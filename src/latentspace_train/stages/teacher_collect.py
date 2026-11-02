"""Stage 2, teacher_collect: Aya Expanse trajectories on the train split, rejection-filtered.

Default teacher: ``CohereLabs/aya-expanse-32b:cohere`` through the HF router
(OpenAI-compatible, ``HF_TOKEN``, optional ``bill_to``) via AgentTune's LiteLLM
``APIRolloutEngine``. ``--max-usd`` (default 5) stops collection when the estimated spend
reaches the budget; tokens used are logged. ``teacher.backend: local`` runs
``teacher.local_fallback`` (aya-expanse-8b in 4-bit) on the GPU instead, since 8B has no
Inference Provider.

Rejection filter (each trajectory, scored with the audit rewards):
- ``audit_tool_validity_reward == 1`` (every attempted call parsed and schema-valid), and
- ``audit_confirm_irreversible_penalty == 0`` (no irreversible call without a "yes"),
- no rollout error, a non-empty final answer;
- optional ``min_f1`` against the reference;
- ``keep_correct_no_call``: a task whose reference is "no call" has nothing to validate
  (validity is 0.5, the no-call score). Keep it when the teacher made no call (F1 == 1), so
  the student also learns when *not* to call a tool.
Raw and filtered JSONL are written incrementally, so a budget stop keeps what was paid for.
"""

from __future__ import annotations

import json
from collections import Counter

from . import free, gen_kwargs, make_budget, make_engine
from ..hub import push_file
from ..rollouts import build_rollout_fn, run_rollouts
from ..runtime import work_dir, write_jsonl
from ..scoring import audit_weights, make_audit_rewards, score_records, summarise
from ..tasks import load_split, load_tools


def _no_call_reference(ref) -> bool:
    if ref is None:
        return False
    try:
        r = json.loads(ref) if isinstance(ref, str) else ref
    except (json.JSONDecodeError, TypeError):
        return False
    flat = r if not (r and isinstance(r[0], list)) else [c for turn in r for c in turn]
    return len(flat) == 0


def reject_reason(rec: dict, s: dict, f: dict) -> str | None:
    """None to keep the trajectory, else the first failed check."""
    if rec.get("error"):
        return "rollout_error"
    if not (rec.get("response") or "").strip():
        return "empty_final_answer"
    if s.get("audit_confirm_irreversible_penalty") not in (0, 0.0):
        return "unconfirmed_irreversible"
    if f.get("min_f1") is not None and (s.get("tool_call_f1") is None or s["tool_call_f1"] < f["min_f1"]):
        return "f1_below_min"
    if s.get("n_calls", 0) == 0:
        if f.get("keep_correct_no_call", True) and _no_call_reference(rec.get("expected_tool_calls")) \
                and s.get("tool_call_f1") == 1.0:
            return None
        return "no_tool_call"
    if s.get("audit_tool_validity_reward") != 1.0:
        return "invalid_tool_call"
    return None


def run(cfg: dict, args) -> tuple[str, dict]:
    t = cfg.get("teacher", {})
    filt = t.get("filter", {})
    tools = load_tools(cfg)
    rows = load_split(cfg, t.get("split", "train"))
    rewards = make_audit_rewards(tools, cfg)
    backend = t.get("backend", "router")
    if cfg.get("dry_run"):
        backend = "local"  # no network in a dry run: the tiny model plays the teacher
    if backend == "router":
        spec = {"backend": "router", "model": t["model"], **(t.get("router") or {}),
                "max_usd": t.get("max_usd", 5.0), "price_usd_per_m": t.get("price_usd_per_m")}
    else:
        spec = {"backend": "local", **(t.get("local_fallback") or {})}
        if cfg.get("dry_run"):
            spec = {"backend": "local"}
    budget = make_budget(spec) if backend == "router" else None
    engine, sp, desc = make_engine(cfg, spec, tools, budget)
    fn = build_rollout_fn(engine, tools, system_prompt=sp,
                          max_steps=cfg.get("rollout", {}).get("max_steps_per_turn", 4))
    d = work_dir(cfg, "teacher_collect")
    raw_path, kept_path = d / "raw.jsonl", d / "filtered.jsonl"
    for p in (raw_path, kept_path):
        p.unlink(missing_ok=True)
    weights = audit_weights(cfg)
    reasons: Counter = Counter()
    kept_by_lang: Counter = Counter()
    seen_by_lang: Counter = Counter()
    scored_all: list[dict] = []

    def on_record(rec: dict) -> None:
        s = score_records([rec], rewards, weights)[0]
        why = reject_reason(rec, s, filt)
        scored_all.append(s)
        seen_by_lang[rec["language"]] += 1
        reasons[why or "kept"] += 1
        row = {**rec, "scores": {k: v for k, v in s.items() if k != "calls"}, "kept": why is None,
               "reject_reason": why, "teacher": desc["model"]}
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        if why is None:
            kept_by_lang[rec["language"]] += 1
            with open(kept_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    n = int(t.get("samples_per_task", 1))
    print(f"[teacher_collect] {len(rows)} tasks x {n} sample(s) with {desc['model']}"
          + (f", budget ${budget.max_usd:.2f}" if budget and budget.max_usd else ""))
    _, status = run_rollouts(fn, rows, trials=n, gen_kwargs=gen_kwargs(cfg, "teacher"),
                             on_record=on_record)
    if not kept_path.exists():
        write_jsonl(kept_path, [])
    out = {
        "teacher": desc,
        "split": t.get("split", "train"),
        "n_tasks": len(rows),
        "samples_per_task": n,
        "rollout_status": status,
        "budget": budget.summary() if budget else None,
        "filter": {"rule": "validity == 1 and confirm_penalty == 0 (see module docstring)",
                   **filt},
        "kept": reasons.get("kept", 0),
        "seen": sum(reasons.values()),
        "reasons": dict(reasons),
        "kept_by_language": {k: {"kept": kept_by_lang[k], "seen": v,
                                 "rate": round(kept_by_lang[k] / v, 3)}
                             for k, v in sorted(seen_by_lang.items())},
        "teacher_summary": summarise(scored_all),
        "files": {"raw": str(raw_path), "filtered": str(kept_path)},
    }
    if cfg.get("push"):
        out["hub"] = [push_file(cfg, "teacher_data", kept_path, path_in_repo="filtered.jsonl"),
                      push_file(cfg, "teacher_data", raw_path, path_in_repo="raw.jsonl")]
    free(engine, fn)
    return "11_teacher_collect", out
