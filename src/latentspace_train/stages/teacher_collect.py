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
- optional ``require_correct``: keep only trajectories that solve the task (the success
  verdict, with semantic argument matching when ``tasks.lenient_score`` is set). Without it
  the student also imitates wrong tool use (round 1: 129 of 301 kept trajectories were wrong);
- optional ``require_reply_language``: the final answer is in the user's script (the 4-bit
  teacher sometimes answers an English question in Hindi);
- optional ``min_f1`` against the reference;
- ``keep_correct_no_call``: a task whose reference is "no call" has nothing to validate
  (validity is 0.5, the no-call score). Keep it when the teacher made no call (F1 == 1), so
  the student also learns when *not* to call a tool.
Raw and filtered JSONL are written incrementally, so a budget stop keeps what was paid for.

``teacher.guided: true`` (round 2): reference-guided collection. Round 1 showed that the
4-bit teacher rarely completes a multi-step task (0 of 54 kept trajectories were correct), so
the student never saw one done right. For a train row whose reference has tool calls, those
calls are replayed turn by turn on the backend and written into the conversation in the
rollout's own format; the teacher then continues from there, so it writes the final answer
from real tool results. Rows whose reference is "no call" are free teacher rollouts as
before. Every trajectory still goes through the same filter (``require_correct``), so a
teacher that adds a wrong call after the replay is rejected. Only the train split is used.
"""

from __future__ import annotations

import json
from collections import Counter

from . import free, gen_kwargs, make_budget, make_engine
from ..hub import push_file
from ..rollouts import build_rollout_fn, run_rollouts
from ..runtime import work_dir, write_jsonl
from ..scoring import audit_weights, lenient_scorer, make_audit_rewards, score_records, summarise
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


def guided_prompt(row: dict, tools: list) -> tuple[list[dict], int] | None:
    """The row's prompt followed by its reference tool calls and their real results, in the
    shape AgentTune's rollout leaves in ``conversation`` for a template without tool turns:
    an assistant turn holding the calls, then one user turn with the folded results.
    Returns ``(messages, n_calls)``, or None when the reference has no calls."""
    import uuid

    from agenttune.agentic.rollout_engines.rollout_factory import DEFAULT_TOOL_RESULT_FORMAT

    from .distill import cohere_call_text

    ref = row.get("expected_tool_calls")
    turns = json.loads(ref) if isinstance(ref, str) and ref.strip() else (ref or [])
    turns = [t if isinstance(t, list) else [t] for t in turns]
    if not any(turns):
        return None
    by_name = {f.__name__: f for f in tools}
    prompt = row["prompt"]
    msgs = [dict(m) for m in prompt] if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    n = 0
    for step in turns:
        calls = [{"type": "function", "id": uuid.uuid4().hex[:9],
                  "function": {"name": c["name"],
                               "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                 for c in step]
        msgs.append({"role": "assistant", "content": cohere_call_text(calls), "tool_calls": calls})
        results = []
        for c in step:
            try:
                out = by_name[c["name"]](**c["arguments"])
            except Exception as e:  # a reference call must run; surfaced by the filter
                out = {"error": f"{type(e).__name__}: {e}"}
            results.append(DEFAULT_TOOL_RESULT_FORMAT.format(name=c["name"], content=out))
            n += 1
        msgs.append({"role": "user", "content": "\n\n".join(results)})
    return msgs, n


def reply_language_ok(language: str, text: str) -> bool:
    """The final answer is in the user's script: Devanagari for ``hi`` / ``mr``, Latin for
    ``en`` (names and a few quoted words in the other script are fine). ``hinglish`` accepts
    both: the teacher answers 3 of 4 romanised-Hindi requests in Devanagari Hindi, which the
    user can read, and rejecting those left no Hinglish example of a tool call at all."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return True
    dev = sum("\u0900" <= ch <= "\u097f" for ch in letters) / len(letters)
    if language in ("hi", "mr"):
        return dev >= 0.5
    if language == "en":
        return dev <= 0.2
    return True


def reject_reason(rec: dict, s: dict, f: dict) -> str | None:
    """None to keep the trajectory, else the first failed check."""
    if rec.get("error"):
        return "rollout_error"
    if not (rec.get("response") or "").strip():
        return "empty_final_answer"
    if f.get("require_reply_language") and not reply_language_ok(
            rec.get("language", ""), rec.get("response") or ""):
        return "wrong_reply_language"
    if s.get("audit_confirm_irreversible_penalty") not in (0, 0.0):
        return "unconfirmed_irreversible"
    if f.get("require_correct"):
        # the trajectory must solve the task: every reference call made (semantic match when
        # tasks.lenient_score is set, else exact), nothing extra, every call parsed and valid
        ok = s.get("lenient_success", s.get("success"))
        if ok != 1.0:
            return "incorrect"
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
    # teacher.resume: keep the rollouts already written (after a crash or restart) and run
    # only the missing (task, sample) pairs; counts below then cover this session only, and
    # scripts/merge_teacher_shards.py recomputes them from the complete raw file.
    done: set = set()
    if t.get("resume") and raw_path.exists():
        with open(raw_path, encoding="utf-8") as f:
            done = {(r["task_id"], r["trial"]) for r in map(json.loads, f)}
        print(f"[teacher_collect] resume: {len(done)} rollouts already collected")
    else:
        for p in (raw_path, kept_path):
            p.unlink(missing_ok=True)
    weights = audit_weights(cfg)
    lenient = lenient_scorer(cfg)
    reasons: Counter = Counter()
    kept_by_lang: Counter = Counter()
    seen_by_lang: Counter = Counter()
    scored_all: list[dict] = []
    kept_guided = [0]

    guided: dict[str, tuple] = {}
    if t.get("guided"):
        new_rows = []
        for r in rows:
            g = guided_prompt(r, tools)
            if g is not None:
                guided[r["id"]] = (r["prompt"], g[1])
                r = {**r, "prompt": g[0]}
            new_rows.append(r)
        rows = new_rows
        print(f"[teacher_collect] reference-guided: {len(guided)} of {len(rows)} tasks")

    def on_record(rec: dict) -> None:
        rec["guided"] = rec["task_id"] in guided
        if rec["guided"]:
            # score against the task as the student will see it: the replayed calls belong
            # to the trajectory, not to the prompt
            rec["prompt"], n_replayed = guided[rec["task_id"]]
            rec["tool_call_count"] = (rec.get("tool_call_count") or 0) + n_replayed
        s = score_records([rec], rewards, weights, lenient=lenient)[0]
        why = reject_reason(rec, s, filt)
        scored_all.append(s)
        seen_by_lang[rec["language"]] += 1
        reasons[why or "kept"] += 1
        row = {**rec, "scores": {k: v for k, v in s.items() if k != "calls"},
               "category": s.get("category"), "kept": why is None,
               "reject_reason": why, "teacher": desc["model"]}
        with open(raw_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
        if why is None:
            kept_guided[0] += int(rec["guided"])
            kept_by_lang[rec["language"]] += 1
            with open(kept_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    n = int(t.get("samples_per_task", 1))
    print(f"[teacher_collect] {len(rows)} tasks x {n} sample(s) with {desc['model']}"
          + (f", budget ${budget.max_usd:.2f}" if budget and budget.max_usd else ""))
    _, status = run_rollouts(fn, rows, trials=n, gen_kwargs=gen_kwargs(cfg, "teacher"),
                             on_record=on_record, skip=done)
    if not kept_path.exists():
        write_jsonl(kept_path, [])
    out = {
        "teacher": desc,
        "split": t.get("split", "train"),
        "n_tasks": len(rows),
        "samples_per_task": n,
        "resumed_with": len(done),
        "guided": {"enabled": bool(t.get("guided")), "tasks": len(guided)},
        "kept_guided": kept_guided[0],
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
