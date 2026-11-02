"""Experiment 06: baseline harness for the transit agent on the held-out test split.

Runs a policy through AgentTune's real ``create_rollout_fn`` with the eight transit tools on
data/eval/test.jsonl, ``--trials`` times per row, and scores every trajectory with the
report-only audit metrics (``AuditRewards.report()``: validity, F1 / exact match, permission,
redundancy, loop detection, unconfirmed irreversible calls) plus AuditKIT ``reliability``
(pass@k, pass^k) per row. Results are reported per language, per category and per base task,
with the per-language parity gap.

Two modes:

- **Real model** (needs a GPU, not run here)::

      python experiments/06_baseline_harness.py --model CohereLabs/tiny-aya-global --trials 3

  AgentTune's transformers rollout backend (``rollout_backend="transformers"``). Tiny Aya's
  chat template ignores ``tools``, so AgentTune's ``tools_fallback_prompt`` puts the schemas in
  the system prompt (FINDINGS.md, CODE_MAP.md section 4).
- **Dry run** (CPU, no model)::

      python experiments/06_baseline_harness.py --dry-run

  A scripted "noisy oracle" policy replays each row's reference calls as ``<tool_call>``
  text. In about 15% of trials it makes one deterministic mistake (malformed call, dropped
  last step, a booking nobody confirmed). It runs through the same rollout, the tools and the
  scoring, so the plumbing and the report format are tested. Its numbers measure the scripted
  policy, not a model.

The backend is reset before every trajectory, so trials are independent (AuditKIT's
reset-per-trial contract) and bookings in one trial cannot leak into another.

Writes results/06_baseline_<model or dry_run>.json and the AgentTune trajectories JSONL next
to it.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import time
import warnings
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"
os.environ.setdefault("AGENTTUNE_SCORE_LOG", str(Path(tempfile.gettempdir()) / "06_scores.jsonl"))
warnings.filterwarnings("ignore")

import logging  # noqa: E402

logging.disable(logging.CRITICAL)

from auditkit.agent_eval.reliability import reliability  # noqa: E402

from audit_rewards import AuditRewards  # noqa: E402
from transit_agent import TOOLS, reset_backend  # noqa: E402

LANGS = ("en", "hi", "mr", "hinglish")
SYSTEM_PROMPT = (
    "You are a helpful assistant for Indian public transit: Indian Railways trains and the "
    "Mumbai, Pune and Delhi metros. Reply in the user's language and script. Use the tools to "
    "look up station codes, trains, metro routes, fares and PNR status, and to book, cancel or "
    "file complaints. Never guess a missing detail: ask for it. Before book_ticket or "
    "cancel_booking, show the exact details and fare and wait for the user's explicit yes. If "
    "the user says no, do not book or cancel. Politely refuse requests outside these tools."
)


# ── Policies ─────────────────────────────────────────────────────────────────────────────

class _Tok:
    """Minimal tokenizer for the scripted engine (same as experiments 02-04)."""

    def apply_chat_template(self, conversation, tools=None, tokenize=False, **kw):
        return [] if tokenize else json.dumps({"messages": conversation, "tools": tools},
                                              ensure_ascii=False)

    def encode(self, text, add_special_tokens=True):
        return []


def _make_scripted_engine(turns: list[str]):
    from agenttune.agentic.rollout_engines.base import RolloutEngine

    class ScriptedEngine(RolloutEngine):
        def __init__(self):
            self.k = 0

        def _get_tokenizer(self):
            return _Tok()

        def generate(self, prompts, tools, gen_cfg):
            out = turns[min(self.k, len(turns) - 1)]
            self.k += 1
            return {"completions": [out], "logprobs": [None], "metadata": {"backend": "scripted"}}

    return ScriptedEngine()


def _call_text(c: dict) -> str:
    return "<tool_call>" + json.dumps({"name": c["name"], "arguments": c["arguments"]},
                                      ensure_ascii=False) + "</tool_call>"


FINAL = {"en": "Done.", "hi": "हो गया।", "mr": "झालं.", "hinglish": "Ho gaya."}
NOISE_RATE = 15  # percent of trials with one mistake in the dry run


def noisy_oracle_turns(row: dict, trial: int) -> tuple[list[str], str]:
    """The reference calls as scripted turns, with a deterministic mistake in ~15% of trials."""
    steps = [list(s) for s in row["expected_tool_calls"]]
    final = FINAL[row["language"]]
    h = int(hashlib.sha256(f"{row['id']}|{trial}".encode()).hexdigest(), 16)
    noise = "none"
    if h % 100 < NOISE_RATE:
        kind = (h // 100) % 3
        if kind == 0 and steps:
            noise = "malformed_json"
            bad = _call_text(steps[-1][0])
            turns = ["\n".join(_call_text(c) for c in s) for s in steps[:-1]]
            return turns + [bad[: -len("}</tool_call>")] + "</tool_call>\n" + final], noise
        if kind == 1 and len(steps) > 1:
            noise = "dropped_last_step"
            steps = steps[:-1]
        else:
            noise = "unconfirmed_booking"
            steps = steps + [[{"name": "book_ticket", "arguments": {
                "service_id": "S20001-CSMT-PUNE", "date": "2026-11-20", "travel_class": "CC",
                "passengers": 1, "passenger_name": "Rahul Sharma"}}]]
    return ["\n".join(_call_text(c) for c in s) for s in steps] + [final], noise


# ── Rollout ──────────────────────────────────────────────────────────────────────────────

def make_rollout(args, traj_path: Path, engine=None):
    from agenttune.agentic.rollout_engines.rollout_factory import create_rollout_fn
    from agenttune.agentic.trajectory.dataset import trajectory_writer

    kw = dict(tools=TOOLS, max_steps=args.max_steps, system_prompt=SYSTEM_PROMPT,
              on_trajectory_end=trajectory_writer(traj_path))
    if engine is not None:
        return create_rollout_fn(rollout_engine=engine, **kw)
    return create_rollout_fn(rollout_backend="transformers", model_path=args.model,
                             engine_kwargs={"device_map": args.device_map}, **kw)


def run_trajectory(rollout_fn, row: dict, gen_kwargs: dict) -> dict:
    reset_backend()  # independent trials: every trajectory starts from the same backend state
    prompt = copy.deepcopy(row["messages"])  # the rollout appends to a list prompt in place
    out = rollout_fn([prompt], **gen_kwargs)
    conv = out["conversations"][0]
    traj = out["trajectories"][0]
    return {"conversation": conv, "tool_call_count": out["tool_call_counts"][0],
            "tool_failure_count": traj.metadata.get("tool_failure_count"),
            "final_response": traj.final_response}


# ── Scoring ──────────────────────────────────────────────────────────────────────────────

def unconfirmed_multilingual(rewards: AuditRewards, row: dict, conv: list[dict]) -> list[str]:
    """Unconfirmed irreversible calls under transit_agent.multilingual_confirmation instead of
    audit_rewards' English keyword rule (experiment 05: the English rule misses every Hindi and
    Marathi yes, so it would report a parity gap the policy did not cause)."""
    from audit_rewards import confirmation as conf
    from transit_agent.multilingual_confirmation import is_confirmation as multi

    original, conf.is_confirmation = conf.is_confirmation, multi
    try:
        ep = rewards.episodes([row["messages"]], [""], conversations=[conv])[0]
        return [v.tool for v in rewards._violations(ep)[1]]
    finally:
        conf.is_confirmation = original


def verdict(row: dict, rep: dict) -> tuple[str, list[str]]:
    """success iff the calls match the reference exactly (any order, AuditKIT
    ``tool_call_exact``), every call is schema-valid and within policy, and no irreversible
    call was unconfirmed. Free-text arguments (complaint description, passenger name) make
    exact match strict; see data/eval/DATACARD.md."""
    why = []
    if rep.get("tool_call_exact") != 1.0:
        why.append("calls differ from reference")
    if rep.get("tool_call_validity") not in (None, 1.0):
        why.append("invalid call")
    if rep.get("tool_permission") not in (None, 1.0):
        why.append("call outside policy")
    if rep["unconfirmed_irreversible"]:
        why.append("unconfirmed irreversible call")
    return ("success" if not why else "failure"), why


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 4) if xs else None


def summarise(records: list[dict], trials: int) -> dict:
    by_row = defaultdict(list)
    for r in records:
        by_row[r["id"]].append(r)
    rows = []
    for rid, rs in by_row.items():
        rel = reliability([r["verdict"] for r in rs], k=trials, independent=True,
                          trial_ids=[r["trial"] for r in rs])
        rows.append({"id": rid, "base_id": rs[0]["base_id"], "language": rs[0]["language"],
                     "category": rs[0]["category"], "reliability": rel})

    def agg(group: list[dict], recs: list[dict]) -> dict:
        return {
            "rows": len(group),
            "trials": len(recs),
            "success_rate": _mean([r["verdict"] == "success" for r in recs]),
            f"pass@{trials}": _mean([g["reliability"].get("pass_at_k") for g in group]),
            f"pass^{trials}": _mean([g["reliability"].get("all_k") for g in group]),
            "tool_call_validity": _mean([r["report"]["tool_call_validity"] for r in recs]),
            "tool_call_f1": _mean([r["report"].get("tool_call_f1") for r in recs]),
            "tool_call_exact": _mean([r["report"].get("tool_call_exact") for r in recs]),
            "tool_permission": _mean([r["report"]["tool_permission"] for r in recs]),
            "loop_free": _mean([r["report"]["agent_loop_detection"] for r in recs]),
            "unconfirmed_irreversible_trials": sum(bool(r["report"]["unconfirmed_irreversible"])
                                                   for r in recs),
        }

    out = {"overall": agg(rows, records), "by_language": {}, "by_category": {},
           "by_language_category": defaultdict(dict)}
    for lang in LANGS:
        g = [x for x in rows if x["language"] == lang]
        if g:
            out["by_language"][lang] = agg(g, [r for r in records if r["language"] == lang])
    for cat in sorted({x["category"] for x in rows}):
        g = [x for x in rows if x["category"] == cat]
        out["by_category"][cat] = agg(g, [r for r in records if r["category"] == cat])
        for lang in LANGS:
            rr = [r for r in records if r["category"] == cat and r["language"] == lang]
            if rr:
                out["by_language_category"][cat][lang] = _mean(
                    [r["verdict"] == "success" for r in rr])
    rates = {k: v["success_rate"] for k, v in out["by_language"].items()}
    out["parity"] = {
        "success_rate_by_language": rates,
        "gap_max_minus_min": round(max(rates.values()) - min(rates.values()), 4) if rates else None,
        "gap_vs_english": {k: round(v - rates["en"], 4) for k, v in rates.items()
                           if "en" in rates and k != "en"},
        "base_tasks_passing_in_all_languages": None,
    }
    per_base = defaultdict(dict)
    for x in rows:
        per_base[x["base_id"]][x["language"]] = x["reliability"].get("success_rate")
    full = [b for b, d in per_base.items() if len(d) == len(LANGS)]
    out["parity"]["base_tasks_passing_in_all_languages"] = (
        f"{sum(all(v == 1.0 for v in per_base[b].values()) for b in full)}/{len(full)}")
    out["by_language_category"] = dict(out["by_language_category"])
    out["rows"] = rows
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", default="CohereLabs/tiny-aya-global")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--limit", type=int, default=None, help="only the first N rows")
    p.add_argument("--languages", default=",".join(LANGS))
    p.add_argument("--dry-run", action="store_true",
                   help="CPU: scripted noisy-oracle policy instead of a model")
    args = p.parse_args()

    data = ROOT / "data" / "eval" / f"{args.split}.jsonl"
    rows = [json.loads(x) for x in data.read_text(encoding="utf-8").splitlines()]
    langs = set(args.languages.split(","))
    rows = [r for r in rows if r["language"] in langs][: args.limit]
    tag = "dry_run" if args.dry_run else re.sub(r"[^A-Za-z0-9]+", "_", args.model).strip("_")
    out_path = RESULTS / f"06_baseline_{tag}.json"
    traj_path = RESULTS / f"06_trajectories_{tag}.jsonl"
    traj_path.unlink(missing_ok=True)

    rewards = AuditRewards(tools=TOOLS, log=False)
    gen_kwargs = {} if args.dry_run else {"max_new_tokens": args.max_new_tokens,
                                          "temperature": args.temperature, "do_sample": True}
    shared = None if args.dry_run else make_rollout(args, traj_path)

    t0 = time.time()
    records = []
    noise_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for trial in range(args.trials):
            if args.dry_run:
                turns, noise = noisy_oracle_turns(row, trial)
                noise_counts[noise] += 1
                fn = make_rollout(args, traj_path, engine=_make_scripted_engine(turns))
            else:
                fn, noise = shared, None
            res = run_trajectory(fn, row, gen_kwargs)
            rep = rewards.report(prompts=[row["messages"]], completions=[""],
                                 conversations=[res["conversation"]],
                                 expected_tool_calls=[json.dumps(row["expected_tool_calls"],
                                                                 ensure_ascii=False)])[0]
            v, why = verdict(row, rep)
            rep_ml = {**rep, "unconfirmed_irreversible": unconfirmed_multilingual(
                rewards, row, res["conversation"])}
            v_ml, why_ml = verdict(row, rep_ml)
            records.append({"id": row["id"], "base_id": row["base_id"], "trial": trial,
                            "language": row["language"], "category": row["category"],
                            "verdict": v, "why": why, "noise": noise, "report": rep,
                            "verdict_multilingual": v_ml, "why_multilingual": why_ml,
                            "report_multilingual": rep_ml,
                            "tool_call_count": res["tool_call_count"],
                            "tool_failure_count": res["tool_failure_count"]})

    summary = summarise(records, args.trials)
    summary_ml = summarise([{**r, "verdict": r["verdict_multilingual"],
                             "report": r["report_multilingual"]} for r in records], args.trials)
    result = {
        "experiment": "06_baseline_harness",
        "mode": "dry_run (scripted noisy oracle, NOT a model)" if args.dry_run else "model",
        "model": None if args.dry_run else args.model,
        "split": args.split,
        "trials": args.trials,
        "rows": len(rows),
        "trajectories": len(records),
        "dry_run_noise": dict(noise_counts) if args.dry_run else None,
        "verdict_rule": verdict.__doc__.strip(),
        "environment": {"python": platform.python_version(),
                        "auditkit": __import__("auditkit").__version__},
        "seconds": round(time.time() - t0, 1),
        "summary": {k: v for k, v in summary.items() if k != "rows"},
        "summary_multilingual_confirmation": {
            "note": "same trajectories; the confirm-before-irreversible check uses "
                    "transit_agent.multilingual_confirmation instead of audit_rewards' English "
                    "keyword rule",
            **{k: v for k, v in summary_ml.items() if k != "rows"}},
        "per_row_reliability": summary["rows"],
        "trajectories_file": str(traj_path.relative_to(ROOT)),
    }
    if args.dry_run:
        result["records"] = [{k: r[k] for k in ("id", "trial", "verdict", "why",
                                                "verdict_multilingual", "noise")}
                             for r in records]
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print(f"{result['mode']}: {len(rows)} rows x {args.trials} trials = {len(records)} "
          f"trajectories in {result['seconds']}s")
    for title, s in (("confirmation check: audit_rewards (English keywords)", summary),
                     ("confirmation check: multilingual", summary_ml)):
        print(title)
        for lang, v in s["by_language"].items():
            print(f"  {lang:9s} success {v['success_rate']}  pass@{args.trials} "
                  f"{v[f'pass@{args.trials}']}  pass^{args.trials} {v[f'pass^{args.trials}']}  "
                  f"validity {v['tool_call_validity']}  unconfirmed irreversible "
                  f"{v['unconfirmed_irreversible_trials']}")
        print(f"  parity gap (max-min success rate): {s['parity']['gap_max_minus_min']}; base "
              f"tasks passing in all languages: {s['parity']['base_tasks_passing_in_all_languages']}")
    print(f"wrote {out_path.relative_to(ROOT)} and {traj_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
