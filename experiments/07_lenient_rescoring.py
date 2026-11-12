"""Experiment 07: re-score the held-out rollouts with lenient argument matching.

Inputs: the round-1 rollouts, ``outputs_round1/evaluate/<checkpoint>/scored.jsonl`` and
``data/eval/test.jsonl``. Nothing is re-generated: the same 540 rollouts per checkpoint are
matched again, once with AuditKIT's exact matcher (to check we reproduce the reported numbers)
and once with ``transit_agent.lenient_match.lenient_arg_match`` (rules in its docstring).

Only the tool-call match changes. Validity, permission, the unconfirmed-irreversible check and
call counts are taken from the scored rows unchanged, and ``success`` uses the same verdict as
the evaluate stage with the lenient F1 in place of the strict one.

Writes ``results/round1/07_lenient_rescoring.json`` and ``.md``. From round 2 on, the evaluate
stage reports the same lenient metrics itself (``tasks.lenient_score``).
Run from the repo root: ``python experiments/07_lenient_rescoring.py``.
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import transit_agent  # noqa: E402
from audit_rewards.tools import tool_schemas  # noqa: E402
from auditkit.metrics.agent import _optional_arg_matcher, _tool_schemas  # noqa: E402
from auditkit.trace import ToolCall, max_matching  # noqa: E402
from latentspace_train.scoring import paired_test, verdict  # noqa: E402
from transit_agent.lenient_match import lenient_arg_match  # noqa: E402

CHECKPOINTS = ["base", "distill", "run_A", "run_B"]
COMPARISONS = [("run_A", "run_B"), ("base", "run_B"), ("distill", "run_B"), ("base", "run_A"),
               ("base", "distill")]
LANGS = ["en", "hi", "mr", "hinglish"]
strict_match = _optional_arg_matcher(_tool_schemas(tool_schemas(transit_agent.TOOLS)))


def f1_exact(preds: list[ToolCall], refs: list[ToolCall], fn) -> tuple[float, float]:
    matched = len(max_matching(preds, refs, "exact", fn))
    if not refs:
        f = 1.0 if not preds else 0.0
    else:
        p = matched / len(preds) if preds else 0.0
        r = matched / len(refs)
        f = 2 * p * r / (p + r) if p + r else 0.0
    return f, 1.0 if matched == len(refs) == len(preds) else 0.0


def rescore(rows: list[dict], tasks: dict) -> list[dict]:
    out = []
    for r in rows:
        t = tasks[r["task_id"]]
        refs = [ToolCall(c["name"], c["arguments"]) for turn in t["expected_tool_calls"] for c in turn]
        preds = [ToolCall(c["name"], c["arguments"]) for c in r.get("calls") or []
                 if isinstance(c.get("arguments"), dict)]
        sf1, sex = f1_exact(preds, refs, strict_match)
        lf1, lex = f1_exact(preds, refs, lenient_arg_match)
        row = dict(r)
        row["strict_f1_recomputed"] = sf1
        row["lenient_f1"], row["lenient_exact"] = lf1, lex
        row["lenient_success"] = 1.0 if verdict({**r, "tool_call_f1": lf1}) == "success" else 0.0
        row["category"], row["language"] = t["category"], t["language"]
        out.append(row)
    return out


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.fmean(xs), 4) if xs else None


def summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "success_strict": mean(r["success"] for r in rows),
        "success_lenient": mean(r["lenient_success"] for r in rows),
        "exact_strict": mean(r["tool_call_exact"] for r in rows),
        "exact_lenient": mean(r["lenient_exact"] for r in rows),
        "f1_strict": mean(r["tool_call_f1"] for r in rows),
        "f1_lenient": mean(r["lenient_f1"] for r in rows),
    }


def pass_k(rows: list[dict], key: str) -> dict:
    by = collections.defaultdict(list)
    for r in rows:
        by[r["task_id"]].append(r[key])
    return {"pass_at_k": mean(float(any(v)) for v in by.values()),
            "pass_hat_k": mean(float(all(v)) for v in by.values())}


def main() -> None:
    tasks = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data/eval/test.jsonl")}
    scored = {}
    for ck in CHECKPOINTS:
        rows = [json.loads(l) for l in open(ROOT / f"outputs_round1/evaluate/{ck}/scored.jsonl")]
        scored[ck] = rescore(rows, tasks)

    # sanity check: the strict re-match must reproduce the F1 the evaluate stage reported
    mismatch = {ck: sum(abs(r["strict_f1_recomputed"] - (r["tool_call_f1"] or 0.0)) > 1e-9
                        for r in rows) for ck, rows in scored.items()}

    res = {"experiment": "07_lenient_rescoring", "rules": "transit_agent.lenient_match docstring",
           "n_tasks": len({r["task_id"] for r in scored["base"]}),
           "strict_recompute_mismatches": mismatch, "checkpoints": {}, "comparisons": {},
           "changed_rollouts": {}}
    cats = sorted({t["category"] for t in tasks.values()})
    for ck, rows in scored.items():
        res["checkpoints"][ck] = {
            "overall": summary(rows),
            "pass_k_strict": pass_k(rows, "success"),
            "pass_k_lenient": pass_k(rows, "lenient_success"),
            "per_language": {lg: summary([r for r in rows if r["language"] == lg]) for lg in LANGS},
            "per_category": {c: summary([r for r in rows if r["category"] == c]) for c in cats},
        }
        flipped = [r for r in rows if r["lenient_success"] > r["success"]]
        res["changed_rollouts"][ck] = {
            "strict_fail_to_lenient_success": len(flipped),
            "lenient_lower_than_strict": sum(r["lenient_f1"] < r["tool_call_f1"] - 1e-9 for r in rows),
            "by_tool": dict(collections.Counter(
                c["name"] for r in flipped for c in r.get("calls") or [])),
            "examples": [{"task_id": r["task_id"],
                          "expected": tasks[r["task_id"]]["expected_tool_calls"],
                          "model": [{"name": c["name"], "arguments": c.get("arguments")}
                                    for c in r.get("calls") or []]} for r in flipped[:4]],
        }

    def per_task(rows, key):
        by = collections.defaultdict(list)
        for r in rows:
            by[r["task_id"]].append(float(r[key]))
        return {t: statistics.fmean(v) for t, v in by.items()}

    for base, cand in COMPARISONS:
        cmp = {}
        for key in ("success", "lenient_success", "tool_call_f1", "lenient_f1"):
            b, c = per_task(scored[base], key), per_task(scored[cand], key)
            cmp[key] = paired_test([(b[t], c[t]) for t in sorted(b.keys() & c.keys())])
        res["comparisons"][f"{base}_vs_{cand}"] = cmp

    (ROOT / "results/round1/07_lenient_rescoring.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    write_md(res, cats)
    print(json.dumps({ck: v["overall"] for ck, v in res["checkpoints"].items()}, indent=1))
    print("strict recompute mismatches:", mismatch)


def write_md(res: dict, cats: list[str]) -> None:
    ck = res["checkpoints"]
    L = ["# Experiment 07: strict vs lenient argument matching (held-out test, 108 tasks x 5)",
         "", "Rules: `src/transit_agent/lenient_match.py`. Only the tool-call match changes; "
         "validity, permission and the confirmation check are unchanged.", "",
         "| Model | success strict | success lenient | exact strict | exact lenient | F1 strict | F1 lenient | pass^5 strict | pass^5 lenient |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, v in ck.items():
        o = v["overall"]
        L.append(f"| {name} | {o['success_strict']:.3f} | **{o['success_lenient']:.3f}** | "
                 f"{o['exact_strict']:.3f} | {o['exact_lenient']:.3f} | {o['f1_strict']:.3f} | "
                 f"{o['f1_lenient']:.3f} | {v['pass_k_strict']['pass_hat_k']:.3f} | "
                 f"{v['pass_k_lenient']['pass_hat_k']:.3f} |")
    L += ["", "## Success per language (strict → lenient)", "",
          "| Model | " + " | ".join(LANGS) + " |", "|---|" + "---|" * len(LANGS)]
    for name, v in ck.items():
        L.append(f"| {name} | " + " | ".join(
            f"{v['per_language'][lg]['success_strict']:.2f} → {v['per_language'][lg]['success_lenient']:.2f}"
            for lg in LANGS) + " |")
    L += ["", "## Success per category (strict → lenient)", "",
          "| Category | " + " | ".join(ck) + " |", "|---|" + "---|" * len(ck)]
    for c in cats:
        L.append(f"| {c} | " + " | ".join(
            f"{ck[n]['per_category'][c]['success_strict']:.2f} → {ck[n]['per_category'][c]['success_lenient']:.2f}"
            for n in ck) + " |")
    L += ["", "## Paired tests over per-task means (sign-flip permutation; * p < 0.05)", "",
          "| Comparison | success strict | success lenient | F1 strict | F1 lenient |",
          "|---|---|---|---|---|"]
    for k, c in res["comparisons"].items():
        def f(r):
            return f"{r['delta']:+.3f} (p {r['p_value']:.3f}){' *' if r['significant'] else ''}"
        L.append(f"| {k.replace('_vs_', ' → ')} | {f(c['success'])} | {f(c['lenient_success'])} | "
                 f"{f(c['tool_call_f1'])} | {f(c['lenient_f1'])} |")
    L += ["", "Rollouts that fail strictly but succeed leniently: " + ", ".join(
        f"{n} {v['strict_fail_to_lenient_success']}" for n, v in res["changed_rollouts"].items()),
        f"Strict re-match mismatches vs reported F1: {res['strict_recompute_mismatches']}"]
    (ROOT / "results/round1/07_lenient_rescoring.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
