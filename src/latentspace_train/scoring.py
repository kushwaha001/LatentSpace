"""Scoring rollouts with the audit-grounded rewards and AuditKIT.

- ``score_records``: per rollout, AuditKIT's raw metrics (``AuditRewards.report()``), the
  train-time rewards and their weighted combination (AgentTune's real ``combine_rewards``),
  and a success verdict.
- ``summarise``: means overall and per language, plus pass@k / pass^k over trials per task
  (``auditkit.agent_eval.reliability``).
- ``bootstrap_compare``: paired comparison of two runs over per-task means, aligned by task
  id: a sign-flip permutation test (the reported p), a centred bootstrap p and a percentile
  95% CI. Not ``auditkit._bootstrap.paired_bootstrap``: it resamples the raw differences
  without centring them under H0, so its p is ~0.5 whatever the effect (FINDINGS.md A3).

Train-time metrics are what run_B optimised; report-only metrics (loop detection, exact
match, success, pass^k) were never part of any reward (src/audit_rewards/README.md,
"Held-out split").
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from typing import Any

from audit_rewards import DEFAULT_WEIGHTS, AuditRewards

TRAIN_TIME = list(DEFAULT_WEIGHTS)
REPORT_ONLY = ["agent_loop_detection", "tool_call_exact", "success"]
AUDIT_METRICS = [
    "tool_call_validity", "tool_call_f1", "tool_call_exact", "redundant_tool_calls",
    "tool_permission", "agent_loop_detection",
]


def make_audit_rewards(tools: list, cfg: dict) -> AuditRewards:
    r = cfg.get("rewards", {}).get("audit", {})
    return AuditRewards(
        tools=tools,
        irreversible_tools=cfg.get("tasks", {}).get("irreversible_tools"),
        denied_tools=cfg.get("tasks", {}).get("denied_tools"),
        no_call_score=r.get("no_call_score", 0.5),
        calls_per_confirmation=r.get("calls_per_confirmation", 1),
        confirmation_penalty=r.get("confirmation_penalty", 1.0),
        log=r.get("log", False),
    )


def audit_weights(cfg: dict) -> dict[str, float]:
    return {**DEFAULT_WEIGHTS, **(cfg.get("rewards", {}).get("audit", {}).get("weights") or {})}


def _has_ref(ref: Any) -> bool:
    return ref is not None and ref != ""


def verdict(row: dict) -> str:
    """success / failure / unknown for one rollout. Success needs: no error, every attempted
    call schema-valid and parsed, all calls within policy, no unconfirmed irreversible call,
    and (with a reference) ToolCallF1 == 1. Without a reference the verdict is unknown."""
    if row.get("error"):
        return "failure"
    if row.get("tool_call_f1") is None:
        return "unknown"
    ok = (
        row["tool_call_f1"] == 1.0
        and row.get("n_malformed", 0) == 0
        and row.get("tool_call_validity") in (None, 1.0)
        and row.get("tool_permission") in (None, 1.0)
        and not row.get("unconfirmed_irreversible")
    )
    return "success" if ok else "failure"


def score_records(records: list[dict], rewards: AuditRewards,
                  weights: dict[str, float] | None = None) -> list[dict]:
    """Score rollouts. Rows with a reference get F1 and the full train-time combination;
    rows without one get the reference-free combination."""
    if not records:
        return []
    weights = weights or DEFAULT_WEIGHTS
    prompts = [r["prompt"] for r in records]
    completions = [r.get("response") or "" for r in records]
    refs = [r.get("expected_tool_calls") if _has_ref(r.get("expected_tool_calls")) else None
            for r in records]
    kw = {"conversations": [r.get("conversation") or None for r in records],
          "expected_tool_calls": refs}
    report = rewards.report(prompts, completions, **kw)
    rew = {fn.__name__: fn(prompts=prompts, completions=completions, **kw)
           for fn in rewards.train_time(include_f1=True)}

    out = []
    for i, rec in enumerate(records):
        row = {k: rec.get(k) for k in ("task_id", "language", "trial", "error", "tool_call_count")}
        row.update({k: v for k, v in report[i].items() if k != "calls"})
        row["calls"] = report[i]["calls"]
        for name, vals in rew.items():
            row[name] = vals[i]
        used = {n: w for n, w in weights.items() if n in rew and rew[n][i] is not None}
        total = sum(used.values())
        # same normalisation as AgentTune's combine_rewards (weights sum to 1)
        row["audit_combined"] = (
            sum(rew[n][i] * w for n, w in used.items()) / total if total else None
        )
        row["success"] = {"success": 1.0, "failure": 0.0}.get(verdict(row))
        row["verdict"] = verdict(row)
        out.append(row)
    return out


def _mean(vals: list) -> float | None:
    vals = [float(v) for v in vals if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def _metric_means(rows: list[dict]) -> dict[str, Any]:
    keys = AUDIT_METRICS + TRAIN_TIME + ["audit_combined", "success", "tool_call_count",
                                         "n_malformed"]
    out = {k: _mean([r.get(k) for r in rows]) for k in keys}
    out["n_rollouts"] = len(rows)
    out["n_errors"] = sum(1 for r in rows if r.get("error"))
    out["unconfirmed_irreversible_rate"] = _mean(
        [1.0 if r.get("unconfirmed_irreversible") else 0.0 for r in rows]
    )
    return out


def pass_k(rows: list[dict], k: int, independent: bool = False) -> dict[str, Any]:
    """pass@k and pass^k (all k trials succeed) per task, averaged over tasks."""
    from auditkit.agent_eval.reliability import reliability

    by_task: dict[str, list[str]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r.get("trial", 0)):
        by_task[str(r["task_id"])].append(r.get("verdict", "unknown"))
    per_task = {t: reliability(v, k=k, independent=independent) for t, v in by_task.items()}
    decided = {t: r for t, r in per_task.items() if r.get("status") != "unknown"}
    return {
        "k": k,
        "n_tasks": len(per_task),
        "n_tasks_decided": len(decided),
        "pass_at_k": _mean([r.get("pass_at_k") for r in decided.values()]),
        "pass_hat_k": _mean([r.get("all_k") for r in decided.values()]),
        "success_rate": _mean([r.get("success_rate") for r in decided.values()]),
        "status_counts": dict(
            sorted(_count(r.get("status") for r in per_task.values()).items())
        ),
        "per_task": {t: {k2: r.get(k2) for k2 in ("pass_at_k", "all_k", "success_rate", "status")}
                     for t, r in per_task.items()},
    }


def _count(items) -> dict:
    out: dict = defaultdict(int)
    for i in items:
        out[i] += 1
    return out


def summarise(rows: list[dict], *, k: int | None = None, independent: bool = False) -> dict:
    """Overall and per-language means; pass@k / pass^k when there are k > 1 trials."""
    langs = sorted({r.get("language", "unknown") for r in rows})
    out: dict[str, Any] = {
        "overall": _metric_means(rows),
        "per_language": {lang: _metric_means([r for r in rows if r.get("language") == lang])
                         for lang in langs},
        "metric_roles": {"train_time": TRAIN_TIME + ["audit_combined"],
                         "report_only": REPORT_ONLY + ["pass_at_k", "pass_hat_k"]},
    }
    if k and k > 1:
        out["overall"]["reliability"] = {
            kk: v for kk, v in pass_k(rows, k, independent).items() if kk != "per_task"
        }
        for lang in langs:
            rel = pass_k([r for r in rows if r.get("language") == lang], k, independent)
            out["per_language"][lang]["reliability"] = {
                kk: v for kk, v in rel.items() if kk != "per_task"
            }
    return out


# Metrics derived from a scored row instead of read as a float field.
DERIVED_METRICS = {
    "unconfirmed_irreversible_rate": lambda r: float(bool(r.get("unconfirmed_irreversible"))),
    "no_call_rate": lambda r: (None if r.get("tool_call_count") is None
                               else float(r["tool_call_count"] == 0)),
}


def _value(r: dict, metric: str) -> float | None:
    if metric in DERIVED_METRICS:
        return DERIVED_METRICS[metric](r)
    v = r.get(metric)
    return None if v is None else float(v)


def per_task_means(rows: list[dict], metric: str) -> dict[str, float]:
    by_task: dict[str, list] = defaultdict(list)
    for r in rows:
        v = _value(r, metric)
        if v is not None:
            by_task[str(r["task_id"])].append(v)
    return {t: statistics.fmean(v) for t, v in by_task.items() if v}


def paired_test(pairs: list[tuple[float, float]], *, n_resamples: int = 10000,
                seed: int = 0) -> dict[str, Any]:
    """Two-sided paired test over ``(baseline, candidate)`` pairs. ``delta`` = mean of
    candidate - baseline. ``p_value`` is a sign-flip permutation test (exact under H0 of
    exchangeable pairs); ``p_bootstrap`` resamples the differences centred on 0 (the H0
    distribution); ``ci95`` is the percentile bootstrap CI of ``delta``."""
    import random

    n = len(pairs)
    if n < 2:
        return {"error": "need at least 2 aligned sample pairs", "n": n}
    d = [c - b for b, c in pairs]
    obs = sum(d) / n
    centred = [x - obs for x in d]
    rng = random.Random(seed)
    tol = 1e-12
    perm = boot = 0
    boots = []
    for _ in range(n_resamples):
        if abs(sum(x if rng.random() < 0.5 else -x for x in d) / n) >= abs(obs) - tol:
            perm += 1
        if abs(sum(rng.choice(centred) for _ in range(n)) / n) >= abs(obs) - tol:
            boot += 1
        boots.append(sum(rng.choice(d) for _ in range(n)) / n)
    boots.sort()
    lo, hi = boots[int(0.025 * n_resamples)], boots[min(n_resamples - 1, int(0.975 * n_resamples))]
    p = (perm + 1) / (n_resamples + 1)
    return {
        "n": n,
        "mean_baseline": round(sum(b for b, _ in pairs) / n, 6),
        "mean_candidate": round(sum(c for _, c in pairs) / n, 6),
        "delta": round(obs, 6),
        "ci95": [round(lo, 6), round(hi, 6)],
        "p_value": round(p, 6),
        "p_bootstrap": round((boot + 1) / (n_resamples + 1), 6),
        "significant": p < 0.05,
    }


def bootstrap_compare(base_rows: list[dict], cand_rows: list[dict], metrics: list[str],
                      *, n_resamples: int = 10000, seed: int = 0) -> dict[str, Any]:
    """``paired_test`` over per-task means (trials averaged first), aligned by task id.
    ``delta`` > 0 means the candidate scored higher."""
    out = {}
    for m in metrics:
        b, c = per_task_means(base_rows, m), per_task_means(cand_rows, m)
        pairs = [(b[t], c[t]) for t in sorted(b.keys() & c.keys())]
        out[m] = paired_test(pairs, n_resamples=n_resamples, seed=seed)
    return out


def dumps_ref(ref: Any) -> str | None:
    return None if ref is None else (ref if isinstance(ref, str) else json.dumps(ref))
