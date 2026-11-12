"""Stage 5, evaluate: score checkpoints on the held-out test split.

For each checkpoint in ``evaluate.checkpoints`` (base model, distilled adapter, run_A,
run_B, or any Hub id / router model): ``trials`` rollouts per task (default 5), scored with

- train-time metrics: the audit rewards run_B optimised, and their weighted combination;
- report-only metrics: AgentLoopDetection, tool_call_exact, success, pass@k and pass^k
  (``auditkit.agent_eval.reliability``), never part of any reward;

overall and per language. Then ``evaluate.comparisons`` pairs (e.g. run_A vs run_B) are
compared over per-task means with a paired sign-flip permutation test (and a centred
bootstrap CI); AuditKIT's paired_bootstrap is not used (uncentred, FINDINGS.md A3).

``--run NAME`` evaluates one checkpoint only; comparisons then use the scored files already
on disk for the other side (``outputs/evaluate/<name>/scored.jsonl``).
"""

from __future__ import annotations

from pathlib import Path

from . import free, gen_kwargs, make_budget, make_engine
from ..config import ROOT
from ..hub import push_folder
from ..rollouts import build_rollout_fn, run_rollouts
from ..runtime import read_jsonl, work_dir, write_jsonl
from ..scoring import (
    audit_weights,
    bootstrap_compare,
    make_audit_rewards,
    lenient_scorer,
    score_records,
    summarise,
)
from ..tasks import load_split, load_tools

COMPARE_METRICS = ["success", "lenient_success", "lenient_f1", "audit_combined", "tool_call_validity", "tool_call_f1",
                   "tool_call_exact", "agent_loop_detection", "audit_confirm_irreversible_penalty",
                   "unconfirmed_irreversible_rate", "tool_call_count", "no_call_rate"]


def _resolve_adapter(path: str | None, cfg: dict) -> tuple[str | None, bool]:
    """(adapter, exists). Relative paths are under output_root; Hub ids pass through."""
    if not path:
        return None, True
    p = Path(path)
    if not p.is_absolute():
        local = ROOT / cfg.get("output_root", "outputs") / p
        if local.exists():
            return str(local), True
        if (ROOT / p).exists():
            return str(ROOT / p), True
        if path.count("/") == 1 and not path.startswith((".", "outputs")):
            return path, True  # a Hub repo id
        return str(local), False
    return str(p), p.exists()


def _scored_file(cfg: dict, name: str) -> Path | None:
    """outputs/evaluate/<name>/scored.jsonl, else the copy pushed to the logs dataset."""
    local = work_dir(cfg, "evaluate", name) / "scored.jsonl"
    if local.exists():
        return local
    if cfg.get("dry_run"):
        return None
    try:
        from huggingface_hub import hf_hub_download

        from ..hub import repo_id

        return Path(hf_hub_download(repo_id(cfg, "logs"), f"evaluate/{name}/scored.jsonl",
                                    repo_type="dataset"))
    except Exception:
        return None


def run(cfg: dict, args) -> tuple[str, dict]:
    e = cfg.get("evaluate", {})
    trials = int(e.get("trials", 5))
    tools = load_tools(cfg)
    rows = load_split(cfg, e.get("split", "test"))
    rewards = make_audit_rewards(tools, cfg)
    checkpoints = e.get("checkpoints") or [{"name": "base"}]
    if args.run:
        checkpoints = [c for c in checkpoints if c["name"] == args.run]
        if not checkpoints:
            raise SystemExit(f"--run {args.run}: no such checkpoint in evaluate.checkpoints")
    out: dict = {"split": e.get("split", "test"), "n_tasks": len(rows), "trials": trials,
                 "independent_trials": e.get("independent_trials", False),
                 "checkpoints": {}, "comparisons": {}}

    for ck in checkpoints:
        name = ck["name"]
        backend = ck.get("backend", "local")
        if getattr(args, "compare_only", False):
            scored_path = _scored_file(cfg, name)
            if scored_path is None:
                out["checkpoints"][name] = {"skipped": "no scored file (run evaluate --run first)"}
            else:
                scored = read_jsonl(scored_path)
                out["checkpoints"][name] = {
                    "from": str(scored_path),
                    "summary": summarise(scored, k=trials,
                                         independent=e.get("independent_trials", False))}
            continue
        if cfg.get("dry_run") and backend != "local":
            out["checkpoints"][name] = {"skipped": "router checkpoint in a dry run"}
            continue
        spec = {"backend": backend, **{k: v for k, v in ck.items() if k not in ("name",)}}
        if backend == "local":
            spec.setdefault("model", cfg["student"]["model"])
            if cfg.get("dry_run"):
                spec["model"] = cfg["student"]["model"]
            if ck.get("base_adapter"):
                spec["base_adapter"], _ = _resolve_adapter(ck.get("base_adapter"), cfg)
            spec["adapter"], ok = _resolve_adapter(ck.get("adapter"), cfg)
            if not ok:
                msg = f"adapter not found: {spec['adapter']}"
                if e.get("skip_missing", True):
                    print(f"[evaluate] skip {name}: {msg}")
                    out["checkpoints"][name] = {"skipped": msg}
                    continue
                raise FileNotFoundError(msg)
        budget = make_budget(spec) if backend == "router" else None
        engine, sp, desc = make_engine(cfg, spec, tools, budget)
        fn = build_rollout_fn(engine, tools, system_prompt=sp,
                              max_steps=cfg.get("rollout", {}).get("max_steps_per_turn", 4))
        print(f"[evaluate:{name}] {len(rows)} tasks x {trials} trials")
        records, status = run_rollouts(fn, rows, trials=trials, gen_kwargs=gen_kwargs(cfg, "evaluate"))
        scored = score_records(records, rewards, audit_weights(cfg), lenient=lenient_scorer(cfg))
        d = work_dir(cfg, "evaluate", name)
        write_jsonl(d / "rollouts.jsonl", records)
        write_jsonl(d / "scored.jsonl", scored)
        out["checkpoints"][name] = {
            "engine": desc, "rollout_status": status,
            "budget": budget.summary() if budget else None,
            "summary": summarise(scored, k=trials, independent=e.get("independent_trials", False)),
            "files": {"rollouts": str(d / "rollouts.jsonl"), "scored": str(d / "scored.jsonl")},
        }
        if cfg.get("push"):
            out["checkpoints"][name]["hub"] = push_folder(
                cfg, "logs", d, repo_type="dataset", path_in_repo=f"evaluate/{name}")
        free(engine, fn)

    for base, cand in e.get("comparisons") or []:
        paths = [_scored_file(cfg, n) for n in (base, cand)]
        if not all(paths):
            out["comparisons"][f"{base}_vs_{cand}"] = {"skipped": "scored files missing"}
            continue
        b, c = (read_jsonl(p) for p in paths)
        out["comparisons"][f"{base}_vs_{cand}"] = {
            "baseline": base, "candidate": cand,
            "method": "paired sign-flip permutation test (p_value) and centred bootstrap "
                      "(p_bootstrap, ci95) over per-task means (trials averaged); not "
                      "auditkit's paired_bootstrap (uncentred, FINDINGS.md A3)",
            "metrics": bootstrap_compare(b, c, e.get("compare_metrics", COMPARE_METRICS),
                                         n_resamples=e.get("n_resamples", 10000),
                                         seed=cfg.get("seed", 42)),
        }
    name = f"14_evaluate_{args.run}" if args.run else "14_evaluate"
    out["mode"] = "compare_only" if getattr(args, "compare_only", False) else "rollouts"
    return name, out
