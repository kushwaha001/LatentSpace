"""Stage 1, baseline: the untrained student on the eval split.

``baseline.modes`` picks the engines, and the results record which ones ran:
- ``local``: Tiny Aya weights on the GPU (4-bit by default), AgentTune's
  ``create_rollout_fn`` with its Cohere ``tools_fallback_prompt``;
- ``api``: the same model through the HF router (``CohereLabs/tiny-aya-global:cohere``),
  a cheap cross-check that needs no GPU. Same fallback prompt, sent as the system prompt.
Both are scored with ``AuditRewards.report()`` and the train-time rewards.
"""

from __future__ import annotations

from . import free, gen_kwargs, make_budget, make_engine
from ..hub import push_folder
from ..rollouts import build_rollout_fn, run_rollouts
from ..runtime import work_dir, write_jsonl
from ..scoring import audit_weights, make_audit_rewards, score_records, summarise
from ..tasks import load_split, load_tools


def run(cfg: dict, args) -> tuple[str, dict]:
    b = cfg.get("baseline", {})
    split = b.get("split", "eval")
    trials = int(b.get("trials", 1))
    tools = load_tools(cfg)
    rows = load_split(cfg, split)
    rewards = make_audit_rewards(tools, cfg)
    modes = b.get("modes", ["local"])
    if cfg.get("dry_run"):
        modes = [m for m in modes if m == "local"] or ["local"]
    out: dict = {"split": split, "n_tasks": len(rows), "trials": trials, "modes": {},
                 "baseline_used": modes}

    for mode in modes:
        spec = {"backend": "local"} if mode == "local" else {"backend": "router", **b.get("api", {})}
        budget = make_budget(spec) if mode == "api" else None
        engine, sp, desc = make_engine(cfg, spec, tools, budget)
        fn = build_rollout_fn(engine, tools, system_prompt=sp,
                              max_steps=cfg.get("rollout", {}).get("max_steps_per_turn", 4))
        print(f"[baseline:{mode}] {len(rows)} tasks x {trials} trial(s) with {desc['model']}")
        records, status = run_rollouts(fn, rows, trials=trials, gen_kwargs=gen_kwargs(cfg, "baseline"))
        scored = score_records(records, rewards, audit_weights(cfg))
        d = work_dir(cfg, "baseline", mode)
        write_jsonl(d / "rollouts.jsonl", records)
        write_jsonl(d / "scored.jsonl", scored)
        out["modes"][mode] = {
            "engine": desc,
            "rollout_status": status,
            "summary": summarise(scored, k=trials if trials > 1 else None),
            "files": {"rollouts": str(d / "rollouts.jsonl"), "scored": str(d / "scored.jsonl")},
            "budget": budget.summary() if budget else None,
        }
        if cfg.get("push"):
            out["modes"][mode]["hub"] = push_folder(
                cfg, "logs", d, repo_type="dataset", path_in_repo=f"baseline/{mode}")
        free(engine, fn)
    return "10_baseline", out
