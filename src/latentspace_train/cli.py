"""Single CLI for every stage.

    python -m latentspace_train <stage> --config configs/<NN_stage>.yaml [options]

Options shared by all stages:
    --set key.path=value   override any config value (YAML-parsed), repeatable
    --run NAME             GRPO preset (run_A | run_B); evaluate: limit to one checkpoint
    --push                 push adapters / logs / datasets to the HF Hub (repo ids in config)
    --dry-run              CPU wiring check: tiny random Llama, stub tasks, 3 steps,
                           results under results/smoke/ (no network)
    --limit N              only the first N task rows
    --max-usd X            teacher_collect / api baseline: stop when estimated spend reaches X
    --compare-only         evaluate: only summarise and compare already scored checkpoints
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

from . import STAGES
from .config import ROOT, load_config, select_run, set_dotted

DEFAULT_CONFIGS = {
    "baseline": "configs/10_baseline.yaml",
    "teacher_collect": "configs/11_teacher_collect.yaml",
    "distill": "configs/12_distill.yaml",
    "grpo": "configs/13_grpo.yaml",
    "evaluate": "configs/14_evaluate.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="latentspace_train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--config", default=None, help="stage YAML (default: configs/NN_<stage>.yaml)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--run", default=None)
    p.add_argument("--push", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-usd", type=float, default=None)
    p.add_argument("--compare-only", action="store_true",
                   help="evaluate: no rollouts; summarise + compare the scored files of every "
                        "checkpoint (local, or the logs dataset on the Hub)")
    p.add_argument("--print-config", action="store_true", help="print the resolved config and exit")
    return p


def resolve(args) -> dict:
    cfg = load_config(args.config or DEFAULT_CONFIGS[args.stage], args.overrides)
    cfg["stage"] = args.stage
    if args.stage == "grpo":
        cfg = select_run(cfg, args.run)
    if args.limit is not None:
        set_dotted(cfg, "tasks.limit", args.limit)
    if args.max_usd is not None:
        set_dotted(cfg, "teacher.max_usd", args.max_usd)
        set_dotted(cfg, "baseline.api.max_usd", args.max_usd)
    cfg["push"] = bool(args.push)
    if args.dry_run:
        from .runtime import work_dir
        from .smoke import apply_dry_run

        cfg["output_root"] = "outputs/smoke"
        cfg = apply_dry_run(cfg, work_dir(cfg, "_models"))
        if args.push:
            raise SystemExit("--push is not allowed with --dry-run")
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sys.path.insert(0, str(ROOT / "src"))
    started = time.time()
    cfg = resolve(args)
    if args.print_config:
        print(json.dumps(cfg, indent=2, default=str))
        return 0

    from .runtime import prepare_env, seed_everything, work_dir, write_results

    prepare_env(work_dir(cfg, "_agenttune"))
    seed_everything(int(cfg.get("seed", 42)))
    stage = importlib.import_module(f".stages.{args.stage}", __package__)
    name, payload = stage.run(cfg, args)
    path = write_results(cfg, name, payload, started)
    print(f"[latentspace_train] {args.stage}: wrote {Path(path).relative_to(ROOT)}")
    return 0
