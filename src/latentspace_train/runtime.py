"""Process setup, hardware detection, seeding and the results/NN_*.json writer."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .config import ROOT

PACKAGES = (
    "torch", "transformers", "trl", "peft", "accelerate", "bitsandbytes", "datasets",
    "agenttune", "auditkit", "litellm", "huggingface_hub",
)


def prepare_env(work_dir: Path) -> None:
    """Environment variables AgentTune reads at import time (FINDINGS.md, environment notes):
    keep its score log and judgment file out of the CWD and the source tree."""
    work_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AGENTTUNE_SCORE_LOG", str(work_dir / "scores.jsonl"))
    os.environ.setdefault("JUDGMENTS_FILE", str(work_dir / "judgments.jsonl"))
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda": False, "gpus": [], "bf16_supported": False}
    try:
        import torch

        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["gpus"].append(
                    {"name": p.name, "memory_gb": round(p.total_memory / 2**30, 1),
                     "capability": f"{p.major}.{p.minor}"}
                )
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except ImportError:
        pass
    return info


def resolve_precision(precision: str, hw: dict | None = None) -> str:
    """``auto`` -> bf16 on Ampere+ (L4, A10G, A100), fp16 on T4 (no bf16), fp32 on CPU."""
    hw = hw or hardware()
    if precision != "auto":
        return precision
    if not hw["cuda"]:
        return "fp32"
    caps = [float(g["capability"]) for g in hw["gpus"]] or [0.0]
    return "bf16" if hw["bf16_supported"] and min(caps) >= 8.0 else "fp16"


def torch_dtype(precision: str):
    import torch

    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": sys.version.split()[0]}
    for p in PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def _git(*args: str, cwd: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except Exception:
        return None


def provenance() -> dict[str, Any]:
    agenttune_dir = None
    try:
        import agenttune

        agenttune_dir = Path(agenttune.__file__).resolve().parents[2]
    except Exception:
        pass
    return {
        "commit": _git("rev-parse", "HEAD", cwd=ROOT),
        "dirty": bool(_git("status", "--porcelain", cwd=ROOT)),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=ROOT),
        "agenttune_commit": _git("rev-parse", "HEAD", cwd=agenttune_dir) if agenttune_dir else None,
        "platform": platform.platform(),
        "hostname_kind": detect_host(),
    }


def detect_host() -> str:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists():
        return "kaggle"
    if "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ:
        return "colab"
    if os.environ.get("SPACE_ID"):
        return "hf_space"
    return "other"


def results_path(cfg: dict, name: str) -> Path:
    base = Path(cfg.get("results_dir", "results"))
    if not base.is_absolute():
        base = ROOT / base
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{name}.json"


def write_results(cfg: dict, name: str, payload: dict[str, Any], started: float) -> Path:
    """results/<name>.json = {stage payload, config, seed, versions, commit, hardware}."""
    path = results_path(cfg, name)
    record = {
        "stage": cfg.get("stage"),
        "run_name": cfg.get("run_name"),
        "dry_run": bool(cfg.get("dry_run")),
        "seed": cfg.get("seed"),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "seconds": round(time.time() - started, 1),
        **payload,
        "provenance": provenance(),
        "versions": versions(),
        "hardware": hardware(),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_") or k == "_config_path"},
    }
    path.write_text(json.dumps(record, indent=2, default=str, ensure_ascii=False) + "\n")
    return path


def work_dir(cfg: dict, *parts: str) -> Path:
    base = Path(cfg.get("output_root", "outputs"))
    if not base.is_absolute():
        base = ROOT / base
    p = base.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
