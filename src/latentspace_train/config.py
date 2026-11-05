"""YAML configs: one file per stage, ``extends:`` a shared base, ``--set a.b=c`` overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]

# Tiny Aya family. Short names, the bare variant word and full repo ids are all accepted.
TINY_AYA = {
    "tiny-aya-global": "CohereLabs/tiny-aya-global",
    "tiny-aya-fire": "CohereLabs/tiny-aya-fire",
    "tiny-aya-water": "CohereLabs/tiny-aya-water",
    "tiny-aya-earth": "CohereLabs/tiny-aya-earth",
}
AYA_EXPANSE = {
    "aya-expanse-8b": "CohereLabs/aya-expanse-8b",
    "aya-expanse-32b": "CohereLabs/aya-expanse-32b",
}
MODEL_ALIASES = {
    **TINY_AYA,
    **{k.removeprefix("tiny-aya-"): v for k, v in TINY_AYA.items()},  # "fire", "global", ...
    **AYA_EXPANSE,
}


def resolve_model(name: str | None) -> str | None:
    """Map a short model name to its Hub id. Paths and other ids pass through unchanged."""
    if not name:
        return name
    return MODEL_ALIASES.get(str(name).lower(), name)


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_file(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"config extends cycle: {' -> '.join(map(str, seen + (path,)))}")
    data = yaml.safe_load(path.read_text()) or {}
    parents = data.pop("extends", None) or []
    if isinstance(parents, str):
        parents = [parents]
    merged: dict = {}
    for p in parents:
        merged = deep_merge(merged, _load_file((path.parent / p), seen + (path,)))
    return deep_merge(merged, data)


def set_dotted(cfg: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if not isinstance(node.get(k), dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def get_dotted(cfg: dict, dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for k in dotted.split("."):
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node


def parse_override(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ValueError(f"--set expects key=value, got {item!r}")
    key, raw = item.split("=", 1)
    return key.strip(), yaml.safe_load(raw) if raw != "" else ""


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load a stage config (with ``extends``), apply ``key.path=value`` overrides and
    resolve model aliases. The result is a plain dict that is written verbatim into the
    stage's results JSON."""
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    cfg = _load_file(path)
    for item in overrides or []:
        set_dotted(cfg, *parse_override(item))
    cfg["_config_path"] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    for key in ("student.model", "teacher.model", "teacher.local_fallback.model"):
        val = get_dotted(cfg, key)
        if val:
            set_dotted(cfg, key, resolve_model(val))
    return cfg


def select_run(cfg: dict, run: str | None) -> dict:
    """GRPO configs carry ``runs: {run_A: {...}, run_B: {...}}``. Each preset is merged on
    top of the shared settings, so the runs differ only where the preset says so."""
    runs = cfg.get("runs") or {}
    if not runs:
        return cfg
    if run is None:
        raise ValueError(f"this config defines runs {sorted(runs)}; pass --run <name>")
    if run not in runs:
        raise ValueError(f"unknown run {run!r}; choose from {sorted(runs)}")
    out = deep_merge({k: v for k, v in cfg.items() if k != "runs"}, runs[run])
    out["run_name"] = run
    out["_run_presets"] = sorted(runs)
    return out
