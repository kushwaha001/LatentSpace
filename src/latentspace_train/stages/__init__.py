"""Stage implementations. Each module has ``run(cfg, args) -> (results_name, payload)``."""

from __future__ import annotations

import gc
from typing import Any

from ..engines import Budget, fallback_system_prompt, make_local_engine, make_router_engine
from ..runtime import resolve_precision
from ..tasks import system_prompt as task_system_prompt


def precision(cfg: dict) -> str:
    return resolve_precision(cfg.get("precision", "auto"))


def gen_kwargs(cfg: dict, section: str | None = None) -> dict[str, Any]:
    g = dict(cfg.get("generation") or {})
    if section:
        g.update((cfg.get(section) or {}).get("generation") or {})
    return g


def make_budget(spec: dict) -> Budget:
    price = spec.get("price_usd_per_m") or {}
    return Budget(
        max_usd=spec.get("max_usd"),
        usd_per_m_input=float(price.get("input", 0.0)),
        usd_per_m_output=float(price.get("output", 0.0)),
    )


def make_engine(cfg: dict, spec: dict, tools: list, budget: Budget | None = None):
    """(engine, system_prompt, description) for a ``backend: local | router`` spec.

    Local weights: AgentTune adds its fallback prompt itself (the template ignores tools).
    Router in prompt mode: we pass the identical prompt as the system prompt."""
    base_sp = task_system_prompt(cfg)
    backend = spec.get("backend", "local")
    if backend == "router":
        router = {**(cfg.get("router") or {}), **spec}
        engine = make_router_engine(router, budget)
        sp = fallback_system_prompt(tools, base_sp) if router.get("tool_mode", "prompt") == "prompt" else base_sp
        desc = {"backend": "router", "model": router["model"], "base_url": router.get("base_url"),
                "tool_mode": router.get("tool_mode", "prompt"), "bill_to": router.get("bill_to")}
        return engine, sp, desc
    if backend == "local":
        local = {**(cfg.get("student") or {}), **spec}
        prec = precision(cfg)
        engine = make_local_engine(local, prec)
        desc = {"backend": "local", "model": local["model"], "adapter": local.get("adapter"),
                "quantization": local.get("quantization"), "precision": prec}
        return engine, base_sp, desc
    raise ValueError(f"unknown backend {backend!r} (local | router)")


def free(*objs) -> None:
    for o in objs:
        del o
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
