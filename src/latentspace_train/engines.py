"""Rollout engines: local Hugging Face weights, or the Hugging Face router (OpenAI-compatible
Inference Providers endpoint) through AgentTune's LiteLLM ``APIRolloutEngine``.

Aya Expanse and Tiny Aya chat templates ignore ``tools`` (CODE_MAP.md section 4), so for
local weights AgentTune adds its Cohere ``DEFAULT_TOOLS_FALLBACK_PROMPT`` itself. An API
engine has no tokenizer, so AgentTune would send the schemas as native ``tools=`` instead.
``RouterEngine`` in ``tool_mode: prompt`` (default) makes an API model see exactly what the
student sees: the same fallback prompt in the system message, no native ``tools``, tool
results folded into user turns. ``tool_mode: native`` passes the schemas through.

``Budget`` tracks tokens and estimated USD per call and raises ``BudgetExceeded`` before
the next call once ``max_usd`` is reached.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any

HF_ROUTER = "https://router.huggingface.co/v1"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    """Token and cost accounting. Prices are USD per million tokens."""

    max_usd: float | None = None
    usd_per_m_input: float = 0.0
    usd_per_m_output: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    failed_calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def usd(self) -> float:
        return (
            self.prompt_tokens * self.usd_per_m_input
            + self.completion_tokens * self.usd_per_m_output
        ) / 1e6

    @property
    def exhausted(self) -> bool:
        return self.max_usd is not None and self.usd >= self.max_usd

    def check(self) -> None:
        if self.exhausted:
            raise BudgetExceeded(f"budget reached: ${self.usd:.4f} >= ${self.max_usd:.2f}")

    def add(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.prompt_tokens += int(prompt_tokens or 0)
            self.completion_tokens += int(completion_tokens or 0)
            self.calls += 1

    def summary(self) -> dict[str, Any]:
        return {
            "max_usd": self.max_usd,
            "usd_estimated": round(self.usd, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "api_calls": self.calls,
            "failed_calls": self.failed_calls,
            "price_usd_per_m": {"input": self.usd_per_m_input, "output": self.usd_per_m_output},
            "note": "estimate from token counts x configured prices; check the HF billing page",
        }


def fallback_system_prompt(tools: list, base: str | None = None) -> str:
    """The system prompt AgentTune builds for a template that ignores ``tools``
    (rollout_factory.py:2193-2205), reproduced with AgentTune's own constant and builders."""
    from agenttune.agentic.rollout_engines.rollout_factory import DEFAULT_TOOLS_FALLBACK_PROMPT

    from audit_rewards import tool_schemas

    schemas = json.dumps(tool_schemas(tools), indent=2)
    try:
        fb = DEFAULT_TOOLS_FALLBACK_PROMPT.format(tools=schemas)
    except (KeyError, ValueError, IndexError):
        fb = DEFAULT_TOOLS_FALLBACK_PROMPT.replace("{tools}", schemas)
    return f"{base}\n\n{fb}" if base else fb


def _prompt_mode_messages(messages: list[dict]) -> list[dict]:
    """Fold tool turns into user turns and drop structured ``tool_calls`` (the raw call text
    stays in ``content``), the way AgentTune renders them for a tool-ignoring template."""
    from agenttune.agentic.rollout_engines.rollout_factory import fold_tool_messages_into_user

    out = []
    for m in fold_tool_messages_into_user([dict(m) for m in messages]):
        m = {k: v for k, v in m.items() if k in ("role", "content")}
        if m["role"] == "assistant" and not m.get("content"):
            m["content"] = ""
        if out and out[-1]["role"] == m["role"] and m["role"] != "system":
            out[-1] = {**out[-1], "content": f"{out[-1]['content']}\n\n{m['content']}"}
        else:
            out.append(m)
    return out


def make_router_engine(spec: dict, budget: Budget | None = None):
    """An ``APIRolloutEngine`` for the HF router (or any OpenAI-compatible endpoint).

    spec keys: model ("CohereLabs/aya-expanse-32b:cohere"), base_url, api_key_env
    ("HF_TOKEN"), bill_to (optional HF org to bill), tool_mode ("prompt" | "native"),
    max_retries, timeout.
    """
    from agenttune.agentic.rollout_engines.api_engine import APIRolloutEngine

    key_env = spec.get("api_key_env", "HF_TOKEN")
    api_key = os.environ.get(key_env)
    if not api_key:
        raise RuntimeError(
            f"{key_env} is not set. On Kaggle: Add-ons > Secrets > {key_env}; on Colab: the key "
            "icon > Secrets; locally: export HF_TOKEN=hf_..."
        )
    extra: dict[str, Any] = {}
    if spec.get("bill_to"):
        extra["extra_headers"] = {"X-HF-Bill-To": spec["bill_to"]}
    if spec.get("timeout"):
        extra["timeout"] = spec["timeout"]

    class RouterEngine(APIRolloutEngine):
        tool_mode = spec.get("tool_mode", "prompt")

        def generate(self, prompts, tools=None, gen_cfg=None):
            if budget is not None:
                budget.check()
            if self.tool_mode == "prompt":
                from agenttune.agentic.rollout_engines.api_engine import _to_messages

                prompts, tools = _prompt_mode_messages(_to_messages(prompts)), None
            try:
                out = super().generate(prompts, tools=tools, gen_cfg=gen_cfg)
            except Exception:
                if budget is not None:
                    budget.failed_calls += 1
                raise
            if budget is not None:
                meta = out.get("metadata", {})
                budget.add(meta.get("prompt_tokens", 0), meta.get("completion_tokens", 0))
            return out

    # LiteLLM routes "openai/<id>" to an OpenAI-compatible server at base_url.
    model = spec["model"]
    litellm_model = model if model.startswith("openai/") else f"openai/{model}"
    return RouterEngine(
        model=litellm_model,
        api_key=api_key,
        base_url=spec.get("base_url", HF_ROUTER),
        max_retries=spec.get("max_retries", 3),
        extra_litellm_kwargs=extra,
    )


def load_local_model(spec: dict, precision: str, *, for_training: bool = False):
    """Load model + tokenizer from local weights / the Hub with memory-safe defaults.

    spec keys: model, quantization ("4bit" | "8bit" | null), device_map, attn_implementation,
    trust_remote_code. 4-bit uses NF4 with double quantization and the compute dtype of
    ``precision`` (fp16 on T4).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .runtime import torch_dtype

    name = _local_or_hub(spec["model"])
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=spec.get("trust_remote_code", False))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch_dtype(precision)
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": spec.get("trust_remote_code", False),
    }
    if torch.cuda.is_available():
        # one GPU per process: DDP / Accelerate places the replica; device_map would shard it
        kwargs["device_map"] = spec.get("device_map") or {"": int(os.environ.get("LOCAL_RANK", 0))}
    if spec.get("attn_implementation"):
        kwargs["attn_implementation"] = spec["attn_implementation"]
    q = spec.get("quantization")
    if q and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig

        if q == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
        elif q == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"unknown quantization {q!r}")
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if q and torch.cuda.is_available() and for_training:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if spec.get("adapter"):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, _local_or_hub(spec["adapter"]),
                                          is_trainable=for_training)
    if not for_training:
        model.eval()
    return model, tok


def _local_or_hub(path: str) -> str:
    """A path relative to the repo root if it exists there, else as given (a Hub id)."""
    from pathlib import Path

    from .config import ROOT

    p = Path(path)
    if not p.is_absolute() and (ROOT / p).exists():
        return str(ROOT / p)
    if p.is_absolute() or path.startswith((".", "outputs/")) or path.count("/") != 1:
        if not p.exists():
            raise FileNotFoundError(
                f"{path} not found. In a fresh Kaggle/Colab session, local outputs are gone: "
                "point to the Hub copy instead, e.g. --set student.adapter=<namespace>/<repo>")
    return path


def make_local_engine(spec: dict, precision: str):
    from agenttune.agentic.rollout_engines.transformers_engine import TransformersRolloutEngine

    model, tok = load_local_model(spec, precision)
    return TransformersRolloutEngine(model, tok)
