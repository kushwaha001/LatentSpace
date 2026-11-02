"""Stage 4, grpo: ``create_agentic_trainer("grpo")`` with LoRA, two presets that differ only
in the reward (configs/13_grpo.yaml, ``runs:``):

- ``run_A``: AgentTune's default tool reward, ``hybrid_prm_reward`` from ``REWARD_REGISTRY``,
  upstream and unpatched (FINDINGS.md B1-B3: it scores a valid call 0.0 and no call 1.0);
- ``run_B``: ``AuditRewards(tools).train_time()`` with ``DEFAULT_WEIGHTS``
  (validity 1, F1 1, permission 1, confirm 1, redundancy 0.25).

Seed, steps, data, model, LoRA and every hyperparameter come from the shared part of the
config. ``run()`` records a fingerprint of those shared settings, so the two results files
can be checked for identical setups.

After training, the run's own training rollouts (``trajectories.jsonl``) are scored with
the audit metrics, so run_A's results show what its reward could not see.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from . import precision
from ..engines import load_local_model
from ..hub import push_folder
from ..runtime import read_jsonl, work_dir
from ..scoring import audit_weights, make_audit_rewards, score_records, summarise
from ..tasks import load_split, load_tools, system_prompt, to_dataset_rows

REWARD_KEYS = ("reward", "reward_funcs", "reward_weights")


def shared_fingerprint(cfg: dict) -> str:
    """Hash of everything except the reward choice and bookkeeping: equal for run_A/run_B."""
    skip = {"reward", "run_name", "_run_presets", "push", "_config_path", "hub"}
    shared = {k: v for k, v in cfg.items() if k not in skip}
    return hashlib.sha256(json.dumps(shared, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _as_text(x: Any) -> Any:
    if isinstance(x, list) and x and isinstance(x[0], dict):
        return "\n".join(str(m.get("content") or "") for m in x)
    return x


def text_inputs(fn):
    """Input adapter for rewards that only accept strings. ``hybrid_prm_reward`` calls
    ``re.sub`` on each completion, so with a conversational dataset (TRL then passes
    message-list completions) it raises ``TypeError``; ``combine_rewards`` retries it with
    ``completions`` as the first positional argument, which lands on ``prompts`` and fails
    again. The scoring itself is untouched: only the input shape changes."""

    def wrapped(prompts=None, completions=None, **kwargs):
        return fn(prompts=[_as_text(p) for p in (prompts or [])],
                  completions=[_as_text(c) for c in completions], **kwargs)

    wrapped.__name__ = wrapped.__qualname__ = fn.__name__
    wrapped.__doc__ = fn.__doc__
    return wrapped


def build_rewards(cfg: dict, tools: list) -> tuple[list, list | None, dict]:
    r = cfg.get("reward", {})
    kind = r.get("kind")
    if kind == "agenttune_default":
        from agenttune.agentic.rewards.builtin_rewards import REWARD_REGISTRY

        names = r.get("funcs", ["hybrid_prm_reward"])
        funcs = [REWARD_REGISTRY[n] for n in names]
        if r.get("text_inputs", True):
            funcs = [text_inputs(f) for f in funcs]
        return funcs, r.get("weights"), {
            "kind": kind, "funcs": names, "weights": r.get("weights"), "patched": False,
            "input_adapter": "message-list prompts/completions flattened to text before the "
                             "unmodified reward is called" if r.get("text_inputs", True) else None}
    if kind == "audit":
        rewards = make_audit_rewards(tools, cfg)
        funcs = rewards.train_time(include_f1=r.get("include_f1", True))
        w = audit_weights(cfg)
        weights = [w[f.__name__] for f in funcs]
        return funcs, weights, {"kind": kind, "funcs": [f.__name__ for f in funcs],
                                "weights": weights}
    raise ValueError(f"reward.kind must be agenttune_default | audit, got {kind!r}")


def _trainer_kwargs(cfg: dict, prec: str) -> dict[str, Any]:
    t = dict(cfg.get("training") or {})
    t["fp16"], t["bf16"] = prec == "fp16", prec == "bf16"
    t["seed"] = cfg.get("seed", 42)
    t.setdefault("report_to", "none")
    t.setdefault("logging_steps", 1)
    if t.get("gradient_checkpointing"):
        t.setdefault("gradient_checkpointing_kwargs", {"use_reentrant": False})
    if not cfg.get("dry_run"):
        t.pop("use_cpu", None)
    return t


def run(cfg: dict, args) -> tuple[str, dict]:
    import torch
    from datasets import Dataset

    from agenttune.core.backend_factory import create_agentic_trainer

    run_name = cfg.get("run_name") or "run"
    tools = load_tools(cfg)
    rows = load_split(cfg, cfg.get("grpo", {}).get("split", "train"))
    ds = Dataset.from_list(to_dataset_rows(rows))
    funcs, weights, reward_desc = build_rewards(cfg, tools)
    prec = precision(cfg)
    out_dir = work_dir(cfg, "grpo", run_name)
    (out_dir / "trajectories.jsonl").unlink(missing_ok=True)

    student = dict(cfg.get("student") or {})
    model, tok = load_local_model(student, prec, for_training=True)
    lora = cfg.get("lora", {})
    peft_config = None
    if not student.get("adapter"):  # a distilled adapter is loaded trainable instead
        peft_config = {
            "r": lora.get("r", 16), "lora_alpha": lora.get("alpha", 32),
            "lora_dropout": lora.get("dropout", 0.05), "bias": "none", "task_type": "CAUSAL_LM",
            "target_modules": lora.get("target_modules") or "all-linear",
        }
    tk = _trainer_kwargs(cfg, prec)
    max_steps_per_turn = tk.pop("max_steps_per_turn", 3)
    trainer = create_agentic_trainer(
        "grpo",
        backend="trl",
        model=model,
        processing_class=tok,
        tools=tools,
        reward_funcs=funcs,
        reward_weights=weights,
        train_dataset=ds,
        output_dir=str(out_dir),
        system_prompt=system_prompt(cfg),
        max_steps_per_turn=max_steps_per_turn,
        peft_config=peft_config,
        **tk,
    )
    error = None
    try:
        trainer.train()
    except Exception as e:  # recorded, never hidden
        error = f"{type(e).__name__}: {e}"
    hf = trainer.trainer
    history = hf.state.log_history if hf is not None else []
    # LoRA B matrices start at zero, so a non-zero B means the optimizer updated it
    lora_b = [p for n, p in hf.model.named_parameters() if "lora_B" in n] if hf is not None else []
    changed = (f"{sum(bool(torch.any(p != 0)) for p in lora_b)}/{len(lora_b)} lora_B tensors non-zero"
               if peft_config is not None else "n/a (started from a trained adapter)")
    adapter_dir = out_dir / "adapter"
    if hf is not None and error is None:
        hf.save_model(str(adapter_dir))
        tok.save_pretrained(str(adapter_dir))

    traj_path = out_dir / "trajectories.jsonl"
    audit_of_training = None
    if traj_path.exists():
        audit_of_training = audit_training_rollouts(cfg, tools, read_jsonl(traj_path))

    out = {
        "run": run_name,
        "reward": reward_desc,
        "shared_settings_fingerprint": shared_fingerprint(cfg),
        "student": {k: student.get(k) for k in ("model", "adapter", "quantization")},
        "precision": prec,
        "lora": peft_config or {"adapter": student.get("adapter")},
        "n_train_rows": len(ds),
        "global_step": hf.state.global_step if hf is not None else None,
        "error": error,
        "trainable_tensors_updated": changed,
        "log_history": [
            {k: v for k, v in h.items() if not isinstance(v, (list, dict))} for h in history
        ],
        "trajectories_written": sum(1 for _ in open(traj_path)) if traj_path.exists() else 0,
        "audit_of_training_rollouts": audit_of_training,
        "adapter_dir": str(adapter_dir) if adapter_dir.exists() else None,
    }
    if cfg.get("push") and adapter_dir.exists():
        out["hub"] = push_folder(cfg, f"grpo_{run_name}", out_dir,
                                 allow_patterns=["adapter/**", "trajectories.jsonl"],
                                 message=f"GRPO {run_name}: {reward_desc['kind']} reward")
    return f"13_grpo_{run_name}", out


def audit_training_rollouts(cfg: dict, tools: list, trajs: list[dict], bins: int = 4) -> dict:
    """Audit metrics of the rollouts GRPO trained on, in ``bins`` chunks in step order."""
    recs = []
    for t in trajs:
        meta = t.get("metadata") or {}
        conv = meta.get("conversation") or []
        prompt = next((m.get("content") for m in conv if m.get("role") == "user"), t.get("task"))
        recs.append({"task_id": t.get("task"), "language": "train", "prompt": prompt,
                     "response": t.get("final_response", ""), "conversation": conv,
                     "tool_call_count": meta.get("tool_call_count"),
                     "expected_tool_calls": None})
    if not recs:
        return {}
    scored = score_records(recs, make_audit_rewards(tools, cfg), audit_weights(cfg))
    size = max(1, -(-len(scored) // bins))
    keys = ("tool_call_validity", "audit_tool_validity_reward", "n_malformed", "tool_call_count",
            "audit_confirm_irreversible_penalty", "agent_loop_detection")
    chunks = []
    for i in range(0, len(scored), size):
        s = summarise(scored[i:i + size])["overall"]
        chunks.append({"rollouts": f"{i}-{min(i + size, len(scored)) - 1}",
                       **{k: s.get(k) for k in keys}})
    return {"n_rollouts": len(scored), "note": "reference-free (training rows' references are "
            "not joined back to the written trajectories)", "by_chunk": chunks}
