"""Stage 3, distill: LoRA SFT of Tiny Aya on the filtered teacher trajectories, through
AgentTune's ``create_distill_trainer`` (``Project.distill`` -> TRL ``SFTTrainer``).

Each kept trajectory's transcript becomes a full-tier AgentTune ``EventLog``. The rows the
student trains on are rendered the way the student will see them at rollout time:

- the system prompt is the task system prompt plus AgentTune's Cohere
  ``tools_fallback_prompt`` with the tool schemas (exactly what ``_execute_trajectory``
  adds for a template that ignores ``tools``);
- tool results are folded into user turns with AgentTune's ``fold_tool_messages_into_user``;
- tool calls are written in the format that prompt asks for,
  ``[{"tool_name": ..., "parameters": {...}}]`` (``distill.tool_call_format: cohere``), or
  AgentTune's ``<tool_call>{...}</tool_call>`` (``agenttune``). Both parse with
  ``rollout_factory._extract_tool_calls``.

The built-in factory is extended (``QuantizedSFTFactory``) to load the student in 4-bit /
fp16 for 16 GB GPUs and to turn on gradient checkpointing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import precision
from ..config import ROOT
from ..engines import fallback_system_prompt, load_local_model
from ..hub import push_folder
from ..runtime import read_jsonl, work_dir
from ..tasks import load_tools, system_prompt


def cohere_call_text(calls: list[dict]) -> str:
    """The call format DEFAULT_TOOLS_FALLBACK_PROMPT asks for."""
    out = []
    for c in calls:
        fn = c.get("function", c)
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        out.append({"tool_name": fn.get("name"), "parameters": args})
    return json.dumps(out, ensure_ascii=False)


def conversation_to_eventlog(conversation: list[dict], call_format: str = "cohere"):
    """A rollout transcript -> full-tier ``EventLog`` (system turns dropped; they are
    re-added by the factory). Tool turns are folded into user turns first, as the rollout
    does for Tiny Aya, so the student sees the same text at train and test time."""
    from agenttune.agentic.events import Event, EventKind, EventLog
    from agenttune.agentic.rollout_engines.rollout_factory import fold_tool_messages_into_user

    log = EventLog(tier="full")
    for m in fold_tool_messages_into_user([dict(m) for m in conversation]):
        role, content = m.get("role"), m.get("content") or ""
        if role == "system":
            continue
        if role == "user":
            log.append(Event(EventKind.OBSERVATION, {"text": content}))
        elif role == "assistant" and m.get("tool_calls"):
            if call_format == "agenttune":
                log.append(Event(EventKind.TOOL_CALL, {"action": {"tool_calls": m["tool_calls"]}}))
            else:
                log.append(Event(EventKind.TEXT, {"text": cohere_call_text(m["tool_calls"])}))
        elif role == "assistant":
            log.append(Event(EventKind.TEXT, {"text": content}))
    return log


def _merge_tool_call_runs(messages: list[dict], call_format: str) -> list[dict]:
    """``as_dataset_rows`` writes one assistant message per call; parallel calls were one
    turn at rollout time, so merge consecutive call messages back into one turn."""
    if call_format != "agenttune":
        return messages
    out: list[dict] = []
    for m in messages:
        is_call = m["role"] == "assistant" and str(m.get("content", "")).startswith("<tool_call>")
        if is_call and out and out[-1]["role"] == "assistant" and out[-1]["content"].startswith("<tool_call>"):
            out[-1] = {**out[-1], "content": out[-1]["content"] + m["content"]}
        else:
            out.append(dict(m))
    return out


def make_factory(cfg: dict, tools: list):
    from agenttune.agentic.distill_trainer import (
        _DefaultSFTTrainerFactory,
        _merge_consecutive_same_role,
    )

    d = cfg.get("distill", {})
    lora = cfg.get("lora", {})
    prec = precision(cfg)
    sys_text = fallback_system_prompt(tools, system_prompt(cfg))
    call_format = d.get("tool_call_format", "cohere")

    class QuantizedSFTFactory(_DefaultSFTTrainerFactory):
        """``_DefaultSFTTrainerFactory`` with 4-bit / fp16 loading, the rollout-time system
        prompt and strict-alternation repair applied up front."""

        def __call__(self, *, model, train_dataset, **kwargs):
            from datasets import Dataset
            from peft import LoraConfig
            from trl import SFTConfig, SFTTrainer

            spec = {**cfg.get("student", {}), "model": model}
            self.model, self.tokenizer = load_local_model(spec, prec, for_training=True)
            rows = []
            for row in train_dataset:
                msgs = _merge_tool_call_runs(row["messages"], call_format)
                msgs = _merge_consecutive_same_role([{"role": "system", "content": sys_text}] + msgs)
                rows.append({"messages": msgs})
            self.rows = rows
            cfg_kwargs = dict(kwargs)
            cfg_kwargs.setdefault("output_dir", "./out_distill")
            sft = SFTConfig(**cfg_kwargs)
            peft_config = None
            if self.use_peft and not spec.get("adapter"):
                peft_config = LoraConfig(
                    r=self.lora_r, lora_alpha=self.lora_alpha, lora_dropout=self.lora_dropout,
                    bias=self.lora_bias, task_type=self.task_type,
                    target_modules=self.lora_targets or "all-linear",
                )
            self.sft_trainer = SFTTrainer(
                model=self.model, args=sft, train_dataset=Dataset.from_list(rows),
                processing_class=self.tokenizer, peft_config=peft_config,
            )
            return self.sft_trainer

    return QuantizedSFTFactory(
        use_peft=True,
        lora_r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.05),
        lora_targets=lora.get("target_modules"),
    )


def _sft_kwargs(cfg: dict, out_dir: Path) -> dict[str, Any]:
    prec = precision(cfg)
    t = dict(cfg.get("distill", {}).get("training") or {})
    if cfg.get("dry_run"):
        t.update({"max_steps": cfg["training"]["max_steps"], "use_cpu": True,
                  "per_device_train_batch_size": 2, "gradient_checkpointing": False})
    t.setdefault("report_to", "none")
    t.setdefault("save_strategy", "no")
    t["fp16"] = prec == "fp16"
    t["bf16"] = prec == "bf16"
    t["seed"] = cfg.get("seed", 42)
    t["data_seed"] = cfg.get("seed", 42)
    t["output_dir"] = str(out_dir)
    if t.get("gradient_checkpointing"):
        t.setdefault("gradient_checkpointing_kwargs", {"use_reentrant": False})
    return t


def _data_path(cfg: dict) -> Path:
    d = cfg.get("distill", {})
    if d.get("data_hub_repo"):
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(d["data_hub_repo"], d.get("data_hub_file", "filtered.jsonl"),
                                    repo_type="dataset"))
    p = Path(d.get("data", "outputs/teacher_collect/filtered.jsonl"))
    if cfg.get("dry_run"):
        p = Path(cfg.get("output_root", "outputs/smoke")) / "teacher_collect" / "filtered.jsonl"
    return p if p.is_absolute() else ROOT / p


def run(cfg: dict, args) -> tuple[str, dict]:
    from agenttune.agentic.distill_trainer import create_distill_trainer

    d = cfg.get("distill", {})
    tools = load_tools(cfg)
    path = _data_path(cfg)
    recs = read_jsonl(path) if path.exists() else []
    recs = [r for r in recs if r.get("kept", True) and r.get("conversation")]
    note = None
    if not recs and cfg.get("dry_run"):
        # the tiny random teacher rarely produces a valid call: use the stub references as
        # demonstrations so the dry run still exercises the trainer path
        from ..stub import rows as stub_rows

        recs = [{"conversation": _demo_conversation(r)} for r in stub_rows("train")]
        note = "dry run: no kept teacher trajectories; used stub reference demonstrations"
    if not recs:
        raise RuntimeError(f"no kept teacher trajectories in {path}; run teacher_collect first")
    fmt = d.get("tool_call_format", "cohere")
    logs = [conversation_to_eventlog(r["conversation"], fmt) for r in recs]
    out_dir = work_dir(cfg, "distill")
    lora = cfg.get("lora", {})
    factory = make_factory(cfg, tools)
    trainer = create_distill_trainer(
        cfg["student"]["model"],
        trajectories=logs,
        trainer_factory=factory,
        output_dir=str(out_dir),
        **{k: v for k, v in _sft_kwargs(cfg, out_dir).items() if k != "output_dir"},
    )
    preview = trainer.sft_dataset()[:1]
    result = trainer.train()
    adapter_dir = out_dir / "adapter"
    factory.sft_trainer.save_model(str(adapter_dir))
    factory.tokenizer.save_pretrained(str(adapter_dir))
    hist = factory.sft_trainer.state.log_history
    out = {
        "student": cfg["student"]["model"],
        "data": str(path),
        "n_trajectories": len(logs),
        "n_rows": len(factory.rows),
        "note": note,
        "tool_call_format": fmt,
        "lora": lora,
        "train_result": getattr(result, "metrics", None) or str(result),
        "log_history": hist,
        "first_row_preview": factory.rows[0]["messages"][:3] if factory.rows else preview,
        "adapter_dir": str(adapter_dir),
    }
    if cfg.get("push"):
        out["hub"] = push_folder(cfg, "distill_adapter", adapter_dir,
                                 message=f"distill LoRA on {len(logs)} teacher trajectories")
    return "12_distill", out


def _demo_conversation(row: dict) -> list[dict]:
    refs = json.loads(row["expected_tool_calls"])
    calls = refs[0] if refs else []
    prompt = row["prompt"]
    conv = list(prompt) if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    if calls:
        conv.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{i}", "type": "function",
             "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
            for i, c in enumerate(calls)]})
        conv.append({"role": "tool", "name": calls[0]["name"], "content": '{"ok": true}'})
    conv.append({"role": "assistant", "content": "Done."})
    return conv
