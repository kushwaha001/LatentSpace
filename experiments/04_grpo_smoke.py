"""Experiment 04: 20-step CPU GRPO smoke run of create_agentic_trainer with the audit rewards.

The plan was a ~135M pretrained model (e.g. SmolLM2-135M-Instruct). That is not possible here:
this sandbox's network policy blocks the Hugging Face Hub (proxy 403), and no model weights are
cached. So this run uses a **randomly initialised** tiny Llama (about 90k parameters) and a
byte-level BPE tokenizer trained locally on a few tool-call strings. It is a plumbing test, not
a learning run: it checks that AgentTune's real trainer path

    create_agentic_trainer("grpo", tools=..., reward_funcs=AuditRewards(...).train_time())
      -> TrlAgenticGrpo.setup_trainer -> combine_rewards -> TRL GRPOTrainer
      -> patched _generate_single_turn -> AgentTune rollout (HF generate on CPU, tool parsing)
      -> reward_funcs(prompts=..., completions=..., conversations=..., expected_tool_calls=...)
      -> advantages -> loss -> optimizer step

runs end to end on CPU, that every adapter reward is called with the rollout's extras, and
that the policy weights change. ``<tool_call>`` is a single token in the vocabulary, so a random
policy sometimes emits tool-call markup that does not parse. The validity reward scores that 0.0
against 0.5 for no call, which gives GRPO a non-constant reward to act on.

Run from the repo root:  python experiments/04_grpo_smoke.py
Writes results/04_grpo_smoke.json.
"""

from __future__ import annotations

import json
import os
import platform
import random
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
WORK = Path(tempfile.mkdtemp(prefix="grpo_smoke_"))
os.environ["AGENTTUNE_SCORE_LOG"] = str(WORK / "scores.jsonl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

import logging  # noqa: E402

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from audit_rewards import AuditRewards, irreversible  # noqa: E402

SEED = 0
random.seed(SEED)
torch.manual_seed(SEED)
STEPS = 20


def get_weather(location: str) -> dict:
    """Get the current weather for a city.

    Args:
        location: City name.
    """
    return {"location": location, "temp": 14}


@irreversible
def book_flight(flight_id: str, passenger: str) -> dict:
    """Book a flight. Irreversible.

    Args:
        flight_id: Flight id.
        passenger: Passenger name.
    """
    return {"status": "booked"}


TOOLS = [get_weather, book_flight]

CHAT_TEMPLATE = (
    "{% if tools %}<tools>{% for t in tools %}{{ t | tojson }}{% endfor %}</tools>\n{% endif %}"
    "{% for m in messages %}<|{{ m['role'] }}|>"
    "{% if m['content'] %}{{ m['content'] }}{% endif %}"
    "{% if m.get('tool_calls') %}{% for c in m['tool_calls'] %}<tool_call>"
    "{{ c['function'] | tojson }}</tool_call>{% endfor %}{% endif %}<|end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def build_tokenizer(path: Path) -> PreTrainedTokenizerFast:
    corpus = [
        '<tool_call>{"name": "get_weather", "arguments": {"location": "Paris"}}</tool_call>',
        '<tool_call>{"name": "book_flight", "arguments": {"flight_id": "UA123"}}</tool_call>',
        "What is the weather in Paris? It is 14 C. Book the flight for Ana.",
    ] * 20
    special = ["<pad>", "<eos>", "<|user|>", "<|assistant|>", "<|tool|>", "<|system|>", "<|end|>"]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=special))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", bos_token="<eos>"
    )
    # Ordinary (non-special) added tokens: they survive skip_special_tokens decoding, so one
    # sampled id puts tool-call markup in the text the rollout parses.
    fast.add_tokens(["<tool_call>", "</tool_call>"])
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(path)
    return fast


def build_model(path: Path, tok) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(cfg)
    model.generation_config.pad_token_id = tok.pad_token_id
    model.save_pretrained(path)
    return model


class Recorder:
    """Wraps a reward to record how the trainer called it (kwargs it got, scores it gave)."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []
        self.__name__ = fn.__name__

    def __call__(self, prompts=None, completions=None, **kwargs):
        scores = self.fn(prompts=prompts, completions=completions, **kwargs)
        self.calls.append({"kwargs": sorted(kwargs), "n": len(completions), "scores": scores})
        return scores


def main() -> dict:
    tok_dir, model_dir, out_dir = WORK / "tok", WORK / "model", WORK / "run"
    tok = build_tokenizer(tok_dir)
    model = build_model(model_dir, tok)
    n_params = sum(p.numel() for p in model.parameters())
    before = {k: v.clone() for k, v in model.state_dict().items()}

    ref = json.dumps([[{"name": "get_weather", "arguments": {"location": "Paris"}}]])
    ds = Dataset.from_list(
        [{"prompt": "What is the weather in Paris?", "expected_tool_calls": ref}] * 8
    )

    adapter = AuditRewards(tools=TOOLS)
    recorders = [Recorder(f) for f in adapter.train_time()]

    from agenttune.core.backend_factory import create_agentic_trainer

    trainer = create_agentic_trainer(
        "grpo",
        backend="trl",
        model=str(model_dir),
        processing_class=tok,
        tools=TOOLS,
        reward_funcs=recorders,
        reward_weights=[1.0, 1.0, 1.0, 1.0, 0.25],
        train_dataset=ds,
        output_dir=str(out_dir),
        max_steps=STEPS,
        per_device_train_batch_size=4,
        num_generations=4,
        gradient_accumulation_steps=1,
        max_completion_length=24,
        max_steps_per_turn=2,
        learning_rate=5e-3,
        warmup_steps=0,
        beta=0.0,
        temperature=1.0,
        logging_steps=1,
        save_steps=10_000,
        report_to="none",
        use_cpu=True,
        bf16=False,
        seed=SEED,
    )
    t0 = time.time()
    error = None
    try:
        trainer.train()
    except Exception as e:  # report, never hide
        error = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0

    hf = trainer.trainer
    history = hf.state.log_history if hf is not None else []
    after = hf.model.state_dict() if hf is not None else {}
    changed = sum(not torch.equal(before[k], after[k].cpu()) for k in before if k in after)
    traj_file = out_dir / "trajectories.jsonl"
    n_traj = sum(1 for _ in open(traj_file)) if traj_file.exists() else 0

    calls = recorders[0].calls
    all_scores = [s for c in calls for s in c["scores"]]
    res = {
        "model": {
            "kind": "randomly initialised LlamaForCausalLM (Hub blocked; see docstring)",
            "parameters": n_params,
            "vocab_size": len(tok),
        },
        "steps_requested": STEPS,
        "global_step": hf.state.global_step if hf is not None else None,
        "seconds": round(elapsed, 1),
        "error": error,
        "device": str(next(hf.model.parameters()).device) if hf is not None else None,
        "tensors_changed": f"{changed}/{len(before)}",
        "trajectories_written": n_traj,
        "reward_calls": {
            r.__name__: {
                "n_calls": len(r.calls),
                "kwargs_seen": r.calls[0]["kwargs"] if r.calls else None,
                "score_values": sorted({round(s, 4) for c in r.calls for s in c["scores"] if s is not None}),
            }
            for r in recorders
        },
        "validity_score_counts": {str(v): all_scores.count(v) for v in sorted(set(all_scores))},
        "log_history": [
            {k: h[k] for k in ("step", "loss", "reward", "reward_std", "grad_norm") if k in h}
            for h in history
        ],
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "threads": torch.get_num_threads(),
        },
    }
    try:
        from importlib.metadata import version

        res["env"].update({p: version(p) for p in ("trl", "transformers", "agenttune", "auditkit")})
    except Exception:
        pass
    shutil.rmtree(WORK, ignore_errors=True)
    return res


if __name__ == "__main__":
    out = main()
    (RESULTS / "04_grpo_smoke.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "log_history"}, indent=2, default=str))
    for h in out["log_history"]:
        print(h)
    print("wrote results/04_grpo_smoke.json")
