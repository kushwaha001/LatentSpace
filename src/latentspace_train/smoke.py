"""A randomly initialised tiny Llama + locally trained tokenizer for CPU dry runs
(same idea as experiments/04_grpo_smoke.py). It proves wiring, not learning.

Its chat template ignores ``tools`` on purpose, like Tiny Aya's, so the dry run goes through
AgentTune's ``tools_fallback_prompt`` path and folded tool results, exactly as on GPU.
"""

from __future__ import annotations

from pathlib import Path

# Ignores `tools` (as Tiny Aya / Aya Expanse do) -> AgentTune's Cohere fallback prompt.
CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>"
    "{% if m['content'] %}{{ m['content'] }}{% endif %}<|end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def build_tiny_model(path: Path, seed: int = 0) -> Path:
    """Write a ~100k-parameter random Llama and its tokenizer to ``path``; return it."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    path = Path(path)
    if (path / "config.json").exists():
        return path
    torch.manual_seed(seed)
    corpus = [
        '[{"tool_name": "search_trains", "parameters": {"origin": "Mumbai"}}]',
        '[{"tool_name": "get_fare", "parameters": {"train_id": "12951"}}]',
        "Find trains from Mumbai to Delhi. 12951 ka 3A fare kitna hai? haan, book kar do",
    ] * 20
    special = ["<pad>", "<eos>", "<|user|>", "<|assistant|>", "<|system|>", "<|end|>"]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=special))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", bos_token="<eos>"
    )
    # one sampled id puts Cohere call markup in the text, so the random policy sometimes
    # produces (malformed) call attempts and the rewards are not constant
    fast.add_tokens(['[{"tool_name": '])
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(path)
    cfg = LlamaConfig(
        vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096,
        pad_token_id=fast.pad_token_id, eos_token_id=fast.eos_token_id,
        bos_token_id=fast.bos_token_id, tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(cfg)
    model.generation_config.pad_token_id = fast.pad_token_id
    model.save_pretrained(path)
    return path


def apply_dry_run(cfg: dict, work: Path) -> dict:
    """Rewrite a stage config for a CPU dry run: tiny random model, stub tasks, 3 steps,
    fp32, no quantization, results under results/smoke/. Nothing touches the network."""
    from .config import deep_merge

    model_dir = build_tiny_model(work / "tiny_llama", seed=cfg.get("seed", 0))
    dry = cfg.get("dry_run_overrides") or {}
    over = {
        "dry_run": True,
        "precision": "fp32",
        "results_dir": "results/smoke",
        "tasks": {"source": "stub", "limit": None},
        "student": {"model": str(model_dir), "quantization": None, "adapter": None},
        "lora": {"target_modules": ["q_proj", "v_proj"]},
        "training": {"max_steps": 3, "per_device_train_batch_size": 4, "num_generations": 4,
                     "max_completion_length": 24, "max_steps_per_turn": 2,
                     "gradient_checkpointing": False, "logging_steps": 1,
                     "save_steps": 10_000, "learning_rate": 5e-3, "warmup_steps": 0,
                     "gradient_accumulation_steps": 1, "use_cpu": True},
        "generation": {"max_new_tokens": 24},
    }
    out = deep_merge(deep_merge(cfg, over), dry)
    return out
