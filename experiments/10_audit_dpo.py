"""Experiment 10: audit-ranked DPO (the audit reward as a ranker instead of a policy gradient).

All three GRPO runs collapsed into silence although reward v2 ranks a call above silence on
every task that needs one: once a prompt's sampled answers are all silent, GRPO has no signal
left. DPO learns from explicit (better, worse) pairs instead, so "silent on a task that needs a
call" can be named as the worse answer directly.

Pairs (train split only, no new sampling): every train-split rollout we already have (teacher
collection and the training rollouts of run_A, run_B, run_B2) is matched to its train task and
scored with the evaluation's audit verdict (semantic matching, ``lenient_success``). For each
task, a successful rollout's first assistant turn is "chosen" and a failed rollout's first
assistant turn is "rejected" when they differ. Hard negatives first: silence on a task that
needs a call, an unconfirmed booking, a call on a task that needs none. At most
``--pairs-per-task`` pairs per task.

Training: AgentTune's ``create_agentic_trainer("dpo")`` (offline mode). The distilled adapter is
merged into the weights first and a new LoRA is trained on top, so DPO's reference model
("adapter disabled") is the distilled student.

    python experiments/10_audit_dpo.py --build-only          # pairs + stats, CPU
    python experiments/10_audit_dpo.py                       # build + train (GPU)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "outputs" / "dpo" / "audit"  # --name changes the last part
SOURCES = {
    "teacher": "outputs/teacher_collect/raw.jsonl",
    "run_A": "outputs/grpo/run_A/trajectories.jsonl",
    "run_B": "outputs/grpo/run_B/trajectories.jsonl",
    "run_B2": "outputs/grpo/run_B2/trajectories.jsonl",
}


def _content(m) -> str:
    return str(m.get("content") or "")


def first_turn_text(msg: dict) -> str:
    """The assistant turn as the student writes it: the Cohere call list, or the text."""
    from latentspace_train.stages.distill import cohere_call_text

    if msg.get("tool_calls"):
        return cohere_call_text(msg["tool_calls"])
    return _content(msg).strip()


def collect(rows: list[dict]) -> list[dict]:
    """Train-split rollouts with their task, as records for ``score_records``."""
    by_first = defaultdict(list)
    for t in rows:
        by_first[_content(t["messages"][0])].append(t)
    out = []
    for src, path in SOURCES.items():
        p = ROOT / path
        if not p.exists():
            continue
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            conv = r.get("conversation") if src == "teacher" else (r.get("metadata") or {}).get("conversation")
            if not conv:
                continue
            body = [m for m in conv if m.get("role") != "system"]
            if not body:
                continue
            for t in by_first.get(_content(body[0]), []):
                k = len(t["messages"])
                if [_content(m) for m in body[:k]] == [_content(m) for m in t["messages"]] \
                        and len(body) > k and body[k].get("role") == "assistant":
                    out.append({"source": src, "task_id": t["id"], "language": t["language"],
                                "category": t["category"], "prompt": t["messages"],
                                "response": (r.get("response") if src == "teacher" else r.get("final_response")) or "",
                                "conversation": conv, "expected_tool_calls": json.dumps(t["expected_tool_calls"]),
                                "first_turn": first_turn_text(body[k]),
                                "first_has_call": bool(body[k].get("tool_calls"))})
                    break
    return out


def hardness(rec: dict, s: dict) -> int:
    """Lower = more useful negative."""
    needs = bool(json.loads(rec["expected_tool_calls"]))
    if needs and not rec["first_has_call"]:
        return 0  # silent although a call is needed (the GRPO collapse)
    if s.get("unconfirmed_irreversible"):
        return 0  # booked / cancelled without a yes
    if not needs and rec["first_has_call"]:
        return 1  # acted although no call was correct
    return 2  # wrong call, wrong arguments, or a later turn failed


def build_pairs(per_task: int, seed: int, balance: bool = False) -> tuple[list[dict], dict]:
    import transit_agent
    from latentspace_train.config import load_config
    from latentspace_train.scoring import audit_weights, lenient_scorer, make_audit_rewards, score_records
    from latentspace_train.tasks import load_split

    cfg = load_config("configs/13_grpo.yaml")
    rows = [json.loads(l) for l in open(ROOT / "data/eval/train.jsonl", encoding="utf-8")]
    recs = collect(rows)
    scored = score_records(recs, make_audit_rewards(transit_agent.TOOLS, cfg), audit_weights(cfg),
                           lenient=lenient_scorer(cfg))
    by_task = defaultdict(lambda: {"pos": [], "neg": []})
    for rec, s in zip(recs, scored):
        ok = s.get("lenient_success") == 1.0
        by_task[rec["task_id"]]["pos" if ok else "neg"].append((rec, s))
    rng = random.Random(seed)
    pairs, stats = [], Counter()
    task_of = {t["id"]: t for t in rows}
    for tid, d in sorted(by_task.items()):
        pos = {r["first_turn"]: r for r, _ in d["pos"]}  # distinct chosen texts
        negs = sorted({r["first_turn"]: (r, s) for r, s in d["neg"]}.values(), key=lambda x: hardness(*x))
        negs = [(r, s) for r, s in negs if r["first_turn"] not in pos]
        if not pos or not negs:
            stats["tasks_without_pair"] += 1
            continue
        chosen = list(pos.values())
        rng.shuffle(chosen)
        for i, (r, s) in enumerate(negs[:per_task]):
            c = chosen[i % len(chosen)]
            pairs.append({"task_id": tid, "category": task_of[tid]["category"], "language": task_of[tid]["language"],
                          "prompt": task_of[tid]["messages"],
                          "chosen": c["first_turn"], "rejected": r["first_turn"],
                          "chosen_source": c["source"], "rejected_source": r["source"],
                          "rejected_kind": ["silent_or_unconfirmed", "acted_needlessly", "wrong"][hardness(r, s)]})
            stats["pairs"] += 1
    if balance:
        # v1 (unbalanced: 327 of 432 negatives were "silent") learned to call tools on
        # everything. Keep as many "silent" negatives as "acted needlessly" ones, plus all
        # wrong-call negatives, so neither direction dominates.
        kinds = defaultdict(list)
        for p in pairs:
            kinds[p["rejected_kind"]].append(p)
        n = len(kinds["acted_needlessly"])
        silent = kinds["silent_or_unconfirmed"]
        rng.shuffle(silent)
        pairs = kinds["acted_needlessly"] + kinds["wrong"] + silent[:n]
        stats["pairs"] = len(pairs)
        stats["balanced"] = 1
    stats.update({"rollouts_matched": len(recs), "tasks_with_pairs": len({p["task_id"] for p in pairs})})
    info = {"stats": dict(stats),
            "by_category": dict(Counter(p["category"] for p in pairs)),
            "by_rejected_kind": dict(Counter(p["rejected_kind"] for p in pairs)),
            "by_source": {"chosen": dict(Counter(p["chosen_source"] for p in pairs)),
                          "rejected": dict(Counter(p["rejected_source"] for p in pairs))},
            "rollouts_by_source": dict(Counter(r["source"] for r in recs))}
    return pairs, info


def train(pairs: list[dict], args) -> dict:
    import torch
    from datasets import Dataset

    import transit_agent
    from agenttune.core.backend_factory import create_agentic_trainer
    from latentspace_train.engines import fallback_system_prompt, load_local_model

    system = fallback_system_prompt(transit_agent.TOOLS, transit_agent.SYSTEM_PROMPT)
    OUT.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list([{"prompt": [{"role": "system", "content": system}] + p["prompt"],
                             "chosen": [{"role": "assistant", "content": p["chosen"]}],
                             "rejected": [{"role": "assistant", "content": p["rejected"]}]} for p in pairs])
    # an in-memory dataset caches .map() results in a temporary directory that can disappear
    # before TRL reads them back (FileNotFoundError on precompute_ref_log_probs): use a disk copy
    from datasets import load_from_disk

    ds.save_to_disk(str(OUT / "dataset"))
    ds = load_from_disk(str(OUT / "dataset"))
    spec = {"model": "CohereLabs/tiny-aya-global", "quantization": None, "attn_implementation": "sdpa",
            "base_adapter": "outputs/distill/adapter", "adapter": None}
    model, tok = load_local_model(spec, "bf16", for_training=True)
    peft = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM",
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
    kw = dict(output_dir=str(OUT), num_train_epochs=args.epochs, per_device_train_batch_size=1,
              gradient_accumulation_steps=8, learning_rate=args.lr, warmup_steps=5, beta=args.beta,
              max_length=6144, truncation_mode="keep_start", logging_steps=5,
              # the weights are already bf16. With mixed-precision autocast, accelerate copies
              # the logits to fp32 (6.6 GB at 2 x 3.7k tokens x 261k vocab) and runs out of memory
              bf16=False, fp16=False,
              # reference log-probs computed once up front (no grad), instead of a second
              # full-vocabulary forward pass inside every training step
              precompute_ref_log_probs=True,
              save_strategy="no", report_to="none", seed=args.seed, gradient_checkpointing=True,
              gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.max_steps:
        kw["max_steps"] = args.max_steps
    trainer = create_agentic_trainer("dpo", backend="trl", model=model, processing_class=tok,
                                     train_dataset=ds, peft_config=peft, **kw)
    t0 = time.time()
    trainer.train()
    hf = trainer.trainer
    hf.save_model(str(OUT / "adapter"))
    return {"seconds": round(time.time() - t0, 1), "log_history": hf.state.log_history,
            "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
            "config": {k: v for k, v in kw.items() if k != "gradient_checkpointing_kwargs"},
            "start": "base + merged distill adapter (reference = distilled student)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--pairs-per-task", type=int, default=4)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balance", action="store_true", help="as many 'silent' as 'acted needlessly' negatives")
    ap.add_argument("--name", default="audit", help="outputs/dpo/<name>, results/10_audit_dpo[_<name>].json")
    args = ap.parse_args()
    global OUT
    OUT = ROOT / "outputs" / "dpo" / args.name
    pairs, info = build_pairs(args.pairs_per_task, args.seed, args.balance)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "pairs.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps(info, indent=1))
    res = {"experiment": "10_audit_dpo", "pairs": info, "args": vars(args)}
    if not args.build_only:
        res["train"] = train(pairs, args)
        print({k: v for k, v in res["train"].items() if k != "log_history"})
    suffix = "" if args.name == "audit" else f"_{args.name}"
    (ROOT / f"results/10_audit_dpo{suffix}.json").write_text(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
