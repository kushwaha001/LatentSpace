"""Publish the submission artefacts to the Hugging Face Hub as PUBLIC repos + two collections.

    python scripts/publish_hf.py --dry-run          # print the plan, upload nothing
    read -rs HF_TOKEN && export HF_TOKEN            # never put the token in a file
    python scripts/publish_hf.py                    # upload (needs a write token)

Models: each LoRA adapter (adapter_config.json + adapter_model.safetensors + a model card).
Tokenizer files are not uploaded: load the tokenizer from the base model (the GRPO adapters were
saved with a training-time eos change, see FINDINGS.md B15). Adapters marked "on top of distill"
were trained on base + the merged distill adapter.
Datasets: the eval set, the teacher trajectories, and every rollout / scored rollout / training
trajectory / DPO pair set / benchmark row / results file used in the write-up.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NS = "kanishk001"
BASE = "CohereLabs/tiny-aya-global"
GH = "https://github.com/kushwaha001/LatentSpace"

MODELS = [  # (repo suffix, adapter dir, on top of distill?, description)
    ("distill", "outputs/distill/adapter", False, "LoRA SFT on audit-filtered Aya Expanse 32B trajectories (epoch selected on dev). Test success 0.513."),
    ("distill-unfiltered", "outputs/ablation_unfiltered/distill/adapter", False, "Ablation: LoRA SFT on all teacher trajectories (no audit filter). 19.4% unconfirmed irreversible actions vs 13.2% filtered."),
    ("grpo-default-reward", "outputs/grpo/run_A/adapter", False, "GRPO with AgentTune's default hybrid_prm_reward from the distilled student. Reward-hacked: 0 tool calls per task."),
    ("grpo-audit-v1", "outputs/grpo/run_B/adapter", False, "GRPO with the audit reward v1 (no KL anchor). Collapsed towards silence (0.09 calls per task)."),
    ("grpo-audit-v2", "outputs/grpo/run_B2/adapter", False, "GRPO with the audit reward v2 (no KL anchor). Collapsed into silence despite a correct reward ranking."),
    ("grpo-kl-default-reward", "outputs/grpo/run_A3/adapter", True, "GRPO + KL anchor to the distilled student, default reward. Still collapses (needs-a-call success 0.023)."),
    ("grpo-kl-audit-v2", "outputs/grpo/run_B3/adapter", True, "GRPO + KL anchor to the distilled student, audit reward v2. Keeps competence (needs-a-call success 0.423, +0.40 vs default reward, p=0.0001)."),
    ("dpo-audit-unbalanced", "outputs/dpo/audit/adapter", True, "Offline DPO on audit-ranked pairs (327/432 'silent' negatives). Over-calls tools (dev success 0.26)."),
    ("dpo-audit-balanced", "outputs/dpo/bal/adapter", True, "Offline DPO on balanced audit-ranked pairs. Test success 0.500."),
]
DATASETS = {
    "latentspace-transit-eval": {
        "desc": "360-row, 4-language (en/hi/mr/Hinglish) tool-calling eval set for Indian public transit, 8 categories, train/dev/test split by base task, plus the mock backend data.",
        "files": {"data/eval": "eval", "data/backend": "backend"},
    },
    "latentspace-teacher-trajectories": {
        "desc": "Aya Expanse 32B (4-bit) trajectories on the train split (reference-guided for tasks with tool calls), with audit scores and the filter verdict; filtered.jsonl is the distillation set.",
        "files": {"outputs/teacher_collect/raw.jsonl": "raw.jsonl", "outputs/teacher_collect/filtered.jsonl": "filtered.jsonl",
                  "results/11_teacher_collect.json": "11_teacher_collect.json"},
    },
    "latentspace-rollouts-and-logs": {
        "desc": "Everything behind the write-up: held-out test rollouts and audit-scored rollouts for every checkpoint, GRPO training trajectories, DPO pairs, dev checks, public-benchmark rows (When2Call, BFCL-Hi), all results JSON files, and the raw training / evaluation logs.",
        "files": {"outputs/evaluate": "evaluate", "outputs/benchmarks": "benchmarks", "outputs/dpo/audit/pairs.jsonl": "dpo/audit_pairs.jsonl",
                  "outputs/dpo/bal/pairs.jsonl": "dpo/balanced_pairs.jsonl", "results": "results", "logs": "logs",
                  **{f"outputs/grpo/{r}/trajectories.jsonl": f"grpo_trajectories/{r}.jsonl" for r in ("run_A", "run_B", "run_B2", "run_A3", "run_B3")}},
    },
}


def model_card(suffix, desc, on_distill):
    load = (f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("{BASE}")
model = AutoModelForCausalLM.from_pretrained("{BASE}", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(model, "{NS}/latentspace-tiny-aya-transit-distill").merge_and_unload()
model = PeftModel.from_pretrained(model, "{NS}/latentspace-tiny-aya-transit-{suffix}")
```""" if on_distill else f"""```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("{BASE}")
model = AutoModelForCausalLM.from_pretrained("{BASE}", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(model, "{NS}/latentspace-tiny-aya-transit-{suffix}")
```""")
    return f"""---
base_model: {BASE}
library_name: peft
license: cc-by-nc-4.0
language: [en, hi, mr]
tags: [lora, tool-calling, agent, grpo, reward-hacking, lexsi-hackathon]
---
# LatentSpace · Tiny Aya transit agent · {suffix}

{desc}

Part of LatentSpace (Lexsi AI Safety & Alignment Hackathon, Track 3, Problem 9: reward design and reward-hacking defences).
Write-up and code: {GH} (`docs/WRITEUP.md`). Use the system prompt `transit_agent.SYSTEM_PROMPT` plus AgentTune's tool fallback prompt (see the repo).
{"This adapter was trained on top of the base model **merged with the distilled adapter**." if on_distill else ""}

{load}

Stop generation at `<|END_RESPONSE|>` as well as the model's own eos (FINDINGS.md B15). Tools and data are a synthetic Indian-transit world; do not use for real bookings.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    plan = [(f"{NS}/latentspace-tiny-aya-transit-{s}", "model", d, on, desc) for s, d, on, desc in MODELS]
    for name, info in DATASETS.items():
        plan.append((f"{NS}/{name}", "dataset", info["files"], None, info["desc"]))
    for repo, kind, src, _, desc in plan:
        miss = [p for p in ([src] if isinstance(src, str) else list(src)) if not (ROOT / p).exists()]
        print(f"{kind:7s} {repo}{'   MISSING: ' + str(miss) if miss else ''}")
    if a.dry_run:
        return
    if not os.environ.get("HF_TOKEN"):
        sys.exit("export HF_TOKEN first (read -rs HF_TOKEN && export HF_TOKEN)")
    from huggingface_hub import HfApi
    api = HfApi()
    for repo, kind, src, on, desc in plan:
        api.create_repo(repo, repo_type=kind, private=False, exist_ok=True)
        api.update_repo_settings(repo, repo_type=kind, private=False)
        if kind == "model":
            d = ROOT / src
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                api.upload_file(path_or_fileobj=str(d / f), path_in_repo=f, repo_id=repo, repo_type=kind)
            card = model_card(repo.split("transit-")[1], desc, on)
        else:
            for local, remote in src.items():
                p = ROOT / local
                if p.is_dir():
                    api.upload_folder(folder_path=str(p), path_in_repo=remote, repo_id=repo, repo_type=kind,
                                      ignore_patterns=["*.lock", "__pycache__/*"])
                else:
                    api.upload_file(path_or_fileobj=str(p), path_in_repo=remote, repo_id=repo, repo_type=kind)
            card = f"---\nlicense: cc-by-nc-4.0\nlanguage: [en, hi, mr]\ntags: [tool-calling, agent, reward-hacking, lexsi-hackathon]\n---\n# {repo.split('/')[1]}\n\n{desc}\n\nWrite-up and code: {GH} (`docs/WRITEUP.md`). Station names and codes are real; timetables, fares and PNRs are synthetic. The Hindi, Marathi and Hinglish text is model-generated (native-speaker review in progress).\n"
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, repo_type=kind)
        print("published", repo)
    out = {}
    for title, kind in (("LatentSpace models (Lexsi hackathon, Track 3 P9)", "model"),
                        ("LatentSpace datasets and logs (Lexsi hackathon, T3 P9)", "dataset")):
        col = api.create_collection(title=title, namespace=NS, private=False, exists_ok=True,
                                    description="Reward hacking in a multilingual tool-calling agent: " + GH)
        for repo, k, *_ in plan:
            if k == kind:
                api.add_collection_item(col.slug, item_id=repo, item_type=kind, exists_ok=True)
        out[kind] = f"https://huggingface.co/collections/{col.slug}"
        print(kind, "collection:", out[kind])
    (ROOT / "docs/HF_LINKS.txt").write_text("\n".join(f"{k}: {v}" for k, v in out.items()) + "\n")


if __name__ == "__main__":
    main()
