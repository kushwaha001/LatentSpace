#!/usr/bin/env bash
# One resumable GRPO run (+ its test evaluation) on one GPU, for runs beyond round2.sh:
#
#   bash scripts/grpo_extra.sh <run> <gpu> <per-device batch> [outroot-suffix]
#   e.g. bash scripts/grpo_extra.sh run_B2 0 2
#
# Resumes from the newest checkpoint; after an out-of-memory stop it continues with 1 completion
# per backward pass (same optimizer step: 1 prompt x 4 generations). Keeps a checkpoint every
# 20 steps (15 kept = the whole run), so any step can be evaluated on dev afterwards.
# Waits until the GPU has MIN_FREE_MB free before starting.
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/ls-venv/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 HF_HOME="${HF_HOME:-/mnt/d/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512,garbage_collection_threshold:0.8}"
R="$1"; G="$2"; BS="${3:-2}"; STEPS="${GRPO_STEPS:-300}"; MIN_FREE_MB="${MIN_FREE_MB:-16500}"
L=logs/round2; LOCAL=(--set precision=bf16 --set student.quantization=null)
say() { echo "== [$(date -u +%F_%T)] $*"; }
finished() { python -c "import json,sys; d=json.load(open('results/13_grpo_$R.json')); sys.exit(0 if d['error'] is None and (d['global_step'] or 0)>=$STEPS else 1)" 2>/dev/null; }
until [ "$(nvidia-smi -i "$G" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')" -ge "$MIN_FREE_MB" ]; do sleep 30; done
say "$R: GRPO on GPU $G (batch $BS)"
for a in $(seq 1 40); do
  finished && break
  CUDA_VISIBLE_DEVICES="$G" python -m latentspace_train grpo --run "$R" "${LOCAL[@]}" \
    --set grpo.resume_from_checkpoint=auto --set training.max_steps="$STEPS" \
    --set training.save_steps=10 --set training.save_total_limit=30 --set training.torch_empty_cache_steps=1 \
    --set training.per_device_train_batch_size="$BS" --set training.gradient_accumulation_steps=$((4 / BS)) \
    >> "$L/13_grpo_$R.log" 2>&1 || true
  finished || { say "$R attempt $a stopped early; resuming with batch 1"; BS=1; sleep 30; }
done
finished || { say "$R failed"; exit 1; }
say "$R trained: test evaluation"
CUDA_VISIBLE_DEVICES="$G" python -m latentspace_train evaluate --run "$R" "${LOCAL[@]}" > "$L/14_eval_$R.log" 2>&1 \
  && date -u > "outputs/.round2/eval_$R"
say "$R done"
