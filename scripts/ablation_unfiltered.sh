#!/usr/bin/env bash
# Ablation: distillation on ALL teacher trajectories (the audit filter's verdict ignored),
# same settings and epoch count as the selected distilled student, then its test evaluation.
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/ls-venv/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 HF_HOME="${HF_HOME:-/mnt/d/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
L=logs/round2; A=outputs/ablation_unfiltered/distill/adapter
say() { echo "== [$(date -u +%F_%T)] $*"; }
if [ ! -f "$A/adapter_model.safetensors" ]; then
  for a in 1 2 3; do
    LS_GPU_MEM_FRACTION=0.85 bash scripts/gpu_run.sh 0 python -m latentspace_train distill --set precision=bf16 \
      --set student.quantization=null --set distill.data=outputs/teacher_collect/raw.jsonl --set distill.use_all=true \
      --set output_root=outputs/ablation_unfiltered --set results_dir=results/ablation_unfiltered > "$L/12_distill_unfiltered.log" 2>&1 && break
    say "attempt $a failed"; sleep 60
  done
fi
[ -f "$A/adapter_model.safetensors" ] || { say "unfiltered distill failed"; exit 1; }
say "unfiltered distill trained"
bash scripts/gpu_run.sh 0 python -m latentspace_train evaluate --run distill_unfiltered --set precision=bf16 \
  --set student.quantization=null > "$L/14_eval_distill_unfiltered.log" 2>&1 && date -u > outputs/.round2/eval_distill_unfiltered
say "unfiltered ablation done"
