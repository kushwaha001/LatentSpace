#!/usr/bin/env bash
# Dev-split check (64 tasks x 2 trials) of one GRPO checkpoint, run next to the training job.
#   bash scripts/dev_check.sh <run> <step> [gpu] [base_adapter]   e.g. run_B3 60 0 outputs/distill/adapter
# Waits for outputs/grpo/<run>/checkpoint-<step> and for ~9 GB of free memory on the GPU, then
# prints the comparison with the selected distilled model (scripts/dev_compare.py).
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/ls-venv/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 HF_HOME="${HF_HOME:-/mnt/d/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
R="$1"; S="$2"; G="${3:-0}"; BASE="${4:-}"; N="${R}_ckpt$S"; CK="outputs/grpo/$R/checkpoint-$S"
until [ -f "$CK/adapter_model.safetensors" ]; do sleep 30; done
sleep 20
if [ ! -f "results/_devsel/$N/10_baseline.json" ]; then
  # a dev check decides what runs next, so it goes before a benchmark: stop a running benchmark
  # on this GPU (it resumes where it stopped, scripts/round2_final.sh retries it after this)
  pkill -TERM -f "python experiments/08_public_benchmarks.py" 2>/dev/null; sleep 20
  bash scripts/gpu_run.sh "$G" python -m latentspace_train baseline --set precision=bf16 --set student.quantization=null \
    --set "student.adapter=$CK" ${BASE:+--set "student.base_adapter=$BASE"} --set baseline.trials=2 \
    --set "output_root=outputs/_devsel/$N" --set "results_dir=results/_devsel/$N" > "logs/round2/12_devcheck_$N.log" 2>&1
fi
python scripts/dev_compare.py "$N"
