#!/usr/bin/env bash
# Use GPU 0 as soon as run_A finishes, while run_B (GPU 1) is still training:
# test evaluation of base, distill and run_A, then public benchmarks for distill and run_A.
# Each job claims its marker first, so scripts/round2.sh skips it; outputs and results are the
# same files round2.sh would write. The final comparisons (evaluate --compare-only and
# 08 --report) are rerun after everything has finished.
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/ls-venv/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 HF_HOME="${HF_HOME:-/mnt/d/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512,garbage_collection_threshold:0.8}"
M=outputs/.round2; L=logs/round2
LOCAL=(--set precision=bf16 --set student.quantization=null)
until [ -f "$M/grpo_run_A" ]; do sleep 60; done
echo "== [$(date -u +%F_%T)] GPU 0 free: early evaluation"
for c in base distill run_A; do
  [ -f "$M/eval_$c" ] && continue
  echo claimed > "$M/eval_$c"
  if CUDA_VISIBLE_DEVICES=0 python -m latentspace_train evaluate --run "$c" "${LOCAL[@]}" > "$L/14_eval_$c.log" 2>&1; then
    date -u > "$M/eval_$c"
  else
    rm -f "$M/eval_$c"; echo "eval $c failed"
  fi
done
for spec in "distill outputs/distill/adapter" "run_A outputs/grpo/run_A/adapter"; do
  set -- $spec
  [ -f "results/08_public_benchmarks_$1.json" ] && continue
  CUDA_VISIBLE_DEVICES=0 python experiments/08_public_benchmarks.py --name "$1" --adapter "$2" > "$L/08_bench_$1.log" 2>&1 \
    || echo "bench $1 failed"
done
echo "== [$(date -u +%F_%T)] GPU 0 early jobs done"
