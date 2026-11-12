#!/usr/bin/env bash
# The tail of round 2 on GPU 1, started automatically when run_B finishes training:
# test evaluation of run_B and the public benchmarks of every model that still lacks them
# (2-3 jobs side by side, memory-aware), then the comparison tables.
# GPU 0 is left to run_B2 (scripts/grpo_extra.sh). Rerunnable: finished jobs are skipped.
#
#   nohup bash scripts/round2_final.sh >> logs/round2/round2_final.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/ls-venv/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 HF_HOME="${HF_HOME:-/mnt/d/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512,garbage_collection_threshold:0.8}"
GPU="${GPU:-1}"; M=outputs/.round2; L=logs/round2
LOCAL=(--set precision=bf16 --set student.quantization=null)
MIN_FREE_MB="${MIN_FREE_MB:-11000}"; MAX_JOBS="${MAX_JOBS:-1}"; GAP="${GAP:-150}"; LAST=0
free_mb() { nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9'; echo; }
running() { jobs -rp | wc -l; }
say() { echo "== [$(date -u +%F_%T)] $*"; }
launch() { # launch <name> <done-test> <function> <args...>
  local name="$1" test="$2"; shift 2
  if eval "$test"; then say "skip $name (done)"; return 0; fi
  while [ "$(running)" -ge "$MAX_JOBS" ] || [ "$(free_mb)" -lt "$MIN_FREE_MB" ] 2>/dev/null \
        || [ $(( $(date +%s) - LAST )) -lt "$GAP" ]; do sleep 15; done
  say "start $name (GPU $GPU free $(free_mb) MiB, $(running) running)"
  LAST=$(date +%s)
  ( for a in 1 2 3 4 5 6; do "$@" && { say "done $name"; exit 0; }; say "$name attempt $a failed"; sleep 60; done; exit 1 ) &
}
do_eval() { bash scripts/gpu_run.sh "$GPU" python -m latentspace_train evaluate --run "$1" "${LOCAL[@]}" \
              > "$L/14_eval_$1.log" 2>&1 && date -u > "$M/eval_$1"; }
do_bench() { bash scripts/gpu_run.sh "$GPU" python experiments/08_public_benchmarks.py --name "$1" --adapter "$2" \
               > "$L/08_bench_$1.log" 2>&1; }
until [ -f "$M/grpo_run_B" ]; do sleep 60; done
say "run_B finished training"; date -u > "$M/grpo"
launch eval_run_B "[ -f $M/eval_run_B ]" do_eval run_B
launch bench_run_B "[ -f results/08_public_benchmarks_run_B.json ]" do_bench run_B outputs/grpo/run_B/adapter
launch bench_distill "[ -f results/08_public_benchmarks_distill.json ]" do_bench distill outputs/distill/adapter
launch bench_run_A "[ -f results/08_public_benchmarks_run_A.json ]" do_bench run_A outputs/grpo/run_A/adapter
wait
until [ -f "$M/eval_distill" ] && [ -f "$M/eval_base" ] && [ -f "$M/eval_run_A" ] && [ -f "$M/eval_run_B" ]; do sleep 60; done
say "all four evaluations present: comparisons"
CUDA_VISIBLE_DEVICES="" python -m latentspace_train evaluate --compare-only > "$L/14_compare.log" 2>&1
python experiments/08_public_benchmarks.py --report > "$L/08_report.log" 2>&1
date -u > "$M/evaluate"; date -u > "$M/bench"
say "ROUND 2 COMPLETE"
