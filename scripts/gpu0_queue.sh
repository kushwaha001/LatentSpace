#!/usr/bin/env bash
# Keep GPU 0 busy while run_B trains on GPU 1.
#
# One generation process uses ~1 CPU core and only 30-40% of a GPU (batch-1 decoding is bound by
# kernel launches, not compute), and this machine has 48 cores. So as soon as run_A has finished,
# this queue runs the held-out evaluations and the public benchmarks of base, distill and run_A
# side by side on GPU 0: a new job starts only when the GPU has MIN_FREE_MB free, at most MAX_JOBS
# at a time, GAP seconds apart (memory settles), and a failed job is retried twice.
# Results go where scripts/round2.sh expects them, so it skips what is already done.
#
#   nohup bash scripts/gpu0_queue.sh > logs/round2/gpu0_queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/ls-venv/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 HF_HOME="${HF_HOME:-/mnt/d/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512,garbage_collection_threshold:0.8}"
M=outputs/.round2; L=logs/round2
LOCAL=(--set precision=bf16 --set student.quantization=null)
MIN_FREE_MB="${MIN_FREE_MB:-11000}"; MAX_JOBS="${MAX_JOBS:-3}"; GAP="${GAP:-150}"; LAST=0

free_mb() { nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
running() { jobs -rp | wc -l; }
say() { echo "== [$(date -u +%F_%T)] $*"; }

launch() { # launch <name> <done-test> <function> <args...>
  local name="$1" test="$2"; shift 2
  if eval "$test"; then say "skip $name (done)"; return 0; fi
  while [ "$(running)" -ge "$MAX_JOBS" ] || [ "$(free_mb)" -lt "$MIN_FREE_MB" ] \
        || [ $(( $(date +%s) - LAST )) -lt "$GAP" ]; do sleep 15; done
  say "start $name (GPU 0 free $(free_mb) MiB, $(running) running)"
  LAST=$(date +%s)
  ( for a in 1 2 3; do "$@" && { say "done $name"; exit 0; }; say "$name attempt $a failed"; sleep 60; done; exit 1 ) &
}

do_eval() { CUDA_VISIBLE_DEVICES=0 python -m latentspace_train evaluate --run "$1" "${LOCAL[@]}" \
              > "$L/14_eval_$1.log" 2>&1 && date -u > "$M/eval_$1"; }
do_bench() { CUDA_VISIBLE_DEVICES=0 python experiments/08_public_benchmarks.py --name "$1" --adapter "$2" \
               > "$L/08_bench_$1.log" 2>&1; }

until [ -f "$M/grpo_run_A" ]; do sleep 30; done
say "run_A finished: GPU 0 queue starts"
launch eval_run_A  "[ -f $M/eval_run_A ]"   do_eval run_A
launch eval_distill "[ -f $M/eval_distill ]" do_eval distill
launch eval_base   "[ -f $M/eval_base ]"    do_eval base
launch bench_distill "[ -f results/08_public_benchmarks_distill.json ]" do_bench distill outputs/distill/adapter
launch bench_run_A "[ -f results/08_public_benchmarks_run_A.json ]" do_bench run_A outputs/grpo/run_A/adapter
wait
date -u > "$M/gpu0_queue_done"
say "GPU 0 queue finished"
