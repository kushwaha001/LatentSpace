#!/usr/bin/env bash
# Second training seed for both GRPO runs, started automatically on whichever GPU is free:
#   run_A (seed 7) on GPU 0 once scripts/gpu0_queue.sh has finished;
#   run_B (seed 7) on GPU 1 once scripts/round2.sh has finished (ROUND 2 COMPLETE).
# Same distilled start, data, steps and settings as seed 42; only the seed differs. Outputs:
# outputs/seed7/, results/seed7/ (the seed-42 results are never touched). After training, each
# run is evaluated on the test split with the same protocol (5 trials).
#
#   SEED=7 nohup bash scripts/round2_seed2.sh > logs/round2/seed2.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/ls-venv/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 HF_HOME="${HF_HOME:-/mnt/d/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512,garbage_collection_threshold:0.8}"
SEED="${SEED:-7}"; STEPS="${GRPO_STEPS:-300}"
OUT="outputs/seed$SEED"; RES="results/seed$SEED"; L=logs/round2
LOCAL=(--set precision=bf16 --set student.quantization=null)
mkdir -p "$OUT/.markers" "$RES" "$L"
say() { echo "== [$(date -u +%F_%T)] $*"; }
finished() { python -c "import json,sys; d=json.load(open('$RES/13_grpo_$1.json')); sys.exit(0 if d['error'] is None and (d['global_step'] or 0)>=$STEPS else 1)" 2>/dev/null; }

seed_run() { # seed_run <run> <gpu> <per-device batch>
  local r="$1" g="$2" bs="$3" a
  for a in 1 2 3 4 5 6; do
    finished "$r" && break
    CUDA_VISIBLE_DEVICES="$g" python -m latentspace_train grpo --run "$r" "${LOCAL[@]}" \
      --set seed="$SEED" --set output_root="$OUT" --set results_dir="$RES" \
      --set grpo.resume_from_checkpoint=auto --set training.max_steps="$STEPS" \
      --set training.per_device_train_batch_size="$bs" --set training.gradient_accumulation_steps=$((4 / bs)) \
      >> "$L/13_grpo_${r}_seed$SEED.log" 2>&1 || true
    finished "$r" || { say "$r seed $SEED attempt $a stopped early; resuming with batch 1"; bs=1; sleep 30; }
  done
  finished "$r" || { say "$r seed $SEED failed"; return 1; }
  say "$r seed $SEED trained; evaluating"
  [ -f "$OUT/.markers/eval_$r" ] || CUDA_VISIBLE_DEVICES="$g" python -m latentspace_train evaluate --run "$r" "${LOCAL[@]}" \
      --set output_root="$OUT" --set results_dir="$RES" > "$L/14_eval_${r}_seed$SEED.log" 2>&1 \
      && date -u > "$OUT/.markers/eval_$r"
  say "$r seed $SEED done"
}

( until [ -f outputs/.round2/gpu0_queue_done ]; do sleep 60; done; seed_run run_A 0 2 ) & pa=$!
( until grep -q "ROUND 2 COMPLETE" "$L/round2.log" 2>/dev/null; do sleep 60; done; seed_run run_B 1 1 ) & pb=$!
wait "$pa"; wait "$pb"
say "seed $SEED complete"
