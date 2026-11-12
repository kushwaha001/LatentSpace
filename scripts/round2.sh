#!/usr/bin/env bash
# Round 2, unattended and resumable: teacher (reference-guided) -> distill -> pick the epoch
# on dev -> GRPO run_A + run_B in parallel -> held-out evaluation -> public benchmarks.
#
#   nohup bash scripts/round2.sh > logs/round2/round2.log 2>&1 &
#
# Every step writes a marker in outputs/.round2/ when it succeeds and is skipped on a rerun,
# so after a crash or a machine restart the same command continues where it stopped
# (teacher: missing rollouts only; GRPO: from the newest checkpoint).
# Needs no HF token: models come from the local cache (HF_HUB_OFFLINE=1). Nothing is uploaded.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-/mnt/d/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512,garbage_collection_threshold:0.8}"
GRPO_STEPS="${GRPO_STEPS:-150}"
LOCAL=(--set precision=bf16 --set student.quantization=null)
M=outputs/.round2; L=logs/round2
mkdir -p "$M" "$L"
done_() { [ -f "$M/$1" ]; }
mark() { date -u > "$M/$1"; echo "== [$(date -u +%F_%T)] done: $1"; }
say() { echo "== [$(date -u +%F_%T)] $*"; }

# 1. teacher: two language shards, one per GPU, resumable
if ! done_ teacher; then
  say "teacher collection (resume)"
  T=(--set precision=bf16 --set teacher.backend=local --set teacher.local_fallback.model=aya-expanse-32b
     --set teacher.local_fallback.quantization=4bit --set teacher.resume=true)
  need() { [ "$(cat "outputs/$1/teacher_collect/raw.jsonl" 2>/dev/null | wc -l)" -lt 188 ]; }
  pids=()
  if need _shardA; then CUDA_VISIBLE_DEVICES=0 python -m latentspace_train teacher_collect "${T[@]}" \
      --set 'tasks.languages=[en,hi]' --set output_root=outputs/_shardA --set results_dir=results/_shardA \
      >> "$L/11_teacher_A_resume.log" 2>&1 & pids+=($!); fi
  if need _shardB; then CUDA_VISIBLE_DEVICES=1 python -m latentspace_train teacher_collect "${T[@]}" \
      --set 'tasks.languages=[mr,hinglish]' --set output_root=outputs/_shardB --set results_dir=results/_shardB \
      >> "$L/11_teacher_B_resume.log" 2>&1 & pids+=($!); fi
  for p in "${pids[@]}"; do wait "$p"; done
  if need _shardA || need _shardB; then echo "teacher shards incomplete"; exit 1; fi
  python scripts/merge_teacher_shards.py outputs/_shardA outputs/_shardB > "$L/11_teacher_merge.log" 2>&1
  mark teacher
fi

# 2. distill: 3 epochs, one checkpoint per epoch
if ! done_ distill; then
  say "distill"
  rm -rf outputs/distill
  CUDA_VISIBLE_DEVICES=1 python -m latentspace_train distill "${LOCAL[@]}" > "$L/12_distill.log" 2>&1
  mark distill
fi

# 3. choose the epoch on the dev split (2 trials per task), never on test
if ! done_ select; then
  say "dev selection of the distill epoch"
  mapfile -t CK < <(ls -d outputs/distill/checkpoint-* | sort -t- -k2 -n)
  dev() { # dev <gpu> <checkpoint>
    local n; n="$(basename "$2")"
    [ -f "results/_devsel/$n/10_baseline.json" ] && return 0
    CUDA_VISIBLE_DEVICES="$1" python -m latentspace_train baseline "${LOCAL[@]}" \
      --set "student.adapter=$2" --set baseline.trials=2 \
      --set "output_root=outputs/_devsel/$n" --set "results_dir=results/_devsel/$n" > "$L/12_devsel_$n.log" 2>&1
  }
  i=0; pids=()
  for c in "${CK[@]}"; do dev $((i % 2)) "$c" & pids+=($!); i=$((i + 1))
    if [ $((i % 2)) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
  done
  for p in "${pids[@]}"; do wait "$p"; done
  python scripts/select_distill_epoch.py > "$L/12_select.log" 2>&1
  cat "$L/12_select.log"
  mark select
fi

# 4. GRPO: run_A (GPU 0) and run_B (GPU 1) at the same time, identical except the reward.
#    Each resumes from its newest checkpoint (save_steps 10). A run that ran out of memory
#    continues with 1 completion per backward pass instead of 2: same optimizer step
#    (1 prompt x 4 generations), less memory. run_B starts that way (it OOMed at 2 on 11 Nov).
grpo_run() { # grpo_run <run> <gpu> <first per-device batch>
  local r="$1" g="$2" bs="$3" a
  for a in 1 2 3 4 5; do
    if python -c "import json,sys; d=json.load(open('results/13_grpo_$r.json')); sys.exit(0 if d['error'] is None and (d['global_step'] or 0) >= $GRPO_STEPS else 1)" 2>/dev/null \
       && [ -f "$M/grpo_$r" ]; then return 0; fi
    CUDA_VISIBLE_DEVICES="$g" python -m latentspace_train grpo --run "$r" "${LOCAL[@]}" \
      --set grpo.resume_from_checkpoint=auto --set training.max_steps="$GRPO_STEPS" \
      --set training.per_device_train_batch_size="$bs" \
      --set training.gradient_accumulation_steps=$((4 / bs)) \
      >> "$L/13_grpo_${r}.log" 2>&1 || true
    if python -c "import json,sys; d=json.load(open('results/13_grpo_$r.json')); sys.exit(0 if d['error'] is None and (d['global_step'] or 0) >= $GRPO_STEPS else 1)"; then
      date -u > "$M/grpo_$r"; return 0
    fi
    say "$r attempt $a stopped early; resuming with batch 1"; bs=1; sleep 30
  done
  return 1
}
if ! done_ grpo; then
  say "GRPO run_A + run_B, $GRPO_STEPS steps (resume)"
  grpo_run run_A 0 2 & pa=$!
  grpo_run run_B 1 1 & pb=$!
  wait "$pa"; wait "$pb"
  mark grpo
fi

# 5. held-out test split, 5 trials, all four checkpoints, then comparisons
if ! done_ evaluate; then
  say "evaluate"
  ev() { [ -f "$M/eval_$2" ] && return 0
         CUDA_VISIBLE_DEVICES="$1" python -m latentspace_train evaluate --run "$2" "${LOCAL[@]}" > "$L/14_eval_$2.log" 2>&1 \
           && date -u > "$M/eval_$2"; }
  ( ev 0 base && ev 0 run_A ) & pa=$!
  ( ev 1 distill && ev 1 run_B ) & pb=$!
  wait "$pa"; wait "$pb"
  CUDA_VISIBLE_DEVICES="" python -m latentspace_train evaluate --compare-only > "$L/14_compare.log" 2>&1
  mark evaluate
fi

# 6. public benchmarks (When2Call, BFCL-Hi), all four checkpoints
if ! done_ bench; then
  say "public benchmarks"
  bn() { [ -f "results/08_public_benchmarks_$2.json" ] && return 0
         CUDA_VISIBLE_DEVICES="$1" python experiments/08_public_benchmarks.py --name "$2" ${3:+--adapter "$3"} > "$L/08_bench_$2.log" 2>&1; }
  ( bn 0 base && bn 0 run_A outputs/grpo/run_A/adapter ) & pa=$!
  ( bn 1 distill outputs/distill/adapter && bn 1 run_B outputs/grpo/run_B/adapter ) & pb=$!
  wait "$pa"; wait "$pb"
  python experiments/08_public_benchmarks.py --report > "$L/08_report.log" 2>&1
  mark bench
fi
say "ROUND 2 COMPLETE"
