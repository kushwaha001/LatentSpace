#!/usr/bin/env bash
# Run the full pipeline on one machine with two 24 GB GPUs (tested target: 2x RTX 4500 Ada).
#
#   export HF_TOKEN=...            # read + Inference Providers (+ write if PUSH=1)
#   bash scripts/run_local_2gpu.sh [stage ...]     # default: all stages in order
#
# Stages: baseline teacher teacher_local distill grpo evaluate
#   teacher       Aya Expanse 32B through the HF router (paid, capped by MAX_USD)
#   teacher_local Aya Expanse 32B in 4-bit on one local GPU (free, ~65 GB download, slower)
# Env:   PUSH=1   upload adapters / datasets / logs to the Hub (namespace kanishk001)
#        GPU_A=0 GPU_B=1   which GPUs run run_A / run_B (single-GPU stages use GPU_B)
#        MAX_USD=5         budget for teacher collection through the HF router
#
# Differences from the Kaggle (2x T4) defaults, set below with --set:
#   precision bf16 (Ada supports it; Cohere models are trained in bf16) and no 4-bit
#   quantization (the 3.35B student fits in 24 GB in bf16), which is faster and avoids
#   bitsandbytes. Everything else, including every GRPO setting shared by run_A and run_B,
#   comes from configs/ unchanged.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"
MAX_USD="${MAX_USD:-5}"
PUSH_FLAG=()
[ "${PUSH:-0}" = "1" ] && PUSH_FLAG=(--push)
LOCAL=(--set precision=bf16 --set student.quantization=null)
mkdir -p logs

run() {  # run <gpu> <log> <args...>
  local gpu="$1" log="$2"; shift 2
  echo "== [GPU $gpu] python -m latentspace_train $*  (log: logs/$log.log)"
  CUDA_VISIBLE_DEVICES="$gpu" python -m latentspace_train "$@" 2>&1 | tee "logs/$log.log"
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(baseline teacher distill grpo evaluate)

for s in "${stages[@]}"; do
  case "$s" in
    baseline)
      # 1. "Before": untrained Tiny Aya on the eval (dev) split.
      run "$GPU_B" 10_baseline baseline "${LOCAL[@]}" "${PUSH_FLAG[@]}" ;;
    teacher)
      # 2. Aya Expanse 32B via the HF router (no GPU), kept only if valid and confirmed.
      : "${HF_TOKEN:?set HF_TOKEN first}"
      run "" 11_teacher_collect teacher_collect --max-usd "$MAX_USD" "${PUSH_FLAG[@]}" ;;
    teacher_local)
      # 2b. Same teacher, run on GPU_B in 4-bit (no HF credits).
      run "$GPU_B" 11_teacher_collect teacher_collect --set precision=bf16 \
        --set teacher.backend=local --set teacher.local_fallback.model=aya-expanse-32b \
        --set teacher.local_fallback.quantization=4bit "${PUSH_FLAG[@]}" ;;
    distill)
      # 3. LoRA SFT of the student on the filtered teacher trajectories.
      run "$GPU_B" 12_distill distill "${LOCAL[@]}" "${PUSH_FLAG[@]}" ;;
    grpo)
      # 4. run_A (AgentTune default reward) and run_B (audit reward) at the same time,
      #    one per GPU, identical settings except the reward.
      run "$GPU_A" 13_grpo_run_A grpo --run run_A "${LOCAL[@]}" "${PUSH_FLAG[@]}" &
      pid_a=$!
      run "$GPU_B" 13_grpo_run_B grpo --run run_B "${LOCAL[@]}" "${PUSH_FLAG[@]}" &
      pid_b=$!
      wait "$pid_a"; wait "$pid_b" ;;
    evaluate)
      # 5. Held-out test split, 5 trials, per language; two checkpoints per GPU, then compare.
      ( run "$GPU_A" 14_eval_base evaluate --run base "${LOCAL[@]}" "${PUSH_FLAG[@]}" &&
        run "$GPU_A" 14_eval_run_A evaluate --run run_A "${LOCAL[@]}" "${PUSH_FLAG[@]}" ) &
      pid_a=$!
      ( run "$GPU_B" 14_eval_distill evaluate --run distill "${LOCAL[@]}" "${PUSH_FLAG[@]}" &&
        run "$GPU_B" 14_eval_run_B evaluate --run run_B "${LOCAL[@]}" "${PUSH_FLAG[@]}" ) &
      pid_b=$!
      wait "$pid_a"; wait "$pid_b"
      run "" 14_evaluate_compare evaluate --compare-only "${PUSH_FLAG[@]}" ;;
    *) echo "unknown stage: $s (baseline teacher teacher_local distill grpo evaluate)"; exit 2 ;;
  esac
done
echo "== done. Results: results/1*.json   Logs: logs/"
