#!/usr/bin/env bash
# CPU wiring check of every stage, in the order the notebooks run them, with a tiny randomly
# initialised Llama and the stub tasks (no network, no GPU). Writes results/smoke/*.json.
#
#   bash scripts/dry_run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
rm -rf outputs/smoke results/smoke
run() { echo "== $*"; python -m latentspace_train "$@" --dry-run 2>&1 | grep -E "^\[|Error|Traceback" | grep -v "offline-deterministic-only" || true; }
# run_A logs litellm "LLM Provider NOT provided ... offline-deterministic-only" on every
# reward call: upstream hybrid_prm_reward calls a judge even with use_llm_judge=False
# (FINDINGS.md B3). Filtered above; it does not affect the score.
run baseline
run teacher_collect
run distill
run grpo --run run_A
run grpo --run run_B
run evaluate --run run_A
run evaluate --run run_B
run evaluate --run base
run evaluate --run distill
run evaluate --compare-only
ls results/smoke
