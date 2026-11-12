#!/usr/bin/env bash
# Audit-ranked DPO on GPU 0, then its dev check and test evaluation (resumable by markers).
set -uo pipefail
cd "$(dirname "$0")/.."
source ~/ls-venv/bin/activate
export PYTHONPATH=src HF_HUB_OFFLINE=1 HF_HOME="${HF_HOME:-/mnt/d/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
NAME="${NAME:-audit}"; EXTRA="${EXTRA:-}"; EVN="${EVN:-dpo_audit}"
L=logs/round2; M=outputs/.round2; A=outputs/dpo/$NAME/adapter
say() { echo "== [$(date -u +%F_%T)] $*"; }
if [ ! -f "$A/adapter_model.safetensors" ]; then
  for a in 1 2 3; do
    # peak 19 GB (2 x 3.7k tokens x 261k vocab): a 20.4 GB cap still leaves ~3.5 GB for Windows
    LS_GPU_MEM_FRACTION=0.85 bash scripts/gpu_run.sh 0 python experiments/10_audit_dpo.py --name $NAME $EXTRA > "$L/10_dpo_$NAME.log" 2>&1 && break
    say "DPO attempt $a failed"; sleep 60
  done
fi
[ -f "$A/adapter_model.safetensors" ] || { say "DPO failed"; exit 1; }
say "DPO trained"
N=dpo_${NAME}_dev
[ -f "results/_devsel/$N/10_baseline.json" ] || bash scripts/gpu_run.sh 0 python -m latentspace_train baseline \
  --set precision=bf16 --set student.quantization=null --set student.adapter=$A \
  --set student.base_adapter=outputs/distill/adapter --set baseline.trials=2 \
  --set output_root=outputs/_devsel/$N --set results_dir=results/_devsel/$N > "$L/12_devcheck_$N.log" 2>&1
python scripts/dev_compare.py "$N"
say "DPO dev check done"
[ -f "$M/eval_$EVN" ] || [ -n "${NO_TEST:-}" ] || bash scripts/gpu_run.sh 0 python -m latentspace_train evaluate --run $EVN \
  --set precision=bf16 --set student.quantization=null > "$L/14_eval_$EVN.log" 2>&1 && date -u > "$M/eval_$EVN"
say "DPO done"
