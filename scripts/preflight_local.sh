#!/usr/bin/env bash
# Checks before a long local run. Prints OK / FAIL per check; nothing is downloaded except
# two small config files. Usage: bash scripts/preflight_local.sh
cd "$(dirname "$0")/.."
export PYTHONPATH=src
fail=0
ok()  { echo "  OK   $*"; }
bad() { echo "  FAIL $*"; fail=1; }

echo "== 1. GPUs"
python - <<'PY' && ok "torch sees 2 GPUs" || bad "torch cannot use 2 GPUs (install the cu126 build, see docs/RUN_LOCAL.md)"
import torch, sys
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
for i in range(n):
    free, total = torch.cuda.mem_get_info(i)
    print(f"       GPU {i}: {torch.cuda.get_device_name(i)}, {free/2**30:.1f} of {total/2**30:.1f} GB free")
sys.exit(0 if n >= 2 else 1)
PY

echo "== 2. bitsandbytes (4-bit teacher)"
python -c "import bitsandbytes, torch; torch.zeros(1, device='cuda'); print('       bitsandbytes', bitsandbytes.__version__)" 2>/dev/null \
  && ok "bitsandbytes imports" || bad "bitsandbytes missing: pip install bitsandbytes"

echo "== 3. Hugging Face token and model access"
python - <<'PY' && ok "token valid, both models accessible" || bad "token or model access problem (see message above)"
import os, sys
from huggingface_hub import whoami, hf_hub_download
if not os.environ.get("HF_TOKEN"):
    print("       HF_TOKEN is not set: export HF_TOKEN=..."); sys.exit(1)
try:
    print("       logged in as", whoami()["name"])
except Exception as e:
    print("       invalid token:", str(e).splitlines()[0]); sys.exit(1)
bad = 0
for repo in ("CohereLabs/tiny-aya-global", "CohereLabs/aya-expanse-32b"):
    try:
        hf_hub_download(repo, "config.json"); print("       access ok:", repo)
    except Exception as e:
        bad = 1
        print("       NO ACCESS:", repo, "->", str(e).splitlines()[0][:160])
        print("         accept the licence at https://huggingface.co/" + repo,
              "and tick 'public gated repos' on the token")
sys.exit(bad)
PY

echo "== 4. Disk space where models are cached"
HF_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_DIR"
free_gb=$(df -BG --output=avail "$HF_DIR" | tail -1 | tr -dc '0-9')
echo "       HF_HOME=$HF_DIR  free: ${free_gb} GB"
[ "${free_gb:-0}" -ge 80 ] && ok ">= 80 GB free (32B teacher ~65 GB + adapters)" \
  || bad "need ~80 GB free in $HF_DIR (export HF_HOME=/mnt/d/hf_cache)"
home_gb=$(df -BG --output=avail "$PWD" | tail -1 | tr -dc '0-9')
echo "       repo folder free: ${home_gb} GB"
[ "${home_gb:-0}" -ge 30 ] && ok ">= 30 GB free for checkpoints/outputs" || bad "need ~30 GB free next to the repo"

echo "== 5. Unit tests"
last=$(python -m pytest -q tests 2>&1 | tail -1); echo "       $last"
echo "$last" | grep -q "passed" && ! echo "$last" | grep -qE "failed|error" \
  && ok "tests pass" || bad "tests fail: python -m pytest -q tests"

echo "== 6. Earlier results"
[ -f results/10_baseline.json ] && ok "results/10_baseline.json exists" || bad "run the baseline first"

echo
[ $fail -eq 0 ] && echo "ALL CHECKS PASSED" || echo "SOME CHECKS FAILED (fix the FAIL lines above)"
exit $fail
