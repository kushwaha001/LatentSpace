#!/usr/bin/env bash
# Install the training stack on Kaggle / Colab / an HF GPU Space / a local GPU box.
#
#   bash scripts/install_deps.sh [AGENTTUNE_DIR]
#
# - AuditKIT 1.1.3 from PyPI
# - AgentTune at the pinned commit d4db4c12, `pip install --no-deps -e`, then its base
#   dependencies EXCEPT torch / torchvision / torchao / compressed-tensors, so the image's
#   preinstalled CUDA torch is kept (a constraints file pins it)
# - bitsandbytes when a GPU is present (4-bit loading)
set -euo pipefail

AT_DIR="${1:-${AGENTTUNE_DIR:-$HOME/AgentTune}}"
AT_COMMIT="d4db4c12d52342227c4113a70183d0f64aa2f345"
PIP="python -m pip"
TMP="$(mktemp -d)"

echo "== AuditKIT"
$PIP install -q "auditkit==1.1.3"

echo "== AgentTune @ ${AT_COMMIT:0:8} -> $AT_DIR"
if [ ! -d "$AT_DIR/.git" ]; then
  git clone -q https://github.com/Lexsi-Labs/AgentTune "$AT_DIR"
fi
git -C "$AT_DIR" fetch -q origin || true
git -C "$AT_DIR" checkout -q "$AT_COMMIT"

EXTRA=""
if ! python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
  echo "WARNING: Python $(python -V 2>&1) < 3.12; AgentTune declares >=3.12. Installing anyway"
  echo "         (--ignore-requires-python). If imports fail, use a Python 3.12 image."
  EXTRA="--ignore-requires-python"
fi
$PIP install -q --no-deps $EXTRA -e "$AT_DIR"

# AgentTune's base dependencies minus the torch family (keep the image's CUDA build).
python - "$AT_DIR/pyproject.toml" > "$TMP/deps.txt" <<'PY'
import re, sys, tomllib
deps = tomllib.load(open(sys.argv[1], "rb"))["project"]["dependencies"]
skip = {"torch", "torchvision", "torchao", "compressed-tensors", "bitsandbytes"}
for d in deps:
    if re.split(r"[<>=!~\[ ;]", d, maxsplit=1)[0].strip().lower() not in skip:
        print(d)
PY
: > "$TMP/constraints.txt"
if python -c "import torch" 2>/dev/null; then
  TV="$(python -c 'import torch; print(torch.__version__)')"
  echo "torch==${TV%%+*}" >> "$TMP/constraints.txt"  # PEP 440: matches the local +cuXXX build
  echo "== keeping preinstalled torch ${TV}"
else
  echo "== no torch found: installing AgentTune's pin (torch==2.11.0)"
  $PIP install -q "torch==2.11.0"
fi
echo "== AgentTune base dependencies ($(wc -l < "$TMP/deps.txt") packages)"
echo "   (pip will warn that agenttune requires torchvision/torchao/...: skipped on purpose)"
if ! $PIP install -q -r "$TMP/deps.txt" -c "$TMP/constraints.txt"; then
  # rouge-score / sqlitedict / word2number sdists fail under some Debian setuptools
  $PIP install -q --no-build-isolation rouge-score sqlitedict word2number
  $PIP install -q -r "$TMP/deps.txt" -c "$TMP/constraints.txt"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "== bitsandbytes (GPU)"
  $PIP install -q "bitsandbytes>=0.45" -c "$TMP/constraints.txt"
fi

# peft>=0.19 raises at train() when an old torchao (<0.16) is installed (Colab ships 0.10).
python - <<'PY' || $PIP uninstall -y -q torchao
from importlib.metadata import version, PackageNotFoundError
try:
    v = tuple(int(x) for x in version("torchao").split(".")[:2])
except PackageNotFoundError:
    raise SystemExit(0)
raise SystemExit(0 if v >= (0, 16) else 1)
PY

python - <<'PY'
from importlib.metadata import version
import torch
pk = ["torch", "transformers", "trl", "peft", "accelerate", "datasets", "agenttune", "auditkit", "litellm"]
print("== installed:", ", ".join(f"{p} {version(p)}" for p in pk))
print("== cuda:", torch.cuda.is_available(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
