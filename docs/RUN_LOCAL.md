# Running the experiments on a local 2-GPU machine

Target machine: Windows 11, **2x NVIDIA RTX 4500 Ada (24 GB each)**, driver 552.74 (CUDA 12.4).
Two 24 GB Ada cards are faster than Kaggle's 2x T4 and support bf16, so the local script turns
4-bit quantization off and uses bf16. Everything else follows `configs/`.

## 1. One-time setup (WSL2 Ubuntu, recommended)

AgentTune and TRL are developed and tested on Linux. WSL2 runs Linux on Windows with full GPU
access through the Windows NVIDIA driver (no Linux driver install).

In **PowerShell as Administrator** (on a lab PC this may need the lab admin):

```powershell
wsl --install -d Ubuntu-24.04
```

Restart if asked, open **Ubuntu** from the Start menu, create a user, then:

```bash
nvidia-smi                      # must list both RTX 4500 Ada cards
sudo apt update && sudo apt install -y git python3.12-venv build-essential
python3.12 -m venv ~/ls-venv && source ~/ls-venv/bin/activate
git clone https://github.com/kushwaha001/LatentSpace.git ~/LatentSpace
cd ~/LatentSpace
pip install "torch==2.11.0" --index-url https://download.pytorch.org/whl/cu126   # CUDA 12.x build
bash scripts/install_deps.sh    # AuditKIT 1.1.3 + AgentTune d4db4c12 + its dependencies
PYTHONPATH=src python -m pytest -q tests     # expect all tests to pass
```

Use the `cu126` build: the plain PyPI wheel is built for CUDA 13.0, and a driver that reports
CUDA 12.4 (552.74) fails with "The NVIDIA driver on your system is too old". Check with
`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` (expect
`True 2`). If it is still False, update the Windows NVIDIA driver and run `wsl --shutdown`.

**Disk:** models and caches download to `~/.cache/huggingface` inside WSL (about 10 GB for Tiny
Aya plus adapters). To use another drive, set `export HF_HOME=/mnt/d/hf_cache` (D: drive) before
running.

## 2. Hugging Face token (never in a file or commit)

Create a fine-grained token on huggingface.co (account `kanishk001`) with *read gated repos*,
*Inference Providers*, and *write* (only if you will `PUSH=1`). In the WSL terminal:

```bash
read -rs HF_TOKEN && export HF_TOKEN   # paste, press Enter; it is not shown or saved
```

On a shared lab PC, do not run `huggingface-cli login` (it saves the token to disk); the
`export` above lasts only for that terminal. Close the terminal when you are done.

## 3. Run the experiments

```bash
cd ~/LatentSpace && source ~/ls-venv/bin/activate
bash scripts/run_local_2gpu.sh baseline     # 10: untrained Tiny Aya        (~20-40 min, GPU 1)
bash scripts/run_local_2gpu.sh teacher      # 11: Aya Expanse 32B via HF     (~1-2 h, API, <= $5)
bash scripts/run_local_2gpu.sh distill      # 12: LoRA SFT on teacher data   (~20-40 min, GPU 1)
bash scripts/run_local_2gpu.sh grpo         # 13: run_A on GPU 0 + run_B on GPU 1 at once (~1.5-3 h)
bash scripts/run_local_2gpu.sh evaluate     # 14: test split x 5 trials, per language (~2-3 h)
```

Or everything in order: `bash scripts/run_local_2gpu.sh`. Add `PUSH=1` to upload adapters,
datasets and logs to the Hub (needed for the submission). Each stage writes
`results/NN_*.json` and a log in `logs/`. Times are estimates; record the real ones.

GPU 0 also drives the display. If a run on GPU 0 runs out of memory, close other GPU apps or
run the GRPO stages one after the other: `GPU_A=1 bash scripts/run_local_2gpu.sh grpo`.

Quick check before a long run: add `--limit 4` through the CLI directly, e.g.
`PYTHONPATH=src python -m latentspace_train baseline --set precision=bf16 --set student.quantization=null --limit 4`.

### Without HF credits: local 32B teacher, overnight

`teacher_local` runs the same teacher (Aya Expanse 32B) in 4-bit on one GPU instead of the HF
router: no credits, about 65 GB download, a few hours. Check first, then start everything after
the baseline in the background (it keeps running if the terminal closes):

```bash
export HF_HOME=/mnt/d/hf_cache
bash scripts/preflight_local.sh          # must end with ALL CHECKS PASSED
bash scripts/overnight_local.sh          # teacher_local -> distill -> grpo (A || B) -> evaluate
tail -f logs/overnight.log               # watch; Ctrl+C stops watching only
```

In the morning, `logs/overnight.log` ends with `== EXIT CODE 0` if every stage finished.

## 4. What each experiment answers

| # | Stage | Question |
|---|---|---|
| 10 | baseline | How good is untrained Tiny Aya at our tools, per language? (the "before") |
| 11 | teacher_collect | Can Aya Expanse 32B produce valid, confirmed examples to learn from? (kept / collected, cost) |
| 12 | distill | Does copying the teacher make the student "sometimes right" so GRPO can learn? |
| 13 | grpo run_A / run_B | Same model, data and settings; only the reward differs. Which reward trains better tool use? |
| 14 | evaluate | On held-out tasks (incl. unseen tools), per language, 5 trials: base vs distill vs run_A vs run_B, with bootstrap significance |

## 5. After each stage, check

- `results/10_baseline.json`: validity and success per language; how often the base model calls a tool at all.
- `results/11_teacher_collect.json`: `kept / collected`, and the `budget` block (tokens and dollars). If `budget` shows 0 tokens, the provider returned no usage: check spend on huggingface.co/settings/billing.
- `results/13_grpo_run_{A,B}.json`: reward curve and the audit metrics of each run's own rollouts.
- `results/14_evaluate*.json`: the final table and the `run_A` vs `run_B` comparison.

Send the results files back (or push them) and they go into the write-up.
