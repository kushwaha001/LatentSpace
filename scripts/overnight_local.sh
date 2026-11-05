#!/usr/bin/env bash
# Whole pipeline after the baseline, unattended: local 32B teacher -> distill -> GRPO A||B -> eval.
# Keeps running if the terminal closes. Usage:
#   bash scripts/overnight_local.sh            # start in the background
#   tail -f logs/overnight.log                 # watch (Ctrl+C stops watching, not the run)
cd "$(dirname "$0")/.."
mkdir -p logs
: "${HF_TOKEN:?export HF_TOKEN first}"
export HF_HOME="${HF_HOME:-/mnt/d/hf_cache}"
nohup bash -c 'bash scripts/run_local_2gpu.sh teacher_local distill grpo evaluate; echo "== EXIT CODE $?"' \
  > logs/overnight.log 2>&1 &
echo "started (pid $!). Watch with: tail -f logs/overnight.log"
