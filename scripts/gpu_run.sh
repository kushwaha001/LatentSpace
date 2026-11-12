#!/usr/bin/env bash
# Run one GPU job safely next to the Windows desktop:
#   bash scripts/gpu_run.sh <gpu> <command ...>
# - one job at a time per GPU (flock): jobs queue instead of sharing and exhausting memory;
# - a hard memory cap per process (GPU 0 drives the display and Remote Desktop: 70% = 17 GB,
#   leaving ~7 GB for Windows; GPU 1: 92%), so a job that needs more fails inside itself
#   (and is retried / resumed by its caller) instead of freezing the desktop;
# - low CPU priority (nice 10), so the desktop always gets the CPU first.
G="$1"; shift
FRAC="${LS_GPU_MEM_FRACTION:-$([ "$G" = 0 ] && echo 0.70 || echo 0.92)}"
LOCK="/tmp/latentspace_gpu$G.lock"
exec flock "$LOCK" env CUDA_VISIBLE_DEVICES="$G" LS_GPU_MEM_FRACTION="$FRAC" nice -n 10 "$@"
