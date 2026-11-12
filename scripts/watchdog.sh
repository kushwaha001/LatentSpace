#!/usr/bin/env bash
# Exit (and print why) as soon as something needs attention: a crash/restart of a job, a stall,
# a GPU error, or a milestone. Used to wake the monitoring session.
cd "$(dirname "$0")/.."
B3L=logs/round2/grpo_B3_outer.log; Q=logs/round2/round2_final.log
n_b3=$(grep -c "stopped early\|failed" $B3L 2>/dev/null); n_q=$(grep -c "failed" $Q 2>/dev/null)
last_step=-1; last_change=$(date +%s)
step() { tr '\r' '\n' < logs/round2/13_grpo_run_B3.log | grep -o '[0-9]*/300' | tail -1 | cut -d/ -f1; }
while true; do
  [ "$(grep -c "stopped early\|failed" $B3L)" -gt "$n_b3" ] && { echo "DISRUPTION: run_B3 crashed: $(tail -1 $B3L)"; exit 0; }
  [ "$(grep -c "failed" $Q)" -gt "$n_q" ] && { echo "DISRUPTION: queue job failed: $(grep failed $Q | tail -1)"; exit 0; }
  D=logs/round2/run_dpo.out
  grep -q "DPO failed" $D 2>/dev/null && [ ! -f /tmp/latentspace_seen_dpofail ] && { touch /tmp/latentspace_seen_dpofail; echo "DISRUPTION: DPO failed"; tail -5 logs/round2/10_dpo.log; exit 0; }
  grep -q "DPO dev check done" $D 2>/dev/null && [ ! -f /tmp/latentspace_seen_dpodev ] && { touch /tmp/latentspace_seen_dpodev; echo "MILESTONE: DPO dev check"; cat $D; exit 0; }
  grep -q "DPO done" $D 2>/dev/null && [ ! -f /tmp/latentspace_seen_dpodone ] && { touch /tmp/latentspace_seen_dpodone; echo "MILESTONE: DPO test evaluation done"; exit 0; }
  grep -q "run_B3 done" $B3L && { echo "MILESTONE: run_B3 trained and evaluated"; exit 0; }
  for s in 60 120; do f=logs/round2/devcheck_B3_$s.out; [ -s $f ] && grep -q category $f && [ ! -f /tmp/latentspace_seen_dev$s ] && { touch /tmp/latentspace_seen_dev$s; echo "MILESTONE: run_B3 dev check step $s"; cat $f; exit 0; }; done
  grep -q "ROUND 2 COMPLETE" $Q && [ ! -f /tmp/latentspace_seen_r2 ] && { touch /tmp/latentspace_seen_r2; echo "MILESTONE: first 5 models compared"; exit 0; }
  nvidia-smi >/dev/null 2>&1 || { sleep 60; nvidia-smi >/dev/null 2>&1 || { echo "DISRUPTION: nvidia-smi failing (GPU access blocked?)"; exit 0; }; }
  s=$(step); now=$(date +%s)
  if [ "$s" != "$last_step" ]; then last_step=$s; last_change=$now
  elif [ $((now - last_change)) -gt 1500 ] && ! grep -q "run_B3 trained" $B3L; then echo "DISRUPTION: run_B3 stalled at step $s for 25 min"; exit 0; fi
  sleep 60
done
