"""Pick the distillation epoch on the dev split and install it as outputs/distill/adapter.

Reads results/_devsel/checkpoint-N/10_baseline.json (dev split, 2 trials per task) for every
epoch checkpoint and keeps the one with the highest task success under semantic matching
(ties: higher strict success, then the earlier epoch). The test split is never used here.
Writes results/12_distill_selection.json.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = []
for d in sorted((ROOT / "results/_devsel").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
    r = json.loads((d / "10_baseline.json").read_text())
    s = r["modes"]["local"]["summary"]
    o = s["overall"]
    rows.append({"checkpoint": d.name, "step": int(d.name.split("-")[1]),
                 "dev_lenient_success": o.get("lenient_success"), "dev_success": o.get("success"),
                 "dev_validity": o.get("tool_call_validity"), "dev_calls": o.get("tool_call_count"),
                 "dev_unconfirmed_rate": o.get("unconfirmed_irreversible_rate"),
                 "per_category": {c: v.get("lenient_success") for c, v in (s.get("per_category") or {}).items()}})
base = json.loads((ROOT / "results/10_baseline.json").read_text())["modes"]["local"]["summary"]["overall"]
best = max(rows, key=lambda r: (r["dev_lenient_success"] or 0, r["dev_success"] or 0, -r["step"]))
src, dst = ROOT / "outputs/distill" / best["checkpoint"], ROOT / "outputs/distill/adapter"
keep = dst.with_name("adapter_last_epoch")
if dst.exists() and not keep.exists():
    dst.rename(keep)
elif dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("optimizer.pt", "scheduler.pt", "rng_state.pth",
                                                        "trainer_state.json", "training_args.bin"))
out = {"rule": "highest dev lenient_success; ties: strict success, then earlier epoch",
       "dev_split": "data/eval/dev.jsonl, 2 trials per task",
       "base_model_on_dev_1_trial": {"lenient_success": base.get("lenient_success"), "success": base.get("success")},
       "candidates": rows, "selected": best["checkpoint"], "installed_as": "outputs/distill/adapter"}
(ROOT / "results/12_distill_selection.json").write_text(json.dumps(out, indent=2))
for r in rows:
    print(r["checkpoint"], "dev lenient", r["dev_lenient_success"], "strict", r["dev_success"],
          "validity", r["dev_validity"], "<== selected" if r is best else "")
