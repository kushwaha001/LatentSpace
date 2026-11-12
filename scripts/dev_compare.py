"""Print a dev-split result next to the selected distilled model: python scripts/dev_compare.py NAME"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1]
s = json.loads((ROOT / f"results/_devsel/{name}/10_baseline.json").read_text())["modes"]["local"]["summary"]
sel = json.loads((ROOT / "results/12_distill_selection.json").read_text())
ref = next(c for c in sel["candidates"] if c["checkpoint"] == sel["selected"])
o = s["overall"]
print(f"{name} (dev, {o['n_rollouts']} rollouts) vs distilled {sel['selected']}:")
print(f"  success (semantic) {o['lenient_success']:.3f} vs {ref['dev_lenient_success']:.3f}   strict {o['success']:.3f} vs {ref['dev_success']:.3f}")
print(f"  calls/task {o['tool_call_count']:.2f} vs {ref['dev_calls']:.2f}   valid {o['tool_call_validity'] if o['tool_call_validity'] is None else round(o['tool_call_validity'], 3)} vs {ref['dev_validity']:.3f}   unconfirmed {o['unconfirmed_irreversible_rate']:.3f} vs {ref['dev_unconfirmed_rate']:.3f}")
print("  category                    success   (distilled)   calls/task")
for c, v in s["per_category"].items():
    print(f"  {c:26s} {v['lenient_success']:.3f}     ({ref['per_category'][c]:.3f})       {v['tool_call_count']:.2f}")
