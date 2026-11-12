"""Merge teacher_collect shards (one per GPU, split by language) into outputs/teacher_collect/.

    python scripts/merge_teacher_shards.py outputs/_shardA outputs/_shardB

Re-applies the current filter (``teacher.filter`` in configs/11_teacher_collect.yaml) to every
raw trajectory, so a filter added after the shards started (``require_reply_language``) is
applied uniformly, and writes results/11_teacher_collect.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from latentspace_train.config import load_config  # noqa: E402
from latentspace_train.scoring import summarise  # noqa: E402
from latentspace_train.stages.teacher_collect import reject_reason  # noqa: E402

cfg = load_config("configs/11_teacher_collect.yaml")
filt = cfg["teacher"]["filter"]
shards = [Path(a) for a in sys.argv[1:]]
rows, metas = [], []
for s in shards:
    rows += [json.loads(l) for l in open(s / "teacher_collect" / "raw.jsonl", encoding="utf-8")]
    res = ROOT / "results" / s.name / "11_teacher_collect.json"
    if res.exists():
        metas.append(json.loads(res.read_text()))
out_dir = ROOT / cfg.get("output_root", "outputs") / "teacher_collect"
out_dir.mkdir(parents=True, exist_ok=True)
reasons, by_cat, by_lang, seen_lang = Counter(), Counter(), Counter(), Counter()
kept = []
for r in rows:
    why = reject_reason(r, r["scores"], filt)
    r["kept"], r["reject_reason"] = why is None, why
    reasons[why or "kept"] += 1
    seen_lang[r["language"]] += 1
    if why is None:
        kept.append(r)
        by_cat[r.get("category")] += 1
        by_lang[r["language"]] += 1
for name, data in (("raw.jsonl", rows), ("filtered.jsonl", kept)):
    with open(out_dir / name, "w", encoding="utf-8") as f:
        for r in data:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
m0 = metas[0] if metas else {}
out = {
    "stage": "teacher_collect", "merged_from": [str(s) for s in shards],
    "teacher": m0.get("teacher"), "split": "train",
    "n_tasks": len({r["task_id"] for r in rows}),
    "samples_per_task": cfg["teacher"]["samples_per_task"],
    "guided": {"enabled": bool(cfg["teacher"].get("guided")),
               "tasks": len({r["task_id"] for r in rows if r.get("guided")})},
    "filter": filt, "seen": len(rows), "kept": len(kept),
    "kept_guided": sum(bool(r.get("guided")) for r in kept),
    "reasons": dict(reasons), "kept_by_category": dict(by_cat),
    "kept_by_language": {k: {"kept": by_lang[k], "seen": v, "rate": round(by_lang[k] / v, 3)}
                         for k, v in sorted(seen_lang.items())},
    "teacher_summary": summarise([{**r["scores"], "language": r["language"],
                                   "category": r.get("category")} for r in rows]),
    "seconds_per_shard": [m.get("seconds") for m in metas],
    "files": {"raw": str(out_dir / "raw.jsonl"), "filtered": str(out_dir / "filtered.jsonl")},
    "system_prompt": cfg.get("tasks", {}).get("system_prompt"),
    "versions": m0.get("versions"), "hardware": m0.get("hardware"), "config": cfg,
}
(ROOT / "results/11_teacher_collect.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str))
print(json.dumps({k: out[k] for k in ("seen", "kept", "kept_guided", "reasons", "kept_by_category",
                                      "kept_by_language")}, indent=1, ensure_ascii=False))
