"""Experiment 09: automatic text checks on the 4-language eval set (before human review).

For every non-English row, compared with its English sibling (same base_id), turn by turn:
- numbers: the digit groups (dates, counts, PNRs, train numbers, amounts; Devanagari digits
  mapped) must be the same multiset;
- ids: Latin tokens with a digit or in capitals (S20001, CSMT, 3A) must all be kept;
- script: Hindi and Marathi turns must contain Devanagari; Hinglish turns must not;
- Marathi rows must not contain Hindi-only function words, and must differ from the Hindi row.

References (all 360 rows): the expected calls are replayed in order on a fresh mock backend.
Every call must succeed, and every station code, service id and PNR used after the first step
must appear in the conversation or in an earlier tool result, so the reference path is one an
agent can actually follow without guessing. No-call categories must have an empty reference.

It cannot judge fluency or meaning: that is the native-speaker review
(data/eval/review_checklist.csv). Writes results/09_dataset_audit.json.
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = str.maketrans("०१२३४५६७८९", "0123456789")
HINDI_ONLY = {"है", "हैं", "कीजिए", "चाहिए", "मुझे", "मेरा", "मेरी", "क्या", "करें", "कृपया", "नहीं", "और", "बताइए", "दीजिए"}
MARATHI_OK = {"कृपया", "आणि"}  # shared or Marathi words that look like Hindi ones


def digits(t): return sorted(re.findall(r"\d+", t.translate(DEV)))
def ids(t): return sorted(set(re.findall(r"\b(?:[A-Z]{2,}\d*|[A-Z]?\d+[A-Z]+\d*|S\d{5}(?:-[A-Z]+-[A-Z]+)?)\b", t)))
def has_dev(t): return bool(re.search(r"[ऀ-ॿ]", t))


def main():
    rows = [json.loads(l) for l in open(ROOT / "data/eval/eval_set.jsonl", encoding="utf-8")]
    by = defaultdict(dict)
    for r in rows:
        by[r["base_id"]][r["language"]] = r
    issues, n_turns = [], 0
    for b, langs in sorted(by.items()):
        en = langs["en"]
        for lg in ("hi", "mr", "hinglish"):
            r = langs[lg]
            if len(r["messages"]) != len(en["messages"]):
                issues.append({"id": r["id"], "split": r["split"], "check": "turn_count"})
                continue
            for i, (m, e) in enumerate(zip(r["messages"], en["messages"])):
                n_turns += 1
                t, et = m["content"], e["content"]
                def add(check, detail):
                    issues.append({"id": r["id"], "split": r["split"], "turn": i, "role": m["role"],
                                   "check": check, "detail": detail, "text": t, "english": et})
                if digits(t) != digits(et):
                    add("numbers_differ", {"got": digits(t), "english": digits(et)})
                missing = [x for x in ids(et) if x not in t and x.upper() not in t.upper()]
                if missing and lg == "hinglish":
                    add("id_missing", missing)
                if lg in ("hi", "mr") and not has_dev(t):
                    add("no_devanagari", None)
                if lg == "hinglish" and has_dev(t):
                    add("devanagari_in_hinglish", None)
                if lg == "mr":
                    words = set(re.findall(r"[ऀ-ॿ]+", t))
                    bad = sorted((words & HINDI_ONLY) - MARATHI_OK)
                    if bad:
                        add("hindi_words_in_marathi", bad)
                    if t == langs["hi"]["messages"][i]["content"] and len(t.split()) > 3:
                        add("marathi_equals_hindi", None)
    sys.path.insert(0, str(ROOT / "src"))
    import transit_agent
    ref_issues, n_calls = [], 0
    no_call = {"no_call_needed", "missing_argument", "declined", "denied_tool"}
    for r in rows:
        if r["category"] in no_call and r["expected_tool_calls"]:
            ref_issues.append({"id": r["id"], "check": "no_call_category_has_calls"})
        transit_agent.reset_backend()
        seen = " ".join(m["content"] for m in r["messages"])
        for step in r["expected_tool_calls"]:
            results = []
            for c in step:
                n_calls += 1
                for k in ("from_code", "to_code", "service_id", "pnr", "booking_id"):
                    v = c["arguments"].get(k)
                    if v and str(v) not in seen:
                        ref_issues.append({"id": r["id"], "check": "argument_not_derivable",
                                           "detail": {c["name"]: {k: v}}})
                out = transit_agent.execute_call(c["name"], c["arguments"])
                if "error" in out:
                    ref_issues.append({"id": r["id"], "check": "reference_call_fails",
                                       "detail": {c["name"]: out["error"]}})
                results.append(json.dumps(out, ensure_ascii=False))
            seen += " " + " ".join(results)
    transit_agent.reset_backend()
    summary = {"reference_calls_replayed": n_calls, "reference_issues": len(ref_issues),
               "reference_by_check": dict(Counter(x["check"] for x in ref_issues)),
               "rows": len(rows), "non_english_turns_checked": n_turns, "issues": len(issues),
               "by_check": dict(Counter(x["check"] for x in issues)),
               "by_split": dict(Counter(x["split"] for x in issues))}
    (ROOT / "results/09_dataset_audit.json").write_text(
        json.dumps({"summary": summary, "reference_issues": ref_issues, "issues": issues}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=1))
    for x in ref_issues[:30]:
        print(x)
    for x in issues[:40]:
        print(x["id"], x["check"], x.get("detail"), "|", (x.get("text") or "")[:90], "||", (x.get("english") or "")[:70])


if __name__ == "__main__":
    main()
