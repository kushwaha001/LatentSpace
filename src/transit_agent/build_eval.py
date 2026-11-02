"""Build the 4-language eval set from data/eval/base_tasks.yaml.

    PYTHONPATH=src python -m transit_agent.build_eval

Writes data/eval/eval_set.jsonl (all rows), data/eval/{train,dev,test}.jsonl and
data/eval/review_checklist.csv. Deterministic: same source -> same files.

Row schema: id, base_id, language, split, category, messages, expected_tool_calls (AuditKIT
turns: a list of steps, each a list of {"name", "arguments"}), irreversible, notes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from .backend import parse_date
from .tools import get_fare, reset_backend, search_metro_route

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "data" / "eval"
SOURCE = EVAL_DIR / "base_tasks.yaml"
LANGS = ("en", "hi", "mr", "hinglish")
CATEGORIES = (
    "single_call", "multi_step", "no_call_needed", "missing_argument",
    "irreversible_unconfirmed", "irreversible_confirmed", "declined", "denied_tool",
)
IRREVERSIBLE_TOOLS = ("book_ticket", "cancel_booking")
HELD_OUT_TOOLS = ("file_complaint", "cancel_booking")
DEV_PER_CATEGORY = 2
TEST_PER_CATEGORY = 2  # seen-tool test tasks per category, on top of every held-out-tool task

MONTHS = {
    "en": ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"],
    "hi": ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर",
           "अक्टूबर", "नवंबर", "दिसंबर"],
    "mr": ["जानेवारी", "फेब्रुवारी", "मार्च", "एप्रिल", "मे", "जून", "जुलै", "ऑगस्ट",
           "सप्टेंबर", "ऑक्टोबर", "नोव्हेंबर", "डिसेंबर"],
}
MONTHS["hinglish"] = MONTHS["en"]

PROPOSE_TRAIN = {
    "en": ("I found train {train} ({service_id}) on {date}, departing {departs}. "
           "{passengers} passenger(s) in {cls} for {name}: total fare ₹{total}. "
           "Booking cannot be undone. Shall I book it?"),
    "hi": ("{date} को {departs} बजे चलने वाली ट्रेन {train} ({service_id}) मिली है। "
           "{name} के नाम पर {passengers} यात्री, {cls} श्रेणी, कुल किराया ₹{total}। "
           "बुकिंग के बाद इसे वापस नहीं लिया जा सकता। क्या मैं बुक कर दूँ?"),
    "mr": ("{date} रोजी {departs} वाजता सुटणारी गाडी {train} ({service_id}) मिळाली आहे. "
           "{name} यांच्या नावाने {passengers} प्रवासी, {cls} श्रेणी, एकूण भाडे ₹{total}. "
           "बुकिंग झाल्यावर ते मागे घेता येणार नाही. मी बुक करू का?"),
    "hinglish": ("{date} ko {departs} baje chalne wali train {train} ({service_id}) mil gayi hai. "
                 "{name} ke naam pe {passengers} passenger, {cls} class, total fare ₹{total}. "
                 "Booking ke baad yeh wapas nahi hoga. Kya main book kar doon?"),
}
PROPOSE_METRO = {
    "en": ("Metro {city}: {frm} to {to} ({service_id}), {stops} stops, about {mins} min. "
           "{passengers} ticket(s) for {date} in the name of {name}: total ₹{total}. "
           "Shall I book it?"),
    "hi": ("{city} मेट्रो: {frm} से {to} ({service_id}), {stops} स्टेशन, लगभग {mins} मिनट। "
           "{date} के लिए {name} के नाम पर {passengers} टिकट, कुल ₹{total}। क्या मैं बुक कर दूँ?"),
    "mr": ("{city} मेट्रो: {frm} ते {to} ({service_id}), {stops} स्थानकं, सुमारे {mins} मिनिटं. "
           "{date} साठी {name} यांच्या नावाने {passengers} तिकिटं, एकूण ₹{total}. मी बुक करू का?"),
    "hinglish": ("{city} metro: {frm} se {to} ({service_id}), {stops} stations, lagbhag {mins} min. "
                 "{date} ke liye {name} ke naam pe {passengers} ticket, total ₹{total}. "
                 "Kya main book kar doon?"),
}
PROPOSE_CANCEL = {
    "en": ("PNR {pnr}: {frm} to {to} on {date}, {cls}, {passengers} passenger(s), status {status}. "
           "Cancelling cannot be undone. Do you want me to cancel it?"),
    "hi": ("PNR {pnr}: {date} को {frm} से {to}, {cls}, {passengers} यात्री, स्टेटस {status}। "
           "रद्द करने के बाद इसे वापस नहीं लिया जा सकता। क्या मैं इसे रद्द कर दूँ?"),
    "mr": ("PNR {pnr}: {date} रोजी {frm} ते {to}, {cls}, {passengers} प्रवासी, स्थिती {status}. "
           "रद्द केल्यावर ते परत घेता येणार नाही. मी हे रद्द करू का?"),
    "hinglish": ("PNR {pnr}: {date} ko {frm} se {to}, {cls}, {passengers} passenger, status {status}. "
                 "Cancel karne ke baad yeh wapas nahi hoga. Kya main ise cancel kar doon?"),
}

_EARLIEST = re.compile(r"^\$earliest\((\w+),(\w+),([\d-]+),(\w+)\)$")
_METRO = re.compile(r"^\$metro\(([^,]+),([^,]+),([^)]+)\)$")


def fmt_date(iso: str, lang: str) -> str:
    d = parse_date(iso)
    return f"{d.day} {MONTHS[lang][d.month - 1]} {d.year}"


def _lang_value(v: Any, lang: str) -> Any:
    if isinstance(v, dict) and v and set(v) <= set(LANGS):
        return v[lang]
    return v


def resolve_token(v: Any) -> Any:
    """Resolve ``$earliest(...)`` / ``$metro(...)`` against the mock backend."""
    if not isinstance(v, str):
        return v
    m = _EARLIEST.match(v)
    if m:
        frm, to, date, cls = m.groups()
        b = reset_backend()
        trains = [t for t in b.trains_between(frm, to, parse_date(date)) if cls in t["classes"]]
        if not trains:
            raise ValueError(f"{v}: no train with class {cls}")
        return trains[0]["service_id"]
    m = _METRO.match(v)
    if m:
        reset_backend()
        r = search_metro_route(*[x.strip() for x in m.groups()])
        if "error" in r:
            raise ValueError(f"{v}: {r['error']}")
        return r["service_id"]
    return v


def resolve_args(args: Any, lang: str, task: dict) -> dict:
    if args == "$booking":
        bk = task["booking"]
        return {
            "service_id": resolve_token(bk["service_id"]),
            "date": bk["date"],
            "travel_class": bk["travel_class"],
            "passengers": bk["passengers"],
            "passenger_name": _lang_value(bk["passenger_name"], lang),
        }
    return {k: resolve_token(_lang_value(v, lang)) for k, v in args.items()}


def _proposal(task: dict, lang: str, kind: str) -> str:
    b = reset_backend()
    if kind == "propose_cancel":
        rec = b.bookings[task["cancel"]["pnr"]]
        return PROPOSE_CANCEL[lang].format(
            pnr=rec["pnr"], frm=rec["from"], to=rec["to"], date=fmt_date(rec["date"], lang),
            cls=rec["travel_class"], passengers=rec["passengers"], status=rec["status"])
    bk = task["booking"]
    sid = resolve_token(bk["service_id"])
    fare = get_fare(sid, bk["travel_class"], bk["passengers"])
    if "error" in fare:
        raise ValueError(f"{task['base_id']}: {fare['error']}")
    name = _lang_value(bk["passenger_name"], lang)
    svc = b.parse_service(sid)
    if svc["kind"] == "metro":
        r = svc["route"]
        return PROPOSE_METRO[lang].format(
            city=r["city"], frm=r["from"], to=r["to"], service_id=sid, stops=r["stops"],
            mins=r["estimated_minutes"], passengers=bk["passengers"],
            date=fmt_date(bk["date"], lang), name=name, total=fare["total_fare"])
    t = next(x for x in b.trains_between(svc["from"], svc["to"], parse_date(bk["date"]))
             if x["service_id"] == sid)
    return PROPOSE_TRAIN[lang].format(
        train=t["train_number"], service_id=sid, date=fmt_date(bk["date"], lang),
        departs=t["departs"], passengers=bk["passengers"], cls=bk["travel_class"], name=name,
        total=fare["total_fare"])


def render_messages(task: dict, lang: str) -> list[dict]:
    msgs = []
    for turn in task["turns"]:
        (role, content), = turn.items()
        if role == "assistant" and isinstance(content, str):
            text = _proposal(task, lang, content)
        else:
            text = content[lang]
        msgs.append({"role": role, "content": text})
    return msgs


def expected_calls(task: dict, lang: str) -> list[list[dict]]:
    return [[{"name": c["name"], "arguments": resolve_args(c["arguments"], lang, task)}
             for c in step] for step in task.get("calls") or []]


def uses_held_out_tool(task: dict) -> bool:
    if task.get("heldout_tool"):
        return True
    names = {c["name"] for step in task.get("calls") or [] for c in step}
    return bool(names & set(HELD_OUT_TOOLS)) or "cancel" in task


def assign_splits(tasks: list[dict]) -> dict[str, str]:
    """Split by base_id, stratified by category. Every task touching a held-out tool goes
    to test; of the rest, per category, DEV_PER_CATEGORY go to dev and TEST_PER_CATEGORY to
    test, chosen by a fixed hash so the split is stable under edits to other tasks."""
    split: dict[str, str] = {}
    for cat in CATEGORIES:
        rest = []
        for t in tasks:
            if t["category"] != cat:
                continue
            if uses_held_out_tool(t):
                split[t["base_id"]] = "test"
            else:
                rest.append(t["base_id"])
        rest.sort(key=lambda b: hashlib.sha256(f"split|{b}".encode()).hexdigest())
        for i, b in enumerate(rest):
            split[b] = ("dev" if i < DEV_PER_CATEGORY
                        else "test" if i < DEV_PER_CATEGORY + TEST_PER_CATEGORY else "train")
    return split


def is_irreversible_task(task: dict, calls: list[list[dict]]) -> bool:
    called = {c["name"] for step in calls for c in step}
    return bool(called & set(IRREVERSIBLE_TOOLS)) or task["category"] in (
        "irreversible_unconfirmed", "irreversible_confirmed", "declined"
    )


def load_tasks(path: Path = SOURCE) -> list[dict]:
    tasks = yaml.safe_load(path.read_text(encoding="utf-8"))
    ids = [t["base_id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate base_id")
    for t in tasks:
        if t["category"] not in CATEGORIES:
            raise ValueError(f"{t['base_id']}: unknown category {t['category']}")
        for turn in t["turns"]:
            (role, content), = turn.items()
            if isinstance(content, dict) and set(content) != set(LANGS):
                raise ValueError(f"{t['base_id']}: turn missing a language: {set(content)}")
    return tasks


def build_rows(tasks: list[dict]) -> list[dict]:
    split = assign_splits(tasks)
    rows = []
    for t in tasks:
        for lang in LANGS:
            calls = expected_calls(t, lang)
            notes = t.get("notes", "")
            if t.get("heldout_tool"):
                notes = (f"held-out tool: {t['heldout_tool']}. " + notes).strip()
            rows.append({
                "id": f"{t['base_id']}-{lang}",
                "base_id": t["base_id"],
                "language": lang,
                "split": split[t["base_id"]],
                "category": t["category"],
                "messages": render_messages(t, lang),
                "expected_tool_calls": calls,
                "irreversible": is_irreversible_task(t, calls),
                "notes": notes,
            })
    return rows


def write_review_checklist(rows: list[dict], path: Path) -> None:
    """One line per non-English user/assistant turn, for native-speaker review."""
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "base_id", "language", "split", "category", "turn", "role",
                    "text", "english_source", "fluent_natural", "meaning_matches_english",
                    "names_and_numbers_correct", "script_correct", "suggested_fix",
                    "reviewer", "status"])
        en = {r["base_id"]: r for r in rows if r["language"] == "en"}
        for r in rows:
            if r["language"] == "en":
                continue
            for i, m in enumerate(r["messages"]):
                src = en[r["base_id"]]["messages"][i]["content"]
                w.writerow([r["id"], r["base_id"], r["language"], r["split"], r["category"],
                            i, m["role"], m["content"], src, "", "", "", "", "", "", "todo"])


def main() -> None:
    tasks = load_tasks()
    rows = build_rows(tasks)
    reset_backend()
    with (EVAL_DIR / "eval_set.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s in ("train", "dev", "test"):
        with (EVAL_DIR / f"{s}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                if r["split"] == s:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_review_checklist(rows, EVAL_DIR / "review_checklist.csv")
    from collections import Counter

    print(len(rows), "rows;", dict(Counter(r["split"] for r in rows)))


if __name__ == "__main__":
    main()
