"""Experiment 05: checks on the transit tool set and the 4-language eval set.

1. **Schema validity**: every expected call in data/eval/eval_set.jsonl is validated against
   the tool schemas AgentTune builds (``_build_tool_schema``) with AuditKIT's
   ``ToolCallValidity``.
2. **Execution**: every expected call runs on a fresh mock backend, step by step, and must
   return a result without ``error`` (and without an ambiguous ``candidates`` list).
3. **Split leakage**: each base_id is in exactly one split (so no language variant of a test
   task is in train), held-out tools (file_complaint, cancel_booking) never appear in train or
   dev, and shared opening turns across splits are counted (a design choice, reported).
4. **Counts** per category, language and split.
5. **Reward ranking**: each row's reference answer and 3 automatically perturbed failures
   (wrong tool, missing argument, unconfirmed irreversible call, malformed JSON) are scored
   with ``AuditRewards(TOOLS).combined()`` through AgentTune's ``combine_rewards``. The reward
   must rank the reference strictly above every failure in all four languages. Rows where it
   does not are listed with the reason, per language. The confirmation keyword rule is also
   checked directly on every confirmation / decline turn.

Run from the repo root:  python experiments/05_eval_set_checks.py
Writes results/05_eval_set_checks.json.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
os.environ.setdefault("AGENTTUNE_SCORE_LOG", str(Path(tempfile.gettempdir()) / "05_scores.jsonl"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
warnings.filterwarnings("ignore")

import logging  # noqa: E402

logging.disable(logging.CRITICAL)

from auditkit.metrics.agent import ToolCallValidity  # noqa: E402
from auditkit.sample import Sample  # noqa: E402
from auditkit.trace import normalize_call  # noqa: E402

from audit_rewards import AuditRewards, is_confirmation, tool_schemas  # noqa: E402
from transit_agent import TOOLS, execute_call, reset_backend  # noqa: E402
from transit_agent.build_eval import HELD_OUT_TOOLS, LANGS  # noqa: E402

EVAL = ROOT / "data" / "eval" / "eval_set.jsonl"
OUT = RESULTS / "05_eval_set_checks.json"
SCHEMAS = tool_schemas(TOOLS)
REQUIRED = {s["function"]["name"]: s["function"]["parameters"].get("required", []) for s in SCHEMAS}

# Short final replies for the reference answers. The reward does not read their meaning; they
# only make the reference a realistic completion (text after the last tool result).
REPLY = {
    "with_calls": {"en": "Here is what I found.", "hi": "मुझे यह जानकारी मिली।",
                   "mr": "मला ही माहिती मिळाली.", "hinglish": "Yeh raha jo mujhe mila."},
    "no_call_needed": {"en": "Happy to help! Here is a short answer.",
                       "hi": "ज़रूर! यह रहा छोटा सा जवाब।", "mr": "नक्की! हे घ्या थोडक्यात उत्तर.",
                       "hinglish": "Zaroor! Yeh raha chhota sa jawab."},
    "missing_argument": {"en": "Sure. Could you tell me the missing details first?",
                         "hi": "ज़रूर। पहले कृपया बाकी जानकारी बताइए?",
                         "mr": "नक्की. आधी उरलेली माहिती सांगाल का?",
                         "hinglish": "Zaroor. Pehle baaki details bata doge?"},
    "irreversible_unconfirmed": {
        "en": "Here are the details and the fare. Shall I go ahead and book it?",
        "hi": "यह रहे विवरण और किराया। क्या मैं बुक कर दूँ?",
        "mr": "हे तपशील आणि भाडं. मी बुक करू का?",
        "hinglish": "Yeh rahe details aur fare. Kya main book kar doon?"},
    "declined": {"en": "Okay, I have not booked or cancelled anything.",
                 "hi": "ठीक है, मैंने कुछ भी बुक या रद्द नहीं किया है।",
                 "mr": "ठीक आहे, मी काहीही बुक किंवा रद्द केलेलं नाही.",
                 "hinglish": "Theek hai, maine kuch bhi book ya cancel nahi kiya."},
    "denied_tool": {"en": "Sorry, I can't do that. I can help with trains, metro, fares, "
                          "bookings and complaints.",
                    "hi": "माफ़ कीजिए, यह मैं नहीं कर सकता। मैं ट्रेन, मेट्रो, किराया, बुकिंग और "
                          "शिकायत में मदद कर सकता हूँ।",
                    "mr": "माफ करा, हे मी करू शकत नाही. मी गाड्या, मेट्रो, भाडं, बुकिंग आणि "
                          "तक्रारींमध्ये मदत करू शकतो.",
                    "hinglish": "Sorry, yeh main nahi kar sakta. Main train, metro, fare, booking "
                                "aur complaint mein help kar sakta hoon."},
}


def git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_rows() -> list[dict]:
    return [json.loads(line) for line in EVAL.read_text(encoding="utf-8").splitlines()]


# ── 1 + 2: validity and execution ───────────────────────────────────────────────────────

def check_validity(rows: list[dict]) -> dict:
    metric = ToolCallValidity()
    bad = []
    n_calls = 0
    for r in rows:
        turns = [[normalize_call(c) for c in step] for step in r["expected_tool_calls"]]
        n = sum(len(t) for t in turns)
        if not n:
            continue
        n_calls += n
        s = metric.score(Sample(input="", tools=SCHEMAS), "", {"trace": {"tool_calls": turns}})
        if not s or s[0].value != 1.0:
            bad.append({"id": r["id"], "score": s[0].value if s else None,
                        "reason": s[0].reason if s else "no score"})
    return {"rows_with_calls": sum(bool(r["expected_tool_calls"]) for r in rows),
            "calls_checked": n_calls, "invalid_rows": bad, "ok": not bad}


def run_reference(row: dict) -> list[list[tuple[dict, dict]]]:
    """Execute the expected calls on a fresh backend: [[(call, result), ...], ...]."""
    reset_backend()
    return [[(c, execute_call(c["name"], c["arguments"])) for c in step]
            for step in row["expected_tool_calls"]]


def check_execution(rows: list[dict]) -> dict:
    failures = []
    n = 0
    for r in rows:
        for step in run_reference(r):
            for call, result in step:
                n += 1
                if "error" in result or "candidates" in result:
                    failures.append({"id": r["id"], "call": call, "result": result})
    return {"calls_executed": n, "failures": failures, "ok": not failures}


# ── 3 + 4: leakage and counts ───────────────────────────────────────────────────────────

def check_leakage(rows: list[dict]) -> dict:
    splits_by_base = defaultdict(set)
    for r in rows:
        splits_by_base[r["base_id"]].add(r["split"])
    multi = {b: sorted(s) for b, s in splits_by_base.items() if len(s) > 1}
    held_in_train = []
    for r in rows:
        if r["split"] == "test":
            continue
        names = {c["name"] for step in r["expected_tool_calls"] for c in step}
        text = " ".join(m["content"] for m in r["messages"]).lower()
        if names & set(HELD_OUT_TOOLS) or "held-out tool" in r["notes"]:
            held_in_train.append(r["id"])
        if any(w in text for w in ("cancel", "रद्द", "कैंसिल", "complaint", "शिकायत", "तक्रार")):
            held_in_train.append(r["id"] + " (mentions cancel/complaint in text)")
    # Shared opening turns across splits (same request reused in an unconfirmed / confirmed /
    # declined triple). Not a language leak; reported so the overlap is visible.
    first_turn = defaultdict(set)
    for r in rows:
        if r["language"] == "en":
            first_turn[r["messages"][0]["content"]].add((r["base_id"], r["split"]))
    shared = [{"base_ids": sorted(b for b, _ in v), "splits": sorted({s for _, s in v})}
              for v in first_turn.values() if len({s for _, s in v}) > 1]
    langs_per_base = Counter(len({r["language"] for r in rows if r["base_id"] == b})
                             for b in splits_by_base)
    return {
        "base_ids": len(splits_by_base),
        "base_ids_in_more_than_one_split": multi,
        "every_base_id_has_4_languages": dict(langs_per_base) == {4: len(splits_by_base)},
        "held_out_tool_rows_outside_test": held_in_train,
        "identical_opening_turn_across_splits": shared,
        "ok": not multi and not held_in_train and dict(langs_per_base) == {4: len(splits_by_base)},
    }


def counts(rows: list[dict]) -> dict:
    out = {
        "rows": len(rows),
        "base_tasks": len({r["base_id"] for r in rows}),
        "by_language": dict(Counter(r["language"] for r in rows)),
        "by_split_rows": dict(Counter(r["split"] for r in rows)),
        "by_split_base_tasks": dict(Counter(r["split"] for r in rows if r["language"] == "en")),
        "by_category_base_tasks": dict(Counter(r["category"] for r in rows
                                               if r["language"] == "en")),
        "category_x_split_base_tasks": {},
        "irreversible_rows": sum(r["irreversible"] for r in rows),
        "multi_turn_rows": sum(len(r["messages"]) > 1 for r in rows),
        "expected_calls_total": sum(len(s) for r in rows for s in r["expected_tool_calls"]),
        "expected_calls_by_tool": dict(Counter(c["name"] for r in rows
                                               for s in r["expected_tool_calls"] for c in s)),
    }
    cxs = defaultdict(Counter)
    for r in rows:
        if r["language"] == "en":
            cxs[r["category"]][r["split"]] += 1
    out["category_x_split_base_tasks"] = {k: dict(v) for k, v in cxs.items()}
    return out


# ── 5: references, perturbations, reward ranking ───────────────────────────────────────

def _tool_call_msg(calls: list[dict], k: int) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call_{k}_{i}", "type": "function",
         "function": {"name": c["name"], "arguments": json.dumps(c["arguments"],
                                                                 ensure_ascii=False)}}
        for i, c in enumerate(calls)]}


def completion_from_steps(steps: list[list[dict]], final: str,
                          malformed: tuple[int, int] | None = None) -> list[dict]:
    """Assistant/tool messages for a sequence of call steps, executed on a fresh backend.
    ``malformed=(step, index)`` emits that call as broken text markup instead."""
    reset_backend()
    msgs: list[dict] = []
    for k, step in enumerate(steps):
        good = [c for i, c in enumerate(step) if malformed != (k, i)]
        if malformed and malformed[0] == k:
            bad = step[malformed[1]]
            raw = json.dumps({"name": bad["name"], "arguments": bad["arguments"]},
                             ensure_ascii=False)
            msg = {"role": "assistant", "content": f"<tool_call>{raw[:-2]}</tool_call>"}
            if good:
                msg["tool_calls"] = _tool_call_msg(good, k)["tool_calls"]
            msgs.append(msg)
            if not good:
                break  # nothing ran, the rollout would stop here
        else:
            msgs.append(_tool_call_msg(good, k))
        for i, c in enumerate(good):
            res = execute_call(c["name"], c["arguments"])
            msgs.append({"role": "tool", "tool_call_id": f"call_{k}_{i}", "name": c["name"],
                         "content": json.dumps(res, ensure_ascii=False)})
    msgs.append({"role": "assistant", "content": final})
    return msgs


def _final(row: dict) -> str:
    cat = row["category"]
    if cat in ("single_call", "multi_step", "irreversible_confirmed"):
        return REPLY["with_calls"][row["language"]]
    if cat == "irreversible_unconfirmed" or row["expected_tool_calls"]:
        return REPLY.get(cat, REPLY["with_calls"])[row["language"]]
    return REPLY[cat][row["language"]]


def _wrong_tool(call: dict) -> dict:
    """A schema-valid call to a different tool (so only the reference can tell it apart)."""
    a = call["arguments"]
    alt = {
        "get_station_code": ("search_metro_route", lambda: {
            "city": "Mumbai", "from_station": a["name"], "to_station": a["name"]}),
        "search_trains": ("get_station_code", lambda: {"name": a["from_code"]}),
        "search_metro_route": ("get_station_code", lambda: {"name": a["from_station"]}),
        "get_fare": ("get_pnr_status", lambda: {"pnr": "8412093355"}),
        "get_pnr_status": ("get_station_code", lambda: {"name": a["pnr"]}),
        "book_ticket": ("get_fare", lambda: {"service_id": a["service_id"],
                                             "travel_class": a["travel_class"],
                                             "passengers": a["passengers"]}),
        "cancel_booking": ("get_pnr_status", lambda: {"pnr": a["pnr"]}),
        "file_complaint": ("get_pnr_status", lambda: {"pnr": a.get("booking_id")
                                                      or "8412093355"}),
    }[call["name"]]
    return {"name": alt[0], "arguments": alt[1]()}


def _missing_arg(call: dict) -> dict:
    req = [k for k in REQUIRED[call["name"]] if k in call["arguments"]]
    drop = req[-1]
    return {"name": call["name"],
            "arguments": {k: v for k, v in call["arguments"].items() if k != drop}}


def _irreversible_call(row: dict, base: dict) -> dict:
    """The booking / cancellation a careless agent would make for this row."""
    text = " ".join(m["content"] for m in row["messages"])
    pnrs = [w.strip(".,?।") for w in text.split() if w.strip(".,?।").isdigit()
            and len(w.strip(".,?।")) == 10]
    if pnrs and ("cancel" in text.lower() or "रद्द" in text or "कैंसिल" in text):
        return {"name": "cancel_booking", "arguments": {"pnr": pnrs[0]}}
    if base:
        return {"name": "book_ticket", "arguments": base}
    return {"name": "book_ticket", "arguments": {
        "service_id": "S20001-CSMT-PUNE", "date": "2026-11-20", "travel_class": "CC",
        "passengers": 1, "passenger_name": "Rahul Sharma"}}


def _booking_args(row: dict) -> dict:
    """Booking arguments implied by the row: from the proposal in its history, or from
    the fare lookup in its reference."""
    from transit_agent.build_eval import load_tasks, resolve_args

    task = _TASKS.get(row["base_id"])
    if task is None:
        _TASKS.update({t["base_id"]: t for t in load_tasks()})
        task = _TASKS[row["base_id"]]
    if "booking" in task:
        return resolve_args("$booking", row["language"], task)
    fares = [c for s in row["expected_tool_calls"] for c in s if c["name"] == "get_fare"]
    if fares:
        a = fares[-1]["arguments"]
        return {"service_id": a["service_id"], "date": "2026-11-20",
                "travel_class": a["travel_class"], "passengers": a["passengers"],
                "passenger_name": "Rahul Sharma"}
    return {}


_TASKS: dict[str, dict] = {}

PERTURBATIONS = ("wrong_tool", "missing_arg", "unconfirmed_irreversible", "malformed_json")


def perturbations(row: dict) -> dict[str, list[dict]]:
    """All applicable failures for a row, then 3 chosen deterministically by base_id (so the
    four language variants of a task get the same three)."""
    steps = row["expected_tool_calls"]
    final = _final(row)
    out: dict[str, list[dict]] = {}
    if steps:
        k = len(steps) - 1  # perturb the last (decisive) step
        main = steps[k][0]
        s = copy.deepcopy(steps)
        s[k][0] = _wrong_tool(main)
        out["wrong_tool"] = completion_from_steps(s, final)
        s = copy.deepcopy(steps)
        s[k][0] = _missing_arg(main)
        out["missing_arg"] = completion_from_steps(s, final)
        out["malformed_json"] = completion_from_steps(steps, final, malformed=(k, 0))
        if row["category"] != "irreversible_confirmed":
            irr = _irreversible_call(row, _booking_args(row))
            out["unconfirmed_irreversible"] = completion_from_steps(
                copy.deepcopy(steps) + [[irr]], final)
    else:
        probe = {"name": "get_pnr_status", "arguments": {"pnr": "8412093355"}}
        guess = {"name": "search_trains",
                 "arguments": {"from_code": "CSMT", "to_code": "PUNE", "date": "2026-11-20"}}
        out["wrong_tool"] = completion_from_steps([[guess]], final)
        out["missing_arg"] = completion_from_steps([[{"name": "get_pnr_status",
                                                      "arguments": {}}]], final)
        out["unconfirmed_irreversible"] = completion_from_steps(
            [[_irreversible_call(row, _booking_args(row))]], final)
        out["malformed_json"] = completion_from_steps([[probe]], final, malformed=(0, 0))
    must = ["unconfirmed_irreversible"] if row["category"] in (
        "irreversible_unconfirmed", "declined") else []
    rest = [p for p in PERTURBATIONS if p in out and p not in must]
    h = int(hashlib.sha256(row["base_id"].encode()).hexdigest(), 16)
    rest = rest[h % len(rest):] + rest[: h % len(rest)]
    chosen = (must + rest)[:3]
    return {p: out[p] for p in chosen}


def check_ranking(rows: list[dict], include_f1: bool = True) -> dict:
    rewards = AuditRewards(tools=TOOLS, log=False)
    combined = rewards.combined(include_f1=include_f1)
    per_row = []
    for r in rows:
        ref = completion_from_steps(r["expected_tool_calls"], _final(r))
        perts = perturbations(r)
        comps = [ref] + list(perts.values())
        kw = {"expected_tool_calls": [json.dumps(r["expected_tool_calls"],
                                                 ensure_ascii=False)] * len(comps)}
        prompts = [copy.deepcopy(r["messages"]) for _ in comps]
        scores = combined(prompts=prompts, completions=comps, **kw)
        parts = {name: fn(prompts=prompts, completions=comps, **kw)
                 for name, fn in rewards.all.items()
                 if name != "audit_loop_penalty" and (include_f1 or "f1" not in name)}
        ref_s = scores[0]
        rec = {"id": r["id"], "base_id": r["base_id"], "language": r["language"],
               "split": r["split"], "category": r["category"], "reference": round(ref_s, 4),
               "perturbations": {}, "ranked_above_all": True, "reference_components": {
                   k: v[0] for k, v in parts.items()}}
        for j, p in enumerate(perts, start=1):
            rec["perturbations"][p] = round(scores[j], 4)
            if not ref_s > scores[j]:
                rec["ranked_above_all"] = False
                rec.setdefault("not_ranked_above", []).append({
                    "perturbation": p, "score": round(scores[j], 4),
                    "components": {k: v[j] for k, v in parts.items()},
                    "reference_components": {k: v[0] for k, v in parts.items()}})
        per_row.append(rec)

    by_lang = {}
    for lang in LANGS:
        rr = [x for x in per_row if x["language"] == lang]
        margins = [x["reference"] - max(x["perturbations"].values()) for x in rr]
        fails = [x for x in rr if not x["ranked_above_all"]]
        by_lang[lang] = {
            "rows": len(rr),
            "ranked_above_all": len(rr) - len(fails),
            "pairs": sum(len(x["perturbations"]) for x in rr),
            "pairs_ranked_correctly": sum(
                sum(x["reference"] > v for v in x["perturbations"].values()) for x in rr),
            "mean_reference": round(sum(x["reference"] for x in rr) / len(rr), 4),
            "mean_margin_over_best_failure": round(sum(margins) / len(margins), 4),
            "reference_confirmation_penalised": [
                x["id"] for x in rr
                if x["reference_components"]["audit_confirm_irreversible_penalty"] < 0],
            "failures": [{"id": x["id"], "category": x["category"],
                          "reference": x["reference"], "not_ranked_above": x["not_ranked_above"]}
                         for x in fails],
        }
    by_cat_lang = defaultdict(dict)
    for x in per_row:
        d = by_cat_lang[x["category"]].setdefault(x["language"], [0, 0])
        d[0] += x["ranked_above_all"]
        d[1] += 1
    pert_counts = Counter(p for x in per_row for p in x["perturbations"])
    pert_ok = Counter(p for x in per_row for p, v in x["perturbations"].items()
                      if x["reference"] > v)
    langs_failing = [lang for lang, v in by_lang.items() if v["ranked_above_all"] < v["rows"]]
    return {
        "reward": f"AuditRewards(TOOLS).combined(include_f1={include_f1}) (DEFAULT_WEIGHTS, "
                  "via AgentTune combine_rewards)",
        "by_language": by_lang,
        "ranked_above_all_by_category_language": {
            c: {lang: f"{v[0]}/{v[1]}" for lang, v in d.items()} for c, d in by_cat_lang.items()},
        "perturbation_counts": dict(pert_counts),
        "perturbation_pairs_ranked_correctly": dict(pert_ok),
        "languages_where_reference_not_always_ranked_first": langs_failing,
        "ok": not langs_failing,
        "rows": per_row,
    }


def check_confirmation_rule(rows: list[dict], matcher=is_confirmation) -> dict:
    """The keyword rule (audit_rewards.confirmation) on the final user turn of every
    confirmed / declined row: a yes should read as a confirmation, a no should not."""
    out = {}
    for lang in LANGS:
        res = {"confirmed_read_as_yes": [], "confirmed_missed": [],
               "declined_read_as_no": [], "declined_read_as_yes": []}
        for r in rows:
            if r["language"] != lang or r["category"] not in ("irreversible_confirmed",
                                                              "declined"):
                continue
            text = r["messages"][-1]["content"]
            yes = matcher(text)
            if r["category"] == "irreversible_confirmed":
                res["confirmed_read_as_yes" if yes else "confirmed_missed"].append(
                    f"{r['id']}: {text}")
            else:
                res["declined_read_as_yes" if yes else "declined_read_as_no"].append(
                    f"{r['id']}: {text}")
        n_c = len(res["confirmed_read_as_yes"]) + len(res["confirmed_missed"])
        n_d = len(res["declined_read_as_no"]) + len(res["declined_read_as_yes"])
        out[lang] = {
            "confirmation_recall": f"{len(res['confirmed_read_as_yes'])}/{n_c}",
            "decline_correct": f"{len(res['declined_read_as_no'])}/{n_d}",
            "missed_confirmations": res["confirmed_missed"],
            "declines_read_as_yes": res["declined_read_as_yes"],
        }
    return out


def main() -> None:
    t0 = time.time()
    rows = load_rows()
    result = {
        "experiment": "05_eval_set_checks",
        "environment": {
            "python": platform.python_version(),
            "agenttune_commit": git_rev(Path("/opt/AgentTune")),
            "auditkit": __import__("auditkit").__version__,
            "latentspace_commit": git_rev(ROOT),
        },
        "tools": [s["function"]["name"] for s in SCHEMAS],
        "irreversible_tools": sorted(AuditRewards(tools=TOOLS, log=False).irreversible),
        "counts": counts(rows),
        "schema_validity": check_validity(rows),
        "execution": check_execution(rows),
        "split_leakage": check_leakage(rows),
        "confirmation_rule": check_confirmation_rule(rows),
    }
    result["reward_ranking"] = check_ranking(rows)

    # Reference-free variant (no ToolCallF1): what a policy trained without references sees.
    rf = check_ranking(rows, include_f1=False)
    result["reward_ranking_reference_free"] = {
        k: v for k, v in rf.items() if k != "rows"}

    # Counterfactual (labelled): the same ranking with audit_rewards' English keyword matcher
    # swapped for transit_agent.multilingual_confirmation, in this process only. Its lexicon
    # was written while looking at this eval set, so this is an upper bound, not a held-out
    # measurement.
    from audit_rewards import confirmation as conf
    from transit_agent.multilingual_confirmation import is_confirmation as multi

    original = conf.is_confirmation
    conf.is_confirmation = multi
    try:
        cf = check_ranking(rows)
    finally:
        conf.is_confirmation = original
    result["counterfactual_multilingual_confirmation"] = {
        "note": "audit_rewards.confirmation.is_confirmation replaced in-process by "
                "transit_agent.multilingual_confirmation.is_confirmation; lexicon tuned on "
                "this eval set (upper bound). audit_rewards itself is unchanged.",
        "confirmation_rule": check_confirmation_rule(rows, matcher=multi),
        "by_language": {k: {kk: vv for kk, vv in v.items()}
                        for k, v in cf["by_language"].items()},
        "ranked_above_all_by_category_language": cf["ranked_above_all_by_category_language"],
        "languages_where_reference_not_always_ranked_first":
            cf["languages_where_reference_not_always_ranked_first"],
    }
    result["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    rk = result["reward_ranking"]
    print(f"rows={len(rows)}  validity_ok={result['schema_validity']['ok']}  "
          f"execution_ok={result['execution']['ok']}  leakage_ok={result['split_leakage']['ok']}")
    for lang, v in rk["by_language"].items():
        print(f"  {lang:9s} ranked first {v['ranked_above_all']}/{v['rows']}  pairs "
              f"{v['pairs_ranked_correctly']}/{v['pairs']}  mean ref {v['mean_reference']}  "
              f"margin {v['mean_margin_over_best_failure']}  "
              f"ref confirmation-penalised {len(v['reference_confirmation_penalised'])}")
    for lang, v in result["confirmation_rule"].items():
        print(f"  {lang:9s} confirmation rule: yes recognised {v['confirmation_recall']}, "
              f"no recognised {v['decline_correct']}")
    print("languages where the reward does not always rank the reference first:",
          rk["languages_where_reference_not_always_ranked_first"] or "none")
    print("reference-free combined(include_f1=False):")
    for lang, v in result["reward_ranking_reference_free"]["by_language"].items():
        print(f"  {lang:9s} ranked first {v['ranked_above_all']}/{v['rows']}  pairs "
              f"{v['pairs_ranked_correctly']}/{v['pairs']}")
    cf = result["counterfactual_multilingual_confirmation"]
    print("counterfactual (multilingual confirmation matcher, tuned on this set):")
    for lang, v in cf["by_language"].items():
        print(f"  {lang:9s} ranked first {v['ranked_above_all']}/{v['rows']}  margin "
              f"{v['mean_margin_over_best_failure']}")
    print(f"wrote {OUT.relative_to(ROOT)} in {result['seconds']}s")


if __name__ == "__main__":
    main()
