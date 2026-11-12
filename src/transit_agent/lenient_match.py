"""Lenient argument matching for scoring held-out rollouts (evaluation only, never a reward).

AuditKIT's ``ToolCallF1`` counts a predicted call as correct only when every argument value is
JSON-equal to the reference. On this eval set that rejects calls the backend treats as the same
request: a station name in another script ("पुणे" for "Pune"), another case ("PUNE"), or a
complaint description in other words. ``lenient_arg_match`` is a drop-in ``arg_match`` for
``auditkit.trace.max_matching`` (and ``ToolCallF1(arg_match=...)``) that accepts those and
nothing else.

The rules, fixed before re-scoring, per argument of the reference call:

1. Station and metro names (``get_station_code.name/city``, ``search_metro_route``): equal if
   the backend resolves both to the same non-empty set of stations (and the same metro city);
   otherwise equal after ``normalize_name`` (case, punctuation, "Jn"/"station" words).
2. Station codes, service ids, travel class, complaint category: equal after trimming and
   case-folding.
3. PNRs and booking ids: equal digit strings ("8412 093 355" == "8412093355").
4. Dates: the same calendar date (``YYYY-MM-DD`` only, as the tools accept).
5. Passenger counts: the same integer (2 == "2").
6. Passenger name: equal after ``normalize_name``.
7. Free text (``file_complaint.description``, ``cancel_booking.reason``): not compared. A
   required free-text argument must still be a non-empty string.

As in AuditKIT's exact mode, every reference argument must be present in the prediction, and
an extra predicted argument is allowed only if the tool declares it. Tool names must be equal.
Anything else falls back to exact equality, so a wrong date, PNR, station, class, count or
category is still a mismatch.

``lenient_score`` scores a whole rollout with that matcher and one task-level rule:

8. In ``irreversible_unconfirmed`` tasks, read-only lookups the reference does not list are
   ignored. The data card defines correct behaviour there as "propose, show the fare, ask;
   lookups allowed, no book_ticket / cancel_booking", but a reference holds one route only.
   An irreversible call is never ignored, and schema validity is checked separately.
"""

from __future__ import annotations

import inspect
from typing import Any

from .backend import normalize_name, parse_date
from .tools import get_backend
from . import tools as _tools

FREE_TEXT = {("file_complaint", "description"), ("cancel_booking", "reason")}
REQUIRED_FREE_TEXT = {("file_complaint", "description")}
CASE_FOLD = {"from_code", "to_code", "service_id", "travel_class", "category"}
DIGITS = {"pnr", "booking_id"}
READ_ONLY = {"get_station_code", "search_trains", "search_metro_route", "get_fare",
             "get_pnr_status"}
LOOKUPS_ALLOWED = {"irreversible_unconfirmed"}


def _declared(name: str) -> set[str]:
    fn = getattr(_tools, name, None)
    return set(inspect.signature(fn).parameters) if callable(fn) else set()


def _fold(v: Any) -> Any:
    return v.strip().casefold() if isinstance(v, str) else v


def _digits(v: Any) -> str:
    return "".join(ch for ch in str(v) if ch.isdigit())


def _int(v: Any) -> int | None:
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else None


def _rail_codes(name: Any, city: Any = None) -> frozenset[str]:
    if not isinstance(name, str) or not name.strip():
        return frozenset()
    city = city if isinstance(city, str) else None
    return frozenset(s["code"] for s in get_backend().find_rail_stations(name, city))


def _metro_ids(city: str | None, name: Any) -> frozenset[str]:
    if city is None or not isinstance(name, str) or not name.strip():
        return frozenset()
    return frozenset(s["id"] for s in get_backend().find_metro_station(city, name))


def _same_name(a: Any, b: Any) -> bool:
    return isinstance(a, str) and isinstance(b, str) and normalize_name(a) == normalize_name(b)


def _value_match(tool: str, key: str, p: Any, r: Any) -> bool:
    if p == r:
        return True
    if (tool, key) in FREE_TEXT:
        return (not (tool, key) in REQUIRED_FREE_TEXT) or (isinstance(p, str) and bool(p.strip()))
    if key in DIGITS:
        return bool(_digits(r)) and _digits(p) == _digits(r)
    if key == "date":
        return parse_date(p) is not None and parse_date(p) == parse_date(r)
    if key == "passengers":
        return _int(p) is not None and _int(p) == _int(r)
    if key == "passenger_name":
        return _same_name(p, r)
    if key in CASE_FOLD:
        return _fold(p) == _fold(r)
    return False


def lenient_arg_match(name: str, pred: Any, ref: Any) -> bool:
    """``(tool_name, pred_args, ref_args) -> bool`` for ``max_matching``. See module doc."""
    if not isinstance(pred, dict) or not isinstance(ref, dict):
        return pred == ref
    declared = _declared(name)
    if any(k not in ref and k not in declared for k in pred):
        return False
    if any(k not in pred for k in ref):
        return False
    if (name, "description") in REQUIRED_FREE_TEXT and "description" in pred:
        if not (isinstance(pred["description"], str) and pred["description"].strip()):
            return False

    if name == "get_station_code":
        pc, rc = _rail_codes(pred.get("name"), pred.get("city")), _rail_codes(ref.get("name"), ref.get("city"))
        if rc and pc == rc:
            return True
        if not _same_name(pred.get("name"), ref.get("name")):
            return False
        return "city" not in ref or _same_name(pred.get("city"), ref.get("city"))

    if name == "search_metro_route":
        b = get_backend()
        pcity, rcity = b.metro_city(str(pred.get("city", ""))), b.metro_city(str(ref.get("city", "")))
        if rcity is None or pcity != rcity:
            return pred == ref
        for key in ("from_station", "to_station"):
            pi, ri = _metro_ids(pcity, pred.get(key)), _metro_ids(rcity, ref.get(key))
            if not ((ri and pi == ri) or _same_name(pred.get(key), ref.get(key))):
                return False
        return True

    return all(_value_match(name, k, pred[k], ref[k]) for k in ref)


def lenient_score(calls: list[dict], expected: Any, category: str | None = None) -> dict:
    """``{"lenient_f1", "lenient_exact"}`` for one rollout.

    ``calls``: predicted ``{"name", "arguments"}`` dicts (parsed calls only).
    ``expected``: the reference turns (list of lists of calls), flat list, or its JSON text."""
    import json

    from auditkit.trace import ToolCall, max_matching

    if isinstance(expected, str):
        expected = json.loads(expected) if expected.strip() else []
    flat = [c for t in (expected or []) for c in (t if isinstance(t, list) else [t])]
    refs = [ToolCall(c["name"], c.get("arguments") or {}) for c in flat]
    preds = [ToolCall(c["name"], c["arguments"]) for c in calls or []
             if isinstance(c.get("arguments"), dict)]
    matching = max_matching(preds, refs, "exact", lenient_arg_match)
    if category in LOOKUPS_ALLOWED:
        used = set(matching.values())
        preds = [p for i, p in enumerate(preds) if i in used or p.name not in READ_ONLY]
        matching = max_matching(preds, refs, "exact", lenient_arg_match)
    m = len(matching)
    if not refs:
        f1 = 1.0 if not preds else 0.0
    else:
        prec = m / len(preds) if preds else 0.0
        rec = m / len(refs)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"lenient_f1": f1, "lenient_exact": 1.0 if m == len(refs) == len(preds) else 0.0}
