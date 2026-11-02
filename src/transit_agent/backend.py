"""Deterministic in-repo mock backend for the transit tools (data/backend/*.json).

Station names and Indian Railways codes are real; timetables, fares, availability and PNRs
are synthetic (data/backend/DATACARD.md). The backend keeps a small amount of state (new
bookings, cancellations, complaints) per instance, so every eval row can start from a fresh
``MockBackend()`` and get the same answer.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import heapq
import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "backend"

RAIL_CLASSES = ("2S", "SL", "CC", "3A", "2A", "EC", "1A")
METRO_CLASS = "standard"
COMPLAINT_CATEGORIES = (
    "cleanliness", "punctuality", "staff_behaviour", "refund", "safety", "catering",
    "accessibility", "other",
)
MAX_PASSENGERS = 6

# Words dropped before matching a station name: "Pune Junction", "पुणे स्टेशन", "Andheri metro".
_STOPWORDS = {
    "junction", "jn", "jct", "station", "railway", "rly", "terminus", "metro", "stn",
    "जंक्शन", "स्टेशन", "रेलवे", "रेल्वे", "मेट्रो", "स्थानक", "टर्मिनस",
}
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_NUKTA = "़"


def normalize_name(text: str) -> str:
    """Case-fold, strip punctuation, nukta and stop-words, and map Devanagari digits, so
    "Pune Jn.", "PUNE junction", "पुणे स्टेशन" and "पुणे" all compare equal."""
    t = unicodedata.normalize("NFC", text or "").translate(_DEV_DIGITS).replace(_NUKTA, "")
    t = unicodedata.normalize("NFC", t).casefold()
    # Python's \w excludes Devanagari vowel signs and virama, so keep the whole block.
    t = re.sub(r"[^\w\s\u0900-\u0963\u0966-\u097F]", " ", t)
    words = [w for w in t.split() if w not in _STOPWORDS]
    return " ".join(words)


def _load(name: str) -> Any:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _static() -> dict[str, Any]:
    rail = _load("rail_stations.json")
    metro = _load("metro.json")
    trains = _load("trains.json")
    return {
        "rail": rail,
        "rail_by_code": {s["code"]: s for s in rail["stations"]},
        "metro": metro,
        "trains": {t["number"]: t for t in trains},
        "bookings": {b["pnr"]: b for b in _load("bookings.json")},
        "fares": _load("fares.json"),
        "rail_alias": _rail_alias_index(rail),
        "metro_alias": {city: _metro_alias_index(m) for city, m in metro.items()},
        "city_alias": _city_alias_index(rail, metro),
    }


def _rail_alias_index(rail: dict) -> dict[str, set[str]]:
    idx: dict[str, set[str]] = {}
    for s in rail["stations"]:
        names = [s["code"], s["name"], s["name"].split(" (")[0], s["hi"], s["mr"], *s["aliases"]]
        for n in names:
            idx.setdefault(normalize_name(n), set()).add(s["code"])
    for name, codes in rail["ambiguous"].items():
        idx.setdefault(normalize_name(name), set()).update(codes)
    return idx


def _metro_alias_index(city: dict) -> dict[str, set[str]]:
    idx: dict[str, set[str]] = {}
    for s in city["stations"]:
        for n in [s["name"], s["hi"], s["mr"], *s["aliases"]]:
            idx.setdefault(normalize_name(n), set()).add(s["id"])
    return idx


_CITY_EXTRA = {
    "Mumbai": ["mumbai", "bombay", "मुंबई", "बंबई"],
    "Pune": ["pune", "poona", "पुणे"],
    "Delhi": ["delhi", "new delhi", "dilli", "दिल्ली", "नई दिल्ली", "नवी दिल्ली"],
}


def _city_alias_index(rail: dict, metro: dict) -> dict[str, str]:
    idx = {normalize_name(s["city"]): s["city"] for s in rail["stations"]}
    for city, names in _CITY_EXTRA.items():
        for n in names:
            idx[normalize_name(n)] = city
    for city in metro:
        idx[normalize_name(city)] = city
    return idx


def _digest(*parts: Any, n: int = 8) -> str:
    h = int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16)
    return f"{h % 10**n:0{n}d}"


class MockBackend:
    """Read-only static data plus per-instance mutable state."""

    def __init__(self) -> None:
        s = _static()
        self.s = s
        self.bookings: dict[str, dict] = {k: dict(v) for k, v in s["bookings"].items()}
        self.complaints: list[dict] = []

    # ── stations ─────────────────────────────────────────────────────────────

    def resolve_city(self, city: str | None) -> str | None:
        if not city:
            return None
        return self.s["city_alias"].get(normalize_name(city))

    def find_rail_stations(self, name: str, city: str | None = None) -> list[dict]:
        key = normalize_name(name)
        idx = self.s["rail_alias"]
        codes = set(idx.get(key, set()))
        if not codes and key:  # prefix / containment match on any alias
            for alias, cs in idx.items():
                if len(key) >= 3 and (alias.startswith(key) or key in alias.split()):
                    codes |= cs
        stations = [self.s["rail_by_code"][c] for c in sorted(codes)]
        want = self.resolve_city(city)
        if want:
            in_city = [st for st in stations if st["city"] == want]
            stations = in_city or stations
        return stations

    def metro_city(self, city: str) -> str | None:
        c = self.resolve_city(city)
        return c if c in self.s["metro"] else None

    def find_metro_station(self, city: str, name: str) -> list[dict]:
        m = self.s["metro"][city]
        key = normalize_name(name)
        idx = self.s["metro_alias"][city]
        ids = set(idx.get(key, set()))
        if not ids and len(key) >= 3:
            for alias, sids in idx.items():
                if alias.startswith(key) or key in alias.split():
                    ids |= sids
        by_id = {s["id"]: s for s in m["stations"]}
        return [by_id[i] for i in sorted(ids)]

    # ── trains ───────────────────────────────────────────────────────────────

    def trains_between(self, frm: str, to: str, date: dt.date) -> list[dict]:
        out = []
        for t in self.s["trains"].values():
            codes = [st["code"] for st in t["stops"]]
            if frm in codes and to in codes and codes.index(frm) < codes.index(to):
                a, b = t["stops"][codes.index(frm)], t["stops"][codes.index(to)]
                start_offset = a["day_offset"]
                run_day = (date - dt.timedelta(days=start_offset)).strftime("%a")
                if run_day not in t["runs_on"]:
                    continue
                mins = b["minute_of_journey"] - a["minute_of_journey"]
                out.append({
                    "service_id": f"{t['number']}-{frm}-{to}",
                    "train_number": t["number"],
                    "train_name": t["name"],
                    "departs": a["time"],
                    "arrives": b["time"],
                    "arrives_day_offset": b["day_offset"] - a["day_offset"],
                    "duration": f"{mins // 60}h{mins % 60:02d}m",
                    "classes": t["classes"],
                    "availability": {c: self._availability(t["number"], c, date)
                                     for c in t["classes"]},
                })
        return sorted(out, key=lambda r: r["departs"])

    def _availability(self, number: str, cls: str, date: dt.date) -> str:
        v = int(_digest(number, cls, date.isoformat(), n=3))
        if v < 600:
            return f"AVAILABLE-{v % 120 + 1}"
        if v < 800:
            return f"RAC-{v % 20 + 1}"
        return f"WL-{v % 60 + 1}"

    def parse_service(self, service_id: str) -> dict | None:
        """``S20000-CSMT-PUNE`` (train) or ``MET-Mumbai-mum_a-mum_b`` (metro)."""
        parts = (service_id or "").strip().split("-")
        if len(parts) == 3 and parts[0] in self.s["trains"]:
            t = self.s["trains"][parts[0]]
            codes = [st["code"] for st in t["stops"]]
            if parts[1] in codes and parts[2] in codes and codes.index(parts[1]) < codes.index(
                parts[2]
            ):
                km = (t["stops"][codes.index(parts[2])]["km"]
                      - t["stops"][codes.index(parts[1])]["km"])
                return {"kind": "train", "train": t, "from": parts[1], "to": parts[2], "km": km}
            return None
        if len(parts) == 4 and parts[0] == "MET" and parts[1] in self.s["metro"]:
            route = self.metro_route(parts[1], parts[2], parts[3])
            if route:
                return {"kind": "metro", "city": parts[1], "route": route}
        return None

    def rail_fare(self, km: float, cls: str) -> int:
        f = self.s["fares"]
        return int(round(km * f["per_km"][cls] + f["reservation"][cls]))

    # ── metro ────────────────────────────────────────────────────────────────

    def metro_route(self, city: str, a: str, b: str) -> dict | None:
        m = self.s["metro"][city]
        by_id = {s["id"]: s for s in m["stations"]}
        if a not in by_id or b not in by_id or a == b:
            return None
        adj: dict[str, list[tuple[str, str]]] = {}
        for line, ids in m["lines"].items():
            for x, y in zip(ids, ids[1:]):
                adj.setdefault(x, []).append((y, line))
                adj.setdefault(y, []).append((x, line))
        # Dijkstra over (station, line) states: a stop costs 1, a line change 5.
        start_lines = by_id[a]["lines"]
        pq = [(0, 0, a, ln, [(a, ln)]) for ln in start_lines]
        heapq.heapify(pq)
        seen: dict[tuple[str, str], int] = {}
        while pq:
            cost, stops, node, line, path = heapq.heappop(pq)
            if node == b:
                return self._route_summary(city, path, by_id)
            if seen.get((node, line), 1 << 30) <= cost:
                continue
            seen[(node, line)] = cost
            for nxt, ln in adj.get(node, []):
                step = 1 if ln == line else 5
                heapq.heappush(pq, (cost + step, stops + 1, nxt, ln, path + [(nxt, ln)]))
        return None

    def _route_summary(self, city: str, path: list[tuple[str, str]], by_id: dict) -> dict:
        legs: list[dict] = []
        for (sid, line), (prev_sid, _prev_line) in zip(path[1:], path[:-1]):
            if not legs or legs[-1]["line"] != line:
                legs.append({"line": line, "board": by_id[prev_sid]["name"], "stops": 0})
            legs[-1]["stops"] += 1
            legs[-1]["alight"] = by_id[sid]["name"]
        n_stops = sum(leg["stops"] for leg in legs)
        changes = len(legs) - 1
        a, b = path[0][0], path[-1][0]
        return {
            "service_id": f"MET-{city}-{a}-{b}",
            "city": city,
            "from": by_id[a]["name"],
            "to": by_id[b]["name"],
            "legs": legs,
            "stops": n_stops,
            "interchanges": [leg["board"] for leg in legs[1:]],
            "estimated_minutes": n_stops * 2 + changes * 5,
            "airport_express": any("Airport Express" in leg["line"] for leg in legs),
        }

    def metro_fare(self, route: dict) -> int:
        f = self.s["fares"]
        fare = next(v for cap, v in f["metro_slabs"] if route["stops"] <= cap)
        if route.get("airport_express"):
            fare += f["metro_airport_express_surcharge"]
        return fare

    # ── bookings ─────────────────────────────────────────────────────────────

    def new_pnr(self, *parts: Any) -> str:
        pnr = "86" + _digest(*parts)
        while pnr in self.bookings:
            pnr = "86" + _digest(pnr, "again")
        return pnr


def parse_date(date: str) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(date).strip())
    except ValueError:
        return None

