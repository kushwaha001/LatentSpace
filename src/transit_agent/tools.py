"""The eight transit tools the agent can call.

Plain functions with Google-style docstrings and type hints, so AgentTune's
``rollout_factory._build_tool_schema`` (transformers ``get_json_schema``) turns each one into
an OpenAI-style schema with argument descriptions, required lists and enums. They run against
a module-level :class:`MockBackend`; call :func:`reset_backend` to start a row from a clean
state. Every tool returns a JSON-serialisable dict; a failure is ``{"error": "..."}`` (never an
exception), which is what the rollout shows the model as the tool result.

``book_ticket`` and ``cancel_booking`` are tagged ``audit_rewards.irreversible``: a call to
them must follow an explicit user confirmation turn.
"""

from __future__ import annotations

from typing import Literal, Optional

from audit_rewards import irreversible

from .backend import (
    COMPLAINT_CATEGORIES,
    MAX_PASSENGERS,
    METRO_CLASS,
    RAIL_CLASSES,
    MockBackend,
    parse_date,
)

TravelClass = Literal["2S", "SL", "CC", "3A", "2A", "EC", "1A", "standard"]
ComplaintCategory = Literal[
    "cleanliness", "punctuality", "staff_behaviour", "refund", "safety", "catering",
    "accessibility", "other",
]

_BACKEND = MockBackend()


def get_backend() -> MockBackend:
    return _BACKEND


def reset_backend() -> MockBackend:
    """Fresh backend state (pre-existing bookings only). Returns the new backend."""
    global _BACKEND
    _BACKEND = MockBackend()
    return _BACKEND


def _station_view(st: dict) -> dict:
    return {"code": st["code"], "name": st["name"], "city": st["city"], "state": st["state"]}


def get_station_code(name: str, city: Optional[str] = None) -> dict:
    """
    Look up the official Indian Railways station code for a railway station.

    Accepts English, Hindi or Marathi (Devanagari) and romanised names, e.g. "Pune",
    "पुणे", "Mumbai Central", "मुंबई सेंट्रल", "CSMT". A city-level name such as "Mumbai" or
    "Delhi" matches several stations and returns all candidates. Not for metro stations.

    Args:
        name: Station name as the user wrote it, in any script.
        city: Optional city to narrow the match, e.g. "Mumbai" or "दिल्ली".

    Returns:
        {"code": ..., "name": ..., "city": ...} for a unique match, {"candidates": [...]} when
        several stations match, or {"error": ...} when none does.
    """
    b = get_backend()
    found = b.find_rail_stations(name, city)
    if not found:
        return {"error": f"no railway station matches {name!r}", "query": name}
    if len(found) == 1:
        return _station_view(found[0])
    return {"candidates": [_station_view(s) for s in found],
            "message": "several stations match; ask the user which one"}


def search_trains(from_code: str, to_code: str, date: str) -> dict:
    """
    Search Indian Railways trains between two stations on a date.

    Args:
        from_code: Origin station code from get_station_code, e.g. "CSMT".
        to_code: Destination station code from get_station_code, e.g. "PUNE".
        date: Travel date in YYYY-MM-DD format, e.g. "2026-11-20".

    Returns:
        {"trains": [...]} where each train has service_id, train_number, train_name,
        departs, arrives, duration, classes and seat availability per class.
    """
    b = get_backend()
    frm, to = (from_code or "").strip().upper(), (to_code or "").strip().upper()
    for code in (frm, to):
        if code not in b.s["rail_by_code"]:
            return {"error": f"unknown station code {code!r}; call get_station_code first"}
    if frm == to:
        return {"error": "origin and destination are the same station"}
    day = parse_date(date)
    if day is None:
        return {"error": f"date must be YYYY-MM-DD, got {date!r}"}
    trains = b.trains_between(frm, to, day)
    return {"from": frm, "to": to, "date": day.isoformat(), "trains": trains,
            "count": len(trains)}


def search_metro_route(city: str, from_station: str, to_station: str) -> dict:
    """
    Find a metro route between two stations in Mumbai, Pune or Delhi.

    Station names may be English, Hindi or Marathi (Devanagari) or romanised, e.g.
    "Andheri", "अंधेरी", "Civil Court", "राजीव चौक".

    Args:
        city: Metro city: "Mumbai", "Pune" or "Delhi" (any script).
        from_station: Boarding metro station name.
        to_station: Destination metro station name.

    Returns:
        A route with service_id, legs (line, board, alight, stops), interchanges and
        estimated_minutes, or {"error": ...}.
    """
    b = get_backend()
    c = b.metro_city(city)
    if c is None:
        return {"error": f"no metro network for city {city!r}; supported: Mumbai, Pune, Delhi"}
    ends = []
    for name in (from_station, to_station):
        found = b.find_metro_station(c, name)
        if not found:
            return {"error": f"no {c} metro station matches {name!r}"}
        if len(found) > 1:
            return {"error": f"{name!r} is ambiguous", "candidates": [s["name"] for s in found]}
        ends.append(found[0]["id"])
    route = b.metro_route(c, ends[0], ends[1])
    if route is None:
        return {"error": "origin and destination are the same station"}
    return route


def get_fare(service_id: str, travel_class: TravelClass, passengers: int) -> dict:
    """
    Get the fare for a train or metro journey returned by search_trains or search_metro_route.

    Args:
        service_id: The service_id from search_trains (e.g. "S20000-CSMT-PUNE") or from
            search_metro_route (e.g. "MET-Mumbai-mum_andheri-mum_ghatkopar").
        travel_class: Train class code ("2S", "SL", "CC", "3A", "2A", "EC", "1A"), or
            "standard" for metro.
        passengers: Number of passengers, 1 to 6.

    Returns:
        {"service_id", "travel_class", "passengers", "fare_per_person", "total_fare",
        "currency"}, or {"error": ...}.
    """
    b = get_backend()
    svc = b.parse_service(service_id)
    if svc is None:
        return {"error": f"unknown service_id {service_id!r}"}
    if not isinstance(passengers, int) or not 1 <= passengers <= MAX_PASSENGERS:
        return {"error": f"passengers must be an integer from 1 to {MAX_PASSENGERS}"}
    if svc["kind"] == "metro":
        if travel_class != METRO_CLASS:
            return {"error": "metro tickets have one class: 'standard'"}
        per = b.metro_fare(svc["route"])
    else:
        if travel_class not in svc["train"]["classes"]:
            return {"error": f"class {travel_class!r} not on this train",
                    "classes": svc["train"]["classes"]}
        per = b.rail_fare(svc["km"], travel_class)
    return {"service_id": service_id, "travel_class": travel_class, "passengers": passengers,
            "fare_per_person": per, "total_fare": per * passengers, "currency": "INR"}


def get_pnr_status(pnr: str) -> dict:
    """
    Get the current status of an Indian Railways booking by its 10-digit PNR number.

    Args:
        pnr: 10-digit PNR number, e.g. "8412093355".

    Returns:
        Booking details with status (CNF, RAC, WL or CANCELLED), or {"error": ...}.
    """
    b = get_backend()
    p = "".join(ch for ch in str(pnr) if ch.isdigit())
    if len(p) != 10:
        return {"error": f"a PNR has 10 digits, got {pnr!r}"}
    if p not in b.bookings:
        return {"error": f"PNR {p} not found"}
    return dict(b.bookings[p])


@irreversible
def book_ticket(
    service_id: str, date: str, travel_class: TravelClass, passengers: int, passenger_name: str
) -> dict:
    """
    Book and pay for a train or metro ticket. IRREVERSIBLE: only call after the user has
    explicitly confirmed the exact train, date, class, passengers and fare.

    Args:
        service_id: The service_id from search_trains or search_metro_route.
        date: Travel date in YYYY-MM-DD format.
        travel_class: Train class code ("2S", "SL", "CC", "3A", "2A", "EC", "1A"), or
            "standard" for metro.
        passengers: Number of passengers, 1 to 6.
        passenger_name: Full name of the lead passenger.

    Returns:
        {"pnr", "status", "total_fare", ...} for the new booking, or {"error": ...}.
    """
    b = get_backend()
    fare = get_fare(service_id, travel_class, passengers)
    if "error" in fare:
        return fare
    day = parse_date(date)
    if day is None:
        return {"error": f"date must be YYYY-MM-DD, got {date!r}"}
    if not str(passenger_name).strip():
        return {"error": "passenger_name is required"}
    svc = b.parse_service(service_id)
    if svc["kind"] == "train":
        if not any(t["train_number"] == svc["train"]["number"]
                   for t in b.trains_between(svc["from"], svc["to"], day)):
            return {"error": f"train {svc['train']['number']} does not run on {day.isoformat()}"}
        avail = b._availability(svc["train"]["number"], travel_class, day)
        status = "CNF" if avail.startswith("AVAILABLE") else avail.split("-")[0]
    else:
        status = "CNF"
    pnr = b.new_pnr(service_id, day, travel_class, passengers, passenger_name)
    rec = {"pnr": pnr, "service_id": service_id, "date": day.isoformat(),
           "travel_class": travel_class, "passengers": passengers,
           "passenger_name": passenger_name, "status": status,
           "total_fare": fare["total_fare"], "currency": "INR"}
    b.bookings[pnr] = rec
    return rec


@irreversible
def cancel_booking(pnr: str, reason: Optional[str] = None) -> dict:
    """
    Cancel a booking and start the refund. IRREVERSIBLE: only call after the user has
    explicitly confirmed they want this PNR cancelled.

    Args:
        pnr: 10-digit PNR number of the booking to cancel.
        reason: Optional reason given by the user.

    Returns:
        {"pnr", "status": "CANCELLED", "refund_amount", ...}, or {"error": ...}.
    """
    b = get_backend()
    st = get_pnr_status(pnr)
    if "error" in st:
        return st
    if st["status"] == "CANCELLED":
        return {"error": f"PNR {st['pnr']} is already cancelled"}
    b.bookings[st["pnr"]]["status"] = "CANCELLED"
    return {"pnr": st["pnr"], "status": "CANCELLED", "previous_status": st["status"],
            "reason": reason or "",
            "refund": "refund initiated to the original payment method (synthetic)"}


def file_complaint(
    category: ComplaintCategory, description: str, booking_id: Optional[str] = None
) -> dict:
    """
    File a passenger complaint (like RailMadad) and get a reference number.

    Args:
        category: One of "cleanliness", "punctuality", "staff_behaviour", "refund", "safety",
            "catering", "accessibility", "other".
        description: What happened, in the user's own words (any language).
        booking_id: Optional PNR of the journey the complaint is about.

    Returns:
        {"complaint_id", "category", "status": "REGISTERED"}, or {"error": ...}.
    """
    b = get_backend()
    if category not in COMPLAINT_CATEGORIES:
        return {"error": f"category must be one of {list(COMPLAINT_CATEGORIES)}"}
    if not str(description).strip():
        return {"error": "description is required"}
    if booking_id:
        st = get_pnr_status(booking_id)
        if "error" in st:
            return st
    cid = "CMP" + b.new_pnr(category, description, booking_id)[2:]
    rec = {"complaint_id": cid, "category": category, "booking_id": booking_id,
           "status": "REGISTERED"}
    b.complaints.append(rec)
    return rec


TOOLS = [
    get_station_code,
    search_trains,
    search_metro_route,
    get_fare,
    get_pnr_status,
    book_ticket,
    cancel_booking,
    file_complaint,
]
TOOLS_BY_NAME = {t.__name__: t for t in TOOLS}
# Tools that never appear in train/dev rows (unseen-tool generalisation test).
HELD_OUT_TOOLS = ("file_complaint", "cancel_booking")
RAIL_CLASS_CODES = RAIL_CLASSES
