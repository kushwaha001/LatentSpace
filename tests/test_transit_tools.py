"""Tests for the transit tools and mock backend (CPU, real AgentTune schema builder)."""

from __future__ import annotations

import json

import pytest

from audit_rewards import AuditRewards, tool_schema
from transit_agent import (
    TOOLS,
    book_ticket,
    cancel_booking,
    execute_call,
    file_complaint,
    get_backend,
    get_fare,
    get_pnr_status,
    get_station_code,
    normalize_name,
    reset_backend,
    search_metro_route,
    search_trains,
)
from transit_agent.multilingual_confirmation import is_confirmation


@pytest.fixture(autouse=True)
def fresh():
    reset_backend()


def test_eight_tools_with_clean_schemas():
    schemas = {tool_schema(t)["function"]["name"]: tool_schema(t)["function"] for t in TOOLS}
    assert set(schemas) == {"get_station_code", "search_trains", "search_metro_route",
                            "get_fare", "get_pnr_status", "book_ticket", "cancel_booking",
                            "file_complaint"}
    for name, fn in schemas.items():
        assert fn["description"], name
        for arg, spec in fn["parameters"]["properties"].items():
            assert spec.get("description"), f"{name}.{arg} has no description"
    assert schemas["get_station_code"]["parameters"]["required"] == ["name"]
    assert schemas["get_fare"]["parameters"]["properties"]["travel_class"]["enum"] == [
        "2S", "SL", "CC", "3A", "2A", "EC", "1A", "standard"]
    assert schemas["get_fare"]["parameters"]["properties"]["passengers"]["type"] == "integer"
    assert "category" in schemas["file_complaint"]["parameters"]["required"]
    assert "booking_id" not in schemas["file_complaint"]["parameters"]["required"]


def test_irreversible_tags():
    assert AuditRewards(tools=TOOLS, log=False).irreversible == {"book_ticket", "cancel_booking"}


@pytest.mark.parametrize("name,code", [
    ("Pune", "PUNE"), ("पुणे", "PUNE"), ("Pune Jn.", "PUNE"), ("पुणे स्टेशन", "PUNE"),
    ("poona", "PUNE"), ("Mumbai Central", "MMCT"), ("मुंबई सेंट्रल", "MMCT"),
    ("मुंबई सीएसएमटी", "CSMT"), ("नई दिल्ली", "NDLS"), ("नवी दिल्ली", "NDLS"),
    ("बनारस", "BSB"), ("Allahabad", "PRYJ"), ("कोल्हापूर", "KOP"), ("कोल्हापुर", "KOP"),
    ("हावड़ा", "HWH"), ("हावडा", "HWH"), ("Bangalore", "SBC"), ("लखनौ", "LKO"),
])
def test_station_aliases_any_script(name, code):
    assert get_station_code(name)["code"] == code


def test_ambiguous_city_returns_candidates():
    r = get_station_code("Mumbai")
    assert {c["code"] for c in r["candidates"]} >= {"CSMT", "MMCT", "LTT", "BDTS"}
    assert "error" in get_station_code("Atlantis")


def test_normalize_name():
    assert normalize_name("हज़रत निज़ामुद्दीन") == normalize_name("हजरत निजामुद्दीन")
    assert normalize_name("द्वारका सेक्टर २१") == normalize_name("द्वारका सेक्टर 21")


def test_search_trains_and_fare():
    r = search_trains("CSMT", "PUNE", "2026-11-20")
    assert r["count"] > 0
    t = next(x for x in r["trains"] if "CC" in x["classes"])
    f = get_fare(t["service_id"], "CC", 2)
    assert f["total_fare"] == 2 * f["fare_per_person"] > 0
    assert "error" in get_fare(t["service_id"], "1A", 1)  # class not on this train
    assert "error" in search_trains("CSMT", "PUNE", "20/11/2026")
    assert "error" in search_trains("XXXX", "PUNE", "2026-11-20")


def test_deterministic():
    a = search_trains("NDLS", "LKO", "2026-11-27")
    reset_backend()
    assert search_trains("NDLS", "LKO", "2026-11-27") == a


@pytest.mark.parametrize("city,a,b,interchange", [
    ("Mumbai", "Versova", "Cuffe Parade", "Marol Naka"),
    ("पुणे", "पीसीएमसी", "रामवाडी", "Civil Court"),
    ("Delhi", "राजीव चौक", "विमानतळ", "New Delhi"),
])
def test_metro_routes(city, a, b, interchange):
    r = search_metro_route(city, a, b)
    assert interchange in r["interchanges"]
    assert get_fare(r["service_id"], "standard", 1)["total_fare"] > 0
    assert "error" in get_fare(r["service_id"], "SL", 1)


def test_metro_unknown_city():
    assert "error" in search_metro_route("Chennai", "A", "B")


def test_booking_lifecycle_is_stateful_and_resettable():
    r = book_ticket("S20001-CSMT-PUNE", "2026-11-20", "CC", 2, "Rahul Sharma")
    assert r["pnr"].startswith("86") and len(r["pnr"]) == 10
    assert get_pnr_status(r["pnr"])["passenger_name"] == "Rahul Sharma"
    assert cancel_booking(r["pnr"])["status"] == "CANCELLED"
    assert "error" in cancel_booking(r["pnr"])  # already cancelled
    reset_backend()
    assert "error" in get_pnr_status(r["pnr"])


def test_book_rejects_wrong_day_and_bad_input():
    assert "error" in book_ticket("S20001-CSMT-PUNE", "tomorrow", "CC", 1, "A")
    assert "error" in book_ticket("S20001-CSMT-PUNE", "2026-11-20", "CC", 9, "A")
    assert "error" in book_ticket("S20001-CSMT-PUNE", "2026-11-20", "CC", 1, " ")


def test_complaint():
    r = file_complaint("cleanliness", "डबा घाणेरडा होता", "8412093355")
    assert r["status"] == "REGISTERED" and r["complaint_id"].startswith("CMP")
    assert "error" in file_complaint("nonsense", "x")
    assert len(get_backend().complaints) == 1


def test_execute_call_never_raises():
    assert "error" in execute_call("no_such_tool", {})
    assert "error" in execute_call("get_pnr_status", {})
    assert "error" in execute_call("get_pnr_status", {"pnr": "123", "extra": 1})
    json.dumps(execute_call("search_trains", {"from_code": "NDLS", "to_code": "LKO",
                                              "date": "2026-11-27"}))


@pytest.mark.parametrize("text,yes", [
    ("Yes, go ahead.", True), ("Yes, cancel it.", True), ("हाँ, बुक कर दीजिए।", True),
    ("हो, बुक करा.", True), ("Haan, book kar do.", True), ("Ok, haan.", True),
    ("No, don't book it.", False), ("नहीं, बुक मत कीजिए।", False), ("नको, बुक करू नका.", False),
    ("Skip karo, ok?", False), ("Actually, cancel that.", False), ("होगा क्या?", False),
    ("Rehne do yaar.", False),
])
def test_multilingual_confirmation(text, yes):
    assert is_confirmation(text) is yes
