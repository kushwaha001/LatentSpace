"""Multilingual Indian public-transit tools (train + metro) on a deterministic mock backend.

    from transit_agent import TOOLS, reset_backend
    trainer = create_agentic_trainer("grpo", tools=TOOLS, ...)

Station names and Indian Railways codes are real; schedules, fares and PNRs are synthetic
(data/backend/DATACARD.md).
"""

from .backend import MockBackend, normalize_name
from .tools import (
    HELD_OUT_TOOLS,
    TOOLS,
    TOOLS_BY_NAME,
    book_ticket,
    cancel_booking,
    file_complaint,
    get_backend,
    get_fare,
    get_pnr_status,
    get_station_code,
    reset_backend,
    search_metro_route,
    search_trains,
)

__all__ = [
    "HELD_OUT_TOOLS",
    "TOOLS",
    "TOOLS_BY_NAME",
    "MockBackend",
    "book_ticket",
    "cancel_booking",
    "execute_call",
    "file_complaint",
    "get_backend",
    "get_fare",
    "get_pnr_status",
    "get_station_code",
    "normalize_name",
    "reset_backend",
    "search_metro_route",
    "search_trains",
]


def execute_call(name: str, arguments: dict) -> dict:
    """Run one tool call by name on the current backend (``{"error"}`` for an unknown tool)."""
    fn = TOOLS_BY_NAME.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}"}
    try:
        return fn(**arguments)
    except TypeError as e:  # wrong / missing arguments
        return {"error": str(e)}
