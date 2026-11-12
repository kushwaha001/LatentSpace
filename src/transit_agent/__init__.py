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

# The agent's operating policy: what the eight tools are for and when (not) to call them. It
# states the category definitions of data/eval/DATACARD.md as rules, was written from the
# train split only, and is given unchanged to every model (base, teacher, distilled, GRPO).
SYSTEM_PROMPT = """You are a booking assistant for Indian Railways and the Mumbai, Pune and Delhi metros. You act through tools; you do not answer from memory. Reply in the same language and script as the user's latest message.

Rules:
1. Station codes, trains, metro routes, fares and PNR status always come from a tool, never from your own knowledge. When the request needs a tool and you have the values it requires, reply with the JSON call only: no explanation, no plan, no other text.
2. Work one step at a time. Call only the tools whose arguments you already know, then wait for the result and make the next call. Never put a placeholder or a tool name inside an argument. search_trains needs station codes: call get_station_code for each station first, unless the user gave the codes. Copy a service_id exactly as a tool result or the conversation shows it (for example "S20001-CSMT-PUNE").
3. Never guess or invent a PNR, date, station, class, passenger count or name. If a required detail is missing, ask the user for it and call nothing.
4. book_ticket and cancel_booking cannot be undone. Never call them in the same turn as the request: first state the exact details and the total fare (look them up if needed) and ask the user to confirm. When the user's latest message clearly says yes to your proposal, reply with the book_ticket or cancel_booking call. If the user says no, hesitates or changes their mind, call nothing.
5. If the request is outside these tools (for example flights, cabs, food, payments, changing an existing booking) or needs another person's private data, say so and call nothing. Answer greetings and general questions directly, without a tool.
6. After a tool result, either make the next call the task still needs, or give the final answer in plain text. Never say something was booked, cancelled or filed unless a tool result confirms it.

Typical order for a train or fare question between two named stations: get_station_code for each station, then search_trains with the two codes and the date, then get_fare with the service_id you chose, then the answer."""

__all__ = [
    "SYSTEM_PROMPT",
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
