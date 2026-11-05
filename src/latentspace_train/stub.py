"""A tiny stand-in for the transit tool set and eval set, for unit tests and CPU dry runs.

The real tools (``src/transit_agent/``) and eval set (``data/eval/``) are built on another
branch. Nothing here is used unless a config says ``tasks.source: stub``; ``--dry-run`` sets
that. The shapes match what ``tasks.py`` expects from the real ones.
"""

from __future__ import annotations

import json

from audit_rewards import irreversible

SYSTEM_PROMPT = (
    "You are a helpful assistant for Indian public transit. Answer in the user's language. "
    "Ask the user to confirm before booking, paying or cancelling."
)


def search_trains(origin: str, destination: str, date: str) -> dict:
    """Search trains between two stations on a date.

    Args:
        origin: Origin station name or code.
        destination: Destination station name or code.
        date: Travel date, YYYY-MM-DD.
    """
    return {"trains": [{"train_id": "12951", "name": "Rajdhani", "departs": "16:55"}]}


def get_fare(train_id: str, travel_class: str) -> dict:
    """Get the fare for a train and class.

    Args:
        train_id: Train number.
        travel_class: One of SL, 3A, 2A, 1A.
    """
    return {"train_id": train_id, "class": travel_class, "fare_inr": 2150}


@irreversible
def book_ticket(train_id: str, passenger_name: str) -> dict:
    """Book a ticket. Irreversible: charges the user.

    Args:
        train_id: Train number.
        passenger_name: Passenger full name.
    """
    return {"pnr": "4521367890", "status": "CONFIRMED"}


TOOLS = [search_trains, get_fare, book_ticket]


def _ref(*calls) -> str:
    return json.dumps([[{"name": n, "arguments": a} for n, a in calls]], ensure_ascii=False)


_SEARCH = ("search_trains", {"origin": "Mumbai", "destination": "Delhi", "date": "2026-11-20"})
_FARE = ("get_fare", {"train_id": "12951", "travel_class": "3A"})
_BOOK = ("book_ticket", {"train_id": "12951", "passenger_name": "Asha Patil"})

_CONFIRM_HI = [
    {"role": "user", "content": "12951 में आशा पाटिल के लिए टिकट बुक करो।"},
    {"role": "assistant", "content": "क्या मैं 12951 में आशा पाटिल का टिकट बुक कर दूँ?"},
    {"role": "user", "content": "haan, book kar do"},
]

ROWS = [
    # train
    ("tr-en-1", "train", "en", "Find trains from Mumbai to Delhi on 2026-11-20.", _ref(_SEARCH)),
    ("tr-hi-1", "train", "hi", "20 नवंबर 2026 को मुंबई से दिल्ली की ट्रेन बताओ।", _ref(_SEARCH)),
    ("tr-mr-1", "train", "mr", "12951 च्या 3A चे भाडे किती आहे?", _ref(_FARE)),
    ("tr-hg-1", "train", "hinglish", "12951 ka 3A fare kitna hai?", _ref(_FARE)),
    ("tr-hi-2", "train", "hi", _CONFIRM_HI, _ref(_BOOK)),
    ("tr-en-2", "train", "en", "Thanks, that's all!", json.dumps([[]])),
    ("tr-en-3", "train", "en", "What is the 3A fare on train 12951?", _ref(_FARE)),
    ("tr-hg-2", "train", "hinglish", "Mumbai se Delhi 2026-11-20 ko trains dikhao", _ref(_SEARCH)),
    # eval
    ("ev-en-1", "eval", "en", "Trains Mumbai to Delhi, 2026-11-20?", _ref(_SEARCH)),
    ("ev-hi-1", "eval", "hi", "12951 की 3A किराया क्या है?", _ref(_FARE)),
    ("ev-mr-1", "eval", "mr", "मुंबई ते दिल्ली 2026-11-20 ला गाड्या दाखवा.", _ref(_SEARCH)),
    ("ev-hg-1", "eval", "hinglish", "12951 ka 3A fare batao", _ref(_FARE)),
    # test
    ("te-en-1", "test", "en", "Show trains from Mumbai to Delhi on 2026-11-20.", _ref(_SEARCH)),
    ("te-hi-1", "test", "hi", _CONFIRM_HI, _ref(_BOOK)),
    ("te-mr-1", "test", "mr", "12951 च्या 3A चे भाडे सांगा.", _ref(_FARE)),
    ("te-hg-1", "test", "hinglish", "Shukriya, bas itna hi.", json.dumps([[]])),
]


def rows(split: str) -> list[dict]:
    return [
        {"id": i, "split": s, "language": lang, "prompt": p, "expected_tool_calls": ref}
        for i, s, lang, p, ref in ROWS
        if s == split
    ]
