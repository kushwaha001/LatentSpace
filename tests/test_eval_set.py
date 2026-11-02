"""Schema and integrity tests for data/eval (the 4-language eval set)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from auditkit.trace import to_turns

from transit_agent import TOOLS_BY_NAME, execute_call, reset_backend
from transit_agent.build_eval import (
    CATEGORIES,
    HELD_OUT_TOOLS,
    LANGS,
    build_rows,
    load_tasks,
)

EVAL = Path(__file__).resolve().parents[1] / "data" / "eval"
FIELDS = {"id", "base_id", "language", "split", "category", "messages",
          "expected_tool_calls", "irreversible", "notes"}


@pytest.fixture(scope="module")
def rows():
    return [json.loads(x) for x in (EVAL / "eval_set.jsonl").read_text("utf-8").splitlines()]


def test_committed_file_matches_source(rows):
    assert build_rows(load_tasks()) == rows, "run: PYTHONPATH=src python -m transit_agent.build_eval"


def test_row_schema(rows):
    for r in rows:
        assert set(r) == FIELDS, r["id"]
        assert r["id"] == f"{r['base_id']}-{r['language']}"
        assert r["language"] in LANGS and r["category"] in CATEGORIES
        assert r["split"] in ("train", "dev", "test")
        assert isinstance(r["irreversible"], bool) and isinstance(r["notes"], str)
        assert r["messages"] and r["messages"][-1]["role"] == "user"
        for m in r["messages"]:
            assert set(m) == {"role", "content"} and m["role"] in ("user", "assistant")
            assert m["content"].strip()
        assert isinstance(r["expected_tool_calls"], list)
        for step in r["expected_tool_calls"]:
            assert step and all(set(c) == {"name", "arguments"} for c in step)
            assert all(c["name"] in TOOLS_BY_NAME for c in step)
        to_turns(r["expected_tool_calls"])  # AuditKIT accepts the shape


def test_counts_balanced_and_paired(rows):
    base = Counter(r["base_id"] for r in rows)
    assert len(base) == 90 and set(base.values()) == {4}
    cats = Counter(r["category"] for r in rows if r["language"] == "en")
    assert set(cats) == set(CATEGORIES) and max(cats.values()) - min(cats.values()) <= 1


def test_scripts(rows):
    deva = lambda s: any("ऀ" <= ch <= "ॿ" for ch in s)  # noqa: E731
    for r in rows:
        user = " ".join(m["content"] for m in r["messages"] if m["role"] == "user")
        if r["language"] in ("hi", "mr"):
            assert deva(user), r["id"]
        else:
            assert not deva(user), r["id"]


def test_split_by_base_id_and_held_out_tools(rows):
    splits = {}
    for r in rows:
        assert splits.setdefault(r["base_id"], r["split"]) == r["split"]
    for r in rows:
        names = {c["name"] for s in r["expected_tool_calls"] for c in s}
        if names & set(HELD_OUT_TOOLS) or "held-out tool" in r["notes"]:
            assert r["split"] == "test", r["id"]
    assert all(Counter(r["split"] for r in rows if r["category"] == c) for c in CATEGORIES)


def test_category_semantics(rows):
    for r in rows:
        names = [c["name"] for s in r["expected_tool_calls"] for c in s]
        cat = r["category"]
        if cat in ("no_call_needed", "missing_argument", "declined", "denied_tool"):
            assert names == [], r["id"]
        if cat == "irreversible_unconfirmed":
            assert not set(names) & {"book_ticket", "cancel_booking"}, r["id"]
        if cat == "irreversible_confirmed":
            assert names[-1] in ("book_ticket", "cancel_booking") and len(r["messages"]) == 3
        if cat == "multi_step":
            assert len(r["expected_tool_calls"]) >= 2, r["id"]
        if cat in ("irreversible_confirmed", "declined"):
            assert [m["role"] for m in r["messages"]] == ["user", "assistant", "user"]


def test_expected_calls_execute(rows):
    for r in rows:
        reset_backend()
        for step in r["expected_tool_calls"]:
            for c in step:
                res = execute_call(c["name"], c["arguments"])
                assert "error" not in res and "candidates" not in res, (r["id"], c, res)


def test_review_checklist_covers_every_non_english_turn(rows):
    import csv

    with (EVAL / "review_checklist.csv").open(encoding="utf-8") as f:
        lines = list(csv.DictReader(f))
    expected = sum(len(r["messages"]) for r in rows if r["language"] != "en")
    assert len(lines) == expected and {x["status"] for x in lines} == {"todo"}
