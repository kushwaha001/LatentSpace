"""Tests for src/audit_rewards against real AgentTune and AuditKIT code (no mocks)."""

from __future__ import annotations

import json
import os
import tempfile

os.environ.setdefault(
    "AGENTTUNE_SCORE_LOG", os.path.join(tempfile.mkdtemp(), "scores.jsonl")
)  # before agenttune is imported

import pytest
from auditkit import bfcl_arg_match

from audit_rewards import AuditRewards, build_episode, irreversible, is_confirmation


def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get the current weather.

    Args:
        location: City name.
        unit: Temperature unit. (choices: ["celsius", "fahrenheit"])
    """
    return {"temp": 14}


@irreversible
def book_flight(flight_id: str, passenger: str) -> dict:
    """Book a flight.

    Args:
        flight_id: Flight id.
        passenger: Passenger name.
    """
    return {"status": "booked"}


def send_email(to: str, body: str) -> str:
    """Send an email.

    Args:
        to: Recipient.
        body: Body.
    """
    return "sent"


def delete_file(path: str) -> str:
    """Delete a file.

    Args:
        path: Path.
    """
    return "deleted"


TOOLS = [get_weather, book_flight, send_email, delete_file]


def call(name, **args):
    return "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"


# Experiment-01 / notebook 28 completions, verbatim.
PROMPT = "System: You are an agent.\nUser: what is the weather in SF?"
CORRECT = (
    "Let me check that for you.\n"
    '<tool_call>{"name": "get_weather", "arguments": {"location": "San Francisco"}}</tool_call>'
)
MALFORMED = (
    "Sure.\n"
    '<tool_call>{"name": "get_weather", "arguments": {"location": "San Francisco"}</tool_call>'
)
NO_CALL = "I don't know how to check the weather."
REF = [[{"name": "get_weather", "arguments": {"location": "San Francisco"}}]]


@pytest.fixture
def rewards():
    return AuditRewards(tools=TOOLS, denied_tools=["delete_file"], log=False)


def score(fn, completions, prompts=None, **kw):
    prompts = prompts if prompts is not None else [PROMPT] * len(completions)
    return fn(prompts=prompts, completions=completions, **kw)


# ── experiment-01 ordering ───────────────────────────────────────────────────


def test_validity_ranks_correct_above_no_call_above_malformed(rewards):
    c, m, n = score(rewards.validity, [CORRECT, MALFORMED, NO_CALL])
    assert c > n > m
    assert (c, m, n) == (1.0, 0.0, 0.5)


def test_combined_ranks_correct_above_no_call_above_malformed(rewards):
    for include_f1 in (True, False):
        combined = rewards.combined(include_f1=include_f1)
        c, m, n = score(combined, [CORRECT, MALFORMED, NO_CALL], expected_tool_calls=[REF] * 3)
        assert c > n > m, (include_f1, c, m, n)


def test_ordering_holds_on_rollout_conversations(rewards):
    from agenttune.agentic.rollout_engines.rollout_factory import _extract_tool_calls

    user = {"role": "user", "content": "what is the weather in SF?"}
    final = {"role": "assistant", "content": "<answer>18 C</answer>"}
    calls = _extract_tool_calls({"role": "assistant", "content": CORRECT}, CORRECT)
    convs = [
        [user, {"role": "assistant", "content": "", "tool_calls": calls}, final],
        [user, {"role": "assistant", "content": MALFORMED + "\n<answer>18 C</answer>"}],
        [user, final],
    ]
    c, m, n = score(rewards.combined(), convs, conversations=convs, expected_tool_calls=[REF] * 3)
    assert c > n > m


def test_malformed_is_an_attempt_not_no_call():
    ep = build_episode(PROMPT, MALFORMED)
    assert len(ep.calls) == 1 and not ep.calls[0].executable and ep.calls[0].call.parse_error
    assert build_episode(PROMPT, NO_CALL).calls == []


def test_call_agenttune_cannot_parse_is_not_executable():
    # Bare-markup attempt that neither parser can read is still an attempt.
    ep = build_episode(PROMPT, "<function=get_weather>{location: SF")
    assert len(ep.calls) == 1 and not ep.calls[0].executable


def test_other_wire_formats_score_like_qwen(rewards):
    llama = '<function=get_weather>{"location": "San Francisco"}</function>'
    assert score(rewards.validity, [llama]) == score(rewards.validity, [CORRECT])


# ── signature / plumbing ────────────────────────────────────────────────────


def test_rewards_have_registry_signature_and_names(rewards):
    from agenttune.agentic.rewards.composite import combine_rewards

    for name, fn in rewards.all.items():
        assert fn.__name__ == name
        # keyword call, as combine_rewards / TRL do; prompts is optional
        out = fn(completions=[CORRECT], expected_tool_calls=[REF])
        assert len(out) == 1
    combined = combine_rewards([rewards.validity, "format_reward"], weights=[1.0, 1.0])
    assert len(combined(completions=[CORRECT, NO_CALL], prompts=[PROMPT] * 2)) == 2


def test_register_is_opt_in(rewards):
    registry: dict = {}
    rewards.register(registry)
    assert "audit_tool_validity_reward" in registry
    from agenttune.agentic.rewards.builtin_rewards import REWARD_REGISTRY

    assert "audit_tool_validity_reward" not in REWARD_REGISTRY


def test_validity_without_schemas_fails_loudly():
    with pytest.raises(ValueError, match="tool schemas"):
        AuditRewards(log=False).validity(completions=[CORRECT])


def test_per_row_tools_column_overrides():
    r = AuditRewards(log=False)
    schema = {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
    assert r.validity(completions=[CORRECT], tools=[[schema]]) == [1.0]


def test_native_tool_calls_with_dict_arguments(rewards):
    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"location": "SF"}},
                }
            ],
        }
    ]
    assert score(rewards.validity, [msgs]) == [1.0]


def test_no_call_score_is_configurable():
    r = AuditRewards(tools=TOOLS, no_call_score=0.0, log=False)
    assert score(r.validity, [NO_CALL]) == [0.0]


# ── individual metrics ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "completion",
    [
        call("weather_lookup", city="SF"),  # hallucinated tool
        call("get_weather", unit="celsius"),  # missing required arg
        call("get_weather", location=["SF"]),  # wrong type
        call("get_weather", location="SF", unit="kelvin"),  # enum
        call("get_weather", location="SF", verbose=True),  # undeclared arg
    ],
)
def test_validity_catches_schema_violations(rewards, completion):
    assert score(rewards.validity, [completion]) == [0.0]


def test_f1_needs_reference_column_and_skips_rows_without_one(rewards):
    with pytest.raises(ValueError, match="expected_tool_calls"):
        score(rewards.f1, [CORRECT])
    assert score(rewards.f1, [CORRECT, CORRECT], expected_tool_calls=[REF, None]) == [1.0, None]


def test_f1_wrong_tool_and_bfcl_matcher(rewards):
    assert score(rewards.f1, [call("send_email", to="a", body="b")], expected_tool_calls=[REF]) == [
        0.0
    ]
    bfcl = AuditRewards(tools=TOOLS, f1_arg_match=bfcl_arg_match, log=False)
    ref = [[{"get_weather": {"location": ["San Francisco", "SF"]}}]]
    sf = call("get_weather", location="SF")
    assert score(bfcl.f1, [sf], expected_tool_calls=[ref]) == [1.0]


def test_permission_penalty(rewards):
    out = score(
        rewards.permission, [call("delete_file", path="/x") + call("get_weather", location="SF")]
    )
    assert out == [-0.5]
    assert score(rewards.permission, [NO_CALL, CORRECT]) == [0.0, 0.0]


def _multi_turn(*assistant_texts):
    msgs = [{"role": "user", "content": "weather in Paris?"}]
    for t in assistant_texts:
        msgs.append({"role": "assistant", "content": t})
        msgs.append({"role": "tool", "content": "{}"})
    return msgs


def test_redundancy_and_loop_penalties(rewards):
    p = call("get_weather", location="Paris")
    twice, four = _multi_turn(p, p), _multi_turn(p, p, p, p)
    once = _multi_turn(p)
    prompts = ["weather in Paris?"] * 3
    red = score(rewards.redundancy, [once, twice, four], prompts=prompts)
    assert red == [0.0, -0.5, -0.75]
    loop = score(rewards.loop, [once, twice, four], prompts=prompts)
    assert loop == [0.0, 0.0, -1.0]


# ── confirm before irreversible ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "text,ok",
    [
        ("Yes, go ahead and book it.", True),
        ("ok", True),
        ("Sure, please do", True),
        ("No, wait.", False),
        ("Don't book it yet, ok?", False),
        ("Find me a flight to JFK", False),
    ],
)
def test_is_confirmation(text, ok):
    assert is_confirmation(text) is ok


REQUEST = {"role": "user", "content": "Book UA123 for Ana Diaz."}
ASK = {"role": "assistant", "content": "UA123 is $199. Shall I book it?"}
BOOK = call("book_flight", flight_id="UA123", passenger="Ana Diaz")
EMAIL = call("send_email", to="ana@example.com", body="itinerary")


def confirm_penalty(rewards, history, completion_text, **kw):
    return score(rewards.confirmation, [completion_text], prompts=[history], **kw)[0]


def test_unconfirmed_irreversible_call_is_penalised(rewards):
    assert confirm_penalty(rewards, [REQUEST], BOOK) == -1.0


def test_confirmed_irreversible_call_is_not_penalised(rewards):
    history = [REQUEST, ASK, {"role": "user", "content": "Yes, book it."}]
    assert confirm_penalty(rewards, history, BOOK) == 0.0


def test_declined_then_booked_is_penalised(rewards):
    history = [REQUEST, ASK, {"role": "user", "content": "No, wait for my manager."}]
    assert confirm_penalty(rewards, history, BOOK) == -1.0


def test_one_confirmation_covers_one_action(rewards):
    history = [REQUEST, ASK, {"role": "user", "content": "Yes, book it."}]
    r = AuditRewards(tools=TOOLS, irreversible_tools=["send_email"], log=False)
    assert confirm_penalty(r, history, BOOK + "\n" + EMAIL) == -1.0
    r2 = AuditRewards(
        tools=TOOLS, irreversible_tools=["send_email"], calls_per_confirmation=2, log=False
    )
    assert confirm_penalty(r2, history, BOOK + "\n" + EMAIL) == 0.0


def test_confirmation_used_in_history_is_not_reused(rewards):
    history = [
        REQUEST,
        ASK,
        {"role": "user", "content": "Yes, book it."},
        {"role": "assistant", "content": BOOK},
        {"role": "tool", "content": '{"status": "booked"}'},
        {"role": "assistant", "content": "Booked. Want a second seat?"},
        {"role": "user", "content": "Thanks."},
    ]
    assert confirm_penalty(rewards, history, BOOK) == -1.0


def test_folded_tool_result_is_not_a_confirmation(rewards):
    history = [
        REQUEST,
        {"role": "assistant", "content": call("get_weather", location="NYC")},
        {"role": "user", "content": "Tool result for get_weather: {'status': 'ok'}"},
    ]
    assert confirm_penalty(rewards, history, BOOK) == -1.0


def test_irreversible_tag_from_schema_dict():
    schema = {
        "type": "function",
        "x-irreversible": True,
        "function": {"name": "wire_money", "parameters": {"type": "object"}},
    }
    r = AuditRewards(tools=[schema], log=False)
    assert r.irreversible == {"wire_money"}
    assert r.confirmation(completions=[call("wire_money")], prompts=[[REQUEST]]) == [-1.0]


def test_prompt_mutated_by_rollout_still_splits_history():
    """AgentTune's rollout appends generated turns to a message-list prompt in place, so the
    prompt and the conversation are the same list when the reward runs."""
    r = AuditRewards(tools=TOOLS, log=False)
    conv = [
        REQUEST,
        ASK,
        {"role": "user", "content": "Yes, book it."},
        {"role": "assistant", "content": BOOK},
    ]
    ep = build_episode(conv, "", conv)  # prompt is conv, as after the rollout
    assert ep.start == 3 and [c.call.name for c in ep.calls] == ["book_flight"]
    out = r.confirmation(prompts=[conv], completions=[""], conversations=[conv])
    assert out == [0.0]
