"""Turn one GRPO completion (plus AgentTune's rollout extras) into an episode AuditKIT can score.

Where the calls come from, in order of preference:

1. ``conversations[i]``, the chat transcript AgentTune's rollout forwards to reward functions
   (``rollout_factory._format_for_grpo``). Assistant turns carry the ``tool_calls`` the rollout
   parsed and executed.
2. A message-list completion (TRL's conversational format).
3. A plain-text completion, parsed with AgentTune's own ``_extract_tool_calls``.

Every generated assistant turn is also scanned with AuditKIT's parser, only to find tool-call
*attempts* that AgentTune's parser could not execute (broken JSON, an unsupported format). Those
become ``ToolCall``s with ``parse_error`` set. AuditKIT's ``ToolCallValidity`` counts them as
invalid, so a malformed call is no longer indistinguishable from no call (FINDINGS.md B11).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from auditkit.trace import ToolCall, normalize_call, parse_tool_calls, strip_thinking

# Opening markers of the tool-call formats AgentTune's parser understands. Markup that neither
# parser can read is still an attempted (malformed) call, not "no call".
TOOL_CALL_MARKUP_RE = re.compile(
    r"<tool_?call|<\|tool_call>|<function=|\[TOOL_CALLS\]|tool▁call|<\|python_tag\|>",
    re.IGNORECASE,
)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# AgentTune's DEFAULT_TOOL_RESULT_FORMAT: a tool result folded into a user turn, not the user.
FOLDED_TOOL_RESULT_PREFIX = "Tool result for "
NOT_EXECUTABLE = "tool call not recognised by AgentTune's parser, so the rollout never ran it"


@dataclass
class CallRecord:
    """One generated tool call (or failed attempt) and where it happened."""

    call: ToolCall
    message_index: int  # index into Episode.messages
    executable: bool  # AgentTune's parser produced it, so the rollout would execute it


@dataclass
class Episode:
    """The full message list (prompt history + generated turns) and the generated calls."""

    messages: list[dict[str, Any]]
    start: int  # index of the first message the policy generated
    calls: list[CallRecord] = field(default_factory=list)

    @property
    def executed(self) -> list[CallRecord]:
        return [c for c in self.calls if c.executable]

    @property
    def malformed(self) -> list[CallRecord]:
        return [c for c in self.calls if c.call.parse_error]

    def turns(self, executable_only: bool = False) -> list[list[ToolCall]]:
        """AuditKIT ``Turns``: one list of calls per generated assistant message."""
        by_msg: dict[int, list[ToolCall]] = {}
        for c in self.calls:
            if executable_only and not c.executable:
                continue
            by_msg.setdefault(c.message_index, []).append(c.call)
        return [by_msg[k] for k in sorted(by_msg)]

    def trace(self, executable_only: bool = False) -> dict[str, Any]:
        """The ``context["trace"]`` AuditKIT metrics read: explicit calls plus the generated
        messages (``AgentLoopDetection`` reads assistant text for reasoning stagnation)."""
        has_calls = {c.message_index for c in self.calls}
        msgs = []
        for i, m in enumerate(self.messages[self.start :], start=self.start):
            m = dict(m)
            if i in has_calls:
                m["tool_calls"] = m.get("tool_calls") or [{"name": "_"}]  # marks a call turn
            msgs.append(m)
        return {"tool_calls": self.turns(executable_only), "messages": msgs}


def _content(msg: dict[str, Any]) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):  # multimodal content parts
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return "" if c is None else str(c)


def _agenttune_calls(msg: dict[str, Any], text: str) -> list[dict]:
    from agenttune.agentic.rollout_engines.rollout_factory import _extract_tool_calls

    return _extract_tool_calls(msg, text) or []


def calls_in_message(msg: dict[str, Any]) -> list[tuple[ToolCall, bool]]:
    """``(call, executable)`` pairs for one assistant message."""
    text = _THINK_RE.sub("", _content(msg))
    native = msg.get("tool_calls")
    raw = native if native else _agenttune_calls(msg, text)
    out: list[tuple[ToolCall, bool]] = []
    for c in raw:
        try:
            out.append((normalize_call(c), True))
        except ValueError as e:  # a parsed shape AuditKIT cannot read: still an invalid call
            out.append((ToolCall(name="", parse_error=str(e)), True))

    # Attempts AgentTune could not execute, found by AuditKIT's independent parser.
    try:
        audit = parse_tool_calls(strip_thinking(text)) if text else []
    except (ValueError, TypeError):  # never let a parser edge case drop the sample
        audit = []
    if out:
        extra = [c for c in audit if c.parse_error][: max(0, len(audit) - len(out))]
        out += [(c, False) for c in extra]
    elif audit:
        for c in audit:
            err = c.parse_error or NOT_EXECUTABLE
            out.append((ToolCall(name=c.name, arguments=c.arguments, parse_error=err), False))
    elif TOOL_CALL_MARKUP_RE.search(text):
        out.append((ToolCall(name="", parse_error="unparseable tool-call markup"), False))
    return out


def _as_messages(x: Any, role: str) -> list[dict[str, Any]]:
    if isinstance(x, list) and all(isinstance(m, dict) for m in x):
        return [dict(m) for m in x]
    if isinstance(x, dict) and "role" in x:
        return [dict(x)]
    return [{"role": role, "content": "" if x is None else str(x)}]


def _last_user_turn_end(conversation: list[dict]) -> int:
    """Index after the last genuine user turn. AgentTune's GRPO rollout never adds a user turn
    of its own, except tool results folded into user turns, so this is where the prompt ends."""
    end = 0
    for i, m in enumerate(conversation):
        if m.get("role") == "user" and not _content(m).startswith(FOLDED_TOOL_RESULT_PREFIX):
            end = i + 1
    return end


def _history_end(conversation: list[dict], prompt: Any) -> int:
    """Index of the first conversation message after the prompt (the policy's first turn)."""
    if isinstance(prompt, list) and prompt and isinstance(prompt[-1], dict):
        # The rollout builds [system prompt?] + prompt messages + generated turns. But
        # _build_conversation returns the caller's prompt list itself and the rollout appends
        # to it (rollout_factory.py:86-89, :2250), so a prompt list seen after the rollout may
        # already hold the generated turns. A prompt at least as long as the transcript is that
        # case, and its length says nothing about where generation started.
        offset = int(
            bool(conversation)
            and conversation[0].get("role") == "system"
            and prompt[0].get("role") != "system"
        )
        end = offset + len(prompt)
        if end < len(conversation) and _content(conversation[end - 1]) == _content(prompt[-1]):
            return end
    elif isinstance(prompt, str):
        p = prompt.strip()
        for i, m in enumerate(conversation):
            if m.get("role") == "user" and p and (_content(m).strip() == p or p in _content(m)):
                return i + 1
    return _last_user_turn_end(conversation)


def build_episode(prompt: Any, completion: Any, conversation: Any = None) -> Episode:
    """Build the episode for one GRPO row."""
    if isinstance(conversation, list) and conversation:
        messages = [dict(m) for m in conversation]
        start = _history_end(messages, prompt)
    else:
        history = _as_messages(prompt, "user") if prompt is not None else []
        messages = history + _as_messages(completion, "assistant")
        start = len(history)

    ep = Episode(messages=messages, start=start)
    for i in range(start, len(messages)):
        if messages[i].get("role") != "assistant":
            continue
        for call, executable in calls_in_message(messages[i]):
            ep.calls.append(CallRecord(call=call, message_index=i, executable=executable))
    return ep
