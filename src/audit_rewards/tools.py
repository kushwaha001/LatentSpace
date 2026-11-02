"""Tool schemas and tags, built the way AgentTune's rollout builds them."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

IRREVERSIBLE_KEY = "x-irreversible"  # schema-level tag, alongside "type"/"function"


def irreversible(tool: Callable) -> Callable:
    """Tag a tool (plain function or ``BaseTool``) as irreversible: booking, paying,
    sending, deleting. Calls to it must follow a user confirmation turn."""
    tool.irreversible = True
    return tool


def tool_name(tool: Any) -> str:
    if isinstance(tool, dict):
        fn = tool.get("function", tool)
        return str(fn.get("name"))
    return tool.name if hasattr(tool, "name") else tool.__name__


def tool_schema(tool: Any) -> dict:
    """OpenAI-style schema for one tool, using the same code path as
    ``rollout_factory._execute_trajectory``: ``BaseTool.to_schema()`` or
    ``_build_tool_schema(fn)`` (transformers ``get_json_schema`` with AgentTune's fallback).
    A schema dict is passed through."""
    if isinstance(tool, dict):
        return tool
    if hasattr(tool, "to_schema"):
        return tool.to_schema()
    from agenttune.agentic.rollout_engines.rollout_factory import _build_tool_schema

    return _build_tool_schema(tool)


def tool_schemas(tools: Iterable[Any] | None) -> list[dict]:
    return [tool_schema(t) for t in tools or []]


def is_irreversible(tool: Any) -> bool:
    if isinstance(tool, dict):
        fn = tool.get("function", tool)
        return bool(
            tool.get(IRREVERSIBLE_KEY) or (isinstance(fn, dict) and fn.get(IRREVERSIBLE_KEY))
        )
    return bool(getattr(tool, "irreversible", False))


def irreversible_names(tools: Iterable[Any] | None, extra: Iterable[str] | None = None) -> set[str]:
    names = {tool_name(t) for t in tools or [] if is_irreversible(t)}
    return names | set(extra or [])
