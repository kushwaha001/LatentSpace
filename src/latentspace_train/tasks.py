"""Tools and task rows for every stage.

``tasks.source: transit`` (default) imports the transit tool set and reads the eval set by
the names in the config:

    tasks:
      tools: transit_agent:TOOLS            # module:attr, a list or a zero-arg callable
      system_prompt: transit_agent:SYSTEM_PROMPT   # optional; a literal string also works
      data:
        dir: data/eval
        files: {train: train.jsonl, eval: eval.jsonl, test: test.jsonl}
        # or one file with a split column:  file: tasks.jsonl, split_field: split
        fields: {id: id, prompt: prompt, language: language,
                 expected_tool_calls: expected_tool_calls}

``tasks.source: stub`` uses ``latentspace_train.stub`` (tests and ``--dry-run``).

Every row comes out as ``{id, language, prompt, expected_tool_calls, ...}`` where
``prompt`` is a string or a message list and ``expected_tool_calls`` is a JSON string
(Arrow-friendly) or ``None``.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from .config import ROOT


class TasksUnavailable(RuntimeError):
    pass


def _import_attr(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError(f"expected 'module:attr', got {spec!r}")
    mod_name, attr = spec.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        raise TasksUnavailable(
            f"cannot import {mod_name!r} ({e}). The transit tools live in src/transit_agent/ "
            "(branch eval-set). Merge that branch, or run with --set tasks.source=stub "
            "(or --dry-run) to use the stub tools."
        ) from e
    obj = mod
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def _resolve(value: Any) -> Any:
    if isinstance(value, str) and ":" in value and " " not in value and not value.startswith(
        ("http:", "https:")
    ):
        obj = _import_attr(value)
        return obj() if callable(obj) and _is_factory(obj) else obj
    return value


def _is_factory(obj: Any) -> bool:
    """A zero-argument callable that is not itself a tool (tools take arguments)."""
    import inspect

    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return False
    return not any(
        p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        for p in sig.parameters.values()
    )


def load_tools(cfg: dict) -> list:
    t = cfg.get("tasks", {})
    if t.get("source") == "stub":
        from . import stub

        return list(stub.TOOLS)
    tools = _resolve(t.get("tools", "transit_agent:TOOLS"))
    if not isinstance(tools, (list, tuple)) or not tools:
        raise TasksUnavailable(f"tasks.tools resolved to {type(tools).__name__}, not a tool list")
    return list(tools)


def system_prompt(cfg: dict) -> str | None:
    t = cfg.get("tasks", {})
    if t.get("source") == "stub":
        from . import stub

        return stub.SYSTEM_PROMPT
    sp = t.get("system_prompt")
    if not sp:
        return None
    try:
        return _resolve(sp)
    except (AttributeError, TasksUnavailable):
        return None if ":" in str(sp) else sp


def _read_rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else data.get("rows", data.get("data", []))
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalise(raw: dict, fields: dict, split: str) -> dict:
    get = lambda k, d=None: raw.get(fields.get(k, k), d)  # noqa: E731
    row = dict(raw)
    row["id"] = str(get("id", raw.get("task_id", "")))
    row["language"] = get("language", raw.get("lang", "unknown")) or "unknown"
    row["prompt"] = get("prompt", raw.get("messages"))
    ref = get("expected_tool_calls")
    if ref is not None and not isinstance(ref, str):
        ref = json.dumps(ref, ensure_ascii=False)
    row["expected_tool_calls"] = ref
    row["split"] = split
    if row["prompt"] is None:
        raise ValueError(f"row {row['id']!r} has no prompt (fields map: {fields})")
    return row


def load_split(cfg: dict, split: str, limit: int | None = None) -> list[dict]:
    t = cfg.get("tasks", {})
    if t.get("source") == "stub":
        from . import stub

        rows = stub.rows(split)
    else:
        d = t.get("data", {})
        base = Path(d.get("dir", "data/eval"))
        base = base if base.is_absolute() else ROOT / base
        fields = d.get("fields", {})
        if d.get("file"):
            path = base / d["file"]
            if not path.exists():
                raise TasksUnavailable(f"{path} not found (eval set: branch eval-set)")
            sf = d.get("split_field", "split")
            raw = [r for r in _read_rows(path) if r.get(sf) == split]
        else:
            name = (d.get("files") or {}).get(split, f"{split}.jsonl")
            path = base / name
            if not path.exists():
                raise TasksUnavailable(
                    f"{path} not found. The eval set lives in data/eval/ (branch eval-set); "
                    "merge it or run with --set tasks.source=stub."
                )
            raw = _read_rows(path)
        rows = [_normalise(r, fields, split) for r in raw]
    langs = t.get("languages")
    if langs:
        rows = [r for r in rows if r["language"] in langs]
    limit = limit if limit is not None else t.get("limit")
    return rows[: int(limit)] if limit else rows


def to_dataset_rows(rows: list[dict]) -> list[dict]:
    """Columns the GRPO trainer and the rewards read. If any prompt is a message list (a
    confirmation turn), every prompt becomes one, so the Arrow column has a single type."""
    as_messages = any(isinstance(r["prompt"], list) for r in rows)

    def prompt(p):
        if as_messages and isinstance(p, str):
            return [{"role": "user", "content": p}]
        return p

    return [
        {
            "prompt": prompt(r["prompt"]),
            "expected_tool_calls": r.get("expected_tool_calls"),
            "language": r.get("language", "unknown"),
            "task_id": r.get("id", ""),
        }
        for r in rows
    ]
