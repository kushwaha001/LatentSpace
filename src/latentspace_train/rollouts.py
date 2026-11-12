"""Run AgentTune rollouts (``create_rollout_fn``) over task rows and keep everything needed
to score them later: original prompt, transcript, parsed calls, final answer."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from typing import Any

from .engines import BudgetExceeded


def build_rollout_fn(engine, tools: list, *, system_prompt: str | None, max_steps: int,
                     on_trajectory_end: Callable | None = None):
    """AgentTune's own rollout loop with its Cohere ``tools_fallback_prompt`` (the default),
    so a tool-ignoring chat template (Tiny Aya, Aya Expanse) gets the schemas in the system
    prompt and tool results folded into user turns."""
    from agenttune.agentic.rollout_engines.rollout_factory import (
        DEFAULT_TOOL_RESULT_FORMAT,
        DEFAULT_TOOLS_FALLBACK_PROMPT,
        create_rollout_fn,
    )

    return create_rollout_fn(
        rollout_engine=engine,
        tools=tools,
        max_steps=max_steps,
        system_prompt=system_prompt,
        on_trajectory_end=on_trajectory_end,
        tools_fallback_prompt=DEFAULT_TOOLS_FALLBACK_PROMPT,
        tool_result_format=DEFAULT_TOOL_RESULT_FORMAT,
    )


def run_rollouts(
    rollout_fn,
    rows: list[dict],
    *,
    trials: int = 1,
    gen_kwargs: dict | None = None,
    on_record: Callable[[dict], None] | None = None,
    log_every: int = 10,
    skip: set | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """One rollout per (row, trial). Rows are copied before the rollout because AgentTune
    appends the transcript to message-list prompts in place (FINDINGS.md B14).

    Stops cleanly on ``BudgetExceeded`` and records the reason; any other per-rollout error
    is recorded on the row and counted, not raised. ``skip``: ``(task_id, trial)`` pairs
    that are already done and are not run again (resume after a crash)."""
    gen_kwargs = dict(gen_kwargs or {})
    records: list[dict] = []
    status = {"stopped_early": False, "stop_reason": None, "errors": 0, "rollouts": 0}
    t0 = time.time()
    for trial in range(trials):
        for row in rows:
            if skip and (row.get("id"), trial) in skip:  # already collected (resume)
                continue
            rec: dict[str, Any] = {
                "task_id": row.get("id"),
                "language": row.get("language", "unknown"),
                "trial": trial,
                "prompt": row["prompt"],
                "expected_tool_calls": row.get("expected_tool_calls"),
                "category": row.get("category"),
            }
            try:
                out = rollout_fn(copy.deepcopy(row["prompt"]), **gen_kwargs)
                traj = out["trajectories"][0]
                rec.update(
                    response=traj.final_response,
                    conversation=out["conversations"][0],
                    tool_call_count=out["tool_call_counts"][0],
                    trajectory=traj.to_dict(),
                    error=None,
                )
            except BudgetExceeded as e:
                status.update(stopped_early=True, stop_reason=str(e))
                return records, _finish(status, t0)
            except Exception as e:  # recorded, never hidden
                status["errors"] += 1
                rec.update(response="", conversation=[], tool_call_count=0, trajectory=None,
                           error=f"{type(e).__name__}: {e}")
            records.append(rec)
            status["rollouts"] += 1
            if on_record:
                on_record(rec)
            if log_every and status["rollouts"] % log_every == 0:
                print(f"  rollouts: {status['rollouts']} ({time.time() - t0:.0f}s, "
                      f"{status['errors']} errors)", flush=True)
    return records, _finish(status, t0)


def _finish(status: dict, t0: float) -> dict:
    status["seconds"] = round(time.time() - t0, 1)
    return status
