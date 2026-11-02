"""Experiment 01: does AgentTune's built-in reward stack tell a good tool call from a bad one?

Scores three completions for the same tool task with every reward in AgentTune's real
``REWARD_REGISTRY`` (no mocks, no reimplementations):

  correct    - a well-formed ``<tool_call>`` for ``get_weather``
  malformed  - the same call with a missing closing brace (unparseable JSON)
  no_call    - no tool call at all

Two scenarios:

  notebook   - the exact three completions from AgentTune's docs/notebooks/28_hybrid_prm_real.ipynb,
               i.e. the setup behind the documented "identical scores" gap.
  controlled - the rollout-shaped version: the correct case also carries the tool result and a
               final ``<answer>``; malformed/no_call end with the *same* final answer, so the only
               difference between the three is the tool-call part.

Each reward is called the way TRL calls it in agentic GRPO: ``completions`` as decoded text
plus the rollout extras AgentTune forwards (``tool_call_counts``, ``conversations``, ``answer``,
``prompts``). ``tool_call_counts`` come from AgentTune's own parser
(``rollout_factory._extract_tool_calls``), exactly as the rollout loop counts calls. The
``controlled`` scenario is also run with message-list completions, because sql.py's rewards only
look at chat-message structure.

Run from the repo root:  python experiments/01_reward_gap.py
Writes results/01_reward_gap.json and results/01_reward_scores.jsonl (AgentTune's own
per-score reasons from ``log_score``).
"""

from __future__ import annotations

import asyncio
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
SCORE_LOG = RESULTS / "01_reward_scores.jsonl"
SCORE_LOG.unlink(missing_ok=True)
# Must be set before agenttune is imported: score_logger reads it at import time and otherwise
# writes data/scores.jsonl relative to the CWD.
os.environ["AGENTTUNE_SCORE_LOG"] = str(SCORE_LOG)
os.environ.setdefault("JUDGMENTS_FILE", str(RESULTS / "01_judgments.jsonl"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # distilled_judge's default model id is not a real repo

import logging  # noqa: E402

logging.disable(logging.CRITICAL)  # litellm/agenttune log a lot; scores and reasons are captured below
import litellm  # noqa: E402

litellm.suppress_debug_info = True

from agenttune.agentic.rewards.builtin_rewards import REWARD_REGISTRY  # noqa: E402
from agenttune.agentic.rewards.composite import combine_rewards  # noqa: E402
from agenttune.agentic.rollout_engines.rollout_factory import _extract_tool_calls  # noqa: E402

CASES = ["correct", "malformed", "no_call"]

# ── The task ──────────────────────────────────────────────────────────────────
PROMPT = "System: You are an agent.\nUser: what is the weather in SF?"
GOLD = "18"  # temperature in C returned by the tool
TOOL_RESULT = '{"location": "San Francisco", "temp_c": 18}'
FINAL = "<answer>It is 18 degrees C in San Francisco.</answer>"

# Verbatim from docs/notebooks/28_hybrid_prm_real.ipynb.
NB_CORRECT = (
    "Let me check that for you.\n"
    '<tool_call>{"name": "get_weather", "arguments": {"location": "San Francisco"}}</tool_call>'
)
NB_MALFORMED = (
    "Sure.\n"
    '<tool_call>{"name": "get_weather", "arguments": {"location": "San Francisco"}</tool_call>'
)
NB_NO_CALL = "I don't know how to check the weather."
NB_PROMPTS = [PROMPT, PROMPT, "System: You are an agent.\nUser: tell me a joke"]


def tool_call_count(text: str) -> int:
    calls = _extract_tool_calls({"role": "assistant", "content": text}, text)
    return len(calls or [])


def conversation(case: str) -> list[dict]:
    """The chat transcript the rollout would hold in Trajectory.metadata['conversation']."""
    user = {"role": "user", "content": "what is the weather in SF?"}
    if case == "correct":
        calls = _extract_tool_calls({"role": "assistant", "content": NB_CORRECT}, NB_CORRECT)
        return [
            user,
            {"role": "assistant", "content": "Let me check that for you.", "tool_calls": calls},
            {"role": "tool", "content": TOOL_RESULT},
            {"role": "assistant", "content": FINAL},
        ]
    if case == "malformed":
        # Unparseable call -> no tool executed; the model then answers anyway.
        return [user, {"role": "assistant", "content": NB_MALFORMED + "\n" + FINAL}]
    return [user, {"role": "assistant", "content": FINAL}]


def scenarios() -> dict[str, dict]:
    nb_texts = [NB_CORRECT, NB_MALFORMED, NB_NO_CALL]
    ctl_texts = [
        NB_CORRECT + "\n" + TOOL_RESULT + "\n" + FINAL,
        NB_MALFORMED + "\n" + FINAL,
        FINAL,
    ]
    convs = [conversation(c) for c in CASES]
    return {
        "notebook_text": {
            "prompts": NB_PROMPTS,
            "completions": nb_texts,
            "kwargs": {"tool_call_counts": [tool_call_count(t) for t in nb_texts]},
        },
        "controlled_text": {
            "prompts": [PROMPT] * 3,
            "completions": ctl_texts,
            "kwargs": {
                "answer": [GOLD] * 3,
                "tool_call_counts": [tool_call_count(t) for t in ctl_texts],
                "conversations": convs,
            },
        },
        "controlled_messages": {
            "prompts": [PROMPT] * 3,
            "completions": convs,
            "kwargs": {
                "answer": [GOLD] * 3,
                "tool_call_counts": [tool_call_count(t) for t in ctl_texts],
                "conversations": convs,
            },
        },
    }


def call_reward(fn, prompts, completions, kwargs) -> dict:
    """Call ``fn`` the way TRL's GRPOTrainer does (keyword ``prompts``/``completions``)."""
    t0 = time.time()
    try:
        scores = fn(prompts=prompts, completions=completions, **kwargs)
        out = {"scores": [None if s is None else float(s) for s in scores]}
    except Exception as e:  # report, never hide
        out = {"scores": None, "error": f"{type(e).__name__}: {e}"}
    out["seconds"] = round(time.time() - t0, 3)
    return out


def verdict(scores: list[float] | None) -> str:
    if scores is None:
        return "ERROR"
    if len(scores) < 3:
        return "SHORT"
    c, m, n = scores
    if c == m == n:
        return "identical"
    if c > m and c > n:
        return "correct>both"
    if c < m or c < n:
        return "INVERTED"
    return "partial"


def run() -> dict:
    results: dict = {"scenarios": {}, "notes": {}}
    for sname, sc in scenarios().items():
        rows = {}
        for key, fn in REWARD_REGISTRY.items():
            r = call_reward(fn, sc["prompts"], sc["completions"], sc["kwargs"])
            r["verdict"] = verdict(r["scores"])
            rows[key] = r
        # AgentTune's actual default stack (rag/trainer.py:34-35), through the real combine_rewards.
        combo = combine_rewards(
            ["search_grounding_reward", "format_reward", "answer_correctness_reward"],
            weights=[1.0, 0.2, 1.0],
        )
        r = call_reward(combo, sc["prompts"], sc["completions"], sc["kwargs"])
        r["verdict"] = verdict(r["scores"])
        rows["combine_rewards(RAG default)"] = r
        results["scenarios"][sname] = {
            "inputs": {
                "prompts": sc["prompts"],
                "completions": sc["completions"],
                **{k: v for k, v in sc["kwargs"].items() if k != "conversations"},
            },
            "rewards": rows,
        }

    # The notebook's own captured output (0.0 x3) came from calling hybrid_prm_reward inside
    # Jupyter's running event loop. Reproduce that path explicitly.
    fn = REWARD_REGISTRY["hybrid_prm_reward"]

    async def inside_loop():
        return call_reward(fn, NB_PROMPTS, [NB_CORRECT, NB_MALFORMED, NB_NO_CALL], {})

    r = asyncio.run(inside_loop())
    r["verdict"] = verdict(r["scores"])
    results["notes"]["hybrid_prm_inside_running_event_loop"] = r
    results["notes"]["parsed_tool_calls"] = {
        c: _extract_tool_calls({"role": "assistant", "content": t}, t)
        for c, t in zip(CASES, [NB_CORRECT, NB_MALFORMED, NB_NO_CALL], strict=True)
    }
    return results


def print_table(results: dict) -> None:
    for sname, block in results["scenarios"].items():
        print(f"\n=== scenario: {sname}   (tool_call_counts={block['inputs'].get('tool_call_counts')})")
        print(f"{'reward':38s} {'correct':>9s} {'malformed':>9s} {'no_call':>9s}  verdict")
        for key, r in block["rewards"].items():
            if r["scores"] is None:
                print(f"{key:38s} {'-':>9s} {'-':>9s} {'-':>9s}  ERROR {r['error'][:60]}")
                continue
            s = [f"{x:9.3f}" if x is not None else f"{'None':>9s}" for x in r["scores"]]
            s += [f"{'(none)':>9s}"] * (3 - len(s))  # reward returned fewer scores than completions
            print(f"{key:38s} {s[0]} {s[1]} {s[2]}  {r['verdict']}")
    n = results["notes"]["hybrid_prm_inside_running_event_loop"]
    print(f"\nhybrid_prm_reward called inside a running event loop: {n['scores']}  ({n['verdict']})")


def summarize(results: dict) -> dict:
    summary = {}
    for sname, block in results["scenarios"].items():
        counts: dict[str, list[str]] = {}
        for key, r in block["rewards"].items():
            counts.setdefault(r["verdict"], []).append(key)
        summary[sname] = counts
    hp = results["scenarios"]["notebook_text"]["rewards"]["hybrid_prm_reward"]["scores"]
    summary["documented_gap"] = {
        "claim": "hybrid_prm_reward scores well-formed, malformed and missing tool calls identically (1.0)",
        "source": "docs/features.md:63-65, docs/notebooks/local-notebook.md:102-107",
        "observed_hybrid_prm_notebook_inputs": hp,
        "identical": hp is not None and len(set(hp)) == 1,
        "all_one": hp == [1.0, 1.0, 1.0],
    }
    return summary


def env_info() -> dict:
    def ver(pkg):
        try:
            from importlib.metadata import version

            return version(pkg)
        except Exception:
            return None

    try:
        commit = subprocess.check_output(
            ["git", "-C", "/opt/AgentTune", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        commit = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "agenttune_commit": commit,
        "packages": {p: ver(p) for p in ("agenttune", "auditkit", "litellm", "torch", "transformers", "trl")},
    }


if __name__ == "__main__":
    res = run()
    print_table(res)
    res["summary"] = summarize(res)
    res["env"] = env_info()
    out = RESULTS / "01_reward_gap.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    g = res["summary"]["documented_gap"]
    print(
        f"\nDocumented gap (identical scores) reproduces: {g['identical']}  "
        f"-> hybrid_prm_reward on notebook inputs = {g['observed_hybrid_prm_notebook_inputs']}"
    )
    print(f"wrote {out.relative_to(ROOT)} and {SCORE_LOG.relative_to(ROOT)}")
