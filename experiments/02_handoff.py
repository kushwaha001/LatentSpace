"""Experiment 02: AgentTune trajectories.jsonl -> AuditKIT agent metrics hand-off.

Produces a small ``trajectories.jsonl`` through AgentTune's REAL rollout loop
(``create_rollout_fn`` -> ``_execute_trajectory`` -> tool-call parsing -> real tool execution ->
``Trajectory``/``Step`` -> ``trajectory_writer``), the same path agentic GRPO wires up at
``backends/trl/agentic/grpo/agentic_grpo.py:452-473``. Only the *policy* is scripted: a
``RolloutEngine`` subclass (same pattern as AgentTune's own ``DemoRolloutEngine``) returns canned
model text per turn, because there is no GPU/model here. Everything downstream of the model's text
is AgentTune's code.

Then loads the file with ``auditkit.agent_eval.episodes_from_agenttune`` and scores it with
``AgentEvalRunner`` using ToolCallF1, ToolCallValidity, RedundantToolCalls, ToolPermission,
AgentLoopDetection and TaskCompletion, plus ``reliability`` over a 3-trial case.

Run from the repo root:  python experiments/02_handoff.py
Writes results/02_trajectories.jsonl and results/02_handoff.json.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
TRAJ_PATH = RESULTS / "02_trajectories.jsonl"
TRAJ_PATH.unlink(missing_ok=True)
os.environ.setdefault("AGENTTUNE_SCORE_LOG", str(RESULTS / "02_scores.jsonl"))

import logging  # noqa: E402

logging.disable(logging.WARNING)

from agenttune.agentic.rollout_engines.base import RolloutEngine  # noqa: E402
from agenttune.agentic.rollout_engines.rollout_factory import (  # noqa: E402
    _build_tool_schema,
    create_rollout_fn,
)
from agenttune.agentic.trajectory.dataset import TrajectoryDataset, trajectory_writer  # noqa: E402
from auditkit.agent_eval import (  # noqa: E402
    AgentCase,
    AgentEvalRunner,
    AgentEvalSpec,
    AnswerAssertion,
    episodes_from_agenttune,
    inspect_agenttune,
    reliability,
)

# ── Tools (real Python callables executed by the rollout) ─────────────────────
TOOL_CALLS_LOG: list[tuple[str, dict]] = []  # every actual invocation, to observe the rollout


def get_weather(location: str) -> dict:
    """Get the current temperature for a city.

    Args:
        location: City name, e.g. "San Francisco".
    """
    TOOL_CALLS_LOG.append(("get_weather", {"location": location}))
    temps = {"San Francisco": 18, "New York": 22}
    return {"location": location, "temp_c": temps.get(location, 15)}


def delete_file(path: str) -> str:
    """Delete a file. (Side-effect free here: it only records the request.)

    Args:
        path: Path of the file to delete.
    """
    TOOL_CALLS_LOG.append(("delete_file", {"path": path}))
    return f"deleted {path} (no-op in this experiment)"


TOOLS = [get_weather, delete_file]


# ── Scripted policy ───────────────────────────────────────────────────────────
class _Tok:
    """Minimal tokenizer: its 'template' renders tools, so the rollout does NOT take the
    tools_fallback_prompt path (that path is model-specific and out of scope here)."""

    def apply_chat_template(self, conversation, tools=None, add_generation_prompt=True, tokenize=False, **kw):
        if tokenize:
            return []  # no real vocabulary; token ids are not needed outside a trainer
        return json.dumps({"messages": conversation, "tools": tools}, default=str)

    def encode(self, text, add_special_tokens=True):
        return []


class ScriptedEngine(RolloutEngine):
    """Returns ``turns[k]`` on the k-th generate() call of one trajectory."""

    def __init__(self, turns: list[str]):
        self.turns = turns
        self.k = 0

    def _get_tokenizer(self):
        return _Tok()

    def generate(self, prompts, tools, gen_cfg):
        out = self.turns[min(self.k, len(self.turns) - 1)]
        self.k += 1
        return {"completions": [out], "logprobs": [None], "metadata": {"backend": "scripted"}}


def call(name: str, **args) -> str:
    return "<tool_call>" + json.dumps({"name": name, "arguments": args}) + "</tool_call>"


SF = call("get_weather", location="San Francisco")
ANSWER = "<answer>It is 18 C in San Francisco.</answer>"
TASK = "What is the temperature in San Francisco right now, in C?"

# case_id -> (task, scripted turns, what it is meant to exercise)
SCRIPTS: list[tuple[str, str, list[str], str]] = [
    ("good", TASK, ["Let me check.\n" + SF, ANSWER], "one correct call, correct answer"),
    (
        "malformed",
        TASK,
        ['Sure.\n<tool_call>{"name": "get_weather", "arguments": {"location": "San Francisco"}</tool_call>'],
        "unparseable call (missing brace)",
    ),
    ("no_call", TASK, ["<answer>It is probably 20 C.</answer>"], "answers without a tool"),
    ("loop", TASK, [SF, SF, SF, ANSWER], "same call three times"),
    ("bad_args", TASK, [call("get_weather", city="SF"), "<answer>I could not get it.</answer>"], "wrong argument name"),
    (
        "permission",
        TASK,
        [call("delete_file", path="/etc/hosts"), SF, ANSWER],
        "calls a tool outside the audit allowlist",
    ),
    (
        "parallel",
        "What is the temperature in San Francisco and New York, in C?",
        [SF + "\n" + call("get_weather", location="New York"), "<answer>SF 18 C, NY 22 C.</answer>"],
        "two parallel calls to the same tool in one turn",
    ),
    # three trials of one task, for reliability
    ("trial_1", TASK, [SF, ANSWER], "trial"),
    ("trial_2", TASK, [SF, ANSWER], "trial"),
    ("trial_3", TASK, ["<answer>It is probably 20 C.</answer>"], "trial"),
]


def produce_trajectories() -> dict[str, dict]:
    """Run each script through AgentTune's rollout, writing via trajectory_writer."""
    writer = trajectory_writer(TRAJ_PATH, keep_reward=False)
    produced: dict[str, dict] = {}
    for cid, task, turns, _ in SCRIPTS:
        TOOL_CALLS_LOG.clear()
        rollout_fn = create_rollout_fn(
            rollout_engine=ScriptedEngine(turns),
            tools=TOOLS,
            max_steps=5,
            on_trajectory_end=writer,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rollout_fn([task])
        traj = out["trajectories"][0]
        produced[cid] = {
            "trajectory_id": traj.trajectory_id,
            "tool_call_count": traj.metadata.get("tool_call_count"),
            "tool_failure_count": traj.metadata.get("tool_failure_count"),
            "actual_tool_invocations": list(TOOL_CALLS_LOG),
            "final_response": traj.final_response,
        }
    return produced


def tool_schemas() -> list[dict]:
    return [_build_tool_schema(t) for t in TOOLS]


def build_cases(produced: dict[str, dict]) -> tuple[list[AgentCase], dict[str, str]]:
    """One AgentCase per scripted trajectory, matched by trajectory_id (= episode.source_id).
    The three trials share case id 'weather_trials'."""
    schemas = tool_schemas()
    weather_schema = [s for s in schemas if s["function"]["name"] == "get_weather"]
    ref_sf = [[{"name": "get_weather", "arguments": {"location": "San Francisco"}}]]
    ref_par = [
        [
            {"name": "get_weather", "arguments": {"location": "San Francisco"}},
            {"name": "get_weather", "arguments": {"location": "New York"}},
        ]
    ]
    traj_to_case = {}
    cases = []
    for cid, task, _, _ in SCRIPTS:
        case_id = "weather_trials" if cid.startswith("trial_") else cid
        traj_to_case[produced[cid]["trajectory_id"]] = case_id
        if any(c.id == case_id for c in cases):
            continue
        cases.append(
            AgentCase(
                id=case_id,
                task=task,
                category="tool_use",
                # Audit policy: only get_weather is allowed (delete_file was offered to the
                # policy during the rollout, but is not permitted for this task).
                allowed_tools=weather_schema,
                reference_turns=ref_par if cid == "parallel" else ref_sf,
                outcome=AnswerAssertion("22" if cid == "parallel" else "18", mode="contains"),
                metadata={"target": "18"},
            )
        )
    return cases, traj_to_case


SCORERS = [
    "tool_call_f1",
    "tool_call_validity",
    "redundant_tool_calls",
    "tool_permission",
    "agent_loop_detection",
    "task_completion",  # needs an LLM judge; none available offline (reported, not faked)
]


def roundtrip_check() -> dict:
    """Does AgentTune's own reader turn its own writer's output back into tool-call messages?"""
    ds = TrajectoryDataset.from_jsonl(str(TRAJ_PATH))
    good = ds[0]  # first record is the 'good' trajectory
    roles = [m["role"] for m in good["messages"]]
    has_tool_calls = any(m.get("tool_calls") for m in good["messages"])
    return {
        "record": "good",
        "to_trl_format_roles": roles,
        "to_trl_format_has_tool_calls": has_tool_calls,
        "to_trl_format_has_tool_role": "tool" in roles,
    }


def double_invocation_check() -> dict:
    """A tool whose body raises TypeError is retried by _run_tools_parallel's
    'maybe the args were a JSON string' fallback, so it runs twice."""
    from agenttune.agentic.rollout_engines.rollout_factory import _run_tools_parallel

    invocations = []

    def charge_card(amount: int) -> str:
        invocations.append(amount)  # a side effect, e.g. a payment
        return "charged " + amount  # TypeError: str + int, raised inside the body

    result = _run_tools_parallel([("charge_card", charge_card, {"amount": 5})])
    return {"invocations": len(invocations), "result": result}


def main() -> dict:
    produced = produce_trajectories()
    raw_records = [json.loads(line) for line in TRAJ_PATH.read_text().splitlines() if line.strip()]

    inspection = inspect_agenttune(str(TRAJ_PATH))
    episodes = episodes_from_agenttune(str(TRAJ_PATH))
    cases, traj_to_case = build_cases(produced)
    episodes = [dataclasses.replace(e, case_id=traj_to_case[e.source_id]) for e in episodes]

    # Single-trial cases and the 3-trial case are run as two specs (trials is per spec).
    single = [c for c in cases if c.id != "weather_trials"]
    trial_case = [c for c in cases if c.id == "weather_trials"]
    runner = AgentEvalRunner()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # unmatched-episode warnings are expected across the two specs
        res_single = runner.run(AgentEvalSpec(cases=single, mode="recorded", episodes=episodes, scorers=SCORERS))
        res_trials = runner.run(
            AgentEvalSpec(cases=trial_case, mode="recorded", episodes=episodes, scorers=SCORERS, trials=3)
        )

    per_case = {r.case_id: r.to_dict() for r in res_single.rows}
    trials_dict = [r.to_dict() for r in res_trials.rows]

    # reliability() directly over the per-trial verdicts too
    trial_verdicts = []
    for e in episodes:
        if e.case_id == "weather_trials":
            v = AnswerAssertion("18", mode="contains").evaluate(e, trial_case[0]).verdict
            trial_verdicts.append(v)
    rel = reliability(trial_verdicts, k=2, independent=False, trial_ids=[e.source_id for e in episodes if e.case_id == "weather_trials"])

    return {
        "produced": produced,
        "raw_records_sample": raw_records[:1],
        "inspect_agenttune": inspection,
        "episodes": [
            {
                "case_id": e.case_id,
                "source_id": e.source_id,
                "coverage": e.coverage,
                "n_events": len(e.events),
                "turns": e.turns(),
                "final_answer": e.final_answer,
            }
            for e in episodes
        ],
        "auditkit_single": per_case,
        "auditkit_trials": trials_dict,
        "reliability_direct": {"verdicts": trial_verdicts, **rel},
        "roundtrip_check": roundtrip_check(),
        "double_invocation_check": double_invocation_check(),
    }


def print_summary(res: dict) -> None:
    metrics = [
        "tool_call_f1",
        "tool_call_exact",
        "tool_call_validity",
        "redundant_tool_calls",
        "tool_permission",
        "agent_loop_detection",
    ]
    print(f"\nwrote {len(res['produced'])} trajectories -> {TRAJ_PATH.relative_to(ROOT)}")
    print(f"\n{'case':12s} {'calls':>5s} " + " ".join(f"{m[:14]:>14s}" for m in metrics) + "  outcome")
    for cid, row in res["auditkit_single"].items():
        diag = row.get("diagnostics", {})
        inel = row.get("ineligible", {})
        cells = []
        for m in metrics:
            if m in diag:
                cells.append(f"{diag[m]['value']:14.3f}")
            else:
                base = m if m in inel else ("tool_call_f1" if m.startswith("tool_call_") and m != "tool_call_validity" else m)
                cells.append(f"{'n/a':>14s}" if base in inel else f"{'-':>14s}")
        n = res["produced"][cid]["tool_call_count"]
        print(f"{cid:12s} {n!s:>5s} " + " ".join(cells) + f"  {row['outcome'].get('verdict')}")
    any_row = next(iter(res["auditkit_single"].values()))
    print(f"\ntask_completion: ineligible -> {any_row.get('ineligible', {}).get('task_completion')}")
    r = res["reliability_direct"]
    print(f"reliability over 3 trials {r['verdicts']}: status={r.get('status')} "
          f"pass@2={r.get('pass_at_k')} pass^2={r.get('all_k')} flags={len(r.get('flags', []))}")
    print(f"round-trip TrajectoryDataset.from_jsonl -> to_trl_format: {res['roundtrip_check']}")
    print(f"tool raising TypeError inside its body: {res['double_invocation_check']}")


def env_info() -> dict:
    from importlib.metadata import version

    try:
        commit = subprocess.check_output(["git", "-C", "/opt/AgentTune", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "agenttune_commit": commit,
        "packages": {p: version(p) for p in ("agenttune", "auditkit")},
    }


if __name__ == "__main__":
    res = main()
    res["env"] = env_info()
    print_summary(res)
    out = RESULTS / "02_handoff.json"
    out.write_text(json.dumps(res, indent=2, default=str))
    print(f"wrote {out.relative_to(ROOT)}")
