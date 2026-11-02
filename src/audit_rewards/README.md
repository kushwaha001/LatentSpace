# audit_rewards: AuditKIT agent metrics as AgentTune GRPO rewards

A small prototype. It turns AuditKIT's reference-free and reference-based agent metrics into
reward functions that plug into AgentTune's agentic GRPO, in place of (or next to) the built-in
rewards. Experiments 01 and 03 show the built-in rewards cannot tell a good tool call from a bad
one.

```python
from audit_rewards import AuditRewards, irreversible
from agenttune.core.backend_factory import create_agentic_trainer

tools = [get_weather, search_flights, irreversible(book_flight)]
rewards = AuditRewards(tools=tools, denied_tools=["delete_file"])

trainer = create_agentic_trainer(
    "grpo",
    model="...",
    tools=tools,
    reward_funcs=rewards.train_time(),          # a list; TrlAgenticGrpo combines it
    reward_weights=[1.0, 1.0, 1.0, 1.0, 0.25],  # same order as DEFAULT_WEIGHTS
    train_dataset=ds,                           # columns: prompt, expected_tool_calls (optional)
)
# or a single callable through AgentTune's own combine_rewards:
reward_fn = rewards.combined(extra=[("answer_correctness_reward", 1.0)])
```

Every reward has the `REWARD_REGISTRY` signature, `fn(prompts, completions, **kwargs) -> list`.
It works with `combine_rewards`, and its `__name__` is what TRL logs. `rewards.register()`
adds them to `REWARD_REGISTRY` so they can be named as strings. This is opt-in; nothing is
registered on import.

## What the rewards read

- **Calls**: AgentTune's own parsed tool calls. The adapter takes `conversations[i]` (the
  rollout transcript TRL forwards) when present; otherwise a message-list completion, otherwise
  the text completion parsed with `rollout_factory._extract_tool_calls`. AuditKIT's parser is
  used only to find call *attempts* AgentTune could not parse. Those are kept as invalid calls,
  so "malformed" is no longer the same as "no call" (FINDINGS.md B11).
- **Schemas**: built from `tools=` with the rollout's own builders (`BaseTool.to_schema()` /
  `_build_tool_schema`). A per-row `tools` column overrides them.
- **Reference**: an optional `expected_tool_calls` dataset column, in any shape AuditKIT accepts
  (a JSON string is fine, which keeps Arrow happy).
- **Prompt history**: only calls the policy generated are scored. History calls and user turns
  are used for the confirmation rule.

## Metric → reward

| Reward | AuditKIT metric | Range | No call / nothing to check |
|---|---|---|---|
| `audit_tool_validity_reward` | `ToolCallValidity` | 0..1 (fraction of attempted calls that are schema-valid) | `no_call_score` (0.5) |
| `audit_tool_f1_reward` | `ToolCallF1` (`arg_mode`, or `arg_match=bfcl_arg_match`) | 0..1 | 0.0 if a call was expected, 1.0 if none was (AuditKIT semantics). `None` for a row without a reference. Raises if the column is missing |
| `audit_tool_permission_penalty` | `ToolPermission` (allowlist defaults to the offered tools; denylist) | -1..0 | 0.0 |
| `audit_redundant_calls_penalty` | `RedundantToolCalls` | -1..0 | 0.0 |
| `audit_loop_penalty` | `AgentLoopDetection` | -1 or 0 | 0.0 |
| `audit_confirm_irreversible_penalty` | new: confirm before irreversible action | -1 or 0 | 0.0 |

The validity reward is the only one that ranks no call between a good and a broken call. The
penalties are 0 when nothing was violated, *including* when there was nothing to check. A
neutral midpoint for "no calls" would quietly turn every penalty into a "call more tools" bonus.

On the experiment-01 completions, validity gives correct / malformed / no call =
`1.0 / 0.0 / 0.5` in all three scenarios. `combined()` gives `0.471 / 0.000 / 0.118` (controlled
scenarios) and `0.471 / 0.000 / 0.353` (notebook scenario, whose no-call prompt is a joke where no
call is correct). The reference-free `combined(include_f1=False)` gives `0.308 / 0.000 / 0.154`.
AgentTune's RAG default gives `0.327 / 0.236 / 0.236`, and `hybrid_prm_reward` gives
`0.0 / 1.0 / 1.0`. See `results/03_audit_rewards.json`.

### Confirm before irreversible action

Tools tagged with `@irreversible`, `"x-irreversible": true` in their schema, or listed in
`irreversible_tools=` must be preceded by a user confirmation turn:

- a user message after the agent has spoken, which affirms ("yes", "go ahead", "confirm", …)
  and does not negate ("no", "wait", "don't", …);
- one confirmation covers `calls_per_confirmation` irreversible calls (default 1). A
  confirmation already used by a call in the prompt history is not reused;
- the opening request does not count as a confirmation, and neither do tool results folded
  into user turns (`"Tool result for …"`).

In AgentTune's GRPO rollout the policy never sees a new user turn, so confirmations come from
the prompt history. Train on rows whose `prompt` is a message list ending in the user's reply.

## Held-out split: train-time vs report-only

A metric you optimise stops being a trustworthy measurement of the thing it measures. GRPO will
find whatever the reward does not check (Goodhart). So keep one set of metrics for the reward
and a separate set, plus held-out data, for evaluation:

| Metric | Use | Why |
|---|---|---|
| `ToolCallValidity` | **train-time** | Checks the actual contract: schema, required args, types, enums. Hard to game without making calls valid. |
| `ToolCallF1` (`subset` / BFCL matcher) on the **train** references | **train-time** | Dense signal for tool choice and arguments. |
| `ToolPermission` | **train-time** (hard constraint) | A policy violation should never pay. |
| confirm-before-irreversible | **train-time** (hard constraint) | Same. Its keyword matcher must also be *evaluated* on held-out confirmation phrasings (below). |
| `RedundantToolCalls` | train-time, low weight (0.25) | Useful shaping. At high weight it pushes the policy to skip legitimate re-checks. |
| `AgentLoopDetection` | **report-only** (excluded from `train_time()`) | It is a heuristic (exact repeats, cycles, text similarity). If it is optimised, the policy learns to perturb arguments or paraphrase, and keeps looping. It stays useful only as an independent check. |
| `tool_call_exact`, `TrajectoryMatch` | **report-only** | Strict whole-trajectory matches. Report them on held-out tasks to see whether the dense F1 reward transferred. |
| `TaskCompletion` (LLM judge), `ToolSelectionJudge` | **report-only** | Expensive, slow, and a judge in the reward loop gets gamed (B3 shows the judge path is not even offline-safe today). |
| `reliability` (pass@k / pass^k) | **report-only** | Needs repeated trials of the final policy. |

**Held-out data.** Evaluate with `AuditRewards.report()` or AuditKIT's `AgentEvalRunner`
(experiment 02) on:

- prompts and references not used for training, including tasks where no call is correct;
- at least one tool the policy never trained on (does validity generalise to new schemas?);
- confirmation turns phrased differently from the keyword list ("let's do it", "make it so",
  "nah, hold off"), to measure the confirmation rule's false-positive and false-negative rate
  rather than trusting it;
- irreversible tools that appear only in eval, to check that the policy learned "ask first"
  and not "never call `book_flight`".

## Limits

- The confirmation rule is a keyword rule. It does not check that the confirmed action is the
  one that was proposed (same flight, same recipient).
- `ToolCallF1`'s exact mode with schemas tolerates any value for a schema-declared optional
  argument the reference omits (BFCL convention), so an enum violation can still get F1 = 1.0.
  Validity catches it.
- A hallucinated tool is penalised twice: invalid (validity) and outside the allowlist
  (permission). That is AuditKIT's default policy; pass `allowed_tools=` to change it.
- AgentTune's rollout mutates message-list prompts in place (FINDINGS.md B14). The adapter
  then falls back to "the last genuine user turn ends the prompt", which is wrong only when a
  prompt ends with a folded tool result.

Tests: `python -m pytest tests` (35 tests, CPU, real AgentTune and AuditKIT code).
