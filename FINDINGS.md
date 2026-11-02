# Findings: AgentTune + AuditKIT, agentic fine-tuning path

Pre-hackathon exploration (Lexsi AI Safety & Alignment Hackathon, Track 3). Everything here
comes from reading the source and running it on CPU. Paths and line numbers:

- AgentTune `Lexsi-Labs/AgentTune` @ `d4db4c12` → paths relative to `/opt/AgentTune/src/agenttune/`
  (`pyproject.toml` and `docs/` are at the repo root).
- AuditKIT `auditkit==1.1.3` → paths relative to `site-packages/auditkit/`.

Evidence: `results/01_reward_gap.json` (with per-score reasons in `results/01_reward_scores.jsonl`)
and `results/02_handoff.json` / `results/02_trajectories.jsonl`. Re-run with
`python experiments/01_reward_gap.py` and `python experiments/02_handoff.py`.
Source map: `docs/CODE_MAP.md`.

## TL;DR

1. **The documented hybrid-PRM gap does not reproduce as documented. The real behaviour is worse.**
   The docs say `hybrid_prm_reward` gives a well-formed, a malformed and a missing tool call the
   same score (1.0). At this commit it returns **`[0.0, 1.0, 1.0]`**: the *correct* call scores
   lowest. Notebook 28's own saved output was `[0.0, 0.0, 0.0]`, which comes from a different
   failure (`asyncio.run` inside Jupyter's event loop). I reproduced that too.
2. **None of the 26 built-in rewards tells a malformed call from no call.** On every scenario I
   ran, every reward gave `malformed == no_call`. The rewards that do favour the correct call
   (`reward_tool_used`, `search_grounding_reward`, `computation_reward`, `exploration_reward`) only
   count parsed calls. They never check that a call is valid.
3. **The AgentTune → AuditKIT hand-off works.** A `trajectories.jsonl` written by AgentTune's real
   rollout loop and `trajectory_writer` loads with `episodes_from_agenttune`, and ToolCallF1,
   ToolCallValidity, RedundantToolCalls, ToolPermission, AgentLoopDetection and `reliability` all
   give sensible numbers. I found four loss/accuracy bugs on the AgentTune side of the format.

## Environment notes (not upstream bugs, but they will bite you)

- The task brief said AgentTune was "installed editable". **It was not importable.** A plain
  `pip install -e` fails here for two reasons. `pyproject.toml:83` hard-pins `torch==2.11.0`; on
  PyPI that is the multi-GB CUDA build. The CPU wheel index (`download.pytorch.org`) is also blocked
  by this sandbox's network policy (proxy 403). What I did instead:
  `pip install --no-deps -e /opt/AgentTune`, then installed every base dependency **except**
  `torch`, `torchvision`, `torchao`, `bitsandbytes`, `compressed-tensors`, `transformers`, keeping
  the preinstalled `torch 2.14.1` / `transformers 5.20.0`. Rewards, the parser, the rollout loop,
  the trajectory writer and the evaluator are pure Python and run fine this way. Training would not.
- `utils/score_logger.py:21`: every reward appends to `data/scores.jsonl` **relative to the CWD**
  unless `AGENTTUNE_SCORE_LOG` is set before import. `eval/agentic/trajectory_eval.py:489-491`
  defaults `JUDGMENTS_FILE` to a path computed from the module file, which lands inside the source
  tree or site-packages. Both experiments set these variables.

## What works

- **Rollout loop on CPU with a scripted policy** (`experiments/02_handoff.py`). A `RolloutEngine`
  subclass, in the same pattern as AgentTune's own `DemoRolloutEngine`, drives the real
  `create_rollout_fn` → `_execute_trajectory`. Tool calls are parsed, real Python tools run, and
  `Trajectory` objects come out and go through `trajectory_writer`. Parallel calls, unknown tools
  and loops all go through the real code path.
- **Tool-call parser** (`rollout_factory._extract_tool_calls`, `:1077`). It parses Qwen
  `<tool_call>` (including two parallel calls) and correctly rejects the missing-brace call.
- **`Trajectory.to_dict` / `trajectory_writer`** (`agentic/trajectory/dataset.py:53`, `:156`). It
  produces portable `{"id","name","arguments": dict}` calls, as the AuditKIT importer expects.
  `reward` is written as `null` unless `keep_reward`. That is documented and correct, because
  trainers score after the hook runs.
- **AuditKIT importer + metrics** (`agent_eval/importers.py:425`, `metrics/agent.py`). Results on
  the 7 single-trial cases:

  | case | calls | F1 | exact | validity | redundant | permission | loop-free | outcome |
  |------|------:|---:|------:|---------:|----------:|-----------:|----------:|---------|
  | good | 1 | 1.0 | 1 | 1.0 | 0.0 | 1.0 | 1 | success |
  | malformed | 0 | 0.0 | 0 | n/a | n/a | n/a | n/a | failure |
  | no_call | 0 | 0.0 | 0 | n/a | n/a | n/a | n/a | failure |
  | loop (3× same call) | 3 | 0.5 | 0 | 1.0 | 0.667 | 1.0 | **0** | success |
  | bad_args (`city=`) | 1 | 0.0 | 0 | **0.0** | 0.0 | 1.0 | 1 | failure |
  | permission (`delete_file`) | 2 | 0.667 | 0 | 0.5 | 0.0 | **0.5** | 1 | success |
  | parallel (SF + NY) | 2 | 1.0 | 1 | 1.0 | 0.0 | 1.0 | 1 | success |

  "n/a" means AuditKIT marked the metric *ineligible* (zero calls leaves nothing to validate). It
  did not return a fake 0, which is the correct behaviour. `task_completion` is reported ineligible
  (`spec.judge`) because no LLM judge is available offline. I did not fake it.
- **`reliability`** (`agent_eval/reliability.py:48`). Over 3 trials (success, success, failure) it
  reports `status=mixed`, pass@2 = 1.0, pass^2 = 0.333, and flags "independence unconfirmed". The
  runner's own aggregate uses k = 3 (pass^3 = 0.0).
- **`combine_rewards`** (`agentic/rewards/composite.py:89`). With AgentTune's actual default stack
  (`rag/trainer.py:34-35`: `search_grounding_reward + format_reward + answer_correctness_reward`,
  weights 1/0.2/1) it ranks correct (0.327) above malformed = no_call (0.236). The entire margin
  comes from `search_grounding_reward` counting one parsed call.

## Experiment 01: reward-gap results

Scenarios: `notebook_text` uses the exact completions from `docs/notebooks/28_hybrid_prm_real.ipynb`.
`controlled_text` and `controlled_messages` give all three cases the same final answer, so only the
tool-call part differs. Verdict counts over the 26 registry rewards plus the RAG default combination:

| scenario | correct > both | identical | inverted | error / short list |
|----------|---------------:|----------:|---------:|-------------------:|
| notebook_text | 5 | 18 | 1 (`hybrid_prm_reward`) | 3 (empty list) |
| controlled_text | 5 | 21 | 1 (`hybrid_prm_reward`) | 0 |
| controlled_messages | 6 | 19 | 0 | 2 (`query_reward`, `hybrid_prm_reward` raise) |

**Is the documented gap reproduced?** I report this as **not as documented**, with a stronger
negative result:

- The claim in `docs/features.md:63-65` and `docs/notebooks/local-notebook.md:102-107` is "all three
  score 1.0, identically". **Not reproduced.** The real call gives `[0.0, 1.0, 1.0]`.
- Identical scores *do* reproduce, at `[0.0, 0.0, 0.0]`, but only when the function is called from
  inside a running event loop. That is the notebook's real captured output, and it is a different
  bug (B2).
- The broader version of the gap is real: no built-in reward separates malformed from missing.

## Experiment 03: AuditKIT reward adapter vs default rewards

`src/audit_rewards` turns ToolCallValidity, ToolCallF1, ToolPermission, RedundantToolCalls and
AgentLoopDetection into REWARD_REGISTRY-style rewards, plus a confirm-before-irreversible-action
penalty. It reads AgentTune's own parsed calls and schemas (README in that directory).
`experiments/03_audit_rewards.py` scores the experiment-01 cases and 29 synthetic multi-turn
rollouts. They are produced by AgentTune's real `create_rollout_fn` with a scripted policy, and
21 of them are failure cases, each paired with a good case. Full table:
`results/03_audit_rewards.md`.

| reward | experiment-01 strict order (of 3) | failures detected (of 21) | inverted |
|---|---:|---:|---:|
| best single default (`search_grounding_reward` / `computation_reward` / `exploration_reward`) | 0 | 3 | 9 |
| `hybrid_prm_reward` (unpatched main) | 0 | 3 | 2 |
| `combine_rewards(RAG default)` | 0 | 6 | 8 |
| adapter `audit_tool_validity_reward` alone | 3 | 10 | 2 |
| adapter `combined()` (train-time, with F1) | **3** | **21** | **0** |
| adapter `combined(include_f1=False)` (reference-free) | 3 | 18 | 1 |

"Strict order" means correct > no_call > malformed. No default reward achieves it in any
scenario. The reference-free combination misses `w_wrong_tool` and `p_partial` (a valid call to
the wrong tool, or an incomplete set of calls, cannot be judged without a reference). It
inverts `i_unneeded_call` (calling a tool on a task that needs none scores above not calling,
because validity's no-call score is 0.5).

## Experiment 04: CPU GRPO smoke run

`experiments/04_grpo_smoke.py` runs `create_agentic_trainer("grpo", tools=...,
reward_funcs=AuditRewards(...).train_time())` for 20 steps on CPU (about 13 s). A pretrained
~135M model was **not** possible: the sandbox blocks the Hugging Face Hub (proxy 403) and has no
cached weights. So the policy is a randomly initialised ~90k-parameter Llama with a locally
trained tokenizer. This is a plumbing test, not a learning run. Verified: all 20 steps run, all 5
rewards are called on every step with the rollout extras (`conversations`, `tool_call_counts`,
`trajectories`, the `expected_tool_calls` column), 80 trajectories are written, and the weights
change. 3 of 80 rollouts emitted unparseable `<tool_call>` markup (validity 0.0 vs 0.5). Those
were the only steps with non-zero reward std and gradient. Three events are too few to call
that learning (`results/04_grpo_smoke.json`).

## Bugs worth upstream PRs

Ordered by how much they would hurt a training run. Each one was observed in an experiment unless
it is marked *(code reading)*.

### AgentTune

**B1. `hybrid_prm_reward` rewards *not* calling tools (inverted signal).**
`agentic/rewards/builtin_rewards/hybrid_prm.py:66` builds `TrajectoryEvaluator(model_name=...)`
**without `tool_schemas`**. In `eval/agentic/trajectory_eval.py`, `_calculate_tac` then returns
**1.0 when there are no tool calls** (`:45-46`, which also covers a malformed call that failed to
parse) and counts every real call as invalid because its tool is not in the empty schema dict
(`:63-65`). Result: valid call → 0.0, malformed → 1.0, no call → 1.0. Under GRPO this teaches the
policy to stop calling tools.
*Fix:* thread `tool_schemas` through kwargs (the rollout already builds them). Score "no call" as
neutral or penalised, not 1.0. Distinguish "unknown tool" from "no schema available".

**B2. `hybrid_prm_reward` returns all zeros inside a running event loop.**
`hybrid_prm.py:67` calls `asyncio.run(...)`. Inside Jupyter, or any async caller, that raises.
The `except` at `:68-79` swallows the error and returns `[0.0]*n`, leaving an un-awaited coroutine
behind. Reproduced in `results/01_reward_gap.json → notes.hybrid_prm_inside_running_event_loop`.
*Fix:* the same thread-pool fallback `_run_tools_parallel` already uses
(`rollout_factory.py:806-815`).

**B3. `hybrid_prm_reward(use_llm_judge=False)` still calls an LLM.**
With the default flag it sets `model_name="offline-deterministic-only"` (`hybrid_prm.py:63`).
`evaluate_batch` then calls `engine.generate_batch` unconditionally (`trajectory_eval.py:457`).
litellm rejects that model name and every row takes the "Parsing failed" fallback. The work is
wasted on every reward call (about 10 s for 3 completions here, including retries), and the
default would hit the network if the model name ever resolved.
*Fix:* skip the judge when `use_llm_judge=False` and compute TAC/ARR/LCF directly.

**B4. `answer_correctness_reward` gives 0.5 when there is no gold answer.**
`use_case.py:135` replaces a missing `answer` with `""`. Then `"" in pred` is always true at
`use_case.py:149`, so every completion gets partial credit, 0.5. Observed in `notebook_text`.
Related defaults hand out free reward with no gold: `template_structure_reward` 0.1
(`use_case.py:220-221`) and `finqa_placeholder_coverage_reward` 0.1.
*Fix:* return 0.0 (or `None`) when gold is empty.

**B5. sql.py rewards silently return a short list when `answer` is missing, and `combine_rewards` hides it.**
`sql.py:31`, `:87` and `:148` use `zip(completions, answer or [], strict=False)`, so with no
`answer` column `correctness_reward`, `query_reward` and `reward_correct_answer` return `[]`.
`combine_rewards` then iterates whatever came back (`composite.py:158`), so the missing reward
contributes 0 with no error. Verified:
`combine_rewards(["correctness_reward","reward_tool_used"])` returns `[0.5, 0.5]` instead of
failing. The comment at `composite.py:150-156` says a silent all-zero signal is exactly what it
wants to avoid.
*Fix:* `strict=True`, or raise in `combine_rewards` when `len(scores) != len(completions)`.

**B6. `query_reward` crashes on AgentTune's own tool-call format.**
`sql.py:94` does `call["function"]["arguments"].get("sql_command", "")`. AgentTune's parser
normalises arguments to a JSON **string** (`rollout_factory.py:1038`, see also the docstring at
`:355`), so on message-list completions this raises
`AttributeError: 'str' object has no attribute 'get'` (`controlled_messages`). It is not a
`TypeError`, so `combine_rewards` does not retry it.
*Fix:* `json.loads` the arguments when they are a string.

**B7. Parallel calls to the same tool lose all but the last result in the step record.**
`rollout_factory.py:2375`: `observation=str(dict(tool_results))`. `tool_results` is a list of
`(name, result)` pairs, so `dict()` keys by tool name. Observed: the parallel SF + NY step records
`{'get_weather': "{'location': 'New York', 'temp_c': 22}"}`, and the SF result is gone. The
conversation in `metadata` still has both. Anything that reads `Step.observation`
(`to_trl_format`, AuditKIT's importer, the heal loop's audit records) sees wrong data.
*Fix:* store a list (or JSON) of per-call results with their call ids.

**B8. A tool whose body raises `TypeError` is executed twice.**
`rollout_factory.py:772-778`: any `TypeError` triggers a retry meant for "the args were a JSON
string", even when `kwargs` is already a dict. The async path at `:785-801` has the same retry.
Observed: a side-effecting `charge_card` tool ran **2×** (`results/02_handoff.json →
double_invocation_check`). This is a safety issue for real-world tools such as payments, writes
and emails.
*Fix:* retry only when `isinstance(kwargs, str)`, before the first call.

**B9. AgentTune cannot read back its own `trajectories.jsonl` as tool-call training data.**
The rollout records `Step.action={"tool_calls": [...]}` (`rollout_factory.py:2374`), and `to_dict`
keeps that shape (`dataset.py:64-66`). But `Trajectory.to_trl_format` only recognises a flat
`action["name"]` (`dataset.py:76`). So `TrajectoryDataset.from_jsonl(path)[i]` turns every tool
step into a plain assistant message whose content is the observation. Observed for the "good"
record: roles `['assistant','assistant']`, with no `tool_calls` and no `tool` message.
*Fix:* handle the `tool_calls` list form in `to_trl_format`.

**B10. Per-step metadata is silently dropped from the written record.**
`rollout_factory.py:2378` and `:2562` set `step.metadata = {...}` (per-call `tool_results`,
`tool_failure_count`, `is_terminal`) as an ad-hoc attribute. `Step` has no such dataclass field
(`dataset.py:34`), so `asdict` in `to_dict` (`dataset.py:58`) drops it. Observed: written step keys
are only `step_number, state, action, observation, thought, reward`.
*Fix:* add `metadata: dict = field(default_factory=dict)` to `Step`.

**B11. Malformed tool calls are invisible downstream.**
When parsing fails, the turn becomes the terminal step with `action={}` (`rollout_factory.py:2555`)
and `tool_call_count=0`. The attempted call survives only as free text in `thought`/`observation`.
AuditKIT and every count-based reward therefore treat it exactly like "no call" (see TL;DR 2).
*Fix:* record a `tool_parse_failures` counter (or the unparsed span) in `Trajectory.metadata`,
and expose it to rewards the same way `tool_call_counts` is.

**B12. Count-based rewards reward any parsed call, including wrong ones** *(code reading plus
experiment 02)*. `search_grounding_reward`, `exploration_reward` and `computation_reward`
(`use_case.py:10`, `:320`, `:246`) and `reward_tool_used` (`sql.py:193`) read only
`tool_call_counts`, which the rollout increments for unknown tools and bad arguments too
(`rollout_factory.py:2303`). `computation_reward`'s docstring says "run_python", but it counts
any tool. The FinQA grounding rewards are pure substring counts over the completion
(`finqa.py:176`, `:214`): writing the word "calculator" earns 0.2. All of these can be gamed by a
GRPO policy.
*Suggested addition:* a schema-validity reward using the schemas the rollout already has. It
would be AgentTune's equivalent of AuditKIT's `ToolCallValidity`.

**B13. Docs are wrong about the PRM** *(docs)*. `docs/features.md:63-65`,
`docs/notebooks/local-notebook.md:102-107` and the notebook 28 markdown all say "all score 1.0".
The notebook's own saved output shows 0.0 ×3, and the current code gives 0.0 / 1.0 / 1.0.

**B14. The rollout mutates message-list prompts in place.** `rollout_factory.py:86-89`:
`_build_conversation` returns the caller's prompt list itself when it is already a list of
messages (and has no system prompt to prepend). The rollout then appends every generated turn to
it (`:2250`, `:2347`, `:2435`). Observed in experiment 03: all 9 message-list prompts came back
holding the full transcript (`results/03_audit_rewards.json → notes.prompt_list_mutated_by_rollout`).
In GRPO the same objects are the `prompts` TRL later passes to reward functions, and possibly the
dataset rows. So a reward that reads `prompts` sees the completion inside the prompt, and the
prompt length no longer marks where generation started. `src/audit_rewards` works around it; see
its README.
*Fix:* `return list(prompt)` (or `[dict(m) for m in prompt]`) in `_build_conversation`.

### AuditKIT

**A1. `call_result_pairing` reported as observed, but results are not paired per call.**
`agent_eval/importers.py:474` sets `call_result_pairing=observed` whenever
`metadata.conversation` exists. The `tool_result` events, however, are built from the turn-level
`step.observation` blob (`importers.py:456-460`), never from the conversation. Combined with B7,
this means the parallel episode claims observed pairing while its only result event holds just
the NY result. The conversation that has both results (with `tool_call_id`s) is ignored.
*Fix:* build `tool_result` events from `metadata.conversation` when it is present. Otherwise
downgrade the flag to `inferred`.

**A2. Design note (not a bug).** AuditKIT's handling of zero-call episodes is principled
(ineligible, not 0). But it means AuditKIT cannot see B11 (malformed calls) at all, because the
information is not in the file. Fixing B11 on the AgentTune side would let AuditKIT add a
"parse-failure rate" metric.

### Safety-relevant observation (design, not a bug)

The rollout executes **any tool it was given**, with no allow/deny policy at execution time.
`delete_file('/etc/hosts')` was invoked during the rollout (the tool here was a no-op stub).
AuditKIT's `ToolPermission` catches it only after the fact (0.5). For agentic RL on real tools,
a permission gate in `_execute_trajectory` is needed: the policy explores, and during training it
will try every tool it has.

## Candidate PRs, smallest first

1. B8: guard the `TypeError` retry with `isinstance(kwargs, str)` (2 places; about 4 lines).
2. B5: `strict=True` in the three sql.py zips, plus a length check in `combine_rewards`.
3. B4: empty gold → 0.0 in `answer_correctness_reward`.
4. B6: `json.loads` string arguments in `query_reward`.
5. B10 + B7: `Step.metadata` field, and a per-call observation list.
6. B9: `to_trl_format` support for `{"tool_calls": [...]}` actions.
7. B2 + B3: no event-loop crash, no LLM call when `use_llm_judge=False`.
8. B1: pass `tool_schemas` into `hybrid_prm_reward`, and fix TAC's no-call = 1.0.
9. A1 (AuditKIT): pair results from `metadata.conversation`.
10. B14: copy message-list prompts in `_build_conversation` (one line).
