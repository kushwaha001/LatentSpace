# Code map: AgentTune + AuditKIT (agentic fine-tuning path)

Pinned versions this map was read against:

- **AgentTune** `Lexsi-Labs/AgentTune` @ `d4db4c12d52342227c4113a70183d0f64aa2f345` (2026-10-07), at `/opt/AgentTune`.
  Paths below are relative to `/opt/AgentTune/src/agenttune/`.
- **AuditKIT** `auditkit==1.1.3` (pip). Paths below are relative to the installed package
  `site-packages/auditkit/`.

Line numbers are for those exact versions.

---

## 1. Reward registry (AgentTune)

**`REWARD_REGISTRY: dict[str, Callable]`** — `agentic/rewards/builtin_rewards/__init__.py:17-56`.
Re-exported as `agenttune.agentic.REWARD_REGISTRY` (`agentic/__init__.py:1`).
Built by `from .sql import *` then `from .use_case import *` (lines 14-15), so the bare name
`format_reward` resolves to **use_case.py's** version; sql.py's and finqa.py's are reachable only
under the prefixed keys.

Two calling conventions coexist:

- **sql.py**: `fn(completions, answer=None, **kwargs)` — `completions` may be a string or a list of
  chat messages; several only work on message lists.
- **use_case.py / finqa.py / PRM / judge**: `fn(prompts, completions, ..., **kwargs)` — `prompts`
  is a **required positional/keyword** argument; `str(completion)` is scanned with regexes.

All 26 entries (registry key → `file:line` signature — what it scores):

| # | Key | Defined at | What it scores |
|---|-----|-----------|----------------|
| 1 | `correctness_reward` | `builtin_rewards/sql.py:28` `(completions, answer=None, **kw)` | `*yes*`/`*no*` in last message vs gold: +0.6 right, −1.0 wrong, −0.5 absent. |
| 2 | `structure_reward` | `sql.py:53` `(completions, **kw)` | Message-list shape: tool call + tool response (+0.05/+0.1), call w/o response −0.15, else 0. Strings → always 0. |
| 3 | `query_reward` | `sql.py:84` `(completions, answer=None, **kw)` | SQL strategy: penalises >3 queries, `LIMIT 1`, missing `WHERE`; ±2.0 by hits vs yes/no gold. Reads `arguments.get("sql_command")` → assumes dict args. |
| 4 | `reward_correct_answer` | `sql.py:145` `(completions, answer=None, **kw)` | Numeric match of final assistant text vs gold: 1.5 exact, 0.5 within 1%. |
| 5 | `reward_tool_used` | `sql.py:193` `(completions, **kw)` | 1.0 if `tool_call_counts[i]>0` or any `role=="tool"` message (also checks `conversations`). |
| 6 | `reward_concise_answer` | `sql.py:223` `(completions, **kw)` | 0.3 if final text ≤ 80 words. |
| 7 | `format_reward` | `use_case.py:75` `(prompts, completions, **kw)` | 0.1 if `<answer>…</answer>` present. (Shadows `sql.py:247`.) |
| 8 | `search_grounding_reward` | `use_case.py:10` `(prompts, completions, tool_call_counts=None, **kw)` | Tiered on `tool_call_counts`: 0/0.2/0.3/0.4 for 0/1/2/3+ calls; all 0 if counts missing. |
| 9 | `message_id_citation_reward` | `use_case.py:45` `(prompts, completions, **kw)` | 0.2 if an email address or `message_id` appears. |
| 10 | `answer_format_reward` | `use_case.py:97` `(prompts, completions, **kw)` | 0.1 if `<answer>` holds ≥10 chars. |
| 11 | `answer_correctness_reward` | `use_case.py:123` `(prompts, completions, answer=None, **kw)` | `<answer>` (or whole text) vs gold: 1.0 exact, 0.5 substring either way. |
| 12 | `placeholder_coverage_reward` | `use_case.py:162` `(prompts, completions, answer=None, **kw)` | Fraction of gold numbers present as substrings: 1.0/0.5/0.2/0. |
| 13 | `template_structure_reward` | `use_case.py:207` `(prompts, completions, answer=None, **kw)` | Fraction of gold `**header**`s preserved ×0.3; flat 0.1 when gold has none. |
| 14 | `computation_reward` | `use_case.py:246` `(prompts, completions, tool_call_counts=None, **kw)` | 0.2 for 1 call, 0.3 for 2+ (counts **any** tool, despite the "run_python" docstring). |
| 15 | `numerical_match_reward` | `use_case.py:271` `(prompts, completions, answer=None, **kw)` | First number in pred vs first in gold: 1.0 exact, 0.5 within 5%. |
| 16 | `exploration_reward` | `use_case.py:320` `(prompts, completions, tool_call_counts=None, **kw)` | Tiered on call count: 0/0.1/0.2/0.3. |
| 17 | `hybrid_prm_reward` | `builtin_rewards/hybrid_prm.py:9` `(prompts: list[str], completions: list[str], **kw) -> list[float]` | Parses calls with `rollout_factory._extract_tool_calls`, runs `TrajectoryEvaluator`; `clamp(TAC − 0.2·ARR − 0.1·LCF, 0, 1)` (+judge terms if `use_llm_judge`). |
| 18 | `distilled_judge_reward` | `builtin_rewards/distilled_judge.py:76` `(prompts, completions, **kw)` | HF `text-classification` pipeline from `distilled_judge_path` (default `"custom_reward_model"`); 0.0 if it can't load. Instance cached on the function object. |
| 19 | `tool_chain_reward` | `use_case.py:353` `(prompts, completions, **kw)` | +0.1 per substring `list_dir`/`read_file`/`run_python`/`write_file` in the text. |
| 20 | `summary_written_reward` | `use_case.py:377` `(prompts, completions, sample_dir=None, **kw)` | 0.2 if `sample_dir` has a file beyond 4 hard-coded names. |
| 21 | `sql_format_reward` | `sql.py:247` `(completions, **kw)` | 0.2 if final text has markdown (`**`, `#`, lists, fences). |
| 22 | `finqa_format_reward` | `builtin_rewards/finqa.py:124` `(prompts, completions, **kw)` | 0.1 if `<answer>` holds ≥3 chars. |
| 23 | `finqa_sql_grounding_reward` | `finqa.py:153` `(prompts, completions, **kw)` | Counts substring `query_finqa_tables` in text: 0/0.2/0.3. |
| 24 | `finqa_calculator_grounding_reward` | `finqa.py:196` `(prompts, completions, **kw)` | 0.2 if the substring `calculator` appears anywhere. |
| 25 | `finqa_placeholder_coverage_reward` | `finqa.py:231` `(prompts, completions, answer=None, **kw)` | Unit-aware numeric coverage (2% tol) of gold numbers in `<answer>`: 0.4/0.2/0.1/0. |
| 26 | `finqa_answer_correctness_reward` | `finqa.py:318` `(prompts, completions, answer=None, **kw)` | Any gold number within 2% → 1.0, 10% → 0.5; else text exact/substring. |

Every reward also calls `agenttune.utils.score_logger.log_score(...)`, which appends to
`$AGENTTUNE_SCORE_LOG` (default `data/scores.jsonl` **relative to CWD**, read once at import —
`utils/score_logger.py:21`).

"Default" reward stacks that actually exist in the code:

- **RAG trainer**: `rag/trainer.py:34-35` —
  `["search_grounding_reward", "format_reward", "answer_correctness_reward"]` weights `[1.0, 0.2, 1.0]`.
- **Agentic GRPO docstring example**: `backends/trl/agentic/grpo/agentic_grpo.py:34` —
  `[correctness_reward, structure_reward, query_reward]`.
- Agentic GRPO itself has **no** default: `setup_trainer` raises if `reward_funcs` is missing
  (`agentic_grpo.py:383-390`).

### Process-reward internals used by `hybrid_prm_reward`

`eval/agentic/trajectory_eval.py`:

- `class TrajectoryEvaluator` (`:15`), `__init__(self, engine=None, model_name="groq/llama-3.3-70b-versatile", api_base=None, tool_schemas=None)` (`:27`). Always builds an `APIEngine` (litellm).
- `_calculate_tac(self, trajectory) -> float` (`:39`): **returns 1.0 when there are no tool calls** (`:45-46`); a call whose name isn't in `self.tool_schemas` counts as invalid (`:63-65`).
- `_calculate_arr` (`:83`) duplicates ratio; `_calculate_lcf` (`:140`) 3-in-a-row repeats.
- `async evaluate_batch(self, trajectories, scoring_mode="absolute")` (`:443`): always calls the LLM judge first; on failure every row takes the "Parsing failed" fallback (`:534-560`) that still fills TAC/ARR/LCF.

## 2. `combine_rewards`

`agentic/rewards/composite.py:89` — `combine_rewards(reward_funcs, weights=None) -> Callable`.

- `_resolve(fn)` (`:69`): str → `REWARD_REGISTRY[fn]`, callable passthrough.
- Weights normalised to sum 1 (`:130-137`).
- Returned `_combined(completions, **kwargs)` (`:145`): tries `fn(completions=..., **kwargs)`, on
  `TypeError` retries `fn(completions, **kwargs)` (`:148-157`); `None` scores count as 0.0.
- Called by agentic GRPO at `backends/trl/agentic/grpo/agentic_grpo.py:391`; `reward_weights` is
  consumed there and popped from `GRPOConfig` (`:395`).

## 3. Agentic GRPO rollout

`backends/trl/agentic/grpo/agentic_grpo.py`:

- `class TrlAgenticGrpo` (`:169`); `setup_data` (`:235`), `setup_trainer` (`:375`), `train` (`:612`).
- `_build_grpo_rollout_fn(rollout_engine=None, rollout_backend=None, tools=None, max_steps=20, system_prompt=None, engine_kwargs=None, **rollout_kwargs) -> Callable` (`:132`) → thin wrapper over `create_rollout_fn`.
- Wiring (`:452-473`): when in agentic mode with `tools`, builds the rollout fn with
  `on_trajectory_end=trajectory_writer(<output_dir>/<trajectories_file>)` and forwards
  `tools_fallback_prompt` / `tool_result_format` if passed.
- Patches `GRPOTrainer._generate_single_turn` to route generation through the rollout fn (comment at `:475+`).

`agentic/rollout_engines/rollout_factory.py` (2829 lines):

- `create_rollout_fn(rollout_engine=None, rollout_backend=None, model=None, tokenizer=None, model_path=None, api_provider=None, api_model=None, api_key=None, api_base_url=None, engine_kwargs=None, tools=None, max_steps=20, reward_fn=None, custom_rollout_fn=None, pre_step_hook=None, post_step_hook=None, on_trajectory_end=None, enable_thinking=False, system_prompt=None, async_rollouts=False, force_final_answer=False, force_action_on_stall=False, context_length=None, tools_fallback_prompt=DEFAULT_TOOLS_FALLBACK_PROMPT, tool_result_format=DEFAULT_TOOL_RESULT_FORMAT) -> Callable` (`:1549`); inner `rollout_fn(prompts, trainer=None, *args, **gen_kwargs) -> dict` (`:1619`); calls `on_trajectory_end(t)` per trajectory (`:1588-1590`, `:1697-1699`).
- `_run_trajectories(...)` (`:1714`) — sync or async fan-out.
- `_execute_trajectory(prompt, tools, tools_enabled, engine, max_steps, pre_step_hook, post_step_hook, gen_kwargs, system_prompt=None, trainer=None, enable_thinking=False, force_final_answer=False, force_action_on_stall=False, context_length=None, tools_fallback_prompt=..., tool_result_format=...) -> Trajectory` (`:1815`) — the multi-turn generate → parse → execute tools loop.
  - No-tools path builds a one-step `Trajectory` (`:2143-2170`).
  - Each tool turn records `Step(action={"tool_calls": tool_calls}, observation=str(dict(tool_results)), thought=raw_output)` (`:2371-2377`); the per-step `step.metadata` is set as an ad-hoc attribute (`:2378`).
  - Terminal step `action={}` (`:2555`); returned `Trajectory.metadata` carries `conversation`, `tool_call_count`, `tool_failure_count`, `retrieved_chunk_ids`, `answer`, token ids/mask (`:2579-2597`).
- `_format_for_grpo(trajectories, prompts, tools_enabled) -> dict` (`:1407`) — what TRL gets back; extra keys `conversations`, `tool_call_counts`, `retrieved_chunk_ids`, `trajectories` are forwarded by TRL to reward functions as kwargs.

### Tool-call parsers (there are three)

1. **`_extract_tool_calls(completion_msg, raw_text) -> list[dict] | None`** — `rollout_factory.py:1077`. The one the rollout loop uses (`:2245`, `:2286`, `:2430`, `:2468`) and that `hybrid_prm_reward` imports. Order: native `tool_calls` → Cohere (action / legacy / json, `:1091-1101`) → `<tool_call>`/`<toolcall>`-stripped first JSON object (`:1103+`) → then Llama `<function=…>`, Mistral `[TOOL_CALLS]`, DeepSeek, GLM (`:1226`), Functionary regexes (defined `:947-1035`). Normalised by `_normalise_tool_calls` (`:1038`) to OpenAI shape with **string** `arguments`.
2. **`extract_tool_calls(completion_msg, raw_text)`** — `rollout_engines/tool_call_parse.py:475`, with `extract_tool_calls_from_text(raw_text)` (`:447`) trying `parse_xml_function_calls` (`:167`), `_parse_tool_call_tagged` (`:257`), `_parse_cohere_action` (`:350`), `_parse_cohere_json` (`:366`), `_parse_harmony_tool_calls` (`:313`), `_parse_mistral` (`:306`), `_parse_react` (`:403`), `_parse_cohere_legacy_action` (`:387`), `_parse_fenced_after_name` (`:425`), then a JSON-blob fallback.
3. **`_parse_tool_call(text)`** — `decide/closed_loop/training_example_generator.py:17`, for synthesized corrections in the heal loop.

## 4. `tools_fallback_prompt` (Cohere / Tiny Aya)

- Constant **`DEFAULT_TOOLS_FALLBACK_PROMPT`** — `rollout_factory.py:537-544`. Tells the model to reply with
  `[{"tool_name": "<name>", "parameters": {...}}]` (Cohere/Command-R list format).
- Detection: `_template_renders_tools(tokenizer, schemas) -> bool` (`:517`) renders the chat template with and without `tools`.
- Applied in `_execute_trajectory` at `:2173-2206`: if the template ignores `tools`, warns, formats
  `{tools}` with the JSON schemas (with a `.replace` fallback when `.format` fails on stray braces),
  prepends to/creates the system message, and sets `fold_tool_results = True`.
- Tool results are then folded into user turns by `fold_tool_messages_into_user(conversation, tool_result_format)` (`:107`) using `DEFAULT_TOOL_RESULT_FORMAT` (`:102`).
- Parsed back by `_parse_cohere_json` (`tool_call_parse.py:366`, docstring mentions Tiny Aya) and the Cohere branch of `_extract_tool_calls` (`rollout_factory.py:1091-1098`).
- Passing `tools_fallback_prompt=None` makes a tool-ignoring template a hard `ValueError` (`:2181-2186`).

## 5. `trajectories.jsonl` writer

`agentic/trajectory/dataset.py`:

- `@dataclass Step(step_number: int, state: str, action: dict, observation: str, thought: str | None = None, reward: float | None = None)` (`:34`).
- `@dataclass Trajectory(task: str, steps: list[Step], reward: float = 0.0, trajectory_id: str = uuid4, final_response: str = "", logprobs: Any = None, metadata: dict = {})` (`:44`).
- `Trajectory.to_dict(self) -> dict` (`:53`): the portable record. Drops `logprobs` and the
  training-only metadata keys `prompt_ids`, `completion_ids`, `tool_mask`, `completions_list`
  (`:9`); rewrites each `action["tool_calls"]` via `_portable_call` (`:12`) to
  `{"id", "name", "arguments": dict}` (unparseable args stay a raw string).
- `Trajectory.to_trl_format(self)` (`:69`) — reads `step.action["name"]` (flat form).
- `TrajectoryDataset.from_jsonl(cls, path)` (`:137`).
- **`trajectory_writer(path, keep_reward=False) -> Callable[[Trajectory], None]`** (`:156`):
  `on_trajectory_end` hook; appends one `json.dumps(to_dict(), default=str)` line; writes
  `reward: null` unless `keep_reward`.
- Wired into agentic GRPO (`backends/trl/agentic/grpo/agentic_grpo.py:454-463`) and RLOO
  (`backends/trl/agentic/rloo/agentic_rloo.py:363-375`) with default filename `trajectories.jsonl`.

## 6. Self-healing closed loop

Detect → classify → generate corrective examples → (gated) retrain → deploy gate.

- **Detection**: `Project.heal(self, *, detector=None, max_revisits=3)` — `agentic/project.py:292`.
  Projects `EventLog.to_audit_records()` to a temp JSONL and runs
  `decide/closed_loop/failure_detector.py:12` `FailureDetector(judge_threshold=0.6, max_revisits=3, offset_file=".audit_read_offset")`, `.scan_audit_log(file_path) -> Iterator[Failure]` (`:38`).
- **Contracts**: `decide/closed_loop/contracts.py` — `Failure` (`:6`), `ClassifiedFailure` (`:16`), `TrainingExample` (`:24`, `has_preference_pair` `:61`, `derive_preference_from_completions` `:69`), `AgenticEvalResult` (`:116`).
- **Orchestrator**: `agentic/heal_loop.py`
  - `class SelfHealLoop` (`:69`), `__init__(self, classifier, generator, *, trainer_factory=None, retraining_trigger=None, evaluate_deploy=None)` (`:103`).
  - `run(self, failures, **trainer_kwargs) -> dict` (`:118`); `run_on(self, project, *, detector=None, max_revisits=3, **trainer_kwargs) -> dict` (`:168`).
  - `build_dataset(examples) -> list[dict]` (`:58`) → `{prompt, chosen, rejected}` rows.
  - `as_sync_classifier(classifier)` (`:182`), `as_sync_generator(generator)` (`:188`) wrap the async stages with `asyncio.run`.
- **LLM stages**: `FailureClassifier(model_name="gpt-4o-mini", api_base=None)` — `failure_classifier.py:9`, `async classify_batch` (`:81`); `TrainingExampleGenerator` — `training_example_generator.py:53`, `async generate_batch` (`:218`).
- **Gating / deploy**: `RetrainingTrigger` — `retraining_trigger.py:320` (with `TrainingBuffer` `:193`, `RewardDriftTracker` `:90`); `deployment_gate.py`.
- **Continuous version**: `FullClosedLoop` — `decide/closed_loop/full_loop.py:148` (`ingest_once` `:246`, `tick` `:280`, `_evaluate` `:360` via `TrajectoryEvaluator`, `run_until` `:405`); `ClosedLoopRunner` — `closed_loop_runner.py:62`.

---

## 7. AuditKIT `agent_eval` importers

`agent_eval/importers.py` (stdlib-only, never imports AgentTune):

- **`episodes_from_agenttune(path: str) -> list[AgentEpisode]`** (`:515`). Accepts a file or a run
  folder (all `*.jsonl`, plus `lexsi_provenance.json`). EventLog JSONL → `episode_from_eventlog`;
  otherwise `_load_agenttune_pairs` (`:280`) + `_detect_format` (`:312`) dispatch to:
  - `_episode_trajectory(raw, sample)` (`:425`) — rollout `trajectories.jsonl` (keys `steps`/`trajectory_id`/`final_response`). Emits `reasoning` (from `thought`), `tool_call` (from `action.tool_calls`), and a turn-level `tool_result` (from `observation`) per step. Coverage: `call_result_pairing` only if `metadata.conversation` exists; `timestamps`/`final_state` unavailable.
  - `_episode_trace` (`:356`) — TraceLogger `trace.jsonl` / `TrajectoryStore.export_jsonl`.
  - `_episode_report` (`:322`) — `run_eval` report (names only).
  - `_episode_run_eval_row` (`:492`) — a task row (not scored).
- `episode_from_eventlog(log, *, source_id=None)` (`:572`), `episode_from_openai_messages(messages, *, mode="recorded", ...)` (`:185`), `episode_from_sample(sample, *, mode="recorded", ...)` (`:220`).
- `inspect_agenttune(path) -> dict` (`:681`); `supported_scorers_from_coverage(coverage) -> list[str]` (`:49`).
- Types: `AgentEvent` (`agent_eval/types.py:314`), `AgentEpisode` (`:347`; `turns()` `:379`, `calls_unnamed()` `:401`, `to_trace()` `:412`), `AgentCase` (`:544`; `allowed_tools`, `reference_turns`, `outcome`, `metadata`).

Scoring recorded episodes — `agent_eval/runner.py`:

- `KNOWN_SCORERS` (`:41`); `eligibility(scorer, case, episode) -> Optional[str]` (`:150`) names the missing field instead of scoring 0.
- `case_from_episode(episode, *, outcome=None, case_id=None) -> AgentCase` (`:256`).
- `AgentEvalRunner.run(spec: AgentEvalSpec) -> AgentEvalResult` (`:303`); per-case `_score` (`:707`) builds `Sample` via `_sample_for(case)` (`:226`) and `context={"trace": episode.to_trace()}`.
- `rescore(episodes, scorers, *, cases=None, judge=None) -> AgentEvalResult` (`:1029`).

## 8. AuditKIT `metrics/agent.py`

All implement `score(self, sample: Sample, output: str, context: Any = None)`; calls come from
`auditkit.trace.predicted_turns(output, context)` (`trace.py:595`).

| Metric | Registry key | Location | Needs | Output |
|--------|-------------|----------|-------|--------|
| `ToolCallF1(arg_mode="exact", arg_match=None)` | `tool_call_f1` | `metrics/agent.py:273` (base `_ReferenceToolMetric` `:194`) | `sample.expected_tool_calls` | `tool_call_precision/recall/f1/exact`; multiset, bipartite matching. |
| `ToolCallValidity()` | `tool_call_validity` | `:687` | `sample.tools` (schemas) | `1 − invalid/n_calls`; checks existence, required, extra args, JSON types, enums, recursively. No calls → no score. |
| `RedundantToolCalls()` | `redundant_tool_calls` | `:715` | trace only | duplicate fraction (minimise). No calls → no score. |
| `ToolPermission(denied_tools=None)` | `tool_permission` | `:978` | allowlist (`metadata.allowed_tools` or `sample.tools`) / denylist | fraction of calls within policy. |
| `AgentLoopDetection(repetition_threshold=3, similarity_threshold=0.85)` | `agent_loop_detection` | `:905` | trace only | 1.0 loop-free / 0.0 loop (repeat, stagnation, call-graph cycle). |
| `TaskCompletion(*, judge_model=None, ...)` | `task_completion` | `:762` | an LLM judge model | complete 1.0 / partial 0.5 / failed 0.0; unparseable → raises. |

Also present: `TrajectoryMatch` (`:373`), `ParallelToolCalls` (`:405`), `ToolSelectionJudge` (`:1062`).

## 9. AuditKIT `agent_eval/reliability.py`

- `aggregate_verdict(verdicts: list[str]) -> tuple[str, str]` (`:28`): majority over decided trials; tie → `unknown`.
- `reliability(verdicts, *, k=None, independent=False, trial_ids=None, n_requested=None) -> dict` (`:48`): `pass_at_k`, `all_k` (pass^k), `success_rate`, `consistency`, `variance`, `status ∈ {reliable, unreliable, mixed, unknown}`; flags non-independent trials and duplicate `trial_id`s; ≤1 decided trial → `unknown`.
- Used by `AgentEvalRunner._aggregate` (`runner.py:332`) when `spec.trials > 1`.
