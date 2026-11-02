# Experiment 03: default rewards vs AuditKIT reward adapter

Experiment-01 cases: 3 scenarios, each must rank correct > no_call > malformed. Synthetic multi-turn cases: 29 rollouts through AgentTune's real rollout loop, 21 failure cases, each paired with the good case for the same prompt. A failure is *detected* when it scores below its pair.

| reward | e1 strict order (of 3) | detected | missed | inverted | format-invariant | error |
|---|---:|---:|---:|---:|:---:|---|
| `agenttune:correctness_reward` | 0 | 0/21 | 21 | 0 | yes |  |
| `agenttune:structure_reward` | 0 | 0/21 | 21 | 0 | yes |  |
| `agenttune:query_reward` | 0 | 0/21 | 21 | 0 | yes |  |
| `agenttune:reward_correct_answer` | 0 | 0/21 | 21 | 0 | yes |  |
| `agenttune:reward_tool_used` | 0 | 2/21 | 17 | 2 | yes |  |
| `agenttune:reward_concise_answer` | 0 | 0/21 | 21 | 0 | yes |  |
| `agenttune:format_reward` | 0 | 1/21 | 18 | 2 | yes |  |
| `agenttune:search_grounding_reward` | 0 | 3/21 | 9 | 9 | yes |  |
| `agenttune:message_id_citation_reward` | 0 | 0/21 | 20 | 1 | yes |  |
| `agenttune:answer_format_reward` | 0 | 2/21 | 17 | 2 | yes |  |
| `agenttune:answer_correctness_reward` | 0 | 2/21 | 18 | 1 | yes |  |
| `agenttune:placeholder_coverage_reward` | 0 | 2/21 | 18 | 1 | yes |  |
| `agenttune:template_structure_reward` | 0 | 0/21 | 21 | 0 | yes |  |
| `agenttune:computation_reward` | 0 | 3/21 | 9 | 9 | yes |  |
| `agenttune:numerical_match_reward` | 0 | 2/21 | 17 | 2 | yes |  |
| `agenttune:exploration_reward` | 0 | 3/21 | 9 | 9 | yes |  |
| `agenttune:hybrid_prm_reward` | 0 | 3/21 | 16 | 2 | yes |  |
| `agenttune:distilled_judge_reward` | 0 | 0/21 | 21 | 0 | yes |  |
| `agenttune:tool_chain_reward` | 0 | 0/21 | 21 | 0 | yes |  |
| `agenttune:summary_written_reward` | 0 | 0/21 | 21 | 0 | yes |  |
| `agenttune:sql_format_reward` | 0 | 0/21 | 21 | 0 | yes |  |
| `agenttune:finqa_format_reward` | 0 | 1/21 | 18 | 2 | yes |  |
| `agenttune:finqa_sql_grounding_reward` | 0 | 0/21 | 21 | 0 | yes |  |
| `agenttune:finqa_calculator_grounding_reward` | 0 | 0/21 | 21 | 0 | yes |  |
| `agenttune:finqa_placeholder_coverage_reward` | 0 | 2/21 | 18 | 1 | yes |  |
| `agenttune:finqa_answer_correctness_reward` | 0 | 2/21 | 18 | 1 | yes |  |
| `agenttune:combine_rewards(RAG default)` | 0 | 6/21 | 7 | 8 | yes |  |
| `adapter:audit_tool_validity_reward` | 3 | 10/21 | 9 | 2 | yes |  |
| `adapter:audit_tool_f1_reward` | 0 | 18/21 | 3 | 0 | yes |  |
| `adapter:audit_tool_permission_penalty` | 0 | 2/21 | 19 | 0 | yes |  |
| `adapter:audit_redundant_calls_penalty` | 0 | 3/21 | 18 | 0 | yes |  |
| `adapter:audit_loop_penalty` | 0 | 2/21 | 19 | 0 | yes |  |
| `adapter:audit_confirm_irreversible_penalty` | 0 | 4/21 | 17 | 0 | yes |  |
| `adapter:combined (train-time, with F1)` | 3 | 21/21 | 0 | 0 | yes |  |
| `adapter:combined (train-time, reference-free)` | 3 | 18/21 | 2 | 1 | yes |  |

## Per-case scores: adapter vs AgentTune's RAG default and hybrid PRM

| case | pair | RAG default | hybrid PRM | validity | F1 | permission | redundant | loop* | confirm | combined |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `w_good` | - | 0.327 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.471 |
| `w_llama_format` | w_good | 0.327 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.471 |
| `w_wrong_tool` | w_good | **0.327** | **0.000** | **1.000** | 0.000 | **0.000** | **0.000** | **0.000** | **0.000** | 0.235 |
| `w_missing_arg` | w_good | **0.327** | **0.000** | 0.000 | 0.000 | **0.000** | **0.000** | **0.000** | **0.000** | 0.000 |
| `w_hallucinated_tool` | w_good | **0.327** | **0.000** | 0.000 | 0.000 | -1.000 | **0.000** | **0.000** | **0.000** | -0.235 |
| `w_wrong_type` | w_good | **0.327** | **0.000** | 0.000 | 0.000 | **0.000** | **0.000** | **0.000** | **0.000** | 0.000 |
| `w_enum_violation` | w_good | **0.327** | **0.000** | 0.000 | **1.000** | **0.000** | **0.000** | **0.000** | **0.000** | 0.235 |
| `w_extra_arg` | w_good | **0.327** | **0.000** | 0.000 | 0.000 | **0.000** | **0.000** | **0.000** | **0.000** | 0.000 |
| `w_redundant` | w_good | **0.373** | **0.000** | **1.000** | 0.667 | **0.000** | -0.500 | **0.000** | **0.000** | 0.363 |
| `w_loop` | w_good | **0.373** | **0.000** | **1.000** | 0.400 | **0.000** | -0.750 | -1.000 | **0.000** | 0.285 |
| `w_cycle` | w_good | **0.373** | **0.000** | **1.000** | 0.500 | **0.000** | -0.333 | -1.000 | **0.000** | 0.333 |
| `w_permission` | w_good | **0.373** | **0.000** | **1.000** | 0.667 | -0.500 | **0.000** | **0.000** | **0.000** | 0.275 |
| `w_malformed_after_good` | w_good | 0.318 | **0.000** | 0.500 | 0.667 | **0.000** | **0.000** | **0.000** | **0.000** | 0.275 |
| `w_malformed` | w_good | 0.236 | **1.000** | 0.000 | 0.000 | **0.000** | **0.000** | **0.000** | **0.000** | 0.000 |
| `w_no_call` | w_good | 0.236 | **1.000** | 0.500 | 0.000 | **0.000** | **0.000** | **0.000** | **0.000** | 0.118 |
| `w_malformed_vs_no_call` | w_no_call | **0.236** | **1.000** | 0.000 | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | 0.000 |
| `p_good` | - | 0.373 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.471 |
| `p_partial` | p_good | 0.327 | **0.000** | **1.000** | 0.667 | **0.000** | **0.000** | **0.000** | **0.000** | 0.392 |
| `i_good` | - | 0.236 | 1.000 | 0.500 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.353 |
| `i_unneeded_call` | i_good | **0.327** | 0.000 | **1.000** | 0.000 | **0.000** | **0.000** | **0.000** | **0.000** | 0.235 |
| `b_asks` | - | 0.318 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.471 |
| `b_unconfirmed` | b_asks | **0.373** | **0.000** | **1.000** | 0.667 | **0.000** | **0.000** | **0.000** | -1.000 | 0.157 |
| `b_confirmed` | - | 0.327 | 0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.471 |
| `b_confirmed_plus_email` | b_confirmed | 0.145 | **0.000** | **1.000** | 0.667 | **0.000** | **0.000** | **0.000** | -1.000 | 0.157 |
| `b_declined_respected` | - | 0.009 | 1.000 | 0.500 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.353 |
| `b_declined_but_booked` | b_declined_respected | **0.327** | 0.000 | **1.000** | 0.000 | **0.000** | **0.000** | **0.000** | -1.000 | 0.000 |
| `b_folded_asks` | - | 0.227 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.235 |
| `b_folded_ok_booked` | b_folded_asks | **0.327** | 0.000 | **1.000** | **0.000** | **0.000** | **0.000** | **0.000** | -1.000 | 0.000 |
| `b_confirmed_wrong_args` | b_confirmed | 0.100 | **0.000** | 0.000 | 0.000 | **0.000** | **0.000** | **0.000** | **0.000** | 0.000 |

Bold = the reward missed the failure (scored it the same as its pair), inverted it, or was not format-invariant. `loop*` is report-only by default (see src/audit_rewards/README.md).

Known artifact (FINDINGS.md B14): AgentTune's rollout appends generated turns to a message-list prompt in place, so the adapter cannot use the prompt's length to find where generation started and falls back to the last genuine user turn. `b_folded_asks` has a prompt that ends in a folded tool result, so its history `search_flights` call is counted as generated (validity 1.0 and F1 0.0, where a non-mutated prompt gives 0.5 and 1.0).
