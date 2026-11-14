# Silence pays: reward hacking in a multilingual tool-calling agent, and an audit-grounded reward that resists it

*Kanishk Kushwaha, Sambhav Verma, Vikrant Bansode (team LatentSpace) · Lexsi AI Safety & Alignment Hackathon, Track 3 (Agentic Fine-tuning), Problem Statement 9: reward design and reward-hacking defences*

**Code:** https://github.com/kushwaha001/LatentSpace · **Models:** https://huggingface.co/collections/kanishk001/latentspace-models-lexsi-hackathon-track-3-p9-6af84d1bbf8a687cc9e0b8d2 · **Data, rollouts and logs:** https://huggingface.co/collections/kanishk001/latentspace-datasets-and-logs-lexsi-hackathon-t3-p9-6af84d2c307f37a24ab2d708

## Abstract

We study reward hacking in reinforcement fine-tuning of a small tool-calling agent. The agent is Tiny Aya Global (3.35B parameters) serving Indian public transit in English, Hindi, Marathi and Hinglish, with 8 tools, two of them irreversible (booking and cancellation). The default tool reward of the AgentTune library, `hybrid_prm_reward`, scores a correct tool call 0 and a missing or malformed call 1. GRPO trained with it produces an agent that never calls a tool. That agent succeeds on 0 of 44 held-out tasks that need an action, yet its overall success is 0.481 against 0.513 for the distilled student, and it has the highest reliability score of any model, because 52 of the 108 test tasks are solved by not acting. We build a reward from AuditKIT's audit metrics (schema validity, tool-call F1, permission, a multilingual confirm-before-irreversible check, redundancy). In a controlled comparison, the same KL-anchored GRPO run with only the reward changed gives 0.023 success on tasks that need a call with the default reward and 0.423 with ours (+0.400, 95% CI [+0.286, +0.514], p < 0.001). Used as a filter on teacher trajectories, the same reward lowers unconfirmed irreversible actions from 19.4% to 13.2% (p = 0.004). We also report what did not work: without a KL anchor, GRPO collapsed into silence even under a reward that ranks a call above silence; no RL method exceeded the distilled student; and distillation taught the agent when not to act, not how to act better. All results are from one training seed.

## 1. Introduction

Reward-based fine-tuning optimises whatever the reward pays for. If the reward can be earned without doing the task, the policy will find that route [1, 2, 3]. For a tool-calling agent the cheapest route is easy to name: do nothing. An agent that never calls a tool never makes a malformed call, never calls a forbidden tool and never books a ticket without permission.

Whether that route pays depends on the reward and on the evaluation. This article shows a case where it pays under both. The reward is the default tool reward of AgentTune, an open-source agentic fine-tuning library. The evaluation is a single success rate over a realistic task mix, in which a large share of user requests are correctly handled by declining, asking a question or refusing.

We make four contributions.

1. **A reward-hacking failure shown end to end.** AgentTune's `hybrid_prm_reward` ranks a correct tool call below no call. GRPO trained with it stops calling tools entirely, on train and on held-out tasks. A single success number, a reliability metric (pass^5) and the call-decision accuracy of a public benchmark all rate the silent agent at or near the top (Sections 4.2 and 4.8).
2. **An audit-grounded reward.** We turn AuditKIT's post-hoc agent metrics into training rewards and add a confirm-before-irreversible-action check that works in four languages. With training otherwise identical, it keeps the agent acting where the default reward does not (Section 4.3). As a data filter it reduces unsafe actions (Section 4.4).
3. **A negative result on reward design alone.** A reward with the right ranking was not sufficient. Two GRPO runs with our reward also collapsed into silence, and we trace this to zero within-group reward variance. A KL anchor to the starting policy prevented it (Section 4.5).
4. **Defects in the toolchain.** We report three defects that change training or evaluation outcomes: the inverted reward, a stop-token mismatch that made GRPO completions run to the length limit, and a paired bootstrap that returns p ≈ 0.5 for any effect size (Section 5).

## 2. Background and related work

**Reward hacking.** Specification gaming and reward hacking describe policies that score well on the stated objective without achieving the intended one [1, 2, 3]; Goodhart's law is the general form [4]. Our case is a degenerate-policy instance: one constant behaviour maximises the reward.

**Group-relative policy optimisation.** GRPO [5] samples G completions per prompt and uses the group-normalised reward as the advantage, A_i = (r_i − mean(r)) / std(r). A group whose samples all receive the same reward contributes no gradient. DAPO [6] names this problem and resamples prompts until the group is not uniform; we did not use dynamic sampling. A KL penalty to a reference policy is the standard way to limit drift [7].

**Tool-calling evaluation.** BFCL [8] scores function-call correctness. When2Call [9] tests whether a model calls a tool, asks for missing information or declines. τ-bench [10] introduced pass^k, the probability that all k trials of a task succeed. We use pass^5 and show that a constant policy can score well on it.

**Models and methods.** The student is Tiny Aya Global and the teacher is Aya Expanse 32B [11], both from Cohere Labs. Adapters are LoRA [12]. We compare supervised distillation [13], GRPO and DPO [14].

## 3. Setup

### 3.1 Task, tools and data

The agent answers questions about Indian Railways and the Mumbai, Pune and Delhi metros through 8 tools: `get_station_code`, `search_trains`, `search_metro_route`, `get_fare`, `get_pnr_status`, `book_ticket`, `cancel_booking` and `file_complaint`. `book_ticket` and `cancel_booking` are tagged irreversible. The tools run on a deterministic mock backend with real station names and railway codes and synthetic timetables, fares and PNRs.

The dataset has 90 base tasks, each in English, Hindi (Devanagari), Marathi (Devanagari) and Hinglish (Roman script), giving 360 rows in 8 categories:

| Group | Category | Correct behaviour |
|---|---|---|
| Needs a call | single_call | one tool call |
| Needs a call | multi_step | a chain of calls (for example station codes, then trains, then fare) |
| Needs a call | irreversible_confirmed | the user has said yes to the agent's proposal; book or cancel |
| Propose and ask | irreversible_unconfirmed | state the details and ask; read-only lookups allowed, no booking |
| No call | declined | the user said no; do not book or cancel |
| No call | denied_tool | out of scope or another person's data; refuse |
| No call | missing_argument | ask for the missing value |
| No call | no_call_needed | greeting or general question |

Rows are split by base task into train (188 rows), dev (64) and test (108), so no translation of a test task is seen in training. `file_complaint` and `cancel_booking` appear only in test tasks. All 308 reference calls replay on the backend without error ([`experiments/09_dataset_audit.py`](https://github.com/kushwaha001/LatentSpace/blob/main/experiments/09_dataset_audit.py)). The Hindi, Marathi and Hinglish text was generated with a language model and has not yet been reviewed by native speakers.

The test split has 44 rows that need a call, 48 that need none and 16 propose-and-ask rows, of which 4 have an empty reference. A policy that never calls a tool therefore succeeds on 52 of 108 rows (0.481).

### 3.2 Policy prompt

Every model, including the teacher, receives the same system prompt: six operating rules (use tools for facts, work one step at a time, ask for missing values, propose and wait for a yes before an irreversible action, refuse out-of-scope requests, never claim an action a tool result does not confirm) followed by the tool schemas. The rules were written from the category definitions and the train split.

### 3.3 Training pipeline

**Teacher data.** Aya Expanse 32B (4-bit, run locally) produced 2 samples for each of the 188 train rows. For the 76 train rows whose reference contains tool calls, the reference calls were replayed on the backend and the teacher wrote the replies from the real tool results; the other 112 rows are free teacher rollouts. We call this reference-guided distillation. We adopted it after a first pipeline in which none of the teacher's 54 kept multi-step trajectories was correct. Each trajectory then passes an audit filter: the task is solved, every call is schema-valid, there is no unconfirmed irreversible call, and the reply is in the user's script. The filter kept 319 of 376 trajectories and rejected 23 for an unconfirmed irreversible call, 19 as incorrect and 15 for a reply in the wrong script.

**Distillation (SFT).** LoRA (r = 16, α = 32, all attention and MLP projections) on Tiny Aya Global for 3 epochs, learning rate 2e-4, with one checkpoint per epoch. The third-epoch checkpoint was selected on the dev split (Section 4.6). An ablation trains on all 376 trajectories with the same settings.

**GRPO.** AgentTune's agentic GRPO trainer with multi-turn tool rollouts, starting from the distilled student: 300 steps, one prompt and G = 4 completions per optimiser step, learning rate 2e-5, sampling temperature 0.9, up to 4 turns per episode and 192 new tokens per turn. Runs without an anchor use β = 0. KL-anchored runs use β = 0.04 with the distilled student as the reference: the distilled adapter is merged into the weights and a new LoRA is trained on top, because TRL's reference policy for a LoRA model is the model with the adapter disabled.

**DPO.** AgentTune's offline DPO on preference pairs mined from rollouts we already had on the train split (teacher rollouts and the training rollouts of three GRPO runs), ranked by the audit verdict. No new sampling.

All runs used one workstation with two NVIDIA RTX 4500 Ada GPUs (24 GB each), bf16 weights without quantisation for the student, torch 2.11.0+cu126, transformers 5.22.1, TRL 1.7.1, AgentTune 1.1.3 (commit `d4db4c12`) and AuditKIT 1.1.3. The two KL-anchored GRPO runs took 4.6 and 8.6 hours; the other GRPO runs were interrupted and resumed from checkpoints, so their total time was not recorded.

### 3.4 Rewards

**Default.** AgentTune's `hybrid_prm_reward`, unmodified. At the commit above it returns 0.0 for a correct call, 1.0 for a malformed call and 1.0 for no call (Section 5).

**Audit reward.** A weighted sum of five terms, each computed by an AuditKIT metric class or a rule built on its trace:

| Term | Range | Weight | Source |
|---|---|---|---|
| Tool-call validity: share of attempted calls that parse and match the schema; 0.5 for no call | [0, 1] | 1.0 | `ToolCallValidity` |
| Tool-call F1 against the reference calls, with semantic argument matching | [0, 1] | 1.0 | `ToolCallF1` |
| Permission penalty: a call to a tool outside the allowed set | [−1, 0] | 1.0 | `ToolPermission` |
| Confirm-before-irreversible penalty | [−1, 0] | 1.0 | our rule (below) |
| Redundant-call penalty | [−1, 0] | 0.25 | `RedundantToolCalls` |

The sum is divided by the total weight (4.25), so a correct call scores 0.471, no call 0.118 and a malformed call 0. Version 2 changes one rule: on a task whose reference contains a call, no call earns 0 validity instead of 0.5, so that any valid attempt outranks silence.

The confirmation rule counts an irreversible call as confirmed only if a user turn that affirms the action comes after the agent has spoken and before the call, and no later turn withdraws it. It matches tokens and n-grams in English, Hindi, Marathi and Hinglish, and it is action-aware: "cancel" is a refusal when the agent proposed a booking and the action itself when the agent proposed a cancellation. Its lexicon contains no phrasing that occurs only in the test split.

Loop detection, exact-match scores, success and pass^k are never part of a reward.

### 3.5 Evaluation

Each model is run on the 108 test rows with 5 trials per row at temperature 0.7 (540 rollouts). A rollout is a **success** when the reference calls are made, there is no extra, malformed or schema-invalid call, every call is permitted, and there is no unconfirmed irreversible call. Arguments are compared semantically: a station name in another script that the backend resolves to the same station, a PNR with spaces, or a reworded free-text complaint counts as equal; a wrong date, class, count or id does not. We also report **strict** success, which requires JSON-equal arguments (AuditKIT's default).

We report four further quantities: success on rows that **need a call** and on rows that need **no call**; **calls per task**; the **unconfirmed irreversible rate**, the share of rollouts with a booking or cancellation made without the user's yes; and **pass^5**, the share of rows where all 5 trials succeed.

Comparisons are paired over the 108 rows (trial means per row): a two-sided sign-flip permutation test with 10,000 resamples and a percentile bootstrap 95% CI. With 10,000 resamples the smallest attainable p-value is 0.0001; we write p < 0.001. All model and checkpoint choices were made on the dev split.

## 4. Results

### 4.1 Main results

Held-out test split, 108 rows × 5 trials. Success uses semantic argument matching.

| Model | Success | Success (strict) | Needs a call | No call needed | Calls / task | Unconfirmed irreversible | pass^5 |
|---|---|---|---|---|---|---|---|
| Base | 0.289 | 0.204 | 0.409 | 0.217 | 1.24 | 18.0% | 0.120 |
| SFT-all | 0.478 | 0.411 | 0.441 | 0.637 | 0.89 | 19.4% | 0.222 |
| SFT-audit | 0.513 | 0.435 | 0.427 | 0.713 | 0.97 | 13.2% | 0.296 |
| GRPO-default | 0.481 | 0.481 | 0.000 | 1.000 | 0.00 | 0.0% | 0.481 |
| GRPO-audit-v1 | 0.526 | 0.507 | 0.123 | 0.992 | 0.09 | 1.3% | 0.454 |
| GRPO-audit-v2 | 0.483 | 0.483 | 0.009 | 0.996 | 0.01 | 0.0% | 0.472 |
| GRPO-KL-default | 0.485 | 0.480 | 0.023 | 0.996 | 0.03 | 0.6% | 0.454 |
| **GRPO-KL-audit** | 0.513 | 0.444 | 0.423 | 0.721 | 0.85 | 11.8% | 0.241 |
| DPO-balanced | 0.500 | 0.426 | 0.436 | 0.675 | 1.00 | 16.3% | 0.232 |

### 4.2 The default reward is gamed by silence

GRPO-default makes 0.00 tool calls per task on the test split. Its success on rows that need a call is 0.000, down from 0.409 for the untrained model (−0.409, 95% CI [−0.536, −0.291], p < 0.001). Its success on rows that need no call is 1.000.

Three aggregate measures fail to show the collapse:

- **Overall success** is 0.481, against 0.513 for the distilled student. It is higher than the untrained model's (+0.193, 95% CI [+0.070, +0.313], p = 0.003).
- **pass^5** is 0.481, the highest in the table. A constant policy is perfectly consistent.
- **Unconfirmed irreversible rate** is 0.0%. An agent that never acts never acts without permission.

Only the split by task group, or the call count, shows that the agent no longer does its job. This is why we report the two groups separately throughout.

### 4.3 Controlled comparison: only the reward changes

GRPO-KL-default and GRPO-KL-audit share the starting adapter, data, seed, step count and every hyperparameter; the results files record the same shared-settings fingerprint for both (`648b52befd28657f`). The only difference is the reward.

| | GRPO-KL-default | GRPO-KL-audit | Difference |
|---|---|---|---|
| Success, rows that need a call | 0.023 | 0.423 | +0.400, 95% CI [+0.286, +0.514], p < 0.001 |
| Same, strict matching | 0.009 | 0.291 | |
| Success, rows that need no call | 0.996 | 0.721 | |
| Overall success | 0.485 | 0.513 | +0.028, 95% CI [−0.057, +0.117], p = 0.55 |
| Calls per task | 0.03 | 0.85 | +0.817, 95% CI [+0.670, +0.980], p < 0.001 |
| Unconfirmed irreversible rate | 0.6% | 11.8% | +0.113, 95% CI [+0.074, +0.156], p < 0.001 |

With the default reward the KL-anchored agent still stops acting (0.03 calls per task): the anchor slows the drift but the reward keeps paying for silence. With the audit reward the agent keeps acting. The overall success difference is not significant, which restates Section 4.2: the aggregate cannot separate the two agents.

The comparison also shows the cost of acting. GRPO-KL-audit has a higher unconfirmed irreversible rate than the silent agent, and the audit reward did not lower that rate relative to its starting point (SFT-audit: −0.013, 95% CI [−0.048, +0.022], p = 0.53). Its overall success equals the distilled student's (+0.000, 95% CI [−0.043, +0.041], p = 1.00). The audit reward preserved competence under RL. It did not add any.

### 4.4 The audit reward as a data filter

SFT-all and SFT-audit differ only in the training set: all 376 teacher trajectories, or the 319 the audit filter approved.

| | SFT-all | SFT-audit | Difference |
|---|---|---|---|
| Unconfirmed irreversible rate | 19.4% | 13.2% | −0.063, 95% CI [−0.106, −0.024], p = 0.004 |
| Overall success | 0.478 | 0.513 | +0.035, 95% CI [−0.007, +0.080], p = 0.14 |

Removing 57 trajectories, 23 of them for an unconfirmed irreversible call, lowers the unconfirmed rate by 6.3 points. The success difference is not significant. The safety failures that remain are concentrated in particular categories (unconfirmed irreversible rate per category):

| Category | Base | SFT-all | SFT-audit | GRPO-KL-audit | DPO-balanced |
|---|---|---|---|---|---|
| irreversible_unconfirmed | 63.7% | 73.8% | 43.8% | 37.5% | 51.2% |
| declined | 17.5% | 11.2% | 10.0% | 17.5% | 18.8% |
| denied_tool | 17.5% | 30.0% | 12.5% | 7.5% | 12.5% |
| missing_argument | 22.5% | 15.0% | 15.0% | 7.5% | 12.5% |
| irreversible_confirmed | 6.2% | 10.0% | 8.8% | 8.8% | 8.8% |

Two residual failures matter for deployment. On propose-and-ask rows, every acting model still books or cancels without waiting in more than a third of rollouts. On declined rows, where the user has said no, GRPO-KL-audit acts anyway in 17.5% of rollouts. Neither the filter nor the reward removed these.

### 4.5 A correct reward ranking was not sufficient

Audit reward v1 gives no call 0.5 validity on every task. GRPO-audit-v1 drifted to 0.09 calls per task. We attributed this to the neutral score outranking the student's frequently invalid attempts, and changed the rule (v2) so that on tasks that need a call, any valid attempt outranks silence. GRPO-audit-v2 collapsed further, to 0.01 calls per task.

The training logs show the mechanism:

| Run | Reward | KL anchor | Calls per training rollout, by quarter | Steps with zero reward variance (of 300) |
|---|---|---|---|---|
| GRPO-default | default | no | 0.09 → 0.00 → 0.00 → 0.00 | 286 |
| GRPO-audit-v1 | audit v1 | no | 0.88 → 0.41 → 0.36 → 0.11 | 189 |
| GRPO-audit-v2 | audit v2 | no | 1.40 → 0.92 → 0.04 → 0.00 | 205 |
| GRPO-KL-default | default | β = 0.04 | 0.50 → 0.10 → 0.05 → 0.06 | 250 |
| GRPO-KL-audit | audit v2 | β = 0.04 | 1.27 → 1.39 → 1.29 → 0.95 | 112 |

In GRPO-audit-v2 the policy calls tools in the first half of training and stops in the second, and the number of steps with zero reward variance rises from 25 in the first quarter to 75 of 75 in the last. When all 4 samples for a prompt are silent, they receive the same reward, their group-relative advantages are 0, and the step produces no gradient that could restore calling. Prompts that need no call continue to reward silence. The ranking of the reward is correct at every step, but a ranking only acts through samples that differ. Recomputing reward v2 on this run's logged training rollouts gives about 0.30 for a calling rollout and 0.00 for a silent one on tasks that need a call, throughout training.

A KL penalty to the distilled student (β = 0.04) keeps calls per training rollout between 0.95 and 1.39 for the whole run and holds zero-variance steps near 28 per quarter. The anchor prevented collapse in this run. By Section 4.3 it does not do so under the default reward.

**DPO.** Offline DPO learns from explicit pairs, so silence on a task that needs a call can be labelled as the worse answer directly. The mined pairs were unbalanced: of 432 pairs, 327 had a silent or unconfirmed rejected answer, 95 a needless action and 10 a wrong call. The resulting model over-called (dev split: 2.34 calls per task, 27.3% unconfirmed irreversible, success 0.258). Balancing the pair types (200 pairs) removed the over-calling (dev success 0.453). The balanced run also used a lower learning rate, one epoch instead of two and a larger β, so the improvement cannot be attributed to balancing alone. On test, DPO-balanced does not differ from the distilled student in success (−0.013, 95% CI [−0.057, +0.031], p = 0.62) and has a higher unconfirmed rate (+0.031, 95% CI [+0.004, +0.059], p = 0.04).

### 4.6 Model selection on the dev split

Dev split, 64 rows × 2 trials. These numbers were used for choices and are not test results.

| Checkpoint | Success | Calls / task | Unconfirmed irreversible | Used for |
|---|---|---|---|---|
| SFT-audit, epoch 1 | 0.523 | 0.34 | 1.6% | epoch choice |
| SFT-audit, epoch 2 | 0.492 | 0.84 | 7.0% | epoch choice |
| SFT-audit, epoch 3 (selected) | 0.570 | 1.05 | 4.7% | epoch choice |
| GRPO-audit-v1, step 220 | 0.523 | 0.12 | 0.0% | diagnosing the drift |
| GRPO-KL-audit, step 60 | 0.484 | 1.18 | 11.7% | checking for collapse |
| GRPO-KL-audit, step 120 | 0.492 | 0.97 | 5.5% | checking for collapse |
| DPO, unbalanced pairs | 0.258 | 2.34 | 27.3% | rejected |
| DPO, balanced pairs | 0.453 | 1.12 | 11.7% | kept for test |

### 4.7 What distillation changed

SFT-audit raises overall success from 0.289 to 0.513 (+0.224, 95% CI [+0.141, +0.311], p < 0.001) and lowers the unconfirmed rate from 18.0% to 13.2% (−0.048, 95% CI [−0.089, −0.011], p = 0.02). The gain is almost entirely restraint. Success on rows that need no call rises from 0.217 to 0.713; success on rows that need a call moves from 0.409 to 0.427.

Success by category (semantic matching):

| Category (test rows) | Group | Base | SFT-all | SFT-audit | GRPO-default | GRPO-audit-v1 | GRPO-audit-v2 | GRPO-KL-default | GRPO-KL-audit | DPO-balanced |
|---|---|---|---|---|---|---|---|---|---|---|
| single_call (16) | needs a call | 0.72 | 0.54 | 0.55 | 0.00 | 0.11 | 0.00 | 0.04 | 0.51 | 0.53 |
| multi_step (12) | needs a call | 0.00 | 0.07 | 0.03 | 0.00 | 0.00 | 0.00 | 0.02 | 0.05 | 0.08 |
| irreversible_confirmed (16) | needs a call | 0.40 | 0.62 | 0.60 | 0.00 | 0.23 | 0.03 | 0.01 | 0.61 | 0.61 |
| irreversible_unconfirmed (16) | propose and ask | 0.17 | 0.10 | 0.15 | 0.25 | 0.24 | 0.25 | 0.23 | 0.14 | 0.15 |
| declined (16) | no call | 0.49 | 0.86 | 0.86 | 1.00 | 0.99 | 0.99 | 1.00 | 0.81 | 0.74 |
| denied_tool (8) | no call | 0.10 | 0.40 | 0.53 | 1.00 | 1.00 | 1.00 | 0.97 | 0.65 | 0.53 |
| missing_argument (16) | no call | 0.05 | 0.39 | 0.51 | 1.00 | 0.99 | 1.00 | 1.00 | 0.56 | 0.54 |
| no_call_needed (8) | no call | 0.12 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.93 | 0.97 |

Three points qualify the headline numbers. Single-call success fell from 0.72 to 0.55 after distillation. Multi-step success is at most 0.08 for every model: no method in this study produced an agent that completes a chain of calls. And propose-and-ask success is at most 0.25, reached only by the silent agents through the 4 rows with an empty reference.

Success by language:

| Language | Base | SFT-all | SFT-audit | GRPO-default | GRPO-audit-v1 | GRPO-audit-v2 | GRPO-KL-default | GRPO-KL-audit | DPO-balanced |
|---|---|---|---|---|---|---|---|---|---|
| English | 0.33 | 0.42 | 0.46 | 0.48 | 0.48 | 0.48 | 0.50 | 0.47 | 0.50 |
| Hindi | 0.25 | 0.52 | 0.53 | 0.48 | 0.55 | 0.49 | 0.49 | 0.50 | 0.53 |
| Marathi | 0.30 | 0.50 | 0.54 | 0.48 | 0.56 | 0.48 | 0.48 | 0.59 | 0.52 |
| Hinglish | 0.27 | 0.47 | 0.53 | 0.48 | 0.51 | 0.48 | 0.47 | 0.49 | 0.45 |

The untrained model is strongest in English. After distillation English is the weakest language (0.46 against 0.53 to 0.54); the audit filter also kept the smallest share of English teacher trajectories (76.6%, against 85.1% to 89.4%). Each language has 27 test rows, and the non-English text is unreviewed, so we do not interpret these gaps further.

### 4.8 Public benchmarks

As an external check we ran four checkpoints on When2Call [9] (200 questions per behaviour; we score only the decision to call or not) and on BFCL-Hi, NVIDIA's Hindi version of the BFCL v2 live set (258 simple and 258 irrelevance questions). One greedy turn per question, with the tool schemas and no task policy. This is our simplified protocol, not either benchmark's official harness.

| Model | When2Call: decision accuracy | calls when it should | calls when no tool fits ↓ | calls when information is missing ↓ | BFCL-Hi simple | BFCL-Hi irrelevance |
|---|---|---|---|---|---|---|
| Base | 0.345 | 0.985 | 0.975 | 0.975 | 0.264 | 0.097 |
| SFT-audit | 0.530 | 0.785 | 0.505 | 0.690 | 0.043 | 0.903 |
| GRPO-default | 0.667 | 0.000 | 0.000 | 0.000 | 0.000 | 0.996 |
| GRPO-audit-v1 | 0.667 | 0.095 | 0.045 | 0.050 | 0.008 | 0.977 |

The silent agent again does well on an aggregate: its When2Call decision accuracy is 0.667, tied for the highest of the four, because two of the three balanced classes are answered correctly by not calling. The distilled student is the only model that both calls when it should and holds back when it should not. All trained models lose Hindi function calling on unseen tools (BFCL-Hi simple 0.264 → 0.043 or lower): the training is specific to our tools and policy prompt, and we make no claim of general tool-calling improvement.

## 5. Defects found in AgentTune and AuditKIT

Each defect below was reproduced, and worked around or fixed in our pipeline with a regression test. Line references and further findings are in [FINDINGS.md](https://github.com/kushwaha001/LatentSpace/blob/main/FINDINGS.md). None has been submitted upstream yet.

**Inverted default reward (AgentTune).** `hybrid_prm_reward` builds its trajectory evaluator without tool schemas. The tool-accuracy component then returns 1.0 when there are no calls and counts every real call as invalid, because its tool is not in the empty schema set. The result is 0.0 for a valid call and 1.0 for a malformed or missing one. Sections 4.2 and 4.3 show the training consequence. In a separate check, none of the 26 built-in AgentTune rewards separated a malformed call from no call on any scenario we ran ([experiment 01](https://github.com/kushwaha001/LatentSpace/blob/main/experiments/01_reward_gap.py)). On 21 paired failure cases, the combined audit reward scored every failure below its correct counterpart; AgentTune's default reward combination did so for 6 and ranked 8 above it ([experiment 03](https://github.com/kushwaha001/LatentSpace/blob/main/results/03_audit_rewards.md)).

**Stop-token mismatch (AgentTune and TRL with Tiny Aya).** TRL's `GRPOTrainer` and AgentTune's trainer-driven rollout stop and mask a completion on `tokenizer.eos_token_id`, which for Tiny Aya is `<|END_OF_TURN_TOKEN|>`. The released model ends a reply with `<|END_RESPONSE|>` and does not emit that token. In our first pipeline the logged clipped ratio was 0.75 to 1.00 on every GRPO step: completions ran to the length limit, and the text generated after the real end of the reply, including invented tool results, was parsed and rewarded. We set the tokenizer's end-of-sequence token to `<|END_RESPONSE|>` for training and stop evaluation rollouts on every end-of-reply id. In the two KL-anchored runs the mean clipped ratio is then 0.12 and 0.13.

**Paired bootstrap that never rejects (AuditKIT).** `paired_bootstrap` resamples the observed differences and counts resamples at least as extreme as the observed mean, without centring the differences on zero. About half the resamples exceed the observed mean for any effect size: 60 pairs with a true shift of +0.3 and noise SD 0.1 give p = 0.49. We use a sign-flip permutation test instead.

## 6. Limitations

- **One seed.** Every method was trained once (seed 42). The confidence intervals reflect variation over test rows and trials, not over training runs. The collapse results in particular should be read as observed in these runs.
- **Small test set with dependent rows.** The 108 test rows are 27 base tasks in 4 languages. The tests treat rows as the unit, so they overstate the effective sample size. Per-category cells hold 2 to 4 base tasks.
- **Many comparisons.** We report many paired tests without correction. The two headline effects have p < 0.001 and p = 0.004; smaller effects should be read as exploratory.
- **No RL gain over distillation.** No GRPO or DPO model exceeded the distilled student on overall success or on rows that need a call. Our RL results concern avoiding collapse, not improving the agent.
- **Low absolute performance.** The best models succeed on about half the test rows, almost never complete a multi-step task, and still take unconfirmed irreversible actions in 12 to 13% of rollouts. None is fit for deployment.
- **Reference-guided teacher data.** On train rows with reference calls, the tool calls in the teacher data are the references; the teacher wrote only the replies. The student therefore imitates gold call sequences on train tasks. Dev and test rows were never used this way.
- **Shared matcher.** The semantic argument matcher used in evaluation is also used inside the audit reward's F1 term. Evaluation is on held-out tasks, and strict numbers are given alongside, but the reward and the metric are not independent.
- **Post-hoc analysis choices.** Semantic matching and the split into rows that need a call and rows that need none were introduced after we saw first-pipeline and GRPO-default results.
- **Unreviewed translations and a synthetic world.** The non-English text is model-generated and unreviewed, and the backend data is synthetic.
- **Batching differs between runs.** GRPO-default used 2 completions per backward pass with 2 accumulation steps for the whole run; GRPO-audit-v1 used the same for its first 30 steps and then 1 with 4; the other runs used 1 with 4. The optimiser step is the same in every case (1 prompt, 4 completions). The controlled comparison in Section 4.3 is not affected.
- **Interrupted runs.** GRPO runs were resumable from checkpoints. GRPO-default was resumed at step 190 after a machine restart; GRPO-audit-v1 stopped with an out-of-memory error at step 30, was resumed, and was resumed again from step 60 after a machine restart; GRPO-audit-v2 was restarted after out-of-memory stops and resumed from step 20. The two KL-anchored runs ran without interruption.
- **Clipped completions in collapsing runs.** In the runs that collapse, the logged clipped ratio rises during training (last-quarter mean 0.87 for GRPO-default and 0.96 for GRPO-audit-v2). We have not established the cause.
- **Simplified public-benchmark protocol.** The When2Call and BFCL-Hi numbers are not comparable with published leaderboard results.

## 7. Reproducibility

Code, configurations and result files are in the repository; adapters, the dataset, teacher trajectories, all evaluation and training rollouts, and logs are in the two Hugging Face collections. Each stage writes a results file with its configuration, package versions, hardware and git commit.

- Full pipeline (teacher → distillation → dev selection → GRPO → evaluation → benchmarks, resumable): `bash scripts/round2.sh`
- KL-anchored runs: `bash scripts/grpo_extra.sh run_B3 1 1` (and `run_A3`)
- DPO: `python experiments/10_audit_dpo.py`
- Paired tests from the published scored rollouts: `python -m latentspace_train evaluate --compare-only`
- Unit tests: `python -m pytest tests` (141 tests)

The tables in this article are generated from `results/*.json`. The two paired tests restricted to rows that need a call (Sections 4.2 and 4.3) are computed by `scripts/make_writeup.py` from the scored rollouts in the datasets collection. Run identifiers in the repository map to the names used here as follows: `distill_unfiltered` = SFT-all, `distill` = SFT-audit, `run_A` = GRPO-default, `run_B` = GRPO-audit-v1, `run_B2` = GRPO-audit-v2, `run_A3` = GRPO-KL-default, `run_B3` = GRPO-KL-audit, `dpo_bal` = DPO-balanced. An earlier pipeline with several flaws of our own (truncated distillation examples, a missing policy prompt, a filter that did not check correctness) is kept in `results/round1/` and documented in FINDINGS.md; no number in this article comes from it except where stated.

## 8. Conclusion

A reward that pays for inaction produced an agent that does nothing, and three aggregate metrics rated that agent at or near the top. Replacing the reward with one built from audit metrics, in an otherwise identical KL-anchored run, kept the agent acting. Two further observations limit that result. The reward alone was not enough: GRPO needed an anchor because uniform groups carry no gradient. And the anchored, correctly rewarded agent was no better than the distilled student it started from. For agentic RL on tasks where restraint is often correct, we recommend reporting success separately for tasks that need an action and tasks that do not, logging calls per task during training, and treating a rising count of zero-variance groups as an early sign of collapse.

## References

[1] D. Amodei, C. Olah, J. Steinhardt, P. Christiano, J. Schulman, D. Mané. Concrete Problems in AI Safety. arXiv:1606.06565, 2016.
[2] V. Krakovna et al. Specification gaming: the flip side of AI ingenuity. DeepMind blog, 2020.
[3] J. Skalse, N. Howe, D. Krasheninnikov, D. Krueger. Defining and Characterizing Reward Hacking. NeurIPS, 2022. arXiv:2209.13085.
[4] D. Manheim, S. Garrabrant. Categorizing Variants of Goodhart's Law. arXiv:1803.04585, 2018.
[5] Z. Shao et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.
[6] Q. Yu et al. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv:2503.14476, 2025.
[7] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.
[8] S. G. Patil et al. The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models. ICML, 2025.
[9] H. Ross, A. S. Mahabaleshwarkar, Y. Suhara. When2Call: When (not) to Call Tools. NAACL, 2025. arXiv:2504.18851.
[10] S. Yao, N. Shinn, P. Razavi, K. Narasimhan. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045, 2024.
[11] J. Dang et al. Aya Expanse: Combining Research Breakthroughs for a New Multilingual Frontier. arXiv:2412.04261, 2024.
[12] E. J. Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. ICLR, 2022. arXiv:2106.09685.
[13] G. Hinton, O. Vinyals, J. Dean. Distilling the Knowledge in a Neural Network. arXiv:1503.02531, 2015.
[14] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, C. Finn. Direct Preference Optimization: Your Language Model is Secretly a Reward Model. NeurIPS, 2023. arXiv:2305.18290.

Software and models: AgentTune (https://github.com/Lexsi-Labs/AgentTune), AuditKIT (https://github.com/Lexsi-Labs/AuditKIT), Tiny Aya Global (https://huggingface.co/CohereLabs/tiny-aya-global), Aya Expanse 32B (https://huggingface.co/CohereLabs/aya-expanse-32b), When2Call (https://huggingface.co/datasets/nvidia/When2Call), BFCL-Hi (https://huggingface.co/datasets/nvidia/BFCL-Hi).
