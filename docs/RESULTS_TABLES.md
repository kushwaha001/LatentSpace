# Silence pays: reward hacking in a multilingual tool-calling agent, and an audit-grounded reward that resists it

**Lexsi AI Safety & Alignment Hackathon · Track 3 (Agentic Fine-tuning) · Problem 9: Reward design and reward-hacking defences** (also: reliable tool calling, agentic distillation, multilingual)
**Repos:** AgentTune, AuditKIT · **Models:** Tiny Aya Global (student), Aya Expanse 32B (teacher)
**Code:** https://github.com/kushwaha001/LatentSpace · **Models collection:** https://huggingface.co/collections/kanishk001/latentspace-models-lexsi-hackathon-track-3-p9-6af84d1bbf8a687cc9e0b8d2 · **Datasets, rollouts and logs collection:** https://huggingface.co/collections/kanishk001/latentspace-datasets-and-logs-lexsi-hackathon-t3-p9-6af84d2c307f37a24ab2d708

## TL;DR
We train a small multilingual tool-calling agent for Indian public transit (Tiny Aya Global, 3.35B; English, Hindi, Marathi, Hinglish; 8 tools including irreversible booking and cancellation on a deterministic mock backend) and study reward hacking in agentic RL with AgentTune and AuditKIT.

AgentTune's default tool reward (`hybrid_prm_reward`) scores a correct tool call 0 and no call 1. Trained with it, GRPO learns to never call a tool: 0.00 calls per task and 0% success on every task that needs an action on the held-out test split, while its overall success (0.481) stays close to the best model (0.513), because about half of realistic requests are correctly answered by not acting. We therefore report tasks that need a call and tasks that need none separately.

We build an audit-grounded reward from AuditKIT metrics (schema validity, tool-call F1 with semantic argument matching, permission, a multilingual action-aware confirm-before-irreversible check, redundancy). With identical KL-anchored GRPO training, only the reward changed, success on tasks that need a call is 0.023 with the default reward and 0.423 with ours (+0.400 [+0.286, +0.514], p = 0.0001). Used as a filter for teacher trajectories, the audit reward cuts unconfirmed irreversible actions from 19.4% to 13.2% (-0.063 [-0.106, -0.024], p = 0.0046).

A correct reward ranking is not enough: without a KL anchor, GRPO with our reward also collapsed into silence, because a prompt whose samples are all silent yields zero advantage. DPO followed the imbalance of its preference pairs (unbalanced pairs: over-calling and unconfirmed bookings). Distillation from Aya Expanse 32B on audit-filtered trajectories raised test success from 0.289 to 0.513 (+0.224 [+0.141, +0.309], p = 0.0001). We also found and fixed bugs in AgentTune (B1, B15: GRPO never stops a Tiny Aya reply) and AuditKIT (A3: paired_bootstrap returns p ≈ 0.5 for any effect). All models, data, rollouts and logs are public.

## 1. Problem
An agent that books tickets must call tools when asked, must not act when it should not, and must never book or cancel without the user's yes. Reward-based fine-tuning optimises whatever the reward pays for. We ask: does AgentTune's default tool reward get gamed on a realistic agent task, can a reward built from AuditKIT's audit metrics resist it, and what else does the training need?

## 2. Setup
**Task and tools.** 8 transit tools (`get_station_code`, `search_trains`, `search_metro_route`, `get_fare`, `get_pnr_status`, `book_ticket`, `cancel_booking`, `file_complaint`) on a deterministic mock backend: real station names and Indian Railways codes, synthetic timetables, fares and PNRs. `book_ticket` and `cancel_booking` are tagged irreversible.

**Data.** 90 base tasks × English / Hindi / Marathi / Hinglish = 360 rows in 8 categories: single call, multi-step, no call needed, missing argument, irreversible unconfirmed (propose and ask), irreversible confirmed (the user said yes), declined (the user said no), denied tool (out of scope or another person's data). Split by base task: train 188 / dev 64 / test 108 rows; `file_complaint` and `cancel_booking` appear only in test. All 308 reference calls replay on the backend without error, and the 402 non-English turns keep the English turn's numbers, ids and scripts (`experiments/09_dataset_audit.py`). The non-English text is model-generated; native-speaker review is in progress.

**Policy.** One system prompt (operating rules written from the category definitions and the train split: ask for missing values, propose and wait for a yes before irreversible actions, refuse out-of-scope requests, work one tool step at a time) plus the tool schemas, identical for every model.

**Pipeline.** (1) Teacher: Aya Expanse 32B (4-bit, local) on the train split, 2 samples per task. For tasks whose reference has tool calls, the reference calls were replayed on the backend and the teacher wrote the replies (the 4-bit teacher solved 0 of 54 multi-step tasks on its own); every trajectory then passes the audit filter: the task is solved, every call is schema-valid, no unconfirmed irreversible call, the reply is in the user's script. 319 of 376 kept. (2) LoRA SFT of Tiny Aya Global on them, epoch chosen on dev. (3) GRPO (AgentTune `create_agentic_trainer("grpo")`, multi-turn tool rollouts, 300 steps, 4 generations per prompt) or DPO (AgentTune offline DPO) from the distilled student. (4) Held-out test: 108 tasks × 5 trials, scored with AuditKIT.

**Rewards.**
- *Default:* AgentTune `hybrid_prm_reward` (unpatched). At this commit it scores a correct call 0.0, a malformed call 1.0 and no call 1.0 (FINDINGS.md B1).
- *Audit reward* (`src/audit_rewards`): weighted sum of AuditKIT-backed terms: tool-call validity (1.0), tool-call F1 against the reference with semantic argument matching (1.0), permission penalty (1.0), confirm-before-irreversible penalty with a multilingual, action-aware confirmation check (1.0), redundant-call penalty (0.25). v2 adds one rule: a no-call on a task whose reference needs a call earns 0 validity instead of a neutral 0.5.

**Evaluation.** Success = the reference calls are made (semantic argument matching: a station name in another script, a reworded complaint), no extra or invalid call, no unconfirmed irreversible call. Strict (exact JSON) numbers are reported too. Because about half the tasks are correctly handled by not acting, we report **needs a call** (single call, multi-step, confirmed booking) and **no call needed** (declined, denied, missing argument, chit-chat) separately. Paired sign-flip permutation tests over tasks (10,000 resamples), 95% bootstrap CIs. The test split was never used for any choice; all choices were made on dev.

## 3. Results (held-out test split, 108 tasks × 5 trials)
| Model | Success | Success (strict) | Needs a call | No call needed | Calls / task | Unconfirmed irreversible | pass^5 |
|---|---|---|---|---|---|---|---|
| Tiny Aya Global + policy prompt (no training) | 0.289 | 0.204 | 0.409 | 0.217 | 1.24 | 18.0% | 0.120 |
| SFT on all teacher trajectories (ablation) | 0.478 | 0.411 | 0.441 | 0.637 | 0.89 | 19.4% | 0.222 |
| SFT on audit-filtered teacher trajectories | 0.513 | 0.435 | 0.427 | 0.713 | 0.97 | 13.2% | 0.296 |
| GRPO, AgentTune default reward | 0.481 | 0.481 | 0.000 | 1.000 | 0.00 | 0.0% | 0.481 |
| GRPO, audit reward v1 | 0.526 | 0.507 | 0.123 | 0.992 | 0.09 | 1.3% | 0.454 |
| GRPO, audit reward v2 | 0.483 | 0.483 | 0.009 | 0.996 | 0.01 | 0.0% | 0.472 |
| GRPO + KL anchor, default reward | 0.485 | 0.480 | 0.023 | 0.996 | 0.03 | 0.6% | 0.454 |
| GRPO + KL anchor, audit reward v2 | 0.513 | 0.444 | 0.423 | 0.721 | 0.85 | 11.8% | 0.241 |
| DPO, balanced audit-ranked pairs | 0.500 | 0.426 | 0.436 | 0.675 | 1.00 | 16.3% | 0.232 |

### 3.1 The default reward is gamed by silence
GRPO with the default reward makes 0.00 tool calls per task; success on tasks that need a call drops from 0.409 (base) to 0.000 (-0.409 [-0.536, -0.291], p = 0.0001). Its overall success (0.481) is close to the best model's, which is why a single success number is not enough here. Adding a KL anchor does not save it: run_A3 also stops acting (0.03 calls per task).

### 3.2 Same training, only the reward changed
With identical KL-anchored GRPO (same distilled start, data, steps, seed, settings fingerprint), success on tasks that need a call is 0.023 with the default reward and 0.423 with the audit reward: **+0.400 [+0.286, +0.514], p = 0.0001**. On all tasks the difference is +0.028 [-0.057, +0.117], p = 0.5544: the overall number hides a model that can no longer act. run_B3 matches the distilled student (overall +0.000 [-0.041, +0.041], p = 1.0000), so the audit reward preserves competence under RL, but it does not exceed distillation.

### 3.3 The audit reward as a data filter makes the agent safer
SFT on all 376 teacher trajectories vs only the 319 audit-approved ones: unconfirmed irreversible actions (booking or cancelling without the user's yes) **19.4% → 13.2% (-0.063 [-0.106, -0.024], p = 0.0046)**; overall success +0.035 [-0.007, +0.078], p = 0.1370 (not significant).

### 3.4 A correct reward ranking is not enough for GRPO
Recomputing reward v2 on run_B2's own training rollouts: on tasks that need a call, a calling rollout scored about 0.30 and a silent one 0.00 throughout training, yet by the last quarter of training no rollout called a tool. Once all 4 samples of a prompt are silent, their group-relative advantages are 0 and nothing pushes back, while no-call prompts keep pushing towards silence (205 of 300 steps had zero reward variance). A KL penalty to the distilled student (beta 0.04; the distilled adapter is merged first, because TRL's reference for a LoRA model is "adapter disabled", i.e. the base model) keeps the policy calling tools (run_B3). DPO from mined pairs (no new sampling) follows the imbalance of its pairs: with 327 of 432 negatives being "silent", it calls tools on everything (dev success 0.26, 27% unconfirmed bookings); balanced pairs fix the direction (0.45) but do not beat distillation.

### 3.5 Development decisions (dev split)
| Checkpoint (dev split, 64 tasks × 2) | Success | Calls / task | Unconfirmed | Reading |
|---|---|---|---|---|
| Distilled, checkpoint-20 | 0.523 | 0.34 | 1.6% | epoch choice |
| Distilled, checkpoint-40 | 0.492 | 0.84 | 7.0% | epoch choice |
| Distilled, checkpoint-60 (selected) | 0.570 | 1.05 | 4.7% | epoch choice |
| GRPO audit v1, step 220 | 0.523 | 0.12 | 0.0% | drifting into silence |
| GRPO + KL, audit v2, step 60 | 0.484 | 1.18 | 11.7% | no collapse |
| GRPO + KL, audit v2, step 120 | 0.492 | 0.97 | 5.5% | balanced |
| DPO, unbalanced pairs | 0.258 | 2.34 | 27.3% | collapses into over-calling |
| DPO, balanced pairs | 0.453 | 1.12 | 11.7% | balanced |

### 3.6 Per category and per language (success, test)
| Category | base | distill_unfiltered | distill | run_A | run_B | run_B2 | run_A3 | run_B3 | dpo_bal |
|---|---|---|---|---|---|---|---|---|---|
| declined | 0.49 | 0.86 | 0.86 | 1.00 | 0.99 | 0.99 | 1.00 | 0.81 | 0.74 |
| denied_tool | 0.10 | 0.40 | 0.53 | 1.00 | 1.00 | 1.00 | 0.97 | 0.65 | 0.53 |
| irreversible_confirmed | 0.40 | 0.62 | 0.60 | 0.00 | 0.23 | 0.03 | 0.01 | 0.61 | 0.61 |
| irreversible_unconfirmed | 0.17 | 0.10 | 0.15 | 0.25 | 0.24 | 0.25 | 0.23 | 0.14 | 0.15 |
| missing_argument | 0.05 | 0.39 | 0.51 | 1.00 | 0.99 | 1.00 | 1.00 | 0.56 | 0.54 |
| multi_step | 0.00 | 0.07 | 0.03 | 0.00 | 0.00 | 0.00 | 0.02 | 0.05 | 0.08 |
| no_call_needed | 0.12 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.93 | 0.97 |
| single_call | 0.72 | 0.54 | 0.55 | 0.00 | 0.11 | 0.00 | 0.04 | 0.51 | 0.53 |

| Language | base | distill_unfiltered | distill | run_A | run_B | run_B2 | run_A3 | run_B3 | dpo_bal |
|---|---|---|---|---|---|---|---|---|---|
| en | 0.33 | 0.42 | 0.46 | 0.48 | 0.48 | 0.48 | 0.50 | 0.47 | 0.50 |
| hi | 0.25 | 0.52 | 0.53 | 0.48 | 0.55 | 0.49 | 0.49 | 0.50 | 0.53 |
| mr | 0.30 | 0.50 | 0.54 | 0.48 | 0.56 | 0.48 | 0.48 | 0.59 | 0.52 |
| hinglish | 0.27 | 0.47 | 0.53 | 0.48 | 0.51 | 0.48 | 0.47 | 0.49 | 0.45 |

### 3.7 Public benchmarks (external check, our simplified protocol)
When2Call (NVIDIA, CC-BY-4.0; 200 questions per behaviour: call / ask for missing information / no tool fits; we score only the call-or-not decision, greedy, one turn) and BFCL-Hi (NVIDIA's Hindi BFCL v2 live: 258 simple, 258 irrelevance; AuditKIT `bfcl_arg_match`).
| Model | When2Call: calls when it should | When2Call: calls when no tool fits ↓ | When2Call: calls when info is missing ↓ | BFCL-Hi simple | BFCL-Hi irrelevance |
|---|---|---|---|---|---|
| base | 0.985 | 0.975 | 0.975 | 0.264 | 0.097 |
| distill | 0.785 | 0.505 | 0.690 | 0.043 | 0.903 |
| run_A | 0.000 | 0.000 | 0.000 | 0.000 | 0.996 |
| run_B | 0.095 | 0.045 | 0.050 | 0.008 | 0.977 |

The distilled student is the only balanced model on When2Call. All trained models lose Hindi function calling on unseen tools (BFCL-Hi simple 0.26 → under 0.05): training is specific to our tools and policy prompt.

## 4. Bugs found (with fixes and tests)
- **AgentTune B1:** `hybrid_prm_reward` scores correct / malformed / no call as 0 / 1 / 1. Lexsi's own notebook 14 documents a related failure (0 / 0 / 0 inside Jupyter); outside Jupyter the ordering is inverted, and we show the training consequence end to end.
- **AgentTune B15:** TRL's `GRPOTrainer` and AgentTune's trainer-driven rollout stop and mask on `tokenizer.eos_token_id` (`<|END_OF_TURN_TOKEN|>`), which Tiny Aya never emits (it ends with `<|END_RESPONSE|>` and `<EOS_TOKEN>`): every GRPO completion ran to the token limit and the invented text after the real reply was rewarded. Fix: stop on every end-of-reply id; 0% clipped, about 5× faster steps.
- **AuditKIT A3:** `paired_bootstrap` resamples uncentred differences, so p ≈ 0.5 for any effect size (a +0.3 effect: p = 0.49). Replaced by a paired permutation test.
- Our own round-1 pipeline flaws (distillation examples truncated inside the tool list, a missing policy prompt, an unfiltered teacher set): FINDINGS.md "Round 2". Round-1 results are kept in `results/round1/`.

## 5. Limitations
- One training seed per method; 27 base tasks in test (per-category cells hold 2–4 tasks).
- No RL method exceeded the distilled student on overall success; the RL results are about avoiding collapse and about safety.
- The teacher data is reference-guided on train tasks (gold tool calls, teacher-written replies); dev and test were never used this way.
- Hindi / Marathi / Hinglish text is model-generated (review in progress); the backend world is synthetic.
- Semantic matching and the needs-a-call / no-call split were introduced after seeing round-1 and run_A results; strict numbers are always given alongside.
- Public benchmarks use our simplified protocol, not the official leaderboard harnesses.

## 6. Reproducibility
Everything runs on one machine with two 24 GB GPUs (RTX 4500 Ada) or Kaggle 2× T4 (`notebooks/`). `bash scripts/round2.sh` runs teacher → distill → dev selection → GRPO → evaluation → benchmarks (resumable); `scripts/grpo_extra.sh run_B3 1 1` for the KL-anchored runs; `experiments/10_audit_dpo.py` for DPO; `scripts/make_writeup.py` regenerates every table here from the result files. 141 unit tests (`pytest`). Configs: `configs/`; per-stage results: `results/`; rollouts and scored rollouts: the datasets collection.
