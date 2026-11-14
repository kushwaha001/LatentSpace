# LatentSpace

**Audit-grounded rewards for training reliable, safe tool-calling agents.**

Our entry to the [Lexsi AI Safety & Alignment Hackathon](https://hackathon.lexsi.ai/), Track 3 (Agentic Fine-tuning), Problem Statement 9: *Reward design and reward-hacking defences*.

**Write-up (article): [Hugging Face](https://huggingface.co/datasets/kanishk001/latentspace-article) · [docs/WRITEUP.md](docs/WRITEUP.md)** · Abstract: [docs/ABSTRACT.md](docs/ABSTRACT.md) · Results: [Results](#results-round-2-final) · Bugs and fixes: [FINDINGS.md](FINDINGS.md)

> **Headline:** with identical KL-anchored GRPO training and only the reward changed, AgentTune's default reward drives the agent to stop calling tools (success on tasks that need a call: 0.023), while our audit-grounded reward keeps it competent (0.423; +0.40, p = 0.0001). Used as a filter for teacher data, the audit reward cuts bookings and cancellations made without the user's yes from 19.4% to 13.2% (p = 0.005).

## The idea in short

An agent trained with reinforcement learning learns exactly what its reward can see. We found that the default tool-call reward in [AgentTune](https://github.com/Lexsi-Labs/AgentTune) scores a correct tool call **0.0** and a malformed or missing call **1.0**. With GRPO, which pushes each answer up or down relative to its group, that reward would teach a model to stop calling tools properly.

[AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) already checks tool calls correctly (schema validity, argument match, permissions, redundancy), but only after training. LatentSpace turns those checks into training rewards:

| Answer | AgentTune `hybrid_prm_reward` | Audit-grounded reward |
|---|---|---|
| Correct call | 0.0 | 1.0 (validity) / 0.471 (combined) |
| Malformed call | 1.0 | 0.0 / 0.000 |
| No call | 1.0 | 0.5 / 0.118 |

It also adds a **confirm-before-irreversible-action** penalty: tools such as booking, paying or cancelling are only rewarded after a clear "yes" from the user.

## What we are building

A small multilingual agent for Indian public transit (train and metro search, fares, booking, cancellation, complaints) in English, Hindi, Marathi and Hinglish, trained and evaluated in four stages:

1. **Distillation:** Aya Expanse 32B trajectories, filtered by the audit-grounded reward, are distilled into Tiny Aya Global (LoRA SFT; epoch chosen on the dev split). An ablation trains on the unfiltered trajectories.
2. **GRPO:** the same model, data and settings with AgentTune's default tool reward or with ours (audit reward v1, v2), without and with a KL anchor to the distilled student.
3. **DPO:** offline DPO on audit-ranked preference pairs mined from existing rollouts (unbalanced and balanced).
4. **Evaluation on the held-out test split** (108 tasks × 5 trials), strict and semantic argument matching, tasks that need a call and tasks that need none reported separately, per language and category, paired permutation tests; plus two public benchmarks (When2Call, BFCL-Hi).

## Stack

| Part | Used for |
|---|---|
| [AgentTune](https://github.com/Lexsi-Labs/AgentTune) | Agentic GRPO, tool-calling rollouts, distillation, trajectory logging |
| [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) | Agent metrics used as rewards and for evaluation |
| [Tiny Aya](https://huggingface.co/CohereLabs/tiny-aya-global) | Student model |
| [Aya Expanse](https://huggingface.co/CohereLabs/aya-expanse-32b) | Teacher model |

Models (LoRA adapters), datasets, rollouts and logs are public on Hugging Face: [models collection](https://huggingface.co/collections/kanishk001/latentspace-models-lexsi-hackathon-track-3-p9-6af84d1bbf8a687cc9e0b8d2) · [datasets collection](https://huggingface.co/collections/kanishk001/latentspace-datasets-and-logs-lexsi-hackathon-t3-p9-6af84d2c307f37a24ab2d708) (`scripts/publish_hf.py`).

## Repository layout

```
src/audit_rewards/         AuditKIT agent metrics as AgentTune GRPO rewards (see its README)
src/transit_agent/         the 8 transit tools, mock backend, policy prompt, semantic argument matching
src/latentspace_train/     the training pipeline (teacher, distill, GRPO, evaluate), one CLI
configs/                   one config per stage (13_grpo.yaml holds run_A/B/B2/A3/B3)
experiments/NN_name.py     one script per experiment -> results/NN_name.json (07 lenient rescoring,
                           08 public benchmarks, 09 dataset audit, 10 audit-ranked DPO)
scripts/                   round2.sh (full resumable pipeline), grpo_extra.sh, gpu_run.sh,
                           make_writeup.py (regenerates docs/WRITEUP.md), publish_hf.py
results/                   per-stage results; results/round1/ keeps the round-1 run
docs/WRITEUP.md            the submission write-up; docs/RUN_LOCAL.md how to run locally
FINDINGS.md                what works, what is broken, with file:line references
tests/                     unit tests (CPU): python -m pytest tests
```

## Early experiments (01-04)

| # | Experiment | Result |
|---|---|---|
| 01 | [Reward gap](experiments/01_reward_gap.py) | `hybrid_prm_reward` scores correct / malformed / no call as 0.0 / 1.0 / 1.0. None of the 26 built-in rewards separates malformed from no call. [results](results/01_reward_gap.json) |
| 02 | [AgentTune → AuditKIT hand-off](experiments/02_handoff.py) | AgentTune's `trajectories.jsonl` loads into AuditKIT; its metrics catch loops, invalid arguments and forbidden tools. [results](results/02_handoff.json) |
| 03 | [Audit rewards vs default rewards](experiments/03_audit_rewards.py) | Combined audit reward catches 21 of 21 paired failure cases with 0 inverted; the best default combination catches 6 and inverts 8. [table](results/03_audit_rewards.md) |
| 04 | [GRPO smoke run](experiments/04_grpo_smoke.py) | The rewards plug into AgentTune's agentic GRPO trainer and are called on every step (20 steps, CPU, tiny model: a plumbing test, not a learning run). [results](results/04_grpo_smoke.json) |

Run any experiment from the repo root with `python experiments/NN_name.py`.

## Transit tools and eval set (experiments 05-06)

- **Tools** (`src/transit_agent/`): `get_station_code`, `search_trains`, `search_metro_route`,
  `get_fare`, `get_pnr_status`, `book_ticket`, `cancel_booking` and `file_complaint`.
  `book_ticket` and `cancel_booking` are tagged irreversible. The tools run on a
  deterministic mock backend (`data/backend/`). Station names and railway codes are real;
  timetables, fares and PNRs are synthetic ([data card](data/backend/DATACARD.md)).
  Names resolve in English, Hindi, Marathi or romanised form.
- **Eval set** (`data/eval/`): 90 base tasks × en / hi / mr / Hinglish = 360 rows, paired by
  `base_id`, in 8 balanced categories. It is split by base_id (47 / 16 / 27). `file_complaint`
  and `cancel_booking` appear only in test. **The non-English text is model-generated and
  still needs native-speaker review** ([data card](data/eval/DATACARD.md),
  `review_checklist.csv`).
- **[05](experiments/05_eval_set_checks.py)**: all 308 expected calls are schema-valid and run
  on the backend, and there is no split leakage. It also found a flaw in our own reward: the
  confirm-before-irreversible rule was an English keyword list, so a correct booking confirmed in
  Hindi or Marathi was penalised. The rule in `audit_rewards.confirmation` is now multilingual
  (en / hi / mr / Hinglish, token-based so Devanagari works) and action-aware ("yes, cancel it"
  confirms a cancellation; "cancel that" still refuses a booking). Its lexicon contains no
  phrasing that appears only in the test split.

  | Reference ranked above all 3 failures | en | hi | mr | Hinglish |
  |---|---|---|---|---|
  | English-only rule (before) | 89/90 | 79/90 | 79/90 | 81/90 |
  | Multilingual rule (now) | 90/90 | 89/90 | 90/90 | 90/90 |

  Yes/no read correctly on held-out test phrasings: en 8/8, hi 7/8, mr 8/8, Hinglish 8/8. The
  one miss ("बढ़िया, बुक कर दो") uses a phrasing that occurs only in test, so it was not added.
  [results](results/05_eval_set_checks.json)
- **[06](experiments/06_baseline_harness.py)**: baseline harness (AgentTune rollout + audit
  metrics + AuditKIT pass^k) for a HF model on the test split. Only `--dry-run` (a scripted
  policy on CPU) has been run. [results](results/06_baseline_dry_run.json)

## Training pipeline

`src/latentspace_train/` runs the full plan on a GPU: baseline, teacher collection (Aya Expanse
32B via HF Inference Providers, rejection-filtered with the audit rewards), LoRA distillation,
GRPO run_A (AgentTune default reward) and run_B (audit reward), and held-out evaluation. One CLI,
one config per stage in `configs/`, and a `results/NN_*.json` file per stage. Start with
[`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) (2x T4) or the Colab notebook.
Details, memory defaults and a cost/time table are in
[its README](src/latentspace_train/README.md). CPU check: `bash scripts/dry_run_all.sh`.
On a local machine with two 24 GB GPUs: [docs/RUN_LOCAL.md](docs/RUN_LOCAL.md)
(`bash scripts/run_local_2gpu.sh`).

## Results (round 2, final)

**Write-up (article): [Hugging Face](https://huggingface.co/datasets/kanishk001/latentspace-article) · [docs/WRITEUP.md](docs/WRITEUP.md)** · abstract: [docs/ABSTRACT.md](docs/ABSTRACT.md) · every table there comes from `results/`; `scripts/make_writeup.py` regenerates them in `docs/RESULTS_TABLES.md`.

Held-out test split (108 tasks × 5 trials). "Needs a call" = single call, multi-step, confirmed booking; "no call" = declined, denied, missing argument, chit-chat.

| Model | Success | Needs a call | No call | Calls / task | Unconfirmed irreversible |
|---|---|---|---|---|---|
| Tiny Aya Global + policy prompt | 0.289 | 0.409 | 0.217 | 1.24 | 18.0% |
| SFT, audit-filtered teacher data | 0.513 | 0.427 | 0.713 | 0.97 | 13.2% |
| GRPO, AgentTune default reward | 0.481 | 0.000 | 1.000 | 0.00 | 0.0% |
| GRPO + KL anchor, default reward | 0.485 | 0.023 | 0.996 | 0.03 | 0.6% |
| **GRPO + KL anchor, audit reward** | **0.513** | **0.423** | 0.721 | 0.85 | 11.8% |

- The default reward is gamed by silence; the overall success number hides it.
- Same KL-anchored GRPO, only the reward changed: needs-a-call success +0.400 (p = 0.0001).
- Audit reward as a data filter: unconfirmed irreversible actions 19.4% → 13.2% (p = 0.005).
- Without a KL anchor, GRPO collapsed into silence even with a correct reward ranking (run_B2).
- Bugs: AgentTune B1, B15; AuditKIT A3 ([FINDINGS.md](FINDINGS.md)). Round-1 results: `results/round1/`.

## Next

- Native-speaker review of the Hindi, Marathi and Hinglish tasks (`data/eval/review_checklist.csv`), then re-evaluation
- 2-3 training seeds per method (all results above are one seed)
- A public demo (Hugging Face Space) and upstream PRs for B15 (AgentTune) and A3 (AuditKIT)
- Reward and optimiser work that beats the distilled student, not only matches it (e.g. dynamic sampling of prompts whose samples disagree)

## Status

Stage 1 submission (Lexsi hackathon, deadline 15 November 2026 AoE). Write-up: [docs/WRITEUP.md](docs/WRITEUP.md). Hugging Face collections: [models](https://huggingface.co/collections/kanishk001/latentspace-models-lexsi-hackathon-track-3-p9-6af84d1bbf8a687cc9e0b8d2) · [datasets, rollouts and logs](https://huggingface.co/collections/kanishk001/latentspace-datasets-and-logs-lexsi-hackathon-t3-p9-6af84d2c307f37a24ab2d708).
