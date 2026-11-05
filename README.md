# LatentSpace

**Audit-grounded rewards for training reliable, safe tool-calling agents.**

Our entry to the [Lexsi AI Safety & Alignment Hackathon](https://hackathon.lexsi.ai/), Track 3 (Agentic Fine-tuning), Problem Statement 9: *Reward design and reward-hacking defences*.

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

A small multilingual agent for Indian public transit (train and metro search, fares, booking, cancellation, complaints) in English, Hindi, Marathi and Hinglish, trained in three stages:

1. **Distillation:** an Aya Expanse teacher's trajectories, filtered by the audit-grounded reward, are distilled into Tiny Aya.
2. **GRPO, two runs:** the same model, data and settings, once with AgentTune's default tool reward and once with ours.
3. **Evaluation on held-out tasks**, using metrics that were never part of the reward (loop detection, pass^k reliability, task completion), per language.

## Stack

| Part | Used for |
|---|---|
| [AgentTune](https://github.com/Lexsi-Labs/AgentTune) | Agentic GRPO, tool-calling rollouts, distillation, trajectory logging |
| [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) | Agent metrics used as rewards and for evaluation |
| [Tiny Aya](https://huggingface.co/CohereLabs/tiny-aya-global) | Student model |
| [Aya Expanse](https://huggingface.co/CohereLabs/aya-expanse-32b) | Teacher model |

Models, adapters, datasets and logs will be published as public Hugging Face collections.

## Repository layout

```
src/audit_rewards/         AuditKIT agent metrics as AgentTune GRPO rewards (see its README)
tests/                     unit tests (CPU): python -m pytest tests
experiments/NN_name.py     one script per experiment -> results/NN_name.json
results/                   experiment outputs
docs/CODE_MAP.md           where everything lives in AgentTune and AuditKIT (file:line)
FINDINGS.md                what works, what is broken, with file:line references
```

## Experiments so far

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
  on the backend, and there is no split leakage. `combined()` ranks the reference above 3
  perturbed failures on every row in 7 of 8 categories, in all languages. It fails on
  irreversible_confirmed in Hindi (0/11), Marathi (0/11) and Hinglish (2/11): the confirmation
  rule in `audit_rewards` is an English keyword list, so a correct confirmed booking is
  penalised. A proposed multilingual rule (`transit_agent/multilingual_confirmation.py`, not yet
  merged into `audit_rewards`) uses no test-only phrasing. It reads yes / no correctly on
  14/14 train and dev rows and 6 to 7 of 8 held-out test rows per language. It still reads
  "yes, cancel it" as a no.
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

## Next

- Transit tool set and a 4-language eval set with a held-out split
- Baseline Tiny Aya, distillation, then the two GRPO runs (default reward vs audit reward)

## Status

Work in progress during the hackathon (1 to 14 November 2026). Results and the Hugging Face article will be linked here.
