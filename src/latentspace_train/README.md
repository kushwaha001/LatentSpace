# latentspace_train: the GPU training pipeline

Five stages, one CLI, one YAML per stage (`configs/`, all extending `configs/base.yaml`).
It runs on Kaggle (2x T4, the main target), Colab (T4 / L4) or a Hugging Face GPU Space.

```
python -m latentspace_train baseline        # 1. untrained Tiny Aya on the eval split
python -m latentspace_train teacher_collect # 2. Aya Expanse 32B via HF router, rejection-filtered
python -m latentspace_train distill         # 3. LoRA SFT on the kept teacher trajectories
python -m latentspace_train grpo --run run_A  # 4a. GRPO, AgentTune default tool reward
python -m latentspace_train grpo --run run_B  # 4b. GRPO, audit-grounded reward
python -m latentspace_train evaluate        # 5. test split, 5 trials, per language, bootstrap
```

Run from the repo root with `PYTHONPATH=src`. Shared flags: `--config`, `--set key.path=value`
(repeatable, YAML-parsed), `--push` (Hub), `--limit N`, `--max-usd X`, `--dry-run`,
`--print-config`. `evaluate` also takes `--run <checkpoint>` and `--compare-only`.

Notebooks: [`notebooks/kaggle_train.ipynb`](../../notebooks/kaggle_train.ipynb) (main) and
[`notebooks/colab_train.ipynb`](../../notebooks/colab_train.ipynb). Both install the stack with
[`scripts/install_deps.sh`](../../scripts/install_deps.sh), which pins AuditKIT 1.1.3 and AgentTune
`d4db4c12` (`pip install --no-deps -e` plus its base dependencies) and keeps the image's torch.

## Stages

| # | Stage | What it does | Writes |
|---|---|---|---|
| 1 | `baseline` | `baseline.modes`: `local` (Tiny Aya weights, fp16) and/or `api` (`CohereLabs/tiny-aya-global:cohere` through the HF router). AgentTune `create_rollout_fn` with its Cohere `tools_fallback_prompt`, scored with `AuditRewards.report()` and the train-time rewards. The modes that ran are recorded in `baseline_used`. | `results/10_baseline.json` |
| 2 | `teacher_collect` | `CohereLabs/aya-expanse-32b:cohere` via `https://router.huggingface.co/v1` (AgentTune's LiteLLM `APIRolloutEngine`, `HF_TOKEN`, optional `bill_to`). `--max-usd` (default 5) stops collection when the token-based estimate reaches the budget. Keeps a trajectory only if `audit_tool_validity_reward == 1` and `audit_confirm_irreversible_penalty == 0` (plus: no error, non-empty answer; correct no-call answers are kept). `--set teacher.backend=local` runs aya-expanse-8b in 4-bit on the GPU instead (8B has no Inference Provider). | `results/11_teacher_collect.json`, `outputs/teacher_collect/{raw,filtered}.jsonl` |
| 3 | `distill` | `create_distill_trainer` (`Project.distill` → TRL `SFTTrainer`) with LoRA, 4-bit, gradient checkpointing. Rows are rendered exactly as the student sees them in a rollout: same fallback system prompt, folded tool results, Cohere call format. | `results/12_distill.json`, `outputs/distill/adapter/` |
| 4 | `grpo` | `create_agentic_trainer("grpo")` with LoRA, `num_generations=4`, starting from the distilled adapter. `run_A`: `hybrid_prm_reward`, upstream and unpatched. `run_B`: `AuditRewards(tools).train_time()` with `DEFAULT_WEIGHTS`. Everything else is shared; `shared_settings_fingerprint` in the results proves it. The run's own training rollouts are then scored with the audit metrics. | `results/13_grpo_run_{A,B}.json`, `outputs/grpo/<run>/{adapter/,trajectories.jsonl}` |
| 5 | `evaluate` | Every checkpoint in `evaluate.checkpoints` (base, distill, run_A, run_B; local paths, Hub ids, or router models) on the test split, 5 trials per task. Train-time metrics (the audit rewards and their combination) and report-only metrics (loop detection, exact match, success, pass@k / pass^k via `auditkit.agent_eval.reliability`), overall and per language. `evaluate.comparisons` pairs are compared with AuditKIT's paired bootstrap over per-task means. | `results/14_evaluate*.json`, `outputs/evaluate/<ckpt>/` |

Every results file carries the stage payload plus `seed`, the full resolved `config`,
`provenance` (repo commit and dirty flag, AgentTune commit, host kind), `versions` (torch,
transformers, trl, peft, agenttune, auditkit, ...) and `hardware` (GPUs, bf16 support).

`--push` uploads to the repos in `hub.repos` (namespace `kanishk001`): the distill and GRPO
adapters as model repos (adapter files at the repo root, so later stages load them by id),
teacher trajectories as a dataset, and rollouts / scored files of every stage to
`kanishk001/latentspace-logs`.

## Memory-safe defaults (16 GB GPUs)

- `precision: auto`: fp16 on T4 (no bf16), bf16 on L4 / A10G / A100, fp32 on CPU.
- Training (distill, GRPO): 4-bit NF4 with double quantization and fp16 compute, LoRA r=16
  on all attention and MLP projections, gradient checkpointing (non-reentrant).
- GRPO: `per_device_train_batch_size=4` with `num_generations=4` (one prompt per step),
  `max_completion_length=128` per turn, `max_steps_per_turn=3`, `beta=0` (no reference
  forward pass).
- Inference (baseline, evaluate): fp16 weights, no quantization (6.7 GB, about 2x faster
  decoding than 4-bit on a T4).
- One model per process. On 2x T4, run_A and run_B train at the same time, one per GPU
  (`CUDA_VISIBLE_DEVICES=0` / `1`). That avoids DDP and 4-bit sharding.

## Time and cost estimate

| Stage | Where | Kaggle 2x T4 (wall) | GPU-h | HF credits |
|---|---|---|---|---|
| install + tests | CPU | 8-12 min | - | - |
| 1 baseline, local fp16 | GPU 0 | 25-35 min per 100 tasks | 0.5 | - |
| 1 baseline, router (optional) | API | 5-10 min per 100 tasks, alongside the local one | - | about $0.05-0.10 |
| 2 teacher_collect, 32B router | API | 1.5-2 h per 600 trajectories (calls are sequential) | - | about $1.5-2.5, capped by `--max-usd` |
| 2' teacher_collect, 8B 4-bit local | GPU 0 | 4-5 h per 300 trajectories | 4-5 | - |
| 3 distill, QLoRA | GPU 0 | 30-60 min per 400 rows x 2 epochs | 0.5-1 | - |
| 4 grpo run_A ‖ run_B, 100 steps each | GPU 0 ‖ GPU 1 | 3-3.5 h (both at once) | 6-7 | - |
| 5 evaluate, 4 ckpts x 100 tasks x 5 trials | GPU 0 ‖ GPU 1 | 4-5 h | 8-10 | - |
| **total** | | **about 10-12 h in 3 sessions** | **about 16-19** | **about $2-3** |

These are estimates, not measurements (no GPU was available when the pipeline was built).
Assumptions: Tiny Aya is 3.35B parameters; HF `generate` runs at about 20 tok/s (fp16) or
10 tok/s (4-bit) per sequence on a T4; a rollout is about 2.5 turns and 300 generated tokens
with a 1.5-2k token prompt (the tool schemas); AgentTune rolls out one trajectory at a time.
The teacher price is $0.50 / $1.50 per M input / output tokens (Cohere's list price for Aya
Expanse 32B); check the provider price on the model page and set
`teacher.price_usd_per_m`. The budget stop uses that price, so it is only as good as the
number you give it. Everything scales linearly with the number of tasks and trials. Kaggle
stops a session at 12 h. Plan three sessions (stages 1-3, stage 4, stage 5) and use `--push`
so each session can resume from the Hub. The 30 h/week Kaggle quota covers one full pass. If
the quota meter charges per GPU rather than per session, the GPU-h column is what counts.

## CPU dry run

`--dry-run` swaps in a randomly initialised ~100k-parameter Llama with a tool-ignoring chat
template (so AgentTune's fallback-prompt path runs, as for Tiny Aya), the stub tasks
(`stub.py`), 3 steps, fp32 and `results/smoke/`. `bash scripts/dry_run_all.sh` runs every
stage in notebook order in about 3 minutes; its output is committed under `results/smoke/`.
This proves wiring, config loading and result writing. It does not prove learning.

## What the tasks look like

`tasks.py` imports the transit tools by name (`tasks.tools: transit_agent:TOOLS`; a list or a
zero-argument factory) and reads `data/eval/{train,eval,test}.jsonl` (or one file with a split
column). Field names are mapped in `tasks.data.fields`. Rows need a prompt (a string, or a
message list ending in the user's turn for confirmation tasks), a language and, ideally,
`expected_tool_calls` (any shape AuditKIT accepts). `[[]]` means "no call is correct". Until the
eval-set branch lands, use `--set tasks.source=stub`.

## Known limits and things not verified on CPU

- No GPU run has happened yet: 4-bit loading, fp16 numerics with Tiny Aya (Cohere models are
  trained in bf16; if the loss goes NaN on a T4, try `--set student.quantization=null` or an L4),
  real memory use and every time in the table above.
- No call to the HF router was made (the sandbox blocks the Hub). The router path is tested
  with a mocked `litellm.completion`. It is not yet confirmed that the `cohere` provider accepts
  `X-HF-Bill-To` or returns `usage`. If it returns no usage, the budget counts 0 tokens; check the
  `budget` block in the first results file.
- `hybrid_prm_reward` (run_A) needs an input adapter: with message-list prompts, TRL passes
  message-list completions and the upstream reward raises `TypeError` in `re.sub`.
  `combine_rewards` then retries with `completions` as the first positional argument, which
  lands on `prompts` and fails again. `stages/grpo.py:text_inputs` flattens the inputs to text and
  leaves the scoring unchanged; `reward.input_adapter` in the results records this. It also
  calls litellm on every batch (FINDINGS.md B3), which logs errors and costs a few seconds. Its
  network use was not checked on Kaggle.
- AgentTune declares Python >= 3.12. On an older image `install_deps.sh` installs with
  `--ignore-requires-python` and warns.
- Teacher collection and rollouts are sequential. For a larger eval set, raise
  `tasks.limit` / `samples_per_task` with the budget in mind.
