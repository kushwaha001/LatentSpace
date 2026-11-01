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

## Planned layout

```
src/latentspace/rewards/   audit-grounded reward functions (AgentTune-compatible)
src/latentspace/tools/     transit tool definitions and schemas
data/                      eval set builder (4 languages, held-out split)
experiments/NN_name.py     one script per experiment -> results/NN_name.json
tests/                     unit tests (CPU)
```

## Status

Work in progress during the hackathon (1 to 14 November 2026). Results and the Hugging Face article will be linked here.
