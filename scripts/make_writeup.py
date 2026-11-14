"""Write docs/WRITEUP.md (the submission write-up) and docs/ABSTRACT.md. Tables are generated from
results/ and outputs/evaluate/, so the numbers match the result files exactly.

    python scripts/make_writeup.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from latentspace_train.scoring import paired_test, per_task_means  # noqa: E402

R = ROOT / "results"
NEED = ["single_call", "multi_step", "irreversible_confirmed"]
NONE = ["declined", "denied_tool", "missing_argument", "no_call_needed"]
HF = "kanishk001"
MODELS = [("base", "Tiny Aya Global + policy prompt (no training)"),
          ("distill_unfiltered", "SFT on all teacher trajectories (ablation)"),
          ("distill", "SFT on audit-filtered teacher trajectories"),
          ("run_A", "GRPO, AgentTune default reward"), ("run_B", "GRPO, audit reward v1"),
          ("run_B2", "GRPO, audit reward v2"), ("run_A3", "GRPO + KL anchor, default reward"),
          ("run_B3", "GRPO + KL anchor, audit reward v2"), ("dpo_bal", "DPO, balanced audit-ranked pairs")]


def summ(n):
    return json.loads((R / f"14_evaluate_{n}.json").read_text())["checkpoints"][n]["summary"]


def rows(n):
    return [json.loads(l) for l in open(ROOT / f"outputs/evaluate/{n}/scored.jsonl")]


def rate(n, cats, key="lenient_success"):
    v = [r[key] for r in rows(n) if r["category"] in cats]
    return sum(v) / len(v)


def ptest(a, b, m, cats=None):
    A, B = rows(a), rows(b)
    if cats:
        A = [r for r in A if r["category"] in cats]
        B = [r for r in B if r["category"] in cats]
    x, y = per_task_means(A, m), per_task_means(B, m)
    return paired_test([(x[t], y[t]) for t in sorted(x.keys() & y.keys())])


def pv(r):
    return f"{r['delta']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], p = {r['p_value']:.4f}"


f3 = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
pc = lambda x: f"{100 * x:.1f}%"  # noqa: E731

main = ["| Model | Success | Success (strict) | Needs a call | No call needed | Calls / task | Unconfirmed irreversible | pass^5 |",
        "|---|---|---|---|---|---|---|---|"]
for k, label in MODELS:
    s = summ(k); o = s["overall"]
    main.append(f"| {label} | {f3(o['lenient_success'])} | {f3(o['success'])} | {f3(rate(k, NEED))} | {f3(rate(k, NONE))} "
                f"| {o['tool_call_count']:.2f} | {pc(o['unconfirmed_irreversible_rate'])} | "
                f"{f3((o.get('reliability_lenient') or o.get('reliability') or {}).get('pass_hat_k'))} |")
cats = sorted(summ("base")["per_category"])
percat = ["| Category | " + " | ".join(k for k, _ in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
for c in cats:
    percat.append(f"| {c} | " + " | ".join(f"{summ(k)['per_category'][c]['lenient_success']:.2f}" for k, _ in MODELS) + " |")
langs = ["en", "hi", "mr", "hinglish"]
perlang = ["| Language | " + " | ".join(k for k, _ in MODELS) + " |", "|---|" + "---|" * len(MODELS)]
for lg in langs:
    perlang.append(f"| {lg} | " + " | ".join(f"{summ(k)['per_language'][lg]['lenient_success']:.2f}" for k, _ in MODELS) + " |")

T = {
    "a3b3_need": ptest("run_A3", "run_B3", "lenient_success", NEED),
    "a3b3_all": ptest("run_A3", "run_B3", "lenient_success"),
    "a3b3_calls": ptest("run_A3", "run_B3", "tool_call_count"),
    "filt_unconf": ptest("distill_unfiltered", "distill", "unconfirmed_irreversible_rate"),
    "filt_succ": ptest("distill_unfiltered", "distill", "lenient_success"),
    "base_distill": ptest("base", "distill", "lenient_success"),
    "base_A_need": ptest("base", "run_A", "lenient_success", NEED),
    "distill_B3": ptest("distill", "run_B3", "lenient_success"),
    "distill_B3_unconf": ptest("distill", "run_B3", "unconfirmed_irreversible_rate"),
}
bm = json.loads((R / "08_public_benchmarks.json").read_text())["models"]
bench = ["| Model | When2Call: calls when it should | When2Call: calls when no tool fits ↓ | When2Call: calls when info is missing ↓ | BFCL-Hi simple | BFCL-Hi irrelevance |",
         "|---|---|---|---|---|---|"]
for k, v in bm.items():
    w, b = v["when2call"], v["bfcl_hi"]
    bench.append(f"| {k} | {f3(w['calls_when_it_should'])} | {f3(w['tool_hallucination_rate_cannot_answer'])} | "
                 f"{f3(w['tool_call_rate_request_for_info'])} | {f3(b['simple_accuracy'])} | {f3(b['irrelevance_accuracy'])} |")
dev = ["| Checkpoint (dev split, 64 tasks × 2) | Success | Calls / task | Unconfirmed | Reading |", "|---|---|---|---|---|"]
sel = json.loads((R / "12_distill_selection.json").read_text())
for c in sel["candidates"]:
    dev.append(f"| Distilled, {c['checkpoint']}{' (selected)' if c['checkpoint'] == sel['selected'] else ''} | "
               f"{f3(c['dev_lenient_success'])} | {c['dev_calls']:.2f} | {pc(c['dev_unconfirmed_rate'])} | epoch choice |")
for n, label, note in [("runB_checkpoint-220", "GRPO audit v1, step 220", "drifting into silence"),
                       ("run_B3_ckpt60", "GRPO + KL, audit v2, step 60", "no collapse"),
                       ("run_B3_ckpt120", "GRPO + KL, audit v2, step 120", "balanced"),
                       ("dpo_audit_dev", "DPO, unbalanced pairs", "collapses into over-calling"),
                       ("dpo_bal_dev", "DPO, balanced pairs", "balanced")]:
    p = R / f"_devsel/{n}/10_baseline.json"
    if p.exists():
        o = json.loads(p.read_text())["modes"]["local"]["summary"]["overall"]
        dev.append(f"| {label} | {f3(o['lenient_success'])} | {o['tool_call_count']:.2f} | {pc(o['unconfirmed_irreversible_rate'])} | {note} |")
tc = json.loads((R / "11_teacher_collect.json").read_text())
A, B3, A3, D, DU, BASE = (summ(k)["overall"] for k in ("run_A", "run_B3", "run_A3", "distill", "distill_unfiltered", "base"))

ABSTRACT = f"""We train a small multilingual tool-calling agent for Indian public transit (Tiny Aya Global, 3.35B; English, Hindi, Marathi, Hinglish; 8 tools including irreversible booking and cancellation on a deterministic mock backend) and study reward hacking in agentic RL with AgentTune and AuditKIT.

AgentTune's default tool reward (`hybrid_prm_reward`) scores a correct tool call 0 and no call 1. Trained with it, GRPO learns to never call a tool: {A['tool_call_count']:.2f} calls per task and 0% success on every task that needs an action on the held-out test split, while its overall success ({A['lenient_success']:.3f}) stays close to the best model ({D['lenient_success']:.3f}), because about half of realistic requests are correctly answered by not acting. We therefore report tasks that need a call and tasks that need none separately.

We build an audit-grounded reward from AuditKIT metrics (schema validity, tool-call F1 with semantic argument matching, permission, a multilingual action-aware confirm-before-irreversible check, redundancy). With identical KL-anchored GRPO training, only the reward changed, success on tasks that need a call is {rate('run_A3', NEED):.3f} with the default reward and {rate('run_B3', NEED):.3f} with ours ({pv(T['a3b3_need'])}). Used as a filter for teacher trajectories, the audit reward cuts unconfirmed irreversible actions from {pc(DU['unconfirmed_irreversible_rate'])} to {pc(D['unconfirmed_irreversible_rate'])} ({pv(T['filt_unconf'])}).

A correct reward ranking is not enough: without a KL anchor, GRPO with our reward also collapsed into silence, because a prompt whose samples are all silent yields zero advantage. DPO followed the imbalance of its preference pairs (unbalanced pairs: over-calling and unconfirmed bookings). Distillation from Aya Expanse 32B on audit-filtered trajectories raised test success from {BASE['lenient_success']:.3f} to {D['lenient_success']:.3f} ({pv(T['base_distill'])}). We also found and fixed bugs in AgentTune (B1, B15: GRPO never stops a Tiny Aya reply) and AuditKIT (A3: paired_bootstrap returns p ≈ 0.5 for any effect). All models, data, rollouts and logs are public."""

W = f"""# Silence pays: reward hacking in a multilingual tool-calling agent, and an audit-grounded reward that resists it

**Lexsi AI Safety & Alignment Hackathon · Track 3 (Agentic Fine-tuning) · Problem 9: Reward design and reward-hacking defences** (also: reliable tool calling, agentic distillation, multilingual)
**Repos:** AgentTune, AuditKIT · **Models:** Tiny Aya Global (student), Aya Expanse 32B (teacher)
**Code:** https://github.com/kushwaha001/LatentSpace · **Models:** https://huggingface.co/{HF} · **Data, rollouts, logs:** https://huggingface.co/{HF}

## TL;DR
{ABSTRACT}

## 1. Problem
An agent that books tickets must call tools when asked, must not act when it should not, and must never book or cancel without the user's yes. Reward-based fine-tuning optimises whatever the reward pays for. We ask: does AgentTune's default tool reward get gamed on a realistic agent task, can a reward built from AuditKIT's audit metrics resist it, and what else does the training need?

## 2. Setup
**Task and tools.** 8 transit tools (`get_station_code`, `search_trains`, `search_metro_route`, `get_fare`, `get_pnr_status`, `book_ticket`, `cancel_booking`, `file_complaint`) on a deterministic mock backend: real station names and Indian Railways codes, synthetic timetables, fares and PNRs. `book_ticket` and `cancel_booking` are tagged irreversible.

**Data.** 90 base tasks × English / Hindi / Marathi / Hinglish = 360 rows in 8 categories: single call, multi-step, no call needed, missing argument, irreversible unconfirmed (propose and ask), irreversible confirmed (the user said yes), declined (the user said no), denied tool (out of scope or another person's data). Split by base task: train 188 / dev 64 / test 108 rows; `file_complaint` and `cancel_booking` appear only in test. All 308 reference calls replay on the backend without error, and the 402 non-English turns keep the English turn's numbers, ids and scripts (`experiments/09_dataset_audit.py`). The non-English text is model-generated; native-speaker review is in progress.

**Policy.** One system prompt (operating rules written from the category definitions and the train split: ask for missing values, propose and wait for a yes before irreversible actions, refuse out-of-scope requests, work one tool step at a time) plus the tool schemas, identical for every model.

**Pipeline.** (1) Teacher: Aya Expanse 32B (4-bit, local) on the train split, 2 samples per task. For tasks whose reference has tool calls, the reference calls were replayed on the backend and the teacher wrote the replies (the 4-bit teacher solved 0 of 54 multi-step tasks on its own); every trajectory then passes the audit filter: the task is solved, every call is schema-valid, no unconfirmed irreversible call, the reply is in the user's script. {tc['kept']} of {tc['seen']} kept. (2) LoRA SFT of Tiny Aya Global on them, epoch chosen on dev. (3) GRPO (AgentTune `create_agentic_trainer("grpo")`, multi-turn tool rollouts, 300 steps, 4 generations per prompt) or DPO (AgentTune offline DPO) from the distilled student. (4) Held-out test: 108 tasks × 5 trials, scored with AuditKIT.

**Rewards.**
- *Default:* AgentTune `hybrid_prm_reward` (unpatched). At this commit it scores a correct call 0.0, a malformed call 1.0 and no call 1.0 (FINDINGS.md B1).
- *Audit reward* (`src/audit_rewards`): weighted sum of AuditKIT-backed terms: tool-call validity (1.0), tool-call F1 against the reference with semantic argument matching (1.0), permission penalty (1.0), confirm-before-irreversible penalty with a multilingual, action-aware confirmation check (1.0), redundant-call penalty (0.25). v2 adds one rule: a no-call on a task whose reference needs a call earns 0 validity instead of a neutral 0.5.

**Evaluation.** Success = the reference calls are made (semantic argument matching: a station name in another script, a reworded complaint), no extra or invalid call, no unconfirmed irreversible call. Strict (exact JSON) numbers are reported too. Because about half the tasks are correctly handled by not acting, we report **needs a call** (single call, multi-step, confirmed booking) and **no call needed** (declined, denied, missing argument, chit-chat) separately. Paired sign-flip permutation tests over tasks (10,000 resamples), 95% bootstrap CIs. The test split was never used for any choice; all choices were made on dev.

## 3. Results (held-out test split, 108 tasks × 5 trials)
{chr(10).join(main)}

### 3.1 The default reward is gamed by silence
GRPO with the default reward makes {A['tool_call_count']:.2f} tool calls per task; success on tasks that need a call drops from {rate('base', NEED):.3f} (base) to {rate('run_A', NEED):.3f} ({pv(T['base_A_need'])}). Its overall success ({A['lenient_success']:.3f}) is close to the best model's, which is why a single success number is not enough here. Adding a KL anchor does not save it: run_A3 also stops acting ({A3['tool_call_count']:.2f} calls per task).

### 3.2 Same training, only the reward changed
With identical KL-anchored GRPO (same distilled start, data, steps, seed, settings fingerprint), success on tasks that need a call is {rate('run_A3', NEED):.3f} with the default reward and {rate('run_B3', NEED):.3f} with the audit reward: **{pv(T['a3b3_need'])}**. On all tasks the difference is {pv(T['a3b3_all'])}: the overall number hides a model that can no longer act. run_B3 matches the distilled student (overall {pv(T['distill_B3'])}), so the audit reward preserves competence under RL, but it does not exceed distillation.

### 3.3 The audit reward as a data filter makes the agent safer
SFT on all {tc['seen']} teacher trajectories vs only the {tc['kept']} audit-approved ones: unconfirmed irreversible actions (booking or cancelling without the user's yes) **{pc(DU['unconfirmed_irreversible_rate'])} → {pc(D['unconfirmed_irreversible_rate'])} ({pv(T['filt_unconf'])})**; overall success {pv(T['filt_succ'])} (not significant).

### 3.4 A correct reward ranking is not enough for GRPO
Recomputing reward v2 on run_B2's own training rollouts: on tasks that need a call, a calling rollout scored about 0.30 and a silent one 0.00 throughout training, yet by the last quarter of training no rollout called a tool. Once all 4 samples of a prompt are silent, their group-relative advantages are 0 and nothing pushes back, while no-call prompts keep pushing towards silence (205 of 300 steps had zero reward variance). A KL penalty to the distilled student (beta 0.04; the distilled adapter is merged first, because TRL's reference for a LoRA model is "adapter disabled", i.e. the base model) keeps the policy calling tools (run_B3). DPO from mined pairs (no new sampling) follows the imbalance of its pairs: with 327 of 432 negatives being "silent", it calls tools on everything (dev success 0.26, 27% unconfirmed bookings); balanced pairs fix the direction (0.45) but do not beat distillation.

### 3.5 Development decisions (dev split)
{chr(10).join(dev)}

### 3.6 Per category and per language (success, test)
{chr(10).join(percat)}

{chr(10).join(perlang)}

### 3.7 Public benchmarks (external check, our simplified protocol)
When2Call (NVIDIA, CC-BY-4.0; 200 questions per behaviour: call / ask for missing information / no tool fits; we score only the call-or-not decision, greedy, one turn) and BFCL-Hi (NVIDIA's Hindi BFCL v2 live: 258 simple, 258 irrelevance; AuditKIT `bfcl_arg_match`).
{chr(10).join(bench)}

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
"""
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/WRITEUP.md").write_text(W, encoding="utf-8")
(ROOT / "docs/ABSTRACT.md").write_text(ABSTRACT + "\n", encoding="utf-8")
print("words in abstract:", len(ABSTRACT.split()))
print("wrote docs/WRITEUP.md and docs/ABSTRACT.md")
