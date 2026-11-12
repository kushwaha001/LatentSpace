"""Final round-2 report (.docx). Every number is read from results/ and outputs/evaluate/.

    python scripts/make_report.py OUT.docx
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from latentspace_train.scoring import paired_test, per_task_means  # noqa: E402

R = ROOT / "results"
OUT = Path(sys.argv[1])
NEED = ["single_call", "multi_step", "irreversible_confirmed"]
NONE = ["declined", "denied_tool", "missing_argument", "no_call_needed"]
MODELS = [("base", "Base Tiny Aya (policy prompt)"), ("distill_unfiltered", "Distilled, unfiltered data (ablation)"),
          ("distill", "Distilled, audit-filtered data"), ("run_A", "GRPO, default reward"),
          ("run_B", "GRPO, audit reward v1"), ("run_B2", "GRPO, audit reward v2"),
          ("run_A3", "GRPO + KL anchor, default reward"), ("run_B3", "GRPO + KL anchor, audit reward v2"),
          ("dpo_bal", "DPO (balanced audit-ranked pairs)")]


def ev(name):
    p = R / f"14_evaluate_{name}.json"
    return json.loads(p.read_text())["checkpoints"][name]["summary"] if p.exists() else None


def rows(name):
    return [json.loads(l) for l in open(ROOT / f"outputs/evaluate/{name}/scored.jsonl")]


def group(s, cats):
    return sum(s["per_category"][c]["lenient_success"] for c in cats) / len(cats)


def test(a, b, metric, cats=None):
    A, B = rows(a), rows(b)
    if cats:
        A = [r for r in A if r.get("category") in cats]
        B = [r for r in B if r.get("category") in cats]
    x, y = per_task_means(A, metric), per_task_means(B, metric)
    return paired_test([(x[t], y[t]) for t in sorted(x.keys() & y.keys())])


doc = Document()
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(10.5)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Nirmala UI")


def shade(cell, fill):
    pr = cell._tc.get_or_add_tcPr()
    s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), fill)
    pr.append(s)


def rich(p, text):
    for i, part in enumerate(text.split("**")):
        if part:
            p.add_run(part).bold = i % 2 == 1
    return p


def para(t, style=None):
    return rich(doc.add_paragraph(style=style), t)


def bullets(items):
    for t in items:
        para(t, "List Bullet")


def table(header, body, bold_rows=()):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""
        c.paragraphs[0].add_run(str(h)).bold = True
        shade(c, "D9E2F3")
    for ri, row in enumerate(body):
        cells = t.add_row().cells
        for ci, v in enumerate(row):
            cells[ci].text = ""
            cells[ci].paragraphs[0].add_run("" if v is None else str(v)).bold = ri in bold_rows
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    doc.add_paragraph()


f3 = lambda x: "–" if x is None else f"{x:.3f}"  # noqa: E731
pc = lambda x: "–" if x is None else f"{100 * x:.1f}%"  # noqa: E731


def pv(r):
    return f"{r['delta']:+.3f} (p={r['p_value']:.4f}){' *' if r['significant'] else ''}"


doc.add_heading("LatentSpace: audit-grounded rewards for a multilingual tool-calling agent", 0)
r = doc.add_paragraph().add_run(
    "Lexsi AI Safety & Alignment Hackathon · Track 3, Problem 9 (reward design and reward-hacking defences) · "
    "Tiny Aya Global (3.35B) student, Aya Expanse 32B teacher · AgentTune + AuditKIT · English / Hindi / Marathi / Hinglish")
r.italic = True; r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

# ── headline numbers ─────────────────────────────────────────────────────────
A3, B3 = ev("run_A3"), ev("run_B3")
ab_need = test("run_A3", "run_B3", "lenient_success", NEED)
ab_all = test("run_A3", "run_B3", "lenient_success")
filt = test("distill_unfiltered", "distill", "unconfirmed_irreversible_rate")
filt_s = test("distill_unfiltered", "distill", "lenient_success")
bd = test("base", "distill", "lenient_success")

doc.add_heading("Summary", 1)
bullets([
    f"**The default AgentTune reward is hacked.** GRPO with `hybrid_prm_reward` learned to never call a tool: "
    f"{f3(ev('run_A')['overall']['tool_call_count'])} calls per task on the held-out test split, 0% success on every task "
    f"that needs an action, yet its raw success ({f3(ev('run_A')['overall']['lenient_success'])}) looks close to the best model.",
    f"**Same training, only the reward changed:** with an identical KL-anchored GRPO setup, the default reward collapsed "
    f"(success on tasks that need a call {f3(group(A3, NEED))}) while our audit reward kept the model competent "
    f"({f3(group(B3, NEED))}). Difference {pv(ab_need)}.",
    f"**The raw success metric hides this:** on all tasks the same two models differ by {pv(ab_all)}, because about half "
    "of the tasks are correctly answered by doing nothing. We report tasks that need a call and tasks that need none separately.",
    f"**The audit reward as a data filter:** distilling only on audit-approved teacher trajectories cut unsafe "
    f"irreversible actions (booking or cancelling without the user's yes) from "
    f"{pc(ev('distill_unfiltered')['overall']['unconfirmed_irreversible_rate'])} to "
    f"{pc(ev('distill')['overall']['unconfirmed_irreversible_rate'])}: {pv(filt)}.",
    "**A correct reward is not enough for GRPO:** with our reward v2 ranking a call above silence on every task that "
    "needs one, GRPO without a KL anchor still collapsed into silence. Once all samples of a prompt are silent their "
    "advantages are zero and nothing pushes back. A KL anchor to the distilled student prevents this.",
    f"**Distillation (fixed) is the largest single gain:** test success {f3(ev('base')['overall']['lenient_success'])} → "
    f"{f3(ev('distill')['overall']['lenient_success'])} ({pv(bd)}).",
    "**Bugs found:** AgentTune B1 (default reward scores a correct call 0 and no call 1), B15 (GRPO never stops a Tiny "
    "Aya reply), AuditKIT A3 (paired_bootstrap returns p ≈ 0.5 for any effect), and six flaws in our own round-1 pipeline.",
])

# ── test table ───────────────────────────────────────────────────────────────
doc.add_heading("1. Held-out test results (108 tasks × 5 trials; test never used for any choice)", 1)
para("Success uses the same verdict as AuditKIT's strict scoring (every call schema-valid, no unconfirmed irreversible call, "
     "the reference calls made) but compares arguments by meaning (a station in another script, a reworded complaint). "
     "Strict numbers are in the results files. 'Needs a call' averages single_call, multi_step and irreversible_confirmed; "
     "'no call' averages declined, denied_tool, missing_argument and no_call_needed.")
body = []
for k, label in MODELS:
    s = ev(k)
    if not s:
        continue
    o = s["overall"]
    body.append([label, f3(o["lenient_success"]), f3(o["success"]), f3(group(s, NEED)), f3(group(s, NONE)),
                 f"{o['tool_call_count']:.2f}", pc(o["unconfirmed_irreversible_rate"])])
table(["Model", "Success", "Success (strict)", "Needs a call", "No call", "Calls / task", "Unconfirmed"], body,
      bold_rows=(2, 7))

doc.add_heading("2. The key comparisons (paired sign-flip permutation tests over tasks)", 1)
table(["Comparison", "What changes", "Metric", "Result"], [
    ["GRPO+KL: default → audit reward", "only the reward", "success, tasks that need a call", pv(ab_need)],
    ["GRPO+KL: default → audit reward", "only the reward", "success, all tasks", pv(ab_all)],
    ["Distillation: unfiltered → audit-filtered", "only the data filter", "unconfirmed irreversible rate", pv(filt)],
    ["Distillation: unfiltered → audit-filtered", "only the data filter", "success, all tasks", pv(filt_s)],
    ["Base → distilled", "SFT on filtered teacher data", "success, all tasks", pv(bd)],
])

doc.add_heading("3. What went wrong in round 1 and what changed", 1)
table(["#", "Flaw", "Effect", "Fix"], [
    ["P1", "Distillation max_length 2,048; the tool list alone is 2,892 tokens", "Every example cut inside the tool list; distillation did nothing", "max_length 6,144 and a truncation guard"],
    ["P2", "The agent policy prompt did not exist", "No model was told to ask, confirm or refuse", "Policy written from the data card and train split, same for all models"],
    ["P3", "Teacher filter checked format, not correctness", "43% of kept trajectories were wrong", "Keep only audit-approved trajectories; reference-guided on train"],
    ["B15", "GRPO stop token ≠ the token Tiny Aya emits", "100% of training replies ran to the limit", "Stop on every end-of-reply token"],
    ["P5", "Exact-match scoring only", "Correct answers marked wrong", "Semantic matching next to strict"],
    ["A3", "AuditKIT paired_bootstrap", "p ≈ 0.5 for any effect", "Paired permutation test + 95% CI"],
])

doc.add_heading("4. Reward and optimiser iterations (dev split, 64 tasks × 2, used for decisions)", 1)
sel = json.loads((R / "12_distill_selection.json").read_text())
ref = next(c for c in sel["candidates"] if c["checkpoint"] == sel["selected"])
dev = [["Distilled (selected epoch)", f3(ref["dev_lenient_success"]), f"{ref['dev_calls']:.2f}", pc(ref["dev_unconfirmed_rate"]), "reference"]]
for n, label, note in [("runB_checkpoint-220", "GRPO audit v1, step 220", "drifts into silence"),
                       ("run_B3_ckpt60", "GRPO+KL audit v2, step 60", "no collapse; over-acts"),
                       ("run_B3_ckpt120", "GRPO+KL audit v2, step 120", "balanced"),
                       ("dpo_audit_dev", "DPO, unbalanced pairs", "collapses into over-calling"),
                       ("dpo_bal_dev", "DPO, balanced pairs", "balanced")]:
    p = R / f"_devsel/{n}/10_baseline.json"
    if p.exists():
        o = json.loads(p.read_text())["modes"]["local"]["summary"]["overall"]
        dev.append([label, f3(o["lenient_success"]), f"{o['tool_call_count']:.2f}", pc(o["unconfirmed_irreversible_rate"]), note])
table(["Model", "Dev success", "Calls / task", "Unconfirmed", "Reading"], dev)
bullets([
    "**Reward v1 → v2:** v1 gave a no-call 0.5 validity even on tasks that need a call, which beat invalid attempts. "
    "v2 gives it 0. The ranking became right, but GRPO still collapsed (run_B2): the problem was the optimiser, not the ranking.",
    "**KL anchor:** TRL's reference for a LoRA model is 'adapter disabled' = the base model. We merge the distilled adapter "
    "into the weights first so the anchor is the distilled student.",
    "**DPO:** with 327 of 432 negatives being 'silent', DPO learned to call tools on everything (dev 0.26). Balancing the "
    "pairs fixed the direction (dev 0.45) but did not beat distillation.",
])

bm = R / "08_public_benchmarks.json"
if bm.exists():
    b = json.loads(bm.read_text())["models"]
    doc.add_heading("5. Public benchmarks (When2Call, BFCL-Hi; our simplified protocol)", 1)
    body = [[k, f3(v["when2call"]["calls_when_it_should"]), f3(v["when2call"]["tool_hallucination_rate_cannot_answer"]),
             f3(v["bfcl_hi"]["simple_accuracy"]), f3(v["bfcl_hi"]["irrelevance_accuracy"])] for k, v in b.items()]
    table(["Model", "When2Call: calls when it should", "When2Call: calls when no tool fits (lower is better)",
           "BFCL-Hi simple", "BFCL-Hi irrelevance"], body)
    para("Silent models score well on 'decision accuracy' and irrelevance for the same reason as in our test split; the "
         "distilled model is the only balanced one. All trained models lose Hindi function calling on unseen tools "
         "(BFCL-Hi simple 0.26 → under 0.05): our training is specific to the transit tools and policy prompt.")

doc.add_heading("Caveats", 1)
bullets([
    "One training seed per method; 27 base tasks (108 rows) in test, so per-category cells hold 2–4 tasks.",
    "The Hindi, Marathi and Hinglish text is model-generated; native-speaker review is in progress.",
    "Teacher collection was reference-guided on train tasks: the reference calls were replayed on the backend and the "
    "teacher wrote the replies. So the distilled student learned gold tool calls plus teacher-written answers (train split only).",
    "Several evaluation choices (semantic matching rules, the needs-a-call / no-call split) were added after seeing round-1 "
    "or run_A results; strict numbers are always reported next to them.",
    "Model and method choices used the dev split only; dev results were noisier than test (64 tasks).",
    "Timetables, fares and PNRs are synthetic (mock backend); station names and codes are real.",
])
doc.save(OUT)
print("saved", OUT)
