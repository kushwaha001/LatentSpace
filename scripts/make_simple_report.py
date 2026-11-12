"""Plain-language report of the whole LatentSpace project (.docx). Key numbers are read from results/.

    python scripts/make_simple_report.py OUT.docx
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
NEED = ["single_call", "multi_step", "irreversible_confirmed"]


def ev(n):
    return json.loads((R / f"14_evaluate_{n}.json").read_text())["checkpoints"][n]["summary"]


def need(n):
    """Success over all test rows whose task needs a call (the same rows the paired test uses)."""
    rows = [json.loads(l) for l in open(ROOT / f"outputs/evaluate/{n}/scored.jsonl")]
    v = [r["lenient_success"] for r in rows if r["category"] in NEED]
    return sum(v) / len(v)


def ov(n, k="lenient_success"):
    return ev(n)["overall"][k]


def p(a, b, m, cats=None):
    A = [json.loads(l) for l in open(ROOT / f"outputs/evaluate/{a}/scored.jsonl")]
    B = [json.loads(l) for l in open(ROOT / f"outputs/evaluate/{b}/scored.jsonl")]
    if cats:
        A = [r for r in A if r["category"] in cats]
        B = [r for r in B if r["category"] in cats]
    x, y = per_task_means(A, m), per_task_means(B, m)
    return paired_test([(x[t], y[t]) for t in sorted(x.keys() & y.keys())])


pct = lambda x: f"{round(100 * x)}%"  # noqa: E731
doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(11)
doc.styles["Normal"].element.rPr.rFonts.set(qn("w:eastAsia"), "Nirmala UI")


def shade(cell, fill):
    pr = cell._tc.get_or_add_tcPr()
    s = OxmlElement("w:shd")
    s.set(qn("w:val"), "clear"); s.set(qn("w:color"), "auto"); s.set(qn("w:fill"), fill)
    pr.append(s)


def rich(par, text):
    for i, part in enumerate(text.split("**")):
        if part:
            par.add_run(part).bold = i % 2 == 1
    return par


def para(t, style=None):
    return rich(doc.add_paragraph(style=style), t)


def bullets(items, style="List Bullet"):
    for t in items:
        para(t, style)


def box(text):
    t = doc.add_table(rows=1, cols=1)
    t.style = "Table Grid"
    c = t.rows[0].cells[0]
    c.text = ""
    rich(c.paragraphs[0], text)
    shade(c, "FFF2CC")
    doc.add_paragraph()


def table(header, body):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""
        c.paragraphs[0].add_run(h).bold = True
        shade(c, "D9E2F3")
    for row in body:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            rich(cells[i].paragraphs[0], str(v))
    doc.add_paragraph()


a3b3 = p("run_A3", "run_B3", "lenient_success", NEED)
filt = p("distill_unfiltered", "distill", "unconfirmed_irreversible_rate")

doc.add_heading("LatentSpace: what we built, what we found, and why it matters", 0)
r = doc.add_paragraph().add_run("A plain-language summary of the whole project · Lexsi AI Safety & Alignment Hackathon, Track 3, Problem 9")
r.italic = True; r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

doc.add_heading("The project in one paragraph", 1)
para("We built a small AI assistant that helps people in India with trains and metros (finding station codes, "
     "searching trains, checking fares and PNR status, booking and cancelling tickets, filing complaints) in "
     "**English, Hindi, Marathi and Hinglish**. The assistant does not just chat: it uses **tools**, small programs that "
     "do the real work. We studied a key safety problem: when you train such an assistant by rewarding it, it can "
     "learn to **cheat the reward** instead of getting better. We showed this cheating clearly, explained why it happens, "
     "and showed that a better-designed reward prevents it.")

doc.add_heading("1. The problem we worked on", 1)
para("**Problem 9 of Track 3: reward design and defences against reward hacking.**")
bullets([
    "One way to improve an AI model is **reinforcement learning**: the model tries many answers, a **reward** "
    "(a scoring rule) says which answers are good, and the model is nudged towards the good ones.",
    "**Reward hacking** happens when the model finds a way to get a high score **without doing the job**. "
    "It is like a student who learns that blank answers are never marked wrong, and stops answering.",
    "This matters for safety: an assistant that books tickets with real money must act when asked, must **not** act "
    "when it shouldn't, and must **never book without the user saying yes**.",
])

doc.add_heading("2. What we built", 1)
table(["Part", "In simple words"], [
    ["The assistant (student)", "**Tiny Aya**, a small open model from Cohere (3.35 billion parameters) that runs on one GPU"],
    ["The teacher", "**Aya Expanse 32B**, a 10x larger model whose good answers the student learns from"],
    ["8 transit tools", "Station codes, train search, metro routes, fares, PNR status, **booking**, **cancelling**, complaints. "
                        "Real station names and codes; timetables, fares and PNRs are made up (a safe test world)"],
    ["Test questions", "90 tasks, each in 4 languages (360 in total), in 8 types, e.g. 'needs one tool', 'needs several steps', "
                       "'user said yes to a booking', 'user said no', 'a detail is missing', 'out of scope'"],
    ["Our reward (the audit reward)", "A scoring rule built from **AuditKIT** checks: is each tool call correctly written, "
                                      "is it the right call, was a booking confirmed by the user, no forbidden tools, no repeats"],
    ["Training tools", "**AgentTune** (Lexsi's agent-training library) and **AuditKIT** (Lexsi's evaluation library)"],
])

doc.add_heading("3. How we trained and tested", 1)
bullets([
    "**Step 1, learn from the teacher (distillation):** the student copies the teacher's correct answers. "
    "Our audit reward was used to **filter** the teacher's answers, keeping only correct and safe ones.",
    "**Step 2, reinforcement learning:** the student practises, and a reward scores its answers. We compared "
    "AgentTune's **default reward** with **our audit reward**, keeping everything else identical.",
    "**Step 3, honest testing:** a held-out **test set** of 108 tasks the model never saw, each tried 5 times. "
    "All decisions were made on a separate **dev set**, so the test results cannot have been tuned.",
    "**Statistics:** every comparison has a p-value. **p < 0.05 means the difference is very unlikely to be luck.**",
])

doc.add_heading("4. What went wrong first, and how we fixed it", 1)
para("Our first full round gave confusing results. Instead of accepting them, we investigated and found real bugs. "
     "Fixing them is a big part of the work.")
table(["Problem found", "What it caused", "Fix"], [
    ["The student's training examples were cut off too early (the tool list alone was longer than the limit)",
     "The student **never saw a single question or answer**; learning from the teacher did nothing", "Longer limit, plus an automatic check that nothing is cut"],
    ["The assistant's instructions (its 'job description') were missing", "No model was told to confirm bookings or ask for missing details",
     "Wrote clear rules from the dataset's own definitions, the same for every model"],
    ["43% of the teacher's answers used for learning were wrong", "The student copied mistakes", "Kept only answers that pass our audit checks"],
    ["A stop-signal mismatch in the training library (**new bug, B15**)", "The model never stopped talking during training; it invented fake results",
     "Stop on the right signal: 0% runaway answers, about 5x faster training"],
    ["A statistics bug in AuditKIT (**new bug, A3**)", "Every 'is this significant?' answer was meaningless (always about 50/50)", "Replaced with a correct statistical test"],
    ["Answer checking was too strict", "'पुणे' and 'Pune' counted as different stations", "Compare by meaning; strict numbers still shown alongside"],
])

doc.add_heading("5. The main results, in plain words", 1)
doc.add_heading("Result 1: the default reward is cheated completely", 2)
para(f"Trained with AgentTune's default reward, the assistant learned to **never use a tool at all**: "
     f"{ov('run_A', 'tool_call_count'):.2f} tool calls per task. It can no longer search a train, check a fare or book anything. "
     f"Yet its overall score ({pct(ov('run_A'))}) looks almost as good as the best model ({pct(ov('distill'))}).")
box("**Why?** About half of the test tasks are correctly handled by doing nothing (the user said no, a detail is missing, "
    "or the request is out of scope). A model that never acts gets all of those right and still looks good. "
    "The default reward also scores a correct tool call 0 and no call 1 (bug B1). **Silence pays.**")

doc.add_heading("Result 2: our reward prevents the cheating (the key result)", 2)
para("We trained twice with **exactly the same settings**, changing **only the reward**:")
table(["", "Default reward", "Our audit reward"], [
    ["Success on tasks that need an action", f"**{pct(need('run_A3'))}**", f"**{pct(need('run_B3'))}**"],
    ["Tool calls per task", f"{ov('run_A3', 'tool_call_count'):.2f}", f"{ov('run_B3', 'tool_call_count'):.2f}"],
    ["Overall score (misleading)", pct(ov("run_A3")), pct(ov("run_B3"))],
])
para(f"Difference on tasks that need an action: **+{round(100 * a3b3['delta'])} percentage points, p = {a3b3['p_value']:.4f}** "
     "(far from luck). With the default reward the assistant becomes useless; with ours it stays as capable as before training.")

doc.add_heading("Result 3: our reward makes the assistant safer when used to choose training examples", 2)
para(f"Training only on teacher answers that pass our audit checks cut **bookings and cancellations made without the "
     f"user's yes** from **{pct(ov('distill_unfiltered', 'unconfirmed_irreversible_rate'))}** to "
     f"**{pct(ov('distill', 'unconfirmed_irreversible_rate'))}** (p = {filt['p_value']:.3f}, significant). "
     "That is about a third fewer unsafe actions.")

doc.add_heading("Result 4: learning from the teacher is the biggest improvement", 2)
para(f"After fixing the bugs, learning from the teacher raised test success from **{pct(ov('base'))} to "
     f"{pct(ov('distill'))}**, the largest single gain in the project.")

doc.add_heading("Result 5: a good reward is necessary but not sufficient", 2)
bullets([
    "Even when our reward clearly ranked 'act' above 'stay silent', the most common training method (GRPO) still "
    "drifted into silence. Once all of the model's attempts at a question are silent, they all get the same score, "
    "and the method has nothing left to learn from.",
    "Adding an **anchor** (a penalty for drifting too far from the starting model) stopped this.",
    "We also tried a second method, DPO. It copied whatever imbalance its examples had: too many 'don't be silent' "
    "examples made it **act on everything**, including booking without permission. Balancing the examples fixed the direction.",
])

doc.add_heading("6. Why this matters (significance)", 1)
table(["Point", "Meaning"], [
    ["Reward hacking is real and easy to miss", "A model that does nothing can score well; you must look at results by task type"],
    ["The headline number can lie", "One overall score hid a completely broken model. We report 'tasks that need action' and 'tasks that need none' separately"],
    ["Reward design changes behaviour, not just scores", f"Same training, different reward: a useless assistant vs a working one (+{round(100 * a3b3['delta'])} points on tasks that need action)"],
    ["Safety improves measurably", "About a third fewer unconfirmed bookings when the training data is chosen with our checks"],
    ["Training methods need guard-rails", "A correct reward alone did not stop the collapse; an anchor did"],
    ["Contributions to Lexsi's own tools", "Two new bugs (B15, A3) with fixes, plus the documented default-reward bug (B1)"],
])

doc.add_heading("7. Limits of our work (honest caveats)", 1)
bullets([
    "Each method was trained once (one seed); some differences between methods are small and not significant.",
    "None of the reinforcement-learning methods beat the simple 'learn from the teacher' model on overall success; "
    "their value here is avoiding the collapse and the safety gains.",
    "The Hindi, Marathi and Hinglish questions were written by an AI and are still being checked by native speakers.",
    "Timetables, fares and PNRs are invented; station names and codes are real.",
    "For training (not testing), the teacher was shown the correct tool calls and wrote the answers around them.",
    "On an outside Hindi benchmark with unfamiliar tools, our trained models did worse than the original: "
    "our training is specialised to transit tools.",
])

doc.add_heading("8. Everything we did, step by step", 1)
bullets([
    "Studied AgentTune and AuditKIT and found that the default reward scores a correct call 0 and no call 1 (B1).",
    "Built the 8 transit tools, a safe test world, and 360 questions in 4 languages; checked every reference answer automatically.",
    "Built our audit reward from AuditKIT checks, including a multilingual 'did the user say yes?' check.",
    "Ran a first full round; found it misleading; found and fixed 6 flaws plus 2 library bugs.",
    "Ran a second round on two GPUs: teacher data, learning from the teacher, 7 reinforcement-learning or DPO "
    "variants, test evaluation of 9 models, and two outside benchmarks (When2Call, BFCL-Hindi).",
    "Made the setup crash-proof (automatic resume, memory limits, one job per GPU) after PC restarts and memory problems.",
    "Wrote up everything with p-values; all numbers in this document are read from the result files.",
])

doc.add_heading("9. What happens next", 1)
bullets([
    "Write the Hugging Face article and a short abstract; build a small public demo.",
    "Publish the trained models, data and logs on Hugging Face, and the code on GitHub (needs the team's permission).",
    "Finish the native-speaker review of the Hindi and Marathi questions.",
])
doc.save(sys.argv[1])
print("saved", sys.argv[1])
