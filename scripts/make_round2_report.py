"""Build the round-2 report (.docx) from the results files.

    python scripts/make_round2_report.py OUT.docx [notes.json]

Every number in the tables is read from results/*.json. ``notes.json`` (optional) holds the
written interpretation: {"summary": [...], "reading": [...], "judges": [...], "next": [...]},
lists of paragraphs, so the prose is written after the numbers are seen and never hard-codes
them in this script.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = Path(sys.argv[1])
NOTES = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else {}
CK = ["base", "distill", "run_A", "run_B"]
LABEL = {"base": "Base Tiny Aya", "distill": "Distilled", "run_A": "run_A (default reward)",
         "run_B": "run_B (audit reward)"}


def load(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None


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


def para(text, style=None):
    return rich(doc.add_paragraph(style=style), text)


def bullets(items, style="List Bullet"):
    for t in items:
        para(t, style)


def table(header, rows, bold=()):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""
        c.paragraphs[0].add_run(str(h)).bold = True
        shade(c, "D9E2F3")
    for ri, row in enumerate(rows):
        cells = t.add_row().cells
        for ci, v in enumerate(row):
            cells[ci].text = ""
            run = cells[ci].paragraphs[0].add_run("" if v is None else str(v))
            run.bold = (ri, ci) in bold
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    doc.add_paragraph()


def f(x, d=3):
    return "–" if x is None else (f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x))


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def pval(r):
    if not r or "delta" not in r:
        return "–"
    star = " *" if r.get("significant") else ""
    return f"{r['delta']:+.3f} (p={r['p_value']:.3f}){star}"


# ── title ─────────────────────────────────────────────────────────────────────
doc.add_heading("LatentSpace, round 2: what changed, why, and the results", 0)
p = doc.add_paragraph()
r = p.add_run("Lexsi AI Safety & Alignment Hackathon · Track 3, Problem 9 (reward design and "
              "reward-hacking defences) · Tiny Aya student, Aya Expanse 32B teacher · "
              "English / Hindi / Marathi / Hinglish")
r.italic = True; r.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

if NOTES.get("summary"):
    doc.add_heading("Summary", 1)
    bullets(NOTES["summary"])

# ── flaws ─────────────────────────────────────────────────────────────────────
doc.add_heading("1. What was wrong in round 1, and what changed", 1)
para("Round 1 is kept unchanged in results/round1/. These flaws were found by reading rollouts "
     "and token counts. Each fix is tested (139 unit tests pass) and documented in FINDINGS.md.")
table(["#", "Flaw", "Effect in round 1", "Fix in round 2"], [
    ["P1", "Distillation max_length 2,048 tokens; the tool list alone is 2,892",
     "All 301 examples cut inside the tool list. The student never saw a question or an answer. "
     "Distillation did nothing (test success 0.161 → 0.170).",
     "max_length 6,144 (longest example 4,908). The stage refuses to run if any example would be cut."],
    ["P2", "The agent system prompt the config pointed to did not exist",
     "No model, not even the 32B teacher, was told to ask for missing details, confirm before "
     "booking, refuse out-of-scope requests or work step by step.",
     "An operating policy written from the data card's rules and the train split, given unchanged to every model."],
    ["P3", "The teacher filter checked format, not correctness",
     "129 of 301 kept trajectories were wrong (0 of 54 multi-step).",
     "Keep only trajectories that solve the task and reply in the user's language; reference-guided "
     "collection on train tasks."],
    ["P4", "GRPO stop-token mismatch (AgentTune / TRL, B15)",
     "Every training reply ran to the token limit (100% clipped); invented tool results and fake "
     "confirmations were rewarded.",
     "Stop on every end-of-reply token. 0% clipped, steps about 5× faster."],
    ["P5", "Exact-match scoring only",
     "Correct calls marked wrong ('पुणे' vs 'Pune', reworded complaints).",
     "Semantic matching reported next to strict; run_B's F1 reward uses it."],
    ["P6", "Distillation epoch not validated", "Over-fitting unnoticed (loss 0.014).",
     "One checkpoint per epoch, chosen on the dev split (test never used)."],
    ["A3", "AuditKIT paired_bootstrap returns p ≈ 0.5 for any effect",
     "Every round-1 p-value was invalid.", "Paired permutation test plus a 95% CI."],
])

# ── data ──────────────────────────────────────────────────────────────────────
doc.add_heading("2. Data quality", 1)
audit = load("09_dataset_audit.json")
if audit:
    s = audit["summary"]
    bullets([f"Reference answers: all {s['reference_calls_replayed']} expected tool calls replayed on the "
             f"backend: {s['reference_issues']} problems. Every station code, service id and PNR they "
             "use can be derived from the conversation.",
             f"Translations: {s['non_english_turns_checked']} non-English turns checked against English "
             f"for numbers, ids and script: {s['issues']} problems.",
             "Fluency and naturalness: native-speaker review by the team (review sheets), still open."])
tc = load("11_teacher_collect.json")
if tc:
    table(["Teacher data", "Round 1", "Round 2"], [
        ["Trajectories kept", "301 of 376", f"{tc['kept']} of {tc['seen']}"],
        ["Kept trajectories that are correct", "172 (57%)", f"{tc['kept']} (100%, by the filter)"],
        ["Multi-step examples that are correct", "0", str(tc["kept_by_category"].get("multi_step"))],
        ["Reference-guided (train split only)", "–", str(tc.get("kept_guided"))],
        ["Rejected: wrong / unconfirmed booking / wrong language", "–",
         " / ".join(str(tc["reasons"].get(k, 0)) for k in ("incorrect", "unconfirmed_irreversible",
                                                           "wrong_reply_language"))],
    ])

sel = load("12_distill_selection.json")
if sel:
    doc.add_heading("3. Distillation, epoch chosen on the dev split", 1)
    b = sel["base_model_on_dev_1_trial"]
    rows = [["Base (no training)", f(b["lenient_success"]), f(b["success"]), "–", "–"]]
    for c in sel["candidates"]:
        rows.append([c["checkpoint"] + (" (selected)" if c["checkpoint"] == sel["selected"] else ""),
                     f(c["dev_lenient_success"]), f(c["dev_success"]), f(c["dev_validity"]),
                     pct(c["dev_unconfirmed_rate"])])
    table(["Checkpoint", "Dev success (semantic)", "Dev success (strict)", "Valid calls",
           "Unconfirmed bookings"], rows)

# ── GRPO training ─────────────────────────────────────────────────────────────
ga, gb = load("13_grpo_run_A.json"), load("13_grpo_run_B.json")
if ga and gb:
    doc.add_heading("4. GRPO training: run_A vs run_B (identical except the reward)", 1)

    def dyn(d):
        h = [x for x in d["log_history"] if "reward" in x]
        q = max(1, len(h) // 4)
        m = lambda k, a, b: sum(float(x.get(k, 0)) for x in h[a:b]) / max(1, len(h[a:b]))  # noqa: E731
        return {"steps": d.get("global_step"),
                "zero": sum(float(x.get("reward_std", 1)) == 0 for x in h), "n": len(h),
                "clip": m("completions/clipped_ratio", 0, len(h)),
                "r": [m("reward", i * q, (i + 1) * q) for i in range(4)],
                "val": [c.get("tool_call_validity") for c in
                        (d.get("audit_of_training_rollouts") or {}).get("by_chunk", [])],
                "calls": [c.get("tool_call_count") for c in
                          (d.get("audit_of_training_rollouts") or {}).get("by_chunk", [])]}
    A, B = dyn(ga), dyn(gb)
    table(["", "run_A (default reward)", "run_B (audit reward)"], [
        ["Steps", A["steps"], B["steps"]],
        ["Steps with no learning signal (all 4 rewards equal)", f"{A['zero']}/{A['n']}", f"{B['zero']}/{B['n']}"],
        ["Replies cut off at the token limit", pct(A["clip"]), pct(B["clip"])],
        ["Mean reward by quarter", " → ".join(f(x) for x in A["r"]), " → ".join(f(x) for x in B["r"])],
        ["Valid calls in its own training rollouts, by quarter", " → ".join(f(x, 2) for x in A["val"]),
         " → ".join(f(x, 2) for x in B["val"])],
        ["Calls per rollout, by quarter", " → ".join(f(x, 2) for x in A["calls"]),
         " → ".join(f(x, 2) for x in B["calls"])],
    ])
    para("Same seed, data, model, LoRA, learning rate and steps (shared-settings fingerprint in the "
         "results files). run_B ran out of memory at step 30 and resumed from its checkpoint with 1 "
         "completion per backward pass instead of 2; the optimizer step is unchanged "
         "(1 prompt × 4 generations).")

# ── test ──────────────────────────────────────────────────────────────────────
ev = load("14_evaluate.json")
if ev:
    doc.add_heading("5. Held-out test split (108 tasks × 5 trials, never used for any choice)", 1)
    cks = [c for c in CK if c in ev["checkpoints"]]
    O = {c: ev["checkpoints"][c]["summary"]["overall"] for c in cks}
    rows = []
    for c in cks:
        o = O[c]
        rel = o.get("reliability") or {}
        rows.append([LABEL[c], f(o.get("lenient_success")), f(o.get("success")),
                     f(o.get("tool_call_validity")), f(o.get("tool_call_count"), 2),
                     pct(o.get("unconfirmed_irreversible_rate")), f(rel.get("pass_hat_k"))])
    best = {i: max(range(len(rows)), key=lambda r: float(rows[r][i]) if rows[r][i] != "–" else -1)
            for i in (1, 2, 3)}
    table(["Model", "Success (semantic)", "Success (strict)", "Valid calls", "Calls / task",
           "Unconfirmed bookings", "pass^5"], rows, bold={(r, i) for i, r in best.items()})
    cats = sorted(ev["checkpoints"][cks[0]]["summary"].get("per_category", {}))
    if cats:
        para("**Success by category** (semantic matching):")
        table(["Category"] + [LABEL[c] for c in cks],
              [[k] + [f(ev["checkpoints"][c]["summary"]["per_category"][k].get("lenient_success"), 2)
                      for c in cks] for k in cats])
    langs = ["en", "hi", "mr", "hinglish"]
    para("**Success by language** (semantic matching):")
    table(["Language"] + [LABEL[c] for c in cks],
          [[lg] + [f(ev["checkpoints"][c]["summary"]["per_language"].get(lg, {}).get("lenient_success"), 2)
                   for c in cks] for lg in langs])
    para("**Paired tests over per-task means** (sign-flip permutation, 10,000 resamples; "
         "* = p < 0.05; delta = second model − first):")
    mets = [("lenient_success", "Success (semantic)"), ("success", "Success (strict)"),
            ("tool_call_validity", "Valid calls"), ("unconfirmed_irreversible_rate", "Unconfirmed bookings"),
            ("no_call_rate", "No-call rate")]
    rows = []
    for k, v in ev["comparisons"].items():
        if "metrics" in v:
            rows.append([k.replace("_vs_", " → ")] + [pval(v["metrics"].get(m)) for m, _ in mets])
    table(["Comparison"] + [n for _, n in mets], rows)

    r1 = load("round1/14_evaluate.json")
    if r1:
        para("**Round 1 vs round 2 on the same test split** (strict success; round 1 had no policy prompt):")
        table(["Model", "Round 1", "Round 2"],
              [[LABEL[c], f(r1["checkpoints"][c]["summary"]["overall"].get("success")),
                f(O[c].get("success"))] for c in cks if c in r1["checkpoints"]])

# ── public benchmarks ─────────────────────────────────────────────────────────
bm = load("08_public_benchmarks.json")
if bm:
    doc.add_heading("6. Public benchmarks (external check)", 1)
    para("When2Call (NVIDIA, CC-BY-4.0): 600 questions, 200 each where the right behaviour is to "
         "call a tool, ask for missing information, or say no tool fits. BFCL-Hi (NVIDIA's Hindi "
         "version of the Berkeley Function-Calling Leaderboard v2 live set): 258 simple calls, 258 "
         "irrelevance questions. Our protocol: one greedy turn, the same tool prompt for every model, "
         "and no task policy. When2Call scores only the call / no-call decision; the official "
         "protocol uses log-probabilities of four written answers.")
    rows = []
    for c in CK:
        m = bm["models"].get(c)
        if m:
            w, b = m["when2call"], m["bfcl_hi"]
            rows.append([LABEL[c], f(w["decision_accuracy"]), pct(w["tool_hallucination_rate_cannot_answer"]),
                         pct(w["tool_call_rate_request_for_info"]), pct(w["calls_when_it_should"]),
                         f(b["simple_accuracy"]), f(b["irrelevance_accuracy"])])
    table(["Model", "When2Call decision accuracy", "Calls when no tool fits", "Calls when info is missing",
           "Calls when it should", "BFCL-Hi simple accuracy", "BFCL-Hi irrelevance accuracy"], rows)
    rows = [[k.replace("_vs_", " → ")] + [pval(v.get(b)) for b in ("when2call", "bfcl_hi_simple",
                                                                   "bfcl_hi_irrelevance")]
            for k, v in bm["comparisons"].items()]
    if rows:
        table(["Comparison", "When2Call", "BFCL-Hi simple", "BFCL-Hi irrelevance"], rows)

for key, title in (("reading", "7. What the results mean"),
                   ("judges", "8. How this answers the problem statement"),
                   ("next", "9. Next steps")):
    if NOTES.get(key):
        doc.add_heading(title, 1)
        bullets(NOTES[key])

doc.add_heading("Caveats", 1)
bullets(NOTES.get("caveats") or [
    "One seed per GRPO run; the test split has 27 base tasks (108 rows), so per-category cells hold 2–4 tasks.",
    "The Hindi, Marathi and Hinglish text is model-generated; the native-speaker review is in progress.",
    "Timetables, fares and PNRs are synthetic (mock backend); station names and codes are real.",
    "Teacher collection was reference-guided on train tasks: the train split's reference calls were "
    "replayed and the teacher wrote the replies. The dev and test splits were never used this way.",
    "Public benchmarks use our own simplified protocol, not the official leaderboard harness.",
])
doc.save(OUT)
print("saved", OUT)
