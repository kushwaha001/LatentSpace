"""Experiment 08: two public tool-calling benchmarks, as an external check of our checkpoints.

- **When2Call** (NVIDIA, CC-BY-4.0, ``nvidia/When2Call``, ``test/when2call_test_mcq.jsonl``):
  each question has one correct behaviour: ``tool_call``, ``request_for_info`` (a required
  value is missing) or ``cannot_answer`` (no tool fits). It measures *when (not) to call*.
- **BFCL-Hi** (NVIDIA's Hindi version of the Berkeley Function-Calling Leaderboard v2 live
  set, CC-BY-4.0, ``nvidia/BFCL-Hi``): Hindi questions, English function schemas.
  ``live_simple`` (one function, a call is correct) and ``live_irrelevance`` (no function
  fits, no call is correct).

Protocol (ours, not the official leaderboards'): one generated turn per question, greedy
decoding, the same AgentTune Cohere fallback prompt the transit agent uses (tool schemas in
the system prompt, no task policy), calls parsed with AgentTune's ``_extract_tool_calls``.

- When2Call: we score the *decision* only (did the model call a tool?), which is
  deterministic and needs no LLM judge. The official protocol compares log-probabilities of
  four written answers and also separates request_for_info from cannot_answer; we do not.
  A seeded, class-balanced sample (default 200 per class) keeps it to minutes per model.
- BFCL-Hi simple: AuditKIT's ``bfcl_arg_match`` (BFCL's accepted-value lists) on the one
  expected call; irrelevance: correct iff no call.

    python experiments/08_public_benchmarks.py --name base
    python experiments/08_public_benchmarks.py --name run_B --adapter outputs/grpo/run_B/adapter
    python experiments/08_public_benchmarks.py --report        # table + paired tests

Writes ``outputs/benchmarks/<name>/*.jsonl`` and ``results/08_public_benchmarks*.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BENCH = Path("/mnt/d/hf_cache/bench")
OUT = ROOT / "outputs" / "benchmarks"


def _as_schema(tool) -> dict:
    t = json.loads(tool) if isinstance(tool, str) else tool
    return t if "function" in t else {"type": "function", "function": t}


def load_when2call(per_class: int, seed: int) -> list[dict]:
    rows = [json.loads(l) for l in open(BENCH / "When2Call/test/when2call_test_mcq.jsonl")]
    rng = random.Random(seed)
    out = []
    for cls in ("tool_call", "request_for_info", "cannot_answer"):
        pool = sorted((r for r in rows if r["correct_answer"] == cls), key=lambda r: r["uuid"])
        for r in rng.sample(pool, min(per_class, len(pool))):
            out.append({"id": r["uuid"], "bench": "when2call", "label": cls,
                        "question": r["question"], "tools": [_as_schema(t) for t in r["tools"]],
                        "target_tool": r.get("target_tool")})
    return out


def load_bfcl_hi(n_irrelevance: int, seed: int) -> list[dict]:
    d = BENCH / "BFCL-Hi"
    gold = {json.loads(l)["id"]: json.loads(l)["ground_truth"]
            for l in open(d / "possible_answer/BFCL_v2_live_simple.json")}
    out = []
    for l in open(d / "BFCL_v2_live_simple.json"):
        r = json.loads(l)
        q = r["question"][0] if r["question"] and isinstance(r["question"][0], list) else r["question"]
        out.append({"id": r["id"], "bench": "bfcl_hi_simple", "label": "tool_call",
                    "messages": q, "tools": [_as_schema(t) for t in r["function"]],
                    "gold": gold[r["id"]]})
    irr = [json.loads(l) for l in open(d / "BFCL_v2_live_irrelevance.json")]
    rng = random.Random(seed)
    for r in rng.sample(sorted(irr, key=lambda r: r["id"]), min(n_irrelevance, len(irr))):
        q = r["question"][0] if r["question"] and isinstance(r["question"][0], list) else r["question"]
        out.append({"id": r["id"], "bench": "bfcl_hi_irrelevance", "label": "no_call",
                    "messages": q, "tools": [_as_schema(t) for t in r["function"]]})
    return out


def build_prompt(tok, item: dict) -> str:
    from agenttune.agentic.rollout_engines.rollout_factory import DEFAULT_TOOLS_FALLBACK_PROMPT

    schemas = json.dumps(item["tools"], indent=2, ensure_ascii=False)
    system = DEFAULT_TOOLS_FALLBACK_PROMPT.format(tools=schemas)
    msgs = item.get("messages") or [{"role": "user", "content": item["question"]}]
    msgs = [{"role": "system", "content": system}] + [m for m in msgs if m["role"] != "system"]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def parse_calls(text: str) -> list[dict]:
    from agenttune.agentic.rollout_engines.rollout_factory import _extract_tool_calls

    out = []
    for c in _extract_tool_calls(None, text) or []:
        fn = c.get("function", c)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        out.append({"name": fn.get("name"), "arguments": args})
    return out


def judge(item: dict, calls: list[dict]) -> dict:
    called = len(calls) > 0
    res = {"called": called}
    if item["bench"] == "when2call":
        res["correct"] = called == (item["label"] == "tool_call")
        if item["label"] == "tool_call" and called and item.get("target_tool"):
            try:
                want = json.loads(item["target_tool"])["name"]
            except (TypeError, ValueError, KeyError):
                want = None
            res["right_tool"] = bool(want) and calls[0]["name"] == want
    elif item["bench"] == "bfcl_hi_irrelevance":
        res["correct"] = not called
    else:
        from auditkit import bfcl_arg_match

        ok = False
        if len(calls) == 1 and isinstance(calls[0]["arguments"], dict):
            for g in item["gold"]:
                (name, ref), = g.items()
                if calls[0]["name"] == name and bfcl_arg_match(name, calls[0]["arguments"], ref):
                    ok = True
        res["correct"] = ok
        res["right_tool"] = called and any(calls[0]["name"] in g for g in item["gold"])
    return res


def run_model(args) -> None:
    import torch

    from latentspace_train.engines import load_local_model

    spec = {"model": args.model, "quantization": None, "adapter": args.adapter,
            "base_adapter": args.base_adapter,
            "attn_implementation": "sdpa"}
    model, tok = load_local_model(spec, "bf16")
    tok.padding_side = "left"
    items = load_when2call(args.per_class, args.seed) + load_bfcl_hi(args.n_irrelevance, args.seed)
    if args.limit:
        items = items[:: max(1, len(items) // args.limit)][: args.limit]
    prompts = [build_prompt(tok, it) for it in items]
    order = sorted(range(len(items)), key=lambda i: len(prompts[i]))
    outd = OUT / args.name
    outd.mkdir(parents=True, exist_ok=True)
    rows, t0 = [None] * len(items), time.time()
    # resumable: every finished row is appended to rows_partial.jsonl and skipped on a rerun
    part = outd / "rows_partial.jsonl"
    pos = {(it["bench"], it["id"]): i for i, it in enumerate(items)}
    if part.exists():
        for line in open(part, encoding="utf-8"):
            r = json.loads(line)
            if (r["bench"], r["id"]) in pos:
                rows[pos[(r["bench"], r["id"])]] = r
        print(f"  resume: {sum(r is not None for r in rows)} rows already done", flush=True)
    order = [i for i in order if rows[i] is None]
    eos = model.generation_config.eos_token_id
    for b in range(0, len(order), args.batch_size):
        idx = order[b:b + args.batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False, truncation=True, max_length=6000).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=eos)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, text in zip(idx, texts):
            calls = parse_calls(text)
            it = items[i]
            rows[i] = {"id": it["id"], "bench": it["bench"], "label": it["label"],
                       "response": text, "calls": calls, **judge(it, calls)}
            with open(part, "a", encoding="utf-8") as f:
                f.write(json.dumps(rows[i], ensure_ascii=False) + "\n")
        if (b // args.batch_size) % 10 == 0:
            print(f"  {b + len(idx)}/{len(items)} ({time.time() - t0:.0f}s)", flush=True)
    with open(outd / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = summarise(rows)
    summ.update({"name": args.name, "model": args.model, "adapter": args.adapter,
                 "seconds": round(time.time() - t0, 1), "n": len(rows), "seed": args.seed,
                 "decoding": "greedy", "max_new_tokens": args.max_new_tokens})
    (ROOT / f"results/08_public_benchmarks_{args.name}.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))


def _rate(rows, key="correct"):
    v = [float(r[key]) for r in rows if key in r]
    return round(sum(v) / len(v), 4) if v else None


def summarise(rows: list[dict]) -> dict:
    w = [r for r in rows if r["bench"] == "when2call"]
    by = lambda lab: [r for r in w if r["label"] == lab]  # noqa: E731
    simple = [r for r in rows if r["bench"] == "bfcl_hi_simple"]
    irr = [r for r in rows if r["bench"] == "bfcl_hi_irrelevance"]
    return {
        "when2call": {
            "n": len(w),
            "decision_accuracy": _rate(w),
            "calls_when_it_should": _rate(by("tool_call"), "called"),
            "right_tool_when_calling": _rate(by("tool_call"), "right_tool"),
            "tool_hallucination_rate_cannot_answer": _rate(by("cannot_answer"), "called"),
            "tool_call_rate_request_for_info": _rate(by("request_for_info"), "called"),
        },
        "bfcl_hi": {
            "n_simple": len(simple), "simple_accuracy": _rate(simple),
            "simple_right_function": _rate(simple, "right_tool"),
            "simple_call_rate": _rate(simple, "called"),
            "n_irrelevance": len(irr), "irrelevance_accuracy": _rate(irr),
        },
    }


def report(names: list[str]) -> None:
    from latentspace_train.scoring import paired_test

    data = {n: {r["id"]: r for r in map(json.loads, open(OUT / n / "rows.jsonl"))}
            for n in names if (OUT / n / "rows.jsonl").exists()}
    out = {"models": {n: json.loads((ROOT / f"results/08_public_benchmarks_{n}.json").read_text())
                      for n in data}, "comparisons": {}}
    names = list(data)
    for a in names:
        for b in names:
            if a >= b and not (a == "run_A" and b == "run_B"):
                continue
            if (a, b) not in [("base", "run_B"), ("base", "run_A"), ("run_A", "run_B"),
                              ("distill", "run_B"), ("base", "distill")]:
                continue
            cmp = {}
            for bench in ("when2call", "bfcl_hi_simple", "bfcl_hi_irrelevance"):
                ids = sorted(i for i, r in data[a].items() if r["bench"] == bench and i in data[b])
                cmp[bench] = paired_test([(float(data[a][i]["correct"]), float(data[b][i]["correct"]))
                                          for i in ids])
            out["comparisons"][f"{a}_vs_{b}"] = cmp
    (ROOT / "results/08_public_benchmarks.json").write_text(json.dumps(out, indent=2))
    L = ["# Experiment 08: public benchmarks (When2Call, BFCL-Hi)", "",
         "| Model | When2Call decision acc. | calls when it should | tool hallucination (cannot answer) | "
         "calls when info is missing | BFCL-Hi simple acc. | BFCL-Hi irrelevance acc. |",
         "|---|---|---|---|---|---|---|"]
    for n, m in out["models"].items():
        w, b = m["when2call"], m["bfcl_hi"]
        L.append(f"| {n} | {w['decision_accuracy']:.3f} | {w['calls_when_it_should']:.3f} | "
                 f"{w['tool_hallucination_rate_cannot_answer']:.3f} | "
                 f"{w['tool_call_rate_request_for_info']:.3f} | {b['simple_accuracy']:.3f} | "
                 f"{b['irrelevance_accuracy']:.3f} |")
    L += ["", "Paired sign-flip permutation tests on per-question correctness (delta = second − first):", ""]
    for k, c in out["comparisons"].items():
        L.append(f"- {k}: " + "; ".join(
            f"{bn} {r['delta']:+.3f} (p {r['p_value']:.3f}{' *' if r['significant'] else ''})"
            for bn, r in c.items() if "delta" in r))
    (ROOT / "results/08_public_benchmarks.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name")
    ap.add_argument("--model", default="CohereLabs/tiny-aya-global")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--base-adapter", default=None, help="LoRA merged into the weights first")
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--n-irrelevance", type=int, default=258)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(["base", "distill", "run_A", "run_B"])
    else:
        run_model(a)
