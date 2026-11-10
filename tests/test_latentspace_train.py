"""CPU unit tests for src/latentspace_train (stub tools, mocked HF router, no network).

The GPU path itself is covered by the dry runs (python -m latentspace_train <stage> --dry-run).
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from types import SimpleNamespace

import pytest

from latentspace_train import config as C
from latentspace_train import stub
from latentspace_train.engines import (
    Budget,
    BudgetExceeded,
    _prompt_mode_messages,
    fallback_system_prompt,
    make_router_engine,
)
from latentspace_train.rollouts import build_rollout_fn, run_rollouts
from latentspace_train.scoring import (
    bootstrap_compare,
    make_audit_rewards,
    pass_k,
    score_records,
    summarise,
)
from latentspace_train.tasks import TasksUnavailable, load_split, load_tools, to_dataset_rows

BASE_CFG = {"tasks": {"source": "stub"}, "rewards": {"audit": {}}}


# ── config ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("variant", ["global", "fire", "water", "earth"])
def test_tiny_aya_variants_resolve(variant):
    full = f"CohereLabs/tiny-aya-{variant}"
    assert C.resolve_model(f"tiny-aya-{variant}") == full
    assert C.resolve_model(variant) == full
    assert C.resolve_model(full) == full
    cfg = C.load_config("configs/10_baseline.yaml", [f"student.model=tiny-aya-{variant}"])
    assert cfg["student"]["model"] == full


def test_stage_configs_load_and_extend_base():
    for stage, path in {
        "baseline": "configs/10_baseline.yaml",
        "teacher_collect": "configs/11_teacher_collect.yaml",
        "distill": "configs/12_distill.yaml",
        "grpo": "configs/13_grpo.yaml",
        "evaluate": "configs/14_evaluate.yaml",
    }.items():
        cfg = C.load_config(path)
        assert cfg["seed"] == 42, stage
        assert cfg["hub"]["namespace"] == "kanishk001", stage
        assert all(r.startswith("kanishk001/") for r in cfg["hub"]["repos"].values())
        assert cfg["student"]["model"] == "CohereLabs/tiny-aya-global"
    t = C.load_config("configs/11_teacher_collect.yaml")
    assert t["teacher"]["model"] == "CohereLabs/aya-expanse-32b:cohere"
    assert t["teacher"]["max_usd"] == 5.0
    assert t["teacher"]["local_fallback"]["model"] == "CohereLabs/aya-expanse-8b"
    assert t["router"]["base_url"] == "https://router.huggingface.co/v1"


def test_overrides_are_yaml_parsed():
    cfg = C.load_config("configs/13_grpo.yaml", ["training.max_steps=7", "tasks.languages=[hi, mr]"])
    assert cfg["training"]["max_steps"] == 7
    assert cfg["tasks"]["languages"] == ["hi", "mr"]


def test_grpo_runs_differ_only_in_reward():
    from latentspace_train.stages.grpo import shared_fingerprint

    cfg = C.load_config("configs/13_grpo.yaml")
    a, b = C.select_run(cfg, "run_A"), C.select_run(cfg, "run_B")
    strip = lambda c: {k: v for k, v in c.items() if k not in ("reward", "run_name")}  # noqa: E731
    assert strip(a) == strip(b)
    assert a["reward"]["kind"] == "agenttune_default" and b["reward"]["kind"] == "audit"
    assert a["reward"]["funcs"] == ["hybrid_prm_reward"]
    assert shared_fingerprint(a) == shared_fingerprint(b)
    assert a["training"]["num_generations"] == 4
    with pytest.raises(ValueError, match="--run"):
        C.select_run(cfg, None)
    with pytest.raises(ValueError, match="unknown run"):
        C.select_run(cfg, "run_C")


def test_extends_cycle_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\nx: 1\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\ny: 2\n")
    with pytest.raises(ValueError, match="cycle"):
        C.load_config(tmp_path / "a.yaml")


# ── tasks ─────────────────────────────────────────────────────────────────────


def test_stub_tasks_cover_four_languages_and_splits():
    for split in ("train", "eval", "test"):
        rows = load_split(BASE_CFG, split)
        assert {r["language"] for r in rows} == {"en", "hi", "mr", "hinglish"}
    tools = load_tools(BASE_CFG)
    assert [t.__name__ for t in tools] == ["search_trains", "get_fare", "book_ticket"]


def test_mixed_prompts_become_message_lists():
    ds = to_dataset_rows(load_split(BASE_CFG, "train"))
    assert all(isinstance(r["prompt"], list) for r in ds)
    assert ds[0]["prompt"] == [{"role": "user", "content": stub.ROWS[0][3]}]


def test_transit_source_missing_is_a_clear_error():
    cfg = {"tasks": {"source": "transit", "tools": "no_such_transit_module_xyz:TOOLS"}}
    with pytest.raises(TasksUnavailable, match="tasks.source=stub"):
        load_tools(cfg)


@pytest.fixture
def fake_transit(tmp_path, monkeypatch):
    """A tiny local stand-in for src/transit_agent + data/eval with renamed fields."""
    pkg = tmp_path / "fake_transit_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(textwrap.dedent('''
        SYSTEM_PROMPT = "Transit helper."
        def get_status(train_id: str) -> dict:
            """Live status.

            Args:
                train_id: Train number.
            """
            return {"late_min": 5}
        def get_tools():
            return [get_status]
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    data = tmp_path / "eval"
    data.mkdir()
    rows = [
        {"task_id": "t1", "lang": "hi", "query": "12951 कहाँ है?", "split": "test",
         "gold_calls": [[{"name": "get_status", "arguments": {"train_id": "12951"}}]]},
        {"task_id": "t2", "lang": "en", "query": "Where is 12951?", "split": "train",
         "gold_calls": [[{"name": "get_status", "arguments": {"train_id": "12951"}}]]},
    ]
    (data / "all.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    yield {
        "tasks": {
            "source": "transit", "tools": "fake_transit_pkg:get_tools",
            "system_prompt": "fake_transit_pkg:SYSTEM_PROMPT",
            "data": {"dir": str(data), "file": "all.jsonl", "split_field": "split",
                     "fields": {"id": "task_id", "language": "lang", "prompt": "query",
                                "expected_tool_calls": "gold_calls"}},
        }
    }
    sys.modules.pop("fake_transit_pkg", None)


def test_transit_loader_with_module_attr_and_field_map(fake_transit):
    from latentspace_train.tasks import system_prompt

    tools = load_tools(fake_transit)
    assert [t.__name__ for t in tools] == ["get_status"]
    assert system_prompt(fake_transit) == "Transit helper."
    test = load_split(fake_transit, "test")
    assert [r["id"] for r in test] == ["t1"]
    assert test[0]["language"] == "hi"
    assert json.loads(test[0]["expected_tool_calls"])[0][0]["name"] == "get_status"


def test_transit_loader_per_split_files(tmp_path):
    (tmp_path / "eval.jsonl").write_text(json.dumps({"id": 1, "prompt": "hi", "language": "en"}))
    cfg = {"tasks": {"data": {"dir": str(tmp_path), "files": {"eval": "eval.jsonl"}}}}
    assert load_split(cfg, "eval")[0]["id"] == "1"
    with pytest.raises(TasksUnavailable, match="eval-set"):
        load_split(cfg, "test")


# ── router engine, budget, rollouts ───────────────────────────────────────────


def test_budget_accounting():
    b = Budget(max_usd=0.01, usd_per_m_input=1.0, usd_per_m_output=2.0)
    b.add(4000, 1000)
    assert b.usd == pytest.approx(0.006)
    b.check()
    b.add(4000, 0)
    assert b.exhausted
    with pytest.raises(BudgetExceeded):
        b.check()
    s = b.summary()
    assert s["prompt_tokens"] == 8000 and s["api_calls"] == 2


def test_prompt_mode_folds_tool_turns_and_drops_native_calls():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "fare?"},
        {"role": "assistant", "content": '[{"tool_name": "get_fare"}]', "tool_calls": [{"id": "1"}]},
        {"role": "tool", "name": "get_fare", "content": "2150"},
    ]
    out = _prompt_mode_messages(msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user"]
    assert all(set(m) == {"role", "content"} for m in out)
    assert out[-1]["content"].startswith("Tool result for get_fare: 2150")


class _FakeLiteLLM:
    """Scripted litellm.completion: first a Cohere-format call, then a final answer."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, **kw):
        self.calls.append(kw)
        text = self.replies.pop(0) if self.replies else "done"
        msg = SimpleNamespace(content=text, tool_calls=None)
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=100)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


@pytest.fixture
def fake_router(monkeypatch, tmp_path):
    import litellm

    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("AGENTTUNE_SCORE_LOG", str(tmp_path / "scores.jsonl"))
    fake = _FakeLiteLLM([
        '[{"tool_name": "get_fare", "parameters": {"train_id": "12951", "travel_class": "3A"}}]',
        "The 3A fare is Rs 2150.",
    ])
    monkeypatch.setattr(litellm, "completion", fake)
    return fake


def test_router_engine_prompt_mode_end_to_end(fake_router):
    tools = stub.TOOLS
    budget = Budget(max_usd=1.0, usd_per_m_input=0.5, usd_per_m_output=1.5)
    spec = {"model": "CohereLabs/aya-expanse-32b:cohere", "bill_to": "my-org",
            "tool_mode": "prompt"}
    engine = make_router_engine(spec, budget)
    sp = fallback_system_prompt(tools, stub.SYSTEM_PROMPT)
    fn = build_rollout_fn(engine, tools, system_prompt=sp, max_steps=3)
    rows = [r for r in stub.rows("eval") if r["id"] == "ev-hg-1"]
    recs, status = run_rollouts(fn, rows, gen_kwargs={"max_new_tokens": 64})
    assert status["errors"] == 0 and len(recs) == 1
    first = fake_router.calls[0]
    assert first["model"] == "openai/CohereLabs/aya-expanse-32b:cohere"
    assert first["base_url"] == "https://router.huggingface.co/v1"
    assert first["api_key"] == "hf_test"
    assert first["extra_headers"] == {"X-HF-Bill-To": "my-org"}
    assert "tools" not in first  # prompt mode: schemas are in the system prompt
    assert first["messages"][0]["content"] == sp
    second = fake_router.calls[1]["messages"]
    assert second[-1]["role"] == "user" and second[-1]["content"].startswith("Tool result for get_fare")
    assert recs[0]["tool_call_count"] == 1
    assert budget.calls == 2 and budget.prompt_tokens == 2000

    rewards = make_audit_rewards(tools, BASE_CFG)
    s = score_records(recs, rewards)[0]
    assert s["tool_call_validity"] == 1.0 and s["tool_call_f1"] == 1.0
    assert s["verdict"] == "success"


def test_rollouts_stop_on_budget(fake_router):
    budget = Budget(max_usd=0.0005, usd_per_m_input=0.5, usd_per_m_output=1.5)
    engine = make_router_engine({"model": "m"}, budget)
    fn = build_rollout_fn(engine, stub.TOOLS, system_prompt=None, max_steps=3)
    recs, status = run_rollouts(fn, stub.rows("train"), trials=2)
    assert status["stopped_early"] and "budget reached" in status["stop_reason"]
    assert len(recs) < 16


def test_missing_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        make_router_engine({"model": "m"})


# ── scoring ───────────────────────────────────────────────────────────────────


def _rec(task, lang, trial, conv, ref, prompt=None):
    return {"task_id": task, "language": lang, "trial": trial,
            "prompt": prompt or conv[0]["content"], "response": conv[-1].get("content", ""),
            "conversation": conv, "expected_tool_calls": ref, "error": None}


def _call(name, args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


FARE = ("get_fare", {"train_id": "12951", "travel_class": "3A"})
FARE_REF = json.dumps([[{"name": FARE[0], "arguments": FARE[1]}]])


def test_score_records_good_malformed_unconfirmed():
    rewards = make_audit_rewards(stub.TOOLS, BASE_CFG)
    good = [{"role": "user", "content": "fare?"}, _call(*FARE),
            {"role": "tool", "content": "2150"}, {"role": "assistant", "content": "Rs 2150"}]
    malformed = [{"role": "user", "content": "fare?"},
                 {"role": "assistant", "content": '<tool_call>{"name": "get_fare", "arguments": {</tool_call>'}]
    book = [{"role": "user", "content": "book 12951 for Asha"},
            _call("book_ticket", {"train_id": "12951", "passenger_name": "Asha"}),
            {"role": "tool", "content": "ok"}, {"role": "assistant", "content": "Booked"}]
    book_ref = json.dumps([[{"name": "book_ticket",
                             "arguments": {"train_id": "12951", "passenger_name": "Asha"}}]])
    rows = score_records([_rec("g", "en", 0, good, FARE_REF), _rec("m", "en", 0, malformed, FARE_REF),
                          _rec("b", "hi", 0, book, book_ref)], rewards)
    g, m, b = rows
    assert g["verdict"] == "success" and g["audit_combined"] > m["audit_combined"]
    assert m["n_malformed"] == 1 and m["audit_tool_validity_reward"] == 0.0 and m["verdict"] == "failure"
    assert b["audit_confirm_irreversible_penalty"] == -1.0 and b["verdict"] == "failure"
    assert b["unconfirmed_irreversible"] == ["book_ticket"]
    s = summarise(rows)
    assert set(s["per_language"]) == {"en", "hi"}
    assert s["overall"]["n_rollouts"] == 3


def test_pass_k_and_bootstrap():
    rows_a = [{"task_id": t, "language": "en", "trial": i, "verdict": v, "success": float(v == "success")}
              for t, vs in {"t1": ["success"] * 5, "t2": ["success", "failure"] * 2 + ["success"],
                            "t3": ["failure"] * 5}.items() for i, v in enumerate(vs)]
    pk = pass_k(rows_a, k=5)
    assert pk["n_tasks"] == 3
    assert pk["pass_hat_k"] == pytest.approx(1 / 3, abs=1e-3)
    assert pk["pass_at_k"] == pytest.approx(2 / 3, abs=1e-3)
    rows_b = [dict(r, success=1.0, verdict="success") for r in rows_a]
    cmp = bootstrap_compare(rows_a, rows_b, ["success"], n_resamples=200)
    assert cmp["success"]["n"] == 3 and cmp["success"]["delta"] > 0


def test_paired_test_detects_effects_and_nulls():
    """Regression for FINDINGS.md A3: AuditKIT's uncentred paired_bootstrap gives p ~ 0.5
    for a large effect; the paired test must give a small p there and a large p for noise."""
    import random

    from latentspace_train.scoring import paired_test

    rng = random.Random(1)
    shifted = [(b, b + 0.3 + rng.gauss(0, 0.1)) for b in (rng.random() for _ in range(60))]
    r = paired_test(shifted, n_resamples=2000)
    assert r["significant"] and r["p_value"] < 0.01 and r["p_bootstrap"] < 0.01
    assert r["ci95"][0] > 0.2 and r["ci95"][1] < 0.4
    noise = [(b, b + rng.gauss(0, 0.1)) for b in (rng.random() for _ in range(60))]
    assert paired_test(noise, n_resamples=2000)["p_value"] > 0.05
    assert paired_test([(0.0, 1.0)])["n"] == 1  # too few pairs -> error dict, no crash


# ── teacher filter ────────────────────────────────────────────────────────────


def test_teacher_reject_reasons():
    from latentspace_train.stages.teacher_collect import reject_reason

    ok = {"response": "fine", "expected_tool_calls": FARE_REF}
    s = {"n_calls": 1, "audit_tool_validity_reward": 1.0, "audit_confirm_irreversible_penalty": 0.0,
         "tool_call_f1": 1.0}
    assert reject_reason(ok, s, {}) is None
    assert reject_reason(ok, {**s, "audit_tool_validity_reward": 0.5}, {}) == "invalid_tool_call"
    assert reject_reason(ok, {**s, "audit_confirm_irreversible_penalty": -1.0}, {}) == "unconfirmed_irreversible"
    assert reject_reason({**ok, "error": "x"}, s, {}) == "rollout_error"
    assert reject_reason(ok, {**s, "tool_call_f1": 0.5}, {"min_f1": 1.0}) == "f1_below_min"
    no_call = {"response": "You're welcome", "expected_tool_calls": json.dumps([[]])}
    s0 = {"n_calls": 0, "audit_tool_validity_reward": 0.5, "audit_confirm_irreversible_penalty": 0.0,
          "tool_call_f1": 1.0}
    assert reject_reason(no_call, s0, {}) is None
    assert reject_reason(no_call, s0, {"keep_correct_no_call": False}) == "no_tool_call"
    assert reject_reason(ok, {**s0, "tool_call_f1": 0.0}, {}) == "no_tool_call"


# ── distill rows ──────────────────────────────────────────────────────────────


def test_conversation_to_sft_rows_match_rollout_format():
    from agenttune.agentic.rollout_engines.rollout_factory import _extract_tool_calls

    from latentspace_train.stages.distill import conversation_to_eventlog

    conv = [{"role": "system", "content": "S"}, {"role": "user", "content": "12951 ka 3A fare?"},
            _call(*FARE), {"role": "tool", "name": "get_fare", "content": "2150"},
            {"role": "assistant", "content": "Rs 2150 hai."}]
    rows = conversation_to_eventlog(conv, "cohere").as_dataset_rows("sft")
    msgs = rows[0]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert json.loads(msgs[1]["content"]) == [{"tool_name": "get_fare", "parameters": FARE[1]}]
    parsed = _extract_tool_calls({"role": "assistant", "content": msgs[1]["content"]}, msgs[1]["content"])
    assert parsed[0]["function"]["name"] == "get_fare"
    assert msgs[2]["content"].startswith("Tool result for get_fare: 2150")
    agt = conversation_to_eventlog(conv, "agenttune").as_dataset_rows("sft")[0]["messages"]
    assert agt[1]["content"].startswith("<tool_call>")


# ── run_A reward input adapter ────────────────────────────────────────────────


def test_hybrid_prm_needs_text_inputs(tmp_path, monkeypatch):
    """Upstream hybrid_prm_reward raises on message-list completions (what TRL passes for a
    conversational dataset). The adapter flattens inputs; the reward itself is unmodified."""
    monkeypatch.setenv("AGENTTUNE_SCORE_LOG", str(tmp_path / "s.jsonl"))
    from agenttune.agentic.rewards.builtin_rewards import REWARD_REGISTRY

    from latentspace_train.stages.grpo import build_rewards

    raw = REWARD_REGISTRY["hybrid_prm_reward"]
    comps = [[{"role": "assistant", "content": "hello"}]]
    prompts = [[{"role": "user", "content": "hi"}]]
    with pytest.raises(TypeError):
        raw(prompts=prompts, completions=comps)
    funcs, weights, desc = build_rewards({"reward": {"kind": "agenttune_default"}}, stub.TOOLS)
    assert funcs[0].__name__ == "hybrid_prm_reward" and desc["patched"] is False
    assert funcs[0](prompts=prompts, completions=comps) == [1.0]  # no call -> 1.0 (B1)
    funcs_b, weights_b, _ = build_rewards({"reward": {"kind": "audit"}, **BASE_CFG}, stub.TOOLS)
    assert [f.__name__ for f in funcs_b][0] == "audit_tool_validity_reward"
    assert weights_b == [1.0, 1.0, 1.0, 1.0, 0.25]


# ── results writer + CLI ──────────────────────────────────────────────────────


def test_write_results_records_provenance(tmp_path):
    from latentspace_train.runtime import write_results

    cfg = {"results_dir": str(tmp_path), "stage": "baseline", "seed": 3, "_config_path": "x.yaml"}
    p = write_results(cfg, "10_baseline", {"n": 1}, time.time())
    d = json.loads(p.read_text())
    assert d["seed"] == 3 and d["n"] == 1 and d["config"]["_config_path"] == "x.yaml"
    assert d["versions"]["agenttune"] and d["versions"]["auditkit"] == "1.1.3"
    assert len(d["provenance"]["commit"]) == 40


def test_cli_print_config(capsys):
    from latentspace_train.cli import main

    assert main(["grpo", "--run", "run_B", "--print-config", "--set", "student.model=fire"]) == 0
    cfg = json.loads(capsys.readouterr().out)
    assert cfg["run_name"] == "run_B" and cfg["student"]["model"] == "CohereLabs/tiny-aya-fire"
    with pytest.raises(ValueError, match="--run"):
        main(["grpo", "--print-config"])


def test_precision_auto():
    from latentspace_train.runtime import resolve_precision

    t4 = {"cuda": True, "bf16_supported": True, "gpus": [{"capability": "7.5"}] * 2}
    l4 = {"cuda": True, "bf16_supported": True, "gpus": [{"capability": "8.9"}]}
    assert resolve_precision("auto", t4) == "fp16"
    assert resolve_precision("auto", l4) == "bf16"
    assert resolve_precision("auto", {"cuda": False, "gpus": []}) == "fp32"
    assert resolve_precision("fp16", l4) == "fp16"
