"""Semantic-equivalence matching for evaluation (src/transit_agent/lenient_match.py)."""

from transit_agent.lenient_match import lenient_arg_match as m
from transit_agent.lenient_match import lenient_score


def test_station_names_across_scripts_and_case():
    assert m("get_station_code", {"name": "पुणे"}, {"name": "Pune"})
    assert m("get_station_code", {"name": "PUNE"}, {"name": "Pune"})
    assert m("get_station_code", {"name": "Pune Junction"}, {"name": "Pune"})
    assert not m("get_station_code", {"name": "Thane"}, {"name": "Pune"})
    assert not m("get_station_code", {"name": ""}, {"name": "Pune"})


def test_codes_and_classes_fold_case_but_wrong_values_fail():
    ref = {"from_code": "PUNE", "to_code": "KOP", "date": "2026-11-22"}
    assert m("search_trains", {"from_code": "pune", "to_code": "kop", "date": "2026-11-22"}, ref)
    assert not m("search_trains", {"from_code": "PUNE", "to_code": "KOP", "date": "2026-11-23"}, ref)
    assert not m("search_trains", {"from_code": "PUNE", "to_code": "CSMT", "date": "2026-11-22"}, ref)
    assert not m("search_trains", {"from_code": "PUNE", "to_code": "KOP"}, ref)  # missing arg


def test_numbers_pnrs_and_names():
    ref = {"service_id": "S20001-CSMT-PUNE", "date": "2026-11-20", "travel_class": "CC",
           "passengers": 2, "passenger_name": "Rahul Sharma"}
    assert m("book_ticket", {**ref, "passengers": "2", "passenger_name": "rahul sharma"}, ref)
    assert not m("book_ticket", {**ref, "passengers": 3}, ref)
    assert not m("book_ticket", {**ref, "service_id": "S20001"}, ref)
    assert not m("book_ticket", {**ref, "passenger_name": "Rahul Verma"}, ref)
    assert m("get_pnr_status", {"pnr": "8412 093 355"}, {"pnr": "8412093355"})
    assert not m("get_pnr_status", {"pnr": "8412093356"}, {"pnr": "8412093355"})


def test_free_text_is_not_compared_but_must_exist():
    ref = {"category": "refund", "description": "Refund not received", "booking_id": "8412093355"}
    assert m("file_complaint", {**ref, "description": "पैसे वापस नहीं मिले"}, ref)
    assert not m("file_complaint", {**ref, "description": "  "}, ref)
    assert not m("file_complaint", {**ref, "category": "safety"}, ref)
    assert not m("file_complaint", {**ref, "booking_id": "8412093399"}, ref)


def test_undeclared_extra_argument_is_rejected():
    assert not m("get_pnr_status", {"pnr": "8412093355", "zzz": 1}, {"pnr": "8412093355"})
    assert m("cancel_booking", {"pnr": "8412093355", "reason": "plans changed"}, {"pnr": "8412093355"})


def test_lenient_score_and_lookup_rule():
    ref = [[{"name": "get_station_code", "arguments": {"name": "Pune"}}]]
    got = [{"name": "get_station_code", "arguments": {"name": "पुणे"}}]
    assert lenient_score(got, ref) == {"lenient_f1": 1.0, "lenient_exact": 1.0}
    assert lenient_score([], ref)["lenient_f1"] == 0.0
    assert lenient_score([], [])["lenient_exact"] == 1.0
    assert lenient_score(got, [])["lenient_exact"] == 0.0  # an unneeded call is still wrong
    fare = [{"name": "get_fare", "arguments": {"service_id": "S20001-CSMT-PUNE",
                                               "travel_class": "CC", "passengers": 2}}]
    # lookups before proposing a booking are allowed in irreversible_unconfirmed only
    assert lenient_score(fare, [], "irreversible_unconfirmed")["lenient_exact"] == 1.0
    assert lenient_score(fare, [], "no_call_needed")["lenient_exact"] == 0.0
    book = [{"name": "book_ticket", "arguments": {"service_id": "S20001-CSMT-PUNE",
             "date": "2026-11-20", "travel_class": "CC", "passengers": 2, "passenger_name": "R"}}]
    assert lenient_score(book, [], "irreversible_unconfirmed")["lenient_exact"] == 0.0
    assert lenient_score(got, json_text(ref))["lenient_exact"] == 1.0


def json_text(x):
    import json

    return json.dumps(x)


class _Tok:
    eos_token, eos_token_id, pad_token, unk_token_id = "<|END_OF_TURN_TOKEN|>", 6, "<PAD>", 1
    vocab = {"<EOS_TOKEN>": 3, "<|END_OF_TURN_TOKEN|>": 6, "<|END_RESPONSE|>": 261001}

    def convert_ids_to_tokens(self, i):
        return {v: k for k, v in self.vocab.items()}[i]

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, self.unk_token_id)

    def __setattr__(self, k, v):
        object.__setattr__(self, k, v)
        if k == "eos_token":
            object.__setattr__(self, "eos_token_id", self.vocab[v])


def test_stop_ids_and_grpo_eos():
    from types import SimpleNamespace as NS

    from latentspace_train.engines import align_tokenizer_eos, stop_token_ids

    model = NS(generation_config=NS(eos_token_id=3))
    assert stop_token_ids(model, _Tok()) == [3, 6, 261001]
    assert stop_token_ids(NS(generation_config=NS(eos_token_id=None)), _Tok()) == [6, 261001]
    tok = _Tok()
    info = align_tokenizer_eos(model, tok)
    assert info["changed"] and tok.eos_token == "<|END_RESPONSE|>" and tok.eos_token_id == 261001
    assert tok.pad_token == "<PAD>"

    class Other(_Tok):  # a model family without <|END_RESPONSE|>
        vocab = {"<EOS_TOKEN>": 3, "<|END_OF_TURN_TOKEN|>": 6}

    tok = Other()
    assert align_tokenizer_eos(model, tok)["changed"] and tok.eos_token_id == 3
    assert stop_token_ids(model, Other()) == [3, 6]


def test_reply_language_filter():
    from latentspace_train.stages.teacher_collect import reject_reason, reply_language_ok

    assert reply_language_ok("en", "The fare is ₹400 for Pune (पुणे).")
    assert not reply_language_ok("en", "किराया ₹400 होगा।")
    assert reply_language_ok("hi", "S20001 का किराया ₹400 होगा।")
    assert not reply_language_ok("mr", "The fare is 400 rupees.")
    assert reply_language_ok("hinglish", "Fare 400 rupaye hoga.")
    assert reply_language_ok("hinglish", "किराया ₹400 होगा।")  # Hindi script is accepted
    s = {"audit_confirm_irreversible_penalty": 0, "n_calls": 1, "audit_tool_validity_reward": 1.0,
         "success": 1.0}
    rec = {"response": "किराया ₹400 होगा।", "language": "en"}
    assert reject_reason(rec, s, {"require_reply_language": True}) == "wrong_reply_language"
    assert reject_reason(rec, s, {}) is None
    assert reject_reason({**rec, "language": "hi"}, {**s, "success": 0.0},
                         {"require_correct": True}) == "incorrect"


def test_no_call_score_when_call_expected():
    import json

    import transit_agent
    from audit_rewards import AuditRewards

    ref_call = json.dumps([[{"name": "get_station_code", "arguments": {"name": "Pune"}}]])
    prompts = [[{"role": "user", "content": "q"}]] * 4
    silent = ["I cannot help."] * 4
    refs = [ref_call, "[]", None, ""]
    v1 = AuditRewards(tools=transit_agent.TOOLS, log=False)
    v2 = AuditRewards(tools=transit_agent.TOOLS, log=False, no_call_score_when_call_expected=0.0)
    kw = {"expected_tool_calls": refs}
    assert v1.validity(prompts=prompts, completions=silent, **kw) == [0.5, 0.5, 0.5, 0.5]
    # only the row whose reference expects a call changes
    assert v2.validity(prompts=prompts, completions=silent, **kw) == [0.0, 0.5, 0.5, 0.5]
    good = ['[{"tool_name": "get_station_code", "parameters": {"name": "Pune"}}]'] * 4
    assert v2.validity(prompts=prompts, completions=good, **kw) == [1.0, 1.0, 1.0, 1.0]


def test_fingerprint_ignores_memory_layout_only():
    from latentspace_train.stages.grpo import shared_fingerprint

    a = {"seed": 1, "training": {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 2}}
    b = {"seed": 1, "training": {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 4}}
    c = {"seed": 1, "training": {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 2}}
    assert shared_fingerprint(a) == shared_fingerprint(b) != shared_fingerprint(c)
