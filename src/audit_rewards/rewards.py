"""AuditKIT agent metrics as AgentTune GRPO reward functions.

Every reward has the REWARD_REGISTRY convention ``fn(prompts, completions, **kwargs) -> list``
and reads the extras AgentTune's rollout forwards (``conversations``) plus dataset columns
(``expected_tool_calls``, and optionally per-row ``tools`` / ``allowed_tools`` /
``denied_tools``). The scoring itself is done by AuditKIT's metric classes, unmodified; this
module only builds their input and maps their output to a reward.

Two kinds of reward:

- **scores in [0, 1]**: ``audit_tool_validity_reward`` and ``audit_tool_f1_reward``. These carry
  the call / no-call / malformed ordering.
- **penalties in [-1, 0]**: permission, redundancy, loop and confirm-before-irreversible.
  They are 0 when nothing was violated, *including* when there was nothing to check (no calls).
  AuditKIT marks such episodes ineligible rather than scoring them. A penalty should not also
  act as a "call more tools" bonus, so no-call gets 0, not a neutral midpoint.
"""

from __future__ import annotations

import json

from collections.abc import Callable, Iterable, Sequence
from typing import Any

from auditkit.metrics.agent import (
    AgentLoopDetection,
    RedundantToolCalls,
    ToolCallF1,
    ToolCallValidity,
    ToolPermission,
)
from auditkit.sample import Sample

from .confirmation import FOLDED_TOOL_RESULT_PREFIX, unconfirmed_irreversible_calls
from .episode import Episode, _content, build_episode
from .tools import irreversible_names, tool_name, tool_schemas

RewardFn = Callable[..., list]

# Train-time default weights for combined(). AgentLoopDetection is deliberately absent:
# it is report-only by default (README, "Held-out split").
DEFAULT_WEIGHTS = {
    "audit_tool_validity_reward": 1.0,
    "audit_tool_f1_reward": 1.0,
    "audit_tool_permission_penalty": 1.0,
    "audit_confirm_irreversible_penalty": 1.0,
    "audit_redundant_calls_penalty": 0.25,
}


def _log(name: str, score: Any, reasons: list[str], index: int) -> None:
    try:
        from agenttune.utils.score_logger import log_score
    except ImportError:  # AgentTune is optional for scoring; logging is best-effort
        return
    log_score(f"audit_rewards.{name}", score, reasons=reasons, meta={"index": index})


def _row(value: Any, i: int) -> Any:
    """Per-row value of a TRL kwarg (a column list), or the value itself if it is a scalar."""
    if isinstance(value, (list, tuple)):
        return value[i] if i < len(value) else None
    return value


def _expects_call(ref: Any) -> bool:
    """True when a reference (JSON text, turns of calls, or a flat list) has at least one call."""
    if ref is None:
        return False
    if isinstance(ref, str):
        try:
            ref = json.loads(ref) if ref.strip() else []
        except json.JSONDecodeError:
            return False
    if isinstance(ref, dict):
        return True
    return any((len(t) > 0) if isinstance(t, (list, tuple)) else bool(t) for t in (ref or []))


def _prompt_text(prompt: Any) -> str:
    if isinstance(prompt, list):
        return "\n".join(_content(m) for m in prompt if isinstance(m, dict))
    return "" if prompt is None else str(prompt)


def _pick(scores: list, prefix: str) -> Any:
    for s in scores:
        if s.name.startswith(prefix):
            return s
    return None


class AuditRewards:
    """A configured set of AuditKIT-backed reward functions for one tool set.

    Args:
        tools: the tools the rollout offers: ``BaseTool`` instances, plain functions or schema
            dicts (the same list passed to ``create_agentic_trainer(tools=...)``). Schemas are
            built with AgentTune's own builders. A per-row ``tools`` column overrides them.
        irreversible_tools: extra tool names that need a user confirmation turn. Tools tagged
            with :func:`audit_rewards.irreversible` or ``"x-irreversible": true`` are included
            automatically.
        allowed_tools / denied_tools: the ToolPermission policy. The allowlist defaults to the
            offered tools (AuditKIT's default). A per-row column of the same name overrides it.
        no_call_score: validity reward for an episode with no tool call and no attempt.
            Default 0.5 (neutral). Use 0.0 when every task needs a call.
        no_call_score_when_call_expected: validity reward for a no-call episode on a row whose
            ``expected_tool_calls`` has at least one call. ``None`` (default) keeps
            ``no_call_score`` for every row. With 0.5, staying silent on a task that needs a
            call out-scores an invalid attempt and so teaches the policy to give up when its
            calls are often invalid (round 2, run_B v1: 0.12 calls per task, dev success on
            single-call tasks 0.63 -> 0.13). 0.0 makes any valid attempt beat silence there,
            and leaves rows without a reference, or with an empty one, unchanged.
        f1_arg_mode / f1_arg_match: passed to ``ToolCallF1``. Use
            ``f1_arg_match=auditkit.bfcl_arg_match`` for BFCL-style references (lists of
            accepted values).
        calls_per_confirmation: irreversible calls one confirmation turn authorizes.
        confirmation_penalty: magnitude of the confirm-before-irreversible penalty.
        log: write every score and its reasons through AgentTune's ``log_score``.
    """

    def __init__(
        self,
        tools: Iterable[Any] | None = None,
        *,
        irreversible_tools: Iterable[str] | None = None,
        allowed_tools: Iterable[str] | None = None,
        denied_tools: Iterable[str] | None = None,
        no_call_score: float = 0.5,
        no_call_score_when_call_expected: float | None = None,
        f1_arg_mode: str = "exact",
        f1_arg_match: Callable | None = None,
        loop_repetition_threshold: int = 3,
        calls_per_confirmation: int = 1,
        confirmation_penalty: float = 1.0,
        folded_tool_result_prefix: str = FOLDED_TOOL_RESULT_PREFIX,
        log: bool = True,
    ) -> None:
        tools = list(tools or [])
        self.schemas = tool_schemas(tools)
        self.irreversible = irreversible_names(tools, irreversible_tools)
        self.allowed_tools = list(allowed_tools) if allowed_tools is not None else None
        self.denied_tools = list(denied_tools) if denied_tools is not None else None
        self.no_call_score = no_call_score
        self.no_call_score_when_call_expected = no_call_score_when_call_expected
        self.calls_per_confirmation = calls_per_confirmation
        self.confirmation_penalty = confirmation_penalty
        self.folded_prefix = folded_tool_result_prefix
        self.log = log

        self._validity = ToolCallValidity()
        self._f1 = ToolCallF1(arg_mode=f1_arg_mode, arg_match=f1_arg_match)
        self._redundant = RedundantToolCalls()
        self._loop = AgentLoopDetection(repetition_threshold=loop_repetition_threshold)

        self.validity = self._reward("audit_tool_validity_reward", self._score_validity)
        self.f1 = self._reward("audit_tool_f1_reward", self._score_f1)
        self.permission = self._reward("audit_tool_permission_penalty", self._score_permission)
        self.redundancy = self._reward("audit_redundant_calls_penalty", self._score_redundancy)
        self.loop = self._reward("audit_loop_penalty", self._score_loop)
        self.confirmation = self._reward(
            "audit_confirm_irreversible_penalty", self._score_confirmation
        )

    # ── public API ───────────────────────────────────────────────────────────

    @property
    def all(self) -> dict[str, RewardFn]:
        fns = [self.validity, self.f1, self.permission, self.redundancy, self.loop]
        return {f.__name__: f for f in fns + [self.confirmation]}

    def train_time(self, include_f1: bool = True) -> list[RewardFn]:
        """The rewards the README recommends training on (loop detection is report-only)."""
        names = [n for n in DEFAULT_WEIGHTS if include_f1 or n != "audit_tool_f1_reward"]
        return [self.all[n] for n in names]

    def combined(
        self,
        weights: dict[str, float] | None = None,
        include_f1: bool = True,
        extra: Sequence[tuple[Any, float]] = (),
    ) -> RewardFn:
        """One reward through AgentTune's real ``combine_rewards``. ``extra`` adds
        ``(reward_or_registry_name, weight)`` pairs, e.g. a task-answer reward."""
        from agenttune.agentic.rewards.composite import combine_rewards

        weights = {**DEFAULT_WEIGHTS, **(weights or {})}
        fns = self.train_time(include_f1=include_f1)
        funcs: list[Any] = fns + [f for f, _ in extra]
        ws = [weights[f.__name__] for f in fns] + [w for _, w in extra]
        return combine_rewards(funcs, weights=ws)

    def register(self, registry: dict | None = None) -> dict[str, RewardFn]:
        """Opt-in: add these rewards to AgentTune's REWARD_REGISTRY (or ``registry``) so they
        can be named as strings in ``reward_funcs=[...]``."""
        if registry is None:
            from agenttune.agentic.rewards.builtin_rewards import REWARD_REGISTRY as registry
        registry.update(self.all)
        return self.all

    def episodes(self, prompts, completions, **kwargs) -> list[Episode]:
        convs = kwargs.get("conversations")
        return [
            build_episode(_row(prompts, i), c, _row(convs, i)) for i, c in enumerate(completions)
        ]

    def report(self, prompts=None, completions=None, **kwargs) -> list[dict[str, Any]]:
        """Raw AuditKIT scores per row (``None`` = ineligible), for report-only use."""
        rows = []
        refs = kwargs.get("expected_tool_calls")
        for i, ep in enumerate(self.episodes(prompts, completions, **kwargs)):
            sample = self._sample(i, prompts, kwargs, ep)
            ctx_all, ctx_exec = {"trace": ep.trace()}, {"trace": ep.trace(executable_only=True)}
            row: dict[str, Any] = {
                "n_calls": len(ep.calls),
                "n_executed": len(ep.executed),
                "n_malformed": len(ep.malformed),
                "calls": [c.call.to_dict() for c in ep.calls],
            }
            v = self._validity.score(sample, "", ctx_all) if sample.tools else []
            row["tool_call_validity"] = v[0].value if v else None
            if _row(refs, i) is not None:
                f = self._f1.score(sample, "", ctx_all)
                for key in ("tool_call_f1", "tool_call_exact"):
                    s = _pick(f, key)
                    row[key] = s.value if s else None
            r = self._redundant.score(sample, "", ctx_exec)
            row["redundant_tool_calls"] = r[0].value if r else None
            p = self._permission_metric(i, kwargs).score(sample, "", ctx_exec)
            row["tool_permission"] = p[0].value if p else None
            lp = self._loop.score(sample, "", ctx_exec)
            row["agent_loop_detection"] = lp[0].value if lp else None
            n_irrev, viol = self._violations(ep)
            row["irreversible_calls"] = n_irrev
            row["unconfirmed_irreversible"] = [v.tool for v in viol]
            rows.append(row)
        return rows

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _reward(self, name: str, scorer: Callable) -> RewardFn:
        def reward(prompts=None, completions=None, **kwargs) -> list:
            if completions is None:
                raise TypeError(f"{name}() missing required argument: 'completions'")
            out = []
            for i, ep in enumerate(self.episodes(prompts, completions, **kwargs)):
                score, reasons = scorer(i, ep, prompts, kwargs)
                if self.log:
                    _log(name, score, reasons, i)
                out.append(score)
            return out

        reward.__name__ = reward.__qualname__ = name
        reward.__doc__ = scorer.__doc__
        return reward

    def _schemas(self, i: int, kwargs: dict) -> list[dict]:
        row = _row(kwargs.get("tools"), i)
        if row:
            return tool_schemas(row)
        return self.schemas

    def _sample(self, i: int, prompts, kwargs: dict, ep: Episode) -> Sample:
        meta: dict[str, Any] = {}
        allowed = _row(kwargs.get("allowed_tools"), i) or self.allowed_tools
        if allowed is not None:
            meta["allowed_tools"] = [a if isinstance(a, str) else tool_name(a) for a in allowed]
        return Sample(
            input=_prompt_text(_row(prompts, i)),
            tools=self._schemas(i, kwargs) or None,
            expected_tool_calls=_row(kwargs.get("expected_tool_calls"), i),
            metadata=meta,
        )

    def _permission_metric(self, i: int, kwargs: dict) -> ToolPermission:
        denied = _row(kwargs.get("denied_tools"), i)
        return ToolPermission(denied_tools=denied if denied is not None else self.denied_tools)

    def _violations(self, ep: Episode):
        return unconfirmed_irreversible_calls(
            ep, self.irreversible, self.calls_per_confirmation, self.folded_prefix
        )

    # ── scorers: (row index, episode, prompts, kwargs) -> (score, reasons) ──────

    def _score_validity(self, i, ep, prompts, kwargs):
        """ToolCallValidity: fraction of attempted calls that are schema-valid. A call the
        rollout could not parse counts as invalid. No call and no attempt: ``no_call_score``."""
        if not ep.calls:
            alt = self.no_call_score_when_call_expected
            if alt is not None and _expects_call(_row(kwargs.get("expected_tool_calls"), i)):
                return alt, [f"no tool call, but the reference expects one -> {alt}"]
            return self.no_call_score, [f"no tool call or attempt -> {self.no_call_score}"]
        sample = self._sample(i, prompts, kwargs, ep)
        if not sample.tools:
            raise ValueError(
                "audit_tool_validity_reward needs tool schemas: pass tools= to AuditRewards "
                "or a per-row 'tools' column"
            )
        s = self._validity.score(sample, "", {"trace": ep.trace()})
        return s[0].value, [s[0].reason or f"{len(ep.calls)} call(s), all valid"]

    def _score_f1(self, i, ep, prompts, kwargs):
        """ToolCallF1 against the row's ``expected_tool_calls`` (attempted calls count as
        predictions; an unparsed attempt never matches). ``None`` for a row without one."""
        refs = kwargs.get("expected_tool_calls")
        if refs is None:
            raise ValueError(
                "audit_tool_f1_reward needs an 'expected_tool_calls' dataset column "
                "(use combined(include_f1=False) for reference-free training)"
            )
        if _row(refs, i) is None:
            return None, ["no reference for this row -> None"]
        sample = self._sample(i, prompts, kwargs, ep)
        s = _pick(self._f1.score(sample, "", {"trace": ep.trace()}), "tool_call_f1")
        if s is None:
            return None, ["ToolCallF1 ineligible -> None"]
        return s.value, [f"tool_call_f1={s.value:.3f}"]

    def _score_permission(self, i, ep, prompts, kwargs):
        """ToolPermission on executed calls: -(fraction of calls outside policy)."""
        if not ep.executed:
            return 0.0, ["no executed call -> nothing to violate"]
        sample = self._sample(i, prompts, kwargs, ep)
        s = self._permission_metric(i, kwargs).score(sample, "", {"trace": ep.trace(True)})
        if not s:
            return 0.0, ["no permission policy -> 0.0"]
        return s[0].value - 1.0, [s[0].reason or "all calls within policy"]

    def _score_redundancy(self, i, ep, prompts, kwargs):
        """RedundantToolCalls on executed calls: -(fraction that repeat an earlier call)."""
        if not ep.executed:
            return 0.0, ["no executed call -> nothing redundant"]
        sample = self._sample(i, prompts, kwargs, ep)
        s = self._redundant.score(sample, "", {"trace": ep.trace(True)})
        return 0.0 - s[0].value, [f"redundant fraction {s[0].value:.3f}"]

    def _score_loop(self, i, ep, prompts, kwargs):
        """AgentLoopDetection: -1.0 if a loop (repeats, cycle, stagnation) was found."""
        sample = self._sample(i, prompts, kwargs, ep)
        s = self._loop.score(sample, "", {"trace": ep.trace(True)})
        if not s:
            return 0.0, ["nothing to check -> 0.0"]
        return s[0].value - 1.0, [s[0].reason or "loop-free"]

    def _score_confirmation(self, i, ep, prompts, kwargs):
        """-confirmation_penalty if any irreversible call has no preceding user confirmation."""
        n, viol = self._violations(ep)
        if not viol:
            return 0.0, [
                f"{n} irreversible call(s), all confirmed" if n else "no irreversible call"
            ]
        why = [f"{v.tool} at message {v.message_index}: {v.reason}" for v in viol]
        return -self.confirmation_penalty, why
