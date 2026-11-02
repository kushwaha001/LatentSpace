"""AuditKIT agent metrics as AgentTune-compatible GRPO reward functions.

    from audit_rewards import AuditRewards, irreversible

    rewards = AuditRewards(tools=[get_weather, irreversible(book_flight)])
    trainer = create_agentic_trainer(..., tools=tools, reward_funcs=rewards.train_time())

See README.md in this directory for the metric-to-reward mapping and the held-out split.
"""

from .confirmation import is_confirmation, unconfirmed_irreversible_calls
from .episode import CallRecord, Episode, build_episode
from .rewards import DEFAULT_WEIGHTS, AuditRewards
from .tools import irreversible, tool_schema, tool_schemas

__all__ = [
    "DEFAULT_WEIGHTS",
    "AuditRewards",
    "CallRecord",
    "Episode",
    "build_episode",
    "irreversible",
    "is_confirmation",
    "tool_schema",
    "tool_schemas",
    "unconfirmed_irreversible_calls",
]
