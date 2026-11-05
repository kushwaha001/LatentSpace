"""Backward-compatible alias. The multilingual (en / hi / mr / Hinglish) yes-no matcher now
lives in ``audit_rewards.confirmation`` and is the reward's default rule; this module re-exports
it so older imports keep working."""

from audit_rewards.confirmation import (  # noqa: F401
    AFFIRM_WORDS as AFFIRM,
    NEGATE_WORDS as NEGATE,
    CANCEL_WORDS,
    is_confirmation,
    is_confirmation_english,
)
