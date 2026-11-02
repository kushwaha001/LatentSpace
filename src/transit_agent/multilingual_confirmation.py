"""A multilingual (en / hi / mr / Hinglish) yes-no matcher for the confirmation rule.

``audit_rewards.confirmation.is_confirmation`` is an English keyword regex. Experiment 05
shows that it misses every Hindi and Marathi "yes" in the eval set, and most Hinglish ones.
It also reads "Yes, cancel it" as a no, because "cancel" is on its negation list. This module
is a drop-in replacement, proposed for audit_rewards. Experiment 05 swaps it in only as a
labelled counterfactual. Nothing in audit_rewards is changed.

The rules, applied to a single user turn:

- Matching is on tokens and token n-grams, not substrings. Python's ``\\b`` does not work
  inside Devanagari words (vowel signs are not ``\\w``), and a substring "हो" (Marathi
  "yes") would also match "होगा".
- **Hard negations** ("no", "नहीं", "नको", "nahi", "don't", "wait", "रुकिए", "रहने दो", ...)
  always win.
- **Soft negations** ("cancel", "रद्द", "stop", ...) name the action itself when the user is
  confirming a cancellation. They count as a no only when the turn has no affirmation.
  "Yes, cancel it" is a yes; "Actually, cancel that" is a no.
- Otherwise a turn is a confirmation if it has an affirmation.

This is still a lexicon. It is not a judge, and its gaps must be measured on held-out
phrasings (the eval set's confirmed and declined rows).
"""

from __future__ import annotations

import re
import unicodedata

AFFIRM = {
    # English
    "yes", "yep", "yeah", "yup", "sure", "ok", "okay", "confirm", "confirmed", "approved",
    "approve", "proceed", "correct", "go ahead", "do it", "book it", "please do",
    "sounds good", "let's do it", "lets do it", "make it so", "that's right", "thats right",
    # Hindi (Devanagari)
    "हाँ", "हां", "जी हाँ", "जी हां", "ठीक है", "कर दीजिए", "कर दो", "कर दीजिये", "बुक कीजिए",
    "बुक करो", "मंज़ूर", "मंजूर", "चलिए", "बढ़िया", "आगे बढ़िए", "कन्फ़र्म", "कन्फर्म", "पक्का",
    # Marathi (Devanagari)
    "हो", "होय", "ठीक आहे", "चालेल", "करा", "करून टाका", "बुक करा", "चला", "छान", "नक्की",
    "पुढे जा",
    # Hinglish (Roman)
    "haan", "haa", "han", "ha", "haan ji", "ji haan", "theek hai", "thik hai", "kar do",
    "kardo", "book karo", "book kar do", "chalo", "pakka", "done", "badhiya", "aage badho",
}
HARD_NEGATE = {
    # English
    "no", "nope", "not", "don't", "dont", "do not", "wait", "hold on", "never mind",
    "nevermind", "leave it", "drop it", "skip", "skip it", "not now", "not today",
    # Hindi
    "नहीं", "नही", "मत", "रुकिए", "रुको", "रहने दीजिए", "रहने दो", "छोड़ दीजिए", "छोड़ दो",
    "अभी नहीं",
    # Marathi
    "नको", "नाही", "नका", "थांबा", "राहू द्या", "राहूदे",
    # Hinglish
    "nahi", "nahin", "nai", "mat", "ruko", "rehne do", "chhod do", "chod do", "abhi nahi",
}
SOFT_NEGATE = {"cancel", "stop", "abort", "रद्द", "कैंसिल", "cancel kar do"}

_NUKTA = "़"
_PUNCT = re.compile(r"[^\w\s'\u0900-\u0963\u0966-\u097F]")  # Devanagari minus danda


def _tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFC", text or "").casefold()
    t = _PUNCT.sub(" ", t).replace("’", "'")
    return t.split()


def _grams(tokens: list[str], n_max: int = 3) -> set[str]:
    out = set()
    for n in range(1, n_max + 1):
        for i in range(len(tokens) - n + 1):
            out.add(" ".join(tokens[i:i + n]))
    return out


def _norm_set(words: set[str]) -> set[str]:
    return {" ".join(_tokens(w)) for w in words} | {
        " ".join(_tokens(w)).replace(_NUKTA, "") for w in words}


_AFFIRM, _HARD, _SOFT = _norm_set(AFFIRM), _norm_set(HARD_NEGATE), _norm_set(SOFT_NEGATE)


def is_confirmation(text: str) -> bool:
    toks = _tokens(text)
    grams = _grams(toks) | _grams([t.replace(_NUKTA, "") for t in toks])
    if grams & _HARD:
        return False
    affirm = bool(grams & _AFFIRM)
    if grams & _SOFT and not affirm:
        return False
    return affirm
