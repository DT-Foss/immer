"""IntentRouter: what does David want, and which organ answers?

Rule-based on purpose — no magic, every classification is inspectable.
Kinds: TEACH, RECALL, MATH, STATUS, CHAT.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    kind: str
    payload: str
    detail: str = ""


_TEACH = re.compile(r"^(merke[:r]|merk dir|notiere|remember)[:\s]+(.+)$", re.IGNORECASE)
_RECALL = re.compile(r"^(was weißt du über|erinnere dich an|erinnere|recall|was weisst du über)\s*[:\s]?(.+)$", re.IGNORECASE)
_STATUS = re.compile(r"^(wie geht es dir|wie geht's|wie gehts|status|bist du da)[?!\s]*$", re.IGNORECASE)
_MATH_HINTS = (
    "how many", "how much", "wie viel", "wie viele", "berechne",
    "calculate", "summe", "differenz", "produkt", "quotient", "z3sum",
    "+", "-", "*", "/",
)
_NUMBER_WORD = (
    r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|null|eins?|zwei|drei|vier|"
    r"fünf|fuenf|sechs|sieben|acht|neun|zehn|elf|zwölf|zwoelf|dreizehn|"
    r"vierzehn|fünfzehn|fuenfzehn|sechzehn)"
)
_WORD_ARITHMETIC = re.compile(
    rf"\b{_NUMBER_WORD}\b\s*(?:plus|minus|less|times|mal|z3sum)\s*\b{_NUMBER_WORD}\b",
    re.IGNORECASE,
)


def classify(text: str) -> Intent:
    stripped = text.strip()
    teach = _TEACH.match(stripped)
    if teach:
        return Intent("TEACH", teach.group(2).strip(), "Lehr-Absage erkannt")
    recall = _RECALL.match(stripped)
    if recall:
        return Intent("RECALL", recall.group(2).strip(), "Abruf erkannt")
    if _STATUS.match(stripped):
        return Intent("STATUS", stripped, "Zustandsfrage")
    has_digit = bool(re.search(r"\d", stripped))
    lowered = stripped.lower()
    if _WORD_ARITHMETIC.search(stripped) or (has_digit and any(hint in lowered for hint in _MATH_HINTS)):
        return Intent("MATH", stripped, "Zahlen + Rechen-Signal")
    return Intent("CHAT", stripped, "Gespräch")
