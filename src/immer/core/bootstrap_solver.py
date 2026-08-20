"""Tiny local fallback for the public smoke path.

This is intentionally narrow. It handles only transparent arithmetic fixtures
used to prove that a fresh checkout starts without model downloads. Anything
outside these forms abstains and is delegated to FERTIG when configured.
"""

from __future__ import annotations

import re
from fractions import Fraction


def solve(question: str) -> str | None:
    text = question.lower()
    numbers = [Fraction(value) for value in re.findall(r"\d+(?:\.\d+)?", text)]
    if not numbers:
        return None
    if "half as many" in text and any(word in text for word in ("total", "altogether", "in all")):
        return str(numbers[0] + numbers[0] / 2)
    percent = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(\d+(?:\.\d+)?)", text)
    if percent:
        return _format(Fraction(percent.group(1)) * Fraction(percent.group(2)) / 100)
    if any(word in text for word in ("total", "altogether", "in all")) and len(numbers) == 2:
        return _format(sum(numbers, Fraction(0)))
    return None


def _format(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else str(float(value))
