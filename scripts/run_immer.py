#!/usr/bin/env python3
"""Start the dependency-free IMMER smoke path."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from immer.core.runtime import ImmerRuntime


def main() -> int:
    question = "A bakery sold 12 cakes on Monday and 15 cakes on Tuesday. How many cakes did they sell in total?"
    result = ImmerRuntime().solve(question)
    print(result.to_dict())
    return 0 if result.answer == "27" else 1


if __name__ == "__main__":
    raise SystemExit(main())
