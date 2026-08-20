"""Command-line entry point for the small public integration shell."""

from __future__ import annotations

import argparse
import json
import sys

from .runtime import ImmerRuntime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="immer", description="IMMER fail-closed integration runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)
    solve = subparsers.add_parser("solve", help="solve an exact-math question through FERTIG")
    solve.add_argument("question")
    solve.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "solve":
        result = ImmerRuntime().solve(args.question)
        if args.as_json:
            print(json.dumps(result.to_dict(), sort_keys=True))
        else:
            print(result.answer if result.answer is not None else f"ABSTAIN: {result.reason}")
        return 0 if result.status.value == "verified" else 2
    return 2


if __name__ == "__main__":
    sys.exit(main())
