from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .cognition.fertig import FertigSolver
from .contracts import Request
from .runtime import ImmerRuntime


COMPONENTS = (
    ("FERTIG", "grounded cognition, executable skills and exact solving", "canonical repo + runtime adapter"),
    ("o1-state", "persistent O(1)-state host", "canonical repo"),
    ("OrganBank", "digest-addressed structured capabilities", "integrated artifact registry"),
    ("CRSA", "causal Local/Balanced/Free attention routing", "integrated operators"),
    ("FLCA", "evidence-gated compilation and replay", "canonical repo"),
    ("QAD", "precision-aware neural deployment", "model/compiler artifacts"),
)


def _components() -> int:
    for name, role, integration in COMPONENTS:
        print(f"{name:10}  {role}  [{integration}]")
    return 0


def _solve(question: str) -> int:
    runtime = ImmerRuntime([FertigSolver()])
    result = runtime.dispatch(Request("exact_math", question))
    print(json.dumps({
        "status": result.status.value,
        "component": result.component,
        "output": result.output,
        "reason": result.reason,
        "evidence": dict(result.evidence),
    }, ensure_ascii=False, sort_keys=True))
    return 0 if result.ok else 2


def _doctor() -> int:
    fertig_root = os.environ.get("IMMER_FERTIG_ROOT")
    organbank = os.environ.get("IMMER_ORGANBANK")
    checks = [
        ("FERTIG", bool(fertig_root and (Path(fertig_root).expanduser() / "fertig" / "solver.py").is_file()), "IMMER_FERTIG_ROOT"),
        ("OrganBank", bool(organbank and Path(organbank).expanduser().is_file()), "IMMER_ORGANBANK"),
    ]
    try:
        import torch  # noqa: F401
        crsa = True
    except ImportError:
        crsa = False
    checks.append(("CRSA/Torch", crsa, "pip install -e '.[neural]'"))
    for name, ready, hint in checks:
        print(f"{'✓' if ready else '·'} {name:12} {hint}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="immer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("components", help="show the system component map")
    sub.add_parser("doctor", help="inspect optional runtime integrations")
    solve = sub.add_parser("solve", help="run the FERTIG exact-math capability")
    solve.add_argument("question")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "components":
        return _components()
    if args.command == "doctor":
        return _doctor()
    if args.command == "solve":
        return _solve(args.question)
    raise AssertionError(args.command)
