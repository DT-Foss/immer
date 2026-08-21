from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .cognition.fertig import FertigSolver
from .contracts import Request
from .runtime import ImmerRuntime
from .substrate import LifeDaemon


COMPONENTS = (
    ("FERTIG", "grounded cognition, executable skills and exact solving", "vendored + runtime adapter"),
    ("o1-state", "persistent O(1)-state host", "vendored organism bridge"),
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
    from .runtimes.o1_state.adapter import is_available as o1state_available

    fertig_root = os.environ.get("IMMER_FERTIG_ROOT")
    organbank = os.environ.get("IMMER_ORGANBANK")
    checks = [
        ("FERTIG", (Path(__file__).parent / "cognition" / "fertig" / "_vendor" / "fertig" / "solver.py").is_file() or bool(fertig_root), "vendored; override via IMMER_FERTIG_ROOT"),
        ("o1-state", o1state_available(), "pip install -e '.[neural]'"),
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


def _organs(args: argparse.Namespace) -> int:
    from .capabilities.organbank import OrganBank

    if not args.manifest:
        print("organ manifest required: --manifest PATH or IMMER_ORGANBANK", file=sys.stderr)
        return 2
    bank = OrganBank.from_manifest(args.manifest)
    if args.organ_command == "list":
        for name in bank.names():
            descriptor = bank.descriptor(name)
            print(f"{name:24} {descriptor.capability:16} {descriptor.group}")
        return 0
    artifact = bank.verify(args.name)
    print(json.dumps({"organ": args.name, "verified": True, "artifact": str(artifact)}, ensure_ascii=False))
    return 0


def _serve(args: argparse.Namespace) -> int:
    """The organism lives in this terminal: type to it, it remembers."""
    from .runtimes.o1_state.adapter import O1StateStream, is_available

    stream = None
    if is_available():
        try:
            sidecar = Path(args.state).expanduser()
            stream = O1StateStream(sidecar=sidecar.with_suffix(".pt"))
        except RuntimeError as exc:
            print(f"· o1-state bridge: {exc}", file=sys.stderr)
    bank = None
    organbank = os.environ.get("IMMER_ORGANBANK")
    if organbank and Path(organbank).expanduser().is_file():
        from .capabilities.organbank import OrganBank

        bank = OrganBank.from_manifest(organbank)
    daemon = LifeDaemon(stream=stream, bank=bank, state_path=args.state)
    daemon.register(FertigSolver())
    print(f"immer serve — turns alive: {daemon.turns}; organs: {', '.join(daemon.rack.mounted()) or 'none'}")
    print("commands: /state /say <text> /quit — anything else is experience")
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/state":
            print(json.dumps(daemon.snapshot(), ensure_ascii=False, sort_keys=True))
            continue
        if text.startswith("/say "):
            daemon.say("utterance", text[5:])
            print("(gesagt)")
            continue
        daemon.submit_user(text)
        answer = daemon.request("exact_math", text)
        if answer.ok:
            print(f"[exact] {answer.output}")
        elif answer.status.value == "abstained":
            print("[exact] weiß ich nicht")
        print(f"(turns: {daemon.turns}, tokens: {getattr(stream, 'tokens', 0)})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="immer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("components", help="show the system component map")
    sub.add_parser("doctor", help="inspect optional runtime integrations")
    solve = sub.add_parser("solve", help="run the FERTIG exact-math capability")
    solve.add_argument("question")
    organs = sub.add_parser("organs", help="inspect the cold organ bank")
    organs.add_argument("--manifest", default=os.environ.get("IMMER_ORGANBANK"))
    organs_sub = organs.add_subparsers(dest="organ_command", required=True)
    organs_sub.add_parser("list", help="list registered organs")
    mount = organs_sub.add_parser("mount", help="verify one organ's digest")
    mount.add_argument("name")
    serve = sub.add_parser("serve", help="let the organism live in this terminal")
    serve.add_argument("--state", default="~/.immer/life.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "components":
        return _components()
    if args.command == "doctor":
        return _doctor()
    if args.command == "solve":
        return _solve(args.question)
    if args.command == "organs":
        return _organs(args)
    if args.command == "serve":
        return _serve(args)
    raise AssertionError(args.command)
