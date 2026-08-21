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
    """The organism lives in this terminal: it learns, remembers and answers."""
    from .intent import classify
    from .library import Library
    from .memory import SpanStore
    from .runtimes.o1_state.adapter import is_available
    from .runtimes.o1_state.plasticity import LearningStream
    from .runtimes.qwen.adapter import QwenBrain
    from .suite import Metrics, start_dashboard

    state = Path(args.state).expanduser()
    state.parent.mkdir(parents=True, exist_ok=True)

    stream = None
    if is_available():
        try:
            stream = LearningStream(sidecar=state.with_suffix(".pt"))
        except RuntimeError as exc:
            print(f"· lernender Strom: {exc}", file=sys.stderr)
    else:
        print("· o1-state bridge nicht verfügbar — Leben ohne Lernen", file=sys.stderr)

    qwen = QwenBrain()
    council = None
    if args.council and qwen.model_id is not None:
        from .council import Council

        council = Council(
            (
                QwenBrain(name="qwen.basis"),
                QwenBrain(
                    name="qwen.kritiker",
                    persona="Du bist der Kritiker im Rat eines Lebewesens. Prüfe Aussagen "
                    "auf Fehler und Widersprüche und korrigiere sie. Antworte kurz.",
                ),
                QwenBrain(
                    name="qwen.freigeist",
                    persona="Du bist der Freigeist im Rat eines Lebewesens. Denk "
                    "unkonventionell und bringe den Blickwinkel, den niemand sonst hat. "
                    "Antworte kurz.",
                ),
            ),
            name="rat",
        )
    library = Library(SpanStore(state.parent / "memory.json"), harvester=qwen)
    metrics = Metrics(status_path=state.parent / "status.json", jsonl_path=state.parent / "metrics.jsonl")

    bank = None
    organbank = os.environ.get("IMMER_ORGANBANK")
    if organbank and Path(organbank).expanduser().is_file():
        from .capabilities.organbank import OrganBank

        bank = OrganBank.from_manifest(organbank)
    daemon = LifeDaemon(stream=stream, bank=bank, state_path=args.state)
    daemon.register(FertigSolver())

    history: list[dict[str, str]] = []

    def gauges() -> dict:
        life = stream.metrics() if stream is not None else {}
        return {
            "turns": daemon.turns,
            "tokens": life.get("tokens", 0),
            "loss_ema": life.get("loss_ema"),
            "surprises": life.get("surprises", 0),
            "updates": life.get("updates", 0),
            "sleeps": life.get("sleeps", 0),
            "span_buffer": life.get("span_buffer", 0),
            "qwen": qwen.loaded,
            "qwen_model": qwen.model_id,
            "spans": library.store.count(),
        }

    port = None if args.no_dashboard else (args.dashboard or 8787)
    if port:
        try:
            start_dashboard(metrics, gauges, port)
            print(f"◆ dashboard: http://127.0.0.1:{port}/  (Maschinenfutter: /status)")
        except OSError as exc:
            print(f"· dashboard aus: {exc}", file=sys.stderr)

    print(f"immer serve — turns alive: {daemon.turns}; organs: {', '.join(daemon.rack.mounted()) or 'none'}")
    print("commands: /state /say <text> /sleep /quit — alles andere ist Erfahrung")
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/state":
            print(json.dumps(metrics.emit(gauges()), ensure_ascii=False, sort_keys=True))
            continue
        if text.startswith("/say "):
            daemon.say("utterance", text[5:])
            print("(gesagt)")
            continue
        if text == "/sleep":
            if stream is not None:
                print(f"(konsolidiert: {stream.sleep()})")
            else:
                print("(kein Strom zum Konsolidieren)")
            continue

        intent = classify(text)
        daemon.submit_user(text)  # das Leben trinkt zuerst alles

        if intent.kind == "TEACH":
            span = library.teach(intent.payload)
            metrics.bump("teaches")
            print(f"(gemerkt: {span['key']})")
        elif intent.kind == "RECALL":
            hits = library.recall(intent.payload)
            metrics.bump("recalls")
            if not hits and qwen.model_id is not None:
                metrics.bump("harvests")
                card = library.grow(intent.payload)
                if card is not None:
                    print(f"[ernte] {card['text']}")
                    print(f"         (quelle: {card['source']} — geprüft?)")
                    hits = [card]
            if hits:
                for span in hits:
                    print(f"[erinnerung] {span['text']} ({span.get('source', '?')})")
            elif not hits:
                print("[erinnerung] da weiß ich noch nichts — lehr mich (merke: …)")
        elif intent.kind == "MATH":
            answer = daemon.request("exact_math", intent.payload)
            if answer.ok:
                metrics.bump("math_ok")
                print(f"[exact] {answer.output}")
            else:
                metrics.bump("math_abstained")
                print("[exact] weiß ich nicht")
        elif intent.kind == "STATUS":
            metrics.bump("status")
            print(json.dumps(metrics.snapshot(gauges()), ensure_ascii=False))
        else:  # CHAT
            metrics.bump("chat")
            meta = {
                "history": history[-6:],
                "life": f"turns={daemon.turns} tokens={getattr(stream, 'tokens', 0)}",
            }
            if council is not None:
                result = council.deliberate("chat", intent.payload, metadata=meta)
            else:
                result = qwen.handle(Request("chat", intent.payload, metadata=meta))
            if result.ok:
                tag = "[rat]" if council is not None else "[qwen]"
                print(f"{tag} {result.output}")
                if council is not None and result.evidence.get("votes"):
                    print(
                        f"         (stimmen: {result.evidence['votes']}, "
                        f"übereinstimmend: {result.evidence['agree']})"
                    )
                history.append({"role": "user", "content": intent.payload})
                history.append({"role": "assistant", "content": result.output})
            else:
                print(f"[stille] (kein Mund: {result.reason})")

        metrics.emit(gauges())
        life_line = f"turns: {daemon.turns}"
        if stream is not None:
            life_line += f", tokens: {stream.tokens}, updates: {stream.updates}, überraschungen: {stream.surprises}"
        print(f"({life_line})")
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
    serve.add_argument(
        "--dashboard", nargs="?", const=8787, type=int, default=8787,
        help="metrics UI port (default 8787)",
    )
    serve.add_argument("--no-dashboard", action="store_true", help="disable the metrics UI")
    serve.add_argument("--council", action="store_true", help="chat through a council of three Qwen roles")
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
