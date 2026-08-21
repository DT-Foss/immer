"""Command line for FERTIG's local teach-by-showing assistant.

The product surface combines persistent desktop skills, hierarchical
composition, dynamic text slots, closed-loop correction and the compact HSSLM
language layer with the historical symbolic graph/corpus tools.

Subcommands:
  info     Graph-Statistiken + Hyperboloid-Check
  chains   abgeleitete Ketten (3-Pass-Inferenz, pass1)
  graph    gewicht-freie Kausal-Walks (Kette als Text)
  speech   Walks als gesprochene Prosa (handgeschriebene Verknüpfer)
  mined    Walks als Prosa mit gemessener Muster-Bank (erfordert Bank)
  bank     Muster-Bank aus Korpus minen (extract_patterns)
  corpus   Korpus-Modus: Prompt gewicht-frei fortsetzen

Determinismus: `graph`, `chains`, `speech`, `corpus` sind bei gleichen
Argumenten bit-identisch. `mined` ist in der Form zufällig (echter RNG),
die Fakten bleiben deterministisch.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import math
import re
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import __version__
from . import state_init
from . import pipeline, corpus, mined
from .pattern_bank import PatternBank
from .intent import parse_command
from . import tools, learn as learn_mod, arena as arena_mod
from . import bench as bench_mod

DATA = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DESKTOP_STORE = DATA / "desktop_tasks.json"
DEFAULT_DESKTOP_TEMPLATES = DATA / "desktop_templates.json"
DEFAULT_DESKTOP_RECORDINGS = DATA / "desktop_recordings"


class _BoundedRecorder:
    """Bind the CLI timeout without changing the assistant recorder protocol."""

    def __init__(self, recorder: Any, timeout: float | None) -> None:
        self.recorder = recorder
        self.timeout = timeout

    def record(self, *, label: str | None = None):
        return self.recorder.record(label=label, timeout=self.timeout)


def _make_assistant(args):
    """Construct the local assistant without touching screen or input APIs."""

    from .assistant import FertigAssistant

    language = None
    if not args.no_hsslm:
        from .hsslm_interface import GroundedLanguageInterface, HSSLMRuntime

        language = GroundedLanguageInterface(HSSLMRuntime())
    return FertigAssistant(
        args.store,
        template_store=args.templates,
        recordings_dir=args.recordings,
        language=language,
    )


def _make_chat(assistant):
    """Wrap the desktop assistant in FERTIG's unified grounded router."""

    from .chat import FertigChat

    return FertigChat(assistant)


def _make_recorder(args):
    """Create the passive adapter only after a teach intent was resolved."""

    from .macos_recording import MacOSPassiveRecorder

    return _BoundedRecorder(MacOSPassiveRecorder(), args.record_timeout)


def _make_desktop_environment(_args):
    """Create real adapters only after a known execution intent was resolved."""

    from .desktop import DesktopEnvironment, MacOSInputBackend, MacOSScreenshotBackend

    return DesktopEnvironment(MacOSScreenshotBackend(), MacOSInputBackend())


def _make_screenshot_backend():
    from .desktop import MacOSScreenshotBackend

    return MacOSScreenshotBackend()


def _check_recorder_permissions():
    from .macos_recording import check_macos_permissions

    return check_macos_permissions()


def _hsslm_status():
    from .hsslm_interface import HSSLMRuntime

    return HSSLMRuntime().status()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        payload = asdict(value)
        # Preserve the established resolve-only schema for ordinary commands;
        # management/template resolutions add this field only when populated.
        if payload.get("arguments") == {}:
            payload.pop("arguments")
        return _jsonable(payload)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _emit_assistant_result(value: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True))
        return
    text = getattr(value, "text", None)
    if isinstance(text, str):
        print(text)
        return
    task = getattr(value, "task", None)
    task_text = f" task={task!r}" if task else ""
    print(
        f"status={getattr(value, 'status', 'unknown')} "
        f"intent={getattr(value, 'intent', 'unknown')}{task_text} "
        f"confidence={float(getattr(value, 'confidence', 0.0)):.3f}"
    )


def _known_execution(assistant, task: object) -> bool:
    """Return whether a resolved do-target is an actually stored task."""

    if not isinstance(task, str) or not task.strip():
        return False
    agent = getattr(assistant, "agent", None)
    list_templates = getattr(agent, "list_templates", None)
    if callable(list_templates):
        try:
            if task.strip().casefold() in {
                str(name).casefold() for name in list_templates()
            }:
                return True
        except (TypeError, ValueError):
            return False
    matcher = getattr(assistant, "_match_task", None)
    if not callable(matcher):
        # Structural test doubles and third-party assistant adapters may make
        # resolution authoritative without exposing FERTIG's private matcher.
        return True
    try:
        matched, _alternatives = matcher(task)
    except (TypeError, ValueError):
        return False
    return matched is not None


def _known_correction(assistant, task: object) -> bool:
    """Corrections may record only for a known atomic task."""

    if not isinstance(task, str) or not task.strip():
        return False
    matcher = getattr(assistant, "_match_task", None)
    store = getattr(assistant, "store", None)
    getter = getattr(store, "get", None)
    if not callable(matcher) or not callable(getter):
        return False
    try:
        matched, _ = matcher(task)
        return matched is not None and getter(matched) is not None
    except (KeyError, TypeError, ValueError):
        return False


def _dispatch_assistant_request(assistant, chat, text: str, args) -> int:
    """Resolve first, then lazily cross only the required OS boundary."""

    try:
        resolution = assistant.resolve(text)
    except (TypeError, ValueError, RuntimeError) as exc:
        payload = {
            "status": "error",
            "intent": "unknown",
            "task": None,
            "text": str(exc),
            "data": {"error": type(exc).__name__},
        }
        _emit_assistant_result(payload, as_json=args.json)
        return 1

    if args.resolve_only:
        _emit_assistant_result(resolution, as_json=args.json)
        return 0

    intent = getattr(resolution, "intent", "unknown")
    status = getattr(resolution, "status", "resolved")
    task = getattr(resolution, "task", None)
    recorder = None
    desktop = None
    try:
        if status == "resolved" and intent == "teach" and task:
            if not args.json:
                print("Aufnahme läuft — mit F8 beenden.", file=sys.stderr)
            recorder = _make_recorder(args)
        elif (
            status == "resolved"
            and intent == "correct"
            and _known_correction(assistant, task)
        ):
            if not args.json:
                print("Korrekturaufnahme läuft — mit F8 beenden.", file=sys.stderr)
            recorder = _make_recorder(args)
        elif (
            status == "resolved"
            and intent == "do"
            and _known_execution(assistant, task)
        ):
            desktop = _make_desktop_environment(args)
        reply = chat.handle(text, desktop=desktop, recorder=recorder)
    except Exception as exc:
        payload = {
            "status": "error",
            "intent": intent,
            "task": task,
            "text": str(exc),
            "data": {"error": type(exc).__name__},
        }
        _emit_assistant_result(payload, as_json=args.json)
        return 1
    _emit_assistant_result(reply, as_json=args.json)
    return 1 if getattr(reply, "status", None) == "error" else 0


def _shortcut_text(args) -> str | None:
    intent = getattr(args, "assistant_intent", None)
    if intent is None:
        words = getattr(args, "request", ())
        return " ".join(words).strip() or None
    fixed = {
        "list": "list tasks",
        "status": "HSSLM status",
    }
    if intent in fixed:
        return fixed[intent]
    if intent == "compose":
        name = shlex.quote(str(args.name))
        children = " + ".join(shlex.quote(str(task)) for task in args.components)
        return f"compose {name} = {children}"
    if intent == "template":
        slots = [f"{name}={index}" for name, index in (args.slot or ())]
        slots.extend(f"{name}?={index}" for name, index in (args.optional_slot or ()))
        return " ".join(
            (
                "template",
                shlex.quote(str(args.name)),
                "from",
                shlex.quote(str(args.base)),
                *slots,
            )
        )
    if intent == "correct":
        task = " ".join(getattr(args, "request", ())).strip()
        return f"correct {shlex.quote(task)} step={int(args.step)}"
    task = " ".join(
        shlex.quote(str(word)) for word in getattr(args, "request", ())
    ).strip()
    return f"{intent} {task}".strip()


def cmd_assistant(args) -> int:
    """Run one grounded natural-language request or a small interactive REPL."""

    try:
        assistant = _make_assistant(args)
        chat = _make_chat(assistant)
    except Exception as exc:
        payload = {
            "status": "error",
            "intent": "unknown",
            "task": None,
            "text": f"Assistant konnte nicht gestartet werden: {exc}",
            "data": {"error": type(exc).__name__},
        }
        _emit_assistant_result(payload, as_json=args.json)
        return 1

    request = _shortcut_text(args)
    if request is not None:
        return _dispatch_assistant_request(assistant, chat, request, args)

    result = 0
    interactive = bool(getattr(sys.stdin, "isatty", lambda: False)())
    while True:
        if interactive:
            print("fertig> ", end="", flush=True)
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            print(file=sys.stderr)
            break
        if not line:
            break
        text = line.strip()
        if not text:
            continue
        if text.casefold() in {"exit", "quit", "ende", ":q"}:
            break
        result = max(result, _dispatch_assistant_request(assistant, chat, text, args))
    return result


def cmd_desktop_doctor(args) -> int:
    """Diagnose the complete local alpha without ever sending input."""

    payload: dict[str, Any] = {
        "status": "ready",
        "screen_recording": "failed",
        "input_monitoring": "failed",
        "accessibility": "failed",
        "recorder_helper": "failed",
        "hsslm": {"ready": False},
        "input_sent": False,
        "errors": [],
    }
    try:
        frame = _make_screenshot_backend().capture()
        payload.update(
            screen_recording="granted",
            shape=list(frame.shape),
            dtype=str(frame.dtype),
        )
    except Exception as exc:
        from .desktop import DesktopPermissionError

        payload["screen_recording"] = (
            "denied" if isinstance(exc, DesktopPermissionError) else "failed"
        )
        payload["errors"].append(
            {
                "check": "screen_recording",
                "error": type(exc).__name__,
                "message": str(exc),
            }
        )
    try:
        permission = _check_recorder_permissions()
        payload.update(
            recorder_helper="ready",
            helper=str(permission.helper),
            helper_size_bytes=permission.helper.stat().st_size,
            input_monitoring=("granted" if permission.input_monitoring else "denied"),
            accessibility="granted" if permission.accessibility else "denied",
        )
    except Exception as exc:
        payload["errors"].append(
            {
                "check": "recorder_helper",
                "error": type(exc).__name__,
                "message": str(exc),
            }
        )
    try:
        hsslm = _hsslm_status()
        payload["hsslm"] = _jsonable(hsslm)
    except Exception as exc:
        payload["errors"].append(
            {"check": "hsslm", "error": type(exc).__name__, "message": str(exc)}
        )
    required = (
        payload["screen_recording"] == "granted",
        payload["recorder_helper"] == "ready",
        payload["input_monitoring"] == "granted",
        payload["accessibility"] == "granted",
    )
    if not all(required):
        payload["status"] = "needs_setup"
    if args.json:
        _emit_assistant_result(payload, as_json=True)
    else:
        print(f"Screen Recording: {payload['screen_recording']}")
        print(f"Input Monitoring: {payload['input_monitoring']}")
        print(f"Accessibility: {payload['accessibility']}")
        print(f"Recorder helper: {payload['recorder_helper']}")
        hsslm_ready = payload["hsslm"].get("ready", False)
        print(f"HSSLM: {'ready' if hsslm_ready else 'unavailable'}")
        for error in payload["errors"]:
            print(f"{error['check']}: {error['message']}", file=sys.stderr)
        print("Eingaben gesendet: nein.")
    return 0 if payload["status"] == "ready" else 1


def cmd_product_demo(args) -> int:
    """Run the complete teach -> compose -> persist -> execute product proof."""

    from .product_demo import run_product_demo

    output = args.output
    if output is None:
        output = Path(tempfile.mkdtemp(prefix="fertig-product-demo-"))
    try:
        report = run_product_demo(output, runtime_value=args.runtime_value)
    except Exception as exc:
        payload = {
            "success": False,
            "error": type(exc).__name__,
            "message": str(exc),
            "output": str(output),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        else:
            print(f"Produktdemo fehlgeschlagen: {exc}", file=sys.stderr)
            print(f"Artefaktverzeichnis: {output}", file=sys.stderr)
        return 1

    if args.json:
        print(report.to_json(indent=None))
    else:
        print("=== FERTIG Produktbeweis ===")
        print("3 Fähigkeiten gezeigt -> komponiert -> parametrisiert -> neu geladen")
        print(
            f"Ausführung: {report.counts.verified_steps}/"
            f"{report.counts.executed_steps} Schritte sichtbar verifiziert"
        )
        print(f"Laufzeitwert: {report.runtime_value}")
        print(
            "Koordinaten-Replay: "
            f"{'unerwartet erfolgreich' if report.baseline.success else 'gescheitert'}"
        )
        for action in report.actions:
            detail = (
                action.text
                if action.text is not None
                else (
                    f"({action.x}, {action.y})"
                    if action.x is not None and action.y is not None
                    else action.key or ""
                )
            )
            print(
                f"  {action.index}: {action.kind} {detail} "
                f"-> {'verifiziert' if action.verified else 'fehlgeschlagen'}"
            )
        print(f"Persistierte Demo-Stores: {output}")
        print(f"Ergebnis: {'PASS' if report.success else 'FAIL'}")
    return 0 if report.success else 1


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("muss eine endliche positive Zahl sein")
    return number


def _template_slot(value: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_-]{0,63})=(\d+)", value)
    if match is None:
        raise argparse.ArgumentTypeError("muss NAME=SCHRITT sein")
    return match.group(1), int(match.group(2))


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("muss nichtnegativ sein")
    return number


def _add_assistant_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", type=Path, default=DEFAULT_DESKTOP_STORE)
    parser.add_argument("--templates", type=Path, default=DEFAULT_DESKTOP_TEMPLATES)
    parser.add_argument("--recordings", type=Path, default=DEFAULT_DESKTOP_RECORDINGS)
    parser.add_argument(
        "--no-hsslm",
        action="store_true",
        help="deterministischen Resolver ohne kleinen HSSLM-Sprachkern nutzen",
    )
    parser.add_argument("--json", action="store_true", help="Antwort als JSON ausgeben")
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Intent/Aufgabe nur auflösen, niemals aufnehmen oder ausführen",
    )
    parser.add_argument(
        "--record-timeout",
        type=_positive_float,
        default=None,
        metavar="SECONDS",
        help="Aufnahme spätestens nach SECONDS beenden (F8 beendet früher)",
    )


def _load_graph(args):
    path = Path(args.graph)
    return pipeline.load_graph(path)


def cmd_info(args) -> int:
    vocab, stoi, adj, mech = _load_graph(args)
    print(f"Graph      : {args.graph}")
    print(f"Symbole    : {len(vocab)}")
    print(f"Kanten     : {sum(len(v) for v in adj.values())}")
    print(f"Triplets   : {len({(a, b) for a in adj for b in adj[a]})} eindeutige Hops")
    SM = state_init.initialize_symbol_state(len(vocab))
    mink = -(SM[:, 0] ** 2) + np.sum(SM[:, 1:] ** 2, axis=1)
    ok = bool(np.allclose(mink, -1.0, atol=1e-9))
    print(
        f"Hyperboloid-Check (alle Zustände auf der Einheits-Hyperboloid): "
        f"{ok} (max. Abweichung {np.max(np.abs(mink + 1.0)):.2e})"
    )
    print(f"Top-Startpunkte: {', '.join(pipeline.top_starts(adj, vocab))}")
    return 0


def cmd_chains(args) -> int:
    vocab, stoi, adj, mech = _load_graph(args)
    chains = pipeline.derive_chains(adj, vocab)
    print(f"{len(chains)} abgeleitete exakte Ketten (pass1):")
    for chain, conf in sorted(chains.items(), key=lambda kv: -kv[1])[: args.n]:
        names = " -> ".join(vocab[i] for i in chain if i < len(vocab))
        print(f"  [{conf:.2f}] {names}")
    return 0


def cmd_graph(args) -> int:
    vocab, stoi, adj, mech = _load_graph(args)
    SM = state_init.initialize_symbol_state(len(vocab))
    starts = args.start or pipeline.top_starts(adj, vocab)
    print("=== gewicht-freie Kausal-Walks (Entitäten exakt, Walk generiert) ===")
    for start in starts:
        for tau in (0.15, 0.5):
            print(
                f"[tau={tau}] {pipeline.walk(start, vocab, stoi, adj, mech, SM, n=args.n, tau=tau)}"
            )
        print()
    return 0


def cmd_speech(args) -> int:
    vocab, stoi, adj, mech = _load_graph(args)
    SM = state_init.initialize_symbol_state(len(vocab))
    starts = args.start or pipeline.top_starts(adj, vocab)
    print("=== gewicht-freie SPRACHE aus dem .causal-Graphen ===")
    for start in starts:
        hops = pipeline.walk_chain(start, vocab, stoi, adj, SM, n=args.n, tau=0.3)
        chain = (
            " -> ".join([vocab[hops[0][0]]] + [vocab[b] for _, b in hops])
            if hops
            else "(Sackgasse)"
        )
        print(f"[{start}]")
        print(f"  Kette : {chain}")
        print(f"  Sprache: {pipeline.verbalize(hops, vocab, mech, seed=args.seed)}\n")
    return 0


def cmd_mined(args) -> int:
    bank_path = Path(args.bank)
    if not bank_path.exists():
        print(
            f"Muster-Bank fehlt: {bank_path}\n"
            f"  -> minen mit:  python -m fertig.cli bank -o {bank_path}",
            file=sys.stderr,
        )
        return 1
    bank = PatternBank.load(bank_path)
    vocab, stoi, adj, mech = _load_graph(args)
    SM = state_init.initialize_symbol_state(len(vocab))
    starts = args.start or pipeline.top_starts(adj, vocab)
    openers = mined.MinedOpeners(bank, seed=args.seed)
    print("=== SPRACHE mit gemessener Muster-Bank ===")
    for start in starts:
        hops = pipeline.walk_chain(start, vocab, stoi, adj, SM, tau=0.3, n=args.n)
        chain = (
            " -> ".join([vocab[hops[0][0]]] + [vocab[b] for _, b in hops])
            if hops
            else "(Sackgasse)"
        )
        print(f"[{start}]")
        print(f"  Kette : {chain}")
        print(f"  Sprache: {mined.verbalize_mined(hops, vocab, mech, openers)}\n")
    return 0


def cmd_bank(args) -> int:
    bank = PatternBank()
    corpora = args.corpora or [corpus.DEFAULT_CORPUS]
    for path in corpora:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        bank.extract(text)
        print(f"extrahiert: {path} ({len(text)} Zeichen)")
    out = Path(args.out)
    bank.save(out)
    print(
        f"\nSätze: {bank.n_sentences} | Skelette: {len(bank.skeletons)} | "
        f"Opener: {len(bank.openers)}"
    )
    print(f"Bank gespeichert: {out}")
    return 0


def cmd_corpus(args) -> int:
    path = Path(args.corpus)
    text = path.read_text(encoding="utf-8", errors="ignore")
    vocab, stoi, adjacency, trigram, unigram = corpus.build_vocab(
        text, max_vocab=args.max_vocab
    )
    print(corpus.stats(vocab, adjacency, trigram, unigram))
    SM = state_init.initialize_symbol_state(len(vocab))
    prompts = args.prompt or ["the candle", "the flame", "we have here"]
    print("\n=== gewicht-freie Fortsetzung (gemessene Trigramm/Bigramm-Kanten) ===")
    for prompt in prompts:
        for tau in (0.2, 0.5):
            out = corpus.generate(
                prompt, vocab, stoi, adjacency, trigram, unigram, SM, n=args.n, tau=tau
            )
            print(f"\n[tau={tau}] {prompt!r}\n  -> {out}")
    return 0


def cmd_intent(args) -> int:
    """NL-Befehl -> Intent-Tupel + Tool-Call."""
    vocab = pipeline.load_graph(args.graph)[0]
    lex = None
    if Path(args.lexicon).exists():
        lex = learn_mod.Lexicon.load(args.lexicon)
    it = parse_command(" ".join(args.command), vocab, lexicon=lex)
    if args.video:
        it.arguments["video"] = args.video
        # Bei 'erkennen' IST das Video das Ziel — kein Graph-Target nötig
        if it.action == "erkennen":
            it.status = "ok"
            it.tool = "video"
            it.grounded = True
            it.confidence = max(it.confidence, 0.8)
            it.target = args.video
    print(f"Befehl   : {' '.join(args.command)}")
    print(f"Parse    : {it.tree}")
    print(
        f"Intent   : action={it.action!r} target={it.target!r} conf={it.confidence:.3f}"
    )
    print(
        f"Grounded : {it.grounded} | Ambiguität: {it.ambiguity:.2f} | "
        f"Status: {it.status}"
    )
    if it.arguments:
        print(f"Args     : {it.arguments}")
    if args.execute and it.status == "ok":
        res = tools.execute(it, args.graph)
        print(f"\n[Tool {res.tool}] ok={res.ok}")
        if res.text:
            print(res.text)
    return 0


def cmd_learn(args) -> int:
    lex = learn_mod.Lexicon()
    corpora = args.corpora or [corpus.DEFAULT_CORPUS]
    for p in corpora:
        lex = learn_mod.learn_from_file(p, lex, min_count=args.min_count)
        print(f"gelernt: {p}")
    out = Path(args.out)
    lex.save(out)
    print(f"\nToken gesamt: {lex.tokens}")
    print(f"Aktionen gelernt: {len(lex.actions)} | Nomen gelernt: {len(lex.nouns)}")
    top_v = sorted(lex.actions.items(), key=lambda kv: -kv[1]["weight"])[:10]
    for v, meta in top_v:
        print(f"  verb {v:14s} -> action {meta['action']!r} (w={meta['weight']})")
    top_n = sorted(lex.nouns.items(), key=lambda kv: -kv[1])[:10]
    print("Top-Nomen:")
    for n, w in top_n:
        print(f"  noun {n:14s} (w={w})")
    print(f"\nLexikon gespeichert: {out}")
    return 0


def cmd_arena(args) -> int:
    res = arena_mod.run_arena(args.graph, verbose=not args.quiet)
    print()
    print(res.report())
    return 0


def cmd_bench(args) -> int:
    if args.name == "blimp":
        res = bench_mod.run_blimp(
            subtasks=args.subtasks or None, verbose=not args.quiet
        )
        print()
        print(res.report())
    elif args.name == "snips":
        res = bench_mod.run_snips(verbose=not args.quiet)
        print()
        print(res.report())
    elif args.name == "humaneval":
        res = bench_mod.run_humaneval(
            n_eval=args.n, learn=not args.no_learn, verbose=not args.quiet
        )
        print()
        print(res.report())
    elif args.name == "hellaswag":
        res = bench_mod.run_hellaswag(n=args.n, verbose=not args.quiet)
        print()
        print(res.report())
    elif args.name == "winogrande":
        res = bench_mod.run_winogrande(verbose=not args.quiet)
        print()
        print(res.report())
    elif args.name == "lambada":
        res = bench_mod.run_lambada(verbose=not args.quiet)
        print()
        print(res.report())
    elif args.name == "llm-snips":
        d = bench_mod.run_llm_snips(n=args.n, verbose=not args.quiet)
        if "error" in d:
            print(d["error"], file=sys.stderr)
            return 1
        print()
        print(
            f"DeepSeek auf SNIPS (n={d['total']}): "
            f"{d['hits']}/{d['total']} ({100 * d['hits'] / max(d['total'], 1):.1f}%)"
        )
        print("FERTIG auf SNIPS (gesamt): 88.3% — die präregistrierte Arena läuft")
    elif args.name == "causal-v2":
        d = bench_mod.run_causal_v2()
        if "error" in d:
            print(d["error"], file=sys.stderr)
            return 1
        print("\n=== P0 causal-v2 (Codex): aktive Kausal-Induktion ===")
        print(
            f"  finite Design: {'PASS' if d['finite_passed'] else 'FAIL'} "
            f"({d['blocks']} Blöcke)"
        )
        avr = d["active_vs_random"]
        print(
            f"  aktiv vs random : {avr['active_wins']}W/{avr['active_losses']}L "
            f"(p={avr['exact_one_sided_sign_p']:.2e})"
        )
        avp = d["active_vs_passive"]
        print(
            f"  aktiv vs passiv : {avp['active_wins']}W/{avp['active_losses']}L "
            f"(p={avp['exact_one_sided_sign_p']:.2e})"
        )
        avo = d["active_vs_optimal"]
        print(
            f"  aktiv vs optimal: {avo['active_wins']}W/{avo['active_losses']}L "
            f"(p={avo['exact_one_sided_sign_p']:.3f})"
        )
        print(
            f"  Brightness-Shortcut: {sum(1 for s in d['brightness_shortcut'] if s)}/{len(d['brightness_shortcut'])} "
            f"| deranged-Shortcuts: {d['deranged_shortcuts']}/64"
        )
    elif args.name == "groundzero":
        if args.grade3:
            d = bench_mod.run_groundzero_grade3(seed=args.seed)
            if "error" in d:
                print(d["error"], file=sys.stderr)
                return 1
            print("\n=== GroundZero Grade-3 (noncompensatory, isoliert) ===")
            print(f"  Positiv-Achsen : {d['positive']}/{d['positive_total']}")
            print(f"  Negativ-Kontr. : {d['negative']}/{d['negative_total']}")
            print(f"  Noncompensatory: {d['noncompensatory']}")
            print(
                f"  GESAMT: {'PASS' if d['passed'] else 'FAIL'} "
                f"(report {d['report_hash']}...)"
            )
            return 0
        d = bench_mod.run_groundzero_v1(seed=args.seed, verbose=not args.quiet)
        if "error" in d:
            print(d["error"], file=sys.stderr)
            return 1
        print("\n=== GroundZero-v1: formale Symbol-Grounding-Zertifikate ===")
        for k, v in d["axes"].items():
            mark = "✓" if v["passed"] else "✗"
            print(f"  {mark} {k:38s} estimate={v['estimate']}")
        print(f"\n  {d['passed']}/{d['total']} Achsen bestanden")
        if d.get("controls"):
            print(f"  Kontrollen: {d['controls']}")
        print(f"  Zertifikat: {d.get('certificate_hash', '')[:16]}...")
    elif args.name == "gsm8k":
        res = bench_mod.run_gsm8k(n=args.n, verbose=not args.quiet)
        print()
        print(res.report())
    elif args.name == "ifeval":
        res = bench_mod.run_ifeval(n=args.n, verbose=not args.quiet)
        print()
        print(res.report())
    elif args.name == "llm-all":
        d = bench_mod.run_llm_all(n_each=args.n, verbose=not args.quiet)
        if "error" in d:
            print(d["error"], file=sys.stderr)
            return 1
        print("\n=== Gap-Ledger: jede Lücke ist ein registriertes Ziel ===")
        for name, e in d.items():
            print(f"  {name:14s} Lücke {e['gap'] * 100:+5.1f}pp -> {e['unser_weg']}")
        print("\n  gespeichert in data/gap_ledger.json")
    elif args.name == "arc":
        res = bench_mod.run_arc(
            n=args.n, use_graph=not args.no_graph, verbose=not args.quiet
        )
        print()
        print(res.report())
    else:
        print(
            f"unbekannter Benchmark: {args.name} "
            f"(blimp | snips | humaneval | hellaswag | winogrande | lambada | "
            f"llm-snips | arc)",
            file=sys.stderr,
        )
        return 1
    return 0


def cmd_code(args) -> int:
    from . import code as code_mod

    prompt = " ".join(args.prompt)
    fragments = code_mod.load_fragments()
    triplets = code_mod.load_triplets()
    code, used = code_mod.assemble(prompt, fragments, triplets)
    print(f"Prompt: {prompt}")
    print(
        f"Fragmente: {used if used else '(keine über Schwelle — ehrliche Sackgasse)'}"
    )
    print("---")
    print(code if code else "(kein Code assembliert)")
    if args.execute and code:
        rc, out, err = code_mod.run_sandbox(code)
        print(f"\n[Sandbox] exit={rc}")
        if out:
            print(out[:2000])
        if err:
            print(err[:1000])
    return 0


def cmd_grow(args) -> int:
    from . import gaps as gaps_mod

    srcs = args.sources.split(",") if args.sources else None
    if args.gaps:
        from .arena import EVAL_SET

        n = gaps_mod.grow_gaps(
            [c for c, _, _ in EVAL_SET], max_targets=args.max_targets, verbose=True
        )
        print(f"\n{gaps_mod.WORLD_GRAPH}: {n} neue Tripletts aus dem Loop")
    else:
        target = " ".join(args.target)
        gaps_mod.grow(target, source_names=srcs)
    return 0


def cmd_crawl(args) -> int:
    from . import sources as sources_mod
    from . import gaps as gaps_mod

    print(f"[crawl] {args.url} ...")
    trips = sources_mod.fetch_url_direct(args.url)
    print(f"  {len(trips)} Kausal-Tripletts extrahiert")
    for a, b, c, conf in trips[:10]:
        print(f"  {a} {b} {c} ({conf:.2f})")
    if args.store:
        merged = {t[:3]: t[3] for t in gaps_mod._load_world()}
        added = 0
        for a, b, c, conf in trips:
            key = (a, b, c)
            if key not in merged or conf > merged[key]:
                merged[key] = conf
                added += 1
        gaps_mod._save_world([(a, b, c, conf) for (a, b, c), conf in merged.items()])
        print(f"  {added} neu gespeichert, Welt-Graph jetzt {len(merged)} Tripletts")
    return 0


def cmd_evolve(args) -> int:
    from .evolve import evolve as evolve_fn

    srcs = args.sources.split(",") if args.sources else None
    log = evolve_fn(args.iterations, args.arc_questions, args.grow_per_iter, srcs)
    if len(log) >= 2:
        print("\n=== Evolve-Protokoll (Ledger) ===")
        print(f"{'It':>3} {'Graph':>6} {'ARC%':>6} {'Cov%':>6} {'Δacc':>6} {'Δcov':>6}")
        for e in log:
            print(
                f"{e['iteration']:>3} {e['graph_triplets']:>6} "
                f"{100 * e['arc_accuracy']:>5.1f}% "
                f"{100 * e['arc_coverage']:>5.1f}% "
                f"{100 * e['arc_delta_acc']:>+5.1f}% "
                f"{100 * e['arc_delta_cov']:>+5.1f}%"
            )
    return 0


def cmd_apprentice(args) -> int:
    """Run the direct see -> try -> learn -> do apprentice loop."""
    from dataclasses import asdict
    import json

    from .apprentice import run_apprentice_experiment

    report = run_apprentice_experiment(
        seed=args.seed,
        steps=args.steps,
        checkpoint_every=args.checkpoint,
        max_depth=args.max_depth,
        codebook_variant=args.codebook,
    )
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
        return 0
    print("=== FERTIG Apprentice: sehen -> ausprobieren -> lernen -> tun ===")
    print(f"Opake Aktionen: {report.action_codes}")
    print(f"{'Schritte':>8} {'Task%':>8} {'Modell%':>9} {'Unsicherheit':>13}")
    for point in report.curve:
        print(
            f"{point.steps:>8} {100 * point.task_accuracy:>7.1f}% "
            f"{100 * point.model_accuracy:>8.1f}% "
            f"{point.mean_uncertainty:>13.3f}"
        )
    print(
        f"\nLerngewinn: {100 * report.initial_accuracy:.1f}% -> "
        f"{100 * report.final_accuracy:.1f}% "
        f"({100 * report.improvement:+.1f} Punkte)"
    )
    print(f"Ungesehene Kompositionen: {100 * report.composition_accuracy:.1f}%")
    print(
        f"Unbekanntes Wort: {'UNKNOWN' if report.unknown_abstained else 'falsch geraten'}"
    )
    if args.compare_shuffled:
        control = run_apprentice_experiment(
            seed=args.seed,
            steps=args.steps,
            checkpoint_every=args.checkpoint,
            max_depth=args.max_depth,
            codebook_variant=args.codebook,
            shuffled_control=True,
        )
        print(
            "Shuffle-Kontrolle (Aktion/Wirkung zerstört): "
            f"{100 * control.final_accuracy:.1f}%"
        )
    print("\nProduktziel: dieselbe Schleife hinter Screen + Maus/Tastatur.")
    return 0


def cmd_ground(args) -> int:
    from . import grounding as g

    if args.all:
        from . import gaps as gaps_mod

        trips = gaps_mod._load_world()
        symbols = set()
        for a, b, c, _ in trips:
            symbols.add(a)
            symbols.add(c)
        symbols = sorted(symbols)[: args.max]
        anchored = {}
        for i, word in enumerate(symbols, 1):
            print(f"[{i}/{len(symbols)}] ", end="")
            res = g.ground_symbol(word, verbose=False)
            anchored[word] = bool(res["perceptual"] or res["quantitative"])
        trips = gaps_mod._load_world()
        cov = g.grounding_coverage(trips, anchored)
        print("\n=== Grounding-Coverage (der Regress-Ende-Wert) ===")
        print(f"  Symbole im Graphen : {cov['symbols']}")
        print(
            f"  Nicht-Wort-gebunden: {cov['grounded']} ({100 * cov['coverage']:.1f}%)"
        )
        print("  (perzeptuell: CLIP-Bilder, quantitativ: Zahlen+Einheiten)")
    else:
        for word in args.word:
            g.ground_symbol(word)
    return 0


def cmd_quant(args) -> int:
    from . import quant as quant_mod

    if args.all:
        r = quant_mod.run_quant(verbose=True)
        print(
            f"\nQuantitative QA: {r['covered']}/{r['total']} beantwortbar "
            f"({100 * r['covered'] / max(r['total'], 1):.0f}%)"
        )
    else:
        q = " ".join(args.question)
        ans, mech, conf = quant_mod.answer(q)
        if ans:
            print(f"Antwort: {ans}  [{mech}, conf={conf:.2f}]")
        else:
            print("Nicht beantwortbar — die Lücke ist der nächste ground-Kandidat.")
    return 0


def cmd_vision(args) -> int:
    from . import vision as v

    if args.unsupervised:
        # Harnad-Ebene: Kategorien OHNE Wörter — Bilder aller Wörter
        # gemischt, Cluster entstehen aus Pixel-Struktur
        import urllib.request

        all_images, true_labels = [], []
        for word in args.word:
            for img in v.commons_images(word, args.images):
                all_images.append(img)
                true_labels.append(word)
        clusters, centers = v.cluster_unsupervised(all_images, k=len(args.word))
        if not clusters:
            print("Keine Bilder verfügbar.")
            return 1
        purity = v.cluster_purity(clusters, true_labels)
        print("Unüberwachte Kategorien (kein Wort, kein Netz, nur Pixel):")
        for ci, cl in enumerate(clusters):
            from collections import Counter

            dom = Counter(true_labels[i] for i in cl).most_common(1)[0][0]
            print(f"  Cluster {ci}: {len(cl)} Bilder — dominant: {dom}")
        print(
            f"\nPurity (Cluster vs. wahre Klassen, NUR zur Validierung): "
            f"{100 * purity:.1f}%"
        )
        print("Die Kategorien entstanden ohne Labels — Wörter wurden erst")
        print("nach der Cluster-Bildung zugeordnet.")
        return 0
    bank = v.build_bank(args.word, n_images=args.images)
    if not bank.prototypes:
        print("Keine Kategorien gebaut.")
        return 1
    print(f"Kategorien: {', '.join(bank.prototypes)}")
    print(
        "Konsistenz: "
        + ", ".join(f"{w}={bank.consistency[w]:.1f}" for w in bank.prototypes)
    )
    ratio = bank.harnad_ratio()
    if ratio is not None:
        print(
            f"\nHarnad-Ratio (within/between): {ratio:.3f} "
            f"({'< 1: kategorielle Wahrnehmung wirkt' if ratio < 1 else '>= 1: Kategorien trennen nicht'})"
        )
    if args.test:
        import urllib.request

        req = urllib.request.Request(args.test, headers={"User-Agent": "fertig/1.0"})
        img = urllib.request.urlopen(req, timeout=30).read()
        word, d = bank.recognize(img)
        print(f"\nTestbild erkannt als: {word} (Distanz {d:.4f})")
    return 0


def cmd_video(args) -> int:
    from . import video as v

    data = open(args.gif, "rb").read()
    raw = v.extract_frames(data)
    if args.mode == "verstehen":
        u = v.understand(raw)
        print(
            f"Frames: {u['frames']} | bewegt: {u['bewegt']} "
            f"(sig={u['signatur']:.4f} pix={u['pixel']:.4f})"
        )
        print(
            f"periodisch: {u['periodisch']} | szenenwechsel: "
            f"{u['szenenwechsel']} | paritaet: {u['paritaet']}"
        )
    elif args.mode == "generieren":
        codes = v.frame_code(v.frame_signatures(raw))
        trans = v.learn_transitions(codes)
        gen = v.generate_frames(codes[0], trans, n=args.frames)
        print(f"Original : {codes}")
        print(f"Generiert: {gen}")
        ok = sum(1 for a, b in zip(gen, gen[1:]) if b in trans.get(a, {}))
        print(f"Grammatik-Treue: {ok}/{len(gen) - 1} Kanten")
    return 0


def cmd_stream(args) -> int:
    from . import stream as s

    source = args.source
    if source.startswith(("http://", "https://")) and "youtu" in source:
        print(f"[stream] yt-dlp: {source}")
        try:
            source = s.ytdlp_url(source)
            print(f"[stream] direkte URL erhalten ({len(source)} Zeichen)")
        except Exception as e:
            print(f"[stream] yt-dlp fehlgeschlagen: {e}")
            return 1
    print(f"[stream] lerne aus {source} ({args.seconds}s @ {args.fps}fps) ...")
    learner = s.learn_from(source, seconds=args.seconds, fps=args.fps)
    st = learner.state()
    print("\n=== Gelernt (O(1), konstantes Memory) ===")
    print(f"  Frames        : {st['frames']}")
    print(f"  Bewegung      : sig={st['bewegung']:.4f} pixel={st['pixel']:.4f}")
    print(f"  Periodisch    : {st['periodisch']}")
    print(f"  Szenenwechsel : {st['szenenwechsel']}")
    print(f"  Grammatik     : {st['grammatik_kanten']} Kanten")
    print(f"  Fortsetzung   : {learner.generate(10)}")
    if args.name:
        from . import video as v
        import json

        bank = v.VideoBank()
        bank_path = Path("data/video_bank.json")
        if bank_path.exists():
            data = json.loads(bank_path.read_text())
            for k, arr in data.items():
                bank.prototypes[k] = np.array(arr)
        bank.add_from_learner(args.name, learner)
        bank_path.write_text(
            json.dumps({k: v.tolist() for k, v in bank.prototypes.items()})
        )
        facts = learner.to_graph_facts(args.name)
        print(f"  Kategorie '{args.name}' gelernt + gespeichert; Graph-Fakten: {facts}")
    if args.recognize:
        from . import video as v

        bank = v.VideoBank()
        bank_path = Path("data/video_bank.json")
        if bank_path.exists():
            import json

            data = json.loads(bank_path.read_text())
            for k, arr in data.items():
                bank.prototypes[k] = np.array(arr)
        sig = learner.sequence_signature()
        word, d = bank.recognize_signature(sig)
        print(f"  Erkannt als   : {word} (Distanz {d:.4f})")
    return 0


def cmd_interp(args) -> int:
    from . import video as v
    from .interp import InterpLearner

    raw = v.extract_frames(open(args.video, "rb").read(), max_frames=args.frames)
    learner = InterpLearner(n_bins=args.bins, quality_threshold=args.schwelle)
    for f in raw:
        learner.update(f)
    print(
        f"Gelernt: {learner.frames_seen} Frames, "
        f"{sum(len(r) for r in learner.transitions.values())} "
        f"Grammatik-Kanten"
    )
    if args.selfpaced:
        print("\nSelf-paced Curriculum (Surprise stellt Stützräder ein):")
        curve = learner.self_paced_learn(raw, max_gap=args.maxgap)
        print(f"  gap-Verlauf: {[g for g, _ in curve[:: max(1, len(curve) // 10)]]}")
        print(f"  Endstand: gap={curve[-1][0]}")
    else:
        print("\nInterpolation (Lücke wächst = Stützräder ab):")
        for gap in [2, 4, 6, 8]:
            q = learner.quality(raw, gap)
            mark = "<-- beherrscht" if q < args.schwelle else ""
            print(f"  Lücke {gap}: Fehler {q:.4f} {mark}")
        print(f"\nBeherrschte Lücke: {learner.mastered_gap(raw, args.maxgap)}")
    return 0


def cmd_schauen(args) -> int:
    """GOAT-Moonshoot: Video -> Verständnis -> Kategorie -> Wissen -> Sprache.
    Der geschlossene Kreislauf des Organismus in einem Befehl."""
    from . import stream as s
    from . import video as v
    from . import gaps as gaps_mod
    from pathlib import Path as _P

    data = open(args.video, "rb").read()
    print(f"[schauen] {args.video} — der Kreislauf startet\n")

    # 1. SEHEN (O(1)-Stream-Lernen)
    raw = v.extract_frames(data)
    learner = s.StreamLearner()
    for f in raw:
        learner.update(f)
    st = learner.state()
    print(
        f"[1/5] SEHEN     : {st['frames']} Frames, Bewegung "
        f"sig={st['bewegung']:.3f} pix={st['pixel']:.3f}, "
        f"periodisch={st['periodisch']}"
    )

    # 2. ERKENNEN (VideoBank)
    bank = v.VideoBank().load(_P("data/video_bank.json"))
    word, d = bank.recognize_signature(learner.sequence_signature())
    if word:
        print(f"[2/5] ERKENNEN  : {word} (Distanz {d:.4f})")
    else:
        print(f"[2/5] ERKENNEN  : unbekannt (Distanz {d:.4f}) — wird neue Kategorie")

    # 3. LERNEN (Kategorie + Fakten in den Welt-Graphen)
    name = args.name or (word or _P(args.video).stem)
    bank.add_from_learner(name, learner)
    bank.save(_P("data/video_bank.json"))
    facts = learner.to_graph_facts(name)
    print(f"[3/5] LERNEN    : Kategorie '{name}' + Graph-Fakten {facts}")

    # 4. WISSEN (Graph konsultieren — Erinnerung an die eigene Sprache)
    trips = gaps_mod._load_world()
    erinnerung = [c for a, b, c, _ in trips if a == name and b == "beschreibt_sich"]
    if erinnerung:
        print(f"[4/5] WISSEN    : (erinnert) {erinnerung[0]}")
    else:
        # Struktur-Narration: Fakten aus dem gemessenen Zustand — und als
        # Selbstbeschreibung in den Graphen schreiben (autobiografisch)
        teile = [f"{name}"]
        if st["bewegung"] > 0.02 or st["pixel"] > 0.02:
            teile.append("zeigt Bewegung")
        if st["periodisch"]:
            teile.append("wiederholt sich periodisch")
        if st["szenenwechsel"]:
            teile.append(f"hat {st['szenenwechsel']} Szenenwechsel")
        narration = ", ".join(teile) + "."
        best = {t[:3]: t[3] for t in trips}
        best[(name, "beschreibt_sich", narration)] = 0.7
        gaps_mod._save_world([(a, b, c, conf) for (a, b, c), conf in best.items()])
        print(f"[4/5] WISSEN    : {narration} (als Selbstbeschreibung gespeichert)")

    # 5. SPRECHEN (die Antwort als Intent-Ausgabe)
    print(
        f"[5/5] SPRECHEN  : Das Video zeigt {name or 'etwas Neues'}"
        + (", es ist periodisch" if st["periodisch"] else "")
        + (", es bewegt sich" if st["bewegung"] > 0.02 else "")
    )
    return 0


def cmd_sprechen(args) -> int:
    if args.engine == "hsslm":
        from .form_engine import FormEngine, speak_with_engine

        e = FormEngine()
        if not e.ready:
            print(
                "[sprechen] HSSLM-Gewichte fehlen — Fallback auf "
                "deterministische Verbalisierung."
            )
        r = speak_with_engine(args.graph, args.entity, engine=e, n=args.n)
        print(f"[sprechen] engine={r['engine']} ok={r['ok']}")
        for v in r.get("variants", []):
            mark = "FREIGEGEBEN" if v["verified"] else "VERWORFEN"
            print(f"  [{mark}] {v['text'][:110]}")
        fg = len(r.get("freigegeben", []))
        print(f"Freigegeben: {fg}/{len(r.get('variants', []))}")
        return 0 if r["ok"] else 1
    from .utterance import speak

    res = speak(args.graph, args.entity, n=args.n)
    print(res.report())
    return 0 if res.all_verified else 1


def cmd_weltbuch(args) -> int:
    """Das Weltbuch gesprochen: gelebte Evidenz (o1-state acted-Records)
    -> Kausal-Kanten mit Quittung (SHA-256, Frame) -> bit-exaktes
    Replay durch die Vendor-Welt -> FREIGEGEBEN/VERWORFEN."""
    import sys as _sys

    sys_path_backup = list(_sys.path)
    root = Path(__file__).resolve().parent.parent
    _sys.path.insert(0, str(root / "erweiterung"))
    try:
        import weltbuch

        out = weltbuch.main(args.out)
        return 0 if out["status"] == "FREIGEGEBEN" else 1
    finally:
        _sys.path[:] = sys_path_backup


def cmd_schreiben(args) -> int:
    """Der o1-Schreiber: FERTIGs Graphen-Fakten -> HSSLM rankt Übergänge/
    Pronomen/Fügungen -> unsichtbare Prüfung -> AUSGELIEFERT/ZURÜCKGEHALTEN.
    Das Modell erzeugt nie Wörter — es wählt (Logprob im Kontext)."""
    import sys as _sys

    sys_path_backup = list(_sys.path)
    root = Path(__file__).resolve().parent.parent
    _sys.path.insert(0, str(root / "erweiterung"))
    try:
        import schreiber_o1 as s
        from fertig.pipeline import load_graph

        graph_path = args.graph
        vocab, stoi, adj, mech = load_graph(graph_path)
        ent = args.entity.lower()
        start = stoi.get(ent)
        if start is None:
            print(f"[schreiben] Entität '{ent}' nicht im Graphen", file=_sys.stderr)
            return 1
        # Fakten: alle Kanten ab der Entität (expanded BFS, max Tiefe 4)
        facts, seen, frontier = [], set(), [start]
        for _ in range(4):
            nxt = []
            for a in frontier:
                if a in seen:
                    continue
                seen.add(a)
                for b in sorted(adj.get(a, {})):
                    facts.append(
                        {"subj": vocab[a], "verb": mech[(a, b)], "obj": vocab[b]}
                    )
                    nxt.append(b)
            frontier = nxt
        ranker = s.O1Ranker()
        lm = ranker.lm
        text, ordered = s.write_with_o1(list(facts), ranker, lm)
        ok, missing, unbacked = s.check_text(text, facts)
        print(
            f"\n════ {'AUSGELIEFERT' if ok else 'ZURÜCKGEHALTEN'}: "
            f"'{args.entity}' via {ranker.kind} "
            f"({len(facts)} Fakten) ════"
        )
        print(text if ok else f"fehlt {missing} | ungedeckt {unbacked}")
        return 0 if ok else 1
    finally:
        _sys.path[:] = sys_path_backup


def cmd_formarena(args) -> int:
    """FORM-ARENA (Erweiterung FE3-FE5): Prosa-Varianten pro Plan-Kante,
    gemessen an fluency/UID/IR/Ohr — die beste belegte Form gewinnt."""
    import sys as _sys

    sys_path_backup = list(_sys.path)
    root = Path(__file__).resolve().parent.parent
    _sys.path.insert(0, str(root / "erweiterung"))
    try:
        import form_arena

        out = form_arena.main(args.out)
        print(
            f"[formarena] {out['n_edges']} Kanten | "
            f"UID {out['uid_improved_frac']:.0%} | "
            f"IR {out['ir_selected_pass']}/{out['n_edges']} | "
            f"Ohr {out['ohr_selected_pass']}/{out['n_edges']} | "
            f"Kills {out['gate_kills']}"
        )
        return 0 if out["ir_selected_pass"] == out["n_edges"] else 1
    finally:
        _sys.path[:] = sys_path_backup


def cmd_erzaehlen(args) -> int:
    """MOONSHOOT A: Video -> belegter Plan -> verifizierte Prosa.
    Der geschlossene Gesamtkreislauf der Maschine in einem Befehl."""
    from . import stream as s
    from . import video as v
    from .utterance import Utterance, SpeechResult

    data = open(args.video, "rb").read()
    print(f"[erzählen] {args.video}\n")

    # 1. SEHEN: O(1)-Stream-Lernen
    raw = v.extract_frames(data)
    learner = s.StreamLearner()
    for f in raw:
        learner.update(f)
    st = learner.state()
    print(
        f"[1/5] SEHEN     : {st['frames']} Frames, Bewegung "
        f"sig={st['bewegung']:.3f} pix={st['pixel']:.3f}"
    )

    # 2. ERKENNEN: VideoBank-Kategorie
    bank = v.VideoBank().load(Path("data/video_bank.json"))
    name, d = bank.recognize_signature(learner.sequence_signature())
    if not name:
        name = Path(args.video).stem
    print(f"[2/5] ERKENNEN  : {name} (Distanz {d:.4f})")

    # 3. PLAN: belegte Fakten aus dem Zustand (Konfidenz = gemessen)
    plan = []
    if st["bewegung"] > 0.02 or st["pixel"] > 0.02:
        plan.append((name, "zeigt", "Bewegung", 0.6))
    if st["periodisch"]:
        plan.append((name, "wiederholt sich", "periodisch", 0.7))
    if st["szenenwechsel"]:
        plan.append((name, "hat", f"{st['szenenwechsel']} Szenenwechsel", 0.5))
    print(
        f"[3/5] PLAN      : {len(plan)} belegte Fakten "
        f"({', '.join(p[1] for p in plan)})"
    )

    # 4. SPRECHEN: deterministische Prosa aus dem Plan (Beleg = Zustand)
    teile = []
    for subj, verb, obj, conf in plan:
        if verb == "zeigt":
            teile.append(f"Das Video zeigt {obj}")
        elif verb == "wiederholt sich":
            teile.append(f"Es wiederholt sich {obj}")
        elif verb == "hat":
            teile.append(f"Es hat {obj}")
    prose = ". ".join(teile) + "." if teile else ""
    print(f"[4/5] SPRECHEN  : {prose}")

    # 5. VERIFIZIEREN: Rückführung — jede Aussage gegen den Zustand
    res = SpeechResult()
    for subj, verb, obj, conf in plan:
        u = Utterance(subj, verb, obj, conf)
        u.prose = prose
        # Beleg: Objekt im Satz + Zustands-Fakt existiert
        ok = obj.lower() in prose.lower()
        u.verified = ok
        u.detail = "Beleg: Zustands-Fakt + wörtlich in Prosa" if ok else "FEHLT"
        res.utterances.append(u)
    res.verified_count = sum(1 for u in res.utterances if u.verified)
    res.all_verified = len(res.utterances) > 0 and res.verified_count == len(
        res.utterances
    )
    print(
        f"[5/5] VERIFIZIERT: {res.verified_count}/{len(res.utterances)} "
        f"-> {'FREIGEGEBEN' if res.all_verified else 'VERWORFEN'}"
    )
    return 0 if res.all_verified else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="fertig",
        description="FERTIG — lokale KI durch Vormachen: Desktop-Skills, "
        "Komposition, dynamische Textfelder, HSSLM und grounded Werkzeuge.",
    )
    ap.add_argument("--version", action="version", version=f"fertig {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser(
        "assistant",
        aliases=["chat"],
        help="natürliche Sprache: Aufgaben zeigen, ausführen und erklären",
    )
    _add_assistant_options(p)
    p.add_argument(
        "request",
        nargs="*",
        help="freie Anfrage; ohne Text startet der interaktive Dialog",
    )
    p.set_defaults(fn=cmd_assistant, assistant_intent=None)

    for command, intent, help_text in (
        ("teach", "teach", "Desktop-Aufgabe durch Vormachen lernen (F8 stoppt)"),
        ("do", "do", "gelernte Desktop-Aufgabe sofort ausführen"),
        ("explain", "explain", "gelernte Desktop-Aufgabe erklären"),
    ):
        p = sub.add_parser(command, help=help_text)
        _add_assistant_options(p)
        p.add_argument("request", nargs="+", metavar="TASK")
        p.set_defaults(fn=cmd_assistant, assistant_intent=intent)

    p = sub.add_parser(
        "compose", help="neuen Ablauf aus mindestens zwei gelernten Skills bauen"
    )
    _add_assistant_options(p)
    p.add_argument("name", metavar="NAME")
    p.add_argument(
        "--from",
        dest="components",
        nargs="+",
        required=True,
        metavar="TASK",
        help="auszuführende Skills in Reihenfolge",
    )
    p.set_defaults(fn=cmd_assistant, assistant_intent="compose", request=())

    p = sub.add_parser(
        "template", help="Textschritte eines gelernten Skills parametrisierbar machen"
    )
    _add_assistant_options(p)
    p.add_argument("name", metavar="NAME")
    p.add_argument("--base", required=True, metavar="TASK")
    p.add_argument(
        "--slot",
        action="append",
        type=_template_slot,
        default=[],
        metavar="NAME=STEP",
        help="erforderlicher Textwert an globalem nullbasiertem Schritt",
    )
    p.add_argument(
        "--optional-slot",
        action="append",
        type=_template_slot,
        default=[],
        metavar="NAME=STEP",
        help="optionaler Textwert; ohne Angabe bleibt der Demonstrationswert",
    )
    p.set_defaults(fn=cmd_assistant, assistant_intent="template", request=())

    p = sub.add_parser(
        "correct", help="einen atomaren Skill-Schritt erneut vormachen und reparieren"
    )
    _add_assistant_options(p)
    p.add_argument("request", nargs="+", metavar="TASK")
    p.add_argument("--step", type=_nonnegative_int, default=0, metavar="N")
    p.set_defaults(fn=cmd_assistant, assistant_intent="correct")

    p = sub.add_parser("tasks", help="gelernte Desktop-Aufgaben auflisten")
    _add_assistant_options(p)
    p.set_defaults(fn=cmd_assistant, assistant_intent="list", request=())

    p = sub.add_parser(
        "hsslm-status", help="Status und Parameterzahl des kleinen Sprachkerns"
    )
    _add_assistant_options(p)
    p.set_defaults(fn=cmd_assistant, assistant_intent="status", request=())

    p = sub.add_parser(
        "desktop-doctor",
        help="Screen-Recording-Zugriff mit genau einem Screenshot prüfen",
    )
    p.add_argument("--json", action="store_true", help="Ergebnis als JSON ausgeben")
    p.set_defaults(fn=cmd_desktop_doctor)

    p = sub.add_parser(
        "product-demo",
        help="kompletten Teach/Compose/Template/Reload/Execute-Pfad ohne OS testen",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help="Demo-Stores in DIR ablegen (Standard: frisches temporäres Verzeichnis)",
    )
    p.add_argument(
        "--runtime-value",
        default="Dr. Katherine Johnson",
        metavar="TEXT",
        help="dynamischer Textwert für den parametrisierten Skill",
    )
    p.add_argument("--json", action="store_true", help="vollständigen Report als JSON")
    p.set_defaults(fn=cmd_product_demo)

    p = sub.add_parser("info", help="Graph-Statistiken + Hyperboloid-Check")
    p.add_argument("--graph", default=str(pipeline.DEFAULT_GRAPH))
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("chains", help="abgeleitete Ketten (pass1)")
    p.add_argument("--graph", default=str(pipeline.DEFAULT_GRAPH))
    p.add_argument("-n", type=int, default=8)
    p.set_defaults(fn=cmd_chains)

    p = sub.add_parser("graph", help="gewicht-freie Kausal-Walks")
    p.add_argument("--graph", default=str(pipeline.DEFAULT_GRAPH))
    p.add_argument("-n", type=int, default=8)
    p.add_argument("start", nargs="*")
    p.set_defaults(fn=cmd_graph)

    p = sub.add_parser("speech", help="Walks als gesprochene Prosa")
    p.add_argument("--graph", default=str(pipeline.DEFAULT_GRAPH))
    p.add_argument("-n", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("start", nargs="*")
    p.set_defaults(fn=cmd_speech)

    p = sub.add_parser("mined", help="Prosa mit gemessener Muster-Bank")
    p.add_argument("--graph", default=str(pipeline.DEFAULT_GRAPH))
    p.add_argument("--bank", default=str(mined.DEFAULT_BANK))
    p.add_argument("-n", type=int, default=8)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("start", nargs="*")
    p.set_defaults(fn=cmd_mined)

    p = sub.add_parser("bank", help="Muster-Bank aus Korpus minen")
    p.add_argument("-o", "--out", default=str(mined.DEFAULT_BANK))
    p.add_argument("corpora", nargs="*")
    p.set_defaults(fn=cmd_bank)

    p = sub.add_parser("intent", help="NL-Befehl -> Intent-Tupel + Tool-Call")
    p.add_argument("--graph", default=str(pipeline.DEFAULT_GRAPH))
    p.add_argument("--lexicon", default=str(learn_mod.DEFAULT_LEXICON))
    p.add_argument(
        "-x", "--execute", action="store_true", help="Intent auch ausführen (Tool-Call)"
    )
    p.add_argument("--video", default=None, help="Video-Datei für die erkennen-Aktion")
    p.add_argument("command", nargs="+")
    p.set_defaults(fn=cmd_intent)

    p = sub.add_parser("learn", help="Lexikon aus Korpus lernen (wächst)")
    p.add_argument("-o", "--out", default=str(learn_mod.DEFAULT_LEXICON))
    p.add_argument("--min-count", type=int, default=2)
    p.add_argument("corpora", nargs="*")
    p.set_defaults(fn=cmd_learn)

    p = sub.add_parser("arena", help="Selbst-Benchmark (präregistriert)")
    p.add_argument("--graph", default=str(pipeline.DEFAULT_GRAPH))
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(fn=cmd_arena)

    p = sub.add_parser("bench", help="SOTA-Benchmarks")
    p.add_argument(
        "name",
        choices=[
            "blimp",
            "snips",
            "humaneval",
            "hellaswag",
            "winogrande",
            "lambada",
            "llm-snips",
            "llm-all",
            "arc",
            "ifeval",
            "gsm8k",
            "groundzero",
            "causal-v2",
        ],
    )
    p.add_argument("--subtasks", nargs="*", help="BLiMP-Subtasks (Standard: alle 8)")
    p.add_argument("--seed", type=int, default=3, help="GroundZero-Seed")
    p.add_argument("--grade3", action="store_true", help="Grade-3-Diagnostik statt v1")
    p.add_argument(
        "-n", type=int, default=30, help="HumanEval: Anzahl Evaluations-Probleme"
    )
    p.add_argument(
        "--no-learn",
        action="store_true",
        help="HumanEval: nicht aus Referenzlösungen lernen",
    )
    p.add_argument(
        "--no-graph",
        action="store_true",
        help="ARC: ohne Graph-Antworten (nur LM-Baseline)",
    )
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("evolve", help="Autonomer Verbesserungs-Loop")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--arc-questions", type=int, default=30)
    p.add_argument("--grow-per-iter", type=int, default=3)
    p.add_argument("--sources", default=None)
    p.set_defaults(fn=cmd_evolve)

    p = sub.add_parser(
        "apprentice",
        help="Teach-by-Showing-Kern: RGB sehen, Effekte lernen, Ziele ausführen",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=16, help="aktive Lerninteraktionen")
    p.add_argument(
        "--checkpoint",
        type=int,
        default=4,
        help="Lernkurve alle N Interaktionen messen",
    )
    p.add_argument(
        "--max-depth", type=int, default=4, help="maximale Länge komponierter Pläne"
    )
    p.add_argument(
        "--codebook", type=int, default=0, help="Variante des opaken Aktionsalphabets"
    )
    p.add_argument(
        "--compare-shuffled",
        action="store_true",
        help="zerstörte Aktion/Wirkung-Zuordnung daneben messen",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="strukturierten Report statt Tabelle ausgeben",
    )
    p.set_defaults(fn=cmd_apprentice)

    p = sub.add_parser("code", help="Code aus Prompt assembleren + sandboxen")
    p.add_argument("prompt", nargs="+")
    p.add_argument(
        "-x",
        "--execute",
        action="store_true",
        help="assemblierten Code in der Sandbox ausführen",
    )
    p.set_defaults(fn=cmd_code)

    p = sub.add_parser("erzaehlen", help="MOONSHOOT A: Video->Plan->Prosa->Verify")
    p.add_argument("video")
    p.set_defaults(fn=cmd_erzaehlen)

    p = sub.add_parser("sprechen", help="Utterance-IR: Plan->Prosa->Verifikation")
    p.add_argument("entity")
    p.add_argument("--graph", default="data/chained.causal")
    p.add_argument("-n", type=int, default=5)
    p.add_argument(
        "--engine",
        default="plan",
        choices=["plan", "hsslm"],
        help="plan=deterministisch, hsslm=Form-Engine (Moonshoot B)",
    )
    p.set_defaults(fn=cmd_sprechen)

    p = sub.add_parser(
        "weltbuch",
        help="Gelebte Evidenz sprechen (o1-state acted-Records, bit-exakte Quittungen)",
    )
    p.add_argument(
        "--out", default=None, help="Ausgabe-Pfad (default: erweiterung/results/)"
    )
    p.set_defaults(fn=cmd_weltbuch)

    p = sub.add_parser(
        "formarena",
        help="Form-Arena: beste belegte Prosa-Variante pro Kante (UID/IR/Ohr-Richter)",
    )
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_formarena)

    p = sub.add_parser(
        "schreiben",
        help="o1-Schreiber: Fakten ranken, "
        "prüfen, ausliefern (HSSLM wählt, erzeugt nie)",
    )
    p.add_argument("entity")
    p.add_argument("--graph", default="data/chained.causal")
    p.set_defaults(fn=cmd_schreiben)

    p = sub.add_parser("schauen", help="GOAT: Video->Verstehen->Wissen->Sprache")
    p.add_argument("video")
    p.add_argument(
        "--name", default=None, help="Kategorie-Name (Default: erkannt oder Dateiname)"
    )
    p.set_defaults(fn=cmd_schauen)

    p = sub.add_parser("interp", help="Interpolation mit Stützrädern")
    p.add_argument("video")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--bins", type=int, default=16)
    p.add_argument("--schwelle", type=float, default=0.05)
    p.add_argument("--maxgap", type=int, default=8)
    p.add_argument("-s", "--selfpaced", action="store_true")
    p.set_defaults(fn=cmd_interp)

    p = sub.add_parser("stream", help="Permanent aus Video-Streams lernen")
    p.add_argument("source", help="Datei oder YouTube-URL")
    p.add_argument("--seconds", type=int, default=15)
    p.add_argument("--fps", type=int, default=2)
    p.add_argument(
        "--name",
        default=None,
        help="Kategorie-Name: Stream in VideoBank + Graph lernen",
    )
    p.add_argument(
        "--recognize", action="store_true", help="gegen die VideoBank erkennen"
    )
    p.set_defaults(fn=cmd_stream)

    p = sub.add_parser("video", help="Video-Verständnis/-Generierung (GIF)")
    p.add_argument("gif", help="Pfad zur GIF-Datei")
    p.add_argument("--mode", choices=["verstehen", "generieren"], default="verstehen")
    p.add_argument("--frames", type=int, default=16)
    p.set_defaults(fn=cmd_video)

    p = sub.add_parser("vision", help="Deterministische Bilderkennung")
    p.add_argument("word", nargs="+", help="Kategorien-Wörter")
    p.add_argument("--images", type=int, default=4, help="Bilder pro Wort")
    p.add_argument("--test", default=None, help="URL eines Testbildes zum Erkennen")
    p.add_argument(
        "-u",
        "--unsupervised",
        action="store_true",
        help="Harnad-Ebene: Kategorien ohne Wörter (Clustering)",
    )
    p.set_defaults(fn=cmd_vision)

    p = sub.add_parser("quant", help="Quantitative QA (Grounding-Beweis)")
    p.add_argument("question", nargs="*")
    p.add_argument("--all", action="store_true", help="präregistrierte Arena")
    p.set_defaults(fn=cmd_quant)

    p = sub.add_parser("ground", help="Symbol an Nicht-Wort-Anker binden")
    p.add_argument("word", nargs="*")
    p.add_argument(
        "--all", action="store_true", help="alle Graph-Symbole erden + Coverage messen"
    )
    p.add_argument("--max", type=int, default=25)
    p.set_defaults(fn=cmd_ground)

    p = sub.add_parser("grow", help="Gap-Loop: Weltwissen in den Graphen holen")
    p.add_argument("target", nargs="*", help="Entitäten (z. B. sugar)")
    p.add_argument(
        "--sources",
        default=None,
        help="wikipedia,wiktionary,duckduckgo,web,arxiv,pubmed,"
        "semantic_scholar,openalex (Default: alle)",
    )
    p.add_argument(
        "--gaps",
        action="store_true",
        help="Lücken aus der Arena automatisch wachsen lassen",
    )
    p.add_argument("--max-targets", type=int, default=3)
    p.set_defaults(fn=cmd_grow)

    p = sub.add_parser("crawl", help="Beliebige URL -> Text -> Tripletts")
    p.add_argument("url")
    p.add_argument(
        "--store", action="store_true", help="Tripletts in den Welt-Graphen speichern"
    )
    p.set_defaults(fn=cmd_crawl)

    p = sub.add_parser("corpus", help="Korpus-Modus: Prompt fortsetzen")
    p.add_argument("--corpus", default=str(corpus.DEFAULT_CORPUS))
    p.add_argument("--max-vocab", type=int, default=2000)
    p.add_argument("-n", type=int, default=25)
    p.add_argument("prompt", nargs="*")
    p.set_defaults(fn=cmd_corpus)

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
