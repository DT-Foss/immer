from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from .contracts import Request
from .resource_paths import s3_ship_manifest
from .substrate import LifeDaemon


COMPONENTS = (
    ("FERTIG", "grounded deterministic execution and verification", "vendored + runtime adapter"),
    ("CRSA", "Causal Prefix Sinkhorn Attention", "integrated operators"),
    ("WorldStream", "exact local and pinned-remote Safetensors ranges", "integrated tensor source"),
    ("LiveCausal", "append-only causal control plane", "integrated lazy graph"),
    ("CausalWeights", "local weight bundle and exact causal range routes", "integrated Qwen pager path"),
    ("Qwen3.8", "complete local causal teacher and novelty fallback", "primary runtime"),
    ("OoE", "persistent PS-Lifted Markov agents and executable Crystals", "integrated runtime"),
    ("AnchorBattery", "authenticated prefix-state restoration", "integrated Qwen path"),
    ("DeepSeekV4", "complete 43-layer frontier decoder", "retained backend"),
    ("MarkovRouter", "label-free next-layer expert transport hints", "integrated evaluation path"),
    ("OrganBank", "digest-addressed exact capabilities", "integrated artifact registry"),
    ("o1-state", "persistent life stream outside frozen execution", "integrated runtime"),
)


def _sorted_layer_list(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layers must be comma-separated integers"
        ) from exc
    if (
        not layers
        or layers != tuple(sorted(set(layers)))
        or any(layer < 0 for layer in layers)
    ):
        raise argparse.ArgumentTypeError(
            "layers must be sorted unique non-negative integers"
        )
    return layers

def _s3_manifest(configured: str | Path | None = None) -> Path:
    """Resolve the one deployment manifest used by solve, serve and organs."""

    selected = (
        configured
        or os.environ.get("IMMER_S3_MANIFEST")
        or os.environ.get("IMMER_ORGANBANK")
        or s3_ship_manifest()
    )
    return Path(selected).expanduser().resolve()


def _artifact_root(manifest: str | Path, configured: str | Path | None = None) -> Path | None:
    from .artifacts import artifact_root

    return artifact_root(manifest, configured)


def _components() -> int:
    for name, role, integration in COMPONENTS:
        print(f"{name:10}  {role}  [{integration}]")
    return 0


def _solve(
    question: str,
    manifest: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> int:
    from .composition import CompositionRoot

    composition = CompositionRoot.build(
        s3_manifest=_s3_manifest(manifest),
        s3_artifact_root=artifact_root,
    )
    result = composition.dispatch("exact_math", question)
    print(json.dumps({
        "status": result.status.value,
        "component": result.component,
        "output": result.output,
        "reason": result.reason,
        "evidence": dict(result.evidence),
    }, ensure_ascii=False, sort_keys=True))
    return 0 if result.ok else 2


def _chat_qwen38(args: argparse.Namespace) -> int:
    """Run one turn or a persistent JSONL stream through local Qwen3.8."""

    from .runtimes.qwen3_8.adapter import Qwen38CausalChat
    from .runtimes.qwen3_8.draft_window import DraftWindowError
    from .runtimes.qwen3_8.semantic_state_cache import SemanticStateAnchorCache

    component = None
    jsonl = bool(getattr(args, "jsonl", False))
    message = getattr(args, "message", None)
    max_requests = getattr(args, "max_requests", None)

    def emit(result, *, request_id=None, include_id: bool = False) -> None:
        value = {
            "status": result.status.value,
            "component": result.component,
            "output": result.output,
            "reason": result.reason,
            "evidence": dict(result.evidence),
        }
        if include_id:
            value["id"] = request_id
        print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)

    def emit_line_error(reason: str, *, request_id=None, include_id=False) -> None:
        value = {
            "status": "error",
            "component": "qwen3.8.causal-chat",
            "output": None,
            "reason": reason,
            "evidence": {},
        }
        if include_id:
            value["id"] = request_id
        print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)

    try:
        if jsonl:
            if message is not None:
                raise ValueError("chat message and --jsonl are mutually exclusive")
            if (
                max_requests is not None
                and (
                    isinstance(max_requests, bool)
                    or not isinstance(max_requests, int)
                    or max_requests <= 0
                )
            ):
                raise ValueError("max_requests must be a positive integer")
        elif not isinstance(message, str) or not message.strip():
            raise ValueError("chat requires a message or --jsonl")
        anchor_cache = (
            None
            if args.qwen38_anchor_cache is None
            else SemanticStateAnchorCache(args.qwen38_anchor_cache)
        )
        component = Qwen38CausalChat(
            args.qwen38_causal_bundle,
            args.qwen38_tokenizer,
            system_prompt=args.system_prompt,
            device=args.device,
            compute_dtype=args.compute_dtype,
            source_budget_mb=args.source_budget_mb,
            max_resident_bytes=int(args.max_resident_mb * 1024**2),
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            max_context_tokens=args.max_context_tokens,
            head_block_rows=args.head_block_rows,
            exact_head_root=args.exact_head,
            exact_head_max_bytes=int(args.exact_head_max_mb * 1024**2),
            q4_root=args.qwen38_q4,
            q4_threads=args.q4_threads,
            anchor_cache=anchor_cache,
            draft_bundle_path=args.draft_bundle,
            draft_mode=args.draft_mode,
            draft_window=args.draft_window,
            draft_source_budget_mb=args.draft_source_budget_mb,
            draft_max_resident_bytes=(
                None
                if args.draft_max_resident_mb is None
                else int(args.draft_max_resident_mb * 1024**2)
            ),
            markov_draft_state_path=args.markov_draft_state,
            draft_window_state_path=args.draft_window_state,
            range_markov_state_path=args.range_markov_state,
            range_prefetch_max_bytes=int(args.range_prefetch_max_mb * 1024**2),
            range_prefetch_min_support=args.range_prefetch_min_support,
            range_prefetch_min_confidence=args.range_prefetch_min_confidence,
            range_prefetch_beam_horizon=args.range_prefetch_beam_horizon,
            range_prefetch_beam_width=args.range_prefetch_beam_width,
            range_prefetch_hint_cooldown=args.range_prefetch_hint_cooldown,
            fast_mlp_root=args.fast_mlp,
            fast_mlp_online_state_path=args.fast_mlp_online_state,
            fast_mlp_source_budget_mb=args.fast_mlp_source_budget_mb,
            fast_mlp_max_resident_bytes=(
                None
                if args.fast_mlp_max_resident_mb is None
                else int(args.fast_mlp_max_resident_mb * 1024**2)
            ),
            fast_mlp_active_layers=args.fast_mlp_layers,
        )
        if jsonl:
            failures = 0
            handled = 0
            for raw in sys.stdin:
                line = raw.strip()
                if not line:
                    continue
                if max_requests is not None and handled >= max_requests:
                    break
                handled += 1
                request_id = None
                include_id = False
                try:
                    if line.startswith("{"):
                        document = json.loads(line)
                        if not isinstance(document, dict) or set(document) - {
                            "id",
                            "message",
                        }:
                            raise ValueError(
                                "JSONL request must contain only id/message"
                            )
                        include_id = "id" in document
                        request_id = document.get("id")
                        line_message = document.get("message")
                    else:
                        line_message = line
                    if not isinstance(line_message, str) or not line_message.strip():
                        raise ValueError("JSONL request message must be non-empty text")
                    result = component.handle(Request("chat", line_message))
                    emit(
                        result,
                        request_id=request_id,
                        include_id=include_id,
                    )
                    failures += int(not result.ok)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    failures += 1
                    emit_line_error(
                        f"{type(exc).__name__}: {exc}",
                        request_id=request_id,
                        include_id=include_id,
                    )
                if max_requests is not None and handled >= max_requests:
                    break
            return 0 if failures == 0 else 2
        result = component.handle(Request("chat", message))
    except (DraftWindowError, OSError, TypeError, ValueError) as exc:
        print(json.dumps({
            "status": "error",
            "component": "qwen3.8.fertig-chat",
            "reason": f"{type(exc).__name__}: {exc}",
        }, ensure_ascii=False, sort_keys=True))
        return 2
    finally:
        close = getattr(component, "close", None)
        if callable(close):
            close()
    emit(result)
    return 0 if result.ok else 2


def _doctor(*, deep: bool = False, artifact_root: str | Path | None = None) -> int:
    from .runtimes.o1_state.adapter import is_available as o1state_available

    fertig_root = os.environ.get("IMMER_FERTIG_ROOT")
    vendor_solver = Path(__file__).parent / "cognition" / "fertig" / "_vendor" / "fertig" / "solver.py"
    configured_solver = (
        Path(fertig_root).expanduser() / "fertig" / "solver.py"
        if fertig_root
        else None
    )
    solver_ready = vendor_solver.is_file() or bool(
        configured_solver is not None and configured_solver.is_file()
    )
    graph_setting = os.environ.get("IMMER_FERTIG_GRAPH")
    graph_ready = bool(graph_setting and Path(graph_setting).expanduser().is_file())
    manifest = _s3_manifest()
    selected_artifact_root = _artifact_root(manifest, artifact_root)
    organ_ready = False
    organ_detail = str(manifest)
    try:
        from .capabilities.organbank import OrganBank

        bank = OrganBank.from_manifest(manifest, artifact_root=selected_artifact_root)
        bank.verify_all()
        organ_ready = len(bank.names()) == 4
        organ_detail = f"{len(bank.names())} SHA-geprüfte Organe"
    except (FileNotFoundError, KeyError, ValueError) as exc:
        organ_detail = f"{exc}; run 'immer artifacts import SOURCE'"
    checks = [
        ("FERTIG-solv", solver_ready, "vendored; override via IMMER_FERTIG_ROOT", True),
        (
            "FERTIG-graph",
            graph_ready,
            str(Path(graph_setting).expanduser()) if graph_setting else "optional; set IMMER_FERTIG_GRAPH",
            False,
        ),
        ("Action-gates", True, "desktop/recorder/mutations require explicit backends", True),
        ("o1-state", o1state_available(), "pip install -e '.[neural]'", True),
        ("SHIP-v6", organ_ready, organ_detail, True),
    ]
    try:
        import torch  # noqa: F401
        crsa = True
    except ImportError:
        crsa = False
    checks.append(("CRSA/Torch", crsa, "pip install -e '.[neural]'", True))
    try:
        import numpy  # noqa: F401
        from .knowledge import Streamer  # noqa: F401

        world_stream = True
    except ImportError:
        world_stream = False
    checks.append(("WorldStream", world_stream, "pip install -e .", True))
    for name, ready, hint, _required in checks:
        print(f"{'✓' if ready else '·'} {name:12} {hint}")
    if deep and organ_ready and crsa:
        from .capabilities.s3_runtime import S3Arithmetic

        benchmark = S3Arithmetic(
            manifest,
            artifact_root=selected_artifact_root,
        ).benchmark()
        passed = (
            benchmark["correct"] == benchmark["cases"]
            and benchmark["route_correct"] == benchmark["cases"]
        )
        print(
            f"{'✓' if passed else '·'} SHIP-eval    "
            f"{benchmark['correct']}/{benchmark['cases']} korrekt; "
            f"Route {benchmark['route_correct']}/{benchmark['cases']}; "
            f"{benchmark['runtime_s']:.3f}s"
        )
        return 0 if passed else 1
    return 0 if all(ready for _, ready, _, required in checks if required) else 1


def _organs(args: argparse.Namespace) -> int:
    from .capabilities.organbank import OrganBank

    manifest = _s3_manifest(args.manifest)
    bank = OrganBank.from_manifest(
        manifest,
        artifact_root=_artifact_root(manifest, args.artifact_root),
    )
    if args.organ_command == "list":
        for name in bank.names():
            descriptor = bank.descriptor(name)
            print(f"{name:24} {descriptor.capability:16} {descriptor.group}")
        return 0
    artifact = bank.verify(args.name)
    print(json.dumps({"organ": args.name, "verified": True, "artifact": str(artifact)}, ensure_ascii=False))
    return 0


def _artifacts(args: argparse.Namespace) -> int:
    from .artifacts import ArtifactBootstrapError, import_artifacts, load_artifact_specs, sha256_file

    manifest = _s3_manifest(args.manifest)
    if args.artifact_command == "verify":
        try:
            _, specs = load_artifact_specs(
                manifest,
                configured_root=args.artifact_root,
            )
            rows = []
            for spec in specs:
                actual = sha256_file(spec.destination) if spec.destination.is_file() else None
                rows.append(
                    {
                        "label": spec.label,
                        "path": str(spec.destination),
                        "present": actual is not None,
                        "verified": actual == spec.sha256,
                        "sha256": actual,
                    }
                )
            passed = all(row["verified"] for row in rows)
            print(json.dumps({"status": "ok" if passed else "missing", "artifacts": rows}, sort_keys=True))
            return 0 if passed else 1
        except ArtifactBootstrapError as exc:
            print(json.dumps({"status": "error", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
            return 2
    try:
        report = import_artifacts(
            manifest,
            args.source,
            dry_run=args.dry_run,
            replace=args.replace,
            configured_root=args.artifact_root,
        )
    except ArtifactBootstrapError as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def _eval_ship(
    manifest: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> int:
    from .capabilities.s3_runtime import S3Arithmetic

    report = dict(
        S3Arithmetic(
            _s3_manifest(manifest),
            artifact_root=artifact_root,
        ).benchmark()
    )
    report["passed"] = (
        report["correct"] == report["cases"]
        and report["route_correct"] == report["cases"]
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


def _export_hf(args: argparse.Namespace) -> int:
    """Build the final offline bundle; never authenticate or upload."""

    from .hf_export import HfExportError, export_hf_poc

    try:
        report = export_hf_poc(
            args.output,
            manifest=_s3_manifest(args.manifest),
            artifact_root=args.artifact_root,
            replace=args.replace,
        )
    except HfExportError as exc:
        print(
            json.dumps({"status": "error", "reason": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def _stream(args: argparse.Namespace) -> int:
    """Inspect or read exact rows without constructing the donor model."""

    from .knowledge import Streamer, TensorSourceError

    try:
        source = (
            Streamer.from_local(
                args.source,
                revision=args.revision,
                budget_mb=args.budget_mb,
                cache_dir=args.cache_dir,
                use_cache=not args.no_cache,
            )
            if args.local
            else Streamer(
                args.source,
                revision=args.revision,
                budget_mb=args.budget_mb,
                cache_dir=args.cache_dir,
                use_cache=not args.no_cache,
            )
        )
        inventory = source.inventory(refresh=args.refresh)
        tensors = source.tensors()
        if args.tensor is None:
            payload = {
                "status": "ok",
                "mode": "inventory",
                "source": source.repo_id,
                "revision": source.revision,
                "tensor_count": len(tensors),
                "shard_count": len(inventory.get("shards", ())),
                "tensors": [entry["name"] for entry in tensors[: args.limit]],
                "truncated": len(tensors) > args.limit,
                "metrics": source.metrics(),
            }
        else:
            rows = source.rows(args.tensor, start_row=args.start_row, n_rows=args.rows)
            raw = rows.tobytes(order="C")
            preview_rows = min(2, int(rows.shape[0]))
            preview_cols = min(8, int(rows.shape[1]))
            payload = {
                "status": "ok",
                "mode": "rows",
                "source": source.repo_id,
                "revision": source.revision,
                "tensor": args.tensor,
                "start_row": args.start_row,
                "shape": list(rows.shape),
                "dtype": str(rows.dtype),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "preview": rows[:preview_rows, :preview_cols].tolist(),
                "metrics": source.metrics(),
            }
    except (FileNotFoundError, KeyError, TensorSourceError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "error", "error": type(exc).__name__, "reason": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _serve(args: argparse.Namespace) -> int:
    """The organism lives in this terminal: it learns, remembers and answers."""
    from .intent import classify
    from .library import Library
    from .memory import SpanStore
    from .runtimes.donor.adapter import DonorBrain
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

    qwen = QwenBrain() if args.local_brain else None
    donor = DonorBrain()
    council = None
    if args.council and donor.available():
        from .runtimes.donor.adapter import build_council

        council = build_council()
    elif args.council and qwen is not None and qwen.model_id is not None:
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
    harvester = donor if donor.available() else qwen
    library = Library(SpanStore(state.parent / "memory.json"), harvester=harvester)
    metrics = Metrics(status_path=state.parent / "status.json", jsonl_path=state.parent / "metrics.jsonl")

    bank = None
    manifest = _s3_manifest(args.manifest)
    if manifest.is_file():
        from .capabilities.organbank import OrganBank

        bank = OrganBank.from_manifest(
            manifest,
            artifact_root=_artifact_root(manifest, args.artifact_root),
        )
    from .composition import CompositionRoot

    composition = CompositionRoot.build(
        life_stream=stream,
        s3_manifest=manifest,
        s3_artifact_root=args.artifact_root,
        fertig_state_dir=state.parent / "fertig",
        fertig_graph=os.environ.get("IMMER_FERTIG_GRAPH"),
    )
    daemon = LifeDaemon(stream=stream, bank=bank, state_path=args.state)
    daemon.register(composition.exact_math)
    if composition.grounded_chat is not None:
        daemon.register(composition.grounded_chat)

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
            "donor": donor.available(),
            "donor_modell": donor.model_id,
            "lokal_gehirn": qwen.loaded if qwen is not None else False,
            "spans": library.store.count(),
        }

    port = None if args.no_dashboard else (args.dashboard or 8787)
    server = None
    if port:
        try:
            server = start_dashboard(metrics, gauges, port)
            print(f"◆ dashboard: http://127.0.0.1:{port}/  (Maschinenfutter: /status)")
        except OSError as exc:
            print(f"· dashboard aus: {exc}", file=sys.stderr)

    available_organs = bank.names() if bank is not None else ()
    print(
        f"immer serve — turns alive: {daemon.turns}; cold bank: "
        f"{', '.join(available_organs) or 'none'}"
    )
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
            if not hits and harvester is not None:
                metrics.bump("harvests")
                card = library.grow(intent.payload)
                if card is not None:
                    print(f"[ernte] {card['text']}")
                    print(f"         (quelle: {card['source']})")
                    hits = [card]
            if hits:
                for span in hits:
                    print(f"[erinnerung] {span['text']} ({span.get('source', '?')})")
            else:
                print("[erinnerung] da weiß ich noch nichts — lehr mich (merke: …)")
        elif intent.kind == "MATH":
            import time as _time

            exact_started = _time.perf_counter()
            answer = daemon.request("exact_math", intent.payload)
            exact_ms = int((_time.perf_counter() - exact_started) * 1000)
            if answer.ok:
                metrics.bump("math_ok")
                metrics.emit_gauge("last_exact_latency_ms", exact_ms)
                route = answer.evidence.get("route", answer.component)
                print(f"[exact] {answer.output}  ({exact_ms} ms, {route})")
            else:
                metrics.bump("math_abstained")
                # Abstinenz heißt nicht Endstation: gleiche Kaskade wie Chat
                _answer_via_cascade(
                    intent.payload, library, council, donor, qwen, daemon, history,
                    metrics, stream,
                )
        elif intent.kind == "STATUS":
            metrics.bump("status")
            print(json.dumps(metrics.snapshot(gauges()), ensure_ascii=False))
        else:  # CHAT — Stufen: Bibliothek (0 ms) → Entwurf → Donor → Veredelung im Hintergrund
            _answer_via_cascade(
                intent.payload, library, council, donor, qwen, daemon, history,
                metrics, stream,
            )
            metrics.bump("chat")

        metrics.emit(gauges())
        life_line = f"turns: {daemon.turns}"
        if stream is not None:
            life_line += f", tokens: {stream.tokens}, updates: {stream.updates}, überraschungen: {stream.surprises}"
        print(f"({life_line})")
    if server is not None:
        server.shutdown()
        server.server_close()
    return 0


def _answer_via_cascade(
    payload: str,
    library,
    council,
    donor,
    qwen,
    daemon,
    history: list[dict[str, str]],
    metrics,
    stream,
) -> None:
    """Stufen des Mundwerks, nach Latenz sortiert — gemessen, nicht behauptet.

    Tier 0  Bibliothek   ~0 ms      (die Karte existiert schon)
    Tier 1  FERTIG       grounded   (Graph, Skills, Hilfe; sichere Aktions-Gates)
    Tier 2  Donor        ~19 s      (SOTA-Orakel, sync wenn nichts Besseres da)
    Tier 3  Veredelung   im Hintergrund (Donor verbessert den Entwurf nachträglich)
    """
    import threading
    import time as _time

    meta = {
        "history": history[-6:],
        "life": f"turns={daemon.turns} tokens={getattr(stream, 'tokens', 0)}",
    }

    # ---- Tier 0: die Bibliothek ist schneller als jedes Modell -------------
    t0 = _time.perf_counter()
    hits = library.recall(payload)
    if hits:
        latency = int((_time.perf_counter() - t0) * 1000)
        metrics.bump("tier_bibliothek")
        metrics.emit_gauge("last_latency_ms", latency)
        print(f"[bibliothek] {hits[0]['text']}  ({latency} ms, quelle: {hits[0].get('source', '?')})")
        history.append({"role": "user", "content": payload})
        history.append({"role": "assistant", "content": hits[0]["text"]})
        return

    # ---- Tier 1: FERTIGs geerdeter Graph-/Skill-Pfad ----------------------
    grounded = daemon.request("grounded_chat", payload, metadata=meta)
    if grounded.ok:
        route = str(grounded.evidence.get("route", "grounded"))
        if route == "math":
            # FERTIGs natürliche Chat-Oberfläche darf den Guard des exakten
            # Pfads nicht umgehen. Nur eine bestätigte Zahl wird gesprochen.
            verified = daemon.request("exact_math", payload, metadata=meta)
            claimed = grounded.evidence.get("data", {}).get("answer")
            if not verified.ok or (
                claimed is not None
                and str(claimed).strip().casefold()
                != str(verified.output).strip().casefold()
            ):
                grounded = None
        if grounded is not None:
            metrics.bump("tier_fertig")
            print(f"[fertig:{route}] {grounded.output}")
            library.capture(payload, grounded.output)
            history.append({"role": "user", "content": payload})
            history.append({"role": "assistant", "content": str(grounded.output)})
            return
    elif (
        grounded.evidence.get("status") in {"needs_input", "error"}
        or grounded.evidence.get("route") == "desktop"
    ):
        # Eine fehlende explizite Desktop-/Recorder-Freigabe ist eine harte
        # Grenze; ein Sprachmodell darf daraus keine scheinbare Aktion machen.
        print(f"[fertig] ({grounded.reason})")
        return

    # ---- Tier 2: der Donor antwortet synchron und die Karte wird sofort geschrieben
    if council is not None:
        result = council.deliberate("chat", payload, metadata=meta)
        tag = "[rat]"
    elif donor.available():
        t1 = _time.perf_counter()
        result = donor.handle(Request("chat", payload, metadata=meta))
        latency = int((_time.perf_counter() - t1) * 1000)
        metrics.emit_gauge("last_latency_ms", latency)
        tag = f"[donor {latency / 1000:.1f}s]"
    elif qwen is not None:
        t1 = _time.perf_counter()
        result = qwen.handle(Request("chat", payload, metadata=meta))
        latency = int((_time.perf_counter() - t1) * 1000)
        metrics.emit_gauge("last_latency_ms", latency)
        tag = f"[entwurf {latency / 1000:.1f}s]"
        # Tier 3: der Donor veredelt den Entwurf, ohne zu blockieren
        if donor.available() and result.ok:

            def _refine() -> None:
                card = library.grow(payload)
                if card is not None:
                    metrics.bump("refined")

            threading.Thread(target=_refine, daemon=True).start()
            print("         (donor veredelt im hintergrund…)")
    else:
        result = None

    if result is not None and result.ok:
        print(f"{tag} {result.output}")
        if council is not None and result.evidence.get("votes"):
            print(
                f"         (stimmen: {result.evidence['votes']}, "
                f"übereinstimmend: {result.evidence['agree']})"
            )
        # Jede gesprochene Antwort wird Karte — die Wiederholung ist gratis.
        library.capture(payload, result.output)
        history.append({"role": "user", "content": payload})
        history.append({"role": "assistant", "content": result.output})
    else:
        reason = result.reason if result is not None else "kein Mund konfiguriert (--local-brain?)"
        print(f"[stille] ({reason})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="immer")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("components", help="show the system component map")
    doctor = sub.add_parser("doctor", help="inspect runtime integrations")
    doctor.add_argument("--deep", action="store_true", help="also run the cold 152-case SHIP eval")
    doctor.add_argument("--artifact-root", help="external SHIP artifact directory")
    solve = sub.add_parser("solve", help="run the guarded S3 + FERTIG exact cascade")
    solve.add_argument("question")
    solve.add_argument("--manifest", help="SHIP-v6 manifest (default: bundled manifest)")
    solve.add_argument("--artifact-root", help="external SHIP artifact directory")
    chat = sub.add_parser(
        "chat",
        help="run one greedy turn through a verified local Qwen3.8 causal bundle",
    )
    chat.add_argument("message", nargs="?")
    chat.add_argument(
        "--jsonl",
        action="store_true",
        help="keep one loaded runtime and process stdin as raw-text or JSONL requests",
    )
    chat.add_argument(
        "--max-requests",
        type=int,
        help="stop the persistent JSONL loop after this many non-empty lines",
    )
    chat.add_argument("--qwen38-causal-bundle", required=True)
    chat.add_argument("--qwen38-tokenizer", required=True)
    chat.add_argument(
        "--qwen38-q4",
        metavar="BANK",
        help="causal-bound mmap Q4/Q8 execution bank for the local Qwen bundle",
    )
    chat.add_argument(
        "--q4-threads",
        type=int,
        help="CPU worker count for the native Q4/Q8 kernel (default: up to 16)",
    )
    chat.add_argument(
        "--draft-bundle",
        help="optional local causal Qwen3.5-0.8B bundle for rolling drafting",
    )
    chat.add_argument(
        "--draft-mode",
        choices=("qwen35", "markov"),
        help="rolling draft provider; inferred as qwen35 when --draft-bundle is set",
    )
    chat.add_argument(
        "--draft-window",
        type=int,
        choices=range(2, 17),
        default=8,
        metavar="K",
        help="target-verified tokens per weight pass (2-16; default: 8)",
    )
    chat.add_argument(
        "--markov-draft-state",
        help="persistent sparse Qwen-token Markov memory",
    )
    chat.add_argument(
        "--draft-window-state",
        help=(
            "persistent target-receipt controller for contextual K=4/8/16; "
            "--draft-window becomes its maximum ceiling"
        ),
    )
    chat.add_argument(
        "--range-markov-state",
        default=os.environ.get("IMMER_QWEN38_RANGE_MARKOV_STATE"),
        help="persistent operation-Markov state for local weight-range prefetch",
    )
    chat.add_argument(
        "--range-prefetch-max-mb",
        type=float,
        default=64.0,
        help="maximum OS-cache hint bytes after one predicted operation",
    )
    chat.add_argument(
        "--range-prefetch-min-support",
        type=int,
        default=2,
        help="minimum learned transition support before a range hint",
    )
    chat.add_argument(
        "--range-prefetch-min-confidence",
        type=float,
        default=0.65,
        help="minimum next-operation probability before a range hint",
    )
    chat.add_argument(
        "--range-prefetch-beam-horizon",
        type=int,
        default=3,
        help="future operation depth for the bounded range beam",
    )
    chat.add_argument(
        "--range-prefetch-beam-width",
        type=int,
        default=4,
        help="maximum hypothetical paths retained at each range depth",
    )
    chat.add_argument(
        "--range-prefetch-hint-cooldown",
        type=int,
        default=2,
        help="operations before an identical exact range can be hinted again",
    )
    chat.add_argument(
        "--fast-mlp",
        metavar="ARTIFACT_ROOT",
        help="mount the existing row-routed sparse MLP banks for fast chat",
    )
    chat.add_argument(
        "--fast-mlp-layers",
        type=_sorted_layer_list,
        help="sorted fitted layer subset, for example 0,9,18,27,36,45,54,63",
    )
    chat.add_argument(
        "--fast-mlp-online-state",
        help="persistent target-confirmed state for a weight-only Fast-MLP bank",
    )
    chat.add_argument(
        "--qwen38-anchor-cache",
        default=os.environ.get("IMMER_QWEN38_ANCHOR_CACHE"),
        help="local authenticated semantic anchor cache",
    )
    chat.add_argument("--system-prompt", default="")
    chat.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    chat.add_argument(
        "--compute-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    chat.add_argument("--source-budget-mb", type=float, default=4194304)
    chat.add_argument("--draft-source-budget-mb", type=float, default=1048576)
    chat.add_argument("--fast-mlp-source-budget-mb", type=float)
    chat.add_argument("--max-resident-mb", type=int, default=192)
    chat.add_argument("--draft-max-resident-mb", type=int, default=64)
    chat.add_argument("--fast-mlp-max-resident-mb", type=int)
    chat.add_argument("--max-prompt-tokens", type=int, default=1024)
    chat.add_argument("--max-new-tokens", type=int, default=64)
    chat.add_argument("--max-context-tokens", type=int, default=2048)
    chat.add_argument("--head-block-rows", type=int, default=2048)
    chat.add_argument(
        "--exact-head",
        help="local residual-certified exact LM-head index directory",
    )
    chat.add_argument("--exact-head-max-mb", type=int, default=128)
    organs = sub.add_parser("organs", help="inspect the cold organ bank")
    organs.add_argument("--manifest", help="SHIP-v6/OrganBank manifest")
    organs.add_argument("--artifact-root", help="external SHIP artifact directory")
    organs_sub = organs.add_subparsers(dest="organ_command", required=True)
    organs_sub.add_parser("list", help="list registered organs")
    mount = organs_sub.add_parser("mount", help="verify one organ's digest")
    mount.add_argument("name")
    artifacts = sub.add_parser("artifacts", help="verify or bootstrap external SHIP artifacts")
    artifacts.add_argument("--manifest", help="SHIP-v6 manifest (default: bundled manifest)")
    artifacts.add_argument("--artifact-root", help="destination directory for the five blobs")
    artifact_sub = artifacts.add_subparsers(dest="artifact_command", required=True)
    artifact_sub.add_parser("verify", help="verify all five host/organ blobs")
    artifact_import = artifact_sub.add_parser(
        "import",
        help="copy SHA-matching blobs from an explicit read-only source directory",
    )
    artifact_import.add_argument("source")
    artifact_import.add_argument("--dry-run", action="store_true")
    artifact_import.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing wrong target only after a matching source is found",
    )
    evaluate = sub.add_parser("eval", help="run the cold, training-free SHIP-v6 proof suite")
    evaluate.add_argument("--manifest", help="SHIP-v6 manifest (default: bundled manifest)")
    evaluate.add_argument("--artifact-root", help="external SHIP artifact directory")
    export_hf = sub.add_parser(
        "export-hf",
        help="build and offline-verify the final license-gated HF PoC folder",
    )
    export_hf.add_argument("output", help="new export directory")
    export_hf.add_argument("--manifest", help="SHIP-v6 manifest (default: bundled manifest)")
    export_hf.add_argument("--artifact-root", help="external SHIP artifact directory")
    export_hf.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing export while preserving it as a backup",
    )
    stream = sub.add_parser(
        "stream",
        help="inspect safetensors or read exact rows under a hard byte budget",
    )
    stream.add_argument("source", help="HF repo id, or a directory together with --local")
    stream.add_argument("--local", action="store_true", help="treat source as an offline directory")
    stream.add_argument("--revision", default="main", help="source revision; pin a commit for remote use")
    stream.add_argument("--budget-mb", type=float, default=200.0, help="hard total transfer ceiling")
    stream.add_argument("--cache-dir", help="verified resume-cache directory")
    stream.add_argument("--no-cache", action="store_true", help="disable inventory/range resume caches")
    stream.add_argument("--refresh", action="store_true", help="bypass cached inventory")
    stream.add_argument("--limit", type=int, default=20, help="maximum tensor names in inventory output")
    stream.add_argument("--tensor", help="read this exact 2D tensor instead of listing the inventory")
    stream.add_argument("--start-row", type=int, default=0)
    stream.add_argument("--rows", type=int, default=8)
    serve = sub.add_parser("serve", help="let the organism live in this terminal")
    serve.add_argument("--state", default="~/.immer/life.json")
    serve.add_argument("--manifest", help="SHIP-v6 manifest (default: bundled manifest)")
    serve.add_argument("--artifact-root", help="external SHIP artifact directory")
    serve.add_argument(
        "--dashboard", nargs="?", const=8787, type=int, default=8787,
        help="metrics UI port (default 8787)",
    )
    serve.add_argument("--no-dashboard", action="store_true", help="disable the metrics UI")
    serve.add_argument("--council", action="store_true", help="chat through the rat (donor council)")
    serve.add_argument("--local-brain", action="store_true", help="allow the local Qwen as fallback mouth")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "components":
        return _components()
    if args.command == "doctor":
        return _doctor(deep=args.deep, artifact_root=args.artifact_root)
    if args.command == "solve":
        return _solve(args.question, args.manifest, args.artifact_root)
    if args.command == "chat":
        return _chat_qwen38(args)
    if args.command == "organs":
        return _organs(args)
    if args.command == "artifacts":
        return _artifacts(args)
    if args.command == "eval":
        return _eval_ship(args.manifest, args.artifact_root)
    if args.command == "export-hf":
        return _export_hf(args)
    if args.command == "stream":
        if args.limit < 0:
            raise SystemExit("--limit must be nonnegative")
        return _stream(args)
    if args.command == "serve":
        return _serve(args)
    raise AssertionError(args.command)
