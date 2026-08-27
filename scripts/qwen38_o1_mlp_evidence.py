#!/usr/bin/env python3
"""Capture, replay-verify, and evaluate exact local Qwen MLP evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import stat
import sys
import time
from typing import Sequence

import numpy as np

from immer.runtimes.ooe.qwen_mlp_evidence import (
    CAPTURE_STAGES,
    CaptureManifest,
    CapturePlanEntry,
    ExactMlpBoundaryCapture,
    MlpEvidenceBudget,
    MlpEvidenceReceipt,
    MlpProjectionVerificationReceipt,
    QwenMlpEvidenceBank,
    canonical_capture_plan,
    run_capture_manifest,
)
from immer.runtimes.ooe.qwen_mlp_live import (
    LiveExactMlpCaptureRunner,
    LiveMlpAuthority,
    MlpCalibrationLock,
)
from immer.runtimes.ooe.operator_harvester import (
    HarvesterConfig,
    HarvesterState,
    QWEN_CONTEXT_EMITTER_SHA256,
    harvester_identity_sha256,
    harvester_state_name,
)
from immer.runtimes.ooe.subspace_battery import (
    SubspaceBatteryFit,
    SubspaceCorpus,
    SubspaceSweepConfig,
    evaluate_subspace_battery,
    fit_subspace_battery,
    graph_revision_sha256,
)
from immer.runtimes.qwen3_8.semantic_atlas import ModelPin, SemanticWeightAtlas


STATUS_SCHEMA = "immer.qwen3.8-mlp-evidence-status/v1"
FIXTURE_SCHEMA = "immer.qwen3.8-mlp-evidence-fixture/v1"
LIVE_ANALYSIS_SCHEMA = "immer.qwen3.8-mlp-live-analysis/v1"
CAPTURE_MANIFEST_NAME = "capture-manifest.json"
CALIBRATION_LOCK_NAME = "calibration-lock.json"
PREFIX_FIT_NAME = "prefix-fit.json"
FULL_FIT_NAME = "full-fit.json"
HOLDOUT_NAME = "holdout.json"


class CliError(RuntimeError):
    pass


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_bytes(path: Path, *, maximum: int = 128 * 1024 * 1024) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise CliError(f"invalid bounded file: {path}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise CliError(f"file exceeds its bound: {path}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise CliError(f"file changed while read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _persist_exact(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or _stable_bytes(path, maximum=max(len(data), 1)) != data:
            raise CliError(f"sealed artifact changed: {path}")
        return
    temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError:
        if _stable_bytes(path, maximum=max(len(data), 1)) != data:
            raise CliError(f"sealed artifact collided: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _cartography_module():
    name = "qwen38_o1_cartography_live_mlp"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    source = Path(__file__).resolve().with_name("qwen38_o1_cartography.py")
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise CliError("cannot import the Qwen O1 cartography authority")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _harvester_config() -> HarvesterConfig:
    return HarvesterConfig(
        minimum_observations=3,
        minimum_fit_rows=4,
        max_observations_per_step=128,
        max_samples_per_group=1024,
        max_groups=4096,
        max_recent_receipts=65_536,
        graph_cas_retries=16,
    )


class FixtureRunner:
    capture_mode = "fixture"
    verifier_sha256 = _hash("fixture-verifier-not-live")

    def __init__(self, manifest: CaptureManifest, fixture: dict[str, object]) -> None:
        if fixture.get("schema") != FIXTURE_SCHEMA or set(fixture) != {
            "hidden_dimension",
            "intermediate_dimension",
            "rows",
            "schema",
            "seed_sha256",
        }:
            raise CliError("fixture schema is invalid")
        for field in ("rows", "hidden_dimension", "intermediate_dimension"):
            value = fixture.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CliError(f"fixture {field} must be positive")
        seed = fixture.get("seed_sha256")
        if not isinstance(seed, str) or len(seed) != 64:
            raise CliError("fixture seed_sha256 is invalid")
        self.manifest = manifest
        self.rows = int(fixture["rows"])
        self.hidden = int(fixture["hidden_dimension"])
        self.intermediate = int(fixture["intermediate_dimension"])
        self.seed = seed

    def capture(
        self, entry: CapturePlanEntry, sink: ExactMlpBoundaryCapture
    ) -> MlpEvidenceReceipt:
        generator = np.random.default_rng(
            int(
                hashlib.sha256(f"{self.seed}:{entry.sha256}".encode()).hexdigest()[:16],
                16,
            )
        )
        tensors = {
            "mlp.input": generator.normal(size=(1, self.rows, self.hidden)).astype(
                np.float32
            ),
            "mlp.gate": generator.normal(size=(1, self.rows, self.intermediate)).astype(
                np.float32
            ),
            "mlp.up": generator.normal(size=(1, self.rows, self.intermediate)).astype(
                np.float32
            ),
            "mlp.output": generator.normal(size=(1, self.rows, self.hidden)).astype(
                np.float32
            ),
        }
        sink.begin_group(entry)
        for stage in CAPTURE_STAGES:
            sink(entry.layer, stage, tensors[stage])
        event = _hash(f"fixture-atlas-event:{entry.ordinal}")
        sketch = _hash(f"fixture-input-sketch:{entry.ordinal}")
        receipt = sink.finalize(
            capture_mode=self.capture_mode,
            manifest_sha256=self.manifest.sha256,
            model_pin_sha256=self.manifest.model_pin_sha256,
            token_sha256=_hash(f"fixture-token:{entry.prompt_sha256}"),
            probe_spec_sha256=_hash(f"fixture-probe-spec:{entry.ordinal}"),
            atlas_sequence=entry.ordinal,
            atlas_event_sha256=event,
            atlas_revision_sha256=graph_revision_sha256(entry.ordinal, event),
            measurement_sha256=_hash(f"fixture-measurement:{entry.ordinal}"),
            weight_revision_sha256=_hash("fixture-weight-revision"),
            access_trace_sha256=_hash(f"fixture-access:{entry.ordinal}"),
            source_receipt_sha256s=tuple(
                sorted(
                    (
                        _hash(f"fixture-gate-range:{entry.ordinal}"),
                        _hash(f"fixture-up-range:{entry.ordinal}"),
                        _hash(f"fixture-down-range:{entry.ordinal}"),
                    )
                )
            ),
            cartography_input_sketch_sha256=sketch,
            recomputed_input_sketch_sha256=sketch,
        )
        return receipt

    def verify(
        self, receipt: MlpEvidenceReceipt, bank: QwenMlpEvidenceBank
    ) -> MlpProjectionVerificationReceipt:
        if receipt.capture_mode != self.capture_mode:
            raise CliError("fixture verifier received non-fixture evidence")
        for ref in receipt.tensors:
            bank.restore_tensor(ref)
        replay_sha256 = _hash(f"fixture-replay:{receipt.sha256}")
        proof_sha256 = bank.publish_verifier_evidence(
            evidence_receipt_sha256=receipt.sha256,
            verifier_sha256=self.verifier_sha256,
            evidence={
                "gate_exact": True,
                "input_sketch_exact": True,
                "kind": "bounded-fixture-storage-proof",
                "output_exact": True,
                "replay_access_trace_sha256": replay_sha256,
                "storage_exact": True,
                "tensor_object_sha256s": [ref.object_sha256 for ref in receipt.tensors],
                "up_exact": True,
            },
        )
        return MlpProjectionVerificationReceipt(
            evidence_receipt_sha256=receipt.sha256,
            verifier_sha256=self.verifier_sha256,
            verifier_evidence_sha256=proof_sha256,
            replay_access_trace_sha256=replay_sha256,
            storage_exact=True,
            input_sketch_exact=True,
            gate_exact=True,
            up_exact=True,
            output_exact=True,
        )


def _load_live_context(
    cartography_root: Path,
    *,
    ooe_root: Path | None,
):
    cartography = _cartography_module()
    base_manifest, body, frontier_status = cartography._effective_manifest(
        cartography_root
    )
    prompt_rows = cartography._manifest_prompts(body)
    if len(prompt_rows) != 5:
        raise CliError("live MLP capture requires exactly five O1 prompts")
    prompt_sha256s = tuple(sorted(row["sha256"] for row in prompt_rows))
    capture_entries = canonical_capture_plan(prompt_sha256s)
    expected_keys = {(entry.prompt_sha256, entry.layer) for entry in capture_entries}
    prepared = cartography._manifest_jobs(body)
    expected_jobs = {job.job_id: job for job, _spec in prepared}
    spec_by_job = {job.job_id: spec for job, spec in prepared}
    scheduler_path = cartography_root / cartography.SCHEDULER_NAME
    scheduler_raw_before, scheduler_document = (
        cartography.O1Cartographer._read_document(scheduler_path)
    )
    if scheduler_document.get("in_flight") is not None:
        raise CliError("O1 scheduler has an in-flight attempt; retry after its commit")
    stream = cartography._stream(
        None,
        seed=body["scheduler"]["seed"],
        sidecar=cartography_root / cartography.O1_STATE_NAME,
    )
    scheduler = cartography.O1Cartographer.restore(
        scheduler_path,
        code_pin=body["code_revision"],
        model_pin=body["model_pin_sha256"],
        stream=stream,
    )
    restored_jobs = {job.job_id: job for job in scheduler.jobs}
    if restored_jobs != expected_jobs:
        raise CliError("O1 scheduler jobs differ from the effective frontier")
    selected: dict[tuple[str, int], tuple[object, object]] = {}
    for job_id, spec in spec_by_job.items():
        job = expected_jobs[job_id]
        layer = spec.coordinate.layer
        key = (spec.prompt_sha256, layer)
        base = f"model.language_model.layers.{layer}.input_layernorm"
        coordinate = spec.coordinate
        target = job.target
        if (
            key not in expected_keys
            or job.probe_family != "contextual.input-layernorm"
            or job.intervention != "passive"
        ):
            continue
        if not (
            spec.intervention_mode == "passive"
            and job.layer == layer
            and job.prompt_sha256 == spec.prompt_sha256
            and job.model_pin == body["model_pin_sha256"]
            and job.code_pin == body["code_revision"]
            and target.module == base
            and target.unit_kind == "module"
            and target.unit_index is None
            and spec.start_layer == layer
            and spec.stop_layer == layer + 1
            and coordinate.module == base
            and coordinate.tensor == f"{base}.weight"
            and coordinate.head_index is None
            and coordinate.row_start is None
            and coordinate.row_end is None
            and coordinate.byte_length is None
            and coordinate.relative_byte_offset == 0
            and spec.hidden_sketch is not None
            and spec.native_head_crsa is None
            and spec.prefix_sinkhorn_operator_capture is None
        ):
            raise CliError(f"canonical MLP authority spec changed for {key}")
        if key in selected:
            raise CliError(f"duplicate canonical MLP authority spec for {key}")
        selected[key] = (job, spec)
    if set(selected) != expected_keys:
        raise CliError("effective frontier lacks exact 40-cell MLP authority")
    projections = {spec.hidden_sketch for _job, spec in selected.values()}
    if len(projections) != 1:
        raise CliError("MLP authority cells do not share one hidden projection")
    projection = next(iter(projections))
    outcomes_by_job = {}
    for outcome in scheduler.outcomes:
        if (
            outcome.job_id in spec_by_job
            and outcome.status == "succeeded"
            and outcome.atlas_receipt_sha256 is not None
        ):
            outcomes_by_job.setdefault(outcome.job_id, []).append(outcome)
    for job, _spec in selected.values():
        if len(outcomes_by_job.get(job.job_id, ())) != 1:
            raise CliError("canonical MLP job lacks one successful Atlas outcome")

    runtime = cartography._open_runtime(body, None)
    try:
        atlas = cartography._open_atlas(
            cartography_root,
            body,
            runtime,
            create=False,
            atlas_factory=SemanticWeightAtlas,
        )
        if atlas is None:
            raise CliError("O1 semantic Atlas is absent")
        atlas.verify_or_raise()
        atlas_head_before = atlas.revision()
        model_pin = ModelPin.from_document(body["model_pin"])
        measurements = {}
        scheduler_observation_sha256s = {}
        for key, (job, spec) in selected.items():
            outcome = outcomes_by_job[job.job_id][0]
            observation = outcome.observation_document()
            if (
                observation is None
                or observation.get("probe_spec_sha256") != spec.sha256
                or observation.get("prompt_sha256") != spec.prompt_sha256
                or observation.get("intervention") != "passive"
            ):
                raise CliError("scheduler MLP observation differs from its spec")
            measurement = cartography._active_outcome_measurement(
                outcome,
                atlas=atlas,
                spec=spec,
                model_pin=model_pin,
                capture_bank=None,
            )
            measurements[key] = measurement
            scheduler_observation_sha256s[key] = hashlib.sha256(
                outcome.observation_bytes
            ).hexdigest()

        selected_ooe_root = (
            cartography_root / cartography.OOE_NAME if ooe_root is None else ooe_root
        )
        compute_root = selected_ooe_root / cartography.OPERATOR_COMPUTE_NAME
        if (
            not compute_root.exists()
            or compute_root.is_symlink()
            or not compute_root.is_dir()
        ):
            raise CliError("persistent operator-compute root is absent or invalid")
        config = _harvester_config()
        state_name = harvester_state_name(
            model_pin_sha256=model_pin.sha256,
            config=config,
            emitter_sha256=QWEN_CONTEXT_EMITTER_SHA256,
        )
        state_store = cartography.CrystalStore(compute_root)
        state_payload = state_store.restore_state(state_name)
        harvester_state = HarvesterState.from_bytes(state_payload)
        expected_identity = harvester_identity_sha256(
            model_pin_sha256=model_pin.sha256,
            config=config,
            emitter_sha256=QWEN_CONTEXT_EMITTER_SHA256,
        )
        if harvester_state.identity_sha256 != expected_identity:
            raise CliError("operator Harvester identity differs from O1 authority")
        observations = tuple(
            row for _group, rows in harvester_state.groups for row in rows
        )
        authorities = {}
        for key, (_job, spec) in selected.items():
            prompt_sha256, layer = key
            measurement = measurements[key]
            source = f"qwen.layer.{layer}.mlp.input-sketch"
            target = f"qwen.layer.{layer}.mlp.output-sketch"
            matches = tuple(
                row
                for row in observations
                if row.measurement.to_document() == measurement.to_document()
                and row.receipt.source_state == source
                and row.receipt.target_state == target
                and row.receipt.emitter_sha256 == QWEN_CONTEXT_EMITTER_SHA256
                and row.receipt.intervention_mode == "passive"
                and row.receipt.granularity == "operator"
                and row.receipt.segment_start is None
                and row.receipt.segment_end is None
                and row.receipt.model_pin_sha256 == model_pin.sha256
                and atlas.contains_revision(row.receipt.atlas_revision)
                and row.input_array.shape
                == (len(spec.prompt_token_ids), projection.output_dimensions)
                and row.output_array.shape
                == (len(spec.prompt_token_ids), projection.output_dimensions)
            )
            if len(matches) != 1:
                raise CliError(
                    f"Harvester lacks one exact MLP input authority for {key}"
                )
            authorities[key] = LiveMlpAuthority(
                layer=layer,
                prompt_sha256=prompt_sha256,
                probe_spec=spec,
                observation=matches[0],
                scheduler_observation_sha256=(scheduler_observation_sha256s[key]),
            )
        scheduler_raw_after, _scheduler_document_after = (
            cartography.O1Cartographer._read_document(scheduler_path)
        )
        if scheduler_raw_after != scheduler_raw_before:
            raise CliError("O1 scheduler changed during MLP authority load")
        if (
            atlas.revision() != atlas_head_before
            or atlas.revision_history()[-1] != atlas_head_before
        ):
            raise CliError("Atlas head/history changed during MLP authority load")
        if state_store.restore_state(state_name) != state_payload:
            raise CliError("Harvester state changed during MLP authority load")
        input_authority_sha256 = _digest(
            {
                "base_manifest_sha256": base_manifest["sha256"],
                "cells": [
                    {
                        "context_receipt_sha256": authorities[
                            key
                        ].observation.receipt.sha256,
                        "measurement_sha256": authorities[key].measurement.sha256,
                        "probe_spec_sha256": authorities[key].probe_spec.sha256,
                        "scheduler_observation_sha256": authorities[
                            key
                        ].scheduler_observation_sha256,
                    }
                    for key in sorted(authorities)
                ],
                "model_pin_sha256": model_pin.sha256,
                "schema": "immer.qwen3.8-mlp-live-input-authority/v1",
            }
        )
        capture_manifest = CaptureManifest(
            model_pin_sha256=model_pin.sha256,
            input_manifest_sha256=input_authority_sha256,
            prompt_sha256s=prompt_sha256s,
            entries=capture_entries,
        )
        return {
            "atlas": atlas,
            "authorities": authorities,
            "base_manifest_sha256": base_manifest["sha256"],
            "capture_manifest": capture_manifest,
            "frontier_head_sha256": frontier_status["head_sha256"],
            "harvester_state_sha256": harvester_state.sha256,
            "model": runtime.model,
            "projection": projection,
            "runtime": runtime,
            "scheduler_sha256": hashlib.sha256(scheduler_raw_before).hexdigest(),
        }
    except Exception:
        runtime.close()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--fixture",
        help="bounded CI fixture JSON; permanently non-production",
    )
    source.add_argument(
        "--cartography-root",
        help="authenticated local O1/Qwen cartography root for live exact capture",
    )
    parser.add_argument("--ooe-root")
    parser.add_argument("--max-total-mb", type=float, default=4096.0)
    parser.add_argument(
        "--max-new-groups",
        type=int,
        default=0,
        help="0 is unlimited; live capture completes the current five-prompt layer",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="fixture-only wall bound; live atomic capture uses --max-new-groups",
    )
    parser.add_argument("--analysis-max-working-gb", type=float, default=8.0)
    parser.add_argument("--capture-max-resident-gb", type=float, default=4.0)
    parser.add_argument(
        "--output",
        help="optional no-replace path for the canonical JSON status",
    )
    parser.add_argument(
        "--authority-only",
        action="store_true",
        help="authenticate the live 40-cell authority without opening an evidence bank",
    )
    return parser


def _maximum_bytes(value: object) -> int:
    maximum = float(value)
    if not np.isfinite(maximum) or maximum <= 0:
        raise CliError("--max-total-mb must be finite and positive")
    return int(maximum * 1024**2)


def _status(
    *,
    bank: QwenMlpEvidenceBank,
    manifest: CaptureManifest,
    fixture: bool,
    publications: int,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    state = bank.state()
    audit = bank.audit()
    return {
        "audit_clean": audit.clean,
        "budget_sha256": bank.budget.sha256,
        "fixture": fixture,
        "manifest_sha256": manifest.sha256,
        "new_publications": publications,
        "receipt_count": state.receipt_count,
        "referenced_tensor_bytes": state.referenced_tensor_bytes,
        "schema": STATUS_SCHEMA,
        "split_counts": list(state.split_counts),
        "state_sha256": state.sha256,
        **({} if extra is None else extra),
    }


def _run_fixture(args: argparse.Namespace) -> dict[str, object]:
    if args.authority_only:
        raise CliError("--authority-only requires --cartography-root")
    if args.manifest is None:
        raise CliError("--manifest is required with --fixture")
    manifest_path = Path(args.manifest).expanduser().absolute()
    fixture_path = Path(args.fixture).expanduser().absolute()
    manifest = CaptureManifest.from_bytes(_stable_bytes(manifest_path))
    try:
        fixture = json.loads(_stable_bytes(fixture_path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError("fixture is not JSON") from exc
    if not isinstance(fixture, dict):
        raise CliError("fixture must be an object")
    bank = QwenMlpEvidenceBank(
        Path(args.root).expanduser().absolute(),
        budget=MlpEvidenceBudget(
            max_total_referenced_bytes=_maximum_bytes(args.max_total_mb)
        ),
    )
    if args.max_new_groups < 0:
        raise CliError("--max-new-groups must be non-negative")
    fixture_seconds = float(args.max_seconds)
    if not np.isfinite(fixture_seconds) or fixture_seconds < 0.0:
        raise CliError("--max-seconds must be finite and non-negative")
    publications = run_capture_manifest(
        manifest,
        bank,
        FixtureRunner(manifest, fixture),
        max_new_groups=(None if args.max_new_groups == 0 else int(args.max_new_groups)),
        max_seconds=None if fixture_seconds == 0.0 else fixture_seconds,
    )
    return _status(
        bank=bank,
        manifest=manifest,
        fixture=True,
        publications=len(publications),
    )


def _prefix_corpus(corpus: SubspaceCorpus) -> SubspaceCorpus:
    if len(corpus.groups) != 40:
        raise CliError("full live corpus does not contain 40 groups")
    return SubspaceCorpus(corpus.model_pin_sha256, corpus.groups[:35])


def _run_live(args: argparse.Namespace) -> dict[str, object]:
    bank_root = Path(args.root).expanduser().absolute()
    cartography_root = Path(args.cartography_root).expanduser().absolute()
    selected_ooe_root = (
        None if args.ooe_root is None else Path(args.ooe_root).expanduser().absolute()
    )
    if args.max_new_groups < 0 or (args.max_new_groups and args.max_new_groups % 5):
        raise CliError("--max-new-groups must be zero or a multiple of five")
    if not np.isfinite(float(args.max_seconds)) or float(args.max_seconds) < 0.0:
        raise CliError("--max-seconds must be finite and non-negative")
    if args.max_seconds:
        raise CliError(
            "live capture cannot promise a hard wall bound across one Qwen forward; "
            "use --max-new-groups"
        )
    analysis_gb = float(args.analysis_max_working_gb)
    if not np.isfinite(analysis_gb) or analysis_gb <= 0.0:
        raise CliError("--analysis-max-working-gb must be finite and positive")
    analysis_config = SubspaceSweepConfig(max_working_bytes=int(analysis_gb * 1024**3))
    capture_gb = float(args.capture_max_resident_gb)
    if not np.isfinite(capture_gb) or capture_gb <= 0.0:
        raise CliError("--capture-max-resident-gb must be finite and positive")
    maximum_new = None if args.max_new_groups == 0 else int(args.max_new_groups)
    maximum_seconds = None
    cartography = _cartography_module()
    started = time.monotonic()
    new_publications = 0
    with cartography._root_lock(cartography_root):
        context = _load_live_context(
            cartography_root,
            ooe_root=selected_ooe_root,
        )
        runtime = context["runtime"]
        try:
            manifest = context["capture_manifest"]
            assert isinstance(manifest, CaptureManifest)
            if args.manifest is not None:
                supplied = CaptureManifest.from_bytes(
                    _stable_bytes(Path(args.manifest).expanduser().absolute())
                )
                if supplied != manifest:
                    raise CliError(
                        "supplied MLP manifest differs from live O1 authority"
                    )
            if args.authority_only:
                return {
                    "authority_cells": len(context["authorities"]),
                    "authority_only": True,
                    "base_manifest_sha256": context["base_manifest_sha256"],
                    "fixture": False,
                    "frontier_head_sha256": context["frontier_head_sha256"],
                    "harvester_state_sha256": context["harvester_state_sha256"],
                    "input_manifest_sha256": manifest.input_manifest_sha256,
                    "manifest_sha256": manifest.sha256,
                    "scheduler_sha256": context["scheduler_sha256"],
                    "schema": STATUS_SCHEMA,
                }
            _persist_exact(bank_root / CAPTURE_MANIFEST_NAME, manifest.to_bytes())
            bank = QwenMlpEvidenceBank(
                bank_root,
                budget=MlpEvidenceBudget(
                    max_total_referenced_bytes=_maximum_bytes(args.max_total_mb)
                ),
            )
            if any(
                receipt.capture_mode != "live-exact"
                for receipt, _verification in bank.committed_pairs()
            ):
                raise CliError("live MLP root contains fixture evidence")
            runner = LiveExactMlpCaptureRunner(
                manifest=manifest,
                model=context["model"],
                atlas=context["atlas"],
                authorities=context["authorities"],
                projection=context["projection"],
                max_resident_capture_bytes=int(capture_gb * 1024**3),
            )
            for receipt, verification in bank.committed_pairs():
                if (
                    receipt.manifest_sha256 != manifest.sha256
                    or receipt.model_pin_sha256 != manifest.model_pin_sha256
                    or verification.verifier_sha256 != runner.verifier_sha256
                ):
                    raise CliError(
                        "live MLP resume crosses its manifest/model/verifier pin"
                    )

            def remaining_groups() -> int | None:
                if maximum_new is None:
                    return None
                return maximum_new - new_publications

            def remaining_seconds() -> float | None:
                if maximum_seconds is None:
                    return None
                return max(0.0, maximum_seconds - (time.monotonic() - started))

            state = bank.state()
            lock_path = bank_root / CALIBRATION_LOCK_NAME
            if state.split_counts[2] and (
                state.split_counts[:2] != (25, 10) or not lock_path.exists()
            ):
                raise CliError(
                    "holdout evidence exists without an exact 25/10 calibration lock"
                )
            group_limit = remaining_groups()
            second_limit = remaining_seconds()
            if (
                state.split_counts[:2] != (25, 10)
                and (group_limit is None or group_limit > 0)
                and (second_limit is None or second_limit > 0.0)
            ):
                rows = run_capture_manifest(
                    manifest,
                    bank,
                    runner,
                    allowed_splits=("train", "calibration"),
                    max_new_groups=group_limit,
                    max_seconds=second_limit,
                )
                new_publications += len(rows)
                state = bank.state()

            lock = None
            prefix_fit = None
            if state.split_counts[:2] == (25, 10):
                if lock_path.exists():
                    lock = MlpCalibrationLock.from_bytes(_stable_bytes(lock_path))
                    if state.split_counts[2]:
                        full = bank.build_subspace_corpus()
                        prefix = _prefix_corpus(full)
                    else:
                        prefix = bank.build_subspace_corpus()
                    prefix_fit = SubspaceBatteryFit.from_bytes(
                        _stable_bytes(bank_root / PREFIX_FIT_NAME), corpus=prefix
                    )
                    if (
                        prefix_fit.sha256 != lock.prefix_fit_sha256
                        or prefix.sha256 != lock.prefix_corpus_sha256
                    ):
                        raise CliError("persisted calibration lock/fit changed")
                else:
                    if state.split_counts[2]:
                        raise CliError("holdout opened before calibration was locked")
                    lock, prefix_fit, _prefix = MlpCalibrationLock.create(
                        manifest=manifest,
                        bank=bank,
                        config=analysis_config,
                    )
                    _persist_exact(lock_path, lock.to_bytes())
                    _persist_exact(bank_root / PREFIX_FIT_NAME, prefix_fit.to_bytes())

            group_limit = remaining_groups()
            second_limit = remaining_seconds()
            if (
                lock is not None
                and state.split_counts[2] != 5
                and (group_limit is None or group_limit > 0)
                and (second_limit is None or second_limit > 0.0)
            ):
                rows = run_capture_manifest(
                    manifest,
                    bank,
                    runner,
                    allowed_splits=("holdout",),
                    max_new_groups=group_limit,
                    max_seconds=second_limit,
                )
                new_publications += len(rows)
                state = bank.state()

            analysis = None
            if state.split_counts == (25, 10, 5):
                if lock is None or prefix_fit is None:
                    raise CliError("complete corpus lost its calibration lock")
                corpus = bank.build_subspace_corpus()
                fit = fit_subspace_battery(
                    corpus,
                    train_group_indices=tuple(range(25)),
                    calibration_group_indices=tuple(range(25, 35)),
                    config=prefix_fit.config,
                )
                lock.verify_full_fit(
                    manifest=manifest,
                    bank=bank,
                    corpus=corpus,
                    fit=fit,
                )
                holdout = evaluate_subspace_battery(
                    fit,
                    corpus,
                    holdout_group_indices=tuple(range(35, 40)),
                )
                _persist_exact(bank_root / FULL_FIT_NAME, fit.to_bytes())
                _persist_exact(bank_root / HOLDOUT_NAME, holdout.to_bytes())
                promoted = tuple(row for row in holdout.results if row.promoted)
                analysis = {
                    "calibration_lock_sha256": lock.sha256,
                    "corpus_sha256": corpus.sha256,
                    "fit_sha256": fit.sha256,
                    "holdout_sha256": holdout.sha256,
                    "promoted_model_sha256s": [row.model_sha256 for row in promoted],
                    "schema": LIVE_ANALYSIS_SCHEMA,
                }
            return _status(
                bank=bank,
                manifest=manifest,
                fixture=False,
                publications=new_publications,
                extra={
                    "analysis": analysis,
                    "base_manifest_sha256": context["base_manifest_sha256"],
                    "frontier_head_sha256": context["frontier_head_sha256"],
                    "harvester_state_sha256": context["harvester_state_sha256"],
                    "scheduler_sha256": context["scheduler_sha256"],
                },
            )
        finally:
            runtime.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    status = _run_fixture(args) if args.fixture is not None else _run_live(args)
    encoded = (
        json.dumps(
            status,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    if args.output is not None:
        _persist_exact(Path(args.output).expanduser().absolute(), encoded)
    print(encoded.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
