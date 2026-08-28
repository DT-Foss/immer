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
from typing import Any, Mapping, Sequence, cast

import numpy as np

from immer.runtimes.ooe.qwen_mlp_evidence import (
    CAPTURE_STAGES,
    CaptureManifest,
    CaptureManifestV2,
    CapturePlanEntry,
    ExactMlpBoundaryCapture,
    MlpEvidenceBudget,
    MlpEvidenceReceipt,
    MlpProjectionVerificationReceipt,
    QwenMlpEvidenceBank,
    canonical_all_layer_capture_plan,
    canonical_capture_plan,
    run_capture_manifest,
)
from immer.runtimes.ooe.qwen_mlp_live import (
    LiveExactMlpCaptureRunner,
    LiveMlpAuthority,
    MlpCalibrationLock,
    MlpCalibrationLockV2,
)
from immer.runtimes.ooe.operator_harvester import (
    HarvesterConfig,
    HarvesterState,
    MAX_STATE_BYTES,
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
ALL_LAYER_AUTHORITY_FAMILY = (
    "contextual.mlp-all-layer-authority.full-span-v2"
)
ALL_LAYER_COUNT = 64
LEGACY_CAPTURE_PLAN = "legacy-v1"
ALL_LAYER_CAPTURE_PLAN = "all-layer-v2"
ALL_LAYER_CALIBRATION_LOCK_NAME = "calibration-lock-v2.json"
ALL_LAYER_ANALYSIS_NAME = "layerwise-analysis-v2.json"
ALL_LAYER_ANALYSIS_SCHEMA = "immer.qwen3.8-mlp-layerwise-analysis/v2"
ALL_LAYER_INPUT_AUTHORITY_SCHEMA = "immer.qwen3.8-mlp-live-input-authority/v2"


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


def _all_layer_prompt_splits(
    cartography: Any,
    *,
    cartography_root: Path,
    base_manifest: Mapping[str, object],
    base_body: Mapping[str, object],
) -> tuple[dict[str, tuple[str, ...]], str]:
    """Recover chronological v2 cohorts without trusting effective sort order."""

    base_prompts = tuple(
        sorted(row["sha256"] for row in cartography._manifest_prompts(base_body))
    )
    if len(base_prompts) != 5 or len(set(base_prompts)) != 5:
        raise CliError("all-layer v2 requires exactly five base-manifest prompts")
    frontier, _frontier_file_sha256 = cartography._load_frontier_document(
        cartography_root,
        manifest_sha256=base_manifest["sha256"],
    )
    prompt_events: list[tuple[frozenset[str], str]] = []
    authority_job_events: list[tuple[frozenset[str], str]] = []
    introduced = set(base_prompts)
    for event in frontier["body"]["events"]:
        event_body = event["body"]
        additions = tuple(
            sorted(
                cartography._prompt_record(value)["sha256"]
                for value in event_body["added_prompts"]
            )
        )
        if len(additions) != len(set(additions)) or introduced.intersection(additions):
            raise CliError("frontier prompt introduction history is inconsistent")
        introduced.update(additions)
        if additions:
            prompt_events.append((frozenset(additions), event["sha256"]))
        authority_prompts = frozenset(
            job.get("prompt_sha256")
            for row in event_body["added_jobs"]
            if isinstance(row, Mapping)
            and isinstance(row.get("job"), Mapping)
            and (job := row["job"]).get("probe_family")
            == ALL_LAYER_AUTHORITY_FAMILY
            and isinstance(job.get("prompt_sha256"), str)
        )
        if authority_prompts:
            authority_job_events.append((authority_prompts, event["sha256"]))
    if len(authority_job_events) != 1:
        raise CliError(
            "all-layer v2 jobs must be introduced together in one frontier event"
        )
    authority_prompts, _authority_generation_sha256 = authority_job_events[0]
    if not set(base_prompts).issubset(authority_prompts):
        raise CliError("all-layer v2 job event does not cover the base prompt cohort")
    holdout_set = authority_prompts - set(base_prompts)
    if len(authority_prompts) != 10 or len(holdout_set) != 5:
        raise CliError("all-layer v2 job event does not bind an exact later cohort")
    matching_prompt_events = tuple(
        event_sha256
        for prompts, event_sha256 in prompt_events
        if prompts == holdout_set
    )
    if len(matching_prompt_events) != 1:
        raise CliError(
            "all-layer v2 holdout jobs do not match one prompt-introduction event"
        )
    holdout = tuple(sorted(holdout_set))
    generation_sha256 = matching_prompt_events[0]
    return (
        {
            "train": base_prompts[:3],
            "calibration": base_prompts[3:],
            "holdout": holdout,
        },
        generation_sha256,
    )


def _select_all_layer_jobs(
    prepared: Sequence[tuple[object, object]],
    *,
    prompt_splits: Mapping[str, Sequence[str]],
    model_pin_sha256: str,
    code_revision: str,
) -> dict[str, tuple[object, object]]:
    expected_prompts = {
        prompt
        for split in ("train", "calibration", "holdout")
        for prompt in prompt_splits[split]
    }
    selected: dict[str, tuple[object, object]] = {}
    base = "model.language_model.layers.63.input_layernorm"
    for job, spec in prepared:
        if getattr(job, "probe_family", None) != ALL_LAYER_AUTHORITY_FAMILY:
            continue
        prompt = getattr(spec, "prompt_sha256", None)
        if prompt not in expected_prompts:
            raise CliError("all-layer v2 job names an unexpected prompt")
        coordinate = getattr(spec, "coordinate", None)
        target = getattr(job, "target", None)
        if not (
            getattr(job, "job_id", None)
            and getattr(job, "layer", None) == ALL_LAYER_COUNT - 1
            and getattr(job, "prompt_sha256", None) == prompt
            and getattr(job, "model_pin", None) == model_pin_sha256
            and getattr(job, "code_pin", None) == code_revision
            and getattr(job, "intervention", None) == "passive"
            and getattr(target, "module", None) == base
            and getattr(target, "unit_kind", None) == "module"
            and getattr(target, "unit_index", None) is None
            and getattr(spec, "intervention_mode", None) == "passive"
            and getattr(spec, "start_layer", None) == 0
            and getattr(spec, "stop_layer", None) == ALL_LAYER_COUNT
            and getattr(coordinate, "layer", None) == ALL_LAYER_COUNT - 1
            and getattr(coordinate, "module", None) == base
            and getattr(coordinate, "tensor", None) == f"{base}.weight"
            and getattr(coordinate, "head_index", None) is None
            and getattr(coordinate, "row_start", None) is None
            and getattr(coordinate, "row_end", None) is None
            and getattr(coordinate, "byte_length", None) is None
            and getattr(coordinate, "relative_byte_offset", None) == 0
            and getattr(spec, "hidden_sketch", None) is not None
            and getattr(spec, "native_head_crsa", None) is None
            and getattr(spec, "prefix_sinkhorn_operator_capture", None) is None
        ):
            raise CliError(f"canonical all-layer MLP job changed for prompt {prompt}")
        if prompt in selected:
            raise CliError(f"duplicate all-layer MLP job for prompt {prompt}")
        selected[prompt] = (job, spec)
    if set(selected) != expected_prompts or len(selected) != 10:
        raise CliError("effective frontier lacks the exact ten all-layer v2 jobs")
    return selected


def _successful_outcomes_by_job(
    outcomes: Sequence[object],
    selected: Mapping[str, tuple[object, object]],
) -> dict[str, object]:
    expected_jobs = {getattr(job, "job_id") for job, _spec in selected.values()}
    matches: dict[str, list[object]] = {job_id: [] for job_id in expected_jobs}
    for outcome in outcomes:
        job_id = getattr(outcome, "job_id", None)
        if (
            job_id in expected_jobs
            and getattr(outcome, "status", None) == "succeeded"
            and getattr(outcome, "atlas_receipt_sha256", None) is not None
        ):
            matches[job_id].append(outcome)
    if any(len(rows) != 1 for rows in matches.values()):
        raise CliError("each all-layer v2 job requires one successful Atlas outcome")
    return {job_id: rows[0] for job_id, rows in matches.items()}


def _all_layer_observation_cells(
    observations: Sequence[object],
    *,
    selected: Mapping[str, tuple[object, object]],
    measurements: Mapping[str, object],
    model_pin_sha256: str,
    atlas: object,
    projection: object,
) -> dict[tuple[str, int], object]:
    cells: dict[tuple[str, int], object] = {}
    dimensions = getattr(projection, "output_dimensions", None)
    by_transition: dict[tuple[object, object, object], list[object]] = {}
    for row in observations:
        receipt = getattr(row, "receipt", None)
        key = (
            getattr(getattr(row, "measurement", None), "sha256", None),
            getattr(receipt, "source_state", None),
            getattr(receipt, "target_state", None),
        )
        by_transition.setdefault(key, []).append(row)
    revision_membership: dict[str, bool] = {}

    def contains_revision(receipt: object) -> bool:
        revision = getattr(receipt, "atlas_revision", None)
        sha256 = getattr(revision, "sha256", None)
        if not isinstance(sha256, str):
            return False
        if sha256 not in revision_membership:
            revision_membership[sha256] = bool(
                getattr(atlas, "contains_revision")(revision)
            )
        return revision_membership[sha256]

    for prompt, (_job, spec) in selected.items():
        measurement = measurements[prompt]
        token_count = len(getattr(spec, "prompt_token_ids"))
        for layer in range(ALL_LAYER_COUNT):
            source = f"qwen.layer.{layer}.mlp.input-sketch"
            target = f"qwen.layer.{layer}.mlp.output-sketch"
            matches = tuple(
                row
                for row in by_transition.get(
                    (getattr(measurement, "sha256", None), source, target), ()
                )
                if getattr(getattr(row, "receipt", None), "emitter_sha256", None)
                == QWEN_CONTEXT_EMITTER_SHA256
                and getattr(getattr(row, "receipt", None), "intervention_mode", None)
                == "passive"
                and getattr(getattr(row, "receipt", None), "granularity", None)
                == "operator"
                and getattr(getattr(row, "receipt", None), "segment_start", None)
                is None
                and getattr(getattr(row, "receipt", None), "segment_end", None)
                is None
                and getattr(getattr(row, "receipt", None), "model_pin_sha256", None)
                == model_pin_sha256
                and contains_revision(getattr(row, "receipt"))
                and getattr(getattr(row, "input_array", None), "shape", None)
                == (token_count, dimensions)
                and getattr(getattr(row, "output_array", None), "shape", None)
                == (token_count, dimensions)
            )
            if len(matches) != 1:
                raise CliError(
                    f"Harvester lacks one exact all-layer MLP authority for {(prompt, layer)}"
                )
            cells[(prompt, layer)] = matches[0]
    if len(cells) != 10 * ALL_LAYER_COUNT:
        raise CliError("all-layer MLP authority did not expand to exactly 640 cells")
    return cells


def _all_layer_input_authority_documents(
    *,
    base_manifest_sha256: str,
    generation_sha256: str,
    prompt_splits: Mapping[str, Sequence[str]],
    selected: Mapping[str, tuple[object, object]],
    outcomes: Mapping[str, object],
    measurements: Mapping[str, object],
    scheduler_observation_sha256s: Mapping[str, str],
    cells: Mapping[tuple[str, int], object],
    model_pin_sha256: str,
) -> tuple[str, str]:
    split_by_prompt = {
        prompt: split
        for split in ("train", "calibration", "holdout")
        for prompt in prompt_splits[split]
    }

    def shared_record(prompt: str) -> dict[str, object]:
        job, spec = selected[prompt]
        outcome = outcomes[getattr(job, "job_id")]
        measurement = measurements[prompt]
        return {
            "atlas_receipt_sha256": getattr(outcome, "atlas_receipt_sha256"),
            "measurement_sha256": getattr(measurement, "sha256"),
            "outcome_attempt_id": getattr(outcome, "attempt_id"),
            "probe_family": getattr(job, "probe_family"),
            "probe_job_sha256": getattr(job, "job_id"),
            "probe_spec_sha256": getattr(spec, "sha256"),
            "prompt_sha256": prompt,
            "scheduler_observation_sha256": scheduler_observation_sha256s[prompt],
            "split": split_by_prompt[prompt],
        }

    def cell_record(key: tuple[str, int]) -> dict[str, object]:
        prompt, layer = key
        receipt = getattr(cells[key], "receipt")
        return {
            "context_receipt_sha256": getattr(receipt, "sha256"),
            "input_sha256": getattr(receipt, "input_sha256"),
            "layer": layer,
            "output_sha256": getattr(receipt, "output_sha256"),
            "prompt_sha256": prompt,
            "source_state": getattr(receipt, "source_state"),
            "split": split_by_prompt[prompt],
            "target_state": getattr(receipt, "target_state"),
        }

    base_prompts = tuple(
        prompt_splits["train"]
    ) + tuple(prompt_splits["calibration"])
    base_authority = _digest(
        {
            "base_manifest_sha256": base_manifest_sha256,
            "cells": [
                cell_record((prompt, layer))
                for prompt in base_prompts
                for layer in range(ALL_LAYER_COUNT)
            ],
            "model_pin_sha256": model_pin_sha256,
            "schema": f"{ALL_LAYER_INPUT_AUTHORITY_SCHEMA}.base",
            "shared_measurements": [shared_record(prompt) for prompt in base_prompts],
        }
    )
    all_prompts = base_prompts + tuple(prompt_splits["holdout"])
    later_authority = _digest(
        {
            "base_input_manifest_sha256": base_authority,
            "base_manifest_sha256": base_manifest_sha256,
            "cells": [
                cell_record((prompt, layer))
                for prompt in all_prompts
                for layer in range(ALL_LAYER_COUNT)
            ],
            "generation_sha256": generation_sha256,
            "model_pin_sha256": model_pin_sha256,
            "prompt_bindings": {
                split: list(prompt_splits[split])
                for split in ("train", "calibration", "holdout")
            },
            "schema": ALL_LAYER_INPUT_AUTHORITY_SCHEMA,
            "shared_measurements": [shared_record(prompt) for prompt in all_prompts],
        }
    )
    return base_authority, later_authority


def _load_live_context_legacy(
    cartography_root: Path,
    *,
    ooe_root: Path | None,
    prompt_sha256s: Sequence[str] | None = None,
):
    cartography = _cartography_module()
    base_manifest, body, frontier_status = cartography._effective_manifest(
        cartography_root
    )
    prompt_rows = cartography._manifest_prompts(body)
    available_prompts = {row["sha256"]: row for row in prompt_rows}
    selected_prompts = (
        tuple(sorted(available_prompts))
        if prompt_sha256s is None
        else tuple(sorted(set(prompt_sha256s)))
    )
    if len(selected_prompts) != 5 or any(
        prompt not in available_prompts for prompt in selected_prompts
    ):
        raise CliError("live MLP capture requires exactly five active O1 prompts")
    capture_entries = canonical_capture_plan(selected_prompts)
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
        state_store = cartography.CrystalStore(
            compute_root,
            max_state_bytes=MAX_STATE_BYTES,
        )
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
            prompt_sha256s=selected_prompts,
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


def _load_live_context_all_layer(
    cartography_root: Path,
    *,
    ooe_root: Path | None,
) -> dict[str, object]:
    cartography = _cartography_module()
    base_manifest, base_body = cartography._load_manifest(cartography_root)
    effective_base, body, frontier_status = cartography._effective_manifest(
        cartography_root
    )
    if effective_base != base_manifest:
        raise CliError("effective frontier changed its base manifest")
    prompt_splits, generation_sha256 = _all_layer_prompt_splits(
        cartography,
        cartography_root=cartography_root,
        base_manifest=base_manifest,
        base_body=base_body,
    )
    prepared = cartography._manifest_jobs(body)
    selected = _select_all_layer_jobs(
        prepared,
        prompt_splits=prompt_splits,
        model_pin_sha256=body["model_pin_sha256"],
        code_revision=body["code_revision"],
    )
    projections = tuple(getattr(spec, "hidden_sketch") for _job, spec in selected.values())
    if not projections or any(value != projections[0] for value in projections[1:]):
        raise CliError("all-layer v2 jobs do not share one hidden projection")
    projection = projections[0]

    expected_jobs = {job.job_id: job for job, _spec in prepared}
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
    if {job.job_id: job for job in scheduler.jobs} != expected_jobs:
        raise CliError("O1 scheduler jobs differ from the effective frontier")
    outcomes = _successful_outcomes_by_job(scheduler.outcomes, selected)

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
        measurements: dict[str, object] = {}
        scheduler_observation_sha256s: dict[str, str] = {}
        for prompt, (job, spec) in selected.items():
            outcome = outcomes[job.job_id]
            observation = outcome.observation_document()
            if (
                observation is None
                or observation.get("probe_spec_sha256") != spec.sha256
                or observation.get("prompt_sha256") != prompt
                or observation.get("intervention") != "passive"
            ):
                raise CliError("scheduler all-layer observation differs from its spec")
            measurement = cartography._active_outcome_measurement(
                outcome,
                atlas=atlas,
                spec=spec,
                model_pin=model_pin,
                capture_bank=None,
            )
            measurements[prompt] = measurement
            scheduler_observation_sha256s[prompt] = hashlib.sha256(
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
        state_store = cartography.CrystalStore(
            compute_root,
            max_state_bytes=MAX_STATE_BYTES,
        )
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
        cells = _all_layer_observation_cells(
            observations,
            selected=selected,
            measurements=measurements,
            model_pin_sha256=model_pin.sha256,
            atlas=atlas,
            projection=projection,
        )
        authorities = {}
        for key, row in cells.items():
            prompt, layer = key
            job, spec = selected[prompt]
            authorities[key] = LiveMlpAuthority(
                layer=layer,
                prompt_sha256=prompt,
                probe_spec=spec,
                observation=row,
                scheduler_observation_sha256=(
                    scheduler_observation_sha256s[prompt]
                ),
                probe_family=job.probe_family,
                probe_job_sha256=job.job_id,
            )
        scheduler_raw_after, _scheduler_document_after = (
            cartography.O1Cartographer._read_document(scheduler_path)
        )
        if scheduler_raw_after != scheduler_raw_before:
            raise CliError("O1 scheduler changed during all-layer authority load")
        if (
            atlas.revision() != atlas_head_before
            or atlas.revision_history()[-1] != atlas_head_before
        ):
            raise CliError("Atlas head/history changed during all-layer authority load")
        if state_store.restore_state(state_name) != state_payload:
            raise CliError("Harvester state changed during all-layer authority load")
        base_authority, later_authority = _all_layer_input_authority_documents(
            base_manifest_sha256=base_manifest["sha256"],
            generation_sha256=generation_sha256,
            prompt_splits=prompt_splits,
            selected=selected,
            outcomes=outcomes,
            measurements=measurements,
            scheduler_observation_sha256s=scheduler_observation_sha256s,
            cells=cells,
            model_pin_sha256=model_pin.sha256,
        )
        entries = canonical_all_layer_capture_plan(
            prompt_splits["train"],
            prompt_splits["calibration"],
            prompt_splits["holdout"],
        )
        capture_manifest = CaptureManifestV2(
            model_pin_sha256=model_pin.sha256,
            base_input_manifest_sha256=base_authority,
            later_generation_input_manifest_sha256=later_authority,
            train_prompt_sha256s=prompt_splits["train"],
            calibration_prompt_sha256s=prompt_splits["calibration"],
            holdout_prompt_sha256s=prompt_splits["holdout"],
            entries=entries,
        )
        return {
            "atlas": atlas,
            "authorities": authorities,
            "authority_measurements": len(measurements),
            "base_manifest_sha256": base_manifest["sha256"],
            "capture_manifest": capture_manifest,
            "frontier_head_sha256": frontier_status["head_sha256"],
            "harvester_state_sha256": harvester_state.sha256,
            "input_authority_sha256": later_authority,
            "model": runtime.model,
            "projection": projection,
            "runtime": runtime,
            "scheduler_sha256": hashlib.sha256(scheduler_raw_before).hexdigest(),
        }
    except Exception:
        runtime.close()
        raise


def _load_live_context(
    cartography_root: Path,
    *,
    ooe_root: Path | None,
    prompt_sha256s: Sequence[str] | None = None,
    capture_plan: str = LEGACY_CAPTURE_PLAN,
) -> dict[str, object]:
    if capture_plan == LEGACY_CAPTURE_PLAN:
        return _load_live_context_legacy(
            cartography_root,
            ooe_root=ooe_root,
            prompt_sha256s=prompt_sha256s,
        )
    if capture_plan == ALL_LAYER_CAPTURE_PLAN:
        if prompt_sha256s is not None:
            raise CliError(
                "all-layer v2 derives chronological cohorts from the frontier journal"
            )
        return _load_live_context_all_layer(
            cartography_root,
            ooe_root=ooe_root,
        )
    raise CliError("unknown capture plan")


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
    parser.add_argument(
        "--capture-plan",
        choices=(LEGACY_CAPTURE_PLAN, ALL_LAYER_CAPTURE_PLAN),
        default=LEGACY_CAPTURE_PLAN,
        help="legacy 40-cell bank or chronological 640-cell all-layer bank",
    )
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt-sha256", action="append")
    prompts.add_argument(
        "--prompt-cohort",
        help="JSON prompt registry whose exact five hashes define this bank",
    )
    parser.add_argument(
        "--max-total-mb",
        type=float,
        help="bank byte budget; defaults to 4096 legacy / 8192 all-layer",
    )
    parser.add_argument(
        "--max-new-groups",
        type=int,
        default=0,
        help="0 is unlimited; live capture stops only on a manifest atomic boundary",
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
        help="authenticate live authority without opening an evidence bank",
    )
    return parser


def _maximum_bytes(value: object, *, default_mb: float = 4096.0) -> int:
    maximum = float(default_mb if value is None else value)
    if not np.isfinite(maximum) or maximum <= 0:
        raise CliError("--max-total-mb must be finite and positive")
    return int(maximum * 1024**2)


def _status(
    *,
    bank: QwenMlpEvidenceBank,
    manifest: CaptureManifest | CaptureManifestV2,
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
    if args.capture_plan != LEGACY_CAPTURE_PLAN:
        raise CliError("all-layer v2 is live-only")
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


def _sealed_status_document(schema: str, body: Mapping[str, object]) -> bytes:
    normalized = dict(body)
    return (
        json.dumps(
            {
                "body": normalized,
                "body_sha256": _digest(normalized),
                "schema": schema,
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _read_sealed_status_document(
    path: Path,
    *,
    schema: str,
    maximum: int = 16 * 1024 * 1024,
) -> Mapping[str, object]:
    raw = _stable_bytes(path, maximum=maximum)
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError(f"sealed artifact is not JSON: {path}") from exc
    if (
        not isinstance(document, Mapping)
        or set(document) != {"body", "body_sha256", "schema"}
        or document.get("schema") != schema
        or not isinstance(document.get("body"), Mapping)
        or document.get("body_sha256") != _digest(document["body"])
        or _sealed_status_document(schema, document["body"]) != raw
    ):
        raise CliError(f"sealed artifact changed: {path}")
    return cast(Mapping[str, object], document["body"])


def _all_layer_fit_path(root: Path, kind: str, layer: int) -> Path:
    if kind not in {"prefix", "full", "holdout"}:
        raise ValueError("unknown all-layer analysis artifact kind")
    return root / "layerwise" / kind / f"layer-{layer:02d}.json"


def _all_layer_calibration_lock(
    *,
    bank_root: Path,
    bank: QwenMlpEvidenceBank,
    manifest: CaptureManifestV2,
    config: SubspaceSweepConfig,
) -> MlpCalibrationLockV2:
    lock_path = bank_root / ALL_LAYER_CALIBRATION_LOCK_NAME
    target = manifest.split_targets
    state = bank.state()
    if state.split_counts[:2] != target[:2]:
        raise CliError("all-layer calibration lock requires exact prefix coverage")
    if lock_path.exists():
        lock = MlpCalibrationLockV2.from_bytes(_stable_bytes(lock_path))
        if (
            lock.manifest_sha256 != manifest.sha256
            or lock.model_pin_sha256 != manifest.model_pin_sha256
            or lock.prefix_state_sha256
            not in {value.sha256 for value in bank.state_history()}
            or lock.config_sha256 != _digest(config.to_record())
        ):
            raise CliError("all-layer calibration lock differs from the live prefix")
        for seal in lock.layer_seals:
            layer = seal.layer
            corpus = bank.build_subspace_corpus(
                allowed_splits=("train", "calibration"),
                allowed_layers=(layer,),
            )
            fit = SubspaceBatteryFit.from_bytes(
                _stable_bytes(_all_layer_fit_path(bank_root, "prefix", layer)),
                corpus=corpus,
            )
            if (
                seal.prefix_corpus_sha256 != corpus.sha256
                or seal.prefix_fit_sha256 != fit.sha256
                or seal.model_sha256s != tuple(model.sha256 for model in fit.models)
            ):
                raise CliError("all-layer prefix fit differs from its lock")
        return lock

    if state.split_counts[2]:
        raise CliError("all-layer holdout opened before calibration was locked")
    lock, fits = MlpCalibrationLockV2.create(
        manifest=manifest,
        bank=bank,
        config=config,
    )
    for layer, fit in fits.items():
        _persist_exact(
            _all_layer_fit_path(bank_root, "prefix", layer),
            fit.to_bytes(),
        )
    _persist_exact(lock_path, lock.to_bytes())
    return lock


def _all_layer_analysis(
    *,
    bank_root: Path,
    bank: QwenMlpEvidenceBank,
    manifest: CaptureManifestV2,
    config: SubspaceSweepConfig,
    calibration_lock: MlpCalibrationLockV2,
) -> Mapping[str, object]:
    if bank.state().split_counts != manifest.split_targets:
        raise CliError("all-layer analysis requires complete manifest coverage")
    layers = manifest.layers_for_split("train")
    train_count = len(manifest.prompts_for_split("train"))
    calibration_count = len(manifest.prompts_for_split("calibration"))
    holdout_count = len(manifest.prompts_for_split("holdout"))
    prefix_count = train_count + calibration_count
    total_count = prefix_count + holdout_count
    train_indices = tuple(range(train_count))
    calibration_indices = tuple(range(train_count, prefix_count))
    holdout_indices = tuple(range(prefix_count, total_count))
    if len(calibration_lock.layer_seals) != len(layers):
        raise CliError("all-layer analysis lost its calibration lock")
    result_path = bank_root / ALL_LAYER_ANALYSIS_NAME
    if result_path.exists():
        body = _read_sealed_status_document(
            result_path,
            schema=ALL_LAYER_ANALYSIS_SCHEMA,
        )
        if (
            body.get("manifest_sha256") != manifest.sha256
            or body.get("complete_state_sha256") != bank.state().sha256
            or body.get("calibration_lock_sha256")
            != calibration_lock.sha256
        ):
            raise CliError("all-layer analysis differs from the complete bank")
        rows = body.get("layers")
        if (
            not isinstance(rows, list)
            or len(rows) != len(layers)
            or [row.get("layer") for row in rows if isinstance(row, Mapping)]
            != list(layers)
        ):
            raise CliError("all-layer analysis cache has an invalid layer inventory")
        for row in rows:
            assert isinstance(row, Mapping)
            layer = cast(int, row["layer"])
            for kind, field in (("full", "fit_sha256"), ("holdout", "holdout_sha256")):
                expected_sha256 = row.get(field)
                if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
                    raise CliError("all-layer analysis cache has an invalid artifact pin")
                try:
                    artifact = _stable_bytes(
                        _all_layer_fit_path(bank_root, kind, layer)
                    )
                except (OSError, CliError) as exc:
                    raise CliError(
                        "all-layer analysis cache lost a referenced layer artifact"
                    ) from exc
                if hashlib.sha256(artifact).hexdigest() != expected_sha256:
                    raise CliError(
                        "all-layer analysis cache layer artifact changed"
                    )
        return body

    fits: dict[int, SubspaceBatteryFit] = {}
    for layer in layers:
        corpus = bank.build_subspace_corpus(allowed_layers=(layer,))
        if len(corpus.groups) != total_count:
            raise CliError("all-layer corpus lacks exact 3/2/5 prompt coverage")
        prefix_corpus = SubspaceCorpus(
            corpus.model_pin_sha256,
            corpus.groups[:prefix_count],
        )
        prefix_fit = SubspaceBatteryFit.from_bytes(
            _stable_bytes(_all_layer_fit_path(bank_root, "prefix", layer)),
            corpus=prefix_corpus,
        )
        full_fit = fit_subspace_battery(
            corpus,
            train_group_indices=train_indices,
            calibration_group_indices=calibration_indices,
            config=config,
        )
        if full_fit.config != prefix_fit.config:
            raise CliError("all-layer full fit changed its frozen configuration")
        fits[layer] = full_fit
    calibration_lock.verify_full_fits(
        manifest=manifest,
        bank=bank,
        fits=fits,
    )
    layer_results: list[dict[str, object]] = []
    for layer in layers:
        corpus = bank.build_subspace_corpus(allowed_layers=(layer,))
        full_fit = fits[layer]
        holdout = evaluate_subspace_battery(
            full_fit,
            corpus,
            holdout_group_indices=holdout_indices,
        )
        _persist_exact(
            _all_layer_fit_path(bank_root, "full", layer),
            full_fit.to_bytes(),
        )
        _persist_exact(
            _all_layer_fit_path(bank_root, "holdout", layer),
            holdout.to_bytes(),
        )
        layer_results.append(
            {
                "corpus_sha256": corpus.sha256,
                "fit_sha256": full_fit.sha256,
                "holdout_sha256": holdout.sha256,
                "layer": layer,
                "promoted_model_sha256s": [
                    row.model_sha256 for row in holdout.results if row.promoted
                ],
            }
        )
    body = {
        "calibration_lock_sha256": calibration_lock.sha256,
        "complete_state_sha256": bank.state().sha256,
        "layers": layer_results,
        "manifest_sha256": manifest.sha256,
        "model_pin_sha256": manifest.model_pin_sha256,
    }
    _persist_exact(
        result_path,
        _sealed_status_document(ALL_LAYER_ANALYSIS_SCHEMA, body),
    )
    return body


def _requested_prompts(
    args: argparse.Namespace, bank_root: Path
) -> tuple[str, ...] | None:
    values: Sequence[object] | None = args.prompt_sha256
    if args.prompt_cohort is not None:
        try:
            document = json.loads(
                _stable_bytes(Path(args.prompt_cohort).expanduser().absolute())
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CliError("--prompt-cohort is not JSON") from exc
        if (
            not isinstance(document, dict)
            or set(document) != {"prompts"}
            or not isinstance(document.get("prompts"), list)
        ):
            raise CliError("--prompt-cohort must be an exact prompt registry")
        values = [
            row.get("sha256") if isinstance(row, dict) else None
            for row in document["prompts"]
        ]
    capture_path = bank_root / CAPTURE_MANIFEST_NAME
    if values is None and capture_path.exists():
        return CaptureManifest.from_bytes(_stable_bytes(capture_path)).prompt_sha256s
    if values is None:
        return None
    raw_values = tuple(values)
    if any(not isinstance(value, str) or len(value) != 64 for value in raw_values):
        raise CliError("prompt selection must contain exactly five SHA-256 values")
    result = tuple(sorted(set(cast(tuple[str, ...], raw_values))))
    if len(result) != 5:
        raise CliError("prompt selection must contain exactly five SHA-256 values")
    return cast(tuple[str, ...], result)


def _run_live_legacy(args: argparse.Namespace) -> dict[str, object]:
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
    requested_prompts = _requested_prompts(args, bank_root)
    with cartography._root_lock(cartography_root):
        context = _load_live_context(
            cartography_root,
            ooe_root=selected_ooe_root,
            prompt_sha256s=requested_prompts,
            capture_plan=LEGACY_CAPTURE_PLAN,
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


def _run_live_all_layer(args: argparse.Namespace) -> dict[str, object]:
    bank_root = Path(args.root).expanduser().absolute()
    cartography_root = Path(args.cartography_root).expanduser().absolute()
    selected_ooe_root = (
        None if args.ooe_root is None else Path(args.ooe_root).expanduser().absolute()
    )
    if args.prompt_sha256 is not None or args.prompt_cohort is not None:
        raise CliError(
            "all-layer v2 derives train/calibration/holdout from frontier history"
        )
    if args.max_new_groups < 0:
        raise CliError("--max-new-groups must be non-negative")
    if not np.isfinite(float(args.max_seconds)) or float(args.max_seconds) < 0.0:
        raise CliError("--max-seconds must be finite and non-negative")
    if args.max_seconds:
        raise CliError(
            "live capture cannot hard-bound one atomic Qwen forward; "
            "use --max-new-groups"
        )
    analysis_gb = float(args.analysis_max_working_gb)
    if not np.isfinite(analysis_gb) or analysis_gb <= 0.0:
        raise CliError("--analysis-max-working-gb must be finite and positive")
    analysis_config = SubspaceSweepConfig(max_working_bytes=int(analysis_gb * 1024**3))
    capture_gb = float(args.capture_max_resident_gb)
    if not np.isfinite(capture_gb) or capture_gb <= 0.0:
        raise CliError("--capture-max-resident-gb must be finite and positive")
    maximum_bytes = _maximum_bytes(args.max_total_mb, default_mb=8192.0)
    if maximum_bytes < 8 * 1024**3:
        raise CliError("all-layer v2 requires --max-total-mb of at least 8192")
    maximum_new = None if args.max_new_groups == 0 else int(args.max_new_groups)
    cartography = _cartography_module()
    new_publications = 0
    with cartography._root_lock(cartography_root):
        context = _load_live_context(
            cartography_root,
            ooe_root=selected_ooe_root,
            capture_plan=ALL_LAYER_CAPTURE_PLAN,
        )
        runtime = context["runtime"]
        try:
            manifest = context["capture_manifest"]
            if not isinstance(manifest, CaptureManifestV2):
                raise CliError("all-layer authority returned a legacy capture manifest")
            atomic_group_size = len(manifest.layers_for_split("train"))
            if (
                atomic_group_size != ALL_LAYER_COUNT
                or maximum_new is not None
                and maximum_new % atomic_group_size
            ):
                raise CliError(
                    "--max-new-groups must align to the manifest's 64-layer prompt group"
                )
            if args.manifest is not None:
                supplied = CaptureManifestV2.from_bytes(
                    _stable_bytes(Path(args.manifest).expanduser().absolute())
                )
                if supplied != manifest:
                    raise CliError(
                        "supplied all-layer manifest differs from live O1 authority"
                    )
            if args.authority_only:
                return {
                    "authority_cells": len(context["authorities"]),
                    "authority_measurements": context["authority_measurements"],
                    "authority_only": True,
                    "base_manifest_sha256": context["base_manifest_sha256"],
                    "capture_plan": ALL_LAYER_CAPTURE_PLAN,
                    "fixture": False,
                    "frontier_head_sha256": context["frontier_head_sha256"],
                    "harvester_state_sha256": context["harvester_state_sha256"],
                    "input_manifest_sha256": context["input_authority_sha256"],
                    "manifest_sha256": manifest.sha256,
                    "scheduler_sha256": context["scheduler_sha256"],
                    "schema": STATUS_SCHEMA,
                }
            _persist_exact(bank_root / CAPTURE_MANIFEST_NAME, manifest.to_bytes())
            bank = QwenMlpEvidenceBank(
                bank_root,
                budget=MlpEvidenceBudget(max_total_referenced_bytes=maximum_bytes),
                capture_manifest=manifest,
            )
            if any(
                receipt.capture_mode != "live-exact"
                for receipt, _verification in bank.committed_pairs()
            ):
                raise CliError("all-layer MLP root contains fixture evidence")
            runner = LiveExactMlpCaptureRunner(
                manifest=manifest,
                model=context["model"],
                atlas=context["atlas"],
                authorities=context["authorities"],
                projection=context["projection"],
                max_resident_capture_bytes=int(capture_gb * 1024**3),
            )
            if runner.atomic_capture_group_size != atomic_group_size:
                raise CliError("live runner atomic group differs from the v2 manifest")
            for receipt, verification in bank.committed_pairs():
                if (
                    receipt.manifest_sha256 != manifest.sha256
                    or receipt.model_pin_sha256 != manifest.model_pin_sha256
                    or verification.verifier_sha256 != runner.verifier_sha256
                ):
                    raise CliError(
                        "all-layer resume crosses its manifest/model/verifier pin"
                    )

            def remaining_groups() -> int | None:
                if maximum_new is None:
                    return None
                return maximum_new - new_publications

            state = bank.state()
            train_target, calibration_target, holdout_target = manifest.split_targets
            prefix_targets = (train_target, calibration_target)
            lock_path = bank_root / ALL_LAYER_CALIBRATION_LOCK_NAME
            if state.split_counts[2] and (
                state.split_counts[:2] != prefix_targets or not lock_path.exists()
            ):
                raise CliError(
                    "all-layer holdout exists without its exact calibration lock"
                )
            group_limit = remaining_groups()
            if state.split_counts[:2] != prefix_targets and (
                group_limit is None or group_limit > 0
            ):
                rows = run_capture_manifest(
                    manifest,
                    bank,
                    runner,
                    allowed_splits=("train", "calibration"),
                    max_new_groups=group_limit,
                )
                new_publications += len(rows)
                state = bank.state()

            calibration_lock = None
            if state.split_counts[:2] == prefix_targets:
                calibration_lock = _all_layer_calibration_lock(
                    bank_root=bank_root,
                    bank=bank,
                    manifest=manifest,
                    config=analysis_config,
                )

            group_limit = remaining_groups()
            if (
                calibration_lock is not None
                and state.split_counts[2] != holdout_target
                and (group_limit is None or group_limit > 0)
            ):
                rows = run_capture_manifest(
                    manifest,
                    bank,
                    runner,
                    allowed_splits=("holdout",),
                    max_new_groups=group_limit,
                )
                new_publications += len(rows)
                state = bank.state()

            analysis = None
            if state.split_counts == manifest.split_targets:
                if calibration_lock is None:
                    raise CliError("complete all-layer corpus lost its calibration lock")
                analysis = dict(
                    _all_layer_analysis(
                        bank_root=bank_root,
                        bank=bank,
                        manifest=manifest,
                        config=analysis_config,
                        calibration_lock=calibration_lock,
                    )
                )
            return _status(
                bank=bank,
                manifest=manifest,
                fixture=False,
                publications=new_publications,
                extra={
                    "analysis": analysis,
                    "authority_cells": len(context["authorities"]),
                    "authority_measurements": context["authority_measurements"],
                    "base_manifest_sha256": context["base_manifest_sha256"],
                    "capture_plan": ALL_LAYER_CAPTURE_PLAN,
                    "frontier_head_sha256": context["frontier_head_sha256"],
                    "harvester_state_sha256": context["harvester_state_sha256"],
                    "scheduler_sha256": context["scheduler_sha256"],
                },
            )
        finally:
            runtime.close()


def _run_live(args: argparse.Namespace) -> dict[str, object]:
    if args.capture_plan == ALL_LAYER_CAPTURE_PLAN:
        return _run_live_all_layer(args)
    return _run_live_legacy(args)


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
