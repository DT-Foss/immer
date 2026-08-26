#!/usr/bin/env python3
"""Execute and verify an offline real-Qwen Atlas-to-OoE warm cohort.

The command never opens a checkpoint, tensor bundle, model runtime, or network
transport.  It reconstructs the exact tensor plans from the primary immutable
Atlas MeasurementReceipts, restores the already-trained OoE controller, and
discharges every attached scheduler outcome through its promoted Crystal.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any

from immer.knowledge.livecausal import LiveGraph
from immer.runtimes.deepseek_v4.causal_weights import TensorRangePlan
from immer.runtimes.o1_state import ProbeJob, ProbeOutcome
from immer.runtimes.ooe.cartography import (
    AtlasProbeExecutor,
    authenticated_measurement_verifier_sha256,
    verify_atlas_probe_execution,
)
from immer.runtimes.ooe.controller import (
    CONTROLLER_STATE_NAME,
    ActionExecution,
    OoeController,
    VerifiedTeacherTransition,
    WarmAccountingReceipt,
)
from immer.runtimes.ooe.crystal import CrystalStore, CrystalStoreAudit
from immer.runtimes.ooe.identity import canonical_json_bytes, require_sha256
from immer.runtimes.ooe.qwen_bridge import QwenOoeFeatureReceipt
from immer.runtimes.qwen3_8.semantic_atlas import (
    ATLAS_EDGE_SCHEMA,
    MEASUREMENT_RECEIPT_SCHEMA,
    MeasurementReceipt,
    SemanticWeightAtlas,
)


RESULT_SCHEMA = "immer.qwen3.8-ooe-cold-warm-cohort/v1"
SCHEDULE_SCHEMA = "immer.qwen3.8-ooe-cold-warm-schedule/v1"
QUALITY_CONTRACT = (
    "active MeasurementReceipt + exact ModelPin + WeightCoordinate + "
    "Atlas journal membership + feature-bound verifier"
)
_SCHEDULER_SCHEMA = "immer.o1-cartography-state/v1"
_STATE_PREFIX = "qwen38-ooe-cold-warm"
_LOCK_NAME = ".qwen38-ooe-cold-warm-cohort.lock"
_MAX_JSON_BYTES = 64 * 1024 * 1024


class CohortError(RuntimeError):
    """The sealed offline cohort contract cannot be satisfied."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stable_regular_bytes(path: Path, *, maximum: int = _MAX_JSON_BYTES) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise CohortError(f"cannot inspect required file: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CohortError(f"required path is not a regular file: {path}")
    if before.st_size > maximum:
        raise CohortError(f"file exceeds {maximum} bytes: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CohortError(f"cannot open required file: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise CohortError(f"required path changed while opening: {path}")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise CohortError(f"required file was truncated: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CohortError(f"required file grew while reading: {path}")
        after = path.lstat()
        if (opened.st_dev, opened.st_ino, opened.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise CohortError(f"required file changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _decode_canonical_json(raw: bytes, *, label: str) -> dict[str, Any]:
    if not raw.endswith(b"\n"):
        raise CohortError(f"{label} is not canonical JSONL")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CohortError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise CohortError(f"{label} root must be an object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise CohortError(f"{label} is not canonical JSON")
    return value


def _sealed_result(body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json_bytes(body))
    return {
        "body": normalized,
        "schema": RESULT_SCHEMA,
        "sha256": _digest(normalized),
    }


def _result_bytes(document: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(document) + b"\n"


def _parse_result(raw: bytes) -> dict[str, Any]:
    document = _decode_canonical_json(raw, label="cohort result")
    if set(document) != {"body", "schema", "sha256"}:
        raise CohortError("cohort result envelope is invalid")
    if document.get("schema") != RESULT_SCHEMA:
        raise CohortError("cohort result schema is invalid")
    body = document.get("body")
    if not isinstance(body, dict):
        raise CohortError("cohort result body is invalid")
    claimed = require_sha256(document.get("sha256"), field="result sha256")
    if claimed != _digest(body):
        raise CohortError("cohort result SHA-256 mismatch")
    return document


@contextmanager
def _cohort_lock(root: Path) -> Iterator[None]:
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    path = root / _LOCK_NAME
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise CohortError(f"cannot open cohort lock: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise CohortError("cohort lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        linked = path.lstat()
        if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
            raise CohortError("cohort lock changed while acquiring it")
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _atomic_result(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if _stable_regular_bytes(path) != data:
            raise CohortError("refusing to overwrite a different cohort result")
        return
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".pending", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError:
            if _stable_regular_bytes(path) != data:
                raise CohortError("cohort result appeared with different bytes")
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _SchedulerState:
    state_sha256: str
    file_sha256: str
    jobs: tuple[ProbeJob, ...]
    outcomes: tuple[ProbeOutcome, ...]


def _load_scheduler(path: Path) -> _SchedulerState:
    raw = _stable_regular_bytes(path)
    document = _decode_canonical_json(raw, label="cartography scheduler")
    expected = {
        "config",
        "generation",
        "histories",
        "in_flight",
        "jobs",
        "last_stop_reason",
        "outcomes",
        "replay",
        "schema",
        "state_sha256",
        "stream_state",
    }
    if set(document) != expected or document.get("schema") != _SCHEDULER_SCHEMA:
        raise CohortError("cartography scheduler envelope is invalid")
    claimed = require_sha256(
        document.get("state_sha256"), field="scheduler state_sha256"
    )
    body = dict(document)
    del body["state_sha256"]
    if _digest(body) != claimed:
        raise CohortError("cartography scheduler state SHA-256 mismatch")
    if document.get("in_flight") not in (None, []):
        raise CohortError("cartography scheduler still has an in-flight probe")
    try:
        jobs = tuple(ProbeJob.from_document(row) for row in document["jobs"])
        outcomes = tuple(ProbeOutcome.from_document(row) for row in document["outcomes"])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CohortError("cartography scheduler records are invalid") from exc
    if not jobs or len({row.job_id for row in jobs}) != len(jobs):
        raise CohortError("cartography scheduler jobs are empty or duplicated")
    if len({row.attempt_id for row in outcomes}) != len(outcomes):
        raise CohortError("cartography scheduler contains duplicate attempts")
    attached = tuple(
        row
        for row in outcomes
        if row.status == "succeeded" and row.atlas_receipt_sha256 is not None
    )
    by_job: dict[str, list[ProbeOutcome]] = {row.job_id: [] for row in jobs}
    for row in attached:
        if row.job_id not in by_job:
            raise CohortError("attached outcome references an unknown job")
        by_job[row.job_id].append(row)
        observation = row.observation_document()
        if (
            observation is None
            or observation.get("atlas_receipt_sha256") != row.atlas_receipt_sha256
            or observation.get("measurement_sha256") != row.atlas_receipt_sha256
        ):
            raise CohortError("attached outcome lost its exact Atlas receipt")
    if len(attached) != len(jobs) or any(len(rows) != 1 for rows in by_job.values()):
        raise CohortError("cartography scheduler is not a completed attached cohort")
    return _SchedulerState(
        state_sha256=claimed,
        file_sha256=hashlib.sha256(raw).hexdigest(),
        jobs=jobs,
        outcomes=outcomes,
    )


@dataclass(frozen=True, slots=True)
class _AtlasState:
    atlas: SemanticWeightAtlas
    measurements: Mapping[str, MeasurementReceipt]
    tensor_plans: tuple[TensorRangePlan, ...]


def _plan_from_measurement(measurement: MeasurementReceipt) -> TensorRangePlan:
    coordinate = measurement.coordinate
    plan = TensorRangePlan(
        name=coordinate.tensor,
        dtype=coordinate.dtype,
        shape=coordinate.shape,
        shard=coordinate.shard,
        absolute_offset=coordinate.tensor_absolute_offset,
        length=coordinate.tensor_length,
    )
    if not coordinate.matches_plan(plan):
        raise CohortError("MeasurementReceipt failed reconstructed tensor-plan binding")
    return plan


def _open_atlas(root: Path) -> _AtlasState:
    atlas_root = root / "atlas"
    if not atlas_root.is_dir() or atlas_root.is_symlink():
        raise CohortError("cartography root has no real Atlas directory")
    try:
        graph = LiveGraph(atlas_root)
        if not graph.store.verify(include_tombstones=True):
            raise CohortError("Atlas LiveGraph segment verification failed")
        measurements: dict[str, MeasurementReceipt] = {}
        for segment_sha256 in graph.store.segments():
            records = tuple(
                record
                for _sha256, _index, record in graph.store.iter_records(segment_sha256)
            )
            for record in records:
                if (
                    record.get("schema") == ATLAS_EDGE_SCHEMA
                    and record.get("edge_kind") == "primary"
                ):
                    document = record.get("document")
                    if (
                        isinstance(document, Mapping)
                        and document.get("schema") == MEASUREMENT_RECEIPT_SCHEMA
                    ):
                        measurement = MeasurementReceipt.from_document(document)
                        if measurement.sha256 != record.get("document_sha256"):
                            raise CohortError("Atlas primary measurement identity mismatch")
                        measurements[measurement.sha256] = measurement
        if not measurements:
            raise CohortError("Atlas contains no primary MeasurementReceipts")
        pins = {row.model_pin.sha256: row.model_pin for row in measurements.values()}
        if len(pins) != 1:
            raise CohortError("Atlas primary measurements span multiple model pins")
        plans: dict[str, TensorRangePlan] = {}
        for measurement in measurements.values():
            plan = _plan_from_measurement(measurement)
            prior = plans.get(plan.name)
            if prior is not None and prior != plan:
                raise CohortError("Atlas contains conflicting tensor plans")
            plans[plan.name] = plan
        model_pin = next(iter(pins.values()))
        atlas = SemanticWeightAtlas(
            graph,
            model_pin=model_pin,
            tensor_plans=tuple(plans.values()),
        )
        atlas.verify_or_raise()
    except CohortError:
        raise
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise CohortError("cannot reconstruct and authenticate the Atlas") from exc
    return _AtlasState(
        atlas=atlas,
        measurements=measurements,
        tensor_plans=tuple(sorted(plans.values(), key=lambda row: row.name)),
    )


@dataclass(frozen=True, slots=True)
class _HistoricalFeature:
    receipt: QwenOoeFeatureReceipt
    source_action: str
    target_action: str


def _controller_document(raw: bytes) -> dict[str, Any]:
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CohortError("OoE controller snapshot is invalid JSON") from exc
    if not isinstance(document, dict) or canonical_json_bytes(document) != raw:
        raise CohortError("OoE controller snapshot is not canonical")
    return document


def _historical_features(raw: bytes) -> tuple[_HistoricalFeature, ...]:
    document = _controller_document(raw)
    try:
        sites = document["body"]["sites"]
    except (KeyError, TypeError) as exc:
        raise CohortError("OoE controller snapshot has no historical sites") from exc
    features: list[_HistoricalFeature] = []
    try:
        for site in sites:
            for row in site["history"]:
                receipt = QwenOoeFeatureReceipt.from_document(row["feature"])
                transition = VerifiedTeacherTransition.from_document(row["transition"])
                transition.assert_bound(receipt)
                features.append(
                    _HistoricalFeature(
                        receipt=receipt,
                        source_action=transition.source_action,
                        target_action=transition.target_action,
                    )
                )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, CohortError):
            raise
        raise CohortError("OoE historical controller replay is invalid") from exc
    return tuple(features)


@dataclass(frozen=True, slots=True)
class _ScheduleRow:
    job: ProbeJob
    outcome: ProbeOutcome
    measurement: MeasurementReceipt
    feature: QwenOoeFeatureReceipt
    source_action: str
    target_action: str

    def identity_record(self) -> dict[str, Any]:
        return {
            "attempt_id": self.outcome.attempt_id,
            "feature_receipt_sha256": self.feature.sha256,
            "job_id": self.job.job_id,
            "measurement_sha256": self.measurement.sha256,
            "source_action": self.source_action,
            "target_action": self.target_action,
            "temporal_index": self.feature.temporal_index,
        }


@dataclass(frozen=True, slots=True)
class _Context:
    root: Path
    store: CrystalStore
    atlas_state: _AtlasState
    scheduler: _SchedulerState
    rows: tuple[_ScheduleRow, ...]
    cohort_identity: str
    cohort_state_name: str
    root_manifest_sha256: str | None


def _build_context(root: Path) -> _Context:
    root = root.expanduser().absolute()
    if not root.is_dir() or root.is_symlink():
        raise CohortError("cartography root must be a real directory")
    scheduler = _load_scheduler(root / "scheduler.json")
    atlas_state = _open_atlas(root)
    store_root = root / "ooe"
    if not store_root.is_dir() or store_root.is_symlink():
        raise CohortError("cartography root has no trained OoE CrystalStore")
    store = CrystalStore(store_root)
    try:
        controller_raw = store.restore_state(CONTROLLER_STATE_NAME)
    except KeyError as exc:
        raise CohortError("OoE controller snapshot is missing") from exc
    historical = _historical_features(controller_raw)
    jobs = {row.job_id: row for row in scheduler.jobs}
    measurements = atlas_state.measurements
    by_attempt: dict[str, tuple[str, str]] = {}
    current_action: dict[str, str] = {}
    attached: list[ProbeOutcome] = []
    for outcome in scheduler.outcomes:
        if outcome.status != "succeeded" or outcome.atlas_receipt_sha256 is None:
            continue
        job = jobs[outcome.job_id]
        source = current_action.get(job.prompt_sha256, "qwen_fallback")
        by_attempt[outcome.attempt_id] = (source, "probe_coordinate")
        current_action[job.prompt_sha256] = "probe_coordinate"
        attached.append(outcome)
    schedule: list[_ScheduleRow] = []
    seen_measurements: set[str] = set()
    for outcome in attached:
        measurement_sha256 = require_sha256(
            outcome.atlas_receipt_sha256, field="atlas receipt SHA-256"
        )
        if measurement_sha256 in seen_measurements:
            raise CohortError("attached cohort repeats one Atlas measurement")
        seen_measurements.add(measurement_sha256)
        measurement = measurements.get(measurement_sha256)
        if measurement is None:
            raise CohortError("attached scheduler measurement is absent from Atlas")
        job = jobs[outcome.job_id]
        if job.model_pin != measurement.model_pin.sha256:
            raise CohortError("scheduler job belongs to a stale Qwen model pin")
        candidates = tuple(
            row
            for row in historical
            if row.receipt.measurement_sha256 == measurement_sha256
            and outcome.attempt_id in row.receipt.evidence_sha256s
        )
        if len(candidates) != 1:
            raise CohortError("attached outcome has no unique historical OoE feature")
        historical_row = candidates[0]
        historical_row.receipt.validate_measurement(measurement)
        expected_source, expected_target = by_attempt[outcome.attempt_id]
        if (
            historical_row.source_action != expected_source
            or historical_row.target_action != expected_target
        ):
            raise CohortError("historical Markov action path differs from scheduler order")
        schedule.append(
            _ScheduleRow(
                job=job,
                outcome=outcome,
                measurement=measurement,
                feature=historical_row.receipt,
                source_action=expected_source,
                target_action=expected_target,
            )
        )
    schedule.sort(key=lambda row: (row.feature.temporal_index, row.outcome.attempt_id))
    temporal = [row.feature.temporal_index for row in schedule]
    if len(set(temporal)) != len(temporal):
        raise CohortError("historical feature temporal order is duplicated")
    pins = {row.measurement.model_pin.sha256 for row in schedule}
    weight_revisions = {row.measurement.weight_rail_revision.sha256 for row in schedule}
    if len(pins) != 1 or len(weight_revisions) != 1:
        raise CohortError("cohort spans multiple Qwen pins or weight graphs")
    atlas = atlas_state.atlas
    for row in schedule:
        result = atlas.query_by_prompt_signature(row.measurement.probe.prompt_signature)
        if row.measurement not in result.measurements:
            raise CohortError("historical measurement is not active in Atlas")
        if not atlas.contains_revision(row.measurement.atlas_head_revision):
            raise CohortError("historical measurement Atlas revision is stale")
    root_manifest = root / "manifest.json"
    root_manifest_sha256 = (
        hashlib.sha256(_stable_regular_bytes(root_manifest)).hexdigest()
        if root_manifest.exists()
        else None
    )
    identity_body = {
        "atlas_revision": atlas.revision().to_document(),
        "model_pin_sha256": next(iter(pins)),
        "rows": [row.identity_record() for row in schedule],
        "scheduler_state_sha256": scheduler.state_sha256,
        "schema": SCHEDULE_SCHEMA,
        "weight_graph_revision_sha256": next(iter(weight_revisions)),
    }
    cohort_identity = _digest(identity_body)
    return _Context(
        root=root,
        store=store,
        atlas_state=atlas_state,
        scheduler=scheduler,
        rows=tuple(schedule),
        cohort_identity=cohort_identity,
        cohort_state_name=f"{_STATE_PREFIX}-{cohort_identity[:24]}",
        root_manifest_sha256=root_manifest_sha256,
    )


@dataclass(frozen=True, slots=True)
class _Committed:
    transaction_sha256: str
    decision_binding_sha256: str
    execution: ActionExecution
    accounting: WarmAccountingReceipt


@dataclass(frozen=True, slots=True)
class _TransactionReplay:
    status: str
    execution: ActionExecution
    committed: _Committed | None


def _warm_transactions(raw: bytes) -> Mapping[str, _TransactionReplay]:
    document = _controller_document(raw)
    try:
        transactions = document["body"]["warm_transactions"]
    except (KeyError, TypeError) as exc:
        raise CohortError("OoE snapshot lost warm transactions") from exc
    by_feature: dict[str, _TransactionReplay] = {}
    try:
        for row in transactions:
            execution = ActionExecution.from_document(row["execution"])
            transaction_sha256 = require_sha256(
                row["transaction_sha256"], field="transaction_sha256"
            )
            decision_binding_sha256 = require_sha256(
                row["decision_binding_sha256"], field="decision_binding_sha256"
            )
            status = row["status"]
            if status not in ("pending", "committed", "rejected"):
                raise CohortError("warm transaction status is invalid")
            raw_accounting = row["final_receipt"]
            accounting = (
                None
                if raw_accounting is None
                else WarmAccountingReceipt.from_dict(raw_accounting)
            )
            if (status == "pending") != (accounting is None):
                raise CohortError("warm transaction disposition is inconsistent")
            if accounting is not None and (
                accounting.transaction_sha256 != transaction_sha256
                or accounting.decision_binding_sha256 != decision_binding_sha256
                or accounting.execution_receipt_sha256 != execution.sha256
                or accounting.disposition != status
            ):
                raise CohortError("warm transaction accounting binding is invalid")
            feature_sha256 = execution.feature_receipt_sha256
            if feature_sha256 in by_feature:
                raise CohortError("one feature has duplicate warm transactions")
            committed = (
                None
                if status != "committed" or accounting is None
                else _Committed(
                    transaction_sha256=transaction_sha256,
                    decision_binding_sha256=decision_binding_sha256,
                    execution=execution,
                    accounting=accounting,
                )
            )
            by_feature[feature_sha256] = _TransactionReplay(
                status=status,
                execution=execution,
                committed=committed,
            )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        if isinstance(exc, CohortError):
            raise
        raise CohortError("OoE warm transaction replay is invalid") from exc
    return by_feature


def _audit_record(audit: CrystalStoreAudit) -> dict[str, Any]:
    return {
        "clean": audit.clean,
        "generation": audit.generation,
        "manifest_sha256": audit.manifest_sha256,
        "missing_objects": list(audit.missing_objects),
        "orphan_objects": list(audit.orphan_objects),
        "staged_files": list(audit.staged_files),
        "tampered_objects": list(audit.tampered_objects),
        "unexpected_files": list(audit.unexpected_files),
        "valid_objects": list(audit.valid_objects),
    }


def _row_record(
    context: _Context,
    schedule: _ScheduleRow,
    committed: _Committed,
) -> dict[str, Any]:
    if committed.execution.action != schedule.target_action:
        raise CohortError("warm execution action differs from historical transition")
    if committed.execution.feature_receipt_sha256 != schedule.feature.sha256:
        raise CohortError("warm execution feature differs from historical receipt")
    if schedule.target_action != "probe_coordinate":
        raise CohortError("cohort schedule is not an authenticated Qwen probe")
    if committed.execution.teacher_baseline_qwen_forwards != 1:
        raise CohortError("Qwen probe baseline must be exactly one forward")
    if committed.execution.qwen_forwards != 0:
        raise CohortError("offline Atlas probe execution must use zero Qwen forwards")
    if (
        committed.execution.saved_qwen_forwards != 1
        or committed.accounting.saved_qwen_forwards != 1
    ):
        raise CohortError("Qwen probe must save exactly one authenticated forward")
    if not verify_atlas_probe_execution(
        context.atlas_state.atlas,
        schedule.feature,
        committed.execution,
    ):
        raise CohortError("final Atlas execution verification failed")
    measurement = MeasurementReceipt.from_document(committed.execution.result)
    if measurement != schedule.measurement:
        raise CohortError("warm execution returned a different Atlas measurement")
    verifier_sha256 = authenticated_measurement_verifier_sha256(
        context.atlas_state.atlas,
        schedule.measurement,
    )
    if committed.execution.verifier_sha256 != verifier_sha256:
        raise CohortError("warm execution verifier differs from final Atlas proof")
    return {
        "accounting": committed.accounting.to_dict(),
        "attempt_id": schedule.outcome.attempt_id,
        "execution": committed.execution.to_document(),
        "feature_receipt_sha256": schedule.feature.sha256,
        "final_atlas_verifier_sha256": verifier_sha256,
        "job_id": schedule.job.job_id,
        "measurement_sha256": schedule.measurement.sha256,
        "source_action": schedule.source_action,
        "target_action": schedule.target_action,
        "temporal_index": schedule.feature.temporal_index,
        "transaction_sha256": committed.transaction_sha256,
    }


def _source_record(context: _Context, controller_snapshot_sha256: str) -> dict[str, Any]:
    rows = context.rows
    model_pin = rows[0].measurement.model_pin
    revision = rows[0].measurement.weight_rail_revision
    coordinates_by_tensor = {
        measurement.coordinate.tensor: measurement.coordinate
        for measurement in context.atlas_state.measurements.values()
    }
    plans = []
    for plan in context.atlas_state.tensor_plans:
        coordinate = coordinates_by_tensor[plan.name]
        plans.append(
            {
                "absolute_offset": plan.absolute_offset,
                "dtype": plan.dtype,
                "length": plan.length,
                "name": plan.name,
                "shape": list(plan.shape),
                "shard": plan.shard,
                "tensor_plan_sha256": coordinate.tensor_plan_sha256,
            }
        )
    return {
        "atlas_revision": context.atlas_state.atlas.revision().to_document(),
        "controller_snapshot_sha256": controller_snapshot_sha256,
        "model_pin": model_pin.to_document(),
        "root_manifest_sha256": context.root_manifest_sha256,
        "scheduler_file_sha256": context.scheduler.file_sha256,
        "scheduler_state_sha256": context.scheduler.state_sha256,
        "tensor_plans": plans,
        "weight_graph_revision": revision.to_document(),
    }


def _build_document(
    context: _Context,
    rows: Sequence[Mapping[str, Any]],
    *,
    controller_snapshot_sha256: str,
    audit: CrystalStoreAudit,
) -> dict[str, Any]:
    if len(rows) != len(context.rows):
        raise CohortError("verified rows do not cover the immutable cohort schedule")
    # Baseline authority is the immutable scheduler: every attached row is one
    # completed Qwen cartography probe.  Persisted warm-accounting integers are
    # checked in `_row_record`, never trusted to define the headline.
    baseline = len(context.rows)
    executed = 0
    saved = len(context.rows)
    if not audit.clean:
        raise CohortError("CrystalStore audit is not clean")
    body = {
        "cohort_state_name": context.cohort_state_name,
        "headline": {
            "attached_scheduler_outcomes": len(rows),
            "executed_qwen_forwards": executed,
            "saved_qwen_forwards": saved,
            "teacher_baseline_qwen_forwards": baseline,
            "teacher_calls": 0,
            "verified_warm_results": len(rows),
        },
        "quality_contract": QUALITY_CONTRACT,
        "rows": list(rows),
        "schedule": {
            "attempt_ids": [row.outcome.attempt_id for row in context.rows],
            "cohort_identity": context.cohort_identity,
            "feature_receipt_sha256s": [row.feature.sha256 for row in context.rows],
            "measurement_sha256s": [
                row.measurement.sha256 for row in context.rows
            ],
            "schema": SCHEDULE_SCHEMA,
        },
        "source": _source_record(context, controller_snapshot_sha256),
        "store_audit": _audit_record(audit),
    }
    return _sealed_result(body)


def _restore_controller(context: _Context) -> OoeController:
    atlas = context.atlas_state.atlas
    executor = AtlasProbeExecutor(atlas)
    first = context.rows[0].measurement
    try:
        return OoeController.restore(
            crystal_store=context.store,
            action_executors={"probe_coordinate": executor},
            atlas_revision_verifier=atlas.contains_revision,
            expected_model_pin_sha256=first.model_pin.sha256,
            expected_weight_graph_revision_sha256=(first.weight_rail_revision.sha256),
        )
    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
        raise CohortError("cannot restore exact historical OoE controller") from exc


def _expected_document(context: _Context) -> dict[str, Any]:
    controller = _restore_controller(context)
    snapshot = controller.snapshot_bytes()
    snapshot_sha256 = hashlib.sha256(snapshot).hexdigest()
    transactions = _warm_transactions(snapshot)
    rows: list[dict[str, Any]] = []
    for schedule in context.rows:
        replay = transactions.get(schedule.feature.sha256)
        if replay is None or replay.committed is None:
            raise CohortError("cohort is not fully committed in the controller snapshot")
        rows.append(_row_record(context, schedule, replay.committed))
    if len({row["transaction_sha256"] for row in rows}) != len(rows):
        raise CohortError("cohort rows reuse one warm transaction")
    return _build_document(
        context,
        rows,
        controller_snapshot_sha256=snapshot_sha256,
        audit=context.store.audit(),
    )


def _verify_document(context: _Context, raw: bytes) -> dict[str, Any]:
    document = _parse_result(raw)
    snapshot_before = context.store.restore_state(CONTROLLER_STATE_NAME)
    expected = _expected_document(context)
    snapshot_after = context.store.restore_state(CONTROLLER_STATE_NAME)
    if snapshot_after != snapshot_before:
        raise CohortError("verification mutated the controller snapshot")
    if document != expected:
        raise CohortError("cohort result differs from exact controller replay")
    try:
        persisted = context.store.restore_state(context.cohort_state_name)
    except KeyError as exc:
        raise CohortError("cohort CAS state is missing") from exc
    if persisted != raw:
        raise CohortError("cohort CAS state differs from result bytes")
    return document


def run(root: str | os.PathLike[str], output: str | os.PathLike[str]) -> dict[str, Any]:
    root_path = Path(root).expanduser().absolute()
    output_path = Path(output).expanduser().absolute()
    with _cohort_lock(root_path):
        context = _build_context(root_path)
        try:
            persisted = context.store.restore_state(context.cohort_state_name)
        except KeyError:
            persisted = None
        if persisted is not None:
            document = _verify_document(context, persisted)
            _atomic_result(output_path, persisted)
            return document
        if output_path.exists() or output_path.is_symlink():
            raise CohortError("result exists without its cohort CAS state")

        controller = _restore_controller(context)
        snapshot_sha256 = hashlib.sha256(controller.snapshot_bytes()).hexdigest()
        teacher_calls_before = controller.metrics.teacher_calls
        for schedule in context.rows:
            transactions = _warm_transactions(controller.snapshot_bytes())
            replay = transactions.get(schedule.feature.sha256)
            if replay is not None:
                if replay.committed is None:
                    raise CohortError(
                        "scheduled feature already has an uncommitted warm execution"
                    )
                continue
            decision = controller.try_warm(
                schedule.feature,
                schedule.source_action,
                quality_verifier=lambda receipt, execution: (
                    verify_atlas_probe_execution(
                        context.atlas_state.atlas,
                        receipt,
                        execution,
                    )
                ),
                stream_id=(
                    f"{context.cohort_identity}:{schedule.outcome.attempt_id}"
                ),
            )
            if (
                decision.origin != "crystal"
                or decision.action != schedule.target_action
                or not decision.quality_verified
                or decision.teacher_called
            ):
                raise CohortError(
                    f"promoted Crystal did not execute: {decision.reason}"
                )
            transaction = controller.commit_warm(decision)
            if transaction.disposition != "committed":
                raise CohortError("final Atlas verification was not committed")
            publication = controller.save_snapshot(expected_sha256=snapshot_sha256)
            snapshot_sha256 = publication.payload_sha256
        if controller.metrics.teacher_calls != teacher_calls_before:
            raise CohortError("offline warm cohort invoked the Qwen teacher")
        final_publication = controller.save_snapshot(expected_sha256=snapshot_sha256)
        snapshot_sha256 = final_publication.payload_sha256
        expected = _expected_document(context)
        if expected["body"]["source"]["controller_snapshot_sha256"] != snapshot_sha256:
            raise CohortError("final CAS snapshot identity changed during replay")
        encoded = _result_bytes(expected)
        publication = context.store.publish_state(context.cohort_state_name, encoded)
        if publication.payload_sha256 != hashlib.sha256(encoded).hexdigest():
            raise CohortError("cohort CAS state publication digest mismatch")
        verified = _verify_document(context, encoded)
        _atomic_result(output_path, encoded)
        return verified


def verify(
    root: str | os.PathLike[str], result: str | os.PathLike[str]
) -> dict[str, Any]:
    root_path = Path(root).expanduser().absolute()
    result_path = Path(result).expanduser().absolute()
    with _cohort_lock(root_path):
        context = _build_context(root_path)
        raw = _stable_regular_bytes(result_path)
        return _verify_document(context, raw)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="execute or replay the cohort")
    run_parser.add_argument("--root", required=True, help="completed cartography root")
    run_parser.add_argument("--output", required=True, help="sealed result JSON path")
    verify_parser = subparsers.add_parser("verify", help="verify without decisions")
    verify_parser.add_argument("--root", required=True, help="completed cartography root")
    verify_parser.add_argument("--result", required=True, help="sealed result JSON path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            document = run(args.root, args.output)
        else:
            document = verify(args.root, args.result)
    except (CohortError, OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(document, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
