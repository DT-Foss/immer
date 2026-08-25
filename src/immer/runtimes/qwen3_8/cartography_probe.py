"""Exact Qwen execution bridge for O1-state semantic cartography.

The bridge measures immutable checkpoint coordinates.  It never asks model
text to label a weight and never commits continuation state.  Every arm starts
from the same token IDs and empty layer state, advances one decoder layer at a
time through :meth:`StreamedQwen38.hidden_stateful_range`, and returns an
authenticated :class:`~.semantic_atlas.MeasurementReceipt` plus the evidence
and access trace from which it was built.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
from typing import Any, Literal, Mapping

import torch

from immer.knowledge import AccessTrace, AccessTraceRecorder

from .bundle import QWEN38_BUNDLE_SCHEMA
from .kernels import AttentionState, DeltaNetProbe, DeltaNetState
from .model import LayerState, StreamedQwen38
from .native_crsa import (
    NATIVE_HEAD_CRSA_LAYER,
    Qwen38NativeHeadCrsa,
)
from .provenance import runtime_dependency_versions, runtime_source_manifest
from .semantic_atlas import (
    GraphRevision,
    InterventionIdentity,
    MeasurementReceipt,
    ModelPin,
    NumericSummary,
    PlaceboEffect,
    ProbeIdentity,
    RuntimeProvenance,
    WeightCoordinate,
)


CARTOGRAPHY_EVIDENCE_SCHEMA = "immer.qwen3.8-cartography-evidence/v1"
CARTOGRAPHY_PROMPT_SCHEMA = "immer.qwen3.8-cartography-prompt/v1"
CARTOGRAPHY_BUNDLE_VIEW_SCHEMA = "immer.qwen3.8-cartography-bundle-view/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CODE_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_MODES = frozenset(("passive", "off", "native", "placebo"))
_UNLABELED_FAMILY_SHA256 = hashlib.sha256(b"immer:unlabeled-family/v1").hexdigest()
_NO_LABEL_SOURCE_SHA256 = hashlib.sha256(b"immer:no-label-source/v1").hexdigest()
_MAX_BUNDLE_MANIFEST_BYTES = 64 * 1024**2
_MAX_RUNTIME_SOURCE_BYTES = 64 * 1024**2


class Qwen38CartographyProbeError(RuntimeError):
    """A requested measurement violates the exact cartography contract."""


class Qwen38CartographyIntegrityError(Qwen38CartographyProbeError):
    """Checkpoint, graph, observer, state, or evidence identity changed."""


class Qwen38CartographyBudgetError(Qwen38CartographyProbeError):
    """A probe exceeds an explicit resource bound before execution."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Qwen38CartographyIntegrityError(
            "cartography evidence is not canonical JSON"
        ) from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_regular_bytes(path: Path, *, label: str, max_bytes: int) -> bytes:
    """Read one bounded regular file through a stable ``O_NOFOLLOW`` fd."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    limit = _positive(max_bytes, "max_bytes")
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int):
        raise Qwen38CartographyIntegrityError(
            "local cartography reads require O_NOFOLLOW"
        )
    try:
        linked_before = path.lstat()
    except OSError as exc:
        raise Qwen38CartographyIntegrityError(
            f"cannot inspect {label}: {path}"
        ) from exc
    if (
        not stat.S_ISREG(linked_before.st_mode)
        or linked_before.st_size < 0
        or linked_before.st_size > limit
    ):
        raise Qwen38CartographyIntegrityError(
            f"{label} must be a bounded non-symlink regular file"
        )
    flags = os.O_RDONLY | nofollow
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_size < 0
            or opened.st_size > limit
            or (opened.st_dev, opened.st_ino)
            != (linked_before.st_dev, linked_before.st_ino)
            or opened.st_size != linked_before.st_size
        ):
            raise Qwen38CartographyIntegrityError(
                f"{label} changed while its descriptor was opened"
            )
        remaining = int(opened.st_size)
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                raise Qwen38CartographyIntegrityError(
                    f"{label} ended before its authenticated size"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise Qwen38CartographyIntegrityError(
                f"{label} grew beyond its authenticated size"
            )
        opened_after = os.fstat(descriptor)
        try:
            linked_after = path.lstat()
        except OSError as exc:
            raise Qwen38CartographyIntegrityError(
                f"{label} path changed after its descriptor read"
            ) from exc
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        expected = tuple(getattr(opened, field) for field in stable_fields)
        if (
            tuple(getattr(opened_after, field) for field in stable_fields) != expected
            or tuple(getattr(linked_before, field) for field in stable_fields)
            != expected
            or tuple(getattr(linked_after, field) for field in stable_fields)
            != expected
            or not stat.S_ISREG(linked_after.st_mode)
        ):
            raise Qwen38CartographyIntegrityError(
                f"{label} changed during its descriptor read"
            )
        return b"".join(chunks)
    except Qwen38CartographyIntegrityError:
        raise
    except OSError as exc:
        raise Qwen38CartographyIntegrityError(
            f"cannot read authenticated {label}: {path}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise Qwen38CartographyProbeError(f"{label} must be a lowercase SHA-256")
    return value


def _positive(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Qwen38CartographyProbeError(f"{label} must be a positive integer")
    return value


def _nonnegative(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Qwen38CartographyProbeError(f"{label} must be a non-negative integer")
    return value


def _finite(value: object, label: str, *, lower: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Qwen38CartographyProbeError(f"{label} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result) or (lower is not None and result < lower):
        raise Qwen38CartographyProbeError(f"{label} must be finite numeric data")
    return 0.0 if result == 0.0 else result


def prompt_token_sha256(token_ids: tuple[int, ...] | list[int]) -> str:
    """Hash an exact one-row prompt under the cartography prompt schema."""

    try:
        values = tuple(token_ids)
    except TypeError as exc:
        raise TypeError("token_ids must be an integer sequence") from exc
    if not values:
        raise ValueError("token_ids must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("token_ids must contain integers")
    if any(value < 0 for value in values):
        raise ValueError("token_ids must be non-negative")
    return _digest({"schema": CARTOGRAPHY_PROMPT_SCHEMA, "token_ids": list(values)})


@dataclass(frozen=True, slots=True)
class ProbeResourceBudget:
    """Hard preflight limits for one complete single- or paired-arm probe."""

    max_prompt_tokens: int = 4096
    max_executed_layers: int = 128
    max_source_bytes: int = 1024**4
    max_trace_operations: int = 8192
    max_trace_leaves: int = 8192
    max_sketch_elements: int = 16 * 1024**2
    max_evidence_bytes: int = 64 * 1024**2

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _positive(getattr(self, name), name))

    def as_record(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HiddenSketchProjection:
    """Deterministic Rademacher projection used only for evidence sketches."""

    seed_sha256: str
    output_dimensions: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed_sha256", _sha(self.seed_sha256, "sketch seed"))
        dimensions = _positive(self.output_dimensions, "output_dimensions")
        if dimensions > 256:
            raise Qwen38CartographyProbeError(
                "output_dimensions exceeds the bounded sketch width of 256"
            )
        object.__setattr__(self, "output_dimensions", dimensions)

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProbeCoordinateSpec:
    """Requested semantic coordinate inside one causally bound tensor plan."""

    layer: int
    module: str
    tensor: str
    head_index: int | None = None
    row_start: int | None = None
    row_end: int | None = None
    relative_byte_offset: int = 0
    byte_length: int | None = None

    def __post_init__(self) -> None:
        layer = _nonnegative(self.layer, "coordinate layer")
        for field in ("module", "tensor"):
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or not value
                or value != value.strip()
                or "\x00" in value
                or len(value) > 4096
            ):
                raise Qwen38CartographyProbeError(
                    f"coordinate {field} must be bounded canonical text"
                )
        if self.tensor != self.module and not self.tensor.startswith(f"{self.module}."):
            raise Qwen38CartographyProbeError(
                "coordinate module is not an exact tensor-name prefix"
            )
        marker = f".layers.{layer}."
        if marker not in f".{self.tensor}":
            raise Qwen38CartographyProbeError(
                "coordinate tensor does not identify its requested layer"
            )
        head = self.head_index
        if head is not None:
            head = _nonnegative(head, "head_index")
        if (self.row_start is None) != (self.row_end is None):
            raise Qwen38CartographyProbeError(
                "row_start and row_end must be supplied together"
            )
        row_start = self.row_start
        row_end = self.row_end
        if row_start is not None and row_end is not None:
            row_start = _nonnegative(row_start, "row_start")
            row_end = _positive(row_end, "row_end")
            if row_start >= row_end:
                raise Qwen38CartographyProbeError("row range must be non-empty")
            if self.relative_byte_offset != 0 or self.byte_length is not None:
                raise Qwen38CartographyProbeError(
                    "row and arbitrary byte coordinates are mutually exclusive"
                )
        relative = _nonnegative(self.relative_byte_offset, "relative_byte_offset")
        byte_length = self.byte_length
        if byte_length is not None:
            byte_length = _positive(byte_length, "byte_length")
        object.__setattr__(self, "layer", layer)
        object.__setattr__(self, "head_index", head)
        object.__setattr__(self, "row_start", row_start)
        object.__setattr__(self, "row_end", row_end)
        object.__setattr__(self, "relative_byte_offset", relative)
        object.__setattr__(self, "byte_length", byte_length)

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProbeSpec:
    """One bounded prompt/range/intervention request for the O1 cartographer."""

    prompt_token_ids: tuple[int, ...]
    prompt_sha256: str
    start_layer: int
    stop_layer: int
    coordinate: ProbeCoordinateSpec
    intervention_mode: Literal["passive", "off", "native", "placebo"] | str
    code_revision: str
    budget: ProbeResourceBudget = ProbeResourceBudget()
    hidden_sketch: HiddenSketchProjection | None = None
    question_sha256: str | None = None
    family_sha256: str = _UNLABELED_FAMILY_SHA256
    label_source_sha256: str = _NO_LABEL_SOURCE_SHA256
    native_head_crsa: Qwen38NativeHeadCrsa | None = None

    def __post_init__(self) -> None:
        try:
            tokens = tuple(self.prompt_token_ids)
        except TypeError as exc:
            raise TypeError("prompt_token_ids must be an integer sequence") from exc
        expected_prompt = prompt_token_sha256(tokens)
        claimed_prompt = _sha(self.prompt_sha256, "prompt_sha256")
        if claimed_prompt != expected_prompt:
            raise Qwen38CartographyIntegrityError(
                "prompt token IDs do not match prompt_sha256"
            )
        start = _nonnegative(self.start_layer, "start_layer")
        stop = _positive(self.stop_layer, "stop_layer")
        if start >= stop:
            raise Qwen38CartographyProbeError("layer range must be non-empty")
        if not isinstance(self.coordinate, ProbeCoordinateSpec):
            raise TypeError("coordinate must be a ProbeCoordinateSpec")
        if not start <= self.coordinate.layer < stop:
            raise Qwen38CartographyProbeError(
                "coordinate layer must lie inside the measured layer range"
            )
        mode = self.intervention_mode
        if not isinstance(mode, str) or mode not in _MODES:
            raise Qwen38CartographyProbeError("intervention_mode is invalid")
        if _CODE_REVISION.fullmatch(self.code_revision) is None:
            raise Qwen38CartographyProbeError(
                "code_revision must be a full 40- or 64-digit SHA revision"
            )
        if not isinstance(self.budget, ProbeResourceBudget):
            raise TypeError("budget must be a ProbeResourceBudget")
        if len(tokens) > self.budget.max_prompt_tokens:
            raise Qwen38CartographyBudgetError(
                "prompt exceeds max_prompt_tokens before execution"
            )
        if self.hidden_sketch is not None and not isinstance(
            self.hidden_sketch, HiddenSketchProjection
        ):
            raise TypeError("hidden_sketch must be a HiddenSketchProjection or None")
        question = (
            claimed_prompt
            if self.question_sha256 is None
            else _sha(self.question_sha256, "question_sha256")
        )
        family = _sha(self.family_sha256, "family_sha256")
        label_source = _sha(self.label_source_sha256, "label_source_sha256")
        native = self.native_head_crsa
        if mode == "native":
            if not isinstance(native, Qwen38NativeHeadCrsa) or not native.active:
                raise Qwen38CartographyProbeError(
                    "native mode requires an active Qwen38NativeHeadCrsa"
                )
        elif mode in ("passive", "off") and native is not None:
            raise Qwen38CartographyProbeError(
                "passive/off probes cannot carry a native intervention"
            )
        elif (
            mode == "placebo"
            and native is not None
            and not isinstance(native, Qwen38NativeHeadCrsa)
        ):
            raise TypeError("placebo native_head_crsa has an invalid type")
        object.__setattr__(self, "prompt_token_ids", tokens)
        object.__setattr__(self, "prompt_sha256", claimed_prompt)
        object.__setattr__(self, "start_layer", start)
        object.__setattr__(self, "stop_layer", stop)
        object.__setattr__(self, "intervention_mode", mode)
        object.__setattr__(self, "question_sha256", question)
        object.__setattr__(self, "family_sha256", family)
        object.__setattr__(self, "label_source_sha256", label_source)

    @property
    def probe_identity(self) -> ProbeIdentity:
        assert self.question_sha256 is not None
        return ProbeIdentity(
            question_sha256=self.question_sha256,
            token_sha256=self.prompt_sha256,
            family_sha256=self.family_sha256,
            label_source_sha256=self.label_source_sha256,
        )

    def as_record(self) -> dict[str, Any]:
        return {
            "budget": self.budget.as_record(),
            "code_revision": self.code_revision,
            "coordinate": self.coordinate.as_record(),
            "family_sha256": self.family_sha256,
            "hidden_sketch": (
                None if self.hidden_sketch is None else self.hidden_sketch.as_record()
            ),
            "intervention_mode": self.intervention_mode,
            "label_source_sha256": self.label_source_sha256,
            "native_head_crsa": (
                None if self.native_head_crsa is None else asdict(self.native_head_crsa)
            ),
            "prompt_sha256": self.prompt_sha256,
            "prompt_token_ids": list(self.prompt_token_ids),
            "question_sha256": self.question_sha256,
            "start_layer": self.start_layer,
            "stop_layer": self.stop_layer,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())


@dataclass(frozen=True, slots=True)
class TensorRangeReceipt:
    """One exact causal tensor-plan subrange observed by an execution arm."""

    arm: str
    operation_index: int
    tensor: str
    tensor_plan_sha256: str
    shard: str
    absolute_offset: int
    relative_offset: int
    length: int
    source_requests: int
    source_bytes: int
    cache_hits: int | None
    access_operation_sha256: str

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CartographyProbeResult:
    """Primary measurement plus optional paired control and raw proof objects."""

    measurement: MeasurementReceipt
    evidence_document: dict[str, Any]
    access_trace: AccessTrace
    tensor_range_receipts: tuple[TensorRangeReceipt, ...]
    control_measurement: MeasurementReceipt | None = None
    control_evidence_document: dict[str, Any] | None = None
    control_access_trace: AccessTrace | None = None
    control_tensor_range_receipts: tuple[TensorRangeReceipt, ...] = ()

    @property
    def measurements_in_append_order(self) -> tuple[MeasurementReceipt, ...]:
        """Return placebo/control first so atlas effect references are resolvable."""

        if self.control_measurement is None:
            return (self.measurement,)
        return (self.control_measurement, self.measurement)

    def verify(self) -> None:
        """Re-parse every sealed object and bind raw proofs to receipt hashes."""

        measured = MeasurementReceipt.from_document(self.measurement.to_document())
        self.access_trace.verify()
        _verify_evidence_document(self.evidence_document)
        if measured.access_trace_sha256 != self.access_trace.sha256:
            raise Qwen38CartographyIntegrityError(
                "measurement access trace does not match its raw proof"
            )
        if measured.evidence_sha256 != self.evidence_document["sha256"]:
            raise Qwen38CartographyIntegrityError(
                "measurement evidence digest does not match its document"
            )
        if self.control_measurement is None:
            if (
                any(
                    value is not None
                    for value in (
                        self.control_evidence_document,
                        self.control_access_trace,
                    )
                )
                or self.control_tensor_range_receipts
            ):
                raise Qwen38CartographyIntegrityError(
                    "unpaired result retains control proof material"
                )
            return
        if self.control_evidence_document is None or self.control_access_trace is None:
            raise Qwen38CartographyIntegrityError(
                "paired result is missing control proof material"
            )
        control = MeasurementReceipt.from_document(
            self.control_measurement.to_document()
        )
        self.control_access_trace.verify()
        _verify_evidence_document(self.control_evidence_document)
        if control.access_trace_sha256 != self.control_access_trace.sha256:
            raise Qwen38CartographyIntegrityError(
                "control access trace does not match its raw proof"
            )
        if control.evidence_sha256 != self.control_evidence_document["sha256"]:
            raise Qwen38CartographyIntegrityError(
                "control evidence digest does not match its document"
            )


@dataclass(slots=True)
class _ArmCapture:
    arm: str
    intervention_mode: str
    intervention_configuration: dict[str, Any]
    layers: list[dict[str, Any]]
    hidden_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]]
    delta_probes: list[dict[str, Any]]
    crsa_evidence: list[dict[str, Any]]
    final_hidden_sha256: str
    final_state_sha256: str
    activation_sha256: str
    summaries: list[NumericSummary]
    access_trace: AccessTrace
    tensor_receipts: tuple[TensorRangeReceipt, ...]
    source_body_bytes: int


def _seal_evidence(body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(_canonical(dict(body)))
    return {
        "body": normalized,
        "schema": CARTOGRAPHY_EVIDENCE_SCHEMA,
        "sha256": _digest(normalized),
    }


def _verify_evidence_document(document: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(document, Mapping) or set(document) != {
        "body",
        "schema",
        "sha256",
    }:
        raise Qwen38CartographyIntegrityError("evidence document shape is invalid")
    if document.get("schema") != CARTOGRAPHY_EVIDENCE_SCHEMA:
        raise Qwen38CartographyIntegrityError("evidence document schema is invalid")
    body = document.get("body")
    if not isinstance(body, Mapping) or document.get("sha256") != _digest(body):
        raise Qwen38CartographyIntegrityError("evidence document SHA-256 mismatch")
    return json.loads(_canonical(dict(body)))


def _tensor_record(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach().contiguous().to(device="cpu")
    if not detached.is_floating_point():
        raise Qwen38CartographyIntegrityError(
            "cartography state contains a non-floating tensor"
        )
    raw = detached.view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(detached.dtype).removeprefix("torch."),
        "shape": list(detached.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _hidden_statistics(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach().contiguous().to(device="cpu", dtype=torch.float64)
    flat = detached.reshape(-1)
    if flat.numel() == 0 or not bool(torch.isfinite(flat).all().item()):
        raise Qwen38CartographyIntegrityError(
            "hidden measurement is empty or non-finite"
        )
    count = int(flat.numel())
    total = float(flat.sum().item())
    total_squares = float(flat.square().sum().item())
    mean = total / count
    variance = max(0.0, total_squares / count - mean * mean)
    tensor = _tensor_record(value)
    return {
        "count": count,
        "maximum": float(flat.max().item()),
        "mean": mean,
        "minimum": float(flat.min().item()),
        "norm": math.sqrt(total_squares),
        "sha256": tensor["sha256"],
        "std": math.sqrt(variance),
        "total": total,
        "total_squares": total_squares,
    }


def _numeric_summary(metric: str, stats: Mapping[str, Any]) -> NumericSummary:
    return NumericSummary(
        metric=metric,
        count=int(stats["count"]),
        total=float(stats["total"]),
        total_squares=float(stats["total_squares"]),
        minimum=float(stats["minimum"]),
        maximum=float(stats["maximum"]),
    )


def _scalar_summary(metric: str, value: float) -> NumericSummary:
    normalized = _finite(value, metric)
    return NumericSummary(
        metric=metric,
        count=1,
        total=normalized,
        total_squares=normalized * normalized,
        minimum=normalized,
        maximum=normalized,
    )


def _state_record(state: LayerState) -> dict[str, Any]:
    if isinstance(state, AttentionState):
        tensors = {
            "key": _tensor_record(state.key),
            "value": _tensor_record(state.value),
        }
        if state.crsa_log_usage is not None:
            tensors["crsa_log_usage"] = _tensor_record(state.crsa_log_usage)
        kind = "attention"
    elif isinstance(state, DeltaNetState):
        tensors = {
            "conv": _tensor_record(state.conv),
            "recurrent": _tensor_record(state.recurrent),
        }
        kind = "deltanet"
    else:  # pragma: no cover - LayerState exhaustiveness.
        raise Qwen38CartographyIntegrityError("layer returned an invalid state kind")
    body = {"kind": kind, "tensors": tensors}
    return {**body, "sha256": _digest(body)}


def _model_state_stamp(model: StreamedQwen38) -> str:
    states: list[Any] = []
    for state in model._layer_states:
        states.append(None if state is None else _state_record(state))
    body = {
        "graft_history": (
            None
            if model._graft_history is None
            else _tensor_record(model._graft_history)
        ),
        "layer_states": states,
        "next_position": model.next_position,
        "pending_block": model._pending_block_stage is not None,
        "state_batch_size": model.state_batch_size,
        "state_poisoned": model.state_poisoned,
    }
    return _digest(body)


def _model_attachment_stamp(model: StreamedQwen38) -> str:
    native = None if model.native_head_crsa is None else asdict(model.native_head_crsa)
    body = {
        "config": asdict(model.config),
        "delta_probe_id": None if model.delta_probe is None else id(model.delta_probe),
        "graft_id": None if model.graft is None else id(model.graft),
        "graft_layer": model.graft_layer,
        "max_batch_size": model.max_batch_size,
        "max_seq_len": model.max_seq_len,
        "native_head_crsa": native,
        "native_observer_id": (
            None
            if model.native_head_crsa_observer is None
            else id(model.native_head_crsa_observer)
        ),
        "pager_id": id(model.pager),
        "source_id": id(model.pager.source),
    }
    return _digest(body)


def _sketch_record(
    value: torch.Tensor,
    projection: HiddenSketchProjection,
    *,
    max_elements: int,
) -> dict[str, Any]:
    rows = (
        value.detach()
        .contiguous()
        .to(device="cpu", dtype=torch.float64)
        .reshape(-1, value.shape[-1])
    )
    matrix_elements = int(value.shape[-1]) * projection.output_dimensions
    output_elements = int(rows.shape[0]) * projection.output_dimensions
    if matrix_elements + output_elements > max_elements:
        raise Qwen38CartographyBudgetError(
            "hidden sketch exceeds max_sketch_elements before allocation"
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(projection.seed_sha256[:16], 16))
    signs = torch.randint(
        0,
        2,
        (value.shape[-1], projection.output_dimensions),
        dtype=torch.int8,
        generator=generator,
    ).to(dtype=torch.float64)
    signs.mul_(2.0).sub_(1.0)
    projected = rows.matmul(signs).div_(math.sqrt(value.shape[-1]))
    stats = _hidden_statistics(projected)
    return {
        "output_dimensions": projection.output_dimensions,
        "seed_sha256": projection.seed_sha256,
        "statistics": stats,
    }


def _paired_statistics(observed: torch.Tensor, control: torch.Tensor) -> dict[str, Any]:
    left = observed.detach().contiguous().to(device="cpu", dtype=torch.float64)
    right = control.detach().contiguous().to(device="cpu", dtype=torch.float64)
    if tuple(left.shape) != tuple(right.shape):
        raise Qwen38CartographyIntegrityError("paired hidden shapes differ")
    delta = left - right
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    delta_flat = delta.reshape(-1)
    left_norm = float(torch.linalg.vector_norm(left_flat).item())
    right_norm = float(torch.linalg.vector_norm(right_flat).item())
    if left_norm == 0.0 or right_norm == 0.0:
        cosine = 1.0 if torch.equal(left_flat, right_flat) else 0.0
    else:
        cosine = float(torch.dot(left_flat, right_flat).item()) / (
            left_norm * right_norm
        )
        cosine = max(-1.0, min(1.0, cosine))
    return {
        "cosine": cosine,
        "delta_l2": float(torch.linalg.vector_norm(delta_flat).item()),
        "delta_max_abs": float(delta_flat.abs().max().item()),
        "delta_mean_abs": float(delta_flat.abs().mean().item()),
        "exact_equal": bool(torch.equal(left, right)),
    }


def _layer_tensor_names(model: StreamedQwen38, layer: int) -> tuple[str, ...]:
    base = f"model.language_model.layers.{layer}"
    names = [
        f"{base}.input_layernorm.weight",
        f"{base}.post_attention_layernorm.weight",
        f"{base}.mlp.gate_proj.weight",
        f"{base}.mlp.up_proj.weight",
        f"{base}.mlp.down_proj.weight",
    ]
    if model.config.is_full_attention(layer):
        attn = f"{base}.self_attn"
        names.extend(
            (
                f"{attn}.q_proj.weight",
                f"{attn}.k_proj.weight",
                f"{attn}.v_proj.weight",
                f"{attn}.o_proj.weight",
                f"{attn}.q_norm.weight",
                f"{attn}.k_norm.weight",
            )
        )
    else:
        attn = f"{base}.linear_attn"
        names.extend(
            (
                f"{attn}.in_proj_qkv.weight",
                f"{attn}.in_proj_z.weight",
                f"{attn}.in_proj_a.weight",
                f"{attn}.in_proj_b.weight",
                f"{attn}.conv1d.weight",
                f"{attn}.A_log",
                f"{attn}.dt_bias",
                f"{attn}.norm.weight",
                f"{attn}.out_proj.weight",
            )
        )
    return tuple(names)


def _graph_revision(model: StreamedQwen38) -> tuple[int, str]:
    reader = model.pager.causal_tensor_reader
    if reader is None or reader.source is not model.pager.source:
        raise Qwen38CartographyIntegrityError(
            "cartography requires this model's exact causal tensor reader"
        )
    revision = reader.graph.store.revision()
    if (
        not isinstance(revision, tuple)
        or len(revision) != 2
        or isinstance(revision[0], bool)
        or not isinstance(revision[0], int)
        or revision[0] < 0
        or not isinstance(revision[1], str)
        or _SHA256.fullmatch(revision[1]) is None
    ):
        raise Qwen38CartographyIntegrityError("causal graph revision is invalid")
    return revision


def _source_identity(model: StreamedQwen38) -> dict[str, str]:
    source = model.pager.source
    source.inventory()
    metrics = source.metrics()
    identity = {
        "repo_id": metrics.get("repo_id"),
        "revision": metrics.get("revision"),
        "bundle_fingerprint": metrics.get("inventory_source_fingerprint"),
    }
    if (
        not isinstance(identity["repo_id"], str)
        or not identity["repo_id"]
        or not isinstance(identity["revision"], str)
        or not identity["revision"]
        or not isinstance(identity["bundle_fingerprint"], str)
        or _SHA256.fullmatch(identity["bundle_fingerprint"]) is None
    ):
        raise Qwen38CartographyIntegrityError(
            "checkpoint source has no immutable inventory identity"
        )
    return identity  # type: ignore[return-value]


def _model_pin(
    model: StreamedQwen38,
    *,
    code_revision: str,
    bundle_manifest_sha256: str,
) -> ModelPin:
    source = _source_identity(model)
    return ModelPin(
        repo_id=source["repo_id"],
        revision=source["revision"],
        bundle_fingerprint=source["bundle_fingerprint"],
        bundle_manifest_sha256=_sha(bundle_manifest_sha256, "bundle_manifest_sha256"),
        code_revision=code_revision,
    )


def _bundle_manifest_sha256(model: StreamedQwen38) -> str:
    """Read and authenticate the real mounted Qwen ``bundle.json`` identity."""

    upstream = getattr(model.pager.source.reader, "upstream", None)
    weights_root = getattr(upstream, "root", None)
    if not isinstance(weights_root, Path):
        raise Qwen38CartographyIntegrityError(
            "cartography requires a local causal bundle manifest"
        )
    candidates = (weights_root / "bundle.json", weights_root.parent / "bundle.json")
    existing: list[Path] = []
    for path in candidates:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise Qwen38CartographyIntegrityError(
                f"cannot inspect causal bundle manifest: {path}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise Qwen38CartographyIntegrityError(
                "causal bundle manifest must be a non-symlink regular file"
            )
        existing.append(path)
    if len(existing) != 1:
        raise Qwen38CartographyIntegrityError(
            "exactly one mounted causal bundle.json must be discoverable"
        )
    path = existing[0]
    try:
        raw = _read_regular_bytes(
            path,
            label="causal bundle manifest",
            max_bytes=_MAX_BUNDLE_MANIFEST_BYTES,
        )
        document = json.loads(raw.decode("utf-8"))
    except Qwen38CartographyIntegrityError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise Qwen38CartographyIntegrityError(
            "causal bundle manifest is unreadable"
        ) from exc
    if (
        not isinstance(document, Mapping)
        or set(document) != {"body", "schema", "sha256"}
        or document.get("schema") != QWEN38_BUNDLE_SCHEMA
        or not isinstance(document.get("body"), Mapping)
        or document.get("sha256") != _digest(document["body"])
    ):
        raise Qwen38CartographyIntegrityError(
            "causal bundle manifest identity is invalid"
        )
    body = document["body"]
    source = _source_identity(model)
    logical_model = {"repo_id": source["repo_id"], "revision": source["revision"]}
    sealed_graph_revision = body.get("graph_revision")
    if (
        body.get("checkpoint_complete") is not True
        or body.get("layout_fingerprint") != source["bundle_fingerprint"]
        or body.get("logical_model") != logical_model
        or not isinstance(sealed_graph_revision, list)
        or len(sealed_graph_revision) != 2
        or isinstance(sealed_graph_revision[0], bool)
        or not isinstance(sealed_graph_revision[0], int)
        or sealed_graph_revision[0] < 0
        or not isinstance(sealed_graph_revision[1], str)
        or _SHA256.fullmatch(sealed_graph_revision[1]) is None
    ):
        raise Qwen38CartographyIntegrityError(
            "causal bundle manifest is stale for the mounted model/layout"
        )
    expected_layout = "flat/v1" if path.parent == weights_root else "nested/v1"
    if body.get("weights_layout", "nested/v1") != expected_layout:
        raise Qwen38CartographyIntegrityError(
            "causal bundle manifest weights layout disagrees with its mount"
        )
    return str(document["sha256"])


def _file_sha256(path: Path) -> str:
    raw = _read_regular_bytes(
        path,
        label="cartography runtime source",
        max_bytes=_MAX_RUNTIME_SOURCE_BYTES,
    )
    return hashlib.sha256(raw).hexdigest()


def _runtime_provenance(
    model: StreamedQwen38,
    spec: ProbeSpec,
    *,
    preflight: Mapping[str, Any],
    graph_revision: tuple[int, str],
) -> RuntimeProvenance:
    source_manifest = runtime_source_manifest(include_transport=True)
    module_root = Path(__file__).resolve().parent
    source_manifest.extend(
        (
            {
                "path": "immer/runtimes/qwen3_8/cartography_probe.py",
                "sha256": _file_sha256(module_root / "cartography_probe.py"),
            },
            {
                "path": "immer/runtimes/qwen3_8/semantic_atlas.py",
                "sha256": _file_sha256(module_root / "semantic_atlas.py"),
            },
        )
    )
    dependencies = runtime_dependency_versions()
    runtime_configuration = {
        "config": asdict(model.config),
        "device": str(model.pager.device),
        "dtype": str(model.pager.compute_dtype).removeprefix("torch."),
        "graph_revision": list(graph_revision),
        "preflight": dict(preflight),
        "probe_spec": spec.as_record(),
    }
    platform_record = {
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "torch": torch.__version__,
    }
    return RuntimeProvenance(
        code_revision=spec.code_revision,
        source_manifest_sha256=_digest(source_manifest),
        dependency_manifest_sha256=_digest(dependencies),
        runtime_configuration_sha256=_digest(runtime_configuration),
        platform_sha256=_digest(platform_record),
    )


def _coordinate(model: StreamedQwen38, spec: ProbeSpec) -> WeightCoordinate:
    reader = model.pager.causal_tensor_reader
    assert reader is not None
    plan = reader.resolve_tensor_plan(spec.coordinate.tensor)
    kwargs: dict[str, Any] = {
        "layer": spec.coordinate.layer,
        "module": spec.coordinate.module,
        "head_index": spec.coordinate.head_index,
    }
    if spec.coordinate.row_start is not None:
        kwargs["row_start"] = spec.coordinate.row_start
        kwargs["row_end"] = spec.coordinate.row_end
    else:
        kwargs["relative_byte_offset"] = spec.coordinate.relative_byte_offset
        kwargs["byte_length"] = spec.coordinate.byte_length
    coordinate = WeightCoordinate.from_plan(plan, **kwargs)
    if spec.coordinate.head_index is not None:
        # The current exact runtime exposes the fixed four-head native hook as
        # one atomic group.  Claiming one head would overstate probe precision.
        raise Qwen38CartographyProbeError(
            "head-specific intervention is unsupported by the current exact hook"
        )
    return coordinate


def _validate_intervention(model: StreamedQwen38, spec: ProbeSpec) -> None:
    if spec.intervention_mode not in ("native", "placebo"):
        return
    native = spec.native_head_crsa
    if spec.intervention_mode == "placebo" and native is None:
        return
    assert native is not None
    if (
        native.layer != NATIVE_HEAD_CRSA_LAYER
        or spec.coordinate.layer != native.layer
        or not spec.coordinate.module.endswith(f".layers.{native.layer}.self_attn")
        or spec.coordinate.tensor.rsplit(".", 2)[0] != spec.coordinate.module
    ):
        raise Qwen38CartographyProbeError(
            "native intervention coordinate must be a layer-27 self-attention tensor"
        )
    if native.layer >= model.config.n_layers or not model.config.is_full_attention(
        native.layer
    ):
        raise Qwen38CartographyProbeError(
            "checkpoint does not expose the validated native intervention layer"
        )
    if model.config.n_heads != 24 or model.config.n_kv_heads != 4:
        raise Qwen38CartographyProbeError(
            "checkpoint does not expose the validated 24-query/4-KV layout"
        )


def _preflight_plans(
    model: StreamedQwen38,
    spec: ProbeSpec,
    *,
    arm_count: int,
) -> tuple[dict[str, Any], int, int, int]:
    if spec.stop_layer > model.config.n_layers:
        raise Qwen38CartographyProbeError("probe layer range exceeds model depth")
    if len(spec.prompt_token_ids) > model.max_seq_len:
        raise Qwen38CartographyBudgetError(
            "probe prompt exceeds model max_seq_len before execution"
        )
    if any(token >= model.config.vocab_size for token in spec.prompt_token_ids):
        raise Qwen38CartographyProbeError(
            "probe prompt token is outside checkpoint vocabulary"
        )
    executed_layers = arm_count * spec.stop_layer
    if executed_layers > spec.budget.max_executed_layers:
        raise Qwen38CartographyBudgetError(
            "probe exceeds max_executed_layers before execution"
        )
    preflight = model.checkpoint_preflight()
    reader = model.pager.causal_tensor_reader
    assert reader is not None
    names = [model.EMBED_NAME]
    for layer in range(spec.stop_layer):
        names.extend(_layer_tensor_names(model, layer))
    if spec.coordinate.tensor not in names:
        raise Qwen38CartographyProbeError(
            "coordinate tensor is not consumed by the requested execution range"
        )
    plans = {name: reader.resolve_tensor_plan(name) for name in names}
    embed = plans[model.EMBED_NAME]
    if len(embed.shape) != 2:
        raise Qwen38CartographyIntegrityError("embedding tensor plan is not 2D")
    row_bytes = embed.length // embed.shape[0]
    embed_bytes = len(set(spec.prompt_token_ids)) * row_bytes
    layer_bytes = sum(plans[name].length for name in names if name != model.EMBED_NAME)
    planned_source_bytes = arm_count * (embed_bytes + layer_bytes)
    runs = model.pager._consecutive_runs(spec.prompt_token_ids)
    planned_operations = arm_count * (len(runs) + len(names) - 1)
    if planned_source_bytes > spec.budget.max_source_bytes:
        raise Qwen38CartographyBudgetError(
            "probe exceeds max_source_bytes before execution"
        )
    if planned_operations > spec.budget.max_trace_operations:
        raise Qwen38CartographyBudgetError(
            "probe exceeds max_trace_operations before execution"
        )
    if planned_operations > spec.budget.max_trace_leaves:
        raise Qwen38CartographyBudgetError(
            "probe exceeds max_trace_leaves before execution"
        )
    return preflight, planned_source_bytes, planned_operations, len(names)


def _trace_receipts(
    trace: AccessTrace,
    *,
    model: StreamedQwen38,
    spec: ProbeSpec,
    arm: str,
) -> tuple[TensorRangeReceipt, ...]:
    reader = model.pager.causal_tensor_reader
    assert reader is not None
    names = [model.EMBED_NAME]
    for layer in range(spec.stop_layer):
        names.extend(_layer_tensor_names(model, layer))
    plans = {name: reader.resolve_tensor_plan(name) for name in names}
    rows: list[TensorRangeReceipt] = []
    observed_names: set[str] = set()
    for operation_index, operation in enumerate(trace.operations):
        tags = dict(operation.tags)
        if tags != {"arm": arm, "probe_sha256": spec.sha256}:
            raise Qwen38CartographyIntegrityError(
                "access trace contains an operation outside the probe scope"
            )
        operation_sha = _digest(operation.to_document())
        for leaf in operation.leaves:
            matches = [
                plan
                for plan in plans.values()
                if plan.shard == leaf.shard
                and plan.absolute_offset <= leaf.offset
                and leaf.offset + leaf.length <= plan.absolute_end
            ]
            if len(matches) != 1:
                raise Qwen38CartographyIntegrityError(
                    "access leaf does not map to one exact requested tensor plan"
                )
            plan = matches[0]
            observed_names.add(plan.name)
            plan_sha = _digest(
                {
                    "absolute_offset": plan.absolute_offset,
                    "dtype": plan.dtype,
                    "length": plan.length,
                    "name": plan.name,
                    "shape": list(plan.shape),
                    "shard": plan.shard,
                }
            )
            rows.append(
                TensorRangeReceipt(
                    arm=arm,
                    operation_index=operation_index,
                    tensor=plan.name,
                    tensor_plan_sha256=plan_sha,
                    shard=leaf.shard,
                    absolute_offset=leaf.offset,
                    relative_offset=leaf.offset - plan.absolute_offset,
                    length=leaf.length,
                    source_requests=operation.source_requests,
                    source_bytes=operation.source_bytes,
                    cache_hits=operation.cache_hits,
                    access_operation_sha256=operation_sha,
                )
            )
    if observed_names != set(names):
        missing = sorted(set(names) - observed_names)
        raise Qwen38CartographyIntegrityError(
            "access trace omits requested tensor reads: " + ", ".join(missing[:4])
        )
    if spec.coordinate.tensor not in observed_names:
        raise Qwen38CartographyIntegrityError(
            "coordinate tensor has no exact access-trace receipt"
        )
    return tuple(rows)


class _DeltaRecorder:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def __call__(self, layer: int, probe: DeltaNetProbe) -> None:
        if not isinstance(probe, DeltaNetProbe):
            raise TypeError("DeltaNet probe callback received an invalid row")
        self.rows.append({"layer": layer, **asdict(probe)})


def _arm_intervention(
    spec: ProbeSpec, arm: str
) -> tuple[Qwen38NativeHeadCrsa | None, str, dict[str, Any]]:
    if arm == "native":
        assert spec.native_head_crsa is not None
        native = spec.native_head_crsa
        return native, "native", asdict(native)
    if arm == "placebo" and spec.native_head_crsa is not None:
        native = replace(spec.native_head_crsa, alpha=0.0)
        return native, "placebo", asdict(native)
    mode = arm if arm in ("passive", "off", "placebo") else "off"
    return None, mode, {"kind": "original-qwen-identity", "alpha": 0.0}


def _execute_arm(
    model: StreamedQwen38,
    spec: ProbeSpec,
    *,
    arm: str,
) -> _ArmCapture:
    native, intervention_mode, intervention_configuration = _arm_intervention(spec, arm)
    delta = _DeltaRecorder()
    child = StreamedQwen38(
        model.config,
        model.pager,
        delta_probe=delta,
        native_head_crsa=native,
        max_batch_size=1,
        max_seq_len=model.max_seq_len,
    )
    source = model.pager.source
    recorder = AccessTraceRecorder(
        max_operations=spec.budget.max_trace_operations,
        max_leaves=spec.budget.max_trace_leaves,
    )
    observer_before = source.metrics()
    previous_observer = source.set_access_observer(recorder, prepare_identity=True)
    start_bytes = int(source.metrics().get("network_or_source_body_bytes", 0))
    hidden_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    layer_records: list[dict[str, Any]] = []
    summaries: list[NumericSummary] = []
    crsa_rows: list[dict[str, Any]] = []
    restored_observer: Any | None = None
    try:
        with recorder.scope(arm=arm, probe_sha256=spec.sha256):
            hidden = child.embed_batch((spec.prompt_token_ids,))
            states: tuple[LayerState | None, ...] = tuple(
                None for _ in range(model.config.n_layers)
            )
            for layer in range(spec.stop_layer):
                pre = hidden.detach().clone()
                result = child.hidden_stateful_range(
                    hidden,
                    states,
                    start_pos=0,
                    start_layer=layer,
                    stop_layer=layer + 1,
                    native_head_crsa_tokenwise_usage=True,
                )
                hidden = result.hidden
                states = result.layer_states
                crsa_rows.extend(
                    row.to_dict() for row in result.native_head_crsa_evidence
                )
                if layer < spec.start_layer:
                    continue
                post = hidden.detach().clone()
                state = states[layer]
                if state is None:  # pragma: no cover - stateful range contract.
                    raise Qwen38CartographyIntegrityError(
                        "measured layer returned no staged continuation state"
                    )
                pre_stats = _hidden_statistics(pre)
                post_stats = _hidden_statistics(post)
                state_record = _state_record(state)
                record: dict[str, Any] = {
                    "evidence": asdict(result.evidence),
                    "layer": layer,
                    "post_hidden": post_stats,
                    "pre_hidden": pre_stats,
                    "state": state_record,
                }
                if spec.hidden_sketch is not None:
                    record["pre_sketch"] = _sketch_record(
                        pre,
                        spec.hidden_sketch,
                        max_elements=spec.budget.max_sketch_elements,
                    )
                    record["post_sketch"] = _sketch_record(
                        post,
                        spec.hidden_sketch,
                        max_elements=spec.budget.max_sketch_elements,
                    )
                layer_records.append(record)
                hidden_by_layer[layer] = (pre, post)
                summaries.extend(
                    (
                        _numeric_summary(f"layer.{layer}.pre.hidden", pre_stats),
                        _numeric_summary(f"layer.{layer}.post.hidden", post_stats),
                    )
                )
        trace = recorder.snapshot()
    finally:
        restored_observer = source.set_access_observer(
            previous_observer, prepare_identity=False
        )
        model.pager.release()
    if restored_observer is not recorder:
        raise Qwen38CartographyIntegrityError(
            "source access observer changed during probe execution"
        )
    observer_after = source.metrics()
    for metric in ("access_observer_drops", "access_observer_errors"):
        before = int(observer_before.get(metric, 0))
        after = int(observer_after.get(metric, 0))
        if after != before:
            raise Qwen38CartographyIntegrityError(
                f"source observer reported {metric} during probe execution"
            )
    recorder_metrics = recorder.metrics()
    if recorder_metrics["dropped_capacity"] or recorder_metrics["dropped_identity"]:
        raise Qwen38CartographyIntegrityError(
            "access trace recorder dropped probe operations"
        )
    source_body_bytes = (
        int(source.metrics().get("network_or_source_body_bytes", 0)) - start_bytes
    )
    if source_body_bytes < 0 or source_body_bytes > spec.budget.max_source_bytes:
        raise Qwen38CartographyBudgetError(
            "actual probe source traffic exceeds max_source_bytes"
        )
    tensor_receipts = _trace_receipts(trace, model=model, spec=spec, arm=arm)
    final_hidden = hidden_by_layer[spec.stop_layer - 1][1]
    final_state_records = [
        None if state is None else _state_record(state) for state in states
    ]
    final_state_sha = _digest(final_state_records)
    activation_body = [
        {
            "layer": row["layer"],
            "post_hidden_sha256": row["post_hidden"]["sha256"],
            "post_sketch_sha256": (
                None
                if "post_sketch" not in row
                else row["post_sketch"]["statistics"]["sha256"]
            ),
            "pre_hidden_sha256": row["pre_hidden"]["sha256"],
            "pre_sketch_sha256": (
                None
                if "pre_sketch" not in row
                else row["pre_sketch"]["statistics"]["sha256"]
            ),
        }
        for row in layer_records
    ]
    return _ArmCapture(
        arm=arm,
        intervention_mode=intervention_mode,
        intervention_configuration=intervention_configuration,
        layers=layer_records,
        hidden_by_layer=hidden_by_layer,
        delta_probes=delta.rows,
        crsa_evidence=crsa_rows,
        final_hidden_sha256=_tensor_record(final_hidden)["sha256"],
        final_state_sha256=final_state_sha,
        activation_sha256=_digest(activation_body),
        summaries=summaries,
        access_trace=trace,
        tensor_receipts=tensor_receipts,
        source_body_bytes=source_body_bytes,
    )


def _pair_capture(observed: _ArmCapture, control: _ArmCapture) -> list[dict[str, Any]]:
    if set(observed.hidden_by_layer) != set(control.hidden_by_layer):
        raise Qwen38CartographyIntegrityError("paired arms cover different layers")
    rows: list[dict[str, Any]] = []
    for layer in sorted(observed.hidden_by_layer):
        observed_pre, observed_post = observed.hidden_by_layer[layer]
        control_pre, control_post = control.hidden_by_layer[layer]
        pre = _paired_statistics(observed_pre, control_pre)
        post = _paired_statistics(observed_post, control_post)
        rows.append({"layer": layer, "post": post, "pre": pre})
        for boundary, values in (("pre", pre), ("post", post)):
            for metric in (
                "cosine",
                "delta_l2",
                "delta_max_abs",
                "delta_mean_abs",
            ):
                observed.summaries.append(
                    _scalar_summary(
                        f"layer.{layer}.{boundary}.{metric}_to_control",
                        float(values[metric]),
                    )
                )
    return rows


def _evidence_body(
    capture: _ArmCapture,
    *,
    spec: ProbeSpec,
    model_pin: ModelPin,
    coordinate: WeightCoordinate,
    graph_revision: tuple[int, str],
    preflight: Mapping[str, Any],
    paired: list[dict[str, Any]] | None,
    control_measurement_sha256: str | None,
    weight_rail_revision: GraphRevision,
    atlas_head_revision: GraphRevision,
) -> dict[str, Any]:
    return {
        "access_trace": capture.access_trace.to_document(),
        "arm": capture.arm,
        "atlas_head_revision": atlas_head_revision.to_document(),
        "control_measurement_sha256": control_measurement_sha256,
        "coordinate": coordinate.to_document(),
        "crsa_evidence": capture.crsa_evidence,
        "deltanet_probes": capture.delta_probes,
        "final_hidden_sha256": capture.final_hidden_sha256,
        "final_state_sha256": capture.final_state_sha256,
        "graph_revision": {
            "sequence": graph_revision[0],
            "sha256": graph_revision[1],
        },
        "intervention_configuration": capture.intervention_configuration,
        "intervention_mode": capture.intervention_mode,
        "layers": capture.layers,
        "model_pin": model_pin.to_document(),
        "paired_comparisons": [] if paired is None else paired,
        "preflight": dict(preflight),
        "probe_spec": spec.as_record(),
        "semantic_label_source": "none-model-output-is-untrusted",
        "source_body_bytes": capture.source_body_bytes,
        "tensor_range_receipts": [row.as_record() for row in capture.tensor_receipts],
        "weight_rail_revision": weight_rail_revision.to_document(),
    }


def _measurement(
    capture: _ArmCapture,
    *,
    evidence: Mapping[str, Any],
    model_pin: ModelPin,
    coordinate: WeightCoordinate,
    probe: ProbeIdentity,
    runtime: RuntimeProvenance,
    weight_rail_revision: GraphRevision,
    atlas_head_revision: GraphRevision,
    placebo_effects: tuple[PlaceboEffect, ...] = (),
) -> MeasurementReceipt:
    return MeasurementReceipt(
        model_pin=model_pin,
        coordinate=coordinate,
        probe=probe,
        intervention=InterventionIdentity(
            mode=capture.intervention_mode,
            configuration_sha256=_digest(capture.intervention_configuration),
        ),
        observation_status="recorded",
        observed_semantic_label=None,
        hidden_sha256=capture.final_hidden_sha256,
        activation_sha256=capture.activation_sha256,
        logits_sha256=_digest(
            {"measured": False, "reason": "semantic-output-is-not-probe-evidence"}
        ),
        state_sha256=capture.final_state_sha256,
        access_trace_sha256=capture.access_trace.sha256,
        evidence_sha256=str(evidence["sha256"]),
        weight_rail_revision=weight_rail_revision,
        atlas_head_revision=atlas_head_revision,
        numeric_summaries=tuple(capture.summaries),
        placebo_effects=placebo_effects,
        runtime=runtime,
    )


class Qwen38CartographyProbe:
    """Execute bounded passive or paired exact probes over one causal Qwen."""

    def __init__(self, model: StreamedQwen38) -> None:
        if not isinstance(model, StreamedQwen38):
            raise TypeError("model must be a StreamedQwen38")
        if model.pager.causal_tensor_reader is None:
            raise Qwen38CartographyIntegrityError(
                "cartography requires a causal tensor reader"
            )
        self.model = model

    def execute(
        self,
        spec: ProbeSpec,
        *,
        atlas_head_revision: GraphRevision,
    ) -> CartographyProbeResult:
        if not isinstance(spec, ProbeSpec):
            raise TypeError("spec must be a ProbeSpec")
        if not isinstance(atlas_head_revision, GraphRevision):
            raise TypeError("atlas_head_revision must be a GraphRevision")
        model = self.model
        state_before = _model_state_stamp(model)
        attachments_before = _model_attachment_stamp(model)
        graph_before = _graph_revision(model)
        weight_rail_revision = GraphRevision.from_live_revision(graph_before)
        source_before = _source_identity(model)
        bundle_manifest_before = _bundle_manifest_sha256(model)
        _validate_intervention(model, spec)
        coordinate = _coordinate(model, spec)
        paired = spec.intervention_mode in ("native", "placebo")
        arm_count = 2 if paired else 1
        preflight, planned_source_bytes, planned_operations, planned_tensors = (
            _preflight_plans(model, spec, arm_count=arm_count)
        )
        model_pin = _model_pin(
            model,
            code_revision=spec.code_revision,
            bundle_manifest_sha256=bundle_manifest_before,
        )
        runtime = _runtime_provenance(
            model,
            spec,
            preflight=preflight,
            graph_revision=graph_before,
        )

        control_capture: _ArmCapture | None = None
        control_evidence: dict[str, Any] | None = None
        control_measurement: MeasurementReceipt | None = None
        primary_capture: _ArmCapture
        try:
            with model.pager._lock:
                if paired:
                    control_arm = (
                        "placebo" if spec.intervention_mode == "native" else "off"
                    )
                    control_capture = _execute_arm(model, spec, arm=control_arm)
                    control_body = _evidence_body(
                        control_capture,
                        spec=spec,
                        model_pin=model_pin,
                        coordinate=coordinate,
                        graph_revision=graph_before,
                        preflight=preflight,
                        paired=None,
                        control_measurement_sha256=None,
                        weight_rail_revision=weight_rail_revision,
                        atlas_head_revision=atlas_head_revision,
                    )
                    control_body["resource_preflight"] = {
                        "planned_operations": planned_operations,
                        "planned_source_bytes": planned_source_bytes,
                        "planned_tensors_per_arm": planned_tensors,
                    }
                    control_evidence = _seal_evidence(control_body)
                    if (
                        len(_canonical(control_evidence))
                        > spec.budget.max_evidence_bytes
                    ):
                        raise Qwen38CartographyBudgetError(
                            "control evidence exceeds max_evidence_bytes"
                        )
                    control_measurement = _measurement(
                        control_capture,
                        evidence=control_evidence,
                        model_pin=model_pin,
                        coordinate=coordinate,
                        probe=spec.probe_identity,
                        runtime=runtime,
                        weight_rail_revision=weight_rail_revision,
                        atlas_head_revision=atlas_head_revision,
                    )
                    primary_arm = (
                        "native" if spec.intervention_mode == "native" else "placebo"
                    )
                    primary_capture = _execute_arm(model, spec, arm=primary_arm)
                    pair_rows = _pair_capture(primary_capture, control_capture)
                else:
                    primary_capture = _execute_arm(
                        model, spec, arm=str(spec.intervention_mode)
                    )
                    pair_rows = None

                effects: tuple[PlaceboEffect, ...] = ()
                if spec.intervention_mode == "native":
                    assert control_capture is not None
                    assert control_measurement is not None
                    control_by_metric = {
                        row.metric: row for row in control_capture.summaries
                    }
                    effects = tuple(
                        PlaceboEffect(
                            metric=row.metric,
                            placebo_measurement_sha256=control_measurement.sha256,
                            observed_mean=row.mean,
                            placebo_mean=control_by_metric[row.metric].mean,
                            delta=row.mean - control_by_metric[row.metric].mean,
                        )
                        for row in primary_capture.summaries
                        if row.metric in control_by_metric
                    )
                primary_body = _evidence_body(
                    primary_capture,
                    spec=spec,
                    model_pin=model_pin,
                    coordinate=coordinate,
                    graph_revision=graph_before,
                    preflight=preflight,
                    paired=pair_rows,
                    control_measurement_sha256=(
                        None
                        if control_measurement is None
                        else control_measurement.sha256
                    ),
                    weight_rail_revision=weight_rail_revision,
                    atlas_head_revision=atlas_head_revision,
                )
                primary_body["resource_preflight"] = {
                    "planned_operations": planned_operations,
                    "planned_source_bytes": planned_source_bytes,
                    "planned_tensors_per_arm": planned_tensors,
                }
                primary_evidence = _seal_evidence(primary_body)
                if len(_canonical(primary_evidence)) > spec.budget.max_evidence_bytes:
                    raise Qwen38CartographyBudgetError(
                        "measurement evidence exceeds max_evidence_bytes"
                    )
                measurement = _measurement(
                    primary_capture,
                    evidence=primary_evidence,
                    model_pin=model_pin,
                    coordinate=coordinate,
                    probe=spec.probe_identity,
                    runtime=runtime,
                    weight_rail_revision=weight_rail_revision,
                    atlas_head_revision=atlas_head_revision,
                    placebo_effects=effects,
                )
        finally:
            model.pager.release()

        graph_after = _graph_revision(model)
        source_after = _source_identity(model)
        bundle_manifest_after = _bundle_manifest_sha256(model)
        state_after = _model_state_stamp(model)
        attachments_after = _model_attachment_stamp(model)
        if graph_after != graph_before:
            raise Qwen38CartographyIntegrityError(
                "causal graph revision changed during probe execution"
            )
        if source_after != source_before:
            raise Qwen38CartographyIntegrityError(
                "checkpoint source identity changed during probe execution"
            )
        if bundle_manifest_after != bundle_manifest_before:
            raise Qwen38CartographyIntegrityError(
                "causal bundle manifest changed during probe execution"
            )
        if state_after != state_before:
            raise Qwen38CartographyIntegrityError(
                "committed model state changed during non-committing probe"
            )
        if attachments_after != attachments_before:
            raise Qwen38CartographyIntegrityError(
                "model observer/intervention attachments changed during probe"
            )
        if model.checkpoint_preflight() != preflight:
            raise Qwen38CartographyIntegrityError(
                "checkpoint tensor contract changed during probe execution"
            )
        result = CartographyProbeResult(
            measurement=measurement,
            evidence_document=primary_evidence,
            access_trace=primary_capture.access_trace,
            tensor_range_receipts=primary_capture.tensor_receipts,
            control_measurement=control_measurement,
            control_evidence_document=control_evidence,
            control_access_trace=(
                None if control_capture is None else control_capture.access_trace
            ),
            control_tensor_range_receipts=(
                () if control_capture is None else control_capture.tensor_receipts
            ),
        )
        result.verify()
        return result


__all__ = [
    "CARTOGRAPHY_BUNDLE_VIEW_SCHEMA",
    "CARTOGRAPHY_EVIDENCE_SCHEMA",
    "CARTOGRAPHY_PROMPT_SCHEMA",
    "CartographyProbeResult",
    "HiddenSketchProjection",
    "ProbeCoordinateSpec",
    "ProbeResourceBudget",
    "ProbeSpec",
    "Qwen38CartographyBudgetError",
    "Qwen38CartographyIntegrityError",
    "Qwen38CartographyProbe",
    "Qwen38CartographyProbeError",
    "TensorRangeReceipt",
    "prompt_token_sha256",
]
