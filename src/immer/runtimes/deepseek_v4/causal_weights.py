"""Exact causal bindings from routed experts to immutable shard byte ranges.

The model identity and its physical safetensors layout are deliberately
separate.  A semantic ``(layer, expert_id)`` key belongs to the logical model;
direct LiveCausal edges point from that key to one layout-specific immutable
range plan.  Repacking shards therefore regenerates bindings without changing
the model identity or any weight byte.

This module is only an address plane.  It never changes router decisions,
decodes tensors, mutates weights, or permits model computation to be skipped.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import threading
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - the runtime target is macOS/Linux.
    fcntl = None  # type: ignore[assignment]

from immer.knowledge.livecausal import LiveGraph
from immer.knowledge.streamer import Streamer, TensorSource

from .pager import (
    ExpertSourceRange,
    ExpertTensorLayout,
    OfficialExpertRangePlan,
)

CAUSAL_WEIGHT_BINDING_SCHEMA = "causal-weight-binding/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EXPERT_BASE = re.compile(r"layers\.(0|[1-9][0-9]*)\.ffn\.experts\.(0|[1-9][0-9]*)\Z")
_DTYPE = re.compile(r"[A-Z][A-Z0-9_]*\Z")
_UINT64_MAX = (1 << 64) - 1
_BINDING_LOCK_NAME = ".causal-weight-bindings.lock"
_PLAN_CACHE_RESOLVE_RETRIES = 3
_RECORD_KEYS = frozenset(
    (
        "layout_fingerprint",
        "logical_model",
        "outcome_key",
        "plan",
        "plan_sha256",
        "record_type",
        "schema",
        "trigger_key",
    )
)

_BINDING_LOCKS_GUARD = threading.Lock()
_BINDING_LOCKS: dict[str, _StoreBindingLock] = {}


class CausalWeightError(ValueError):
    """A causal weight binding or read request is invalid."""


class CausalWeightConflictError(CausalWeightError):
    """One physical layout maps an expert to two different range plans."""


class CausalWeightIdentityError(CausalWeightError):
    """The mounted tensor source is not the layout selected by the reader."""


class CausalWeightNotFoundError(CausalWeightError, KeyError):
    """No range binding exists for an expert in the selected layout."""


class CausalWeightIntegrityError(CausalWeightError):
    """A graph record or tensor-source receipt violates the binding schema."""


class _StoreBindingLock:
    """Process-local half of one reentrant, cross-process store lock."""

    def __init__(self) -> None:
        self.thread_lock = threading.RLock()
        self.local = threading.local()


def _store_binding_lock(root: Path) -> _StoreBindingLock:
    key = os.path.normcase(str(root.resolve()))
    with _BINDING_LOCKS_GUARD:
        return _BINDING_LOCKS.setdefault(key, _StoreBindingLock())


def _same_regular_file(path: Path, opened: os.stat_result) -> bool:
    try:
        linked = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(opened.st_mode)
        and stat.S_ISREG(linked.st_mode)
        and opened.st_dev == linked.st_dev
        and opened.st_ino == linked.st_ino
    )


def _acquire_binding_file_lock(path: Path) -> int:
    if fcntl is None:  # pragma: no cover - the runtime target is macOS/Linux.
        raise CausalWeightIntegrityError(
            "causal weight binding requires POSIX fcntl.flock"
        )
    descriptor: int | None = None
    acquired = False
    success = False
    try:
        flags = os.O_RDWR | os.O_CREAT
        flags |= int(getattr(os, "O_CLOEXEC", 0))
        flags |= int(getattr(os, "O_NOFOLLOW", 0))
        descriptor = os.open(path, flags, 0o600)
        opened = os.fstat(descriptor)
        if not _same_regular_file(path, opened):
            raise CausalWeightIntegrityError(
                "causal weight binding lock must be an unchanged regular file"
            )
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        acquired = True
        if not _same_regular_file(path, opened):
            raise CausalWeightIntegrityError(
                "causal weight binding lock changed while acquiring it"
            )
        success = True
        return descriptor
    except OSError as exc:
        raise CausalWeightIntegrityError(
            f"cannot acquire causal weight binding lock: {path}"
        ) from exc
    finally:
        if descriptor is not None and not success:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


@contextmanager
def _binding_transaction(graph: LiveGraph) -> Iterator[None]:
    root = Path(graph.store.root)
    state = _store_binding_lock(root)
    with state.thread_lock:
        depth = int(getattr(state.local, "depth", 0))
        if depth:
            state.local.depth = depth + 1
            try:
                yield
            finally:
                state.local.depth = depth
            return

        descriptor = _acquire_binding_file_lock(root / _BINDING_LOCK_NAME)
        state.local.depth = 1
        try:
            yield
        finally:
            state.local.depth = 0
            assert fcntl is not None
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CausalWeightIntegrityError("value is not canonical JSON") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _text(value: object, label: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CausalWeightError(f"{label} must be non-empty canonical text")
    if len(value) > maximum or "\x00" in value:
        raise CausalWeightError(f"{label} exceeds its representation bound")
    return value


def _uint64(value: object, label: str, *, positive: bool = False) -> int:
    lower = 1 if positive else 0
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not lower <= value <= _UINT64_MAX
    ):
        qualifier = "positive " if positive else ""
        raise CausalWeightError(f"{label} must be a {qualifier}unsigned 64-bit integer")
    return value


def _coordinate(value: object, label: str) -> int:
    return _uint64(value, label)


def _safe_shard(value: object) -> str:
    shard = _text(value, "shard", maximum=1024)
    if "\\" in shard:
        raise CausalWeightError("shard must use POSIX separators")
    path = PurePosixPath(shard)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise CausalWeightError("shard must be a safe relative path")
    return shard


@dataclass(frozen=True, slots=True)
class LogicalModelIdentity:
    """Repository and revision identity, independent of shard packing."""

    repo_id: str
    revision: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "repo_id", _text(self.repo_id, "repo_id"))
        object.__setattr__(self, "revision", _text(self.revision, "revision"))

    def as_record(self) -> dict[str, str]:
        return {"repo_id": self.repo_id, "revision": self.revision}

    @property
    def key(self) -> str:
        return f"logical-model:v1:{_digest(self.as_record())}"


@dataclass(frozen=True, slots=True)
class CausalWeightLayoutIdentity:
    """One immutable physical tensor layout of a logical model."""

    model: LogicalModelIdentity
    layout_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, LogicalModelIdentity):
            raise CausalWeightError("model must be a LogicalModelIdentity")
        fingerprint = _text(
            self.layout_fingerprint,
            "layout_fingerprint",
            maximum=64,
        )
        if _SHA256.fullmatch(fingerprint) is None:
            raise CausalWeightError(
                "layout_fingerprint must be a lowercase SHA-256 digest"
            )
        object.__setattr__(self, "layout_fingerprint", fingerprint)

    def as_record(self) -> dict[str, Any]:
        return {
            "layout_fingerprint": self.layout_fingerprint,
            "logical_model": self.model.as_record(),
        }

    @property
    def key(self) -> str:
        return f"weight-layout:v1:{_digest(self.as_record())}"

    @classmethod
    def from_source(
        cls,
        source: TensorSource,
        *,
        model: LogicalModelIdentity | None = None,
    ) -> CausalWeightLayoutIdentity:
        """Prepare a physical inventory and mount it under a logical model.

        A local source normally reports ``local:/path`` as transport identity;
        callers mounting official model bytes locally pass ``model`` so that
        filesystem location never becomes the model's semantic identity.
        """

        inventory = getattr(source, "inventory", None)
        metrics = getattr(source, "metrics", None)
        if not callable(inventory) or not callable(metrics):
            raise CausalWeightIdentityError(
                "tensor source must expose inventory() and metrics()"
            )
        inventory()
        snapshot = metrics()
        if not isinstance(snapshot, Mapping):
            raise CausalWeightIdentityError("tensor source metrics are invalid")
        if model is None:
            model = LogicalModelIdentity(
                repo_id=snapshot.get("repo_id"),
                revision=snapshot.get("revision"),
            )
        elif not isinstance(model, LogicalModelIdentity):
            raise TypeError("model must be a LogicalModelIdentity")
        return cls(
            model=model,
            layout_fingerprint=snapshot.get("inventory_source_fingerprint"),
        )


@dataclass(frozen=True, slots=True)
class ExpertBindingReceipt:
    """Exact graph citation created or reused for one expert plan."""

    layer: int
    expert_id: int
    segment_sha256: str
    record_index: int
    appended: bool


@dataclass(frozen=True, slots=True)
class CausalWeightBindingReceipt:
    """Receipt for one arbitrary-fanout binding call."""

    layout: CausalWeightLayoutIdentity
    bindings: tuple[ExpertBindingReceipt, ...]
    appended_segment_sha256: str | None

    @property
    def appended_count(self) -> int:
        return sum(receipt.appended for receipt in self.bindings)


@dataclass(frozen=True, slots=True)
class CausalWeightLeaf:
    """One exact half-open range returned in plan/range order."""

    layer: int
    expert_id: int
    range_index: int
    shard: str
    absolute_offset: int
    length: int

    @property
    def absolute_end(self) -> int:
        return self.absolute_offset + self.length


@dataclass(frozen=True, slots=True)
class CausalWeightReadReceipt:
    """Resolved plans, byte views, and physical-I/O accounting for one read."""

    plans: tuple[OfficialExpertRangePlan, ...]
    leaves: tuple[CausalWeightLeaf, ...]
    parts: tuple[memoryview, ...]
    requested_bytes: int
    resident_bytes: int
    source_requests: int
    source_bytes: int


def semantic_expert_key(
    model: LogicalModelIdentity,
    *,
    layer: int,
    expert_id: int,
) -> str:
    """Return the layout-independent key for one official routed expert."""

    if not isinstance(model, LogicalModelIdentity):
        raise CausalWeightError("model must be a LogicalModelIdentity")
    coordinate = {
        "expert_id": _coordinate(expert_id, "expert_id"),
        "layer": _coordinate(layer, "layer"),
        "logical_model": model.as_record(),
    }
    return f"semantic-expert:v1:{_digest(coordinate)}"


def _plan_record(plan: OfficialExpertRangePlan) -> dict[str, Any]:
    return {
        "base": plan.base,
        "expert_id": plan.expert_id,
        "layer": plan.layer,
        "payload_bytes": plan.payload_bytes,
        "ranges": [
            {
                "absolute_offset": source_range.absolute_offset,
                "length": source_range.length,
                "shard": source_range.shard,
                "tensors": [
                    {
                        "absolute_offset": tensor.absolute_offset,
                        "dtype": tensor.dtype,
                        "length": tensor.length,
                        "name": tensor.name,
                        "range_offset": tensor.range_offset,
                        "shape": list(tensor.shape),
                    }
                    for tensor in source_range.tensors
                ],
            }
            for source_range in plan.ranges
        ],
    }


def _normalize_plan(plan: OfficialExpertRangePlan) -> OfficialExpertRangePlan:
    if not isinstance(plan, OfficialExpertRangePlan):
        raise CausalWeightError("plans must contain OfficialExpertRangePlan values")
    layer = _coordinate(plan.layer, "plan.layer")
    expert_id = _coordinate(plan.expert_id, "plan.expert_id")
    base = _text(plan.base, "plan.base", maximum=1024)
    match = _EXPERT_BASE.fullmatch(base)
    if match is None or (int(match.group(1)), int(match.group(2))) != (
        layer,
        expert_id,
    ):
        raise CausalWeightError("plan.base does not match plan layer/expert_id")
    if not isinstance(plan.ranges, tuple) or not plan.ranges:
        raise CausalWeightError("plan.ranges must be a non-empty tuple")

    normalized_ranges: list[ExpertSourceRange] = []
    tensor_names: set[str] = set()
    intervals: dict[str, list[tuple[int, int]]] = {}
    payload_bytes = 0
    for range_index, source_range in enumerate(plan.ranges):
        if not isinstance(source_range, ExpertSourceRange):
            raise CausalWeightError(
                f"plan.ranges[{range_index}] must be an ExpertSourceRange"
            )
        shard = _safe_shard(source_range.shard)
        absolute = _uint64(
            source_range.absolute_offset,
            f"plan.ranges[{range_index}].absolute_offset",
        )
        length = _uint64(
            source_range.length,
            f"plan.ranges[{range_index}].length",
            positive=True,
        )
        if absolute + length > _UINT64_MAX:
            raise CausalWeightError("source range end exceeds unsigned 64-bit space")
        if not isinstance(source_range.tensors, tuple) or not source_range.tensors:
            raise CausalWeightError("source ranges must contain tensor layouts")

        normalized_tensors: list[ExpertTensorLayout] = []
        cursor = 0
        for tensor_index, tensor in enumerate(source_range.tensors):
            label = f"plan.ranges[{range_index}].tensors[{tensor_index}]"
            if not isinstance(tensor, ExpertTensorLayout):
                raise CausalWeightError(f"{label} must be an ExpertTensorLayout")
            name = _text(tensor.name, f"{label}.name", maximum=2048)
            if name in tensor_names:
                raise CausalWeightError(f"duplicate tensor layout {name!r}")
            tensor_names.add(name)
            dtype = _text(tensor.dtype, f"{label}.dtype", maximum=64)
            if _DTYPE.fullmatch(dtype) is None:
                raise CausalWeightError(f"{label}.dtype is not canonical")
            if not isinstance(tensor.shape, tuple):
                raise CausalWeightError(f"{label}.shape must be a tuple")
            shape = tuple(
                _uint64(dimension, f"{label}.shape[{index}]")
                for index, dimension in enumerate(tensor.shape)
            )
            tensor_absolute = _uint64(
                tensor.absolute_offset,
                f"{label}.absolute_offset",
            )
            tensor_length = _uint64(
                tensor.length,
                f"{label}.length",
                positive=True,
            )
            range_offset = _uint64(tensor.range_offset, f"{label}.range_offset")
            if tensor_absolute + tensor_length > _UINT64_MAX:
                raise CausalWeightError(
                    "tensor range end exceeds unsigned 64-bit space"
                )
            if tensor_absolute != absolute + range_offset:
                raise CausalWeightError("tensor absolute/range offsets disagree")
            if range_offset != cursor or range_offset + tensor_length > length:
                raise CausalWeightError(
                    "tensor layouts must exactly tile their source range without gaps"
                )
            cursor += tensor_length
            normalized_tensors.append(
                ExpertTensorLayout(
                    name=name,
                    dtype=dtype,
                    shape=shape,
                    absolute_offset=tensor_absolute,
                    length=tensor_length,
                    range_offset=range_offset,
                )
            )
        if cursor != length:
            raise CausalWeightError(
                "tensor layouts must exactly cover their source range"
            )
        intervals.setdefault(shard, []).append((absolute, absolute + length))
        payload_bytes += length
        if payload_bytes > _UINT64_MAX:
            raise CausalWeightError("plan payload exceeds unsigned 64-bit space")
        normalized_ranges.append(
            ExpertSourceRange(
                shard=shard,
                absolute_offset=absolute,
                length=length,
                tensors=tuple(normalized_tensors),
            )
        )

    for shard, shard_intervals in intervals.items():
        ordered = sorted(shard_intervals)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise CausalWeightError(f"overlapping source ranges exist in {shard}")
    declared_payload = _uint64(plan.payload_bytes, "plan.payload_bytes", positive=True)
    if declared_payload != payload_bytes:
        raise CausalWeightError("plan.payload_bytes does not match its exact ranges")
    return OfficialExpertRangePlan(
        base=base,
        layer=layer,
        expert_id=expert_id,
        ranges=tuple(normalized_ranges),
        payload_bytes=payload_bytes,
    )


def _plan_from_record(value: object) -> OfficialExpertRangePlan:
    if not isinstance(value, Mapping):
        raise CausalWeightIntegrityError("binding plan must be an object")
    required = {"base", "expert_id", "layer", "payload_bytes", "ranges"}
    if set(value) != required or not isinstance(value.get("ranges"), list):
        raise CausalWeightIntegrityError("binding plan schema is invalid")
    ranges: list[ExpertSourceRange] = []
    for raw_range in value["ranges"]:
        if not isinstance(raw_range, Mapping) or set(raw_range) != {
            "absolute_offset",
            "length",
            "shard",
            "tensors",
        }:
            raise CausalWeightIntegrityError("binding source-range schema is invalid")
        raw_tensors = raw_range.get("tensors")
        if not isinstance(raw_tensors, list):
            raise CausalWeightIntegrityError("binding tensors must be a list")
        tensors: list[ExpertTensorLayout] = []
        for raw_tensor in raw_tensors:
            if not isinstance(raw_tensor, Mapping) or set(raw_tensor) != {
                "absolute_offset",
                "dtype",
                "length",
                "name",
                "range_offset",
                "shape",
            }:
                raise CausalWeightIntegrityError("binding tensor schema is invalid")
            shape = raw_tensor.get("shape")
            if not isinstance(shape, list):
                raise CausalWeightIntegrityError("binding tensor shape must be a list")
            tensors.append(
                ExpertTensorLayout(
                    name=raw_tensor.get("name"),
                    dtype=raw_tensor.get("dtype"),
                    shape=tuple(shape),
                    absolute_offset=raw_tensor.get("absolute_offset"),
                    length=raw_tensor.get("length"),
                    range_offset=raw_tensor.get("range_offset"),
                )
            )
        ranges.append(
            ExpertSourceRange(
                shard=raw_range.get("shard"),
                absolute_offset=raw_range.get("absolute_offset"),
                length=raw_range.get("length"),
                tensors=tuple(tensors),
            )
        )
    try:
        candidate = OfficialExpertRangePlan(
            base=value.get("base"),
            layer=value.get("layer"),
            expert_id=value.get("expert_id"),
            ranges=tuple(ranges),
            payload_bytes=value.get("payload_bytes"),
        )
        return _normalize_plan(candidate)
    except (CausalWeightError, TypeError, ValueError) as exc:
        if isinstance(exc, CausalWeightIntegrityError):
            raise
        raise CausalWeightIntegrityError("stored range plan is invalid") from exc


def _range_plan_key(
    layout: CausalWeightLayoutIdentity,
    plan: OfficialExpertRangePlan,
) -> str:
    identity = {
        "layout": layout.as_record(),
        "plan_sha256": _digest(_plan_record(plan)),
    }
    return f"expert-range-plan:v1:{_digest(identity)}"


def _binding_record(
    layout: CausalWeightLayoutIdentity,
    plan: OfficialExpertRangePlan,
) -> dict[str, Any]:
    plan_record = _plan_record(plan)
    return {
        "layout_fingerprint": layout.layout_fingerprint,
        "logical_model": layout.model.as_record(),
        "outcome_key": _range_plan_key(layout, plan),
        "plan": plan_record,
        "plan_sha256": _digest(plan_record),
        "record_type": "expert_range_plan",
        "schema": CAUSAL_WEIGHT_BINDING_SCHEMA,
        "trigger_key": semantic_expert_key(
            layout.model,
            layer=plan.layer,
            expert_id=plan.expert_id,
        ),
    }


def _identity_from_record(record: Mapping[str, Any]) -> CausalWeightLayoutIdentity:
    model = record.get("logical_model")
    if not isinstance(model, Mapping) or set(model) != {"repo_id", "revision"}:
        raise CausalWeightIntegrityError("binding logical-model identity is invalid")
    try:
        return CausalWeightLayoutIdentity(
            model=LogicalModelIdentity(
                repo_id=model.get("repo_id"),
                revision=model.get("revision"),
            ),
            layout_fingerprint=record.get("layout_fingerprint"),
        )
    except (CausalWeightError, TypeError, ValueError) as exc:
        raise CausalWeightIntegrityError("binding layout identity is invalid") from exc


def _parse_binding_record(
    record: object,
) -> tuple[CausalWeightLayoutIdentity, OfficialExpertRangePlan]:
    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        raise CausalWeightIntegrityError("causal weight record schema is invalid")
    if (
        record.get("schema") != CAUSAL_WEIGHT_BINDING_SCHEMA
        or record.get("record_type") != "expert_range_plan"
    ):
        raise CausalWeightIntegrityError("causal weight record type is invalid")
    layout = _identity_from_record(record)
    plan = _plan_from_record(record.get("plan"))
    plan_record = _plan_record(plan)
    if record.get("plan_sha256") != _digest(plan_record):
        raise CausalWeightIntegrityError("binding plan digest is invalid")
    expected_trigger = semantic_expert_key(
        layout.model,
        layer=plan.layer,
        expert_id=plan.expert_id,
    )
    expected_outcome = _range_plan_key(layout, plan)
    if (
        record.get("trigger_key") != expected_trigger
        or record.get("outcome_key") != expected_outcome
    ):
        raise CausalWeightIntegrityError("binding graph keys are invalid")
    return layout, plan


def _direct_bindings(
    graph: LiveGraph,
    model: LogicalModelIdentity,
    *,
    layer: int,
    expert_id: int,
) -> list[tuple[CausalWeightLayoutIdentity, OfficialExpertRangePlan, list[Any]]]:
    trigger = semantic_expert_key(model, layer=layer, expert_id=expert_id)
    resolved: list[
        tuple[CausalWeightLayoutIdentity, OfficialExpertRangePlan, list[Any]]
    ] = []
    for edge in graph.query_base(trigger):
        if (
            edge.get("kind") != "base"
            or edge.get("depth") != 1
            or edge.get("from_key") != trigger
        ):
            raise CausalWeightIntegrityError("direct binding edge is invalid")
        derivation = edge.get("derivation")
        records = graph.resolve_derivation(derivation)
        if not records:
            raise CausalWeightIntegrityError("direct binding has no citation")
        if not isinstance(derivation, list) or len(derivation) != len(records):
            raise CausalWeightIntegrityError("direct binding citations are invalid")
        for citation, record in zip(derivation, records, strict=True):
            layout, plan = _parse_binding_record(record)
            if layout.model != model or (plan.layer, plan.expert_id) != (
                layer,
                expert_id,
            ):
                raise CausalWeightIntegrityError(
                    "binding resolved under the wrong semantic expert key"
                )
            if edge.get("to_key") != _range_plan_key(layout, plan):
                raise CausalWeightIntegrityError("binding outcome edge is invalid")
            resolved.append((layout, plan, list(citation)))
    return resolved


def _bind_normalized_plans(
    graph: LiveGraph,
    layout: CausalWeightLayoutIdentity,
    ordered: list[OfficialExpertRangePlan],
) -> CausalWeightBindingReceipt:
    """Scan and append while the store-wide binding transaction is held."""

    existing_citations: dict[tuple[int, int], list[Any]] = {}
    missing: list[OfficialExpertRangePlan] = []
    for plan in ordered:
        coordinate = (plan.layer, plan.expert_id)
        matching: list[list[Any]] = []
        for stored_layout, stored_plan, citation in _direct_bindings(
            graph,
            layout.model,
            layer=plan.layer,
            expert_id=plan.expert_id,
        ):
            if stored_layout.layout_fingerprint != layout.layout_fingerprint:
                continue
            if stored_layout != layout or stored_plan != plan:
                raise CausalWeightConflictError(
                    "layout already binds this expert to a different range plan"
                )
            matching.append(citation)
        if matching:
            existing_citations[coordinate] = sorted(
                matching,
                key=lambda pair: (str(pair[0]), int(pair[1])),
            )[0]
        else:
            missing.append(plan)

    appended_sha: str | None = None
    appended_citations: dict[tuple[int, int], list[Any]] = {}
    if missing:
        appended_sha = graph.append_segment(
            [_binding_record(layout, plan) for plan in missing]
        )
        appended_citations = {
            (plan.layer, plan.expert_id): [appended_sha, index]
            for index, plan in enumerate(missing)
        }

    receipts: list[ExpertBindingReceipt] = []
    for plan in ordered:
        coordinate = (plan.layer, plan.expert_id)
        appended = coordinate in appended_citations
        citation = (
            appended_citations[coordinate]
            if appended
            else existing_citations[coordinate]
        )
        receipts.append(
            ExpertBindingReceipt(
                layer=plan.layer,
                expert_id=plan.expert_id,
                segment_sha256=str(citation[0]),
                record_index=int(citation[1]),
                appended=appended,
            )
        )
    return CausalWeightBindingReceipt(
        layout=layout,
        bindings=tuple(receipts),
        appended_segment_sha256=appended_sha,
    )


def bind_causal_weight_plans(
    graph: LiveGraph,
    layout: CausalWeightLayoutIdentity,
    plans: Iterable[OfficialExpertRangePlan],
) -> CausalWeightBindingReceipt:
    """Append missing exact plans as one delta segment, with conflict checks.

    Existing identical bindings are replayed without a manifest write.  A
    second plan for the same ``(logical model, layout, layer, expert)`` is a
    conflict.  Plans for another layout fingerprint coexist under the same
    semantic expert key and are selected only by a matching reader.
    """

    if not isinstance(graph, LiveGraph):
        raise TypeError("graph must be a LiveGraph")
    if not isinstance(layout, CausalWeightLayoutIdentity):
        raise TypeError("layout must be a CausalWeightLayoutIdentity")
    try:
        raw_plans = tuple(plans)
    except TypeError as exc:
        raise CausalWeightError("plans must be iterable") from exc

    ordered: list[OfficialExpertRangePlan] = []
    by_coordinate: dict[tuple[int, int], OfficialExpertRangePlan] = {}
    for raw_plan in raw_plans:
        plan = _normalize_plan(raw_plan)
        coordinate = (plan.layer, plan.expert_id)
        prior = by_coordinate.get(coordinate)
        if prior is not None:
            if prior != plan:
                raise CausalWeightConflictError(
                    "one binding batch contains conflicting plans for an expert"
                )
            continue
        by_coordinate[coordinate] = plan
        ordered.append(plan)

    with _binding_transaction(graph):
        return _bind_normalized_plans(graph, layout, ordered)


class CausalWeightReader:
    """Resolve and read exact expert ranges without tensor-name discovery."""

    def __init__(
        self,
        graph: LiveGraph,
        layout: CausalWeightLayoutIdentity,
        *,
        source: TensorSource | None = None,
    ) -> None:
        if not isinstance(graph, LiveGraph):
            raise TypeError("graph must be a LiveGraph")
        if not isinstance(layout, CausalWeightLayoutIdentity):
            raise TypeError("layout must be a CausalWeightLayoutIdentity")
        self.graph = graph
        self.layout = layout
        self.source = source
        self._plan_cache_lock = threading.RLock()
        self._plan_cache_revision: tuple[int, str] | None = None
        self._plan_cache: dict[tuple[int, int], OfficialExpertRangePlan] = {}
        self._plan_cache_hits = 0
        self._plan_cache_misses = 0
        self._plan_cache_invalidations = 0
        self._metrics_lock = threading.Lock()
        self._read_calls = 0
        self._requested_bytes = 0
        self._resident_bytes = 0
        self._source_requests = 0
        self._source_bytes = 0
        if source is not None:
            self._validate_source_identity()

    def _validate_source_identity(self) -> None:
        source = self.source
        if source is None:
            raise CausalWeightIdentityError("no tensor source is attached")
        metrics = getattr(source, "metrics", None)
        if not callable(metrics):
            raise CausalWeightIdentityError("tensor source has no metrics identity")
        snapshot = metrics()
        if not isinstance(snapshot, Mapping):
            raise CausalWeightIdentityError("tensor source metrics are invalid")
        # repo_id/revision here identify the physical transport.  A local
        # mount quite correctly reports local:/path; only its verified layout
        # fingerprint is compared to the binding selected by the caller.
        observed = snapshot.get("inventory_source_fingerprint")
        if observed != self.layout.layout_fingerprint:
            raise CausalWeightIdentityError(
                "mounted tensor layout identity does not match causal bindings"
            )

    @staticmethod
    def _requested_coordinates(
        layer: int,
        expert_ids: Iterable[int],
    ) -> tuple[tuple[int, int], ...]:
        normalized_layer = _coordinate(layer, "layer")
        if isinstance(expert_ids, (str, bytes)):
            raise CausalWeightError("expert_ids must be an iterable of integers")
        try:
            raw_ids = tuple(expert_ids)
        except TypeError as exc:
            raise CausalWeightError("expert_ids must be iterable") from exc
        seen: set[int] = set()
        coordinates: list[tuple[int, int]] = []
        for raw_id in raw_ids:
            expert_id = _coordinate(raw_id, "expert_id")
            if expert_id not in seen:
                seen.add(expert_id)
                coordinates.append((normalized_layer, expert_id))
        return tuple(coordinates)

    def resolve_expert_plans(
        self,
        layer: int,
        expert_ids: Iterable[int],
    ) -> tuple[OfficialExpertRangePlan, ...]:
        """Resolve arbitrary fanout through a revision-bound immutable cache.

        A warm hit performs exactly one manifest-revision check.  Misses are
        resolved through direct graph edges and published only when a second
        revision check proves that the observed graph snapshot stayed stable.
        """

        coordinates = self._requested_coordinates(layer, expert_ids)
        if not coordinates:
            return ()
        with self._plan_cache_lock:
            for _attempt in range(_PLAN_CACHE_RESOLVE_RETRIES):
                revision = self.graph.store.revision()
                self._adopt_plan_cache_revision(revision)
                cached = {
                    coordinate: self._plan_cache[coordinate]
                    for coordinate in coordinates
                    if coordinate in self._plan_cache
                }
                missing = tuple(
                    coordinate for coordinate in coordinates if coordinate not in cached
                )
                if not missing:
                    self._plan_cache_hits += len(coordinates)
                    return tuple(cached[coordinate] for coordinate in coordinates)

                try:
                    resolved = {
                        coordinate: self._resolve_expert_plan_uncached(*coordinate)
                        for coordinate in missing
                    }
                except Exception:
                    after = self.graph.store.revision()
                    if after != revision:
                        self._adopt_plan_cache_revision(after)
                        continue
                    self._plan_cache_hits += len(cached)
                    self._plan_cache_misses += len(missing)
                    raise

                after = self.graph.store.revision()
                if after != revision:
                    self._adopt_plan_cache_revision(after)
                    continue
                self._plan_cache.update(resolved)
                self._plan_cache_hits += len(cached)
                self._plan_cache_misses += len(missing)
                return tuple(self._plan_cache[coordinate] for coordinate in coordinates)

            self._plan_cache_misses += len(coordinates)
            raise CausalWeightIntegrityError(
                "causal graph revision changed during every bounded plan-cache retry"
            )

    def _adopt_plan_cache_revision(self, revision: tuple[int, str]) -> None:
        if self._plan_cache_revision == revision:
            return
        if self._plan_cache_revision is not None:
            self._plan_cache_invalidations += 1
        self._plan_cache.clear()
        self._plan_cache_revision = revision

    def _resolve_expert_plan_uncached(
        self,
        layer: int,
        expert_id: int,
    ) -> OfficialExpertRangePlan:
        matches: list[OfficialExpertRangePlan] = []
        for stored_layout, plan, _citation in _direct_bindings(
            self.graph,
            self.layout.model,
            layer=layer,
            expert_id=expert_id,
        ):
            if stored_layout.layout_fingerprint == self.layout.layout_fingerprint:
                if stored_layout != self.layout:
                    raise CausalWeightIntegrityError(
                        "stored layout identity is internally inconsistent"
                    )
                matches.append(plan)
        if not matches:
            raise CausalWeightNotFoundError(
                "no causal range binding exists for "
                f"layer {layer} expert {expert_id} in layout "
                f"{self.layout.layout_fingerprint}"
            )
        if any(plan != matches[0] for plan in matches[1:]):
            raise CausalWeightConflictError(
                "graph contains conflicting range plans for one layout/expert"
            )
        return matches[0]

    def read_experts(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        resident_limit_bytes: int | None = None,
    ) -> CausalWeightReadReceipt:
        """Read exact resolved leaves with zero gap bytes via ``raw_bytes_many``."""

        self._validate_source_identity()
        assert self.source is not None
        raw_bytes_many = getattr(self.source, "raw_bytes_many", None)
        if not callable(raw_bytes_many):
            raise CausalWeightError(
                "attached tensor source must implement raw_bytes_many"
            )
        plans = self.resolve_expert_plans(layer, expert_ids)
        leaves: list[CausalWeightLeaf] = []
        indexed: dict[str, list[tuple[int, CausalWeightLeaf]]] = {}
        for plan in plans:
            for range_index, source_range in enumerate(plan.ranges):
                leaf = CausalWeightLeaf(
                    layer=plan.layer,
                    expert_id=plan.expert_id,
                    range_index=range_index,
                    shard=source_range.shard,
                    absolute_offset=source_range.absolute_offset,
                    length=source_range.length,
                )
                index = len(leaves)
                leaves.append(leaf)
                indexed.setdefault(leaf.shard, []).append((index, leaf))
        requested_bytes = sum(leaf.length for leaf in leaves)
        if resident_limit_bytes is None:
            resident_limit = requested_bytes
        else:
            resident_limit = _uint64(
                resident_limit_bytes,
                "resident_limit_bytes",
            )
            if resident_limit < requested_bytes:
                raise CausalWeightError(
                    "resident_limit_bytes is smaller than the exact requested payload"
                )

        parts: list[memoryview | None] = [None] * len(leaves)
        resident_bytes = 0
        source_requests = 0
        source_bytes = 0
        for shard, entries in indexed.items():
            requested = tuple(
                (leaf.absolute_offset, leaf.length) for _index, leaf in entries
            )
            result = raw_bytes_many(
                shard,
                requested,
                resident_limit_bytes=sum(length for _offset, length in requested),
                max_gap_bytes=0,
            )
            result_parts = tuple(getattr(result, "parts", ()))
            if len(result_parts) != len(entries):
                raise CausalWeightIntegrityError(
                    f"raw_bytes_many returned {len(result_parts)}/{len(entries)} parts"
                )
            for (index, leaf), part in zip(entries, result_parts, strict=True):
                try:
                    view = part if isinstance(part, memoryview) else memoryview(part)
                except TypeError as exc:
                    raise CausalWeightIntegrityError(
                        "raw_bytes_many returned a non-buffer part"
                    ) from exc
                if not view.readonly or len(view) != leaf.length:
                    raise CausalWeightIntegrityError(
                        "raw_bytes_many returned a mutable or short exact leaf"
                    )
                parts[index] = view
            counters: list[int] = []
            for name in ("resident_bytes", "source_requests", "source_bytes"):
                value = getattr(result, name, None)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise CausalWeightIntegrityError(
                        f"raw_bytes_many returned invalid {name}"
                    )
                counters.append(value)
            resident_bytes += counters[0]
            source_requests += counters[1]
            source_bytes += counters[2]
        if any(part is None for part in parts):  # pragma: no cover - indexed above.
            raise CausalWeightIntegrityError("one exact source leaf was not returned")
        if resident_bytes > resident_limit:
            raise CausalWeightIntegrityError(
                "raw_bytes_many exceeded the global resident_limit_bytes guard"
            )
        immutable_parts = tuple(part for part in parts if part is not None)
        receipt = CausalWeightReadReceipt(
            plans=plans,
            leaves=tuple(leaves),
            parts=immutable_parts,
            requested_bytes=requested_bytes,
            resident_bytes=resident_bytes,
            source_requests=source_requests,
            source_bytes=source_bytes,
        )
        with self._metrics_lock:
            self._read_calls += 1
            self._requested_bytes += requested_bytes
            self._resident_bytes += resident_bytes
            self._source_requests += source_requests
            self._source_bytes += source_bytes
        return receipt

    def metrics(self) -> dict[str, int | str]:
        """Return cumulative direct-range receipts for benchmark accounting."""

        with self._plan_cache_lock:
            cache_revision = self._plan_cache_revision
            cache_metrics: dict[str, int | str] = {
                "plan_cache_entries": len(self._plan_cache),
                "plan_cache_hits": self._plan_cache_hits,
                "plan_cache_invalidations": self._plan_cache_invalidations,
                "plan_cache_misses": self._plan_cache_misses,
                "plan_cache_revision_sequence": (
                    -1 if cache_revision is None else cache_revision[0]
                ),
                "plan_cache_revision_sha256": (
                    "" if cache_revision is None else cache_revision[1]
                ),
            }
        with self._metrics_lock:
            return {
                "layout_fingerprint": self.layout.layout_fingerprint,
                "read_calls": self._read_calls,
                "requested_bytes": self._requested_bytes,
                "resident_bytes": self._resident_bytes,
                "source_requests": self._source_requests,
                "source_bytes": self._source_bytes,
                **cache_metrics,
            }


class CausalWeightMount:
    """Open one local ``weights/`` + persistent ``causal/`` model bundle.

    Weight files stay in place and are read by local ``pread`` ranges with no
    disk payload cache.  The sibling causal directory stores only the durable
    address graph and can be appended while this mount is alive.
    """

    WEIGHTS_DIRECTORY = "weights"
    CAUSAL_DIRECTORY = "causal"

    def __init__(
        self,
        root: str | os.PathLike[str],
        model: LogicalModelIdentity,
        *,
        budget_mb: float = 200.0,
        max_metadata_bytes: int = 64 * 1024 * 1024,
        max_open_files: int | None = 64,
        verbose: bool = False,
    ) -> None:
        if not isinstance(model, LogicalModelIdentity):
            raise TypeError("model must be a LogicalModelIdentity")
        bundle_root = Path(root).expanduser().absolute()
        weights_root = bundle_root / self.WEIGHTS_DIRECTORY
        causal_root = bundle_root / self.CAUSAL_DIRECTORY
        self._require_plain_directory(bundle_root, "causal bundle root")
        self._require_plain_directory(weights_root, "causal bundle weights root")
        self._require_plain_directory(causal_root, "causal bundle graph root")

        pinned_inventory = None
        pinned_fingerprint = None
        pinned_path = weights_root / "inventory.pinned.json"
        if pinned_path.exists():
            try:
                pinned_document = json.loads(pinned_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise CausalWeightIntegrityError(
                    "causal bundle pinned inventory is unreadable"
                ) from exc
            if (
                not isinstance(pinned_document, Mapping)
                or pinned_document.get("schema") != "immer.tensor-inventory-cache/v1"
                or not isinstance(pinned_document.get("inventory"), Mapping)
                or not isinstance(pinned_document.get("source_fingerprint"), str)
            ):
                raise CausalWeightIntegrityError(
                    "causal bundle pinned inventory schema is invalid"
                )
            pinned_inventory = pinned_document["inventory"]
            pinned_fingerprint = pinned_document["source_fingerprint"]

        source = Streamer.from_local(
            weights_root,
            repo_id=model.repo_id,
            revision=model.revision,
            pinned_inventory=pinned_inventory,
            pinned_fingerprint=pinned_fingerprint,
            budget_mb=budget_mb,
            use_cache=False,
            max_metadata_bytes=max_metadata_bytes,
            max_open_files=max_open_files,
            verbose=verbose,
        )
        try:
            layout = CausalWeightLayoutIdentity.from_source(source, model=model)
            graph = LiveGraph(causal_root)
            reader = CausalWeightReader(graph, layout, source=source)
        except Exception:
            source.close()
            raise

        self.root = bundle_root
        self.weights_root = weights_root
        self.causal_root = causal_root
        self.model = model
        self.source = source
        self.layout = layout
        self.graph = graph
        self.reader = reader
        self._closed = False

    @staticmethod
    def _require_plain_directory(path: Path, label: str) -> None:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise CausalWeightIntegrityError(f"{label} is missing: {path}") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise CausalWeightIntegrityError(
                f"{label} must be a non-symlink directory: {path}"
            )

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise CausalWeightError("causal weight mount is closed")

    def bind_plans(
        self,
        plans: Iterable[OfficialExpertRangePlan],
    ) -> CausalWeightBindingReceipt:
        self._require_open()
        return bind_causal_weight_plans(self.graph, self.layout, plans)

    def resolve_expert_plans(
        self,
        layer: int,
        expert_ids: Iterable[int],
    ) -> tuple[OfficialExpertRangePlan, ...]:
        self._require_open()
        return self.reader.resolve_expert_plans(layer, expert_ids)

    def read_experts(
        self,
        layer: int,
        expert_ids: Iterable[int],
        *,
        resident_limit_bytes: int | None = None,
    ) -> CausalWeightReadReceipt:
        self._require_open()
        return self.reader.read_experts(
            layer,
            expert_ids,
            resident_limit_bytes=resident_limit_bytes,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.source.close()
        finally:
            self._closed = True

    def __enter__(self) -> CausalWeightMount:
        self._require_open()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()


__all__ = [
    "CAUSAL_WEIGHT_BINDING_SCHEMA",
    "CausalWeightBindingReceipt",
    "CausalWeightConflictError",
    "CausalWeightError",
    "CausalWeightIdentityError",
    "CausalWeightIntegrityError",
    "CausalWeightLayoutIdentity",
    "CausalWeightLeaf",
    "CausalWeightMount",
    "CausalWeightNotFoundError",
    "CausalWeightReadReceipt",
    "CausalWeightReader",
    "ExpertBindingReceipt",
    "LogicalModelIdentity",
    "bind_causal_weight_plans",
    "semantic_expert_key",
]
