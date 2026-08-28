"""Crash-safe exact Qwen MLP boundary evidence.

The Qwen runtime already emits ordered ``mlp.input/gate/up/output`` tensors.
This module owns only their bounded binary persistence, exact receipts, replay
verification interface, and append-only journal.  It never changes attention,
never reconstructs context from embeddings, and never emits tensor bytes as
JSON.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import struct
import time
from typing import Protocol, cast, runtime_checkable

import numpy as np

from .identity import canonical_json_bytes, require_sha256
from .subspace_battery import (
    SUBSPACE_ATTENTION_KIND,
    SUBSPACE_CONTEXT_KIND,
    SubspaceCorpus,
    SubspaceObservationGroup,
    graph_revision_sha256,
    output_evidence_sha256,
    projection_evidence_sha256,
)


TENSOR_SCHEMA = "immer.qwen3.8-mlp-tensor-object/v1"
EVIDENCE_SCHEMA = "immer.qwen3.8-mlp-evidence/v1"
VERIFICATION_SCHEMA = "immer.qwen3.8-mlp-evidence-verification/v1"
VERIFIER_EVIDENCE_SCHEMA = "immer.qwen3.8-mlp-verifier-evidence/v1"
JOURNAL_STATE_SCHEMA = "immer.qwen3.8-mlp-evidence-journal/v1"
JOURNAL_INTENT_SCHEMA = "immer.qwen3.8-mlp-evidence-intent/v1"
CAPTURE_MANIFEST_SCHEMA = "immer.qwen3.8-mlp-capture-manifest/v1"
EVIDENCE_BUDGET_SCHEMA = "immer.qwen3.8-mlp-evidence-budget/v1"
CAPTURE_STAGES = ("mlp.input", "mlp.gate", "mlp.up", "mlp.output")
CAPTURE_SPLITS = ("train", "calibration", "holdout")
CAPTURE_PLAN_LAYERS = {
    "train": (45, 18, 36, 63, 27),
    "calibration": (0, 9),
    "holdout": (54,),
}
_MAGIC = b"IMMLP01\0"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_HEAD = "HEAD"
_INTENT = "PREPARED"
_LOCK = "LOCK"
_BUDGET = "BUDGET.json"


class QwenMlpEvidenceError(RuntimeError):
    pass


class QwenMlpEvidenceIntegrityError(QwenMlpEvidenceError):
    pass


class QwenMlpEvidenceCapacityError(QwenMlpEvidenceError):
    pass


class QwenMlpEvidenceConflictError(QwenMlpEvidenceError):
    pass


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sealed(schema: str, body: Mapping[str, object]) -> dict[str, object]:
    normalized = json.loads(canonical_json_bytes(dict(body)))
    return {"body": normalized, "body_sha256": _digest(normalized), "schema": schema}


def _strict_json(
    data: bytes, *, schema: str, label: str, maximum: int
) -> Mapping[str, object]:
    if not isinstance(data, bytes) or not data or len(data) > maximum:
        raise QwenMlpEvidenceIntegrityError(f"{label} exceeds its byte bound")

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise QwenMlpEvidenceIntegrityError(f"{label} is not strict JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != {"body", "body_sha256", "schema"}
        or value.get("schema") != schema
        or not isinstance(value.get("body"), Mapping)
        or value.get("body_sha256") != _digest(value.get("body"))
        or canonical_json_bytes(value) != data
    ):
        raise QwenMlpEvidenceIntegrityError(f"{label} envelope is invalid")
    return cast(Mapping[str, object], value)


@dataclass(frozen=True, slots=True)
class MlpEvidenceBudget:
    max_groups: int = 4096
    max_rows_per_group: int = 512
    max_feature_dimension: int = 65_536
    max_tensor_bytes: int = 64 * 1024 * 1024
    max_group_bytes: int = 192 * 1024 * 1024
    max_total_referenced_bytes: int = 4 * 1024**3
    max_receipt_bytes: int = 64 * 1024
    max_header_bytes: int = 4096

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def sha256(self) -> str:
        return _digest(
            {name: getattr(self, name) for name in self.__dataclass_fields__}
        )


@dataclass(frozen=True, slots=True)
class BinaryTensorRef:
    stage: str
    layer: int
    object_sha256: str
    header_sha256: str
    raw_sha256: str
    source_dtype: str
    storage_dtype: str
    shape: tuple[int, ...]
    raw_nbytes: int
    row_sha256s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.stage not in CAPTURE_STAGES:
            raise ValueError("tensor stage is invalid")
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or self.layer < 0
        ):
            raise ValueError("tensor layer is invalid")
        for field in ("object_sha256", "header_sha256", "raw_sha256"):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        shape = tuple(self.shape)
        if not shape or any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in shape
        ):
            raise ValueError("tensor shape is invalid")
        if self.source_dtype not in {"bfloat16", "float16", "float32", "float64"}:
            raise ValueError("source dtype is unsupported")
        expected_storage = {
            "bfloat16": "bfloat16-bits-le",
            "float16": "float16-bits-le",
            "float32": "float32-le",
            "float64": "float64-le",
        }[self.source_dtype]
        if self.storage_dtype != expected_storage:
            raise ValueError("storage dtype differs from source dtype")
        if (
            isinstance(self.raw_nbytes, bool)
            or not isinstance(self.raw_nbytes, int)
            or self.raw_nbytes < 1
        ):
            raise ValueError("raw_nbytes is invalid")
        rows = tuple(require_sha256(v, field="row_sha256s") for v in self.row_sha256s)
        if rows and (self.stage != "mlp.output" or len(rows) != math.prod(shape[:-1])):
            raise ValueError("row hashes are valid only for every output row")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "row_sha256s", rows)

    def to_record(self) -> dict[str, object]:
        return {
            "header_sha256": self.header_sha256,
            "layer": self.layer,
            "object_sha256": self.object_sha256,
            "raw_nbytes": self.raw_nbytes,
            "raw_sha256": self.raw_sha256,
            "row_sha256s": list(self.row_sha256s),
            "shape": list(self.shape),
            "source_dtype": self.source_dtype,
            "stage": self.stage,
            "storage_dtype": self.storage_dtype,
        }

    @classmethod
    def from_record(cls, value: object) -> "BinaryTensorRef":
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "header_sha256",
                "layer",
                "object_sha256",
                "raw_nbytes",
                "raw_sha256",
                "row_sha256s",
                "shape",
                "source_dtype",
                "stage",
                "storage_dtype",
            }
            or not isinstance(value.get("shape"), list)
            or not isinstance(value.get("row_sha256s"), list)
        ):
            raise QwenMlpEvidenceIntegrityError("tensor reference is invalid")
        try:
            return cls(
                stage=cast(str, value.get("stage")),
                layer=cast(int, value.get("layer")),
                object_sha256=cast(str, value.get("object_sha256")),
                header_sha256=cast(str, value.get("header_sha256")),
                raw_sha256=cast(str, value.get("raw_sha256")),
                source_dtype=cast(str, value.get("source_dtype")),
                storage_dtype=cast(str, value.get("storage_dtype")),
                shape=tuple(cast(list[int], value.get("shape"))),
                raw_nbytes=cast(int, value.get("raw_nbytes")),
                row_sha256s=tuple(cast(list[str], value.get("row_sha256s"))),
            )
        except (TypeError, ValueError) as exc:
            raise QwenMlpEvidenceIntegrityError(
                "tensor reference validation failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class CapturePlanEntry:
    ordinal: int
    split: str
    layer: int
    prompt_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 1
        ):
            raise ValueError("ordinal must be positive")
        if self.split not in CAPTURE_SPLITS:
            raise ValueError("capture split is invalid")
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or self.layer < 0
        ):
            raise ValueError("capture layer is invalid")
        object.__setattr__(
            self,
            "prompt_sha256",
            require_sha256(self.prompt_sha256, field="prompt_sha256"),
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "ordinal": self.ordinal,
            "prompt_sha256": self.prompt_sha256,
            "split": self.split,
        }

    @classmethod
    def from_record(cls, value: object) -> "CapturePlanEntry":
        if not isinstance(value, Mapping) or set(value) != {
            "layer",
            "ordinal",
            "prompt_sha256",
            "split",
        }:
            raise QwenMlpEvidenceIntegrityError("capture plan entry is invalid")
        return cls(
            cast(int, value.get("ordinal")),
            cast(str, value.get("split")),
            cast(int, value.get("layer")),
            cast(str, value.get("prompt_sha256")),
        )


def canonical_capture_plan(
    prompt_sha256s: Sequence[str],
) -> tuple[CapturePlanEntry, ...]:
    prompts = tuple(
        sorted({require_sha256(v, field="prompt_sha256s") for v in prompt_sha256s})
    )
    if not prompts:
        raise ValueError("capture plan requires prompts")
    rows: list[CapturePlanEntry] = []
    for split in CAPTURE_SPLITS:
        for layer in CAPTURE_PLAN_LAYERS[split]:
            for prompt in prompts:
                rows.append(CapturePlanEntry(len(rows) + 1, split, layer, prompt))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class CaptureManifest:
    model_pin_sha256: str
    input_manifest_sha256: str
    prompt_sha256s: tuple[str, ...]
    entries: tuple[CapturePlanEntry, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_pin_sha256",
            require_sha256(self.model_pin_sha256, field="model_pin_sha256"),
        )
        object.__setattr__(
            self,
            "input_manifest_sha256",
            require_sha256(self.input_manifest_sha256, field="input_manifest_sha256"),
        )
        prompts = tuple(
            sorted(
                {require_sha256(v, field="prompt_sha256s") for v in self.prompt_sha256s}
            )
        )
        if len(prompts) != 5:
            raise ValueError("exact capture manifest requires five prompts")
        entries = tuple(self.entries)
        if entries != canonical_capture_plan(prompts):
            raise ValueError("manifest entries differ from canonical 25/10/5 plan")
        object.__setattr__(self, "prompt_sha256s", prompts)
        object.__setattr__(self, "entries", entries)

    @property
    def sha256(self) -> str:
        return _digest(self.to_document())

    def to_document(self) -> dict[str, object]:
        return _sealed(
            CAPTURE_MANIFEST_SCHEMA,
            {
                "attention_kind": SUBSPACE_ATTENTION_KIND,
                "context_kind": SUBSPACE_CONTEXT_KIND,
                "entries": [v.to_record() for v in self.entries],
                "input_manifest_sha256": self.input_manifest_sha256,
                "model_pin_sha256": self.model_pin_sha256,
                "prompt_sha256s": list(self.prompt_sha256s),
            },
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_document())

    @classmethod
    def from_bytes(cls, data: bytes) -> "CaptureManifest":
        env = _strict_json(
            data,
            schema=CAPTURE_MANIFEST_SCHEMA,
            label="capture manifest",
            maximum=1024 * 1024,
        )
        body = cast(Mapping[str, object], env["body"])
        if (
            set(body)
            != {
                "attention_kind",
                "context_kind",
                "entries",
                "input_manifest_sha256",
                "model_pin_sha256",
                "prompt_sha256s",
            }
            or body.get("attention_kind") != SUBSPACE_ATTENTION_KIND
            or body.get("context_kind") != SUBSPACE_CONTEXT_KIND
            or not isinstance(body.get("entries"), list)
            or not isinstance(body.get("prompt_sha256s"), list)
        ):
            raise QwenMlpEvidenceIntegrityError("capture manifest body is invalid")
        result = cls(
            model_pin_sha256=cast(str, body.get("model_pin_sha256")),
            input_manifest_sha256=cast(str, body.get("input_manifest_sha256")),
            prompt_sha256s=tuple(cast(list[str], body.get("prompt_sha256s"))),
            entries=tuple(
                CapturePlanEntry.from_record(row)
                for row in cast(list[object], body.get("entries"))
            ),
        )
        if result.to_bytes() != data:
            raise QwenMlpEvidenceIntegrityError("capture manifest changed")
        return result


@dataclass(frozen=True, slots=True)
class MlpEvidenceReceipt:
    entry: CapturePlanEntry
    capture_mode: str
    manifest_sha256: str
    model_pin_sha256: str
    token_sha256: str
    probe_spec_sha256: str
    atlas_sequence: int
    atlas_event_sha256: str
    atlas_revision_sha256: str
    measurement_sha256: str
    weight_revision_sha256: str
    access_trace_sha256: str
    source_receipt_sha256s: tuple[str, ...]
    tensors: tuple[BinaryTensorRef, ...]
    cartography_input_sketch_sha256: str
    recomputed_input_sketch_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.entry, CapturePlanEntry):
            raise TypeError("entry must be CapturePlanEntry")
        if self.capture_mode not in {"live-exact", "fixture"}:
            raise ValueError("capture_mode is invalid")
        for field in (
            "manifest_sha256",
            "model_pin_sha256",
            "token_sha256",
            "probe_spec_sha256",
            "atlas_event_sha256",
            "atlas_revision_sha256",
            "measurement_sha256",
            "weight_revision_sha256",
            "access_trace_sha256",
            "cartography_input_sketch_sha256",
            "recomputed_input_sketch_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        if (
            isinstance(self.atlas_sequence, bool)
            or not isinstance(self.atlas_sequence, int)
            or self.atlas_sequence < 0
        ):
            raise ValueError("atlas_sequence is invalid")
        if self.atlas_revision_sha256 != graph_revision_sha256(
            self.atlas_sequence, self.atlas_event_sha256
        ):
            raise ValueError("Atlas revision differs from sequence/event")
        if self.cartography_input_sketch_sha256 != self.recomputed_input_sketch_sha256:
            raise ValueError("full mlp.input does not reproduce cartography sketch")
        sources = tuple(
            sorted(
                {
                    require_sha256(v, field="source_receipt_sha256s")
                    for v in self.source_receipt_sha256s
                }
            )
        )
        if not sources or len(sources) > 64:
            raise ValueError("source_receipt_sha256s must contain 1..64 hashes")
        tensors = tuple(self.tensors)
        if tuple(v.stage for v in tensors) != CAPTURE_STAGES or any(
            v.layer != self.entry.layer for v in tensors
        ):
            raise ValueError("evidence tensor stages/layers are invalid")
        shapes = {v.stage: v.shape for v in tensors}
        if (
            shapes["mlp.input"][:-1] != shapes["mlp.gate"][:-1]
            or shapes["mlp.gate"] != shapes["mlp.up"]
            or shapes["mlp.input"] != shapes["mlp.output"]
        ):
            raise ValueError("MLP evidence tensor shapes are incompatible")
        if not tensors[-1].row_sha256s:
            raise ValueError("MLP output requires exact row payload hashes")
        object.__setattr__(self, "source_receipt_sha256s", sources)
        object.__setattr__(self, "tensors", tensors)

    @property
    def identity_sha256(self) -> str:
        return _digest(
            {
                "entry_sha256": self.entry.sha256,
                "capture_mode": self.capture_mode,
                "manifest_sha256": self.manifest_sha256,
                "model_pin_sha256": self.model_pin_sha256,
                "probe_spec_sha256": self.probe_spec_sha256,
            }
        )

    @property
    def capture_event_sha256(self) -> str:
        """Deterministic event for the manifest-ordered MLP evidence graph."""

        return _digest(
            {
                "access_trace_sha256": self.access_trace_sha256,
                "atlas_revision_sha256": self.atlas_revision_sha256,
                "cartography_input_sketch_sha256": (
                    self.cartography_input_sketch_sha256
                ),
                "entry_sha256": self.entry.sha256,
                "manifest_sha256": self.manifest_sha256,
                "measurement_sha256": self.measurement_sha256,
                "model_pin_sha256": self.model_pin_sha256,
                "probe_spec_sha256": self.probe_spec_sha256,
                "schema": "immer.qwen3.8-mlp-capture-event/v1",
                "source_receipt_sha256s": list(self.source_receipt_sha256s),
                "tensor_raw_sha256s": [row.raw_sha256 for row in self.tensors],
                "token_sha256": self.token_sha256,
                "weight_revision_sha256": self.weight_revision_sha256,
            }
        )

    @property
    def capture_revision_sha256(self) -> str:
        return graph_revision_sha256(self.entry.ordinal, self.capture_event_sha256)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        return _sealed(
            EVIDENCE_SCHEMA,
            {
                "access_trace_sha256": self.access_trace_sha256,
                "atlas_event_sha256": self.atlas_event_sha256,
                "atlas_revision_sha256": self.atlas_revision_sha256,
                "atlas_sequence": self.atlas_sequence,
                "attention_kind": SUBSPACE_ATTENTION_KIND,
                "cartography_input_sketch_sha256": self.cartography_input_sketch_sha256,
                "capture_mode": self.capture_mode,
                "capture_event_sha256": self.capture_event_sha256,
                "capture_revision_sha256": self.capture_revision_sha256,
                "capture_sequence": self.entry.ordinal,
                "context_kind": SUBSPACE_CONTEXT_KIND,
                "entry": self.entry.to_record(),
                "identity_sha256": self.identity_sha256,
                "manifest_sha256": self.manifest_sha256,
                "measurement_sha256": self.measurement_sha256,
                "model_pin_sha256": self.model_pin_sha256,
                "probe_spec_sha256": self.probe_spec_sha256,
                "recomputed_input_sketch_sha256": self.recomputed_input_sketch_sha256,
                "source_receipt_sha256s": list(self.source_receipt_sha256s),
                "tensors": [v.to_record() for v in self.tensors],
                "token_sha256": self.token_sha256,
                "weight_revision_sha256": self.weight_revision_sha256,
            },
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_document())

    @classmethod
    def from_bytes(
        cls, data: bytes, *, maximum: int = 64 * 1024
    ) -> "MlpEvidenceReceipt":
        env = _strict_json(
            data, schema=EVIDENCE_SCHEMA, label="evidence receipt", maximum=maximum
        )
        body = cast(Mapping[str, object], env["body"])
        expected = {
            "access_trace_sha256",
            "atlas_event_sha256",
            "atlas_revision_sha256",
            "atlas_sequence",
            "attention_kind",
            "cartography_input_sketch_sha256",
            "capture_mode",
            "capture_event_sha256",
            "capture_revision_sha256",
            "capture_sequence",
            "context_kind",
            "entry",
            "identity_sha256",
            "manifest_sha256",
            "measurement_sha256",
            "model_pin_sha256",
            "probe_spec_sha256",
            "recomputed_input_sketch_sha256",
            "source_receipt_sha256s",
            "tensors",
            "token_sha256",
            "weight_revision_sha256",
        }
        if (
            set(body) != expected
            or body.get("attention_kind") != SUBSPACE_ATTENTION_KIND
            or body.get("context_kind") != SUBSPACE_CONTEXT_KIND
            or not isinstance(body.get("entry"), Mapping)
            or not isinstance(body.get("tensors"), list)
            or not isinstance(body.get("source_receipt_sha256s"), list)
        ):
            raise QwenMlpEvidenceIntegrityError("evidence receipt body is invalid")
        entry_body = cast(Mapping[str, object], body["entry"])
        entry = CapturePlanEntry(
            cast(int, entry_body.get("ordinal")),
            cast(str, entry_body.get("split")),
            cast(int, entry_body.get("layer")),
            cast(str, entry_body.get("prompt_sha256")),
        )
        result = cls(
            entry=entry,
            capture_mode=cast(str, body.get("capture_mode")),
            manifest_sha256=cast(str, body.get("manifest_sha256")),
            model_pin_sha256=cast(str, body.get("model_pin_sha256")),
            token_sha256=cast(str, body.get("token_sha256")),
            probe_spec_sha256=cast(str, body.get("probe_spec_sha256")),
            atlas_sequence=cast(int, body.get("atlas_sequence")),
            atlas_event_sha256=cast(str, body.get("atlas_event_sha256")),
            atlas_revision_sha256=cast(str, body.get("atlas_revision_sha256")),
            measurement_sha256=cast(str, body.get("measurement_sha256")),
            weight_revision_sha256=cast(str, body.get("weight_revision_sha256")),
            access_trace_sha256=cast(str, body.get("access_trace_sha256")),
            source_receipt_sha256s=tuple(
                cast(list[str], body.get("source_receipt_sha256s"))
            ),
            tensors=tuple(
                BinaryTensorRef.from_record(v)
                for v in cast(list[object], body.get("tensors"))
            ),
            cartography_input_sketch_sha256=cast(
                str, body.get("cartography_input_sketch_sha256")
            ),
            recomputed_input_sketch_sha256=cast(
                str, body.get("recomputed_input_sketch_sha256")
            ),
        )
        if (
            body.get("identity_sha256") != result.identity_sha256
            or body.get("capture_sequence") != result.entry.ordinal
            or body.get("capture_event_sha256") != result.capture_event_sha256
            or body.get("capture_revision_sha256") != result.capture_revision_sha256
            or result.to_bytes() != data
        ):
            raise QwenMlpEvidenceIntegrityError(
                "evidence receipt reconstruction changed"
            )
        return result


@dataclass(frozen=True, slots=True)
class MlpProjectionVerificationReceipt:
    evidence_receipt_sha256: str
    verifier_sha256: str
    verifier_evidence_sha256: str
    replay_access_trace_sha256: str
    storage_exact: bool
    input_sketch_exact: bool
    gate_exact: bool
    up_exact: bool
    output_exact: bool

    def __post_init__(self) -> None:
        for field in (
            "evidence_receipt_sha256",
            "verifier_sha256",
            "verifier_evidence_sha256",
            "replay_access_trace_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        for field in (
            "storage_exact",
            "input_sketch_exact",
            "gate_exact",
            "up_exact",
            "output_exact",
        ):
            if not isinstance(getattr(self, field), bool):
                raise TypeError(f"{field} must be bool")

    @property
    def accepted(self) -> bool:
        return all(
            (
                self.storage_exact,
                self.input_sketch_exact,
                self.gate_exact,
                self.up_exact,
                self.output_exact,
            )
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        return _sealed(
            VERIFICATION_SCHEMA,
            {
                "evidence_receipt_sha256": self.evidence_receipt_sha256,
                "gate_exact": self.gate_exact,
                "input_sketch_exact": self.input_sketch_exact,
                "output_exact": self.output_exact,
                "replay_access_trace_sha256": self.replay_access_trace_sha256,
                "storage_exact": self.storage_exact,
                "up_exact": self.up_exact,
                "verifier_evidence_sha256": self.verifier_evidence_sha256,
                "verifier_sha256": self.verifier_sha256,
            },
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_document())

    @classmethod
    def from_bytes(cls, data: bytes) -> "MlpProjectionVerificationReceipt":
        env = _strict_json(
            data, schema=VERIFICATION_SCHEMA, label="verification", maximum=64 * 1024
        )
        body = cast(Mapping[str, object], env["body"])
        if set(body) != {
            "evidence_receipt_sha256",
            "gate_exact",
            "input_sketch_exact",
            "output_exact",
            "replay_access_trace_sha256",
            "storage_exact",
            "up_exact",
            "verifier_evidence_sha256",
            "verifier_sha256",
        }:
            raise QwenMlpEvidenceIntegrityError("verification body is invalid")
        result = cls(**dict(body))  # type: ignore[arg-type]
        if result.to_bytes() != data:
            raise QwenMlpEvidenceIntegrityError("verification reconstruction changed")
        return result


@runtime_checkable
class MlpProjectionVerifier(Protocol):
    verifier_sha256: str

    def verify(
        self, receipt: MlpEvidenceReceipt, bank: "QwenMlpEvidenceBank"
    ) -> MlpProjectionVerificationReceipt: ...


@dataclass(frozen=True, slots=True)
class MlpEvidenceJournalState:
    generation: int
    previous_state_sha256: str | None
    appended_receipt_sha256: str | None
    appended_verification_sha256: str | None
    receipt_count: int
    referenced_tensor_bytes: int
    split_counts: tuple[int, int, int]

    def __post_init__(self) -> None:
        for field in ("generation", "receipt_count", "referenced_tensor_bytes"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field} is invalid")
        counts = tuple(self.split_counts)
        if (
            len(counts) != 3
            or any(
                isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts
            )
            or sum(counts) != self.receipt_count
        ):
            raise ValueError("split counts are invalid")
        if self.generation == 0:
            if (
                any(
                    v is not None
                    for v in (
                        self.previous_state_sha256,
                        self.appended_receipt_sha256,
                        self.appended_verification_sha256,
                    )
                )
                or self.receipt_count
                or self.referenced_tensor_bytes
            ):
                raise ValueError("root journal state is invalid")
        else:
            for field in (
                "previous_state_sha256",
                "appended_receipt_sha256",
                "appended_verification_sha256",
            ):
                require_sha256(getattr(self, field), field=field)
            if self.generation != self.receipt_count:
                raise ValueError("journal generation/count differ")
        object.__setattr__(self, "split_counts", counts)

    @classmethod
    def initial(cls) -> "MlpEvidenceJournalState":
        return cls(0, None, None, None, 0, 0, (0, 0, 0))

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        return _sealed(
            JOURNAL_STATE_SCHEMA,
            {
                "appended_receipt_sha256": self.appended_receipt_sha256,
                "appended_verification_sha256": self.appended_verification_sha256,
                "generation": self.generation,
                "previous_state_sha256": self.previous_state_sha256,
                "receipt_count": self.receipt_count,
                "referenced_tensor_bytes": self.referenced_tensor_bytes,
                "split_counts": list(self.split_counts),
            },
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_document())

    @classmethod
    def from_bytes(cls, data: bytes) -> "MlpEvidenceJournalState":
        env = _strict_json(
            data, schema=JOURNAL_STATE_SCHEMA, label="journal state", maximum=64 * 1024
        )
        body = cast(Mapping[str, object], env["body"])
        if set(body) != {
            "appended_receipt_sha256",
            "appended_verification_sha256",
            "generation",
            "previous_state_sha256",
            "receipt_count",
            "referenced_tensor_bytes",
            "split_counts",
        } or not isinstance(body.get("split_counts"), list):
            raise QwenMlpEvidenceIntegrityError("journal state body is invalid")
        result = cls(
            generation=cast(int, body.get("generation")),
            previous_state_sha256=cast(str | None, body.get("previous_state_sha256")),
            appended_receipt_sha256=cast(
                str | None, body.get("appended_receipt_sha256")
            ),
            appended_verification_sha256=cast(
                str | None, body.get("appended_verification_sha256")
            ),
            receipt_count=cast(int, body.get("receipt_count")),
            referenced_tensor_bytes=cast(int, body.get("referenced_tensor_bytes")),
            split_counts=tuple(cast(list[int], body.get("split_counts"))),
        )
        if result.to_bytes() != data:
            raise QwenMlpEvidenceIntegrityError("journal state reconstruction changed")
        return result


@dataclass(frozen=True, slots=True)
class MlpEvidencePublication:
    receipt_sha256: str
    verification_sha256: str
    state_sha256: str
    generation: int
    changed: bool


@dataclass(frozen=True, slots=True)
class MlpEvidenceAudit:
    clean: bool
    state_sha256: str
    receipt_count: int
    referenced_objects: int
    orphan_objects: tuple[str, ...]
    orphan_histories: tuple[str, ...]
    orphan_receipts: tuple[str, ...]
    orphan_verifications: tuple[str, ...]
    orphan_verifier_evidence: tuple[str, ...]


class QwenMlpEvidenceBank:
    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        budget: MlpEvidenceBudget | None = None,
        fault_injector: Callable[[str], None] | None = None,
        deferred_tensor_splits: Sequence[str] = (),
    ) -> None:
        self.root = Path(root)
        self.budget = MlpEvidenceBudget() if budget is None else budget
        if not isinstance(self.budget, MlpEvidenceBudget):
            raise TypeError("budget must be MlpEvidenceBudget")
        if fault_injector is not None and not callable(fault_injector):
            raise TypeError("fault_injector must be callable")
        deferred = tuple(deferred_tensor_splits)
        if len(set(deferred)) != len(deferred) or any(
            split not in CAPTURE_SPLITS for split in deferred
        ):
            raise ValueError("deferred_tensor_splits are invalid or duplicated")
        self.fault_injector = fault_injector
        self._eager_tensor_splits = frozenset(CAPTURE_SPLITS) - frozenset(deferred)
        self._ensure_dir(self.root)
        for name in (
            "objects",
            "receipts",
            "verifications",
            "verifier-evidence",
            "history",
            "commits",
            "staging",
        ):
            self._ensure_dir(self.root / name)
        self._initialize()

    @staticmethod
    def _ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, mode=0o700, exist_ok=True)
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise QwenMlpEvidenceIntegrityError(
                f"managed path is not a real directory: {path}"
            )

    @contextmanager
    def _locked(self):
        fd = os.open(
            self.root / _LOCK,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise QwenMlpEvidenceIntegrityError("lock is not a regular file")
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _fault(self, stage: str) -> None:
        if self.fault_injector is not None:
            self.fault_injector(stage)

    @staticmethod
    def _validate_split_transition(
        split_counts: tuple[int, int, int], split: str
    ) -> None:
        train, calibration, holdout = split_counts
        if split == "train":
            valid = calibration == 0 and holdout == 0 and train < 25
        elif split == "calibration":
            valid = train == 25 and holdout == 0 and calibration < 10
        elif split == "holdout":
            valid = train == 25 and calibration == 10 and holdout < 5
        else:  # protected by CapturePlanEntry, kept fail-closed.
            valid = False
        if not valid:
            raise QwenMlpEvidenceIntegrityError(
                "MLP evidence split violates the canonical 25/10/5 state machine"
            )

    def preflight_split_append(self, split: str) -> None:
        if split not in CAPTURE_SPLITS:
            raise ValueError("capture split is invalid")
        with self._locked():
            self._recover_unlocked()
            self._validate_split_transition(
                self._load_head_unlocked().split_counts, split
            )

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        fd = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _write_all(fd: int, data: bytes) -> None:
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            count = os.write(fd, view[offset:])
            if count <= 0:
                raise OSError("short write")
            offset += count

    @staticmethod
    def _stable_read(path: Path, maximum: int) -> bytes:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
                raise QwenMlpEvidenceIntegrityError(f"invalid bounded file: {path}")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(fd, 1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum:
                    raise QwenMlpEvidenceIntegrityError(
                        f"file grew beyond bound: {path}"
                    )
                chunks.append(chunk)
            after = os.fstat(fd)
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise QwenMlpEvidenceIntegrityError(f"file changed while read: {path}")
            return b"".join(chunks)
        finally:
            os.close(fd)

    def _shard(self, category: str, digest: str) -> Path:
        require_sha256(digest, field="digest")
        path = self.root / category / digest[:2]
        self._ensure_dir(path)
        return path

    def _path(self, category: str, digest: str, suffix: str) -> Path:
        return self._shard(category, digest) / f"{digest}.{suffix}"

    def _publish_content(
        self, category: str, suffix: str, data: bytes, maximum: int
    ) -> tuple[str, bool]:
        if not isinstance(data, bytes) or len(data) > maximum:
            raise QwenMlpEvidenceCapacityError("immutable payload exceeds bound")
        digest = hashlib.sha256(data).hexdigest()
        final = self._path(category, digest, suffix)
        if final.exists() or final.is_symlink():
            existing = self._stable_read(final, maximum)
            if existing != data:
                raise QwenMlpEvidenceConflictError(
                    "content address contains other bytes"
                )
            return digest, False
        temporary = self.root / "staging" / f".{digest}.{secrets.token_hex(12)}.tmp"
        fd = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            self._write_all(fd, data)
            os.fsync(fd)
            os.fchmod(fd, 0o444)
        finally:
            os.close(fd)
        try:
            try:
                os.link(temporary, final, follow_symlinks=False)
                created = True
                self._fsync_dir(final.parent)
            except FileExistsError:
                if self._stable_read(final, maximum) != data:
                    raise QwenMlpEvidenceConflictError("concurrent content collision")
                created = False
        finally:
            temporary.unlink(missing_ok=True)
            self._fsync_dir(self.root / "staging")
        return digest, created

    def _replace_pointer(self, name: str, data: bytes) -> None:
        temporary = self.root / "staging" / f".{name}.{secrets.token_hex(12)}.tmp"
        fd = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            self._write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(temporary, self.root / name)
        self._fsync_dir(self.root / "staging")
        directory = os.open(
            self.root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _commit_path(self, digest: str) -> Path:
        return self._shard("commits", digest) / f"{digest}.commit"

    def _publish_commit(self, digest: str) -> None:
        data = canonical_json_bytes(
            {"schema": JOURNAL_STATE_SCHEMA, "state_sha256": digest}
        )
        path = self._commit_path(digest)
        if path.exists():
            if self._stable_read(path, 4096) != data:
                raise QwenMlpEvidenceIntegrityError(
                    "commit marker contains other bytes"
                )
            return
        temp = self.root / "staging" / f".commit.{secrets.token_hex(12)}.tmp"
        fd = os.open(
            temp,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
        try:
            self._write_all(fd, data)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            os.link(temp, path, follow_symlinks=False)
            self._fsync_dir(path.parent)
        except FileExistsError:
            if self._stable_read(path, 4096) != data:
                raise QwenMlpEvidenceIntegrityError("commit collision")
        finally:
            temp.unlink(missing_ok=True)
            self._fsync_dir(self.root / "staging")

    def _initialize(self) -> None:
        with self._locked():
            head = self.root / _HEAD
            budget_path = self.root / _BUDGET
            if (
                head.is_symlink()
                or (self.root / _INTENT).is_symlink()
                or budget_path.is_symlink()
            ):
                raise QwenMlpEvidenceIntegrityError(
                    "journal pointers cannot be symlinks"
                )
            budget_data = canonical_json_bytes(
                _sealed(
                    EVIDENCE_BUDGET_SCHEMA,
                    {
                        "budget_sha256": self.budget.sha256,
                        "limits": {
                            name: getattr(self.budget, name)
                            for name in self.budget.__dataclass_fields__
                        },
                    },
                )
            )
            if budget_path.exists():
                if self._stable_read(budget_path, 64 * 1024) != budget_data:
                    raise QwenMlpEvidenceConflictError(
                        "evidence bank budget changed across resume"
                    )
            else:
                self._replace_pointer(_BUDGET, budget_data)
            if not head.exists():
                state = MlpEvidenceJournalState.initial()
                digest, _ = self._publish_content(
                    "history", "json", state.to_bytes(), 64 * 1024
                )
                if digest != state.sha256:
                    raise AssertionError("root history digest changed")
                self._publish_commit(digest)
                self._replace_pointer(_HEAD, state.to_bytes())
            self._recover_unlocked()

    @staticmethod
    def _tensor_raw(value: object) -> tuple[str, str, tuple[int, ...], bytes]:
        try:
            import torch
        except ImportError:  # pragma: no cover - workspace runtime includes torch
            torch = None  # type: ignore[assignment]
        if torch is not None and isinstance(value, torch.Tensor):
            tensor = value.detach().contiguous().to(device="cpu")
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError("tensor contains non-finite values")
            shape = tuple(int(v) for v in tensor.shape)
            if tensor.dtype == torch.bfloat16:
                return (
                    "bfloat16",
                    "bfloat16-bits-le",
                    shape,
                    tensor.view(torch.uint16)
                    .numpy()
                    .astype("<u2", copy=False)
                    .tobytes(),
                )
            if tensor.dtype == torch.float16:
                return (
                    "float16",
                    "float16-bits-le",
                    shape,
                    tensor.view(torch.uint16)
                    .numpy()
                    .astype("<u2", copy=False)
                    .tobytes(),
                )
            if tensor.dtype == torch.float32:
                return (
                    "float32",
                    "float32-le",
                    shape,
                    tensor.numpy().astype("<f4", copy=False).tobytes(),
                )
            if tensor.dtype == torch.float64:
                return (
                    "float64",
                    "float64-le",
                    shape,
                    tensor.numpy().astype("<f8", copy=False).tobytes(),
                )
            raise TypeError("tensor dtype is unsupported")
        if type(value) is not np.ndarray or value.dtype not in (
            np.dtype(np.float16),
            np.dtype(np.float32),
            np.dtype(np.float64),
        ):
            raise TypeError(
                "capture value must be a floating torch tensor or numpy array"
            )
        if not np.isfinite(value).all():
            raise ValueError("tensor contains non-finite values")
        source = {
            np.dtype(np.float16): "float16",
            np.dtype(np.float32): "float32",
            np.dtype(np.float64): "float64",
        }[value.dtype]
        storage = {
            "float16": "float16-bits-le",
            "float32": "float32-le",
            "float64": "float64-le",
        }[source]
        dtype = {"float16": "<f2", "float32": "<f4", "float64": "<f8"}[source]
        return (
            source,
            storage,
            tuple(int(v) for v in value.shape),
            np.asarray(value, dtype=dtype, order="C").tobytes(),
        )

    @staticmethod
    def tensor_nbytes(value: object) -> tuple[tuple[int, ...], int]:
        try:
            import torch
        except ImportError:  # pragma: no cover
            torch = None  # type: ignore[assignment]
        if torch is not None and isinstance(value, torch.Tensor):
            if value.dtype not in {
                torch.bfloat16,
                torch.float16,
                torch.float32,
                torch.float64,
            }:
                raise TypeError("tensor dtype is unsupported")
            shape = tuple(int(v) for v in value.shape)
            return shape, math.prod(shape) * int(value.element_size())
        if type(value) is not np.ndarray or value.dtype not in (
            np.dtype(np.float16),
            np.dtype(np.float32),
            np.dtype(np.float64),
        ):
            raise TypeError(
                "capture value must be a floating torch tensor or numpy array"
            )
        return tuple(int(v) for v in value.shape), int(value.nbytes)

    def publish_tensor(self, stage: str, layer: int, value: object) -> BinaryTensorRef:
        if stage not in CAPTURE_STAGES:
            raise ValueError("capture stage is invalid")
        shape, planned_bytes = self.tensor_nbytes(value)
        if (
            len(shape) < 2
            or math.prod(shape[:-1]) > self.budget.max_rows_per_group
            or shape[-1] > self.budget.max_feature_dimension
        ):
            raise QwenMlpEvidenceCapacityError("tensor shape exceeds capture budget")
        if planned_bytes > self.budget.max_tensor_bytes:
            raise QwenMlpEvidenceCapacityError("tensor bytes exceed capture budget")
        source, storage, actual_shape, raw = self._tensor_raw(value)
        if actual_shape != shape or len(raw) != planned_bytes:
            raise QwenMlpEvidenceIntegrityError("tensor changed after preflight")
        raw_sha = hashlib.sha256(raw).hexdigest()
        header = canonical_json_bytes(
            {
                "byte_order": "little",
                "finite_policy": "finite",
                "layer": layer,
                "raw_nbytes": len(raw),
                "raw_sha256": raw_sha,
                "schema": TENSOR_SCHEMA,
                "shape": list(shape),
                "source_dtype": source,
                "stage": stage,
                "storage_dtype": storage,
            }
        )
        if len(header) > self.budget.max_header_bytes:
            raise QwenMlpEvidenceCapacityError("tensor header exceeds budget")
        frame = _MAGIC + struct.pack("<I", len(header)) + header + raw
        digest, _ = self._publish_content(
            "objects",
            "tensor",
            frame,
            self.budget.max_tensor_bytes + self.budget.max_header_bytes + 12,
        )
        row_hashes: tuple[str, ...] = ()
        if stage == "mlp.output":
            rows = math.prod(shape[:-1])
            row_bytes = len(raw) // rows
            row_hashes = tuple(
                hashlib.sha256(
                    canonical_json_bytes(
                        {"row": i, "shape": [shape[-1]], "source_dtype": source}
                    )
                    + raw[i * row_bytes : (i + 1) * row_bytes]
                ).hexdigest()
                for i in range(rows)
            )
        ref = BinaryTensorRef(
            stage,
            layer,
            digest,
            hashlib.sha256(header).hexdigest(),
            raw_sha,
            source,
            storage,
            shape,
            len(raw),
            row_hashes,
        )
        self.restore_tensor(ref)
        return ref

    def _read_tensor_frame(
        self, ref: BinaryTensorRef
    ) -> tuple[Mapping[str, object], bytes]:
        path = self._path("objects", ref.object_sha256, "tensor")
        data = self._stable_read(
            path, self.budget.max_tensor_bytes + self.budget.max_header_bytes + 12
        )
        if (
            hashlib.sha256(data).hexdigest() != ref.object_sha256
            or len(data) < 12
            or data[:8] != _MAGIC
        ):
            raise QwenMlpEvidenceIntegrityError("tensor object framing/hash is invalid")
        header_len = struct.unpack("<I", data[8:12])[0]
        if not 0 < header_len <= self.budget.max_header_bytes or 12 + header_len > len(
            data
        ):
            raise QwenMlpEvidenceIntegrityError("tensor header length is invalid")
        header_raw = data[12 : 12 + header_len]
        try:
            header = json.loads(header_raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QwenMlpEvidenceIntegrityError(
                "tensor header is invalid JSON"
            ) from exc
        if (
            not isinstance(header, Mapping)
            or canonical_json_bytes(header) != header_raw
            or hashlib.sha256(header_raw).hexdigest() != ref.header_sha256
        ):
            raise QwenMlpEvidenceIntegrityError(
                "tensor header hash/canonical form is invalid"
            )
        raw = data[12 + header_len :]
        expected = {
            "byte_order": "little",
            "finite_policy": "finite",
            "layer": ref.layer,
            "raw_nbytes": ref.raw_nbytes,
            "raw_sha256": ref.raw_sha256,
            "schema": TENSOR_SCHEMA,
            "shape": list(ref.shape),
            "source_dtype": ref.source_dtype,
            "stage": ref.stage,
            "storage_dtype": ref.storage_dtype,
        }
        if (
            dict(header) != expected
            or len(raw) != ref.raw_nbytes
            or hashlib.sha256(raw).hexdigest() != ref.raw_sha256
        ):
            raise QwenMlpEvidenceIntegrityError(
                "tensor header/raw differs from reference"
            )
        return header, raw

    def restore_tensor(self, ref: BinaryTensorRef) -> np.ndarray:
        if not isinstance(ref, BinaryTensorRef):
            raise TypeError("ref must be BinaryTensorRef")
        _header, raw = self._read_tensor_frame(ref)
        if ref.storage_dtype == "bfloat16-bits-le":
            bits = np.frombuffer(raw, dtype="<u2").astype("<u4") << 16
            result = bits.view("<f4").astype(np.float64)
        elif ref.storage_dtype == "float16-bits-le":
            result = np.frombuffer(raw, dtype="<f2").astype(np.float64)
        elif ref.storage_dtype == "float32-le":
            result = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        else:
            result = np.frombuffer(raw, dtype="<f8").astype(np.float64)
        result = result.reshape(ref.shape)
        if not np.isfinite(result).all():
            raise QwenMlpEvidenceIntegrityError("restored tensor is non-finite")
        result.flags.writeable = False
        return result

    def _history_path(self, digest: str) -> Path:
        return self._path("history", digest, "json")

    def _load_head_unlocked(self) -> MlpEvidenceJournalState:
        return MlpEvidenceJournalState.from_bytes(
            self._stable_read(self.root / _HEAD, 64 * 1024)
        )

    def _load_history(self, digest: str) -> MlpEvidenceJournalState:
        data = self._stable_read(self._history_path(digest), 64 * 1024)
        state = MlpEvidenceJournalState.from_bytes(data)
        if state.sha256 != digest:
            raise QwenMlpEvidenceIntegrityError("history content address mismatch")
        return state

    def _intent_bytes(self, old: str, new: str) -> bytes:
        return canonical_json_bytes(
            _sealed(
                JOURNAL_INTENT_SCHEMA,
                {"new_state_sha256": new, "old_state_sha256": old},
            )
        )

    def _recover_unlocked(self, *, tensor_splits: frozenset[str] | None = None) -> None:
        selected_splits = (
            self._eager_tensor_splits if tensor_splits is None else tensor_splits
        )
        intent_path = self.root / _INTENT
        if not intent_path.exists():
            self._replay_unlocked(
                self._load_head_unlocked(), tensor_splits=selected_splits
            )
            return
        env = _strict_json(
            self._stable_read(intent_path, 64 * 1024),
            schema=JOURNAL_INTENT_SCHEMA,
            label="intent",
            maximum=64 * 1024,
        )
        body = cast(Mapping[str, object], env["body"])
        if set(body) != {"new_state_sha256", "old_state_sha256"}:
            raise QwenMlpEvidenceIntegrityError("prepared intent is invalid")
        old = require_sha256(body.get("old_state_sha256"), field="old_state_sha256")
        new = require_sha256(body.get("new_state_sha256"), field="new_state_sha256")
        head = self._load_head_unlocked()
        candidate = self._load_history(new)
        if candidate.previous_state_sha256 != old:
            raise QwenMlpEvidenceIntegrityError(
                "prepared state does not extend old head"
            )
        if head.sha256 == old:
            self._replace_pointer(_HEAD, candidate.to_bytes())
        elif head.sha256 != new:
            raise QwenMlpEvidenceIntegrityError("prepared intent matches neither head")
        self._publish_commit(new)
        intent_path.unlink()
        self._replay_unlocked(candidate, tensor_splits=selected_splits)

    def _commit_addresses(self) -> set[str]:
        result: set[str] = set()
        for shard in (self.root / "commits").iterdir():
            if shard.is_symlink() or not shard.is_dir():
                raise QwenMlpEvidenceIntegrityError(
                    "commit inventory contains a non-directory"
                )
            for path in shard.iterdir():
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or not path.name.endswith(".commit")
                ):
                    raise QwenMlpEvidenceIntegrityError(
                        "commit inventory contains an invalid entry"
                    )
                address = path.name[: -len(".commit")]
                require_sha256(address, field="commit address")
                if path != self._commit_path(address):
                    raise QwenMlpEvidenceIntegrityError(
                        "commit filename is not content-bound"
                    )
                result.add(address)
        return result

    def _replay_unlocked(
        self,
        head: MlpEvidenceJournalState,
        *,
        tensor_splits: frozenset[str] | None = None,
    ) -> tuple[MlpEvidenceJournalState, ...]:
        selected_splits = (
            self._eager_tensor_splits if tensor_splits is None else tensor_splits
        )
        chain = [head]
        while chain[-1].generation:
            previous = cast(str, chain[-1].previous_state_sha256)
            parent = self._load_history(previous)
            if (
                parent.generation + 1 != chain[-1].generation
                or not self._commit_path(chain[-1].sha256).is_file()
            ):
                raise QwenMlpEvidenceIntegrityError(
                    "journal ancestry/commit is invalid"
                )
            receipt = self.restore_receipt(cast(str, chain[-1].appended_receipt_sha256))
            verification = self.restore_verification(
                cast(str, chain[-1].appended_verification_sha256)
            )
            self._verify_pair(
                receipt,
                verification,
                restore_tensors=receipt.entry.split in selected_splits,
            )
            expected_counts = list(parent.split_counts)
            expected_counts[CAPTURE_SPLITS.index(receipt.entry.split)] += 1
            if (
                chain[-1].receipt_count != parent.receipt_count + 1
                or chain[-1].referenced_tensor_bytes
                != parent.referenced_tensor_bytes
                + sum(ref.raw_nbytes for ref in receipt.tensors)
                or chain[-1].split_counts != tuple(expected_counts)
            ):
                raise QwenMlpEvidenceIntegrityError(
                    "journal cumulative extension is invalid"
                )
            chain.append(parent)
        if (
            chain[-1] != MlpEvidenceJournalState.initial()
            or not self._commit_path(chain[-1].sha256).is_file()
        ):
            raise QwenMlpEvidenceIntegrityError("journal root is invalid")
        result = tuple(reversed(chain))
        if self._commit_addresses() != {state.sha256 for state in result}:
            raise QwenMlpEvidenceIntegrityError(
                "committed journal contains a fork or disconnected state"
            )
        return result

    def state(self) -> MlpEvidenceJournalState:
        with self._locked():
            self._recover_unlocked()
            return self._load_head_unlocked()

    def state_history(self) -> tuple[MlpEvidenceJournalState, ...]:
        """Return the complete authenticated MLP journal ancestry."""

        with self._locked():
            self._recover_unlocked()
            return self._replay_unlocked(self._load_head_unlocked())

    def _receipt_path(self, digest: str) -> Path:
        return self._path("receipts", digest, "json")

    def _verification_path(self, digest: str) -> Path:
        return self._path("verifications", digest, "json")

    def restore_receipt(self, digest: str) -> MlpEvidenceReceipt:
        address = require_sha256(digest, field="receipt_sha256")
        data = self._stable_read(
            self._receipt_path(address), self.budget.max_receipt_bytes
        )
        if hashlib.sha256(data).hexdigest() != address:
            raise QwenMlpEvidenceIntegrityError("receipt address mismatch")
        return MlpEvidenceReceipt.from_bytes(
            data, maximum=self.budget.max_receipt_bytes
        )

    def restore_verification(self, digest: str) -> MlpProjectionVerificationReceipt:
        address = require_sha256(digest, field="verification_sha256")
        data = self._stable_read(self._verification_path(address), 64 * 1024)
        if hashlib.sha256(data).hexdigest() != address:
            raise QwenMlpEvidenceIntegrityError("verification address mismatch")
        return MlpProjectionVerificationReceipt.from_bytes(data)

    def publish_verifier_evidence(
        self,
        *,
        evidence_receipt_sha256: str,
        verifier_sha256: str,
        evidence: Mapping[str, object],
    ) -> str:
        if not isinstance(evidence, Mapping):
            raise TypeError("verifier evidence must be a mapping")
        document = _sealed(
            VERIFIER_EVIDENCE_SCHEMA,
            {
                "evidence": dict(evidence),
                "evidence_receipt_sha256": require_sha256(
                    evidence_receipt_sha256, field="evidence_receipt_sha256"
                ),
                "verifier_sha256": require_sha256(
                    verifier_sha256, field="verifier_sha256"
                ),
            },
        )
        digest, _ = self._publish_content(
            "verifier-evidence", "json", canonical_json_bytes(document), 64 * 1024
        )
        return digest

    def restore_verifier_evidence(self, digest: str) -> Mapping[str, object]:
        address = require_sha256(digest, field="verifier_evidence_sha256")
        data = self._stable_read(
            self._path("verifier-evidence", address, "json"), 64 * 1024
        )
        if hashlib.sha256(data).hexdigest() != address:
            raise QwenMlpEvidenceIntegrityError("verifier evidence address mismatch")
        envelope = _strict_json(
            data,
            schema=VERIFIER_EVIDENCE_SCHEMA,
            label="verifier evidence",
            maximum=64 * 1024,
        )
        body = cast(Mapping[str, object], envelope["body"])
        if set(body) != {
            "evidence",
            "evidence_receipt_sha256",
            "verifier_sha256",
        } or not isinstance(body.get("evidence"), Mapping):
            raise QwenMlpEvidenceIntegrityError("verifier evidence body is invalid")
        return body

    def _committed_pairs_unlocked(
        self,
        *,
        tensor_splits: frozenset[str] | None = None,
    ) -> tuple[tuple[MlpEvidenceReceipt, MlpProjectionVerificationReceipt], ...]:
        chain = self._replay_unlocked(
            self._load_head_unlocked(), tensor_splits=tensor_splits
        )
        rows = []
        for state in chain[1:]:
            receipt = self.restore_receipt(cast(str, state.appended_receipt_sha256))
            verification = self.restore_verification(
                cast(str, state.appended_verification_sha256)
            )
            rows.append((receipt, verification))
        return tuple(rows)

    def committed_pairs(
        self,
    ) -> tuple[tuple[MlpEvidenceReceipt, MlpProjectionVerificationReceipt], ...]:
        with self._locked():
            self._recover_unlocked()
            return self._committed_pairs_unlocked()

    def _verify_pair(
        self,
        receipt: MlpEvidenceReceipt,
        verification: MlpProjectionVerificationReceipt,
        *,
        restore_tensors: bool = True,
    ) -> None:
        if (
            verification.evidence_receipt_sha256 != receipt.sha256
            or not verification.accepted
        ):
            raise QwenMlpEvidenceIntegrityError(
                "MLP evidence verification was not exact"
            )
        proof = self.restore_verifier_evidence(verification.verifier_evidence_sha256)
        if (
            proof.get("evidence_receipt_sha256") != receipt.sha256
            or proof.get("verifier_sha256") != verification.verifier_sha256
        ):
            raise QwenMlpEvidenceIntegrityError(
                "MLP verifier evidence is bound to another receipt/verifier"
            )
        evidence = cast(Mapping[str, object], proof.get("evidence"))
        expected_proof = {
            "gate_exact": verification.gate_exact,
            "input_sketch_exact": verification.input_sketch_exact,
            "output_exact": verification.output_exact,
            "replay_access_trace_sha256": (verification.replay_access_trace_sha256),
            "storage_exact": verification.storage_exact,
            "tensor_object_sha256s": [row.object_sha256 for row in receipt.tensors],
            "up_exact": verification.up_exact,
        }
        if any(evidence.get(field) != value for field, value in expected_proof.items()):
            raise QwenMlpEvidenceIntegrityError(
                "MLP verifier proof differs from its verification receipt/tensors"
            )
        if restore_tensors:
            for ref in receipt.tensors:
                self.restore_tensor(ref)

    def append_verified(
        self,
        receipt: MlpEvidenceReceipt,
        verifier: MlpProjectionVerifier,
        *,
        expected_head_sha256: str | None = None,
    ) -> MlpEvidencePublication:
        if not isinstance(receipt, MlpEvidenceReceipt) or not isinstance(
            verifier, MlpProjectionVerifier
        ):
            raise TypeError("receipt/verifier types are invalid")
        verification = verifier.verify(receipt, self)
        if not isinstance(verification, MlpProjectionVerificationReceipt):
            raise QwenMlpEvidenceIntegrityError(
                "MLP verifier returned an invalid verification receipt"
            )
        if verification.verifier_sha256 != require_sha256(
            verifier.verifier_sha256, field="verifier.verifier_sha256"
        ):
            raise QwenMlpEvidenceIntegrityError(
                "MLP verifier identity differs from its verification receipt"
            )
        if len(receipt.to_bytes()) > self.budget.max_receipt_bytes:
            raise QwenMlpEvidenceCapacityError("receipt exceeds budget")
        self._verify_pair(receipt, verification)
        rsha, _ = self._publish_content(
            "receipts", "json", receipt.to_bytes(), self.budget.max_receipt_bytes
        )
        vsha, _ = self._publish_content(
            "verifications", "json", verification.to_bytes(), 64 * 1024
        )
        self._fault("after-receipt-verification")
        with self._locked():
            self._recover_unlocked()
            head = self._load_head_unlocked()
            if (
                expected_head_sha256 is not None
                and require_sha256(expected_head_sha256, field="expected_head_sha256")
                != head.sha256
            ):
                raise QwenMlpEvidenceConflictError("journal head CAS failed")
            committed = self._committed_pairs_unlocked()
            same = [
                row
                for row in committed
                if row[0].identity_sha256 == receipt.identity_sha256
            ]
            if same:
                if (
                    len(same) != 1
                    or same[0][0].to_bytes() != receipt.to_bytes()
                    or same[0][1].to_bytes() != verification.to_bytes()
                ):
                    raise QwenMlpEvidenceConflictError("capture identity was rebound")
                return MlpEvidencePublication(
                    rsha, vsha, head.sha256, head.generation, False
                )
            if head.receipt_count >= self.budget.max_groups:
                raise QwenMlpEvidenceCapacityError("journal group bound reached")
            self._validate_split_transition(head.split_counts, receipt.entry.split)
            added_bytes = sum(ref.raw_nbytes for ref in receipt.tensors)
            total_bytes = head.referenced_tensor_bytes + added_bytes
            if total_bytes > self.budget.max_total_referenced_bytes:
                raise QwenMlpEvidenceCapacityError("journal tensor byte bound reached")
            counts = list(head.split_counts)
            counts[CAPTURE_SPLITS.index(receipt.entry.split)] += 1
            updated = MlpEvidenceJournalState(
                head.generation + 1,
                head.sha256,
                rsha,
                vsha,
                head.receipt_count + 1,
                total_bytes,
                tuple(counts),
            )
            hsha, _ = self._publish_content(
                "history", "json", updated.to_bytes(), 64 * 1024
            )
            if hsha != updated.sha256:
                raise AssertionError("history address changed")
            self._fault("after-history")
            self._replace_pointer(
                _INTENT, self._intent_bytes(head.sha256, updated.sha256)
            )
            self._fault("after-intent")
            if self._load_head_unlocked().sha256 != head.sha256:
                raise QwenMlpEvidenceConflictError("journal head changed before CAS")
            self._replace_pointer(_HEAD, updated.to_bytes())
            self._fault("after-head")
            self._publish_commit(updated.sha256)
            self._fault("after-commit")
            (self.root / _INTENT).unlink(missing_ok=True)
            if self._load_head_unlocked().to_bytes() != updated.to_bytes():
                raise QwenMlpEvidenceIntegrityError("journal append failed roundtrip")
            return MlpEvidencePublication(
                rsha, vsha, updated.sha256, updated.generation, True
            )

    def audit(self) -> MlpEvidenceAudit:
        all_splits = frozenset(CAPTURE_SPLITS)
        with self._locked():
            self._recover_unlocked(tensor_splits=all_splits)
            pairs = self._committed_pairs_unlocked(tensor_splits=all_splits)
            head = self._load_head_unlocked()
            referenced = {
                ref.object_sha256 for receipt, _ in pairs for ref in receipt.tensors
            }
            objects = {
                path.stem
                for shard in (self.root / "objects").iterdir()
                if shard.is_dir()
                for path in shard.glob("*.tensor")
            }
            histories = {
                path.stem
                for shard in (self.root / "history").iterdir()
                if shard.is_dir()
                for path in shard.glob("*.json")
            }
            chain = {
                state.sha256
                for state in self._replay_unlocked(head, tensor_splits=all_splits)
            }
            committed_receipts = {receipt.sha256 for receipt, _ in pairs}
            committed_verifications = {verification.sha256 for _, verification in pairs}
            receipts = {
                path.stem
                for shard in (self.root / "receipts").iterdir()
                if shard.is_dir()
                for path in shard.glob("*.json")
            }
            verifications = {
                path.stem
                for shard in (self.root / "verifications").iterdir()
                if shard.is_dir()
                for path in shard.glob("*.json")
            }
            verifier_evidence = {
                path.stem
                for shard in (self.root / "verifier-evidence").iterdir()
                if shard.is_dir()
                for path in shard.glob("*.json")
            }
            orphan_objects = tuple(sorted(objects - referenced))
            orphan_histories = tuple(sorted(histories - chain))
            orphan_receipts = tuple(sorted(receipts - committed_receipts))
            orphan_verifications = tuple(
                sorted(verifications - committed_verifications)
            )
            referenced_verifier_evidence = {
                verification.verifier_evidence_sha256 for _, verification in pairs
            }
            orphan_verifier_evidence = tuple(
                sorted(verifier_evidence - referenced_verifier_evidence)
            )
            # The verifier proof is a first-class replay artifact. Orphans
            # are never silently treated as a clean bank.
            return MlpEvidenceAudit(
                not orphan_objects
                and not orphan_histories
                and not orphan_receipts
                and not orphan_verifications
                and not orphan_verifier_evidence,
                head.sha256,
                len(pairs),
                len(referenced),
                orphan_objects,
                orphan_histories,
                orphan_receipts,
                orphan_verifications,
                orphan_verifier_evidence,
            )

    def build_subspace_corpus(
        self, *, allowed_splits: Sequence[str] | None = None
    ) -> SubspaceCorpus:
        selected_splits = (
            CAPTURE_SPLITS if allowed_splits is None else tuple(allowed_splits)
        )
        if (
            not selected_splits
            or len(set(selected_splits)) != len(selected_splits)
            or any(split not in CAPTURE_SPLITS for split in selected_splits)
        ):
            raise ValueError("allowed_splits are invalid or duplicated")
        tensor_splits = frozenset(selected_splits)
        with self._locked():
            self._recover_unlocked(tensor_splits=tensor_splits)
            pairs = tuple(
                pair
                for pair in self._committed_pairs_unlocked(tensor_splits=tensor_splits)
                if pair[0].entry.split in selected_splits
            )
        if not pairs:
            raise QwenMlpEvidenceIntegrityError("bank contains no verified MLP groups")
        if (
            len({receipt.manifest_sha256 for receipt, _ in pairs}) != 1
            or len({receipt.model_pin_sha256 for receipt, _ in pairs}) != 1
            or len({verification.verifier_sha256 for _receipt, verification in pairs})
            != 1
            or any(receipt.capture_mode != "live-exact" for receipt, _ in pairs)
        ):
            raise QwenMlpEvidenceIntegrityError(
                "production SubspaceCorpus requires one live manifest/model/verifier pin"
            )
        groups = []
        for receipt, verification in pairs:
            refs = {ref.stage: ref for ref in receipt.tensors}
            context = self.restore_tensor(refs["mlp.input"]).reshape(
                -1, refs["mlp.input"].shape[-1]
            )
            gate = self.restore_tensor(refs["mlp.gate"]).reshape(
                -1, refs["mlp.gate"].shape[-1]
            )
            up = self.restore_tensor(refs["mlp.up"]).reshape(
                -1, refs["mlp.up"].shape[-1]
            )
            outputs = refs["mlp.output"].row_sha256s
            source = tuple(
                sorted(
                    set(receipt.source_receipt_sha256s)
                    | {
                        receipt.probe_spec_sha256,
                        receipt.measurement_sha256,
                        receipt.access_trace_sha256,
                        verification.sha256,
                    }
                )
            )
            projection_verifier = verification.verifier_sha256
            projection_evidence = projection_evidence_sha256(
                model_pin_sha256=receipt.model_pin_sha256,
                graph_revision_sha256=receipt.capture_revision_sha256,
                layer=receipt.entry.layer,
                source_receipt_sha256s=source,
                projection_verifier_sha256=projection_verifier,
                context_states=context,
                gate_projection=gate,
                up_projection=up,
            )
            output_verifier = verification.verifier_sha256
            groups.append(
                SubspaceObservationGroup(
                    logical_time=receipt.entry.ordinal,
                    group_sha256=receipt.identity_sha256,
                    model_pin_sha256=receipt.model_pin_sha256,
                    graph_revision_sha256=receipt.capture_revision_sha256,
                    graph_sequence=receipt.entry.ordinal,
                    graph_event_sha256=receipt.capture_event_sha256,
                    layer=receipt.entry.layer,
                    prompt_sha256=receipt.entry.prompt_sha256,
                    source_receipt_sha256s=source,
                    projection_verifier_sha256=projection_verifier,
                    projection_evidence_sha256=projection_evidence,
                    output_verifier_sha256=output_verifier,
                    output_evidence_sha256=output_evidence_sha256(
                        output_payload_sha256s=outputs,
                        output_verifier_sha256=output_verifier,
                        source_receipt_sha256s=source,
                    ),
                    context_states=context,
                    gate_projection=gate,
                    up_projection=up,
                    output_payload_sha256s=outputs,
                )
            )
        groups.sort(key=lambda row: (row.logical_time, row.group_sha256))
        return SubspaceCorpus(groups[0].model_pin_sha256, tuple(groups))


class ExactMlpBoundaryCapture:
    def __init__(self, bank: QwenMlpEvidenceBank) -> None:
        if not isinstance(bank, QwenMlpEvidenceBank):
            raise TypeError("bank must be QwenMlpEvidenceBank")
        self.bank = bank
        self.entry: CapturePlanEntry | None = None
        self.refs: list[BinaryTensorRef] = []

    def begin_group(self, entry: CapturePlanEntry) -> None:
        if self.entry is not None:
            raise QwenMlpEvidenceConflictError("capture group already active")
        if not isinstance(entry, CapturePlanEntry):
            raise TypeError("entry must be CapturePlanEntry")
        self.entry = entry
        self.refs = []

    def __call__(self, layer: int, stage: str, value: object) -> None:
        if self.entry is None:
            raise QwenMlpEvidenceConflictError("capture group was not begun")
        expected = (
            CAPTURE_STAGES[len(self.refs)]
            if len(self.refs) < len(CAPTURE_STAGES)
            else None
        )
        if layer != self.entry.layer or stage != expected:
            raise QwenMlpEvidenceIntegrityError(
                "MLP boundary stage order/layer changed"
            )
        _shape, planned_bytes = self.bank.tensor_nbytes(value)
        if (
            sum(v.raw_nbytes for v in self.refs) + planned_bytes
            > self.bank.budget.max_group_bytes
        ):
            raise QwenMlpEvidenceCapacityError("capture group bytes exceed budget")
        ref = self.bank.publish_tensor(stage, layer, value)
        self.refs.append(ref)

    def finalize(self, **pins: object) -> MlpEvidenceReceipt:
        if self.entry is None or len(self.refs) != len(CAPTURE_STAGES):
            raise QwenMlpEvidenceIntegrityError("capture group is incomplete")
        entry = self.entry
        self.entry = None
        try:
            return MlpEvidenceReceipt(entry=entry, tensors=tuple(self.refs), **pins)  # type: ignore[arg-type]
        finally:
            self.refs = []


@runtime_checkable
class ExactMlpCaptureRunner(MlpProjectionVerifier, Protocol):
    capture_mode: str

    def capture(
        self, entry: CapturePlanEntry, sink: ExactMlpBoundaryCapture
    ) -> MlpEvidenceReceipt: ...


def run_capture_manifest(
    manifest: CaptureManifest,
    bank: QwenMlpEvidenceBank,
    runner: ExactMlpCaptureRunner,
    *,
    allowed_splits: Sequence[str] | None = None,
    max_new_groups: int | None = None,
    max_seconds: float | None = None,
) -> tuple[MlpEvidencePublication, ...]:
    if (
        not isinstance(manifest, CaptureManifest)
        or not isinstance(bank, QwenMlpEvidenceBank)
        or not isinstance(runner, ExactMlpCaptureRunner)
    ):
        raise TypeError("manifest/bank/runner types are invalid")
    if runner.capture_mode not in {"fixture", "live-exact"}:
        raise ValueError("runner capture_mode is invalid")
    selected_splits = (
        CAPTURE_SPLITS if allowed_splits is None else tuple(allowed_splits)
    )
    if (
        not selected_splits
        or len(set(selected_splits)) != len(selected_splits)
        or any(split not in CAPTURE_SPLITS for split in selected_splits)
    ):
        raise ValueError("allowed_splits are invalid or duplicated")
    if max_new_groups is not None and (
        isinstance(max_new_groups, bool)
        or not isinstance(max_new_groups, int)
        or max_new_groups < 1
    ):
        raise ValueError("max_new_groups must be positive or None")
    atomic_group_size = getattr(runner, "atomic_capture_group_size", 1)
    if (
        isinstance(atomic_group_size, bool)
        or not isinstance(atomic_group_size, int)
        or atomic_group_size < 1
    ):
        raise QwenMlpEvidenceIntegrityError(
            "runner atomic_capture_group_size is invalid"
        )
    if max_new_groups is not None and max_new_groups % atomic_group_size != 0:
        raise ValueError("max_new_groups must align to the runner atomic capture group")
    if max_seconds is not None and atomic_group_size > 1:
        raise ValueError(
            "max_seconds cannot hard-bound an atomic multi-forward capture; "
            "use max_new_groups"
        )
    if max_seconds is not None and (
        isinstance(max_seconds, bool)
        or not isinstance(max_seconds, (int, float))
        or not math.isfinite(float(max_seconds))
        or float(max_seconds) <= 0.0
    ):
        raise ValueError("max_seconds must be finite positive or None")
    started = time.monotonic()
    active_atomic_group: str | None = None
    publications = []
    committed = {
        (
            receipt.entry.sha256,
            receipt.manifest_sha256,
            receipt.model_pin_sha256,
            receipt.capture_mode,
        )
        for receipt, _ in bank.committed_pairs()
    }
    atomic_group_fn = getattr(runner, "atomic_capture_group", None)
    pending_by_atomic_group: dict[str, int] = {}
    if callable(atomic_group_fn):
        for planned in manifest.entries:
            if (
                planned.split not in selected_splits
                or (
                    planned.sha256,
                    manifest.sha256,
                    manifest.model_pin_sha256,
                    runner.capture_mode,
                )
                in committed
            ):
                continue
            planned_group = atomic_group_fn(planned)
            if not isinstance(planned_group, str) or not planned_group:
                raise QwenMlpEvidenceIntegrityError(
                    "runner returned an invalid atomic capture group"
                )
            pending_by_atomic_group[planned_group] = (
                pending_by_atomic_group.get(planned_group, 0) + 1
            )
    for entry in manifest.entries:
        if entry.split not in selected_splits:
            continue
        if (
            entry.sha256,
            manifest.sha256,
            manifest.model_pin_sha256,
            runner.capture_mode,
        ) in committed:
            continue
        atomic_group = None
        if callable(atomic_group_fn):
            atomic_group = atomic_group_fn(entry)
            if not isinstance(atomic_group, str) or not atomic_group:
                raise QwenMlpEvidenceIntegrityError(
                    "runner returned an invalid atomic capture group"
                )
        at_boundary = atomic_group is None or atomic_group != active_atomic_group
        pending_in_group = (
            1 if atomic_group is None else pending_by_atomic_group[atomic_group]
        )
        if at_boundary and (
            (
                max_new_groups is not None
                and len(publications) + pending_in_group > max_new_groups
            )
            or (max_seconds is not None and time.monotonic() - started >= max_seconds)
        ):
            break
        active_atomic_group = atomic_group
        bank.preflight_split_append(entry.split)
        sink = ExactMlpBoundaryCapture(bank)
        receipt = runner.capture(entry, sink)
        if (
            receipt.manifest_sha256 != manifest.sha256
            or receipt.model_pin_sha256 != manifest.model_pin_sha256
            or receipt.entry != entry
            or receipt.capture_mode != runner.capture_mode
        ):
            raise QwenMlpEvidenceIntegrityError(
                "runner returned evidence for another manifest entry"
            )
        publications.append(bank.append_verified(receipt, runner))
    return tuple(publications)


__all__ = [
    "CAPTURE_PLAN_LAYERS",
    "CAPTURE_SPLITS",
    "CAPTURE_STAGES",
    "EVIDENCE_BUDGET_SCHEMA",
    "VERIFIER_EVIDENCE_SCHEMA",
    "BinaryTensorRef",
    "CaptureManifest",
    "CapturePlanEntry",
    "ExactMlpBoundaryCapture",
    "ExactMlpCaptureRunner",
    "MlpEvidenceAudit",
    "MlpEvidenceBudget",
    "MlpEvidenceJournalState",
    "MlpEvidencePublication",
    "MlpEvidenceReceipt",
    "MlpProjectionVerificationReceipt",
    "MlpProjectionVerifier",
    "QwenMlpEvidenceBank",
    "QwenMlpEvidenceCapacityError",
    "QwenMlpEvidenceConflictError",
    "QwenMlpEvidenceError",
    "QwenMlpEvidenceIntegrityError",
    "canonical_capture_plan",
    "run_capture_manifest",
]
