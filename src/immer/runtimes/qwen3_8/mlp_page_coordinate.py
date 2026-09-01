"""Exact K1 MLP page coordinates keyed by Qwen BF16 activation rows.

The bank stores route coordinates, never activations or MLP outputs.  One key
seals the transport-neutral runtime math, Q4 identity, static page-router
identity, layer, absolute position, and exact BF16 MLP input.  Captures and
hits remain request-local until an exclusive accepted-position boundary is
committed, so rejected speculative work cannot enter the bank or its savings
counters.

Persistence reuses the hardened lock/read/atomic-replace primitives from the
attention-output crystal bank.  The coordinate document itself is compact,
canonical JSON with independent identity, key, payload, cell, body, and
envelope hashes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import threading
from typing import Any, Iterator

import torch

from .attention_output_crystal import (
    AttentionOutputCrystalError,
    _atomic_write as _crystal_atomic_write,
    _exclusive_state_lock as _crystal_exclusive_state_lock,
    _stable_regular_bytes as _crystal_stable_regular_bytes,
    _stable_signature,
    _thread_lock,
    canonical_bf16_tensor_sha256,
)


MLP_PAGE_COORDINATE_IDENTITY_SCHEMA = (
    "immer.qwen3.8-mlp-page-coordinate-identity/v1"
)
MLP_PAGE_COORDINATE_KEY_SCHEMA = "immer.qwen3.8-mlp-page-coordinate-key/v1"
MLP_PAGE_COORDINATE_PAYLOAD_SCHEMA = (
    "immer.qwen3.8-mlp-page-coordinate-payload/v1"
)
MLP_PAGE_COORDINATE_CELL_SCHEMA = "immer.qwen3.8-mlp-page-coordinate-cell/v1"
MLP_PAGE_COORDINATE_STATE_SCHEMA = (
    "immer.qwen3.8-mlp-page-coordinate-bank-state/v1"
)
MLP_PAGE_COORDINATE_ENVELOPE_SCHEMA = (
    "immer.qwen3.8-mlp-page-coordinate-bank-envelope/v1"
)
MLP_PAGE_COORDINATE_EVIDENCE_SCHEMA = (
    "immer.qwen3.8-mlp-page-coordinate-evidence/v1"
)

_MAX_COUNTER = (1 << 63) - 1
_MAX_LAYER_INDEX = 4095
_MAX_ABSOLUTE_POSITION = (1 << 31) - 1
_MAX_PAGE_ID = (1 << 31) - 1
_MAX_SELECTED_WIDTH = 4096
_MAX_CELLS = 1 << 16
_MAX_STATE_BYTES = 128 * 1024 * 1024
_DEFAULT_MAX_STATE_BYTES = 64 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


class MlpPageCoordinateError(RuntimeError):
    """An exact MLP page coordinate or persistent bank is invalid."""


class MlpPageCoordinateIntegrityError(MlpPageCoordinateError):
    """Persistent state is malformed, unstable, conflicting, or tampered."""


class MlpPageCoordinateIdentityError(MlpPageCoordinateError):
    """A key or bank belongs to another immutable runtime identity."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MlpPageCoordinateIntegrityError(
            "MLP page coordinate data is not canonical JSON"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_document(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - _HEX)
    )


def _digest(value: object, field: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _uint(
    value: object,
    *,
    field: str,
    positive: bool = False,
    maximum: int = _MAX_COUNTER,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < int(positive)
        or value > maximum
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a bounded {qualifier} integer")
    return value


def _bounded_add(left: int, right: int) -> int:
    return min(_MAX_COUNTER, left + right)


def _require_k1(row_count: object) -> int:
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        raise TypeError("row_count must be an integer")
    if row_count != 1:
        raise ValueError("MLP page coordinates currently require row_count == 1")
    return row_count


def _page_ids(value: object) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        if value.ndim < 1 or value.shape[-1] < 1:
            raise ValueError("page_ids must contain one non-empty K1 row")
        row_count = value.numel() // value.shape[-1]
        if row_count != 1:
            raise ValueError("page_ids tensor currently requires row_count == 1")
        value = value.detach().to(device="cpu").reshape(-1).tolist()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise TypeError("page_ids must be one flat sequence")
    result = tuple(value)
    if not 1 <= len(result) <= _MAX_SELECTED_WIDTH:
        raise ValueError("page_ids width is outside the supported range")
    if any(
        isinstance(page_id, (Sequence, torch.Tensor))
        and not isinstance(page_id, (str, bytes, bytearray))
        for page_id in result
    ):
        raise ValueError("page_ids must be one flat K1 row, not ragged")
    for index, page_id in enumerate(result):
        _uint(page_id, field=f"page_ids[{index}]", maximum=_MAX_PAGE_ID)
    if len(set(result)) != len(result):
        raise ValueError("page_ids must be unique in exact execution order")
    return result


def _real(value: object, *, field: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{field} must contain one K1 scalar")
        value = value.detach().to(device="cpu").item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return result


def _scores(value: object) -> tuple[float, ...]:
    if isinstance(value, torch.Tensor):
        if value.ndim < 1 or value.shape[-1] < 1:
            raise ValueError("ranked_page_scores must contain one K1 row")
        row_count = value.numel() // value.shape[-1]
        if row_count != 1:
            raise ValueError(
                "ranked_page_scores tensor currently requires row_count == 1"
            )
        value = value.detach().to(device="cpu").reshape(-1).tolist()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise TypeError("ranked_page_scores must be one flat sequence")
    result = tuple(
        _real(item, field=f"ranked_page_scores[{index}]")
        for index, item in enumerate(value)
    )
    if not result:
        raise ValueError("ranked_page_scores must not be empty")
    tolerance = max(1e-15, result[0] * 1e-15)
    if any(
        result[index] + tolerance < result[index + 1]
        for index in range(len(result) - 1)
    ):
        raise ValueError("ranked_page_scores must be non-increasing")
    return result


def _float_hex(value: float) -> str:
    return value.hex()


def _float_from_hex(value: object, *, field: str) -> float:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a hexadecimal float string")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not a hexadecimal float") from exc
    result = _real(result, field=field)
    if result.hex() != value:
        raise ValueError(f"{field} is not canonical hexadecimal float data")
    return result


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MlpPageCoordinateIntegrityError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _decode_canonical_json(value: bytes) -> Any:
    try:
        document = json.loads(value, object_pairs_hook=_json_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MlpPageCoordinateIntegrityError(
            "MLP page coordinate state is not valid JSON"
        ) from exc
    if _canonical_json(document) != value:
        raise MlpPageCoordinateIntegrityError(
            "MLP page coordinate state is not canonical JSON"
        )
    return document


def _wrap_crystal_error(exc: AttentionOutputCrystalError) -> MlpPageCoordinateIntegrityError:
    return MlpPageCoordinateIntegrityError(
        f"MLP page coordinate filesystem transaction failed: {exc}"
    )


def _stable_regular_bytes(
    path: Path,
    *,
    max_bytes: int,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    try:
        return _crystal_stable_regular_bytes(path, max_bytes=max_bytes)
    except FileNotFoundError:
        raise
    except AttentionOutputCrystalError as exc:
        raise _wrap_crystal_error(exc) from exc


@contextmanager
def _exclusive_state_lock(path: Path) -> Iterator[None]:
    try:
        with _crystal_exclusive_state_lock(path):
            yield
    except AttentionOutputCrystalError as exc:
        raise _wrap_crystal_error(exc) from exc


def _atomic_write(path: Path, data: bytes) -> tuple[int, int, int, int, int]:
    try:
        return _crystal_atomic_write(path, data)
    except AttentionOutputCrystalError as exc:
        raise _wrap_crystal_error(exc) from exc


@dataclass(frozen=True, slots=True)
class MlpPageCoordinateIdentity:
    """Static identities that can change exact MLP route coordinates."""

    runtime_math_sha256: str
    q4_identity_sha256: str
    page_router_identity_sha256: str

    def __post_init__(self) -> None:
        _digest(self.runtime_math_sha256, "runtime_math_sha256")
        _digest(self.q4_identity_sha256, "q4_identity_sha256")
        _digest(
            self.page_router_identity_sha256,
            "page_router_identity_sha256",
        )

    @property
    def static_identity_sha256(self) -> str:
        return _sha256_document(
            {
                "page_router_identity_sha256": self.page_router_identity_sha256,
                "q4_identity_sha256": self.q4_identity_sha256,
                "schema": "immer.qwen3.8-mlp-page-coordinate-static/v1",
            }
        )

    @property
    def identity_sha256(self) -> str:
        return _sha256_document(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "page_router_identity_sha256": self.page_router_identity_sha256,
            "q4_identity_sha256": self.q4_identity_sha256,
            "runtime_math_sha256": self.runtime_math_sha256,
            "schema": MLP_PAGE_COORDINATE_IDENTITY_SCHEMA,
            "static_identity_sha256": self.static_identity_sha256,
        }

    @classmethod
    def from_record(cls, value: object) -> "MlpPageCoordinateIdentity":
        if not isinstance(value, Mapping) or set(value) != {
            "page_router_identity_sha256",
            "q4_identity_sha256",
            "runtime_math_sha256",
            "schema",
            "static_identity_sha256",
        }:
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate identity fields are invalid"
            )
        if value["schema"] != MLP_PAGE_COORDINATE_IDENTITY_SCHEMA:
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate identity schema is invalid"
            )
        try:
            identity = cls(
                runtime_math_sha256=value["runtime_math_sha256"],
                q4_identity_sha256=value["q4_identity_sha256"],
                page_router_identity_sha256=value[
                    "page_router_identity_sha256"
                ],
            )
            if value["static_identity_sha256"] != identity.static_identity_sha256:
                raise ValueError("static identity SHA-256 mismatch")
            return identity
        except (TypeError, ValueError) as exc:
            raise MlpPageCoordinateIntegrityError(
                f"MLP page coordinate identity values are invalid: {exc}"
            ) from exc


@dataclass(frozen=True, slots=True)
class MlpPageCoordinateKey:
    identity_sha256: str
    runtime_math_sha256: str
    static_identity_sha256: str
    layer_index: int
    absolute_position: int
    row_count: int
    mlp_input_sha256: str
    key_sha256: str

    def __post_init__(self) -> None:
        _digest(self.identity_sha256, "key identity_sha256")
        _digest(self.runtime_math_sha256, "key runtime_math_sha256")
        _digest(self.static_identity_sha256, "key static_identity_sha256")
        _uint(self.layer_index, field="layer_index", maximum=_MAX_LAYER_INDEX)
        _uint(
            self.absolute_position,
            field="absolute_position",
            maximum=_MAX_ABSOLUTE_POSITION,
        )
        _require_k1(self.row_count)
        _digest(self.mlp_input_sha256, "mlp_input_sha256")
        _digest(self.key_sha256, "key_sha256")
        if self.key_sha256 != _sha256_document(self._address_record()):
            raise ValueError("MLP page coordinate key SHA-256 mismatch")

    def _address_record(self) -> dict[str, object]:
        return {
            "absolute_position": self.absolute_position,
            "identity_sha256": self.identity_sha256,
            "layer_index": self.layer_index,
            "mlp_input_sha256": self.mlp_input_sha256,
            "row_count": self.row_count,
            "runtime_math_sha256": self.runtime_math_sha256,
            "schema": MLP_PAGE_COORDINATE_KEY_SCHEMA,
            "static_identity_sha256": self.static_identity_sha256,
        }

    def to_record(self) -> dict[str, object]:
        return self._address_record() | {"key_sha256": self.key_sha256}

    @classmethod
    def create(
        cls,
        *,
        identity: MlpPageCoordinateIdentity,
        layer_index: int,
        absolute_position: int,
        row_count: int,
        mlp_input_sha256: str,
    ) -> "MlpPageCoordinateKey":
        address = {
            "absolute_position": absolute_position,
            "identity_sha256": identity.identity_sha256,
            "layer_index": layer_index,
            "mlp_input_sha256": mlp_input_sha256,
            "row_count": row_count,
            "runtime_math_sha256": identity.runtime_math_sha256,
            "schema": MLP_PAGE_COORDINATE_KEY_SCHEMA,
            "static_identity_sha256": identity.static_identity_sha256,
        }
        return cls(
            identity_sha256=identity.identity_sha256,
            runtime_math_sha256=identity.runtime_math_sha256,
            static_identity_sha256=identity.static_identity_sha256,
            layer_index=layer_index,
            absolute_position=absolute_position,
            row_count=row_count,
            mlp_input_sha256=mlp_input_sha256,
            key_sha256=_sha256_document(address),
        )

    @classmethod
    def from_record(cls, value: object) -> "MlpPageCoordinateKey":
        if not isinstance(value, Mapping) or set(value) != {
            "absolute_position",
            "identity_sha256",
            "key_sha256",
            "layer_index",
            "mlp_input_sha256",
            "row_count",
            "runtime_math_sha256",
            "schema",
            "static_identity_sha256",
        }:
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate key fields are invalid"
            )
        if value["schema"] != MLP_PAGE_COORDINATE_KEY_SCHEMA:
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate key schema is invalid"
            )
        try:
            return cls(
                identity_sha256=value["identity_sha256"],
                runtime_math_sha256=value["runtime_math_sha256"],
                static_identity_sha256=value["static_identity_sha256"],
                layer_index=value["layer_index"],
                absolute_position=value["absolute_position"],
                row_count=value["row_count"],
                mlp_input_sha256=value["mlp_input_sha256"],
                key_sha256=value["key_sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise MlpPageCoordinateIntegrityError(
                f"MLP page coordinate key values are invalid: {exc}"
            ) from exc


def _payload_record(
    *,
    page_ids: tuple[int, ...],
    selected_width: int,
    ranked_page_scores: tuple[float, ...] | None,
    total_energy: float | None,
    logical_page_weight_bytes_saved: int,
) -> dict[str, object]:
    return {
        "logical_page_weight_bytes_saved": logical_page_weight_bytes_saved,
        "page_ids": list(page_ids),
        "ranked_page_scores_hex": (
            None
            if ranked_page_scores is None
            else [_float_hex(value) for value in ranked_page_scores]
        ),
        "schema": MLP_PAGE_COORDINATE_PAYLOAD_SCHEMA,
        "selected_width": selected_width,
        "total_energy_hex": (
            None if total_energy is None else _float_hex(total_energy)
        ),
    }


def _validated_payload(
    *,
    page_ids: object,
    selected_width: object,
    ranked_page_scores: object | None,
    total_energy: object | None,
    logical_page_weight_bytes_saved: object,
) -> tuple[tuple[int, ...], int, tuple[float, ...] | None, float | None, int]:
    pages = _page_ids(page_ids)
    width = _uint(
        selected_width,
        field="selected_width",
        positive=True,
        maximum=_MAX_SELECTED_WIDTH,
    )
    if width > len(pages):
        raise ValueError("selected_width cannot exceed the ranked page_ids width")
    logical = _uint(
        logical_page_weight_bytes_saved,
        field="logical_page_weight_bytes_saved",
    )
    if (ranked_page_scores is None) != (total_energy is None):
        raise ValueError(
            "ranked_page_scores and total_energy must be present together"
        )
    if ranked_page_scores is None:
        return pages, width, None, None, logical
    scores = _scores(ranked_page_scores)
    if len(scores) != len(pages):
        raise ValueError("ranked_page_scores width must match page_ids")
    energy = _real(total_energy, field="total_energy")
    tolerance = max(1e-12, energy * 1e-12)
    if sum(scores) > energy + tolerance:
        raise ValueError("ranked_page_scores exceed total_energy")
    return pages, width, scores, energy, logical


@dataclass(frozen=True, slots=True)
class StagedMlpPageCoordinate:
    """Immutable cold-teacher coordinate awaiting accepted publication."""

    key: MlpPageCoordinateKey
    page_ids: tuple[int, ...]
    selected_width: int
    ranked_page_scores: tuple[float, ...] | None
    total_energy: float | None
    logical_page_weight_bytes_saved: int
    payload_sha256: str
    cell_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, MlpPageCoordinateKey):
            raise TypeError("staged key must be an MlpPageCoordinateKey")
        pages, width, scores, energy, logical = _validated_payload(
            page_ids=self.page_ids,
            selected_width=self.selected_width,
            ranked_page_scores=self.ranked_page_scores,
            total_energy=self.total_energy,
            logical_page_weight_bytes_saved=(
                self.logical_page_weight_bytes_saved
            ),
        )
        object.__setattr__(self, "page_ids", pages)
        object.__setattr__(self, "selected_width", width)
        object.__setattr__(self, "ranked_page_scores", scores)
        object.__setattr__(self, "total_energy", energy)
        object.__setattr__(self, "logical_page_weight_bytes_saved", logical)
        _digest(self.payload_sha256, "payload_sha256")
        _digest(self.cell_sha256, "cell_sha256")
        if self.payload_sha256 != _sha256_document(self.payload_record()):
            raise ValueError("MLP page coordinate payload SHA-256 mismatch")
        if self.cell_sha256 != self.expected_cell_sha256():
            raise ValueError("MLP page coordinate cell SHA-256 mismatch")

    @property
    def absolute_position(self) -> int:
        return self.key.absolute_position

    @property
    def layer_index(self) -> int:
        return self.key.layer_index

    @property
    def selected_page_ids(self) -> tuple[int, ...]:
        return self.page_ids[: self.selected_width]

    def payload_record(self) -> dict[str, object]:
        return _payload_record(
            page_ids=self.page_ids,
            selected_width=self.selected_width,
            ranked_page_scores=self.ranked_page_scores,
            total_energy=self.total_energy,
            logical_page_weight_bytes_saved=(
                self.logical_page_weight_bytes_saved
            ),
        )

    def expected_cell_sha256(self) -> str:
        return _sha256_document(
            {
                "key_sha256": self.key.key_sha256,
                "payload_sha256": self.payload_sha256,
                "schema": MLP_PAGE_COORDINATE_CELL_SCHEMA,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "absolute_position": self.absolute_position,
            "cell_sha256": self.cell_sha256,
            "key_sha256": self.key.key_sha256,
            "layer_index": self.layer_index,
            "logical_page_weight_bytes_saved": (
                self.logical_page_weight_bytes_saved
            ),
            "page_ids": list(self.page_ids),
            "payload_sha256": self.payload_sha256,
            "selected_width": self.selected_width,
            "selected_page_ids": list(self.selected_page_ids),
            "teacher_evidence": self.ranked_page_scores is not None,
        }


MlpPageCoordinateStage = StagedMlpPageCoordinate


@dataclass(frozen=True, slots=True)
class _CoordinateCell:
    cell_sha256: str
    key: MlpPageCoordinateKey
    page_ids: tuple[int, ...]
    selected_width: int
    ranked_page_scores: tuple[float, ...] | None
    total_energy: float | None
    logical_page_weight_bytes_saved: int
    payload_sha256: str
    support: int
    hit_count: int
    created_clock: int
    last_used_clock: int

    def __post_init__(self) -> None:
        staged = StagedMlpPageCoordinate(
            key=self.key,
            page_ids=self.page_ids,
            selected_width=self.selected_width,
            ranked_page_scores=self.ranked_page_scores,
            total_energy=self.total_energy,
            logical_page_weight_bytes_saved=(
                self.logical_page_weight_bytes_saved
            ),
            payload_sha256=self.payload_sha256,
            cell_sha256=self.cell_sha256,
        )
        object.__setattr__(self, "page_ids", staged.page_ids)
        object.__setattr__(self, "ranked_page_scores", staged.ranked_page_scores)
        object.__setattr__(self, "total_energy", staged.total_energy)
        for field in ("support", "created_clock", "last_used_clock"):
            _uint(getattr(self, field), field=field, positive=True)
        _uint(self.hit_count, field="hit_count")
        if self.last_used_clock < self.created_clock:
            raise ValueError("coordinate last-used clock precedes creation")

    @property
    def value_units(self) -> int:
        observations = min(_MAX_COUNTER, self.support + self.hit_count)
        logical = min(
            _MAX_COUNTER,
            self.logical_page_weight_bytes_saved + 1,
        )
        return min(_MAX_COUNTER, observations * logical)

    def to_record(self) -> dict[str, object]:
        return {
            "cell_sha256": self.cell_sha256,
            "created_clock": self.created_clock,
            "hit_count": self.hit_count,
            "key": self.key.to_record(),
            "last_used_clock": self.last_used_clock,
            "payload": _payload_record(
                page_ids=self.page_ids,
                selected_width=self.selected_width,
                ranked_page_scores=self.ranked_page_scores,
                total_energy=self.total_energy,
                logical_page_weight_bytes_saved=(
                    self.logical_page_weight_bytes_saved
                ),
            )
            | {"payload_sha256": self.payload_sha256},
            "schema": MLP_PAGE_COORDINATE_CELL_SCHEMA,
            "support": self.support,
        }

    @classmethod
    def from_record(cls, value: object) -> "_CoordinateCell":
        if not isinstance(value, Mapping) or set(value) != {
            "cell_sha256",
            "created_clock",
            "hit_count",
            "key",
            "last_used_clock",
            "payload",
            "schema",
            "support",
        }:
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate cell fields are invalid"
            )
        if value["schema"] != MLP_PAGE_COORDINATE_CELL_SCHEMA:
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate cell schema is invalid"
            )
        payload = value["payload"]
        if not isinstance(payload, Mapping) or set(payload) != {
            "logical_page_weight_bytes_saved",
            "page_ids",
            "payload_sha256",
            "ranked_page_scores_hex",
            "schema",
            "selected_width",
            "total_energy_hex",
        }:
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate payload fields are invalid"
            )
        if payload["schema"] != MLP_PAGE_COORDINATE_PAYLOAD_SCHEMA:
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate payload schema is invalid"
            )
        scores_hex = payload["ranked_page_scores_hex"]
        if scores_hex is not None and not isinstance(scores_hex, list):
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate score table is invalid"
            )
        try:
            scores = (
                None
                if scores_hex is None
                else tuple(
                    _float_from_hex(item, field="ranked_page_score")
                    for item in scores_hex
                )
            )
            energy_hex = payload["total_energy_hex"]
            energy = (
                None
                if energy_hex is None
                else _float_from_hex(energy_hex, field="total_energy")
            )
            return cls(
                cell_sha256=value["cell_sha256"],
                key=MlpPageCoordinateKey.from_record(value["key"]),
                page_ids=tuple(payload["page_ids"]),
                selected_width=payload["selected_width"],
                ranked_page_scores=scores,
                total_energy=energy,
                logical_page_weight_bytes_saved=payload[
                    "logical_page_weight_bytes_saved"
                ],
                payload_sha256=payload["payload_sha256"],
                support=value["support"],
                hit_count=value["hit_count"],
                created_clock=value["created_clock"],
                last_used_clock=value["last_used_clock"],
            )
        except (TypeError, ValueError) as exc:
            raise MlpPageCoordinateIntegrityError(
                f"MLP page coordinate cell values are invalid: {exc}"
            ) from exc


def _cell_for_stage(
    staged: StagedMlpPageCoordinate,
    *,
    clock: int,
) -> _CoordinateCell:
    return _CoordinateCell(
        cell_sha256=staged.cell_sha256,
        key=staged.key,
        page_ids=staged.page_ids,
        selected_width=staged.selected_width,
        ranked_page_scores=staged.ranked_page_scores,
        total_energy=staged.total_energy,
        logical_page_weight_bytes_saved=staged.logical_page_weight_bytes_saved,
        payload_sha256=staged.payload_sha256,
        support=1,
        hit_count=0,
        created_clock=clock,
        last_used_clock=clock,
    )


@dataclass(frozen=True, slots=True)
class _CoordinateState:
    identity: MlpPageCoordinateIdentity
    max_cells: int
    max_state_bytes: int
    clock: int = 0
    transactions: int = 0
    staged_captures: int = 0
    accepted_captures: int = 0
    rejected_captures: int = 0
    hit_count: int = 0
    physical_pages_saved: int = 0
    logical_page_weight_bytes_saved: int = 0
    evictions: int = 0
    cells: tuple[_CoordinateCell, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.identity, MlpPageCoordinateIdentity):
            raise ValueError("coordinate state identity is invalid")
        _uint(self.max_cells, field="max_cells", positive=True, maximum=_MAX_CELLS)
        _uint(
            self.max_state_bytes,
            field="max_state_bytes",
            positive=True,
            maximum=_MAX_STATE_BYTES,
        )
        for field in (
            "clock",
            "transactions",
            "staged_captures",
            "accepted_captures",
            "rejected_captures",
            "hit_count",
            "physical_pages_saved",
            "logical_page_weight_bytes_saved",
            "evictions",
        ):
            _uint(getattr(self, field), field=field)
        cells = tuple(self.cells)
        if len(cells) > self.max_cells:
            raise ValueError("coordinate state exceeds max_cells")
        if tuple(sorted(cells, key=lambda cell: cell.key.key_sha256)) != cells:
            raise ValueError("coordinate cells are not canonically ordered")
        if len({cell.key.key_sha256 for cell in cells}) != len(cells):
            raise ValueError("coordinate state contains duplicate exact keys")
        if len({cell.cell_sha256 for cell in cells}) != len(cells):
            raise ValueError("coordinate state contains duplicate cells")
        if any(
            cell.key.identity_sha256 != self.identity.identity_sha256
            or cell.key.runtime_math_sha256
            != self.identity.runtime_math_sha256
            or cell.key.static_identity_sha256
            != self.identity.static_identity_sha256
            or cell.last_used_clock > self.clock
            for cell in cells
        ):
            raise ValueError("coordinate cell identity or clock is invalid")
        if self.accepted_captures > self.staged_captures:
            raise ValueError("accepted capture count exceeds staged captures")
        if self.rejected_captures > self.staged_captures:
            raise ValueError("rejected capture count exceeds staged captures")
        if min(_MAX_COUNTER, sum(cell.hit_count for cell in cells)) > self.hit_count:
            raise ValueError("cell hits exceed the persistent hit counter")
        object.__setattr__(self, "cells", cells)

    def to_record(self) -> dict[str, object]:
        return {
            "accepted_captures": self.accepted_captures,
            "cells": [cell.to_record() for cell in self.cells],
            "clock": self.clock,
            "evictions": self.evictions,
            "hit_count": self.hit_count,
            "identity": self.identity.to_record(),
            "identity_sha256": self.identity.identity_sha256,
            "logical_page_weight_bytes_saved": (
                self.logical_page_weight_bytes_saved
            ),
            "max_cells": self.max_cells,
            "max_state_bytes": self.max_state_bytes,
            "physical_pages_saved": self.physical_pages_saved,
            "rejected_captures": self.rejected_captures,
            "schema": MLP_PAGE_COORDINATE_STATE_SCHEMA,
            "staged_captures": self.staged_captures,
            "transactions": self.transactions,
        }

    @classmethod
    def from_record(cls, value: object) -> "_CoordinateState":
        if not isinstance(value, Mapping) or set(value) != {
            "accepted_captures",
            "cells",
            "clock",
            "evictions",
            "hit_count",
            "identity",
            "identity_sha256",
            "logical_page_weight_bytes_saved",
            "max_cells",
            "max_state_bytes",
            "physical_pages_saved",
            "rejected_captures",
            "schema",
            "staged_captures",
            "transactions",
        }:
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate state fields are invalid"
            )
        if value["schema"] != MLP_PAGE_COORDINATE_STATE_SCHEMA:
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate state schema is invalid"
            )
        cells = value["cells"]
        if not isinstance(cells, list):
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate cell table is invalid"
            )
        try:
            identity = MlpPageCoordinateIdentity.from_record(value["identity"])
            if value["identity_sha256"] != identity.identity_sha256:
                raise ValueError("coordinate identity SHA-256 mismatch")
            return cls(
                identity=identity,
                max_cells=value["max_cells"],
                max_state_bytes=value["max_state_bytes"],
                clock=value["clock"],
                transactions=value["transactions"],
                staged_captures=value["staged_captures"],
                accepted_captures=value["accepted_captures"],
                rejected_captures=value["rejected_captures"],
                hit_count=value["hit_count"],
                physical_pages_saved=value["physical_pages_saved"],
                logical_page_weight_bytes_saved=value[
                    "logical_page_weight_bytes_saved"
                ],
                evictions=value["evictions"],
                cells=tuple(_CoordinateCell.from_record(cell) for cell in cells),
            )
        except (TypeError, ValueError) as exc:
            raise MlpPageCoordinateIntegrityError(
                f"MLP page coordinate state values are invalid: {exc}"
            ) from exc

    def to_bytes(self) -> bytes:
        body = self.to_record()
        envelope = {
            "body": body,
            "body_sha256": _sha256_document(body),
            "schema": MLP_PAGE_COORDINATE_ENVELOPE_SCHEMA,
        }
        encoded = _canonical_json(envelope)
        if len(encoded) > _MAX_STATE_BYTES:
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate state exceeds its absolute byte limit"
            )
        return encoded

    @classmethod
    def from_bytes(cls, value: bytes) -> "_CoordinateState":
        document = _decode_canonical_json(value)
        if not isinstance(document, Mapping) or set(document) != {
            "body",
            "body_sha256",
            "schema",
        }:
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate envelope fields are invalid"
            )
        if document["schema"] != MLP_PAGE_COORDINATE_ENVELOPE_SCHEMA:
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate envelope schema is invalid"
            )
        if not _is_sha256(document["body_sha256"]):
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate body digest is invalid"
            )
        if document["body_sha256"] != _sha256_document(document["body"]):
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate body SHA-256 mismatch"
            )
        return cls.from_record(document["body"])


@dataclass(frozen=True, slots=True)
class MlpPageCoordinateHit:
    cell_sha256: str
    key: MlpPageCoordinateKey
    page_ids: tuple[int, ...]
    selected_width: int
    ranked_page_scores: tuple[float, ...] | None
    total_energy: float | None
    logical_page_weight_bytes_saved: int
    payload_sha256: str
    support: int
    cell_hit_count: int

    @property
    def absolute_position(self) -> int:
        return self.key.absolute_position

    @property
    def layer_index(self) -> int:
        return self.key.layer_index

    @property
    def teacher_evidence(self) -> bool:
        return self.ranked_page_scores is not None

    @property
    def selected_page_ids(self) -> tuple[int, ...]:
        """Exact execution prefix chosen from the stored ranked page order."""

        return self.page_ids[: self.selected_width]

    def to_dict(self) -> dict[str, object]:
        return {
            "absolute_position": self.absolute_position,
            "cell_hit_count": self.cell_hit_count,
            "cell_sha256": self.cell_sha256,
            "key_sha256": self.key.key_sha256,
            "layer_index": self.layer_index,
            "logical_page_weight_bytes_saved": (
                self.logical_page_weight_bytes_saved
            ),
            "page_ids": list(self.page_ids),
            "payload_sha256": self.payload_sha256,
            "selected_width": self.selected_width,
            "selected_page_ids": list(self.selected_page_ids),
            "support": self.support,
            "teacher_evidence": self.teacher_evidence,
        }


@dataclass(frozen=True, slots=True)
class MlpPageCoordinateMetrics:
    identity_sha256: str
    state_sha256: str
    max_cells: int
    max_state_bytes: int
    persisted_bytes: int
    cell_count: int
    clock: int
    transactions: int
    staged_captures: int
    accepted_captures: int
    rejected_captures: int
    hit_count: int
    physical_pages_saved: int
    logical_page_weight_bytes_saved: int
    evictions: int
    support: int

    @property
    def hits(self) -> int:
        return self.hit_count

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted_captures": self.accepted_captures,
            "cell_count": self.cell_count,
            "clock": self.clock,
            "evictions": self.evictions,
            "hit_count": self.hit_count,
            "hits": self.hits,
            "identity_sha256": self.identity_sha256,
            "logical_page_weight_bytes_saved": (
                self.logical_page_weight_bytes_saved
            ),
            "max_cells": self.max_cells,
            "max_state_bytes": self.max_state_bytes,
            "persisted_bytes": self.persisted_bytes,
            "physical_pages_saved": self.physical_pages_saved,
            "rejected_captures": self.rejected_captures,
            "staged_captures": self.staged_captures,
            "state_sha256": self.state_sha256,
            "support": self.support,
            "transactions": self.transactions,
        }


@dataclass(frozen=True, slots=True)
class _StagedHit:
    position: int
    key_sha256: str
    cell_sha256: str
    physical_pages_saved: int
    logical_page_weight_bytes_saved: int

    def __post_init__(self) -> None:
        _uint(
            self.position,
            field="hit position",
            maximum=_MAX_ABSOLUTE_POSITION,
        )
        _digest(self.key_sha256, "hit key_sha256")
        _digest(self.cell_sha256, "hit cell_sha256")
        _uint(self.physical_pages_saved, field="physical_pages_saved")
        _uint(
            self.logical_page_weight_bytes_saved,
            field="logical_page_weight_bytes_saved",
        )


class MlpPageCoordinateTransaction:
    """Request-local K1 captures and hits awaiting an accepted boundary."""

    def __init__(self, bank: "MlpPageCoordinateBank") -> None:
        self._bank = bank
        self._captures: list[tuple[int, StagedMlpPageCoordinate]] = []
        self._hits: list[_StagedHit] = []
        self._closed = False
        self._lock = threading.RLock()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise MlpPageCoordinateError("MLP page coordinate transaction is closed")

    def stage_capture(
        self,
        position: int,
        coordinate: StagedMlpPageCoordinate,
        *,
        row_count: int = 1,
    ) -> StagedMlpPageCoordinate:
        with self._lock:
            self._require_open()
            _require_k1(row_count)
            position = _uint(
                position,
                field="capture position",
                maximum=_MAX_ABSOLUTE_POSITION,
            )
            self._bank._validate_staged(coordinate)
            if position != coordinate.key.absolute_position:
                raise ValueError("capture position disagrees with its exact key")
            if any(
                previous.key.key_sha256 == coordinate.key.key_sha256
                for _position, previous in self._captures
            ):
                raise ValueError("one exact MLP page capture was staged twice")
            self._captures.append((position, coordinate))
            return coordinate

    def stage_hit(
        self,
        position: int,
        hit_or_key: MlpPageCoordinateHit | MlpPageCoordinateKey,
        *,
        physical_pages_saved: int,
        logical_page_weight_bytes_saved: int,
        row_count: int = 1,
    ) -> MlpPageCoordinateHit:
        with self._lock:
            self._require_open()
            _require_k1(row_count)
            position = _uint(
                position,
                field="hit position",
                maximum=_MAX_ABSOLUTE_POSITION,
            )
            if isinstance(hit_or_key, MlpPageCoordinateKey):
                hit = self._bank.peek(hit_or_key)
                if hit is None:
                    raise MlpPageCoordinateIntegrityError(
                        "cannot stage an unknown MLP page coordinate hit"
                    )
            elif isinstance(hit_or_key, MlpPageCoordinateHit):
                hit = hit_or_key
                self._bank._validate_hit(hit)
            else:
                raise TypeError(
                    "hit_or_key must be an MlpPageCoordinateHit or key"
                )
            if position != hit.key.absolute_position:
                raise ValueError("hit position disagrees with its exact key")
            if any(
                previous.position == position
                and previous.key_sha256 == hit.key.key_sha256
                for previous in self._hits
            ):
                raise ValueError("one exact MLP page hit was staged twice")
            self._hits.append(
                _StagedHit(
                    position=position,
                    key_sha256=hit.key.key_sha256,
                    cell_sha256=hit.cell_sha256,
                    physical_pages_saved=physical_pages_saved,
                    logical_page_weight_bytes_saved=(
                        logical_page_weight_bytes_saved
                    ),
                )
            )
            return hit

    def lookup(
        self,
        key: MlpPageCoordinateKey,
        *,
        physical_pages_saved: int,
        logical_page_weight_bytes_saved: int,
        row_count: int = 1,
    ) -> MlpPageCoordinateHit | None:
        with self._lock:
            self._require_open()
            _require_k1(row_count)
            hit = self._bank.peek(key)
            if hit is None:
                return None
            return self.stage_hit(
                key.absolute_position,
                hit,
                physical_pages_saved=physical_pages_saved,
                logical_page_weight_bytes_saved=(
                    logical_page_weight_bytes_saved
                ),
                row_count=1,
            )

    def commit(
        self,
        *,
        accepted_end_position: int,
    ) -> MlpPageCoordinateMetrics:
        with self._lock:
            self._require_open()
            boundary = _uint(
                accepted_end_position,
                field="accepted_end_position",
                maximum=_MAX_ABSOLUTE_POSITION + 1,
            )
            captures = tuple(
                coordinate
                for position, coordinate in self._captures
                if position < boundary
            )
            hits = tuple(hit for hit in self._hits if hit.position < boundary)
            metrics = self._bank._commit(
                captures=captures,
                staged_capture_count=len(self._captures),
                rejected_capture_count=len(self._captures) - len(captures),
                hits=hits,
            )
            self._closed = True
            return metrics

    def rollback(self) -> None:
        with self._lock:
            self._require_open()
            self._captures.clear()
            self._hits.clear()
            self._closed = True

    def __enter__(self) -> "MlpPageCoordinateTransaction":
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        with self._lock:
            if not self._closed:
                self.rollback()


class MlpPageCoordinateBank:
    """Bounded exact K1 MLP page route bank."""

    def __init__(
        self,
        state_path: str | os.PathLike[str],
        identity: MlpPageCoordinateIdentity,
        *,
        max_cells: int = 4096,
        max_state_bytes: int = _DEFAULT_MAX_STATE_BYTES,
    ) -> None:
        if not isinstance(identity, MlpPageCoordinateIdentity):
            raise TypeError("identity must be an MlpPageCoordinateIdentity")
        self.state_path = Path(state_path)
        if not self.state_path.name:
            raise ValueError("state_path must name a file")
        _uint(max_cells, field="max_cells", positive=True, maximum=_MAX_CELLS)
        _uint(
            max_state_bytes,
            field="max_state_bytes",
            positive=True,
            maximum=_MAX_STATE_BYTES,
        )
        self.identity = identity
        self.max_cells = max_cells
        self.max_state_bytes = max_state_bytes
        self._lock = _thread_lock(self.state_path)
        self._file_signature: tuple[int, int, int, int, int] | None = None
        self._state = _CoordinateState(
            identity=identity,
            max_cells=max_cells,
            max_state_bytes=max_state_bytes,
        )
        self._index: dict[str, _CoordinateCell] = {}
        with self._lock:
            self._reload(required=False)

    def _validate_state(self, state: _CoordinateState) -> None:
        if state.identity.identity_sha256 != self.identity.identity_sha256:
            raise MlpPageCoordinateIdentityError(
                "MLP page coordinate bank belongs to another runtime identity"
            )
        if state.max_cells != self.max_cells:
            raise MlpPageCoordinateIdentityError(
                "MLP page coordinate max_cells differs from persistent identity"
            )
        if state.max_state_bytes != self.max_state_bytes:
            raise MlpPageCoordinateIdentityError(
                "MLP page coordinate max_state_bytes differs from persistent identity"
            )

    def _reload(self, *, required: bool) -> None:
        try:
            raw, signature = _stable_regular_bytes(
                self.state_path,
                max_bytes=self.max_state_bytes,
            )
        except FileNotFoundError:
            if required:
                raise MlpPageCoordinateIntegrityError(
                    "MLP page coordinate state disappeared"
                )
            self._state = _CoordinateState(
                identity=self.identity,
                max_cells=self.max_cells,
                max_state_bytes=self.max_state_bytes,
            )
            self._file_signature = None
            self._rebuild_index()
            return
        state = _CoordinateState.from_bytes(raw)
        self._validate_state(state)
        self._state = state
        self._file_signature = signature
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index = {cell.key.key_sha256: cell for cell in self._state.cells}

    def _refresh_if_changed(self) -> None:
        try:
            linked = os.lstat(self.state_path)
        except FileNotFoundError:
            if self._file_signature is not None:
                raise MlpPageCoordinateIntegrityError(
                    "MLP page coordinate state disappeared"
                )
            return
        if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
            raise MlpPageCoordinateIntegrityError(
                "MLP page coordinate state must be a regular file"
            )
        if _stable_signature(linked) == self._file_signature:
            return
        last_error: MlpPageCoordinateIntegrityError | None = None
        for _ in range(3):
            try:
                self._reload(required=True)
                return
            except MlpPageCoordinateIntegrityError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def _validate_key(self, key: MlpPageCoordinateKey) -> None:
        if not isinstance(key, MlpPageCoordinateKey):
            raise TypeError("key must be an MlpPageCoordinateKey")
        if (
            key.identity_sha256 != self.identity.identity_sha256
            or key.runtime_math_sha256 != self.identity.runtime_math_sha256
            or key.static_identity_sha256
            != self.identity.static_identity_sha256
        ):
            raise MlpPageCoordinateIdentityError(
                "MLP page coordinate key belongs to another runtime identity"
            )

    def _validate_staged(self, staged: StagedMlpPageCoordinate) -> None:
        if not isinstance(staged, StagedMlpPageCoordinate):
            raise TypeError("coordinate must be a StagedMlpPageCoordinate")
        self._validate_key(staged.key)

    def _validate_hit(self, hit: MlpPageCoordinateHit) -> None:
        if not isinstance(hit, MlpPageCoordinateHit):
            raise TypeError("hit must be an MlpPageCoordinateHit")
        self._validate_key(hit.key)
        with self._lock:
            self._refresh_if_changed()
            cell = self._index.get(hit.key.key_sha256)
            if cell is None or (
                cell.cell_sha256 != hit.cell_sha256
                or cell.payload_sha256 != hit.payload_sha256
                or cell.page_ids != hit.page_ids
                or cell.selected_width != hit.selected_width
            ):
                raise MlpPageCoordinateIntegrityError(
                    "cannot stage a stale or forged MLP page coordinate hit"
                )

    def make_key(
        self,
        layer_index: int,
        absolute_position: int,
        mlp_input: torch.Tensor,
        *,
        row_count: int = 1,
    ) -> MlpPageCoordinateKey:
        """Seal one exact finite BF16 ``[1, 1, hidden]`` MLP input."""

        _require_k1(row_count)
        _uint(layer_index, field="layer_index", maximum=_MAX_LAYER_INDEX)
        _uint(
            absolute_position,
            field="absolute_position",
            maximum=_MAX_ABSOLUTE_POSITION,
        )
        if not isinstance(mlp_input, torch.Tensor):
            raise TypeError("mlp_input must be a torch.Tensor")
        if mlp_input.ndim != 3 or tuple(mlp_input.shape[:2]) != (1, 1):
            raise ValueError("mlp_input must have exact K1 shape [1, 1, hidden]")
        return MlpPageCoordinateKey.create(
            identity=self.identity,
            layer_index=layer_index,
            absolute_position=absolute_position,
            row_count=1,
            mlp_input_sha256=canonical_bf16_tensor_sha256(mlp_input),
        )

    def stage(
        self,
        key: MlpPageCoordinateKey,
        page_ids: Sequence[int] | torch.Tensor,
        *,
        selected_width: int,
        ranked_page_scores: Sequence[float] | torch.Tensor | None = None,
        total_energy: float | torch.Tensor | None = None,
        logical_page_weight_bytes_saved: int,
        row_count: int = 1,
    ) -> StagedMlpPageCoordinate:
        """Build an immutable exact route and optional cold-teacher evidence."""

        self._validate_key(key)
        _require_k1(row_count)
        pages, width, scores, energy, logical = _validated_payload(
            page_ids=page_ids,
            selected_width=selected_width,
            ranked_page_scores=ranked_page_scores,
            total_energy=total_energy,
            logical_page_weight_bytes_saved=(
                logical_page_weight_bytes_saved
            ),
        )
        payload = _payload_record(
            page_ids=pages,
            selected_width=width,
            ranked_page_scores=scores,
            total_energy=energy,
            logical_page_weight_bytes_saved=logical,
        )
        payload_sha256 = _sha256_document(payload)
        cell_sha256 = _sha256_document(
            {
                "key_sha256": key.key_sha256,
                "payload_sha256": payload_sha256,
                "schema": MLP_PAGE_COORDINATE_CELL_SCHEMA,
            }
        )
        return StagedMlpPageCoordinate(
            key=key,
            page_ids=pages,
            selected_width=width,
            ranked_page_scores=scores,
            total_energy=energy,
            logical_page_weight_bytes_saved=logical,
            payload_sha256=payload_sha256,
            cell_sha256=cell_sha256,
        )

    @staticmethod
    def _hit_from_cell(cell: _CoordinateCell) -> MlpPageCoordinateHit:
        return MlpPageCoordinateHit(
            cell_sha256=cell.cell_sha256,
            key=cell.key,
            page_ids=cell.page_ids,
            selected_width=cell.selected_width,
            ranked_page_scores=cell.ranked_page_scores,
            total_energy=cell.total_energy,
            logical_page_weight_bytes_saved=(
                cell.logical_page_weight_bytes_saved
            ),
            payload_sha256=cell.payload_sha256,
            support=cell.support,
            cell_hit_count=cell.hit_count,
        )

    def peek(self, key: MlpPageCoordinateKey) -> MlpPageCoordinateHit | None:
        """Read a coordinate without publishing hit or savings counters."""

        self._validate_key(key)
        with self._lock:
            self._refresh_if_changed()
            cell = self._index.get(key.key_sha256)
            if cell is None:
                return None
            if cell.key != key:
                raise MlpPageCoordinateIntegrityError(
                    "one key address names two MLP page coordinates"
                )
            return self._hit_from_cell(cell)

    def begin_transaction(self) -> MlpPageCoordinateTransaction:
        return MlpPageCoordinateTransaction(self)

    @staticmethod
    def _victim(cells: Mapping[str, _CoordinateCell]) -> _CoordinateCell:
        return min(
            cells.values(),
            key=lambda cell: (
                cell.value_units,
                cell.last_used_clock,
                cell.support,
                cell.key.key_sha256,
            ),
        )

    def _commit(
        self,
        *,
        captures: Sequence[StagedMlpPageCoordinate],
        staged_capture_count: int,
        rejected_capture_count: int,
        hits: Sequence[_StagedHit],
    ) -> MlpPageCoordinateMetrics:
        captures = tuple(captures)
        hits = tuple(hits)
        staged_capture_count = _uint(
            staged_capture_count,
            field="staged_capture_count",
        )
        rejected_capture_count = _uint(
            rejected_capture_count,
            field="rejected_capture_count",
            maximum=staged_capture_count,
        )
        if len(captures) + rejected_capture_count != staged_capture_count:
            raise ValueError("accepted and rejected captures do not cover the stage")
        for capture in captures:
            self._validate_staged(capture)
        if any(not isinstance(hit, _StagedHit) for hit in hits):
            raise TypeError("hits contains an invalid staged hit")
        if not staged_capture_count and not hits:
            return self.metrics()

        with _exclusive_state_lock(self.state_path):
            self._reload(required=self._file_signature is not None)
            state = self._state
            clock = _bounded_add(state.clock, 1)
            cells = {cell.key.key_sha256: cell for cell in state.cells}

            for event in hits:
                cell = cells.get(event.key_sha256)
                if cell is None:
                    # The exact route was already executed.  Concurrent
                    # eviction cannot erase its request-local economics.
                    continue
                if cell.cell_sha256 != event.cell_sha256:
                    raise MlpPageCoordinateIntegrityError(
                        "staged coordinate hit changed under its exact key"
                    )
                cells[event.key_sha256] = replace(
                    cell,
                    hit_count=_bounded_add(cell.hit_count, 1),
                    last_used_clock=clock,
                )

            for staged in captures:
                incoming = _cell_for_stage(staged, clock=clock)
                previous = cells.get(incoming.key.key_sha256)
                if previous is None:
                    cells[incoming.key.key_sha256] = incoming
                    continue
                if (
                    previous.cell_sha256 != incoming.cell_sha256
                    or previous.key != incoming.key
                    or previous.page_ids != incoming.page_ids
                    or previous.selected_width != incoming.selected_width
                    or previous.ranked_page_scores
                    != incoming.ranked_page_scores
                    or previous.total_energy != incoming.total_energy
                    or previous.logical_page_weight_bytes_saved
                    != incoming.logical_page_weight_bytes_saved
                ):
                    raise MlpPageCoordinateIntegrityError(
                        "one exact MLP input produced conflicting coordinates"
                    )
                cells[incoming.key.key_sha256] = replace(
                    previous,
                    support=_bounded_add(previous.support, 1),
                    last_used_clock=clock,
                )

            evicted = 0
            while len(cells) > self.max_cells:
                victim = self._victim(cells)
                del cells[victim.key.key_sha256]
                evicted += 1

            physical_saved = sum(hit.physical_pages_saved for hit in hits)
            logical_saved = sum(
                hit.logical_page_weight_bytes_saved for hit in hits
            )

            def make_state() -> _CoordinateState:
                return _CoordinateState(
                    identity=self.identity,
                    max_cells=self.max_cells,
                    max_state_bytes=self.max_state_bytes,
                    clock=clock,
                    transactions=_bounded_add(state.transactions, 1),
                    staged_captures=_bounded_add(
                        state.staged_captures,
                        staged_capture_count,
                    ),
                    accepted_captures=_bounded_add(
                        state.accepted_captures,
                        len(captures),
                    ),
                    rejected_captures=_bounded_add(
                        state.rejected_captures,
                        rejected_capture_count,
                    ),
                    hit_count=_bounded_add(state.hit_count, len(hits)),
                    physical_pages_saved=_bounded_add(
                        state.physical_pages_saved,
                        physical_saved,
                    ),
                    logical_page_weight_bytes_saved=_bounded_add(
                        state.logical_page_weight_bytes_saved,
                        logical_saved,
                    ),
                    evictions=_bounded_add(state.evictions, evicted),
                    cells=tuple(
                        sorted(
                            cells.values(),
                            key=lambda cell: cell.key.key_sha256,
                        )
                    ),
                )

            next_state = make_state()
            encoded = next_state.to_bytes()
            while len(encoded) > self.max_state_bytes:
                if not cells:
                    raise MlpPageCoordinateIntegrityError(
                        "coordinate capacity cannot hold its empty state"
                    )
                victim = self._victim(cells)
                del cells[victim.key.key_sha256]
                evicted += 1
                next_state = make_state()
                encoded = next_state.to_bytes()

            signature = _atomic_write(self.state_path, encoded)
            self._state = next_state
            self._file_signature = signature
            self._rebuild_index()
            return self._metrics(next_state)

    def refresh(self) -> MlpPageCoordinateMetrics:
        with self._lock:
            self._refresh_if_changed()
            return self._metrics(self._state)

    def metrics(self) -> MlpPageCoordinateMetrics:
        with self._lock:
            self._refresh_if_changed()
            return self._metrics(self._state)

    def _metrics(self, state: _CoordinateState) -> MlpPageCoordinateMetrics:
        return MlpPageCoordinateMetrics(
            identity_sha256=state.identity.identity_sha256,
            state_sha256=_sha256_document(state.to_record()),
            max_cells=state.max_cells,
            max_state_bytes=state.max_state_bytes,
            persisted_bytes=(
                0 if self._file_signature is None else self._file_signature[2]
            ),
            cell_count=len(state.cells),
            clock=state.clock,
            transactions=state.transactions,
            staged_captures=state.staged_captures,
            accepted_captures=state.accepted_captures,
            rejected_captures=state.rejected_captures,
            hit_count=state.hit_count,
            physical_pages_saved=state.physical_pages_saved,
            logical_page_weight_bytes_saved=(
                state.logical_page_weight_bytes_saved
            ),
            evictions=state.evictions,
            support=sum(cell.support for cell in state.cells),
        )


# Acronym-preserving aliases for callers that spell MLP as an initialism.
MLPPageCoordinateBank = MlpPageCoordinateBank
MLPPageCoordinateError = MlpPageCoordinateError
MLPPageCoordinateHit = MlpPageCoordinateHit
MLPPageCoordinateIdentity = MlpPageCoordinateIdentity
MLPPageCoordinateIdentityError = MlpPageCoordinateIdentityError
MLPPageCoordinateIntegrityError = MlpPageCoordinateIntegrityError
MLPPageCoordinateKey = MlpPageCoordinateKey
MLPPageCoordinateMetrics = MlpPageCoordinateMetrics
MLPPageCoordinateStage = MlpPageCoordinateStage
MLPPageCoordinateTransaction = MlpPageCoordinateTransaction


__all__ = [
    "MLP_PAGE_COORDINATE_CELL_SCHEMA",
    "MLP_PAGE_COORDINATE_ENVELOPE_SCHEMA",
    "MLP_PAGE_COORDINATE_EVIDENCE_SCHEMA",
    "MLP_PAGE_COORDINATE_IDENTITY_SCHEMA",
    "MLP_PAGE_COORDINATE_KEY_SCHEMA",
    "MLP_PAGE_COORDINATE_PAYLOAD_SCHEMA",
    "MLP_PAGE_COORDINATE_STATE_SCHEMA",
    "MLPPageCoordinateBank",
    "MLPPageCoordinateError",
    "MLPPageCoordinateHit",
    "MLPPageCoordinateIdentity",
    "MLPPageCoordinateIdentityError",
    "MLPPageCoordinateIntegrityError",
    "MLPPageCoordinateKey",
    "MLPPageCoordinateMetrics",
    "MLPPageCoordinateStage",
    "MLPPageCoordinateTransaction",
    "MlpPageCoordinateBank",
    "MlpPageCoordinateError",
    "MlpPageCoordinateHit",
    "MlpPageCoordinateIdentity",
    "MlpPageCoordinateIdentityError",
    "MlpPageCoordinateIntegrityError",
    "MlpPageCoordinateKey",
    "MlpPageCoordinateMetrics",
    "MlpPageCoordinateStage",
    "MlpPageCoordinateTransaction",
    "StagedMlpPageCoordinate",
]
