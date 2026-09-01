"""Exact, persistent full-attention transition cells for Qwen3.8.

An attention-output crystal is deliberately narrower than a general activation
cache.  Its address seals the immutable runtime math, layer, absolute token
position, exact BF16 attention input, and exact prior ``AttentionState``.  Its
payload contains only the data needed to skip the four attention projections:
the post-``o_proj`` row, the newly appended RoPE key/value rows, and the
optional layer-27 CRSA continuation state and evidence.

Speculative work is first converted to immutable, byte-backed staged cells.
``publish`` commits only the caller-confirmed prefix.  Every persistent update
is a locked read-modify-replace transaction, and every document and tensor is
canonically content-addressed.  No pickle or ambient RNG state is involved.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass, replace
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
from typing import Any, Iterator

import torch


ATTENTION_OUTPUT_CRYSTAL_IDENTITY_SCHEMA = (
    "immer.qwen3.8-attention-output-crystal-identity/v1"
)
ATTENTION_OUTPUT_CRYSTAL_KEY_SCHEMA = (
    "immer.qwen3.8-attention-output-crystal-key/v1"
)
ATTENTION_OUTPUT_CRYSTAL_TENSOR_SCHEMA = (
    "immer.qwen3.8-attention-output-crystal-tensor/v1"
)
ATTENTION_OUTPUT_CRYSTAL_ATTENTION_STATE_SCHEMA = (
    "immer.qwen3.8-attention-output-crystal-attention-state/v1"
)
ATTENTION_OUTPUT_CRYSTAL_PAYLOAD_SCHEMA = (
    "immer.qwen3.8-attention-output-crystal-payload/v1"
)
ATTENTION_OUTPUT_CRYSTAL_CELL_SCHEMA = (
    "immer.qwen3.8-attention-output-crystal-cell/v1"
)
ATTENTION_OUTPUT_CRYSTAL_STATE_SCHEMA = (
    "immer.qwen3.8-attention-output-crystal-bank-state/v1"
)
ATTENTION_OUTPUT_CRYSTAL_ENVELOPE_SCHEMA = (
    "immer.qwen3.8-attention-output-crystal-bank-envelope/v1"
)
ATTENTION_OUTPUT_CRYSTAL_EVIDENCE_SCHEMA = (
    "immer.qwen3.8-attention-output-crystal-evidence/v1"
)
ATTENTION_OUTPUT_CRYSTAL_TENSOR_ABI = (
    "immer.qwen3.8/little-endian-c-contiguous-exact-tensor/v1"
)

NATIVE_HEAD_CRSA_LAYER = 27

_MAX_COUNTER = (1 << 63) - 1
_MAX_LAYER_INDEX = 4095
_MAX_ABSOLUTE_POSITION = (1 << 31) - 1
_MAX_CELLS = 1 << 16
_MAX_TENSOR_RANK = 8
_MAX_TENSOR_DIMENSION = 1 << 24
_MAX_TENSOR_ELEMENTS = 1 << 24
_MAX_CELL_TENSOR_BYTES = 96 * 1024 * 1024
_MAX_EVIDENCE_BYTES = 1024 * 1024
_ABSOLUTE_MAX_STATE_BYTES = 1024 * 1024 * 1024
_DEFAULT_MAX_STATE_BYTES = 512 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_HEX = frozenset("0123456789abcdef")
_DTYPE_SIZES = {"bfloat16": 2, "float32": 4}


class AttentionOutputCrystalError(RuntimeError):
    """An attention-output crystal input or persistent state is invalid."""


class AttentionOutputCrystalIntegrityError(AttentionOutputCrystalError):
    """The bank is malformed, unstable, conflicting, or hash-inconsistent."""


class AttentionOutputCrystalIdentityError(AttentionOutputCrystalError):
    """A key or bank belongs to a different immutable runtime identity."""


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
        raise AttentionOutputCrystalIntegrityError(
            "attention-output crystal data is not canonical JSON"
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


def _shape(value: object, *, field: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise TypeError(f"{field} must be a shape sequence")
    result = tuple(value)
    if not 1 <= len(result) <= _MAX_TENSOR_RANK:
        raise ValueError(f"{field} rank is outside the supported range")
    elements = 1
    for index, dimension in enumerate(result):
        _uint(
            dimension,
            field=f"{field}[{index}]",
            positive=True,
            maximum=_MAX_TENSOR_DIMENSION,
        )
        elements *= dimension
        if elements > _MAX_TENSOR_ELEMENTS:
            raise ValueError(f"{field} exceeds the tensor element limit")
    return result


def _canonical_little_endian(raw: bytes, *, element_size: int) -> bytes:
    if sys.byteorder == "little" or element_size == 1:
        return raw
    if len(raw) % element_size:
        raise AttentionOutputCrystalIntegrityError(
            "exact tensor byte width is invalid"
        )
    result = bytearray(len(raw))
    for start in range(0, len(raw), element_size):
        result[start : start + element_size] = raw[
            start : start + element_size
        ][::-1]
    return bytes(result)


def _native_endian(raw: bytes, *, element_size: int) -> bytes:
    # Reversing each element is its own inverse.
    return _canonical_little_endian(raw, element_size=element_size)


def _tensor_bytes(
    value: torch.Tensor,
    *,
    field: str,
    expected_dtype: torch.dtype,
    allow_negative_infinity: bool,
) -> tuple[tuple[int, ...], bytes]:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{field} must be a torch.Tensor")
    if value.layout != torch.strided:
        raise TypeError(f"{field} must be a strided tensor")
    if value.dtype != expected_dtype:
        raise TypeError(f"{field} must have dtype {expected_dtype}")
    shape = _shape(tuple(value.shape), field=f"{field} shape")
    cpu = value.detach().to(device="cpu").contiguous()
    if allow_negative_infinity:
        invalid = torch.isnan(cpu) | torch.isposinf(cpu)
        if bool(invalid.any().item()):
            raise ValueError(f"{field} may contain only finite values or -inf")
    elif not bool(torch.isfinite(cpu).all().item()):
        raise ValueError(f"{field} must contain only finite values")
    raw = cpu.view(torch.uint8).numpy().reshape(-1).tobytes()
    raw = _canonical_little_endian(raw, element_size=cpu.element_size())
    if len(raw) > _MAX_CELL_TENSOR_BYTES:
        raise ValueError(f"{field} exceeds the tensor byte limit")
    return shape, raw


def _tensor_header(dtype: str, shape: tuple[int, ...]) -> dict[str, object]:
    return {
        "abi": ATTENTION_OUTPUT_CRYSTAL_TENSOR_ABI,
        "dtype": dtype,
        "schema": ATTENTION_OUTPUT_CRYSTAL_TENSOR_SCHEMA,
        "shape": list(shape),
    }


def _exact_tensor_sha256(
    *,
    dtype: str,
    shape: tuple[int, ...],
    data: bytes,
) -> str:
    return _sha256_bytes(_canonical_json(_tensor_header(dtype, shape)) + b"\0" + data)


def canonical_bf16_tensor_bytes(value: torch.Tensor) -> bytes:
    """Return canonical little-endian bytes for one finite exact BF16 tensor."""

    _shape_value, raw = _tensor_bytes(
        value,
        field="BF16 tensor",
        expected_dtype=torch.bfloat16,
        allow_negative_infinity=False,
    )
    return raw


def canonical_bf16_tensor_sha256(value: torch.Tensor) -> str:
    """Hash BF16 dtype, shape, ABI, and exact canonical storage bytes."""

    shape, raw = _tensor_bytes(
        value,
        field="BF16 tensor",
        expected_dtype=torch.bfloat16,
        allow_negative_infinity=False,
    )
    return _exact_tensor_sha256(dtype="bfloat16", shape=shape, data=raw)


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AttentionOutputCrystalIntegrityError(
                f"duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


def _decode_canonical_json(value: bytes, *, label: str) -> Any:
    try:
        document = json.loads(value, object_pairs_hook=_json_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttentionOutputCrystalIntegrityError(
            f"{label} is not valid JSON"
        ) from exc
    if _canonical_json(document) != value:
        raise AttentionOutputCrystalIntegrityError(f"{label} is not canonical JSON")
    return document


def _canonical_evidence(value: object | None) -> bytes | None:
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise TypeError("crsa_evidence must be a mapping, dataclass, or None")
    raw = _canonical_json(value)
    if len(raw) > _MAX_EVIDENCE_BYTES:
        raise ValueError("crsa_evidence exceeds its byte limit")
    decoded = _decode_canonical_json(raw, label="CRSA evidence")
    if not isinstance(decoded, Mapping):
        raise ValueError("crsa_evidence must encode one JSON object")
    return raw


@dataclass(frozen=True, slots=True)
class _ExactTensor:
    dtype: str
    shape: tuple[int, ...]
    data: bytes
    data_sha256: str
    tensor_sha256: str

    def __post_init__(self) -> None:
        if self.dtype not in _DTYPE_SIZES:
            raise ValueError("exact tensor dtype is unsupported")
        shape = _shape(self.shape, field="exact tensor shape")
        if not isinstance(self.data, bytes):
            raise TypeError("exact tensor data must be immutable bytes")
        expected_bytes = _DTYPE_SIZES[self.dtype]
        for dimension in shape:
            expected_bytes *= dimension
        if len(self.data) != expected_bytes:
            raise ValueError("exact tensor byte length disagrees with its shape")
        if len(self.data) > _MAX_CELL_TENSOR_BYTES:
            raise ValueError("exact tensor exceeds its byte limit")
        _digest(self.data_sha256, "exact tensor data_sha256")
        _digest(self.tensor_sha256, "exact tensor tensor_sha256")
        if self.data_sha256 != _sha256_bytes(self.data):
            raise ValueError("exact tensor data SHA-256 mismatch")
        expected = _exact_tensor_sha256(
            dtype=self.dtype,
            shape=shape,
            data=self.data,
        )
        if self.tensor_sha256 != expected:
            raise ValueError("exact tensor SHA-256 mismatch")
        object.__setattr__(self, "shape", shape)

    @classmethod
    def capture(
        cls,
        value: torch.Tensor,
        *,
        field: str,
        dtype: torch.dtype,
        allow_negative_infinity: bool = False,
    ) -> "_ExactTensor":
        shape, raw = _tensor_bytes(
            value,
            field=field,
            expected_dtype=dtype,
            allow_negative_infinity=allow_negative_infinity,
        )
        dtype_name = str(dtype).removeprefix("torch.")
        return cls(
            dtype=dtype_name,
            shape=shape,
            data=raw,
            data_sha256=_sha256_bytes(raw),
            tensor_sha256=_exact_tensor_sha256(
                dtype=dtype_name,
                shape=shape,
                data=raw,
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "abi": ATTENTION_OUTPUT_CRYSTAL_TENSOR_ABI,
            "data_base64": base64.b64encode(self.data).decode("ascii"),
            "data_sha256": self.data_sha256,
            "dtype": self.dtype,
            "schema": ATTENTION_OUTPUT_CRYSTAL_TENSOR_SCHEMA,
            "shape": list(self.shape),
            "tensor_sha256": self.tensor_sha256,
        }

    @classmethod
    def from_record(cls, value: object) -> "_ExactTensor":
        if not isinstance(value, Mapping) or set(value) != {
            "abi",
            "data_base64",
            "data_sha256",
            "dtype",
            "schema",
            "shape",
            "tensor_sha256",
        }:
            raise AttentionOutputCrystalIntegrityError(
                "exact tensor fields are invalid"
            )
        if value["schema"] != ATTENTION_OUTPUT_CRYSTAL_TENSOR_SCHEMA:
            raise AttentionOutputCrystalIntegrityError(
                "exact tensor schema is invalid"
            )
        if value["abi"] != ATTENTION_OUTPUT_CRYSTAL_TENSOR_ABI:
            raise AttentionOutputCrystalIntegrityError("exact tensor ABI is invalid")
        encoded = value["data_base64"]
        if not isinstance(encoded, str) or not encoded.isascii():
            raise AttentionOutputCrystalIntegrityError(
                "exact tensor base64 is invalid"
            )
        try:
            data = base64.b64decode(encoded.encode("ascii"), validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise AttentionOutputCrystalIntegrityError(
                "exact tensor base64 is invalid"
            ) from exc
        if base64.b64encode(data).decode("ascii") != encoded:
            raise AttentionOutputCrystalIntegrityError(
                "exact tensor base64 is not canonical"
            )
        try:
            return cls(
                dtype=value["dtype"],
                shape=tuple(value["shape"]),
                data=data,
                data_sha256=value["data_sha256"],
                tensor_sha256=value["tensor_sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise AttentionOutputCrystalIntegrityError(
                f"exact tensor values are invalid: {exc}"
            ) from exc

    def tensor(self, *, device: torch.device | str | None = None) -> torch.Tensor:
        native = _native_endian(self.data, element_size=_DTYPE_SIZES[self.dtype])
        dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.float32
        result = torch.frombuffer(bytearray(native), dtype=dtype).clone().reshape(
            self.shape
        )
        if device is not None:
            result = result.to(device=device)
        return result.contiguous()


def canonical_attention_state_sha256(state: object | None) -> str:
    """Hash an exact BF16 ``AttentionState`` without importing model code.

    ``None`` has a stable empty-state address.  Non-empty states are accepted
    structurally and must expose ``key``, ``value``, and ``crsa_log_usage``.
    K/V are exact BF16; native CRSA usage, when present, is exact FP32 and may
    contain ``-inf`` but never NaN or positive infinity.
    """

    if state is None:
        record: dict[str, object] = {
            "crsa_log_usage": None,
            "key": None,
            "schema": ATTENTION_OUTPUT_CRYSTAL_ATTENTION_STATE_SCHEMA,
            "value": None,
        }
        return _sha256_document(record)
    if not all(hasattr(state, field) for field in ("key", "value", "crsa_log_usage")):
        raise TypeError("state must be AttentionState-like or None")
    key = _ExactTensor.capture(
        getattr(state, "key"),
        field="AttentionState key",
        dtype=torch.bfloat16,
    )
    value = _ExactTensor.capture(
        getattr(state, "value"),
        field="AttentionState value",
        dtype=torch.bfloat16,
    )
    if key.shape != value.shape:
        raise ValueError("AttentionState key/value shapes must match")
    usage_value = getattr(state, "crsa_log_usage")
    usage = (
        None
        if usage_value is None
        else _ExactTensor.capture(
            usage_value,
            field="AttentionState CRSA usage",
            dtype=torch.float32,
            allow_negative_infinity=True,
        )
    )
    if usage is not None and usage.shape != (key.shape[0], 4, key.shape[2]):
        raise ValueError("AttentionState CRSA usage shape is invalid")
    return _sha256_document(
        {
            "crsa_log_usage": None if usage is None else usage.tensor_sha256,
            "key": key.tensor_sha256,
            "schema": ATTENTION_OUTPUT_CRYSTAL_ATTENTION_STATE_SCHEMA,
            "value": value.tensor_sha256,
        }
    )


EMPTY_ATTENTION_STATE_SHA256 = canonical_attention_state_sha256(None)


@dataclass(frozen=True, slots=True)
class AttentionOutputCrystalIdentity:
    """Immutable runtime-math identity allowed to use one crystal bank."""

    runtime_math_sha256: str
    tensor_abi: str = ATTENTION_OUTPUT_CRYSTAL_TENSOR_ABI

    def __post_init__(self) -> None:
        _digest(self.runtime_math_sha256, "runtime_math_sha256")
        if self.tensor_abi != ATTENTION_OUTPUT_CRYSTAL_TENSOR_ABI:
            raise ValueError("tensor ABI is not implemented by this runtime")

    @property
    def identity_sha256(self) -> str:
        return _sha256_document(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "runtime_math_sha256": self.runtime_math_sha256,
            "schema": ATTENTION_OUTPUT_CRYSTAL_IDENTITY_SCHEMA,
            "tensor_abi": self.tensor_abi,
        }

    @classmethod
    def from_record(cls, value: object) -> "AttentionOutputCrystalIdentity":
        if not isinstance(value, Mapping) or set(value) != {
            "runtime_math_sha256",
            "schema",
            "tensor_abi",
        }:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal identity fields are invalid"
            )
        if value["schema"] != ATTENTION_OUTPUT_CRYSTAL_IDENTITY_SCHEMA:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal identity schema is invalid"
            )
        try:
            return cls(
                runtime_math_sha256=value["runtime_math_sha256"],
                tensor_abi=value["tensor_abi"],
            )
        except (TypeError, ValueError) as exc:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal identity values are invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class AttentionOutputCrystalKey:
    """Exact full-attention transition address."""

    identity_sha256: str
    runtime_math_sha256: str
    layer_index: int
    absolute_position: int
    attention_input_sha256: str
    prior_state_sha256: str
    key_sha256: str

    def __post_init__(self) -> None:
        _digest(self.identity_sha256, "key identity_sha256")
        _digest(self.runtime_math_sha256, "key runtime_math_sha256")
        _uint(
            self.layer_index,
            field="layer_index",
            maximum=_MAX_LAYER_INDEX,
        )
        _uint(
            self.absolute_position,
            field="absolute_position",
            maximum=_MAX_ABSOLUTE_POSITION,
        )
        _digest(self.attention_input_sha256, "attention_input_sha256")
        _digest(self.prior_state_sha256, "prior_state_sha256")
        _digest(self.key_sha256, "key_sha256")
        if self.key_sha256 != _sha256_document(self._address_record()):
            raise ValueError("attention-output crystal key SHA-256 mismatch")

    def _address_record(self) -> dict[str, object]:
        return {
            "absolute_position": self.absolute_position,
            "attention_input_sha256": self.attention_input_sha256,
            "identity_sha256": self.identity_sha256,
            "layer_index": self.layer_index,
            "prior_state_sha256": self.prior_state_sha256,
            "runtime_math_sha256": self.runtime_math_sha256,
            "schema": ATTENTION_OUTPUT_CRYSTAL_KEY_SCHEMA,
        }

    def to_record(self) -> dict[str, object]:
        return self._address_record() | {"key_sha256": self.key_sha256}

    @classmethod
    def create(
        cls,
        *,
        identity: AttentionOutputCrystalIdentity,
        layer_index: int,
        absolute_position: int,
        attention_input_sha256: str,
        prior_state_sha256: str,
    ) -> "AttentionOutputCrystalKey":
        address = {
            "absolute_position": absolute_position,
            "attention_input_sha256": attention_input_sha256,
            "identity_sha256": identity.identity_sha256,
            "layer_index": layer_index,
            "prior_state_sha256": prior_state_sha256,
            "runtime_math_sha256": identity.runtime_math_sha256,
            "schema": ATTENTION_OUTPUT_CRYSTAL_KEY_SCHEMA,
        }
        return cls(
            identity_sha256=identity.identity_sha256,
            runtime_math_sha256=identity.runtime_math_sha256,
            layer_index=layer_index,
            absolute_position=absolute_position,
            attention_input_sha256=attention_input_sha256,
            prior_state_sha256=prior_state_sha256,
            key_sha256=_sha256_document(address),
        )

    @classmethod
    def from_record(cls, value: object) -> "AttentionOutputCrystalKey":
        if not isinstance(value, Mapping) or set(value) != {
            "absolute_position",
            "attention_input_sha256",
            "identity_sha256",
            "key_sha256",
            "layer_index",
            "prior_state_sha256",
            "runtime_math_sha256",
            "schema",
        }:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal key fields are invalid"
            )
        if value["schema"] != ATTENTION_OUTPUT_CRYSTAL_KEY_SCHEMA:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal key schema is invalid"
            )
        try:
            return cls(
                identity_sha256=value["identity_sha256"],
                runtime_math_sha256=value["runtime_math_sha256"],
                layer_index=value["layer_index"],
                absolute_position=value["absolute_position"],
                attention_input_sha256=value["attention_input_sha256"],
                prior_state_sha256=value["prior_state_sha256"],
                key_sha256=value["key_sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal key values are invalid"
            ) from exc


def _require_tensor_values(
    value: _ExactTensor,
    *,
    field: str,
    allow_negative_infinity: bool,
) -> None:
    tensor = value.tensor()
    if allow_negative_infinity:
        invalid = torch.isnan(tensor) | torch.isposinf(tensor)
        if bool(invalid.any().item()):
            raise ValueError(f"{field} may contain only finite values or -inf")
    elif not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{field} must contain only finite values")


@dataclass(frozen=True, slots=True)
class _AttentionPayload:
    post_o_proj: _ExactTensor
    appended_rope_key: _ExactTensor
    appended_value: _ExactTensor
    next_crsa_usage: _ExactTensor | None
    crsa_evidence_json: bytes | None
    crsa_evidence_sha256: str | None
    payload_sha256: str

    def __post_init__(self) -> None:
        for field in ("post_o_proj", "appended_rope_key", "appended_value"):
            value = getattr(self, field)
            if not isinstance(value, _ExactTensor) or value.dtype != "bfloat16":
                raise ValueError(f"{field} must be an exact BF16 tensor")
            _require_tensor_values(
                value,
                field=field,
                allow_negative_infinity=False,
            )
        if (
            len(self.post_o_proj.shape) != 3
            or self.post_o_proj.shape[0:2] != (1, 1)
        ):
            raise ValueError("post_o_proj must have exact shape [1, 1, hidden]")
        key_shape = self.appended_rope_key.shape
        if (
            len(key_shape) != 4
            or key_shape[0] != 1
            or key_shape[2] != 1
        ):
            raise ValueError(
                "appended_rope_key must have exact shape [1, kv_heads, 1, head_dim]"
            )
        if self.appended_value.shape != key_shape:
            raise ValueError("appended key/value shapes must match exactly")
        if self.next_crsa_usage is not None:
            usage = self.next_crsa_usage
            if not isinstance(usage, _ExactTensor) or usage.dtype != "float32":
                raise ValueError("next_crsa_usage must be an exact FP32 tensor")
            if (
                len(usage.shape) != 3
                or usage.shape[0] != 1
                or usage.shape[1] != 4
            ):
                raise ValueError(
                    "next_crsa_usage must have exact shape [1, 4, history]"
                )
            _require_tensor_values(
                usage,
                field="next_crsa_usage",
                allow_negative_infinity=True,
            )
        if self.crsa_evidence_json is None:
            if self.crsa_evidence_sha256 is not None:
                raise ValueError("CRSA evidence digest exists without evidence")
        else:
            if not isinstance(self.crsa_evidence_json, bytes):
                raise TypeError("CRSA evidence JSON must be immutable bytes")
            if len(self.crsa_evidence_json) > _MAX_EVIDENCE_BYTES:
                raise ValueError("CRSA evidence exceeds its byte limit")
            decoded = _decode_canonical_json(
                self.crsa_evidence_json,
                label="CRSA evidence",
            )
            if not isinstance(decoded, Mapping):
                raise ValueError("CRSA evidence must encode one JSON object")
            _digest(self.crsa_evidence_sha256, "CRSA evidence SHA-256")
            if self.crsa_evidence_sha256 != _sha256_bytes(
                self.crsa_evidence_json
            ):
                raise ValueError("CRSA evidence SHA-256 mismatch")
        tensor_bytes = sum(
            len(value.data)
            for value in (
                self.post_o_proj,
                self.appended_rope_key,
                self.appended_value,
                self.next_crsa_usage,
            )
            if value is not None
        )
        if tensor_bytes > _MAX_CELL_TENSOR_BYTES:
            raise ValueError("attention-output payload exceeds its tensor byte limit")
        _digest(self.payload_sha256, "payload_sha256")
        if self.payload_sha256 != _sha256_document(self._address_record()):
            raise ValueError("attention-output payload SHA-256 mismatch")

    def _address_record(self) -> dict[str, object]:
        return {
            "appended_rope_key_sha256": self.appended_rope_key.tensor_sha256,
            "appended_value_sha256": self.appended_value.tensor_sha256,
            "crsa_evidence_sha256": self.crsa_evidence_sha256,
            "next_crsa_usage_sha256": (
                None
                if self.next_crsa_usage is None
                else self.next_crsa_usage.tensor_sha256
            ),
            "post_o_proj_sha256": self.post_o_proj.tensor_sha256,
            "schema": ATTENTION_OUTPUT_CRYSTAL_PAYLOAD_SCHEMA,
        }

    @classmethod
    def capture(
        cls,
        *,
        post_o_proj: torch.Tensor,
        appended_rope_key: torch.Tensor,
        appended_value: torch.Tensor,
        next_crsa_usage: torch.Tensor | None,
        crsa_evidence: object | None,
    ) -> "_AttentionPayload":
        output = _ExactTensor.capture(
            post_o_proj,
            field="post_o_proj",
            dtype=torch.bfloat16,
        )
        key = _ExactTensor.capture(
            appended_rope_key,
            field="appended_rope_key",
            dtype=torch.bfloat16,
        )
        value = _ExactTensor.capture(
            appended_value,
            field="appended_value",
            dtype=torch.bfloat16,
        )
        usage = (
            None
            if next_crsa_usage is None
            else _ExactTensor.capture(
                next_crsa_usage,
                field="next_crsa_usage",
                dtype=torch.float32,
                allow_negative_infinity=True,
            )
        )
        evidence_json = _canonical_evidence(crsa_evidence)
        evidence_sha256 = (
            None if evidence_json is None else _sha256_bytes(evidence_json)
        )
        address = {
            "appended_rope_key_sha256": key.tensor_sha256,
            "appended_value_sha256": value.tensor_sha256,
            "crsa_evidence_sha256": evidence_sha256,
            "next_crsa_usage_sha256": (
                None if usage is None else usage.tensor_sha256
            ),
            "post_o_proj_sha256": output.tensor_sha256,
            "schema": ATTENTION_OUTPUT_CRYSTAL_PAYLOAD_SCHEMA,
        }
        return cls(
            post_o_proj=output,
            appended_rope_key=key,
            appended_value=value,
            next_crsa_usage=usage,
            crsa_evidence_json=evidence_json,
            crsa_evidence_sha256=evidence_sha256,
            payload_sha256=_sha256_document(address),
        )

    @property
    def tensor_bytes(self) -> int:
        return sum(
            len(value.data)
            for value in (
                self.post_o_proj,
                self.appended_rope_key,
                self.appended_value,
                self.next_crsa_usage,
            )
            if value is not None
        )

    def to_record(self) -> dict[str, object]:
        return {
            "appended_rope_key": self.appended_rope_key.to_record(),
            "appended_value": self.appended_value.to_record(),
            "crsa_evidence_json": (
                None
                if self.crsa_evidence_json is None
                else self.crsa_evidence_json.decode("utf-8")
            ),
            "crsa_evidence_sha256": self.crsa_evidence_sha256,
            "next_crsa_usage": (
                None
                if self.next_crsa_usage is None
                else self.next_crsa_usage.to_record()
            ),
            "payload_sha256": self.payload_sha256,
            "post_o_proj": self.post_o_proj.to_record(),
            "schema": ATTENTION_OUTPUT_CRYSTAL_PAYLOAD_SCHEMA,
        }

    @classmethod
    def from_record(cls, value: object) -> "_AttentionPayload":
        if not isinstance(value, Mapping) or set(value) != {
            "appended_rope_key",
            "appended_value",
            "crsa_evidence_json",
            "crsa_evidence_sha256",
            "next_crsa_usage",
            "payload_sha256",
            "post_o_proj",
            "schema",
        }:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output payload fields are invalid"
            )
        if value["schema"] != ATTENTION_OUTPUT_CRYSTAL_PAYLOAD_SCHEMA:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output payload schema is invalid"
            )
        evidence_value = value["crsa_evidence_json"]
        if evidence_value is not None and not isinstance(evidence_value, str):
            raise AttentionOutputCrystalIntegrityError(
                "CRSA evidence JSON is invalid"
            )
        try:
            return cls(
                post_o_proj=_ExactTensor.from_record(value["post_o_proj"]),
                appended_rope_key=_ExactTensor.from_record(
                    value["appended_rope_key"]
                ),
                appended_value=_ExactTensor.from_record(value["appended_value"]),
                next_crsa_usage=(
                    None
                    if value["next_crsa_usage"] is None
                    else _ExactTensor.from_record(value["next_crsa_usage"])
                ),
                crsa_evidence_json=(
                    None if evidence_value is None else evidence_value.encode("utf-8")
                ),
                crsa_evidence_sha256=value["crsa_evidence_sha256"],
                payload_sha256=value["payload_sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output payload values are invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class StagedAttentionOutputCrystal:
    """Immutable, non-persistent exact transition awaiting acceptance."""

    key: AttentionOutputCrystalKey
    payload: _AttentionPayload
    logical_projection_bytes: int
    skipped_projection_calls: int
    cell_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, AttentionOutputCrystalKey):
            raise TypeError("staged key must be an AttentionOutputCrystalKey")
        if not isinstance(self.payload, _AttentionPayload):
            raise TypeError("staged payload is invalid")
        _uint(
            self.logical_projection_bytes,
            field="logical_projection_bytes",
            positive=True,
        )
        _uint(
            self.skipped_projection_calls,
            field="skipped_projection_calls",
            positive=True,
        )
        _digest(self.cell_sha256, "staged cell_sha256")
        if self.cell_sha256 != self.expected_cell_sha256():
            raise ValueError("staged attention-output cell SHA-256 mismatch")
        _validate_crsa_contract(self.key, self.payload)

    def expected_cell_sha256(self) -> str:
        return _sha256_document(
            {
                "key_sha256": self.key.key_sha256,
                "logical_projection_bytes": self.logical_projection_bytes,
                "payload_sha256": self.payload.payload_sha256,
                "schema": ATTENTION_OUTPUT_CRYSTAL_CELL_SCHEMA,
                "skipped_projection_calls": self.skipped_projection_calls,
            }
        )

    @property
    def absolute_position(self) -> int:
        return self.key.absolute_position

    @property
    def layer_index(self) -> int:
        return self.key.layer_index

    @property
    def payload_sha256(self) -> str:
        return self.payload.payload_sha256

    @property
    def tensor_bytes(self) -> int:
        return self.payload.tensor_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "absolute_position": self.absolute_position,
            "cell_sha256": self.cell_sha256,
            "key_sha256": self.key.key_sha256,
            "layer_index": self.layer_index,
            "logical_projection_bytes": self.logical_projection_bytes,
            "payload_sha256": self.payload_sha256,
            "skipped_projection_calls": self.skipped_projection_calls,
            "tensor_bytes": self.tensor_bytes,
        }


# Concise compatibility spelling for integrations that name the object by role.
AttentionOutputCrystalStage = StagedAttentionOutputCrystal


def _validate_crsa_contract(
    key: AttentionOutputCrystalKey,
    payload: _AttentionPayload,
) -> None:
    usage = payload.next_crsa_usage
    evidence_json = payload.crsa_evidence_json
    if usage is None and evidence_json is None:
        return
    if key.layer_index != NATIVE_HEAD_CRSA_LAYER:
        raise ValueError("CRSA continuation payload is valid only for layer 27")
    if usage is not None and usage.shape != (
        1,
        4,
        key.absolute_position + 1,
    ):
        raise ValueError(
            "next_crsa_usage history must end at the key absolute position"
        )
    if evidence_json is None:
        return
    evidence = _decode_canonical_json(evidence_json, label="CRSA evidence")
    assert isinstance(evidence, Mapping)
    expected_fields = {
        "layer": key.layer_index,
        "query_start": key.absolute_position,
        "query_length": 1,
        "history_length_after": key.absolute_position + 1,
    }
    for field, expected in expected_fields.items():
        if field in evidence and evidence[field] != expected:
            raise ValueError(f"CRSA evidence {field} disagrees with the exact key")


@dataclass(frozen=True, slots=True)
class _AttentionCell:
    cell_sha256: str
    key: AttentionOutputCrystalKey
    payload: _AttentionPayload
    logical_projection_bytes: int
    skipped_projection_calls: int
    support: int
    hit_count: int
    created_clock: int
    last_used_clock: int

    def __post_init__(self) -> None:
        _digest(self.cell_sha256, "cell_sha256")
        if not isinstance(self.key, AttentionOutputCrystalKey):
            raise ValueError("attention-output cell key is invalid")
        if not isinstance(self.payload, _AttentionPayload):
            raise ValueError("attention-output cell payload is invalid")
        for field in (
            "logical_projection_bytes",
            "skipped_projection_calls",
            "support",
            "created_clock",
            "last_used_clock",
        ):
            _uint(getattr(self, field), field=field, positive=True)
        _uint(self.hit_count, field="hit_count")
        if self.last_used_clock < self.created_clock:
            raise ValueError("cell last-used clock precedes creation")
        if self.cell_sha256 != self.expected_cell_sha256():
            raise ValueError("attention-output cell SHA-256 mismatch")
        _validate_crsa_contract(self.key, self.payload)

    def expected_cell_sha256(self) -> str:
        return _sha256_document(
            {
                "key_sha256": self.key.key_sha256,
                "logical_projection_bytes": self.logical_projection_bytes,
                "payload_sha256": self.payload.payload_sha256,
                "schema": ATTENTION_OUTPUT_CRYSTAL_CELL_SCHEMA,
                "skipped_projection_calls": self.skipped_projection_calls,
            }
        )

    @property
    def value_units(self) -> int:
        """Integer reuse value used by deterministic value/LRU eviction."""

        observations = min(_MAX_COUNTER, self.support + self.hit_count)
        return min(_MAX_COUNTER, observations * self.logical_projection_bytes)

    @property
    def tensor_bytes(self) -> int:
        return self.payload.tensor_bytes

    def to_record(self) -> dict[str, object]:
        return {
            "cell_sha256": self.cell_sha256,
            "created_clock": self.created_clock,
            "hit_count": self.hit_count,
            "key": self.key.to_record(),
            "last_used_clock": self.last_used_clock,
            "logical_projection_bytes": self.logical_projection_bytes,
            "payload": self.payload.to_record(),
            "schema": ATTENTION_OUTPUT_CRYSTAL_CELL_SCHEMA,
            "skipped_projection_calls": self.skipped_projection_calls,
            "support": self.support,
        }

    @classmethod
    def from_record(cls, value: object) -> "_AttentionCell":
        if not isinstance(value, Mapping) or set(value) != {
            "cell_sha256",
            "created_clock",
            "hit_count",
            "key",
            "last_used_clock",
            "logical_projection_bytes",
            "payload",
            "schema",
            "skipped_projection_calls",
            "support",
        }:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output cell fields are invalid"
            )
        if value["schema"] != ATTENTION_OUTPUT_CRYSTAL_CELL_SCHEMA:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output cell schema is invalid"
            )
        try:
            return cls(
                cell_sha256=value["cell_sha256"],
                key=AttentionOutputCrystalKey.from_record(value["key"]),
                payload=_AttentionPayload.from_record(value["payload"]),
                logical_projection_bytes=value["logical_projection_bytes"],
                skipped_projection_calls=value["skipped_projection_calls"],
                support=value["support"],
                hit_count=value["hit_count"],
                created_clock=value["created_clock"],
                last_used_clock=value["last_used_clock"],
            )
        except (TypeError, ValueError) as exc:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output cell values are invalid"
            ) from exc


def _cell_for_stage(
    staged: StagedAttentionOutputCrystal,
    *,
    clock: int,
) -> _AttentionCell:
    return _AttentionCell(
        cell_sha256=staged.cell_sha256,
        key=staged.key,
        payload=staged.payload,
        logical_projection_bytes=staged.logical_projection_bytes,
        skipped_projection_calls=staged.skipped_projection_calls,
        support=1,
        hit_count=0,
        created_clock=clock,
        last_used_clock=clock,
    )


@dataclass(frozen=True, slots=True)
class _AttentionState:
    identity: AttentionOutputCrystalIdentity
    max_cells: int
    max_state_bytes: int
    clock: int = 0
    publish_transactions: int = 0
    staged_captures: int = 0
    accepted_captures: int = 0
    rejected_captures: int = 0
    hit_count: int = 0
    evictions: int = 0
    skipped_projection_calls_saved: int = 0
    logical_projection_bytes_saved: int = 0
    cells: tuple[_AttentionCell, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.identity, AttentionOutputCrystalIdentity):
            raise ValueError("attention-output state identity is invalid")
        _uint(self.max_cells, field="max_cells", positive=True, maximum=_MAX_CELLS)
        _uint(
            self.max_state_bytes,
            field="max_state_bytes",
            positive=True,
            maximum=_ABSOLUTE_MAX_STATE_BYTES,
        )
        for field in (
            "clock",
            "publish_transactions",
            "staged_captures",
            "accepted_captures",
            "rejected_captures",
            "hit_count",
            "evictions",
            "skipped_projection_calls_saved",
            "logical_projection_bytes_saved",
        ):
            _uint(getattr(self, field), field=field)
        cells = tuple(self.cells)
        if len(cells) > self.max_cells:
            raise ValueError("attention-output state exceeds max_cells")
        if tuple(sorted(cells, key=lambda cell: cell.key.key_sha256)) != cells:
            raise ValueError("attention-output cells are not canonically ordered")
        if len({cell.key.key_sha256 for cell in cells}) != len(cells):
            raise ValueError("attention-output state contains duplicate exact keys")
        if len({cell.cell_sha256 for cell in cells}) != len(cells):
            raise ValueError("attention-output state contains duplicate cells")
        if any(
            cell.key.identity_sha256 != self.identity.identity_sha256
            or cell.key.runtime_math_sha256 != self.identity.runtime_math_sha256
            or cell.last_used_clock > self.clock
            for cell in cells
        ):
            raise ValueError("attention-output cell identity or clock is invalid")
        if self.accepted_captures > self.staged_captures:
            raise ValueError("accepted capture count exceeds staged captures")
        if self.rejected_captures > self.staged_captures:
            raise ValueError("rejected capture count exceeds staged captures")
        if min(_MAX_COUNTER, sum(cell.hit_count for cell in cells)) > self.hit_count:
            raise ValueError("cell hits exceed the persistent bank hit counter")
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
            "logical_projection_bytes_saved": (
                self.logical_projection_bytes_saved
            ),
            "max_cells": self.max_cells,
            "max_state_bytes": self.max_state_bytes,
            "publish_transactions": self.publish_transactions,
            "rejected_captures": self.rejected_captures,
            "schema": ATTENTION_OUTPUT_CRYSTAL_STATE_SCHEMA,
            "skipped_projection_calls_saved": (
                self.skipped_projection_calls_saved
            ),
            "staged_captures": self.staged_captures,
        }

    @classmethod
    def from_record(cls, value: object) -> "_AttentionState":
        if not isinstance(value, Mapping) or set(value) != {
            "accepted_captures",
            "cells",
            "clock",
            "evictions",
            "hit_count",
            "identity",
            "identity_sha256",
            "logical_projection_bytes_saved",
            "max_cells",
            "max_state_bytes",
            "publish_transactions",
            "rejected_captures",
            "schema",
            "skipped_projection_calls_saved",
            "staged_captures",
        }:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal state fields are invalid"
            )
        if value["schema"] != ATTENTION_OUTPUT_CRYSTAL_STATE_SCHEMA:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal state schema is invalid"
            )
        cells_value = value["cells"]
        if not isinstance(cells_value, list):
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal cell table is invalid"
            )
        try:
            identity = AttentionOutputCrystalIdentity.from_record(value["identity"])
            if value["identity_sha256"] != identity.identity_sha256:
                raise ValueError("attention-output state identity SHA-256 mismatch")
            return cls(
                identity=identity,
                max_cells=value["max_cells"],
                max_state_bytes=value["max_state_bytes"],
                clock=value["clock"],
                publish_transactions=value["publish_transactions"],
                staged_captures=value["staged_captures"],
                accepted_captures=value["accepted_captures"],
                rejected_captures=value["rejected_captures"],
                hit_count=value["hit_count"],
                evictions=value["evictions"],
                skipped_projection_calls_saved=value[
                    "skipped_projection_calls_saved"
                ],
                logical_projection_bytes_saved=value[
                    "logical_projection_bytes_saved"
                ],
                cells=tuple(_AttentionCell.from_record(cell) for cell in cells_value),
            )
        except (TypeError, ValueError) as exc:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal state values are invalid"
            ) from exc

    def to_bytes(self) -> bytes:
        body = self.to_record()
        envelope = {
            "body": body,
            "body_sha256": _sha256_document(body),
            "schema": ATTENTION_OUTPUT_CRYSTAL_ENVELOPE_SCHEMA,
        }
        encoded = _canonical_json(envelope)
        if len(encoded) > _ABSOLUTE_MAX_STATE_BYTES:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal state exceeds its absolute byte limit"
            )
        return encoded

    @classmethod
    def from_bytes(cls, value: bytes) -> "_AttentionState":
        document = _decode_canonical_json(
            value,
            label="attention-output crystal state",
        )
        if not isinstance(document, Mapping) or set(document) != {
            "body",
            "body_sha256",
            "schema",
        }:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal envelope fields are invalid"
            )
        if document["schema"] != ATTENTION_OUTPUT_CRYSTAL_ENVELOPE_SCHEMA:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal envelope schema is invalid"
            )
        if not _is_sha256(document["body_sha256"]):
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal body digest is invalid"
            )
        if document["body_sha256"] != _sha256_document(document["body"]):
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal body SHA-256 mismatch"
            )
        return cls.from_record(document["body"])


def _stable_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _stable_regular_bytes(
    path: Path,
    *,
    max_bytes: int,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise AttentionOutputCrystalIntegrityError(
            "attention-output crystal state cannot be opened"
        ) from exc
    try:
        before = os.fstat(descriptor)
        linked_before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(linked_before.st_mode)
            or not _same_inode(before, linked_before)
        ):
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal state must be a stable regular file"
            )
        if before.st_size < 0 or before.st_size > max_bytes:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal state exceeds its byte limit"
            )
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
            if not chunk:
                raise AttentionOutputCrystalIntegrityError(
                    "attention-output crystal state was truncated"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal state grew while reading"
            )
        after = os.fstat(descriptor)
        linked_after = os.lstat(path)
        if (
            _stable_signature(before) != _stable_signature(after)
            or _stable_signature(linked_before) != _stable_signature(linked_after)
            or not _same_inode(after, linked_after)
        ):
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal state changed while reading"
            )
        return b"".join(chunks), _stable_signature(after)
    except OSError as exc:
        raise AttentionOutputCrystalIntegrityError(
            "attention-output crystal state cannot be read safely"
        ) from exc
    finally:
        os.close(descriptor)


_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.RLock] = {}


def _thread_lock(path: Path) -> threading.RLock:
    key = os.path.abspath(os.fspath(path))
    with _THREAD_LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


def _validate_parent(path: Path) -> tuple[Path, os.stat_result]:
    parent = path.parent
    try:
        value = os.lstat(parent)
    except OSError as exc:
        raise AttentionOutputCrystalIntegrityError(
            "attention-output crystal state parent is unavailable"
        ) from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise AttentionOutputCrystalIntegrityError(
            "attention-output crystal state parent must be a directory"
        )
    return parent, value


@contextmanager
def _exclusive_state_lock(path: Path) -> Iterator[None]:
    """Serialize one bank transaction across processes and threads."""

    lock = _thread_lock(path)
    with lock:
        parent, parent_before = _validate_parent(path)
        lock_path = parent / f".{path.name}.lock"
        flags = (
            os.O_RDWR
            | os.O_CREAT
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal lock is unavailable"
            ) from exc
        try:
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            opened = os.fstat(descriptor)
            linked = os.lstat(lock_path)
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(linked.st_mode)
                or not _same_inode(opened, linked)
            ):
                raise AttentionOutputCrystalIntegrityError(
                    "attention-output crystal lock is not a stable regular file"
                )
            yield
            parent_after = os.lstat(parent)
            linked_after = os.lstat(lock_path)
            if (
                not _same_inode(opened, linked_after)
                or not _same_inode(parent_before, parent_after)
            ):
                raise AttentionOutputCrystalIntegrityError(
                    "attention-output crystal lock changed during transaction"
                )
        except OSError as exc:
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal lock failed"
            ) from exc
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> tuple[int, int, int, int, int]:
    parent, parent_before = _validate_parent(path)
    try:
        destination = os.lstat(path)
    except FileNotFoundError:
        destination = None
    except OSError as exc:
        raise AttentionOutputCrystalIntegrityError(
            "attention-output crystal destination is unavailable"
        ) from exc
    if destination is not None and (
        stat.S_ISLNK(destination.st_mode) or not stat.S_ISREG(destination.st_mode)
    ):
        raise AttentionOutputCrystalIntegrityError(
            "attention-output crystal destination must be a regular file"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("zero-byte state write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1

        parent_after = os.lstat(parent)
        if not _same_inode(parent_before, parent_after):
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal parent changed before publication"
            )
        try:
            current = os.lstat(path)
        except FileNotFoundError:
            current = None
        if current is not None and (
            stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode)
        ):
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal destination changed type"
            )
        os.replace(temporary, path)
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_DIRECTORY", 0
        )
        directory = os.open(parent, directory_flags)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        published = os.lstat(path)
        if not stat.S_ISREG(published.st_mode):
            raise AttentionOutputCrystalIntegrityError(
                "published attention-output crystal state is not regular"
            )
        return _stable_signature(published)
    except AttentionOutputCrystalError:
        raise
    except OSError as exc:
        raise AttentionOutputCrystalIntegrityError(
            "attention-output crystal state could not be published atomically"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True, slots=True)
class AttentionOutputCrystalHit:
    """One exact transition reconstructed from canonical stored bytes."""

    cell_sha256: str
    key: AttentionOutputCrystalKey
    post_o_proj: torch.Tensor
    appended_rope_key: torch.Tensor
    appended_value: torch.Tensor
    next_crsa_usage: torch.Tensor | None
    crsa_evidence: Mapping[str, Any] | None
    support: int
    cell_hit_count: int
    logical_projection_bytes: int
    skipped_projection_calls: int
    payload_sha256: str

    @property
    def attention_output(self) -> torch.Tensor:
        """Integration alias for the exact post-``o_proj`` row."""

        return self.post_o_proj

    @property
    def appended_key(self) -> torch.Tensor:
        """Integration alias for the appended RoPE key row."""

        return self.appended_rope_key

    @property
    def next_crsa_evidence(self) -> Mapping[str, Any] | None:
        return self.crsa_evidence

    @property
    def absolute_position(self) -> int:
        return self.key.absolute_position

    @property
    def layer_index(self) -> int:
        return self.key.layer_index

    def to_dict(self) -> dict[str, object]:
        return {
            "absolute_position": self.absolute_position,
            "cell_hit_count": self.cell_hit_count,
            "cell_sha256": self.cell_sha256,
            "has_crsa_evidence": self.crsa_evidence is not None,
            "has_crsa_usage": self.next_crsa_usage is not None,
            "key_sha256": self.key.key_sha256,
            "layer_index": self.layer_index,
            "logical_projection_bytes": self.logical_projection_bytes,
            "payload_sha256": self.payload_sha256,
            "skipped_projection_calls": self.skipped_projection_calls,
            "support": self.support,
        }


@dataclass(frozen=True, slots=True)
class AttentionOutputCrystalMetrics:
    identity_sha256: str
    state_sha256: str
    max_cells: int
    max_state_bytes: int
    persisted_bytes: int
    cell_count: int
    clock: int
    publish_transactions: int
    staged_captures: int
    accepted_captures: int
    rejected_captures: int
    hit_count: int
    evictions: int
    support: int
    stored_tensor_bytes: int
    stored_logical_projection_bytes: int
    skipped_projection_calls_saved: int
    logical_projection_bytes_saved: int

    @property
    def hits(self) -> int:
        return self.hit_count

    @property
    def logical_saved_projection_bytes(self) -> int:
        return self.logical_projection_bytes_saved

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted_captures": self.accepted_captures,
            "cell_count": self.cell_count,
            "clock": self.clock,
            "evictions": self.evictions,
            "hit_count": self.hit_count,
            "identity_sha256": self.identity_sha256,
            "logical_projection_bytes_saved": (
                self.logical_projection_bytes_saved
            ),
            "max_cells": self.max_cells,
            "max_state_bytes": self.max_state_bytes,
            "persisted_bytes": self.persisted_bytes,
            "publish_transactions": self.publish_transactions,
            "rejected_captures": self.rejected_captures,
            "skipped_projection_calls_saved": (
                self.skipped_projection_calls_saved
            ),
            "staged_captures": self.staged_captures,
            "state_sha256": self.state_sha256,
            "stored_logical_projection_bytes": (
                self.stored_logical_projection_bytes
            ),
            "stored_tensor_bytes": self.stored_tensor_bytes,
            "support": self.support,
        }


@dataclass(frozen=True, slots=True)
class _StagedHit:
    position: int
    key_sha256: str
    cell_sha256: str
    skipped_projection_calls: int
    logical_projection_bytes_saved: int

    def __post_init__(self) -> None:
        _uint(
            self.position,
            field="hit position",
            maximum=_MAX_ABSOLUTE_POSITION,
        )
        _digest(self.key_sha256, "hit key_sha256")
        _digest(self.cell_sha256, "hit cell_sha256")
        _uint(
            self.skipped_projection_calls,
            field="hit skipped_projection_calls",
        )
        _uint(
            self.logical_projection_bytes_saved,
            field="hit logical_projection_bytes_saved",
        )


@dataclass(frozen=True, slots=True)
class _StagedSavings:
    position: int
    skipped_projection_calls: int
    logical_projection_bytes_saved: int

    def __post_init__(self) -> None:
        _uint(
            self.position,
            field="savings position",
            maximum=_MAX_ABSOLUTE_POSITION,
        )
        calls = _uint(
            self.skipped_projection_calls,
            field="savings skipped_projection_calls",
        )
        logical = _uint(
            self.logical_projection_bytes_saved,
            field="savings logical_projection_bytes_saved",
        )
        if calls == 0 and logical == 0:
            raise ValueError("a savings event must record calls or logical bytes")


class AttentionOutputCrystalTransaction:
    """Request-local speculative owner for captures, hits, and economics.

    ``accepted_end_position`` is an exclusive absolute token boundary.  Only
    events whose position is below that boundary are committed.  In
    particular, hit and savings counters for rejected speculative rows are
    never published.
    """

    def __init__(self, bank: "AttentionOutputCrystalBank") -> None:
        self._bank = bank
        self._captures: list[tuple[int, StagedAttentionOutputCrystal]] = []
        self._hits: list[_StagedHit] = []
        self._savings: list[_StagedSavings] = []
        self._closed = False
        self._lock = threading.RLock()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise AttentionOutputCrystalError(
                "attention-output crystal transaction is closed"
            )

    def stage_capture(
        self,
        position: int,
        cell: StagedAttentionOutputCrystal,
    ) -> StagedAttentionOutputCrystal:
        with self._lock:
            self._require_open()
            position = _uint(
                position,
                field="capture position",
                maximum=_MAX_ABSOLUTE_POSITION,
            )
            if not isinstance(cell, StagedAttentionOutputCrystal):
                raise TypeError("cell must be a StagedAttentionOutputCrystal")
            self._bank._validate_staged(cell)
            if position != cell.key.absolute_position:
                raise ValueError("capture position disagrees with its exact key")
            self._captures.append((position, cell))
            return cell

    def stage_hit(
        self,
        position: int,
        hit_or_key: AttentionOutputCrystalHit | AttentionOutputCrystalKey,
        *,
        skipped_projection_calls: int = 0,
        logical_projection_bytes_saved: int = 0,
    ) -> AttentionOutputCrystalHit:
        with self._lock:
            self._require_open()
            position = _uint(
                position,
                field="hit position",
                maximum=_MAX_ABSOLUTE_POSITION,
            )
            if isinstance(hit_or_key, AttentionOutputCrystalKey):
                hit = self._bank.peek(hit_or_key)
                if hit is None:
                    raise AttentionOutputCrystalIntegrityError(
                        "cannot stage an unknown attention-output crystal hit"
                    )
            elif isinstance(hit_or_key, AttentionOutputCrystalHit):
                hit = hit_or_key
                self._bank._validate_hit(hit)
            else:
                raise TypeError(
                    "hit_or_key must be an AttentionOutputCrystalHit or key"
                )
            if position != hit.key.absolute_position:
                raise ValueError("hit position disagrees with its exact key")
            if any(
                previous.position == position
                and previous.key_sha256 == hit.key.key_sha256
                for previous in self._hits
            ):
                raise ValueError("one exact attention-output hit was staged twice")
            event = _StagedHit(
                position=position,
                key_sha256=hit.key.key_sha256,
                cell_sha256=hit.cell_sha256,
                skipped_projection_calls=skipped_projection_calls,
                logical_projection_bytes_saved=logical_projection_bytes_saved,
            )
            self._hits.append(event)
            return hit

    def lookup(
        self,
        key: AttentionOutputCrystalKey,
        *,
        device: torch.device | str | None = None,
        skipped_projection_calls: int = 0,
        logical_projection_bytes_saved: int = 0,
    ) -> AttentionOutputCrystalHit | None:
        """Read one exact cell and stage its hit only when it exists."""

        with self._lock:
            self._require_open()
            hit = self._bank.peek(key, device=device)
            if hit is None:
                return None
            self.stage_hit(
                key.absolute_position,
                hit,
                skipped_projection_calls=skipped_projection_calls,
                logical_projection_bytes_saved=logical_projection_bytes_saved,
            )
            return hit

    def stage_savings(
        self,
        position: int,
        *,
        skipped_projection_calls: int,
        logical_projection_bytes_saved: int,
    ) -> None:
        """Stage one caller-measured all-hit layer-block savings event."""

        with self._lock:
            self._require_open()
            self._savings.append(
                _StagedSavings(
                    position=position,
                    skipped_projection_calls=skipped_projection_calls,
                    logical_projection_bytes_saved=logical_projection_bytes_saved,
                )
            )

    def commit(
        self,
        *,
        accepted_end_position: int,
    ) -> AttentionOutputCrystalMetrics:
        with self._lock:
            self._require_open()
            boundary = _uint(
                accepted_end_position,
                field="accepted_end_position",
                maximum=_MAX_ABSOLUTE_POSITION + 1,
            )
            captures = tuple(
                cell for position, cell in self._captures if position < boundary
            )
            rejected_captures = len(self._captures) - len(captures)
            hits = tuple(hit for hit in self._hits if hit.position < boundary)
            savings = tuple(
                event for event in self._savings if event.position < boundary
            )
            metrics = self._bank._commit(
                captures=captures,
                staged_capture_count=len(self._captures),
                rejected_capture_count=rejected_captures,
                hits=hits,
                savings=savings,
            )
            self._closed = True
            return metrics

    def rollback(self) -> None:
        with self._lock:
            self._require_open()
            self._captures.clear()
            self._hits.clear()
            self._savings.clear()
            self._closed = True

    def __enter__(self) -> "AttentionOutputCrystalTransaction":
        with self._lock:
            self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        with self._lock:
            if not self._closed:
                self.rollback()


class AttentionOutputCrystalBank:
    """Bounded exact transition bank with atomic speculative publication."""

    def __init__(
        self,
        state_path: str | os.PathLike[str],
        identity: AttentionOutputCrystalIdentity,
        *,
        max_cells: int = 4096,
        max_state_bytes: int = _DEFAULT_MAX_STATE_BYTES,
    ) -> None:
        if not isinstance(identity, AttentionOutputCrystalIdentity):
            raise TypeError("identity must be an AttentionOutputCrystalIdentity")
        self.state_path = Path(state_path)
        if not self.state_path.name:
            raise ValueError("state_path must name a file")
        _uint(max_cells, field="max_cells", positive=True, maximum=_MAX_CELLS)
        _uint(
            max_state_bytes,
            field="max_state_bytes",
            positive=True,
            maximum=_ABSOLUTE_MAX_STATE_BYTES,
        )
        self.identity = identity
        self.max_cells = max_cells
        self.max_state_bytes = max_state_bytes
        self._lock = _thread_lock(self.state_path)
        self._file_signature: tuple[int, int, int, int, int] | None = None
        self._state = _AttentionState(
            identity=identity,
            max_cells=max_cells,
            max_state_bytes=max_state_bytes,
        )
        self._index: dict[str, _AttentionCell] = {}
        with self._lock:
            self._reload(required=False)

    def _validate_state(self, state: _AttentionState) -> None:
        if state.identity.identity_sha256 != self.identity.identity_sha256:
            raise AttentionOutputCrystalIdentityError(
                "attention-output bank belongs to a different runtime identity"
            )
        if state.max_cells != self.max_cells:
            raise AttentionOutputCrystalIdentityError(
                "attention-output bank max_cells differs from persistent identity"
            )
        if state.max_state_bytes != self.max_state_bytes:
            raise AttentionOutputCrystalIdentityError(
                "attention-output bank max_state_bytes differs from persistent identity"
            )

    def _reload(self, *, required: bool) -> None:
        try:
            raw, signature = _stable_regular_bytes(
                self.state_path,
                max_bytes=min(
                    self.max_state_bytes,
                    _ABSOLUTE_MAX_STATE_BYTES,
                ),
            )
        except FileNotFoundError:
            if required:
                raise AttentionOutputCrystalIntegrityError(
                    "attention-output crystal state disappeared"
                )
            self._file_signature = None
            self._state = _AttentionState(
                identity=self.identity,
                max_cells=self.max_cells,
                max_state_bytes=self.max_state_bytes,
            )
            self._rebuild_index()
            return
        state = _AttentionState.from_bytes(raw)
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
                raise AttentionOutputCrystalIntegrityError(
                    "attention-output crystal state disappeared"
                )
            return
        if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
            raise AttentionOutputCrystalIntegrityError(
                "attention-output crystal state must be a regular file"
            )
        if _stable_signature(linked) == self._file_signature:
            return
        last_error: AttentionOutputCrystalIntegrityError | None = None
        for _ in range(3):
            try:
                self._reload(required=True)
                return
            except AttentionOutputCrystalIntegrityError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def _validate_key(self, key: AttentionOutputCrystalKey) -> None:
        if not isinstance(key, AttentionOutputCrystalKey):
            raise TypeError("key must be an AttentionOutputCrystalKey")
        if (
            key.identity_sha256 != self.identity.identity_sha256
            or key.runtime_math_sha256 != self.identity.runtime_math_sha256
        ):
            raise AttentionOutputCrystalIdentityError(
                "attention-output key belongs to a different runtime identity"
            )

    def _validate_staged(self, staged: StagedAttentionOutputCrystal) -> None:
        if not isinstance(staged, StagedAttentionOutputCrystal):
            raise TypeError("staged value must be a StagedAttentionOutputCrystal")
        self._validate_key(staged.key)

    def _validate_hit(self, hit: AttentionOutputCrystalHit) -> None:
        if not isinstance(hit, AttentionOutputCrystalHit):
            raise TypeError("hit must be an AttentionOutputCrystalHit")
        self._validate_key(hit.key)
        with self._lock:
            self._refresh_if_changed()
            current = self._index.get(hit.key.key_sha256)
            if current is None or (
                current.cell_sha256 != hit.cell_sha256
                or current.payload.payload_sha256 != hit.payload_sha256
                or current.logical_projection_bytes
                != hit.logical_projection_bytes
                or current.skipped_projection_calls
                != hit.skipped_projection_calls
            ):
                raise AttentionOutputCrystalIntegrityError(
                    "cannot stage a stale or forged attention-output hit"
                )

    def make_key(
        self,
        layer_index: int,
        absolute_position: int,
        attention_input: torch.Tensor,
        prior_state_sha256: str,
    ) -> AttentionOutputCrystalKey:
        """Seal one exact BF16 attention boundary into a lookup key."""

        _uint(layer_index, field="layer_index", maximum=_MAX_LAYER_INDEX)
        _uint(
            absolute_position,
            field="absolute_position",
            maximum=_MAX_ABSOLUTE_POSITION,
        )
        _digest(prior_state_sha256, "prior_state_sha256")
        if not isinstance(attention_input, torch.Tensor):
            raise TypeError("attention_input must be a torch.Tensor")
        if (
            attention_input.ndim != 3
            or tuple(attention_input.shape[:2]) != (1, 1)
        ):
            raise ValueError(
                "attention_input must have exact shape [1, 1, hidden]"
            )
        return AttentionOutputCrystalKey.create(
            identity=self.identity,
            layer_index=layer_index,
            absolute_position=absolute_position,
            attention_input_sha256=canonical_bf16_tensor_sha256(attention_input),
            prior_state_sha256=prior_state_sha256,
        )

    def stage(
        self,
        key: AttentionOutputCrystalKey,
        *,
        post_o_proj: torch.Tensor,
        appended_rope_key: torch.Tensor,
        appended_value: torch.Tensor,
        logical_projection_bytes: int,
        skipped_projection_calls: int = 4,
        next_crsa_usage: torch.Tensor | None = None,
        crsa_evidence: object | None = None,
    ) -> StagedAttentionOutputCrystal:
        """Detach one result into an immutable staged exact transition."""

        self._validate_key(key)
        logical_projection_bytes = _uint(
            logical_projection_bytes,
            field="logical_projection_bytes",
            positive=True,
        )
        skipped_projection_calls = _uint(
            skipped_projection_calls,
            field="skipped_projection_calls",
            positive=True,
        )
        payload = _AttentionPayload.capture(
            post_o_proj=post_o_proj,
            appended_rope_key=appended_rope_key,
            appended_value=appended_value,
            next_crsa_usage=next_crsa_usage,
            crsa_evidence=crsa_evidence,
        )
        address = {
            "key_sha256": key.key_sha256,
            "logical_projection_bytes": logical_projection_bytes,
            "payload_sha256": payload.payload_sha256,
            "schema": ATTENTION_OUTPUT_CRYSTAL_CELL_SCHEMA,
            "skipped_projection_calls": skipped_projection_calls,
        }
        return StagedAttentionOutputCrystal(
            key=key,
            payload=payload,
            logical_projection_bytes=logical_projection_bytes,
            skipped_projection_calls=skipped_projection_calls,
            cell_sha256=_sha256_document(address),
        )

    def begin_transaction(self) -> AttentionOutputCrystalTransaction:
        """Begin one request-local owner spanning initial and extended blocks."""

        return AttentionOutputCrystalTransaction(self)

    @staticmethod
    def _hit_from_cell(
        cell: _AttentionCell,
        *,
        device: torch.device | str | None,
    ) -> AttentionOutputCrystalHit:
        evidence: Mapping[str, Any] | None = None
        if cell.payload.crsa_evidence_json is not None:
            decoded = _decode_canonical_json(
                cell.payload.crsa_evidence_json,
                label="CRSA evidence",
            )
            assert isinstance(decoded, Mapping)
            evidence = decoded
        return AttentionOutputCrystalHit(
            cell_sha256=cell.cell_sha256,
            key=cell.key,
            post_o_proj=cell.payload.post_o_proj.tensor(device=device),
            appended_rope_key=cell.payload.appended_rope_key.tensor(device=device),
            appended_value=cell.payload.appended_value.tensor(device=device),
            next_crsa_usage=(
                None
                if cell.payload.next_crsa_usage is None
                else cell.payload.next_crsa_usage.tensor(device=device)
            ),
            crsa_evidence=evidence,
            support=cell.support,
            cell_hit_count=cell.hit_count,
            logical_projection_bytes=cell.logical_projection_bytes,
            skipped_projection_calls=cell.skipped_projection_calls,
            payload_sha256=cell.payload.payload_sha256,
        )

    def peek(
        self,
        key: AttentionOutputCrystalKey,
        *,
        device: torch.device | str | None = None,
    ) -> AttentionOutputCrystalHit | None:
        """Read an exact cell without publishing hit or savings counters."""

        self._validate_key(key)
        with self._lock:
            self._refresh_if_changed()
            cell = self._index.get(key.key_sha256)
            if cell is None:
                return None
            if cell.key != key:
                raise AttentionOutputCrystalIntegrityError(
                    "one key address names two attention-output transitions"
                )
            return self._hit_from_cell(cell, device=device)

    def lookup(
        self,
        key: AttentionOutputCrystalKey,
        *,
        device: torch.device | str | None = None,
    ) -> AttentionOutputCrystalHit | None:
        """Alias for read-only :meth:`peek`; transactions account accepted hits."""

        return self.peek(key, device=device)

    def publish(
        self,
        staged_sequence: Sequence[StagedAttentionOutputCrystal],
        *,
        accepted_rows: int,
    ) -> AttentionOutputCrystalMetrics:
        """Atomically publish only the accepted prefix of staged row cells."""

        if isinstance(staged_sequence, (str, bytes, bytearray)) or not isinstance(
            staged_sequence, Sequence
        ):
            raise TypeError("staged_sequence must be a sequence")
        staged = tuple(staged_sequence)
        if any(not isinstance(item, StagedAttentionOutputCrystal) for item in staged):
            raise TypeError(
                "staged_sequence contains a non-StagedAttentionOutputCrystal value"
            )
        for item in staged:
            self._validate_staged(item)
        positions = tuple(item.key.absolute_position for item in staged)
        if any(
            left >= right for left, right in zip(positions, positions[1:])
        ):
            raise ValueError(
                "staged_sequence must contain strictly increasing row positions"
            )
        accepted_rows = _uint(
            accepted_rows,
            field="accepted_rows",
            maximum=len(staged),
        )
        return self._commit(
            captures=staged[:accepted_rows],
            staged_capture_count=len(staged),
            rejected_capture_count=len(staged) - accepted_rows,
            hits=(),
            savings=(),
        )

    @staticmethod
    def _victim(cells: Mapping[str, _AttentionCell]) -> _AttentionCell:
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
        captures: Sequence[StagedAttentionOutputCrystal],
        staged_capture_count: int,
        rejected_capture_count: int,
        hits: Sequence[_StagedHit],
        savings: Sequence[_StagedSavings],
    ) -> AttentionOutputCrystalMetrics:
        captures = tuple(captures)
        hits = tuple(hits)
        savings = tuple(savings)
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
        if any(not isinstance(event, _StagedSavings) for event in savings):
            raise TypeError("savings contains an invalid staged savings event")
        if not staged_capture_count and not hits and not savings:
            return self.metrics()

        with _exclusive_state_lock(self.state_path):
            self._reload(required=self._file_signature is not None)
            state = self._state
            clock = _bounded_add(state.clock, 1)
            cells = {cell.key.key_sha256: cell for cell in state.cells}

            committed_calls = sum(
                event.skipped_projection_calls for event in (*hits, *savings)
            )
            committed_bytes = sum(
                event.logical_projection_bytes_saved
                for event in (*hits, *savings)
            )
            for event in hits:
                cell = cells.get(event.key_sha256)
                if cell is None:
                    # The replay already happened.  A concurrent deterministic
                    # eviction cannot erase its request economics.
                    continue
                if cell.cell_sha256 != event.cell_sha256:
                    raise AttentionOutputCrystalIntegrityError(
                        "staged hit cell changed under its exact key"
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
                    or previous.payload != incoming.payload
                    or previous.logical_projection_bytes
                    != incoming.logical_projection_bytes
                    or previous.skipped_projection_calls
                    != incoming.skipped_projection_calls
                ):
                    raise AttentionOutputCrystalIntegrityError(
                        "one exact attention transition produced conflicting payloads"
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

            def make_state() -> _AttentionState:
                return _AttentionState(
                    identity=self.identity,
                    max_cells=self.max_cells,
                    max_state_bytes=self.max_state_bytes,
                    clock=clock,
                    publish_transactions=_bounded_add(
                        state.publish_transactions,
                        1,
                    ),
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
                    evictions=_bounded_add(state.evictions, evicted),
                    skipped_projection_calls_saved=_bounded_add(
                        state.skipped_projection_calls_saved,
                        committed_calls,
                    ),
                    logical_projection_bytes_saved=_bounded_add(
                        state.logical_projection_bytes_saved,
                        committed_bytes,
                    ),
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
                    raise AttentionOutputCrystalIntegrityError(
                        "attention-output crystal capacity cannot hold empty state"
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

    def refresh(self) -> AttentionOutputCrystalMetrics:
        """Read a newer complete atomic generation when one exists."""

        with self._lock:
            self._refresh_if_changed()
            return self._metrics(self._state)

    def metrics(self) -> AttentionOutputCrystalMetrics:
        with self._lock:
            self._refresh_if_changed()
            return self._metrics(self._state)

    def _metrics(self, state: _AttentionState) -> AttentionOutputCrystalMetrics:
        cells = state.cells
        return AttentionOutputCrystalMetrics(
            identity_sha256=state.identity.identity_sha256,
            state_sha256=_sha256_document(state.to_record()),
            max_cells=state.max_cells,
            max_state_bytes=state.max_state_bytes,
            persisted_bytes=(
                0 if self._file_signature is None else self._file_signature[2]
            ),
            cell_count=len(cells),
            clock=state.clock,
            publish_transactions=state.publish_transactions,
            staged_captures=state.staged_captures,
            accepted_captures=state.accepted_captures,
            rejected_captures=state.rejected_captures,
            hit_count=state.hit_count,
            evictions=state.evictions,
            support=sum(cell.support for cell in cells),
            stored_tensor_bytes=sum(cell.tensor_bytes for cell in cells),
            stored_logical_projection_bytes=sum(
                cell.logical_projection_bytes for cell in cells
            ),
            skipped_projection_calls_saved=(
                state.skipped_projection_calls_saved
            ),
            logical_projection_bytes_saved=(
                state.logical_projection_bytes_saved
            ),
        )


__all__ = [
    "ATTENTION_OUTPUT_CRYSTAL_ATTENTION_STATE_SCHEMA",
    "ATTENTION_OUTPUT_CRYSTAL_CELL_SCHEMA",
    "ATTENTION_OUTPUT_CRYSTAL_ENVELOPE_SCHEMA",
    "ATTENTION_OUTPUT_CRYSTAL_EVIDENCE_SCHEMA",
    "ATTENTION_OUTPUT_CRYSTAL_IDENTITY_SCHEMA",
    "ATTENTION_OUTPUT_CRYSTAL_KEY_SCHEMA",
    "ATTENTION_OUTPUT_CRYSTAL_PAYLOAD_SCHEMA",
    "ATTENTION_OUTPUT_CRYSTAL_STATE_SCHEMA",
    "ATTENTION_OUTPUT_CRYSTAL_TENSOR_ABI",
    "ATTENTION_OUTPUT_CRYSTAL_TENSOR_SCHEMA",
    "EMPTY_ATTENTION_STATE_SHA256",
    "AttentionOutputCrystalBank",
    "AttentionOutputCrystalError",
    "AttentionOutputCrystalHit",
    "AttentionOutputCrystalIdentity",
    "AttentionOutputCrystalIdentityError",
    "AttentionOutputCrystalIntegrityError",
    "AttentionOutputCrystalKey",
    "AttentionOutputCrystalMetrics",
    "AttentionOutputCrystalStage",
    "AttentionOutputCrystalTransaction",
    "StagedAttentionOutputCrystal",
    "canonical_attention_state_sha256",
    "canonical_bf16_tensor_bytes",
    "canonical_bf16_tensor_sha256",
]
