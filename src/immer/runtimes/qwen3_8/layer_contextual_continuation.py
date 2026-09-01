"""Persistent layer-scoped, target-confirmed continuation bank (v2).

The bank stores only a layer number, one known token, a contiguous signed-Q8
sketch, and a target-confirmed continuation.  Hidden states and prompt text are
never part of the persistent format.  Multi-layer captures and verifier
feedback are published in one atomic, idempotent transaction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import lru_cache
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import tempfile
import threading
from types import MappingProxyType
from typing import Any, Iterator

import torch


LAYER_CONTEXTUAL_CONTINUATION_IDENTITY_SCHEMA = (
    "immer.qwen3.8-layer-contextual-continuation-identity/v2"
)
LAYER_CONTEXTUAL_CONTINUATION_CELL_SCHEMA = (
    "immer.qwen3.8-layer-contextual-continuation-cell/v2"
)
LAYER_CONTEXTUAL_CONTINUATION_RECEIPT_SCHEMA = (
    "immer.qwen3.8-layer-contextual-continuation-receipt/v2"
)
LAYER_CONTEXTUAL_CONTINUATION_STATE_SCHEMA = (
    "immer.qwen3.8-layer-contextual-continuation-bank-state/v2"
)
LAYER_CONTEXTUAL_CONTINUATION_ENVELOPE_SCHEMA = (
    "immer.qwen3.8-layer-contextual-continuation-bank-envelope/v2"
)
LAYER_CONTEXTUAL_CONTINUATION_TRANSACTION_SCHEMA = (
    "immer.qwen3.8-layer-contextual-continuation-transaction/v2"
)
LAYER_CONTEXTUAL_CONTINUATION_METRICS_SCHEMA = (
    "immer.qwen3.8-layer-contextual-continuation-metrics/v2"
)
LAYER_CONTEXTUAL_PROJECTION_ABI = (
    "immer.qwen3.8/layer-output-rademacher-shake256-lsb-f64-l2-q8/v2"
)
LAYER_CONTEXTUAL_STAGE = "layer.output"
DEFAULT_LAYER_CONTEXTUAL_SKETCH_DIM = 128
DEFAULT_LAYER_CONTEXTUAL_MAX_RECEIPTS = 65_536
MAX_CONTINUATION_TOKENS = 15

_MAX_COUNTER = (1 << 63) - 1
_MAX_TOKEN_ID = (1 << 32) - 1
_MAX_BOUNDARY_INDEX = (1 << 63) - 1
_MAX_LAYER_INDEX = (1 << 16) - 1
_MAX_LAYERS = 1024
_MAX_HIDDEN_DIM = 1 << 16
_MAX_SKETCH_DIM = 4096
_MAX_PROJECTION_ELEMENTS = 1 << 22
_MAX_CELLS = 1 << 16
_MAX_RECEIPTS = 1 << 16
_MAX_STATE_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_HEX = frozenset("0123456789abcdef")


class LayerContextualContinuationError(RuntimeError):
    """A layer-continuation input or persistent state is invalid."""


class LayerContextualContinuationIntegrityError(LayerContextualContinuationError):
    """Persistent state is malformed, unstable, or hash-inconsistent."""


class LayerContextualContinuationIdentityError(LayerContextualContinuationError):
    """A transaction or state belongs to another immutable identity."""


class LayerContextualContinuationConflictError(LayerContextualContinuationError):
    """An already-settled transaction was presented with another outcome."""


class LayerContextualContinuationCapacityError(
    LayerContextualContinuationIntegrityError
):
    """A new settlement cannot retain its mandatory idempotence receipt."""


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
        raise LayerContextualContinuationIntegrityError(
            "layer continuation state is not canonical JSON"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_document(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - _HEX)


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


def _token(value: object, field: str = "known_token") -> int:
    return _uint(value, field=field, maximum=_MAX_TOKEN_ID)


def _layer(value: object, field: str = "layer") -> int:
    return _uint(value, field=field, maximum=_MAX_LAYER_INDEX)


def _tail(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("verified_prefix must be a sequence of token IDs")
    result = tuple(value)
    if not 1 <= len(result) <= MAX_CONTINUATION_TOKENS:
        raise ValueError(
            f"verified_prefix must contain 1..{MAX_CONTINUATION_TOKENS} tokens"
        )
    for token_id in result:
        _token(token_id, "continuation token")
    return result


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LayerContextualContinuationIntegrityError(
                f"duplicate JSON key: {key!r}"
            )
        result[key] = value
    return result


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


def _stable_regular_bytes(path: Path) -> tuple[bytes, tuple[int, int, int, int, int]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise LayerContextualContinuationIntegrityError(
            "layer continuation state cannot be opened"
        ) from exc
    try:
        before = os.fstat(descriptor)
        linked_before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(linked_before.st_mode)
            or not _same_inode(before, linked_before)
        ):
            raise LayerContextualContinuationIntegrityError(
                "layer continuation state must be a stable regular file"
            )
        if before.st_size < 0 or before.st_size > _MAX_STATE_BYTES:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation state exceeds its byte limit"
            )
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
            if not chunk:
                raise LayerContextualContinuationIntegrityError(
                    "layer continuation state was truncated"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise LayerContextualContinuationIntegrityError(
                "layer continuation state grew while reading"
            )
        after = os.fstat(descriptor)
        linked_after = os.lstat(path)
        if (
            _stable_signature(before) != _stable_signature(after)
            or _stable_signature(linked_before) != _stable_signature(linked_after)
            or not _same_inode(after, linked_after)
        ):
            raise LayerContextualContinuationIntegrityError(
                "layer continuation state changed while reading"
            )
        return b"".join(chunks), _stable_signature(after)
    except OSError as exc:
        raise LayerContextualContinuationIntegrityError(
            "layer continuation state cannot be read safely"
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
        raise LayerContextualContinuationIntegrityError(
            "layer continuation state parent is unavailable"
        ) from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise LayerContextualContinuationIntegrityError(
            "layer continuation state parent must be a directory"
        )
    return parent, value


@contextmanager
def _exclusive_state_lock(path: Path) -> Iterator[None]:
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
            raise LayerContextualContinuationIntegrityError(
                "layer continuation lock is unavailable"
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
                raise LayerContextualContinuationIntegrityError(
                    "layer continuation lock is not a stable regular file"
                )
            yield
            parent_after = os.lstat(parent)
            linked_after = os.lstat(lock_path)
            if not _same_inode(opened, linked_after) or not _same_inode(
                parent_before, parent_after
            ):
                raise LayerContextualContinuationIntegrityError(
                    "layer continuation lock changed during transaction"
                )
        except OSError as exc:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation lock failed"
            ) from exc
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> tuple[int, int, int, int, int]:
    if len(data) > _MAX_STATE_BYTES:
        raise LayerContextualContinuationIntegrityError(
            "layer continuation state exceeds its byte limit"
        )
    parent, parent_before = _validate_parent(path)
    try:
        destination = os.lstat(path)
    except FileNotFoundError:
        destination = None
    except OSError as exc:
        raise LayerContextualContinuationIntegrityError(
            "layer continuation destination is unavailable"
        ) from exc
    if destination is not None and (
        stat.S_ISLNK(destination.st_mode) or not stat.S_ISREG(destination.st_mode)
    ):
        raise LayerContextualContinuationIntegrityError(
            "layer continuation destination must be a regular file"
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
        if not _same_inode(parent_before, os.lstat(parent)):
            raise LayerContextualContinuationIntegrityError(
                "layer continuation parent changed before publication"
            )
        try:
            current = os.lstat(path)
        except FileNotFoundError:
            current = None
        if current is not None and (
            stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode)
        ):
            raise LayerContextualContinuationIntegrityError(
                "layer continuation destination changed type"
            )
        os.replace(temporary, path)
        directory_flags = (
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
        )
        directory = os.open(parent, directory_flags)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        published = os.lstat(path)
        if not stat.S_ISREG(published.st_mode):
            raise LayerContextualContinuationIntegrityError(
                "published layer continuation state is not regular"
            )
        return _stable_signature(published)
    except LayerContextualContinuationError:
        raise
    except OSError as exc:
        raise LayerContextualContinuationIntegrityError(
            "layer continuation state could not be published atomically"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _q8_values(value: object, *, sketch_dim: int | None = None) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("Q8 vector must be on CPU")
        if value.dtype != torch.int8:
            raise TypeError("Q8 vector must have torch.int8 dtype")
        if value.ndim != 1:
            raise ValueError("Q8 vector must be one-dimensional")
        if not value.is_contiguous():
            raise ValueError("Q8 vector must be contiguous")
        result = tuple(int(item) for item in value.tolist())
    elif isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("Q8 vector must be a contiguous int8 tensor or sequence")
    else:
        result = tuple(value)
    if sketch_dim is not None and len(result) != sketch_dim:
        raise ValueError(f"Q8 vector must have {sketch_dim} dimensions")
    if not result:
        raise ValueError("Q8 vector must not be empty")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or not -127 <= item <= 127
        for item in result
    ):
        raise ValueError("Q8 vector contains a value outside [-127, 127]")
    norm_sq = sum(item * item for item in result)
    tolerance = math.ceil(math.sqrt(len(result)))
    lower = max(1, 127 - tolerance)
    upper = 127 + tolerance
    if not lower * lower <= norm_sq <= upper * upper:
        raise ValueError("Q8 vector is not a normalised signed-Q8 sketch")
    return result


class LayerContextualContinuationKey:
    """A key containing only ``(layer, known_token, contiguous int8 Q8)``."""

    __slots__ = ("_q8_bytes", "known_token", "layer")

    def __init__(
        self,
        layer: int,
        known_token: int,
        q8: torch.Tensor | Sequence[int],
    ) -> None:
        self.layer = _layer(layer)
        self.known_token = _token(known_token)
        values = _q8_values(q8)
        self._q8_bytes = bytes(value & 0xFF for value in values)

    @property
    def q8(self) -> torch.Tensor:
        """Return an isolated, contiguous CPU int8 vector."""

        return torch.frombuffer(
            bytearray(self._q8_bytes), dtype=torch.int8
        ).contiguous()

    @property
    def q8_values(self) -> tuple[int, ...]:
        return tuple(value if value < 128 else value - 256 for value in self._q8_bytes)

    @property
    def sketch_dim(self) -> int:
        return len(self._q8_bytes)

    @property
    def norm_sq(self) -> int:
        return sum(value * value for value in self.q8_values)

    @property
    def key_sha256(self) -> str:
        return _sha256_document(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "known_token": self.known_token,
            "layer": self.layer,
            "q8": list(self.q8_values),
        }

    @classmethod
    def from_record(
        cls,
        value: object,
        *,
        sketch_dim: int,
    ) -> LayerContextualContinuationKey:
        if not isinstance(value, Mapping) or set(value) != {
            "known_token",
            "layer",
            "q8",
        }:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation key fields are invalid"
            )
        try:
            values = _q8_values(value["q8"], sketch_dim=sketch_dim)
            return cls(value["layer"], value["known_token"], values)
        except (TypeError, ValueError) as exc:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation key values are invalid"
            ) from exc

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, LayerContextualContinuationKey)
            and self.layer == other.layer
            and self.known_token == other.known_token
            and self._q8_bytes == other._q8_bytes
        )

    def __hash__(self) -> int:
        return hash((self.layer, self.known_token, self._q8_bytes))

    def __repr__(self) -> str:
        return (
            "LayerContextualContinuationKey("
            f"layer={self.layer}, known_token={self.known_token}, "
            f"q8=<int8[{self.sketch_dim}]>)"
        )


@dataclass(frozen=True, slots=True)
class LayerContextualContinuationIdentity:
    """Immutable authority and projection identity for one layer bank."""

    runtime_sha256: str
    model_sha256: str
    q4_sha256: str
    tokenizer_sha256: str
    hidden_dim: int
    layers: tuple[int, ...]
    projection_seed: int = 0
    sketch_dim: int = DEFAULT_LAYER_CONTEXTUAL_SKETCH_DIM
    stage: str = LAYER_CONTEXTUAL_STAGE
    projection_abi: str = LAYER_CONTEXTUAL_PROJECTION_ABI

    def __post_init__(self) -> None:
        for field in (
            "runtime_sha256",
            "model_sha256",
            "q4_sha256",
            "tokenizer_sha256",
        ):
            _digest(getattr(self, field), field)
        _uint(
            self.hidden_dim,
            field="hidden_dim",
            positive=True,
            maximum=_MAX_HIDDEN_DIM,
        )
        _uint(
            self.sketch_dim,
            field="sketch_dim",
            positive=True,
            maximum=_MAX_SKETCH_DIM,
        )
        if self.hidden_dim * self.sketch_dim > _MAX_PROJECTION_ELEMENTS:
            raise ValueError("projection exceeds its element limit")
        _uint(self.projection_seed, field="projection_seed", maximum=(1 << 64) - 1)
        if isinstance(self.layers, (str, bytes, bytearray)) or not isinstance(
            self.layers, Sequence
        ):
            raise TypeError("layers must be a sequence of layer indices")
        raw_layers = tuple(self.layers)
        if not raw_layers or len(raw_layers) > _MAX_LAYERS:
            raise ValueError("layers must contain a bounded non-empty layer set")
        for layer_index in raw_layers:
            _layer(layer_index, "identity layer")
        if len(set(raw_layers)) != len(raw_layers):
            raise ValueError("layers contains a duplicate layer")
        object.__setattr__(self, "layers", tuple(sorted(raw_layers)))
        if self.stage != LAYER_CONTEXTUAL_STAGE:
            raise ValueError("layer continuation stage must be 'layer.output'")
        if self.projection_abi != LAYER_CONTEXTUAL_PROJECTION_ABI:
            raise ValueError("projection ABI is not implemented by this runtime")

    @property
    def hidden_width(self) -> int:
        """Compatibility alias for code that calls the model width hidden_width."""

        return self.hidden_dim

    @property
    def identity_sha256(self) -> str:
        return _sha256_document(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "layers": list(self.layers),
            "model_sha256": self.model_sha256,
            "projection": {
                "abi": self.projection_abi,
                "seed": self.projection_seed,
                "sketch_dim": self.sketch_dim,
            },
            "q4_sha256": self.q4_sha256,
            "runtime_sha256": self.runtime_sha256,
            "schema": LAYER_CONTEXTUAL_CONTINUATION_IDENTITY_SCHEMA,
            "stage": self.stage,
            "tokenizer_sha256": self.tokenizer_sha256,
        }

    @classmethod
    def from_record(cls, value: object) -> LayerContextualContinuationIdentity:
        if not isinstance(value, Mapping) or set(value) != {
            "hidden_dim",
            "layers",
            "model_sha256",
            "projection",
            "q4_sha256",
            "runtime_sha256",
            "schema",
            "stage",
            "tokenizer_sha256",
        }:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation identity fields are invalid"
            )
        if value["schema"] != LAYER_CONTEXTUAL_CONTINUATION_IDENTITY_SCHEMA:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation identity schema is invalid"
            )
        projection = value["projection"]
        if not isinstance(projection, Mapping) or set(projection) != {
            "abi",
            "seed",
            "sketch_dim",
        }:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation projection identity is invalid"
            )
        try:
            raw_layers = tuple(value["layers"])
            if raw_layers != tuple(sorted(raw_layers)):
                raise ValueError("persisted identity layers are not sorted")
            return cls(
                runtime_sha256=value["runtime_sha256"],
                model_sha256=value["model_sha256"],
                q4_sha256=value["q4_sha256"],
                tokenizer_sha256=value["tokenizer_sha256"],
                hidden_dim=value["hidden_dim"],
                layers=raw_layers,
                projection_seed=projection["seed"],
                sketch_dim=projection["sketch_dim"],
                stage=value["stage"],
                projection_abi=projection["abi"],
            )
        except (TypeError, ValueError) as exc:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation identity values are invalid"
            ) from exc


def _transaction_address(
    *,
    identity_sha256: str,
    boundary_index: int,
    known_token: int,
    transaction_nonce: str,
    keys: Sequence[LayerContextualContinuationKey],
) -> dict[str, object]:
    return {
        "boundary_index": boundary_index,
        "identity_sha256": identity_sha256,
        "keys": [key.to_record() for key in keys],
        "known_token": known_token,
        "schema": LAYER_CONTEXTUAL_CONTINUATION_TRANSACTION_SCHEMA,
        "transaction_nonce": transaction_nonce,
    }


@dataclass(frozen=True, slots=True)
class LayerContextualContinuationTransaction:
    """One retry-safe, multi-layer boundary capture."""

    identity_sha256: str
    boundary_index: int
    known_token: int
    keys: tuple[LayerContextualContinuationKey, ...]
    transaction_nonce: str
    transaction_sha256: str

    def __post_init__(self) -> None:
        _digest(self.identity_sha256, "transaction identity_sha256")
        _uint(
            self.boundary_index,
            field="boundary_index",
            maximum=_MAX_BOUNDARY_INDEX,
        )
        _token(self.known_token)
        if (
            not isinstance(self.transaction_nonce, str)
            or len(self.transaction_nonce) != 32
            or set(self.transaction_nonce) - _HEX
        ):
            raise ValueError("transaction_nonce must be 16 lowercase hexadecimal bytes")
        keys = tuple(self.keys)
        if not keys or any(
            not isinstance(key, LayerContextualContinuationKey) for key in keys
        ):
            raise TypeError("transaction keys must be layer continuation keys")
        if tuple(sorted(keys, key=lambda key: key.layer)) != keys:
            raise ValueError("transaction keys must be sorted by layer")
        if len({key.layer for key in keys}) != len(keys):
            raise ValueError("transaction contains a duplicate layer")
        if any(key.known_token != self.known_token for key in keys):
            raise ValueError("transaction key known_token differs from transaction")
        object.__setattr__(self, "keys", keys)
        expected = _sha256_document(
            _transaction_address(
                identity_sha256=self.identity_sha256,
                boundary_index=self.boundary_index,
                known_token=self.known_token,
                transaction_nonce=self.transaction_nonce,
                keys=keys,
            )
        )
        if self.transaction_sha256 != expected:
            raise ValueError("layer continuation transaction SHA-256 mismatch")

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(key.layer for key in self.keys)

    def key_for_layer(self, layer: int) -> LayerContextualContinuationKey:
        selected = _layer(layer)
        for key in self.keys:
            if key.layer == selected:
                return key
        raise KeyError(selected)

    def to_record(self) -> dict[str, object]:
        return _transaction_address(
            identity_sha256=self.identity_sha256,
            boundary_index=self.boundary_index,
            known_token=self.known_token,
            transaction_nonce=self.transaction_nonce,
            keys=self.keys,
        ) | {"transaction_sha256": self.transaction_sha256}

    @classmethod
    def create(
        cls,
        *,
        identity_sha256: str,
        boundary_index: int,
        known_token: int,
        keys: Sequence[LayerContextualContinuationKey],
        transaction_nonce: str | None = None,
    ) -> LayerContextualContinuationTransaction:
        ordered = tuple(keys)
        nonce = (
            secrets.token_hex(16) if transaction_nonce is None else transaction_nonce
        )
        address = _transaction_address(
            identity_sha256=identity_sha256,
            boundary_index=boundary_index,
            known_token=known_token,
            transaction_nonce=nonce,
            keys=ordered,
        )
        return cls(
            identity_sha256=identity_sha256,
            boundary_index=boundary_index,
            known_token=known_token,
            keys=ordered,
            transaction_nonce=nonce,
            transaction_sha256=_sha256_document(address),
        )

    @classmethod
    def from_record(
        cls,
        value: object,
        *,
        sketch_dim: int,
    ) -> LayerContextualContinuationTransaction:
        expected_fields = {
            "boundary_index",
            "identity_sha256",
            "keys",
            "known_token",
            "schema",
            "transaction_nonce",
            "transaction_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation transaction fields are invalid"
            )
        if value["schema"] != LAYER_CONTEXTUAL_CONTINUATION_TRANSACTION_SCHEMA:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation transaction schema is invalid"
            )
        try:
            return cls(
                identity_sha256=value["identity_sha256"],
                boundary_index=value["boundary_index"],
                known_token=value["known_token"],
                keys=tuple(
                    LayerContextualContinuationKey.from_record(
                        key,
                        sketch_dim=sketch_dim,
                    )
                    for key in value["keys"]
                ),
                transaction_nonce=value["transaction_nonce"],
                transaction_sha256=value["transaction_sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation transaction values are invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class LayerContextualContinuationOption:
    """One target tail ranked within exactly one layer/token bucket."""

    transaction_sha256: str
    cell_sha256: str
    layer: int
    known_token: int
    target_tail: tuple[int, ...]
    cosine: float
    runner_up_cosine: float | None
    margin: float | None
    support: int
    position_verified: tuple[int, ...]
    position_hits: tuple[int, ...]
    collapsed_cells: int

    def __post_init__(self) -> None:
        _digest(self.transaction_sha256, "option transaction_sha256")
        _digest(self.cell_sha256, "option cell_sha256")
        _layer(self.layer)
        _token(self.known_token)
        tail = _tail(self.target_tail)
        if not math.isfinite(self.cosine) or not -1.0 <= self.cosine <= 1.0:
            raise ValueError("option cosine must be finite and in [-1, 1]")
        if self.runner_up_cosine is None:
            if self.margin is not None:
                raise ValueError("option margin requires a runner-up")
        else:
            if (
                not math.isfinite(self.runner_up_cosine)
                or not -1.0 <= self.runner_up_cosine <= 1.0
            ):
                raise ValueError("runner-up cosine must be finite and in [-1, 1]")
            if self.margin is None or not math.isfinite(self.margin):
                raise ValueError("option margin must be finite")
            if not math.isclose(
                self.margin,
                self.cosine - self.runner_up_cosine,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("option margin differs from its cosines")
        _uint(self.support, field="option support", positive=True)
        _uint(self.collapsed_cells, field="collapsed_cells", positive=True)
        verified = tuple(self.position_verified)
        hits = tuple(self.position_hits)
        if len(verified) != len(tail) or len(hits) != len(tail):
            raise ValueError("option position counters do not match its tail")
        for index, (verified_count, hit_count) in enumerate(zip(verified, hits)):
            _uint(verified_count, field=f"position_verified[{index}]")
            _uint(hit_count, field=f"position_hits[{index}]")
            if hit_count > verified_count:
                raise ValueError("option hits exceed verified observations")
        object.__setattr__(self, "target_tail", tail)
        object.__setattr__(self, "position_verified", verified)
        object.__setattr__(self, "position_hits", hits)

    @property
    def width(self) -> int:
        return len(self.target_tail)

    @property
    def tail(self) -> tuple[int, ...]:
        return self.target_tail

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_sha256": self.cell_sha256,
            "collapsed_cells": self.collapsed_cells,
            "cosine": self.cosine,
            "known_token": self.known_token,
            "layer": self.layer,
            "margin": self.margin,
            "position_hits": list(self.position_hits),
            "position_verified": list(self.position_verified),
            "runner_up_cosine": self.runner_up_cosine,
            "support": self.support,
            "target_tail": list(self.target_tail),
            "transaction_sha256": self.transaction_sha256,
            "width": self.width,
        }


@dataclass(frozen=True, slots=True)
class _Cell:
    content_sha256: str
    key: LayerContextualContinuationKey
    target_tail: tuple[int, ...]
    support: int
    position_verified: tuple[int, ...]
    position_hits: tuple[int, ...]
    created_clock: int
    last_used_clock: int
    identity_sha256: str

    def __post_init__(self) -> None:
        _digest(self.content_sha256, "cell content_sha256")
        _digest(self.identity_sha256, "cell identity_sha256")
        if not isinstance(self.key, LayerContextualContinuationKey):
            raise ValueError("cell key is invalid")
        tail = _tail(self.target_tail)
        verified = tuple(self.position_verified)
        hits = tuple(self.position_hits)
        if len(verified) != len(tail) or len(hits) != len(tail):
            raise ValueError("cell position counters do not match its tail")
        for index, (verified_count, hit_count) in enumerate(zip(verified, hits)):
            _uint(verified_count, field=f"cell position_verified[{index}]")
            _uint(hit_count, field=f"cell position_hits[{index}]")
            if hit_count > verified_count:
                raise ValueError("cell hits exceed verified observations")
        _uint(self.support, field="cell support", positive=True)
        _uint(self.created_clock, field="cell created_clock", positive=True)
        _uint(self.last_used_clock, field="cell last_used_clock", positive=True)
        if self.last_used_clock < self.created_clock:
            raise ValueError("cell last-used clock precedes creation")
        object.__setattr__(self, "target_tail", tail)
        object.__setattr__(self, "position_verified", verified)
        object.__setattr__(self, "position_hits", hits)
        if self.content_sha256 != self.expected_content_sha256():
            raise ValueError("cell content SHA-256 mismatch")

    def expected_content_sha256(self) -> str:
        return _sha256_document(
            {
                "identity_sha256": self.identity_sha256,
                "key": self.key.to_record(),
                "schema": LAYER_CONTEXTUAL_CONTINUATION_CELL_SCHEMA,
                "target_tail": list(self.target_tail),
            }
        )

    @property
    def value_units(self) -> int:
        width = len(self.target_tail)
        evidence = sum(
            (width - index) * (2 * hits - verified)
            for index, (hits, verified) in enumerate(
                zip(self.position_hits, self.position_verified)
            )
        )
        return self.support * width + evidence

    def to_record(self) -> dict[str, object]:
        return {
            "content_sha256": self.content_sha256,
            "created_clock": self.created_clock,
            "key": self.key.to_record(),
            "last_used_clock": self.last_used_clock,
            "position_hits": list(self.position_hits),
            "position_verified": list(self.position_verified),
            "schema": LAYER_CONTEXTUAL_CONTINUATION_CELL_SCHEMA,
            "support": self.support,
            "target_tail": list(self.target_tail),
        }

    @classmethod
    def from_record(
        cls,
        value: object,
        *,
        identity_sha256: str,
        sketch_dim: int,
    ) -> _Cell:
        if not isinstance(value, Mapping) or set(value) != {
            "content_sha256",
            "created_clock",
            "key",
            "last_used_clock",
            "position_hits",
            "position_verified",
            "schema",
            "support",
            "target_tail",
        }:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation cell fields are invalid"
            )
        if value["schema"] != LAYER_CONTEXTUAL_CONTINUATION_CELL_SCHEMA:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation cell schema is invalid"
            )
        try:
            return cls(
                content_sha256=value["content_sha256"],
                key=LayerContextualContinuationKey.from_record(
                    value["key"], sketch_dim=sketch_dim
                ),
                target_tail=tuple(value["target_tail"]),
                support=value["support"],
                position_verified=tuple(value["position_verified"]),
                position_hits=tuple(value["position_hits"]),
                created_clock=value["created_clock"],
                last_used_clock=value["last_used_clock"],
                identity_sha256=identity_sha256,
            )
        except (TypeError, ValueError) as exc:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation cell values are invalid"
            ) from exc


def _cell_for_capture(
    key: LayerContextualContinuationKey,
    target_tail: tuple[int, ...],
    *,
    identity_sha256: str,
    clock: int,
) -> _Cell:
    address = {
        "identity_sha256": identity_sha256,
        "key": key.to_record(),
        "schema": LAYER_CONTEXTUAL_CONTINUATION_CELL_SCHEMA,
        "target_tail": list(target_tail),
    }
    return _Cell(
        content_sha256=_sha256_document(address),
        key=key,
        target_tail=target_tail,
        support=1,
        position_verified=(0,) * len(target_tail),
        position_hits=(0,) * len(target_tail),
        created_clock=clock,
        last_used_clock=clock,
        identity_sha256=identity_sha256,
    )


@dataclass(frozen=True, slots=True)
class _Counter:
    crystal_queries: int = 0
    crystal_query_hits: int = 0
    crystal_option_calls: int = 0
    crystal_proposed_tokens: int = 0
    crystal_verified_tokens: int = 0
    crystal_accepted_tokens: int = 0
    crystal_mismatches: int = 0
    crystal_captures: int = 0
    crystal_failures: int = 0
    crystal_last_cosine: float = 0.0
    crystal_last_margin: float = 0.0
    crystal_last_cell_sha256: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "crystal_queries",
            "crystal_query_hits",
            "crystal_option_calls",
            "crystal_proposed_tokens",
            "crystal_verified_tokens",
            "crystal_accepted_tokens",
            "crystal_mismatches",
            "crystal_captures",
            "crystal_failures",
        ):
            _uint(getattr(self, field), field=field)
        if self.crystal_query_hits > self.crystal_queries:
            raise ValueError("crystal query hits exceed queries")
        if self.crystal_option_calls != self.crystal_queries:
            raise ValueError("crystal option calls differ from queries")
        if self.crystal_accepted_tokens > self.crystal_verified_tokens:
            raise ValueError("crystal accepted tokens exceed verified tokens")
        if self.crystal_mismatches > self.crystal_verified_tokens:
            raise ValueError("crystal mismatches exceed verified tokens")
        if (
            not math.isfinite(self.crystal_last_cosine)
            or not -1.0 <= (self.crystal_last_cosine) <= 1.0
        ):
            raise ValueError("crystal last cosine is invalid")
        if (
            not math.isfinite(self.crystal_last_margin)
            or not 0.0 <= (self.crystal_last_margin) <= 2.0
        ):
            raise ValueError("crystal last margin is invalid")
        if self.crystal_last_cell_sha256 is not None:
            _digest(self.crystal_last_cell_sha256, "crystal_last_cell_sha256")

    def to_record(self) -> dict[str, object]:
        return {
            "crystal_accepted_tokens": self.crystal_accepted_tokens,
            "crystal_captures": self.crystal_captures,
            "crystal_failures": self.crystal_failures,
            "crystal_last_cell_sha256": self.crystal_last_cell_sha256,
            "crystal_last_cosine": self.crystal_last_cosine,
            "crystal_last_margin": self.crystal_last_margin,
            "crystal_mismatches": self.crystal_mismatches,
            "crystal_option_calls": self.crystal_option_calls,
            "crystal_proposed_tokens": self.crystal_proposed_tokens,
            "crystal_queries": self.crystal_queries,
            "crystal_query_hits": self.crystal_query_hits,
            "crystal_verified_tokens": self.crystal_verified_tokens,
        }

    @classmethod
    def from_record(cls, value: object) -> _Counter:
        fields = {
            "crystal_accepted_tokens",
            "crystal_captures",
            "crystal_failures",
            "crystal_last_cell_sha256",
            "crystal_last_cosine",
            "crystal_last_margin",
            "crystal_mismatches",
            "crystal_option_calls",
            "crystal_proposed_tokens",
            "crystal_queries",
            "crystal_query_hits",
            "crystal_verified_tokens",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation counter fields are invalid"
            )
        try:
            return cls(**{field: value[field] for field in fields})
        except (TypeError, ValueError) as exc:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation counter values are invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class _Receipt:
    transaction_sha256: str
    outcome_sha256: str
    settled_clock: int

    def __post_init__(self) -> None:
        _digest(self.transaction_sha256, "receipt transaction_sha256")
        _digest(self.outcome_sha256, "receipt outcome_sha256")
        _uint(self.settled_clock, field="receipt settled_clock", positive=True)

    def to_record(self) -> dict[str, object]:
        return {
            "outcome_sha256": self.outcome_sha256,
            "schema": LAYER_CONTEXTUAL_CONTINUATION_RECEIPT_SCHEMA,
            "settled_clock": self.settled_clock,
            "transaction_sha256": self.transaction_sha256,
        }

    @classmethod
    def from_record(cls, value: object) -> _Receipt:
        if not isinstance(value, Mapping) or set(value) != {
            "outcome_sha256",
            "schema",
            "settled_clock",
            "transaction_sha256",
        }:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation receipt fields are invalid"
            )
        if value["schema"] != LAYER_CONTEXTUAL_CONTINUATION_RECEIPT_SCHEMA:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation receipt schema is invalid"
            )
        try:
            return cls(
                transaction_sha256=value["transaction_sha256"],
                outcome_sha256=value["outcome_sha256"],
                settled_clock=value["settled_clock"],
            )
        except (TypeError, ValueError) as exc:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation receipt values are invalid"
            ) from exc


def _zero_layer_counters(
    identity: LayerContextualContinuationIdentity,
) -> tuple[tuple[int, _Counter], ...]:
    return tuple((layer, _Counter()) for layer in identity.layers)


@dataclass(frozen=True, slots=True)
class _State:
    identity: LayerContextualContinuationIdentity
    max_cells: int
    max_receipts: int
    clock: int = 0
    settlements: int = 0
    evictions: int = 0
    counter: _Counter = _Counter()
    layer_counters: tuple[tuple[int, _Counter], ...] = ()
    cells: tuple[_Cell, ...] = ()
    receipts: tuple[_Receipt, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.identity, LayerContextualContinuationIdentity):
            raise ValueError("layer continuation state identity is invalid")
        _uint(self.max_cells, field="max_cells", positive=True, maximum=_MAX_CELLS)
        _uint(
            self.max_receipts,
            field="max_receipts",
            positive=True,
            maximum=_MAX_RECEIPTS,
        )
        _uint(self.clock, field="clock")
        _uint(self.settlements, field="settlements")
        _uint(self.evictions, field="evictions")
        if not isinstance(self.counter, _Counter):
            raise ValueError("layer continuation global counter is invalid")
        layer_counters = tuple(self.layer_counters)
        if not layer_counters:
            layer_counters = _zero_layer_counters(self.identity)
            object.__setattr__(self, "layer_counters", layer_counters)
        if tuple(layer for layer, _counter in layer_counters) != self.identity.layers:
            raise ValueError("layer continuation counters do not match identity layers")
        if any(not isinstance(counter, _Counter) for _, counter in layer_counters):
            raise ValueError("layer continuation per-layer counter is invalid")
        aggregate_fields = (
            "crystal_queries",
            "crystal_query_hits",
            "crystal_option_calls",
            "crystal_proposed_tokens",
            "crystal_verified_tokens",
            "crystal_accepted_tokens",
            "crystal_mismatches",
            "crystal_captures",
            "crystal_failures",
        )
        for field in aggregate_fields:
            expected = min(
                _MAX_COUNTER,
                sum(getattr(counter, field) for _, counter in layer_counters),
            )
            if getattr(self.counter, field) != expected:
                raise ValueError(f"global {field} differs from per-layer counters")
        cells = tuple(self.cells)
        if len(cells) > self.max_cells:
            raise ValueError("layer continuation state exceeds max_cells")
        if tuple(sorted(cells, key=lambda cell: cell.content_sha256)) != cells:
            raise ValueError("layer continuation cells are not canonically ordered")
        if len({cell.content_sha256 for cell in cells}) != len(cells):
            raise ValueError("layer continuation state contains duplicate cells")
        if any(
            cell.identity_sha256 != self.identity.identity_sha256
            or cell.key.layer not in self.identity.layers
            or cell.key.sketch_dim != self.identity.sketch_dim
            or cell.last_used_clock > self.clock
            for cell in cells
        ):
            raise ValueError("layer continuation cell scope or clock is invalid")
        receipts = tuple(self.receipts)
        if len(receipts) > self.max_receipts:
            raise ValueError("layer continuation receipt ledger exceeds its bound")
        if (
            tuple(sorted(receipts, key=lambda item: item.transaction_sha256))
            != receipts
        ):
            raise ValueError("layer continuation receipts are not canonically ordered")
        if len({item.transaction_sha256 for item in receipts}) != len(receipts):
            raise ValueError("layer continuation state contains duplicate receipts")
        if any(item.settled_clock > self.clock for item in receipts):
            raise ValueError("layer continuation receipt clock is invalid")
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "receipts", receipts)

    def to_record(self) -> dict[str, object]:
        return {
            "cells": [cell.to_record() for cell in self.cells],
            "clock": self.clock,
            "counter": self.counter.to_record(),
            "evictions": self.evictions,
            "identity": self.identity.to_record(),
            "identity_sha256": self.identity.identity_sha256,
            "layer_counters": [
                {"counter": counter.to_record(), "layer": layer}
                for layer, counter in self.layer_counters
            ],
            "max_cells": self.max_cells,
            "max_receipts": self.max_receipts,
            "receipts": [receipt.to_record() for receipt in self.receipts],
            "schema": LAYER_CONTEXTUAL_CONTINUATION_STATE_SCHEMA,
            "settlements": self.settlements,
        }

    @classmethod
    def from_record(cls, value: object) -> _State:
        if not isinstance(value, Mapping) or set(value) != {
            "cells",
            "clock",
            "counter",
            "evictions",
            "identity",
            "identity_sha256",
            "layer_counters",
            "max_cells",
            "max_receipts",
            "receipts",
            "schema",
            "settlements",
        }:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation state fields are invalid"
            )
        if value["schema"] != LAYER_CONTEXTUAL_CONTINUATION_STATE_SCHEMA:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation state schema is invalid"
            )
        try:
            identity = LayerContextualContinuationIdentity.from_record(
                value["identity"]
            )
            if value["identity_sha256"] != identity.identity_sha256:
                raise ValueError("layer continuation identity SHA-256 mismatch")
            raw_layer_counters = value["layer_counters"]
            if not isinstance(raw_layer_counters, Sequence):
                raise TypeError("layer_counters must be a sequence")
            parsed_layer_counters: list[tuple[int, _Counter]] = []
            for item in raw_layer_counters:
                if not isinstance(item, Mapping) or set(item) != {"counter", "layer"}:
                    raise ValueError("per-layer counter record is invalid")
                parsed_layer_counters.append(
                    (_layer(item["layer"]), _Counter.from_record(item["counter"]))
                )
            return cls(
                identity=identity,
                max_cells=value["max_cells"],
                max_receipts=value["max_receipts"],
                clock=value["clock"],
                settlements=value["settlements"],
                evictions=value["evictions"],
                counter=_Counter.from_record(value["counter"]),
                layer_counters=tuple(parsed_layer_counters),
                cells=tuple(
                    _Cell.from_record(
                        cell,
                        identity_sha256=identity.identity_sha256,
                        sketch_dim=identity.sketch_dim,
                    )
                    for cell in value["cells"]
                ),
                receipts=tuple(
                    _Receipt.from_record(item) for item in value["receipts"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation state values are invalid"
            ) from exc

    def to_bytes(self) -> bytes:
        body = self.to_record()
        envelope = {
            "body": body,
            "body_sha256": _sha256_document(body),
            "schema": LAYER_CONTEXTUAL_CONTINUATION_ENVELOPE_SCHEMA,
        }
        encoded = _canonical_json(envelope)
        if len(encoded) > _MAX_STATE_BYTES:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation state exceeds its byte limit"
            )
        return encoded

    @classmethod
    def from_bytes(cls, value: bytes) -> _State:
        try:
            document = json.loads(value, object_pairs_hook=_json_no_duplicates)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation state is not valid JSON"
            ) from exc
        if _canonical_json(document) != value:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation state is not canonical JSON"
            )
        if not isinstance(document, Mapping) or set(document) != {
            "body",
            "body_sha256",
            "schema",
        }:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation envelope fields are invalid"
            )
        if document["schema"] != LAYER_CONTEXTUAL_CONTINUATION_ENVELOPE_SCHEMA:
            raise LayerContextualContinuationIntegrityError(
                "layer continuation envelope schema is invalid"
            )
        if not _is_sha256(document["body_sha256"]):
            raise LayerContextualContinuationIntegrityError(
                "layer continuation body digest is invalid"
            )
        if document["body_sha256"] != _sha256_document(document["body"]):
            raise LayerContextualContinuationIntegrityError(
                "layer continuation body SHA-256 mismatch"
            )
        return cls.from_record(document["body"])


@dataclass(frozen=True, slots=True)
class LayerContextualContinuationLayerMetrics:
    crystal_queries: int
    crystal_query_hits: int
    crystal_option_calls: int
    crystal_proposed_tokens: int
    crystal_verified_tokens: int
    crystal_accepted_tokens: int
    crystal_mismatches: int
    crystal_captures: int
    crystal_failures: int
    crystal_bank_cells: int
    crystal_bank_support: int
    crystal_last_cosine: float
    crystal_last_margin: float
    crystal_last_cell_sha256: str | None
    persisted_crystal_queries: int
    persisted_crystal_query_hits: int
    persisted_crystal_option_calls: int
    persisted_crystal_proposed_tokens: int
    process_crystal_queries_delta: int
    process_crystal_query_hits_delta: int
    process_crystal_option_calls_delta: int
    process_crystal_proposed_tokens_delta: int

    def to_dict(self) -> dict[str, object]:
        return {
            "crystal_accepted_tokens": self.crystal_accepted_tokens,
            "crystal_bank_cells": self.crystal_bank_cells,
            "crystal_bank_support": self.crystal_bank_support,
            "crystal_captures": self.crystal_captures,
            "crystal_failures": self.crystal_failures,
            "crystal_last_cell_sha256": self.crystal_last_cell_sha256,
            "crystal_last_cosine": self.crystal_last_cosine,
            "crystal_last_margin": self.crystal_last_margin,
            "crystal_mismatches": self.crystal_mismatches,
            "crystal_option_calls": self.crystal_option_calls,
            "crystal_proposed_tokens": self.crystal_proposed_tokens,
            "crystal_queries": self.crystal_queries,
            "crystal_query_hits": self.crystal_query_hits,
            "crystal_verified_tokens": self.crystal_verified_tokens,
            "persisted_crystal_option_calls": self.persisted_crystal_option_calls,
            "persisted_crystal_proposed_tokens": (
                self.persisted_crystal_proposed_tokens
            ),
            "persisted_crystal_queries": self.persisted_crystal_queries,
            "persisted_crystal_query_hits": self.persisted_crystal_query_hits,
            "process_crystal_option_calls_delta": (
                self.process_crystal_option_calls_delta
            ),
            "process_crystal_proposed_tokens_delta": (
                self.process_crystal_proposed_tokens_delta
            ),
            "process_crystal_queries_delta": self.process_crystal_queries_delta,
            "process_crystal_query_hits_delta": (self.process_crystal_query_hits_delta),
        }

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]


@dataclass(frozen=True, slots=True)
class LayerContextualContinuationMetrics:
    schema: str
    identity_sha256: str
    state_sha256: str
    max_cells: int
    max_receipts: int
    clock: int
    settlements: int
    evictions: int
    receipt_count: int
    crystal_enabled: bool
    crystal_queries: int
    crystal_query_hits: int
    crystal_option_calls: int
    crystal_proposed_tokens: int
    crystal_verified_tokens: int
    crystal_accepted_tokens: int
    crystal_mismatches: int
    crystal_captures: int
    crystal_failures: int
    crystal_bank_cells: int
    crystal_bank_support: int
    crystal_last_cosine: float
    crystal_last_margin: float
    crystal_last_cell_sha256: str | None
    query_metrics_scope: str
    persisted_crystal_queries: int
    persisted_crystal_query_hits: int
    persisted_crystal_option_calls: int
    persisted_crystal_proposed_tokens: int
    process_crystal_queries_delta: int
    process_crystal_query_hits_delta: int
    process_crystal_option_calls_delta: int
    process_crystal_proposed_tokens_delta: int
    layers: Mapping[str, LayerContextualContinuationLayerMetrics]

    def to_dict(self) -> dict[str, object]:
        return {
            "clock": self.clock,
            "crystal_accepted_tokens": self.crystal_accepted_tokens,
            "crystal_bank_cells": self.crystal_bank_cells,
            "crystal_bank_support": self.crystal_bank_support,
            "crystal_captures": self.crystal_captures,
            "crystal_enabled": self.crystal_enabled,
            "crystal_failures": self.crystal_failures,
            "crystal_last_cell_sha256": self.crystal_last_cell_sha256,
            "crystal_last_cosine": self.crystal_last_cosine,
            "crystal_last_margin": self.crystal_last_margin,
            "crystal_mismatches": self.crystal_mismatches,
            "crystal_option_calls": self.crystal_option_calls,
            "crystal_proposed_tokens": self.crystal_proposed_tokens,
            "crystal_queries": self.crystal_queries,
            "crystal_query_hits": self.crystal_query_hits,
            "crystal_verified_tokens": self.crystal_verified_tokens,
            "evictions": self.evictions,
            "identity_sha256": self.identity_sha256,
            "layers": {
                layer: metrics.to_dict() for layer, metrics in self.layers.items()
            },
            "max_cells": self.max_cells,
            "max_receipts": self.max_receipts,
            "persisted_crystal_option_calls": self.persisted_crystal_option_calls,
            "persisted_crystal_proposed_tokens": (
                self.persisted_crystal_proposed_tokens
            ),
            "persisted_crystal_queries": self.persisted_crystal_queries,
            "persisted_crystal_query_hits": self.persisted_crystal_query_hits,
            "process_crystal_option_calls_delta": (
                self.process_crystal_option_calls_delta
            ),
            "process_crystal_proposed_tokens_delta": (
                self.process_crystal_proposed_tokens_delta
            ),
            "process_crystal_queries_delta": self.process_crystal_queries_delta,
            "process_crystal_query_hits_delta": (self.process_crystal_query_hits_delta),
            "query_metrics_scope": self.query_metrics_scope,
            "receipt_count": self.receipt_count,
            "schema": self.schema,
            "settlements": self.settlements,
            "state_sha256": self.state_sha256,
        }

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]


@lru_cache(maxsize=4)
def _rademacher_projection(
    identity_sha256: str,
    projection_seed: int,
    layer: int,
    hidden_dim: int,
    sketch_dim: int,
) -> torch.Tensor:
    material = _canonical_json(
        {
            "hidden_dim": hidden_dim,
            "identity_sha256": identity_sha256,
            "layer": layer,
            "projection_abi": LAYER_CONTEXTUAL_PROJECTION_ABI,
            "projection_seed": projection_seed,
            "sketch_dim": sketch_dim,
            "stage": LAYER_CONTEXTUAL_STAGE,
        }
    )
    count = hidden_dim * sketch_dim
    raw = hashlib.shake_256(material).digest((count + 7) // 8)
    octets = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    shifts = torch.arange(8, dtype=torch.uint8)
    bits = torch.bitwise_and(
        torch.bitwise_right_shift(octets[:, None], shifts[None, :]),
        1,
    ).reshape(-1)[:count]
    projection = bits.to(dtype=torch.float64).mul_(2.0).sub_(1.0)
    projection = projection.reshape(hidden_dim, sketch_dim)
    projection.mul_(1.0 / math.sqrt(sketch_dim))
    return projection.contiguous()


def _project_hidden(
    hidden: torch.Tensor,
    *,
    identity: LayerContextualContinuationIdentity,
    layer: int,
) -> torch.Tensor:
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("hidden must be a torch.Tensor")
    if tuple(hidden.shape) != (1, 1, identity.hidden_dim):
        raise ValueError(f"hidden must have exact shape [1, 1, {identity.hidden_dim}]")
    if not hidden.dtype.is_floating_point:
        raise TypeError("hidden must have a floating-point dtype")
    vector = hidden.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("hidden contains a non-finite value")
    projection = _rademacher_projection(
        identity.identity_sha256,
        identity.projection_seed,
        layer,
        identity.hidden_dim,
        identity.sketch_dim,
    )
    sketch = torch.matmul(vector, projection)
    norm = torch.linalg.vector_norm(sketch)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise ValueError("hidden has no projectable norm")
    q8 = torch.round(sketch.div(norm).mul(127.0)).to(dtype=torch.int8).contiguous()
    _q8_values(q8, sketch_dim=identity.sketch_dim)
    return q8


def project_layer_contextual_key(
    identity: LayerContextualContinuationIdentity,
    hidden: torch.Tensor,
    layer: int,
    known_token: int,
) -> LayerContextualContinuationKey:
    """Project one ``layer.output`` row without opening or owning a bank.

    This is the path-free projection boundary used by the streamed model.  It
    deliberately returns the same signed-Q8 key as
    :meth:`LayerContextualContinuationBank.project` while retaining neither
    the supplied hidden row nor a persistence handle.
    """

    if not isinstance(identity, LayerContextualContinuationIdentity):
        raise TypeError(
            "identity must be a LayerContextualContinuationIdentity"
        )
    selected_layer = _layer(layer)
    if selected_layer not in identity.layers:
        raise ValueError("layer is outside the continuation identity")
    known = _token(known_token)
    return LayerContextualContinuationKey(
        selected_layer,
        known,
        _project_hidden(hidden, identity=identity, layer=selected_layer),
    )


def _cosine(
    left: LayerContextualContinuationKey,
    right: LayerContextualContinuationKey,
) -> float:
    left_values = left.q8_values
    right_values = right.q8_values
    dot = sum(a * b for a, b in zip(left_values, right_values))
    denominator = math.sqrt(left.norm_sq * right.norm_sq)
    return max(-1.0, min(1.0, dot / denominator))


def _counter_for_query(
    counter: _Counter,
    options: Sequence[LayerContextualContinuationOption],
) -> _Counter:
    top = options[0] if options else None
    return replace(
        counter,
        crystal_queries=_bounded_add(counter.crystal_queries, 1),
        crystal_query_hits=_bounded_add(
            counter.crystal_query_hits,
            int(top is not None),
        ),
        crystal_option_calls=_bounded_add(counter.crystal_option_calls, 1),
        crystal_proposed_tokens=_bounded_add(
            counter.crystal_proposed_tokens,
            0 if top is None else top.width,
        ),
        crystal_last_cosine=(
            counter.crystal_last_cosine if top is None else top.cosine
        ),
        crystal_last_margin=(
            counter.crystal_last_margin if top is None else float(top.margin or 0.0)
        ),
        crystal_last_cell_sha256=(
            counter.crystal_last_cell_sha256 if top is None else top.cell_sha256
        ),
    )


def _counter_for_feedback(
    counter: _Counter,
    *,
    verified_tokens: int,
    accepted_tokens: int,
) -> _Counter:
    return replace(
        counter,
        crystal_verified_tokens=_bounded_add(
            counter.crystal_verified_tokens, verified_tokens
        ),
        crystal_accepted_tokens=_bounded_add(
            counter.crystal_accepted_tokens, accepted_tokens
        ),
        crystal_mismatches=_bounded_add(
            counter.crystal_mismatches,
            int(accepted_tokens < verified_tokens),
        ),
    )


def _counter_for_capture(counter: _Counter) -> _Counter:
    return replace(
        counter,
        crystal_captures=_bounded_add(counter.crystal_captures, 1),
    )


def _merge_counters(persisted: _Counter, process_delta: _Counter) -> _Counter:
    """Combine durable work with this bank instance's read-only query delta."""

    has_process_hit = process_delta.crystal_last_cell_sha256 is not None
    return _Counter(
        crystal_queries=_bounded_add(
            persisted.crystal_queries, process_delta.crystal_queries
        ),
        crystal_query_hits=_bounded_add(
            persisted.crystal_query_hits, process_delta.crystal_query_hits
        ),
        crystal_option_calls=_bounded_add(
            persisted.crystal_option_calls, process_delta.crystal_option_calls
        ),
        crystal_proposed_tokens=_bounded_add(
            persisted.crystal_proposed_tokens,
            process_delta.crystal_proposed_tokens,
        ),
        crystal_verified_tokens=_bounded_add(
            persisted.crystal_verified_tokens,
            process_delta.crystal_verified_tokens,
        ),
        crystal_accepted_tokens=_bounded_add(
            persisted.crystal_accepted_tokens,
            process_delta.crystal_accepted_tokens,
        ),
        crystal_mismatches=_bounded_add(
            persisted.crystal_mismatches, process_delta.crystal_mismatches
        ),
        crystal_captures=_bounded_add(
            persisted.crystal_captures, process_delta.crystal_captures
        ),
        crystal_failures=_bounded_add(
            persisted.crystal_failures, process_delta.crystal_failures
        ),
        crystal_last_cosine=(
            process_delta.crystal_last_cosine
            if has_process_hit
            else persisted.crystal_last_cosine
        ),
        crystal_last_margin=(
            process_delta.crystal_last_margin
            if has_process_hit
            else persisted.crystal_last_margin
        ),
        crystal_last_cell_sha256=(
            process_delta.crystal_last_cell_sha256
            if has_process_hit
            else persisted.crystal_last_cell_sha256
        ),
    )


class LayerContextualContinuationBank:
    """Bounded persistent nearest-neighbour bank over layer-output sketches."""

    @classmethod
    def read_identity(
        cls,
        state_path: str | os.PathLike[str],
    ) -> LayerContextualContinuationIdentity:
        path = Path(state_path)
        if not path.name:
            raise ValueError("state_path must name a file")
        raw, _signature = _stable_regular_bytes(path)
        return _State.from_bytes(raw).identity

    def __init__(
        self,
        state_path: str | os.PathLike[str],
        identity: LayerContextualContinuationIdentity,
        *,
        max_cells: int = 4096,
        max_receipts: int = DEFAULT_LAYER_CONTEXTUAL_MAX_RECEIPTS,
    ) -> None:
        if not isinstance(identity, LayerContextualContinuationIdentity):
            raise TypeError("identity must be a LayerContextualContinuationIdentity")
        self.state_path = Path(state_path)
        if not self.state_path.name:
            raise ValueError("state_path must name a file")
        _uint(max_cells, field="max_cells", positive=True, maximum=_MAX_CELLS)
        _uint(
            max_receipts,
            field="max_receipts",
            positive=True,
            maximum=_MAX_RECEIPTS,
        )
        self.identity = identity
        self.max_cells = max_cells
        self.max_receipts = max_receipts
        self._lock = _thread_lock(self.state_path)
        self._file_signature: tuple[int, int, int, int, int] | None = None
        self._state = _State(
            identity=identity,
            max_cells=max_cells,
            max_receipts=max_receipts,
        )
        self._index: dict[tuple[int, int], tuple[_Cell, ...]] = {}
        self._process_query_counter = _Counter()
        self._process_query_layer_counters = dict(_zero_layer_counters(identity))
        with self._lock:
            self._reload(required=False)

    def _validate_state(self, state: _State) -> None:
        if state.identity.identity_sha256 != self.identity.identity_sha256:
            raise LayerContextualContinuationIdentityError(
                "layer continuation bank belongs to another runtime identity"
            )
        if state.max_cells != self.max_cells:
            raise LayerContextualContinuationIdentityError(
                "layer continuation max_cells differs from persistent state"
            )
        if state.max_receipts != self.max_receipts:
            raise LayerContextualContinuationIdentityError(
                "layer continuation max_receipts differs from persistent state"
            )

    def _reload(self, *, required: bool) -> None:
        try:
            raw, signature = _stable_regular_bytes(self.state_path)
        except FileNotFoundError:
            if required:
                raise LayerContextualContinuationIntegrityError(
                    "layer continuation state disappeared"
                )
            self._state = _State(
                identity=self.identity,
                max_cells=self.max_cells,
                max_receipts=self.max_receipts,
            )
            self._file_signature = None
            self._rebuild_index()
            return
        state = _State.from_bytes(raw)
        self._validate_state(state)
        self._state = state
        self._file_signature = signature
        self._rebuild_index()

    def _refresh_if_changed(self) -> None:
        try:
            linked = os.lstat(self.state_path)
        except FileNotFoundError:
            if self._file_signature is not None:
                raise LayerContextualContinuationIntegrityError(
                    "layer continuation state disappeared"
                )
            return
        if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
            raise LayerContextualContinuationIntegrityError(
                "layer continuation state must be a regular file"
            )
        if _stable_signature(linked) == self._file_signature:
            return
        last_error: LayerContextualContinuationIntegrityError | None = None
        for _ in range(3):
            try:
                self._reload(required=True)
                return
            except LayerContextualContinuationIntegrityError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def _rebuild_index(self) -> None:
        grouped: dict[tuple[int, int], list[_Cell]] = {}
        for cell in self._state.cells:
            grouped.setdefault((cell.key.layer, cell.key.known_token), []).append(cell)
        self._index = {bucket: tuple(cells) for bucket, cells in grouped.items()}

    def _publish(self, state: _State) -> None:
        encoded = state.to_bytes()
        signature = _atomic_write(self.state_path, encoded)
        self._state = state
        self._file_signature = signature
        self._rebuild_index()

    def project(
        self,
        hidden: torch.Tensor,
        layer: int,
        known_token: int,
    ) -> LayerContextualContinuationKey:
        return project_layer_contextual_key(
            self.identity,
            hidden,
            layer,
            known_token,
        )

    def capture_boundaries(
        self,
        layer_outputs: Mapping[int, torch.Tensor] | Sequence[torch.Tensor],
        known_token: int,
        boundary_index: int,
        *,
        transaction_nonce: str | None = None,
    ) -> LayerContextualContinuationTransaction:
        """Project every configured layer into one retry-safe transaction."""

        known = _token(known_token)
        boundary = _uint(
            boundary_index,
            field="boundary_index",
            maximum=_MAX_BOUNDARY_INDEX,
        )
        if isinstance(layer_outputs, Mapping):
            supplied: dict[int, torch.Tensor] = {}
            for layer_index, hidden in layer_outputs.items():
                selected = _layer(layer_index, "layer_outputs key")
                if selected in supplied:
                    raise ValueError("layer_outputs contains a duplicate layer")
                supplied[selected] = hidden
            if set(supplied) != set(self.identity.layers):
                raise ValueError("layer_outputs must exactly match identity layers")
        elif isinstance(layer_outputs, (str, bytes, bytearray)) or not isinstance(
            layer_outputs, Sequence
        ):
            raise TypeError("layer_outputs must be a mapping or ordered sequence")
        else:
            values = tuple(layer_outputs)
            if len(values) != len(self.identity.layers):
                raise ValueError("layer_outputs must exactly match identity layers")
            supplied = dict(zip(self.identity.layers, values))
        keys = tuple(
            self.project(supplied[layer_index], layer_index, known)
            for layer_index in self.identity.layers
        )
        return LayerContextualContinuationTransaction.create(
            identity_sha256=self.identity.identity_sha256,
            boundary_index=boundary,
            known_token=known,
            keys=keys,
            transaction_nonce=transaction_nonce,
        )

    def _validate_transaction(
        self,
        transaction: LayerContextualContinuationTransaction,
    ) -> None:
        if not isinstance(transaction, LayerContextualContinuationTransaction):
            raise TypeError(
                "transaction must be a LayerContextualContinuationTransaction"
            )
        if transaction.identity_sha256 != self.identity.identity_sha256:
            raise LayerContextualContinuationIdentityError(
                "transaction belongs to another layer continuation identity"
            )
        if transaction.layers != self.identity.layers:
            raise LayerContextualContinuationIdentityError(
                "transaction layers differ from the bank identity"
            )
        if any(key.sketch_dim != self.identity.sketch_dim for key in transaction.keys):
            raise LayerContextualContinuationIdentityError(
                "transaction sketch width differs from the bank identity"
            )

    def _rank_key(
        self,
        transaction_sha256: str,
        key: LayerContextualContinuationKey,
        *,
        limit: int,
    ) -> tuple[LayerContextualContinuationOption, ...]:
        cells = self._index.get((key.layer, key.known_token), ())
        grouped: dict[tuple[int, ...], list[tuple[_Cell, float]]] = {}
        for cell in cells:
            grouped.setdefault(cell.target_tail, []).append(
                (cell, _cosine(cell.key, key))
            )
        ranked: list[
            tuple[
                tuple[int, ...],
                _Cell,
                float,
                int,
                tuple[int, ...],
                tuple[int, ...],
                int,
            ]
        ] = []
        for tail, members in grouped.items():
            representative, cosine = min(
                members,
                key=lambda item: (
                    -item[1],
                    -item[0].value_units,
                    -item[0].last_used_clock,
                    item[0].content_sha256,
                ),
            )
            ranked.append(
                (
                    tail,
                    representative,
                    cosine,
                    representative.support,
                    representative.position_verified,
                    representative.position_hits,
                    len(members),
                )
            )
        ranked.sort(
            key=lambda item: (
                -item[2],
                -item[3],
                item[0],
                item[1].content_sha256,
            )
        )
        selected = ranked[:limit]
        options: list[LayerContextualContinuationOption] = []
        for index, (
            tail,
            representative,
            cosine,
            support,
            verified,
            hits,
            collapsed,
        ) in enumerate(selected):
            runner_up = ranked[index + 1][2] if index + 1 < len(ranked) else None
            options.append(
                LayerContextualContinuationOption(
                    transaction_sha256=transaction_sha256,
                    cell_sha256=representative.content_sha256,
                    layer=key.layer,
                    known_token=key.known_token,
                    target_tail=tail,
                    cosine=cosine,
                    runner_up_cosine=runner_up,
                    margin=None if runner_up is None else cosine - runner_up,
                    support=support,
                    position_verified=verified,
                    position_hits=hits,
                    collapsed_cells=collapsed,
                )
            )
        return tuple(options)

    def query_options(
        self,
        transaction: LayerContextualContinuationTransaction,
        *,
        limit: int = 1,
        limit_per_layer: int | None = None,
    ) -> tuple[LayerContextualContinuationOption, ...]:
        """Rank options independently in each ``(layer, known_token)`` bucket."""

        self._validate_transaction(transaction)
        if limit_per_layer is not None:
            if limit != 1:
                raise ValueError("provide only one of limit and limit_per_layer")
            limit = limit_per_layer
        _uint(limit, field="limit", positive=True, maximum=256)
        with self._lock:
            self._refresh_if_changed()
            by_layer: dict[int, tuple[LayerContextualContinuationOption, ...]] = {
                key.layer: self._rank_key(
                    transaction.transaction_sha256,
                    key,
                    limit=limit,
                )
                for key in transaction.keys
            }
            for key in transaction.keys:
                options = by_layer[key.layer]
                self._process_query_layer_counters[key.layer] = _counter_for_query(
                    self._process_query_layer_counters[key.layer], options
                )
                self._process_query_counter = _counter_for_query(
                    self._process_query_counter, options
                )
            return tuple(
                option for layer in self.identity.layers for option in by_layer[layer]
            )

    query = query_options

    @staticmethod
    def _accepted_prefix(
        option: LayerContextualContinuationOption,
        verified_prefix: tuple[int, ...],
    ) -> tuple[int, int]:
        verified = min(option.width, len(verified_prefix))
        accepted = 0
        for expected, observed in zip(
            option.target_tail[:verified], verified_prefix[:verified]
        ):
            if expected != observed:
                break
            accepted += 1
        return accepted, verified

    def settle_verified_prefix(
        self,
        transaction: LayerContextualContinuationTransaction,
        verified_prefix: Sequence[int] | None = None,
        options: Sequence[LayerContextualContinuationOption] = (),
        *,
        target_tail: Sequence[int] | None = None,
        selected_options: Sequence[LayerContextualContinuationOption] | None = None,
    ) -> LayerContextualContinuationMetrics:
        """Atomically capture all layers and settle verifier feedback once.

        ``verified_prefix`` is itself the target-confirmed tail to learn.  The
        optional queried options are scored against that prefix; callers never
        provide or forge accepted-token counts.
        """

        self._validate_transaction(transaction)
        if verified_prefix is None:
            if target_tail is None:
                raise TypeError("verified_prefix or target_tail is required")
            verified_prefix = target_tail
        elif target_tail is not None:
            raise ValueError("provide only one of verified_prefix and target_tail")
        tail = _tail(verified_prefix)
        if selected_options is not None:
            if options:
                raise ValueError("provide only one of options and selected_options")
            options = selected_options
        if isinstance(options, (str, bytes, bytearray)) or not isinstance(
            options, Sequence
        ):
            raise TypeError("options must be a sequence")
        selected = tuple(options)
        if any(
            not isinstance(option, LayerContextualContinuationOption)
            for option in selected
        ):
            raise TypeError("options contains a non-layer-continuation option")
        if len({option.cell_sha256 for option in selected}) != len(selected):
            raise ValueError("options contains a duplicate cell")
        transaction_layers = set(transaction.layers)
        for option in selected:
            if option.transaction_sha256 != transaction.transaction_sha256:
                raise LayerContextualContinuationIdentityError(
                    "option belongs to another boundary transaction"
                )
            if (
                option.layer not in transaction_layers
                or option.known_token != transaction.known_token
            ):
                raise LayerContextualContinuationIdentityError(
                    "option scope differs from its boundary transaction"
                )
        outcome = {
            "options": [
                {
                    "cell_sha256": option.cell_sha256,
                    "layer": option.layer,
                    "target_tail": list(option.target_tail),
                }
                for option in sorted(
                    selected,
                    key=lambda item: (item.layer, item.cell_sha256),
                )
            ],
            "transaction_sha256": transaction.transaction_sha256,
            "verified_prefix": list(tail),
        }
        outcome_sha256 = _sha256_document(outcome)

        with _exclusive_state_lock(self.state_path):
            self._reload(required=self._file_signature is not None)
            receipts = {
                receipt.transaction_sha256: receipt for receipt in self._state.receipts
            }
            previous_receipt = receipts.get(transaction.transaction_sha256)
            if previous_receipt is not None:
                if previous_receipt.outcome_sha256 != outcome_sha256:
                    raise LayerContextualContinuationConflictError(
                        "transaction was already settled with another outcome"
                    )
                return self._metrics(self._state)
            if len(receipts) >= self.max_receipts:
                raise LayerContextualContinuationCapacityError(
                    "layer continuation receipt capacity is exhausted"
                )

            state = self._state
            clock = _bounded_add(state.clock, 1)
            cells = {cell.content_sha256: cell for cell in state.cells}
            layer_counters = dict(state.layer_counters)
            global_counter = state.counter

            feedback_rows: list[
                tuple[LayerContextualContinuationOption, _Cell, int, int]
            ] = []
            for option in selected:
                cell = cells.get(option.cell_sha256)
                if cell is None:
                    raise LayerContextualContinuationIntegrityError(
                        "option references an unknown layer continuation cell"
                    )
                if (
                    cell.key.layer != option.layer
                    or cell.key.known_token != option.known_token
                    or cell.target_tail != option.target_tail
                ):
                    raise LayerContextualContinuationIntegrityError(
                        "option snapshot differs from its persistent cell"
                    )
                accepted, verified = self._accepted_prefix(option, tail)
                feedback_rows.append((option, cell, accepted, verified))

            for option, cell, accepted, verified in feedback_rows:
                verified_counts = list(cell.position_verified)
                hit_counts = list(cell.position_hits)
                for index in range(verified):
                    verified_counts[index] = _bounded_add(verified_counts[index], 1)
                    if index < accepted:
                        hit_counts[index] = _bounded_add(hit_counts[index], 1)
                cells[cell.content_sha256] = replace(
                    cell,
                    position_verified=tuple(verified_counts),
                    position_hits=tuple(hit_counts),
                    last_used_clock=clock,
                )
                layer_counters[option.layer] = _counter_for_feedback(
                    layer_counters[option.layer],
                    verified_tokens=verified,
                    accepted_tokens=accepted,
                )
                global_counter = _counter_for_feedback(
                    global_counter,
                    verified_tokens=verified,
                    accepted_tokens=accepted,
                )

            for key in transaction.keys:
                incoming = _cell_for_capture(
                    key,
                    tail,
                    identity_sha256=self.identity.identity_sha256,
                    clock=clock,
                )
                previous = cells.get(incoming.content_sha256)
                if previous is None:
                    cells[incoming.content_sha256] = incoming
                else:
                    if previous.key != key or previous.target_tail != tail:
                        raise LayerContextualContinuationIntegrityError(
                            "one content address names two continuation cells"
                        )
                    cells[incoming.content_sha256] = replace(
                        previous,
                        support=_bounded_add(previous.support, 1),
                        last_used_clock=clock,
                    )
                layer_counters[key.layer] = _counter_for_capture(
                    layer_counters[key.layer]
                )
                global_counter = _counter_for_capture(global_counter)

            evicted = 0
            while len(cells) > self.max_cells:
                victim = min(
                    cells.values(),
                    key=lambda cell: (
                        cell.value_units,
                        cell.last_used_clock,
                        cell.support,
                        cell.key.layer,
                        cell.content_sha256,
                    ),
                )
                del cells[victim.content_sha256]
                evicted += 1

            receipts[transaction.transaction_sha256] = _Receipt(
                transaction_sha256=transaction.transaction_sha256,
                outcome_sha256=outcome_sha256,
                settled_clock=clock,
            )

            next_state = _State(
                identity=self.identity,
                max_cells=self.max_cells,
                max_receipts=self.max_receipts,
                clock=clock,
                settlements=_bounded_add(state.settlements, 1),
                evictions=_bounded_add(state.evictions, evicted),
                counter=global_counter,
                layer_counters=tuple(
                    (layer, layer_counters[layer]) for layer in self.identity.layers
                ),
                cells=tuple(
                    sorted(cells.values(), key=lambda cell: cell.content_sha256)
                ),
                receipts=tuple(
                    sorted(
                        receipts.values(),
                        key=lambda receipt: receipt.transaction_sha256,
                    )
                ),
            )
            self._publish(next_state)
            return self._metrics(next_state)

    settle = settle_verified_prefix

    def refresh(self) -> LayerContextualContinuationMetrics:
        with self._lock:
            self._refresh_if_changed()
            return self._metrics(self._state)

    def metrics(self) -> LayerContextualContinuationMetrics:
        with self._lock:
            self._refresh_if_changed()
            return self._metrics(self._state)

    def _metrics(self, state: _State) -> LayerContextualContinuationMetrics:
        cells_by_layer: dict[int, list[_Cell]] = {
            layer: [] for layer in state.identity.layers
        }
        for cell in state.cells:
            cells_by_layer[cell.key.layer].append(cell)
        layer_metrics: dict[str, LayerContextualContinuationLayerMetrics] = {}
        for layer, persisted_counter in state.layer_counters:
            cells = cells_by_layer[layer]
            process_delta = self._process_query_layer_counters[layer]
            counter = _merge_counters(persisted_counter, process_delta)
            layer_metrics[str(layer)] = LayerContextualContinuationLayerMetrics(
                **counter.to_record(),
                crystal_bank_cells=len(cells),
                crystal_bank_support=sum(cell.support for cell in cells),
                persisted_crystal_queries=persisted_counter.crystal_queries,
                persisted_crystal_query_hits=(persisted_counter.crystal_query_hits),
                persisted_crystal_option_calls=(persisted_counter.crystal_option_calls),
                persisted_crystal_proposed_tokens=(
                    persisted_counter.crystal_proposed_tokens
                ),
                process_crystal_queries_delta=process_delta.crystal_queries,
                process_crystal_query_hits_delta=(process_delta.crystal_query_hits),
                process_crystal_option_calls_delta=(process_delta.crystal_option_calls),
                process_crystal_proposed_tokens_delta=(
                    process_delta.crystal_proposed_tokens
                ),
            )
        persisted_counter = state.counter
        process_delta = self._process_query_counter
        counter = _merge_counters(persisted_counter, process_delta)
        return LayerContextualContinuationMetrics(
            schema=LAYER_CONTEXTUAL_CONTINUATION_METRICS_SCHEMA,
            identity_sha256=state.identity.identity_sha256,
            state_sha256=_sha256_document(state.to_record()),
            max_cells=state.max_cells,
            max_receipts=state.max_receipts,
            clock=state.clock,
            settlements=state.settlements,
            evictions=state.evictions,
            receipt_count=len(state.receipts),
            crystal_enabled=True,
            **counter.to_record(),
            crystal_bank_cells=len(state.cells),
            crystal_bank_support=sum(cell.support for cell in state.cells),
            query_metrics_scope="persisted_plus_process_local_delta",
            persisted_crystal_queries=persisted_counter.crystal_queries,
            persisted_crystal_query_hits=persisted_counter.crystal_query_hits,
            persisted_crystal_option_calls=persisted_counter.crystal_option_calls,
            persisted_crystal_proposed_tokens=(
                persisted_counter.crystal_proposed_tokens
            ),
            process_crystal_queries_delta=process_delta.crystal_queries,
            process_crystal_query_hits_delta=process_delta.crystal_query_hits,
            process_crystal_option_calls_delta=process_delta.crystal_option_calls,
            process_crystal_proposed_tokens_delta=(
                process_delta.crystal_proposed_tokens
            ),
            layers=MappingProxyType(layer_metrics),
        )


__all__ = [
    "DEFAULT_LAYER_CONTEXTUAL_MAX_RECEIPTS",
    "DEFAULT_LAYER_CONTEXTUAL_SKETCH_DIM",
    "LAYER_CONTEXTUAL_CONTINUATION_ENVELOPE_SCHEMA",
    "LAYER_CONTEXTUAL_CONTINUATION_IDENTITY_SCHEMA",
    "LAYER_CONTEXTUAL_CONTINUATION_METRICS_SCHEMA",
    "LAYER_CONTEXTUAL_CONTINUATION_STATE_SCHEMA",
    "LAYER_CONTEXTUAL_PROJECTION_ABI",
    "LAYER_CONTEXTUAL_STAGE",
    "MAX_CONTINUATION_TOKENS",
    "LayerContextualContinuationBank",
    "LayerContextualContinuationCapacityError",
    "LayerContextualContinuationConflictError",
    "LayerContextualContinuationError",
    "LayerContextualContinuationIdentity",
    "LayerContextualContinuationIdentityError",
    "LayerContextualContinuationIntegrityError",
    "LayerContextualContinuationKey",
    "LayerContextualContinuationLayerMetrics",
    "LayerContextualContinuationMetrics",
    "LayerContextualContinuationOption",
    "LayerContextualContinuationTransaction",
]
