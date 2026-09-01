"""Persistent target-confirmed continuations keyed by Qwen hidden state.

The bank turns one ``[1, 1, H]`` target hidden state into a deterministic
256-dimensional Rademacher sketch, normalises it, and stores only its signed
Q8 representation.  A single known token is part of every key.  The bank
never stores prompt text or a prompt-token sequence.

Queries are read-only.  Target-confirmed captures and candidate feedback are
committed together under one process/thread lock and one atomic filesystem
replace, so the file is always either the previous complete generation or the
next complete generation.
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
import stat
import tempfile
import threading
from typing import Any, Iterator

import torch


CONTEXTUAL_CONTINUATION_IDENTITY_SCHEMA = (
    "immer.qwen3.8-contextual-continuation-identity/v1"
)
CONTEXTUAL_CONTINUATION_KEY_SCHEMA = "immer.qwen3.8-contextual-q8-key/v1"
CONTEXTUAL_CONTINUATION_CELL_SCHEMA = "immer.qwen3.8-contextual-cell/v1"
CONTEXTUAL_CONTINUATION_STATE_SCHEMA = "immer.qwen3.8-contextual-bank-state/v1"
CONTEXTUAL_CONTINUATION_ENVELOPE_SCHEMA = (
    "immer.qwen3.8-contextual-bank-envelope/v1"
)
CONTEXTUAL_PROJECTION_ABI = (
    "immer.qwen3.8/rademacher-shake256-lsb-f64-l2-q8-256/v1"
)

CONTEXTUAL_KEY_DIMENSIONS = 256
MAX_CONTINUATION_TOKENS = 15

_MAX_COUNTER = (1 << 63) - 1
_MAX_TOKEN_ID = (1 << 32) - 1
_MAX_HIDDEN_WIDTH = 1 << 16
_MAX_CELLS = 1 << 16
_MAX_STATE_BYTES = 256 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_HEX = frozenset("0123456789abcdef")


class ContextualContinuationError(RuntimeError):
    """The contextual bank input or persistent state is invalid."""


class ContextualContinuationIntegrityError(ContextualContinuationError):
    """The persistent bank is malformed, unstable, or hash-inconsistent."""


class ContextualContinuationIdentityError(ContextualContinuationError):
    """A key or state belongs to another immutable runtime identity."""


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
        raise ContextualContinuationIntegrityError(
            "contextual continuation state is not canonical JSON"
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


def _token(value: object, field: str) -> int:
    return _uint(value, field=field, maximum=_MAX_TOKEN_ID)


def _tail(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise TypeError("target_tail must be a sequence of token IDs")
    result = tuple(value)
    if not 1 <= len(result) <= MAX_CONTINUATION_TOKENS:
        raise ValueError(
            f"target_tail must contain 1..{MAX_CONTINUATION_TOKENS} tokens"
        )
    for token_id in result:
        _token(token_id, "target-tail token")
    return result


def _q8_vector(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise TypeError("Q8 key must be an integer sequence")
    result = tuple(value)
    if len(result) != CONTEXTUAL_KEY_DIMENSIONS:
        raise ValueError(
            f"Q8 key must have {CONTEXTUAL_KEY_DIMENSIONS} dimensions"
        )
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or not -127 <= item <= 127
        for item in result
    ):
        raise ValueError("Q8 key contains a value outside [-127, 127]")
    norm_sq = sum(item * item for item in result)
    # A rounded unit vector scaled by 127 has norm very close to 127.  This
    # range rejects unnormalised or all-zero persisted vectors while allowing
    # the worst possible component-wise Q8 rounding error.
    if not 112 * 112 <= norm_sq <= 144 * 144:
        raise ValueError("Q8 key is not a normalised 256D vector")
    return result


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContextualContinuationIntegrityError(
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
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(
        os, "O_NOFOLLOW", 0
    )
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ContextualContinuationIntegrityError(
            "contextual continuation state cannot be opened"
        ) from exc
    try:
        before = os.fstat(descriptor)
        linked_before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(linked_before.st_mode)
            or not _same_inode(before, linked_before)
        ):
            raise ContextualContinuationIntegrityError(
                "contextual continuation state must be a stable regular file"
            )
        if before.st_size < 0 or before.st_size > _MAX_STATE_BYTES:
            raise ContextualContinuationIntegrityError(
                "contextual continuation state exceeds its byte limit"
            )
        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, _READ_CHUNK_BYTES))
            if not chunk:
                raise ContextualContinuationIntegrityError(
                    "contextual continuation state was truncated"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ContextualContinuationIntegrityError(
                "contextual continuation state grew while reading"
            )
        after = os.fstat(descriptor)
        linked_after = os.lstat(path)
        if (
            _stable_signature(before) != _stable_signature(after)
            or _stable_signature(linked_before) != _stable_signature(linked_after)
            or not _same_inode(after, linked_after)
        ):
            raise ContextualContinuationIntegrityError(
                "contextual continuation state changed while reading"
            )
        return b"".join(chunks), _stable_signature(after)
    except OSError as exc:
        raise ContextualContinuationIntegrityError(
            "contextual continuation state cannot be read safely"
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
        raise ContextualContinuationIntegrityError(
            "contextual continuation state parent is unavailable"
        ) from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise ContextualContinuationIntegrityError(
            "contextual continuation state parent must be a directory"
        )
    return parent, value


@contextmanager
def _exclusive_state_lock(path: Path) -> Iterator[None]:
    """Serialise one read-modify-replace transaction across threads/processes."""

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
            raise ContextualContinuationIntegrityError(
                "contextual continuation lock is unavailable"
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
                raise ContextualContinuationIntegrityError(
                    "contextual continuation lock is not a stable regular file"
                )
            yield
            parent_after = os.lstat(parent)
            linked_after = os.lstat(lock_path)
            if (
                not _same_inode(opened, linked_after)
                or not _same_inode(parent_before, parent_after)
            ):
                raise ContextualContinuationIntegrityError(
                    "contextual continuation lock changed during transaction"
                )
        except OSError as exc:
            raise ContextualContinuationIntegrityError(
                "contextual continuation lock failed"
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
        raise ContextualContinuationIntegrityError(
            "contextual continuation destination is unavailable"
        ) from exc
    if destination is not None and (
        stat.S_ISLNK(destination.st_mode) or not stat.S_ISREG(destination.st_mode)
    ):
        raise ContextualContinuationIntegrityError(
            "contextual continuation destination must be a regular file"
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
            raise ContextualContinuationIntegrityError(
                "contextual continuation parent changed before publication"
            )
        try:
            current = os.lstat(path)
        except FileNotFoundError:
            current = None
        if current is not None and (
            stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode)
        ):
            raise ContextualContinuationIntegrityError(
                "contextual continuation destination changed type"
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
            raise ContextualContinuationIntegrityError(
                "published contextual continuation state is not regular"
            )
        return _stable_signature(published)
    except ContextualContinuationError:
        raise
    except OSError as exc:
        raise ContextualContinuationIntegrityError(
            "contextual continuation state could not be published atomically"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True, slots=True)
class ContextualContinuationIdentity:
    """Immutable identity of the only runtime allowed to use one bank."""

    runtime_sha256: str
    model_sha256: str
    q4_sha256: str
    tokenizer_sha256: str
    hidden_width: int
    projection_abi: str = CONTEXTUAL_PROJECTION_ABI

    def __post_init__(self) -> None:
        for field in (
            "runtime_sha256",
            "model_sha256",
            "q4_sha256",
            "tokenizer_sha256",
        ):
            _digest(getattr(self, field), field)
        _uint(
            self.hidden_width,
            field="hidden_width",
            positive=True,
            maximum=_MAX_HIDDEN_WIDTH,
        )
        if self.projection_abi != CONTEXTUAL_PROJECTION_ABI:
            raise ValueError("projection ABI is not implemented by this runtime")

    @property
    def identity_sha256(self) -> str:
        return _sha256_document(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "hidden_width": self.hidden_width,
            "model_sha256": self.model_sha256,
            "projection_abi": self.projection_abi,
            "q4_sha256": self.q4_sha256,
            "runtime_sha256": self.runtime_sha256,
            "schema": CONTEXTUAL_CONTINUATION_IDENTITY_SCHEMA,
            "tokenizer_sha256": self.tokenizer_sha256,
        }

    @classmethod
    def from_record(cls, value: object) -> "ContextualContinuationIdentity":
        if not isinstance(value, Mapping) or set(value) != {
            "hidden_width",
            "model_sha256",
            "projection_abi",
            "q4_sha256",
            "runtime_sha256",
            "schema",
            "tokenizer_sha256",
        }:
            raise ContextualContinuationIntegrityError(
                "contextual continuation identity fields are invalid"
            )
        if value["schema"] != CONTEXTUAL_CONTINUATION_IDENTITY_SCHEMA:
            raise ContextualContinuationIntegrityError(
                "contextual continuation identity schema is invalid"
            )
        try:
            return cls(
                runtime_sha256=value["runtime_sha256"],
                model_sha256=value["model_sha256"],
                q4_sha256=value["q4_sha256"],
                tokenizer_sha256=value["tokenizer_sha256"],
                hidden_width=value["hidden_width"],
                projection_abi=value["projection_abi"],
            )
        except (TypeError, ValueError) as exc:
            raise ContextualContinuationIntegrityError(
                "contextual continuation identity values are invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class ContextualKey:
    """One identity- and known-token-bound normalised Q8 hidden-state key."""

    identity_sha256: str
    known_token: int
    q8: tuple[int, ...]
    key_sha256: str

    def __post_init__(self) -> None:
        _digest(self.identity_sha256, "key identity_sha256")
        _token(self.known_token, "known_token")
        q8 = _q8_vector(self.q8)
        object.__setattr__(self, "q8", q8)
        expected = _sha256_document(self._address_record())
        if self.key_sha256 != expected:
            raise ValueError("contextual key SHA-256 mismatch")

    @property
    def norm_sq(self) -> int:
        return sum(value * value for value in self.q8)

    def _address_record(self) -> dict[str, object]:
        return {
            "identity_sha256": self.identity_sha256,
            "known_token": self.known_token,
            "q8": list(self.q8),
            "schema": CONTEXTUAL_CONTINUATION_KEY_SCHEMA,
        }

    @classmethod
    def create(
        cls,
        *,
        identity_sha256: str,
        known_token: int,
        q8: Sequence[int],
    ) -> "ContextualKey":
        vector = _q8_vector(q8)
        address = {
            "identity_sha256": identity_sha256,
            "known_token": known_token,
            "q8": list(vector),
            "schema": CONTEXTUAL_CONTINUATION_KEY_SCHEMA,
        }
        return cls(
            identity_sha256=identity_sha256,
            known_token=known_token,
            q8=vector,
            key_sha256=_sha256_document(address),
        )


@dataclass(frozen=True, slots=True)
class ContextualCapture:
    """One caller-derived target-confirmed continuation boundary."""

    key: ContextualKey
    target_tail: tuple[int, ...]
    boundary_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, ContextualKey):
            raise TypeError("capture key must be a ContextualKey")
        object.__setattr__(self, "target_tail", _tail(self.target_tail))
        _uint(self.boundary_index, field="boundary_index")


@dataclass(frozen=True, slots=True)
class ContextualCandidateFeedback:
    """Target-verifier feedback for one previously queried cell.

    ``accepted_tokens`` is the exact matching prefix length.  Every position
    below ``verified_tokens`` increments its verified count, and positions
    below ``accepted_tokens`` also increment their hit count.
    """

    cell_sha256: str
    accepted_tokens: int
    verified_tokens: int

    def __post_init__(self) -> None:
        _digest(self.cell_sha256, "feedback cell_sha256")
        accepted = _uint(
            self.accepted_tokens,
            field="accepted_tokens",
            maximum=MAX_CONTINUATION_TOKENS,
        )
        verified = _uint(
            self.verified_tokens,
            field="verified_tokens",
            positive=True,
            maximum=MAX_CONTINUATION_TOKENS,
        )
        if accepted > verified:
            raise ValueError("accepted_tokens exceeds verified_tokens")


@dataclass(frozen=True, slots=True)
class _ContextualCell:
    content_sha256: str
    key: ContextualKey
    target_tail: tuple[int, ...]
    support: int
    position_verified: tuple[int, ...]
    position_hits: tuple[int, ...]
    created_clock: int
    last_used_clock: int

    def __post_init__(self) -> None:
        _digest(self.content_sha256, "cell content_sha256")
        if not isinstance(self.key, ContextualKey):
            raise ValueError("cell key is invalid")
        tail = _tail(self.target_tail)
        verified = tuple(self.position_verified)
        hits = tuple(self.position_hits)
        if len(verified) != len(tail) or len(hits) != len(tail):
            raise ValueError("cell position counters do not match its tail")
        for index, (verified_count, hit_count) in enumerate(zip(verified, hits)):
            _uint(verified_count, field=f"position_verified[{index}]")
            _uint(hit_count, field=f"position_hits[{index}]")
            if hit_count > verified_count:
                raise ValueError("cell position hits exceed verified observations")
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
                "key_sha256": self.key.key_sha256,
                "schema": CONTEXTUAL_CONTINUATION_CELL_SCHEMA,
                "target_tail": list(self.target_tail),
            }
        )

    @property
    def value_units(self) -> int:
        """Integer target-confirmed value used by deterministic eviction."""

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
            "key": self.key._address_record() | {"key_sha256": self.key.key_sha256},
            "last_used_clock": self.last_used_clock,
            "position_hits": list(self.position_hits),
            "position_verified": list(self.position_verified),
            "schema": CONTEXTUAL_CONTINUATION_CELL_SCHEMA,
            "support": self.support,
            "target_tail": list(self.target_tail),
        }

    @classmethod
    def from_record(cls, value: object) -> "_ContextualCell":
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
            raise ContextualContinuationIntegrityError(
                "contextual cell fields are invalid"
            )
        if value["schema"] != CONTEXTUAL_CONTINUATION_CELL_SCHEMA:
            raise ContextualContinuationIntegrityError(
                "contextual cell schema is invalid"
            )
        key_record = value["key"]
        if not isinstance(key_record, Mapping) or set(key_record) != {
            "identity_sha256",
            "key_sha256",
            "known_token",
            "q8",
            "schema",
        }:
            raise ContextualContinuationIntegrityError(
                "contextual cell key fields are invalid"
            )
        if key_record["schema"] != CONTEXTUAL_CONTINUATION_KEY_SCHEMA:
            raise ContextualContinuationIntegrityError(
                "contextual cell key schema is invalid"
            )
        try:
            key = ContextualKey(
                identity_sha256=key_record["identity_sha256"],
                known_token=key_record["known_token"],
                q8=tuple(key_record["q8"]),
                key_sha256=key_record["key_sha256"],
            )
            return cls(
                content_sha256=value["content_sha256"],
                key=key,
                target_tail=tuple(value["target_tail"]),
                support=value["support"],
                position_verified=tuple(value["position_verified"]),
                position_hits=tuple(value["position_hits"]),
                created_clock=value["created_clock"],
                last_used_clock=value["last_used_clock"],
            )
        except (TypeError, ValueError) as exc:
            raise ContextualContinuationIntegrityError(
                "contextual cell values are invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class _ContextualState:
    identity: ContextualContinuationIdentity
    max_cells: int
    clock: int = 0
    settlements: int = 0
    capture_count: int = 0
    feedback_count: int = 0
    evictions: int = 0
    cells: tuple[_ContextualCell, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.identity, ContextualContinuationIdentity):
            raise ValueError("contextual state identity is invalid")
        _uint(self.max_cells, field="max_cells", positive=True, maximum=_MAX_CELLS)
        for field in (
            "clock",
            "settlements",
            "capture_count",
            "feedback_count",
            "evictions",
        ):
            _uint(getattr(self, field), field=field)
        cells = tuple(self.cells)
        if len(cells) > self.max_cells:
            raise ValueError("contextual state exceeds max_cells")
        if tuple(sorted(cells, key=lambda cell: cell.content_sha256)) != cells:
            raise ValueError("contextual cells are not canonically ordered")
        if len({cell.content_sha256 for cell in cells}) != len(cells):
            raise ValueError("contextual state contains duplicate cells")
        if any(
            cell.key.identity_sha256 != self.identity.identity_sha256
            or cell.last_used_clock > self.clock
            for cell in cells
        ):
            raise ValueError("contextual cell identity or clock is invalid")
        object.__setattr__(self, "cells", cells)

    def to_record(self) -> dict[str, object]:
        return {
            "capture_count": self.capture_count,
            "cells": [cell.to_record() for cell in self.cells],
            "clock": self.clock,
            "evictions": self.evictions,
            "feedback_count": self.feedback_count,
            "identity": self.identity.to_record(),
            "identity_sha256": self.identity.identity_sha256,
            "max_cells": self.max_cells,
            "schema": CONTEXTUAL_CONTINUATION_STATE_SCHEMA,
            "settlements": self.settlements,
        }

    @classmethod
    def from_record(cls, value: object) -> "_ContextualState":
        if not isinstance(value, Mapping) or set(value) != {
            "capture_count",
            "cells",
            "clock",
            "evictions",
            "feedback_count",
            "identity",
            "identity_sha256",
            "max_cells",
            "schema",
            "settlements",
        }:
            raise ContextualContinuationIntegrityError(
                "contextual continuation state fields are invalid"
            )
        if value["schema"] != CONTEXTUAL_CONTINUATION_STATE_SCHEMA:
            raise ContextualContinuationIntegrityError(
                "contextual continuation state schema is invalid"
            )
        try:
            identity = ContextualContinuationIdentity.from_record(value["identity"])
            if value["identity_sha256"] != identity.identity_sha256:
                raise ValueError("contextual state identity SHA-256 mismatch")
            return cls(
                identity=identity,
                max_cells=value["max_cells"],
                clock=value["clock"],
                settlements=value["settlements"],
                capture_count=value["capture_count"],
                feedback_count=value["feedback_count"],
                evictions=value["evictions"],
                cells=tuple(
                    _ContextualCell.from_record(cell) for cell in value["cells"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ContextualContinuationIntegrityError(
                "contextual continuation state values are invalid"
            ) from exc

    def to_bytes(self) -> bytes:
        body = self.to_record()
        envelope = {
            "body": body,
            "body_sha256": _sha256_document(body),
            "schema": CONTEXTUAL_CONTINUATION_ENVELOPE_SCHEMA,
        }
        encoded = _canonical_json(envelope)
        if len(encoded) > _MAX_STATE_BYTES:
            raise ContextualContinuationIntegrityError(
                "contextual continuation state exceeds its byte limit"
            )
        return encoded

    @classmethod
    def from_bytes(cls, value: bytes) -> "_ContextualState":
        try:
            document = json.loads(value, object_pairs_hook=_json_no_duplicates)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContextualContinuationIntegrityError(
                "contextual continuation state is not valid JSON"
            ) from exc
        if _canonical_json(document) != value:
            raise ContextualContinuationIntegrityError(
                "contextual continuation state is not canonical JSON"
            )
        if not isinstance(document, Mapping) or set(document) != {
            "body",
            "body_sha256",
            "schema",
        }:
            raise ContextualContinuationIntegrityError(
                "contextual continuation envelope fields are invalid"
            )
        if document["schema"] != CONTEXTUAL_CONTINUATION_ENVELOPE_SCHEMA:
            raise ContextualContinuationIntegrityError(
                "contextual continuation envelope schema is invalid"
            )
        if not _is_sha256(document["body_sha256"]):
            raise ContextualContinuationIntegrityError(
                "contextual continuation body digest is invalid"
            )
        if document["body_sha256"] != _sha256_document(document["body"]):
            raise ContextualContinuationIntegrityError(
                "contextual continuation body SHA-256 mismatch"
            )
        return cls.from_record(document["body"])


@dataclass(frozen=True, slots=True)
class ContextualCandidate:
    """One unique target tail ranked against a read-only query key."""

    cell_sha256: str
    target_tail: tuple[int, ...]
    cosine: float
    runner_up_cosine: float | None
    margin: float | None
    support: int
    position_verified: tuple[int, ...]
    position_hits: tuple[int, ...]
    collapsed_cells: int

    @property
    def width(self) -> int:
        return len(self.target_tail)

    @property
    def tail(self) -> tuple[int, ...]:
        """Concise integration alias for :attr:`target_tail`."""

        return self.target_tail

    def to_dict(self) -> dict[str, object]:
        return {
            "cell_sha256": self.cell_sha256,
            "collapsed_cells": self.collapsed_cells,
            "cosine": self.cosine,
            "margin": self.margin,
            "position_hits": list(self.position_hits),
            "position_verified": list(self.position_verified),
            "runner_up_cosine": self.runner_up_cosine,
            "support": self.support,
            "target_tail": list(self.target_tail),
            "width": self.width,
        }


@dataclass(frozen=True, slots=True)
class ContextualContinuationMetrics:
    identity_sha256: str
    state_sha256: str
    max_cells: int
    cell_count: int
    unique_tail_count: int
    clock: int
    settlements: int
    capture_count: int
    feedback_count: int
    evictions: int
    support: int
    verified_positions: int
    hit_positions: int

    def to_dict(self) -> dict[str, object]:
        return {
            "capture_count": self.capture_count,
            "cell_count": self.cell_count,
            "clock": self.clock,
            "evictions": self.evictions,
            "feedback_count": self.feedback_count,
            "hit_positions": self.hit_positions,
            "identity_sha256": self.identity_sha256,
            "max_cells": self.max_cells,
            "settlements": self.settlements,
            "state_sha256": self.state_sha256,
            "support": self.support,
            "unique_tail_count": self.unique_tail_count,
            "verified_positions": self.verified_positions,
        }


@lru_cache(maxsize=8)
def _rademacher_projection(
    identity_sha256: str,
    hidden_width: int,
) -> torch.Tensor:
    """Build one ABI-pinned Rademacher matrix without ambient RNG state."""

    material = _canonical_json(
        {
            "hidden_width": hidden_width,
            "identity_sha256": identity_sha256,
            "projection_abi": CONTEXTUAL_PROJECTION_ABI,
        }
    )
    count = hidden_width * CONTEXTUAL_KEY_DIMENSIONS
    raw = hashlib.shake_256(material).digest((count + 7) // 8)
    octets = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    shifts = torch.arange(8, dtype=torch.uint8)
    bits = torch.bitwise_and(
        torch.bitwise_right_shift(octets[:, None], shifts[None, :]),
        1,
    ).reshape(-1)[:count]
    projection = bits.to(dtype=torch.float64).mul_(2.0).sub_(1.0)
    projection = projection.reshape(hidden_width, CONTEXTUAL_KEY_DIMENSIONS)
    projection.mul_(1.0 / math.sqrt(CONTEXTUAL_KEY_DIMENSIONS))
    return projection.contiguous()


def _normalised_q8(
    hidden: torch.Tensor,
    *,
    identity: ContextualContinuationIdentity,
) -> tuple[int, ...]:
    if not isinstance(hidden, torch.Tensor):
        raise TypeError("hidden must be a torch.Tensor")
    if tuple(hidden.shape) != (1, 1, identity.hidden_width):
        raise ValueError(
            "hidden must have exact shape "
            f"[1, 1, {identity.hidden_width}]"
        )
    if not hidden.dtype.is_floating_point:
        raise TypeError("hidden must have a floating-point dtype")
    vector = hidden.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("hidden contains a non-finite value")
    projection = _rademacher_projection(
        identity.identity_sha256,
        identity.hidden_width,
    )
    sketch = torch.matmul(vector, projection)
    norm = torch.linalg.vector_norm(sketch)
    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
        raise ValueError("hidden has no projectable norm")
    quantised = torch.round(sketch.div(norm).mul(127.0)).to(dtype=torch.int16)
    return _q8_vector(tuple(int(value) for value in quantised.tolist()))


def _cell_for_capture(
    capture: ContextualCapture,
    *,
    clock: int,
) -> _ContextualCell:
    address = {
        "key_sha256": capture.key.key_sha256,
        "schema": CONTEXTUAL_CONTINUATION_CELL_SCHEMA,
        "target_tail": list(capture.target_tail),
    }
    return _ContextualCell(
        content_sha256=_sha256_document(address),
        key=capture.key,
        target_tail=capture.target_tail,
        support=1,
        position_verified=(0,) * len(capture.target_tail),
        position_hits=(0,) * len(capture.target_tail),
        created_clock=clock,
        last_used_clock=clock,
    )


class ContextualContinuationBank:
    """Bounded persistent nearest-neighbour bank for confirmed continuations."""

    def __init__(
        self,
        state_path: str | os.PathLike[str],
        identity: ContextualContinuationIdentity,
        *,
        max_cells: int = 4096,
    ) -> None:
        if not isinstance(identity, ContextualContinuationIdentity):
            raise TypeError("identity must be a ContextualContinuationIdentity")
        self.state_path = Path(state_path)
        if not self.state_path.name:
            raise ValueError("state_path must name a file")
        _uint(max_cells, field="max_cells", positive=True, maximum=_MAX_CELLS)
        self.identity = identity
        self.max_cells = max_cells
        self._lock = _thread_lock(self.state_path)
        self._file_signature: tuple[int, int, int, int, int] | None = None
        self._state = _ContextualState(identity=identity, max_cells=max_cells)
        self._index: dict[
            int,
            tuple[tuple[_ContextualCell, ...], torch.Tensor, torch.Tensor],
        ] = {}
        with self._lock:
            self._reload(required=False)

    def _validate_state(self, state: _ContextualState) -> None:
        if state.identity.identity_sha256 != self.identity.identity_sha256:
            raise ContextualContinuationIdentityError(
                "contextual bank belongs to a different runtime identity"
            )
        if state.max_cells != self.max_cells:
            raise ContextualContinuationIdentityError(
                "contextual bank max_cells differs from its persistent identity"
            )

    def _reload(self, *, required: bool) -> None:
        try:
            raw, signature = _stable_regular_bytes(self.state_path)
        except FileNotFoundError:
            if required:
                raise ContextualContinuationIntegrityError(
                    "contextual continuation state disappeared"
                )
            self._file_signature = None
            self._state = _ContextualState(
                identity=self.identity,
                max_cells=self.max_cells,
            )
            self._rebuild_index()
            return
        state = _ContextualState.from_bytes(raw)
        self._validate_state(state)
        self._state = state
        self._file_signature = signature
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        grouped: dict[int, list[_ContextualCell]] = {}
        for cell in self._state.cells:
            grouped.setdefault(cell.key.known_token, []).append(cell)
        index: dict[
            int,
            tuple[tuple[_ContextualCell, ...], torch.Tensor, torch.Tensor],
        ] = {}
        for known_token, members in grouped.items():
            cells = tuple(members)
            matrix = torch.tensor(
                [cell.key.q8 for cell in cells],
                dtype=torch.float32,
                device="cpu",
            ).contiguous()
            norms = torch.linalg.vector_norm(matrix, dim=1)
            index[known_token] = (cells, matrix, norms)
        self._index = index

    def _refresh_if_changed(self) -> None:
        try:
            linked = os.lstat(self.state_path)
        except FileNotFoundError:
            if self._file_signature is not None:
                raise ContextualContinuationIntegrityError(
                    "contextual continuation state disappeared"
                )
            return
        if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
            raise ContextualContinuationIntegrityError(
                "contextual continuation state must be a regular file"
            )
        if _stable_signature(linked) == self._file_signature:
            return
        # Atomic publication can replace the path between lstat and open.  A
        # bounded retry observes either complete generation without writing.
        last_error: ContextualContinuationIntegrityError | None = None
        for _ in range(3):
            try:
                self._reload(required=True)
                return
            except ContextualContinuationIntegrityError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    def project(self, hidden: torch.Tensor, known_token: int) -> ContextualKey:
        """Return the deterministic identity/token-bound Q8 projection."""

        known = _token(known_token, "known_token")
        return ContextualKey.create(
            identity_sha256=self.identity.identity_sha256,
            known_token=known,
            q8=_normalised_q8(hidden, identity=self.identity),
        )

    def make_capture(
        self,
        hidden: torch.Tensor,
        known_token: int,
        boundary_index: int,
        target_tail: Sequence[int],
    ) -> ContextualCapture:
        """Project one boundary and attach its caller-confirmed target tail."""

        return ContextualCapture(
            key=self.project(hidden, known_token),
            target_tail=tuple(target_tail),
            boundary_index=boundary_index,
        )

    def query(
        self,
        hidden: torch.Tensor,
        known_token: int,
        *,
        limit: int = 8,
    ) -> tuple[ContextualCandidate, ...]:
        return self.query_key(self.project(hidden, known_token), limit=limit)

    def query_key(
        self,
        key: ContextualKey,
        known_token: int | None = None,
        *,
        limit: int = 8,
    ) -> tuple[ContextualCandidate, ...]:
        """Rank unique tails by Q8 cosine without changing persistent state."""

        if not isinstance(key, ContextualKey):
            raise TypeError("key must be a ContextualKey")
        if key.identity_sha256 != self.identity.identity_sha256:
            raise ContextualContinuationIdentityError(
                "query key belongs to a different runtime identity"
            )
        if known_token is not None and _token(
            known_token, "known_token"
        ) != key.known_token:
            raise ValueError("known_token disagrees with the bound query key")
        _uint(limit, field="limit", positive=True, maximum=256)

        with self._lock:
            self._refresh_if_changed()
            indexed = self._index.get(key.known_token)
        if indexed is None:
            return ()

        cells, matrix, norms = indexed
        query = torch.tensor(key.q8, dtype=torch.float32, device="cpu")
        cosines = torch.mv(matrix, query).div_(
            norms * math.sqrt(key.norm_sq)
        ).tolist()
        grouped: dict[tuple[int, ...], list[tuple[_ContextualCell, float]]] = {}
        for cell, cosine in zip(cells, cosines):
            bounded_cosine = max(-1.0, min(1.0, float(cosine)))
            grouped.setdefault(cell.target_tail, []).append(
                (cell, bounded_cosine)
            )

        ranked: list[
            tuple[
                tuple[int, ...],
                _ContextualCell,
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
        result: list[ContextualCandidate] = []
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
            result.append(
                ContextualCandidate(
                    cell_sha256=representative.content_sha256,
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
        return tuple(result)

    def settle(
        self,
        *,
        captures: Sequence[ContextualCapture] = (),
        feedback: Sequence[ContextualCandidateFeedback] = (),
    ) -> ContextualContinuationMetrics:
        """Atomically apply feedback and target-confirmed captured tails."""

        if isinstance(captures, (str, bytes, bytearray)) or not isinstance(
            captures, Sequence
        ):
            raise TypeError("captures must be a sequence")
        if isinstance(feedback, (str, bytes, bytearray)) or not isinstance(
            feedback, Sequence
        ):
            raise TypeError("feedback must be a sequence")
        captures = tuple(captures)
        feedback = tuple(feedback)
        if any(not isinstance(item, ContextualCapture) for item in captures):
            raise TypeError("captures contains a non-ContextualCapture value")
        if any(
            not isinstance(item, ContextualCandidateFeedback) for item in feedback
        ):
            raise TypeError(
                "feedback contains a non-ContextualCandidateFeedback value"
            )
        if not captures and not feedback:
            return self.metrics()
        for capture in captures:
            if capture.key.identity_sha256 != self.identity.identity_sha256:
                raise ContextualContinuationIdentityError(
                    "capture key belongs to a different runtime identity"
                )

        with _exclusive_state_lock(self.state_path):
            self._reload(required=self._file_signature is not None)
            state = self._state
            clock = _bounded_add(state.clock, 1)
            cells = {cell.content_sha256: cell for cell in state.cells}

            for observation in feedback:
                cell = cells.get(observation.cell_sha256)
                if cell is None:
                    raise ContextualContinuationIntegrityError(
                        "feedback references an unknown contextual cell"
                    )
                if observation.verified_tokens > len(cell.target_tail):
                    raise ValueError(
                        "verified_tokens exceeds the contextual cell tail"
                    )
                verified = list(cell.position_verified)
                hits = list(cell.position_hits)
                for index in range(observation.verified_tokens):
                    verified[index] = _bounded_add(verified[index], 1)
                    if index < observation.accepted_tokens:
                        hits[index] = _bounded_add(hits[index], 1)
                cells[cell.content_sha256] = replace(
                    cell,
                    position_verified=tuple(verified),
                    position_hits=tuple(hits),
                    last_used_clock=clock,
                )

            for capture in captures:
                incoming = _cell_for_capture(capture, clock=clock)
                previous = cells.get(incoming.content_sha256)
                if previous is None:
                    cells[incoming.content_sha256] = incoming
                else:
                    if (
                        previous.key != incoming.key
                        or previous.target_tail != incoming.target_tail
                    ):
                        raise ContextualContinuationIntegrityError(
                            "one content address names two contextual cells"
                        )
                    cells[incoming.content_sha256] = replace(
                        previous,
                        support=_bounded_add(previous.support, 1),
                        last_used_clock=clock,
                    )

            evicted = 0
            while len(cells) > self.max_cells:
                victim = min(
                    cells.values(),
                    key=lambda cell: (
                        cell.value_units,
                        cell.last_used_clock,
                        cell.support,
                        cell.content_sha256,
                    ),
                )
                del cells[victim.content_sha256]
                evicted += 1

            next_state = _ContextualState(
                identity=self.identity,
                max_cells=self.max_cells,
                clock=clock,
                settlements=_bounded_add(state.settlements, 1),
                capture_count=_bounded_add(state.capture_count, len(captures)),
                feedback_count=_bounded_add(state.feedback_count, len(feedback)),
                evictions=_bounded_add(state.evictions, evicted),
                cells=tuple(sorted(cells.values(), key=lambda cell: cell.content_sha256)),
            )
            encoded = next_state.to_bytes()
            signature = _atomic_write(self.state_path, encoded)
            self._state = next_state
            self._file_signature = signature
            self._rebuild_index()
            return self._metrics(next_state)

    def refresh(self) -> ContextualContinuationMetrics:
        """Read a newer atomically published generation, if one exists."""

        with self._lock:
            self._refresh_if_changed()
            return self._metrics(self._state)

    def metrics(self) -> ContextualContinuationMetrics:
        with self._lock:
            self._refresh_if_changed()
            return self._metrics(self._state)

    @staticmethod
    def _metrics(state: _ContextualState) -> ContextualContinuationMetrics:
        cells = state.cells
        return ContextualContinuationMetrics(
            identity_sha256=state.identity.identity_sha256,
            state_sha256=_sha256_document(state.to_record()),
            max_cells=state.max_cells,
            cell_count=len(cells),
            unique_tail_count=len(
                {(cell.key.known_token, cell.target_tail) for cell in cells}
            ),
            clock=state.clock,
            settlements=state.settlements,
            capture_count=state.capture_count,
            feedback_count=state.feedback_count,
            evictions=state.evictions,
            support=sum(cell.support for cell in cells),
            verified_positions=sum(
                sum(cell.position_verified) for cell in cells
            ),
            hit_positions=sum(sum(cell.position_hits) for cell in cells),
        )


__all__ = [
    "CONTEXTUAL_CONTINUATION_ENVELOPE_SCHEMA",
    "CONTEXTUAL_CONTINUATION_IDENTITY_SCHEMA",
    "CONTEXTUAL_CONTINUATION_KEY_SCHEMA",
    "CONTEXTUAL_CONTINUATION_STATE_SCHEMA",
    "CONTEXTUAL_KEY_DIMENSIONS",
    "CONTEXTUAL_PROJECTION_ABI",
    "MAX_CONTINUATION_TOKENS",
    "ContextualCandidate",
    "ContextualCandidateFeedback",
    "ContextualCapture",
    "ContextualContinuationBank",
    "ContextualContinuationError",
    "ContextualContinuationIdentity",
    "ContextualContinuationIdentityError",
    "ContextualContinuationIntegrityError",
    "ContextualContinuationMetrics",
    "ContextualKey",
]
