"""Append-only causal segments with bounded, on-demand graph inference.

This is the small runtime-facing subset of the O1 ``livecausal`` prototype.
The tensor/checkpoint plane remains immutable; this module is the durable
control-plane sidecar.  Segment files are content addressed, while activation
and deactivation are committed by an append-only hash-chained journal plus a
small atomic head.  Consequently a steady-state append never rewrites a list
that grows with the store.

Only exact ``trigger_key -> outcome_key`` joins are implemented here.  There
is deliberately no eager closure, fuzzy matching, canonicalization, spaCy, or
model-dependent inference in this core.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
from typing import Any

try:  # POSIX is the production target; the thread lock remains a safe fallback.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX Python.
    fcntl = None  # type: ignore[assignment]


MANIFEST_VERSION = 2
SEGMENT_VERSION = 2
MANIFEST_JOURNAL_NAME = "manifest.jsonl"
MANIFEST_HEAD_NAME = "manifest.head.json"
LOCK_NAME = ".manifest.lock"
DEFAULT_NODE_BUDGET = 5_000
DEFAULT_MAX_DEPTH = 5
MAX_HEAD_BYTES = 64 * 1024
MAX_RECORD_BYTES = 16 * 1024 * 1024
ZERO_DIGEST = "0" * 64
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")

_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[str, threading.RLock] = {}


class LiveCausalError(RuntimeError):
    """Base class for live-causal persistence and query failures."""


class LiveCausalIntegrityError(LiveCausalError):
    """Stored bytes do not satisfy their committed hashes or schema."""


class LiveCausalValidationError(LiveCausalError, ValueError):
    """Caller input cannot be represented by the live-causal schema."""


def _validate_json_value(value: Any, *, path: str = "record") -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LiveCausalValidationError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _validate_json_value(child, path=f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise LiveCausalValidationError(f"{path} has a non-string object key")
            normalized[key] = _validate_json_value(child, path=f"{path}.{key}")
        return normalized
    raise LiveCausalValidationError(
        f"{path} contains unsupported JSON value {type(value).__name__}"
    )


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
        raise LiveCausalValidationError("value is not canonical JSON") from exc


def _canonical_line(value: Any) -> bytes:
    return _canonical_json(value) + b"\n"


def canonical_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    """Return the exact canonical JSONL bytes hashed by a segment."""

    normalized = _normalize_records(records)
    return b"".join(_canonical_line(record) for record in normalized)


def segment_sha(records: Iterable[Mapping[str, Any]]) -> str:
    """Return the SHA-256 identity of canonical record bytes."""

    return hashlib.sha256(canonical_bytes(records)).hexdigest()


def _normalize_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    try:
        materialized = tuple(records)
    except TypeError as exc:
        raise LiveCausalValidationError("records must be iterable") from exc
    if not materialized:
        raise LiveCausalValidationError("a segment must contain at least one record")
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(materialized):
        if not isinstance(record, Mapping):
            raise LiveCausalValidationError(f"record[{index}] must be an object")
        value = _validate_json_value(record, path=f"record[{index}]")
        assert isinstance(value, dict)
        line = _canonical_line(value)
        if len(line) > MAX_RECORD_BYTES:
            raise LiveCausalValidationError(f"record[{index}] exceeds the size bound")
        normalized.append(value)
    return tuple(normalized)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reject_constant(value: str) -> Any:
    raise LiveCausalIntegrityError(f"non-finite JSON constant {value!r}")


def _decode_json(data: bytes, *, label: str) -> Any:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LiveCausalIntegrityError(
                    f"duplicate JSON key in {label}: {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(
            data,
            object_pairs_hook=no_duplicates,
            parse_constant=_reject_constant,
        )
    except LiveCausalIntegrityError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LiveCausalIntegrityError(f"cannot decode {label}") from exc


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_stat(path: Path, *, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LiveCausalIntegrityError(f"cannot inspect {label}: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise LiveCausalIntegrityError(f"{label} must be a regular file: {path}")
    return metadata


def _read_regular(path: Path, *, label: str, maximum: int | None = None) -> bytes:
    metadata = _regular_stat(path, label=label)
    if maximum is not None and metadata.st_size > maximum:
        raise LiveCausalIntegrityError(f"{label} exceeds its size bound")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LiveCausalIntegrityError(f"cannot open {label}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise LiveCausalIntegrityError(f"{label} changed while opening")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise LiveCausalIntegrityError(f"{label} was truncated while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise LiveCausalIntegrityError(f"{label} grew while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        _regular_stat(path, label=path.name)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".pending", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _thread_lock(path: Path) -> threading.RLock:
    key = os.path.normcase(str(path))
    with _LOCKS_GUARD:
        return _THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _exclusive_store_lock(root: Path) -> Iterator[None]:
    lock = _thread_lock(root)
    with lock:
        lock_path = root / LOCK_NAME
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise LiveCausalIntegrityError(
                f"cannot open store lock: {lock_path}"
            ) from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise LiveCausalIntegrityError("store lock must be a regular file")
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _head_without_digest(
    *,
    sequence: int,
    offset: int,
    event_sha256: str,
    active_count: int,
    tombstone_count: int,
) -> dict[str, Any]:
    return {
        "active_count": active_count,
        "event_sha256": event_sha256,
        "journal_offset": offset,
        "sequence": sequence,
        "tombstone_count": tombstone_count,
        "version": MANIFEST_VERSION,
    }


def _make_head(
    *,
    sequence: int,
    offset: int,
    event_sha256: str,
    active_count: int,
    tombstone_count: int,
) -> dict[str, Any]:
    body = _head_without_digest(
        sequence=sequence,
        offset=offset,
        event_sha256=event_sha256,
        active_count=active_count,
        tombstone_count=tombstone_count,
    )
    return {**body, "head_sha256": _sha256(_canonical_json(body))}


def _event_core(
    *, sequence: int, previous: str, operation: str, shas: Sequence[str]
) -> dict[str, Any]:
    return {
        "op": operation,
        "previous": previous,
        "sequence": sequence,
        "shas": list(shas),
        "version": MANIFEST_VERSION,
    }


def _make_event(
    *, sequence: int, previous: str, operation: str, shas: Sequence[str]
) -> dict[str, Any]:
    core = _event_core(
        sequence=sequence,
        previous=previous,
        operation=operation,
        shas=shas,
    )
    return {**core, "event_sha256": _sha256(_canonical_json(core))}


class LiveStore:
    """Crash-consistent append-only store for content-addressed segments.

    ``manifest.jsonl`` is an append-only hash chain.  ``manifest.head.json``
    is the sole small atomic commit point and records the committed journal
    byte offset.  Bytes beyond that offset are an interrupted transaction and
    are discarded on the next locked mount.  Dropping a segment changes only
    journal state; its immutable ``.seg`` object remains available for audit
    and reactivation.
    """

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        candidate = Path(directory).expanduser().absolute()
        candidate.mkdir(parents=True, exist_ok=True)
        if candidate.is_symlink() or not candidate.is_dir():
            raise LiveCausalIntegrityError("store root must be a real directory")
        self.root = candidate.resolve()
        self._active: list[str] = []
        self._active_set: set[str] = set()
        self._tombstones: set[str] = set()
        self._sequence = 0
        self._journal_offset = 0
        self._event_sha256 = ZERO_DIGEST
        self._record_offsets: dict[str, tuple[int, ...]] = {}
        self._verified_stats: dict[str, tuple[int, int, int, int, int]] = {}
        with _exclusive_store_lock(self.root):
            self._initialize_locked()
            self._reload_locked(verify_segments=True)

    @property
    def journal_path(self) -> Path:
        return self.root / MANIFEST_JOURNAL_NAME

    @property
    def head_path(self) -> Path:
        return self.root / MANIFEST_HEAD_NAME

    def segment_path(self, sha256: str) -> Path:
        self._validate_digest(sha256)
        return self.root / f"{sha256}.seg"

    @staticmethod
    def _validate_digest(value: str) -> None:
        if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
            raise LiveCausalValidationError(
                "segment identity must be lowercase SHA-256"
            )

    def _initialize_locked(self) -> None:
        journal_exists = self.journal_path.exists() or self.journal_path.is_symlink()
        head_exists = self.head_path.exists() or self.head_path.is_symlink()
        if journal_exists != head_exists:
            raise LiveCausalIntegrityError("manifest journal/head pair is incomplete")
        if journal_exists:
            _regular_stat(self.journal_path, label="manifest journal")
            _regular_stat(self.head_path, label="manifest head")
            return

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.journal_path, flags, 0o600)
        except OSError as exc:
            raise LiveCausalIntegrityError("cannot create manifest journal") from exc
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self.root)
        head = _make_head(
            sequence=0,
            offset=0,
            event_sha256=ZERO_DIGEST,
            active_count=0,
            tombstone_count=0,
        )
        _atomic_write(self.head_path, _canonical_line(head))

    def _read_head_locked(self) -> dict[str, Any]:
        raw = _read_regular(
            self.head_path,
            label="manifest head",
            maximum=MAX_HEAD_BYTES,
        )
        value = _decode_json(raw, label="manifest head")
        if not isinstance(value, dict) or raw != _canonical_line(value):
            raise LiveCausalIntegrityError("manifest head is not canonical JSON")
        required = {
            "active_count",
            "event_sha256",
            "head_sha256",
            "journal_offset",
            "sequence",
            "tombstone_count",
            "version",
        }
        if set(value) != required or value.get("version") != MANIFEST_VERSION:
            raise LiveCausalIntegrityError("manifest head schema/version mismatch")
        integers = ("active_count", "journal_offset", "sequence", "tombstone_count")
        if any(
            isinstance(value[name], bool)
            or not isinstance(value[name], int)
            or value[name] < 0
            for name in integers
        ):
            raise LiveCausalIntegrityError("manifest head contains an invalid counter")
        event_digest = value["event_sha256"]
        head_digest = value["head_sha256"]
        if (
            not isinstance(event_digest, str)
            or _DIGEST.fullmatch(event_digest) is None
            or not isinstance(head_digest, str)
            or _DIGEST.fullmatch(head_digest) is None
        ):
            raise LiveCausalIntegrityError("manifest head contains an invalid digest")
        body = {key: child for key, child in value.items() if key != "head_sha256"}
        if _sha256(_canonical_json(body)) != head_digest:
            raise LiveCausalIntegrityError("manifest head digest mismatch")
        if value["sequence"] == 0 and event_digest != ZERO_DIGEST:
            raise LiveCausalIntegrityError("empty manifest has a nonzero event digest")
        return value

    @staticmethod
    def _validate_event(
        value: Any, *, expected_sequence: int, previous: str
    ) -> dict[str, Any]:
        required = {
            "event_sha256",
            "op",
            "previous",
            "sequence",
            "shas",
            "version",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise LiveCausalIntegrityError("manifest journal event schema mismatch")
        if value.get("version") != MANIFEST_VERSION:
            raise LiveCausalIntegrityError("manifest journal version mismatch")
        if (
            value.get("sequence") != expected_sequence
            or value.get("previous") != previous
        ):
            raise LiveCausalIntegrityError("manifest journal hash-chain discontinuity")
        operation = value.get("op")
        shas = value.get("shas")
        if operation not in {"add", "drop"} or not isinstance(shas, list) or not shas:
            raise LiveCausalIntegrityError("manifest journal operation is invalid")
        if operation == "add" and len(shas) != 1:
            raise LiveCausalIntegrityError("an add event must name exactly one segment")
        if len(set(shas)) != len(shas):
            raise LiveCausalIntegrityError("manifest journal event repeats a segment")
        for sha in shas:
            if not isinstance(sha, str) or _DIGEST.fullmatch(sha) is None:
                raise LiveCausalIntegrityError(
                    "manifest journal has an invalid segment digest"
                )
        claimed = value.get("event_sha256")
        if not isinstance(claimed, str) or _DIGEST.fullmatch(claimed) is None:
            raise LiveCausalIntegrityError("manifest journal event digest is invalid")
        core = {key: child for key, child in value.items() if key != "event_sha256"}
        if _sha256(_canonical_json(core)) != claimed:
            raise LiveCausalIntegrityError("manifest journal event digest mismatch")
        return value

    @staticmethod
    def _apply_event_to(
        event: Mapping[str, Any],
        active: list[str],
        active_set: set[str],
        tombstones: set[str],
    ) -> None:
        if event["op"] == "add":
            sha = event["shas"][0]
            if sha in active_set:
                raise LiveCausalIntegrityError(
                    "manifest journal contains a duplicate active add"
                )
            active.append(sha)
            active_set.add(sha)
            tombstones.discard(sha)
            return
        for sha in event["shas"]:
            if sha not in active_set:
                raise LiveCausalIntegrityError(
                    "manifest journal drops an inactive segment"
                )
            active_set.remove(sha)
            active.remove(sha)
            tombstones.add(sha)

    def _replay_locked(self, head: Mapping[str, Any]) -> None:
        journal_stat = _regular_stat(self.journal_path, label="manifest journal")
        committed = head["journal_offset"]
        if journal_stat.st_size < committed:
            raise LiveCausalIntegrityError(
                "manifest journal is shorter than committed head"
            )

        active: list[str] = []
        active_set: set[str] = set()
        tombstones: set[str] = set()
        sequence = 0
        previous = ZERO_DIGEST
        consumed = 0
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.journal_path, flags)
        except OSError as exc:
            raise LiveCausalIntegrityError("cannot open manifest journal") from exc
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                while consumed < committed:
                    remaining = committed - consumed
                    line = handle.readline(remaining + 1)
                    if not line or len(line) > remaining or not line.endswith(b"\n"):
                        raise LiveCausalIntegrityError(
                            "committed manifest event is truncated"
                        )
                    consumed += len(line)
                    value = _decode_json(line, label="manifest journal event")
                    if not isinstance(value, dict) or line != _canonical_line(value):
                        raise LiveCausalIntegrityError(
                            "manifest journal event is not canonical"
                        )
                    sequence += 1
                    event = self._validate_event(
                        value,
                        expected_sequence=sequence,
                        previous=previous,
                    )
                    self._apply_event_to(event, active, active_set, tombstones)
                    previous = event["event_sha256"]
        finally:
            os.close(descriptor)
        if consumed != committed:
            raise LiveCausalIntegrityError(
                "manifest committed offset is not an event boundary"
            )
        if (
            sequence != head["sequence"]
            or previous != head["event_sha256"]
            or len(active) != head["active_count"]
            or len(tombstones) != head["tombstone_count"]
        ):
            raise LiveCausalIntegrityError("manifest head does not match journal state")

        if journal_stat.st_size > committed:
            flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self.journal_path, flags)
            try:
                os.ftruncate(descriptor, committed)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

        self._active = active
        self._active_set = active_set
        self._tombstones = tombstones
        self._sequence = sequence
        self._journal_offset = committed
        self._event_sha256 = previous

    def _reload_locked(self, *, verify_segments: bool) -> None:
        head = self._read_head_locked()
        self._replay_locked(head)
        if verify_segments:
            for sha in (*self._active, *sorted(self._tombstones)):
                self._verify_segment_locked(sha, force=True)

    def _refresh_locked(self) -> None:
        head = self._read_head_locked()
        if (
            head["sequence"] == self._sequence
            and head["journal_offset"] == self._journal_offset
            and head["event_sha256"] == self._event_sha256
        ):
            journal_stat = _regular_stat(self.journal_path, label="manifest journal")
            if journal_stat.st_size < self._journal_offset:
                raise LiveCausalIntegrityError("manifest journal was rolled back")
            if journal_stat.st_size > self._journal_offset:
                descriptor = os.open(
                    self.journal_path,
                    os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    os.ftruncate(descriptor, self._journal_offset)
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            return
        if (
            head["sequence"] < self._sequence
            or head["journal_offset"] < self._journal_offset
        ):
            raise LiveCausalIntegrityError("manifest head was rolled back")
        # Replaying is mount-only in the common case.  A second process can
        # advance the head, in which case a full authenticated replay keeps
        # the implementation simple and fail-closed; steady single-process
        # append remains O(delta) and never rewrites prior journal bytes.
        self._replay_locked(head)
        for sha in (*self._active, *sorted(self._tombstones)):
            self._verify_segment_locked(sha, force=False)

    def _segment_signature(self, path: Path) -> tuple[int, int, int, int, int]:
        metadata = _regular_stat(path, label="causal segment")
        return (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    def _verify_segment_locked(self, sha: str, *, force: bool) -> tuple[int, ...]:
        path = self.segment_path(sha)
        signature = self._segment_signature(path)
        if not force and self._verified_stats.get(sha) == signature:
            return self._record_offsets[sha]

        digest = hashlib.sha256()
        offsets: list[int] = []
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                header_line = handle.readline(MAX_RECORD_BYTES + 1)
                if (
                    not header_line.endswith(b"\n")
                    or len(header_line) > MAX_RECORD_BYTES
                ):
                    raise LiveCausalIntegrityError(
                        "segment header is missing or oversized"
                    )
                header = _decode_json(header_line, label="segment header")
                required = {"count", "sha256", "version"}
                if not isinstance(header, dict) or set(header) != required:
                    raise LiveCausalIntegrityError("segment header schema mismatch")
                if header_line != _canonical_line(header):
                    raise LiveCausalIntegrityError("segment header is not canonical")
                if (
                    header.get("version") != SEGMENT_VERSION
                    or header.get("sha256") != sha
                    or isinstance(header.get("count"), bool)
                    or not isinstance(header.get("count"), int)
                    or header["count"] <= 0
                ):
                    raise LiveCausalIntegrityError("segment header values are invalid")

                while True:
                    offset = handle.tell()
                    line = handle.readline(MAX_RECORD_BYTES + 1)
                    if not line:
                        break
                    if not line.endswith(b"\n") or len(line) > MAX_RECORD_BYTES:
                        raise LiveCausalIntegrityError(
                            "segment record is truncated or oversized"
                        )
                    value = _decode_json(line, label="segment record")
                    if not isinstance(value, dict) or line != _canonical_line(value):
                        raise LiveCausalIntegrityError(
                            "segment record is not canonical"
                        )
                    offsets.append(offset)
                    digest.update(line)
        finally:
            os.close(descriptor)
        if len(offsets) != header["count"] or digest.hexdigest() != sha:
            raise LiveCausalIntegrityError("segment count or SHA-256 mismatch")
        self._record_offsets[sha] = tuple(offsets)
        self._verified_stats[sha] = signature
        return tuple(offsets)

    def _install_segment_locked(
        self, sha: str, records: Sequence[Mapping[str, Any]]
    ) -> None:
        path = self.segment_path(sha)
        if path.exists() or path.is_symlink():
            self._verify_segment_locked(sha, force=True)
            return
        body = b"".join(_canonical_line(record) for record in records)
        header = _canonical_line(
            {"count": len(records), "sha256": sha, "version": SEGMENT_VERSION}
        )
        _atomic_write(path, header + body)
        self._verify_segment_locked(sha, force=True)

    def _append_event_locked(self, operation: str, shas: Sequence[str]) -> None:
        event = _make_event(
            sequence=self._sequence + 1,
            previous=self._event_sha256,
            operation=operation,
            shas=shas,
        )
        line = _canonical_line(event)
        flags = os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.journal_path, flags)
        try:
            metadata = os.fstat(descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_size != self._journal_offset
            ):
                raise LiveCausalIntegrityError("manifest journal changed before append")
            view = memoryview(line)
            written = 0
            while written < len(view):
                count = os.write(descriptor, view[written:])
                if count <= 0:
                    raise LiveCausalIntegrityError(
                        "manifest journal append made no progress"
                    )
                written += count
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

        if operation == "add":
            sha = shas[0]
            active_count = len(self._active) + 1
            tombstone_count = len(self._tombstones) - int(sha in self._tombstones)
        else:
            active_count = len(self._active) - len(shas)
            tombstone_count = len(self._tombstones) + len(shas)
        new_offset = self._journal_offset + len(line)
        head = _make_head(
            sequence=self._sequence + 1,
            offset=new_offset,
            event_sha256=event["event_sha256"],
            active_count=active_count,
            tombstone_count=tombstone_count,
        )
        _atomic_write(self.head_path, _canonical_line(head))
        self._apply_event_to(
            event,
            self._active,
            self._active_set,
            self._tombstones,
        )
        self._sequence += 1
        self._journal_offset = new_offset
        self._event_sha256 = event["event_sha256"]

    def append_segment(self, records: Iterable[Mapping[str, Any]]) -> str:
        """Seal and activate one segment, idempotently.

        Existing active content causes no journal/head write.  Reactivating a
        tombstoned segment appends a new ``add`` event but reuses and verifies
        the original immutable object.
        """

        sha, _start_sequence, _end_sequence, _activated = (
            self._append_segment_with_receipt(records)
        )
        return sha

    def _append_segment_with_receipt(
        self, records: Iterable[Mapping[str, Any]]
    ) -> tuple[str, int, int, bool]:
        """Append plus the locked sequence interval, for ``LazyGraph`` delta sync."""

        normalized = _normalize_records(records)
        body = b"".join(_canonical_line(record) for record in normalized)
        sha = _sha256(body)
        with _exclusive_store_lock(self.root):
            self._refresh_locked()
            start_sequence = self._sequence
            self._install_segment_locked(sha, normalized)
            activated = sha not in self._active_set
            if activated:
                self._append_event_locked("add", (sha,))
            return sha, start_sequence, self._sequence, activated

    def drop_segments(self, shas: Iterable[str]) -> tuple[str, ...]:
        """Deactivate active segments without deleting their immutable files."""

        requested = tuple(dict.fromkeys(shas))
        for sha in requested:
            self._validate_digest(sha)
        with _exclusive_store_lock(self.root):
            self._refresh_locked()
            dropped = tuple(sorted(sha for sha in requested if sha in self._active_set))
            if dropped:
                self._append_event_locked("drop", dropped)
            return dropped

    def segments(self) -> tuple[str, ...]:
        """Return active segment identities in committed activation order."""

        with _exclusive_store_lock(self.root):
            self._refresh_locked()
            return tuple(self._active)

    def tombstones(self) -> tuple[str, ...]:
        """Return currently inactive, reversibly dropped segment identities."""

        with _exclusive_store_lock(self.root):
            self._refresh_locked()
            return tuple(sorted(self._tombstones))

    def revision(self) -> tuple[int, str]:
        """Return the constant-size committed manifest revision token."""

        with _exclusive_store_lock(self.root):
            self._refresh_locked()
            return self._sequence, self._event_sha256

    def _read_record_locked(self, sha: str, index: int) -> dict[str, Any]:
        offsets = self._verify_segment_locked(sha, force=False)
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(offsets)
        ):
            raise LiveCausalValidationError("record index is outside the segment")
        path = self.segment_path(sha)
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            os.lseek(descriptor, offsets[index], os.SEEK_SET)
            chunks: list[bytes] = []
            total = 0
            while True:
                byte = os.read(descriptor, min(64 * 1024, MAX_RECORD_BYTES + 1 - total))
                if not byte:
                    raise LiveCausalIntegrityError("segment record is truncated")
                newline = byte.find(b"\n")
                if newline >= 0:
                    chunks.append(byte[: newline + 1])
                    break
                chunks.append(byte)
                total += len(byte)
                if total > MAX_RECORD_BYTES:
                    raise LiveCausalIntegrityError(
                        "segment record exceeds its size bound"
                    )
        finally:
            os.close(descriptor)
        line = b"".join(chunks)
        value = _decode_json(line, label="segment record")
        if not isinstance(value, dict) or line != _canonical_line(value):
            raise LiveCausalIntegrityError("segment record is not canonical")
        return value

    def record(self, sha: str, index: int) -> dict[str, Any]:
        """Resolve ``[sha, index]`` by direct byte offset, without a segment scan."""

        self._validate_digest(sha)
        with _exclusive_store_lock(self.root):
            self._refresh_locked()
            if sha not in self._active_set and sha not in self._tombstones:
                raise LiveCausalValidationError(
                    "segment is not referenced by this store"
                )
            return self._read_record_locked(sha, index)

    def iter_records(
        self, sha: str | None = None
    ) -> Iterator[tuple[str, int, dict[str, Any]]]:
        """Return a stable snapshot of active records and exact citations."""

        if sha is not None:
            self._validate_digest(sha)
        with _exclusive_store_lock(self.root):
            self._refresh_locked()
            selected = tuple(self._active) if sha is None else (sha,)
            if sha is not None and sha not in self._active_set:
                raise LiveCausalValidationError("segment is not active")
            rows: list[tuple[str, int, dict[str, Any]]] = []
            for segment in selected:
                offsets = self._verify_segment_locked(segment, force=False)
                rows.extend(
                    (segment, index, self._read_record_locked(segment, index))
                    for index in range(len(offsets))
                )
        return iter(rows)

    def verify(self, *, include_tombstones: bool = True) -> bool:
        """Authenticate the committed journal and every referenced segment."""

        try:
            with _exclusive_store_lock(self.root):
                self._reload_locked(verify_segments=False)
                selected = list(self._active)
                if include_tombstones:
                    selected.extend(sorted(self._tombstones))
                for sha in selected:
                    self._verify_segment_locked(sha, force=True)
        except LiveCausalError:
            return False
        return True


def _edge_from_record(record: Mapping[str, Any]) -> tuple[str, str]:
    from_key = record.get("trigger_key")
    to_key = record.get("outcome_key")
    if (
        not isinstance(from_key, str)
        or not from_key
        or not isinstance(to_key, str)
        or not to_key
    ):
        raise LiveCausalIntegrityError(
            "graph records require non-empty string trigger_key/outcome_key"
        )
    return from_key, to_key


def _derivation_key(derivation: Sequence[Sequence[Any]]) -> tuple[tuple[str, int], ...]:
    return tuple((str(sha), int(index)) for sha, index in derivation)


class LazyGraph:
    """Exact base adjacency plus bounded transitive inference at query time."""

    def __init__(
        self,
        store: LiveStore | str | os.PathLike[str],
        *,
        max_depth: int = DEFAULT_MAX_DEPTH,
        node_budget: int = DEFAULT_NODE_BUDGET,
    ) -> None:
        if (
            isinstance(max_depth, bool)
            or not isinstance(max_depth, int)
            or max_depth < 1
        ):
            raise LiveCausalValidationError("max_depth must be a positive integer")
        if (
            isinstance(node_budget, bool)
            or not isinstance(node_budget, int)
            or node_budget < 1
        ):
            raise LiveCausalValidationError("node_budget must be a positive integer")
        self.store = store if isinstance(store, LiveStore) else LiveStore(store)
        self.max_depth = max_depth
        self.default_node_budget = node_budget
        self._base_edges: dict[str, dict[str, list[list[Any]]]] = {}
        self._segments: list[str] = []
        self._segment_set: set[str] = set()
        self._store_revision: tuple[int, str] = (-1, "")
        self._lock = threading.RLock()
        with self._lock:
            self._sync_locked()

    def _add_segment_locked(self, sha: str) -> None:
        for _segment, index, record in self.store.iter_records(sha):
            from_key, to_key = _edge_from_record(record)
            citations = self._base_edges.setdefault(from_key, {}).setdefault(to_key, [])
            citation = [sha, index]
            if citation not in citations:
                citations.append(citation)
                citations.sort(key=lambda pair: (pair[0], pair[1]))

    def _drop_from_adjacency_locked(self, dropped: set[str]) -> None:
        rebuilt: dict[str, dict[str, list[list[Any]]]] = {}
        for from_key, targets in self._base_edges.items():
            for to_key, citations in targets.items():
                kept = [pair for pair in citations if pair[0] not in dropped]
                if kept:
                    rebuilt.setdefault(from_key, {})[to_key] = kept
        self._base_edges = rebuilt

    def _sync_locked(self, *, force: bool = False) -> None:
        revision = self.store.revision()
        if not force and revision == self._store_revision:
            return
        active = list(self.store.segments())
        prior = set(self._segment_set)
        current = set(active)
        dropped = prior - current
        if dropped:
            self._drop_from_adjacency_locked(dropped)
        for sha in active:
            if sha not in prior:
                self._add_segment_locked(sha)
        self._segments = active
        self._segment_set = current
        # segments() refreshed under the store lock after revision(), so use
        # the store instance's exact replay point rather than the earlier token.
        self._store_revision = (self.store._sequence, self.store._event_sha256)

    def append_segment(self, records: Iterable[Mapping[str, Any]]) -> str:
        """Persist and index only the appended segment's records."""

        with self._lock:
            sha, start_sequence, end_sequence, _activated = (
                self.store._append_segment_with_receipt(records)
            )
            if sha not in self._segment_set:
                self._add_segment_locked(sha)
                self._segments.append(sha)
                self._segment_set.add(sha)
            if start_sequence == self._store_revision[0]:
                self._store_revision = (end_sequence, self.store._event_sha256)
            return sha

    def drop_segments(self, shas: Iterable[str]) -> tuple[str, ...]:
        """Reversibly deactivate segments and remove only their citations."""

        with self._lock:
            dropped = self.store.drop_segments(shas)
            self._sync_locked()
            return dropped

    def query(
        self,
        key: str,
        *,
        node_budget: int | None = None,
        return_truncated: bool = True,
    ) -> tuple[list[dict[str, Any]], bool] | list[dict[str, Any]]:
        """Return deterministic base/inferred edges and visible truncation.

        The budget counts DFS frames.  Each inferred derivation is an exact
        root-to-leaf list of ``[segment_sha256, record_index]`` citations.
        ``return_truncated`` defaults to true so a bounded undercount is not
        accidentally presented as complete; passing false retains the older
        list-only convenience shape.
        """

        if not isinstance(key, str) or not key:
            raise LiveCausalValidationError("query key must be a non-empty string")
        budget = self.default_node_budget if node_budget is None else node_budget
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise LiveCausalValidationError("node_budget must be a positive integer")
        with self._lock:
            self._sync_locked()
            edges, truncated = self._bounded_query_locked(key, budget)
        if return_truncated:
            return edges, truncated
        return edges

    def query_base(self, key: str) -> list[dict[str, Any]]:
        """Return every direct outgoing edge without bounded inference.

        This path never enters the DFS and therefore cannot be truncated by
        ``default_node_budget``.  One result is emitted per direct target; all
        records supporting that target remain attached as exact ``[sha, idx]``
        citations in deterministic order.
        """

        if not isinstance(key, str) or not key:
            raise LiveCausalValidationError("query key must be a non-empty string")
        with self._lock:
            self._sync_locked()
            return [
                {
                    "depth": 1,
                    "derivation": [list(pair) for pair in citations],
                    "from_key": key,
                    "kind": "base",
                    "to_key": to_key,
                }
                for to_key, citations in sorted(self._base_edges.get(key, {}).items())
            ]

    def _bounded_query_locked(
        self, key: str, node_budget: int
    ) -> tuple[list[dict[str, Any]], bool]:
        edges: list[dict[str, Any]] = []
        seen: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
        remaining = node_budget
        truncated = False

        def emit(to_key: str, derivation: list[list[Any]]) -> None:
            identity = (to_key, _derivation_key(derivation))
            if identity in seen:
                return
            seen.add(identity)
            edges.append(
                {
                    "depth": len(derivation),
                    "derivation": [list(pair) for pair in derivation],
                    "from_key": key,
                    "kind": "inferred",
                    "to_key": to_key,
                }
            )

        def walk(path: list[str], derivation: list[list[Any]]) -> None:
            nonlocal remaining, truncated
            if remaining <= 0:
                truncated = True
                return
            remaining -= 1
            if len(derivation) >= self.max_depth:
                return
            current = path[-1]
            for to_key, citations in sorted(self._base_edges.get(current, {}).items()):
                if to_key in path:
                    continue
                for citation in citations:
                    next_derivation = derivation + [list(citation)]
                    if len(next_derivation) >= 2:
                        emit(to_key, next_derivation)
                    walk(path + [to_key], next_derivation)
                    if truncated:
                        return

        walk([key], [])
        for to_key, citations in sorted(self._base_edges.get(key, {}).items()):
            edges.append(
                {
                    "depth": 1,
                    "derivation": [list(pair) for pair in citations],
                    "from_key": key,
                    "kind": "base",
                    "to_key": to_key,
                }
            )
        edges.sort(
            key=lambda edge: (
                edge["kind"],
                edge["to_key"],
                edge["depth"],
                _derivation_key(edge["derivation"]),
            )
        )
        return edges, truncated

    def base_edge_citations(self, from_key: str, to_key: str) -> list[list[Any]]:
        """Return a defensive, deterministic copy of one base edge's citations."""

        with self._lock:
            self._sync_locked()
            return [
                list(pair)
                for pair in self._base_edges.get(from_key, {}).get(to_key, ())
            ]

    def resolve_derivation(
        self, derivation: Iterable[Sequence[Any]]
    ) -> list[dict[str, Any]]:
        """Resolve exact citations through the store's offset index."""

        records: list[dict[str, Any]] = []
        for pair in derivation:
            if (
                not isinstance(pair, Sequence)
                or isinstance(pair, (str, bytes))
                or len(pair) != 2
            ):
                raise LiveCausalValidationError("each citation must be [sha256, index]")
            sha, index = pair
            if not isinstance(sha, str):
                raise LiveCausalValidationError("citation SHA-256 must be a string")
            records.append(self.store.record(sha, index))
        return records


# Source-compatible name for callers that selected ``inference='lazy'`` in O1.
LiveGraph = LazyGraph


__all__ = [
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_NODE_BUDGET",
    "LiveCausalError",
    "LiveCausalIntegrityError",
    "LiveCausalValidationError",
    "LiveGraph",
    "LiveStore",
    "LazyGraph",
    "canonical_bytes",
    "segment_sha",
]
