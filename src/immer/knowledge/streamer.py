"""Budgeted, range-only access to tensors stored in safetensors files.

``Streamer`` is deliberately a tensor source, not a model loader. It reads a
safetensors inventory and exact byte ranges; it never instantiates a donor
model. The same contract works against Hugging Face and a local directory::

    remote = Streamer("Qwen/Qwen2.5-0.5B", revision="<commit>")
    offline = Streamer.from_local("./fixture")
    rows = offline.rows("weight", 0, 8)
    tensor = offline.rows_torch("weight", 0, 8)

Every transferred response body is charged to a hard byte budget. Inventory
and range caches are atomic and SHA-256 verified, so a later process can resume
without silently trusting a partial cache entry. Pinning ``revision`` to a
commit is still required for freshness; a digest can prove cached bytes, not
that a mutable ``main`` revision has not moved.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
import threading
import time
from contextlib import contextmanager
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ._hf_source import (
    Budget,
    BudgetExceeded,
    HFRangeReader,
    SourceNotFound,
    scan_inventory,
)


class TensorSourceError(RuntimeError):
    """Base error for the explicit tensor-source contract."""


class ByteBudgetExceeded(BudgetExceeded):
    """A request cannot fit into the remaining transfer budget."""


class RangeValidationError(TensorSourceError, ValueError):
    """A byte or tensor-row range is malformed or outside its source."""


class CacheIntegrityError(TensorSourceError):
    """A cache entry is partial, stale relative to known identity, or corrupt."""


class InventoryValidationError(TensorSourceError):
    """A tensor inventory violates the safetensors source contract."""


class TensorEncodingError(InventoryValidationError):
    """A tensor payload contains a reserved or non-finite raw encoding."""


@dataclass(frozen=True, slots=True)
class RawBytesManyResult:
    """Immutable zero-copy result of one exact multi-range read.

    ``parts`` are readonly views in caller order. Adjacent cold leaves can
    share one backing ``bytes`` owner, while independently cached leaves keep
    their own verified owners. ``resident_bytes`` counts those owners once.
    """

    parts: tuple[memoryview, ...]
    resident_bytes: int
    source_requests: int
    source_bytes: int


@runtime_checkable
class TensorSource(Protocol):
    """Small public contract consumed by retrieval and analysis code."""

    def prepare_parallel_reads(self) -> int: ...

    def inventory(self, *, refresh: bool = False) -> dict[str, Any]: ...

    def find(self, name: str) -> dict[str, Any]: ...

    def tensor(self, tensor_name: str) -> Any: ...

    def rows(
        self,
        tensor_name: str,
        start_row: int = 0,
        n_rows: int = 8,
        n_blocks: int = 8,
    ) -> Any: ...

    def raw_bytes(self, shard: str, offset: int, length: int) -> bytes: ...

    def cache_priority(self, priority: int) -> Any: ...

    def clear_cache_priorities(self) -> None: ...

    def metrics(self) -> dict[str, Any]: ...


class HardByteBudget(Budget):
    """Thread-safe budget whose counters never exceed the limit."""

    def __init__(self, limit_mb: float) -> None:
        if isinstance(limit_mb, bool) or float(limit_mb) < 0:
            raise ValueError("budget_mb muss eine nichtnegative Zahl sein")
        super().__init__(float(limit_mb))
        self.exceeded_error = ByteBudgetExceeded
        self.rejected_charges = 0
        self._charge_lock = threading.Lock()
        self._reserved_bytes = 0
        self._reservation_local = threading.local()

    def _reservation_stack(self) -> list[dict[str, Any]]:
        stack = getattr(self._reservation_local, "stack", None)
        if stack is None:
            stack = []
            self._reservation_local.stack = stack
        return stack

    def thread_charge_snapshot(self) -> tuple[int, int, int]:
        """Return charges made by the calling thread only."""

        return (
            int(getattr(self._reservation_local, "charged_body", 0)),
            int(getattr(self._reservation_local, "charged_overhead", 0)),
            int(getattr(self._reservation_local, "charged_requests", 0)),
        )

    @contextmanager
    def reservation(self, amount: int, tag: str) -> Any:
        """Reserve a hard-budget slice for charges made by this thread.

        The upstream range reader charges the shared budget itself. A
        thread-local receipt lets those charges atomically consume this
        reservation without serializing independent network reads.
        """

        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError("Budget-Reservierung darf nicht negativ sein")
        receipt: dict[str, Any] = {
            "remaining": amount,
            "tag": str(tag),
            "body": 0,
            "overhead": 0,
            "requests": 0,
        }
        with self._charge_lock:
            attempted = self.total + self._reserved_bytes + amount
            if attempted > self.limit:
                self.rejected_charges += 1
                raise ByteBudgetExceeded(
                    f"Bytebudget reicht vor I/O nicht fuer {tag!r}: "
                    f"{attempted}/{self.limit} Bytes"
                )
            self._reserved_bytes += amount
        stack = self._reservation_stack()
        stack.append(receipt)
        try:
            yield receipt
        finally:
            popped = stack.pop()
            if popped is not receipt:  # pragma: no cover - internal invariant
                raise RuntimeError("Budget-Reservierungen wurden nicht LIFO beendet")
            with self._charge_lock:
                self._reserved_bytes -= int(receipt["remaining"])
                if self._reserved_bytes < 0:  # pragma: no cover - invariant
                    self._reserved_bytes = 0
                    raise RuntimeError("Budget-Reservierung ist untergelaufen")

    def charge(self, body: int, overhead: int, tag: str) -> None:
        body = int(body)
        overhead = int(overhead)
        if body < 0 or overhead < 0:
            raise ValueError("Budget-Charge darf nicht negativ sein")
        amount = body + overhead
        with self._charge_lock:
            stack = self._reservation_stack()
            receipt = stack[-1] if stack else None
            thread_body, thread_overhead, thread_requests = (
                self.thread_charge_snapshot()
            )
            if receipt is not None and amount > int(receipt["remaining"]):
                self.rejected_charges += 1
                raise ByteBudgetExceeded(
                    f"Budget-Charge fuer {tag!r} uebersteigt die vorab "
                    f"reservierten Bytes: {amount}/{receipt['remaining']}"
                )
            attempted = self.total + self._reserved_bytes + (
                0 if receipt is not None else amount
            )
            if attempted > self.limit:
                self.rejected_charges += 1
                raise ByteBudgetExceeded(
                    f"Bytebudget reicht nicht fuer {tag!r}: "
                    f"{attempted}/{self.limit} Bytes"
                )
            if receipt is not None:
                receipt["remaining"] = int(receipt["remaining"]) - amount
                receipt["body"] = int(receipt["body"]) + body
                receipt["overhead"] = int(receipt["overhead"]) + overhead
                receipt["requests"] = int(receipt["requests"]) + 1
                self._reserved_bytes -= amount
            self.body += body
            self.overhead += overhead
            self.requests += 1
            self.log.append((str(tag), body, overhead))
            self._reservation_local.charged_body = thread_body + body
            self._reservation_local.charged_overhead = thread_overhead + overhead
            self._reservation_local.charged_requests = thread_requests + 1

    def as_dict(self) -> dict[str, int]:
        result = dict(super().as_dict())
        result["rejected_charges"] = int(self.rejected_charges)
        return result


class LocalRangeReader:
    """Offline inclusive-range reader for a directory of source files."""

    range_overhead_reserve = 0
    transport_policy = "local-range/v1"
    transport_connection_limit = 0

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        repo_id: str | None = None,
        revision: str = "local",
        budget: HardByteBudget | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Lokale Tensorquelle fehlt: {self.root}")
        self.repo = repo_id or f"local:{self.root}"
        self.rev = revision
        self.budget = budget or HardByteBudget(200.0)
        self.file_info: dict[str, dict[str, Any]] = {}

    def _path(self, filename: str) -> Path:
        if not isinstance(filename, str) or not filename or "\x00" in filename:
            raise RangeValidationError("Dateiname muss ein nichtleerer String sein")
        candidate = (self.root / filename).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise RangeValidationError(
                f"Pfad verlaesst lokale Quelle: {filename!r}"
            ) from exc
        return candidate

    def file_size(self, filename: str) -> int:
        path = self._path(filename)
        try:
            return int(path.stat().st_size)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Quelldatei fehlt: {filename}") from exc

    def source_identity(self, filename: str) -> dict[str, str]:
        """Return the current cheap local identity used to reject stale caches."""

        path = self._path(filename)
        try:
            stat = path.stat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Quelldatei fehlt: {filename}") from exc
        return {
            "size": str(int(stat.st_size)),
            "etag": f"local-{stat.st_size:x}-{stat.st_mtime_ns:x}",
        }

    def _remember(self, filename: str, path: Path) -> None:
        identity = self.source_identity(filename)
        self.file_info.setdefault(filename, {}).update(
            {"size": int(identity["size"]), "etag": identity["etag"]}
        )

    def get_range(self, filename: str, start: int, end: int) -> bytes:
        start, end = _validate_inclusive_range(start, end)
        path = self._path(filename)
        size = self.file_size(filename)
        if end >= size:
            raise RangeValidationError(
                f"Range ausserhalb {filename}: [{start}, {end}] bei {size} Bytes"
            )
        length = end - start + 1
        with path.open("rb") as handle:
            handle.seek(start)
            body = handle.read(length)
        if len(body) != length:
            raise RangeValidationError(
                f"Kurzer lokaler Read {filename}: {len(body)}/{length} Bytes"
            )
        self.budget.charge(length, 0, f"local-range:{filename}:{start}")
        self._remember(filename, path)
        return body

    def fetch_file(self, filename: str) -> bytes:
        path = self._path(filename)
        body = path.read_bytes()
        self.budget.charge(len(body), 0, f"local-file:{filename}")
        self._remember(filename, path)
        return body

    def transport_metrics(self) -> dict[str, Any]:
        return {
            "transport_policy": self.transport_policy,
            "transport_connection_limit": self.transport_connection_limit,
            "transport_active_lease_limit": 0,
            "transport_requests": int(self.budget.requests),
            "transport_retries": 0,
            "transport_active_leases": 0,
            "transport_peak_leases": 0,
            "transport_connection_objects_seen": 0,
            "transport_closed": False,
        }

    def close(self) -> None:
        """Match the remote reader lifecycle; local files are per-call."""


def _validate_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RangeValidationError(f"{label} muss eine nichtnegative Ganzzahl sein")
    return value


def _validate_inclusive_range(start: Any, end: Any) -> tuple[int, int]:
    start = _validate_nonnegative_int(start, "Range-Start")
    end = _validate_nonnegative_int(end, "Range-Ende")
    if end < start:
        raise RangeValidationError(f"Leere/verkehrte Range: [{start}, {end}]")
    return start, end


def _canonical_json(document: Any) -> bytes:
    return json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes | memoryview) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _default_cache_dir() -> Path:
    """Use the checkout cache while developing and a user cache from wheels."""
    project_root = Path(__file__).resolve().parents[3]
    if (project_root / "pyproject.toml").is_file():
        return project_root / "hf-cache" / "streamer"
    configured = os.environ.get("IMMER_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME")
    base = Path(configured).expanduser() if configured else Path.home() / ".cache"
    return base.resolve() / "immer" / "streamer"


class _CacheCoordinator:
    """Process-local serialization for readers sharing one exact cache root."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.key_locks: dict[str, threading.Lock] = {}


_CACHE_COORDINATORS_LOCK = threading.Lock()
_CACHE_COORDINATORS: dict[str, _CacheCoordinator] = {}


def _cache_coordinator(cache_dir: Path | None) -> _CacheCoordinator | None:
    if cache_dir is None:
        return None
    identity = os.fspath(cache_dir)
    with _CACHE_COORDINATORS_LOCK:
        return _CACHE_COORDINATORS.setdefault(identity, _CacheCoordinator())


class _ContractReader:
    """Budget, cache and exact-range guard around any compatible reader."""

    _CACHE_SCHEMA = "immer.range-cache/v1"

    def __init__(
        self,
        upstream: Any,
        budget: HardByteBudget,
        cache_dir: Path | None,
        *,
        max_metadata_bytes: int,
        max_cache_bytes: int | None,
    ) -> None:
        self.upstream = upstream
        self.budget = budget
        self.upstream.budget = budget
        self.repo = str(getattr(upstream, "repo", "unknown"))
        self.rev = str(getattr(upstream, "rev", "unknown"))
        self.file_info = getattr(upstream, "file_info", None)
        if not isinstance(self.file_info, dict):
            self.file_info = {}
            upstream.file_info = self.file_info
        self.cache_dir = cache_dir
        self.max_cache_bytes = max_cache_bytes
        if self.max_cache_bytes is not None and (
            isinstance(self.max_cache_bytes, bool)
            or not isinstance(self.max_cache_bytes, int)
            or self.max_cache_bytes < 0
        ):
            raise ValueError(
                "max_cache_bytes muss None oder eine nichtnegative Ganzzahl sein"
            )
        self._cache_coordinator = _cache_coordinator(cache_dir)
        self.max_metadata_bytes = int(max_metadata_bytes)
        if self.max_metadata_bytes <= 0:
            raise ValueError("max_metadata_bytes muss positiv sein")
        self._lock = threading.Lock()
        self._key_locks: dict[str, threading.Lock] = {}
        self._stats = {
            "range_logical_leaves": 0,
            "range_logical_leaf_bytes": 0,
            "range_requests": 0,
            "range_bytes_requested": 0,
            "range_source_requests": 0,
            "range_source_bytes": 0,
            "file_requests": 0,
            "failed_requests": 0,
            "optional_misses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_writes": 0,
            "cache_bytes_written": 0,
            "cache_bytes_reused": 0,
            "cache_integrity_checks": 0,
            "cache_bytes": 0,
            "cache_evictions": 0,
            "cache_evicted_bytes": 0,
            "cache_write_skips_oversize": 0,
            "cache_priority_promotions": 0,
            "cache_recoveries": 0,
        }
        self._cache_bypass = threading.local()
        self._cache_priority = threading.local()
        # Residency hints belong to this reader. Sharing them through the
        # process-wide coordinator would let one experiment silently bias the
        # eviction order of a later independent Streamer on the same root.
        self._cache_priorities: dict[str, int] = {}
        if self.max_cache_bytes is not None:
            self._enforce_cache_limit()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.upstream, name)

    def _bump(self, field: str, amount: int = 1) -> None:
        with self._lock:
            self._stats[field] += int(amount)

    def stats(self) -> dict[str, int | None]:
        self._refresh_cache_bytes()
        with self._lock:
            result: dict[str, int | None] = dict(self._stats)
        result["cache_limit_bytes"] = self.max_cache_bytes
        return result

    def _key_lock(self, key: str) -> threading.Lock:
        if self._cache_coordinator is not None:
            with self._cache_coordinator.lock:
                return self._cache_coordinator.key_locks.setdefault(
                    key, threading.Lock()
                )
        with self._lock:
            return self._key_locks.setdefault(key, threading.Lock())

    @contextmanager
    def _locked_cache_key(self, key: str) -> Any:
        lock = self._key_lock(key)
        try:
            with lock:
                yield
        finally:
            # Run after releasing the current key. Concurrent active entries
            # remain protected by their shared key locks.
            if self.max_cache_bytes is not None:
                self._enforce_cache_limit()

    @contextmanager
    def _locked_cache_keys(self, keys: Iterable[str]) -> Any:
        """Lock an exact key set in canonical order for batch single-flight."""

        locks = [self._key_lock(key) for key in sorted(set(keys))]
        acquired: list[threading.Lock] = []
        try:
            for lock in locks:
                lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()
            if self.max_cache_bytes is not None:
                self._enforce_cache_limit()

    @contextmanager
    def uncached(self) -> Any:
        """Temporarily bypass cache reads while still replacing fresh entries."""
        previous = bool(getattr(self._cache_bypass, "active", False))
        self._cache_bypass.active = True
        try:
            yield
        finally:
            self._cache_bypass.active = previous

    @contextmanager
    def cache_priority(self, priority: int) -> Any:
        """Prefer retaining ranges touched inside this request-local scope."""

        if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
            raise ValueError("cache priority must be a non-negative integer")
        previous = int(getattr(self._cache_priority, "value", 0))
        self._cache_priority.value = priority
        try:
            yield
        finally:
            self._cache_priority.value = previous

    def _mark_priority(self, key: str) -> None:
        coordinator = self._cache_coordinator
        if coordinator is None:
            return
        priority = int(getattr(self._cache_priority, "value", 0))
        with coordinator.lock:
            previous = self._cache_priorities.get(key, 0)
            if priority:
                self._cache_priorities[key] = priority
            else:
                self._cache_priorities.pop(key, None)
            if priority > previous:
                self._bump("cache_priority_promotions")

    def clear_cache_priorities(self) -> None:
        """Start a new cache-admission request with no stale residency hints."""

        coordinator = self._cache_coordinator
        if coordinator is None:
            return
        with coordinator.lock:
            self._cache_priorities.clear()

    def update_file_info(
        self,
        filename: str,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Merge source metadata through the upstream's synchronization API."""

        update = getattr(self.upstream, "update_file_info", None)
        if callable(update):
            return dict(update(filename, values))
        with self._lock:
            info = self.file_info.setdefault(filename, {})
            info.update(values)
            return dict(info)

    def file_info_snapshot(self, filename: str) -> dict[str, Any]:
        snapshot = getattr(self.upstream, "file_info_snapshot", None)
        if callable(snapshot):
            return dict(snapshot(filename))
        with self._lock:
            return dict(self.file_info.get(filename, {}))

    def _identity(self, filename: str) -> dict[str, str]:
        identity_snapshot = getattr(self.upstream, "source_identity_snapshot", None)
        if callable(identity_snapshot):
            return {
                str(key): str(value)
                for key, value in identity_snapshot(filename).items()
            }
        current_identity = getattr(self.upstream, "source_identity", None)
        if callable(current_identity):
            try:
                return {
                    str(key): str(value)
                    for key, value in current_identity(filename).items()
                }
            except FileNotFoundError as exc:
                raise CacheIntegrityError(
                    f"Cache-Quelldatei fehlt inzwischen: {filename}"
                ) from exc
        info = self.file_info_snapshot(filename)
        return {
            key: str(info[key])
            for key in ("cas_url_hash", "etag", "size")
            if info.get(key) is not None
        }

    def _cache_key(
        self, kind: str, filename: str, start: int | None, end: int | None
    ) -> tuple[str, dict[str, Any]]:
        contract = {
            "repo": self.repo,
            "revision": self.rev,
            "kind": kind,
            "filename": filename,
            "start": start,
            "end": end,
        }
        return _sha256(_canonical_json(contract)), contract

    def _cache_paths(self, kind: str, key: str) -> tuple[Path, Path] | None:
        if self.cache_dir is None:
            return None
        directory = self.cache_dir / ("ranges" if kind == "range" else "files")
        return directory / f"{key}.bin", directory / f"{key}.json"

    def _cache_entries_locked(
        self,
    ) -> list[tuple[int, str, Path, Path, int]]:
        """List complete owned cache pairs; caller holds the coordinator lock."""

        if self.cache_dir is None:
            return []
        entries: list[tuple[int, str, Path, Path, int]] = []
        for leaf in ("ranges", "files"):
            directory = self.cache_dir / leaf
            try:
                children = tuple(directory.iterdir())
            except FileNotFoundError:
                continue
            except OSError:
                # A cache measurement must never expand the deletion scope or
                # make tensor reads fail merely because cache metadata is odd.
                continue
            for meta_path in children:
                match = re.fullmatch(r"([0-9a-f]{64})\.json", meta_path.name)
                if match is None or meta_path.is_symlink():
                    continue
                key = match.group(1)
                blob_path = directory / f"{key}.bin"
                if blob_path.is_symlink():
                    continue
                try:
                    if not meta_path.is_file() or not blob_path.is_file():
                        continue
                    meta_stat = meta_path.stat()
                    blob_stat = blob_path.stat()
                except OSError:
                    continue
                size = int(meta_stat.st_size) + int(blob_stat.st_size)
                entries.append(
                    (int(meta_stat.st_mtime_ns), key, blob_path, meta_path, size)
                )
        return entries

    def _record_cache_state(
        self,
        cache_bytes: int,
        *,
        evictions: int = 0,
        evicted_bytes: int = 0,
    ) -> None:
        with self._lock:
            self._stats["cache_bytes"] = int(cache_bytes)
            self._stats["cache_evictions"] += int(evictions)
            self._stats["cache_evicted_bytes"] += int(evicted_bytes)

    def _refresh_cache_bytes(self) -> int:
        coordinator = self._cache_coordinator
        if coordinator is None:
            self._record_cache_state(0)
            return 0
        with coordinator.lock:
            total = sum(item[4] for item in self._cache_entries_locked())
            self._record_cache_state(total)
            return total

    def _evict_locked(
        self,
        entries: list[tuple[int, str, Path, Path, int]],
        total: int,
        target: int,
        *,
        protected_keys: frozenset[str] = frozenset(),
    ) -> tuple[int, int, int]:
        """Evict verified complete LRU pairs below ``target`` bytes."""

        coordinator = self._cache_coordinator
        if coordinator is None:
            return total, 0, 0
        evictions = 0
        evicted_bytes = 0
        for _recency, key, blob_path, meta_path, size in sorted(
            entries,
            key=lambda item: (
                self._cache_priorities.get(item[1], 0),
                item[0],
                item[1],
            ),
        ):
            if total <= target:
                break
            active_lock = coordinator.key_locks.get(key)
            if key in protected_keys or (
                active_lock is not None and active_lock.locked()
            ):
                continue
            try:
                # Payload first: an exceptional partial cleanup can only leave
                # the tiny metadata file behind, never the large tensor range.
                blob_path.unlink()
                meta_path.unlink()
            except FileNotFoundError:
                continue
            except OSError:
                continue
            total -= size
            evictions += 1
            evicted_bytes += size
            self._cache_priorities.pop(key, None)
        return total, evictions, evicted_bytes

    def _enforce_cache_limit(self) -> None:
        coordinator = self._cache_coordinator
        if coordinator is None:
            self._record_cache_state(0)
            return
        with coordinator.lock:
            entries = self._cache_entries_locked()
            total = sum(item[4] for item in entries)
            evictions = 0
            evicted_bytes = 0
            if self.max_cache_bytes is not None and total > self.max_cache_bytes:
                total, evictions, evicted_bytes = self._evict_locked(
                    entries,
                    total,
                    self.max_cache_bytes,
                )
            self._record_cache_state(
                total,
                evictions=evictions,
                evicted_bytes=evicted_bytes,
            )

    def _load_cache(
        self,
        kind: str,
        key: str,
        contract: Mapping[str, Any],
        expected_size: int | None,
    ) -> bytes | None:
        paths = self._cache_paths(kind, key)
        if paths is None or bool(getattr(self._cache_bypass, "active", False)):
            return None
        blob_path, meta_path = paths
        pending_path = meta_path.with_suffix(".pending")
        if pending_path.exists() or pending_path.is_symlink():
            if pending_path.is_symlink() or not pending_path.is_file():
                raise CacheIntegrityError(
                    f"Cache-Transaktionsmarker ist ungueltig: {pending_path}"
                )
            # A zero-byte marker means the process stopped between the two
            # atomic pair writes. Only this exact cache key is discarded; the
            # source bytes are fetched and verified again below.
            for partial in (blob_path, meta_path, pending_path):
                try:
                    partial.unlink()
                except FileNotFoundError:
                    pass
            self._cache_priorities.pop(key, None)
            self._bump("cache_recoveries")
            return None
        if not blob_path.exists() and not meta_path.exists():
            return None
        if blob_path.is_symlink() or meta_path.is_symlink():
            raise CacheIntegrityError(f"Cache-Eintrag darf kein Symlink sein: {key}")
        if not blob_path.is_file() or not meta_path.is_file():
            raise CacheIntegrityError(f"Partieller Cache-Eintrag: {key}")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CacheIntegrityError(f"Cache-Metadaten unlesbar: {meta_path}") from exc
        if meta.get("schema") != self._CACHE_SCHEMA or meta.get("contract") != dict(
            contract
        ):
            raise CacheIntegrityError(f"Cache-Contract stimmt nicht: {meta_path}")
        try:
            body = blob_path.read_bytes()
        except OSError as exc:
            raise CacheIntegrityError(f"Cache-Payload unlesbar: {blob_path}") from exc
        if int(meta.get("size", -1)) != len(body):
            raise CacheIntegrityError(f"Cache-Laenge stimmt nicht: {blob_path}")
        if expected_size is not None and len(body) != expected_size:
            raise CacheIntegrityError(
                f"Cache-Range hat {len(body)} statt {expected_size} Bytes: {blob_path}"
            )
        if meta.get("sha256") != _sha256(body):
            raise CacheIntegrityError(f"Cache-SHA256 stimmt nicht: {blob_path}")
        known_identity = self._identity(str(contract["filename"]))
        cached_identity = meta.get("source_identity")
        if known_identity:
            if not isinstance(cached_identity, Mapping) or not cached_identity:
                raise CacheIntegrityError(
                    f"Quellidentitaet fehlt im Cache-Eintrag: {blob_path}"
                )
            if known_identity != dict(cached_identity):
                raise CacheIntegrityError(
                    f"Quellidentitaet fuer Cache-Eintrag hat sich geaendert: {blob_path}"
                )
        self._bump("cache_hits")
        self._bump("cache_bytes_reused", len(body))
        self._bump("cache_integrity_checks")
        self._mark_priority(key)
        try:
            os.utime(meta_path, None, follow_symlinks=False)
        except OSError:
            # Recency is an optimization; verified cached bytes stay usable on
            # read-only or unusually mounted cache directories.
            pass
        return body

    def _write_cache(
        self,
        kind: str,
        key: str,
        contract: Mapping[str, Any],
        body: bytes | memoryview,
    ) -> None:
        paths = self._cache_paths(kind, key)
        if paths is None:
            return
        blob_path, meta_path = paths
        pending_path = meta_path.with_suffix(".pending")
        meta = {
            "schema": self._CACHE_SCHEMA,
            "contract": dict(contract),
            "source_identity": self._identity(str(contract["filename"])),
            "size": len(body),
            "sha256": _sha256(body),
        }
        encoded_meta = _canonical_json(meta)
        entry_size = len(body) + len(encoded_meta)
        coordinator = self._cache_coordinator
        if coordinator is None:
            return
        if self.max_cache_bytes is None:
            _atomic_write(pending_path, b"")
            try:
                _atomic_write(blob_path, body)
                _atomic_write(meta_path, encoded_meta)
            except BaseException:
                for partial in (blob_path, meta_path, pending_path):
                    try:
                        partial.unlink()
                    except FileNotFoundError:
                        pass
                raise
            pending_path.unlink()
            self._bump("cache_writes")
            self._bump("cache_bytes_written", len(body))
            self._mark_priority(key)
            return

        evictions = 0
        evicted_bytes = 0
        skipped = False
        with coordinator.lock:
            entries = self._cache_entries_locked()
            total = sum(item[4] for item in entries)
            current_size = next(
                (item[4] for item in entries if item[1] == key),
                0,
            )
            if self.max_cache_bytes is not None:
                if entry_size > self.max_cache_bytes:
                    skipped = True
                else:
                    # Make room before the atomic payload write. This prevents
                    # one DeepSeek range from temporarily becoming persistent
                    # cache growth beyond the configured hard limit.
                    target = self.max_cache_bytes - entry_size + current_size
                    total, evictions, evicted_bytes = self._evict_locked(
                        entries,
                        total,
                        max(0, target),
                        protected_keys=frozenset({key}),
                    )
                    if total - current_size + entry_size > self.max_cache_bytes:
                        skipped = True
            if not skipped:
                # A same-key refresh cannot keep the old pair while staging
                # the replacement: the temporary payload would exceed a cap
                # sized for one entry. The key lock makes the remove+replace
                # sequence invisible to other readers in this process, and a
                # cache miss after a crash is safer than exceeding the bound.
                _atomic_write(pending_path, b"")
                try:
                    for old_path in (blob_path, meta_path):
                        try:
                            old_path.unlink()
                        except FileNotFoundError:
                            pass
                    _atomic_write(blob_path, body)
                    _atomic_write(meta_path, encoded_meta)
                except BaseException:
                    for partial in (blob_path, meta_path, pending_path):
                        try:
                            partial.unlink()
                        except FileNotFoundError:
                            pass
                    total -= current_size
                    self._cache_priorities.pop(key, None)
                    self._record_cache_state(
                        total,
                        evictions=evictions,
                        evicted_bytes=evicted_bytes,
                    )
                    raise
                pending_path.unlink()
                total = total - current_size + entry_size
            self._record_cache_state(
                total,
                evictions=evictions,
                evicted_bytes=evicted_bytes,
            )

        if skipped:
            self._bump("cache_write_skips_oversize")
            return
        self._bump("cache_writes")
        self._bump("cache_bytes_written", len(body))
        self._mark_priority(key)

    def get_range(self, filename: str, start: int, end: int) -> bytes:
        start, end = _validate_inclusive_range(start, end)
        expected = end - start + 1
        self._bump("range_logical_leaves")
        self._bump("range_logical_leaf_bytes", expected)
        key, contract = self._cache_key("range", filename, start, end)
        with self._locked_cache_key(key):
            cached = self._load_cache("range", key, contract, expected)
            if cached is not None:
                return cached
            self._bump("cache_misses")
            overhead = int(
                getattr(
                    self.upstream,
                    "range_overhead_reserve",
                    8192 if isinstance(self.upstream, HFRangeReader) else 0,
                )
            )
            reserve = expected + max(0, overhead)
            with self.budget.reservation(
                reserve, f"range:{filename}:{start}-{end}"
            ) as receipt:
                self._bump("range_requests")
                self._bump("range_bytes_requested", expected)
                self._bump("range_source_requests")
                try:
                    result = self.upstream.get_range(filename, start, end)
                    body = bytes(result)
                    if len(body) != expected:
                        raise RangeValidationError(
                            f"Range {filename}[{start}:{end}] lieferte "
                            f"{len(body)} statt {expected} Bytes"
                        )
                    charged_body = int(receipt["body"])
                    if charged_body == 0:
                        self.budget.charge(
                            len(body), 0, f"range:{filename}:{start}"
                        )
                    elif charged_body < len(body):
                        raise RangeValidationError(
                            f"Reader verbuchte fuer {filename} weniger als die gelieferte Bytezahl"
                        )
                    self._bump("range_source_bytes", int(receipt["body"]))
                except Exception:
                    self._bump("failed_requests")
                    raise
            self._write_cache("range", key, contract, body)
            return body

    def get_ranges(
        self,
        filename: str,
        ranges: tuple[tuple[int, int], ...],
        resident_limit_bytes: int,
    ) -> RawBytesManyResult:
        """Resolve exact half-open leaves, coalescing only cold adjacency."""

        nonempty = tuple((start, length) for start, length in ranges if length)
        self._bump("range_logical_leaves", len(nonempty))
        self._bump(
            "range_logical_leaf_bytes",
            sum(length for _start, length in nonempty),
        )
        if not nonempty:
            empty = memoryview(b"").toreadonly()
            return RawBytesManyResult(
                parts=tuple(empty for _item in ranges),
                resident_bytes=0,
                source_requests=0,
                source_bytes=0,
            )

        unique: dict[tuple[int, int], tuple[str, dict[str, Any]]] = {}
        for start, length in nonempty:
            bounds = (start, start + length - 1)
            if bounds not in unique:
                unique[bounds] = self._cache_key(
                    "range", filename, bounds[0], bounds[1]
                )
        resident_preflight = sum(end - start + 1 for start, end in unique)
        if resident_preflight > resident_limit_bytes:
            raise RangeValidationError(
                "Multi-Range-Resultat ueberschreitet resident_limit_bytes: "
                f"{resident_preflight}/{resident_limit_bytes} Bytes"
            )

        known_size: int | None = None
        info_size = self.file_info_snapshot(filename).get("size")
        if info_size is not None:
            try:
                known_size = int(info_size)
            except (TypeError, ValueError) as exc:
                raise RangeValidationError(
                    f"Ungueltige bekannte Quelldateigroesse fuer {filename!r}"
                ) from exc
        else:
            size_method = getattr(self.upstream, "file_size", None)
            if callable(size_method):
                known_size = int(size_method(filename))
        if known_size is not None:
            for start, end in unique:
                if end >= known_size:
                    raise RangeValidationError(
                        f"Range ausserhalb {filename}: [{start}, {end}] "
                        f"bei {known_size} Bytes"
                    )

        ordered_leaves = sorted(
            (
                (start, end, key, contract)
                for (start, end), (key, contract) in unique.items()
            ),
            key=lambda item: (item[2], item[0], item[1]),
        )
        resolved: dict[tuple[int, int], memoryview] = {}
        source_requests = 0
        source_bytes = 0
        with self._locked_cache_keys(item[2] for item in ordered_leaves):
            misses: list[tuple[int, int, str, dict[str, Any]]] = []
            for start, end, key, contract in ordered_leaves:
                cached = self._load_cache(
                    "range", key, contract, end - start + 1
                )
                if cached is None:
                    self._bump("cache_misses")
                    misses.append((start, end, key, contract))
                else:
                    resolved[(start, end)] = memoryview(cached).toreadonly()

            envelopes: list[
                tuple[int, int, list[tuple[int, int, str, dict[str, Any]]]]
            ] = []
            for leaf in sorted(misses, key=lambda item: (item[0], item[1], item[2])):
                if envelopes and leaf[0] == envelopes[-1][1] + 1:
                    envelope_start, _envelope_end, leaves = envelopes[-1]
                    leaves.append(leaf)
                    envelopes[-1] = (envelope_start, leaf[1], leaves)
                else:
                    envelopes.append((leaf[0], leaf[1], [leaf]))

            overhead = int(
                getattr(
                    self.upstream,
                    "range_overhead_reserve",
                    8192 if isinstance(self.upstream, HFRangeReader) else 0,
                )
            )
            planned_source_bytes = sum(
                end - start + 1 for start, end, _ in envelopes
            )
            reserve = planned_source_bytes + len(envelopes) * max(0, overhead)
            with self.budget.reservation(
                reserve, f"ranges:{filename}:{len(envelopes)}"
            ) as receipt:
                for envelope_start, envelope_end, leaves in envelopes:
                    expected = envelope_end - envelope_start + 1
                    before_receipt_body = int(receipt["body"])
                    self._bump("range_requests")
                    self._bump("range_bytes_requested", expected)
                    self._bump("range_source_requests")
                    source_requests += 1
                    try:
                        result = self.upstream.get_range(
                            filename, envelope_start, envelope_end
                        )
                        body = bytes(result)
                        if len(body) != expected:
                            raise RangeValidationError(
                                f"Range {filename}[{envelope_start}:{envelope_end}] "
                                f"lieferte {len(body)} statt {expected} Bytes"
                            )
                        charged_body = (
                            int(receipt["body"]) - before_receipt_body
                        )
                        if charged_body == 0:
                            self.budget.charge(
                                len(body),
                                0,
                                f"ranges:{filename}:{envelope_start}",
                            )
                        elif charged_body < len(body):
                            raise RangeValidationError(
                                f"Reader verbuchte fuer {filename} weniger als "
                                "die gelieferte Bytezahl"
                            )
                        physical_body = (
                            int(receipt["body"]) - before_receipt_body
                        )
                        self._bump("range_source_bytes", physical_body)
                        source_bytes += physical_body
                    except Exception:
                        self._bump("failed_requests")
                        raise

                    owner = memoryview(body).toreadonly()
                    for start, end, key, contract in leaves:
                        leaf = owner[
                            start - envelope_start : end - envelope_start + 1
                        ].toreadonly()
                        self._write_cache("range", key, contract, leaf)
                        resolved[(start, end)] = leaf

        empty = memoryview(b"").toreadonly()
        parts = tuple(
            empty if length == 0 else resolved[(start, start + length - 1)]
            for start, length in ranges
        )
        owners: dict[int, int] = {}
        for part in parts:
            owner = part.obj
            owners.setdefault(id(owner), memoryview(owner).nbytes)
        resident_bytes = sum(owners.values())
        if resident_bytes > resident_limit_bytes:  # pragma: no cover - invariant
            raise RangeValidationError(
                "Multi-Range-Owner ueberschreiten resident_limit_bytes: "
                f"{resident_bytes}/{resident_limit_bytes} Bytes"
            )
        return RawBytesManyResult(
            parts=parts,
            resident_bytes=resident_bytes,
            source_requests=source_requests,
            source_bytes=source_bytes,
        )

    def fetch_file(
        self,
        filename: str,
        max_bytes: int | None = None,
        *,
        _missing_is_expected: bool = False,
    ) -> bytes:
        if not isinstance(filename, str) or not filename:
            raise RangeValidationError("Dateiname muss ein nichtleerer String sein")
        if filename.lower().endswith(
            (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx", ".ckpt")
        ):
            raise RangeValidationError(
                f"Gewichtsdatei {filename!r} darf nicht voll geladen werden; "
                "get_range()/rows() verwenden"
            )
        ceiling = self.max_metadata_bytes if max_bytes is None else int(max_bytes)
        if ceiling <= 0 or ceiling > self.max_metadata_bytes:
            raise RangeValidationError(
                f"max_bytes muss in [1, {self.max_metadata_bytes}] liegen"
            )
        key, contract = self._cache_key("file", filename, None, None)
        with self._locked_cache_key(key):
            cached = self._load_cache("file", key, contract, None)
            if cached is not None:
                if len(cached) > ceiling:
                    raise RangeValidationError(
                        f"Gecachte Metadatei {filename!r} ueberschreitet max_bytes"
                    )
                return cached
            self._bump("cache_misses")
            self._bump("file_requests")
            before_thread_body = self.budget.thread_charge_snapshot()[0]
            known_size = None
            try:
                bounded_fetch = getattr(self.upstream, "fetch_file_bounded", None)
                if callable(bounded_fetch):
                    body = bytes(bounded_fetch(filename, ceiling))
                else:
                    size_method = getattr(self.upstream, "file_size", None)
                    if callable(size_method):
                        known_size = int(size_method(filename))
                        if known_size > ceiling:
                            raise RangeValidationError(
                                f"Metadatei {filename!r} ist {known_size} Bytes gross; "
                                f"Grenze ist {ceiling}"
                            )
                    if known_size is None:
                        body = bytes(self.upstream.fetch_file(filename))
                    else:
                        with self.budget.reservation(
                            known_size, f"file:{filename}"
                        ):
                            body = bytes(self.upstream.fetch_file(filename))
                if len(body) > ceiling:
                    raise RangeValidationError(
                        f"Reader lieferte fuer {filename!r} {len(body)} Bytes; "
                        f"Grenze ist {ceiling}"
                    )
                if known_size is not None and len(body) != known_size:
                    raise RangeValidationError(
                        f"Metadatei {filename!r} aenderte ihre Groesse beim Lesen"
                    )
                charged_body = (
                    self.budget.thread_charge_snapshot()[0] - before_thread_body
                )
                if charged_body == 0:
                    self.budget.charge(len(body), 0, f"file:{filename}")
                elif charged_body < len(body):
                    raise RangeValidationError(
                        f"Reader verbuchte fuer {filename} weniger als die gelieferte Bytezahl"
                    )
            except (FileNotFoundError, SourceNotFound) as exc:
                self._bump(
                    "optional_misses" if _missing_is_expected else "failed_requests"
                )
                if isinstance(exc, SourceNotFound):
                    raise
                raise SourceNotFound(str(exc)) from exc
            except Exception:
                self._bump("failed_requests")
                raise
            self._write_cache("file", key, contract, body)
            return body

    def fetch_optional_file(
        self,
        filename: str,
        max_bytes: int | None = None,
    ) -> bytes:
        """Read optional metadata without classifying absence as a failed request."""

        return self.fetch_file(
            filename,
            max_bytes,
            _missing_is_expected=True,
        )

    def fetch_st_header(
        self, filename: str
    ) -> tuple[dict[str, Any], int, dict[str, Any]]:
        first = self.get_range(filename, 0, 7)
        (header_length,) = struct.unpack("<Q", first)
        if not (0 < header_length <= min(64 * 1024 * 1024, self.max_metadata_bytes)):
            raise InventoryValidationError(
                f"Unsinnige Safetensors-Headerlaenge {header_length} bei {filename}"
            )
        encoded = self.get_range(filename, 8, 8 + header_length - 1)
        try:
            header = json.loads(encoded.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise InventoryValidationError(
                f"Safetensors-Header unlesbar bei {filename}"
            ) from exc
        if not isinstance(header, dict):
            raise InventoryValidationError(
                f"Safetensors-Header ist kein Objekt: {filename}"
            )
        metadata = header.pop("__metadata__", None)
        updates: dict[str, Any] = {"header_len": header_length}
        if isinstance(metadata, Mapping):
            updates["st_metadata"] = {
                str(key): str(value)[:200] for key, value in list(metadata.items())[:8]
            }
        info = self.update_file_info(filename, updates)
        return header, 8 + header_length, info


class Streamer:
    """Range-only tensor source with verified resume cache and hard budget."""

    _INVENTORY_SCHEMA = "immer.tensor-inventory-cache/v1"
    _ITEMSIZE = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "I16": 2,
        "U16": 2,
        "BF16": 2,
        "F16": 2,
        "F8_E4M3": 1,
        "F8_E4M3FN": 1,
        "F8_E8M0": 1,
        "I32": 4,
        "U32": 4,
        "F32": 4,
        "I64": 8,
        "U64": 8,
        "F64": 8,
    }
    _NUMPY_DTYPES = {
        "BOOL": "?",
        "U8": "u1",
        "I8": "i1",
        "I16": "<i2",
        "U16": "<u2",
        "F16": "<f2",
        "I32": "<i4",
        "U32": "<u4",
        "F32": "<f4",
        "I64": "<i8",
        "U64": "<u8",
        "F64": "<f8",
    }

    def __init__(
        self,
        repo_id: str,
        revision: str = "main",
        budget_mb: float = 200.0,
        *,
        reader: Any | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        use_cache: bool = True,
        max_metadata_bytes: int = 64 * 1024 * 1024,
        max_cache_bytes: int | None = None,
        verbose: bool = False,
    ) -> None:
        if not isinstance(repo_id, str) or not repo_id.strip():
            raise ValueError("repo_id muss ein nichtleerer String sein")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision muss ein nichtleerer String sein")
        self.repo_id = repo_id
        self.revision = revision
        self.budget = HardByteBudget(budget_mb)
        self._upstream = reader
        resolved_cache = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else _default_cache_dir()
        )
        self._cache_dir = resolved_cache if use_cache else None
        self._reader: _ContractReader | None = None
        self._inventory: dict[str, Any] | None = None
        self._tensor_index: dict[str, dict[str, Any]] | None = None
        self._tensor_index_inventory_id: int | None = None
        self._state_lock = threading.RLock()
        self._inventory_cache_hits = 0
        self._inventory_cache_writes = 0
        self._inventory_fingerprint: str | None = None
        self._max_metadata_bytes = int(max_metadata_bytes)
        if max_cache_bytes is not None and (
            isinstance(max_cache_bytes, bool)
            or not isinstance(max_cache_bytes, int)
            or max_cache_bytes < 0
        ):
            raise ValueError(
                "max_cache_bytes muss None oder eine nichtnegative Ganzzahl sein"
            )
        self._max_cache_bytes = max_cache_bytes
        self.verbose = bool(verbose)

    @classmethod
    def from_local(
        cls,
        root: str | os.PathLike[str],
        *,
        revision: str = "local",
        budget_mb: float = 200.0,
        cache_dir: str | os.PathLike[str] | None = None,
        use_cache: bool = True,
        max_metadata_bytes: int = 64 * 1024 * 1024,
        max_cache_bytes: int | None = None,
        verbose: bool = False,
    ) -> "Streamer":
        path = Path(root).expanduser().resolve()
        repo_id = f"local:{path}"
        local = LocalRangeReader(path, repo_id=repo_id, revision=revision)
        return cls(
            repo_id,
            revision=revision,
            budget_mb=budget_mb,
            reader=local,
            cache_dir=cache_dir,
            use_cache=use_cache,
            max_metadata_bytes=max_metadata_bytes,
            max_cache_bytes=max_cache_bytes,
            verbose=verbose,
        )

    @property
    def reader(self) -> _ContractReader:
        reader = self._reader
        if reader is not None:
            return reader
        with self._state_lock:
            reader = self._reader
            if reader is not None:
                return reader
            upstream = self._upstream
            if upstream is None:
                upstream = HFRangeReader(
                    self.repo_id,
                    revision=self.revision,
                    budget=self.budget,
                )
            if hasattr(upstream, "repo"):
                upstream.repo = self.repo_id
            if hasattr(upstream, "rev"):
                upstream.rev = self.revision
            reader = _ContractReader(
                upstream,
                self.budget,
                self._cache_dir,
                max_metadata_bytes=self._max_metadata_bytes,
                max_cache_bytes=self._max_cache_bytes,
            )
            self._reader = reader
            return reader

    def _cache_slug(self) -> str:
        readable = re.sub(r"[^A-Za-z0-9_-]+", "-", self.repo_id).strip("-")[:48]
        identity = _sha256(f"{self.repo_id}\0{self.revision}".encode("utf-8"))[:16]
        return f"{readable or 'source'}-{identity}"

    def _cache_path(self) -> Path:
        if self._cache_dir is None:
            return Path(__file__).resolve().parents[3] / "hf-cache" / "disabled.json"
        return self._cache_dir / "inventories" / f"{self._cache_slug()}.json"

    def _legacy_cache_path(self) -> Path:
        slug = self.repo_id.replace("/", "_").replace(".", "-")
        return Path(__file__).resolve().parents[3] / "results" / f"hf_scan_{slug}.json"

    @staticmethod
    def _source_fingerprint(inventory: Mapping[str, Any]) -> str:
        shards = []
        for raw in inventory.get("shards", []):
            if not isinstance(raw, Mapping):
                continue
            shards.append(
                {
                    key: raw.get(key)
                    for key in (
                        "file",
                        "size",
                        "etag",
                        "cas_url_hash",
                        "header_len",
                        "data_start",
                    )
                }
            )
        return _sha256(
            _canonical_json(sorted(shards, key=lambda item: str(item["file"])))
        )

    def _inventory_envelope(self, inventory: Mapping[str, Any]) -> dict[str, Any]:
        document = dict(inventory)
        return {
            "schema": self._INVENTORY_SCHEMA,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "inventory_sha256": _sha256(_canonical_json(document)),
            "source_fingerprint": self._source_fingerprint(document),
            "inventory": document,
        }

    def _load_inventory_path(
        self, path: Path, *, allow_legacy: bool
    ) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CacheIntegrityError(f"Inventar-Cache unlesbar: {path}") from exc
        if isinstance(raw, Mapping) and raw.get("schema") == self._INVENTORY_SCHEMA:
            if (
                raw.get("repo_id") != self.repo_id
                or raw.get("revision") != self.revision
            ):
                raise CacheIntegrityError(
                    f"Inventar-Cache gehoert zu anderer Quelle: {path}"
                )
            document = raw.get("inventory")
            if not isinstance(document, Mapping):
                raise CacheIntegrityError(f"Inventar-Payload fehlt: {path}")
            document = dict(document)
            if raw.get("inventory_sha256") != _sha256(_canonical_json(document)):
                raise CacheIntegrityError(f"Inventar-SHA256 stimmt nicht: {path}")
            fingerprint = self._source_fingerprint(document)
            if raw.get("source_fingerprint") != fingerprint:
                raise CacheIntegrityError(
                    f"Inventar-Quellfingerprint stimmt nicht: {path}"
                )
        elif allow_legacy and isinstance(raw, Mapping):
            document = dict(raw)
            # The historical cache path was keyed only by the repository
            # slug.  A mutable ``main`` scan can therefore sit next to a
            # pinned snapshot.  It is not corruption; it is simply not a
            # cache candidate for this source and must be rescanned.
            if (
                document.get("repo") != self.repo_id
                or document.get("revision") != self.revision
            ):
                return None
        else:
            raise CacheIntegrityError(f"Unbekanntes Inventar-Cacheformat: {path}")
        self._validate_inventory(document)
        self._validate_cached_local_source(document, path)
        self._inventory_fingerprint = self._source_fingerprint(document)
        self._hydrate_reader_identity(document)
        self._inventory_cache_hits += 1
        return document

    def _validate_cached_local_source(
        self,
        document: Mapping[str, Any],
        cache_path: Path,
    ) -> None:
        """Reject a local resume cache when the underlying files changed."""

        current_identity = getattr(self.reader.upstream, "source_identity", None)
        if not callable(current_identity):
            return
        for shard in document.get("shards", ()):
            if not isinstance(shard, Mapping) or not isinstance(shard.get("file"), str):
                continue
            filename = shard["file"]
            cached = {
                key: str(shard[key])
                for key in ("size", "etag")
                if shard.get(key) is not None
            }
            try:
                current = {
                    str(key): str(value)
                    for key, value in current_identity(filename).items()
                }
            except FileNotFoundError as exc:
                raise CacheIntegrityError(
                    f"Lokale Quelle fuer Inventar-Cache fehlt: {filename}"
                ) from exc
            if cached and cached != current:
                raise CacheIntegrityError(
                    f"Lokale Quelle hat sich seit dem Inventar-Cache geaendert: "
                    f"{filename} ({cache_path})"
                )

    def _write_inventory_cache(self, inventory: Mapping[str, Any]) -> None:
        if self._cache_dir is None:
            return
        envelope = self._inventory_envelope(inventory)
        _atomic_write(self._cache_path(), _canonical_json(envelope))
        self._inventory_cache_writes += 1

    def _hydrate_reader_identity(self, inventory: Mapping[str, Any]) -> None:
        for shard in inventory.get("shards", []):
            if not isinstance(shard, Mapping) or not isinstance(shard.get("file"), str):
                continue
            values: dict[str, Any] = {}
            for key in ("size", "etag", "cas_url_hash", "header_len"):
                if shard.get(key) is not None:
                    values[key] = shard[key]
            if values:
                self.reader.update_file_info(shard["file"], values)

    def _validate_inventory(self, document: Mapping[str, Any]) -> None:
        if (
            document.get("repo") != self.repo_id
            or document.get("revision") != self.revision
        ):
            raise InventoryValidationError(
                "Inventarquelle stimmt nicht mit repo_id/revision des Streamers ueberein"
            )
        tensors = document.get("tensors")
        if not isinstance(tensors, list):
            raise InventoryValidationError("Inventar enthaelt keine Tensorliste")
        seen: set[str] = set()
        for index, entry in enumerate(tensors):
            if not isinstance(entry, Mapping):
                raise InventoryValidationError(
                    f"Tensor-Eintrag {index} ist kein Objekt"
                )
            name = entry.get("name")
            if not isinstance(name, str) or not name or name in seen:
                raise InventoryValidationError(
                    f"Ungueltiger/doppelter Tensorname: {name!r}"
                )
            seen.add(name)
            shape = entry.get("shape")
            if not isinstance(shape, list) or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 0
                for dim in shape
            ):
                raise InventoryValidationError(f"Ungueltige Shape bei {name}")
            offsets = entry.get("offset_in_shard")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in offsets
                )
                or offsets[0] < 0
                or offsets[1] < offsets[0]
            ):
                raise InventoryValidationError(f"Ungueltige Offsets bei {name}")
            if not isinstance(entry.get("shard"), str) or not entry["shard"]:
                raise InventoryValidationError(f"Shard fehlt bei {name}")
            data_start = entry.get("data_start")
            if (
                isinstance(data_start, bool)
                or not isinstance(data_start, int)
                or data_start < 8
            ):
                raise InventoryValidationError(f"data_start ungueltig bei {name}")
            dtype = str(entry.get("dtype", "")).upper()
            itemsize = self._ITEMSIZE.get(dtype)
            if itemsize is not None:
                numel = 1
                for dim in shape:
                    numel *= dim
                expected = numel * itemsize
                if offsets[1] - offsets[0] != expected:
                    raise InventoryValidationError(
                        f"Shape/Offset-Bytezahl stimmt bei {name} nicht: "
                        f"{expected} != {offsets[1] - offsets[0]}"
                    )

    def inventory(self, *, refresh: bool = False) -> dict[str, Any]:
        """Return the header-only inventory, using a verified resume cache."""
        with self._state_lock:
            return self._inventory_locked(refresh=refresh)

    def _inventory_locked(self, *, refresh: bool) -> dict[str, Any]:
        if self._inventory is not None and not refresh:
            return self._inventory
        if not refresh and self._cache_dir is not None:
            cached = self._load_inventory_path(self._cache_path(), allow_legacy=False)
            if cached is None:
                cached = self._load_inventory_path(
                    self._legacy_cache_path(), allow_legacy=True
                )
                if cached is not None:
                    self._write_inventory_cache(cached)
            if cached is not None:
                self._inventory = cached
                return self._inventory

        candidate: dict[str, Any] | None = None
        last: Exception | None = None
        for attempt in range(3):
            try:
                if refresh:
                    with self.reader.uncached():
                        candidate = scan_inventory(self.reader, budget=self.budget)
                else:
                    candidate = scan_inventory(self.reader, budget=self.budget)
                self._validate_inventory(candidate)
                break
            except (ConnectionError, TimeoutError, OSError) as exc:
                last = exc
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
        if candidate is None:
            if last is not None:
                raise TensorSourceError(
                    f"Inventar nach 3 Versuchen fehlgeschlagen: {last}"
                ) from last
            raise TensorSourceError("Inventar konnte nicht erstellt werden")
        self._inventory = candidate
        self._inventory_fingerprint = self._source_fingerprint(self._inventory)
        self._hydrate_reader_identity(self._inventory)
        self._write_inventory_cache(self._inventory)
        return self._inventory

    def _ensure_tensor_index_locked(
        self,
        inventory: Mapping[str, Any],
    ) -> dict[str, dict[str, Any]]:
        if self._tensor_index is None or self._tensor_index_inventory_id != id(
            inventory
        ):
            self._tensor_index = {
                str(entry["name"]): dict(entry)
                for entry in inventory.get("tensors", [])
            }
            self._tensor_index_inventory_id = id(inventory)
        return self._tensor_index

    def prepare_parallel_reads(self) -> int:
        """Materialize shared metadata before launching ``raw_bytes`` workers.

        Call this on the owner/main thread, resolve tensor names there with
        :meth:`find`, and give workers only exact shard/offset/length triples.
        Workers then share one contract reader and one hard byte budget while
        independent cache keys remain able to perform upstream I/O in parallel.

        Returns the number of indexed tensors.
        """

        with self._state_lock:
            self.reader
            inventory = self._inventory_locked(refresh=False)
            return len(self._ensure_tensor_index_locked(inventory))

    def tensors(self) -> list[dict[str, Any]]:
        """Return a shallow copy of all tensor metadata entries."""
        return [dict(entry) for entry in self.inventory().get("tensors", [])]

    def find(self, name: str) -> dict[str, Any]:
        if not isinstance(name, str) or not name:
            raise ValueError("Tensorname muss ein nichtleerer String sein")
        with self._state_lock:
            inventory = self._inventory_locked(refresh=False)
            entry = self._ensure_tensor_index_locked(inventory).get(name)
        if entry is not None:
            return dict(entry)
        raise KeyError(f"Tensor {name!r} nicht im Inventar von {self.repo_id}")

    @classmethod
    def _decode_payload(
        cls,
        raw: bytes,
        dtype: str,
        shape: tuple[int, ...],
    ) -> Any:
        """Decode one exact safetensors payload into an owned NumPy array."""
        import numpy as np

        dtype = str(dtype).upper()
        itemsize = cls._ITEMSIZE.get(dtype)
        if itemsize is None or (
            dtype not in {"BF16", "F8_E4M3", "F8_E4M3FN", "F8_E8M0"}
            and dtype not in cls._NUMPY_DTYPES
        ):
            raise RangeValidationError(
                f"Nicht unterstuetztes safetensors-dtype {dtype!r}"
            )
        numel = 1
        for dim in shape:
            numel *= int(dim)
        expected = numel * itemsize
        if len(raw) != expected:
            raise InventoryValidationError(
                f"Tensor-Payload hat {len(raw)} statt {expected} Bytes fuer "
                f"dtype={dtype}, shape={list(shape)}"
            )

        # Float8 weights are dequantized downstream under the assumption that
        # every stored scale/value is finite.  Reject the reserved encodings in
        # the raw payload instead of allowing a NaN to silently poison an
        # activation.  ``find`` keeps this check allocation-free for the large
        # streamed tensors while the minimum index makes the error stable when
        # more than one invalid byte is present.
        invalid_index = -1
        if dtype in {"F8_E4M3", "F8_E4M3FN"}:
            positive_nan = raw.find(b"\x7f")
            negative_nan = raw.find(b"\xff")
            present = (index for index in (positive_nan, negative_nan) if index >= 0)
            invalid_index = min(present, default=-1)
        elif dtype == "F8_E8M0":
            invalid_index = raw.find(b"\xff")
        if invalid_index >= 0:
            raise TensorEncodingError(
                f"Reservierte/nicht-endliche {dtype}-Kodierung "
                f"0x{raw[invalid_index]:02X} bei Element {invalid_index}"
            )

        if dtype == "BF16":
            words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            words <<= 16
            decoded = words.view(np.float32)
        elif dtype in {"F8_E4M3", "F8_E4M3FN"}:
            bits = np.frombuffer(raw, dtype=np.uint8)
            exponent = ((bits >> 3) & 0x0F).astype(np.int16)
            mantissa = (bits & 0x07).astype(np.float32)
            decoded = np.empty(bits.shape, dtype=np.float32)
            subnormal = exponent == 0
            decoded[subnormal] = np.ldexp(mantissa[subnormal], -9)
            normal = ~subnormal
            decoded[normal] = np.ldexp(
                np.float32(1.0) + mantissa[normal] * np.float32(0.125),
                exponent[normal] - 7,
            )
            signs = np.where(
                bits & 0x80,
                np.float32(-1.0),
                np.float32(1.0),
            )
            decoded = np.copysign(decoded, signs)
        elif dtype == "F8_E8M0":
            bits = np.frombuffer(raw, dtype=np.uint8)
            finite_bits = np.minimum(bits, np.uint8(0xFE))
            decoded = np.ldexp(
                np.ones(bits.shape, dtype=np.float32),
                finite_bits.astype(np.int16) - 127,
            )
        else:
            decoded = np.frombuffer(raw, dtype=np.dtype(cls._NUMPY_DTYPES[dtype]))
        return decoded.reshape(shape).copy()

    def tensor(self, tensor_name: str) -> Any:
        """Read and decode one complete tensor with one exact payload range.

        Standard safetensors dtypes retain their native NumPy dtype. BF16 and
        OCP float8 encodings are returned as writable float32 arrays.
        """
        meta = self.find(tensor_name)
        shape = tuple(int(dim) for dim in meta["shape"])
        dtype = str(meta["dtype"]).upper()
        itemsize = self._ITEMSIZE.get(dtype)
        if itemsize is None or (
            dtype not in {"BF16", "F8_E4M3", "F8_E4M3FN", "F8_E8M0"}
            and dtype not in self._NUMPY_DTYPES
        ):
            raise RangeValidationError(
                f"Nicht unterstuetztes safetensors-dtype {dtype!r}"
            )
        offset_begin, offset_end = (int(value) for value in meta["offset_in_shard"])
        length = offset_end - offset_begin
        expected = itemsize
        for dim in shape:
            expected *= dim
        if length != expected:
            raise InventoryValidationError(
                f"Shape/Offset-Bytezahl stimmt bei {tensor_name} nicht: "
                f"{expected} != {length}"
            )
        absolute = int(meta["data_start"]) + offset_begin
        raw = self.raw_bytes(str(meta["shard"]), absolute, length)
        return self._decode_payload(raw, dtype, shape)

    def rows(
        self,
        tensor_name: str,
        start_row: int = 0,
        n_rows: int = 8,
        n_blocks: int = 8,
    ) -> Any:
        """Read exactly ``n_rows`` contiguous rows as an owned NumPy array.

        ``n_blocks`` remains accepted for compatibility. Exact retrieval is one
        contiguous range; stochastic block sampling belongs in analysis code.
        """
        del n_blocks
        start_row = _validate_nonnegative_int(start_row, "start_row")
        n_rows = _validate_nonnegative_int(n_rows, "n_rows")
        meta = self.find(tensor_name)
        shape = meta["shape"]
        if len(shape) != 2:
            raise RangeValidationError(
                f"rows() braucht einen 2D-Tensor, {tensor_name!r} hat Shape {shape}"
            )
        total_rows, n_cols = int(shape[0]), int(shape[1])
        if start_row > total_rows or n_rows > total_rows - start_row:
            raise RangeValidationError(
                f"Zeilen [{start_row}, {start_row + n_rows}) ausserhalb "
                f"{tensor_name}[0:{total_rows})"
            )
        dtype = str(meta["dtype"]).upper()
        itemsize = self._ITEMSIZE.get(dtype)
        if itemsize is None or (
            dtype not in {"BF16", "F8_E4M3", "F8_E4M3FN", "F8_E8M0"}
            and dtype not in self._NUMPY_DTYPES
        ):
            raise RangeValidationError(
                f"Nicht unterstuetztes safetensors-dtype {dtype!r}"
            )
        if n_rows == 0:
            return self._decode_payload(b"", dtype, (0, n_cols))

        row_bytes = n_cols * itemsize
        offset_begin, offset_end = (int(value) for value in meta["offset_in_shard"])
        relative = offset_begin + start_row * row_bytes
        length = n_rows * row_bytes
        if relative + length > offset_end:
            raise InventoryValidationError(
                f"Zeilenrange ueberschreitet Tensor-Offsets bei {tensor_name}"
            )
        absolute = int(meta["data_start"]) + relative
        raw = self.raw_bytes(str(meta["shard"]), absolute, length)
        return self._decode_payload(raw, dtype, (n_rows, n_cols))

    def rows_torch(
        self,
        tensor_name: str,
        start_row: int = 0,
        n_rows: int = 8,
        *,
        dtype: Any | None = None,
        device: str | Any = "cpu",
    ) -> Any:
        """Bridge selected NumPy rows to torch without loading donor weights."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise TensorSourceError(
                "rows_torch() braucht das optionale Paket torch"
            ) from exc
        array = self.rows(tensor_name, start_row=start_row, n_rows=n_rows)
        result = torch.from_numpy(array)
        if dtype is not None or str(device) != "cpu":
            result = result.to(device=device, dtype=dtype or result.dtype)
        result.requires_grad_(False)
        return result

    torch_rows = rows_torch

    def budget_line(self) -> str:
        stats = self.reader.stats()
        return (
            f"{self.budget.total / 1048576:.3f}/{self.budget.limit / 1048576:.3f} MB "
            f"(body={self.budget.body / 1048576:.3f} MB, "
            f"reqs={self.budget.requests}, cache_hits={stats['cache_hits']})"
        )

    def raw_bytes(self, shard: str, offset: int, length: int) -> bytes:
        offset = _validate_nonnegative_int(offset, "offset")
        length = _validate_nonnegative_int(length, "length")
        if not isinstance(shard, str) or not shard:
            raise RangeValidationError("shard muss ein nichtleerer String sein")
        if length == 0:
            return b""
        # HFRangeReader uses an inclusive end. The old facade accidentally
        # requested length+1 bytes here.
        return self.reader.get_range(shard, offset, offset + length - 1)

    def raw_bytes_many(
        self,
        shard: str,
        ranges: Iterable[tuple[int, int]],
        resident_limit_bytes: int,
        *,
        max_gap_bytes: int = 0,
    ) -> RawBytesManyResult:
        """Read exact leaves and merge only adjacent cold source ranges.

        Cache identity stays leaf-exact, so scalar and batch reads warm one
        another. Gap reads are intentionally unsupported: transferred bytes
        must always belong to a requested leaf.
        """

        if not isinstance(shard, str) or not shard:
            raise RangeValidationError("shard muss ein nichtleerer String sein")
        resident_limit_bytes = _validate_nonnegative_int(
            resident_limit_bytes, "resident_limit_bytes"
        )
        max_gap_bytes = _validate_nonnegative_int(max_gap_bytes, "max_gap_bytes")
        if max_gap_bytes != 0:
            raise RangeValidationError(
                "max_gap_bytes muss 0 bleiben; Gap-Bytes sind nicht zugelassen"
            )
        try:
            raw_ranges = tuple(ranges)
        except TypeError as exc:
            raise RangeValidationError(
                "ranges muss ein Iterable aus (offset, length)-Paaren sein"
            ) from exc
        validated: list[tuple[int, int]] = []
        for index, raw in enumerate(raw_ranges):
            try:
                pair = tuple(raw)
            except TypeError as exc:
                raise RangeValidationError(
                    f"Range {index} ist kein (offset, length)-Paar"
                ) from exc
            if len(pair) != 2:
                raise RangeValidationError(
                    f"Range {index} ist kein (offset, length)-Paar"
                )
            offset = _validate_nonnegative_int(pair[0], f"ranges[{index}].offset")
            length = _validate_nonnegative_int(pair[1], f"ranges[{index}].length")
            validated.append((offset, length))
        if not validated:
            return RawBytesManyResult((), 0, 0, 0)
        return self.reader.get_ranges(
            shard,
            tuple(validated),
            resident_limit_bytes,
        )

    @contextmanager
    def cache_priority(self, priority: int) -> Any:
        """Give ranges read in this scope a reader-local eviction priority."""

        with self.reader.cache_priority(priority):
            yield

    def clear_cache_priorities(self) -> None:
        """Clear admission hints before an independent model request."""

        self.reader.clear_cache_priorities()

    def bytes_moved(self) -> int:
        """Network/local source body bytes; verified cache hits count as zero."""
        return int(self.budget.body)

    def metrics(self) -> dict[str, Any]:
        reader_stats = self.reader.stats()
        transport_method = getattr(self.reader.upstream, "transport_metrics", None)
        transport = dict(transport_method()) if callable(transport_method) else {}
        pinned_revision = bool(re.fullmatch(r"[0-9a-fA-F]{40,64}", self.revision))
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "revision_is_pinned": pinned_revision
            and not self.repo_id.startswith("local:"),
            "revision_is_mutable": self.repo_id.startswith("local:")
            or not pinned_revision,
            "budget": self.budget.as_dict(),
            "network_or_source_body_bytes": int(self.budget.body),
            "inventory_cache_hits": int(self._inventory_cache_hits),
            "inventory_cache_writes": int(self._inventory_cache_writes),
            "inventory_source_fingerprint": self._inventory_fingerprint,
            **transport,
            **reader_stats,
        }

    def close(self) -> None:
        """Close persistent source transport after all reads have finished."""

        with self._state_lock:
            upstream = self._upstream
            if upstream is None and self._reader is not None:
                upstream = self._reader.upstream
        close = getattr(upstream, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> Streamer:
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def available() -> bool:
    return True
