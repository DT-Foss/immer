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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ._hf_source import (
    Budget,
    BudgetExceeded,
    HFRangeReader,
    SourceNotFound,
    bf16_rows_to_f32,
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


@runtime_checkable
class TensorSource(Protocol):
    """Small public contract consumed by retrieval and analysis code."""

    def inventory(self, *, refresh: bool = False) -> dict[str, Any]: ...

    def find(self, name: str) -> dict[str, Any]: ...

    def rows(
        self,
        tensor_name: str,
        start_row: int = 0,
        n_rows: int = 8,
        n_blocks: int = 8,
    ) -> Any: ...

    def raw_bytes(self, shard: str, offset: int, length: int) -> bytes: ...

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

    def charge(self, body: int, overhead: int, tag: str) -> None:
        body = int(body)
        overhead = int(overhead)
        if body < 0 or overhead < 0:
            raise ValueError("Budget-Charge darf nicht negativ sein")
        with self._charge_lock:
            attempted = self.total + body + overhead
            if attempted > self.limit:
                self.rejected_charges += 1
                raise ByteBudgetExceeded(
                    f"Bytebudget reicht nicht fuer {tag!r}: "
                    f"{attempted}/{self.limit} Bytes"
                )
            self.body += body
            self.overhead += overhead
            self.requests += 1
            self.log.append((str(tag), body, overhead))

    def as_dict(self) -> dict[str, int]:
        result = dict(super().as_dict())
        result["rejected_charges"] = int(self.rejected_charges)
        return result


class LocalRangeReader:
    """Offline inclusive-range reader for a directory of source files."""

    range_overhead_reserve = 0

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
            raise RangeValidationError(f"Pfad verlaesst lokale Quelle: {filename!r}") from exc
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


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
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
        self.max_metadata_bytes = int(max_metadata_bytes)
        if self.max_metadata_bytes <= 0:
            raise ValueError("max_metadata_bytes muss positiv sein")
        self._lock = threading.Lock()
        self._reserved = 0
        self._key_locks: dict[str, threading.Lock] = {}
        self._stats = {
            "range_requests": 0,
            "range_bytes_requested": 0,
            "file_requests": 0,
            "failed_requests": 0,
            "optional_misses": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "cache_writes": 0,
            "cache_bytes_written": 0,
            "cache_bytes_reused": 0,
            "cache_integrity_checks": 0,
        }
        self._cache_bypass = threading.local()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.upstream, name)

    def _bump(self, field: str, amount: int = 1) -> None:
        with self._lock:
            self._stats[field] += int(amount)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def _key_lock(self, key: str) -> threading.Lock:
        with self._lock:
            return self._key_locks.setdefault(key, threading.Lock())

    @contextmanager
    def uncached(self) -> Any:
        """Temporarily bypass cache reads while still replacing fresh entries."""
        previous = bool(getattr(self._cache_bypass, "active", False))
        self._cache_bypass.active = True
        try:
            yield
        finally:
            self._cache_bypass.active = previous

    def _reserve(self, amount: int, tag: str) -> None:
        amount = int(amount)
        with self._lock:
            projected = self.budget.total + self._reserved + amount
            if projected > self.budget.limit:
                self.budget.rejected_charges += 1
                raise ByteBudgetExceeded(
                    f"Bytebudget reicht vor I/O nicht fuer {tag!r}: "
                    f"{projected}/{self.budget.limit} Bytes"
                )
            self._reserved += amount

    def _release(self, amount: int) -> None:
        with self._lock:
            self._reserved -= int(amount)
            if self._reserved < 0:
                self._reserved = 0

    def _identity(self, filename: str) -> dict[str, str]:
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
        info = self.file_info.get(filename, {})
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
        if not blob_path.exists() and not meta_path.exists():
            return None
        if not blob_path.is_file() or not meta_path.is_file():
            raise CacheIntegrityError(f"Partieller Cache-Eintrag: {key}")
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CacheIntegrityError(f"Cache-Metadaten unlesbar: {meta_path}") from exc
        if meta.get("schema") != self._CACHE_SCHEMA or meta.get("contract") != dict(contract):
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
        cached_identity = meta.get("source_identity") or {}
        if known_identity and cached_identity and known_identity != cached_identity:
            raise CacheIntegrityError(
                f"Quellidentitaet fuer Cache-Eintrag hat sich geaendert: {blob_path}"
            )
        self._bump("cache_hits")
        self._bump("cache_bytes_reused", len(body))
        self._bump("cache_integrity_checks")
        return body

    def _write_cache(
        self,
        kind: str,
        key: str,
        contract: Mapping[str, Any],
        body: bytes,
    ) -> None:
        paths = self._cache_paths(kind, key)
        if paths is None:
            return
        blob_path, meta_path = paths
        meta = {
            "schema": self._CACHE_SCHEMA,
            "contract": dict(contract),
            "source_identity": self._identity(str(contract["filename"])),
            "size": len(body),
            "sha256": _sha256(body),
        }
        _atomic_write(blob_path, body)
        _atomic_write(meta_path, _canonical_json(meta))
        self._bump("cache_writes")
        self._bump("cache_bytes_written", len(body))

    def get_range(self, filename: str, start: int, end: int) -> bytes:
        start, end = _validate_inclusive_range(start, end)
        expected = end - start + 1
        key, contract = self._cache_key("range", filename, start, end)
        with self._key_lock(key):
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
            self._reserve(reserve, f"range:{filename}:{start}-{end}")
            before_body = self.budget.body
            before_total = self.budget.total
            self._bump("range_requests")
            self._bump("range_bytes_requested", expected)
            try:
                result = self.upstream.get_range(filename, start, end)
                body = bytes(result)
                if len(body) != expected:
                    raise RangeValidationError(
                        f"Range {filename}[{start}:{end}] lieferte "
                        f"{len(body)} statt {expected} Bytes"
                    )
                if self.budget.total == before_total:
                    self.budget.charge(len(body), 0, f"range:{filename}:{start}")
                elif self.budget.body - before_body < len(body):
                    raise RangeValidationError(
                        f"Reader verbuchte fuer {filename} weniger als die gelieferte Bytezahl"
                    )
            except Exception:
                self._bump("failed_requests")
                raise
            finally:
                self._release(reserve)
            self._write_cache("range", key, contract, body)
            return body

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
        with self._key_lock(key):
            cached = self._load_cache("file", key, contract, None)
            if cached is not None:
                if len(cached) > ceiling:
                    raise RangeValidationError(
                        f"Gecachte Metadatei {filename!r} ueberschreitet max_bytes"
                    )
                return cached
            self._bump("cache_misses")
            self._bump("file_requests")
            before_body = self.budget.body
            before_total = self.budget.total
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
                        self._reserve(known_size, f"file:{filename}")
                    try:
                        body = bytes(self.upstream.fetch_file(filename))
                    finally:
                        if known_size is not None:
                            self._release(known_size)
                if len(body) > ceiling:
                    raise RangeValidationError(
                        f"Reader lieferte fuer {filename!r} {len(body)} Bytes; "
                        f"Grenze ist {ceiling}"
                    )
                if known_size is not None and len(body) != known_size:
                    raise RangeValidationError(
                        f"Metadatei {filename!r} aenderte ihre Groesse beim Lesen"
                    )
                if self.budget.total == before_total:
                    self.budget.charge(len(body), 0, f"file:{filename}")
                elif self.budget.body - before_body < len(body):
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

    def fetch_st_header(self, filename: str) -> tuple[dict[str, Any], int, dict[str, Any]]:
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
            raise InventoryValidationError(f"Safetensors-Header ist kein Objekt: {filename}")
        metadata = header.pop("__metadata__", None)
        info = self.file_info.setdefault(filename, {})
        info["header_len"] = header_length
        if isinstance(metadata, Mapping):
            info["st_metadata"] = {
                str(key): str(value)[:200] for key, value in list(metadata.items())[:8]
            }
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
        self._inventory_cache_hits = 0
        self._inventory_cache_writes = 0
        self._inventory_fingerprint: str | None = None
        self._max_metadata_bytes = int(max_metadata_bytes)
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
            verbose=verbose,
        )

    @property
    def reader(self) -> _ContractReader:
        if self._reader is None:
            upstream = self._upstream
            if upstream is None:
                upstream = HFRangeReader(self.repo_id, revision=self.revision)
            if hasattr(upstream, "repo"):
                upstream.repo = self.repo_id
            if hasattr(upstream, "rev"):
                upstream.rev = self.revision
            self._reader = _ContractReader(
                upstream,
                self.budget,
                self._cache_dir,
                max_metadata_bytes=self._max_metadata_bytes,
            )
        return self._reader

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
        return _sha256(_canonical_json(sorted(shards, key=lambda item: str(item["file"]))))

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

    def _load_inventory_path(self, path: Path, *, allow_legacy: bool) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CacheIntegrityError(f"Inventar-Cache unlesbar: {path}") from exc
        if isinstance(raw, Mapping) and raw.get("schema") == self._INVENTORY_SCHEMA:
            if raw.get("repo_id") != self.repo_id or raw.get("revision") != self.revision:
                raise CacheIntegrityError(f"Inventar-Cache gehoert zu anderer Quelle: {path}")
            document = raw.get("inventory")
            if not isinstance(document, Mapping):
                raise CacheIntegrityError(f"Inventar-Payload fehlt: {path}")
            document = dict(document)
            if raw.get("inventory_sha256") != _sha256(_canonical_json(document)):
                raise CacheIntegrityError(f"Inventar-SHA256 stimmt nicht: {path}")
            fingerprint = self._source_fingerprint(document)
            if raw.get("source_fingerprint") != fingerprint:
                raise CacheIntegrityError(f"Inventar-Quellfingerprint stimmt nicht: {path}")
        elif allow_legacy and isinstance(raw, Mapping):
            document = dict(raw)
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
            info = self.reader.file_info.setdefault(shard["file"], {})
            for key in ("size", "etag", "cas_url_hash", "header_len"):
                if shard.get(key) is not None:
                    info[key] = shard[key]

    def _validate_inventory(self, document: Mapping[str, Any]) -> None:
        if document.get("repo") != self.repo_id or document.get("revision") != self.revision:
            raise InventoryValidationError(
                "Inventarquelle stimmt nicht mit repo_id/revision des Streamers ueberein"
            )
        tensors = document.get("tensors")
        if not isinstance(tensors, list):
            raise InventoryValidationError("Inventar enthaelt keine Tensorliste")
        seen: set[str] = set()
        for index, entry in enumerate(tensors):
            if not isinstance(entry, Mapping):
                raise InventoryValidationError(f"Tensor-Eintrag {index} ist kein Objekt")
            name = entry.get("name")
            if not isinstance(name, str) or not name or name in seen:
                raise InventoryValidationError(f"Ungueltiger/doppelter Tensorname: {name!r}")
            seen.add(name)
            shape = entry.get("shape")
            if not isinstance(shape, list) or any(
                isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
            ):
                raise InventoryValidationError(f"Ungueltige Shape bei {name}")
            offsets = entry.get("offset_in_shard")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
                or offsets[0] < 0
                or offsets[1] < offsets[0]
            ):
                raise InventoryValidationError(f"Ungueltige Offsets bei {name}")
            if not isinstance(entry.get("shard"), str) or not entry["shard"]:
                raise InventoryValidationError(f"Shard fehlt bei {name}")
            data_start = entry.get("data_start")
            if isinstance(data_start, bool) or not isinstance(data_start, int) or data_start < 8:
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
        if self._inventory is not None and not refresh:
            return self._inventory
        if not refresh and self._cache_dir is not None:
            cached = self._load_inventory_path(self._cache_path(), allow_legacy=False)
            if cached is None:
                cached = self._load_inventory_path(self._legacy_cache_path(), allow_legacy=True)
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
                raise TensorSourceError(f"Inventar nach 3 Versuchen fehlgeschlagen: {last}") from last
            raise TensorSourceError("Inventar konnte nicht erstellt werden")
        self._inventory = candidate
        self._inventory_fingerprint = self._source_fingerprint(self._inventory)
        self._hydrate_reader_identity(self._inventory)
        self._write_inventory_cache(self._inventory)
        return self._inventory

    def tensors(self) -> list[dict[str, Any]]:
        """Return a shallow copy of all tensor metadata entries."""
        return [dict(entry) for entry in self.inventory().get("tensors", [])]

    def find(self, name: str) -> dict[str, Any]:
        if not isinstance(name, str) or not name:
            raise ValueError("Tensorname muss ein nichtleerer String sein")
        for entry in self.inventory().get("tensors", []):
            if entry["name"] == name:
                return dict(entry)
        raise KeyError(f"Tensor {name!r} nicht im Inventar von {self.repo_id}")

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
        import numpy as np

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
        if itemsize is None or (dtype != "BF16" and dtype not in self._NUMPY_DTYPES):
            raise RangeValidationError(f"Nicht unterstuetztes safetensors-dtype {dtype!r}")
        if n_rows == 0:
            result_dtype = np.float32 if dtype == "BF16" else np.dtype(self._NUMPY_DTYPES[dtype])
            return np.empty((0, n_cols), dtype=result_dtype)

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
        if dtype == "BF16":
            decoded = bf16_rows_to_f32(
                np.frombuffer(raw, dtype="<u2"), (n_rows, n_cols)
            )
            return np.ascontiguousarray(decoded, dtype=np.float32)
        decoded = np.frombuffer(raw, dtype=np.dtype(self._NUMPY_DTYPES[dtype]))
        return decoded.reshape(n_rows, n_cols).copy()

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
            raise TensorSourceError("rows_torch() braucht das optionale Paket torch") from exc
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

    def bytes_moved(self) -> int:
        """Network/local source body bytes; verified cache hits count as zero."""
        return int(self.budget.body)

    def metrics(self) -> dict[str, Any]:
        reader_stats = self.reader.stats()
        pinned_revision = bool(re.fullmatch(r"[0-9a-fA-F]{40,64}", self.revision))
        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "revision_is_pinned": pinned_revision and not self.repo_id.startswith("local:"),
            "revision_is_mutable": self.repo_id.startswith("local:") or not pinned_revision,
            "budget": self.budget.as_dict(),
            "network_or_source_body_bytes": int(self.budget.body),
            "inventory_cache_hits": int(self._inventory_cache_hits),
            "inventory_cache_writes": int(self._inventory_cache_writes),
            "inventory_source_fingerprint": self._inventory_fingerprint,
            **reader_stats,
        }


def available() -> bool:
    return True
