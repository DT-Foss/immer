"""Causal-bound mmap Q4/Q8 execution plane for local Qwen3.8.

The original BF16 bundle remains untouched.  This module builds and mounts a
derived row-addressable view whose manifest is bound to the original model,
inventory, bundle manifest, and causal graph revision.  Matrix payloads are
never expanded as full floating tensors during inference: the native kernel
reads packed mmap pages directly.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path, PurePosixPath
import platform
import secrets
import shutil
import stat
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Callable, Mapping

import numpy as np


Q4_BANK_SCHEMA = "immer.qwen3.8-causal-q4-bank/v1"
Q4_BUILD_STATE_SCHEMA = "immer.qwen3.8-causal-q4-build-state/v1"
Q4_VERIFY_CACHE_SCHEMA = "immer.qwen3.8-causal-q4-verify-cache/v1"
Q4_VERIFY_CACHE_NAME = ".verify-cache-v1.json"
Q4_NATIVE_ABI = 2
Q4_0 = "q4_0"
Q8_0 = "q8_0"
Q4_BLOCK_SIZE = 32
Q4_BLOCK_BYTES = 18
Q8_BLOCK_BYTES = 34
Q4_NATIVE_SOURCE = "q4_native.c"
Q4_HEAD_TENSORS = frozenset(
    {
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    }
)
Q4_BASE_POLICY = "q4_0-text-matrices+q8_0-embedding-head/v1"
Q4_BALANCED_POLICY = (
    "q4_0-gate-up-full-attn+q8_0-linear-attn-down-embedding-head/v1"
)
Q4_RECURRENT_POLICY = (
    "q4_0-mlp-full-attn+q8_0-linear-attn-embedding-head/v1"
)
Q4_FORMAT_POLICIES = frozenset(
    (Q4_BASE_POLICY, Q4_BALANCED_POLICY, Q4_RECURRENT_POLICY)
)
_FORMAT_CODES = {Q4_0: 4, Q8_0: 8}
_FORMAT_BLOCK_BYTES = {Q4_0: Q4_BLOCK_BYTES, Q8_0: Q8_BLOCK_BYTES}


class Q4BankError(RuntimeError):
    """The derived Q4 execution plane cannot be built or executed."""


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
        raise Q4BankError("Q4 metadata is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, value: object) -> None:
    data = _canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
            0o600,
        )
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short Q4 metadata write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _strict_document(path: Path, *, schema: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Q4BankError(f"cannot read Q4 metadata: {path}") from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"body", "schema", "sha256"}
        or document.get("schema") != schema
        or not isinstance(document.get("body"), dict)
        or document.get("sha256") != _digest(document["body"])
        or _canonical(document) != raw
    ):
        raise Q4BankError(f"Q4 metadata contract is invalid: {path}")
    return document


def _regular_file(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise Q4BankError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise Q4BankError(f"{label} must be a non-symlink regular file")
    return metadata


def _cpu_flags() -> frozenset[str]:
    if not sys.platform.startswith("linux"):
        return frozenset()
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="ascii", errors="ignore")
    except OSError:
        return frozenset()
    for line in text.splitlines():
        if line.lower().startswith(("flags", "features")) and ":" in line:
            return frozenset(line.split(":", 1)[1].split())
    return frozenset()


def _compiler_command(source: Path, output: Path) -> tuple[str, ...]:
    compiler_name = os.environ.get("CC", "cc")
    compiler = shutil.which(compiler_name)
    if compiler is None:
        raise Q4BankError(f"native Q4 compiler is unavailable: {compiler_name}")
    flags = [compiler, "-std=c11", "-O3", "-DNDEBUG"]
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        flags.extend(("-dynamiclib", "-fPIC"))
    else:
        flags.extend(("-shared", "-fPIC", "-fopenmp"))
    cpu = _cpu_flags()
    if machine in {"x86_64", "amd64"} and {"avx2", "fma"}.issubset(cpu):
        flags.extend(("-mavx2", "-mfma", "-mssse3"))
    flags.extend((str(source), "-o", str(output), "-lm"))
    return tuple(flags)


@lru_cache(maxsize=1)
def _native_library() -> "Q4NativeKernel":
    return Q4NativeKernel.load()


class Q4NativeKernel:
    """Small ctypes bridge to the package-shipped native CPU kernel."""

    def __init__(self, library: ctypes.CDLL, *, path: Path, build_seconds: float) -> None:
        self.library = library
        self.path = path
        self.build_seconds = float(build_seconds)
        self._configure()
        if self.library.immer_q4_abi() != Q4_NATIVE_ABI:
            raise Q4BankError("native Q4 ABI differs from the Python runtime")
        self.avx2 = bool(self.library.immer_q4_has_avx2())

    @classmethod
    def load(cls) -> "Q4NativeKernel":
        source = Path(__file__).with_name(Q4_NATIVE_SOURCE)
        _regular_file(source, "native Q4 source")
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        seed = {
            "abi": Q4_NATIVE_ABI,
            "machine": platform.machine(),
            "platform": sys.platform,
            "python": sys.version_info[:3],
            "source_sha256": _file_sha256(source),
            "flags": _compiler_command(source, Path("OUTPUT"))[:-4],
        }
        identity = _digest(seed)
        cache_root = Path(
            os.environ.get(
                "IMMER_NATIVE_CACHE",
                str(Path.home() / ".cache" / "immer" / "native"),
            )
        ).expanduser()
        target_root = cache_root / f"q4-{identity}"
        library_path = target_root / f"libimmer_q4{suffix}"
        built = 0.0
        if not library_path.exists():
            target_root.mkdir(parents=True, exist_ok=True)
            temporary = target_root / f".{library_path.name}.{secrets.token_hex(8)}.tmp"
            started = time.perf_counter()
            command = _compiler_command(source, temporary)
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if completed.returncode != 0:
                    reason = completed.stderr.strip() or completed.stdout.strip()
                    raise Q4BankError(f"native Q4 compilation failed: {reason}")
                os.replace(temporary, library_path)
            finally:
                temporary.unlink(missing_ok=True)
            built = time.perf_counter() - started
        _regular_file(library_path, "native Q4 library")
        try:
            library = ctypes.CDLL(str(library_path))
        except OSError as exc:
            raise Q4BankError("cannot load native Q4 library") from exc
        return cls(library, path=library_path, build_seconds=built)

    def _configure(self) -> None:
        void = ctypes.c_void_p
        i64 = ctypes.c_int64
        integer = ctypes.c_int
        self.library.immer_q4_abi.argtypes = ()
        self.library.immer_q4_abi.restype = ctypes.c_uint32
        self.library.immer_q4_has_avx2.argtypes = ()
        self.library.immer_q4_has_avx2.restype = integer
        self.library.immer_q4_row_bytes.argtypes = (integer, i64)
        self.library.immer_q4_row_bytes.restype = i64
        self.library.immer_q4_quantize_f32.argtypes = (
            void,
            i64,
            i64,
            integer,
            void,
            integer,
        )
        self.library.immer_q4_quantize_f32.restype = integer
        self.library.immer_q4_dequantize_rows_f32.argtypes = (
            void,
            integer,
            i64,
            i64,
            void,
            i64,
            void,
            integer,
        )
        self.library.immer_q4_dequantize_rows_f32.restype = integer
        self.library.immer_q4_linear_f32.argtypes = (
            void,
            i64,
            i64,
            void,
            integer,
            i64,
            void,
            integer,
        )
        self.library.immer_q4_linear_f32.restype = integer
        self.library.immer_q4_linear_group_f32.argtypes = (
            void,
            i64,
            i64,
            void,
            void,
            void,
            void,
            integer,
            integer,
        )
        self.library.immer_q4_linear_group_f32.restype = integer

    @staticmethod
    def _pointer(value: Any) -> ctypes.c_void_p:
        data_ptr = getattr(value, "data_ptr", None)
        if callable(data_ptr):
            return ctypes.c_void_p(int(data_ptr()))
        interface = getattr(value, "ctypes", None)
        if interface is None:
            raise TypeError("native Q4 value has no stable data pointer")
        return ctypes.c_void_p(int(interface.data))

    def row_bytes(self, fmt: str, cols: int) -> int:
        if fmt not in _FORMAT_CODES:
            raise ValueError(f"unknown Q4 format: {fmt}")
        value = int(self.library.immer_q4_row_bytes(_FORMAT_CODES[fmt], cols))
        if value <= 0:
            raise Q4BankError("Q4 tensor width must be divisible by 32")
        return value

    def quantize(self, values: Any, output: np.ndarray, *, fmt: str, threads: int) -> None:
        if values.ndim != 2 or values.dtype != values.new_empty(()).float().dtype:
            raise TypeError("native Q4 quantization requires a 2D float32 tensor")
        expected = values.shape[0] * self.row_bytes(fmt, values.shape[1])
        if output.dtype != np.uint8 or not output.flags.c_contiguous or output.size != expected:
            raise ValueError("native Q4 output buffer has the wrong size")
        code = self.library.immer_q4_quantize_f32(
            self._pointer(values),
            values.shape[0],
            values.shape[1],
            _FORMAT_CODES[fmt],
            self._pointer(output),
            threads,
        )
        if code:
            raise Q4BankError(f"native Q4 quantization failed with code {code}")


@dataclass(frozen=True, slots=True)
class Q4TensorEntry:
    name: str
    shape: tuple[int, int]
    format: str
    file: str
    row_bytes: int
    payload_bytes: int
    payload_sha256: str
    source_dtype: str = "BF16"

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "Q4TensorEntry":
        try:
            shape = tuple(value["shape"])
            entry = cls(
                name=value["name"],
                shape=shape,  # type: ignore[arg-type]
                format=value["format"],
                file=value["file"],
                row_bytes=value["row_bytes"],
                payload_bytes=value["payload_bytes"],
                payload_sha256=value["payload_sha256"],
                source_dtype=value["source_dtype"],
            )
        except (KeyError, TypeError) as exc:
            raise Q4BankError("Q4 tensor entry is incomplete") from exc
        if (
            not isinstance(entry.name, str)
            or not entry.name
            or len(entry.shape) != 2
            or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in entry.shape)
            or entry.shape[1] % Q4_BLOCK_SIZE
            or entry.format not in _FORMAT_CODES
            or not isinstance(entry.file, str)
            or PurePosixPath(entry.file).name != entry.file
            or any(
                isinstance(x, bool) or not isinstance(x, int) or x <= 0
                for x in (entry.row_bytes, entry.payload_bytes)
            )
            or entry.row_bytes
            != entry.shape[1] // Q4_BLOCK_SIZE * _FORMAT_BLOCK_BYTES[entry.format]
            or entry.payload_bytes != entry.shape[0] * entry.row_bytes
            or not isinstance(entry.payload_sha256, str)
            or len(entry.payload_sha256) != 64
            or entry.source_dtype != "BF16"
        ):
            raise Q4BankError("Q4 tensor entry is invalid")
        return entry


def _payload_verification_row(path: Path, entry: Q4TensorEntry) -> dict[str, Any]:
    metadata = _regular_file(path, "Q4 payload")
    if metadata.st_size != entry.payload_bytes:
        raise Q4BankError("Q4 tensor payload size differs from its manifest")
    return {
        "ctime_ns": metadata.st_ctime_ns,
        "device": metadata.st_dev,
        "file": entry.file,
        "inode": metadata.st_ino,
        "mtime_ns": metadata.st_mtime_ns,
        "sha256": entry.payload_sha256,
        "size": metadata.st_size,
    }


def _write_payload_verification_cache(
    root: Path,
    document: Mapping[str, Any],
    entries: Mapping[str, Q4TensorEntry],
) -> None:
    rows = [
        _payload_verification_row(root / "weights" / entry.file, entry)
        for entry in sorted(entries.values(), key=lambda value: value.file)
    ]
    body = {"manifest_sha256": document["sha256"], "files": rows}
    cache = {
        "schema": Q4_VERIFY_CACHE_SCHEMA,
        "body": body,
        "sha256": _digest(body),
    }
    try:
        _atomic_write(root / Q4_VERIFY_CACHE_NAME, cache)
    except OSError:
        # Read-only banks remain usable; the next mount re-hashes changed or
        # uncached files instead of trusting incomplete metadata.
        pass


def _verified_payload_entries(
    root: Path,
    document: Mapping[str, Any],
) -> dict[str, Q4TensorEntry]:
    raw_entries = document["body"].get("tensors")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise Q4BankError("Q4 bank has no tensor inventory")
    cached: dict[str, Mapping[str, Any]] = {}
    cache_path = root / Q4_VERIFY_CACHE_NAME
    if cache_path.exists() or cache_path.is_symlink():
        try:
            cache = _strict_document(cache_path, schema=Q4_VERIFY_CACHE_SCHEMA)
            if cache["body"].get("manifest_sha256") == document["sha256"]:
                rows = cache["body"].get("files")
                if isinstance(rows, list):
                    cached = {
                        row["file"]: row
                        for row in rows
                        if isinstance(row, Mapping) and isinstance(row.get("file"), str)
                    }
        except Q4BankError:
            cached = {}
    entries: dict[str, Q4TensorEntry] = {}
    files: set[str] = set()
    cache_changed = False
    for raw in raw_entries:
        if not isinstance(raw, Mapping):
            raise Q4BankError("Q4 tensor inventory contains a non-object")
        entry = Q4TensorEntry.from_document(raw)
        if entry.name in entries or entry.file in files:
            raise Q4BankError("Q4 tensor inventory contains a duplicate")
        path = root / "weights" / entry.file
        row = _payload_verification_row(path, entry)
        if cached.get(entry.file) != row:
            if _file_sha256(path) != entry.payload_sha256:
                raise Q4BankError(f"Q4 payload digest differs: {entry.file}")
            cache_changed = True
        entries[entry.name] = entry
        files.add(entry.file)
    if len(cached) != len(entries):
        cache_changed = True
    if cache_changed:
        _write_payload_verification_cache(root, document, entries)
    return entries


class _MappedTensor:
    def __init__(self, path: Path, entry: Q4TensorEntry) -> None:
        flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
        flags |= int(getattr(os, "O_NOFOLLOW", 0))
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise Q4BankError(f"cannot open Q4 payload: {path}") from exc
        try:
            metadata = os.fstat(descriptor)
            linked = _regular_file(path, "Q4 payload")
            if (
                metadata.st_size != entry.payload_bytes
                or (metadata.st_dev, metadata.st_ino) != (linked.st_dev, linked.st_ino)
            ):
                raise Q4BankError(f"Q4 payload identity differs: {path}")
            self.mapping = mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ)
        finally:
            os.close(descriptor)
        self.bytes = np.frombuffer(self.mapping, dtype=np.uint8)

    def close(self) -> None:
        self.bytes = np.empty(0, dtype=np.uint8)
        self.mapping.close()


@dataclass(slots=True)
class Q4BankMetrics:
    mapped_tensors: int = 0
    mapped_payload_bytes: int = 0
    linear_calls: int = 0
    linear_group_calls: int = 0
    input_quantizations: int = 0
    linear_input_rows: int = 0
    embedding_rows: int = 0
    candidate_rows: int = 0
    head_calls: int = 0
    logical_weight_bytes: int = 0
    output_bytes: int = 0


class Q4Bank:
    """Lazy mmap view over one source-bound Q4/Q8 tensor bank."""

    def __init__(
        self,
        root: Path,
        *,
        manifest: Mapping[str, Any],
        entries: Mapping[str, Q4TensorEntry],
        threads: int,
    ) -> None:
        self.root = root
        self.manifest = dict(manifest)
        self.entries = dict(entries)
        self.threads = threads
        self.native = _native_library()
        self._mapped: dict[str, _MappedTensor] = {}
        self._stats = Q4BankMetrics()
        self._lock = threading.RLock()
        self._closed = False

    @classmethod
    def load(
        cls,
        root: str | Path,
        *,
        bundle_receipt: Mapping[str, Any],
        repo_id: str,
        revision: str,
        inventory_fingerprint: str,
        threads: int | None = None,
    ) -> "Q4Bank":
        source = Path(root).expanduser().resolve()
        document = _strict_document(source / "manifest.json", schema=Q4_BANK_SCHEMA)
        body = document["body"]
        if (
            body.get("native_abi") != Q4_NATIVE_ABI
            or body.get("format_policy") not in Q4_FORMAT_POLICIES
        ):
            raise Q4BankError("Q4 bank codec identity differs from this runtime")
        expected = {
            "repo_id": repo_id,
            "revision": revision,
            "inventory_fingerprint": inventory_fingerprint,
            "bundle_manifest_sha256": bundle_receipt.get("manifest_sha256"),
            "layout_fingerprint": bundle_receipt.get("layout_fingerprint"),
            "graph_revision": bundle_receipt.get("graph_revision"),
        }
        if body.get("source") != expected:
            raise Q4BankError("Q4 bank is bound to a different causal bundle")
        entries = _verified_payload_entries(source, document)
        if body.get("tensor_count") != len(entries) or body.get("payload_bytes") != sum(
            entry.payload_bytes for entry in entries.values()
        ):
            raise Q4BankError("Q4 aggregate inventory differs from its entries")
        if threads is None:
            threads = min(16, os.cpu_count() or 1)
        if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
            raise ValueError("Q4 thread count must be a positive integer")
        return cls(source, manifest=document, entries=entries, threads=threads)

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "schema": Q4_BANK_SCHEMA,
            "manifest_sha256": self.manifest["sha256"],
            "native_abi": Q4_NATIVE_ABI,
            "native_avx2": self.native.avx2,
        }

    def has(self, name: str) -> bool:
        return name in self.entries

    def _mapping(self, name: str) -> tuple[Q4TensorEntry, _MappedTensor]:
        if self._closed:
            raise Q4BankError("Q4 bank is closed")
        try:
            entry = self.entries[name]
        except KeyError:
            raise KeyError(name) from None
        mapped = self._mapped.get(name)
        if mapped is None:
            mapped = _MappedTensor(self.root / "weights" / entry.file, entry)
            self._mapped[name] = mapped
            self._stats.mapped_tensors += 1
            self._stats.mapped_payload_bytes += entry.payload_bytes
        return entry, mapped

    def linear(self, values: Any, name: str, *, output_dtype: Any | None = None) -> Any:
        import torch

        with self._lock:
            entry, mapped = self._mapping(name)
            if not isinstance(values, torch.Tensor):
                values = torch.as_tensor(values)
            if values.ndim < 1 or values.shape[-1] != entry.shape[1]:
                raise Q4BankError(f"Q4 input width disagrees with {name!r}")
            if values.device.type != "cpu":
                raise Q4BankError("Q4 execution is currently CPU-only")
            leading = tuple(values.shape[:-1])
            input_rows = values.numel() // values.shape[-1]
            compute = values.detach().to(dtype=torch.float32).reshape(
                input_rows, entry.shape[1]
            ).contiguous()
            output = torch.empty((input_rows, entry.shape[0]), dtype=torch.float32)
            code = self.native.library.immer_q4_linear_f32(
                self.native._pointer(compute),
                input_rows,
                entry.shape[1],
                self.native._pointer(mapped.bytes),
                _FORMAT_CODES[entry.format],
                entry.shape[0],
                self.native._pointer(output),
                self.threads,
            )
            if code:
                raise Q4BankError(f"native Q4 linear failed with code {code}")
            dtype = values.dtype if output_dtype is None else output_dtype
            result = output.to(dtype=dtype).reshape(*leading, entry.shape[0])
            self._stats.linear_calls += 1
            self._stats.input_quantizations += 1
            self._stats.linear_input_rows += input_rows
            self._stats.logical_weight_bytes += entry.payload_bytes
            self._stats.output_bytes += result.numel() * result.element_size()
            return result

    def linear_group(
        self,
        values: Any,
        names: tuple[str, ...],
        *,
        output_dtype: Any | None = None,
    ) -> tuple[Any, ...]:
        import torch

        with self._lock:
            if len(names) < 2 or len(set(names)) != len(names):
                raise ValueError("Q4 linear group requires distinct matrix names")
            resolved = [self._mapping(name) for name in names]
            entries = [row[0] for row in resolved]
            mapped = [row[1] for row in resolved]
            input_columns = entries[0].shape[1]
            if any(entry.shape[1] != input_columns for entry in entries):
                raise Q4BankError("Q4 linear group input widths differ")
            if not isinstance(values, torch.Tensor):
                values = torch.as_tensor(values)
            if values.ndim < 1 or values.shape[-1] != input_columns:
                raise Q4BankError("Q4 grouped input width disagrees with its weights")
            if values.device.type != "cpu":
                raise Q4BankError("Q4 execution is currently CPU-only")
            leading = tuple(values.shape[:-1])
            input_rows = values.numel() // input_columns
            compute = values.detach().to(dtype=torch.float32).reshape(
                input_rows, input_columns
            ).contiguous()
            outputs = [
                torch.empty((input_rows, entry.shape[0]), dtype=torch.float32)
                for entry in entries
            ]
            count = len(entries)
            weight_pointers = (ctypes.c_void_p * count)(
                *(self.native._pointer(row.bytes).value for row in mapped)
            )
            formats = (ctypes.c_int * count)(
                *(_FORMAT_CODES[entry.format] for entry in entries)
            )
            output_rows = (ctypes.c_int64 * count)(
                *(entry.shape[0] for entry in entries)
            )
            output_pointers = (ctypes.c_void_p * count)(
                *(self.native._pointer(output).value for output in outputs)
            )
            code = self.native.library.immer_q4_linear_group_f32(
                self.native._pointer(compute),
                input_rows,
                input_columns,
                ctypes.cast(weight_pointers, ctypes.c_void_p),
                ctypes.cast(formats, ctypes.c_void_p),
                ctypes.cast(output_rows, ctypes.c_void_p),
                ctypes.cast(output_pointers, ctypes.c_void_p),
                count,
                self.threads,
            )
            if code:
                raise Q4BankError(f"native Q4 linear group failed with code {code}")
            dtype = values.dtype if output_dtype is None else output_dtype
            results = tuple(
                output.to(dtype=dtype).reshape(*leading, entry.shape[0])
                for output, entry in zip(outputs, entries, strict=True)
            )
            self._stats.linear_calls += count
            self._stats.linear_group_calls += 1
            self._stats.input_quantizations += 1
            self._stats.linear_input_rows += input_rows * count
            self._stats.logical_weight_bytes += sum(
                entry.payload_bytes for entry in entries
            )
            self._stats.output_bytes += sum(
                result.numel() * result.element_size() for result in results
            )
            return results

    def rows(self, name: str, row_ids: tuple[int, ...], *, dtype: Any) -> Any:
        import torch

        with self._lock:
            entry, mapped = self._mapping(name)
            unique_ids = tuple(dict.fromkeys(row_ids))
            ids = torch.tensor(unique_ids, dtype=torch.int64)
            output = torch.empty((len(unique_ids), entry.shape[1]), dtype=torch.float32)
            if unique_ids:
                code = self.native.library.immer_q4_dequantize_rows_f32(
                    self.native._pointer(mapped.bytes),
                    _FORMAT_CODES[entry.format],
                    entry.shape[0],
                    entry.shape[1],
                    self.native._pointer(ids),
                    len(unique_ids),
                    self.native._pointer(output),
                    self.threads,
                )
                if code:
                    raise Q4BankError(f"native Q4 row decode failed with code {code}")
            unique = output.to(dtype=dtype)
            if unique_ids == row_ids:
                result = unique
            else:
                by_id = {value: index for index, value in enumerate(unique_ids)}
                restore = torch.tensor(
                    [by_id[value] for value in row_ids],
                    dtype=torch.long,
                )
                result = unique.index_select(0, restore)
            self._stats.logical_weight_bytes += len(unique_ids) * entry.row_bytes
            self._stats.output_bytes += result.numel() * result.element_size()
            return result

    def record_embedding(self, rows: int) -> None:
        with self._lock:
            self._stats.embedding_rows += rows

    def record_candidates(self, rows: int) -> None:
        with self._lock:
            self._stats.candidate_rows += rows

    def record_head(self) -> None:
        with self._lock:
            self._stats.head_calls += 1

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                **asdict(self._stats),
                **self.identity,
                "tensor_count": len(self.entries),
                "payload_bytes": sum(x.payload_bytes for x in self.entries.values()),
                "threads": self.threads,
                "native_library": str(self.native.path),
                "native_build_seconds": self.native.build_seconds,
                "closed": self._closed,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            for mapped in self._mapped.values():
                mapped.close()
            self._mapped.clear()
            self._closed = True


class Q4BankBuilder:
    """Tensor-resumable Q4/Q8 conversion from one verified BF16 pager."""

    def __init__(
        self,
        root: str | Path,
        *,
        pager: Any,
        bundle_receipt: Mapping[str, Any],
        row_chunk: int = 128,
        threads: int | None = None,
        format_policy: str = Q4_BASE_POLICY,
        reuse_root: str | Path | None = None,
    ) -> None:
        if isinstance(row_chunk, bool) or not isinstance(row_chunk, int) or row_chunk <= 0:
            raise ValueError("Q4 row chunk must be a positive integer")
        if threads is None:
            threads = min(16, os.cpu_count() or 1)
        if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
            raise ValueError("Q4 thread count must be a positive integer")
        if format_policy not in Q4_FORMAT_POLICIES:
            raise ValueError("unknown Q4 format policy")
        self.root = Path(root).expanduser().resolve()
        self.pager = pager
        self.source = pager.source
        self.bundle_receipt = dict(bundle_receipt)
        self.row_chunk = row_chunk
        self.threads = threads
        self.format_policy = format_policy
        self.reuse_root = (
            None if reuse_root is None else Path(reuse_root).expanduser().resolve()
        )
        self.native = _native_library()

    def _format_for_name(self, name: str) -> str:
        if name in Q4_HEAD_TENSORS:
            return Q8_0
        if self.format_policy in {Q4_BALANCED_POLICY, Q4_RECURRENT_POLICY} and (
            ".linear_attn." in name
            or (
                self.format_policy == Q4_BALANCED_POLICY
                and ".mlp.down_proj.weight" in name
            )
        ):
            return Q8_0
        return Q4_0

    def _source_binding(self) -> dict[str, Any]:
        metrics = self.source.metrics()
        fingerprint = metrics.get("inventory_source_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise Q4BankError("Q4 build source lacks an inventory fingerprint")
        return {
            "repo_id": self.source.repo_id,
            "revision": self.source.revision,
            "inventory_fingerprint": fingerprint,
            "bundle_manifest_sha256": self.bundle_receipt.get("manifest_sha256"),
            "layout_fingerprint": self.bundle_receipt.get("layout_fingerprint"),
            "graph_revision": self.bundle_receipt.get("graph_revision"),
        }

    def _eligible(self) -> tuple[tuple[str, tuple[int, int], str], ...]:
        rows: list[tuple[str, tuple[int, int], str]] = []
        for raw in self.source.inventory().get("tensors", ()):
            if not isinstance(raw, Mapping):
                continue
            name = raw.get("name")
            shape = raw.get("shape")
            dtype = str(raw.get("dtype", "")).upper()
            if (
                not isinstance(name, str)
                or not (name.startswith("model.language_model.") or name == "lm_head.weight")
                or not isinstance(shape, (list, tuple))
                or len(shape) != 2
                or dtype != "BF16"
                or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in shape)
                or shape[1] % Q4_BLOCK_SIZE
            ):
                continue
            fmt = self._format_for_name(name)
            rows.append((name, (shape[0], shape[1]), fmt))
        rows.sort(key=lambda row: row[0])
        if not rows:
            raise Q4BankError("verified source exposes no eligible text matrices")
        return tuple(rows)

    def _initial_state(self, source: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "source": dict(source),
            "native_abi": Q4_NATIVE_ABI,
            "row_chunk": self.row_chunk,
            "format_policy": self.format_policy,
            "completed": [],
        }

    def plan(self) -> dict[str, Any]:
        tensors = []
        reusable_payload_bytes = 0
        for name, shape, fmt in self._eligible():
            row_bytes = self.native.row_bytes(fmt, shape[1])
            payload_bytes = shape[0] * row_bytes
            reused = self._reusable_entry(name=name, shape=shape, fmt=fmt)
            if reused is not None:
                reusable_payload_bytes += payload_bytes
            tensors.append(
                {
                    "name": name,
                    "shape": list(shape),
                    "format": fmt,
                    "source_bf16_bytes": shape[0] * shape[1] * 2,
                    "payload_bytes": payload_bytes,
                    "reused": reused is not None,
                }
            )
        return {
            "schema": "immer.qwen3.8-causal-q4-build-plan/v1",
            "source": self._source_binding(),
            "tensor_count": len(tensors),
            "source_bf16_bytes": sum(row["source_bf16_bytes"] for row in tensors),
            "payload_bytes": sum(row["payload_bytes"] for row in tensors),
            "reused_payload_bytes": reusable_payload_bytes,
            "new_payload_bytes": (
                sum(row["payload_bytes"] for row in tensors)
                - reusable_payload_bytes
            ),
            "tensors": tensors,
        }

    def _load_state(self, source: Mapping[str, Any]) -> dict[str, Any]:
        path = self.root / ".build-state.json"
        if not path.exists():
            return self._initial_state(source)
        body = _strict_document(path, schema=Q4_BUILD_STATE_SCHEMA)["body"]
        if (
            body.get("source") != dict(source)
            or body.get("native_abi") != Q4_NATIVE_ABI
            or body.get("row_chunk") != self.row_chunk
            or body.get("format_policy") != self.format_policy
            or not isinstance(body.get("completed"), list)
        ):
            raise Q4BankError("existing Q4 build state belongs to another build")
        return body

    def _save_state(self, body: Mapping[str, Any]) -> None:
        document = {
            "schema": Q4_BUILD_STATE_SCHEMA,
            "body": dict(body),
            "sha256": _digest(body),
        }
        _atomic_write(self.root / ".build-state.json", document)

    @lru_cache(maxsize=1)
    def _reuse_entries(self) -> dict[str, Q4TensorEntry]:
        if self.reuse_root is None:
            return {}
        document = _strict_document(
            self.reuse_root / "manifest.json",
            schema=Q4_BANK_SCHEMA,
        )
        if document["body"].get("source") != self._source_binding():
            raise Q4BankError("Q4 reuse bank belongs to another causal source")
        if document["body"].get("native_abi") != Q4_NATIVE_ABI:
            raise Q4BankError("Q4 reuse bank has another codec ABI")
        return _verified_payload_entries(self.reuse_root, document)

    def _reusable_entry(
        self,
        *,
        name: str,
        shape: tuple[int, int],
        fmt: str,
    ) -> Q4TensorEntry | None:
        entry = self._reuse_entries().get(name)
        if entry is None or entry.shape != shape or entry.format != fmt:
            return None
        return entry

    def _build_tensor(
        self,
        *,
        name: str,
        shape: tuple[int, int],
        fmt: str,
        progress: Callable[[Mapping[str, Any]], None] | None,
    ) -> Q4TensorEntry:
        rows, cols = shape
        row_bytes = self.native.row_bytes(fmt, cols)
        payload_bytes = rows * row_bytes
        file = f"{hashlib.sha256(name.encode('utf-8')).hexdigest()[:24]}.{fmt}.bin"
        final = self.root / "weights" / file
        final.parent.mkdir(parents=True, exist_ok=True)
        reused = self._reusable_entry(name=name, shape=shape, fmt=fmt)
        if reused is not None:
            assert self.reuse_root is not None
            source = self.reuse_root / "weights" / reused.file
            temporary_link = final.parent / f".{file}.{secrets.token_hex(8)}.link"
            try:
                os.link(source, temporary_link)
                os.replace(temporary_link, final)
            except OSError as exc:
                raise Q4BankError(
                    "Q4 reuse requires source and destination on one filesystem"
                ) from exc
            finally:
                temporary_link.unlink(missing_ok=True)
            return Q4TensorEntry(
                name=name,
                shape=shape,
                format=fmt,
                file=file,
                row_bytes=row_bytes,
                payload_bytes=payload_bytes,
                payload_sha256=reused.payload_sha256,
            )
        temporary = final.parent / f".{file}.{secrets.token_hex(8)}.part"
        digest = hashlib.sha256()
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | int(getattr(os, "O_CLOEXEC", 0))
                | int(getattr(os, "O_NOFOLLOW", 0)),
                0o600,
            )
            for start in range(0, rows, self.row_chunk):
                count = min(self.row_chunk, rows - start)
                with self.pager._lock:
                    tensor = self.pager._read_rows(
                        name,
                        start,
                        count,
                        dtype=self.pager.torch.bfloat16,
                        device=self.pager.torch.device("cpu"),
                    )
                compute = tensor.float().contiguous()
                packed = np.empty(count * row_bytes, dtype=np.uint8)
                self.native.quantize(compute, packed, fmt=fmt, threads=self.threads)
                view = memoryview(packed)
                offset = 0
                while offset < view.nbytes:
                    written = os.write(descriptor, view[offset:])
                    if written <= 0:
                        raise OSError("short Q4 tensor write")
                    offset += written
                digest.update(view)
                del packed, compute, tensor
                if progress is not None:
                    progress(
                        {
                            "event": "q4_tensor_rows",
                            "name": name,
                            "format": fmt,
                            "rows_done": start + count,
                            "rows": rows,
                            "payload_bytes": payload_bytes,
                        }
                    )
            os.fsync(descriptor)
            metadata = os.fstat(descriptor)
            if metadata.st_size != payload_bytes:
                raise Q4BankError(f"Q4 payload size differs for {name!r}")
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, final)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
        return Q4TensorEntry(
            name=name,
            shape=shape,
            format=fmt,
            file=file,
            row_bytes=row_bytes,
            payload_bytes=payload_bytes,
            payload_sha256=digest.hexdigest(),
        )

    def build(
        self,
        *,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            document = _strict_document(manifest_path, schema=Q4_BANK_SCHEMA)
            if (
                document["body"].get("source") != self._source_binding()
                or document["body"].get("native_abi") != Q4_NATIVE_ABI
                or document["body"].get("format_policy") != self.format_policy
            ):
                raise Q4BankError("existing Q4 manifest belongs to another build")
            _verified_payload_entries(self.root, document)
            return document
        self.root.mkdir(parents=True, exist_ok=True)
        source = self._source_binding()
        state = self._load_state(source)
        completed: dict[str, Q4TensorEntry] = {}
        for raw in state["completed"]:
            if not isinstance(raw, Mapping):
                raise Q4BankError("Q4 build state contains an invalid tensor")
            entry = Q4TensorEntry.from_document(raw)
            path = self.root / "weights" / entry.file
            if _regular_file(path, "completed Q4 payload").st_size != entry.payload_bytes:
                raise Q4BankError("completed Q4 payload differs from build state")
            if _file_sha256(path) != entry.payload_sha256:
                raise Q4BankError("completed Q4 payload digest differs from build state")
            completed[entry.name] = entry
        eligible = self._eligible()
        started = time.time_ns()
        # Publish zero-copy hardlinks before the disk preflight.  This lets a
        # caller retire the superseded source-bank directory while the shared
        # inodes remain owned by this resumable build.
        for name, shape, fmt in eligible:
            if name in completed:
                continue
            reused = self._reusable_entry(name=name, shape=shape, fmt=fmt)
            if reused is None:
                continue
            entry = self._build_tensor(
                name=name,
                shape=shape,
                fmt=fmt,
                progress=None,
            )
            completed[name] = entry
            state = {
                **state,
                "completed": [asdict(completed[key]) for key in sorted(completed)],
            }
            self._save_state(state)
            if progress is not None:
                progress(
                    {
                        "event": "q4_tensor_reused",
                        "name": name,
                        "format": fmt,
                        "tensor_count": len(completed),
                    }
                )
        planned = self.plan()
        remaining_bytes = planned["payload_bytes"] - sum(
            entry.payload_bytes for entry in completed.values()
        )
        reusable_bytes = 0
        for name, shape, fmt in self._eligible():
            if name not in completed:
                reused = self._reusable_entry(name=name, shape=shape, fmt=fmt)
                if reused is not None:
                    reusable_bytes += reused.payload_bytes
        remaining_bytes -= reusable_bytes
        free_bytes = shutil.disk_usage(self.root).free
        if free_bytes < remaining_bytes + 512 * 1024**2:
            raise Q4BankError(
                "Q4 build needs "
                f"{remaining_bytes + 512 * 1024**2} free bytes including headroom, "
                f"only {free_bytes} are available"
            )
        for name, shape, fmt in eligible:
            entry = completed.get(name)
            if entry is not None:
                if entry.shape != shape or entry.format != fmt:
                    raise Q4BankError("resumed Q4 tensor plan changed")
                continue
            entry = self._build_tensor(
                name=name,
                shape=shape,
                fmt=fmt,
                progress=progress,
            )
            completed[name] = entry
            state = {**state, "completed": [asdict(completed[key]) for key in sorted(completed)]}
            self._save_state(state)
            if progress is not None:
                progress(
                    {
                        "event": "q4_tensor_complete",
                        "name": name,
                        "format": fmt,
                        "tensor_count": len(completed),
                    }
                )
        entries = [asdict(completed[key]) for key in sorted(completed)]
        body = {
            "source": source,
            "native_abi": Q4_NATIVE_ABI,
            "format_policy": self.format_policy,
            "block_size": Q4_BLOCK_SIZE,
            "tensor_count": len(entries),
            "payload_bytes": sum(row["payload_bytes"] for row in entries),
            "source_bf16_bytes": sum(
                row["shape"][0] * row["shape"][1] * 2 for row in entries
            ),
            "tensors": entries,
            "created_unix_ns": started,
        }
        document = {"schema": Q4_BANK_SCHEMA, "body": body, "sha256": _digest(body)}
        _atomic_write(manifest_path, document)
        (self.root / ".build-state.json").unlink(missing_ok=True)
        document = _strict_document(manifest_path, schema=Q4_BANK_SCHEMA)
        _write_payload_verification_cache(self.root, document, completed)
        return document


__all__ = [
    "Q4_0",
    "Q8_0",
    "Q4_BANK_SCHEMA",
    "Q4_BASE_POLICY",
    "Q4_BALANCED_POLICY",
    "Q4_FORMAT_POLICIES",
    "Q4_RECURRENT_POLICY",
    "Q4Bank",
    "Q4BankBuilder",
    "Q4BankError",
    "Q4NativeKernel",
    "Q4TensorEntry",
]
