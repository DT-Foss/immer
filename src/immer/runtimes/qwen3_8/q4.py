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
import math
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
Q4_BALANCED_POLICY = "q4_0-gate-up-full-attn+q8_0-linear-attn-down-embedding-head/v1"
Q4_RECURRENT_POLICY = "q4_0-mlp-full-attn+q8_0-linear-attn-embedding-head/v1"
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
        if "f16c" in cpu:
            flags.append("-mf16c")
    flags.extend((str(source), "-o", str(output), "-lm"))
    return tuple(flags)


@lru_cache(maxsize=1)
def _native_library() -> "Q4NativeKernel":
    return Q4NativeKernel.load()


class Q4NativeKernel:
    """Small ctypes bridge to the package-shipped native CPU kernel."""

    def __init__(
        self, library: ctypes.CDLL, *, path: Path, build_seconds: float
    ) -> None:
        self.library = library
        self.path = path
        self.build_seconds = float(build_seconds)
        self._configure()
        if self.library.immer_q4_abi() != Q4_NATIVE_ABI:
            raise Q4BankError("native Q4 ABI differs from the Python runtime")
        self.avx2 = bool(self.library.immer_q4_has_avx2())
        self.silu_bf16_table = self._build_silu_bf16_table()

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
        self.library.immer_q4_topk_bf16_f32.argtypes = (
            void,
            i64,
            i64,
            void,
            integer,
            i64,
            i64,
            i64,
            void,
            void,
            void,
            void,
            integer,
        )
        self.library.immer_q4_topk_bf16_f32.restype = integer
        self.library.immer_q4_linear_rows_f32.argtypes = (
            void,
            i64,
            i64,
            void,
            integer,
            i64,
            void,
            i64,
            void,
            integer,
        )
        self.library.immer_q4_linear_rows_f32.restype = integer
        self.library.immer_q4_linear_rows_pair_f32.argtypes = (
            void,
            i64,
            i64,
            void,
            integer,
            i64,
            void,
            integer,
            i64,
            void,
            i64,
            void,
            void,
            integer,
        )
        self.library.immer_q4_linear_rows_pair_f32.restype = integer
        self.library.immer_q4_linear_selected_blocks_f32.argtypes = (
            void,
            i64,
            i64,
            void,
            i64,
            void,
            integer,
            i64,
            void,
            integer,
        )
        self.library.immer_q4_linear_selected_blocks_f32.restype = integer
        self.library.immer_q4_linear_routed_f32.argtypes = (
            void,
            i64,
            i64,
            void,
            void,
            void,
            i64,
            i64,
            void,
            integer,
            i64,
            void,
            integer,
        )
        self.library.immer_q4_linear_routed_f32.restype = integer
        self.library.immer_q4_sparse_mlp_f32.argtypes = (
            void,
            i64,
            i64,
            void,
            integer,
            void,
            integer,
            void,
            integer,
            i64,
            i64,
            void,
            i64,
            i64,
            i64,
            void,
            i64,
            void,
            void,
            void,
            void,
            integer,
        )
        self.library.immer_q4_sparse_mlp_f32.restype = integer
        self.library.immer_q4_deltanet_step_f32.argtypes = (
            void,
            i64,
            void,
            integer,
            i64,
            void,
            integer,
            i64,
            void,
            integer,
            i64,
            void,
            integer,
            i64,
            void,
            void,
            void,
            void,
            void,
            void,
            i64,
            i64,
            i64,
            i64,
            i64,
            ctypes.c_float,
            integer,
            void,
            void,
            void,
            void,
            void,
            void,
            void,
            void,
            integer,
        )
        self.library.immer_q4_deltanet_step_f32.restype = integer
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
        self.library.immer_q4_mlp_bf16_f32.argtypes = (
            void,
            i64,
            i64,
            void,
            integer,
            void,
            integer,
            void,
            integer,
            i64,
            i64,
            void,
            void,
            integer,
        )
        self.library.immer_q4_mlp_bf16_f32.restype = integer

    @staticmethod
    def _pointer(value: Any) -> ctypes.c_void_p:
        data_ptr = getattr(value, "data_ptr", None)
        if callable(data_ptr):
            return ctypes.c_void_p(int(data_ptr()))
        interface = getattr(value, "ctypes", None)
        if interface is None:
            raise TypeError("native Q4 value has no stable data pointer")
        return ctypes.c_void_p(int(interface.data))

    @staticmethod
    def _build_silu_bf16_table() -> np.ndarray:
        import torch

        bits = np.arange(65_536, dtype=np.uint16)
        values = torch.from_numpy(bits.copy()).view(torch.bfloat16)
        outputs = torch.nn.functional.silu(values).contiguous()
        return np.asarray(outputs.view(torch.uint16).numpy(), dtype=np.uint16).copy()

    def row_bytes(self, fmt: str, cols: int) -> int:
        if fmt not in _FORMAT_CODES:
            raise ValueError(f"unknown Q4 format: {fmt}")
        value = int(self.library.immer_q4_row_bytes(_FORMAT_CODES[fmt], cols))
        if value <= 0:
            raise Q4BankError("Q4 tensor width must be divisible by 32")
        return value

    def quantize(
        self, values: Any, output: np.ndarray, *, fmt: str, threads: int
    ) -> None:
        if values.ndim != 2 or values.dtype != values.new_empty(()).float().dtype:
            raise TypeError("native Q4 quantization requires a 2D float32 tensor")
        expected = values.shape[0] * self.row_bytes(fmt, values.shape[1])
        if (
            output.dtype != np.uint8
            or not output.flags.c_contiguous
            or output.size != expected
        ):
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
            or any(
                isinstance(x, bool) or not isinstance(x, int) or x <= 0
                for x in entry.shape
            )
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
            if metadata.st_size != entry.payload_bytes or (
                metadata.st_dev,
                metadata.st_ino,
            ) != (linked.st_dev, linked.st_ino):
                raise Q4BankError(f"Q4 payload identity differs: {path}")
            self.mapping = mmap.mmap(descriptor, 0, access=mmap.ACCESS_READ)
        finally:
            os.close(descriptor)
        self.bytes = np.frombuffer(self.mapping, dtype=np.uint8)

    def discard(self, *, offset: int = 0, length: int | None = None) -> bool:
        """Drop resident read-only pages while keeping the stable mapping open."""

        advice = getattr(mmap, "MADV_DONTNEED", None)
        madvise = getattr(self.mapping, "madvise", None)
        if advice is None or not callable(madvise):
            return False
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or (
                length is not None
                and (
                    isinstance(length, bool)
                    or not isinstance(length, int)
                    or length <= 0
                )
            )
        ):
            raise ValueError("mmap discard range is invalid")
        size = len(self.mapping)
        if offset >= size:
            raise ValueError("mmap discard offset exceeds the payload")
        stop = size if length is None else min(size, offset + length)
        page = int(getattr(mmap, "PAGESIZE", 4096))
        start_aligned = offset - offset % page
        stop_aligned = min(size, ((stop + page - 1) // page) * page)
        try:
            madvise(advice, start_aligned, stop_aligned - start_aligned)
        except (OSError, ValueError):
            return False
        return True

    def close(self) -> None:
        self.bytes = np.empty(0, dtype=np.uint8)
        self.mapping.close()


@dataclass(slots=True)
class Q4BankMetrics:
    mapped_tensors: int = 0
    mapped_payload_bytes: int = 0
    mapping_discard_calls: int = 0
    mapping_discard_bytes: int = 0
    mapping_discard_fallback_closes: int = 0
    mapping_reopens: int = 0
    native_topk_calls: int = 0
    native_topk_rows: int = 0
    native_topk_discard_bytes: int = 0
    full_mlp_calls: int = 0
    full_mlp_rows: int = 0
    linear_calls: int = 0
    linear_group_calls: int = 0
    linear_row_calls: int = 0
    sparse_block_calls: int = 0
    sparse_coordinate_calls: int = 0
    fused_mlp_calls: int = 0
    fused_mlp_rows: int = 0
    fused_deltanet_calls: int = 0
    fused_deltanet_rows: int = 0
    input_quantizations: int = 0
    linear_input_rows: int = 0
    selected_output_rows: int = 0
    selected_input_blocks: int = 0
    selected_input_coordinates: int = 0
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
        self._mapped_once: set[str] = set()
        self._touched: set[str] = set()
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
            if name in self._mapped_once:
                self._stats.mapping_reopens += 1
            else:
                self._mapped_once.add(name)
        self._touched.add(name)
        return entry, mapped

    def discard_rows(self, name: str, start_row: int, row_count: int) -> None:
        """Discard one consumed packed row interval without changing its bytes."""

        with self._lock:
            if self._closed:
                raise Q4BankError("Q4 bank is closed")
            entry = self.entries.get(name)
            mapped = self._mapped.get(name)
            if entry is None:
                raise KeyError(name)
            if (
                isinstance(start_row, bool)
                or not isinstance(start_row, int)
                or start_row < 0
                or isinstance(row_count, bool)
                or not isinstance(row_count, int)
                or row_count <= 0
                or start_row + row_count > entry.shape[0]
            ):
                raise ValueError("Q4 discard row interval is invalid")
            if mapped is None:
                return
            offset = start_row * entry.row_bytes
            length = row_count * entry.row_bytes
            if mapped.discard(offset=offset, length=length):
                self._stats.mapping_discard_calls += 1
                self._stats.mapping_discard_bytes += length

    def release_touched(self) -> None:
        """Remove residency accumulated since the previous execution boundary."""

        with self._lock:
            if self._closed:
                raise Q4BankError("Q4 bank is closed")
            names = tuple(self._touched)
            self._touched.clear()
            for name in names:
                mapped = self._mapped.get(name)
                if mapped is None:
                    continue
                entry = self.entries[name]
                if mapped.discard():
                    self._stats.mapping_discard_calls += 1
                    self._stats.mapping_discard_bytes += entry.payload_bytes
                    continue
                # Platforms without MADV_DONTNEED still get the same bounded
                # residency by unmapping. The next exact access reopens bytes.
                self._mapped.pop(name).close()
                self._stats.mapping_discard_fallback_closes += 1
                self._stats.mapping_discard_calls += 1
                self._stats.mapping_discard_bytes += entry.payload_bytes

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
            compute = (
                values.detach()
                .to(dtype=torch.float32)
                .reshape(input_rows, entry.shape[1])
                .contiguous()
            )
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

    def topk(
        self,
        values: Any,
        name: str,
        *,
        k: int,
        block_rows: int,
        output_dtype: Any,
    ) -> tuple[Any, Any]:
        """Scan one packed matrix once and discard each consumed row interval."""

        import torch

        with self._lock:
            entry, mapped = self._mapping(name)
            if not isinstance(values, torch.Tensor):
                values = torch.as_tensor(values)
            if (
                values.ndim < 1
                or values.shape[-1] != entry.shape[1]
                or values.device.type != "cpu"
                or output_dtype != torch.bfloat16
            ):
                raise ValueError("native Q4 top-k requires BF16-output CPU rows")
            if (
                isinstance(k, bool)
                or not isinstance(k, int)
                or not 1 <= k <= min(256, entry.shape[0])
                or isinstance(block_rows, bool)
                or not isinstance(block_rows, int)
                or block_rows <= 0
            ):
                raise ValueError("native Q4 top-k dimensions are invalid")
            leading = tuple(values.shape[:-1])
            input_rows = values.numel() // values.shape[-1]
            compute = (
                values.detach()
                .to(dtype=torch.float32)
                .reshape(input_rows, entry.shape[1])
                .contiguous()
            )
            top_values = torch.empty((input_rows, k), dtype=torch.float32)
            top_ids = torch.empty((input_rows, k), dtype=torch.int64)
            discarded_bytes = ctypes.c_int64()
            discard_calls = ctypes.c_int64()
            native_block_rows = min(entry.shape[0], max(block_rows, 8192))
            code = self.native.library.immer_q4_topk_bf16_f32(
                self.native._pointer(compute),
                input_rows,
                entry.shape[1],
                self.native._pointer(mapped.bytes),
                _FORMAT_CODES[entry.format],
                entry.shape[0],
                k,
                native_block_rows,
                self.native._pointer(top_values),
                self.native._pointer(top_ids),
                ctypes.byref(discarded_bytes),
                ctypes.byref(discard_calls),
                self.threads,
            )
            if code == 2:
                raise ValueError("native Q4 top-k values are non-finite")
            if code:
                raise Q4BankError(f"native Q4 top-k failed with code {code}")
            result_values = top_values.to(dtype=output_dtype).reshape(*leading, k)
            result_ids = top_ids.reshape(*leading, k)
            self._stats.linear_calls += 1
            self._stats.input_quantizations += input_rows
            self._stats.linear_input_rows += input_rows
            self._stats.logical_weight_bytes += entry.payload_bytes
            self._stats.output_bytes += (
                result_values.numel() * result_values.element_size()
            )
            self._stats.native_topk_calls += 1
            self._stats.native_topk_rows += input_rows * entry.shape[0]
            if discard_calls.value > 0:
                self._stats.mapping_discard_calls += discard_calls.value
                self._stats.mapping_discard_bytes += discarded_bytes.value
                self._stats.native_topk_discard_bytes += discarded_bytes.value
            return result_values, result_ids

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
            compute = (
                values.detach()
                .to(dtype=torch.float32)
                .reshape(input_rows, input_columns)
                .contiguous()
            )
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

    def mlp(
        self,
        values: Any,
        names: tuple[str, str, str],
        *,
        output_dtype: Any,
    ) -> Any:
        """Execute the complete BF16-rounded SwiGLU MLP in one native team."""

        import torch

        with self._lock:
            if len(names) != 3 or len(set(names)) != 3:
                raise ValueError("Q4 full MLP requires Gate, Up, and Down names")
            gate, up, down = (self._mapping(name) for name in names)
            gate_entry, gate_map = gate
            up_entry, up_map = up
            down_entry, down_map = down
            if (
                gate_entry.shape != up_entry.shape
                or down_entry.shape[1] != gate_entry.shape[0]
                or output_dtype != torch.bfloat16
            ):
                raise Q4BankError("Q4 full MLP shapes or BF16 ABI disagree")
            if not isinstance(values, torch.Tensor):
                values = torch.as_tensor(values)
            if (
                values.ndim < 1
                or values.shape[-1] != gate_entry.shape[1]
                or values.device.type != "cpu"
            ):
                raise ValueError("Q4 full MLP input differs from Gate/Up")
            leading = tuple(values.shape[:-1])
            input_rows = values.numel() // values.shape[-1]
            compute = (
                values.detach()
                .to(dtype=torch.float32)
                .reshape(input_rows, values.shape[-1])
                .contiguous()
            )
            output = torch.empty((input_rows, down_entry.shape[0]), dtype=torch.float32)
            code = self.native.library.immer_q4_mlp_bf16_f32(
                self.native._pointer(compute),
                input_rows,
                gate_entry.shape[1],
                self.native._pointer(gate_map.bytes),
                _FORMAT_CODES[gate_entry.format],
                self.native._pointer(up_map.bytes),
                _FORMAT_CODES[up_entry.format],
                self.native._pointer(down_map.bytes),
                _FORMAT_CODES[down_entry.format],
                gate_entry.shape[0],
                down_entry.shape[0],
                self.native._pointer(self.native.silu_bf16_table),
                self.native._pointer(output),
                self.threads,
            )
            if code == 2:
                raise ValueError("native Q4 full MLP values are non-finite")
            if code:
                raise Q4BankError(f"native Q4 full MLP failed with code {code}")
            result = output.to(dtype=output_dtype).reshape(
                *leading,
                down_entry.shape[0],
            )
            self._stats.linear_calls += 3
            self._stats.input_quantizations += input_rows * 2
            self._stats.linear_input_rows += input_rows * 3
            self._stats.logical_weight_bytes += sum(
                entry.payload_bytes for entry in (gate_entry, up_entry, down_entry)
            )
            self._stats.output_bytes += result.numel() * result.element_size()
            self._stats.full_mlp_calls += 1
            self._stats.full_mlp_rows += input_rows
            return result

    def linear_rows(
        self,
        values: Any,
        name: str,
        row_ids: tuple[int, ...],
        *,
        output_dtype: Any | None = None,
    ) -> Any:
        """Apply only selected matrix rows without dequantizing them."""

        import torch

        with self._lock:
            entry, mapped = self._mapping(name)
            if not row_ids or any(
                isinstance(row, bool)
                or not isinstance(row, int)
                or not 0 <= row < entry.shape[0]
                for row in row_ids
            ):
                raise ValueError("Q4 selected linear rows are invalid")
            if not isinstance(values, torch.Tensor):
                values = torch.as_tensor(values)
            if values.ndim < 1 or values.shape[-1] != entry.shape[1]:
                raise Q4BankError(f"Q4 input width disagrees with {name!r}")
            if values.device.type != "cpu":
                raise Q4BankError("Q4 execution is currently CPU-only")
            leading = tuple(values.shape[:-1])
            input_rows = values.numel() // values.shape[-1]
            compute = (
                values.detach()
                .to(dtype=torch.float32)
                .reshape(input_rows, entry.shape[1])
                .contiguous()
            )
            ids = torch.tensor(row_ids, dtype=torch.int64)
            output = torch.empty((input_rows, len(row_ids)), dtype=torch.float32)
            code = self.native.library.immer_q4_linear_rows_f32(
                self.native._pointer(compute),
                input_rows,
                entry.shape[1],
                self.native._pointer(mapped.bytes),
                _FORMAT_CODES[entry.format],
                entry.shape[0],
                self.native._pointer(ids),
                len(row_ids),
                self.native._pointer(output),
                self.threads,
            )
            if code:
                raise Q4BankError(
                    f"native Q4 selected-row linear failed with code {code}"
                )
            dtype = values.dtype if output_dtype is None else output_dtype
            result = output.to(dtype=dtype).reshape(*leading, len(row_ids))
            self._stats.linear_calls += 1
            self._stats.linear_row_calls += 1
            self._stats.input_quantizations += 1
            self._stats.linear_input_rows += input_rows
            self._stats.selected_output_rows += len(row_ids)
            self._stats.logical_weight_bytes += len(row_ids) * entry.row_bytes
            self._stats.output_bytes += result.numel() * result.element_size()
            return result

    def linear_rows_pair(
        self,
        values: Any,
        names: tuple[str, str],
        row_ids: tuple[int, ...],
        *,
        output_dtype: Any | None = None,
    ) -> tuple[Any, Any]:
        """Apply the same selected rows from two matrices with one input pass."""

        import torch

        with self._lock:
            if len(names) != 2 or names[0] == names[1]:
                raise ValueError("Q4 selected-row pair requires two distinct matrices")
            resolved = tuple(self._mapping(name) for name in names)
            entries = tuple(row[0] for row in resolved)
            mapped = tuple(row[1] for row in resolved)
            if entries[0].shape[1] != entries[1].shape[1]:
                raise Q4BankError("Q4 selected-row pair input widths differ")
            if not row_ids or any(
                isinstance(row, bool)
                or not isinstance(row, int)
                or row < 0
                or any(row >= entry.shape[0] for entry in entries)
                for row in row_ids
            ):
                raise ValueError("Q4 selected linear rows are invalid")
            if not isinstance(values, torch.Tensor):
                values = torch.as_tensor(values)
            input_columns = entries[0].shape[1]
            if values.ndim < 1 or values.shape[-1] != input_columns:
                raise Q4BankError("Q4 selected-row pair input width differs")
            if values.device.type != "cpu":
                raise Q4BankError("Q4 execution is currently CPU-only")
            leading = tuple(values.shape[:-1])
            input_rows = values.numel() // input_columns
            compute = (
                values.detach()
                .to(dtype=torch.float32)
                .reshape(input_rows, input_columns)
                .contiguous()
            )
            ids = torch.tensor(row_ids, dtype=torch.int64)
            outputs = tuple(
                torch.empty((input_rows, len(row_ids)), dtype=torch.float32)
                for _ in entries
            )
            code = self.native.library.immer_q4_linear_rows_pair_f32(
                self.native._pointer(compute),
                input_rows,
                input_columns,
                self.native._pointer(mapped[0].bytes),
                _FORMAT_CODES[entries[0].format],
                entries[0].shape[0],
                self.native._pointer(mapped[1].bytes),
                _FORMAT_CODES[entries[1].format],
                entries[1].shape[0],
                self.native._pointer(ids),
                len(row_ids),
                self.native._pointer(outputs[0]),
                self.native._pointer(outputs[1]),
                self.threads,
            )
            if code:
                raise Q4BankError(
                    f"native Q4 selected-row pair failed with code {code}"
                )
            dtype = values.dtype if output_dtype is None else output_dtype
            results = tuple(
                output.to(dtype=dtype).reshape(*leading, len(row_ids))
                for output in outputs
            )
            self._stats.linear_calls += 2
            self._stats.linear_row_calls += 2
            self._stats.input_quantizations += 1
            self._stats.linear_input_rows += input_rows * 2
            self._stats.selected_output_rows += 2 * len(row_ids)
            self._stats.logical_weight_bytes += sum(
                len(row_ids) * entry.row_bytes for entry in entries
            )
            self._stats.output_bytes += sum(
                result.numel() * result.element_size() for result in results
            )
            return results[0], results[1]

    def linear_selected_blocks(
        self,
        block_values: Any,
        block_ids: Any,
        name: str,
        *,
        output_dtype: Any | None = None,
    ) -> Any:
        """Multiply compact 32-value activation blocks by packed matrix columns."""

        import torch

        with self._lock:
            entry, mapped = self._mapping(name)
            if not isinstance(block_values, torch.Tensor):
                block_values = torch.as_tensor(block_values)
            if not isinstance(block_ids, torch.Tensor):
                block_ids = torch.as_tensor(block_ids)
            if (
                block_values.ndim != 3
                or block_values.shape[2] != Q4_BLOCK_SIZE
                or block_values.shape[:2] != block_ids.shape
                or block_ids.ndim != 2
                or block_values.shape[0] < 1
                or block_values.shape[1] < 1
            ):
                raise ValueError("Q4 selected blocks differ from their block IDs")
            if block_values.device.type != "cpu" or block_ids.device.type != "cpu":
                raise Q4BankError("Q4 execution is currently CPU-only")
            ids = block_ids.detach().to(dtype=torch.int64).contiguous()
            total_blocks = entry.shape[1] // Q4_BLOCK_SIZE
            if bool((ids < 0).any()) or bool((ids >= total_blocks).any()):
                raise ValueError("Q4 selected block ID is outside the matrix")
            if any(len(set(row)) != len(row) for row in ids.tolist()):
                raise ValueError("Q4 selected block IDs must be unique per row")
            compute = block_values.detach().to(dtype=torch.float32).contiguous()
            input_rows, selected_blocks, _ = compute.shape
            output = torch.empty((input_rows, entry.shape[0]), dtype=torch.float32)
            code = self.native.library.immer_q4_linear_selected_blocks_f32(
                self.native._pointer(compute),
                input_rows,
                selected_blocks,
                self.native._pointer(ids),
                entry.shape[1],
                self.native._pointer(mapped.bytes),
                _FORMAT_CODES[entry.format],
                entry.shape[0],
                self.native._pointer(output),
                self.threads,
            )
            if code:
                raise Q4BankError(
                    f"native Q4 selected-block linear failed with code {code}"
                )
            dtype = block_values.dtype if output_dtype is None else output_dtype
            result = output.to(dtype=dtype)
            block_bytes = _FORMAT_BLOCK_BYTES[entry.format]
            self._stats.linear_calls += 1
            self._stats.sparse_block_calls += 1
            self._stats.input_quantizations += input_rows * selected_blocks
            self._stats.linear_input_rows += input_rows
            self._stats.selected_input_blocks += input_rows * selected_blocks
            self._stats.logical_weight_bytes += (
                input_rows * selected_blocks * entry.shape[0] * block_bytes
            )
            self._stats.output_bytes += result.numel() * result.element_size()
            return result

    def linear_routed(
        self,
        full_block_values: Any,
        full_block_ids: Any,
        sparse_values: Any,
        sparse_coords: Any,
        name: str,
        *,
        output_dtype: Any | None = None,
    ) -> Any:
        """Run dense selected Q4 blocks plus scattered packed coordinates."""

        import torch

        with self._lock:
            entry, mapped = self._mapping(name)
            tensors = []
            for value in (
                full_block_values,
                full_block_ids,
                sparse_values,
                sparse_coords,
            ):
                tensors.append(
                    value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                )
            full_value, full_ids, sparse_value, sparse_ids = tensors
            if (
                full_value.ndim != 3
                or full_value.shape[2] != Q4_BLOCK_SIZE
                or full_ids.ndim != 2
                or full_value.shape[:2] != full_ids.shape
                or sparse_value.ndim != 2
                or sparse_ids.ndim != 2
                or sparse_value.shape != sparse_ids.shape
                or full_value.shape[0] != sparse_value.shape[0]
                or full_value.shape[0] < 1
                or full_value.shape[1] < 1
                or sparse_value.shape[1] < 1
            ):
                raise ValueError("Q4 routed blocks and coordinates are misaligned")
            if any(value.device.type != "cpu" for value in tensors):
                raise Q4BankError("Q4 execution is currently CPU-only")
            full_ids = full_ids.detach().to(dtype=torch.int64).contiguous()
            sparse_ids = sparse_ids.detach().to(dtype=torch.int64).contiguous()
            total_blocks = entry.shape[1] // Q4_BLOCK_SIZE
            if (
                bool((full_ids < 0).any())
                or bool((full_ids >= total_blocks).any())
                or bool((sparse_ids < 0).any())
                or bool((sparse_ids >= entry.shape[1]).any())
            ):
                raise ValueError("Q4 routed coordinate is outside the matrix")
            full_rows = full_ids.tolist()
            sparse_rows = sparse_ids.tolist()
            for blocks, coordinates in zip(full_rows, sparse_rows, strict=True):
                if (
                    len(set(blocks)) != len(blocks)
                    or len(set(coordinates)) != len(coordinates)
                    or {coordinate // Q4_BLOCK_SIZE for coordinate in coordinates}
                    & set(blocks)
                ):
                    raise ValueError("Q4 routed coordinates overlap or repeat")
            full_compute = full_value.detach().to(dtype=torch.float32).contiguous()
            sparse_compute = sparse_value.detach().to(dtype=torch.float32).contiguous()
            input_rows, full_blocks, _ = full_compute.shape
            sparse_count = sparse_compute.shape[1]
            output = torch.empty((input_rows, entry.shape[0]), dtype=torch.float32)
            code = self.native.library.immer_q4_linear_routed_f32(
                self.native._pointer(full_compute),
                input_rows,
                full_blocks,
                self.native._pointer(full_ids),
                self.native._pointer(sparse_compute),
                self.native._pointer(sparse_ids),
                sparse_count,
                entry.shape[1],
                self.native._pointer(mapped.bytes),
                _FORMAT_CODES[entry.format],
                entry.shape[0],
                self.native._pointer(output),
                self.threads,
            )
            if code:
                raise Q4BankError(f"native Q4 routed linear failed with code {code}")
            dtype = full_value.dtype if output_dtype is None else output_dtype
            result = output.to(dtype=dtype)
            block_bytes = _FORMAT_BLOCK_BYTES[entry.format]
            sparse_touched_blocks = sum(
                len({coordinate // Q4_BLOCK_SIZE for coordinate in row})
                for row in sparse_rows
            )
            self._stats.linear_calls += 1
            self._stats.sparse_coordinate_calls += 1
            self._stats.input_quantizations += input_rows * full_blocks
            self._stats.linear_input_rows += input_rows
            self._stats.selected_input_blocks += input_rows * full_blocks
            self._stats.selected_input_coordinates += input_rows * sparse_count
            self._stats.logical_weight_bytes += (
                (input_rows * full_blocks + sparse_touched_blocks)
                * entry.shape[0]
                * block_bytes
            )
            self._stats.output_bytes += result.numel() * result.element_size()
            return result

    def sparse_mlp(
        self,
        values: Any,
        names: tuple[str, str, str],
        *,
        pilot_ids: Any,
        coefficients: Any,
        block_size: int,
        selected_block_count: int,
        affine_scale: Any,
        affine_bias: Any,
        output_dtype: Any | None = None,
    ) -> tuple[Any, Any]:
        """Execute routing, selected SwiGLU, and selected Down in one kernel."""

        import torch

        with self._lock:
            if len(names) != 3 or len(set(names)) != 3:
                raise ValueError("Q4 fused MLP requires Gate, Up, and Down names")
            gate, up, down = (self._mapping(name) for name in names)
            gate_entry, gate_map = gate
            up_entry, up_map = up
            down_entry, down_map = down
            if (
                gate_entry.shape != up_entry.shape
                or down_entry.shape[1] != gate_entry.shape[0]
            ):
                raise Q4BankError("Q4 fused MLP tensor shapes disagree")
            if not isinstance(values, torch.Tensor):
                values = torch.as_tensor(values)
            if (
                values.ndim < 1
                or values.shape[-1] != gate_entry.shape[1]
                or values.device.type != "cpu"
            ):
                raise ValueError("Q4 fused MLP input differs from Gate/Up")
            if (
                isinstance(block_size, bool)
                or not isinstance(block_size, int)
                or block_size <= 0
                or gate_entry.shape[0] % block_size
                or isinstance(selected_block_count, bool)
                or not isinstance(selected_block_count, int)
                or not 0 < selected_block_count < gate_entry.shape[0] // block_size
            ):
                raise ValueError("Q4 fused MLP block topology is invalid")
            ids = (
                (
                    pilot_ids
                    if isinstance(pilot_ids, torch.Tensor)
                    else torch.tensor(pilot_ids)
                )
                .detach()
                .to(dtype=torch.int64)
                .contiguous()
            )
            coeff = (
                (
                    coefficients
                    if isinstance(coefficients, torch.Tensor)
                    else torch.tensor(coefficients)
                )
                .detach()
                .to(dtype=torch.float64)
                .contiguous()
            )
            block_count = gate_entry.shape[0] // block_size
            if (
                ids.ndim != 2
                or ids.shape[0] != block_count
                or ids.shape[1] < 1
                or coeff.shape != (block_count, ids.shape[1] + 1)
            ):
                raise ValueError("Q4 fused MLP pilots and coefficients disagree")
            scale = (
                (
                    affine_scale
                    if isinstance(affine_scale, torch.Tensor)
                    else torch.tensor(affine_scale)
                )
                .detach()
                .to(dtype=torch.float32)
                .contiguous()
            )
            bias = (
                (
                    affine_bias
                    if isinstance(affine_bias, torch.Tensor)
                    else torch.tensor(affine_bias)
                )
                .detach()
                .to(dtype=torch.float32)
                .contiguous()
            )
            if scale.shape != (down_entry.shape[0],) or bias.shape != scale.shape:
                raise ValueError("Q4 fused MLP affine vectors disagree with Down")
            leading = tuple(values.shape[:-1])
            input_rows = values.numel() // values.shape[-1]
            compute = (
                values.detach()
                .to(dtype=torch.float32)
                .reshape(input_rows, values.shape[-1])
                .contiguous()
            )
            output = torch.empty((input_rows, down_entry.shape[0]), dtype=torch.float32)
            selected = torch.empty(
                (input_rows, selected_block_count), dtype=torch.int64
            )
            code = self.native.library.immer_q4_sparse_mlp_f32(
                self.native._pointer(compute),
                input_rows,
                gate_entry.shape[1],
                self.native._pointer(gate_map.bytes),
                _FORMAT_CODES[gate_entry.format],
                self.native._pointer(up_map.bytes),
                _FORMAT_CODES[up_entry.format],
                self.native._pointer(down_map.bytes),
                _FORMAT_CODES[down_entry.format],
                gate_entry.shape[0],
                down_entry.shape[0],
                self.native._pointer(ids),
                block_count,
                block_size,
                ids.shape[1],
                self.native._pointer(coeff),
                selected_block_count,
                self.native._pointer(scale),
                self.native._pointer(bias),
                self.native._pointer(output),
                self.native._pointer(selected),
                self.threads,
            )
            if code:
                raise Q4BankError(f"native Q4 fused MLP failed with code {code}")
            dtype = values.dtype if output_dtype is None else output_dtype
            result = output.to(dtype=dtype).reshape(*leading, down_entry.shape[0])
            pilot_count = ids.numel()
            selected_neurons = selected_block_count * block_size
            down_blocks = selected_neurons // Q4_BLOCK_SIZE
            actual = input_rows * (
                (pilot_count + selected_neurons)
                * (gate_entry.row_bytes + up_entry.row_bytes)
                + down_blocks
                * down_entry.shape[0]
                * _FORMAT_BLOCK_BYTES[down_entry.format]
            )
            self._stats.linear_calls += 3
            self._stats.input_quantizations += input_rows * (1 + down_blocks)
            self._stats.linear_input_rows += input_rows * 3
            self._stats.selected_input_blocks += input_rows * down_blocks
            self._stats.selected_output_rows += (
                input_rows * 2 * (pilot_count + selected_neurons)
            )
            self._stats.logical_weight_bytes += actual
            self._stats.output_bytes += result.numel() * result.element_size()
            self._stats.fused_mlp_calls += 1
            self._stats.fused_mlp_rows += input_rows
            return result, selected

    def deltanet_step(
        self,
        hidden: Any,
        names: tuple[str, str, str, str],
        *,
        conv_weight: Any,
        A_log: Any,
        dt_bias: Any,
        norm_weight: Any,
        conv_state: Any,
        recurrent_state: Any,
        key_heads: int,
        value_heads: int,
        key_dim: int,
        value_dim: int,
        rms_eps: float,
        output_dtype: Any,
    ) -> tuple[Any, Any, Any, Any, Any]:
        """Fuse packed input projections with one exact DeltaNet state step."""

        import torch

        with self._lock:
            if len(names) != 4 or len(set(names)) != 4:
                raise ValueError("Q4 fused DeltaNet requires QKV/Z/B/A names")
            resolved = tuple(self._mapping(name) for name in names)
            entries = tuple(row[0] for row in resolved)
            mapped = tuple(row[1] for row in resolved)
            if not isinstance(hidden, torch.Tensor):
                hidden = torch.as_tensor(hidden)
            if (
                hidden.device.type != "cpu"
                or hidden.ndim != 3
                or hidden.shape[0] != 1
                or hidden.shape[1] != 1
                or hidden.shape[2] != entries[0].shape[1]
                or any(entry.shape[1] != hidden.shape[2] for entry in entries)
            ):
                raise ValueError("Q4 fused DeltaNet hidden shape is invalid")
            for field, value in (
                ("key_heads", key_heads),
                ("value_heads", value_heads),
                ("key_dim", key_dim),
                ("value_dim", value_dim),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{field} must be a positive integer")
            qkv_rows = 2 * key_heads * key_dim + value_heads * value_dim
            z_rows = value_heads * value_dim
            if (
                entries[0].shape[0] != qkv_rows
                or entries[1].shape[0] != z_rows
                or entries[2].shape[0] != value_heads
                or entries[3].shape[0] != value_heads
            ):
                raise Q4BankError("Q4 fused DeltaNet projection topology changed")

            def f32(value: Any, field: str) -> Any:
                tensor = (
                    value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
                )
                if tensor.device.type != "cpu" or not tensor.is_floating_point():
                    raise ValueError(f"{field} must be a floating CPU tensor")
                return tensor.detach().to(dtype=torch.float32).contiguous()

            compute = f32(hidden, "hidden").reshape(1, hidden.shape[2])
            conv = f32(conv_weight, "conv_weight")
            if conv.ndim == 3 and conv.shape[1] == 1:
                conv = conv[:, 0, :].contiguous()
            if conv.ndim != 2 or conv.shape[0] != qkv_rows or conv.shape[1] < 1:
                raise ValueError("conv_weight shape is invalid")
            kernel_size = int(conv.shape[1])
            a_log = f32(A_log, "A_log").reshape(-1)
            delta_bias = f32(dt_bias, "dt_bias").reshape(-1)
            norm = f32(norm_weight, "norm_weight").reshape(-1)
            previous_conv = f32(conv_state, "conv_state")
            previous_recurrent = f32(recurrent_state, "recurrent_state")
            if (
                a_log.shape != (value_heads,)
                or delta_bias.shape != (value_heads,)
                or norm.shape != (value_dim,)
                or previous_conv.shape != (1, qkv_rows, kernel_size)
                or previous_recurrent.shape != (1, value_heads, key_dim, value_dim)
            ):
                raise ValueError("Q4 fused DeltaNet control/state shape is invalid")
            if (
                isinstance(rms_eps, bool)
                or not isinstance(rms_eps, (int, float))
                or not math.isfinite(float(rms_eps))
                or float(rms_eps) <= 0.0
            ):
                raise ValueError("rms_eps must be finite and positive")
            projected_qkv = torch.empty((1, 1, qkv_rows), dtype=torch.float32)
            projected_z = torch.empty((1, 1, z_rows), dtype=torch.float32)
            projected_b = torch.empty((1, 1, value_heads), dtype=torch.float32)
            projected_a = torch.empty((1, 1, value_heads), dtype=torch.float32)
            next_conv = torch.empty((1, qkv_rows, kernel_size), dtype=torch.float32)
            null = ctypes.c_void_p()
            code = self.native.library.immer_q4_deltanet_step_f32(
                self.native._pointer(compute),
                hidden.shape[2],
                self.native._pointer(mapped[0].bytes),
                _FORMAT_CODES[entries[0].format],
                qkv_rows,
                self.native._pointer(mapped[1].bytes),
                _FORMAT_CODES[entries[1].format],
                z_rows,
                self.native._pointer(mapped[2].bytes),
                _FORMAT_CODES[entries[2].format],
                value_heads,
                self.native._pointer(mapped[3].bytes),
                _FORMAT_CODES[entries[3].format],
                value_heads,
                self.native._pointer(conv),
                self.native._pointer(a_log),
                self.native._pointer(delta_bias),
                self.native._pointer(norm),
                self.native._pointer(previous_conv),
                self.native._pointer(previous_recurrent),
                key_heads,
                value_heads,
                key_dim,
                value_dim,
                kernel_size,
                float(rms_eps),
                1,
                null,
                self.native._pointer(projected_qkv),
                self.native._pointer(projected_z),
                self.native._pointer(projected_b),
                self.native._pointer(projected_a),
                null,
                self.native._pointer(next_conv),
                null,
                self.threads,
            )
            if code == 2:
                raise ValueError("Q4 fused DeltaNet values are non-finite")
            if code:
                raise Q4BankError(f"native Q4 DeltaNet step failed with code {code}")
            qkv_result = projected_qkv.to(dtype=output_dtype)
            z_result = projected_z.to(dtype=output_dtype)
            b_result = projected_b.to(dtype=output_dtype)
            a_result = projected_a.to(dtype=output_dtype)
            conv_result = next_conv.to(dtype=output_dtype)
            self._stats.linear_calls += 4
            self._stats.linear_group_calls += 1
            self._stats.input_quantizations += 1
            self._stats.linear_input_rows += 4
            self._stats.logical_weight_bytes += sum(
                entry.payload_bytes for entry in entries
            )
            self._stats.output_bytes += (
                qkv_result.numel() * qkv_result.element_size()
                + z_result.numel() * z_result.element_size()
                + b_result.numel() * b_result.element_size()
                + a_result.numel() * a_result.element_size()
                + conv_result.numel() * conv_result.element_size()
            )
            self._stats.fused_deltanet_calls += 1
            self._stats.fused_deltanet_rows += 1
            return qkv_result, z_result, b_result, a_result, conv_result

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
            self._touched.clear()
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
        if (
            isinstance(row_chunk, bool)
            or not isinstance(row_chunk, int)
            or row_chunk <= 0
        ):
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
                or not (
                    name.startswith(("model.language_model.", "mtp."))
                    or name == "lm_head.weight"
                )
                or not isinstance(shape, (list, tuple))
                or len(shape) != 2
                or dtype != "BF16"
                or any(
                    isinstance(x, bool) or not isinstance(x, int) or x <= 0
                    for x in shape
                )
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
                sum(row["payload_bytes"] for row in tensors) - reusable_payload_bytes
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
            if (
                _regular_file(path, "completed Q4 payload").st_size
                != entry.payload_bytes
            ):
                raise Q4BankError("completed Q4 payload differs from build state")
            if _file_sha256(path) != entry.payload_sha256:
                raise Q4BankError(
                    "completed Q4 payload digest differs from build state"
                )
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
            state = {
                **state,
                "completed": [asdict(completed[key]) for key in sorted(completed)],
            }
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
