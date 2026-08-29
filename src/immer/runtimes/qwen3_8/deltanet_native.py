"""Package-built float32 CPU kernel for one recurrent DeltaNet token."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import platform
import secrets
import shutil
import stat
import subprocess
import sys
import time
from functools import lru_cache
from typing import Any


DELTANET_NATIVE_ABI = 1
DELTANET_NATIVE_SOURCE = "deltanet_native.c"


class DeltaNetNativeError(RuntimeError):
    """The package-built DeltaNet kernel could not be built or executed."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise DeltaNetNativeError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DeltaNetNativeError(f"{label} must be a non-symlink regular file")
    return metadata


def _compiler_command(source: Path, output: Path) -> tuple[str, ...]:
    compiler_name = os.environ.get("CC", "cc")
    compiler = shutil.which(compiler_name)
    if compiler is None:
        raise DeltaNetNativeError(
            f"native DeltaNet compiler is unavailable: {compiler_name}"
        )
    flags = [compiler, "-std=c11", "-O3", "-DNDEBUG"]
    if sys.platform == "darwin":
        flags.extend(("-dynamiclib", "-fPIC"))
    else:
        flags.extend(("-shared", "-fPIC", "-fopenmp"))
    flags.extend((str(source), "-o", str(output), "-lm"))
    return tuple(flags)


class _DeltaNetNativeKernel:
    def __init__(
        self,
        library: ctypes.CDLL,
        *,
        path: Path,
        build_seconds: float,
    ) -> None:
        self.library = library
        self.path = path
        self.build_seconds = float(build_seconds)
        void = ctypes.c_void_p
        i64 = ctypes.c_int64
        integer = ctypes.c_int
        self.library.immer_deltanet_abi.argtypes = ()
        self.library.immer_deltanet_abi.restype = ctypes.c_uint32
        self.library.immer_deltanet_step_f32.argtypes = (
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
            void,
            void,
            integer,
        )
        self.library.immer_deltanet_step_f32.restype = integer
        if self.library.immer_deltanet_abi() != DELTANET_NATIVE_ABI:
            raise DeltaNetNativeError(
                "native DeltaNet ABI differs from the Python runtime"
            )

    @classmethod
    def load(cls) -> "_DeltaNetNativeKernel":
        source = Path(__file__).with_name(DELTANET_NATIVE_SOURCE)
        _regular_file(source, "native DeltaNet source")
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        seed = repr(
            (
                DELTANET_NATIVE_ABI,
                platform.machine(),
                sys.platform,
                sys.version_info[:3],
                _file_sha256(source),
                _compiler_command(Path("SOURCE"), Path("OUTPUT")),
            )
        ).encode("utf-8")
        identity = hashlib.sha256(seed).hexdigest()
        cache_root = Path(
            os.environ.get(
                "IMMER_NATIVE_CACHE",
                str(Path.home() / ".cache" / "immer" / "native"),
            )
        ).expanduser()
        target_root = cache_root / f"deltanet-{identity}"
        library_path = target_root / f"libimmer_deltanet{suffix}"
        build_seconds = 0.0
        if not library_path.exists():
            target_root.mkdir(parents=True, exist_ok=True)
            temporary = target_root / (
                f".{library_path.name}.{secrets.token_hex(8)}.tmp"
            )
            started = time.perf_counter()
            try:
                completed = subprocess.run(
                    _compiler_command(source, temporary),
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if completed.returncode != 0:
                    reason = completed.stderr.strip() or completed.stdout.strip()
                    raise DeltaNetNativeError(
                        f"native DeltaNet compilation failed: {reason}"
                    )
                os.replace(temporary, library_path)
            finally:
                temporary.unlink(missing_ok=True)
            build_seconds = time.perf_counter() - started
        _regular_file(library_path, "native DeltaNet library")
        try:
            library = ctypes.CDLL(str(library_path))
        except OSError as exc:
            raise DeltaNetNativeError(
                "cannot load native DeltaNet library"
            ) from exc
        return cls(library, path=library_path, build_seconds=build_seconds)

    @staticmethod
    def pointer(value: Any) -> ctypes.c_void_p:
        return ctypes.c_void_p(int(value.data_ptr()))


@lru_cache(maxsize=1)
def _native_library() -> _DeltaNetNativeKernel:
    return _DeltaNetNativeKernel.load()


def _float32_cpu_tensor(value: Any, name: str, *, ndim: int) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if value.dtype != torch.float32:
        raise ValueError(f"{name} must use float32")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU")
    return value.detach().contiguous()


def deltanet_sequence_one(
    query: Any,
    key: Any,
    value: Any,
    log_decay: Any,
    beta: Any,
    initial_state: Any,
    *,
    threads: int | None = None,
) -> tuple[Any, Any]:
    """Execute one already-normalized/scaled recurrent DeltaNet token on CPU.

    Inputs use ``q/k=[B,1,H,K]``, ``v=[B,1,H,V]``, scalar controls
    ``[B,1,H]``, and state ``[B,H,K,V]``. Inputs are never mutated; both
    returned tensors own distinct storage.
    """

    import torch

    q = _float32_cpu_tensor(query, "query", ndim=4)
    k = _float32_cpu_tensor(key, "key", ndim=4)
    v = _float32_cpu_tensor(value, "value", ndim=4)
    decay = _float32_cpu_tensor(log_decay, "log_decay", ndim=3)
    step = _float32_cpu_tensor(beta, "beta", ndim=3)
    state = _float32_cpu_tensor(initial_state, "initial_state", ndim=4)
    if q.shape != k.shape:
        raise ValueError("query and key must have identical shapes")
    batch, sequence, heads, key_width = q.shape
    if sequence != 1 or min(batch, heads, key_width) <= 0:
        raise ValueError("query/key must have shape [B,1,H,K] with non-empty axes")
    if v.shape[:3] != (batch, 1, heads) or v.shape[-1] <= 0:
        raise ValueError("value must have shape [B,1,H,V]")
    value_width = v.shape[-1]
    if decay.shape != (batch, 1, heads) or step.shape != decay.shape:
        raise ValueError("log_decay and beta must have shape [B,1,H]")
    if state.shape != (batch, heads, key_width, value_width):
        raise ValueError("initial_state must have shape [B,H,K,V]")
    if threads is None:
        threads = max(1, min(batch * heads, os.cpu_count() or 1))
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise ValueError("threads must be a positive integer")

    output = torch.empty(
        (batch, 1, heads, value_width),
        dtype=torch.float32,
        device="cpu",
    )
    new_state = torch.empty(
        (batch, heads, key_width, value_width),
        dtype=torch.float32,
        device="cpu",
    )
    native = _native_library()
    code = native.library.immer_deltanet_step_f32(
        native.pointer(q),
        native.pointer(k),
        native.pointer(v),
        native.pointer(decay),
        native.pointer(step),
        native.pointer(state),
        batch,
        heads,
        key_width,
        value_width,
        native.pointer(output),
        native.pointer(new_state),
        threads,
    )
    if code == 2:
        raise ValueError("DeltaNet inputs or recurrent result are non-finite")
    if code:
        raise DeltaNetNativeError(f"native DeltaNet step failed with code {code}")
    return output, new_state


__all__ = [
    "DELTANET_NATIVE_ABI",
    "DeltaNetNativeError",
    "deltanet_sequence_one",
]
