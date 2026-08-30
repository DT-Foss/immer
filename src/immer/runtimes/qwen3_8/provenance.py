"""Canonical source and dependency identity for stateful Qwen3.8 execution."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
from pathlib import Path
import platform
import stat


_QWEN_RUNTIME_FILES = (
    "__init__.py",
    "config.py",
    "deltanet_native.c",
    "deltanet_native.py",
    "encoding.py",
    "graft.py",
    "kernels.py",
    "model.py",
    "mlp_page_markov.py",
    "native_crsa.py",
    "native_fork.py",
    "pager.py",
    "provenance.py",
    "q4.py",
    "q4_delta_router.py",
    "q4_fast_mlp.py",
    "q4_native.c",
    "snapshot.py",
)

_TENSOR_SOURCE_FILES = (
    "knowledge/__init__.py",
    "knowledge/_hf_source.py",
    "knowledge/streamer.py",
)

_SHARED_RUNTIME_FILES = (
    ("immer/attention/crsa/operators.py", "attention/crsa/operators.py"),
    ("immer/runtimes/deepseek_v4/graft.py", "runtimes/deepseek_v4/graft.py"),
    (
        "immer/runtimes/deepseek_v4/stable_graft.py",
        "runtimes/deepseek_v4/stable_graft.py",
    ),
    (
        "immer/runtimes/deepseek_v4/snapshot.py",
        "runtimes/deepseek_v4/snapshot.py",
    ),
)

_SHARED_TRANSPORT_FILES = (
    (
        "immer/runtimes/deepseek_v4/causal_weights.py",
        "runtimes/deepseek_v4/causal_weights.py",
    ),
)

_RUNTIME_DISTRIBUTIONS = (
    "torch",
    "numpy",
    "requests",
    "safetensors",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_source_manifest(*, include_transport: bool = True) -> list[dict[str, str]]:
    """Hash every Python source that can change Qwen continuation values."""

    if not isinstance(include_transport, bool):
        raise TypeError("include_transport must be a boolean")

    root = Path(__file__).resolve().parent
    immer_root = root.parent.parent
    sources: list[tuple[str, Path]] = [
        (f"immer/runtimes/qwen3_8/{name}", root / name) for name in _QWEN_RUNTIME_FILES
    ]
    sources.extend(
        (logical_path, immer_root / relative_path)
        for logical_path, relative_path in _SHARED_RUNTIME_FILES
    )
    if include_transport:
        sources.extend(
            (logical_path, immer_root / relative_path)
            for logical_path, relative_path in _SHARED_TRANSPORT_FILES
        )
        sources.extend(
            (f"immer/{name}", immer_root / name) for name in _TENSOR_SOURCE_FILES
        )

    result: list[dict[str, str]] = []
    for logical_path, path in sources:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise RuntimeError(f"cannot inspect Qwen runtime source: {path}") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"Qwen runtime source is not regular: {path}")
        result.append({"path": logical_path, "sha256": _sha256_file(path)})
    return result


def runtime_dependency_versions() -> dict[str, str]:
    """Return the interpreter and package versions bound by a continuation."""

    versions = {"python": platform.python_version()}
    for distribution in _RUNTIME_DISTRIBUTIONS:
        try:
            value = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"required runtime dependency metadata is missing: {distribution}"
            ) from exc
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(f"runtime dependency version is invalid: {distribution}")
        versions[distribution] = value
    if not versions["python"]:
        raise RuntimeError("Python runtime version is unavailable")
    return versions


__all__ = ["runtime_dependency_versions", "runtime_source_manifest"]
