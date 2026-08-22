"""Canonical source fingerprints for reproducible DeepSeek-V4 execution."""

from __future__ import annotations

import hashlib
from importlib import metadata as importlib_metadata
from pathlib import Path
import platform
import stat
from typing import Iterable


_CORE_RUNTIME_SOURCE_FILES = (
    "__init__.py",
    "config.py",
    "encoding.py",
    "graft.py",
    "kernels.py",
    "model.py",
    "pager.py",
    "provenance.py",
    "quantization.py",
    "snapshot.py",
    "stateful.py",
)

_TENSOR_SOURCE_FILES = (
    "knowledge/__init__.py",
    "knowledge/_hf_source.py",
    "knowledge/streamer.py",
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


def runtime_source_manifest(
    *,
    extra_files: Iterable[str] = (),
    project_files: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Return ordered hashes for every source that can affect streamed execution.

    ``extra_files`` remains restricted to local DeepSeek-V4 Python modules.
    ``project_files`` is restricted to direct Python children of ``scripts/``
    and lets an execution proof bind its actual command-line orchestrator.
    Every path is inspected with ``lstat`` before it is opened; symlinks and
    other non-regular source objects fail closed.
    """

    root = Path(__file__).resolve().parent
    immer_root = root.parent.parent
    project_root = root.parents[3]
    local_names = tuple(dict.fromkeys((*_CORE_RUNTIME_SOURCE_FILES, *extra_files)))
    sources: list[tuple[str, Path]] = []
    for name in local_names:
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("runtime source names must be local Python filenames")
        sources.append((f"immer/runtimes/deepseek_v4/{name}", root / name))
    sources.extend(
        (f"immer/{name}", immer_root / name) for name in _TENSOR_SOURCE_FILES
    )
    for name in dict.fromkeys(project_files):
        candidate = Path(name)
        if (
            candidate.parts != ("scripts", candidate.name)
            or not candidate.name.endswith(".py")
        ):
            raise ValueError(
                "project runtime sources must be direct scripts/*.py files"
            )
        sources.append((candidate.as_posix(), project_root / candidate))

    result: list[dict[str, str]] = []
    for logical_path, path in sources:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise RuntimeError(
                f"cannot inspect DeepSeek-V4 runtime source: {path}"
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"DeepSeek-V4 runtime source is not regular: {path}")
        result.append(
            {
                "path": logical_path,
                "sha256": _sha256_file(path),
            }
        )
    return result


def runtime_dependency_versions() -> dict[str, str]:
    """Return the canonical interpreter and package versions used at runtime."""

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
