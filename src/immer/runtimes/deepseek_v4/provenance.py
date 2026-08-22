"""Canonical source fingerprints for reproducible DeepSeek-V4 execution."""

from __future__ import annotations

import hashlib
from pathlib import Path
import stat
from typing import Iterable


_CORE_RUNTIME_SOURCE_FILES = (
    "config.py",
    "encoding.py",
    "graft.py",
    "kernels.py",
    "model.py",
    "pager.py",
    "provenance.py",
    "quantization.py",
    "stateful.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_source_manifest(
    *, extra_files: Iterable[str] = ()
) -> list[dict[str, str]]:
    """Return ordered hashes for core runtime sources plus named local extras."""

    root = Path(__file__).resolve().parent
    names = tuple(dict.fromkeys((*_CORE_RUNTIME_SOURCE_FILES, *extra_files)))
    result: list[dict[str, str]] = []
    for name in names:
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("runtime source names must be local Python filenames")
        path = root / name
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise RuntimeError(f"cannot inspect DeepSeek-V4 runtime source: {path}") from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise RuntimeError(f"DeepSeek-V4 runtime source is not regular: {path}")
        result.append(
            {
                "path": f"immer/runtimes/deepseek_v4/{name}",
                "sha256": _sha256_file(path),
            }
        )
    return result


__all__ = ["runtime_source_manifest"]
