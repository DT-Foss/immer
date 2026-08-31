#!/usr/bin/env python3
"""Inspect built IMMER distributions as the exact public release surface."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import zipfile


_SDIST_TOP_LEVEL = frozenset(
    {
        "CHANGELOG.md",
        "CITATION.cff",
        "MANIFEST.in",
        "MODEL_CARD.md",
        "NOTICE.md",
        "PKG-INFO",
        "README.md",
        "SECURITY.md",
        "deploy",
        "docs",
        "manifests",
        "pyproject.toml",
        "setup.cfg",
        "setup.py",
        "src",
    }
)
_FORBIDDEN_PARTS = frozenset(
    {
        ".git",
        "__pycache__",
        "artifacts",
        "evals",
        "hf-cache",
        "kandidaten",
        "research",
        "results",
        "scripts",
        "tests",
        "vendor",
    }
)
_FORBIDDEN_SUFFIXES = frozenset(
    {
        ".bin",
        ".causal",
        ".ckpt",
        ".env",
        ".jsonl",
        ".lock",
        ".log",
        ".npy",
        ".npz",
        ".parquet",
        ".pt",
        ".pth",
        ".safetensors",
        ".seg",
    }
)
_TEXT_SUFFIXES = frozenset(
    {
        "",
        ".c",
        ".cff",
        ".example",
        ".in",
        ".json",
        ".md",
        ".plist",
        ".py",
        ".service",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
_MACHINE_PATHS = (
    b"/" + b"Users" + b"/",
    b"/" + b"home" + b"/",
    b"/" + b"root" + b"/",
)
_CREDENTIALS = (
    b"BEGIN " + b"OPENSSH PRIVATE KEY",
    b"BEGIN " + b"RSA PRIVATE KEY",
    b"ghp_" + b"[A-Za-z0-9]",
)


class ReleasePreflightError(ValueError):
    """The built public surface contains a forbidden member or value."""


def _normalized_members(names: Iterable[str], *, sdist: bool) -> tuple[str, ...]:
    normalized: list[str] = []
    root: str | None = None
    for raw in names:
        name = raw.rstrip("/")
        if not name:
            continue
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ReleasePreflightError(f"archive path escapes its root: {raw}")
        parts = path.parts
        if sdist:
            if len(parts) < 2:
                continue
            if root is None:
                root = parts[0]
            if parts[0] != root:
                raise ReleasePreflightError("sdist has more than one archive root")
            parts = parts[1:]
        normalized.append(PurePosixPath(*parts).as_posix())
    return tuple(normalized)


def _check_name(name: str, *, sdist: bool) -> None:
    path = PurePosixPath(name)
    if any(part in _FORBIDDEN_PARTS for part in path.parts):
        raise ReleasePreflightError(f"private release member: {name}")
    if path.suffix.casefold() in _FORBIDDEN_SUFFIXES:
        raise ReleasePreflightError(f"payload release member: {name}")
    if sdist and path.parts and path.parts[0] not in _SDIST_TOP_LEVEL:
        raise ReleasePreflightError(f"sdist member is outside the allowlist: {name}")
    if not sdist and path.parts:
        top = path.parts[0]
        if top != "immer" and re.fullmatch(r"immer-1\.0\.0\.dist-info", top) is None:
            raise ReleasePreflightError(f"wheel member is outside the allowlist: {name}")


def _check_text(name: str, payload: bytes) -> None:
    if len(payload) > 16 * 1024**2:
        return
    if PurePosixPath(name).suffix.casefold() not in _TEXT_SUFFIXES:
        return
    for pattern in _MACHINE_PATHS:
        if pattern in payload:
            raise ReleasePreflightError(f"machine path in release member: {name}")
    for pattern in _CREDENTIALS:
        if pattern in payload:
            raise ReleasePreflightError(f"credential material in release member: {name}")


def inspect_distribution(path: str | Path) -> dict[str, object]:
    source = Path(path)
    if source.suffix == ".whl":
        with zipfile.ZipFile(source) as archive:
            raw_names = archive.namelist()
            names = _normalized_members(raw_names, sdist=False)
            for raw, name in zip(raw_names, names, strict=True):
                _check_name(name, sdist=False)
                if not raw.endswith("/"):
                    _check_text(name, archive.read(raw))
        kind = "wheel"
    elif source.name.endswith(".tar.gz"):
        with tarfile.open(source, "r:gz") as archive:
            members = [row for row in archive.getmembers() if row.isfile()]
            names = _normalized_members((row.name for row in members), sdist=True)
            if len(names) != len(members):
                raise ReleasePreflightError("sdist contains a root-level file")
            for member, name in zip(members, names, strict=True):
                _check_name(name, sdist=True)
                handle = archive.extractfile(member)
                if handle is None:
                    raise ReleasePreflightError(f"cannot read sdist member: {name}")
                _check_text(name, handle.read())
        kind = "sdist"
    else:
        raise ReleasePreflightError(f"unsupported distribution: {source.name}")
    if not names:
        raise ReleasePreflightError(f"empty distribution: {source.name}")
    return {"archive": source.name, "kind": kind, "members": len(names)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("distributions", nargs="+")
    args = parser.parse_args(argv)
    reports = [inspect_distribution(path) for path in args.distributions]
    print(json.dumps({"status": "clean", "distributions": reports}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
