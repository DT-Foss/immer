"""Safe bootstrap for the external artifacts named by an IMMER manifest.

The repository intentionally does not commit model blobs.  This module closes
the fresh-checkout gap without weakening that boundary: bytes are copied from
an explicitly supplied source directory, verified against the committed
SHA-256 values, and installed atomically.  The source is never opened for
writing and an existing invalid destination is never replaced implicitly.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ArtifactBootstrapError(RuntimeError):
    """The manifest, source, or destination violates the bootstrap contract."""


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    label: str
    destination: Path
    sha256: str


def cache_root() -> Path:
    """Return IMMER's writable cache root without touching the filesystem."""

    configured = os.environ.get("IMMER_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "immer").resolve()


def artifact_root(
    manifest: str | Path,
    configured: str | Path | None = None,
) -> Path | None:
    """Choose artifact storage.

    Editable checkouts retain their committed relative layout.  Installed
    wheels use an explicit ``IMMER_ARTIFACT_ROOT`` or the user's cache; package
    resources under site-packages are never treated as writable blob storage.
    """

    manifest_path = Path(manifest).expanduser().resolve()
    selected = configured or os.environ.get("IMMER_ARTIFACT_ROOT")
    if selected:
        return Path(selected).expanduser().resolve()
    repository = manifest_path.parent.parent
    if (
        manifest_path.parent.name == "manifests"
        and (repository / "pyproject.toml").is_file()
        and (repository / "src" / "immer").is_dir()
    ):
        return None
    return cache_root() / "ship-v6"


def resolve_artifact_path(
    manifest: str | Path,
    artifact: str | Path,
    *,
    configured_root: str | Path | None = None,
) -> Path:
    """Resolve one manifest artifact in checkout or installed-wheel mode."""

    manifest_path = Path(manifest).expanduser().resolve()
    raw = Path(artifact)
    root = artifact_root(manifest_path, configured_root)
    if root is not None:
        if raw.is_absolute():
            raise ArtifactBootstrapError(
                f"absolute artifact path cannot be rebased below artifact root: {raw}"
            )
        parts = tuple(part for part in raw.parts if part not in ("", "."))
        if ".." in parts:
            # The committed SHIP manifest uses repository-relative legacy paths.
            # In an installed wheel those two known prefixes map to the flat
            # cache contract; arbitrary traversal remains an error.
            legacy_prefix = parts[:4]
            if (
                len(parts) == 5
                and legacy_prefix[:3] == ("..", "vendor", "o1state")
                and legacy_prefix[3] in {"results", "s3_ship"}
            ):
                parts = (parts[-1],)
            else:
                raise ArtifactBootstrapError(
                    f"artifact path traverses outside configured root: {raw}"
                )
        if not parts:
            raise ArtifactBootstrapError("artifact path is empty")
        destination = (root / Path(*parts)).resolve()
        _inside(destination, root.resolve(), label=str(raw))
        return destination
    return raw.resolve() if raw.is_absolute() else (manifest_path.parent / raw).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_digest(value: object, *, label: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ArtifactBootstrapError(f"invalid SHA-256 for {label}")
    return digest


def _inside(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactBootstrapError(
            f"artifact destination for {label} leaves target root: {resolved}"
        ) from exc
    return resolved


def load_artifact_specs(
    manifest: str | Path,
    *,
    configured_root: str | Path | None = None,
) -> tuple[Path, tuple[ArtifactSpec, ...]]:
    """Load host + organ targets from one SHIP manifest."""

    manifest_path = Path(manifest).expanduser().resolve()
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArtifactBootstrapError(f"cannot read manifest {manifest_path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise ArtifactBootstrapError("artifact manifest must be a JSON object")
    external_root = artifact_root(manifest_path, configured_root)
    target_root = (
        manifest_path.parent.parent.resolve()
        if external_root is None
        else external_root.resolve()
    )

    records: list[tuple[str, Mapping[str, Any]]] = []
    s3 = document.get("s3")
    if isinstance(s3, Mapping) and isinstance(s3.get("host"), Mapping):
        host = s3["host"]
        records.append((f"host:{host.get('name', 'unnamed')}", host))
    organs = document.get("organs")
    if not isinstance(organs, list):
        raise ArtifactBootstrapError("artifact manifest has no organs list")
    for index, raw in enumerate(organs):
        if not isinstance(raw, Mapping):
            raise ArtifactBootstrapError(f"organ entry {index} is not an object")
        records.append((f"organ:{raw.get('name', index)}", raw))
    if not records:
        raise ArtifactBootstrapError("artifact manifest names no artifacts")

    specs: list[ArtifactSpec] = []
    destinations: set[Path] = set()
    for label, record in records:
        artifact = record.get("artifact")
        if not isinstance(artifact, str) or not artifact:
            raise ArtifactBootstrapError(f"missing artifact path for {label}")
        raw_path = Path(artifact)
        destination = (
            resolve_artifact_path(
                manifest_path,
                raw_path,
                configured_root=configured_root,
            )
            if external_root is not None
            else raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
        )
        destination = _inside(destination, target_root, label=label)
        if destination in destinations:
            raise ArtifactBootstrapError(f"duplicate artifact destination: {destination}")
        destinations.add(destination)
        specs.append(
            ArtifactSpec(
                label=label,
                destination=destination,
                sha256=_valid_digest(record.get("sha256"), label=label),
            )
        )
    return target_root, tuple(specs)


def _source_candidates(source_root: Path, target_root: Path, spec: ArtifactSpec) -> tuple[Path, ...]:
    relative = spec.destination.relative_to(target_root)
    name = spec.destination.name
    candidates = (
        source_root / relative,
        source_root / name,
        source_root / "results" / name,
        source_root / "s3_ship" / name,
        source_root / "vendor" / "o1state" / "results" / name,
        source_root / "vendor" / "o1state" / "s3_ship" / name,
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(source_root)
        except ValueError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return tuple(unique)


def _find_source(source_root: Path, target_root: Path, spec: ArtifactSpec) -> Path:
    mismatches: list[str] = []
    for candidate in _source_candidates(source_root, target_root, spec):
        if not candidate.is_file():
            continue
        actual = sha256_file(candidate)
        if actual == spec.sha256:
            return candidate
        mismatches.append(f"{candidate}={actual}")
    detail = f"; wrong candidates: {', '.join(mismatches)}" if mismatches else ""
    raise ArtifactBootstrapError(
        f"no SHA-matching source found for {spec.label} ({spec.destination.name}){detail}"
    )


def _atomic_copy(source: Path, destination: Path, expected: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary)
    try:
        with source.open("rb") as reader, os.fdopen(fd, "wb") as writer:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        actual = sha256_file(temporary_path)
        if actual != expected:
            raise ArtifactBootstrapError(
                f"copied bytes failed SHA-256: expected {expected}, got {actual}"
            )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def import_artifacts(
    manifest: str | Path,
    source: str | Path,
    *,
    dry_run: bool = False,
    replace: bool = False,
    configured_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Verify and atomically copy every artifact required by ``manifest``."""

    source_root = Path(source).expanduser().resolve()
    if not source_root.is_dir():
        raise ArtifactBootstrapError(f"artifact source is not a directory: {source_root}")
    target_root, specs = load_artifact_specs(manifest, configured_root=configured_root)
    actions: list[dict[str, str]] = []
    for spec in specs:
        existed = spec.destination.exists()
        if spec.destination.is_file():
            actual = sha256_file(spec.destination)
            if actual == spec.sha256:
                actions.append(
                    {"label": spec.label, "status": "verified", "path": str(spec.destination)}
                )
                continue
            if not replace:
                raise ArtifactBootstrapError(
                    f"existing destination has wrong SHA-256 and was not touched: "
                    f"{spec.destination}"
                )
        source_path = _find_source(source_root, target_root, spec)
        status = "would_replace" if spec.destination.exists() else "would_install"
        if not dry_run:
            _atomic_copy(source_path, spec.destination, spec.sha256)
            status = "replaced" if existed else "installed"
        actions.append(
            {
                "label": spec.label,
                "status": status,
                "source": str(source_path),
                "path": str(spec.destination),
            }
        )
    return {
        "schema": "immer.artifact-bootstrap/v1",
        "source": str(source_root),
        "target_root": str(target_root),
        "dry_run": dry_run,
        "artifacts": actions,
    }


__all__ = [
    "ArtifactBootstrapError",
    "ArtifactSpec",
    "artifact_root",
    "cache_root",
    "import_artifacts",
    "load_artifact_specs",
    "resolve_artifact_path",
    "sha256_file",
]
