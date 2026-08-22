"""Stable locators for IMMER's packaged deployment manifests.

The JSON documents remain canonical in the repository-level ``manifests``
directory.  A wheel contains byte-for-byte copies under ``immer/resources``.
This module selects the canonical source in an editable checkout and the
packaged copy after installation.
"""

from __future__ import annotations

from pathlib import Path


S3_SHIP_V6 = "s3_ship_v6.json"
CRSA_ROUTER_V1 = "crsa_router_v1.json"
MANIFEST_NAMES = (S3_SHIP_V6, CRSA_ROUTER_V1)
_KNOWN_MANIFESTS = frozenset(MANIFEST_NAMES)


def _editable_manifest_dir() -> Path | None:
    package_dir = Path(__file__).resolve().parent
    repository = package_dir.parents[1]
    expected_package = repository / "src" / "immer"
    if (
        (repository / "pyproject.toml").is_file()
        and expected_package.is_dir()
        and expected_package.resolve() == package_dir
    ):
        return repository / "manifests"
    return None


def packaged_manifest_dir() -> Path:
    """Return the materialized manifest directory inside an installation."""

    return Path(__file__).resolve().parent / "resources"


def manifest_path(name: str) -> Path:
    """Locate one known deployment manifest or fail with checked locations."""

    if name not in _KNOWN_MANIFESTS:
        allowed = ", ".join(MANIFEST_NAMES)
        raise ValueError(f"unknown IMMER manifest {name!r}; expected one of: {allowed}")

    checked: list[Path] = []
    editable = _editable_manifest_dir()
    if editable is not None:
        candidate = editable / name
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"canonical IMMER manifest {name!r} is missing from editable checkout: "
            f"{candidate}"
        )

    packaged = packaged_manifest_dir() / name
    checked.append(packaged)
    if packaged.is_file():
        return packaged

    locations = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(f"IMMER manifest {name!r} not found; checked: {locations}")


def s3_ship_manifest() -> Path:
    """Locate the canonical SHIP-v6 organ/host manifest."""

    return manifest_path(S3_SHIP_V6)


def crsa_router_manifest() -> Path:
    """Locate the calibrated CRSA router state."""

    return manifest_path(CRSA_ROUTER_V1)


__all__ = [
    "CRSA_ROUTER_V1",
    "MANIFEST_NAMES",
    "S3_SHIP_V6",
    "crsa_router_manifest",
    "manifest_path",
    "packaged_manifest_dir",
    "s3_ship_manifest",
]
