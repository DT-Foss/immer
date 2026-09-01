"""Authenticated, pickle-free continuation snapshots for Qwen3.8."""

from __future__ import annotations

from collections.abc import Mapping
import os
from typing import Any

from ..deepseek_v4.snapshot import (
    DeepSeekV4SnapshotError,
    DeepSeekV4SnapshotIdentityMismatch,
    LoadedSnapshot,
    SnapshotLimits,
    SnapshotTensor,
    read_snapshot,
    write_snapshot,
)


QWEN38_SNAPSHOT_SCHEMA = "immer.qwen3.8-continuation/v1"


class Qwen38SnapshotError(ValueError):
    """A Qwen continuation is corrupt, unsafe, or runtime-incompatible."""


class Qwen38SnapshotIdentityMismatch(Qwen38SnapshotError):
    """The snapshot is valid but belongs to a different Qwen runtime."""


def write_qwen38_snapshot(
    path: str | os.PathLike[str],
    *,
    identity: Mapping[str, Any],
    state: Mapping[str, Any],
    tensors: Mapping[str, SnapshotTensor],
    limits: SnapshotLimits | None = None,
) -> dict[str, Any]:
    try:
        return write_snapshot(
            path,
            identity=identity,
            state=state,
            tensors=tensors,
            limits=limits,
            schema=QWEN38_SNAPSHOT_SCHEMA,
        )
    except DeepSeekV4SnapshotError as exc:
        raise Qwen38SnapshotError(str(exc)) from exc


def read_qwen38_snapshot(
    path: str | os.PathLike[str],
    *,
    expected_identity: Mapping[str, Any],
    limits: SnapshotLimits | None = None,
    resident_bytes: int = 0,
    max_restore_peak_bytes: int | None = None,
) -> LoadedSnapshot:
    try:
        return read_snapshot(
            path,
            expected_identity=expected_identity,
            limits=limits,
            resident_bytes=resident_bytes,
            max_restore_peak_bytes=max_restore_peak_bytes,
            schema=QWEN38_SNAPSHOT_SCHEMA,
        )
    except DeepSeekV4SnapshotIdentityMismatch as exc:
        raise Qwen38SnapshotIdentityMismatch(str(exc)) from exc
    except DeepSeekV4SnapshotError as exc:
        raise Qwen38SnapshotError(str(exc)) from exc


__all__ = [
    "LoadedSnapshot",
    "QWEN38_SNAPSHOT_SCHEMA",
    "Qwen38SnapshotError",
    "Qwen38SnapshotIdentityMismatch",
    "SnapshotLimits",
    "SnapshotTensor",
    "read_qwen38_snapshot",
    "write_qwen38_snapshot",
]
