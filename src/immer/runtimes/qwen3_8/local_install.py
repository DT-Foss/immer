"""Fast structural inspection of a configured local Qwen3.8 causal runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePath
import stat
from typing import Any

from ..ooe.identity import canonical_json_bytes
from .bundle import QWEN38_BUNDLE_SCHEMA
from .config import OFFICIAL_REPO_ID, OFFICIAL_REVISION, Qwen38Config
from .q4 import Q4_BANK_SCHEMA


_MAX_METADATA_BYTES = 64 * 1024**2


class QwenLocalInstallError(ValueError):
    """A configured local runtime is incomplete or structurally inconsistent."""


def _pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in entries:
        if key in result:
            raise QwenLocalInstallError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _regular(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise QwenLocalInstallError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise QwenLocalInstallError(f"{label} must be a regular non-symlink file")
    return metadata


def _read_metadata(path: Path, label: str) -> bytes:
    metadata = _regular(path, label)
    if metadata.st_size > _MAX_METADATA_BYTES:
        raise QwenLocalInstallError(f"{label} exceeds 64 MiB")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise QwenLocalInstallError(f"cannot read {label}: {path}") from exc
    if len(payload) != metadata.st_size:
        raise QwenLocalInstallError(f"{label} returned a short read")
    return payload


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            _read_metadata(path, label).decode("utf-8"),
            object_pairs_hook=_pairs,
        )
    except QwenLocalInstallError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QwenLocalInstallError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise QwenLocalInstallError(f"{label} root must be an object")
    return value


def _document(path: Path, *, label: str, schema: str) -> Mapping[str, Any]:
    document = _json(path, label)
    body = document.get("body")
    if (
        set(document) != {"body", "schema", "sha256"}
        or document.get("schema") != schema
        or not isinstance(body, Mapping)
        or document.get("sha256")
        != hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    ):
        raise QwenLocalInstallError(f"{label} envelope is invalid")
    return body


def _sha256_file(path: Path, label: str) -> str:
    return hashlib.sha256(_read_metadata(path, label)).hexdigest()


def _safe_child(root: Path, name: object, label: str) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or PurePath(name).name != name
        or name in {".", ".."}
    ):
        raise QwenLocalInstallError(f"{label} file name is invalid")
    return root / name


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QwenLocalInstallError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class QwenLocalInstall:
    root: Path
    tokenizer_path: Path
    q4_root: Path
    checkpoint_bytes: int
    checkpoint_shards: int
    q4_payload_bytes: int
    q4_tensors: int

    def summary(self) -> str:
        return (
            f"{self.checkpoint_shards} shards / "
            f"{self.checkpoint_bytes / 1024**3:.2f} GiB source / "
            f"{self.q4_tensors} Q4/Q8 tensors / "
            f"{self.q4_payload_bytes / 1024**3:.2f} GiB packed"
        )


def inspect_local_qwen(
    root: str | os.PathLike[str],
    *,
    tokenizer_path: str | os.PathLike[str] | None = None,
    q4_root: str | os.PathLike[str] | None = None,
) -> QwenLocalInstall:
    """Validate local metadata and payload sizes without hashing model payloads."""

    model_root = Path(root).expanduser().absolute()
    if not model_root.is_dir():
        raise QwenLocalInstallError(f"Qwen root is not a directory: {model_root}")
    tokenizer = (
        model_root / "tokenizer.json"
        if tokenizer_path is None
        else Path(tokenizer_path).expanduser().absolute()
    )
    packed = (
        model_root / "causal" / "q4-base-v3-mtp"
        if q4_root is None
        else Path(q4_root).expanduser().absolute()
    )

    Qwen38Config.from_file(model_root / "config.json", require_official=True)
    tokenizer_document = _json(tokenizer, "Qwen tokenizer")
    if not isinstance(tokenizer_document.get("model"), Mapping):
        raise QwenLocalInstallError("Qwen tokenizer lacks its model object")

    bundle = _document(
        model_root / "bundle.json",
        label="Qwen causal bundle",
        schema=QWEN38_BUNDLE_SCHEMA,
    )
    logical_model = bundle.get("logical_model")
    if logical_model != {
        "repo_id": OFFICIAL_REPO_ID,
        "revision": OFFICIAL_REVISION,
    }:
        raise QwenLocalInstallError("Qwen causal bundle model pin is invalid")
    if bundle.get("checkpoint_complete") is not True:
        raise QwenLocalInstallError("Qwen causal bundle is not complete")
    if bundle.get("config_sha256") != _sha256_file(
        model_root / "config.json",
        "Qwen config",
    ):
        raise QwenLocalInstallError("Qwen config digest differs from the bundle")
    if bundle.get("index_sha256") != _sha256_file(
        model_root / "model.safetensors.index.json",
        "Qwen tensor index",
    ):
        raise QwenLocalInstallError("Qwen tensor-index digest differs from the bundle")
    shards = bundle.get("shards")
    if not isinstance(shards, list) or not shards:
        raise QwenLocalInstallError("Qwen causal bundle has no shards")
    checkpoint_bytes = 0
    shard_names: set[str] = set()
    for row in shards:
        if not isinstance(row, Mapping):
            raise QwenLocalInstallError("Qwen shard entry is invalid")
        path = _safe_child(model_root, row.get("file"), "Qwen shard")
        if path.name in shard_names:
            raise QwenLocalInstallError("Qwen shard is listed twice")
        shard_names.add(path.name)
        expected = _positive_int(row.get("size"), "Qwen shard size")
        if _regular(path, "Qwen shard").st_size != expected:
            raise QwenLocalInstallError(f"Qwen shard size differs: {path.name}")
        checkpoint_bytes += expected
    if bundle.get("checkpoint_bytes") != checkpoint_bytes:
        raise QwenLocalInstallError("Qwen checkpoint byte total is invalid")
    _regular(model_root / "causal" / "manifest.head.json", "causal graph head")
    _regular(model_root / "causal" / "manifest.jsonl", "causal graph manifest")

    q4 = _document(
        packed / "manifest.json",
        label="Qwen Q4/Q8 bank",
        schema=Q4_BANK_SCHEMA,
    )
    source = q4.get("source")
    if not isinstance(source, Mapping) or (
        source.get("repo_id"),
        source.get("revision"),
    ) != (OFFICIAL_REPO_ID, OFFICIAL_REVISION):
        raise QwenLocalInstallError("Qwen Q4/Q8 source pin is invalid")
    tensors = q4.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise QwenLocalInstallError("Qwen Q4/Q8 bank has no tensors")
    q4_payload_bytes = 0
    tensor_files: set[str] = set()
    for row in tensors:
        if not isinstance(row, Mapping):
            raise QwenLocalInstallError("Qwen Q4/Q8 tensor entry is invalid")
        path = _safe_child(
            packed / "weights",
            row.get("file"),
            "Qwen Q4/Q8 tensor",
        )
        if path.name in tensor_files:
            raise QwenLocalInstallError("Qwen Q4/Q8 payload is listed twice")
        tensor_files.add(path.name)
        expected = _positive_int(row.get("payload_bytes"), "Q4/Q8 payload size")
        if _regular(path, "Qwen Q4/Q8 payload").st_size != expected:
            raise QwenLocalInstallError(f"Q4/Q8 payload size differs: {path.name}")
        q4_payload_bytes += expected
    if q4.get("tensor_count") != len(tensors):
        raise QwenLocalInstallError("Qwen Q4/Q8 tensor count is invalid")
    if q4.get("payload_bytes") != q4_payload_bytes:
        raise QwenLocalInstallError("Qwen Q4/Q8 byte total is invalid")

    return QwenLocalInstall(
        root=model_root,
        tokenizer_path=tokenizer,
        q4_root=packed,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_shards=len(shards),
        q4_payload_bytes=q4_payload_bytes,
        q4_tensors=len(tensors),
    )


__all__ = [
    "QwenLocalInstall",
    "QwenLocalInstallError",
    "inspect_local_qwen",
]
