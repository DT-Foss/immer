#!/usr/bin/env python3
"""Build and verify a complete local Qwen3.8 ``weights/ + causal/`` bundle.

Every official shard is size- and SHA-256-verified before any tensor binding is
published. The builder copies into a resumable staging directory, mounts the
copied layout under the pinned inventory fingerprint, appends all tensor plans
to LiveCausal, verifies the finished artifact, and promotes it atomically.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - production targets are macOS/Linux.
    fcntl = None  # type: ignore[assignment]

from immer.runtimes.qwen3_8 import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    CausalWeightMount,
    LogicalModelIdentity,
    Qwen38Config,
    tensor_range_plan_from_source,
)


ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_INVENTORY_FINGERPRINT = (
    "ad46a48f49c75b301f029a1d48ad9171e157629d9245ffb5c893724a3265879f"
)
DEFAULT_INVENTORY = (
    ROOT
    / "artifacts"
    / "private"
    / "qwen3.8-cache"
    / "inventories"
    / "Qwen-Qwen3-8-27B-5a5baa00ed547aaa.json"
)
BUNDLE_SCHEMA = "immer.qwen3.8-complete-causal-bundle/v1"
BUNDLE_HEADROOM_BYTES = 64 * 1024**2
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class QwenCausalBundleError(RuntimeError):
    """A complete Qwen causal bundle cannot be authenticated or published."""


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
        raise QwenCausalBundleError("value is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> Any:
    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in entries:
            if key in result:
                raise QwenCausalBundleError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except QwenCausalBundleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QwenCausalBundleError(f"cannot read JSON: {path}") from exc


def _strict_json_bytes(value: bytes, label: str) -> Any:
    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, child in entries:
            if key in result:
                raise QwenCausalBundleError(f"duplicate JSON key: {key!r}")
            result[key] = child
        return result

    try:
        return json.loads(value, object_pairs_hook=pairs)
    except QwenCausalBundleError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QwenCausalBundleError(f"cannot decode {label}") from exc


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".pending", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_new_bytes(path: Path, value: bytes) -> None:
    """Publish a new regular file without replacing any late-created target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".pending", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise QwenCausalBundleError(
                f"bundle manifest appeared during adoption: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _regular_file(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise QwenCausalBundleError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise QwenCausalBundleError(f"{label} must be a non-symlink regular file")
    return metadata


def _read_regular_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise QwenCausalBundleError(f"cannot open {label}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        linked = _regular_file(path, label)
        if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
            raise QwenCausalBundleError(f"{label} changed while opening")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024**2):
            chunks.append(chunk)
        if os.fstat(descriptor).st_size != opened.st_size:
            raise QwenCausalBundleError(f"{label} changed while reading")
        value = b"".join(chunks)
        if len(value) != opened.st_size:
            raise QwenCausalBundleError(f"{label} returned a short read")
        return value
    finally:
        os.close(descriptor)


@contextmanager
def _bundle_parent_lock(parent: Path):
    if fcntl is None:  # pragma: no cover
        raise QwenCausalBundleError("bundle promotion requires POSIX flock")
    lock_path = parent / ".qwen38-causal-bundle.lock"
    flags = os.O_RDWR | os.O_CREAT | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise QwenCausalBundleError("cannot open bundle parent lock") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise QwenCausalBundleError("bundle parent lock is not regular")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        linked = lock_path.lstat()
        if not stat.S_ISREG(linked.st_mode) or (linked.st_dev, linked.st_ino) != (
            opened.st_dev,
            opened.st_ino,
        ):
            raise QwenCausalBundleError("bundle parent lock changed while acquiring")
        yield
    except OSError as exc:
        raise QwenCausalBundleError("bundle parent lock failed") from exc
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _rename_no_replace(source: Path, target: Path) -> None:
    """Atomically promote one directory and fail if the target exists."""

    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(source_bytes, target_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, target_bytes, 0x00000001)
    else:  # pragma: no cover - fail closed on unsupported platforms.
        raise QwenCausalBundleError("atomic no-replace rename is unavailable")
    if result != 0:
        error = ctypes.get_errno()
        raise QwenCausalBundleError(
            f"bundle promotion refused target: {os.strerror(error)}"
        )


def _plain_directory(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise QwenCausalBundleError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise QwenCausalBundleError(f"{label} must be a non-symlink directory")


def _load_inventory(
    path: Path,
    *,
    repo_id: str,
    revision: str,
    expected_fingerprint: str | None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    document = _strict_json(path)
    if (
        not isinstance(document, dict)
        or document.get("schema") != "immer.tensor-inventory-cache/v1"
        or document.get("repo_id") != repo_id
        or document.get("revision") != revision
    ):
        raise QwenCausalBundleError("pinned inventory identity is invalid")
    inventory = document.get("inventory")
    fingerprint = document.get("source_fingerprint")
    if not isinstance(inventory, dict) or not isinstance(fingerprint, str):
        raise QwenCausalBundleError("pinned inventory payload is invalid")
    if document.get("inventory_sha256") != _sha256_bytes(_canonical(inventory)):
        raise QwenCausalBundleError("pinned inventory SHA-256 mismatch")
    if expected_fingerprint is not None and fingerprint != expected_fingerprint:
        raise QwenCausalBundleError("pinned inventory fingerprint is not official")
    if inventory.get("repo") != repo_id or inventory.get("revision") != revision:
        raise QwenCausalBundleError("inventory logical model identity is invalid")
    return inventory, fingerprint, document


def _expected_shard_digest(shard: Mapping[str, Any]) -> str | None:
    values: set[str] = set()
    for raw in (shard.get("etag"), shard.get("cas_url_hash")):
        if raw is None:
            continue
        value = str(raw).strip().strip('"').lower()
        if _SHA256.fullmatch(value):
            values.add(value)
    if len(values) > 1:
        raise QwenCausalBundleError("shard ETag and CAS digest disagree")
    return next(iter(values), None)


def _copy_verified_shard(
    source: Path,
    target: Path,
    *,
    expected_size: int,
    expected_sha256: str | None,
    resume: bool,
) -> dict[str, Any]:
    metadata = _regular_file(source, "checkpoint shard")
    if metadata.st_size != expected_size:
        raise QwenCausalBundleError(f"checkpoint shard size mismatch: {source.name}")
    source_digest = expected_sha256
    if source_digest is None:
        source_digest = _sha256_file(source)
    if target.exists() or target.is_symlink():
        if not resume:
            raise QwenCausalBundleError(
                f"staged shard already exists; use --resume: {target.name}"
            )
        target_metadata = _regular_file(target, "staged checkpoint shard")
        if (
            target_metadata.st_size == expected_size
            and _sha256_file(target) == source_digest
        ):
            return {
                "file": target.name,
                "reused": True,
                "sha256": source_digest,
                "size": expected_size,
            }
        raise QwenCausalBundleError(f"staged shard is corrupt: {target.name}")

    partial = target.with_name(f".{target.name}.partial")
    if partial.exists() or partial.is_symlink():
        if not resume:
            raise QwenCausalBundleError(
                f"partial shard exists; use --resume: {partial.name}"
            )
        partial.unlink()
    digest = hashlib.sha256()
    copied = 0
    with source.open("rb") as reader, partial.open("xb") as writer:
        while chunk := reader.read(4 * 1024**2):
            writer.write(chunk)
            digest.update(chunk)
            copied += len(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    actual = digest.hexdigest()
    if copied != expected_size or actual != source_digest:
        partial.unlink(missing_ok=True)
        raise QwenCausalBundleError(f"checkpoint shard SHA-256 mismatch: {source.name}")
    os.replace(partial, target)
    return {
        "file": target.name,
        "reused": False,
        "sha256": actual,
        "size": expected_size,
    }


def _validate_index_bytes(
    value: bytes,
    inventory: Mapping[str, Any],
) -> bytes:
    document = _strict_json_bytes(value, "checkpoint index")
    if not isinstance(document, Mapping):
        raise QwenCausalBundleError("checkpoint index root is invalid")
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise QwenCausalBundleError("checkpoint index weight_map is invalid")
    expected = {
        str(row["name"]): str(row["shard"]) for row in inventory.get("tensors", ())
    }
    if dict(weight_map) != expected:
        raise QwenCausalBundleError("checkpoint index disagrees with pinned inventory")
    return value


def _validate_config_bytes(value: bytes, *, require_official: bool) -> bytes:
    document = _strict_json_bytes(value, "checkpoint config")
    if not isinstance(document, Mapping):
        raise QwenCausalBundleError("checkpoint config root is invalid")
    try:
        Qwen38Config.from_mapping(document, require_official=require_official)
    except (TypeError, ValueError) as exc:
        raise QwenCausalBundleError("checkpoint config is not executable") from exc
    return value


def _copy_metadata(
    source: Path,
    weights: Path,
    inventory: Mapping[str, Any],
    *,
    require_official: bool,
) -> dict[str, str | None]:
    config, index, receipt = _validated_metadata(
        source,
        inventory,
        require_official=require_official,
    )
    _atomic_bytes(weights / "config.json", config)
    if index is not None:
        _atomic_bytes(weights / "model.safetensors.index.json", index)
    return receipt


def _validated_metadata(
    source: Path,
    inventory: Mapping[str, Any],
    *,
    require_official: bool,
) -> tuple[bytes, bytes | None, dict[str, str | None]]:
    config_path = source / "config.json"
    config = _validate_config_bytes(
        _read_regular_bytes(config_path, "checkpoint config"),
        require_official=require_official,
    )

    index_sha: str | None = None
    index: bytes | None = None
    index_path = source / "model.safetensors.index.json"
    if len(inventory.get("shards", ())) > 1 or index_path.exists():
        index = _validate_index_bytes(
            _read_regular_bytes(index_path, "checkpoint index"), inventory
        )
        index_sha = _sha256_bytes(index)
    receipt = {
        "config_sha256": _sha256_bytes(config),
        "index_sha256": index_sha,
    }
    return config, index, receipt


def _manifest(path: Path) -> dict[str, Any]:
    document = _strict_json(path)
    if not isinstance(document, dict) or set(document) != {
        "body",
        "schema",
        "sha256",
    }:
        raise QwenCausalBundleError("bundle manifest schema is invalid")
    if document.get("schema") != BUNDLE_SCHEMA:
        raise QwenCausalBundleError("bundle manifest type is invalid")
    body = document.get("body")
    if not isinstance(body, dict) or document.get("sha256") != _sha256_bytes(
        _canonical(body)
    ):
        raise QwenCausalBundleError("bundle manifest SHA-256 mismatch")
    return document


def verify_bundle(
    bundle: Path,
    *,
    require_remote_hashes: bool = True,
    expected_repo_id: str | None = OFFICIAL_REPO_ID,
    expected_revision: str | None = OFFICIAL_REVISION,
    expected_fingerprint: str | None = OFFICIAL_INVENTORY_FINGERPRINT,
    require_official: bool = True,
) -> dict[str, Any]:
    root = bundle.expanduser().absolute()
    _plain_directory(root, "bundle root")
    _plain_directory(root / "weights", "bundle weights")
    _plain_directory(root / "causal", "bundle graph")
    document = _manifest(root / "bundle.json")
    body = document["body"]
    model = body.get("logical_model")
    if not isinstance(model, Mapping) or set(model) != {"repo_id", "revision"}:
        raise QwenCausalBundleError("bundle logical model identity is invalid")
    if expected_repo_id is not None and model.get("repo_id") != expected_repo_id:
        raise QwenCausalBundleError("bundle repository identity is not expected")
    if expected_revision is not None and model.get("revision") != expected_revision:
        raise QwenCausalBundleError("bundle revision identity is not expected")
    if (
        expected_fingerprint is not None
        and body.get("layout_fingerprint") != expected_fingerprint
    ):
        raise QwenCausalBundleError("bundle layout fingerprint is not expected")
    inventory, fingerprint, _pinned = _load_inventory(
        root / "weights" / "inventory.pinned.json",
        repo_id=str(model["repo_id"]),
        revision=str(model["revision"]),
        expected_fingerprint=str(body.get("layout_fingerprint")),
    )
    if body.get("inventory_sha256") != _pinned.get("inventory_sha256"):
        raise QwenCausalBundleError("bundle inventory receipt is invalid")
    config_path = root / "weights" / "config.json"
    config = _validate_config_bytes(
        _read_regular_bytes(config_path, "bundle config"),
        require_official=require_official,
    )
    if _sha256_bytes(config) != body.get("config_sha256"):
        raise QwenCausalBundleError("bundle config SHA-256 mismatch")
    index_path = root / "weights" / "model.safetensors.index.json"
    if body.get("index_sha256") is None:
        if index_path.exists() or index_path.is_symlink():
            raise QwenCausalBundleError("unexpected bundle checkpoint index")
    else:
        index = _validate_index_bytes(
            _read_regular_bytes(index_path, "bundle checkpoint index"), inventory
        )
        if _sha256_bytes(index) != body.get("index_sha256"):
            raise QwenCausalBundleError("bundle index SHA-256 mismatch")
    receipts = body.get("shards")
    if not isinstance(receipts, list) or len(receipts) != len(
        inventory.get("shards", ())
    ):
        raise QwenCausalBundleError("bundle shard receipts are incomplete")
    by_name = {
        str(row.get("file")): row for row in receipts if isinstance(row, Mapping)
    }
    total = 0
    for shard in inventory.get("shards", ()):
        name = str(shard["file"])
        receipt = by_name.get(name)
        path = root / "weights" / name
        metadata = _regular_file(path, "bundle shard")
        expected_size = int(shard["size"])
        expected_digest = _expected_shard_digest(shard)
        if require_remote_hashes and expected_digest is None:
            raise QwenCausalBundleError(f"shard lacks remote SHA-256: {name}")
        digest = _sha256_file(path)
        if (
            receipt is None
            or metadata.st_size != expected_size
            or receipt.get("size") != expected_size
            or receipt.get("sha256") != digest
            or (expected_digest is not None and digest != expected_digest)
        ):
            raise QwenCausalBundleError(f"bundle shard verification failed: {name}")
        total += expected_size
    if total != body.get("checkpoint_bytes") or not body.get("checkpoint_complete"):
        raise QwenCausalBundleError("bundle checkpoint completeness receipt is invalid")

    identity = LogicalModelIdentity(str(model["repo_id"]), str(model["revision"]))
    with CausalWeightMount(root, identity, budget_mb=64) as mount:
        if mount.layout.layout_fingerprint != fingerprint:
            raise QwenCausalBundleError("mounted bundle layout fingerprint changed")
        plans = tuple(
            tensor_range_plan_from_source(mount.source, str(row["name"]))
            for row in inventory.get("tensors", ())
        )
        if len(plans) != body.get("tensor_bindings"):
            raise QwenCausalBundleError("bundle tensor binding count is invalid")
        for plan in plans:
            if mount.resolve_tensor_plan(plan.name) != plan:
                raise QwenCausalBundleError(
                    f"bundle tensor binding changed: {plan.name}"
                )
        revision = mount.graph.store.revision()
    if [revision[0], revision[1]] != body.get("graph_revision"):
        raise QwenCausalBundleError("bundle graph revision receipt changed")
    return {
        "bundle": str(root),
        "checkpoint_bytes": total,
        "layout_fingerprint": fingerprint,
        "sha256": document["sha256"],
        "tensor_bindings": len(plans),
    }


def _build_bundle_locked(
    source: Path,
    inventory_path: Path,
    output: Path,
    *,
    repo_id: str = OFFICIAL_REPO_ID,
    revision: str = OFFICIAL_REVISION,
    expected_fingerprint: str | None = OFFICIAL_INVENTORY_FINGERPRINT,
    require_official: bool = True,
    require_remote_hashes: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    source_root = source.expanduser().absolute()
    target = output.expanduser().absolute()
    staging = target.with_name(f".{target.name}.building")
    _plain_directory(source_root, "checkpoint source")
    if target.exists() or target.is_symlink():
        raise QwenCausalBundleError(f"bundle output already exists: {target}")
    inventory, fingerprint, pinned = _load_inventory(
        inventory_path.expanduser().absolute(),
        repo_id=repo_id,
        revision=revision,
        expected_fingerprint=expected_fingerprint,
    )
    checkpoint_bytes = sum(int(row["size"]) for row in inventory.get("shards", ()))
    target.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists() or staging.is_symlink():
        if not resume:
            raise QwenCausalBundleError(
                f"staging directory exists; use --resume: {staging}"
            )
        _plain_directory(staging, "bundle staging root")
    else:
        (staging / "weights").mkdir(parents=True)
        (staging / "causal").mkdir()
    weights = staging / "weights"
    _plain_directory(weights, "bundle staging weights")
    _plain_directory(staging / "causal", "bundle staging graph")
    remaining_bytes = sum(
        int(shard["size"])
        for shard in inventory.get("shards", ())
        if not (weights / str(shard["file"])).is_file()
    )
    free = shutil.disk_usage(target.parent).free
    if free < remaining_bytes + BUNDLE_HEADROOM_BYTES:
        raise QwenCausalBundleError(
            f"bundle needs {remaining_bytes + BUNDLE_HEADROOM_BYTES} "
            "additional free bytes, "
            f"found {free}"
        )

    shard_receipts: list[dict[str, Any]] = []
    for shard in inventory.get("shards", ()):
        name = str(shard["file"])
        expected_digest = _expected_shard_digest(shard)
        if require_remote_hashes and expected_digest is None:
            raise QwenCausalBundleError(f"shard lacks remote SHA-256: {name}")
        shard_receipts.append(
            _copy_verified_shard(
                source_root / name,
                weights / name,
                expected_size=int(shard["size"]),
                expected_sha256=expected_digest,
                resume=resume,
            )
        )

    metadata = _copy_metadata(
        source_root,
        weights,
        inventory,
        require_official=require_official,
    )
    _atomic_bytes(
        weights / "inventory.pinned.json",
        _canonical(pinned) + b"\n",
    )
    identity = LogicalModelIdentity(repo_id, revision)
    with CausalWeightMount(staging, identity, budget_mb=64) as mount:
        if mount.layout.layout_fingerprint != fingerprint:
            raise QwenCausalBundleError("copied layout fingerprint changed")
        plans = tuple(
            tensor_range_plan_from_source(mount.source, str(row["name"]))
            for row in inventory.get("tensors", ())
        )
        receipt = mount.bind_tensor_plans(plans)
        if receipt.appended_count not in (0, len(plans)):
            # A resumed graph may contain all bindings or a prefix published by
            # one earlier atomic segment; mixed coverage is not promoted.
            raise QwenCausalBundleError("staged graph has partial tensor bindings")
        for plan in plans:
            if mount.resolve_tensor_plan(plan.name) != plan:
                raise QwenCausalBundleError(f"tensor binding mismatch: {plan.name}")
        graph_revision = mount.graph.store.revision()

    body = {
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_complete": True,
        "config_sha256": metadata["config_sha256"],
        "graph_revision": [graph_revision[0], graph_revision[1]],
        "index_sha256": metadata["index_sha256"],
        "inventory_sha256": pinned["inventory_sha256"],
        "layout_fingerprint": fingerprint,
        "logical_model": {"repo_id": repo_id, "revision": revision},
        "shards": sorted(shard_receipts, key=lambda row: str(row["file"])),
        "tensor_bindings": len(plans),
    }
    document = {
        "body": body,
        "schema": BUNDLE_SCHEMA,
        "sha256": _sha256_bytes(_canonical(body)),
    }
    _atomic_bytes(staging / "bundle.json", _canonical(document) + b"\n")
    verified = verify_bundle(
        staging,
        require_remote_hashes=require_remote_hashes,
        expected_repo_id=repo_id,
        expected_revision=revision,
        expected_fingerprint=fingerprint,
        require_official=require_official,
    )
    _rename_no_replace(staging, target)
    return {**verified, "bundle": str(target), "resumed": resume}


def build_bundle(
    source: Path,
    inventory_path: Path,
    output: Path,
    *,
    repo_id: str = OFFICIAL_REPO_ID,
    revision: str = OFFICIAL_REVISION,
    expected_fingerprint: str | None = OFFICIAL_INVENTORY_FINGERPRINT,
    require_official: bool = True,
    require_remote_hashes: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    target = output.expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    _plain_directory(target.parent, "bundle output parent")
    with _bundle_parent_lock(target.parent):
        return _build_bundle_locked(
            source,
            inventory_path,
            output,
            repo_id=repo_id,
            revision=revision,
            expected_fingerprint=expected_fingerprint,
            require_official=require_official,
            require_remote_hashes=require_remote_hashes,
            resume=resume,
        )


def adopt_bundle(
    bundle: Path,
    inventory_path: Path,
    *,
    repo_id: str = OFFICIAL_REPO_ID,
    revision: str = OFFICIAL_REVISION,
    expected_fingerprint: str | None = OFFICIAL_INVENTORY_FINGERPRINT,
    require_official: bool = True,
    require_remote_hashes: bool = True,
) -> dict[str, Any]:
    """Causalize a complete ``bundle/weights`` tree without copying shards."""

    root = bundle.expanduser().absolute()
    parent = root.parent
    _plain_directory(parent, "bundle parent")
    with _bundle_parent_lock(parent):
        _plain_directory(root, "bundle root")
        weights = root / "weights"
        _plain_directory(weights, "bundle weights")
        causal = root / "causal"
        if causal.exists() or causal.is_symlink():
            _plain_directory(causal, "bundle graph")
        else:
            causal.mkdir()
        manifest_path = root / "bundle.json"
        if manifest_path.exists() or manifest_path.is_symlink():
            verified = verify_bundle(
                root,
                require_remote_hashes=require_remote_hashes,
                expected_repo_id=repo_id,
                expected_revision=revision,
                expected_fingerprint=expected_fingerprint,
                require_official=require_official,
            )
            return {**verified, "adopted": True, "resumed": True}

        inventory, fingerprint, pinned = _load_inventory(
            inventory_path.expanduser().absolute(),
            repo_id=repo_id,
            revision=revision,
            expected_fingerprint=expected_fingerprint,
        )
        shard_receipts: list[dict[str, Any]] = []
        checkpoint_bytes = 0
        for shard in inventory.get("shards", ()):
            name = str(shard["file"])
            path = weights / name
            metadata = _regular_file(path, "adopted checkpoint shard")
            expected_size = int(shard["size"])
            expected_digest = _expected_shard_digest(shard)
            if require_remote_hashes and expected_digest is None:
                raise QwenCausalBundleError(f"shard lacks remote SHA-256: {name}")
            digest = _sha256_file(path)
            if metadata.st_size != expected_size or (
                expected_digest is not None and digest != expected_digest
            ):
                raise QwenCausalBundleError(
                    f"adopted checkpoint shard verification failed: {name}"
                )
            shard_receipts.append(
                {
                    "adopted": True,
                    "file": name,
                    "reused": True,
                    "sha256": digest,
                    "size": expected_size,
                }
            )
            checkpoint_bytes += expected_size

        _config, _index, metadata = _validated_metadata(
            weights,
            inventory,
            require_official=require_official,
        )
        _atomic_bytes(
            weights / "inventory.pinned.json",
            _canonical(pinned) + b"\n",
        )
        identity = LogicalModelIdentity(repo_id, revision)
        with CausalWeightMount(root, identity, budget_mb=64) as mount:
            if mount.layout.layout_fingerprint != fingerprint:
                raise QwenCausalBundleError("adopted layout fingerprint changed")
            plans = tuple(
                tensor_range_plan_from_source(mount.source, str(row["name"]))
                for row in inventory.get("tensors", ())
            )
            receipt = mount.bind_tensor_plans(plans)
            if receipt.appended_count not in (0, len(plans)):
                raise QwenCausalBundleError("adopted graph has partial tensor bindings")
            for plan in plans:
                if mount.resolve_tensor_plan(plan.name) != plan:
                    raise QwenCausalBundleError(
                        f"adopted tensor binding mismatch: {plan.name}"
                    )
            graph_revision = mount.graph.store.revision()

        body = {
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_complete": True,
            "config_sha256": metadata["config_sha256"],
            "graph_revision": [graph_revision[0], graph_revision[1]],
            "index_sha256": metadata["index_sha256"],
            "inventory_sha256": pinned["inventory_sha256"],
            "layout_fingerprint": fingerprint,
            "logical_model": {"repo_id": repo_id, "revision": revision},
            "shards": sorted(shard_receipts, key=lambda row: str(row["file"])),
            "tensor_bindings": len(plans),
        }
        document = {
            "body": body,
            "schema": BUNDLE_SCHEMA,
            "sha256": _sha256_bytes(_canonical(body)),
        }
        _atomic_new_bytes(manifest_path, _canonical(document) + b"\n")
        verified = verify_bundle(
            root,
            require_remote_hashes=require_remote_hashes,
            expected_repo_id=repo_id,
            expected_revision=revision,
            expected_fingerprint=fingerprint,
            require_official=require_official,
        )
        return {**verified, "adopted": True, "resumed": False}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source", required=True)
    build.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    build.add_argument("--output", required=True)
    build.add_argument("--resume", action="store_true")
    adopt = subparsers.add_parser("adopt")
    adopt.add_argument("--bundle", required=True)
    adopt.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            result = build_bundle(
                Path(args.source),
                Path(args.inventory),
                Path(args.output),
                resume=args.resume,
            )
        elif args.command == "adopt":
            result = adopt_bundle(Path(args.bundle), Path(args.inventory))
        else:
            result = verify_bundle(Path(args.bundle))
    except QwenCausalBundleError as exc:
        raise SystemExit(f"Qwen causal bundle failed: {exc}") from exc
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
