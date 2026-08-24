"""Authenticated complete-bundle verification for Qwen3.8 runtimes."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any

from immer.knowledge import Streamer

from ..deepseek_v4.causal_weights import (
    CausalWeightMount,
    tensor_range_plan_from_source,
)
from .config import Qwen38Config


QWEN38_BUNDLE_SCHEMA = "immer.qwen3.8-complete-causal-bundle/v1"
NESTED_WEIGHTS_LAYOUT = "nested/v1"
FLAT_WEIGHTS_LAYOUT = "flat/v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Qwen38BundleError(ValueError):
    """A mounted Qwen causal bundle is incomplete or unauthenticated."""


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
        raise Qwen38BundleError("bundle value is not canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor, opened = _open_regular(path, "bundle shard")
    try:
        while chunk := os.read(descriptor, 4 * 1024**2):
            digest.update(chunk)
        current = os.fstat(descriptor)
        if (
            current.st_size != opened.st_size
            or current.st_mtime_ns != opened.st_mtime_ns
            or current.st_ctime_ns != opened.st_ctime_ns
        ):
            raise Qwen38BundleError("bundle file changed while hashing")
        linked = _regular_file(path, "bundle shard")
        if (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino):
            raise Qwen38BundleError("bundle shard changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in entries:
        if key in result:
            raise Qwen38BundleError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    value = _read_regular_bytes(path, label)
    try:
        document = json.loads(value.decode("utf-8"), object_pairs_hook=_pairs)
    except Qwen38BundleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Qwen38BundleError(f"cannot read {label}: {path}") from exc
    if not isinstance(document, dict):
        raise Qwen38BundleError(f"{label} root is invalid")
    return document


def _regular_file(path: Path, label: str):
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise Qwen38BundleError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise Qwen38BundleError(f"{label} must be a non-symlink regular file")
    return metadata


def _open_regular(path: Path, label: str) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Qwen38BundleError(f"cannot open {label}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        linked = _regular_file(path, label)
        if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
            raise Qwen38BundleError(f"{label} changed while opening")
        return descriptor, opened
    except Exception:
        os.close(descriptor)
        raise


def _read_regular_bytes(path: Path, label: str) -> bytes:
    descriptor, opened = _open_regular(path, label)
    if opened.st_size > 64 * 1024**2:
        os.close(descriptor)
        raise Qwen38BundleError(f"{label} exceeds its metadata bound")
    chunks: list[bytes] = []
    try:
        while chunk := os.read(descriptor, 1024**2):
            chunks.append(chunk)
        current = os.fstat(descriptor)
        if (
            current.st_size != opened.st_size
            or current.st_mtime_ns != opened.st_mtime_ns
            or current.st_ctime_ns != opened.st_ctime_ns
        ):
            raise Qwen38BundleError(f"{label} changed while reading")
        linked = _regular_file(path, label)
        if (linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino):
            raise Qwen38BundleError(f"{label} changed while reading")
    finally:
        os.close(descriptor)
    value = b"".join(chunks)
    if len(value) != opened.st_size:
        raise Qwen38BundleError(f"{label} returned a short read")
    return value


def _payload_sha256(shard: Mapping[str, Any]) -> str:
    values: set[str] = set()
    for raw in (shard.get("payload_sha256"), shard.get("linked_etag")):
        if raw is None:
            continue
        value = str(raw).strip().strip('"').lower()
        if _SHA256.fullmatch(value):
            values.add(value)
    if len(values) != 1:
        raise Qwen38BundleError("pinned shard lacks one unambiguous payload SHA-256")
    return next(iter(values))


def _pinned_inventory(
    path: Path,
    *,
    repo_id: str,
    revision: str,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    document = _strict_json(path, "pinned inventory")
    if (
        set(document)
        != {
            "inventory",
            "inventory_sha256",
            "repo_id",
            "revision",
            "schema",
            "source_fingerprint",
        }
        or document.get("schema") != "immer.tensor-inventory-cache/v1"
    ):
        raise Qwen38BundleError("pinned inventory schema is invalid")
    inventory = document.get("inventory")
    if not isinstance(inventory, dict):
        raise Qwen38BundleError("pinned inventory body is invalid")
    if (
        document.get("repo_id") != repo_id
        or document.get("revision") != revision
        or inventory.get("repo") != repo_id
        or inventory.get("revision") != revision
        or document.get("inventory_sha256") != _sha256(inventory)
    ):
        raise Qwen38BundleError("pinned inventory identity is invalid")
    fingerprint = Streamer._source_fingerprint(inventory)
    if document.get("source_fingerprint") != fingerprint:
        raise Qwen38BundleError("pinned inventory fingerprint is invalid")
    return inventory, fingerprint, document


def verify_qwen38_causal_mount(
    mount: CausalWeightMount,
    *,
    require_official_config: bool = True,
) -> dict[str, Any]:
    """Fully re-hash one mounted payload and replay every tensor binding."""

    if not isinstance(mount, CausalWeightMount):
        raise TypeError("mount must be a CausalWeightMount")
    manifest = _strict_json(mount.root / "bundle.json", "bundle manifest")
    if (
        set(manifest) != {"body", "schema", "sha256"}
        or manifest.get("schema") != QWEN38_BUNDLE_SCHEMA
        or not isinstance(manifest.get("body"), Mapping)
        or manifest.get("sha256") != _sha256(manifest["body"])
    ):
        raise Qwen38BundleError("bundle manifest identity is invalid")
    body = manifest["body"]
    weights_layout = body.get("weights_layout", NESTED_WEIGHTS_LAYOUT)
    if weights_layout not in (NESTED_WEIGHTS_LAYOUT, FLAT_WEIGHTS_LAYOUT):
        raise Qwen38BundleError("bundle weights layout is invalid")
    if weights_layout != f"{mount.weights_layout}/v1":
        raise Qwen38BundleError("mounted weights layout differs from manifest")
    if (
        body.get("checkpoint_complete") is not True
        or body.get("layout_fingerprint") != mount.layout.layout_fingerprint
        or body.get("logical_model") != mount.model.as_record()
    ):
        raise Qwen38BundleError("bundle completeness identity is invalid")

    inventory, fingerprint, pinned = _pinned_inventory(
        mount.weights_root / "inventory.pinned.json",
        repo_id=mount.model.repo_id,
        revision=mount.model.revision,
    )
    if (
        fingerprint != mount.layout.layout_fingerprint
        or body.get("inventory_sha256") != pinned["inventory_sha256"]
    ):
        raise Qwen38BundleError("bundle inventory receipt is invalid")

    config_path = mount.weights_root / "config.json"
    config_bytes = _read_regular_bytes(config_path, "bundle config")
    try:
        config_document = json.loads(config_bytes, object_pairs_hook=_pairs)
        Qwen38Config.from_mapping(
            config_document,
            require_official=require_official_config,
        )
    except (TypeError, ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise Qwen38BundleError("bundle config is not executable") from exc
    if hashlib.sha256(config_bytes).hexdigest() != body.get("config_sha256"):
        raise Qwen38BundleError("bundle config SHA-256 mismatch")

    index_path = mount.weights_root / "model.safetensors.index.json"
    if body.get("index_sha256") is None:
        if index_path.exists() or index_path.is_symlink():
            raise Qwen38BundleError("bundle has an unexpected checkpoint index")
    else:
        index_bytes = _read_regular_bytes(index_path, "checkpoint index")
        try:
            index = json.loads(index_bytes.decode("utf-8"), object_pairs_hook=_pairs)
        except (Qwen38BundleError, UnicodeError, json.JSONDecodeError) as exc:
            raise Qwen38BundleError("checkpoint index is invalid") from exc
        if not isinstance(index, Mapping):
            raise Qwen38BundleError("checkpoint index root is invalid")
        expected_map = {
            str(row["name"]): str(row["shard"]) for row in inventory.get("tensors", ())
        }
        if index.get("weight_map") != expected_map:
            raise Qwen38BundleError("checkpoint index differs from inventory")
        if hashlib.sha256(index_bytes).hexdigest() != body.get("index_sha256"):
            raise Qwen38BundleError("bundle index SHA-256 mismatch")

    raw_receipts = body.get("shards")
    if not isinstance(raw_receipts, list) or len(raw_receipts) != len(
        inventory.get("shards", ())
    ):
        raise Qwen38BundleError("bundle shard receipts are incomplete")
    receipts = {
        str(row.get("file")): row for row in raw_receipts if isinstance(row, Mapping)
    }
    payload_receipts: list[dict[str, Any]] = []
    checkpoint_bytes = 0
    for shard in inventory.get("shards", ()):
        if not isinstance(shard, Mapping):
            raise Qwen38BundleError("pinned shard entry is invalid")
        name = str(shard.get("file"))
        shard_path = PurePosixPath(name)
        if (
            shard_path.is_absolute()
            or len(shard_path.parts) != 1
            or any(part in ("", ".", "..") for part in shard_path.parts)
        ):
            raise Qwen38BundleError("pinned shard filename is unsafe")
        path = mount.weights_root / name
        metadata = _regular_file(path, "bundle shard")
        expected_size = int(shard.get("size", -1))
        expected_sha256 = _payload_sha256(shard)
        actual_sha256 = _sha256_file(path)
        receipt = receipts.get(name)
        if (
            receipt is None
            or metadata.st_size != expected_size
            or receipt.get("size") != expected_size
            or receipt.get("sha256") != actual_sha256
            or actual_sha256 != expected_sha256
        ):
            raise Qwen38BundleError(f"bundle shard verification failed: {name}")
        payload_receipts.append(
            {"file": name, "sha256": actual_sha256, "size": expected_size}
        )
        checkpoint_bytes += expected_size
    if checkpoint_bytes != body.get("checkpoint_bytes"):
        raise Qwen38BundleError("bundle checkpoint byte receipt is invalid")

    plans = tuple(
        tensor_range_plan_from_source(mount.source, str(row["name"]))
        for row in inventory.get("tensors", ())
    )
    if len(plans) != body.get("tensor_bindings"):
        raise Qwen38BundleError("bundle tensor binding count is invalid")
    for plan in plans:
        if mount.resolve_tensor_plan(plan.name) != plan:
            raise Qwen38BundleError(f"causal tensor plan differs: {plan.name}")
    graph_revision = mount.graph.store.revision()
    if body.get("graph_revision") != [graph_revision[0], graph_revision[1]]:
        raise Qwen38BundleError("bundle graph revision differs")

    return {
        "checkpoint_bytes": checkpoint_bytes,
        "graph_revision": [graph_revision[0], graph_revision[1]],
        "kind": "complete-causal-bundle/v1",
        "layout_fingerprint": fingerprint,
        "manifest_sha256": manifest["sha256"],
        "shards": len(payload_receipts),
        "shards_sha256": _sha256(payload_receipts),
        "tensor_bindings": len(plans),
        "weights_layout": weights_layout,
    }


__all__ = [
    "FLAT_WEIGHTS_LAYOUT",
    "NESTED_WEIGHTS_LAYOUT",
    "QWEN38_BUNDLE_SCHEMA",
    "Qwen38BundleError",
    "verify_qwen38_causal_mount",
]
