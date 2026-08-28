"""Authenticated complete-bundle verification for Qwen3.8 runtimes."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
from typing import Any, cast

from immer.knowledge import Streamer

from ..deepseek_v4.causal_weights import (
    CausalWeightMount,
    tensor_range_plan_from_source,
)
from .config import Qwen38Config


QWEN38_BUNDLE_SCHEMA = "immer.qwen3.8-complete-causal-bundle/v1"
QWEN38_BUNDLE_VERIFY_CACHE_SCHEMA = "immer.qwen3.8-bundle-verify-cache/v1"
QWEN38_BUNDLE_VERIFY_CACHE_NAME = ".bundle-verify-cache-v1.json"
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


def _cache_identity(
    *,
    manifest: Mapping[str, Any],
    pinned: Mapping[str, Any],
    fingerprint: str,
    weights_layout: str,
) -> dict[str, Any]:
    body = cast(Mapping[str, Any], manifest["body"])
    return {
        "config_sha256": body.get("config_sha256"),
        "index_sha256": body.get("index_sha256"),
        "inventory_sha256": pinned.get("inventory_sha256"),
        "layout_fingerprint": fingerprint,
        "manifest_sha256": manifest.get("sha256"),
        "weights_layout": weights_layout,
    }


def _load_verify_cache(
    path: Path,
    *,
    identity: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if not path.exists() and not path.is_symlink():
        return {}
    try:
        document = _strict_json(path, "bundle verification cache")
    except Qwen38BundleError:
        return {}
    if (
        set(document) != {"body", "schema", "sha256"}
        or document.get("schema") != QWEN38_BUNDLE_VERIFY_CACHE_SCHEMA
        or not isinstance(document.get("body"), Mapping)
        or document.get("sha256") != _sha256(document["body"])
    ):
        return {}
    body = document["body"]
    if (
        not isinstance(body, Mapping)
        or body.get("identity") != dict(identity)
        or not isinstance(body.get("shards"), list)
    ):
        return {}
    rows: dict[str, Mapping[str, Any]] = {}
    for raw in body["shards"]:
        if not isinstance(raw, Mapping) or set(raw) != {
            "ctime_ns",
            "device",
            "file",
            "inode",
            "mtime_ns",
            "sha256",
            "size",
        }:
            return {}
        name = raw.get("file")
        if (
            not isinstance(name, str)
            or PurePosixPath(name).name != name
            or name in rows
            or not isinstance(raw.get("sha256"), str)
            or _SHA256.fullmatch(cast(str, raw["sha256"])) is None
            or any(
                isinstance(raw.get(field), bool)
                or not isinstance(raw.get(field), int)
                or cast(int, raw[field]) < 0
                for field in ("ctime_ns", "device", "inode", "mtime_ns", "size")
            )
        ):
            return {}
        rows[name] = raw
    return rows


def _write_verify_cache(
    path: Path,
    *,
    identity: Mapping[str, Any],
    shards: list[dict[str, Any]],
) -> None:
    body = {"identity": dict(identity), "shards": shards}
    document = {
        "body": body,
        "schema": QWEN38_BUNDLE_VERIFY_CACHE_SCHEMA,
        "sha256": _sha256(body),
    }
    data = _canonical(document)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
            0o600,
        )
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short verification-cache write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
    except OSError:
        # Read-only bundles still work through the original full verification.
        pass
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


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
    use_verification_cache: bool = True,
) -> dict[str, Any]:
    """Verify one mount, reusing unchanged local shard digests when available."""

    if not isinstance(mount, CausalWeightMount):
        raise TypeError("mount must be a CausalWeightMount")
    if not isinstance(use_verification_cache, bool):
        raise TypeError("use_verification_cache must be boolean")
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
    cache_path = mount.causal_root / QWEN38_BUNDLE_VERIFY_CACHE_NAME
    cache_identity = _cache_identity(
        manifest=manifest,
        pinned=pinned,
        fingerprint=fingerprint,
        weights_layout=weights_layout,
    )
    cached_shards = (
        _load_verify_cache(cache_path, identity=cache_identity)
        if use_verification_cache
        else {}
    )
    next_cache_rows: list[dict[str, Any]] = []
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
        identity_descriptor, metadata = _open_regular(path, "bundle shard")
        os.close(identity_descriptor)
        expected_size = int(shard.get("size", -1))
        expected_sha256 = _payload_sha256(shard)
        receipt = receipts.get(name)
        cached = cached_shards.get(name)
        cache_matches = cached is not None and all(
            cached.get(field) == value
            for field, value in (
                ("ctime_ns", metadata.st_ctime_ns),
                ("device", metadata.st_dev),
                ("inode", metadata.st_ino),
                ("mtime_ns", metadata.st_mtime_ns),
                ("sha256", expected_sha256),
                ("size", metadata.st_size),
            )
        )
        actual_sha256 = expected_sha256 if cache_matches else _sha256_file(path)
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
        next_cache_rows.append(
            {
                "ctime_ns": metadata.st_ctime_ns,
                "device": metadata.st_dev,
                "file": name,
                "inode": metadata.st_ino,
                "mtime_ns": metadata.st_mtime_ns,
                "sha256": actual_sha256,
                "size": metadata.st_size,
            }
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

    if use_verification_cache:
        _write_verify_cache(
            cache_path,
            identity=cache_identity,
            shards=next_cache_rows,
        )

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
    "QWEN38_BUNDLE_VERIFY_CACHE_NAME",
    "QWEN38_BUNDLE_VERIFY_CACHE_SCHEMA",
    "Qwen38BundleError",
    "verify_qwen38_causal_mount",
]
