#!/usr/bin/env python3
"""Build a trace-complete sparse local DeepSeek-V4 ``weights/ + causal/`` bundle.

The logical shard sizes and tensor offsets remain byte-identical to the pinned
checkpoint. Only verified access-trace leaves consume physical disk blocks;
all other shard regions are sparse holes. Causal expert bindings are created
only when every byte of that expert is materialized, so unseen experts fail
closed instead of reading a hole.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
from typing import Any
import uuid

from immer.knowledge import AccessTrace, Streamer
from immer.runtimes.deepseek_v4 import (
    CausalWeightMount,
    DeepSeekWeightPager,
    LogicalModelIdentity,
)

ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_SOURCE = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
OFFICIAL_LAYOUT_FINGERPRINT = (
    "61600c552f3e52ae382b3eca0370001905fc4e2ecdd2da5f3e66fa95809206c7"
)
DEFAULT_CACHE = ROOT / "artifacts" / "private" / "deepseek-v4-cache"
BUNDLE_SCHEMA = "immer.deepseek-v4-sparse-causal-bundle/v1"
_EXPERT_TENSOR = re.compile(
    r"^layers\.(0|[1-9][0-9]*)\.ffn\.experts\."
    r"(0|[1-9][0-9]*)\.(w[123])\.(weight|scale)$"
)
_EXPERT_PARTS = frozenset(
    (f"w{index}.{kind}" for index in (1, 2, 3) for kind in ("scale", "weight"))
)


class SparseBundleError(RuntimeError):
    """The sparse local bundle cannot be proven complete for its trace."""


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
        raise SparseBundleError("value is not canonical JSON") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: object) -> str:
    return _sha256_bytes(_canonical(value))


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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


def _strict_json(path: Path) -> Any:
    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in entries:
            if key in result:
                raise SparseBundleError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except SparseBundleError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SparseBundleError(f"cannot read JSON: {path}") from exc


@dataclass(frozen=True, slots=True)
class CachedRange:
    shard: str
    start: int
    stop: int
    payload: Path
    sha256: str


class VerifiedRangeCache:
    def __init__(self, root: Path, *, repo_id: str, revision: str) -> None:
        self.root = root
        self.repo_id = repo_id
        self.revision = revision
        self.by_shard: dict[str, list[CachedRange]] = defaultdict(list)
        self._verified: set[Path] = set()
        if not root.is_dir():
            raise SparseBundleError(f"range cache is missing: {root}")
        for metadata in root.glob("*.json"):
            document = _strict_json(metadata)
            if not isinstance(document, Mapping) or document.get("schema") != (
                "immer.range-cache/v1"
            ):
                continue
            contract = document.get("contract")
            payload = metadata.with_suffix(".bin")
            if (
                not isinstance(contract, Mapping)
                or contract.get("kind") != "range"
                or contract.get("repo") != repo_id
                or contract.get("revision") != revision
                or not payload.is_file()
            ):
                continue
            start = int(contract["start"])
            stop = int(contract["end"]) + 1
            size = int(document.get("size", -1))
            digest = document.get("sha256")
            if (
                start < 0
                or stop <= start
                or stop - start != size
                or payload.stat().st_size != size
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise SparseBundleError(f"invalid cached range metadata: {metadata}")
            shard = str(contract["filename"])
            self.by_shard[shard].append(
                CachedRange(shard, start, stop, payload, digest)
            )
        for ranges in self.by_shard.values():
            ranges.sort(key=lambda row: (row.start, row.stop, row.payload.name))

    def resolve(self, shard: str, offset: int, length: int) -> tuple[CachedRange, int]:
        stop = offset + length
        for row in self.by_shard.get(shard, ()):
            if row.start <= offset and row.stop >= stop:
                return row, offset - row.start
        raise SparseBundleError(f"cache does not cover {shard}[{offset}:{stop}]")

    def read(self, shard: str, offset: int, length: int) -> bytes:
        row, relative = self.resolve(shard, offset, length)
        if row.payload not in self._verified:
            encoded = row.payload.read_bytes()
            if _sha256_bytes(encoded) != row.sha256:
                raise SparseBundleError(f"cached range SHA-256 mismatch: {row.payload}")
            self._verified.add(row.payload)
        descriptor = os.open(row.payload, os.O_RDONLY)
        try:
            encoded = os.pread(descriptor, length, relative)
        finally:
            os.close(descriptor)
        if len(encoded) != length:
            raise SparseBundleError("cached range returned a short payload")
        return encoded


def _load_inventory(path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    document = _strict_json(path)
    if not isinstance(document, Mapping) or document.get("schema") != (
        "immer.tensor-inventory-cache/v1"
    ):
        raise SparseBundleError("inventory cache schema is invalid")
    inventory = document.get("inventory")
    fingerprint = document.get("source_fingerprint")
    if not isinstance(inventory, dict) or not isinstance(fingerprint, str):
        raise SparseBundleError("inventory cache identity is invalid")
    return inventory, fingerprint, dict(document)


def _load_traces(
    paths: Iterable[Path],
    *,
    repo_id: str,
    revision: str,
    fingerprint: str,
) -> tuple[tuple[AccessTrace, ...], tuple[tuple[str, int, int], ...]]:
    traces: list[AccessTrace] = []
    leaves: set[tuple[str, int, int]] = set()
    for path in paths:
        trace = AccessTrace.from_bytes(path.read_bytes())
        trace.verify()
        if (
            trace.repo_id != repo_id
            or trace.revision != revision
            or trace.inventory_fingerprint != fingerprint
        ):
            raise SparseBundleError(f"access trace identity mismatch: {path}")
        traces.append(trace)
        leaves.update(
            (leaf.shard, leaf.offset, leaf.length)
            for operation in trace.operations
            for leaf in operation.leaves
        )
    if not traces or not leaves:
        raise SparseBundleError("at least one non-empty access trace is required")
    return tuple(traces), tuple(sorted(leaves))


def _header_bytes(
    shard: Mapping[str, Any], tensors: Sequence[Mapping[str, Any]]
) -> bytes:
    header: dict[str, Any] = {}
    if shard.get("st_metadata") is not None:
        header["__metadata__"] = shard["st_metadata"]
    for tensor in sorted(tensors, key=lambda row: int(row["offset_in_shard"][0])):
        header[str(tensor["name"])] = {
            "dtype": str(tensor["dtype"]),
            "shape": [int(value) for value in tensor["shape"]],
            "data_offsets": [int(value) for value in tensor["offset_in_shard"]],
        }
    encoded = json.dumps(
        header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    capacity = int(shard["data_start"]) - 8
    if len(encoded) > capacity:
        raise SparseBundleError(
            f"reconstructed header exceeds original capacity for {shard['file']}"
        )
    return struct.pack("<Q", capacity) + encoded + b" " * (capacity - len(encoded))


def _interval_covered(
    intervals: Mapping[str, Sequence[tuple[int, int]]],
    shard: str,
    start: int,
    stop: int,
) -> bool:
    cursor = start
    for left, right in intervals.get(shard, ()):
        if right <= cursor:
            continue
        if left > cursor:
            return False
        cursor = max(cursor, right)
        if cursor >= stop:
            return True
    return False


def _copy_config(
    cache_root: Path, weights: Path, *, repo_id: str, revision: str
) -> str:
    for metadata in (cache_root / "files").glob("*.json"):
        document = _strict_json(metadata)
        contract = document.get("contract") if isinstance(document, Mapping) else None
        if (
            isinstance(contract, Mapping)
            and contract.get("filename") == "config.json"
            and contract.get("repo") == repo_id
            and contract.get("revision") == revision
        ):
            payload = metadata.with_suffix(".bin")
            encoded = payload.read_bytes()
            if _sha256_bytes(encoded) != document.get("sha256"):
                raise SparseBundleError("cached config SHA-256 mismatch")
            _atomic_bytes(weights / "config.json", encoded)
            return _sha256_bytes(encoded)
    raise SparseBundleError("verified config.json is absent from the cache")


def _write_sparse_weights(
    weights: Path,
    inventory: Mapping[str, Any],
    leaves: Sequence[tuple[str, int, int]],
    cache: VerifiedRangeCache,
) -> tuple[list[dict[str, Any]], dict[str, list[tuple[int, int]]]]:
    tensors_by_shard: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for tensor in inventory.get("tensors", []):
        tensors_by_shard[str(tensor["shard"])].append(tensor)
    leaves_by_shard: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for shard, offset, length in leaves:
        leaves_by_shard[shard].append((offset, offset + length))
    for intervals in leaves_by_shard.values():
        intervals.sort()

    shard_receipts: list[dict[str, Any]] = []
    for shard in inventory.get("shards", []):
        filename = str(shard["file"])
        path = weights / filename
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        descriptor = os.open(path, flags, 0o600)
        try:
            size = int(shard["size"])
            os.ftruncate(descriptor, size)
            header = _header_bytes(shard, tensors_by_shard[filename])
            if os.pwrite(descriptor, header, 0) != len(header):
                raise SparseBundleError(f"short header write for {filename}")
            payload_bytes = 0
            for offset, stop in leaves_by_shard.get(filename, ()):
                encoded = cache.read(filename, offset, stop - offset)
                if os.pwrite(descriptor, encoded, offset) != len(encoded):
                    raise SparseBundleError(f"short payload write for {filename}")
                payload_bytes += len(encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        metadata = path.stat()
        shard_receipts.append(
            {
                "file": filename,
                "header_bytes": len(header),
                "logical_bytes": metadata.st_size,
                "materialized_leaf_bytes": payload_bytes,
                "physical_bytes": metadata.st_blocks * 512,
            }
        )
    return shard_receipts, leaves_by_shard


def _write_index(weights: Path, inventory: Mapping[str, Any]) -> str:
    document = {
        "metadata": {"total_size": int(inventory["index_total_size"])},
        "weight_map": {
            str(tensor["name"]): str(tensor["shard"]) for tensor in inventory["tensors"]
        },
    }
    encoded = _canonical(document)
    _atomic_bytes(weights / "model.safetensors.index.json", encoded)
    return _sha256_bytes(encoded)


def _covered_experts(
    inventory: Mapping[str, Any],
    intervals: Mapping[str, Sequence[tuple[int, int]]],
) -> dict[int, tuple[int, ...]]:
    parts: dict[tuple[int, int], set[str]] = defaultdict(set)
    for tensor in inventory["tensors"]:
        match = _EXPERT_TENSOR.fullmatch(str(tensor["name"]))
        if match is None:
            continue
        shard = str(tensor["shard"])
        start = int(tensor["data_start"]) + int(tensor["offset_in_shard"][0])
        stop = int(tensor["data_start"]) + int(tensor["offset_in_shard"][1])
        if _interval_covered(intervals, shard, start, stop):
            parts[(int(match.group(1)), int(match.group(2)))].add(
                f"{match.group(3)}.{match.group(4)}"
            )
    by_layer: dict[int, list[int]] = defaultdict(list)
    for (layer, expert), observed in parts.items():
        if observed == _EXPERT_PARTS:
            by_layer[layer].append(expert)
    return {
        layer: tuple(sorted(experts)) for layer, experts in sorted(by_layer.items())
    }


def build_bundle(args: argparse.Namespace) -> dict[str, Any]:
    inventory, fingerprint, inventory_document = _load_inventory(
        Path(args.inventory).expanduser().resolve()
    )
    if fingerprint != args.layout_fingerprint:
        raise SparseBundleError(
            "inventory fingerprint does not match the pinned layout"
        )
    traces, leaves = _load_traces(
        (Path(path).expanduser().resolve() for path in args.access_trace),
        repo_id=args.repo_id,
        revision=args.revision,
        fingerprint=fingerprint,
    )
    cache_root = Path(args.cache_dir).expanduser().resolve()
    cache = VerifiedRangeCache(
        cache_root / "ranges", repo_id=args.repo_id, revision=args.revision
    )
    for shard, offset, length in leaves:
        cache.resolve(shard, offset, length)

    target = Path(args.output).expanduser().resolve()
    if target.exists() or target.is_symlink():
        raise SparseBundleError(f"output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.parent / f".{target.name}.pending-{uuid.uuid4().hex}"
    weights = pending / "weights"
    causal = pending / "causal"
    weights.mkdir(parents=True)
    causal.mkdir()
    try:
        config_sha = _copy_config(
            cache_root, weights, repo_id=args.repo_id, revision=args.revision
        )
        pinned_inventory_bytes = _canonical(inventory_document)
        _atomic_bytes(weights / "inventory.pinned.json", pinned_inventory_bytes)
        index_sha = _write_index(weights, inventory)
        shard_receipts, intervals = _write_sparse_weights(
            weights, inventory, leaves, cache
        )
        covered = _covered_experts(inventory, intervals)
        if not covered:
            raise SparseBundleError("trace materialized no complete routed expert")

        source = Streamer.from_local(
            weights,
            repo_id=args.repo_id,
            revision=args.revision,
            pinned_inventory=inventory,
            pinned_fingerprint=fingerprint,
            use_cache=False,
            budget_mb=args.budget_mb,
        )
        try:
            source.inventory()
            observed_fingerprint = source.metrics().get("inventory_source_fingerprint")
            if observed_fingerprint != fingerprint:
                raise SparseBundleError(
                    "reconstructed sparse layout fingerprint does not match source"
                )
            pager = DeepSeekWeightPager(
                source,
                device="cpu",
                compute_dtype="bfloat16",
                expert_prefetch=False,
            )
            plans = tuple(
                plan
                for layer, experts in covered.items()
                for plan in pager.plan_expert_ranges(layer, experts)
            )
        finally:
            source.close()

        model = LogicalModelIdentity(repo_id=args.repo_id, revision=args.revision)
        with CausalWeightMount(
            pending,
            model,
            budget_mb=args.budget_mb,
        ) as mount:
            receipt = mount.bind_plans(plans)
            if len(receipt.bindings) != len(plans):
                raise SparseBundleError("causal binding receipt lost expert plans")
            first = plans[0]
            if mount.resolve_expert_plans(first.layer, (first.expert_id,)) != (first,):
                raise SparseBundleError("causal reader did not replay the first plan")

        physical_bytes = sum(row["physical_bytes"] for row in shard_receipts)
        logical_bytes = sum(row["logical_bytes"] for row in shard_receipts)
        identity = {
            "causal_bindings": len(plans),
            "config_sha256": config_sha,
            "covered_experts": {
                str(layer): list(experts) for layer, experts in covered.items()
            },
            "index_sha256": index_sha,
            "pinned_inventory_sha256": _sha256_bytes(pinned_inventory_bytes),
            "layout_fingerprint": fingerprint,
            "logical_model": model.as_record(),
            "logical_shard_bytes": logical_bytes,
            "materialized_leaf_bytes": sum(length for _, _, length in leaves),
            "physical_shard_bytes": physical_bytes,
            "schema": BUNDLE_SCHEMA,
            "shards": shard_receipts,
            "trace_sha256": [trace.sha256 for trace in traces],
            "unique_leaves": len(leaves),
        }
        manifest = {**identity, "sha256": _sha256(identity)}
        _atomic_bytes(pending / "bundle.json", _canonical(manifest))
        os.replace(pending, target)
        return manifest
    except BaseException:
        if pending.exists():
            shutil.rmtree(pending)
        raise


def verify_bundle(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.bundle).expanduser().resolve()
    document = _strict_json(root / "bundle.json")
    if not isinstance(document, dict) or document.get("schema") != BUNDLE_SCHEMA:
        raise SparseBundleError("bundle manifest schema is invalid")
    identity = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != _sha256(identity):
        raise SparseBundleError("bundle manifest digest does not match")
    model = LogicalModelIdentity(repo_id=args.repo_id, revision=args.revision)
    with CausalWeightMount(root, model, budget_mb=args.budget_mb) as mount:
        if mount.layout.layout_fingerprint != document["layout_fingerprint"]:
            raise SparseBundleError(
                "mounted layout fingerprint does not match manifest"
            )
        covered = document.get("covered_experts")
        if not isinstance(covered, Mapping):
            raise SparseBundleError("bundle covered-expert table is invalid")
        resolved = sum(
            len(mount.resolve_expert_plans(int(layer), experts))
            for layer, experts in covered.items()
        )
        if resolved != int(document["causal_bindings"]):
            raise SparseBundleError("bundle causal binding count does not match")
        metrics = mount.reader.metrics()
    return {
        "causal_bindings": resolved,
        "layout_fingerprint": document["layout_fingerprint"],
        "manifest_sha256": document["sha256"],
        "reader": metrics,
        "schema": f"{BUNDLE_SCHEMA}:verification/v1",
    }


def _positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--inventory", required=True)
    build.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    build.add_argument("--access-trace", action="append", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--repo-id", default=OFFICIAL_SOURCE)
    build.add_argument("--revision", default=OFFICIAL_REVISION)
    build.add_argument("--layout-fingerprint", default=OFFICIAL_LAYOUT_FINGERPRINT)
    build.add_argument("--budget-mb", type=_positive_float, default=512.0)
    build.set_defaults(handler=build_bundle)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    verify.add_argument("--repo-id", default=OFFICIAL_SOURCE)
    verify.add_argument("--revision", default=OFFICIAL_REVISION)
    verify.add_argument("--budget-mb", type=_positive_float, default=512.0)
    verify.set_defaults(handler=verify_bundle)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = args.handler(args)
    except SparseBundleError as exc:
        raise SystemExit(f"sparse causal bundle failed: {exc}") from exc
    print(json.dumps(document, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
