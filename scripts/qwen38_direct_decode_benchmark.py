#!/usr/bin/env python3
"""Benchmark one Qwen3.8 decode token from an authenticated shared prefix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time
from typing import Any

import torch

from immer.knowledge import AccessTraceRecorder, Streamer
from immer.runtimes.qwen3_8 import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    CausalWeightMount,
    DELTANET_PROBE_SCHEMA,
    DeltaNetProbeRecorder,
    LogicalModelIdentity,
    Qwen38Config,
    Qwen38WeightPager,
    StreamedQwen38,
    build_probe_document,
    tensor_range_plan_from_source,
    verify_probe_document,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "artifacts" / "private" / "qwen3.8-cache"
INPUT_SCHEMA = "immer.qwen3.8-direct-decode-input/v1"
PREFIX_SCHEMA = "immer.qwen3.8-direct-prefix/v1"
DECODE_SCHEMA = "immer.qwen3.8-direct-decode/v1"
COMPARISON_SCHEMA = "immer.qwen3.8-direct-decode-comparison/v1"
FERTIG_INPUT_SCHEMA = "immer.qwen3.8-fertig-draft-inputs/v1"


class QwenDirectDecodeError(RuntimeError):
    """The shared-prefix Qwen decode contract cannot be satisfied."""


@dataclass(slots=True)
class RuntimeSource:
    source: Streamer
    mount: CausalWeightMount | None = None
    verification: dict[str, Any] | None = None

    def close(self) -> None:
        if self.mount is not None:
            self.mount.close()
        else:
            self.source.close()


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
        raise QwenDirectDecodeError("value is not canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _strict_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()

    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in entries:
            if key in result:
                raise QwenDirectDecodeError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        document = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=pairs
        )
    except QwenDirectDecodeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QwenDirectDecodeError(f"cannot read JSON: {source}") from exc
    if not isinstance(document, dict):
        raise QwenDirectDecodeError("JSON root must be an object")
    return document


def _atomic_bytes(path: str | os.PathLike[str], value: bytes) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".pending", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def _write_json(path: str | os.PathLike[str], document: Mapping[str, Any]) -> Path:
    return _atomic_bytes(path, _canonical(dict(document)) + b"\n")


def _result(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(identity), "sha256": _sha256(identity)}


def _hidden_sha256(hidden: torch.Tensor) -> str:
    cpu = hidden.detach().to(device="cpu").contiguous()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def _token_rows(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise QwenDirectDecodeError(f"{label} must be a non-empty token list")
    tokens: list[int] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise QwenDirectDecodeError(f"{label} contains an invalid token")
        tokens.append(raw)
    return tuple(tokens)


def _load_input(path: str | os.PathLike[str]) -> dict[str, Any]:
    document = _strict_json(path)
    if document.get("schema") != INPUT_SCHEMA or set(document) != {
        "decode_token_id",
        "item_id",
        "prefix_token_ids",
        "schema",
        "sha256",
    }:
        raise QwenDirectDecodeError("direct-decode input schema is invalid")
    identity = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != _sha256(identity):
        raise QwenDirectDecodeError("direct-decode input SHA-256 mismatch")
    if not isinstance(document.get("item_id"), str) or not document["item_id"]:
        raise QwenDirectDecodeError("direct-decode item_id is invalid")
    prefix = _token_rows(document.get("prefix_token_ids"), "prefix_token_ids")
    decode = document.get("decode_token_id")
    if isinstance(decode, bool) or not isinstance(decode, int) or decode < 0:
        raise QwenDirectDecodeError("decode_token_id is invalid")
    return {**document, "prefix_token_ids": list(prefix)}


def select_input(args: argparse.Namespace) -> dict[str, Any]:
    document = _strict_json(args.inputs)
    if document.get("schema") != FERTIG_INPUT_SCHEMA:
        raise QwenDirectDecodeError("Qwen FERTIG input schema is invalid")
    rows = document.get("items")
    if not isinstance(rows, list):
        raise QwenDirectDecodeError("Qwen FERTIG input rows are invalid")
    matching = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("item_id") == args.item_id
    ]
    if len(matching) != 1:
        raise QwenDirectDecodeError("item_id does not select exactly one input")
    row = matching[0]
    prefix = _token_rows(row.get("prompt_token_ids"), "prompt_token_ids")
    draft = _token_rows(row.get("draft_token_ids"), "draft_token_ids")
    identity = {
        "decode_token_id": draft[0],
        "item_id": args.item_id,
        "prefix_token_ids": list(prefix),
        "schema": INPUT_SCHEMA,
    }
    return _result(identity)


def _local_path(value: str) -> Path | None:
    path = Path(value).expanduser()
    return path.resolve() if path.exists() else None


def _pinned_inventory(
    path: str | None,
    *,
    repo_id: str,
    revision: str,
) -> tuple[Mapping[str, Any] | None, str | None]:
    if path is None:
        return None, None
    document = _strict_json(path)
    if (
        document.get("schema") != "immer.tensor-inventory-cache/v1"
        or document.get("repo_id") != repo_id
        or document.get("revision") != revision
        or not isinstance(document.get("inventory"), Mapping)
        or not isinstance(document.get("source_fingerprint"), str)
    ):
        raise QwenDirectDecodeError("pinned inventory schema is invalid")
    inventory = document["inventory"]
    if (
        inventory.get("repo") != repo_id
        or inventory.get("revision") != revision
        or document.get("inventory_sha256") != _sha256(inventory)
    ):
        raise QwenDirectDecodeError("pinned inventory identity is invalid")
    return inventory, document["source_fingerprint"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_shard_digest(shard: Mapping[str, Any]) -> str:
    values: set[str] = set()
    for raw in (shard.get("etag"), shard.get("cas_url_hash")):
        if raw is None:
            continue
        value = str(raw).strip().strip('"').lower()
        if re.fullmatch(r"[0-9a-f]{64}", value):
            values.add(value)
    if len(values) != 1:
        raise QwenDirectDecodeError("local shard lacks one unambiguous SHA-256")
    return next(iter(values))


def _verify_local_payload(
    root: Path,
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise QwenDirectDecodeError("local weight root is missing") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise QwenDirectDecodeError("local weight root must be a plain directory")
    receipts: list[dict[str, Any]] = []
    total = 0
    for shard in inventory.get("shards", ()):
        if not isinstance(shard, Mapping):
            raise QwenDirectDecodeError("pinned shard table is invalid")
        name = shard.get("file")
        size = shard.get("size")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise QwenDirectDecodeError("pinned shard coordinate is invalid")
        path = root / name
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise QwenDirectDecodeError(f"local shard is missing: {name}") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != size
        ):
            raise QwenDirectDecodeError(f"local shard shape is invalid: {name}")
        expected = _expected_shard_digest(shard)
        actual = _sha256_file(path)
        if actual != expected:
            raise QwenDirectDecodeError(f"local shard SHA-256 mismatch: {name}")
        receipts.append({"file": name, "sha256": actual, "size": size})
        total += size
    if not receipts:
        raise QwenDirectDecodeError("pinned inventory contains no shards")
    return {
        "checkpoint_bytes": total,
        "kind": "complete-local-shards/v1",
        "shards": len(receipts),
        "shards_sha256": _sha256(receipts),
    }


def _verify_causal_mount(mount: CausalWeightMount) -> dict[str, Any]:
    manifest = _strict_json(mount.root / "bundle.json")
    if (
        manifest.get("schema") != "immer.qwen3.8-complete-causal-bundle/v1"
        or set(manifest) != {"body", "schema", "sha256"}
        or not isinstance(manifest.get("body"), Mapping)
        or manifest.get("sha256") != _sha256(manifest["body"])
    ):
        raise QwenDirectDecodeError("causal bundle manifest is invalid")
    body = manifest["body"]
    manifest_weights_layout = body.get("weights_layout", "nested/v1")
    mounted_weights_layout = f"{mount.weights_layout}/v1"
    if (
        body.get("checkpoint_complete") is not True
        or body.get("layout_fingerprint") != mount.layout.layout_fingerprint
        or body.get("logical_model") != mount.model.as_record()
        or manifest_weights_layout != mounted_weights_layout
    ):
        raise QwenDirectDecodeError("causal bundle completeness identity is invalid")
    inventory, fingerprint = _pinned_inventory(
        str(mount.weights_root / "inventory.pinned.json"),
        repo_id=mount.model.repo_id,
        revision=mount.model.revision,
    )
    assert inventory is not None and fingerprint is not None
    payload = _verify_local_payload(mount.weights_root, inventory)
    config_path = mount.weights_root / "config.json"
    if (
        not config_path.is_file()
        or config_path.is_symlink()
        or _sha256_file(config_path) != body.get("config_sha256")
    ):
        raise QwenDirectDecodeError("causal bundle config receipt is invalid")
    index_path = mount.weights_root / "model.safetensors.index.json"
    if body.get("index_sha256") is None:
        if index_path.exists() or index_path.is_symlink():
            raise QwenDirectDecodeError("causal bundle has an unexpected index")
    elif (
        not index_path.is_file()
        or index_path.is_symlink()
        or _sha256_file(index_path) != body.get("index_sha256")
    ):
        raise QwenDirectDecodeError("causal bundle index receipt is invalid")
    if payload["checkpoint_bytes"] != body.get("checkpoint_bytes") or body.get(
        "tensor_bindings"
    ) != len(inventory.get("tensors", ())):
        raise QwenDirectDecodeError("causal bundle payload receipt is invalid")
    for row in inventory.get("tensors", ()):
        plan = tensor_range_plan_from_source(mount.source, str(row["name"]))
        if mount.resolve_tensor_plan(plan.name) != plan:
            raise QwenDirectDecodeError(f"causal tensor plan differs: {plan.name}")
    revision = mount.graph.store.revision()
    if body.get("graph_revision") != [revision[0], revision[1]]:
        raise QwenDirectDecodeError("causal bundle graph revision differs")
    return {
        **payload,
        "graph_revision": [revision[0], revision[1]],
        "kind": "complete-causal-bundle/v1",
        "manifest_sha256": manifest["sha256"],
        "tensor_bindings": body["tensor_bindings"],
        "weights_layout": manifest_weights_layout,
    }


def _build_source(
    args: argparse.Namespace, recorder: AccessTraceRecorder
) -> RuntimeSource:
    if args.causal_bundle is not None:
        mount = CausalWeightMount(
            args.causal_bundle,
            LogicalModelIdentity(args.logical_repo_id, args.revision),
            budget_mb=args.source_budget_mb,
        )
        try:
            verification = _verify_causal_mount(mount)
            mount.source.set_access_observer(recorder)
            return RuntimeSource(mount.source, mount, verification)
        except Exception:
            mount.close()
            raise
    local = _local_path(args.source)
    common = {
        "revision": args.revision,
        "budget_mb": args.source_budget_mb,
        "cache_dir": Path(args.cache_dir).expanduser().resolve(),
        "max_cache_bytes": int(args.max_cache_gb * 1024**3),
        "access_observer": recorder,
        "verbose": False,
    }
    if local is not None:
        inventory, fingerprint = _pinned_inventory(
            args.pinned_inventory,
            repo_id=args.logical_repo_id,
            revision=args.revision,
        )
        if inventory is None or fingerprint is None:
            raise QwenDirectDecodeError(
                "local source requires a pinned inventory with shard hashes"
            )
        verification = _verify_local_payload(local, inventory)
        return RuntimeSource(
            Streamer.from_local(
                local,
                repo_id=args.logical_repo_id,
                pinned_inventory=inventory,
                pinned_fingerprint=fingerprint,
                use_cache=False,
                **common,
            ),
            verification=verification,
        )
    return RuntimeSource(
        Streamer(args.source, **common),
        verification={"kind": "remote-pinned-range-source/v1"},
    )


def _runtime(
    args: argparse.Namespace,
    recorder: AccessTraceRecorder,
    delta_probe: DeltaNetProbeRecorder | None = None,
) -> tuple[RuntimeSource, StreamedQwen38]:
    runtime = _build_source(args, recorder)
    pager: Qwen38WeightPager | None = None
    try:
        raw_config = runtime.source.reader.fetch_file("config.json")
        config_document = json.loads(raw_config)
        if not isinstance(config_document, Mapping):
            raise QwenDirectDecodeError("checkpoint config root is invalid")
        config = Qwen38Config.from_mapping(
            config_document,
            require_official=getattr(args, "_require_official", True),
        )
        pager = Qwen38WeightPager(
            runtime.source,
            device=args.device,
            compute_dtype=args.dtype,
            max_resident_bytes=args.max_resident_mb * 1024**2,
            require_source_identity=True,
            causal_tensor_reader=(
                None if runtime.mount is None else runtime.mount.tensor_reader
            ),
        )
        model = StreamedQwen38(
            config,
            pager,
            delta_probe=delta_probe,
            max_batch_size=1,
            max_seq_len=args.max_seq_len,
        )
        model.checkpoint_preflight()
        return runtime, model
    except Exception:
        if pager is not None:
            pager.close()
        runtime.close()
        raise


def _progress(event: Mapping[str, Any]) -> None:
    sys.stderr.write(json.dumps(dict(event), sort_keys=True) + "\n")
    sys.stderr.flush()


def _trace_receipt(
    recorder: AccessTraceRecorder, path: str | os.PathLike[str]
) -> dict[str, Any]:
    metrics = recorder.metrics()
    if metrics["dropped_capacity"] or metrics["dropped_identity"]:
        raise QwenDirectDecodeError("access trace recorder dropped operations")
    trace = recorder.snapshot()
    trace.verify()
    output = _atomic_bytes(path, trace.to_bytes())
    return {
        "inventory_fingerprint": trace.inventory_fingerprint,
        "leaves": metrics["leaves"],
        "operations": metrics["operations"],
        "path": str(output),
        "sha256": trace.sha256,
    }


def _checkpoint(model: StreamedQwen38) -> dict[str, str]:
    source = model.pager.source
    source.inventory()
    metrics = source.metrics()
    return {
        "inventory_fingerprint": str(metrics["inventory_source_fingerprint"]),
        "repo_id": str(metrics["repo_id"]),
        "revision": str(metrics["revision"]),
    }


def _delta_probe_receipt(
    args: argparse.Namespace,
    recorder: DeltaNetProbeRecorder | None,
    *,
    checkpoint: Mapping[str, Any],
    context_mode: str,
    start_pos: int,
    end_pos: int,
    inputs: Mapping[str, Any],
    hidden_sha256: str,
) -> dict[str, Any] | None:
    path = getattr(args, "delta_probe", None)
    if path is None:
        if recorder is not None:  # pragma: no cover - internal contract.
            raise QwenDirectDecodeError("unused DeltaNet probe recorder")
        return None
    if recorder is None:  # pragma: no cover - internal contract.
        raise QwenDirectDecodeError("DeltaNet probe recorder is missing")
    document = build_probe_document(
        recorder,
        checkpoint=checkpoint,
        context_mode=context_mode,
        start_pos=start_pos,
        end_pos=end_pos,
        item_id=str(inputs["item_id"]),
        input_sha256=str(inputs["sha256"]),
        hidden_sha256=hidden_sha256,
    )
    output = _atomic_bytes(path, _canonical(document) + b"\n")
    return {
        "path": str(output),
        "records": len(document["body"]["records"]),
        "schema": DELTANET_PROBE_SCHEMA,
        "sha256": document["sha256"],
    }


def _cleanup(runtime: RuntimeSource, model: StreamedQwen38) -> None:
    active_error = sys.exc_info()[1]
    cleanup_error: Exception | None = None
    for action in (
        lambda: model.reset_state(release=True),
        model.pager.close,
        runtime.close,
    ):
        try:
            action()
        except Exception as exc:
            if cleanup_error is None:
                cleanup_error = exc
    if active_error is None and cleanup_error is not None:
        raise cleanup_error


def prepare_prefix(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _load_input(args.input)
    prefix = inputs["prefix_token_ids"]
    if len(prefix) >= args.max_seq_len:
        raise QwenDirectDecodeError("prefix leaves no room for decode")
    recorder = AccessTraceRecorder()
    delta_probe = DeltaNetProbeRecorder() if args.delta_probe is not None else None
    runtime, model = _runtime(args, recorder, delta_probe)
    try:
        if max((*prefix, int(inputs["decode_token_id"]))) >= model.config.vocab_size:
            raise QwenDirectDecodeError("input token exceeds checkpoint vocabulary")
        hidden, evidence = model.prefill(
            [prefix], tokenwise=False, reset=True, progress=_progress
        )
        snapshot = model.save_state(args.snapshot, transport_neutral=True)
        trace = _trace_receipt(recorder, args.access_trace)
        checkpoint = _checkpoint(model)
        hidden_sha256 = _hidden_sha256(hidden)
        probe_receipt = _delta_probe_receipt(
            args,
            delta_probe,
            checkpoint=checkpoint,
            context_mode="prefill",
            start_pos=0,
            end_pos=len(prefix),
            inputs=inputs,
            hidden_sha256=hidden_sha256,
        )
        identity = {
            "checkpoint": checkpoint,
            "evidence": [asdict(row) for row in evidence],
            "hidden_dtype": str(hidden.dtype).removeprefix("torch."),
            "hidden_shape": list(hidden.shape),
            "hidden_sha256": hidden_sha256,
            "input_sha256": inputs["sha256"],
            "pager": model.pager.metrics(),
            "schema": PREFIX_SCHEMA,
            "snapshot": snapshot,
            "source_verification": runtime.verification,
            "trace": trace,
        }
        if probe_receipt is not None:
            identity["delta_probe"] = probe_receipt
        return _result(identity)
    finally:
        _cleanup(runtime, model)


def decode_arm(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _load_input(args.input)
    prefix_result = _load_result(args.prefix_result, PREFIX_SCHEMA)
    if prefix_result["input_sha256"] != inputs["sha256"]:
        raise QwenDirectDecodeError("prefix result belongs to another input")
    snapshot_path = Path(args.snapshot).expanduser().resolve()
    if Path(prefix_result["snapshot"]["manifest"]).resolve() != snapshot_path:
        raise QwenDirectDecodeError("prefix result belongs to another snapshot")
    prefix = inputs["prefix_token_ids"]
    recorder = AccessTraceRecorder()
    delta_probe = DeltaNetProbeRecorder() if args.delta_probe is not None else None
    runtime, model = _runtime(args, recorder, delta_probe)
    try:
        restore_started = time.perf_counter()
        restored = model.load_state(args.snapshot, transport_neutral=True)
        if restored["payload_sha256"] != prefix_result["snapshot"]["payload_sha256"]:
            raise QwenDirectDecodeError("restored snapshot payload differs from prefix")
        restore_seconds = time.perf_counter() - restore_started
        if model.next_position != len(prefix):
            raise QwenDirectDecodeError("snapshot cursor differs from input prefix")
        checkpoint = _checkpoint(model)
        if checkpoint != prefix_result["checkpoint"]:
            raise QwenDirectDecodeError("prefix/decode checkpoint identity differs")
        hidden, evidence = model.decode(
            [[int(inputs["decode_token_id"])]], progress=_progress
        )
        trace = _trace_receipt(recorder, args.access_trace)
        hidden_sha256 = _hidden_sha256(hidden)
        probe_receipt = _delta_probe_receipt(
            args,
            delta_probe,
            checkpoint=checkpoint,
            context_mode="decode",
            start_pos=len(prefix),
            end_pos=len(prefix) + 1,
            inputs=inputs,
            hidden_sha256=hidden_sha256,
        )
        identity = {
            "checkpoint": checkpoint,
            "evidence": asdict(evidence),
            "hidden_dtype": str(hidden.dtype).removeprefix("torch."),
            "hidden_shape": list(hidden.shape),
            "hidden_sha256": hidden_sha256,
            "input_sha256": inputs["sha256"],
            "pager": model.pager.metrics(),
            "restore_seconds": restore_seconds,
            "schema": DECODE_SCHEMA,
            "snapshot": restored,
            "source_verification": runtime.verification,
            "trace": trace,
        }
        if probe_receipt is not None:
            identity["delta_probe"] = probe_receipt
        return _result(identity)
    finally:
        _cleanup(runtime, model)


def _load_result(path: str, schema: str) -> dict[str, Any]:
    document = _strict_json(path)
    if document.get("schema") != schema or not isinstance(document.get("sha256"), str):
        raise QwenDirectDecodeError("benchmark result schema is invalid")
    identity = {key: value for key, value in document.items() if key != "sha256"}
    if document["sha256"] != _sha256(identity):
        raise QwenDirectDecodeError("benchmark result SHA-256 mismatch")
    return document


def _verified_delta_probe(
    result: Mapping[str, Any],
) -> dict[str, Any] | None:
    receipt = result.get("delta_probe")
    if receipt is None:
        return None
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "path",
        "records",
        "schema",
        "sha256",
    }:
        raise QwenDirectDecodeError("DeltaNet probe receipt is invalid")
    if receipt.get("schema") != DELTANET_PROBE_SCHEMA:
        raise QwenDirectDecodeError("DeltaNet probe receipt schema is invalid")
    try:
        document = verify_probe_document(_strict_json(str(receipt["path"])))
    except Exception as exc:
        raise QwenDirectDecodeError("DeltaNet probe artifact is invalid") from exc
    body = document["body"]
    if (
        document["sha256"] != receipt.get("sha256")
        or len(body["records"]) != receipt.get("records")
        or body["checkpoint"] != result.get("checkpoint")
        or body["input_sha256"] != result.get("input_sha256")
        or body["hidden_sha256"] != result.get("hidden_sha256")
        or body["context_mode"] != "decode"
    ):
        raise QwenDirectDecodeError("DeltaNet probe/result identity differs")
    return document


def compare(args: argparse.Namespace) -> dict[str, Any]:
    remote = _load_result(args.remote, DECODE_SCHEMA)
    local = _load_result(args.local, DECODE_SCHEMA)
    for key in ("hidden_dtype", "hidden_shape", "hidden_sha256", "input_sha256"):
        if remote[key] != local[key]:
            raise QwenDirectDecodeError(f"remote/local {key} differs")
    for key in ("inventory_fingerprint", "repo_id", "revision"):
        if remote["checkpoint"][key] != local["checkpoint"][key]:
            raise QwenDirectDecodeError(f"remote/local checkpoint {key} differs")
    if remote["snapshot"]["payload_sha256"] != local["snapshot"]["payload_sha256"]:
        raise QwenDirectDecodeError("remote/local snapshot payload differs")
    remote_probe = _verified_delta_probe(remote)
    local_probe = _verified_delta_probe(local)
    if (remote_probe is None) != (local_probe is None):
        raise QwenDirectDecodeError("remote/local DeltaNet instrumentation differs")
    probe_sha256 = None
    if remote_probe is not None and local_probe is not None:
        if remote_probe["sha256"] != local_probe["sha256"]:
            raise QwenDirectDecodeError("remote/local DeltaNet probes differ")
        probe_sha256 = remote_probe["sha256"]
    remote_seconds = float(remote["evidence"]["seconds"])
    local_seconds = float(local["evidence"]["seconds"])
    if (
        not math.isfinite(remote_seconds)
        or not math.isfinite(local_seconds)
        or remote_seconds <= 0.0
        or local_seconds <= 0.0
    ):
        raise QwenDirectDecodeError("remote/local decode seconds are invalid")
    identity = {
        "hidden_sha256": remote["hidden_sha256"],
        "hidden_dtype": remote["hidden_dtype"],
        "hidden_shape": remote["hidden_shape"],
        "input_sha256": remote["input_sha256"],
        "delta_probe_sha256": probe_sha256,
        "local": {
            "result_sha256": local["sha256"],
            "seconds": local_seconds,
            "source_body_bytes": local["evidence"]["source_body_bytes"],
        },
        "remote": {
            "result_sha256": remote["sha256"],
            "seconds": remote_seconds,
            "source_body_bytes": remote["evidence"]["source_body_bytes"],
        },
        "schema": COMPARISON_SCHEMA,
        "seconds_ratio_local_over_remote": local_seconds / remote_seconds,
        "speedup_remote_over_local": remote_seconds / local_seconds,
    }
    return _result(identity)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--access-trace", required=True)
    parser.add_argument("--delta-probe")
    parser.add_argument("--source", default=OFFICIAL_REPO_ID)
    parser.add_argument("--causal-bundle")
    parser.add_argument("--logical-repo-id", default=OFFICIAL_REPO_ID)
    parser.add_argument("--revision", default=OFFICIAL_REVISION)
    parser.add_argument("--pinned-inventory")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--max-cache-gb", type=_positive_float, default=1.0)
    parser.add_argument("--source-budget-mb", type=_positive_int, default=65536)
    parser.add_argument("--max-resident-mb", type=_positive_int, default=384)
    parser.add_argument("--max-seq-len", type=_positive_int, default=256)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select-input")
    select.add_argument("--inputs", required=True)
    select.add_argument("--item-id", required=True)
    select.add_argument("--output", required=True)
    select.set_defaults(handler=select_input)
    prepare = subparsers.add_parser("prepare-prefix")
    _runtime_arguments(prepare)
    prepare.set_defaults(handler=prepare_prefix)
    decode = subparsers.add_parser("decode")
    _runtime_arguments(decode)
    decode.add_argument("--prefix-result", required=True)
    decode.set_defaults(handler=decode_arm)
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--remote", required=True)
    comparison.add_argument("--local", required=True)
    comparison.add_argument("--output", required=True)
    comparison.set_defaults(handler=compare)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = args.handler(args)
        _write_json(args.output, document)
    except QwenDirectDecodeError as exc:
        raise SystemExit(f"Qwen direct decode failed: {exc}") from exc
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "schema": document["schema"],
                "sha256": document["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
