#!/usr/bin/env python3
"""Build the local row-addressable transpose bank for sparse Qwen down projections."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
from typing import Sequence

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.mlp_pilot_residual import MlpPilotAffineFit
from immer.runtimes.ooe.mlp_pilot_router import MlpPilotRouterFit
from immer.runtimes.ooe.mlp_pilot_runtime import (
    MlpPilotSparseExecutor,
    MlpPilotTransposeEntry,
    MlpPilotTransposeManifest,
)

from qwen38_mlp_pilot_router import _persist_exact, _stable_read


MANIFEST_NAME = "pilot-transpose-manifest.json"
INDEX_NAME = "model.safetensors.index.json"


class CliError(RuntimeError):
    pass


def _raw_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path, maximum: int) -> str:
    return hashlib.sha256(_stable_read(path, maximum)).hexdigest()


def _publish_shard(path: Path, tensor_name: str, tensor: torch.Tensor) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise CliError(f"transpose shard cannot be a symlink: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            existing = reader.get_tensor(tensor_name)
        if not torch.equal(existing, tensor):
            raise CliError(f"transpose shard changed: {path}")
        return
    temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
    save_file(
        {tensor_name: tensor},
        str(temporary),
        metadata={"format": "pt", "kind": "immer-qwen-mlp-down-transpose/v1"},
    )
    descriptor = os.open(temporary, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(temporary, 0o444)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError:
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            existing = reader.get_tensor(tensor_name)
        if not torch.equal(existing, tensor):
            raise CliError(f"transpose shard collided: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, object]:
    analysis = Path(args.analysis_root).expanduser().absolute()
    weights_root = Path(args.weights_root).expanduser().absolute()
    output = Path(args.output_root).expanduser().absolute()
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    router_fit = MlpPilotRouterFit.from_bytes(_stable_read(analysis / "fit.json"))
    affine_fit = MlpPilotAffineFit.from_bytes(
        _stable_read(analysis / "affine-fit.json")
    )
    if (
        affine_fit.router_fit_sha256 != router_fit.sha256
        or affine_fit.model_pin_sha256 != router_fit.model_pin_sha256
    ):
        raise CliError("router and affine fits disagree")
    index_raw = _stable_read(weights_root / INDEX_NAME)
    if hashlib.sha256(index_raw).hexdigest() != affine_fit.weights_index_sha256:
        raise CliError("source weights index differs from the affine fit")
    try:
        source_index = json.loads(index_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError("source weights index is not JSON") from exc
    if not isinstance(source_index, dict) or not isinstance(
        source_index.get("weight_map"), dict
    ):
        raise CliError("source weights index has no tensor map")
    weight_map = source_index["weight_map"]
    affine_by_layer = {row.layer: row for row in affine_fit.models}
    entries = []
    derived_map = {}
    tensor_payload_bytes = 0
    maximum_shard_bytes = int(args.max_shard_gb * 1024**3)
    for model in router_fit.models:
        layer = model.layer
        source_name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
        source_shard = weight_map.get(source_name)
        if not isinstance(source_shard, str) or Path(source_shard).name != source_shard:
            raise CliError(f"source index lacks a safe shard for {source_name}")
        with safe_open(
            str(weights_root / source_shard), framework="pt", device="cpu"
        ) as reader:
            source = reader.get_tensor(source_name)
        expected = (
            affine_by_layer[layer].output_dimension,
            model.intermediate_dimension,
        )
        if source.dtype != torch.bfloat16 or tuple(source.shape) != expected:
            raise CliError(f"source down projection changed at layer {layer}")
        source_raw_sha256 = _raw_sha256(source)
        transpose = source.T.contiguous()
        tensor_name = MlpPilotSparseExecutor.transpose_name(layer)
        shard = f"pilot-down-transpose-layer-{layer:02d}.safetensors"
        shard_path = output / shard
        _publish_shard(shard_path, tensor_name, transpose)
        if shard_path.stat().st_size > maximum_shard_bytes:
            raise CliError(f"transpose shard exceeds its bound: {shard_path}")
        transpose_raw_sha256 = _raw_sha256(transpose)
        entries.append(
            MlpPilotTransposeEntry(
                layer=layer,
                source_tensor=source_name,
                transpose_tensor=tensor_name,
                shard=shard,
                shape=tuple(transpose.shape),
                source_raw_sha256=source_raw_sha256,
                transpose_raw_sha256=transpose_raw_sha256,
                shard_sha256=_file_sha256(shard_path, maximum_shard_bytes),
                shard_bytes=shard_path.stat().st_size,
            )
        )
        derived_map[tensor_name] = shard
        tensor_payload_bytes += transpose.numel() * transpose.element_size()
        del source, transpose
    manifest = MlpPilotTransposeManifest(
        model_pin_sha256=router_fit.model_pin_sha256,
        router_fit_sha256=router_fit.sha256,
        affine_fit_sha256=affine_fit.sha256,
        weights_index_sha256=affine_fit.weights_index_sha256,
        entries=tuple(entries),
    )
    if manifest.total_shard_bytes > int(args.max_total_gb * 1024**3):
        raise CliError("transpose bank exceeds its total byte bound")
    derived_index = {
        "metadata": {"total_size": tensor_payload_bytes},
        "weight_map": dict(sorted(derived_map.items())),
    }
    _persist_exact(output / INDEX_NAME, canonical_json_bytes(derived_index) + b"\n")
    _persist_exact(output / MANIFEST_NAME, manifest.to_bytes())
    return {
        "affine_fit_sha256": affine_fit.sha256,
        "manifest_sha256": manifest.sha256,
        "model_pin_sha256": manifest.model_pin_sha256,
        "schema": "immer.qwen3.8-mlp-pilot-transpose-bank-status/v1",
        "shard_bytes": manifest.total_shard_bytes,
        "tensor_payload_bytes": tensor_payload_bytes,
        "tensors": len(entries),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-shard-gb", type=float, default=1.0)
    parser.add_argument("--max-total-gb", type=float, default=4.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        not math.isfinite(args.max_shard_gb)
        or args.max_shard_gb <= 0.0
        or not math.isfinite(args.max_total_gb)
        or args.max_total_gb <= 0.0
    ):
        raise CliError("transpose byte bounds must be finite and positive")
    status = run(args)
    print(json.dumps(status, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
