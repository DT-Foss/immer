#!/usr/bin/env python3
"""Pack fixed Qwen MLP pilot weights into one contiguous local shard per layer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
from typing import Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import save_file
import torch

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.mlp_pilot_residual import MlpPilotAffineFit
from immer.runtimes.ooe.mlp_pilot_router import MlpPilotRouterFit
from immer.runtimes.ooe.mlp_pilot_runtime import (
    MlpPilotSparseExecutor,
    MlpPilotTransposeManifest,
)

from qwen38_mlp_pilot_router import _digest, _persist_exact, _stable_read


MANIFEST_NAME = "pilot-weight-manifest.json"
INDEX_NAME = "model.safetensors.index.json"
MANIFEST_SCHEMA = "immer.qwen3.8-mlp-pilot-weight-bank/v1"


class CliError(RuntimeError):
    pass


def _raw_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def _publish_shard(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise CliError(f"pilot shard cannot be a symlink: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            if set(reader.keys()) != set(tensors) or any(
                not torch.equal(reader.get_tensor(name), tensor)
                for name, tensor in tensors.items()
            ):
                raise CliError(f"pilot shard changed: {path}")
        return
    temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
    save_file(
        dict(tensors),
        str(temporary),
        metadata={"format": "pt", "kind": MANIFEST_SCHEMA},
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
            if set(reader.keys()) != set(tensors) or any(
                not torch.equal(reader.get_tensor(name), tensor)
                for name, tensor in tensors.items()
            ):
                raise CliError(f"pilot shard collided: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _source_tensor(
    weights_root: Path, weight_map: Mapping[str, object], name: str
) -> torch.Tensor:
    shard = weight_map.get(name)
    if not isinstance(shard, str) or Path(shard).name != shard:
        raise CliError(f"source index lacks a safe shard for {name}")
    with safe_open(str(weights_root / shard), framework="pt", device="cpu") as reader:
        return reader.get_tensor(name)


def run(args: argparse.Namespace) -> dict[str, object]:
    analysis = Path(args.analysis_root).expanduser().absolute()
    weights_root = Path(args.weights_root).expanduser().absolute()
    transpose_root = Path(args.transpose_root).expanduser().absolute()
    output = Path(args.output_root).expanduser().absolute()
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    router_fit = MlpPilotRouterFit.from_bytes(_stable_read(analysis / "fit.json"))
    affine_fit = MlpPilotAffineFit.from_bytes(
        _stable_read(analysis / "affine-fit.json")
    )
    transpose_manifest = MlpPilotTransposeManifest.from_bytes(
        _stable_read(transpose_root / "pilot-transpose-manifest.json")
    )
    if (
        affine_fit.router_fit_sha256 != router_fit.sha256
        or transpose_manifest.router_fit_sha256 != router_fit.sha256
        or transpose_manifest.affine_fit_sha256 != affine_fit.sha256
    ):
        raise CliError("pilot source artifacts disagree")
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
    entries = []
    derived_map = {}
    total_payload_bytes = 0
    maximum_shard_bytes = int(args.max_shard_mb * 1024**2)
    affine_by_layer = {row.layer: row for row in affine_fit.models}
    for model in router_fit.models:
        layer = model.layer
        base = f"model.language_model.layers.{layer}.mlp"
        pilots = torch.from_numpy(model.pilot_neuron_indices())
        gate = _source_tensor(weights_root, weight_map, f"{base}.gate_proj.weight")
        up = _source_tensor(weights_root, weight_map, f"{base}.up_proj.weight")
        down = _source_tensor(weights_root, weight_map, f"{base}.down_proj.weight")
        hidden_dimension = affine_by_layer[layer].output_dimension
        expected_gate = (model.intermediate_dimension, hidden_dimension)
        if (
            gate.dtype != torch.bfloat16
            or up.dtype != torch.bfloat16
            or down.dtype != torch.bfloat16
            or tuple(gate.shape) != expected_gate
            or tuple(up.shape) != expected_gate
            or tuple(down.shape) != (hidden_dimension, model.intermediate_dimension)
        ):
            raise CliError(f"pilot source ABI changed at layer {layer}")
        tensors = {
            MlpPilotSparseExecutor.pilot_name(layer, "gate"): gate[pilots].contiguous(),
            MlpPilotSparseExecutor.pilot_name(layer, "up"): up[pilots].contiguous(),
            MlpPilotSparseExecutor.pilot_name(
                layer, "down_transpose"
            ): down.T.contiguous()[pilots].contiguous(),
        }
        shard = f"pilot-weights-layer-{layer:02d}.safetensors"
        shard_path = output / shard
        _publish_shard(shard_path, tensors)
        if shard_path.stat().st_size > maximum_shard_bytes:
            raise CliError(f"pilot shard exceeds its bound: {shard_path}")
        raw = _stable_read(shard_path, maximum_shard_bytes)
        entries.append(
            {
                "layer": layer,
                "pilot_indices_sha256": hashlib.sha256(
                    canonical_json_bytes(model.pilot_neuron_indices().tolist())
                ).hexdigest(),
                "shard": shard,
                "shard_bytes": len(raw),
                "shard_sha256": hashlib.sha256(raw).hexdigest(),
                "tensors": [
                    {
                        "name": name,
                        "raw_sha256": _raw_sha256(tensor),
                        "shape": list(tensor.shape),
                    }
                    for name, tensor in sorted(tensors.items())
                ],
            }
        )
        for name, tensor in tensors.items():
            derived_map[name] = shard
            total_payload_bytes += tensor.numel() * tensor.element_size()
        del gate, up, down, tensors
    body = {
        "affine_fit_sha256": affine_fit.sha256,
        "entries": entries,
        "model_pin_sha256": router_fit.model_pin_sha256,
        "router_fit_sha256": router_fit.sha256,
        "tensor_payload_bytes": total_payload_bytes,
        "transpose_manifest_sha256": transpose_manifest.sha256,
        "weights_index_sha256": affine_fit.weights_index_sha256,
    }
    manifest = {
        "body": body,
        "body_sha256": _digest(body),
        "schema": MANIFEST_SCHEMA,
    }
    total_shard_bytes = sum(entry["shard_bytes"] for entry in entries)
    if total_shard_bytes > int(args.max_total_gb * 1024**3):
        raise CliError("pilot bank exceeds its total byte bound")
    index = {
        "metadata": {"total_size": total_payload_bytes},
        "weight_map": dict(sorted(derived_map.items())),
    }
    _persist_exact(output / INDEX_NAME, canonical_json_bytes(index) + b"\n")
    _persist_exact(output / MANIFEST_NAME, canonical_json_bytes(manifest))
    return {
        "manifest_body_sha256": manifest["body_sha256"],
        "model_pin_sha256": router_fit.model_pin_sha256,
        "schema": "immer.qwen3.8-mlp-pilot-weight-bank-status/v1",
        "shard_bytes": total_shard_bytes,
        "tensor_payload_bytes": total_payload_bytes,
        "tensors": len(derived_map),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", required=True)
    parser.add_argument("--weights-root", required=True)
    parser.add_argument("--transpose-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-shard-mb", type=float, default=64.0)
    parser.add_argument("--max-total-gb", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if (
        not math.isfinite(args.max_shard_mb)
        or args.max_shard_mb <= 0.0
        or not math.isfinite(args.max_total_gb)
        or args.max_total_gb <= 0.0
    ):
        raise CliError("pilot byte bounds must be finite and positive")
    status = run(args)
    print(json.dumps(status, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
