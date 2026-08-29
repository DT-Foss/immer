#!/usr/bin/env python3
"""Build a complete p4/k32 Fast-MLP bank from a local causal Qwen bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Mapping, Sequence

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from immer.runtimes.deepseek_v4.causal_weights import (
    CausalWeightMount,
    LogicalModelIdentity,
)
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.mlp_pilot_router import (
    DEFAULT_RANDOM_SEED_SHA256,
    MlpPilotRouterConfig,
)
from immer.runtimes.ooe.mlp_pilot_runtime import (
    MlpPilotSparseExecutor,
    MlpPilotTransposeEntry,
    MlpPilotTransposeManifest,
)
from immer.runtimes.ooe.mlp_pilot_weight_only import (
    MlpPilotOnlineConfig,
    MlpPilotWeightOnlyPlan,
    build_weight_only_layer_model,
    gaussian_joint_moments,
    weight_only_model_pin,
)
from immer.runtimes.qwen3_8.bundle import verify_qwen38_causal_mount
from immer.runtimes.qwen3_8.config import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    Qwen38Config,
)
from immer.runtimes.qwen3_8.fast_mlp import (
    PILOT_WEIGHT_MANIFEST_V2_SCHEMA,
    WEIGHT_ONLY_PLAN_NAME,
    Qwen38FastMlpPaths,
)
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

INDEX_NAME = "model.safetensors.index.json"
TRANSPOSE_MANIFEST_NAME = "pilot-transpose-manifest.json"
PILOT_MANIFEST_NAME = "pilot-weight-manifest.json"
BUILD_REPORT_SCHEMA = "immer.qwen3.8-fast-mlp-weight-only-build-report/v1"


class CliError(RuntimeError):
    pass


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _plain_directory(path: Path) -> None:
    if path.exists() or path.is_symlink():
        row = path.lstat()
        if stat.S_ISLNK(row.st_mode) or not stat.S_ISDIR(row.st_mode):
            raise CliError(f"artifact root must be a non-symlink directory: {path}")
        return
    path.mkdir(parents=True, mode=0o700)


def _stable_read(path: Path, maximum: int = 64 * 1024 * 1024) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise CliError(f"invalid bounded input: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise CliError(f"short input read: {path}")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise CliError(f"input changed while read: {path}")
        return b"".join(chunks)
    except CliError:
        raise
    except OSError as exc:
        raise CliError(f"cannot read input: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _persist_exact(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or _stable_read(path, max(1, len(data))) != data:
            raise CliError(f"sealed artifact changed: {path}")
        return
    temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0)),
        0o600,
    )
    try:
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short artifact write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError:
        if path.is_symlink() or _stable_read(path, max(1, len(data))) != data:
            raise CliError(f"sealed artifact collided: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _raw_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(array).hexdigest()


def _file_sha256(path: Path, expected_size: int) -> str:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise CliError(f"shard identity changed: {path}")
        digest = hashlib.sha256()
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(4 * 1024 * 1024, remaining))
            if not chunk:
                raise CliError(f"short shard read: {path}")
            digest.update(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise CliError(f"shard changed while hashed: {path}")
        return digest.hexdigest()
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_shard(
    path: Path, tensors: Mapping[str, torch.Tensor], *, kind: str
) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink():
            raise CliError(f"shard cannot be a symlink: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as reader:
            if set(reader.keys()) != set(tensors) or any(
                not torch.equal(reader.get_tensor(name), tensor)
                for name, tensor in tensors.items()
            ):
                raise CliError(f"immutable shard changed: {path}")
        return
    temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
    save_file(dict(tensors), str(temporary), metadata={"format": "pt", "kind": kind})
    descriptor = os.open(
        temporary,
        os.O_RDONLY
        | int(getattr(os, "O_CLOEXEC", 0))
        | int(getattr(os, "O_NOFOLLOW", 0)),
    )
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
                raise CliError(f"immutable shard collided: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _layer_moments(
    pager: Qwen38WeightPager,
    *,
    layer: int,
    config: Qwen38Config,
    router_config: MlpPilotRouterConfig,
    chunk_rows: int,
) -> np.ndarray:
    base = f"model.language_model.layers.{layer}.mlp"
    moments = np.empty(config.intermediate_size, dtype=np.float64)
    for start in range(0, config.intermediate_size, chunk_rows):
        stop = min(config.intermediate_size, start + chunk_rows)
        ids = tuple(range(start, stop))
        gate = pager.tensor_rows(f"{base}.gate_proj.weight", ids)
        up = pager.tensor_rows(f"{base}.up_proj.weight", ids)
        try:
            if (
                gate.dtype != torch.bfloat16
                or up.dtype != torch.bfloat16
                or tuple(gate.shape) != (len(ids), config.dim)
                or tuple(up.shape) != tuple(gate.shape)
            ):
                raise CliError(f"Gate/Up ABI changed at layer {layer}")
            moments[start:stop] = gaussian_joint_moments(gate, up)
        finally:
            del gate, up
    return moments.reshape(-1, router_config.block_size)


def _source_layer_sha256(
    *,
    layer: int,
    moments: np.ndarray,
    bundle_manifest_sha256: str,
    weights_index_sha256: str,
) -> str:
    raw = np.asarray(moments, dtype="<f8", order="C").tobytes(order="C")
    return _digest(
        {
            "bundle_manifest_sha256": bundle_manifest_sha256,
            "initializer_moment_sha256": hashlib.sha256(raw).hexdigest(),
            "layer": layer,
            "schema": "immer.qwen-mlp-pilot-weight-only-layer-source/v1",
            "weights_index_sha256": weights_index_sha256,
        }
    )


def _router_config(args: argparse.Namespace) -> MlpPilotRouterConfig:
    return MlpPilotRouterConfig(
        block_size=args.block_size,
        pilot_count=args.pilot_count,
        selected_block_count=args.selected_block_count,
        ridge=1e-6,
        random_seed_sha256=args.random_seed_sha256,
        max_working_bytes=int(args.max_working_gb * 1024**3),
    )


def _online_config(args: argparse.Namespace) -> MlpPilotOnlineConfig:
    return MlpPilotOnlineConfig(
        min_confirmed_rows=args.min_confirmed_rows,
        min_capture=args.min_capture,
        confirmation_interval=args.confirmation_interval,
        cold_start_sparse_waves=args.cold_start_sparse_waves,
        statistics_decay=args.statistics_decay,
        confidence_decay=args.confidence_decay,
        max_sparse_rows=args.max_sparse_rows,
    )


def _build_plan(
    *,
    pager: Qwen38WeightPager,
    model_identity: LogicalModelIdentity,
    bundle: Mapping[str, object],
    config: Qwen38Config,
    router_config: MlpPilotRouterConfig,
    online_config: MlpPilotOnlineConfig,
    weights_index_sha256: str,
    chunk_rows: int,
) -> MlpPilotWeightOnlyPlan:
    models = []
    sources = []
    for layer in range(config.n_layers):
        moments = _layer_moments(
            pager,
            layer=layer,
            config=config,
            router_config=router_config,
            chunk_rows=chunk_rows,
        )
        source_sha256 = _source_layer_sha256(
            layer=layer,
            moments=moments,
            bundle_manifest_sha256=str(bundle["manifest_sha256"]),
            weights_index_sha256=weights_index_sha256,
        )
        models.append(
            build_weight_only_layer_model(
                layer=layer,
                moments=moments,
                config=router_config,
                source_layer_sha256=source_sha256,
            )
        )
        sources.append((layer, source_sha256))
    model_pin_sha256 = weight_only_model_pin(
        repo_id=model_identity.repo_id,
        revision=model_identity.revision,
        bundle_manifest_sha256=str(bundle["manifest_sha256"]),
        layout_fingerprint=str(bundle["layout_fingerprint"]),
        weights_index_sha256=weights_index_sha256,
    )
    return MlpPilotWeightOnlyPlan(
        repo_id=model_identity.repo_id,
        revision=model_identity.revision,
        model_pin_sha256=model_pin_sha256,
        bundle_manifest_sha256=str(bundle["manifest_sha256"]),
        layout_fingerprint=str(bundle["layout_fingerprint"]),
        weights_index_sha256=weights_index_sha256,
        hidden_dimension=config.dim,
        n_layers=config.n_layers,
        config=router_config,
        online_config=online_config,
        models=tuple(models),
        source_layer_sha256s=tuple(sources),
    )


def _build_weight_banks(
    *,
    pager: Qwen38WeightPager,
    plan: MlpPilotWeightOnlyPlan,
    paths: Qwen38FastMlpPaths,
    max_transpose_shard_bytes: int,
    max_transpose_total_bytes: int,
    max_pilot_shard_bytes: int,
    max_pilot_total_bytes: int,
) -> tuple[MlpPilotTransposeManifest, Mapping[str, object]]:
    transpose_entries = []
    transpose_map: dict[str, str] = {}
    pilot_entries = []
    pilot_map: dict[str, str] = {}
    transpose_payload_bytes = 0
    pilot_payload_bytes = 0
    affine = plan.affine_fit
    for model in plan.models:
        layer = model.layer
        base = f"model.language_model.layers.{layer}.mlp"
        source_name = f"{base}.down_proj.weight"
        down = pager.tensor_torch(source_name, dtype=torch.bfloat16, device="cpu")
        expected = (plan.hidden_dimension, model.intermediate_dimension)
        if down.dtype != torch.bfloat16 or tuple(down.shape) != expected:
            raise CliError(f"Down projection ABI changed at layer {layer}")
        transpose = down.T.contiguous()
        transpose_name = MlpPilotSparseExecutor.transpose_name(layer)
        transpose_shard = f"pilot-down-transpose-layer-{layer:02d}.safetensors"
        transpose_path = paths.transpose_root / transpose_shard
        _publish_shard(
            transpose_path,
            {transpose_name: transpose},
            kind="immer-qwen-mlp-down-transpose/v1",
        )
        transpose_size = transpose_path.stat().st_size
        if transpose_size > max_transpose_shard_bytes:
            raise CliError(f"transpose shard exceeds its bound: {transpose_path}")
        transpose_entries.append(
            MlpPilotTransposeEntry(
                layer=layer,
                source_tensor=source_name,
                transpose_tensor=transpose_name,
                shard=transpose_shard,
                shape=tuple(transpose.shape),
                source_raw_sha256=_raw_sha256(down),
                transpose_raw_sha256=_raw_sha256(transpose),
                shard_sha256=_file_sha256(transpose_path, transpose_size),
                shard_bytes=transpose_size,
            )
        )
        transpose_map[transpose_name] = transpose_shard
        transpose_payload_bytes += transpose.numel() * transpose.element_size()

        pilot_ids = tuple(int(row) for row in model.pilot_neuron_indices())
        gate_pilot = pager.tensor_rows(f"{base}.gate_proj.weight", pilot_ids)
        up_pilot = pager.tensor_rows(f"{base}.up_proj.weight", pilot_ids)
        down_pilot = transpose.index_select(
            0, torch.tensor(pilot_ids, dtype=torch.long)
        ).contiguous()
        tensors = {
            MlpPilotSparseExecutor.pilot_name(layer, "gate"): gate_pilot.contiguous(),
            MlpPilotSparseExecutor.pilot_name(layer, "up"): up_pilot.contiguous(),
            MlpPilotSparseExecutor.pilot_name(layer, "down_transpose"): down_pilot,
        }
        pilot_shard = f"pilot-weights-layer-{layer:02d}.safetensors"
        pilot_path = paths.pilot_root / pilot_shard
        _publish_shard(
            pilot_path,
            tensors,
            kind=PILOT_WEIGHT_MANIFEST_V2_SCHEMA,
        )
        pilot_size = pilot_path.stat().st_size
        if pilot_size > max_pilot_shard_bytes:
            raise CliError(f"pilot shard exceeds its bound: {pilot_path}")
        pilot_entries.append(
            {
                "layer": layer,
                "pilot_indices_sha256": hashlib.sha256(
                    canonical_json_bytes(list(pilot_ids))
                ).hexdigest(),
                "shard": pilot_shard,
                "shard_bytes": pilot_size,
                "shard_sha256": _file_sha256(pilot_path, pilot_size),
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
            pilot_map[name] = pilot_shard
            pilot_payload_bytes += tensor.numel() * tensor.element_size()
        del down, transpose, gate_pilot, up_pilot, down_pilot, tensors

    transpose_manifest = MlpPilotTransposeManifest(
        model_pin_sha256=plan.model_pin_sha256,
        router_fit_sha256=plan.sha256,
        affine_fit_sha256=affine.sha256,
        weights_index_sha256=plan.weights_index_sha256,
        entries=tuple(transpose_entries),
    )
    if transpose_manifest.total_shard_bytes > max_transpose_total_bytes:
        raise CliError("transpose bank exceeds its total byte bound")
    pilot_total_bytes = sum(int(row["shard_bytes"]) for row in pilot_entries)
    if pilot_total_bytes > max_pilot_total_bytes:
        raise CliError("pilot bank exceeds its total byte bound")

    transpose_index = {
        "metadata": {"total_size": transpose_payload_bytes},
        "weight_map": dict(sorted(transpose_map.items())),
    }
    pilot_index = {
        "metadata": {"total_size": pilot_payload_bytes},
        "weight_map": dict(sorted(pilot_map.items())),
    }
    pilot_body = {
        "entries": pilot_entries,
        "identity_affine_sha256": affine.sha256,
        "model_pin_sha256": plan.model_pin_sha256,
        "tensor_payload_bytes": pilot_payload_bytes,
        "transpose_manifest_sha256": transpose_manifest.sha256,
        "weight_plan_sha256": plan.sha256,
        "weights_index_sha256": plan.weights_index_sha256,
    }
    pilot_manifest = {
        "body": pilot_body,
        "body_sha256": _digest(pilot_body),
        "schema": PILOT_WEIGHT_MANIFEST_V2_SCHEMA,
    }
    _persist_exact(
        paths.transpose_root / INDEX_NAME,
        canonical_json_bytes(transpose_index) + b"\n",
    )
    _persist_exact(
        paths.transpose_root / TRANSPOSE_MANIFEST_NAME,
        transpose_manifest.to_bytes(),
    )
    _persist_exact(
        paths.pilot_root / INDEX_NAME,
        canonical_json_bytes(pilot_index) + b"\n",
    )
    _persist_exact(
        paths.pilot_root / PILOT_MANIFEST_NAME,
        canonical_json_bytes(pilot_manifest),
    )
    return transpose_manifest, pilot_manifest


def run(args: argparse.Namespace) -> dict[str, object]:
    bundle_root = Path(args.bundle).expanduser().absolute()
    output_root = Path(args.output).expanduser().absolute()
    paths = Qwen38FastMlpPaths.from_root(output_root)
    for path in (
        output_root,
        paths.analysis_root,
        paths.transpose_root,
        paths.pilot_root,
    ):
        _plain_directory(path)
    router_config = _router_config(args)
    online_config = _online_config(args)
    if args.moment_chunk_rows % router_config.block_size:
        raise CliError("--moment-chunk-rows must be block aligned")
    identity = LogicalModelIdentity(args.repo_id, args.revision)
    with CausalWeightMount(
        bundle_root,
        identity,
        budget_mb=args.source_budget_mb,
    ) as mount:
        bundle = verify_qwen38_causal_mount(
            mount,
            require_official_config=not args.allow_nonofficial_config,
        )
        config = Qwen38Config.from_file(
            mount.weights_root / "config.json",
            require_official=not args.allow_nonofficial_config,
        )
        if config.intermediate_size % router_config.block_size:
            raise CliError("checkpoint intermediate dimension is not block aligned")
        if router_config.selected_block_count >= (
            config.intermediate_size // router_config.block_size
        ):
            raise CliError("selected p4/k32 blocks exhaust the checkpoint MLP")
        estimated_working_bytes = max(
            args.moment_chunk_rows * config.dim * 32,
            config.dim * config.intermediate_size * 4,
        )
        if estimated_working_bytes > router_config.max_working_bytes:
            raise CliError("weight-only build exceeds --max-working-gb")
        index_raw = _stable_read(mount.weights_root / INDEX_NAME)
        index_sha256 = hashlib.sha256(index_raw).hexdigest()
        pager = Qwen38WeightPager(
            mount.source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=args.max_resident_mb * 1024**2,
            close_source=False,
            require_source_identity=True,
            causal_tensor_reader=mount.tensor_reader,
        )
        try:
            plan = _build_plan(
                pager=pager,
                model_identity=identity,
                bundle=bundle,
                config=config,
                router_config=router_config,
                online_config=online_config,
                weights_index_sha256=index_sha256,
                chunk_rows=args.moment_chunk_rows,
            )
            _persist_exact(paths.analysis_root / WEIGHT_ONLY_PLAN_NAME, plan.to_bytes())
            transpose, pilot = _build_weight_banks(
                pager=pager,
                plan=plan,
                paths=paths,
                max_transpose_shard_bytes=int(args.max_transpose_shard_gb * 1024**3),
                max_transpose_total_bytes=int(args.max_transpose_total_gb * 1024**3),
                max_pilot_shard_bytes=int(args.max_pilot_shard_mb * 1024**2),
                max_pilot_total_bytes=int(args.max_pilot_total_gb * 1024**3),
            )
            selected_fraction = plan.models[0].selected_neuron_count / (
                plan.models[0].intermediate_dimension
            )
            transport_fraction = (
                plan.models[0].block_count * plan.models[0].pilot_count
                + plan.models[0].selected_block_count * plan.models[0].block_size
            ) / plan.models[0].intermediate_dimension
            report = {
                "active_layers": list(range(config.n_layers)),
                "bundle": bundle,
                "cold_start": "weight-only/no-forward",
                "online_config_sha256": online_config.sha256,
                "pager": pager.metrics(),
                "pilot_manifest_body_sha256": pilot["body_sha256"],
                "schema": BUILD_REPORT_SCHEMA,
                "selected_neuron_fraction": selected_fraction,
                "transpose_manifest_sha256": transpose.sha256,
                "transport_row_fraction": transport_fraction,
                "weight_plan_sha256": plan.sha256,
            }
            return report
        finally:
            pager.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo-id", default=OFFICIAL_REPO_ID)
    parser.add_argument("--revision", default=OFFICIAL_REVISION)
    parser.add_argument("--allow-nonofficial-config", action="store_true")
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--pilot-count", type=int, default=4)
    parser.add_argument("--selected-block-count", type=int, default=32)
    parser.add_argument("--random-seed-sha256", default=DEFAULT_RANDOM_SEED_SHA256)
    parser.add_argument("--moment-chunk-rows", type=int, default=1024)
    parser.add_argument("--min-confirmed-rows", type=int, default=8)
    parser.add_argument("--min-capture", type=float, default=0.25)
    parser.add_argument("--confirmation-interval", type=int, default=32)
    parser.add_argument("--cold-start-sparse-waves", type=int, default=0)
    parser.add_argument("--statistics-decay", type=float, default=0.99)
    parser.add_argument("--confidence-decay", type=float, default=0.80)
    parser.add_argument("--max-sparse-rows", type=int, default=16)
    parser.add_argument("--source-budget-mb", type=float, default=8192.0)
    parser.add_argument("--max-resident-mb", type=int, default=384)
    parser.add_argument("--max-working-gb", type=float, default=1.0)
    parser.add_argument("--max-transpose-shard-gb", type=float, default=1.0)
    parser.add_argument("--max-transpose-total-gb", type=float, default=16.0)
    parser.add_argument("--max-pilot-shard-mb", type=float, default=64.0)
    parser.add_argument("--max-pilot-total-gb", type=float, default=4.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    positive_floats = (
        args.source_budget_mb,
        args.max_working_gb,
        args.max_transpose_shard_gb,
        args.max_transpose_total_gb,
        args.max_pilot_shard_mb,
        args.max_pilot_total_gb,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive_floats):
        raise CliError("builder byte bounds must be finite and positive")
    if args.max_resident_mb < 1 or args.moment_chunk_rows < 1:
        raise CliError("builder row and resident bounds must be positive")
    report = run(args)
    print(json.dumps(report, allow_nan=False, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
