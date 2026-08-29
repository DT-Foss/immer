#!/usr/bin/env python3
"""Build the tiny all-layer Fast-MLP route plan directly from packed Q4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import time
from typing import Mapping, cast

import numpy as np
import torch

from immer.runtimes.deepseek_v4.causal_weights import (
    CausalWeightMount,
    LogicalModelIdentity,
)
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.mlp_pilot_router import (
    DEFAULT_RANDOM_SEED_SHA256,
    MlpPilotLayerModel,
    MlpPilotRouterConfig,
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
    WEIGHT_ONLY_PLAN_NAME,
    Qwen38FastMlpPaths,
)
from immer.runtimes.qwen3_8.q4 import Q4Bank


STATE_SCHEMA = "immer.qwen3.8-q4-fast-mlp-plan-state/v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--q4", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--pilot-count", type=int, default=4)
    parser.add_argument("--selected-block-count", type=int, default=32)
    parser.add_argument("--row-chunk", type=int, default=1024)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--source-budget-mb", type=float, default=65_536)
    parser.add_argument("--random-seed-sha256", default=DEFAULT_RANDOM_SEED_SHA256)
    return parser


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | int(getattr(os, "O_CLOEXEC", 0)),
            0o600,
        )
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("short plan write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _layer_moments(
    bank: Q4Bank,
    *,
    layer: int,
    config: Qwen38Config,
    block_size: int,
    row_chunk: int,
) -> np.ndarray:
    base = f"model.language_model.layers.{layer}.mlp"
    result = np.empty(config.intermediate_size, dtype=np.float64)
    for start in range(0, config.intermediate_size, row_chunk):
        stop = min(config.intermediate_size, start + row_chunk)
        ids = tuple(range(start, stop))
        gate = bank.rows(f"{base}.gate_proj.weight", ids, dtype=torch.float32)
        up = bank.rows(f"{base}.up_proj.weight", ids, dtype=torch.float32)
        result[start:stop] = gaussian_joint_moments(gate, up)
    return result.reshape(-1, block_size)


def _state_identity(
    *,
    bank: Q4Bank,
    bundle: Mapping[str, object],
    weights_index_sha256: str,
    router: MlpPilotRouterConfig,
) -> dict[str, object]:
    return {
        "bundle_manifest_sha256": bundle["manifest_sha256"],
        "q4_manifest_sha256": bank.identity["manifest_sha256"],
        "router": router.to_record(),
        "weights_index_sha256": weights_index_sha256,
    }


def _load_state(
    path: Path,
    *,
    identity: Mapping[str, object],
) -> tuple[list[MlpPilotLayerModel], list[tuple[int, str]]]:
    if not path.exists():
        return [], []
    document = json.loads(path.read_bytes())
    if (
        not isinstance(document, dict)
        or document.get("schema") != STATE_SCHEMA
        or document.get("identity") != identity
        or not isinstance(document.get("models"), list)
        or not isinstance(document.get("sources"), list)
        or canonical_json_bytes(document) != path.read_bytes()
    ):
        raise ValueError("plan resume state differs from this Q4 bank")
    models = [MlpPilotLayerModel.from_record(row) for row in document["models"]]
    sources = []
    for row in document["sources"]:
        if not isinstance(row, dict) or set(row) != {"layer", "sha256"}:
            raise ValueError("plan resume source row is invalid")
        sources.append((int(row["layer"]), str(row["sha256"])))
    if [row.layer for row in models] != list(range(len(models))) or [
        layer for layer, _sha in sources
    ] != list(range(len(sources))):
        raise ValueError("plan resume layers are not contiguous")
    return models, sources


def _save_state(
    path: Path,
    *,
    identity: Mapping[str, object],
    models: list[MlpPilotLayerModel],
    sources: list[tuple[int, str]],
) -> None:
    _atomic_bytes(
        path,
        canonical_json_bytes(
            {
                "identity": dict(identity),
                "models": [row.to_record() for row in models],
                "schema": STATE_SCHEMA,
                "sources": [
                    {"layer": layer, "sha256": sha} for layer, sha in sources
                ],
            }
        ),
    )


def run(args: argparse.Namespace) -> int:
    if (
        args.block_size <= 0
        or args.pilot_count <= 0
        or args.selected_block_count <= 0
        or args.row_chunk <= 0
    ):
        raise ValueError("plan dimensions must be positive")
    mount = CausalWeightMount(
        args.bundle.expanduser().resolve(),
        LogicalModelIdentity(OFFICIAL_REPO_ID, OFFICIAL_REVISION),
        budget_mb=args.source_budget_mb,
    )
    bank: Q4Bank | None = None
    started = time.perf_counter()
    try:
        bundle = verify_qwen38_causal_mount(mount, require_official_config=True)
        config = Qwen38Config.from_file(
            mount.weights_root / "config.json", require_official=True
        )
        source_metrics = mount.source.metrics()
        fingerprint = source_metrics.get("inventory_source_fingerprint")
        if not isinstance(fingerprint, str):
            raise ValueError("Qwen source has no inventory fingerprint")
        bank = Q4Bank.load(
            args.q4.expanduser().resolve(),
            bundle_receipt=bundle,
            repo_id=OFFICIAL_REPO_ID,
            revision=OFFICIAL_REVISION,
            inventory_fingerprint=fingerprint,
            threads=args.threads,
        )
        router = MlpPilotRouterConfig(
            block_size=args.block_size,
            pilot_count=args.pilot_count,
            selected_block_count=args.selected_block_count,
            random_seed_sha256=args.random_seed_sha256,
            max_working_bytes=1024**3,
        )
        if config.intermediate_size % router.block_size:
            raise ValueError("Qwen intermediate width is not route-block aligned")
        index_sha = _sha256(mount.weights_root / "model.safetensors.index.json")
        paths = Qwen38FastMlpPaths.from_root(args.output)
        paths.analysis_root.mkdir(parents=True, exist_ok=True)
        target = paths.analysis_root / WEIGHT_ONLY_PLAN_NAME
        state_path = paths.analysis_root / ".q4-plan-state.json"
        identity = _state_identity(
            bank=bank,
            bundle=bundle,
            weights_index_sha256=index_sha,
            router=router,
        )
        models, sources = _load_state(state_path, identity=identity)
        for layer in range(len(models), config.n_layers):
            layer_started = time.perf_counter()
            moments = _layer_moments(
                bank,
                layer=layer,
                config=config,
                block_size=router.block_size,
                row_chunk=args.row_chunk,
            )
            moment_sha = hashlib.sha256(
                np.asarray(moments, dtype="<f8", order="C").tobytes(order="C")
            ).hexdigest()
            source_sha = hashlib.sha256(
                canonical_json_bytes(
                    {
                        "layer": layer,
                        "moment_sha256": moment_sha,
                        "q4_manifest_sha256": bank.identity["manifest_sha256"],
                        "schema": "immer.qwen3.8-q4-pilot-layer-source/v1",
                    }
                )
            ).hexdigest()
            models.append(
                build_weight_only_layer_model(
                    layer=layer,
                    moments=moments,
                    config=router,
                    source_layer_sha256=source_sha,
                )
            )
            sources.append((layer, source_sha))
            _save_state(
                state_path,
                identity=identity,
                models=models,
                sources=sources,
            )
            _print(
                {
                    "event": "q4_fast_mlp_layer_complete",
                    "layer": layer,
                    "layers": config.n_layers,
                    "seconds": time.perf_counter() - layer_started,
                }
            )
        plan = MlpPilotWeightOnlyPlan(
            repo_id=OFFICIAL_REPO_ID,
            revision=OFFICIAL_REVISION,
            model_pin_sha256=weight_only_model_pin(
                repo_id=OFFICIAL_REPO_ID,
                revision=OFFICIAL_REVISION,
                bundle_manifest_sha256=cast(str, bundle["manifest_sha256"]),
                layout_fingerprint=cast(str, bundle["layout_fingerprint"]),
                weights_index_sha256=index_sha,
            ),
            bundle_manifest_sha256=cast(str, bundle["manifest_sha256"]),
            layout_fingerprint=cast(str, bundle["layout_fingerprint"]),
            weights_index_sha256=index_sha,
            hidden_dimension=config.dim,
            n_layers=config.n_layers,
            config=router,
            online_config=MlpPilotOnlineConfig(),
            models=tuple(models),
            source_layer_sha256s=tuple(sources),
        )
        data = plan.to_bytes()
        if target.exists() and target.read_bytes() != data:
            raise ValueError("a different all-layer Q4 route plan already exists")
        if not target.exists():
            _atomic_bytes(target, data)
        state_path.unlink(missing_ok=True)
        _print(
            {
                "active_layers": config.n_layers,
                "bytes": len(data),
                "output": str(args.output.expanduser().resolve()),
                "plan_sha256": plan.sha256,
                "seconds": time.perf_counter() - started,
                "status": "complete",
            }
        )
        return 0
    finally:
        if bank is not None:
            bank.close()
        mount.close()


def main() -> int:
    try:
        return run(_parser().parse_args())
    except Exception as exc:
        _print({"status": "error", "reason": f"{type(exc).__name__}: {exc}"})
        return 2


if __name__ == "__main__":
    sys.exit(main())
