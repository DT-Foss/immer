from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Any, TextIO

from .contracts import Request, Result
from .resource_paths import s3_ship_manifest
from .substrate import LifeDaemon


COMPONENTS = (
    (
        "FERTIG",
        "grounded deterministic execution and verification",
        "vendored + runtime adapter",
    ),
    ("CRSA", "Causal Prefix Sinkhorn Attention", "integrated operators"),
    (
        "WorldStream",
        "exact local and pinned-remote Safetensors ranges",
        "integrated tensor source",
    ),
    ("LiveCausal", "append-only causal control plane", "integrated lazy graph"),
    (
        "CausalWeights",
        "local weight bundle and exact causal range routes",
        "integrated Qwen pager path",
    ),
    (
        "Qwen3.8",
        "complete local causal teacher and novelty fallback",
        "primary runtime",
    ),
    (
        "OoE",
        "persistent PS-Lifted Markov agents and executable Crystals",
        "integrated runtime",
    ),
    ("AnchorBattery", "authenticated prefix-state restoration", "integrated Qwen path"),
    ("DeepSeekV4", "complete 43-layer frontier decoder", "retained backend"),
    (
        "MarkovRouter",
        "label-free next-layer expert transport hints",
        "integrated evaluation path",
    ),
    (
        "OrganBank",
        "digest-addressed exact capabilities",
        "integrated artifact registry",
    ),
    (
        "o1-state",
        "persistent life stream outside frozen execution",
        "integrated runtime",
    ),
)


def _sorted_layer_list(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "layers must be comma-separated integers"
        ) from exc
    if (
        not layers
        or layers != tuple(sorted(set(layers)))
        or any(layer < 0 for layer in layers)
    ):
        raise argparse.ArgumentTypeError(
            "layers must be sorted unique non-negative integers"
        )
    return layers


def _finite_non_negative_float(value: str | int | float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "value must be a finite non-negative number"
        ) from exc
    if isinstance(value, bool) or not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be a finite non-negative number")
    return result


def _s3_manifest(configured: str | Path | None = None) -> Path:
    """Resolve the one deployment manifest used by solve, serve and organs."""

    selected = (
        configured
        or os.environ.get("IMMER_S3_MANIFEST")
        or os.environ.get("IMMER_ORGANBANK")
        or s3_ship_manifest()
    )
    return Path(selected).expanduser().resolve()


def _artifact_root(
    manifest: str | Path, configured: str | Path | None = None
) -> Path | None:
    from .artifacts import artifact_root

    return artifact_root(manifest, configured)


_QWEN38_DEPLOYMENT_ROOT = Path("/") / "app" / "models" / "Qwen3.8-27B"
_QWEN35_DEPLOYMENT_DRAFT_ROOT = Path("/") / "app" / "models" / "Qwen3.5-0.8B"
_QWEN38_DEPLOYMENT_PRIVATE = (
    Path("/") / "root" / "immer-runtime" / "artifacts" / "private"
)
_QWEN38_DEPLOYMENT_STATE = Path("/") / "root" / "immer-state"
_QWEN38_DEPLOYMENT_WARM_ROOT = _QWEN38_DEPLOYMENT_PRIVATE / "qwen3.8-ooe-chat-real"
_QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE = (
    _QWEN38_DEPLOYMENT_STATE / "qwen-markov-q4-v1.bin"
)
_QWEN38_DEPLOYMENT_MTP_STATE = _QWEN38_DEPLOYMENT_STATE / "qwen-mtp-q4-v1.json"
_QWEN38_DEPLOYMENT_MARKOV_ATLAS = _QWEN38_DEPLOYMENT_STATE / "qwen-markov-atlas-v1.bin"
_QWEN38_DEPLOYMENT_O1_RETENTION = (
    _QWEN38_DEPLOYMENT_STATE / "qwen-markov-o1-retention-v1.json"
)
_QWEN38_DEPLOYMENT_MLP_PAGE_STATE = (
    _QWEN38_DEPLOYMENT_STATE / "qwen-mlp-page-markov-v1.json"
)
_QWEN38_DEPLOYMENT_MLP_PAGE_COORDINATE_STATE = (
    _QWEN38_DEPLOYMENT_STATE / "qwen-mlp-page-coordinate-v1.json"
)
_QWEN38_DEPLOYMENT_DRAFT_WINDOW_STATE = (
    _QWEN38_DEPLOYMENT_STATE / "qwen-draft-window-v1.bin"
)
_QWEN38_DEPLOYMENT_INFERENCE_ECONOMICS = (
    _QWEN38_DEPLOYMENT_STATE / "qwen-inference-economics-v1"
)
_QWEN38_DEPLOYMENT_SERVICE_SOCKET = _QWEN38_DEPLOYMENT_STATE / "qwen3.8-service.sock"
_QWEN38_DEPLOYMENT_ANCHOR_CACHE = _QWEN38_DEPLOYMENT_STATE / "qwen-chat-prefix-anchors"
_QWEN38_DEPLOYMENT_Q4_RESIDENT_BUDGET_MB = 16_384
_QWEN38_DEPLOYMENT_CONTEXT_CRYSTAL_STATE = (
    _QWEN38_DEPLOYMENT_STATE / "qwen-contextual-continuation-v1.json"
)
_QWEN38_DEPLOYMENT_LAYER_CONTEXTUAL_CONTINUATION_STATE = (
    _QWEN38_DEPLOYMENT_STATE / "qwen-layer-contextual-continuation-v2.json"
)
_QWEN38_DEPLOYMENT_ATTENTION_OUTPUT_CRYSTAL_STATE = (
    _QWEN38_DEPLOYMENT_STATE / "qwen-attention-output-crystal-v1.json"
)
_QWEN38_DEPLOYMENT_LAYER_MLP_O1_STATE = (
    _QWEN38_DEPLOYMENT_STATE / "qwen-layer63-mlp-o1-decode-r128-v1.json"
)
_QWEN38_DEPLOYMENT_LAYER_MLP_O1_ATLAS = (
    _QWEN38_DEPLOYMENT_PRIVATE
    / "qwen3.8-o1-cartography"
    / "promptgrid-bcf3d28"
    / "atlas"
)
_QWEN38_DEPLOYMENT_LAYER_MLP_O1_COMPUTE = (
    _QWEN38_DEPLOYMENT_PRIVATE
    / "qwen3.8-o1-cartography"
    / "promptgrid-bcf3d28"
    / "ooe"
    / "operator-compute"
)
_QWEN38_DEPLOYMENT_LAYER_MLP_CRYSTAL_REGISTRY = (
    _QWEN38_DEPLOYMENT_STATE
    / "qwen-layer-mlp-crystals-v2"
    / "layer-mlp-o1-registry.json"
)
_QWEN38_MARKOV_DRAFT_ABI = "immer.qwen3.8-markov-draft-provider/v49"
_QWEN38_HYBRID_DRAFT_ABI = "immer.qwen3.8-markov-mtp-hybrid-provider/v31"
_QWEN38_MTP_DRAFT_ABI = "immer.qwen3.5-mtp-draft-provider/v6"
_QWEN38_GROWING_WARM_ABI_SHA256 = hashlib.sha256(
    b"immer:qwen3.8-growing-warm-runtime/v3"
).hexdigest()


def _chat_path(
    explicit: str | Path | None,
    environment: str,
) -> Path | None:
    value = explicit or os.environ.get(environment)
    if value is None:
        return None
    return Path(value).expanduser().absolute()


def _path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while chunk := handle.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _require_existing_real_directory(path: Path, option: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ValueError(f"{option} must name an existing real directory") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{option} must name an existing real directory")


def _authenticate_layer_transition_crystal_atlas(
    path: Path,
    *,
    atlas_revision_sha256: str,
    option: str = "--layer-transition-crystal-atlas",
    subject: str = "layer-transition Crystal",
) -> None:
    _require_existing_real_directory(path, option)
    from .knowledge.livecausal import LiveGraph
    from .runtimes.qwen3_8.semantic_atlas import GraphRevision

    try:
        atlas = LiveGraph(path)
        revision_history = atlas.store.revision_history()
    except Exception as exc:
        raise ValueError(f"{option} is unavailable or unauthenticated") from exc
    if not any(
        GraphRevision(sequence, event_sha256).sha256 == atlas_revision_sha256
        for sequence, event_sha256 in revision_history
    ):
        raise ValueError(f"{subject} bank belongs to a foreign Atlas authority")


def _authenticate_layer_transition_crystal_compute_graph(
    path: Path,
    *,
    graph_revision_sha256: str,
    option: str = "--layer-transition-crystal-compute-root",
    subject: str = "layer-transition Crystal",
) -> Any:
    _require_existing_real_directory(path, option)
    from .runtimes.ooe.compute_crystals import ComputeCrystalBank
    from .runtimes.ooe.compute_graph import (
        ComputeOperatorGraph,
        ComputeOperatorGraphState,
    )

    try:
        bank = ComputeCrystalBank(path)
        graph = ComputeOperatorGraph(bank)
        current = graph.state()
        seen: set[str] = set()
        while True:
            current_sha256 = current.sha256
            if current_sha256 in seen:
                raise ValueError(f"{subject} Compute graph history contains a cycle")
            seen.add(current_sha256)
            if current_sha256 == graph_revision_sha256:
                return current
            previous_sha256 = current.previous_state_sha256
            if previous_sha256 is None:
                break
            payload = bank.store.restore_state(
                graph.history_state_name(previous_sha256)
            )
            previous = ComputeOperatorGraphState.from_bytes(payload)
            if (
                previous.sha256 != previous_sha256
                or current.generation != previous.generation + 1
                or current.previous_state_sha256 != previous.sha256
            ):
                raise ValueError(f"{subject} Compute graph history is unauthenticated")
            current = previous
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"{option} is unavailable or unauthenticated") from exc
    raise ValueError(f"{subject} bank belongs to a foreign Compute graph authority")


def _qwen38_layer_contextual_continuation_policy(
    args: argparse.Namespace,
) -> dict[str, object] | None:
    """Return the canonical path-free v2 layer-continuation configuration."""

    configured = getattr(args, "layer_contextual_continuation_state", None)
    if configured is None:
        return None
    from .runtimes.qwen3_8.adapter import (
        DEFAULT_LAYER_CONTEXTUAL_CONTINUATION_LAYERS,
        DEFAULT_LAYER_CONTEXTUAL_CONTINUATION_PROJECTION_SEED,
        _layer_contextual_continuation_runtime_sha256,
    )
    from .runtimes.qwen3_8.layer_contextual_continuation import (
        DEFAULT_LAYER_CONTEXTUAL_SKETCH_DIM,
        LAYER_CONTEXTUAL_CONTINUATION_IDENTITY_SCHEMA,
        LAYER_CONTEXTUAL_PROJECTION_ABI,
        LAYER_CONTEXTUAL_STAGE,
    )

    max_cells = getattr(args, "layer_contextual_continuation_max_cells", 4096)
    max_receipts = getattr(
        args,
        "layer_contextual_continuation_max_receipts",
        65_536,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= 65_536
        for value in (max_cells, max_receipts)
    ):
        raise ValueError(
            "layer contextual continuation capacities must lie in [1, 65536]"
        )
    return {
        "enabled": True,
        "identity": {
            "identity_schema": LAYER_CONTEXTUAL_CONTINUATION_IDENTITY_SCHEMA,
            "layers": list(DEFAULT_LAYER_CONTEXTUAL_CONTINUATION_LAYERS),
            "projection": {
                "abi": LAYER_CONTEXTUAL_PROJECTION_ABI,
                "seed": DEFAULT_LAYER_CONTEXTUAL_CONTINUATION_PROJECTION_SEED,
                "sketch_dim": DEFAULT_LAYER_CONTEXTUAL_SKETCH_DIM,
            },
            "runtime_sha256": _layer_contextual_continuation_runtime_sha256(),
            "stage": LAYER_CONTEXTUAL_STAGE,
        },
        "max_cells": max_cells,
        "max_receipts": max_receipts,
        "schema": "immer.qwen3.8-layer-contextual-continuation-policy/v2",
    }


def _qwen38_layer_transition_crystal_policy(
    args: argparse.Namespace,
    *,
    request_applied: bool | None = None,
) -> dict[str, object] | None:
    """Return a path-free policy for one configured private layer-63 bank."""

    configured = getattr(args, "layer_transition_crystal_state", None)
    configured_atlas = getattr(args, "layer_transition_crystal_atlas", None)
    configured_compute_root = getattr(
        args,
        "layer_transition_crystal_compute_root",
        None,
    )
    radius = getattr(args, "layer_transition_crystal_max_error_radius", 0.0)
    try:
        radius = _finite_non_negative_float(radius)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(
            "layer-transition Crystal max-error radius must be finite and non-negative"
        ) from exc
    if configured is None:
        if configured_atlas is not None:
            raise ValueError(
                "--layer-transition-crystal-atlas requires "
                "--layer-transition-crystal-state"
            )
        if configured_compute_root is not None:
            raise ValueError(
                "--layer-transition-crystal-compute-root requires "
                "--layer-transition-crystal-state"
            )
        if radius != 0.0:
            raise ValueError(
                "--layer-transition-crystal-max-error-radius requires "
                "--layer-transition-crystal-state"
            )
        return None
    if configured_atlas is None:
        raise ValueError(
            "--layer-transition-crystal-state requires --layer-transition-crystal-atlas"
        )
    if configured_compute_root is None:
        raise ValueError(
            "--layer-transition-crystal-state requires "
            "--layer-transition-crystal-compute-root"
        )
    if request_applied is not None and not isinstance(request_applied, bool):
        raise TypeError("request_applied must be boolean or null")
    path = Path(configured).expanduser().absolute()
    if not path.is_file():
        raise ValueError(
            "--layer-transition-crystal-state must name an existing sealed file"
        )
    from .runtimes.qwen3_8.layer_transition_crystal import (
        LayerTransitionCrystalBank,
        LayerTransitionCrystalIdentity,
    )

    bank = LayerTransitionCrystalBank.load(path)
    identity = getattr(bank, "identity", None)
    if not isinstance(identity, LayerTransitionCrystalIdentity):
        raise ValueError("layer-transition Crystal bank identity is invalid")
    atlas_path = Path(configured_atlas).expanduser().absolute()
    _authenticate_layer_transition_crystal_atlas(
        atlas_path,
        atlas_revision_sha256=identity.atlas_revision_sha256,
    )
    compute_root = Path(configured_compute_root).expanduser().absolute()
    compute_state = _authenticate_layer_transition_crystal_compute_graph(
        compute_root,
        graph_revision_sha256=identity.graph_revision_sha256,
    )
    edges = {edge.sha256: edge for edge in compute_state.edges}
    for crystal in bank.crystals:
        source = crystal.source_compute_crystal_sha256
        edge_sha256 = crystal.source_compute_edge_sha256
        if source is None and edge_sha256 is None:
            continue
        edge = None if edge_sha256 is None else edges.get(edge_sha256)
        if edge is None or source is None or edge.crystal_sha256 != source:
            raise ValueError(
                "layer-transition Crystal source lineage differs from Compute graph"
            )
    return {
        "atlas_revision_sha256": identity.atlas_revision_sha256,
        "enabled": True,
        "graph_revision_sha256": identity.graph_revision_sha256,
        "identity_sha256": identity.identity_sha256,
        "max_error_radius": radius.hex(),
        "model_sha256": identity.model_sha256,
        "q4_sha256": identity.q4_sha256,
        "request_applied": True if request_applied is None else request_applied,
        "schema": "immer.qwen3.8-layer-transition-crystal-policy/v1",
    }


def _qwen38_layer_mlp_crystal_policy(
    args: argparse.Namespace,
    *,
    request_applied: bool | None = None,
) -> dict[str, object] | None:
    """Return a path-free policy for a legacy bank or immutable registry."""

    configured = getattr(args, "layer_mlp_crystal_state", None)
    configured_registry = getattr(args, "layer_mlp_crystal_registry", None)
    configured_atlas = getattr(args, "layer_mlp_crystal_atlas", None)
    configured_compute_root = getattr(args, "layer_mlp_crystal_compute_root", None)
    radius = getattr(args, "layer_mlp_crystal_max_error_radius", 0.0)
    try:
        radius = _finite_non_negative_float(radius)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(
            "layer-MLP Crystal max-error radius must be finite and non-negative"
        ) from exc
    if configured is not None and configured_registry is not None:
        raise ValueError(
            "--layer-mlp-crystal-state and --layer-mlp-crystal-registry are "
            "mutually exclusive"
        )
    if configured is None and configured_registry is None:
        if configured_atlas is not None:
            raise ValueError(
                "--layer-mlp-crystal-atlas requires a layer-MLP Crystal bank"
            )
        if configured_compute_root is not None:
            raise ValueError(
                "--layer-mlp-crystal-compute-root requires a layer-MLP Crystal bank"
            )
        if radius != 0.0:
            raise ValueError(
                "--layer-mlp-crystal-max-error-radius requires "
                "--layer-mlp-crystal-state"
            )
        return None
    if configured_atlas is None:
        raise ValueError("layer-MLP Crystals require --layer-mlp-crystal-atlas")
    if configured_compute_root is None:
        raise ValueError(
            "layer-MLP Crystals require --layer-mlp-crystal-compute-root"
        )
    if request_applied is not None and not isinstance(request_applied, bool):
        raise TypeError("request_applied must be boolean or null")
    if configured_registry is not None:
        if radius != 0.0:
            raise ValueError(
                "--layer-mlp-crystal-max-error-radius applies only to the legacy "
                "--layer-mlp-crystal-state"
            )
        from .runtimes.qwen3_8.layer_mlp_registry import (
            read_layer_mlp_o1_registry_manifest,
        )

        registry = read_layer_mlp_o1_registry_manifest(
            Path(configured_registry).expanduser().absolute(),
            require_mountable=True,
        )
        pins = registry.pins
        atlas_path = Path(configured_atlas).expanduser().absolute()
        _authenticate_layer_transition_crystal_atlas(
            atlas_path,
            atlas_revision_sha256=pins.atlas_revision_sha256,
            option="--layer-mlp-crystal-atlas",
            subject="layer-MLP Crystal registry",
        )
        compute_root = Path(configured_compute_root).expanduser().absolute()
        _authenticate_layer_transition_crystal_compute_graph(
            compute_root,
            graph_revision_sha256=pins.graph_revision_sha256,
            option="--layer-mlp-crystal-compute-root",
            subject="layer-MLP Crystal registry",
        )
        applied = True if request_applied is None else request_applied
        return {
            "atlas_revision_sha256": pins.atlas_revision_sha256,
            "enabled": True,
            "enabled_layer": max(registry.mounted_layers) if applied else None,
            "entries": [
                {
                    "action_abi": entry.action_abi,
                    "bank_file_sha256": entry.bank_file_sha256,
                    "bank_identity_sha256": entry.bank_identity_sha256,
                    "crystal_sha256": entry.crystal_sha256,
                    "error_radius": entry.error_radius.hex(),
                    "feature_radius": entry.feature_radius.hex(),
                    "layer_index": entry.layer_index,
                    "max_error_radius": entry.max_error_radius.hex(),
                    "packed_weight_bytes_avoided": (
                        entry.packed_weight_bytes_avoided
                    ),
                    "source_o1_generation": entry.source_o1_generation,
                    "source_o1_state_sha256": entry.source_o1_state_sha256,
                }
                for entry in registry.entries
            ],
            "graph_revision_sha256": pins.graph_revision_sha256,
            "installed_layers": list(registry.mounted_layers),
            "manifest_file_sha256": registry.manifest_file_sha256,
            "model_sha256": pins.model_sha256,
            "projection_sha256": pins.projection.projection_sha256,
            "q4_sha256": pins.q4_sha256,
            "registry_sha256": registry.registry_sha256,
            "request_applied": applied,
            "schema": "immer.qwen3.8-layer-mlp-residual-crystal-policy/v2",
        }

    assert configured is not None
    path = Path(configured).expanduser().absolute()
    if not path.is_file():
        raise ValueError("--layer-mlp-crystal-state must name an existing sealed file")
    from .runtimes.qwen3_8.layer_mlp_crystal import (
        LAYER_MLP_RESIDUAL_ACTION_ABI,
        Layer63MlpResidualCrystalBank,
        Layer63MlpResidualCrystalIdentity,
    )

    bank = Layer63MlpResidualCrystalBank.load(path)
    identity = getattr(bank, "identity", None)
    if not isinstance(identity, Layer63MlpResidualCrystalIdentity):
        raise ValueError("layer-MLP Crystal bank identity is invalid")
    if identity.action_abi != LAYER_MLP_RESIDUAL_ACTION_ABI:
        raise ValueError("layer-MLP Crystal action ABI is invalid")
    atlas_path = Path(configured_atlas).expanduser().absolute()
    _authenticate_layer_transition_crystal_atlas(
        atlas_path,
        atlas_revision_sha256=identity.atlas_revision_sha256,
        option="--layer-mlp-crystal-atlas",
        subject="layer-MLP Crystal",
    )
    compute_root = Path(configured_compute_root).expanduser().absolute()
    _authenticate_layer_transition_crystal_compute_graph(
        compute_root,
        graph_revision_sha256=identity.graph_revision_sha256,
        option="--layer-mlp-crystal-compute-root",
        subject="layer-MLP Crystal",
    )
    return {
        "action_abi": identity.action_abi,
        "atlas_revision_sha256": identity.atlas_revision_sha256,
        "enabled": True,
        "graph_revision_sha256": identity.graph_revision_sha256,
        "identity_sha256": identity.identity_sha256,
        "max_error_radius": radius.hex(),
        "model_sha256": identity.model_sha256,
        "q4_sha256": identity.q4_sha256,
        "request_applied": True if request_applied is None else request_applied,
        "schema": "immer.qwen3.8-layer-mlp-residual-crystal-policy/v1",
    }


def _qwen38_layer_mlp_o1_policy(args: argparse.Namespace) -> dict[str, object] | None:
    """Return a path-free identity/configuration for passive MLP O1 charging."""

    configured = getattr(args, "layer_mlp_o1_state", None)
    configured_atlas = getattr(args, "layer_mlp_o1_atlas", None)
    configured_compute = getattr(args, "layer_mlp_o1_compute_root", None)
    configured_layers = getattr(args, "layer_mlp_o1_layers", None)
    if configured is None:
        if (
            configured_atlas is not None
            or configured_compute is not None
            or configured_layers is not None
        ):
            raise ValueError("layer-MLP O1 authorities require --layer-mlp-o1-state")
        return None
    if configured_atlas is None or configured_compute is None:
        raise ValueError("--layer-mlp-o1-state requires Atlas and Compute authorities")
    from .runtimes.ooe.compute_crystals import ComputeCrystalBank
    from .runtimes.ooe.compute_graph import ComputeOperatorGraph
    from .knowledge.livecausal import LiveGraph
    from .runtimes.qwen3_8.adapter import (
        DEFAULT_LAYER_MLP_O1_PROJECTION_SEED,
        _layer_mlp_o1_state_path_for_layer,
    )
    from .runtimes.qwen3_8.layer_mlp_crystal import (
        Layer63MlpResidualCrystalIdentity,
        LayerMlpResidualCrystalIdentity,
    )
    from .runtimes.qwen3_8.layer_mlp_o1 import (
        Layer63MlpO1Accumulator,
        LayerMlpO1Accumulator,
    )
    from .runtimes.qwen3_8.layer_transition_crystal import (
        LayerTransitionProjectionIdentity,
    )
    from .runtimes.qwen3_8.semantic_atlas import GraphRevision

    state_path = Path(configured).expanduser().absolute()
    atlas_path = Path(configured_atlas).expanduser().absolute()
    compute_root = Path(configured_compute).expanduser().absolute()
    sketch_dim = int(getattr(args, "layer_mlp_o1_sketch_dim", 128))
    seed = getattr(
        args,
        "layer_mlp_o1_seed",
        DEFAULT_LAYER_MLP_O1_PROJECTION_SEED,
    )
    ridge = float(getattr(args, "layer_mlp_o1_ridge", 1e-8))
    coverage_guard = float(getattr(args, "layer_mlp_o1_coverage_guard", 0.0))
    error_guard = float(getattr(args, "layer_mlp_o1_error_guard", 0.0))
    queue_capacity = int(getattr(args, "layer_mlp_o1_queue_capacity", 8))
    layers = (63,) if configured_layers is None else tuple(configured_layers)
    if (
        not layers
        or any(
            isinstance(layer, bool) or not isinstance(layer, int) for layer in layers
        )
        or layers != tuple(sorted(set(layers)))
        or any(not 0 <= layer <= 63 for layer in layers)
    ):
        raise ValueError(
            "layer-MLP O1 layers must be sorted unique integers in [0, 63]"
        )
    if not 0 < sketch_dim <= 256:
        raise ValueError("layer-MLP O1 sketch_dim must lie in [1, 256]")
    if (
        not isinstance(seed, str)
        or len(seed) != 64
        or set(seed) - set("0123456789abcdef")
    ):
        raise ValueError("layer-MLP O1 seed must be a lowercase SHA-256")
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("layer-MLP O1 ridge must be positive and finite")
    if any(
        not math.isfinite(value) or value < 0 for value in (coverage_guard, error_guard)
    ):
        raise ValueError("layer-MLP O1 guards must be finite and non-negative")
    if queue_capacity <= 0:
        raise ValueError("layer-MLP O1 queue capacity must be positive")
    try:
        atlas = LiveGraph(atlas_path)
        sequence, event_sha256 = atlas.store.revision()
        atlas_revision = GraphRevision(sequence, event_sha256).sha256
        compute_revision = (
            ComputeOperatorGraph(ComputeCrystalBank(compute_root)).state().sha256
        )
    except Exception as exc:
        raise ValueError(
            "layer-MLP O1 authorities are unavailable or unauthenticated"
        ) from exc
    authority_accumulator: Layer63MlpO1Accumulator | LayerMlpO1Accumulator | None = None
    authority_layer: int | None = None
    authority_candidates = (
        (63, *tuple(layer for layer in layers if layer != 63))
        if 63 in layers
        else layers
    )
    for layer_index in authority_candidates:
        candidate = _layer_mlp_o1_state_path_for_layer(
            state_path,
            layer_index=layer_index,
            sketch_dim=sketch_dim,
        )
        if not candidate.exists():
            continue
        authority_accumulator = (
            Layer63MlpO1Accumulator.load(candidate)
            if layer_index == 63
            else LayerMlpO1Accumulator.load(candidate)
        )
        expected_type = (
            Layer63MlpResidualCrystalIdentity
            if layer_index == 63
            else LayerMlpResidualCrystalIdentity
        )
        if not isinstance(authority_accumulator.identity, expected_type):
            raise ValueError(
                f"layer-{layer_index} MLP O1 state has another identity generation"
            )
        if authority_accumulator.identity.layer_index != layer_index:
            raise ValueError(
                f"layer-{layer_index} MLP O1 state has another layer identity"
            )
        authority_layer = layer_index
        break

    if authority_accumulator is not None:
        assert authority_layer is not None
        authority = authority_accumulator.identity
        _authenticate_layer_transition_crystal_atlas(
            atlas_path,
            atlas_revision_sha256=authority.atlas_revision_sha256,
            option="--layer-mlp-o1-atlas",
            subject="layer-MLP O1",
        )
        _authenticate_layer_transition_crystal_compute_graph(
            compute_root,
            graph_revision_sha256=authority.graph_revision_sha256,
            option="--layer-mlp-o1-compute-root",
            subject="layer-MLP O1",
        )
        expected_projection = LayerTransitionProjectionIdentity(
            hidden_dim=authority.hidden_dim,
            sketch_dim=sketch_dim,
            seed_sha256=seed,
        )
        if (
            authority.projection != expected_projection
            or authority_accumulator.ridge != ridge
            or authority_accumulator.coverage_guard != coverage_guard
            or authority_accumulator.error_guard != error_guard
        ):
            raise ValueError(
                f"layer-{authority_layer} MLP O1 authority differs from its CLI "
                "configuration"
            )
        atlas_revision = authority.atlas_revision_sha256
        compute_revision = authority.graph_revision_sha256

    if layers == (63,):
        identity = None
        if authority_accumulator is not None:
            state_identity = authority_accumulator.identity
            identity = {
                "atlas_revision_sha256": state_identity.atlas_revision_sha256,
                "graph_revision_sha256": state_identity.graph_revision_sha256,
                "identity_sha256": state_identity.identity_sha256,
                "model_sha256": state_identity.model_sha256,
                "q4_sha256": state_identity.q4_sha256,
            }
        return {
            "authorities": {
                "atlas_revision_sha256": atlas_revision,
                "compute_revision_sha256": compute_revision,
            },
            "coverage_guard": coverage_guard.hex(),
            "enabled": True,
            "error_guard": error_guard.hex(),
            "identity": identity,
            "projection_seed_sha256": seed,
            "queue_capacity": queue_capacity,
            "ridge": ridge.hex(),
            "schema": "immer.qwen3.8-layer63-mlp-o1-policy/v1",
            "sketch_dim": sketch_dim,
        }

    return {
        "authorities": {
            "atlas_revision_sha256": atlas_revision,
            "compute_revision_sha256": compute_revision,
        },
        "coverage_guard": coverage_guard.hex(),
        "enabled": True,
        "error_guard": error_guard.hex(),
        "layers": list(layers),
        "projection_seed_sha256": seed,
        "queue_capacity": queue_capacity,
        "ridge": ridge.hex(),
        "schema": "immer.qwen3.8-layer-mlp-o1-policy/v2",
        "sketch_dim": sketch_dim,
    }


def _resolve_qwen38_chat_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path | None, Path | None]:
    """Resolve the deployed local chat stack without requiring flag repetition."""

    root = _chat_path(getattr(args, "qwen38_root", None), "IMMER_QWEN38_ROOT")
    custom_layout = any(
        (
            getattr(args, "qwen38_causal_bundle", None),
            getattr(args, "qwen38_tokenizer", None),
            os.environ.get("IMMER_QWEN38_CAUSAL_BUNDLE"),
            os.environ.get("IMMER_QWEN38_TOKENIZER"),
        )
    )
    if root is None and not custom_layout and _QWEN38_DEPLOYMENT_ROOT.is_dir():
        root = _QWEN38_DEPLOYMENT_ROOT

    bundle = _chat_path(
        getattr(args, "qwen38_causal_bundle", None),
        "IMMER_QWEN38_CAUSAL_BUNDLE",
    )
    if bundle is None:
        bundle = root

    tokenizer = _chat_path(
        getattr(args, "qwen38_tokenizer", None),
        "IMMER_QWEN38_TOKENIZER",
    )
    if tokenizer is None and root is not None:
        tokenizer = root / "tokenizer.json"

    q4 = _chat_path(
        getattr(args, "qwen38_q4", None),
        "IMMER_QWEN38_Q4",
    )
    if q4 is None and root is not None:
        draft_mode = getattr(args, "draft_mode", None)
        if draft_mode in {"hybrid", "mtp"}:
            candidates = (root / "causal" / "q4-base-v3-mtp",)
        elif draft_mode in {None, "markov"}:
            candidates = (
                root / "causal" / "q4-base-v3-mtp",
                root / "causal" / "q4-base-v2",
            )
        else:
            candidates = (root / "causal" / "q4-base-v2",)
        q4 = next((candidate for candidate in candidates if candidate.is_dir()), None)

    disable_fast_mlp = bool(getattr(args, "no_fast_mlp", False))
    if disable_fast_mlp and getattr(args, "fast_mlp", None) is not None:
        raise ValueError("--fast-mlp and --no-fast-mlp are mutually exclusive")
    fast_mlp = (
        None
        if disable_fast_mlp
        else _chat_path(
            getattr(args, "fast_mlp", None),
            "IMMER_QWEN38_FAST_MLP",
        )
    )
    if bundle is None or tokenizer is None:
        raise ValueError(
            "local Qwen is not configured; set --qwen38-root or IMMER_QWEN38_ROOT"
        )
    return bundle, tokenizer, q4, fast_mlp


def _resolve_qwen38_prefix_sinkhorn(
    args: argparse.Namespace,
    bundle_path: Path,
    q4_root: Path | None,
) -> bool:
    """Resolve the deployment default while preserving both explicit overrides."""

    selected = getattr(args, "prefix_sinkhorn", None)
    if selected is None:
        selected = bundle_path == _QWEN38_DEPLOYMENT_ROOT and q4_root is not None
    elif not isinstance(selected, bool):
        raise TypeError("prefix_sinkhorn must be boolean or None")
    args.prefix_sinkhorn = selected
    return selected


def _resolve_qwen38_warm_root(
    args: argparse.Namespace,
    bundle_path: Path,
) -> Path | None:
    if bool(getattr(args, "no_ooe_warm", False)):
        if getattr(args, "ooe_warm_root", None) is not None:
            raise ValueError("--ooe-warm-root and --no-ooe-warm are mutually exclusive")
        return None
    configured = _chat_path(
        getattr(args, "ooe_warm_root", None),
        "IMMER_QWEN38_OOE_WARM_ROOT",
    )
    if configured is not None:
        return configured
    if bundle_path == _QWEN38_DEPLOYMENT_ROOT and _QWEN38_DEPLOYMENT_WARM_ROOT.is_dir():
        return _QWEN38_DEPLOYMENT_WARM_ROOT
    return None


def _resolve_qwen38_inference_economics(
    args: argparse.Namespace,
    bundle_path: Path,
) -> Path | None:
    disabled = bool(getattr(args, "no_inference_economics", False))
    configured_value = getattr(args, "inference_economics_state", None)
    if disabled and configured_value is not None:
        raise ValueError(
            "--inference-economics-state and --no-inference-economics "
            "are mutually exclusive"
        )
    if disabled:
        if bool(getattr(args, "service", False)):
            raise ValueError("the Qwen service requires inference economics")
        return None
    configured = _chat_path(
        configured_value,
        "IMMER_QWEN38_INFERENCE_ECONOMICS",
    )
    if configured is not None:
        return configured
    if bundle_path == _QWEN38_DEPLOYMENT_ROOT and _QWEN38_DEPLOYMENT_STATE.is_dir():
        return _QWEN38_DEPLOYMENT_INFERENCE_ECONOMICS
    if bool(getattr(args, "service", False)):
        service_socket = getattr(args, "service_socket", None)
        if service_socket is None:
            raise ValueError("the Qwen service socket is unresolved")
        return Path(service_socket).parent / "qwen-inference-economics-v1"
    return None


def _resolve_qwen38_service_socket(
    args: argparse.Namespace,
    bundle_path: Path,
) -> Path:
    configured = _chat_path(
        getattr(args, "service_socket", None),
        "IMMER_QWEN38_SOCKET",
    )
    if configured is not None:
        return configured
    if bundle_path == _QWEN38_DEPLOYMENT_ROOT and _QWEN38_DEPLOYMENT_STATE.is_dir():
        return _QWEN38_DEPLOYMENT_SERVICE_SOCKET
    return (Path.home() / ".immer" / "qwen3.8-service.sock").absolute()


def _qwen38_service_profile(
    args: argparse.Namespace,
    *,
    bundle_path: Path,
    tokenizer_path: Path,
    q4_root: Path | None,
    fast_mlp_root: Path | None,
    warm_root: Path | None,
    draft_mode: str | None,
    markov_draft_state: str | None,
    mtp_draft_state: str | None,
    markov_atlas_path: Path | None,
    markov_o1_retention_path: Path | None,
    mlp_page_state_path: Path | None,
    draft_window_state_path: Path | None,
    runtime_code_revision: str,
    attention_output_crystal_state_path: Path | None = None,
    mlp_page_coordinate_enabled: bool | None = None,
) -> str:
    """Bind socket clients to the exact output-affecting runtime configuration."""

    if mlp_page_coordinate_enabled is None:
        mlp_page_coordinate_enabled = (
            getattr(args, "mlp_page_coordinate_state", None) is not None
        )
    if not isinstance(mlp_page_coordinate_enabled, bool):
        raise TypeError("mlp_page_coordinate_enabled must be boolean")

    argument_names = (
        "compute_dtype",
        "context_crystal_state",
        "delta_head_layers",
        "delta_head_online_state",
        "device",
        "draft_bundle",
        "draft_max_resident_mb",
        "draft_source_budget_mb",
        "draft_window",
        "exact_head",
        "exact_head_max_mb",
        "fast_mlp_blocks",
        "fast_mlp_layers",
        "fast_mlp_max_resident_mb",
        "fast_mlp_online_state",
        "fast_mlp_policy",
        "fast_mlp_source_budget_mb",
        "head_block_rows",
        "inference_economics_state",
        "layer_mlp_o1_layers",
        "max_context_tokens",
        "max_new_tokens",
        "max_prompt_tokens",
        "max_resident_mb",
        "mlp_page_width",
        "no_inference_economics",
        "no_external_drafter",
        "no_context_crystal",
        "no_attention_output_crystal",
        "prefix_sinkhorn",
        "q4_threads",
        "q4_resident_budget_mb",
        "qwen38_anchor_cache",
        "range_markov_state",
        "range_prefetch_beam_horizon",
        "range_prefetch_beam_width",
        "range_prefetch_hint_cooldown",
        "range_prefetch_max_mb",
        "range_prefetch_min_confidence",
        "range_prefetch_min_support",
        "raw_qwen",
        "source_budget_mb",
        "system_prompt",
    )

    def normalized(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, tuple):
            return list(value)
        return value

    profile = {
        "attention_output_crystal": (attention_output_crystal_state_path is not None),
        "layer_contextual_continuation": (
            _qwen38_layer_contextual_continuation_policy(args)
        ),
        "layer_mlp_crystal": _qwen38_layer_mlp_crystal_policy(args),
        "layer_mlp_o1": _qwen38_layer_mlp_o1_policy(args),
        "layer_transition_crystal": (_qwen38_layer_transition_crystal_policy(args)),
        "mlp_page_coordinate": mlp_page_coordinate_enabled,
        "arguments": {
            name: normalized(getattr(args, name, None)) for name in argument_names
        },
        "paths": {
            "bundle": str(bundle_path),
            "draft_window_state": normalized(draft_window_state_path),
            "fast_mlp": normalized(fast_mlp_root),
            "markov_atlas": normalized(markov_atlas_path),
            "markov_draft_state": markov_draft_state,
            "markov_o1_retention": normalized(markov_o1_retention_path),
            "mlp_page_state": normalized(mlp_page_state_path),
            "mtp_draft_state": mtp_draft_state,
            "q4": normalized(q4_root),
            "tokenizer": str(tokenizer_path),
            "warm_root": normalized(warm_root),
        },
        "draft_mode": draft_mode,
        "runtime_code_revision": runtime_code_revision,
        "schema": "immer.qwen3.8-service-profile/v1",
    }
    return hashlib.sha256(
        json.dumps(
            profile,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _resolve_qwen38_markov_draft(
    args: argparse.Namespace,
    bundle_path: Path,
    q4_root: Path | None,
) -> tuple[str | None, str | None, str | None]:
    draft_mode = getattr(args, "draft_mode", None)
    markov_state = getattr(args, "markov_draft_state", None)
    mtp_state = getattr(args, "mtp_draft_state", None)
    anchor_active = getattr(args, "qwen38_anchor_cache", None) is not None and not bool(
        getattr(args, "no_anchor_cache", False)
    )
    disabled = bool(getattr(args, "no_markov_draft", False))
    if disabled:
        if draft_mode in {"hybrid", "markov"} or markov_state is not None:
            raise ValueError(
                "Markov draft options and --no-markov-draft are mutually exclusive"
            )
        if draft_mode is None and mtp_state is not None:
            draft_mode = "mtp"
        return draft_mode, markov_state, mtp_state
    if draft_mode is None and getattr(args, "draft_bundle", None) is None:
        if markov_state is not None and mtp_state is not None:
            draft_mode = "hybrid"
        elif mtp_state is not None:
            draft_mode = "mtp"
    if (
        draft_mode is None
        and getattr(args, "draft_bundle", None) is None
        and markov_state is None
        and mtp_state is None
        and bundle_path == _QWEN38_DEPLOYMENT_ROOT
        and q4_root is not None
        and q4_root.name == "q4-base-v3-mtp"
        and not anchor_active
        and _QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE.is_file()
    ):
        return (
            "hybrid",
            str(_QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE),
            str(_QWEN38_DEPLOYMENT_MTP_STATE),
        )
    if (
        draft_mode is None
        and getattr(args, "draft_bundle", None) is None
        and markov_state is None
        and bundle_path == _QWEN38_DEPLOYMENT_ROOT
        and q4_root is not None
        and _QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE.is_file()
    ):
        return "markov", str(_QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE), mtp_state
    if (
        draft_mode == "markov"
        and markov_state is None
        and bundle_path == _QWEN38_DEPLOYMENT_ROOT
        and q4_root is not None
        and _QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE.is_file()
    ):
        markov_state = str(_QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE)
    if (
        draft_mode == "mtp"
        and mtp_state is None
        and bundle_path == _QWEN38_DEPLOYMENT_ROOT
        and q4_root is not None
    ):
        mtp_state = markov_state or str(_QWEN38_DEPLOYMENT_MTP_STATE)
    if (
        draft_mode == "hybrid"
        and bundle_path == _QWEN38_DEPLOYMENT_ROOT
        and q4_root is not None
    ):
        markov_state = markov_state or str(_QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE)
        mtp_state = mtp_state or str(_QWEN38_DEPLOYMENT_MTP_STATE)
    return draft_mode, markov_state, mtp_state


def _qwen38_growing_warm_profile(
    args: argparse.Namespace,
    *,
    tokenizer_path: Path,
    q4_root: Path | None,
    fast_mlp_root: Path | None,
    mlp_page_state_path: Path | None = None,
    draft_mode: str | None,
    markov_atlas_path: Path | None,
    markov_o1_retention_path: Path | None,
    runtime_code_revision: str,
) -> str | None:
    """Bind reusable cold cells to the exact pre-load product runtime."""

    if q4_root is None or fast_mlp_root is not None:
        return None
    q4_manifest = q4_root / "manifest.json"
    if not q4_manifest.is_file() or not tokenizer_path.is_file():
        return None
    external_drafter = None
    if draft_mode == "hybrid" and getattr(args, "draft_bundle", None) is not None:
        from .runtimes.qwen3_8.config import (
            QWEN35_DRAFTER_REPO_ID,
            QWEN35_DRAFTER_REVISION,
        )
        from .runtimes.qwen3_8.local_draft import (
            QWEN35_K4_DRAFT_PROVIDER_SCHEMA,
        )

        external_drafter = {
            "bundle_path_sha256": hashlib.sha256(
                str(Path(args.draft_bundle).expanduser().absolute()).encode("utf-8")
            ).hexdigest(),
            "provider_abi": QWEN35_K4_DRAFT_PROVIDER_SCHEMA,
            "repo_id": QWEN35_DRAFTER_REPO_ID,
            "revision": QWEN35_DRAFTER_REVISION,
        }
    mlp_page_route = None
    if mlp_page_state_path is not None:
        from .runtimes.qwen3_8.mlp_page_markov import (
            MLP_PAGE_MARKOV_POLICY,
            MLP_PAGE_MARKOV_SCHEMA,
            MlpPageMarkov,
        )

        route_width = getattr(args, "mlp_page_width", None)
        if (
            isinstance(route_width, bool)
            or not isinstance(route_width, int)
            or route_width <= 0
        ):
            raise ValueError("MLP page width must be a positive integer")
        mlp_page_route = {
            "energy_coverage": MlpPageMarkov.ENERGY_COVERAGE.hex(),
            "policy": MLP_PAGE_MARKOV_POLICY,
            "route_width": route_width,
            "schema": MLP_PAGE_MARKOV_SCHEMA,
            "width_actions": list(MlpPageMarkov.width_actions_for(route_width)),
        }
    profile = {
        "abi_sha256": _QWEN38_GROWING_WARM_ABI_SHA256,
        "anchor_cache": args.qwen38_anchor_cache is not None,
        "attention_output_crystal": (
            None
            if getattr(args, "attention_output_crystal_state", None) is None
            else {
                "evidence_schema": (
                    "immer.qwen3.8-attention-output-crystal-evidence/v1"
                ),
                "output": "exact-full-attention-token-transition/v1",
            }
        ),
        "layer_transition_crystal": (_qwen38_layer_transition_crystal_policy(args)),
        "layer_contextual_continuation": (
            _qwen38_layer_contextual_continuation_policy(args)
        ),
        "layer_mlp_crystal": _qwen38_layer_mlp_crystal_policy(args),
        "layer_mlp_o1": _qwen38_layer_mlp_o1_policy(args),
        "mlp_page_coordinate": (
            getattr(args, "mlp_page_coordinate_state", None) is not None
        ),
        "compute_dtype": args.compute_dtype,
        "device": "cpu" if args.device == "auto" else args.device,
        "draft_mode": draft_mode,
        "external_drafter": external_drafter,
        "draft_window": args.draft_window,
        "head_block_rows": args.head_block_rows,
        "draft_provider_abi": (
            _QWEN38_HYBRID_DRAFT_ABI
            if draft_mode == "hybrid"
            else _QWEN38_MARKOV_DRAFT_ABI
            if draft_mode == "markov"
            else _QWEN38_MTP_DRAFT_ABI
            if draft_mode == "mtp"
            else None
        ),
        "delta_head_router": (
            None
            if getattr(args, "delta_head_online_state", None) is None
            else {
                "active_layers": (
                    "all-linear-attention"
                    if getattr(args, "delta_head_layers", None) is None
                    else list(args.delta_head_layers)
                ),
                "max_selected_heads": 40,
                "policy": "mean-square+sinkhorn-first-order/v1",
                "width_actions": [24, 32, 40],
            }
        ),
        "contextual_continuation_crystal": (
            None
            if getattr(args, "context_crystal_state", None) is None
            else {
                "key_abi": "known-token+normalized-rademacher-q8-256/v1",
                "markov_provider_abi": _QWEN38_MARKOV_DRAFT_ABI,
                "maximum_tail_tokens": 15,
                "target_verified": True,
            }
        ),
        "markov_provider_abi": (
            _QWEN38_MARKOV_DRAFT_ABI if draft_mode in {"hybrid", "markov"} else None
        ),
        "markov_atlas_sha256": (
            None if markov_atlas_path is None else _path_sha256(markov_atlas_path)
        ),
        "markov_o1_retention": markov_o1_retention_path is not None,
        "mtp_provider_abi": (
            _QWEN38_MTP_DRAFT_ABI if draft_mode in {"hybrid", "mtp"} else None
        ),
        "max_context_tokens": args.max_context_tokens,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "mlp_page_route": mlp_page_route,
        "prefix_sinkhorn": bool(getattr(args, "prefix_sinkhorn", False)),
        "q4_manifest_file_sha256": _path_sha256(q4_manifest),
        "q4_threads": args.q4_threads or min(16, os.cpu_count() or 1),
        "runtime_code_revision": runtime_code_revision,
        "schema": "immer.qwen3.8-growing-warm-runtime/v3",
        "system_prompt_sha256": hashlib.sha256(
            args.system_prompt.strip().encode("utf-8")
        ).hexdigest(),
        "tokenizer_sha256": _path_sha256(tokenizer_path),
    }
    return hashlib.sha256(
        json.dumps(
            profile,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _qwen38_output_semantics(
    args: argparse.Namespace,
    *,
    tokenizer_path: Path,
    q4_root: Path | None,
    mlp_page_state_path: Path | None,
):
    # A private layer-transition bank is a bounded approximate same-runtime
    # action.  It cannot share the cross-profile exact-output replay authority.
    if (
        _qwen38_layer_transition_crystal_policy(args) is not None
        or _qwen38_layer_mlp_crystal_policy(args) is not None
    ):
        return None
    if q4_root is None:
        return None
    q4_manifest = q4_root / "manifest.json"
    if not q4_manifest.is_file() or not tokenizer_path.is_file():
        return None
    from .runtimes.qwen3_8.config import OFFICIAL_REPO_ID, OFFICIAL_REVISION
    from .runtimes.qwen3_8.encoding import END_OF_TEXT_TOKEN_ID, IM_END_TOKEN_ID
    from .runtimes.qwen3_8.mlp_page_markov import MlpPageMarkov
    from .runtimes.qwen3_8.output_semantics import QwenOutputSemantics
    from .runtimes.qwen3_8.q4 import Q4_BANK_CODEC_ABI, Q4_NATIVE_ABI

    route_width = None
    width_actions = ()
    energy_coverage = None
    if mlp_page_state_path is not None:
        route_width = getattr(args, "mlp_page_width", None)
        if (
            isinstance(route_width, bool)
            or not isinstance(route_width, int)
            or route_width <= 0
        ):
            raise ValueError("MLP page width must be a positive integer")
        width_actions = MlpPageMarkov.width_actions_for(route_width)
        energy_coverage = MlpPageMarkov.ENERGY_COVERAGE
    return QwenOutputSemantics(
        repo_id=OFFICIAL_REPO_ID,
        revision=OFFICIAL_REVISION,
        q4_manifest_file_sha256=_path_sha256(q4_manifest),
        q4_native_abi=Q4_NATIVE_ABI,
        q4_bank_codec_abi=Q4_BANK_CODEC_ABI,
        tokenizer_sha256=_path_sha256(tokenizer_path),
        compute_dtype=args.compute_dtype,
        max_context_tokens=args.max_context_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=(IM_END_TOKEN_ID, END_OF_TEXT_TOKEN_ID),
        mlp_page_route_width=route_width,
        mlp_page_width_actions=width_actions,
        mlp_page_energy_coverage=energy_coverage,
    )


def _qwen38_runtime_code_paths() -> tuple[Path, ...]:
    package = Path(__file__).resolve().parent
    fixed = (
        package / "cli.py",
        package / "contracts.py",
        package / "cognition" / "qwen_fertig_chat.py",
        package / "knowledge" / "livecausal.py",
        package / "runtimes" / "ooe" / "chat.py",
        package / "runtimes" / "ooe" / "controller.py",
        package / "runtimes" / "ooe" / "qwen_warm_bank.py",
        package / "runtimes" / "ooe" / "qwen_warm_growth.py",
        package / "runtimes" / "ooe" / "result_cells.py",
    )
    qwen_runtime = tuple(
        sorted(
            (
                path
                for path in (package / "runtimes" / "qwen3_8").glob("*.py")
                if path.name
                not in {"layer_transition_builder.py", "layer_mlp_builder.py"}
            ),
            key=lambda path: path.name,
        )
    )
    return tuple(sorted({*fixed, *qwen_runtime}, key=lambda path: str(path)))


def _qwen38_runtime_code_revision() -> str:
    package = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in _qwen38_runtime_code_paths():
        digest.update(str(path.relative_to(package)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb", buffering=0) as handle:
            while chunk := handle.read(1024**2):
                digest.update(chunk)
    return digest.hexdigest()


class _LiveTextWriter:
    """Render cumulative decoder snapshots as one readable terminal stream."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._snapshot = ""
        self._wrote = False
        self._finished = False

    def update(self, snapshot: str) -> None:
        if not isinstance(snapshot, str):
            raise TypeError("streaming snapshot must be text")
        if self._finished:
            raise RuntimeError("text stream is already finished")
        snapshot = snapshot.strip()
        if snapshot == self._snapshot:
            return
        if snapshot.startswith(self._snapshot):
            delta = snapshot[len(self._snapshot) :]
        else:
            delta = ("\n[rewrite]\n" if self._wrote else "") + snapshot
        if delta:
            self._stream.write(delta)
            self._stream.flush()
            self._wrote = True
        self._snapshot = snapshot

    def finish(self, final_text: str) -> None:
        if self._finished:
            return
        if not isinstance(final_text, str):
            raise TypeError("final chat output must be text")
        final_text = final_text.strip()
        if not self._wrote:
            self._stream.write(final_text)
        elif final_text.startswith(self._snapshot):
            self._stream.write(final_text[len(self._snapshot) :])
        elif final_text != self._snapshot:
            self._stream.write(f"\n[final]\n{final_text}")
        self._stream.write("\n")
        self._stream.flush()
        self._snapshot = final_text
        self._wrote = self._wrote or bool(final_text)
        self._finished = True

    def fail(self) -> None:
        if not self._finished and self._wrote:
            self._stream.write("\n")
            self._stream.flush()
        self._finished = True

    def reset(self) -> None:
        """Start the next response while keeping the loaded chat runtime alive."""

        if not self._finished:
            raise RuntimeError("cannot reset an unfinished text stream")
        self._snapshot = ""
        self._wrote = False
        self._finished = False


class _LiveProgressWriter:
    """Show decode progress without exposing provisional routed text."""

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._tokens = 0
        self._active = False

    def update(self, _snapshot: str) -> None:
        self._tokens += 1
        self._stream.write(f"\rQwen: {self._tokens} tokens")
        self._stream.flush()
        self._active = True

    def finish(self) -> None:
        if self._active:
            self._stream.write("\r\x1b[2K")
            self._stream.flush()
        self._active = False
        self._tokens = 0


def _components() -> int:
    for name, role, integration in COMPONENTS:
        print(f"{name:10}  {role}  [{integration}]")
    return 0


def _solve(
    question: str,
    manifest: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> int:
    from .composition import CompositionRoot

    composition = CompositionRoot.build(
        s3_manifest=_s3_manifest(manifest),
        s3_artifact_root=artifact_root,
    )
    result = composition.dispatch("exact_math", question)
    print(
        json.dumps(
            {
                "status": result.status.value,
                "component": result.component,
                "output": result.output,
                "reason": result.reason,
                "evidence": dict(result.evidence),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if result.ok else 2


def _chat_qwen38(args: argparse.Namespace) -> int:
    """Run one turn or a persistent JSONL/interactive local Qwen3.8 session."""

    from .cognition.fertig.adapter import FertigSolver
    from .cognition.qwen_fertig_chat import QwenFertigChat
    from .runtimes.ooe.qwen_warm_bank import (
        open_verified_qwen_warm_bank,
    )
    from .runtimes.qwen3_8.adapter import (
        QWEN38_CHAT_HISTORY_METADATA,
        QWEN38_CHAT_SESSION_METADATA,
        QWEN38_INFERENCE_ACTION_METADATA,
        Qwen38CausalChat,
    )
    from .runtimes.qwen3_8.draft_window import DraftWindowError
    from .runtimes.qwen3_8.semantic_state_cache import SemanticStateAnchorCache
    from .runtimes.qwen3_8.service import (
        QwenChatServiceApplication,
        QwenServiceError,
        SnapshotEventBridge,
        UnixQwenServiceClient,
        UnixQwenServiceServer,
    )

    component = None
    jsonl = bool(getattr(args, "jsonl", False))
    interactive = bool(getattr(args, "interactive", False))
    service = bool(getattr(args, "service", False))
    direct = bool(getattr(args, "direct", False))
    message = getattr(args, "message", None)
    max_requests = getattr(args, "max_requests", None)
    q4_resident_budget_mb = getattr(args, "q4_resident_budget_mb", None)
    if q4_resident_budget_mb is not None and (
        isinstance(q4_resident_budget_mb, bool)
        or not isinstance(q4_resident_budget_mb, int)
        or q4_resident_budget_mb < 0
    ):
        raise ValueError("--q4-resident-budget-mb must be non-negative")
    output_mode = getattr(args, "output", None) or ("json" if jsonl else "text")
    stream_enabled = (
        not service
        and not jsonl
        and output_mode == "text"
        and not bool(getattr(args, "no_stream", False))
    )
    live_writer = (
        _LiveTextWriter(sys.stdout)
        if stream_enabled and bool(getattr(args, "raw_qwen", False))
        else None
    )
    progress_writer = (
        _LiveProgressWriter(sys.stderr)
        if stream_enabled
        and not bool(getattr(args, "raw_qwen", False))
        and sys.stderr.isatty()
        else None
    )

    def emit(result, *, request_id=None, include_id: bool = False) -> None:
        if output_mode == "text":
            if progress_writer is not None:
                progress_writer.finish()
            if result.ok and isinstance(result.output, str):
                if live_writer is None:
                    print(result.output, flush=True)
                else:
                    live_writer.finish(result.output)
            else:
                if live_writer is not None:
                    live_writer.fail()
                print(
                    f"error: {result.reason or 'chat returned no text'}",
                    file=sys.stderr,
                    flush=True,
                )
            return
        value = {
            "status": result.status.value,
            "component": result.component,
            "output": result.output,
            "reason": result.reason,
            "evidence": dict(result.evidence),
        }
        if include_id:
            value["id"] = request_id
        print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)

    def emit_line_error(reason: str, *, request_id=None, include_id=False) -> None:
        value = {
            "status": "error",
            "component": "qwen3.8.causal-chat",
            "output": None,
            "reason": reason,
            "evidence": {},
        }
        if include_id:
            value["id"] = request_id
        print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)

    def interactive_summary(result) -> str | None:
        evidence = result.evidence
        if not isinstance(evidence, dict):
            evidence = dict(evidence)
        parts = []
        economics = evidence.get("inference_economics")
        if isinstance(economics, dict) and economics.get("status") in {
            "recorded",
            "duplicate",
        }:
            receipt = economics.get("receipt")
            body = receipt.get("body") if isinstance(receipt, dict) else None
            rollup = economics.get("rollup")
            if isinstance(body, dict):
                target = body.get("target_forwards")
                saved = body.get("saved_qwen_forwards")
                if isinstance(target, int) and isinstance(saved, int):
                    parts.append(f"economics {target} target / {saved} saved")
            if isinstance(rollup, dict):
                requests = rollup.get("requests")
                saved = rollup.get("saved_qwen_forwards")
                largest = rollup.get("largest_avoidable_cost_class")
                if isinstance(requests, int) and isinstance(saved, int):
                    summary = f"ledger {requests} requests / {saved} saved"
                    if isinstance(largest, str) and largest:
                        summary += f" / next {largest}"
                    parts.append(summary)
        action_bank = evidence.get("inference_action_bank")
        if isinstance(action_bank, dict) and action_bank.get("status") in {
            "recorded",
            "duplicate",
        }:
            receipt = action_bank.get("receipt")
            body = receipt.get("body") if isinstance(receipt, dict) else None
            snapshot = action_bank.get("snapshot")
            if isinstance(body, dict) and isinstance(body.get("actions"), list):
                parts.append("actions " + "+".join(body["actions"]))
            if isinstance(snapshot, dict):
                requests = snapshot.get("requests")
                if isinstance(requests, int) and not isinstance(requests, bool):
                    parts.append(f"action bank {requests} requests")
        generation = evidence.get("generation")
        runtime_metrics = evidence.get("runtime_metrics")
        if not isinstance(generation, dict):
            return "; ".join(parts) or None
        conversation = evidence.get("conversation")
        if isinstance(conversation, dict):
            history_turns = conversation.get("history_turns")
            if isinstance(history_turns, int) and not isinstance(history_turns, bool):
                parts.append(f"{history_turns} prior turns")
            reuse_status = conversation.get("reuse_status")
            reused_prefix = conversation.get("reused_prefix_tokens")
            if reuse_status == "hit" and isinstance(reused_prefix, int):
                parts.append(f"{reused_prefix} cached prefix tokens")
            retained_prefix = conversation.get("state_retained_tokens")
            if (
                reuse_status != "hit"
                and isinstance(retained_prefix, int)
                and retained_prefix > 0
            ):
                parts.append(f"{retained_prefix} prefix tokens cached")
            mtp_carry_status = conversation.get("mtp_carry_status")
            mtp_carry_bytes = conversation.get("mtp_carry_bytes")
            if (
                mtp_carry_status in {"stored", "reused+stored"}
                and isinstance(mtp_carry_bytes, int)
                and mtp_carry_bytes > 0
            ):
                parts.append(f"{mtp_carry_bytes / 1024:.1f} KiB MTP carry")
        generated_tokens = generation.get("generated_tokens")
        if isinstance(generated_tokens, int) and not isinstance(generated_tokens, bool):
            parts.append(f"{generated_tokens} tokens")
        forwards = generation.get("forward_passes")
        if isinstance(forwards, int) and not isinstance(forwards, bool):
            parts.append(f"{forwards} Qwen forwards")
        seconds = generation.get("seconds")
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
            parts.append(f"{float(seconds):.2f} s")
        if isinstance(runtime_metrics, dict):
            peak_rss = runtime_metrics.get("process_peak_rss_bytes")
            if isinstance(peak_rss, int) and not isinstance(peak_rss, bool):
                parts.append(f"{peak_rss / 1024**3:.2f} GiB peak")
        page_route = evidence.get("mlp_page_route")
        if isinstance(page_route, dict):
            page_request = page_route.get("request")
            if isinstance(page_request, dict):
                dynamic_calls = page_request.get("dynamic_route_calls")
                dynamic_changes = page_request.get("dynamic_route_changes")
                exact_rows = page_request.get("exact_rows")
                coactive_updates = page_request.get("coactive_updates")
                width_saved = page_request.get("adaptive_width_pages_saved")
                energy_rows = page_request.get("energy_feedback_rows")
                page_parts = []
                if (
                    isinstance(dynamic_calls, int)
                    and not isinstance(dynamic_calls, bool)
                    and dynamic_calls > 0
                ):
                    page_parts.append(
                        f"{dynamic_calls} dynamic routes"
                        + (
                            f", {dynamic_changes} changed"
                            if isinstance(dynamic_changes, int)
                            and not isinstance(dynamic_changes, bool)
                            and dynamic_changes > 0
                            else ""
                        )
                    )
                if (
                    isinstance(exact_rows, int)
                    and not isinstance(exact_rows, bool)
                    and exact_rows > 0
                ):
                    page_parts.append(f"{exact_rows} exact rows learned")
                if (
                    isinstance(coactive_updates, int)
                    and not isinstance(coactive_updates, bool)
                    and coactive_updates > 0
                ):
                    page_parts.append(f"{coactive_updates} coactive edges")
                if (
                    isinstance(width_saved, int)
                    and not isinstance(width_saved, bool)
                    and width_saved > 0
                ):
                    page_parts.append(f"{width_saved} pages skipped")
                if (
                    isinstance(energy_rows, int)
                    and not isinstance(energy_rows, bool)
                    and energy_rows > 0
                ):
                    page_parts.append(f"{energy_rows} energy labels")
                page_runtime = page_route.get("runtime")
                if isinstance(page_runtime, dict):
                    width_min = page_runtime.get("last_width_min")
                    width_mean = page_runtime.get("last_width_mean")
                    if (
                        isinstance(width_min, int)
                        and not isinstance(width_min, bool)
                        and isinstance(width_mean, (int, float))
                        and not isinstance(width_mean, bool)
                    ):
                        page_parts.append(f"width {width_min}-{float(width_mean):.1f}")
                if page_parts:
                    parts.append("MLP pages " + ", ".join(page_parts))
        q4 = evidence.get("q4")
        if isinstance(q4, dict):
            q4_request = q4.get("request")
            if isinstance(q4_request, dict):
                prefetch_calls = q4_request.get("page_mlp_prefetch_calls")
                prefetch_pages = q4_request.get("page_mlp_prefetch_pages")
                requested_pages = q4_request.get("page_mlp_prefetch_requested_pages")
                selected_pages = q4_request.get("page_mlp_prefetch_selected_pages")
                prefetch_bytes = q4_request.get("page_mlp_prefetch_bytes")
                consumed = q4_request.get("page_mlp_prefetch_consumed_leases")
                declines = q4_request.get("page_mlp_prefetch_budget_declines")
                budget_fraction_sum = q4_request.get(
                    "page_mlp_prefetch_budget_fraction_sum_ppm"
                )
                trims = q4_request.get("page_mlp_prefetch_trimmed_pages")
                failures = q4_request.get("page_mlp_prefetch_failures")
                if (
                    isinstance(prefetch_calls, int)
                    and not isinstance(prefetch_calls, bool)
                    and prefetch_calls > 0
                    and isinstance(prefetch_pages, int)
                    and not isinstance(prefetch_pages, bool)
                    and isinstance(prefetch_bytes, int)
                    and not isinstance(prefetch_bytes, bool)
                ):
                    detail = (
                        f"Q4 lookahead {prefetch_calls} calls, "
                        f"{prefetch_pages} pages fully advised, "
                        f"{prefetch_bytes / 1024**2:.1f} MiB advised"
                    )
                    if isinstance(budget_fraction_sum, int) and not isinstance(
                        budget_fraction_sum, bool
                    ):
                        detail += (
                            f", {budget_fraction_sum / prefetch_calls / 10_000:.1f}% "
                            "mean budget"
                        )
                    if isinstance(consumed, int) and not isinstance(consumed, bool):
                        detail += f", {consumed} leases consumed"
                    if (
                        isinstance(selected_pages, int)
                        and not isinstance(selected_pages, bool)
                        and selected_pages > prefetch_pages
                    ):
                        detail += f", {selected_pages} pages budget-selected"
                    if (
                        isinstance(requested_pages, int)
                        and not isinstance(requested_pages, bool)
                        and requested_pages > prefetch_pages
                    ):
                        detail += f", {requested_pages} pages requested"
                    if (
                        isinstance(trims, int)
                        and not isinstance(trims, bool)
                        and trims > 0
                    ):
                        detail += f", {trims} pages budget-trimmed"
                    if (
                        isinstance(declines, int)
                        and not isinstance(declines, bool)
                        and declines > 0
                    ):
                        detail += f", {declines} budget declines"
                    if (
                        isinstance(failures, int)
                        and not isinstance(failures, bool)
                        and failures > 0
                    ):
                        detail += f", {failures} advice failures"
                    parts.append(detail)
        runtime_reward = evidence.get("runtime_reward")
        if isinstance(runtime_reward, dict):
            reward = runtime_reward.get("reward")
            accepted_reward = runtime_reward.get("accepted_draft_tokens")
            saved_reward = runtime_reward.get("page_actions_saved")
            o1_reward = runtime_reward.get("o1_priority")
            if (
                isinstance(reward, (int, float))
                and not isinstance(reward, bool)
                and isinstance(accepted_reward, int)
                and not isinstance(accepted_reward, bool)
                and isinstance(saved_reward, int)
                and not isinstance(saved_reward, bool)
                and isinstance(o1_reward, (int, float))
                and not isinstance(o1_reward, bool)
            ):
                parts.append(
                    f"joint reward {float(reward):.2f} "
                    f"(draft {accepted_reward}, pages {saved_reward}, "
                    f"O1 {float(o1_reward):.2f})"
                )
        draft = evidence.get("draft")
        if isinstance(draft, dict):
            accepted = draft.get("accepted_draft_tokens")
            if (
                isinstance(accepted, int)
                and not isinstance(accepted, bool)
                and accepted > 0
            ):
                parts.append(f"{accepted} accepted draft tokens")
            provider = draft.get("provider")
            markov = None
            mtp = None
            if isinstance(provider, dict):
                provider_parts = []
                provider_tournaments = provider.get("provider_tournament_calls")
                if (
                    isinstance(provider_tournaments, int)
                    and not isinstance(provider_tournaments, bool)
                    and provider_tournaments > 0
                ):
                    provider_selections = tuple(
                        provider.get(name, 0)
                        for name in (
                            "provider_tournament_markov_selections",
                            "provider_tournament_mtp_selections",
                        )
                    )
                    if all(
                        isinstance(value, int) and not isinstance(value, bool)
                        for value in provider_selections
                    ):
                        provider_parts.append(
                            f"{provider_tournaments} provider tournaments "
                            f"Markov{provider_selections[0]}/MTP{provider_selections[1]}"
                        )
                provider_feedback = provider.get("provider_trace_feedback_tokens")
                if (
                    isinstance(provider_feedback, int)
                    and not isinstance(provider_feedback, bool)
                    and provider_feedback > 0
                ):
                    provider_parts.append(
                        f"{provider_feedback} provider counterfactual labels"
                    )
                if provider_parts:
                    parts.append("Hybrid " + ", ".join(provider_parts))
                nested_markov = provider.get("markov")
                nested_mtp = provider.get("mtp")
                markov = (
                    nested_markov
                    if isinstance(nested_markov, dict)
                    else provider
                    if "planner_tournament_calls" in provider
                    else None
                )
                mtp = (
                    nested_mtp
                    if isinstance(nested_mtp, dict)
                    else provider
                    if "teacher_verifications" in provider
                    else None
                )
            if isinstance(markov, dict):
                markov_parts = []
                tournaments = markov.get("planner_tournament_calls")
                if (
                    isinstance(tournaments, int)
                    and not isinstance(tournaments, bool)
                    and tournaments > 0
                ):
                    selections = tuple(
                        markov.get(name, 0)
                        for name in (
                            "planner_beam_selections",
                            "planner_council_selections",
                            "planner_phrase_selections",
                        )
                    )
                    if all(
                        isinstance(value, int) and not isinstance(value, bool)
                        for value in selections
                    ):
                        markov_parts.append(
                            f"{tournaments} tournaments "
                            f"B{selections[0]}/C{selections[1]}/P{selections[2]}"
                        )
                counterfactual = markov.get("planner_trace_feedback_tokens")
                if (
                    isinstance(counterfactual, int)
                    and not isinstance(counterfactual, bool)
                    and counterfactual > 0
                ):
                    markov_parts.append(f"{counterfactual} counterfactual labels")
                deep = markov.get("recursive_trace_feedback_tokens")
                depth = markov.get("recursive_trace_max_position")
                if isinstance(deep, int) and not isinstance(deep, bool) and deep > 0:
                    markov_parts.append(
                        f"{deep} deep labels"
                        + (
                            f" through p{depth}"
                            if isinstance(depth, int)
                            and not isinstance(depth, bool)
                            and depth > 0
                            else ""
                        )
                    )
                similarity = markov.get("active_dialect_similarity")
                if (
                    isinstance(similarity, (int, float))
                    and not isinstance(similarity, bool)
                    and float(similarity) > 0.0
                ):
                    markov_parts.append(f"dialect {float(similarity):.2f}")
                ricci_builds = markov.get("ricci_working_set_builds")
                ricci_episodes = markov.get("ricci_working_set_selected_episodes")
                ricci_tokens = markov.get("ricci_working_set_selected_tokens")
                ricci_age = markov.get("ricci_working_set_oldest_age")
                if (
                    isinstance(ricci_builds, int)
                    and not isinstance(ricci_builds, bool)
                    and ricci_builds > 0
                    and isinstance(ricci_episodes, int)
                    and not isinstance(ricci_episodes, bool)
                    and isinstance(ricci_tokens, int)
                    and not isinstance(ricci_tokens, bool)
                ):
                    markov_parts.append(
                        f"Ricci PPM {ricci_episodes} episodes/"
                        f"{ricci_tokens} tokens"
                        + (
                            f", age {ricci_age}"
                            if isinstance(ricci_age, int)
                            and not isinstance(ricci_age, bool)
                            and ricci_age > 0
                            else ""
                        )
                    )
                if markov_parts:
                    parts.append("Markov " + ", ".join(markov_parts))
            if isinstance(mtp, dict):
                teacher = mtp.get("teacher_verifications")
                if (
                    isinstance(teacher, int)
                    and not isinstance(teacher, bool)
                    and teacher > 0
                ):
                    parts.append(f"{teacher} free MTP teacher labels")
                recursive_mtp = mtp.get("recursive_trace_feedback_tokens")
                recursive_mtp_depth = mtp.get("recursive_trace_max_position")
                if (
                    isinstance(recursive_mtp, int)
                    and not isinstance(recursive_mtp, bool)
                    and recursive_mtp > 0
                ):
                    parts.append(
                        f"{recursive_mtp} recursive MTP labels"
                        + (
                            f" through p{recursive_mtp_depth}"
                            if isinstance(recursive_mtp_depth, int)
                            and not isinstance(recursive_mtp_depth, bool)
                            and recursive_mtp_depth > 0
                            else ""
                        )
                    )
        return None if not parts else "[" + " · ".join(parts) + "]"

    def service_event_sink(event) -> None:
        if event.event != "candidate_delta":
            return
        snapshot = event.body.get("snapshot")
        if not isinstance(snapshot, str):
            raise QwenServiceError("service candidate snapshot is invalid")
        if live_writer is not None:
            live_writer.update(snapshot)
        elif progress_writer is not None:
            progress_writer.update(snapshot)

    def run_socket_frontend(client: UnixQwenServiceClient) -> int:
        if jsonl:
            failures = 0
            handled = 0
            for raw in sys.stdin:
                line = raw.strip()
                if not line:
                    continue
                if max_requests is not None and handled >= max_requests:
                    break
                handled += 1
                request_id = None
                include_id = False
                try:
                    if line.startswith("{"):
                        document = json.loads(line)
                        if not isinstance(document, dict) or set(document) - {
                            "id",
                            "message",
                        }:
                            raise ValueError(
                                "JSONL request must contain only id/message"
                            )
                        include_id = "id" in document
                        request_id = document.get("id")
                        line_message = document.get("message")
                    else:
                        line_message = line
                    if not isinstance(line_message, str) or not line_message.strip():
                        raise ValueError("JSONL request message must be non-empty text")
                    internal_request_id = (
                        hashlib.sha256(
                            json.dumps(
                                {
                                    "external_id": request_id,
                                    "message_sha256": hashlib.sha256(
                                        line_message.strip().encode("utf-8")
                                    ).hexdigest(),
                                },
                                allow_nan=False,
                                ensure_ascii=True,
                                separators=(",", ":"),
                                sort_keys=True,
                            ).encode("ascii")
                        ).hexdigest()
                        if include_id
                        else secrets.token_hex(16)
                    )
                    response = client.request(
                        "chat",
                        session_id=None,
                        message=line_message,
                        request_id=internal_request_id,
                        event_sink=service_event_sink,
                    )
                    emit(
                        response.result,
                        request_id=request_id,
                        include_id=include_id,
                    )
                    failures += int(not response.result.ok)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    failures += 1
                    emit_line_error(
                        f"{type(exc).__name__}: {exc}",
                        request_id=request_id,
                        include_id=include_id,
                    )
                if max_requests is not None and handled >= max_requests:
                    break
            return 0 if failures == 0 else 2

        if interactive:
            failures = 0
            handled = 0
            terminal = sys.stdin.isatty() and sys.stdout.isatty()
            session_id = f"interactive:{secrets.token_hex(16)}"
            if terminal:
                print(
                    "IMMER local Qwen service — /help, /stats, /clear, /quit",
                    flush=True,
                )
            while max_requests is None or handled < max_requests:
                if terminal:
                    print("you> ", end="", flush=True)
                raw = sys.stdin.readline()
                if raw == "":
                    break
                line_message = raw.strip()
                if not line_message:
                    continue
                if line_message in {"/exit", "/quit"}:
                    break
                if line_message == "/help":
                    print(
                        "Enter any prompt. /stats shows the last real runtime cost. "
                        "/clear drops conversation context. /quit disconnects while "
                        "the local Qwen service stays loaded.",
                        flush=True,
                    )
                    continue
                if line_message == "/stats":
                    response = client.request("stats", session_id=session_id)
                    print(response.result.output, flush=True)
                    continue
                if line_message == "/clear":
                    client.request("clear", session_id=session_id)
                    if terminal:
                        print("Conversation context cleared.", flush=True)
                    continue
                handled += 1
                if terminal:
                    print("immer> ", end="", flush=True)
                response = client.request(
                    "chat",
                    session_id=session_id,
                    message=line_message,
                    event_sink=service_event_sink,
                )
                emit(response.result)
                failures += int(not response.result.ok)
                summary = interactive_summary(response.result)
                if terminal and summary is not None:
                    print(summary, file=sys.stderr, flush=True)
                if live_writer is not None:
                    live_writer.reset()
            client.request("clear", session_id=session_id)
            return 0 if failures == 0 else 2

        response = client.request(
            "chat",
            session_id=None,
            message=message,
            event_sink=service_event_sink,
        )
        emit(response.result)
        return 0 if response.result.ok else 2

    try:
        if service:
            if direct:
                raise ValueError("--service and --direct are mutually exclusive")
            if message is not None or jsonl or interactive:
                raise ValueError(
                    "--service is mutually exclusive with a message, --jsonl, "
                    "and --interactive"
                )
            if max_requests is not None:
                raise ValueError("--max-requests does not apply to --service")
        elif jsonl:
            if output_mode != "json":
                raise ValueError("JSONL chat requires --output json")
            if message is not None:
                raise ValueError("chat message and --jsonl are mutually exclusive")
            if interactive:
                raise ValueError("--jsonl and --interactive are mutually exclusive")
            if max_requests is not None and (
                isinstance(max_requests, bool)
                or not isinstance(max_requests, int)
                or max_requests <= 0
            ):
                raise ValueError("max_requests must be a positive integer")
        elif interactive:
            if message is not None:
                raise ValueError(
                    "chat message and --interactive are mutually exclusive"
                )
            if output_mode != "text":
                raise ValueError("interactive chat requires --output text")
            if max_requests is not None and (
                isinstance(max_requests, bool)
                or not isinstance(max_requests, int)
                or max_requests <= 0
            ):
                raise ValueError("max_requests must be a positive integer")
        elif not isinstance(message, str) or not message.strip():
            raise ValueError("chat requires a message, --jsonl, or --interactive")
        bundle_path, tokenizer_path, q4_root, fast_mlp_root = (
            _resolve_qwen38_chat_paths(args)
        )
        if q4_resident_budget_mb is None:
            q4_resident_budget_mb = (
                _QWEN38_DEPLOYMENT_Q4_RESIDENT_BUDGET_MB
                if bundle_path == _QWEN38_DEPLOYMENT_ROOT and q4_root is not None
                else 0
            )
        args.q4_resident_budget_mb = q4_resident_budget_mb
        if args.q4_resident_budget_mb and q4_root is None:
            raise ValueError("--q4-resident-budget-mb requires local Q4 execution")
        _resolve_qwen38_prefix_sinkhorn(args, bundle_path, q4_root)
        service_socket = _resolve_qwen38_service_socket(args, bundle_path)
        args.service_socket = service_socket
        draft_mode, markov_draft_state, mtp_draft_state = _resolve_qwen38_markov_draft(
            args,
            bundle_path,
            q4_root,
        )
        disable_external_drafter = bool(
            getattr(args, "no_external_drafter", False)
        )
        if disable_external_drafter and getattr(args, "draft_bundle", None) is not None:
            raise ValueError(
                "--draft-bundle and --no-external-drafter are mutually exclusive"
            )
        if (
            not disable_external_drafter
            and draft_mode == "hybrid"
            and getattr(args, "draft_bundle", None) is None
            and bundle_path == _QWEN38_DEPLOYMENT_ROOT
            and _QWEN35_DEPLOYMENT_DRAFT_ROOT.is_dir()
        ):
            args.draft_bundle = str(_QWEN35_DEPLOYMENT_DRAFT_ROOT)
        markov_atlas_path = _chat_path(
            getattr(args, "markov_atlas", None),
            "IMMER_QWEN38_MARKOV_ATLAS",
        )
        markov_o1_retention_path = _chat_path(
            getattr(args, "markov_o1_retention", None),
            "IMMER_QWEN38_MARKOV_O1_RETENTION",
        )
        disable_mlp_page_route = bool(getattr(args, "no_mlp_page_route", False))
        if disable_mlp_page_route and getattr(args, "mlp_page_state", None):
            raise ValueError(
                "--mlp-page-state and --no-mlp-page-route are mutually exclusive"
            )
        mlp_page_state_path = (
            None
            if disable_mlp_page_route
            else _chat_path(
                getattr(args, "mlp_page_state", None),
                "IMMER_QWEN38_MLP_PAGE_STATE",
            )
        )
        if (
            not disable_mlp_page_route
            and mlp_page_state_path is None
            and bundle_path == _QWEN38_DEPLOYMENT_ROOT
            and q4_root is not None
            and q4_root.name == "q4-base-v3-mtp"
        ):
            mlp_page_state_path = _QWEN38_DEPLOYMENT_MLP_PAGE_STATE
        if mlp_page_state_path is not None:
            if q4_root is None:
                raise ValueError("MLP page routing requires local Q4 execution")
            if getattr(args, "fast_mlp", None) is not None:
                raise ValueError(
                    "--mlp-page-state and --fast-mlp are mutually exclusive"
                )
            fast_mlp_root = None
        disable_mlp_page_coordinate = bool(
            getattr(args, "no_mlp_page_coordinate", False)
        )
        if (
            disable_mlp_page_coordinate
            and getattr(
                args,
                "mlp_page_coordinate_state",
                None,
            )
            is not None
        ):
            raise ValueError(
                "--mlp-page-coordinate-state and "
                "--no-mlp-page-coordinate are mutually exclusive"
            )
        mlp_page_coordinate_state_path = (
            None
            if disable_mlp_page_coordinate
            else _chat_path(
                getattr(args, "mlp_page_coordinate_state", None),
                "IMMER_QWEN38_MLP_PAGE_COORDINATE_STATE",
            )
        )
        if (
            not disable_mlp_page_coordinate
            and mlp_page_coordinate_state_path is None
            and bundle_path == _QWEN38_DEPLOYMENT_ROOT
            and q4_root is not None
            and mlp_page_state_path is not None
        ):
            mlp_page_coordinate_state_path = (
                _QWEN38_DEPLOYMENT_MLP_PAGE_COORDINATE_STATE
            )
        if mlp_page_coordinate_state_path is not None and (
            q4_root is None or mlp_page_state_path is None
        ):
            raise ValueError("MLP page coordinates require local Q4 MLP page routing")
        if mlp_page_coordinate_state_path is not None and args.compute_dtype not in {
            "auto",
            "bfloat16",
        }:
            raise ValueError(
                "MLP page coordinates require bfloat16 compute; pass "
                "--no-mlp-page-coordinate to use another dtype"
            )
        args.mlp_page_coordinate_state = mlp_page_coordinate_state_path
        if (
            bool(getattr(args, "no_markov_draft", False))
            and markov_atlas_path is not None
        ):
            raise ValueError(
                "--markov-atlas and --no-markov-draft are mutually exclusive"
            )
        if (
            bool(getattr(args, "no_markov_draft", False))
            and markov_o1_retention_path is not None
        ):
            raise ValueError(
                "--markov-o1-retention and --no-markov-draft are mutually exclusive"
            )
        if draft_mode is None and markov_atlas_path is not None:
            draft_mode = "markov"
        if draft_mode is None and markov_o1_retention_path is not None:
            draft_mode = "markov"
        if (
            markov_atlas_path is None
            and draft_mode in {None, "hybrid", "markov"}
            and not bool(getattr(args, "no_markov_draft", False))
            and bundle_path == _QWEN38_DEPLOYMENT_ROOT
            and _QWEN38_DEPLOYMENT_MARKOV_ATLAS.is_file()
        ):
            markov_atlas_path = _QWEN38_DEPLOYMENT_MARKOV_ATLAS
            if draft_mode is None:
                draft_mode = "markov"
        if (
            markov_o1_retention_path is None
            and draft_mode in {"hybrid", "markov"}
            and bundle_path == _QWEN38_DEPLOYMENT_ROOT
        ):
            markov_o1_retention_path = _QWEN38_DEPLOYMENT_O1_RETENTION
        disable_context_crystal = bool(getattr(args, "no_context_crystal", False))
        if (
            disable_context_crystal
            and getattr(
                args,
                "context_crystal_state",
                None,
            )
            is not None
        ):
            raise ValueError(
                "--context-crystal-state and --no-context-crystal are "
                "mutually exclusive"
            )
        context_crystal_state_path = (
            None
            if disable_context_crystal
            else _chat_path(
                getattr(args, "context_crystal_state", None),
                "IMMER_QWEN38_CONTEXT_CRYSTAL_STATE",
            )
        )
        if (
            not disable_context_crystal
            and context_crystal_state_path is None
            and draft_mode in {"hybrid", "markov"}
            and bundle_path == _QWEN38_DEPLOYMENT_ROOT
            and q4_root is not None
        ):
            context_crystal_state_path = _QWEN38_DEPLOYMENT_CONTEXT_CRYSTAL_STATE
        if context_crystal_state_path is not None and (
            draft_mode not in {"hybrid", "markov"} or q4_root is None
        ):
            raise ValueError(
                "contextual continuation Crystals require Q4 Markov or hybrid drafting"
            )
        args.context_crystal_state = context_crystal_state_path
        disable_layer_contextual_continuation = bool(
            getattr(args, "no_layer_contextual_continuation", False)
        )
        if disable_layer_contextual_continuation and getattr(
            args,
            "layer_contextual_continuation_state",
            None,
        ) is not None:
            raise ValueError(
                "--layer-contextual-continuation-state and "
                "--no-layer-contextual-continuation are mutually exclusive"
            )
        layer_contextual_continuation_state_path = (
            None
            if disable_layer_contextual_continuation
            else _chat_path(
                getattr(args, "layer_contextual_continuation_state", None),
                "IMMER_QWEN38_LAYER_CONTEXTUAL_CONTINUATION_STATE",
            )
        )
        if (
            not disable_layer_contextual_continuation
            and layer_contextual_continuation_state_path is None
            and draft_mode in {"hybrid", "markov"}
            and bundle_path == _QWEN38_DEPLOYMENT_ROOT
            and q4_root is not None
        ):
            layer_contextual_continuation_state_path = (
                _QWEN38_DEPLOYMENT_LAYER_CONTEXTUAL_CONTINUATION_STATE
            )
        if layer_contextual_continuation_state_path is not None and (
            draft_mode not in {"hybrid", "markov"} or q4_root is None
        ):
            raise ValueError(
                "layer contextual continuation requires Q4 Markov or hybrid "
                "drafting"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 65_536
            for value in (
                args.layer_contextual_continuation_max_cells,
                args.layer_contextual_continuation_max_receipts,
            )
        ):
            raise ValueError(
                "layer contextual continuation capacities must lie in [1, 65536]"
            )
        args.layer_contextual_continuation_state = (
            layer_contextual_continuation_state_path
        )
        disable_attention_output_crystal = bool(
            getattr(args, "no_attention_output_crystal", False)
        )
        if (
            disable_attention_output_crystal
            and getattr(
                args,
                "attention_output_crystal_state",
                None,
            )
            is not None
        ):
            raise ValueError(
                "--attention-output-crystal-state and "
                "--no-attention-output-crystal are mutually exclusive"
            )
        attention_output_crystal_state_path = (
            None
            if disable_attention_output_crystal
            else _chat_path(
                getattr(args, "attention_output_crystal_state", None),
                "IMMER_QWEN38_ATTENTION_OUTPUT_CRYSTAL_STATE",
            )
        )
        if (
            not disable_attention_output_crystal
            and attention_output_crystal_state_path is None
            and bundle_path == _QWEN38_DEPLOYMENT_ROOT
            and q4_root is not None
        ):
            attention_output_crystal_state_path = (
                _QWEN38_DEPLOYMENT_ATTENTION_OUTPUT_CRYSTAL_STATE
            )
        if attention_output_crystal_state_path is not None and q4_root is None:
            raise ValueError("attention-output Crystals require local Q4 execution")
        args.attention_output_crystal_state = attention_output_crystal_state_path
        layer_transition_crystal_state_path = _chat_path(
            getattr(args, "layer_transition_crystal_state", None),
            "IMMER_QWEN38_LAYER_TRANSITION_CRYSTAL_STATE",
        )
        layer_transition_crystal_atlas_path = _chat_path(
            getattr(args, "layer_transition_crystal_atlas", None),
            "IMMER_QWEN38_LAYER_TRANSITION_ATLAS",
        )
        layer_transition_crystal_compute_root = _chat_path(
            getattr(args, "layer_transition_crystal_compute_root", None),
            "IMMER_QWEN38_LAYER_TRANSITION_COMPUTE_ROOT",
        )
        layer_transition_crystal_max_error_radius = _finite_non_negative_float(
            getattr(
                args,
                "layer_transition_crystal_max_error_radius",
                0.0,
            )
        )
        if layer_transition_crystal_state_path is not None and q4_root is None:
            raise ValueError("layer-transition Crystals require local Q4 execution")
        if (
            layer_transition_crystal_state_path is not None
            and not layer_transition_crystal_state_path.is_file()
        ):
            raise ValueError(
                "--layer-transition-crystal-state must name an existing sealed file"
            )
        if (
            layer_transition_crystal_state_path is not None
            and layer_transition_crystal_atlas_path is None
        ):
            raise ValueError(
                "--layer-transition-crystal-state requires "
                "--layer-transition-crystal-atlas"
            )
        if (
            layer_transition_crystal_state_path is not None
            and layer_transition_crystal_compute_root is None
        ):
            raise ValueError(
                "--layer-transition-crystal-state requires "
                "--layer-transition-crystal-compute-root"
            )
        if (
            layer_transition_crystal_state_path is None
            and layer_transition_crystal_atlas_path is not None
        ):
            raise ValueError(
                "--layer-transition-crystal-atlas requires "
                "--layer-transition-crystal-state"
            )
        if (
            layer_transition_crystal_state_path is None
            and layer_transition_crystal_compute_root is not None
        ):
            raise ValueError(
                "--layer-transition-crystal-compute-root requires "
                "--layer-transition-crystal-state"
            )
        if layer_transition_crystal_atlas_path is not None:
            _require_existing_real_directory(
                layer_transition_crystal_atlas_path,
                "--layer-transition-crystal-atlas",
            )
        if layer_transition_crystal_compute_root is not None:
            _require_existing_real_directory(
                layer_transition_crystal_compute_root,
                "--layer-transition-crystal-compute-root",
            )
        if (
            layer_transition_crystal_state_path is not None
            and args.compute_dtype not in {"auto", "bfloat16"}
        ):
            raise ValueError("layer-transition Crystals require bfloat16 compute")
        if (
            layer_transition_crystal_state_path is None
            and layer_transition_crystal_max_error_radius != 0.0
        ):
            raise ValueError(
                "--layer-transition-crystal-max-error-radius requires "
                "--layer-transition-crystal-state"
            )
        args.layer_transition_crystal_state = layer_transition_crystal_state_path
        args.layer_transition_crystal_atlas = layer_transition_crystal_atlas_path
        args.layer_transition_crystal_compute_root = (
            layer_transition_crystal_compute_root
        )
        args.layer_transition_crystal_max_error_radius = (
            layer_transition_crystal_max_error_radius
        )
        layer_mlp_crystal_state_path = _chat_path(
            getattr(args, "layer_mlp_crystal_state", None),
            "IMMER_QWEN38_LAYER_MLP_CRYSTAL_STATE",
        )
        layer_mlp_crystal_registry_path = _chat_path(
            getattr(args, "layer_mlp_crystal_registry", None),
            "IMMER_QWEN38_LAYER_MLP_CRYSTAL_REGISTRY",
        )
        if (
            layer_mlp_crystal_state_path is not None
            and layer_mlp_crystal_registry_path is not None
        ):
            raise ValueError(
                "--layer-mlp-crystal-state and --layer-mlp-crystal-registry "
                "are mutually exclusive"
            )
        layer_mlp_crystal_atlas_path = _chat_path(
            getattr(args, "layer_mlp_crystal_atlas", None),
            "IMMER_QWEN38_LAYER_MLP_ATLAS",
        )
        layer_mlp_crystal_compute_root = _chat_path(
            getattr(args, "layer_mlp_crystal_compute_root", None),
            "IMMER_QWEN38_LAYER_MLP_COMPUTE_ROOT",
        )
        layer_mlp_crystal_max_error_radius = _finite_non_negative_float(
            getattr(args, "layer_mlp_crystal_max_error_radius", 0.0)
        )
        from .runtimes.qwen3_8.layer_mlp_registry import (
            LayerMlpO1RegistryNotMountableError,
            read_layer_mlp_o1_registry_manifest,
        )

        if (
            layer_mlp_crystal_state_path is None
            and layer_mlp_crystal_registry_path is None
            and bundle_path == _QWEN38_DEPLOYMENT_ROOT
            and q4_root is not None
            and _QWEN38_DEPLOYMENT_LAYER_MLP_CRYSTAL_REGISTRY.is_file()
        ):
            try:
                read_layer_mlp_o1_registry_manifest(
                    _QWEN38_DEPLOYMENT_LAYER_MLP_CRYSTAL_REGISTRY,
                    require_mountable=True,
                )
            except LayerMlpO1RegistryNotMountableError:
                pass
            else:
                layer_mlp_crystal_registry_path = (
                    _QWEN38_DEPLOYMENT_LAYER_MLP_CRYSTAL_REGISTRY
                )
                if layer_mlp_crystal_atlas_path is None:
                    layer_mlp_crystal_atlas_path = (
                        _QWEN38_DEPLOYMENT_LAYER_MLP_O1_ATLAS
                    )
                if layer_mlp_crystal_compute_root is None:
                    layer_mlp_crystal_compute_root = (
                        _QWEN38_DEPLOYMENT_LAYER_MLP_O1_COMPUTE
                    )
        layer_mlp_crystal_configured = (
            layer_mlp_crystal_state_path is not None
            or layer_mlp_crystal_registry_path is not None
        )
        if layer_mlp_crystal_configured and q4_root is None:
            raise ValueError("layer-MLP Crystals require local Q4 execution")
        if (
            layer_mlp_crystal_state_path is not None
            and not layer_mlp_crystal_state_path.is_file()
        ):
            raise ValueError(
                "--layer-mlp-crystal-state must name an existing sealed file"
            )
        if layer_mlp_crystal_registry_path is not None:
            if not layer_mlp_crystal_registry_path.is_file():
                raise ValueError(
                    "--layer-mlp-crystal-registry must name an existing manifest"
                )
            read_layer_mlp_o1_registry_manifest(
                layer_mlp_crystal_registry_path,
                require_mountable=True,
            )
        if (
            layer_mlp_crystal_configured
            and layer_mlp_crystal_atlas_path is None
        ):
            raise ValueError(
                "layer-MLP Crystals require --layer-mlp-crystal-atlas"
            )
        if (
            layer_mlp_crystal_configured
            and layer_mlp_crystal_compute_root is None
        ):
            raise ValueError(
                "layer-MLP Crystals require --layer-mlp-crystal-compute-root"
            )
        if (
            not layer_mlp_crystal_configured
            and layer_mlp_crystal_atlas_path is not None
        ):
            raise ValueError(
                "--layer-mlp-crystal-atlas requires a layer-MLP Crystal bank"
            )
        if (
            not layer_mlp_crystal_configured
            and layer_mlp_crystal_compute_root is not None
        ):
            raise ValueError(
                "--layer-mlp-crystal-compute-root requires a layer-MLP Crystal bank"
            )
        if layer_mlp_crystal_atlas_path is not None:
            _require_existing_real_directory(
                layer_mlp_crystal_atlas_path,
                "--layer-mlp-crystal-atlas",
            )
        if layer_mlp_crystal_compute_root is not None:
            _require_existing_real_directory(
                layer_mlp_crystal_compute_root,
                "--layer-mlp-crystal-compute-root",
            )
        if layer_mlp_crystal_configured and args.compute_dtype not in {
            "auto",
            "bfloat16",
        }:
            raise ValueError("layer-MLP Crystals require bfloat16 compute")
        if (
            layer_mlp_crystal_state_path is None
            and layer_mlp_crystal_max_error_radius != 0.0
        ):
            raise ValueError(
                "--layer-mlp-crystal-max-error-radius requires "
                "--layer-mlp-crystal-state"
            )
        args.layer_mlp_crystal_state = layer_mlp_crystal_state_path
        args.layer_mlp_crystal_registry = layer_mlp_crystal_registry_path
        args.layer_mlp_crystal_atlas = layer_mlp_crystal_atlas_path
        args.layer_mlp_crystal_compute_root = layer_mlp_crystal_compute_root
        args.layer_mlp_crystal_max_error_radius = layer_mlp_crystal_max_error_radius
        layer_mlp_o1_state_path = _chat_path(
            getattr(args, "layer_mlp_o1_state", None),
            "IMMER_QWEN38_LAYER_MLP_O1_STATE",
        )
        layer_mlp_o1_atlas_path = _chat_path(
            getattr(args, "layer_mlp_o1_atlas", None),
            "IMMER_QWEN38_LAYER_MLP_O1_ATLAS",
        )
        layer_mlp_o1_compute_root = _chat_path(
            getattr(args, "layer_mlp_o1_compute_root", None),
            "IMMER_QWEN38_LAYER_MLP_O1_COMPUTE_ROOT",
        )
        layer_mlp_o1_layers = getattr(args, "layer_mlp_o1_layers", None)
        if (
            layer_mlp_o1_state_path is None
            and bundle_path == _QWEN38_DEPLOYMENT_ROOT
            and q4_root is not None
            and _QWEN38_DEPLOYMENT_LAYER_MLP_O1_ATLAS.is_dir()
            and _QWEN38_DEPLOYMENT_LAYER_MLP_O1_COMPUTE.is_dir()
        ):
            layer_mlp_o1_state_path = _QWEN38_DEPLOYMENT_LAYER_MLP_O1_STATE
            layer_mlp_o1_atlas_path = _QWEN38_DEPLOYMENT_LAYER_MLP_O1_ATLAS
            layer_mlp_o1_compute_root = _QWEN38_DEPLOYMENT_LAYER_MLP_O1_COMPUTE
            if layer_mlp_o1_layers is None:
                layer_mlp_o1_layers = tuple(range(64))
        if layer_mlp_o1_state_path is not None:
            if q4_root is None:
                raise ValueError("layer-MLP O1 collection requires local Q4 execution")
            if args.compute_dtype not in {"auto", "bfloat16"}:
                raise ValueError("layer-MLP O1 collection requires bfloat16 compute")
            if layer_mlp_o1_atlas_path is None or layer_mlp_o1_compute_root is None:
                raise ValueError(
                    "--layer-mlp-o1-state requires --layer-mlp-o1-atlas and "
                    "--layer-mlp-o1-compute-root"
                )
            if (
                layer_mlp_o1_state_path.exists()
                and not layer_mlp_o1_state_path.is_file()
            ):
                raise ValueError(
                    "--layer-mlp-o1-state must name a regular file or new file"
                )
            _require_existing_real_directory(
                layer_mlp_o1_state_path.parent,
                "--layer-mlp-o1-state parent",
            )
        elif (
            layer_mlp_o1_atlas_path is not None or layer_mlp_o1_compute_root is not None
        ):
            raise ValueError("layer-MLP O1 authorities require --layer-mlp-o1-state")
        if layer_mlp_o1_state_path is None and layer_mlp_o1_layers is not None:
            raise ValueError("--layer-mlp-o1-layers requires --layer-mlp-o1-state")
        if layer_mlp_o1_layers is not None and any(
            layer > 63 for layer in layer_mlp_o1_layers
        ):
            raise ValueError("--layer-mlp-o1-layers must lie in [0, 63]")
        if layer_mlp_o1_atlas_path is not None:
            _require_existing_real_directory(
                layer_mlp_o1_atlas_path,
                "--layer-mlp-o1-atlas",
            )
        if layer_mlp_o1_compute_root is not None:
            _require_existing_real_directory(
                layer_mlp_o1_compute_root,
                "--layer-mlp-o1-compute-root",
            )
        if not 0 < args.layer_mlp_o1_sketch_dim <= 256:
            raise ValueError("--layer-mlp-o1-sketch-dim must lie in [1, 256]")
        if len(args.layer_mlp_o1_seed) != 64 or set(args.layer_mlp_o1_seed) - set(
            "0123456789abcdef"
        ):
            raise ValueError("--layer-mlp-o1-seed must be a lowercase SHA-256")
        if not math.isfinite(args.layer_mlp_o1_ridge) or args.layer_mlp_o1_ridge <= 0:
            raise ValueError("--layer-mlp-o1-ridge must be positive and finite")
        for value, option in (
            (args.layer_mlp_o1_coverage_guard, "--layer-mlp-o1-coverage-guard"),
            (args.layer_mlp_o1_error_guard, "--layer-mlp-o1-error-guard"),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{option} must be finite and non-negative")
        if args.layer_mlp_o1_queue_capacity <= 0:
            raise ValueError("--layer-mlp-o1-queue-capacity must be positive")
        args.layer_mlp_o1_state = layer_mlp_o1_state_path
        args.layer_mlp_o1_atlas = layer_mlp_o1_atlas_path
        args.layer_mlp_o1_compute_root = layer_mlp_o1_compute_root
        args.layer_mlp_o1_layers = layer_mlp_o1_layers
        _qwen38_layer_mlp_o1_policy(args)
        disable_draft_window = bool(getattr(args, "no_draft_window_controller", False))
        if disable_draft_window and getattr(args, "draft_window_state", None):
            raise ValueError(
                "--draft-window-state and --no-draft-window-controller are mutually exclusive"
            )
        draft_window_state_path = (
            None
            if disable_draft_window
            else _chat_path(
                getattr(args, "draft_window_state", None),
                "IMMER_QWEN38_DRAFT_WINDOW_STATE",
            )
        )
        if (
            not disable_draft_window
            and draft_window_state_path is None
            and draft_mode in {"hybrid", "markov"}
            and bundle_path == _QWEN38_DEPLOYMENT_ROOT
        ):
            draft_window_state_path = _QWEN38_DEPLOYMENT_DRAFT_WINDOW_STATE
        if bool(getattr(args, "no_anchor_cache", False)):
            if args.qwen38_anchor_cache is not None:
                raise ValueError(
                    "--qwen38-anchor-cache and --no-anchor-cache are mutually exclusive"
                )
            anchor_cache_path = None
        else:
            anchor_cache_path = args.qwen38_anchor_cache
            if (
                anchor_cache_path is None
                and bundle_path == _QWEN38_DEPLOYMENT_ROOT
                and _QWEN38_DEPLOYMENT_ANCHOR_CACHE.is_dir()
            ):
                anchor_cache_path = str(_QWEN38_DEPLOYMENT_ANCHOR_CACHE)
        args.qwen38_anchor_cache = anchor_cache_path
        runtime_code_revision = _qwen38_runtime_code_revision()
        warm_runtime_code_revision = None if args.raw_qwen else runtime_code_revision
        warm_profile_sha256 = (
            None
            if args.raw_qwen
            else _qwen38_growing_warm_profile(
                args,
                tokenizer_path=tokenizer_path,
                q4_root=q4_root,
                fast_mlp_root=fast_mlp_root,
                mlp_page_state_path=mlp_page_state_path,
                draft_mode=draft_mode,
                markov_atlas_path=markov_atlas_path,
                markov_o1_retention_path=markov_o1_retention_path,
                runtime_code_revision=warm_runtime_code_revision,
            )
        )
        warm_root = (
            None if args.raw_qwen else _resolve_qwen38_warm_root(args, bundle_path)
        )
        service_profile_sha256 = _qwen38_service_profile(
            args,
            bundle_path=bundle_path,
            tokenizer_path=tokenizer_path,
            q4_root=q4_root,
            fast_mlp_root=fast_mlp_root,
            warm_root=warm_root,
            draft_mode=draft_mode,
            markov_draft_state=markov_draft_state,
            mtp_draft_state=mtp_draft_state,
            markov_atlas_path=markov_atlas_path,
            markov_o1_retention_path=markov_o1_retention_path,
            mlp_page_state_path=mlp_page_state_path,
            attention_output_crystal_state_path=(attention_output_crystal_state_path),
            mlp_page_coordinate_enabled=(mlp_page_coordinate_state_path is not None),
            draft_window_state_path=draft_window_state_path,
            runtime_code_revision=runtime_code_revision,
        )
        if not service and not direct:
            service_client = UnixQwenServiceClient(service_socket)
            try:
                service_client.connect()
                ping = service_client.request(
                    "ping",
                    session_id=None,
                    request_id=secrets.token_hex(16),
                )
                remote_profile = ping.result.evidence.get("runtime_profile_sha256")
            except (OSError, QwenServiceError):
                service_client.close()
            else:
                if remote_profile == service_profile_sha256:
                    try:
                        return run_socket_frontend(service_client)
                    finally:
                        service_client.close()
                service_client.close()
        output_semantics = (
            None
            if args.raw_qwen
            or warm_profile_sha256 is None
            or bool(getattr(args, "prefix_sinkhorn", False))
            or getattr(args, "delta_head_online_state", None) is not None
            else _qwen38_output_semantics(
                args,
                tokenizer_path=tokenizer_path,
                q4_root=q4_root,
                mlp_page_state_path=mlp_page_state_path,
            )
        )
        economics_root = _resolve_qwen38_inference_economics(
            args,
            bundle_path,
        )
        economics_ledger = None
        economics_initialization_error = None
        action_bank = None
        action_bank_initialization_error = None
        if economics_root is not None:
            try:
                from .runtimes.qwen3_8.inference_economics import (
                    InferenceEconomicsLedger,
                )

                economics_ledger = InferenceEconomicsLedger(economics_root)
                from .runtimes.qwen3_8.action_bank import (
                    InferenceActionBank,
                    executed_actions_from_result,
                )

                action_bank = InferenceActionBank(
                    economics_root.parent / "qwen-inference-action-bank-v1"
                )
                action_bank.reconcile(economics_ledger.receipts())
            except Exception as exc:
                error = f"{type(exc).__module__}.{type(exc).__qualname__}"
                if economics_ledger is None:
                    economics_initialization_error = error
                else:
                    action_bank_initialization_error = error
        economics_runtime_profile = (
            None
            if economics_root is None
            else warm_profile_sha256
            or hashlib.sha256(
                json.dumps(
                    {
                        "bundle_path_sha256": hashlib.sha256(
                            str(bundle_path).encode("utf-8")
                        ).hexdigest(),
                        "q4_path_sha256": (
                            None
                            if q4_root is None
                            else hashlib.sha256(
                                str(q4_root).encode("utf-8")
                            ).hexdigest()
                        ),
                        "q4_manifest_file_sha256": (
                            None
                            if q4_root is None
                            or not (q4_root / "manifest.json").is_file()
                            else _path_sha256(q4_root / "manifest.json")
                        ),
                        "layer_transition_crystal": (
                            _qwen38_layer_transition_crystal_policy(args)
                        ),
                        "layer_mlp_crystal": _qwen38_layer_mlp_crystal_policy(args),
                        "layer_mlp_o1": _qwen38_layer_mlp_o1_policy(args),
                        "prefix_sinkhorn": bool(args.prefix_sinkhorn),
                        "raw_qwen": bool(args.raw_qwen),
                        "runtime_code_revision": runtime_code_revision,
                        "schema": "immer.qwen3.8-economics-runtime-profile/v2",
                        "tokenizer_path_sha256": hashlib.sha256(
                            str(tokenizer_path).encode("utf-8")
                        ).hexdigest(),
                        "tokenizer_file_sha256": (
                            _path_sha256(tokenizer_path)
                            if tokenizer_path.is_file()
                            else None
                        ),
                    },
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
            ).hexdigest()
        )
        economics_session_nonce = (
            None if economics_root is None else secrets.token_hex(16)
        )
        economics_request_ordinal = 0
        prompt_tokenizer = None
        if (
            warm_profile_sha256 is not None
            or output_semantics is not None
            or interactive
            or service
        ):
            from .runtimes.qwen3_8.encoding import Qwen38Tokenizer

            prompt_tokenizer = Qwen38Tokenizer(
                tokenizer_path,
                require_official=True,
            )

        def request_metadata_for(
            text: str,
            history: tuple[tuple[str, str], ...] = (),
            session_id: str | None = None,
        ) -> dict[str, object]:
            text = text.strip()
            metadata: dict[str, object] = {}
            if history:
                metadata[QWEN38_CHAT_HISTORY_METADATA] = history
            if session_id is not None:
                metadata[QWEN38_CHAT_SESSION_METADATA] = session_id
            if action_bank is not None and economics_runtime_profile is not None:
                try:
                    directive = action_bank.recommend(
                        question_sha256=hashlib.sha256(
                            text.encode("utf-8")
                        ).hexdigest(),
                        runtime_profile_sha256=economics_runtime_profile,
                    )
                except Exception:
                    directive = None
                if directive is not None:
                    metadata[QWEN38_INFERENCE_ACTION_METADATA] = directive.to_document()
            if warm_profile_sha256 is None or prompt_tokenizer is None:
                return metadata
            from .runtimes.qwen3_8.cartography_probe import (
                prompt_token_sha256,
            )

            rendered = (
                prompt_tokenizer.render_no_thinking_messages(
                    args.system_prompt,
                    (*history, ("user", text)),
                )
                if history
                else prompt_tokenizer.render_no_thinking_prompt(
                    args.system_prompt,
                    text,
                )
            )
            token_sha256 = prompt_token_sha256(prompt_tokenizer.encode(rendered))
            semantic_key = None
            if output_semantics is not None:
                from .runtimes.qwen3_8.output_semantics import (
                    QWEN_SEMANTIC_REPLAY_METADATA_KEY,
                    semantic_replay_key_for_prompt,
                )

                semantic_key = semantic_replay_key_for_prompt(
                    output_semantics,
                    question=text.strip(),
                    rendered_prompt=rendered,
                    rendered_prompt_token_sha256=token_sha256,
                    system_prompt=args.system_prompt.strip(),
                )
            metadata.update(
                {
                    "qwen_token_sha256": token_sha256,
                    "qwen_warm_runtime_profile_sha256": warm_profile_sha256,
                }
            )
            if semantic_key is not None:
                metadata[QWEN_SEMANTIC_REPLAY_METADATA_KEY] = semantic_key.to_document()
            return metadata

        def fit_interactive_history(
            text: str,
            history: tuple[tuple[str, str], ...],
        ) -> tuple[tuple[tuple[str, str], ...], int, int]:
            if prompt_tokenizer is None:
                raise RuntimeError("interactive tokenizer is unavailable")
            retained = history
            dropped = 0
            while True:
                rendered = prompt_tokenizer.render_no_thinking_messages(
                    args.system_prompt,
                    (*retained, ("user", text)),
                )
                token_count = len(prompt_tokenizer.encode(rendered))
                if (
                    token_count <= args.max_prompt_tokens and len(retained) <= 128
                ) or not retained:
                    return retained, dropped, token_count
                retained = retained[2:]
                dropped += 1

        def verify_prompt_token(question: str, claimed: str) -> bool:
            metadata = request_metadata_for(question)
            expected = metadata.get("qwen_token_sha256")
            return isinstance(expected, str) and hmac.compare_digest(
                expected,
                claimed,
            )

        def verify_semantic_key(
            question: str,
            claimed: object,
            metadata: object,
        ) -> bool:
            if not isinstance(metadata, dict):
                return False
            raw_history = metadata.get(QWEN38_CHAT_HISTORY_METADATA, ())
            if not isinstance(raw_history, tuple):
                return False
            try:
                expected = request_metadata_for(
                    question,
                    history=raw_history,
                ).get("qwen_semantic_replay_key")
                left = hashlib.sha256(
                    json.dumps(
                        expected,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("ascii")
                ).hexdigest()
                right = hashlib.sha256(
                    json.dumps(
                        claimed,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("ascii")
                ).hexdigest()
            except (TypeError, ValueError):
                return False
            return hmac.compare_digest(left, right)

        def attach_inference_economics(
            question: str,
            result: Result,
            *,
            request_id: object | None = None,
        ) -> Result:
            nonlocal economics_request_ordinal
            if economics_root is None:
                return result
            assert economics_runtime_profile is not None
            assert economics_session_nonce is not None
            question_sha256 = hashlib.sha256(
                question.strip().encode("utf-8")
            ).hexdigest()
            ordinal = economics_request_ordinal
            economics_request_ordinal += 1
            action_evidence: dict[str, object] | None = None
            try:
                request_identity = (
                    {
                        "external_id": request_id,
                        "question_sha256": question_sha256,
                        "runtime_profile_sha256": economics_runtime_profile,
                    }
                    if request_id is not None
                    else {
                        "ordinal": ordinal,
                        "question_sha256": question_sha256,
                        "runtime_profile_sha256": economics_runtime_profile,
                        "session_nonce": economics_session_nonce,
                    }
                )
                request_sha256 = hashlib.sha256(
                    json.dumps(
                        request_identity,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("ascii")
                ).hexdigest()
                if economics_ledger is None:
                    raise RuntimeError(
                        economics_initialization_error
                        or "inference economics is unavailable"
                    )
                observation = economics_ledger.observe(
                    result,
                    question_sha256=question_sha256,
                    runtime_profile_sha256=economics_runtime_profile,
                    request_sha256=request_sha256,
                )
                economics_evidence: dict[str, object] = {
                    "duplicate": observation.duplicate,
                    "receipt": observation.receipt.to_document(),
                    "rollup": dict(observation.rollup),
                    "status": "duplicate" if observation.duplicate else "recorded",
                }
                try:
                    if action_bank is None:
                        raise RuntimeError(
                            action_bank_initialization_error
                            or "inference action bank is unavailable"
                        )
                    action_observation = action_bank.observe(
                        observation.receipt,
                        executed_actions=executed_actions_from_result(
                            result,
                            observation.receipt,
                        ),
                    )
                    action_evidence = {
                        "duplicate": action_observation.duplicate,
                        "receipt": action_observation.receipt.to_document(),
                        "snapshot": dict(action_observation.snapshot),
                        "status": (
                            "duplicate" if action_observation.duplicate else "recorded"
                        ),
                    }
                except Exception as exc:
                    action_evidence = {
                        "error": f"{type(exc).__module__}.{type(exc).__qualname__}",
                        "status": "error",
                    }
            except Exception as exc:
                economics_evidence = {
                    "error": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "status": "error",
                }
            evidence = dict(result.evidence)
            evidence["inference_economics"] = economics_evidence
            if action_evidence is not None:
                evidence["inference_action_bank"] = action_evidence
            return Result(
                result.status,
                result.component,
                output=result.output,
                reason=result.reason,
                evidence=evidence,
            )

        warm_mount = None
        if not args.raw_qwen:
            if warm_root is not None:
                warm_mount = open_verified_qwen_warm_bank(
                    warm_root,
                    runtime_profile_sha256=warm_profile_sha256,
                    runtime_code_revision=(
                        warm_runtime_code_revision
                        if warm_profile_sha256 is not None
                        else None
                    ),
                    template_output_character_limit=(
                        args.max_new_tokens if warm_profile_sha256 is not None else None
                    ),
                    prompt_token_verifier=(
                        verify_prompt_token if warm_profile_sha256 is not None else None
                    ),
                    semantic_key_verifier=(
                        verify_semantic_key if warm_profile_sha256 is not None else None
                    ),
                )
        anchor_cache = (
            None
            if args.qwen38_anchor_cache is None
            else SemanticStateAnchorCache(args.qwen38_anchor_cache)
        )
        fast_mlp_layers = args.fast_mlp_layers
        fast_mlp_blocks = args.fast_mlp_blocks
        fast_mlp_policy = args.fast_mlp_policy
        if fast_mlp_policy == "auto":
            fast_mlp_policy = (
                "structure-edge"
                if q4_root is not None and fast_mlp_root is not None
                else "manual"
            )
        if fast_mlp_policy == "structure-edge":
            if fast_mlp_layers is not None:
                raise ValueError(
                    "--fast-mlp-policy structure-edge replaces --fast-mlp-layers"
                )
            fast_mlp_layers = (*range(18), *range(55, 64))
            if fast_mlp_blocks is None:
                fast_mlp_blocks = 64 if args.fast_mlp_online_state else 32
        native_head_crsa = None
        if bool(getattr(args, "prefix_sinkhorn", False)):
            from .runtimes.qwen3_8.native_crsa import Qwen38NativeHeadCrsa

            native_head_crsa = Qwen38NativeHeadCrsa(
                alpha=1.0,
                replace_base_softmax=True,
            )
        snapshot_bridge = SnapshotEventBridge() if service else None
        qwen = Qwen38CausalChat(
            str(bundle_path),
            str(tokenizer_path),
            system_prompt=args.system_prompt,
            device=args.device,
            compute_dtype=args.compute_dtype,
            source_budget_mb=args.source_budget_mb,
            max_resident_bytes=int(args.max_resident_mb * 1024**2),
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            max_context_tokens=args.max_context_tokens,
            head_block_rows=args.head_block_rows,
            exact_head_root=args.exact_head,
            exact_head_max_bytes=int(args.exact_head_max_mb * 1024**2),
            q4_root=None if q4_root is None else str(q4_root),
            q4_threads=args.q4_threads,
            q4_resident_budget_bytes=int(args.q4_resident_budget_mb * 1024**2),
            anchor_cache=anchor_cache,
            draft_bundle_path=args.draft_bundle,
            draft_mode=draft_mode,
            draft_window=args.draft_window,
            draft_source_budget_mb=args.draft_source_budget_mb,
            draft_max_resident_bytes=(
                None
                if args.draft_max_resident_mb is None
                else int(args.draft_max_resident_mb * 1024**2)
            ),
            markov_draft_state_path=markov_draft_state,
            markov_atlas_path=(
                None if markov_atlas_path is None else str(markov_atlas_path)
            ),
            markov_o1_retention_path=(
                None
                if markov_o1_retention_path is None
                else str(markov_o1_retention_path)
            ),
            contextual_continuation_state_path=(
                None
                if context_crystal_state_path is None
                else str(context_crystal_state_path)
            ),
            layer_contextual_continuation_state_path=(
                None
                if layer_contextual_continuation_state_path is None
                else str(layer_contextual_continuation_state_path)
            ),
            layer_contextual_continuation_max_cells=(
                args.layer_contextual_continuation_max_cells
            ),
            layer_contextual_continuation_max_receipts=(
                args.layer_contextual_continuation_max_receipts
            ),
            attention_output_crystal_state_path=(
                None
                if attention_output_crystal_state_path is None
                else str(attention_output_crystal_state_path)
            ),
            layer_transition_crystal_state_path=(
                None
                if layer_transition_crystal_state_path is None
                else str(layer_transition_crystal_state_path)
            ),
            layer_transition_crystal_atlas_path=(
                None
                if layer_transition_crystal_atlas_path is None
                else str(layer_transition_crystal_atlas_path)
            ),
            layer_transition_crystal_compute_root=(
                None
                if layer_transition_crystal_compute_root is None
                else str(layer_transition_crystal_compute_root)
            ),
            layer_transition_crystal_max_error_radius=(
                layer_transition_crystal_max_error_radius
            ),
            layer_mlp_crystal_state_path=(
                None
                if layer_mlp_crystal_state_path is None
                else str(layer_mlp_crystal_state_path)
            ),
            layer_mlp_crystal_registry_path=(
                None
                if layer_mlp_crystal_registry_path is None
                else str(layer_mlp_crystal_registry_path)
            ),
            layer_mlp_crystal_atlas_path=(
                None
                if layer_mlp_crystal_atlas_path is None
                else str(layer_mlp_crystal_atlas_path)
            ),
            layer_mlp_crystal_compute_root=(
                None
                if layer_mlp_crystal_compute_root is None
                else str(layer_mlp_crystal_compute_root)
            ),
            layer_mlp_crystal_max_error_radius=(layer_mlp_crystal_max_error_radius),
            layer_mlp_o1_state_path=(
                None
                if layer_mlp_o1_state_path is None
                else str(layer_mlp_o1_state_path)
            ),
            layer_mlp_o1_atlas_path=(
                None
                if layer_mlp_o1_atlas_path is None
                else str(layer_mlp_o1_atlas_path)
            ),
            layer_mlp_o1_compute_root=(
                None
                if layer_mlp_o1_compute_root is None
                else str(layer_mlp_o1_compute_root)
            ),
            layer_mlp_o1_layers=layer_mlp_o1_layers,
            layer_mlp_o1_sketch_dim=args.layer_mlp_o1_sketch_dim,
            layer_mlp_o1_projection_seed=args.layer_mlp_o1_seed,
            layer_mlp_o1_ridge=args.layer_mlp_o1_ridge,
            layer_mlp_o1_coverage_guard=args.layer_mlp_o1_coverage_guard,
            layer_mlp_o1_error_guard=args.layer_mlp_o1_error_guard,
            layer_mlp_o1_queue_capacity=args.layer_mlp_o1_queue_capacity,
            mtp_draft_state_path=mtp_draft_state,
            draft_window_state_path=(
                None
                if draft_window_state_path is None
                else str(draft_window_state_path)
            ),
            range_markov_state_path=args.range_markov_state,
            range_prefetch_max_bytes=int(args.range_prefetch_max_mb * 1024**2),
            range_prefetch_min_support=args.range_prefetch_min_support,
            range_prefetch_min_confidence=args.range_prefetch_min_confidence,
            range_prefetch_beam_horizon=args.range_prefetch_beam_horizon,
            range_prefetch_beam_width=args.range_prefetch_beam_width,
            range_prefetch_hint_cooldown=args.range_prefetch_hint_cooldown,
            fast_mlp_root=None if fast_mlp_root is None else str(fast_mlp_root),
            fast_mlp_online_state_path=args.fast_mlp_online_state,
            mlp_page_state_path=mlp_page_state_path,
            mlp_page_coordinate_state_path=(
                None
                if mlp_page_coordinate_state_path is None
                else str(mlp_page_coordinate_state_path)
            ),
            mlp_page_route_width=args.mlp_page_width,
            fast_mlp_source_budget_mb=args.fast_mlp_source_budget_mb,
            fast_mlp_max_resident_bytes=(
                None
                if args.fast_mlp_max_resident_mb is None
                else int(args.fast_mlp_max_resident_mb * 1024**2)
            ),
            fast_mlp_active_layers=fast_mlp_layers,
            fast_mlp_selected_block_count=fast_mlp_blocks,
            delta_head_state_path=args.delta_head_online_state,
            delta_head_active_layers=args.delta_head_layers,
            result_cell_code_revision=(
                None
                if warm_mount is None or warm_profile_sha256 is None
                else warm_mount.result_cell_code_revision
            ),
            native_head_crsa=native_head_crsa,
            text_snapshot_sink=(
                snapshot_bridge
                if snapshot_bridge is not None
                else live_writer.update
                if live_writer is not None
                else progress_writer.update
                if progress_writer is not None
                else None
            ),
        )
        component = qwen
        if not args.raw_qwen:
            component = QwenFertigChat(
                qwen,
                FertigSolver(),
                ooe_hook=None if warm_mount is None else warm_mount.hook,
            )
        if service:
            assert snapshot_bridge is not None
            application = QwenChatServiceApplication(
                qwen=qwen,
                component=component,
                snapshot_bridge=snapshot_bridge,
                request_metadata_for=request_metadata_for,
                fit_history=fit_interactive_history,
                attach_inference_economics=attach_inference_economics,
                result_summary=interactive_summary,
                runtime_profile_sha256=service_profile_sha256,
            )
            server = UnixQwenServiceServer(service_socket, application)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
            finally:
                server.close()
            return 0
        if jsonl:
            failures = 0
            handled = 0
            for raw in sys.stdin:
                line = raw.strip()
                if not line:
                    continue
                if max_requests is not None and handled >= max_requests:
                    break
                handled += 1
                request_id = None
                include_id = False
                try:
                    if line.startswith("{"):
                        document = json.loads(line)
                        if not isinstance(document, dict) or set(document) - {
                            "id",
                            "message",
                        }:
                            raise ValueError(
                                "JSONL request must contain only id/message"
                            )
                        include_id = "id" in document
                        request_id = document.get("id")
                        line_message = document.get("message")
                    else:
                        line_message = line
                    if not isinstance(line_message, str) or not line_message.strip():
                        raise ValueError("JSONL request message must be non-empty text")
                    result = component.handle(
                        Request(
                            "chat",
                            line_message,
                            request_metadata_for(line_message),
                        )
                    )
                    result = attach_inference_economics(
                        line_message,
                        result,
                        request_id=request_id if include_id else None,
                    )
                    emit(
                        result,
                        request_id=request_id,
                        include_id=include_id,
                    )
                    failures += int(not result.ok)
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    failures += 1
                    emit_line_error(
                        f"{type(exc).__name__}: {exc}",
                        request_id=request_id,
                        include_id=include_id,
                    )
                if max_requests is not None and handled >= max_requests:
                    break
            return 0 if failures == 0 else 2
        if interactive:
            failures = 0
            handled = 0
            last_summary = None
            history: tuple[tuple[str, str], ...] = ()
            session_generation = 0
            terminal = sys.stdin.isatty() and sys.stdout.isatty()
            if terminal:
                print(
                    "IMMER local Qwen — /help, /stats, /clear, /quit",
                    flush=True,
                )
            while max_requests is None or handled < max_requests:
                if terminal:
                    print("you> ", end="", flush=True)
                raw = sys.stdin.readline()
                if raw == "":
                    break
                line_message = raw.strip()
                if not line_message:
                    continue
                if line_message in {"/exit", "/quit"}:
                    break
                if line_message == "/help":
                    print(
                        "Enter any prompt. /stats shows the last real runtime cost. "
                        "/clear drops conversation context. /quit closes the loaded "
                        "local runtime.",
                        flush=True,
                    )
                    continue
                if line_message == "/stats":
                    print(last_summary or "No completed Qwen response yet.", flush=True)
                    continue
                if line_message == "/clear":
                    history = ()
                    last_summary = None
                    session_generation += 1
                    qwen.clear_conversation()
                    if terminal:
                        print("Conversation context cleared.", flush=True)
                    continue
                handled += 1
                retained, dropped_turns, prompt_tokens = fit_interactive_history(
                    line_message,
                    history,
                )
                if terminal:
                    if dropped_turns:
                        print(
                            f"[context: dropped {dropped_turns} oldest turns; "
                            f"{prompt_tokens} prompt tokens]",
                            file=sys.stderr,
                            flush=True,
                        )
                    print("immer> ", end="", flush=True)
                turn_component = qwen if history else component
                result = turn_component.handle(
                    Request(
                        "chat",
                        line_message,
                        request_metadata_for(
                            line_message,
                            retained,
                            f"interactive:{session_generation}",
                        ),
                    )
                )
                result = attach_inference_economics(line_message, result)
                emit(result)
                failures += int(not result.ok)
                if result.ok and isinstance(result.output, str):
                    history = (
                        *retained,
                        ("user", line_message),
                        ("assistant", result.output),
                    )
                last_summary = interactive_summary(result)
                if terminal and last_summary is not None:
                    print(last_summary, file=sys.stderr, flush=True)
                if live_writer is not None:
                    live_writer.reset()
            return 0 if failures == 0 else 2
        result = component.handle(
            Request("chat", message, request_metadata_for(message))
        )
        result = attach_inference_economics(message, result)
    except (
        DraftWindowError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        if live_writer is not None:
            live_writer.fail()
        if progress_writer is not None:
            progress_writer.finish()
        if output_mode == "text":
            print(
                f"error: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                json.dumps(
                    {
                        "status": "error",
                        "component": "qwen3.8.fertig-chat",
                        "reason": f"{type(exc).__name__}: {exc}",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        return 2
    finally:
        close = getattr(component, "close", None)
        if callable(close):
            close()
    emit(result)
    return 0 if result.ok else 2


def _doctor(
    *,
    deep: bool = False,
    artifact_root: str | Path | None = None,
    qwen38_root: str | Path | None = None,
    qwen38_causal_bundle: str | Path | None = None,
    qwen38_tokenizer: str | Path | None = None,
    qwen38_q4: str | Path | None = None,
) -> int:
    from .runtimes.o1_state.adapter import is_available as o1state_available

    fertig_root = os.environ.get("IMMER_FERTIG_ROOT")
    vendor_solver = (
        Path(__file__).parent
        / "cognition"
        / "fertig"
        / "_vendor"
        / "fertig"
        / "solver.py"
    )
    configured_solver = (
        Path(fertig_root).expanduser() / "fertig" / "solver.py" if fertig_root else None
    )
    solver_ready = vendor_solver.is_file() or bool(
        configured_solver is not None and configured_solver.is_file()
    )
    graph_setting = os.environ.get("IMMER_FERTIG_GRAPH")
    graph_ready = bool(graph_setting and Path(graph_setting).expanduser().is_file())
    manifest = _s3_manifest()
    selected_artifact_root = _artifact_root(manifest, artifact_root)
    organ_ready = False
    organ_detail = str(manifest)
    try:
        from .capabilities.organbank import OrganBank

        bank = OrganBank.from_manifest(manifest, artifact_root=selected_artifact_root)
        bank.verify_all()
        organ_ready = len(bank.names()) == 4
        organ_detail = f"{len(bank.names())} SHA-geprüfte Organe"
    except (FileNotFoundError, KeyError, ValueError) as exc:
        organ_detail = f"{exc}; run 'immer artifacts import SOURCE'"
    qwen_requested = bool(
        qwen38_root
        or qwen38_causal_bundle
        or qwen38_tokenizer
        or qwen38_q4
        or os.environ.get("IMMER_QWEN38_ROOT")
        or os.environ.get("IMMER_QWEN38_CAUSAL_BUNDLE")
        or os.environ.get("IMMER_QWEN38_TOKENIZER")
        or os.environ.get("IMMER_QWEN38_Q4")
        or _QWEN38_DEPLOYMENT_ROOT.is_dir()
    )
    qwen_ready = False
    qwen_detail = "optional; set --qwen38-root or IMMER_QWEN38_ROOT"
    qwen_paths = argparse.Namespace(
        draft_mode=None,
        fast_mlp=None,
        no_fast_mlp=False,
        qwen38_causal_bundle=qwen38_causal_bundle,
        qwen38_q4=qwen38_q4,
        qwen38_root=qwen38_root,
        qwen38_tokenizer=qwen38_tokenizer,
    )
    bundle_path = None
    if qwen_requested:
        try:
            from .runtimes.qwen3_8.local_install import inspect_local_qwen

            bundle_path, tokenizer_path, q4_root, _fast_mlp = (
                _resolve_qwen38_chat_paths(qwen_paths)
            )
            if q4_root is None:
                raise ValueError("configured Qwen runtime has no Q4/Q8 bank")
            install = inspect_local_qwen(
                bundle_path,
                tokenizer_path=tokenizer_path,
                q4_root=q4_root,
            )
            qwen_ready = True
            qwen_detail = install.summary()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            qwen_detail = f"{type(exc).__name__}: {exc}"
    checks = [
        ("FERTIG-solv", solver_ready, "vendored; override via IMMER_FERTIG_ROOT", True),
        (
            "FERTIG-graph",
            graph_ready,
            str(Path(graph_setting).expanduser())
            if graph_setting
            else "optional; set IMMER_FERTIG_GRAPH",
            False,
        ),
        ("Qwen3.8", qwen_ready, qwen_detail, qwen_requested),
        (
            "Action-gates",
            True,
            "desktop/recorder/mutations require explicit backends",
            True,
        ),
        ("o1-state", o1state_available(), "pip install -e '.[neural]'", True),
        ("SHIP-v6", organ_ready, organ_detail, False),
    ]
    try:
        import torch  # noqa: F401

        crsa = True
    except ImportError:
        crsa = False
    checks.append(("CRSA/Torch", crsa, "pip install -e '.[neural]'", True))
    try:
        import numpy  # noqa: F401
        from .knowledge import Streamer  # noqa: F401

        world_stream = True
    except ImportError:
        world_stream = False
    checks.append(("WorldStream", world_stream, "pip install -e .", True))
    service_ready = False
    service_detail = "optional; start 'immer chat --service'"
    if qwen_ready and bundle_path is not None:
        try:
            from .runtimes.qwen3_8.service import UnixQwenServiceClient

            socket_path = _resolve_qwen38_service_socket(qwen_paths, bundle_path)
            client = UnixQwenServiceClient(socket_path, timeout=0.2)
            try:
                ping = client.request("ping")
            finally:
                client.close()
            profile = ping.result.evidence.get("runtime_profile_sha256")
            if not isinstance(profile, str) or len(profile) != 64:
                raise RuntimeError("service returned no runtime profile")
            service_ready = True
            service_detail = f"resident profile {profile[:12]}"
        except (OSError, RuntimeError, TypeError, ValueError):
            pass
    checks.append(("Qwen-service", service_ready, service_detail, False))
    for name, ready, hint, _required in checks:
        print(f"{'✓' if ready else '·'} {name:12} {hint}")
    if deep and organ_ready and crsa:
        from .capabilities.s3_runtime import S3Arithmetic

        benchmark = S3Arithmetic(
            manifest,
            artifact_root=selected_artifact_root,
        ).benchmark()
        passed = (
            benchmark["correct"] == benchmark["cases"]
            and benchmark["route_correct"] == benchmark["cases"]
        )
        print(
            f"{'✓' if passed else '·'} SHIP-eval    "
            f"{benchmark['correct']}/{benchmark['cases']} korrekt; "
            f"Route {benchmark['route_correct']}/{benchmark['cases']}; "
            f"{benchmark['runtime_s']:.3f}s"
        )
        return 0 if passed else 1
    return 0 if all(ready for _, ready, _, required in checks if required) else 1


def _organs(args: argparse.Namespace) -> int:
    from .capabilities.organbank import OrganBank

    manifest = _s3_manifest(args.manifest)
    bank = OrganBank.from_manifest(
        manifest,
        artifact_root=_artifact_root(manifest, args.artifact_root),
    )
    if args.organ_command == "list":
        for name in bank.names():
            descriptor = bank.descriptor(name)
            print(f"{name:24} {descriptor.capability:16} {descriptor.group}")
        return 0
    artifact = bank.verify(args.name)
    print(
        json.dumps(
            {"organ": args.name, "verified": True, "artifact": str(artifact)},
            ensure_ascii=False,
        )
    )
    return 0


def _artifacts(args: argparse.Namespace) -> int:
    from .artifacts import (
        ArtifactBootstrapError,
        import_artifacts,
        load_artifact_specs,
        sha256_file,
    )

    manifest = _s3_manifest(args.manifest)
    if args.artifact_command == "verify":
        try:
            _, specs = load_artifact_specs(
                manifest,
                configured_root=args.artifact_root,
            )
            rows = []
            for spec in specs:
                actual = (
                    sha256_file(spec.destination)
                    if spec.destination.is_file()
                    else None
                )
                rows.append(
                    {
                        "label": spec.label,
                        "path": str(spec.destination),
                        "present": actual is not None,
                        "verified": actual == spec.sha256,
                        "sha256": actual,
                    }
                )
            passed = all(row["verified"] for row in rows)
            print(
                json.dumps(
                    {"status": "ok" if passed else "missing", "artifacts": rows},
                    sort_keys=True,
                )
            )
            return 0 if passed else 1
        except ArtifactBootstrapError as exc:
            print(
                json.dumps({"status": "error", "reason": str(exc)}, sort_keys=True),
                file=sys.stderr,
            )
            return 2
    try:
        report = import_artifacts(
            manifest,
            args.source,
            dry_run=args.dry_run,
            replace=args.replace,
            configured_root=args.artifact_root,
        )
    except ArtifactBootstrapError as exc:
        print(
            json.dumps({"status": "error", "reason": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def _eval_ship(
    manifest: str | Path | None = None,
    artifact_root: str | Path | None = None,
) -> int:
    from .capabilities.s3_runtime import S3Arithmetic

    report = dict(
        S3Arithmetic(
            _s3_manifest(manifest),
            artifact_root=artifact_root,
        ).benchmark()
    )
    report["passed"] = (
        report["correct"] == report["cases"]
        and report["route_correct"] == report["cases"]
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


def _export_hf(args: argparse.Namespace) -> int:
    """Build the final offline bundle; never authenticate or upload."""

    from .hf_export import HfExportError, export_hf_poc

    try:
        report = export_hf_poc(
            args.output,
            manifest=_s3_manifest(args.manifest),
            artifact_root=args.artifact_root,
            replace=args.replace,
        )
    except HfExportError as exc:
        print(
            json.dumps({"status": "error", "reason": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


def _stream(args: argparse.Namespace) -> int:
    """Inspect or read exact rows without constructing the donor model."""

    from .knowledge import Streamer, TensorSourceError

    try:
        source = (
            Streamer.from_local(
                args.source,
                revision=args.revision,
                budget_mb=args.budget_mb,
                cache_dir=args.cache_dir,
                use_cache=not args.no_cache,
            )
            if args.local
            else Streamer(
                args.source,
                revision=args.revision,
                budget_mb=args.budget_mb,
                cache_dir=args.cache_dir,
                use_cache=not args.no_cache,
            )
        )
        inventory = source.inventory(refresh=args.refresh)
        tensors = source.tensors()
        if args.tensor is None:
            payload = {
                "status": "ok",
                "mode": "inventory",
                "source": source.repo_id,
                "revision": source.revision,
                "tensor_count": len(tensors),
                "shard_count": len(inventory.get("shards", ())),
                "tensors": [entry["name"] for entry in tensors[: args.limit]],
                "truncated": len(tensors) > args.limit,
                "metrics": source.metrics(),
            }
        else:
            rows = source.rows(args.tensor, start_row=args.start_row, n_rows=args.rows)
            raw = rows.tobytes(order="C")
            preview_rows = min(2, int(rows.shape[0]))
            preview_cols = min(8, int(rows.shape[1]))
            payload = {
                "status": "ok",
                "mode": "rows",
                "source": source.repo_id,
                "revision": source.revision,
                "tensor": args.tensor,
                "start_row": args.start_row,
                "shape": list(rows.shape),
                "dtype": str(rows.dtype),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "preview": rows[:preview_rows, :preview_cols].tolist(),
                "metrics": source.metrics(),
            }
    except (FileNotFoundError, KeyError, TensorSourceError, ValueError) as exc:
        print(
            json.dumps(
                {"status": "error", "error": type(exc).__name__, "reason": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _serve(args: argparse.Namespace) -> int:
    """The organism lives in this terminal: it learns, remembers and answers."""
    from .intent import classify
    from .library import Library
    from .memory import SpanStore
    from .runtimes.donor.adapter import DonorBrain
    from .runtimes.o1_state.adapter import is_available
    from .runtimes.o1_state.plasticity import LearningStream
    from .runtimes.qwen.adapter import QwenBrain
    from .suite import Metrics, start_dashboard

    state = Path(args.state).expanduser()
    state.parent.mkdir(parents=True, exist_ok=True)

    stream = None
    if is_available():
        try:
            stream = LearningStream(sidecar=state.with_suffix(".pt"))
        except RuntimeError as exc:
            print(f"· lernender Strom: {exc}", file=sys.stderr)
    else:
        print("· o1-state bridge nicht verfügbar — Leben ohne Lernen", file=sys.stderr)

    qwen = QwenBrain() if args.local_brain else None
    donor = DonorBrain()
    council = None
    if args.council and donor.available():
        from .runtimes.donor.adapter import build_council

        council = build_council()
    elif args.council and qwen is not None and qwen.model_id is not None:
        from .council import Council

        council = Council(
            (
                QwenBrain(name="qwen.basis"),
                QwenBrain(
                    name="qwen.kritiker",
                    persona="Du bist der Kritiker im Rat eines Lebewesens. Prüfe Aussagen "
                    "auf Fehler und Widersprüche und korrigiere sie. Antworte kurz.",
                ),
                QwenBrain(
                    name="qwen.freigeist",
                    persona="Du bist der Freigeist im Rat eines Lebewesens. Denk "
                    "unkonventionell und bringe den Blickwinkel, den niemand sonst hat. "
                    "Antworte kurz.",
                ),
            ),
            name="rat",
        )
    harvester = donor if donor.available() else qwen
    library = Library(SpanStore(state.parent / "memory.json"), harvester=harvester)
    metrics = Metrics(
        status_path=state.parent / "status.json",
        jsonl_path=state.parent / "metrics.jsonl",
    )

    bank = None
    manifest = _s3_manifest(args.manifest)
    if manifest.is_file():
        from .capabilities.organbank import OrganBank

        bank = OrganBank.from_manifest(
            manifest,
            artifact_root=_artifact_root(manifest, args.artifact_root),
        )
    from .composition import CompositionRoot

    composition = CompositionRoot.build(
        life_stream=stream,
        s3_manifest=manifest,
        s3_artifact_root=args.artifact_root,
        fertig_state_dir=state.parent / "fertig",
        fertig_graph=os.environ.get("IMMER_FERTIG_GRAPH"),
    )
    daemon = LifeDaemon(stream=stream, bank=bank, state_path=args.state)
    daemon.register(composition.exact_math)
    if composition.grounded_chat is not None:
        daemon.register(composition.grounded_chat)

    history: list[dict[str, str]] = []

    def gauges() -> dict:
        life = stream.metrics() if stream is not None else {}
        return {
            "turns": daemon.turns,
            "tokens": life.get("tokens", 0),
            "loss_ema": life.get("loss_ema"),
            "surprises": life.get("surprises", 0),
            "updates": life.get("updates", 0),
            "sleeps": life.get("sleeps", 0),
            "span_buffer": life.get("span_buffer", 0),
            "donor": donor.available(),
            "donor_modell": donor.model_id,
            "lokal_gehirn": qwen.loaded if qwen is not None else False,
            "spans": library.store.count(),
        }

    port = None if args.no_dashboard else (args.dashboard or 8787)
    server = None
    if port:
        try:
            server = start_dashboard(metrics, gauges, port)
            print(f"◆ dashboard: http://127.0.0.1:{port}/  (Maschinenfutter: /status)")
        except OSError as exc:
            print(f"· dashboard aus: {exc}", file=sys.stderr)

    available_organs = bank.names() if bank is not None else ()
    print(
        f"immer serve — turns alive: {daemon.turns}; cold bank: "
        f"{', '.join(available_organs) or 'none'}"
    )
    print("commands: /state /say <text> /sleep /quit — alles andere ist Erfahrung")
    for line in sys.stdin:
        text = line.strip()
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/state":
            print(
                json.dumps(metrics.emit(gauges()), ensure_ascii=False, sort_keys=True)
            )
            continue
        if text.startswith("/say "):
            daemon.say("utterance", text[5:])
            print("(gesagt)")
            continue
        if text == "/sleep":
            if stream is not None:
                print(f"(konsolidiert: {stream.sleep()})")
            else:
                print("(kein Strom zum Konsolidieren)")
            continue

        intent = classify(text)
        daemon.submit_user(text)  # das Leben trinkt zuerst alles

        if intent.kind == "TEACH":
            span = library.teach(intent.payload)
            metrics.bump("teaches")
            print(f"(gemerkt: {span['key']})")
        elif intent.kind == "RECALL":
            hits = library.recall(intent.payload)
            metrics.bump("recalls")
            if not hits and harvester is not None:
                metrics.bump("harvests")
                card = library.grow(intent.payload)
                if card is not None:
                    print(f"[ernte] {card['text']}")
                    print(f"         (quelle: {card['source']})")
                    hits = [card]
            if hits:
                for span in hits:
                    print(f"[erinnerung] {span['text']} ({span.get('source', '?')})")
            else:
                print("[erinnerung] da weiß ich noch nichts — lehr mich (merke: …)")
        elif intent.kind == "MATH":
            import time as _time

            exact_started = _time.perf_counter()
            answer = daemon.request("exact_math", intent.payload)
            exact_ms = int((_time.perf_counter() - exact_started) * 1000)
            if answer.ok:
                metrics.bump("math_ok")
                metrics.emit_gauge("last_exact_latency_ms", exact_ms)
                route = answer.evidence.get("route", answer.component)
                print(f"[exact] {answer.output}  ({exact_ms} ms, {route})")
            else:
                metrics.bump("math_abstained")
                # Abstinenz heißt nicht Endstation: gleiche Kaskade wie Chat
                _answer_via_cascade(
                    intent.payload,
                    library,
                    council,
                    donor,
                    qwen,
                    daemon,
                    history,
                    metrics,
                    stream,
                )
        elif intent.kind == "STATUS":
            metrics.bump("status")
            print(json.dumps(metrics.snapshot(gauges()), ensure_ascii=False))
        else:  # CHAT — Stufen: Bibliothek (0 ms) → Entwurf → Donor → Veredelung im Hintergrund
            _answer_via_cascade(
                intent.payload,
                library,
                council,
                donor,
                qwen,
                daemon,
                history,
                metrics,
                stream,
            )
            metrics.bump("chat")

        metrics.emit(gauges())
        life_line = f"turns: {daemon.turns}"
        if stream is not None:
            life_line += f", tokens: {stream.tokens}, updates: {stream.updates}, überraschungen: {stream.surprises}"
        print(f"({life_line})")
    if server is not None:
        server.shutdown()
        server.server_close()
    return 0


def _answer_via_cascade(
    payload: str,
    library,
    council,
    donor,
    qwen,
    daemon,
    history: list[dict[str, str]],
    metrics,
    stream,
) -> None:
    """Stufen des Mundwerks, nach Latenz sortiert — gemessen, nicht behauptet.

    Tier 0  Bibliothek   ~0 ms      (die Karte existiert schon)
    Tier 1  FERTIG       grounded   (Graph, Skills, Hilfe; sichere Aktions-Gates)
    Tier 2  Donor        ~19 s      (SOTA-Orakel, sync wenn nichts Besseres da)
    Tier 3  Veredelung   im Hintergrund (Donor verbessert den Entwurf nachträglich)
    """
    import threading
    import time as _time

    meta = {
        "history": history[-6:],
        "life": f"turns={daemon.turns} tokens={getattr(stream, 'tokens', 0)}",
    }

    # ---- Tier 0: die Bibliothek ist schneller als jedes Modell -------------
    t0 = _time.perf_counter()
    hits = library.recall(payload)
    if hits:
        latency = int((_time.perf_counter() - t0) * 1000)
        metrics.bump("tier_bibliothek")
        metrics.emit_gauge("last_latency_ms", latency)
        print(
            f"[bibliothek] {hits[0]['text']}  ({latency} ms, quelle: {hits[0].get('source', '?')})"
        )
        history.append({"role": "user", "content": payload})
        history.append({"role": "assistant", "content": hits[0]["text"]})
        return

    # ---- Tier 1: FERTIGs geerdeter Graph-/Skill-Pfad ----------------------
    grounded = daemon.request("grounded_chat", payload, metadata=meta)
    if grounded.ok:
        route = str(grounded.evidence.get("route", "grounded"))
        if route == "math":
            # FERTIGs natürliche Chat-Oberfläche darf den Guard des exakten
            # Pfads nicht umgehen. Nur eine bestätigte Zahl wird gesprochen.
            verified = daemon.request("exact_math", payload, metadata=meta)
            claimed = grounded.evidence.get("data", {}).get("answer")
            if not verified.ok or (
                claimed is not None
                and str(claimed).strip().casefold()
                != str(verified.output).strip().casefold()
            ):
                grounded = None
        if grounded is not None:
            metrics.bump("tier_fertig")
            print(f"[fertig:{route}] {grounded.output}")
            library.capture(payload, grounded.output)
            history.append({"role": "user", "content": payload})
            history.append({"role": "assistant", "content": str(grounded.output)})
            return
    elif (
        grounded.evidence.get("status") in {"needs_input", "error"}
        or grounded.evidence.get("route") == "desktop"
    ):
        # Eine fehlende explizite Desktop-/Recorder-Freigabe ist eine harte
        # Grenze; ein Sprachmodell darf daraus keine scheinbare Aktion machen.
        print(f"[fertig] ({grounded.reason})")
        return

    # ---- Tier 2: der Donor antwortet synchron und die Karte wird sofort geschrieben
    if council is not None:
        result = council.deliberate("chat", payload, metadata=meta)
        tag = "[rat]"
    elif donor.available():
        t1 = _time.perf_counter()
        result = donor.handle(Request("chat", payload, metadata=meta))
        latency = int((_time.perf_counter() - t1) * 1000)
        metrics.emit_gauge("last_latency_ms", latency)
        tag = f"[donor {latency / 1000:.1f}s]"
    elif qwen is not None:
        t1 = _time.perf_counter()
        result = qwen.handle(Request("chat", payload, metadata=meta))
        latency = int((_time.perf_counter() - t1) * 1000)
        metrics.emit_gauge("last_latency_ms", latency)
        tag = f"[entwurf {latency / 1000:.1f}s]"
        # Tier 3: der Donor veredelt den Entwurf, ohne zu blockieren
        if donor.available() and result.ok:

            def _refine() -> None:
                card = library.grow(payload)
                if card is not None:
                    metrics.bump("refined")

            threading.Thread(target=_refine, daemon=True).start()
            print("         (donor veredelt im hintergrund…)")
    else:
        result = None

    if result is not None and result.ok:
        print(f"{tag} {result.output}")
        if council is not None and result.evidence.get("votes"):
            print(
                f"         (stimmen: {result.evidence['votes']}, "
                f"übereinstimmend: {result.evidence['agree']})"
            )
        # Jede gesprochene Antwort wird Karte — die Wiederholung ist gratis.
        library.capture(payload, result.output)
        history.append({"role": "user", "content": payload})
        history.append({"role": "assistant", "content": result.output})
    else:
        reason = (
            result.reason
            if result is not None
            else "kein Mund konfiguriert (--local-brain?)"
        )
        print(f"[stille] ({reason})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="immer")
    from . import __version__

    parser.add_argument("--version", action="version", version=f"immer {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("components", help="show the system component map")
    doctor = sub.add_parser("doctor", help="inspect runtime integrations")
    doctor.add_argument(
        "--deep", action="store_true", help="also run the cold 152-case SHIP eval"
    )
    doctor.add_argument("--artifact-root", help="external SHIP artifact directory")
    doctor.add_argument(
        "--qwen38-root",
        help="local Qwen3.8 root containing bundle, tokenizer, causal graph and Q4/Q8",
    )
    doctor.add_argument(
        "--qwen38-causal-bundle", help="override the causal bundle root"
    )
    doctor.add_argument("--qwen38-tokenizer", help="override tokenizer.json")
    doctor.add_argument("--qwen38-q4", help="override the packed Q4/Q8 bank")
    solve = sub.add_parser("solve", help="run the guarded S3 + FERTIG exact cascade")
    solve.add_argument("question")
    solve.add_argument(
        "--manifest", help="SHIP-v6 manifest (default: bundled manifest)"
    )
    solve.add_argument("--artifact-root", help="external SHIP artifact directory")
    chat = sub.add_parser(
        "chat",
        help="run one greedy turn through a verified local Qwen3.8 causal bundle",
    )
    chat.add_argument("message", nargs="?")
    chat.add_argument(
        "--jsonl",
        action="store_true",
        help="keep one loaded runtime and process stdin as raw-text or JSONL requests",
    )
    chat.add_argument(
        "--interactive",
        action="store_true",
        help="keep one loaded runtime and accept arbitrary prompts until /quit",
    )
    chat.add_argument(
        "--service",
        action="store_true",
        help="keep the canonical Qwen/FERTIG runtime behind a local Unix socket",
    )
    chat.add_argument(
        "--socket",
        "--service-socket",
        dest="service_socket",
        help=(
            "local Qwen service socket; default: IMMER_QWEN38_SOCKET, the "
            "deployment state directory, or ~/.immer/qwen3.8-service.sock"
        ),
    )
    chat.add_argument(
        "--direct",
        action="store_true",
        help="bypass an available local Qwen service and mount the runtime here",
    )
    chat.add_argument(
        "--output",
        choices=("text", "json"),
        help="single-request output (default: live text; JSONL always uses json)",
    )
    chat.add_argument(
        "--no-stream",
        action="store_true",
        help="buffer a text response and print it only after generation completes",
    )
    chat.add_argument(
        "--max-requests",
        type=int,
        help="stop the persistent JSONL loop after this many non-empty lines",
    )
    chat.add_argument(
        "--qwen38-root",
        help=(
            "local Qwen root containing config, weights, tokenizer and causal/; "
            "default: IMMER_QWEN38_ROOT or the deployed /app model"
        ),
    )
    chat.add_argument(
        "--qwen38-causal-bundle",
        help="override the local causal bundle (default: Qwen root)",
    )
    chat.add_argument(
        "--qwen38-tokenizer",
        help="override tokenizer.json (default: Qwen root/tokenizer.json)",
    )
    chat.add_argument(
        "--qwen38-q4",
        metavar="BANK",
        help="causal-bound mmap Q4/Q8 execution bank for the local Qwen bundle",
    )
    chat.add_argument(
        "--q4-threads",
        type=int,
        help="CPU worker count for the native Q4/Q8 kernel (default: up to 16)",
    )
    chat.add_argument(
        "--q4-resident-budget-mb",
        type=int,
        default=None,
        metavar="MIB",
        help=(
            "protect recurring packed Q4/Q8 pages across layer/token boundaries "
            "under a sticky page budget; canonical deployment: 16384, "
            "other layouts: 0, explicit 0 keeps eager MADV_DONTNEED"
        ),
    )
    chat.add_argument(
        "--draft-bundle",
        help="optional local causal Qwen3.5-0.8B bundle for rolling drafting",
    )
    chat.add_argument(
        "--no-external-drafter",
        action="store_true",
        help="disable the deployed Qwen3.5-0.8B hybrid novelty expert",
    )
    chat.add_argument(
        "--draft-mode",
        choices=("qwen35", "markov", "mtp", "hybrid"),
        help=(
            "rolling draft provider; the deployed MTP-capable bank defaults to the "
            "Markov/MTP hybrid, while an explicit value overrides it"
        ),
    )
    chat.add_argument(
        "--draft-window",
        type=int,
        choices=range(2, 17),
        default=16,
        metavar="K",
        help="target-verified tokens per weight pass (2-16; default: 16)",
    )
    chat.add_argument(
        "--markov-draft-state",
        help="persistent sparse Qwen-token Markov memory",
    )
    chat.add_argument(
        "--markov-atlas",
        help="corpus-scale zero-model-byte Qwen-token Markov atlas",
    )
    chat.add_argument(
        "--markov-o1-retention",
        help="persistent O1 surprise scorer for confirmed Markov episodes",
    )
    chat.add_argument(
        "--context-crystal-state",
        help=(
            "persistent target-hidden continuation Crystals for verified "
            "K1-K16 drafting"
        ),
    )
    chat.add_argument(
        "--no-context-crystal",
        action="store_true",
        help="disable the deployed target-hidden continuation Crystal bank",
    )
    chat.add_argument(
        "--layer-contextual-continuation-state",
        "--layer-context-crystal-state",
        dest="layer_contextual_continuation_state",
        help=(
            "persistent multi-layer Q8 continuation bank for target-verified "
            "Markov drafting"
        ),
    )
    chat.add_argument(
        "--no-layer-contextual-continuation",
        action="store_true",
        help="disable the deployed multi-layer continuation bank",
    )
    chat.add_argument(
        "--layer-contextual-continuation-max-cells",
        "--layer-context-crystal-max-cells",
        dest="layer_contextual_continuation_max_cells",
        type=int,
        default=4096,
        metavar="N",
        help="maximum persistent multi-layer continuation cells (default: 4096)",
    )
    chat.add_argument(
        "--layer-contextual-continuation-max-receipts",
        "--layer-context-crystal-max-receipts",
        dest="layer_contextual_continuation_max_receipts",
        type=int,
        default=65_536,
        metavar="N",
        help=(
            "maximum retry-safe multi-layer settlement receipts "
            "(default: 65536)"
        ),
    )
    chat.add_argument(
        "--attention-output-crystal-state",
        help=(
            "persistent exact full-attention outputs that skip repeated "
            "Q/K/V/O projections"
        ),
    )
    chat.add_argument(
        "--no-attention-output-crystal",
        action="store_true",
        help="disable the deployed exact attention-output Crystal bank",
    )
    chat.add_argument(
        "--layer-transition-crystal-state",
        help=(
            "existing sealed private layer-63 K1 transition Crystal bank; "
            "no bank is mounted by default"
        ),
    )
    chat.add_argument(
        "--layer-transition-crystal-atlas",
        help=(
            "existing O1 Semantic Atlas authority whose authenticated history "
            "contains the private layer-63 bank revision"
        ),
    )
    chat.add_argument(
        "--layer-transition-crystal-compute-root",
        help=(
            "existing Compute Crystal bank whose authenticated operator-graph "
            "history contains the private layer-63 bank revision"
        ),
    )
    chat.add_argument(
        "--layer-transition-crystal-max-error-radius",
        type=_finite_non_negative_float,
        default=0.0,
        metavar="RADIUS",
        help=(
            "maximum admitted artifact error radius for the private BF16 "
            "layer-63 bank (default: 0)"
        ),
    )
    chat.add_argument(
        "--layer-mlp-crystal-state",
        help=(
            "existing sealed private layer-63 MLP residual Crystal bank; "
            "no bank is mounted by default"
        ),
    )
    chat.add_argument(
        "--layer-mlp-crystal-registry",
        help=(
            "existing sealed offline multi-layer MLP Crystal registry; "
            "mutually exclusive with the legacy layer-63 state"
        ),
    )
    chat.add_argument(
        "--layer-mlp-crystal-atlas",
        help=(
            "existing O1 Semantic Atlas authority whose authenticated history "
            "contains the private layer-63 MLP bank revision"
        ),
    )
    chat.add_argument(
        "--layer-mlp-crystal-compute-root",
        help=(
            "existing Compute Crystal bank whose authenticated operator-graph "
            "history contains the private layer-63 MLP bank revision"
        ),
    )
    chat.add_argument(
        "--layer-mlp-crystal-max-error-radius",
        type=_finite_non_negative_float,
        default=0.0,
        metavar="RADIUS",
        help=(
            "maximum admitted artifact error radius for the private BF16 "
            "layer-63 MLP bank (default: 0)"
        ),
    )
    chat.add_argument(
        "--layer-mlp-o1-state",
        help=(
            "persistent passive layer-63 MLP O1 sufficient statistics; "
            "never mounts the learned approximate bank"
        ),
    )
    chat.add_argument(
        "--layer-mlp-o1-layers",
        type=_sorted_layer_list,
        help=(
            "sorted decoder-layer registry to charge over ordinary requests; "
            "explicit 63 keeps the legacy single-layer runtime"
        ),
    )
    chat.add_argument(
        "--layer-mlp-o1-atlas",
        help="authenticated O1 Semantic Atlas authority for passive MLP charging",
    )
    chat.add_argument(
        "--layer-mlp-o1-compute-root",
        help="authenticated ComputeOperatorGraph authority for passive MLP charging",
    )
    chat.add_argument(
        "--layer-mlp-o1-sketch-dim",
        type=int,
        default=128,
        metavar="D",
        help="O1 Rademacher sketch width (default: 128)",
    )
    chat.add_argument(
        "--layer-mlp-o1-seed",
        default=("2b188a99b4b36f51bd910866e6d9a007fd02ec7256d6596fc87eb76f0444eccf"),
        metavar="SHA256",
        help="deterministic O1 projection seed",
    )
    chat.add_argument(
        "--layer-mlp-o1-ridge",
        type=float,
        default=1e-8,
        metavar="LAMBDA",
        help="positive centered-ridge coefficient (default: 1e-8)",
    )
    chat.add_argument(
        "--layer-mlp-o1-coverage-guard",
        type=float,
        default=0.0,
        metavar="RADIUS",
        help="non-negative persistent feature-coverage guard",
    )
    chat.add_argument(
        "--layer-mlp-o1-error-guard",
        type=float,
        default=0.0,
        metavar="RADIUS",
        help="non-negative persistent error-envelope guard",
    )
    chat.add_argument(
        "--layer-mlp-o1-queue-capacity",
        type=int,
        default=8,
        metavar="N",
        help="bounded asynchronous request-batch queue (default: 8)",
    )
    chat.add_argument(
        "--mtp-draft-state",
        help="persistent embedded-MTP reliability state",
    )
    chat.add_argument(
        "--no-markov-draft",
        action="store_true",
        help="disable the deployed target-verified Markov token council",
    )
    chat.add_argument(
        "--draft-window-state",
        help=(
            "persistent target-receipt controller for contextual K=4/8/16; "
            "--draft-window becomes its maximum ceiling"
        ),
    )
    chat.add_argument(
        "--no-draft-window-controller",
        action="store_true",
        help="disable the deployed joint runtime-reward window controller",
    )
    chat.add_argument(
        "--range-markov-state",
        default=os.environ.get("IMMER_QWEN38_RANGE_MARKOV_STATE"),
        help="persistent operation-Markov state for local weight-range prefetch",
    )
    chat.add_argument(
        "--range-prefetch-max-mb",
        type=float,
        default=64.0,
        help="maximum OS-cache hint bytes after one predicted operation",
    )
    chat.add_argument(
        "--range-prefetch-min-support",
        type=int,
        default=2,
        help="minimum learned transition support before a range hint",
    )
    chat.add_argument(
        "--range-prefetch-min-confidence",
        type=float,
        default=0.65,
        help="minimum next-operation probability before a range hint",
    )
    chat.add_argument(
        "--range-prefetch-beam-horizon",
        type=int,
        default=3,
        help="future operation depth for the bounded range beam",
    )
    chat.add_argument(
        "--range-prefetch-beam-width",
        type=int,
        default=4,
        help="maximum hypothetical paths retained at each range depth",
    )
    chat.add_argument(
        "--range-prefetch-hint-cooldown",
        type=int,
        default=2,
        help="operations before an identical exact range can be hinted again",
    )
    chat.add_argument(
        "--fast-mlp",
        metavar="ARTIFACT_ROOT",
        help="mount the existing row-routed sparse MLP banks for fast chat",
    )
    chat.add_argument(
        "--no-fast-mlp",
        action="store_true",
        help="run the full Q4 MLP path instead of the deployed sparse plan",
    )
    chat.add_argument(
        "--fast-mlp-layers",
        type=_sorted_layer_list,
        help="sorted fitted layer subset, for example 0,9,18,27,36,45,54,63",
    )
    chat.add_argument(
        "--fast-mlp-blocks",
        type=int,
        help="override the packed Q4 route width per active MLP layer",
    )
    chat.add_argument(
        "--fast-mlp-policy",
        choices=("auto", "manual", "structure-edge"),
        default="auto",
        help=(
            "auto uses structure-edge for the packed Q4 plan; structure-edge "
            "keeps layers 18-54 full and routes only the edges"
        ),
    )
    chat.add_argument(
        "--fast-mlp-online-state",
        help="persistent target-confirmed state for a weight-only Fast-MLP bank",
    )
    chat.add_argument(
        "--mlp-page-state",
        help=("persistent Markov page routes that execute selected Q4 MLP pages"),
    )
    chat.add_argument(
        "--mlp-page-width",
        type=int,
        default=192,
        help="64-neuron Q4 MLP pages executed per layer (default: 192 of 272)",
    )
    chat.add_argument(
        "--no-mlp-page-route",
        action="store_true",
        help="disable the deployed direct Markov MLP page route",
    )
    chat.add_argument(
        "--mlp-page-coordinate-state",
        help=("persistent exact K1 MLP page actions keyed by target BF16 inputs"),
    )
    chat.add_argument(
        "--no-mlp-page-coordinate",
        action="store_true",
        help="disable the deployed exact K1 MLP page coordinate bank",
    )
    chat.add_argument(
        "--delta-head-online-state",
        help="persistent Markov-Sinkhorn state for packed DeltaNet head routing",
    )
    chat.add_argument(
        "--delta-head-layers",
        type=_sorted_layer_list,
        help=(
            "linear-attention layers using packed DeltaNet head coordinates; "
            "defaults to every DeltaNet layer"
        ),
    )
    prefix_sinkhorn = chat.add_mutually_exclusive_group()
    prefix_sinkhorn.add_argument(
        "--prefix-sinkhorn",
        dest="prefix_sinkhorn",
        action="store_true",
        help=(
            "replace base softmax with native Prefix-Sinkhorn on the routed "
            "layer-27 heads; this is the canonical local-Q4 deployment default"
        ),
    )
    prefix_sinkhorn.add_argument(
        "--no-prefix-sinkhorn",
        dest="prefix_sinkhorn",
        action="store_false",
        help="use checkpoint base softmax instead of deployed Prefix-Sinkhorn",
    )
    chat.set_defaults(prefix_sinkhorn=None)
    chat.add_argument(
        "--raw-qwen",
        action="store_true",
        help="bypass FERTIG exact short-circuiting and call raw Qwen directly",
    )
    chat.add_argument(
        "--ooe-warm-root",
        help="verified local Markov/OoE ResultCell bank",
    )
    chat.add_argument(
        "--no-ooe-warm",
        action="store_true",
        help="disable the deployed zero-Qwen-forward warm path",
    )
    chat.add_argument(
        "--inference-economics-state",
        help="append-only local inference-economics ledger directory",
    )
    chat.add_argument(
        "--no-inference-economics",
        action="store_true",
        help="disable passive inference-economics receipts",
    )
    chat.add_argument(
        "--qwen38-anchor-cache",
        default=os.environ.get("IMMER_QWEN38_ANCHOR_CACHE"),
        help="local authenticated semantic anchor cache",
    )
    chat.add_argument(
        "--no-anchor-cache",
        action="store_true",
        help="disable the deployed shared chat-template state battery",
    )
    chat.add_argument("--system-prompt", default="")
    chat.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    chat.add_argument(
        "--compute-dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
    )
    chat.add_argument("--source-budget-mb", type=float, default=4194304)
    chat.add_argument("--draft-source-budget-mb", type=float, default=1048576)
    chat.add_argument("--fast-mlp-source-budget-mb", type=float)
    chat.add_argument("--max-resident-mb", type=int, default=192)
    chat.add_argument("--draft-max-resident-mb", type=int, default=64)
    chat.add_argument("--fast-mlp-max-resident-mb", type=int)
    chat.add_argument("--max-prompt-tokens", type=int, default=1024)
    chat.add_argument("--max-new-tokens", type=int, default=128)
    chat.add_argument("--max-context-tokens", type=int, default=2048)
    chat.add_argument("--head-block-rows", type=int, default=2048)
    chat.add_argument(
        "--exact-head",
        help="local residual-certified exact LM-head index directory",
    )
    chat.add_argument("--exact-head-max-mb", type=int, default=128)
    organs = sub.add_parser("organs", help="inspect the cold organ bank")
    organs.add_argument("--manifest", help="SHIP-v6/OrganBank manifest")
    organs.add_argument("--artifact-root", help="external SHIP artifact directory")
    organs_sub = organs.add_subparsers(dest="organ_command", required=True)
    organs_sub.add_parser("list", help="list registered organs")
    mount = organs_sub.add_parser("mount", help="verify one organ's digest")
    mount.add_argument("name")
    artifacts = sub.add_parser(
        "artifacts", help="verify or bootstrap external SHIP artifacts"
    )
    artifacts.add_argument(
        "--manifest", help="SHIP-v6 manifest (default: bundled manifest)"
    )
    artifacts.add_argument(
        "--artifact-root", help="destination directory for the five blobs"
    )
    artifact_sub = artifacts.add_subparsers(dest="artifact_command", required=True)
    artifact_sub.add_parser("verify", help="verify all five host/organ blobs")
    artifact_import = artifact_sub.add_parser(
        "import",
        help="copy SHA-matching blobs from an explicit read-only source directory",
    )
    artifact_import.add_argument("source")
    artifact_import.add_argument("--dry-run", action="store_true")
    artifact_import.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing wrong target only after a matching source is found",
    )
    evaluate = sub.add_parser(
        "eval", help="run the cold, training-free SHIP-v6 proof suite"
    )
    evaluate.add_argument(
        "--manifest", help="SHIP-v6 manifest (default: bundled manifest)"
    )
    evaluate.add_argument("--artifact-root", help="external SHIP artifact directory")
    export_hf = sub.add_parser(
        "export-hf",
        help="build and offline-verify the final license-gated HF PoC folder",
    )
    export_hf.add_argument("output", help="new export directory")
    export_hf.add_argument(
        "--manifest", help="SHIP-v6 manifest (default: bundled manifest)"
    )
    export_hf.add_argument("--artifact-root", help="external SHIP artifact directory")
    export_hf.add_argument(
        "--replace",
        action="store_true",
        help="replace an existing export while preserving it as a backup",
    )
    stream = sub.add_parser(
        "stream",
        help="inspect safetensors or read exact rows under a hard byte budget",
    )
    stream.add_argument(
        "source", help="HF repo id, or a directory together with --local"
    )
    stream.add_argument(
        "--local", action="store_true", help="treat source as an offline directory"
    )
    stream.add_argument(
        "--revision",
        default="main",
        help="source revision; pin a commit for remote use",
    )
    stream.add_argument(
        "--budget-mb", type=float, default=200.0, help="hard total transfer ceiling"
    )
    stream.add_argument("--cache-dir", help="verified resume-cache directory")
    stream.add_argument(
        "--no-cache", action="store_true", help="disable inventory/range resume caches"
    )
    stream.add_argument(
        "--refresh", action="store_true", help="bypass cached inventory"
    )
    stream.add_argument(
        "--limit", type=int, default=20, help="maximum tensor names in inventory output"
    )
    stream.add_argument(
        "--tensor", help="read this exact 2D tensor instead of listing the inventory"
    )
    stream.add_argument("--start-row", type=int, default=0)
    stream.add_argument("--rows", type=int, default=8)
    serve = sub.add_parser("serve", help="let the organism live in this terminal")
    serve.add_argument("--state", default="~/.immer/life.json")
    serve.add_argument(
        "--manifest", help="SHIP-v6 manifest (default: bundled manifest)"
    )
    serve.add_argument("--artifact-root", help="external SHIP artifact directory")
    serve.add_argument(
        "--dashboard",
        nargs="?",
        const=8787,
        type=int,
        default=8787,
        help="metrics UI port (default 8787)",
    )
    serve.add_argument(
        "--no-dashboard", action="store_true", help="disable the metrics UI"
    )
    serve.add_argument(
        "--council", action="store_true", help="chat through the rat (donor council)"
    )
    serve.add_argument(
        "--local-brain",
        action="store_true",
        help="allow the local Qwen as fallback mouth",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "components":
        return _components()
    if args.command == "doctor":
        return _doctor(
            deep=args.deep,
            artifact_root=args.artifact_root,
            qwen38_root=args.qwen38_root,
            qwen38_causal_bundle=args.qwen38_causal_bundle,
            qwen38_tokenizer=args.qwen38_tokenizer,
            qwen38_q4=args.qwen38_q4,
        )
    if args.command == "solve":
        return _solve(args.question, args.manifest, args.artifact_root)
    if args.command == "chat":
        return _chat_qwen38(args)
    if args.command == "organs":
        return _organs(args)
    if args.command == "artifacts":
        return _artifacts(args)
    if args.command == "eval":
        return _eval_ship(args.manifest, args.artifact_root)
    if args.command == "export-hf":
        return _export_hf(args)
    if args.command == "stream":
        if args.limit < 0:
            raise SystemExit("--limit must be nonnegative")
        return _stream(args)
    if args.command == "serve":
        return _serve(args)
    raise AssertionError(args.command)
