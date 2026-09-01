"""One-shot O1/Q4 charger for the private Qwen layer-63 transition bank."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import stat
from typing import Any

import numpy as np
import torch

from ...knowledge.livecausal import LiveGraph
from ..ooe.compute_crystals import AFFINE_FLOAT64, ComputeCrystal, ComputeCrystalBank
from ..ooe.compute_graph import ComputeOperatorGraph
from ..ooe.crystal import CrystalStore
from ..ooe.operator_harvester import MAX_STATE_BYTES
from .layer_transition_crystal import (
    TARGET_LAYER_INDEX,
    LayerTransitionCrystal,
    LayerTransitionCrystalBank,
    LayerTransitionCrystalIdentity,
)
from .semantic_atlas import GraphRevision


LAYER63_TRANSITION_ENDPOINTS = frozenset(
    {
        (
            "qwen.layer.63.pre-hidden-sketch",
            "qwen.layer.63.post-hidden-sketch",
        ),
        (
            "qwen.layer.63.layer.input-sketch",
            "qwen.layer.63.layer.output-sketch",
        ),
    }
)


class LayerTransitionBuildError(RuntimeError):
    """The O1 operator, Q4 capture, or destination bank is unusable."""


def _existing_real_directory(value: str | Path, label: str) -> Path:
    path = Path(value)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LayerTransitionBuildError(
            f"{label} must name an existing real directory"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise LayerTransitionBuildError(f"{label} must name an existing real directory")
    return path


@dataclass(frozen=True, slots=True)
class PromotedLayerAffine:
    graph_revision_sha256: str
    edge_sha256: str
    compute_crystal_sha256: str
    operator: torch.Tensor
    bias: torch.Tensor


@dataclass(frozen=True, slots=True)
class LayerTransitionBuildResult:
    bank_path: Path
    bank_identity_sha256: str
    crystal_sha256: str
    source_compute_crystal_sha256: str | None
    source_edge_sha256: str | None
    graph_revision_sha256: str
    atlas_revision_sha256: str
    calibration_rows: int
    max_observed_error: float
    error_radius: float
    sketch_radius: float
    packed_weight_bytes_avoided: int


def current_atlas_revision_sha256(atlas_root: str | Path) -> str:
    """Return the authenticated current Semantic Atlas revision identity."""

    path = _existing_real_directory(atlas_root, "Semantic Atlas root")
    try:
        sequence, event_sha256 = LiveGraph(path).store.revision()
        return GraphRevision(sequence, event_sha256).sha256
    except Exception as exc:
        raise LayerTransitionBuildError(
            "Semantic Atlas authority is unavailable or unauthenticated"
        ) from exc


def _decode_affine(
    crystal: ComputeCrystal, sketch_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if crystal.operator_kind != AFFINE_FLOAT64:
        raise LayerTransitionBuildError("selected ComputeCrystal is not affine")
    if crystal.input_abi.trailing_shape != (sketch_dim,):
        raise LayerTransitionBuildError(
            "ComputeCrystal input width differs from projection"
        )
    zeros = np.zeros((1, sketch_dim), dtype=np.float64)
    identity = np.eye(sketch_dim, dtype=np.float64)
    try:
        bias = np.asarray(crystal.apply(zeros)[0], dtype=np.float64)
        operator = np.asarray(crystal.apply(identity), dtype=np.float64) - bias
    except Exception as exc:
        raise LayerTransitionBuildError("ComputeCrystal affine decode failed") from exc
    if operator.shape != (sketch_dim, sketch_dim) or bias.shape != (sketch_dim,):
        raise LayerTransitionBuildError("ComputeCrystal affine output shape is invalid")
    if not np.isfinite(operator).all() or not np.isfinite(bias).all():
        raise LayerTransitionBuildError(
            "ComputeCrystal affine contains non-finite values"
        )
    # ComputeCrystal applies x @ M.T + b.  LayerTransitionCrystal applies x @ A + b.
    return torch.from_numpy(operator.copy()), torch.from_numpy(bias.copy())


def restore_promoted_layer63_affine(
    compute_root: str | Path,
    *,
    sketch_dim: int,
    compute_crystal_sha256: str | None = None,
) -> tuple[str, PromotedLayerAffine | None]:
    """Restore one authenticated promoted whole-layer affine, if one exists."""

    try:
        root = _existing_real_directory(compute_root, "Compute graph root")
        store = CrystalStore(root, max_state_bytes=MAX_STATE_BYTES)
        bank = ComputeCrystalBank(store)
        state = ComputeOperatorGraph(bank).state()
    except Exception as exc:
        raise LayerTransitionBuildError(
            "ComputeOperatorGraph authority is unavailable or unauthenticated"
        ) from exc
    matches: list[tuple[Any, ComputeCrystal]] = []
    for edge in state.edges:
        if (edge.source_state, edge.target_state) not in LAYER63_TRANSITION_ENDPOINTS:
            continue
        if (
            compute_crystal_sha256 is not None
            and edge.crystal_sha256 != compute_crystal_sha256
        ):
            continue
        crystal = bank.restore_crystal(edge.crystal_sha256)
        if crystal.operator_kind == AFFINE_FLOAT64:
            matches.append((edge, crystal))
    if not matches:
        return state.sha256, None
    if len(matches) != 1:
        raise LayerTransitionBuildError(
            "multiple layer-63 affine edges exist; select one ComputeCrystal SHA"
        )
    edge, crystal = matches[0]
    operator, bias = _decode_affine(crystal, sketch_dim)
    return state.sha256, PromotedLayerAffine(
        graph_revision_sha256=state.sha256,
        edge_sha256=edge.sha256,
        compute_crystal_sha256=crystal.sha256,
        operator=operator,
        bias=bias,
    )


class Layer63BoundaryCollector:
    """Pair exact layer.input/layer.output tensors without persisting them."""

    def __init__(self, hidden_dim: int) -> None:
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim <= 0
        ):
            raise ValueError("hidden_dim must be a positive integer")
        self.hidden_dim = hidden_dim
        self._pending: torch.Tensor | None = None
        self._source: list[torch.Tensor] = []
        self._target: list[torch.Tensor] = []

    def __call__(self, layer: int, stage: str, value: torch.Tensor) -> None:
        if layer != TARGET_LAYER_INDEX:
            raise LayerTransitionBuildError("collector received a foreign layer")
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.bfloat16
            or value.ndim != 3
            or value.shape[0] != 1
            or value.shape[-1] != self.hidden_dim
        ):
            raise LayerTransitionBuildError(
                "collector received an invalid BF16 hidden tensor"
            )
        row = (
            value.detach()
            .to(device="cpu", dtype=torch.bfloat16)
            .reshape(-1, self.hidden_dim)
        )
        if stage == "layer.input":
            if self._pending is not None:
                raise LayerTransitionBuildError("layer-63 input was not settled")
            self._pending = row.contiguous()
            return
        if stage != "layer.output":
            raise LayerTransitionBuildError(
                "collector received an unknown boundary stage"
            )
        if self._pending is None or self._pending.shape != row.shape:
            raise LayerTransitionBuildError("layer-63 output has no matching input")
        self._source.append(self._pending)
        self._target.append(row.contiguous())
        self._pending = None

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._pending is not None or not self._source:
            raise LayerTransitionBuildError("layer-63 capture is incomplete")
        return torch.cat(self._source, dim=0), torch.cat(self._target, dim=0)


def capture_exact_layer63(
    model: Any,
    prompt_token_ids: tuple[int, ...],
    *,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], Any]:
    """Run one ordinary generation and return transient exact layer-63 pairs."""

    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must not be empty")
    collector = Layer63BoundaryCollector(int(model.config.dim))
    previous = (
        model.layer_boundary_observer,
        model.layer_boundary_stages,
        model.layer_boundary_layers,
    )
    model.layer_boundary_observer = collector
    model.layer_boundary_stages = ("layer.input", "layer.output")
    model.layer_boundary_layers = (TARGET_LAYER_INDEX,)
    try:
        model.reset_state(release=True)
        generated, evidence = model.generate_greedy(
            [list(prompt_token_ids)],
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            retain_final_state=True,
        )
        source, target = collector.tensors()
        return source, target, tuple(generated), evidence
    finally:
        try:
            model.reset_state(release=True)
        finally:
            (
                model.layer_boundary_observer,
                model.layer_boundary_stages,
                model.layer_boundary_layers,
            ) = previous


def publish_layer_transition_crystal(
    *,
    bank_path: str | Path,
    identity: LayerTransitionCrystalIdentity,
    source_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    packed_weight_bytes_avoided: int,
    promoted: PromotedLayerAffine | None,
    direct_fit: bool,
    coverage_guard: float = 0.0,
    error_guard: float = 0.0,
) -> LayerTransitionBuildResult:
    """Calibrate or fit one crystal, publish it, and discard caller-owned pairs."""

    if promoted is None and not direct_fit:
        raise LayerTransitionBuildError(
            "no promoted layer-63 affine exists; pass direct_fit explicitly"
        )
    if (
        promoted is not None
        and promoted.graph_revision_sha256 != identity.graph_revision_sha256
    ):
        raise LayerTransitionBuildError(
            "promoted affine belongs to a different Compute graph revision"
        )
    if promoted is None and int(source_hidden.shape[0]) < identity.sketch_dim + 1:
        raise LayerTransitionBuildError(
            "direct affine fit needs at least sketch_dim + 1 captured rows"
        )
    if promoted is not None:
        crystal = LayerTransitionCrystal.calibrate_affine(
            identity=identity,
            operator=promoted.operator,
            bias=promoted.bias,
            source_hidden=source_hidden,
            target_hidden=target_hidden,
            logical_weight_bytes_replaced=packed_weight_bytes_avoided,
            source_compute_crystal_sha256=promoted.compute_crystal_sha256,
            source_compute_edge_sha256=promoted.edge_sha256,
            coverage_guard=coverage_guard,
            error_guard=error_guard,
        )
    else:
        crystal = LayerTransitionCrystal.fit(
            identity=identity,
            source_hidden=source_hidden,
            target_hidden=target_hidden,
            logical_weight_bytes_replaced=packed_weight_bytes_avoided,
            coverage_guard=coverage_guard,
            error_guard=error_guard,
        )
    bank = LayerTransitionCrystalBank(bank_path, identity)
    bank.publish(crystal)
    coverage = crystal.coverage
    return LayerTransitionBuildResult(
        bank_path=Path(bank_path).absolute(),
        bank_identity_sha256=identity.identity_sha256,
        crystal_sha256=crystal.crystal_sha256,
        source_compute_crystal_sha256=crystal.source_compute_crystal_sha256,
        source_edge_sha256=None if promoted is None else promoted.edge_sha256,
        graph_revision_sha256=identity.graph_revision_sha256,
        atlas_revision_sha256=identity.atlas_revision_sha256,
        calibration_rows=int(source_hidden.shape[0]),
        max_observed_error=coverage.max_observed_error,
        error_radius=coverage.error_radius,
        sketch_radius=coverage.sketch_radius,
        packed_weight_bytes_avoided=packed_weight_bytes_avoided,
    )


__all__ = [
    "LAYER63_TRANSITION_ENDPOINTS",
    "Layer63BoundaryCollector",
    "LayerTransitionBuildError",
    "LayerTransitionBuildResult",
    "PromotedLayerAffine",
    "capture_exact_layer63",
    "current_atlas_revision_sha256",
    "publish_layer_transition_crystal",
    "restore_promoted_layer63_affine",
]
