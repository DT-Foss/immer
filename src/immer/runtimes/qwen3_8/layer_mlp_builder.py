"""One-shot O1/Q4 charger for the private layer-63 MLP residual bank."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import stat
from typing import Any

import torch

from ..ooe.compute_crystals import ComputeCrystalBank
from ..ooe.compute_graph import ComputeOperatorGraph
from ..ooe.crystal import CrystalStore
from ..ooe.operator_harvester import MAX_STATE_BYTES
from .layer_mlp_crystal import (
    FEATURE_STAGE,
    SOURCE_STAGE,
    TARGET_LAYER_INDEX,
    TARGET_STAGE,
    Layer63MlpResidualCrystal,
    Layer63MlpResidualCrystalBank,
    Layer63MlpResidualCrystalIdentity,
)
from .layer_transition_builder import current_atlas_revision_sha256


class LayerMlpBuildError(RuntimeError):
    """The compute authority, exact capture, or destination bank is unusable."""


def _existing_real_directory(value: str | Path, label: str) -> Path:
    path = Path(value)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LayerMlpBuildError(
            f"{label} must name an existing real directory"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise LayerMlpBuildError(f"{label} must name an existing real directory")
    return path


@dataclass(frozen=True, slots=True)
class LayerMlpBuildResult:
    bank_path: Path
    bank_identity_sha256: str
    crystal_sha256: str
    graph_revision_sha256: str
    atlas_revision_sha256: str
    calibration_rows: int
    feature_rank: int
    max_observed_error: float
    error_radius: float
    feature_radius: float
    packed_weight_bytes_avoided: int


def current_compute_graph_revision_sha256(compute_root: str | Path) -> str:
    """Return the authenticated current ComputeOperatorGraph revision."""

    root = _existing_real_directory(compute_root, "Compute graph root")
    try:
        store = CrystalStore(root, max_state_bytes=MAX_STATE_BYTES)
        bank = ComputeCrystalBank(store)
        return ComputeOperatorGraph(bank).state().sha256
    except Exception as exc:
        raise LayerMlpBuildError(
            "ComputeOperatorGraph authority is unavailable or unauthenticated"
        ) from exc


class Layer63MlpTripleCollector:
    """Pair exact attention-residual/MLP-input/layer-output triples in RAM."""

    def __init__(self, hidden_dim: int) -> None:
        if (
            isinstance(hidden_dim, bool)
            or not isinstance(hidden_dim, int)
            or hidden_dim <= 0
        ):
            raise ValueError("hidden_dim must be a positive integer")
        self.hidden_dim = hidden_dim
        self._base_pending: torch.Tensor | None = None
        self._feature_pending: torch.Tensor | None = None
        self._base: list[torch.Tensor] = []
        self._feature: list[torch.Tensor] = []
        self._target: list[torch.Tensor] = []

    def _rows(self, value: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.bfloat16
            or value.layout != torch.strided
            or value.ndim != 3
            or value.shape[0] != 1
            or value.shape[-1] != self.hidden_dim
        ):
            raise LayerMlpBuildError("collector received an invalid BF16 hidden tensor")
        rows = (
            value.detach()
            .to(device="cpu", dtype=torch.bfloat16)
            .reshape(-1, self.hidden_dim)
            .contiguous()
        )
        if rows.shape[0] < 1:
            raise LayerMlpBuildError("collector received an empty hidden tensor")
        return rows

    def __call__(self, layer: int, stage: str, value: torch.Tensor) -> None:
        if layer != TARGET_LAYER_INDEX:
            raise LayerMlpBuildError("collector received a foreign layer")
        rows = self._rows(value)
        if stage == SOURCE_STAGE:
            if self._base_pending is not None or self._feature_pending is not None:
                raise LayerMlpBuildError("previous MLP triple was not settled")
            self._base_pending = rows
            return
        if stage == FEATURE_STAGE:
            if self._base_pending is None or self._feature_pending is not None:
                raise LayerMlpBuildError("MLP input has no matching attention residual")
            if self._base_pending.shape != rows.shape:
                raise LayerMlpBuildError("MLP input row count differs from residual")
            self._feature_pending = rows
            return
        if stage != TARGET_STAGE:
            raise LayerMlpBuildError("collector received an unknown boundary stage")
        if self._base_pending is None or self._feature_pending is None:
            raise LayerMlpBuildError("layer output has no complete MLP input pair")
        if self._base_pending.shape != rows.shape:
            raise LayerMlpBuildError("layer output row count differs from MLP input")
        self._base.append(self._base_pending)
        self._feature.append(self._feature_pending)
        self._target.append(rows)
        self._base_pending = None
        self._feature_pending = None

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self._base_pending is not None
            or self._feature_pending is not None
            or not self._base
        ):
            raise LayerMlpBuildError("layer-63 MLP capture is incomplete")
        return (
            torch.cat(self._base, dim=0),
            torch.cat(self._feature, dim=0),
            torch.cat(self._target, dim=0),
        )


def capture_exact_layer63_mlp(
    model: Any,
    prompt_token_ids: tuple[int, ...],
    *,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...], Any]:
    """Run one ordinary generation and return transient exact MLP triples."""

    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must not be empty")
    collector = Layer63MlpTripleCollector(int(model.config.dim))
    previous = (
        model.layer_boundary_observer,
        model.layer_boundary_stages,
        model.layer_boundary_layers,
    )
    model.layer_boundary_observer = collector
    model.layer_boundary_stages = (SOURCE_STAGE, FEATURE_STAGE, TARGET_STAGE)
    model.layer_boundary_layers = (TARGET_LAYER_INDEX,)
    try:
        model.reset_state(release=True)
        generated, evidence = model.generate_greedy(
            [list(prompt_token_ids)],
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            retain_final_state=True,
        )
        base, feature, target = collector.tensors()
        return base, feature, target, tuple(generated), evidence
    finally:
        try:
            model.reset_state(release=True)
        finally:
            (
                model.layer_boundary_observer,
                model.layer_boundary_stages,
                model.layer_boundary_layers,
            ) = previous


def publish_layer_mlp_residual_crystal(
    *,
    bank_path: str | Path,
    identity: Layer63MlpResidualCrystalIdentity,
    base_hidden: torch.Tensor,
    mlp_input: torch.Tensor,
    target_hidden: torch.Tensor,
    packed_weight_bytes_avoided: int,
    direct_fit: bool,
    ridge: float = 1e-8,
    coverage_guard: float = 0.0,
    error_guard: float = 0.0,
) -> LayerMlpBuildResult:
    """Fit, authenticate, and atomically publish one direct MLP action."""

    if not direct_fit:
        raise LayerMlpBuildError(
            "layer-63 MLP residual publication requires explicit direct_fit"
        )
    if not all(
        isinstance(value, torch.Tensor)
        for value in (base_hidden, mlp_input, target_hidden)
    ):
        raise TypeError("captured MLP triples must be tensors")
    row_count = int(base_hidden.shape[0])
    if row_count < identity.sketch_dim + 1:
        raise LayerMlpBuildError(
            "direct MLP fit needs at least sketch_dim + 1 captured rows"
        )
    try:
        feature_rows = mlp_input.detach().to(device="cpu", dtype=torch.float64)
        feature_rows = feature_rows.reshape(row_count, identity.hidden_dim)
        features = feature_rows @ identity.projection.matrix()
        rank = int(torch.linalg.matrix_rank(features - features.mean(dim=0)).item())
    except Exception as exc:
        raise LayerMlpBuildError(
            "captured MLP feature rank could not be measured"
        ) from exc
    if rank != identity.sketch_dim:
        raise LayerMlpBuildError(
            "captured centered MLP features do not span the sketch dimension"
        )
    try:
        crystal = Layer63MlpResidualCrystal.fit(
            identity=identity,
            base_hidden=base_hidden,
            mlp_input=mlp_input,
            target_hidden=target_hidden,
            packed_weight_bytes_avoided=packed_weight_bytes_avoided,
            ridge=ridge,
            coverage_guard=coverage_guard,
            error_guard=error_guard,
        )
        bank = Layer63MlpResidualCrystalBank(bank_path, identity)
        bank.publish(crystal)
    except (TypeError, ValueError) as exc:
        raise LayerMlpBuildError("direct layer-63 MLP residual fit failed") from exc
    coverage = crystal.coverage
    return LayerMlpBuildResult(
        bank_path=Path(bank_path).absolute(),
        bank_identity_sha256=identity.identity_sha256,
        crystal_sha256=crystal.crystal_sha256,
        graph_revision_sha256=identity.graph_revision_sha256,
        atlas_revision_sha256=identity.atlas_revision_sha256,
        calibration_rows=row_count,
        feature_rank=rank,
        max_observed_error=coverage.max_observed_error,
        error_radius=coverage.error_radius,
        feature_radius=coverage.feature_radius,
        packed_weight_bytes_avoided=crystal.packed_weight_bytes_avoided,
    )


__all__ = [
    "Layer63MlpTripleCollector",
    "LayerMlpBuildError",
    "LayerMlpBuildResult",
    "capture_exact_layer63_mlp",
    "current_atlas_revision_sha256",
    "current_compute_graph_revision_sha256",
    "publish_layer_mlp_residual_crystal",
]
