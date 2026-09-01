"""Additive O1 sufficient statistics for layer-parametric MLP crystals.

Each observation is reduced immediately to float64 sufficient statistics.  The
state therefore survives arbitrarily many ordinary Qwen requests without ever
persisting a hidden row.  Once the centered feature Gram reaches full sketch
rank, the accumulator exposes one centered-ridge crystal and carries a
conservative error/coverage envelope through every later refit.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import stat
import threading

import torch

from .layer_mlp_crystal import (
    Layer63MlpResidualCoverage,
    Layer63MlpResidualCrystal,
    Layer63MlpResidualCrystalIdentity,
    LayerMlpResidualCoverage,
    LayerMlpResidualCrystal,
    LayerMlpResidualCrystalIdentity,
    LayerMlpCrystalIdentityError,
    LayerMlpCrystalIntegrityError,
    _LayerMlpIdentity,
    _as_integrity,
    _generic_identity,
    _identity_from_record,
    _publish_bytes,
    _read_state,
    _state_lock,
)
from .layer_transition_crystal import (
    LayerTransitionCrystalIntegrityError,
    _bounded_add,
    _digest,
    _finite_non_negative,
    _hidden_batch,
    _sealed_document,
    _sha256_document,
    _stable_signature,
    _tensor_from_record,
    _tensor_record,
    _thread_lock,
    _uint,
    _unseal_document,
)


LAYER_MLP_O1_STATS_SCHEMA = "immer.qwen3.8-layer63-mlp-o1-stats/v1"
LAYER_MLP_O1_REFIT_STATS_SCHEMA = "immer.qwen3.8-layer63-mlp-o1-stats/v2"
LAYER_MLP_O1_STATS_ENVELOPE_SCHEMA = "immer.qwen3.8-layer63-mlp-o1-stats-envelope/v1"
LAYER_MLP_O1_SOLUTION_SCHEMA = "immer.qwen3.8-layer63-mlp-o1-solution/v1"
LAYER_MLP_GENERIC_O1_STATS_SCHEMA = "immer.qwen3.8-layer-mlp-o1-stats/v2"
LAYER_MLP_GENERIC_O1_REFIT_STATS_SCHEMA = "immer.qwen3.8-layer-mlp-o1-stats/v3"
LAYER_MLP_GENERIC_O1_STATS_ENVELOPE_SCHEMA = (
    "immer.qwen3.8-layer-mlp-o1-stats-envelope/v2"
)
LAYER_MLP_GENERIC_O1_SOLUTION_SCHEMA = "immer.qwen3.8-layer-mlp-o1-solution/v2"


def _positive_finite(value: object, field: str) -> float:
    result = _finite_non_negative(value, field)
    if result == 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _tensor(
    value: torch.Tensor,
    *,
    shape: tuple[int, ...],
    field: str,
) -> torch.Tensor:
    result = value.detach().to(device="cpu", dtype=torch.float64).contiguous()
    if tuple(result.shape) != shape:
        raise ValueError(f"{field} has an invalid shape")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError(f"{field} contains a non-finite value")
    return result


def _centered_gram(
    count: int,
    sum_f: torch.Tensor,
    sum_ff: torch.Tensor,
) -> torch.Tensor:
    gram = sum_ff - torch.outer(sum_f, sum_f) / count
    return ((gram + gram.T) * 0.5).contiguous()


def _gram_rank(gram: torch.Tensor) -> int:
    return int(torch.linalg.matrix_rank(gram, hermitian=True).item())


@dataclass(frozen=True, slots=True)
class _O1Solution:
    feature_mean: torch.Tensor
    residual_mean: torch.Tensor
    operator: torch.Tensor
    feature_radius: float
    max_observed_error: float
    error_radius: float

    def __post_init__(self) -> None:
        feature = self.feature_mean.detach().to(torch.float64).cpu().contiguous()
        residual = self.residual_mean.detach().to(torch.float64).cpu().contiguous()
        operator = self.operator.detach().to(torch.float64).cpu().contiguous()
        if feature.ndim != 1 or residual.ndim != 1:
            raise ValueError("O1 solution means must be vectors")
        if tuple(operator.shape) != (feature.numel(), residual.numel()):
            raise ValueError("O1 solution operator shape is invalid")
        if not all(
            bool(item.isfinite().all().item()) for item in (feature, residual, operator)
        ):
            raise ValueError("O1 solution contains a non-finite tensor")
        radius = _finite_non_negative(self.feature_radius, "feature_radius")
        observed = _finite_non_negative(
            self.max_observed_error,
            "max_observed_error",
        )
        error = _finite_non_negative(self.error_radius, "error_radius")
        if error < observed:
            raise ValueError("error_radius is below the observed error envelope")
        object.__setattr__(self, "feature_mean", feature)
        object.__setattr__(self, "residual_mean", residual)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "feature_radius", radius)
        object.__setattr__(self, "max_observed_error", observed)
        object.__setattr__(self, "error_radius", error)

    def to_record(
        self,
        *,
        schema: str = LAYER_MLP_O1_SOLUTION_SCHEMA,
    ) -> dict[str, object]:
        if schema not in (
            LAYER_MLP_O1_SOLUTION_SCHEMA,
            LAYER_MLP_GENERIC_O1_SOLUTION_SCHEMA,
        ):
            raise ValueError("MLP O1 solution schema is invalid")
        return {
            "error_radius": self.error_radius,
            "feature_mean": _tensor_record(
                self.feature_mean,
                field="O1 feature_mean",
            ),
            "feature_radius": self.feature_radius,
            "max_observed_error": self.max_observed_error,
            "operator": _tensor_record(self.operator, field="O1 operator"),
            "residual_mean": _tensor_record(
                self.residual_mean,
                field="O1 residual_mean",
            ),
            "schema": schema,
        }

    @classmethod
    def from_record(
        cls,
        value: object,
        *,
        sketch_dim: int,
        hidden_dim: int,
        schema: str = LAYER_MLP_O1_SOLUTION_SCHEMA,
    ) -> "_O1Solution":
        fields = {
            "error_radius",
            "feature_mean",
            "feature_radius",
            "max_observed_error",
            "operator",
            "residual_mean",
            "schema",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != fields
            or value["schema"] != schema
        ):
            raise LayerMlpCrystalIntegrityError("MLP O1 solution fields are invalid")
        try:
            return cls(
                feature_mean=_tensor(
                    _tensor_from_record(
                        value["feature_mean"],
                        field="O1 feature_mean",
                    ),
                    shape=(sketch_dim,),
                    field="feature_mean",
                ),
                residual_mean=_tensor(
                    _tensor_from_record(
                        value["residual_mean"],
                        field="O1 residual_mean",
                    ),
                    shape=(hidden_dim,),
                    field="residual_mean",
                ),
                operator=_tensor(
                    _tensor_from_record(value["operator"], field="O1 operator"),
                    shape=(sketch_dim, hidden_dim),
                    field="operator",
                ),
                feature_radius=value["feature_radius"],
                max_observed_error=value["max_observed_error"],
                error_radius=value["error_radius"],
            )
        except (TypeError, ValueError, LayerTransitionCrystalIntegrityError) as exc:
            raise LayerMlpCrystalIntegrityError(
                "MLP O1 solution values are invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class _O1State:
    identity: _LayerMlpIdentity
    packed_weight_bytes_avoided: int
    ridge: float
    coverage_guard: float
    error_guard: float
    generation: int
    observation_batches: int
    sample_count: int
    sum_f: torch.Tensor
    sum_d: torch.Tensor
    sum_ff: torch.Tensor
    sum_fd: torch.Tensor
    sum_d2: float
    max_feature_norm: float
    max_residual_norm: float
    max_base_norm: float
    feature_rank: int
    solution: _O1Solution | None
    source_o1_state_sha256: str | None = None
    source_o1_generation: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.identity,
            (Layer63MlpResidualCrystalIdentity, LayerMlpResidualCrystalIdentity),
        ):
            raise TypeError("O1 state identity is invalid")
        s = self.identity.sketch_dim
        h = self.identity.hidden_dim
        _uint(
            self.packed_weight_bytes_avoided,
            field="packed_weight_bytes_avoided",
            positive=True,
        )
        _positive_finite(self.ridge, "ridge")
        _finite_non_negative(self.coverage_guard, "coverage_guard")
        _finite_non_negative(self.error_guard, "error_guard")
        _uint(self.generation, field="generation")
        _uint(self.observation_batches, field="observation_batches", positive=True)
        _uint(self.sample_count, field="sample_count", positive=True)
        if self.observation_batches > self.sample_count:
            raise ValueError("observation_batches exceeds sample_count")
        tensors = (
            ("sum_f", self.sum_f, (s,)),
            ("sum_d", self.sum_d, (h,)),
            ("sum_ff", self.sum_ff, (s, s)),
            ("sum_fd", self.sum_fd, (s, h)),
        )
        for field, value, shape in tensors:
            object.__setattr__(self, field, _tensor(value, shape=shape, field=field))
        sum_d2 = _finite_non_negative(self.sum_d2, "sum_d2")
        max_feature = _finite_non_negative(
            self.max_feature_norm,
            "max_feature_norm",
        )
        max_residual = _finite_non_negative(
            self.max_residual_norm,
            "max_residual_norm",
        )
        max_base = _finite_non_negative(self.max_base_norm, "max_base_norm")
        _uint(self.feature_rank, field="feature_rank", maximum=s)
        actual_rank = _gram_rank(
            _centered_gram(self.sample_count, self.sum_f, self.sum_ff)
        )
        if self.feature_rank != actual_rank:
            raise ValueError("feature_rank differs from sufficient statistics")
        if self.solution is not None and actual_rank < s:
            raise ValueError("O1 solution exists before full feature rank")
        if self.solution is not None:
            if tuple(self.solution.feature_mean.shape) != (s,):
                raise ValueError("O1 solution feature width differs from identity")
            if tuple(self.solution.residual_mean.shape) != (h,):
                raise ValueError("O1 solution hidden width differs from identity")
        if (self.source_o1_state_sha256 is None) != (self.source_o1_generation is None):
            raise ValueError("O1 refit provenance must be supplied together")
        if self.source_o1_state_sha256 is not None:
            _digest(self.source_o1_state_sha256, "source_o1_state_sha256")
            _uint(
                self.source_o1_generation,
                field="source_o1_generation",
                positive=True,
            )
        object.__setattr__(self, "sum_d2", sum_d2)
        object.__setattr__(self, "max_feature_norm", max_feature)
        object.__setattr__(self, "max_residual_norm", max_residual)
        object.__setattr__(self, "max_base_norm", max_base)

    @property
    def ready(self) -> bool:
        return self.solution is not None

    @property
    def state_sha256(self) -> str:
        return _sha256_document(self.to_record())

    def to_record(self) -> dict[str, object]:
        generic = _generic_identity(self.identity)
        solution_schema = (
            LAYER_MLP_GENERIC_O1_SOLUTION_SCHEMA
            if generic
            else LAYER_MLP_O1_SOLUTION_SCHEMA
        )
        record: dict[str, object] = {
            "coverage_guard": self.coverage_guard,
            "error_guard": self.error_guard,
            "feature_rank": self.feature_rank,
            "generation": self.generation,
            "identity": self.identity.to_record(),
            "identity_sha256": self.identity.identity_sha256,
            "max_base_norm": self.max_base_norm,
            "max_feature_norm": self.max_feature_norm,
            "max_residual_norm": self.max_residual_norm,
            "observation_batches": self.observation_batches,
            "packed_weight_bytes_avoided": self.packed_weight_bytes_avoided,
            "ridge": self.ridge,
            "sample_count": self.sample_count,
            "schema": (
                LAYER_MLP_GENERIC_O1_STATS_SCHEMA
                if generic
                else LAYER_MLP_O1_STATS_SCHEMA
            ),
            "solution": (
                None
                if self.solution is None
                else self.solution.to_record(schema=solution_schema)
            ),
            "sum_d": _tensor_record(self.sum_d, field="O1 sum_d"),
            "sum_d2": self.sum_d2,
            "sum_f": _tensor_record(self.sum_f, field="O1 sum_f"),
            "sum_fd": _tensor_record(self.sum_fd, field="O1 sum_fd"),
            "sum_ff": _tensor_record(self.sum_ff, field="O1 sum_ff"),
        }
        if self.source_o1_state_sha256 is not None:
            record.update(
                {
                    "schema": (
                        LAYER_MLP_GENERIC_O1_REFIT_STATS_SCHEMA
                        if generic
                        else LAYER_MLP_O1_REFIT_STATS_SCHEMA
                    ),
                    "source_o1_generation": self.source_o1_generation,
                    "source_o1_state_sha256": self.source_o1_state_sha256,
                }
            )
        return record

    def to_bytes(self) -> bytes:
        envelope_schema = (
            LAYER_MLP_GENERIC_O1_STATS_ENVELOPE_SCHEMA
            if _generic_identity(self.identity)
            else LAYER_MLP_O1_STATS_ENVELOPE_SCHEMA
        )
        return _sealed_document(self.to_record(), envelope_schema)

    @classmethod
    def from_bytes(cls, value: bytes) -> "_O1State":
        generic_envelope = False
        try:
            try:
                body = _unseal_document(
                    value,
                    schema=LAYER_MLP_O1_STATS_ENVELOPE_SCHEMA,
                    kind="layer-63 MLP O1 state",
                )
            except LayerTransitionCrystalIntegrityError:
                generic_envelope = True
                body = _unseal_document(
                    value,
                    schema=LAYER_MLP_GENERIC_O1_STATS_ENVELOPE_SCHEMA,
                    kind="layer-parametric MLP O1 state",
                )
        except Exception as exc:
            raise _as_integrity(exc) from exc
        legacy_fields = {
            "coverage_guard",
            "error_guard",
            "feature_rank",
            "generation",
            "identity",
            "identity_sha256",
            "max_base_norm",
            "max_feature_norm",
            "max_residual_norm",
            "observation_batches",
            "packed_weight_bytes_avoided",
            "ridge",
            "sample_count",
            "schema",
            "solution",
            "sum_d",
            "sum_d2",
            "sum_f",
            "sum_fd",
            "sum_ff",
        }
        refit_fields = legacy_fields | {
            "source_o1_generation",
            "source_o1_state_sha256",
        }
        actual_fields = set(body)
        if actual_fields not in (legacy_fields, refit_fields):
            raise LayerMlpCrystalIntegrityError("MLP O1 state fields are invalid")
        is_refit = actual_fields == refit_fields
        schema = body["schema"]
        if schema not in (
            LAYER_MLP_O1_REFIT_STATS_SCHEMA if is_refit else LAYER_MLP_O1_STATS_SCHEMA,
            (
                LAYER_MLP_GENERIC_O1_REFIT_STATS_SCHEMA
                if is_refit
                else LAYER_MLP_GENERIC_O1_STATS_SCHEMA
            ),
        ):
            raise LayerMlpCrystalIntegrityError("MLP O1 state schema is invalid")
        generic_schema = schema in (
            LAYER_MLP_GENERIC_O1_STATS_SCHEMA,
            LAYER_MLP_GENERIC_O1_REFIT_STATS_SCHEMA,
        )
        if generic_envelope != generic_schema:
            raise LayerMlpCrystalIntegrityError(
                "MLP O1 envelope and body generation differ"
            )
        try:
            identity = _identity_from_record(body["identity"])
            generic = _generic_identity(identity)
            if generic != generic_schema:
                raise ValueError("MLP O1 schema and identity generation differ")
            if body["identity_sha256"] != identity.identity_sha256:
                raise ValueError("MLP O1 identity SHA-256 mismatch")
            solution = body["solution"]
            result = cls(
                identity=identity,
                packed_weight_bytes_avoided=body["packed_weight_bytes_avoided"],
                ridge=body["ridge"],
                coverage_guard=body["coverage_guard"],
                error_guard=body["error_guard"],
                generation=body["generation"],
                observation_batches=body["observation_batches"],
                sample_count=body["sample_count"],
                sum_f=_tensor_from_record(body["sum_f"], field="O1 sum_f"),
                sum_d=_tensor_from_record(body["sum_d"], field="O1 sum_d"),
                sum_ff=_tensor_from_record(body["sum_ff"], field="O1 sum_ff"),
                sum_fd=_tensor_from_record(body["sum_fd"], field="O1 sum_fd"),
                sum_d2=body["sum_d2"],
                max_feature_norm=body["max_feature_norm"],
                max_residual_norm=body["max_residual_norm"],
                max_base_norm=body["max_base_norm"],
                feature_rank=body["feature_rank"],
                solution=(
                    None
                    if solution is None
                    else _O1Solution.from_record(
                        solution,
                        sketch_dim=identity.sketch_dim,
                        hidden_dim=identity.hidden_dim,
                        schema=(
                            LAYER_MLP_GENERIC_O1_SOLUTION_SCHEMA
                            if generic
                            else LAYER_MLP_O1_SOLUTION_SCHEMA
                        ),
                    )
                ),
                source_o1_state_sha256=(
                    body["source_o1_state_sha256"] if is_refit else None
                ),
                source_o1_generation=(
                    body["source_o1_generation"] if is_refit else None
                ),
            )
            if result.solution is None and result.feature_rank == identity.sketch_dim:
                raise ValueError("full-rank MLP O1 state lacks its fitted solution")
            return result
        except (TypeError, ValueError, LayerTransitionCrystalIntegrityError) as exc:
            raise LayerMlpCrystalIntegrityError(
                "MLP O1 state values are invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class Layer63MlpO1Snapshot:
    """Caller-owned summary of one atomically persisted O1 generation."""

    identity_sha256: str
    state_sha256: str
    generation: int
    observation_batches: int
    accumulated_rows: int
    feature_rank: int
    sketch_dim: int
    ready: bool
    feature_radius: float | None
    max_observed_error: float | None
    error_radius: float | None
    crystal_sha256: str | None
    source_o1_state_sha256: str | None
    source_o1_generation: int | None


@dataclass(frozen=True, slots=True)
class LayerMlpO1Snapshot(Layer63MlpO1Snapshot):
    """Snapshot carrying the fully qualified v2 layer boundary identity."""

    layer_index: int
    source_stage: str
    feature_stage: str
    target_stage: str


def _fit_coefficients(
    state: _O1State,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = state.sample_count
    feature_mean = state.sum_f / count
    residual_mean = state.sum_d / count
    gram = _centered_gram(count, state.sum_f, state.sum_ff)
    cross = state.sum_fd - torch.outer(state.sum_f, state.sum_d) / count
    regularizer = (
        torch.eye(state.identity.sketch_dim, dtype=torch.float64) * state.ridge
    )
    operator = torch.linalg.solve(gram + regularizer, cross).contiguous()
    if not bool(operator.isfinite().all().item()):
        raise LayerMlpCrystalIntegrityError("MLP O1 ridge fit is non-finite")
    return feature_mean.contiguous(), residual_mean.contiguous(), operator


def _predictor_transfer_bound(old: _O1Solution, new: _O1Solution) -> float:
    delta_mu = torch.linalg.vector_norm(new.residual_mean - old.residual_mean).item()
    delta_operator = torch.linalg.matrix_norm(
        new.operator - old.operator,
        ord=2,
    ).item()
    delta_feature = torch.linalg.vector_norm(new.feature_mean - old.feature_mean).item()
    new_operator = torch.linalg.matrix_norm(new.operator, ord=2).item()
    return float(
        delta_mu + old.feature_radius * delta_operator + delta_feature * new_operator
    )


def _pending_error_bound(state: _O1State, solution: _O1Solution) -> float:
    operator_norm = float(torch.linalg.matrix_norm(solution.operator, ord=2).item())
    intercept = solution.residual_mean - solution.feature_mean @ solution.operator
    feature_prediction_sum = state.sum_f @ solution.operator
    squared_error = (
        state.sum_d2
        - 2.0
        * float(
            torch.dot(intercept, state.sum_d).item()
            + torch.sum(solution.operator * state.sum_fd).item()
        )
        + state.sample_count * float(torch.dot(intercept, intercept).item())
        + 2.0 * float(torch.dot(intercept, feature_prediction_sum).item())
        + float(
            torch.sum(solution.operator * (state.sum_ff @ solution.operator)).item()
        )
    )
    if squared_error < 0.0:
        tolerance = 1e-10 * max(1.0, state.sum_d2)
        if squared_error < -tolerance:
            raise LayerMlpCrystalIntegrityError(
                "MLP O1 sufficient statistics produced negative residual energy"
            )
        squared_error = 0.0
    predicted_residual = float(
        torch.linalg.vector_norm(solution.residual_mean).item()
        + (
            state.max_feature_norm
            + torch.linalg.vector_norm(solution.feature_mean).item()
        )
        * operator_norm
    )
    # One full BF16 ulp is deliberately used; this remains conservative at
    # exponent boundaries while avoiding any dependency on historical rows.
    rounding = (state.max_base_norm + predicted_residual) / 128.0 + math.sqrt(
        state.identity.hidden_dim
    ) * math.ldexp(1.0, -133)
    return math.sqrt(squared_error) + rounding


def _pending_coverage_bound(state: _O1State, feature_mean: torch.Tensor) -> float:
    return state.max_feature_norm + float(torch.linalg.vector_norm(feature_mean).item())


def _solve_aggregate(
    aggregate: _O1State,
    *,
    old_states: tuple[_O1State, ...],
    exact_batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
) -> _O1State:
    s = aggregate.identity.sketch_dim
    rank = _gram_rank(
        _centered_gram(aggregate.sample_count, aggregate.sum_f, aggregate.sum_ff)
    )
    if rank < s:
        return replace(aggregate, feature_rank=rank, solution=None)
    feature_mean, residual_mean, operator = _fit_coefficients(aggregate)
    provisional = _O1Solution(
        feature_mean=feature_mean,
        residual_mean=residual_mean,
        operator=operator,
        feature_radius=0.0,
        max_observed_error=0.0,
        error_radius=0.0,
    )
    coverage_candidates: list[float] = []
    error_candidates: list[float] = []
    for old in old_states:
        if old.solution is None:
            coverage_candidates.append(
                _pending_coverage_bound(old, feature_mean) + aggregate.coverage_guard
            )
            error_candidates.append(_pending_error_bound(old, provisional))
        else:
            center_shift = float(
                torch.linalg.vector_norm(
                    feature_mean - old.solution.feature_mean
                ).item()
            )
            coverage_candidates.append(old.solution.feature_radius + center_shift)
            error_candidates.append(
                old.solution.max_observed_error
                + _predictor_transfer_bound(old.solution, provisional)
            )
    if exact_batch is not None:
        base, features, target = exact_batch
        centered = features - feature_mean
        predicted = base + residual_mean + centered @ operator
        predicted_bf16 = predicted.to(torch.bfloat16).to(torch.float64)
        errors = torch.linalg.vector_norm(predicted_bf16 - target, dim=1)
        distances = torch.linalg.vector_norm(centered, dim=1)
        error_candidates.append(float(errors.max().item()))
        coverage_candidates.append(
            float(distances.max().item()) + aggregate.coverage_guard
        )
    if not coverage_candidates or not error_candidates:
        raise LayerMlpCrystalIntegrityError("MLP O1 refit has no coverage authority")
    observed = max(error_candidates)
    feature_radius = math.nextafter(max(coverage_candidates), math.inf)
    error_radius = math.nextafter(observed + aggregate.error_guard, math.inf)
    return replace(
        aggregate,
        feature_rank=rank,
        solution=_O1Solution(
            feature_mean=feature_mean,
            residual_mean=residual_mean,
            operator=operator,
            feature_radius=feature_radius,
            max_observed_error=observed,
            error_radius=error_radius,
        ),
    )


class Layer63MlpO1Accumulator:
    """Race-safe additive sufficient-statistics state for one exact identity."""

    def __init__(
        self,
        state_path: str | os.PathLike[str],
        identity: _LayerMlpIdentity,
        *,
        packed_weight_bytes_avoided: int,
        ridge: float = 1e-8,
        coverage_guard: float = 0.0,
        error_guard: float = 0.0,
    ) -> None:
        if not isinstance(
            identity,
            (Layer63MlpResidualCrystalIdentity, LayerMlpResidualCrystalIdentity),
        ):
            raise TypeError("identity must be an MLP residual crystal identity")
        self.state_path = Path(state_path)
        if not self.state_path.name:
            raise ValueError("state_path must name a file")
        self.identity = identity
        self.packed_weight_bytes_avoided = _uint(
            packed_weight_bytes_avoided,
            field="packed_weight_bytes_avoided",
            positive=True,
        )
        self.ridge = _positive_finite(ridge, "ridge")
        self.coverage_guard = _finite_non_negative(
            coverage_guard,
            "coverage_guard",
        )
        self.error_guard = _finite_non_negative(error_guard, "error_guard")
        self._lock: threading.RLock = _thread_lock(self.state_path)
        self._file_signature: tuple[int, int, int, int, int] | None = None
        self._had_persistent_state = False
        self._state: _O1State | None = None
        with self._lock:
            self._reload(required=False)

    @classmethod
    def load(cls, state_path: str | os.PathLike[str]) -> "Layer63MlpO1Accumulator":
        path = Path(state_path)
        raw, signature = _read_state(path)
        state = _O1State.from_bytes(raw)
        result = cls.__new__(cls)
        result.state_path = path
        result.identity = state.identity
        result.packed_weight_bytes_avoided = state.packed_weight_bytes_avoided
        result.ridge = state.ridge
        result.coverage_guard = state.coverage_guard
        result.error_guard = state.error_guard
        result._lock = _thread_lock(path)
        result._file_signature = signature
        result._had_persistent_state = True
        result._state = state
        return result

    def _validate_state(self, state: _O1State) -> None:
        if state.identity.identity_sha256 != self.identity.identity_sha256:
            raise LayerMlpCrystalIdentityError("MLP O1 state has another identity")
        if (
            state.packed_weight_bytes_avoided != self.packed_weight_bytes_avoided
            or state.ridge != self.ridge
            or state.coverage_guard != self.coverage_guard
            or state.error_guard != self.error_guard
        ):
            raise LayerMlpCrystalIdentityError(
                "MLP O1 fit parameters differ from persistent state"
            )

    def _reload(self, *, required: bool) -> None:
        try:
            raw, signature = _read_state(self.state_path)
        except FileNotFoundError:
            if required or self._had_persistent_state:
                raise LayerMlpCrystalIntegrityError("MLP O1 state disappeared")
            self._state = None
            self._file_signature = None
            return
        state = _O1State.from_bytes(raw)
        self._validate_state(state)
        self._state = state
        self._file_signature = signature
        self._had_persistent_state = True

    def _refresh_if_changed(self) -> None:
        try:
            linked = os.lstat(self.state_path)
        except FileNotFoundError:
            if self._had_persistent_state:
                raise LayerMlpCrystalIntegrityError("MLP O1 state disappeared")
            return
        if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
            raise LayerMlpCrystalIntegrityError("MLP O1 state must be a regular file")
        if _stable_signature(linked) != self._file_signature:
            self._reload(required=True)

    def _publish(self, state: _O1State) -> None:
        if state.solution is None and state.feature_rank == self.identity.sketch_dim:
            raise LayerMlpCrystalIntegrityError(
                "full-rank MLP O1 state cannot be published without a solution"
            )
        _publish_bytes(self.state_path, state.to_bytes())
        self._state = state
        self._file_signature = _stable_signature(os.lstat(self.state_path))
        self._had_persistent_state = True

    def _snapshot(self, state: _O1State) -> Layer63MlpO1Snapshot:
        crystal = self._crystal(state)
        solution = state.solution
        fields: dict[str, object] = {}
        snapshot_cls: type[Layer63MlpO1Snapshot] = Layer63MlpO1Snapshot
        if isinstance(self.identity, LayerMlpResidualCrystalIdentity):
            snapshot_cls = LayerMlpO1Snapshot
            fields = {
                "layer_index": self.identity.layer_index,
                "source_stage": self.identity.source_stage,
                "feature_stage": self.identity.feature_stage,
                "target_stage": self.identity.target_stage,
            }
        return snapshot_cls(
            identity_sha256=self.identity.identity_sha256,
            state_sha256=state.state_sha256,
            generation=state.generation,
            observation_batches=state.observation_batches,
            accumulated_rows=state.sample_count,
            feature_rank=state.feature_rank,
            sketch_dim=self.identity.sketch_dim,
            ready=state.ready,
            feature_radius=(None if solution is None else solution.feature_radius),
            max_observed_error=(
                None if solution is None else solution.max_observed_error
            ),
            error_radius=None if solution is None else solution.error_radius,
            crystal_sha256=None if crystal is None else crystal.crystal_sha256,
            source_o1_state_sha256=state.source_o1_state_sha256,
            source_o1_generation=state.source_o1_generation,
            **fields,
        )

    def _crystal(self, state: _O1State) -> Layer63MlpResidualCrystal | None:
        solution = state.solution
        if solution is None:
            return None
        generic = _generic_identity(self.identity)
        crystal_cls = LayerMlpResidualCrystal if generic else Layer63MlpResidualCrystal
        coverage_cls = (
            LayerMlpResidualCoverage if generic else Layer63MlpResidualCoverage
        )
        return crystal_cls(
            identity=self.identity,
            operator=solution.operator,
            feature_mean=solution.feature_mean,
            residual_mean=solution.residual_mean,
            coverage=coverage_cls(
                center=solution.feature_mean,
                feature_radius=solution.feature_radius,
                error_radius=solution.error_radius,
                sample_count=state.sample_count,
                max_observed_error=solution.max_observed_error,
            ),
            packed_weight_bytes_avoided=self.packed_weight_bytes_avoided,
            ridge=self.ridge,
            source_o1_state_sha256=state.state_sha256,
            source_o1_generation=state.generation,
        )

    def observe_and_crystal(
        self,
        base_hidden: torch.Tensor,
        mlp_input: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> tuple[Layer63MlpO1Snapshot, Layer63MlpResidualCrystal | None]:
        """Add one batch and return its exact persisted generation and crystal."""

        base = _hidden_batch(
            base_hidden,
            hidden_dim=self.identity.hidden_dim,
            field="base_hidden",
        )
        feature_input = _hidden_batch(
            mlp_input,
            hidden_dim=self.identity.hidden_dim,
            field="mlp_input",
        )
        target = _hidden_batch(
            target_hidden,
            hidden_dim=self.identity.hidden_dim,
            field="target_hidden",
        )
        if base.shape != feature_input.shape or base.shape != target.shape:
            raise ValueError("captured MLP triple shapes differ")
        features = (feature_input @ self.identity.projection.matrix()).contiguous()
        residuals = (target - base).contiguous()
        rows = int(base.shape[0])
        batch_sum_f = features.sum(dim=0)
        batch_sum_d = residuals.sum(dim=0)
        batch_sum_ff = features.T @ features
        batch_sum_fd = features.T @ residuals
        batch_sum_d2 = float(residuals.square().sum().item())
        batch_max_feature = float(
            torch.linalg.vector_norm(features, dim=1).max().item()
        )
        batch_max_residual = float(
            torch.linalg.vector_norm(residuals, dim=1).max().item()
        )
        batch_max_base = float(torch.linalg.vector_norm(base, dim=1).max().item())
        with self._lock, _state_lock(self.state_path):
            self._reload(required=self._had_persistent_state)
            old = self._state
            if old is None:
                count = rows
                batches = 1
                sum_f = batch_sum_f
                sum_d = batch_sum_d
                sum_ff = batch_sum_ff
                sum_fd = batch_sum_fd
                sum_d2 = batch_sum_d2
                max_feature = batch_max_feature
                max_residual = batch_max_residual
                max_base = batch_max_base
                old_states: tuple[_O1State, ...] = ()
                generation = 1
                source_state_sha256 = None
                source_generation = None
            else:
                count = _bounded_add(old.sample_count, rows)
                if count != old.sample_count + rows:
                    raise LayerMlpCrystalIntegrityError("MLP O1 row counter overflow")
                batches = _bounded_add(old.observation_batches, 1)
                sum_f = old.sum_f + batch_sum_f
                sum_d = old.sum_d + batch_sum_d
                sum_ff = old.sum_ff + batch_sum_ff
                sum_fd = old.sum_fd + batch_sum_fd
                sum_d2 = old.sum_d2 + batch_sum_d2
                max_feature = max(old.max_feature_norm, batch_max_feature)
                max_residual = max(old.max_residual_norm, batch_max_residual)
                max_base = max(old.max_base_norm, batch_max_base)
                old_states = (old,)
                generation = _bounded_add(old.generation, 1)
                source_state_sha256 = old.source_o1_state_sha256
                source_generation = old.source_o1_generation
            pending = _O1State(
                identity=self.identity,
                packed_weight_bytes_avoided=self.packed_weight_bytes_avoided,
                ridge=self.ridge,
                coverage_guard=self.coverage_guard,
                error_guard=self.error_guard,
                generation=generation,
                observation_batches=batches,
                sample_count=count,
                sum_f=sum_f,
                sum_d=sum_d,
                sum_ff=sum_ff,
                sum_fd=sum_fd,
                sum_d2=sum_d2,
                max_feature_norm=max_feature,
                max_residual_norm=max_residual,
                max_base_norm=max_base,
                feature_rank=_gram_rank(_centered_gram(count, sum_f, sum_ff)),
                solution=None,
                source_o1_state_sha256=source_state_sha256,
                source_o1_generation=source_generation,
            )
            state = _solve_aggregate(
                pending,
                old_states=old_states,
                exact_batch=(base, features, target),
            )
            self._publish(state)
            return self._snapshot(state), self._crystal(state)

    def observe(
        self,
        base_hidden: torch.Tensor,
        mlp_input: torch.Tensor,
        target_hidden: torch.Tensor,
    ) -> Layer63MlpO1Snapshot:
        """Atomically add one transient exact batch and refit when rank permits."""

        snapshot, _crystal = self.observe_and_crystal(
            base_hidden,
            mlp_input,
            target_hidden,
        )
        return snapshot

    def merge_from(
        self,
        other: "Layer63MlpO1Accumulator | str | os.PathLike[str]",
    ) -> Layer63MlpO1Snapshot:
        """Add another sealed sufficient-statistics state without raw rows."""

        source = (
            other
            if isinstance(other, Layer63MlpO1Accumulator)
            else type(self).load(other)
        )
        if source.state_path.absolute() == self.state_path.absolute():
            raise ValueError("cannot merge an O1 state into itself")
        with source._lock:
            source._refresh_if_changed()
            incoming = source._state
        if incoming is None:
            raise LayerMlpCrystalIntegrityError("source MLP O1 state is empty")
        self._validate_state(incoming)
        with self._lock, _state_lock(self.state_path):
            self._reload(required=self._had_persistent_state)
            old = self._state
            if old is None:
                aggregate = replace(incoming, generation=1)
                old_states = (incoming,)
            else:
                count = _bounded_add(old.sample_count, incoming.sample_count)
                batches = _bounded_add(
                    old.observation_batches,
                    incoming.observation_batches,
                )
                if count != old.sample_count + incoming.sample_count:
                    raise LayerMlpCrystalIntegrityError("MLP O1 row counter overflow")
                shared_provenance = (
                    old.source_o1_state_sha256 == incoming.source_o1_state_sha256
                    and old.source_o1_generation == incoming.source_o1_generation
                )
                aggregate = _O1State(
                    identity=self.identity,
                    packed_weight_bytes_avoided=self.packed_weight_bytes_avoided,
                    ridge=self.ridge,
                    coverage_guard=self.coverage_guard,
                    error_guard=self.error_guard,
                    generation=_bounded_add(old.generation, 1),
                    observation_batches=batches,
                    sample_count=count,
                    sum_f=old.sum_f + incoming.sum_f,
                    sum_d=old.sum_d + incoming.sum_d,
                    sum_ff=old.sum_ff + incoming.sum_ff,
                    sum_fd=old.sum_fd + incoming.sum_fd,
                    sum_d2=old.sum_d2 + incoming.sum_d2,
                    max_feature_norm=max(
                        old.max_feature_norm,
                        incoming.max_feature_norm,
                    ),
                    max_residual_norm=max(
                        old.max_residual_norm,
                        incoming.max_residual_norm,
                    ),
                    max_base_norm=max(old.max_base_norm, incoming.max_base_norm),
                    feature_rank=_gram_rank(
                        _centered_gram(
                            count,
                            old.sum_f + incoming.sum_f,
                            old.sum_ff + incoming.sum_ff,
                        )
                    ),
                    solution=None,
                    source_o1_state_sha256=(
                        old.source_o1_state_sha256 if shared_provenance else None
                    ),
                    source_o1_generation=(
                        old.source_o1_generation if shared_provenance else None
                    ),
                )
                old_states = (old, incoming)
            state = _solve_aggregate(
                aggregate,
                old_states=old_states,
                exact_batch=None,
            )
            self._publish(state)
            return self._snapshot(state)

    def centered_gram_scale(self) -> float:
        """Return ``trace(centered Gram) / sketch_dim`` for ridge scaling."""

        with self._lock:
            self._refresh_if_changed()
            state = self._state
            if state is None:
                raise LayerMlpCrystalIntegrityError("MLP O1 state has no observations")
            gram = _centered_gram(state.sample_count, state.sum_f, state.sum_ff)
            scale = float(torch.trace(gram).item()) / self.identity.sketch_dim
            if not math.isfinite(scale) or scale <= 0.0:
                raise LayerMlpCrystalIntegrityError(
                    "MLP O1 centered Gram has no positive ridge scale"
                )
            return scale

    def fork_with_ridge(
        self,
        destination_state_path: str | os.PathLike[str],
        ridge: float,
    ) -> Layer63MlpResidualCrystal:
        """Refit sealed statistics under a new ridge without loading Qwen rows."""

        ridge_value = _positive_finite(ridge, "ridge")
        destination_path = Path(destination_state_path)
        if destination_path.absolute() == self.state_path.absolute():
            raise ValueError("ridge refit destination must differ from its source")
        with self._lock:
            self._refresh_if_changed()
            source = self._state
            if source is None:
                raise LayerMlpCrystalIntegrityError("MLP O1 state has no observations")
            if source.feature_rank != self.identity.sketch_dim:
                raise LayerMlpCrystalIntegrityError(
                    "MLP O1 ridge refit requires full feature rank"
                )
            source_sha256 = source.state_sha256
            generation = _bounded_add(source.generation, 1)
            if generation == source.generation:
                raise LayerMlpCrystalIntegrityError(
                    "MLP O1 generation cannot advance for ridge refit"
                )
            pending = replace(
                source,
                ridge=ridge_value,
                generation=generation,
                solution=None,
                source_o1_state_sha256=source_sha256,
                source_o1_generation=source.generation,
            )
            refitted = _solve_aggregate(
                pending,
                old_states=(pending,),
                exact_batch=None,
            )
            if source.solution is None or refitted.solution is None:
                raise LayerMlpCrystalIntegrityError(
                    "full-rank MLP O1 ridge refit produced no solution"
                )
            # Ridge changes W, not the sufficient-statistics feature mean.
            # The source coverage ball therefore remains exact and tighter
            # than the raw-norm pending bound.
            refitted = replace(
                refitted,
                solution=replace(
                    refitted.solution,
                    feature_radius=source.solution.feature_radius,
                ),
            )
        with _state_lock(destination_path):
            try:
                os.lstat(destination_path)
            except FileNotFoundError:
                pass
            else:
                raise LayerMlpCrystalIntegrityError(
                    "MLP O1 ridge refit destination already exists"
                )
            destination = type(self)(
                destination_path,
                self.identity,
                packed_weight_bytes_avoided=self.packed_weight_bytes_avoided,
                ridge=ridge_value,
                coverage_guard=self.coverage_guard,
                error_guard=self.error_guard,
            )
            destination._publish(refitted)
            crystal = destination._crystal(refitted)
            if crystal is None:
                raise LayerMlpCrystalIntegrityError(
                    "full-rank MLP O1 ridge refit produced no crystal"
                )
            return crystal

    def snapshot(self) -> Layer63MlpO1Snapshot:
        with self._lock:
            self._refresh_if_changed()
            if self._state is None:
                raise LayerMlpCrystalIntegrityError("MLP O1 state has no observations")
            return self._snapshot(self._state)

    def current_crystal(self) -> Layer63MlpResidualCrystal | None:
        with self._lock:
            self._refresh_if_changed()
            if self._state is None:
                return None
            return self._crystal(self._state)


class LayerMlpO1Accumulator(Layer63MlpO1Accumulator):
    """Additive O1 state for a layer-parametric v2 MLP identity."""

    def __init__(
        self,
        state_path: str | os.PathLike[str],
        identity: LayerMlpResidualCrystalIdentity,
        *,
        packed_weight_bytes_avoided: int,
        ridge: float = 1e-8,
        coverage_guard: float = 0.0,
        error_guard: float = 0.0,
    ) -> None:
        if not isinstance(identity, LayerMlpResidualCrystalIdentity):
            raise TypeError("identity must be a LayerMlpResidualCrystalIdentity")
        super().__init__(
            state_path,
            identity,
            packed_weight_bytes_avoided=packed_weight_bytes_avoided,
            ridge=ridge,
            coverage_guard=coverage_guard,
            error_guard=error_guard,
        )

    @classmethod
    def load(cls, state_path: str | os.PathLike[str]) -> "LayerMlpO1Accumulator":
        accumulator = super().load(state_path)
        if not isinstance(accumulator.identity, LayerMlpResidualCrystalIdentity):
            raise LayerMlpCrystalIdentityError(
                "generic MLP O1 loader rejected a layer-63 v1 identity"
            )
        return accumulator


__all__ = [
    "LAYER_MLP_GENERIC_O1_REFIT_STATS_SCHEMA",
    "LAYER_MLP_GENERIC_O1_SOLUTION_SCHEMA",
    "LAYER_MLP_GENERIC_O1_STATS_ENVELOPE_SCHEMA",
    "LAYER_MLP_GENERIC_O1_STATS_SCHEMA",
    "LAYER_MLP_O1_REFIT_STATS_SCHEMA",
    "LAYER_MLP_O1_STATS_ENVELOPE_SCHEMA",
    "LAYER_MLP_O1_STATS_SCHEMA",
    "Layer63MlpO1Accumulator",
    "Layer63MlpO1Snapshot",
    "LayerMlpO1Accumulator",
    "LayerMlpO1Snapshot",
]
