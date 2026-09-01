"""Private layer-parametric MLP residual crystals for the local Qwen3.8 runtime.

The crystal keeps the exact attention result ``x`` and replaces only the
three packed MLP projections.  Given the exact normalised MLP input ``u`` it
computes::

    f = u P
    y = BF16(x + residual_mean + (f - feature_mean) W)

``P`` is the existing deterministic O1 cartography Rademacher projection.
The wide residual operator ``W`` is fitted from exact BF16 runtime triples and
is useful only inside its authenticated finite coverage ball and explicit
error budget.

Artifacts and banks are canonical hash-sealed JSON containing exact
little-endian float64 tensor records.  Raw hidden-state samples are never
persisted.  Publication uses the same race-safe, symlink-safe, fsync-backed
atomic storage primitives as the layer-transition bank.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
import stat
import threading
from typing import Iterator

import torch

from .layer_transition_crystal import (
    LayerTransitionCrystalError,
    LayerTransitionCrystalIntegrityError,
    LayerTransitionProjectionIdentity,
    _atomic_write,
    _bounded_add,
    _digest,
    _exclusive_state_lock,
    _finite_non_negative,
    _hidden_batch,
    _k1_hidden,
    _sealed_document,
    _sha256_document,
    _stable_regular_bytes,
    _stable_signature,
    _tensor_from_record,
    _tensor_record,
    _thread_lock,
    _uint,
    _unseal_document,
)


TARGET_LAYER_INDEX = 63
MAX_LAYER_INDEX = 63
SOURCE_STAGE = "attention.residual"
FEATURE_STAGE = "mlp.input"
TARGET_STAGE = "layer.output"

LAYER_MLP_RESIDUAL_ACTION_ABI = (
    "immer.qwen3.8/layer63-mlp-residual-rademacher-centered-ridge-bf16/v1"
)
LAYER_MLP_RESIDUAL_IDENTITY_SCHEMA = (
    "immer.qwen3.8-layer63-mlp-residual-crystal-identity/v1"
)
LAYER_MLP_RESIDUAL_COVERAGE_SCHEMA = (
    "immer.qwen3.8-layer63-mlp-residual-crystal-coverage/v1"
)
LAYER_MLP_RESIDUAL_CRYSTAL_SCHEMA = "immer.qwen3.8-layer63-mlp-residual-crystal/v1"
LAYER_MLP_RESIDUAL_O1_CRYSTAL_SCHEMA = "immer.qwen3.8-layer63-mlp-residual-crystal/v2"
LAYER_MLP_RESIDUAL_CRYSTAL_ENVELOPE_SCHEMA = (
    "immer.qwen3.8-layer63-mlp-residual-crystal-envelope/v1"
)
LAYER_MLP_RESIDUAL_BANK_SCHEMA = "immer.qwen3.8-layer63-mlp-residual-bank/v1"
LAYER_MLP_RESIDUAL_BANK_ENVELOPE_SCHEMA = (
    "immer.qwen3.8-layer63-mlp-residual-bank-envelope/v1"
)

# Layer-63 v1 artifacts are deployed private state.  The generic lane therefore
# has its own ABI and schemas instead of silently assigning new semantics to
# any existing digest.
LAYER_MLP_RESIDUAL_GENERIC_ACTION_ABI = (
    "immer.qwen3.8/layer-mlp-residual-rademacher-centered-ridge-bf16/v2"
)
LAYER_MLP_RESIDUAL_GENERIC_IDENTITY_SCHEMA = (
    "immer.qwen3.8-layer-mlp-residual-crystal-identity/v2"
)
LAYER_MLP_RESIDUAL_GENERIC_COVERAGE_SCHEMA = (
    "immer.qwen3.8-layer-mlp-residual-crystal-coverage/v2"
)
LAYER_MLP_RESIDUAL_GENERIC_CRYSTAL_SCHEMA = (
    "immer.qwen3.8-layer-mlp-residual-crystal/v2"
)
LAYER_MLP_RESIDUAL_GENERIC_O1_CRYSTAL_SCHEMA = (
    "immer.qwen3.8-layer-mlp-residual-crystal/v3"
)
LAYER_MLP_RESIDUAL_GENERIC_CRYSTAL_ENVELOPE_SCHEMA = (
    "immer.qwen3.8-layer-mlp-residual-crystal-envelope/v2"
)
LAYER_MLP_RESIDUAL_GENERIC_BANK_SCHEMA = "immer.qwen3.8-layer-mlp-residual-bank/v2"
LAYER_MLP_RESIDUAL_GENERIC_BANK_ENVELOPE_SCHEMA = (
    "immer.qwen3.8-layer-mlp-residual-bank-envelope/v2"
)

_MAX_COUNTER = (1 << 63) - 1
_MAX_CRYSTALS = 1 << 16
_MAX_STATE_BYTES = 512 * 1024 * 1024


class LayerMlpCrystalError(RuntimeError):
    """A layer-63 MLP residual crystal input or state is invalid."""


class LayerMlpCrystalIntegrityError(LayerMlpCrystalError):
    """A crystal or bank is malformed, unstable, or hash-inconsistent."""


class LayerMlpCrystalIdentityError(LayerMlpCrystalError):
    """A crystal belongs to another immutable runtime identity."""


class LayerMlpCrystalCapacityError(LayerMlpCrystalError):
    """A bounded MLP residual bank has no room for another crystal."""


def _as_integrity(exc: Exception) -> LayerMlpCrystalIntegrityError:
    return LayerMlpCrystalIntegrityError(str(exc))


def _read_state(path: Path) -> tuple[bytes, tuple[int, int, int, int, int]]:
    try:
        return _stable_regular_bytes(path)
    except FileNotFoundError:
        raise
    except LayerTransitionCrystalIntegrityError as exc:
        raise _as_integrity(exc) from exc


@contextmanager
def _state_lock(path: Path) -> Iterator[None]:
    try:
        with _exclusive_state_lock(path):
            yield
    except LayerTransitionCrystalIntegrityError as exc:
        raise _as_integrity(exc) from exc


def _publish_bytes(path: Path, data: bytes) -> None:
    try:
        _atomic_write(path, data)
    except LayerTransitionCrystalError as exc:
        raise _as_integrity(exc) from exc


def _layer_index(value: object) -> int:
    return _uint(value, field="layer_index", maximum=MAX_LAYER_INDEX)


def _qualified_stage(layer_index: int, stage: str) -> str:
    return f"qwen.layer.{_layer_index(layer_index)}.{stage}"


@dataclass(frozen=True, slots=True)
class Layer63MlpResidualCrystalIdentity:
    """Complete immutable scope of one layer-63 MLP residual action."""

    model_sha256: str
    q4_sha256: str
    graph_revision_sha256: str
    atlas_revision_sha256: str
    projection: LayerTransitionProjectionIdentity
    layer_index: int = TARGET_LAYER_INDEX
    source_stage: str = SOURCE_STAGE
    feature_stage: str = FEATURE_STAGE
    target_stage: str = TARGET_STAGE
    action_abi: str = LAYER_MLP_RESIDUAL_ACTION_ABI

    def __post_init__(self) -> None:
        for field in (
            "model_sha256",
            "q4_sha256",
            "graph_revision_sha256",
            "atlas_revision_sha256",
        ):
            _digest(getattr(self, field), field)
        if not isinstance(self.projection, LayerTransitionProjectionIdentity):
            raise TypeError("projection must be a LayerTransitionProjectionIdentity")
        if self.layer_index != TARGET_LAYER_INDEX:
            raise ValueError(f"only layer {TARGET_LAYER_INDEX} is supported")
        if self.source_stage != SOURCE_STAGE:
            raise ValueError(f"source_stage must be {SOURCE_STAGE!r}")
        if self.feature_stage != FEATURE_STAGE:
            raise ValueError(f"feature_stage must be {FEATURE_STAGE!r}")
        if self.target_stage != TARGET_STAGE:
            raise ValueError(f"target_stage must be {TARGET_STAGE!r}")
        if self.action_abi != LAYER_MLP_RESIDUAL_ACTION_ABI:
            raise ValueError("MLP residual action ABI is not implemented")

    @property
    def hidden_dim(self) -> int:
        return self.projection.hidden_dim

    @property
    def sketch_dim(self) -> int:
        return self.projection.sketch_dim

    @property
    def identity_sha256(self) -> str:
        return _sha256_document(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "action_abi": self.action_abi,
            "atlas_revision_sha256": self.atlas_revision_sha256,
            "feature_stage": self.feature_stage,
            "graph_revision_sha256": self.graph_revision_sha256,
            "layer_index": self.layer_index,
            "model_sha256": self.model_sha256,
            "projection": self.projection.to_record(),
            "q4_sha256": self.q4_sha256,
            "schema": LAYER_MLP_RESIDUAL_IDENTITY_SCHEMA,
            "source_stage": self.source_stage,
            "target_stage": self.target_stage,
        }

    @classmethod
    def from_record(cls, value: object) -> "Layer63MlpResidualCrystalIdentity":
        fields = {
            "action_abi",
            "atlas_revision_sha256",
            "feature_stage",
            "graph_revision_sha256",
            "layer_index",
            "model_sha256",
            "projection",
            "q4_sha256",
            "schema",
            "source_stage",
            "target_stage",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise LayerMlpCrystalIntegrityError(
                "MLP crystal identity fields are invalid"
            )
        if value["schema"] != LAYER_MLP_RESIDUAL_IDENTITY_SCHEMA:
            raise LayerMlpCrystalIntegrityError(
                "MLP crystal identity schema is invalid"
            )
        try:
            return cls(
                model_sha256=value["model_sha256"],
                q4_sha256=value["q4_sha256"],
                graph_revision_sha256=value["graph_revision_sha256"],
                atlas_revision_sha256=value["atlas_revision_sha256"],
                projection=LayerTransitionProjectionIdentity.from_record(
                    value["projection"]
                ),
                layer_index=value["layer_index"],
                source_stage=value["source_stage"],
                feature_stage=value["feature_stage"],
                target_stage=value["target_stage"],
                action_abi=value["action_abi"],
            )
        except (TypeError, ValueError, LayerTransitionCrystalIntegrityError) as exc:
            raise LayerMlpCrystalIntegrityError(
                "MLP crystal identity values are invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class LayerMlpResidualCrystalIdentity:
    """Immutable v2 identity for one MLP action at any Qwen layer."""

    model_sha256: str
    q4_sha256: str
    graph_revision_sha256: str
    atlas_revision_sha256: str
    projection: LayerTransitionProjectionIdentity
    layer_index: int
    action_abi: str = LAYER_MLP_RESIDUAL_GENERIC_ACTION_ABI

    def __post_init__(self) -> None:
        for field in (
            "model_sha256",
            "q4_sha256",
            "graph_revision_sha256",
            "atlas_revision_sha256",
        ):
            _digest(getattr(self, field), field)
        if not isinstance(self.projection, LayerTransitionProjectionIdentity):
            raise TypeError("projection must be a LayerTransitionProjectionIdentity")
        _layer_index(self.layer_index)
        if self.action_abi != LAYER_MLP_RESIDUAL_GENERIC_ACTION_ABI:
            raise ValueError("generic MLP residual action ABI is not implemented")

    @property
    def source_stage(self) -> str:
        return _qualified_stage(self.layer_index, SOURCE_STAGE)

    @property
    def feature_stage(self) -> str:
        return _qualified_stage(self.layer_index, FEATURE_STAGE)

    @property
    def target_stage(self) -> str:
        return _qualified_stage(self.layer_index, TARGET_STAGE)

    @property
    def hidden_dim(self) -> int:
        return self.projection.hidden_dim

    @property
    def sketch_dim(self) -> int:
        return self.projection.sketch_dim

    @property
    def identity_sha256(self) -> str:
        return _sha256_document(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "action_abi": self.action_abi,
            "atlas_revision_sha256": self.atlas_revision_sha256,
            "feature_stage": self.feature_stage,
            "graph_revision_sha256": self.graph_revision_sha256,
            "layer_index": self.layer_index,
            "model_sha256": self.model_sha256,
            "projection": self.projection.to_record(),
            "q4_sha256": self.q4_sha256,
            "schema": LAYER_MLP_RESIDUAL_GENERIC_IDENTITY_SCHEMA,
            "source_stage": self.source_stage,
            "target_stage": self.target_stage,
        }

    @classmethod
    def from_record(cls, value: object) -> "LayerMlpResidualCrystalIdentity":
        fields = {
            "action_abi",
            "atlas_revision_sha256",
            "feature_stage",
            "graph_revision_sha256",
            "layer_index",
            "model_sha256",
            "projection",
            "q4_sha256",
            "schema",
            "source_stage",
            "target_stage",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise LayerMlpCrystalIntegrityError(
                "generic MLP crystal identity fields are invalid"
            )
        if value["schema"] != LAYER_MLP_RESIDUAL_GENERIC_IDENTITY_SCHEMA:
            raise LayerMlpCrystalIntegrityError(
                "generic MLP crystal identity schema is invalid"
            )
        try:
            result = cls(
                model_sha256=value["model_sha256"],
                q4_sha256=value["q4_sha256"],
                graph_revision_sha256=value["graph_revision_sha256"],
                atlas_revision_sha256=value["atlas_revision_sha256"],
                projection=LayerTransitionProjectionIdentity.from_record(
                    value["projection"]
                ),
                layer_index=value["layer_index"],
                action_abi=value["action_abi"],
            )
            if (
                value["source_stage"] != result.source_stage
                or value["feature_stage"] != result.feature_stage
                or value["target_stage"] != result.target_stage
            ):
                raise ValueError("generic MLP stages differ from layer_index")
            return result
        except (TypeError, ValueError, LayerTransitionCrystalIntegrityError) as exc:
            raise LayerMlpCrystalIntegrityError(
                "generic MLP crystal identity values are invalid"
            ) from exc


_LayerMlpIdentity = Layer63MlpResidualCrystalIdentity | LayerMlpResidualCrystalIdentity


def _identity_from_record(value: object) -> _LayerMlpIdentity:
    if not isinstance(value, Mapping):
        raise LayerMlpCrystalIntegrityError("MLP crystal identity is invalid")
    schema = value.get("schema")
    if schema == LAYER_MLP_RESIDUAL_IDENTITY_SCHEMA:
        return Layer63MlpResidualCrystalIdentity.from_record(value)
    if schema == LAYER_MLP_RESIDUAL_GENERIC_IDENTITY_SCHEMA:
        return LayerMlpResidualCrystalIdentity.from_record(value)
    raise LayerMlpCrystalIntegrityError("MLP crystal identity schema is invalid")


@dataclass(frozen=True, slots=True)
class Layer63MlpResidualCoverage:
    """Finite feature-space coverage and observed BF16 error envelope."""

    center: torch.Tensor
    feature_radius: float
    error_radius: float
    sample_count: int
    max_observed_error: float

    def __post_init__(self) -> None:
        center = self.center.detach().to(device="cpu", dtype=torch.float64).clone()
        if center.ndim != 1 or center.numel() < 1:
            raise ValueError("coverage center must be a non-empty vector")
        if not bool(torch.isfinite(center).all().item()):
            raise ValueError("coverage center must contain only finite values")
        radius = _finite_non_negative(self.feature_radius, "feature_radius")
        error = _finite_non_negative(self.error_radius, "error_radius")
        observed = _finite_non_negative(
            self.max_observed_error,
            "max_observed_error",
        )
        _uint(self.sample_count, field="sample_count", positive=True)
        if error < observed:
            raise ValueError("error_radius cannot be below max_observed_error")
        object.__setattr__(self, "center", center.contiguous())
        object.__setattr__(self, "feature_radius", radius)
        object.__setattr__(self, "error_radius", error)
        object.__setattr__(self, "max_observed_error", observed)

    @property
    def sketch_dim(self) -> int:
        return int(self.center.numel())

    def to_record(self) -> dict[str, object]:
        return {
            "center": _tensor_record(self.center, field="coverage center"),
            "error_radius": self.error_radius,
            "feature_radius": self.feature_radius,
            "max_observed_error": self.max_observed_error,
            "sample_count": self.sample_count,
            "schema": LAYER_MLP_RESIDUAL_COVERAGE_SCHEMA,
        }

    @classmethod
    def from_record(cls, value: object) -> "Layer63MlpResidualCoverage":
        fields = {
            "center",
            "error_radius",
            "feature_radius",
            "max_observed_error",
            "sample_count",
            "schema",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise LayerMlpCrystalIntegrityError("MLP coverage fields are invalid")
        if value["schema"] != LAYER_MLP_RESIDUAL_COVERAGE_SCHEMA:
            raise LayerMlpCrystalIntegrityError("MLP coverage schema is invalid")
        try:
            return cls(
                center=_tensor_from_record(value["center"], field="coverage center"),
                feature_radius=value["feature_radius"],
                error_radius=value["error_radius"],
                sample_count=value["sample_count"],
                max_observed_error=value["max_observed_error"],
            )
        except (TypeError, ValueError, LayerTransitionCrystalIntegrityError) as exc:
            raise LayerMlpCrystalIntegrityError(
                "MLP coverage values are invalid"
            ) from exc


class LayerMlpResidualCoverage(Layer63MlpResidualCoverage):
    """Generic v2 coverage record shared by layer-parametric crystals."""

    __slots__ = ()

    def to_record(self) -> dict[str, object]:
        record = super().to_record()
        record["schema"] = LAYER_MLP_RESIDUAL_GENERIC_COVERAGE_SCHEMA
        return record

    @classmethod
    def from_record(cls, value: object) -> "LayerMlpResidualCoverage":
        fields = {
            "center",
            "error_radius",
            "feature_radius",
            "max_observed_error",
            "sample_count",
            "schema",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise LayerMlpCrystalIntegrityError(
                "generic MLP coverage fields are invalid"
            )
        if value["schema"] != LAYER_MLP_RESIDUAL_GENERIC_COVERAGE_SCHEMA:
            raise LayerMlpCrystalIntegrityError(
                "generic MLP coverage schema is invalid"
            )
        try:
            return cls(
                center=_tensor_from_record(value["center"], field="coverage center"),
                feature_radius=value["feature_radius"],
                error_radius=value["error_radius"],
                sample_count=value["sample_count"],
                max_observed_error=value["max_observed_error"],
            )
        except (TypeError, ValueError, LayerTransitionCrystalIntegrityError) as exc:
            raise LayerMlpCrystalIntegrityError(
                "generic MLP coverage values are invalid"
            ) from exc


def _generic_identity(identity: _LayerMlpIdentity) -> bool:
    return isinstance(identity, LayerMlpResidualCrystalIdentity)


def _coverage_type(
    identity: _LayerMlpIdentity,
) -> type[Layer63MlpResidualCoverage]:
    if _generic_identity(identity):
        return LayerMlpResidualCoverage
    return Layer63MlpResidualCoverage


class Layer63MlpResidualCrystal:
    """One immutable centered-ridge MLP residual action."""

    __slots__ = (
        "identity",
        "packed_weight_bytes_avoided",
        "ridge",
        "_body_sha256",
        "_coverage",
        "_feature_mean",
        "_operator",
        "_residual_mean",
        "source_o1_generation",
        "source_o1_state_sha256",
    )

    def __init__(
        self,
        *,
        identity: _LayerMlpIdentity,
        operator: torch.Tensor,
        feature_mean: torch.Tensor,
        residual_mean: torch.Tensor,
        coverage: Layer63MlpResidualCoverage,
        packed_weight_bytes_avoided: int,
        ridge: float,
        source_o1_state_sha256: str | None = None,
        source_o1_generation: int | None = None,
    ) -> None:
        if not isinstance(
            identity,
            (Layer63MlpResidualCrystalIdentity, LayerMlpResidualCrystalIdentity),
        ):
            raise TypeError("identity must be an MLP residual crystal identity")
        if not isinstance(coverage, Layer63MlpResidualCoverage):
            raise TypeError("coverage must be a Layer63MlpResidualCoverage")
        weight = operator.detach().to(device="cpu", dtype=torch.float64).clone()
        feature = feature_mean.detach().to(device="cpu", dtype=torch.float64).clone()
        residual = residual_mean.detach().to(device="cpu", dtype=torch.float64).clone()
        if tuple(weight.shape) != (identity.sketch_dim, identity.hidden_dim):
            raise ValueError(
                "operator must have shape "
                f"[{identity.sketch_dim}, {identity.hidden_dim}]"
            )
        if tuple(feature.shape) != (identity.sketch_dim,):
            raise ValueError("feature_mean shape differs from projection")
        if tuple(residual.shape) != (identity.hidden_dim,):
            raise ValueError("residual_mean shape differs from hidden width")
        if coverage.sketch_dim != identity.sketch_dim:
            raise ValueError("coverage center dimension differs from projection")
        if not torch.equal(feature, coverage.center):
            raise ValueError("coverage center must equal feature_mean")
        if not all(
            bool(torch.isfinite(value).all().item())
            for value in (weight, feature, residual)
        ):
            raise ValueError("MLP crystal tensors must contain only finite values")
        self.identity = identity
        self._operator = weight.contiguous()
        self._feature_mean = feature.contiguous()
        self._residual_mean = residual.contiguous()
        self._coverage = _coverage_type(identity)(
            center=coverage.center,
            feature_radius=coverage.feature_radius,
            error_radius=coverage.error_radius,
            sample_count=coverage.sample_count,
            max_observed_error=coverage.max_observed_error,
        )
        self.packed_weight_bytes_avoided = _uint(
            packed_weight_bytes_avoided,
            field="packed_weight_bytes_avoided",
            positive=True,
        )
        ridge_value = _finite_non_negative(ridge, "ridge")
        if ridge_value == 0.0:
            raise ValueError("ridge must be positive")
        self.ridge = ridge_value
        if (source_o1_state_sha256 is None) != (source_o1_generation is None):
            raise ValueError(
                "O1 state SHA-256 and generation must be supplied together"
            )
        if source_o1_state_sha256 is not None:
            source_o1_state_sha256 = _digest(
                source_o1_state_sha256,
                "source_o1_state_sha256",
            )
            source_o1_generation = _uint(
                source_o1_generation,
                field="source_o1_generation",
                positive=True,
            )
        self.source_o1_state_sha256 = source_o1_state_sha256
        self.source_o1_generation = source_o1_generation
        self._body_sha256 = _sha256_document(self.to_record())

    @property
    def crystal_sha256(self) -> str:
        return self._body_sha256

    @property
    def content_sha256(self) -> str:
        return self.crystal_sha256

    @property
    def logical_weight_bytes_replaced(self) -> int:
        return self.packed_weight_bytes_avoided

    @property
    def operator(self) -> torch.Tensor:
        return self._operator.clone()

    @property
    def feature_mean(self) -> torch.Tensor:
        return self._feature_mean.clone()

    @property
    def residual_mean(self) -> torch.Tensor:
        return self._residual_mean.clone()

    @property
    def coverage(self) -> Layer63MlpResidualCoverage:
        return _coverage_type(self.identity)(
            center=self._coverage.center,
            feature_radius=self._coverage.feature_radius,
            error_radius=self._coverage.error_radius,
            sample_count=self._coverage.sample_count,
            max_observed_error=self._coverage.max_observed_error,
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "coverage": self._coverage.to_record(),
            "feature_mean": _tensor_record(self._feature_mean, field="feature_mean"),
            "identity": self.identity.to_record(),
            "identity_sha256": self.identity.identity_sha256,
            "operator": _tensor_record(self._operator, field="operator"),
            "packed_weight_bytes_avoided": self.packed_weight_bytes_avoided,
            "residual_mean": _tensor_record(
                self._residual_mean,
                field="residual_mean",
            ),
            "ridge": self.ridge,
            "schema": (
                LAYER_MLP_RESIDUAL_GENERIC_CRYSTAL_SCHEMA
                if _generic_identity(self.identity)
                else LAYER_MLP_RESIDUAL_CRYSTAL_SCHEMA
            ),
        }
        if self.source_o1_state_sha256 is not None:
            record.update(
                {
                    "schema": (
                        LAYER_MLP_RESIDUAL_GENERIC_O1_CRYSTAL_SCHEMA
                        if _generic_identity(self.identity)
                        else LAYER_MLP_RESIDUAL_O1_CRYSTAL_SCHEMA
                    ),
                    "source_o1_generation": self.source_o1_generation,
                    "source_o1_state_sha256": self.source_o1_state_sha256,
                }
            )
        return record

    def to_bytes(self) -> bytes:
        envelope_schema = (
            LAYER_MLP_RESIDUAL_GENERIC_CRYSTAL_ENVELOPE_SCHEMA
            if _generic_identity(self.identity)
            else LAYER_MLP_RESIDUAL_CRYSTAL_ENVELOPE_SCHEMA
        )
        return _sealed_document(self.to_record(), envelope_schema)

    @classmethod
    def from_record(cls, value: object) -> "Layer63MlpResidualCrystal":
        legacy_fields = {
            "coverage",
            "feature_mean",
            "identity",
            "identity_sha256",
            "operator",
            "packed_weight_bytes_avoided",
            "residual_mean",
            "ridge",
            "schema",
        }
        o1_fields = legacy_fields | {
            "source_o1_generation",
            "source_o1_state_sha256",
        }
        if not isinstance(value, Mapping):
            raise LayerMlpCrystalIntegrityError("MLP crystal fields are invalid")
        actual_fields = frozenset(value)
        if actual_fields not in {
            frozenset(legacy_fields),
            frozenset(o1_fields),
        }:
            raise LayerMlpCrystalIntegrityError("MLP crystal fields are invalid")
        is_o1 = actual_fields == frozenset(o1_fields)
        schema = value["schema"]
        legacy_schema = (
            LAYER_MLP_RESIDUAL_O1_CRYSTAL_SCHEMA
            if is_o1
            else LAYER_MLP_RESIDUAL_CRYSTAL_SCHEMA
        )
        generic_schema = (
            LAYER_MLP_RESIDUAL_GENERIC_O1_CRYSTAL_SCHEMA
            if is_o1
            else LAYER_MLP_RESIDUAL_GENERIC_CRYSTAL_SCHEMA
        )
        if schema not in (legacy_schema, generic_schema):
            raise LayerMlpCrystalIntegrityError("MLP crystal schema is invalid")
        try:
            identity = _identity_from_record(value["identity"])
            if (schema == generic_schema) != _generic_identity(identity):
                raise ValueError("MLP crystal schema and identity generation differ")
            if value["identity_sha256"] != identity.identity_sha256:
                raise ValueError("MLP crystal identity SHA-256 mismatch")
            coverage_cls = _coverage_type(identity)
            return cls(
                identity=identity,
                operator=_tensor_from_record(value["operator"], field="operator"),
                feature_mean=_tensor_from_record(
                    value["feature_mean"],
                    field="feature_mean",
                ),
                residual_mean=_tensor_from_record(
                    value["residual_mean"],
                    field="residual_mean",
                ),
                coverage=coverage_cls.from_record(value["coverage"]),
                packed_weight_bytes_avoided=value["packed_weight_bytes_avoided"],
                ridge=value["ridge"],
                source_o1_state_sha256=(
                    value["source_o1_state_sha256"] if is_o1 else None
                ),
                source_o1_generation=(value["source_o1_generation"] if is_o1 else None),
            )
        except (TypeError, ValueError, LayerTransitionCrystalIntegrityError) as exc:
            raise LayerMlpCrystalIntegrityError(
                "MLP crystal values are invalid"
            ) from exc

    @classmethod
    def from_bytes(cls, value: bytes) -> "Layer63MlpResidualCrystal":
        generic_envelope = False
        try:
            try:
                body = _unseal_document(
                    value,
                    schema=LAYER_MLP_RESIDUAL_CRYSTAL_ENVELOPE_SCHEMA,
                    kind="layer-63 MLP residual crystal",
                )
            except LayerTransitionCrystalIntegrityError:
                generic_envelope = True
                body = _unseal_document(
                    value,
                    schema=LAYER_MLP_RESIDUAL_GENERIC_CRYSTAL_ENVELOPE_SCHEMA,
                    kind="layer-parametric MLP residual crystal",
                )
        except LayerTransitionCrystalIntegrityError as exc:
            raise _as_integrity(exc) from exc
        result = cls.from_record(body)
        if generic_envelope != _generic_identity(result.identity):
            raise LayerMlpCrystalIntegrityError(
                "MLP crystal envelope and identity generation differ"
            )
        return result

    @classmethod
    def fit(
        cls,
        *,
        identity: _LayerMlpIdentity,
        base_hidden: torch.Tensor,
        mlp_input: torch.Tensor,
        target_hidden: torch.Tensor,
        packed_weight_bytes_avoided: int,
        ridge: float = 1e-8,
        coverage_guard: float = 0.0,
        error_guard: float = 0.0,
    ) -> "Layer63MlpResidualCrystal":
        """Fit ``D = mu + (uP - fbar)W`` from exact BF16 triples."""

        if not isinstance(
            identity,
            (Layer63MlpResidualCrystalIdentity, LayerMlpResidualCrystalIdentity),
        ):
            raise TypeError("identity must be an MLP residual crystal identity")
        ridge_value = _finite_non_negative(ridge, "ridge")
        if ridge_value == 0.0:
            raise ValueError("ridge must be positive")
        coverage_guard_value = _finite_non_negative(
            coverage_guard,
            "coverage_guard",
        )
        error_guard_value = _finite_non_negative(error_guard, "error_guard")
        base = _hidden_batch(
            base_hidden,
            hidden_dim=identity.hidden_dim,
            field="base_hidden",
        )
        feature_input = _hidden_batch(
            mlp_input,
            hidden_dim=identity.hidden_dim,
            field="mlp_input",
        )
        target = _hidden_batch(
            target_hidden,
            hidden_dim=identity.hidden_dim,
            field="target_hidden",
        )
        if base.shape != feature_input.shape or base.shape != target.shape:
            raise ValueError("captured MLP triple shapes differ")
        row_count = int(base.shape[0])
        if identity.sketch_dim > row_count - 1:
            raise ValueError("sketch_dim must not exceed sample_count - 1")
        projection = identity.projection.matrix()
        features = feature_input @ projection
        residuals = target - base
        feature_mean = features.mean(dim=0)
        residual_mean = residuals.mean(dim=0)
        centered_features = features - feature_mean
        centered_residuals = residuals - residual_mean
        rank = int(torch.linalg.matrix_rank(centered_features).item())
        if rank != identity.sketch_dim:
            raise ValueError("centered feature matrix is not full sketch rank")
        gram = centered_features.T @ centered_features
        regulariser = torch.eye(identity.sketch_dim, dtype=torch.float64) * ridge_value
        operator = torch.linalg.solve(
            gram + regulariser,
            centered_features.T @ centered_residuals,
        ).contiguous()
        predicted = base + residual_mean + centered_features @ operator
        predicted_bf16 = predicted.to(dtype=torch.bfloat16).to(dtype=torch.float64)
        errors = torch.linalg.vector_norm(predicted_bf16 - target, dim=1)
        max_observed = float(errors.max().item())
        distances = torch.linalg.vector_norm(centered_features, dim=1)
        feature_radius = math.nextafter(
            float(distances.max().item()) + coverage_guard_value,
            math.inf,
        )
        error_radius = math.nextafter(
            max_observed + error_guard_value,
            math.inf,
        )
        return cls(
            identity=identity,
            operator=operator,
            feature_mean=feature_mean,
            residual_mean=residual_mean,
            coverage=_coverage_type(identity)(
                center=feature_mean,
                feature_radius=feature_radius,
                error_radius=error_radius,
                sample_count=row_count,
                max_observed_error=max_observed,
            ),
            packed_weight_bytes_avoided=packed_weight_bytes_avoided,
            ridge=ridge_value,
        )

    def coverage_distance(self, mlp_input: torch.Tensor) -> float:
        feature, _device = _k1_hidden(
            mlp_input,
            hidden_dim=self.identity.hidden_dim,
        )
        sketch = feature @ self.identity.projection.matrix()
        return float(torch.linalg.vector_norm(sketch - self._coverage.center).item())

    def eligible(
        self,
        mlp_input: torch.Tensor,
        *,
        max_error_radius: float,
    ) -> bool:
        allowed = _finite_non_negative(max_error_radius, "max_error_radius")
        return (
            self.coverage_distance(mlp_input) <= self._coverage.feature_radius
            and self._coverage.error_radius <= allowed
        )

    def apply(
        self,
        base_hidden: torch.Tensor,
        mlp_input: torch.Tensor,
        *,
        max_error_radius: float,
    ) -> "Layer63MlpResidualReplacement | None":
        allowed = _finite_non_negative(max_error_radius, "max_error_radius")
        base, device = _k1_hidden(
            base_hidden,
            hidden_dim=self.identity.hidden_dim,
        )
        feature, _feature_device = _k1_hidden(
            mlp_input,
            hidden_dim=self.identity.hidden_dim,
        )
        sketch = feature @ self.identity.projection.matrix()
        distance = float(
            torch.linalg.vector_norm(sketch - self._coverage.center).item()
        )
        return self._apply_precomputed(
            base,
            device=device,
            sketch=sketch,
            distance=distance,
            allowed=allowed,
        )

    def _apply_precomputed(
        self,
        base: torch.Tensor,
        *,
        device: torch.device,
        sketch: torch.Tensor,
        distance: float,
        allowed: float,
    ) -> "Layer63MlpResidualReplacement | None":
        if (
            distance > self._coverage.feature_radius
            or self._coverage.error_radius > allowed
        ):
            return None
        predicted = (
            base + self._residual_mean + (sketch - self._feature_mean) @ self._operator
        )
        if not bool(torch.isfinite(predicted).all().item()):
            raise LayerMlpCrystalIntegrityError(
                "MLP residual action produced a non-finite hidden row"
            )
        output = predicted.reshape(1, 1, self.identity.hidden_dim).to(
            device=device,
            dtype=torch.bfloat16,
        )
        if not bool(torch.isfinite(output).all().item()):
            raise LayerMlpCrystalIntegrityError(
                "BF16 MLP residual action produced a non-finite hidden row"
            )
        replacement_cls = (
            LayerMlpResidualReplacement
            if _generic_identity(self.identity)
            else Layer63MlpResidualReplacement
        )
        return replacement_cls(
            output=output,
            crystal_sha256=self.crystal_sha256,
            coverage_distance=distance,
            coverage_radius=self._coverage.feature_radius,
            error_radius=self._coverage.error_radius,
            logical_weight_bytes_replaced=self.packed_weight_bytes_avoided,
        )


@dataclass(frozen=True, slots=True)
class Layer63MlpResidualReplacement:
    """One successful physical gate/up/down replacement."""

    output: torch.Tensor
    crystal_sha256: str
    coverage_distance: float
    coverage_radius: float
    error_radius: float
    logical_weight_bytes_replaced: int

    def __post_init__(self) -> None:
        _digest(self.crystal_sha256, "crystal_sha256")
        if not isinstance(self.output, torch.Tensor):
            raise TypeError("replacement output must be a torch.Tensor")
        if self.output.dtype != torch.bfloat16 or self.output.ndim != 3:
            raise ValueError("replacement output must be a rank-three BF16 tensor")
        _finite_non_negative(self.coverage_distance, "coverage_distance")
        _finite_non_negative(self.coverage_radius, "coverage_radius")
        _finite_non_negative(self.error_radius, "error_radius")
        _uint(
            self.logical_weight_bytes_replaced,
            field="logical_weight_bytes_replaced",
            positive=True,
        )

    @property
    def hidden(self) -> torch.Tensor:
        return self.output

    @property
    def packed_weight_bytes_avoided(self) -> int:
        return self.logical_weight_bytes_replaced


@dataclass(frozen=True, slots=True)
class Layer63MlpResidualCrystalMetrics:
    identity_sha256: str
    crystal_count: int
    attempts: int
    replacements: int
    fallbacks: int
    logical_weight_bytes_replaced: int
    output_bytes_emitted: int

    def __post_init__(self) -> None:
        _digest(self.identity_sha256, "identity_sha256")
        for field in (
            "crystal_count",
            "attempts",
            "replacements",
            "fallbacks",
            "logical_weight_bytes_replaced",
            "output_bytes_emitted",
        ):
            _uint(getattr(self, field), field=field)
        if self.replacements + self.fallbacks != self.attempts:
            raise ValueError("attempt metrics do not settle exactly once")

    @property
    def bytes_replaced(self) -> int:
        return self.logical_weight_bytes_replaced

    @property
    def packed_weight_bytes_avoided(self) -> int:
        return self.logical_weight_bytes_replaced

    def to_dict(self) -> dict[str, object]:
        return {
            "attempts": self.attempts,
            "bytes_replaced": self.logical_weight_bytes_replaced,
            "crystal_count": self.crystal_count,
            "fallbacks": self.fallbacks,
            "identity_sha256": self.identity_sha256,
            "logical_weight_bytes_replaced": self.logical_weight_bytes_replaced,
            "output_bytes_emitted": self.output_bytes_emitted,
            "packed_weight_bytes_avoided": self.logical_weight_bytes_replaced,
            "replacements": self.replacements,
        }


class LayerMlpResidualCrystal(Layer63MlpResidualCrystal):
    """Layer-parametric v2 centered-ridge MLP residual action."""

    __slots__ = ()

    def __init__(
        self,
        *,
        identity: LayerMlpResidualCrystalIdentity,
        operator: torch.Tensor,
        feature_mean: torch.Tensor,
        residual_mean: torch.Tensor,
        coverage: LayerMlpResidualCoverage,
        packed_weight_bytes_avoided: int,
        ridge: float,
        source_o1_state_sha256: str | None = None,
        source_o1_generation: int | None = None,
    ) -> None:
        if not isinstance(identity, LayerMlpResidualCrystalIdentity):
            raise TypeError("identity must be a LayerMlpResidualCrystalIdentity")
        if not isinstance(coverage, LayerMlpResidualCoverage):
            raise TypeError("coverage must be a LayerMlpResidualCoverage")
        super().__init__(
            identity=identity,
            operator=operator,
            feature_mean=feature_mean,
            residual_mean=residual_mean,
            coverage=coverage,
            packed_weight_bytes_avoided=packed_weight_bytes_avoided,
            ridge=ridge,
            source_o1_state_sha256=source_o1_state_sha256,
            source_o1_generation=source_o1_generation,
        )


class LayerMlpResidualReplacement(Layer63MlpResidualReplacement):
    """Successful replacement emitted by a generic MLP crystal."""

    __slots__ = ()


class LayerMlpResidualCrystalMetrics(Layer63MlpResidualCrystalMetrics):
    """Settled metrics emitted by a generic MLP crystal bank."""

    __slots__ = ()


@dataclass(frozen=True, slots=True)
class _LayerMlpBankState:
    identity: _LayerMlpIdentity
    max_crystals: int
    generation: int = 0
    publications: int = 0
    crystals: tuple[Layer63MlpResidualCrystal, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(
            self.identity,
            (Layer63MlpResidualCrystalIdentity, LayerMlpResidualCrystalIdentity),
        ):
            raise TypeError("bank identity is invalid")
        _uint(
            self.max_crystals,
            field="max_crystals",
            positive=True,
            maximum=_MAX_CRYSTALS,
        )
        _uint(self.generation, field="generation")
        _uint(self.publications, field="publications")
        crystals = tuple(self.crystals)
        if len(crystals) > self.max_crystals:
            raise ValueError("bank exceeds max_crystals")
        if tuple(sorted(crystals, key=lambda item: item.crystal_sha256)) != crystals:
            raise ValueError("bank crystals are not canonically ordered")
        if len({item.crystal_sha256 for item in crystals}) != len(crystals):
            raise ValueError("bank contains duplicate crystals")
        if any(
            item.identity.identity_sha256 != self.identity.identity_sha256
            for item in crystals
        ):
            raise ValueError("bank contains a foreign crystal identity")
        object.__setattr__(self, "crystals", crystals)

    def to_record(self) -> dict[str, object]:
        return {
            "crystals": [crystal.to_record() for crystal in self.crystals],
            "generation": self.generation,
            "identity": self.identity.to_record(),
            "identity_sha256": self.identity.identity_sha256,
            "max_crystals": self.max_crystals,
            "publications": self.publications,
            "schema": (
                LAYER_MLP_RESIDUAL_GENERIC_BANK_SCHEMA
                if _generic_identity(self.identity)
                else LAYER_MLP_RESIDUAL_BANK_SCHEMA
            ),
        }

    def to_bytes(self) -> bytes:
        envelope_schema = (
            LAYER_MLP_RESIDUAL_GENERIC_BANK_ENVELOPE_SCHEMA
            if _generic_identity(self.identity)
            else LAYER_MLP_RESIDUAL_BANK_ENVELOPE_SCHEMA
        )
        encoded = _sealed_document(self.to_record(), envelope_schema)
        if len(encoded) > _MAX_STATE_BYTES:
            raise LayerMlpCrystalIntegrityError("MLP residual bank exceeds byte limit")
        return encoded

    @classmethod
    def from_bytes(cls, value: bytes) -> "_LayerMlpBankState":
        generic_envelope = False
        try:
            try:
                body = _unseal_document(
                    value,
                    schema=LAYER_MLP_RESIDUAL_BANK_ENVELOPE_SCHEMA,
                    kind="layer-63 MLP residual bank",
                )
            except LayerTransitionCrystalIntegrityError:
                generic_envelope = True
                body = _unseal_document(
                    value,
                    schema=LAYER_MLP_RESIDUAL_GENERIC_BANK_ENVELOPE_SCHEMA,
                    kind="layer-parametric MLP residual bank",
                )
        except LayerTransitionCrystalIntegrityError as exc:
            raise _as_integrity(exc) from exc
        fields = {
            "crystals",
            "generation",
            "identity",
            "identity_sha256",
            "max_crystals",
            "publications",
            "schema",
        }
        if set(body) != fields:
            raise LayerMlpCrystalIntegrityError("MLP residual bank fields are invalid")
        if body["schema"] not in (
            LAYER_MLP_RESIDUAL_BANK_SCHEMA,
            LAYER_MLP_RESIDUAL_GENERIC_BANK_SCHEMA,
        ):
            raise LayerMlpCrystalIntegrityError("MLP residual bank schema is invalid")
        if generic_envelope != (
            body["schema"] == LAYER_MLP_RESIDUAL_GENERIC_BANK_SCHEMA
        ):
            raise LayerMlpCrystalIntegrityError(
                "MLP bank envelope and body generation differ"
            )
        try:
            identity = _identity_from_record(body["identity"])
            if (
                body["schema"] == LAYER_MLP_RESIDUAL_GENERIC_BANK_SCHEMA
            ) != _generic_identity(identity):
                raise ValueError("MLP bank schema and identity generation differ")
            if body["identity_sha256"] != identity.identity_sha256:
                raise ValueError("bank identity SHA-256 mismatch")
            raw_crystals = body["crystals"]
            if isinstance(raw_crystals, (str, bytes, bytearray)) or not isinstance(
                raw_crystals,
                Sequence,
            ):
                raise TypeError("bank crystals are not a sequence")
            crystal_cls = (
                LayerMlpResidualCrystal
                if _generic_identity(identity)
                else Layer63MlpResidualCrystal
            )
            return cls(
                identity=identity,
                max_crystals=body["max_crystals"],
                generation=body["generation"],
                publications=body["publications"],
                crystals=tuple(crystal_cls.from_record(item) for item in raw_crystals),
            )
        except (TypeError, ValueError) as exc:
            raise LayerMlpCrystalIntegrityError(
                "MLP residual bank values are invalid"
            ) from exc


class Layer63MlpResidualCrystalBank:
    """Persistent bounded MLP residual bank with settled runtime metrics."""

    def __init__(
        self,
        state_path: str | os.PathLike[str],
        identity: _LayerMlpIdentity,
        *,
        max_crystals: int = 64,
    ) -> None:
        if not isinstance(
            identity,
            (Layer63MlpResidualCrystalIdentity, LayerMlpResidualCrystalIdentity),
        ):
            raise TypeError("identity must be an MLP residual crystal identity")
        self.state_path = Path(state_path)
        if not self.state_path.name:
            raise ValueError("state_path must name a file")
        _uint(
            max_crystals,
            field="max_crystals",
            positive=True,
            maximum=_MAX_CRYSTALS,
        )
        self.identity = identity
        self.max_crystals = max_crystals
        self._lock: threading.RLock = _thread_lock(self.state_path)
        self._file_signature: tuple[int, int, int, int, int] | None = None
        self._had_persistent_state = False
        self._state = _LayerMlpBankState(
            identity=identity,
            max_crystals=max_crystals,
        )
        self._attempts = 0
        self._replacements = 0
        self._fallbacks = 0
        self._logical_weight_bytes_replaced = 0
        self._output_bytes_emitted = 0
        with self._lock:
            self._reload(required=False)

    @classmethod
    def load(
        cls,
        state_path: str | os.PathLike[str],
    ) -> "Layer63MlpResidualCrystalBank":
        path = Path(state_path)
        raw, signature = _read_state(path)
        state = _LayerMlpBankState.from_bytes(raw)
        bank = cls.__new__(cls)
        bank.state_path = path
        bank.identity = state.identity
        bank.max_crystals = state.max_crystals
        bank._lock = _thread_lock(path)
        bank._file_signature = signature
        bank._had_persistent_state = True
        bank._state = state
        bank._attempts = 0
        bank._replacements = 0
        bank._fallbacks = 0
        bank._logical_weight_bytes_replaced = 0
        bank._output_bytes_emitted = 0
        return bank

    def _validate_state(self, state: _LayerMlpBankState) -> None:
        if state.identity.identity_sha256 != self.identity.identity_sha256:
            raise LayerMlpCrystalIdentityError(
                "MLP residual bank belongs to a different identity"
            )
        if state.max_crystals != self.max_crystals:
            raise LayerMlpCrystalIdentityError(
                "MLP residual bank max_crystals differs from persistent state"
            )

    def _reload(self, *, required: bool) -> None:
        try:
            raw, signature = _read_state(self.state_path)
        except FileNotFoundError:
            if required or self._had_persistent_state:
                raise LayerMlpCrystalIntegrityError("MLP residual bank disappeared")
            self._file_signature = None
            self._state = _LayerMlpBankState(
                identity=self.identity,
                max_crystals=self.max_crystals,
            )
            return
        state = _LayerMlpBankState.from_bytes(raw)
        self._validate_state(state)
        self._state = state
        self._file_signature = signature
        self._had_persistent_state = True

    def _refresh_if_changed(self) -> None:
        try:
            linked = os.lstat(self.state_path)
        except FileNotFoundError:
            if self._had_persistent_state:
                raise LayerMlpCrystalIntegrityError("MLP residual bank disappeared")
            return
        if stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode):
            raise LayerMlpCrystalIntegrityError(
                "MLP residual bank must be a regular file"
            )
        if _stable_signature(linked) != self._file_signature:
            self._reload(required=True)

    def publish(self, crystal: Layer63MlpResidualCrystal) -> str:
        if not isinstance(crystal, Layer63MlpResidualCrystal):
            raise TypeError("crystal must be a Layer63MlpResidualCrystal")
        if crystal.identity.identity_sha256 != self.identity.identity_sha256:
            raise LayerMlpCrystalIdentityError(
                "cannot publish a crystal from another identity"
            )
        with _state_lock(self.state_path):
            self._reload(required=self._had_persistent_state)
            if any(
                item.crystal_sha256 == crystal.crystal_sha256
                for item in self._state.crystals
            ):
                return crystal.crystal_sha256
            if len(self._state.crystals) >= self.max_crystals:
                raise LayerMlpCrystalCapacityError("MLP residual bank is full")
            next_state = _LayerMlpBankState(
                identity=self.identity,
                max_crystals=self.max_crystals,
                generation=_bounded_add(self._state.generation, 1),
                publications=_bounded_add(self._state.publications, 1),
                crystals=tuple(
                    sorted(
                        (*self._state.crystals, crystal),
                        key=lambda item: item.crystal_sha256,
                    )
                ),
            )
            _publish_bytes(self.state_path, next_state.to_bytes())
            self._state = next_state
            self._file_signature = _stable_signature(os.lstat(self.state_path))
            self._had_persistent_state = True
        return crystal.crystal_sha256

    def publish_latest(self, crystal: Layer63MlpResidualCrystal) -> str:
        """Atomically make one refitted O1 crystal the only current action."""

        if not isinstance(crystal, Layer63MlpResidualCrystal):
            raise TypeError("crystal must be a Layer63MlpResidualCrystal")
        if crystal.identity.identity_sha256 != self.identity.identity_sha256:
            raise LayerMlpCrystalIdentityError(
                "cannot publish a crystal from another identity"
            )
        if (
            crystal.source_o1_state_sha256 is None
            or crystal.source_o1_generation is None
        ):
            raise ValueError("publish_latest requires O1 state provenance")
        with _state_lock(self.state_path):
            self._reload(required=self._had_persistent_state)
            if (
                len(self._state.crystals) == 1
                and self._state.crystals[0].crystal_sha256 == crystal.crystal_sha256
            ):
                return crystal.crystal_sha256
            if len(self._state.crystals) == 1:
                current = self._state.crystals[0]
                current_generation = current.source_o1_generation
                if current_generation is not None:
                    if current_generation > crystal.source_o1_generation:
                        return current.crystal_sha256
                    if current_generation == crystal.source_o1_generation:
                        if (
                            current.source_o1_state_sha256
                            != crystal.source_o1_state_sha256
                        ):
                            raise LayerMlpCrystalIntegrityError(
                                "equal O1 generations name different states"
                            )
                        raise LayerMlpCrystalIntegrityError(
                            "one O1 state produced two different crystals"
                        )
            next_state = _LayerMlpBankState(
                identity=self.identity,
                max_crystals=self.max_crystals,
                generation=_bounded_add(self._state.generation, 1),
                publications=_bounded_add(self._state.publications, 1),
                crystals=(crystal,),
            )
            _publish_bytes(self.state_path, next_state.to_bytes())
            self._state = next_state
            self._file_signature = _stable_signature(os.lstat(self.state_path))
            self._had_persistent_state = True
        return crystal.crystal_sha256

    replace_current = publish_latest

    def replace(
        self,
        base_hidden: torch.Tensor,
        mlp_input: torch.Tensor,
        *,
        max_error_radius: float,
    ) -> Layer63MlpResidualReplacement | None:
        """Choose the tightest eligible action and settle one runtime attempt."""

        allowed = _finite_non_negative(max_error_radius, "max_error_radius")
        base, device = _k1_hidden(
            base_hidden,
            hidden_dim=self.identity.hidden_dim,
        )
        feature, _feature_device = _k1_hidden(
            mlp_input,
            hidden_dim=self.identity.hidden_dim,
        )
        sketch = feature @ self.identity.projection.matrix()
        with self._lock:
            self._refresh_if_changed()
            ranked: list[tuple[float, float, str, Layer63MlpResidualCrystal]] = []
            for crystal in self._state.crystals:
                distance = float(
                    torch.linalg.vector_norm(sketch - crystal._coverage.center).item()
                )
                if (
                    distance <= crystal._coverage.feature_radius
                    and crystal._coverage.error_radius <= allowed
                ):
                    ranked.append(
                        (
                            crystal._coverage.error_radius,
                            distance,
                            crystal.crystal_sha256,
                            crystal,
                        )
                    )
            self._attempts = _bounded_add(self._attempts, 1)
            if not ranked:
                self._fallbacks = _bounded_add(self._fallbacks, 1)
                return None
            selected_row = min(ranked)
            selected_distance = selected_row[1]
            selected = selected_row[-1]
            try:
                replacement = selected._apply_precomputed(
                    base,
                    device=device,
                    sketch=sketch,
                    distance=selected_distance,
                    allowed=allowed,
                )
            except Exception:
                self._fallbacks = _bounded_add(self._fallbacks, 1)
                raise
            if replacement is None:
                self._fallbacks = _bounded_add(self._fallbacks, 1)
                raise LayerMlpCrystalIntegrityError(
                    "eligible MLP crystal rejected the same feature row"
                )
            self._replacements = _bounded_add(self._replacements, 1)
            self._logical_weight_bytes_replaced = _bounded_add(
                self._logical_weight_bytes_replaced,
                replacement.logical_weight_bytes_replaced,
            )
            self._output_bytes_emitted = _bounded_add(
                self._output_bytes_emitted,
                replacement.output.numel() * replacement.output.element_size(),
            )
            return replacement

    def try_replace(
        self,
        base_hidden: torch.Tensor,
        mlp_input: torch.Tensor,
        *,
        max_error_radius: float,
    ) -> Layer63MlpResidualReplacement | None:
        return self.replace(
            base_hidden,
            mlp_input,
            max_error_radius=max_error_radius,
        )

    def metrics(self) -> Layer63MlpResidualCrystalMetrics:
        with self._lock:
            self._refresh_if_changed()
            metrics_cls = (
                LayerMlpResidualCrystalMetrics
                if _generic_identity(self.identity)
                else Layer63MlpResidualCrystalMetrics
            )
            return metrics_cls(
                identity_sha256=self.identity.identity_sha256,
                crystal_count=len(self._state.crystals),
                attempts=self._attempts,
                replacements=self._replacements,
                fallbacks=self._fallbacks,
                logical_weight_bytes_replaced=(self._logical_weight_bytes_replaced),
                output_bytes_emitted=self._output_bytes_emitted,
            )

    @property
    def crystals(self) -> tuple[Layer63MlpResidualCrystal, ...]:
        with self._lock:
            self._refresh_if_changed()
            return tuple(self._state.crystals)


class LayerMlpResidualCrystalBank(Layer63MlpResidualCrystalBank):
    """Persistent bounded bank for a layer-parametric v2 identity."""

    def __init__(
        self,
        state_path: str | os.PathLike[str],
        identity: LayerMlpResidualCrystalIdentity,
        *,
        max_crystals: int = 64,
    ) -> None:
        if not isinstance(identity, LayerMlpResidualCrystalIdentity):
            raise TypeError("identity must be a LayerMlpResidualCrystalIdentity")
        super().__init__(state_path, identity, max_crystals=max_crystals)

    @classmethod
    def load(
        cls,
        state_path: str | os.PathLike[str],
    ) -> "LayerMlpResidualCrystalBank":
        bank = super().load(state_path)
        if not isinstance(bank.identity, LayerMlpResidualCrystalIdentity):
            raise LayerMlpCrystalIdentityError(
                "generic MLP bank loader rejected a layer-63 v1 identity"
            )
        return bank


__all__ = [
    "FEATURE_STAGE",
    "LAYER_MLP_RESIDUAL_ACTION_ABI",
    "LAYER_MLP_RESIDUAL_GENERIC_ACTION_ABI",
    "LAYER_MLP_RESIDUAL_GENERIC_BANK_ENVELOPE_SCHEMA",
    "LAYER_MLP_RESIDUAL_GENERIC_BANK_SCHEMA",
    "LAYER_MLP_RESIDUAL_GENERIC_COVERAGE_SCHEMA",
    "LAYER_MLP_RESIDUAL_GENERIC_CRYSTAL_ENVELOPE_SCHEMA",
    "LAYER_MLP_RESIDUAL_GENERIC_CRYSTAL_SCHEMA",
    "LAYER_MLP_RESIDUAL_GENERIC_IDENTITY_SCHEMA",
    "LAYER_MLP_RESIDUAL_GENERIC_O1_CRYSTAL_SCHEMA",
    "LAYER_MLP_RESIDUAL_O1_CRYSTAL_SCHEMA",
    "Layer63MlpResidualCoverage",
    "Layer63MlpResidualCrystal",
    "Layer63MlpResidualCrystalBank",
    "Layer63MlpResidualCrystalIdentity",
    "Layer63MlpResidualCrystalMetrics",
    "Layer63MlpResidualReplacement",
    "LayerMlpResidualCoverage",
    "LayerMlpResidualCrystal",
    "LayerMlpResidualCrystalBank",
    "LayerMlpResidualCrystalIdentity",
    "LayerMlpResidualCrystalMetrics",
    "LayerMlpResidualReplacement",
    "LayerMlpCrystalCapacityError",
    "LayerMlpCrystalError",
    "LayerMlpCrystalIdentityError",
    "LayerMlpCrystalIntegrityError",
    "SOURCE_STAGE",
    "MAX_LAYER_INDEX",
    "TARGET_LAYER_INDEX",
    "TARGET_STAGE",
]
