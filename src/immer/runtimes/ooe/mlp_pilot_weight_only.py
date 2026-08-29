"""Weight-only cold starts and bounded online confirmation for Fast-MLP.

The original p4/k32 router is fitted from prompt activations.  This module
provides a separate, explicitly non-evidentiary initializer for layers that do
not have such a fit.  It derives one isotropic-Gaussian joint second moment per
Gate/Up neuron, chooses four deterministic stratified representatives per
64-neuron block, and keeps the existing sparse executor ABI.

Ordinary exact MLP executions can then call :meth:`observe_full`.  Only fixed
size sufficient statistics are retained; no prompt text, token IDs, hidden
states, activations, or outputs are persisted.  A confidence decision requests
the unchanged full MLP whenever the route has not yet been confirmed, its
measured activation-energy capture is low, or a periodic confirmation is due.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import threading
from typing import Mapping, Sequence, cast

try:
    import fcntl
except ImportError:  # pragma: no cover - production targets are POSIX.
    fcntl = None  # type: ignore[assignment]

import numpy as np
import torch
import torch.nn.functional as F

from .identity import canonical_json_bytes, require_sha256
from .mlp_pilot_residual import MlpPilotAffineLayer
from .mlp_pilot_router import (
    MlpPilotLayerModel,
    MlpPilotRouterConfig,
    _random_offsets,
)

WEIGHT_ONLY_PLAN_SCHEMA = "immer.qwen-mlp-pilot-weight-only-plan/v1"
WEIGHT_ONLY_AFFINE_SCHEMA = "immer.qwen-mlp-pilot-identity-affine/v1"
WEIGHT_ONLY_INITIALIZER = "isotropic-gaussian-joint-moment-stratified/v1"
PILOT_ONLINE_CONFIG_SCHEMA = "immer.qwen-mlp-pilot-online-config/v1"
PILOT_ADAPTIVE_WIDTH_CONFIG_SCHEMA = "immer.qwen-mlp-pilot-adaptive-width-config/v1"
LEGACY_PILOT_ONLINE_STATE_SCHEMA = "immer.qwen-mlp-pilot-online-state/v1"
V2_PILOT_ONLINE_STATE_SCHEMA = "immer.qwen-mlp-pilot-online-state/v2"
PILOT_ONLINE_STATE_SCHEMA = "immer.qwen-mlp-pilot-online-state/v3"
PILOT_SPARSE_DECISION_SCHEMA = "immer.qwen-mlp-pilot-sparse-decision/v2"
PILOT_ONLINE_OBSERVATION_SCHEMA = "immer.qwen-mlp-pilot-online-observation/v2"
_MAX_PLAN_BYTES = 64 * 1024 * 1024
_MAX_STATE_BYTES = 16 * 1024 * 1024


class MlpPilotWeightOnlyError(RuntimeError):
    """A weight-only plan or its online state violates the runtime ABI."""


class MlpPilotOnlineStateError(MlpPilotWeightOnlyError):
    """The mutable online confirmation state is corrupt or unavailable."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_document(
    data: bytes,
    *,
    schema: str,
    label: str,
    maximum: int,
) -> Mapping[str, object]:
    if not isinstance(data, bytes) or not 0 < len(data) <= maximum:
        raise MlpPilotWeightOnlyError(f"{label} exceeds its byte bound")

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MlpPilotWeightOnlyError(f"{label} is not canonical JSON") from exc
    if (
        not isinstance(document, Mapping)
        or set(document) != {"body", "body_sha256", "schema"}
        or document.get("schema") != schema
        or not isinstance(document.get("body"), Mapping)
        or document.get("body_sha256") != _digest(document.get("body"))
        or canonical_json_bytes(document) != data
    ):
        raise MlpPilotWeightOnlyError(f"{label} seal is invalid")
    return cast(Mapping[str, object], document)


def _sealed(schema: str, body: Mapping[str, object]) -> bytes:
    normalized = json.loads(canonical_json_bytes(dict(body)))
    return canonical_json_bytes(
        {"body": normalized, "body_sha256": _digest(normalized), "schema": schema}
    )


def _finite_probability(value: object, *, field: str, inclusive_zero: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    lower_ok = result >= 0.0 if inclusive_zero else result > 0.0
    if not math.isfinite(result) or not lower_ok or result > 1.0:
        raise ValueError(f"{field} must be a finite probability")
    return result


@dataclass(frozen=True, slots=True)
class MlpPilotOnlineConfig:
    """Bounded confidence and recursive-ridge policy for one mounted plan."""

    min_confirmed_rows: int = 8
    min_capture: float = 0.25
    confirmation_interval: int = 32
    cold_start_sparse_waves: int = 0
    statistics_decay: float = 0.99
    confidence_decay: float = 0.80
    ridge: float = 1e-6
    prior_strength: float = 1e-3
    max_confirmed_rows: int = 1_000_000
    max_sparse_rows: int = 16

    def __post_init__(self) -> None:
        for field in (
            "min_confirmed_rows",
            "confirmation_interval",
            "max_confirmed_rows",
            "max_sparse_rows",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if (
            isinstance(self.cold_start_sparse_waves, bool)
            or not isinstance(self.cold_start_sparse_waves, int)
            or self.cold_start_sparse_waves < 0
        ):
            raise ValueError("cold_start_sparse_waves must be non-negative")
        object.__setattr__(
            self,
            "min_capture",
            _finite_probability(
                self.min_capture, field="min_capture", inclusive_zero=True
            ),
        )
        for field in ("statistics_decay", "confidence_decay"):
            value = _finite_probability(
                getattr(self, field), field=field, inclusive_zero=False
            )
            if value >= 1.0:
                raise ValueError(f"{field} must be smaller than one")
            object.__setattr__(self, field, value)
        for field in ("ridge", "prior_strength"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{field} must be finite and positive")
            object.__setattr__(self, field, float(value))
        if self.min_confirmed_rows > self.max_confirmed_rows:
            raise ValueError("min_confirmed_rows exceeds max_confirmed_rows")

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "cold_start_sparse_waves": self.cold_start_sparse_waves,
            "confidence_decay": self.confidence_decay,
            "confirmation_interval": self.confirmation_interval,
            "max_confirmed_rows": self.max_confirmed_rows,
            "max_sparse_rows": self.max_sparse_rows,
            "min_capture": self.min_capture,
            "min_confirmed_rows": self.min_confirmed_rows,
            "prior_strength": self.prior_strength,
            "ridge": self.ridge,
            "schema": PILOT_ONLINE_CONFIG_SCHEMA,
            "statistics_decay": self.statistics_decay,
        }

    @classmethod
    def from_record(cls, value: object) -> "MlpPilotOnlineConfig":
        expected = {
            "cold_start_sparse_waves",
            "confidence_decay",
            "confirmation_interval",
            "max_confirmed_rows",
            "max_sparse_rows",
            "min_capture",
            "min_confirmed_rows",
            "prior_strength",
            "ridge",
            "schema",
            "statistics_decay",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != PILOT_ONLINE_CONFIG_SCHEMA
        ):
            raise MlpPilotWeightOnlyError("online pilot config is invalid")
        try:
            return cls(
                min_confirmed_rows=cast(int, value["min_confirmed_rows"]),
                min_capture=cast(float, value["min_capture"]),
                confirmation_interval=cast(int, value["confirmation_interval"]),
                cold_start_sparse_waves=cast(int, value["cold_start_sparse_waves"]),
                statistics_decay=cast(float, value["statistics_decay"]),
                confidence_decay=cast(float, value["confidence_decay"]),
                ridge=cast(float, value["ridge"]),
                prior_strength=cast(float, value["prior_strength"]),
                max_confirmed_rows=cast(int, value["max_confirmed_rows"]),
                max_sparse_rows=cast(int, value["max_sparse_rows"]),
            )
        except (TypeError, ValueError) as exc:
            raise MlpPilotWeightOnlyError(
                "online pilot config validation failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class MlpPilotAdaptiveWidthConfig:
    """Bound the dynamic p4 route width without changing bank payloads."""

    max_selected_block_count: int = 128
    selected_block_step: int = 8
    score_mass_margin: float = 0.02
    target_capture: float = 0.50
    min_output_cosine: float = 0.999
    max_output_relative_l2: float = 0.05
    min_output_confirmed_rows: int = 8
    output_metric_window_rows: int = 8
    max_shadow_rows: int = 1
    max_sparse_transport_fraction: float = 0.90

    def __post_init__(self) -> None:
        for field in (
            "max_selected_block_count",
            "selected_block_step",
            "min_output_confirmed_rows",
            "output_metric_window_rows",
            "max_shadow_rows",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        margin = _finite_probability(
            self.score_mass_margin,
            field="score_mass_margin",
            inclusive_zero=True,
        )
        if margin >= 1.0:
            raise ValueError("score_mass_margin must be smaller than one")
        object.__setattr__(self, "score_mass_margin", margin)
        object.__setattr__(
            self,
            "target_capture",
            _finite_probability(
                self.target_capture,
                field="target_capture",
                inclusive_zero=False,
            ),
        )
        object.__setattr__(
            self,
            "max_sparse_transport_fraction",
            _finite_probability(
                self.max_sparse_transport_fraction,
                field="max_sparse_transport_fraction",
                inclusive_zero=False,
            ),
        )
        object.__setattr__(
            self,
            "min_output_cosine",
            _finite_probability(
                self.min_output_cosine,
                field="min_output_cosine",
                inclusive_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "max_output_relative_l2",
            _finite_probability(
                self.max_output_relative_l2,
                field="max_output_relative_l2",
                inclusive_zero=True,
            ),
        )
        if self.min_output_confirmed_rows > self.output_metric_window_rows:
            raise ValueError(
                "min_output_confirmed_rows exceeds output_metric_window_rows"
            )

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "max_selected_block_count": self.max_selected_block_count,
            "schema": PILOT_ADAPTIVE_WIDTH_CONFIG_SCHEMA,
            "score_mass_margin": self.score_mass_margin,
            "selected_block_step": self.selected_block_step,
            "target_capture": self.target_capture,
            "min_output_cosine": self.min_output_cosine,
            "max_output_relative_l2": self.max_output_relative_l2,
            "min_output_confirmed_rows": self.min_output_confirmed_rows,
            "output_metric_window_rows": self.output_metric_window_rows,
            "max_shadow_rows": self.max_shadow_rows,
            "max_sparse_transport_fraction": self.max_sparse_transport_fraction,
        }

    def to_bytes(self) -> bytes:
        return _sealed(PILOT_ADAPTIVE_WIDTH_CONFIG_SCHEMA, self.to_record())

    @classmethod
    def from_bytes(cls, data: bytes) -> "MlpPilotAdaptiveWidthConfig":
        document = _strict_document(
            data,
            schema=PILOT_ADAPTIVE_WIDTH_CONFIG_SCHEMA,
            label="adaptive width config",
            maximum=64 * 1024,
        )
        body = cast(Mapping[str, object], document["body"])
        expected = {
            "max_selected_block_count",
            "schema",
            "score_mass_margin",
            "selected_block_step",
            "target_capture",
            "min_output_cosine",
            "max_output_relative_l2",
            "min_output_confirmed_rows",
            "output_metric_window_rows",
            "max_shadow_rows",
            "max_sparse_transport_fraction",
        }
        if (
            set(body) != expected
            or body.get("schema") != PILOT_ADAPTIVE_WIDTH_CONFIG_SCHEMA
        ):
            raise MlpPilotWeightOnlyError("adaptive width config is invalid")
        try:
            result = cls(
                max_selected_block_count=cast(int, body["max_selected_block_count"]),
                selected_block_step=cast(int, body["selected_block_step"]),
                score_mass_margin=cast(float, body["score_mass_margin"]),
                target_capture=cast(float, body["target_capture"]),
                min_output_cosine=cast(float, body["min_output_cosine"]),
                max_output_relative_l2=cast(float, body["max_output_relative_l2"]),
                min_output_confirmed_rows=cast(int, body["min_output_confirmed_rows"]),
                output_metric_window_rows=cast(int, body["output_metric_window_rows"]),
                max_shadow_rows=cast(int, body["max_shadow_rows"]),
                max_sparse_transport_fraction=cast(
                    float, body["max_sparse_transport_fraction"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise MlpPilotWeightOnlyError(
                "adaptive width config validation failed"
            ) from exc
        if result.to_bytes() != data:
            raise MlpPilotWeightOnlyError(
                "adaptive width config reconstruction changed"
            )
        return result


@dataclass(frozen=True, slots=True)
class MlpPilotIdentityAffinePlan:
    """Identity correction used until a separately confirmed residual exists."""

    model_pin_sha256: str
    router_fit_sha256: str
    weights_index_sha256: str
    models: tuple[MlpPilotAffineLayer, ...]

    def __post_init__(self) -> None:
        for field in (
            "model_pin_sha256",
            "router_fit_sha256",
            "weights_index_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        models = tuple(self.models)
        if (
            not models
            or tuple(row.layer for row in models) != tuple(range(len(models)))
            or any(
                not np.array_equal(row.scale, np.ones(row.output_dimension))
                for row in models
            )
            or any(
                not np.array_equal(row.bias, np.zeros(row.output_dimension))
                for row in models
            )
        ):
            raise ValueError("identity affine inventory is invalid")
        object.__setattr__(self, "models", models)

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "model_pin_sha256": self.model_pin_sha256,
            "models": [row.to_record() for row in self.models],
            "router_fit_sha256": self.router_fit_sha256,
            "schema": WEIGHT_ONLY_AFFINE_SCHEMA,
            "weights_index_sha256": self.weights_index_sha256,
        }


@dataclass(frozen=True, slots=True)
class MlpPilotWeightOnlyPlan:
    """All-layer p4/k32 plan derived only from authenticated local weights."""

    repo_id: str
    revision: str
    model_pin_sha256: str
    bundle_manifest_sha256: str
    layout_fingerprint: str
    weights_index_sha256: str
    hidden_dimension: int
    n_layers: int
    config: MlpPilotRouterConfig
    online_config: MlpPilotOnlineConfig
    models: tuple[MlpPilotLayerModel, ...]
    source_layer_sha256s: tuple[tuple[int, str], ...]
    initializer: str = WEIGHT_ONLY_INITIALIZER

    def __post_init__(self) -> None:
        if (
            not isinstance(self.repo_id, str)
            or not self.repo_id
            or not isinstance(self.revision, str)
            or not self.revision
        ):
            raise ValueError("weight-only logical model identity is invalid")
        for field in (
            "model_pin_sha256",
            "bundle_manifest_sha256",
            "layout_fingerprint",
            "weights_index_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        expected_model_pin = _digest(
            {
                "bundle_manifest_sha256": self.bundle_manifest_sha256,
                "layout_fingerprint": self.layout_fingerprint,
                "repo_id": self.repo_id,
                "revision": self.revision,
                "schema": "immer.qwen-mlp-pilot-weight-only-model-pin/v1",
                "weights_index_sha256": self.weights_index_sha256,
            }
        )
        if self.model_pin_sha256 != expected_model_pin:
            raise ValueError("weight-only model pin differs from its source identity")
        for field in ("hidden_dimension", "n_layers"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be positive")
        if not isinstance(self.config, MlpPilotRouterConfig):
            raise TypeError("config must be MlpPilotRouterConfig")
        if not isinstance(self.online_config, MlpPilotOnlineConfig):
            raise TypeError("online_config must be MlpPilotOnlineConfig")
        if self.initializer != WEIGHT_ONLY_INITIALIZER:
            raise ValueError("weight-only initializer ABI changed")
        models = tuple(self.models)
        if (
            len(models) != self.n_layers
            or tuple(row.layer for row in models) != tuple(range(self.n_layers))
            or any(row.block_size != self.config.block_size for row in models)
            or any(row.pilot_count != self.config.pilot_count for row in models)
            or any(
                row.selected_block_count != self.config.selected_block_count
                for row in models
            )
        ):
            raise ValueError("weight-only plan must cover every layer exactly once")
        sources = tuple(self.source_layer_sha256s)
        if tuple(layer for layer, _sha in sources) != tuple(
            range(self.n_layers)
        ) or any(
            require_sha256(sha, field="source_layer_sha256") != sha
            for _, sha in sources
        ):
            raise ValueError("weight-only source layer inventory is invalid")
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "source_layer_sha256s", sources)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @property
    def affine_fit(self) -> MlpPilotIdentityAffinePlan:
        plan_sha256 = self.sha256
        models = tuple(
            MlpPilotAffineLayer(
                layer=layer,
                output_dimension=self.hidden_dimension,
                scale=np.ones(self.hidden_dimension, dtype=np.float64),
                bias=np.zeros(self.hidden_dimension, dtype=np.float64),
                training_group_sha256s=(
                    _digest(
                        {
                            "initializer": "identity/no-output-claim/v1",
                            "layer": layer,
                            "weight_plan_sha256": plan_sha256,
                        }
                    ),
                ),
            )
            for layer in range(self.n_layers)
        )
        return MlpPilotIdentityAffinePlan(
            model_pin_sha256=self.model_pin_sha256,
            router_fit_sha256=plan_sha256,
            weights_index_sha256=self.weights_index_sha256,
            models=models,
        )

    def to_document(self) -> dict[str, object]:
        body = {
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "config": self.config.to_record(),
            "hidden_dimension": self.hidden_dimension,
            "initializer": self.initializer,
            "layout_fingerprint": self.layout_fingerprint,
            "model_pin_sha256": self.model_pin_sha256,
            "models": [row.to_record() for row in self.models],
            "n_layers": self.n_layers,
            "online_config": self.online_config.to_record(),
            "repo_id": self.repo_id,
            "revision": self.revision,
            "source_layer_sha256s": [
                {"layer": layer, "sha256": sha}
                for layer, sha in self.source_layer_sha256s
            ],
            "weights_index_sha256": self.weights_index_sha256,
        }
        normalized = json.loads(canonical_json_bytes(body))
        return {
            "body": normalized,
            "body_sha256": _digest(normalized),
            "schema": WEIGHT_ONLY_PLAN_SCHEMA,
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > _MAX_PLAN_BYTES:
            raise MlpPilotWeightOnlyError("weight-only plan exceeds its byte bound")
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> "MlpPilotWeightOnlyPlan":
        document = _strict_document(
            data,
            schema=WEIGHT_ONLY_PLAN_SCHEMA,
            label="weight-only plan",
            maximum=_MAX_PLAN_BYTES,
        )
        body = cast(Mapping[str, object], document["body"])
        expected = {
            "bundle_manifest_sha256",
            "config",
            "hidden_dimension",
            "initializer",
            "layout_fingerprint",
            "model_pin_sha256",
            "models",
            "n_layers",
            "online_config",
            "repo_id",
            "revision",
            "source_layer_sha256s",
            "weights_index_sha256",
        }
        if (
            set(body) != expected
            or not isinstance(body.get("models"), list)
            or not isinstance(body.get("source_layer_sha256s"), list)
        ):
            raise MlpPilotWeightOnlyError("weight-only plan body is invalid")
        try:
            sources = []
            for row in cast(list[object], body["source_layer_sha256s"]):
                if not isinstance(row, Mapping) or set(row) != {"layer", "sha256"}:
                    raise ValueError("invalid source layer row")
                sources.append((cast(int, row["layer"]), cast(str, row["sha256"])))
            result = cls(
                repo_id=cast(str, body["repo_id"]),
                revision=cast(str, body["revision"]),
                model_pin_sha256=cast(str, body["model_pin_sha256"]),
                bundle_manifest_sha256=cast(str, body["bundle_manifest_sha256"]),
                layout_fingerprint=cast(str, body["layout_fingerprint"]),
                weights_index_sha256=cast(str, body["weights_index_sha256"]),
                hidden_dimension=cast(int, body["hidden_dimension"]),
                n_layers=cast(int, body["n_layers"]),
                config=MlpPilotRouterConfig.from_record(body["config"]),
                online_config=MlpPilotOnlineConfig.from_record(body["online_config"]),
                models=tuple(
                    MlpPilotLayerModel.from_record(row)
                    for row in cast(list[object], body["models"])
                ),
                source_layer_sha256s=tuple(sources),
                initializer=cast(str, body["initializer"]),
            )
        except (TypeError, ValueError) as exc:
            raise MlpPilotWeightOnlyError("weight-only plan validation failed") from exc
        if result.to_bytes() != data:
            raise MlpPilotWeightOnlyError("weight-only plan reconstruction changed")
        return result


def weight_only_model_pin(
    *,
    repo_id: str,
    revision: str,
    bundle_manifest_sha256: str,
    layout_fingerprint: str,
    weights_index_sha256: str,
) -> str:
    """Return the code-independent identity used by a weight-only plan."""

    return _digest(
        {
            "bundle_manifest_sha256": require_sha256(
                bundle_manifest_sha256, field="bundle_manifest_sha256"
            ),
            "layout_fingerprint": require_sha256(
                layout_fingerprint, field="layout_fingerprint"
            ),
            "repo_id": repo_id,
            "revision": revision,
            "schema": "immer.qwen-mlp-pilot-weight-only-model-pin/v1",
            "weights_index_sha256": require_sha256(
                weights_index_sha256, field="weights_index_sha256"
            ),
        }
    )


def gaussian_joint_moments(
    gate_weight: torch.Tensor | np.ndarray,
    up_weight: torch.Tensor | np.ndarray,
) -> np.ndarray:
    """Compute ``E[g²u²]`` under an isotropic Gaussian hidden vector.

    The common hidden-dimension scale is retained only to keep values bounded;
    it does not affect stable ranking.  NumPy reductions use float64 and a
    fixed row order so a local rebuild is deterministic.
    """

    if isinstance(gate_weight, torch.Tensor):
        gate = gate_weight.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        gate = np.asarray(gate_weight)
    if isinstance(up_weight, torch.Tensor):
        up = up_weight.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        up = np.asarray(up_weight)
    if (
        gate.ndim != 2
        or gate.shape != up.shape
        or gate.shape[0] < 1
        or gate.shape[1] < 1
        or not np.issubdtype(gate.dtype, np.floating)
        or not np.issubdtype(up.dtype, np.floating)
    ):
        raise ValueError("Gate/Up weights differ from the weight initializer ABI")
    gate64 = np.asarray(gate, dtype=np.float64)
    up64 = np.asarray(up, dtype=np.float64)
    if not bool(np.isfinite(gate64).all()) or not bool(np.isfinite(up64).all()):
        raise ValueError("Gate/Up weights contain non-finite values")
    gate_norm = np.sum(np.square(gate64), axis=1, dtype=np.float64)
    up_norm = np.sum(np.square(up64), axis=1, dtype=np.float64)
    covariance = np.sum(gate64 * up64, axis=1, dtype=np.float64)
    dimension = float(gate64.shape[1])
    result = (gate_norm * up_norm + 2.0 * np.square(covariance)) / (
        dimension * dimension
    )
    result[result == 0.0] = 0.0
    if not bool(np.isfinite(result).all()) or bool((result < 0.0).any()):
        raise MlpPilotWeightOnlyError("weight-only joint moments are invalid")
    return result


def _stratified_pilots(moment: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    width = len(moment)
    if width <= count:
        raise ValueError("pilot count must leave non-pilot rows")
    order = np.argsort(moment, kind="stable")
    partitions = np.array_split(order, count)
    selected: list[int] = []
    coefficient_by_offset: dict[int, float] = {}
    for partition in partitions:
        values = moment[partition]
        mean = float(values.mean())
        distance = np.abs(values - mean)
        local = int(np.argmin(distance))
        offset = int(partition[local])
        selected.append(offset)
        denominator = float(moment[offset])
        coefficient_by_offset[offset] = (
            float(values.sum()) / denominator
            if denominator > np.finfo(np.float64).tiny
            else float(len(partition))
        )
    offsets = np.asarray(sorted(selected), dtype=np.int64)
    coefficients = np.asarray(
        [coefficient_by_offset[int(offset)] for offset in offsets],
        dtype=np.float64,
    )
    if (
        len(set(offsets.tolist())) != count
        or not bool(np.isfinite(coefficients).all())
        or bool((coefficients <= 0.0).any())
    ):
        raise MlpPilotWeightOnlyError("stratified pilot construction failed")
    return offsets, coefficients


def build_weight_only_layer_model(
    *,
    layer: int,
    moments: np.ndarray,
    config: MlpPilotRouterConfig,
    source_layer_sha256: str,
) -> MlpPilotLayerModel:
    """Build one existing layer-model ABI from block-local weight moments."""

    if not isinstance(config, MlpPilotRouterConfig):
        raise TypeError("config must be MlpPilotRouterConfig")
    values = np.asarray(moments)
    if (
        values.dtype != np.dtype(np.float64)
        or values.ndim != 2
        or values.shape[1] != config.block_size
        or not bool(np.isfinite(values).all())
        or bool((values < 0.0).any())
    ):
        raise ValueError("moments must be a finite block-by-neuron float64 matrix")
    block_count = values.shape[0]
    if config.selected_block_count >= block_count:
        raise ValueError("selected_block_count exhausts the weight-only layer")
    pilots: list[tuple[int, ...]] = []
    coefficients = np.zeros((block_count, config.pilot_count + 1), dtype=np.float64)
    random_pilots: list[tuple[int, ...]] = []
    random_coefficients = np.zeros_like(coefficients)
    for block in range(block_count):
        offsets, slopes = _stratified_pilots(values[block], config.pilot_count)
        pilots.append(tuple(int(row) for row in offsets))
        coefficients[block, 1:] = slopes
        controls = _random_offsets(
            config.random_seed_sha256,
            layer,
            block,
            config.pilot_count,
            config.block_size,
        )
        random_pilots.append(tuple(int(row) for row in controls))
        random_coefficients[block, 1:] = config.block_size / config.pilot_count
    return MlpPilotLayerModel(
        layer=layer,
        intermediate_dimension=block_count * config.block_size,
        block_size=config.block_size,
        selected_block_count=config.selected_block_count,
        pilot_offsets=tuple(pilots),
        coefficients=coefficients,
        random_pilot_offsets=tuple(random_pilots),
        random_coefficients=random_coefficients,
        marginal_block_scores=values.sum(axis=1, dtype=np.float64),
        training_group_sha256s=(
            require_sha256(source_layer_sha256, field="source_layer_sha256"),
        ),
    )


@dataclass(frozen=True, slots=True)
class MlpPilotSparseDecision:
    layer: int
    row_count: int
    use_sparse: bool
    reason: str
    confirmed_rows: int
    capture_ema: float
    sparse_waves_since_confirmation: int
    selected_block_count: int = 0
    max_selected_block_count: int = 0
    required_score_mass: float = 0.0

    def to_record(self) -> dict[str, object]:
        return {
            "capture_ema": self.capture_ema,
            "confirmed_rows": self.confirmed_rows,
            "layer": self.layer,
            "reason": self.reason,
            "row_count": self.row_count,
            "schema": PILOT_SPARSE_DECISION_SCHEMA,
            "selected_block_count": self.selected_block_count,
            "max_selected_block_count": self.max_selected_block_count,
            "required_score_mass": self.required_score_mass,
            "sparse_waves_since_confirmation": self.sparse_waves_since_confirmation,
            "use_sparse": self.use_sparse,
        }


@dataclass(frozen=True, slots=True)
class MlpPilotOnlineObservation:
    layer: int
    rows: int
    route_capture: float
    capture_ema: float
    confirmed_rows: int
    surprising: bool
    selected_block_count: int = 0
    predicted_score_mass: float = 0.0
    required_score_mass: float = 0.0
    shadow_row_indices: tuple[int, ...] = ()
    shadow_selected_blocks: tuple[tuple[int, ...], ...] = ()
    output_confirmed_rows: int = 0
    output_worst_cosine: float = 0.0
    output_worst_relative_l2: float = 1.0

    def to_record(self) -> dict[str, object]:
        return {
            "capture_ema": self.capture_ema,
            "confirmed_rows": self.confirmed_rows,
            "layer": self.layer,
            "route_capture": self.route_capture,
            "rows": self.rows,
            "schema": PILOT_ONLINE_OBSERVATION_SCHEMA,
            "selected_block_count": self.selected_block_count,
            "predicted_score_mass": self.predicted_score_mass,
            "required_score_mass": self.required_score_mass,
            "shadow_row_indices": list(self.shadow_row_indices),
            "shadow_selected_blocks": [
                list(row) for row in self.shadow_selected_blocks
            ],
            "output_confirmed_rows": self.output_confirmed_rows,
            "output_worst_cosine": self.output_worst_cosine,
            "output_worst_relative_l2": self.output_worst_relative_l2,
            "surprising": self.surprising,
        }


@dataclass(slots=True)
class _LayerState:
    gram: np.ndarray
    rhs: np.ndarray
    coefficients: np.ndarray
    confirmed_rows: int = 0
    exact_waves: int = 0
    sparse_waves: int = 0
    sparse_rows: int = 0
    sparse_since_confirmation: int = 0
    capture_ema: float = 0.0
    last_capture: float = 0.0
    surprises: int = 0
    selected_block_count: int = 0
    confirmed_capture_at_width: float = 0.0
    predicted_score_mass: float = 0.0
    required_score_mass: float = 0.0
    last_sparse_block_count: int = 0
    width_updates: int = 0
    output_element_count: int = 0
    output_sum_x: float = 0.0
    output_sum_y: float = 0.0
    output_sum_xx: float = 0.0
    output_sum_xy: float = 0.0
    output_scale: float = 1.0
    output_bias: float = 0.0
    output_cosines: tuple[float, ...] = ()
    output_relative_l2s: tuple[float, ...] = ()
    output_confirmed_rows: int = 0
    output_shadow_waves: int = 0
    output_calibration_block_count: int = 0


def _array_record(value: np.ndarray) -> dict[str, object]:
    normalized = np.asarray(value, dtype="<f8", order="C")
    raw = normalized.tobytes(order="C")
    return {
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "dtype": "float64-le",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(normalized.shape),
    }


def _array_from_record(
    value: object, *, shape: tuple[int, ...], field: str
) -> np.ndarray:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"data_base64", "dtype", "sha256", "shape"}
        or value.get("dtype") != "float64-le"
        or value.get("shape") != list(shape)
        or not isinstance(value.get("data_base64"), str)
    ):
        raise MlpPilotOnlineStateError(f"{field} array record is invalid")
    try:
        raw = base64.b64decode(cast(str, value["data_base64"]), validate=True)
    except (TypeError, ValueError) as exc:
        raise MlpPilotOnlineStateError(f"{field} array encoding is invalid") from exc
    if len(raw) != math.prod(shape) * 8 or hashlib.sha256(
        raw
    ).hexdigest() != require_sha256(value.get("sha256"), field=f"{field}.sha256"):
        raise MlpPilotOnlineStateError(f"{field} array payload is invalid")
    result = np.frombuffer(raw, dtype="<f8").reshape(shape).copy()
    if not bool(np.isfinite(result).all()):
        raise MlpPilotOnlineStateError(f"{field} contains non-finite values")
    return result


def _read_regular(path: Path, maximum: int) -> bytes:
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
            raise MlpPilotOnlineStateError("online state is not a bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise MlpPilotOnlineStateError("online state returned a short read")
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
            raise MlpPilotOnlineStateError("online state changed while read")
        return b"".join(chunks)
    except MlpPilotOnlineStateError:
        raise
    except OSError as exc:
        raise MlpPilotOnlineStateError("cannot read online state") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


class MlpPilotOnlineController:
    """Fixed-memory recursive router state with exact-path confidence gates."""

    def __init__(
        self,
        plan: MlpPilotWeightOnlyPlan,
        *,
        state_path: str | Path | None = None,
        width_config: MlpPilotAdaptiveWidthConfig = MlpPilotAdaptiveWidthConfig(),
    ) -> None:
        if not isinstance(plan, MlpPilotWeightOnlyPlan):
            raise TypeError("plan must be MlpPilotWeightOnlyPlan")
        self.plan = plan
        self.config = plan.online_config
        if not isinstance(width_config, MlpPilotAdaptiveWidthConfig):
            raise TypeError("width_config must be MlpPilotAdaptiveWidthConfig")
        self.width_config = width_config
        self._models = {row.layer: row for row in plan.models}
        self._max_blocks = {
            row.layer: min(
                width_config.max_selected_block_count,
                row.block_count - 1,
            )
            for row in plan.models
        }
        if any(
            self._max_blocks[row.layer] < row.selected_block_count
            for row in plan.models
        ):
            raise ValueError("adaptive width maximum is below the p4 plan width")
        self.width_policy_sha256 = _digest(
            {
                "maximum_by_layer": {
                    str(layer): count
                    for layer, count in sorted(self._max_blocks.items())
                },
                "minimum_by_layer": {
                    str(row.layer): row.selected_block_count for row in plan.models
                },
                "plan_sha256": plan.sha256,
                "schema": "immer.qwen-mlp-pilot-adaptive-width-policy/v1",
                "width_config": width_config.to_record(),
            }
        )
        self._lock = threading.RLock()
        self._closed = False
        self._dirty = False
        self._state_path = (
            None if state_path is None else Path(state_path).expanduser().absolute()
        )
        self._lock_descriptor: int | None = None
        self._states = {row.layer: self._initial_state(row) for row in plan.models}
        if self._state_path is not None:
            self._acquire_file_lock()
            try:
                if self._state_path.exists() or self._state_path.is_symlink():
                    self._load_state()
            except Exception:
                self._release_file_lock()
                raise

    def _initial_state(self, model: MlpPilotLayerModel) -> _LayerState:
        width = model.pilot_count + 1
        identity = np.eye(width, dtype=np.float64)
        gram = np.repeat(
            (identity * self.config.prior_strength)[None, :, :],
            model.block_count,
            axis=0,
        )
        coefficients = np.array(model.coefficients, dtype=np.float64, copy=True)
        rhs = np.einsum("bij,bj->bi", gram, coefficients)
        return _LayerState(
            gram=gram,
            rhs=rhs,
            coefficients=coefficients,
            selected_block_count=model.selected_block_count,
            last_sparse_block_count=model.selected_block_count,
            output_calibration_block_count=self._max_blocks[model.layer],
        )

    def _require_open(self) -> None:
        if self._closed:
            raise MlpPilotOnlineStateError("online pilot controller is closed")

    def _width_candidates(self, layer: int) -> tuple[int, ...]:
        model = self._models[layer]
        maximum = self._max_blocks[layer]
        candidates = list(
            range(
                model.selected_block_count,
                maximum + 1,
                self.width_config.selected_block_step,
            )
        )
        if candidates[-1] != maximum:
            candidates.append(maximum)
        return tuple(candidates)

    def selected_block_count(self, *, layer: int) -> int:
        """Return the latest exact-confirmed width for one sparse wave."""

        with self._lock:
            self._require_open()
            state = self._states.get(layer)
            if state is None:
                raise KeyError(f"no online pilot model for layer {layer}")
            output_ready = (
                state.output_confirmed_rows
                >= self.width_config.min_output_confirmed_rows
                and len(state.output_cosines)
                >= self.width_config.min_output_confirmed_rows
                and min(state.output_cosines) >= self.width_config.min_output_cosine
                and max(state.output_relative_l2s)
                <= self.width_config.max_output_relative_l2
            )
            return (
                state.output_calibration_block_count
                if output_ready
                else state.selected_block_count
            )

    def output_calibrated(self, *, layer: int) -> bool:
        """Return whether recent prequential output error opens target use."""

        with self._lock:
            self._require_open()
            state = self._states.get(layer)
            if state is None:
                raise KeyError(f"no online pilot model for layer {layer}")
            return (
                state.output_confirmed_rows
                >= self.width_config.min_output_confirmed_rows
                and len(state.output_cosines)
                >= self.width_config.min_output_confirmed_rows
                and min(state.output_cosines) >= self.width_config.min_output_cosine
                and max(state.output_relative_l2s)
                <= self.width_config.max_output_relative_l2
            )

    def max_selected_block_count(self, *, layer: int) -> int:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError("layer must be an integer")
        try:
            return self._max_blocks[layer]
        except KeyError as exc:
            raise KeyError(f"no online pilot model for layer {layer}") from exc

    def route_block_count(self, *, layer: int, scores: np.ndarray) -> int:
        """Widen a confirmed route until every row reaches its score-mass floor."""

        with self._lock:
            self._require_open()
            model = self._models.get(layer)
            state = self._states.get(layer)
            if model is None or state is None:
                raise KeyError(f"no online pilot model for layer {layer}")
            values = np.asarray(scores)
            if (
                values.dtype != np.dtype(np.float64)
                or values.ndim != 2
                or values.shape[1] != model.block_count
                or not bool(np.isfinite(values).all())
                or bool((values < 0.0).any())
            ):
                raise ValueError("adaptive route scores differ from the width ABI")
            required = state.required_score_mass
            if required <= 0.0:
                return state.selected_block_count
            ranked = np.sort(values, axis=1)[:, ::-1]
            cumulative = np.cumsum(ranked, axis=1, dtype=np.float64)
            totals = cumulative[:, -1]
            maximum = self._max_blocks[layer]
            selected = self.selected_block_count(layer=layer)
            for row in range(len(values)):
                if totals[row] <= np.finfo(np.float64).tiny:
                    selected = maximum
                    continue
                threshold = required * totals[row]
                reached = (
                    int(np.searchsorted(cumulative[row], threshold, side="left")) + 1
                )
                selected = max(selected, min(maximum, reached))
            return selected

    def _acquire_file_lock(self) -> None:
        assert self._state_path is not None
        if fcntl is None:  # pragma: no cover - production targets are POSIX.
            raise MlpPilotOnlineStateError("persistent online state requires fcntl")
        self._state_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        lock_path = self._state_path.with_name(f".{self._state_path.name}.lock")
        descriptor: int | None = None
        try:
            descriptor = os.open(
                lock_path,
                os.O_RDWR
                | os.O_CREAT
                | int(getattr(os, "O_CLOEXEC", 0))
                | int(getattr(os, "O_NOFOLLOW", 0)),
                0o600,
            )
            opened = os.fstat(descriptor)
            linked = lock_path.lstat()
            if (
                not stat.S_ISREG(opened.st_mode)
                or not stat.S_ISREG(linked.st_mode)
                or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
            ):
                raise MlpPilotOnlineStateError(
                    "online state lock must be an unchanged regular file"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            self._lock_descriptor = descriptor
        except MlpPilotOnlineStateError:
            if descriptor is not None:
                os.close(descriptor)
            raise
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            raise MlpPilotOnlineStateError("cannot lock online pilot state") from exc

    def _release_file_lock(self) -> None:
        descriptor = self._lock_descriptor
        self._lock_descriptor = None
        if descriptor is None:
            return
        assert fcntl is not None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    @staticmethod
    def _counter(value: object, *, field: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MlpPilotOnlineStateError(f"{field} counter is invalid")
        return value

    @staticmethod
    def _capture(value: object, *, field: str) -> float:
        try:
            return _finite_probability(value, field=field, inclusive_zero=True)
        except (TypeError, ValueError) as exc:
            raise MlpPilotOnlineStateError(f"{field} is invalid") from exc

    def _load_state(self) -> None:
        assert self._state_path is not None
        data = _read_regular(self._state_path, _MAX_STATE_BYTES)
        state_version = 3
        try:
            try:
                document = _strict_document(
                    data,
                    schema=PILOT_ONLINE_STATE_SCHEMA,
                    label="online pilot state",
                    maximum=_MAX_STATE_BYTES,
                )
            except MlpPilotWeightOnlyError:
                try:
                    document = _strict_document(
                        data,
                        schema=V2_PILOT_ONLINE_STATE_SCHEMA,
                        label="v2 online pilot state",
                        maximum=_MAX_STATE_BYTES,
                    )
                    state_version = 2
                except MlpPilotWeightOnlyError:
                    document = _strict_document(
                        data,
                        schema=LEGACY_PILOT_ONLINE_STATE_SCHEMA,
                        label="legacy online pilot state",
                        maximum=_MAX_STATE_BYTES,
                    )
                    state_version = 1
        except MlpPilotOnlineStateError:
            raise
        except MlpPilotWeightOnlyError as exc:
            raise MlpPilotOnlineStateError("online pilot state is corrupt") from exc
        body = cast(Mapping[str, object], document["body"])
        body_fields = (
            {"config_sha256", "layers", "plan_sha256"}
            if state_version == 1
            else {
                "config_sha256",
                "layers",
                "plan_sha256",
                "width_policy_sha256",
            }
        )
        if (
            set(body) != body_fields
            or body.get("plan_sha256") != self.plan.sha256
            or body.get("config_sha256") != self.config.sha256
            or (
                state_version >= 3
                and body.get("width_policy_sha256") != self.width_policy_sha256
            )
            or not isinstance(body.get("layers"), list)
        ):
            raise MlpPilotOnlineStateError("online pilot state identity changed")
        restored: dict[int, _LayerState] = {}
        for raw in cast(list[object], body["layers"]):
            legacy_fields = {
                "capture_ema",
                "coefficients",
                "confirmed_rows",
                "exact_waves",
                "gram",
                "last_capture",
                "layer",
                "rhs",
                "sparse_rows",
                "sparse_since_confirmation",
                "sparse_waves",
                "surprises",
            }
            expected = (
                legacy_fields
                if state_version == 1
                else legacy_fields
                | {
                    "confirmed_capture_at_width",
                    "last_sparse_block_count",
                    "predicted_score_mass",
                    "required_score_mass",
                    "selected_block_count",
                    "width_updates",
                }
            )
            if state_version >= 3:
                expected |= {
                    "output_bias",
                    "output_calibration_block_count",
                    "output_confirmed_rows",
                    "output_cosines",
                    "output_element_count",
                    "output_relative_l2s",
                    "output_scale",
                    "output_shadow_waves",
                    "output_sum_x",
                    "output_sum_xx",
                    "output_sum_xy",
                    "output_sum_y",
                }
            if not isinstance(raw, Mapping) or set(raw) != expected:
                raise MlpPilotOnlineStateError("online layer state is invalid")
            layer = raw.get("layer")
            if (
                isinstance(layer, bool)
                or not isinstance(layer, int)
                or layer not in self._models
            ):
                raise MlpPilotOnlineStateError("online layer identity is invalid")
            model = self._models[layer]
            width = model.pilot_count + 1
            state = _LayerState(
                gram=_array_from_record(
                    raw["gram"],
                    shape=(model.block_count, width, width),
                    field=f"layer {layer} gram",
                ),
                rhs=_array_from_record(
                    raw["rhs"],
                    shape=(model.block_count, width),
                    field=f"layer {layer} rhs",
                ),
                coefficients=_array_from_record(
                    raw["coefficients"],
                    shape=(model.block_count, width),
                    field=f"layer {layer} coefficients",
                ),
                confirmed_rows=self._counter(
                    raw["confirmed_rows"], field="confirmed_rows"
                ),
                exact_waves=self._counter(raw["exact_waves"], field="exact_waves"),
                sparse_waves=self._counter(raw["sparse_waves"], field="sparse_waves"),
                sparse_rows=self._counter(raw["sparse_rows"], field="sparse_rows"),
                sparse_since_confirmation=self._counter(
                    raw["sparse_since_confirmation"],
                    field="sparse_since_confirmation",
                ),
                capture_ema=self._capture(raw["capture_ema"], field="capture_ema"),
                last_capture=self._capture(raw["last_capture"], field="last_capture"),
                surprises=self._counter(raw["surprises"], field="surprises"),
                selected_block_count=(
                    model.selected_block_count
                    if state_version == 1
                    else self._counter(
                        raw["selected_block_count"], field="selected_block_count"
                    )
                ),
                confirmed_capture_at_width=(
                    self._capture(raw["capture_ema"], field="capture_ema")
                    if state_version == 1
                    else self._capture(
                        raw["confirmed_capture_at_width"],
                        field="confirmed_capture_at_width",
                    )
                ),
                predicted_score_mass=(
                    0.0
                    if state_version == 1
                    else self._capture(
                        raw["predicted_score_mass"], field="predicted_score_mass"
                    )
                ),
                required_score_mass=(
                    0.0
                    if state_version == 1
                    else self._capture(
                        raw["required_score_mass"], field="required_score_mass"
                    )
                ),
                last_sparse_block_count=(
                    model.selected_block_count
                    if state_version == 1
                    else self._counter(
                        raw["last_sparse_block_count"],
                        field="last_sparse_block_count",
                    )
                ),
                width_updates=(
                    0
                    if state_version == 1
                    else self._counter(raw["width_updates"], field="width_updates")
                ),
                output_element_count=(
                    0
                    if state_version < 3
                    else self._counter(
                        raw["output_element_count"], field="output_element_count"
                    )
                ),
                output_sum_x=(0.0 if state_version < 3 else float(raw["output_sum_x"])),
                output_sum_y=(0.0 if state_version < 3 else float(raw["output_sum_y"])),
                output_sum_xx=(
                    0.0 if state_version < 3 else float(raw["output_sum_xx"])
                ),
                output_sum_xy=(
                    0.0 if state_version < 3 else float(raw["output_sum_xy"])
                ),
                output_scale=(1.0 if state_version < 3 else float(raw["output_scale"])),
                output_bias=(0.0 if state_version < 3 else float(raw["output_bias"])),
                output_cosines=(
                    ()
                    if state_version < 3
                    else tuple(float(value) for value in raw["output_cosines"])
                ),
                output_relative_l2s=(
                    ()
                    if state_version < 3
                    else tuple(float(value) for value in raw["output_relative_l2s"])
                ),
                output_confirmed_rows=(
                    0
                    if state_version < 3
                    else self._counter(
                        raw["output_confirmed_rows"], field="output_confirmed_rows"
                    )
                ),
                output_shadow_waves=(
                    0
                    if state_version < 3
                    else self._counter(
                        raw["output_shadow_waves"], field="output_shadow_waves"
                    )
                ),
                output_calibration_block_count=(
                    self._max_blocks[layer]
                    if state_version < 3
                    else self._counter(
                        raw["output_calibration_block_count"],
                        field="output_calibration_block_count",
                    )
                ),
            )
            if state.confirmed_rows > self.config.max_confirmed_rows:
                raise MlpPilotOnlineStateError("confirmed row bound changed")
            if not (
                model.selected_block_count
                <= state.selected_block_count
                <= self._max_blocks[layer]
                and model.selected_block_count
                <= state.last_sparse_block_count
                <= self._max_blocks[layer]
            ):
                raise MlpPilotOnlineStateError("adaptive selected width is invalid")
            output_values = (
                state.output_sum_x,
                state.output_sum_y,
                state.output_sum_xx,
                state.output_sum_xy,
                state.output_scale,
                state.output_bias,
                *state.output_cosines,
                *state.output_relative_l2s,
            )
            if (
                any(not math.isfinite(value) for value in output_values)
                or any(not 0.0 <= value <= 1.0 for value in state.output_cosines)
                or any(value < 0.0 for value in state.output_relative_l2s)
                or len(state.output_cosines) != len(state.output_relative_l2s)
                or len(state.output_cosines)
                > self.width_config.output_metric_window_rows
                or state.output_confirmed_rows < len(state.output_cosines)
                or state.output_calibration_block_count != self._max_blocks[layer]
            ):
                raise MlpPilotOnlineStateError("output calibration state is invalid")
            restored[layer] = state
        if set(restored) != set(self._models):
            raise MlpPilotOnlineStateError("online state layer inventory changed")
        self._states = restored
        if state_version < 3:
            self._dirty = True

    def _state_bytes(self) -> bytes:
        layers = []
        for layer, state in sorted(self._states.items()):
            layers.append(
                {
                    "capture_ema": state.capture_ema,
                    "coefficients": _array_record(state.coefficients),
                    "confirmed_capture_at_width": state.confirmed_capture_at_width,
                    "confirmed_rows": state.confirmed_rows,
                    "exact_waves": state.exact_waves,
                    "gram": _array_record(state.gram),
                    "last_capture": state.last_capture,
                    "last_sparse_block_count": state.last_sparse_block_count,
                    "layer": layer,
                    "output_bias": state.output_bias,
                    "output_calibration_block_count": state.output_calibration_block_count,
                    "output_confirmed_rows": state.output_confirmed_rows,
                    "output_cosines": list(state.output_cosines),
                    "output_element_count": state.output_element_count,
                    "output_relative_l2s": list(state.output_relative_l2s),
                    "output_scale": state.output_scale,
                    "output_shadow_waves": state.output_shadow_waves,
                    "output_sum_x": state.output_sum_x,
                    "output_sum_xx": state.output_sum_xx,
                    "output_sum_xy": state.output_sum_xy,
                    "output_sum_y": state.output_sum_y,
                    "predicted_score_mass": state.predicted_score_mass,
                    "required_score_mass": state.required_score_mass,
                    "rhs": _array_record(state.rhs),
                    "sparse_rows": state.sparse_rows,
                    "sparse_since_confirmation": state.sparse_since_confirmation,
                    "sparse_waves": state.sparse_waves,
                    "surprises": state.surprises,
                    "selected_block_count": state.selected_block_count,
                    "width_updates": state.width_updates,
                }
            )
        data = _sealed(
            PILOT_ONLINE_STATE_SCHEMA,
            {
                "config_sha256": self.config.sha256,
                "layers": layers,
                "plan_sha256": self.plan.sha256,
                "width_policy_sha256": self.width_policy_sha256,
            },
        )
        if len(data) > _MAX_STATE_BYTES:
            raise MlpPilotOnlineStateError("online pilot state exceeds its byte bound")
        return data

    def _persist_locked(self) -> None:
        if self._state_path is None:
            return
        data = self._state_bytes()
        path = self._state_path
        temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | int(getattr(os, "O_CLOEXEC", 0))
                | int(getattr(os, "O_NOFOLLOW", 0)),
                0o600,
            )
            view = memoryview(data)
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written <= 0:
                    raise OSError("short online state write")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            if path.is_symlink():
                raise MlpPilotOnlineStateError("online state path cannot be a symlink")
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except MlpPilotOnlineStateError:
            raise
        except OSError as exc:
            raise MlpPilotOnlineStateError("cannot persist online pilot state") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def decision(self, *, layer: int, row_count: int) -> MlpPilotSparseDecision:
        """Return a non-mutating sparse/full decision for the next MLP wave."""

        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError("layer must be an integer")
        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 1
        ):
            raise ValueError("row_count must be positive")
        with self._lock:
            self._require_open()
            state = self._states.get(layer)
            if state is None:
                return MlpPilotSparseDecision(
                    layer, row_count, False, "unsupported-layer", 0, 0.0, 0
                )
            target_capture = max(
                self.config.min_capture,
                self.width_config.target_capture,
            )
            output_ready = (
                state.output_confirmed_rows
                >= self.width_config.min_output_confirmed_rows
                and len(state.output_cosines)
                >= self.width_config.min_output_confirmed_rows
                and min(state.output_cosines) >= self.width_config.min_output_cosine
                and max(state.output_relative_l2s)
                <= self.width_config.max_output_relative_l2
            )
            if row_count > self.config.max_sparse_rows:
                use_sparse, reason = False, "row-width"
            elif state.confirmed_rows == 0:
                use_sparse = state.sparse_waves < self.config.cold_start_sparse_waves
                reason = (
                    "weight-only-cold-start" if use_sparse else "confirmation-required"
                )
            elif state.confirmed_rows < self.config.min_confirmed_rows:
                use_sparse, reason = False, "confirmation-warmup"
            elif state.confirmed_capture_at_width < target_capture:
                use_sparse, reason = False, "low-confirmed-capture"
            elif (
                state.output_confirmed_rows
                < self.width_config.min_output_confirmed_rows
            ):
                use_sparse, reason = False, "output-calibration-required"
            elif not output_ready:
                use_sparse, reason = False, "output-calibration-failed"
            elif state.sparse_since_confirmation >= self.config.confirmation_interval:
                use_sparse, reason = False, "periodic-confirmation"
            else:
                use_sparse, reason = True, "target-confirmed"
            return MlpPilotSparseDecision(
                layer=layer,
                row_count=row_count,
                use_sparse=use_sparse,
                reason=reason,
                confirmed_rows=state.confirmed_rows,
                capture_ema=state.confirmed_capture_at_width,
                sparse_waves_since_confirmation=state.sparse_since_confirmation,
                selected_block_count=(
                    state.output_calibration_block_count
                    if output_ready
                    else state.selected_block_count
                ),
                max_selected_block_count=self._max_blocks[layer],
                required_score_mass=state.required_score_mass,
            )

    def _pilot_square_numpy(
        self, model: MlpPilotLayerModel, gate: np.ndarray, up: np.ndarray
    ) -> np.ndarray:
        expected = model.block_count * model.pilot_count
        if gate.shape != up.shape or gate.ndim != 2 or gate.shape[1] != expected:
            raise ValueError("pilot Gate/Up arrays differ from the online ABI")
        gate64 = np.asarray(gate, dtype=np.float64)
        up64 = np.asarray(up, dtype=np.float64)
        if not bool(np.isfinite(gate64).all()) or not bool(np.isfinite(up64).all()):
            raise ValueError("pilot Gate/Up arrays contain non-finite values")
        silu = gate64 / (1.0 + np.exp(-np.clip(gate64, -60.0, 60.0)))
        return np.square(silu * up64).reshape(
            len(gate64), model.block_count, model.pilot_count
        )

    def score_pilot_arrays(
        self, *, layer: int, gate: np.ndarray, up: np.ndarray
    ) -> np.ndarray:
        """Score blocks with the current target-confirmed coefficients."""

        with self._lock:
            self._require_open()
            model = self._models.get(layer)
            if model is None:
                raise KeyError(f"no online pilot model for layer {layer}")
            pilot_square = self._pilot_square_numpy(model, gate, up)
            denominator = np.maximum(
                pilot_square.sum(axis=(1, 2), dtype=np.float64), 1e-300
            )
            normalized = pilot_square / denominator[:, None, None]
            state = self._states[layer]
            scores = state.coefficients[None, :, 0] + np.sum(
                normalized * state.coefficients[None, :, 1:], axis=2
            )
            return np.maximum(scores, 0.0)

    def record_sparse(
        self,
        *,
        layer: int,
        row_count: int,
        selected_block_count: int | None = None,
    ) -> None:
        """Commit one successfully executed sparse wave to the bounded ledger."""

        if (
            isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 1
        ):
            raise ValueError("row_count must be positive")
        with self._lock:
            self._require_open()
            state = self._states.get(layer)
            if state is None:
                raise KeyError(f"no online pilot model for layer {layer}")
            executed_width = (
                state.selected_block_count
                if selected_block_count is None
                else selected_block_count
            )
            if (
                isinstance(executed_width, bool)
                or not isinstance(executed_width, int)
                or not self._models[layer].selected_block_count
                <= executed_width
                <= self._max_blocks[layer]
            ):
                raise ValueError("executed adaptive block width is invalid")
            state.sparse_waves += 1
            state.sparse_rows += row_count
            state.sparse_since_confirmation += 1
            state.last_sparse_block_count = executed_width
            self._dirty = True

    @staticmethod
    def _observation_rows(
        value: torch.Tensor | Sequence[torch.Tensor],
        *,
        width: int,
        field: str,
    ) -> torch.Tensor:
        rows = (value,) if isinstance(value, torch.Tensor) else tuple(value)
        if not rows or any(
            not isinstance(row, torch.Tensor)
            or row.ndim < 1
            or row.shape[-1] != width
            or not row.is_floating_point()
            for row in rows
        ):
            raise ValueError(f"{field} observation differs from the online ABI")
        try:
            return torch.cat(
                [row.reshape(-1, width) for row in rows],
                dim=0,
            )
        except RuntimeError as exc:
            raise ValueError(f"{field} observation tensors are not aligned") from exc

    def observe_full(
        self,
        *,
        layer: int,
        gate: torch.Tensor | Sequence[torch.Tensor],
        up: torch.Tensor | Sequence[torch.Tensor],
        output: torch.Tensor | Sequence[torch.Tensor],
        activated: torch.Tensor | Sequence[torch.Tensor] | None = None,
    ) -> MlpPilotOnlineObservation:
        """Learn from an already executed exact MLP without retaining its rows."""

        with self._lock:
            self._require_open()
            model = self._models.get(layer)
            if model is None:
                raise KeyError(f"no online pilot model for layer {layer}")
            flat_gate = self._observation_rows(
                gate,
                width=model.intermediate_dimension,
                field="Gate",
            )
            flat_up = self._observation_rows(
                up,
                width=model.intermediate_dimension,
                field="Up",
            )
            flat_output = self._observation_rows(
                output,
                width=self.plan.hidden_dimension,
                field="output",
            )
            if len(flat_gate) != len(flat_up) or len(flat_gate) != len(flat_output):
                raise ValueError("exact MLP observation differs from the online ABI")
            if activated is None:
                activation = F.silu(flat_gate.float()) * flat_up.float()
            else:
                activation = self._observation_rows(
                    activated,
                    width=model.intermediate_dimension,
                    field="activated",
                )
                if len(activation) != len(flat_gate):
                    raise ValueError("activated observation differs from Gate/Up")
                activation = activation.float()
            if not bool(torch.isfinite(activation).all()) or not bool(
                torch.isfinite(flat_output).all()
            ):
                raise ValueError("exact MLP observation contains non-finite values")
            flat_square = (
                activation.reshape(-1, model.intermediate_dimension).double().square()
            )
            rows = len(flat_square)
            block_square = flat_square.reshape(
                rows, model.block_count, model.block_size
            )
            target = block_square.sum(dim=2, dtype=torch.float64).cpu().numpy()
            pilot_indices = torch.tensor(
                model.pilot_neuron_indices(),
                device=flat_square.device,
                dtype=torch.long,
            )
            pilot_square = flat_square.index_select(1, pilot_indices).reshape(
                rows, model.block_count, model.pilot_count
            )
            pilot = pilot_square.to(device="cpu", dtype=torch.float64).numpy()
            total = np.maximum(target.sum(axis=1, dtype=np.float64), 1e-300)

            state = self._states[layer]
            raw_pilot_denominator = pilot.sum(axis=(1, 2), dtype=np.float64)
            valid = raw_pilot_denominator > np.maximum(
                total * np.finfo(np.float64).eps,
                np.finfo(np.float64).tiny,
            )
            normalized = np.divide(
                pilot,
                raw_pilot_denominator[:, None, None],
                out=np.zeros_like(pilot),
                where=valid[:, None, None],
            )
            pre_scores = state.coefficients[None, :, 0] + np.sum(
                normalized * state.coefficients[None, :, 1:], axis=2
            )
            pre_scores = np.maximum(pre_scores, 0.0)
            ranked = np.argsort(-pre_scores, axis=1, kind="stable")
            pilot_total = pilot.sum(axis=(1, 2), dtype=np.float64)
            pilot_by_block = pilot.sum(axis=2, dtype=np.float64)
            extra_target = np.maximum(target - pilot_by_block, 0.0)
            ranked_extra = np.take_along_axis(extra_target, ranked, axis=1)
            cumulative_extra = np.cumsum(ranked_extra, axis=1, dtype=np.float64)
            ranked_scores = np.take_along_axis(pre_scores, ranked, axis=1)
            cumulative_scores = np.cumsum(ranked_scores, axis=1, dtype=np.float64)
            score_total = pre_scores.sum(axis=1, dtype=np.float64)
            candidates = self._width_candidates(layer)
            capture_by_width: dict[int, float] = {}
            mass_by_width: dict[int, float] = {}
            for count in candidates:
                capture = np.clip(
                    (pilot_total + cumulative_extra[:, count - 1]) / total,
                    0.0,
                    1.0,
                )
                score_mass = np.divide(
                    cumulative_scores[:, count - 1],
                    score_total,
                    out=np.full(
                        rows,
                        count / model.block_count,
                        dtype=np.float64,
                    ),
                    where=score_total > np.finfo(np.float64).tiny,
                )
                capture_by_width[count] = float(np.min(capture))
                mass_by_width[count] = float(np.max(np.clip(score_mass, 0.0, 1.0)))
            target_capture = max(
                self.config.min_capture,
                self.width_config.target_capture,
            )
            selected_block_count = candidates[-1]
            for count in candidates:
                if capture_by_width[count] >= target_capture:
                    selected_block_count = count
                    break
            route_capture = capture_by_width[selected_block_count]
            predicted_score_mass = mass_by_width[selected_block_count]
            required_score_mass = min(
                1.0,
                predicted_score_mass + self.width_config.score_mass_margin,
            )
            calibration_width = self._max_blocks[layer]
            calibration_capture = np.clip(
                (pilot_total + cumulative_extra[:, calibration_width - 1]) / total,
                0.0,
                1.0,
            )
            shadow_row_indices = (
                ()
                if route_capture < target_capture
                else tuple(
                    int(row)
                    for row in np.argsort(calibration_capture, kind="stable")[
                        : self.width_config.max_shadow_rows
                    ]
                )
            )
            shadow_selected_blocks = tuple(
                tuple(int(block) for block in ranked[row, :calibration_width])
                for row in shadow_row_indices
            )

            if bool(valid.any()):
                valid_normalized = normalized[valid]
                valid_rows = len(valid_normalized)
                features = np.concatenate(
                    (
                        np.ones((valid_rows, model.block_count, 1), dtype=np.float64),
                        valid_normalized,
                    ),
                    axis=2,
                )
                normalized_target = target[valid] / raw_pilot_denominator[valid, None]
                decay = self.config.statistics_decay
                candidate_gram = state.gram * decay + np.einsum(
                    "rbf,rbg->bfg", features, features
                )
                candidate_rhs = state.rhs * decay + np.einsum(
                    "rbf,rb->bf", features, normalized_target
                )
                ridge = (
                    np.eye(model.pilot_count + 1, dtype=np.float64) * self.config.ridge
                )
                coefficients = np.linalg.solve(
                    candidate_gram + ridge[None, :, :],
                    candidate_rhs[:, :, None],
                )[:, :, 0]
                if bool(np.isfinite(coefficients).all()):
                    state.gram = candidate_gram
                    state.rhs = candidate_rhs
                    state.coefficients = coefficients
            previous_rows = state.confirmed_rows
            previous_width = state.selected_block_count
            state.confirmed_rows = min(
                self.config.max_confirmed_rows, previous_rows + rows
            )
            state.exact_waves += 1
            state.sparse_since_confirmation = 0
            state.last_capture = route_capture
            state.selected_block_count = selected_block_count
            state.predicted_score_mass = predicted_score_mass
            state.required_score_mass = required_score_mass
            state.output_calibration_block_count = calibration_width
            if selected_block_count != previous_width:
                state.width_updates += 1
            state.capture_ema = (
                route_capture
                if previous_rows == 0 or selected_block_count != previous_width
                else self.config.confidence_decay * state.capture_ema
                + (1.0 - self.config.confidence_decay) * route_capture
            )
            state.confirmed_capture_at_width = state.capture_ema
            surprising = route_capture < target_capture
            if surprising:
                state.surprises += 1
            self._dirty = True
            return MlpPilotOnlineObservation(
                layer=layer,
                rows=rows,
                route_capture=route_capture,
                capture_ema=state.capture_ema,
                confirmed_rows=state.confirmed_rows,
                surprising=surprising,
                selected_block_count=selected_block_count,
                predicted_score_mass=predicted_score_mass,
                required_score_mass=required_score_mass,
                shadow_row_indices=shadow_row_indices,
                shadow_selected_blocks=shadow_selected_blocks,
                output_confirmed_rows=state.output_confirmed_rows,
                output_worst_cosine=(
                    min(state.output_cosines) if state.output_cosines else 0.0
                ),
                output_worst_relative_l2=(
                    max(state.output_relative_l2s) if state.output_relative_l2s else 1.0
                ),
            )

    def observe_output_shadow(
        self,
        *,
        layer: int,
        sparse_output: torch.Tensor,
        exact_output: torch.Tensor,
    ) -> dict[str, object]:
        """Prequentially score and then fit one scalar output correction."""

        with self._lock:
            self._require_open()
            if (
                not isinstance(sparse_output, torch.Tensor)
                or not isinstance(exact_output, torch.Tensor)
                or sparse_output.shape != exact_output.shape
                or sparse_output.ndim != 2
                or sparse_output.shape[1] != self.plan.hidden_dimension
                or not bool(torch.isfinite(sparse_output).all())
                or not bool(torch.isfinite(exact_output).all())
            ):
                raise ValueError("output shadow differs from the calibration ABI")
            state = self._states[layer]
            raw = sparse_output.detach().to(device="cpu", dtype=torch.float64)
            truth = exact_output.detach().to(device="cpu", dtype=torch.float64)
            corrected = raw * state.output_scale + state.output_bias
            dot = (corrected * truth).sum(dim=1)
            predicted_norm = torch.linalg.vector_norm(corrected, dim=1)
            truth_norm = torch.linalg.vector_norm(truth, dim=1)
            cosine = torch.where(
                (predicted_norm == 0) & (truth_norm == 0),
                torch.ones_like(dot),
                torch.where(
                    (predicted_norm > 0) & (truth_norm > 0),
                    dot / (predicted_norm * truth_norm),
                    torch.zeros_like(dot),
                ),
            ).clamp(0.0, 1.0)
            relative_l2 = torch.linalg.vector_norm(
                corrected - truth, dim=1
            ) / torch.clamp(
                truth_norm,
                min=torch.finfo(torch.float64).tiny,
            )
            window = self.width_config.output_metric_window_rows
            state.output_cosines = (
                *state.output_cosines,
                *(float(value) for value in cosine.tolist()),
            )[-window:]
            state.output_relative_l2s = (
                *state.output_relative_l2s,
                *(float(value) for value in relative_l2.tolist()),
            )[-window:]
            state.output_confirmed_rows += len(raw)
            state.output_shadow_waves += 1

            flat_x = raw.reshape(-1).numpy()
            flat_y = truth.reshape(-1).numpy()
            state.output_element_count += len(flat_x)
            state.output_sum_x += float(flat_x.sum(dtype=np.float64))
            state.output_sum_y += float(flat_y.sum(dtype=np.float64))
            state.output_sum_xx += float(np.square(flat_x).sum(dtype=np.float64))
            state.output_sum_xy += float((flat_x * flat_y).sum(dtype=np.float64))
            count = state.output_element_count
            centered_xx = state.output_sum_xx - state.output_sum_x**2 / count
            centered_xy = (
                state.output_sum_xy - state.output_sum_x * state.output_sum_y / count
            )
            scale = centered_xy / centered_xx if centered_xx > 1e-12 else 1.0
            bias = state.output_sum_y / count - scale * state.output_sum_x / count
            if not math.isfinite(scale) or not math.isfinite(bias):
                raise MlpPilotOnlineStateError(
                    "output shadow correction became non-finite"
                )
            state.output_scale = max(-4.0, min(4.0, scale))
            state.output_bias = bias
            self._dirty = True
            return {
                "output_bias": state.output_bias,
                "output_confirmed_rows": state.output_confirmed_rows,
                "output_scale": state.output_scale,
                "output_worst_cosine": min(state.output_cosines),
                "output_worst_relative_l2": max(state.output_relative_l2s),
                "schema": "immer.qwen-mlp-pilot-output-shadow/v1",
            }

    def correct_sparse_output(
        self, *, layer: int, sparse_output: torch.Tensor
    ) -> torch.Tensor:
        with self._lock:
            self._require_open()
            state = self._states[layer]
            return sparse_output.float() * state.output_scale + state.output_bias

    def flush(self) -> None:
        """Persist all layer updates once at the surrounding request boundary."""

        with self._lock:
            self._require_open()
            if not self._dirty:
                return
            self._persist_locked()
            self._dirty = False

    def metrics(self) -> dict[str, object]:
        with self._lock:
            layers = [
                {
                    "capture_ema": state.capture_ema,
                    "confirmed_capture_at_width": state.confirmed_capture_at_width,
                    "confirmed_rows": state.confirmed_rows,
                    "exact_waves": state.exact_waves,
                    "last_capture": state.last_capture,
                    "last_sparse_block_count": state.last_sparse_block_count,
                    "layer": layer,
                    "max_selected_block_count": self._max_blocks[layer],
                    "output_bias": state.output_bias,
                    "output_calibration_block_count": state.output_calibration_block_count,
                    "output_confirmed_rows": state.output_confirmed_rows,
                    "output_scale": state.output_scale,
                    "output_shadow_waves": state.output_shadow_waves,
                    "output_worst_cosine": (
                        min(state.output_cosines) if state.output_cosines else 0.0
                    ),
                    "output_worst_relative_l2": (
                        max(state.output_relative_l2s)
                        if state.output_relative_l2s
                        else 1.0
                    ),
                    "predicted_score_mass": state.predicted_score_mass,
                    "required_score_mass": state.required_score_mass,
                    "selected_block_count": state.selected_block_count,
                    "sparse_rows": state.sparse_rows,
                    "sparse_waves": state.sparse_waves,
                    "surprises": state.surprises,
                    "width_updates": state.width_updates,
                }
                for layer, state in sorted(self._states.items())
            ]
            return {
                "config_sha256": self.config.sha256,
                "layers": layers,
                "persistent": self._state_path is not None,
                "plan_sha256": self.plan.sha256,
                "schema": "immer.qwen-mlp-pilot-online-metrics/v2",
                "width_policy_sha256": self.width_policy_sha256,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            failure: Exception | None = None
            try:
                if self._dirty:
                    self._persist_locked()
                    self._dirty = False
            except Exception as exc:  # release the process lock even on I/O failure.
                failure = exc
            self._closed = True
            self._release_file_lock()
            if failure is not None:
                raise MlpPilotOnlineStateError(
                    "online pilot controller cleanup failed"
                ) from failure


__all__ = [
    "LEGACY_PILOT_ONLINE_STATE_SCHEMA",
    "V2_PILOT_ONLINE_STATE_SCHEMA",
    "PILOT_ADAPTIVE_WIDTH_CONFIG_SCHEMA",
    "PILOT_ONLINE_CONFIG_SCHEMA",
    "PILOT_ONLINE_OBSERVATION_SCHEMA",
    "PILOT_ONLINE_STATE_SCHEMA",
    "PILOT_SPARSE_DECISION_SCHEMA",
    "WEIGHT_ONLY_AFFINE_SCHEMA",
    "WEIGHT_ONLY_INITIALIZER",
    "WEIGHT_ONLY_PLAN_SCHEMA",
    "MlpPilotIdentityAffinePlan",
    "MlpPilotAdaptiveWidthConfig",
    "MlpPilotOnlineConfig",
    "MlpPilotOnlineController",
    "MlpPilotOnlineObservation",
    "MlpPilotOnlineStateError",
    "MlpPilotSparseDecision",
    "MlpPilotWeightOnlyError",
    "MlpPilotWeightOnlyPlan",
    "build_weight_only_layer_model",
    "gaussian_joint_moments",
    "weight_only_model_pin",
]
