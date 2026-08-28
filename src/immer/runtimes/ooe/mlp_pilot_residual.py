"""Train-only diagonal affine residual Crystal for sparse Qwen MLP drafts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence, cast

import numpy as np
import torch

from .identity import canonical_json_bytes, require_sha256
from .mlp_pilot_output import MlpPilotOutputMetrics, PILOT_OUTPUT_KERNEL
from .mlp_pilot_router import (
    MlpPilotRouterIntegrityError,
    _array_from_record,
    _array_record,
    _sha256s,
)


PILOT_AFFINE_LAYER_SCHEMA = "immer.qwen-mlp-pilot-affine-layer/v1"
PILOT_AFFINE_FIT_SCHEMA = "immer.qwen-mlp-pilot-affine-fit/v1"
PILOT_AFFINE_EVALUATION_SCHEMA = "immer.qwen-mlp-pilot-affine-evaluation/v1"
PILOT_AFFINE_VERIFIER_SHA256 = hashlib.sha256(
    b"immer:qwen-mlp-pilot-residual/train-only-diagonal-affine/v1"
).hexdigest()
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


class MlpPilotResidualError(RuntimeError):
    pass


class MlpPilotResidualIntegrityError(MlpPilotResidualError):
    pass


class MlpPilotResidualCapacityError(MlpPilotResidualError):
    pass


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sealed(schema: str, body: Mapping[str, object]) -> dict[str, object]:
    normalized = json.loads(canonical_json_bytes(dict(body)))
    return {"body": normalized, "body_sha256": _digest(normalized), "schema": schema}


def _strict_json(data: bytes, *, schema: str, label: str) -> Mapping[str, object]:
    if not isinstance(data, bytes) or not data or len(data) > _MAX_ARTIFACT_BYTES:
        raise MlpPilotResidualIntegrityError(f"{label} exceeds its byte bound")

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MlpPilotResidualIntegrityError(f"{label} is not strict JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != {"body", "body_sha256", "schema"}
        or value.get("schema") != schema
        or not isinstance(value.get("body"), Mapping)
        or value.get("body_sha256") != _digest(value.get("body"))
        or canonical_json_bytes(value) != data
    ):
        raise MlpPilotResidualIntegrityError(f"{label} seal is invalid")
    return cast(Mapping[str, object], value)


@dataclass(frozen=True, slots=True)
class MlpPilotAffineLayer:
    layer: int
    output_dimension: int
    scale: np.ndarray
    bias: np.ndarray
    training_group_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or self.layer < 0
            or isinstance(self.output_dimension, bool)
            or not isinstance(self.output_dimension, int)
            or self.output_dimension < 1
        ):
            raise ValueError("affine layer ABI is invalid")
        scale = np.asarray(self.scale)
        bias = np.asarray(self.bias)
        if (
            scale.dtype != np.dtype(np.float64)
            or bias.dtype != np.dtype(np.float64)
            or scale.shape != (self.output_dimension,)
            or bias.shape != scale.shape
            or not bool(np.isfinite(scale).all())
            or not bool(np.isfinite(bias).all())
        ):
            raise ValueError("affine layer coefficients are invalid")
        scale = np.array(scale, dtype="<f8", order="C", copy=True)
        bias = np.array(bias, dtype="<f8", order="C", copy=True)
        scale[scale == 0.0] = 0.0
        bias[bias == 0.0] = 0.0
        scale.flags.writeable = False
        bias.flags.writeable = False
        training = _sha256s(self.training_group_sha256s, field="training_group_sha256s")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "training_group_sha256s", training)

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def apply(self, sparse_output: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(sparse_output, torch.Tensor)
            or sparse_output.ndim != 2
            or sparse_output.shape[1] != self.output_dimension
            or not sparse_output.is_floating_point()
            or not bool(torch.isfinite(sparse_output).all())
        ):
            raise ValueError("sparse output differs from the affine ABI")
        scale = torch.tensor(
            self.scale, dtype=torch.float32, device=sparse_output.device
        )
        bias = torch.tensor(self.bias, dtype=torch.float32, device=sparse_output.device)
        return sparse_output.float() * scale + bias

    def to_record(self) -> dict[str, object]:
        return {
            "bias": _array_record(self.bias.reshape(1, -1)),
            "layer": self.layer,
            "output_dimension": self.output_dimension,
            "scale": _array_record(self.scale.reshape(1, -1)),
            "schema": PILOT_AFFINE_LAYER_SCHEMA,
            "training_group_sha256s": list(self.training_group_sha256s),
        }

    @classmethod
    def from_record(cls, value: object) -> "MlpPilotAffineLayer":
        expected = {
            "bias",
            "layer",
            "output_dimension",
            "scale",
            "schema",
            "training_group_sha256s",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != PILOT_AFFINE_LAYER_SCHEMA
            or not isinstance(value.get("training_group_sha256s"), list)
        ):
            raise MlpPilotResidualIntegrityError("affine layer record is invalid")
        try:
            dimension = cast(int, value["output_dimension"])
            return cls(
                layer=cast(int, value["layer"]),
                output_dimension=dimension,
                scale=_array_from_record(
                    value["scale"], field="scale", shape=(1, dimension)
                ).reshape(-1),
                bias=_array_from_record(
                    value["bias"], field="bias", shape=(1, dimension)
                ).reshape(-1),
                training_group_sha256s=tuple(
                    cast(list[str], value["training_group_sha256s"])
                ),
            )
        except (TypeError, ValueError, MlpPilotRouterIntegrityError) as exc:
            raise MlpPilotResidualIntegrityError(
                "affine layer validation failed"
            ) from exc


class MlpPilotAffineAccumulator:
    """Sufficient statistics for one exact diagonal OLS fit."""

    __slots__ = ("count", "sum_x", "sum_xx", "sum_xy", "sum_y")

    def __init__(self, output_dimension: int) -> None:
        if (
            isinstance(output_dimension, bool)
            or not isinstance(output_dimension, int)
            or output_dimension < 1
        ):
            raise ValueError("output_dimension must be positive")
        self.count = 0
        self.sum_x = np.zeros(output_dimension, dtype=np.float64)
        self.sum_y = np.zeros(output_dimension, dtype=np.float64)
        self.sum_xx = np.zeros(output_dimension, dtype=np.float64)
        self.sum_xy = np.zeros(output_dimension, dtype=np.float64)

    def add(self, sparse_output: torch.Tensor, truth: torch.Tensor) -> None:
        if (
            not isinstance(sparse_output, torch.Tensor)
            or not isinstance(truth, torch.Tensor)
            or sparse_output.shape != truth.shape
            or sparse_output.ndim != 2
            or sparse_output.shape[1] != len(self.sum_x)
            or not sparse_output.is_floating_point()
            or not truth.is_floating_point()
        ):
            raise ValueError("affine fit tensors are invalid")
        x = sparse_output.detach().to(dtype=torch.float64, device="cpu").numpy()
        y = truth.detach().to(dtype=torch.float64, device="cpu").numpy()
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("affine fit tensors contain non-finite values")
        self.count += len(x)
        self.sum_x += x.sum(axis=0)
        self.sum_y += y.sum(axis=0)
        self.sum_xx += np.square(x).sum(axis=0)
        self.sum_xy += (x * y).sum(axis=0)

    def fit(
        self, *, layer: int, training_group_sha256s: Sequence[str]
    ) -> MlpPilotAffineLayer:
        if self.count < 2:
            raise MlpPilotResidualError("affine fit needs at least two rows")
        centered_xx = self.sum_xx - np.square(self.sum_x) / self.count
        centered_xy = self.sum_xy - self.sum_x * self.sum_y / self.count
        scale = np.divide(
            centered_xy,
            centered_xx,
            out=np.ones_like(centered_xy),
            where=centered_xx > 1e-12,
        )
        bias = self.sum_y / self.count - scale * self.sum_x / self.count
        return MlpPilotAffineLayer(
            layer=layer,
            output_dimension=len(scale),
            scale=scale,
            bias=bias,
            training_group_sha256s=tuple(sorted(training_group_sha256s)),
        )


@dataclass(frozen=True, slots=True)
class MlpPilotAffineFit:
    model_pin_sha256: str
    router_fit_sha256: str
    source_bank_state_sha256: str
    source_corpus_sha256: str
    source_row_role_sha256: str
    weights_index_sha256: str
    source_row_count: int
    models: tuple[MlpPilotAffineLayer, ...]

    def __post_init__(self) -> None:
        for field in (
            "model_pin_sha256",
            "router_fit_sha256",
            "source_bank_state_sha256",
            "source_corpus_sha256",
            "source_row_role_sha256",
            "weights_index_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        if (
            isinstance(self.source_row_count, bool)
            or not isinstance(self.source_row_count, int)
            or self.source_row_count < 2
        ):
            raise ValueError("source_row_count must be at least two")
        models = tuple(self.models)
        if (
            not models
            or any(not isinstance(model, MlpPilotAffineLayer) for model in models)
            or tuple(model.layer for model in models)
            != tuple(sorted(model.layer for model in models))
            or len({model.layer for model in models}) != len(models)
            or len({model.output_dimension for model in models}) != 1
        ):
            raise ValueError("affine model inventory is invalid")
        object.__setattr__(self, "models", models)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        return _sealed(
            PILOT_AFFINE_FIT_SCHEMA,
            {
                "kernel": PILOT_OUTPUT_KERNEL,
                "model_pin_sha256": self.model_pin_sha256,
                "models": [model.to_record() for model in self.models],
                "router_fit_sha256": self.router_fit_sha256,
                "source_bank_state_sha256": self.source_bank_state_sha256,
                "source_corpus_sha256": self.source_corpus_sha256,
                "source_row_count": self.source_row_count,
                "source_row_role_sha256": self.source_row_role_sha256,
                "verifier_sha256": PILOT_AFFINE_VERIFIER_SHA256,
                "weights_index_sha256": self.weights_index_sha256,
            },
        )

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > _MAX_ARTIFACT_BYTES:
            raise MlpPilotResidualCapacityError("affine fit exceeds its byte bound")
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> "MlpPilotAffineFit":
        envelope = _strict_json(
            data, schema=PILOT_AFFINE_FIT_SCHEMA, label="affine fit"
        )
        body = cast(Mapping[str, object], envelope["body"])
        expected = {
            "kernel",
            "model_pin_sha256",
            "models",
            "router_fit_sha256",
            "source_bank_state_sha256",
            "source_corpus_sha256",
            "source_row_count",
            "source_row_role_sha256",
            "verifier_sha256",
            "weights_index_sha256",
        }
        if (
            set(body) != expected
            or body.get("kernel") != PILOT_OUTPUT_KERNEL
            or body.get("verifier_sha256") != PILOT_AFFINE_VERIFIER_SHA256
            or not isinstance(body.get("models"), list)
        ):
            raise MlpPilotResidualIntegrityError("affine fit body is invalid")
        try:
            result = cls(
                model_pin_sha256=cast(str, body["model_pin_sha256"]),
                router_fit_sha256=cast(str, body["router_fit_sha256"]),
                source_bank_state_sha256=cast(str, body["source_bank_state_sha256"]),
                source_corpus_sha256=cast(str, body["source_corpus_sha256"]),
                source_row_role_sha256=cast(str, body["source_row_role_sha256"]),
                weights_index_sha256=cast(str, body["weights_index_sha256"]),
                source_row_count=cast(int, body["source_row_count"]),
                models=tuple(
                    MlpPilotAffineLayer.from_record(row)
                    for row in cast(list[object], body["models"])
                ),
            )
        except (TypeError, ValueError) as exc:
            raise MlpPilotResidualIntegrityError(
                "affine fit validation failed"
            ) from exc
        if result.to_bytes() != data:
            raise MlpPilotResidualIntegrityError("affine fit reconstruction changed")
        return result


@dataclass(frozen=True, slots=True)
class MlpPilotAffineEvaluation:
    fit_sha256: str
    router_evaluation_sha256: str
    holdout_bank_state_sha256: str
    holdout_corpus_sha256: str
    holdout_row_role_sha256: str
    row_count: int
    raw_metrics: MlpPilotOutputMetrics
    corrected_metrics: MlpPilotOutputMetrics
    layer_metrics: tuple[tuple[int, MlpPilotOutputMetrics, MlpPilotOutputMetrics], ...]

    def __post_init__(self) -> None:
        for field in (
            "fit_sha256",
            "router_evaluation_sha256",
            "holdout_bank_state_sha256",
            "holdout_corpus_sha256",
            "holdout_row_role_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        if (
            isinstance(self.row_count, bool)
            or not isinstance(self.row_count, int)
            or self.row_count < 1
            or not isinstance(self.raw_metrics, MlpPilotOutputMetrics)
            or not isinstance(self.corrected_metrics, MlpPilotOutputMetrics)
        ):
            raise ValueError("affine evaluation metrics are invalid")
        layers = tuple(self.layer_metrics)
        if (
            not layers
            or tuple(row[0] for row in layers)
            != tuple(sorted(row[0] for row in layers))
            or len({row[0] for row in layers}) != len(layers)
            or any(
                len(row) != 3
                or not isinstance(row[1], MlpPilotOutputMetrics)
                or not isinstance(row[2], MlpPilotOutputMetrics)
                for row in layers
            )
        ):
            raise ValueError("affine layer evaluation inventory is invalid")
        object.__setattr__(self, "layer_metrics", layers)

    @property
    def l2_error_reduction(self) -> float:
        return 1.0 - (
            self.corrected_metrics.relative_l2_error
            / self.raw_metrics.relative_l2_error
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        return _sealed(
            PILOT_AFFINE_EVALUATION_SCHEMA,
            {
                "corrected_metrics": self.corrected_metrics.to_record(),
                "fit_sha256": self.fit_sha256,
                "holdout_bank_state_sha256": self.holdout_bank_state_sha256,
                "holdout_corpus_sha256": self.holdout_corpus_sha256,
                "holdout_row_role_sha256": self.holdout_row_role_sha256,
                "l2_error_reduction": self.l2_error_reduction,
                "layer_metrics": [
                    {
                        "corrected": corrected.to_record(),
                        "layer": layer,
                        "raw": raw.to_record(),
                    }
                    for layer, raw, corrected in self.layer_metrics
                ],
                "raw_metrics": self.raw_metrics.to_record(),
                "router_evaluation_sha256": self.router_evaluation_sha256,
                "row_count": self.row_count,
                "verifier_sha256": PILOT_AFFINE_VERIFIER_SHA256,
            },
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_document())

    @classmethod
    def from_bytes(cls, data: bytes) -> "MlpPilotAffineEvaluation":
        envelope = _strict_json(
            data, schema=PILOT_AFFINE_EVALUATION_SCHEMA, label="affine evaluation"
        )
        body = cast(Mapping[str, object], envelope["body"])
        expected = {
            "corrected_metrics",
            "fit_sha256",
            "holdout_bank_state_sha256",
            "holdout_corpus_sha256",
            "holdout_row_role_sha256",
            "l2_error_reduction",
            "layer_metrics",
            "raw_metrics",
            "router_evaluation_sha256",
            "row_count",
            "verifier_sha256",
        }
        if (
            set(body) != expected
            or body.get("verifier_sha256") != PILOT_AFFINE_VERIFIER_SHA256
            or not isinstance(body.get("layer_metrics"), list)
        ):
            raise MlpPilotResidualIntegrityError("affine evaluation body is invalid")
        try:
            layers = []
            for row in cast(list[object], body["layer_metrics"]):
                if not isinstance(row, Mapping) or set(row) != {
                    "corrected",
                    "layer",
                    "raw",
                }:
                    raise ValueError("affine layer metric row is invalid")
                layers.append(
                    (
                        cast(int, row["layer"]),
                        MlpPilotOutputMetrics.from_record(row["raw"]),
                        MlpPilotOutputMetrics.from_record(row["corrected"]),
                    )
                )
            result = cls(
                fit_sha256=cast(str, body["fit_sha256"]),
                router_evaluation_sha256=cast(str, body["router_evaluation_sha256"]),
                holdout_bank_state_sha256=cast(str, body["holdout_bank_state_sha256"]),
                holdout_corpus_sha256=cast(str, body["holdout_corpus_sha256"]),
                holdout_row_role_sha256=cast(str, body["holdout_row_role_sha256"]),
                row_count=cast(int, body["row_count"]),
                raw_metrics=MlpPilotOutputMetrics.from_record(body["raw_metrics"]),
                corrected_metrics=MlpPilotOutputMetrics.from_record(
                    body["corrected_metrics"]
                ),
                layer_metrics=tuple(layers),
            )
        except (TypeError, ValueError) as exc:
            raise MlpPilotResidualIntegrityError(
                "affine evaluation validation failed"
            ) from exc
        if body.get("l2_error_reduction") != result.l2_error_reduction:
            raise MlpPilotResidualIntegrityError("affine evaluation reduction changed")
        if result.to_bytes() != data:
            raise MlpPilotResidualIntegrityError(
                "affine evaluation reconstruction changed"
            )
        return result


__all__ = [
    "PILOT_AFFINE_EVALUATION_SCHEMA",
    "PILOT_AFFINE_FIT_SCHEMA",
    "PILOT_AFFINE_LAYER_SCHEMA",
    "PILOT_AFFINE_VERIFIER_SHA256",
    "MlpPilotAffineAccumulator",
    "MlpPilotAffineEvaluation",
    "MlpPilotAffineFit",
    "MlpPilotAffineLayer",
    "MlpPilotResidualCapacityError",
    "MlpPilotResidualError",
    "MlpPilotResidualIntegrityError",
]
