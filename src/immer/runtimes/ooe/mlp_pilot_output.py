"""Exact two-pass down-projection kernel and output metrics for pilot routes."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, cast

import numpy as np
import torch
import torch.nn.functional as F

from .mlp_pilot_router import MlpPilotLayerModel


PILOT_OUTPUT_METRICS_SCHEMA = "immer.qwen-mlp-pilot-output-metrics/v1"
PILOT_OUTPUT_KERNEL = "bf16-pilot-pass+selected-block-residual-pass"


class MlpPilotOutputError(RuntimeError):
    pass


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class MlpPilotOutputMetrics:
    relative_l2_error: float
    cosine: float
    predicted_energy_ratio: float
    optimal_scalar: float
    scalar_oracle_l2_error: float
    max_abs_error: float
    values: int

    def __post_init__(self) -> None:
        for field in (
            "relative_l2_error",
            "cosine",
            "predicted_energy_ratio",
            "optimal_scalar",
            "scalar_oracle_l2_error",
            "max_abs_error",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), field=field))
        if (
            self.relative_l2_error < 0.0
            or not -1.0 <= self.cosine <= 1.0
            or self.predicted_energy_ratio < 0.0
            or self.scalar_oracle_l2_error < 0.0
            or self.max_abs_error < 0.0
            or isinstance(self.values, bool)
            or not isinstance(self.values, int)
            or self.values < 1
        ):
            raise ValueError("pilot output metrics are outside their domain")

    def to_record(self) -> dict[str, object]:
        return {
            "cosine": self.cosine,
            "max_abs_error": self.max_abs_error,
            "optimal_scalar": self.optimal_scalar,
            "predicted_energy_ratio": self.predicted_energy_ratio,
            "relative_l2_error": self.relative_l2_error,
            "scalar_oracle_l2_error": self.scalar_oracle_l2_error,
            "schema": PILOT_OUTPUT_METRICS_SCHEMA,
            "values": self.values,
        }

    @classmethod
    def from_record(cls, value: object) -> "MlpPilotOutputMetrics":
        expected = {
            "cosine",
            "max_abs_error",
            "optimal_scalar",
            "predicted_energy_ratio",
            "relative_l2_error",
            "scalar_oracle_l2_error",
            "schema",
            "values",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != PILOT_OUTPUT_METRICS_SCHEMA
        ):
            raise ValueError("pilot output metrics record is invalid")
        return cls(
            relative_l2_error=cast(float, value["relative_l2_error"]),
            cosine=cast(float, value["cosine"]),
            predicted_energy_ratio=cast(float, value["predicted_energy_ratio"]),
            optimal_scalar=cast(float, value["optimal_scalar"]),
            scalar_oracle_l2_error=cast(float, value["scalar_oracle_l2_error"]),
            max_abs_error=cast(float, value["max_abs_error"]),
            values=cast(int, value["values"]),
        )


class MlpPilotOutputMetricAccumulator:
    """Stable aggregate metrics without retaining output tensors."""

    __slots__ = ("dot", "max_abs", "predicted_square", "sse", "true_square", "values")

    def __init__(self) -> None:
        self.sse = 0.0
        self.true_square = 0.0
        self.dot = 0.0
        self.predicted_square = 0.0
        self.max_abs = 0.0
        self.values = 0

    def add(self, predicted: torch.Tensor, truth: torch.Tensor) -> None:
        if (
            not isinstance(predicted, torch.Tensor)
            or not isinstance(truth, torch.Tensor)
            or predicted.shape != truth.shape
            or predicted.numel() < 1
            or not predicted.is_floating_point()
            or not truth.is_floating_point()
            or not bool(torch.isfinite(predicted).all())
            or not bool(torch.isfinite(truth).all())
        ):
            raise ValueError("predicted/truth output tensors are invalid")
        candidate = predicted.detach().to(dtype=torch.float64, device="cpu")
        target = truth.detach().to(dtype=torch.float64, device="cpu")
        difference = candidate - target
        self.sse += float(torch.square(difference).sum())
        self.true_square += float(torch.square(target).sum())
        self.dot += float((candidate * target).sum())
        self.predicted_square += float(torch.square(candidate).sum())
        self.max_abs = max(self.max_abs, float(difference.abs().max()))
        self.values += target.numel()

    def result(self) -> MlpPilotOutputMetrics:
        if self.values < 1 or self.true_square <= 0.0 or self.predicted_square <= 0.0:
            raise MlpPilotOutputError(
                "output metric accumulator lacks nonzero evidence"
            )
        squared_cosine = (self.dot * self.dot) / (
            self.predicted_square * self.true_square
        )
        return MlpPilotOutputMetrics(
            relative_l2_error=math.sqrt(self.sse / self.true_square),
            cosine=self.dot / math.sqrt(self.predicted_square * self.true_square),
            predicted_energy_ratio=self.predicted_square / self.true_square,
            optimal_scalar=self.dot / self.predicted_square,
            scalar_oracle_l2_error=math.sqrt(max(0.0, 1.0 - squared_cosine)),
            max_abs_error=self.max_abs,
            values=self.values,
        )


def _linear_inputs(
    model: MlpPilotLayerModel,
    activated: torch.Tensor,
    down_weight: torch.Tensor,
) -> None:
    if (
        not isinstance(model, MlpPilotLayerModel)
        or not isinstance(activated, torch.Tensor)
        or not isinstance(down_weight, torch.Tensor)
        or activated.ndim != 2
        or down_weight.ndim != 2
        or activated.shape[1] != model.intermediate_dimension
        or down_weight.shape[1] != model.intermediate_dimension
        or activated.device != down_weight.device
        or activated.dtype != down_weight.dtype
        or not activated.is_floating_point()
        or not down_weight.is_floating_point()
    ):
        raise ValueError("pilot down-projection kernel ABI is invalid")


def pilot_sparse_down_projection(
    model: MlpPilotLayerModel,
    activated: torch.Tensor,
    down_weight: torch.Tensor,
    selected_blocks: np.ndarray,
) -> torch.Tensor:
    """Execute pilots once, then add the non-pilot rows of selected blocks."""

    _linear_inputs(model, activated, down_weight)
    if (
        type(selected_blocks) is not np.ndarray
        or selected_blocks.dtype != np.dtype(np.int64)
        or selected_blocks.shape != (activated.shape[0], model.selected_block_count)
        or bool((selected_blocks < 0).any())
        or bool((selected_blocks >= model.block_count).any())
        or any(
            len(set(row.tolist())) != model.selected_block_count
            for row in selected_blocks
        )
    ):
        raise ValueError("selected_blocks differ from the router action ABI")
    pilots = torch.as_tensor(
        model.pilot_neuron_indices(), dtype=torch.long, device=activated.device
    )
    pilot_set = set(int(row) for row in pilots.tolist())
    pilot_output = F.linear(activated[:, pilots], down_weight[:, pilots])
    result = torch.empty(
        (activated.shape[0], down_weight.shape[0]),
        dtype=activated.dtype,
        device=activated.device,
    )
    for row, blocks in enumerate(selected_blocks):
        neurons = np.asarray(
            sorted(
                neuron
                for block in blocks
                for neuron in range(
                    int(block) * model.block_size,
                    (int(block) + 1) * model.block_size,
                )
                if neuron not in pilot_set
            ),
            dtype=np.int64,
        )
        expected = model.selected_block_count * (model.block_size - model.pilot_count)
        if len(neurons) != expected:
            raise ValueError("selected blocks are duplicated or invalid")
        indices = torch.as_tensor(neurons, dtype=torch.long, device=activated.device)
        result[row] = pilot_output[row] + F.linear(
            activated[row, indices], down_weight[:, indices]
        )
    return result


def marginal_sparse_down_projection(
    model: MlpPilotLayerModel,
    activated: torch.Tensor,
    down_weight: torch.Tensor,
) -> torch.Tensor:
    """Execute the equal-or-higher-compute static block baseline."""

    _linear_inputs(model, activated, down_weight)
    block_count = math.ceil(model.selected_neuron_count / model.block_size)
    blocks = np.argsort(-model.marginal_block_scores, kind="stable")[:block_count]
    neurons = np.concatenate(
        [
            np.arange(block * model.block_size, (block + 1) * model.block_size)
            for block in blocks
        ]
    ).astype(np.int64, copy=False)
    indices = torch.as_tensor(neurons, dtype=torch.long, device=activated.device)
    return F.linear(activated[:, indices], down_weight[:, indices])


__all__ = [
    "PILOT_OUTPUT_KERNEL",
    "PILOT_OUTPUT_METRICS_SCHEMA",
    "MlpPilotOutputError",
    "MlpPilotOutputMetricAccumulator",
    "MlpPilotOutputMetrics",
    "marginal_sparse_down_projection",
    "pilot_sparse_down_projection",
]
