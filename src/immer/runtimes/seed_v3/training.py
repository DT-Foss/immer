"""Masked multi-head training for the receipt-native Seed v3 shadow path."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math
from typing import Mapping

import torch
import torch.nn.functional as F

from .model import ImmerSeedV3, SeedV3Output


@dataclass(frozen=True, slots=True)
class SeedV3LossWeights:
    operator: float = 2.0
    state: float = 0.5
    final: float = 0.15
    novelty: float = 0.1
    predictive: float = 0.15
    quotient: float = 0.1
    key: float = 0.8
    key_margin: float = 0.2
    route_value: float = 1.0
    expected_work: float = 0.5

    def __post_init__(self) -> None:
        values = tuple(getattr(self, field.name) for field in fields(self))
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in values
        ):
            raise ValueError("loss weights must be finite and non-negative")
        if not any(float(value) > 0.0 for value in values):
            raise ValueError("at least one loss weight must be positive")


@dataclass(frozen=True, slots=True)
class SeedV3ShadowBatch:
    """One padded ``[batch, sequence, feature]`` receipt-training batch."""

    receipt_features: torch.Tensor
    attention_mask: torch.Tensor
    answer_pos: torch.Tensor
    operator_targets: torch.Tensor | None = None
    state_targets: torch.Tensor | None = None
    final_targets: torch.Tensor | None = None
    novelty_targets: torch.Tensor | None = None
    predictive_targets: torch.Tensor | None = None
    quotient_targets: torch.Tensor | None = None
    key_targets: torch.Tensor | None = None
    key_mask: torch.Tensor | None = None
    route_value_targets: torch.Tensor | None = None
    expected_work_targets: torch.Tensor | None = None
    route_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        features = self.receipt_features
        if not isinstance(features, torch.Tensor) or features.ndim != 3:
            raise ValueError("receipt_features must have shape [batch, sequence, feature]")
        if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
            raise ValueError("receipt_features must be finite floating point")
        batch, sequence, _feature = features.shape
        if self.attention_mask.shape != (batch, sequence):
            raise ValueError("attention_mask must match receipt_features")
        if self.answer_pos.shape != (batch,):
            raise ValueError("answer_pos must have shape [batch]")
        for name in (
            "operator_targets",
            "state_targets",
            "novelty_targets",
            "predictive_targets",
            "quotient_targets",
        ):
            value = getattr(self, name)
            if value is not None and value.shape != (batch, sequence):
                raise ValueError(f"{name} must have shape [batch, sequence]")
        if self.final_targets is not None and self.final_targets.shape != (batch,):
            raise ValueError("final_targets must have shape [batch]")
        if self.key_targets is not None:
            if self.key_targets.ndim != 3 or self.key_targets.shape[:2] != (
                batch,
                sequence,
            ):
                raise ValueError("key_targets must have shape [batch, sequence, key]")
            if not self.key_targets.is_floating_point() or not bool(
                torch.isfinite(self.key_targets).all()
            ):
                raise ValueError("key_targets must be finite floating point")
            if bool((self.key_targets.abs() > 1.0).any()):
                raise ValueError("key_targets must lie in [-1, 1]")
        if self.key_mask is not None and self.key_mask.shape != (batch, sequence):
            raise ValueError("key_mask must have shape [batch, sequence]")
        for name in ("route_value_targets", "expected_work_targets"):
            value = getattr(self, name)
            if value is not None:
                if value.ndim != 2 or value.shape[0] != batch:
                    raise ValueError(f"{name} must have shape [batch, route]")
                if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"{name} must be finite floating point")
        if self.expected_work_targets is not None and bool(
            (self.expected_work_targets < 0.0).any()
        ):
            raise ValueError("expected_work_targets cannot be negative")
        if self.route_mask is not None:
            route_shape = None
            if self.route_value_targets is not None:
                route_shape = self.route_value_targets.shape
            if self.expected_work_targets is not None:
                if route_shape is not None and self.expected_work_targets.shape != route_shape:
                    raise ValueError("route target shapes differ")
                route_shape = self.expected_work_targets.shape
            if route_shape is None or self.route_mask.shape != route_shape:
                raise ValueError("route_mask must match configured route targets")
        if all(
            getattr(self, name) is None
            for name in (
                "operator_targets",
                "state_targets",
                "final_targets",
                "novelty_targets",
                "predictive_targets",
                "quotient_targets",
                "key_targets",
                "route_value_targets",
                "expected_work_targets",
            )
        ):
            raise ValueError("shadow batch requires at least one training target")

    def to(self, device: torch.device | str) -> "SeedV3ShadowBatch":
        values: dict[str, torch.Tensor | None] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = None if value is None else value.to(device)
        return SeedV3ShadowBatch(**values)  # type: ignore[arg-type]


def _classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor | None,
    *,
    name: str,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if targets is None:
        return None
    if targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError(f"{name} targets must contain integers")
    targets = targets.to(device=logits.device, dtype=torch.long)
    valid = targets != -100
    if valid_mask is not None:
        if valid_mask.shape != targets.shape:
            raise ValueError(f"{name} mask does not match targets")
        valid = valid & valid_mask.to(device=logits.device, dtype=torch.bool)
    if not bool(valid.any()):
        return logits.sum() * 0.0
    selected = targets[valid]
    if bool(((selected < 0) | (selected >= logits.shape[-1])).any()):
        raise ValueError(f"{name} targets exceed the configured class count")
    return F.cross_entropy(logits[valid], selected)


def seed_v3_shadow_losses(
    output: SeedV3Output,
    batch: SeedV3ShadowBatch,
    *,
    weights: SeedV3LossWeights | None = None,
) -> dict[str, torch.Tensor]:
    """Compute only the heads whose exact receipt targets are present."""

    active_weights = weights or SeedV3LossWeights()
    losses: dict[str, torch.Tensor] = {}
    classification = (
        ("operator", output.operator_logits, batch.operator_targets),
        ("state", output.state_logits, batch.state_targets),
        ("novelty", output.novelty_logits, batch.novelty_targets),
        ("predictive", output.predictive_logits, batch.predictive_targets),
        ("quotient", output.quotient_logits, batch.quotient_targets),
    )
    for name, logits, targets in classification:
        value = _classification_loss(
            logits,
            targets,
            name=name,
            valid_mask=batch.attention_mask,
        )
        if value is not None:
            losses[name] = value
    final = _classification_loss(output.final_logits, batch.final_targets, name="final")
    if final is not None:
        losses["final"] = final

    if batch.key_targets is not None:
        targets = batch.key_targets.to(device=output.mlp_keys.device, dtype=output.mlp_keys.dtype)
        if targets.shape != output.mlp_keys.shape:
            raise ValueError("key_targets do not match the Seed joint-key width")
        mask = batch.attention_mask if batch.key_mask is None else batch.key_mask
        mask = mask.to(device=output.mlp_keys.device, dtype=torch.bool)
        if not bool(mask.any()):
            losses["key"] = output.mlp_keys.sum() * 0.0
            losses["key_margin"] = output.mlp_keys.sum() * 0.0
        else:
            bounded = torch.tanh(output.mlp_keys[mask])
            selected_targets = targets[mask]
            losses["key"] = F.mse_loss(bounded, selected_targets)
            losses["key_margin"] = torch.relu(0.65 - bounded.abs()).mean()

    route_mask = batch.route_mask
    for name, prediction, targets in (
        ("route_value", output.route_values, batch.route_value_targets),
        ("expected_work", output.expected_work, batch.expected_work_targets),
    ):
        if targets is None:
            continue
        targets = targets.to(device=prediction.device, dtype=prediction.dtype)
        if targets.shape != prediction.shape:
            raise ValueError(f"{name} targets do not match configured routes")
        mask = (
            torch.ones_like(prediction, dtype=torch.bool)
            if route_mask is None
            else route_mask.to(device=prediction.device, dtype=torch.bool)
        )
        losses[name] = (
            F.mse_loss(prediction[mask], targets[mask])
            if bool(mask.any())
            else prediction.sum() * 0.0
        )

    if not losses:
        raise ValueError("shadow batch has no active targets")
    total = output.hidden.sum() * 0.0
    for name, value in losses.items():
        total = total + float(getattr(active_weights, name)) * value
    losses["total"] = total
    return losses


class SeedV3ShadowTrainer:
    """Bounded optimizer for the proposal path; it never executes a proposal."""

    def __init__(
        self,
        model: ImmerSeedV3,
        *,
        learning_rate: float = 3.0e-3,
        weight_decay: float = 1.0e-4,
        max_grad_norm: float = 1.0,
        device: torch.device | str = "cpu",
        weights: SeedV3LossWeights | None = None,
    ) -> None:
        if not isinstance(model, ImmerSeedV3):
            raise TypeError("model must be ImmerSeedV3")
        for value, name in (
            (learning_rate, "learning_rate"),
            (max_grad_norm, "max_grad_norm"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be positive and finite")
        if (
            isinstance(weight_decay, bool)
            or not isinstance(weight_decay, (int, float))
            or not math.isfinite(float(weight_decay))
            or float(weight_decay) < 0.0
        ):
            raise ValueError("weight_decay must be finite and non-negative")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.weights = weights or SeedV3LossWeights()
        self.max_grad_norm = float(max_grad_norm)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(learning_rate),
            weight_decay=float(weight_decay),
        )
        self.steps = 0

    def losses(self, batch: SeedV3ShadowBatch) -> dict[str, torch.Tensor]:
        active = batch.to(self.device)
        output = self.model.forward_receipts(
            active.receipt_features,
            active.attention_mask,
            active.answer_pos,
        )
        return seed_v3_shadow_losses(output, active, weights=self.weights)

    def step(self, batch: SeedV3ShadowBatch) -> Mapping[str, float]:
        self.model.train()
        losses = self.losses(batch)
        self.optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.steps += 1
        return {name: float(value.detach().cpu()) for name, value in losses.items()}


__all__ = [
    "SeedV3LossWeights",
    "SeedV3ShadowBatch",
    "SeedV3ShadowTrainer",
    "seed_v3_shadow_losses",
]
