"""Bridge sealed OoE Seed projections into the neural shadow trainer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from immer.runtimes.ooe.seed_projection import SeedSplit, SeedTrainingBatch

from .training import SeedV3ShadowBatch


@dataclass(frozen=True, slots=True)
class SeedV3ProjectionViews:
    """Aligned typed views; target meaning remains an explicit caller contract."""

    feature: torch.Tensor
    action: torch.Tensor
    consequence: torch.Tensor
    quality: torch.Tensor
    work_cost: torch.Tensor
    attention_mask: torch.Tensor
    answer_pos: torch.Tensor
    group_sha256s: tuple[str, ...]
    batch_sha256: str
    batch_manifest_sha256: str
    feature_schema_sha256: str


TargetProjector = Callable[[SeedV3ProjectionViews], Mapping[str, torch.Tensor | None]]


def seed_v3_projection_views(
    batch: SeedTrainingBatch,
    split: SeedSplit,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> SeedV3ProjectionViews:
    if not isinstance(batch, SeedTrainingBatch):
        raise TypeError("batch must be SeedTrainingBatch")
    if not dtype.is_floating_point:
        raise TypeError("Seed projection dtype must be floating point")
    arrays: dict[str, torch.Tensor] = {}
    expected_mask = None
    expected_groups: tuple[str, ...] | None = None
    for name in ("feature", "action", "consequence", "quality", "work_cost"):
        values, mask, groups = batch.vector_tensor(split, name)
        if expected_mask is None:
            expected_mask = mask
            expected_groups = groups
        elif groups != expected_groups or not bool((mask == expected_mask).all()):
            raise ValueError("Seed projection vector views lost sequence alignment")
        arrays[name] = torch.as_tensor(values.copy(), dtype=dtype, device=device)
    if expected_mask is None or expected_groups is None or not expected_groups:
        raise ValueError(f"Seed projection split {split!r} is empty")
    attention_mask = torch.as_tensor(
        expected_mask.copy(), dtype=torch.bool, device=device
    )
    answer_pos = attention_mask.sum(dim=1, dtype=torch.long) - 1
    if bool((answer_pos < 0).any()):
        raise ValueError("Seed projection contains an empty sequence group")
    return SeedV3ProjectionViews(
        feature=arrays["feature"],
        action=arrays["action"],
        consequence=arrays["consequence"],
        quality=arrays["quality"],
        work_cost=arrays["work_cost"],
        attention_mask=attention_mask,
        answer_pos=answer_pos,
        group_sha256s=expected_groups,
        batch_sha256=batch.sha256,
        batch_manifest_sha256=batch.manifest_sha256,
        feature_schema_sha256=batch.feature_schema_sha256,
    )


def shadow_batch_from_projection(
    batch: SeedTrainingBatch,
    split: SeedSplit,
    *,
    target_projector: TargetProjector,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> tuple[SeedV3ShadowBatch, SeedV3ProjectionViews]:
    """Create a trainer batch while keeping every target mapping explicit."""

    if not callable(target_projector):
        raise TypeError("target_projector must be callable")
    views = seed_v3_projection_views(batch, split, dtype=dtype, device=device)
    projected = target_projector(views)
    if not isinstance(projected, Mapping):
        raise TypeError("target_projector must return a mapping")
    allowed = {
        "operator_targets",
        "state_targets",
        "final_targets",
        "novelty_targets",
        "predictive_targets",
        "quotient_targets",
        "key_targets",
        "key_mask",
        "route_value_targets",
        "expected_work_targets",
        "route_mask",
    }
    if set(projected) - allowed:
        raise ValueError("target_projector returned an unknown Seed target")
    result = SeedV3ShadowBatch(
        receipt_features=views.feature,
        attention_mask=views.attention_mask,
        answer_pos=views.answer_pos,
        **dict(projected),
    )
    return result, views


__all__ = [
    "SeedV3ProjectionViews",
    "TargetProjector",
    "seed_v3_projection_views",
    "shadow_batch_from_projection",
]
