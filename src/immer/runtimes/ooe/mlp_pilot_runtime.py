"""True selected-row execution for the Qwen MLP pilot and residual Crystals."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping, Protocol, Sequence, cast, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F

from ..qwen3_8.kernels import swiglu
from .identity import canonical_json_bytes, require_sha256
from .mlp_pilot_residual import MlpPilotAffineFit, MlpPilotAffineLayer
from .mlp_pilot_router import MlpPilotLayerModel, MlpPilotRouterFit
from .mlp_pilot_weight_only import (
    MlpPilotIdentityAffinePlan,
    MlpPilotOnlineController,
    MlpPilotOnlineObservation,
    MlpPilotSparseDecision,
    MlpPilotWeightOnlyPlan,
)

PILOT_SPARSE_RUNTIME_SCHEMA = "immer.qwen-mlp-pilot-sparse-runtime/v3"
PILOT_TRANSPOSE_ENTRY_SCHEMA = "immer.qwen-mlp-pilot-transpose-entry/v1"
PILOT_TRANSPOSE_MANIFEST_SCHEMA = "immer.qwen-mlp-pilot-transpose-manifest/v1"


class MlpPilotSparseRuntimeError(RuntimeError):
    pass


class MlpPilotNonBeneficialRoute(MlpPilotSparseRuntimeError):
    """Pilot scoring proved that sparse transport cannot beat the full MLP."""

    exact_mlp_fallback = True

    def __init__(self, *, layer: int, estimated_fraction: float) -> None:
        self.layer = layer
        self.estimated_fraction = estimated_fraction
        super().__init__(
            f"layer {layer} adaptive route needs {estimated_fraction:.6f} "
            "of full MLP row transport"
        )


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class MlpPilotTransposeEntry:
    layer: int
    source_tensor: str
    transpose_tensor: str
    shard: str
    shape: tuple[int, int]
    source_raw_sha256: str
    transpose_raw_sha256: str
    shard_sha256: str
    shard_bytes: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer, bool)
            or not isinstance(self.layer, int)
            or self.layer < 0
        ):
            raise ValueError("transpose layer is invalid")
        for field in ("source_tensor", "transpose_tensor", "shard"):
            value = getattr(self, field)
            if (
                not isinstance(value, str)
                or not value
                or (field == "shard" and value != value.split("/")[-1])
            ):
                raise ValueError(f"{field} is invalid")
        shape = tuple(self.shape)
        if len(shape) != 2 or any(
            isinstance(row, bool) or not isinstance(row, int) or row < 1
            for row in shape
        ):
            raise ValueError("transpose shape is invalid")
        for field in (
            "source_raw_sha256",
            "transpose_raw_sha256",
            "shard_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        if (
            isinstance(self.shard_bytes, bool)
            or not isinstance(self.shard_bytes, int)
            or self.shard_bytes < 1
        ):
            raise ValueError("shard_bytes is invalid")
        object.__setattr__(self, "shape", cast(tuple[int, int], shape))

    def to_record(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "schema": PILOT_TRANSPOSE_ENTRY_SCHEMA,
            "shape": list(self.shape),
            "shard": self.shard,
            "shard_bytes": self.shard_bytes,
            "shard_sha256": self.shard_sha256,
            "source_raw_sha256": self.source_raw_sha256,
            "source_tensor": self.source_tensor,
            "transpose_raw_sha256": self.transpose_raw_sha256,
            "transpose_tensor": self.transpose_tensor,
        }

    @classmethod
    def from_record(cls, value: object) -> "MlpPilotTransposeEntry":
        expected = {
            "layer",
            "schema",
            "shape",
            "shard",
            "shard_bytes",
            "shard_sha256",
            "source_raw_sha256",
            "source_tensor",
            "transpose_raw_sha256",
            "transpose_tensor",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != PILOT_TRANSPOSE_ENTRY_SCHEMA
            or not isinstance(value.get("shape"), list)
        ):
            raise ValueError("transpose entry record is invalid")
        return cls(
            layer=cast(int, value["layer"]),
            source_tensor=cast(str, value["source_tensor"]),
            transpose_tensor=cast(str, value["transpose_tensor"]),
            shard=cast(str, value["shard"]),
            shape=tuple(cast(list[int], value["shape"])),  # type: ignore[arg-type]
            source_raw_sha256=cast(str, value["source_raw_sha256"]),
            transpose_raw_sha256=cast(str, value["transpose_raw_sha256"]),
            shard_sha256=cast(str, value["shard_sha256"]),
            shard_bytes=cast(int, value["shard_bytes"]),
        )


@dataclass(frozen=True, slots=True)
class MlpPilotTransposeManifest:
    model_pin_sha256: str
    router_fit_sha256: str
    affine_fit_sha256: str
    weights_index_sha256: str
    entries: tuple[MlpPilotTransposeEntry, ...]

    def __post_init__(self) -> None:
        for field in (
            "model_pin_sha256",
            "router_fit_sha256",
            "affine_fit_sha256",
            "weights_index_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        entries = tuple(self.entries)
        if (
            not entries
            or any(not isinstance(row, MlpPilotTransposeEntry) for row in entries)
            or tuple(row.layer for row in entries)
            != tuple(sorted(row.layer for row in entries))
            or len({row.layer for row in entries}) != len(entries)
            or len({row.transpose_tensor for row in entries}) != len(entries)
        ):
            raise ValueError("transpose entry inventory is invalid")
        object.__setattr__(self, "entries", entries)

    @property
    def total_shard_bytes(self) -> int:
        return sum(row.shard_bytes for row in self.entries)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        body = {
            "affine_fit_sha256": self.affine_fit_sha256,
            "entries": [row.to_record() for row in self.entries],
            "model_pin_sha256": self.model_pin_sha256,
            "router_fit_sha256": self.router_fit_sha256,
            "total_shard_bytes": self.total_shard_bytes,
            "weights_index_sha256": self.weights_index_sha256,
        }
        return {
            "body": body,
            "body_sha256": _digest(body),
            "schema": PILOT_TRANSPOSE_MANIFEST_SCHEMA,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_document())

    @classmethod
    def from_bytes(cls, data: bytes) -> "MlpPilotTransposeManifest":
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
            raise ValueError("transpose manifest is not JSON") from exc
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "body_sha256", "schema"}
            or value.get("schema") != PILOT_TRANSPOSE_MANIFEST_SCHEMA
            or not isinstance(value.get("body"), Mapping)
            or value.get("body_sha256") != _digest(value.get("body"))
            or canonical_json_bytes(value) != data
        ):
            raise ValueError("transpose manifest seal is invalid")
        body = cast(Mapping[str, object], value["body"])
        expected = {
            "affine_fit_sha256",
            "entries",
            "model_pin_sha256",
            "router_fit_sha256",
            "total_shard_bytes",
            "weights_index_sha256",
        }
        if set(body) != expected or not isinstance(body.get("entries"), list):
            raise ValueError("transpose manifest body is invalid")
        result = cls(
            model_pin_sha256=cast(str, body["model_pin_sha256"]),
            router_fit_sha256=cast(str, body["router_fit_sha256"]),
            affine_fit_sha256=cast(str, body["affine_fit_sha256"]),
            weights_index_sha256=cast(str, body["weights_index_sha256"]),
            entries=tuple(
                MlpPilotTransposeEntry.from_record(row)
                for row in cast(list[object], body["entries"])
            ),
        )
        if (
            body.get("total_shard_bytes") != result.total_shard_bytes
            or result.to_bytes() != data
        ):
            raise ValueError("transpose manifest reconstruction changed")
        return result


@runtime_checkable
class SelectedRowPager(Protocol):
    device: Any
    compute_dtype: Any

    def tensor_rows(self, name: str, row_ids: Sequence[int]) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class MlpPilotSparseTrace:
    layer: int
    row_count: int
    selected_blocks: tuple[tuple[int, ...], ...]
    selected_neuron_count: int
    intermediate_dimension: int
    source_weight_rows: int
    full_weight_rows: int
    dynamic_requested_rows: int
    dynamic_unique_rows: int
    down_requested_rows: int
    down_loaded_rows: int
    output_dtype: str
    range_mode: str = "scattered-pilot+union-target+route-cache-down"

    def __post_init__(self) -> None:
        for field in (
            "layer",
            "row_count",
            "selected_neuron_count",
            "intermediate_dimension",
            "source_weight_rows",
            "full_weight_rows",
            "dynamic_requested_rows",
            "dynamic_unique_rows",
            "down_requested_rows",
            "down_loaded_rows",
        ):
            value = getattr(self, field)
            minimum = 0 if field == "layer" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field} is invalid")
        blocks = tuple(tuple(row) for row in self.selected_blocks)
        if (
            len(blocks) != self.row_count
            or any(not row or len(set(row)) != len(row) for row in blocks)
            or self.selected_neuron_count >= self.intermediate_dimension
            or self.full_weight_rows != 3 * self.intermediate_dimension
            or self.dynamic_unique_rows > self.dynamic_requested_rows
            or self.down_loaded_rows > self.down_requested_rows
            or not isinstance(self.output_dtype, str)
            or not self.output_dtype
            or self.range_mode
            not in {
                "scattered-pilot+union-target+route-cache-down",
                "consolidated-pilot+union-target+route-cache-down",
                "packed-q4-pilot+selected-blocks",
            }
        ):
            raise ValueError("sparse runtime trace is inconsistent")
        object.__setattr__(self, "selected_blocks", blocks)

    @property
    def weight_row_fraction(self) -> float:
        return self.source_weight_rows / self.full_weight_rows

    @property
    def dynamic_row_reuse(self) -> float:
        return self.dynamic_requested_rows / self.dynamic_unique_rows

    @property
    def down_row_reuse(self) -> float:
        return self.down_requested_rows / self.down_loaded_rows

    def to_record(self) -> dict[str, object]:
        return {
            "full_weight_rows": self.full_weight_rows,
            "intermediate_dimension": self.intermediate_dimension,
            "layer": self.layer,
            "dynamic_requested_rows": self.dynamic_requested_rows,
            "dynamic_row_reuse": self.dynamic_row_reuse,
            "dynamic_unique_rows": self.dynamic_unique_rows,
            "down_loaded_rows": self.down_loaded_rows,
            "down_requested_rows": self.down_requested_rows,
            "down_row_reuse": self.down_row_reuse,
            "output_dtype": self.output_dtype,
            "row_count": self.row_count,
            "range_mode": self.range_mode,
            "schema": PILOT_SPARSE_RUNTIME_SCHEMA,
            "selected_blocks": [list(row) for row in self.selected_blocks],
            "selected_neuron_count": self.selected_neuron_count,
            "source_weight_rows": self.source_weight_rows,
            "weight_row_fraction": self.weight_row_fraction,
        }


class MlpPilotSparseExecutor:
    """Execute only pilot and chosen Qwen MLP rows through two range pagers."""

    def __init__(
        self,
        router_fit: MlpPilotRouterFit | MlpPilotWeightOnlyPlan,
        affine_fit: MlpPilotAffineFit | MlpPilotIdentityAffinePlan | None,
        weight_pager: SelectedRowPager,
        down_transpose_pager: SelectedRowPager,
        *,
        pilot_pager: SelectedRowPager | None = None,
        active_layers: Sequence[int] | None = None,
        output_dtype: torch.dtype = torch.float32,
        online_controller: MlpPilotOnlineController | None = None,
    ) -> None:
        if not isinstance(router_fit, (MlpPilotRouterFit, MlpPilotWeightOnlyPlan)):
            raise TypeError("router_fit must be a fitted or weight-only pilot plan")
        if isinstance(router_fit, MlpPilotWeightOnlyPlan):
            expected_affine = router_fit.affine_fit
            if affine_fit is None:
                affine_fit = expected_affine
            if (
                not isinstance(affine_fit, MlpPilotIdentityAffinePlan)
                or affine_fit.sha256 != expected_affine.sha256
            ):
                raise ValueError("weight-only plan requires its identity affine plan")
        elif not isinstance(affine_fit, MlpPilotAffineFit):
            raise TypeError("fitted router requires MlpPilotAffineFit")
        if affine_fit.router_fit_sha256 != router_fit.sha256:
            raise ValueError("affine fit differs from the router fit")
        if affine_fit.model_pin_sha256 != router_fit.model_pin_sha256:
            raise ValueError("affine and router fits cross model pins")
        if not isinstance(weight_pager, SelectedRowPager) or not isinstance(
            down_transpose_pager, SelectedRowPager
        ):
            raise TypeError("sparse execution requires selected-row pagers")
        if weight_pager.device != down_transpose_pager.device:
            raise ValueError("selected-row pagers must share a device")
        if weight_pager.compute_dtype != down_transpose_pager.compute_dtype:
            raise ValueError("selected-row pagers must share a compute dtype")
        if pilot_pager is not None and (
            not isinstance(pilot_pager, SelectedRowPager)
            or pilot_pager.device != weight_pager.device
            or pilot_pager.compute_dtype != weight_pager.compute_dtype
        ):
            raise ValueError("pilot pager must share the selected-row pager ABI")
        if output_dtype not in {torch.bfloat16, torch.float16, torch.float32}:
            raise ValueError("output_dtype must be bfloat16, float16, or float32")
        if online_controller is not None and (
            not isinstance(router_fit, MlpPilotWeightOnlyPlan)
            or not isinstance(online_controller, MlpPilotOnlineController)
            or online_controller.plan.sha256 != router_fit.sha256
        ):
            raise ValueError("online controller differs from the weight-only plan")
        router_models = {model.layer: model for model in router_fit.models}
        affine_models = {model.layer: model for model in affine_fit.models}
        if set(router_models) != set(affine_models):
            raise ValueError("router and affine fits cover different layers")
        fitted_layers = set(router_models)
        if active_layers is None:
            selected_layers = tuple(sorted(fitted_layers))
        else:
            try:
                selected_layers = tuple(active_layers)
            except TypeError as exc:
                raise TypeError("active_layers must be an integer sequence") from exc
            if (
                not selected_layers
                or selected_layers != tuple(sorted(set(selected_layers)))
                or any(
                    isinstance(layer, bool)
                    or not isinstance(layer, int)
                    or layer not in fitted_layers
                    for layer in selected_layers
                )
            ):
                raise ValueError(
                    "active_layers must be a sorted non-empty fitted subset"
                )
        self.router_fit = router_fit
        self.affine_fit = affine_fit
        self.weight_pager = weight_pager
        self.down_transpose_pager = down_transpose_pager
        self.pilot_pager = pilot_pager
        self.output_dtype = output_dtype
        self.online_controller = online_controller
        self._router_models = router_models
        self._affine_models = affine_models
        self._active_layers = frozenset(selected_layers)
        transport_sources = []
        for label, selected_pager in (
            ("weight", self.weight_pager),
            ("down_transpose", self.down_transpose_pager),
            ("pilot", self.pilot_pager),
        ):
            if selected_pager is None:
                continue
            source = getattr(selected_pager, "source", None)
            inventory = getattr(source, "inventory", None)
            if callable(inventory):
                inventory()
            metrics = source.metrics() if source is not None else {}
            transport_sources.append(
                {
                    "compute_dtype": str(selected_pager.compute_dtype).removeprefix(
                        "torch."
                    ),
                    "device": str(selected_pager.device),
                    "inventory_fingerprint": metrics.get(
                        "inventory_source_fingerprint"
                    ),
                    "label": label,
                    "repo_id": metrics.get("repo_id", getattr(source, "repo_id", None)),
                    "revision": metrics.get(
                        "revision", getattr(source, "revision", None)
                    ),
                }
            )
        self._transport_sources = tuple(transport_sources)
        self.last_trace: MlpPilotSparseTrace | None = None

    @staticmethod
    def _base(layer: int) -> str:
        return f"model.language_model.layers.{layer}.mlp"

    @classmethod
    def transpose_name(cls, layer: int) -> str:
        return f"{cls._base(layer)}.down_proj.weight.transpose"

    @classmethod
    def pilot_name(cls, layer: int, projection: str) -> str:
        if projection not in {"gate", "up", "down_transpose"}:
            raise ValueError("pilot projection is invalid")
        return f"{cls._base(layer)}.pilot.{projection}.weight"

    def supports_layer(self, layer: int) -> bool:
        return layer in self._active_layers

    def decision(self, *, layer: int, row_count: int) -> MlpPilotSparseDecision:
        """Choose sparse execution or the caller's unchanged exact MLP path."""

        if not self.supports_layer(layer):
            return MlpPilotSparseDecision(
                layer=layer,
                row_count=row_count,
                use_sparse=False,
                reason="unsupported-layer",
                confirmed_rows=0,
                capture_ema=0.0,
                sparse_waves_since_confirmation=0,
            )
        if self.online_controller is not None:
            return self.online_controller.decision(layer=layer, row_count=row_count)
        return MlpPilotSparseDecision(
            layer=layer,
            row_count=row_count,
            use_sparse=True,
            reason="sealed-prompt-fit",
            confirmed_rows=0,
            capture_ema=0.0,
            sparse_waves_since_confirmation=0,
            selected_block_count=self._router_models[layer].selected_block_count,
            max_selected_block_count=self._router_models[layer].selected_block_count,
        )

    def observe_full(
        self,
        *,
        layer: int,
        gate: torch.Tensor | Sequence[torch.Tensor],
        up: torch.Tensor | Sequence[torch.Tensor],
        output: torch.Tensor | Sequence[torch.Tensor],
        activated: torch.Tensor | Sequence[torch.Tensor] | None = None,
        down_weight: torch.Tensor | None = None,
    ) -> MlpPilotOnlineObservation | None:
        """Ingest values already produced by an exact full-MLP fallback."""

        if self.online_controller is None:
            return None
        observation = self.online_controller.observe_full(
            layer=layer,
            gate=gate,
            up=up,
            output=output,
            activated=activated,
        )
        if not observation.shadow_row_indices:
            return observation
        model = self._router_models[layer]
        affine = self._affine_models[layer]
        if activated is None:
            activated_rows = self.online_controller._observation_rows(
                gate,
                width=model.intermediate_dimension,
                field="Gate",
            )
            up_rows = self.online_controller._observation_rows(
                up,
                width=model.intermediate_dimension,
                field="Up",
            )
            activated_rows = swiglu(activated_rows, up_rows)
        else:
            activated_rows = self.online_controller._observation_rows(
                activated,
                width=model.intermediate_dimension,
                field="activated",
            )
        exact_rows = self.online_controller._observation_rows(
            output,
            width=affine.output_dimension,
            field="output",
        )
        pilots = model.pilot_neuron_indices()
        if down_weight is not None:
            if (
                not isinstance(down_weight, torch.Tensor)
                or not down_weight.is_floating_point()
                or tuple(down_weight.shape)
                != (affine.output_dimension, model.intermediate_dimension)
                or down_weight.device != activated_rows.device
            ):
                raise ValueError("resident Down weight differs from shadow ABI")
            shadow_outputs = []
            truth_outputs = []
            for row, blocks in zip(
                observation.shadow_row_indices,
                observation.shadow_selected_blocks,
                strict=True,
            ):
                selected = set(int(value) for value in pilots)
                for block in blocks:
                    selected.update(
                        range(
                            block * model.block_size,
                            (block + 1) * model.block_size,
                        )
                    )
                ids = torch.tensor(
                    sorted(selected),
                    device=activated_rows.device,
                    dtype=torch.long,
                )
                source = activated_rows[row : row + 1]
                masked = torch.zeros_like(source)
                masked.index_copy_(1, ids, source.index_select(1, ids))
                shadow_outputs.append(
                    torch.nn.functional.linear(
                        masked.to(down_weight.dtype),
                        down_weight,
                    )
                )
                truth_outputs.append(exact_rows[row : row + 1])
            metrics = self.online_controller.observe_output_shadow(
                layer=layer,
                sparse_output=torch.cat(shadow_outputs, dim=0),
                exact_output=torch.cat(truth_outputs, dim=0),
            )
            return replace(
                observation,
                output_confirmed_rows=cast(int, metrics["output_confirmed_rows"]),
                output_worst_cosine=cast(float, metrics["output_worst_cosine"]),
                output_worst_relative_l2=cast(
                    float,
                    metrics["output_worst_relative_l2"],
                ),
            )
        if self.pilot_pager is None:
            down_pilot_weight = self._rows(
                self.down_transpose_pager,
                self.transpose_name(layer),
                pilots,
                columns=affine.output_dimension,
            )
        else:
            down_pilot_weight = self._rows(
                self.pilot_pager,
                self.pilot_name(layer, "down_transpose"),
                np.arange(len(pilots), dtype=np.int64),
                columns=affine.output_dimension,
            )
        pilot_ids = torch.tensor(
            pilots,
            device=activated_rows.device,
            dtype=torch.long,
        )
        shadow_outputs = []
        truth_outputs = []
        pilot_set = set(int(value) for value in pilots)
        try:
            for row, blocks in zip(
                observation.shadow_row_indices,
                observation.shadow_selected_blocks,
                strict=True,
            ):
                ids = np.asarray(
                    [
                        neuron
                        for block in blocks
                        for neuron in range(
                            block * model.block_size,
                            (block + 1) * model.block_size,
                        )
                        if self.pilot_pager is not None or neuron not in pilot_set
                    ],
                    dtype=np.int64,
                )
                down_weight = self._rows(
                    self.down_transpose_pager,
                    self.transpose_name(layer),
                    ids,
                    columns=affine.output_dimension,
                )
                dynamic_ids = torch.tensor(
                    ids,
                    device=activated_rows.device,
                    dtype=torch.long,
                )
                dynamic = activated_rows[row : row + 1].index_select(1, dynamic_ids)
                if self.pilot_pager is not None:
                    dynamic = dynamic.clone()
                    for block_index, block in enumerate(blocks):
                        for offset in model.pilot_offsets[block]:
                            dynamic[:, block_index * model.block_size + offset] = 0
                pilot_activation = activated_rows[row : row + 1].index_select(
                    1, pilot_ids
                )
                shadow_outputs.append(
                    pilot_activation.to(down_pilot_weight.dtype) @ down_pilot_weight
                    + dynamic.to(down_weight.dtype) @ down_weight
                )
                truth_outputs.append(exact_rows[row : row + 1])
                del down_weight
        finally:
            del down_pilot_weight
        metrics = self.online_controller.observe_output_shadow(
            layer=layer,
            sparse_output=torch.cat(shadow_outputs, dim=0),
            exact_output=torch.cat(truth_outputs, dim=0),
        )
        return replace(
            observation,
            output_confirmed_rows=cast(int, metrics["output_confirmed_rows"]),
            output_worst_cosine=cast(float, metrics["output_worst_cosine"]),
            output_worst_relative_l2=cast(float, metrics["output_worst_relative_l2"]),
        )

    def online_metrics(self) -> dict[str, object] | None:
        return (
            None if self.online_controller is None else self.online_controller.metrics()
        )

    @property
    def active_layers(self) -> tuple[int, ...]:
        return tuple(sorted(self._active_layers))

    def snapshot_identity(
        self, *, transport_neutral: bool = False
    ) -> dict[str, object]:
        identity: dict[str, object] = {
            "affine_fit_sha256": self.affine_fit.sha256,
            "fitted_layers": sorted(self._router_models),
            "layers": sorted(self._active_layers),
            "output_dtype": str(self.output_dtype).removeprefix("torch."),
            "range_mode": (
                "scattered-pilot+union-target+route-cache-down"
                if self.pilot_pager is None
                else "consolidated-pilot+union-target+route-cache-down"
            ),
            "router_fit_sha256": self.router_fit.sha256,
            "schema": PILOT_SPARSE_RUNTIME_SCHEMA,
        }
        if self.online_controller is not None:
            identity["online_config_sha256"] = self.online_controller.config.sha256
            identity["weight_only_plan_sha256"] = self.router_fit.sha256
            identity["adaptive_width_policy_sha256"] = (
                self.online_controller.width_policy_sha256
            )
        if transport_neutral:
            return identity
        return {**identity, "sources": list(self._transport_sources)}

    @staticmethod
    def _rows(
        pager: SelectedRowPager,
        name: str,
        ids: np.ndarray,
        *,
        columns: int,
    ) -> torch.Tensor:
        rows = pager.tensor_rows(name, tuple(int(row) for row in ids))
        if (
            not isinstance(rows, torch.Tensor)
            or rows.ndim != 2
            or rows.shape != (len(ids), columns)
            or rows.device != pager.device
            or rows.dtype != pager.compute_dtype
            or not bool(torch.isfinite(rows).all())
        ):
            raise MlpPilotSparseRuntimeError(
                f"selected rows for {name!r} differ from the sparse ABI"
            )
        return rows

    def _project_union_rows(
        self,
        hidden: torch.Tensor,
        *,
        name: str,
        extra_by_row: Sequence[np.ndarray],
        union_ids: np.ndarray,
        chunk_rows: int,
        columns: int,
    ) -> tuple[torch.Tensor, ...]:
        """Read each routed target row once with one-route-bounded residency."""

        pieces: list[list[torch.Tensor]] = [[] for _ in extra_by_row]
        output_positions: list[list[int]] = [[] for _ in extra_by_row]
        union = tuple(int(neuron) for neuron in union_ids)
        for start in range(0, len(union), chunk_rows):
            chunk = union[start : start + chunk_rows]
            chunk_offsets = {neuron: index for index, neuron in enumerate(chunk)}
            weight = self._rows(
                self.weight_pager,
                name,
                np.asarray(chunk, dtype=np.int64),
                columns=columns,
            )
            try:
                for row, extra in enumerate(extra_by_row):
                    selected = [
                        (index, chunk_offsets[int(neuron)])
                        for index, neuron in enumerate(extra)
                        if int(neuron) in chunk_offsets
                    ]
                    if not selected:
                        continue
                    weight_positions = torch.tensor(
                        [position for _output, position in selected],
                        device=self.weight_pager.device,
                        dtype=torch.long,
                    )
                    pieces[row].append(
                        F.linear(
                            hidden[row : row + 1],
                            weight.index_select(0, weight_positions),
                        )
                    )
                    output_positions[row].extend(
                        output for output, _position in selected
                    )
            finally:
                del weight

        result = []
        for row, extra in enumerate(extra_by_row):
            if not pieces[row] or len(output_positions[row]) != len(extra):
                raise MlpPilotSparseRuntimeError(
                    "union target projection lost routed neurons"
                )
            projected = torch.cat(pieces[row], dim=-1)
            by_output = {
                output: index for index, output in enumerate(output_positions[row])
            }
            restore = torch.tensor(
                [by_output[index] for index in range(len(extra))],
                device=self.weight_pager.device,
                dtype=torch.long,
            )
            result.append(projected.index_select(-1, restore))
        return tuple(result)

    @staticmethod
    def _route_processing_order(
        block_orders: Sequence[Sequence[int]],
    ) -> tuple[int, ...]:
        """Order rows by exact unique-route DP, then bounded greedy fallback."""

        routes = tuple(frozenset(int(block) for block in row) for row in block_orders)
        if not routes or any(
            not route or len(route) != len(tuple(row))
            for route, row in zip(routes, block_orders, strict=True)
        ):
            raise MlpPilotSparseRuntimeError("down route inventory is invalid")

        unique_routes: list[frozenset[int]] = []
        rows_by_route: list[list[int]] = []
        route_index: dict[frozenset[int], int] = {}
        for row, route in enumerate(routes):
            index = route_index.get(route)
            if index is None:
                index = len(unique_routes)
                route_index[route] = index
                unique_routes.append(route)
                rows_by_route.append([])
            rows_by_route[index].append(row)

        unique_order: tuple[int, ...]
        count = len(unique_routes)
        if count <= 12:
            states: dict[tuple[int, int], tuple[int, tuple[int, ...]]] = {
                (1 << index, index): (len(route), (index,))
                for index, route in enumerate(unique_routes)
            }
            for mask in range(1, 1 << count):
                for last in range(count):
                    current = states.get((mask, last))
                    if current is None:
                        continue
                    cost, path = current
                    for following in range(count):
                        bit = 1 << following
                        if mask & bit:
                            continue
                        candidate = (
                            cost + len(unique_routes[following] - unique_routes[last]),
                            (*path, following),
                        )
                        key = (mask | bit, following)
                        previous = states.get(key)
                        if previous is None or candidate < previous:
                            states[key] = candidate
            complete = (1 << count) - 1
            unique_order = min(states[(complete, last)] for last in range(count))[1]
        else:
            order: list[int] = []
            remaining = set(range(count))
            active: frozenset[int] = frozenset()
            while remaining:
                index = min(
                    remaining,
                    key=lambda candidate: (
                        len(unique_routes[candidate] - active),
                        candidate,
                    ),
                )
                order.append(index)
                remaining.remove(index)
                active = unique_routes[index]
            unique_order = tuple(order)

        return tuple(row for index in unique_order for row in rows_by_route[index])

    def _execute_flat(
        self, hidden: torch.Tensor, *, layer: int
    ) -> tuple[torch.Tensor, MlpPilotSparseTrace]:
        if not self.supports_layer(layer):
            raise KeyError(f"no sparse MLP model for layer {layer}")
        model: MlpPilotLayerModel = self._router_models[layer]
        affine: MlpPilotAffineLayer = self._affine_models[layer]
        selected_block_count = (
            model.selected_block_count
            if self.online_controller is None
            else self.online_controller.selected_block_count(layer=layer)
        )
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 2
            or hidden.shape[1] != affine.output_dimension
            or not hidden.is_floating_point()
            or not bool(torch.isfinite(hidden).all())
        ):
            raise ValueError("hidden rows differ from the sparse MLP input ABI")
        compute_hidden = hidden.to(
            device=self.weight_pager.device,
            dtype=self.weight_pager.compute_dtype,
        )
        base = self._base(layer)
        pilots = model.pilot_neuron_indices()
        if self.pilot_pager is None:
            gate_pilot_weight = self._rows(
                self.weight_pager,
                f"{base}.gate_proj.weight",
                pilots,
                columns=affine.output_dimension,
            )
            up_pilot_weight = self._rows(
                self.weight_pager,
                f"{base}.up_proj.weight",
                pilots,
                columns=affine.output_dimension,
            )
            down_pilot_weight = self._rows(
                self.down_transpose_pager,
                self.transpose_name(layer),
                pilots,
                columns=affine.output_dimension,
            )
        else:
            packed = np.arange(len(pilots), dtype=np.int64)
            gate_pilot_weight = self._rows(
                self.pilot_pager,
                self.pilot_name(layer, "gate"),
                packed,
                columns=affine.output_dimension,
            )
            up_pilot_weight = self._rows(
                self.pilot_pager,
                self.pilot_name(layer, "up"),
                packed,
                columns=affine.output_dimension,
            )
            down_pilot_weight = self._rows(
                self.pilot_pager,
                self.pilot_name(layer, "down_transpose"),
                packed,
                columns=affine.output_dimension,
            )
        try:
            gate_pilot = F.linear(compute_hidden, gate_pilot_weight)
            up_pilot = F.linear(compute_hidden, up_pilot_weight)
        finally:
            del gate_pilot_weight, up_pilot_weight
        pilot_activation = swiglu(gate_pilot, up_pilot)
        gate_pilot_array = (
            gate_pilot.detach().to(device="cpu", dtype=torch.float64).numpy()
        )
        up_pilot_array = up_pilot.detach().to(device="cpu", dtype=torch.float64).numpy()
        scores = (
            model.score_pilot_arrays(gate_pilot_array, up_pilot_array)
            if self.online_controller is None
            else self.online_controller.score_pilot_arrays(
                layer=layer,
                gate=gate_pilot_array,
                up=up_pilot_array,
            )
        )
        if self.online_controller is not None:
            selected_block_count = self.online_controller.route_block_count(
                layer=layer,
                scores=np.asarray(scores, dtype=np.float64),
            )
        selected = np.argsort(-scores, axis=1, kind="stable")[
            :, :selected_block_count
        ].astype(np.int64, copy=False)
        pilot_set = set(int(row) for row in pilots)
        extra_by_row: list[np.ndarray] = []
        for row, blocks in enumerate(selected):
            if self.pilot_pager is None:
                extra = np.asarray(
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
                expected = selected_block_count * (model.block_size - model.pilot_count)
            else:
                extra = np.concatenate(
                    [
                        np.arange(
                            int(block) * model.block_size,
                            (int(block) + 1) * model.block_size,
                        )
                        for block in blocks
                    ]
                ).astype(np.int64, copy=False)
                expected = selected_block_count * model.block_size
            if len(extra) != expected or len(set(extra.tolist())) != len(extra):
                raise MlpPilotSparseRuntimeError("router emitted invalid block ranges")
            extra_by_row.append(extra)

        union_extra = np.asarray(
            sorted({int(neuron) for extra in extra_by_row for neuron in extra}),
            dtype=np.int64,
        )
        per_row_dynamic = selected_block_count * (
            model.block_size - model.pilot_count
            if self.pilot_pager is None
            else model.block_size
        )
        if self.online_controller is not None:
            estimated_orders = tuple(
                (
                    tuple(sorted(int(block) for block in blocks))
                    if self.pilot_pager is None
                    else tuple(int(block) for block in blocks)
                )
                for blocks in selected
            )
            estimated_routes = tuple(frozenset(row) for row in estimated_orders)
            estimated_order = self._route_processing_order(estimated_orders)
            active_route: frozenset[int] = frozenset()
            planned_blocks = 0
            for row in estimated_order:
                planned_blocks += len(estimated_routes[row] - active_route)
                active_route = estimated_routes[row]
            dtype_bytes = int(
                torch.empty(
                    (), dtype=self.down_transpose_pager.compute_dtype
                ).element_size()
            )
            route_bytes = per_row_dynamic * affine.output_dimension * dtype_bytes
            assembled_bytes = 0 if selected_block_count == 1 else route_bytes
            same_pager_pilots = (
                len(pilots) * affine.output_dimension * dtype_bytes
                if self.pilot_pager is None
                else 0
            )
            resident_limit = getattr(
                self.down_transpose_pager, "max_resident_bytes", None
            )
            cache_fits = not (
                isinstance(resident_limit, int)
                and not isinstance(resident_limit, bool)
                and route_bytes + assembled_bytes + same_pager_pilots > resident_limit
            )
            requested_blocks = len(hidden) * selected_block_count
            use_cache = planned_blocks < requested_blocks and cache_fits
            down_rows_per_block = (
                model.block_size - model.pilot_count
                if self.pilot_pager is None
                else model.block_size
            )
            estimated_down_rows = (
                planned_blocks * down_rows_per_block
                if use_cache
                else len(hidden) * per_row_dynamic
            )
            estimated_rows = (
                3 * len(pilots) + 2 * len(union_extra) + estimated_down_rows
            )
            estimated_fraction = estimated_rows / (3 * model.intermediate_dimension)
            if (
                estimated_fraction
                > self.online_controller.width_config.max_sparse_transport_fraction
            ):
                raise MlpPilotNonBeneficialRoute(
                    layer=layer,
                    estimated_fraction=estimated_fraction,
                )
        gate_extra = self._project_union_rows(
            compute_hidden,
            name=f"{base}.gate_proj.weight",
            extra_by_row=extra_by_row,
            union_ids=union_extra,
            chunk_rows=per_row_dynamic,
            columns=affine.output_dimension,
        )
        up_extra = self._project_union_rows(
            compute_hidden,
            name=f"{base}.up_proj.weight",
            extra_by_row=extra_by_row,
            union_ids=union_extra,
            chunk_rows=per_row_dynamic,
            columns=affine.output_dimension,
        )

        selected_blocks = tuple(
            tuple(int(block) for block in blocks) for blocks in selected
        )
        down_block_orders = tuple(
            tuple(sorted(blocks)) if self.pilot_pager is None else blocks
            for blocks in selected_blocks
        )
        down_ids_by_block = {
            block: np.asarray(
                [
                    neuron
                    for neuron in range(
                        block * model.block_size,
                        (block + 1) * model.block_size,
                    )
                    if self.pilot_pager is not None or neuron not in pilot_set
                ],
                dtype=np.int64,
            )
            for blocks in down_block_orders
            for block in blocks
        }
        for blocks, extra in zip(down_block_orders, extra_by_row, strict=True):
            reconstructed = tuple(
                int(neuron) for block in blocks for neuron in down_ids_by_block[block]
            )
            if reconstructed != tuple(int(neuron) for neuron in extra):
                raise MlpPilotSparseRuntimeError(
                    "down route order differs from its activation order"
                )

        routes = tuple(frozenset(blocks) for blocks in down_block_orders)
        processing_order = self._route_processing_order(down_block_orders)
        dtype_bytes = int(
            torch.empty(
                (), dtype=self.down_transpose_pager.compute_dtype
            ).element_size()
        )
        route_bytes = per_row_dynamic * affine.output_dimension * dtype_bytes
        same_pager_pilot_bytes = (
            len(pilots) * affine.output_dimension * dtype_bytes
            if self.pilot_pager is None
            else 0
        )
        assembled_route_bytes = 0 if selected_block_count == 1 else route_bytes
        resident_required = route_bytes + assembled_route_bytes + same_pager_pilot_bytes
        resident_limit = getattr(
            self.down_transpose_pager,
            "max_resident_bytes",
            None,
        )
        active: frozenset[int] = frozenset()
        planned_loaded_blocks = 0
        for row in processing_order:
            planned_loaded_blocks += len(routes[row] - active)
            active = routes[row]
        requested_blocks = len(hidden) * selected_block_count
        cache_reuses_blocks = planned_loaded_blocks < requested_blocks
        cache_fits = not (
            isinstance(resident_limit, int)
            and not isinstance(resident_limit, bool)
            and resident_required > resident_limit
        )
        use_down_cache = cache_reuses_blocks and cache_fits

        outputs: list[torch.Tensor | None] = [None] * len(hidden)
        down_loaded_rows = 0

        def execute_down_row(row: int, weight: torch.Tensor) -> torch.Tensor:
            extra_activation = swiglu(gate_extra[row], up_extra[row])
            if self.pilot_pager is not None:
                extra_activation = extra_activation.clone()
                for block_index, block in enumerate(selected_blocks[row]):
                    for offset in model.pilot_offsets[block]:
                        extra_activation[:, block_index * model.block_size + offset] = 0
            pilot_output = pilot_activation[row : row + 1] @ down_pilot_weight
            return pilot_output + extra_activation @ weight

        if not use_down_cache:
            for row, extra in enumerate(extra_by_row):
                down_extra_weight = self._rows(
                    self.down_transpose_pager,
                    self.transpose_name(layer),
                    extra,
                    columns=affine.output_dimension,
                )
                down_loaded_rows += len(extra)
                outputs[row] = execute_down_row(row, down_extra_weight)
                del down_extra_weight
        else:
            down_cache: dict[int, torch.Tensor] = {}
            for row in processing_order:
                blocks = down_block_orders[row]
                active = routes[row]
                for cached in tuple(down_cache):
                    if cached not in active:
                        del down_cache[cached]
                for block in blocks:
                    if block in down_cache:
                        continue
                    ids = down_ids_by_block[block]
                    down_cache[block] = self._rows(
                        self.down_transpose_pager,
                        self.transpose_name(layer),
                        ids,
                        columns=affine.output_dimension,
                    )
                    down_loaded_rows += len(ids)
                down_extra_weight = (
                    down_cache[blocks[0]]
                    if len(blocks) == 1
                    else torch.cat(
                        tuple(down_cache[block] for block in blocks),
                        dim=0,
                    )
                )
                outputs[row] = execute_down_row(row, down_extra_weight)
                del down_extra_weight
        if any(output is None for output in outputs):  # pragma: no cover - permutation.
            raise MlpPilotSparseRuntimeError("down route cache lost an output row")
        sparse = torch.cat(
            [output for output in outputs if output is not None],
            dim=0,
        )
        corrected = affine.apply(sparse)
        if self.online_controller is not None:
            corrected = self.online_controller.correct_sparse_output(
                layer=layer,
                sparse_output=corrected,
            )
        corrected = corrected.to(dtype=self.output_dtype)
        trace = MlpPilotSparseTrace(
            layer=layer,
            row_count=len(hidden),
            selected_blocks=tuple(
                tuple(int(block) for block in row) for row in selected
            ),
            selected_neuron_count=(
                model.block_count * model.pilot_count
                + selected_block_count * (model.block_size - model.pilot_count)
            ),
            intermediate_dimension=model.intermediate_dimension,
            source_weight_rows=(
                3 * len(pilots) + 2 * len(union_extra) + down_loaded_rows
            ),
            full_weight_rows=3 * model.intermediate_dimension,
            dynamic_requested_rows=len(hidden) * per_row_dynamic,
            dynamic_unique_rows=len(union_extra),
            down_requested_rows=len(hidden) * per_row_dynamic,
            down_loaded_rows=down_loaded_rows,
            output_dtype=str(corrected.dtype).removeprefix("torch."),
            range_mode=(
                "scattered-pilot+union-target+route-cache-down"
                if self.pilot_pager is None
                else "consolidated-pilot+union-target+route-cache-down"
            ),
        )
        if self.online_controller is not None:
            self.online_controller.record_sparse(
                layer=layer,
                row_count=len(hidden),
                selected_block_count=selected_block_count,
            )
        return corrected, trace

    def execute(
        self, hidden: torch.Tensor, *, layer: int
    ) -> tuple[torch.Tensor, MlpPilotSparseTrace]:
        if not isinstance(hidden, torch.Tensor) or hidden.ndim < 1:
            raise TypeError("hidden must be a tensor with a feature axis")
        shape = tuple(hidden.shape)
        flat = hidden.reshape(-1, shape[-1])
        output, trace = self._execute_flat(flat, layer=layer)
        result = output.reshape((*shape[:-1], output.shape[-1]))
        self.last_trace = trace
        return result, trace

    def execute_many(
        self, hidden: Sequence[torch.Tensor], *, layer: int
    ) -> tuple[tuple[torch.Tensor, ...], MlpPilotSparseTrace]:
        values = tuple(hidden)
        if not values:
            raise ValueError("execute_many requires hidden tensors")
        if any(
            not isinstance(value, torch.Tensor) or value.ndim < 1 for value in values
        ):
            raise ValueError("hidden values must be tensors with a feature axis")
        widths = {value.shape[-1] for value in values}
        if len(widths) != 1:
            raise ValueError("hidden tensors disagree on their feature ABI")
        shapes = [tuple(value.shape) for value in values]
        counts = [math.prod(shape[:-1]) for shape in shapes]
        flat = torch.cat(
            [
                value.reshape(-1, shape[-1])
                for value, shape in zip(values, shapes, strict=True)
            ],
            dim=0,
        )
        output, trace = self._execute_flat(flat, layer=layer)
        rows = []
        offset = 0
        for shape, count in zip(shapes, counts, strict=True):
            rows.append(output[offset : offset + count].reshape((*shape[:-1], -1)))
            offset += count
        self.last_trace = trace
        return tuple(rows), trace


__all__ = [
    "PILOT_SPARSE_RUNTIME_SCHEMA",
    "PILOT_TRANSPOSE_ENTRY_SCHEMA",
    "PILOT_TRANSPOSE_MANIFEST_SCHEMA",
    "MlpPilotSparseExecutor",
    "MlpPilotNonBeneficialRoute",
    "MlpPilotSparseRuntimeError",
    "MlpPilotSparseTrace",
    "MlpPilotTransposeEntry",
    "MlpPilotTransposeManifest",
    "SelectedRowPager",
]
