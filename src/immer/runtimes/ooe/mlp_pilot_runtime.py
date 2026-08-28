"""True selected-row execution for the Qwen MLP pilot and residual Crystals."""

from __future__ import annotations

from dataclasses import dataclass
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


PILOT_SPARSE_RUNTIME_SCHEMA = "immer.qwen-mlp-pilot-sparse-runtime/v1"
PILOT_TRANSPOSE_ENTRY_SCHEMA = "immer.qwen-mlp-pilot-transpose-entry/v1"
PILOT_TRANSPOSE_MANIFEST_SCHEMA = "immer.qwen-mlp-pilot-transpose-manifest/v1"


class MlpPilotSparseRuntimeError(RuntimeError):
    pass


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
    output_dtype: str
    range_mode: str = "scattered-union"

    def __post_init__(self) -> None:
        for field in (
            "layer",
            "row_count",
            "selected_neuron_count",
            "intermediate_dimension",
            "source_weight_rows",
            "full_weight_rows",
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
            or not isinstance(self.output_dtype, str)
            or not self.output_dtype
            or self.range_mode
            not in {"scattered-union", "consolidated-pilot+full-blocks"}
        ):
            raise ValueError("sparse runtime trace is inconsistent")
        object.__setattr__(self, "selected_blocks", blocks)

    @property
    def weight_row_fraction(self) -> float:
        return self.source_weight_rows / self.full_weight_rows

    def to_record(self) -> dict[str, object]:
        return {
            "full_weight_rows": self.full_weight_rows,
            "intermediate_dimension": self.intermediate_dimension,
            "layer": self.layer,
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
        router_fit: MlpPilotRouterFit,
        affine_fit: MlpPilotAffineFit,
        weight_pager: SelectedRowPager,
        down_transpose_pager: SelectedRowPager,
        *,
        pilot_pager: SelectedRowPager | None = None,
        active_layers: Sequence[int] | None = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(router_fit, MlpPilotRouterFit):
            raise TypeError("router_fit must be MlpPilotRouterFit")
        if not isinstance(affine_fit, MlpPilotAffineFit):
            raise TypeError("affine_fit must be MlpPilotAffineFit")
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
                "scattered-union"
                if self.pilot_pager is None
                else "consolidated-pilot+full-blocks"
            ),
            "router_fit_sha256": self.router_fit.sha256,
            "schema": PILOT_SPARSE_RUNTIME_SCHEMA,
        }
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

    def _execute_flat(
        self, hidden: torch.Tensor, *, layer: int
    ) -> tuple[torch.Tensor, MlpPilotSparseTrace]:
        if not self.supports_layer(layer):
            raise KeyError(f"no sparse MLP model for layer {layer}")
        model: MlpPilotLayerModel = self._router_models[layer]
        affine: MlpPilotAffineLayer = self._affine_models[layer]
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
        gate_pilot = F.linear(compute_hidden, gate_pilot_weight)
        up_pilot = F.linear(compute_hidden, up_pilot_weight)
        pilot_activation = swiglu(gate_pilot, up_pilot)
        scores = model.score_pilot_arrays(
            gate_pilot.detach().to(device="cpu", dtype=torch.float64).numpy(),
            up_pilot.detach().to(device="cpu", dtype=torch.float64).numpy(),
        )
        selected = np.argsort(-scores, axis=1, kind="stable")[
            :, : model.selected_block_count
        ].astype(np.int64, copy=False)
        pilot_set = set(int(row) for row in pilots)
        outputs = []
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
                expected = model.selected_block_count * (
                    model.block_size - model.pilot_count
                )
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
                expected = model.selected_block_count * model.block_size
            if len(extra) != expected or len(set(extra.tolist())) != len(extra):
                raise MlpPilotSparseRuntimeError("router emitted invalid block ranges")
            gate_extra_weight = self._rows(
                self.weight_pager,
                f"{base}.gate_proj.weight",
                extra,
                columns=affine.output_dimension,
            )
            up_extra_weight = self._rows(
                self.weight_pager,
                f"{base}.up_proj.weight",
                extra,
                columns=affine.output_dimension,
            )
            down_extra_weight = self._rows(
                self.down_transpose_pager,
                self.transpose_name(layer),
                extra,
                columns=affine.output_dimension,
            )
            gate_extra = F.linear(compute_hidden[row : row + 1], gate_extra_weight)
            up_extra = F.linear(compute_hidden[row : row + 1], up_extra_weight)
            extra_activation = swiglu(gate_extra, up_extra)
            if self.pilot_pager is not None:
                extra_activation = extra_activation.clone()
                for block_index, block in enumerate(blocks):
                    for offset in model.pilot_offsets[int(block)]:
                        extra_activation[:, block_index * model.block_size + offset] = 0
            pilot_output = pilot_activation[row : row + 1] @ down_pilot_weight
            extra_output = extra_activation @ down_extra_weight
            outputs.append(pilot_output + extra_output)
        sparse = torch.cat(outputs, dim=0)
        corrected = affine.apply(sparse).to(dtype=self.output_dtype)
        per_row_dynamic = model.selected_block_count * (
            model.block_size - model.pilot_count
            if self.pilot_pager is None
            else model.block_size
        )
        trace = MlpPilotSparseTrace(
            layer=layer,
            row_count=len(hidden),
            selected_blocks=tuple(
                tuple(int(block) for block in row) for row in selected
            ),
            selected_neuron_count=model.selected_neuron_count,
            intermediate_dimension=model.intermediate_dimension,
            source_weight_rows=3 * (len(pilots) + len(hidden) * per_row_dynamic),
            full_weight_rows=3 * model.intermediate_dimension,
            output_dtype=str(corrected.dtype).removeprefix("torch."),
            range_mode=(
                "scattered-union"
                if self.pilot_pager is None
                else "consolidated-pilot+full-blocks"
            ),
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
    "MlpPilotSparseRuntimeError",
    "MlpPilotSparseTrace",
    "MlpPilotTransposeEntry",
    "MlpPilotTransposeManifest",
    "SelectedRowPager",
]
