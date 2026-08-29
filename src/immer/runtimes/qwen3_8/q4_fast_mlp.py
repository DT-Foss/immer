"""Pilot-routed sparse MLP execution directly on the packed Qwen Q4 bank."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import threading
from typing import Mapping, Sequence

import torch

from ..ooe.mlp_pilot_residual import MlpPilotAffineFit, MlpPilotAffineLayer
from ..ooe.mlp_pilot_router import MlpPilotLayerModel, MlpPilotRouterFit
from ..ooe.mlp_pilot_runtime import MlpPilotSparseTrace
from ..ooe.mlp_pilot_weight_only import MlpPilotSparseDecision
from ..ooe.mlp_pilot_weight_only import (
    MlpPilotIdentityAffinePlan,
    MlpPilotWeightOnlyPlan,
)
from ..ooe.identity import canonical_json_bytes
from .config import Qwen38Config
from .fast_mlp import Qwen38FastMlpPaths, _stable_read
from .q4 import Q4Bank, Q4_BLOCK_SIZE


PACKED_FAST_MLP_SCHEMA = "immer.qwen3.8-packed-q4-fast-mlp/v1"
PACKED_FAST_MLP_MARKOV_SCHEMA = "immer.qwen3.8-packed-fast-mlp-markov/v1"


class Qwen38PackedFastMlpError(RuntimeError):
    """The packed Q4 bank and sparse MLP route do not share one ABI."""


class PackedFastMlpMarkovController:
    """Persistent first-order route-overlap controller for k32/k48/k64."""

    def __init__(
        self,
        path: Path,
        *,
        identity: Mapping[str, object],
        layers: Sequence[int],
        maximum_width: int,
    ) -> None:
        self.identity = dict(identity)
        requested_path = path.expanduser().absolute()
        namespace = hashlib.sha256(canonical_json_bytes(self.identity)).hexdigest()[:16]
        suffix = requested_path.suffix or ".json"
        self.path = requested_path.with_name(
            f"{requested_path.stem}.packed-{namespace}{suffix}"
        )
        self.layers = tuple(layers)
        self.maximum_width = maximum_width
        actions = {width for width in (32, 48, 64) if width <= maximum_width}
        actions.add(maximum_width)
        self.actions = tuple(sorted(actions))
        self._state = {
            layer: {"overlap_ema": 0.0, "previous": (), "support": 0}
            for layer in self.layers
        }
        self._width_counts = {width: 0 for width in self.actions}
        self._dirty = False
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        raw = self.path.read_bytes()
        try:
            document = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise Qwen38PackedFastMlpError("packed Markov state is corrupt") from exc
        if (
            not isinstance(document, dict)
            or document.get("schema") != PACKED_FAST_MLP_MARKOV_SCHEMA
            or document.get("identity") != self.identity
            or canonical_json_bytes(document) != raw
            or not isinstance(document.get("layers"), dict)
            or not isinstance(document.get("width_counts"), dict)
        ):
            raise Qwen38PackedFastMlpError("packed Markov state identity changed")
        restored = {}
        for layer in self.layers:
            row = document["layers"].get(str(layer))
            if not isinstance(row, dict) or set(row) != {
                "overlap_ema",
                "previous",
                "support",
            }:
                raise Qwen38PackedFastMlpError("packed Markov layer state is invalid")
            ema = row["overlap_ema"]
            previous = row["previous"]
            support = row["support"]
            if (
                isinstance(ema, bool)
                or not isinstance(ema, (int, float))
                or not math.isfinite(float(ema))
                or not 0.0 <= float(ema) <= 1.0
                or not isinstance(previous, list)
                or any(
                    isinstance(block, bool) or not isinstance(block, int) or block < 0
                    for block in previous
                )
                or previous != sorted(set(previous))
                or isinstance(support, bool)
                or not isinstance(support, int)
                or support < 0
            ):
                raise Qwen38PackedFastMlpError("packed Markov layer values are invalid")
            restored[layer] = {
                "overlap_ema": float(ema),
                "previous": tuple(previous),
                "support": support,
            }
        counts = {}
        for width in self.actions:
            value = document["width_counts"].get(str(width), 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise Qwen38PackedFastMlpError("packed Markov counters are invalid")
            counts[width] = value
        self._state = restored
        self._width_counts = counts

    def width(self, layer: int) -> int:
        with self._lock:
            row = self._state[layer]
            if row["support"] < 1:
                width = self.actions[-1]
            elif row["overlap_ema"] >= 0.38:
                width = self.actions[0]
            elif row["overlap_ema"] >= 0.22:
                width = self.actions[min(1, len(self.actions) - 1)]
            else:
                width = self.actions[-1]
            self._width_counts[width] += 1
            self._dirty = True
            return width

    def observe(self, layer: int, selected: Sequence[int]) -> None:
        current = tuple(sorted(set(int(block) for block in selected)))
        if not current:
            raise Qwen38PackedFastMlpError("packed Markov action is empty")
        with self._lock:
            row = self._state[layer]
            previous = set(row["previous"])
            if previous:
                active = set(current)
                overlap = len(previous & active) / len(previous | active)
                row["overlap_ema"] = (
                    overlap
                    if row["support"] == 0
                    else 0.8 * row["overlap_ema"] + 0.2 * overlap
                )
                row["support"] += 1
            row["previous"] = current
            self._dirty = True

    def metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "markov_learned_layers": sum(
                    row["support"] > 0 for row in self._state.values()
                ),
                "markov_transitions": sum(
                    row["support"] for row in self._state.values()
                ),
                **{
                    f"markov_width_{width}": count
                    for width, count in self._width_counts.items()
                },
            }

    def flush(self) -> None:
        with self._lock:
            if not self._dirty:
                return
            document = {
                "identity": self.identity,
                "layers": {
                    str(layer): {
                        "overlap_ema": row["overlap_ema"],
                        "previous": list(row["previous"]),
                        "support": row["support"],
                    }
                    for layer, row in self._state.items()
                },
                "schema": PACKED_FAST_MLP_MARKOV_SCHEMA,
                "width_counts": {
                    str(width): count for width, count in self._width_counts.items()
                },
            }
            data = canonical_json_bytes(document)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.parent / (
                f".{self.path.name}.{secrets.token_hex(8)}.tmp"
            )
            descriptor: int | None = None
            try:
                descriptor = os.open(
                    temporary,
                    os.O_CREAT
                    | os.O_EXCL
                    | os.O_WRONLY
                    | int(getattr(os, "O_CLOEXEC", 0)),
                    0o600,
                )
                offset = 0
                while offset < len(data):
                    written = os.write(descriptor, data[offset:])
                    if written <= 0:
                        raise OSError("short packed Markov state write")
                    offset += written
                os.fsync(descriptor)
                os.close(descriptor)
                descriptor = None
                os.replace(temporary, self.path)
                self._dirty = False
            finally:
                if descriptor is not None:
                    os.close(descriptor)
                temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class Qwen38PackedFastMlpReceipt:
    model_pin_sha256: str
    router_fit_sha256: str
    affine_fit_sha256: str
    weights_index_sha256: str
    q4_manifest_sha256: str
    fitted_layers: tuple[int, ...]
    active_layers: tuple[int, ...]
    selected_neuron_fraction_by_layer: tuple[tuple[int, float], ...]
    transport_fraction_by_layer: tuple[tuple[int, float], ...]
    initialization: str
    selected_block_count: int
    markov_state_persistent: bool
    markov_width_actions: tuple[int, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "active_layers": list(self.active_layers),
            "affine_fit_sha256": self.affine_fit_sha256,
            "auxiliary_payload_bytes": 0,
            "execution": "packed-q4-pilot-routed-sparse-mlp",
            "fitted_layers": list(self.fitted_layers),
            "initialization": self.initialization,
            "model_pin_sha256": self.model_pin_sha256,
            "markov_state_persistent": self.markov_state_persistent,
            "markov_width_actions": list(self.markov_width_actions),
            "q4_manifest_sha256": self.q4_manifest_sha256,
            "router_fit_sha256": self.router_fit_sha256,
            "schema": PACKED_FAST_MLP_SCHEMA,
            "selected_block_count": self.selected_block_count,
            "selected_neuron_fraction_by_layer": {
                str(layer): fraction
                for layer, fraction in self.selected_neuron_fraction_by_layer
            },
            "transport_fraction_by_layer": {
                str(layer): fraction
                for layer, fraction in self.transport_fraction_by_layer
            },
            "weights_index_sha256": self.weights_index_sha256,
        }


class Qwen38PackedFastMlpExecutor:
    """Route MLP blocks with pilots, then execute only those packed Q4 blocks."""

    def __init__(
        self,
        router: MlpPilotRouterFit | MlpPilotWeightOnlyPlan,
        affine: MlpPilotAffineFit | MlpPilotIdentityAffinePlan,
        bank: Q4Bank,
        *,
        active_layers: Sequence[int],
        output_dtype: torch.dtype,
        selected_block_count: int | None = None,
        markov_state_path: Path | None = None,
    ) -> None:
        self.router_fit = router
        self.affine_fit = affine
        self.bank = bank
        self.output_dtype = output_dtype
        self.online_controller = None
        self._models = {row.layer: row for row in router.models}
        self._affine = {row.layer: row for row in affine.models}
        self._active_layers = frozenset(active_layers)
        selected_counts = {row.selected_block_count for row in router.models}
        if len(selected_counts) != 1:
            raise Qwen38PackedFastMlpError("route models disagree on their width")
        stored_count = next(iter(selected_counts))
        self.selected_block_count = (
            stored_count if selected_block_count is None else selected_block_count
        )
        if (
            isinstance(self.selected_block_count, bool)
            or not isinstance(self.selected_block_count, int)
            or self.selected_block_count < 1
            or any(
                self.selected_block_count >= row.block_count for row in router.models
            )
        ):
            raise Qwen38PackedFastMlpError("packed sparse route width is invalid")
        self.last_trace: MlpPilotSparseTrace | None = None
        self._pending_width: dict[int, int] = {}
        self.markov_controller = (
            None
            if markov_state_path is None
            else PackedFastMlpMarkovController(
                markov_state_path,
                identity={
                    "active_layers": list(self.active_layers),
                    "maximum_width": self.selected_block_count,
                    "q4_manifest_sha256": self.bank.identity["manifest_sha256"],
                    "router_fit_sha256": self.router_fit.sha256,
                },
                layers=self.active_layers,
                maximum_width=self.selected_block_count,
            )
        )
        self._metrics = {
            "full_equivalent_logical_weight_bytes": 0,
            "logical_weight_bytes": 0,
            "packed_sparse_calls": 0,
            "packed_sparse_rows": 0,
            "q4_weight_bytes_saved": 0,
            "selected_blocks": 0,
        }

    @property
    def active_layers(self) -> tuple[int, ...]:
        return tuple(sorted(self._active_layers))

    @staticmethod
    def _base(layer: int) -> str:
        return f"model.language_model.layers.{layer}.mlp"

    def supports_layer(self, layer: int) -> bool:
        return layer in self._active_layers

    def decision(self, *, layer: int, row_count: int) -> MlpPilotSparseDecision:
        supported = self.supports_layer(layer)
        count = (
            0
            if not supported
            else (
                self.selected_block_count
                if self.markov_controller is None
                else self.markov_controller.width(layer)
            )
        )
        if supported:
            self._pending_width[layer] = count
        return MlpPilotSparseDecision(
            layer=layer,
            row_count=row_count,
            use_sparse=supported,
            reason=(
                "packed-q4-markov-route"
                if supported and self.markov_controller is not None
                else "packed-q4-route" if supported else "unsupported-layer"
            ),
            confirmed_rows=0,
            capture_ema=0.0,
            sparse_waves_since_confirmation=0,
            selected_block_count=count,
            max_selected_block_count=count,
        )

    def metrics(self) -> dict[str, int]:
        return {
            **self._metrics,
            **(
                {}
                if self.markov_controller is None
                else self.markov_controller.metrics()
            ),
        }

    def snapshot_identity(
        self, *, transport_neutral: bool = False
    ) -> dict[str, object]:
        identity: dict[str, object] = {
            "affine_fit_sha256": self.affine_fit.sha256,
            "layers": list(self.active_layers),
            "output_dtype": str(self.output_dtype).removeprefix("torch."),
            "q4_manifest_sha256": self.bank.identity["manifest_sha256"],
            "range_mode": "packed-q4-pilot+selected-blocks",
            "router_fit_sha256": self.router_fit.sha256,
            "schema": PACKED_FAST_MLP_SCHEMA,
            "selected_block_count": self.selected_block_count,
            "markov_width_policy": (
                "off"
                if self.markov_controller is None
                else "route-overlap-k32-k48-k64/v2"
            ),
        }
        if not transport_neutral:
            identity["bank_root"] = str(self.bank.root)
        return identity

    def _execute_flat(
        self,
        hidden: torch.Tensor,
        *,
        layer: int,
    ) -> tuple[torch.Tensor, MlpPilotSparseTrace]:
        if not self.supports_layer(layer):
            raise KeyError(f"no packed sparse MLP route for layer {layer}")
        model: MlpPilotLayerModel = self._models[layer]
        affine: MlpPilotAffineLayer = self._affine[layer]
        if (
            hidden.ndim != 2
            or hidden.shape[1] != affine.output_dimension
            or not hidden.is_floating_point()
            or not bool(torch.isfinite(hidden).all())
        ):
            raise ValueError("hidden rows differ from the packed sparse MLP ABI")
        if model.block_size % Q4_BLOCK_SIZE:
            raise Qwen38PackedFastMlpError(
                "pilot block size must be divisible by the Q4 block size"
            )

        base = self._base(layer)
        gate_name = f"{base}.gate_proj.weight"
        up_name = f"{base}.up_proj.weight"
        down_name = f"{base}.down_proj.weight"
        compute = hidden.detach().to(device="cpu", dtype=torch.float32).contiguous()
        pilots = tuple(int(row) for row in model.pilot_neuron_indices())
        before = int(self.bank.metrics()["logical_weight_bytes"])
        route_width = self._pending_width.pop(layer, self.selected_block_count)
        corrected, selected_tensor = self.bank.sparse_mlp(
            compute,
            (gate_name, up_name, down_name),
            pilot_ids=model.pilot_neuron_indices().reshape(
                model.block_count, model.pilot_count
            ),
            coefficients=model.coefficients,
            block_size=model.block_size,
            selected_block_count=route_width,
            affine_scale=affine.scale,
            affine_bias=affine.bias,
            output_dtype=self.output_dtype,
        )
        selected = tuple(
            tuple(int(block) for block in row) for row in selected_tensor.tolist()
        )
        if self.markov_controller is not None:
            for row in selected:
                self.markov_controller.observe(layer, row)
        pilot_set = frozenset(pilots)
        extra_neurons = tuple(
            tuple(
                neuron
                for block in row
                for neuron in range(
                    int(block) * model.block_size,
                    (int(block) + 1) * model.block_size,
                )
                if neuron not in pilot_set
            )
            for row in selected
        )
        union_extra = tuple(
            sorted({neuron for row in extra_neurons for neuron in row})
        )
        actual = int(self.bank.metrics()["logical_weight_bytes"]) - before
        full = sum(
            self.bank.entries[name].payload_bytes
            for name in (gate_name, up_name, down_name)
        )
        saved = max(0, full - actual)
        selected_count = route_width * model.block_size
        rows = len(hidden)
        dynamic_per_row = route_width * (
            model.block_size - model.pilot_count
        )
        requested = rows * dynamic_per_row
        trace = MlpPilotSparseTrace(
            layer=layer,
            row_count=rows,
            selected_blocks=selected,
            selected_neuron_count=selected_count,
            intermediate_dimension=model.intermediate_dimension,
            source_weight_rows=(
                2 * len(pilots) + 3 * selected_count
            ),
            full_weight_rows=3 * model.intermediate_dimension,
            dynamic_requested_rows=requested,
            dynamic_unique_rows=len(union_extra),
            down_requested_rows=rows * selected_count,
            down_loaded_rows=rows * selected_count,
            output_dtype=str(corrected.dtype).removeprefix("torch."),
            range_mode="packed-q4-pilot+selected-blocks",
        )
        self._metrics["full_equivalent_logical_weight_bytes"] += full
        self._metrics["logical_weight_bytes"] += actual
        self._metrics["packed_sparse_calls"] += 1
        self._metrics["packed_sparse_rows"] += rows
        self._metrics["q4_weight_bytes_saved"] += saved
        self._metrics["selected_blocks"] += rows * route_width
        return corrected, trace

    def execute(
        self,
        hidden: torch.Tensor,
        *,
        layer: int,
    ) -> tuple[torch.Tensor, MlpPilotSparseTrace]:
        shape = tuple(hidden.shape)
        output, trace = self._execute_flat(hidden.reshape(-1, shape[-1]), layer=layer)
        result = output.reshape(*shape[:-1], output.shape[-1])
        self.last_trace = trace
        return result, trace

    def execute_many(
        self,
        hidden: Sequence[torch.Tensor],
        *,
        layer: int,
    ) -> tuple[tuple[torch.Tensor, ...], MlpPilotSparseTrace]:
        values = tuple(hidden)
        if not values:
            raise ValueError("execute_many requires hidden tensors")
        shapes = [tuple(value.shape) for value in values]
        counts = [value.numel() // value.shape[-1] for value in values]
        flat = torch.cat(
            [value.reshape(-1, value.shape[-1]) for value in values], dim=0
        )
        output, trace = self._execute_flat(flat, layer=layer)
        rows = []
        start = 0
        for shape, count in zip(shapes, counts, strict=True):
            rows.append(output[start : start + count].reshape(*shape[:-1], -1))
            start += count
        self.last_trace = trace
        return tuple(rows), trace


class Qwen38PackedFastMlpMount:
    def __init__(
        self,
        executor: Qwen38PackedFastMlpExecutor,
        receipt: Qwen38PackedFastMlpReceipt,
    ) -> None:
        self.executor = executor
        self.receipt = receipt

    def metrics(self) -> dict[str, int]:
        if self.executor.markov_controller is not None:
            self.executor.markov_controller.flush()
        metrics = self.executor.metrics()
        return {
            **metrics,
            "pilot_logical_weight_bytes": 0,
            "pilot_source_body_bytes": 0,
            "source_body_bytes": 0,
            "transpose_logical_weight_bytes": 0,
            "transpose_source_body_bytes": 0,
        }

    def close(self) -> None:
        if self.executor.markov_controller is not None:
            self.executor.markov_controller.flush()


def open_qwen38_packed_fast_mlp(
    *,
    paths: Qwen38FastMlpPaths,
    bank: Q4Bank,
    config: Qwen38Config,
    weights_root: Path,
    active_layers: Sequence[int] | None,
    output_dtype: torch.dtype,
    selected_block_count: int | None = None,
    markov_state_path: Path | None = None,
) -> Qwen38PackedFastMlpMount:
    """Mount only the small route/affine files; all weights stay in Q4."""

    plan_path = paths.analysis_root / "weight-only-plan.json"
    if plan_path.exists() and not plan_path.is_symlink():
        fit: MlpPilotRouterFit | MlpPilotWeightOnlyPlan = (
            MlpPilotWeightOnlyPlan.from_bytes(_stable_read(plan_path))
        )
        affine: MlpPilotAffineFit | MlpPilotIdentityAffinePlan = fit.affine_fit
        initialization = fit.initializer
    else:
        fit = MlpPilotRouterFit.from_bytes(
            _stable_read(paths.analysis_root / "fit.json")
        )
        affine = MlpPilotAffineFit.from_bytes(
            _stable_read(paths.analysis_root / "affine-fit.json")
        )
        initialization = "contextual-pilot-fit/v1"
    index = weights_root / "model.safetensors.index.json"
    index_sha256 = hashlib.sha256(_stable_read(index)).hexdigest()
    models = {row.layer: row for row in fit.models}
    affine_models = {row.layer: row for row in affine.models}
    if (
        affine.router_fit_sha256 != fit.sha256
        or affine.model_pin_sha256 != fit.model_pin_sha256
        or affine.weights_index_sha256 != index_sha256
        or set(models) != set(affine_models)
        or any(row.intermediate_dimension != config.intermediate_size for row in models.values())
        or any(row.output_dimension != config.dim for row in affine_models.values())
        or (
            isinstance(fit, MlpPilotWeightOnlyPlan)
            and (
                fit.n_layers != config.n_layers
                or fit.hidden_dimension != config.dim
                or fit.weights_index_sha256 != index_sha256
                or fit.repo_id != bank.manifest["body"]["source"]["repo_id"]
                or fit.revision != bank.manifest["body"]["source"]["revision"]
                or fit.bundle_manifest_sha256
                != bank.manifest["body"]["source"]["bundle_manifest_sha256"]
                or fit.layout_fingerprint
                != bank.manifest["body"]["source"]["layout_fingerprint"]
            )
        )
    ):
        raise Qwen38PackedFastMlpError("packed sparse route differs from local Qwen")
    fitted = tuple(sorted(models))
    selected = fitted if active_layers is None else tuple(active_layers)
    if (
        not selected
        or selected != tuple(sorted(set(selected)))
        or any(layer not in models for layer in selected)
    ):
        raise Qwen38PackedFastMlpError(
            "active packed sparse layers must be a sorted fitted subset"
        )
    for layer in selected:
        model = models[layer]
        if model.block_size % Q4_BLOCK_SIZE:
            raise Qwen38PackedFastMlpError("route block size differs from Q4")
        base = f"model.language_model.layers.{layer}.mlp"
        expected: Mapping[str, tuple[int, int]] = {
            f"{base}.gate_proj.weight": (config.intermediate_size, config.dim),
            f"{base}.up_proj.weight": (config.intermediate_size, config.dim),
            f"{base}.down_proj.weight": (config.dim, config.intermediate_size),
        }
        for name, shape in expected.items():
            entry = bank.entries.get(name)
            if entry is None or entry.shape != shape:
                raise Qwen38PackedFastMlpError(
                    f"packed sparse tensor ABI changed: {name}"
                )
    executor = Qwen38PackedFastMlpExecutor(
        fit,
        affine,
        bank,
        active_layers=selected,
        output_dtype=output_dtype,
        selected_block_count=selected_block_count,
        markov_state_path=markov_state_path,
    )
    selected_fractions = tuple(
        (
            layer,
            executor.selected_block_count
            * models[layer].block_size
            / models[layer].intermediate_dimension,
        )
        for layer in selected
    )
    transport_fractions = tuple(
        (
            layer,
            (
                2
                * (
                    len(models[layer].pilot_neuron_indices())
                    + executor.selected_block_count * models[layer].block_size
                )
                + executor.selected_block_count * models[layer].block_size
            )
            / (3 * models[layer].intermediate_dimension),
        )
        for layer in selected
    )
    receipt = Qwen38PackedFastMlpReceipt(
        model_pin_sha256=fit.model_pin_sha256,
        router_fit_sha256=fit.sha256,
        affine_fit_sha256=affine.sha256,
        weights_index_sha256=index_sha256,
        q4_manifest_sha256=str(bank.identity["manifest_sha256"]),
        fitted_layers=fitted,
        active_layers=selected,
        selected_neuron_fraction_by_layer=selected_fractions,
        transport_fraction_by_layer=transport_fractions,
        initialization=initialization,
        selected_block_count=executor.selected_block_count,
        markov_state_persistent=executor.markov_controller is not None,
        markov_width_actions=(
            ()
            if executor.markov_controller is None
            else executor.markov_controller.actions
        ),
    )
    return Qwen38PackedFastMlpMount(executor, receipt)


__all__ = [
    "PACKED_FAST_MLP_SCHEMA",
    "Qwen38PackedFastMlpError",
    "Qwen38PackedFastMlpExecutor",
    "Qwen38PackedFastMlpMount",
    "Qwen38PackedFastMlpReceipt",
    "open_qwen38_packed_fast_mlp",
]
