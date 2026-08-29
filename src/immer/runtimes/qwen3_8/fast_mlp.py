"""Production mount for IMMER's row-routed Qwen fast MLP artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Mapping, Sequence, cast

import torch

from ...knowledge.streamer import Streamer
from ..deepseek_v4.causal_weights import CausalWeightMount
from ..ooe.identity import canonical_json_bytes, require_sha256
from ..ooe.mlp_pilot_residual import MlpPilotAffineFit
from ..ooe.mlp_pilot_router import MlpPilotRouterFit
from ..ooe.mlp_pilot_runtime import (
    MlpPilotSparseExecutor,
    MlpPilotTransposeManifest,
)
from ..ooe.mlp_pilot_weight_only import (
    WEIGHT_ONLY_INITIALIZER,
    MlpPilotAdaptiveWidthConfig,
    MlpPilotIdentityAffinePlan,
    MlpPilotOnlineController,
    MlpPilotWeightOnlyPlan,
)
from .pager import Qwen38WeightPager
from .config import Qwen38Config

FAST_MLP_MOUNT_SCHEMA = "immer.qwen3.8-fast-mlp-mount/v1"
PILOT_WEIGHT_MANIFEST_SCHEMA = "immer.qwen3.8-mlp-pilot-weight-bank/v1"
PILOT_WEIGHT_MANIFEST_V2_SCHEMA = "immer.qwen3.8-mlp-pilot-weight-bank/v2"
WEIGHT_ONLY_PLAN_NAME = "weight-only-plan.json"
ADAPTIVE_WIDTH_CONFIG_NAME = "adaptive-width.json"


class Qwen38FastMlpError(RuntimeError):
    """The configured fast-MLP artifacts cannot be mounted as one runtime."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stable_read(path: Path, maximum: int = 64 * 1024 * 1024) -> bytes:
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise ValueError("maximum must be a positive integer")
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
            raise Qwen38FastMlpError(f"invalid bounded fast-MLP artifact: {path}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise Qwen38FastMlpError(f"short fast-MLP artifact read: {path}")
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
            raise Qwen38FastMlpError(f"fast-MLP artifact changed while read: {path}")
        return b"".join(chunks)
    except Qwen38FastMlpError:
        raise
    except OSError as exc:
        raise Qwen38FastMlpError(f"cannot read fast-MLP artifact: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _plain_directory(path: Path) -> None:
    try:
        row = path.lstat()
    except OSError as exc:
        raise Qwen38FastMlpError(f"fast-MLP directory is missing: {path}") from exc
    if stat.S_ISLNK(row.st_mode) or not stat.S_ISDIR(row.st_mode):
        raise Qwen38FastMlpError(
            f"fast-MLP root must be a non-symlink directory: {path}"
        )


def _regular_shard(root: Path, name: object, size: object) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
    ):
        raise Qwen38FastMlpError("fast-MLP shard entry is invalid")
    path = root / name
    try:
        row = path.lstat()
    except OSError as exc:
        raise Qwen38FastMlpError(f"fast-MLP shard is missing: {path}") from exc
    if (
        stat.S_ISLNK(row.st_mode)
        or not stat.S_ISREG(row.st_mode)
        or row.st_size != size
    ):
        raise Qwen38FastMlpError(f"fast-MLP shard identity changed: {path}")
    return path


def _stable_sha256(path: Path, expected_size: int) -> str:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
            raise Qwen38FastMlpError(f"fast-MLP shard identity changed: {path}")
        digest = hashlib.sha256()
        remaining = expected_size
        while remaining:
            chunk = os.read(descriptor, min(4 * 1024 * 1024, remaining))
            if not chunk:
                raise Qwen38FastMlpError(f"short fast-MLP shard read: {path}")
            digest.update(chunk)
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
            raise Qwen38FastMlpError(f"fast-MLP shard changed while hashed: {path}")
        return digest.hexdigest()
    except Qwen38FastMlpError:
        raise
    except OSError as exc:
        raise Qwen38FastMlpError(f"cannot hash fast-MLP shard: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _strict_json(data: bytes, *, label: str) -> Mapping[str, object]:
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
        raise Qwen38FastMlpError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != data:
        raise Qwen38FastMlpError(f"{label} is not canonical JSON")
    return value


@dataclass(frozen=True, slots=True)
class Qwen38FastMlpPaths:
    analysis_root: Path
    transpose_root: Path
    pilot_root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "Qwen38FastMlpPaths":
        if not isinstance(root, (str, Path)):
            raise TypeError("fast_mlp_root must be a filesystem path")
        base = Path(root).expanduser().absolute()
        return cls(
            analysis_root=base / "mlp-pilot-router-p4-k32-v1",
            transpose_root=base / "mlp-pilot-down-transpose-v1",
            pilot_root=base / "mlp-pilot-packed-v1",
        )


@dataclass(frozen=True, slots=True)
class Qwen38FastMlpReceipt:
    model_pin_sha256: str
    router_fit_sha256: str
    affine_fit_sha256: str
    transpose_manifest_sha256: str
    pilot_manifest_body_sha256: str
    weights_index_sha256: str
    fitted_layers: tuple[int, ...]
    active_layers: tuple[int, ...]
    selected_neuron_fraction_by_layer: tuple[tuple[int, float], ...]
    transport_row_fraction_by_layer: tuple[tuple[int, float], ...]
    initialization: str = "sealed-prompt-fit/v1"
    online_config_sha256: str | None = None
    online_state_persistent: bool = False
    adaptive_width_policy_sha256: str | None = None
    maximum_selected_block_count_by_layer: tuple[tuple[int, int], ...] = ()
    maximum_transport_row_fraction_by_layer: tuple[tuple[int, float], ...] = ()

    def to_record(self) -> dict[str, object]:
        return {
            "active_layers": list(self.active_layers),
            "adaptive_width_policy_sha256": self.adaptive_width_policy_sha256,
            "affine_fit_sha256": self.affine_fit_sha256,
            "execution": "row-routed-sparse-mlp",
            "fitted_layers": list(self.fitted_layers),
            "initialization": self.initialization,
            "model_pin_sha256": self.model_pin_sha256,
            "maximum_selected_block_count_by_layer": {
                str(layer): count
                for layer, count in self.maximum_selected_block_count_by_layer
            },
            "maximum_transport_row_fraction_by_layer": {
                str(layer): fraction
                for layer, fraction in self.maximum_transport_row_fraction_by_layer
            },
            "online_config_sha256": self.online_config_sha256,
            "online_state_persistent": self.online_state_persistent,
            "pilot_manifest_body_sha256": self.pilot_manifest_body_sha256,
            "router_fit_sha256": self.router_fit_sha256,
            "schema": FAST_MLP_MOUNT_SCHEMA,
            "selected_neuron_fraction_by_layer": {
                str(layer): fraction
                for layer, fraction in self.selected_neuron_fraction_by_layer
            },
            "transport_row_fraction_by_layer": {
                str(layer): fraction
                for layer, fraction in self.transport_row_fraction_by_layer
            },
            "transpose_manifest_sha256": self.transpose_manifest_sha256,
            "weights_index_sha256": self.weights_index_sha256,
        }


class Qwen38FastMlpMount:
    """Own the auxiliary pagers used by one mounted sparse MLP executor."""

    def __init__(
        self,
        executor: MlpPilotSparseExecutor,
        transpose_pager: Qwen38WeightPager,
        pilot_pager: Qwen38WeightPager,
        receipt: Qwen38FastMlpReceipt,
    ) -> None:
        self.executor = executor
        self.transpose_pager = transpose_pager
        self.pilot_pager = pilot_pager
        self.receipt = receipt
        self._closed = False

    def metrics(self) -> dict[str, int]:
        if self.executor.online_controller is not None:
            self.executor.online_controller.flush()
        result: dict[str, int] = {}
        for prefix, pager in (
            ("pilot", self.pilot_pager),
            ("transpose", self.transpose_pager),
        ):
            metrics = pager.metrics()
            for source, target in (
                ("network_or_source_body_bytes", "source_body_bytes"),
                ("logical_weight_bytes", "logical_weight_bytes"),
                ("row_reads", "row_reads"),
            ):
                value = metrics.get(source, 0)
                result[f"{prefix}_{target}"] = (
                    int(value)
                    if isinstance(value, int) and not isinstance(value, bool)
                    else 0
                )
        result["source_body_bytes"] = (
            result["pilot_source_body_bytes"] + result["transpose_source_body_bytes"]
        )
        result["logical_weight_bytes"] = (
            result["pilot_logical_weight_bytes"]
            + result["transpose_logical_weight_bytes"]
        )
        online = self.executor.online_metrics()
        layers = None if online is None else online.get("layers")
        if isinstance(layers, list):
            for field in (
                "confirmed_rows",
                "exact_waves",
                "sparse_rows",
                "sparse_waves",
                "surprises",
                "width_updates",
                "output_confirmed_rows",
                "output_shadow_waves",
            ):
                result[f"online_{field}"] = sum(
                    int(row.get(field, 0))
                    for row in layers
                    if isinstance(row, Mapping)
                    and isinstance(row.get(field, 0), int)
                    and not isinstance(row.get(field, 0), bool)
                )
        return result

    def close(self) -> None:
        if self._closed:
            return
        failures: list[Exception] = []
        if self.executor.online_controller is not None:
            try:
                self.executor.online_controller.close()
            except Exception as exc:
                failures.append(exc)
        for pager in (self.pilot_pager, self.transpose_pager):
            try:
                pager.close()
            except Exception as exc:
                failures.append(exc)
        self._closed = True
        if failures:
            raise Qwen38FastMlpError(
                "fast-MLP auxiliary pager cleanup failed"
            ) from failures[0]


def _pilot_manifest(
    root: Path,
    *,
    router: MlpPilotRouterFit | MlpPilotWeightOnlyPlan,
    affine: MlpPilotAffineFit | MlpPilotIdentityAffinePlan,
    transpose: MlpPilotTransposeManifest,
) -> tuple[Mapping[str, object], str]:
    data = _stable_read(root / "pilot-weight-manifest.json")
    document = _strict_json(data, label="pilot weight manifest")
    schema = document.get("schema")
    if set(document) != {"body", "body_sha256", "schema"} or not isinstance(
        document.get("body"), Mapping
    ):
        raise Qwen38FastMlpError("pilot weight manifest envelope is invalid")
    body = cast(Mapping[str, object], document["body"])
    if schema == PILOT_WEIGHT_MANIFEST_SCHEMA and isinstance(router, MlpPilotRouterFit):
        expected = {
            "affine_fit_sha256",
            "entries",
            "model_pin_sha256",
            "router_fit_sha256",
            "tensor_payload_bytes",
            "transpose_manifest_sha256",
            "weights_index_sha256",
        }
        identities_match = (
            body.get("router_fit_sha256") == router.sha256
            and body.get("affine_fit_sha256") == affine.sha256
        )
    elif schema == PILOT_WEIGHT_MANIFEST_V2_SCHEMA and isinstance(
        router, MlpPilotWeightOnlyPlan
    ):
        expected = {
            "entries",
            "identity_affine_sha256",
            "model_pin_sha256",
            "tensor_payload_bytes",
            "transpose_manifest_sha256",
            "weight_plan_sha256",
            "weights_index_sha256",
        }
        identities_match = (
            body.get("weight_plan_sha256") == router.sha256
            and body.get("identity_affine_sha256") == affine.sha256
        )
    else:
        raise Qwen38FastMlpError("pilot weight manifest initialization mode changed")
    if (
        set(body) != expected
        or document.get("body_sha256") != _digest(body)
        or body.get("model_pin_sha256") != router.model_pin_sha256
        or not identities_match
        or body.get("transpose_manifest_sha256") != transpose.sha256
        or body.get("weights_index_sha256") != affine.weights_index_sha256
        or not isinstance(body.get("entries"), list)
    ):
        raise Qwen38FastMlpError("pilot weight manifest crosses artifact identities")
    return body, cast(str, document["body_sha256"])


def _validate_artifact_inventory(
    *,
    router: MlpPilotRouterFit | MlpPilotWeightOnlyPlan,
    affine: MlpPilotAffineFit | MlpPilotIdentityAffinePlan,
    transpose: MlpPilotTransposeManifest,
    pilot_body: Mapping[str, object],
    transpose_root: Path,
    pilot_root: Path,
    config: Qwen38Config,
    active_layers: tuple[int, ...],
) -> None:
    models = {row.layer: row for row in router.models}
    affine_models = {row.layer: row for row in affine.models}
    transpose_entries = {row.layer: row for row in transpose.entries}
    if set(models) != set(affine_models) or set(models) != set(transpose_entries):
        raise Qwen38FastMlpError("fast-MLP fitted layer inventories disagree")
    for layer, model in models.items():
        affine_row = affine_models[layer]
        entry = transpose_entries[layer]
        if (
            layer >= config.n_layers
            or model.intermediate_dimension != config.intermediate_size
            or affine_row.output_dimension != config.dim
            or entry.source_tensor
            != f"model.language_model.layers.{layer}.mlp.down_proj.weight"
            or entry.transpose_tensor != MlpPilotSparseExecutor.transpose_name(layer)
            or entry.shape
            != (model.intermediate_dimension, affine_row.output_dimension)
        ):
            raise Qwen38FastMlpError(f"transpose ABI changed at layer {layer}")
        require_sha256(entry.shard_sha256, field="transpose shard SHA")
        if layer in active_layers:
            shard = _regular_shard(transpose_root, entry.shard, entry.shard_bytes)
            if _stable_sha256(shard, entry.shard_bytes) != entry.shard_sha256:
                raise Qwen38FastMlpError(
                    f"transpose shard hash changed at layer {layer}"
                )

    raw_entries = cast(list[object], pilot_body["entries"])
    pilot_entries: dict[int, Mapping[str, object]] = {}
    for raw in raw_entries:
        if not isinstance(raw, Mapping) or set(raw) != {
            "layer",
            "pilot_indices_sha256",
            "shard",
            "shard_bytes",
            "shard_sha256",
            "tensors",
        }:
            raise Qwen38FastMlpError("pilot weight entry is invalid")
        layer = raw.get("layer")
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or layer in pilot_entries
        ):
            raise Qwen38FastMlpError("pilot weight layers are invalid")
        pilot_entries[layer] = raw
    if set(pilot_entries) != set(models):
        raise Qwen38FastMlpError("pilot weight layers differ from the router fit")
    for layer, raw in pilot_entries.items():
        model = models[layer]
        affine_row = affine_models[layer]
        expected_rows = model.block_count * model.pilot_count
        pilot_indices_sha256 = require_sha256(
            raw.get("pilot_indices_sha256"), field="pilot index SHA"
        )
        if (
            pilot_indices_sha256
            != hashlib.sha256(
                canonical_json_bytes(model.pilot_neuron_indices().tolist())
            ).hexdigest()
        ):
            raise Qwen38FastMlpError(f"pilot indices changed at layer {layer}")
        require_sha256(raw.get("shard_sha256"), field="pilot shard SHA")
        if layer in active_layers:
            pilot_shard = _regular_shard(
                pilot_root, raw.get("shard"), raw.get("shard_bytes")
            )
            if _stable_sha256(pilot_shard, cast(int, raw["shard_bytes"])) != raw.get(
                "shard_sha256"
            ):
                raise Qwen38FastMlpError(f"pilot shard hash changed at layer {layer}")
        tensors = raw.get("tensors")
        if not isinstance(tensors, list) or len(tensors) != 3:
            raise Qwen38FastMlpError(f"pilot tensor inventory changed at layer {layer}")
        expected_names = {
            MlpPilotSparseExecutor.pilot_name(layer, projection)
            for projection in ("gate", "up", "down_transpose")
        }
        actual_names: set[str] = set()
        for tensor in tensors:
            if not isinstance(tensor, Mapping) or set(tensor) != {
                "name",
                "raw_sha256",
                "shape",
            }:
                raise Qwen38FastMlpError("pilot tensor entry is invalid")
            name = tensor.get("name")
            if not isinstance(name, str):
                raise Qwen38FastMlpError("pilot tensor name is invalid")
            require_sha256(tensor.get("raw_sha256"), field="pilot tensor SHA")
            if tensor.get("shape") != [expected_rows, affine_row.output_dimension]:
                raise Qwen38FastMlpError(f"pilot tensor shape changed at layer {layer}")
            actual_names.add(name)
        if actual_names != expected_names:
            raise Qwen38FastMlpError(f"pilot tensor names changed at layer {layer}")


def _validate_source_tensor(
    source: Streamer,
    *,
    name: str,
    shape: tuple[int, int],
    shard: str,
) -> None:
    try:
        row = source.find(name)
    except Exception as exc:
        raise Qwen38FastMlpError(f"fast-MLP tensor is absent: {name}") from exc
    if (
        str(row.get("dtype", "")).upper() != "BF16"
        or tuple(row.get("shape", ())) != shape
        or row.get("shard") != shard
    ):
        raise Qwen38FastMlpError(f"fast-MLP tensor ABI changed: {name}")


def _validate_open_sources(
    *,
    router: MlpPilotRouterFit | MlpPilotWeightOnlyPlan,
    affine: MlpPilotAffineFit | MlpPilotIdentityAffinePlan,
    transpose: MlpPilotTransposeManifest,
    pilot_body: Mapping[str, object],
    active_layers: tuple[int, ...],
    transpose_source: Streamer,
    pilot_source: Streamer,
) -> None:
    models = {row.layer: row for row in router.models}
    affine_models = {row.layer: row for row in affine.models}
    transpose_entries = {row.layer: row for row in transpose.entries}
    pilot_entries = {
        cast(int, row["layer"]): row
        for row in cast(list[Mapping[str, object]], pilot_body["entries"])
    }
    for layer in active_layers:
        model = models[layer]
        width = affine_models[layer].output_dimension
        transpose_entry = transpose_entries[layer]
        _validate_source_tensor(
            transpose_source,
            name=transpose_entry.transpose_tensor,
            shape=(model.intermediate_dimension, width),
            shard=transpose_entry.shard,
        )
        pilot_entry = pilot_entries[layer]
        pilot_rows = model.block_count * model.pilot_count
        pilot_shard = cast(str, pilot_entry["shard"])
        for projection in ("gate", "up", "down_transpose"):
            _validate_source_tensor(
                pilot_source,
                name=MlpPilotSparseExecutor.pilot_name(layer, projection),
                shape=(pilot_rows, width),
                shard=pilot_shard,
            )


def open_qwen38_fast_mlp(
    *,
    paths: Qwen38FastMlpPaths,
    target_mount: CausalWeightMount,
    target_pager: Qwen38WeightPager,
    config: Qwen38Config,
    source_budget_mb: float,
    max_resident_bytes: int,
    active_layers: Sequence[int] | None = None,
    online_state_path: str | Path | None = None,
) -> Qwen38FastMlpMount:
    """Mount an existing sparse MLP bank without opening any training corpus."""

    if not isinstance(paths, Qwen38FastMlpPaths):
        raise TypeError("paths must be Qwen38FastMlpPaths")
    if not isinstance(target_mount, CausalWeightMount):
        raise TypeError("target_mount must be a CausalWeightMount")
    if not isinstance(target_pager, Qwen38WeightPager):
        raise TypeError("target_pager must be a Qwen38WeightPager")
    if not isinstance(config, Qwen38Config):
        raise TypeError("config must be a Qwen38Config")
    if online_state_path is not None and not isinstance(online_state_path, (str, Path)):
        raise TypeError("online_state_path must be a filesystem path or None")
    if target_pager.resolved_dtype != "bfloat16":
        raise Qwen38FastMlpError("fast-MLP execution requires target BF16 compute")
    if target_pager.source is not target_mount.source:
        raise Qwen38FastMlpError(
            "target pager must borrow the exact target causal mount source"
        )
    weights_root = target_mount.weights_root
    for root in (
        paths.analysis_root,
        paths.transpose_root,
        paths.pilot_root,
    ):
        _plain_directory(root)
    plan_path = paths.analysis_root / WEIGHT_ONLY_PLAN_NAME
    legacy_paths = (
        paths.analysis_root / "fit.json",
        paths.analysis_root / "affine-fit.json",
    )
    plan_present = plan_path.exists() or plan_path.is_symlink()
    legacy_present = tuple(path.exists() or path.is_symlink() for path in legacy_paths)
    width_config: MlpPilotAdaptiveWidthConfig | None = None
    if plan_present and any(legacy_present):
        raise Qwen38FastMlpError(
            "fast-MLP analysis mixes fitted and weight-only initialization"
        )
    if plan_present:
        router: MlpPilotRouterFit | MlpPilotWeightOnlyPlan = (
            MlpPilotWeightOnlyPlan.from_bytes(_stable_read(plan_path))
        )
        if (
            router.n_layers != config.n_layers
            or router.hidden_dimension != config.dim
            or any(
                row.intermediate_dimension != config.intermediate_size
                for row in router.models
            )
        ):
            raise Qwen38FastMlpError(
                "weight-only plan differs from the mounted Qwen topology"
            )
        mounted_model = getattr(target_mount, "model", None)
        if mounted_model is not None and (
            getattr(mounted_model, "repo_id", None) != router.repo_id
            or getattr(mounted_model, "revision", None) != router.revision
        ):
            raise Qwen38FastMlpError(
                "weight-only plan differs from the mounted logical model"
            )
        mounted_layout = getattr(target_mount, "layout", None)
        if mounted_layout is not None and (
            getattr(mounted_layout, "layout_fingerprint", None)
            != router.layout_fingerprint
        ):
            raise Qwen38FastMlpError(
                "weight-only plan differs from the mounted weight layout"
            )
        affine: MlpPilotAffineFit | MlpPilotIdentityAffinePlan = router.affine_fit
        width_path = paths.analysis_root / ADAPTIVE_WIDTH_CONFIG_NAME
        width_config = (
            MlpPilotAdaptiveWidthConfig.from_bytes(_stable_read(width_path, 64 * 1024))
            if width_path.exists() or width_path.is_symlink()
            else MlpPilotAdaptiveWidthConfig()
        )
        initialization = WEIGHT_ONLY_INITIALIZER
    else:
        if legacy_present != (True, True):
            raise Qwen38FastMlpError("fast-MLP analysis artifacts are incomplete")
        router = MlpPilotRouterFit.from_bytes(_stable_read(legacy_paths[0]))
        affine = MlpPilotAffineFit.from_bytes(_stable_read(legacy_paths[1]))
        initialization = "sealed-prompt-fit/v1"
        if online_state_path is not None:
            raise Qwen38FastMlpError(
                "online state is available only for a weight-only plan"
            )
    transpose = MlpPilotTransposeManifest.from_bytes(
        _stable_read(paths.transpose_root / "pilot-transpose-manifest.json")
    )
    index_path = weights_root / "model.safetensors.index.json"
    if not index_path.is_file() or index_path.is_symlink():
        raise Qwen38FastMlpError(
            "this fast-MLP bank requires its indexed source checkpoint"
        )
    index_sha256 = hashlib.sha256(_stable_read(index_path)).hexdigest()
    if (
        affine.router_fit_sha256 != router.sha256
        or affine.model_pin_sha256 != router.model_pin_sha256
        or affine.weights_index_sha256 != index_sha256
        or transpose.model_pin_sha256 != router.model_pin_sha256
        or transpose.router_fit_sha256 != router.sha256
        or transpose.affine_fit_sha256 != affine.sha256
        or transpose.weights_index_sha256 != index_sha256
    ):
        raise Qwen38FastMlpError(
            "fast-MLP artifacts differ from the mounted Qwen weights"
        )
    pilot_body, pilot_body_sha256 = _pilot_manifest(
        paths.pilot_root,
        router=router,
        affine=affine,
        transpose=transpose,
    )
    fitted_layers = tuple(row.layer for row in router.models)
    if active_layers is None:
        selected_layers = fitted_layers
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
            raise Qwen38FastMlpError(
                "active fast-MLP layers must be a sorted fitted subset"
            )
    _validate_artifact_inventory(
        router=router,
        affine=affine,
        transpose=transpose,
        pilot_body=pilot_body,
        transpose_root=paths.transpose_root,
        pilot_root=paths.pilot_root,
        config=config,
        active_layers=selected_layers,
    )

    transpose_source: Streamer | None = None
    pilot_source: Streamer | None = None
    transpose_pager: Qwen38WeightPager | None = None
    pilot_pager: Qwen38WeightPager | None = None
    online_controller: MlpPilotOnlineController | None = None
    try:
        transpose_source = Streamer.from_local(
            paths.transpose_root,
            revision=transpose.sha256,
            budget_mb=source_budget_mb,
            use_cache=False,
        )
        pilot_source = Streamer.from_local(
            paths.pilot_root,
            revision=pilot_body_sha256,
            budget_mb=source_budget_mb,
            use_cache=False,
        )
        _validate_open_sources(
            router=router,
            affine=affine,
            transpose=transpose,
            pilot_body=pilot_body,
            active_layers=selected_layers,
            transpose_source=transpose_source,
            pilot_source=pilot_source,
        )
        transpose_pager = Qwen38WeightPager(
            transpose_source,
            device=target_pager.resolved_device,
            compute_dtype=target_pager.resolved_dtype,
            max_resident_bytes=max_resident_bytes,
            close_source=True,
        )
        pilot_pager = Qwen38WeightPager(
            pilot_source,
            device=target_pager.resolved_device,
            compute_dtype=target_pager.resolved_dtype,
            max_resident_bytes=max_resident_bytes,
            close_source=True,
        )
        if isinstance(router, MlpPilotWeightOnlyPlan):
            assert width_config is not None
            block_bytes = (
                router.config.block_size
                * router.hidden_dimension
                * torch.empty((), dtype=target_pager.compute_dtype).element_size()
            )
            resident_limits = (
                max_resident_bytes,
                getattr(target_pager, "max_resident_bytes", max_resident_bytes),
            )
            resident_block_limit = min(
                int(limit) // block_bytes
                for limit in resident_limits
                if isinstance(limit, int) and not isinstance(limit, bool)
            )
            effective_maximum = min(
                width_config.max_selected_block_count,
                resident_block_limit,
            )
            if effective_maximum < router.config.selected_block_count:
                raise Qwen38FastMlpError(
                    "fast-MLP residency cannot hold its minimum adaptive route"
                )
            effective_width_config = MlpPilotAdaptiveWidthConfig(
                max_selected_block_count=effective_maximum,
                selected_block_step=width_config.selected_block_step,
                score_mass_margin=width_config.score_mass_margin,
                target_capture=width_config.target_capture,
                min_output_cosine=width_config.min_output_cosine,
                max_output_relative_l2=width_config.max_output_relative_l2,
                min_output_confirmed_rows=width_config.min_output_confirmed_rows,
                output_metric_window_rows=width_config.output_metric_window_rows,
                max_shadow_rows=width_config.max_shadow_rows,
                max_sparse_transport_fraction=(
                    width_config.max_sparse_transport_fraction
                ),
            )
            online_controller = MlpPilotOnlineController(
                router,
                state_path=online_state_path,
                width_config=effective_width_config,
            )
        executor = MlpPilotSparseExecutor(
            router,
            affine,
            target_pager,
            transpose_pager,
            pilot_pager=pilot_pager,
            active_layers=selected_layers,
            output_dtype=target_pager.compute_dtype,
            online_controller=online_controller,
        )
        selected_fractions = tuple(
            (row.layer, row.selected_neuron_count / row.intermediate_dimension)
            for row in router.models
            if executor.supports_layer(row.layer)
        )
        transport_fractions = tuple(
            (
                row.layer,
                (
                    row.block_count * row.pilot_count
                    + row.selected_block_count * row.block_size
                )
                / row.intermediate_dimension,
            )
            for row in router.models
            if executor.supports_layer(row.layer)
        )
        maximum_block_counts = (
            ()
            if online_controller is None
            else tuple(
                (
                    row.layer,
                    online_controller.max_selected_block_count(layer=row.layer),
                )
                for row in router.models
                if executor.supports_layer(row.layer)
            )
        )
        maximum_transport_fractions = (
            ()
            if online_controller is None
            else tuple(
                (
                    row.layer,
                    (
                        row.block_count * row.pilot_count
                        + online_controller.max_selected_block_count(layer=row.layer)
                        * row.block_size
                    )
                    / row.intermediate_dimension,
                )
                for row in router.models
                if executor.supports_layer(row.layer)
            )
        )
        receipt = Qwen38FastMlpReceipt(
            model_pin_sha256=router.model_pin_sha256,
            router_fit_sha256=router.sha256,
            affine_fit_sha256=affine.sha256,
            transpose_manifest_sha256=transpose.sha256,
            pilot_manifest_body_sha256=pilot_body_sha256,
            weights_index_sha256=index_sha256,
            fitted_layers=fitted_layers,
            active_layers=executor.active_layers,
            selected_neuron_fraction_by_layer=selected_fractions,
            transport_row_fraction_by_layer=transport_fractions,
            initialization=initialization,
            online_config_sha256=(
                None if online_controller is None else online_controller.config.sha256
            ),
            online_state_persistent=online_state_path is not None,
            adaptive_width_policy_sha256=(
                None
                if online_controller is None
                else online_controller.width_policy_sha256
            ),
            maximum_selected_block_count_by_layer=maximum_block_counts,
            maximum_transport_row_fraction_by_layer=maximum_transport_fractions,
        )
        return Qwen38FastMlpMount(executor, transpose_pager, pilot_pager, receipt)
    except Exception:
        if online_controller is not None:
            online_controller.close()
        if pilot_pager is not None:
            pilot_pager.close()
        elif pilot_source is not None:
            pilot_source.close()
        if transpose_pager is not None:
            transpose_pager.close()
        elif transpose_source is not None:
            transpose_source.close()
        raise


__all__ = [
    "ADAPTIVE_WIDTH_CONFIG_NAME",
    "FAST_MLP_MOUNT_SCHEMA",
    "PILOT_WEIGHT_MANIFEST_SCHEMA",
    "PILOT_WEIGHT_MANIFEST_V2_SCHEMA",
    "WEIGHT_ONLY_PLAN_NAME",
    "Qwen38FastMlpError",
    "Qwen38FastMlpMount",
    "Qwen38FastMlpPaths",
    "Qwen38FastMlpReceipt",
    "open_qwen38_fast_mlp",
]
