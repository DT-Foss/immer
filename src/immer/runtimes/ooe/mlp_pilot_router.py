"""Layer-local pilot-neuron routing for sparse causal Qwen MLP ranges."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence, cast

import numpy as np

from .identity import canonical_json_bytes, require_sha256
from .subspace_battery import SubspaceCorpus, SubspaceObservationGroup


PILOT_ROUTER_CONFIG_SCHEMA = "immer.qwen-mlp-pilot-router-config/v1"
PILOT_LAYER_MODEL_SCHEMA = "immer.qwen-mlp-pilot-layer-model/v1"
PILOT_ROUTER_METRICS_SCHEMA = "immer.qwen-mlp-pilot-router-metrics/v1"
PILOT_ROUTER_FIT_SCHEMA = "immer.qwen-mlp-pilot-router-fit/v1"
PILOT_ROUTER_EVALUATION_SCHEMA = "immer.qwen-mlp-pilot-router-evaluation/v1"
PILOT_ROUTER_VERIFIER_SHA256 = hashlib.sha256(
    b"immer:qwen-mlp-pilot-router/content-row+layer-local+residual-omp/v1"
).hexdigest()
DEFAULT_RANDOM_SEED_SHA256 = hashlib.sha256(
    b"immer:qwen-mlp-pilot-router/random-control/v1"
).hexdigest()
_MAX_ARTIFACT_BYTES = 64 * 1024 * 1024


class MlpPilotRouterError(RuntimeError):
    pass


class MlpPilotRouterIntegrityError(MlpPilotRouterError):
    pass


class MlpPilotRouterCapacityError(MlpPilotRouterError):
    pass


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sealed(schema: str, body: Mapping[str, object]) -> dict[str, object]:
    normalized = json.loads(canonical_json_bytes(dict(body)))
    return {"body": normalized, "body_sha256": _digest(normalized), "schema": schema}


def _strict_json(data: bytes, *, schema: str, label: str) -> Mapping[str, object]:
    if not isinstance(data, bytes) or not data or len(data) > _MAX_ARTIFACT_BYTES:
        raise MlpPilotRouterIntegrityError(f"{label} exceeds its byte bound")

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
        raise MlpPilotRouterIntegrityError(f"{label} is not strict JSON") from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != {"body", "body_sha256", "schema"}
        or value.get("schema") != schema
        or not isinstance(value.get("body"), Mapping)
        or value.get("body_sha256") != _digest(value.get("body"))
        or canonical_json_bytes(value) != data
    ):
        raise MlpPilotRouterIntegrityError(f"{label} seal is invalid")
    return cast(Mapping[str, object], value)


def _sha256s(values: Sequence[str], *, field: str) -> tuple[str, ...]:
    result = tuple(require_sha256(value, field=field) for value in values)
    if not result or result != tuple(sorted(set(result))):
        raise ValueError(f"{field} must be sorted, unique, and non-empty")
    return result


def _finite(value: object, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(
            f"{field} must be finite" + (" and positive" if positive else "")
        )
    return result


def _float_array(value: object, *, field: str, shape: tuple[int, int]) -> np.ndarray:
    if type(value) is not np.ndarray or value.dtype != np.dtype(np.float64):
        raise TypeError(f"{field} must be an exact float64 numpy.ndarray")
    if value.shape != shape or not bool(np.isfinite(value).all()):
        raise ValueError(f"{field} shape or finite policy is invalid")
    result = np.array(value, dtype="<f8", order="C", copy=True)
    result[result == 0.0] = 0.0
    result.flags.writeable = False
    return result


def _array_record(value: np.ndarray) -> dict[str, object]:
    raw = np.asarray(value, dtype="<f8", order="C").tobytes(order="C")
    return {
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "dtype": "float64-le",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": [int(row) for row in value.shape],
    }


def _array_from_record(
    value: object, *, field: str, shape: tuple[int, int]
) -> np.ndarray:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"data_base64", "dtype", "sha256", "shape"}
        or value.get("dtype") != "float64-le"
        or value.get("shape") != list(shape)
        or not isinstance(value.get("data_base64"), str)
    ):
        raise MlpPilotRouterIntegrityError(f"{field} array record is invalid")
    try:
        raw = base64.b64decode(cast(str, value["data_base64"]), validate=True)
    except (TypeError, ValueError) as exc:
        raise MlpPilotRouterIntegrityError(f"{field} encoding is invalid") from exc
    if len(raw) != math.prod(shape) * 8 or hashlib.sha256(
        raw
    ).hexdigest() != require_sha256(value.get("sha256"), field=f"{field}.sha256"):
        raise MlpPilotRouterIntegrityError(f"{field} bytes are invalid")
    return _float_array(
        np.frombuffer(raw, dtype="<f8").reshape(shape), field=field, shape=shape
    )


@dataclass(frozen=True, slots=True)
class MlpPilotRouterConfig:
    block_size: int = 64
    pilot_count: int = 4
    selected_block_count: int = 32
    ridge: float = 1e-6
    random_seed_sha256: str = DEFAULT_RANDOM_SEED_SHA256
    max_working_bytes: int = 8 * 1024**3

    def __post_init__(self) -> None:
        for field in (
            "block_size",
            "pilot_count",
            "selected_block_count",
            "max_working_bytes",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.pilot_count >= self.block_size:
            raise ValueError("pilot_count must be smaller than block_size")
        object.__setattr__(
            self, "ridge", _finite(self.ridge, field="ridge", positive=True)
        )
        object.__setattr__(
            self,
            "random_seed_sha256",
            require_sha256(self.random_seed_sha256, field="random_seed_sha256"),
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "block_size": self.block_size,
            "max_working_bytes": self.max_working_bytes,
            "pilot_count": self.pilot_count,
            "random_seed_sha256": self.random_seed_sha256,
            "ridge": self.ridge,
            "schema": PILOT_ROUTER_CONFIG_SCHEMA,
            "selected_block_count": self.selected_block_count,
        }

    @classmethod
    def from_record(cls, value: object) -> "MlpPilotRouterConfig":
        expected = {
            "block_size",
            "max_working_bytes",
            "pilot_count",
            "random_seed_sha256",
            "ridge",
            "schema",
            "selected_block_count",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != PILOT_ROUTER_CONFIG_SCHEMA
        ):
            raise MlpPilotRouterIntegrityError("pilot router config is invalid")
        try:
            return cls(
                block_size=cast(int, value["block_size"]),
                pilot_count=cast(int, value["pilot_count"]),
                selected_block_count=cast(int, value["selected_block_count"]),
                ridge=cast(float, value["ridge"]),
                random_seed_sha256=cast(str, value["random_seed_sha256"]),
                max_working_bytes=cast(int, value["max_working_bytes"]),
            )
        except (TypeError, ValueError) as exc:
            raise MlpPilotRouterIntegrityError(
                "pilot router config validation failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class MlpPilotRouterMetrics:
    row_count: int
    selected_neuron_count: int
    intermediate_dimension: int
    pilot_energy_capture: float
    random_energy_capture: float
    marginal_energy_capture: float
    oracle_energy_capture: float

    def __post_init__(self) -> None:
        for field in ("row_count", "selected_neuron_count", "intermediate_dimension"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be positive")
        if self.selected_neuron_count >= self.intermediate_dimension:
            raise ValueError("router must select a strict neuron subset")
        for field in (
            "pilot_energy_capture",
            "random_energy_capture",
            "marginal_energy_capture",
            "oracle_energy_capture",
        ):
            value = _finite(getattr(self, field), field=field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be a probability")
            object.__setattr__(self, field, value)

    @property
    def selected_fraction(self) -> float:
        return self.selected_neuron_count / self.intermediate_dimension

    @property
    def gain_over_marginal(self) -> float:
        return self.pilot_energy_capture - self.marginal_energy_capture

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "gain_over_marginal": self.gain_over_marginal,
            "intermediate_dimension": self.intermediate_dimension,
            "marginal_energy_capture": self.marginal_energy_capture,
            "oracle_energy_capture": self.oracle_energy_capture,
            "pilot_energy_capture": self.pilot_energy_capture,
            "random_energy_capture": self.random_energy_capture,
            "row_count": self.row_count,
            "schema": PILOT_ROUTER_METRICS_SCHEMA,
            "selected_fraction": self.selected_fraction,
            "selected_neuron_count": self.selected_neuron_count,
        }

    @classmethod
    def from_record(cls, value: object) -> "MlpPilotRouterMetrics":
        expected = {
            "gain_over_marginal",
            "intermediate_dimension",
            "marginal_energy_capture",
            "oracle_energy_capture",
            "pilot_energy_capture",
            "random_energy_capture",
            "row_count",
            "schema",
            "selected_fraction",
            "selected_neuron_count",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != PILOT_ROUTER_METRICS_SCHEMA
        ):
            raise MlpPilotRouterIntegrityError("pilot router metrics are invalid")
        try:
            result = cls(
                row_count=cast(int, value["row_count"]),
                selected_neuron_count=cast(int, value["selected_neuron_count"]),
                intermediate_dimension=cast(int, value["intermediate_dimension"]),
                pilot_energy_capture=cast(float, value["pilot_energy_capture"]),
                random_energy_capture=cast(float, value["random_energy_capture"]),
                marginal_energy_capture=cast(float, value["marginal_energy_capture"]),
                oracle_energy_capture=cast(float, value["oracle_energy_capture"]),
            )
        except (TypeError, ValueError) as exc:
            raise MlpPilotRouterIntegrityError(
                "pilot router metrics validation failed"
            ) from exc
        if (
            value.get("gain_over_marginal") != result.gain_over_marginal
            or value.get("selected_fraction") != result.selected_fraction
        ):
            raise MlpPilotRouterIntegrityError("derived pilot metrics changed")
        return result


def _activation_square(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    if (
        type(gate) is not np.ndarray
        or type(up) is not np.ndarray
        or gate.dtype != np.dtype(np.float64)
        or up.dtype != np.dtype(np.float64)
        or gate.shape != up.shape
        or gate.ndim != 2
        or not bool(np.isfinite(gate).all())
        or not bool(np.isfinite(up).all())
    ):
        raise TypeError("gate/up must be matching finite float64 matrices")
    silu = gate / (1.0 + np.exp(-np.clip(gate, -60.0, 60.0)))
    return np.square(silu * up)


def _block_features(group: SubspaceObservationGroup, block_size: int) -> np.ndarray:
    if group.intermediate_dimension % block_size:
        raise ValueError("intermediate dimension is not divisible by block_size")
    features = _activation_square(group.gate_projection, group.up_projection)
    return features.reshape(group.row_count, -1, block_size)


def _fit_coefficients(
    features: np.ndarray, target: np.ndarray, offsets: Sequence[int], ridge: float
) -> np.ndarray:
    selected = tuple(int(row) for row in offsets)
    design = np.column_stack((np.ones(len(features)), features[:, selected]))
    regularizer = np.eye(len(selected) + 1, dtype=np.float64) * ridge
    regularizer[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + regularizer, design.T @ target).astype(
        np.float64, copy=False
    )


def _residual_omp(
    features: np.ndarray, target: np.ndarray, count: int, ridge: float
) -> tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or target.shape != (len(features),) or len(features) < 2:
        raise ValueError("OMP requires aligned non-empty training rows")
    selected: list[int] = []
    centered = features - features.mean(axis=0)
    for _ in range(count):
        if selected:
            coefficients = _fit_coefficients(features, target, selected, ridge)
            residual = target - (
                coefficients[0] + features[:, selected] @ coefficients[1:]
            )
        else:
            residual = target - target.mean()
        residual = residual - residual.mean()
        denominator = np.sqrt(
            np.square(centered).sum(axis=0) * np.square(residual).sum()
        )
        score = np.divide(
            np.abs(centered.T @ residual),
            denominator,
            out=np.zeros(features.shape[1], dtype=np.float64),
            where=denominator > 0.0,
        )
        if selected:
            score[np.asarray(selected, dtype=np.int64)] = -1.0
        selected.append(int(np.argmax(score)))
    # Runtime inventories are canonical block-local ranges. Refit after sorting
    # because OMP discovery order is semantic, while persisted offset order is
    # purely an address order.
    offsets = np.asarray(sorted(selected), dtype=np.int64)
    return offsets, _fit_coefficients(features, target, offsets, ridge)


def _random_offsets(
    seed: str, layer: int, block: int, count: int, width: int
) -> np.ndarray:
    material = hashlib.sha256(f"{seed}:{layer}:{block}".encode("ascii")).digest()
    generator = np.random.default_rng(int.from_bytes(material[:8], "big"))
    return np.sort(generator.choice(width, size=count, replace=False)).astype(np.int64)


@dataclass(frozen=True, slots=True)
class MlpPilotLayerModel:
    layer: int
    intermediate_dimension: int
    block_size: int
    selected_block_count: int
    pilot_offsets: tuple[tuple[int, ...], ...]
    coefficients: np.ndarray
    random_pilot_offsets: tuple[tuple[int, ...], ...]
    random_coefficients: np.ndarray
    marginal_block_scores: np.ndarray
    training_group_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "layer",
            "intermediate_dimension",
            "block_size",
            "selected_block_count",
        ):
            value = getattr(self, field)
            minimum = 0 if field == "layer" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field} is invalid")
        if self.intermediate_dimension % self.block_size:
            raise ValueError("layer intermediate dimension is not block aligned")
        block_count = self.intermediate_dimension // self.block_size
        if self.selected_block_count >= block_count:
            raise ValueError("selected_block_count must leave at least one block out")
        pilots = tuple(tuple(row) for row in self.pilot_offsets)
        random_pilots = tuple(tuple(row) for row in self.random_pilot_offsets)
        if (
            len(pilots) != block_count
            or len(random_pilots) != block_count
            or not pilots[0]
            or len({len(row) for row in pilots}) != 1
            or len({len(row) for row in random_pilots}) != 1
            or len(pilots[0]) != len(random_pilots[0])
        ):
            raise ValueError("layer pilot inventory shape is invalid")
        for inventory in (pilots, random_pilots):
            if any(
                row != tuple(sorted(set(row)))
                or any(
                    isinstance(offset, bool)
                    or not isinstance(offset, int)
                    or not 0 <= offset < self.block_size
                    for offset in row
                )
                for row in inventory
            ):
                raise ValueError("layer pilot offsets are invalid")
        pilot_count = len(pilots[0])
        coefficients = _float_array(
            self.coefficients,
            field="coefficients",
            shape=(block_count, pilot_count + 1),
        )
        random_coefficients = _float_array(
            self.random_coefficients,
            field="random_coefficients",
            shape=(block_count, pilot_count + 1),
        )
        marginal = _float_array(
            np.asarray(self.marginal_block_scores).reshape(1, -1),
            field="marginal_block_scores",
            shape=(1, block_count),
        ).reshape(-1)
        marginal.flags.writeable = False
        training = _sha256s(self.training_group_sha256s, field="training_group_sha256s")
        object.__setattr__(self, "pilot_offsets", pilots)
        object.__setattr__(self, "random_pilot_offsets", random_pilots)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "random_coefficients", random_coefficients)
        object.__setattr__(self, "marginal_block_scores", marginal)
        object.__setattr__(self, "training_group_sha256s", training)

    @property
    def block_count(self) -> int:
        return self.intermediate_dimension // self.block_size

    @property
    def pilot_count(self) -> int:
        return len(self.pilot_offsets[0])

    @property
    def selected_neuron_count(self) -> int:
        return self.block_count * self.pilot_count + self.selected_block_count * (
            self.block_size - self.pilot_count
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def pilot_neuron_indices(self, *, random_control: bool = False) -> np.ndarray:
        inventory = self.random_pilot_offsets if random_control else self.pilot_offsets
        return np.asarray(
            [
                block * self.block_size + offset
                for block, offsets in enumerate(inventory)
                for offset in offsets
            ],
            dtype=np.int64,
        )

    def score_pilot_arrays(
        self,
        gate: np.ndarray,
        up: np.ndarray,
        *,
        random_control: bool = False,
    ) -> np.ndarray:
        expected = self.block_count * self.pilot_count
        if gate.shape != up.shape or gate.ndim != 2 or gate.shape[1] != expected:
            raise ValueError("pilot gate/up arrays differ from the layer pilot ABI")
        features = _activation_square(gate, up).reshape(
            len(gate), self.block_count, self.pilot_count
        )
        coefficients = self.random_coefficients if random_control else self.coefficients
        scores = coefficients[None, :, 0] + np.sum(
            features * coefficients[None, :, 1:], axis=2
        )
        return np.maximum(scores, 0.0)

    def select_blocks_from_full(
        self,
        gate: np.ndarray,
        up: np.ndarray,
        *,
        random_control: bool = False,
    ) -> np.ndarray:
        indices = self.pilot_neuron_indices(random_control=random_control)
        scores = self.score_pilot_arrays(
            gate[:, indices], up[:, indices], random_control=random_control
        )
        return np.argsort(-scores, axis=1, kind="stable")[
            :, : self.selected_block_count
        ]

    def selected_neurons(
        self,
        selected_blocks: Sequence[int],
        *,
        random_control: bool = False,
    ) -> tuple[int, ...]:
        blocks = tuple(int(row) for row in selected_blocks)
        if (
            len(blocks) != self.selected_block_count
            or len(set(blocks)) != len(blocks)
            or any(not 0 <= block < self.block_count for block in blocks)
        ):
            raise ValueError("selected block action is invalid")
        neurons = set(self.pilot_neuron_indices(random_control=random_control).tolist())
        for block in blocks:
            neurons.update(
                range(block * self.block_size, (block + 1) * self.block_size)
            )
        result = tuple(sorted(neurons))
        if len(result) != self.selected_neuron_count:
            raise AssertionError("pilot/block neuron union changed size")
        return result

    def to_record(self) -> dict[str, object]:
        return {
            "block_size": self.block_size,
            "coefficients": _array_record(self.coefficients),
            "intermediate_dimension": self.intermediate_dimension,
            "layer": self.layer,
            "marginal_block_scores": _array_record(
                self.marginal_block_scores.reshape(1, -1)
            ),
            "pilot_offsets": [list(row) for row in self.pilot_offsets],
            "random_coefficients": _array_record(self.random_coefficients),
            "random_pilot_offsets": [list(row) for row in self.random_pilot_offsets],
            "schema": PILOT_LAYER_MODEL_SCHEMA,
            "selected_block_count": self.selected_block_count,
            "training_group_sha256s": list(self.training_group_sha256s),
        }

    @classmethod
    def from_record(cls, value: object) -> "MlpPilotLayerModel":
        expected = {
            "block_size",
            "coefficients",
            "intermediate_dimension",
            "layer",
            "marginal_block_scores",
            "pilot_offsets",
            "random_coefficients",
            "random_pilot_offsets",
            "schema",
            "selected_block_count",
            "training_group_sha256s",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != PILOT_LAYER_MODEL_SCHEMA
            or not isinstance(value.get("pilot_offsets"), list)
            or not isinstance(value.get("random_pilot_offsets"), list)
            or not isinstance(value.get("training_group_sha256s"), list)
        ):
            raise MlpPilotRouterIntegrityError("pilot layer model is invalid")
        try:
            intermediate = cast(int, value["intermediate_dimension"])
            block_size = cast(int, value["block_size"])
            block_count = intermediate // block_size
            pilot_rows = cast(list[list[int]], value["pilot_offsets"])
            pilot_count = len(pilot_rows[0])
            return cls(
                layer=cast(int, value["layer"]),
                intermediate_dimension=intermediate,
                block_size=block_size,
                selected_block_count=cast(int, value["selected_block_count"]),
                pilot_offsets=tuple(tuple(row) for row in pilot_rows),
                coefficients=_array_from_record(
                    value["coefficients"],
                    field="coefficients",
                    shape=(block_count, pilot_count + 1),
                ),
                random_pilot_offsets=tuple(
                    tuple(row)
                    for row in cast(list[list[int]], value["random_pilot_offsets"])
                ),
                random_coefficients=_array_from_record(
                    value["random_coefficients"],
                    field="random_coefficients",
                    shape=(block_count, pilot_count + 1),
                ),
                marginal_block_scores=_array_from_record(
                    value["marginal_block_scores"],
                    field="marginal_block_scores",
                    shape=(1, block_count),
                ).reshape(-1),
                training_group_sha256s=tuple(
                    cast(list[str], value["training_group_sha256s"])
                ),
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise MlpPilotRouterIntegrityError(
                "pilot layer model validation failed"
            ) from exc


def _rectangular_groups(
    corpus: SubspaceCorpus, prompts: tuple[str, ...]
) -> dict[tuple[int, str], SubspaceObservationGroup]:
    prompt_set = set(prompts)
    groups = [group for group in corpus.groups if group.prompt_sha256 in prompt_set]
    if len(groups) != len(corpus.groups):
        raise ValueError("pilot corpus contains prompts outside the sealed split")
    by_key = {(group.layer, group.prompt_sha256): group for group in groups}
    layers = {group.layer for group in groups}
    if len(by_key) != len(groups) or set(by_key) != {
        (layer, prompt) for layer in layers for prompt in prompts
    }:
        raise ValueError("pilot corpus is not a rectangular layer/prompt grid")
    return by_key


def _fit_layer(
    groups: Sequence[SubspaceObservationGroup], config: MlpPilotRouterConfig
) -> MlpPilotLayerModel:
    first = groups[0]
    block_count = first.intermediate_dimension // config.block_size
    if config.selected_block_count >= block_count:
        raise ValueError("selected block count exhausts the layer")
    estimated = (
        sum(group.row_count for group in groups) * first.intermediate_dimension * 24
    )
    if estimated > config.max_working_bytes:
        raise MlpPilotRouterCapacityError("pilot fit exceeds its working-byte bound")
    features = np.concatenate(
        [_block_features(group, config.block_size) for group in groups], axis=0
    )
    target = features.sum(axis=2)
    pilot_offsets = []
    coefficients = []
    random_offsets = []
    random_coefficients = []
    for block in range(block_count):
        offsets, fitted = _residual_omp(
            features[:, block, :], target[:, block], config.pilot_count, config.ridge
        )
        controls = _random_offsets(
            config.random_seed_sha256,
            first.layer,
            block,
            config.pilot_count,
            config.block_size,
        )
        pilot_offsets.append(tuple(int(row) for row in offsets))
        coefficients.append(fitted)
        random_offsets.append(tuple(int(row) for row in controls))
        random_coefficients.append(
            _fit_coefficients(
                features[:, block, :], target[:, block], controls, config.ridge
            )
        )
    return MlpPilotLayerModel(
        layer=first.layer,
        intermediate_dimension=first.intermediate_dimension,
        block_size=config.block_size,
        selected_block_count=config.selected_block_count,
        pilot_offsets=tuple(pilot_offsets),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        random_pilot_offsets=tuple(random_offsets),
        random_coefficients=np.asarray(random_coefficients, dtype=np.float64),
        marginal_block_scores=target.mean(axis=0).astype(np.float64),
        training_group_sha256s=tuple(sorted(group.sha256 for group in groups)),
    )


def _captured_energy(
    features: np.ndarray,
    model: MlpPilotLayerModel,
    selected_blocks: np.ndarray,
    *,
    random_control: bool,
) -> float:
    inventory = model.random_pilot_offsets if random_control else model.pilot_offsets
    total = features.sum(axis=(1, 2))
    captured = np.zeros(len(features), dtype=np.float64)
    for row in range(len(features)):
        neurons = model.selected_neurons(
            selected_blocks[row], random_control=random_control
        )
        flat = features[row].reshape(-1)
        captured[row] = flat[np.asarray(neurons, dtype=np.int64)].sum()
    if any(not offsets for offsets in inventory):
        raise AssertionError("empty pilot inventory passed validation")
    return float(np.sum(captured / np.maximum(total, 1e-300)))


def _evaluate_groups(
    models: Sequence[MlpPilotLayerModel], groups: Sequence[SubspaceObservationGroup]
) -> MlpPilotRouterMetrics:
    by_layer = {model.layer: model for model in models}
    if not groups or {group.layer for group in groups} != set(by_layer):
        raise ValueError("evaluation groups differ from the fitted layer inventory")
    pilot_sum = 0.0
    random_sum = 0.0
    marginal_sum = 0.0
    oracle_sum = 0.0
    rows = 0
    selected_neurons = {model.selected_neuron_count for model in models}
    intermediate = {model.intermediate_dimension for model in models}
    if len(selected_neurons) != 1 or len(intermediate) != 1:
        raise ValueError("layer models disagree on the action ABI")
    for group in groups:
        model = by_layer[group.layer]
        features = _block_features(group, model.block_size)
        pilot_blocks = model.select_blocks_from_full(
            group.gate_projection, group.up_projection
        )
        random_blocks = model.select_blocks_from_full(
            group.gate_projection, group.up_projection, random_control=True
        )
        pilot_sum += _captured_energy(
            features, model, pilot_blocks, random_control=False
        )
        random_sum += _captured_energy(
            features, model, random_blocks, random_control=True
        )
        equal_blocks = math.ceil(model.selected_neuron_count / model.block_size)
        marginal_blocks = np.argsort(-model.marginal_block_scores, kind="stable")[
            :equal_blocks
        ]
        block_energy = features.sum(axis=2)
        total = np.maximum(block_energy.sum(axis=1), 1e-300)
        marginal_sum += float(
            block_energy[:, marginal_blocks].sum(axis=1).dot(1.0 / total)
        )
        oracle = np.sort(block_energy, axis=1)[:, -equal_blocks:].sum(axis=1)
        oracle_sum += float(np.sum(oracle / total))
        rows += group.row_count
    return MlpPilotRouterMetrics(
        row_count=rows,
        selected_neuron_count=next(iter(selected_neurons)),
        intermediate_dimension=next(iter(intermediate)),
        pilot_energy_capture=pilot_sum / rows,
        random_energy_capture=random_sum / rows,
        marginal_energy_capture=marginal_sum / rows,
        oracle_energy_capture=oracle_sum / rows,
    )


@dataclass(frozen=True, slots=True)
class MlpPilotRouterFit:
    model_pin_sha256: str
    corpus_sha256: str
    row_role_sha256: str
    train_prompt_sha256s: tuple[str, ...]
    calibration_prompt_sha256s: tuple[str, ...]
    config: MlpPilotRouterConfig
    models: tuple[MlpPilotLayerModel, ...]
    calibration_metrics: MlpPilotRouterMetrics

    def __post_init__(self) -> None:
        for field in ("model_pin_sha256", "corpus_sha256", "row_role_sha256"):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        train = _sha256s(self.train_prompt_sha256s, field="train_prompt_sha256s")
        calibration = _sha256s(
            self.calibration_prompt_sha256s, field="calibration_prompt_sha256s"
        )
        if set(train) & set(calibration):
            raise ValueError("fit prompt partitions overlap")
        if not isinstance(self.config, MlpPilotRouterConfig):
            raise TypeError("config must be MlpPilotRouterConfig")
        models = tuple(self.models)
        if (
            not models
            or tuple(model.layer for model in models)
            != tuple(sorted(model.layer for model in models))
            or len({model.layer for model in models}) != len(models)
            or any(not isinstance(model, MlpPilotLayerModel) for model in models)
            or any(model.block_size != self.config.block_size for model in models)
            or any(model.pilot_count != self.config.pilot_count for model in models)
            or any(
                model.selected_block_count != self.config.selected_block_count
                for model in models
            )
        ):
            raise ValueError("fit layer model inventory is invalid")
        if not isinstance(self.calibration_metrics, MlpPilotRouterMetrics):
            raise TypeError("calibration_metrics must be MlpPilotRouterMetrics")
        object.__setattr__(self, "train_prompt_sha256s", train)
        object.__setattr__(self, "calibration_prompt_sha256s", calibration)
        object.__setattr__(self, "models", models)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        return _sealed(
            PILOT_ROUTER_FIT_SCHEMA,
            {
                "calibration_metrics": self.calibration_metrics.to_record(),
                "calibration_prompt_sha256s": list(self.calibration_prompt_sha256s),
                "config": self.config.to_record(),
                "corpus_sha256": self.corpus_sha256,
                "model_pin_sha256": self.model_pin_sha256,
                "models": [model.to_record() for model in self.models],
                "row_role_sha256": self.row_role_sha256,
                "train_prompt_sha256s": list(self.train_prompt_sha256s),
                "verifier_sha256": PILOT_ROUTER_VERIFIER_SHA256,
            },
        )

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > _MAX_ARTIFACT_BYTES:
            raise MlpPilotRouterCapacityError("pilot fit exceeds its byte bound")
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> "MlpPilotRouterFit":
        envelope = _strict_json(data, schema=PILOT_ROUTER_FIT_SCHEMA, label="pilot fit")
        body = cast(Mapping[str, object], envelope["body"])
        expected = {
            "calibration_metrics",
            "calibration_prompt_sha256s",
            "config",
            "corpus_sha256",
            "model_pin_sha256",
            "models",
            "row_role_sha256",
            "train_prompt_sha256s",
            "verifier_sha256",
        }
        if (
            set(body) != expected
            or body.get("verifier_sha256") != PILOT_ROUTER_VERIFIER_SHA256
            or not isinstance(body.get("models"), list)
            or not isinstance(body.get("train_prompt_sha256s"), list)
            or not isinstance(body.get("calibration_prompt_sha256s"), list)
        ):
            raise MlpPilotRouterIntegrityError("pilot fit body is invalid")
        try:
            result = cls(
                model_pin_sha256=cast(str, body["model_pin_sha256"]),
                corpus_sha256=cast(str, body["corpus_sha256"]),
                row_role_sha256=cast(str, body["row_role_sha256"]),
                train_prompt_sha256s=tuple(
                    cast(list[str], body["train_prompt_sha256s"])
                ),
                calibration_prompt_sha256s=tuple(
                    cast(list[str], body["calibration_prompt_sha256s"])
                ),
                config=MlpPilotRouterConfig.from_record(body["config"]),
                models=tuple(
                    MlpPilotLayerModel.from_record(row)
                    for row in cast(list[object], body["models"])
                ),
                calibration_metrics=MlpPilotRouterMetrics.from_record(
                    body["calibration_metrics"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise MlpPilotRouterIntegrityError("pilot fit validation failed") from exc
        if result.to_bytes() != data:
            raise MlpPilotRouterIntegrityError("pilot fit reconstruction changed")
        return result


def fit_mlp_pilot_router(
    corpus: SubspaceCorpus,
    *,
    train_prompt_sha256s: Sequence[str],
    calibration_prompt_sha256s: Sequence[str],
    row_role_sha256: str,
    config: MlpPilotRouterConfig = MlpPilotRouterConfig(),
) -> MlpPilotRouterFit:
    if not isinstance(corpus, SubspaceCorpus):
        raise TypeError("corpus must be SubspaceCorpus")
    if not isinstance(config, MlpPilotRouterConfig):
        raise TypeError("config must be MlpPilotRouterConfig")
    train = _sha256s(tuple(train_prompt_sha256s), field="train_prompt_sha256s")
    calibration = _sha256s(
        tuple(calibration_prompt_sha256s), field="calibration_prompt_sha256s"
    )
    if set(train) & set(calibration):
        raise ValueError("train and calibration prompts overlap")
    prompts = tuple(sorted((*train, *calibration)))
    by_key = _rectangular_groups(corpus, prompts)
    layers = tuple(sorted({layer for layer, _prompt in by_key}))
    models = tuple(
        _fit_layer([by_key[layer, prompt] for prompt in train], config)
        for layer in layers
    )
    calibration_groups = tuple(
        sorted(
            (by_key[layer, prompt] for layer in layers for prompt in calibration),
            key=lambda group: (group.logical_time, group.group_sha256),
        )
    )
    return MlpPilotRouterFit(
        model_pin_sha256=corpus.model_pin_sha256,
        corpus_sha256=corpus.sha256,
        row_role_sha256=require_sha256(row_role_sha256, field="row_role_sha256"),
        train_prompt_sha256s=train,
        calibration_prompt_sha256s=calibration,
        config=config,
        models=models,
        calibration_metrics=_evaluate_groups(models, calibration_groups),
    )


def verify_mlp_pilot_router_fit(fit: MlpPilotRouterFit, corpus: SubspaceCorpus) -> None:
    if not isinstance(fit, MlpPilotRouterFit):
        raise TypeError("fit must be MlpPilotRouterFit")
    rebuilt = fit_mlp_pilot_router(
        corpus,
        train_prompt_sha256s=fit.train_prompt_sha256s,
        calibration_prompt_sha256s=fit.calibration_prompt_sha256s,
        row_role_sha256=fit.row_role_sha256,
        config=fit.config,
    )
    if rebuilt.to_bytes() != fit.to_bytes():
        raise MlpPilotRouterIntegrityError("pilot fit differs from its source corpus")


@dataclass(frozen=True, slots=True)
class MlpPilotRouterEvaluation:
    fit_sha256: str
    holdout_corpus_sha256: str
    holdout_authority_sha256: str
    holdout_row_role_sha256: str
    holdout_prompt_sha256s: tuple[str, ...]
    metrics: MlpPilotRouterMetrics

    def __post_init__(self) -> None:
        for field in (
            "fit_sha256",
            "holdout_corpus_sha256",
            "holdout_authority_sha256",
            "holdout_row_role_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        prompts = _sha256s(self.holdout_prompt_sha256s, field="holdout_prompt_sha256s")
        if not isinstance(self.metrics, MlpPilotRouterMetrics):
            raise TypeError("metrics must be MlpPilotRouterMetrics")
        object.__setattr__(self, "holdout_prompt_sha256s", prompts)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        return _sealed(
            PILOT_ROUTER_EVALUATION_SCHEMA,
            {
                "fit_sha256": self.fit_sha256,
                "holdout_authority_sha256": self.holdout_authority_sha256,
                "holdout_corpus_sha256": self.holdout_corpus_sha256,
                "holdout_prompt_sha256s": list(self.holdout_prompt_sha256s),
                "holdout_row_role_sha256": self.holdout_row_role_sha256,
                "metrics": self.metrics.to_record(),
                "verifier_sha256": PILOT_ROUTER_VERIFIER_SHA256,
            },
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_document())

    @classmethod
    def from_bytes(cls, data: bytes) -> "MlpPilotRouterEvaluation":
        envelope = _strict_json(
            data, schema=PILOT_ROUTER_EVALUATION_SCHEMA, label="pilot evaluation"
        )
        body = cast(Mapping[str, object], envelope["body"])
        expected = {
            "fit_sha256",
            "holdout_authority_sha256",
            "holdout_corpus_sha256",
            "holdout_prompt_sha256s",
            "holdout_row_role_sha256",
            "metrics",
            "verifier_sha256",
        }
        if (
            set(body) != expected
            or body.get("verifier_sha256") != PILOT_ROUTER_VERIFIER_SHA256
            or not isinstance(body.get("holdout_prompt_sha256s"), list)
        ):
            raise MlpPilotRouterIntegrityError("pilot evaluation body is invalid")
        try:
            result = cls(
                fit_sha256=cast(str, body["fit_sha256"]),
                holdout_corpus_sha256=cast(str, body["holdout_corpus_sha256"]),
                holdout_authority_sha256=cast(str, body["holdout_authority_sha256"]),
                holdout_row_role_sha256=cast(str, body["holdout_row_role_sha256"]),
                holdout_prompt_sha256s=tuple(
                    cast(list[str], body["holdout_prompt_sha256s"])
                ),
                metrics=MlpPilotRouterMetrics.from_record(body["metrics"]),
            )
        except (TypeError, ValueError) as exc:
            raise MlpPilotRouterIntegrityError(
                "pilot evaluation validation failed"
            ) from exc
        if result.to_bytes() != data:
            raise MlpPilotRouterIntegrityError(
                "pilot evaluation reconstruction changed"
            )
        return result


def evaluate_mlp_pilot_router(
    fit: MlpPilotRouterFit,
    holdout_corpus: SubspaceCorpus,
    *,
    holdout_authority_sha256: str,
    holdout_row_role_sha256: str,
) -> MlpPilotRouterEvaluation:
    if not isinstance(fit, MlpPilotRouterFit):
        raise TypeError("fit must be MlpPilotRouterFit")
    if not isinstance(holdout_corpus, SubspaceCorpus):
        raise TypeError("holdout_corpus must be SubspaceCorpus")
    if holdout_corpus.model_pin_sha256 != fit.model_pin_sha256:
        raise MlpPilotRouterIntegrityError("holdout crosses the fitted model pin")
    prompts = tuple(sorted({group.prompt_sha256 for group in holdout_corpus.groups}))
    if set(prompts) & set((*fit.train_prompt_sha256s, *fit.calibration_prompt_sha256s)):
        raise MlpPilotRouterIntegrityError("holdout prompt identities overlap fit")
    _rectangular_groups(holdout_corpus, prompts)
    layers = {group.layer for group in holdout_corpus.groups}
    if layers != {model.layer for model in fit.models}:
        raise MlpPilotRouterIntegrityError("holdout layer inventory differs from fit")
    return MlpPilotRouterEvaluation(
        fit_sha256=fit.sha256,
        holdout_corpus_sha256=holdout_corpus.sha256,
        holdout_authority_sha256=require_sha256(
            holdout_authority_sha256, field="holdout_authority_sha256"
        ),
        holdout_row_role_sha256=require_sha256(
            holdout_row_role_sha256, field="holdout_row_role_sha256"
        ),
        holdout_prompt_sha256s=prompts,
        metrics=_evaluate_groups(fit.models, holdout_corpus.groups),
    )


def verify_mlp_pilot_router_evaluation(
    evaluation: MlpPilotRouterEvaluation,
    fit: MlpPilotRouterFit,
    holdout_corpus: SubspaceCorpus,
) -> None:
    rebuilt = evaluate_mlp_pilot_router(
        fit,
        holdout_corpus,
        holdout_authority_sha256=evaluation.holdout_authority_sha256,
        holdout_row_role_sha256=evaluation.holdout_row_role_sha256,
    )
    if rebuilt.to_bytes() != evaluation.to_bytes():
        raise MlpPilotRouterIntegrityError(
            "pilot evaluation differs from its holdout corpus"
        )


__all__ = [
    "DEFAULT_RANDOM_SEED_SHA256",
    "PILOT_LAYER_MODEL_SCHEMA",
    "PILOT_ROUTER_CONFIG_SCHEMA",
    "PILOT_ROUTER_EVALUATION_SCHEMA",
    "PILOT_ROUTER_FIT_SCHEMA",
    "PILOT_ROUTER_METRICS_SCHEMA",
    "PILOT_ROUTER_VERIFIER_SHA256",
    "MlpPilotLayerModel",
    "MlpPilotRouterCapacityError",
    "MlpPilotRouterConfig",
    "MlpPilotRouterError",
    "MlpPilotRouterEvaluation",
    "MlpPilotRouterFit",
    "MlpPilotRouterIntegrityError",
    "MlpPilotRouterMetrics",
    "evaluate_mlp_pilot_router",
    "fit_mlp_pilot_router",
    "verify_mlp_pilot_router_evaluation",
    "verify_mlp_pilot_router_fit",
]
