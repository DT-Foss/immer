"""Train-only Markov search over exact Qwen joint gate/up coordinates.

The selector treats one canonical basis as a state and adding one coordinate as
an action.  Every transition is scored by the exact chronological cache
verifier on training evidence.  Calibration chooses among the frozen beam;
holdout is evaluated only after the fit is sealed and is reported both frozen
and online-adaptive.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from .identity import canonical_json_bytes, require_sha256
from .subspace_battery import (
    DEFAULT_RANDOM_SEED_SHA256,
    CollisionMetrics,
    SubspaceCorpus,
    SubspaceKeyModel,
    SubspaceObservationGroup,
    evaluate_subspace_key_model,
    fit_marginal_subspace_model,
    fit_subspace_key_model,
)
from . import subspace_battery as _subspace


MARKOV_SELECTOR_CONFIG_SCHEMA = "immer.qwen-markov-coordinate-config/v1"
MARKOV_NOMINATION_SCHEMA = "immer.qwen-markov-coordinate-nomination/v1"
MARKOV_STATE_SCHEMA = "immer.qwen-markov-coordinate-state/v1"
MARKOV_FIT_SCHEMA = "immer.qwen-markov-coordinate-fit/v1"
MARKOV_HOLDOUT_SCHEMA = "immer.qwen-markov-coordinate-holdout/v1"
MARKOV_SELECTOR_VERIFIER_SHA256 = hashlib.sha256(
    b"immer:qwen-markov-coordinate-selector/train-only-beam+calibration-lock/v1"
).hexdigest()
_MAX_RECEIPT_BYTES = 64 * 1024 * 1024


class MarkovCoordinateSelectorError(RuntimeError):
    pass


class MarkovCoordinateSelectorIntegrityError(MarkovCoordinateSelectorError):
    pass


class MarkovCoordinateSelectorCapacityError(MarkovCoordinateSelectorError):
    pass


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sealed(schema: str, body: Mapping[str, object]) -> dict[str, object]:
    normalized = json.loads(canonical_json_bytes(dict(body)))
    return {"body": normalized, "body_sha256": _digest(normalized), "schema": schema}


def _strict(data: bytes, *, schema: str, label: str) -> Mapping[str, object]:
    if not isinstance(data, bytes) or not data or len(data) > _MAX_RECEIPT_BYTES:
        raise MarkovCoordinateSelectorIntegrityError(f"{label} exceeds its bound")

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(key)
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MarkovCoordinateSelectorIntegrityError(
            f"{label} is not strict JSON"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or set(value) != {"body", "body_sha256", "schema"}
        or value.get("schema") != schema
        or not isinstance(value.get("body"), Mapping)
        or value.get("body_sha256") != _digest(value.get("body"))
        or canonical_json_bytes(value) != data
    ):
        raise MarkovCoordinateSelectorIntegrityError(f"{label} seal is invalid")
    return cast(Mapping[str, object], value)


def _positive(value: object, *, field: str, maximum: int = 1 << 31) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{field} must be a bounded positive integer")
    return value


def _hashes(
    values: Sequence[str], *, field: str, count: int | None = None
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError(f"{field} must be a sequence")
    result = tuple(require_sha256(value, field=field) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{field} must be non-empty and unique")
    if count is not None and len(result) != count:
        raise ValueError(f"{field} has the wrong exact count")
    return result


@dataclass(frozen=True, slots=True)
class MarkovCoordinateSelectorConfig:
    quant_bits: int = 16
    max_depth: int = 4
    beam_width: int = 8
    internal_validation_groups: int = 5
    energy_candidates: int = 96
    fisher_candidates: int = 96
    variance_candidates: int = 64
    random_candidates: int = 64
    random_rank_tags: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    max_candidate_pool: int = 512
    max_evaluated_states: int = 100_000
    fisher_block_dimensions: int = 256
    minimum_raw_key_bits: int = 64
    scale_floor: float = 1e-12
    random_seed_sha256: str = DEFAULT_RANDOM_SEED_SHA256
    max_working_bytes: int = 8 * 1024**3

    def __post_init__(self) -> None:
        bits = _positive(self.quant_bits, field="quant_bits", maximum=16)
        if bits < 2:
            raise ValueError("quant_bits must lie in [2, 16]")
        for field in (
            "max_depth",
            "beam_width",
            "internal_validation_groups",
            "energy_candidates",
            "fisher_candidates",
            "variance_candidates",
            "random_candidates",
            "max_candidate_pool",
            "max_evaluated_states",
            "fisher_block_dimensions",
            "minimum_raw_key_bits",
        ):
            object.__setattr__(
                self, field, _positive(getattr(self, field), field=field)
            )
        object.__setattr__(
            self,
            "max_working_bytes",
            _positive(
                self.max_working_bytes,
                field="max_working_bytes",
                maximum=(1 << 63) - 1,
            ),
        )
        tags = tuple(self.random_rank_tags)
        if (
            not tags
            or tags != tuple(sorted(set(tags)))
            or any(
                isinstance(tag, bool) or not isinstance(tag, int) or tag < 1
                for tag in tags
            )
        ):
            raise ValueError("random_rank_tags must be sorted unique positive integers")
        floor = float(self.scale_floor)
        if not math.isfinite(floor) or floor <= 0.0:
            raise ValueError("scale_floor must be finite and positive")
        object.__setattr__(
            self,
            "random_seed_sha256",
            require_sha256(self.random_seed_sha256, field="random_seed_sha256"),
        )
        object.__setattr__(self, "random_rank_tags", tags)
        object.__setattr__(self, "scale_floor", floor)

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "beam_width": self.beam_width,
            "energy_candidates": self.energy_candidates,
            "fisher_block_dimensions": self.fisher_block_dimensions,
            "fisher_candidates": self.fisher_candidates,
            "internal_validation_groups": self.internal_validation_groups,
            "max_candidate_pool": self.max_candidate_pool,
            "max_depth": self.max_depth,
            "max_evaluated_states": self.max_evaluated_states,
            "max_working_bytes": self.max_working_bytes,
            "minimum_raw_key_bits": self.minimum_raw_key_bits,
            "quant_bits": self.quant_bits,
            "random_candidates": self.random_candidates,
            "random_rank_tags": list(self.random_rank_tags),
            "random_seed_sha256": self.random_seed_sha256,
            "scale_floor": self.scale_floor,
            "schema": MARKOV_SELECTOR_CONFIG_SCHEMA,
            "variance_candidates": self.variance_candidates,
        }

    @classmethod
    def from_record(cls, value: object) -> "MarkovCoordinateSelectorConfig":
        if not isinstance(value, Mapping):
            raise MarkovCoordinateSelectorIntegrityError("selector config is invalid")
        expected = {
            "beam_width",
            "energy_candidates",
            "fisher_block_dimensions",
            "fisher_candidates",
            "internal_validation_groups",
            "max_candidate_pool",
            "max_depth",
            "max_evaluated_states",
            "max_working_bytes",
            "minimum_raw_key_bits",
            "quant_bits",
            "random_candidates",
            "random_rank_tags",
            "random_seed_sha256",
            "scale_floor",
            "schema",
            "variance_candidates",
        }
        if (
            set(value) != expected
            or value.get("schema") != MARKOV_SELECTOR_CONFIG_SCHEMA
        ):
            raise MarkovCoordinateSelectorIntegrityError(
                "selector config shape changed"
            )
        body = dict(value)
        body.pop("schema")
        body["random_rank_tags"] = tuple(cast(list[int], body["random_rank_tags"]))
        try:
            return cls(**body)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise MarkovCoordinateSelectorIntegrityError(
                "selector config validation failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class CoordinateNominationReceipt:
    config_sha256: str
    source_group_sha256s: tuple[str, ...]
    intermediate_dimension: int
    arm_indices: tuple[tuple[str, tuple[int, ...]], ...]
    arm_score_sha256s: tuple[tuple[str, str], ...]
    candidate_pool: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "config_sha256",
            require_sha256(self.config_sha256, field="config_sha256"),
        )
        object.__setattr__(
            self,
            "source_group_sha256s",
            _hashes(self.source_group_sha256s, field="source_group_sha256s"),
        )
        dimension = _positive(
            self.intermediate_dimension,
            field="intermediate_dimension",
            maximum=1_000_000,
        )
        arms = tuple((name, tuple(indices)) for name, indices in self.arm_indices)
        if (
            not arms
            or tuple(name for name, _ in arms)
            != tuple(sorted(name for name, _ in arms))
            or len({name for name, _ in arms}) != len(arms)
            or any(
                not name
                or not indices
                or len(set(indices)) != len(indices)
                or any(not 0 <= index < dimension for index in indices)
                for name, indices in arms
            )
        ):
            raise ValueError("nomination arm inventory is invalid")
        score_rows = tuple(
            (name, require_sha256(digest, field="arm_score_sha256s"))
            for name, digest in self.arm_score_sha256s
        )
        if tuple(name for name, _ in score_rows) != tuple(name for name, _ in arms):
            raise ValueError("nomination score inventory differs from arms")
        pool = tuple(self.candidate_pool)
        if (
            not pool
            or len(set(pool)) != len(pool)
            or any(not 0 <= index < dimension for index in pool)
        ):
            raise ValueError("candidate_pool is invalid")
        object.__setattr__(self, "intermediate_dimension", dimension)
        object.__setattr__(self, "arm_indices", arms)
        object.__setattr__(self, "arm_score_sha256s", score_rows)
        object.__setattr__(self, "candidate_pool", pool)

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "arm_indices": [
                {"indices": list(indices), "name": name}
                for name, indices in self.arm_indices
            ],
            "arm_score_sha256s": [
                {"name": name, "sha256": digest}
                for name, digest in self.arm_score_sha256s
            ],
            "candidate_pool": list(self.candidate_pool),
            "config_sha256": self.config_sha256,
            "intermediate_dimension": self.intermediate_dimension,
            "schema": MARKOV_NOMINATION_SCHEMA,
            "source_group_sha256s": list(self.source_group_sha256s),
        }

    @classmethod
    def from_record(cls, value: object) -> "CoordinateNominationReceipt":
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "arm_indices",
                "arm_score_sha256s",
                "candidate_pool",
                "config_sha256",
                "intermediate_dimension",
                "schema",
                "source_group_sha256s",
            }
            or value.get("schema") != MARKOV_NOMINATION_SCHEMA
            or not isinstance(value.get("arm_indices"), list)
            or not isinstance(value.get("arm_score_sha256s"), list)
            or not isinstance(value.get("candidate_pool"), list)
            or not isinstance(value.get("source_group_sha256s"), list)
        ):
            raise MarkovCoordinateSelectorIntegrityError(
                "nomination receipt is invalid"
            )
        try:
            arms = tuple(
                (cast(str, row["name"]), tuple(cast(list[int], row["indices"])))
                for row in cast(list[Mapping[str, object]], value["arm_indices"])
            )
            scores = tuple(
                (cast(str, row["name"]), cast(str, row["sha256"]))
                for row in cast(list[Mapping[str, object]], value["arm_score_sha256s"])
            )
            return cls(
                config_sha256=cast(str, value["config_sha256"]),
                source_group_sha256s=tuple(
                    cast(list[str], value["source_group_sha256s"])
                ),
                intermediate_dimension=cast(int, value["intermediate_dimension"]),
                arm_indices=arms,
                arm_score_sha256s=scores,
                candidate_pool=tuple(cast(list[int], value["candidate_pool"])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MarkovCoordinateSelectorIntegrityError(
                "nomination receipt validation failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class MarkovCoordinateState:
    depth: int
    basis_indices: tuple[int, ...]
    parent_state_sha256: str | None
    action_coordinate: int
    model: SubspaceKeyModel

    def __post_init__(self) -> None:
        depth = _positive(self.depth, field="depth")
        basis = tuple(self.basis_indices)
        if (
            len(basis) != depth
            or basis != tuple(sorted(set(basis)))
            or self.action_coordinate not in basis
        ):
            raise ValueError("Markov coordinate state basis/action is invalid")
        parent = self.parent_state_sha256
        if depth == 1:
            if parent is not None:
                raise ValueError("depth-one state cannot carry a parent")
        else:
            parent = require_sha256(parent, field="parent_state_sha256")
        if (
            not isinstance(self.model, SubspaceKeyModel)
            or self.model.family != "markov"
        ):
            raise TypeError("Markov state requires a markov key model")
        if self.model.basis_indices != basis:
            raise ValueError("Markov state basis differs from its exact model")
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "basis_indices", basis)
        object.__setattr__(self, "parent_state_sha256", parent)

    @property
    def raw_key_bits(self) -> int:
        return 2 * self.depth * self.model.quant_bits

    @property
    def reward_key(self) -> tuple[object, ...]:
        metrics = self.model.calibration_metrics
        wrong = self.model.train_wrong_collision_rows + metrics.wrong_collisions
        return (
            int(bool(wrong)),
            wrong,
            -metrics.exact_verified_hits,
            metrics.misses,
            metrics.stored_bytes,
            self.basis_indices,
        )

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "action_coordinate": self.action_coordinate,
            "basis_indices": list(self.basis_indices),
            "depth": self.depth,
            "model": self.model.to_record(),
            "parent_state_sha256": self.parent_state_sha256,
            "raw_key_bits": self.raw_key_bits,
            "schema": MARKOV_STATE_SCHEMA,
        }

    @classmethod
    def from_record(cls, value: object) -> "MarkovCoordinateState":
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "action_coordinate",
                "basis_indices",
                "depth",
                "model",
                "parent_state_sha256",
                "raw_key_bits",
                "schema",
            }
            or value.get("schema") != MARKOV_STATE_SCHEMA
            or not isinstance(value.get("basis_indices"), list)
        ):
            raise MarkovCoordinateSelectorIntegrityError("Markov state is invalid")
        try:
            result = cls(
                depth=cast(int, value["depth"]),
                basis_indices=tuple(cast(list[int], value["basis_indices"])),
                parent_state_sha256=cast(str | None, value["parent_state_sha256"]),
                action_coordinate=cast(int, value["action_coordinate"]),
                model=SubspaceKeyModel.from_record(value["model"]),
            )
        except (TypeError, ValueError) as exc:
            raise MarkovCoordinateSelectorIntegrityError(
                "Markov state validation failed"
            ) from exc
        if value.get("raw_key_bits") != result.raw_key_bits:
            raise MarkovCoordinateSelectorIntegrityError("Markov state bits changed")
        return result


@dataclass(frozen=True, slots=True)
class MarkovCoordinateSelectorFit:
    corpus_sha256: str
    model_pin_sha256: str
    train_group_sha256s: tuple[str, ...]
    calibration_group_sha256s: tuple[str, ...]
    config: MarkovCoordinateSelectorConfig
    nomination: CoordinateNominationReceipt
    beam_states: tuple[MarkovCoordinateState, ...]
    evaluated_state_count: int
    evaluated_inventory_sha256: str
    locked_model: SubspaceKeyModel
    energy_control: SubspaceKeyModel
    random_control: SubspaceKeyModel
    full_control: SubspaceKeyModel
    marginal_control: SubspaceKeyModel
    selection_reason: str

    def __post_init__(self) -> None:
        for field in (
            "corpus_sha256",
            "model_pin_sha256",
            "evaluated_inventory_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        object.__setattr__(
            self,
            "train_group_sha256s",
            _hashes(self.train_group_sha256s, field="train_group_sha256s"),
        )
        object.__setattr__(
            self,
            "calibration_group_sha256s",
            _hashes(self.calibration_group_sha256s, field="calibration_group_sha256s"),
        )
        if not isinstance(self.config, MarkovCoordinateSelectorConfig):
            raise TypeError("config must be MarkovCoordinateSelectorConfig")
        if not isinstance(self.nomination, CoordinateNominationReceipt):
            raise TypeError("nomination must be CoordinateNominationReceipt")
        states = tuple(self.beam_states)
        if not states or any(
            not isinstance(state, MarkovCoordinateState) for state in states
        ):
            raise TypeError("beam_states are invalid")
        if len({state.sha256 for state in states}) != len(states):
            raise ValueError("beam_states are duplicated")
        evaluated = _positive(self.evaluated_state_count, field="evaluated_state_count")
        if evaluated < len(states) or evaluated > self.config.max_evaluated_states:
            raise ValueError("evaluated_state_count is inconsistent")
        for model, family in (
            (self.locked_model, "markov"),
            (self.energy_control, "candidate"),
            (self.random_control, "random"),
            (self.full_control, "full"),
            (self.marginal_control, "marginal"),
        ):
            if not isinstance(model, SubspaceKeyModel) or model.family != family:
                raise TypeError(f"{family} model is invalid")
        if (
            self.locked_model.k != self.energy_control.k
            or self.locked_model.k != self.random_control.k
            or self.locked_model.quant_bits != self.energy_control.quant_bits
            or self.locked_model.quant_bits != self.random_control.quant_bits
        ):
            raise ValueError("selector controls do not share k/bits")
        if (
            self.locked_model.train_wrong_collision_rows
            or self.locked_model.calibration_metrics.wrong_collisions
            or not self.locked_model.calibration_metrics.exact_verified_hits
            or 2 * self.locked_model.k * self.locked_model.quant_bits
            < self.config.minimum_raw_key_bits
        ):
            raise ValueError("locked model violates safety/calibration gates")
        reason = self.selection_reason
        if reason != "minimum-safe-bits+zero-wrong+verified-calibration-hit":
            raise ValueError("selection_reason is invalid")
        object.__setattr__(self, "beam_states", states)
        object.__setattr__(self, "evaluated_state_count", evaluated)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        body = {
            "beam_states": [state.to_record() for state in self.beam_states],
            "calibration_group_sha256s": list(self.calibration_group_sha256s),
            "config": self.config.to_record(),
            "corpus_sha256": self.corpus_sha256,
            "energy_control": self.energy_control.to_record(),
            "evaluated_inventory_sha256": self.evaluated_inventory_sha256,
            "evaluated_state_count": self.evaluated_state_count,
            "locked_model": self.locked_model.to_record(),
            "full_control": self.full_control.to_record(),
            "marginal_control": self.marginal_control.to_record(),
            "model_pin_sha256": self.model_pin_sha256,
            "nomination": self.nomination.to_record(),
            "random_control": self.random_control.to_record(),
            "selection_reason": self.selection_reason,
            "train_group_sha256s": list(self.train_group_sha256s),
            "verifier_sha256": MARKOV_SELECTOR_VERIFIER_SHA256,
        }
        return _sealed(MARKOV_FIT_SCHEMA, body)

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > _MAX_RECEIPT_BYTES:
            raise MarkovCoordinateSelectorCapacityError(
                "selector fit exceeds its bound"
            )
        return data

    @classmethod
    def from_bytes(
        cls, data: bytes, *, corpus: SubspaceCorpus
    ) -> "MarkovCoordinateSelectorFit":
        envelope = _strict(data, schema=MARKOV_FIT_SCHEMA, label="selector fit")
        body = cast(Mapping[str, object], envelope["body"])
        expected = {
            "beam_states",
            "calibration_group_sha256s",
            "config",
            "corpus_sha256",
            "energy_control",
            "evaluated_inventory_sha256",
            "evaluated_state_count",
            "locked_model",
            "full_control",
            "marginal_control",
            "model_pin_sha256",
            "nomination",
            "random_control",
            "selection_reason",
            "train_group_sha256s",
            "verifier_sha256",
        }
        if (
            set(body) != expected
            or body.get("verifier_sha256") != MARKOV_SELECTOR_VERIFIER_SHA256
            or not isinstance(body.get("beam_states"), list)
            or not isinstance(body.get("train_group_sha256s"), list)
            or not isinstance(body.get("calibration_group_sha256s"), list)
        ):
            raise MarkovCoordinateSelectorIntegrityError("selector fit body is invalid")
        try:
            result = cls(
                corpus_sha256=cast(str, body["corpus_sha256"]),
                model_pin_sha256=cast(str, body["model_pin_sha256"]),
                train_group_sha256s=tuple(cast(list[str], body["train_group_sha256s"])),
                calibration_group_sha256s=tuple(
                    cast(list[str], body["calibration_group_sha256s"])
                ),
                config=MarkovCoordinateSelectorConfig.from_record(body["config"]),
                nomination=CoordinateNominationReceipt.from_record(body["nomination"]),
                beam_states=tuple(
                    MarkovCoordinateState.from_record(row)
                    for row in cast(list[object], body["beam_states"])
                ),
                evaluated_state_count=cast(int, body["evaluated_state_count"]),
                evaluated_inventory_sha256=cast(
                    str, body["evaluated_inventory_sha256"]
                ),
                locked_model=SubspaceKeyModel.from_record(body["locked_model"]),
                energy_control=SubspaceKeyModel.from_record(body["energy_control"]),
                random_control=SubspaceKeyModel.from_record(body["random_control"]),
                full_control=SubspaceKeyModel.from_record(body["full_control"]),
                marginal_control=SubspaceKeyModel.from_record(body["marginal_control"]),
                selection_reason=cast(str, body["selection_reason"]),
            )
        except (TypeError, ValueError) as exc:
            raise MarkovCoordinateSelectorIntegrityError(
                "selector fit reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise MarkovCoordinateSelectorIntegrityError("selector fit bytes changed")
        result.verify_against(corpus)
        return result

    def verify_against(self, corpus: SubspaceCorpus) -> None:
        train_indices = tuple(
            index
            for index, group in enumerate(corpus.groups)
            if group.group_sha256 in set(self.train_group_sha256s)
        )
        calibration_indices = tuple(
            index
            for index, group in enumerate(corpus.groups)
            if group.group_sha256 in set(self.calibration_group_sha256s)
        )
        recomputed = fit_markov_coordinate_selector(
            corpus,
            train_group_indices=train_indices,
            calibration_group_indices=calibration_indices,
            config=self.config,
        )
        if recomputed.to_bytes() != self.to_bytes():
            raise MarkovCoordinateSelectorIntegrityError(
                "selector fit differs from complete recomputation"
            )


@dataclass(frozen=True, slots=True)
class MarkovHoldoutResult:
    name: str
    model_sha256: str
    frozen_metrics: CollisionMetrics
    adaptive_metrics: CollisionMetrics
    promoted: bool

    def __post_init__(self) -> None:
        if self.name not in {"energy", "full", "marginal", "markov", "random"}:
            raise ValueError("holdout result name is invalid")
        object.__setattr__(
            self,
            "model_sha256",
            require_sha256(self.model_sha256, field="model_sha256"),
        )
        if not isinstance(self.frozen_metrics, CollisionMetrics) or not isinstance(
            self.adaptive_metrics, CollisionMetrics
        ):
            raise TypeError("holdout metrics are invalid")
        if not isinstance(self.promoted, bool) or self.promoted != (
            self.name == "markov"
            and not self.adaptive_metrics.any_wrong_collision
            and self.adaptive_metrics.exact_verified_hits > 0
            and not self.frozen_metrics.any_wrong_collision
        ):
            raise ValueError("holdout promotion verdict is inconsistent")

    def to_record(self) -> dict[str, object]:
        return {
            "adaptive_metrics": self.adaptive_metrics.to_record(),
            "frozen_metrics": self.frozen_metrics.to_record(),
            "model_sha256": self.model_sha256,
            "name": self.name,
            "promoted": self.promoted,
        }

    @classmethod
    def from_record(cls, value: object) -> "MarkovHoldoutResult":
        if not isinstance(value, Mapping) or set(value) != {
            "adaptive_metrics",
            "frozen_metrics",
            "model_sha256",
            "name",
            "promoted",
        }:
            raise MarkovCoordinateSelectorIntegrityError("holdout result is invalid")
        return cls(
            name=cast(str, value["name"]),
            model_sha256=cast(str, value["model_sha256"]),
            frozen_metrics=CollisionMetrics.from_record(value["frozen_metrics"]),
            adaptive_metrics=CollisionMetrics.from_record(value["adaptive_metrics"]),
            promoted=cast(bool, value["promoted"]),
        )


@dataclass(frozen=True, slots=True)
class MarkovCoordinateSelectorEvaluation:
    fit_sha256: str
    holdout_group_sha256s: tuple[str, ...]
    holdout_authority_sha256: str
    results: tuple[MarkovHoldoutResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "fit_sha256", require_sha256(self.fit_sha256, field="fit_sha256")
        )
        object.__setattr__(
            self,
            "holdout_group_sha256s",
            _hashes(self.holdout_group_sha256s, field="holdout_group_sha256s"),
        )
        object.__setattr__(
            self,
            "holdout_authority_sha256",
            require_sha256(
                self.holdout_authority_sha256, field="holdout_authority_sha256"
            ),
        )
        results = tuple(self.results)
        if tuple(result.name for result in results) != (
            "markov",
            "energy",
            "random",
            "full",
            "marginal",
        ):
            raise ValueError("holdout result inventory is invalid")
        object.__setattr__(self, "results", results)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        return _sealed(
            MARKOV_HOLDOUT_SCHEMA,
            {
                "fit_sha256": self.fit_sha256,
                "holdout_authority_sha256": self.holdout_authority_sha256,
                "holdout_group_sha256s": list(self.holdout_group_sha256s),
                "results": [result.to_record() for result in self.results],
                "verifier_sha256": MARKOV_SELECTOR_VERIFIER_SHA256,
            },
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_document())

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        fit: MarkovCoordinateSelectorFit,
        fit_corpus: SubspaceCorpus,
        holdout_corpus: SubspaceCorpus,
        holdout_authority_sha256: str,
    ) -> "MarkovCoordinateSelectorEvaluation":
        envelope = _strict(data, schema=MARKOV_HOLDOUT_SCHEMA, label="selector holdout")
        body = cast(Mapping[str, object], envelope["body"])
        if (
            set(body)
            != {
                "fit_sha256",
                "holdout_authority_sha256",
                "holdout_group_sha256s",
                "results",
                "verifier_sha256",
            }
            or body.get("verifier_sha256") != MARKOV_SELECTOR_VERIFIER_SHA256
            or not isinstance(body.get("holdout_group_sha256s"), list)
            or not isinstance(body.get("results"), list)
        ):
            raise MarkovCoordinateSelectorIntegrityError(
                "selector holdout body is invalid"
            )
        result = cls(
            fit_sha256=cast(str, body["fit_sha256"]),
            holdout_group_sha256s=tuple(cast(list[str], body["holdout_group_sha256s"])),
            holdout_authority_sha256=cast(str, body["holdout_authority_sha256"]),
            results=tuple(
                MarkovHoldoutResult.from_record(row)
                for row in cast(list[object], body["results"])
            ),
        )
        if result.to_bytes() != data:
            raise MarkovCoordinateSelectorIntegrityError(
                "selector holdout bytes changed"
            )
        recomputed = evaluate_markov_coordinate_selector(
            fit,
            fit_corpus=fit_corpus,
            holdout_corpus=holdout_corpus,
            holdout_authority_sha256=holdout_authority_sha256,
        )
        if recomputed.to_bytes() != data:
            raise MarkovCoordinateSelectorIntegrityError(
                "selector holdout differs from complete recomputation"
            )
        return result


def _indices(values: Sequence[int], *, count: int, field: str) -> tuple[int, ...]:
    result = tuple(values)
    if (
        not result
        or result != tuple(sorted(set(result)))
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < count
            for index in result
        )
    ):
        raise ValueError(f"{field} must be sorted unique corpus indices")
    return result


def _rank(values: NDArray[np.float64], count: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(int(values.shape[0])),
            key=lambda index: (-float(values[index]), index),
        )[:count]
    )


def _matrix(
    groups: Sequence[SubspaceObservationGroup], field: str
) -> NDArray[np.float64]:
    return cast(
        NDArray[np.float64],
        np.concatenate([getattr(group, field) for group in groups], axis=0),
    )


def _labels(groups: Sequence[SubspaceObservationGroup]) -> NDArray[np.object_]:
    return np.asarray(
        [label for group in groups for label in group.output_payload_sha256s],
        dtype=object,
    )


def _fisher_scores(
    gate: NDArray[np.float64],
    up: NDArray[np.float64],
    labels: NDArray[np.object_],
    *,
    block_dimensions: int,
) -> NDArray[np.float64]:
    _unique, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    dimension = int(gate.shape[1])
    score = np.zeros(dimension, dtype=np.float64)
    for start in range(0, dimension, block_dimensions):
        stop = min(start + block_dimensions, dimension)
        width = stop - start
        sum_gate = np.zeros((len(counts), width), dtype=np.float64)
        sum_up = np.zeros((len(counts), width), dtype=np.float64)
        np.add.at(sum_gate, inverse, gate[:, start:stop])
        np.add.at(sum_up, inverse, up[:, start:stop])
        mean_gate = sum_gate / counts[:, None]
        mean_up = sum_up / counts[:, None]
        global_gate = gate[:, start:stop].mean(axis=0)
        global_up = up[:, start:stop].mean(axis=0)
        between = (counts[:, None] * (mean_gate - global_gate) ** 2).sum(axis=0)
        between += (counts[:, None] * (mean_up - global_up) ** 2).sum(axis=0)
        within = (gate[:, start:stop] ** 2).sum(axis=0)
        within -= (counts[:, None] * mean_gate * mean_gate).sum(axis=0)
        within += (up[:, start:stop] ** 2).sum(axis=0)
        within -= (counts[:, None] * mean_up * mean_up).sum(axis=0)
        score[start:stop] = between / np.maximum(within, 1e-30)
    if not np.isfinite(score).all():
        raise MarkovCoordinateSelectorIntegrityError("Fisher nomination is non-finite")
    return score


def _nominate(
    groups: Sequence[SubspaceObservationGroup], config: MarkovCoordinateSelectorConfig
) -> CoordinateNominationReceipt:
    gate = _matrix(groups, "gate_projection")
    up = _matrix(groups, "up_projection")
    labels = _labels(groups)
    required = int(gate.nbytes + up.nbytes + 8 * gate.shape[1] * 4)
    if required > config.max_working_bytes:
        raise MarkovCoordinateSelectorCapacityError(
            "selector nomination exceeds max_working_bytes"
        )
    energy = np.mean(gate * gate + up * up, axis=0)
    variance = np.var(gate, axis=0) + np.var(up, axis=0)
    fisher = _fisher_scores(
        gate,
        up,
        labels,
        block_dimensions=config.fisher_block_dimensions,
    )
    arms: dict[str, tuple[int, ...]] = {
        "energy": _rank(energy, min(config.energy_candidates, gate.shape[1])),
        "fisher": _rank(fisher, min(config.fisher_candidates, gate.shape[1])),
        "variance": _rank(variance, min(config.variance_candidates, gate.shape[1])),
    }
    for tag in config.random_rank_tags:
        arms[f"random-{tag:04d}"] = _subspace._random_basis(
            config.random_seed_sha256, int(gate.shape[1]), tag
        )[: config.random_candidates]
    ordered: list[int] = []
    for name in sorted(arms):
        for index in arms[name]:
            if index not in ordered:
                ordered.append(index)
            if len(ordered) == config.max_candidate_pool:
                break
        if len(ordered) == config.max_candidate_pool:
            break
    scores = {
        "energy": energy,
        "fisher": fisher,
        "variance": variance,
    }
    score_hashes = tuple(
        (
            name,
            (
                hashlib.sha256(
                    np.asarray(scores[name], dtype="<f8", order="C").tobytes(order="C")
                ).hexdigest()
                if name in scores
                else _digest(
                    {
                        "indices": list(indices),
                        "name": name,
                        "seed_sha256": config.random_seed_sha256,
                    }
                )
            ),
        )
        for name, indices in sorted(arms.items())
    )
    return CoordinateNominationReceipt(
        config_sha256=config.sha256,
        source_group_sha256s=tuple(group.group_sha256 for group in groups),
        intermediate_dimension=int(gate.shape[1]),
        arm_indices=tuple(sorted(arms.items())),
        arm_score_sha256s=score_hashes,
        candidate_pool=tuple(ordered),
    )


def _state_score(state: MarkovCoordinateState) -> tuple[object, ...]:
    return state.reward_key


def fit_markov_coordinate_selector(
    corpus: SubspaceCorpus,
    *,
    train_group_indices: Sequence[int],
    calibration_group_indices: Sequence[int],
    config: MarkovCoordinateSelectorConfig | None = None,
) -> MarkovCoordinateSelectorFit:
    if not isinstance(corpus, SubspaceCorpus):
        raise TypeError("corpus must be a SubspaceCorpus")
    selected_config = MarkovCoordinateSelectorConfig() if config is None else config
    if not isinstance(selected_config, MarkovCoordinateSelectorConfig):
        raise TypeError("config must be MarkovCoordinateSelectorConfig")
    train_indices = _indices(
        train_group_indices, count=len(corpus.groups), field="train_group_indices"
    )
    calibration_indices = _indices(
        calibration_group_indices,
        count=len(corpus.groups),
        field="calibration_group_indices",
    )
    if set(train_indices) & set(calibration_indices) or max(train_indices) >= min(
        calibration_indices
    ):
        raise ValueError("selector calibration must follow disjoint training groups")
    train = tuple(corpus.groups[index] for index in train_indices)
    calibration = tuple(corpus.groups[index] for index in calibration_indices)
    validation_count = selected_config.internal_validation_groups
    if len(train) <= validation_count:
        raise ValueError("training split is too short for internal validation")
    search_train = train[:-validation_count]
    search_validation = train[-validation_count:]
    nomination = _nominate(search_train, selected_config)
    pool = nomination.candidate_pool
    beam: tuple[tuple[int, ...], ...] = ((),)
    beam_states: list[MarkovCoordinateState] = []
    state_by_basis: dict[tuple[int, ...], MarkovCoordinateState] = {}
    evaluated_records: list[dict[str, object]] = []
    evaluated_count = 0
    for depth in range(1, selected_config.max_depth + 1):
        candidates: dict[tuple[int, ...], MarkovCoordinateState] = {}
        for parent_basis in beam:
            parent_state = state_by_basis.get(parent_basis)
            for coordinate in pool:
                if coordinate in parent_basis:
                    continue
                basis = tuple(sorted((*parent_basis, coordinate)))
                if basis in candidates:
                    continue
                evaluated_count += 1
                if evaluated_count > selected_config.max_evaluated_states:
                    raise MarkovCoordinateSelectorCapacityError(
                        "selector exceeded max_evaluated_states"
                    )
                model = fit_subspace_key_model(
                    search_train,
                    search_validation,
                    basis_indices=basis,
                    quant_bits=selected_config.quant_bits,
                    family="markov",
                    scale_floor=selected_config.scale_floor,
                )
                state = MarkovCoordinateState(
                    depth=depth,
                    basis_indices=basis,
                    parent_state_sha256=(
                        None if parent_state is None else parent_state.sha256
                    ),
                    action_coordinate=coordinate,
                    model=model,
                )
                candidates[basis] = state
                evaluated_records.append(
                    {
                        "basis_indices": list(basis),
                        "model_sha256": model.sha256,
                        "parent_state_sha256": state.parent_state_sha256,
                        "reward_key": list(state.reward_key[:-1]),
                    }
                )
        ranked = tuple(sorted(candidates.values(), key=_state_score))
        selected = ranked[: selected_config.beam_width]
        if not selected:
            raise MarkovCoordinateSelectorIntegrityError("selector beam became empty")
        beam = tuple(state.basis_indices for state in selected)
        for state in selected:
            state_by_basis[state.basis_indices] = state
            beam_states.append(state)
    official: list[tuple[tuple[object, ...], SubspaceKeyModel]] = []
    for basis in dict.fromkeys(state.basis_indices for state in beam_states):
        model = fit_subspace_key_model(
            train,
            calibration,
            basis_indices=basis,
            quant_bits=selected_config.quant_bits,
            family="markov",
            scale_floor=selected_config.scale_floor,
        )
        metrics = model.calibration_metrics
        safe = (
            2 * model.k * model.quant_bits >= selected_config.minimum_raw_key_bits
            and model.train_wrong_collision_rows == 0
            and metrics.wrong_collisions == 0
            and metrics.exact_verified_hits > 0
        )
        official.append(
            (
                (
                    int(not safe),
                    model.k,
                    model.train_wrong_collision_rows + metrics.wrong_collisions,
                    -metrics.exact_verified_hits,
                    metrics.stored_bytes,
                    model.basis_indices,
                ),
                model,
            )
        )
    official.sort(key=lambda row: row[0])
    if not official or official[0][0][0]:
        raise MarkovCoordinateSelectorIntegrityError(
            "selector found no calibration-safe entropy-guarded basis"
        )
    locked = official[0][1]
    energy = np.mean(
        _matrix(train, "gate_projection") ** 2 + _matrix(train, "up_projection") ** 2,
        axis=0,
    )
    energy_basis = _rank(energy, locked.k)
    random_basis = _subspace._random_basis(
        selected_config.random_seed_sha256,
        corpus.intermediate_dimension,
        locked.k,
    )
    energy_control = fit_subspace_key_model(
        train,
        calibration,
        basis_indices=energy_basis,
        quant_bits=locked.quant_bits,
        family="candidate",
        scale_floor=selected_config.scale_floor,
    )
    random_control = fit_subspace_key_model(
        train,
        calibration,
        basis_indices=random_basis,
        quant_bits=locked.quant_bits,
        family="random",
        scale_floor=selected_config.scale_floor,
    )
    full_control = fit_subspace_key_model(
        train,
        calibration,
        basis_indices=tuple(range(corpus.intermediate_dimension)),
        quant_bits=locked.quant_bits,
        family="full",
        scale_floor=selected_config.scale_floor,
    )
    marginal_control = fit_marginal_subspace_model(train, calibration)
    return MarkovCoordinateSelectorFit(
        corpus_sha256=corpus.sha256,
        model_pin_sha256=corpus.model_pin_sha256,
        train_group_sha256s=tuple(group.group_sha256 for group in train),
        calibration_group_sha256s=tuple(group.group_sha256 for group in calibration),
        config=selected_config,
        nomination=nomination,
        beam_states=tuple(beam_states),
        evaluated_state_count=evaluated_count,
        evaluated_inventory_sha256=_digest(
            {
                "config_sha256": selected_config.sha256,
                "records": evaluated_records,
                "schema": "immer.qwen-markov-evaluated-state-inventory/v1",
            }
        ),
        locked_model=locked,
        energy_control=energy_control,
        random_control=random_control,
        full_control=full_control,
        marginal_control=marginal_control,
        selection_reason="minimum-safe-bits+zero-wrong+verified-calibration-hit",
    )


def evaluate_markov_coordinate_selector(
    fit: MarkovCoordinateSelectorFit,
    *,
    fit_corpus: SubspaceCorpus,
    holdout_corpus: SubspaceCorpus,
    holdout_authority_sha256: str,
) -> MarkovCoordinateSelectorEvaluation:
    if not isinstance(fit, MarkovCoordinateSelectorFit):
        raise TypeError("fit must be MarkovCoordinateSelectorFit")
    if not isinstance(fit_corpus, SubspaceCorpus) or not isinstance(
        holdout_corpus, SubspaceCorpus
    ):
        raise TypeError("fit_corpus and holdout_corpus must be SubspaceCorpus values")
    fit.verify_against(fit_corpus)
    if (
        fit_corpus.model_pin_sha256 != fit.model_pin_sha256
        or holdout_corpus.model_pin_sha256 != fit.model_pin_sha256
    ):
        raise MarkovCoordinateSelectorIntegrityError(
            "selector fit/holdout crosses Qwen model pins"
        )
    authority = require_sha256(
        holdout_authority_sha256, field="holdout_authority_sha256"
    )
    train_map = {group.group_sha256: group for group in fit_corpus.groups}
    try:
        train = tuple(train_map[digest] for digest in fit.train_group_sha256s)
        calibration = tuple(
            train_map[digest] for digest in fit.calibration_group_sha256s
        )
    except KeyError as exc:
        raise MarkovCoordinateSelectorIntegrityError(
            "fit corpus lost a locked selector group"
        ) from exc
    holdout = tuple(holdout_corpus.groups)
    prior_groups = set(fit.train_group_sha256s) | set(fit.calibration_group_sha256s)
    if prior_groups & {group.group_sha256 for group in holdout}:
        raise MarkovCoordinateSelectorIntegrityError(
            "holdout overlaps selector fit groups"
        )
    results = []
    for name, model in (
        ("markov", fit.locked_model),
        ("energy", fit.energy_control),
        ("random", fit.random_control),
        ("full", fit.full_control),
        ("marginal", fit.marginal_control),
    ):
        frozen = evaluate_subspace_key_model(
            model, train, calibration, holdout, adaptive=False
        )
        adaptive = evaluate_subspace_key_model(
            model, train, calibration, holdout, adaptive=True
        )
        results.append(
            MarkovHoldoutResult(
                name=name,
                model_sha256=model.sha256,
                frozen_metrics=frozen,
                adaptive_metrics=adaptive,
                promoted=(
                    name == "markov"
                    and not frozen.any_wrong_collision
                    and not adaptive.any_wrong_collision
                    and adaptive.exact_verified_hits > 0
                ),
            )
        )
    return MarkovCoordinateSelectorEvaluation(
        fit_sha256=fit.sha256,
        holdout_group_sha256s=tuple(group.group_sha256 for group in holdout),
        holdout_authority_sha256=authority,
        results=tuple(results),
    )


__all__ = [
    "MARKOV_FIT_SCHEMA",
    "MARKOV_HOLDOUT_SCHEMA",
    "MARKOV_NOMINATION_SCHEMA",
    "MARKOV_SELECTOR_CONFIG_SCHEMA",
    "MARKOV_SELECTOR_VERIFIER_SHA256",
    "MARKOV_STATE_SCHEMA",
    "CoordinateNominationReceipt",
    "MarkovCoordinateSelectorCapacityError",
    "MarkovCoordinateSelectorConfig",
    "MarkovCoordinateSelectorError",
    "MarkovCoordinateSelectorEvaluation",
    "MarkovCoordinateSelectorFit",
    "MarkovCoordinateSelectorIntegrityError",
    "MarkovCoordinateState",
    "MarkovHoldoutResult",
    "evaluate_markov_coordinate_selector",
    "fit_markov_coordinate_selector",
]
