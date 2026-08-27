"""Verified non-contiguous lag blankets for append-only route demand.

The generic conditional-blanket learner owns the statistical mathematics.  This
module owns the Demand evidence boundary: it reconstructs execution chronology
from sealed scheduler events, keeps explicit episodes intact across temporal
splits, and binds every fit and prediction back to an authenticated scheduler
and operator-graph revision.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from typing import cast

from .compute_graph import ComputeOperatorGraphState
from .conditional_blanket import (
    CategoricalFeatureAtom,
    CategoricalSample,
    ConditionalBlanketConfig,
    ConditionalBlanketFitReceipt,
    ConditionalBlanketIntegrityError,
    ConditionalBlanketValidationReceipt,
    fit_conditional_blanket,
    predict_probabilities,
    validate_conditional_blanket,
)
from .demand_scheduler import (
    MAX_CONTEXT_ORDER,
    DemandOutcomeReceipt,
    OperatorDemandScheduler,
    OutcomeEvent,
    SelectionEvent,
)
from .identity import canonical_json_bytes, require_sha256


DEMAND_LAG_SAMPLE_SCHEMA = "immer-ooe-demand-lag-sample/v1"
DEMAND_LAG_CORPUS_SCHEMA = "immer-ooe-demand-lag-corpus/v1"
DEMAND_LAG_SPLIT_SCHEMA = "immer-ooe-demand-lag-split/v1"
DEMAND_LAG_FIT_SCHEMA = "immer-ooe-demand-lag-blanket-fit/v1"
DEMAND_LAG_VALIDATION_SCHEMA = "immer-ooe-demand-lag-blanket-validation/v1"
DEMAND_LAG_PREDICTION_SCHEMA = "immer-ooe-demand-lag-blanket-prediction/v1"
MAX_DEMAND_LAG_RECEIPT_BYTES = 48 * 1024 * 1024


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


DEMAND_LAG_BOS = _digest(
    {
        "schema": "immer-ooe-demand-lag-bos/v1",
        "meaning": "explicit-before-episode-boundary",
    }
)


class DemandBlanketError(RuntimeError):
    """Base error for verified Demand lag-blanket learning."""


class DemandBlanketIntegrityError(DemandBlanketError):
    """A corpus, split, model, validation, or prediction failed closed."""


class DemandBlanketUnavailableError(DemandBlanketError):
    """The validated blanket has no authorized singleton prediction."""


def _strict_json(data: bytes, *, label: str) -> object:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if len(data) > MAX_DEMAND_LAG_RECEIPT_BYTES:
        raise DemandBlanketIntegrityError(f"{label} exceeds its byte bound")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DemandBlanketIntegrityError(f"{label} is not strict JSON") from exc
    if canonical_json_bytes(value) != data:
        raise DemandBlanketIntegrityError(f"{label} is not canonical JSON")
    return value


def _sealed(schema: str, body: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(body)
    return {"schema": schema, "body": normalized, "body_sha256": _digest(normalized)}


def _sealed_body(value: object, *, schema: str, label: str) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "body", "body_sha256"}
        or value.get("schema") != schema
    ):
        raise DemandBlanketIntegrityError(f"invalid {label} envelope")
    body = value.get("body")
    if not isinstance(body, Mapping):
        raise DemandBlanketIntegrityError(f"invalid {label} body")
    try:
        claimed = require_sha256(value.get("body_sha256"), field="body_sha256")
    except ValueError as exc:
        raise DemandBlanketIntegrityError(f"invalid {label} body hash") from exc
    if claimed != _digest(body):
        raise DemandBlanketIntegrityError(f"{label} body hash mismatch")
    return body


def _uint(value: object, *, field: str, positive: bool = False) -> int:
    lower = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a {qualifier} integer")
    return value


def _hashes(
    values: Sequence[str], *, field: str, sorted_unique: bool = False
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence")
    result = tuple(require_sha256(value, field=field) for value in values)
    if sorted_unique and result != tuple(sorted(set(result))):
        raise ValueError(f"{field} must be sorted and unique")
    return result


def _fraction_dict(value: Fraction) -> dict[str, int]:
    if not isinstance(value, Fraction):
        raise TypeError("exact ratio must be a Fraction")
    if value < 0:
        raise ValueError("exact ratio must be non-negative")
    return {"numerator": value.numerator, "denominator": value.denominator}


def _parse_fraction(value: object, *, field: str) -> Fraction:
    if not isinstance(value, Mapping) or set(value) != {"numerator", "denominator"}:
        raise DemandBlanketIntegrityError(f"invalid {field}")
    try:
        numerator = _uint(value.get("numerator"), field=f"{field}.numerator")
        denominator = _uint(
            value.get("denominator"), field=f"{field}.denominator", positive=True
        )
        result = Fraction(numerator, denominator)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise DemandBlanketIntegrityError(f"invalid {field}") from exc
    if result < 0 or _fraction_dict(result) != dict(value):
        raise DemandBlanketIntegrityError(f"non-canonical {field}")
    return result


@dataclass(frozen=True, slots=True)
class DemandLagSample:
    """One next-route observation with explicit non-contiguous lag columns."""

    logical_time: int
    settlement_logical_time: int
    episode_sha256: str
    source_sha256: str
    position: int
    lag_route_sha256s: tuple[str, ...]
    target_route_sha256: str
    outcome_receipt_sha256: str
    outcome_verifier_sha256: str
    outcome_evidence_sha256: str
    selection_event_sha256: str | None

    def __post_init__(self) -> None:
        logical = _uint(self.logical_time, field="logical_time", positive=True)
        settlement = _uint(
            self.settlement_logical_time,
            field="settlement_logical_time",
            positive=True,
        )
        if logical > settlement:
            raise ValueError("execution cannot follow outcome settlement")
        position = _uint(self.position, field="position")
        lags = _hashes(self.lag_route_sha256s, field="lag_route_sha256s")
        if not lags or len(lags) > MAX_CONTEXT_ORDER:
            raise ValueError("lag inventory is empty or exceeds its bound")
        for field in (
            "episode_sha256",
            "source_sha256",
            "target_route_sha256",
            "outcome_receipt_sha256",
            "outcome_verifier_sha256",
            "outcome_evidence_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        selection = self.selection_event_sha256
        if selection is not None:
            selection = require_sha256(selection, field="selection_event_sha256")
        object.__setattr__(self, "logical_time", logical)
        object.__setattr__(self, "settlement_logical_time", settlement)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "lag_route_sha256s", lags)
        object.__setattr__(self, "selection_event_sha256", selection)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": DEMAND_LAG_SAMPLE_SCHEMA,
            "logical_time": self.logical_time,
            "settlement_logical_time": self.settlement_logical_time,
            "episode_sha256": self.episode_sha256,
            "source_sha256": self.source_sha256,
            "position": self.position,
            "lag_route_sha256s": list(self.lag_route_sha256s),
            "target_route_sha256": self.target_route_sha256,
            "outcome_receipt_sha256": self.outcome_receipt_sha256,
            "outcome_verifier_sha256": self.outcome_verifier_sha256,
            "outcome_evidence_sha256": self.outcome_evidence_sha256,
            "selection_event_sha256": self.selection_event_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "DemandLagSample":
        expected = {
            "schema",
            "logical_time",
            "settlement_logical_time",
            "episode_sha256",
            "source_sha256",
            "position",
            "lag_route_sha256s",
            "target_route_sha256",
            "outcome_receipt_sha256",
            "outcome_verifier_sha256",
            "outcome_evidence_sha256",
            "selection_event_sha256",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != DEMAND_LAG_SAMPLE_SCHEMA
            or not isinstance(value.get("lag_route_sha256s"), list)
        ):
            raise DemandBlanketIntegrityError("invalid Demand lag sample")
        try:
            sample = cls(
                logical_time=cast(int, value.get("logical_time")),
                settlement_logical_time=cast(
                    int, value.get("settlement_logical_time")
                ),
                episode_sha256=cast(str, value.get("episode_sha256")),
                source_sha256=cast(str, value.get("source_sha256")),
                position=cast(int, value.get("position")),
                lag_route_sha256s=tuple(
                    cast(list[str], value.get("lag_route_sha256s"))
                ),
                target_route_sha256=cast(
                    str, value.get("target_route_sha256")
                ),
                outcome_receipt_sha256=cast(
                    str, value.get("outcome_receipt_sha256")
                ),
                outcome_verifier_sha256=cast(
                    str, value.get("outcome_verifier_sha256")
                ),
                outcome_evidence_sha256=cast(
                    str, value.get("outcome_evidence_sha256")
                ),
                selection_event_sha256=cast(
                    str | None, value.get("selection_event_sha256")
                ),
            )
        except (TypeError, ValueError) as exc:
            raise DemandBlanketIntegrityError(
                "Demand lag sample failed validation"
            ) from exc
        if sample.to_dict() != dict(value):
            raise DemandBlanketIntegrityError(
                "Demand lag sample failed canonical reconstruction"
            )
        return sample


@dataclass(frozen=True, slots=True)
class DemandLagCorpusReceipt:
    scheduler_state_sha256: str
    scheduler_config_sha256: str
    graph_generation: int
    graph_state_sha256: str
    input_abi_sha256: str
    output_abi_sha256: str | None
    max_context_order: int
    bos_category: str
    route_alphabet_sha256s: tuple[str, ...]
    samples: tuple[DemandLagSample, ...]
    sample_sha256s: tuple[str, ...]
    outcome_receipt_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "scheduler_state_sha256",
            "scheduler_config_sha256",
            "graph_state_sha256",
            "input_abi_sha256",
            "bos_category",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        if self.output_abi_sha256 is not None:
            object.__setattr__(
                self,
                "output_abi_sha256",
                require_sha256(self.output_abi_sha256, field="output_abi_sha256"),
            )
        generation = _uint(self.graph_generation, field="graph_generation")
        order = _uint(
            self.max_context_order, field="max_context_order", positive=True
        )
        if order > MAX_CONTEXT_ORDER:
            raise ValueError("max_context_order exceeds the Demand bound")
        alphabet = _hashes(
            self.route_alphabet_sha256s,
            field="route_alphabet_sha256s",
            sorted_unique=True,
        )
        if not alphabet or self.bos_category in alphabet:
            raise ValueError("route alphabet is empty or collides with BOS")
        samples = tuple(self.samples)
        if not samples or any(not isinstance(row, DemandLagSample) for row in samples):
            raise ValueError("corpus samples must be non-empty DemandLagSample values")
        expected_order = tuple(
            sorted(
                samples,
                key=lambda row: (
                    row.logical_time,
                    row.settlement_logical_time,
                    row.episode_sha256,
                    row.position,
                    row.outcome_receipt_sha256,
                ),
            )
        )
        if samples != expected_order:
            raise ValueError("Demand lag samples must be execution-chronological")
        sample_hashes = tuple(row.sha256 for row in samples)
        outcome_hashes = tuple(row.outcome_receipt_sha256 for row in samples)
        if tuple(self.sample_sha256s) != sample_hashes:
            raise ValueError("sample hash inventory differs from corpus samples")
        if tuple(self.outcome_receipt_sha256s) != outcome_hashes:
            raise ValueError("outcome hash inventory differs from corpus samples")
        if len(set(sample_hashes)) != len(samples) or len(set(outcome_hashes)) != len(samples):
            raise ValueError("Demand lag corpus contains duplicate evidence")
        by_episode: dict[str, list[DemandLagSample]] = defaultdict(list)
        allowed = set(alphabet) | {self.bos_category}
        for row in samples:
            if len(row.lag_route_sha256s) != order:
                raise ValueError("sample lag width differs from corpus configuration")
            if row.target_route_sha256 not in alphabet:
                raise ValueError("sample target is outside the route alphabet")
            if any(value not in allowed for value in row.lag_route_sha256s):
                raise ValueError("sample lag category is outside the sealed alphabet")
            if row.source_sha256 != row.episode_sha256:
                raise ValueError("Demand episode is its indivisible source group")
            by_episode[row.episode_sha256].append(row)
        for rows in by_episode.values():
            ordered = sorted(rows, key=lambda row: row.position)
            if [row.position for row in ordered] != list(range(len(ordered))):
                raise ValueError("Demand episode positions are not contiguous")
            logical_times = [row.logical_time for row in ordered]
            if logical_times != sorted(set(logical_times)):
                raise ValueError(
                    "Demand episode positions differ from execution chronology"
                )
            route_prefix: list[str] = []
            for row in ordered:
                expected_lags = tuple(
                    route_prefix[-lag] if len(route_prefix) >= lag else self.bos_category
                    for lag in range(1, order + 1)
                )
                if row.lag_route_sha256s != expected_lags:
                    raise ValueError("Demand lag sample differs from episode chronology")
                route_prefix.append(row.target_route_sha256)
        object.__setattr__(self, "graph_generation", generation)
        object.__setattr__(self, "max_context_order", order)
        object.__setattr__(self, "route_alphabet_sha256s", alphabet)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "sample_sha256s", sample_hashes)
        object.__setattr__(self, "outcome_receipt_sha256s", outcome_hashes)

    def to_dict(self) -> dict[str, object]:
        return _sealed(
            DEMAND_LAG_CORPUS_SCHEMA,
            {
                "scheduler_state_sha256": self.scheduler_state_sha256,
                "scheduler_config_sha256": self.scheduler_config_sha256,
                "graph_generation": self.graph_generation,
                "graph_state_sha256": self.graph_state_sha256,
                "input_abi_sha256": self.input_abi_sha256,
                "output_abi_sha256": self.output_abi_sha256,
                "max_context_order": self.max_context_order,
                "bos_category": self.bos_category,
                "route_alphabet_sha256s": list(self.route_alphabet_sha256s),
                "samples": [row.to_dict() for row in self.samples],
                "sample_sha256s": list(self.sample_sha256s),
                "outcome_receipt_sha256s": list(
                    self.outcome_receipt_sha256s
                ),
            },
        )

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_DEMAND_LAG_RECEIPT_BYTES:
            raise ValueError("Demand lag corpus exceeds its byte bound")
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "DemandLagCorpusReceipt":
        value = _strict_json(data, label="Demand lag corpus")
        body = _sealed_body(
            value, schema=DEMAND_LAG_CORPUS_SCHEMA, label="Demand lag corpus"
        )
        expected = {
            "scheduler_state_sha256",
            "scheduler_config_sha256",
            "graph_generation",
            "graph_state_sha256",
            "input_abi_sha256",
            "output_abi_sha256",
            "max_context_order",
            "bos_category",
            "route_alphabet_sha256s",
            "samples",
            "sample_sha256s",
            "outcome_receipt_sha256s",
        }
        if set(body) != expected:
            raise DemandBlanketIntegrityError("invalid Demand lag corpus body")
        for field in (
            "route_alphabet_sha256s",
            "samples",
            "sample_sha256s",
            "outcome_receipt_sha256s",
        ):
            if not isinstance(body.get(field), list):
                raise DemandBlanketIntegrityError(
                    f"invalid Demand lag corpus {field}"
                )
        try:
            result = cls(
                scheduler_state_sha256=cast(
                    str, body.get("scheduler_state_sha256")
                ),
                scheduler_config_sha256=cast(
                    str, body.get("scheduler_config_sha256")
                ),
                graph_generation=cast(int, body.get("graph_generation")),
                graph_state_sha256=cast(str, body.get("graph_state_sha256")),
                input_abi_sha256=cast(str, body.get("input_abi_sha256")),
                output_abi_sha256=cast(
                    str | None, body.get("output_abi_sha256")
                ),
                max_context_order=cast(int, body.get("max_context_order")),
                bos_category=cast(str, body.get("bos_category")),
                route_alphabet_sha256s=tuple(
                    cast(list[str], body.get("route_alphabet_sha256s"))
                ),
                samples=tuple(
                    DemandLagSample.from_dict(row)
                    for row in cast(list[object], body.get("samples"))
                ),
                sample_sha256s=tuple(
                    cast(list[str], body.get("sample_sha256s"))
                ),
                outcome_receipt_sha256s=tuple(
                    cast(list[str], body.get("outcome_receipt_sha256s"))
                ),
            )
        except (TypeError, ValueError) as exc:
            raise DemandBlanketIntegrityError(
                "Demand lag corpus failed validation"
            ) from exc
        if result.to_bytes() != data:
            raise DemandBlanketIntegrityError(
                "Demand lag corpus changed during reconstruction"
            )
        return result


def build_demand_lag_corpus(
    scheduler: OperatorDemandScheduler,
    graph_state: ComputeOperatorGraphState,
    *,
    input_abi_sha256: str,
    output_abi_sha256: str | None = None,
    max_context_order: int | None = None,
) -> DemandLagCorpusReceipt:
    """Derive lag rows only from successful, explicit, exact-revision episodes."""

    if not isinstance(scheduler, OperatorDemandScheduler):
        raise TypeError("scheduler must be an OperatorDemandScheduler")
    if not isinstance(graph_state, ComputeOperatorGraphState):
        raise TypeError("graph_state must be a ComputeOperatorGraphState")
    input_abi = require_sha256(input_abi_sha256, field="input_abi_sha256")
    output_abi = (
        None
        if output_abi_sha256 is None
        else require_sha256(output_abi_sha256, field="output_abi_sha256")
    )
    order = scheduler.config.max_context_order if max_context_order is None else _uint(
        max_context_order, field="max_context_order", positive=True
    )
    if order > scheduler.config.max_context_order or order > MAX_CONTEXT_ORDER:
        raise ValueError("max_context_order exceeds the scheduler configuration")
    state = scheduler.state()
    if state.config_sha256 != scheduler.config.sha256:
        raise DemandBlanketIntegrityError("scheduler state configuration changed")
    alphabet = tuple(
        sorted(
            route.sha256
            for route in graph_state.materialized_routes
            if route.input_abi_sha256 == input_abi
            and (output_abi is None or route.output_abi_sha256 == output_abi)
        )
    )
    if not alphabet:
        raise DemandBlanketUnavailableError(
            "graph has no materialized route for the requested ABI"
        )
    allowed = set(alphabet)
    selection_time = {
        event.sha256: event.logical_time
        for event in state.events
        if isinstance(event, SelectionEvent)
    }
    grouped: dict[
        str, list[tuple[int, int, str, DemandOutcomeReceipt]]
    ] = defaultdict(list)
    for event in state.events:
        if not isinstance(event, OutcomeEvent):
            continue
        outcome = event.outcome
        if (
            not outcome.success
            or outcome.episode_sha256 is None
            or outcome.graph_generation != graph_state.generation
            or outcome.graph_state_sha256 != graph_state.sha256
            or outcome.input_abi_sha256 != input_abi
            or (output_abi is not None and outcome.output_abi_sha256 != output_abi)
            or outcome.route_sha256 not in allowed
        ):
            continue
        execution_time = (
            event.logical_time
            if outcome.selection_event_sha256 is None
            else selection_time[outcome.selection_event_sha256]
        )
        grouped[outcome.episode_sha256].append(
            (execution_time, event.logical_time, outcome.sha256, outcome)
        )
    samples: list[DemandLagSample] = []
    episodes = sorted(
        grouped.items(),
        key=lambda item: (
            min(row[0] for row in item[1]),
            item[0],
        ),
    )
    for episode, raw_rows in episodes:
        route_prefix: list[str] = []
        for position, row in enumerate(sorted(raw_rows)):
            execution_time, settlement_time, _outcome_sha, outcome = row
            lags = tuple(
                route_prefix[-lag] if len(route_prefix) >= lag else DEMAND_LAG_BOS
                for lag in range(1, order + 1)
            )
            samples.append(
                DemandLagSample(
                    logical_time=execution_time,
                    settlement_logical_time=settlement_time,
                    episode_sha256=episode,
                    source_sha256=episode,
                    position=position,
                    lag_route_sha256s=lags,
                    target_route_sha256=outcome.route_sha256,
                    outcome_receipt_sha256=outcome.sha256,
                    outcome_verifier_sha256=outcome.outcome_verifier_sha256,
                    outcome_evidence_sha256=outcome.outcome_evidence_sha256,
                    selection_event_sha256=outcome.selection_event_sha256,
                )
            )
            route_prefix.append(outcome.route_sha256)
    if not samples:
        raise DemandBlanketUnavailableError(
            "no successful explicit episode matches the exact graph and ABI"
        )
    samples.sort(
        key=lambda row: (
            row.logical_time,
            row.settlement_logical_time,
            row.episode_sha256,
            row.position,
            row.outcome_receipt_sha256,
        )
    )
    return DemandLagCorpusReceipt(
        scheduler_state_sha256=state.sha256,
        scheduler_config_sha256=state.config_sha256,
        graph_generation=graph_state.generation,
        graph_state_sha256=graph_state.sha256,
        input_abi_sha256=input_abi,
        output_abi_sha256=output_abi,
        max_context_order=order,
        bos_category=DEMAND_LAG_BOS,
        route_alphabet_sha256s=alphabet,
        samples=tuple(samples),
        sample_sha256s=tuple(row.sha256 for row in samples),
        outcome_receipt_sha256s=tuple(
            row.outcome_receipt_sha256 for row in samples
        ),
    )


@dataclass(frozen=True, slots=True)
class DemandEpisodeSplitReceipt:
    corpus_sha256: str
    train_episode_sha256s: tuple[str, ...]
    calibration_episode_sha256s: tuple[str, ...]
    holdout_episode_sha256s: tuple[str, ...]
    train_sample_sha256s: tuple[str, ...]
    calibration_sample_sha256s: tuple[str, ...]
    holdout_sample_sha256s: tuple[str, ...]
    train_outcome_sha256s: tuple[str, ...]
    calibration_outcome_sha256s: tuple[str, ...]
    holdout_outcome_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "corpus_sha256",
            require_sha256(self.corpus_sha256, field="corpus_sha256"),
        )
        episode_sets: list[set[str]] = []
        for field in (
            "train_episode_sha256s",
            "calibration_episode_sha256s",
            "holdout_episode_sha256s",
        ):
            values = _hashes(getattr(self, field), field=field)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{field} must be non-empty and unique")
            object.__setattr__(self, field, values)
            episode_sets.append(set(values))
        if any(
            left & right
            for index, left in enumerate(episode_sets)
            for right in episode_sets[index + 1 :]
        ):
            raise ValueError("episode groups leak across temporal partitions")
        inventories: list[set[str]] = []
        for field in (
            "train_sample_sha256s",
            "calibration_sample_sha256s",
            "holdout_sample_sha256s",
            "train_outcome_sha256s",
            "calibration_outcome_sha256s",
            "holdout_outcome_sha256s",
        ):
            values = _hashes(getattr(self, field), field=field)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{field} must be non-empty and unique")
            object.__setattr__(self, field, values)
            inventories.append(set(values))
        sample_sets = inventories[:3]
        outcome_sets = inventories[3:]
        if any(
            left & right
            for collection in (sample_sets, outcome_sets)
            for index, left in enumerate(collection)
            for right in collection[index + 1 :]
        ):
            raise ValueError("sample or outcome evidence leaks across partitions")

    def to_dict(self) -> dict[str, object]:
        return _sealed(
            DEMAND_LAG_SPLIT_SCHEMA,
            {
                "corpus_sha256": self.corpus_sha256,
                "train_episode_sha256s": list(self.train_episode_sha256s),
                "calibration_episode_sha256s": list(
                    self.calibration_episode_sha256s
                ),
                "holdout_episode_sha256s": list(self.holdout_episode_sha256s),
                "train_sample_sha256s": list(self.train_sample_sha256s),
                "calibration_sample_sha256s": list(
                    self.calibration_sample_sha256s
                ),
                "holdout_sample_sha256s": list(self.holdout_sample_sha256s),
                "train_outcome_sha256s": list(self.train_outcome_sha256s),
                "calibration_outcome_sha256s": list(
                    self.calibration_outcome_sha256s
                ),
                "holdout_outcome_sha256s": list(self.holdout_outcome_sha256s),
            },
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "DemandEpisodeSplitReceipt":
        value = _strict_json(data, label="Demand episode split")
        body = _sealed_body(
            value, schema=DEMAND_LAG_SPLIT_SCHEMA, label="Demand episode split"
        )
        expected = {
            "corpus_sha256",
            "train_episode_sha256s",
            "calibration_episode_sha256s",
            "holdout_episode_sha256s",
            "train_sample_sha256s",
            "calibration_sample_sha256s",
            "holdout_sample_sha256s",
            "train_outcome_sha256s",
            "calibration_outcome_sha256s",
            "holdout_outcome_sha256s",
        }
        if set(body) != expected or any(
            not isinstance(body.get(field), list)
            for field in expected - {"corpus_sha256"}
        ):
            raise DemandBlanketIntegrityError("invalid Demand episode split body")
        try:
            result = cls(
                corpus_sha256=cast(str, body.get("corpus_sha256")),
                train_episode_sha256s=tuple(
                    cast(list[str], body.get("train_episode_sha256s"))
                ),
                calibration_episode_sha256s=tuple(
                    cast(list[str], body.get("calibration_episode_sha256s"))
                ),
                holdout_episode_sha256s=tuple(
                    cast(list[str], body.get("holdout_episode_sha256s"))
                ),
                train_sample_sha256s=tuple(
                    cast(list[str], body.get("train_sample_sha256s"))
                ),
                calibration_sample_sha256s=tuple(
                    cast(list[str], body.get("calibration_sample_sha256s"))
                ),
                holdout_sample_sha256s=tuple(
                    cast(list[str], body.get("holdout_sample_sha256s"))
                ),
                train_outcome_sha256s=tuple(
                    cast(list[str], body.get("train_outcome_sha256s"))
                ),
                calibration_outcome_sha256s=tuple(
                    cast(list[str], body.get("calibration_outcome_sha256s"))
                ),
                holdout_outcome_sha256s=tuple(
                    cast(list[str], body.get("holdout_outcome_sha256s"))
                ),
            )
        except (TypeError, ValueError) as exc:
            raise DemandBlanketIntegrityError(
                "Demand episode split failed validation"
            ) from exc
        if result.to_bytes() != data:
            raise DemandBlanketIntegrityError(
                "Demand episode split changed during reconstruction"
            )
        return result


def chronological_episode_group_split(
    corpus: DemandLagCorpusReceipt,
    *,
    train_fraction: Fraction = Fraction(3, 5),
    calibration_fraction: Fraction = Fraction(1, 5),
) -> DemandEpisodeSplitReceipt:
    """Split whole episode/source groups in their first-execution order."""

    if not isinstance(corpus, DemandLagCorpusReceipt):
        raise TypeError("corpus must be a DemandLagCorpusReceipt")
    for name, value in (
        ("train_fraction", train_fraction),
        ("calibration_fraction", calibration_fraction),
    ):
        if not isinstance(value, Fraction) or value <= 0 or value >= 1:
            raise ValueError(f"{name} must be an exact Fraction in (0, 1)")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train plus calibration fraction must leave a holdout")
    episode_rows: dict[str, list[DemandLagSample]] = defaultdict(list)
    for row in corpus.samples:
        episode_rows[row.episode_sha256].append(row)
    intervals = sorted(
        (
            min(row.logical_time for row in rows),
            max(row.logical_time for row in rows),
            episode,
        )
        for episode, rows in episode_rows.items()
    )
    # Concurrent/interleaved episodes form one indivisible temporal block.
    # Splitting them by first appearance would put a later training row after
    # an earlier calibration row and leak time despite disjoint group hashes.
    blocks: list[list[tuple[int, int, str]]] = []
    block_max = -1
    for interval in intervals:
        if not blocks or interval[0] > block_max:
            blocks.append([interval])
            block_max = interval[1]
        else:
            blocks[-1].append(interval)
            block_max = max(block_max, interval[1])
    count = len(blocks)
    if count < 3:
        raise DemandBlanketUnavailableError(
            "at least three non-overlapping episode blocks are required for "
            "train/calibration/holdout"
        )
    train_count = max(1, min(count - 2, (count * train_fraction.numerator) // train_fraction.denominator))
    calibration_count = max(
        1,
        min(
            count - train_count - 1,
            (count * calibration_fraction.numerator)
            // calibration_fraction.denominator,
        ),
    )
    train = tuple(
        interval[2]
        for block in blocks[:train_count]
        for interval in block
    )
    calibration = tuple(
        interval[2]
        for block in blocks[train_count : train_count + calibration_count]
        for interval in block
    )
    holdout = tuple(
        interval[2]
        for block in blocks[train_count + calibration_count :]
        for interval in block
    )
    partitions = (set(train), set(calibration), set(holdout))

    def inventory(
        groups: set[str], *, outcome: bool
    ) -> tuple[str, ...]:
        return tuple(
            row.outcome_receipt_sha256 if outcome else row.sha256
            for row in corpus.samples
            if row.episode_sha256 in groups
        )

    result = DemandEpisodeSplitReceipt(
        corpus_sha256=corpus.sha256,
        train_episode_sha256s=train,
        calibration_episode_sha256s=calibration,
        holdout_episode_sha256s=holdout,
        train_sample_sha256s=inventory(partitions[0], outcome=False),
        calibration_sample_sha256s=inventory(partitions[1], outcome=False),
        holdout_sample_sha256s=inventory(partitions[2], outcome=False),
        train_outcome_sha256s=inventory(partitions[0], outcome=True),
        calibration_outcome_sha256s=inventory(partitions[1], outcome=True),
        holdout_outcome_sha256s=inventory(partitions[2], outcome=True),
    )
    # In Demand, explicit episode is also the source identity.  Keep this
    # assertion here so that a future source field cannot silently leak.
    source_partition: dict[str, int] = {}
    sample_partition = {
        sample_sha: index
        for index, values in enumerate(
            (
                result.train_sample_sha256s,
                result.calibration_sample_sha256s,
                result.holdout_sample_sha256s,
            )
        )
        for sample_sha in values
    }
    for row in corpus.samples:
        partition = sample_partition[row.sha256]
        prior = source_partition.setdefault(row.source_sha256, partition)
        if prior != partition:
            raise DemandBlanketIntegrityError(
                "source identity leaks across temporal partitions"
            )
    temporal_partitions = tuple(
        tuple(
            row.logical_time
            for row in corpus.samples
            if sample_partition[row.sha256] == index
        )
        for index in range(3)
    )
    if not (
        max(temporal_partitions[0]) < min(temporal_partitions[1])
        and max(temporal_partitions[1]) < min(temporal_partitions[2])
    ):
        raise DemandBlanketIntegrityError(
            "episode-group split is not strictly chronological"
        )
    return result


# Short alias used by runtime callers.
chronological_demand_split = chronological_episode_group_split


def _feature_name(lag: int) -> str:
    value = _uint(lag, field="lag", positive=True)
    if value > MAX_CONTEXT_ORDER:
        raise ValueError("lag exceeds the Demand bound")
    return f"lag-{value:04d}"


def _lag_from_feature(name: str) -> int:
    if (
        not isinstance(name, str)
        or len(name) != 8
        or not name.startswith("lag-")
        or not name[4:].isdigit()
    ):
        raise DemandBlanketIntegrityError("generic fit selected a non-lag feature")
    lag = int(name[4:])
    if not 1 <= lag <= MAX_CONTEXT_ORDER or _feature_name(lag) != name:
        raise DemandBlanketIntegrityError("generic fit selected an invalid lag")
    return lag


def _categorical_sample(
    corpus: DemandLagCorpusReceipt,
    sample: DemandLagSample,
) -> CategoricalSample:
    return CategoricalSample(
        temporal_index=sample.logical_time,
        group_sha256=sample.episode_sha256,
        source_receipt_sha256=sample.outcome_receipt_sha256,
        source_revision_sha256=corpus.graph_state_sha256,
        verifier_sha256=sample.outcome_verifier_sha256,
        evidence_sha256=sample.outcome_evidence_sha256,
        target_category=sample.target_route_sha256,
        features=tuple(
            CategoricalFeatureAtom(_feature_name(index), category)
            for index, category in enumerate(
                sample.lag_route_sha256s, start=1
            )
        ),
    )


def _partition_samples(
    corpus: DemandLagCorpusReceipt,
    split: DemandEpisodeSplitReceipt,
) -> tuple[
    tuple[CategoricalSample, ...],
    tuple[CategoricalSample, ...],
    tuple[CategoricalSample, ...],
]:
    if split.corpus_sha256 != corpus.sha256:
        raise DemandBlanketIntegrityError("split names another Demand corpus")
    by_sha = {row.sha256: row for row in corpus.samples}
    partitions: list[tuple[CategoricalSample, ...]] = []
    expected_episodes = (
        split.train_episode_sha256s,
        split.calibration_episode_sha256s,
        split.holdout_episode_sha256s,
    )
    expected_samples = (
        split.train_sample_sha256s,
        split.calibration_sample_sha256s,
        split.holdout_sample_sha256s,
    )
    expected_outcomes = (
        split.train_outcome_sha256s,
        split.calibration_outcome_sha256s,
        split.holdout_outcome_sha256s,
    )
    for episode_inventory, sample_inventory, outcome_inventory in zip(
        expected_episodes,
        expected_samples,
        expected_outcomes,
        strict=True,
    ):
        try:
            rows = tuple(by_sha[address] for address in sample_inventory)
        except KeyError as exc:
            raise DemandBlanketIntegrityError(
                "split sample is absent from the Demand corpus"
            ) from exc
        if (
            tuple(row.outcome_receipt_sha256 for row in rows)
            != outcome_inventory
            or tuple(dict.fromkeys(row.episode_sha256 for row in rows))
            != episode_inventory
        ):
            raise DemandBlanketIntegrityError(
                "split episode or outcome inventory changed"
            )
        partitions.append(tuple(_categorical_sample(corpus, row) for row in rows))
    if set().union(*(set(values) for values in expected_samples)) != set(
        corpus.sample_sha256s
    ):
        raise DemandBlanketIntegrityError("split does not exactly cover the corpus")
    return cast(
        tuple[
            tuple[CategoricalSample, ...],
            tuple[CategoricalSample, ...],
            tuple[CategoricalSample, ...],
        ],
        tuple(partitions),
    )


@dataclass(frozen=True, slots=True)
class DemandLagBlanketFitReceipt:
    """Demand pins wrapped around one replayable generic exact fit."""

    corpus: DemandLagCorpusReceipt
    split: DemandEpisodeSplitReceipt
    generic_fit: ConditionalBlanketFitReceipt
    selected_lags: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.corpus, DemandLagCorpusReceipt):
            raise TypeError("corpus must be a DemandLagCorpusReceipt")
        if not isinstance(self.split, DemandEpisodeSplitReceipt):
            raise TypeError("split must be a DemandEpisodeSplitReceipt")
        if not isinstance(self.generic_fit, ConditionalBlanketFitReceipt):
            raise TypeError("generic_fit must be a ConditionalBlanketFitReceipt")
        train, calibration, _holdout = _partition_samples(self.corpus, self.split)
        expected_lags = tuple(
            _lag_from_feature(name)
            for name in self.generic_fit.selected_feature_names
        )
        lags = tuple(self.selected_lags)
        if lags != tuple(sorted(set(lags))) or lags != expected_lags:
            raise DemandBlanketIntegrityError(
                "selected lag inventory differs from the generic fit"
            )
        if (
            self.generic_fit.config.target_alphabet
            != self.corpus.route_alphabet_sha256s
            or self.generic_fit.feature_schema
            != tuple(
                _feature_name(lag)
                for lag in range(1, self.corpus.max_context_order + 1)
            )
            or self.generic_fit.train_samples != train
            or self.generic_fit.calibration_samples != calibration
        ):
            raise DemandBlanketIntegrityError(
                "generic fit differs from the pinned Demand evidence"
            )
        object.__setattr__(self, "selected_lags", lags)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @property
    def scheduler_state_sha256(self) -> str:
        return self.corpus.scheduler_state_sha256

    @property
    def graph_state_sha256(self) -> str:
        return self.corpus.graph_state_sha256

    def to_dict(self) -> dict[str, object]:
        return _sealed(
            DEMAND_LAG_FIT_SCHEMA,
            {
                "corpus": self.corpus.to_dict(),
                "corpus_sha256": self.corpus.sha256,
                "split": self.split.to_dict(),
                "split_sha256": self.split.sha256,
                "generic_fit": self.generic_fit.to_document(),
                "generic_fit_sha256": self.generic_fit.sha256,
                "selected_lags": list(self.selected_lags),
                "scheduler_state_sha256": self.corpus.scheduler_state_sha256,
                "scheduler_config_sha256": self.corpus.scheduler_config_sha256,
                "graph_generation": self.corpus.graph_generation,
                "graph_state_sha256": self.corpus.graph_state_sha256,
                "input_abi_sha256": self.corpus.input_abi_sha256,
                "output_abi_sha256": self.corpus.output_abi_sha256,
                "route_alphabet_sha256s": list(
                    self.corpus.route_alphabet_sha256s
                ),
                "train_sample_sha256s": list(self.split.train_sample_sha256s),
                "calibration_sample_sha256s": list(
                    self.split.calibration_sample_sha256s
                ),
                "train_outcome_sha256s": list(self.split.train_outcome_sha256s),
                "calibration_outcome_sha256s": list(
                    self.split.calibration_outcome_sha256s
                ),
            },
        )

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_DEMAND_LAG_RECEIPT_BYTES:
            raise ValueError("Demand lag fit exceeds its byte bound")
        return data

    def verify_or_raise(self) -> bool:
        self.generic_fit.verify_or_raise()
        expected = fit_demand_lag_blanket(
            self.corpus,
            self.split,
            config=self.generic_fit.config,
        )
        if expected.to_dict() != self.to_dict():
            raise DemandBlanketIntegrityError(
                "Demand lag fit does not recompute exactly"
            )
        return True

    @classmethod
    def from_bytes(cls, data: bytes) -> "DemandLagBlanketFitReceipt":
        value = _strict_json(data, label="Demand lag fit")
        body = _sealed_body(value, schema=DEMAND_LAG_FIT_SCHEMA, label="Demand lag fit")
        expected = {
            "corpus",
            "corpus_sha256",
            "split",
            "split_sha256",
            "generic_fit",
            "generic_fit_sha256",
            "selected_lags",
            "scheduler_state_sha256",
            "scheduler_config_sha256",
            "graph_generation",
            "graph_state_sha256",
            "input_abi_sha256",
            "output_abi_sha256",
            "route_alphabet_sha256s",
            "train_sample_sha256s",
            "calibration_sample_sha256s",
            "train_outcome_sha256s",
            "calibration_outcome_sha256s",
        }
        if set(body) != expected or not isinstance(body.get("selected_lags"), list):
            raise DemandBlanketIntegrityError("invalid Demand lag fit body")
        try:
            corpus = DemandLagCorpusReceipt.from_bytes(
                canonical_json_bytes(body.get("corpus"))
            )
            split = DemandEpisodeSplitReceipt.from_bytes(
                canonical_json_bytes(body.get("split"))
            )
            generic = ConditionalBlanketFitReceipt.from_document(
                cast(Mapping[str, object], body.get("generic_fit"))
            )
            result = cls(
                corpus=corpus,
                split=split,
                generic_fit=generic,
                selected_lags=tuple(cast(list[int], body.get("selected_lags"))),
            )
            pins = {
                "corpus_sha256": corpus.sha256,
                "split_sha256": split.sha256,
                "generic_fit_sha256": generic.sha256,
                "scheduler_state_sha256": corpus.scheduler_state_sha256,
                "scheduler_config_sha256": corpus.scheduler_config_sha256,
                "graph_generation": corpus.graph_generation,
                "graph_state_sha256": corpus.graph_state_sha256,
                "input_abi_sha256": corpus.input_abi_sha256,
                "output_abi_sha256": corpus.output_abi_sha256,
                "route_alphabet_sha256s": list(corpus.route_alphabet_sha256s),
                "train_sample_sha256s": list(split.train_sample_sha256s),
                "calibration_sample_sha256s": list(
                    split.calibration_sample_sha256s
                ),
                "train_outcome_sha256s": list(split.train_outcome_sha256s),
                "calibration_outcome_sha256s": list(
                    split.calibration_outcome_sha256s
                ),
            }
            if any(body.get(field) != actual for field, actual in pins.items()):
                raise DemandBlanketIntegrityError("Demand lag fit pin changed")
        except DemandBlanketIntegrityError:
            raise
        except (ConditionalBlanketIntegrityError, TypeError, ValueError) as exc:
            raise DemandBlanketIntegrityError(
                "Demand lag fit failed validation"
            ) from exc
        if result.to_bytes() != data:
            raise DemandBlanketIntegrityError(
                "Demand lag fit changed during reconstruction"
            )
        result.verify_or_raise()
        return result


def fit_demand_lag_blanket(
    corpus: DemandLagCorpusReceipt,
    split: DemandEpisodeSplitReceipt,
    *,
    config: ConditionalBlanketConfig | None = None,
) -> DemandLagBlanketFitReceipt:
    """Fit an exact generic blanket under Demand-specific evidence pins."""

    if not isinstance(corpus, DemandLagCorpusReceipt):
        raise TypeError("corpus must be a DemandLagCorpusReceipt")
    if not isinstance(split, DemandEpisodeSplitReceipt):
        raise TypeError("split must be a DemandEpisodeSplitReceipt")
    train, calibration, _holdout = _partition_samples(corpus, split)
    if config is None:
        config = ConditionalBlanketConfig(
            target_alphabet=corpus.route_alphabet_sha256s,
            max_subset_size=corpus.max_context_order,
            max_exhaustive_subsets=min(
                1 << corpus.max_context_order,
                1_000_000,
            ),
            max_pair_checks=256,
        )
    if not isinstance(config, ConditionalBlanketConfig):
        raise TypeError("config must be a ConditionalBlanketConfig")
    if config.target_alphabet != corpus.route_alphabet_sha256s:
        raise DemandBlanketIntegrityError(
            "generic target alphabet differs from the Demand route alphabet"
        )
    try:
        generic = fit_conditional_blanket(train, calibration, config=config)
    except (ConditionalBlanketIntegrityError, ValueError) as exc:
        raise DemandBlanketIntegrityError(
            "generic conditional blanket rejected Demand evidence"
        ) from exc
    selected_lags = tuple(
        _lag_from_feature(name) for name in generic.selected_feature_names
    )
    return DemandLagBlanketFitReceipt(
        corpus=corpus,
        split=split,
        generic_fit=generic,
        selected_lags=selected_lags,
    )


def _validation_verdict(
    fit: DemandLagBlanketFitReceipt,
    generic: ConditionalBlanketValidationReceipt,
    minimum_accuracy: Fraction,
) -> tuple[bool, str]:
    selected = generic.selected_metrics
    full = generic.full_metrics
    random = generic.random_metrics
    marginal = generic.marginal_metrics
    config = fit.generic_fit.config
    if fit.generic_fit.capacity_exhausted or not fit.generic_fit.closure_verified:
        return False, "blanket-capacity-or-closure-unverified"
    if selected.accuracy < minimum_accuracy:
        return False, "holdout-accuracy-below-threshold"
    if (
        selected.accuracy < random.accuracy
        or selected.accuracy < marginal.accuracy
        or selected.brier > random.brier
        or selected.brier > marginal.brier
    ):
        return False, "holdout-placebo-or-marginal-not-beaten"
    if (
        selected.accuracy + config.max_accuracy_regret < full.accuracy
        or selected.brier > full.brier + config.max_brier_regret
        or selected.coverage + config.max_coverage_regret < full.coverage
    ):
        return False, "holdout-full-arm-regret-exceeded"
    return True, "validated-exact-holdout"


@dataclass(frozen=True, slots=True)
class DemandLagBlanketValidationReceipt:
    fit: DemandLagBlanketFitReceipt
    generic_validation: ConditionalBlanketValidationReceipt
    minimum_holdout_accuracy: Fraction
    accepted: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.fit, DemandLagBlanketFitReceipt):
            raise TypeError("fit must be a DemandLagBlanketFitReceipt")
        if not isinstance(
            self.generic_validation, ConditionalBlanketValidationReceipt
        ):
            raise TypeError(
                "generic_validation must be a ConditionalBlanketValidationReceipt"
            )
        if isinstance(self.minimum_holdout_accuracy, bool) or not isinstance(
            self.minimum_holdout_accuracy, (int, Fraction)
        ):
            raise TypeError(
                "minimum_holdout_accuracy must be exact int/Fraction data"
            )
        minimum = Fraction(self.minimum_holdout_accuracy)
        if not 0 <= minimum <= 1:
            raise ValueError("minimum_holdout_accuracy must lie in [0, 1]")
        if self.generic_validation.fit.sha256 != self.fit.generic_fit.sha256:
            raise DemandBlanketIntegrityError(
                "generic validation names another Demand fit"
            )
        _train, _calibration, holdout = _partition_samples(
            self.fit.corpus, self.fit.split
        )
        if self.generic_validation.holdout_samples != holdout:
            raise DemandBlanketIntegrityError(
                "generic validation differs from Demand holdout evidence"
            )
        expected_accepted, expected_reason = _validation_verdict(
            self.fit, self.generic_validation, minimum
        )
        if (
            not isinstance(self.accepted, bool)
            or self.accepted != expected_accepted
            or self.reason != expected_reason
        ):
            raise DemandBlanketIntegrityError(
                "Demand validation verdict is not reproducible"
            )
        object.__setattr__(self, "minimum_holdout_accuracy", minimum)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return _sealed(
            DEMAND_LAG_VALIDATION_SCHEMA,
            {
                "fit": self.fit.to_dict(),
                "fit_sha256": self.fit.sha256,
                "generic_validation": self.generic_validation.to_document(),
                "generic_validation_sha256": self.generic_validation.sha256,
                "minimum_holdout_accuracy": _fraction_dict(
                    self.minimum_holdout_accuracy
                ),
                "accepted": self.accepted,
                "reason": self.reason,
                "selected_lags": list(self.fit.selected_lags),
                "holdout_sample_sha256s": list(
                    self.fit.split.holdout_sample_sha256s
                ),
                "holdout_outcome_sha256s": list(
                    self.fit.split.holdout_outcome_sha256s
                ),
            },
        )

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_dict())
        if len(data) > MAX_DEMAND_LAG_RECEIPT_BYTES:
            raise ValueError("Demand lag validation exceeds its byte bound")
        return data

    def verify_or_raise(self) -> bool:
        self.fit.verify_or_raise()
        self.generic_validation.verify_or_raise()
        expected = validate_demand_lag_blanket(
            self.fit,
            minimum_holdout_accuracy=self.minimum_holdout_accuracy,
        )
        if expected.to_dict() != self.to_dict():
            raise DemandBlanketIntegrityError(
                "Demand lag validation does not recompute exactly"
            )
        return True

    @classmethod
    def from_bytes(cls, data: bytes) -> "DemandLagBlanketValidationReceipt":
        value = _strict_json(data, label="Demand lag validation")
        body = _sealed_body(
            value,
            schema=DEMAND_LAG_VALIDATION_SCHEMA,
            label="Demand lag validation",
        )
        expected = {
            "fit",
            "fit_sha256",
            "generic_validation",
            "generic_validation_sha256",
            "minimum_holdout_accuracy",
            "accepted",
            "reason",
            "selected_lags",
            "holdout_sample_sha256s",
            "holdout_outcome_sha256s",
        }
        if set(body) != expected:
            raise DemandBlanketIntegrityError(
                "invalid Demand lag validation body"
            )
        try:
            fit = DemandLagBlanketFitReceipt.from_bytes(
                canonical_json_bytes(body.get("fit"))
            )
            generic = ConditionalBlanketValidationReceipt.from_document(
                cast(Mapping[str, object], body.get("generic_validation"))
            )
            result = cls(
                fit=fit,
                generic_validation=generic,
                minimum_holdout_accuracy=_parse_fraction(
                    body.get("minimum_holdout_accuracy"),
                    field="minimum_holdout_accuracy",
                ),
                accepted=cast(bool, body.get("accepted")),
                reason=cast(str, body.get("reason")),
            )
            pins = {
                "fit_sha256": fit.sha256,
                "generic_validation_sha256": generic.sha256,
                "selected_lags": list(fit.selected_lags),
                "holdout_sample_sha256s": list(
                    fit.split.holdout_sample_sha256s
                ),
                "holdout_outcome_sha256s": list(
                    fit.split.holdout_outcome_sha256s
                ),
            }
            if any(body.get(field) != actual for field, actual in pins.items()):
                raise DemandBlanketIntegrityError(
                    "Demand lag validation pin changed"
                )
        except DemandBlanketIntegrityError:
            raise
        except (ConditionalBlanketIntegrityError, TypeError, ValueError) as exc:
            raise DemandBlanketIntegrityError(
                "Demand lag validation failed validation"
            ) from exc
        if result.to_bytes() != data:
            raise DemandBlanketIntegrityError(
                "Demand lag validation changed during reconstruction"
            )
        result.verify_or_raise()
        return result


def validate_demand_lag_blanket(
    fit: DemandLagBlanketFitReceipt,
    *,
    minimum_holdout_accuracy: Fraction = Fraction(3, 4),
) -> DemandLagBlanketValidationReceipt:
    if not isinstance(fit, DemandLagBlanketFitReceipt):
        raise TypeError("fit must be a DemandLagBlanketFitReceipt")
    if isinstance(minimum_holdout_accuracy, bool) or not isinstance(
        minimum_holdout_accuracy, (int, Fraction)
    ):
        raise TypeError(
            "minimum_holdout_accuracy must be exact int/Fraction data"
        )
    minimum = Fraction(minimum_holdout_accuracy)
    if not 0 <= minimum <= 1:
        raise ValueError("minimum_holdout_accuracy must lie in [0, 1]")
    _train, _calibration, holdout = _partition_samples(fit.corpus, fit.split)
    try:
        generic = validate_conditional_blanket(fit.generic_fit, holdout)
    except (ConditionalBlanketIntegrityError, ValueError) as exc:
        raise DemandBlanketIntegrityError(
            "generic conditional blanket rejected Demand holdout evidence"
        ) from exc
    accepted, reason = _validation_verdict(fit, generic, minimum)
    return DemandLagBlanketValidationReceipt(
        fit=fit,
        generic_validation=generic,
        minimum_holdout_accuracy=minimum,
        accepted=accepted,
        reason=reason,
    )


_PREDICTION_REASONS = frozenset(
    {
        "validated-high-confidence-singleton",
        "validation-rejected",
        "history-outside-route-alphabet",
        "probability-below-threshold",
        "top-probability-tie",
    }
)

_PREDICTION_CONTEXT_SCHEMA = "immer-ooe-demand-lag-prediction-context/v1"
_PREDICTION_CELL_SCHEMA = "immer-ooe-demand-lag-prediction-cell/v1"


def _prediction_context_sha256(
    arm_sha256: str,
    feature_names: Sequence[str],
    context: Sequence[str],
) -> str:
    return _digest(
        {
            "schema": _PREDICTION_CONTEXT_SCHEMA,
            "arm_sha256": require_sha256(arm_sha256, field="arm_sha256"),
            "feature_names": list(feature_names),
            "context": list(context),
        }
    )


def _prediction_cell_sha256(
    *,
    arm_sha256: str,
    feature_names: Sequence[str],
    context: Sequence[str],
    target_counts: Sequence[int],
    probabilities: Sequence[Fraction],
) -> str:
    return _digest(
        {
            "schema": _PREDICTION_CELL_SCHEMA,
            "arm_sha256": require_sha256(arm_sha256, field="arm_sha256"),
            "feature_names": list(feature_names),
            "context": list(context),
            "target_counts": list(target_counts),
            "probabilities": [
                _fraction_dict(value) for value in probabilities
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class DemandLagBlanketPredictionReceipt:
    generic_fit_sha256: str
    validation_sha256: str
    fit_scheduler_state_sha256: str
    scheduler_config_sha256: str
    scheduler_state_sha256: str
    graph_generation: int
    graph_state_sha256: str
    input_abi_sha256: str
    output_abi_sha256: str | None
    route_alphabet_sha256s: tuple[str, ...]
    max_context_order: int
    selected_lags: tuple[int, ...]
    history_route_sha256s: tuple[str, ...]
    lag_route_sha256s: tuple[str, ...]
    model_arm: str
    model_arm_sha256: str
    model_feature_names: tuple[str, ...]
    model_context: tuple[str, ...]
    model_context_sha256: str
    context_seen: bool
    matched_cell_sha256: str | None
    target_counts: tuple[int, ...]
    laplace_alpha: Fraction
    probabilities: tuple[Fraction, ...]
    minimum_probability: Fraction
    validation_accepted: bool
    candidate_route_sha256: str | None
    probability: Fraction
    reason: str

    def __post_init__(self) -> None:
        for field in (
            "generic_fit_sha256",
            "validation_sha256",
            "fit_scheduler_state_sha256",
            "scheduler_config_sha256",
            "scheduler_state_sha256",
            "graph_state_sha256",
            "input_abi_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        if self.output_abi_sha256 is not None:
            object.__setattr__(
                self,
                "output_abi_sha256",
                require_sha256(self.output_abi_sha256, field="output_abi_sha256"),
            )
        generation = _uint(self.graph_generation, field="graph_generation")
        order = _uint(
            self.max_context_order, field="max_context_order", positive=True
        )
        if order > MAX_CONTEXT_ORDER:
            raise ValueError("prediction context order exceeds its bound")
        alphabet = _hashes(
            self.route_alphabet_sha256s,
            field="route_alphabet_sha256s",
            sorted_unique=True,
        )
        if len(alphabet) < 2:
            raise ValueError("prediction requires at least two route categories")
        lags = tuple(self.selected_lags)
        if (
            lags != tuple(sorted(set(lags)))
            or any(
                isinstance(lag, bool)
                or not isinstance(lag, int)
                or not 1 <= lag <= order
                for lag in lags
            )
        ):
            raise ValueError("selected_lags are invalid")
        history = _hashes(
            self.history_route_sha256s, field="history_route_sha256s"
        )
        lag_values = _hashes(
            self.lag_route_sha256s, field="lag_route_sha256s"
        )
        if len(lag_values) != order:
            raise ValueError("prediction lag width changed")
        expected_lags = tuple(
            history[-lag] if len(history) >= lag else DEMAND_LAG_BOS
            for lag in range(1, order + 1)
        )
        if lag_values != expected_lags:
            raise ValueError("prediction lags differ from exact history")
        if self.model_arm not in {"selected", "marginal"}:
            raise ValueError("prediction model arm must be selected or marginal")
        arm_sha = require_sha256(
            self.model_arm_sha256, field="model_arm_sha256"
        )
        feature_names = tuple(self.model_feature_names)
        if (
            feature_names != tuple(sorted(set(feature_names)))
            or any(
                not isinstance(name, str)
                or not name
                or name != name.strip()
                for name in feature_names
            )
        ):
            raise ValueError("prediction model feature names are invalid")
        context = tuple(self.model_context)
        if len(context) != len(feature_names) or any(
            not isinstance(category, str)
            or not category
            or category != category.strip()
            for category in context
        ):
            raise ValueError("prediction model context is invalid")
        context_sha = require_sha256(
            self.model_context_sha256, field="model_context_sha256"
        )
        if context_sha != _prediction_context_sha256(
            arm_sha, feature_names, context
        ):
            raise ValueError("prediction model context hash is inconsistent")
        if not isinstance(self.context_seen, bool):
            raise TypeError("context_seen must be bool")
        matched_cell = self.matched_cell_sha256
        if matched_cell is not None:
            matched_cell = require_sha256(
                matched_cell, field="matched_cell_sha256"
            )
        if self.context_seen != (matched_cell is not None):
            raise ValueError("context_seen differs from matched-cell evidence")
        counts = tuple(self.target_counts)
        if (
            len(counts) != len(alphabet)
            or any(
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                for count in counts
            )
            or sum(counts) <= 0
        ):
            raise ValueError("prediction target counts are invalid")
        if isinstance(self.laplace_alpha, bool) or not isinstance(
            self.laplace_alpha, (int, Fraction)
        ):
            raise TypeError("laplace_alpha must be exact int/Fraction data")
        alpha = Fraction(self.laplace_alpha)
        if alpha <= 0:
            raise ValueError("laplace_alpha must be positive")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, Fraction))
            for value in self.probabilities
        ):
            raise TypeError("probabilities must be exact int/Fraction data")
        probabilities = tuple(Fraction(value) for value in self.probabilities)
        denominator = Fraction(sum(counts)) + alpha * len(counts)
        expected_probabilities = tuple(
            (Fraction(count) + alpha) / denominator for count in counts
        )
        if probabilities != expected_probabilities:
            raise ValueError("prediction probabilities differ from exact counts")
        if matched_cell is not None and matched_cell != _prediction_cell_sha256(
            arm_sha256=arm_sha,
            feature_names=feature_names,
            context=context,
            target_counts=counts,
            probabilities=probabilities,
        ):
            raise ValueError("matched prediction cell hash is inconsistent")
        if isinstance(self.minimum_probability, bool) or not isinstance(
            self.minimum_probability, (int, Fraction)
        ):
            raise TypeError(
                "minimum_probability must be exact int/Fraction data"
            )
        threshold = Fraction(self.minimum_probability)
        if not 0 <= threshold <= 1:
            raise ValueError("minimum_probability must lie in [0, 1]")
        if not isinstance(self.validation_accepted, bool):
            raise TypeError("validation_accepted must be bool")
        maximum = max(probabilities)
        winners = tuple(
            index
            for index, value in enumerate(probabilities)
            if value == maximum
        )
        history_compatible = all(
            value == DEMAND_LAG_BOS or value in alphabet
            for value in lag_values
        )
        if not self.validation_accepted:
            expected_candidate = None
            expected_reason = "validation-rejected"
        elif not history_compatible:
            expected_candidate = None
            expected_reason = "history-outside-route-alphabet"
        elif len(winners) != 1:
            expected_candidate = None
            expected_reason = "top-probability-tie"
        elif maximum < threshold:
            expected_candidate = None
            expected_reason = "probability-below-threshold"
        else:
            expected_candidate = alphabet[winners[0]]
            expected_reason = "validated-high-confidence-singleton"
        candidate = self.candidate_route_sha256
        if candidate is not None:
            candidate = require_sha256(candidate, field="candidate_route_sha256")
        if isinstance(self.probability, bool) or not isinstance(
            self.probability, (int, Fraction)
        ):
            raise TypeError("probability must be exact int/Fraction data")
        probability = Fraction(self.probability)
        if (
            candidate != expected_candidate
            or probability != maximum
            or self.reason != expected_reason
            or self.reason not in _PREDICTION_REASONS
        ):
            raise ValueError("Demand lag prediction verdict is inconsistent")
        object.__setattr__(self, "graph_generation", generation)
        object.__setattr__(self, "max_context_order", order)
        object.__setattr__(self, "route_alphabet_sha256s", alphabet)
        object.__setattr__(self, "selected_lags", lags)
        object.__setattr__(self, "history_route_sha256s", history)
        object.__setattr__(self, "lag_route_sha256s", lag_values)
        object.__setattr__(self, "model_arm_sha256", arm_sha)
        object.__setattr__(self, "model_feature_names", feature_names)
        object.__setattr__(self, "model_context", context)
        object.__setattr__(self, "model_context_sha256", context_sha)
        object.__setattr__(self, "matched_cell_sha256", matched_cell)
        object.__setattr__(self, "target_counts", counts)
        object.__setattr__(self, "laplace_alpha", alpha)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "minimum_probability", threshold)
        object.__setattr__(self, "candidate_route_sha256", candidate)
        object.__setattr__(self, "probability", probability)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @property
    def selected_route_sha256(self) -> str | None:
        return self.candidate_route_sha256

    def to_dict(self) -> dict[str, object]:
        return _sealed(
            DEMAND_LAG_PREDICTION_SCHEMA,
            {
                "generic_fit_sha256": self.generic_fit_sha256,
                "validation_sha256": self.validation_sha256,
                "fit_scheduler_state_sha256": self.fit_scheduler_state_sha256,
                "scheduler_config_sha256": self.scheduler_config_sha256,
                "scheduler_state_sha256": self.scheduler_state_sha256,
                "graph_generation": self.graph_generation,
                "graph_state_sha256": self.graph_state_sha256,
                "input_abi_sha256": self.input_abi_sha256,
                "output_abi_sha256": self.output_abi_sha256,
                "route_alphabet_sha256s": list(self.route_alphabet_sha256s),
                "max_context_order": self.max_context_order,
                "selected_lags": list(self.selected_lags),
                "history_route_sha256s": list(self.history_route_sha256s),
                "lag_route_sha256s": list(self.lag_route_sha256s),
                "model_arm": self.model_arm,
                "model_arm_sha256": self.model_arm_sha256,
                "model_feature_names": list(self.model_feature_names),
                "model_context": list(self.model_context),
                "model_context_sha256": self.model_context_sha256,
                "context_seen": self.context_seen,
                "matched_cell_sha256": self.matched_cell_sha256,
                "target_counts": list(self.target_counts),
                "laplace_alpha": _fraction_dict(self.laplace_alpha),
                "probabilities": [
                    _fraction_dict(value) for value in self.probabilities
                ],
                "minimum_probability": _fraction_dict(
                    self.minimum_probability
                ),
                "validation_accepted": self.validation_accepted,
                "candidate_route_sha256": self.candidate_route_sha256,
                "probability": _fraction_dict(self.probability),
                "reason": self.reason,
            },
        )

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        validation: DemandLagBlanketValidationReceipt | None = None,
    ) -> "DemandLagBlanketPredictionReceipt":
        value = _strict_json(data, label="Demand lag prediction")
        body = _sealed_body(
            value,
            schema=DEMAND_LAG_PREDICTION_SCHEMA,
            label="Demand lag prediction",
        )
        expected = {
            "generic_fit_sha256",
            "validation_sha256",
            "fit_scheduler_state_sha256",
            "scheduler_config_sha256",
            "scheduler_state_sha256",
            "graph_generation",
            "graph_state_sha256",
            "input_abi_sha256",
            "output_abi_sha256",
            "route_alphabet_sha256s",
            "max_context_order",
            "selected_lags",
            "history_route_sha256s",
            "lag_route_sha256s",
            "model_arm",
            "model_arm_sha256",
            "model_feature_names",
            "model_context",
            "model_context_sha256",
            "context_seen",
            "matched_cell_sha256",
            "target_counts",
            "laplace_alpha",
            "probabilities",
            "minimum_probability",
            "validation_accepted",
            "candidate_route_sha256",
            "probability",
            "reason",
        }
        sequence_fields = (
            "route_alphabet_sha256s",
            "selected_lags",
            "history_route_sha256s",
            "lag_route_sha256s",
            "model_feature_names",
            "model_context",
            "target_counts",
            "probabilities",
        )
        if set(body) != expected or any(
            not isinstance(body.get(field), list) for field in sequence_fields
        ):
            raise DemandBlanketIntegrityError(
                "invalid Demand lag prediction body"
            )
        try:
            result = cls(
                generic_fit_sha256=cast(str, body.get("generic_fit_sha256")),
                validation_sha256=cast(str, body.get("validation_sha256")),
                fit_scheduler_state_sha256=cast(
                    str, body.get("fit_scheduler_state_sha256")
                ),
                scheduler_config_sha256=cast(
                    str, body.get("scheduler_config_sha256")
                ),
                scheduler_state_sha256=cast(
                    str, body.get("scheduler_state_sha256")
                ),
                graph_generation=cast(int, body.get("graph_generation")),
                graph_state_sha256=cast(str, body.get("graph_state_sha256")),
                input_abi_sha256=cast(str, body.get("input_abi_sha256")),
                output_abi_sha256=cast(
                    str | None, body.get("output_abi_sha256")
                ),
                route_alphabet_sha256s=tuple(
                    cast(list[str], body.get("route_alphabet_sha256s"))
                ),
                max_context_order=cast(int, body.get("max_context_order")),
                selected_lags=tuple(
                    cast(list[int], body.get("selected_lags"))
                ),
                history_route_sha256s=tuple(
                    cast(list[str], body.get("history_route_sha256s"))
                ),
                lag_route_sha256s=tuple(
                    cast(list[str], body.get("lag_route_sha256s"))
                ),
                model_arm=cast(str, body.get("model_arm")),
                model_arm_sha256=cast(str, body.get("model_arm_sha256")),
                model_feature_names=tuple(
                    cast(list[str], body.get("model_feature_names"))
                ),
                model_context=tuple(
                    cast(list[str], body.get("model_context"))
                ),
                model_context_sha256=cast(
                    str, body.get("model_context_sha256")
                ),
                context_seen=cast(bool, body.get("context_seen")),
                matched_cell_sha256=cast(
                    str | None, body.get("matched_cell_sha256")
                ),
                target_counts=tuple(
                    cast(list[int], body.get("target_counts"))
                ),
                laplace_alpha=_parse_fraction(
                    body.get("laplace_alpha"), field="laplace_alpha"
                ),
                probabilities=tuple(
                    _parse_fraction(row, field="probability")
                    for row in cast(list[object], body.get("probabilities"))
                ),
                minimum_probability=_parse_fraction(
                    body.get("minimum_probability"),
                    field="minimum_probability",
                ),
                validation_accepted=cast(
                    bool, body.get("validation_accepted")
                ),
                candidate_route_sha256=cast(
                    str | None, body.get("candidate_route_sha256")
                ),
                probability=_parse_fraction(
                    body.get("probability"), field="probability"
                ),
                reason=cast(str, body.get("reason")),
            )
        except (TypeError, ValueError) as exc:
            raise DemandBlanketIntegrityError(
                "Demand lag prediction failed validation"
            ) from exc
        if result.to_bytes() != data:
            raise DemandBlanketIntegrityError(
                "Demand lag prediction changed during reconstruction"
            )
        if validation is not None:
            result.verify_against(validation)
        return result

    def verify_against(
        self, validation: DemandLagBlanketValidationReceipt
    ) -> bool:
        """Replay the table-cell proof against the exact frozen fit."""

        if not isinstance(validation, DemandLagBlanketValidationReceipt):
            raise TypeError(
                "validation must be a DemandLagBlanketValidationReceipt"
            )
        validation.verify_or_raise()
        fit = validation.fit
        corpus = fit.corpus
        if (
            self.validation_sha256 != validation.sha256
            or self.generic_fit_sha256 != fit.generic_fit.sha256
            or self.fit_scheduler_state_sha256
            != corpus.scheduler_state_sha256
            or self.scheduler_config_sha256 != corpus.scheduler_config_sha256
            or self.graph_generation != corpus.graph_generation
            or self.graph_state_sha256 != corpus.graph_state_sha256
            or self.input_abi_sha256 != corpus.input_abi_sha256
            or self.output_abi_sha256 != corpus.output_abi_sha256
            or self.route_alphabet_sha256s
            != corpus.route_alphabet_sha256s
            or self.max_context_order != corpus.max_context_order
            or self.selected_lags != fit.selected_lags
            or self.validation_accepted != validation.accepted
        ):
            raise DemandBlanketIntegrityError(
                "Demand prediction pins differ from its validation"
            )
        arm = (
            fit.generic_fit.selected_arm
            if validation.accepted
            else fit.generic_fit.marginal_arm
        )
        feature_map = {
            _feature_name(index): category
            for index, category in enumerate(
                self.lag_route_sha256s, start=1
            )
        }
        context = tuple(feature_map[name] for name in arm.feature_names)
        cell = next((row for row in arm.cells if row.context == context), None)
        probabilities = predict_probabilities(
            fit.generic_fit, feature_map, arm=arm.name
        )
        counts = arm.marginal_counts if cell is None else cell.target_counts
        matched_cell = (
            None
            if cell is None
            else _prediction_cell_sha256(
                arm_sha256=arm.sha256,
                feature_names=arm.feature_names,
                context=context,
                target_counts=counts,
                probabilities=probabilities,
            )
        )
        if (
            self.model_arm != arm.name
            or self.model_arm_sha256 != arm.sha256
            or self.model_feature_names != arm.feature_names
            or self.model_context != context
            or self.model_context_sha256
            != _prediction_context_sha256(
                arm.sha256, arm.feature_names, context
            )
            or self.context_seen != (cell is not None)
            or self.matched_cell_sha256 != matched_cell
            or self.target_counts != counts
            or self.laplace_alpha != fit.generic_fit.config.laplace_alpha
            or self.probabilities != probabilities
        ):
            raise DemandBlanketIntegrityError(
                "Demand prediction cell differs from its frozen fit"
            )
        return True


def predict_demand_lag_blanket(
    validation: DemandLagBlanketValidationReceipt,
    scheduler: OperatorDemandScheduler,
    graph_state: ComputeOperatorGraphState,
    history_route_sha256s: Sequence[str],
    *,
    input_abi_sha256: str,
    output_abi_sha256: str | None = None,
    minimum_probability: Fraction = Fraction(3, 4),
) -> DemandLagBlanketPredictionReceipt:
    """Predict one route while authenticating a frozen fit as an ancestor."""

    if not isinstance(validation, DemandLagBlanketValidationReceipt):
        raise TypeError("validation must be a DemandLagBlanketValidationReceipt")
    if not isinstance(scheduler, OperatorDemandScheduler):
        raise TypeError("scheduler must be an OperatorDemandScheduler")
    if not isinstance(graph_state, ComputeOperatorGraphState):
        raise TypeError("graph_state must be a ComputeOperatorGraphState")
    fit = validation.fit
    corpus = fit.corpus
    input_abi = require_sha256(input_abi_sha256, field="input_abi_sha256")
    output_abi = (
        None
        if output_abi_sha256 is None
        else require_sha256(output_abi_sha256, field="output_abi_sha256")
    )
    if (
        graph_state.generation != corpus.graph_generation
        or graph_state.sha256 != corpus.graph_state_sha256
        or input_abi != corpus.input_abi_sha256
        or output_abi != corpus.output_abi_sha256
        or scheduler.config.sha256 != corpus.scheduler_config_sha256
    ):
        raise DemandBlanketIntegrityError(
            "Demand blanket graph, ABI, or scheduler configuration changed"
        )
    # Validation hashes are evidence addresses, never capabilities.  Recompute
    # the full generic fit and holdout result on every public authorization so
    # a caller cannot construct a forged accepted object and bless it with its
    # own matching hashes.
    validation.verify_or_raise()
    validation_address = validation.sha256
    generic_fit_address = fit.generic_fit.sha256
    current = scheduler.state()
    if not scheduler.is_state_ancestor(
        corpus.scheduler_state_sha256, current.sha256
    ):
        raise DemandBlanketIntegrityError(
            "Demand blanket fit head is not an authenticated scheduler ancestor"
        )
    history = _hashes(
        history_route_sha256s, field="history_route_sha256s"
    )
    lag_values = tuple(
        history[-lag] if len(history) >= lag else DEMAND_LAG_BOS
        for lag in range(1, corpus.max_context_order + 1)
    )
    feature_map = {
        _feature_name(index): category
        for index, category in enumerate(lag_values, start=1)
    }
    generic_arm_name = "selected" if validation.accepted else "marginal"
    try:
        probabilities = predict_probabilities(
            fit.generic_fit, feature_map, arm=generic_arm_name
        )
    except (ConditionalBlanketIntegrityError, ValueError) as exc:
        raise DemandBlanketIntegrityError(
            "generic conditional blanket rejected Demand prediction"
        ) from exc
    arm = (
        fit.generic_fit.selected_arm
        if validation.accepted
        else fit.generic_fit.marginal_arm
    )
    context = tuple(feature_map[name] for name in arm.feature_names)
    cell = next((row for row in arm.cells if row.context == context), None)
    counts = arm.marginal_counts if cell is None else cell.target_counts
    if probabilities != tuple(
        value.fraction
        for value in (
            arm.marginal_probabilities if cell is None else cell.probabilities
        )
    ):
        raise DemandBlanketIntegrityError(
            "generic prediction differs from its exact selected-arm cell"
        )
    if isinstance(minimum_probability, bool) or not isinstance(
        minimum_probability, (int, Fraction)
    ):
        raise TypeError("minimum_probability must be exact int/Fraction data")
    threshold = Fraction(minimum_probability)
    maximum = max(probabilities)
    winners = tuple(
        index for index, value in enumerate(probabilities) if value == maximum
    )
    compatible = all(
        value == DEMAND_LAG_BOS
        or value in corpus.route_alphabet_sha256s
        for value in lag_values
    )
    if not validation.accepted:
        candidate = None
        reason = "validation-rejected"
    elif not compatible:
        candidate = None
        reason = "history-outside-route-alphabet"
    elif len(winners) != 1:
        candidate = None
        reason = "top-probability-tie"
    elif maximum < threshold:
        candidate = None
        reason = "probability-below-threshold"
    else:
        candidate = corpus.route_alphabet_sha256s[winners[0]]
        reason = "validated-high-confidence-singleton"
    return DemandLagBlanketPredictionReceipt(
        generic_fit_sha256=generic_fit_address,
        validation_sha256=validation_address,
        fit_scheduler_state_sha256=corpus.scheduler_state_sha256,
        scheduler_config_sha256=corpus.scheduler_config_sha256,
        scheduler_state_sha256=current.sha256,
        graph_generation=corpus.graph_generation,
        graph_state_sha256=corpus.graph_state_sha256,
        input_abi_sha256=corpus.input_abi_sha256,
        output_abi_sha256=corpus.output_abi_sha256,
        route_alphabet_sha256s=corpus.route_alphabet_sha256s,
        max_context_order=corpus.max_context_order,
        selected_lags=fit.selected_lags,
        history_route_sha256s=history,
        lag_route_sha256s=lag_values,
        model_arm=arm.name,
        model_arm_sha256=arm.sha256,
        model_feature_names=arm.feature_names,
        model_context=context,
        model_context_sha256=_prediction_context_sha256(
            arm.sha256, arm.feature_names, context
        ),
        context_seen=cell is not None,
        matched_cell_sha256=(
            None
            if cell is None
            else _prediction_cell_sha256(
                arm_sha256=arm.sha256,
                feature_names=arm.feature_names,
                context=context,
                target_counts=counts,
                probabilities=probabilities,
            )
        ),
        target_counts=counts,
        laplace_alpha=fit.generic_fit.config.laplace_alpha,
        probabilities=probabilities,
        minimum_probability=threshold,
        validation_accepted=validation.accepted,
        candidate_route_sha256=candidate,
        probability=maximum,
        reason=reason,
    )


__all__ = [
    "DEMAND_LAG_BOS",
    "DEMAND_LAG_CORPUS_SCHEMA",
    "DEMAND_LAG_FIT_SCHEMA",
    "DEMAND_LAG_PREDICTION_SCHEMA",
    "DEMAND_LAG_SAMPLE_SCHEMA",
    "DEMAND_LAG_SPLIT_SCHEMA",
    "DEMAND_LAG_VALIDATION_SCHEMA",
    "DemandBlanketError",
    "DemandBlanketIntegrityError",
    "DemandBlanketUnavailableError",
    "DemandEpisodeSplitReceipt",
    "DemandLagBlanketFitReceipt",
    "DemandLagBlanketPredictionReceipt",
    "DemandLagBlanketValidationReceipt",
    "DemandLagCorpusReceipt",
    "DemandLagSample",
    "MAX_DEMAND_LAG_RECEIPT_BYTES",
    "build_demand_lag_corpus",
    "chronological_demand_split",
    "chronological_episode_group_split",
    "fit_demand_lag_blanket",
    "predict_demand_lag_blanket",
    "validate_demand_lag_blanket",
]
