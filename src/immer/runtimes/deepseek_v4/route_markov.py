"""Token-row Markov hints for exact DeepSeek-V4 expert prefetching.

The official router remains authoritative.  This module consumes immutable
router traces and produces only expert-ordering scores for offline evaluation
or cache prefetch; it never changes a router row or skips model computation.
"""

from __future__ import annotations

from array import array
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import random
import re
from typing import Any

_SCHEMA = "deepseek-v4-token-row-markov-v1"
_MODES = ("markov", "marginal", "passthrough")
_UINT64_MAX = (1 << 64) - 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class RouteMarkovError(ValueError):
    """A route trace, model snapshot, or evaluation request is invalid."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RouteMarkovError("value is not canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RouteMarkovError(f"{label} must be an integer >= {minimum}")
    return value


def _positive_float(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise RouteMarkovError(f"{label} must be a finite positive number")
    return float(value)


def _observation_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RouteMarkovError("observation_id must be non-empty text")
    normalized = value.strip()
    if normalized != value or len(normalized) > 512:
        raise RouteMarkovError(
            "observation_id must be trimmed and at most 512 characters"
        )
    return normalized


def _seed(value: object) -> int:
    return _integer(value, "seed")


@dataclass(frozen=True, slots=True)
class LayerTokenRoutes:
    """Official expert IDs for every token row at one transformer layer."""

    layer: int
    rows: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layer", _integer(self.layer, "layer"))
        try:
            raw_rows = tuple(self.rows)
        except TypeError as exc:
            raise RouteMarkovError("rows must be an iterable of expert rows") from exc
        frozen: list[tuple[int, ...]] = []
        for raw_row in raw_rows:
            if isinstance(raw_row, (str, bytes)):
                raise RouteMarkovError(
                    "each route row must be an iterable of expert IDs"
                )
            try:
                row = tuple(_integer(expert, "expert ID") for expert in tuple(raw_row))
            except TypeError as exc:
                raise RouteMarkovError(
                    "each route row must be an iterable of expert IDs"
                ) from exc
            frozen.append(row)
        object.__setattr__(self, "rows", tuple(frozen))

    def as_record(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "rows": [list(row) for row in self.rows],
        }


@dataclass(frozen=True, slots=True)
class PromptRouteObservation:
    """One indivisible prompt trace; train/test splitting never splits its rows."""

    observation_id: str
    layers: tuple[LayerTokenRoutes, ...]
    target_overrides: tuple[LayerTokenRoutes, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_id", _observation_id(self.observation_id))
        try:
            raw_layers = tuple(self.layers)
        except TypeError as exc:
            raise RouteMarkovError("layers must be iterable") from exc
        frozen: list[LayerTokenRoutes] = []
        prior_layer = -1
        for raw in raw_layers:
            layer = raw if isinstance(raw, LayerTokenRoutes) else LayerTokenRoutes(*raw)
            if layer.layer <= prior_layer:
                raise RouteMarkovError("layers must be unique and strictly increasing")
            prior_layer = layer.layer
            frozen.append(layer)
        if not frozen:
            raise RouteMarkovError("an observation must contain at least one layer")
        for source, target in zip(frozen, frozen[1:], strict=False):
            if target.layer == source.layer + 1 and len(source.rows) != len(
                target.rows
            ):
                raise RouteMarkovError(
                    "consecutive layers must preserve token-row alignment"
                )
        object.__setattr__(self, "layers", tuple(frozen))

        try:
            raw_overrides = tuple(self.target_overrides)
        except TypeError as exc:
            raise RouteMarkovError("target_overrides must be iterable") from exc
        overrides: list[LayerTokenRoutes] = []
        layer_by_id = {layer.layer: layer for layer in frozen}
        prior_override = -1
        for raw in raw_overrides:
            override = (
                raw if isinstance(raw, LayerTokenRoutes) else LayerTokenRoutes(*raw)
            )
            if override.layer <= prior_override:
                raise RouteMarkovError(
                    "target overrides must be unique and strictly increasing"
                )
            base = layer_by_id.get(override.layer)
            if base is None or override.layer - 1 not in layer_by_id:
                raise RouteMarkovError(
                    "target override must belong to a consecutive observed layer"
                )
            if len(override.rows) != len(base.rows):
                raise RouteMarkovError("target override lost token-row alignment")
            prior_override = override.layer
            overrides.append(override)
        object.__setattr__(self, "target_overrides", tuple(overrides))

    @property
    def payload_sha256(self) -> str:
        return _sha256(self.payload_record())

    def payload_record(self) -> dict[str, Any]:
        return {
            "layers": [layer.as_record() for layer in self.layers],
            "schema": _SCHEMA,
            "target_overrides": [layer.as_record() for layer in self.target_overrides],
        }

    def as_record(self) -> dict[str, Any]:
        return {
            **self.payload_record(),
            "observation_id": self.observation_id,
        }

    def consecutive_pairs(
        self,
    ) -> tuple[tuple[LayerTokenRoutes, LayerTokenRoutes], ...]:
        """Return transitions with target-only interventions applied."""

        overrides = {layer.layer: layer for layer in self.target_overrides}
        return tuple(
            (source, overrides.get(target.layer, target))
            for source, target in zip(self.layers, self.layers[1:], strict=False)
            if target.layer == source.layer + 1
        )

    def base_consecutive_pairs(
        self,
    ) -> tuple[tuple[LayerTokenRoutes, LayerTokenRoutes], ...]:
        """Return the untouched official source and target trace."""

        return tuple(
            (source, target)
            for source, target in zip(self.layers, self.layers[1:], strict=False)
            if target.layer == source.layer + 1
        )


def _active_row_pairs(
    source: LayerTokenRoutes,
    target: LayerTokenRoutes,
) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...]], ...]:
    if target.layer != source.layer + 1:
        raise RouteMarkovError("route transition layers must be consecutive")
    if len(source.rows) != len(target.rows):
        raise RouteMarkovError("route transition lost token-row alignment")
    active: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = []
    for row_index, (source_row, target_row) in enumerate(
        zip(source.rows, target.rows, strict=True)
    ):
        if not source_row and not target_row:
            continue
        if not source_row or not target_row:
            raise RouteMarkovError(
                "padding rows must be empty in both consecutive layers"
            )
        active.append((row_index, source_row, target_row))
    return tuple(active)


@dataclass(frozen=True, slots=True)
class ObservationReceipt:
    observation_id: str
    payload_sha256: str
    appended: bool


@dataclass(frozen=True, slots=True)
class ScoreDistribution:
    """A normalized score for every official expert ID."""

    mode: str
    source_layer: int
    target_layer: int
    scores: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise RouteMarkovError(f"unknown prediction mode: {self.mode!r}")
        if not self.scores or any(
            not math.isfinite(score) or score < 0 for score in self.scores
        ):
            raise RouteMarkovError("scores must be finite non-negative values")
        if not math.isclose(sum(self.scores), 1.0, rel_tol=1e-10, abs_tol=1e-12):
            raise RouteMarkovError("scores must sum to one")

    @property
    def ranking(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                range(len(self.scores)),
                key=lambda expert: (-self.scores[expert], expert),
            )
        )

    def top_k(self, k: int) -> tuple[int, ...]:
        width = _integer(k, "k", minimum=1)
        if width > len(self.scores):
            raise RouteMarkovError("k exceeds the expert inventory")
        return self.ranking[:width]


@dataclass(frozen=True, slots=True)
class MicroWindowPrediction:
    """One ordered candidate set for consecutive active token rows."""

    source_layer: int
    target_layer: int
    active_row_start: int
    active_row_stop: int
    candidate_experts: tuple[int, ...]

    def __post_init__(self) -> None:
        source = _integer(self.source_layer, "source_layer")
        target = _integer(self.target_layer, "target_layer", minimum=1)
        start = _integer(self.active_row_start, "active_row_start")
        stop = _integer(self.active_row_stop, "active_row_stop", minimum=1)
        if target != source + 1:
            raise RouteMarkovError("micro-window target layer must be consecutive")
        if stop <= start:
            raise RouteMarkovError("micro-window active row interval is empty")
        candidates = tuple(self.candidate_experts)
        if not candidates or len(set(candidates)) != len(candidates):
            raise RouteMarkovError(
                "micro-window candidates must be non-empty and unique"
            )
        object.__setattr__(self, "source_layer", source)
        object.__setattr__(self, "target_layer", target)
        object.__setattr__(self, "active_row_start", start)
        object.__setattr__(self, "active_row_stop", stop)
        object.__setattr__(self, "candidate_experts", candidates)


@dataclass(frozen=True, slots=True)
class LayerMicroWindowPlan:
    """Causal row-local expert candidates for the next decoder layer."""

    source_layer: int
    target_layer: int
    active_rows: int
    window_rows: int
    windows: tuple[MicroWindowPrediction, ...]

    def __post_init__(self) -> None:
        source = _integer(self.source_layer, "source_layer")
        target = _integer(self.target_layer, "target_layer", minimum=1)
        active = _integer(self.active_rows, "active_rows", minimum=1)
        width = _integer(self.window_rows, "window_rows", minimum=1)
        windows = tuple(self.windows)
        if target != source + 1 or not windows:
            raise RouteMarkovError("micro-window plan has no consecutive target")
        cursor = 0
        for window in windows:
            if not isinstance(window, MicroWindowPrediction):
                raise RouteMarkovError("micro-window plan must contain predictions")
            if (
                window.source_layer != source
                or window.target_layer != target
                or window.active_row_start != cursor
                or window.active_row_stop - window.active_row_start > width
            ):
                raise RouteMarkovError("micro-window plan lost active-row alignment")
            cursor = window.active_row_stop
        if cursor != active:
            raise RouteMarkovError("micro-window plan does not cover every active row")
        object.__setattr__(self, "source_layer", source)
        object.__setattr__(self, "target_layer", target)
        object.__setattr__(self, "active_rows", active)
        object.__setattr__(self, "window_rows", width)
        object.__setattr__(self, "windows", windows)


class LayerMarkovExpertPredictor:
    """Per-layer dense uint64 transition counts with full-distribution scoring."""

    def __init__(self, *, n_experts: int = 256) -> None:
        self.n_experts = _integer(n_experts, "n_experts", minimum=1)
        self._matrices: dict[int, array[int]] = {}
        self._target_marginals: dict[int, array[int]] = {}
        self._target_rows: dict[int, int] = {}
        self._observation_digests: dict[str, str] = {}

    def _matrix(self, source_layer: int) -> array[int]:
        matrix = self._matrices.get(source_layer)
        if matrix is None:
            matrix = array("Q", [0]) * (self.n_experts * self.n_experts)
            self._matrices[source_layer] = matrix
        return matrix

    def _marginal(self, target_layer: int) -> array[int]:
        counts = self._target_marginals.get(target_layer)
        if counts is None:
            counts = array("Q", [0]) * self.n_experts
            self._target_marginals[target_layer] = counts
        return counts

    def _validate_expert(self, expert: int) -> None:
        if expert >= self.n_experts:
            raise RouteMarkovError(
                f"expert ID {expert} is outside the {self.n_experts}-expert inventory"
            )

    def _validate_observation(self, observation: PromptRouteObservation) -> None:
        if not isinstance(observation, PromptRouteObservation):
            raise RouteMarkovError("observation must be a PromptRouteObservation")
        for layer in (*observation.layers, *observation.target_overrides):
            for row in layer.rows:
                for expert in row:
                    self._validate_expert(expert)
        for source, target in observation.consecutive_pairs():
            _active_row_pairs(source, target)

    @property
    def observation_count(self) -> int:
        return len(self._observation_digests)

    @property
    def source_layers(self) -> tuple[int, ...]:
        return tuple(sorted(self._matrices))

    def observe(self, observation: PromptRouteObservation) -> ObservationReceipt:
        """Add one whole prompt atomically, or accept its idempotent replay."""

        self._validate_observation(observation)
        digest = observation.payload_sha256
        prior = self._observation_digests.get(observation.observation_id)
        if prior is not None:
            if prior != digest:
                raise RouteMarkovError(
                    "observation_id is already bound to different route evidence"
                )
            return ObservationReceipt(observation.observation_id, digest, False)

        matrix_deltas: dict[int, Counter[int]] = {}
        marginal_deltas: dict[int, Counter[int]] = {}
        row_deltas: Counter[int] = Counter()
        for source, target in observation.consecutive_pairs():
            matrix_delta = matrix_deltas.setdefault(source.layer, Counter())
            marginal_delta = marginal_deltas.setdefault(target.layer, Counter())
            for _row_index, source_row, target_row in _active_row_pairs(source, target):
                source_ids = tuple(sorted(set(source_row)))
                target_ids = tuple(sorted(set(target_row)))
                for source_id in source_ids:
                    base = source_id * self.n_experts
                    for target_id in target_ids:
                        matrix_delta[base + target_id] += 1
                marginal_delta.update(target_ids)
                row_deltas[target.layer] += 1

        for source_layer, deltas in matrix_deltas.items():
            matrix = self._matrices.get(source_layer)
            for index, delta in deltas.items():
                current = 0 if matrix is None else matrix[index]
                if current > _UINT64_MAX - delta:
                    raise OverflowError("Markov transition count exceeds uint64")
        for target_layer, deltas in marginal_deltas.items():
            marginal = self._target_marginals.get(target_layer)
            for expert, delta in deltas.items():
                current = 0 if marginal is None else marginal[expert]
                if current > _UINT64_MAX - delta:
                    raise OverflowError("target marginal count exceeds uint64")
            current_rows = self._target_rows.get(target_layer, 0)
            if current_rows > _UINT64_MAX - row_deltas[target_layer]:
                raise OverflowError("target row count exceeds uint64")

        for source_layer, deltas in matrix_deltas.items():
            matrix = self._matrix(source_layer)
            for index, delta in deltas.items():
                matrix[index] += delta
        for target_layer, deltas in marginal_deltas.items():
            marginal = self._marginal(target_layer)
            for expert, delta in deltas.items():
                marginal[expert] += delta
            self._target_rows[target_layer] = (
                self._target_rows.get(target_layer, 0) + row_deltas[target_layer]
            )
        self._observation_digests[observation.observation_id] = digest
        return ObservationReceipt(observation.observation_id, digest, True)

    def fit(self, observations: Iterable[PromptRouteObservation]) -> int:
        appended = 0
        for observation in observations:
            appended += int(self.observe(observation).appended)
        return appended

    def transition_count(self, source_layer: int, source: int, target: int) -> int:
        layer = _integer(source_layer, "source_layer")
        source_id = _integer(source, "source expert")
        target_id = _integer(target, "target expert")
        self._validate_expert(source_id)
        self._validate_expert(target_id)
        matrix = self._matrices.get(layer)
        if matrix is None:
            return 0
        return int(matrix[source_id * self.n_experts + target_id])

    def _row(self, row: Iterable[int]) -> tuple[int, ...]:
        if isinstance(row, (str, bytes)):
            raise RouteMarkovError("current row must contain expert IDs")
        try:
            selected = tuple(sorted(set(_integer(value, "expert ID") for value in row)))
        except TypeError as exc:
            raise RouteMarkovError("current row must contain expert IDs") from exc
        if not selected:
            raise RouteMarkovError("current row must not be empty")
        for expert in selected:
            self._validate_expert(expert)
        return selected

    def predict_distribution(
        self,
        *,
        source_layer: int,
        current_row: Iterable[int],
        alpha: float = 1.0,
    ) -> ScoreDistribution:
        """Average P(target expert | each current expert) with Dirichlet alpha."""

        layer = _integer(source_layer, "source_layer")
        selected = self._row(current_row)
        prior = _positive_float(alpha, "alpha")
        matrix = self._matrices.get(layer)
        scores = [0.0] * self.n_experts
        for source in selected:
            offset = source * self.n_experts
            row_total = (
                0 if matrix is None else sum(matrix[offset : offset + self.n_experts])
            )
            denominator = row_total + prior * self.n_experts
            for target in range(self.n_experts):
                count = 0 if matrix is None else matrix[offset + target]
                scores[target] += (count + prior) / denominator
        scale = 1.0 / len(selected)
        return ScoreDistribution(
            mode="markov",
            source_layer=layer,
            target_layer=layer + 1,
            scores=tuple(score * scale for score in scores),
        )

    def marginal_distribution(
        self,
        *,
        target_layer: int,
        alpha: float = 1.0,
    ) -> ScoreDistribution:
        """Layer marginal baseline, independent of the current router row."""

        layer = _integer(target_layer, "target_layer", minimum=1)
        prior = _positive_float(alpha, "alpha")
        marginal = self._target_marginals.get(layer)
        total = 0 if marginal is None else sum(marginal)
        denominator = total + prior * self.n_experts
        scores = tuple(
            ((0 if marginal is None else marginal[expert]) + prior) / denominator
            for expert in range(self.n_experts)
        )
        return ScoreDistribution(
            mode="marginal",
            source_layer=layer - 1,
            target_layer=layer,
            scores=scores,
        )

    def predict_window_distribution(
        self,
        *,
        source_layer: int,
        current_rows: Iterable[Iterable[int]],
        alpha: float = 1.0,
    ) -> ScoreDistribution:
        """Average full expert scores across one causal token micro-window."""

        if isinstance(current_rows, (str, bytes)):
            raise RouteMarkovError("current_rows must contain expert rows")
        try:
            rows = tuple(tuple(row) for row in current_rows)
        except TypeError as exc:
            raise RouteMarkovError("current_rows must contain expert rows") from exc
        active = tuple(row for row in rows if row)
        if not active:
            raise RouteMarkovError("current_rows must contain an active expert row")
        distributions = tuple(
            self.predict_distribution(
                source_layer=source_layer,
                current_row=row,
                alpha=alpha,
            )
            for row in active
        )
        scale = 1.0 / len(distributions)
        scores = tuple(
            sum(distribution.scores[expert] for distribution in distributions) * scale
            for expert in range(self.n_experts)
        )
        return ScoreDistribution(
            mode="markov",
            source_layer=distributions[0].source_layer,
            target_layer=distributions[0].target_layer,
            scores=scores,
        )

    def passthrough_distribution(
        self,
        *,
        source_layer: int,
        current_row: Iterable[int],
    ) -> ScoreDistribution:
        """Exact-current-ID baseline with all remaining experts tied at zero."""

        layer = _integer(source_layer, "source_layer")
        selected = self._row(current_row)
        scores = [0.0] * self.n_experts
        mass = 1.0 / len(selected)
        for expert in selected:
            scores[expert] = mass
        return ScoreDistribution(
            mode="passthrough",
            source_layer=layer,
            target_layer=layer + 1,
            scores=tuple(scores),
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a canonical sparse JSON snapshot of the compact dense counters."""

        layers: list[dict[str, Any]] = []
        all_source_layers = sorted(
            set(self._matrices) | {layer - 1 for layer in self._target_marginals}
        )
        for source_layer in all_source_layers:
            matrix = self._matrices.get(source_layer)
            marginal = self._target_marginals.get(source_layer + 1)
            transitions = []
            if matrix is not None:
                transitions = [
                    [index // self.n_experts, index % self.n_experts, int(count)]
                    for index, count in enumerate(matrix)
                    if count
                ]
            target_counts = []
            if marginal is not None:
                target_counts = [
                    [expert, int(count)]
                    for expert, count in enumerate(marginal)
                    if count
                ]
            layers.append(
                {
                    "source_layer": source_layer,
                    "target_counts": target_counts,
                    "target_rows": self._target_rows.get(source_layer + 1, 0),
                    "transitions": transitions,
                }
            )
        return {
            "layers": layers,
            "n_experts": self.n_experts,
            "observations": [
                [observation_id, digest]
                for observation_id, digest in sorted(self._observation_digests.items())
            ],
            "schema": _SCHEMA,
        }

    @classmethod
    def from_snapshot(cls, snapshot: object) -> LayerMarkovExpertPredictor:
        """Restore and fully validate one canonical sparse counter snapshot."""

        if not isinstance(snapshot, dict) or set(snapshot) != {
            "layers",
            "n_experts",
            "observations",
            "schema",
        }:
            raise RouteMarkovError("Markov snapshot has unknown or missing fields")
        if snapshot.get("schema") != _SCHEMA:
            raise RouteMarkovError("Markov snapshot schema is unsupported")
        n_experts = _integer(snapshot.get("n_experts"), "n_experts", minimum=1)
        model = cls(n_experts=n_experts)

        raw_layers = snapshot.get("layers")
        if not isinstance(raw_layers, list):
            raise RouteMarkovError("Markov snapshot layers must be a list")
        prior_source_layer = -1
        for raw_layer in raw_layers:
            if not isinstance(raw_layer, dict) or set(raw_layer) != {
                "source_layer",
                "target_counts",
                "target_rows",
                "transitions",
            }:
                raise RouteMarkovError("Markov snapshot layer is invalid")
            source_layer = _integer(raw_layer.get("source_layer"), "source_layer")
            if source_layer <= prior_source_layer:
                raise RouteMarkovError(
                    "Markov snapshot source layers must be strictly increasing"
                )
            prior_source_layer = source_layer
            target_layer = source_layer + 1
            target_rows = _integer(raw_layer.get("target_rows"), "target_rows")

            raw_transitions = raw_layer.get("transitions")
            if not isinstance(raw_transitions, list):
                raise RouteMarkovError("Markov snapshot transitions must be a list")
            prior_transition: tuple[int, int] | None = None
            matrix: array[int] | None = None
            for raw_transition in raw_transitions:
                if not isinstance(raw_transition, list) or len(raw_transition) != 3:
                    raise RouteMarkovError("Markov snapshot transition is invalid")
                source = _integer(raw_transition[0], "source expert")
                target = _integer(raw_transition[1], "target expert")
                count = _integer(raw_transition[2], "transition count", minimum=1)
                model._validate_expert(source)
                model._validate_expert(target)
                coordinate = (source, target)
                if prior_transition is not None and coordinate <= prior_transition:
                    raise RouteMarkovError(
                        "Markov snapshot transitions must be strictly increasing"
                    )
                if count > _UINT64_MAX:
                    raise RouteMarkovError("Markov transition count exceeds uint64")
                if matrix is None:
                    matrix = model._matrix(source_layer)
                matrix[source * n_experts + target] = count
                prior_transition = coordinate

            raw_counts = raw_layer.get("target_counts")
            if not isinstance(raw_counts, list):
                raise RouteMarkovError("Markov snapshot target counts must be a list")
            prior_expert = -1
            marginal: array[int] | None = None
            for raw_count in raw_counts:
                if not isinstance(raw_count, list) or len(raw_count) != 2:
                    raise RouteMarkovError("Markov snapshot target count is invalid")
                expert = _integer(raw_count[0], "target expert")
                count = _integer(raw_count[1], "target count", minimum=1)
                model._validate_expert(expert)
                if expert <= prior_expert:
                    raise RouteMarkovError(
                        "Markov snapshot target counts must be strictly increasing"
                    )
                if count > target_rows or count > _UINT64_MAX:
                    raise RouteMarkovError(
                        "Markov snapshot target count exceeds its row total"
                    )
                if marginal is None:
                    marginal = model._marginal(target_layer)
                marginal[expert] = count
                prior_expert = expert
            if bool(raw_counts) != bool(target_rows):
                raise RouteMarkovError(
                    "Markov snapshot target counts and row total disagree"
                )
            if target_rows:
                model._target_rows[target_layer] = target_rows
            if not raw_transitions and not raw_counts:
                raise RouteMarkovError("Markov snapshot layer contains no evidence")

        raw_observations = snapshot.get("observations")
        if not isinstance(raw_observations, list):
            raise RouteMarkovError("Markov snapshot observations must be a list")
        prior_observation = ""
        for raw_observation in raw_observations:
            if not isinstance(raw_observation, list) or len(raw_observation) != 2:
                raise RouteMarkovError("Markov snapshot observation is invalid")
            observation_id = _observation_id(raw_observation[0])
            digest = raw_observation[1]
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise RouteMarkovError(
                    "Markov snapshot observation digest must be SHA-256"
                )
            if observation_id <= prior_observation:
                raise RouteMarkovError(
                    "Markov snapshot observations must be strictly increasing"
                )
            model._observation_digests[observation_id] = digest
            prior_observation = observation_id

        if model.snapshot() != snapshot:
            raise RouteMarkovError("Markov snapshot is not canonical")
        return model

    @property
    def snapshot_sha256(self) -> str:
        return _sha256(self.snapshot())


def plan_micro_window_prefetch(
    predictor: LayerMarkovExpertPredictor,
    *,
    source_layer: int,
    current_rows: Iterable[Iterable[int]],
    window_rows: int = 2,
    k: int | None = None,
    alpha: float = 1.0,
) -> LayerMicroWindowPlan:
    """Convert row-local Markov distributions into an executable next-layer plan.

    Empty padding rows are removed without changing the order of active rows.
    With ``k=None`` each window predicts exactly as many candidates as it has
    official source routing slots, matching the measured micro-window protocol.
    """

    if not isinstance(predictor, LayerMarkovExpertPredictor):
        raise RouteMarkovError("predictor must be a LayerMarkovExpertPredictor")
    layer = _integer(source_layer, "source_layer")
    width = _integer(window_rows, "window_rows", minimum=1)
    fixed_k = None if k is None else _integer(k, "k", minimum=1)
    if fixed_k is not None and fixed_k > predictor.n_experts:
        raise RouteMarkovError("k exceeds the expert inventory")
    prior = _positive_float(alpha, "alpha")
    if isinstance(current_rows, (str, bytes)):
        raise RouteMarkovError("current_rows must contain expert rows")
    try:
        rows = tuple(tuple(row) for row in current_rows)
    except TypeError as exc:
        raise RouteMarkovError("current_rows must contain expert rows") from exc
    active = tuple(row for row in rows if row)
    if not active:
        raise RouteMarkovError("current_rows must contain an active expert row")

    windows: list[MicroWindowPrediction] = []
    for start in range(0, len(active), width):
        chunk = active[start : start + width]
        candidate_count = (
            min(predictor.n_experts, sum(len(row) for row in chunk))
            if fixed_k is None
            else fixed_k
        )
        distribution = predictor.predict_window_distribution(
            source_layer=layer,
            current_rows=chunk,
            alpha=prior,
        )
        windows.append(
            MicroWindowPrediction(
                source_layer=layer,
                target_layer=layer + 1,
                active_row_start=start,
                active_row_stop=start + len(chunk),
                candidate_experts=distribution.top_k(candidate_count),
            )
        )
    return LayerMicroWindowPlan(
        source_layer=layer,
        target_layer=layer + 1,
        active_rows=len(active),
        window_rows=width,
        windows=tuple(windows),
    )


@dataclass(frozen=True, slots=True)
class PromptSplit:
    train: tuple[PromptRouteObservation, ...]
    test: tuple[PromptRouteObservation, ...]


def _unique_observations(
    observations: Iterable[PromptRouteObservation],
) -> tuple[PromptRouteObservation, ...]:
    by_id: dict[str, PromptRouteObservation] = {}
    try:
        raw_observations = tuple(observations)
    except TypeError as exc:
        raise RouteMarkovError("observations must be iterable") from exc
    for observation in raw_observations:
        if not isinstance(observation, PromptRouteObservation):
            raise RouteMarkovError("observations must contain whole prompt traces")
        prior = by_id.get(observation.observation_id)
        if prior is not None and prior.payload_sha256 != observation.payload_sha256:
            raise RouteMarkovError(
                "duplicate observation_id has conflicting route evidence"
            )
        by_id[observation.observation_id] = observation
    return tuple(by_id[key] for key in sorted(by_id))


def split_prompt_observations(
    observations: Iterable[PromptRouteObservation],
    *,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> PromptSplit:
    """Deterministically split only at prompt boundaries."""

    prompts = _unique_observations(observations)
    if len(prompts) < 2:
        raise RouteMarkovError("a train/test split requires at least two prompts")
    if (
        isinstance(test_fraction, bool)
        or not isinstance(test_fraction, (int, float))
        or not math.isfinite(test_fraction)
        or not 0 < test_fraction < 1
    ):
        raise RouteMarkovError("test_fraction must be strictly between zero and one")
    normalized_seed = _seed(seed)
    ranked = sorted(
        prompts,
        key=lambda observation: (
            _sha256(
                {
                    "observation_id": observation.observation_id,
                    "seed": normalized_seed,
                }
            ),
            observation.observation_id,
        ),
    )
    n_test = min(len(ranked) - 1, max(1, round(len(ranked) * float(test_fraction))))
    test_ids = {observation.observation_id for observation in ranked[:n_test]}
    train = tuple(
        observation
        for observation in prompts
        if observation.observation_id not in test_ids
    )
    test = tuple(
        observation for observation in prompts if observation.observation_id in test_ids
    )
    return PromptSplit(train=train, test=test)


@dataclass(frozen=True, slots=True)
class PlaceboAssignment:
    observation_id: str
    source_layer: int
    target_layer: int
    row_index: int
    original_target: tuple[int, ...]
    shuffled_target: tuple[int, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "original_target": list(self.original_target),
            "row_index": self.row_index,
            "shuffled_target": list(self.shuffled_target),
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
        }


@dataclass(frozen=True, slots=True)
class TargetLayerPlacebo:
    """Target-only permutation; sources stay untouched during evaluation."""

    seed: int
    assignments: tuple[PlaceboAssignment, ...]

    @property
    def sha256(self) -> str:
        return _sha256(
            {
                "assignments": [entry.as_record() for entry in self.assignments],
                "schema": f"{_SCHEMA}:target-shuffle-v1",
                "seed": self.seed,
            }
        )

    def lookup(self) -> dict[tuple[str, int, int], tuple[int, ...]]:
        return {
            (entry.observation_id, entry.source_layer, entry.row_index): (
                entry.shuffled_target
            )
            for entry in self.assignments
        }


def build_target_layer_placebo(
    observations: Iterable[PromptRouteObservation],
    *,
    seed: int,
) -> TargetLayerPlacebo:
    """Shuffle target rows within layer and row width, preserving all marginals."""

    prompts = _unique_observations(observations)
    normalized_seed = _seed(seed)
    groups: dict[
        tuple[int, int, int],
        list[tuple[str, int, int, tuple[int, ...]]],
    ] = {}
    for observation in prompts:
        if observation.target_overrides:
            raise RouteMarkovError(
                "build the placebo from untouched official prompt observations"
            )
        for source, target in observation.base_consecutive_pairs():
            for row_index, _source_row, target_row in _active_row_pairs(source, target):
                group = (target.layer, len(target_row), len(set(target_row)))
                groups.setdefault(group, []).append(
                    (observation.observation_id, source.layer, row_index, target_row)
                )

    assignments: list[PlaceboAssignment] = []
    for group in sorted(groups):
        entries = sorted(groups[group], key=lambda entry: entry[:3])
        donor_rows = [entry[3] for entry in entries]
        if len(donor_rows) > 1:
            generator = random.Random(
                int(
                    _sha256(
                        {
                            "group": list(group),
                            "schema": f"{_SCHEMA}:target-shuffle-v1",
                            "seed": normalized_seed,
                        }
                    ),
                    16,
                )
            )
            generator.shuffle(donor_rows)
            if donor_rows == [entry[3] for entry in entries]:
                donor_rows = donor_rows[1:] + donor_rows[:1]
        for entry, shuffled in zip(entries, donor_rows, strict=True):
            observation_id, source_layer, row_index, original = entry
            assignments.append(
                PlaceboAssignment(
                    observation_id=observation_id,
                    source_layer=source_layer,
                    target_layer=source_layer + 1,
                    row_index=row_index,
                    original_target=original,
                    shuffled_target=shuffled,
                )
            )
    assignments.sort(
        key=lambda entry: (entry.observation_id, entry.source_layer, entry.row_index)
    )
    return TargetLayerPlacebo(normalized_seed, tuple(assignments))


def apply_target_layer_placebo(
    observations: Iterable[PromptRouteObservation],
    placebo: TargetLayerPlacebo,
) -> tuple[PromptRouteObservation, ...]:
    """Bind shuffled targets to whole prompts while retaining official sources.

    Overrides are target-only: layer ``l+1`` is shuffled when training ``M_l``,
    but its untouched official rows remain the source when training ``M_l+1``.
    This makes a real-vs-placebo predictor A/B possible without mutating traces.
    """

    if not isinstance(placebo, TargetLayerPlacebo):
        raise RouteMarkovError("placebo must be a TargetLayerPlacebo")
    prompts = _unique_observations(observations)
    if any(prompt.target_overrides for prompt in prompts):
        raise RouteMarkovError("cannot apply a placebo to an already overridden trace")

    assignments = {
        (entry.observation_id, entry.source_layer, entry.row_index): entry
        for entry in placebo.assignments
    }
    if len(assignments) != len(placebo.assignments):
        raise RouteMarkovError("placebo contains duplicate row assignments")
    originals: dict[tuple[int, int, int], Counter[tuple[int, ...]]] = {}
    shuffled: dict[tuple[int, int, int], Counter[tuple[int, ...]]] = {}
    for entry in placebo.assignments:
        group = (
            entry.target_layer,
            len(entry.original_target),
            len(set(entry.original_target)),
        )
        if (
            len(entry.shuffled_target) != group[1]
            or len(set(entry.shuffled_target)) != group[2]
        ):
            raise RouteMarkovError("placebo changed a target row width")
        originals.setdefault(group, Counter())[entry.original_target] += 1
        shuffled.setdefault(group, Counter())[entry.shuffled_target] += 1
    if originals != shuffled:
        raise RouteMarkovError("placebo changed target-layer row marginals")
    expected: set[tuple[str, int, int]] = set()
    transformed: list[PromptRouteObservation] = []
    for prompt in prompts:
        rows_by_target = {layer.layer: list(layer.rows) for layer in prompt.layers}
        overridden_layers: set[int] = set()
        for source, target in prompt.base_consecutive_pairs():
            for row_index, _source_row, target_row in _active_row_pairs(source, target):
                key = (prompt.observation_id, source.layer, row_index)
                expected.add(key)
                assignment = assignments.get(key)
                if assignment is None:
                    raise RouteMarkovError("placebo is missing an aligned target row")
                if (
                    assignment.target_layer != target.layer
                    or assignment.original_target != target_row
                ):
                    raise RouteMarkovError(
                        "placebo assignment does not match its official target row"
                    )
                rows_by_target[target.layer][row_index] = assignment.shuffled_target
                overridden_layers.add(target.layer)
        transformed.append(
            PromptRouteObservation(
                observation_id=prompt.observation_id,
                layers=prompt.layers,
                target_overrides=tuple(
                    LayerTokenRoutes(layer, tuple(rows_by_target[layer]))
                    for layer in sorted(overridden_layers)
                ),
            )
        )
    if set(assignments) != expected:
        raise RouteMarkovError(
            "placebo contains rows outside these prompt observations"
        )
    return tuple(transformed)


@dataclass(frozen=True, slots=True)
class KSweepPoint:
    k: int
    set_recall: float
    set_precision: float
    selection_mass_recall: float
    set_hits: int
    target_set_total: int
    predicted_total: int
    selection_mass_hits: int
    target_selection_mass: int


@dataclass(frozen=True, slots=True)
class LayerKSweep:
    target_layer: int
    evaluated_rows: int
    curve: tuple[KSweepPoint, ...]


@dataclass(frozen=True, slots=True)
class KSweepEvaluation:
    mode: str
    n_experts: int
    prompt_ids: tuple[str, ...]
    evaluated_rows: int
    curve: tuple[KSweepPoint, ...]
    layers: tuple[LayerKSweep, ...]
    placebo_sha256: str | None = None


@dataclass(slots=True)
class _SweepAccumulator:
    rows: int
    set_hits: list[int]
    target_set_total: int
    selection_hits: list[int]
    target_selection_mass: int

    @classmethod
    def create(cls, n_experts: int) -> _SweepAccumulator:
        return cls(0, [0] * n_experts, 0, [0] * n_experts, 0)

    def add(self, ranking: Sequence[int], target_row: Sequence[int]) -> None:
        target_counts = Counter(target_row)
        target_set = set(target_counts)
        cumulative_set = 0
        cumulative_mass = 0
        for index, expert in enumerate(ranking):
            if expert in target_set:
                cumulative_set += 1
                cumulative_mass += target_counts[expert]
            self.set_hits[index] += cumulative_set
            self.selection_hits[index] += cumulative_mass
        self.rows += 1
        self.target_set_total += len(target_set)
        self.target_selection_mass += len(target_row)

    def curve(self) -> tuple[KSweepPoint, ...]:
        return tuple(
            KSweepPoint(
                k=index + 1,
                set_recall=(
                    self.set_hits[index] / self.target_set_total
                    if self.target_set_total
                    else 0.0
                ),
                set_precision=(
                    self.set_hits[index] / (self.rows * (index + 1))
                    if self.rows
                    else 0.0
                ),
                selection_mass_recall=(
                    self.selection_hits[index] / self.target_selection_mass
                    if self.target_selection_mass
                    else 0.0
                ),
                set_hits=self.set_hits[index],
                target_set_total=self.target_set_total,
                predicted_total=self.rows * (index + 1),
                selection_mass_hits=self.selection_hits[index],
                target_selection_mass=self.target_selection_mass,
            )
            for index in range(len(self.set_hits))
        )


def evaluate_k_sweep(
    predictor: LayerMarkovExpertPredictor,
    observations: Iterable[PromptRouteObservation],
    *,
    mode: str = "markov",
    alpha: float = 1.0,
    placebo_seed: int | None = None,
) -> KSweepEvaluation:
    """Evaluate K=1..N without splitting any prompt into train/test rows."""

    if not isinstance(predictor, LayerMarkovExpertPredictor):
        raise RouteMarkovError("predictor must be a LayerMarkovExpertPredictor")
    if mode not in _MODES:
        raise RouteMarkovError(f"mode must be one of: {', '.join(_MODES)}")
    prior = _positive_float(alpha, "alpha")
    prompts = _unique_observations(observations)
    for observation in prompts:
        predictor._validate_observation(observation)
    placebo = (
        None
        if placebo_seed is None
        else build_target_layer_placebo(prompts, seed=placebo_seed)
    )
    overrides = {} if placebo is None else placebo.lookup()
    overall = _SweepAccumulator.create(predictor.n_experts)
    by_layer: dict[int, _SweepAccumulator] = {}
    for observation in prompts:
        for source, target in observation.consecutive_pairs():
            accumulator = by_layer.setdefault(
                target.layer, _SweepAccumulator.create(predictor.n_experts)
            )
            for row_index, source_row, target_row in _active_row_pairs(source, target):
                evaluated_target = overrides.get(
                    (observation.observation_id, source.layer, row_index),
                    target_row,
                )
                if mode == "markov":
                    distribution = predictor.predict_distribution(
                        source_layer=source.layer,
                        current_row=source_row,
                        alpha=prior,
                    )
                elif mode == "marginal":
                    distribution = predictor.marginal_distribution(
                        target_layer=target.layer,
                        alpha=prior,
                    )
                else:
                    distribution = predictor.passthrough_distribution(
                        source_layer=source.layer,
                        current_row=source_row,
                    )
                ranking = distribution.ranking
                overall.add(ranking, evaluated_target)
                accumulator.add(ranking, evaluated_target)
    if overall.rows == 0:
        raise RouteMarkovError("observations contain no evaluable aligned transitions")
    layers = tuple(
        LayerKSweep(
            target_layer=target_layer,
            evaluated_rows=by_layer[target_layer].rows,
            curve=by_layer[target_layer].curve(),
        )
        for target_layer in sorted(by_layer)
        if by_layer[target_layer].rows
    )
    return KSweepEvaluation(
        mode=mode,
        n_experts=predictor.n_experts,
        prompt_ids=tuple(observation.observation_id for observation in prompts),
        evaluated_rows=overall.rows,
        curve=overall.curve(),
        layers=layers,
        placebo_sha256=None if placebo is None else placebo.sha256,
    )


def evaluate_all_baselines(
    predictor: LayerMarkovExpertPredictor,
    observations: Iterable[PromptRouteObservation],
    *,
    alpha: float = 1.0,
    placebo_seed: int | None = None,
) -> dict[str, KSweepEvaluation]:
    """Run Markov, target-layer marginal, and exact-ID passthrough together."""

    prompts = _unique_observations(observations)
    return {
        mode: evaluate_k_sweep(
            predictor,
            prompts,
            mode=mode,
            alpha=alpha,
            placebo_seed=placebo_seed,
        )
        for mode in _MODES
    }


__all__ = [
    "KSweepEvaluation",
    "KSweepPoint",
    "LayerKSweep",
    "LayerMarkovExpertPredictor",
    "LayerMicroWindowPlan",
    "LayerTokenRoutes",
    "MicroWindowPrediction",
    "ObservationReceipt",
    "PlaceboAssignment",
    "PromptRouteObservation",
    "PromptSplit",
    "RouteMarkovError",
    "ScoreDistribution",
    "TargetLayerPlacebo",
    "apply_target_layer_placebo",
    "build_target_layer_placebo",
    "evaluate_all_baselines",
    "evaluate_k_sweep",
    "plan_micro_window_prefetch",
    "split_prompt_observations",
]
