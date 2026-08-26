from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .identity import canonical_json_bytes
from .math_core import RapidityLedger, normalize_rows


FloatArray = NDArray[np.float64]


def _finite_vector(value: NDArray[np.floating], *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _probability_vector(
    value: NDArray[np.floating], *, size: int, name: str
) -> FloatArray:
    array = _finite_vector(value, name=name)
    if array.size != size:
        raise ValueError(f"{name} must have length {size}")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    total = float(array.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive finite mass")
    return array / total


@dataclass(slots=True)
class _Cluster:
    count: int
    mean: FloatArray


@dataclass(frozen=True, slots=True)
class RouteDecision:
    label: str | None
    accepted: bool
    reason: str
    nearest_distance: float
    second_distance: float
    margin: float
    radius: float
    min_margin: float


class AttractorRouter:
    """Centroid router with independent radius and ambiguity gates.

    Radius rejects sketches outside every calibrated basin.  Margin rejects a
    sketch even inside a basin when the nearest and second-nearest sites are too
    similar.  A route is emitted only when both conditions pass.
    """

    def __init__(self, *, radius: float, min_margin: float = 0.05) -> None:
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("radius must be positive and finite")
        if not math.isfinite(min_margin) or min_margin < 0.0:
            raise ValueError("min_margin must be finite and non-negative")
        self.radius = float(radius)
        self.min_margin = float(min_margin)
        self._clusters: dict[str, _Cluster] = {}
        self._dimension: int | None = None
        self._calibration_sha256: str | None = None

    @property
    def dimension(self) -> int | None:
        return self._dimension

    @property
    def calibration_sha256(self) -> str | None:
        return self._calibration_sha256

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self._clusters))

    def observe(self, label: str, sketch: NDArray[np.floating]) -> None:
        if not isinstance(label, str) or not label:
            raise ValueError("label must be a non-empty string")
        value = _finite_vector(sketch, name="sketch")
        if self._dimension is None:
            self._dimension = int(value.size)
        elif value.size != self._dimension:
            raise ValueError(f"sketch must have length {self._dimension}")
        cluster = self._clusters.get(label)
        if cluster is None:
            self._clusters[label] = _Cluster(1, value.copy())
        else:
            cluster.count += 1
            cluster.mean += (value - cluster.mean) / cluster.count
        # New evidence changes the centroids.  Existing numerical thresholds
        # remain usable, but their calibration receipt no longer describes them.
        self._calibration_sha256 = None

    def calibrate(
        self,
        samples: Iterable[tuple[str, NDArray[np.floating]]],
        *,
        radius_quantile: float = 0.99,
        margin_quantile: float = 0.01,
        radius_ceiling: float | None = None,
    ) -> str:
        """Fit both abstention thresholds and return their evidence digest."""

        if not 0.0 < radius_quantile <= 1.0:
            raise ValueError("radius_quantile must lie in (0, 1]")
        if not 0.0 <= margin_quantile < 1.0:
            raise ValueError("margin_quantile must lie in [0, 1)")
        if radius_ceiling is not None and (
            not math.isfinite(radius_ceiling) or radius_ceiling <= 0.0
        ):
            raise ValueError("radius_ceiling must be positive and finite")
        if not self._clusters:
            raise RuntimeError("observe router centroids before calibration")

        observations: list[dict[str, object]] = []
        within: list[float] = []
        margins: list[float] = []
        for label, sketch in samples:
            if label not in self._clusters:
                raise ValueError(f"unknown calibration label: {label}")
            value = _finite_vector(sketch, name="calibration sketch")
            if value.size != self._dimension:
                raise ValueError(
                    f"calibration sketch must have length {self._dimension}"
                )
            own = float(np.linalg.norm(value - self._clusters[label].mean))
            competing = [
                float(np.linalg.norm(value - cluster.mean))
                for other, cluster in self._clusters.items()
                if other != label
            ]
            margin = min(competing) - own if competing else math.inf
            within.append(own)
            if math.isfinite(margin):
                margins.append(margin)
            observations.append({"label": label, "sketch": value.tolist()})
        if not observations:
            raise ValueError("calibration requires at least one sample")

        radius = float(np.quantile(within, radius_quantile, method="higher"))
        radius = max(radius, np.finfo(np.float64).eps)
        if radius_ceiling is not None:
            radius = min(radius, float(radius_ceiling))
        margin = (
            max(0.0, float(np.quantile(margins, margin_quantile, method="lower")))
            if margins
            else 0.0
        )
        self.radius = radius
        self.min_margin = margin
        receipt = {
            "clusters": [
                {
                    "count": self._clusters[label].count,
                    "label": label,
                    "mean": self._clusters[label].mean.tolist(),
                }
                for label in sorted(self._clusters)
            ],
            "dimension": self._dimension,
            "format": "immer-ooe-router-calibration/v1",
            "margin_quantile": margin_quantile,
            "min_margin": margin,
            "observations": observations,
            "radius": radius,
            "radius_ceiling": radius_ceiling,
            "radius_quantile": radius_quantile,
        }
        self._calibration_sha256 = hashlib.sha256(
            canonical_json_bytes(receipt)
        ).hexdigest()
        return self._calibration_sha256

    def decision(self, sketch: NDArray[np.floating]) -> RouteDecision:
        if not self._clusters:
            return RouteDecision(
                None,
                False,
                "untrained",
                math.inf,
                math.inf,
                0.0,
                self.radius,
                self.min_margin,
            )
        value = _finite_vector(sketch, name="sketch")
        if value.size != self._dimension:
            raise ValueError(f"sketch must have length {self._dimension}")
        ordered = sorted(
            (
                (float(np.linalg.norm(value - cluster.mean)), label)
                for label, cluster in self._clusters.items()
            ),
            key=lambda item: (item[0], item[1]),
        )
        nearest, label = ordered[0]
        second = ordered[1][0] if len(ordered) > 1 else math.inf
        margin = second - nearest
        if nearest > self.radius:
            return RouteDecision(
                None,
                False,
                "outside-radius",
                nearest,
                second,
                margin,
                self.radius,
                self.min_margin,
            )
        if margin < self.min_margin:
            return RouteDecision(
                None,
                False,
                "ambiguous-margin",
                nearest,
                second,
                margin,
                self.radius,
                self.min_margin,
            )
        return RouteDecision(
            label,
            True,
            "accepted",
            nearest,
            second,
            margin,
            self.radius,
            self.min_margin,
        )

    def classify(self, sketch: NDArray[np.floating]) -> str | None:
        return self.decision(sketch).label

    def centroid(self, label: str) -> FloatArray:
        try:
            return self._clusters[label].mean.copy()
        except KeyError as exc:
            raise KeyError(f"unknown attractor label: {label}") from exc


@dataclass(slots=True)
class MarkovPDAgent:
    """General finite-state perception/decision/action kernel."""

    state_size: int
    action_size: int | None = None
    prior: float = 0.05
    decision_counts: FloatArray = field(init=False, repr=False)
    observation_mass_by_state: FloatArray = field(init=False, repr=False)
    # Raw observation-event count.  This is deliberately not scaled when
    # probabilistic evidence is weighted or when another agent is merged.
    observations: int = 0
    # Sum of evidence weights.  Unlike ``observations``, merge weights scale it.
    observation_mass: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.state_size, bool)
            or not isinstance(self.state_size, (int, np.integer))
            or self.state_size < 2
        ):
            raise ValueError("state_size must be an integer of at least two")
        self.state_size = int(self.state_size)
        if self.action_size is None:
            self.action_size = self.state_size
        if (
            isinstance(self.action_size, bool)
            or not isinstance(self.action_size, (int, np.integer))
            or self.action_size < 2
        ):
            raise ValueError("action_size must be an integer of at least two")
        self.action_size = int(self.action_size)
        if not math.isfinite(self.prior) or self.prior <= 0.0:
            raise ValueError("prior must be positive and finite")
        self.decision_counts = np.full(
            (self.state_size, self.action_size), self.prior, dtype=np.float64
        )
        self.observation_mass_by_state = np.zeros(self.state_size, dtype=np.float64)

    @staticmethod
    def _index(value: int, *, upper: int, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
        result = int(value)
        if not 0 <= result < upper:
            raise IndexError(f"{name} outside [0, {upper})")
        return result

    def observe(self, source: int, target: int, *, weight: float = 1.0) -> None:
        source_index = self._index(source, upper=self.state_size, name="source")
        target_index = self._index(target, upper=self.action_size, name="target")
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("observation weight must be positive and finite")
        value = float(weight)
        self.decision_counts[source_index, target_index] += value
        self.observation_mass_by_state[source_index] += value
        self.observation_mass += value
        self.observations += 1

    def observe_distribution(
        self,
        source: int,
        weights: NDArray[np.floating],
        *,
        total_weight: float = 1.0,
    ) -> None:
        source_index = self._index(source, upper=self.state_size, name="source")
        distribution = _probability_vector(
            weights, size=self.action_size, name="observation distribution"
        )
        if not math.isfinite(total_weight) or total_weight <= 0.0:
            raise ValueError("total_weight must be positive and finite")
        mass = distribution * float(total_weight)
        self.decision_counts[source_index] += mass
        self.observation_mass_by_state[source_index] += float(total_weight)
        self.observation_mass += float(total_weight)
        self.observations += 1

    def active_states(self, *, min_mass: float = 0.0) -> tuple[int, ...]:
        if not math.isfinite(min_mass) or min_mass < 0.0:
            raise ValueError("min_mass must be finite and non-negative")
        active = (self.observation_mass_by_state > 0.0) & (
            self.observation_mass_by_state >= min_mass
        )
        return tuple(int(index) for index in np.flatnonzero(active))

    @property
    def event_count(self) -> int:
        """Number of raw observation events represented by this agent."""

        return self.observations

    @property
    def evidence_mass(self) -> float:
        """Total (possibly probabilistic and consensus-weighted) evidence mass."""

        return self.observation_mass

    def decision_kernel(self) -> FloatArray:
        return normalize_rows(self.decision_counts)

    def transition_kernel(self) -> FloatArray:
        return self.decision_kernel()

    def predict(self, belief: NDArray[np.floating]) -> FloatArray:
        state = _probability_vector(belief, size=self.state_size, name="belief")
        result = state @ self.decision_kernel()
        return _probability_vector(result, size=self.action_size, name="prediction")

    def merge_from(self, other: "MarkovPDAgent", *, weight: float = 1.0) -> None:
        if not isinstance(other, MarkovPDAgent):
            raise TypeError("other must be a MarkovPDAgent")
        if (self.state_size, self.action_size) != (other.state_size, other.action_size):
            raise ValueError("cannot merge agents with different kernel shapes")
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("merge weight must be positive and finite")
        value = float(weight)
        evidence = other.decision_counts - other.prior
        self.decision_counts += value * evidence
        self.observation_mass_by_state += value * other.observation_mass_by_state
        self.observation_mass += value * other.observation_mass
        self.observations += other.observations


def _fold_feature(feature: FloatArray, size: int) -> FloatArray:
    folded = np.zeros(size, dtype=np.float64)
    for index, value in enumerate(feature):
        slot = index % size
        sign = -1.0 if (index // size) & 1 else 1.0
        folded[slot] += sign * value / math.sqrt(1.0 + index // size)
    return np.tanh(folded)


@dataclass(slots=True)
class MobileMarkovToken:
    """Mobile belief carrying fading memory and compositional confidence."""

    belief: FloatArray
    reservoir: FloatArray
    ledger: RapidityLedger = field(default_factory=RapidityLedger)
    route: list[str] = field(default_factory=list)
    branch_mass: float = 1.0
    fallbacks: int = 0
    reservoir_decay: float = 0.95
    _reservoir_coherence: float = 0.0

    def __post_init__(self) -> None:
        belief = _finite_vector(self.belief, name="belief")
        if belief.size < 2 or np.any(belief < 0.0) or float(belief.sum()) <= 0.0:
            raise ValueError("belief must be a non-negative distribution of size >= 2")
        self.belief = belief / float(belief.sum())
        reservoir = _finite_vector(self.reservoir, name="reservoir")
        if reservoir.size < 8:
            raise ValueError("reservoir must have at least eight dimensions")
        self.reservoir = reservoir.copy()
        if not isinstance(self.ledger, RapidityLedger):
            raise TypeError("ledger must be a RapidityLedger")
        if not math.isfinite(self.branch_mass) or not 0.0 <= self.branch_mass <= 1.0:
            raise ValueError("branch_mass must lie in [0, 1]")
        if (
            isinstance(self.fallbacks, bool)
            or not isinstance(self.fallbacks, int)
            or self.fallbacks < 0
        ):
            raise ValueError("fallbacks must be a non-negative integer")
        if (
            not math.isfinite(self.reservoir_decay)
            or not 0.0 <= self.reservoir_decay < 1.0
        ):
            raise ValueError("reservoir_decay must lie in [0, 1)")
        if (
            not math.isfinite(self._reservoir_coherence)
            or not 0.0 <= self._reservoir_coherence <= 1.0
        ):
            raise ValueError("reservoir coherence must lie in [0, 1]")
        if any(not isinstance(site, str) or not site for site in self.route):
            raise ValueError("route entries must be non-empty strings")

    @classmethod
    def from_state(
        cls, state: int, *, state_size: int, reservoir_size: int = 8
    ) -> "MobileMarkovToken":
        if (
            isinstance(state_size, bool)
            or not isinstance(state_size, (int, np.integer))
            or state_size < 2
        ):
            raise ValueError("state_size must be an integer of at least two")
        if isinstance(state, bool) or not isinstance(state, (int, np.integer)):
            raise TypeError("state must be an integer")
        if not 0 <= int(state) < state_size:
            raise IndexError("state outside token belief")
        if (
            isinstance(reservoir_size, bool)
            or not isinstance(reservoir_size, (int, np.integer))
            or reservoir_size < 8
        ):
            raise ValueError("reservoir_size must be an integer of at least eight")
        belief = np.zeros(state_size, dtype=np.float64)
        belief[int(state)] = 1.0
        return cls(belief=belief, reservoir=np.zeros(reservoir_size, dtype=np.float64))

    @property
    def belief_margin(self) -> float:
        ordered = np.partition(self.belief, -2)
        return float(np.clip(ordered[-1] - ordered[-2], 0.0, 1.0))

    @property
    def reservoir_signal(self) -> float:
        energy = math.tanh(
            float(np.linalg.norm(self.reservoir)) / math.sqrt(self.reservoir.size)
        )
        return float(np.clip(math.sqrt(energy * self._reservoir_coherence), 0.0, 1.0))

    @property
    def rapidity_signal(self) -> float:
        return float(np.clip(self.ledger.value, 0.0, 1.0))

    @property
    def execution_confidence(self) -> float:
        # A zero in any factor closes the execution gate.  Reservoir memory and
        # accumulated rapidity therefore cannot be removed without changing the
        # executable behavior of the token.
        product = self.belief_margin * self.reservoir_signal * self.rapidity_signal
        return float(np.clip(product ** (1.0 / 3.0), 0.0, 1.0))

    def update(
        self,
        distribution: NDArray[np.floating],
        sketch: NDArray[np.floating],
        route: str,
        *,
        coupling: float | None = None,
    ) -> float:
        if not isinstance(route, str) or not route:
            raise ValueError("route must be a non-empty string")
        self.belief = _probability_vector(
            distribution, size=self.belief.size, name="distribution"
        )
        feature = _finite_vector(sketch, name="sketch")
        injected = _fold_feature(feature, self.reservoir.size)
        previous = self.reservoir.copy()
        self.reservoir = (
            self.reservoir_decay * previous + (1.0 - self.reservoir_decay) * injected
        )
        reservoir_norm = float(np.linalg.norm(self.reservoir))
        feature_norm = float(np.linalg.norm(injected))
        if reservoir_norm > 0.0 and feature_norm > 0.0:
            cosine = float(
                np.dot(self.reservoir, injected) / (reservoir_norm * feature_norm)
            )
            self._reservoir_coherence = float(np.clip(0.5 + 0.5 * cosine, 0.0, 1.0))
        else:
            self._reservoir_coherence = 0.0

        if coupling is None:
            coupling_value = self.belief_margin * (0.25 + 0.75 * self.reservoir_signal)
        else:
            if not math.isfinite(coupling) or not -1.0 < coupling < 1.0:
                raise ValueError("coupling must be finite and lie in (-1, 1)")
            coupling_value = float(coupling) * (0.25 + 0.75 * self.reservoir_signal)
        self.ledger.add(float(np.clip(coupling_value, -0.999999, 0.999999)))
        self.route.append(route)
        return self.execution_confidence

    def gate(
        self,
        *,
        min_confidence: float,
        min_branch_mass: float = 0.0,
        record_fallback: bool = True,
    ) -> bool:
        if not math.isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must lie in [0, 1]")
        if not math.isfinite(min_branch_mass) or not 0.0 <= min_branch_mass <= 1.0:
            raise ValueError("min_branch_mass must lie in [0, 1]")
        accepted = (
            self.execution_confidence >= min_confidence
            and self.branch_mass >= min_branch_mass
        )
        if not accepted and record_fallback:
            self.fallbacks += 1
        return accepted

    def mark_fallback(self) -> None:
        self.fallbacks += 1

    def spawn(self, *, top_k: int) -> list["MobileMarkovToken"]:
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, (int, np.integer))
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer")
        indices = np.argsort(self.belief, kind="stable")[::-1][:top_k]
        children: list[MobileMarkovToken] = []
        for index in indices:
            belief = np.zeros_like(self.belief)
            belief[index] = 1.0
            children.append(
                MobileMarkovToken(
                    belief=belief,
                    reservoir=self.reservoir.copy(),
                    ledger=RapidityLedger(self.ledger.xi, self.ledger.limit),
                    route=list(self.route),
                    branch_mass=self.branch_mass * float(self.belief[index]),
                    fallbacks=self.fallbacks,
                    reservoir_decay=self.reservoir_decay,
                    _reservoir_coherence=self._reservoir_coherence,
                )
            )
        return children
