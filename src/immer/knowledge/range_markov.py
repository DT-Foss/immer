"""Persistent Markov prediction over settled logical Streamer operations."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
import threading
from typing import Any
import zlib

from .access_trace import AccessLeaf, AccessOperation, canonical_tags

try:
    import fcntl
except ImportError:  # pragma: no cover - production targets are POSIX.
    fcntl = None  # type: ignore[assignment]


RANGE_MARKOV_STATE_SCHEMA = "immer.range-markov-state/v2"
V1_RANGE_MARKOV_STATE_SCHEMA = "immer.range-markov-state/v1"
RANGE_MARKOV_PREDICTION_SCHEMA = "immer.range-markov-prediction/v1"
RANGE_MARKOV_METRICS_SCHEMA = "immer.range-markov-metrics/v3"
RANGE_MARKOV_BEAM_STEP_SCHEMA = "immer.range-markov-beam-step/v2"
RANGE_MARKOV_BEAM_PLAN_SCHEMA = "immer.range-markov-beam-plan/v2"

_STATE_PREFIX = b"IMRM\x02"
_V1_STATE_PREFIX = b"IMRM\x01"
_MAX_STATE_BYTES = 16 * 1024 * 1024
_MAX_JSON_BYTES = 64 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")
_SEMANTIC_TAGS = frozenset({"layer", "phase", "read_kind", "tensor"})
_MAX_DISTANCE_AGENTS = 8


class RangeMarkovError(RuntimeError):
    """Range intelligence state or input is invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or bool(set(value) - _HEX)
    ):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _uint(value: object, *, field: str, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < int(positive)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be a {qualifier} integer")
    return value


def _finite(
    value: object,
    *,
    field: str,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be finite")
    result = float(value)
    if (
        not math.isfinite(result)
        or (lower is not None and result < lower)
        or (upper is not None and result > upper)
    ):
        raise ValueError(f"{field} is outside its finite bound")
    return result


def _float_record(value: float) -> str:
    return float(value).hex()


def _record_float(value: object, *, field: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a hexadecimal float")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} is not a hexadecimal float") from exc
    return _finite(result, field=field)


@dataclass(frozen=True, slots=True)
class AccessState:
    """One grouped logical demand operation used as a Markov state."""

    operation: str
    leaves: tuple[AccessLeaf, ...]
    tags: tuple[tuple[str, Any], ...] = ()
    key: str = ""

    def __post_init__(self) -> None:
        if self.operation not in {"raw_bytes", "raw_bytes_many"}:
            raise ValueError("access state operation is invalid")
        leaves = tuple(self.leaves)
        if not leaves or any(not isinstance(leaf, AccessLeaf) for leaf in leaves):
            raise ValueError("access state requires exact leaves")
        if len(set(leaves)) != len(leaves):
            raise ValueError("access state leaves must be unique")
        tags = canonical_tags(dict(self.tags))
        identity = {
            "leaves": [leaf.to_document() for leaf in leaves],
            "operation": self.operation,
            "tags": dict(tags),
        }
        expected = _digest(identity)
        if self.key and self.key != expected:
            raise ValueError("access state key differs from its operation")
        object.__setattr__(self, "leaves", leaves)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "key", expected)

    @classmethod
    def from_operation(cls, operation: AccessOperation) -> "AccessState":
        if not isinstance(operation, AccessOperation):
            raise TypeError("operation must be AccessOperation")
        leaves = tuple(dict.fromkeys(operation.leaves))
        tags = tuple(
            (key, value)
            for key, value in operation.tags
            if key in _SEMANTIC_TAGS
        )
        return cls(operation.operation, leaves, tags)

    @property
    def total_bytes(self) -> int:
        return sum(leaf.length for leaf in self.leaves)

    def to_record(self) -> dict[str, object]:
        return {
            "key": self.key,
            "leaves": [leaf.to_document() for leaf in self.leaves],
            "operation": self.operation,
            "tags": dict(self.tags),
        }

    @classmethod
    def from_record(cls, value: object) -> "AccessState":
        if not isinstance(value, Mapping) or set(value) != {
            "key",
            "leaves",
            "operation",
            "tags",
        }:
            raise ValueError("access state record is invalid")
        leaves = value["leaves"]
        if not isinstance(leaves, list):
            raise ValueError("access state leaves must be a list")
        return cls(
            operation=value["operation"],
            leaves=tuple(AccessLeaf.from_document(row) for row in leaves),
            tags=canonical_tags(value["tags"]),
            key=value["key"],
        )


@dataclass(frozen=True, slots=True)
class RangeNode:
    state: AccessState
    visits: int
    last_seen: int

    def __post_init__(self) -> None:
        if not isinstance(self.state, AccessState):
            raise TypeError("range node requires AccessState")
        _uint(self.visits, field="node visits", positive=True)
        _uint(self.last_seen, field="node last_seen", positive=True)

    def ricci_value(self, *, clock: int, alpha: float) -> float:
        return self.visits * math.exp(-alpha * max(0, clock - self.last_seen))

    def to_record(self) -> dict[str, object]:
        return {
            "last_seen": self.last_seen,
            "state": self.state.to_record(),
            "visits": self.visits,
        }

    @classmethod
    def from_record(cls, value: object) -> "RangeNode":
        if not isinstance(value, Mapping) or set(value) != {
            "last_seen",
            "state",
            "visits",
        }:
            raise ValueError("range node record is invalid")
        return cls(
            state=AccessState.from_record(value["state"]),
            visits=value["visits"],
            last_seen=value["last_seen"],
        )


@dataclass(frozen=True, slots=True)
class RangeContext:
    history: tuple[str, ...]
    targets: tuple[tuple[str, int], ...]
    visits: int
    last_seen: int

    def __post_init__(self) -> None:
        history = tuple(self.history)
        targets = tuple(self.targets)
        if (
            not 1 <= len(history) <= 2
            or any(len(key) != 64 or bool(set(key) - _HEX) for key in history)
            or not targets
            or tuple(sorted(targets)) != targets
            or len({key for key, _count in targets}) != len(targets)
            or any(
                len(key) != 64
                or bool(set(key) - _HEX)
                or _uint(count, field="target count", positive=True) < 1
                for key, count in targets
            )
        ):
            raise ValueError("range context topology is invalid")
        _uint(self.visits, field="context visits", positive=True)
        _uint(self.last_seen, field="context last_seen", positive=True)
        if self.visits < sum(count for _key, count in targets):
            raise ValueError("range context visits trail retained target counts")
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "targets", targets)

    def distribution(self) -> dict[str, float]:
        return {key: count / self.visits for key, count in self.targets}

    def ricci_value(self, *, clock: int, alpha: float) -> float:
        return self.visits * math.exp(-alpha * max(0, clock - self.last_seen))

    def to_record(self) -> dict[str, object]:
        return {
            "history": list(self.history),
            "last_seen": self.last_seen,
            "targets": [list(row) for row in self.targets],
            "visits": self.visits,
        }

    @classmethod
    def from_record(cls, value: object) -> "RangeContext":
        if not isinstance(value, Mapping) or set(value) != {
            "history",
            "last_seen",
            "targets",
            "visits",
        }:
            raise ValueError("range context record is invalid")
        try:
            return cls(
                history=tuple(value["history"]),
                targets=tuple((row[0], row[1]) for row in value["targets"]),
                visits=value["visits"],
                last_seen=value["last_seen"],
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError("range context values are invalid") from exc


@dataclass(frozen=True, slots=True)
class RangeMarkovState:
    repo_id: str | None = None
    revision: str | None = None
    inventory_fingerprint: str | None = None
    clock: int = 0
    observations: int = 0
    nodes: tuple[RangeNode, ...] = ()
    contexts: tuple[RangeContext, ...] = ()
    history: tuple[str, ...] = ()
    expert_rapidities: tuple[float, float] = (0.0, 0.0)
    expert_observations: tuple[int, int] = (0, 0)
    expert_hits: tuple[int, int] = (0, 0)
    surprise_mean: float = 0.0
    surprise_deviation: float = 1.0
    surprise_cusum: float = 0.0
    regime_generation: int = 0
    node_evictions: int = 0
    context_evictions: int = 0
    distance_rapidities: tuple[float, ...] = (0.0,) * _MAX_DISTANCE_AGENTS
    distance_observations: tuple[int, ...] = (0,) * _MAX_DISTANCE_AGENTS
    distance_hits: tuple[int, ...] = (0,) * _MAX_DISTANCE_AGENTS
    distance_hinted_bytes: tuple[int, ...] = (0,) * _MAX_DISTANCE_AGENTS
    distance_useful_bytes: tuple[int, ...] = (0,) * _MAX_DISTANCE_AGENTS
    hint_feedback_count: int = 0
    hint_utility_ema: float = 1.0

    def __post_init__(self) -> None:
        identities = (self.repo_id, self.revision, self.inventory_fingerprint)
        if any(value is not None for value in identities):
            if not all(isinstance(value, str) and value for value in identities):
                raise ValueError("range Markov source identity is incomplete")
            _sha256(self.inventory_fingerprint, field="inventory_fingerprint")
        _uint(self.clock, field="clock")
        _uint(self.observations, field="observations")
        nodes = tuple(self.nodes)
        contexts = tuple(self.contexts)
        if (
            tuple(sorted(nodes, key=lambda row: row.state.key)) != nodes
            or len({row.state.key for row in nodes}) != len(nodes)
            or any(row.last_seen > self.clock for row in nodes)
        ):
            raise ValueError("range Markov nodes are invalid")
        node_keys = {row.state.key for row in nodes}
        if (
            tuple(sorted(contexts, key=lambda row: row.history)) != contexts
            or len({row.history for row in contexts}) != len(contexts)
            or any(
                row.last_seen > self.clock
                or any(key not in node_keys for key in row.history)
                or any(key not in node_keys for key, _count in row.targets)
                for row in contexts
            )
        ):
            raise ValueError("range Markov contexts are invalid")
        history = tuple(self.history)
        if len(history) > 2 or any(key not in node_keys for key in history):
            raise ValueError("range Markov history is invalid")
        rapidities = tuple(float(value) for value in self.expert_rapidities)
        expert_observations = tuple(self.expert_observations)
        expert_hits = tuple(self.expert_hits)
        if (
            len(rapidities) != 2
            or any(not math.isfinite(value) or abs(value) > 20 for value in rapidities)
            or len(expert_observations) != 2
            or len(expert_hits) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (*expert_observations, *expert_hits)
            )
            or any(hit > seen for hit, seen in zip(expert_hits, expert_observations))
        ):
            raise ValueError("range Markov expert state is invalid")
        for value, field in (
            (self.surprise_mean, "surprise_mean"),
            (self.surprise_deviation, "surprise_deviation"),
            (self.surprise_cusum, "surprise_cusum"),
        ):
            _finite(value, field=field, lower=0.0)
        for value, field in (
            (self.regime_generation, "regime_generation"),
            (self.node_evictions, "node_evictions"),
            (self.context_evictions, "context_evictions"),
        ):
            _uint(value, field=field)
        distance_rapidities = tuple(float(value) for value in self.distance_rapidities)
        distance_observations = tuple(self.distance_observations)
        distance_hits = tuple(self.distance_hits)
        distance_hinted_bytes = tuple(self.distance_hinted_bytes)
        distance_useful_bytes = tuple(self.distance_useful_bytes)
        if (
            any(
                len(row) != _MAX_DISTANCE_AGENTS
                for row in (
                    distance_rapidities,
                    distance_observations,
                    distance_hits,
                    distance_hinted_bytes,
                    distance_useful_bytes,
                )
            )
            or any(
                not math.isfinite(value) or abs(value) > 20.0
                for value in distance_rapidities
            )
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for row in (
                    distance_observations,
                    distance_hits,
                    distance_hinted_bytes,
                    distance_useful_bytes,
                )
                for value in row
            )
            or any(
                hit > seen
                for hit, seen in zip(
                    distance_hits,
                    distance_observations,
                    strict=True,
                )
            )
            or any(
                useful > hinted
                for useful, hinted in zip(
                    distance_useful_bytes,
                    distance_hinted_bytes,
                    strict=True,
                )
            )
        ):
            raise ValueError("range Markov distance-agent state is invalid")
        _uint(self.hint_feedback_count, field="hint_feedback_count")
        _finite(
            self.hint_utility_ema,
            field="hint_utility_ema",
            lower=0.0,
            upper=1.0,
        )
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "contexts", contexts)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "expert_rapidities", rapidities)
        object.__setattr__(self, "expert_observations", expert_observations)
        object.__setattr__(self, "expert_hits", expert_hits)
        object.__setattr__(self, "distance_rapidities", distance_rapidities)
        object.__setattr__(self, "distance_observations", distance_observations)
        object.__setattr__(self, "distance_hits", distance_hits)
        object.__setattr__(self, "distance_hinted_bytes", distance_hinted_bytes)
        object.__setattr__(self, "distance_useful_bytes", distance_useful_bytes)

    @property
    def source_identity(self) -> tuple[str, str, str] | None:
        if self.repo_id is None:
            return None
        return self.repo_id, self.revision, self.inventory_fingerprint

    def to_record(self) -> dict[str, object]:
        return {
            "clock": self.clock,
            "context_evictions": self.context_evictions,
            "contexts": [row.to_record() for row in self.contexts],
            "distance_hinted_bytes": list(self.distance_hinted_bytes),
            "distance_hits": list(self.distance_hits),
            "distance_observations": list(self.distance_observations),
            "distance_rapidities": [
                _float_record(value) for value in self.distance_rapidities
            ],
            "distance_useful_bytes": list(self.distance_useful_bytes),
            "expert_hits": list(self.expert_hits),
            "expert_observations": list(self.expert_observations),
            "expert_rapidities": [
                _float_record(value) for value in self.expert_rapidities
            ],
            "history": list(self.history),
            "hint_feedback_count": self.hint_feedback_count,
            "hint_utility_ema": _float_record(self.hint_utility_ema),
            "inventory_fingerprint": self.inventory_fingerprint,
            "node_evictions": self.node_evictions,
            "nodes": [row.to_record() for row in self.nodes],
            "observations": self.observations,
            "regime_generation": self.regime_generation,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "schema": RANGE_MARKOV_STATE_SCHEMA,
            "surprise_cusum": _float_record(self.surprise_cusum),
            "surprise_deviation": _float_record(self.surprise_deviation),
            "surprise_mean": _float_record(self.surprise_mean),
        }

    def to_bytes(self) -> bytes:
        body = self.to_record()
        envelope = _canonical({"body": body, "body_sha256": _digest(body)})
        encoded = _STATE_PREFIX + zlib.compress(envelope, level=9)
        if len(encoded) > _MAX_STATE_BYTES:
            raise RangeMarkovError("range Markov state exceeds its byte bound")
        return encoded

    @classmethod
    def from_record(cls, value: object) -> "RangeMarkovState":
        fields = {
            "clock",
            "context_evictions",
            "contexts",
            "distance_hinted_bytes",
            "distance_hits",
            "distance_observations",
            "distance_rapidities",
            "distance_useful_bytes",
            "expert_hits",
            "expert_observations",
            "expert_rapidities",
            "history",
            "hint_feedback_count",
            "hint_utility_ema",
            "inventory_fingerprint",
            "node_evictions",
            "nodes",
            "observations",
            "regime_generation",
            "repo_id",
            "revision",
            "schema",
            "surprise_cusum",
            "surprise_deviation",
            "surprise_mean",
        }
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("range Markov state record is invalid")
        if value["schema"] != RANGE_MARKOV_STATE_SCHEMA:
            raise ValueError("range Markov state schema is invalid")
        return cls(
            repo_id=value["repo_id"],
            revision=value["revision"],
            inventory_fingerprint=value["inventory_fingerprint"],
            clock=value["clock"],
            observations=value["observations"],
            nodes=tuple(RangeNode.from_record(row) for row in value["nodes"]),
            contexts=tuple(
                RangeContext.from_record(row) for row in value["contexts"]
            ),
            distance_rapidities=tuple(
                _record_float(row, field="distance rapidity")
                for row in value["distance_rapidities"]
            ),
            distance_observations=tuple(value["distance_observations"]),
            distance_hits=tuple(value["distance_hits"]),
            distance_hinted_bytes=tuple(value["distance_hinted_bytes"]),
            distance_useful_bytes=tuple(value["distance_useful_bytes"]),
            hint_feedback_count=value["hint_feedback_count"],
            hint_utility_ema=_record_float(
                value["hint_utility_ema"], field="hint_utility_ema"
            ),
            history=tuple(value["history"]),
            expert_rapidities=tuple(
                _record_float(row, field="expert rapidity")
                for row in value["expert_rapidities"]
            ),
            expert_observations=tuple(value["expert_observations"]),
            expert_hits=tuple(value["expert_hits"]),
            surprise_mean=_record_float(
                value["surprise_mean"], field="surprise_mean"
            ),
            surprise_deviation=_record_float(
                value["surprise_deviation"], field="surprise_deviation"
            ),
            surprise_cusum=_record_float(
                value["surprise_cusum"], field="surprise_cusum"
            ),
            regime_generation=value["regime_generation"],
            node_evictions=value["node_evictions"],
            context_evictions=value["context_evictions"],
        )

    @classmethod
    def _from_v1_record(cls, value: object) -> "RangeMarkovState":
        if not isinstance(value, Mapping) or value.get("schema") != (
            V1_RANGE_MARKOV_STATE_SCHEMA
        ):
            raise ValueError("v1 range Markov state schema is invalid")
        migrated = dict(value)
        migrated["schema"] = RANGE_MARKOV_STATE_SCHEMA
        migrated["distance_rapidities"] = [
            _float_record(0.0)
        ] * _MAX_DISTANCE_AGENTS
        migrated["distance_observations"] = [0] * _MAX_DISTANCE_AGENTS
        migrated["distance_hits"] = [0] * _MAX_DISTANCE_AGENTS
        migrated["distance_hinted_bytes"] = [0] * _MAX_DISTANCE_AGENTS
        migrated["distance_useful_bytes"] = [0] * _MAX_DISTANCE_AGENTS
        migrated["hint_feedback_count"] = 0
        migrated["hint_utility_ema"] = _float_record(1.0)
        return cls.from_record(migrated)

    @classmethod
    def from_bytes(cls, value: bytes) -> "RangeMarkovState":
        if not isinstance(value, bytes) or not 5 < len(value) <= _MAX_STATE_BYTES:
            raise RangeMarkovError("range Markov state header is invalid")
        if value.startswith(_STATE_PREFIX):
            version = 2
            payload = value[len(_STATE_PREFIX) :]
        elif value.startswith(_V1_STATE_PREFIX):
            version = 1
            payload = value[len(_V1_STATE_PREFIX) :]
        else:
            raise RangeMarkovError("range Markov state header is invalid")
        try:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(payload, _MAX_JSON_BYTES + 1)
            raw += decoder.flush()
            if (
                len(raw) > _MAX_JSON_BYTES
                or not decoder.eof
                or decoder.unused_data
                or decoder.unconsumed_tail
            ):
                raise ValueError("range Markov state body exceeds its bound")
            envelope = json.loads(raw)
            if not isinstance(envelope, Mapping) or set(envelope) != {
                "body",
                "body_sha256",
            }:
                raise ValueError("range Markov envelope is invalid")
            if envelope["body_sha256"] != _digest(envelope["body"]):
                raise ValueError("range Markov checksum differs")
            if version == 1:
                return cls._from_v1_record(envelope["body"])
            return cls.from_record(envelope["body"])
        except (TypeError, ValueError, zlib.error, json.JSONDecodeError) as exc:
            raise RangeMarkovError("range Markov state is corrupt") from exc


@dataclass(frozen=True, slots=True)
class RangePrediction:
    state: AccessState
    confidence: float
    support: int
    total: int
    expert_weights: tuple[tuple[str, float], ...]
    ricci_value: float
    expected_reuse_distance: float

    def __post_init__(self) -> None:
        if not isinstance(self.state, AccessState):
            raise TypeError("range prediction requires AccessState")
        _finite(self.confidence, field="prediction confidence", lower=0.0, upper=1.0)
        _uint(self.support, field="prediction support", positive=True)
        _uint(self.total, field="prediction total", positive=True)
        if self.support > self.total:
            raise ValueError("prediction support exceeds total")
        weights = tuple(self.expert_weights)
        if (
            tuple(name for name, _weight in weights) != ("order-1", "order-2")
            or not math.isclose(sum(weight for _name, weight in weights), 1.0)
        ):
            raise ValueError("range prediction expert weights are invalid")
        _finite(self.ricci_value, field="prediction Ricci value", lower=0.0)
        _finite(
            self.expected_reuse_distance,
            field="expected reuse distance",
            lower=0.0,
        )
        object.__setattr__(self, "expert_weights", weights)

    def to_dict(self) -> dict[str, object]:
        return {
            "confidence": self.confidence,
            "expected_reuse_distance": self.expected_reuse_distance,
            "expert_weights": dict(self.expert_weights),
            "ricci_value": self.ricci_value,
            "schema": RANGE_MARKOV_PREDICTION_SCHEMA,
            "state": self.state.to_record(),
            "support": self.support,
            "total": self.total,
        }


@dataclass(frozen=True, slots=True)
class RangeBeamStep:
    distance: int
    state: AccessState
    transition_probability: float
    path_probability: float
    path_min_transition_probability: float
    support: int
    total: int
    path_min_support: int
    ricci_value: float
    expected_reuse_distance: float
    distance_weight: float
    score: float
    path: tuple[str, ...]

    def __post_init__(self) -> None:
        _uint(self.distance, field="beam distance", positive=True)
        if not isinstance(self.state, AccessState):
            raise TypeError("beam step requires AccessState")
        _finite(
            self.transition_probability,
            field="transition_probability",
            lower=0.0,
            upper=1.0,
        )
        _finite(
            self.path_probability,
            field="path_probability",
            lower=0.0,
            upper=1.0,
        )
        if self.path_probability > self.transition_probability + 1e-15:
            raise ValueError("beam path probability exceeds its final transition")
        _finite(
            self.path_min_transition_probability,
            field="path_min_transition_probability",
            lower=0.0,
            upper=1.0,
        )
        if self.path_min_transition_probability > self.transition_probability:
            raise ValueError("beam path minimum exceeds final transition")
        _uint(self.support, field="beam support", positive=True)
        _uint(self.total, field="beam total", positive=True)
        if self.support > self.total:
            raise ValueError("beam support exceeds total")
        _uint(self.path_min_support, field="beam path_min_support", positive=True)
        if self.path_min_support > self.support:
            raise ValueError("beam path support minimum exceeds final support")
        _finite(self.ricci_value, field="beam Ricci value", lower=0.0)
        _finite(
            self.expected_reuse_distance,
            field="beam expected reuse distance",
            lower=0.0,
        )
        _finite(self.distance_weight, field="beam distance weight", lower=0.0)
        _finite(self.score, field="beam score", lower=0.0)
        path = tuple(self.path)
        if (
            len(path) != self.distance
            or path[-1] != self.state.key
            or any(len(key) != 64 or bool(set(key) - _HEX) for key in path)
        ):
            raise ValueError("beam path is invalid")
        object.__setattr__(self, "path", path)

    def to_dict(self) -> dict[str, object]:
        return {
            "distance": self.distance,
            "distance_weight": self.distance_weight,
            "expected_reuse_distance": self.expected_reuse_distance,
            "path": list(self.path),
            "path_probability": self.path_probability,
            "path_min_support": self.path_min_support,
            "path_min_transition_probability": (
                self.path_min_transition_probability
            ),
            "ricci_value": self.ricci_value,
            "schema": RANGE_MARKOV_BEAM_STEP_SCHEMA,
            "score": self.score,
            "state": self.state.to_record(),
            "support": self.support,
            "total": self.total,
            "transition_probability": self.transition_probability,
        }


@dataclass(frozen=True, slots=True)
class RangeHint:
    leaf: AccessLeaf
    source_state_key: str
    distance: int
    score: float
    path_probability: float
    original_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.leaf, AccessLeaf):
            raise TypeError("range hint requires AccessLeaf")
        _sha256(self.source_state_key, field="source_state_key")
        _uint(self.distance, field="hint distance", positive=True)
        _finite(self.score, field="hint score", lower=0.0)
        _finite(
            self.path_probability,
            field="hint path_probability",
            lower=0.0,
            upper=1.0,
        )
        _uint(self.original_length, field="hint original_length", positive=True)
        if self.leaf.length > self.original_length:
            raise ValueError("hint length exceeds original leaf")

    def to_dict(self) -> dict[str, object]:
        return {
            "distance": self.distance,
            "leaf": self.leaf.to_document(),
            "original_length": self.original_length,
            "path_probability": self.path_probability,
            "score": self.score,
            "source_state_key": self.source_state_key,
        }


@dataclass(frozen=True, slots=True)
class _PendingHint:
    created_operation: int
    due_operation: int
    distance: int
    leaf: AccessLeaf
    original_length: int
    covered_intervals: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        _uint(self.created_operation, field="hint created_operation", positive=True)
        _uint(self.due_operation, field="hint due_operation", positive=True)
        _uint(self.distance, field="hint feedback distance", positive=True)
        if self.due_operation != self.created_operation + self.distance:
            raise ValueError("hint due operation differs from its distance")
        if not isinstance(self.leaf, AccessLeaf):
            raise TypeError("pending hint requires AccessLeaf")
        _uint(self.original_length, field="pending original_length", positive=True)
        if self.leaf.length > self.original_length:
            raise ValueError("pending hint exceeds its original range")
        intervals = tuple(self.covered_intervals)
        if (
            tuple(sorted(intervals)) != intervals
            or any(
                not isinstance(start, int)
                or not isinstance(stop, int)
                or not self.leaf.offset <= start < stop
                or stop > self.leaf.offset + self.leaf.length
                for start, stop in intervals
            )
            or any(
                right[0] <= left[1]
                for left, right in zip(intervals, intervals[1:], strict=False)
            )
        ):
            raise ValueError("pending hint coverage intervals are invalid")
        object.__setattr__(self, "covered_intervals", intervals)

    @property
    def covered_bytes(self) -> int:
        return sum(stop - start for start, stop in self.covered_intervals)

    @property
    def original_key(self) -> tuple[str, int, int]:
        return self.leaf.shard, self.leaf.offset, self.original_length

    def observe_demand(self, demand: Sequence[AccessLeaf]) -> "_PendingHint":
        intervals = [*self.covered_intervals, *_range_overlap_intervals(self.leaf, demand)]
        if not intervals:
            return self
        intervals.sort()
        merged = []
        for start, stop in intervals:
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
            else:
                merged.append((start, stop))
        return replace(self, covered_intervals=tuple(merged))


def _range_overlap_intervals(
    hint: AccessLeaf,
    demand: Sequence[AccessLeaf],
) -> tuple[tuple[int, int], ...]:
    intervals = []
    hint_start = hint.offset
    hint_stop = hint.offset + hint.length
    for leaf in demand:
        if leaf.shard != hint.shard:
            continue
        start = max(hint_start, leaf.offset)
        stop = min(hint_stop, leaf.offset + leaf.length)
        if stop > start:
            intervals.append((start, stop))
    if not intervals:
        return ()
    intervals.sort()
    merged = []
    for start, stop in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return tuple(merged)


def _range_overlap_bytes(hint: AccessLeaf, demand: Sequence[AccessLeaf]) -> int:
    return sum(
        stop - start for start, stop in _range_overlap_intervals(hint, demand)
    )


@dataclass(frozen=True, slots=True)
class RangeBeamPlan:
    horizon: int
    beam_width: int
    candidates_considered: int
    steps: tuple[RangeBeamStep, ...]
    hints: tuple[RangeHint, ...]
    distance_weights: tuple[tuple[int, float], ...]
    configured_hint_bytes: int
    effective_hint_budget_bytes: int
    duplicate_leaves_avoided: int
    cooldown_leaves_skipped: int
    truncated_by_bytes: bool
    truncated_by_leaves: bool

    def __post_init__(self) -> None:
        _uint(self.horizon, field="beam horizon", positive=True)
        _uint(self.beam_width, field="beam width", positive=True)
        _uint(
            self.candidates_considered,
            field="beam candidates_considered",
        )
        steps = tuple(self.steps)
        hints = tuple(self.hints)
        distance_weights = tuple(self.distance_weights)
        if (
            any(not isinstance(row, RangeBeamStep) for row in steps)
            or any(row.distance > self.horizon for row in steps)
            or any(not isinstance(row, RangeHint) for row in hints)
        ):
            raise ValueError("range beam plan contents are invalid")
        if (
            tuple(distance for distance, _weight in distance_weights)
            != tuple(range(1, self.horizon + 1))
            or any(
                not math.isfinite(weight) or not 0.0 < weight <= 1.0
                for _distance, weight in distance_weights
            )
            or not math.isclose(
                sum(weight for _distance, weight in distance_weights),
                1.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("beam distance weights are invalid")
        _uint(
            self.configured_hint_bytes,
            field="configured_hint_bytes",
            positive=True,
        )
        _uint(
            self.effective_hint_budget_bytes,
            field="effective_hint_budget_bytes",
            positive=True,
        )
        if self.effective_hint_budget_bytes > self.configured_hint_bytes:
            raise ValueError("effective hint budget exceeds configured maximum")
        _uint(
            self.duplicate_leaves_avoided,
            field="duplicate_leaves_avoided",
        )
        _uint(
            self.cooldown_leaves_skipped,
            field="cooldown_leaves_skipped",
        )
        if not isinstance(self.truncated_by_bytes, bool) or not isinstance(
            self.truncated_by_leaves,
            bool,
        ):
            raise TypeError("beam truncation flags must be boolean")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "hints", hints)
        object.__setattr__(self, "distance_weights", distance_weights)

    @property
    def hint_bytes(self) -> int:
        return sum(row.leaf.length for row in self.hints)

    @property
    def predicted_states(self) -> int:
        return len({row.state.key for row in self.steps})

    @property
    def path_probability(self) -> float:
        if not self.steps:
            return 0.0
        return max(row.path_probability for row in self.steps)

    @property
    def score(self) -> float:
        return sum(row.score for row in self.steps)

    def to_dict(self) -> dict[str, object]:
        return {
            "beam_width": self.beam_width,
            "candidates_considered": self.candidates_considered,
            "cooldown_leaves_skipped": self.cooldown_leaves_skipped,
            "duplicate_leaves_avoided": self.duplicate_leaves_avoided,
            "configured_hint_bytes": self.configured_hint_bytes,
            "distance_weights": dict(self.distance_weights),
            "effective_hint_budget_bytes": self.effective_hint_budget_bytes,
            "hint_bytes": self.hint_bytes,
            "hints": [row.to_dict() for row in self.hints],
            "horizon": self.horizon,
            "path_probability": self.path_probability,
            "predicted_states": self.predicted_states,
            "schema": RANGE_MARKOV_BEAM_PLAN_SCHEMA,
            "score": self.score,
            "steps": [row.to_dict() for row in self.steps],
            "truncated_by_bytes": self.truncated_by_bytes,
            "truncated_by_leaves": self.truncated_by_leaves,
        }


class MarkovRangePrefetcher:
    """Two-agent Markov predictor and bounded local OS-page-cache warmer."""

    FIXED_SHARE = 0.05
    RAPIDITY_DECAY = 0.995
    LEARNING_RATE = 0.25
    SURPRISE_RATE = 0.05
    CUSUM_DECAY = 0.90
    CUSUM_DRIFT = 0.50
    CUSUM_THRESHOLD = 8.0
    REGIME_WARMUP = 16
    REGIME_SHRINK = 0.25
    RICCI_ALPHA = 0.001
    DISTANCE_FIXED_SHARE = 0.05
    DISTANCE_RAPIDITY_DECAY = 0.995
    DISTANCE_LEARNING_RATE = 0.35
    HINT_UTILITY_RATE = 0.10
    MIN_BUDGET_FRACTION = 0.125

    def __init__(
        self,
        state_path: str | Path | None,
        *,
        prefetch_range: Callable[[str, int, int], bool],
        min_support: int = 2,
        min_confidence: float = 0.65,
        max_prefetch_bytes: int = 64 * 1024**2,
        max_prefetch_leaves: int = 4,
        max_operation_leaves: int = 64,
        max_nodes: int = 2048,
        max_contexts: int = 4096,
        max_targets_per_context: int = 16,
        beam_horizon: int = 3,
        beam_width: int = 4,
        hint_cooldown_operations: int = 2,
        flush_interval: int = 1024,
    ) -> None:
        if state_path is not None and not isinstance(state_path, (str, Path)):
            raise TypeError("state_path must be a local path or None")
        if not callable(prefetch_range):
            raise TypeError("prefetch_range must be callable")
        for value, field in (
            (min_support, "min_support"),
            (max_prefetch_bytes, "max_prefetch_bytes"),
            (max_prefetch_leaves, "max_prefetch_leaves"),
            (max_operation_leaves, "max_operation_leaves"),
            (max_nodes, "max_nodes"),
            (max_contexts, "max_contexts"),
            (max_targets_per_context, "max_targets_per_context"),
            (beam_horizon, "beam_horizon"),
            (beam_width, "beam_width"),
            (hint_cooldown_operations, "hint_cooldown_operations"),
            (flush_interval, "flush_interval"),
        ):
            _uint(value, field=field, positive=True)
        _finite(min_confidence, field="min_confidence", lower=0.0, upper=1.0)
        if max_nodes < 4 or max_contexts < 4:
            raise ValueError("range Markov capacities are too small")
        if beam_horizon > 8 or beam_width > 16:
            raise ValueError("range Markov beam exceeds its bounded topology")
        self.state_path = (
            None if state_path is None else Path(state_path).expanduser().absolute()
        )
        self.prefetch_range = prefetch_range
        self.min_support = min_support
        self.min_confidence = float(min_confidence)
        self.max_prefetch_bytes = max_prefetch_bytes
        self.max_prefetch_leaves = max_prefetch_leaves
        self.max_operation_leaves = max_operation_leaves
        self.max_nodes = max_nodes
        self.max_contexts = max_contexts
        self.max_targets_per_context = max_targets_per_context
        self.beam_horizon = beam_horizon
        self.beam_width = beam_width
        self.hint_cooldown_operations = hint_cooldown_operations
        self.flush_interval = flush_interval
        self._lock = threading.RLock()
        self._closed = False
        self._state_lock_descriptor: int | None = None
        self._dirty_operations = 0
        self._last_operation_sequence = 0
        self._pending_prediction_key: str | None = None
        self._last_prediction: RangePrediction | None = None
        self._last_beam_plan: RangeBeamPlan | None = None
        self._pending_forecasts: dict[
            int,
            dict[int, frozenset[str]],
        ] = {}
        self._recent_hint_operations: dict[tuple[str, int, int], int] = {}
        self._pending_hints: list[_PendingHint] = []
        self._inflight_hint_keys: set[tuple[str, int, int]] = set()
        self._horizon_hits = {distance: 0 for distance in range(1, beam_horizon + 1)}
        self._horizon_misses = {
            distance: 0 for distance in range(1, beam_horizon + 1)
        }
        self._hints_by_distance = {
            distance: 0 for distance in range(1, beam_horizon + 1)
        }
        self._metrics = {
            "demand_operations": 0,
            "operations": 0,
            "leaves": 0,
            "predictions": 0,
            "prediction_hits": 0,
            "prediction_misses": 0,
            "prefetch_attempts": 0,
            "prefetch_hints": 0,
            "prefetch_hint_bytes": 0,
            "prefetch_declines": 0,
            "prefetch_errors": 0,
            "low_support": 0,
            "low_confidence": 0,
            "oversize_operations": 0,
            "out_of_order_operations": 0,
            "ignored_prefetch_operations": 0,
            "flushes": 0,
            "beam_candidates_considered": 0,
            "beam_plans": 0,
            "beam_steps": 0,
            "reservoir_states": 0,
            "reservoir_duplicate_leaves_avoided": 0,
            "reservoir_cooldown_leaves_skipped": 0,
            "reservoir_truncated_by_bytes": 0,
            "reservoir_truncated_by_leaves": 0,
            "hint_feedback": 0,
            "hint_useful_bytes": 0,
            "hint_wasted_bytes": 0,
            "hint_early_hits": 0,
        }
        self._acquire_state_lock()
        try:
            self._state = self._load_state()
        except Exception:
            self._release_state_lock()
            raise

    def _acquire_state_lock(self) -> None:
        path = self.state_path
        if path is None:
            return
        if fcntl is None:  # pragma: no cover - production targets are POSIX.
            raise RangeMarkovError("persistent range Markov state requires fcntl")
        path.parent.mkdir(parents=True, exist_ok=True)
        parent = path.parent.lstat()
        if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
            raise RangeMarkovError("range Markov directory must be real")
        lock_path = path.parent / f".{path.name}.lock"
        descriptor = os.open(
            lock_path,
            os.O_CREAT
            | os.O_RDWR
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
            0o600,
        )
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise RangeMarkovError("range Markov lock is not a regular file")
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except Exception:
            os.close(descriptor)
            raise
        self._state_lock_descriptor = descriptor

    def _release_state_lock(self) -> None:
        descriptor = self._state_lock_descriptor
        self._state_lock_descriptor = None
        if descriptor is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    @staticmethod
    def _weights(rapidities: Sequence[float]) -> tuple[float, float]:
        maximum = max(rapidities)
        raw = tuple(math.exp(value - maximum) for value in rapidities)
        total = sum(raw)
        return tuple(
            (1.0 - MarkovRangePrefetcher.FIXED_SHARE) * value / total
            + MarkovRangePrefetcher.FIXED_SHARE / 2.0
            for value in raw
        )

    def _distance_weights(self, state: RangeMarkovState) -> tuple[float, ...]:
        rapidities = state.distance_rapidities[: self.beam_horizon]
        maximum = max(rapidities)
        raw = tuple(math.exp(value - maximum) for value in rapidities)
        total = sum(raw)
        count = len(raw)
        return tuple(
            (1.0 - self.DISTANCE_FIXED_SHARE) * value / total
            + self.DISTANCE_FIXED_SHARE / count
            for value in raw
        )

    def _adaptive_prefetch_bytes(self, state: RangeMarkovState) -> int:
        if state.hint_feedback_count == 0:
            return self.max_prefetch_bytes
        fraction = self.MIN_BUDGET_FRACTION + (
            1.0 - self.MIN_BUDGET_FRACTION
        ) * state.hint_utility_ema
        return max(1, min(self.max_prefetch_bytes, int(self.max_prefetch_bytes * fraction)))

    @staticmethod
    def _source_identity(operation: AccessOperation) -> tuple[str, str, str]:
        return (
            operation.repo_id,
            operation.revision,
            operation.inventory_fingerprint,
        )

    def bind_source_identity(
        self,
        repo_id: str,
        revision: str,
        inventory_fingerprint: str,
    ) -> None:
        if not isinstance(repo_id, str) or not repo_id:
            raise ValueError("repo_id must be non-empty")
        if not isinstance(revision, str) or not revision:
            raise ValueError("revision must be non-empty")
        _sha256(inventory_fingerprint, field="inventory_fingerprint")
        identity = (repo_id, revision, inventory_fingerprint)
        with self._lock:
            if self._closed:
                raise RangeMarkovError("range Markov prefetcher is closed")
            if self._state.source_identity is None:
                self._state = replace(
                    self._state,
                    repo_id=repo_id,
                    revision=revision,
                    inventory_fingerprint=inventory_fingerprint,
                )
            elif self._state.source_identity != identity:
                raise RangeMarkovError(
                    "range Markov state belongs to a different source"
                )

    def _load_state(self) -> RangeMarkovState:
        path = self.state_path
        if path is None:
            return RangeMarkovState()
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return RangeMarkovState()
        except OSError as exc:
            raise RangeMarkovError("cannot inspect range Markov state") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RangeMarkovError("range Markov state must be a regular file")
        try:
            return RangeMarkovState.from_bytes(path.read_bytes())
        except OSError as exc:
            raise RangeMarkovError("cannot read range Markov state") from exc

    def _persist_locked(self) -> None:
        path = self.state_path
        if path is None or self._dirty_operations == 0:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            parent = path.parent.lstat()
        except OSError as exc:
            raise RangeMarkovError("cannot inspect range Markov directory") from exc
        if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
            raise RangeMarkovError("range Markov directory must be real")
        try:
            current = path.lstat()
        except FileNotFoundError:
            current = None
        if current is not None and (
            stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode)
        ):
            raise RangeMarkovError("range Markov state target is invalid")
        body = self._state.to_bytes()
        temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | int(getattr(os, "O_CLOEXEC", 0))
                | int(getattr(os, "O_NOFOLLOW", 0)),
                0o600,
            )
            offset = 0
            while offset < len(body):
                written = os.write(descriptor, body[offset:])
                if written <= 0:
                    raise OSError("short range Markov state write")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, path)
            directory = os.open(
                path.parent,
                os.O_RDONLY
                | int(getattr(os, "O_DIRECTORY", 0))
                | int(getattr(os, "O_CLOEXEC", 0)),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as exc:
            raise RangeMarkovError("cannot persist range Markov state") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
        self._dirty_operations = 0
        self._metrics["flushes"] += 1

    def _expert_distributions(
        self,
        state: RangeMarkovState,
        history: Sequence[str] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        selected_history = state.history if history is None else tuple(history)[-2:]
        contexts = {row.history: row for row in state.contexts}
        order_1 = (
            {}
            if not selected_history
            else contexts.get((selected_history[-1],), None)
        )
        order_2 = (
            {}
            if len(selected_history) < 2
            else contexts.get(selected_history[-2:], None)
        )
        return (
            {} if not isinstance(order_1, RangeContext) else order_1.distribution(),
            {} if not isinstance(order_2, RangeContext) else order_2.distribution(),
        )

    def _mixture_for_history(
        self,
        state: RangeMarkovState,
        history: Sequence[str],
    ) -> tuple[
        dict[str, float],
        tuple[float, float],
        dict[str, int],
        dict[str, int],
    ]:
        selected_history = tuple(history)[-2:]
        distributions = self._expert_distributions(state, selected_history)
        available = tuple(index for index, row in enumerate(distributions) if row)
        weights = self._weights(state.expert_rapidities)
        if not available:
            return {}, weights, {}, {}
        available_mass = sum(weights[index] for index in available)
        mixture: dict[str, float] = {}
        for index in available:
            scaled = weights[index] / available_mass
            for key, probability in distributions[index].items():
                mixture[key] = mixture.get(key, 0.0) + scaled * probability
        contexts = {row.history: row for row in state.contexts}
        supports: dict[str, int] = {}
        totals: dict[str, int] = {}
        for order in (1, 2):
            if len(selected_history) < order:
                continue
            context = contexts.get(selected_history[-order:])
            if context is None:
                continue
            for key, count in context.targets:
                supports[key] = max(supports.get(key, 0), count)
                totals[key] = max(totals.get(key, 0), context.visits)
        return mixture, weights, supports, totals

    def _prediction_locked(self, state: RangeMarkovState) -> RangePrediction | None:
        mixture, weights, supports, totals = self._mixture_for_history(
            state,
            state.history,
        )
        if not mixture:
            return None
        nodes = {row.state.key: row for row in state.nodes}
        target_key, confidence = max(
            mixture.items(),
            key=lambda row: (
                row[1],
                nodes[row[0]].ricci_value(
                    clock=state.clock,
                    alpha=self.RICCI_ALPHA,
                ),
                row[0],
            ),
        )
        node = nodes[target_key]
        age = max(0, state.clock - node.last_seen)
        return RangePrediction(
            state=node.state,
            confidence=confidence,
            support=supports[target_key],
            total=totals[target_key],
            expert_weights=(
                ("order-1", weights[0]),
                ("order-2", weights[1]),
            ),
            ricci_value=node.ricci_value(clock=state.clock, alpha=self.RICCI_ALPHA),
            expected_reuse_distance=(age + 1.0) / node.visits,
        )

    def _beam_plan_locked(
        self,
        state: RangeMarkovState,
        *,
        current_operation: int | None = None,
    ) -> RangeBeamPlan | None:
        if current_operation is None:
            current_operation = int(self._metrics["demand_operations"]) + 1
        nodes = {row.state.key: row for row in state.nodes}
        distance_weights = self._distance_weights(state)
        frontier: list[
            tuple[tuple[str, ...], float, tuple[str, ...], float, int]
        ] = [
            (state.history, 1.0, (), 1.0, 2**63 - 1)
        ]
        steps: list[RangeBeamStep] = []
        candidates_considered = 0
        for distance in range(1, self.beam_horizon + 1):
            candidates: list[
                tuple[RangeBeamStep, tuple[str, ...]]
            ] = []
            for (
                history,
                path_probability,
                path,
                path_min_probability,
                path_min_support,
            ) in frontier:
                mixture, _weights, supports, totals = self._mixture_for_history(
                    state,
                    history,
                )
                candidates_considered += len(mixture)
                for key, transition_probability in mixture.items():
                    node = nodes.get(key)
                    if node is None:
                        continue
                    next_path_probability = (
                        path_probability * transition_probability
                    )
                    age = max(0, state.clock - node.last_seen)
                    reuse_distance = (age + distance) / node.visits
                    ricci = node.ricci_value(
                        clock=state.clock,
                        alpha=self.RICCI_ALPHA,
                    )
                    ricci_factor = 1.0 + min(4.0, math.log1p(ricci))
                    score = (
                        next_path_probability
                        * ricci_factor
                        * distance_weights[distance - 1]
                        * self.beam_horizon
                        / (distance * max(1.0, reuse_distance))
                    )
                    next_path = (*path, key)
                    step = RangeBeamStep(
                        distance=distance,
                        state=node.state,
                        transition_probability=transition_probability,
                        path_probability=next_path_probability,
                        path_min_transition_probability=min(
                            path_min_probability,
                            transition_probability,
                        ),
                        support=supports[key],
                        total=totals[key],
                        path_min_support=min(path_min_support, supports[key]),
                        ricci_value=ricci,
                        expected_reuse_distance=reuse_distance,
                        distance_weight=distance_weights[distance - 1],
                        score=score,
                        path=next_path,
                    )
                    next_history = (*history, key)[-2:]
                    candidates.append((step, next_history))
            if not candidates:
                break
            candidates.sort(
                key=lambda row: (
                    -row[0].score,
                    -row[0].path_probability,
                    row[0].state.key,
                    row[0].path,
                )
            )
            kept = candidates[: self.beam_width]
            steps.extend(row[0] for row in kept)
            frontier = [
                (
                    row[1],
                    row[0].path_probability,
                    row[0].path,
                    row[0].path_min_transition_probability,
                    row[0].path_min_support,
                )
                for row in kept
            ]
        if not steps:
            return None
        effective_hint_budget = self._adaptive_prefetch_bytes(state)
        hints, duplicates, cooldown, truncated_bytes, truncated_leaves = (
            self._reservoir_hints_locked(
                steps,
                effective_hint_budget=effective_hint_budget,
                current_operation=current_operation,
            )
        )
        return RangeBeamPlan(
            horizon=self.beam_horizon,
            beam_width=self.beam_width,
            candidates_considered=candidates_considered,
            steps=tuple(steps),
            hints=hints,
            distance_weights=tuple(
                (index + 1, value)
                for index, value in enumerate(distance_weights)
            ),
            configured_hint_bytes=self.max_prefetch_bytes,
            effective_hint_budget_bytes=effective_hint_budget,
            duplicate_leaves_avoided=duplicates,
            cooldown_leaves_skipped=cooldown,
            truncated_by_bytes=truncated_bytes,
            truncated_by_leaves=truncated_leaves,
        )

    def _reservoir_hints_locked(
        self,
        steps: Sequence[RangeBeamStep],
        *,
        effective_hint_budget: int,
        current_operation: int,
    ) -> tuple[tuple[RangeHint, ...], int, int, bool, bool]:
        ranked = sorted(
            (
                row
                for row in steps
                if row.path_min_support >= self.min_support
                and row.path_min_transition_probability >= self.min_confidence
            ),
            key=lambda row: (
                -row.score,
                row.distance,
                row.state.key,
                row.path,
            ),
        )
        seen: set[tuple[str, int, int]] = set()
        candidates: list[tuple[RangeBeamStep, AccessLeaf]] = []
        duplicates = 0
        cooldown = 0
        pending_keys = {
            row.original_key for row in self._pending_hints
        } | self._inflight_hint_keys
        state_keys: set[str] = set()
        states = []
        for step in ranked:
            if step.state.key in state_keys:
                duplicates += len(step.state.leaves)
                continue
            state_keys.add(step.state.key)
            states.append(step)
        max_state_leaves = max(
            (len(step.state.leaves) for step in states),
            default=0,
        )
        for leaf_index in range(max_state_leaves):
            for step in states:
                if leaf_index >= len(step.state.leaves):
                    continue
                leaf = step.state.leaves[leaf_index]
                key = (leaf.shard, leaf.offset, leaf.length)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                if key in pending_keys:
                    cooldown += 1
                    continue
                last_hint = self._recent_hint_operations.get(key)
                if (
                    last_hint is not None
                    and current_operation - last_hint
                    <= self.hint_cooldown_operations
                ):
                    cooldown += 1
                    continue
                candidates.append((step, leaf))
        truncated_by_leaves = len(candidates) > self.max_prefetch_leaves
        selected = candidates[: self.max_prefetch_leaves]
        if not selected:
            return (), duplicates, cooldown, False, truncated_by_leaves
        state_order = tuple(dict.fromkeys(step.state.key for step, _leaf in selected))
        base = effective_hint_budget // len(state_order)
        extra = effective_hint_budget % len(state_order)
        state_budgets = {
            key: base + int(index < extra)
            for index, key in enumerate(state_order)
        }
        allocations = [0] * len(selected)
        for index, (step, leaf) in enumerate(selected):
            amount = min(leaf.length, state_budgets[step.state.key])
            allocations[index] = amount
            state_budgets[step.state.key] -= amount
        remaining = effective_hint_budget - sum(allocations)
        for index, (_step, leaf) in enumerate(selected):
            if remaining <= 0:
                break
            missing = leaf.length - allocations[index]
            addition = min(missing, remaining)
            allocations[index] += addition
            remaining -= addition
        hints = tuple(
            RangeHint(
                leaf=AccessLeaf(leaf.shard, leaf.offset, amount),
                source_state_key=step.state.key,
                distance=step.distance,
                score=step.score,
                path_probability=step.path_probability,
                original_length=leaf.length,
            )
            for (step, leaf), amount in zip(selected, allocations, strict=True)
            if amount > 0
        )
        truncated_by_bytes = any(
            amount < leaf.length
            for (_step, leaf), amount in zip(selected, allocations, strict=True)
        ) or (bool(candidates) and not hints)
        return (
            hints,
            duplicates,
            cooldown,
            truncated_by_bytes,
            truncated_by_leaves,
        )

    def _update_experts(
        self,
        state: RangeMarkovState,
        actual_key: str,
    ) -> tuple[
        tuple[float, float],
        tuple[int, int],
        tuple[int, int],
        float,
        float,
        float,
        bool,
    ]:
        distributions = self._expert_distributions(state)
        logs = [self.RAPIDITY_DECAY * value for value in state.expert_rapidities]
        observations = list(state.expert_observations)
        hits = list(state.expert_hits)
        weights = self._weights(state.expert_rapidities)
        mixture_probability = 0.0
        available_mass = sum(
            weights[index] for index, row in enumerate(distributions) if row
        )
        for index, distribution in enumerate(distributions):
            if not distribution:
                continue
            prediction = max(distribution.items(), key=lambda row: (row[1], row[0]))[0]
            hit = prediction == actual_key
            observations[index] += 1
            hits[index] += int(hit)
            logs[index] += self.LEARNING_RATE * (1.0 if hit else -1.0)
            mixture_probability += (
                weights[index]
                / available_mass
                * distribution.get(actual_key, 1e-12)
            )
        center = sum(logs) / len(logs)
        logs = [max(-20.0, min(20.0, value - center)) for value in logs]
        if available_mass == 0.0:
            return (
                tuple(logs),
                tuple(observations),
                tuple(hits),
                state.surprise_mean,
                state.surprise_deviation,
                state.surprise_cusum,
                False,
            )
        surprise = -math.log(max(mixture_probability, 1e-12))
        z_score = (
            0.0
            if state.observations == 0
            else (surprise - state.surprise_mean)
            / max(state.surprise_deviation, 1e-6)
        )
        mean = (
            (1.0 - self.SURPRISE_RATE) * state.surprise_mean
            + self.SURPRISE_RATE * surprise
        )
        deviation = (
            (1.0 - self.SURPRISE_RATE) * state.surprise_deviation
            + self.SURPRISE_RATE * abs(surprise - state.surprise_mean)
        )
        cusum = max(
            0.0,
            self.CUSUM_DECAY * state.surprise_cusum
            + z_score
            - self.CUSUM_DRIFT,
        )
        regime = (
            state.observations + 1 >= self.REGIME_WARMUP
            and cusum > self.CUSUM_THRESHOLD
        )
        if regime:
            logs = [self.REGIME_SHRINK * value for value in logs]
            cusum = 0.0
        return (
            tuple(logs),
            tuple(observations),
            tuple(hits),
            mean,
            deviation,
            cusum,
            regime,
        )

    def _settle_pending_hints_locked(
        self,
        state: RangeMarkovState,
        *,
        current_operation: int,
        demand: AccessState,
    ) -> RangeMarkovState:
        remaining = []
        settled: list[tuple[_PendingHint, int]] = []
        for hint in self._pending_hints:
            updated = hint.observe_demand(demand.leaves)
            if updated.covered_bytes == updated.leaf.length:
                settled.append((updated, updated.covered_bytes))
                if current_operation < hint.due_operation:
                    self._metrics["hint_early_hits"] += 1
            elif current_operation >= hint.due_operation:
                settled.append((updated, updated.covered_bytes))
            else:
                remaining.append(updated)
        self._pending_hints = remaining
        if not settled:
            return state
        rapidities = list(state.distance_rapidities)
        observations = list(state.distance_observations)
        hits = list(state.distance_hits)
        hinted_bytes = list(state.distance_hinted_bytes)
        useful_bytes = list(state.distance_useful_bytes)
        by_distance: dict[int, list[int]] = {}
        for hint, useful in settled:
            index = hint.distance - 1
            totals = by_distance.setdefault(index, [0, 0])
            totals[0] += hint.leaf.length
            totals[1] += useful
        rapidities = [
            self.DISTANCE_RAPIDITY_DECAY * value for value in rapidities
        ]
        for index, (distance_hinted, distance_useful) in sorted(
            by_distance.items()
        ):
            ratio = distance_useful / distance_hinted
            rapidities[index] += self.DISTANCE_LEARNING_RATE * (2.0 * ratio - 1.0)
            observations[index] += 1
            hits[index] += int(distance_useful > 0)
            hinted_bytes[index] += distance_hinted
            useful_bytes[index] += distance_useful
        center = sum(rapidities) / len(rapidities)
        rapidities = [
            max(-20.0, min(20.0, value - center))
            for value in rapidities
        ]
        total_hinted = sum(row[0] for row in by_distance.values())
        total_useful = sum(row[1] for row in by_distance.values())
        ratio = total_useful / total_hinted
        rate = (
            1.0
            if state.hint_feedback_count == 0
            else self.HINT_UTILITY_RATE
        )
        utility_ema = (1.0 - rate) * state.hint_utility_ema + rate * ratio
        feedback_updates = len(by_distance)
        self._metrics["hint_feedback"] += feedback_updates
        self._metrics["hint_useful_bytes"] += total_useful
        self._metrics["hint_wasted_bytes"] += total_hinted - total_useful
        return replace(
            state,
            distance_rapidities=tuple(rapidities),
            distance_observations=tuple(observations),
            distance_hits=tuple(hits),
            distance_hinted_bytes=tuple(hinted_bytes),
            distance_useful_bytes=tuple(useful_bytes),
            hint_feedback_count=state.hint_feedback_count + feedback_updates,
            hint_utility_ema=utility_ema,
        )

    def _updated_state(
        self,
        state: RangeMarkovState,
        access: AccessState,
    ) -> RangeMarkovState:
        clock = state.clock + 1
        (
            rapidities,
            expert_observations,
            expert_hits,
            surprise_mean,
            surprise_deviation,
            surprise_cusum,
            regime,
        ) = self._update_experts(state, access.key)
        nodes = {row.state.key: row for row in state.nodes}
        prior = nodes.get(access.key)
        nodes[access.key] = RangeNode(
            state=access,
            visits=1 if prior is None else prior.visits + 1,
            last_seen=clock,
        )
        contexts = {row.history: row for row in state.contexts}
        for order in (1, 2):
            if len(state.history) < order:
                continue
            history = state.history[-order:]
            prior_context = contexts.get(history)
            targets = {} if prior_context is None else dict(prior_context.targets)
            targets[access.key] = targets.get(access.key, 0) + 1
            while len(targets) > self.max_targets_per_context:
                evicted_target = min(
                    targets,
                    key=lambda key: (targets[key], key),
                )
                del targets[evicted_target]
            contexts[history] = RangeContext(
                history=history,
                targets=tuple(sorted(targets.items())),
                visits=(
                    1 if prior_context is None else prior_context.visits + 1
                ),
                last_seen=clock,
            )
        history = (*state.history, access.key)[-2:]
        node_evictions = state.node_evictions
        protected = set(history)
        while len(nodes) > self.max_nodes:
            candidates = [row for key, row in nodes.items() if key not in protected]
            if not candidates:
                break
            evicted = min(
                candidates,
                key=lambda row: (
                    row.ricci_value(clock=clock, alpha=self.RICCI_ALPHA),
                    row.state.key,
                ),
            )
            del nodes[evicted.state.key]
            contexts = {
                key: row
                for key, row in contexts.items()
                if evicted.state.key not in key
                and evicted.state.key not in dict(row.targets)
            }
            node_evictions += 1
        context_evictions = state.context_evictions
        while len(contexts) > self.max_contexts:
            evicted = min(
                contexts.values(),
                key=lambda row: (
                    row.ricci_value(clock=clock, alpha=self.RICCI_ALPHA),
                    row.history,
                ),
            )
            del contexts[evicted.history]
            context_evictions += 1
        return RangeMarkovState(
            repo_id=state.repo_id,
            revision=state.revision,
            inventory_fingerprint=state.inventory_fingerprint,
            clock=clock,
            observations=state.observations + 1,
            nodes=tuple(sorted(nodes.values(), key=lambda row: row.state.key)),
            contexts=tuple(sorted(contexts.values(), key=lambda row: row.history)),
            history=history,
            expert_rapidities=rapidities,
            expert_observations=expert_observations,
            expert_hits=expert_hits,
            surprise_mean=surprise_mean,
            surprise_deviation=surprise_deviation,
            surprise_cusum=surprise_cusum,
            regime_generation=state.regime_generation + int(regime),
            node_evictions=node_evictions,
            context_evictions=context_evictions,
            distance_rapidities=state.distance_rapidities,
            distance_observations=state.distance_observations,
            distance_hits=state.distance_hits,
            distance_hinted_bytes=state.distance_hinted_bytes,
            distance_useful_bytes=state.distance_useful_bytes,
            hint_feedback_count=state.hint_feedback_count,
            hint_utility_ema=state.hint_utility_ema,
        )

    def observe(self, operation: AccessOperation) -> bool:
        if not isinstance(operation, AccessOperation):
            raise TypeError("operation must be AccessOperation")
        if dict(operation.tags).get("access_role") == "prefetch":
            with self._lock:
                self._metrics["ignored_prefetch_operations"] += 1
            return False
        access = AccessState.from_operation(operation)
        with self._lock:
            if self._closed:
                return False
            if (
                self._last_operation_sequence
                and operation.operation_sequence <= self._last_operation_sequence
            ):
                self._metrics["out_of_order_operations"] += 1
                return False
            self._last_operation_sequence = operation.operation_sequence
            current_operation = operation.operation_sequence
            self._metrics["demand_operations"] += 1
            identity = self._source_identity(operation)
            if self._state.source_identity is None:
                self._state = replace(
                    self._state,
                    repo_id=identity[0],
                    revision=identity[1],
                    inventory_fingerprint=identity[2],
                )
            elif self._state.source_identity != identity:
                raise RangeMarkovError(
                    "range Markov state belongs to a different source"
                )
            hint_cutoff = (
                current_operation
                - self.hint_cooldown_operations
                - self.beam_horizon
                - 1
            )
            self._recent_hint_operations = {
                key: value
                for key, value in self._recent_hint_operations.items()
                if value >= hint_cutoff
            }
            due_forecasts = tuple(
                due
                for due in self._pending_forecasts
                if due <= current_operation
            )
            for due in sorted(due_forecasts):
                forecasts = self._pending_forecasts.pop(due)
                for distance, candidates in forecasts.items():
                    if due == current_operation and access.key in candidates:
                        self._horizon_hits[distance] += 1
                    else:
                        self._horizon_misses[distance] += 1
            settled_state = self._settle_pending_hints_locked(
                self._state,
                current_operation=current_operation,
                demand=access,
            )
            feedback_changed = settled_state != self._state
            self._state = settled_state
            if len(access.leaves) > self.max_operation_leaves:
                self._metrics["oversize_operations"] += 1
                self._dirty_operations += int(feedback_changed)
                if self._dirty_operations >= self.flush_interval:
                    self._persist_locked()
                return False
            if self._pending_prediction_key is not None:
                if self._pending_prediction_key == access.key:
                    self._metrics["prediction_hits"] += 1
                else:
                    self._metrics["prediction_misses"] += 1
            self._state = self._updated_state(self._state, access)
            prediction = self._prediction_locked(self._state)
            beam_plan = self._beam_plan_locked(
                self._state,
                current_operation=current_operation,
            )
            self._last_prediction = prediction
            self._last_beam_plan = beam_plan
            self._pending_prediction_key = (
                None if prediction is None else prediction.state.key
            )
            self._metrics["operations"] += 1
            self._metrics["leaves"] += len(access.leaves)
            self._dirty_operations += 1
            if prediction is not None:
                self._metrics["predictions"] += 1
            if beam_plan is not None:
                self._metrics["beam_plans"] += 1
                self._metrics["beam_candidates_considered"] += (
                    beam_plan.candidates_considered
                )
                self._metrics["beam_steps"] += len(beam_plan.steps)
                self._metrics["reservoir_states"] += beam_plan.predicted_states
                self._metrics["reservoir_duplicate_leaves_avoided"] += (
                    beam_plan.duplicate_leaves_avoided
                )
                self._metrics["reservoir_cooldown_leaves_skipped"] += (
                    beam_plan.cooldown_leaves_skipped
                )
                self._metrics["reservoir_truncated_by_bytes"] += int(
                    beam_plan.truncated_by_bytes
                )
                self._metrics["reservoir_truncated_by_leaves"] += int(
                    beam_plan.truncated_by_leaves
                )
                for distance in range(1, self.beam_horizon + 1):
                    candidates = frozenset(
                        row.state.key
                        for row in beam_plan.steps
                        if row.distance == distance
                    )
                    if candidates:
                        self._pending_forecasts.setdefault(
                            current_operation + distance,
                            {},
                        )[distance] = candidates
            if prediction is None or prediction.support < self.min_support:
                self._metrics["low_support"] += int(prediction is not None)
            elif prediction.confidence < self.min_confidence:
                self._metrics["low_confidence"] += 1
            hints = () if beam_plan is None else beam_plan.hints
            if self._dirty_operations >= self.flush_interval:
                self._persist_locked()
        for hint in hints:
            original_key = (
                hint.leaf.shard,
                hint.leaf.offset,
                hint.original_length,
            )
            with self._lock:
                if original_key in self._inflight_hint_keys or any(
                    row.original_key == original_key
                    for row in self._pending_hints
                ):
                    self._metrics["reservoir_cooldown_leaves_skipped"] += 1
                    continue
                self._inflight_hint_keys.add(original_key)
                self._metrics["prefetch_attempts"] += 1
                self._hints_by_distance[hint.distance] += 1
            try:
                accepted = self.prefetch_range(
                    hint.leaf.shard,
                    hint.leaf.offset,
                    hint.leaf.length,
                )
            except Exception:
                with self._lock:
                    self._inflight_hint_keys.discard(original_key)
                    self._metrics["prefetch_errors"] += 1
                continue
            with self._lock:
                self._inflight_hint_keys.discard(original_key)
                if accepted:
                    self._metrics["prefetch_hints"] += 1
                    self._metrics["prefetch_hint_bytes"] += hint.leaf.length
                    self._pending_hints.append(
                        _PendingHint(
                            created_operation=current_operation,
                            due_operation=current_operation + hint.distance,
                            distance=hint.distance,
                            leaf=hint.leaf,
                            original_length=hint.original_length,
                        )
                    )
                    self._recent_hint_operations[original_key] = current_operation
                else:
                    self._metrics["prefetch_declines"] += 1
        return True

    @property
    def last_prediction(self) -> RangePrediction | None:
        with self._lock:
            return self._last_prediction

    @property
    def last_beam_plan(self) -> RangeBeamPlan | None:
        with self._lock:
            return self._last_beam_plan

    def flush(self) -> None:
        with self._lock:
            if self._closed:
                raise RangeMarkovError("range Markov prefetcher is closed")
            self._persist_locked()

    def metrics(self) -> dict[str, object]:
        with self._lock:
            weights = self._weights(self._state.expert_rapidities)
            distance_weights = self._distance_weights(self._state)
            effective_prefetch_bytes = self._adaptive_prefetch_bytes(self._state)
            return {
                **self._metrics,
                "beam_horizon": self.beam_horizon,
                "beam_width": self.beam_width,
                "clock": self._state.clock,
                "context_evictions": self._state.context_evictions,
                "contexts": len(self._state.contexts),
                "effective_experts": 1.0 / sum(value * value for value in weights),
                "effective_prefetch_bytes": effective_prefetch_bytes,
                "effective_prefetch_fraction": (
                    effective_prefetch_bytes / self.max_prefetch_bytes
                ),
                "expert_accuracy": {
                    name: (
                        0.0 if seen == 0 else hit / seen
                    )
                    for name, hit, seen in zip(
                        ("order-1", "order-2"),
                        self._state.expert_hits,
                        self._state.expert_observations,
                        strict=True,
                    )
                },
                "expert_weights": {
                    "order-1": weights[0],
                    "order-2": weights[1],
                },
                "distance_weights": {
                    str(index + 1): value
                    for index, value in enumerate(distance_weights)
                },
                "distance_accuracy": {
                    str(index + 1): (
                        0.0
                        if self._state.distance_observations[index] == 0
                        else self._state.distance_hits[index]
                        / self._state.distance_observations[index]
                    )
                    for index in range(self.beam_horizon)
                },
                "distance_byte_utility": {
                    str(index + 1): (
                        0.0
                        if self._state.distance_hinted_bytes[index] == 0
                        else self._state.distance_useful_bytes[index]
                        / self._state.distance_hinted_bytes[index]
                    )
                    for index in range(self.beam_horizon)
                },
                "hint_feedback_count": self._state.hint_feedback_count,
                "hint_utility_ema": self._state.hint_utility_ema,
                "last_prediction": (
                    None
                    if self._last_prediction is None
                    else self._last_prediction.to_dict()
                ),
                "last_beam_plan": (
                    None
                    if self._last_beam_plan is None
                    else self._last_beam_plan.to_dict()
                ),
                "horizon_hits": {
                    str(key): value for key, value in self._horizon_hits.items()
                },
                "horizon_misses": {
                    str(key): value for key, value in self._horizon_misses.items()
                },
                "prefetch_hints_by_distance": {
                    str(key): value for key, value in self._hints_by_distance.items()
                },
                "node_evictions": self._state.node_evictions,
                "nodes": len(self._state.nodes),
                "observations": self._state.observations,
                "pending_hint_feedback": len(self._pending_hints),
                "regime_generation": self._state.regime_generation,
                "schema": RANGE_MARKOV_METRICS_SCHEMA,
                "state_bytes": len(self._state.to_bytes()),
                "surprise_cusum": self._state.surprise_cusum,
                "surprise_mean": self._state.surprise_mean,
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._persist_locked()
            finally:
                self._closed = True
                self._release_state_lock()

    def __enter__(self) -> "MarkovRangePrefetcher":
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


__all__ = [
    "RANGE_MARKOV_BEAM_PLAN_SCHEMA",
    "RANGE_MARKOV_BEAM_STEP_SCHEMA",
    "RANGE_MARKOV_METRICS_SCHEMA",
    "RANGE_MARKOV_PREDICTION_SCHEMA",
    "RANGE_MARKOV_STATE_SCHEMA",
    "AccessState",
    "MarkovRangePrefetcher",
    "RangeContext",
    "RangeBeamPlan",
    "RangeBeamStep",
    "RangeHint",
    "RangeMarkovError",
    "RangeMarkovState",
    "RangeNode",
    "RangePrediction",
]
