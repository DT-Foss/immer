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


RANGE_MARKOV_STATE_SCHEMA = "immer.range-markov-state/v1"
RANGE_MARKOV_PREDICTION_SCHEMA = "immer.range-markov-prediction/v1"
RANGE_MARKOV_METRICS_SCHEMA = "immer.range-markov-metrics/v1"

_STATE_PREFIX = b"IMRM\x01"
_MAX_STATE_BYTES = 16 * 1024 * 1024
_MAX_JSON_BYTES = 64 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")
_SEMANTIC_TAGS = frozenset({"layer", "phase", "read_kind", "tensor"})


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
        if self.visits != sum(count for _key, count in targets):
            raise ValueError("range context visits differ from target counts")
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
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "contexts", contexts)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "expert_rapidities", rapidities)
        object.__setattr__(self, "expert_observations", expert_observations)
        object.__setattr__(self, "expert_hits", expert_hits)

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
            "expert_hits": list(self.expert_hits),
            "expert_observations": list(self.expert_observations),
            "expert_rapidities": [
                _float_record(value) for value in self.expert_rapidities
            ],
            "history": list(self.history),
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
            "expert_hits",
            "expert_observations",
            "expert_rapidities",
            "history",
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
    def from_bytes(cls, value: bytes) -> "RangeMarkovState":
        if (
            not isinstance(value, bytes)
            or not value.startswith(_STATE_PREFIX)
            or not 5 < len(value) <= _MAX_STATE_BYTES
        ):
            raise RangeMarkovError("range Markov state header is invalid")
        try:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(value[len(_STATE_PREFIX) :], _MAX_JSON_BYTES + 1)
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
            (flush_interval, "flush_interval"),
        ):
            _uint(value, field=field, positive=True)
        _finite(min_confidence, field="min_confidence", lower=0.0, upper=1.0)
        if max_nodes < 4 or max_contexts < 4:
            raise ValueError("range Markov capacities are too small")
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
        self.flush_interval = flush_interval
        self._lock = threading.RLock()
        self._closed = False
        self._state_lock_descriptor: int | None = None
        self._dirty_operations = 0
        self._last_operation_sequence = 0
        self._pending_prediction_key: str | None = None
        self._last_prediction: RangePrediction | None = None
        self._metrics = {
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
    ) -> tuple[dict[str, float], dict[str, float]]:
        contexts = {row.history: row for row in state.contexts}
        order_1 = (
            {}
            if not state.history
            else contexts.get((state.history[-1],), None)
        )
        order_2 = (
            {}
            if len(state.history) < 2
            else contexts.get(state.history[-2:], None)
        )
        return (
            {} if not isinstance(order_1, RangeContext) else order_1.distribution(),
            {} if not isinstance(order_2, RangeContext) else order_2.distribution(),
        )

    def _prediction_locked(self, state: RangeMarkovState) -> RangePrediction | None:
        distributions = self._expert_distributions(state)
        available = tuple(index for index, row in enumerate(distributions) if row)
        if not available:
            return None
        weights = self._weights(state.expert_rapidities)
        available_mass = sum(weights[index] for index in available)
        mixture: dict[str, float] = {}
        for index in available:
            scaled = weights[index] / available_mass
            for key, probability in distributions[index].items():
                mixture[key] = mixture.get(key, 0.0) + scaled * probability
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
        contexts = {row.history: row for row in state.contexts}
        supports = []
        totals = []
        for order in (1, 2):
            if len(state.history) < order:
                continue
            context = contexts.get(state.history[-order:])
            if context is None:
                continue
            supports.append(dict(context.targets).get(target_key, 0))
            totals.append(context.visits)
        node = nodes[target_key]
        age = max(0, state.clock - node.last_seen)
        return RangePrediction(
            state=node.state,
            confidence=confidence,
            support=max(supports),
            total=max(totals),
            expert_weights=(
                ("order-1", weights[0]),
                ("order-2", weights[1]),
            ),
            ricci_value=node.ricci_value(clock=state.clock, alpha=self.RICCI_ALPHA),
            expected_reuse_distance=(age + 1.0) / node.visits,
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
                visits=sum(targets.values()),
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
        )

    def _hint_plan(self, prediction: RangePrediction) -> tuple[AccessLeaf, ...]:
        remaining = self.max_prefetch_bytes
        hints = []
        for leaf in prediction.state.leaves[: self.max_prefetch_leaves]:
            if remaining <= 0:
                break
            length = min(leaf.length, remaining)
            if length <= 0:
                continue
            hints.append(AccessLeaf(leaf.shard, leaf.offset, length))
            remaining -= length
        return tuple(hints)

    def observe(self, operation: AccessOperation) -> bool:
        if not isinstance(operation, AccessOperation):
            raise TypeError("operation must be AccessOperation")
        if dict(operation.tags).get("access_role") == "prefetch":
            with self._lock:
                self._metrics["ignored_prefetch_operations"] += 1
            return False
        access = AccessState.from_operation(operation)
        if len(access.leaves) > self.max_operation_leaves:
            with self._lock:
                self._metrics["oversize_operations"] += 1
            return False
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
            if self._pending_prediction_key is not None:
                if self._pending_prediction_key == access.key:
                    self._metrics["prediction_hits"] += 1
                else:
                    self._metrics["prediction_misses"] += 1
            self._state = self._updated_state(self._state, access)
            prediction = self._prediction_locked(self._state)
            self._last_prediction = prediction
            self._pending_prediction_key = (
                None if prediction is None else prediction.state.key
            )
            self._metrics["operations"] += 1
            self._metrics["leaves"] += len(access.leaves)
            self._dirty_operations += 1
            if prediction is not None:
                self._metrics["predictions"] += 1
            if prediction is None or prediction.support < self.min_support:
                self._metrics["low_support"] += int(prediction is not None)
                hints = ()
            elif prediction.confidence < self.min_confidence:
                self._metrics["low_confidence"] += 1
                hints = ()
            else:
                hints = self._hint_plan(prediction)
            if self._dirty_operations >= self.flush_interval:
                self._persist_locked()
        for leaf in hints:
            with self._lock:
                self._metrics["prefetch_attempts"] += 1
            try:
                accepted = self.prefetch_range(
                    leaf.shard,
                    leaf.offset,
                    leaf.length,
                )
            except Exception:
                with self._lock:
                    self._metrics["prefetch_errors"] += 1
                continue
            with self._lock:
                if accepted:
                    self._metrics["prefetch_hints"] += 1
                    self._metrics["prefetch_hint_bytes"] += leaf.length
                else:
                    self._metrics["prefetch_declines"] += 1
        return True

    @property
    def last_prediction(self) -> RangePrediction | None:
        with self._lock:
            return self._last_prediction

    def flush(self) -> None:
        with self._lock:
            if self._closed:
                raise RangeMarkovError("range Markov prefetcher is closed")
            self._persist_locked()

    def metrics(self) -> dict[str, object]:
        with self._lock:
            weights = self._weights(self._state.expert_rapidities)
            return {
                **self._metrics,
                "clock": self._state.clock,
                "context_evictions": self._state.context_evictions,
                "contexts": len(self._state.contexts),
                "effective_experts": 1.0 / sum(value * value for value in weights),
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
                "last_prediction": (
                    None
                    if self._last_prediction is None
                    else self._last_prediction.to_dict()
                ),
                "node_evictions": self._state.node_evictions,
                "nodes": len(self._state.nodes),
                "observations": self._state.observations,
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
    "RANGE_MARKOV_METRICS_SCHEMA",
    "RANGE_MARKOV_PREDICTION_SCHEMA",
    "RANGE_MARKOV_STATE_SCHEMA",
    "AccessState",
    "MarkovRangePrefetcher",
    "RangeContext",
    "RangeMarkovError",
    "RangeMarkovState",
    "RangeNode",
    "RangePrediction",
]
