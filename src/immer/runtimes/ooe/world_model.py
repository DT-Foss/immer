"""Action-conditioned finite world models for Markov-OoE.

The world model stores only verified one-step evidence.  It does not cache
teacher-produced plans: longer behaviour is composed from action-conditioned
transition rows at planning time.  Replica statistics are fused by executing
the existing PS-Lifted push-sum, and every topology, payload and result is
bound into a canonical receipt.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .consensus import (
    ConsensusReceipt,
    adaptive_lift_parameters,
    adaptive_ps_lift_matrix,
    measure_consensus,
    ps_lifted_matrix,
    validate_adjacency,
)
from .identity import canonical_json_bytes, require_sha256
from .math_core import (
    DEFAULT_MAX_DENSE_BYTES,
    DEFAULT_MAX_NODES,
    FloatArray,
    array_sha256,
)


WORLD_MODEL_SCHEMA = "immer-ooe-action-world-model/v1"
TRANSITION_EVIDENCE_SCHEMA = "immer-ooe-transition-evidence/v1"
REGIME_CHANGE_SCHEMA = "immer-ooe-regime-change/v1"
WORLD_FUSION_SCHEMA = "immer-ooe-world-fusion/v1"
WORLD_MODEL_PAYLOAD_SCHEMA = "immer-ooe-action-world-model-payload/v1"
DEFAULT_MAX_ACTIONS = 256
DEFAULT_MAX_PROVENANCE = 1_000_000
DEFAULT_MAX_REGIME_CHANGES = 4096
DEFAULT_MAX_EVIDENCE_MASS = float(2**48)
MAX_WORLD_MODEL_PAYLOAD_BYTES = 512 * 1024 * 1024


class WorldModelTamperError(ValueError):
    """A serialized world model violated its canonical integrity contract."""


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json(data: bytes) -> object:
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
        raise WorldModelTamperError("world-model payload is not strict JSON") from exc
    if canonical_json_bytes(value) != data:
        raise WorldModelTamperError("world-model payload is not canonical JSON")
    return value


def _payload_limit(value: object, *, field: str) -> int:
    limit = _uint(value, field=field, positive=True)
    if limit > MAX_WORLD_MODEL_PAYLOAD_BYTES:
        raise ValueError(
            f"{field} exceeds MAX_WORLD_MODEL_PAYLOAD_BYTES="
            f"{MAX_WORLD_MODEL_PAYLOAD_BYTES}"
        )
    return limit


def _label(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > 1024
    ):
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _labels(
    values: Sequence[str],
    *,
    field: str,
    minimum: int,
    maximum: int,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{field} must be a sequence")
    result = tuple(_label(value, field=field) for value in values)
    if not minimum <= len(result) <= maximum:
        raise ValueError(f"{field} count must lie in [{minimum}, {maximum}]")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} must be unique")
    return result


def _finite_positive(value: object, *, field: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    lower_ok = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not lower_ok:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field} must be finite and {qualifier}")
    return result


def _probability(value: object, *, field: str) -> float:
    result = _finite_positive(value, field=field, allow_zero=True)
    if result > 1.0:
        raise ValueError(f"{field} must lie in [0, 1]")
    return result


def _uint(value: object, *, field: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{field} must be an integer")
    result = int(value)
    if result < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{field} must be {qualifier}")
    return result


def _normalize_row(counts: NDArray[np.float64]) -> FloatArray:
    """Normalize a positive row and make the represented mass exactly one."""

    total = float(counts.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("transition row has no evidence mass")
    result = np.asarray(counts / total, dtype=np.float64)
    # Deterministically absorb the last floating residual into the largest bin.
    pivot = int(np.argmax(result))
    residual = 1.0 - float(result.sum())
    result[pivot] += residual
    if np.any(result < 0.0) or not np.isfinite(result).all():
        raise ArithmeticError("row normalization produced an invalid distribution")
    return np.ascontiguousarray(result)


def label_sha256(kind: str, label: str) -> str:
    return _sha256({"kind": _label(kind, field="kind"), "label": label})


@dataclass(frozen=True, slots=True)
class TransitionEvidence:
    """One verified, weighted, single-step teacher observation."""

    source_state: str
    action: str
    target_state: str
    weight: float
    verifier_sha256: str
    evidence_sha256: str
    epoch: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source_state", _label(self.source_state, field="source_state")
        )
        object.__setattr__(self, "action", _label(self.action, field="action"))
        object.__setattr__(
            self, "target_state", _label(self.target_state, field="target_state")
        )
        weight = _finite_positive(self.weight, field="weight")
        if weight > 1_000_000.0:
            raise ValueError("weight exceeds the per-observation bound")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(
            self,
            "verifier_sha256",
            require_sha256(self.verifier_sha256, field="verifier_sha256"),
        )
        object.__setattr__(
            self,
            "evidence_sha256",
            require_sha256(self.evidence_sha256, field="evidence_sha256"),
        )
        object.__setattr__(self, "epoch", _uint(self.epoch, field="epoch"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TRANSITION_EVIDENCE_SCHEMA,
            "source_state": self.source_state,
            "action": self.action,
            "target_state": self.target_state,
            "weight": self.weight,
            "verifier_sha256": self.verifier_sha256,
            "evidence_sha256": self.evidence_sha256,
            "epoch": self.epoch,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class TransitionPrediction:
    source_state: str
    action: str
    probabilities: tuple[float, ...]
    state_labels: tuple[str, ...]
    evidence_mass: float
    event_count: int
    coverage: float
    normalized_entropy: float
    peak_probability: float
    confidence: float
    abstained: bool
    reason: str | None
    row_sha256: str
    world_model_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "immer-ooe-transition-prediction/v1",
            "source_state": self.source_state,
            "action": self.action,
            "probabilities": list(self.probabilities),
            "state_labels": list(self.state_labels),
            "evidence_mass": self.evidence_mass,
            "event_count": self.event_count,
            "coverage": self.coverage,
            "normalized_entropy": self.normalized_entropy,
            "peak_probability": self.peak_probability,
            "confidence": self.confidence,
            "abstained": self.abstained,
            "reason": self.reason,
            "row_sha256": self.row_sha256,
            "world_model_sha256": self.world_model_sha256,
        }


@dataclass(frozen=True, slots=True)
class RegimeChangeSignal:
    """Explicit nonstationarity signal and requested evidence retention."""

    signal_sha256: str
    verifier_sha256: str
    strength: float
    retention: float
    epoch: int
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "signal_sha256",
            require_sha256(self.signal_sha256, field="signal_sha256"),
        )
        object.__setattr__(
            self,
            "verifier_sha256",
            require_sha256(self.verifier_sha256, field="verifier_sha256"),
        )
        object.__setattr__(
            self, "strength", _probability(self.strength, field="strength")
        )
        object.__setattr__(
            self, "retention", _probability(self.retention, field="retention")
        )
        object.__setattr__(self, "epoch", _uint(self.epoch, field="epoch"))
        object.__setattr__(self, "name", _label(self.name, field="name"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "immer-ooe-regime-change-signal/v1",
            "signal_sha256": self.signal_sha256,
            "verifier_sha256": self.verifier_sha256,
            "strength": self.strength,
            "retention": self.retention,
            "epoch": self.epoch,
            "name": self.name,
        }


@dataclass(frozen=True, slots=True)
class RegimeChangeReceipt:
    signal: RegimeChangeSignal
    revision: int
    before_model_sha256: str
    after_model_sha256: str
    before_counts_sha256: str
    after_counts_sha256: str
    before_evidence_mass: float
    after_evidence_mass: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REGIME_CHANGE_SCHEMA,
            "signal": self.signal.to_dict(),
            "revision": self.revision,
            "before_model_sha256": self.before_model_sha256,
            "after_model_sha256": self.after_model_sha256,
            "before_counts_sha256": self.before_counts_sha256,
            "after_counts_sha256": self.after_counts_sha256,
            "before_evidence_mass": self.before_evidence_mass,
            "after_evidence_mass": self.after_evidence_mass,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class WorldFusionReceipt:
    topology: str
    adjacency_sha256: str
    lifted_transition_sha256: str
    replica_model_sha256s: tuple[str, ...]
    fused_model_sha256: str
    counts_sha256: str
    event_counts_sha256: str
    consensus: ConsensusReceipt
    lift_parameters_sha256: str
    fiedler_eigenvalue: float
    pc: float
    ps: float
    adaptive_lift: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": WORLD_FUSION_SCHEMA,
            "topology": self.topology,
            "adjacency_sha256": self.adjacency_sha256,
            "lifted_transition_sha256": self.lifted_transition_sha256,
            "replica_model_sha256s": list(self.replica_model_sha256s),
            "fused_model_sha256": self.fused_model_sha256,
            "counts_sha256": self.counts_sha256,
            "event_counts_sha256": self.event_counts_sha256,
            "consensus": self.consensus.to_dict(),
            "lift_parameters_sha256": self.lift_parameters_sha256,
            "fiedler_eigenvalue": self.fiedler_eigenvalue,
            "pc": self.pc,
            "ps": self.ps,
            "adaptive_lift": self.adaptive_lift,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class FusedWorldModel:
    model: "ActionConditionedWorldModel"
    receipt: WorldFusionReceipt


class ActionConditionedWorldModel:
    """Bounded finite ``P(next_state | state, action)`` evidence model.

    Observation and regime-adaptation methods are the only mutations.  All
    prediction, hashing, kernel and planning reads are pure.
    """

    def __init__(
        self,
        states: Sequence[str],
        actions: Sequence[str],
        *,
        min_evidence_mass: float = 1.0,
        max_normalized_entropy: float = 0.98,
        min_peak_probability: float = 0.34,
        max_states: int = DEFAULT_MAX_NODES,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        max_bytes: int = DEFAULT_MAX_DENSE_BYTES,
        max_provenance: int = DEFAULT_MAX_PROVENANCE,
        max_regime_changes: int = DEFAULT_MAX_REGIME_CHANGES,
        max_evidence_mass: float = DEFAULT_MAX_EVIDENCE_MASS,
    ) -> None:
        state_limit = _uint(max_states, field="max_states", positive=True)
        action_limit = _uint(max_actions, field="max_actions", positive=True)
        self.states = _labels(states, field="states", minimum=2, maximum=state_limit)
        self.actions = _labels(
            actions, field="actions", minimum=1, maximum=action_limit
        )
        self.min_evidence_mass = _finite_positive(
            min_evidence_mass, field="min_evidence_mass"
        )
        self.max_normalized_entropy = _probability(
            max_normalized_entropy, field="max_normalized_entropy"
        )
        self.min_peak_probability = _probability(
            min_peak_probability, field="min_peak_probability"
        )
        self.max_states = state_limit
        self.max_actions = action_limit
        self.max_bytes = _uint(max_bytes, field="max_bytes", positive=True)
        self.max_provenance = _uint(
            max_provenance, field="max_provenance", positive=True
        )
        self.max_regime_changes = _uint(
            max_regime_changes, field="max_regime_changes", positive=True
        )
        self.max_evidence_mass = _finite_positive(
            max_evidence_mass, field="max_evidence_mass"
        )
        cells = len(self.actions) * len(self.states) * len(self.states)
        required = cells * (
            np.dtype(np.float64).itemsize + np.dtype(np.uint64).itemsize
        )
        if required > self.max_bytes:
            raise ValueError(
                f"world model requires {required} bytes, exceeding {self.max_bytes}"
            )
        self._state_index = {label: index for index, label in enumerate(self.states)}
        self._action_index = {label: index for index, label in enumerate(self.actions)}
        shape = (len(self.actions), len(self.states), len(self.states))
        self._counts = np.zeros(shape, dtype=np.float64)
        self._event_counts = np.zeros(shape, dtype=np.uint64)
        self._evidence_hashes: set[str] = set()
        self._verifier_hashes: set[str] = set()
        self._regime_signal_hashes: list[str] = []
        self._regime_signals: list[RegimeChangeSignal] = []
        self._regime_revision = 0
        self._cached_sha256: str | None = None

    def _state(self, label: str, *, field: str = "state") -> int:
        try:
            return self._state_index[label]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"unknown {field}: {label!r}") from exc

    def _action(self, label: str) -> int:
        try:
            return self._action_index[label]
        except (KeyError, TypeError) as exc:
            raise KeyError(f"unknown action: {label!r}") from exc

    @property
    def counts(self) -> FloatArray:
        value = self._counts.copy()
        value.flags.writeable = False
        return value

    @property
    def event_counts(self) -> NDArray[np.uint64]:
        value = self._event_counts.copy()
        value.flags.writeable = False
        return value

    @property
    def evidence_hashes(self) -> tuple[str, ...]:
        return tuple(sorted(self._evidence_hashes))

    @property
    def verifier_hashes(self) -> tuple[str, ...]:
        return tuple(sorted(self._verifier_hashes))

    @property
    def regime_revision(self) -> int:
        return self._regime_revision

    @property
    def total_evidence_mass(self) -> float:
        return float(self._counts.sum())

    @property
    def total_events(self) -> int:
        return int(self._event_counts.sum(dtype=np.uint64))

    @property
    def contract_sha256(self) -> str:
        return _sha256(self._contract_dict())

    def _contract_dict(self) -> dict[str, Any]:
        return {
            "schema": WORLD_MODEL_SCHEMA,
            "states": list(self.states),
            "actions": list(self.actions),
            "min_evidence_mass": self.min_evidence_mass,
            "max_normalized_entropy": self.max_normalized_entropy,
            "min_peak_probability": self.min_peak_probability,
            "max_states": self.max_states,
            "max_actions": self.max_actions,
            "max_bytes": self.max_bytes,
            "max_provenance": self.max_provenance,
            "max_regime_changes": self.max_regime_changes,
            "max_evidence_mass": self.max_evidence_mass,
        }

    @property
    def regime_lineage_sha256(self) -> str:
        return _sha256(
            {
                "revision": self._regime_revision,
                "signals": [signal.to_dict() for signal in self._regime_signals],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": WORLD_MODEL_SCHEMA,
            "contract_sha256": self.contract_sha256,
            "states": list(self.states),
            "actions": list(self.actions),
            "counts_sha256": array_sha256(self._counts),
            "event_counts_sha256": array_sha256(self._event_counts),
            "evidence_hashes": list(self.evidence_hashes),
            "verifier_hashes": list(self.verifier_hashes),
            "total_evidence_mass": self.total_evidence_mass,
            "total_events": self.total_events,
            "regime_revision": self._regime_revision,
            "regime_signal_hashes": list(self._regime_signal_hashes),
            "regime_signals": [signal.to_dict() for signal in self._regime_signals],
            "regime_lineage_sha256": self.regime_lineage_sha256,
        }

    @property
    def sha256(self) -> str:
        if self._cached_sha256 is None:
            self._cached_sha256 = _sha256(self.to_dict())
        return self._cached_sha256

    def _invalidate_identity(self) -> None:
        self._cached_sha256 = None

    def clone(self) -> "ActionConditionedWorldModel":
        result = ActionConditionedWorldModel(
            self.states,
            self.actions,
            min_evidence_mass=self.min_evidence_mass,
            max_normalized_entropy=self.max_normalized_entropy,
            min_peak_probability=self.min_peak_probability,
            max_states=self.max_states,
            max_actions=self.max_actions,
            max_bytes=self.max_bytes,
            max_provenance=self.max_provenance,
            max_regime_changes=self.max_regime_changes,
            max_evidence_mass=self.max_evidence_mass,
        )
        result._counts[...] = self._counts
        result._event_counts[...] = self._event_counts
        result._evidence_hashes = set(self._evidence_hashes)
        result._verifier_hashes = set(self._verifier_hashes)
        result._regime_signal_hashes = list(self._regime_signal_hashes)
        result._regime_signals = list(self._regime_signals)
        result._regime_revision = self._regime_revision
        result._cached_sha256 = self._cached_sha256
        return result

    @staticmethod
    def _tensor_descriptor(data: bytes, *, dtype: str) -> dict[str, Any]:
        return {
            "dtype": dtype,
            "byte_count": len(data),
            "raw_sha256": hashlib.sha256(data).hexdigest(),
            "data_base64": base64.b64encode(data).decode("ascii"),
        }

    def to_bytes(
        self,
        *,
        max_payload_bytes: int = MAX_WORLD_MODEL_PAYLOAD_BYTES,
    ) -> bytes:
        """Serialize the complete executable model in one canonical payload."""

        limit = _payload_limit(max_payload_bytes, field="max_payload_bytes")
        if not np.isfinite(self._counts).all() or np.any(self._counts < 0.0):
            raise ValueError("world-model counts must be finite and non-negative")
        if np.any((self._counts == 0.0) & np.signbit(self._counts)):
            raise ValueError("world-model counts contain non-canonical negative zero")
        if self.total_evidence_mass > self.max_evidence_mass:
            raise ValueError("world-model evidence mass exceeds its bound")
        if self.total_events != len(self._evidence_hashes):
            raise ValueError("event counts do not match evidence provenance")
        if len(self._evidence_hashes) > self.max_provenance:
            raise ValueError("evidence provenance exceeds its bound")
        if len(self._verifier_hashes) > self.max_provenance + self.max_regime_changes:
            raise ValueError("verifier provenance exceeds its bound")
        if (
            len(self._regime_signals) != self._regime_revision
            or len(self._regime_signal_hashes) != self._regime_revision
        ):
            raise ValueError("regime revision does not match its signal lineage")
        if self._regime_signal_hashes != [
            signal.signal_sha256 for signal in self._regime_signals
        ]:
            raise ValueError("regime signal hashes do not match signal records")
        if not {signal.verifier_sha256 for signal in self._regime_signals}.issubset(
            self._verifier_hashes
        ):
            raise ValueError("regime verifier is missing from model provenance")

        counts_data = np.asarray(self._counts, dtype="<f8", order="C").tobytes(
            order="C"
        )
        event_data = np.asarray(self._event_counts, dtype="<u8", order="C").tobytes(
            order="C"
        )
        shape = [len(self.actions), len(self.states), len(self.states)]
        payload = canonical_json_bytes(
            {
                "format": WORLD_MODEL_PAYLOAD_SCHEMA,
                "model_sha256": self.sha256,
                "contract_sha256": self.contract_sha256,
                "contract": self._contract_dict(),
                "tensors": {
                    "shape": shape,
                    "counts": self._tensor_descriptor(counts_data, dtype="<f8"),
                    "event_counts": self._tensor_descriptor(event_data, dtype="<u8"),
                },
                "evidence_hashes": list(self.evidence_hashes),
                "verifier_hashes": list(self.verifier_hashes),
                "regime": {
                    "revision": self._regime_revision,
                    "signal_hashes": list(self._regime_signal_hashes),
                    "signals": [signal.to_dict() for signal in self._regime_signals],
                    "lineage_sha256": self.regime_lineage_sha256,
                },
            }
        )
        if len(payload) > limit:
            raise ValueError(
                f"world-model payload requires {len(payload)} bytes, exceeding {limit}"
            )
        return payload

    @staticmethod
    def _decode_tensor(
        value: object,
        *,
        field: str,
        dtype: str,
        expected_bytes: int,
    ) -> bytes:
        if not isinstance(value, dict) or set(value) != {
            "dtype",
            "byte_count",
            "raw_sha256",
            "data_base64",
        }:
            raise WorldModelTamperError(f"invalid {field} tensor descriptor")
        if value["dtype"] != dtype:
            raise WorldModelTamperError(f"invalid {field} tensor dtype")
        byte_count = value["byte_count"]
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count != expected_bytes
        ):
            raise WorldModelTamperError(f"invalid {field} tensor byte count")
        try:
            digest = require_sha256(value["raw_sha256"], field=f"{field}.raw_sha256")
        except ValueError as exc:
            raise WorldModelTamperError(f"invalid {field} tensor hash") from exc
        encoded = value["data_base64"]
        expected_encoded_bytes = 4 * ((expected_bytes + 2) // 3)
        if (
            not isinstance(encoded, str)
            or not encoded.isascii()
            or len(encoded) != expected_encoded_bytes
        ):
            raise WorldModelTamperError(f"invalid {field} tensor encoding length")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise WorldModelTamperError(f"invalid {field} tensor base64") from exc
        if (
            len(decoded) != expected_bytes
            or base64.b64encode(decoded).decode("ascii") != encoded
        ):
            raise WorldModelTamperError(f"non-canonical {field} tensor base64")
        if hashlib.sha256(decoded).hexdigest() != digest:
            raise WorldModelTamperError(f"{field} tensor hash mismatch")
        return decoded

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        max_payload_bytes: int = MAX_WORLD_MODEL_PAYLOAD_BYTES,
        max_states: int = DEFAULT_MAX_NODES,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        max_decoded_bytes: int = DEFAULT_MAX_DENSE_BYTES,
        max_provenance: int = DEFAULT_MAX_PROVENANCE,
        max_regime_changes: int = DEFAULT_MAX_REGIME_CHANGES,
        max_evidence_mass: float = DEFAULT_MAX_EVIDENCE_MASS,
    ) -> "ActionConditionedWorldModel":
        """Restore a complete model after strict structural and hash checks."""

        if not isinstance(data, bytes):
            raise TypeError("world-model payload must be immutable bytes")
        payload_limit = _payload_limit(max_payload_bytes, field="max_payload_bytes")
        if len(data) > payload_limit:
            raise WorldModelTamperError("world-model payload exceeds its byte bound")
        root = _strict_json(data)
        if not isinstance(root, dict) or set(root) != {
            "format",
            "model_sha256",
            "contract_sha256",
            "contract",
            "tensors",
            "evidence_hashes",
            "verifier_hashes",
            "regime",
        }:
            raise WorldModelTamperError("invalid world-model payload structure")
        if root["format"] != WORLD_MODEL_PAYLOAD_SCHEMA:
            raise WorldModelTamperError("unsupported world-model payload format")
        try:
            model_digest = require_sha256(root["model_sha256"], field="model_sha256")
            contract_digest = require_sha256(
                root["contract_sha256"], field="contract_sha256"
            )
        except ValueError as exc:
            raise WorldModelTamperError("invalid world-model identity hash") from exc
        contract = root["contract"]
        expected_contract_fields = {
            "schema",
            "states",
            "actions",
            "min_evidence_mass",
            "max_normalized_entropy",
            "min_peak_probability",
            "max_states",
            "max_actions",
            "max_bytes",
            "max_provenance",
            "max_regime_changes",
            "max_evidence_mass",
        }
        if not isinstance(contract, dict) or set(contract) != expected_contract_fields:
            raise WorldModelTamperError("invalid world-model contract")
        if contract["schema"] != WORLD_MODEL_SCHEMA:
            raise WorldModelTamperError("unsupported world-model contract schema")

        state_cap = _uint(max_states, field="max_states", positive=True)
        action_cap = _uint(max_actions, field="max_actions", positive=True)
        decoded_cap = _uint(max_decoded_bytes, field="max_decoded_bytes", positive=True)
        provenance_cap = _uint(max_provenance, field="max_provenance", positive=True)
        regime_cap = _uint(
            max_regime_changes, field="max_regime_changes", positive=True
        )
        mass_cap = _finite_positive(max_evidence_mass, field="max_evidence_mass")
        try:
            stored_state_cap = _uint(
                contract["max_states"], field="contract.max_states", positive=True
            )
            stored_action_cap = _uint(
                contract["max_actions"], field="contract.max_actions", positive=True
            )
            stored_decoded_cap = _uint(
                contract["max_bytes"], field="contract.max_bytes", positive=True
            )
            stored_provenance_cap = _uint(
                contract["max_provenance"],
                field="contract.max_provenance",
                positive=True,
            )
            stored_regime_cap = _uint(
                contract["max_regime_changes"],
                field="contract.max_regime_changes",
                positive=True,
            )
            stored_mass_cap = _finite_positive(
                contract["max_evidence_mass"],
                field="contract.max_evidence_mass",
            )
        except (TypeError, ValueError) as exc:
            raise WorldModelTamperError("invalid world-model contract bounds") from exc
        if (
            stored_state_cap > state_cap
            or stored_action_cap > action_cap
            or stored_decoded_cap > decoded_cap
            or stored_provenance_cap > provenance_cap
            or stored_regime_cap > regime_cap
            or stored_mass_cap > mass_cap
        ):
            raise WorldModelTamperError("world-model contract exceeds restore bounds")
        try:
            model = cls(
                contract["states"],
                contract["actions"],
                min_evidence_mass=contract["min_evidence_mass"],
                max_normalized_entropy=contract["max_normalized_entropy"],
                min_peak_probability=contract["min_peak_probability"],
                max_states=stored_state_cap,
                max_actions=stored_action_cap,
                max_bytes=stored_decoded_cap,
                max_provenance=stored_provenance_cap,
                max_regime_changes=stored_regime_cap,
                max_evidence_mass=stored_mass_cap,
            )
        except (MemoryError, TypeError, ValueError) as exc:
            raise WorldModelTamperError(
                "world-model contract validation failed"
            ) from exc
        if (
            model._contract_dict() != contract
            or model.contract_sha256 != contract_digest
        ):
            raise WorldModelTamperError("world-model contract hash mismatch")

        tensors = root["tensors"]
        if not isinstance(tensors, dict) or set(tensors) != {
            "shape",
            "counts",
            "event_counts",
        }:
            raise WorldModelTamperError("invalid world-model tensors")
        expected_shape = [len(model.actions), len(model.states), len(model.states)]
        if tensors["shape"] != expected_shape:
            raise WorldModelTamperError("world-model tensor shape mismatch")
        decoded_bytes = math.prod(expected_shape) * 8
        if decoded_bytes * 2 > stored_decoded_cap:
            raise WorldModelTamperError(
                "world-model tensors exceed contract byte bound"
            )
        counts_data = cls._decode_tensor(
            tensors["counts"],
            field="counts",
            dtype="<f8",
            expected_bytes=decoded_bytes,
        )
        event_data = cls._decode_tensor(
            tensors["event_counts"],
            field="event_counts",
            dtype="<u8",
            expected_bytes=decoded_bytes,
        )
        counts = np.frombuffer(counts_data, dtype="<f8").reshape(expected_shape)
        if not np.isfinite(counts).all() or np.any(counts < 0.0):
            raise WorldModelTamperError("counts tensor is non-finite or negative")
        if np.any((counts == 0.0) & np.signbit(counts)):
            raise WorldModelTamperError("counts tensor contains negative zero")
        event_counts = np.frombuffer(event_data, dtype="<u8").reshape(expected_shape)

        def restore_hashes(
            value: object,
            *,
            field: str,
            limit: int,
            sorted_set: bool = True,
        ) -> tuple[str, ...]:
            if not isinstance(value, list) or len(value) > limit:
                raise WorldModelTamperError(f"invalid {field}")
            try:
                result = tuple(
                    require_sha256(digest, field=f"{field} digest") for digest in value
                )
            except ValueError as exc:
                raise WorldModelTamperError(f"invalid {field}") from exc
            if len(set(result)) != len(result):
                raise WorldModelTamperError(f"{field} must be unique")
            if sorted_set and tuple(sorted(result)) != result:
                raise WorldModelTamperError(f"{field} must be sorted")
            return result

        evidence_hashes = restore_hashes(
            root["evidence_hashes"],
            field="evidence_hashes",
            limit=stored_provenance_cap,
        )
        verifier_hashes = restore_hashes(
            root["verifier_hashes"],
            field="verifier_hashes",
            limit=stored_provenance_cap + stored_regime_cap,
        )
        nonzero_events = event_counts[event_counts > 0]
        if nonzero_events.size > stored_provenance_cap:
            raise WorldModelTamperError("event counts exceed provenance bound")
        event_total = 0
        for count in nonzero_events:
            event_total += int(count)
            if event_total > stored_provenance_cap:
                raise WorldModelTamperError("event counts exceed provenance bound")
        if event_total != len(evidence_hashes):
            raise WorldModelTamperError(
                "event counts do not match unique evidence provenance"
            )

        regime = root["regime"]
        if not isinstance(regime, dict) or set(regime) != {
            "revision",
            "signal_hashes",
            "signals",
            "lineage_sha256",
        }:
            raise WorldModelTamperError("invalid regime lineage")
        try:
            revision = _uint(regime["revision"], field="regime.revision")
        except ValueError as exc:
            raise WorldModelTamperError("invalid regime revision") from exc
        if revision > stored_regime_cap:
            raise WorldModelTamperError("regime revision exceeds its bound")
        signal_hashes = restore_hashes(
            regime["signal_hashes"],
            field="regime.signal_hashes",
            limit=stored_regime_cap,
            sorted_set=False,
        )
        raw_signals = regime["signals"]
        if not isinstance(raw_signals, list) or len(raw_signals) != revision:
            raise WorldModelTamperError("regime signals do not match revision")
        signals: list[RegimeChangeSignal] = []
        expected_signal_fields = {
            "schema",
            "signal_sha256",
            "verifier_sha256",
            "strength",
            "retention",
            "epoch",
            "name",
        }
        for raw_signal in raw_signals:
            if (
                not isinstance(raw_signal, dict)
                or set(raw_signal) != expected_signal_fields
                or raw_signal["schema"] != "immer-ooe-regime-change-signal/v1"
            ):
                raise WorldModelTamperError("invalid regime signal record")
            try:
                signal = RegimeChangeSignal(
                    signal_sha256=raw_signal["signal_sha256"],
                    verifier_sha256=raw_signal["verifier_sha256"],
                    strength=raw_signal["strength"],
                    retention=raw_signal["retention"],
                    epoch=raw_signal["epoch"],
                    name=raw_signal["name"],
                )
            except (TypeError, ValueError) as exc:
                raise WorldModelTamperError("invalid regime signal record") from exc
            signals.append(signal)
        if tuple(signal.signal_sha256 for signal in signals) != signal_hashes:
            raise WorldModelTamperError("regime signal hash sequence mismatch")
        if not {signal.verifier_sha256 for signal in signals}.issubset(verifier_hashes):
            raise WorldModelTamperError("regime verifier provenance is incomplete")

        model._counts[...] = np.asarray(counts, dtype=np.float64)
        model._event_counts[...] = np.asarray(event_counts, dtype=np.uint64)
        model._evidence_hashes = set(evidence_hashes)
        model._verifier_hashes = set(verifier_hashes)
        model._regime_revision = revision
        model._regime_signal_hashes = list(signal_hashes)
        model._regime_signals = signals
        model._invalidate_identity()
        if model.total_evidence_mass > model.max_evidence_mass:
            raise WorldModelTamperError("restored evidence mass exceeds its bound")
        try:
            lineage_digest = require_sha256(
                regime["lineage_sha256"], field="regime.lineage_sha256"
            )
        except ValueError as exc:
            raise WorldModelTamperError("invalid regime lineage hash") from exc
        if model.regime_lineage_sha256 != lineage_digest:
            raise WorldModelTamperError("regime lineage hash mismatch")
        if model.sha256 != model_digest:
            raise WorldModelTamperError("world-model semantic hash mismatch")
        if model.to_bytes(max_payload_bytes=payload_limit) != data:
            raise WorldModelTamperError(
                "world-model payload failed canonical roundtrip"
            )
        return model

    def observe(self, evidence: TransitionEvidence) -> None:
        if not isinstance(evidence, TransitionEvidence):
            raise TypeError("evidence must be TransitionEvidence")
        source = self._state(evidence.source_state, field="source state")
        action = self._action(evidence.action)
        target = self._state(evidence.target_state, field="target state")
        if evidence.evidence_sha256 in self._evidence_hashes:
            raise ValueError("duplicate transition evidence")
        if len(self._evidence_hashes) >= self.max_provenance:
            raise ValueError("transition evidence provenance limit reached")
        new_mass = self.total_evidence_mass + evidence.weight
        if new_mass > self.max_evidence_mass:
            raise ValueError("world-model evidence mass limit reached")
        current_events = int(self._event_counts[action, source, target])
        if current_events == np.iinfo(np.uint64).max:
            raise OverflowError("transition event count overflow")
        # All validation precedes this explicit state transition.
        self._counts[action, source, target] += evidence.weight
        self._event_counts[action, source, target] += np.uint64(1)
        self._evidence_hashes.add(evidence.evidence_sha256)
        self._verifier_hashes.add(evidence.verifier_sha256)
        self._invalidate_identity()

    def observe_many(self, evidence: Iterable[TransitionEvidence]) -> None:
        records = tuple(evidence)
        staged = self.clone()
        for record in records:
            staged.observe(record)
        self._counts[...] = staged._counts
        self._event_counts[...] = staged._event_counts
        self._evidence_hashes = staged._evidence_hashes
        self._verifier_hashes = staged._verifier_hashes
        self._cached_sha256 = staged._cached_sha256

    def row_evidence_mass(self, source_state: str, action: str) -> float:
        source = self._state(source_state, field="source state")
        action_index = self._action(action)
        return float(self._counts[action_index, source].sum())

    def row_event_count(self, source_state: str, action: str) -> int:
        source = self._state(source_state, field="source state")
        action_index = self._action(action)
        return int(self._event_counts[action_index, source].sum(dtype=np.uint64))

    def transition_row(self, source_state: str, action: str) -> FloatArray:
        source = self._state(source_state, field="source state")
        action_index = self._action(action)
        return _normalize_row(self._counts[action_index, source])

    def action_kernel(self, action: str) -> FloatArray:
        """Return an exactly row-normalized kernel; unknown rows are uniform.

        Unknown rows remain epistemically distinguishable through the separate
        count tensor and are rejected by :meth:`predict` and the planner.
        """

        action_index = self._action(action)
        result = np.empty((len(self.states), len(self.states)), dtype=np.float64)
        uniform = np.full(len(self.states), 1.0 / len(self.states), dtype=np.float64)
        for source in range(len(self.states)):
            row = self._counts[action_index, source]
            result[source] = uniform if float(row.sum()) == 0.0 else _normalize_row(row)
        return np.ascontiguousarray(result)

    def predict(
        self,
        source_state: str,
        action: str,
        *,
        min_evidence_mass: float | None = None,
        max_normalized_entropy: float | None = None,
        min_peak_probability: float | None = None,
    ) -> TransitionPrediction:
        source = self._state(source_state, field="source state")
        action_index = self._action(action)
        mass_threshold = (
            self.min_evidence_mass
            if min_evidence_mass is None
            else _finite_positive(min_evidence_mass, field="min_evidence_mass")
        )
        entropy_threshold = (
            self.max_normalized_entropy
            if max_normalized_entropy is None
            else _probability(max_normalized_entropy, field="max_normalized_entropy")
        )
        peak_threshold = (
            self.min_peak_probability
            if min_peak_probability is None
            else _probability(min_peak_probability, field="min_peak_probability")
        )
        row = self._counts[action_index, source]
        mass = float(row.sum())
        events = int(self._event_counts[action_index, source].sum(dtype=np.uint64))
        if mass > 0.0:
            probabilities = _normalize_row(row)
        else:
            probabilities = np.full(
                len(self.states), 1.0 / len(self.states), dtype=np.float64
            )
        positive = probabilities[probabilities > 0.0]
        entropy = float(-np.sum(positive * np.log(positive)))
        if len(self.states) > 1:
            entropy /= math.log(len(self.states))
        peak = float(probabilities.max())
        coverage = min(1.0, mass / mass_threshold)
        confidence = float(coverage * peak * (1.0 - entropy))
        reason: str | None = None
        if mass == 0.0:
            reason = "novel-transition"
        elif mass < mass_threshold:
            reason = "insufficient-evidence"
        elif entropy > entropy_threshold:
            reason = "transition-entropy"
        elif peak < peak_threshold:
            reason = "transition-ambiguity"
        row_hash = _sha256(
            {
                "source_state": source_state,
                "action": action,
                "probabilities_sha256": array_sha256(probabilities),
                "evidence_mass": mass,
                "event_count": events,
                "regime_revision": self._regime_revision,
            }
        )
        return TransitionPrediction(
            source_state=source_state,
            action=action,
            probabilities=tuple(float(value) for value in probabilities),
            state_labels=self.states,
            evidence_mass=mass,
            event_count=events,
            coverage=coverage,
            normalized_entropy=entropy,
            peak_probability=peak,
            confidence=confidence,
            abstained=reason is not None,
            reason=reason,
            row_sha256=row_hash,
            world_model_sha256=self.sha256,
        )

    def apply_regime_change(self, signal: RegimeChangeSignal) -> RegimeChangeReceipt:
        if not isinstance(signal, RegimeChangeSignal):
            raise TypeError("signal must be RegimeChangeSignal")
        if signal.signal_sha256 in self._regime_signal_hashes:
            raise ValueError("duplicate regime-change signal")
        if len(self._regime_signal_hashes) >= self.max_regime_changes:
            raise ValueError("regime-change history limit reached")
        before_model = self.sha256
        before_counts = array_sha256(self._counts)
        before_mass = self.total_evidence_mass
        updated = np.ascontiguousarray(self._counts * signal.retention)
        if not np.isfinite(updated).all() or np.any(updated < 0.0):
            raise ArithmeticError("regime decay produced invalid evidence")
        self._counts[...] = updated
        self._regime_revision += 1
        self._regime_signal_hashes.append(signal.signal_sha256)
        self._regime_signals.append(signal)
        self._verifier_hashes.add(signal.verifier_sha256)
        self._invalidate_identity()
        after_model = self.sha256
        return RegimeChangeReceipt(
            signal=signal,
            revision=self._regime_revision,
            before_model_sha256=before_model,
            after_model_sha256=after_model,
            before_counts_sha256=before_counts,
            after_counts_sha256=array_sha256(self._counts),
            before_evidence_mass=before_mass,
            after_evidence_mass=self.total_evidence_mass,
        )

    @classmethod
    def fuse_ps_lifted(
        cls,
        replicas: Sequence["ActionConditionedWorldModel"],
        adjacency: NDArray[np.floating],
        *,
        tolerance: float = 1e-9,
        max_rounds: int = 4096,
        topology: str = "ps-lifted-z2-world",
        pc: float | None = None,
        ps: float = 0.003,
    ) -> FusedWorldModel:
        if isinstance(replicas, (str, bytes)) or not isinstance(replicas, Sequence):
            raise TypeError("replicas must be a sequence")
        nodes = len(replicas)
        if nodes < 2:
            raise ValueError("PS-Lifted fusion requires at least two replicas")
        if any(not isinstance(model, cls) for model in replicas):
            raise TypeError("every replica must be an ActionConditionedWorldModel")
        first = replicas[0]
        if nodes > first.max_states:
            raise ValueError("replica count exceeds the configured node bound")
        if any(model.contract_sha256 != first.contract_sha256 for model in replicas):
            raise ValueError("replica world-model contracts differ")
        if any(
            model.regime_lineage_sha256 != first.regime_lineage_sha256
            for model in replicas
        ):
            raise ValueError("replica regime lineages differ")
        threshold = _finite_positive(tolerance, field="tolerance")
        if threshold * nodes >= 0.25:
            raise ValueError(
                "tolerance must keep aggregate event-count error below 0.25"
            )
        rounds_limit = _uint(max_rounds, field="max_rounds", positive=True)
        topology_name = _label(topology, field="topology")
        seen_evidence: set[str] = set()
        for model in replicas:
            overlap = seen_evidence.intersection(model._evidence_hashes)
            if overlap:
                raise ValueError("replicas contain duplicate transition evidence")
            seen_evidence.update(model._evidence_hashes)
        if len(seen_evidence) > first.max_provenance:
            raise ValueError("fused provenance exceeds its configured bound")

        graph = validate_adjacency(
            adjacency,
            max_nodes=first.max_states,
            max_bytes=first.max_bytes,
        )
        if graph.shape != (nodes, nodes):
            raise ValueError("adjacency size must equal replica count")
        adaptive = pc is None
        if adaptive:
            lifted, lift_parameters = adaptive_ps_lift_matrix(
                graph,
                ps=ps,
                max_nodes=first.max_states,
                max_bytes=first.max_bytes,
            )
            selected_pc = lift_parameters.pc
        else:
            lift_parameters = adaptive_lift_parameters(
                graph,
                ps=ps,
                max_nodes=first.max_states,
                max_bytes=first.max_bytes,
            )
            selected_pc = float(pc)
            lifted = ps_lifted_matrix(
                graph,
                pc=selected_pc,
                ps=ps,
                max_nodes=first.max_states,
                max_bytes=first.max_bytes,
            )
        count_width = int(first._counts.size)
        event_width = int(first._event_counts.size)
        payload_bytes = nodes * (count_width + event_width) * 8
        if payload_bytes > first.max_bytes:
            raise ValueError(
                f"fusion payload requires {payload_bytes} bytes, exceeding "
                f"{first.max_bytes}"
            )
        payload = np.empty((nodes, count_width + event_width), dtype=np.float64)
        for node, model in enumerate(replicas):
            payload[node, :count_width] = model._counts.reshape(-1)
            payload[node, count_width:] = model._event_counts.reshape(-1)
        result = measure_consensus(
            lifted,
            payload,
            tolerance=threshold,
            max_rounds=rounds_limit,
            lifted_nodes=nodes,
            topology=topology_name,
            max_nodes=first.max_states,
            max_bytes=first.max_bytes,
        )
        # Entry zero is the actual distributed estimate.  Multiplication by the
        # known replica cardinality converts push-sum's mean into the aggregate;
        # no central count summation is used.
        aggregate = np.ascontiguousarray(result.estimates[0] * nodes)
        negative = aggregate < 0.0
        if np.any(aggregate[negative] < -max(threshold * nodes, 1e-12)):
            raise ArithmeticError("consensus produced negative sufficient statistics")
        aggregate[negative] = 0.0
        fused = first.clone()
        fused._counts[...] = aggregate[:count_width].reshape(first._counts.shape)
        raw_events = aggregate[count_width:].reshape(first._event_counts.shape)
        rounded_events = np.rint(raw_events)
        if np.max(np.abs(raw_events - rounded_events), initial=0.0) > max(
            threshold * nodes * 4.0, 1e-7
        ):
            raise ArithmeticError("consensus event counts did not recover integers")
        if np.any(rounded_events > np.iinfo(np.uint64).max):
            raise OverflowError("fused transition event count overflow")
        fused._event_counts[...] = rounded_events.astype(np.uint64)
        fused._evidence_hashes = seen_evidence
        fused._verifier_hashes = set().union(
            *(model._verifier_hashes for model in replicas)
        )
        if fused.total_events != len(seen_evidence):
            raise ArithmeticError(
                "fused event counts do not match unique evidence provenance"
            )
        fused._invalidate_identity()
        if fused.total_evidence_mass > fused.max_evidence_mass:
            raise ValueError("fused evidence mass exceeds its configured bound")
        receipt = WorldFusionReceipt(
            topology=topology_name,
            adjacency_sha256=array_sha256(graph),
            lifted_transition_sha256=array_sha256(lifted),
            replica_model_sha256s=tuple(model.sha256 for model in replicas),
            fused_model_sha256=fused.sha256,
            counts_sha256=array_sha256(fused._counts),
            event_counts_sha256=array_sha256(fused._event_counts),
            consensus=result.receipt,
            lift_parameters_sha256=(
                lift_parameters.sha256
                if adaptive
                else _sha256(
                    {
                        "adjacency_sha256": lift_parameters.adjacency_sha256,
                        "adaptive": False,
                        "fiedler_eigenvalue": lift_parameters.fiedler_eigenvalue,
                        "pc": selected_pc,
                        "ps": float(ps),
                    }
                )
            ),
            fiedler_eigenvalue=lift_parameters.fiedler_eigenvalue,
            pc=selected_pc,
            ps=float(ps),
            adaptive_lift=adaptive,
        )
        return FusedWorldModel(model=fused, receipt=receipt)
