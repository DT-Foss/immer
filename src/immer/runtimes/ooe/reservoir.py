"""Continuous PS-Lifted reservoirs with distributed closed-form learning.

The finite action world model is the exact discrete path.  This module adds a
continuous path whose recurrent body is fixed by a PS-Lifted Markov operator.
Only sufficient statistics and a small ridge readout learn.  Separate agents
can therefore fuse what they learned through the same PS-Lifted consensus used
by the discrete world model, without exchanging their raw sequences.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .consensus import (
    ConsensusReceipt,
    measure_consensus,
    ps_lifted_matrix,
    validate_adjacency,
)
from .identity import canonical_json_bytes, require_sha256
from .math_core import array_sha256


RESERVOIR_SCHEMA = "immer-ooe-ps-lifted-reservoir/v1"
RESERVOIR_PAYLOAD_SCHEMA = "immer-ooe-ps-lifted-reservoir-payload/v1"
RESERVOIR_FUSION_SCHEMA = "immer-ooe-ps-lifted-reservoir-fusion/v1"
MAX_RESERVOIR_NODES = 512
MAX_RESERVOIR_INPUTS = 16_384
MAX_RESERVOIR_OUTPUTS = 16_384
MAX_RESERVOIR_FEATURES = 65_536
MAX_RESERVOIR_EVIDENCE = 1_000_000
MAX_RESERVOIR_PAYLOAD_BYTES = 512 * 1024 * 1024

FloatArray = NDArray[np.float64]


class ReservoirIntegrityError(ValueError):
    """Raised when a reservoir artifact is malformed or hash-inconsistent."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _integer(
    value: object,
    *,
    name: str,
    minimum: int = 1,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not minimum <= int(value) <= maximum
    ):
        raise ValueError(f"{name} must lie in [{minimum}, {maximum}]")
    return int(value)


def _finite(
    value: object,
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _vector(value: ArrayLike, *, size: int, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size != size:
        raise ValueError(f"{name} must be a vector of length {size}")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must be finite")
    result = np.ascontiguousarray(array)
    result[result == 0.0] = 0.0
    return result


def _matrix(
    value: ArrayLike,
    *,
    shape: tuple[int, int],
    name: str,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} must be a finite matrix of shape {shape}")
    result = np.ascontiguousarray(array)
    result[result == 0.0] = 0.0
    return result


def _matvec(matrix: FloatArray, vector: FloatArray) -> FloatArray:
    # ``optimize=False`` avoids platform BLAS kernels that have emitted bogus
    # overflow warnings for small, finite matrices on macOS Accelerate.
    return np.einsum("ij,j->i", matrix, vector, optimize=False)


def _outer(left: FloatArray, right: FloatArray) -> FloatArray:
    return np.einsum("i,j->ij", left, right, optimize=False)


def _array_descriptor(value: FloatArray) -> dict[str, object]:
    data = np.asarray(value, dtype="<f8", order="C").tobytes(order="C")
    return {
        "byte_count": len(data),
        "data_base64": base64.b64encode(data).decode("ascii"),
        "dtype": "<f8",
        "raw_sha256": hashlib.sha256(data).hexdigest(),
        "shape": list(value.shape),
    }


def _decode_array(
    value: object,
    *,
    shape: tuple[int, ...],
    name: str,
    max_bytes: int,
) -> FloatArray:
    if not isinstance(value, dict) or set(value) != {
        "byte_count",
        "data_base64",
        "dtype",
        "raw_sha256",
        "shape",
    }:
        raise ReservoirIntegrityError(f"invalid {name} descriptor")
    if value["dtype"] != "<f8" or value["shape"] != list(shape):
        raise ReservoirIntegrityError(f"invalid {name} shape or dtype")
    expected = math.prod(shape) * 8
    if expected > max_bytes or value["byte_count"] != expected:
        raise ReservoirIntegrityError(f"invalid {name} byte count")
    encoded = value["data_base64"]
    if (
        not isinstance(encoded, str)
        or not encoded.isascii()
        or len(encoded) != 4 * ((expected + 2) // 3)
    ):
        raise ReservoirIntegrityError(f"invalid {name} encoding")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ReservoirIntegrityError(f"invalid {name} base64") from exc
    try:
        digest = require_sha256(value["raw_sha256"], field=f"{name}.raw_sha256")
    except ValueError as exc:
        raise ReservoirIntegrityError(f"invalid {name} digest") from exc
    if len(data) != expected or hashlib.sha256(data).hexdigest() != digest:
        raise ReservoirIntegrityError(f"{name} digest mismatch")
    result = np.frombuffer(data, dtype="<f8").reshape(shape).copy()
    if not bool(np.isfinite(result).all()):
        raise ReservoirIntegrityError(f"{name} contains non-finite values")
    if bool(np.any((result == 0.0) & np.signbit(result))):
        raise ReservoirIntegrityError(f"{name} contains negative zero")
    return np.ascontiguousarray(result)


@dataclass(frozen=True, slots=True)
class PSLiftedReservoirConfig:
    input_size: int
    output_size: int
    nodes: int = 24
    seed: int = 42
    edge_probability: float = 0.2
    pc: float = 0.8
    ps: float = 0.003
    spectral_radius: float = 0.95
    input_scale: float = 0.2
    leak_rate: float = 1.0
    ridge: float = 1e-3

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_size",
            _integer(
                self.input_size,
                name="input_size",
                maximum=MAX_RESERVOIR_INPUTS,
            ),
        )
        object.__setattr__(
            self,
            "output_size",
            _integer(
                self.output_size,
                name="output_size",
                maximum=MAX_RESERVOIR_OUTPUTS,
            ),
        )
        object.__setattr__(
            self,
            "nodes",
            _integer(
                self.nodes,
                name="nodes",
                minimum=4,
                maximum=MAX_RESERVOIR_NODES,
            ),
        )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, (int, np.integer))
            or not 0 <= int(self.seed) <= 2**63 - 1
        ):
            raise ValueError("seed must be a non-negative signed 64-bit integer")
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(
            self,
            "edge_probability",
            _finite(
                self.edge_probability,
                name="edge_probability",
                strictly_positive=True,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "pc",
            _finite(self.pc, name="pc", strictly_positive=True, maximum=1.0),
        )
        object.__setattr__(
            self,
            "ps",
            _finite(self.ps, name="ps", strictly_positive=True, maximum=1.0),
        )
        if self.pc + self.ps >= 1.0:
            raise ValueError("pc + ps must be smaller than one")
        object.__setattr__(
            self,
            "spectral_radius",
            _finite(
                self.spectral_radius,
                name="spectral_radius",
                strictly_positive=True,
                maximum=0.999999999,
            ),
        )
        object.__setattr__(
            self,
            "input_scale",
            _finite(self.input_scale, name="input_scale", strictly_positive=True),
        )
        object.__setattr__(
            self,
            "leak_rate",
            _finite(
                self.leak_rate,
                name="leak_rate",
                strictly_positive=True,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "ridge",
            _finite(self.ridge, name="ridge", strictly_positive=True),
        )
        if self.feature_size > MAX_RESERVOIR_FEATURES:
            raise ValueError("reservoir feature vector exceeds its hard bound")

    @property
    def state_size(self) -> int:
        return 2 * self.nodes

    @property
    def feature_size(self) -> int:
        # Both lifted layers, their Z2 parity, direct input, and an intercept.
        return 3 * self.nodes + self.input_size + 1

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_probability": self.edge_probability,
            "input_scale": self.input_scale,
            "input_size": self.input_size,
            "leak_rate": self.leak_rate,
            "nodes": self.nodes,
            "output_size": self.output_size,
            "pc": self.pc,
            "ps": self.ps,
            "ridge": self.ridge,
            "schema": RESERVOIR_SCHEMA,
            "seed": self.seed,
            "spectral_radius": self.spectral_radius,
        }

    @classmethod
    def from_dict(cls, value: object) -> "PSLiftedReservoirConfig":
        expected = {
            "edge_probability",
            "input_scale",
            "input_size",
            "leak_rate",
            "nodes",
            "output_size",
            "pc",
            "ps",
            "ridge",
            "schema",
            "seed",
            "spectral_radius",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or value.get("schema") != RESERVOIR_SCHEMA
        ):
            raise ReservoirIntegrityError("invalid reservoir configuration")
        try:
            return cls(
                input_size=value["input_size"],
                output_size=value["output_size"],
                nodes=value["nodes"],
                seed=value["seed"],
                edge_probability=value["edge_probability"],
                pc=value["pc"],
                ps=value["ps"],
                spectral_radius=value["spectral_radius"],
                input_scale=value["input_scale"],
                leak_rate=value["leak_rate"],
                ridge=value["ridge"],
            )
        except (TypeError, ValueError) as exc:
            raise ReservoirIntegrityError("reservoir configuration failed") from exc

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class ReservoirPrediction:
    values: tuple[float, ...]
    selected: int
    margin: float
    parity_surprise: float
    state_sha256: str
    readout_sha256: str


@dataclass(frozen=True, slots=True)
class ReservoirFusionReceipt:
    core_sha256: str
    replica_sha256s: tuple[str, ...]
    evidence_sha256s: tuple[str, ...]
    consensus: ConsensusReceipt
    fused_statistics_sha256: str
    sample_count: int
    evidence_mass: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "core_sha256",
            require_sha256(self.core_sha256, field="core_sha256"),
        )
        replicas = tuple(
            require_sha256(value, field="replica_sha256")
            for value in self.replica_sha256s
        )
        if not replicas or tuple(sorted(replicas)) != replicas:
            raise ValueError("replica hashes must be non-empty and sorted")
        object.__setattr__(self, "replica_sha256s", replicas)
        evidence = tuple(
            require_sha256(value, field="evidence_sha256")
            for value in self.evidence_sha256s
        )
        if tuple(sorted(evidence)) != evidence or len(set(evidence)) != len(evidence):
            raise ValueError("evidence hashes must be sorted and unique")
        object.__setattr__(self, "evidence_sha256s", evidence)
        object.__setattr__(
            self,
            "fused_statistics_sha256",
            require_sha256(
                self.fused_statistics_sha256,
                field="fused_statistics_sha256",
            ),
        )
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 0
        ):
            raise ValueError("sample_count must be non-negative")
        object.__setattr__(
            self,
            "evidence_mass",
            _finite(
                self.evidence_mass,
                name="evidence_mass",
                minimum=0.0,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "consensus": self.consensus.to_dict(),
            "core_sha256": self.core_sha256,
            "evidence_mass": self.evidence_mass,
            "evidence_sha256s": list(self.evidence_sha256s),
            "fused_statistics_sha256": self.fused_statistics_sha256,
            "replica_sha256s": list(self.replica_sha256s),
            "sample_count": self.sample_count,
            "schema": RESERVOIR_FUSION_SCHEMA,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class FusedReservoir:
    agent: "PSLiftedReservoirAgent"
    receipt: ReservoirFusionReceipt


class PSLiftedReservoirAgent:
    """Fixed PS-Lifted recurrent body with an online closed-form readout."""

    def __init__(self, config: PSLiftedReservoirConfig) -> None:
        if not isinstance(config, PSLiftedReservoirConfig):
            raise TypeError("config must be a PSLiftedReservoirConfig")
        self.config = config
        self._adjacency, self._transition, self._input_weights = self._build_core(
            config
        )
        self._state = np.zeros(config.state_size, dtype=np.float64)
        self._xtx = np.zeros(
            (config.feature_size, config.feature_size), dtype=np.float64
        )
        self._xty = np.zeros(
            (config.feature_size, config.output_size), dtype=np.float64
        )
        self._readout = np.zeros(
            (config.feature_size, config.output_size), dtype=np.float64
        )
        self._evidence_sha256s: set[str] = set()
        self._sample_count = 0
        self._evidence_mass = 0.0

    @staticmethod
    def _build_core(
        config: PSLiftedReservoirConfig,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        rng = np.random.default_rng(config.seed)
        nodes = config.nodes
        random_upper = rng.random((nodes, nodes))
        adjacency = np.triu(
            (random_upper < config.edge_probability).astype(np.float64), 1
        )
        adjacency += adjacency.T
        # A deterministic chain guarantees connectivity without a retry loop.
        for index in range(nodes - 1):
            adjacency[index, index + 1] = 1.0
            adjacency[index + 1, index] = 1.0
        adjacency = validate_adjacency(adjacency)
        transition = ps_lifted_matrix(
            adjacency,
            pc=config.pc,
            ps=config.ps,
        )
        input_weights = rng.normal(
            0.0,
            config.input_scale,
            size=(config.state_size, config.input_size),
        )
        input_weights[input_weights == 0.0] = 0.0
        return (
            np.ascontiguousarray(adjacency),
            np.ascontiguousarray(transition),
            np.ascontiguousarray(input_weights),
        )

    @property
    def state(self) -> FloatArray:
        value = self._state.copy()
        value.flags.writeable = False
        return value

    @property
    def readout(self) -> FloatArray:
        value = self._readout.copy()
        value.flags.writeable = False
        return value

    @property
    def evidence_sha256s(self) -> tuple[str, ...]:
        return tuple(sorted(self._evidence_sha256s))

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def evidence_mass(self) -> float:
        return self._evidence_mass

    @property
    def core_sha256(self) -> str:
        return _digest(
            {
                "adjacency_sha256": array_sha256(self._adjacency),
                "config_sha256": self.config.sha256,
                "input_weights_sha256": array_sha256(self._input_weights),
                "transition_sha256": array_sha256(self._transition),
            }
        )

    @property
    def statistics_sha256(self) -> str:
        return _digest(
            {
                "evidence_mass": self._evidence_mass,
                "evidence_sha256s": list(self.evidence_sha256s),
                "sample_count": self._sample_count,
                "xtx_sha256": array_sha256(self._xtx),
                "xty_sha256": array_sha256(self._xty),
            }
        )

    @property
    def readout_sha256(self) -> str:
        return array_sha256(self._readout)

    @property
    def state_sha256(self) -> str:
        return array_sha256(self._state)

    @property
    def sha256(self) -> str:
        return _digest(
            {
                "core_sha256": self.core_sha256,
                "readout_sha256": self.readout_sha256,
                "state_sha256": self.state_sha256,
                "statistics_sha256": self.statistics_sha256,
            }
        )

    def reset_state(self) -> None:
        self._state.fill(0.0)

    def advance(self, inputs: ArrayLike) -> FloatArray:
        value = _vector(
            inputs,
            size=self.config.input_size,
            name="reservoir input",
        )
        propagated = _matvec(self._transition.T, self._state)
        injected = _matvec(self._input_weights, value)
        candidate = np.tanh(
            self.config.spectral_radius * propagated + injected
        )
        self._state = np.ascontiguousarray(
            (1.0 - self.config.leak_rate) * self._state
            + self.config.leak_rate * candidate
        )
        nodes = self.config.nodes
        parity = self._state[:nodes] - self._state[nodes:]
        feature = np.concatenate(
            (self._state, parity, value, np.ones(1, dtype=np.float64))
        )
        feature[feature == 0.0] = 0.0
        return np.ascontiguousarray(feature)

    def predict(
        self,
        inputs: ArrayLike,
        *,
        advance: bool = True,
    ) -> ReservoirPrediction:
        if advance:
            feature = self.advance(inputs)
        else:
            value = _vector(
                inputs,
                size=self.config.input_size,
                name="reservoir input",
            )
            nodes = self.config.nodes
            feature = np.concatenate(
                (
                    self._state,
                    self._state[:nodes] - self._state[nodes:],
                    value,
                    np.ones(1, dtype=np.float64),
                )
            )
        scores = _matvec(self._readout.T, feature)
        order = np.argsort(scores, kind="stable")
        selected = int(order[-1])
        margin = (
            float(scores[order[-1]] - scores[order[-2]])
            if scores.size > 1
            else math.inf
        )
        parity = self._state[: self.config.nodes] - self._state[self.config.nodes :]
        parity_surprise = float(
            np.linalg.norm(parity) / math.sqrt(self.config.nodes)
        )
        return ReservoirPrediction(
            values=tuple(float(value) for value in scores),
            selected=selected,
            margin=margin,
            parity_surprise=parity_surprise,
            state_sha256=self.state_sha256,
            readout_sha256=self.readout_sha256,
        )

    def observe(
        self,
        inputs: ArrayLike,
        target: ArrayLike,
        *,
        evidence_sha256: str,
        weight: float = 1.0,
        solve: bool = True,
    ) -> FloatArray:
        digest = require_sha256(evidence_sha256, field="evidence_sha256")
        if digest in self._evidence_sha256s:
            raise ValueError("duplicate reservoir evidence")
        if len(self._evidence_sha256s) >= MAX_RESERVOIR_EVIDENCE:
            raise ValueError("reservoir evidence bound exceeded")
        mass = _finite(weight, name="weight", strictly_positive=True)
        output = _vector(
            target,
            size=self.config.output_size,
            name="reservoir target",
        )
        feature = self.advance(inputs)
        self._xtx += mass * _outer(feature, feature)
        self._xty += mass * _outer(feature, output)
        self._xtx[self._xtx == 0.0] = 0.0
        self._xty[self._xty == 0.0] = 0.0
        self._evidence_sha256s.add(digest)
        self._sample_count += 1
        self._evidence_mass += mass
        if solve:
            self.solve_readout()
        return feature

    def solve_readout(self) -> FloatArray:
        regularized = self._xtx.copy()
        diagonal = np.diag_indices_from(regularized)
        regularized[diagonal] += self.config.ridge
        try:
            readout = np.linalg.solve(regularized, self._xty)
        except np.linalg.LinAlgError:
            readout = np.linalg.lstsq(regularized, self._xty, rcond=None)[0]
        if not bool(np.isfinite(readout).all()):
            raise ArithmeticError("ridge solution produced non-finite readout")
        readout = np.ascontiguousarray(readout, dtype=np.float64)
        readout[readout == 0.0] = 0.0
        self._readout = readout
        value = self._readout.copy()
        value.flags.writeable = False
        return value

    def apply_retention(self, retention: float) -> None:
        """Decay learned evidence while preserving the fixed Markov body."""

        value = _finite(
            retention,
            name="retention",
            minimum=0.0,
            maximum=1.0,
        )
        self._xtx *= value
        self._xty *= value
        self._evidence_mass *= value
        self.solve_readout()

    def _statistics_vector(self) -> FloatArray:
        return np.ascontiguousarray(
            np.concatenate((self._xtx.ravel(), self._xty.ravel()))
        )

    @classmethod
    def fuse_ps_lifted(
        cls,
        agents: Sequence["PSLiftedReservoirAgent"],
        adjacency: ArrayLike,
        *,
        tolerance: float = 1e-9,
        max_rounds: int = 4096,
        topology: str = "reservoir-ps-lifted",
    ) -> FusedReservoir:
        replicas = tuple(agents)
        if len(replicas) < 2 or any(
            not isinstance(agent, PSLiftedReservoirAgent) for agent in replicas
        ):
            raise ValueError("at least two reservoir agents are required")
        core = replicas[0].core_sha256
        if any(agent.core_sha256 != core for agent in replicas[1:]):
            raise ValueError("reservoir cores differ")
        all_evidence: list[str] = []
        for agent in replicas:
            all_evidence.extend(agent.evidence_sha256s)
        if len(set(all_evidence)) != len(all_evidence):
            raise ValueError("replicas contain duplicate evidence")
        graph = validate_adjacency(adjacency)
        if graph.shape[0] != len(replicas):
            raise ValueError("consensus adjacency does not match replica count")
        transition = ps_lifted_matrix(graph)
        values = np.stack([agent._statistics_vector() for agent in replicas])
        result = measure_consensus(
            transition,
            values,
            tolerance=tolerance,
            max_rounds=max_rounds,
            lifted_nodes=len(replicas),
            topology=topology,
            max_bytes=max(256 * 1024 * 1024, values.nbytes * 16),
        )
        # Every entry converges to the mean.  One visible entry is the actual
        # decentralized output; multiplying by N reconstructs sufficient sums.
        fused_vector = np.ascontiguousarray(result.estimates[0] * len(replicas))
        config = replicas[0].config
        split = config.feature_size * config.feature_size
        fused = cls(config)
        fused._xtx = fused_vector[:split].reshape(
            config.feature_size, config.feature_size
        )
        fused._xty = fused_vector[split:].reshape(
            config.feature_size, config.output_size
        )
        fused._xtx = np.ascontiguousarray(0.5 * (fused._xtx + fused._xtx.T))
        fused._xty = np.ascontiguousarray(fused._xty)
        fused._xtx[fused._xtx == 0.0] = 0.0
        fused._xty[fused._xty == 0.0] = 0.0
        fused._evidence_sha256s = set(all_evidence)
        fused._sample_count = sum(agent.sample_count for agent in replicas)
        fused._evidence_mass = sum(agent.evidence_mass for agent in replicas)
        fused.solve_readout()
        receipt = ReservoirFusionReceipt(
            core_sha256=core,
            replica_sha256s=tuple(sorted(agent.sha256 for agent in replicas)),
            evidence_sha256s=tuple(sorted(all_evidence)),
            consensus=result.receipt,
            fused_statistics_sha256=fused.statistics_sha256,
            sample_count=fused.sample_count,
            evidence_mass=fused.evidence_mass,
        )
        return FusedReservoir(fused, receipt)

    def to_bytes(self) -> bytes:
        payload = canonical_json_bytes(
            {
                "arrays": {
                    "adjacency": _array_descriptor(self._adjacency),
                    "input_weights": _array_descriptor(self._input_weights),
                    "readout": _array_descriptor(self._readout),
                    "state": _array_descriptor(self._state),
                    "transition": _array_descriptor(self._transition),
                    "xtx": _array_descriptor(self._xtx),
                    "xty": _array_descriptor(self._xty),
                },
                "config": self.config.to_dict(),
                "core_sha256": self.core_sha256,
                "evidence_mass": self._evidence_mass,
                "evidence_sha256s": list(self.evidence_sha256s),
                "format": RESERVOIR_PAYLOAD_SCHEMA,
                "readout_sha256": self.readout_sha256,
                "reservoir_sha256": self.sha256,
                "sample_count": self._sample_count,
                "state_sha256": self.state_sha256,
                "statistics_sha256": self.statistics_sha256,
            }
        )
        if len(payload) > MAX_RESERVOIR_PAYLOAD_BYTES:
            raise ValueError("reservoir payload exceeds its hard byte bound")
        return payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "PSLiftedReservoirAgent":
        if not isinstance(data, bytes):
            raise TypeError("reservoir payload must be immutable bytes")
        if len(data) > MAX_RESERVOIR_PAYLOAD_BYTES:
            raise ReservoirIntegrityError("reservoir payload is oversized")
        try:
            root = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ReservoirIntegrityError("reservoir payload is not JSON") from exc
        expected = {
            "arrays",
            "config",
            "core_sha256",
            "evidence_mass",
            "evidence_sha256s",
            "format",
            "readout_sha256",
            "reservoir_sha256",
            "sample_count",
            "state_sha256",
            "statistics_sha256",
        }
        if (
            not isinstance(root, dict)
            or set(root) != expected
            or root.get("format") != RESERVOIR_PAYLOAD_SCHEMA
        ):
            raise ReservoirIntegrityError("invalid reservoir payload structure")
        if canonical_json_bytes(root) != data:
            raise ReservoirIntegrityError("reservoir payload is not canonical")
        config = PSLiftedReservoirConfig.from_dict(root["config"])
        arrays = root["arrays"]
        if not isinstance(arrays, dict) or set(arrays) != {
            "adjacency",
            "input_weights",
            "readout",
            "state",
            "transition",
            "xtx",
            "xty",
        }:
            raise ReservoirIntegrityError("invalid reservoir array collection")
        feature = config.feature_size
        state = config.state_size
        decoded_bound = min(MAX_RESERVOIR_PAYLOAD_BYTES, max(1, len(data) * 2))
        restored = cls(config)
        restored._adjacency = _decode_array(
            arrays["adjacency"],
            shape=(config.nodes, config.nodes),
            name="adjacency",
            max_bytes=decoded_bound,
        )
        restored._transition = _decode_array(
            arrays["transition"],
            shape=(state, state),
            name="transition",
            max_bytes=decoded_bound,
        )
        restored._input_weights = _decode_array(
            arrays["input_weights"],
            shape=(state, config.input_size),
            name="input_weights",
            max_bytes=decoded_bound,
        )
        restored._state = _decode_array(
            arrays["state"],
            shape=(state,),
            name="state",
            max_bytes=decoded_bound,
        )
        restored._xtx = _decode_array(
            arrays["xtx"],
            shape=(feature, feature),
            name="xtx",
            max_bytes=decoded_bound,
        )
        restored._xty = _decode_array(
            arrays["xty"],
            shape=(feature, config.output_size),
            name="xty",
            max_bytes=decoded_bound,
        )
        restored._readout = _decode_array(
            arrays["readout"],
            shape=(feature, config.output_size),
            name="readout",
            max_bytes=decoded_bound,
        )
        try:
            evidence = tuple(
                require_sha256(value, field="evidence_sha256")
                for value in root["evidence_sha256s"]
            )
        except (TypeError, ValueError) as exc:
            raise ReservoirIntegrityError("invalid reservoir evidence") from exc
        if (
            tuple(sorted(evidence)) != evidence
            or len(set(evidence)) != len(evidence)
            or len(evidence) > MAX_RESERVOIR_EVIDENCE
        ):
            raise ReservoirIntegrityError("reservoir evidence is non-canonical")
        restored._evidence_sha256s = set(evidence)
        sample_count = root["sample_count"]
        if (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count != len(evidence)
        ):
            raise ReservoirIntegrityError("reservoir sample count mismatch")
        restored._sample_count = sample_count
        try:
            restored._evidence_mass = _finite(
                root["evidence_mass"],
                name="evidence_mass",
                minimum=0.0,
            )
            if not np.array_equal(restored._xtx, restored._xtx.T):
                raise ReservoirIntegrityError(
                    "reservoir X^T X sufficient statistic is not symmetric"
                )
            diagonal = np.diag(restored._xtx)
            if bool(np.any(diagonal < 0.0)):
                raise ReservoirIntegrityError(
                    "reservoir X^T X has a negative diagonal"
                )
            # Every feature vector ends in an exact intercept of one, hence
            # its accumulated diagonal mass equals total evidence mass.  A
            # decentralized consensus may differ only by its declared numeric
            # convergence tolerance; local artifacts are bit-exact here.
            mass_tolerance = 1e-7 * max(1.0, restored._evidence_mass)
            if (
                abs(float(restored._xtx[-1, -1]) - restored._evidence_mass)
                > mass_tolerance
            ):
                raise ReservoirIntegrityError(
                    "reservoir sufficient-statistic mass is inconsistent"
                )
            scale = max(1.0, float(np.max(diagonal, initial=0.0)))
            cauchy_tolerance = 1e-10 * scale * scale
            if bool(
                np.any(
                    restored._xtx * restored._xtx
                    > diagonal[:, None] * diagonal[None, :] + cauchy_tolerance
                )
            ):
                raise ReservoirIntegrityError(
                    "reservoir X^T X violates Gram-matrix bounds"
                )
            stored_readout = restored._readout.copy()
            restored.solve_readout()
            readout_tolerance = 1e-10 * max(
                1.0,
                float(np.max(np.abs(restored._readout), initial=0.0)),
            )
            if (
                float(
                    np.max(
                        np.abs(restored._readout - stored_readout),
                        initial=0.0,
                    )
                )
                > readout_tolerance
            ):
                raise ReservoirIntegrityError(
                    "reservoir readout does not match its sufficient statistics"
                )
            restored._readout = stored_readout
            expected_hashes = {
                "core_sha256": restored.core_sha256,
                "readout_sha256": restored.readout_sha256,
                "reservoir_sha256": restored.sha256,
                "state_sha256": restored.state_sha256,
                "statistics_sha256": restored.statistics_sha256,
            }
            for field, actual in expected_hashes.items():
                declared = require_sha256(root[field], field=field)
                if declared != actual:
                    raise ReservoirIntegrityError(f"{field} mismatch")
        except ValueError as exc:
            if isinstance(exc, ReservoirIntegrityError):
                raise
            raise ReservoirIntegrityError("reservoir identity failed") from exc
        generated_adjacency, generated_transition, generated_input = cls._build_core(
            config
        )
        if not (
            np.array_equal(restored._adjacency, generated_adjacency)
            and np.array_equal(restored._transition, generated_transition)
            and np.array_equal(restored._input_weights, generated_input)
        ):
            raise ReservoirIntegrityError("fixed reservoir core was altered")
        if restored.to_bytes() != data:
            raise ReservoirIntegrityError("reservoir failed canonical roundtrip")
        return restored


__all__ = [
    "FusedReservoir",
    "MAX_RESERVOIR_EVIDENCE",
    "MAX_RESERVOIR_FEATURES",
    "MAX_RESERVOIR_INPUTS",
    "MAX_RESERVOIR_NODES",
    "MAX_RESERVOIR_OUTPUTS",
    "MAX_RESERVOIR_PAYLOAD_BYTES",
    "PSLiftedReservoirAgent",
    "PSLiftedReservoirConfig",
    "RESERVOIR_FUSION_SCHEMA",
    "RESERVOIR_PAYLOAD_SCHEMA",
    "RESERVOIR_SCHEMA",
    "ReservoirFusionReceipt",
    "ReservoirIntegrityError",
    "ReservoirPrediction",
]
