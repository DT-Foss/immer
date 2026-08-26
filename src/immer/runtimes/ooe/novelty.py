"""Hopfield-energy novelty gates for contextual OoE routing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .identity import canonical_json_bytes, require_sha256


HOPFIELD_CALIBRATION_SCHEMA = "immer-ooe-hopfield-calibration/v1"
HOPFIELD_MODEL_SCHEMA = "immer-ooe-hopfield-novelty-model/v1"
MAX_HOPFIELD_DIMENSIONS = 4096
MAX_HOPFIELD_PATTERNS = 1_000_000

FloatArray = NDArray[np.float64]


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _vector(value: ArrayLike, *, dimension: int | None = None) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or not 1 <= array.size <= MAX_HOPFIELD_DIMENSIONS:
        raise ValueError("Hopfield vector has invalid dimensions")
    if dimension is not None and array.size != dimension:
        raise ValueError("Hopfield vector dimension mismatch")
    if not bool(np.isfinite(array).all()):
        raise ValueError("Hopfield vector contains non-finite data")
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError("Hopfield vector must have nonzero norm")
    normalized = array.copy() if abs(norm - 1.0) <= 1e-15 else array / norm
    normalized[normalized == 0.0] = 0.0
    return normalized


def _matrix(value: ArrayLike, *, dimension: int | None = None) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 2
        or not 1 <= array.shape[0] <= MAX_HOPFIELD_PATTERNS
        or not 1 <= array.shape[1] <= MAX_HOPFIELD_DIMENSIONS
    ):
        raise ValueError("Hopfield pattern matrix has invalid dimensions")
    if dimension is not None and array.shape[1] != dimension:
        raise ValueError("Hopfield pattern dimension mismatch")
    if not bool(np.isfinite(array).all()):
        raise ValueError("Hopfield pattern matrix contains non-finite data")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if bool(np.any(norms <= 0.0)):
        raise ValueError("Hopfield patterns must have nonzero norm")
    normalized = array.copy()
    rows = np.abs(norms[:, 0] - 1.0) > 1e-15
    normalized[rows] /= norms[rows]
    normalized[normalized == 0.0] = 0.0
    return normalized


def _positive_beta(value: float) -> float:
    beta = float(value)
    if not math.isfinite(beta) or beta <= 0.0 or beta > 1_000_000.0:
        raise ValueError("beta must be positive, finite, and bounded")
    return beta


def hopfield_energy(
    patterns: ArrayLike,
    query: ArrayLike,
    *,
    beta: float = 8.0,
) -> float:
    """Return ``-LSE(beta X q) + 0.5 q^T q`` with stable log-sum-exp."""

    temperature = _positive_beta(beta)
    matrix = _matrix(patterns)
    vector = _vector(query, dimension=matrix.shape[1])
    scores = temperature * (matrix @ vector)
    maximum = float(np.max(scores))
    logsumexp = maximum + math.log(float(np.exp(scores - maximum).sum()))
    return float(-logsumexp + 0.5 * np.dot(vector, vector))


@dataclass(frozen=True, slots=True)
class HopfieldCalibrationReceipt:
    dimension: int
    beta: float
    pattern_sha256s: tuple[tuple[str, tuple[str, ...]], ...]
    thresholds: tuple[tuple[str, float], ...]
    minimum_energy_gap: float
    energy_quantile: float
    gap_quantile: float
    threshold_slack: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, int)
            or not 1 <= self.dimension <= MAX_HOPFIELD_DIMENSIONS
        ):
            raise ValueError("dimension is invalid")
        object.__setattr__(self, "beta", _positive_beta(self.beta))
        patterns = tuple(self.pattern_sha256s)
        thresholds = tuple(self.thresholds)
        labels = tuple(label for label, _ in patterns)
        if not labels or labels != tuple(sorted(labels)) or len(set(labels)) != len(labels):
            raise ValueError("pattern labels must be sorted and unique")
        if tuple(label for label, _ in thresholds) != labels:
            raise ValueError("threshold labels differ from pattern labels")
        normalized_patterns = []
        normalized_thresholds = []
        for (label, hashes), (_, threshold) in zip(patterns, thresholds, strict=True):
            if not isinstance(label, str) or not label or label != label.strip():
                raise ValueError("Hopfield label is invalid")
            values = tuple(require_sha256(value, field="pattern_sha256") for value in hashes)
            if not values or values != tuple(sorted(values)) or len(set(values)) != len(values):
                raise ValueError("pattern hashes must be sorted, unique, and non-empty")
            numeric_threshold = float(threshold)
            if not math.isfinite(numeric_threshold):
                raise ValueError("Hopfield threshold must be finite")
            normalized_patterns.append((label, values))
            normalized_thresholds.append((label, numeric_threshold))
        gap = float(self.minimum_energy_gap)
        if not math.isfinite(gap) or gap < 0.0:
            raise ValueError("minimum_energy_gap must be finite and non-negative")
        for name in ("energy_quantile", "gap_quantile"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        slack = float(self.threshold_slack)
        if not math.isfinite(slack) or slack < 0.0:
            raise ValueError("threshold_slack must be finite and non-negative")
        object.__setattr__(self, "threshold_slack", slack)
        object.__setattr__(self, "pattern_sha256s", tuple(normalized_patterns))
        object.__setattr__(self, "thresholds", tuple(normalized_thresholds))
        object.__setattr__(self, "minimum_energy_gap", gap)

    def to_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "dimension": self.dimension,
            "energy_quantile": self.energy_quantile,
            "gap_quantile": self.gap_quantile,
            "minimum_energy_gap": self.minimum_energy_gap,
            "pattern_sha256s": [
                {"label": label, "sha256s": list(values)}
                for label, values in self.pattern_sha256s
            ],
            "schema": HOPFIELD_CALIBRATION_SCHEMA,
            "threshold_slack": self.threshold_slack,
            "thresholds": [
                {"label": label, "value": value}
                for label, value in self.thresholds
            ],
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class HopfieldNoveltyDecision:
    label: str | None
    accepted: bool
    reason: str
    best_energy: float
    second_energy: float
    energy_gap: float
    threshold: float | None
    calibration_sha256: str


class HopfieldNoveltyModel:
    """Calibrated per-site associative energy plus inter-site energy gap."""

    def __init__(self, *, beta: float = 8.0) -> None:
        self._beta = _positive_beta(beta)
        self._patterns: dict[str, list[FloatArray]] = {}
        self._dimension: int | None = None
        self._receipt: HopfieldCalibrationReceipt | None = None

    @property
    def dimension(self) -> int | None:
        return self._dimension

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def calibration_receipt(self) -> HopfieldCalibrationReceipt | None:
        return self._receipt

    @property
    def calibration_sha256(self) -> str | None:
        return None if self._receipt is None else self._receipt.sha256

    def observe(self, label: str, sketch: ArrayLike) -> None:
        if not isinstance(label, str) or not label or label != label.strip():
            raise ValueError("label must be canonical non-empty text")
        vector = _vector(sketch, dimension=self._dimension)
        if self._dimension is None:
            self._dimension = vector.size
        count = sum(len(values) for values in self._patterns.values())
        if count >= MAX_HOPFIELD_PATTERNS:
            raise ValueError("Hopfield pattern limit reached")
        self._patterns.setdefault(label, []).append(vector.copy())
        self._receipt = None

    @staticmethod
    def _quantile(values: Sequence[float], q: float) -> float:
        return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="higher"))

    def calibrate(
        self,
        *,
        energy_quantile: float = 1.0,
        gap_quantile: float = 0.0,
        threshold_slack: float = 1e-12,
    ) -> HopfieldCalibrationReceipt:
        if self._dimension is None or not self._patterns:
            raise ValueError("observe patterns before calibration")
        for name, value in (
            ("energy_quantile", energy_quantile),
            ("gap_quantile", gap_quantile),
        ):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        slack = float(threshold_slack)
        if not math.isfinite(slack) or slack < 0.0:
            raise ValueError("threshold_slack must be finite and non-negative")
        matrices = {
            label: _matrix(np.stack(values), dimension=self._dimension)
            for label, values in sorted(self._patterns.items())
        }
        thresholds = []
        gaps = []
        for label, matrix in matrices.items():
            own_energies = [
                hopfield_energy(matrix, query, beta=self.beta) for query in matrix
            ]
            thresholds.append(
                (
                    label,
                    self._quantile(own_energies, float(energy_quantile)) + slack,
                )
            )
            for query in matrix:
                energies = sorted(
                    (
                        hopfield_energy(candidate, query, beta=self.beta),
                        candidate_label,
                    )
                    for candidate_label, candidate in matrices.items()
                )
                if len(energies) > 1 and energies[0][1] == label:
                    gaps.append(energies[1][0] - energies[0][0])
        minimum_gap = (
            0.0
            if not gaps
            else max(0.0, self._quantile(gaps, float(gap_quantile)) - slack)
        )
        pattern_hashes = tuple(
            (
                label,
                tuple(
                    sorted(
                        _digest(
                            {
                                "dimension": self._dimension,
                                "index": index,
                                "values": row.tolist(),
                            }
                        )
                        for index, row in enumerate(matrix)
                    )
                ),
            )
            for label, matrix in matrices.items()
        )
        receipt = HopfieldCalibrationReceipt(
            dimension=self._dimension,
            beta=self.beta,
            pattern_sha256s=pattern_hashes,
            thresholds=tuple(thresholds),
            minimum_energy_gap=minimum_gap,
            energy_quantile=float(energy_quantile),
            gap_quantile=float(gap_quantile),
            threshold_slack=slack,
        )
        self._receipt = receipt
        return receipt

    def decision(self, sketch: ArrayLike) -> HopfieldNoveltyDecision:
        receipt = self._receipt
        if receipt is None or self._dimension is None:
            raise RuntimeError("calibrate Hopfield novelty model before decision")
        if self.beta != receipt.beta:
            raise RuntimeError("Hopfield runtime beta differs from calibration")
        vector = _vector(sketch, dimension=self._dimension)
        energies = sorted(
            (
                hopfield_energy(np.stack(self._patterns[label]), vector, beta=self.beta),
                label,
            )
            for label in sorted(self._patterns)
        )
        best_energy, label = energies[0]
        second = math.inf if len(energies) == 1 else energies[1][0]
        gap = math.inf if len(energies) == 1 else second - best_energy
        thresholds = dict(receipt.thresholds)
        threshold = thresholds[label]
        if best_energy > threshold:
            accepted = False
            reason = "energy-threshold"
            selected = None
        elif gap < receipt.minimum_energy_gap:
            accepted = False
            reason = "energy-gap"
            selected = None
        else:
            accepted = True
            reason = "accepted"
            selected = label
        return HopfieldNoveltyDecision(
            label=selected,
            accepted=accepted,
            reason=reason,
            best_energy=best_energy,
            second_energy=second,
            energy_gap=gap,
            threshold=threshold,
            calibration_sha256=receipt.sha256,
        )

    def to_document(self) -> dict[str, Any]:
        if self._receipt is None or self._dimension is None:
            raise RuntimeError("calibrate Hopfield novelty model before serialization")
        body = {
            "beta": self.beta,
            "calibration": self._receipt.to_dict(),
            "calibration_sha256": self._receipt.sha256,
            "dimension": self._dimension,
            "patterns": [
                {
                    "label": label,
                    "values": [row.tolist() for row in values],
                }
                for label, values in sorted(self._patterns.items())
            ],
        }
        return {"body": body, "schema": HOPFIELD_MODEL_SCHEMA, "sha256": _digest(body)}

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "HopfieldNoveltyModel":
        if not isinstance(document, Mapping) or set(document) != {
            "body",
            "schema",
            "sha256",
        }:
            raise ValueError("Hopfield model envelope is invalid")
        if document.get("schema") != HOPFIELD_MODEL_SCHEMA:
            raise ValueError("Hopfield model schema is invalid")
        body = document.get("body")
        if not isinstance(body, Mapping) or set(body) != {
            "beta",
            "calibration",
            "calibration_sha256",
            "dimension",
            "patterns",
        }:
            raise ValueError("Hopfield model body is invalid")
        claimed = require_sha256(document.get("sha256"), field="sha256")
        if claimed != _digest(body):
            raise ValueError("Hopfield model SHA-256 mismatch")
        model = cls(beta=body["beta"])
        patterns = body["patterns"]
        if not isinstance(patterns, list) or not patterns:
            raise ValueError("Hopfield model patterns are invalid")
        for row in patterns:
            if not isinstance(row, Mapping) or set(row) != {"label", "values"}:
                raise ValueError("Hopfield pattern row is invalid")
            for vector in row["values"]:
                model.observe(row["label"], vector)
        calibration = body["calibration"]
        if not isinstance(calibration, Mapping) or calibration.get("schema") != HOPFIELD_CALIBRATION_SCHEMA:
            raise ValueError("Hopfield calibration body is invalid")
        pattern_rows = calibration.get("pattern_sha256s")
        threshold_rows = calibration.get("thresholds")
        if not isinstance(pattern_rows, list) or not isinstance(threshold_rows, list):
            raise ValueError("Hopfield calibration rows are invalid")
        receipt = HopfieldCalibrationReceipt(
            dimension=calibration["dimension"],
            beta=calibration["beta"],
            pattern_sha256s=tuple(
                (row["label"], tuple(row["sha256s"])) for row in pattern_rows
            ),
            thresholds=tuple(
                (row["label"], row["value"]) for row in threshold_rows
            ),
            minimum_energy_gap=calibration["minimum_energy_gap"],
            energy_quantile=calibration["energy_quantile"],
            gap_quantile=calibration["gap_quantile"],
            threshold_slack=calibration["threshold_slack"],
        )
        if (
            body["dimension"] != model._dimension
            or body["calibration_sha256"] != receipt.sha256
        ):
            raise ValueError("Hopfield model calibration binding mismatch")
        derived = model.calibrate(
            energy_quantile=receipt.energy_quantile,
            gap_quantile=receipt.gap_quantile,
            threshold_slack=receipt.threshold_slack,
        )
        if derived != receipt:
            raise ValueError("Hopfield calibration cannot be reproduced")
        model._receipt = receipt
        if model.to_document() != dict(document):
            raise ValueError("Hopfield model failed canonical reconstruction")
        return model


__all__ = [
    "HOPFIELD_CALIBRATION_SCHEMA",
    "HOPFIELD_MODEL_SCHEMA",
    "HopfieldCalibrationReceipt",
    "HopfieldNoveltyDecision",
    "HopfieldNoveltyModel",
    "MAX_HOPFIELD_DIMENSIONS",
    "MAX_HOPFIELD_PATTERNS",
    "hopfield_energy",
]
