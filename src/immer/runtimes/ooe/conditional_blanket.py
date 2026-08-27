"""Exact categorical conditional-independence blankets for OoE actions.

This module learns the smallest categorical feature set that preserves a
verified action distribution.  Selection is an exhaustive, bounded subset
search; all probabilities, losses, coverage values, and conditional-
dependence residuals use :class:`fractions.Fraction`.  Training, calibration,
and later holdout evidence are separated by source/group identity and by a
strict forward temporal boundary.

The fitted receipt is self-contained.  It stores the full, selected,
deterministic same-size placebo, and marginal arms together with every source
sample.  Deserialization recomputes the entire search, so a newly sealed but
semantically forged table is rejected as readily as a flipped byte.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from itertools import combinations
import json
import math
from typing import Any, cast

from .identity import canonical_json_bytes, require_sha256


FEATURE_ATOM_SCHEMA = "immer-ooe-categorical-feature-atom/v1"
CATEGORICAL_SAMPLE_SCHEMA = "immer-ooe-categorical-sample/v1"
CONDITIONAL_BLANKET_CONFIG_SCHEMA = "immer-ooe-conditional-blanket-config/v1"
CONDITIONAL_BLANKET_ARM_SCHEMA = "immer-ooe-conditional-blanket-arm/v1"
CONDITIONAL_BLANKET_FIT_SCHEMA = "immer-ooe-conditional-blanket-fit/v1"
CONDITIONAL_BLANKET_VALIDATION_SCHEMA = (
    "immer-ooe-conditional-blanket-validation/v1"
)
MAX_FEATURES = 32
MAX_TARGET_CATEGORIES = 128
MAX_SAMPLES_PER_SPLIT = 100_000
MAX_EXHAUSTIVE_SUBSETS = 1_000_000
MAX_PAIR_CHECKS = 10_000
MAX_DOCUMENT_BYTES = 256 * 1024 * 1024
MAX_RATIO_BITS = 1_000_000
DEFAULT_PLACEBO_SEED_SHA256 = hashlib.sha256(
    b"immer-ooe-conditional-blanket-placebo/v1"
).hexdigest()


class ConditionalBlanketError(ValueError):
    """Categorical evidence cannot satisfy the blanket contract."""


class ConditionalBlanketIntegrityError(ConditionalBlanketError):
    """A source, model arm, metric, seal, or replay was modified."""


class ConditionalBlanketLeakageError(ConditionalBlanketError):
    """Training, calibration, or holdout evidence is not isolated."""


class ConditionalBlanketBoundsError(ConditionalBlanketError):
    """The requested exact search exceeds its explicit bound."""


class ConditionalBlanketAbstentionError(ConditionalBlanketError):
    """The selected blanket is not authorized for runtime prediction."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _canonical_text(value: object, *, field: str, maximum: int = 256) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise ConditionalBlanketError(f"{field} must be canonical non-empty text")
    return value


def _bounded_uint(
    value: object,
    *,
    field: str,
    minimum: int = 0,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ConditionalBlanketBoundsError(
            f"{field} must be an integer in {minimum}..{maximum}"
        )
    return value


def _fraction(
    value: object,
    *,
    field: str,
    minimum: Fraction = Fraction(),
    maximum: Fraction | None = None,
    positive: bool = False,
) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ConditionalBlanketError(f"{field} must be exact integer/Fraction data")
    result = Fraction(value)
    if (positive and result <= 0) or result < minimum:
        raise ConditionalBlanketError(f"{field} is below its exact bound")
    if maximum is not None and result > maximum:
        raise ConditionalBlanketError(f"{field} exceeds its exact bound")
    return result


def _seal(schema: str, body: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json_bytes(dict(body)))
    return {"body": normalized, "schema": schema, "sha256": _digest(normalized)}


def _unseal(
    document: Mapping[str, Any],
    *,
    schema: str,
    fields: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(document, Mapping) or set(document) != {
        "body",
        "schema",
        "sha256",
    }:
        raise ConditionalBlanketIntegrityError(f"{label} envelope is invalid")
    if document.get("schema") != schema:
        raise ConditionalBlanketIntegrityError(f"{label} schema is invalid")
    body = document.get("body")
    if not isinstance(body, Mapping) or set(body) != fields:
        raise ConditionalBlanketIntegrityError(f"{label} body is invalid")
    try:
        claimed = require_sha256(document.get("sha256"), field=f"{label}.sha256")
    except ValueError as exc:
        raise ConditionalBlanketIntegrityError(f"{label} seal is invalid") from exc
    if claimed != _digest(body):
        raise ConditionalBlanketIntegrityError(f"{label} SHA-256 mismatch")
    return cast(dict[str, Any], json.loads(canonical_json_bytes(body)))


def _document_bytes(document: Mapping[str, Any], *, label: str) -> bytes:
    try:
        data = canonical_json_bytes(dict(document))
    except ValueError as exc:
        raise ConditionalBlanketIntegrityError(f"{label} is not canonical JSON") from exc
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ConditionalBlanketBoundsError(f"{label} exceeds its byte bound")
    return data


def _parse_bytes(data: bytes, *, label: str) -> dict[str, Any]:
    if not isinstance(data, bytes) or not data or len(data) > MAX_DOCUMENT_BYTES:
        raise ConditionalBlanketIntegrityError(f"{label} must be bounded bytes")
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ConditionalBlanketIntegrityError(f"{label} is not JSON") from exc
    try:
        canonical = canonical_json_bytes(document)
    except ValueError as exc:
        raise ConditionalBlanketIntegrityError(
            f"{label} contains non-canonical values"
        ) from exc
    if not isinstance(document, dict) or canonical != data:
        raise ConditionalBlanketIntegrityError(f"{label} is not canonical JSON")
    return document


@dataclass(frozen=True, slots=True, order=True)
class ExactRatio:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        numerator = self.numerator
        denominator = self.denominator
        if (
            isinstance(numerator, bool)
            or not isinstance(numerator, int)
            or numerator < 0
            or numerator.bit_length() > MAX_RATIO_BITS
        ):
            raise ConditionalBlanketBoundsError(
                "ratio numerator is negative, non-integer, or too large"
            )
        if (
            isinstance(denominator, bool)
            or not isinstance(denominator, int)
            or denominator <= 0
            or denominator.bit_length() > MAX_RATIO_BITS
        ):
            raise ConditionalBlanketBoundsError(
                "ratio denominator is non-positive, non-integer, or too large"
            )
        reduced = Fraction(numerator, denominator)
        if (reduced.numerator, reduced.denominator) != (numerator, denominator):
            raise ConditionalBlanketIntegrityError("ratio must be canonically reduced")

    @classmethod
    def from_fraction(cls, value: Fraction) -> "ExactRatio":
        exact = _fraction(value, field="ratio")
        return cls(exact.numerator, exact.denominator)

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def to_dict(self) -> dict[str, int]:
        return {"denominator": self.denominator, "numerator": self.numerator}

    @classmethod
    def from_dict(cls, value: object) -> "ExactRatio":
        if not isinstance(value, Mapping) or set(value) != {
            "denominator",
            "numerator",
        }:
            raise ConditionalBlanketIntegrityError("exact ratio is invalid")
        return cls(numerator=value["numerator"], denominator=value["denominator"])


@dataclass(frozen=True, slots=True, order=True)
class CategoricalFeatureAtom:
    """One named categorical observation."""

    name: str
    category: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _canonical_text(self.name, field="feature.name", maximum=128)
        )
        object.__setattr__(
            self,
            "category",
            _canonical_text(self.category, field="feature.category", maximum=512),
        )

    def as_record(self) -> dict[str, str]:
        return {"category": self.category, "name": self.name}

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(FEATURE_ATOM_SCHEMA, self.as_record())

    def to_bytes(self) -> bytes:
        return _document_bytes(self.to_document(), label="feature atom")

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "CategoricalFeatureAtom":
        body = _unseal(
            document,
            schema=FEATURE_ATOM_SCHEMA,
            fields=frozenset({"category", "name"}),
            label="feature atom",
        )
        result = cls(name=body["name"], category=body["category"])
        if result.to_document() != document:
            raise ConditionalBlanketIntegrityError("feature atom replay changed")
        return result

    @classmethod
    def from_bytes(cls, data: bytes) -> "CategoricalFeatureAtom":
        return cls.from_document(_parse_bytes(data, label="feature atom"))


@dataclass(frozen=True, slots=True)
class CategoricalSample:
    """One temporally and cryptographically bound categorical action sample."""

    temporal_index: int
    group_sha256: str
    source_receipt_sha256: str
    source_revision_sha256: str
    verifier_sha256: str
    evidence_sha256: str
    target_category: str
    features: tuple[CategoricalFeatureAtom, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "temporal_index",
            _bounded_uint(
                self.temporal_index,
                field="temporal_index",
                maximum=2**63 - 1,
            ),
        )
        for name in (
            "group_sha256",
            "source_receipt_sha256",
            "source_revision_sha256",
            "verifier_sha256",
            "evidence_sha256",
        ):
            try:
                value = require_sha256(getattr(self, name), field=name)
            except ValueError as exc:
                raise ConditionalBlanketError(str(exc)) from exc
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "target_category",
            _canonical_text(
                self.target_category, field="target_category", maximum=512
            ),
        )
        if isinstance(self.features, (str, bytes, bytearray)):
            raise ConditionalBlanketError("features must be a bounded atom sequence")
        try:
            features = tuple(self.features)
        except TypeError as exc:
            raise ConditionalBlanketError(
                "features must be a bounded atom sequence"
            ) from exc
        if not 1 <= len(features) <= MAX_FEATURES or any(
            not isinstance(atom, CategoricalFeatureAtom) for atom in features
        ):
            raise ConditionalBlanketBoundsError(
                f"features must contain 1..{MAX_FEATURES} atoms"
            )
        features = tuple(sorted(features, key=lambda atom: atom.name))
        if len({atom.name for atom in features}) != len(features):
            raise ConditionalBlanketIntegrityError("feature names must be unique")
        object.__setattr__(self, "features", features)

    @property
    def feature_schema(self) -> tuple[str, ...]:
        return tuple(atom.name for atom in self.features)

    @property
    def feature_map(self) -> dict[str, str]:
        return {atom.name: atom.category for atom in self.features}

    def as_record(self) -> dict[str, Any]:
        return {
            "evidence_sha256": self.evidence_sha256,
            "feature_schema": list(self.feature_schema),
            "features": [atom.to_document() for atom in self.features],
            "group_sha256": self.group_sha256,
            "source_receipt_sha256": self.source_receipt_sha256,
            "source_revision_sha256": self.source_revision_sha256,
            "target_category": self.target_category,
            "temporal_index": self.temporal_index,
            "verifier_sha256": self.verifier_sha256,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(CATEGORICAL_SAMPLE_SCHEMA, self.as_record())

    def to_bytes(self) -> bytes:
        return _document_bytes(self.to_document(), label="categorical sample")

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "CategoricalSample":
        body = _unseal(
            document,
            schema=CATEGORICAL_SAMPLE_SCHEMA,
            fields=frozenset(
                {
                    "evidence_sha256",
                    "feature_schema",
                    "features",
                    "group_sha256",
                    "source_receipt_sha256",
                    "source_revision_sha256",
                    "target_category",
                    "temporal_index",
                    "verifier_sha256",
                }
            ),
            label="categorical sample",
        )
        raw_features = body.pop("features")
        if not isinstance(raw_features, list):
            raise ConditionalBlanketIntegrityError("sample features are invalid")
        features = tuple(
            CategoricalFeatureAtom.from_document(item) for item in raw_features
        )
        schema = body.pop("feature_schema")
        if schema != [atom.name for atom in features]:
            raise ConditionalBlanketIntegrityError("sample feature schema changed")
        result = cls(features=features, **body)
        if result.to_document() != document:
            raise ConditionalBlanketIntegrityError("categorical sample replay changed")
        return result

    @classmethod
    def from_bytes(cls, data: bytes) -> "CategoricalSample":
        return cls.from_document(_parse_bytes(data, label="categorical sample"))


@dataclass(frozen=True, slots=True)
class ConditionalBlanketConfig:
    """Pinned alphabet, exact thresholds, and finite exhaustive-search budget."""

    target_alphabet: tuple[str, ...]
    max_subset_size: int = 4
    max_exhaustive_subsets: int = 100_000
    max_pair_checks: int = 256
    laplace_alpha: Fraction = Fraction(1)
    max_brier_regret: Fraction = Fraction()
    max_accuracy_regret: Fraction = Fraction()
    max_coverage_regret: Fraction = Fraction()
    dependence_tolerance: Fraction = Fraction()
    placebo_seed_sha256: str = DEFAULT_PLACEBO_SEED_SHA256

    def __post_init__(self) -> None:
        if isinstance(self.target_alphabet, (str, bytes, bytearray)):
            raise ConditionalBlanketError("target_alphabet must be a sequence")
        try:
            alphabet = tuple(
                _canonical_text(value, field="target_alphabet", maximum=512)
                for value in self.target_alphabet
            )
        except TypeError as exc:
            raise ConditionalBlanketError("target_alphabet must be a sequence") from exc
        alphabet = tuple(sorted(alphabet))
        if not 2 <= len(alphabet) <= MAX_TARGET_CATEGORIES:
            raise ConditionalBlanketBoundsError(
                f"target_alphabet must contain 2..{MAX_TARGET_CATEGORIES} categories"
            )
        if len(set(alphabet)) != len(alphabet):
            raise ConditionalBlanketIntegrityError(
                "target_alphabet categories must be unique"
            )
        object.__setattr__(self, "target_alphabet", alphabet)
        object.__setattr__(
            self,
            "max_subset_size",
            _bounded_uint(
                self.max_subset_size,
                field="max_subset_size",
                maximum=MAX_FEATURES,
            ),
        )
        object.__setattr__(
            self,
            "max_exhaustive_subsets",
            _bounded_uint(
                self.max_exhaustive_subsets,
                field="max_exhaustive_subsets",
                minimum=1,
                maximum=MAX_EXHAUSTIVE_SUBSETS,
            ),
        )
        object.__setattr__(
            self,
            "max_pair_checks",
            _bounded_uint(
                self.max_pair_checks,
                field="max_pair_checks",
                minimum=1,
                maximum=MAX_PAIR_CHECKS,
            ),
        )
        object.__setattr__(
            self,
            "laplace_alpha",
            _fraction(
                self.laplace_alpha,
                field="laplace_alpha",
                positive=True,
                maximum=Fraction(1_000_000),
            ),
        )
        object.__setattr__(
            self,
            "max_brier_regret",
            _fraction(
                self.max_brier_regret,
                field="max_brier_regret",
                maximum=Fraction(2),
            ),
        )
        for name in (
            "max_accuracy_regret",
            "max_coverage_regret",
            "dependence_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _fraction(
                    getattr(self, name), field=name, maximum=Fraction(1)
                ),
            )
        try:
            seed = require_sha256(
                self.placebo_seed_sha256, field="placebo_seed_sha256"
            )
        except ValueError as exc:
            raise ConditionalBlanketError(str(exc)) from exc
        object.__setattr__(self, "placebo_seed_sha256", seed)

    @staticmethod
    def _ratio(value: Fraction) -> dict[str, int]:
        return ExactRatio.from_fraction(value).to_dict()

    def as_record(self) -> dict[str, Any]:
        return {
            "dependence_tolerance": self._ratio(self.dependence_tolerance),
            "laplace_alpha": self._ratio(self.laplace_alpha),
            "max_accuracy_regret": self._ratio(self.max_accuracy_regret),
            "max_brier_regret": self._ratio(self.max_brier_regret),
            "max_coverage_regret": self._ratio(self.max_coverage_regret),
            "max_exhaustive_subsets": self.max_exhaustive_subsets,
            "max_pair_checks": self.max_pair_checks,
            "max_subset_size": self.max_subset_size,
            "placebo_seed_sha256": self.placebo_seed_sha256,
            "target_alphabet": list(self.target_alphabet),
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(CONDITIONAL_BLANKET_CONFIG_SCHEMA, self.as_record())

    def to_bytes(self) -> bytes:
        return _document_bytes(self.to_document(), label="blanket config")

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "ConditionalBlanketConfig":
        body = _unseal(
            document,
            schema=CONDITIONAL_BLANKET_CONFIG_SCHEMA,
            fields=frozenset(
                {
                    "dependence_tolerance",
                    "laplace_alpha",
                    "max_accuracy_regret",
                    "max_brier_regret",
                    "max_coverage_regret",
                    "max_exhaustive_subsets",
                    "max_pair_checks",
                    "max_subset_size",
                    "placebo_seed_sha256",
                    "target_alphabet",
                }
            ),
            label="blanket config",
        )
        for name in (
            "dependence_tolerance",
            "laplace_alpha",
            "max_accuracy_regret",
            "max_brier_regret",
            "max_coverage_regret",
        ):
            body[name] = ExactRatio.from_dict(body[name]).fraction
        body["target_alphabet"] = tuple(body["target_alphabet"])
        result = cls(**body)
        if result.to_document() != document:
            raise ConditionalBlanketIntegrityError("blanket config replay changed")
        return result

    @classmethod
    def from_bytes(cls, data: bytes) -> "ConditionalBlanketConfig":
        return cls.from_document(_parse_bytes(data, label="blanket config"))


@dataclass(frozen=True, slots=True)
class ExactCategoricalMetrics:
    sample_count: int
    multiclass_brier: ExactRatio
    top1_accuracy: ExactRatio
    context_coverage: ExactRatio

    def __post_init__(self) -> None:
        _bounded_uint(
            self.sample_count,
            field="metrics.sample_count",
            minimum=1,
            maximum=MAX_SAMPLES_PER_SPLIT,
        )
        for name in ("multiclass_brier", "top1_accuracy", "context_coverage"):
            if not isinstance(getattr(self, name), ExactRatio):
                raise TypeError(f"{name} must be ExactRatio")
        if self.multiclass_brier.fraction > 2:
            raise ConditionalBlanketIntegrityError("multiclass Brier exceeds two")
        if self.top1_accuracy.fraction > 1 or self.context_coverage.fraction > 1:
            raise ConditionalBlanketIntegrityError("accuracy/coverage exceeds one")

    @property
    def brier(self) -> Fraction:
        return self.multiclass_brier.fraction

    @property
    def accuracy(self) -> Fraction:
        return self.top1_accuracy.fraction

    @property
    def coverage(self) -> Fraction:
        return self.context_coverage.fraction

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_coverage": self.context_coverage.to_dict(),
            "multiclass_brier": self.multiclass_brier.to_dict(),
            "sample_count": self.sample_count,
            "top1_accuracy": self.top1_accuracy.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ExactCategoricalMetrics":
        if not isinstance(value, Mapping) or set(value) != {
            "context_coverage",
            "multiclass_brier",
            "sample_count",
            "top1_accuracy",
        }:
            raise ConditionalBlanketIntegrityError("categorical metrics are invalid")
        return cls(
            sample_count=value["sample_count"],
            multiclass_brier=ExactRatio.from_dict(value["multiclass_brier"]),
            top1_accuracy=ExactRatio.from_dict(value["top1_accuracy"]),
            context_coverage=ExactRatio.from_dict(value["context_coverage"]),
        )


@dataclass(frozen=True, slots=True, order=True)
class ConditionalCell:
    context: tuple[str, ...]
    target_counts: tuple[int, ...]
    probabilities: tuple[ExactRatio, ...]

    def __post_init__(self) -> None:
        context = tuple(
            _canonical_text(value, field="cell.context", maximum=512)
            for value in self.context
        )
        counts = tuple(self.target_counts)
        probabilities = tuple(self.probabilities)
        if not counts or len(counts) != len(probabilities):
            raise ConditionalBlanketIntegrityError("cell target vectors are invalid")
        for count in counts:
            _bounded_uint(
                count,
                field="cell.target_count",
                maximum=MAX_SAMPLES_PER_SPLIT,
            )
        if sum(counts) <= 0:
            raise ConditionalBlanketIntegrityError("cell cannot be empty")
        if any(not isinstance(value, ExactRatio) for value in probabilities):
            raise TypeError("cell probabilities must be ExactRatio values")
        if sum((value.fraction for value in probabilities), Fraction()) != 1:
            raise ConditionalBlanketIntegrityError(
                "cell probabilities do not sum exactly to one"
            )
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "target_counts", counts)
        object.__setattr__(self, "probabilities", probabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": list(self.context),
            "probabilities": [value.to_dict() for value in self.probabilities],
            "target_counts": list(self.target_counts),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConditionalCell":
        if not isinstance(value, Mapping) or set(value) != {
            "context",
            "probabilities",
            "target_counts",
        }:
            raise ConditionalBlanketIntegrityError("conditional cell is invalid")
        return cls(
            context=tuple(value["context"]),
            target_counts=tuple(value["target_counts"]),
            probabilities=tuple(
                ExactRatio.from_dict(row) for row in value["probabilities"]
            ),
        )


@dataclass(frozen=True, slots=True)
class ConditionalBlanketArm:
    """One exact Laplace table and its isolated calibration metrics."""

    name: str
    feature_names: tuple[str, ...]
    target_alphabet: tuple[str, ...]
    laplace_alpha: ExactRatio
    training_sample_count: int
    marginal_counts: tuple[int, ...]
    marginal_probabilities: tuple[ExactRatio, ...]
    cells: tuple[ConditionalCell, ...]
    calibration_metrics: ExactCategoricalMetrics
    selection_sha256: str

    def __post_init__(self) -> None:
        if self.name not in {"full", "selected", "random", "marginal"}:
            raise ConditionalBlanketError("arm name is invalid")
        features = tuple(
            sorted(
                _canonical_text(value, field="arm.feature_name", maximum=128)
                for value in self.feature_names
            )
        )
        if len(set(features)) != len(features) or len(features) > MAX_FEATURES:
            raise ConditionalBlanketIntegrityError("arm feature names are invalid")
        alphabet = tuple(
            sorted(
                _canonical_text(value, field="arm.target", maximum=512)
                for value in self.target_alphabet
            )
        )
        if not 2 <= len(alphabet) <= MAX_TARGET_CATEGORIES or len(set(alphabet)) != len(
            alphabet
        ):
            raise ConditionalBlanketIntegrityError("arm target alphabet is invalid")
        if not isinstance(self.laplace_alpha, ExactRatio) or (
            self.laplace_alpha.fraction <= 0
        ):
            raise ConditionalBlanketIntegrityError("arm Laplace alpha is invalid")
        training_count = _bounded_uint(
            self.training_sample_count,
            field="arm.training_sample_count",
            minimum=1,
            maximum=MAX_SAMPLES_PER_SPLIT,
        )
        counts = tuple(self.marginal_counts)
        probabilities = tuple(self.marginal_probabilities)
        if len(counts) != len(alphabet) or len(probabilities) != len(alphabet):
            raise ConditionalBlanketIntegrityError("arm marginal vectors are invalid")
        for count in counts:
            _bounded_uint(
                count,
                field="arm.marginal_count",
                maximum=MAX_SAMPLES_PER_SPLIT,
            )
        if sum(counts) != training_count:
            raise ConditionalBlanketIntegrityError("arm marginal counts do not bind N")
        if any(not isinstance(value, ExactRatio) for value in probabilities) or sum(
            (value.fraction for value in probabilities), Fraction()
        ) != 1:
            raise ConditionalBlanketIntegrityError(
                "arm marginal probabilities are invalid"
            )
        cells = tuple(sorted(self.cells, key=lambda cell: cell.context))
        if not cells or len(cells) > training_count:
            raise ConditionalBlanketIntegrityError("arm conditional table is invalid")
        if any(not isinstance(cell, ConditionalCell) for cell in cells):
            raise TypeError("arm cells must be ConditionalCell values")
        if len({cell.context for cell in cells}) != len(cells):
            raise ConditionalBlanketIntegrityError("arm context is duplicated")
        if any(
            len(cell.context) != len(features)
            or len(cell.target_counts) != len(alphabet)
            for cell in cells
        ):
            raise ConditionalBlanketIntegrityError("arm cell shape is invalid")
        if sum(sum(cell.target_counts) for cell in cells) != training_count:
            raise ConditionalBlanketIntegrityError("arm cells do not cover training")
        if not isinstance(self.calibration_metrics, ExactCategoricalMetrics):
            raise TypeError("calibration_metrics must be ExactCategoricalMetrics")
        try:
            selection = require_sha256(
                self.selection_sha256, field="selection_sha256"
            )
        except ValueError as exc:
            raise ConditionalBlanketError(str(exc)) from exc
        object.__setattr__(self, "feature_names", features)
        object.__setattr__(self, "target_alphabet", alphabet)
        object.__setattr__(self, "training_sample_count", training_count)
        object.__setattr__(self, "marginal_counts", counts)
        object.__setattr__(self, "marginal_probabilities", probabilities)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "selection_sha256", selection)

    def as_record(self) -> dict[str, Any]:
        return {
            "calibration_metrics": self.calibration_metrics.to_dict(),
            "cells": [cell.to_dict() for cell in self.cells],
            "feature_names": list(self.feature_names),
            "laplace_alpha": self.laplace_alpha.to_dict(),
            "marginal_counts": list(self.marginal_counts),
            "marginal_probabilities": [
                value.to_dict() for value in self.marginal_probabilities
            ],
            "name": self.name,
            "selection_sha256": self.selection_sha256,
            "target_alphabet": list(self.target_alphabet),
            "training_sample_count": self.training_sample_count,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        return _seal(CONDITIONAL_BLANKET_ARM_SCHEMA, self.as_record())

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> "ConditionalBlanketArm":
        body = _unseal(
            document,
            schema=CONDITIONAL_BLANKET_ARM_SCHEMA,
            fields=frozenset(
                {
                    "calibration_metrics",
                    "cells",
                    "feature_names",
                    "laplace_alpha",
                    "marginal_counts",
                    "marginal_probabilities",
                    "name",
                    "selection_sha256",
                    "target_alphabet",
                    "training_sample_count",
                }
            ),
            label="blanket arm",
        )
        result = cls(
            name=body["name"],
            feature_names=tuple(body["feature_names"]),
            target_alphabet=tuple(body["target_alphabet"]),
            laplace_alpha=ExactRatio.from_dict(body["laplace_alpha"]),
            training_sample_count=body["training_sample_count"],
            marginal_counts=tuple(body["marginal_counts"]),
            marginal_probabilities=tuple(
                ExactRatio.from_dict(row) for row in body["marginal_probabilities"]
            ),
            cells=tuple(ConditionalCell.from_dict(row) for row in body["cells"]),
            calibration_metrics=ExactCategoricalMetrics.from_dict(
                body["calibration_metrics"]
            ),
            selection_sha256=body["selection_sha256"],
        )
        if result.to_document() != document:
            raise ConditionalBlanketIntegrityError("blanket arm replay changed")
        return result


@dataclass(frozen=True, slots=True, order=True)
class ConditionalDependenceCheck:
    excluded_feature_names: tuple[str, ...]
    residual: ExactRatio

    def __post_init__(self) -> None:
        names = tuple(
            sorted(
                _canonical_text(value, field="excluded_feature", maximum=128)
                for value in self.excluded_feature_names
            )
        )
        if not 1 <= len(names) <= 2 or len(set(names)) != len(names):
            raise ConditionalBlanketIntegrityError(
                "dependence check must bind one or two excluded features"
            )
        if not isinstance(self.residual, ExactRatio) or self.residual.fraction > 1:
            raise ConditionalBlanketIntegrityError("dependence residual is invalid")
        object.__setattr__(self, "excluded_feature_names", names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "excluded_feature_names": list(self.excluded_feature_names),
            "residual": self.residual.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ConditionalDependenceCheck":
        if not isinstance(value, Mapping) or set(value) != {
            "excluded_feature_names",
            "residual",
        }:
            raise ConditionalBlanketIntegrityError("dependence check is invalid")
        return cls(
            excluded_feature_names=tuple(value["excluded_feature_names"]),
            residual=ExactRatio.from_dict(value["residual"]),
        )


def _source_binding(sample: CategoricalSample) -> dict[str, Any]:
    return {
        "evidence_sha256": sample.evidence_sha256,
        "group_sha256": sample.group_sha256,
        "sample_sha256": sample.sha256,
        "source_receipt_sha256": sample.source_receipt_sha256,
        "source_revision_sha256": sample.source_revision_sha256,
        "temporal_index": sample.temporal_index,
        "verifier_sha256": sample.verifier_sha256,
    }


@dataclass(frozen=True, slots=True)
class ConditionalBlanketFitReceipt:
    config: ConditionalBlanketConfig
    feature_schema: tuple[str, ...]
    train_samples: tuple[CategoricalSample, ...]
    calibration_samples: tuple[CategoricalSample, ...]
    status: str
    candidate_count: int
    candidate_subset_inventory_sha256: str
    qualifying_candidate_count: int
    closure_candidate_count: int
    closure_subset_inventory_sha256: str
    closure_evaluation_sha256: str
    closure_verified: bool
    capacity_exhausted: bool
    selected_dependence_checks: tuple[ConditionalDependenceCheck, ...]
    full_arm: ConditionalBlanketArm
    selected_arm: ConditionalBlanketArm
    random_arm: ConditionalBlanketArm
    marginal_arm: ConditionalBlanketArm

    def __post_init__(self) -> None:
        if not isinstance(self.config, ConditionalBlanketConfig):
            raise TypeError("config must be ConditionalBlanketConfig")
        schema = tuple(
            sorted(
                _canonical_text(value, field="feature_schema", maximum=128)
                for value in self.feature_schema
            )
        )
        if not 1 <= len(schema) <= MAX_FEATURES or len(set(schema)) != len(schema):
            raise ConditionalBlanketIntegrityError("fit feature schema is invalid")
        train, calibration, derived_schema = _normalize_fit_evidence(
            self.train_samples,
            self.calibration_samples,
            self.config.target_alphabet,
        )
        if schema != derived_schema:
            raise ConditionalBlanketIntegrityError("fit feature schema changed")
        if self.status not in {"sparse", "full", "capacity_exhausted"}:
            raise ConditionalBlanketError(
                "fit status must be sparse, full, or capacity_exhausted"
            )
        candidate_count = _bounded_uint(
            self.candidate_count,
            field="candidate_count",
            minimum=1,
            maximum=self.config.max_exhaustive_subsets,
        )
        qualifying = _bounded_uint(
            self.qualifying_candidate_count,
            field="qualifying_candidate_count",
            maximum=candidate_count,
        )
        closure_count = _bounded_uint(
            self.closure_candidate_count,
            field="closure_candidate_count",
            maximum=MAX_FEATURES,
        )
        for name in (
            "candidate_subset_inventory_sha256",
            "closure_subset_inventory_sha256",
            "closure_evaluation_sha256",
        ):
            try:
                value = require_sha256(getattr(self, name), field=name)
            except ValueError as exc:
                raise ConditionalBlanketError(str(exc)) from exc
            object.__setattr__(self, name, value)
        if not isinstance(self.closure_verified, bool) or not isinstance(
            self.capacity_exhausted, bool
        ):
            raise TypeError("closure/capacity flags must be bool")
        checks = tuple(
            sorted(
                self.selected_dependence_checks,
                key=lambda item: item.excluded_feature_names,
            )
        )
        if any(not isinstance(item, ConditionalDependenceCheck) for item in checks):
            raise TypeError("selected_dependence_checks contain invalid values")
        if len({item.excluded_feature_names for item in checks}) != len(checks):
            raise ConditionalBlanketIntegrityError("dependence check is duplicated")
        arms = (self.full_arm, self.selected_arm, self.random_arm, self.marginal_arm)
        if any(not isinstance(arm, ConditionalBlanketArm) for arm in arms):
            raise TypeError("fit arms must be ConditionalBlanketArm values")
        if tuple(arm.name for arm in arms) != (
            "full",
            "selected",
            "random",
            "marginal",
        ):
            raise ConditionalBlanketIntegrityError("fit arm roles changed")
        if any(
            arm.target_alphabet != self.config.target_alphabet
            or arm.training_sample_count != len(train)
            or arm.calibration_metrics.sample_count != len(calibration)
            or arm.laplace_alpha.fraction != self.config.laplace_alpha
            for arm in arms
        ):
            raise ConditionalBlanketIntegrityError("fit arm pins changed")
        if self.full_arm.feature_names != schema:
            raise ConditionalBlanketIntegrityError("full arm is not full")
        if self.marginal_arm.feature_names:
            raise ConditionalBlanketIntegrityError("marginal arm is not marginal")
        if len(self.random_arm.feature_names) != len(self.selected_arm.feature_names):
            raise ConditionalBlanketIntegrityError(
                "random placebo size differs from selected blanket"
            )
        expected_status = (
            "capacity_exhausted"
            if self.capacity_exhausted
            else (
                "sparse"
                if len(self.selected_arm.feature_names) < len(schema)
                else "full"
            )
        )
        if self.status != expected_status:
            raise ConditionalBlanketIntegrityError("fit status contradicts blanket size")
        if self.capacity_exhausted and self.closure_verified:
            raise ConditionalBlanketIntegrityError(
                "capacity exhaustion cannot claim verified closure"
            )
        if (
            closure_count == 0
            and not self.closure_verified
            and not self.capacity_exhausted
        ):
            raise ConditionalBlanketIntegrityError(
                "an empty closure frontier must be verified"
            )
        object.__setattr__(self, "feature_schema", schema)
        object.__setattr__(self, "train_samples", train)
        object.__setattr__(self, "calibration_samples", calibration)
        object.__setattr__(self, "candidate_count", candidate_count)
        object.__setattr__(self, "qualifying_candidate_count", qualifying)
        object.__setattr__(self, "closure_candidate_count", closure_count)
        object.__setattr__(self, "selected_dependence_checks", checks)

    @property
    def selected_feature_names(self) -> tuple[str, ...]:
        return self.selected_arm.feature_names

    @property
    def full_feature_count(self) -> int:
        return len(self.feature_schema)

    @property
    def selected_feature_count(self) -> int:
        return len(self.selected_feature_names)

    @property
    def compression(self) -> Fraction:
        return Fraction(self.full_feature_count, max(1, self.selected_feature_count))

    @property
    def feature_reduction(self) -> Fraction:
        return Fraction(
            self.full_feature_count - self.selected_feature_count,
            self.full_feature_count,
        )

    @property
    def abstained(self) -> bool:
        return self.capacity_exhausted

    def as_record(self) -> dict[str, Any]:
        return {
            "calibration_bindings": [
                _source_binding(sample) for sample in self.calibration_samples
            ],
            "calibration_samples": [
                sample.to_document() for sample in self.calibration_samples
            ],
            "abstained": self.abstained,
            "candidate_count": self.candidate_count,
            "candidate_subset_inventory_sha256": (
                self.candidate_subset_inventory_sha256
            ),
            "capacity_exhausted": self.capacity_exhausted,
            "closure_candidate_count": self.closure_candidate_count,
            "closure_evaluation_sha256": self.closure_evaluation_sha256,
            "closure_subset_inventory_sha256": (
                self.closure_subset_inventory_sha256
            ),
            "closure_verified": self.closure_verified,
            "compression": ExactRatio.from_fraction(self.compression).to_dict(),
            "config": self.config.to_document(),
            "config_sha256": self.config.sha256,
            "feature_reduction": ExactRatio.from_fraction(
                self.feature_reduction
            ).to_dict(),
            "feature_schema": list(self.feature_schema),
            "full_arm": self.full_arm.to_document(),
            "marginal_arm": self.marginal_arm.to_document(),
            "qualifying_candidate_count": self.qualifying_candidate_count,
            "random_arm": self.random_arm.to_document(),
            "selected_arm": self.selected_arm.to_document(),
            "selected_dependence_checks": [
                item.to_dict() for item in self.selected_dependence_checks
            ],
            "status": self.status,
            "train_bindings": [
                _source_binding(sample) for sample in self.train_samples
            ],
            "train_samples": [sample.to_document() for sample in self.train_samples],
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        document = _seal(CONDITIONAL_BLANKET_FIT_SCHEMA, self.as_record())
        _document_bytes(document, label="blanket fit")
        return document

    def to_bytes(self) -> bytes:
        return _document_bytes(self.to_document(), label="blanket fit")

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any]
    ) -> "ConditionalBlanketFitReceipt":
        _document_bytes(document, label="blanket fit")
        body = _unseal(
            document,
            schema=CONDITIONAL_BLANKET_FIT_SCHEMA,
            fields=frozenset(
                {
                    "calibration_bindings",
                    "calibration_samples",
                    "abstained",
                    "candidate_count",
                    "candidate_subset_inventory_sha256",
                    "capacity_exhausted",
                    "closure_candidate_count",
                    "closure_evaluation_sha256",
                    "closure_subset_inventory_sha256",
                    "closure_verified",
                    "compression",
                    "config",
                    "config_sha256",
                    "feature_reduction",
                    "feature_schema",
                    "full_arm",
                    "marginal_arm",
                    "qualifying_candidate_count",
                    "random_arm",
                    "selected_arm",
                    "selected_dependence_checks",
                    "status",
                    "train_bindings",
                    "train_samples",
                }
            ),
            label="blanket fit",
        )
        config = ConditionalBlanketConfig.from_document(body["config"])
        train = tuple(
            CategoricalSample.from_document(row) for row in body["train_samples"]
        )
        calibration = tuple(
            CategoricalSample.from_document(row)
            for row in body["calibration_samples"]
        )
        if body["config_sha256"] != config.sha256:
            raise ConditionalBlanketIntegrityError("fit config identity changed")
        if body["train_bindings"] != [_source_binding(row) for row in train] or body[
            "calibration_bindings"
        ] != [_source_binding(row) for row in calibration]:
            raise ConditionalBlanketIntegrityError("fit source bindings changed")
        result = cls(
            config=config,
            feature_schema=tuple(body["feature_schema"]),
            train_samples=train,
            calibration_samples=calibration,
            status=body["status"],
            candidate_count=body["candidate_count"],
            candidate_subset_inventory_sha256=body[
                "candidate_subset_inventory_sha256"
            ],
            qualifying_candidate_count=body["qualifying_candidate_count"],
            closure_candidate_count=body["closure_candidate_count"],
            closure_subset_inventory_sha256=body[
                "closure_subset_inventory_sha256"
            ],
            closure_evaluation_sha256=body["closure_evaluation_sha256"],
            closure_verified=body["closure_verified"],
            capacity_exhausted=body["capacity_exhausted"],
            selected_dependence_checks=tuple(
                ConditionalDependenceCheck.from_dict(row)
                for row in body["selected_dependence_checks"]
            ),
            full_arm=ConditionalBlanketArm.from_document(body["full_arm"]),
            selected_arm=ConditionalBlanketArm.from_document(body["selected_arm"]),
            random_arm=ConditionalBlanketArm.from_document(body["random_arm"]),
            marginal_arm=ConditionalBlanketArm.from_document(body["marginal_arm"]),
        )
        if (
            body["abstained"] != result.abstained
            or body["compression"]
            != ExactRatio.from_fraction(result.compression).to_dict()
            or body["feature_reduction"]
            != ExactRatio.from_fraction(result.feature_reduction).to_dict()
            or result.as_record() != body
        ):
            raise ConditionalBlanketIntegrityError("fit derived fields changed")
        result.verify_or_raise()
        return result

    @classmethod
    def from_bytes(cls, data: bytes) -> "ConditionalBlanketFitReceipt":
        return cls.from_document(_parse_bytes(data, label="blanket fit"))

    def verify_or_raise(self) -> bool:
        expected = _fit_core(self.train_samples, self.calibration_samples, self.config)
        if expected.as_record() != self.as_record():
            raise ConditionalBlanketIntegrityError(
                "blanket fit does not recompute exactly"
            )
        return True


@dataclass(frozen=True, slots=True)
class ConditionalBlanketValidationReceipt:
    fit: ConditionalBlanketFitReceipt
    holdout_samples: tuple[CategoricalSample, ...]
    full_metrics: ExactCategoricalMetrics
    selected_metrics: ExactCategoricalMetrics
    random_metrics: ExactCategoricalMetrics
    marginal_metrics: ExactCategoricalMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.fit, ConditionalBlanketFitReceipt):
            raise TypeError("fit must be ConditionalBlanketFitReceipt")
        holdout = _normalize_holdout(self.fit, self.holdout_samples)
        metrics = (
            self.full_metrics,
            self.selected_metrics,
            self.random_metrics,
            self.marginal_metrics,
        )
        if any(not isinstance(item, ExactCategoricalMetrics) for item in metrics):
            raise TypeError("validation metrics must be ExactCategoricalMetrics")
        if any(item.sample_count != len(holdout) for item in metrics):
            raise ConditionalBlanketIntegrityError(
                "validation metric sample count changed"
            )
        object.__setattr__(self, "holdout_samples", holdout)

    @property
    def compression(self) -> Fraction:
        return self.fit.compression

    @property
    def feature_reduction(self) -> Fraction:
        return self.fit.feature_reduction

    def as_record(self) -> dict[str, Any]:
        all_sources = sorted(
            sample.source_receipt_sha256
            for sample in (
                *self.fit.train_samples,
                *self.fit.calibration_samples,
                *self.holdout_samples,
            )
        )
        return {
            "all_source_receipt_sha256s": all_sources,
            "compression": ExactRatio.from_fraction(self.compression).to_dict(),
            "feature_reduction": ExactRatio.from_fraction(
                self.feature_reduction
            ).to_dict(),
            "fit": self.fit.to_document(),
            "fit_sha256": self.fit.sha256,
            "full_metrics": self.full_metrics.to_dict(),
            "holdout_bindings": [
                _source_binding(sample) for sample in self.holdout_samples
            ],
            "holdout_samples": [
                sample.to_document() for sample in self.holdout_samples
            ],
            "marginal_metrics": self.marginal_metrics.to_dict(),
            "random_metrics": self.random_metrics.to_dict(),
            "selected_metrics": self.selected_metrics.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return _digest(self.as_record())

    def to_document(self) -> dict[str, Any]:
        document = _seal(CONDITIONAL_BLANKET_VALIDATION_SCHEMA, self.as_record())
        _document_bytes(document, label="blanket validation")
        return document

    def to_bytes(self) -> bytes:
        return _document_bytes(self.to_document(), label="blanket validation")

    @classmethod
    def from_document(
        cls, document: Mapping[str, Any]
    ) -> "ConditionalBlanketValidationReceipt":
        _document_bytes(document, label="blanket validation")
        body = _unseal(
            document,
            schema=CONDITIONAL_BLANKET_VALIDATION_SCHEMA,
            fields=frozenset(
                {
                    "all_source_receipt_sha256s",
                    "compression",
                    "feature_reduction",
                    "fit",
                    "fit_sha256",
                    "full_metrics",
                    "holdout_bindings",
                    "holdout_samples",
                    "marginal_metrics",
                    "random_metrics",
                    "selected_metrics",
                }
            ),
            label="blanket validation",
        )
        fit = ConditionalBlanketFitReceipt.from_document(body["fit"])
        holdout = tuple(
            CategoricalSample.from_document(row) for row in body["holdout_samples"]
        )
        result = cls(
            fit=fit,
            holdout_samples=holdout,
            full_metrics=ExactCategoricalMetrics.from_dict(body["full_metrics"]),
            selected_metrics=ExactCategoricalMetrics.from_dict(
                body["selected_metrics"]
            ),
            random_metrics=ExactCategoricalMetrics.from_dict(body["random_metrics"]),
            marginal_metrics=ExactCategoricalMetrics.from_dict(
                body["marginal_metrics"]
            ),
        )
        if body["fit_sha256"] != fit.sha256:
            raise ConditionalBlanketIntegrityError("validation fit identity changed")
        if body["holdout_bindings"] != [
            _source_binding(row) for row in holdout
        ]:
            raise ConditionalBlanketIntegrityError(
                "validation holdout bindings changed"
            )
        if result.as_record() != body:
            raise ConditionalBlanketIntegrityError(
                "validation derived fields changed"
            )
        result.verify_or_raise()
        return result

    @classmethod
    def from_bytes(cls, data: bytes) -> "ConditionalBlanketValidationReceipt":
        return cls.from_document(_parse_bytes(data, label="blanket validation"))

    def verify_or_raise(self) -> bool:
        self.fit.verify_or_raise()
        expected = _validate_core(self.fit, self.holdout_samples)
        if expected.as_record() != self.as_record():
            raise ConditionalBlanketIntegrityError(
                "blanket validation does not recompute exactly"
            )
        return True


def _bounded_samples(
    values: Sequence[CategoricalSample], *, field: str
) -> tuple[CategoricalSample, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ConditionalBlanketError(f"{field} must be a sample sequence")
    try:
        rows = tuple(values)
    except TypeError as exc:
        raise ConditionalBlanketError(f"{field} must be a sample sequence") from exc
    if not 1 <= len(rows) <= MAX_SAMPLES_PER_SPLIT or any(
        not isinstance(row, CategoricalSample) for row in rows
    ):
        raise ConditionalBlanketBoundsError(
            f"{field} must contain 1..{MAX_SAMPLES_PER_SPLIT} categorical samples"
        )
    return tuple(sorted(rows, key=lambda row: (row.temporal_index, row.sha256)))


def _assert_unique_source_group_time(
    rows: Sequence[CategoricalSample], *, split: str
) -> None:
    for field, values in (
        ("temporal index", [row.temporal_index for row in rows]),
        ("source receipt", [row.source_receipt_sha256 for row in rows]),
    ):
        if len(set(values)) != len(values):
            raise ConditionalBlanketLeakageError(
                f"{split} repeats a {field}; leakage is rejected"
            )


def _normalize_fit_evidence(
    train_values: Sequence[CategoricalSample],
    calibration_values: Sequence[CategoricalSample],
    alphabet: Sequence[str],
) -> tuple[
    tuple[CategoricalSample, ...], tuple[CategoricalSample, ...], tuple[str, ...]
]:
    train = _bounded_samples(train_values, field="train")
    calibration = _bounded_samples(calibration_values, field="calibration")
    _assert_unique_source_group_time(train, split="train")
    _assert_unique_source_group_time(calibration, split="calibration")
    schema = train[0].feature_schema
    if any(row.feature_schema != schema for row in (*train, *calibration)):
        raise ConditionalBlanketIntegrityError(
            "all samples must have an identical sorted feature schema"
        )
    target_alphabet = set(alphabet)
    if any(row.target_category not in target_alphabet for row in (*train, *calibration)):
        raise ConditionalBlanketIntegrityError(
            "sample target is absent from the pinned target alphabet"
        )
    train_sources = {row.source_receipt_sha256 for row in train}
    calibration_sources = {row.source_receipt_sha256 for row in calibration}
    train_groups = {row.group_sha256 for row in train}
    calibration_groups = {row.group_sha256 for row in calibration}
    train_times = {row.temporal_index for row in train}
    calibration_times = {row.temporal_index for row in calibration}
    if (
        train_sources & calibration_sources
        or train_groups & calibration_groups
        or train_times & calibration_times
    ):
        raise ConditionalBlanketLeakageError(
            "train/calibration source, group, or temporal overlap is rejected"
        )
    if train[-1].temporal_index >= calibration[0].temporal_index:
        raise ConditionalBlanketLeakageError(
            "calibration must be strictly later than all training evidence"
        )
    return train, calibration, schema


def _normalize_holdout(
    fit: ConditionalBlanketFitReceipt,
    holdout_values: Sequence[CategoricalSample],
) -> tuple[CategoricalSample, ...]:
    holdout = _bounded_samples(holdout_values, field="holdout")
    _assert_unique_source_group_time(holdout, split="holdout")
    if any(row.feature_schema != fit.feature_schema for row in holdout):
        raise ConditionalBlanketIntegrityError("holdout feature schema changed")
    if any(row.target_category not in fit.config.target_alphabet for row in holdout):
        raise ConditionalBlanketIntegrityError(
            "holdout target is absent from the pinned alphabet"
        )
    prior = (*fit.train_samples, *fit.calibration_samples)
    if (
        {row.source_receipt_sha256 for row in prior}
        & {row.source_receipt_sha256 for row in holdout}
        or {row.group_sha256 for row in prior}
        & {row.group_sha256 for row in holdout}
        or {row.temporal_index for row in prior}
        & {row.temporal_index for row in holdout}
    ):
        raise ConditionalBlanketLeakageError(
            "holdout overlaps a fit source, group, or temporal index"
        )
    if fit.calibration_samples[-1].temporal_index >= holdout[0].temporal_index:
        raise ConditionalBlanketLeakageError(
            "holdout must be strictly later than calibration evidence"
        )
    return holdout


def _context(sample: CategoricalSample, feature_names: Sequence[str]) -> tuple[str, ...]:
    values = sample.feature_map
    try:
        return tuple(values[name] for name in feature_names)
    except KeyError as exc:
        raise ConditionalBlanketIntegrityError(
            "sample is missing a selected feature"
        ) from exc


def _laplace_probabilities(
    counts: Sequence[int], alpha: Fraction
) -> tuple[ExactRatio, ...]:
    total = sum(counts)
    denominator = Fraction(total) + alpha * len(counts)
    return tuple(
        ExactRatio.from_fraction((Fraction(count) + alpha) / denominator)
        for count in counts
    )


def _selection_sha(
    name: str,
    feature_names: Sequence[str],
    *,
    config: ConditionalBlanketConfig,
) -> str:
    return _digest(
        {
            "config_sha256": config.sha256,
            "feature_names": list(feature_names),
            "name": name,
            "schema": "immer-ooe-conditional-blanket-arm-selection/v1",
        }
    )


def _fit_table(
    name: str,
    feature_names: tuple[str, ...],
    train: Sequence[CategoricalSample],
    calibration: Sequence[CategoricalSample],
    config: ConditionalBlanketConfig,
) -> ConditionalBlanketArm:
    alphabet = config.target_alphabet
    index = {target: position for position, target in enumerate(alphabet)}
    marginal = [0] * len(alphabet)
    counts: dict[tuple[str, ...], list[int]] = {}
    for sample in train:
        target_index = index[sample.target_category]
        marginal[target_index] += 1
        context = _context(sample, feature_names)
        counts.setdefault(context, [0] * len(alphabet))[target_index] += 1
    cells = tuple(
        ConditionalCell(
            context=context,
            target_counts=tuple(target_counts),
            probabilities=_laplace_probabilities(
                target_counts, config.laplace_alpha
            ),
        )
        for context, target_counts in sorted(counts.items())
    )
    marginal_probabilities = _laplace_probabilities(
        marginal, config.laplace_alpha
    )
    provisional = ConditionalBlanketArm(
        name=name,
        feature_names=feature_names,
        target_alphabet=alphabet,
        laplace_alpha=ExactRatio.from_fraction(config.laplace_alpha),
        training_sample_count=len(train),
        marginal_counts=tuple(marginal),
        marginal_probabilities=marginal_probabilities,
        cells=cells,
        calibration_metrics=ExactCategoricalMetrics(
            sample_count=len(calibration),
            multiclass_brier=ExactRatio(0, 1),
            top1_accuracy=ExactRatio(0, 1),
            context_coverage=ExactRatio(0, 1),
        ),
        selection_sha256=_selection_sha(name, feature_names, config=config),
    )
    metrics = _evaluate_arm(provisional, calibration)
    return ConditionalBlanketArm(
        name=name,
        feature_names=feature_names,
        target_alphabet=alphabet,
        laplace_alpha=provisional.laplace_alpha,
        training_sample_count=len(train),
        marginal_counts=provisional.marginal_counts,
        marginal_probabilities=provisional.marginal_probabilities,
        cells=provisional.cells,
        calibration_metrics=metrics,
        selection_sha256=provisional.selection_sha256,
    )


def _arm_probabilities(
    arm: ConditionalBlanketArm, sample: CategoricalSample
) -> tuple[Fraction, ...]:
    return _arm_probabilities_for_values(arm, sample.feature_map)


def _arm_probabilities_for_values(
    arm: ConditionalBlanketArm, values: Mapping[str, str]
) -> tuple[Fraction, ...]:
    try:
        context = tuple(values[name] for name in arm.feature_names)
    except KeyError as exc:
        raise ConditionalBlanketIntegrityError(
            "prediction is missing a selected feature"
        ) from exc
    cells = {cell.context: cell for cell in arm.cells}
    cell = cells.get(context)
    values = (
        arm.marginal_probabilities if cell is None else cell.probabilities
    )
    return tuple(value.fraction for value in values)


def _evaluate_arm(
    arm: ConditionalBlanketArm,
    samples: Sequence[CategoricalSample],
) -> ExactCategoricalMetrics:
    if not samples:
        raise ConditionalBlanketBoundsError("metrics require at least one sample")
    target_index = {target: index for index, target in enumerate(arm.target_alphabet)}
    contexts = {cell.context for cell in arm.cells}
    cell_map = {cell.context: cell for cell in arm.cells}
    by_group: dict[str, list[CategoricalSample]] = defaultdict(list)
    for sample in samples:
        by_group[sample.group_sha256].append(sample)
    brier = Fraction()
    accuracy = Fraction()
    coverage = Fraction()
    for group_samples in by_group.values():
        group_brier = Fraction()
        group_correct = 0
        group_covered = 0
        for sample in group_samples:
            context = _context(sample, arm.feature_names)
            cell = cell_map.get(context)
            probability_rows = (
                arm.marginal_probabilities if cell is None else cell.probabilities
            )
            probabilities = tuple(value.fraction for value in probability_rows)
            actual = target_index[sample.target_category]
            group_brier += sum(
                (
                    probability
                    - (Fraction(1) if index == actual else Fraction())
                )
                ** 2
                for index, probability in enumerate(probabilities)
            )
            maximum = max(probabilities)
            prediction = next(
                index
                for index, probability in enumerate(probabilities)
                if probability == maximum
            )
            group_correct += prediction == actual
            group_covered += context in contexts
        group_size = len(group_samples)
        brier += group_brier / group_size
        accuracy += Fraction(group_correct, group_size)
        coverage += Fraction(group_covered, group_size)
    group_count = len(by_group)
    count = len(samples)
    return ExactCategoricalMetrics(
        sample_count=count,
        multiclass_brier=ExactRatio.from_fraction(brier / group_count),
        top1_accuracy=ExactRatio.from_fraction(accuracy / group_count),
        context_coverage=ExactRatio.from_fraction(coverage / group_count),
    )


def _dependence_residual(
    selected: tuple[str, ...],
    excluded: tuple[str, ...],
    samples: Sequence[CategoricalSample],
    alphabet: tuple[str, ...],
) -> Fraction:
    target_index = {target: index for index, target in enumerate(alphabet)}
    selected_counts: dict[tuple[str, ...], list[Fraction]] = defaultdict(
        lambda: [Fraction()] * len(alphabet)
    )
    joint_counts: dict[
        tuple[tuple[str, ...], tuple[str, ...]], list[Fraction]
    ] = defaultdict(
        lambda: [Fraction()] * len(alphabet)
    )
    group_sizes = Counter(sample.group_sha256 for sample in samples)
    group_count = len(group_sizes)
    for sample in samples:
        selected_context = _context(sample, selected)
        excluded_context = _context(sample, excluded)
        index = target_index[sample.target_category]
        weight = Fraction(1, group_count * group_sizes[sample.group_sha256])
        selected_counts[selected_context][index] += weight
        joint_counts[(selected_context, excluded_context)][index] += weight
    residual = Fraction()
    for (selected_context, _), joint in joint_counts.items():
        selected_row = selected_counts[selected_context]
        joint_total = sum(joint)
        selected_total = sum(selected_row)
        total_variation = sum(
            abs(Fraction(joint_y, joint_total) - Fraction(selected_y, selected_total))
            for joint_y, selected_y in zip(joint, selected_row, strict=True)
        ) / 2
        residual += joint_total * total_variation
    return residual


def _pair_checks(
    excluded: Sequence[str], config: ConditionalBlanketConfig
) -> tuple[tuple[str, str], ...]:
    pairs = tuple(combinations(excluded, 2))
    if len(pairs) <= config.max_pair_checks:
        return pairs
    ranked = sorted(
        pairs,
        key=lambda pair: (
            _digest(
                {
                    "config_sha256": config.sha256,
                    "features": list(pair),
                    "schema": "immer-ooe-blanket-pair-rank/v1",
                }
            ),
            pair,
        ),
    )
    return tuple(sorted(ranked[: config.max_pair_checks]))


def _dependence_checks(
    selected: tuple[str, ...],
    schema: tuple[str, ...],
    calibration: Sequence[CategoricalSample],
    config: ConditionalBlanketConfig,
) -> tuple[ConditionalDependenceCheck, ...]:
    excluded = tuple(name for name in schema if name not in selected)
    variables: Iterable[tuple[str, ...]] = (
        *((name,) for name in excluded),
        *_pair_checks(excluded, config),
    )
    return tuple(
        ConditionalDependenceCheck(
            excluded_feature_names=names,
            residual=ExactRatio.from_fraction(
                _dependence_residual(
                    selected,
                    names,
                    calibration,
                    config.target_alphabet,
                )
            ),
        )
        for names in variables
    )


def _qualifies(
    metrics: ExactCategoricalMetrics,
    checks: Sequence[ConditionalDependenceCheck],
    full: ExactCategoricalMetrics,
    config: ConditionalBlanketConfig,
) -> bool:
    maximum_residual = max(
        (item.residual.fraction for item in checks), default=Fraction()
    )
    return (
        metrics.brier <= full.brier + config.max_brier_regret
        and metrics.accuracy + config.max_accuracy_regret >= full.accuracy
        and metrics.coverage + config.max_coverage_regret >= full.coverage
        and maximum_residual <= config.dependence_tolerance
    )


def _candidate_subsets(
    schema: tuple[str, ...], config: ConditionalBlanketConfig
) -> tuple[tuple[str, ...], ...]:
    maximum = min(len(schema), config.max_subset_size)
    count = sum(math.comb(len(schema), size) for size in range(maximum + 1))
    if count > config.max_exhaustive_subsets:
        raise ConditionalBlanketBoundsError(
            f"exact search needs {count} subsets, above max_exhaustive_subsets="
            f"{config.max_exhaustive_subsets}"
        )
    return tuple(
        subset
        for size in range(maximum + 1)
        for subset in combinations(schema, size)
    )


def _placebo_subset(
    schema: tuple[str, ...],
    selected: tuple[str, ...],
    config: ConditionalBlanketConfig,
) -> tuple[str, ...]:
    choices = tuple(combinations(schema, len(selected)))
    alternatives = tuple(choice for choice in choices if choice != selected)
    pool = alternatives or choices
    return min(
        pool,
        key=lambda subset: (
            _digest(
                {
                    "feature_names": list(subset),
                    "seed_sha256": config.placebo_seed_sha256,
                    "schema": "immer-ooe-blanket-placebo-rank/v1",
                }
            ),
            subset,
        ),
    )


def _fit_core(
    train_values: Sequence[CategoricalSample],
    calibration_values: Sequence[CategoricalSample],
    config: ConditionalBlanketConfig,
) -> ConditionalBlanketFitReceipt:
    train, calibration, schema = _normalize_fit_evidence(
        train_values, calibration_values, config.target_alphabet
    )
    full_arm = _fit_table("full", schema, train, calibration, config)
    candidates = _candidate_subsets(schema, config)
    candidate_inventory_sha256 = _digest(
        {
            "config_sha256": config.sha256,
            "schema": "immer-ooe-blanket-subset-inventory/v1",
            "subsets": [list(subset) for subset in candidates],
        }
    )
    qualified: list[
        tuple[
            tuple[Any, ...],
            tuple[str, ...],
            ConditionalBlanketArm,
            tuple[ConditionalDependenceCheck, ...],
        ]
    ] = []
    for subset in candidates:
        candidate = _fit_table("selected", subset, train, calibration, config)
        checks = _dependence_checks(subset, schema, calibration, config)
        metrics = candidate.calibration_metrics
        if _qualifies(metrics, checks, full_arm.calibration_metrics, config):
            residual = max(
                (item.residual.fraction for item in checks), default=Fraction()
            )
            key = (
                len(subset),
                metrics.brier,
                -metrics.accuracy,
                -metrics.coverage,
                residual,
                subset,
            )
            qualified.append((key, subset, candidate, checks))
    if qualified:
        _, selected, selected_arm, selected_checks = min(
            qualified, key=lambda item: item[0]
        )
    else:
        selected = schema
        selected_arm = _fit_table("selected", schema, train, calibration, config)
        selected_checks = ()
    search_covers_full_power_set = config.max_subset_size >= len(schema)
    capacity_exhausted = not search_covers_full_power_set
    closure_verified = search_covers_full_power_set
    closure_subsets: tuple[tuple[str, ...], ...] = ()
    closure_evaluations: list[dict[str, Any]] = []
    boundary = min(config.max_subset_size, len(schema))
    if qualified and len(selected) == boundary and boundary < len(schema):
        closure_subsets = tuple(
            tuple(sorted((*selected, excluded)))
            for excluded in schema
            if excluded not in selected
        )
        selected_metrics = selected_arm.calibration_metrics
        for superset in closure_subsets:
            arm = _fit_table("selected", superset, train, calibration, config)
            checks = _dependence_checks(superset, schema, calibration, config)
            metrics = arm.calibration_metrics
            preserved = (
                selected_metrics.brier
                <= metrics.brier + config.max_brier_regret
                and selected_metrics.accuracy + config.max_accuracy_regret
                >= metrics.accuracy
                and selected_metrics.coverage + config.max_coverage_regret
                >= metrics.coverage
            )
            closure_evaluations.append(
                {
                    "dependence_residual": ExactRatio.from_fraction(
                        max(
                            (
                                item.residual.fraction
                                for item in checks
                            ),
                            default=Fraction(),
                        )
                    ).to_dict(),
                    "feature_names": list(superset),
                    "metrics": metrics.to_dict(),
                    "selected_preserved": preserved,
                }
            )
        # This frontier remains a useful diagnostic, but it is not a closure
        # proof: a higher-order interaction may be invisible to every single,
        # pair, and one-step check.  Only the complete power set closes that
        # possibility.
    closure_inventory_sha256 = _digest(
        {
            "config_sha256": config.sha256,
            "schema": "immer-ooe-blanket-closure-inventory/v1",
            "subsets": [list(subset) for subset in closure_subsets],
        }
    )
    closure_evaluation_sha256 = _digest(
        {
            "evaluations": closure_evaluations,
            "inventory_sha256": closure_inventory_sha256,
            "schema": "immer-ooe-blanket-closure-evaluation/v1",
        }
    )
    random_features = _placebo_subset(schema, selected, config)
    random_arm = _fit_table("random", random_features, train, calibration, config)
    marginal_arm = _fit_table("marginal", (), train, calibration, config)
    return ConditionalBlanketFitReceipt(
        config=config,
        feature_schema=schema,
        train_samples=train,
        calibration_samples=calibration,
        status=(
            "capacity_exhausted"
            if capacity_exhausted
            else ("sparse" if len(selected) < len(schema) else "full")
        ),
        candidate_count=len(candidates),
        candidate_subset_inventory_sha256=candidate_inventory_sha256,
        qualifying_candidate_count=len(qualified),
        closure_candidate_count=len(closure_subsets),
        closure_subset_inventory_sha256=closure_inventory_sha256,
        closure_evaluation_sha256=closure_evaluation_sha256,
        closure_verified=closure_verified,
        capacity_exhausted=capacity_exhausted,
        selected_dependence_checks=selected_checks,
        full_arm=full_arm,
        selected_arm=selected_arm,
        random_arm=random_arm,
        marginal_arm=marginal_arm,
    )


def fit_conditional_blanket(
    train: Sequence[CategoricalSample],
    calibration: Sequence[CategoricalSample],
    *,
    config: ConditionalBlanketConfig,
) -> ConditionalBlanketFitReceipt:
    """Fit the smallest qualified blanket without observing any holdout row."""

    if not isinstance(config, ConditionalBlanketConfig):
        raise TypeError("config must be ConditionalBlanketConfig")
    return _fit_core(train, calibration, config)


def predict_probabilities(
    fit: ConditionalBlanketFitReceipt,
    sample: CategoricalSample
    | Mapping[str, str]
    | Sequence[CategoricalFeatureAtom],
    *,
    arm: str = "selected",
) -> tuple[Fraction, ...]:
    """Return exact probabilities; unseen contexts use the fitted marginal."""

    if not isinstance(fit, ConditionalBlanketFitReceipt):
        raise TypeError("fit must be ConditionalBlanketFitReceipt")
    if isinstance(sample, CategoricalSample):
        feature_map = sample.feature_map
        feature_schema = sample.feature_schema
    elif isinstance(sample, Mapping):
        feature_map = {
            _canonical_text(name, field="prediction.feature_name", maximum=128): (
                _canonical_text(
                    category, field="prediction.feature_category", maximum=512
                )
            )
            for name, category in sample.items()
        }
        feature_schema = tuple(sorted(feature_map))
    else:
        if isinstance(sample, (str, bytes, bytearray)):
            raise TypeError("prediction features must be atoms or a mapping")
        try:
            atoms = tuple(sample)
        except TypeError as exc:
            raise TypeError("prediction features must be atoms or a mapping") from exc
        if any(not isinstance(atom, CategoricalFeatureAtom) for atom in atoms):
            raise TypeError("prediction features must contain categorical atoms")
        feature_map = {atom.name: atom.category for atom in atoms}
        if len(feature_map) != len(atoms):
            raise ConditionalBlanketIntegrityError(
                "prediction feature names are duplicated"
            )
        feature_schema = tuple(sorted(feature_map))
    if feature_schema != fit.feature_schema:
        raise ConditionalBlanketIntegrityError("prediction feature schema changed")
    arms = {
        "full": fit.full_arm,
        "selected": fit.selected_arm,
        "random": fit.random_arm,
        "marginal": fit.marginal_arm,
    }
    if arm not in arms:
        raise ConditionalBlanketError("arm must be full/selected/random/marginal")
    if arm == "selected" and fit.abstained:
        raise ConditionalBlanketAbstentionError(
            "selected blanket search did not cover the full power set"
        )
    return _arm_probabilities_for_values(arms[arm], feature_map)


def predict_category(
    fit: ConditionalBlanketFitReceipt,
    sample: CategoricalSample
    | Mapping[str, str]
    | Sequence[CategoricalFeatureAtom],
    *,
    arm: str = "selected",
) -> str:
    probabilities = predict_probabilities(fit, sample, arm=arm)
    maximum = max(probabilities)
    index = next(
        index for index, probability in enumerate(probabilities) if probability == maximum
    )
    return fit.config.target_alphabet[index]


def _validate_core(
    fit: ConditionalBlanketFitReceipt,
    holdout_values: Sequence[CategoricalSample],
) -> ConditionalBlanketValidationReceipt:
    holdout = _normalize_holdout(fit, holdout_values)
    return ConditionalBlanketValidationReceipt(
        fit=fit,
        holdout_samples=holdout,
        full_metrics=_evaluate_arm(fit.full_arm, holdout),
        selected_metrics=_evaluate_arm(fit.selected_arm, holdout),
        random_metrics=_evaluate_arm(fit.random_arm, holdout),
        marginal_metrics=_evaluate_arm(fit.marginal_arm, holdout),
    )


def validate_conditional_blanket(
    fit: ConditionalBlanketFitReceipt,
    holdout: Sequence[CategoricalSample],
) -> ConditionalBlanketValidationReceipt:
    """Evaluate all four frozen arms on strictly later, disjoint evidence."""

    if not isinstance(fit, ConditionalBlanketFitReceipt):
        raise TypeError("fit must be ConditionalBlanketFitReceipt")
    fit.verify_or_raise()
    return _validate_core(fit, holdout)


def verify_conditional_blanket_fit(fit: ConditionalBlanketFitReceipt) -> bool:
    if not isinstance(fit, ConditionalBlanketFitReceipt):
        raise TypeError("fit must be ConditionalBlanketFitReceipt")
    return fit.verify_or_raise()


def verify_conditional_blanket_validation(
    receipt: ConditionalBlanketValidationReceipt,
) -> bool:
    if not isinstance(receipt, ConditionalBlanketValidationReceipt):
        raise TypeError("receipt must be ConditionalBlanketValidationReceipt")
    return receipt.verify_or_raise()


__all__ = [
    "CATEGORICAL_SAMPLE_SCHEMA",
    "CONDITIONAL_BLANKET_CONFIG_SCHEMA",
    "CONDITIONAL_BLANKET_FIT_SCHEMA",
    "CONDITIONAL_BLANKET_VALIDATION_SCHEMA",
    "FEATURE_ATOM_SCHEMA",
    "CategoricalFeatureAtom",
    "CategoricalSample",
    "ConditionalBlanketArm",
    "ConditionalBlanketAbstentionError",
    "ConditionalBlanketBoundsError",
    "ConditionalBlanketConfig",
    "ConditionalBlanketError",
    "ConditionalBlanketFitReceipt",
    "ConditionalBlanketIntegrityError",
    "ConditionalBlanketLeakageError",
    "ConditionalBlanketValidationReceipt",
    "ConditionalDependenceCheck",
    "ExactCategoricalMetrics",
    "ExactRatio",
    "fit_conditional_blanket",
    "predict_category",
    "predict_probabilities",
    "validate_conditional_blanket",
    "verify_conditional_blanket_fit",
    "verify_conditional_blanket_validation",
]
