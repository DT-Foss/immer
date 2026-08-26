"""Prompt-preserving contextual sequence prediction.

This module is an instrument, not an executable ComputeCrystal.  It measures
whether a fixed recurrent substrate can predict contextual hidden-state
residuals on a later prompt.  Training, validation, and final holdout are
separate APIs so holdout evidence can never enter fitted model bytes.

For an input sequence ``X`` the model evaluates

``Y_hat = X + Psi(X) @ C``

where ``Psi`` contains a bias, token-wise RMS-normalized input, and optionally
one or more fixed recurrent traces

``r_t = (1-eta) r_(t-1) + eta tanh(Wr r_(t-1) + Win x_t)``.

The recurrent state is reset for every prompt.  Only the ridge readout ``C``
is fitted.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field as dataclass_field
import hashlib
import json
import math
import os
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .crystal import CrystalStore, CrystalStoreError, StatePublication
from .identity import canonical_json_bytes, require_sha256


SAMPLE_SCHEMA = "immer-ooe-contextual-sequence-sample/v1"
SAMPLE_IDENTITY_SCHEMA = "immer-ooe-contextual-sequence-sample-identity/v1"
FEATURE_CONFIG_SCHEMA = "immer-ooe-contextual-sequence-feature-config/v1"
FEATURE_SEMANTIC_SCHEMA = "immer-ooe-contextual-sequence-feature-semantic/v1"
MODEL_SCHEMA = "immer-ooe-contextual-sequence-model/v1"
FIT_RECEIPT_SCHEMA = "immer-ooe-contextual-sequence-fit-receipt/v1"
FIT_ARTIFACT_SCHEMA = "immer-ooe-contextual-sequence-fit/v1"
HOLDOUT_RECEIPT_SCHEMA = "immer-ooe-contextual-sequence-holdout/v1"
SUBSTRATE_SCHEMA = "immer-fixed-seeded-multitimescale-substrate/v1"

MAX_TOKENS = 1_000_000
MAX_DIMENSION = 65_536
MAX_RESERVOIR_SIZE = 4_096
MAX_LEAKS = 16
MAX_SAMPLE_BYTES = 512 * 1024 * 1024
MAX_MODEL_BYTES = 512 * 1024 * 1024
MAX_ARTIFACT_BYTES = 768 * 1024 * 1024
DEFAULT_MAX_FIT_WORKING_BYTES = 1024 * 1024 * 1024

FloatArray = NDArray[np.float64]


class ContextualSequenceError(RuntimeError):
    """Base error for contextual sequence prediction."""


class ContextualSequenceIntegrityError(ContextualSequenceError, ValueError):
    """A sequence artifact failed structural or hash validation."""


class ContextualSequenceCapacityError(ContextualSequenceError, MemoryError):
    """A fit would exceed its aggregate working-memory contract."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


SEQUENCE_FIT_PROTOCOL_SHA256 = _digest(
    {
        "feature_map": "[1,rms(x),r_eta,(r_eta)^2?]",
        "fit": "ridge-on-ordered-training-prompts-only",
        "holdout": "separate-predictive-evaluation",
        "model": "Yhat=X+Psi@C",
        "recurrence": ("r_t=(1-eta)r_(t-1)+eta*tanh(Wr@r_(t-1)+Win@rms(x_t))"),
        "schema": "immer-contextual-sequence-protocol/v1",
        "selection": "one-later-validation-prompt",
        "state_reset": "per-prompt",
        "substrate": SUBSTRATE_SCHEMA,
    }
)


def _identifier(value: object, *, field: str, maximum: int = 128) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > maximum
    ):
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _positive_integer(value: object, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 1 <= int(value) <= maximum
    ):
        raise ValueError(f"{field} must lie in [1, {maximum}]")
    return int(value)


def _seed(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 0 <= int(value) <= 2**63 - 1
    ):
        raise ValueError("seed must be a non-negative signed 64-bit integer")
    return int(value)


def _finite_positive(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field} must be a finite positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be a finite positive number")
    return result


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return 0.0 if result == 0.0 else result


def _strict_json(data: bytes, *, label: str, maximum: int) -> object:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if not 1 <= len(data) <= maximum:
        raise ContextualSequenceIntegrityError(f"{label} exceeds its byte bound")

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
        raise ContextualSequenceIntegrityError(f"{label} is not strict JSON") from exc
    if canonical_json_bytes(value) != data:
        raise ContextualSequenceIntegrityError(f"{label} is not canonical JSON")
    return value


def _exact_keys(
    value: object, expected: set[str], *, field: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ContextualSequenceIntegrityError(f"{field} has unknown or missing fields")
    return value


def _seal(schema: str, body: Mapping[str, object]) -> bytes:
    canonical_body = dict(body)
    return canonical_json_bytes(
        {"body": canonical_body, "schema": schema, "sha256": _digest(canonical_body)}
    )


def _open(
    data: bytes, *, schema: str, label: str, maximum: int
) -> Mapping[str, object]:
    root = _strict_json(data, label=label, maximum=maximum)
    envelope = _exact_keys(root, {"body", "schema", "sha256"}, field=label)
    if envelope["schema"] != schema:
        raise ContextualSequenceIntegrityError(f"unsupported {label} schema")
    body = envelope["body"]
    if not isinstance(body, Mapping):
        raise ContextualSequenceIntegrityError(f"{label} body is invalid")
    try:
        claimed = require_sha256(envelope["sha256"], field=f"{label}.sha256")
    except ValueError as exc:
        raise ContextualSequenceIntegrityError(f"{label} digest is invalid") from exc
    if claimed != _digest(body):
        raise ContextualSequenceIntegrityError(f"{label} digest mismatch")
    return body


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(value: object, *, field: str, maximum: int) -> bytes:
    if (
        not isinstance(value, str)
        or not value.isascii()
        or len(value) > 4 * ((maximum + 2) // 3)
    ):
        raise ContextualSequenceIntegrityError(f"{field} encoding is invalid")
    try:
        data = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ContextualSequenceIntegrityError(f"{field} encoding is invalid") from exc
    if not 1 <= len(data) <= maximum:
        raise ContextualSequenceIntegrityError(f"{field} exceeds its byte bound")
    if _b64(data) != value:
        raise ContextualSequenceIntegrityError(f"{field} encoding is not canonical")
    return data


def _freeze_matrix(value: ArrayLike, *, field: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{field} must be a rank-two array")
    rows = _positive_integer(array.shape[0], field=f"{field} rows", maximum=MAX_TOKENS)
    columns = _positive_integer(
        array.shape[1], field=f"{field} columns", maximum=MAX_DIMENSION
    )
    if rows * columns * 8 > MAX_SAMPLE_BYTES:
        raise ValueError(f"{field} exceeds its byte bound")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{field} must contain only finite values")
    canonical = np.array(array, dtype="<f8", order="C", copy=True)
    canonical[canonical == 0.0] = 0.0
    raw = canonical.tobytes(order="C")
    # A bytes-backed view cannot have WRITEABLE re-enabled by a caller.
    return np.frombuffer(raw, dtype="<f8").reshape((rows, columns))


def _array_record(value: FloatArray) -> dict[str, object]:
    array = _freeze_matrix(value, field="array")
    raw = array.tobytes(order="C")
    return {
        "byte_count": len(raw),
        "data_base64": _b64(raw),
        "dtype": "<f8",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "shape": [int(array.shape[0]), int(array.shape[1])],
    }


def _array_from_record(value: object, *, field: str) -> FloatArray:
    descriptor = _exact_keys(
        value,
        {"byte_count", "data_base64", "dtype", "raw_sha256", "shape"},
        field=field,
    )
    shape = descriptor["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in shape)
    ):
        raise ContextualSequenceIntegrityError(f"{field} shape is invalid")
    rows = _positive_integer(shape[0], field=f"{field} rows", maximum=MAX_TOKENS)
    columns = _positive_integer(
        shape[1], field=f"{field} columns", maximum=MAX_DIMENSION
    )
    expected = rows * columns * 8
    if (
        descriptor["dtype"] != "<f8"
        or isinstance(descriptor["byte_count"], bool)
        or descriptor["byte_count"] != expected
        or expected > MAX_SAMPLE_BYTES
    ):
        raise ContextualSequenceIntegrityError(f"{field} descriptor is invalid")
    raw = _unb64(
        descriptor["data_base64"], field=f"{field}.data_base64", maximum=expected
    )
    try:
        claimed = require_sha256(descriptor["raw_sha256"], field=f"{field}.raw_sha256")
    except ValueError as exc:
        raise ContextualSequenceIntegrityError(f"{field} digest is invalid") from exc
    if len(raw) != expected or hashlib.sha256(raw).hexdigest() != claimed:
        raise ContextualSequenceIntegrityError(f"{field} payload digest mismatch")
    array = np.frombuffer(raw, dtype="<f8").reshape((rows, columns))
    if not bool(np.isfinite(array).all()):
        raise ContextualSequenceIntegrityError(f"{field} contains non-finite values")
    if bool(np.any((array == 0.0) & np.signbit(array))):
        raise ContextualSequenceIntegrityError(f"{field} contains negative zero")
    return array


def _sequence_content_sha256(x: FloatArray, y: FloatArray) -> str:
    """Identify raw prompt content independently of relabelable metadata."""

    x_raw = x.tobytes(order="C")
    y_raw = y.tobytes(order="C")
    return _digest(
        {
            "schema": "immer-ooe-contextual-sequence-content/v1",
            "x": {
                "raw_sha256": hashlib.sha256(x_raw).hexdigest(),
                "shape": [int(x.shape[0]), int(x.shape[1])],
            },
            "y": {
                "raw_sha256": hashlib.sha256(y_raw).hexdigest(),
                "shape": [int(y.shape[0]), int(y.shape[1])],
            },
        }
    )


@dataclass(frozen=True, slots=True, eq=False)
class SequenceSample:
    """One complete prompt sequence with immutable contextual evidence."""

    x: FloatArray
    y: FloatArray
    prompt_sha256: str
    evidence_sha256: str
    _content_sha256: str = dataclass_field(init=False, repr=False)

    def __post_init__(self) -> None:
        frozen_x = _freeze_matrix(self.x, field="x")
        frozen_y = _freeze_matrix(self.y, field="y")
        if frozen_x.shape != frozen_y.shape:
            raise ValueError("x and y must have the same prompt-preserving shape")
        object.__setattr__(self, "x", frozen_x)
        object.__setattr__(self, "y", frozen_y)
        object.__setattr__(
            self,
            "prompt_sha256",
            require_sha256(self.prompt_sha256, field="prompt_sha256"),
        )
        object.__setattr__(
            self,
            "evidence_sha256",
            require_sha256(self.evidence_sha256, field="evidence_sha256"),
        )
        object.__setattr__(
            self,
            "_content_sha256",
            _sequence_content_sha256(frozen_x, frozen_y),
        )

    @property
    def token_count(self) -> int:
        return int(self.x.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.x.shape[1])

    @property
    def content_sha256(self) -> str:
        """Hash of canonical X/Y content, excluding prompt/evidence labels."""

        return self._content_sha256

    @property
    def X(self) -> FloatArray:
        """Mathematical alias for the immutable input sequence."""

        return self.x

    @property
    def Y(self) -> FloatArray:
        """Mathematical alias for the immutable target sequence."""

        return self.y

    def _body(self) -> dict[str, object]:
        return {
            "content_sha256": self.content_sha256,
            "evidence_sha256": self.evidence_sha256,
            "prompt_sha256": self.prompt_sha256,
            "x": _array_record(self.x),
            "y": _array_record(self.y),
        }

    def to_bytes(self) -> bytes:
        return _seal(SAMPLE_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @property
    def identity(self) -> "SequenceSampleIdentity":
        return SequenceSampleIdentity(
            prompt_sha256=self.prompt_sha256,
            evidence_sha256=self.evidence_sha256,
            sample_sha256=self.sha256,
            content_sha256=self.content_sha256,
            token_count=self.token_count,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "SequenceSample":
        body = _open(
            data,
            schema=SAMPLE_SCHEMA,
            label="sequence sample",
            maximum=MAX_SAMPLE_BYTES * 3,
        )
        _exact_keys(
            body,
            {"content_sha256", "evidence_sha256", "prompt_sha256", "x", "y"},
            field="sequence sample body",
        )
        try:
            result = cls(
                x=_array_from_record(body["x"], field="x"),
                y=_array_from_record(body["y"], field="y"),
                prompt_sha256=body["prompt_sha256"],
                evidence_sha256=body["evidence_sha256"],
            )
        except (TypeError, ValueError) as exc:
            raise ContextualSequenceIntegrityError("invalid sequence sample") from exc
        try:
            claimed_content = require_sha256(
                body["content_sha256"], field="content_sha256"
            )
        except ValueError as exc:
            raise ContextualSequenceIntegrityError(
                "sequence content digest is invalid"
            ) from exc
        if result.content_sha256 != claimed_content:
            raise ContextualSequenceIntegrityError("sequence content digest mismatch")
        if result._body() != body or result.to_bytes() != data:
            raise ContextualSequenceIntegrityError("sequence sample bindings changed")
        return result


@dataclass(frozen=True, slots=True)
class SequenceSampleIdentity:
    prompt_sha256: str
    evidence_sha256: str
    sample_sha256: str
    content_sha256: str
    token_count: int

    def __post_init__(self) -> None:
        for field in (
            "prompt_sha256",
            "evidence_sha256",
            "sample_sha256",
            "content_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        object.__setattr__(
            self,
            "token_count",
            _positive_integer(
                self.token_count, field="token_count", maximum=MAX_TOKENS
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "content_sha256": self.content_sha256,
            "evidence_sha256": self.evidence_sha256,
            "format": SAMPLE_IDENTITY_SCHEMA,
            "prompt_sha256": self.prompt_sha256,
            "sample_sha256": self.sample_sha256,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SequenceSampleIdentity":
        body = _exact_keys(
            value,
            {
                "evidence_sha256",
                "content_sha256",
                "format",
                "prompt_sha256",
                "sample_sha256",
                "token_count",
            },
            field="sample identity",
        )
        if body["format"] != SAMPLE_IDENTITY_SCHEMA:
            raise ContextualSequenceIntegrityError("sample identity format is invalid")
        try:
            return cls(
                prompt_sha256=body["prompt_sha256"],
                evidence_sha256=body["evidence_sha256"],
                sample_sha256=body["sample_sha256"],
                content_sha256=body["content_sha256"],
                token_count=body["token_count"],
            )
        except (TypeError, ValueError) as exc:
            raise ContextualSequenceIntegrityError("invalid sample identity") from exc


@dataclass(frozen=True, slots=True)
class SequenceFeatureConfig:
    """A fixed feature-map candidate; an empty leak tuple is pointwise."""

    name: str
    leaks: tuple[float, ...] = ()
    square_lift: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, field="config name"))
        if not isinstance(self.square_lift, bool):
            raise ValueError("square_lift must be boolean")
        try:
            leaks = tuple(float(value) for value in self.leaks)
        except (TypeError, ValueError) as exc:
            raise ValueError("leaks must be a finite sequence") from exc
        if len(leaks) > MAX_LEAKS:
            raise ValueError(f"at most {MAX_LEAKS} leak timescales are supported")
        if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in leaks):
            raise ValueError("every leak must lie in (0, 1]")
        if len(set(leaks)) != len(leaks):
            raise ValueError("leak timescales must be unique")
        if not leaks and self.square_lift:
            raise ValueError("a pointwise configuration cannot square-lift recurrence")
        object.__setattr__(self, "leaks", leaks)

    @property
    def pointwise(self) -> bool:
        return not self.leaks

    @property
    def semantic_sha256(self) -> str:
        """Feature-map identity that cannot be changed by renaming a config."""

        return _digest(
            {
                "fit_protocol_sha256": SEQUENCE_FIT_PROTOCOL_SHA256,
                "format": FEATURE_SEMANTIC_SCHEMA,
                "leaks": list(self.leaks),
                "square_lift": self.square_lift,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "format": FEATURE_CONFIG_SCHEMA,
            "leaks": list(self.leaks),
            "name": self.name,
            "semantic_sha256": self.semantic_sha256,
            "square_lift": self.square_lift,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "SequenceFeatureConfig":
        body = _exact_keys(
            value,
            {"format", "leaks", "name", "semantic_sha256", "square_lift"},
            field="feature config",
        )
        if body["format"] != FEATURE_CONFIG_SCHEMA or not isinstance(
            body["leaks"], list
        ):
            raise ContextualSequenceIntegrityError("feature config is invalid")
        try:
            result = cls(
                name=body["name"],
                leaks=tuple(body["leaks"]),
                square_lift=body["square_lift"],
            )
        except (TypeError, ValueError) as exc:
            raise ContextualSequenceIntegrityError("invalid feature config") from exc
        try:
            claimed_semantic = require_sha256(
                body["semantic_sha256"], field="semantic_sha256"
            )
        except ValueError as exc:
            raise ContextualSequenceIntegrityError(
                "feature semantic digest is invalid"
            ) from exc
        if result.semantic_sha256 != claimed_semantic:
            raise ContextualSequenceIntegrityError("feature semantic digest mismatch")
        if result.to_dict() != body:
            raise ContextualSequenceIntegrityError("feature config bindings changed")
        return result


DEFAULT_FEATURE_CONFIGS = (
    SequenceFeatureConfig("pointwise"),
    SequenceFeatureConfig("leak-0.5", (0.5,)),
    SequenceFeatureConfig("leak-0.5-lift", (0.5,), True),
    SequenceFeatureConfig("leak-0.9", (0.9,)),
    SequenceFeatureConfig("leak-0.9-lift", (0.9,), True),
    SequenceFeatureConfig("multiscale-0.5-0.3-0.1", (0.5, 0.3, 0.1)),
    SequenceFeatureConfig("multiscale-0.5-0.3-0.1-lift", (0.5, 0.3, 0.1), True),
)
DEFAULT_RIDGE_GRID = (1e-8, 1e-6, 1e-4, 1e-2, 1.0)


@dataclass(frozen=True, slots=True)
class _FixedSubstrate:
    wr: FloatArray
    win: FloatArray

    @classmethod
    def create(
        cls, *, dimension: int, reservoir_size: int, seed: int
    ) -> "_FixedSubstrate":
        generator = np.random.Generator(np.random.PCG64(seed))
        wr = generator.standard_normal((reservoir_size, reservoir_size))
        # Deterministic infinity-norm scaling guarantees a contractive body
        # without an eigensolver or platform-dependent spectral routine.
        row_bound = float(np.max(np.sum(np.abs(wr), axis=1)))
        if not math.isfinite(row_bound) or row_bound <= 0.0:
            raise ContextualSequenceError("seeded recurrent substrate is degenerate")
        wr *= 0.9 / row_bound
        win = generator.standard_normal((reservoir_size, dimension))
        win *= 1.0 / math.sqrt(float(dimension))
        return cls(
            wr=_freeze_matrix(wr, field="wr"),
            win=_freeze_matrix(win, field="win"),
        )


def _rms_normalize(x: FloatArray, *, epsilon: float) -> FloatArray:
    # Stable RMS avoids overflow for large but finite hidden coordinates.
    scale = np.max(np.abs(x), axis=1, keepdims=True)
    scaled = np.divide(x, scale, out=np.zeros_like(x), where=scale != 0.0)
    rms = scale * np.sqrt(np.mean(scaled * scaled, axis=1, keepdims=True))
    denominator = np.maximum(rms, epsilon)
    normalized = np.divide(x, denominator)
    if not bool(np.isfinite(normalized).all()):
        raise ContextualSequenceError("RMS normalization produced non-finite values")
    normalized[normalized == 0.0] = 0.0
    return normalized


def _feature_dimension(
    *, dimension: int, reservoir_size: int, config: SequenceFeatureConfig
) -> int:
    recurrent = len(config.leaks) * reservoir_size
    if config.square_lift:
        recurrent *= 2
    return 1 + dimension + recurrent


def _feature_map(
    x: FloatArray,
    *,
    config: SequenceFeatureConfig,
    substrate: _FixedSubstrate,
    epsilon: float,
) -> FloatArray:
    normalized = _rms_normalize(x, epsilon=epsilon)
    columns: list[FloatArray] = [
        np.ones((x.shape[0], 1), dtype=np.float64),
        normalized,
    ]
    recurrent: list[FloatArray] = []
    for leak in config.leaks:
        state = np.zeros(substrate.wr.shape[0], dtype=np.float64)
        trace = np.empty((x.shape[0], substrate.wr.shape[0]), dtype=np.float64)
        for token_index in range(x.shape[0]):
            proposal = np.tanh(
                np.einsum("ij,j->i", substrate.wr, state, optimize=False)
                + np.einsum(
                    "ij,j->i", substrate.win, normalized[token_index], optimize=False
                )
            )
            state = (1.0 - leak) * state + leak * proposal
            trace[token_index] = state
        recurrent.append(trace)
    columns.extend(recurrent)
    if config.square_lift:
        columns.extend(trace * trace for trace in recurrent)
    result = np.concatenate(columns, axis=1)
    if not bool(np.isfinite(result).all()):
        raise ContextualSequenceError("sequence feature map is non-finite")
    result[result == 0.0] = 0.0
    return result


@dataclass(frozen=True, slots=True)
class CandidateValidationScore:
    ordinal: int
    config: SequenceFeatureConfig
    ridge: float
    validation_mse: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
        ):
            raise ValueError("candidate ordinal must be a non-negative integer")
        if not isinstance(self.config, SequenceFeatureConfig):
            raise TypeError("config must be a SequenceFeatureConfig")
        object.__setattr__(self, "ridge", _finite_positive(self.ridge, field="ridge"))
        object.__setattr__(
            self,
            "validation_mse",
            _finite_nonnegative(self.validation_mse, field="validation_mse"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "ordinal": self.ordinal,
            "ridge": self.ridge,
            "validation_mse": self.validation_mse,
        }

    @classmethod
    def from_dict(cls, value: object) -> "CandidateValidationScore":
        body = _exact_keys(
            value,
            {"config", "ordinal", "ridge", "validation_mse"},
            field="candidate validation score",
        )
        try:
            return cls(
                ordinal=body["ordinal"],
                config=SequenceFeatureConfig.from_dict(body["config"]),
                ridge=body["ridge"],
                validation_mse=body["validation_mse"],
            )
        except (TypeError, ValueError) as exc:
            raise ContextualSequenceIntegrityError(
                "invalid candidate validation score"
            ) from exc


@dataclass(frozen=True, slots=True, eq=False)
class ContextualSequenceModel:
    """Selected predictive readout with a reproducible fixed substrate."""

    dimension: int
    reservoir_size: int
    seed: int
    epsilon: float
    config: SequenceFeatureConfig
    ridge: float
    coefficient: FloatArray

    def __post_init__(self) -> None:
        dimension = _positive_integer(
            self.dimension, field="dimension", maximum=MAX_DIMENSION
        )
        reservoir_size = _positive_integer(
            self.reservoir_size,
            field="reservoir_size",
            maximum=MAX_RESERVOIR_SIZE,
        )
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "reservoir_size", reservoir_size)
        object.__setattr__(self, "seed", _seed(self.seed))
        object.__setattr__(
            self, "epsilon", _finite_positive(self.epsilon, field="epsilon")
        )
        if not isinstance(self.config, SequenceFeatureConfig):
            raise TypeError("config must be a SequenceFeatureConfig")
        object.__setattr__(self, "ridge", _finite_positive(self.ridge, field="ridge"))
        coefficient = _freeze_matrix(self.coefficient, field="coefficient")
        expected_shape = (
            _feature_dimension(
                dimension=dimension,
                reservoir_size=reservoir_size,
                config=self.config,
            ),
            dimension,
        )
        if coefficient.shape != expected_shape:
            raise ValueError(f"coefficient must have shape {expected_shape}")
        object.__setattr__(self, "coefficient", coefficient)

    def _body(self) -> dict[str, object]:
        return {
            "coefficient": _array_record(self.coefficient),
            "config": self.config.to_dict(),
            "dimension": self.dimension,
            "epsilon": self.epsilon,
            "reservoir_size": self.reservoir_size,
            "ridge": self.ridge,
            "seed": self.seed,
            "substrate_schema": SUBSTRATE_SCHEMA,
        }

    def to_bytes(self) -> bytes:
        return _seal(MODEL_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def feature_map(self, x: ArrayLike) -> FloatArray:
        sequence = _freeze_matrix(x, field="x")
        if sequence.shape[1] != self.dimension:
            raise ValueError(f"x must have feature dimension {self.dimension}")
        substrate = _FixedSubstrate.create(
            dimension=self.dimension,
            reservoir_size=self.reservoir_size,
            seed=self.seed,
        )
        features = _feature_map(
            sequence,
            config=self.config,
            substrate=substrate,
            epsilon=self.epsilon,
        )
        return _freeze_matrix(features, field="features")

    def predict(self, x: ArrayLike) -> FloatArray:
        sequence = _freeze_matrix(x, field="x")
        if sequence.shape[1] != self.dimension:
            raise ValueError(f"x must have feature dimension {self.dimension}")
        features = self.feature_map(sequence)
        residual = np.einsum("tf,fd->td", features, self.coefficient, optimize=False)
        prediction = np.asarray(sequence, dtype=np.float64) + residual
        if not bool(np.isfinite(prediction).all()):
            raise ContextualSequenceError("sequence prediction is non-finite")
        return _freeze_matrix(prediction, field="prediction")

    @classmethod
    def from_bytes(cls, data: bytes) -> "ContextualSequenceModel":
        body = _open(
            data,
            schema=MODEL_SCHEMA,
            label="contextual sequence model",
            maximum=MAX_MODEL_BYTES,
        )
        _exact_keys(
            body,
            {
                "coefficient",
                "config",
                "dimension",
                "epsilon",
                "reservoir_size",
                "ridge",
                "seed",
                "substrate_schema",
            },
            field="contextual sequence model body",
        )
        if body["substrate_schema"] != SUBSTRATE_SCHEMA:
            raise ContextualSequenceIntegrityError("substrate schema is invalid")
        try:
            result = cls(
                dimension=body["dimension"],
                reservoir_size=body["reservoir_size"],
                seed=body["seed"],
                epsilon=body["epsilon"],
                config=SequenceFeatureConfig.from_dict(body["config"]),
                ridge=body["ridge"],
                coefficient=_array_from_record(
                    body["coefficient"], field="coefficient"
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ContextualSequenceIntegrityError(
                "invalid contextual sequence model"
            ) from exc
        if result._body() != body or result.to_bytes() != data:
            raise ContextualSequenceIntegrityError("model bindings changed")
        return result


def _chronology_sha256(
    train_samples: Sequence[SequenceSampleIdentity],
    validation_sample: SequenceSampleIdentity,
) -> str:
    return _digest(
        {
            "ordered_train_sample_sha256": [
                item.sample_sha256 for item in train_samples
            ],
            "validation_sample_sha256": validation_sample.sample_sha256,
        }
    )


@dataclass(frozen=True, slots=True)
class ContextualSequenceFitReceipt:
    """Sealed proof of chronological training and validation selection."""

    model_sha256: str
    train_samples: tuple[SequenceSampleIdentity, ...]
    validation_sample: SequenceSampleIdentity
    candidates: tuple[CandidateValidationScore, ...]
    selected_ordinal: int
    chronology_sha256: str
    protocol_sha256: str = SEQUENCE_FIT_PROTOCOL_SHA256
    predictive_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_sha256",
            require_sha256(self.model_sha256, field="model_sha256"),
        )
        train_samples = tuple(self.train_samples)
        candidates = tuple(self.candidates)
        if not train_samples or any(
            not isinstance(item, SequenceSampleIdentity) for item in train_samples
        ):
            raise ValueError("train_samples must contain sample identities")
        if not isinstance(self.validation_sample, SequenceSampleIdentity):
            raise TypeError("validation_sample must be a SequenceSampleIdentity")
        if not candidates or any(
            not isinstance(item, CandidateValidationScore) for item in candidates
        ):
            raise ValueError("candidates must contain validation scores")
        object.__setattr__(self, "train_samples", train_samples)
        object.__setattr__(self, "candidates", candidates)
        all_samples = train_samples + (self.validation_sample,)
        prompts = [item.prompt_sha256 for item in all_samples]
        evidence = [item.evidence_sha256 for item in all_samples]
        sample_hashes = [item.sample_sha256 for item in all_samples]
        content_hashes = [item.content_sha256 for item in all_samples]
        if len(set(prompts)) != len(prompts):
            raise ValueError("training and validation prompt identities must be unique")
        if len(set(evidence)) != len(evidence):
            raise ValueError(
                "training and validation evidence identities must be unique"
            )
        if len(set(sample_hashes)) != len(sample_hashes):
            raise ValueError("training and validation samples must be unique")
        if len(set(content_hashes)) != len(content_hashes):
            raise ValueError(
                "training and validation raw sequence content must be unique"
            )
        if tuple(item.ordinal for item in candidates) != tuple(range(len(candidates))):
            raise ValueError("candidate ordinals must be contiguous and chronological")
        keys = [(item.config.semantic_sha256, item.ridge) for item in candidates]
        if len(set(keys)) != len(keys):
            raise ValueError("candidate config/ridge pairs must be unique")
        if (
            isinstance(self.selected_ordinal, bool)
            or not isinstance(self.selected_ordinal, int)
            or not 0 <= self.selected_ordinal < len(candidates)
        ):
            raise ValueError("selected_ordinal is invalid")
        best = min(candidates, key=lambda item: (item.validation_mse, item.ordinal))
        if best.ordinal != self.selected_ordinal:
            raise ValueError("selected candidate is not the validation minimum")
        object.__setattr__(
            self,
            "chronology_sha256",
            require_sha256(self.chronology_sha256, field="chronology_sha256"),
        )
        if self.chronology_sha256 != _chronology_sha256(
            train_samples, self.validation_sample
        ):
            raise ValueError("fit chronology digest mismatch")
        object.__setattr__(
            self,
            "protocol_sha256",
            require_sha256(self.protocol_sha256, field="protocol_sha256"),
        )
        if self.protocol_sha256 != SEQUENCE_FIT_PROTOCOL_SHA256:
            raise ValueError("fit protocol identity mismatch")
        if self.predictive_only is not True:
            raise ValueError("contextual sequence fits are predictive-only")

    @property
    def selected(self) -> CandidateValidationScore:
        return self.candidates[self.selected_ordinal]

    def _body(self) -> dict[str, object]:
        return {
            "candidates": [item.to_dict() for item in self.candidates],
            "chronology_sha256": self.chronology_sha256,
            "model_sha256": self.model_sha256,
            "predictive_only": True,
            "protocol_sha256": self.protocol_sha256,
            "selected_ordinal": self.selected_ordinal,
            "train_samples": [item.to_dict() for item in self.train_samples],
            "validation_sample": self.validation_sample.to_dict(),
        }

    def to_bytes(self) -> bytes:
        return _seal(FIT_RECEIPT_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ContextualSequenceFitReceipt":
        body = _open(
            data,
            schema=FIT_RECEIPT_SCHEMA,
            label="contextual sequence fit receipt",
            maximum=16 * 1024 * 1024,
        )
        _exact_keys(
            body,
            {
                "candidates",
                "chronology_sha256",
                "model_sha256",
                "predictive_only",
                "protocol_sha256",
                "selected_ordinal",
                "train_samples",
                "validation_sample",
            },
            field="fit receipt body",
        )
        if not isinstance(body["train_samples"], list) or not isinstance(
            body["candidates"], list
        ):
            raise ContextualSequenceIntegrityError("fit receipt lists are invalid")
        try:
            result = cls(
                model_sha256=body["model_sha256"],
                train_samples=tuple(
                    SequenceSampleIdentity.from_dict(item)
                    for item in body["train_samples"]
                ),
                validation_sample=SequenceSampleIdentity.from_dict(
                    body["validation_sample"]
                ),
                candidates=tuple(
                    CandidateValidationScore.from_dict(item)
                    for item in body["candidates"]
                ),
                selected_ordinal=body["selected_ordinal"],
                chronology_sha256=body["chronology_sha256"],
                protocol_sha256=body["protocol_sha256"],
                predictive_only=body["predictive_only"],
            )
        except (TypeError, ValueError) as exc:
            raise ContextualSequenceIntegrityError("invalid fit receipt") from exc
        if result._body() != body or result.to_bytes() != data:
            raise ContextualSequenceIntegrityError("fit receipt bindings changed")
        return result


@dataclass(frozen=True, slots=True)
class ContextualSequenceFit:
    """Atomic fitted model plus its train/validation receipt."""

    model: ContextualSequenceModel
    receipt: ContextualSequenceFitReceipt

    def __post_init__(self) -> None:
        if not isinstance(self.model, ContextualSequenceModel):
            raise TypeError("model must be a ContextualSequenceModel")
        if not isinstance(self.receipt, ContextualSequenceFitReceipt):
            raise TypeError("receipt must be a ContextualSequenceFitReceipt")
        if self.receipt.model_sha256 != self.model.sha256:
            raise ValueError("fit receipt does not bind the fitted model")
        selected = self.receipt.selected
        if selected.config != self.model.config or selected.ridge != self.model.ridge:
            raise ValueError(
                "fitted model does not match selected validation candidate"
            )

    def _body(self) -> dict[str, object]:
        model_bytes = self.model.to_bytes()
        receipt_bytes = self.receipt.to_bytes()
        return {
            "model_base64": _b64(model_bytes),
            "model_sha256": self.model.sha256,
            "receipt_base64": _b64(receipt_bytes),
            "receipt_sha256": self.receipt.sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(FIT_ARTIFACT_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ContextualSequenceFit":
        body = _open(
            data,
            schema=FIT_ARTIFACT_SCHEMA,
            label="contextual sequence fit",
            maximum=MAX_ARTIFACT_BYTES,
        )
        _exact_keys(
            body,
            {"model_base64", "model_sha256", "receipt_base64", "receipt_sha256"},
            field="contextual sequence fit body",
        )
        model_bytes = _unb64(
            body["model_base64"], field="model_base64", maximum=MAX_MODEL_BYTES
        )
        receipt_bytes = _unb64(
            body["receipt_base64"],
            field="receipt_base64",
            maximum=16 * 1024 * 1024,
        )
        try:
            model_sha256 = require_sha256(body["model_sha256"], field="model_sha256")
            receipt_sha256 = require_sha256(
                body["receipt_sha256"], field="receipt_sha256"
            )
        except ValueError as exc:
            raise ContextualSequenceIntegrityError(
                "fit child digest is invalid"
            ) from exc
        if hashlib.sha256(model_bytes).hexdigest() != model_sha256:
            raise ContextualSequenceIntegrityError("fit model digest mismatch")
        if hashlib.sha256(receipt_bytes).hexdigest() != receipt_sha256:
            raise ContextualSequenceIntegrityError("fit receipt digest mismatch")
        result = cls(
            model=ContextualSequenceModel.from_bytes(model_bytes),
            receipt=ContextualSequenceFitReceipt.from_bytes(receipt_bytes),
        )
        if result._body() != body or result.to_bytes() != data:
            raise ContextualSequenceIntegrityError("fit bindings changed")
        return result


def _validate_split(
    train_samples: Sequence[SequenceSample], validation_sample: SequenceSample
) -> tuple[SequenceSample, ...]:
    if not isinstance(validation_sample, SequenceSample):
        raise TypeError("validation_sample must be a SequenceSample")
    train = tuple(train_samples)
    if not train or any(not isinstance(item, SequenceSample) for item in train):
        raise ValueError("train_samples must contain at least one SequenceSample")
    all_samples = train + (validation_sample,)
    dimension = train[0].dimension
    if any(item.dimension != dimension for item in all_samples):
        raise ValueError("all train and validation samples must share one dimension")
    prompts = [item.prompt_sha256 for item in all_samples]
    evidence = [item.evidence_sha256 for item in all_samples]
    content = [item.content_sha256 for item in all_samples]
    if len(set(prompts)) != len(prompts):
        raise ValueError("train and validation prompts must be unique")
    if len(set(evidence)) != len(evidence):
        raise ValueError("train and validation evidence must be unique")
    if len(set(content)) != len(content):
        raise ValueError("train and validation raw sequence content must be unique")
    return train


def _fit_memory_components(
    train: Sequence[SequenceSample],
    validation_sample: SequenceSample,
    *,
    configs: Sequence[SequenceFeatureConfig],
    reservoir_size: int,
) -> dict[str, int]:
    """Conservatively account peak aggregate fit allocations in bytes."""

    dimension = train[0].dimension
    train_tokens = sum(item.token_count for item in train)
    validation_tokens = validation_sample.token_count
    largest_prompt = max([validation_tokens, *(item.token_count for item in train)])
    maximum_leaks = max(len(config.leaks) for config in configs)
    feature_width = max(
        _feature_dimension(
            dimension=dimension,
            reservoir_size=reservoir_size,
            config=config,
        )
        for config in configs
    )
    float_bytes = 8
    return {
        # Fixed seeded matrices are resident throughout every candidate fit.
        "substrate": (reservoir_size * reservoir_size + reservoir_size * dimension)
        * float_bytes,
        # Per-prompt blocks coexist with their concatenated training matrix.
        "training_feature_blocks_and_stack": 2
        * train_tokens
        * feature_width
        * float_bytes,
        "validation_features": validation_tokens * feature_width * float_bytes,
        # One feature-map call holds normalization, recurrent traces, and output.
        "feature_map_scratch": largest_prompt
        * (dimension + maximum_leaks * reservoir_size + feature_width)
        * float_bytes,
        # Keep a second Gram-sized allowance for the linear solver workspace.
        "gram_and_solver": 2 * feature_width * feature_width * float_bytes,
        # X^T Y and the selected readout coexist.
        "xty_and_readout": 2 * feature_width * dimension * float_bytes,
        # Per-prompt residual blocks coexist with their concatenated target.
        "training_residual_blocks_and_stack": 2
        * train_tokens
        * dimension
        * float_bytes,
        # Prediction, residual, squared-error, and target-sized validation buffers.
        "validation_prediction_buffers": 4
        * validation_tokens
        * dimension
        * float_bytes,
    }


def estimate_contextual_sequence_fit_bytes(
    train_samples: Sequence[SequenceSample],
    validation_sample: SequenceSample,
    *,
    configs: Sequence[SequenceFeatureConfig] = DEFAULT_FEATURE_CONFIGS,
    reservoir_size: int = 24,
) -> int:
    """Return the deterministic conservative aggregate fit-memory estimate."""

    train = _validate_split(train_samples, validation_sample)
    candidates = tuple(configs)
    if not candidates or any(
        not isinstance(item, SequenceFeatureConfig) for item in candidates
    ):
        raise ValueError("configs must contain feature configurations")
    if len({item.semantic_sha256 for item in candidates}) != len(candidates):
        raise ValueError("feature configurations must be semantically unique")
    reservoir_size = _positive_integer(
        reservoir_size, field="reservoir_size", maximum=MAX_RESERVOIR_SIZE
    )
    return sum(
        _fit_memory_components(
            train,
            validation_sample,
            configs=candidates,
            reservoir_size=reservoir_size,
        ).values()
    )


def _ridge_readout(
    features: FloatArray, residual: FloatArray, ridge: float
) -> FloatArray:
    gram = np.einsum("tf,tg->fg", features, features, optimize=False)
    target = np.einsum("tf,td->fd", features, residual, optimize=False)
    gram.flat[:: gram.shape[0] + 1] += ridge
    try:
        coefficient = np.linalg.solve(gram, target)
    except np.linalg.LinAlgError as exc:
        raise ContextualSequenceError("ridge system could not be solved") from exc
    if not bool(np.isfinite(coefficient).all()):
        raise ContextualSequenceError("ridge readout is non-finite")
    return coefficient


def _mse(prediction: FloatArray, target: FloatArray) -> float:
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    squared = difference * difference
    result = float(np.mean(squared))
    if not math.isfinite(result):
        raise ContextualSequenceError("mean-squared error is non-finite")
    return 0.0 if result == 0.0 else result


def fit_contextual_sequence(
    train_samples: Sequence[SequenceSample],
    validation_sample: SequenceSample,
    *,
    configs: Sequence[SequenceFeatureConfig] = DEFAULT_FEATURE_CONFIGS,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
    reservoir_size: int = 24,
    seed: int = 42,
    epsilon: float = 1e-12,
    max_working_bytes: int = DEFAULT_MAX_FIT_WORKING_BYTES,
) -> ContextualSequenceFit:
    """Fit on ordered train prompts and select once on one later prompt.

    The selected readout remains the train-only readout used during selection.
    No holdout argument exists in this API.
    """

    train = _validate_split(train_samples, validation_sample)
    candidates = tuple(configs)
    if not candidates or any(
        not isinstance(item, SequenceFeatureConfig) for item in candidates
    ):
        raise ValueError("configs must contain feature configurations")
    if len({item.semantic_sha256 for item in candidates}) != len(candidates):
        raise ValueError("feature configurations must be semantically unique")
    ridges = tuple(_finite_positive(item, field="ridge") for item in ridge_grid)
    if not ridges or len(set(ridges)) != len(ridges):
        raise ValueError("ridge_grid must contain unique positive values")
    reservoir_size = _positive_integer(
        reservoir_size, field="reservoir_size", maximum=MAX_RESERVOIR_SIZE
    )
    seed = _seed(seed)
    epsilon = _finite_positive(epsilon, field="epsilon")
    if (
        isinstance(max_working_bytes, bool)
        or not isinstance(max_working_bytes, int)
        or not 1 <= max_working_bytes <= 2**63 - 1
    ):
        raise ValueError("max_working_bytes must be a positive signed 64-bit integer")
    memory_components = _fit_memory_components(
        train,
        validation_sample,
        configs=candidates,
        reservoir_size=reservoir_size,
    )
    estimated_working_bytes = sum(memory_components.values())
    if estimated_working_bytes > max_working_bytes:
        feature_width = max(
            _feature_dimension(
                dimension=train[0].dimension,
                reservoir_size=reservoir_size,
                config=config,
            )
            for config in candidates
        )
        raise ContextualSequenceCapacityError(
            "contextual sequence fit requires "
            f"{estimated_working_bytes} aggregate working bytes for "
            f"feature_width={feature_width}, cap={max_working_bytes}"
        )
    dimension = train[0].dimension
    substrate = _FixedSubstrate.create(
        dimension=dimension, reservoir_size=reservoir_size, seed=seed
    )
    train_residuals = [
        np.asarray(item.y, dtype=np.float64) - np.asarray(item.x, dtype=np.float64)
        for item in train
    ]
    if any(not bool(np.isfinite(item).all()) for item in train_residuals):
        raise ContextualSequenceError("training residual contains non-finite values")

    scores: list[CandidateValidationScore] = []
    selected_score: CandidateValidationScore | None = None
    selected_coefficient: FloatArray | None = None
    ordinal = 0
    for config in candidates:
        feature_blocks = [
            _feature_map(item.x, config=config, substrate=substrate, epsilon=epsilon)
            for item in train
        ]
        stacked_features = np.concatenate(feature_blocks, axis=0)
        stacked_residual = np.concatenate(train_residuals, axis=0)
        validation_features = _feature_map(
            validation_sample.x,
            config=config,
            substrate=substrate,
            epsilon=epsilon,
        )
        for ridge in ridges:
            coefficient = _ridge_readout(stacked_features, stacked_residual, ridge)
            prediction = np.asarray(validation_sample.x, dtype=np.float64) + np.einsum(
                "tf,fd->td", validation_features, coefficient, optimize=False
            )
            score = CandidateValidationScore(
                ordinal=ordinal,
                config=config,
                ridge=ridge,
                validation_mse=_mse(prediction, validation_sample.y),
            )
            scores.append(score)
            if selected_score is None or (
                score.validation_mse,
                score.ordinal,
            ) < (
                selected_score.validation_mse,
                selected_score.ordinal,
            ):
                selected_score = score
                selected_coefficient = coefficient
            ordinal += 1
    if selected_score is None or selected_coefficient is None:
        raise ContextualSequenceError("candidate selection produced no fitted model")
    selected = selected_score
    coefficient = selected_coefficient
    model = ContextualSequenceModel(
        dimension=dimension,
        reservoir_size=reservoir_size,
        seed=seed,
        epsilon=epsilon,
        config=selected.config,
        ridge=selected.ridge,
        coefficient=coefficient,
    )
    train_identities = tuple(item.identity for item in train)
    validation_identity = validation_sample.identity
    receipt = ContextualSequenceFitReceipt(
        model_sha256=model.sha256,
        train_samples=train_identities,
        validation_sample=validation_identity,
        candidates=tuple(scores),
        selected_ordinal=selected.ordinal,
        chronology_sha256=_chronology_sha256(train_identities, validation_identity),
    )
    return ContextualSequenceFit(model=model, receipt=receipt)


def _permutation(
    length: int,
    *,
    fit: ContextualSequenceFit,
    holdout: SequenceSample,
    label: str,
) -> tuple[int, ...]:
    entropy = canonical_json_bytes(
        {
            "fit_sha256": fit.sha256,
            "holdout_evidence_sha256": holdout.evidence_sha256,
            "holdout_prompt_sha256": holdout.prompt_sha256,
            "label": label,
        }
    )
    seed = int.from_bytes(hashlib.sha256(entropy).digest()[:8], "big")
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(length)
    if length > 1 and bool(np.array_equal(permutation, np.arange(length))):
        permutation = np.roll(permutation, 1)
    return tuple(int(item) for item in permutation)


@dataclass(frozen=True, slots=True)
class ContextualSequenceHoldoutReceipt:
    """Predictive-only evaluation on evidence absent from fitted bytes."""

    fit_sha256: str
    model_sha256: str
    fit_receipt_sha256: str
    holdout_sample: SequenceSampleIdentity
    actual_mse: float
    shuffled_token_mse: float
    shuffled_output_mse: float
    token_permutation: tuple[int, ...]
    output_permutation: tuple[int, ...]
    predictive_only: bool = True

    def __post_init__(self) -> None:
        for field in ("fit_sha256", "model_sha256", "fit_receipt_sha256"):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        if not isinstance(self.holdout_sample, SequenceSampleIdentity):
            raise TypeError("holdout_sample must be a SequenceSampleIdentity")
        for field in ("actual_mse", "shuffled_token_mse", "shuffled_output_mse"):
            object.__setattr__(
                self,
                field,
                _finite_nonnegative(getattr(self, field), field=field),
            )
        for field in ("token_permutation", "output_permutation"):
            value = tuple(getattr(self, field))
            length = self.holdout_sample.token_count
            if (
                len(value) != length
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in value
                )
                or set(value) != set(range(length))
            ):
                raise ValueError(f"{field} must be a complete token permutation")
            if length > 1 and value == tuple(range(length)):
                raise ValueError(f"{field} must be non-trivial")
            object.__setattr__(self, field, value)
        if self.predictive_only is not True:
            raise ValueError("holdout receipt must be predictive-only")

    def _body(self) -> dict[str, object]:
        return {
            "actual_mse": self.actual_mse,
            "fit_receipt_sha256": self.fit_receipt_sha256,
            "fit_sha256": self.fit_sha256,
            "holdout_sample": self.holdout_sample.to_dict(),
            "model_sha256": self.model_sha256,
            "output_permutation": list(self.output_permutation),
            "predictive_only": True,
            "shuffled_output_mse": self.shuffled_output_mse,
            "shuffled_token_mse": self.shuffled_token_mse,
            "token_permutation": list(self.token_permutation),
        }

    def to_bytes(self) -> bytes:
        return _seal(HOLDOUT_RECEIPT_SCHEMA, self._body())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def verify(self, fit: ContextualSequenceFit, holdout: SequenceSample) -> None:
        expected = evaluate_contextual_sequence_holdout(fit, holdout)
        if self.to_bytes() != expected.to_bytes():
            raise ContextualSequenceIntegrityError(
                "holdout receipt does not replay against supplied evidence"
            )

    @classmethod
    def from_bytes(cls, data: bytes) -> "ContextualSequenceHoldoutReceipt":
        body = _open(
            data,
            schema=HOLDOUT_RECEIPT_SCHEMA,
            label="contextual sequence holdout receipt",
            maximum=16 * 1024 * 1024,
        )
        _exact_keys(
            body,
            {
                "actual_mse",
                "fit_receipt_sha256",
                "fit_sha256",
                "holdout_sample",
                "model_sha256",
                "output_permutation",
                "predictive_only",
                "shuffled_output_mse",
                "shuffled_token_mse",
                "token_permutation",
            },
            field="holdout receipt body",
        )
        if not isinstance(body["token_permutation"], list) or not isinstance(
            body["output_permutation"], list
        ):
            raise ContextualSequenceIntegrityError("holdout permutations are invalid")
        try:
            result = cls(
                fit_sha256=body["fit_sha256"],
                model_sha256=body["model_sha256"],
                fit_receipt_sha256=body["fit_receipt_sha256"],
                holdout_sample=SequenceSampleIdentity.from_dict(body["holdout_sample"]),
                actual_mse=body["actual_mse"],
                shuffled_token_mse=body["shuffled_token_mse"],
                shuffled_output_mse=body["shuffled_output_mse"],
                token_permutation=tuple(body["token_permutation"]),
                output_permutation=tuple(body["output_permutation"]),
                predictive_only=body["predictive_only"],
            )
        except (TypeError, ValueError) as exc:
            raise ContextualSequenceIntegrityError("invalid holdout receipt") from exc
        if result._body() != body or result.to_bytes() != data:
            raise ContextualSequenceIntegrityError("holdout receipt bindings changed")
        return result


def evaluate_contextual_sequence_holdout(
    fit: ContextualSequenceFit,
    holdout: SequenceSample,
) -> ContextualSequenceHoldoutReceipt:
    """Evaluate an untouched prompt without mutating or rebuilding ``fit``."""

    if not isinstance(fit, ContextualSequenceFit):
        raise TypeError("fit must be a ContextualSequenceFit")
    if not isinstance(holdout, SequenceSample):
        raise TypeError("holdout must be a SequenceSample")
    if holdout.dimension != fit.model.dimension:
        raise ValueError("holdout dimension does not match fitted model")
    prior = fit.receipt.train_samples + (fit.receipt.validation_sample,)
    if holdout.prompt_sha256 in {item.prompt_sha256 for item in prior}:
        raise ValueError("holdout prompt was already used by fit selection")
    if holdout.evidence_sha256 in {item.evidence_sha256 for item in prior}:
        raise ValueError("holdout evidence was already used by fit selection")
    if holdout.content_sha256 in {item.content_sha256 for item in prior}:
        raise ValueError(
            "holdout raw sequence content was already used by fit selection"
        )
    token_permutation = _permutation(
        holdout.token_count,
        fit=fit,
        holdout=holdout,
        label="shuffled-token-input",
    )
    output_permutation = _permutation(
        holdout.token_count,
        fit=fit,
        holdout=holdout,
        label="shuffled-output-target",
    )
    actual_prediction = fit.model.predict(holdout.x)
    shuffled_token_prediction = fit.model.predict(
        holdout.x[np.asarray(token_permutation, dtype=np.int64)]
    )
    shuffled_output = holdout.y[np.asarray(output_permutation, dtype=np.int64)]
    return ContextualSequenceHoldoutReceipt(
        fit_sha256=fit.sha256,
        model_sha256=fit.model.sha256,
        fit_receipt_sha256=fit.receipt.sha256,
        holdout_sample=holdout.identity,
        actual_mse=_mse(actual_prediction, holdout.y),
        shuffled_token_mse=_mse(shuffled_token_prediction, holdout.y),
        shuffled_output_mse=_mse(actual_prediction, shuffled_output),
        token_permutation=token_permutation,
        output_permutation=output_permutation,
    )


class ContextualSequenceBank:
    """Content-addressed CrystalStore persistence for predictive fit artifacts."""

    _PREFIX = "ooe-contextual-sequence-fit/v1:"

    def __init__(self, store: CrystalStore | str | os.PathLike[str]) -> None:
        self.store = store if isinstance(store, CrystalStore) else CrystalStore(store)

    @classmethod
    def state_name(cls, fit_sha256: str) -> str:
        return cls._PREFIX + require_sha256(fit_sha256, field="fit_sha256")

    def _require_clean(self) -> None:
        try:
            audit = self.store.audit()
        except CrystalStoreError as exc:
            raise ContextualSequenceIntegrityError(
                "sequence bank audit failed"
            ) from exc
        if not audit.clean:
            raise ContextualSequenceIntegrityError(
                "sequence bank refuses persistence while CrystalStore is not clean"
            )

    def publish(self, fit: ContextualSequenceFit) -> StatePublication:
        if not isinstance(fit, ContextualSequenceFit):
            raise TypeError("fit must be a ContextualSequenceFit")
        self._require_clean()
        payload = fit.to_bytes()
        name = self.state_name(fit.sha256)
        try:
            try:
                existing = self.store.restore_state(name)
            except KeyError:
                publication = self.store.publish_state(name, payload)
            else:
                if existing != payload:
                    raise ContextualSequenceIntegrityError(
                        "content-addressed fit state collided with different bytes"
                    )
                publication = self.store.publish_state(
                    name, payload, expected_sha256=fit.sha256
                )
        except ContextualSequenceIntegrityError:
            raise
        except CrystalStoreError as exc:
            raise ContextualSequenceIntegrityError(
                "sequence fit publication failed"
            ) from exc
        if publication.payload_sha256 != fit.sha256:
            raise ContextualSequenceIntegrityError(
                "published sequence fit digest changed"
            )
        return publication

    def restore(self, fit_sha256: str) -> ContextualSequenceFit:
        digest = require_sha256(fit_sha256, field="fit_sha256")
        self._require_clean()
        try:
            payload = self.store.restore_state(self.state_name(digest))
        except KeyError:
            raise
        except CrystalStoreError as exc:
            raise ContextualSequenceIntegrityError(
                "sequence fit state failed store integrity"
            ) from exc
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ContextualSequenceIntegrityError(
                "content-addressed fit state digest mismatch"
            )
        fit = ContextualSequenceFit.from_bytes(payload)
        if fit.sha256 != digest:
            raise ContextualSequenceIntegrityError(
                "restored sequence fit failed semantic rehash"
            )
        return fit
