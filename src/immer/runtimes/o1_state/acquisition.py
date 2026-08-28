"""Deterministic D-optimal acquisition for finite O1 probe frontiers.

The planner consumes only pre-outcome feature vectors.  It greedily maximizes
the incremental log determinant of a ridge-initialized Gram matrix, seals the
complete selection contract, and can later replay the decision against the
same content-addressed design.  Target and holdout values are never selection
inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from .cartographer import ProbeJob


D_OPTIMAL_ACQUISITION_SCHEMA = "immer.o1-d-optimal-acquisition/v1"
D_OPTIMAL_ALGORITHM = "greedy-logdet-sherman-morrison/v1"
D_OPTIMAL_NUMERIC_ABI = "ieee754-float64/log1p/symmetric-rank1/v1"
D_OPTIMAL_DESIGN_ENCODING = "job-id-sha256/float64-be/v1"
D_OPTIMAL_TIE_BREAK = "gain-desc/job-id-asc/v1"
PROBE_JOB_STRUCTURAL_DESIGN_ABI = "o1.probe-job-structural-design/v1"
PROBE_JOB_FEATURE_SCHEMA = "immer.o1-probe-job-feature-schema/v1"

MAX_CANDIDATES = 65_536
MAX_FEATURES = 4_096
MAX_SELECTIONS = 4_096
MAX_DESIGN_CELLS = 8_388_608
MAX_GRAM_CELLS = 4_194_304
MAX_RECEIPT_BYTES = 16 * 1024 * 1024
MIN_RIDGE = 1.0e-15
MAX_RIDGE = 1.0e15
MAX_ABS_FEATURE = 1.0e100
MAX_STRUCTURAL_COORDINATE = 2**63 - 1

_DESIGN_DIGEST_DOMAIN = b"immer.o1-d-optimal-design/v1\x00"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ABI_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,255}")

Normalization = Literal["none", "l2"]


class AcquisitionError(ValueError):
    """The acquisition request or design is invalid."""


class AcquisitionLeakageError(AcquisitionError):
    """Outcome or holdout values were offered to the acquisition planner."""


class AcquisitionIntegrityError(AcquisitionError):
    """A receipt or replay no longer matches its sealed acquisition."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AcquisitionIntegrityError("value is not canonical JSON") from exc


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AcquisitionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _pin(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 256
        or "\x00" in value
        or unicodedata.normalize("NFC", value) != value
    ):
        raise AcquisitionError(f"{label} must be a bounded NFC string")
    return value


def _design_abi(value: Any) -> str:
    if not isinstance(value, str) or _ABI_RE.fullmatch(value) is None:
        raise AcquisitionError("design ABI is not a canonical identifier")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AcquisitionError(f"{label} must be a positive integer")
    return value


def _bounded_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AcquisitionError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AcquisitionError(f"{label} must be a finite number")
    return 0.0 if result == 0.0 else result


def _float_hex(value: float) -> str:
    return float(value).hex()


def _parse_float_hex(value: Any, label: str) -> float:
    if not isinstance(value, str) or len(value) > 32:
        raise AcquisitionIntegrityError(f"{label} is not canonical float64 hex")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise AcquisitionIntegrityError(
            f"{label} is not canonical float64 hex"
        ) from exc
    if not math.isfinite(result) or _float_hex(result) != value:
        raise AcquisitionIntegrityError(f"{label} is not canonical float64 hex")
    return result


@dataclass(frozen=True, slots=True)
class AcquisitionLimits:
    """Explicit memory and frontier bounds sealed into each receipt."""

    candidate_capacity: int = MAX_CANDIDATES
    feature_capacity: int = MAX_FEATURES
    selection_capacity: int = MAX_SELECTIONS
    design_cell_capacity: int = MAX_DESIGN_CELLS
    gram_cell_capacity: int = MAX_GRAM_CELLS

    def __post_init__(self) -> None:
        bounds = (
            ("candidate capacity", self.candidate_capacity, MAX_CANDIDATES),
            ("feature capacity", self.feature_capacity, MAX_FEATURES),
            ("selection capacity", self.selection_capacity, MAX_SELECTIONS),
            ("design-cell capacity", self.design_cell_capacity, MAX_DESIGN_CELLS),
            ("Gram-cell capacity", self.gram_cell_capacity, MAX_GRAM_CELLS),
        )
        for label, value, hard_limit in bounds:
            _positive_int(value, label)
            if value > hard_limit:
                raise AcquisitionError(f"{label} exceeds the hard limit")

    def to_document(self) -> dict[str, int]:
        return {
            "candidate_capacity": self.candidate_capacity,
            "design_cell_capacity": self.design_cell_capacity,
            "feature_capacity": self.feature_capacity,
            "gram_cell_capacity": self.gram_cell_capacity,
            "selection_capacity": self.selection_capacity,
        }

    @classmethod
    def from_document(cls, value: Any) -> AcquisitionLimits:
        expected = {
            "candidate_capacity",
            "design_cell_capacity",
            "feature_capacity",
            "gram_cell_capacity",
            "selection_capacity",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise AcquisitionIntegrityError("acquisition limits are malformed")
        try:
            return cls(
                candidate_capacity=value["candidate_capacity"],
                feature_capacity=value["feature_capacity"],
                selection_capacity=value["selection_capacity"],
                design_cell_capacity=value["design_cell_capacity"],
                gram_cell_capacity=value["gram_cell_capacity"],
            )
        except AcquisitionIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise AcquisitionIntegrityError("acquisition limits are invalid") from exc


@dataclass(frozen=True, slots=True)
class DOptimalAcquisitionReceipt:
    """Content-addressed and replayable proof of one acquisition decision."""

    code_pin: str
    model_pin: str
    design_abi: str
    feature_pin_sha256: str
    design_sha256: str
    feature_dimension: int
    candidate_job_ids: tuple[str, ...]
    ridge_hex: str
    budget: int
    normalization: Normalization
    selected_job_ids: tuple[str, ...]
    step_logdet_gains_hex: tuple[str, ...]
    limits: AcquisitionLimits
    target_values_used: bool = False
    holdout_values_used: bool = False
    seal_sha256: str = ""

    def __post_init__(self) -> None:
        _pin(self.code_pin, "code pin")
        _pin(self.model_pin, "model pin")
        _design_abi(self.design_abi)
        _sha256(self.feature_pin_sha256, "feature pin")
        _sha256(self.design_sha256, "design SHA-256")
        dimension = _positive_int(self.feature_dimension, "feature dimension")
        budget = _positive_int(self.budget, "acquisition budget")
        if not isinstance(self.limits, AcquisitionLimits):
            raise AcquisitionError("acquisition limits are invalid")
        if dimension > self.limits.feature_capacity:
            raise AcquisitionError("feature dimension exceeds its capacity")
        if dimension * dimension > self.limits.gram_cell_capacity:
            raise AcquisitionError("Gram matrix exceeds its cell capacity")
        if not isinstance(self.candidate_job_ids, tuple):
            raise AcquisitionError("candidate job ids must be an immutable tuple")
        candidate_ids = self.candidate_job_ids
        if not candidate_ids or len(candidate_ids) > self.limits.candidate_capacity:
            raise AcquisitionError("candidate frontier exceeds its capacity")
        if candidate_ids != tuple(sorted(candidate_ids)):
            raise AcquisitionIntegrityError("candidate job ids are not canonical")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise AcquisitionIntegrityError("candidate job ids are not unique")
        for job_id in candidate_ids:
            _sha256(job_id, "candidate job id")
        if len(candidate_ids) * dimension > self.limits.design_cell_capacity:
            raise AcquisitionError("feature design exceeds its cell capacity")
        if budget > len(candidate_ids) or budget > self.limits.selection_capacity:
            raise AcquisitionError("acquisition budget exceeds its capacity")
        ridge = _parse_float_hex(self.ridge_hex, "ridge")
        if not MIN_RIDGE <= ridge <= MAX_RIDGE:
            raise AcquisitionError("ridge lies outside the numeric ABI")
        if self.normalization not in {"none", "l2"}:
            raise AcquisitionError("normalization must be none or l2")
        if not isinstance(self.selected_job_ids, tuple):
            raise AcquisitionError("selected job ids must be an immutable tuple")
        if len(self.selected_job_ids) != budget:
            raise AcquisitionIntegrityError("selected job count differs from budget")
        if len(set(self.selected_job_ids)) != budget:
            raise AcquisitionIntegrityError("selected job ids are not unique")
        if not set(self.selected_job_ids).issubset(candidate_ids):
            raise AcquisitionIntegrityError("selection is outside the frontier")
        if not isinstance(self.step_logdet_gains_hex, tuple):
            raise AcquisitionError("step gains must be an immutable tuple")
        if len(self.step_logdet_gains_hex) != budget:
            raise AcquisitionIntegrityError("step gain count differs from budget")
        for index, value in enumerate(self.step_logdet_gains_hex):
            gain = _parse_float_hex(value, f"step gain {index}")
            if gain < 0.0:
                raise AcquisitionIntegrityError("step gains must be non-negative")
        if self.target_values_used is not False or self.holdout_values_used is not False:
            raise AcquisitionLeakageError(
                "D-optimal acquisition cannot use target or holdout values"
            )
        claimed = self.seal_sha256
        if claimed:
            _sha256(claimed, "acquisition seal")
        expected = hashlib.sha256(_canonical_json(self.body())).hexdigest()
        if claimed and claimed != expected:
            raise AcquisitionIntegrityError("acquisition seal mismatch")
        object.__setattr__(self, "seal_sha256", expected)

    @property
    def ridge(self) -> float:
        return _parse_float_hex(self.ridge_hex, "ridge")

    @property
    def step_logdet_gains(self) -> tuple[float, ...]:
        return tuple(
            _parse_float_hex(value, f"step gain {index}")
            for index, value in enumerate(self.step_logdet_gains_hex)
        )

    @property
    def sha256(self) -> str:
        return self.seal_sha256

    def body(self) -> dict[str, Any]:
        return {
            "algorithm": D_OPTIMAL_ALGORITHM,
            "budget": self.budget,
            "candidate_job_ids": list(self.candidate_job_ids),
            "code_pin": self.code_pin,
            "design_abi": self.design_abi,
            "design_encoding": D_OPTIMAL_DESIGN_ENCODING,
            "design_sha256": self.design_sha256,
            "feature_dimension": self.feature_dimension,
            "feature_pin_sha256": self.feature_pin_sha256,
            "holdout_values_used": self.holdout_values_used,
            "limits": self.limits.to_document(),
            "model_pin": self.model_pin,
            "normalization": self.normalization,
            "numeric_abi": D_OPTIMAL_NUMERIC_ABI,
            "ridge_hex": self.ridge_hex,
            "selected_job_ids": list(self.selected_job_ids),
            "step_logdet_gains_hex": list(self.step_logdet_gains_hex),
            "target_values_used": self.target_values_used,
            "tie_break": D_OPTIMAL_TIE_BREAK,
        }

    def verify_seal(self) -> None:
        expected = hashlib.sha256(_canonical_json(self.body())).hexdigest()
        if self.seal_sha256 != expected:
            raise AcquisitionIntegrityError("acquisition seal mismatch")

    def to_document(self) -> dict[str, Any]:
        self.verify_seal()
        return {
            "body": self.body(),
            "schema": D_OPTIMAL_ACQUISITION_SCHEMA,
            "sha256": self.seal_sha256,
        }

    def to_bytes(self) -> bytes:
        encoded = _canonical_json(self.to_document())
        if len(encoded) > MAX_RECEIPT_BYTES:
            raise AcquisitionIntegrityError("acquisition receipt exceeds its byte bound")
        return encoded

    @classmethod
    def from_document(cls, document: Any) -> DOptimalAcquisitionReceipt:
        if not isinstance(document, Mapping) or set(document) != {
            "body",
            "schema",
            "sha256",
        }:
            raise AcquisitionIntegrityError("acquisition envelope is malformed")
        if document.get("schema") != D_OPTIMAL_ACQUISITION_SCHEMA:
            raise AcquisitionIntegrityError("acquisition schema is unknown")
        body = document.get("body")
        expected_body = {
            "algorithm",
            "budget",
            "candidate_job_ids",
            "code_pin",
            "design_abi",
            "design_encoding",
            "design_sha256",
            "feature_dimension",
            "feature_pin_sha256",
            "holdout_values_used",
            "limits",
            "model_pin",
            "normalization",
            "numeric_abi",
            "ridge_hex",
            "selected_job_ids",
            "step_logdet_gains_hex",
            "target_values_used",
            "tie_break",
        }
        if not isinstance(body, Mapping) or set(body) != expected_body:
            raise AcquisitionIntegrityError("acquisition body is malformed")
        if (
            body.get("algorithm") != D_OPTIMAL_ALGORITHM
            or body.get("numeric_abi") != D_OPTIMAL_NUMERIC_ABI
            or body.get("design_encoding") != D_OPTIMAL_DESIGN_ENCODING
            or body.get("tie_break") != D_OPTIMAL_TIE_BREAK
        ):
            raise AcquisitionIntegrityError("acquisition algorithm ABI is unknown")
        try:
            claimed = _sha256(document.get("sha256"), "acquisition SHA-256")
            if claimed != hashlib.sha256(_canonical_json(body)).hexdigest():
                raise AcquisitionIntegrityError("acquisition SHA-256 mismatch")
            candidate_ids = body["candidate_job_ids"]
            selected_ids = body["selected_job_ids"]
            gains = body["step_logdet_gains_hex"]
            if not all(isinstance(value, list) for value in (candidate_ids, selected_ids, gains)):
                raise AcquisitionIntegrityError("acquisition arrays are malformed")
            receipt = cls(
                code_pin=body["code_pin"],
                model_pin=body["model_pin"],
                design_abi=body["design_abi"],
                feature_pin_sha256=body["feature_pin_sha256"],
                design_sha256=body["design_sha256"],
                feature_dimension=body["feature_dimension"],
                candidate_job_ids=tuple(candidate_ids),
                ridge_hex=body["ridge_hex"],
                budget=body["budget"],
                normalization=body["normalization"],
                selected_job_ids=tuple(selected_ids),
                step_logdet_gains_hex=tuple(gains),
                limits=AcquisitionLimits.from_document(body["limits"]),
                target_values_used=body["target_values_used"],
                holdout_values_used=body["holdout_values_used"],
                seal_sha256=claimed,
            )
        except AcquisitionIntegrityError:
            raise
        except AcquisitionError as exc:
            raise AcquisitionIntegrityError(
                "acquisition receipt fields are invalid"
            ) from exc
        except (KeyError, TypeError, ValueError) as exc:
            raise AcquisitionIntegrityError(
                "acquisition receipt reconstruction failed"
            ) from exc
        if receipt.to_document() != dict(document):
            raise AcquisitionIntegrityError(
                "acquisition canonical reconstruction mismatch"
            )
        return receipt

    @classmethod
    def from_bytes(cls, data: bytes) -> DOptimalAcquisitionReceipt:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if not data or len(data) > MAX_RECEIPT_BYTES:
            raise AcquisitionIntegrityError("acquisition byte length is invalid")
        try:
            document = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AcquisitionIntegrityError("acquisition receipt is invalid JSON") from exc
        if _canonical_json(document) != data:
            raise AcquisitionIntegrityError("acquisition JSON is not canonical")
        return cls.from_document(document)


def _canonical_jobs(
    jobs: Iterable[ProbeJob], limits: AcquisitionLimits
) -> tuple[ProbeJob, ...]:
    try:
        materialized = tuple(jobs)
    except TypeError as exc:
        raise AcquisitionError("probe frontier must be iterable") from exc
    if not materialized:
        raise AcquisitionError("probe frontier cannot be empty")
    if len(materialized) > limits.candidate_capacity:
        raise AcquisitionError("probe frontier exceeds candidate capacity")
    if any(not isinstance(job, ProbeJob) for job in materialized):
        raise AcquisitionError("probe frontier must contain ProbeJob instances")
    ids = [job.job_id for job in materialized]
    if len(set(ids)) != len(ids):
        raise AcquisitionError("probe frontier contains duplicate jobs")
    ordered = tuple(sorted(materialized, key=lambda job: job.job_id))
    code_pin = ordered[0].code_pin
    model_pin = ordered[0].model_pin
    if any(job.code_pin != code_pin or job.model_pin != model_pin for job in ordered):
        raise AcquisitionError("probe frontier crosses code or model pins")
    return ordered


def _feature_row(value: Any, *, job_id: str) -> tuple[float, ...]:
    if isinstance(value, Mapping):
        raise AcquisitionLeakageError(
            "design rows must be feature vectors only; target/holdout fields are forbidden"
        )
    if isinstance(value, (str, bytes, bytearray)):
        raise AcquisitionError(f"feature row for {job_id} is not numeric")
    try:
        raw_values = tuple(value)
    except TypeError as exc:
        raise AcquisitionError(f"feature row for {job_id} is not iterable") from exc
    if not raw_values:
        raise AcquisitionError("feature rows cannot be empty")
    result: list[float] = []
    for raw in raw_values:
        if isinstance(raw, bool):
            raise AcquisitionError("boolean feature values are forbidden")
        try:
            feature = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AcquisitionError(f"feature row for {job_id} is not numeric") from exc
        if not math.isfinite(feature) or abs(feature) > MAX_ABS_FEATURE:
            raise AcquisitionError("feature values exceed the numeric ABI")
        result.append(0.0 if feature == 0.0 else feature)
    return tuple(result)


def _prepare_design(
    jobs: tuple[ProbeJob, ...],
    rows_by_job: Mapping[str, Iterable[float]],
    *,
    design_abi: str,
    feature_pin_sha256: str,
    limits: AcquisitionLimits,
) -> tuple[NDArray[np.float64], str]:
    if not isinstance(rows_by_job, Mapping):
        raise AcquisitionError("design rows must be keyed by probe job id")
    if any(not isinstance(key, str) for key in rows_by_job):
        raise AcquisitionError("design row keys must be probe job ids")
    expected = {job.job_id for job in jobs}
    actual = set(rows_by_job)
    if actual != expected:
        raise AcquisitionError("design rows do not exactly cover the probe frontier")
    rows = tuple(_feature_row(rows_by_job[job.job_id], job_id=job.job_id) for job in jobs)
    dimension = len(rows[0])
    if any(len(row) != dimension for row in rows):
        raise AcquisitionError("feature rows have inconsistent dimensions")
    if dimension > limits.feature_capacity:
        raise AcquisitionError("feature dimension exceeds its capacity")
    if len(rows) * dimension > limits.design_cell_capacity:
        raise AcquisitionError("feature design exceeds its cell capacity")
    if dimension * dimension > limits.gram_cell_capacity:
        raise AcquisitionError("Gram matrix exceeds its cell capacity")
    matrix = np.asarray(rows, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(_DESIGN_DIGEST_DOMAIN)
    abi_bytes = design_abi.encode("ascii")
    digest.update(struct.pack(">I", len(abi_bytes)))
    digest.update(abi_bytes)
    digest.update(bytes.fromhex(feature_pin_sha256))
    digest.update(struct.pack(">QI", len(jobs), dimension))
    for job, row in zip(jobs, matrix, strict=True):
        digest.update(bytes.fromhex(job.job_id))
        digest.update(np.asarray(row, dtype=">f8").tobytes(order="C"))
    return matrix, digest.hexdigest()


def encode_probe_job_structural_design(
    jobs: Iterable[ProbeJob],
    *,
    limits: AcquisitionLimits | None = None,
) -> tuple[dict[str, tuple[float, ...]], str]:
    """Encode a normal ProbeJob frontier without outcomes or caller features.

    The schema contains bounded layer polynomials, one-hot frontier categories,
    and layer/unit-index interactions with target module and target unit kind.
    Sorted category universes and normalization scales are part of the returned
    feature-schema pin.
    """

    if limits is None:
        limits = AcquisitionLimits()
    if not isinstance(limits, AcquisitionLimits):
        raise AcquisitionError("limits must be AcquisitionLimits")
    ordered = _canonical_jobs(jobs, limits)
    if any(job.layer > MAX_STRUCTURAL_COORDINATE for job in ordered):
        raise AcquisitionError("probe layer exceeds the structural design ABI")
    unit_indices = tuple(
        job.target.unit_index
        for job in ordered
        if job.target.unit_index is not None
    )
    if any(index > MAX_STRUCTURAL_COORDINATE for index in unit_indices):
        raise AcquisitionError("target unit index exceeds the structural design ABI")

    families = tuple(sorted({job.probe_family for job in ordered}))
    interventions = tuple(sorted({job.intervention for job in ordered}))
    modules = tuple(sorted({job.target.module for job in ordered}))
    unit_kinds = tuple(sorted({job.target.unit_kind for job in ordered}))
    prompts = tuple(sorted({job.prompt_sha256 or "none" for job in ordered}))
    layer_scale = max(1, max(job.layer for job in ordered))
    unit_index_scale = max((1, *unit_indices))

    feature_names = (
        "intercept",
        "layer_norm",
        "layer_norm_squared",
        "unit_index_present",
        "unit_index_norm",
        "layer_x_unit_index",
        *(f"probe_family={value}" for value in families),
        *(f"intervention={value}" for value in interventions),
        *(f"target_module={value}" for value in modules),
        *(f"target_unit_kind={value}" for value in unit_kinds),
        *(f"prompt_sha256={value}" for value in prompts),
        *(f"layer_x_target_module={value}" for value in modules),
        *(f"layer_x_target_unit_kind={value}" for value in unit_kinds),
        *(f"layer_x_prompt_sha256={value}" for value in prompts),
        *(f"unit_index_x_target_module={value}" for value in modules),
        *(f"unit_index_x_target_unit_kind={value}" for value in unit_kinds),
    )
    if len(feature_names) > limits.feature_capacity:
        raise AcquisitionError("structural feature schema exceeds feature capacity")

    rows: dict[str, tuple[float, ...]] = {}
    for job in ordered:
        layer = float(job.layer / layer_scale)
        present = 1.0 if job.target.unit_index is not None else 0.0
        unit_index = (
            float(job.target.unit_index / unit_index_scale)
            if job.target.unit_index is not None
            else 0.0
        )
        family = tuple(float(job.probe_family == value) for value in families)
        intervention = tuple(
            float(job.intervention == value) for value in interventions
        )
        module = tuple(float(job.target.module == value) for value in modules)
        unit_kind = tuple(
            float(job.target.unit_kind == value) for value in unit_kinds
        )
        prompt = tuple(
            float((job.prompt_sha256 or "none") == value) for value in prompts
        )
        rows[job.job_id] = (
            1.0,
            layer,
            layer * layer,
            present,
            unit_index,
            layer * unit_index,
            *family,
            *intervention,
            *module,
            *unit_kind,
            *prompt,
            *(layer * value for value in module),
            *(layer * value for value in unit_kind),
            *(layer * value for value in prompt),
            *(unit_index * value for value in module),
            *(unit_index * value for value in unit_kind),
        )
    schema = {
        "categories": {
            "intervention": list(interventions),
            "probe_family": list(families),
            "prompt_sha256": list(prompts),
            "target_module": list(modules),
            "target_unit_kind": list(unit_kinds),
        },
        "design_abi": PROBE_JOB_STRUCTURAL_DESIGN_ABI,
        "feature_dimension": len(feature_names),
        "feature_names": list(feature_names),
        "layer_scale": layer_scale,
        "max_structural_coordinate": MAX_STRUCTURAL_COORDINATE,
        "numeric_abi": "float64/bounded-onehot-interactions/v1",
        "schema": PROBE_JOB_FEATURE_SCHEMA,
        "unit_index_scale": unit_index_scale,
    }
    feature_schema_sha256 = hashlib.sha256(_canonical_json(schema)).hexdigest()
    return rows, feature_schema_sha256


def _normalize_rows(
    design: NDArray[np.float64], normalization: Normalization
) -> NDArray[np.float64]:
    if normalization == "none":
        return design
    if normalization != "l2":
        raise AcquisitionError("normalization must be none or l2")
    normalized = design.copy()
    for index, row in enumerate(normalized):
        scale = float(np.max(np.abs(row)))
        if scale == 0.0:
            continue
        norm = scale * math.sqrt(math.fsum(float(value / scale) ** 2 for value in row))
        normalized[index] = row / norm
    return normalized


def _greedy_logdet(
    design: NDArray[np.float64],
    job_ids: tuple[str, ...],
    *,
    budget: int,
    ridge: float,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    dimension = int(design.shape[1])
    gram_inverse = np.eye(dimension, dtype=np.float64) / ridge
    available = np.ones(len(job_ids), dtype=np.bool_)
    selected: list[str] = []
    gains: list[float] = []
    for _ in range(budget):
        # NumPy/Accelerate can emit floating warnings from vectorized kernels
        # before returning a perfectly inspectable result.  Admission is based
        # on the explicit finite/PSD checks below, never on warning policy.
        with np.errstate(all="ignore"):
            projected = design @ gram_inverse
            leverage = np.einsum("ij,ij->i", projected, design, optimize=False)
        if not np.isfinite(leverage).all():
            raise AcquisitionError("D-optimal leverage left the numeric ABI")
        scale = max(1.0, float(np.max(np.abs(leverage))))
        tolerance = 128.0 * np.finfo(np.float64).eps * scale * max(1, dimension)
        if float(np.min(leverage)) < -tolerance:
            raise AcquisitionError("D-optimal Gram inverse lost positive definiteness")
        leverage = np.maximum(leverage, 0.0)
        leverage[~available] = -math.inf
        index = int(np.argmax(leverage))
        best_leverage = float(leverage[index])
        if not math.isfinite(best_leverage) or best_leverage < 0.0:
            raise AcquisitionError("D-optimal frontier has no selectable row")
        gain = math.log1p(best_leverage)
        x = design[index]
        with np.errstate(all="ignore"):
            inverse_x = gram_inverse @ x
            denominator = 1.0 + float(x @ inverse_x)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise AcquisitionError("Sherman-Morrison denominator is invalid")
        with np.errstate(all="ignore"):
            gram_inverse -= np.outer(inverse_x, inverse_x) / denominator
        gram_inverse = (gram_inverse + gram_inverse.T) * 0.5
        if not np.isfinite(gram_inverse).all():
            raise AcquisitionError("Sherman-Morrison update left the numeric ABI")
        selected.append(job_ids[index])
        gains.append(0.0 if gain == 0.0 else gain)
        available[index] = False
    return tuple(selected), tuple(gains)


def plan_d_optimal_acquisition(
    jobs: Iterable[ProbeJob],
    design_rows_by_job: Mapping[str, Iterable[float]],
    *,
    design_abi: str,
    feature_pin_sha256: str,
    budget: int,
    ridge: float = 1.0e-9,
    normalization: Normalization = "none",
    limits: AcquisitionLimits | None = None,
    target_values: object | None = None,
    holdout_values: object | None = None,
) -> DOptimalAcquisitionReceipt:
    """Select a bounded probe subset without reading outcomes or holdout labels."""

    if target_values is not None or holdout_values is not None:
        raise AcquisitionLeakageError(
            "D-optimal acquisition cannot use target or holdout values"
        )
    if limits is None:
        limits = AcquisitionLimits()
    if not isinstance(limits, AcquisitionLimits):
        raise AcquisitionError("limits must be AcquisitionLimits")
    ordered_jobs = _canonical_jobs(jobs, limits)
    design_abi = _design_abi(design_abi)
    feature_pin_sha256 = _sha256(feature_pin_sha256, "feature pin")
    budget = _positive_int(budget, "acquisition budget")
    if budget > len(ordered_jobs) or budget > limits.selection_capacity:
        raise AcquisitionError("acquisition budget exceeds its capacity")
    ridge = _bounded_float(ridge, "ridge")
    if not MIN_RIDGE <= ridge <= MAX_RIDGE:
        raise AcquisitionError("ridge lies outside the numeric ABI")
    if normalization not in {"none", "l2"}:
        raise AcquisitionError("normalization must be none or l2")
    design, design_sha256 = _prepare_design(
        ordered_jobs,
        design_rows_by_job,
        design_abi=design_abi,
        feature_pin_sha256=feature_pin_sha256,
        limits=limits,
    )
    normalized_design = _normalize_rows(design, normalization)
    job_ids = tuple(job.job_id for job in ordered_jobs)
    selected, gains = _greedy_logdet(
        normalized_design,
        job_ids,
        budget=budget,
        ridge=ridge,
    )
    return DOptimalAcquisitionReceipt(
        code_pin=ordered_jobs[0].code_pin,
        model_pin=ordered_jobs[0].model_pin,
        design_abi=design_abi,
        feature_pin_sha256=feature_pin_sha256,
        design_sha256=design_sha256,
        feature_dimension=int(design.shape[1]),
        candidate_job_ids=job_ids,
        ridge_hex=_float_hex(ridge),
        budget=budget,
        normalization=normalization,
        selected_job_ids=selected,
        step_logdet_gains_hex=tuple(_float_hex(gain) for gain in gains),
        limits=limits,
    )


def plan_probe_job_acquisition(
    jobs: Iterable[ProbeJob],
    *,
    budget: int,
    ridge: float = 1.0e-9,
    normalization: Normalization = "l2",
    limits: AcquisitionLimits | None = None,
    target_values: object | None = None,
    holdout_values: object | None = None,
) -> DOptimalAcquisitionReceipt:
    """Plan directly from the pinned, pre-outcome structure of ProbeJobs."""

    if target_values is not None or holdout_values is not None:
        raise AcquisitionLeakageError(
            "D-optimal acquisition cannot use target or holdout values"
        )
    materialized = tuple(jobs)
    if limits is None:
        limits = AcquisitionLimits()
    rows, feature_schema_sha256 = encode_probe_job_structural_design(
        materialized,
        limits=limits,
    )
    return plan_d_optimal_acquisition(
        materialized,
        rows,
        design_abi=PROBE_JOB_STRUCTURAL_DESIGN_ABI,
        feature_pin_sha256=feature_schema_sha256,
        budget=budget,
        ridge=ridge,
        normalization=normalization,
        limits=limits,
        target_values=target_values,
        holdout_values=holdout_values,
    )


def selected_probe_jobs(
    receipt: DOptimalAcquisitionReceipt,
    jobs: Iterable[ProbeJob],
) -> tuple[ProbeJob, ...]:
    """Resolve the sealed order into the exact jobs accepted by O1Cartographer."""

    if not isinstance(receipt, DOptimalAcquisitionReceipt):
        raise TypeError("receipt must be a DOptimalAcquisitionReceipt")
    receipt.verify_seal()
    ordered = _canonical_jobs(jobs, receipt.limits)
    if tuple(job.job_id for job in ordered) != receipt.candidate_job_ids:
        raise AcquisitionIntegrityError("probe frontier differs from the receipt")
    if any(
        job.code_pin != receipt.code_pin or job.model_pin != receipt.model_pin
        for job in ordered
    ):
        raise AcquisitionIntegrityError("probe frontier pins differ from the receipt")
    by_id = {job.job_id: job for job in ordered}
    return tuple(by_id[job_id] for job_id in receipt.selected_job_ids)


def replay_d_optimal_acquisition(
    receipt: DOptimalAcquisitionReceipt,
    jobs: Iterable[ProbeJob],
    design_rows_by_job: Mapping[str, Iterable[float]],
) -> tuple[ProbeJob, ...]:
    """Recompute every selection step and return the verified O1 probe subset."""

    if not isinstance(receipt, DOptimalAcquisitionReceipt):
        raise TypeError("receipt must be a DOptimalAcquisitionReceipt")
    receipt.verify_seal()
    materialized_jobs = tuple(jobs)
    try:
        replay = plan_d_optimal_acquisition(
            materialized_jobs,
            design_rows_by_job,
            design_abi=receipt.design_abi,
            feature_pin_sha256=receipt.feature_pin_sha256,
            budget=receipt.budget,
            ridge=receipt.ridge,
            normalization=receipt.normalization,
            limits=receipt.limits,
        )
    except AcquisitionIntegrityError:
        raise
    except AcquisitionError as exc:
        raise AcquisitionIntegrityError("acquisition replay input is invalid") from exc
    if replay.to_document() != receipt.to_document():
        raise AcquisitionIntegrityError("acquisition replay differs from the receipt")
    return selected_probe_jobs(receipt, materialized_jobs)


def replay_probe_job_acquisition(
    receipt: DOptimalAcquisitionReceipt,
    jobs: Iterable[ProbeJob],
) -> tuple[ProbeJob, ...]:
    """Regenerate the built-in structural design and replay its acquisition."""

    if not isinstance(receipt, DOptimalAcquisitionReceipt):
        raise TypeError("receipt must be a DOptimalAcquisitionReceipt")
    if receipt.design_abi != PROBE_JOB_STRUCTURAL_DESIGN_ABI:
        raise AcquisitionIntegrityError(
            "receipt does not use the ProbeJob structural design ABI"
        )
    materialized = tuple(jobs)
    try:
        rows, feature_schema_sha256 = encode_probe_job_structural_design(
            materialized,
            limits=receipt.limits,
        )
    except AcquisitionError as exc:
        raise AcquisitionIntegrityError(
            "ProbeJob structural design replay is invalid"
        ) from exc
    if feature_schema_sha256 != receipt.feature_pin_sha256:
        raise AcquisitionIntegrityError(
            "ProbeJob feature schema differs from the receipt"
        )
    return replay_d_optimal_acquisition(receipt, materialized, rows)


__all__ = [
    "D_OPTIMAL_ACQUISITION_SCHEMA",
    "D_OPTIMAL_ALGORITHM",
    "D_OPTIMAL_DESIGN_ENCODING",
    "D_OPTIMAL_NUMERIC_ABI",
    "D_OPTIMAL_TIE_BREAK",
    "PROBE_JOB_FEATURE_SCHEMA",
    "PROBE_JOB_STRUCTURAL_DESIGN_ABI",
    "AcquisitionError",
    "AcquisitionIntegrityError",
    "AcquisitionLeakageError",
    "AcquisitionLimits",
    "DOptimalAcquisitionReceipt",
    "encode_probe_job_structural_design",
    "plan_d_optimal_acquisition",
    "plan_probe_job_acquisition",
    "replay_d_optimal_acquisition",
    "replay_probe_job_acquisition",
    "selected_probe_jobs",
]
