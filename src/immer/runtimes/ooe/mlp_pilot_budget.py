"""Receipt-bound brake-only budgets for sparse Qwen MLP layer actions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence, cast

from .identity import canonical_json_bytes, require_sha256


MLP_PILOT_DECODE_REPORT_SCHEMA = (
    "immer.qwen3.8-mlp-pilot-sparse-decode-compare/v2"
)
MLP_PILOT_DECODE_REPORT_SCHEMA_V3 = (
    "immer.qwen3.8-mlp-pilot-sparse-decode-compare/v3"
)
MLP_PILOT_LAYER_BUDGET_CONFIG_SCHEMA = (
    "immer.qwen3.8-mlp-pilot-layer-budget-config/v1"
)
MLP_PILOT_LAYER_CANDIDATE_SCHEMA = (
    "immer.qwen3.8-mlp-pilot-layer-budget-candidate/v2"
)
MLP_PILOT_LAYER_POLICY_SCHEMA = "immer.qwen3.8-mlp-pilot-layer-budget-policy/v2"
MLP_PILOT_LAYER_POLICY_VERIFIER_SHA256 = hashlib.sha256(
    b"immer:qwen-mlp-pilot:receipt-domain+token-path+prefix-horizon+brake-only/v2"
).hexdigest()
_MAX_REPORT_BYTES = 8 * 1024 * 1024
_MAX_POLICY_BYTES = 8 * 1024 * 1024


class MlpPilotLayerBudgetError(RuntimeError):
    pass


class MlpPilotLayerBudgetIntegrityError(MlpPilotLayerBudgetError):
    pass


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json(data: bytes, *, maximum: int, label: str) -> Mapping[str, object]:
    if not isinstance(data, bytes) or not data or len(data) > maximum:
        raise MlpPilotLayerBudgetIntegrityError(f"{label} exceeds its byte bound")
    canonical = data[:-1] if data.endswith(b"\n") else data
    if not canonical or canonical.endswith(b"\n"):
        raise MlpPilotLayerBudgetIntegrityError(f"{label} has invalid framing")

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            canonical,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MlpPilotLayerBudgetIntegrityError(f"{label} is not strict JSON") from exc
    if not isinstance(value, Mapping) or canonical_json_bytes(value) != canonical:
        raise MlpPilotLayerBudgetIntegrityError(f"{label} is not canonical JSON")
    return cast(Mapping[str, object], value)


def _uint(value: object, *, field: str, positive: bool = False) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < (1 if positive else 0)
    ):
        raise MlpPilotLayerBudgetIntegrityError(f"{field} is invalid")
    return value


def _finite(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MlpPilotLayerBudgetIntegrityError(f"{field} is invalid")
    result = float(value)
    if minimum is not None and result < minimum:
        raise MlpPilotLayerBudgetIntegrityError(f"{field} is below its bound")
    if maximum is not None and result > maximum:
        raise MlpPilotLayerBudgetIntegrityError(f"{field} exceeds its bound")
    return result


def _layers(value: object, *, field: str, allow_empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise MlpPilotLayerBudgetIntegrityError(f"{field} is invalid")
    rows = tuple(_uint(row, field=field) for row in value)
    if (
        (not rows and not allow_empty)
        or rows != tuple(sorted(set(rows)))
    ):
        raise MlpPilotLayerBudgetIntegrityError(f"{field} is not a sorted set")
    return rows


def _token_ids(value: object, *, field: str, count: int | None = None) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise MlpPilotLayerBudgetIntegrityError(f"{field} is invalid")
    rows = tuple(_uint(row, field=field) for row in value)
    if (count is not None and len(rows) != count) or not rows:
        raise MlpPilotLayerBudgetIntegrityError(f"{field} has the wrong size")
    return rows


def _close(actual: object, expected: float, *, field: str) -> None:
    value = _finite(actual, field=field)
    if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise MlpPilotLayerBudgetIntegrityError(f"{field} differs from step receipts")


@dataclass(frozen=True, slots=True)
class MlpPilotDecodeStep:
    input_token_id: int
    top1_equal: bool
    top10_overlap: int
    hidden_cosine: float
    hidden_relative_l2: float
    candidate_logit_max_abs_error: float
    full_primary_weight_bytes: int
    sparse_weight_bytes: int
    full_seconds: float
    sparse_seconds: float


@dataclass(frozen=True, slots=True)
class MlpPilotDecodeEvidence:
    report_schema: str
    report_sha256: str
    report_body_sha256: str
    model_pin_sha256: str
    router_fit_sha256: str
    affine_fit_sha256: str
    prefix_sha256: str
    sparse_layers: tuple[int, ...]
    steps: int
    teacher_forced_input_token_ids: tuple[int, ...]
    top1_equal_steps: int
    top10_overlap_min: int
    top10_overlap_mean: float
    hidden_cosine_min: float
    hidden_relative_l2_max: float
    candidate_logit_max_abs_error: float
    full_primary_weight_bytes: int
    sparse_weight_bytes: int
    full_decode_seconds: float
    sparse_decode_seconds: float
    layer_budget_policy_sha256: str | None
    step_evidence: tuple[MlpPilotDecodeStep, ...]

    @property
    def all_top1_equal(self) -> bool:
        return self.top1_equal_steps == self.steps

    @property
    def byte_saving_fraction(self) -> float:
        return 1.0 - self.sparse_weight_bytes / self.full_primary_weight_bytes

    @property
    def speedup(self) -> float:
        return self.full_decode_seconds / self.sparse_decode_seconds


def parse_mlp_pilot_decode_report(data: bytes) -> MlpPilotDecodeEvidence:
    envelope = _strict_json(data, maximum=_MAX_REPORT_BYTES, label="decode report")
    if (
        set(envelope) != {"body", "body_sha256", "schema"}
        or envelope.get("schema")
        not in {MLP_PILOT_DECODE_REPORT_SCHEMA, MLP_PILOT_DECODE_REPORT_SCHEMA_V3}
        or not isinstance(envelope.get("body"), Mapping)
        or envelope.get("body_sha256") != _digest(envelope.get("body"))
    ):
        raise MlpPilotLayerBudgetIntegrityError("decode report seal is invalid")
    body = cast(Mapping[str, object], envelope["body"])
    expected = {
        "affine_fit_sha256",
        "candidate_logit_max_abs_error",
        "full_decode_seconds",
        "full_primary_weight_bytes",
        "hidden_cosine_mean",
        "hidden_cosine_min",
        "hidden_relative_l2_max",
        "hidden_relative_l2_mean",
        "model_pin_sha256",
        "prefix_seconds",
        "prefix_sha256",
        "prefix_top1_value",
        "prefix_tokens",
        "router_fit_sha256",
        "sparse_layers",
        "sparse_decode_seconds",
        "sparse_pilot_weight_bytes",
        "sparse_primary_weight_bytes",
        "sparse_top10_overlap_mean",
        "sparse_transpose_weight_bytes",
        "speedup_full_over_sparse",
        "step_receipts",
        "steps",
        "teacher_forced_input_token_ids",
        "top1_equal_steps",
        "unused_full_auxiliary_bytes",
    }
    report_schema = cast(str, envelope["schema"])
    if report_schema == MLP_PILOT_DECODE_REPORT_SCHEMA_V3:
        expected |= {"layer_budget_policy_sha256"}
    if set(body) != expected or not isinstance(body.get("step_receipts"), list):
        raise MlpPilotLayerBudgetIntegrityError("decode report body is invalid")
    steps = _uint(body["steps"], field="steps", positive=True)
    rows = cast(list[object], body["step_receipts"])
    if len(rows) != steps:
        raise MlpPilotLayerBudgetIntegrityError("step receipt count is invalid")
    full_bytes = sparse_primary = sparse_pilot = sparse_transpose = 0
    full_seconds = sparse_seconds = 0.0
    top1_equal = 0
    overlaps: list[int] = []
    cosines: list[float] = []
    relative_l2: list[float] = []
    logit_errors: list[float] = []
    inputs: list[int] = []
    step_evidence: list[MlpPilotDecodeStep] = []
    step_expected = {
        "candidate_logit_max_abs_error",
        "candidate_token_ids",
        "full_primary_weight_bytes",
        "full_seconds",
        "full_top10",
        "full_top10_values",
        "hidden_metrics",
        "input_token_id",
        "sparse_pilot_weight_bytes",
        "sparse_primary_weight_bytes",
        "sparse_seconds",
        "sparse_top10",
        "sparse_top10_overlap",
        "sparse_top10_values",
        "sparse_transpose_weight_bytes",
        "step",
        "top1_equal",
    }
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or set(raw) != step_expected:
            raise MlpPilotLayerBudgetIntegrityError("step receipt is invalid")
        row = cast(Mapping[str, object], raw)
        if _uint(row["step"], field="step") != index:
            raise MlpPilotLayerBudgetIntegrityError("step receipts are not contiguous")
        full_top = _token_ids(row["full_top10"], field="full_top10", count=10)
        sparse_top = _token_ids(row["sparse_top10"], field="sparse_top10", count=10)
        if len(set(full_top)) != 10 or len(set(sparse_top)) != 10:
            raise MlpPilotLayerBudgetIntegrityError("top-10 token ids are not unique")
        candidate_ids = _token_ids(row["candidate_token_ids"], field="candidate ids")
        if candidate_ids != tuple(sorted(set(full_top) | set(sparse_top))):
            raise MlpPilotLayerBudgetIntegrityError("candidate token union is invalid")
        overlap = len(set(full_top) & set(sparse_top))
        if _uint(row["sparse_top10_overlap"], field="top10 overlap") != overlap:
            raise MlpPilotLayerBudgetIntegrityError("top-10 overlap is invalid")
        equal = full_top[0] == sparse_top[0]
        if not isinstance(row["top1_equal"], bool) or row["top1_equal"] != equal:
            raise MlpPilotLayerBudgetIntegrityError("top-1 equality is invalid")
        hidden = row["hidden_metrics"]
        if (
            not isinstance(hidden, Mapping)
            or set(hidden) != {"cosine", "max_abs_error", "relative_l2_error"}
        ):
            raise MlpPilotLayerBudgetIntegrityError("hidden metrics are invalid")
        hidden = cast(Mapping[str, object], hidden)
        cosines.append(
            _finite(hidden["cosine"], field="hidden cosine", minimum=-1.0, maximum=1.0)
        )
        _finite(hidden["max_abs_error"], field="hidden max error", minimum=0.0)
        relative_l2.append(
            _finite(hidden["relative_l2_error"], field="hidden relative L2", minimum=0.0)
        )
        logit_errors.append(
            _finite(
                row["candidate_logit_max_abs_error"],
                field="candidate logit error",
                minimum=0.0,
            )
        )
        full_row_bytes = _uint(
            row["full_primary_weight_bytes"], field="full weight bytes", positive=True
        )
        sparse_row_primary = _uint(
            row["sparse_primary_weight_bytes"],
            field="sparse primary weight bytes",
        )
        sparse_row_pilot = _uint(
            row["sparse_pilot_weight_bytes"], field="sparse pilot weight bytes"
        )
        sparse_row_transpose = _uint(
            row["sparse_transpose_weight_bytes"],
            field="sparse transpose weight bytes",
        )
        full_row_seconds = _finite(
            row["full_seconds"], field="full seconds", minimum=0.0
        )
        sparse_row_seconds = _finite(
            row["sparse_seconds"], field="sparse seconds", minimum=0.0
        )
        full_bytes += full_row_bytes
        sparse_primary += sparse_row_primary
        sparse_pilot += sparse_row_pilot
        sparse_transpose += sparse_row_transpose
        full_seconds += full_row_seconds
        sparse_seconds += sparse_row_seconds
        overlaps.append(overlap)
        top1_equal += int(equal)
        input_token_id = _uint(row["input_token_id"], field="input token")
        inputs.append(input_token_id)
        step_evidence.append(
            MlpPilotDecodeStep(
                input_token_id=input_token_id,
                top1_equal=equal,
                top10_overlap=overlap,
                hidden_cosine=cosines[-1],
                hidden_relative_l2=relative_l2[-1],
                candidate_logit_max_abs_error=logit_errors[-1],
                full_primary_weight_bytes=full_row_bytes,
                sparse_weight_bytes=(
                    sparse_row_primary + sparse_row_pilot + sparse_row_transpose
                ),
                full_seconds=full_row_seconds,
                sparse_seconds=sparse_row_seconds,
            )
        )
    if full_seconds <= 0.0 or sparse_seconds <= 0.0:
        raise MlpPilotLayerBudgetIntegrityError("decode seconds must be positive")
    sparse_bytes = sparse_primary + sparse_pilot + sparse_transpose
    if sparse_bytes <= 0:
        raise MlpPilotLayerBudgetIntegrityError("sparse report read no weight bytes")
    aggregates = {
        "candidate_logit_max_abs_error": max(logit_errors),
        "full_decode_seconds": full_seconds,
        "hidden_cosine_mean": sum(cosines) / steps,
        "hidden_cosine_min": min(cosines),
        "hidden_relative_l2_max": max(relative_l2),
        "hidden_relative_l2_mean": sum(relative_l2) / steps,
        "sparse_decode_seconds": sparse_seconds,
        "sparse_top10_overlap_mean": sum(overlaps) / steps,
        "speedup_full_over_sparse": full_seconds / sparse_seconds,
    }
    for field, expected_value in aggregates.items():
        _close(body[field], expected_value, field=field)
    integer_aggregates = {
        "full_primary_weight_bytes": full_bytes,
        "sparse_primary_weight_bytes": sparse_primary,
        "sparse_pilot_weight_bytes": sparse_pilot,
        "sparse_transpose_weight_bytes": sparse_transpose,
        "top1_equal_steps": top1_equal,
    }
    for field, expected_value in integer_aggregates.items():
        if _uint(body[field], field=field) != expected_value:
            raise MlpPilotLayerBudgetIntegrityError(
                f"{field} differs from step receipts"
            )
    teacher = _token_ids(
        body["teacher_forced_input_token_ids"],
        field="teacher-forced tokens",
        count=steps,
    )
    if teacher != tuple(inputs):
        raise MlpPilotLayerBudgetIntegrityError(
            "teacher-forced tokens differ from step receipts"
        )
    for field in ("model_pin_sha256", "router_fit_sha256", "affine_fit_sha256"):
        require_sha256(body[field], field=field)
    policy_sha256 = body.get("layer_budget_policy_sha256")
    if policy_sha256 is not None:
        policy_sha256 = require_sha256(
            policy_sha256, field="layer_budget_policy_sha256"
        )
    return MlpPilotDecodeEvidence(
        report_schema=report_schema,
        report_sha256=hashlib.sha256(canonical_json_bytes(envelope)).hexdigest(),
        report_body_sha256=cast(str, envelope["body_sha256"]),
        model_pin_sha256=cast(str, body["model_pin_sha256"]),
        router_fit_sha256=cast(str, body["router_fit_sha256"]),
        affine_fit_sha256=cast(str, body["affine_fit_sha256"]),
        prefix_sha256=require_sha256(body["prefix_sha256"], field="prefix_sha256"),
        sparse_layers=_layers(body["sparse_layers"], field="sparse_layers"),
        steps=steps,
        teacher_forced_input_token_ids=teacher,
        top1_equal_steps=top1_equal,
        top10_overlap_min=min(overlaps),
        top10_overlap_mean=sum(overlaps) / steps,
        hidden_cosine_min=min(cosines),
        hidden_relative_l2_max=max(relative_l2),
        candidate_logit_max_abs_error=max(logit_errors),
        full_primary_weight_bytes=full_bytes,
        sparse_weight_bytes=sparse_bytes,
        full_decode_seconds=full_seconds,
        sparse_decode_seconds=sparse_seconds,
        layer_budget_policy_sha256=cast(str | None, policy_sha256),
        step_evidence=tuple(step_evidence),
    )


@dataclass(frozen=True, slots=True)
class MlpPilotLayerBudgetConfig:
    min_reports_per_candidate: int = 1
    min_distinct_prefixes: int = 1
    min_verified_steps: int = 4
    min_top10_overlap: int = 8
    min_hidden_cosine: float = 0.90
    max_hidden_relative_l2: float = 0.45
    max_candidate_logit_error: float = 64.0
    min_byte_saving_fraction: float = 0.01
    min_speedup: float = 0.0
    max_sparse_layers: int = 64

    def __post_init__(self) -> None:
        for field in (
            "min_reports_per_candidate",
            "min_distinct_prefixes",
            "min_verified_steps",
            "max_sparse_layers",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if (
            isinstance(self.min_top10_overlap, bool)
            or not isinstance(self.min_top10_overlap, int)
            or not 0 <= self.min_top10_overlap <= 10
        ):
            raise ValueError("min_top10_overlap must be in [0, 10]")
        bounds = (
            ("min_hidden_cosine", -1.0, 1.0),
            ("max_hidden_relative_l2", 0.0, None),
            ("max_candidate_logit_error", 0.0, None),
            ("min_byte_saving_fraction", -1.0, 1.0),
            ("min_speedup", 0.0, None),
        )
        for field, minimum, maximum in bounds:
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < minimum
                or (maximum is not None and float(value) > maximum)
            ):
                raise ValueError(f"{field} is outside its bound")

    def to_record(self) -> dict[str, object]:
        return {
            "max_candidate_logit_error": float(self.max_candidate_logit_error),
            "max_hidden_relative_l2": float(self.max_hidden_relative_l2),
            "max_sparse_layers": self.max_sparse_layers,
            "min_byte_saving_fraction": float(self.min_byte_saving_fraction),
            "min_distinct_prefixes": self.min_distinct_prefixes,
            "min_hidden_cosine": float(self.min_hidden_cosine),
            "min_reports_per_candidate": self.min_reports_per_candidate,
            "min_speedup": float(self.min_speedup),
            "min_top10_overlap": self.min_top10_overlap,
            "min_verified_steps": self.min_verified_steps,
            "schema": MLP_PILOT_LAYER_BUDGET_CONFIG_SCHEMA,
        }

    @classmethod
    def from_record(cls, value: object) -> "MlpPilotLayerBudgetConfig":
        expected = {
            "max_candidate_logit_error",
            "max_hidden_relative_l2",
            "max_sparse_layers",
            "min_byte_saving_fraction",
            "min_distinct_prefixes",
            "min_hidden_cosine",
            "min_reports_per_candidate",
            "min_speedup",
            "min_top10_overlap",
            "min_verified_steps",
            "schema",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != MLP_PILOT_LAYER_BUDGET_CONFIG_SCHEMA
        ):
            raise MlpPilotLayerBudgetIntegrityError("layer budget config is invalid")
        try:
            return cls(
                min_reports_per_candidate=cast(int, value["min_reports_per_candidate"]),
                min_distinct_prefixes=cast(int, value["min_distinct_prefixes"]),
                min_verified_steps=cast(int, value["min_verified_steps"]),
                min_top10_overlap=cast(int, value["min_top10_overlap"]),
                min_hidden_cosine=cast(float, value["min_hidden_cosine"]),
                max_hidden_relative_l2=cast(
                    float, value["max_hidden_relative_l2"]
                ),
                max_candidate_logit_error=cast(
                    float, value["max_candidate_logit_error"]
                ),
                min_byte_saving_fraction=cast(
                    float, value["min_byte_saving_fraction"]
                ),
                min_speedup=cast(float, value["min_speedup"]),
                max_sparse_layers=cast(int, value["max_sparse_layers"]),
            )
        except (TypeError, ValueError) as exc:
            raise MlpPilotLayerBudgetIntegrityError(
                "layer budget config validation failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class MlpPilotLayerCandidate:
    layers: tuple[int, ...]
    prefix_sha256s: tuple[str, ...]
    report_sha256s: tuple[str, ...]
    report_body_sha256s: tuple[str, ...]
    token_paths_sha256: str
    verified_steps: int
    top1_equal: bool
    top10_overlap_min: int
    hidden_cosine_min: float
    hidden_relative_l2_max: float
    candidate_logit_max_abs_error: float
    byte_saving_fraction: float
    speedup: float
    admissible: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", _layers(self.layers, field="candidate layers"))
        prefixes = tuple(
            sorted(require_sha256(row, field="prefix_sha256") for row in self.prefix_sha256s)
        )
        reports = tuple(
            sorted(require_sha256(row, field="report_sha256") for row in self.report_sha256s)
        )
        bodies = tuple(
            sorted(
                require_sha256(row, field="report_body_sha256")
                for row in self.report_body_sha256s
            )
        )
        object.__setattr__(
            self,
            "token_paths_sha256",
            require_sha256(self.token_paths_sha256, field="token_paths_sha256"),
        )
        if (
            not prefixes
            or len(set(prefixes)) != len(prefixes)
            or len(reports) != len(prefixes)
            or len(bodies) != len(prefixes)
            or len(set(reports)) != len(reports)
            or len(set(bodies)) != len(bodies)
        ):
            raise ValueError("candidate evidence inventory is invalid")
        if self.verified_steps < 1 or not 0 <= self.top10_overlap_min <= 10:
            raise ValueError("candidate horizon metrics are invalid")
        for field in (
            "hidden_cosine_min",
            "hidden_relative_l2_max",
            "candidate_logit_max_abs_error",
            "byte_saving_fraction",
            "speedup",
        ):
            if not math.isfinite(float(getattr(self, field))):
                raise ValueError(f"candidate {field} is not finite")
        reasons = tuple(self.rejection_reasons)
        if (
            any(not isinstance(row, str) or not row for row in reasons)
            or len(set(reasons)) != len(reasons)
            or self.admissible == bool(reasons)
        ):
            raise ValueError("candidate admission reasons are inconsistent")
        object.__setattr__(self, "prefix_sha256s", prefixes)
        object.__setattr__(self, "report_sha256s", reports)
        object.__setattr__(self, "report_body_sha256s", bodies)
        object.__setattr__(self, "rejection_reasons", reasons)

    def to_record(self) -> dict[str, object]:
        return {
            "admissible": self.admissible,
            "byte_saving_fraction": self.byte_saving_fraction,
            "candidate_logit_max_abs_error": self.candidate_logit_max_abs_error,
            "hidden_cosine_min": self.hidden_cosine_min,
            "hidden_relative_l2_max": self.hidden_relative_l2_max,
            "layers": list(self.layers),
            "prefix_sha256s": list(self.prefix_sha256s),
            "rejection_reasons": list(self.rejection_reasons),
            "report_body_sha256s": list(self.report_body_sha256s),
            "report_sha256s": list(self.report_sha256s),
            "schema": MLP_PILOT_LAYER_CANDIDATE_SCHEMA,
            "speedup": self.speedup,
            "token_paths_sha256": self.token_paths_sha256,
            "top10_overlap_min": self.top10_overlap_min,
            "top1_equal": self.top1_equal,
            "verified_steps": self.verified_steps,
        }

    @classmethod
    def from_record(cls, value: object) -> "MlpPilotLayerCandidate":
        expected = {
            "admissible",
            "byte_saving_fraction",
            "candidate_logit_max_abs_error",
            "hidden_cosine_min",
            "hidden_relative_l2_max",
            "layers",
            "prefix_sha256s",
            "rejection_reasons",
            "report_body_sha256s",
            "report_sha256s",
            "schema",
            "speedup",
            "token_paths_sha256",
            "top10_overlap_min",
            "top1_equal",
            "verified_steps",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("schema") != MLP_PILOT_LAYER_CANDIDATE_SCHEMA
            or not isinstance(value.get("layers"), list)
            or not isinstance(value.get("prefix_sha256s"), list)
            or not isinstance(value.get("report_sha256s"), list)
            or not isinstance(value.get("report_body_sha256s"), list)
            or not isinstance(value.get("rejection_reasons"), list)
            or not isinstance(value.get("admissible"), bool)
            or not isinstance(value.get("top1_equal"), bool)
        ):
            raise MlpPilotLayerBudgetIntegrityError("layer candidate is invalid")
        try:
            return cls(
                layers=tuple(cast(list[int], value["layers"])),
                prefix_sha256s=tuple(cast(list[str], value["prefix_sha256s"])),
                report_sha256s=tuple(cast(list[str], value["report_sha256s"])),
                report_body_sha256s=tuple(
                    cast(list[str], value["report_body_sha256s"])
                ),
                verified_steps=cast(int, value["verified_steps"]),
                top1_equal=cast(bool, value["top1_equal"]),
                top10_overlap_min=cast(int, value["top10_overlap_min"]),
                hidden_cosine_min=cast(float, value["hidden_cosine_min"]),
                hidden_relative_l2_max=cast(
                    float, value["hidden_relative_l2_max"]
                ),
                candidate_logit_max_abs_error=cast(
                    float, value["candidate_logit_max_abs_error"]
                ),
                byte_saving_fraction=cast(float, value["byte_saving_fraction"]),
                speedup=cast(float, value["speedup"]),
                token_paths_sha256=cast(str, value["token_paths_sha256"]),
                admissible=cast(bool, value["admissible"]),
                rejection_reasons=tuple(
                    cast(list[str], value["rejection_reasons"])
                ),
            )
        except (TypeError, ValueError) as exc:
            raise MlpPilotLayerBudgetIntegrityError(
                "layer candidate validation failed"
            ) from exc


@dataclass(frozen=True, slots=True)
class MlpPilotMeasuredPath:
    prefix_sha256: str
    input_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prefix_sha256",
            require_sha256(self.prefix_sha256, field="prefix_sha256"),
        )
        object.__setattr__(
            self,
            "input_token_ids",
            _token_ids(self.input_token_ids, field="measured input tokens"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "input_token_ids": list(self.input_token_ids),
            "prefix_sha256": self.prefix_sha256,
        }

    @classmethod
    def from_record(cls, value: object) -> "MlpPilotMeasuredPath":
        if (
            not isinstance(value, Mapping)
            or set(value) != {"input_token_ids", "prefix_sha256"}
            or not isinstance(value.get("input_token_ids"), list)
        ):
            raise MlpPilotLayerBudgetIntegrityError("measured token path is invalid")
        return cls(
            prefix_sha256=cast(str, value["prefix_sha256"]),
            input_token_ids=tuple(cast(list[int], value["input_token_ids"])),
        )


def _token_paths_sha256(paths: Sequence[MlpPilotMeasuredPath]) -> str:
    return _digest(
        {
            "paths": [row.to_record() for row in paths],
            "schema": "immer.qwen3.8-mlp-pilot-measured-token-paths/v1",
        }
    )


@dataclass(frozen=True, slots=True)
class MlpPilotBudgetState:
    decode_step: int = 0
    braked: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.decode_step, bool)
            or not isinstance(self.decode_step, int)
            or self.decode_step < 0
            or not isinstance(self.braked, bool)
        ):
            raise ValueError("layer budget state is invalid")


@dataclass(frozen=True, slots=True)
class MlpPilotBudgetDecision:
    action: str
    layers: tuple[int, ...]
    reason: str
    state: MlpPilotBudgetState
    policy_sha256: str


@dataclass(frozen=True, slots=True)
class MlpPilotLayerBudgetPolicy:
    model_pin_sha256: str
    router_fit_sha256: str
    affine_fit_sha256: str
    config: MlpPilotLayerBudgetConfig
    candidates: tuple[MlpPilotLayerCandidate, ...]
    selected_layers: tuple[int, ...]
    scope_prefix_sha256s: tuple[str, ...]
    measured_paths: tuple[MlpPilotMeasuredPath, ...]
    verified_steps: int

    def __post_init__(self) -> None:
        for field in ("model_pin_sha256", "router_fit_sha256", "affine_fit_sha256"):
            object.__setattr__(
                self, field, require_sha256(getattr(self, field), field=field)
            )
        if not isinstance(self.config, MlpPilotLayerBudgetConfig):
            raise TypeError("config must be MlpPilotLayerBudgetConfig")
        candidates = tuple(self.candidates)
        if (
            not candidates
            or any(not isinstance(row, MlpPilotLayerCandidate) for row in candidates)
            or tuple(row.layers for row in candidates)
            != tuple(sorted(row.layers for row in candidates))
            or len({row.layers for row in candidates}) != len(candidates)
        ):
            raise ValueError("layer policy candidate inventory is invalid")
        selected = _layers(
            self.selected_layers, field="selected_layers", allow_empty=True
        )
        prefixes = tuple(
            sorted(
                require_sha256(row, field="scope prefix")
                for row in self.scope_prefix_sha256s
            )
        )
        if not prefixes or len(set(prefixes)) != len(prefixes):
            raise ValueError("layer policy prefix scope is invalid")
        chosen = next((row for row in candidates if row.layers == selected), None)
        if selected and (chosen is None or not chosen.admissible):
            raise ValueError("selected layer action is not admissible")
        if not selected and any(row.admissible for row in candidates):
            raise ValueError("layer policy ignored an admissible candidate")
        expected_steps = 0 if chosen is None else chosen.verified_steps
        if self.verified_steps != expected_steps:
            raise ValueError("layer policy horizon differs from selected evidence")
        if any(row.prefix_sha256s != prefixes for row in candidates):
            raise ValueError("layer policy candidates have unequal prompt coverage")
        paths = tuple(self.measured_paths)
        if selected:
            if (
                any(not isinstance(row, MlpPilotMeasuredPath) for row in paths)
                or tuple(row.prefix_sha256 for row in paths) != prefixes
                or any(len(row.input_token_ids) != self.verified_steps for row in paths)
                or chosen is None
                or _token_paths_sha256(paths) != chosen.token_paths_sha256
            ):
                raise ValueError("layer policy measured token paths are invalid")
        elif paths:
            raise ValueError("fallback-only layer policy cannot carry measured paths")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "selected_layers", selected)
        object.__setattr__(self, "scope_prefix_sha256s", prefixes)
        object.__setattr__(self, "measured_paths", paths)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    def to_document(self) -> dict[str, object]:
        body = {
            "affine_fit_sha256": self.affine_fit_sha256,
            "candidates": [row.to_record() for row in self.candidates],
            "config": self.config.to_record(),
            "model_pin_sha256": self.model_pin_sha256,
            "measured_paths": [row.to_record() for row in self.measured_paths],
            "router_fit_sha256": self.router_fit_sha256,
            "scope_prefix_sha256s": list(self.scope_prefix_sha256s),
            "selected_layers": list(self.selected_layers),
            "verified_steps": self.verified_steps,
            "verifier_sha256": MLP_PILOT_LAYER_POLICY_VERIFIER_SHA256,
        }
        return {
            "body": body,
            "body_sha256": _digest(body),
            "schema": MLP_PILOT_LAYER_POLICY_SCHEMA,
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > _MAX_POLICY_BYTES:
            raise MlpPilotLayerBudgetIntegrityError("layer policy exceeds its byte bound")
        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> "MlpPilotLayerBudgetPolicy":
        envelope = _strict_json(data, maximum=_MAX_POLICY_BYTES, label="layer policy")
        if (
            set(envelope) != {"body", "body_sha256", "schema"}
            or envelope.get("schema") != MLP_PILOT_LAYER_POLICY_SCHEMA
            or not isinstance(envelope.get("body"), Mapping)
            or envelope.get("body_sha256") != _digest(envelope.get("body"))
        ):
            raise MlpPilotLayerBudgetIntegrityError("layer policy seal is invalid")
        body = cast(Mapping[str, object], envelope["body"])
        expected = {
            "affine_fit_sha256",
            "candidates",
            "config",
            "model_pin_sha256",
            "measured_paths",
            "router_fit_sha256",
            "scope_prefix_sha256s",
            "selected_layers",
            "verified_steps",
            "verifier_sha256",
        }
        if (
            set(body) != expected
            or body.get("verifier_sha256") != MLP_PILOT_LAYER_POLICY_VERIFIER_SHA256
            or not isinstance(body.get("candidates"), list)
            or not isinstance(body.get("measured_paths"), list)
            or not isinstance(body.get("scope_prefix_sha256s"), list)
            or not isinstance(body.get("selected_layers"), list)
        ):
            raise MlpPilotLayerBudgetIntegrityError("layer policy body is invalid")
        try:
            result = cls(
                model_pin_sha256=cast(str, body["model_pin_sha256"]),
                router_fit_sha256=cast(str, body["router_fit_sha256"]),
                affine_fit_sha256=cast(str, body["affine_fit_sha256"]),
                config=MlpPilotLayerBudgetConfig.from_record(body["config"]),
                candidates=tuple(
                    MlpPilotLayerCandidate.from_record(row)
                    for row in cast(list[object], body["candidates"])
                ),
                measured_paths=tuple(
                    MlpPilotMeasuredPath.from_record(row)
                    for row in cast(list[object], body["measured_paths"])
                ),
                selected_layers=tuple(cast(list[int], body["selected_layers"])),
                scope_prefix_sha256s=tuple(
                    cast(list[str], body["scope_prefix_sha256s"])
                ),
                verified_steps=cast(int, body["verified_steps"]),
            )
        except (TypeError, ValueError) as exc:
            raise MlpPilotLayerBudgetIntegrityError(
                "layer policy validation failed"
            ) from exc
        if result.to_bytes() != data:
            raise MlpPilotLayerBudgetIntegrityError(
                "layer policy reconstruction changed"
            )
        return result

    def measured_input_token_ids(self, prefix_sha256: str) -> tuple[int, ...]:
        prefix = require_sha256(prefix_sha256, field="prefix_sha256")
        return next(
            (
                row.input_token_ids
                for row in self.measured_paths
                if row.prefix_sha256 == prefix
            ),
            (),
        )

    def authorize_teacher_forced(
        self, prefix_sha256: str, input_token_ids: Sequence[int]
    ) -> tuple[int, ...]:
        prefix = require_sha256(prefix_sha256, field="prefix_sha256")
        tokens = _token_ids(input_token_ids, field="teacher-forced input tokens")
        expected = self.measured_input_token_ids(prefix)
        if (
            not self.selected_layers
            or not expected
            or len(tokens) > self.verified_steps
            or tokens != expected[: len(tokens)]
        ):
            return ()
        return self.selected_layers

    def decide(
        self,
        prefix_sha256: str,
        state: MlpPilotBudgetState = MlpPilotBudgetState(),
        *,
        input_token_id: int,
        verifier_ok: bool = True,
    ) -> MlpPilotBudgetDecision:
        prefix = require_sha256(prefix_sha256, field="prefix_sha256")
        if not isinstance(state, MlpPilotBudgetState):
            raise TypeError("state must be MlpPilotBudgetState")
        if not isinstance(verifier_ok, bool):
            raise TypeError("verifier_ok must be bool")
        token = _uint(input_token_id, field="input_token_id")
        expected_path = self.measured_input_token_ids(prefix)
        reason = "verified-domain"
        if state.braked:
            reason = "brake-latched"
        elif not verifier_ok:
            reason = "verifier-brake"
        elif not self.selected_layers:
            reason = "no-admissible-sparse-action"
        elif prefix not in self.scope_prefix_sha256s:
            reason = "unmeasured-prefix"
        elif state.decode_step >= self.verified_steps:
            reason = "verified-horizon-exhausted"
        elif not expected_path or token != expected_path[state.decode_step]:
            reason = "token-path-divergence"
        if reason != "verified-domain":
            return MlpPilotBudgetDecision(
                action="qwen_fallback",
                layers=(),
                reason=reason,
                state=MlpPilotBudgetState(state.decode_step, True),
                policy_sha256=self.sha256,
            )
        return MlpPilotBudgetDecision(
            action="sparse_mlp",
            layers=self.selected_layers,
            reason=reason,
            state=MlpPilotBudgetState(state.decode_step + 1, False),
            policy_sha256=self.sha256,
        )


def _candidate(
    layers: tuple[int, ...],
    evidence: Sequence[MlpPilotDecodeEvidence],
    config: MlpPilotLayerBudgetConfig,
) -> MlpPilotLayerCandidate:
    def safe(step: MlpPilotDecodeStep) -> bool:
        return (
            step.top1_equal
            and step.top10_overlap >= config.min_top10_overlap
            and step.hidden_cosine >= config.min_hidden_cosine
            and step.hidden_relative_l2 <= config.max_hidden_relative_l2
            and step.candidate_logit_max_abs_error
            <= config.max_candidate_logit_error
        )

    horizons = []
    for row in evidence:
        horizon = 0
        for step in row.step_evidence:
            if not safe(step):
                break
            horizon += 1
        horizons.append(horizon)
    verified_steps = min(horizons)
    metric_steps = max(1, verified_steps)
    selected_steps = tuple(
        step for row in evidence for step in row.step_evidence[:metric_steps]
    )
    total_full_bytes = sum(row.full_primary_weight_bytes for row in selected_steps)
    total_sparse_bytes = sum(row.sparse_weight_bytes for row in selected_steps)
    total_full_seconds = sum(row.full_seconds for row in selected_steps)
    total_sparse_seconds = sum(row.sparse_seconds for row in selected_steps)
    saving = 1.0 - total_sparse_bytes / total_full_bytes
    speedup = total_full_seconds / total_sparse_seconds
    top1 = all(row.top1_equal for row in selected_steps)
    overlap = min(row.top10_overlap for row in selected_steps)
    cosine = min(row.hidden_cosine for row in selected_steps)
    l2 = max(row.hidden_relative_l2 for row in selected_steps)
    logit = max(row.candidate_logit_max_abs_error for row in selected_steps)
    reasons = []
    checks = (
        (len(evidence) < config.min_reports_per_candidate, "insufficient-reports"),
        (
            len({row.prefix_sha256 for row in evidence})
            < config.min_distinct_prefixes,
            "insufficient-prefixes",
        ),
        (verified_steps < config.min_verified_steps, "short-horizon"),
        (saving < config.min_byte_saving_fraction, "byte-saving"),
        (speedup < config.min_speedup, "speedup"),
        (len(layers) > config.max_sparse_layers, "layer-count"),
    )
    reasons.extend(reason for failed, reason in checks if failed)
    if verified_steps == 0:
        first = tuple(row.step_evidence[0] for row in evidence)
        first_checks = (
            (not all(row.top1_equal for row in first), "top1-divergence"),
            (
                min(row.top10_overlap for row in first) < config.min_top10_overlap,
                "top10-overlap",
            ),
            (
                min(row.hidden_cosine for row in first) < config.min_hidden_cosine,
                "hidden-cosine",
            ),
            (
                max(row.hidden_relative_l2 for row in first)
                > config.max_hidden_relative_l2,
                "hidden-relative-l2",
            ),
            (
                max(row.candidate_logit_max_abs_error for row in first)
                > config.max_candidate_logit_error,
                "candidate-logit-error",
            ),
        )
        reasons.extend(reason for failed, reason in first_checks if failed)
    measured_paths = (
        ()
        if verified_steps == 0
        else tuple(
            MlpPilotMeasuredPath(
                prefix_sha256=row.prefix_sha256,
                input_token_ids=row.teacher_forced_input_token_ids[:verified_steps],
            )
            for row in evidence
        )
    )
    return MlpPilotLayerCandidate(
        layers=layers,
        prefix_sha256s=tuple(row.prefix_sha256 for row in evidence),
        report_sha256s=tuple(row.report_sha256 for row in evidence),
        report_body_sha256s=tuple(row.report_body_sha256 for row in evidence),
        token_paths_sha256=_token_paths_sha256(measured_paths),
        verified_steps=verified_steps,
        top1_equal=top1,
        top10_overlap_min=overlap,
        hidden_cosine_min=cosine,
        hidden_relative_l2_max=l2,
        candidate_logit_max_abs_error=logit,
        byte_saving_fraction=saving,
        speedup=speedup,
        admissible=not reasons,
        rejection_reasons=tuple(reasons),
    )


def fit_mlp_pilot_layer_budget(
    reports: Sequence[bytes],
    *,
    config: MlpPilotLayerBudgetConfig = MlpPilotLayerBudgetConfig(),
) -> MlpPilotLayerBudgetPolicy:
    if not isinstance(config, MlpPilotLayerBudgetConfig):
        raise TypeError("config must be MlpPilotLayerBudgetConfig")
    evidence = tuple(parse_mlp_pilot_decode_report(row) for row in reports)
    if not evidence:
        raise ValueError("at least one decode report is required")
    identities = {
        (row.model_pin_sha256, row.router_fit_sha256, row.affine_fit_sha256)
        for row in evidence
    }
    if len(identities) != 1:
        raise MlpPilotLayerBudgetIntegrityError("decode reports cross fitted identities")
    grouped: dict[tuple[int, ...], dict[str, MlpPilotDecodeEvidence]] = {}
    for row in evidence:
        candidate = grouped.setdefault(row.sparse_layers, {})
        if row.prefix_sha256 in candidate:
            raise MlpPilotLayerBudgetIntegrityError(
                "duplicate candidate report for one prefix"
            )
        candidate[row.prefix_sha256] = row
    prefix_sets = {tuple(sorted(rows)) for rows in grouped.values()}
    if len(prefix_sets) != 1:
        raise MlpPilotLayerBudgetIntegrityError(
            "candidate actions have unequal prompt coverage"
        )
    prefixes = next(iter(prefix_sets))
    for prefix in prefixes:
        paths = [rows[prefix] for rows in grouped.values()]
        if len({row.steps for row in paths}) != 1 or len(
            {row.teacher_forced_input_token_ids for row in paths}
        ) != 1:
            raise MlpPilotLayerBudgetIntegrityError(
                "candidate actions were not measured on the same token path"
            )
    candidates = tuple(
        _candidate(layers, tuple(rows[prefix] for prefix in prefixes), config)
        for layers, rows in sorted(grouped.items())
    )
    admissible = tuple(row for row in candidates if row.admissible)
    selected = (
        max(
            admissible,
            key=lambda row: (
                row.byte_saving_fraction,
                row.speedup,
                row.hidden_cosine_min,
                row.top10_overlap_min,
                -row.hidden_relative_l2_max,
                -len(row.layers),
                tuple(-layer for layer in row.layers),
            ),
        )
        if admissible
        else None
    )
    model_pin, router_fit, affine_fit = next(iter(identities))
    measured_paths = (
        ()
        if selected is None
        else tuple(
            MlpPilotMeasuredPath(
                prefix_sha256=prefix,
                input_token_ids=grouped[selected.layers][
                    prefix
                ].teacher_forced_input_token_ids[: selected.verified_steps],
            )
            for prefix in prefixes
        )
    )
    return MlpPilotLayerBudgetPolicy(
        model_pin_sha256=model_pin,
        router_fit_sha256=router_fit,
        affine_fit_sha256=affine_fit,
        config=config,
        candidates=candidates,
        selected_layers=() if selected is None else selected.layers,
        scope_prefix_sha256s=prefixes,
        measured_paths=measured_paths,
        verified_steps=0 if selected is None else selected.verified_steps,
    )


def verify_mlp_pilot_layer_budget(
    policy: MlpPilotLayerBudgetPolicy, reports: Sequence[bytes]
) -> None:
    if not isinstance(policy, MlpPilotLayerBudgetPolicy):
        raise TypeError("policy must be MlpPilotLayerBudgetPolicy")
    rebuilt = fit_mlp_pilot_layer_budget(reports, config=policy.config)
    if rebuilt.to_bytes() != policy.to_bytes():
        raise MlpPilotLayerBudgetIntegrityError(
            "layer policy differs from its decode receipts"
        )


__all__ = [
    "MLP_PILOT_DECODE_REPORT_SCHEMA",
    "MLP_PILOT_DECODE_REPORT_SCHEMA_V3",
    "MLP_PILOT_LAYER_BUDGET_CONFIG_SCHEMA",
    "MLP_PILOT_LAYER_CANDIDATE_SCHEMA",
    "MLP_PILOT_LAYER_POLICY_SCHEMA",
    "MLP_PILOT_LAYER_POLICY_VERIFIER_SHA256",
    "MlpPilotBudgetDecision",
    "MlpPilotBudgetState",
    "MlpPilotDecodeEvidence",
    "MlpPilotDecodeStep",
    "MlpPilotLayerBudgetConfig",
    "MlpPilotLayerBudgetError",
    "MlpPilotLayerBudgetIntegrityError",
    "MlpPilotLayerBudgetPolicy",
    "MlpPilotLayerCandidate",
    "MlpPilotMeasuredPath",
    "fit_mlp_pilot_layer_budget",
    "parse_mlp_pilot_decode_report",
    "verify_mlp_pilot_layer_budget",
]
