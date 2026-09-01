"""Crash-safe economics receipts from already completed inference evidence."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import stat
import threading
from typing import Any

try:  # pragma: no cover - every production platform provides fcntl.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from ...contracts import Result
from ..ooe.identity import canonical_json_bytes, require_sha256


INFERENCE_ECONOMICS_RECEIPT_SCHEMA = "immer.qwen3.8-inference-economics-receipt/v2"
INFERENCE_ECONOMICS_EVENT_SCHEMA = "immer.qwen3.8-inference-economics-event/v2"
INFERENCE_ECONOMICS_ROLLUP_SCHEMA = "immer.qwen3.8-inference-economics-rollup/v2"
_LEGACY_RECEIPT_SCHEMA = "immer.qwen3.8-inference-economics-receipt/v1"
_LEGACY_EVENT_SCHEMA = "immer.qwen3.8-inference-economics-event/v1"
MAX_RECEIPT_BYTES = 64 * 1024
MAX_EVENT_BYTES = 80 * 1024
MAX_ROLLUP_BYTES = 64 * 1024
_EVENT = re.compile(r"([0-9]{20})-([0-9a-f]{64})\.json")
_ZERO_SHA256 = "0" * 64

_RECEIPT_V1_FIELDS = (
    "request_sha256",
    "question_sha256",
    "runtime_profile_sha256",
    "result_sha256",
    "output_sha256",
    "status",
    "component",
    "route",
    "target_forwards",
    "saved_qwen_forwards",
    "generated_tokens",
    "accepted_draft_tokens",
    "proposed_draft_tokens",
    "source_body_bytes",
    "target_source_body_bytes",
    "draft_source_body_bytes",
    "logical_weight_bytes",
    "page_mlp_weight_bytes",
    "prefetch_bytes",
    "physical_read_bytes",
    "linear_calls",
    "process_peak_rss_bytes",
    "generation_seconds",
    "request_wall_seconds",
    "time_to_first_token_seconds",
    "selected_pages",
    "saved_pages",
    "o1_priority",
    "runtime_reward",
    "warm_hit",
    "fertig_exact",
    "draft_active",
    "page_active",
    "avoidable_work_bytes",
)
_RECEIPT_V2_FIELDS = (
    *_RECEIPT_V1_FIELDS,
    "output_tokens_per_second",
    "component_timings",
)


class InferenceEconomicsError(RuntimeError):
    """The economics journal or one of its receipts failed integrity."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _uint(value: object, *, default: int = 0) -> int:
    return (
        int(value)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else default
    )


def _nonnegative_float(value: object, *, default: float = 0.0) -> float:
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
        else default
    )


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(child) for child in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError("inference evidence contains a non-canonical value")


def _component_timings(value: object) -> dict[str, object]:
    """Copy one request timing receipt without weakening its integer costs."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("component_timings must be a mapping")
    if not value:
        return {}
    safe = _json_safe(value)
    assert isinstance(safe, dict)
    components = safe.get("components")
    if not isinstance(components, dict):
        raise ValueError("component_timings components must be a mapping")
    normalized_components: dict[str, dict[str, object]] = {}
    for name, row in components.items():
        if not isinstance(name, str) or not name or not isinstance(row, dict):
            raise ValueError("component_timings contains an invalid component")
        calls = row.get("calls")
        nanoseconds = row.get("nanoseconds")
        if (
            isinstance(calls, bool)
            or not isinstance(calls, int)
            or calls < 0
            or isinstance(nanoseconds, bool)
            or not isinstance(nanoseconds, int)
            or nanoseconds < 0
        ):
            raise ValueError("component timing costs must be non-negative integers")
        boundary = row.get("boundary")
        if boundary is not None and (not isinstance(boundary, str) or not boundary):
            raise ValueError("component timing boundary must be non-empty text")
        normalized_components[name] = dict(sorted(row.items()))
    for key in ("schema", "source_schema", "clock", "unit", "status"):
        item = safe.get(key)
        if item is not None and (not isinstance(item, str) or not item):
            raise ValueError(f"component_timings {key} must be non-empty text")
    accounting_error = safe.get("accounting_error")
    if accounting_error is not None and (
        not isinstance(accounting_error, str) or not accounting_error
    ):
        raise ValueError("component_timings accounting_error is invalid")
    if "accounting_failures" in safe:
        failures = safe["accounting_failures"]
        if isinstance(failures, bool) or not isinstance(failures, int) or failures < 0:
            raise ValueError("component_timings accounting_failures is invalid")
    measured = safe.get("measured_nanoseconds")
    if measured is not None and (
        isinstance(measured, bool) or not isinstance(measured, int) or measured < 0
    ):
        raise ValueError("component_timings measured_nanoseconds is invalid")
    safe["components"] = dict(sorted(normalized_components.items()))
    return dict(sorted(safe.items()))


def _generation_and_context(
    result: Result,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    str,
]:
    evidence = _mapping(result.evidence)
    final_receipt = _mapping(evidence.get("receipt"))
    if final_receipt:
        qwen = _mapping(final_receipt.get("qwen"))
        generation = _mapping(qwen.get("generation"))
        draft = _mapping(qwen.get("draft"))
        runtime_metrics = _mapping(qwen.get("runtime_metrics"))
        q4 = _mapping(qwen.get("q4"))
        page = _mapping(qwen.get("mlp_page_route"))
        reward = _mapping(qwen.get("runtime_reward"))
        route = str(final_receipt.get("route", result.component))
        return generation, draft, runtime_metrics, q4, page, reward, route
    return (
        _mapping(evidence.get("generation")),
        _mapping(evidence.get("draft")),
        _mapping(evidence.get("runtime_metrics")),
        _mapping(evidence.get("q4")),
        _mapping(evidence.get("mlp_page_route")),
        _mapping(evidence.get("runtime_reward")),
        result.component,
    )


def _saved_qwen_forwards(
    result: Result,
    generation: Mapping[str, Any],
) -> int:
    receipt = _mapping(_mapping(result.evidence).get("receipt"))
    accounting = _mapping(_mapping(receipt.get("ooe")).get("accounting"))
    if "saved_qwen_forwards" in accounting:
        return _uint(accounting.get("saved_qwen_forwards"))
    generated = _uint(generation.get("generated_tokens"))
    target = _uint(generation.get("forward_passes"))
    return max(0, generated - target)


def _authenticated_warm_execution(result: Result, route: str) -> bool:
    receipt = _mapping(_mapping(result.evidence).get("receipt"))
    accounting = _mapping(_mapping(receipt.get("ooe")).get("accounting"))
    return "saved_qwen_forwards" in accounting or (
        route.startswith("ooe_") and route not in {"ooe_failure", "ooe_integrity_error"}
    )


def _proposed_draft_tokens(draft: Mapping[str, Any]) -> int:
    direct = draft.get("proposed_draft_tokens")
    if isinstance(direct, int) and not isinstance(direct, bool) and direct >= 0:
        return direct
    rounds = draft.get("rounds")
    if not isinstance(rounds, (tuple, list)):
        return 0
    proposed = 0
    for row in rounds:
        record = _mapping(row)
        tokens = record.get("proposed_token_ids")
        if isinstance(tokens, (tuple, list)):
            proposed += len(tokens)
        else:
            proposed += _uint(record.get("proposed_draft_tokens"))
    return proposed


@dataclass(frozen=True, slots=True)
class InferenceEconomicsReceipt:
    request_sha256: str
    question_sha256: str
    runtime_profile_sha256: str
    result_sha256: str
    output_sha256: str
    status: str
    component: str
    route: str
    target_forwards: int
    saved_qwen_forwards: int
    generated_tokens: int
    accepted_draft_tokens: int
    proposed_draft_tokens: int
    source_body_bytes: int
    target_source_body_bytes: int
    draft_source_body_bytes: int
    logical_weight_bytes: int
    page_mlp_weight_bytes: int
    prefetch_bytes: int
    physical_read_bytes: int
    linear_calls: int
    process_peak_rss_bytes: int
    generation_seconds: float
    request_wall_seconds: float
    time_to_first_token_seconds: float
    selected_pages: int
    saved_pages: int
    o1_priority: float
    runtime_reward: float
    warm_hit: bool
    fertig_exact: bool
    draft_active: bool
    page_active: bool
    avoidable_work_bytes: Mapping[str, int]
    output_tokens_per_second: float = 0.0
    component_timings: Mapping[str, object] = field(default_factory=dict)
    _document_schema: str = field(
        default=INFERENCE_ECONOMICS_RECEIPT_SCHEMA,
        repr=False,
    )

    def __post_init__(self) -> None:
        for field_name in (
            "request_sha256",
            "question_sha256",
            "runtime_profile_sha256",
            "result_sha256",
            "output_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        for field_name in (
            "status",
            "component",
            "route",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be non-empty text")
        for field_name in (
            "target_forwards",
            "saved_qwen_forwards",
            "generated_tokens",
            "accepted_draft_tokens",
            "proposed_draft_tokens",
            "source_body_bytes",
            "target_source_body_bytes",
            "draft_source_body_bytes",
            "logical_weight_bytes",
            "page_mlp_weight_bytes",
            "prefetch_bytes",
            "physical_read_bytes",
            "linear_calls",
            "process_peak_rss_bytes",
            "selected_pages",
            "saved_pages",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        for field_name in (
            "generation_seconds",
            "request_wall_seconds",
            "time_to_first_token_seconds",
            "o1_priority",
            "runtime_reward",
            "output_tokens_per_second",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{field_name} must be finite")
            if field_name != "runtime_reward" and float(value) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        for field_name in (
            "warm_hit",
            "fertig_exact",
            "draft_active",
            "page_active",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        work = dict(self.avoidable_work_bytes)
        if any(
            not isinstance(key, str)
            or not key
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in work.items()
        ):
            raise ValueError("avoidable_work_bytes is invalid")
        object.__setattr__(self, "avoidable_work_bytes", dict(sorted(work.items())))
        try:
            timings = _component_timings(self.component_timings)
        except (TypeError, ValueError) as exc:
            raise ValueError("component_timings is invalid") from exc
        object.__setattr__(self, "component_timings", timings)
        if self._document_schema not in {
            _LEGACY_RECEIPT_SCHEMA,
            INFERENCE_ECONOMICS_RECEIPT_SCHEMA,
        }:
            raise ValueError("economics receipt schema is unsupported")
        if self._document_schema == _LEGACY_RECEIPT_SCHEMA and (
            self.output_tokens_per_second != 0.0 or timings
        ):
            raise ValueError("legacy economics receipts cannot contain v2 costs")

    @property
    def sha256(self) -> str:
        return _digest(self.body())

    def body(self) -> dict[str, object]:
        fields = (
            _RECEIPT_V1_FIELDS
            if self._document_schema == _LEGACY_RECEIPT_SCHEMA
            else _RECEIPT_V2_FIELDS
        )
        return {
            name: (
                dict(self.avoidable_work_bytes)
                if name == "avoidable_work_bytes"
                else dict(self.component_timings)
                if name == "component_timings"
                else getattr(self, name)
            )
            for name in fields
        }

    def to_document(self) -> dict[str, object]:
        body = self.body()
        document = {
            "body": body,
            "schema": self._document_schema,
            "sha256": _digest(body),
        }
        if len(canonical_json_bytes(document)) > MAX_RECEIPT_BYTES:
            raise InferenceEconomicsError("economics receipt exceeds 64 KiB")
        return document

    @classmethod
    def from_document(cls, value: object) -> "InferenceEconomicsReceipt":
        if not isinstance(value, Mapping):
            raise InferenceEconomicsError("economics receipt envelope is invalid")
        schema = value.get("schema")
        body = value.get("body")
        if (
            set(value) != {"body", "schema", "sha256"}
            or schema
            not in {_LEGACY_RECEIPT_SCHEMA, INFERENCE_ECONOMICS_RECEIPT_SCHEMA}
            or not isinstance(body, Mapping)
            or value.get("sha256") != _digest(body)
        ):
            raise InferenceEconomicsError("economics receipt envelope is invalid")
        expected_fields = (
            _RECEIPT_V1_FIELDS
            if schema == _LEGACY_RECEIPT_SCHEMA
            else _RECEIPT_V2_FIELDS
        )
        if set(body) != set(expected_fields):
            raise InferenceEconomicsError("economics receipt fields are invalid")
        arguments = dict(body)
        arguments["_document_schema"] = schema
        try:
            receipt = cls(**arguments)
        except (TypeError, ValueError) as exc:
            raise InferenceEconomicsError("economics receipt is invalid") from exc
        if receipt.to_document() != dict(value):
            raise InferenceEconomicsError("economics receipt is not canonical")
        return receipt


def receipt_from_result(
    result: Result,
    *,
    question_sha256: str,
    runtime_profile_sha256: str,
    request_sha256: str | None = None,
) -> InferenceEconomicsReceipt:
    if not isinstance(result, Result):
        raise TypeError("result must be a Result")
    question_digest = require_sha256(question_sha256, field="question_sha256")
    profile_digest = require_sha256(
        runtime_profile_sha256,
        field="runtime_profile_sha256",
    )
    request_digest = (
        None
        if request_sha256 is None
        else require_sha256(request_sha256, field="request_sha256")
    )
    generation, draft, runtime_metrics, q4, page, reward, route = (
        _generation_and_context(result)
    )
    q4_request = _mapping(q4.get("request"))
    page_request = _mapping(page.get("request"))
    accepted = _uint(draft.get("accepted_draft_tokens"))
    proposed = max(accepted, _proposed_draft_tokens(draft))
    target_bytes = _uint(draft.get("target_source_body_bytes"))
    draft_bytes = _uint(draft.get("draft_source_body_bytes"))
    source_bytes = _uint(generation.get("source_body_bytes"))
    logical_bytes = _uint(q4_request.get("logical_weight_bytes"))
    page_bytes = _uint(q4_request.get("page_mlp_weight_bytes"))
    target_work = max(target_bytes, source_bytes, logical_bytes)
    rejected = max(0, proposed - accepted)
    draft_waste = 0 if proposed <= 0 else int(draft_bytes * rejected / proposed)
    work = {
        "draft_miss": draft_waste,
        "mlp_target": page_bytes,
        "q4_discard": _uint(q4_request.get("resident_unprotected_discard_bytes")),
        "target_fallback": target_work,
    }
    evidence_safe = _json_safe(dict(result.evidence))
    output_safe = _json_safe(result.output)
    output_sha256 = _digest(output_safe)
    final_receipt = _mapping(_mapping(result.evidence).get("receipt"))
    if "receipt_sha256" in final_receipt:
        result_sha256 = require_sha256(
            final_receipt["receipt_sha256"],
            field="result receipt",
        )
        receipt_core = {
            key: _json_safe(value)
            for key, value in final_receipt.items()
            if key != "receipt_sha256"
        }
        if result_sha256 != _digest(receipt_core):
            raise InferenceEconomicsError("wrapped result receipt SHA-256 mismatch")
    else:
        result_sha256 = _digest(
            {
                "component": result.component,
                "evidence": evidence_safe,
                "output_sha256": output_sha256,
                "reason": result.reason,
                "status": result.status.value,
            }
        )
    saved_forwards = _saved_qwen_forwards(result, generation)
    warm_hit = result.ok and _authenticated_warm_execution(result, route)
    selected_pages = _uint(q4_request.get("page_mlp_selected_pages"))
    full_page_actions = _uint(page.get("page_count")) * _uint(
        q4_request.get("page_mlp_rows")
    )
    route_saved_pages = max(
        _uint(page_request.get("physical_pages_saved")),
        max(0, full_page_actions - selected_pages),
    )
    receipt = InferenceEconomicsReceipt(
        request_sha256=(
            request_digest
            or _digest(
                {
                    "question_sha256": question_digest,
                    "result_sha256": result_sha256,
                    "runtime_profile_sha256": profile_digest,
                }
            )
        ),
        question_sha256=question_digest,
        runtime_profile_sha256=profile_digest,
        result_sha256=result_sha256,
        output_sha256=output_sha256,
        status=result.status.value,
        component=result.component,
        route=route,
        target_forwards=_uint(generation.get("forward_passes")),
        saved_qwen_forwards=saved_forwards,
        generated_tokens=_uint(generation.get("generated_tokens")),
        accepted_draft_tokens=accepted,
        proposed_draft_tokens=proposed,
        source_body_bytes=source_bytes,
        target_source_body_bytes=target_bytes,
        draft_source_body_bytes=draft_bytes,
        logical_weight_bytes=logical_bytes,
        page_mlp_weight_bytes=page_bytes,
        prefetch_bytes=_uint(q4_request.get("page_mlp_prefetch_bytes")),
        physical_read_bytes=_uint(runtime_metrics.get("physical_read_bytes")),
        linear_calls=_uint(generation.get("linear_calls")),
        process_peak_rss_bytes=_uint(runtime_metrics.get("process_peak_rss_bytes")),
        generation_seconds=_nonnegative_float(generation.get("seconds")),
        request_wall_seconds=_nonnegative_float(
            runtime_metrics.get("generation_wall_seconds")
        ),
        time_to_first_token_seconds=_nonnegative_float(
            generation.get("time_to_first_token_seconds")
        ),
        selected_pages=selected_pages,
        saved_pages=max(
            route_saved_pages,
            _uint(page_request.get("adaptive_width_pages_saved")),
        ),
        o1_priority=_nonnegative_float(reward.get("o1_priority")),
        runtime_reward=(
            float(reward.get("reward"))
            if isinstance(reward.get("reward"), (int, float))
            and not isinstance(reward.get("reward"), bool)
            and math.isfinite(float(reward.get("reward")))
            else 0.0
        ),
        warm_hit=warm_hit,
        fertig_exact=route == "fertig_exact_short_circuit",
        draft_active=bool(draft),
        page_active=bool(page),
        avoidable_work_bytes=work,
        output_tokens_per_second=_nonnegative_float(
            generation.get("output_tokens_per_second")
        ),
        component_timings=_component_timings(runtime_metrics.get("component_timings")),
    )
    receipt.to_document()
    return receipt


@dataclass(frozen=True, slots=True)
class InferenceEconomicsObservation:
    receipt: InferenceEconomicsReceipt
    rollup: Mapping[str, object]
    duplicate: bool


def _same_receipt_result(
    existing: InferenceEconomicsReceipt,
    observed: InferenceEconomicsReceipt,
) -> bool:
    if existing.sha256 == observed.sha256:
        return True
    if (
        existing._document_schema != _LEGACY_RECEIPT_SCHEMA
        or observed._document_schema != INFERENCE_ECONOMICS_RECEIPT_SCHEMA
    ):
        return False
    observed_body = observed.body()
    return existing.body() == {name: observed_body[name] for name in _RECEIPT_V1_FIELDS}


def _empty_rollup() -> dict[str, Any]:
    return {
        "accepted_draft_tokens": 0,
        "avoidable_work_bytes": {
            "draft_miss": 0,
            "mlp_target": 0,
            "q4_discard": 0,
            "target_fallback": 0,
        },
        "component_timings": {
            "accounting_failures": 0,
            "components": {},
            "measured_nanoseconds": 0,
            "requests": 0,
            "statuses": {},
        },
        "draft_requests": 0,
        "generated_tokens": 0,
        "head_event_sha256": _ZERO_SHA256,
        "largest_avoidable_cost_class": None,
        "output_tokens_per_second": 0.0,
        "output_tokens_per_second_observations": 0,
        "output_tokens_per_second_sum": 0.0,
        "page_requests": 0,
        "peak_output_tokens_per_second": 0.0,
        "requests": 0,
        "saved_pages": 0,
        "saved_qwen_forwards": 0,
        "schema": INFERENCE_ECONOMICS_ROLLUP_SCHEMA,
        "source_body_bytes": 0,
        "target_forwards": 0,
        "total_generation_seconds": 0.0,
        "total_request_wall_seconds": 0.0,
        "warm_hits": 0,
    }


def _accumulate(
    rollup: Mapping[str, Any],
    receipt: InferenceEconomicsReceipt,
    *,
    head_event_sha256: str,
) -> dict[str, Any]:
    result = dict(rollup)
    result["requests"] += 1
    result["target_forwards"] += receipt.target_forwards
    result["saved_qwen_forwards"] += receipt.saved_qwen_forwards
    result["generated_tokens"] += receipt.generated_tokens
    result["accepted_draft_tokens"] += receipt.accepted_draft_tokens
    result["source_body_bytes"] += max(
        receipt.source_body_bytes,
        receipt.target_source_body_bytes + receipt.draft_source_body_bytes,
    )
    result["saved_pages"] += receipt.saved_pages
    result["total_generation_seconds"] += receipt.generation_seconds
    result["total_request_wall_seconds"] += receipt.request_wall_seconds
    if (
        receipt._document_schema == INFERENCE_ECONOMICS_RECEIPT_SCHEMA
        and receipt.output_tokens_per_second > 0.0
    ):
        result["output_tokens_per_second_observations"] += 1
        result["output_tokens_per_second_sum"] += receipt.output_tokens_per_second
        result["output_tokens_per_second"] = (
            result["output_tokens_per_second_sum"]
            / result["output_tokens_per_second_observations"]
        )
        result["peak_output_tokens_per_second"] = max(
            result["peak_output_tokens_per_second"],
            receipt.output_tokens_per_second,
        )
    result["warm_hits"] += int(
        receipt.warm_hit and receipt.target_forwards == 0 and receipt.status == "ok"
    )
    result["draft_requests"] += int(receipt.draft_active)
    result["page_requests"] += int(receipt.page_active)
    if receipt.component_timings:
        timing = receipt.component_timings
        timing_rollup = dict(result["component_timings"])
        timing_rollup["requests"] += 1
        timing_rollup["accounting_failures"] += _uint(timing.get("accounting_failures"))
        measured_nanoseconds = timing.get("measured_nanoseconds")
        if isinstance(measured_nanoseconds, int) and not isinstance(
            measured_nanoseconds, bool
        ):
            timing_rollup["measured_nanoseconds"] += measured_nanoseconds
        status = timing.get("status")
        status_name = status if isinstance(status, str) and status else "unknown"
        statuses = dict(timing_rollup["statuses"])
        statuses[status_name] = statuses.get(status_name, 0) + 1
        timing_rollup["statuses"] = dict(sorted(statuses.items()))
        components = {
            name: dict(row) for name, row in dict(timing_rollup["components"]).items()
        }
        for name, raw_row in _mapping(timing.get("components")).items():
            row = _mapping(raw_row)
            aggregate = components.setdefault(
                name,
                {
                    "boundary": row.get("boundary"),
                    "calls": 0,
                    "nanoseconds": 0,
                },
            )
            boundary = row.get("boundary")
            known_boundary = aggregate.get("boundary")
            if known_boundary is None and boundary is not None:
                aggregate["boundary"] = boundary
            elif (
                boundary is not None
                and known_boundary is not None
                and boundary != known_boundary
            ):
                raise InferenceEconomicsError(
                    f"component timing boundary changed for {name!r}"
                )
            aggregate["calls"] += _uint(row.get("calls"))
            aggregate["nanoseconds"] += _uint(row.get("nanoseconds"))
        timing_rollup["components"] = dict(sorted(components.items()))
        result["component_timings"] = timing_rollup
    work = dict(result["avoidable_work_bytes"])
    for key, value in receipt.avoidable_work_bytes.items():
        work[key] = work.get(key, 0) + value
    result["avoidable_work_bytes"] = dict(sorted(work.items()))
    result["largest_avoidable_cost_class"] = (
        None
        if not work or max(work.values()) <= 0
        else min(key for key, value in work.items() if value == max(work.values()))
    )
    result["head_event_sha256"] = require_sha256(
        head_event_sha256,
        field="head_event_sha256",
    )
    return result


def _rollup_document(body: Mapping[str, Any]) -> dict[str, object]:
    document = {
        "body": dict(body),
        "schema": INFERENCE_ECONOMICS_ROLLUP_SCHEMA,
        "sha256": _digest(body),
    }
    if len(canonical_json_bytes(document)) > MAX_ROLLUP_BYTES:
        raise InferenceEconomicsError("economics rollup exceeds its byte limit")
    return document


def _parse_event_envelope(
    document: Mapping[str, object],
    *,
    sequence: int,
    previous_event_sha256: str,
    filename_sha256: str,
    event_schema: str,
    receipt_schema: str,
) -> tuple[str, InferenceEconomicsReceipt]:
    body = document.get("body")
    if (
        set(document) != {"body", "schema", "sha256"}
        or document.get("schema") != event_schema
        or not isinstance(body, Mapping)
        or set(body) != {"previous_event_sha256", "receipt", "sequence"}
        or body.get("sequence") != sequence
        or body.get("previous_event_sha256") != previous_event_sha256
        or document.get("sha256") != _digest(body)
        or document.get("sha256") != filename_sha256
    ):
        raise InferenceEconomicsError("economics event envelope is invalid")
    receipt_document = body.get("receipt")
    if (
        not isinstance(receipt_document, Mapping)
        or receipt_document.get("schema") != receipt_schema
    ):
        raise InferenceEconomicsError("economics event receipt schema is invalid")
    receipt = InferenceEconomicsReceipt.from_document(receipt_document)
    return filename_sha256, receipt


def _parse_legacy_event_document(
    document: Mapping[str, object],
    *,
    sequence: int,
    previous_event_sha256: str,
    filename_sha256: str,
) -> tuple[str, InferenceEconomicsReceipt]:
    """Read an immutable v1 event without upgrading or rewriting its body."""

    return _parse_event_envelope(
        document,
        sequence=sequence,
        previous_event_sha256=previous_event_sha256,
        filename_sha256=filename_sha256,
        event_schema=_LEGACY_EVENT_SCHEMA,
        receipt_schema=_LEGACY_RECEIPT_SCHEMA,
    )


def _parse_current_event_document(
    document: Mapping[str, object],
    *,
    sequence: int,
    previous_event_sha256: str,
    filename_sha256: str,
) -> tuple[str, InferenceEconomicsReceipt]:
    return _parse_event_envelope(
        document,
        sequence=sequence,
        previous_event_sha256=previous_event_sha256,
        filename_sha256=filename_sha256,
        event_schema=INFERENCE_ECONOMICS_EVENT_SCHEMA,
        receipt_schema=INFERENCE_ECONOMICS_RECEIPT_SCHEMA,
    )


class InferenceEconomicsLedger:
    """Append immutable receipt segments and maintain one bounded rollup."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().absolute()
        self.events = self.root / "events"
        self.staging = self.root / "staging"
        self._thread_lock = threading.RLock()
        for index, path in enumerate((self.root, self.events, self.staging)):
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                try:
                    path.mkdir(
                        parents=index == 0,
                        exist_ok=False,
                        mode=0o700,
                    )
                except FileExistsError:
                    pass
                metadata = path.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise InferenceEconomicsError(
                    "economics storage path is not a plain directory"
                )
            path.chmod(0o700)
        with self._locked():
            _events, rollup = self._restore_unlocked()
            self._write_rollup_unlocked(rollup)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        lock_path = self.root / "LOCK"
        descriptor = os.open(
            lock_path,
            os.O_CREAT
            | os.O_RDWR
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
            0o600,
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise InferenceEconomicsError("economics lock is not a regular file")
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _write_all(descriptor: int, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short economics journal write")
            view = view[written:]

    def _atomic_write(self, destination: Path, payload: bytes) -> None:
        temporary = self.staging / (
            f"{destination.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
        )
        descriptor = os.open(
            temporary,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
            0o600,
        )
        try:
            self._write_all(descriptor, payload)
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def _event_paths(self) -> tuple[Path, ...]:
        rows = []
        for path in self.events.iterdir():
            metadata = path.lstat()
            if (
                not stat.S_ISREG(metadata.st_mode)
                or _EVENT.fullmatch(path.name) is None
            ):
                raise InferenceEconomicsError("economics event inventory is invalid")
            rows.append(path)
        return tuple(sorted(rows, key=lambda path: path.name))

    def _restore_unlocked(
        self,
    ) -> tuple[tuple[tuple[str, InferenceEconomicsReceipt], ...], dict[str, Any]]:
        previous = _ZERO_SHA256
        rollup = _empty_rollup()
        events = []
        seen_receipts = set()
        seen_requests: dict[str, str] = {}
        for sequence, path in enumerate(self._event_paths()):
            match = _EVENT.fullmatch(path.name)
            assert match is not None
            if int(match.group(1)) != sequence:
                raise InferenceEconomicsError("economics sequence is not contiguous")
            raw = path.read_bytes()
            if len(raw) > MAX_EVENT_BYTES:
                raise InferenceEconomicsError("economics event exceeds its byte limit")
            try:
                document = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise InferenceEconomicsError(
                    "economics event is invalid JSON"
                ) from exc
            if canonical_json_bytes(document) != raw:
                raise InferenceEconomicsError("economics event is not canonical")
            if not isinstance(document, Mapping):
                raise InferenceEconomicsError("economics event envelope is invalid")
            schema = document.get("schema")
            if schema == _LEGACY_EVENT_SCHEMA:
                event_sha256, receipt = _parse_legacy_event_document(
                    document,
                    sequence=sequence,
                    previous_event_sha256=previous,
                    filename_sha256=match.group(2),
                )
            elif schema == INFERENCE_ECONOMICS_EVENT_SCHEMA:
                event_sha256, receipt = _parse_current_event_document(
                    document,
                    sequence=sequence,
                    previous_event_sha256=previous,
                    filename_sha256=match.group(2),
                )
            else:
                raise InferenceEconomicsError("economics event schema is unsupported")
            if receipt.sha256 in seen_receipts:
                raise InferenceEconomicsError("economics receipt is duplicated")
            prior_receipt = seen_requests.get(receipt.request_sha256)
            if prior_receipt is not None and prior_receipt != receipt.sha256:
                raise InferenceEconomicsError(
                    "one economics request has conflicting results"
                )
            seen_receipts.add(receipt.sha256)
            seen_requests[receipt.request_sha256] = receipt.sha256
            previous = event_sha256
            rollup = _accumulate(rollup, receipt, head_event_sha256=previous)
            events.append((previous, receipt))
        expected = _rollup_document(rollup)
        rollup_path = self.root / "rollup.json"
        if rollup_path.exists():
            if not stat.S_ISREG(rollup_path.lstat().st_mode):
                raise InferenceEconomicsError("economics rollup is not a regular file")
            raw_rollup = rollup_path.read_bytes()
            try:
                observed = json.loads(raw_rollup)
            except (UnicodeDecodeError, json.JSONDecodeError):
                observed = None
            if (
                not isinstance(observed, Mapping)
                or canonical_json_bytes(observed) != raw_rollup
                or observed != expected
            ):
                # The immutable event chain is the authority. A crash can land
                # its event before the replaceable rollup; callers repair it.
                pass
        return tuple(events), rollup

    def _write_rollup_unlocked(self, body: Mapping[str, Any]) -> None:
        payload = canonical_json_bytes(_rollup_document(body))
        path = self.root / "rollup.json"
        if path.exists():
            if not stat.S_ISREG(path.lstat().st_mode):
                raise InferenceEconomicsError("economics rollup is not a regular file")
            if path.read_bytes() == payload:
                return
        self._atomic_write(path, payload)

    def snapshot(self) -> dict[str, Any]:
        with self._thread_lock, self._locked():
            _events, rollup = self._restore_unlocked()
            self._write_rollup_unlocked(rollup)
            return dict(rollup)

    def receipts(self) -> tuple[InferenceEconomicsReceipt, ...]:
        """Return the immutable receipt sequence for derived local consumers."""

        with self._thread_lock, self._locked():
            events, rollup = self._restore_unlocked()
            self._write_rollup_unlocked(rollup)
            return tuple(receipt for _event_sha256, receipt in events)

    def observe(
        self,
        result: Result,
        *,
        question_sha256: str,
        runtime_profile_sha256: str,
        request_sha256: str | None = None,
    ) -> InferenceEconomicsObservation:
        receipt = receipt_from_result(
            result,
            question_sha256=question_sha256,
            runtime_profile_sha256=runtime_profile_sha256,
            request_sha256=request_sha256,
        )
        with self._thread_lock, self._locked():
            events, rollup = self._restore_unlocked()
            for _event_sha256, existing in events:
                if existing.request_sha256 == receipt.request_sha256:
                    if not _same_receipt_result(existing, receipt):
                        raise InferenceEconomicsError(
                            "one economics request produced conflicting results"
                        )
                    self._write_rollup_unlocked(rollup)
                    return InferenceEconomicsObservation(
                        receipt=existing,
                        rollup=dict(rollup),
                        duplicate=True,
                    )
            sequence = len(events)
            previous = _ZERO_SHA256 if not events else events[-1][0]
            body = {
                "previous_event_sha256": previous,
                "receipt": receipt.to_document(),
                "sequence": sequence,
            }
            event_sha256 = _digest(body)
            event = {
                "body": body,
                "schema": INFERENCE_ECONOMICS_EVENT_SCHEMA,
                "sha256": event_sha256,
            }
            payload = canonical_json_bytes(event)
            if len(payload) > MAX_EVENT_BYTES:
                raise InferenceEconomicsError("economics event exceeds its byte limit")
            destination = self.events / f"{sequence:020d}-{event_sha256}.json"
            self._atomic_write(destination, payload)
            updated = _accumulate(
                rollup,
                receipt,
                head_event_sha256=event_sha256,
            )
            self._write_rollup_unlocked(updated)
            return InferenceEconomicsObservation(
                receipt=receipt,
                rollup=dict(updated),
                duplicate=False,
            )


__all__ = [
    "INFERENCE_ECONOMICS_EVENT_SCHEMA",
    "INFERENCE_ECONOMICS_RECEIPT_SCHEMA",
    "INFERENCE_ECONOMICS_ROLLUP_SCHEMA",
    "InferenceEconomicsError",
    "InferenceEconomicsLedger",
    "InferenceEconomicsObservation",
    "InferenceEconomicsReceipt",
    "receipt_from_result",
]
