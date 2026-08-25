"""Gold-label-free audit of the sealed FERTIG GSM8K abstentions.

The benchmark report is the admission boundary: this script validates its
complete outcome partition, then re-runs every baseline-abstained question
through the vendored legacy binder and the closed structural parser.
Gold answers and predictions are deliberately never read by the classifier.

    PYTHONPATH=src python3 scripts/fertig_abstention_audit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from immer.cognition.fertig.adapter import _solver_from_checkout
from immer.cognition.fertig.structural import parse_structural_problem


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK_PATH = ROOT / "results" / "bench_gsm8k.json"
DEFAULT_OUTPUT_PATH = ROOT / "results" / "fertig-abstention-audit.json"
VENDOR_ROOT = ROOT / "src" / "immer" / "cognition" / "fertig" / "_vendor"
LEGACY_BINDINGS_PATH = VENDOR_ROOT / "fertig" / "bindings.py"
STRUCTURAL_PATH = ROOT / "src" / "immer" / "cognition" / "fertig" / "structural.py"
SIGNED_EVENT_PATH = (
    ROOT / "src" / "immer" / "cognition" / "fertig" / "signed_event_frontend.py"
)
SIGNED_EXPRESSION_PATH = (
    ROOT / "src" / "immer" / "cognition" / "fertig" / "signed_expression.py"
)
DISCOURSE_SSA_PATH = (
    ROOT / "src" / "immer" / "cognition" / "fertig" / "discourse_ssa.py"
)

SCHEMA = "immer.fertig-abstention-audit/v2"
REPORT_REVISION = 7
ALLOWED_STATUSES = ("correct", "abstained", "incorrect", "error")
LEGACY_CATEGORIES = (
    "target_parse_failed",
    "target_evidence_missing",
    "relation_incomplete",
    "equation_guard",
)
STRUCTURAL_CATEGORIES = (
    "exact_recovery",
    "numeric_pronoun_ambiguous",
    "question_pronoun_ambiguous",
    "numeric_clause_unsupported",
    "target_unsupported",
    "relation_ambiguous",
    "relation_unsupported",
    "relation_invalid",
)
CURRENT_LEGACY_COUNTS = {
    "target_parse_failed": 57,
    "target_evidence_missing": 154,
    "relation_incomplete": 18,
    "equation_guard": 1,
}
CURRENT_STRUCTURAL_COUNTS = {
    "exact_recovery": 70,
    "numeric_pronoun_ambiguous": 71,
    "question_pronoun_ambiguous": 3,
    "numeric_clause_unsupported": 80,
    "target_unsupported": 5,
    "relation_ambiguous": 1,
    "relation_unsupported": 0,
    "relation_invalid": 0,
}
CURRENT_SCOPE_COUNTS = {
    "exact_recovery_scope": 70,
    "potential_coreference_scope": 74,
    "grammar_scope": 86,
}
CURRENT_EXACT_RECOVERY_INDICES = (
    547,
    550,
    574,
    578,
    587,
    603,
    610,
    613,
    619,
    627,
    631,
    643,
    651,
    672,
    682,
    692,
    694,
    701,
    722,
    724,
    731,
    745,
    746,
    747,
    754,
    763,
    770,
    778,
    780,
    782,
    797,
    802,
    810,
    819,
    823,
    825,
    833,
    836,
    837,
    840,
    844,
    861,
    865,
    868,
    883,
    892,
    900,
    901,
    913,
    916,
    924,
    930,
    934,
    944,
    959,
    992,
    1016,
    1064,
    1172,
    1192,
    1194,
    1217,
    1219,
    1246,
    1252,
    1253,
    1261,
    1293,
    1300,
    1304,
)
BASELINE_EXACT_RECOVERIES = 42
GROUND_WAVE_EXACT_RECOVERY_INDICES = (672, 780, 944, 1219, 1261)
GROUND_WAVE_EXACT_MECHANISMS = {
    672: "signed_event:temporal_categorical_block_remainder",
    780: "signed_event:absolute_weighted_score_difference",
    944: "signed_event:exhaustive_unit_rate_ledger",
    1219: "signed_event:exhaustive_unit_rate_ledger",
    1261: "signed_event:recurring_pronoun_rate_ledger",
}
GROUND_WAVE_ID = "ground-operators-conservative-coreference/v1"
DISCOURSE_SSA_WAVE_EXACT_RECOVERY_INDICES = (
    578,
    603,
    802,
    833,
    900,
    959,
    1064,
    1194,
    1252,
    1304,
)
DISCOURSE_SSA_WAVE_EXACT_MECHANISMS = {
    578: "signed_event:grounded_value_pipeline",
    603: "signed_event:shared_duration_affine_rates",
    802: "signed_event:closed_collection_share_completion",
    833: "signed_event:typed_scale_chain_conversion",
    900: "signed_event:temporal_reader_affine_difference",
    959: "signed_event:closed_named_scale_group_total",
    1064: "signed_event:ordered_affine_category_ledger",
    1194: "signed_event:typed_species_scale_total",
    1252: "signed_event:typed_ratio_property_chain_total",
    1304: "signed_event:typed_scaled_measure_difference",
}
DISCOURSE_SSA_WAVE_ID = "typed-discourse-ssa-affine/v1"
CLOSED_SCHEDULE_WAVE_EXACT_RECOVERY_INDICES = (
    627,
    782,
    992,
    1192,
    1217,
    1246,
    1253,
    1293,
    1300,
)
CLOSED_SCHEDULE_WAVE_EXACT_MECHANISMS = {
    627: "signed_event:closed_week_complement_schedule",
    782: "signed_event:closed_disjoint_week_schedule",
    992: "signed_event:calendar_frequency_ledger",
    1192: "signed_event:explicit_weekly_pay_schedule",
    1217: "signed_event:closed_piecewise_period_cost",
    1246: "signed_event:explicit_period_score_total",
    1253: "signed_event:canonical_duration_rate_conversion",
    1293: "signed_event:canonical_weekly_sales_total",
    1300: "signed_event:explicit_weekday_exception_schedule",
}
CLOSED_SCHEDULE_WAVE_ID = "closed-calendar-schedule-algebra/v1"
RECURRENCE_WAVE_EXACT_RECOVERY_INDICES = (547, 810, 1016, 1172)
RECURRENCE_WAVE_EXACT_MECHANISMS = {
    547: "signed_event:closed_phone_tree_recurrence",
    810: "signed_event:closed_monthly_state_recurrence",
    1016: "signed_event:fixed_base_percentage_recurrence",
    1172: "signed_event:closed_daily_geometric_total",
}
RECURRENCE_WAVE_ID = "closed-recurrence-algebra/v1"
RECOVERY_WAVES = (
    (
        GROUND_WAVE_ID,
        GROUND_WAVE_EXACT_RECOVERY_INDICES,
        GROUND_WAVE_EXACT_MECHANISMS,
    ),
    (
        DISCOURSE_SSA_WAVE_ID,
        DISCOURSE_SSA_WAVE_EXACT_RECOVERY_INDICES,
        DISCOURSE_SSA_WAVE_EXACT_MECHANISMS,
    ),
    (
        CLOSED_SCHEDULE_WAVE_ID,
        CLOSED_SCHEDULE_WAVE_EXACT_RECOVERY_INDICES,
        CLOSED_SCHEDULE_WAVE_EXACT_MECHANISMS,
    ),
    (
        RECURRENCE_WAVE_ID,
        RECURRENCE_WAVE_EXACT_RECOVERY_INDICES,
        RECURRENCE_WAVE_EXACT_MECHANISMS,
    ),
)
CURRENT_EXACT_RECOVERY_MECHANISMS = {
    547: "signed_event:closed_phone_tree_recurrence",
    550: "signed_event:discounted_purchase_ledger",
    574: "signed_event:batch_sale_profit",
    578: "signed_event:grounded_value_pipeline",
    587: "signed_event:balanced_percent_category_difference",
    603: "signed_event:shared_duration_affine_rates",
    610: "signed_event:rate_length_difference",
    613: "signed_event:calendar_daily_total",
    619: "signed_event:functioning_count",
    627: "signed_event:closed_week_complement_schedule",
    631: "signed_event:affine_price_chain_total",
    643: "signed_event:combined_daily_total",
    651: "signed_event:part_scaled_period_total",
    672: "signed_event:temporal_categorical_block_remainder",
    682: "preexisting_generic_structural_exact",
    692: "signed_event:bundle_relative_price_dag",
    694: "signed_event:typed_chair_capacity_deficit",
    701: "signed_event:group_seat_purchase",
    722: "signed_event:avoided_cost_transaction",
    724: "signed_event:fractional_group_consumption_remainder",
    731: "signed_event:exact_packaging_capacity",
    745: "signed_event:profit_contribution_ledger",
    746: "signed_event:alternative_cost_savings",
    747: "signed_event:old_new_rate_savings",
    754: "signed_event:typed_percentage_trade_transitions",
    763: "signed_event:ordinal_ratio_partition",
    770: "signed_event:category_sales_difference",
    778: "clause_compiler",
    780: "signed_event:absolute_weighted_score_difference",
    782: "signed_event:closed_disjoint_week_schedule",
    797: "signed_event:repeated_duration_total",
    802: "signed_event:closed_collection_share_completion",
    810: "signed_event:closed_monthly_state_recurrence",
    819: "signed_event:equal_share_residual",
    823: "signed_event:temporal_affine_score_chain",
    825: "clause_compiler",
    833: "signed_event:typed_scale_chain_conversion",
    836: "signed_event:entity_affine_chain_total",
    837: "signed_event:funding_balance_residual",
    840: "signed_event:fractional_remnant_total",
    844: "signed_event:cross_entity_property_dag",
    861: "signed_event:inverse_rate_time_difference",
    865: "signed_event:chained_inventory_residual",
    868: "signed_event:exact_package_demand_cost",
    883: "signed_event:reverse_affine_state_duration",
    892: "signed_event:unit_cost_residual",
    900: "signed_event:temporal_reader_affine_difference",
    901: "signed_event:typed_bowl_capacity_leftover",
    913: "signed_event:weighted_bundle_residual_count",
    916: "signed_event:equal_allowance_purchase_balance",
    924: "signed_event:exact_trip_capacity_minimum",
    930: "signed_event:inventory_total_residual",
    934: "signed_event:mean_participant_totals",
    944: "signed_event:exhaustive_unit_rate_ledger",
    959: "signed_event:closed_named_scale_group_total",
    992: "signed_event:calendar_frequency_ledger",
    1016: "signed_event:fixed_base_percentage_recurrence",
    1064: "signed_event:ordered_affine_category_ledger",
    1172: "signed_event:closed_daily_geometric_total",
    1192: "signed_event:explicit_weekly_pay_schedule",
    1194: "signed_event:typed_species_scale_total",
    1217: "signed_event:closed_piecewise_period_cost",
    1219: "signed_event:exhaustive_unit_rate_ledger",
    1246: "signed_event:explicit_period_score_total",
    1252: "signed_event:typed_ratio_property_chain_total",
    1253: "signed_event:canonical_duration_rate_conversion",
    1261: "signed_event:recurring_pronoun_rate_ledger",
    1293: "signed_event:canonical_weekly_sales_total",
    1300: "signed_event:explicit_weekday_exception_schedule",
    1304: "signed_event:typed_scaled_measure_difference",
}
_INCOMPLETE_BINDING = re.compile(
    r"^Bindung unvollständig: (?P<qty>\d+) qty, "
    r"(?P<ratio>\d+) ratio, op=(?P<op>[a-z_]+)$"
)
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AuditError(RuntimeError):
    """The input or extracted evidence cannot support a sealed audit."""


def _canonical_json_bytes(document: Any) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_int(document: Mapping[str, Any], key: str) -> int:
    value = document.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuditError(f"benchmark field {key!r} must be a non-negative integer")
    return value


def _require_sha256(document: Mapping[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise AuditError(f"benchmark field {key!r} must be a lowercase SHA-256")
    return value


def load_benchmark_report(path: Path) -> Mapping[str, Any]:
    """Load a JSON object without weakening validation on malformed input."""

    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot load benchmark report {path}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise AuditError("benchmark report must be a JSON object")
    return document


def validate_benchmark_report(
    report: Mapping[str, Any], *, require_zero_wrong: bool = True
) -> tuple[tuple[Mapping[str, Any], ...], dict[str, int]]:
    """Validate the full baseline partition without reading any gold answer."""

    if report.get("schema") != "immer.benchmark/v1":
        raise AuditError("unsupported benchmark schema")
    if report.get("benchmark") != "GSM8K-test":
        raise AuditError("audit requires the GSM8K-test benchmark")
    n = _require_int(report, "n")
    for key in ("dataset_sha256", "harness_sha256", "items_sha256"):
        _require_sha256(report, key)

    raw_items = report.get("items")
    if not isinstance(raw_items, list):
        raise AuditError("benchmark items must be a JSON array")
    if len(raw_items) != n:
        raise AuditError(f"benchmark n={n} but contains {len(raw_items)} items")

    counts: Counter[str] = Counter()
    indexes: set[int] = set()
    item_ids: set[str] = set()
    items: list[Mapping[str, Any]] = []
    for position, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, Mapping):
            raise AuditError(f"benchmark item {position} must be an object")
        index = raw_item.get("index")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise AuditError(f"benchmark item {position} has an invalid index")
        if index in indexes:
            raise AuditError(f"duplicate benchmark index {index}")
        indexes.add(index)

        item_id = raw_item.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise AuditError(f"benchmark item {position} has an invalid item_id")
        if item_id in item_ids:
            raise AuditError(f"duplicate benchmark item_id {item_id!r}")
        item_ids.add(item_id)

        status = raw_item.get("status")
        if status not in ALLOWED_STATUSES:
            raise AuditError(f"benchmark item {item_id!r} has status {status!r}")
        question = raw_item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise AuditError(f"benchmark item {item_id!r} has no question text")
        counts[str(status)] += 1
        items.append(raw_item)

    if indexes != set(range(n)):
        raise AuditError("benchmark indexes do not form the complete range 0..n-1")
    normalized_counts = {status: counts[status] for status in ALLOWED_STATUSES}
    if sum(normalized_counts.values()) != n:
        raise AuditError("benchmark statuses do not form a complete partition")
    for status, observed in normalized_counts.items():
        claimed = _require_int(report, status if status != "error" else "errors")
        if claimed != observed:
            raise AuditError(
                f"benchmark claims {claimed} {status} items, observed {observed}"
            )

    wrong = normalized_counts["incorrect"] + normalized_counts["error"]
    if _require_int(report, "wrong") != wrong:
        raise AuditError("benchmark wrong count is inconsistent with its partition")
    gate = report.get("wrong_must_be_zero")
    if not isinstance(gate, bool) or gate is not (wrong == 0):
        raise AuditError("benchmark wrong_must_be_zero gate is inconsistent")
    if require_zero_wrong and wrong:
        raise AuditError(f"benchmark has {wrong} wrong/error outcomes; expected zero")
    return tuple(items), normalized_counts


def classify_legacy_binding(result: Any) -> dict[str, Any]:
    """Classify one failed legacy bind from its extracted contract only."""

    if bool(getattr(result, "ok", False)):
        raise AuditError("baseline abstention now succeeds in the legacy binder")
    if getattr(result, "answer", None) is not None:
        raise AuditError("failed legacy binding unexpectedly carries an answer")
    reason = getattr(result, "reason", None)
    target = getattr(result, "target", None)
    if not isinstance(reason, str) or target is None:
        raise AuditError("legacy binder returned an incomplete diagnostic result")
    target_ok = bool(getattr(target, "ok", False))

    if reason == "kein Frageziel":
        if target_ok:
            raise AuditError("legacy target is marked valid despite kein Frageziel")
        return {"category": "target_parse_failed", "reason": reason}
    if not target_ok:
        raise AuditError(f"unrecognized failed-target legacy reason: {reason!r}")
    if reason == "Zeit-/Geldbilanz nicht strukturell bewiesen":
        return {"category": "equation_guard", "reason": reason}

    match = _INCOMPLETE_BINDING.fullmatch(reason)
    if match is None:
        raise AuditError(f"unrecognized legacy abstention reason: {reason!r}")
    qty = int(match.group("qty"))
    details = {
        "qty_for_target": qty,
        "ratios_for_target": int(match.group("ratio")),
        "operation": match.group("op"),
    }
    category = "target_evidence_missing" if qty == 0 else "relation_incomplete"
    return {"category": category, "reason": reason, "details": details}


def _status_value(result: Any) -> str:
    status = getattr(result, "status", None)
    value = getattr(status, "value", status)
    if not isinstance(value, str):
        raise AuditError("structural parser returned no status")
    return value


def classify_structural_parse(result: Any) -> dict[str, str]:
    """Classify one current closed-parser outcome without reading gold."""

    if bool(getattr(result, "ok", False)):
        status = _status_value(result)
        if status != "parsed":
            raise AuditError(
                f"successful structural parse has status {status!r}, expected 'parsed'"
            )
        reason = getattr(result, "reason", "")
        if not isinstance(reason, str):
            raise AuditError("successful structural parse returned no reason")
        return {
            "category": "exact_recovery",
            "status": status,
            "reason": reason,
        }

    status = _status_value(result)
    reason = getattr(result, "reason", None)
    if not isinstance(reason, str):
        raise AuditError("structural parser returned no diagnostic reason")

    if reason == "numeric pronoun binding is not proven":
        category = "numeric_pronoun_ambiguous"
        expected_status = "ambiguous"
    elif reason == "question pronoun binding is not proven":
        category = "question_pronoun_ambiguous"
        expected_status = "ambiguous"
    elif reason.startswith("unparsed numeric clause at "):
        category = "numeric_clause_unsupported"
        expected_status = "unsupported"
    elif reason.startswith("unsupported target at "):
        category = "target_unsupported"
        expected_status = "unsupported"
    elif status == "ambiguous":
        category = "relation_ambiguous"
        expected_status = "ambiguous"
    elif status == "unsupported":
        category = "relation_unsupported"
        expected_status = "unsupported"
    elif status == "invalid":
        category = "relation_invalid"
        expected_status = "invalid"
    else:
        raise AuditError(
            f"unrecognized structural status {status!r} for reason {reason!r}"
        )
    if status != expected_status:
        raise AuditError(
            f"structural reason {reason!r} has status {status!r}, "
            f"expected {expected_status!r}"
        )
    return {"category": category, "status": status, "reason": reason}


def _scope_for_structural(category: str) -> str:
    if category == "exact_recovery":
        return "exact_recovery_scope"
    if category in {"numeric_pronoun_ambiguous", "question_pronoun_ambiguous"}:
        return "potential_coreference_scope"
    if category in {
        "numeric_clause_unsupported",
        "target_unsupported",
        "relation_ambiguous",
        "relation_unsupported",
        "relation_invalid",
    }:
        return "grammar_scope"
    raise AuditError(f"unrecognized structural category {category!r}")


def _ordered_counts(counter: Counter[str], categories: Sequence[str]) -> dict[str, int]:
    unknown = set(counter) - set(categories)
    if unknown:
        raise AuditError(f"unexpected classification categories: {sorted(unknown)}")
    return {category: counter[category] for category in categories}


def _exact_recovery_mechanism(structural: Mapping[str, Any]) -> str:
    """Name the exact mechanism without assigning old recoveries to a new wave."""

    if structural.get("category") != "exact_recovery":
        raise AuditError("mechanism attribution requires an exact recovery")
    reason = structural.get("reason")
    if not isinstance(reason, str):
        raise AuditError("exact recovery reason must be text")
    prefix = "evidence-closed signed event grammar"
    if reason == prefix:
        return "preexisting_generic_signed_event_exact"
    if reason.startswith(f"{prefix}: "):
        family = reason.removeprefix(f"{prefix}: ")
        if re.fullmatch(r"[a-z][a-z0-9_]*", family) is None:
            raise AuditError(f"invalid signed-event mechanism family {family!r}")
        return f"signed_event:{family}"
    if reason == "evidence-closed clause compiler":
        return "clause_compiler"
    if reason == "":
        return "preexisting_generic_structural_exact"
    raise AuditError(f"unrecognized exact recovery mechanism reason {reason!r}")


def _indices_identity(indices: Sequence[int]) -> dict[str, Any]:
    ordered = list(indices)
    return {
        "count": len(ordered),
        "indices": ordered,
        "indices_sha256": _sha256_bytes(_canonical_json_bytes(ordered)),
    }


def _mechanism_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        mechanism = row.get("exact_recovery_mechanism")
        index = row.get("index")
        if not isinstance(mechanism, str) or not mechanism:
            raise AuditError("exact recovery item lacks mechanism attribution")
        if isinstance(index, bool) or not isinstance(index, int):
            raise AuditError("exact recovery item has an invalid index")
        grouped.setdefault(mechanism, []).append(index)
    return {
        mechanism: _indices_identity(sorted(indices))
        for mechanism, indices in sorted(grouped.items())
    }


def _mechanism_identity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = [
        {
            "index": int(row["index"]),
            "mechanism": str(row["exact_recovery_mechanism"]),
        }
        for row in sorted(rows, key=lambda row: int(row["index"]))
    ]
    return {
        "count": len(ordered),
        "rows": ordered,
        "rows_sha256": _sha256_bytes(_canonical_json_bytes(ordered)),
    }


def _recovery_attribution(
    exact_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    ordered_rows = sorted(exact_rows, key=lambda row: int(row["index"]))
    current_indices = tuple(int(row["index"]) for row in ordered_rows)
    current_total = len(current_indices)
    cumulative = {
        "current_total": current_total,
        "identity": _indices_identity(current_indices),
        "mechanisms": _mechanism_groups(ordered_rows),
        "mechanism_identity": _mechanism_identity(ordered_rows),
        "attribution": (
            "The current total is cumulative across every listed exact mechanism; "
            "it is not attributed to one planner wave."
        ),
    }

    wave_indices_seen: set[int] = set()
    for wave_id, indices, mechanisms in RECOVERY_WAVES:
        index_set = set(indices)
        if len(index_set) != len(indices) or wave_indices_seen.intersection(index_set):
            raise AuditError(f"recovery-wave indices overlap at {wave_id}")
        if set(mechanisms) != index_set:
            raise AuditError(f"recovery-wave mechanisms are incomplete at {wave_id}")
        wave_indices_seen.update(index_set)

    baseline_rows = [
        row for row in ordered_rows if int(row["index"]) not in wave_indices_seen
    ]
    accumulated_rows = list(baseline_rows)
    wave_deltas: list[dict[str, Any]] = []
    for wave_id, indices, _ in RECOVERY_WAVES:
        index_set = set(indices)
        wave_rows = [row for row in ordered_rows if int(row["index"]) in index_set]
        prior_rows = sorted(accumulated_rows, key=lambda row: int(row["index"]))
        accumulated_rows.extend(wave_rows)
        accumulated_rows.sort(key=lambda row: int(row["index"]))
        wave_indices = tuple(int(row["index"]) for row in wave_rows)
        prior_indices = tuple(int(row["index"]) for row in prior_rows)
        wave_total = len(wave_indices)
        prior_total = len(prior_indices)
        after_total = len(accumulated_rows)
        wave_deltas.append(
            {
                "wave_id": wave_id,
                "previously_sealed_exact_recoveries": prior_total,
                "added_exact_recoveries": wave_total,
                "current_exact_recoveries": after_total,
                "previously_sealed_identity": _indices_identity(prior_indices),
                "added_identity": _indices_identity(wave_indices),
                "mechanisms": _mechanism_groups(wave_rows),
                "mechanism_identity": _mechanism_identity(wave_rows),
                "attribution": (
                    f"{after_total} exact recoveries after {wave_id} = "
                    f"{prior_total} previously sealed + {wave_total} added; this "
                    f"wave claims {wave_total}, not {after_total}."
                ),
            }
        )

    if tuple(int(row["index"]) for row in accumulated_rows) != current_indices:
        raise AuditError("recovery-wave history does not cover the current identity")
    baseline_indices = tuple(int(row["index"]) for row in baseline_rows)
    history = {
        "baseline_exact_recoveries": len(baseline_indices),
        "baseline_identity": _indices_identity(baseline_indices),
        "waves": wave_deltas,
        "current_exact_recoveries": current_total,
        "current_identity": _indices_identity(current_indices),
        "attribution": (
            f"{current_total} current exact recoveries = {len(baseline_indices)} "
            + "sealed baseline"
            + "".join(
                f" + {wave['added_exact_recoveries']} by {wave['wave_id']}"
                for wave in wave_deltas
            )
            + "; wave deltas are disjoint and cumulative."
        ),
    }
    return cumulative, wave_deltas[-1], history


def _validate_current_evidence(report: Mapping[str, Any]) -> None:
    if (
        report.get("schema") != SCHEMA
        or report.get("report_revision") != REPORT_REVISION
    ):
        raise AuditError("current audit schema or report revision drifted")
    observed = report["classification_counts"]
    expected = {
        "legacy": CURRENT_LEGACY_COUNTS,
        "structural": CURRENT_STRUCTURAL_COUNTS,
        "scope": CURRENT_SCOPE_COUNTS,
    }
    if observed != expected:
        raise AuditError(
            "current FERTIG evidence drifted; expected "
            f"{expected!r}, observed {observed!r}"
        )
    raw_items = report.get("items")
    if not isinstance(raw_items, list):
        raise AuditError("current audit evidence has no item array")
    exact_rows = [
        row
        for row in raw_items
        if isinstance(row, Mapping)
        and isinstance(row.get("structural"), Mapping)
        and row["structural"].get("category") == "exact_recovery"
    ]
    exact_rows.sort(key=lambda row: int(row["index"]))
    observed_indices = tuple(int(row["index"]) for row in exact_rows)
    if observed_indices != CURRENT_EXACT_RECOVERY_INDICES:
        raise AuditError(
            "current exact-recovery identity drifted; expected "
            f"{CURRENT_EXACT_RECOVERY_INDICES!r}, observed {observed_indices!r}"
        )
    for row in exact_rows:
        derived = _exact_recovery_mechanism(row["structural"])
        if row.get("exact_recovery_mechanism") != derived:
            raise AuditError(
                f"exact recovery {row['index']} has inconsistent mechanism attribution"
            )
    observed_mechanisms = {
        int(row["index"]): str(row["exact_recovery_mechanism"]) for row in exact_rows
    }
    if observed_mechanisms != CURRENT_EXACT_RECOVERY_MECHANISMS:
        raise AuditError(
            "current exact-recovery mechanisms drifted; expected "
            f"{CURRENT_EXACT_RECOVERY_MECHANISMS!r}, observed "
            f"{observed_mechanisms!r}"
        )
    cumulative, wave, history = _recovery_attribution(exact_rows)
    if report.get("exact_recovery_attribution") != cumulative:
        raise AuditError("current exact-recovery attribution drifted")
    if report.get("recovery_wave_delta") != wave:
        raise AuditError("current recovery-wave delta drifted")
    if report.get("recovery_wave_history") != history:
        raise AuditError("current recovery-wave history drifted")
    if history["baseline_exact_recoveries"] != BASELINE_EXACT_RECOVERIES:
        raise AuditError("baseline exact-recovery total drifted")
    expected_prior = BASELINE_EXACT_RECOVERIES
    for delta, (wave_id, indices, mechanisms) in zip(
        history["waves"], RECOVERY_WAVES, strict=True
    ):
        if delta["wave_id"] != wave_id:
            raise AuditError("recovery-wave order drifted")
        if delta["previously_sealed_exact_recoveries"] != expected_prior:
            raise AuditError(f"pre-wave exact-recovery total drifted at {wave_id}")
        if tuple(delta["added_identity"]["indices"]) != indices:
            raise AuditError(f"recovery-wave identity drifted at {wave_id}")
        observed_wave_mechanisms = {
            row["index"]: row["mechanism"]
            for row in delta["mechanism_identity"]["rows"]
        }
        if observed_wave_mechanisms != mechanisms:
            raise AuditError(
                f"recovery-wave mechanism attribution drifted at {wave_id}"
            )
        expected_prior += len(indices)
        if delta["current_exact_recoveries"] != expected_prior:
            raise AuditError(f"post-wave exact-recovery total drifted at {wave_id}")
    if expected_prior != len(CURRENT_EXACT_RECOVERY_INDICES):
        raise AuditError("recovery-wave chain does not reach the current total")


def build_audit_report(
    benchmark: Mapping[str, Any],
    *,
    legacy_bind: Callable[[str], Any],
    structural_parse: Callable[[str], Any],
    provenance: Mapping[str, Any],
    require_zero_wrong: bool = True,
    require_current_evidence: bool = True,
) -> dict[str, Any]:
    """Build a deterministic audit using only text and parser diagnostics."""

    benchmark_items, baseline_counts = validate_benchmark_report(
        benchmark, require_zero_wrong=require_zero_wrong
    )
    audited_items: list[dict[str, Any]] = []
    legacy_counts: Counter[str] = Counter()
    structural_counts: Counter[str] = Counter()
    scope_counts: Counter[str] = Counter()

    for item in benchmark_items:
        if item["status"] != "abstained":
            continue
        item_id = str(item["item_id"])
        question = str(item["question"])
        try:
            legacy = classify_legacy_binding(legacy_bind(question))
            structural = classify_structural_parse(structural_parse(question))
        except Exception as exc:
            if isinstance(exc, AuditError):
                raise AuditError(f"{item_id}: {exc}") from exc
            raise AuditError(
                f"{item_id}: classifier failed with {type(exc).__name__}: {exc}"
            ) from exc

        legacy_counts[legacy["category"]] += 1
        structural_counts[structural["category"]] += 1
        scope = _scope_for_structural(structural["category"])
        scope_counts[scope] += 1
        audited_item = {
            "item_id": item_id,
            "index": item["index"],
            "question_sha256": _sha256_bytes(question.encode("utf-8")),
            "legacy": legacy,
            "structural": structural,
            "current_scope": scope,
            "hungarian_exclusive_assignment_eligible": False,
            "hungarian_ineligibility_reason": (
                "no_extracted_exclusive_candidate_sets_or_cost_matrix"
            ),
        }
        if structural["category"] == "exact_recovery":
            audited_item["exact_recovery_mechanism"] = _exact_recovery_mechanism(
                structural
            )
        audited_items.append(audited_item)

    audited = len(audited_items)
    if audited != baseline_counts["abstained"]:
        raise AuditError("not every baseline abstention was audited")
    counts = {
        "legacy": _ordered_counts(legacy_counts, LEGACY_CATEGORIES),
        "structural": _ordered_counts(structural_counts, STRUCTURAL_CATEGORIES),
        "scope": _ordered_counts(
            scope_counts,
            (
                "exact_recovery_scope",
                "potential_coreference_scope",
                "grammar_scope",
            ),
        ),
    }
    for name, partition in counts.items():
        if sum(partition.values()) != audited:
            raise AuditError(f"{name} classifications are not a complete partition")

    recoveries = counts["structural"]["exact_recovery"]
    remaining = audited - recoveries
    exact_rows = [
        item
        for item in audited_items
        if item["structural"]["category"] == "exact_recovery"
    ]
    exact_attribution, wave_delta, wave_history = _recovery_attribution(exact_rows)

    eligible = sum(
        bool(item["hungarian_exclusive_assignment_eligible"]) for item in audited_items
    )
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "report_revision": REPORT_REVISION,
        "benchmark": "GSM8K-test",
        "gold_label_free": True,
        "classification_inputs": [
            "baseline status",
            "question text",
            "vendored legacy binding diagnostics",
            "structural parse diagnostics",
        ],
        "baseline_partition": {"n": len(benchmark_items), **baseline_counts},
        "audited_abstentions": audited,
        "current_exact_recoveries": recoveries,
        "current_remaining_abstentions": remaining,
        "exact_recovery_attribution": exact_attribution,
        "recovery_wave_delta": wave_delta,
        "recovery_wave_history": wave_history,
        "classification_counts": counts,
        "hungarian_verdict": {
            "contract": (
                "an instance is directly eligible only when extraction supplies "
                "two exclusive candidate sets and a complete pairwise cost matrix"
            ),
            "directly_eligible_exclusive_instances": eligible,
            "verdict": (
                "0 directly eligible exclusive Hungarian instances under the "
                "current extracted contract; exact recoveries need no assignment, "
                "and the remaining scopes are coreference or grammar/relation "
                "construction, not global one-to-one assignment"
            ),
        },
        "baseline_bindings": {
            "dataset_sha256": benchmark["dataset_sha256"],
            "harness_sha256": benchmark["harness_sha256"],
            "items_sha256": benchmark["items_sha256"],
        },
        "provenance": dict(provenance),
        "items_sha256": _sha256_bytes(_canonical_json_bytes(audited_items)),
        "items": audited_items,
    }
    if eligible != 0:
        raise AuditError("Hungarian eligibility changed from the closed zero baseline")
    if require_current_evidence:
        _validate_current_evidence(document)
    document["report_sha256"] = _sha256_bytes(_canonical_json_bytes(document))
    return document


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def current_provenance(benchmark_path: Path) -> dict[str, Any]:
    """Bind the audit to its input, code, parsers, and classification contract."""

    files = {
        "benchmark_report": benchmark_path,
        "audit_runner": Path(__file__).resolve(),
        "legacy_bindings": LEGACY_BINDINGS_PATH,
        "structural_parser": STRUCTURAL_PATH,
        "signed_event_frontend": SIGNED_EVENT_PATH,
        "signed_expression_compiler": SIGNED_EXPRESSION_PATH,
        "typed_discourse_ssa": DISCOURSE_SSA_PATH,
    }
    file_evidence = {
        name: {"path": _path_label(path), "sha256": _sha256_file(path)}
        for name, path in files.items()
    }
    contract = {
        "legacy_categories": list(LEGACY_CATEGORIES),
        "structural_categories": list(STRUCTURAL_CATEGORIES),
        "hungarian_eligibility": (
            "two exclusive candidate sets plus a complete pairwise cost matrix"
        ),
        "gold_used_for_classification": False,
    }
    return {
        "files": file_evidence,
        "classification_contract": contract,
        "classification_contract_sha256": _sha256_bytes(
            _canonical_json_bytes(contract)
        ),
    }


@contextmanager
def vendored_legacy_bind() -> Iterator[Callable[[str], Any]]:
    """Load the exact vendored binder without consulting ambient packages."""

    with _solver_from_checkout(VENDOR_ROOT) as solver:
        yield solver.bindings_mod.bind


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Atomically replace a JSON report, refusing an output symlink."""

    expanded = path.expanduser()
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    target = expanded.parent.resolve() / expanded.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise AuditError(f"refusing to replace output symlink: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            payload = (
                json.dumps(
                    report,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    indent=2,
                )
                + "\n"
            ).encode("utf-8")
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=DEFAULT_BENCHMARK_PATH,
        help="sealed complete GSM8K benchmark report",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="atomically replaced audit report",
    )
    parser.add_argument(
        "--allow-wrong",
        action="store_true",
        help="audit abstentions even when the baseline wrong gate is non-zero",
    )
    parser.add_argument(
        "--allow-evidence-drift",
        action="store_true",
        help="do not require the committed evidence-count seal",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    benchmark_path = args.benchmark_json.expanduser().resolve()
    benchmark = load_benchmark_report(benchmark_path)
    provenance = current_provenance(benchmark_path)
    with vendored_legacy_bind() as legacy_bind:
        report = build_audit_report(
            benchmark,
            legacy_bind=legacy_bind,
            structural_parse=parse_structural_problem,
            provenance=provenance,
            require_zero_wrong=not args.allow_wrong,
            require_current_evidence=not args.allow_evidence_drift,
        )
    write_report(args.output_json, report)
    summary = {key: value for key, value in report.items() if key != "items"}
    summary["items_in_json"] = len(report["items"])
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"geschrieben: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
