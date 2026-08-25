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

SCHEMA = "immer.fertig-abstention-audit/v2"
REPORT_REVISION = 3
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
    "exact_recovery": 42,
    "numeric_pronoun_ambiguous": 80,
    "question_pronoun_ambiguous": 5,
    "numeric_clause_unsupported": 93,
    "target_unsupported": 8,
    "relation_ambiguous": 1,
    "relation_unsupported": 1,
    "relation_invalid": 0,
}
CURRENT_SCOPE_COUNTS = {
    "exact_recovery_scope": 42,
    "potential_coreference_scope": 85,
    "grammar_scope": 103,
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


def _validate_current_evidence(report: Mapping[str, Any]) -> None:
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
        audited_items.append(
            {
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
        )

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
