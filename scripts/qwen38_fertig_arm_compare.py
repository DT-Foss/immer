#!/usr/bin/env python3
"""Compare two completed Qwen3.8 FERTIG verification arms fail-closed.

The comparator admits only the same checkpoint, causal bundle, cohort, tokenized
prompts, and candidate drafts.  It keeps correct-draft verification separate
from wrong-draft agreement so a graft cannot win by confidently preserving more
wrong answers.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = ROOT / "artifacts" / "private" / "qwen3.8-fertig-local"
DEFAULT_OFF = DEFAULT_RUN_DIR / "result-off.json"
DEFAULT_CANDIDATE = DEFAULT_RUN_DIR / "result-stable-crsa-L27-a001.json"
DEFAULT_OUTPUT = ROOT / "results" / "qwen38_fertig_arm_compare.json"
VERIFICATION_SCHEMA = "immer.qwen3.8-fertig-draft-verification/v1"
RESULT_SCHEMA = "immer.qwen3.8-fertig-arm-compare/v1"


class CliError(RuntimeError):
    """Arm evidence violates the paired-comparison contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", default=str(DEFAULT_OFF))
    parser.add_argument("--candidate", default=str(DEFAULT_CANDIDATE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CliError(f"cannot read {label}: {source}") from exc
    if not isinstance(document, dict):
        raise CliError(f"{label} root must be an object")
    return document


def _canonical_digest(document: Mapping[str, Any]) -> str:
    try:
        payload = json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CliError("comparison report is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: str | Path, document: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CliError(f"cannot prepare comparison path: {destination}") from exc
    body = (
        json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except OSError as exc:
        raise CliError(f"cannot write comparison: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CliError(f"{label} must be an object")
    return value


def _finite_seconds(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CliError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise CliError(f"{label} must be finite and non-negative")
    return result


def _rows(document: Mapping[str, Any], label: str) -> list[Mapping[str, Any]]:
    raw = document.get("items")
    if not isinstance(raw, list) or not raw:
        raise CliError(f"{label} has no items")
    rows: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for row in raw:
        if not isinstance(row, Mapping):
            raise CliError(f"{label} contains a non-object row")
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise CliError(f"{label} item IDs are invalid or duplicated")
        seen.add(item_id)
        rows.append(row)
    return rows


def _identity_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(document.get("source"), "source")
    verification = _mapping(source.get("verification"), "source verification")
    protocol = _mapping(document.get("protocol"), "protocol")
    evidence = _mapping(document.get("evidence"), "evidence")
    identity_fields = (
        "kind",
        "checkpoint_bytes",
        "graph_revision",
        "layout_fingerprint",
        "manifest_sha256",
        "shards",
        "shards_sha256",
        "tensor_bindings",
        "weights_layout",
    )
    protocol_fields = (
        "batch_size",
        "padding",
        "padding_token_id",
        "eos_token_id",
        "accepted_eos_token_ids",
        "weight_order",
        "dynamic_cohort",
        "item_ids",
    )
    evidence_fields = (
        "batch_size",
        "draft_lengths",
        "eos_token_id",
        "fixed_model_batch",
        "padded_sequence_length",
        "prompt_lengths",
        "right_prefix_mask",
        "verification_rows",
    )
    return {
        "checkpoint": source.get("checkpoint"),
        "revision": source.get("revision"),
        "bundle": {name: verification.get(name) for name in identity_fields},
        "protocol": {name: protocol.get(name) for name in protocol_fields},
        "evidence": {name: evidence.get(name) for name in evidence_fields},
    }


def _item_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "item_id",
        "question",
        "gold",
        "text",
        "answer",
        "candidate_correct",
        "candidate_status",
        "finish_reason",
        "prompt_token_ids",
        "draft_token_ids",
    )
    return {name: row.get(name) for name in fields}


def _verification(row: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    report = _mapping(row.get("verification"), f"{label} verification")
    for name in ("draft_verified", "fully_verified"):
        if not isinstance(report.get(name), bool):
            raise CliError(f"{label} has invalid {name}")
    accepted = report.get("accepted_prefix_length")
    draft_ids = report.get("draft_token_ids")
    if (
        isinstance(accepted, bool)
        or not isinstance(accepted, int)
        or accepted < 0
        or not isinstance(draft_ids, list)
        or accepted > len(draft_ids)
    ):
        raise CliError(f"{label} has invalid accepted-prefix evidence")
    if report.get("fully_verified") and (
        accepted != len(draft_ids) or report.get("eos_verified") is not True
    ):
        raise CliError(f"{label} has inconsistent full verification")
    if draft_ids != row.get("draft_token_ids"):
        raise CliError(f"{label} draft tokens differ from the admitted candidate")
    return report


def _quality_metrics(
    document: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Any]:
    reports = [_verification(row, f"{label} {row['item_id']}") for row in rows]
    total = len(rows)
    metrics = {
        "total": total,
        "candidate_correct": sum(row.get("candidate_correct") is True for row in rows),
        "candidate_incomplete": sum(
            row.get("candidate_correct") is None for row in rows
        ),
        "content_verified": sum(bool(report["draft_verified"]) for report in reports),
        "eos_verified": sum(report.get("eos_verified") is True for report in reports),
        "fully_verified": sum(bool(report["fully_verified"]) for report in reports),
        "verified_correct": sum(
            bool(report["fully_verified"] and row.get("candidate_correct") is True)
            for row, report in zip(rows, reports, strict=True)
        ),
    }
    summary = _mapping(document.get("summary"), f"{label} summary")
    for name in (
        "total",
        "candidate_correct",
        "content_verified",
        "eos_verified",
        "fully_verified",
        "verified_correct",
    ):
        if summary.get(name) != metrics[name]:
            raise CliError(f"{label} summary field {name} is inconsistent")
    if (
        "candidate_incomplete" in summary
        and summary.get("candidate_incomplete") != metrics["candidate_incomplete"]
    ):
        raise CliError(f"{label} summary field candidate_incomplete is inconsistent")
    return metrics


def _arm_metrics(
    document: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], label: str
) -> dict[str, Any]:
    evidence = _mapping(document.get("evidence"), "evidence")
    source = _mapping(document.get("source"), "source")
    source_verification = _mapping(source.get("verification"), "source verification")
    model_seconds = _finite_seconds(evidence.get("seconds"), "model seconds")
    verify_seconds = _finite_seconds(
        source_verification.get("seconds"), "bundle verification seconds"
    )
    return {
        **_quality_metrics(document, rows, label),
        "model_seconds": model_seconds,
        "bundle_verification_seconds": verify_seconds,
        "end_to_end_seconds": model_seconds + verify_seconds,
        "source_body_bytes": evidence.get("source_body_bytes"),
        "layer_calls": evidence.get("layer_calls"),
        "layer_retry_count": evidence.get("layer_retry_count"),
        "head_scans": evidence.get("head_scans"),
        "head_retry_count": evidence.get("head_retry_count"),
    }


def compare_arms(
    off: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    for label, document in (("off", off), ("candidate", candidate)):
        if document.get("schema") != VERIFICATION_SCHEMA:
            raise CliError(f"{label} verification schema mismatch")
        if document.get("status") != "complete":
            raise CliError(f"{label} arm is incomplete")
    off_protocol = _mapping(off.get("protocol"), "off protocol")
    candidate_protocol = _mapping(candidate.get("protocol"), "candidate protocol")
    if off_protocol.get("mode") != "off":
        raise CliError("baseline arm must use off mode")
    if candidate_protocol.get("mode") != "stable-crsa":
        raise CliError("candidate arm must use stable-crsa mode")
    layer = candidate_protocol.get("graft_layer")
    alpha = candidate_protocol.get("graft_alpha")
    if (
        isinstance(layer, bool)
        or not isinstance(layer, int)
        or layer < 0
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0 <= float(alpha) <= 1
    ):
        raise CliError("candidate graft contract is invalid")

    identity = _identity_projection(off)
    if identity != _identity_projection(candidate):
        raise CliError("arm checkpoint, cohort, or execution identity differs")
    off_rows = _rows(off, "off arm")
    candidate_rows = _rows(candidate, "candidate arm")
    if len(off_rows) != len(candidate_rows):
        raise CliError("arm item counts differ")

    changes: list[dict[str, Any]] = []
    counts = {
        "correct_promotions": 0,
        "correct_regressions": 0,
        "wrong_agreement_promotions": 0,
        "wrong_agreement_regressions": 0,
        "incomplete_promotions": 0,
        "incomplete_regressions": 0,
        "unchanged": 0,
    }
    prefix_delta = 0
    for off_row, candidate_row in zip(off_rows, candidate_rows, strict=True):
        if _item_identity(off_row) != _item_identity(candidate_row):
            raise CliError("arm item or token identity differs")
        item_id = str(off_row["item_id"])
        off_report = _verification(off_row, f"off {item_id}")
        candidate_report = _verification(candidate_row, f"candidate {item_id}")
        if off_report.get("draft_token_ids") != candidate_report.get("draft_token_ids"):
            raise CliError(f"arm draft verification identity differs for {item_id}")
        off_full = bool(off_report["fully_verified"])
        candidate_full = bool(candidate_report["fully_verified"])
        candidate_correct = off_row.get("candidate_correct")
        if (
            candidate_correct is not True
            and candidate_correct is not False
            and candidate_correct is not None
        ):
            raise CliError(f"invalid candidate correctness for {item_id}")
        if off_full == candidate_full:
            counts["unchanged"] += 1
        elif candidate_correct is True:
            counts[
                "correct_promotions" if candidate_full else "correct_regressions"
            ] += 1
        elif candidate_correct is False:
            counts[
                "wrong_agreement_promotions"
                if candidate_full
                else "wrong_agreement_regressions"
            ] += 1
        else:
            counts[
                "incomplete_promotions" if candidate_full else "incomplete_regressions"
            ] += 1
        off_prefix = int(off_report["accepted_prefix_length"])
        candidate_prefix = int(candidate_report["accepted_prefix_length"])
        prefix_delta += candidate_prefix - off_prefix
        if off_full != candidate_full or off_prefix != candidate_prefix:
            changes.append(
                {
                    "item_id": item_id,
                    "candidate_correct": candidate_correct,
                    "off_fully_verified": off_full,
                    "candidate_fully_verified": candidate_full,
                    "off_accepted_prefix": off_prefix,
                    "candidate_accepted_prefix": candidate_prefix,
                }
            )

    if counts["wrong_agreement_promotions"]:
        verdict = "unsafe_regression"
    elif counts["correct_regressions"] > counts["correct_promotions"]:
        verdict = "regressed"
    elif counts["correct_promotions"] > counts["correct_regressions"]:
        verdict = "improved"
    elif counts["wrong_agreement_regressions"]:
        verdict = "safer"
    else:
        verdict = "neutral"

    off_metrics = _arm_metrics(off, off_rows, "off")
    candidate_metrics = _arm_metrics(candidate, candidate_rows, "candidate")
    comparison: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "identity": identity,
        "candidate_graft": {
            "mode": "stable-crsa",
            "layer": layer,
            "alpha": float(alpha),
        },
        "contract": {
            "same_checkpoint_bundle_cohort_and_tokens": True,
            "agreement_evidence_is_not_an_exact_certificate": True,
            "wrong_draft_agreement_cannot_count_as_quality": True,
        },
        "off": off_metrics,
        "candidate": candidate_metrics,
        "comparison": {
            **counts,
            "accepted_prefix_delta": prefix_delta,
            "model_speed_ratio_off_over_candidate": (
                off_metrics["model_seconds"] / candidate_metrics["model_seconds"]
                if candidate_metrics["model_seconds"]
                else None
            ),
            "model_runtime_delta_seconds": (
                candidate_metrics["model_seconds"] - off_metrics["model_seconds"]
            ),
            "verdict": verdict,
        },
        "changed_items": changes,
    }
    comparison["report_sha256"] = _canonical_digest(comparison)
    return comparison


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = compare_arms(
            _read_json(args.off, "off arm"),
            _read_json(args.candidate, "candidate arm"),
        )
        output = _atomic_write_json(args.output, report)
    except CliError as exc:
        print(f"qwen38_fertig_arm_compare: error: {exc}")
        return 2
    print(
        json.dumps(
            {"output": str(output), **report["comparison"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
