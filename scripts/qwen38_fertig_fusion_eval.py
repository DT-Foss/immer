#!/usr/bin/env python3
"""Fuse exact FERTIG certificates with completed Qwen3.8 draft verification.

The decision order is intentionally asymmetric:

1. an exact, full-rank Fraction/RREF certificate wins;
2. otherwise a stopped Q3 answer may be surfaced as ``model_verified`` only
   when the streamed BF16 checkpoint accepted its complete content and EOS;
3. every other row abstains.

Gold answers are read only after the decision is fixed and are used solely for
evaluation.  Model agreement is never renamed to mathematical correctness.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from immer.cognition.fertig.arithmetic_ir import SolveStatus, solve
from immer.cognition.fertig.structural import parse_structural_problem


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DRAFTS = ROOT / "results" / "qwen38_beast_baseline.json"
DEFAULT_VERIFICATION = (
    ROOT / "artifacts" / "private" / "qwen3.8-fertig-draft-verify" / "result-off.json"
)
DEFAULT_OUTPUT = ROOT / "results" / "qwen38_fertig_fusion.json"
RESULT_SCHEMA = "immer.qwen3.8-fertig-fusion/v1"
BASELINE_SCHEMA = "immer.qwen-local-fertig-baseline/v1"
VERIFICATION_SCHEMA = "immer.qwen3.8-fertig-draft-verification/v1"
CANONICAL_ANSWER_PREFIX_TOKENS = 2  # ``####`` followed by one space.


class CliError(RuntimeError):
    """The fusion evidence does not satisfy its fail-closed contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", default=str(DEFAULT_DRAFTS))
    parser.add_argument("--verification", default=str(DEFAULT_VERIFICATION))
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


def _indexed_items(
    document: Mapping[str, Any], label: str
) -> dict[str, Mapping[str, Any]]:
    raw_items = document.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise CliError(f"{label} has no items")
    indexed: dict[str, Mapping[str, Any]] = {}
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            raise CliError(f"{label} contains a non-object item")
        item_id = raw.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise CliError(f"{label} item has no item_id")
        if item_id in indexed:
            raise CliError(f"{label} contains duplicate item_id {item_id}")
        indexed[item_id] = raw
    return indexed


def _canonical_numeric(value: Any) -> str:
    raw = str(value).strip().replace(",", "")
    try:
        if raw.count("/") == 1:
            numerator, denominator = raw.split("/", 1)
            left = Decimal(numerator.strip())
            right = Decimal(denominator.strip())
            if not left.is_finite() or not right.is_finite() or right == 0:
                raise ValueError("fraction must be finite with a non-zero denominator")
            number = Fraction(left) / Fraction(right)
        elif "/" not in raw:
            decimal = Decimal(raw)
            if not decimal.is_finite():
                raise ValueError("number must be finite")
            number = Fraction(decimal)
        else:
            raise ValueError("answer has more than one fraction separator")
    except (InvalidOperation, ValueError, ZeroDivisionError) as exc:
        raise CliError(f"invalid numeric answer: {value!r}") from exc
    return _format_fraction(number)


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    denominator = value.denominator
    powers_of_two = 0
    while denominator % 2 == 0:
        denominator //= 2
        powers_of_two += 1
    powers_of_five = 0
    while denominator % 5 == 0:
        denominator //= 5
        powers_of_five += 1
    if denominator != 1:
        return f"{value.numerator}/{value.denominator}"
    places = max(powers_of_two, powers_of_five)
    scaled = value.numerator
    scaled *= 2 ** (places - powers_of_two)
    scaled *= 5 ** (places - powers_of_five)
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).rjust(places + 1, "0")
    rendered = f"{sign}{digits[:-places]}.{digits[-places:]}"
    return rendered.rstrip("0").rstrip(".")


def _exact_evidence(question: str) -> dict[str, Any]:
    parsed = parse_structural_problem(question)
    evidence: dict[str, Any] = {
        "parse_status": parsed.status.value,
        "parse_reason": parsed.reason,
        "solve_status": None,
        "answer": None,
        "certificate": None,
    }
    if not parsed.ok:
        return evidence
    assert parsed.problem is not None
    solution = solve(parsed.problem)
    evidence["solve_status"] = solution.status.value
    if solution.status is not SolveStatus.UNIQUE:
        return evidence
    if (
        solution.target_value is None
        or solution.certificate is None
        or not solution.certificate.verified
    ):
        raise CliError("unique arithmetic solution lacks a verified certificate")
    certificate = solution.certificate
    evidence["answer"] = _format_fraction(solution.target_value)
    evidence["certificate"] = {
        "rank": certificate.rank,
        "variable_count": certificate.variable_count,
        "equation_count": certificate.equation_count,
        "zero_residuals": all(row.value == 0 for row in certificate.residuals),
    }
    return evidence


def fuse_documents(
    drafts: Mapping[str, Any], verification: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a gold-label-independent decision followed by evaluation fields."""

    if drafts.get("schema") != BASELINE_SCHEMA:
        raise CliError("unexpected Q3 baseline schema")
    if verification.get("schema") != VERIFICATION_SCHEMA:
        raise CliError("unexpected BF16 verification schema")
    if verification.get("status") != "complete":
        raise CliError("BF16 verification is not complete")

    draft_rows = _indexed_items(drafts, "Q3 baseline")
    verification_rows = _indexed_items(verification, "BF16 verification")
    if set(draft_rows) != set(verification_rows):
        raise CliError("Q3 and BF16 item sets differ")

    rows: list[dict[str, Any]] = []
    for item_id, verified in verification_rows.items():
        draft = draft_rows[item_id]
        question = verified.get("question")
        if not isinstance(question, str) or question != draft.get("question"):
            raise CliError(f"question mismatch for {item_id}")
        q3_answer_raw = draft.get("predicted")
        q3_answer = (
            _canonical_numeric(q3_answer_raw) if q3_answer_raw is not None else None
        )
        verified_answer_raw = verified.get("answer")
        verified_answer = (
            _canonical_numeric(verified_answer_raw)
            if verified_answer_raw is not None
            else None
        )
        if q3_answer != verified_answer or draft.get("text") != verified.get("text"):
            raise CliError(f"Q3 draft identity mismatch for {item_id}")

        raw_report = verified.get("verification")
        if not isinstance(raw_report, Mapping):
            raise CliError(f"missing verification report for {item_id}")
        fully_verified = raw_report.get("fully_verified")
        accepted_prefix = raw_report.get("accepted_prefix_length")
        if not isinstance(fully_verified, bool):
            raise CliError(f"invalid fully_verified flag for {item_id}")
        if (
            isinstance(accepted_prefix, bool)
            or not isinstance(accepted_prefix, int)
            or accepted_prefix < 0
        ):
            raise CliError(f"invalid accepted prefix for {item_id}")
        draft_token_ids = raw_report.get("draft_token_ids")
        if not isinstance(draft_token_ids, list):
            raise CliError(f"missing draft tokens for {item_id}")
        if fully_verified:
            if (
                accepted_prefix != len(draft_token_ids)
                or raw_report.get("eos_verified") is not True
            ):
                raise CliError(f"inconsistent full verification for {item_id}")
        elif raw_report.get("first_mismatch_index") is None:
            raise CliError(f"rejected draft has no mismatch for {item_id}")

        exact = _exact_evidence(question)
        decision: str
        answer: str | None
        accepted: bool
        if exact["answer"] is not None:
            decision = "exact_ir"
            answer = str(exact["answer"])
            accepted = True
        elif (
            fully_verified
            and q3_answer is not None
            and draft.get("finish_reason") == "stop"
        ):
            decision = "model_verified"
            answer = q3_answer
            accepted = True
        else:
            answer = None
            accepted = False
            decision = (
                "surface_quarantine"
                if not fully_verified
                and accepted_prefix < CANONICAL_ANSWER_PREFIX_TOKENS
                else "content_rejected"
            )

        # Labels enter only after the answer/abstention decision is immutable.
        gold = _canonical_numeric(verified.get("gold"))
        if gold != _canonical_numeric(draft.get("gold")):
            raise CliError(f"gold mismatch for {item_id}")
        rows.append(
            {
                "item_id": item_id,
                "gold": gold,
                "q3_answer": q3_answer,
                "q3_finish_reason": draft.get("finish_reason"),
                "bf16_fully_verified": fully_verified,
                "bf16_accepted_prefix_length": accepted_prefix,
                "exact": exact,
                "decision": decision,
                "accepted": accepted,
                "answer": answer,
                "correct": answer == gold if accepted else None,
            }
        )

    total = len(rows)
    accepted_rows = [row for row in rows if row["accepted"]]
    correct = sum(row["correct"] is True for row in rows)
    wrong = sum(row["correct"] is False for row in rows)
    counts = {
        decision: sum(row["decision"] == decision for row in rows)
        for decision in (
            "exact_ir",
            "model_verified",
            "content_rejected",
            "surface_quarantine",
        )
    }
    raw_source = verification.get("source")
    source = (
        {
            key: raw_source[key]
            for key in ("checkpoint", "revision")
            if isinstance(raw_source.get(key), str)
        }
        if isinstance(raw_source, Mapping)
        else {}
    )
    source.update(
        {
            "draft_schema": BASELINE_SCHEMA,
            "verification_schema": VERIFICATION_SCHEMA,
        }
    )
    return {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "source": source,
        "protocol": {
            "decision_order": ["exact_ir", "model_verified", "abstain"],
            "gold_used_for_decisions": False,
            "model_verified_is_exact": False,
            "canonical_answer_prefix_tokens": CANONICAL_ANSWER_PREFIX_TOKENS,
        },
        "items": rows,
        "summary": {
            "total": total,
            **counts,
            "answered": len(accepted_rows),
            "abstained": total - len(accepted_rows),
            "correct": correct,
            "wrong": wrong,
            "coverage": len(accepted_rows) / total,
            "overall_accuracy": correct / total,
            "answered_accuracy": correct / len(accepted_rows)
            if accepted_rows
            else None,
        },
    }


def _atomic_write_json(path: str | Path, document: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        ).encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
    except (OSError, TypeError, ValueError) as exc:
        raise CliError(f"cannot prepare output: {destination}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        raise CliError(f"cannot write output: {destination}") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = fuse_documents(
            _read_json(args.drafts, "Q3 baseline"),
            _read_json(args.verification, "BF16 verification"),
        )
        output = _atomic_write_json(args.output, report)
    except CliError as exc:
        print(f"qwen38_fertig_fusion_eval: error: {exc}")
        return 2
    print(json.dumps({"output": str(output), **report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
