#!/usr/bin/env python3
"""Fuse exact FERTIG certificates with streamed DeepSeek-V4 verification.

Decision order is fail-closed and independent of benchmark labels:

1. semantically under-specified targets are quarantined;
2. an exact, full-rank Fraction/RREF certificate wins;
3. otherwise a DeepSeek API draft is surfaced as ``model_verified`` only
   when the pinned local BF16 checkpoint accepts every token and EOS;
4. every other row abstains.

Gold answers enter only after the decision is fixed and are used solely for
evaluation.  Greedy agreement is evidence of checkpoint behavior, not an
exact mathematical certificate.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from immer.cognition.fertig.arithmetic_ir import SolveStatus, solve
from immer.cognition.fertig.structural import parse_structural_problem
from immer.runtimes.deepseek_v4.benchmark import extract_gsm8k_answer


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN_DIR = ROOT / "artifacts" / "private" / "deepseek-v4-fertig-draft-verify"
DEFAULT_DRAFTS = DEFAULT_RUN_DIR / "drafts.json"
DEFAULT_VERIFICATION = DEFAULT_RUN_DIR / "result-off.json"
DEFAULT_BENCHMARK = ROOT / "results" / "bench_gsm8k.json"
DEFAULT_OUTPUT = ROOT / "results" / "deepseek-v4-fertig-fusion.json"
RESULT_SCHEMA = "immer.deepseek-v4-fertig-fusion/v1"
DRAFT_SCHEMA = "immer.deepseek-v4-fertig-drafts/v1"
VERIFICATION_SCHEMA = "immer.deepseek-v4-fertig-draft-verification/v1"
BENCHMARK_SCHEMA = "immer.benchmark/v1"
OFFICIAL_SOURCE = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
API_MODEL = "deepseek-v4-flash"
DRAFT_MAX_TOKENS = 256

_AVERAGE_LOSS_QUERY = re.compile(
    r"(?:\b(?:lose|lost)\b.{0,32}\bon average\b|"
    r"\baverage\b.{0,32}\b(?:loss|lose|lost)\b)",
    re.IGNORECASE,
)
_GAIN_EVENT = re.compile(r"\b(?:win|wins|won|gain|gains|gained)\b", re.IGNORECASE)
_LOSS_EVENT = re.compile(r"\b(?:lose|loses|lost|loss)\b", re.IGNORECASE)


class CliError(RuntimeError):
    """Fusion evidence failed its closed admission contract."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", default=str(DEFAULT_DRAFTS))
    parser.add_argument("--verification", default=str(DEFAULT_VERIFICATION))
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
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


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CliError("benchmark items are not canonically serializable") from exc


def _framed_digest(values: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = _canonical_json_bytes(value)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    digest.update(len(values).to_bytes(8, "big"))
    return digest.hexdigest()


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


def _optional_numeric(value: Any) -> str | None:
    return None if value is None else _canonical_numeric(value)


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
    zero_residuals = all(row.value == 0 for row in certificate.residuals)
    if (
        certificate.rank != certificate.variable_count
        or certificate.equation_count < certificate.variable_count
        or not zero_residuals
    ):
        raise CliError("verified arithmetic certificate is not full-rank and exact")
    evidence["answer"] = _format_fraction(solution.target_value)
    evidence["certificate"] = {
        "rank": certificate.rank,
        "variable_count": certificate.variable_count,
        "equation_count": certificate.equation_count,
        "zero_residuals": zero_residuals,
    }
    return evidence


def _semantic_ambiguity(question: str) -> dict[str, Any]:
    """Detect an average-loss target with mixed signed events.

    Such wording does not determine whether "average loss" means the mean
    signed outcome, gross losses divided by all events, or losses divided by
    losing events.  A model agreeing with one interpretation cannot certify
    which convention the benchmark author intended.
    """

    context, separator, query = question.rpartition(".")
    if not separator:
        context, query = "", question
    mixed_signed_events = bool(
        _GAIN_EVENT.search(context) and _LOSS_EVENT.search(context)
    )
    average_loss_target = bool(_AVERAGE_LOSS_QUERY.search(query))
    ambiguous = mixed_signed_events and average_loss_target
    return {
        "ambiguous": ambiguous,
        "reason": (
            "mixed gain/loss events leave the denominator and sign convention "
            "of the requested average loss under-specified"
            if ambiguous
            else None
        ),
    }


def _require_draft_protocol(drafts: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol = drafts.get("protocol")
    expected = {
        "model": API_MODEL,
        "assistant_prefix": "Answer:",
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": DRAFT_MAX_TOKENS,
        "logprobs": True,
    }
    if not isinstance(protocol, Mapping) or dict(protocol) != expected:
        raise CliError("DeepSeek API draft protocol is not the pinned contract")
    return protocol


def _require_protocol(verification: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol = verification.get("protocol")
    if not isinstance(protocol, Mapping):
        raise CliError("verification protocol is missing")
    expected = {
        "checkpoint": OFFICIAL_SOURCE,
        "revision": OFFICIAL_REVISION,
        "api_model": API_MODEL,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise CliError(f"verification {key} does not match the pinned source")
    mode = protocol.get("mode")
    if mode not in {"off", "stable-crsa"}:
        raise CliError("verification mode is unsupported")
    layer = protocol.get("graft_layer")
    alpha = protocol.get("graft_alpha")
    if mode == "off":
        if layer is not None or alpha is not None:
            raise CliError("off verification must not carry graft parameters")
    elif (
        isinstance(layer, bool)
        or not isinstance(layer, int)
        or not 0 <= layer < 43
        or isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0 < float(alpha) <= 1
    ):
        raise CliError("stable-crsa verification has invalid graft parameters")
    return protocol


def _require_benchmark(
    benchmark: Mapping[str, Any], selected: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Mapping[str, Any]], str]:
    if benchmark.get("schema") != BENCHMARK_SCHEMA:
        raise CliError("unexpected GSM8K benchmark schema")
    raw_items = benchmark.get("items")
    if not isinstance(raw_items, list):
        raise CliError("canonical GSM8K benchmark has no item list")
    indexed = _indexed_items(benchmark, "canonical GSM8K benchmark")
    digest = benchmark.get("items_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise CliError("canonical GSM8K benchmark has no item digest")
    if _framed_digest(raw_items) != digest:
        raise CliError("canonical GSM8K benchmark item digest does not verify")
    canonical: dict[str, Mapping[str, Any]] = {}
    for item_id, draft in selected.items():
        row = indexed.get(item_id)
        if row is None:
            raise CliError(f"canonical GSM8K item is missing: {item_id}")
        if row.get("question") != draft.get("question"):
            raise CliError(f"canonical question mismatch for {item_id}")
        if _canonical_numeric(row.get("gold")) != _canonical_numeric(draft.get("gold")):
            raise CliError(f"canonical gold mismatch for {item_id}")
        canonical[item_id] = row
    return canonical, digest


def _validate_verification_summary(
    verification: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    summary = verification.get("summary")
    if not isinstance(summary, Mapping):
        raise CliError("verification summary is missing")
    expected = {
        "total": len(rows),
        "content_verified": sum(row.get("content_verified") is True for row in rows),
        "eos_verified": sum(row.get("eos_verified") is True for row in rows),
        "fully_verified": sum(row.get("fully_verified") is True for row in rows),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise CliError(f"verification summary field {key} is inconsistent")


def _validate_local_row(
    item_id: str,
    draft: Mapping[str, Any],
    verified: Mapping[str, Any],
) -> tuple[bool, str | None, str]:
    content = draft.get("content")
    token_ids = draft.get("token_ids")
    if not isinstance(content, str) or not isinstance(token_ids, list):
        raise CliError(f"invalid API draft for {item_id}")
    if draft.get("finish_reason") != "stop" or draft.get("response_model") != API_MODEL:
        raise CliError(f"API draft contract mismatch for {item_id}")
    api_content = "Answer:" + content
    if verified.get("api_content") != api_content:
        raise CliError(f"API content mismatch for {item_id}")
    if verified.get("api_response_model") != draft.get("response_model"):
        raise CliError(f"API model mismatch for {item_id}")
    if verified.get("api_system_fingerprint") != draft.get("system_fingerprint"):
        raise CliError(f"API fingerprint mismatch for {item_id}")

    extracted_api = _optional_numeric(extract_gsm8k_answer(api_content))
    reported_api = _optional_numeric(verified.get("api_answer"))
    if extracted_api != reported_api:
        raise CliError(f"API answer mismatch for {item_id}")

    raw = verified.get("verification")
    if not isinstance(raw, Mapping):
        raise CliError(f"local verification row is missing for {item_id}")
    if raw.get("draft_token_ids") != token_ids:
        raise CliError(f"local draft tokens differ for {item_id}")
    accepted = raw.get("accepted_prefix_length")
    if (
        isinstance(accepted, bool)
        or not isinstance(accepted, int)
        or not 0 <= accepted <= len(token_ids)
    ):
        raise CliError(f"invalid accepted prefix for {item_id}")
    draft_verified = raw.get("draft_verified")
    fully_verified = raw.get("fully_verified")
    if not isinstance(draft_verified, bool) or not isinstance(fully_verified, bool):
        raise CliError(f"invalid local verification flags for {item_id}")
    mismatch = raw.get("first_mismatch_index")
    eos_verified = raw.get("eos_verified")
    if (
        verified.get("content_verified") is not draft_verified
        or verified.get("fully_verified") is not fully_verified
        or verified.get("eos_verified") != eos_verified
    ):
        raise CliError(f"top-level verification flags differ for {item_id}")
    if fully_verified:
        if (
            not draft_verified
            or eos_verified is not True
            or accepted != len(token_ids)
            or mismatch is not None
        ):
            raise CliError(f"inconsistent full verification for {item_id}")
    elif draft_verified:
        if (
            accepted != len(token_ids)
            or eos_verified is not False
            or mismatch != len(token_ids)
        ):
            raise CliError(f"inconsistent EOS rejection for {item_id}")
    elif accepted >= len(token_ids) or eos_verified is not None or mismatch != accepted:
        raise CliError(f"inconsistent draft rejection for {item_id}")

    local_content = verified.get("locally_fully_verified_content")
    local_answer = _optional_numeric(verified.get("local_answer"))
    if fully_verified:
        if local_content != api_content or local_answer != extracted_api:
            raise CliError(f"fully verified local content differs for {item_id}")
    elif local_content is not None or local_answer is not None:
        raise CliError(f"rejected draft exposes local content for {item_id}")
    return (
        fully_verified,
        local_answer,
        ("termination_rejected" if draft_verified else "draft_rejected"),
    )


def fuse_documents(
    drafts: Mapping[str, Any],
    verification: Mapping[str, Any],
    benchmark: Mapping[str, Any],
) -> dict[str, Any]:
    """Return gold-independent decisions followed by evaluation fields."""

    if drafts.get("schema") != DRAFT_SCHEMA:
        raise CliError("unexpected DeepSeek draft schema")
    _require_draft_protocol(drafts)
    if verification.get("schema") != VERIFICATION_SCHEMA:
        raise CliError("unexpected DeepSeek verification schema")
    if verification.get("status") != "complete":
        raise CliError("DeepSeek verification is not complete")
    protocol = _require_protocol(verification)

    draft_rows = _indexed_items(drafts, "DeepSeek API drafts")
    verification_rows = _indexed_items(verification, "DeepSeek local verification")
    if tuple(draft_rows) != tuple(verification_rows):
        raise CliError("draft and verification item order differs")
    canonical_rows, benchmark_digest = _require_benchmark(benchmark, draft_rows)
    _validate_verification_summary(verification, tuple(verification_rows.values()))

    rows: list[dict[str, Any]] = []
    for item_id, draft in draft_rows.items():
        verified = verification_rows[item_id]
        question = canonical_rows[item_id].get("question")
        if not isinstance(question, str) or not question.strip():
            raise CliError(f"question is missing for {item_id}")
        fully_verified, model_answer, rejection = _validate_local_row(
            item_id, draft, verified
        )
        semantic = _semantic_ambiguity(question)
        exact = _exact_evidence(question)
        if semantic["ambiguous"]:
            decision = "semantic_ambiguity"
            answer = None
            accepted = False
        elif exact["answer"] is not None:
            decision = "exact_ir"
            answer = str(exact["answer"])
            accepted = True
        elif fully_verified and model_answer is not None:
            decision = "model_verified"
            answer = model_answer
            accepted = True
        else:
            decision = "non_numeric_model_output" if fully_verified else rejection
            answer = None
            accepted = False

        # Labels enter only after answer/abstention is immutable.
        gold = _canonical_numeric(draft.get("gold"))
        if gold != _canonical_numeric(verified.get("gold")):
            raise CliError(f"gold mismatch for {item_id}")
        rows.append(
            {
                "item_id": item_id,
                "gold": gold,
                "api_answer": _optional_numeric(verified.get("api_answer")),
                "bf16_fully_verified": fully_verified,
                "bf16_accepted_prefix_length": verified["verification"][
                    "accepted_prefix_length"
                ],
                "semantic": semantic,
                "exact": exact,
                "decision": decision,
                "accepted": accepted,
                "answer": answer,
                "correct": answer == gold if accepted else None,
            }
        )

    decisions = (
        "semantic_ambiguity",
        "exact_ir",
        "model_verified",
        "draft_rejected",
        "termination_rejected",
        "non_numeric_model_output",
    )
    accepted_rows = [row for row in rows if row["accepted"]]
    correct = sum(row["correct"] is True for row in rows)
    wrong = sum(row["correct"] is False for row in rows)
    total = len(rows)
    return {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "source": {
            "checkpoint": OFFICIAL_SOURCE,
            "revision": OFFICIAL_REVISION,
            "api_model": API_MODEL,
            "mode": protocol["mode"],
            "graft_layer": protocol.get("graft_layer"),
            "graft_alpha": protocol.get("graft_alpha"),
            "draft_schema": DRAFT_SCHEMA,
            "verification_schema": VERIFICATION_SCHEMA,
            "benchmark_schema": BENCHMARK_SCHEMA,
            "benchmark_items_sha256": benchmark_digest,
        },
        "protocol": {
            "decision_order": [
                "semantic_ambiguity",
                "exact_ir",
                "model_verified",
                "abstain",
            ],
            "gold_used_for_decisions": False,
            "model_verified_is_exact": False,
            "requires_complete_draft_and_eos": True,
        },
        "items": rows,
        "summary": {
            "total": total,
            **{
                decision: sum(row["decision"] == decision for row in rows)
                for decision in decisions
            },
            "answered": len(accepted_rows),
            "abstained": total - len(accepted_rows),
            "correct": correct,
            "wrong": wrong,
            "coverage": len(accepted_rows) / total,
            "overall_accuracy": correct / total,
            "answered_accuracy": (
                correct / len(accepted_rows) if accepted_rows else None
            ),
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
            _read_json(args.drafts, "DeepSeek API drafts"),
            _read_json(args.verification, "DeepSeek local verification"),
            _read_json(args.benchmark, "canonical GSM8K benchmark"),
        )
        output = _atomic_write_json(args.output, report)
    except CliError as exc:
        print(f"deepseek_v4_fertig_fusion_eval: error: {exc}")
        return 2
    print(json.dumps({"output": str(output), **report["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
