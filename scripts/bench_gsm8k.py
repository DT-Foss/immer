"""bench_gsm8k.py — FERTIG gegen die Großen, reproduzierbar gemessen.

Läuft den vendorten FERTIG-Solver über den GSM8K-Test-Split und berichtet
correct / abstained / incorrect / error. Der Architektur-Claim bleibt streng:
``wrong`` ist ``incorrect + errors`` und muss null sein. Der vollständige
Report wird atomar nach ``results/bench_gsm8k.json`` geschrieben.

    PYTHONPATH=src python3 scripts/bench_gsm8k.py [--limit 100]
    PYTHONPATH=src python3 scripts/bench_gsm8k.py \
        --failures-jsonl results/bench_gsm8k_failures.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import immer.cognition.fertig as fertig_package
from immer.cognition.fertig import FertigSolver
from immer.contracts import ExecutionStatus, Request

ROOT = Path(__file__).resolve().parent.parent
EVAL_PATH = ROOT / "evals" / "gsm8k_test.parquet"
RESULTS_DIR = ROOT / "results"
DEFAULT_OUTPUT_PATH = RESULTS_DIR / "bench_gsm8k.json"
SCHEMA = "immer.benchmark/v1"
REPORT_REVISION = 3


def gold_number(answer_field: str) -> float | None:
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    if not match:
        return None
    return float(match.group(1).replace(",", ""))


def solver_number(output: Any | None) -> float | None:
    if output is None:
        return None
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", str(output))
    if not numbers:
        return None
    return float(numbers[-1].replace(",", ""))


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _framed_digest(values: Iterable[Any]) -> str:
    digest = hashlib.sha256()
    count = 0
    for value in values:
        encoded = _canonical_json_bytes(value)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        count += 1
    digest.update(count.to_bytes(8, "big"))
    return digest.hexdigest()


def records_digest(records: Iterable[Mapping[str, str]]) -> str:
    """Hash ordered logical rows with length framing (not parquet encoding)."""

    return _framed_digest(
        {
            "answer": str(record["answer"]),
            "question": str(record["question"]),
        }
        for record in records
    )


def deterministic_item_id(index: int, question: str, answer: str) -> str:
    """Return an order- and content-bound stable GSM8K item identifier."""

    if index < 0:
        raise ValueError("item index must be non-negative")
    content_sha256 = _sha256_bytes(
        _canonical_json_bytes({"answer": answer, "question": question})
    )
    return f"gsm8k-test-{index:04d}-{content_sha256[:16]}"


def _source_tree_digest(root: Path) -> tuple[str, int]:
    """Digest all Python sources that implement the vendored FERTIG solver."""

    digest = hashlib.sha256()
    paths = sorted(path for path in root.rglob("*.py") if path.is_file())
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    digest.update(len(paths).to_bytes(8, "big"))
    return digest.hexdigest(), len(paths)


def harness_provenance() -> dict[str, Any]:
    """Describe and digest the exact runner plus solver source implementation."""

    runner_sha256 = _sha256_file(Path(__file__).resolve())
    package_file = Path(fertig_package.__file__ or "").resolve()
    solver_root = package_file.parent
    solver_sha256, solver_source_files = _source_tree_digest(solver_root)
    digest_material = {
        "runner_sha256": runner_sha256,
        "solver_sha256": solver_sha256,
    }
    solver_name = (
        "immer.cognition.fertig.FertigSolver "
        "(guarded formulas->structural IR->bindings->semantic; "
        "optional explicit rules; math templates proposal-only), "
        "no neural net"
    )
    return {
        "name": "FERTIG GSM8K exact-math harness",
        "entrypoint": "scripts/bench_gsm8k.py",
        "sha256": _sha256_bytes(_canonical_json_bytes(digest_material)),
        "runner_sha256": runner_sha256,
        "solver": solver_name,
        "admission_policy": {
            "answer_engines": [
                "guarded_formula_certificates",
                "structural_fraction_rref",
                "bindings",
                "semantic",
                "explicit_rules_if_supplied",
            ],
            "proposal_only": ["math_templates"],
            "hard_gate": "incorrect + errors == 0",
        },
        "solver_source_files": solver_source_files,
        "solver_source_sha256": solver_sha256,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


def dataset_provenance(
    all_records: Sequence[Mapping[str, str]],
    selected_records: Sequence[Mapping[str, str]],
    *,
    path: Path,
    limit: int,
) -> dict[str, Any]:
    all_sha256 = records_digest(all_records)
    selected_sha256 = records_digest(selected_records)
    try:
        artifact = path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        artifact = str(path.resolve())
    return {
        "name": "openai/gsm8k",
        "configuration": "main",
        "split": "test",
        "artifact": artifact,
        "artifact_sha256": _sha256_file(path),
        "sha256": selected_sha256,
        "records_sha256": all_sha256,
        "selected_records_sha256": selected_sha256,
        "num_items": len(all_records),
        "selected_items": len(selected_records),
        "selection": {"method": "head", "limit": limit},
    }


def _reason(code: str, detail: Any | None = None) -> str:
    if detail is None or not str(detail).strip():
        return code
    return f"{code}: {str(detail).strip()}"


def evaluate_records(
    records: Sequence[Mapping[str, str]], solver: Any
) -> list[dict[str, Any]]:
    """Evaluate every selected row and retain a complete outcome partition."""

    items: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        question = str(record["question"])
        answer = str(record["answer"])
        gold = gold_number(answer)
        item: dict[str, Any] = {
            "item_id": deterministic_item_id(index, question, answer),
            "index": index,
            "status": "error",
            "question": question,
            "gold": gold,
            "predicted": None,
            "reason": "gold_answer_unparseable",
        }
        if gold is None:
            items.append(item)
            continue

        try:
            result = solver.handle(Request("exact_math", question))
            got = solver_number(result.output)
            item["predicted"] = got
            if result.status is ExecutionStatus.ABSTAINED:
                item["status"] = "abstained"
                item["reason"] = _reason("solver_abstained", result.reason)
            elif result.status is ExecutionStatus.ERROR:
                item["reason"] = _reason("solver_error", result.reason)
            elif result.status is not ExecutionStatus.OK:
                item["status"] = "incorrect"
                item["reason"] = _reason(
                    f"solver_status_{result.status.value}", result.reason
                )
            elif got is None:
                # Preserve the historical policy: a clean but unparseable result
                # is safe abstention, not a guessed answer.
                item["status"] = "abstained"
                item["reason"] = "solver_output_unparseable"
            elif abs(got - gold) < 1e-6:
                item["status"] = "correct"
                item["reason"] = "numeric_match"
            else:
                item["status"] = "incorrect"
                item["reason"] = "numeric_mismatch"
        except Exception as exc:  # a solver/harness crash is never a dropped row
            item["status"] = "error"
            item["reason"] = _reason("solver_exception", f"{type(exc).__name__}: {exc}")
        items.append(item)
    return items


def summarize_items(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    correct = sum(item["status"] == "correct" for item in items)
    abstained = sum(item["status"] == "abstained" for item in items)
    incorrect = sum(item["status"] == "incorrect" for item in items)
    errors = sum(item["status"] == "error" for item in items)
    if correct + abstained + incorrect + errors != len(items):
        raise ValueError("item outcomes do not form a complete partition")

    wrong = incorrect + errors
    attempted = correct + wrong
    scored = correct + incorrect
    return {
        "n": len(items),
        "correct": correct,
        "abstained": abstained,
        "incorrect": incorrect,
        "errors": errors,
        # Backward-compatible gate and attempted-accuracy semantics: runtime
        # errors counted as wrong before they gained their own category.
        "wrong": wrong,
        "wrong_must_be_zero": wrong == 0,
        "accuracy_on_attempted": round(correct / attempted, 4) if attempted else 0.0,
        "accuracy_on_scored_attempts": (round(correct / scored, 4) if scored else 0.0),
        "coverage": round(scored / len(items), 4) if items else 0.0,
    }


def build_report(
    items: Sequence[Mapping[str, Any]],
    *,
    seconds: float,
    dataset: Mapping[str, Any],
    harness: Mapping[str, Any],
) -> dict[str, Any]:
    summary = summarize_items(items)
    item_list = [dict(item) for item in items]
    return {
        "schema": SCHEMA,
        "report_revision": REPORT_REVISION,
        "benchmark": "GSM8K-test",
        **summary,
        "seconds": round(seconds, 1),
        "engine": harness["solver"],
        "dataset_sha256": dataset["sha256"],
        "harness_sha256": harness["sha256"],
        "items_sha256": _framed_digest(item_list),
        "provenance": {"dataset": dict(dataset), "harness": dict(harness)},
        "items": item_list,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
    )


def write_failures_jsonl(path: Path, items: Sequence[Mapping[str, Any]]) -> int:
    failures = [item for item in items if item["status"] != "correct"]
    lines = [
        json.dumps(
            item,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for item in failures
    ]
    _atomic_write_text(path, "".join(f"{line}\n" for line in lines))
    return len(failures)


def _load_records(path: Path) -> list[dict[str, str]]:
    import pandas as pd

    frame = pd.read_parquet(path)
    missing = {"question", "answer"} - set(frame.columns)
    if missing:
        raise ValueError(f"GSM8K artifact misses columns: {sorted(missing)}")
    return [
        {"question": str(row["question"]), "answer": str(row["answer"])}
        for _, row in frame.iterrows()
    ]


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=_nonnegative_int, default=0, help="0 = alle 1319"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="atomarer vollständiger JSON-Report",
    )
    parser.add_argument(
        "--failures-jsonl",
        type=Path,
        help="optionales atomares JSONL mit allen nicht-korrekten Items",
    )
    args = parser.parse_args(argv)

    all_records = _load_records(EVAL_PATH)
    selected_records = all_records[: args.limit] if args.limit else all_records
    dataset = dataset_provenance(
        all_records,
        selected_records,
        path=EVAL_PATH,
        limit=args.limit,
    )
    harness = harness_provenance()

    solver = FertigSolver()
    t0 = time.perf_counter()
    items = evaluate_records(selected_records, solver)
    seconds = time.perf_counter() - t0
    report = build_report(
        items,
        seconds=seconds,
        dataset=dataset,
        harness=harness,
    )
    write_report(args.output_json, report)
    if args.failures_jsonl is not None:
        write_failures_jsonl(args.failures_jsonl, items)

    stdout_report = {key: value for key, value in report.items() if key != "items"}
    stdout_report["items_in_json"] = len(items)
    print(json.dumps(stdout_report, ensure_ascii=False, indent=2))
    print(f"geschrieben: {args.output_json}")
    if args.failures_jsonl is not None:
        print(f"Fehler/Abstinenzen: {args.failures_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
