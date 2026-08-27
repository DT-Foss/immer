#!/usr/bin/env python3
"""Build a sealed live Prefix-Sinkhorn head-transport report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.operator_transport import (
    OperatorTransportConfig,
    OperatorTransportError,
    QwenPrefixSinkhornCaptureBank,
    chronological_operator_transport_split,
    evaluate_operator_transport_holdout,
    fit_operator_transport,
    qwen_operator_transport_corpus_from_captures,
)
from immer.runtimes.qwen3_8.semantic_atlas import GraphRevision


REPORT_SCHEMA = "immer-ooe-live-prefix-sinkhorn-transport-report/v1"
SELECTION_POLICY = "minimum-per-head-calibration-residual-then-lexicographic-pair"


class LiveOperatorTransportError(RuntimeError):
    """The live capture inventory cannot reproduce its transport report."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _metric_map(rows: Sequence[Any]) -> dict[str, Any]:
    return {row.model_name: row for row in rows}


def _metrics_record(rows: Sequence[Any]) -> dict[str, object]:
    metrics = _metric_map(rows)
    expected = ("per_head", "global", "identity", "random")
    if tuple(metrics) != expected:
        raise LiveOperatorTransportError("transport metric inventory changed")
    return {name: metrics[name].to_dict() for name in expected}


def _kernel_conditions(kernel: Any) -> dict[str, object]:
    return {
        "kernel_sha256": kernel.sha256,
        "nullspace_gap": kernel.nullspace_gap,
        "system_condition": kernel.system_condition,
        "system_smallest_singular": kernel.system_smallest_singular,
        "transport_condition": kernel.transport_condition,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0.0:
        return None
    value = numerator / denominator
    if not math.isfinite(value) or value < 0.0:
        raise LiveOperatorTransportError("transport residual ratio is invalid")
    return 0.0 if value == 0.0 else value


def _accepted_pair_record(
    pair: tuple[int, int],
    corpus: Any,
    split: Any,
    fit: Any,
) -> dict[str, object]:
    calibration = _metric_map(fit.calibration_metrics)
    kernel = fit.per_head_kernels[0]
    if (kernel.source_head, kernel.target_head) != pair:
        raise LiveOperatorTransportError("per-head kernel names another pair")
    return {
        "calibration_metrics": _metrics_record(fit.calibration_metrics),
        "calibration_residual": calibration["per_head"].mean_relative_residual,
        "conditions": {
            "global": _kernel_conditions(fit.global_kernel),
            "per_head": _kernel_conditions(kernel),
        },
        "corpus_sha256": corpus.sha256,
        "fit_sha256": fit.sha256,
        "pair": list(pair),
        "rejection": None,
        "split_sha256": split.sha256,
        "status": "accepted",
        "train_metrics": _metrics_record(fit.train_metrics),
    }


def _rejected_pair_record(
    pair: tuple[int, int],
    *,
    stage: str,
    error: OperatorTransportError,
    corpus_sha256: str | None,
    split_sha256: str | None,
) -> dict[str, object]:
    return {
        "calibration_metrics": None,
        "calibration_residual": None,
        "conditions": None,
        "corpus_sha256": corpus_sha256,
        "fit_sha256": None,
        "pair": list(pair),
        "rejection": {
            "error_type": type(error).__name__,
            "reason": str(error),
            "stage": stage,
        },
        "split_sha256": split_sha256,
        "status": "rejected",
        "train_metrics": None,
    }


def build_report(
    bank_root: str | os.PathLike[str],
    *,
    graph_revision: GraphRevision,
    config: OperatorTransportConfig | None = None,
    _verify: bool = True,
) -> dict[str, object]:
    if not isinstance(graph_revision, GraphRevision):
        raise TypeError("graph_revision must be GraphRevision")
    active_config = config or OperatorTransportConfig()
    if not isinstance(active_config, OperatorTransportConfig):
        raise TypeError("config must be OperatorTransportConfig")
    bank = QwenPrefixSinkhornCaptureBank(bank_root)
    captures = bank.receipts()
    if len(captures) < 3:
        raise LiveOperatorTransportError("at least three live captures are required")
    heads = captures[0].head_indices
    if any(row.head_indices != heads for row in captures):
        raise LiveOperatorTransportError("capture head inventory changed")
    pairs = tuple(
        (source, target)
        for source in heads
        for target in heads
        if source != target
    )
    pair_records: list[dict[str, object]] = []
    fit_by_pair: dict[tuple[int, int], Any] = {}
    for pair in pairs:
        corpus = None
        split = None
        try:
            corpus = qwen_operator_transport_corpus_from_captures(
                captures,
                graph_revision=graph_revision,
                head_pairs=(pair,),
            )
            split = chronological_operator_transport_split(corpus)
            fit = fit_operator_transport(corpus, split, config=active_config)
        except OperatorTransportError as exc:
            stage = (
                "corpus"
                if corpus is None
                else ("split" if split is None else "fit-condition")
            )
            pair_records.append(
                _rejected_pair_record(
                    pair,
                    stage=stage,
                    error=exc,
                    corpus_sha256=None if corpus is None else corpus.sha256,
                    split_sha256=None if split is None else split.sha256,
                )
            )
            continue
        fit_by_pair[pair] = fit
        pair_records.append(_accepted_pair_record(pair, corpus, split, fit))
    accepted = tuple(
        row for row in pair_records if row["status"] == "accepted"
    )
    if not accepted:
        raise LiveOperatorTransportError("no directed head pair passed fit conditions")
    selected_record = min(
        accepted,
        key=lambda row: (
            float(row["calibration_residual"]),
            tuple(row["pair"]),
        ),
    )
    selected_pair = tuple(selected_record["pair"])
    selected_fit = fit_by_pair[selected_pair]
    holdout = evaluate_operator_transport_holdout(selected_fit)
    holdout_metrics = _metric_map(holdout.holdout_metrics)
    per_head = holdout_metrics["per_head"].mean_relative_residual
    identity = holdout_metrics["identity"].mean_relative_residual
    random = holdout_metrics["random"].mean_relative_residual
    selection_inputs = [
        {
            "calibration_residual": row["calibration_residual"],
            "fit_sha256": row["fit_sha256"],
            "pair": row["pair"],
        }
        for row in accepted
    ]
    audit = bank.audit()
    body = {
        "audit": {
            "audit_clean": not audit.orphan_state_filenames,
            "inventory_sha256": audit.inventory_sha256,
            "orphan_state_filenames": list(audit.orphan_state_filenames),
        },
        "capture_receipt_sha256s": [row.sha256 for row in captures],
        "capture_count": len(captures),
        "config": active_config.to_dict(),
        "config_sha256": active_config.sha256,
        "directed_pair_count": len(pairs),
        "graph_revision": graph_revision.to_document(),
        "holdout": {
            "evaluated_pairs": [list(selected_pair)],
            "holdout_sha256": holdout.sha256,
            "metrics": _metrics_record(holdout.holdout_metrics),
            "ratios": {
                "per_head_to_identity": _ratio(per_head, identity),
                "per_head_to_random": _ratio(per_head, random),
                "identity_over_per_head": _ratio(identity, per_head),
                "random_over_per_head": _ratio(random, per_head),
            },
        },
        "pair_records": pair_records,
        "selection": {
            "accepted_pair_count": len(accepted),
            "holdout_used": False,
            "inputs_sha256": _digest(selection_inputs),
            "policy": SELECTION_POLICY,
            "selected_pair": list(selected_pair),
            "selected_fit_sha256": selected_fit.sha256,
        },
    }
    report = {"schema": REPORT_SCHEMA, "body": body, "body_sha256": _digest(body)}
    if _verify:
        verify_report(report, bank_root)
    return report


def verify_report(
    report: object,
    bank_root: str | os.PathLike[str],
) -> bool:
    if (
        not isinstance(report, Mapping)
        or set(report) != {"schema", "body", "body_sha256"}
        or report.get("schema") != REPORT_SCHEMA
        or not isinstance(report.get("body"), Mapping)
        or report.get("body_sha256") != _digest(report.get("body"))
    ):
        raise LiveOperatorTransportError("live transport report seal is invalid")
    body = report["body"]
    try:
        graph_revision = GraphRevision.from_document(body["graph_revision"])
        config = OperatorTransportConfig.from_dict(body["config"])
    except (KeyError, OperatorTransportError, TypeError, ValueError) as exc:
        raise LiveOperatorTransportError(
            "live transport report authority is invalid"
        ) from exc
    expected = build_report(
        bank_root,
        graph_revision=graph_revision,
        config=config,
        _verify=False,
    )
    if canonical_json_bytes(expected) != canonical_json_bytes(report):
        raise LiveOperatorTransportError(
            "live transport report does not recompute from its capture bank"
        )
    return True


def write_report_no_replace(path: Path, report: Mapping[str, object]) -> None:
    target = Path(path).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(report)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".pending", dir=target.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary_path.unlink(missing_ok=True)


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", required=True)
    parser.add_argument("--graph-sequence", required=True, type=_nonnegative_int)
    parser.add_argument("--graph-sha256", required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(
        args.bank_root,
        graph_revision=GraphRevision(args.graph_sequence, args.graph_sha256),
    )
    if args.output is not None:
        write_report_no_replace(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
