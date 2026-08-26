#!/usr/bin/env python3
"""Fit predictive-only sequence readouts from a persisted Qwen harvester state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from immer.runtimes.ooe.contextual_sequence import (
    DEFAULT_FEATURE_CONFIGS,
    ContextualSequenceBank,
    SequenceSample,
    evaluate_contextual_sequence_holdout,
    fit_contextual_sequence,
)
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.operator_harvester import HarvesterState


REPORT_SCHEMA = "immer-ooe-qwen-contextual-sequence-report/v1"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _mse(left: np.ndarray, right: np.ndarray) -> float:
    difference = np.asarray(left, dtype=np.float64) - np.asarray(
        right, dtype=np.float64
    )
    return float(np.mean(difference * difference))


def _ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0.0 else numerator / denominator


def build_report(
    state: HarvesterState,
    *,
    reservoir_size: int = 24,
    seed: int = 42,
    minimum_prompts: int = 5,
    max_working_bytes: int = 1024**3,
    sequence_bank: ContextualSequenceBank | None = None,
) -> dict[str, Any]:
    """Fit train/validation readouts and evaluate each latest prompt once."""

    if not isinstance(state, HarvesterState):
        raise TypeError("state must be a HarvesterState")
    if minimum_prompts < 3:
        raise ValueError("minimum_prompts must reserve validation and holdout")
    groups = []
    for group_sha256, observations in state.groups:
        if len(observations) < minimum_prompts:
            continue
        samples = tuple(
            SequenceSample(
                x=row.input_array,
                y=row.output_array,
                prompt_sha256=row.measurement.probe.prompt_signature,
                evidence_sha256=row.receipt.sha256,
            )
            for row in observations
        )
        train = samples[:-2]
        validation = samples[-2]
        holdout = samples[-1]
        fit = fit_contextual_sequence(
            train,
            validation,
            reservoir_size=reservoir_size,
            seed=seed,
            max_working_bytes=max_working_bytes,
        )
        pointwise = fit_contextual_sequence(
            train,
            validation,
            configs=(DEFAULT_FEATURE_CONFIGS[0],),
            reservoir_size=reservoir_size,
            seed=seed,
            max_working_bytes=max_working_bytes,
        )
        evaluation = evaluate_contextual_sequence_holdout(fit, holdout)
        pointwise_evaluation = evaluate_contextual_sequence_holdout(pointwise, holdout)
        identity_mse = _mse(holdout.x, holdout.y)
        publication = None
        if sequence_bank is not None:
            published = sequence_bank.publish(fit)
            publication = {
                "changed": published.changed,
                "generation": published.generation,
                "payload_sha256": published.payload_sha256,
            }
        first = observations[0].receipt
        groups.append(
            {
                "actual_mse": evaluation.actual_mse,
                "evaluation_sha256": evaluation.sha256,
                "fit_sha256": fit.sha256,
                "group_sha256": group_sha256,
                "holdout_content_sha256": holdout.content_sha256,
                "holdout_evidence_sha256": holdout.evidence_sha256,
                "holdout_prompt_sha256": holdout.prompt_sha256,
                "identity_mse": identity_mse,
                "model_sha256": fit.model.sha256,
                "output_placebo_mse": evaluation.shuffled_output_mse,
                "pointwise_fit_sha256": pointwise.sha256,
                "pointwise_mse": pointwise_evaluation.actual_mse,
                "predictive_only": True,
                "publication": publication,
                "selected_config": fit.model.config.to_dict(),
                "selected_ridge": fit.model.ridge,
                "sequence_over_identity": _ratio(evaluation.actual_mse, identity_mse),
                "sequence_over_pointwise": _ratio(
                    evaluation.actual_mse, pointwise_evaluation.actual_mse
                ),
                "source_state": first.source_state,
                "target_state": first.target_state,
                "token_placebo_mse": evaluation.shuffled_token_mse,
                "train_content_sha256s": [item.content_sha256 for item in train],
                "train_evidence_sha256s": [item.evidence_sha256 for item in train],
                "train_prompt_sha256s": [item.prompt_sha256 for item in train],
                "validation_content_sha256": validation.content_sha256,
                "validation_evidence_sha256": validation.evidence_sha256,
                "validation_prompt_sha256": validation.prompt_sha256,
            }
        )
    groups.sort(key=lambda row: (row["source_state"], row["target_state"]))
    body = {
        "fit_seed": seed,
        "group_count": len(groups),
        "groups": groups,
        "harvester_identity_sha256": state.identity_sha256,
        "harvester_state_sha256": state.sha256,
        "minimum_prompts": minimum_prompts,
        "predictive_only": True,
        "reservoir_size": reservoir_size,
    }
    return {
        "body": body,
        "schema": REPORT_SCHEMA,
        "sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-root", required=True)
    parser.add_argument("--harvester-state-name", required=True)
    parser.add_argument("--sequence-bank")
    parser.add_argument("--reservoir-size", type=_positive_int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-prompts", type=_positive_int, default=5)
    parser.add_argument("--max-working-mb", type=_positive_float, default=1024.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    store = CrystalStore(Path(args.compute_root).expanduser().absolute())
    state = HarvesterState.from_bytes(store.restore_state(args.harvester_state_name))
    bank = (
        None
        if args.sequence_bank is None
        else ContextualSequenceBank(Path(args.sequence_bank).expanduser().absolute())
    )
    report = build_report(
        state,
        reservoir_size=args.reservoir_size,
        seed=args.seed,
        minimum_prompts=args.minimum_prompts,
        max_working_bytes=int(args.max_working_mb * 1024**2),
        sequence_bank=bank,
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
