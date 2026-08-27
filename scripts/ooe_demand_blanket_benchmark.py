#!/usr/bin/env python3
"""Benchmark non-contiguous Demand memory on authenticated compute routes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from immer.runtimes.ooe.compute_crystals import (
    ComputeCrystal,
    ComputeCrystalBank,
    tensor_sha256,
)
from immer.runtimes.ooe.compute_graph import (
    ComputeOperatorGraph,
    ComputeOperatorGraphState,
    ComputeRoutePlan,
    MaterializedRoute,
    OperatorEdge,
)
from immer.runtimes.ooe.conditional_blanket import ConditionalBlanketConfig
from immer.runtimes.ooe.demand_blanket import (
    DemandLagBlanketPredictionReceipt,
    DemandLagBlanketValidationReceipt,
    DemandLagCorpusReceipt,
    build_demand_lag_corpus,
    chronological_episode_group_split,
    fit_demand_lag_blanket,
    predict_demand_lag_blanket,
    validate_demand_lag_blanket,
)
from immer.runtimes.ooe.demand_execution import (
    DemandExecutionVerification,
    DemandRoutedExecutionReceipt,
    DemandRoutedExecutor,
)
from immer.runtimes.ooe.demand_scheduler import (
    DemandOutcomeReceipt,
    OperatorDemandConfig,
    OperatorDemandScheduler,
    OperatorDemandState,
    OutcomeEvent,
)
from immer.runtimes.ooe.identity import canonical_json_bytes, require_sha256
from immer.runtimes.ooe.residual_execution import ResidualRouteExecution


REPORT_SCHEMA = "immer-ooe-demand-blanket-intelligence-benchmark/v1"
SEED_REPORT_SCHEMA = "immer-ooe-demand-blanket-intelligence-seed/v1"
EPISODE_COUNT = 3
EPISODE_LENGTH = 10
MAX_CONTEXT_ORDER = 3


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


PARITY_LAW_SHA256 = _digest(
    {
        "schema": "immer-ooe-demand-parity-law/v1",
        "law": "route[t] = route[bit(route[t-1]) XOR bit(route[t-3])]",
        "selected_lags": [1, 3],
    }
)
EXACT_OUTPUT_VERIFIER_SHA256 = _digest(
    {
        "schema": "immer-ooe-exact-residual-output-verifier/v1",
        "comparison": "numpy.array_equal",
        "binding": "residual-receipt-and-output-sha256",
    }
)
ALGORITHM_SHA256 = _digest(
    {
        "schema": "immer-ooe-demand-blanket-algorithm/v1",
        "fit": "exhaustive-exact-conditional-blanket",
        "selection": "non-contiguous-lags",
        "baseline": "train-only-longest-contiguous-prefix-PPM",
        "runtime_priority": ["lag-blanket", "PPM", "full-UCB1"],
    }
)
PROTOCOL_SHA256 = _digest(
    {
        "schema": "immer-ooe-demand-blanket-benchmark-protocol/v1",
        "episode_count": EPISODE_COUNT,
        "episode_length": EPISODE_LENGTH,
        "max_context_order": MAX_CONTEXT_ORDER,
        "split": {
            "algorithm": "chronological-episode-group-split",
            "requested_train_fraction": "3/5",
            "requested_calibration_fraction": "1/5",
            "realized_episode_counts": {"train": 1, "calibration": 1, "holdout": 1},
        },
        "replication": "fixed-causal-intervention-independent-evidence-identities",
        "required_selected_lags": [1, 3],
        "execution_priority": ["lag-blanket", "PPM", "full-UCB1"],
    }
)


class DemandBlanketBenchmarkError(RuntimeError):
    """The benchmark could not reproduce its declared mechanism."""


def _ratio(value: Fraction | int) -> dict[str, int]:
    exact = Fraction(value)
    return {"numerator": exact.numerator, "denominator": exact.denominator}


def _from_ratio(value: object, *, field: str) -> Fraction:
    if not isinstance(value, Mapping) or set(value) != {"numerator", "denominator"}:
        raise DemandBlanketBenchmarkError(f"{field} is not an exact ratio")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise DemandBlanketBenchmarkError(f"{field} is not an exact ratio")
    result = Fraction(numerator, denominator)
    if _ratio(result) != dict(value):
        raise DemandBlanketBenchmarkError(f"{field} is not canonical")
    return result


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _crystals() -> tuple[ComputeCrystal, ...]:
    return (
        ComputeCrystal.affine([[1.0, 2.0], [-1.0, 1.0]], [3.0, -2.0]),
        ComputeCrystal.affine([[0.0, 1.0], [1.0, 0.0]], [1.0, 0.0]),
        ComputeCrystal.affine([[2.0, 0.0], [0.0, -1.0]], [-4.0, 5.0]),
        ComputeCrystal.affine([[0.5, 0.0], [1.0, 1.0]], [2.0, -3.0]),
    )


def _apply(crystals: Sequence[ComputeCrystal], value: np.ndarray, length: int) -> np.ndarray:
    result = value
    for crystal in crystals[:length]:
        result = crystal.apply(result)
    return result


def _parity_sequence(episode: int) -> tuple[int, ...]:
    # The seven non-zero third-order states are phase-rotated across episodes.
    initial = 1 + (episode % 7)
    bits = [(initial >> index) & 1 for index in range(3)]
    while len(bits) < EPISODE_LENGTH:
        bits.append(bits[-1] ^ bits[-3])
    return tuple(bits)


def _route_matches_plan(route: MaterializedRoute, plan: ComputeRoutePlan) -> bool:
    length = len(route.primitive_edge_sha256s)
    return bool(
        0 < length <= len(plan.primitive_edge_sha256s)
        and route.source_state == plan.source_state
        and route.goal_state == plan.finite_plan.expected_states[length]
        and route.primitive_edge_sha256s == plan.primitive_edge_sha256s[:length]
    )


class _ExactVerifier:
    def __init__(self, expected: np.ndarray) -> None:
        self.expected = expected
        self.calls = 0

    def __call__(self, execution: ResidualRouteExecution) -> DemandExecutionVerification:
        self.calls += 1
        accepted = bool(np.array_equal(execution.output, self.expected))
        return DemandExecutionVerification.create(
            execution,
            accepted=accepted,
            verifier_sha256=EXACT_OUTPUT_VERIFIER_SHA256,
            evidence_sha256=_digest(
                {
                    "schema": "immer-ooe-exact-residual-output-evidence/v1",
                    "residual_receipt_sha256": execution.receipt.sha256,
                    "accepted": accepted,
                }
            ),
            reason="exact-output-match" if accepted else "exact-output-mismatch",
        )


def _record_parity_episodes(
    *,
    seed: int,
    scheduler: OperatorDemandScheduler,
    graph_state: Any,
    routes: tuple[MaterializedRoute, MaterializedRoute],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    episode_hashes: list[str] = []
    outcome_hashes: list[str] = []
    for episode in range(EPISODE_COUNT):
        episode_sha256 = _digest(
            {"protocol_sha256": PROTOCOL_SHA256, "seed": seed, "episode": episode}
        )
        episode_hashes.append(episode_sha256)
        for position, bit in enumerate(_parity_sequence(episode)):
            route = routes[bit]
            outcome = DemandOutcomeReceipt.create(
                route=route,
                graph_state=graph_state,
                success=True,
                reward=1.0,
                outcome_evidence_sha256=_digest(
                    {
                        "law_sha256": PARITY_LAW_SHA256,
                        "seed": seed,
                        "episode": episode,
                        "position": position,
                        "route_sha256": route.sha256,
                    }
                ),
                outcome_verifier_sha256=PARITY_LAW_SHA256,
                episode_sha256=episode_sha256,
            )
            scheduler.record_outcome(outcome, graph_state)
            outcome_hashes.append(outcome.sha256)
    return tuple(episode_hashes), tuple(outcome_hashes)


def _ppm_holdout_baseline(corpus: DemandLagCorpusReceipt, split: Any) -> dict[str, object]:
    train_groups = set(split.train_episode_sha256s)
    holdout_groups = set(split.holdout_episode_sha256s)
    grouped: dict[str, list[Any]] = defaultdict(list)
    for sample in corpus.samples:
        grouped[sample.episode_sha256].append(sample)
    train_sequences = [
        tuple(row.target_route_sha256 for row in sorted(rows, key=lambda row: row.position))
        for episode, rows in grouped.items()
        if episode in train_groups
    ]
    records: list[dict[str, object]] = []
    correct = 0
    full_context = 0
    matched_order_sum = 0
    brier_sum = Fraction()
    alphabet = corpus.route_alphabet_sha256s
    for episode in split.holdout_episode_sha256s:
        history: list[str] = []
        for row in sorted(grouped[episode], key=lambda item: item.position):
            selected_counts: dict[str, int] = {}
            matched_order = 0
            for order in range(min(MAX_CONTEXT_ORDER, len(history)), -1, -1):
                context = tuple(history[-order:]) if order else ()
                counts: dict[str, int] = defaultdict(int)
                for sequence in train_sequences:
                    for index, target in enumerate(sequence):
                        if index >= order and tuple(sequence[index - order : index]) == context:
                            counts[target] += 1
                if counts:
                    selected_counts = dict(counts)
                    matched_order = order
                    break
            total = sum(selected_counts.values())
            if total <= 0:
                raise DemandBlanketBenchmarkError("train-only PPM found no marginal evidence")
            best = max(selected_counts.values())
            predicted = min(route for route, count in selected_counts.items() if count == best)
            correct += int(predicted == row.target_route_sha256)
            full_context += int(matched_order == MAX_CONTEXT_ORDER)
            matched_order_sum += matched_order
            for route in alphabet:
                probability = Fraction(selected_counts.get(route, 0), total)
                truth = int(route == row.target_route_sha256)
                brier_sum += (probability - truth) ** 2
            records.append(
                {
                    "sample_sha256": row.sha256,
                    "matched_order": matched_order,
                    "prediction": predicted,
                    "target": row.target_route_sha256,
                    "counts": sorted(selected_counts.items()),
                }
            )
            history.append(row.target_route_sha256)
    count = len(records)
    if count == 0 or set(grouped) & holdout_groups != holdout_groups:
        raise DemandBlanketBenchmarkError("PPM holdout inventory is empty or incomplete")
    return {
        "accuracy": _ratio(Fraction(correct, count)),
        "multiclass_brier": _ratio(brier_sum / count),
        "full_context_coverage": _ratio(Fraction(full_context, count)),
        "mean_matched_context_order": _ratio(Fraction(matched_order_sum, count)),
        "sample_count": count,
        "prediction_inventory_sha256": _digest(records),
    }


def _find_history(
    validation: DemandLagBlanketValidationReceipt,
    scheduler: OperatorDemandScheduler,
    graph_state: Any,
    *,
    target: str | None = None,
) -> tuple[tuple[str, ...], DemandLagBlanketPredictionReceipt]:
    alphabet = validation.fit.corpus.route_alphabet_sha256s
    for suffix in product(alphabet, repeat=3):
        # Five tokens prove the complete history was absent from the width-three corpus.
        history = (alphabet[1], alphabet[0], *suffix)
        expected_bit = alphabet.index(history[-1]) ^ alphabet.index(history[-3])
        expected = alphabet[expected_bit]
        if target is not None and expected != target:
            continue
        prediction = predict_demand_lag_blanket(
            validation,
            scheduler,
            graph_state,
            history,
            input_abi_sha256=validation.fit.corpus.input_abi_sha256,
            minimum_probability=Fraction(3, 4),
        )
        if prediction.candidate_route_sha256 == expected:
            return history, prediction
    raise DemandBlanketBenchmarkError("no exact unseen parity history was predicted")


def _execution_record(
    result: Any, validation: DemandLagBlanketValidationReceipt
) -> dict[str, object]:
    receipt = DemandRoutedExecutionReceipt.from_bytes(
        result.receipt.to_bytes(), lag_blanket_validation=validation
    )
    return _execution_receipt_record(receipt)


def _execution_receipt_record(
    receipt: DemandRoutedExecutionReceipt,
) -> dict[str, object]:
    return {
        "execution_sha256": receipt.sha256,
        "selected_prefix_route_sha256": receipt.selected_prefix_route_sha256,
        "blanket_prediction_sha256": (
            None if receipt.blanket_prediction is None else receipt.blanket_prediction.sha256
        ),
        "blanket_candidate_route_sha256": (
            None
            if receipt.blanket_prediction is None
            else receipt.blanket_prediction.candidate_route_sha256
        ),
        "ppm_prediction_sha256": (
            None if receipt.ppm_prediction is None else receipt.ppm_prediction.sha256
        ),
        "ppm_selected_route_sha256": (
            None if receipt.ppm_prediction is None else receipt.ppm_prediction.selected_route_sha256
        ),
        "selection_score_count": len(receipt.selection.event.scores),
        "output_sha256": receipt.residual.output_sha256,
        "verified": receipt.verification.accepted,
        "historical_work_released": receipt.residual.historical_work_released,
        "equivalent_source_work_units": receipt.residual.equivalent_source_work_units,
        "live_work_units": receipt.residual.live_work_units,
    }


def run_seed(seed: int) -> dict[str, object]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    with tempfile.TemporaryDirectory(prefix="immer-demand-blanket-") as temporary:
        bank = ComputeCrystalBank(Path(temporary) / "bank")
        graph = ComputeOperatorGraph(bank)
        crystals = _crystals()
        states = ("root", "layer-1", "layer-2", "layer-3", "terminal")
        for index, crystal in enumerate(crystals):
            bank.publish_crystal(crystal)
            graph.append_edge(
                OperatorEdge(
                    source_state=states[index],
                    target_state=states[index + 1],
                    crystal_sha256=crystal.sha256,
                    verifier_sha256=_digest({"edge-verifier": index}),
                    evidence_sha256=_digest({"edge-evidence": index}),
                )
            )
        two_plan = graph.plan_route("root", "layer-2").plan
        three_plan = graph.plan_route("root", "layer-3").plan
        if two_plan is None or three_plan is None:
            raise DemandBlanketBenchmarkError("prefix plans are unavailable")
        two = graph.charge_route(two_plan).route
        three = graph.charge_route(three_plan).route
        graph_state = graph.state()
        full_plan = graph.plan_route("root", "terminal").plan
        short_plan = graph.plan_route("root", "layer-2").plan
        if full_plan is None or short_plan is None:
            raise DemandBlanketBenchmarkError("execution plans are unavailable")
        routes = tuple(sorted((two, three), key=lambda route: route.sha256))
        scheduler = OperatorDemandScheduler(
            bank.store,
            config=OperatorDemandConfig(max_context_order=MAX_CONTEXT_ORDER),
        )
        episode_hashes, outcome_hashes = _record_parity_episodes(
            seed=seed,
            scheduler=scheduler,
            graph_state=graph_state,
            routes=routes,
        )
        training_scheduler_state = scheduler.state()
        corpus = build_demand_lag_corpus(
            scheduler,
            graph_state,
            input_abi_sha256=routes[0].input_abi_sha256,
            max_context_order=MAX_CONTEXT_ORDER,
        )
        split = chronological_episode_group_split(corpus)
        config = ConditionalBlanketConfig(
            target_alphabet=corpus.route_alphabet_sha256s,
            max_subset_size=MAX_CONTEXT_ORDER,
            max_exhaustive_subsets=100,
            max_pair_checks=16,
            laplace_alpha=Fraction(1),
            max_brier_regret=Fraction(1, 20),
            max_accuracy_regret=Fraction(1, 20),
            max_coverage_regret=Fraction(1),
            dependence_tolerance=Fraction(1, 10),
            placebo_seed_sha256=_digest(
                {"protocol_sha256": PROTOCOL_SHA256, "seed": seed, "placebo": True}
            ),
        )
        fit = fit_demand_lag_blanket(corpus, split, config=config)
        validation = validate_demand_lag_blanket(
            fit, minimum_holdout_accuracy=Fraction(3, 4)
        )
        validation.verify_or_raise()
        if fit.selected_lags != (1, 3) or not validation.accepted:
            raise DemandBlanketBenchmarkError(
                f"seed {seed} recovered {fit.selected_lags!r}; "
                f"validation accepted={validation.accepted} ({validation.reason})"
            )
        metrics = validation.generic_validation
        ppm_baseline = _ppm_holdout_baseline(corpus, split)

        unseen_history, prediction = _find_history(
            validation, scheduler, graph_state
        )
        prediction = DemandLagBlanketPredictionReceipt.from_bytes(
            prediction.to_bytes(), validation=validation
        )
        value = np.array([[3.5, -2.0], [11.0, 0.25], [-7.0, 9.0]], dtype=np.float64)
        expected_full = _apply(crystals, value, len(crystals))
        blanket_executor = DemandRoutedExecutor(
            graph=graph,
            scheduler=scheduler,
            verifier=_ExactVerifier(expected_full),
            lag_blanket_validation=validation,
        )
        blanket_execution = blanket_executor.execute(
            full_plan,
            value,
            candidate_route_sha256s=tuple(route.sha256 for route in routes),
            history_route_sha256s=unseen_history,
        )

        fallback_executor = DemandRoutedExecutor(
            graph=graph,
            scheduler=scheduler,
            verifier=_ExactVerifier(expected_full),
            lag_blanket_validation=validation,
            lag_blanket_min_probability=Fraction(1),
        )
        ppm_execution = fallback_executor.execute(
            full_plan,
            value,
            candidate_route_sha256s=tuple(route.sha256 for route in routes),
            history_route_sha256s=unseen_history,
        )
        ucb_execution = fallback_executor.execute(
            full_plan,
            value,
            candidate_route_sha256s=tuple(route.sha256 for route in routes),
        )

        incompatible_route = three
        compatible_route = two
        incompatible_history, _ = _find_history(
            validation,
            scheduler,
            graph_state,
            target=incompatible_route.sha256,
        )
        expected_short = _apply(crystals, value, 2)
        guarded_executor = DemandRoutedExecutor(
            graph=graph,
            scheduler=scheduler,
            verifier=_ExactVerifier(expected_short),
            lag_blanket_validation=validation,
        )
        guarded_execution = guarded_executor.execute(
            short_plan,
            value,
            candidate_route_sha256s=(compatible_route.sha256,),
            history_route_sha256s=incompatible_history,
        )

        executions = {
            "blanket": _execution_record(blanket_execution, validation),
            "ppm_fallback": _execution_record(ppm_execution, validation),
            "full_ucb_fallback": _execution_record(ucb_execution, validation),
            "incompatible_guard": _execution_record(guarded_execution, validation),
        }
        checks = {
            "selected_noncontiguous_lags": fit.selected_lags == (1, 3),
            "validation_accepted": validation.accepted,
            "unseen_history_exact": (
                prediction.candidate_route_sha256
                == blanket_execution.receipt.selected_prefix_route_sha256
            ),
            "blanket_precedes_ppm": blanket_execution.receipt.ppm_prediction is None,
            "ppm_precedes_ucb": (
                ppm_execution.receipt.blanket_prediction is not None
                and ppm_execution.receipt.blanket_prediction.candidate_route_sha256 is None
                and ppm_execution.receipt.ppm_prediction is not None
                and ppm_execution.receipt.ppm_prediction.selected_route_sha256
                == ppm_execution.receipt.selected_prefix_route_sha256
            ),
            "full_ucb_preserves_all_arms": (
                ucb_execution.receipt.blanket_prediction is not None
                and ucb_execution.receipt.blanket_prediction.candidate_route_sha256 is None
                and ucb_execution.receipt.ppm_prediction is None
                and len(ucb_execution.receipt.selection.event.scores) == 2
            ),
            "incompatible_prediction_never_executed": (
                guarded_execution.receipt.blanket_prediction is not None
                and guarded_execution.receipt.blanket_prediction.candidate_route_sha256
                == incompatible_route.sha256
                and guarded_execution.receipt.selected_prefix_route_sha256
                == compatible_route.sha256
                and not _route_matches_plan(incompatible_route, short_plan)
            ),
            "all_outputs_exact": all(
                execution.receipt.verification.accepted
                for execution in (
                    blanket_execution,
                    ppm_execution,
                    ucb_execution,
                    guarded_execution,
                )
            ),
            "all_executions_release_past_compute": all(
                execution.receipt.residual.historical_work_released > 0
                for execution in (
                    blanket_execution,
                    ppm_execution,
                    ucb_execution,
                    guarded_execution,
                )
            ),
        }
        if not all(checks.values()):
            failed = sorted(name for name, passed in checks.items() if not passed)
            raise DemandBlanketBenchmarkError(f"seed {seed} failed: {', '.join(failed)}")
        body = {
            "seed": seed,
            "receipt_sha256": _digest(list(outcome_hashes)),
            "model_sha256": fit.sha256,
            "validation_sha256": validation.sha256,
            "prediction_sha256": prediction.sha256,
            "execution_sha256s": {
                name: record["execution_sha256"] for name, record in executions.items()
            },
            "bank_manifest_sha256": bank.manifest().sha256,
            "graph_state_sha256": graph_state.sha256,
            "graph_state_receipt": graph_state.to_document(),
            "scheduler_state_receipt": training_scheduler_state.to_document(),
            "crystal_receipts": [crystal.to_document() for crystal in crystals],
            "graph_generation": graph_state.generation,
            "full_plan_sha256": full_plan.sha256,
            "short_plan_sha256": short_plan.sha256,
            "route_sha256s": [route.sha256 for route in routes],
            "episode_sha256s": list(episode_hashes),
            "outcome_receipt_sha256s": list(outcome_hashes),
            "corpus_sha256": corpus.sha256,
            "split_sha256": split.sha256,
            "fit_config_sha256": config.sha256,
            "selected_lags": list(fit.selected_lags),
            "fit_status": fit.generic_fit.status,
            "closure_verified": fit.generic_fit.closure_verified,
            "train_episode_count": len(split.train_episode_sha256s),
            "calibration_episode_count": len(split.calibration_episode_sha256s),
            "holdout_episode_count": len(split.holdout_episode_sha256s),
            "holdout_metrics": {
                "selected": metrics.selected_metrics.to_dict(),
                "full": metrics.full_metrics.to_dict(),
                "random": metrics.random_metrics.to_dict(),
                "marginal": metrics.marginal_metrics.to_dict(),
                "contiguous_ppm": ppm_baseline,
            },
            "unseen_history_route_sha256s": list(unseen_history),
            "unseen_history_sha256": _digest(list(unseen_history)),
            "input_tensor": value.tolist(),
            "validation_receipt": validation.to_dict(),
            "prediction_receipt": prediction.to_dict(),
            "execution_receipts": {
                "blanket": blanket_execution.receipt.to_dict(),
                "ppm_fallback": ppm_execution.receipt.to_dict(),
                "full_ucb_fallback": ucb_execution.receipt.to_dict(),
                "incompatible_guard": guarded_execution.receipt.to_dict(),
            },
            "executions": executions,
            "checks": checks,
        }
        return {
            "schema": SEED_REPORT_SCHEMA,
            "body": body,
            "body_sha256": _digest(body),
        }


def _aggregate(seed_reports: Sequence[Mapping[str, object]]) -> dict[str, object]:
    bodies = [report["body"] for report in seed_reports]
    if any(not isinstance(body, Mapping) for body in bodies):
        raise DemandBlanketBenchmarkError("seed report body is invalid")
    seed_count = len(bodies)
    if seed_count == 0:
        raise DemandBlanketBenchmarkError("at least one seed is required")

    def success(check: str) -> Fraction:
        return Fraction(sum(bool(body["checks"][check]) for body in bodies), seed_count)  # type: ignore[index]

    def pooled_accuracy(arm: str) -> Fraction:
        correct = 0
        total = 0
        for body in bodies:
            metric = body["holdout_metrics"][arm]  # type: ignore[index]
            count = metric["sample_count"]
            accuracy = _from_ratio(metric["top1_accuracy"], field=f"{arm}.accuracy")
            exact = accuracy * count
            if exact.denominator != 1:
                raise DemandBlanketBenchmarkError("metric accuracy is not sample-exact")
            correct += exact.numerator
            total += count
        return Fraction(correct, total)

    ppm_correct = 0
    ppm_total = 0
    work_released = 0
    source_work = 0
    for body in bodies:
        ppm = body["holdout_metrics"]["contiguous_ppm"]  # type: ignore[index]
        count = ppm["sample_count"]
        exact = _from_ratio(ppm["accuracy"], field="ppm.accuracy") * count
        if exact.denominator != 1:
            raise DemandBlanketBenchmarkError("PPM accuracy is not sample-exact")
        ppm_correct += exact.numerator
        ppm_total += count
        for execution in body["executions"].values():  # type: ignore[index,union-attr]
            work_released += execution["historical_work_released"]
            source_work += execution["equivalent_source_work_units"]
    return {
        "seed_count": seed_count,
        "selected_lags_1_3_ratio": _ratio(success("selected_noncontiguous_lags")),
        "validation_acceptance_ratio": _ratio(success("validation_accepted")),
        "unseen_history_exact_ratio": _ratio(success("unseen_history_exact")),
        "blanket_priority_ratio": _ratio(success("blanket_precedes_ppm")),
        "ppm_priority_ratio": _ratio(success("ppm_precedes_ucb")),
        "full_ucb_priority_ratio": _ratio(success("full_ucb_preserves_all_arms")),
        "incompatible_never_executed_ratio": _ratio(
            success("incompatible_prediction_never_executed")
        ),
        "exact_output_ratio": _ratio(success("all_outputs_exact")),
        "past_compute_release_ratio": _ratio(
            success("all_executions_release_past_compute")
        ),
        "selected_holdout_accuracy": _ratio(pooled_accuracy("selected")),
        "full_holdout_accuracy": _ratio(pooled_accuracy("full")),
        "random_holdout_accuracy": _ratio(pooled_accuracy("random")),
        "marginal_holdout_accuracy": _ratio(pooled_accuracy("marginal")),
        "contiguous_ppm_holdout_accuracy": _ratio(Fraction(ppm_correct, ppm_total)),
        "lag_feature_reduction": _ratio(Fraction(1, 3)),
        "historical_work_released": work_released,
        "equivalent_source_work_units": source_work,
        "historical_work_release_ratio": _ratio(Fraction(work_released, source_work)),
        "seed_report_sha256s": [report["body_sha256"] for report in seed_reports],
    }


def build_report(*, seeds: int = 1) -> dict[str, object]:
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds <= 0:
        raise ValueError("seeds must be a positive integer")
    seed_reports = [run_seed(seed) for seed in range(seeds)]
    body = {
        "algorithm_sha256": ALGORITHM_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "parity_law_sha256": PARITY_LAW_SHA256,
        "exact_output_verifier_sha256": EXACT_OUTPUT_VERIFIER_SHA256,
        "seed_reports": seed_reports,
        "aggregate": _aggregate(seed_reports),
    }
    report = {"schema": REPORT_SCHEMA, "body": body, "body_sha256": _digest(body)}
    verify_report(report)
    return report


def _replay_seed_body(seed_body: Mapping[str, object]) -> None:
    """Recompute the model, predictions, executions, outputs, and root evidence."""

    graph_state = ComputeOperatorGraphState.from_bytes(
        canonical_json_bytes(seed_body["graph_state_receipt"])
    )
    scheduler_state = OperatorDemandState.from_bytes(
        canonical_json_bytes(seed_body["scheduler_state_receipt"])
    )
    validation = DemandLagBlanketValidationReceipt.from_bytes(
        canonical_json_bytes(seed_body["validation_receipt"])
    )
    prediction = DemandLagBlanketPredictionReceipt.from_bytes(
        canonical_json_bytes(seed_body["prediction_receipt"]),
        validation=validation,
    )
    raw_crystals = seed_body["crystal_receipts"]
    raw_executions = seed_body["execution_receipts"]
    if not isinstance(raw_crystals, list) or not isinstance(raw_executions, Mapping):
        raise DemandBlanketBenchmarkError("embedded executable evidence is invalid")
    crystals = tuple(
        ComputeCrystal.from_bytes(canonical_json_bytes(document))
        for document in raw_crystals
    )
    execution_names = (
        "blanket",
        "ppm_fallback",
        "full_ucb_fallback",
        "incompatible_guard",
    )
    if set(raw_executions) != set(execution_names):
        raise DemandBlanketBenchmarkError("embedded execution inventory changed")
    execution_receipts = {
        name: DemandRoutedExecutionReceipt.from_bytes(
            canonical_json_bytes(raw_executions[name]),
            lag_blanket_validation=validation,
        )
        for name in execution_names
    }
    corpus = validation.fit.corpus
    split = validation.fit.split
    if (
        graph_state.sha256 != seed_body["graph_state_sha256"]
        or graph_state.generation != seed_body["graph_generation"]
        or scheduler_state.sha256 != corpus.scheduler_state_sha256
        or validation.fit.sha256 != seed_body["model_sha256"]
        or validation.sha256 != seed_body["validation_sha256"]
        or prediction.sha256 != seed_body["prediction_sha256"]
        or corpus.sha256 != seed_body["corpus_sha256"]
        or split.sha256 != seed_body["split_sha256"]
        or validation.fit.generic_fit.config.sha256 != seed_body["fit_config_sha256"]
    ):
        raise DemandBlanketBenchmarkError("embedded model or root identity changed")

    outcomes = tuple(
        event.outcome
        for event in scheduler_state.events
        if isinstance(event, OutcomeEvent)
    )
    outcome_hashes = tuple(outcome.sha256 for outcome in outcomes)
    if (
        len(outcomes) != EPISODE_COUNT * EPISODE_LENGTH
        or list(outcome_hashes) != seed_body["outcome_receipt_sha256s"]
        or _digest(list(outcome_hashes)) != seed_body["receipt_sha256"]
        or outcome_hashes != corpus.outcome_receipt_sha256s
        or any(
            not outcome.success
            or outcome.episode_sha256 is None
            or outcome.outcome_verifier_sha256 != PARITY_LAW_SHA256
            for outcome in outcomes
        )
    ):
        raise DemandBlanketBenchmarkError("explicit successful episode roots changed")
    episodes = tuple(dict.fromkeys(outcome.episode_sha256 for outcome in outcomes))
    if list(episodes) != seed_body["episode_sha256s"]:
        raise DemandBlanketBenchmarkError("episode group inventory changed")
    outcome_by_sha = {outcome.sha256: outcome for outcome in outcomes}
    alphabet = corpus.route_alphabet_sha256s
    route_bit = {route: bit for bit, route in enumerate(alphabet)}
    for sample in corpus.samples:
        outcome = outcome_by_sha.get(sample.outcome_receipt_sha256)
        if outcome is None:
            raise DemandBlanketBenchmarkError("corpus names absent outcome evidence")
        episode_index = episodes.index(sample.episode_sha256)
        expected_evidence = _digest(
            {
                "law_sha256": PARITY_LAW_SHA256,
                "seed": seed_body["seed"],
                "episode": episode_index,
                "position": sample.position,
                "route_sha256": sample.target_route_sha256,
            }
        )
        if outcome.outcome_evidence_sha256 != expected_evidence:
            raise DemandBlanketBenchmarkError("parity evidence authority changed")
        if sample.position >= 3:
            expected_bit = (
                route_bit[sample.lag_route_sha256s[0]]
                ^ route_bit[sample.lag_route_sha256s[2]]
            )
            if route_bit[sample.target_route_sha256] != expected_bit:
                raise DemandBlanketBenchmarkError("episode violates lag-1 XOR lag-3")

    if (
        validation.fit.selected_lags != (1, 3)
        or not validation.accepted
        or validation.fit.generic_fit.status != seed_body["fit_status"]
        or validation.fit.generic_fit.closure_verified
        != seed_body["closure_verified"]
        or len(split.train_episode_sha256s) != seed_body["train_episode_count"]
        or len(split.calibration_episode_sha256s)
        != seed_body["calibration_episode_count"]
        or len(split.holdout_episode_sha256s) != seed_body["holdout_episode_count"]
    ):
        raise DemandBlanketBenchmarkError("embedded blanket result changed")
    metrics = validation.generic_validation
    expected_metrics = {
        "selected": metrics.selected_metrics.to_dict(),
        "full": metrics.full_metrics.to_dict(),
        "random": metrics.random_metrics.to_dict(),
        "marginal": metrics.marginal_metrics.to_dict(),
        "contiguous_ppm": _ppm_holdout_baseline(corpus, split),
    }
    if seed_body["holdout_metrics"] != expected_metrics:
        raise DemandBlanketBenchmarkError("embedded holdout metrics changed")

    history = tuple(seed_body["unseen_history_route_sha256s"])
    if len(history) != 5 or _digest(list(history)) != seed_body["unseen_history_sha256"]:
        raise DemandBlanketBenchmarkError("unseen history binding changed")
    expected_prediction = alphabet[route_bit[history[-1]] ^ route_bit[history[-3]]]
    if prediction.candidate_route_sha256 != expected_prediction:
        raise DemandBlanketBenchmarkError("unseen parity prediction changed")
    if {
        name: receipt.sha256 for name, receipt in execution_receipts.items()
    } != seed_body["execution_sha256s"]:
        raise DemandBlanketBenchmarkError("execution receipt hash inventory changed")
    expected_execution_records = {
        name: _execution_receipt_record(receipt)
        for name, receipt in execution_receipts.items()
    }
    if seed_body["executions"] != expected_execution_records:
        raise DemandBlanketBenchmarkError("execution summary changed")

    crystal_by_sha = {crystal.sha256: crystal for crystal in crystals}
    edge_by_sha = {edge.sha256: edge for edge in graph_state.edges}
    if set(crystal_by_sha) != {edge.crystal_sha256 for edge in graph_state.edges}:
        raise DemandBlanketBenchmarkError("crystal inventory differs from graph edges")
    value = np.asarray(seed_body["input_tensor"], dtype=np.float64)
    for receipt in execution_receipts.values():
        output = value
        final_crystal: ComputeCrystal | None = None
        for edge_sha256 in receipt.residual.plan.primitive_edge_sha256s:
            edge = edge_by_sha.get(edge_sha256)
            if edge is None:
                raise DemandBlanketBenchmarkError("execution plan names absent edge")
            final_crystal = crystal_by_sha[edge.crystal_sha256]
            output = final_crystal.apply(output)
        if (
            final_crystal is None
            or tensor_sha256(output, final_crystal.output_abi)
            != receipt.residual.output_sha256
            or receipt.verification.verifier_sha256
            != EXACT_OUTPUT_VERIFIER_SHA256
            or not receipt.verification.accepted
        ):
            raise DemandBlanketBenchmarkError("exact output replay failed")

    blanket = execution_receipts["blanket"]
    ppm = execution_receipts["ppm_fallback"]
    ucb = execution_receipts["full_ucb_fallback"]
    guard = execution_receipts["incompatible_guard"]
    if (
        guard.blanket_prediction is None
        or guard.blanket_prediction.candidate_route_sha256 is None
    ):
        raise DemandBlanketBenchmarkError(
            "guard execution carries no incompatible blanket prediction"
        )
    routes = {route.sha256: route for route in graph_state.materialized_routes}
    incompatible = routes[guard.blanket_prediction.candidate_route_sha256]
    expected_checks = {
        "selected_noncontiguous_lags": validation.fit.selected_lags == (1, 3),
        "validation_accepted": validation.accepted,
        "unseen_history_exact": (
            prediction.candidate_route_sha256
            == blanket.selected_prefix_route_sha256
            == expected_prediction
        ),
        "blanket_precedes_ppm": blanket.ppm_prediction is None,
        "ppm_precedes_ucb": (
            ppm.blanket_prediction is not None
            and ppm.blanket_prediction.candidate_route_sha256 is None
            and ppm.ppm_prediction is not None
            and ppm.ppm_prediction.selected_route_sha256
            == ppm.selected_prefix_route_sha256
        ),
        "full_ucb_preserves_all_arms": (
            ucb.blanket_prediction is not None
            and ucb.blanket_prediction.candidate_route_sha256 is None
            and ucb.ppm_prediction is None
            and len(ucb.selection.event.scores) == 2
        ),
        "incompatible_prediction_never_executed": (
            guard.blanket_prediction.candidate_route_sha256
            != guard.selected_prefix_route_sha256
            and not _route_matches_plan(incompatible, guard.residual.plan)
        ),
        "all_outputs_exact": all(
            receipt.verification.accepted for receipt in execution_receipts.values()
        ),
        "all_executions_release_past_compute": all(
            receipt.residual.historical_work_released > 0
            for receipt in execution_receipts.values()
        ),
    }
    if seed_body["checks"] != expected_checks or not all(expected_checks.values()):
        raise DemandBlanketBenchmarkError("execution priority proof changed")
    if (
        blanket.residual.plan.sha256 != seed_body["full_plan_sha256"]
        or guard.residual.plan.sha256 != seed_body["short_plan_sha256"]
        or list(alphabet) != seed_body["route_sha256s"]
    ):
        raise DemandBlanketBenchmarkError("plan or route binding changed")


def verify_report(report: object) -> bool:
    if (
        not isinstance(report, Mapping)
        or set(report) != {"schema", "body", "body_sha256"}
        or report.get("schema") != REPORT_SCHEMA
        or not isinstance(report.get("body"), Mapping)
    ):
        raise DemandBlanketBenchmarkError("invalid benchmark report envelope")
    body = report["body"]
    if require_sha256(report.get("body_sha256"), field="body_sha256") != _digest(body):
        raise DemandBlanketBenchmarkError("benchmark report body hash mismatch")
    expected = {
        "algorithm_sha256",
        "protocol_sha256",
        "parity_law_sha256",
        "exact_output_verifier_sha256",
        "seed_reports",
        "aggregate",
    }
    if set(body) != expected or not isinstance(body.get("seed_reports"), list):
        raise DemandBlanketBenchmarkError("invalid benchmark report body")
    authorities = {
        "algorithm_sha256": ALGORITHM_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "parity_law_sha256": PARITY_LAW_SHA256,
        "exact_output_verifier_sha256": EXACT_OUTPUT_VERIFIER_SHA256,
    }
    if any(body.get(name) != value for name, value in authorities.items()):
        raise DemandBlanketBenchmarkError("benchmark authority hash changed")
    seed_reports = body["seed_reports"]
    for index, seed_report in enumerate(seed_reports):
        if (
            not isinstance(seed_report, Mapping)
            or set(seed_report) != {"schema", "body", "body_sha256"}
            or seed_report.get("schema") != SEED_REPORT_SCHEMA
            or not isinstance(seed_report.get("body"), Mapping)
            or seed_report["body"].get("seed") != index
            or seed_report.get("body_sha256") != _digest(seed_report["body"])
        ):
            raise DemandBlanketBenchmarkError("invalid seed report envelope")
        seed_body = seed_report["body"]
        for field in (
            "receipt_sha256",
            "model_sha256",
            "validation_sha256",
            "prediction_sha256",
            "bank_manifest_sha256",
            "graph_state_sha256",
            "corpus_sha256",
            "split_sha256",
        ):
            require_sha256(seed_body.get(field), field=field)
        if seed_body.get("selected_lags") != [1, 3] or not all(
            seed_body.get("checks", {}).values()
        ):
            raise DemandBlanketBenchmarkError("seed report mechanism check failed")
        _replay_seed_body(seed_body)
    if body.get("aggregate") != _aggregate(seed_reports):
        raise DemandBlanketBenchmarkError("aggregate differs from seed reports")
    return True


def write_report_no_replace(path: Path, report: Mapping[str, object]) -> None:
    """Publish complete canonical bytes atomically without replacing a target."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(report)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=_positive_int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_report(seeds=args.seeds)
    if args.output is not None:
        write_report_no_replace(args.output, report)
    rendered = canonical_json_bytes(report)
    print(json.dumps(json.loads(rendered), allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
