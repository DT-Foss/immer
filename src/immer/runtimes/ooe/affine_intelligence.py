"""Mechanism benchmark for exact guarded affine-monoid intelligence."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from typing import Any

from .affine_monoid import (
    AffineMonoidBank,
    AffineMonoidRuntime,
    AffineProgram,
    build_anbncn_machine,
    build_decimal_horner_machine,
    build_fingerprint_machine,
    build_group_map_bridge,
    build_stack_machine,
    compose_actions,
    verify_fingerprint_equality,
)
from .algebra_agents import (
    AlgebraRouterBank,
    AlgebraRouterState,
    ExactVerifierArtifact,
    HeterogeneousProgramEnsemble,
    OperatorAlgebraCandidate,
    ParallelProgramLane,
    VerifierBoundOutcome,
)
from .identity import canonical_json_bytes


AFFINE_MONOID_INTELLIGENCE_SCHEMA = "immer-ooe-affine-monoid-intelligence/v1"


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stack_experiment() -> dict[str, Any]:
    capacity = 64
    depth = 53
    stack = build_stack_machine(
        capacity=capacity,
        symbols=tuple(range(1, capacity + 2)),
    )
    symbols = tuple(range(1, depth + 1))
    program = AffineProgram(
        stack.schema,
        tuple(stack.push(symbol) for symbol in symbols),
        "StackUnseenDepth53",
    )
    execution = AffineMonoidRuntime.execute(program)
    state = execution.state
    recovered = []
    for _ in symbols:
        recovered.append(stack.top(state))
        state = stack.action("pop").apply(stack.schema, state)
    capacity_program = AffineProgram(
        stack.schema,
        tuple(stack.push(index + 1) for index in range(capacity)),
        "StackCapacity64",
    )
    full = AffineMonoidRuntime.execute(capacity_program).state
    overflow = stack.push(capacity + 1).apply(stack.schema, full)
    return {
        "capacity": capacity,
        "tested_depth": depth,
        "buried_lifo_exact": recovered == list(reversed(symbols)),
        "empty_after_pop": state == stack.schema.state(),
        "capacity_state_live": not stack.schema.is_dead(full),
        "capacity_plus_one_dead": stack.schema.is_dead(overflow),
        "program_sha256": program.sha256,
        "execution_receipt_sha256": execution.execution_receipt.sha256,
        "work_units": execution.execution_receipt.work_units,
    }


def _counter_experiment() -> dict[str, Any]:
    machine = build_anbncn_machine(max_count=128)
    counts = (1, 12, 47, 100)
    accepted = tuple(
        machine.accepts(machine.run("a" * count + "b" * count + "c" * count).state)
        for count in counts
    )
    placebos = (
        "",
        "acb",
        "abcabc",
        "aabccb",
        "aaabbbcc",
        "aabbbccc",
        "aaabbbcccc",
    )
    rejected = tuple(not machine.accepts(machine.run(value).state) for value in placebos)
    return {
        "unseen_counts": list(counts),
        "accepted": sum(accepted),
        "accepted_total": len(accepted),
        "order_and_count_placebos_rejected": sum(rejected),
        "placebo_total": len(rejected),
        "schema_sha256": machine.schema.sha256,
    }


def _fingerprint_experiment() -> dict[str, Any]:
    machine = build_fingerprint_machine(max_length=32, base=2, moduli=(3, 5))
    unequal_length = machine.compare(b"\x00", b"\x00\x0d")
    collision_left = b"\x00"
    collision_right = b"\x0f"
    collision = machine.compare(collision_left, collision_right)
    collision_verification = verify_fingerprint_equality(
        machine,
        collision,
        left=collision_left,
        right=collision_right,
        verifier_sha256=_digest({"kind": "exact-byte-verifier"}),
    )
    exact = machine.compare(b"same-sequence", b"same-sequence")
    exact_verification = verify_fingerprint_equality(
        machine,
        exact,
        left=b"same-sequence",
        right=b"same-sequence",
        verifier_sha256=_digest({"kind": "exact-byte-verifier"}),
    )
    return {
        "unequal_length_candidate": machine.candidate_equal(unequal_length.state),
        "modular_collision_candidate": machine.candidate_equal(collision.state),
        "modular_collision_exact": collision_verification.exact_equal,
        "exact_candidate": machine.candidate_equal(exact.state),
        "exact_verified": exact_verification.exact_equal,
        "collision_verification_sha256": collision_verification.sha256,
        "exact_verification_sha256": exact_verification.sha256,
    }


def _decimal_experiment() -> tuple[dict[str, Any], AffineProgram, object]:
    parser = build_decimal_horner_machine(max_digits=128)
    examples = ("12", "21", "90", "100", "+100", "-100", "0")
    values = {text: parser.value(parser.parse(text).state) for text in examples}
    digits = "9" * 128
    maximum = parser.parse(digits)
    over = parser.parse("1" * 129)
    malformed = ("", "+", "-", "00", "01", "1+2", " 12", "12 ")
    rejected = sum(parser.schema.is_dead(parser.parse(text).state) for text in malformed)
    # Persist one exact nontrivial parse program and its whole execution as one bundle.
    exact_program = AffineProgram(
        parser.schema,
        (
            parser.action("digit:1"),
            parser.action("digit:0:next"),
            parser.action("digit:0:next"),
            parser.action("finish"),
        ),
        "Decimal100Persistence",
    )
    exact_execution = AffineMonoidRuntime.execute(exact_program)
    return (
        {
            "examples": values,
            "digits_128_exact": parser.value(maximum.state) == int(digits),
            "digits_129_dead": parser.schema.is_dead(over.state),
            "malformed_rejected": rejected,
            "malformed_total": len(malformed),
            "maximum_execution_sha256": maximum.execution_receipt.sha256,
        },
        exact_program,
        exact_execution,
    )


def _group_experiment() -> dict[str, Any]:
    additive = build_group_map_bridge("additive", max_abs_bits=256)
    fused, composition = compose_actions(
        additive.schema,
        additive.action(12),
        additive.action(21),
        name="Add12Then21",
    )
    future = (-(10**30), -7, 0, 90, 10**30)
    additive_exact = all(
        additive.value(
            fused.apply(additive.schema, additive.schema.state((value, 0)))
        )
        == value + 33
        for value in future
    )
    multiplicative = build_group_map_bridge(
        "multiplicative", initial=2, max_abs_bits=128
    )
    product, _ = compose_actions(
        multiplicative.schema,
        multiplicative.action(12),
        multiplicative.action(21),
        name="Mul12Then21",
    )
    product_value = multiplicative.value(
        product.apply(multiplicative.schema, multiplicative.schema.state())
    )
    cyclic = build_group_map_bridge("cyclic", modulus=90, initial=89)
    wrapped, _ = compose_actions(
        cyclic.schema,
        cyclic.action(12),
        cyclic.action(100),
        name="Z90Add112",
    )
    cyclic_value = cyclic.value(
        wrapped.apply(cyclic.schema, cyclic.schema.state())
    )
    return {
        "additive_future_inputs": len(future),
        "additive_future_exact": additive_exact,
        "multiplicative_result": product_value,
        "multiplicative_expected": 504,
        "cyclic_result": cyclic_value,
        "cyclic_expected": 21,
        "composition_receipt_sha256": composition.sha256,
    }


def _agent_programs() -> dict[str, AffineProgram]:
    stack = build_stack_machine(capacity=4, symbols=(1, 2, 3))
    fingerprint = build_fingerprint_machine(
        max_length=8,
        base=17,
        moduli=(101, 103),
    )
    decimal = build_decimal_horner_machine(max_digits=8)
    return {
        "stack": AffineProgram(
            stack.schema,
            (stack.push(1), stack.push(2), stack.action("peek")),
            "StackAgentProgram",
        ),
        "fingerprint": AffineProgram(
            fingerprint.schema,
            (
                fingerprint.left(ord("x")),
                fingerprint.separator(),
                fingerprint.right(ord("x")),
                fingerprint.finish(),
            ),
            "FingerprintAgentProgram",
        ),
        "decimal": AffineProgram(
            decimal.schema,
            (
                decimal.action("digit:1"),
                decimal.action("digit:2"),
                decimal.action("digit:3"),
                decimal.action("finish"),
            ),
            "DecimalAgentProgram",
        ),
    }


def _agent_candidates(
    programs: dict[str, AffineProgram],
) -> tuple[OperatorAlgebraCandidate, ...]:
    descriptors = {
        "stack": (0, 0),
        "fingerprint": (1, 0),
        "decimal": (2, 0),
    }
    return tuple(
        OperatorAlgebraCandidate(
            family=family,
            program=programs[family],
            verifier_sha256=_digest({"family": family, "kind": "verifier"}),
            evidence_sha256s=tuple(
                sorted(
                    (
                        _digest({"family": family, "replica": 0}),
                        _digest({"family": family, "replica": 1}),
                    )
                )
            ),
            behavior_descriptor=descriptors[family],
            objectives=(1.0, 1.0),
        )
        for family in ("stack", "fingerprint", "decimal")
    )


def _agent_step(
    state: AlgebraRouterState,
    *,
    context: str,
    winner: str,
    nonce: int,
) -> AlgebraRouterState:
    candidate, selection, advanced = state.choose(context)
    success = candidate.family == winner
    outcome = VerifierBoundOutcome.issue(
        selection,
        candidate,
        receipt_payload=canonical_json_bytes(
            {
                "candidate_sha256": candidate.sha256,
                "nonce": nonce,
                "success": success,
                "verifier_sha256": candidate.verifier_sha256,
            }
        ),
        success=success,
    )
    updated, _receipt = advanced.observe(selection, outcome)
    return updated


def _train_agent(
    state: AlgebraRouterState,
    winners: dict[str, str],
    *,
    rounds: int,
    nonce_start: int = 0,
) -> AlgebraRouterState:
    nonce = nonce_start
    for _ in range(rounds):
        for context, winner in sorted(winners.items()):
            state = _agent_step(
                state,
                context=context,
                winner=winner,
                nonce=nonce,
            )
            nonce += 1
    return state


def _evaluate_agent(
    state: AlgebraRouterState,
    winners: dict[str, str],
    *,
    rounds: int,
) -> tuple[int, AlgebraRouterState]:
    correct = 0
    for _ in range(rounds):
        for context, winner in sorted(winners.items()):
            candidate, _selection, state = state.choose(context)
            correct += candidate.family == winner
    return correct, state


def _algebra_agent_experiment() -> dict[str, Any]:
    programs = _agent_programs()
    router = AlgebraRouterState.bootstrap(
        _agent_candidates(programs),
        seed_sha256=_digest({"kind": "algebra-router-seed"}),
        bin_counts=(3, 2),
        objective_count=2,
        max_elites_per_cell=2,
    )
    winners = {
        "context-stack": "stack",
        "context-fingerprint": "fingerprint",
        "context-decimal": "decimal",
    }
    trained = _train_agent(router, winners, rounds=80)
    exact_correct, exact_state = _evaluate_agent(
        AlgebraRouterState.from_bytes(trained.to_bytes()),
        winners,
        rounds=40,
    )
    shuffled = {
        "context-stack": "decimal",
        "context-fingerprint": "stack",
        "context-decimal": "fingerprint",
    }
    placebo_correct, _ = _evaluate_agent(
        AlgebraRouterState.from_bytes(trained.to_bytes()),
        shuffled,
        rounds=30,
    )

    regime = _train_agent(
        router,
        {"regime-context": "stack"},
        rounds=45,
    )
    regime = _train_agent(
        regime,
        {"regime-context": "decimal"},
        rounds=140,
        nonce_start=45,
    )
    recovered, regime = _evaluate_agent(
        regime,
        {"regime-context": "decimal"},
        rounds=30,
    )

    fingerprint_verifier = _digest({"kind": "ensemble-fingerprint-verifier"})
    verifier_artifact = ExactVerifierArtifact(
        fingerprint_verifier,
        canonical_json_bytes(
            {
                "exact_equal": True,
                "left_sha256": _digest({"value": "x"}),
                "right_sha256": _digest({"value": "x"}),
            }
        ),
    )
    ensemble = HeterogeneousProgramEnsemble(
        "ThreeIndependentABIs",
        (
            ParallelProgramLane(
                "decimal",
                programs["decimal"],
                programs["decimal"].schema.state(),
            ),
            ParallelProgramLane(
                "fingerprint",
                programs["fingerprint"],
                programs["fingerprint"].schema.state(),
                (fingerprint_verifier,),
            ),
            ParallelProgramLane(
                "stack",
                programs["stack"],
                programs["stack"].schema.state(),
            ),
        ),
    )
    ensemble_result = ensemble.execute(
        {"fingerprint": (verifier_artifact,)},
        max_workers=3,
    )
    with tempfile.TemporaryDirectory() as temporary:
        bank = AlgebraRouterBank(temporary)
        publication = bank.publish("primary", exact_state)
        restored = bank.restore("primary")
        persistence = {
            "changed": publication.changed,
            "restored_exact": restored.to_bytes() == exact_state.to_bytes(),
        }
    return {
        "context_correct": exact_correct,
        "context_total": 120,
        "context_accuracy": exact_correct / 120,
        "shuffled_placebo_correct": placebo_correct,
        "shuffled_placebo_total": 90,
        "shuffled_placebo_accuracy": placebo_correct / 90,
        "regime_recovered": recovered,
        "regime_total": 30,
        "regime_accuracy": recovered / 30,
        "archive_occupied_cells": trained.archive.occupied_cells,
        "router_sha256": exact_state.sha256,
        "router_restored_exact": persistence["restored_exact"],
        "ensemble_heterogeneous": ensemble.heterogeneous,
        "ensemble_lane_count": len(ensemble.lanes),
        "ensemble_total_work_units": ensemble_result.receipt.total_work_units,
        "ensemble_receipt_sha256": ensemble_result.receipt.sha256,
        "ensemble_replay_exact": (
            ensemble_result.to_bytes()
            == type(ensemble_result).from_bytes(ensemble_result.to_bytes()).to_bytes()
        ),
    }


def run_affine_monoid_intelligence_benchmark() -> dict[str, Any]:
    stack = _stack_experiment()
    counter = _counter_experiment()
    fingerprint = _fingerprint_experiment()
    decimal, decimal_program, decimal_execution = _decimal_experiment()
    groups = _group_experiment()
    algebra_agent = _algebra_agent_experiment()
    with tempfile.TemporaryDirectory() as temporary:
        bank = AffineMonoidBank(Path(temporary) / "affine-bank")
        bundle, publication = bank.capture_and_publish(
            decimal_program,
            decimal_execution,
        )
        restored = bank.restore_bundle(bundle.sha256)
        persistence = {
            "bundle_sha256": bundle.sha256,
            "publication_changed": publication.changed,
            "restored_exact": restored.to_bytes() == bundle.to_bytes(),
            "single_payload_bytes": len(bundle.to_bytes()),
        }
    body = {
        "stack": stack,
        "counter_dfa": counter,
        "fingerprint": fingerprint,
        "decimal_horner": decimal,
        "group_maps": groups,
        "persistence": persistence,
        "algebra_agent": algebra_agent,
    }
    headline = {
        "stack_depth_53_exact": stack["buried_lifo_exact"],
        "stack_capacity_boundary_exact": (
            stack["capacity_state_live"] and stack["capacity_plus_one_dead"]
        ),
        "anbncn_unseen_exact": counter["accepted"] == counter["accepted_total"],
        "anbncn_placebos_rejected": (
            counter["order_and_count_placebos_rejected"] == counter["placebo_total"]
        ),
        "fingerprint_collision_contained": (
            fingerprint["modular_collision_candidate"]
            and not fingerprint["modular_collision_exact"]
        ),
        "decimal_128_exact": decimal["digits_128_exact"],
        "group_composition_exact": (
            groups["additive_future_exact"]
            and groups["multiplicative_result"] == groups["multiplicative_expected"]
            and groups["cyclic_result"] == groups["cyclic_expected"]
        ),
        "atomic_bundle_restored": persistence["restored_exact"],
        "contextual_algebra_learned": (
            algebra_agent["context_accuracy"] >= 116 / 120
        ),
        "shuffled_context_placebo_rejected": (
            algebra_agent["shuffled_placebo_accuracy"] <= 4 / 90
        ),
        "algebra_regime_recovered": algebra_agent["regime_accuracy"] >= 27 / 30,
        "heterogeneous_ensemble_exact": (
            algebra_agent["ensemble_heterogeneous"]
            and algebra_agent["ensemble_replay_exact"]
        ),
    }
    report_body = {"body": body, "headline": headline}
    return {
        **report_body,
        "schema": AFFINE_MONOID_INTELLIGENCE_SCHEMA,
        "sha256": _digest(report_body),
    }


__all__ = [
    "AFFINE_MONOID_INTELLIGENCE_SCHEMA",
    "run_affine_monoid_intelligence_benchmark",
]
