"""Mechanism benchmark for continuous PS-Lifted Markov intelligence."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .consensus import barbell_adjacency, complete_adjacency
from .identity import canonical_json_bytes
from .reservoir import PSLiftedReservoirAgent, PSLiftedReservoirConfig


RESERVOIR_INTELLIGENCE_SCHEMA = "immer-ooe-reservoir-intelligence/v1"


def _hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _target(bit: float) -> np.ndarray:
    return np.asarray([bit < 0.0, bit > 0.0], dtype=np.float64)


def _train_delay(
    agent: PSLiftedReservoirAgent,
    sequence: np.ndarray,
    *,
    delay: int,
    replica: int,
    placebo: bool,
) -> None:
    agent.reset_state()
    labels = np.roll(sequence, 31) if placebo else sequence
    for index, bit in enumerate(sequence):
        value = np.asarray([bit], dtype=np.float64)
        if index < delay:
            agent.advance(value)
            continue
        agent.observe(
            value,
            _target(float(labels[index - delay])),
            evidence_sha256=_hash(
                {
                    "delay": delay,
                    "index": index,
                    "placebo": placebo,
                    "replica": replica,
                }
            ),
            solve=False,
        )
    agent.solve_readout()


def _accuracy(
    agent: PSLiftedReservoirAgent,
    sequence: np.ndarray,
    *,
    delay: int,
) -> float:
    agent.reset_state()
    correct = 0
    total = 0
    for index, bit in enumerate(sequence):
        prediction = agent.predict(np.asarray([bit], dtype=np.float64))
        if index < delay:
            continue
        correct += int(prediction.selected == int(sequence[index - delay] > 0.0))
        total += 1
    return correct / total


def run_reservoir_intelligence_benchmark(
    *,
    seed: int = 20_260_826,
    replicas: int = 8,
    train_steps_per_replica: int = 500,
    test_steps: int = 1_000,
    delay: int = 7,
) -> dict[str, Any]:
    """Learn a delayed latent bit from distributed raw sequence fragments.

    The target is deliberately independent of the current input.  A no-memory
    learner therefore sits at chance, while the fixed PS-Lifted recurrence can
    retain the missing temporal state.  Replicas exchange only ridge sufficient
    statistics; shuffled temporal labels form the matched placebo.
    """

    if (
        isinstance(replicas, bool)
        or not isinstance(replicas, int)
        or replicas < 4
        or replicas % 2
        or replicas > 32
    ):
        raise ValueError("replicas must be an even integer in [4, 32]")
    for value, name, lower, upper in (
        (train_steps_per_replica, "train_steps_per_replica", 64, 100_000),
        (test_steps, "test_steps", 64, 100_000),
        (delay, "delay", 2, 64),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not lower <= value <= upper
        ):
            raise ValueError(f"{name} must lie in [{lower}, {upper}]")
    rng = np.random.default_rng(seed)
    training = tuple(
        rng.choice(
            np.asarray([-1.0, 1.0]),
            size=train_steps_per_replica,
        )
        for _ in range(replicas)
    )
    test = rng.choice(np.asarray([-1.0, 1.0]), size=test_steps)
    config = PSLiftedReservoirConfig(
        input_size=1,
        output_size=2,
        nodes=32,
        seed=42,
        spectral_radius=0.95,
        input_scale=0.2,
        leak_rate=1.0,
        ridge=1e-3,
    )
    learned = tuple(PSLiftedReservoirAgent(config) for _ in range(replicas))
    placebo = tuple(PSLiftedReservoirAgent(config) for _ in range(replicas))
    for index, sequence in enumerate(training):
        _train_delay(
            learned[index],
            sequence,
            delay=delay,
            replica=index,
            placebo=False,
        )
        _train_delay(
            placebo[index],
            sequence,
            delay=delay,
            replica=index,
            placebo=True,
        )

    local_accuracies = tuple(
        _accuracy(agent, test, delay=delay) for agent in learned
    )
    graph_barbell = barbell_adjacency(replicas // 2, replicas // 2)
    graph_complete = complete_adjacency(replicas)
    fused_barbell = PSLiftedReservoirAgent.fuse_ps_lifted(
        learned,
        graph_barbell,
        tolerance=1e-8,
        max_rounds=4_096,
        topology="reservoir-delay-barbell",
    )
    fused_complete = PSLiftedReservoirAgent.fuse_ps_lifted(
        learned,
        graph_complete,
        tolerance=1e-8,
        max_rounds=4_096,
        topology="reservoir-delay-complete",
    )
    fused_placebo = PSLiftedReservoirAgent.fuse_ps_lifted(
        placebo,
        graph_barbell,
        tolerance=1e-8,
        max_rounds=4_096,
        topology="reservoir-delay-shuffled-placebo",
    )
    barbell_accuracy = _accuracy(fused_barbell.agent, test, delay=delay)
    complete_accuracy = _accuracy(fused_complete.agent, test, delay=delay)
    placebo_accuracy = _accuracy(fused_placebo.agent, test, delay=delay)
    no_memory_accuracy = float(
        np.mean((test[delay:] > 0.0) == (test[:-delay] > 0.0))
    )
    local_mean = float(np.mean(local_accuracies))
    body = {
        "config": config.to_dict(),
        "delay": delay,
        "evidence_events": fused_barbell.agent.sample_count,
        "fused": {
            "accuracy": barbell_accuracy,
            "receipt_sha256": fused_barbell.receipt.sha256,
            "rounds": fused_barbell.receipt.consensus.rounds,
            "statistics_sha256": fused_barbell.agent.statistics_sha256,
            "topology": "barbell",
        },
        "local": {
            "accuracies": list(local_accuracies),
            "mean_accuracy": local_mean,
        },
        "no_memory": {
            "accuracy": no_memory_accuracy,
            "state_dimensions": 0,
        },
        "placebo": {
            "accuracy": placebo_accuracy,
            "receipt_sha256": fused_placebo.receipt.sha256,
            "temporal_labels": "circularly-shuffled-31",
        },
        "replicas": replicas,
        "seed": seed,
        "test_steps": test_steps,
        "topology_shift": {
            "accuracy": complete_accuracy,
            "complete_receipt_sha256": fused_complete.receipt.sha256,
            "complete_rounds": fused_complete.receipt.consensus.rounds,
            "barbell_rounds": fused_barbell.receipt.consensus.rounds,
            "readout_max_absolute_delta": float(
                np.max(
                    np.abs(
                        fused_barbell.agent.readout
                        - fused_complete.agent.readout
                    )
                )
            ),
        },
        "train_steps_per_replica": train_steps_per_replica,
    }
    headline = {
        "fused_accuracy": barbell_accuracy,
        "fused_minus_local": barbell_accuracy - local_mean,
        "fused_minus_no_memory": barbell_accuracy - no_memory_accuracy,
        "fused_minus_placebo": barbell_accuracy - placebo_accuracy,
        "local_mean_accuracy": local_mean,
        "no_memory_accuracy": no_memory_accuracy,
        "placebo_accuracy": placebo_accuracy,
        "topology_shift_accuracy": complete_accuracy,
    }
    report_body = {"body": body, "headline": headline}
    return {
        **report_body,
        "schema": RESERVOIR_INTELLIGENCE_SCHEMA,
        "sha256": _hash(report_body),
    }


__all__ = [
    "RESERVOIR_INTELLIGENCE_SCHEMA",
    "run_reservoir_intelligence_benchmark",
]
