#!/usr/bin/env python3
"""Build a sealed predictive quotient from persistent execution-learning traces."""

from __future__ import annotations

import argparse
from pathlib import Path

from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.execution_learning import ExecutionLearningBank
from immer.runtimes.ooe.identity import canonical_json_bytes, require_sha256
from immer.runtimes.ooe.predictive_quotient import (
    build_predictive_quotient,
    derive_execution_transitions,
    derive_local_agent_transitions,
)

from ooe_controller_language_bootstrap import (
    _existing_directory,
    _output_directory,
    _write_atomic_immutable,
)
from ooe_fertig_action_learning import (
    _assert_output_outside_stores,
    _assert_planned_output_outside_stores,
)


class PredictiveQuotientCliError(RuntimeError):
    """Persistent execution evidence cannot produce the requested quotient."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a persistent ExecutionLearningBank and publish its exact "
            "probabilistic bisimulation quotient."
        )
    )
    parser.add_argument("--learning-store", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=("local-agent", "trace"),
        default="local-agent",
    )
    parser.add_argument(
        "--trace-tail",
        choices=("censored", "terminal"),
        default="censored",
    )
    parser.add_argument("--expected-learning-head-sha256")
    return parser


def run(args: argparse.Namespace):
    root = _existing_directory(args.learning_store, label="learning store")
    output = Path(args.output).expanduser().absolute()
    _assert_planned_output_outside_stores(output, root)
    _output_directory(str(output.parent), label="output parent")
    _assert_output_outside_stores(output, root)
    expected = (
        None
        if args.expected_learning_head_sha256 is None
        else require_sha256(
            args.expected_learning_head_sha256,
            field="expected_learning_head_sha256",
        )
    )
    head = ExecutionLearningBank(CrystalStore(root)).head()
    if expected is not None and head.sha256 != expected:
        raise PredictiveQuotientCliError("execution-learning head is stale")
    if not head.receipts:
        raise PredictiveQuotientCliError("execution-learning bank is empty")
    if args.mode == "local-agent":
        transitions = derive_local_agent_transitions(head.receipts, head.traces)
    else:
        transitions = derive_execution_transitions(
            head.receipts,
            head.traces,
            terminal=args.trace_tail == "terminal",
        )
    quotient = build_predictive_quotient(transitions)
    _write_atomic_immutable(output, canonical_json_bytes(quotient.to_document()))
    return quotient


def main() -> int:
    args = _parser().parse_args()
    try:
        quotient = run(args)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"ooe_predictive_quotient: {exc}") from exc
    print(quotient.sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
