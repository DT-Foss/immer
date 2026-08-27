#!/usr/bin/env python3
"""Seal exact top-two router blankets for persistent execution features."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from immer.runtimes.ooe.controller import CONTROLLER_STATE_NAME, OoeController
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.execution_learning import ExecutionLearningBank
from immer.runtimes.ooe.identity import canonical_json_bytes, require_sha256
from immer.runtimes.ooe.predictive_blanket import (
    RouterDecisionRecord,
    apply_exact_router_blanket,
    replay_exact_router_blanket,
)

from ooe_controller_language_bootstrap import (
    _existing_directory,
    _open_atlas,
    _output_directory,
    _write_atomic_immutable,
)


REPORT_SCHEMA = "immer-ooe-live-router-blanket-report/v1"


class RouterBlanketCliError(RuntimeError):
    """The live controller cannot reproduce its exact router blankets."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _pin(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    try:
        return require_sha256(value, field=field)
    except ValueError as exc:
        raise RouterBlanketCliError(f"{field} is not a SHA-256") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build full-scan-proven exact router closures for every persistent "
            "ExecutionLearning feature."
        )
    )
    parser.add_argument("--controller-store", required=True)
    parser.add_argument("--learning-store", required=True)
    parser.add_argument("--atlas-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--controller-state-name", default=CONTROLLER_STATE_NAME)
    parser.add_argument("--expected-controller-snapshot-sha256")
    parser.add_argument("--expected-learning-head-sha256")
    parser.add_argument("--expected-model-pin-sha256")
    parser.add_argument("--expected-weight-graph-revision-sha256")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    controller_root = _existing_directory(
        args.controller_store,
        label="controller store",
    )
    learning_root = _existing_directory(args.learning_store, label="learning store")
    atlas_root = _existing_directory(args.atlas_root, label="Atlas root")
    output = Path(args.output).expanduser().absolute()
    if any(
        output.resolve(strict=False) == root.resolve(strict=True)
        or root.resolve(strict=True) in output.resolve(strict=False).parents
        for root in (controller_root, learning_root, atlas_root)
    ):
        raise RouterBlanketCliError("output must be outside managed runtime roots")
    _output_directory(str(output.parent), label="output parent")

    expected_controller = _pin(
        args.expected_controller_snapshot_sha256,
        field="expected_controller_snapshot_sha256",
    )
    expected_learning = _pin(
        args.expected_learning_head_sha256,
        field="expected_learning_head_sha256",
    )
    expected_model = _pin(
        args.expected_model_pin_sha256,
        field="expected_model_pin_sha256",
    )
    expected_weight = _pin(
        args.expected_weight_graph_revision_sha256,
        field="expected_weight_graph_revision_sha256",
    )

    atlas = _open_atlas(atlas_root)
    controller_store = CrystalStore(controller_root)
    controller = OoeController.restore(
        crystal_store=controller_store,
        name=args.controller_state_name,
        atlas_revision_verifier=atlas.contains_revision,
        expected_model_pin_sha256=expected_model,
        expected_weight_graph_revision_sha256=expected_weight,
    )
    snapshot = controller_store.restore_state(args.controller_state_name)
    snapshot_sha256 = hashlib.sha256(snapshot).hexdigest()
    if controller.snapshot_bytes() != snapshot:
        raise RouterBlanketCliError("controller differs from its persistent snapshot")
    if expected_controller is not None and snapshot_sha256 != expected_controller:
        raise RouterBlanketCliError("controller snapshot is stale")
    if expected_model is not None and controller.model_pin_sha256 != expected_model:
        raise RouterBlanketCliError("controller model pin is stale")
    if (
        expected_weight is not None
        and controller.weight_graph_revision_sha256 != expected_weight
    ):
        raise RouterBlanketCliError("controller weight graph is stale")

    learning = ExecutionLearningBank(CrystalStore(learning_root)).head()
    if expected_learning is not None and learning.sha256 != expected_learning:
        raise RouterBlanketCliError("execution-learning head is stale")
    receipts = []
    full_distances = 0
    sparse_distances = 0
    for learning_receipt in learning.receipts:
        feature = learning_receipt.feature
        blanket = controller.build_router_blanket(feature)
        replay_exact_router_blanket(controller.router, feature, blanket)

        controller.router.reset_distance_evaluations()
        full = controller.router.decision(feature.sketch_array)
        full_distances += controller.router.distance_evaluations
        controller.router.reset_distance_evaluations()
        sparse = apply_exact_router_blanket(controller.router, feature, blanket)
        sparse_distances += controller.router.distance_evaluations
        if full != sparse or RouterDecisionRecord.from_decision(full) != (
            blanket.full_decision
        ):
            raise RouterBlanketCliError("full and sparse decisions differ")
        receipts.append(
            {
                "blanket": blanket.to_document(),
                "blanket_sha256": blanket.sha256,
                "candidate_count": len(blanket.candidate_closure),
                "feature_receipt_sha256": feature.sha256,
                "site_identity_sha256": feature.site_identity.sha256,
                "temporal_index": feature.temporal_index,
            }
        )
    if not receipts or full_distances <= 0 or not 0 < sparse_distances <= full_distances:
        raise RouterBlanketCliError("router blanket distance accounting is invalid")
    body = {
        "atlas_graph_revision_sha256": controller.atlas_graph_revision_sha256,
        "controller_snapshot_sha256": snapshot_sha256,
        "distance_reduction": {
            "denominator": full_distances,
            "numerator": full_distances - sparse_distances,
        },
        "feature_count": len(receipts),
        "format": REPORT_SCHEMA,
        "full_distance_evaluations": full_distances,
        "label_count": len(controller.router.labels),
        "learning_head_sha256": learning.sha256,
        "learning_stream_head_sha256": learning.stream_head_sha256,
        "model_pin_sha256": controller.model_pin_sha256,
        "receipts": receipts,
        "router_calibration_sha256": controller.router.calibration_sha256,
        "router_centroid_universe_sha256": (
            controller.router.centroid_universe_sha256
        ),
        "sparse_distance_evaluations": sparse_distances,
        "weight_graph_revision_sha256": controller.weight_graph_revision_sha256,
    }
    report = {"body": body, "schema": REPORT_SCHEMA, "sha256": _digest(body)}
    _write_atomic_immutable(output, canonical_json_bytes(report))
    return report


def main() -> int:
    args = _parser().parse_args()
    try:
        report = run(args)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"ooe_router_blanket: {exc}") from exc
    print(report["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
