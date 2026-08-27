from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import unittest

import numpy as np

from immer.runtimes.ooe.agents import AttractorRouter
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.predictive_blanket import (
    ExactRouterBlanketCoverageError,
    ExactRouterBlanketIntegrityError,
    ExactRouterBlanketReceipt,
    ExactRouterBlanketStaleError,
    RouterDecisionRecord,
    apply_exact_router_blanket,
    build_exact_router_blanket,
    replay_exact_router_blanket,
    verify_exact_router_blanket,
)
from immer.runtimes.ooe.qwen_bridge import (
    ACTION_SCHEMA_SHA256,
    QwenOoeFeatureReceipt,
    feature_schema_sha256,
)
from immer.runtimes.qwen3_8.semantic_atlas import GraphRevision, ProbeIdentity


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _feature(
    sketch: tuple[float, ...],
    *,
    temporal_index: int = 0,
    coordinate: str = "input-site",
    atlas: GraphRevision | None = None,
) -> QwenOoeFeatureReceipt:
    return QwenOoeFeatureReceipt(
        temporal_index=temporal_index,
        measurement_sha256=_hash(f"measurement-{temporal_index}-{coordinate}"),
        model_pin_sha256=_hash("model"),
        weight_coordinate_sha256=_hash(coordinate),
        weight_graph_revision=GraphRevision(7, _hash("weight-graph")),
        atlas_graph_revision=atlas or GraphRevision(11, _hash("atlas-graph")),
        probe=ProbeIdentity(
            question_sha256=_hash(f"question-{temporal_index}"),
            token_sha256=_hash(f"tokens-{temporal_index}"),
            family_sha256=_hash("family"),
            label_source_sha256=_hash("labels"),
        ),
        feature_schema_sha256=feature_schema_sha256(len(sketch)),
        action_schema_sha256=ACTION_SCHEMA_SHA256,
        verifier_sha256s=(_hash("verifier"),),
        evidence_sha256s=(_hash("evidence"),),
        o1_surprise=0.25,
        o1_learning_progress=0.5,
        feature_sketch=sketch,
    )


def _calibrated_router(
    feature: QwenOoeFeatureReceipt,
    *,
    labels: int = 7,
) -> AttractorRouter:
    router = AttractorRouter(radius=2.0, min_margin=0.0)
    input_label = feature.site_identity.sha256
    centroids: list[tuple[str, np.ndarray]] = [
        (input_label, feature.sketch_array.copy())
    ]
    for index in range(1, labels):
        value = feature.sketch_array.copy()
        value[0] += 0.1 * index
        centroids.append((_hash(f"router-site-{index}"), value))
    for label, centroid in reversed(centroids):
        router.observe(label, centroid)
    router.calibrate(
        centroids,
        radius_quantile=1.0,
        margin_quantile=0.0,
        radius_ceiling=2.0,
    )
    return router


class ExactRouterBlanketTests(unittest.TestCase):
    def test_full_scan_sparse_replay_restart_and_distance_reduction(self) -> None:
        feature = _feature((0.0,) * 8)
        router = _calibrated_router(feature, labels=9)
        blanket = build_exact_router_blanket(router, feature)

        self.assertEqual(blanket.full_decision, blanket.sparse_decision)
        self.assertEqual(len(blanket.label_universe), 9)
        self.assertEqual(len(blanket.candidate_closure), 2)
        self.assertIn(feature.site_identity.sha256, blanket.candidate_closure)
        self.assertTrue(verify_exact_router_blanket(router, feature, blanket))
        restored = ExactRouterBlanketReceipt.from_bytes(blanket.to_bytes())
        self.assertEqual(restored, blanket)
        self.assertEqual(restored.to_bytes(), blanket.to_bytes())

        router.reset_distance_evaluations()
        full = router.decision(feature.sketch_array)
        full_evaluations = router.distance_evaluations
        router.reset_distance_evaluations()
        sparse = apply_exact_router_blanket(router, feature, restored)
        sparse_evaluations = router.distance_evaluations
        self.assertEqual(sparse, full)
        self.assertEqual(full_evaluations, 9)
        self.assertEqual(sparse_evaluations, 2)
        self.assertLess(sparse_evaluations, full_evaluations)

    def test_skipped_global_runner_up_cannot_publish_false_accept(self) -> None:
        feature = _feature((0.0,) * 8)
        input_label = feature.site_identity.sha256
        competitor = next(
            _hash(f"competitor-{index}")
            for index in range(10_000)
            if _hash(f"competitor-{index}") > input_label
        )
        far = _hash("far-site")
        router = AttractorRouter(radius=2.0, min_margin=0.0)
        input_centroid = np.array((-0.01,) + (0.0,) * 7)
        competitor_centroid = np.array((0.01,) + (0.0,) * 7)
        far_centroid = np.array((1.0,) + (0.0,) * 7)
        centroids = (
            (input_label, input_centroid),
            (competitor, competitor_centroid),
            (far, far_centroid),
        )
        for label, centroid in centroids:
            router.observe(label, centroid)
        router.calibrate(
            (
                (input_label, np.array((-0.02,) + (0.0,) * 7)),
                (competitor, np.array((0.02,) + (0.0,) * 7)),
                (far, np.array((1.01,) + (0.0,) * 7)),
            ),
            radius_quantile=1.0,
            margin_quantile=0.0,
        )

        full = router.decision(feature.sketch_array)
        omitted = tuple(sorted((input_label, far)))
        unsafe_sparse = router.decision_for_candidates(
            feature.sketch_array,
            omitted,
        )
        self.assertEqual(full.reason, "ambiguous-margin")
        self.assertEqual(unsafe_sparse.reason, "accepted")
        with self.assertRaisesRegex(
            ExactRouterBlanketCoverageError,
            "global runner-up",
        ):
            build_exact_router_blanket(
                router,
                feature,
                candidate_labels=omitted,
            )

    def test_wrong_feature_stale_router_and_tamper_fail_before_distance(self) -> None:
        feature = _feature((0.0,) * 8)
        router = _calibrated_router(feature, labels=5)
        blanket = build_exact_router_blanket(router, feature)

        other_feature = replace(feature, temporal_index=feature.temporal_index + 1)
        router.reset_distance_evaluations()
        with self.assertRaisesRegex(
            ExactRouterBlanketStaleError,
            "another exact",
        ):
            apply_exact_router_blanket(router, other_feature, blanket)
        self.assertEqual(router.distance_evaluations, 0)

        tampered = json.loads(blanket.to_bytes())
        tampered["body"]["candidate_closure"].pop()
        with self.assertRaisesRegex(
            ExactRouterBlanketIntegrityError,
            "SHA-256 mismatch",
        ):
            ExactRouterBlanketReceipt.from_bytes(canonical_json_bytes(tampered))

        removable = next(
            label
            for label in blanket.label_universe
            if label not in blanket.candidate_closure
        )
        resealed_wrong_universe = replace(
            blanket,
            label_universe=tuple(
                label for label in blanket.label_universe if label != removable
            ),
            seal_sha256="",
        )
        router.reset_distance_evaluations()
        with self.assertRaisesRegex(
            ExactRouterBlanketStaleError,
            "centroid universe",
        ):
            apply_exact_router_blanket(
                router,
                feature,
                resealed_wrong_universe,
            )
        self.assertEqual(router.distance_evaluations, 0)

        router.observe(router.labels[0], feature.sketch_array + 0.001)
        router.reset_distance_evaluations()
        with self.assertRaisesRegex(
            ExactRouterBlanketStaleError,
            "calibration",
        ):
            apply_exact_router_blanket(router, feature, blanket)
        self.assertEqual(router.distance_evaluations, 0)

        router.calibrate(
            tuple((label, router.centroid(label)) for label in router.labels),
            radius_quantile=1.0,
            margin_quantile=0.0,
        )
        router.observe(_hash("new-router-site"), np.ones(8))
        router.calibrate(
            tuple((label, router.centroid(label)) for label in router.labels),
            radius_quantile=1.0,
            margin_quantile=0.0,
        )
        with self.assertRaises(ExactRouterBlanketStaleError):
            apply_exact_router_blanket(router, feature, blanket)

    def test_one_label_infinity_is_none_and_never_nonfinite_json(self) -> None:
        feature = _feature((0.125,) * 8)
        router = _calibrated_router(feature, labels=1)
        blanket = build_exact_router_blanket(router, feature)
        raw = blanket.to_bytes()
        document = json.loads(raw)
        decision = document["body"]["full_decision"]

        self.assertIsNone(decision["second_distance"])
        self.assertIsNone(decision["margin"])
        self.assertNotIn(b"Infinity", raw)
        self.assertNotIn(b"NaN", raw)
        applied = apply_exact_router_blanket(router, feature, blanket)
        self.assertTrue(math.isinf(applied.second_distance))
        self.assertTrue(math.isinf(applied.margin))
        self.assertEqual(
            RouterDecisionRecord.from_decision(applied),
            blanket.full_decision,
        )
        self.assertEqual(replay_exact_router_blanket(router, feature, blanket), applied)

    def test_candidate_labels_are_strict_sorted_unique_known_values(self) -> None:
        router = AttractorRouter(radius=1.0)
        router.observe("a", np.zeros(8))
        router.observe("b", np.ones(8))
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            router.decision_for_candidates(np.zeros(8), ("b", "a"))
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            router.decision_for_candidates(np.zeros(8), ("a", "a"))
        with self.assertRaisesRegex(KeyError, "unknown attractor"):
            router.decision_for_candidates(np.zeros(8), ("missing",))


if __name__ == "__main__":
    unittest.main()
