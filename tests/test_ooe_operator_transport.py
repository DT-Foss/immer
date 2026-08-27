from __future__ import annotations

import base64
import hashlib
import json
import unittest

import numpy as np

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.operator_transport import (
    QWEN_PREFIX_SINKHORN_ATTENTION_MODE,
    OperatorTransportConfig,
    OperatorTransportCorpus,
    OperatorTransportFitError,
    OperatorTransportFitReceipt,
    OperatorTransportHoldoutReceipt,
    OperatorTransportIntegrityError,
    OperatorTransportMissingMeasurementError,
    OperatorTransportObservation,
    OperatorTransportSplit,
    QwenOperatorTransportAvailability,
    causal_prefix_sinkhorn_operator,
    chronological_operator_transport_split,
    evaluate_operator_transport_holdout,
    fit_operator_transport,
    inspect_qwen_operator_transport_availability,
    qwen_operator_transport_corpus_from_atlas,
)
from immer.runtimes.qwen3_8.semantic_atlas import GraphRevision


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class OperatorTransportTests(unittest.TestCase):
    model_pin = _digest("qwen-operator-transport-model")
    graph_revision = GraphRevision(17, _digest("qwen-operator-transport-graph"))

    @staticmethod
    def _transports() -> tuple[np.ndarray, ...]:
        return (
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.01, 0.02, 0.99]],
                dtype=np.float64,
            ),
            np.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.02, 0.01, 1.01]],
                dtype=np.float64,
            ),
        )

    def _corpus(
        self,
        *,
        additive: float = 0.0,
        degenerate: bool = False,
    ) -> OperatorTransportCorpus:
        observations: list[OperatorTransportObservation] = []
        for group_index in range(15):
            for head, coefficient in enumerate(self._transports()):
                if degenerate:
                    source = np.eye(3, dtype=np.float64)
                    target = np.eye(3, dtype=np.float64)
                else:
                    logits = np.random.default_rng(
                        100 * group_index + head
                    ).normal(size=(3, 3))
                    source = causal_prefix_sinkhorn_operator(logits)
                    residual = np.zeros((3, 3), dtype=np.float64)
                    residual[2, 0] = additive * (head + 1)
                    residual[2, 1] = -additive * (head + 1)
                    target = (
                        (coefficient @ source + residual)
                        @ np.linalg.inv(coefficient)
                    )
                    target[np.abs(target) < 1.0e-14] = 0.0
                    target = target / target.sum(axis=1, keepdims=True)
                observations.append(
                    OperatorTransportObservation(
                        temporal_index=group_index + 1,
                        model_pin_sha256=self.model_pin,
                        graph_revision=self.graph_revision,
                        attention_mode=QWEN_PREFIX_SINKHORN_ATTENTION_MODE,
                        prompt_sha256=_digest(f"prompt:{group_index}"),
                        position_group_sha256=_digest(
                            f"position:{group_index}"
                        ),
                        source_head=head,
                        target_head=head + 12,
                        source_measurement_sha256=_digest(
                            f"source-measurement:{group_index}:{head}"
                        ),
                        target_measurement_sha256=_digest(
                            f"target-measurement:{group_index}:{head}"
                        ),
                        evidence_sha256=_digest(
                            f"operator-evidence:{group_index}:{head}"
                        ),
                        source_operator=np.asarray(source, dtype=np.float64),
                        target_operator=np.asarray(target, dtype=np.float64),
                    )
                )
        ordered = tuple(
            sorted(
                observations,
                key=lambda row: (
                    row.temporal_index,
                    row.split_group_sha256,
                    row.source_head,
                    row.target_head,
                    row.sha256,
                ),
            )
        )
        return OperatorTransportCorpus(
            model_pin_sha256=self.model_pin,
            graph_revision=self.graph_revision,
            attention_mode=QWEN_PREFIX_SINKHORN_ATTENTION_MODE,
            observations=ordered,
            observation_sha256s=tuple(row.sha256 for row in ordered),
            evidence_sha256s=tuple(row.evidence_sha256 for row in ordered),
        )

    def test_prefix_sinkhorn_corpus_and_strict_group_split_roundtrip(self) -> None:
        operator = causal_prefix_sinkhorn_operator(
            np.array(
                [[0.1, 99.0, 99.0], [0.3, -0.2, 99.0], [0.4, 0.2, -0.1]],
                dtype=np.float64,
            ),
            alpha=0.8,
            diagonal_debit=0.2,
        )
        self.assertEqual(float(np.max(np.abs(np.triu(operator, 1)))), 0.0)
        np.testing.assert_allclose(operator.sum(axis=1), 1.0, rtol=0.0, atol=1e-14)

        corpus = self._corpus()
        split = chronological_operator_transport_split(corpus)
        restored_corpus = OperatorTransportCorpus.from_bytes(corpus.to_bytes())
        self.assertEqual(restored_corpus.to_bytes(), corpus.to_bytes())
        self.assertEqual(restored_corpus.sha256, corpus.sha256)
        self.assertEqual(OperatorTransportSplit.from_bytes(split.to_bytes()), split)
        self.assertFalse(
            set(split.train_group_sha256s)
            & set(split.holdout_group_sha256s)
        )
        train_times = {
            row.temporal_index
            for row in corpus.observations
            if row.split_group_sha256 in split.train_group_sha256s
        }
        holdout_times = {
            row.temporal_index
            for row in corpus.observations
            if row.split_group_sha256 in split.holdout_group_sha256s
        }
        self.assertLess(max(train_times), min(holdout_times))

    def test_per_head_transport_wins_true_holdout_and_replays(self) -> None:
        corpus = self._corpus()
        split = chronological_operator_transport_split(corpus)
        fit = fit_operator_transport(corpus, split)
        holdout = evaluate_operator_transport_holdout(fit)
        calibration = {
            row.model_name: row for row in fit.calibration_metrics
        }
        held = {row.model_name: row for row in holdout.holdout_metrics}

        self.assertEqual(fit.best_calibration_model, "per_head")
        self.assertEqual(holdout.best_holdout_model, "per_head")
        self.assertLess(calibration["per_head"].mean_relative_residual, 1e-12)
        self.assertLess(held["per_head"].mean_relative_residual, 1e-12)
        self.assertLess(
            held["per_head"].mean_relative_residual,
            held["global"].mean_relative_residual,
        )
        self.assertLess(
            held["per_head"].mean_relative_residual,
            held["identity"].mean_relative_residual,
        )
        self.assertLess(
            held["per_head"].mean_relative_residual,
            held["random"].mean_relative_residual,
        )
        restored_fit = OperatorTransportFitReceipt.from_bytes(fit.to_bytes())
        restored_holdout = OperatorTransportHoldoutReceipt.from_bytes(
            holdout.to_bytes()
        )
        self.assertEqual(restored_fit.to_bytes(), fit.to_bytes())
        self.assertEqual(restored_holdout.to_bytes(), holdout.to_bytes())
        for restored, expected in zip(
            restored_fit.per_head_kernels,
            fit.per_head_kernels,
            strict=True,
        ):
            np.testing.assert_array_equal(
                restored.coefficient, expected.coefficient
            )

    def test_additive_residual_improves_and_degenerate_fit_is_rejected(self) -> None:
        corpus = self._corpus(additive=5.0e-4)
        split = chronological_operator_transport_split(corpus)
        plain = fit_operator_transport(
            corpus, split, config=OperatorTransportConfig(additive_residual=False)
        )
        additive = fit_operator_transport(
            corpus, split, config=OperatorTransportConfig(additive_residual=True)
        )
        plain_metric = {
            row.model_name: row.mean_relative_residual
            for row in plain.calibration_metrics
        }["per_head"]
        additive_metric = {
            row.model_name: row.mean_relative_residual
            for row in additive.calibration_metrics
        }["per_head"]
        self.assertLess(additive_metric, plain_metric)
        self.assertTrue(
            any(np.any(kernel.additive_residual) for kernel in additive.per_head_kernels)
        )
        with self.assertRaisesRegex(OperatorTransportFitError, "ill-conditioned"):
            fit_operator_transport(
                self._corpus(),
                chronological_operator_transport_split(self._corpus()),
                config=OperatorTransportConfig(
                    maximum_transport_condition=1.001
                ),
            )

        degenerate = self._corpus(degenerate=True)
        with self.assertRaisesRegex(OperatorTransportFitError, "rank deficient"):
            fit_operator_transport(
                degenerate,
                chronological_operator_transport_split(degenerate),
            )

    def test_resealed_metric_and_coefficient_tamper_fail_recomputation(self) -> None:
        corpus = self._corpus()
        fit = fit_operator_transport(
            corpus, chronological_operator_transport_split(corpus)
        )

        metric_document = json.loads(fit.to_bytes())
        metric_document["body"]["calibration_metrics"][0][
            "mean_relative_residual"
        ] += 0.01
        metric_document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(metric_document["body"])
        ).hexdigest()
        with self.assertRaises(OperatorTransportIntegrityError):
            OperatorTransportFitReceipt.from_bytes(
                canonical_json_bytes(metric_document)
            )

        coefficient_document = json.loads(fit.to_bytes())
        kernel = coefficient_document["body"]["per_head_kernels"][0]
        record = kernel["body"]["coefficient"]
        raw = bytearray(base64.b64decode(record["data_base64"]))
        changed = np.frombuffer(raw, dtype="<f8").copy()
        changed[0] += 0.01
        changed_raw = changed.astype("<f8").tobytes()
        record["data_base64"] = base64.b64encode(changed_raw).decode("ascii")
        record["data_sha256"] = hashlib.sha256(changed_raw).hexdigest()
        kernel["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(kernel["body"])
        ).hexdigest()
        coefficient_document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(coefficient_document["body"])
        ).hexdigest()
        with self.assertRaises(OperatorTransportIntegrityError):
            OperatorTransportFitReceipt.from_bytes(
                canonical_json_bytes(coefficient_document)
            )

    def test_qwen_atlas_adapter_seals_exact_missing_native_measurement(self) -> None:
        availability = inspect_qwen_operator_transport_availability(
            (),
            model_pin_sha256=self.model_pin,
            graph_revision=self.graph_revision,
        )
        self.assertFalse(availability.available)
        self.assertEqual(
            availability.missing_measurements,
            ("per-head-prefix-sinkhorn-operator-matrix",),
        )
        self.assertIn("Qwen38NativeHeadCrsa.route", availability.required_runtime_seam)
        self.assertEqual(
            QwenOperatorTransportAvailability.from_bytes(
                availability.to_bytes()
            ),
            availability,
        )
        with self.assertRaisesRegex(
            OperatorTransportMissingMeasurementError,
            "instead of reconstructing archive Softmax",
        ):
            qwen_operator_transport_corpus_from_atlas(
                (),
                model_pin_sha256=self.model_pin,
                graph_revision=self.graph_revision,
            )


if __name__ == "__main__":
    unittest.main()
