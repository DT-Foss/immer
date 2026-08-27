from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json
import tempfile
import unittest

import numpy as np

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.operator_transport import (
    QWEN_PREFIX_SINKHORN_ATTENTION_MODE,
    OperatorTransportConfig,
    OperatorTransportCaptureConflictError,
    OperatorTransportCorpus,
    OperatorTransportFitError,
    OperatorTransportFitReceipt,
    OperatorTransportHoldoutReceipt,
    OperatorTransportIntegrityError,
    OperatorTransportMissingMeasurementError,
    OperatorTransportObservation,
    OperatorTransportSplit,
    QwenOperatorTransportAvailability,
    QwenPrefixSinkhornCaptureBank,
    QwenPrefixSinkhornCaptureReceipt,
    causal_prefix_sinkhorn_operator,
    chronological_operator_transport_split,
    evaluate_operator_transport_holdout,
    fit_operator_transport,
    inspect_qwen_operator_transport_availability,
    operator_transport_evidence_sha256,
    qwen_operator_transport_corpus_from_atlas,
    qwen_operator_transport_corpus_from_captures,
)
from immer.runtimes.qwen3_8.semantic_atlas import (
    GraphRevision,
    InterventionIdentity,
    MeasurementReceipt,
    ModelPin,
    NumericSummary,
    ProbeIdentity,
    RuntimeProvenance,
    TensorRangePlan,
    WeightCoordinate,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _measurement() -> MeasurementReceipt:
    code_revision = "7" * 40
    plan = TensorRangePlan(
        name="model.language_model.layers.27.self_attn.q_proj.weight",
        dtype="BF16",
        shape=(8, 8),
        shard="model-00008-of-00018.safetensors",
        absolute_offset=1024,
        length=128,
    )
    pin = ModelPin(
        repo_id="Qwen/Qwen3.8-27B",
        revision="0123456789abcdef",
        bundle_fingerprint=_digest("capture-bundle"),
        bundle_manifest_sha256=_digest("capture-bundle-manifest"),
        code_revision=code_revision,
    )
    probe = ProbeIdentity(
        question_sha256=_digest("capture-question"),
        token_sha256=_digest("capture-tokens"),
        family_sha256=_digest("capture-family"),
        label_source_sha256=_digest("capture-label-source"),
    )
    summary = NumericSummary(
        metric="native_prefix_sinkhorn_delta",
        count=1,
        total=0.5,
        total_squares=0.25,
        minimum=0.5,
        maximum=0.5,
    )
    return MeasurementReceipt(
        model_pin=pin,
        coordinate=WeightCoordinate.from_plan(
            plan,
            layer=27,
            module="model.language_model.layers.27.self_attn.q_proj",
            head_index=2,
            row_start=0,
            row_end=4,
        ),
        probe=probe,
        intervention=InterventionIdentity(
            mode="native",
            configuration_sha256=_digest("capture-intervention"),
        ),
        observation_status="recorded",
        observed_semantic_label=None,
        hidden_sha256=_digest("capture-hidden"),
        activation_sha256=_digest("capture-activation"),
        logits_sha256=_digest("capture-logits"),
        state_sha256=_digest("capture-state"),
        access_trace_sha256=_digest("capture-access"),
        evidence_sha256=_digest("capture-evidence"),
        weight_rail_revision=GraphRevision(9, _digest("capture-weight-rail")),
        atlas_head_revision=GraphRevision(0, "0" * 64),
        numeric_summaries=(summary,),
        placebo_effects=(),
        runtime=RuntimeProvenance(
            code_revision=code_revision,
            source_manifest_sha256=_digest("capture-sources"),
            dependency_manifest_sha256=_digest("capture-dependencies"),
            runtime_configuration_sha256=_digest("capture-runtime"),
            platform_sha256=_digest("capture-platform"),
        ),
    )


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
                prompt_sha256 = _digest(f"prompt:{group_index}")
                position_group_sha256 = _digest(f"position:{group_index}")
                source_measurement_sha256 = _digest(
                    f"source-measurement:{group_index}:{head}"
                )
                target_measurement_sha256 = _digest(
                    f"target-measurement:{group_index}:{head}"
                )
                observations.append(
                    OperatorTransportObservation(
                        temporal_index=group_index + 1,
                        model_pin_sha256=self.model_pin,
                        graph_revision=self.graph_revision,
                        attention_mode=QWEN_PREFIX_SINKHORN_ATTENTION_MODE,
                        prompt_sha256=prompt_sha256,
                        position_group_sha256=position_group_sha256,
                        source_head=head,
                        target_head=head + 12,
                        source_measurement_sha256=source_measurement_sha256,
                        target_measurement_sha256=target_measurement_sha256,
                        evidence_sha256=operator_transport_evidence_sha256(
                            temporal_index=group_index + 1,
                            model_pin_sha256=self.model_pin,
                            graph_revision=self.graph_revision,
                            attention_mode=QWEN_PREFIX_SINKHORN_ATTENTION_MODE,
                            prompt_sha256=prompt_sha256,
                            position_group_sha256=position_group_sha256,
                            source_head=head,
                            target_head=head + 12,
                            source_measurement_sha256=(source_measurement_sha256),
                            target_measurement_sha256=(target_measurement_sha256),
                            source_operator=np.asarray(source, dtype=np.float64),
                            target_operator=np.asarray(target, dtype=np.float64),
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

        evidence_document = json.loads(corpus.to_bytes())
        observation = evidence_document["body"]["observations"][0]
        observation["body"]["evidence_sha256"] = "0" * 64
        observation["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(observation["body"])
        ).hexdigest()
        observation_bytes = canonical_json_bytes(observation)
        evidence_document["body"]["observation_sha256s"][0] = hashlib.sha256(
            observation_bytes
        ).hexdigest()
        evidence_document["body"]["evidence_sha256s"][0] = "0" * 64
        evidence_document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(evidence_document["body"])
        ).hexdigest()
        with self.assertRaises(OperatorTransportIntegrityError):
            OperatorTransportCorpus.from_bytes(
                canonical_json_bytes(evidence_document)
            )

    def test_native_capture_sidecar_is_atomic_bound_and_collision_safe(self) -> None:
        measurement = _measurement()
        atlas_revision = GraphRevision(1, _digest("capture-atlas-append"))
        operators = tuple(
            causal_prefix_sinkhorn_operator(
                np.random.default_rng(800 + head).normal(size=(4, 4))
            )
            for head in range(4)
        )
        receipt = QwenPrefixSinkhornCaptureReceipt.create(
            measurement,
            atlas_revision=atlas_revision,
            capture_spec_sha256=_digest("capture-spec"),
            attention_spec_sha256=_digest("attention-spec"),
            operators=operators,
        )
        restored_receipt = QwenPrefixSinkhornCaptureReceipt.from_bytes(
            receipt.to_bytes()
        )
        self.assertEqual(restored_receipt.to_bytes(), receipt.to_bytes())
        self.assertTrue(
            restored_receipt.verify_against_measurement(
                measurement,
                atlas_revision=atlas_revision,
            )
        )
        with self.assertRaises(OperatorTransportIntegrityError):
            receipt.verify_against_measurement(
                replace(measurement, evidence_sha256=_digest("foreign-evidence")),
                atlas_revision=atlas_revision,
            )

        with tempfile.TemporaryDirectory() as temporary:
            bank = QwenPrefixSinkhornCaptureBank(temporary)
            empty = bank.audit()
            self.assertEqual(empty.receipt_count, 0)
            self.assertFalse(empty.orphan_state_filenames)
            self.assertEqual(bank.publish(receipt), receipt)
            self.assertEqual(bank.publish(receipt), receipt)
            restored = bank.restore(
                receipt.measurement_sha256,
                receipt.capture_spec_sha256,
            )
            self.assertEqual(restored.to_bytes(), receipt.to_bytes())
            reopened = QwenPrefixSinkhornCaptureBank(temporary)
            self.assertEqual(
                reopened.restore(
                    receipt.measurement_sha256,
                    receipt.capture_spec_sha256,
                ).to_bytes(),
                receipt.to_bytes(),
            )
            audit = reopened.audit()
            self.assertEqual(audit.receipt_count, 1)
            self.assertEqual(audit.receipt_sha256s, (receipt.sha256,))
            self.assertFalse(audit.orphan_state_filenames)

            conflicting = replace(
                receipt,
                attention_spec_sha256=_digest("different-attention-spec"),
            )
            with self.assertRaises(OperatorTransportCaptureConflictError):
                reopened.publish(conflicting)
            reopened.store.publish_state("foreign-orphan-state", b"orphan")
            orphan_audit = reopened.audit()
            self.assertEqual(len(orphan_audit.orphan_state_filenames), 1)

        document = json.loads(receipt.to_bytes())
        document["body"]["operators"][0]["data_sha256"] = _digest(
            "forged-operator"
        )
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaises(OperatorTransportIntegrityError):
            QwenPrefixSinkhornCaptureReceipt.from_bytes(
                canonical_json_bytes(document)
            )

    def test_authenticated_capture_receipts_build_transport_corpus(self) -> None:
        base_measurement = _measurement()
        captures = []
        for index in range(15):
            probe = ProbeIdentity(
                question_sha256=_digest(f"capture-question:{index}"),
                token_sha256=_digest(f"capture-token:{index}"),
                family_sha256=base_measurement.probe.family_sha256,
                label_source_sha256=base_measurement.probe.label_source_sha256,
            )
            measurement = replace(base_measurement, probe=probe)
            head_operators = []
            for head, coefficient in enumerate(self._transports()):
                source = causal_prefix_sinkhorn_operator(
                    np.random.default_rng(9_000 + 10 * index + head).normal(
                        size=(4, 4)
                    )
                )
                target = coefficient
                padded = np.eye(4, dtype=np.float64)
                padded[:3, :3] = target
                conjugated = padded @ source @ np.linalg.inv(padded)
                conjugated[np.abs(conjugated) < 1.0e-14] = 0.0
                conjugated = conjugated / conjugated.sum(axis=1, keepdims=True)
                head_operators.extend((source, conjugated))
            captures.append(
                QwenPrefixSinkhornCaptureReceipt.create(
                    measurement,
                    atlas_revision=GraphRevision(
                        index + 1, _digest(f"capture-atlas:{index}")
                    ),
                    capture_spec_sha256=_digest("capture-spec"),
                    attention_spec_sha256=_digest("attention-spec"),
                    operators=head_operators,
                )
            )
        corpus = qwen_operator_transport_corpus_from_captures(
            tuple(reversed(captures)),
            graph_revision=GraphRevision(99, _digest("capture-final-graph")),
        )
        self.assertEqual(len(corpus.observations), 30)
        self.assertEqual(
            OperatorTransportCorpus.from_bytes(corpus.to_bytes()).to_bytes(),
            corpus.to_bytes(),
        )
        fit = fit_operator_transport(
            corpus,
            chronological_operator_transport_split(corpus),
        )
        self.assertEqual(fit.best_calibration_model, "per_head")
        self.assertEqual(
            evaluate_operator_transport_holdout(fit).best_holdout_model,
            "per_head",
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
