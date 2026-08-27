from __future__ import annotations

import hashlib
import json
import unittest
import warnings

import numpy as np

from immer.runtimes.ooe import subspace_battery
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.operator_harvester import HarvesterState
from immer.runtimes.ooe.subspace_battery import (
    SUBSPACE_ATTENTION_KIND,
    SUBSPACE_CONTEXT_KIND,
    SubspaceBatteryCapacityError,
    SubspaceBatteryEvaluation,
    SubspaceBatteryFit,
    SubspaceBatteryIntegrityError,
    SubspaceCorpus,
    SubspaceInstrumentationRequired,
    SubspaceInstrumentationRequest,
    SubspaceObservationGroup,
    SubspaceSweepConfig,
    evaluate_subspace_battery,
    fit_subspace_battery,
    inspect_harvester_subspace_evidence,
    subspace_corpus_from_harvester_state,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _group(
    logical_time: int, *, holdout_scale: float = 1.0
) -> SubspaceObservationGroup:
    # Neuron 0 has the greatest train energy but is identical for both outputs.
    # Neuron 1 carries the missing class distinction.  Consequently k=1 must
    # collide and k>=2 can become exact.
    gate = np.array(
        [
            [10.0, 5.0, 0.2, 0.1],
            [10.0, -5.0, 0.2, -0.1],
            [10.0, 5.0, -0.2, 0.1],
            [10.0, -5.0, -0.2, -0.1],
        ],
        dtype=np.float64,
    )
    up = np.array(
        [
            [9.0, 4.0, 0.1, 0.2],
            [9.0, -4.0, 0.1, -0.2],
            [9.0, 4.0, -0.1, 0.2],
            [9.0, -4.0, -0.1, -0.2],
        ],
        dtype=np.float64,
    )
    if logical_time >= 5:
        gate = gate * holdout_scale
        up = up * holdout_scale
    context = np.array(
        [
            [logical_time, 1.0, -1.0],
            [logical_time, -1.0, 1.0],
            [logical_time + 0.5, 1.0, 1.0],
            [logical_time + 0.5, -1.0, -1.0],
        ],
        dtype=np.float64,
    )
    outputs = (_hash("A"), _hash("B"), _hash("A"), _hash("B"))
    sources = tuple(
        sorted(
            (
                _hash(f"context-receipt:{logical_time}"),
                _hash(f"gate-receipt:{logical_time}"),
                _hash(f"up-receipt:{logical_time}"),
            )
        )
    )
    event_sha = _hash(f"graph-event:{logical_time}")
    graph_revision = subspace_battery.graph_revision_sha256(logical_time, event_sha)
    projection_verifier = _hash("joint-projection-verifier")
    output_verifier = _hash("exact-output-verifier")
    return SubspaceObservationGroup(
        logical_time=logical_time,
        group_sha256=_hash(f"group:{logical_time}:{holdout_scale}"),
        model_pin_sha256=_hash("qwen-model-pin"),
        graph_revision_sha256=graph_revision,
        graph_sequence=logical_time,
        graph_event_sha256=event_sha,
        layer=18 + logical_time,
        prompt_sha256=_hash(f"prompt:{logical_time % 2}"),
        source_receipt_sha256s=sources,
        projection_verifier_sha256=projection_verifier,
        projection_evidence_sha256=subspace_battery.projection_evidence_sha256(
            model_pin_sha256=_hash("qwen-model-pin"),
            graph_revision_sha256=graph_revision,
            layer=18 + logical_time,
            source_receipt_sha256s=sources,
            projection_verifier_sha256=projection_verifier,
            context_states=context,
            gate_projection=gate,
            up_projection=up,
        ),
        output_verifier_sha256=output_verifier,
        output_evidence_sha256=subspace_battery.output_evidence_sha256(
            output_payload_sha256s=outputs,
            output_verifier_sha256=output_verifier,
            source_receipt_sha256s=sources,
        ),
        context_states=context,
        gate_projection=gate,
        up_projection=up,
        output_payload_sha256s=outputs,
    )


def _corpus(*, holdout_scale: float = 1.0) -> SubspaceCorpus:
    return SubspaceCorpus(
        model_pin_sha256=_hash("qwen-model-pin"),
        groups=tuple(
            _group(index, holdout_scale=holdout_scale) for index in range(1, 7)
        ),
    )


def _config(**overrides: object) -> SubspaceSweepConfig:
    values = {
        "k_values": (1, 2, 4),
        "quant_bits": (2, 4, 8),
        "scale_floor": 1e-12,
        "random_seed_sha256": _hash("random-control"),
        "max_working_bytes": 128 * 1024 * 1024,
        **overrides,
    }
    return SubspaceSweepConfig(**values)  # type: ignore[arg-type]


def _fit(corpus: SubspaceCorpus) -> SubspaceBatteryFit:
    return fit_subspace_battery(
        corpus,
        train_group_indices=(0, 1, 2),
        calibration_group_indices=(3,),
        config=_config(),
    )


class SubspaceBatteryTests(unittest.TestCase):
    def test_joint_gate_up_sweep_finds_collision_boundary_and_controls(self) -> None:
        corpus = _corpus()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fit = _fit(corpus)
            holdout = evaluate_subspace_battery(
                fit, corpus, holdout_group_indices=(4, 5)
            )

        self.assertEqual(
            SubspaceCorpus.from_bytes(corpus.to_bytes()).to_bytes(), corpus.to_bytes()
        )
        self.assertEqual(
            SubspaceBatteryFit.from_bytes(fit.to_bytes(), corpus=corpus).to_bytes(),
            fit.to_bytes(),
        )
        self.assertEqual(
            SubspaceBatteryEvaluation.from_bytes(
                holdout.to_bytes(), fit=fit, corpus=corpus
            ).to_bytes(),
            holdout.to_bytes(),
        )
        self.assertEqual(
            {row.family for row in fit.models},
            {"candidate", "full", "marginal", "random"},
        )
        k1 = [row for row in fit.models if row.family == "candidate" and row.k == 1]
        self.assertTrue(k1)
        self.assertTrue(all(row.train_wrong_collision_rows > 0 for row in k1))
        self.assertTrue(all(not row.calibration_safe for row in k1))
        promoted = [row for row in holdout.results if row.promoted]
        self.assertTrue(promoted)
        self.assertTrue(
            all(row.family == "candidate" and row.k >= 2 for row in promoted)
        )
        by_model = {row.sha256: row for row in fit.models}
        for row in promoted:
            model = by_model[row.model_sha256]
            self.assertEqual(model.train_wrong_collision_rows, 0)
            self.assertEqual(model.calibration_metrics.wrong_collisions, 0)
            self.assertEqual(row.metrics.wrong_collisions, 0)
            self.assertGreater(row.metrics.exact_verified_hits, 0)
            self.assertGreater(row.metrics.stored_bytes, 0)
        self.assertFalse(
            any(row.promoted for row in holdout.results if row.family != "candidate")
        )

    def test_basis_and_scale_are_train_only_while_corpus_identity_changes(self) -> None:
        first = _corpus(holdout_scale=1.0)
        shifted = _corpus(holdout_scale=1000.0)
        first_fit = _fit(first)
        shifted_fit = _fit(shifted)
        self.assertNotEqual(first.sha256, shifted.sha256)
        for left, right in zip(first_fit.models, shifted_fit.models, strict=True):
            self.assertEqual(left.key, right.key)
            self.assertEqual(left.basis_indices, right.basis_indices)
            if left.scale is None:
                self.assertIsNone(right.scale)
            else:
                np.testing.assert_array_equal(left.scale, right.scale)
            self.assertEqual(left.calibration_metrics, right.calibration_metrics)

    def test_fit_and_holdout_reject_resealed_scale_and_metric_forgery(self) -> None:
        corpus = _corpus()
        fit = _fit(corpus)
        fit_document = json.loads(fit.to_bytes())
        candidate = next(
            row
            for row in fit_document["body"]["models"]
            if row["family"] == "candidate" and row["k"] == 2
        )
        scale = subspace_battery._array_from_record(candidate["scale"], field="scale")
        forged_scale = scale * 2.0
        candidate["scale"] = subspace_battery._array_record(forged_scale)
        candidate["quantizer_sha256"] = subspace_battery._quantizer_sha256(
            candidate["quant_bits"], forged_scale, family=candidate["family"]
        )
        candidate["calibration_metrics"]["stored_bytes"] += 1
        fit_document["body_sha256"] = subspace_battery._digest(fit_document["body"])
        with self.assertRaisesRegex(
            SubspaceBatteryIntegrityError, "complete recomputation"
        ):
            SubspaceBatteryFit.from_bytes(
                canonical_json_bytes(fit_document), corpus=corpus
            )

        evaluation = evaluate_subspace_battery(
            fit, corpus, holdout_group_indices=(4, 5)
        )
        evaluation_document = json.loads(evaluation.to_bytes())
        evaluation_document["body"]["results"][0]["metrics"]["stored_bytes"] += 1
        evaluation_document["body_sha256"] = subspace_battery._digest(
            evaluation_document["body"]
        )
        with self.assertRaisesRegex(
            SubspaceBatteryIntegrityError, "complete recomputation"
        ):
            SubspaceBatteryEvaluation.from_bytes(
                canonical_json_bytes(evaluation_document), fit=fit, corpus=corpus
            )

    def test_group_chronology_overlap_and_capacity_fail_closed(self) -> None:
        corpus = _corpus()
        with self.assertRaisesRegex(ValueError, "follow"):
            fit_subspace_battery(
                corpus,
                train_group_indices=(2, 3),
                calibration_group_indices=(1,),
                config=_config(),
            )
        fit = fit_subspace_battery(
            corpus,
            train_group_indices=(1, 2),
            calibration_group_indices=(3,),
            config=_config(),
        )
        with self.assertRaisesRegex(ValueError, "must follow"):
            evaluate_subspace_battery(fit, corpus, holdout_group_indices=(0,))
        with self.assertRaisesRegex(ValueError, "overlaps"):
            evaluate_subspace_battery(fit, corpus, holdout_group_indices=(3, 4))
        with self.assertRaisesRegex(SubspaceBatteryCapacityError, "requires"):
            fit_subspace_battery(
                corpus,
                train_group_indices=(0, 1, 2),
                calibration_group_indices=(3,),
                config=_config(max_working_bytes=1),
            )

    def test_current_harvester_adapter_emits_request_and_never_claims_live_fit(
        self,
    ) -> None:
        state = HarvesterState.empty(_hash("harvester-identity"))
        request = inspect_harvester_subspace_evidence(state)
        self.assertEqual(
            SubspaceInstrumentationRequest.from_bytes(request.to_bytes()).to_bytes(),
            request.to_bytes(),
        )
        body = request.to_document()["body"]
        self.assertFalse(body["static_embedding_means_allowed"])
        self.assertFalse(body["softmax_attention_allowed"])
        self.assertEqual(body["attention_kind"], SUBSPACE_ATTENTION_KIND)
        self.assertEqual(body["context_kind"], SUBSPACE_CONTEXT_KIND)
        with self.assertRaises(SubspaceInstrumentationRequired) as raised:
            subspace_corpus_from_harvester_state(state)
        self.assertEqual(raised.exception.request.sha256, request.sha256)


if __name__ == "__main__":
    unittest.main()
