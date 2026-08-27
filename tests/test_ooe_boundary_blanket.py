from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from immer.runtimes.deepseek_v4.causal_weights import TensorRangePlan
from immer.runtimes.ooe import boundary_blanket
from immer.runtimes.ooe import operator_harvester
from immer.runtimes.ooe.boundary_blanket import (
    BoundaryBlanketCapacityError,
    BoundaryBlanketConditionError,
    BoundaryBlanketFit,
    BoundaryBlanketIntegrityError,
    BoundaryCorpus,
    BoundaryFitConfig,
    FIXED_RESIDUAL_SUM_FAMILY,
    evaluate_boundary_blanket_holdout,
    evaluate_boundary_blanket_loqo,
    fit_boundary_blanket,
    verify_boundary_blanket_holdout,
    verify_boundary_blanket_loqo,
)
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.operator_harvester import (
    ContextualOperatorObservation,
    HARVESTER_STATE_PREFIX,
    HarvesterState,
)
from immer.runtimes.qwen3_8.semantic_atlas import (
    GraphRevision,
    InterventionIdentity,
    MeasurementReceipt,
    ModelPin,
    NumericSummary,
    ProbeIdentity,
    RuntimeProvenance,
    WeightCoordinate,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


_CODE_REVISION = "8" * 40
_PIN = ModelPin(
    repo_id="Qwen/Qwen3.8-27B",
    revision="boundary-test",
    bundle_fingerprint=_hash("bundle"),
    bundle_manifest_sha256=_hash("bundle-manifest"),
    code_revision=_CODE_REVISION,
)
_RUNTIME = RuntimeProvenance(
    code_revision=_CODE_REVISION,
    source_manifest_sha256=_hash("source"),
    dependency_manifest_sha256=_hash("dependencies"),
    runtime_configuration_sha256=_hash("runtime"),
    platform_sha256=_hash("platform"),
)
_PLAN = TensorRangePlan(
    name="model.language_model.layers.0.input_layernorm.weight",
    dtype="F64",
    shape=(8, 2),
    shard="model-test.safetensors",
    absolute_offset=4096,
    length=128,
)
_WEIGHT_REVISION = GraphRevision(3, _hash("weight-rail"))
_EMITTER = _hash("boundary-emitter")
_FEATURE_SCHEMA = _hash("boundary-feature-schema")
_ACTION_SCHEMA = _hash("boundary-action-schema")


def _measurement(sequence: int, layer: int, prompt: int) -> MeasurementReceipt:
    plan = TensorRangePlan(
        name=f"model.language_model.layers.{layer}.input_layernorm.weight",
        dtype=_PLAN.dtype,
        shape=_PLAN.shape,
        shard=_PLAN.shard,
        absolute_offset=_PLAN.absolute_offset,
        length=_PLAN.length,
    )
    coordinate = WeightCoordinate.from_plan(
        plan,
        layer=layer,
        module=f"model.language_model.layers.{layer}.input_layernorm",
    )
    return MeasurementReceipt(
        model_pin=_PIN,
        coordinate=coordinate,
        probe=ProbeIdentity(
            question_sha256=_hash(f"question-{prompt}"),
            token_sha256=_hash(f"tokens-{prompt}"),
            family_sha256=_hash("family"),
            label_source_sha256=_hash("label-free"),
        ),
        intervention=InterventionIdentity(
            mode="native", configuration_sha256=_hash(f"intervention-{sequence}")
        ),
        observation_status="recorded",
        observed_semantic_label=None,
        hidden_sha256=_hash(f"hidden-{sequence}"),
        activation_sha256=_hash(f"activation-{sequence}"),
        logits_sha256=_hash(f"logits-{sequence}"),
        state_sha256=_hash(f"state-{sequence}"),
        access_trace_sha256=_hash(f"trace-{sequence}"),
        evidence_sha256=_hash(f"evidence-{sequence}"),
        weight_rail_revision=_WEIGHT_REVISION,
        atlas_head_revision=GraphRevision(
            max(0, sequence - 1), _hash(f"atlas-head-{sequence - 1}")
        ),
        numeric_summaries=(
            NumericSummary(
                metric="probe",
                count=1,
                total=float(sequence),
                total_squares=float(sequence * sequence),
                minimum=float(sequence),
                maximum=float(sequence),
            ),
        ),
        placebo_effects=(),
        runtime=_RUNTIME,
    )


def _observation(
    measurement: MeasurementReceipt,
    revision: GraphRevision,
    source: str,
    target: str,
    x: np.ndarray,
    y: np.ndarray,
) -> ContextualOperatorObservation:
    return ContextualOperatorObservation.capture(
        measurement,
        atlas_revision=revision,
        emitter_sha256=_EMITTER,
        feature_schema_sha256=_FEATURE_SCHEMA,
        action_schema_sha256=_ACTION_SCHEMA,
        source_state=source,
        target_state=target,
        input_array=np.asarray(x, dtype=np.float64),
        output_array=np.asarray(y, dtype=np.float64),
    )


def _state(
    *,
    layers: tuple[int, ...] = (0, 1, 2, 3),
    prompts: int = 3,
    zero: bool = False,
    holdout_offset: float = 0.0,
    variable_rows: bool = False,
) -> HarvesterState:
    observations = []
    sequence = 0
    for layer in layers:
        for prompt in range(prompts):
            sequence += 1
            rows = 2 + ((sequence * 3) % 5) if variable_rows else 5
            generator = np.random.default_rng(1000 + sequence)
            arrays = [generator.normal(size=(rows, 2)) for _ in range(6)]
            if zero:
                arrays = [np.zeros((rows, 2), dtype=np.float64) for _ in range(6)]
            (
                layer_input,
                attention_input,
                attention_output,
                attention_residual,
                mlp_input,
                mlp_output,
            ) = arrays
            target = attention_residual + mlp_output
            if layer == layers[-1] and holdout_offset:
                # Preserve the exact residual identity while changing only the
                # unseen holdout evidence.
                mlp_output = mlp_output + holdout_offset
                target = attention_residual + mlp_output
            measurement = _measurement(sequence, layer, prompt)
            revision = GraphRevision(sequence, _hash(f"atlas-{sequence}"))
            prefix = f"qwen.layer.{layer}"
            observations.extend(
                (
                    _observation(
                        measurement,
                        revision,
                        f"{prefix}.pre-hidden-sketch",
                        f"{prefix}.post-hidden-sketch",
                        layer_input,
                        target,
                    ),
                    _observation(
                        measurement,
                        revision,
                        f"{prefix}.attention.input-sketch",
                        f"{prefix}.attention.output-sketch",
                        attention_input,
                        attention_output,
                    ),
                    _observation(
                        measurement,
                        revision,
                        f"{prefix}.layer.input-sketch",
                        f"{prefix}.attention.residual-sketch",
                        layer_input,
                        attention_residual,
                    ),
                    _observation(
                        measurement,
                        revision,
                        f"{prefix}.mlp.input-sketch",
                        f"{prefix}.mlp.output-sketch",
                        mlp_input,
                        mlp_output,
                    ),
                    _observation(
                        measurement,
                        revision,
                        f"{prefix}.attention.residual-sketch",
                        f"{prefix}.layer.output-sketch",
                        attention_residual,
                        target,
                    ),
                )
            )
    grouped: dict[str, list[ContextualOperatorObservation]] = {}
    for row in observations:
        group_sha = operator_harvester._group_sha256(row.receipt)
        grouped.setdefault(group_sha, []).append(row)
    return HarvesterState(
        identity_sha256=_hash(
            f"harvester-{layers}-{prompts}-{zero}-{holdout_offset}-{variable_rows}"
        ),
        provider_cursor=None,
        groups=tuple((key, tuple(value)) for key, value in sorted(grouped.items())),
        promotions=(),
        recent_receipt_sha256s=(),
    )


def _fit(corpus: BoundaryCorpus) -> BoundaryBlanketFit:
    return fit_boundary_blanket(
        corpus,
        train_indices=tuple(range(6)),
        calibration_indices=tuple(range(6, 9)),
        config=BoundaryFitConfig(selection_tolerance=1e-10),
    )


class BoundaryBlanketTests(unittest.TestCase):
    def test_exact_structural_residual_finds_two_of_six_and_holds_out_layer(
        self,
    ) -> None:
        corpus = BoundaryCorpus.from_harvester_state(_state(variable_rows=True))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fit = _fit(corpus)
            evaluation = evaluate_boundary_blanket_holdout(
                fit, corpus, holdout_indices=(9, 10, 11)
            )

        self.assertEqual(fit.selected_subset, ("attention_residual", "mlp_output"))
        self.assertEqual(fit.selected_model_family, FIXED_RESIDUAL_SUM_FAMILY)
        self.assertEqual(len(fit.models), 64)
        body = evaluation["body"]
        self.assertIsInstance(body, dict)
        assert isinstance(body, dict)
        selected = body["selected"]
        baselines = body["baselines"]
        self.assertIsInstance(selected, dict)
        self.assertIsInstance(baselines, dict)
        assert isinstance(selected, dict) and isinstance(baselines, dict)
        self.assertLess(float(selected["score"]), 1e-12)
        self.assertLess(float(baselines["attention_residual_plus_mlp_output"]), 1e-15)
        self.assertGreater(float(baselines["no_memory_mean"]), 0.1)
        self.assertGreater(float(baselines["layer_input_identity"]), 0.1)
        self.assertGreater(float(baselines["selected_target_permutation_placebo"]), 0.1)
        self.assertEqual(len(body["equal_size_alternatives"]), 16)
        self.assertTrue(body["topology_holdout"])
        self.assertEqual(body["unseen_layer_indices"], [3])
        self.assertEqual(len(body["repeated_prompt_sha256s"]), 3)
        self.assertEqual(
            body["winner"]["name"],
            "fixed_attention_residual_plus_mlp_output",
        )
        self.assertEqual(body["selected"]["model_family"], FIXED_RESIDUAL_SUM_FAMILY)
        self.assertEqual(body["best_fitted_affine"]["model_family"], "fitted_affine")

    def test_macro_weighting_uses_one_vote_per_cluster_not_token_rows(self) -> None:
        corpus = BoundaryCorpus.from_harvester_state(_state(variable_rows=True))
        fit = _fit(corpus)
        expected = (
            sum(
                np.mean(corpus.macros[index].target, axis=0, keepdims=True)
                for index in range(6)
            )
            / 6.0
        )
        np.testing.assert_allclose(fit.mean_target, expected, atol=0.0, rtol=0.0)
        row_weighted = np.mean(
            np.concatenate([corpus.macros[index].target for index in range(6)], axis=0),
            axis=0,
            keepdims=True,
        )
        self.assertGreater(float(np.max(np.abs(fit.mean_target - row_weighted))), 1e-5)

    def test_holdout_is_separate_and_cannot_change_fitted_coefficients(self) -> None:
        first = BoundaryCorpus.from_harvester_state(_state(holdout_offset=0.0))
        second = BoundaryCorpus.from_harvester_state(_state(holdout_offset=100.0))
        first_fit = _fit(first)
        second_fit = _fit(second)
        self.assertEqual(first_fit.selected_subset, second_fit.selected_subset)
        for left, right in zip(first_fit.models, second_fit.models, strict=True):
            np.testing.assert_array_equal(left.coefficient, right.coefficient)
            self.assertEqual(left.calibration_score, right.calibration_score)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            evaluate_boundary_blanket_holdout(first_fit, first, holdout_indices=(8, 9))
        with self.assertRaisesRegex(BoundaryBlanketIntegrityError, "another corpus"):
            evaluate_boundary_blanket_holdout(
                first_fit, second, holdout_indices=(9, 10, 11)
            )
        later_fit = fit_boundary_blanket(
            first,
            train_indices=(3, 4, 5),
            calibration_indices=(6, 7, 8),
        )
        with self.assertRaisesRegex(ValueError, "must follow"):
            evaluate_boundary_blanket_holdout(
                later_fit, first, holdout_indices=(0, 1, 2)
            )

    def test_tamper_stale_and_order_fail_closed(self) -> None:
        state = _state()
        with self.assertRaisesRegex(BoundaryBlanketIntegrityError, "identity is stale"):
            BoundaryCorpus.from_harvester_state(
                state, expected_harvester_identity_sha256=_hash("wrong")
            )
        corpus = BoundaryCorpus.from_harvester_state(state)
        fit = _fit(corpus)
        payload = bytearray(fit.to_bytes())
        payload[len(payload) // 2] ^= 1
        with self.assertRaises(BoundaryBlanketIntegrityError):
            BoundaryBlanketFit.from_bytes(bytes(payload), corpus=corpus)
        document = json.loads(fit.to_bytes())
        structural = next(
            row
            for row in document["body"]["models"]
            if row["model_family"] == FIXED_RESIDUAL_SUM_FAMILY
        )
        coefficient = structural["coefficient"]
        raw = bytearray(base64.b64decode(coefficient["data_base64"]))
        raw[-8:] = np.array([1.0], dtype="<f8").tobytes()
        coefficient["data_base64"] = base64.b64encode(raw).decode("ascii")
        coefficient["sha256"] = hashlib.sha256(raw).hexdigest()
        document["body_sha256"] = operator_harvester._digest(document["body"])
        with self.assertRaises(BoundaryBlanketIntegrityError):
            BoundaryBlanketFit.from_bytes(
                operator_harvester.canonical_json_bytes(document), corpus=corpus
            )
        split_document = json.loads(fit.to_bytes())
        split_document["body"]["train"]["prompt_sha256s"][0] = _hash(
            "resealed-wrong-prompt"
        )
        split_document["body_sha256"] = operator_harvester._digest(
            split_document["body"]
        )
        with self.assertRaisesRegex(BoundaryBlanketIntegrityError, "train split"):
            BoundaryBlanketFit.from_bytes(
                operator_harvester.canonical_json_bytes(split_document),
                corpus=corpus,
            )
        forged_fit = json.loads(fit.to_bytes())
        forged_model = next(
            row
            for row in forged_fit["body"]["models"]
            if row["model_family"] == "fitted_affine"
            and row["subset"] == ["layer_input"]
        )
        forged_coefficient = np.full(
            (corpus.feature_dimension + 1, corpus.feature_dimension),
            123.0,
            dtype=np.float64,
        )
        forged_model["coefficient"] = boundary_blanket._array_record(forged_coefficient)
        train_macros = tuple(corpus.macros[index] for index in range(6))
        calibration_macros = tuple(corpus.macros[index] for index in range(6, 9))
        forged_model["training_score"] = boundary_blanket._score_model(
            (("layer_input",), forged_coefficient),
            train_macros,
            floor=fit.config.normalization_floor,
        )
        forged_model["calibration_score"] = boundary_blanket._score_model(
            (("layer_input",), forged_coefficient),
            calibration_macros,
            floor=fit.config.normalization_floor,
        )
        selected_ridge = forged_model["ridge"]
        selected_diagnostic = next(
            row
            for row in forged_model["ridge_diagnostics"]
            if row["ridge"] == selected_ridge
        )
        selected_diagnostic["calibration_score"] = forged_model["calibration_score"]
        forged_fit["body_sha256"] = operator_harvester._digest(forged_fit["body"])
        with self.assertRaisesRegex(
            BoundaryBlanketIntegrityError, "complete authenticated recomputation"
        ):
            BoundaryBlanketFit.from_bytes(
                operator_harvester.canonical_json_bytes(forged_fit), corpus=corpus
            )
        self.assertEqual(
            BoundaryBlanketFit.from_bytes(fit.to_bytes(), corpus=corpus).to_bytes(),
            fit.to_bytes(),
        )
        holdout = evaluate_boundary_blanket_holdout(
            fit, corpus, holdout_indices=(9, 10, 11)
        )
        holdout["body"]["holdout"]["prompt_sha256s"][0] = _hash(
            "resealed-wrong-holdout-prompt"
        )
        holdout["body_sha256"] = operator_harvester._digest(holdout["body"])
        with self.assertRaisesRegex(BoundaryBlanketIntegrityError, "holdout split"):
            verify_boundary_blanket_holdout(holdout, fit=fit, corpus=corpus)
        forged_holdout = evaluate_boundary_blanket_holdout(
            fit, corpus, holdout_indices=(9, 10, 11)
        )
        forged_holdout["body"]["selected"]["score"] = 123.0
        forged_holdout["body_sha256"] = operator_harvester._digest(
            forged_holdout["body"]
        )
        with self.assertRaisesRegex(
            BoundaryBlanketIntegrityError, "complete authenticated recomputation"
        ):
            verify_boundary_blanket_holdout(forged_holdout, fit=fit, corpus=corpus)
        with self.assertRaisesRegex(ValueError, "calibration must follow"):
            fit_boundary_blanket(
                corpus,
                train_indices=(3, 4, 5),
                calibration_indices=(0, 1, 2),
            )

    def test_unresolved_condition_and_memory_bound_fail_before_result(self) -> None:
        corpus = BoundaryCorpus.from_harvester_state(_state(zero=True))
        with self.assertRaises(BoundaryBlanketConditionError):
            fit_boundary_blanket(
                corpus,
                train_indices=tuple(range(6)),
                calibration_indices=tuple(range(6, 9)),
                config=BoundaryFitConfig(ridge_grid=(0.0,)),
            )
        with self.assertRaisesRegex(BoundaryBlanketCapacityError, "requires"):
            fit_boundary_blanket(
                corpus,
                train_indices=tuple(range(6)),
                calibration_indices=tuple(range(6, 9)),
                config=BoundaryFitConfig(max_working_bytes=1),
            )

    def test_loqo_is_distinct_prompt_holdout(self) -> None:
        corpus = BoundaryCorpus.from_harvester_state(_state())
        report = evaluate_boundary_blanket_loqo(corpus)
        body = report["body"]
        self.assertIsInstance(body, dict)
        assert isinstance(body, dict)
        self.assertEqual(body["prompt_count"], 3)
        self.assertEqual(len(body["folds"]), 3)
        self.assertEqual(
            {tuple(row["selected_subset"]) for row in body["folds"]},
            {("attention_residual", "mlp_output")},
        )
        self.assertEqual(
            {row["selected_model_family"] for row in body["folds"]},
            {FIXED_RESIDUAL_SUM_FAMILY},
        )
        forged = json.loads(operator_harvester.canonical_json_bytes(report))
        forged["body"]["folds"][0]["selected_score"] = 123.0
        forged["body_sha256"] = operator_harvester._digest(forged["body"])
        with self.assertRaisesRegex(
            BoundaryBlanketIntegrityError, "complete authenticated recomputation"
        ):
            verify_boundary_blanket_loqo(corpus, forged)

    def test_cli_discovers_store_and_publishes_once_atomically(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1] / "scripts" / "ooe_boundary_blanket.py"
        )
        spec = importlib.util.spec_from_file_location(
            "ooe_boundary_blanket", script_path
        )
        assert spec is not None and spec.loader is not None
        script = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(script)
        state = _state()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "compute"
            store = CrystalStore(root)
            name = f"{HARVESTER_STATE_PREFIX}{state.identity_sha256}"
            store.publish_state(name, state.to_bytes())
            self.assertEqual(script.discover_harvester_state_name(root), name)
            corpus = BoundaryCorpus.from_harvester_state(state)
            report = script.build_report(
                corpus,
                harvester_state_name=name,
                train_end=6,
                calibration_end=9,
                holdout_end=12,
            )
            self.assertEqual(
                report["body"]["evaluation_kinds"], ["strict_topology_holdout"]
            )
            self.assertNotIn("holdout", report["body"])
            self.assertNotIn("loqo", report["body"])
            self.assertIsNone(report["body"]["loqo_prompt_audit"])
            forged_report = json.loads(operator_harvester.canonical_json_bytes(report))
            strict = forged_report["body"]["strict_topology_holdout"]
            strict["body"]["selected"]["score"] = 123.0
            strict["body_sha256"] = operator_harvester._digest(strict["body"])
            forged_report["body"]["strict_topology_holdout_sha256"] = strict[
                "body_sha256"
            ]
            forged_report["body_sha256"] = operator_harvester._digest(
                forged_report["body"]
            )
            with self.assertRaises(BoundaryBlanketIntegrityError):
                script.verify_report(corpus, forged_report)
            output = Path(temporary) / "report.json"
            payload = operator_harvester.canonical_json_bytes(report)
            script.write_new_atomic(output, payload)
            self.assertEqual(output.read_bytes(), payload)
            with self.assertRaises(FileExistsError):
                script.write_new_atomic(output, payload)


if __name__ == "__main__":
    unittest.main()
