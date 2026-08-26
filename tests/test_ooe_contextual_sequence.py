from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from immer.runtimes.ooe import contextual_sequence
from immer.runtimes.ooe.contextual_sequence import (
    ContextualSequenceBank,
    ContextualSequenceCapacityError,
    ContextualSequenceFit,
    ContextualSequenceFitReceipt,
    ContextualSequenceHoldoutReceipt,
    ContextualSequenceIntegrityError,
    DEFAULT_FEATURE_CONFIGS,
    SequenceFeatureConfig,
    SequenceSample,
    evaluate_contextual_sequence_holdout,
    fit_contextual_sequence,
)
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.identity import canonical_json_bytes


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _sequence_sample(index: int, token_count: int) -> SequenceSample:
    generator = np.random.default_rng(100 + index)
    x = generator.normal(size=(token_count, 3))
    normalized = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True))
    state = np.zeros(3, dtype=np.float64)
    residual: list[np.ndarray] = []
    for token in normalized:
        state = 0.7 * state + 0.3 * token
        residual.append(0.8 * state.copy())
    y = x + np.asarray(residual)
    return SequenceSample(
        x=x,
        y=y,
        prompt_sha256=_sha(f"prompt-{index}"),
        evidence_sha256=_sha(f"evidence-{index}"),
    )


class ContextualSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.samples = tuple(
            _sequence_sample(index, token_count)
            for index, token_count in enumerate((70, 85, 77, 93))
        )
        cls.fit = fit_contextual_sequence(
            cls.samples[:2],
            cls.samples[2],
            reservoir_size=24,
            seed=42,
        )
        cls.pointwise = fit_contextual_sequence(
            cls.samples[:2],
            cls.samples[2],
            configs=(DEFAULT_FEATURE_CONFIGS[0],),
            reservoir_size=24,
            seed=42,
        )

    def test_reservoir_beats_pointwise_on_untouched_prompt(self) -> None:
        actual = evaluate_contextual_sequence_holdout(self.fit, self.samples[3])
        pointwise = evaluate_contextual_sequence_holdout(
            self.pointwise, self.samples[3]
        )

        self.assertFalse(self.fit.model.config.pointwise)
        self.assertLess(actual.actual_mse, pointwise.actual_mse * 0.02)
        self.assertTrue(actual.predictive_only)
        self.assertEqual(actual.model_sha256, self.fit.model.sha256)

    def test_variable_lengths_reset_and_causal_prefix(self) -> None:
        receipt = self.fit.receipt
        self.assertEqual([item.token_count for item in receipt.train_samples], [70, 85])
        self.assertEqual(receipt.validation_sample.token_count, 77)

        holdout = self.samples[3]
        first = self.fit.model.predict(holdout.x)
        self.fit.model.predict(self.samples[0].x)
        second = self.fit.model.predict(holdout.x)
        np.testing.assert_array_equal(first, second)

        prefix = self.fit.model.predict(holdout.x[:25])
        np.testing.assert_allclose(prefix, first[:25], rtol=0.0, atol=0.0)
        expected_width = 1 + 3 + len(self.fit.model.config.leaks) * 24
        if self.fit.model.config.square_lift:
            expected_width += len(self.fit.model.config.leaks) * 24
        self.assertEqual(
            self.fit.model.feature_map(holdout.x[:7]).shape, (7, expected_width)
        )

    def test_shuffled_token_and_output_placebos(self) -> None:
        evaluation = evaluate_contextual_sequence_holdout(self.fit, self.samples[3])
        self.assertLess(evaluation.actual_mse, evaluation.shuffled_token_mse)
        self.assertLess(evaluation.actual_mse, evaluation.shuffled_output_mse)
        self.assertNotEqual(
            evaluation.token_permutation, tuple(range(self.samples[3].token_count))
        )
        self.assertNotEqual(
            evaluation.output_permutation, tuple(range(self.samples[3].token_count))
        )
        restored = ContextualSequenceHoldoutReceipt.from_bytes(evaluation.to_bytes())
        restored.verify(self.fit, self.samples[3])
        self.assertEqual(restored.to_bytes(), evaluation.to_bytes())

    def test_holdout_mutation_cannot_change_fitted_model_hash(self) -> None:
        before_model = self.fit.model.sha256
        before_fit = self.fit.sha256
        original = evaluate_contextual_sequence_holdout(self.fit, self.samples[3])
        mutated = SequenceSample(
            x=self.samples[3].x,
            y=np.asarray(self.samples[3].y) + 0.25,
            prompt_sha256=self.samples[3].prompt_sha256,
            evidence_sha256=self.samples[3].evidence_sha256,
        )
        changed = evaluate_contextual_sequence_holdout(self.fit, mutated)

        self.assertEqual(self.fit.model.sha256, before_model)
        self.assertEqual(self.fit.sha256, before_fit)
        self.assertNotEqual(
            original.holdout_sample.sample_sha256, changed.holdout_sample.sample_sha256
        )
        self.assertNotEqual(original.actual_mse, changed.actual_mse)

    def test_arrays_are_immutable_and_serialization_is_canonical(self) -> None:
        sample = self.samples[0]
        self.assertFalse(sample.x.flags.writeable)
        self.assertFalse(sample.y.flags.writeable)
        self.assertFalse(self.fit.model.coefficient.flags.writeable)
        with self.assertRaises(ValueError):
            sample.x.flags.writeable = True
        with self.assertRaises(ValueError):
            sample.x[0, 0] = 9.0

        restored_sample = SequenceSample.from_bytes(sample.to_bytes())
        self.assertEqual(restored_sample.to_bytes(), sample.to_bytes())
        self.assertFalse(restored_sample.x.flags.writeable)
        restored_fit = ContextualSequenceFit.from_bytes(self.fit.to_bytes())
        self.assertEqual(restored_fit.to_bytes(), self.fit.to_bytes())
        self.assertEqual(restored_fit.model.sha256, self.fit.model.sha256)

    def test_duplicate_prompt_and_resigned_receipt_are_rejected(self) -> None:
        duplicate = SequenceSample(
            x=self.samples[1].x,
            y=self.samples[1].y,
            prompt_sha256=self.samples[0].prompt_sha256,
            evidence_sha256=_sha("fresh-evidence"),
        )
        with self.assertRaisesRegex(ValueError, "prompts must be unique"):
            fit_contextual_sequence(
                (self.samples[0], duplicate),
                self.samples[2],
                configs=(DEFAULT_FEATURE_CONFIGS[0],),
                ridge_grid=(1e-4,),
            )

        document = json.loads(self.fit.receipt.to_bytes())
        document["body"]["train_samples"][1]["prompt_sha256"] = document["body"][
            "train_samples"
        ][0]["prompt_sha256"]
        document["sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaises(ContextualSequenceIntegrityError):
            ContextualSequenceFitReceipt.from_bytes(canonical_json_bytes(document))

    def test_relabelled_identical_content_is_rejected_everywhere(self) -> None:
        relabelled_train = SequenceSample(
            x=self.samples[1].x,
            y=self.samples[1].y,
            prompt_sha256=_sha("relabelled-train-prompt"),
            evidence_sha256=_sha("relabelled-train-evidence"),
        )
        self.assertEqual(
            relabelled_train.content_sha256, self.samples[1].content_sha256
        )
        self.assertNotEqual(
            relabelled_train.prompt_sha256, self.samples[1].prompt_sha256
        )
        with self.assertRaisesRegex(ValueError, "raw sequence content"):
            fit_contextual_sequence(
                (self.samples[0], self.samples[1]),
                relabelled_train,
                configs=(DEFAULT_FEATURE_CONFIGS[0],),
            )

        relabelled_holdout = SequenceSample(
            x=self.samples[0].x,
            y=self.samples[0].y,
            prompt_sha256=_sha("relabelled-holdout-prompt"),
            evidence_sha256=_sha("relabelled-holdout-evidence"),
        )
        with self.assertRaisesRegex(ValueError, "raw sequence content"):
            evaluate_contextual_sequence_holdout(self.fit, relabelled_holdout)

    def test_feature_semantics_are_name_independent(self) -> None:
        first = SequenceFeatureConfig("first-name", (0.5,), True)
        renamed = SequenceFeatureConfig("renamed", (0.5,), True)
        self.assertNotEqual(first.sha256, renamed.sha256)
        self.assertEqual(first.semantic_sha256, renamed.semantic_sha256)
        with self.assertRaisesRegex(ValueError, "semantically unique"):
            fit_contextual_sequence(
                self.samples[:2],
                self.samples[2],
                configs=(first, renamed),
                ridge_grid=(1e-4,),
            )

        document = first.to_dict()
        document["semantic_sha256"] = "0" * 64
        with self.assertRaises(ContextualSequenceIntegrityError):
            SequenceFeatureConfig.from_dict(document)

    def test_oversized_fit_is_rejected_before_large_allocations(self) -> None:
        leaks = tuple((index + 1) / 17.0 for index in range(16))
        oversized = SequenceFeatureConfig("oversized", leaks, True)
        with mock.patch.object(
            contextual_sequence._FixedSubstrate,
            "create",
            side_effect=AssertionError("substrate allocation must not start"),
        ):
            with self.assertRaises(ContextualSequenceCapacityError):
                fit_contextual_sequence(
                    self.samples[:2],
                    self.samples[2],
                    configs=(oversized,),
                    ridge_grid=(1e-4,),
                    reservoir_size=4096,
                )

    def test_sample_and_holdout_tamper_are_rejected(self) -> None:
        sample_document = json.loads(self.samples[0].to_bytes())
        sample_document["body"]["x"]["raw_sha256"] = "0" * 64
        with self.assertRaises(ContextualSequenceIntegrityError):
            SequenceSample.from_bytes(canonical_json_bytes(sample_document))

        evaluation = evaluate_contextual_sequence_holdout(self.fit, self.samples[3])
        evaluation_document = json.loads(evaluation.to_bytes())
        evaluation_document["body"]["actual_mse"] += 1.0
        with self.assertRaises(ContextualSequenceIntegrityError):
            ContextualSequenceHoldoutReceipt.from_bytes(
                canonical_json_bytes(evaluation_document)
            )

    def test_fit_is_deterministic_and_content_addressed_reopen_is_exact(self) -> None:
        repeated = fit_contextual_sequence(
            self.samples[:2],
            self.samples[2],
            reservoir_size=24,
            seed=42,
        )
        self.assertEqual(repeated.to_bytes(), self.fit.to_bytes())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sequence-bank"
            first_bank = ContextualSequenceBank(CrystalStore(root))
            first = first_bank.publish(self.fit)
            second = first_bank.publish(self.fit)
            self.assertTrue(first.changed)
            self.assertFalse(second.changed)
            self.assertEqual(first.payload_sha256, self.fit.sha256)

            reopened = ContextualSequenceBank(CrystalStore(root)).restore(
                self.fit.sha256
            )
            self.assertEqual(reopened.to_bytes(), self.fit.to_bytes())
            self.assertEqual(reopened.model.sha256, self.fit.model.sha256)
            self.assertTrue(CrystalStore(root).audit().clean)

    def test_holdout_identity_cannot_overlap_fit(self) -> None:
        with self.assertRaisesRegex(ValueError, "already used"):
            evaluate_contextual_sequence_holdout(self.fit, self.samples[0])


if __name__ == "__main__":
    unittest.main()
