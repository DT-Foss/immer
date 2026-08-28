from __future__ import annotations

import json
import unittest

from immer.runtimes.ooe.markov_coordinate_selector import (
    MarkovCoordinateSelectorConfig,
    MarkovCoordinateSelectorEvaluation,
    MarkovCoordinateSelectorFit,
    MarkovCoordinateSelectorIntegrityError,
    evaluate_markov_coordinate_selector,
    fit_markov_coordinate_selector,
)
from immer.runtimes.ooe.subspace_battery import (
    SubspaceBatteryIntegrityError,
    SubspaceCorpus,
)

from test_ooe_subspace_battery import _corpus, _hash


class MarkovCoordinateSelectorTests(unittest.TestCase):
    def test_train_only_beam_entropy_gate_and_dual_holdout_are_recomputable(
        self,
    ) -> None:
        complete = _corpus()
        fit_corpus = SubspaceCorpus(complete.model_pin_sha256, complete.groups[:4])
        holdout_corpus = SubspaceCorpus(complete.model_pin_sha256, complete.groups[4:])
        config = MarkovCoordinateSelectorConfig(
            quant_bits=16,
            max_depth=2,
            beam_width=3,
            internal_validation_groups=1,
            energy_candidates=4,
            fisher_candidates=4,
            variance_candidates=4,
            random_candidates=4,
            random_rank_tags=(1, 2),
            max_candidate_pool=4,
            max_evaluated_states=64,
            fisher_block_dimensions=2,
            minimum_raw_key_bits=64,
            max_working_bytes=16 * 1024**2,
        )
        fit = fit_markov_coordinate_selector(
            fit_corpus,
            train_group_indices=(0, 1, 2),
            calibration_group_indices=(3,),
            config=config,
        )
        self.assertEqual(fit.locked_model.family, "markov")
        self.assertEqual(fit.locked_model.k, 2)
        self.assertGreaterEqual(
            2 * fit.locked_model.k * fit.locked_model.quant_bits,
            config.minimum_raw_key_bits,
        )
        self.assertTrue(fit.locked_model.calibration_safe)
        self.assertEqual(
            MarkovCoordinateSelectorFit.from_bytes(
                fit.to_bytes(), corpus=fit_corpus
            ).to_bytes(),
            fit.to_bytes(),
        )
        authority = _hash("later-external-holdout-authority")
        evaluation = evaluate_markov_coordinate_selector(
            fit,
            fit_corpus=fit_corpus,
            holdout_corpus=holdout_corpus,
            holdout_authority_sha256=authority,
        )
        self.assertEqual(
            tuple(row.name for row in evaluation.results),
            ("markov", "energy", "random", "full", "marginal"),
        )
        self.assertEqual(
            MarkovCoordinateSelectorEvaluation.from_bytes(
                evaluation.to_bytes(),
                fit=fit,
                fit_corpus=fit_corpus,
                holdout_corpus=holdout_corpus,
                holdout_authority_sha256=authority,
            ).to_bytes(),
            evaluation.to_bytes(),
        )
        markov = evaluation.results[0]
        self.assertEqual(markov.frozen_metrics.wrong_collisions, 0)
        self.assertEqual(markov.adaptive_metrics.wrong_collisions, 0)

    def test_fit_and_holdout_tamper_fail_closed(self) -> None:
        complete = _corpus()
        fit_corpus = SubspaceCorpus(complete.model_pin_sha256, complete.groups[:4])
        holdout_corpus = SubspaceCorpus(complete.model_pin_sha256, complete.groups[4:])
        config = MarkovCoordinateSelectorConfig(
            max_depth=2,
            beam_width=2,
            internal_validation_groups=1,
            energy_candidates=4,
            fisher_candidates=4,
            variance_candidates=4,
            random_candidates=4,
            random_rank_tags=(1,),
            max_candidate_pool=4,
            max_evaluated_states=32,
            fisher_block_dimensions=2,
            max_working_bytes=16 * 1024**2,
        )
        fit = fit_markov_coordinate_selector(
            fit_corpus,
            train_group_indices=(0, 1, 2),
            calibration_group_indices=(3,),
            config=config,
        )
        document = json.loads(fit.to_bytes())
        document["body"]["locked_model"]["basis_indices"][0] += 1
        document["body_sha256"] = _body_hash(document["body"])
        with self.assertRaises(
            (
                ValueError,
                MarkovCoordinateSelectorIntegrityError,
                SubspaceBatteryIntegrityError,
            )
        ):
            MarkovCoordinateSelectorFit.from_bytes(
                _canonical(document), corpus=fit_corpus
            )
        evaluation = evaluate_markov_coordinate_selector(
            fit,
            fit_corpus=fit_corpus,
            holdout_corpus=holdout_corpus,
            holdout_authority_sha256=_hash("holdout-authority"),
        )
        document = json.loads(evaluation.to_bytes())
        document["body"]["results"][0]["adaptive_metrics"]["wrong_collisions"] += 1
        document["body_sha256"] = _body_hash(document["body"])
        with self.assertRaises(
            (
                ValueError,
                MarkovCoordinateSelectorIntegrityError,
                SubspaceBatteryIntegrityError,
            )
        ):
            MarkovCoordinateSelectorEvaluation.from_bytes(
                _canonical(document),
                fit=fit,
                fit_corpus=fit_corpus,
                holdout_corpus=holdout_corpus,
                holdout_authority_sha256=_hash("holdout-authority"),
            )


def _canonical(value: object) -> bytes:
    from immer.runtimes.ooe.identity import canonical_json_bytes

    return canonical_json_bytes(value)


def _body_hash(value: object) -> str:
    import hashlib

    return hashlib.sha256(_canonical(value)).hexdigest()


if __name__ == "__main__":
    unittest.main()
