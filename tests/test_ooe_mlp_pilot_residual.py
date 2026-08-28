from __future__ import annotations

import json
import unittest

import numpy as np
import torch

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.mlp_pilot_output import MlpPilotOutputMetricAccumulator
from immer.runtimes.ooe.mlp_pilot_residual import (
    MlpPilotAffineAccumulator,
    MlpPilotAffineEvaluation,
    MlpPilotAffineFit,
    MlpPilotResidualIntegrityError,
)

from test_ooe_mlp_pilot_router import _hash


class MlpPilotResidualTests(unittest.TestCase):
    def test_diagonal_affine_crystal_fits_applies_and_roundtrips(self) -> None:
        generator = np.random.default_rng(7)
        x = generator.normal(size=(128, 4)).astype(np.float32)
        scale = np.array([2.0, -0.5, 0.75, 1.25], dtype=np.float32)
        bias = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float32)
        y = x * scale + bias
        accumulator = MlpPilotAffineAccumulator(4)
        accumulator.add(torch.from_numpy(x), torch.from_numpy(y))
        layer = accumulator.fit(
            layer=9,
            training_group_sha256s=(_hash("group-a"), _hash("group-b")),
        )
        corrected = layer.apply(torch.from_numpy(x))
        self.assertTrue(torch.allclose(corrected, torch.from_numpy(y), atol=1e-5))

        fit = MlpPilotAffineFit(
            model_pin_sha256=_hash("model"),
            router_fit_sha256=_hash("router-fit"),
            source_bank_state_sha256=_hash("source-bank"),
            source_corpus_sha256=_hash("source-corpus"),
            source_row_role_sha256=_hash("source-roles"),
            weights_index_sha256=_hash("weights-index"),
            source_row_count=len(x),
            models=(layer,),
        )
        restored = MlpPilotAffineFit.from_bytes(fit.to_bytes())
        self.assertEqual(restored.to_bytes(), fit.to_bytes())

        raw_accumulator = MlpPilotOutputMetricAccumulator()
        raw_accumulator.add(torch.from_numpy(x), torch.from_numpy(y))
        corrected_accumulator = MlpPilotOutputMetricAccumulator()
        corrected_accumulator.add(corrected, torch.from_numpy(y))
        raw_metrics = raw_accumulator.result()
        corrected_metrics = corrected_accumulator.result()
        evaluation = MlpPilotAffineEvaluation(
            fit_sha256=fit.sha256,
            router_evaluation_sha256=_hash("router-evaluation"),
            holdout_bank_state_sha256=_hash("holdout-bank"),
            holdout_corpus_sha256=_hash("holdout-corpus"),
            holdout_row_role_sha256=_hash("holdout-roles"),
            row_count=len(x),
            raw_metrics=raw_metrics,
            corrected_metrics=corrected_metrics,
            layer_metrics=((9, raw_metrics, corrected_metrics),),
        )
        self.assertGreater(evaluation.l2_error_reduction, 0.99)
        self.assertEqual(
            MlpPilotAffineEvaluation.from_bytes(evaluation.to_bytes()).to_bytes(),
            evaluation.to_bytes(),
        )

    def test_resealed_affine_coefficient_tamper_fails(self) -> None:
        x = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 8.0]])
        y = x * 2.0 + 1.0
        accumulator = MlpPilotAffineAccumulator(2)
        accumulator.add(x, y)
        layer = accumulator.fit(
            layer=0, training_group_sha256s=(_hash("training-group"),)
        )
        fit = MlpPilotAffineFit(
            model_pin_sha256=_hash("model"),
            router_fit_sha256=_hash("router-fit"),
            source_bank_state_sha256=_hash("source-bank"),
            source_corpus_sha256=_hash("source-corpus"),
            source_row_role_sha256=_hash("source-roles"),
            weights_index_sha256=_hash("weights-index"),
            source_row_count=3,
            models=(layer,),
        )
        document = json.loads(fit.to_bytes())
        document["body"]["models"][0]["scale"]["data_base64"] = "AAAA"
        document["body_sha256"] = _hash_body(document["body"])
        with self.assertRaises((ValueError, MlpPilotResidualIntegrityError)):
            MlpPilotAffineFit.from_bytes(canonical_json_bytes(document))


def _hash_body(value: object) -> str:
    import hashlib

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


if __name__ == "__main__":
    unittest.main()
