from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from immer.runtimes.deepseek_v4.causal_prefetch import CheckpointIdentity
from immer.runtimes.deepseek_v4.route_markov import (
    LayerMarkovExpertPredictor,
    LayerTokenRoutes,
    PromptRouteObservation,
)
from immer.runtimes.deepseek_v4.route_model import (
    RouteModelArtifactError,
    build_route_model_artifact,
    load_route_model_artifact,
    validate_route_model_artifact,
    write_route_model_artifact,
)

CHECKPOINT = CheckpointIdentity(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    revision="a" * 40,
    inventory_fingerprint="b" * 64,
)


def _predictor() -> LayerMarkovExpertPredictor:
    predictor = LayerMarkovExpertPredictor(n_experts=4)
    predictor.observe(
        PromptRouteObservation(
            "training-prompt",
            (
                LayerTokenRoutes(2, ((0,), (1,))),
                LayerTokenRoutes(3, ((2,), (3,))),
            ),
        )
    )
    return predictor


class RouteModelArtifactTests(unittest.TestCase):
    def test_round_trip_restores_exact_predictor_and_metadata(self) -> None:
        predictor = _predictor()
        document = build_route_model_artifact(
            predictor,
            checkpoint=CHECKPOINT,
            role="real_markov",
            metadata={"split_seed": 17, "train_ids": ["training-prompt"]},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "route-model.json"
            write_route_model_artifact(path, document)
            loaded = load_route_model_artifact(
                path,
                expected_checkpoint=CHECKPOINT,
                expected_role="real_markov",
            )

            self.assertEqual(loaded.predictor.snapshot(), predictor.snapshot())
            self.assertEqual(loaded.sha256, document["sha256"])
            self.assertEqual(loaded.metadata["split_seed"], 17)
            self.assertEqual(
                path.read_bytes(),
                json.dumps(
                    document,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8"),
            )

    def test_tamper_role_and_checkpoint_mismatch_are_rejected(self) -> None:
        document = build_route_model_artifact(
            _predictor(),
            checkpoint=CHECKPOINT,
            role="real_markov",
        )
        tampered = copy.deepcopy(document)
        tampered["snapshot"]["layers"][0]["transitions"][0][2] += 1
        with self.assertRaisesRegex(RouteModelArtifactError, "snapshot digest"):
            validate_route_model_artifact(tampered)
        with self.assertRaisesRegex(RouteModelArtifactError, "role"):
            validate_route_model_artifact(document, expected_role="placebo_markov")
        other = CheckpointIdentity(
            repo_id=CHECKPOINT.repo_id,
            revision="c" * 40,
            inventory_fingerprint=CHECKPOINT.inventory_fingerprint,
        )
        with self.assertRaisesRegex(RouteModelArtifactError, "another checkpoint"):
            validate_route_model_artifact(document, expected_checkpoint=other)


if __name__ == "__main__":
    unittest.main()
