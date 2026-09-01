from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from scripts.qwen38_layer_mlp_crystal_build import (
    _rank_sweep,
    build_parser as build_script_parser,
    run as run_build_script,
)

from immer.knowledge.livecausal import LiveGraph
from immer.runtimes.qwen3_8.layer_mlp_builder import (
    Layer63MlpTripleCollector,
    LayerMlpBuildError,
    capture_exact_layer63_mlp,
    current_atlas_revision_sha256,
    current_compute_graph_revision_sha256,
    publish_layer_mlp_residual_crystal,
)
from immer.runtimes.qwen3_8.layer_mlp_crystal import (
    FEATURE_STAGE,
    SOURCE_STAGE,
    TARGET_STAGE,
    Layer63MlpResidualCrystalBank,
    Layer63MlpResidualCrystalIdentity,
)
from immer.runtimes.qwen3_8.layer_transition_crystal import (
    LayerTransitionProjectionIdentity,
)


def _identity(
    *, graph: str = "3", atlas: str = "4"
) -> Layer63MlpResidualCrystalIdentity:
    return Layer63MlpResidualCrystalIdentity(
        model_sha256="1" * 64,
        q4_sha256="2" * 64,
        graph_revision_sha256=graph if len(graph) == 64 else graph * 64,
        atlas_revision_sha256=atlas if len(atlas) == 64 else atlas * 64,
        projection=LayerTransitionProjectionIdentity(
            hidden_dim=8,
            sketch_dim=3,
            seed_sha256="5" * 64,
        ),
    )


def _triples() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base = torch.tensor(
        [
            [1, 0, 2, -1, 3, 1, 0, -2],
            [0, 1, -1, 2, 1, -2, 3, 0],
            [2, -1, 0, 1, -2, 3, 1, 0],
            [-1, 2, 1, 0, 3, 0, -2, 1],
            [3, 1, -2, 0, 1, 2, 0, -1],
            [1, -2, 3, 1, 0, -1, 2, 0],
        ],
        dtype=torch.bfloat16,
    )
    feature = torch.tensor(
        [
            [1, 2, 0, -1, 3, 0, 1, -2],
            [2, -1, 1, 0, -2, 3, 0, 1],
            [-1, 0, 3, 1, 2, -2, 1, 0],
            [0, 3, -2, 2, 1, 1, -1, 0],
            [3, 1, 2, -2, 0, -1, 1, 0],
            [1, -2, 0, 3, -1, 2, 0, 1],
        ],
        dtype=torch.bfloat16,
    )
    target = (base.to(torch.float32) + 0.25 * feature.to(torch.float32)).to(
        torch.bfloat16
    )
    return base, feature, target


class LayerMlpBuilderTests(unittest.TestCase):
    def test_build_script_requires_exactly_one_fit_mode_before_runtime_load(
        self,
    ) -> None:
        common = [
            "--text",
            "enough text",
            "--output-bank",
            "/tmp/bank.json",
            "--atlas-root",
            "/tmp/atlas",
            "--compute-root",
            "/tmp/compute",
        ]
        for extra in ((), ("--direct-fit", "--o1-state", "/tmp/o1.json")):
            with self.subTest(extra=extra):
                args = build_script_parser().parse_args([*common, *extra])
                with self.assertRaisesRegex(LayerMlpBuildError, "exactly one"):
                    run_build_script(args)

    def test_rank_sweep_reuses_one_capture_and_reports_selected_width(self) -> None:
        base, feature, target = _triples()
        records = _rank_sweep(
            base=base,
            feature=feature,
            target=target,
            identity_fields={
                "model_sha256": "1" * 64,
                "q4_sha256": "2" * 64,
                "graph_revision_sha256": "3" * 64,
                "atlas_revision_sha256": "4" * 64,
            },
            selected_sketch_dim=3,
            projection_seed="5" * 64,
            packed_weight_bytes_avoided=123456,
            ridge=1e-6,
        )

        self.assertEqual([row["sketch_dim"] for row in records], [3])
        self.assertGreater(records[0]["operator_payload_bytes_f64"], 0)
        self.assertGreaterEqual(records[0]["error_l2_rms"], 0.0)

    def test_collector_pairs_three_existing_stages_in_strict_order(self) -> None:
        collector = Layer63MlpTripleCollector(8)
        base = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
        base = base.to(torch.bfloat16)
        feature = base + 1
        target = base + 2
        collector(63, SOURCE_STAGE, base)
        collector(63, FEATURE_STAGE, feature)
        collector(63, TARGET_STAGE, target)
        captured_base, captured_feature, captured_target = collector.tensors()
        self.assertEqual(tuple(captured_base.shape), (3, 8))
        self.assertTrue(torch.equal(captured_base, base.reshape(3, 8)))
        self.assertTrue(torch.equal(captured_feature, feature.reshape(3, 8)))
        self.assertTrue(torch.equal(captured_target, target.reshape(3, 8)))

        invalid = Layer63MlpTripleCollector(8)
        with self.assertRaisesRegex(LayerMlpBuildError, "no matching"):
            invalid(63, FEATURE_STAGE, feature)

    def test_capture_is_one_ordinary_generation_and_restores_observer(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.config = SimpleNamespace(dim=8)
                self.layer_boundary_observer = None
                self.layer_boundary_stages = ()
                self.layer_boundary_layers = None
                self.resets = 0

            def reset_state(self, *, release=True):
                self.resets += int(release)

            def generate_greedy(self, _prompt, **kwargs):
                self.testcase.assertTrue(kwargs["retain_final_state"])
                self.testcase.assertEqual(
                    self.layer_boundary_stages,
                    (SOURCE_STAGE, FEATURE_STAGE, TARGET_STAGE),
                )
                base = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
                base = base.to(torch.bfloat16)
                self.layer_boundary_observer(63, SOURCE_STAGE, base)
                self.layer_boundary_observer(63, FEATURE_STAGE, base + 1)
                self.layer_boundary_observer(63, TARGET_STAGE, base + 2)
                return (7, 8), SimpleNamespace(seconds=1.25)

        model = FakeModel()
        model.testcase = self
        base, feature, target, generated, evidence = capture_exact_layer63_mlp(
            model,
            (1, 2, 3),
            max_new_tokens=2,
            include_prefill=True,
        )
        self.assertEqual(tuple(base.shape), (3, 8))
        self.assertTrue(torch.equal(feature, base + 1))
        self.assertTrue(torch.equal(target, base + 2))
        self.assertEqual(generated, (7, 8))
        self.assertEqual(evidence.seconds, 1.25)
        self.assertEqual(model.resets, 2)
        self.assertIsNone(model.layer_boundary_observer)
        self.assertEqual(model.layer_boundary_stages, ())
        self.assertIsNone(model.layer_boundary_layers)

    def test_capture_default_excludes_prefill_and_keeps_k1_decode_rows(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.config = SimpleNamespace(dim=8)
                self.layer_boundary_observer = None
                self.layer_boundary_stages = ()
                self.layer_boundary_layers = None

            @staticmethod
            def reset_state(*, release=True):
                pass

            def generate_greedy(self, _prompt, **_kwargs):
                for width in (3, 1, 1):
                    base = torch.arange(width * 8, dtype=torch.float32).reshape(
                        1, width, 8
                    )
                    base = base.to(torch.bfloat16)
                    self.layer_boundary_observer(63, SOURCE_STAGE, base)
                    self.layer_boundary_observer(63, FEATURE_STAGE, base + 1)
                    self.layer_boundary_observer(63, TARGET_STAGE, base + 2)
                return (7, 8), SimpleNamespace(seconds=1.0)

        base, feature, target, _generated, _evidence = capture_exact_layer63_mlp(
            FakeModel(),
            (1, 2, 3),
            max_new_tokens=2,
        )

        self.assertEqual(tuple(base.shape), (2, 8))
        self.assertTrue(torch.equal(feature, base + 1))
        self.assertTrue(torch.equal(target, base + 2))

    def test_direct_publish_enforces_rank_and_never_persists_raw_triples(self) -> None:
        base, feature, target = _triples()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layer63-mlp.json"
            result = publish_layer_mlp_residual_crystal(
                bank_path=path,
                identity=_identity(),
                base_hidden=base,
                mlp_input=feature,
                target_hidden=target,
                packed_weight_bytes_avoided=123456,
                direct_fit=True,
                ridge=1e-6,
            )
            crystal = Layer63MlpResidualCrystalBank.load(path).crystals[0]
            encoded = path.read_text(encoding="ascii")

        self.assertEqual(result.calibration_rows, 6)
        self.assertEqual(result.feature_rank, 3)
        self.assertEqual(result.packed_weight_bytes_avoided, 123456)
        self.assertEqual(crystal.coverage.sample_count, 6)
        self.assertNotIn("base_hidden", encoded)
        self.assertNotIn("mlp_input", encoded)
        self.assertNotIn("target_hidden", encoded)
        self.assertFalse(hasattr(crystal, "base_hidden"))

    def test_direct_publish_rejects_missing_opt_in_rows_and_rank(self) -> None:
        base, feature, target = _triples()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layer63-mlp.json"
            with self.assertRaisesRegex(LayerMlpBuildError, "direct_fit"):
                publish_layer_mlp_residual_crystal(
                    bank_path=path,
                    identity=_identity(),
                    base_hidden=base,
                    mlp_input=feature,
                    target_hidden=target,
                    packed_weight_bytes_avoided=1,
                    direct_fit=False,
                )
            with self.assertRaisesRegex(LayerMlpBuildError, r"sketch_dim \+ 1"):
                publish_layer_mlp_residual_crystal(
                    bank_path=path,
                    identity=_identity(),
                    base_hidden=base[:3],
                    mlp_input=feature[:3],
                    target_hidden=target[:3],
                    packed_weight_bytes_avoided=1,
                    direct_fit=True,
                )
            with self.assertRaisesRegex(LayerMlpBuildError, "do not span"):
                publish_layer_mlp_residual_crystal(
                    bank_path=path,
                    identity=_identity(),
                    base_hidden=base,
                    mlp_input=torch.ones_like(feature),
                    target_hidden=target,
                    packed_weight_bytes_avoided=1,
                    direct_fit=True,
                )

    def test_graph_and_atlas_revision_pins_are_authenticated_state_digests(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            compute = root / "compute"
            compute.mkdir()
            graph_sha = current_compute_graph_revision_sha256(compute)
            atlas_root = root / "atlas"
            atlas = LiveGraph(atlas_root)
            atlas.append_segment(
                [{"outcome_key": "mlp-residual", "trigger_key": "ordinary-request"}]
            )
            atlas_sha = current_atlas_revision_sha256(atlas_root)
            _sequence, event_sha = atlas.store.revision()
        self.assertEqual(len(graph_sha), 64)
        self.assertEqual(len(atlas_sha), 64)
        self.assertNotEqual(atlas_sha, event_sha)


if __name__ == "__main__":
    unittest.main()
