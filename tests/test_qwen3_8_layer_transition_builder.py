from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import torch

from immer.knowledge.livecausal import LiveGraph
from immer.runtimes.ooe.compute_crystals import ComputeCrystal, ComputeCrystalBank
from immer.runtimes.ooe.compute_graph import ComputeOperatorGraph, OperatorEdge
from immer.runtimes.qwen3_8.layer_transition_builder import (
    LayerTransitionBuildError,
    PromotedLayerAffine,
    capture_exact_layer63,
    current_atlas_revision_sha256,
    publish_layer_transition_crystal,
    restore_promoted_layer63_affine,
)
from immer.runtimes.qwen3_8.layer_transition_crystal import (
    LayerTransitionCrystalBank,
    LayerTransitionCrystalIdentity,
    LayerTransitionProjectionIdentity,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(*, graph: str = "3", atlas: str = "4") -> LayerTransitionCrystalIdentity:
    return LayerTransitionCrystalIdentity(
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


class LayerTransitionBuilderTests(unittest.TestCase):
    def test_promoted_compute_affine_orientation_and_graph_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "compute"
            matrix = np.array(
                [[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [4.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            bias = np.array([0.25, -0.5, 0.125], dtype=np.float64)
            crystal = ComputeCrystal.affine(matrix, bias)
            bank = ComputeCrystalBank(root)
            bank.publish_crystal(crystal)
            graph = ComputeOperatorGraph(bank)
            state, changed = graph.append_edge(
                OperatorEdge(
                    source_state="qwen.layer.63.pre-hidden-sketch",
                    target_state="qwen.layer.63.post-hidden-sketch",
                    crystal_sha256=crystal.sha256,
                    verifier_sha256=_sha("verifier"),
                    evidence_sha256=_sha("evidence"),
                )
            )
            self.assertTrue(changed)

            graph_sha, promoted = restore_promoted_layer63_affine(
                root,
                sketch_dim=3,
            )

        self.assertEqual(graph_sha, state.sha256)
        self.assertIsNotNone(promoted)
        assert promoted is not None
        self.assertTrue(torch.equal(promoted.operator, torch.from_numpy(matrix.T)))
        self.assertTrue(torch.equal(promoted.bias, torch.from_numpy(bias)))
        self.assertEqual(promoted.compute_crystal_sha256, crystal.sha256)

    def test_empty_compute_graph_requires_explicit_direct_fit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            (Path(temporary) / "compute").mkdir()
            graph_sha, promoted = restore_promoted_layer63_affine(
                Path(temporary) / "compute",
                sketch_dim=3,
            )
            self.assertEqual(len(graph_sha), 64)
            self.assertIsNone(promoted)
            source = torch.ones((4, 8), dtype=torch.bfloat16)
            with self.assertRaisesRegex(LayerTransitionBuildError, "direct_fit"):
                publish_layer_transition_crystal(
                    bank_path=Path(temporary) / "bank.json",
                    identity=_identity(graph=graph_sha),
                    source_hidden=source,
                    target_hidden=source,
                    packed_weight_bytes_avoided=4096,
                    promoted=None,
                    direct_fit=False,
                )

    def test_boundary_capture_is_transient_and_pairs_every_row(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.config = SimpleNamespace(dim=8)
                self.layer_boundary_observer = None
                self.layer_boundary_stages = ()
                self.layer_boundary_layers = None
                self.resets = 0

            def reset_state(self, *, release=True):
                self.resets += int(release)

            def generate_greedy(self, _prompt, **_kwargs):
                source = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
                source = source.to(torch.bfloat16)
                self.layer_boundary_observer(63, "layer.input", source)
                self.layer_boundary_observer(63, "layer.output", source + 1)
                return (7, 8), SimpleNamespace(seconds=1.25)

        model = FakeModel()
        source, target, generated, evidence = capture_exact_layer63(
            model,
            (1, 2, 3),
            max_new_tokens=2,
        )
        self.assertEqual(tuple(source.shape), (3, 8))
        self.assertTrue(torch.equal(target, source + 1))
        self.assertEqual(generated, (7, 8))
        self.assertEqual(evidence.seconds, 1.25)
        self.assertIsNone(model.layer_boundary_observer)
        self.assertEqual(model.resets, 2)

    def test_boundary_observer_is_restored_when_state_cleanup_fails(self) -> None:
        class FailingCleanupModel:
            def __init__(self) -> None:
                self.config = SimpleNamespace(dim=8)
                self.layer_boundary_observer = None
                self.layer_boundary_stages = ()
                self.layer_boundary_layers = None
                self.resets = 0

            def reset_state(self, *, release=True):
                self.resets += int(release)
                if self.resets == 2:
                    raise RuntimeError("cleanup failed")

            def generate_greedy(self, _prompt, **_kwargs):
                source = torch.ones((1, 1, 8), dtype=torch.bfloat16)
                self.layer_boundary_observer(63, "layer.input", source)
                self.layer_boundary_observer(63, "layer.output", source)
                return (7,), SimpleNamespace(seconds=1.0)

        model = FailingCleanupModel()
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            capture_exact_layer63(model, (1,), max_new_tokens=1)
        self.assertIsNone(model.layer_boundary_observer)
        self.assertEqual(model.layer_boundary_stages, ())
        self.assertIsNone(model.layer_boundary_layers)

    def test_direct_and_promoted_publication_bind_source_without_raw_hidden(
        self,
    ) -> None:
        source = torch.tensor(
            [
                [1, 0, 1, 0, 1, 0, 1, 0],
                [0, 1, 0, 1, 0, 1, 0, 1],
                [1, 1, 0, 0, 1, 1, 0, 0],
                [0, 0, 1, 1, 0, 0, 1, 1],
            ],
            dtype=torch.bfloat16,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            direct = publish_layer_transition_crystal(
                bank_path=root / "direct.json",
                identity=_identity(),
                source_hidden=source,
                target_hidden=source,
                packed_weight_bytes_avoided=4096,
                promoted=None,
                direct_fit=True,
            )
            source_sha = _sha("compute")
            promoted = publish_layer_transition_crystal(
                bank_path=root / "promoted.json",
                identity=_identity(),
                source_hidden=source,
                target_hidden=source,
                packed_weight_bytes_avoided=4096,
                promoted=PromotedLayerAffine(
                    graph_revision_sha256="3" * 64,
                    edge_sha256=_sha("edge"),
                    compute_crystal_sha256=source_sha,
                    operator=torch.eye(3, dtype=torch.float64),
                    bias=torch.zeros(3, dtype=torch.float64),
                ),
                direct_fit=False,
            )
            direct_crystal = LayerTransitionCrystalBank.load(
                root / "direct.json"
            ).crystals[0]
            promoted_crystal = LayerTransitionCrystalBank.load(
                root / "promoted.json"
            ).crystals[0]

        self.assertIsNone(direct.source_compute_crystal_sha256)
        self.assertIsNone(direct_crystal.source_compute_crystal_sha256)
        self.assertEqual(promoted.source_compute_crystal_sha256, source_sha)
        self.assertEqual(promoted_crystal.source_compute_crystal_sha256, source_sha)
        self.assertEqual(promoted.source_edge_sha256, _sha("edge"))
        self.assertEqual(promoted_crystal.source_compute_edge_sha256, _sha("edge"))
        self.assertFalse(hasattr(direct, "source_hidden"))
        self.assertFalse(hasattr(direct, "target_hidden"))

    def test_promoted_affine_cannot_rebind_its_compute_graph(self) -> None:
        source = torch.ones((4, 8), dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(LayerTransitionBuildError, "different Compute"):
                publish_layer_transition_crystal(
                    bank_path=Path(temporary) / "bank.json",
                    identity=_identity(graph="3"),
                    source_hidden=source,
                    target_hidden=source,
                    packed_weight_bytes_avoided=4096,
                    promoted=PromotedLayerAffine(
                        graph_revision_sha256="9" * 64,
                        edge_sha256=_sha("edge"),
                        compute_crystal_sha256=_sha("compute"),
                        operator=torch.eye(3, dtype=torch.float64),
                        bias=torch.zeros(3, dtype=torch.float64),
                    ),
                    direct_fit=False,
                )

    def test_atlas_revision_uses_semantic_graph_revision_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "atlas"
            atlas = LiveGraph(root)
            atlas.append_segment(
                [{"outcome_key": "layer-63", "trigger_key": "normal-request"}]
            )
            expected = current_atlas_revision_sha256(root)
            sequence, event = atlas.store.revision()
        self.assertEqual(len(expected), 64)
        self.assertNotEqual(expected, event)
        self.assertGreater(sequence, 0)


if __name__ == "__main__":
    unittest.main()
