from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import unittest

import torch

from scripts.qwen38_layer_mlp_o1_refit import (
    build_parser as build_refit_parser,
    run as run_refit,
)

from immer.runtimes.qwen3_8.layer_mlp_builder import (
    accumulate_layer_mlp_residual_crystal,
)
from immer.runtimes.qwen3_8.layer_mlp_crystal import (
    Layer63MlpResidualCoverage,
    Layer63MlpResidualCrystal,
    Layer63MlpResidualCrystalBank,
    Layer63MlpResidualCrystalIdentity,
    LayerMlpResidualCrystal,
    LayerMlpResidualCrystalIdentity,
    LayerMlpCrystalIdentityError,
    LayerMlpCrystalIntegrityError,
)
from immer.runtimes.qwen3_8.layer_mlp_o1 import (
    LAYER_MLP_GENERIC_O1_STATS_ENVELOPE_SCHEMA,
    Layer63MlpO1Accumulator,
    LayerMlpO1Accumulator,
    LayerMlpO1Snapshot,
)
from immer.runtimes.qwen3_8.layer_transition_crystal import (
    LayerTransitionProjectionIdentity,
)


def _identity(
    *,
    hidden_dim: int = 8,
    sketch_dim: int = 3,
) -> Layer63MlpResidualCrystalIdentity:
    return Layer63MlpResidualCrystalIdentity(
        model_sha256="1" * 64,
        q4_sha256="2" * 64,
        graph_revision_sha256="3" * 64,
        atlas_revision_sha256="4" * 64,
        projection=LayerTransitionProjectionIdentity(
            hidden_dim=hidden_dim,
            sketch_dim=sketch_dim,
            seed_sha256="5" * 64,
        ),
    )


def _generic_identity(
    layer_index: int,
    *,
    hidden_dim: int = 8,
    sketch_dim: int = 3,
) -> LayerMlpResidualCrystalIdentity:
    return LayerMlpResidualCrystalIdentity(
        model_sha256="1" * 64,
        q4_sha256="2" * 64,
        graph_revision_sha256="3" * 64,
        atlas_revision_sha256="4" * 64,
        projection=LayerTransitionProjectionIdentity(
            hidden_dim=hidden_dim,
            sketch_dim=sketch_dim,
            seed_sha256="5" * 64,
        ),
        layer_index=layer_index,
    )


def _triples(
    rows: int,
    hidden_dim: int = 8,
    *,
    seed: int = 7,
    shift: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(rows, hidden_dim, generator=generator).to(torch.bfloat16)
    feature = (torch.randn(rows, hidden_dim, generator=generator) + shift).to(
        torch.bfloat16
    )
    mixing = torch.randn(hidden_dim, hidden_dim, generator=generator) * 0.04
    residual = feature.float() @ mixing + 0.15
    target = (base.float() + residual).to(torch.bfloat16)
    return base, feature, target


class LayerMlpO1Tests(unittest.TestCase):
    def test_generic_o1_accumulates_and_reopens_any_layer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for layer in (0, 18, 63):
                identity = _generic_identity(layer)
                path = root / f"layer-{layer}.json"
                accumulator = LayerMlpO1Accumulator(
                    path,
                    identity,
                    packed_weight_bytes_avoided=909 + layer,
                )
                snapshot = accumulator.observe(*_triples(12, seed=layer + 3))
                self.assertIsInstance(snapshot, LayerMlpO1Snapshot)
                self.assertTrue(snapshot.ready)
                self.assertEqual(snapshot.layer_index, layer)
                self.assertEqual(snapshot.source_stage, identity.source_stage)
                self.assertEqual(snapshot.feature_stage, identity.feature_stage)
                self.assertEqual(snapshot.target_stage, identity.target_stage)
                self.assertEqual(
                    json.loads(path.read_bytes())["schema"],
                    LAYER_MLP_GENERIC_O1_STATS_ENVELOPE_SCHEMA,
                )
                reopened = LayerMlpO1Accumulator.load(path)
                self.assertEqual(reopened.identity, identity)
                self.assertIsInstance(
                    reopened.current_crystal(),
                    LayerMlpResidualCrystal,
                )

    def test_generic_o1_rejects_cross_layer_merge_and_legacy_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = LayerMlpO1Accumulator(
                root / "left.json",
                _generic_identity(7),
                packed_weight_bytes_avoided=909,
            )
            right = LayerMlpO1Accumulator(
                root / "right.json",
                _generic_identity(8),
                packed_weight_bytes_avoided=909,
            )
            left.observe(*_triples(12, seed=7))
            right.observe(*_triples(12, seed=8))
            with self.assertRaises(LayerMlpCrystalIdentityError):
                left.merge_from(right)

            legacy_path = root / "legacy.json"
            legacy = Layer63MlpO1Accumulator(
                legacy_path,
                _identity(),
                packed_weight_bytes_avoided=909,
            )
            legacy.observe(*_triples(12, seed=9))
            with self.assertRaises(LayerMlpCrystalIdentityError):
                LayerMlpO1Accumulator.load(legacy_path)

    def test_offline_relative_ridge_refit_preserves_source_and_provenance(
        self,
    ) -> None:
        identity = _identity()
        rows = _triples(12, seed=3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.json"
            output_path = root / "refit.json"
            bank_path = root / "refit-bank.json"
            source = Layer63MlpO1Accumulator(
                source_path,
                identity,
                packed_weight_bytes_avoided=909,
                ridge=1e-8,
            )
            source.observe(*rows)
            source_before = source_path.read_bytes()
            source_snapshot = source.snapshot()
            scale = source.centered_gram_scale()
            alpha = 0.375
            receipt = run_refit(
                build_refit_parser().parse_args(
                    [
                        "--source-state",
                        str(source_path),
                        "--output-state",
                        str(output_path),
                        "--output-bank",
                        str(bank_path),
                        "--relative-ridge-alpha",
                        str(alpha),
                    ]
                )
            )
            destination = Layer63MlpO1Accumulator.load(output_path)
            destination_snapshot = destination.snapshot()
            bank_crystal = Layer63MlpResidualCrystalBank.load(bank_path).crystals[0]

            self.assertEqual(source_path.read_bytes(), source_before)

        body = receipt["body"]
        self.assertAlmostEqual(body["ridge"], alpha * scale, places=12)
        self.assertEqual(
            body["gram_scale_trace_per_sketch_dim"],
            scale,
        )
        self.assertTrue(body["source_unchanged"])
        self.assertEqual(destination.ridge, alpha * scale)
        self.assertEqual(
            destination_snapshot.source_o1_state_sha256,
            source_snapshot.state_sha256,
        )
        self.assertEqual(
            destination_snapshot.source_o1_generation,
            source_snapshot.generation,
        )
        self.assertEqual(
            bank_crystal.source_o1_state_sha256,
            destination_snapshot.state_sha256,
        )
        self.assertEqual(
            bank_crystal.source_o1_generation,
            destination_snapshot.generation,
        )
        self.assertEqual(body["crystal_sha256"], bank_crystal.crystal_sha256)
        self.assertGreater(body["operator_spectral_norm"], 0.0)

    def test_absolute_ridge_fork_reopens_and_cannot_overwrite_destination(
        self,
    ) -> None:
        identity = _identity()
        rows = _triples(9, seed=5)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = Layer63MlpO1Accumulator(
                root / "source.json",
                identity,
                packed_weight_bytes_avoided=10,
                ridge=1e-8,
            )
            source.observe(*rows)
            crystal = source.fork_with_ridge(root / "destination.json", 2.5)
            reopened = Layer63MlpO1Accumulator.load(root / "destination.json")
            reopened_snapshot = reopened.snapshot()
            with self.assertRaisesRegex(
                LayerMlpCrystalIntegrityError,
                "destination already exists",
            ):
                source.fork_with_ridge(root / "destination.json", 3.0)

        self.assertEqual(reopened.ridge, 2.5)
        self.assertEqual(
            reopened_snapshot.crystal_sha256,
            crystal.crystal_sha256,
        )

    def test_pending_then_ready_reopens_without_persisting_raw_rows(self) -> None:
        identity = _identity()
        base, feature, target = _triples(7)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layer63-mlp-o1.json"
            accumulator = Layer63MlpO1Accumulator(
                path,
                identity,
                packed_weight_bytes_avoided=123456,
                ridge=1e-6,
            )
            pending = accumulator.observe(
                base[:2],
                feature[:2],
                target[:2],
            )
            self.assertFalse(pending.ready)
            self.assertEqual(pending.feature_rank, 1)
            self.assertIsNone(accumulator.current_crystal())

            ready = accumulator.observe(base[2:], feature[2:], target[2:])
            encoded = path.read_text(encoding="ascii")
            reopened = Layer63MlpO1Accumulator.load(path)
            reopened_summary = reopened.snapshot()
            reopened_crystal = reopened.current_crystal()

        self.assertTrue(ready.ready)
        self.assertEqual(ready.accumulated_rows, 7)
        self.assertEqual(ready.observation_batches, 2)
        self.assertEqual(ready.feature_rank, identity.sketch_dim)
        self.assertEqual(reopened_summary, ready)
        self.assertIsNotNone(reopened_crystal)
        self.assertNotIn("base_hidden", encoded)
        self.assertNotIn("mlp_input", encoded)
        self.assertNotIn("target_hidden", encoded)

    def test_refit_transfers_old_error_envelope_and_covers_old_rows(self) -> None:
        identity = _identity()
        first = _triples(7, seed=11)
        second = _triples(5, seed=19, shift=2.0)
        with tempfile.TemporaryDirectory() as temporary:
            accumulator = Layer63MlpO1Accumulator(
                Path(temporary) / "o1.json",
                identity,
                packed_weight_bytes_avoided=99,
                ridge=1e-6,
            )
            accumulator.observe(*first)
            old = accumulator.current_crystal()
            self.assertIsNotNone(old)
            updated = accumulator.observe(*second)
            new = accumulator.current_crystal()
            self.assertIsNotNone(new)

        delta_mu = torch.linalg.vector_norm(
            new.residual_mean - old.residual_mean
        ).item()
        delta_w = torch.linalg.matrix_norm(
            new.operator - old.operator,
            ord=2,
        ).item()
        delta_f = torch.linalg.vector_norm(new.feature_mean - old.feature_mean).item()
        new_w = torch.linalg.matrix_norm(new.operator, ord=2).item()
        transferred = (
            old.coverage.max_observed_error
            + delta_mu
            + old.coverage.feature_radius * delta_w
            + delta_f * new_w
        )
        self.assertGreaterEqual(updated.max_observed_error, transferred - 1e-12)
        projection = identity.projection.matrix()
        base64 = first[0].to(torch.float64)
        feature64 = first[1].to(torch.float64) @ projection
        target64 = first[2].to(torch.float64)
        predicted = (
            (base64 + new.residual_mean + (feature64 - new.feature_mean) @ new.operator)
            .to(torch.bfloat16)
            .to(torch.float64)
        )
        actual = torch.linalg.vector_norm(predicted - target64, dim=1).max().item()
        self.assertLessEqual(actual, updated.max_observed_error + 1e-12)

    def test_merge_matches_additive_stats_and_reopens(self) -> None:
        identity = _identity()
        left_rows = _triples(4, seed=23)
        right_rows = _triples(5, seed=29)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = Layer63MlpO1Accumulator(
                root / "left.json",
                identity,
                packed_weight_bytes_avoided=101,
                ridge=1e-6,
            )
            right = Layer63MlpO1Accumulator(
                root / "right.json",
                identity,
                packed_weight_bytes_avoided=101,
                ridge=1e-6,
            )
            merged = Layer63MlpO1Accumulator(
                root / "merged.json",
                identity,
                packed_weight_bytes_avoided=101,
                ridge=1e-6,
            )
            direct = Layer63MlpO1Accumulator(
                root / "direct.json",
                identity,
                packed_weight_bytes_avoided=101,
                ridge=1e-6,
            )
            left.observe(*left_rows)
            right.observe(*right_rows)
            merged.merge_from(left)
            summary = merged.merge_from(root / "right.json")
            direct.observe(
                torch.cat((left_rows[0], right_rows[0])),
                torch.cat((left_rows[1], right_rows[1])),
                torch.cat((left_rows[2], right_rows[2])),
            )
            merged_crystal = merged.current_crystal()
            direct_crystal = direct.current_crystal()
            reopened = Layer63MlpO1Accumulator.load(root / "merged.json")
            reopened_summary = reopened.snapshot()

        self.assertEqual(summary.accumulated_rows, 9)
        self.assertEqual(summary.observation_batches, 2)
        self.assertTrue(summary.ready)
        self.assertTrue(
            torch.allclose(
                merged_crystal.operator,
                direct_crystal.operator,
                rtol=1e-8,
                atol=1e-8,
            )
        )
        self.assertEqual(reopened_summary, summary)

    def test_tamper_is_rejected(self) -> None:
        identity = _identity()
        rows = _triples(6)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "o1.json"
            accumulator = Layer63MlpO1Accumulator(
                path,
                identity,
                packed_weight_bytes_avoided=1,
            )
            accumulator.observe(*rows)
            raw = bytearray(path.read_bytes())
            raw[len(raw) // 2] ^= 1
            path.write_bytes(raw)
            with self.assertRaises(LayerMlpCrystalIntegrityError):
                Layer63MlpO1Accumulator.load(path)

    def test_two_writers_merge_under_one_file_lock(self) -> None:
        identity = _identity()
        first = _triples(5, seed=31)
        second = _triples(6, seed=37)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "o1.json"
            left = Layer63MlpO1Accumulator(
                path,
                identity,
                packed_weight_bytes_avoided=1,
                ridge=1e-6,
            )
            right = Layer63MlpO1Accumulator(
                path,
                identity,
                packed_weight_bytes_avoided=1,
                ridge=1e-6,
            )
            barrier = threading.Barrier(2)

            def observe(
                accumulator: Layer63MlpO1Accumulator,
                rows: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            ) -> None:
                barrier.wait()
                accumulator.observe(*rows)

            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = (
                    executor.submit(observe, left, first),
                    executor.submit(observe, right, second),
                )
                for future in futures:
                    future.result()
            final = Layer63MlpO1Accumulator.load(path).snapshot()

        self.assertEqual(final.accumulated_rows, 11)
        self.assertEqual(final.observation_batches, 2)
        self.assertEqual(final.generation, 2)

    def test_builder_publishes_only_ready_latest_crystal(self) -> None:
        identity = _identity()
        first = _triples(2, seed=41)
        second = _triples(6, seed=43)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state_path = root / "o1.json"
            bank_path = root / "bank.json"
            pending = accumulate_layer_mlp_residual_crystal(
                bank_path=bank_path,
                o1_state_path=state_path,
                identity=identity,
                base_hidden=first[0],
                mlp_input=first[1],
                target_hidden=first[2],
                packed_weight_bytes_avoided=77,
                ridge=1e-6,
            )
            self.assertFalse(pending.ready)
            self.assertFalse(bank_path.exists())
            ready = accumulate_layer_mlp_residual_crystal(
                bank_path=bank_path,
                o1_state_path=state_path,
                identity=identity,
                base_hidden=second[0],
                mlp_input=second[1],
                target_hidden=second[2],
                packed_weight_bytes_avoided=77,
                ridge=1e-6,
            )
            bank = Layer63MlpResidualCrystalBank.load(bank_path)
            first_ready = bank.crystals[0]
            third = _triples(4, seed=47, shift=1.5)
            latest = accumulate_layer_mlp_residual_crystal(
                bank_path=bank_path,
                o1_state_path=state_path,
                identity=identity,
                base_hidden=third[0],
                mlp_input=third[1],
                target_hidden=third[2],
                packed_weight_bytes_avoided=77,
                ridge=1e-6,
            )
            bank = Layer63MlpResidualCrystalBank.load(bank_path)
            self.assertEqual(len(bank.crystals), 1)
            self.assertEqual(bank.crystals[0].crystal_sha256, latest.crystal_sha256)
            self.assertNotEqual(first_ready.crystal_sha256, latest.crystal_sha256)
            stale_result = bank.publish_latest(first_ready)

            direct_bank = Layer63MlpResidualCrystalBank(
                root / "direct-bank.json",
                identity,
            )
            direct = Layer63MlpResidualCrystal(
                identity=identity,
                operator=first_ready.operator,
                feature_mean=first_ready.feature_mean,
                residual_mean=first_ready.residual_mean,
                coverage=Layer63MlpResidualCoverage(
                    center=first_ready.feature_mean,
                    feature_radius=first_ready.coverage.feature_radius,
                    error_radius=first_ready.coverage.error_radius,
                    sample_count=10_000,
                    max_observed_error=(first_ready.coverage.max_observed_error),
                ),
                packed_weight_bytes_avoided=77,
                ridge=first_ready.ridge,
            )
            with self.assertRaisesRegex(ValueError, "requires O1 state provenance"):
                direct_bank.publish_latest(direct)
            direct_bank.publish(direct)
            promoted = direct_bank.publish_latest(first_ready)
            promoted_source_sha256 = direct_bank.crystals[0].source_o1_state_sha256

            fork = Layer63MlpResidualCrystal(
                identity=identity,
                operator=first_ready.operator,
                feature_mean=first_ready.feature_mean,
                residual_mean=first_ready.residual_mean + 1e-6,
                coverage=Layer63MlpResidualCoverage(
                    center=first_ready.feature_mean,
                    feature_radius=first_ready.coverage.feature_radius,
                    error_radius=first_ready.coverage.error_radius,
                    sample_count=first_ready.coverage.sample_count,
                    max_observed_error=(first_ready.coverage.max_observed_error),
                ),
                packed_weight_bytes_avoided=77,
                ridge=first_ready.ridge,
                source_o1_state_sha256="f" * 64,
                source_o1_generation=first_ready.source_o1_generation,
            )
            conflict_bank = Layer63MlpResidualCrystalBank(
                root / "conflict-bank.json",
                identity,
            )
            conflict_bank.publish_latest(first_ready)
            with self.assertRaisesRegex(
                LayerMlpCrystalIntegrityError,
                "equal O1 generations",
            ):
                conflict_bank.publish_latest(fork)

        self.assertTrue(ready.ready)
        self.assertEqual(ready.crystal_sha256, first_ready.crystal_sha256)
        self.assertEqual(stale_result, latest.crystal_sha256)
        self.assertEqual(promoted, first_ready.crystal_sha256)
        self.assertEqual(
            promoted_source_sha256,
            first_ready.source_o1_state_sha256,
        )

    def test_sketch128_becomes_ready_across_two_109_row_requests(self) -> None:
        identity = _identity(hidden_dim=256, sketch_dim=128)
        first = _triples(109, hidden_dim=256, seed=53)
        second = _triples(109, hidden_dim=256, seed=59)
        with tempfile.TemporaryDirectory() as temporary:
            accumulator = Layer63MlpO1Accumulator(
                Path(temporary) / "o1-128.json",
                identity,
                packed_weight_bytes_avoided=1000,
                ridge=1e-6,
            )
            pending = accumulator.observe(*first)
            ready = accumulator.observe(*second)

        self.assertFalse(pending.ready)
        self.assertLess(pending.feature_rank, 128)
        self.assertTrue(ready.ready)
        self.assertEqual(ready.feature_rank, 128)
        self.assertEqual(ready.accumulated_rows, 218)


if __name__ == "__main__":
    unittest.main()
