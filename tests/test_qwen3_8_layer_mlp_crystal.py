from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from immer.runtimes.qwen3_8.layer_mlp_crystal import (
    FEATURE_STAGE,
    LAYER_MLP_RESIDUAL_GENERIC_BANK_ENVELOPE_SCHEMA,
    LAYER_MLP_RESIDUAL_GENERIC_CRYSTAL_ENVELOPE_SCHEMA,
    LAYER_MLP_RESIDUAL_GENERIC_IDENTITY_SCHEMA,
    SOURCE_STAGE,
    TARGET_STAGE,
    Layer63MlpResidualCoverage,
    Layer63MlpResidualCrystal,
    Layer63MlpResidualCrystalBank,
    Layer63MlpResidualCrystalIdentity,
    LayerMlpResidualCoverage,
    LayerMlpResidualCrystal,
    LayerMlpResidualCrystalBank,
    LayerMlpResidualCrystalIdentity,
    LayerMlpCrystalIdentityError,
    LayerMlpCrystalIntegrityError,
)
from immer.runtimes.qwen3_8.layer_transition_crystal import (
    LayerTransitionProjectionIdentity,
)


def _sha(character: str) -> str:
    return character * 64


def _identity(
    *,
    model: str = "1",
    q4: str = "2",
    graph: str = "3",
    atlas: str = "4",
    seed: str = "5",
) -> Layer63MlpResidualCrystalIdentity:
    return Layer63MlpResidualCrystalIdentity(
        model_sha256=_sha(model),
        q4_sha256=_sha(q4),
        graph_revision_sha256=_sha(graph),
        atlas_revision_sha256=_sha(atlas),
        projection=LayerTransitionProjectionIdentity(
            hidden_dim=8,
            sketch_dim=3,
            seed_sha256=_sha(seed),
        ),
    )


def _crystal(
    identity: Layer63MlpResidualCrystalIdentity | None = None,
    *,
    residual: float = 0.25,
    feature_radius: float = 100.0,
    error_radius: float = 0.5,
    packed_bytes: int = 12288,
) -> Layer63MlpResidualCrystal:
    identity = identity or _identity()
    center = torch.zeros(identity.sketch_dim, dtype=torch.float64)
    return Layer63MlpResidualCrystal(
        identity=identity,
        operator=torch.zeros(
            (identity.sketch_dim, identity.hidden_dim),
            dtype=torch.float64,
        ),
        feature_mean=center,
        residual_mean=torch.full(
            (identity.hidden_dim,),
            residual,
            dtype=torch.float64,
        ),
        coverage=Layer63MlpResidualCoverage(
            center=center,
            feature_radius=feature_radius,
            error_radius=error_radius,
            sample_count=9,
            max_observed_error=error_radius / 2.0,
        ),
        packed_weight_bytes_avoided=packed_bytes,
        ridge=1e-8,
    )


def _generic_identity(layer_index: int) -> LayerMlpResidualCrystalIdentity:
    return LayerMlpResidualCrystalIdentity(
        model_sha256=_sha("1"),
        q4_sha256=_sha("2"),
        graph_revision_sha256=_sha("3"),
        atlas_revision_sha256=_sha("4"),
        projection=LayerTransitionProjectionIdentity(
            hidden_dim=8,
            sketch_dim=3,
            seed_sha256=_sha("5"),
        ),
        layer_index=layer_index,
    )


def _generic_crystal(
    identity: LayerMlpResidualCrystalIdentity,
) -> LayerMlpResidualCrystal:
    center = torch.zeros(identity.sketch_dim, dtype=torch.float64)
    return LayerMlpResidualCrystal(
        identity=identity,
        operator=torch.zeros(
            (identity.sketch_dim, identity.hidden_dim),
            dtype=torch.float64,
        ),
        feature_mean=center,
        residual_mean=torch.full((identity.hidden_dim,), 0.25, dtype=torch.float64),
        coverage=LayerMlpResidualCoverage(
            center=center,
            feature_radius=100.0,
            error_radius=0.5,
            sample_count=9,
            max_observed_error=0.25,
        ),
        packed_weight_bytes_avoided=12288,
        ridge=1e-8,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class LayerMlpMathTests(unittest.TestCase):
    def test_generic_identity_binds_every_layer_and_qualified_stage(self) -> None:
        identities = tuple(_generic_identity(layer) for layer in range(64))
        self.assertEqual(len({row.identity_sha256 for row in identities}), 64)
        for layer, identity in enumerate(identities):
            self.assertEqual(identity.layer_index, layer)
            self.assertEqual(
                identity.source_stage,
                f"qwen.layer.{layer}.attention.residual",
            )
            self.assertEqual(identity.feature_stage, f"qwen.layer.{layer}.mlp.input")
            self.assertEqual(identity.target_stage, f"qwen.layer.{layer}.layer.output")
            self.assertEqual(
                identity.to_record()["schema"],
                LAYER_MLP_RESIDUAL_GENERIC_IDENTITY_SCHEMA,
            )
            self.assertEqual(
                LayerMlpResidualCrystalIdentity.from_record(identity.to_record()),
                identity,
            )
        self.assertNotEqual(identities[-1].identity_sha256, _identity().identity_sha256)
        for invalid in (-1, 64, True):
            with self.assertRaises(ValueError):
                _generic_identity(invalid)

    def test_generic_identity_rejects_stage_not_derived_from_layer(self) -> None:
        record = _generic_identity(18).to_record()
        record["source_stage"] = "qwen.layer.19.attention.residual"
        with self.assertRaises(LayerMlpCrystalIntegrityError):
            LayerMlpResidualCrystalIdentity.from_record(record)

    def test_centered_ridge_matches_closed_form_and_bf16_action(self) -> None:
        identity = _identity()
        base = torch.tensor(
            [
                [1, 0, 2, -1, 3, 1, 0, -2],
                [0, 1, -1, 2, 1, -2, 3, 0],
                [2, -1, 0, 1, -2, 3, 1, 0],
                [-1, 2, 1, 0, 3, 0, -2, 1],
                [3, 1, -2, 0, 1, 2, 0, -1],
                [1, -2, 3, 1, 0, -1, 2, 0],
                [-2, 0, 1, 3, -1, 2, 1, 0],
                [0, 3, 1, -2, 2, 0, -1, 1],
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
                [-2, 1, 3, 0, 1, 0, 2, -1],
                [0, 2, -1, 1, 3, -2, 0, 1],
            ],
            dtype=torch.bfloat16,
        )
        projection = identity.projection.matrix()
        f64 = feature.to(torch.float64) @ projection
        fbar = f64.mean(dim=0)
        known_w = torch.tensor(
            [
                [0.2, -0.1, 0.3, 0.0, 0.1, -0.2, 0.05, 0.25],
                [-0.3, 0.2, 0.0, 0.1, -0.1, 0.15, 0.2, -0.05],
                [0.1, 0.0, -0.2, 0.3, 0.05, 0.1, -0.15, 0.2],
            ],
            dtype=torch.float64,
        )
        mu = torch.tensor(
            [0.25, -0.5, 0.125, 0.0, 0.25, -0.25, 0.5, -0.125],
            dtype=torch.float64,
        )
        target = (base.to(torch.float64) + mu + (f64 - fbar) @ known_w).to(
            torch.bfloat16
        )
        ridge = 1e-5

        crystal = Layer63MlpResidualCrystal.fit(
            identity=identity,
            base_hidden=base,
            mlp_input=feature,
            target_hidden=target,
            packed_weight_bytes_avoided=987654,
            ridge=ridge,
        )

        d64 = target.to(torch.float64) - base.to(torch.float64)
        expected_mu = d64.mean(dim=0)
        fc = f64 - fbar
        expected_w = torch.linalg.solve(
            fc.T @ fc + torch.eye(3, dtype=torch.float64) * ridge,
            fc.T @ (d64 - expected_mu),
        )
        self.assertTrue(torch.equal(crystal.feature_mean, fbar))
        self.assertTrue(torch.equal(crystal.residual_mean, expected_mu))
        self.assertTrue(
            torch.allclose(crystal.operator, expected_w, rtol=0, atol=1e-12)
        )

        row = 2
        replacement = crystal.apply(
            base[row : row + 1].reshape(1, 1, 8),
            feature[row : row + 1].reshape(1, 1, 8),
            max_error_radius=crystal.coverage.error_radius,
        )
        self.assertIsNotNone(replacement)
        assert replacement is not None
        expected = (
            base[row].to(torch.float64)
            + crystal.residual_mean
            + (f64[row] - crystal.feature_mean) @ crystal.operator
        ).reshape(1, 1, 8)
        self.assertTrue(torch.equal(replacement.output, expected.to(torch.bfloat16)))
        predicted = (
            base.to(torch.float64)
            + crystal.residual_mean
            + (f64 - crystal.feature_mean) @ crystal.operator
        ).to(torch.bfloat16)
        observed = torch.linalg.vector_norm(
            predicted.to(torch.float64) - target.to(torch.float64),
            dim=1,
        ).max()
        self.assertEqual(crystal.coverage.max_observed_error, float(observed.item()))

    def test_scope_pins_every_authority_and_fixed_boundary(self) -> None:
        baseline = _identity().identity_sha256
        changed = (
            _identity(model="6"),
            _identity(q4="7"),
            _identity(graph="8"),
            _identity(atlas="9"),
            _identity(seed="a"),
        )
        self.assertEqual(len({baseline, *(row.identity_sha256 for row in changed)}), 6)
        identity = _identity()
        self.assertEqual(identity.source_stage, SOURCE_STAGE)
        self.assertEqual(identity.feature_stage, FEATURE_STAGE)
        self.assertEqual(identity.target_stage, TARGET_STAGE)
        with self.assertRaisesRegex(ValueError, "source_stage"):
            Layer63MlpResidualCrystalIdentity(
                model_sha256=identity.model_sha256,
                q4_sha256=identity.q4_sha256,
                graph_revision_sha256=identity.graph_revision_sha256,
                atlas_revision_sha256=identity.atlas_revision_sha256,
                projection=identity.projection,
                source_stage="layer.input",
            )

    def test_eligibility_and_input_contract_fail_closed(self) -> None:
        base = torch.ones((1, 1, 8), dtype=torch.bfloat16)
        feature = torch.ones((1, 1, 8), dtype=torch.bfloat16)
        narrow = _crystal(feature_radius=0.0)
        self.assertIsNone(narrow.apply(base, feature, max_error_radius=1.0))
        limited = _crystal(error_radius=0.5)
        self.assertIsNone(limited.apply(base, feature, max_error_radius=0.49))
        self.assertIsNotNone(limited.apply(base, feature, max_error_radius=0.5))
        with self.assertRaisesRegex(TypeError, "bfloat16"):
            limited.apply(base.float(), feature, max_error_radius=1.0)
        with self.assertRaisesRegex(ValueError, "exact K1"):
            limited.apply(
                base,
                torch.ones((1, 2, 8), dtype=torch.bfloat16),
                max_error_radius=1.0,
            )


class LayerMlpStorageTests(unittest.TestCase):
    def test_legacy_v1_hashes_remain_byte_identical(self) -> None:
        identity = _identity()
        crystal = _crystal(identity)
        self.assertEqual(
            identity.identity_sha256,
            "0fb1a874254ff3e18cc44f23d5c58e7e718662bfd1f2b5df4917a535a1e97912",
        )
        self.assertEqual(
            crystal.crystal_sha256,
            "f74489e0a0dcb20c6ee5d781ee8e375190f5cd34926fa8bdc1bb26366a9315b1",
        )
        self.assertEqual(
            hashlib.sha256(crystal.to_bytes()).hexdigest(),
            "f82ec7abd82756642f7256606ec608e667f93ffcc8e523bcd813860a607e9e9e",
        )

    def test_generic_crystal_and_bank_roundtrip_without_legacy_relabeling(
        self,
    ) -> None:
        identity = _generic_identity(18)
        crystal = _generic_crystal(identity)
        crystal_document = json.loads(crystal.to_bytes())
        self.assertEqual(
            crystal_document["schema"],
            LAYER_MLP_RESIDUAL_GENERIC_CRYSTAL_ENVELOPE_SCHEMA,
        )
        restored = LayerMlpResidualCrystal.from_bytes(crystal.to_bytes())
        self.assertIsInstance(restored.identity, LayerMlpResidualCrystalIdentity)
        self.assertEqual(restored.to_bytes(), crystal.to_bytes())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layer18-mlp-v2.json"
            bank = LayerMlpResidualCrystalBank(path, identity)
            bank.publish(crystal)
            bank_document = json.loads(path.read_bytes())
            self.assertEqual(
                bank_document["schema"],
                LAYER_MLP_RESIDUAL_GENERIC_BANK_ENVELOPE_SCHEMA,
            )
            loaded = LayerMlpResidualCrystalBank.load(path)
            self.assertEqual(loaded.identity, identity)
            self.assertIsInstance(loaded.crystals[0], LayerMlpResidualCrystal)
            with self.assertRaises(LayerMlpCrystalIdentityError):
                LayerMlpResidualCrystalBank(path, _generic_identity(19))

    def test_generic_loaders_reject_legacy_v1_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layer63-v1.json"
            identity = _identity()
            legacy = Layer63MlpResidualCrystalBank(path, identity)
            legacy.publish(_crystal(identity))
            with self.assertRaises(LayerMlpCrystalIdentityError):
                LayerMlpResidualCrystalBank.load(path)

    def test_crystal_roundtrip_is_hash_stable_and_has_no_raw_samples(self) -> None:
        crystal = _crystal()
        encoded = crystal.to_bytes()
        restored = Layer63MlpResidualCrystal.from_bytes(encoded)
        self.assertEqual(restored.to_bytes(), encoded)
        self.assertEqual(restored.crystal_sha256, crystal.crystal_sha256)
        self.assertTrue(torch.equal(restored.operator, crystal.operator))
        text = encoded.decode("ascii")
        self.assertNotIn("base_hidden", text)
        self.assertNotIn("mlp_input", text)
        self.assertNotIn("target_hidden", text)
        self.assertFalse(hasattr(crystal, "base_hidden"))
        self.assertFalse(hasattr(crystal, "target_hidden"))

    def test_inner_tensor_tamper_is_rejected_even_after_outer_reseal(self) -> None:
        document = json.loads(_crystal().to_bytes())
        operator = document["body"]["operator"]
        raw = bytearray(base64.b64decode(operator["data"]))
        raw[0] ^= 1
        operator["data"] = base64.b64encode(raw).decode("ascii")
        document["body_sha256"] = hashlib.sha256(
            _canonical(document["body"])
        ).hexdigest()
        with self.assertRaises(LayerMlpCrystalIntegrityError):
            Layer63MlpResidualCrystal.from_bytes(_canonical(document))

    def test_publish_reopen_replace_metrics_and_identity_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layer63-mlp.json"
            identity = _identity()
            bank = Layer63MlpResidualCrystalBank(path, identity, max_crystals=4)
            crystal = _crystal(identity, packed_bytes=16384)
            bank.publish(crystal)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            reopened = Layer63MlpResidualCrystalBank(path, identity, max_crystals=4)
            loaded = Layer63MlpResidualCrystalBank.load(path)
            self.assertEqual(loaded.identity, identity)
            self.assertEqual(len(loaded.crystals), 1)
            base = torch.ones((1, 1, 8), dtype=torch.bfloat16)
            feature = torch.zeros((1, 1, 8), dtype=torch.bfloat16)
            self.assertIsNotNone(reopened.replace(base, feature, max_error_radius=0.5))
            self.assertIsNone(reopened.replace(base, feature, max_error_radius=0.49))
            metrics = reopened.metrics()
            self.assertEqual(metrics.attempts, 2)
            self.assertEqual(metrics.replacements, 1)
            self.assertEqual(metrics.fallbacks, 1)
            self.assertEqual(metrics.packed_weight_bytes_avoided, 16384)
            self.assertEqual(metrics.output_bytes_emitted, 16)
            with self.assertRaises(LayerMlpCrystalIdentityError):
                Layer63MlpResidualCrystalBank(
                    path,
                    _identity(atlas="b"),
                    max_crystals=4,
                )

    def test_concurrent_publish_preserves_both_atomic_actions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layer63-mlp.json"
            identity = _identity()
            first_bank = Layer63MlpResidualCrystalBank(path, identity)
            second_bank = Layer63MlpResidualCrystalBank(path, identity)
            first = _crystal(identity, residual=0.125)
            second = _crystal(identity, residual=0.5)
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = (
                    pool.submit(first_bank.publish, first),
                    pool.submit(second_bank.publish, second),
                )
                for future in futures:
                    future.result()
            restored = Layer63MlpResidualCrystalBank.load(path)
            self.assertEqual(
                {row.crystal_sha256 for row in restored.crystals},
                {first.crystal_sha256, second.crystal_sha256},
            )

    def test_failed_atomic_publish_keeps_memory_and_disk_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layer63-mlp.json"
            identity = _identity()
            bank = Layer63MlpResidualCrystalBank(path, identity)
            first = _crystal(identity, residual=0.125)
            second = _crystal(identity, residual=0.5)
            bank.publish(first)
            before = path.read_bytes()
            with patch(
                "immer.runtimes.qwen3_8.layer_mlp_crystal._publish_bytes",
                side_effect=LayerMlpCrystalIntegrityError("injected"),
            ):
                with self.assertRaises(LayerMlpCrystalIntegrityError):
                    bank.publish(second)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(
                tuple(row.crystal_sha256 for row in bank.crystals),
                (first.crystal_sha256,),
            )


if __name__ == "__main__":
    unittest.main()
