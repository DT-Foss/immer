from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import immer.runtimes.qwen3_8.layer_transition_crystal as layer_transition_module
from immer.runtimes.qwen3_8.cartography_probe import (
    HiddenSketchProjection,
    project_hidden_sketch,
)
from immer.runtimes.qwen3_8.layer_transition_crystal import (
    LAYER_TRANSITION_PROJECTION_ABI,
    LayerTransitionCoverage,
    LayerTransitionCrystal,
    LayerTransitionCrystalBank,
    LayerTransitionCrystalCapacityError,
    LayerTransitionCrystalIdentity,
    LayerTransitionCrystalIdentityError,
    LayerTransitionCrystalIntegrityError,
    LayerTransitionProjectionIdentity,
)


def _sha(character: str) -> str:
    return character * 64


def _identity(
    *,
    model: str = "2",
    graph: str = "4",
    atlas: str = "5",
    seed: str = "1",
) -> LayerTransitionCrystalIdentity:
    return LayerTransitionCrystalIdentity(
        model_sha256=_sha(model),
        q4_sha256=_sha("3"),
        graph_revision_sha256=_sha(graph),
        atlas_revision_sha256=_sha(atlas),
        projection=LayerTransitionProjectionIdentity(
            hidden_dim=8,
            sketch_dim=3,
            seed_sha256=_sha(seed),
        ),
    )


def _crystal(
    identity: LayerTransitionCrystalIdentity | None = None,
    *,
    bias: tuple[float, float, float] = (0.25, -0.5, 0.125),
    coverage_radius: float = 100.0,
    error_radius: float = 0.25,
    logical_bytes: int = 8192,
) -> LayerTransitionCrystal:
    identity = identity or _identity()
    return LayerTransitionCrystal(
        identity=identity,
        operator=torch.eye(identity.sketch_dim, dtype=torch.float64),
        bias=torch.tensor(bias, dtype=torch.float64),
        coverage=LayerTransitionCoverage(
            center=torch.zeros(identity.sketch_dim, dtype=torch.float64),
            sketch_radius=coverage_radius,
            error_radius=error_radius,
            sample_count=17,
            max_observed_error=error_radius / 2.0,
        ),
        logical_weight_bytes_replaced=logical_bytes,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class ProjectionIdentityTests(unittest.TestCase):
    def test_rademacher_projection_is_deterministic_full_rank_and_transport_neutral(
        self,
    ) -> None:
        first = _identity()
        second = _identity()
        projection = first.projection.matrix()

        self.assertTrue(torch.equal(projection, second.projection.matrix()))
        self.assertEqual(
            first.projection.projection_sha256,
            second.projection.projection_sha256,
        )
        self.assertEqual(projection.shape, (8, 3))
        self.assertEqual(int(torch.linalg.matrix_rank(projection).item()), 3)
        expected_magnitude = 1.0 / math.sqrt(8)
        self.assertTrue(
            torch.allclose(
                projection.abs(),
                torch.full_like(projection, expected_magnitude),
                rtol=0.0,
                atol=0.0,
            )
        )
        pinv = first.projection.pseudoinverse()
        self.assertTrue(
            torch.allclose(
                pinv @ projection,
                torch.eye(3, dtype=torch.float64),
                rtol=1e-12,
                atol=1e-12,
            )
        )
        record = first.to_record()
        self.assertNotIn("path", _canonical(record).decode("ascii"))
        self.assertNotIn("device", _canonical(record).decode("ascii"))
        self.assertEqual(
            record["projection"]["abi"],
            LAYER_TRANSITION_PROJECTION_ABI,
        )

    def test_projection_is_the_existing_o1_cartography_kernel(self) -> None:
        identity = _identity()
        hidden = torch.tensor(
            [[[1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 2.0, -0.75]]],
            dtype=torch.bfloat16,
        )
        _record, o1_array = project_hidden_sketch(
            hidden,
            HiddenSketchProjection(
                seed_sha256=identity.projection.seed_sha256,
                output_dimensions=identity.sketch_dim,
            ),
            max_elements=1024,
        )
        expected = (
            hidden.to(torch.float64).reshape(1, identity.hidden_dim)
            @ identity.projection.matrix()
        ).numpy()
        self.assertTrue(
            torch.allclose(
                torch.from_numpy(o1_array.copy()),
                torch.from_numpy(expected),
                rtol=1e-15,
                atol=1e-15,
            )
        )

    def test_every_identity_axis_changes_the_crystal_identity(self) -> None:
        baseline = _identity().identity_sha256
        identities = (
            _identity(model="6"),
            LayerTransitionCrystalIdentity(
                model_sha256=_sha("2"),
                q4_sha256=_sha("7"),
                graph_revision_sha256=_sha("4"),
                atlas_revision_sha256=_sha("5"),
                projection=_identity().projection,
            ),
            _identity(graph="8"),
            _identity(atlas="9"),
            _identity(seed="a"),
        )
        self.assertEqual(
            len({baseline, *(item.identity_sha256 for item in identities)}), 6
        )


class QuotientLiftTests(unittest.TestCase):
    def test_affine_sketch_operator_uses_moore_penrose_residual_lift(self) -> None:
        identity = _identity()
        crystal = _crystal(identity)
        hidden = torch.tensor(
            [[[1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 2.0, -0.75]]],
            dtype=torch.bfloat16,
        )
        replacement = crystal.apply(hidden, max_error_radius=0.25)
        self.assertIsNotNone(replacement)
        assert replacement is not None

        vector = hidden.to(dtype=torch.float64).reshape(-1)
        projection = identity.projection.matrix()
        pinv = identity.projection.pseudoinverse()
        sketch = vector @ projection
        next_sketch = sketch @ crystal.operator + crystal.bias
        expected = (vector + (next_sketch - sketch) @ pinv).reshape(1, 1, -1)
        expected = expected.to(dtype=torch.bfloat16)

        self.assertTrue(torch.equal(replacement.output, expected))
        self.assertEqual(replacement.output.dtype, torch.bfloat16)
        self.assertEqual(replacement.output.shape, (1, 1, 8))
        self.assertEqual(replacement.logical_weight_bytes_replaced, 8192)

    def test_exact_k1_bf16_contract_is_enforced_before_execution(self) -> None:
        crystal = _crystal()
        good = torch.ones((1, 1, 8), dtype=torch.bfloat16)
        self.assertIsNotNone(crystal.apply(good, max_error_radius=1.0))
        with self.assertRaisesRegex(TypeError, "bfloat16"):
            crystal.apply(good.float(), max_error_radius=1.0)
        with self.assertRaisesRegex(ValueError, "exact K1"):
            crystal.apply(
                torch.ones((1, 2, 8), dtype=torch.bfloat16),
                max_error_radius=1.0,
            )
        bad = good.clone()
        bad[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            crystal.apply(bad, max_error_radius=1.0)

    def test_coverage_and_error_budget_fail_closed(self) -> None:
        hidden = torch.ones((1, 1, 8), dtype=torch.bfloat16)
        too_narrow = _crystal(coverage_radius=0.0)
        self.assertIsNone(too_narrow.apply(hidden, max_error_radius=1.0))
        error_limited = _crystal(error_radius=0.5)
        self.assertIsNone(error_limited.apply(hidden, max_error_radius=0.49))
        self.assertIsNotNone(error_limited.apply(hidden, max_error_radius=0.5))
        with self.assertRaises(ValueError):
            error_limited.apply(hidden, max_error_radius=float("inf"))

    def test_fit_derives_finite_upward_error_and_coverage_radii(self) -> None:
        identity = _identity()
        source = torch.tensor(
            [
                [1.0, 0.0, 0.5, -1.0, 2.0, 0.25, -0.5, 1.5],
                [-1.0, 2.0, 1.0, 0.0, 0.5, -0.25, 1.5, -2.0],
                [0.5, 1.5, -0.5, 2.0, -1.5, 1.0, 0.0, 0.25],
                [2.0, -1.0, 0.25, 1.0, 0.0, -2.0, 0.5, 1.5],
                [-0.5, 0.25, 2.0, -1.5, 1.0, 0.5, -2.0, 0.0],
            ],
            dtype=torch.bfloat16,
        )
        target = source.clone()
        crystal = LayerTransitionCrystal.fit(
            identity=identity,
            source_hidden=source,
            target_hidden=target,
            logical_weight_bytes_replaced=123456,
            ridge=1e-9,
            coverage_guard=0.125,
            error_guard=0.25,
        )

        self.assertEqual(crystal.coverage.sample_count, source.shape[0])
        self.assertTrue(math.isfinite(crystal.coverage.sketch_radius))
        self.assertTrue(math.isfinite(crystal.coverage.error_radius))
        self.assertGreaterEqual(
            crystal.coverage.error_radius,
            crystal.coverage.max_observed_error + 0.25,
        )
        projection = identity.projection.matrix()
        sketches = source.to(torch.float64) @ projection
        center = sketches.mean(dim=0)
        observed_radius = float(
            torch.linalg.vector_norm(sketches - center, dim=1).max().item()
        )
        self.assertGreaterEqual(
            crystal.coverage.sketch_radius,
            observed_radius + 0.125,
        )

    def test_fixed_o1_affine_is_calibrated_without_refitting(self) -> None:
        identity = _identity()
        source = torch.tensor(
            [
                [1.0, 0.0, 0.5, -1.0, 2.0, 0.25, -0.5, 1.5],
                [-1.0, 2.0, 1.0, 0.0, 0.5, -0.25, 1.5, -2.0],
                [0.5, 1.5, -0.5, 2.0, -1.5, 1.0, 0.0, 0.25],
            ],
            dtype=torch.bfloat16,
        )
        operator = torch.tensor(
            [[1.0, 0.2, 0.0], [0.0, 0.9, -0.1], [0.1, 0.0, 1.1]],
            dtype=torch.float64,
        )
        bias = torch.tensor([0.25, -0.5, 0.125], dtype=torch.float64)
        projection = identity.projection.matrix()
        pinv = identity.projection.pseudoinverse()
        vectors = source.to(torch.float64)
        sketches = vectors @ projection
        target = (vectors + (sketches @ operator + bias - sketches) @ pinv).to(
            torch.bfloat16
        )
        source_sha256 = _sha("e")
        edge_sha256 = _sha("d")

        crystal = LayerTransitionCrystal.calibrate_affine(
            identity=identity,
            operator=operator,
            bias=bias,
            source_hidden=source,
            target_hidden=target,
            logical_weight_bytes_replaced=4096,
            source_compute_crystal_sha256=source_sha256,
            source_compute_edge_sha256=edge_sha256,
        )

        self.assertTrue(torch.equal(crystal.operator, operator))
        self.assertTrue(torch.equal(crystal.bias, bias))
        self.assertEqual(crystal.source_compute_crystal_sha256, source_sha256)
        self.assertEqual(crystal.source_compute_edge_sha256, edge_sha256)
        self.assertEqual(crystal.coverage.max_observed_error, 0.0)
        restored = LayerTransitionCrystal.from_bytes(crystal.to_bytes())
        self.assertEqual(restored.source_compute_crystal_sha256, source_sha256)
        self.assertEqual(restored.source_compute_edge_sha256, edge_sha256)


class SerializationTests(unittest.TestCase):
    def test_crystal_roundtrip_is_exact_and_hash_stable(self) -> None:
        crystal = _crystal()
        encoded = crystal.to_bytes()
        restored = LayerTransitionCrystal.from_bytes(encoded)
        self.assertEqual(restored.to_bytes(), encoded)
        self.assertEqual(restored.crystal_sha256, crystal.crystal_sha256)
        self.assertTrue(torch.equal(restored.operator, crystal.operator))
        self.assertTrue(torch.equal(restored.bias, crystal.bias))

    def test_tensor_accessors_cannot_mutate_the_sealed_artifact(self) -> None:
        crystal = _crystal()
        encoded = crystal.to_bytes()
        operator = crystal.operator
        bias = crystal.bias
        coverage = crystal.coverage
        operator.zero_()
        bias.zero_()
        coverage.center.fill_(999.0)
        self.assertEqual(crystal.to_bytes(), encoded)

    def test_tamper_duplicate_keys_and_noncanonical_json_are_rejected(self) -> None:
        encoded = _crystal().to_bytes()
        document = json.loads(encoded)
        operator = document["body"]["operator"]
        raw = bytearray(base64.b64decode(operator["data"]))
        raw[0] ^= 0x01
        operator["data"] = base64.b64encode(raw).decode("ascii")
        document["body_sha256"] = hashlib.sha256(
            _canonical(document["body"])
        ).hexdigest()
        with self.assertRaisesRegex(
            LayerTransitionCrystalIntegrityError,
            "tensor SHA-256 mismatch",
        ):
            LayerTransitionCrystal.from_bytes(_canonical(document))
        with self.assertRaisesRegex(
            LayerTransitionCrystalIntegrityError,
            "duplicate JSON key",
        ):
            LayerTransitionCrystal.from_bytes(b'{"schema":"a","schema":"b"}')
        with self.assertRaisesRegex(
            LayerTransitionCrystalIntegrityError,
            "not canonical JSON",
        ):
            LayerTransitionCrystal.from_bytes(encoded + b"\n")

    def test_non_finite_coverage_and_underreported_error_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            LayerTransitionCoverage(
                center=torch.zeros(3, dtype=torch.float64),
                sketch_radius=float("inf"),
                error_radius=1.0,
                sample_count=1,
                max_observed_error=0.0,
            )
        with self.assertRaisesRegex(ValueError, "below"):
            LayerTransitionCoverage(
                center=torch.zeros(3, dtype=torch.float64),
                sketch_radius=1.0,
                error_radius=0.1,
                sample_count=1,
                max_observed_error=0.2,
            )


class PersistentBankTests(unittest.TestCase):
    def test_publish_reopen_replace_and_metrics_settle_attempts_and_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layer63.json"
            identity = _identity()
            bank = LayerTransitionCrystalBank(path, identity, max_crystals=4)
            crystal = _crystal(identity, logical_bytes=16384)
            self.assertEqual(bank.publish(crystal), crystal.crystal_sha256)
            self.assertEqual(bank.publish(crystal), crystal.crystal_sha256)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

            reopened = LayerTransitionCrystalBank(path, identity, max_crystals=4)
            loaded = LayerTransitionCrystalBank.load(path)
            self.assertEqual(loaded.identity, identity)
            self.assertEqual(loaded.max_crystals, 4)
            self.assertEqual(
                tuple(item.crystal_sha256 for item in loaded.crystals),
                (crystal.crystal_sha256,),
            )
            hidden = torch.ones((1, 1, 8), dtype=torch.bfloat16)
            hit = reopened.replace(hidden, max_error_radius=0.25)
            miss = reopened.replace(hidden, max_error_radius=0.24)
            self.assertIsNotNone(hit)
            self.assertIsNone(miss)
            metrics = reopened.metrics()
            self.assertEqual(metrics.attempts, 2)
            self.assertEqual(metrics.replacements, 1)
            self.assertEqual(metrics.fallbacks, 1)
            self.assertEqual(metrics.logical_weight_bytes_replaced, 16384)
            self.assertEqual(metrics.output_bytes_emitted, 16)
            self.assertEqual(metrics.bytes_replaced, 16384)
            self.assertEqual(
                metrics.to_dict()["bytes_replaced"],
                metrics.to_dict()["logical_weight_bytes_replaced"],
            )

    def test_load_uses_one_authenticated_snapshot_and_never_reopens_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layer63.json"
            identity = _identity()
            bank = LayerTransitionCrystalBank(path, identity)
            crystal = _crystal(identity)
            bank.publish(crystal)
            snapshot = layer_transition_module._stable_regular_bytes(path)
            with patch.object(
                layer_transition_module,
                "_stable_regular_bytes",
                side_effect=(snapshot, FileNotFoundError()),
            ) as stable_read:
                loaded = LayerTransitionCrystalBank.load(path)
            self.assertEqual(stable_read.call_count, 1)
            self.assertEqual(
                tuple(item.crystal_sha256 for item in loaded.crystals),
                (crystal.crystal_sha256,),
            )

    def test_foreign_identity_capacity_and_symlink_state_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "layer63.json"
            identity = _identity()
            bank = LayerTransitionCrystalBank(path, identity, max_crystals=1)
            bank.publish(_crystal(identity))
            with self.assertRaises(LayerTransitionCrystalIdentityError):
                LayerTransitionCrystalBank(
                    path,
                    _identity(atlas="b"),
                    max_crystals=1,
                )
            with self.assertRaises(LayerTransitionCrystalIdentityError):
                bank.publish(_crystal(_identity(graph="c")))
            with self.assertRaises(LayerTransitionCrystalCapacityError):
                bank.publish(_crystal(identity, bias=(0.0, 0.0, 0.0)))

            link = root / "link.json"
            os.symlink(path, link)
            with self.assertRaises(LayerTransitionCrystalIntegrityError):
                LayerTransitionCrystalBank(link, identity, max_crystals=1)

    def test_failed_atomic_publication_leaves_memory_and_disk_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layer63.json"
            identity = _identity()
            bank = LayerTransitionCrystalBank(path, identity, max_crystals=2)
            first = _crystal(identity)
            second = _crystal(identity, bias=(0.0, 0.0, 0.0))
            bank.publish(first)
            before = path.read_bytes()
            with patch(
                "immer.runtimes.qwen3_8.layer_transition_crystal._atomic_write",
                side_effect=OSError("injected publication failure"),
            ):
                with self.assertRaises(LayerTransitionCrystalIntegrityError):
                    bank.publish(second)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(
                tuple(item.crystal_sha256 for item in bank.crystals),
                (first.crystal_sha256,),
            )

    def test_bank_file_and_inner_tensor_tamper_are_rejected_on_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layer63.json"
            identity = _identity()
            bank = LayerTransitionCrystalBank(path, identity)
            bank.publish(_crystal(identity))
            document = json.loads(path.read_bytes())
            document["body"]["crystals"][0]["bias"]["data_sha256"] = _sha("d")
            document["body_sha256"] = hashlib.sha256(
                _canonical(document["body"])
            ).hexdigest()
            path.write_bytes(_canonical(document))
            with self.assertRaises(LayerTransitionCrystalIntegrityError):
                LayerTransitionCrystalBank(path, identity)

    def test_loaded_bank_disappearance_is_not_silently_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layer63.json"
            identity = _identity()
            bank = LayerTransitionCrystalBank(path, identity)
            bank.publish(_crystal(identity))
            path.unlink()
            with self.assertRaisesRegex(
                LayerTransitionCrystalIntegrityError,
                "disappeared",
            ):
                bank.metrics()


if __name__ == "__main__":
    unittest.main()
