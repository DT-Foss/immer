from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from immer.runtimes.ooe.mlp_pilot_router import MlpPilotRouterConfig
from immer.runtimes.ooe.mlp_pilot_weight_only import (
    MlpPilotOnlineConfig,
    MlpPilotOnlineController,
    MlpPilotOnlineStateError,
    MlpPilotWeightOnlyPlan,
    build_weight_only_layer_model,
    gaussian_joint_moments,
    weight_only_model_pin,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _plan(*, min_capture: float = 0.0) -> MlpPilotWeightOnlyPlan:
    config = MlpPilotRouterConfig(
        block_size=4,
        pilot_count=1,
        selected_block_count=1,
        random_seed_sha256=_hash("weight-only-random"),
        max_working_bytes=1024**2,
    )
    moments = np.asarray(
        [[1.0, 2.0, 4.0, 8.0], [3.0, 5.0, 7.0, 11.0]],
        dtype=np.float64,
    )
    models = tuple(
        build_weight_only_layer_model(
            layer=layer,
            moments=moments + layer,
            config=config,
            source_layer_sha256=_hash(f"weight-only-source:{layer}"),
        )
        for layer in range(2)
    )
    return MlpPilotWeightOnlyPlan(
        repo_id="local/tiny-qwen",
        revision="tiny-revision",
        model_pin_sha256=weight_only_model_pin(
            repo_id="local/tiny-qwen",
            revision="tiny-revision",
            bundle_manifest_sha256=_hash("weight-only-bundle"),
            layout_fingerprint=_hash("weight-only-layout"),
            weights_index_sha256=_hash("weight-only-index"),
        ),
        bundle_manifest_sha256=_hash("weight-only-bundle"),
        layout_fingerprint=_hash("weight-only-layout"),
        weights_index_sha256=_hash("weight-only-index"),
        hidden_dimension=3,
        n_layers=2,
        config=config,
        online_config=MlpPilotOnlineConfig(
            min_confirmed_rows=2,
            min_capture=min_capture,
            confirmation_interval=2,
            cold_start_sparse_waves=1,
            statistics_decay=0.9,
            confidence_decay=0.5,
            max_confirmed_rows=32,
            max_sparse_rows=4,
        ),
        models=models,
        source_layer_sha256s=tuple(
            (layer, _hash(f"weight-only-source:{layer}")) for layer in range(2)
        ),
    )


class WeightOnlyPilotPlanTests(unittest.TestCase):
    def test_gaussian_joint_moment_is_exact_and_deterministic(self) -> None:
        gate = np.asarray([[1.0, 2.0], [2.0, 0.0]], dtype=np.float32)
        up = np.asarray([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)

        actual = gaussian_joint_moments(gate, up)

        expected = np.asarray(
            [
                (5.0 * 25.0 + 2.0 * 11.0**2) / 4.0,
                (4.0 * 25.0 + 2.0 * 0.0**2) / 4.0,
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            actual,
            gaussian_joint_moments(torch.from_numpy(gate), torch.from_numpy(up)),
        )

    def test_plan_round_trip_covers_every_layer_with_identity_affine(self) -> None:
        plan = _plan()

        restored = MlpPilotWeightOnlyPlan.from_bytes(plan.to_bytes())

        self.assertEqual(restored.to_bytes(), plan.to_bytes())
        self.assertEqual(tuple(row.layer for row in restored.models), (0, 1))
        self.assertEqual(tuple(row.layer for row in restored.affine_fit.models), (0, 1))
        for row in restored.affine_fit.models:
            np.testing.assert_array_equal(row.scale, np.ones(3))
            np.testing.assert_array_equal(row.bias, np.zeros(3))
        self.assertEqual(
            restored.models[0].pilot_neuron_indices().shape,
            (restored.models[0].block_count * restored.models[0].pilot_count,),
        )

    def test_plan_rejects_partial_layer_inventory(self) -> None:
        plan = _plan()
        with self.assertRaisesRegex(ValueError, "every layer"):
            MlpPilotWeightOnlyPlan(
                repo_id=plan.repo_id,
                revision=plan.revision,
                model_pin_sha256=plan.model_pin_sha256,
                bundle_manifest_sha256=plan.bundle_manifest_sha256,
                layout_fingerprint=plan.layout_fingerprint,
                weights_index_sha256=plan.weights_index_sha256,
                hidden_dimension=plan.hidden_dimension,
                n_layers=plan.n_layers,
                config=plan.config,
                online_config=plan.online_config,
                models=plan.models[:1],
                source_layer_sha256s=plan.source_layer_sha256s[:1],
            )


class OnlinePilotControllerTests(unittest.TestCase):
    def _exact_rows(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(918)
        return (
            torch.randn(2, 8, generator=generator),
            torch.randn(2, 8, generator=generator),
            torch.randn(2, 3, generator=generator),
        )

    def test_cold_start_requires_confirmation_then_uses_confirmed_route(self) -> None:
        controller = MlpPilotOnlineController(_plan())
        gate, up, output = self._exact_rows()
        activated = torch.nn.functional.silu(gate) * up
        try:
            cold = controller.decision(layer=0, row_count=1)
            self.assertTrue(cold.use_sparse)
            self.assertEqual(cold.reason, "weight-only-cold-start")

            controller.record_sparse(layer=0, row_count=1)
            due = controller.decision(layer=0, row_count=1)
            self.assertFalse(due.use_sparse)
            self.assertEqual(due.reason, "confirmation-required")

            observation = controller.observe_full(
                layer=0,
                gate=(gate[:1], gate[1:]),
                up=(up[:1], up[1:]),
                activated=(activated[:1], activated[1:]),
                output=(output[:1], output[1:]),
            )
            self.assertEqual(observation.rows, 2)
            self.assertEqual(observation.confirmed_rows, 2)
            confirmed = controller.decision(layer=0, row_count=1)
            self.assertTrue(confirmed.use_sparse)
            self.assertEqual(confirmed.reason, "target-confirmed")
        finally:
            controller.close()

    def test_low_confirmed_capture_falls_back_and_periodic_check_is_bounded(
        self,
    ) -> None:
        controller = MlpPilotOnlineController(_plan(min_capture=1.0))
        gate, up, output = self._exact_rows()
        try:
            observation = controller.observe_full(
                layer=0,
                gate=gate,
                up=up,
                output=output,
            )
            self.assertTrue(observation.surprising)
            decision = controller.decision(layer=0, row_count=1)
            self.assertFalse(decision.use_sparse)
            self.assertEqual(decision.reason, "low-confirmed-capture")
            wide = controller.decision(layer=0, row_count=5)
            self.assertFalse(wide.use_sparse)
            self.assertEqual(wide.reason, "row-width")
        finally:
            controller.close()

    def test_persistent_state_round_trip_and_tamper_rejection(self) -> None:
        plan = _plan()
        gate, up, output = self._exact_rows()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "online-state.json"
            controller = MlpPilotOnlineController(plan, state_path=path)
            controller.observe_full(layer=0, gate=gate, up=up, output=output)
            controller.close()
            first_size = path.stat().st_size
            self.assertLess(first_size, 128 * 1024)

            restored = MlpPilotOnlineController(plan, state_path=path)
            try:
                metrics = restored.metrics()["layers"]
                self.assertEqual(metrics[0]["confirmed_rows"], 2)
                for _ in range(4):
                    restored.observe_full(layer=0, gate=gate, up=up, output=output)
            finally:
                restored.close()
            self.assertLess(path.stat().st_size, first_size + 128)

            payload = bytearray(path.read_bytes())
            payload[-1] ^= 1
            path.write_bytes(payload)
            with self.assertRaises(MlpPilotOnlineStateError):
                MlpPilotOnlineController(plan, state_path=path)

    def test_layer_updates_flush_once_at_request_boundary(self) -> None:
        controller = MlpPilotOnlineController(_plan())
        gate, up, output = self._exact_rows()
        with mock.patch.object(
            controller,
            "_persist_locked",
            wraps=controller._persist_locked,
        ) as persist:
            controller.observe_full(layer=0, gate=gate, up=up, output=output)
            controller.observe_full(layer=1, gate=gate, up=up, output=output)
            self.assertEqual(persist.call_count, 0)
            controller.flush()
            self.assertEqual(persist.call_count, 1)
            controller.flush()
            self.assertEqual(persist.call_count, 1)
            controller.record_sparse(layer=0, row_count=1)
            controller.close()
            self.assertEqual(persist.call_count, 2)


if __name__ == "__main__":
    unittest.main()
