from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.torch import save_file
import torch

from immer.knowledge import Streamer
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.mlp_pilot_residual import (
    MlpPilotAffineFit,
    MlpPilotAffineLayer,
)
from immer.runtimes.ooe.mlp_pilot_router import (
    MlpPilotRouterConfig,
    fit_mlp_pilot_router,
)
from immer.runtimes.ooe.mlp_pilot_runtime import (
    MlpPilotSparseExecutor,
    MlpPilotTransposeEntry,
    MlpPilotTransposeManifest,
)
from immer.runtimes.ooe.mlp_pilot_weight_only import (
    MlpPilotOnlineConfig,
    MlpPilotWeightOnlyPlan,
    weight_only_model_pin,
)
from immer.runtimes.deepseek_v4.causal_weights import CausalWeightMount
from immer.runtimes.qwen3_8.fast_mlp import (
    PILOT_WEIGHT_MANIFEST_SCHEMA,
    PILOT_WEIGHT_MANIFEST_V2_SCHEMA,
    WEIGHT_ONLY_PLAN_NAME,
    Qwen38FastMlpError,
    Qwen38FastMlpPaths,
    open_qwen38_fast_mlp,
)
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager

from test_ooe_mlp_pilot_router import _corpus, _hash
from test_qwen3_8_model import _tiny_config


def _raw_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = replace(
            _tiny_config(),
            n_layers=2,
            layer_types=("linear_attention", "linear_attention"),
        )
        corpus, prompts = _corpus("fast-mount", 5)
        self.router = fit_mlp_pilot_router(
            corpus,
            train_prompt_sha256s=prompts[:3],
            calibration_prompt_sha256s=prompts[3:],
            row_role_sha256=_hash("fast-mount-roles"),
            config=MlpPilotRouterConfig(
                block_size=4,
                pilot_count=1,
                selected_block_count=1,
                random_seed_sha256=_hash("fast-mount-random"),
                max_working_bytes=16 * 1024**2,
            ),
        )
        self.weights_root = root / "weights"
        self.weights_root.mkdir()
        generator = torch.Generator().manual_seed(812)
        target_tensors: dict[str, torch.Tensor] = {}
        self.gate: dict[int, torch.Tensor] = {}
        self.up: dict[int, torch.Tensor] = {}
        self.down: dict[int, torch.Tensor] = {}
        for model in self.router.models:
            layer = model.layer
            base = f"model.language_model.layers.{layer}.mlp"
            self.gate[layer] = torch.randn(
                model.intermediate_dimension,
                self.config.dim,
                generator=generator,
                dtype=torch.bfloat16,
            )
            self.up[layer] = torch.randn(
                model.intermediate_dimension,
                self.config.dim,
                generator=generator,
                dtype=torch.bfloat16,
            )
            self.down[layer] = torch.randn(
                self.config.dim,
                model.intermediate_dimension,
                generator=generator,
                dtype=torch.bfloat16,
            )
            target_tensors[f"{base}.gate_proj.weight"] = self.gate[layer]
            target_tensors[f"{base}.up_proj.weight"] = self.up[layer]
            target_tensors[f"{base}.down_proj.weight"] = self.down[layer]
        target_shard = "model.safetensors"
        save_file(target_tensors, self.weights_root / target_shard)
        target_index = {
            "metadata": {
                "total_size": sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in target_tensors.values()
                )
            },
            "weight_map": {name: target_shard for name in sorted(target_tensors)},
        }
        self.index_bytes = canonical_json_bytes(target_index) + b"\n"
        (self.weights_root / "model.safetensors.index.json").write_bytes(
            self.index_bytes
        )
        index_sha256 = hashlib.sha256(self.index_bytes).hexdigest()
        self.affine = MlpPilotAffineFit(
            model_pin_sha256=self.router.model_pin_sha256,
            router_fit_sha256=self.router.sha256,
            source_bank_state_sha256=_hash("fast-mount-bank"),
            source_corpus_sha256=_hash("fast-mount-corpus"),
            source_row_role_sha256=_hash("fast-mount-source-roles"),
            weights_index_sha256=index_sha256,
            source_row_count=24,
            models=tuple(
                MlpPilotAffineLayer(
                    layer=model.layer,
                    output_dimension=self.config.dim,
                    scale=np.ones(self.config.dim, dtype=np.float64),
                    bias=np.zeros(self.config.dim, dtype=np.float64),
                    training_group_sha256s=(
                        _hash(f"fast-mount-training:{model.layer}"),
                    ),
                )
                for model in self.router.models
            ),
        )

        self.paths = Qwen38FastMlpPaths.from_root(root / "artifacts")
        for path in (
            self.paths.analysis_root,
            self.paths.transpose_root,
            self.paths.pilot_root,
        ):
            path.mkdir(parents=True)
        (self.paths.analysis_root / "fit.json").write_bytes(self.router.to_bytes())
        (self.paths.analysis_root / "affine-fit.json").write_bytes(
            self.affine.to_bytes()
        )

        transpose_entries = []
        transpose_map: dict[str, str] = {}
        for model in self.router.models:
            layer = model.layer
            tensor_name = MlpPilotSparseExecutor.transpose_name(layer)
            tensor = self.down[layer].T.contiguous()
            shard = f"pilot-down-transpose-layer-{layer:02d}.safetensors"
            shard_path = self.paths.transpose_root / shard
            save_file({tensor_name: tensor}, shard_path)
            transpose_map[tensor_name] = shard
            transpose_entries.append(
                MlpPilotTransposeEntry(
                    layer=layer,
                    source_tensor=(
                        f"model.language_model.layers.{layer}.mlp." "down_proj.weight"
                    ),
                    transpose_tensor=tensor_name,
                    shard=shard,
                    shape=tuple(tensor.shape),
                    source_raw_sha256=_raw_sha256(self.down[layer]),
                    transpose_raw_sha256=_raw_sha256(tensor),
                    shard_sha256=_file_sha256(shard_path),
                    shard_bytes=shard_path.stat().st_size,
                )
            )
        self.transpose_manifest = MlpPilotTransposeManifest(
            model_pin_sha256=self.router.model_pin_sha256,
            router_fit_sha256=self.router.sha256,
            affine_fit_sha256=self.affine.sha256,
            weights_index_sha256=index_sha256,
            entries=tuple(transpose_entries),
        )
        (self.paths.transpose_root / "pilot-transpose-manifest.json").write_bytes(
            self.transpose_manifest.to_bytes()
        )
        (self.paths.transpose_root / "model.safetensors.index.json").write_bytes(
            canonical_json_bytes(
                {"metadata": {}, "weight_map": dict(sorted(transpose_map.items()))}
            )
            + b"\n"
        )

        pilot_entries = []
        pilot_map: dict[str, str] = {}
        total_payload = 0
        for model in self.router.models:
            layer = model.layer
            indices = torch.from_numpy(model.pilot_neuron_indices())
            tensors = {
                MlpPilotSparseExecutor.pilot_name(layer, "gate"): self.gate[layer][
                    indices
                ].contiguous(),
                MlpPilotSparseExecutor.pilot_name(layer, "up"): self.up[layer][
                    indices
                ].contiguous(),
                MlpPilotSparseExecutor.pilot_name(layer, "down_transpose"): self.down[
                    layer
                ]
                .T.contiguous()[indices]
                .contiguous(),
            }
            shard = f"pilot-weights-layer-{layer:02d}.safetensors"
            shard_path = self.paths.pilot_root / shard
            save_file(tensors, shard_path)
            pilot_map.update({name: shard for name in tensors})
            total_payload += sum(
                tensor.numel() * tensor.element_size() for tensor in tensors.values()
            )
            pilot_entries.append(
                {
                    "layer": layer,
                    "pilot_indices_sha256": hashlib.sha256(
                        canonical_json_bytes(model.pilot_neuron_indices().tolist())
                    ).hexdigest(),
                    "shard": shard,
                    "shard_bytes": shard_path.stat().st_size,
                    "shard_sha256": _file_sha256(shard_path),
                    "tensors": [
                        {
                            "name": name,
                            "raw_sha256": _raw_sha256(tensor),
                            "shape": list(tensor.shape),
                        }
                        for name, tensor in sorted(tensors.items())
                    ],
                }
            )
        pilot_body = {
            "affine_fit_sha256": self.affine.sha256,
            "entries": pilot_entries,
            "model_pin_sha256": self.router.model_pin_sha256,
            "router_fit_sha256": self.router.sha256,
            "tensor_payload_bytes": total_payload,
            "transpose_manifest_sha256": self.transpose_manifest.sha256,
            "weights_index_sha256": index_sha256,
        }
        self.pilot_manifest = {
            "body": pilot_body,
            "body_sha256": hashlib.sha256(canonical_json_bytes(pilot_body)).hexdigest(),
            "schema": PILOT_WEIGHT_MANIFEST_SCHEMA,
        }
        (self.paths.pilot_root / "pilot-weight-manifest.json").write_bytes(
            canonical_json_bytes(self.pilot_manifest)
        )
        (self.paths.pilot_root / "model.safetensors.index.json").write_bytes(
            canonical_json_bytes(
                {"metadata": {}, "weight_map": dict(sorted(pilot_map.items()))}
            )
            + b"\n"
        )

        self.target_source = Streamer.from_local(
            self.weights_root,
            revision="target-fixture",
            budget_mb=64,
            use_cache=False,
        )
        self.target_pager = Qwen38WeightPager(
            self.target_source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=4 * 1024**2,
            close_source=False,
        )
        self.target_mount = object.__new__(CausalWeightMount)
        self.target_mount.source = self.target_source
        self.target_mount.weights_root = self.weights_root

    def close(self) -> None:
        self.target_pager.close()
        self.target_source.close()

    def open(
        self,
        *,
        layers: tuple[int, ...] | None = None,
        online_state_path: Path | None = None,
    ):
        return open_qwen38_fast_mlp(
            paths=self.paths,
            target_mount=self.target_mount,
            target_pager=self.target_pager,
            config=self.config,
            source_budget_mb=64,
            max_resident_bytes=4 * 1024**2,
            active_layers=layers,
            online_state_path=online_state_path,
        )

    def enable_weight_only(self) -> MlpPilotWeightOnlyPlan:
        index_sha256 = hashlib.sha256(self.index_bytes).hexdigest()
        plan = MlpPilotWeightOnlyPlan(
            repo_id="local/tiny-qwen",
            revision="tiny-revision",
            model_pin_sha256=weight_only_model_pin(
                repo_id="local/tiny-qwen",
                revision="tiny-revision",
                bundle_manifest_sha256=_hash("fast-mount-bundle"),
                layout_fingerprint=_hash("fast-mount-layout"),
                weights_index_sha256=index_sha256,
            ),
            bundle_manifest_sha256=_hash("fast-mount-bundle"),
            layout_fingerprint=_hash("fast-mount-layout"),
            weights_index_sha256=index_sha256,
            hidden_dimension=self.config.dim,
            n_layers=self.config.n_layers,
            config=self.router.config,
            online_config=MlpPilotOnlineConfig(
                min_confirmed_rows=2,
                min_capture=0.0,
                confirmation_interval=2,
                cold_start_sparse_waves=1,
                max_sparse_rows=4,
            ),
            models=self.router.models,
            source_layer_sha256s=tuple(
                (model.layer, _hash(f"fast-mount-weight-source:{model.layer}"))
                for model in self.router.models
            ),
        )
        affine = plan.affine_fit
        transpose = MlpPilotTransposeManifest(
            model_pin_sha256=plan.model_pin_sha256,
            router_fit_sha256=plan.sha256,
            affine_fit_sha256=affine.sha256,
            weights_index_sha256=index_sha256,
            entries=self.transpose_manifest.entries,
        )
        (self.paths.transpose_root / "pilot-transpose-manifest.json").write_bytes(
            transpose.to_bytes()
        )
        pilot_body = {
            "entries": self.pilot_manifest["body"]["entries"],
            "identity_affine_sha256": affine.sha256,
            "model_pin_sha256": plan.model_pin_sha256,
            "tensor_payload_bytes": self.pilot_manifest["body"]["tensor_payload_bytes"],
            "transpose_manifest_sha256": transpose.sha256,
            "weight_plan_sha256": plan.sha256,
            "weights_index_sha256": index_sha256,
        }
        pilot_manifest = {
            "body": pilot_body,
            "body_sha256": hashlib.sha256(canonical_json_bytes(pilot_body)).hexdigest(),
            "schema": PILOT_WEIGHT_MANIFEST_V2_SCHEMA,
        }
        (self.paths.pilot_root / "pilot-weight-manifest.json").write_bytes(
            canonical_json_bytes(pilot_manifest)
        )
        (self.paths.analysis_root / "fit.json").unlink()
        (self.paths.analysis_root / "affine-fit.json").unlink()
        (self.paths.analysis_root / WEIGHT_ONLY_PLAN_NAME).write_bytes(plan.to_bytes())
        return plan


class Qwen38FastMlpMountTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = _Fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.fixture.close()
        self.temporary.cleanup()

    def test_valid_artifacts_mount_execute_and_leave_target_pager_borrowed(
        self,
    ) -> None:
        mount = self.fixture.open(layers=(0,))
        hidden = torch.randn(1, self.fixture.config.dim, dtype=torch.bfloat16)

        source_before = self.fixture.target_source.bytes_moved()
        aux_before = mount.metrics()["source_body_bytes"]
        output, trace = mount.executor.execute(hidden, layer=0)
        single_target_bytes = self.fixture.target_source.bytes_moved() - source_before
        single_aux_bytes = mount.metrics()["source_body_bytes"] - aux_before
        source_before = self.fixture.target_source.bytes_moved()
        aux_before = mount.metrics()["source_body_bytes"]
        many, many_trace = mount.executor.execute_many((hidden, hidden), layer=0)
        repeated_k2_target_bytes = (
            self.fixture.target_source.bytes_moved() - source_before
        )
        repeated_k2_aux_bytes = mount.metrics()["source_body_bytes"] - aux_before

        self.assertEqual(tuple(output.shape), tuple(hidden.shape))
        self.assertEqual(trace.layer, 0)
        self.assertEqual(len(many), 2)
        self.assertEqual(many_trace.row_count, 2)
        self.assertEqual(repeated_k2_target_bytes, single_target_bytes)
        self.assertEqual(repeated_k2_aux_bytes, single_aux_bytes)
        self.assertEqual(many_trace.dynamic_row_reuse, 2.0)
        self.assertEqual(many_trace.down_row_reuse, 2.0)
        self.assertEqual(mount.receipt.active_layers, (0,))
        model = self.fixture.router.models[0]
        selected = dict(mount.receipt.selected_neuron_fraction_by_layer)
        transport = dict(mount.receipt.transport_row_fraction_by_layer)
        self.assertEqual(
            selected[0],
            model.selected_neuron_count / model.intermediate_dimension,
        )
        self.assertGreaterEqual(transport[0], selected[0])
        self.assertTrue(all(0.0 < value <= 1.0 for value in selected.values()))
        metrics = mount.metrics()
        self.assertGreater(metrics["source_body_bytes"], 0)
        self.assertGreater(
            self.fixture.target_pager.metrics()["direct_tensor_fills"], 0
        )
        self.assertGreater(mount.pilot_pager.metrics()["direct_tensor_fills"], 0)
        self.assertGreater(mount.transpose_pager.metrics()["direct_tensor_fills"], 0)
        mount.close()
        mount.close()
        rows = self.fixture.target_pager.tensor_rows(
            "model.language_model.layers.0.mlp.gate_proj.weight",
            (0,),
        )
        self.assertEqual(tuple(rows.shape), (1, self.fixture.config.dim))

    def test_default_mount_activates_every_fitted_layer(self) -> None:
        mount = self.fixture.open()
        try:
            self.assertEqual(mount.executor.active_layers, (0, 1))
            self.assertEqual(mount.receipt.fitted_layers, (0, 1))
        finally:
            mount.close()

    def test_target_index_mismatch_fails_without_closing_borrowed_pager(self) -> None:
        index = self.fixture.weights_root / "model.safetensors.index.json"
        index.write_bytes(canonical_json_bytes({"changed": True}))
        with self.assertRaisesRegex(Qwen38FastMlpError, "mounted Qwen weights"):
            self.fixture.open(layers=(0,))
        index.write_bytes(self.fixture.index_bytes)
        rows = self.fixture.target_pager.tensor_rows(
            "model.language_model.layers.0.mlp.gate_proj.weight",
            (0,),
        )
        self.assertEqual(tuple(rows.shape), (1, self.fixture.config.dim))

    def test_fast_bank_rejects_a_target_without_its_source_index(self) -> None:
        index = self.fixture.weights_root / "model.safetensors.index.json"
        index.unlink()
        with self.assertRaisesRegex(Qwen38FastMlpError, "indexed source"):
            self.fixture.open(layers=(0,))

    def test_target_pager_must_borrow_the_exact_mount_source(self) -> None:
        crossed = object.__new__(CausalWeightMount)
        crossed.source = object()
        crossed.weights_root = self.fixture.weights_root
        with self.assertRaisesRegex(Qwen38FastMlpError, "exact target"):
            open_qwen38_fast_mlp(
                paths=self.fixture.paths,
                target_mount=crossed,
                target_pager=self.fixture.target_pager,
                config=self.fixture.config,
                source_budget_mb=64,
                max_resident_bytes=4 * 1024**2,
                active_layers=(0,),
            )

    def test_active_shard_tamper_is_rejected_before_mount(self) -> None:
        entry = self.fixture.transpose_manifest.entries[0]
        shard = self.fixture.paths.transpose_root / entry.shard
        payload = bytearray(shard.read_bytes())
        payload[-1] ^= 1
        shard.write_bytes(payload)

        with self.assertRaisesRegex(Qwen38FastMlpError, "hash changed"):
            self.fixture.open(layers=(entry.layer,))

    def test_nonfitted_and_unsorted_layer_sets_are_rejected(self) -> None:
        for layers in ((3,), (1, 0)):
            with self.subTest(layers=layers):
                with self.assertRaisesRegex(Qwen38FastMlpError, "sorted fitted"):
                    self.fixture.open(layers=layers)

    def test_weight_only_plan_mounts_every_layer_and_persists_confirmation(
        self,
    ) -> None:
        plan = self.fixture.enable_weight_only()
        state_path = Path(self.temporary.name) / "fast-mlp-online.json"
        mount = self.fixture.open(online_state_path=state_path)
        generator = torch.Generator().manual_seed(441)
        hidden = torch.randn(
            1,
            self.fixture.config.dim,
            generator=generator,
            dtype=torch.bfloat16,
        )
        try:
            self.assertEqual(mount.executor.active_layers, (0, 1))
            self.assertEqual(mount.receipt.initialization, plan.initializer)
            self.assertEqual(
                mount.receipt.online_config_sha256,
                plan.online_config.sha256,
            )
            self.assertTrue(mount.receipt.online_state_persistent)
            cold = mount.executor.decision(layer=0, row_count=1)
            self.assertTrue(cold.use_sparse)
            mount.executor.execute(hidden, layer=0)
            self.assertFalse(mount.executor.decision(layer=0, row_count=1).use_sparse)

            gate = hidden.float() @ self.fixture.gate[0].float().T
            up = hidden.float() @ self.fixture.up[0].float().T
            activated = torch.nn.functional.silu(gate) * up
            output = activated @ self.fixture.down[0].float().T
            second_gate = gate * 0.75
            second_up = up * 1.25
            second_activated = torch.nn.functional.silu(second_gate) * second_up
            second_output = second_activated @ self.fixture.down[0].float().T
            observation = mount.executor.observe_full(
                layer=0,
                gate=torch.cat((gate, second_gate), dim=0),
                up=torch.cat((up, second_up), dim=0),
                activated=torch.cat((activated, second_activated), dim=0),
                output=torch.cat((output, second_output), dim=0),
            )
            self.assertIsNotNone(observation)
            self.assertTrue(mount.executor.decision(layer=0, row_count=1).use_sparse)
            self.assertEqual(mount.metrics()["online_confirmed_rows"], 2)
        finally:
            mount.close()

        reopened = self.fixture.open(online_state_path=state_path)
        try:
            decision = reopened.executor.decision(layer=0, row_count=1)
            self.assertEqual(decision.reason, "target-confirmed")
            self.assertEqual(decision.confirmed_rows, 2)
        finally:
            reopened.close()

    def test_weight_only_plan_rejects_mixed_legacy_analysis(self) -> None:
        self.fixture.enable_weight_only()
        (self.fixture.paths.analysis_root / "fit.json").write_bytes(
            self.fixture.router.to_bytes()
        )
        with self.assertRaisesRegex(Qwen38FastMlpError, "mixes fitted"):
            self.fixture.open()


if __name__ == "__main__":
    unittest.main()
