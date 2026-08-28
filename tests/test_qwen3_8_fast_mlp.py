from __future__ import annotations

import hashlib
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
from immer.runtimes.deepseek_v4.causal_weights import CausalWeightMount
from immer.runtimes.qwen3_8.fast_mlp import (
    PILOT_WEIGHT_MANIFEST_SCHEMA,
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
        self.config = _tiny_config()
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
            "weight_map": {
                name: target_shard for name in sorted(target_tensors)
            },
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
                        f"model.language_model.layers.{layer}.mlp."
                        "down_proj.weight"
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
                MlpPilotSparseExecutor.pilot_name(
                    layer, "down_transpose"
                ): self.down[layer].T.contiguous()[indices].contiguous(),
            }
            shard = f"pilot-weights-layer-{layer:02d}.safetensors"
            shard_path = self.paths.pilot_root / shard
            save_file(tensors, shard_path)
            pilot_map.update({name: shard for name in tensors})
            total_payload += sum(
                tensor.numel() * tensor.element_size()
                for tensor in tensors.values()
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
            "body_sha256": hashlib.sha256(
                canonical_json_bytes(pilot_body)
            ).hexdigest(),
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

    def open(self, *, layers: tuple[int, ...] | None = None):
        return open_qwen38_fast_mlp(
            paths=self.paths,
            target_mount=self.target_mount,
            target_pager=self.target_pager,
            config=self.config,
            source_budget_mb=64,
            max_resident_bytes=4 * 1024**2,
            active_layers=layers,
        )


class Qwen38FastMlpMountTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = _Fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.fixture.close()
        self.temporary.cleanup()

    def test_valid_artifacts_mount_execute_and_leave_target_pager_borrowed(self) -> None:
        mount = self.fixture.open(layers=(0,))
        hidden = torch.randn(1, self.fixture.config.dim, dtype=torch.bfloat16)

        source_before = self.fixture.target_source.bytes_moved()
        output, trace = mount.executor.execute(hidden, layer=0)
        single_target_bytes = (
            self.fixture.target_source.bytes_moved() - source_before
        )
        source_before = self.fixture.target_source.bytes_moved()
        many, many_trace = mount.executor.execute_many((hidden, hidden), layer=0)
        repeated_k2_target_bytes = (
            self.fixture.target_source.bytes_moved() - source_before
        )

        self.assertEqual(tuple(output.shape), tuple(hidden.shape))
        self.assertEqual(trace.layer, 0)
        self.assertEqual(len(many), 2)
        self.assertEqual(many_trace.row_count, 2)
        self.assertEqual(repeated_k2_target_bytes, single_target_bytes)
        self.assertEqual(many_trace.dynamic_row_reuse, 2.0)
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


if __name__ == "__main__":
    unittest.main()
