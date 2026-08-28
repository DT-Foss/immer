from __future__ import annotations

import unittest
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

from immer.runtimes.ooe.mlp_pilot_output import pilot_sparse_down_projection
from immer.runtimes.ooe.mlp_pilot_residual import (
    MlpPilotAffineFit,
    MlpPilotAffineLayer,
)
from immer.runtimes.ooe.mlp_pilot_router import (
    MlpPilotRouterConfig,
    fit_mlp_pilot_router,
)
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.mlp_pilot_runtime import (
    MlpPilotSparseExecutor,
    MlpPilotTransposeEntry,
    MlpPilotTransposeManifest,
)
from immer.runtimes.qwen3_8.kernels import swiglu

from test_ooe_mlp_pilot_router import _corpus, _hash


class _Pager:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.device = torch.device("cpu")
        self.compute_dtype = torch.bfloat16
        self.tensors = tensors
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def tensor_rows(self, name: str, row_ids) -> torch.Tensor:
        ids = tuple(int(row) for row in row_ids)
        self.calls.append((name, ids))
        return self.tensors[name][torch.tensor(ids)]


class MlpPilotRuntimeTests(unittest.TestCase):
    def test_transpose_manifest_roundtrips_and_resealed_total_tamper_fails(
        self,
    ) -> None:
        entry = MlpPilotTransposeEntry(
            layer=9,
            source_tensor="layers.9.down.weight",
            transpose_tensor="layers.9.down.weight.transpose",
            shard="layer-9.safetensors",
            shape=(16, 3),
            source_raw_sha256=_hash("source"),
            transpose_raw_sha256=_hash("transpose"),
            shard_sha256=_hash("shard"),
            shard_bytes=512,
        )
        manifest = MlpPilotTransposeManifest(
            model_pin_sha256=_hash("model"),
            router_fit_sha256=_hash("router"),
            affine_fit_sha256=_hash("affine"),
            weights_index_sha256=_hash("index"),
            entries=(entry,),
        )
        self.assertEqual(
            MlpPilotTransposeManifest.from_bytes(manifest.to_bytes()).to_bytes(),
            manifest.to_bytes(),
        )
        document = json.loads(manifest.to_bytes())
        document["body"]["total_shard_bytes"] += 1
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaises(ValueError):
            MlpPilotTransposeManifest.from_bytes(canonical_json_bytes(document))

    def test_selected_row_executor_matches_two_pass_reference_and_counts_rows(
        self,
    ) -> None:
        corpus, prompts = _corpus("runtime", 5)
        router = fit_mlp_pilot_router(
            corpus,
            train_prompt_sha256s=prompts[:3],
            calibration_prompt_sha256s=prompts[3:],
            row_role_sha256=_hash("runtime-row-roles"),
            config=MlpPilotRouterConfig(
                block_size=4,
                pilot_count=1,
                selected_block_count=1,
                random_seed_sha256=_hash("runtime-random"),
                max_working_bytes=16 * 1024**2,
            ),
        )
        model = router.models[0]
        affine_layers = tuple(
            MlpPilotAffineLayer(
                layer=row.layer,
                output_dimension=3,
                scale=np.array([1.5, 0.75, 2.0], dtype=np.float64),
                bias=np.array([0.25, -0.5, 1.0], dtype=np.float64),
                training_group_sha256s=(_hash(f"runtime-training:{row.layer}"),),
            )
            for row in router.models
        )
        affine_layer = affine_layers[0]
        affine = MlpPilotAffineFit(
            model_pin_sha256=router.model_pin_sha256,
            router_fit_sha256=router.sha256,
            source_bank_state_sha256=_hash("runtime-source-bank"),
            source_corpus_sha256=_hash("runtime-source-corpus"),
            source_row_role_sha256=_hash("runtime-source-roles"),
            weights_index_sha256=_hash("runtime-weight-index"),
            source_row_count=24,
            models=affine_layers,
        )
        generator = torch.Generator().manual_seed(23)
        gate_weight = torch.randn(
            model.intermediate_dimension,
            3,
            generator=generator,
            dtype=torch.bfloat16,
        )
        up_weight = torch.randn(
            model.intermediate_dimension,
            3,
            generator=generator,
            dtype=torch.bfloat16,
        )
        down_weight = torch.randn(
            3,
            model.intermediate_dimension,
            generator=generator,
            dtype=torch.bfloat16,
        )
        base = f"model.language_model.layers.{model.layer}.mlp"
        weight_pager = _Pager(
            {
                f"{base}.gate_proj.weight": gate_weight,
                f"{base}.up_proj.weight": up_weight,
            }
        )
        transpose_pager = _Pager(
            {
                MlpPilotSparseExecutor.transpose_name(
                    model.layer
                ): down_weight.T.contiguous()
            }
        )
        executor = MlpPilotSparseExecutor(router, affine, weight_pager, transpose_pager)
        hidden = torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float32)
        actual, trace = executor.execute(hidden, layer=model.layer)

        compute_hidden = hidden.to(torch.bfloat16)
        gate = F.linear(compute_hidden, gate_weight)
        up = F.linear(compute_hidden, up_weight)
        activated = swiglu(gate, up)
        selected = model.select_blocks_from_full(
            gate.float().numpy().astype(np.float64),
            up.float().numpy().astype(np.float64),
        )
        sparse = pilot_sparse_down_projection(model, activated, down_weight, selected)
        expected = affine_layer.apply(sparse)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(trace.selected_neuron_count, model.selected_neuron_count)
        self.assertEqual(
            trace.weight_row_fraction,
            model.selected_neuron_count / model.intermediate_dimension,
        )
        self.assertEqual(
            sum(len(ids) for _name, ids in weight_pager.calls),
            2 * model.selected_neuron_count,
        )
        self.assertEqual(
            sum(len(ids) for _name, ids in transpose_pager.calls),
            model.selected_neuron_count,
        )

        pilot_ids = model.pilot_neuron_indices()
        packed_pager = _Pager(
            {
                MlpPilotSparseExecutor.pilot_name(model.layer, "gate"): gate_weight[
                    torch.from_numpy(pilot_ids)
                ],
                MlpPilotSparseExecutor.pilot_name(model.layer, "up"): up_weight[
                    torch.from_numpy(pilot_ids)
                ],
                MlpPilotSparseExecutor.pilot_name(
                    model.layer, "down_transpose"
                ): down_weight.T.contiguous()[torch.from_numpy(pilot_ids)],
            }
        )
        consolidated_weight_pager = _Pager(
            {
                f"{base}.gate_proj.weight": gate_weight,
                f"{base}.up_proj.weight": up_weight,
            }
        )
        consolidated_transpose_pager = _Pager(
            {
                MlpPilotSparseExecutor.transpose_name(
                    model.layer
                ): down_weight.T.contiguous()
            }
        )
        consolidated = MlpPilotSparseExecutor(
            router,
            affine,
            consolidated_weight_pager,
            consolidated_transpose_pager,
            pilot_pager=packed_pager,
        )
        consolidated_output, consolidated_trace = consolidated.execute(
            hidden, layer=model.layer
        )
        torch.testing.assert_close(consolidated_output, expected, rtol=0, atol=0)
        self.assertEqual(
            consolidated_trace.range_mode, "consolidated-pilot+full-blocks"
        )
        self.assertEqual(
            consolidated_trace.weight_row_fraction,
            (model.block_count * model.pilot_count + model.block_size)
            / model.intermediate_dimension,
        )


if __name__ == "__main__":
    unittest.main()
