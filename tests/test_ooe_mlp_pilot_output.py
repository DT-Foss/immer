from __future__ import annotations

import unittest

import numpy as np
import torch
import torch.nn.functional as F

from immer.runtimes.ooe.mlp_pilot_output import (
    MlpPilotOutputMetricAccumulator,
    MlpPilotOutputMetrics,
    marginal_sparse_down_projection,
    pilot_sparse_down_projection,
)
from immer.runtimes.ooe.mlp_pilot_router import (
    MlpPilotRouterConfig,
    fit_mlp_pilot_router,
)

from test_ooe_mlp_pilot_router import _corpus, _hash


class MlpPilotOutputTests(unittest.TestCase):
    def test_two_pass_kernel_and_metrics_are_exactly_replayable(self) -> None:
        corpus, prompts = _corpus("output", 5)
        fit = fit_mlp_pilot_router(
            corpus,
            train_prompt_sha256s=prompts[:3],
            calibration_prompt_sha256s=prompts[3:],
            row_role_sha256=_hash("output-row-roles"),
            config=MlpPilotRouterConfig(
                block_size=4,
                pilot_count=1,
                selected_block_count=1,
                random_seed_sha256=_hash("output-random"),
                max_working_bytes=16 * 1024**2,
            ),
        )
        model = fit.models[0]
        group = next(row for row in corpus.groups if row.layer == model.layer)
        gate = torch.from_numpy(group.gate_projection.astype(np.float32)).to(
            torch.bfloat16
        )
        up = torch.from_numpy(group.up_projection.astype(np.float32)).to(torch.bfloat16)
        activated = F.silu(gate) * up
        generator = torch.Generator().manual_seed(17)
        weight = torch.randn(
            6,
            model.intermediate_dimension,
            generator=generator,
            dtype=torch.bfloat16,
        )
        selected = model.select_blocks_from_full(
            group.gate_projection, group.up_projection
        )
        predicted = pilot_sparse_down_projection(model, activated, weight, selected)
        pilots = torch.from_numpy(model.pilot_neuron_indices())
        pilot_set = set(pilots.tolist())
        blocks = selected[0]
        extra = torch.tensor(
            sorted(
                neuron
                for block in blocks
                for neuron in range(
                    int(block) * model.block_size,
                    (int(block) + 1) * model.block_size,
                )
                if neuron not in pilot_set
            )
        )
        expected_first = F.linear(activated[0, pilots], weight[:, pilots]) + F.linear(
            activated[0, extra], weight[:, extra]
        )
        self.assertTrue(torch.equal(predicted[0], expected_first))
        marginal = marginal_sparse_down_projection(model, activated, weight)
        self.assertEqual(marginal.shape, predicted.shape)

        truth = F.linear(activated, weight)
        accumulator = MlpPilotOutputMetricAccumulator()
        accumulator.add(predicted, truth)
        metrics = accumulator.result()
        self.assertEqual(
            MlpPilotOutputMetrics.from_record(metrics.to_record()).to_record(),
            metrics.to_record(),
        )
        self.assertEqual(metrics.values, truth.numel())
        self.assertGreater(metrics.cosine, 0.0)

        invalid = selected.copy()
        invalid[:, 0] = model.block_count
        with self.assertRaises(ValueError):
            pilot_sparse_down_projection(model, activated, weight, invalid)


if __name__ == "__main__":
    unittest.main()
