from __future__ import annotations

import hashlib
import json
import unittest

import numpy as np

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.mlp_pilot_router import (
    MlpPilotRouterConfig,
    MlpPilotRouterEvaluation,
    MlpPilotRouterFit,
    MlpPilotRouterIntegrityError,
    evaluate_mlp_pilot_router,
    fit_mlp_pilot_router,
    verify_mlp_pilot_router_evaluation,
    verify_mlp_pilot_router_fit,
)
from immer.runtimes.ooe.subspace_battery import (
    SubspaceCorpus,
    SubspaceObservationGroup,
    graph_revision_sha256,
    output_evidence_sha256,
    projection_evidence_sha256,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _group(
    *, logical_time: int, layer: int, prompt: str, prompt_index: int
) -> SubspaceObservationGroup:
    rows = 24
    block_size = 4
    blocks = 4
    gate = np.empty((rows, blocks * block_size), dtype=np.float64)
    up = np.ones_like(gate)
    for row in range(rows):
        active = (row + prompt_index + layer) % blocks
        for block in range(blocks):
            amplitude = 5.0 + 0.1 * row if block == active else 0.05
            gate[row, block * block_size : (block + 1) * block_size] = amplitude
    context = np.column_stack(
        (
            np.full(rows, float(layer)),
            np.arange(rows, dtype=np.float64),
            np.full(rows, float(prompt_index)),
        )
    )
    outputs = tuple(_hash(f"output:{logical_time}:{row}") for row in range(rows))
    sources = tuple(
        sorted(
            (
                _hash(f"source:context:{logical_time}"),
                _hash(f"source:projection:{logical_time}"),
            )
        )
    )
    event = _hash(f"event:{logical_time}")
    revision = graph_revision_sha256(logical_time, event)
    projection_verifier = _hash("projection-verifier")
    output_verifier = _hash("output-verifier")
    model_pin = _hash("qwen-model")
    return SubspaceObservationGroup(
        logical_time=logical_time,
        group_sha256=_hash(f"group:{logical_time}"),
        model_pin_sha256=model_pin,
        graph_revision_sha256=revision,
        graph_sequence=logical_time,
        graph_event_sha256=event,
        layer=layer,
        prompt_sha256=prompt,
        source_receipt_sha256s=sources,
        projection_verifier_sha256=projection_verifier,
        projection_evidence_sha256=projection_evidence_sha256(
            model_pin_sha256=model_pin,
            graph_revision_sha256=revision,
            layer=layer,
            source_receipt_sha256s=sources,
            projection_verifier_sha256=projection_verifier,
            context_states=context,
            gate_projection=gate,
            up_projection=up,
        ),
        output_verifier_sha256=output_verifier,
        output_evidence_sha256=output_evidence_sha256(
            output_payload_sha256s=outputs,
            output_verifier_sha256=output_verifier,
            source_receipt_sha256s=sources,
        ),
        context_states=context,
        gate_projection=gate,
        up_projection=up,
        output_payload_sha256s=outputs,
    )


def _corpus(prefix: str, prompt_count: int) -> tuple[SubspaceCorpus, tuple[str, ...]]:
    prompts = tuple(
        sorted(_hash(f"{prefix}:prompt:{row}") for row in range(prompt_count))
    )
    groups = []
    logical_time = 1
    for layer in (0, 1):
        for prompt_index, prompt in enumerate(prompts):
            groups.append(
                _group(
                    logical_time=logical_time,
                    layer=layer,
                    prompt=prompt,
                    prompt_index=prompt_index,
                )
            )
            logical_time += 1
    groups.sort(key=lambda group: (group.logical_time, group.group_sha256))
    return SubspaceCorpus(_hash("qwen-model"), tuple(groups)), prompts


class MlpPilotRouterTests(unittest.TestCase):
    def test_layer_local_residual_pilots_beat_static_ranges_and_roundtrip(self) -> None:
        corpus, prompts = _corpus("fit", 5)
        config = MlpPilotRouterConfig(
            block_size=4,
            pilot_count=1,
            selected_block_count=1,
            ridge=1e-6,
            random_seed_sha256=_hash("random-control"),
            max_working_bytes=16 * 1024**2,
        )
        fit = fit_mlp_pilot_router(
            corpus,
            train_prompt_sha256s=prompts[:3],
            calibration_prompt_sha256s=prompts[3:],
            row_role_sha256=_hash("fit-row-roles"),
            config=config,
        )
        self.assertGreater(
            fit.calibration_metrics.pilot_energy_capture,
            fit.calibration_metrics.marginal_energy_capture,
        )
        self.assertLess(fit.calibration_metrics.selected_fraction, 0.5)
        restored = MlpPilotRouterFit.from_bytes(fit.to_bytes())
        self.assertEqual(restored.to_bytes(), fit.to_bytes())
        verify_mlp_pilot_router_fit(restored, corpus)

        model = fit.models[0]
        group = corpus.groups[-1]
        blocks = model.select_blocks_from_full(
            group.gate_projection, group.up_projection
        )
        self.assertEqual(blocks.shape, (group.row_count, 1))
        self.assertEqual(
            len(model.selected_neurons(blocks[0])), model.selected_neuron_count
        )

        holdout, _holdout_prompts = _corpus("holdout", 2)
        evaluation = evaluate_mlp_pilot_router(
            fit,
            holdout,
            holdout_authority_sha256=_hash("future-generation-authority"),
            holdout_row_role_sha256=_hash("holdout-row-roles"),
        )
        self.assertGreater(
            evaluation.metrics.pilot_energy_capture,
            evaluation.metrics.marginal_energy_capture,
        )
        restored_evaluation = MlpPilotRouterEvaluation.from_bytes(evaluation.to_bytes())
        self.assertEqual(restored_evaluation.to_bytes(), evaluation.to_bytes())
        verify_mlp_pilot_router_evaluation(restored_evaluation, fit, holdout)

    def test_resealed_pilot_offset_tamper_fails(self) -> None:
        corpus, prompts = _corpus("tamper", 5)
        fit = fit_mlp_pilot_router(
            corpus,
            train_prompt_sha256s=prompts[:3],
            calibration_prompt_sha256s=prompts[3:],
            row_role_sha256=_hash("row-roles"),
            config=MlpPilotRouterConfig(
                block_size=4,
                pilot_count=1,
                selected_block_count=1,
                random_seed_sha256=_hash("random-control"),
                max_working_bytes=16 * 1024**2,
            ),
        )
        document = json.loads(fit.to_bytes())
        document["body"]["models"][0]["pilot_offsets"][0][0] = 99
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaises((ValueError, MlpPilotRouterIntegrityError)):
            MlpPilotRouterFit.from_bytes(canonical_json_bytes(document))


if __name__ == "__main__":
    unittest.main()
