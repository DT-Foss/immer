from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file

from immer.knowledge import LiveGraph
from immer.runtimes.deepseek_v4.causal_weights import (
    CausalWeightMount,
    LogicalModelIdentity,
    tensor_range_plan_from_source,
)
from immer.runtimes.ooe.operator_harvester import (
    contextual_observations_from_probe_result,
)
from immer.runtimes.ooe.qwen_mlp_evidence import (
    CaptureManifest,
    ExactMlpBoundaryCapture,
    QwenMlpEvidenceBank,
    canonical_capture_plan,
    run_capture_manifest,
)
from immer.runtimes.ooe.qwen_mlp_live import (
    LiveExactMlpCaptureRunner,
    LiveMlpAuthority,
    MlpCalibrationLock,
)
from immer.runtimes.ooe.subspace_battery import (
    SubspaceSweepConfig,
    evaluate_subspace_battery,
    fit_subspace_battery,
)
from immer.runtimes.qwen3_8.cartography_probe import (
    HiddenSketchProjection,
    ProbeCoordinateSpec,
    ProbeSpec,
    Qwen38CartographyProbe,
    prompt_token_sha256,
)
from immer.runtimes.qwen3_8.model import StreamedQwen38
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager
from immer.runtimes.qwen3_8.semantic_atlas import (
    GraphRevision,
    SemanticWeightAtlas,
)

from test_qwen3_8_cartography_probe import _write_bundle_manifest
from test_qwen3_8_model import _official_topology_tiny_config, _tiny_weights


_IDENTITY = LogicalModelIdentity(repo_id="local:mlp-live-test", revision="fixture")
_CODE_REVISION = "c" * 40


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class LiveQwenMlpCaptureTests(unittest.TestCase):
    def test_five_prompt_live_capture_replay_and_subspace_holdout(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".qwen-mlp-live-test-", dir=Path.cwd()
        ) as temporary:
            root = Path(temporary)
            (root / "weights").mkdir()
            (root / "causal").mkdir()
            config = _official_topology_tiny_config()
            save_file(_tiny_weights(config), root / "weights" / "model.safetensors")
            mount = CausalWeightMount(root, _IDENTITY, budget_mb=1024)
            names = tuple(
                str(row["name"]) for row in mount.source.inventory().get("tensors", ())
            )
            mount.bind_tensor_plans(
                tuple(
                    tensor_range_plan_from_source(mount.source, name) for name in names
                )
            )
            _write_bundle_manifest(root, mount, _IDENTITY)
            pager = Qwen38WeightPager(
                mount.source,
                device="cpu",
                compute_dtype="bfloat16",
                max_resident_bytes=4 * 1024**2,
                causal_tensor_reader=mount.tensor_reader,
            )
            model = StreamedQwen38(config, pager, max_batch_size=1, max_seq_len=8)
            try:
                projection = HiddenSketchProjection(
                    seed_sha256=_hash("mlp-live-projection"),
                    output_dimensions=4,
                )
                token_rows = (
                    (1, 4, 5),
                    (1, 4, 6),
                    (1, 4, 7),
                    (1, 4, 8),
                    (1, 4, 9),
                )
                prompt_sha256s = tuple(
                    prompt_token_sha256(tokens) for tokens in token_rows
                )
                entries = canonical_capture_plan(prompt_sha256s)
                executor = Qwen38CartographyProbe(model)
                graph = LiveGraph(root / "atlas")
                head = GraphRevision.from_live_revision(graph.store.revision())
                atlas = None
                authorities = {}
                model_pin_sha256 = None
                tokens_by_prompt = dict(zip(prompt_sha256s, token_rows, strict=True))
                for entry in entries:
                    base = f"model.language_model.layers.{entry.layer}.input_layernorm"
                    spec = ProbeSpec(
                        prompt_token_ids=tokens_by_prompt[entry.prompt_sha256],
                        prompt_sha256=entry.prompt_sha256,
                        start_layer=0,
                        stop_layer=entry.layer + 1,
                        coordinate=ProbeCoordinateSpec(
                            layer=entry.layer,
                            module=base,
                            tensor=f"{base}.weight",
                        ),
                        intervention_mode="passive",
                        code_revision=_CODE_REVISION,
                        hidden_sketch=projection,
                    )
                    result = executor.execute(spec, atlas_head_revision=head)
                    if atlas is None:
                        plans = tuple(mount.resolve_tensor_plan(name) for name in names)
                        atlas = SemanticWeightAtlas(
                            graph,
                            model_pin=result.measurement.model_pin,
                            tensor_plans=plans,
                        )
                        model_pin_sha256 = result.measurement.model_pin.sha256
                    atlas.append_measurement(result.measurement)
                    head = atlas.revision()
                    observations = contextual_observations_from_probe_result(
                        result, atlas_revision=head
                    )
                    source_state = f"qwen.layer.{entry.layer}.mlp.input-sketch"
                    matches = tuple(
                        row
                        for row in observations
                        if row.receipt.source_state == source_state
                    )
                    self.assertEqual(len(matches), 1)
                    authorities[(entry.prompt_sha256, entry.layer)] = LiveMlpAuthority(
                        layer=entry.layer,
                        prompt_sha256=entry.prompt_sha256,
                        probe_spec=spec,
                        observation=matches[0],
                        scheduler_observation_sha256=_hash(
                            f"scheduler:{entry.sha256}:{result.measurement.sha256}"
                        ),
                    )
                assert atlas is not None and model_pin_sha256 is not None
                manifest = CaptureManifest(
                    model_pin_sha256=model_pin_sha256,
                    input_manifest_sha256=_hash("live-input-manifest"),
                    prompt_sha256s=prompt_sha256s,
                    entries=entries,
                )
                bank = QwenMlpEvidenceBank(root / "mlp-bank")
                runner = LiveExactMlpCaptureRunner(
                    manifest=manifest,
                    model=model,
                    atlas=atlas,
                    authorities=authorities,
                    projection=projection,
                )
                with self.assertRaisesRegex(ValueError, "atomic capture group"):
                    run_capture_manifest(
                        manifest,
                        bank,
                        runner,
                        allowed_splits=("train", "calibration"),
                        max_new_groups=1,
                    )
                self.assertEqual(bank.state().receipt_count, 0)
                for entry in manifest.entries[:3]:
                    sink = ExactMlpBoundaryCapture(bank)
                    receipt = runner.capture(entry, sink)
                    bank.append_verified(receipt, runner)
                publications = run_capture_manifest(
                    manifest,
                    bank,
                    runner,
                    allowed_splits=("train", "calibration"),
                    max_new_groups=5,
                )
                self.assertEqual(len(publications), 2)
                remaining_publications = run_capture_manifest(
                    manifest,
                    bank,
                    runner,
                    allowed_splits=("train", "calibration"),
                )
                self.assertEqual(len(remaining_publications), 30)
                self.assertEqual(bank.state().split_counts, (25, 10, 0))
                config = SubspaceSweepConfig(k_values=(1, 2, 4, 8), quant_bits=(4, 8))
                lock, prefix_fit, _prefix_corpus = MlpCalibrationLock.create(
                    manifest=manifest, bank=bank, config=config
                )
                self.assertEqual(MlpCalibrationLock.from_bytes(lock.to_bytes()), lock)
                self.assertEqual(prefix_fit.sha256, lock.prefix_fit_sha256)
                holdout_publications = run_capture_manifest(
                    manifest, bank, runner, allowed_splits=("holdout",)
                )
                self.assertEqual(len(holdout_publications), 5)
                self.assertEqual(bank.state().split_counts, (25, 10, 5))
                self.assertTrue(bank.audit().clean)
                corpus = bank.build_subspace_corpus()
                self.assertEqual(
                    tuple(row.logical_time for row in corpus.groups),
                    tuple(range(1, 41)),
                )
                fit = fit_subspace_battery(
                    corpus,
                    train_group_indices=tuple(range(25)),
                    calibration_group_indices=tuple(range(25, 35)),
                    config=config,
                )
                lock.verify_full_fit(
                    manifest=manifest, bank=bank, corpus=corpus, fit=fit
                )
                holdout = evaluate_subspace_battery(
                    fit, corpus, holdout_group_indices=tuple(range(35, 40))
                )
                holdout.verify_against(fit, corpus)
                self.assertEqual(len(holdout.results), len(fit.models))
            finally:
                pager.close()
                mount.close()


if __name__ == "__main__":
    unittest.main()
