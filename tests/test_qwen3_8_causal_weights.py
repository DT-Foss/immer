from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from safetensors.torch import save_file

from immer.knowledge import Streamer
from immer.runtimes.deepseek_v4.causal_weights import (
    CausalWeightConflictError,
    CausalWeightMount,
    CausalWeightNotFoundError,
    LogicalModelIdentity,
    semantic_tensor_key,
    tensor_range_plan_from_source,
)
from immer.runtimes.qwen3_8 import (
    OFFICIAL_REVISION,
    Qwen38WeightPager,
    StreamedQwen38,
)

from test_qwen3_8_model import _tiny_config, _tiny_weights


_MODEL = LogicalModelIdentity(
    repo_id="Qwen/Qwen3.8-27B",
    revision=OFFICIAL_REVISION,
)


def _write_bundle(root: Path) -> tuple[Path, Path]:
    bundle = root / "model.causal"
    weights = bundle / "weights"
    weights.mkdir(parents=True)
    (bundle / "causal").mkdir()
    save_file(_tiny_weights(_tiny_config()), weights / "model.safetensors")
    return bundle, weights


class Qwen38CausalTensorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix=".qwen-causal-tensor-test-", dir=Path.cwd()
        )
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_tensor_bindings_are_idempotent_live_and_fail_closed(self) -> None:
        bundle, _weights = _write_bundle(self.root)
        with CausalWeightMount(bundle, _MODEL, budget_mb=20) as mount:
            names = (
                "model.language_model.embed_tokens.weight",
                "model.language_model.layers.0.mlp.gate_proj.weight",
            )
            plans = tuple(
                tensor_range_plan_from_source(mount.source, name) for name in names
            )
            first = mount.bind_tensor_plans(plans[:1])
            self.assertEqual(first.appended_count, 1)
            self.assertEqual(mount.resolve_tensor_plan(names[0]), plans[0])

            replay = mount.bind_tensor_plans(plans[:1])
            self.assertEqual(replay.appended_count, 0)
            with self.assertRaises(CausalWeightConflictError):
                mount.bind_tensor_plans(
                    (replace(plans[0], absolute_offset=plans[0].absolute_offset + 1),)
                )

            second = mount.bind_tensor_plans(plans[1:])
            self.assertEqual(second.appended_count, 1)
            self.assertEqual(mount.resolve_tensor_plan(names[1]), plans[1])
            metrics = mount.tensor_reader.metrics()
            self.assertEqual(metrics["plan_cache_invalidations"], 1)

            expected = mount.source.raw_bytes(
                plans[0].shard,
                plans[0].absolute_offset + 2,
                7,
            )
            with mock.patch.object(
                mount.source,
                "find",
                side_effect=AssertionError("causal tensor read used inventory lookup"),
            ):
                read = mount.read_tensor_range(names[0], relative_offset=2, length=7)
            self.assertEqual(bytes(read.part), bytes(expected))
            self.assertEqual(read.plan, plans[0])
            self.assertEqual(
                len(mount.graph.query_base(semantic_tensor_key(_MODEL, name=names[0]))),
                1,
            )

            assert second.appended_segment_sha256 is not None
            mount.graph.drop_segments((second.appended_segment_sha256,))
            with self.assertRaises(CausalWeightNotFoundError):
                mount.resolve_tensor_plan(names[1])
            self.assertEqual(
                mount.tensor_reader.metrics()["plan_cache_invalidations"], 2
            )

    def test_qwen_pager_executes_and_generates_without_tensor_discovery(self) -> None:
        bundle, weights = _write_bundle(self.root)
        config = _tiny_config()
        baseline_source = Streamer.from_local(
            weights,
            repo_id=_MODEL.repo_id,
            revision=_MODEL.revision,
            use_cache=False,
            budget_mb=20,
        )
        baseline_pager = Qwen38WeightPager(
            baseline_source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=2 * 1024**2,
        )
        baseline = StreamedQwen38(
            config, baseline_pager, max_batch_size=1, max_seq_len=16
        )
        try:
            expected, _ = baseline.forward_prefill([[1, 4, 9]])
            expected_tokens, _ = baseline.generate_greedy(
                [[1, 4]], max_new_tokens=2, head_block_rows=7
            )

            with CausalWeightMount(bundle, _MODEL, budget_mb=20) as mount:
                names = tuple(
                    str(row["name"])
                    for row in mount.source.inventory().get("tensors", ())
                )
                plans = tuple(
                    tensor_range_plan_from_source(mount.source, name) for name in names
                )
                bound = mount.bind_tensor_plans(plans)
                self.assertEqual(bound.appended_count, len(names))
                causal_pager = Qwen38WeightPager(
                    mount.source,
                    device="cpu",
                    compute_dtype="bfloat16",
                    max_resident_bytes=2 * 1024**2,
                    causal_tensor_reader=mount.tensor_reader,
                )
                causal = StreamedQwen38(
                    config, causal_pager, max_batch_size=1, max_seq_len=16
                )
                try:
                    with mock.patch.object(
                        mount.source,
                        "find",
                        side_effect=AssertionError(
                            "Qwen causal pager used tensor inventory discovery"
                        ),
                    ):
                        actual, _ = causal.forward_prefill([[1, 4, 9]])
                        actual_tokens, _ = causal.generate_greedy(
                            [[1, 4]], max_new_tokens=2, head_block_rows=7
                        )
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(actual_tokens, expected_tokens)
                    metrics = causal_pager.metrics()
                    self.assertTrue(metrics["causal_tensor_reader_attached"])
                    self.assertEqual(
                        metrics["causal_tensor_plan_cache_entries"], len(names)
                    )
                    self.assertGreater(metrics["causal_tensor_read_calls"], len(names))
                finally:
                    causal_pager.close()
        finally:
            baseline_pager.close()
            baseline_source.close()

    def test_missing_qwen_tensor_binding_never_falls_back_to_inventory(self) -> None:
        bundle, _weights = _write_bundle(self.root)
        name = "model.language_model.embed_tokens.weight"
        missing = "model.language_model.norm.weight"
        with CausalWeightMount(bundle, _MODEL, budget_mb=20) as mount:
            mount.bind_tensor_plans(
                (tensor_range_plan_from_source(mount.source, name),)
            )
            pager = Qwen38WeightPager(
                mount.source,
                device="cpu",
                compute_dtype="bfloat16",
                max_resident_bytes=2 * 1024**2,
                causal_tensor_reader=mount.tensor_reader,
            )
            try:
                with mock.patch.object(
                    mount.source,
                    "find",
                    side_effect=AssertionError("missing causal binding fell back"),
                ):
                    embedded = pager.embedding((1,), name=name)
                    self.assertEqual(tuple(embedded.shape), (1, _tiny_config().dim))
                    with self.assertRaises(CausalWeightNotFoundError):
                        pager.tensor_torch(missing)
            finally:
                pager.close()


if __name__ == "__main__":
    unittest.main()
