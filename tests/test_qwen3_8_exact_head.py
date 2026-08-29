from __future__ import annotations

import hashlib
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from immer.runtimes.qwen3_8.exact_head import (
    EXACT_HEAD_SCORE_ABI,
    ExactHeadBinding,
    ExactHeadConfig,
    ExactHeadError,
    ExactHeadIndex,
    ExactHeadNotApplicable,
)
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager
from immer.runtimes.qwen3_8.config import OFFICIAL_REVISION

from test_qwen3_8_config_pager import _RawBF16IntoSource


_FINGERPRINT = "a" * 64


def _raw_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


class _BoundHeadSource(_RawBF16IntoSource):
    def __init__(self, head: torch.Tensor) -> None:
        super().__init__({"lm_head.weight": head.float().numpy()})
        self.repo_id = "Qwen/Qwen3.8-27B"
        self.revision = OFFICIAL_REVISION

    def metrics(self) -> dict:
        return {
            **super().metrics(),
            "repo_id": self.repo_id,
            "revision": self.revision,
            "inventory_source_fingerprint": _FINGERPRINT,
        }


class _ExplicitFullScan:
    def topk_logits(self, *_args, **_kwargs):
        return None

    def metrics(self) -> dict[str, int]:
        return {}


def _binding(head: torch.Tensor) -> ExactHeadBinding:
    return ExactHeadBinding(
        repo_id="Qwen/Qwen3.8-27B",
        revision=OFFICIAL_REVISION,
        inventory_fingerprint=_FINGERPRINT,
        tensor_name="lm_head.weight",
        tensor_sha256=_raw_sha(head),
        vocab_size=head.shape[0],
        hidden_size=head.shape[1],
        score_abi=EXACT_HEAD_SCORE_ABI,
    )


def _config() -> ExactHeadConfig:
    return ExactHeadConfig(
        subspace_width=2,
        codebook_size=4,
        page_rows=4,
        fanout=2,
        kmeans_iterations=3,
        assignment_chunk_rows=8,
        max_query_rows=16,
    )


class Qwen38ExactHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        generator = torch.Generator().manual_seed(404)
        self.head = torch.randn(
            (16, 8), generator=generator, dtype=torch.bfloat16
        )
        self.binding = _binding(self.head)
        self.index = ExactHeadIndex.build(
            self.head,
            binding=self.binding,
            config=_config(),
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _pager(
        self, *, index: object | None = None
    ) -> tuple[Qwen38WeightPager, _BoundHeadSource]:
        source = _BoundHeadSource(self.head)
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=2048,
            exact_head_index=index,
        )
        return pager, source

    def test_build_save_load_is_deterministic_and_tamper_closed(self) -> None:
        first = self.root / "first"
        second = self.root / "second"
        receipt = self.index.save(first)
        rebuilt = ExactHeadIndex.build(
            self.head,
            binding=self.binding,
            config=_config(),
        )
        rebuilt.save(second)

        self.assertEqual(
            (first / "manifest.json").read_bytes(),
            (second / "manifest.json").read_bytes(),
        )
        self.assertEqual(
            (first / "index.safetensors").read_bytes(),
            (second / "index.safetensors").read_bytes(),
        )
        restored = ExactHeadIndex.load(first, expected_binding=self.binding)
        self.assertEqual(restored.receipt.tensor_sha256, self.binding.tensor_sha256)
        self.assertEqual(restored.receipt.payload_sha256, receipt.payload_sha256)
        with self.assertRaisesRegex(ExactHeadError, "destination is not empty"):
            self.index.save(first)
        with self.assertRaisesRegex(ExactHeadError, "invalid bounded"):
            ExactHeadIndex.load(first, max_payload_bytes=1)
        with self.assertRaisesRegex(ExactHeadError, "binding differs"):
            ExactHeadIndex.load(
                first,
                expected_binding=replace(
                    self.binding,
                    tensor_sha256="f" * 64,
                ),
            )
        link = self.root / "linked"
        os.symlink(first, link)
        with self.assertRaisesRegex(ExactHeadError, "plain directory"):
            ExactHeadIndex.load(link)

        payload = first / "index.safetensors"
        damaged = bytearray(payload.read_bytes())
        damaged[-1] ^= 1
        payload.write_bytes(damaged)
        with self.assertRaisesRegex(ExactHeadError, "payload hash changed"):
            ExactHeadIndex.load(first)

    def test_weight_only_pager_builder_streams_without_model_forward(self) -> None:
        pager, source = self._pager()

        built = ExactHeadIndex.build_from_pager(
            pager,
            config=_config(),
            sample_rows=8,
        )

        self.assertEqual(built.binding, self.binding)
        self.assertGreater(
            source.metrics()["network_or_source_body_bytes"],
            self.head.numel() * self.head.element_size(),
        )
        pager.attach_exact_head_index(built)
        hidden = torch.ones((1, self.head.shape[1]), dtype=torch.bfloat16)
        actual = pager.topk_logits(hidden, k=3, block_rows=4)
        baseline, _source = self._pager(index=_ExplicitFullScan())
        expected = baseline.topk_logits(hidden, k=3, block_rows=4)
        self.assertTrue(torch.equal(actual[0], expected[0]))
        self.assertTrue(torch.equal(actual[1], expected[1]))

    def test_indexed_topk_matches_full_scan_through_sixteen_queries(self) -> None:
        generator = torch.Generator().manual_seed(91)
        for query_rows in (1, 2, 4, 8, 16):
            hidden = torch.randn(
                (query_rows, self.head.shape[1]),
                generator=generator,
                dtype=torch.bfloat16,
            )
            for k in (1, 3, 7):
                with self.subTest(query_rows=query_rows, k=k):
                    baseline, baseline_source = self._pager(
                        index=_ExplicitFullScan()
                    )
                    indexed, indexed_source = self._pager(index=self.index)
                    expected = baseline.topk_logits(
                        hidden, k=k, block_rows=_config().page_rows
                    )
                    actual = indexed.topk_logits(
                        hidden, k=k, block_rows=_config().page_rows
                    )
                    self.assertTrue(torch.equal(actual[0], expected[0]))
                    self.assertTrue(torch.equal(actual[1], expected[1]))
                    self.assertLessEqual(
                        indexed_source.metrics()["network_or_source_body_bytes"],
                        baseline_source.metrics()["network_or_source_body_bytes"],
                    )

    def test_caps_dominate_every_canonical_page_score(self) -> None:
        hidden = torch.tensor(
            [[1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 2.0, -0.75]],
            dtype=torch.bfloat16,
        )
        caps = self.index.leaf_caps(hidden)
        for page in range(self.index.leaf_count):
            start = page * _config().page_rows
            rows = self.head[start : start + _config().page_rows]
            scores = torch.nn.functional.linear(
                hidden.float(), rows.float()
            ).to(torch.bfloat16)
            self.assertGreaterEqual(caps[0, page], float(scores.max()))

    def test_row_caps_dominate_every_score_through_sixteen_queries(self) -> None:
        generator = torch.Generator().manual_seed(812)
        for query_rows in (1, 4, 8, 16):
            hidden = torch.randn(
                (query_rows, self.head.shape[1]),
                generator=generator,
                dtype=torch.bfloat16,
            )
            row_ids = tuple(range(self.head.shape[0]))
            caps = self.index.row_caps(hidden, row_ids)
            scores = torch.nn.functional.linear(
                hidden.float(), self.head.float()
            ).to(torch.bfloat16)
            with self.subTest(query_rows=query_rows):
                self.assertTrue(
                    bool(
                        (
                            torch.from_numpy(caps)
                            >= scores.to(torch.float64)
                        ).all()
                    )
                )

    def test_selected_row_transport_requires_large_low_run_savings(self) -> None:
        economic = ExactHeadIndex._selected_read_is_economic

        self.assertTrue(economic((4,), 4))
        self.assertTrue(economic((4, 5), 4))
        self.assertFalse(economic((4, 5, 6), 4))
        self.assertTrue(economic((0, 2, 4, 6), 16))
        self.assertFalse(economic((0, 2, 4, 6, 8), 16))
        self.assertFalse(economic(tuple(range(9)), 16))

    def test_row_certificate_keeps_an_omitted_winner_and_reads_it_once(self) -> None:
        head = torch.tensor(
            [
                [40.0, 0.0],
                [0.0, 40.0],
                [40.0, 0.0],
                [0.0, 40.0],
                [50.0, -5.0],
                [30.0, -5.0],
                [30.0, -5.0],
                [30.0, -5.0],
            ],
            dtype=torch.bfloat16,
        )
        config = ExactHeadConfig(
            subspace_width=1,
            codebook_size=8,
            page_rows=4,
            fanout=2,
            kmeans_iterations=2,
            assignment_chunk_rows=8,
            max_query_rows=16,
        )
        index = ExactHeadIndex.build(
            head,
            binding=_binding(head),
            config=config,
        )
        source = _BoundHeadSource(head)
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=256,
            exact_head_index=index,
        )

        score_shapes: list[tuple[int, ...]] = []
        scorer = pager._score_head_rows

        def observe(hidden: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
            score_shapes.append(tuple(rows.shape))
            return scorer(hidden, rows)

        with mock.patch.object(pager, "_score_head_rows", side_effect=observe):
            values, ids = pager.topk_logits(
                torch.ones((1, 2), dtype=torch.bfloat16),
                k=1,
                block_rows=4,
            )

        self.assertEqual((float(values[0, 0]), int(ids[0, 0])), (45.0, 4))
        self.assertEqual(score_shapes, [(4, 2), (4, 2)])
        metrics = index.metrics()
        self.assertEqual(metrics["selected_row_reads"], 1)
        self.assertEqual(metrics["selected_rows_scored"], 1)
        self.assertEqual(metrics["row_bound_rows"], 4)
        self.assertEqual(metrics["row_certificate_logical_bytes_avoided"], 12)
        self.assertEqual(metrics["selected_row_logical_bytes"], 4)
        self.assertEqual(metrics["full_leaf_fallbacks"], 1)
        self.assertEqual(
            source.metrics()["network_or_source_body_bytes"],
            5 * head.shape[1] * head.element_size(),
        )

    def test_row_certificate_retains_lower_id_on_cross_page_tie(self) -> None:
        head = torch.tensor(
            [
                [40.0, 0.0],
                [0.0, 40.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [50.0, -10.0],
                [-10.0, 50.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.bfloat16,
        )
        config = ExactHeadConfig(
            subspace_width=1,
            codebook_size=8,
            page_rows=4,
            fanout=2,
            kmeans_iterations=2,
            assignment_chunk_rows=8,
            max_query_rows=16,
        )
        index = ExactHeadIndex.build(
            head,
            binding=_binding(head),
            config=config,
        )
        pager = Qwen38WeightPager(
            _BoundHeadSource(head),
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=256,
            exact_head_index=index,
        )

        values, ids = pager.topk_logits(
            torch.ones((1, 2), dtype=torch.bfloat16),
            k=1,
            block_rows=4,
        )

        self.assertEqual((float(values[0, 0]), int(ids[0, 0])), (40.0, 0))
        self.assertGreater(index.metrics()["row_certificate_logical_bytes_avoided"], 0)

    def test_row_survivors_are_unioned_once_across_sixteen_queries(self) -> None:
        head = torch.tensor(
            [
                [40.0, 0.0, 40.0, 0.0],
                [0.0, 40.0, 0.0, 40.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
                [45.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 45.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=torch.bfloat16,
        )
        config = ExactHeadConfig(
            subspace_width=1,
            codebook_size=8,
            page_rows=4,
            fanout=2,
            kmeans_iterations=2,
            assignment_chunk_rows=8,
            max_query_rows=16,
        )
        index = ExactHeadIndex.build(
            head,
            binding=_binding(head),
            config=config,
        )
        source = _BoundHeadSource(head)
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=2048,
            exact_head_index=index,
        )
        first = torch.tensor([1.0, 1.0, 0.0, 0.0], dtype=torch.bfloat16)
        second = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.bfloat16)
        hidden = torch.stack((first, second) * 8)

        values, ids = pager.topk_logits(hidden, k=1, block_rows=4)

        self.assertTrue(torch.equal(values, torch.full_like(values, 45.0)))
        self.assertEqual(tuple(int(value) for value in ids[:, 0]), (4, 5) * 8)
        metrics = index.metrics()
        self.assertEqual(metrics["selected_row_reads"], 1)
        self.assertEqual(metrics["selected_rows_scored"], 2)
        self.assertEqual(metrics["row_certificate_logical_bytes_avoided"], 16)
        self.assertEqual(
            source.metrics()["network_or_source_body_bytes"],
            6 * head.shape[1] * head.element_size(),
        )

    def test_zero_row_pruning_uses_unchanged_full_leaf_read(self) -> None:
        head = torch.arange(8, dtype=torch.float32).reshape(4, 2).to(
            torch.bfloat16
        )
        config = ExactHeadConfig(
            subspace_width=1,
            codebook_size=4,
            page_rows=4,
            fanout=2,
            kmeans_iterations=2,
            assignment_chunk_rows=4,
            max_query_rows=16,
        )
        index = ExactHeadIndex.build(
            head,
            binding=_binding(head),
            config=config,
        )
        source = _BoundHeadSource(head)
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=1024,
            exact_head_index=index,
        )

        actual = pager.topk_logits(
            torch.ones((16, 2), dtype=torch.bfloat16),
            k=3,
            block_rows=4,
        )
        baseline = torch.nn.functional.linear(
            torch.ones((16, 2), dtype=torch.float32),
            head.float(),
        ).to(torch.bfloat16)

        self.assertTrue(torch.equal(actual[0], torch.topk(baseline, 3).values))
        self.assertEqual(index.metrics()["selected_row_reads"], 0)
        self.assertEqual(index.metrics()["full_leaf_fallbacks"], 1)
        self.assertEqual(index.metrics()["rows_pruned"], 0)
        self.assertEqual(
            source.metrics()["network_or_source_body_bytes"],
            head.numel() * head.element_size(),
        )

    def test_useless_row_bounds_stop_after_sixty_four_leaf_probes(self) -> None:
        head = torch.arange(520, dtype=torch.float32).reshape(260, 2).to(
            torch.bfloat16
        )
        config = ExactHeadConfig(
            subspace_width=1,
            codebook_size=4,
            page_rows=4,
            fanout=4,
            kmeans_iterations=2,
            assignment_chunk_rows=32,
            max_query_rows=16,
        )
        index = ExactHeadIndex.build(
            head,
            binding=_binding(head),
            config=config,
        )
        source = _BoundHeadSource(head)
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=4096,
            exact_head_index=index,
        )
        node_count = len(index._tensors["node_presence"])

        with mock.patch.object(
            index,
            "_node_caps_from_tables",
            return_value=np.full((1, node_count), np.inf),
        ), mock.patch.object(
            index,
            "_row_caps_from_tables",
            side_effect=lambda _tables, ids: np.full((1, len(ids)), np.inf),
        ):
            actual = pager.topk_logits(
                torch.ones((1, 2), dtype=torch.bfloat16),
                k=1,
                block_rows=4,
            )

        baseline = torch.nn.functional.linear(
            torch.ones((1, 2), dtype=torch.float32),
            head.float(),
        ).to(torch.bfloat16)
        self.assertEqual(float(actual[0][0, 0]), float(baseline.max()))
        metrics = index.metrics()
        self.assertEqual(metrics["row_bound_probe_pages"], 64)
        self.assertEqual(metrics["row_bound_rows"], 256)
        self.assertEqual(metrics["row_bound_disabled_calls"], 1)
        self.assertEqual(metrics["selected_row_reads"], 0)
        self.assertEqual(metrics["full_leaf_fallbacks"], 65)
        self.assertEqual(
            source.metrics()["network_or_source_body_bytes"],
            head.numel() * head.element_size(),
        )

    def test_resealed_negative_radius_and_unused_mask_bits_are_rejected(self) -> None:
        negative = {
            key: value.clone() for key, value in self.index._tensors.items()
        }
        negative["residual_radii"][0] = -1
        with self.assertRaisesRegex(ExactHeadError, "PQ tensors are invalid"):
            ExactHeadIndex(
                config=_config(),
                binding=self.binding,
                tensors=negative,
            )

        mask = {key: value.clone() for key, value in self.index._tensors.items()}
        mask["node_presence"][0, 0, -1] |= 0b1000_0000
        with self.assertRaisesRegex(ExactHeadError, "unused bits"):
            ExactHeadIndex(
                config=_config(),
                binding=self.binding,
                tensors=mask,
            )

        tree = {key: value.clone() for key, value in self.index._tensors.items()}
        tree["node_child_start"][-1] = 0
        tree["node_child_count"][-1] = 2
        with self.assertRaisesRegex(ExactHeadError, "tree|aggregate"):
            ExactHeadIndex(
                config=_config(),
                binding=self.binding,
                tensors=tree,
            )

    def test_subnormal_values_and_foreign_score_policy_fall_back_closed(self) -> None:
        subnormal = torch.tensor(2.0**-133, dtype=torch.bfloat16)
        head = self.head.clone()
        head[0, 0] = subnormal
        torch.set_flush_denormal(True)
        try:
            with self.assertRaisesRegex(ExactHeadError, "subnormal weights"):
                ExactHeadIndex.build(
                    head,
                    binding=_binding(head),
                    config=_config(),
                )
        finally:
            torch.set_flush_denormal(False)

        pager, _source = self._pager(index=self.index)
        hidden = torch.ones((1, self.head.shape[1]), dtype=torch.bfloat16)
        hidden[0, 0] = subnormal
        self.assertIsNone(
            self.index.topk_logits(
                pager,
                hidden,
                k=1,
                name="lm_head.weight",
                block_rows=4,
            )
        )
        self.assertEqual(
            self.index.metrics()["last_fallback_reason"], "query-subnormal"
        )

        pager.HEAD_SCORE_POLICY = "foreign"
        with self.assertRaises(ExactHeadNotApplicable):
            self.index.validate_mount(
                pager,
                name="lm_head.weight",
                block_rows=4,
            )

    def test_separated_and_tied_pages_are_certifiably_pruned(self) -> None:
        separated = torch.zeros((8, 4), dtype=torch.bfloat16)
        separated[:4, 0] = 10
        separated[4:, 0] = -10
        config = ExactHeadConfig(
            subspace_width=2,
            codebook_size=2,
            page_rows=4,
            fanout=2,
            kmeans_iterations=2,
            assignment_chunk_rows=8,
        )
        index = ExactHeadIndex.build(
            separated,
            binding=_binding(separated),
            config=config,
        )
        source = _BoundHeadSource(separated)
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=256,
            exact_head_index=index,
        )
        values, ids = pager.topk_logits(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.bfloat16),
            block_rows=4,
        )
        self.assertEqual((float(values[0, 0]), int(ids[0, 0])), (10.0, 0))
        self.assertEqual(index.metrics()["pages_pruned"], 1)
        self.assertEqual(index.metrics()["rows_pruned"], 4)
        self.assertGreater(index.metrics()["bound_nodes"], index.leaf_count)

        tied = torch.ones((8, 4), dtype=torch.bfloat16)
        tied_index = ExactHeadIndex.build(
            tied,
            binding=_binding(tied),
            config=config,
        )
        tied_pager = Qwen38WeightPager(
            _BoundHeadSource(tied),
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=256,
            exact_head_index=tied_index,
        )
        _values, tied_ids = tied_pager.topk_logits(
            torch.ones((1, 4), dtype=torch.bfloat16),
            k=3,
            block_rows=4,
        )
        self.assertEqual(tuple(int(value) for value in tied_ids[0]), (0, 1, 2))

    def test_overflow_ties_never_prune_the_lower_token_page(self) -> None:
        maximum = torch.finfo(torch.bfloat16).max
        head = torch.zeros((4, 2), dtype=torch.bfloat16)
        head[:2, 0] = maximum / 2
        head[2:, 0] = maximum
        config = ExactHeadConfig(
            subspace_width=2,
            codebook_size=2,
            page_rows=2,
            fanout=2,
            kmeans_iterations=2,
            assignment_chunk_rows=4,
        )
        index = ExactHeadIndex.build(
            head,
            binding=_binding(head),
            config=config,
        )
        pager = Qwen38WeightPager(
            _BoundHeadSource(head),
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=64,
            exact_head_index=index,
        )

        values, ids = pager.topk_logits(
            torch.tensor([[4.0, 0.0]], dtype=torch.bfloat16),
            block_rows=2,
        )

        self.assertTrue(torch.isposinf(values[0, 0]))
        self.assertEqual(int(ids[0, 0]), 0)

    def test_inapplicable_index_falls_back_without_claiming_pruning(self) -> None:
        pager, source = self._pager(index=self.index)
        hidden = torch.ones((1, self.head.shape[1]), dtype=torch.bfloat16)
        progress = []

        expected = pager.topk_logits(
            hidden,
            k=5,
            block_rows=2,
            progress=progress.append,
        )

        self.assertEqual(len(progress), 8)
        self.assertEqual(tuple(expected[0].shape), (1, 5))
        self.assertEqual(
            source.metrics()["network_or_source_body_bytes"],
            self.head.numel() * self.head.element_size(),
        )
        self.assertEqual(self.index.metrics()["calls"], 0)

        fallback_values, fallback_ids = pager.topk_logits(
            hidden,
            k=5,
            block_rows=2,
        )
        self.assertTrue(torch.equal(fallback_values, expected[0]))
        self.assertTrue(torch.equal(fallback_ids, expected[1]))
        self.assertEqual(self.index.metrics()["calls"], 1)
        self.assertEqual(self.index.metrics()["fallback_calls"], 1)

        direct = self.index.topk_logits(
            pager,
            torch.full_like(hidden, float("nan")),
            k=1,
            name="lm_head.weight",
            block_rows=4,
        )
        self.assertIsNone(direct)
        self.assertEqual(self.index.metrics()["last_fallback_reason"], "query-values")


if __name__ == "__main__":
    unittest.main()
