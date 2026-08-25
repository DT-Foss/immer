from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from immer.knowledge.livecausal import LiveGraph
from immer.knowledge.streamer import Streamer
from immer.runtimes.deepseek_v4 import (
    CausalTensorReader,
    CausalWeightLayoutIdentity,
    DeepSeekWeightPager,
    LogicalModelIdentity,
    StreamedDeepSeekV4,
    bind_causal_tensor_plans,
    tensor_range_plan_from_source,
)
from immer.runtimes.deepseek_v4.pager import DeepSeekPagerError


_MODEL = LogicalModelIdentity("fixture/deepseek-v4", "d" * 40)


def _bf16(values: np.ndarray) -> bytes:
    words = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return (words >> 16).astype("<u2").tobytes()


def _fixture_tensors() -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    embed = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]],
        dtype=np.float32,
    )
    head = np.asarray(
        [[0.5, 0.0], [0.0, 0.5], [1.0, 1.0], [-1.0, 2.0]],
        dtype=np.float32,
    )
    return {
        "dense.weight": (
            "F32",
            (2, 2),
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="<f4").tobytes(),
        ),
        "embed.weight": ("BF16", embed.shape, _bf16(embed)),
        "head.weight": ("BF16", head.shape, _bf16(head)),
        "f16.control": (
            "F16",
            (2,),
            np.asarray([1.5, -2.0], dtype="<f2").tobytes(),
        ),
        "i8.control": ("I8", (3,), np.asarray([-2, 0, 7], dtype=np.int8).tobytes()),
        "tid2eid": (
            "I64",
            (4, 2),
            np.asarray([[0, 1], [2, 3], [4, 5], [6, 7]], dtype="<i8").tobytes(),
        ),
        "fp8.weight": ("F8_E4M3", (2, 128), bytes([0x38]) * (2 * 128)),
        "fp8.scale": ("F8_E8M0", (1, 1), bytes([0x7F])),
        "fp4.weight": ("I8", (2, 16), bytes([0x22]) * (2 * 16)),
        "fp4.scale": ("F8_E8M0", (2, 1), bytes([0x7F]) * 2),
    }


def _write_fixture(root: Path) -> None:
    root.mkdir()
    header: dict[str, object] = {}
    payloads: list[bytes] = []
    weight_map: dict[str, str] = {}
    offset = 0
    for name, (dtype, shape, payload) in _fixture_tensors().items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        weight_map[name] = "model.safetensors"
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads)
    )
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": offset}, "weight_map": weight_map},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _source(root: Path, cache: Path) -> Streamer:
    return Streamer.from_local(root, budget_mb=8, cache_dir=cache)


def _reader(
    source: Streamer,
    graph_root: Path,
    names: tuple[str, ...],
) -> CausalTensorReader:
    layout = CausalWeightLayoutIdentity.from_source(source, model=_MODEL)
    graph = LiveGraph(graph_root)
    bind_causal_tensor_plans(
        graph,
        layout,
        tuple(tensor_range_plan_from_source(source, name) for name in names),
    )
    return CausalTensorReader(graph, layout, source=source)


class DenseCausalReaderTests(unittest.TestCase):
    def test_every_dense_pager_path_uses_only_authenticated_reader_payloads(
        self,
    ) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "weights"
            _write_fixture(root)
            reference_source = _source(root, base / "cache-reference")
            causal_source = _source(root, base / "cache-causal")
            names = tuple(_fixture_tensors())
            reader = _reader(causal_source, base / "graph", names)
            reference = DeepSeekWeightPager(
                reference_source,
                device="cpu",
                compute_dtype="float32",
                simulate_activation_quantization=False,
                expert_prefetch=False,
            )
            causal = DeepSeekWeightPager(
                causal_source,
                device="cpu",
                compute_dtype="float32",
                simulate_activation_quantization=False,
                expert_prefetch=False,
                causal_tensor_reader=reader,
                causal_missing_fallback=False,
            )
            x2 = torch.asarray([[2.0, -1.0]])
            x_fp8 = torch.ones((1, 128), dtype=torch.float32)
            x_fp4 = torch.ones((1, 32), dtype=torch.float32)
            expected = {
                "dense": reference.linear(x2, "dense"),
                "embedding": reference.embedding([3, 1, 2, 3]),
                "candidate": reference.candidate_logits(x2, [3, 1, 2]),
                "head_scalar": reference.topk_logits(
                    x2, k=2, block_rows=1, transport_range_batch_blocks=1
                ),
                "head_batch": reference.topk_logits(
                    x2, k=2, block_rows=1, transport_range_batch_blocks=4
                ),
                "f16": reference.tensor_torch("f16.control"),
                "i8": reference.tensor_torch("i8.control"),
                "i64": reference.tensor_rows(
                    "tid2eid", [3, 1, 3], dtype=torch.long, device="cpu"
                ),
                "fp8": reference.linear(x_fp8, "fp8"),
                "fp4": reference.linear(x_fp4, "fp4"),
            }

            original_many = causal_source.raw_bytes_many
            with ExitStack() as stack:
                for method in ("find", "tensor", "rows", "raw_bytes"):
                    stack.enter_context(
                        mock.patch.object(
                            causal_source,
                            method,
                            side_effect=AssertionError(
                                f"causal pager bypassed reader through {method}"
                            ),
                        )
                    )
                raw_many = stack.enter_context(
                    mock.patch.object(
                        causal_source,
                        "raw_bytes_many",
                        wraps=original_many,
                    )
                )
                plans = causal.preflight_tensor_plans(names)
                self.assertEqual(set(plans), set(names))
                raw_many.assert_not_called()

                actual = {
                    "dense": causal.linear(x2, "dense"),
                    "embedding": causal.embedding([3, 1, 2, 3]),
                    "candidate": causal.candidate_logits(x2, [3, 1, 2]),
                    "head_scalar": causal.topk_logits(
                        x2, k=2, block_rows=1, transport_range_batch_blocks=1
                    ),
                    "head_batch": causal.topk_logits(
                        x2, k=2, block_rows=1, transport_range_batch_blocks=4
                    ),
                    "f16": causal.tensor_torch("f16.control"),
                    "i8": causal.tensor_torch("i8.control"),
                    "i64": causal.tensor_rows(
                        "tid2eid", [3, 1, 3], dtype=torch.long, device="cpu"
                    ),
                    "fp8": causal.linear(x_fp8, "fp8"),
                    "fp4": causal.linear(x_fp4, "fp4"),
                }

            for key in ("dense", "embedding", "candidate", "f16", "i8", "i64", "fp8", "fp4"):
                with self.subTest(path=key):
                    self.assertTrue(torch.equal(actual[key], expected[key]))
            for key in ("head_scalar", "head_batch"):
                with self.subTest(path=key):
                    self.assertTrue(torch.equal(actual[key][0], expected[key][0]))
                    self.assertTrue(torch.equal(actual[key][1], expected[key][1]))
            self.assertGreater(raw_many.call_count, 0)
            metrics = causal.metrics()
            self.assertTrue(metrics["causal_tensor_reader_attached"])
            self.assertFalse(metrics["causal_missing_fallback"])
            self.assertGreater(metrics["causal_tensor_reads"], 0)
            self.assertGreater(metrics["causal_tensor_multi_reads"], 0)
            self.assertGreater(metrics["causal_tensor_requested_bytes"], 0)
            self.assertGreater(metrics["causal_tensor_reader"]["source_requests"], 0)
            reference.release()
            causal.release()
            reference_source.close()
            causal_source.close()

    def test_missing_quantization_pair_fails_before_any_payload(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "weights"
            _write_fixture(root)
            source = _source(root, base / "cache")
            reader = _reader(source, base / "graph", ("fp8.weight",))
            pager = DeepSeekWeightPager(
                source,
                device="cpu",
                compute_dtype="float32",
                simulate_activation_quantization=False,
                expert_prefetch=False,
                causal_tensor_reader=reader,
            )
            with mock.patch.object(
                source,
                "raw_bytes_many",
                side_effect=AssertionError("missing scale reached payload transport"),
            ) as raw_many:
                with self.assertRaisesRegex(
                    DeepSeekPagerError, "causal tensor binding is missing"
                ):
                    pager.linear(torch.ones((1, 128)), "fp8")
            raw_many.assert_not_called()
            pager.release()
            source.close()

    def test_malformed_plan_and_receipt_fail_closed_without_streamer_fallback(
        self,
    ) -> None:
        import torch

        class Source:
            def metrics(self):
                return {"network_or_source_body_bytes": 0}

            def __getattr__(self, name):
                if name in {"find", "tensor", "rows", "raw_bytes", "raw_bytes_many"}:
                    raise AssertionError(f"pager fell back to source.{name}")
                raise AttributeError(name)

        class Reader:
            def __init__(self, source, plan, receipt=None):
                self.source = source
                self.layout = SimpleNamespace(layout_fingerprint="a" * 64)
                self.plan = plan
                self.receipt = receipt

            def resolve_tensor_plan(self, _name):
                return self.plan

            def resolve_tensor_plans(self, names):
                return tuple(self.plan for _name in names)

            def read_tensor_range(self, *_args, **_kwargs):
                raise AssertionError("pager must use authenticated multi-range reads")

            def read_tensor_ranges(self, *_args, **_kwargs):
                return self.receipt

            def metrics(self):
                return {}

        source = Source()
        malformed = SimpleNamespace(
            name="dense.weight",
            dtype="U8",
            shape=(2, 2),
            shard="model.safetensors",
            absolute_offset=8,
            length=4,
        )
        pager = DeepSeekWeightPager(
            source,
            device="cpu",
            compute_dtype="float32",
            expert_prefetch=False,
            causal_tensor_reader=Reader(source, malformed),
        )
        with self.assertRaisesRegex(DeepSeekPagerError, "unsupported dense tensor"):
            pager.linear(torch.ones((1, 2)), "dense")
        with self.assertRaisesRegex(
            DeepSeekPagerError, "requires the separate expert reader"
        ):
            pager.plan_expert_ranges(0, (0,))

        expert_reader = SimpleNamespace(
            resolve_expert_plans=lambda _layer, _expert_ids: ()
        )
        with self.assertRaisesRegex(
            DeepSeekPagerError, "forbids missing expert fallback"
        ):
            pager.attach_causal_weight_reader(
                expert_reader,
                fallback_on_missing=True,
            )

        valid = SimpleNamespace(
            name="dense.weight",
            dtype="F32",
            shape=(2, 2),
            shard="model.safetensors",
            absolute_offset=8,
            length=16,
        )
        mutable = memoryview(bytearray(16))
        receipt = SimpleNamespace(
            plan=valid,
            ranges=((0, 16),),
            parts=(mutable,),
            requested_bytes=16,
            resident_bytes=16,
            source_requests=1,
            source_bytes=16,
        )
        invalid_receipt = DeepSeekWeightPager(
            source,
            device="cpu",
            compute_dtype="float32",
            expert_prefetch=False,
            causal_tensor_reader=Reader(source, valid, receipt),
        )
        with self.assertRaisesRegex(DeepSeekPagerError, "invalid causal tensor"):
            invalid_receipt.linear(torch.ones((1, 2)), "dense")

    def test_model_hash_rows_delegate_to_pager_address_plane(self) -> None:
        import torch

        pager = mock.Mock()
        pager.tensor_rows.return_value = torch.asarray([[6, 7], [2, 3], [6, 7]])
        model = SimpleNamespace(pager=pager, torch=torch)
        token_ids = torch.asarray([3, 1, 3])
        result = StreamedDeepSeekV4._hash_expert_rows(
            model,
            "layers.0.ffn.gate.tid2eid",
            token_ids,
            torch.device("cpu"),
        )
        self.assertTrue(torch.equal(result, pager.tensor_rows.return_value))
        pager.tensor_rows.assert_called_once_with(
            "layers.0.ffn.gate.tid2eid",
            [3, 1, 3],
            dtype=torch.long,
            device=torch.device("cpu"),
        )

    def test_dense_reader_multi_range_receipt_is_exact_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "weights"
            _write_fixture(root)
            source = _source(root, base / "cache")
            reader = _reader(source, base / "graph", ("head.weight",))
            plan = reader.resolve_tensor_plan("head.weight")
            original = source.raw_bytes_many
            with mock.patch.object(
                source, "raw_bytes_many", wraps=original
            ) as raw_many:
                receipt = reader.read_tensor_ranges(
                    "head.weight",
                    ((0, 4), (8, 4)),
                    resident_limit_bytes=8,
                    max_gap_bytes=0,
                )
            self.assertEqual(receipt.plan, plan)
            self.assertEqual(receipt.ranges, ((0, 4), (8, 4)))
            self.assertEqual(tuple(len(part) for part in receipt.parts), (4, 4))
            self.assertEqual(receipt.requested_bytes, 8)
            self.assertEqual(receipt.resident_bytes, 8)
            raw_many.assert_called_once_with(
                plan.shard,
                (
                    (plan.absolute_offset, 4),
                    (plan.absolute_offset + 8, 4),
                ),
                resident_limit_bytes=8,
                max_gap_bytes=0,
            )
            metrics = reader.metrics()
            self.assertEqual(metrics["read_calls"], 1)
            self.assertEqual(metrics["multi_range_read_calls"], 1)
            self.assertEqual(metrics["requested_bytes"], 8)
            source.close()

    def test_dense_reader_rejects_unbounded_and_corrupt_receipts_before_use(
        self,
    ) -> None:
        from immer.knowledge.streamer import RawBytesManyResult
        from immer.runtimes.deepseek_v4 import (
            CausalWeightError,
            CausalWeightIntegrityError,
        )

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "weights"
            _write_fixture(root)
            source = _source(root, base / "cache")
            reader = _reader(source, base / "graph", ("head.weight",))
            with mock.patch.object(source, "raw_bytes_many") as raw_many:
                with self.assertRaisesRegex(CausalWeightError, "zero gaps"):
                    reader.read_tensor_ranges(
                        "head.weight", ((0, 4),), max_gap_bytes=1
                    )
                with self.assertRaisesRegex(
                    CausalWeightError, "smaller than the exact tensor payload"
                ):
                    reader.read_tensor_ranges(
                        "head.weight",
                        ((0, 4), (8, 4)),
                        resident_limit_bytes=7,
                    )
                with self.assertRaisesRegex(CausalWeightError, "exceeds"):
                    reader.read_tensor_ranges("head.weight", ((15, 2),))
                raw_many.assert_not_called()

                raw_many.return_value = RawBytesManyResult(
                    parts=(memoryview(bytearray(4)),),
                    resident_bytes=4,
                    source_requests=1,
                    source_bytes=4,
                )
                with self.assertRaisesRegex(
                    CausalWeightIntegrityError, "mutable or short"
                ):
                    reader.read_tensor_ranges("head.weight", ((0, 4),))

                raw_many.return_value = RawBytesManyResult(
                    parts=(memoryview(bytes(4)),),
                    resident_bytes=3,
                    source_requests=1,
                    source_bytes=4,
                )
                with self.assertRaisesRegex(
                    CausalWeightIntegrityError, "impossible tensor accounting"
                ):
                    reader.read_tensor_ranges("head.weight", ((0, 4),))
            source.close()


if __name__ == "__main__":
    unittest.main()
