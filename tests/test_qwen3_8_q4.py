from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
import torch

from immer.runtimes.qwen3_8.q4 import (
    Q4_0,
    Q8_0,
    Q4_BANK_SCHEMA,
    Q4_BALANCED_POLICY,
    Q4Bank,
    Q4BankBuilder,
    Q4BankError,
    Q4NativeKernel,
)


_BUNDLE = {
    "manifest_sha256": "a" * 64,
    "layout_fingerprint": "b" * 64,
    "graph_revision": [4, "c" * 64],
}
_SOURCE = {
    "repo_id": "Qwen/Qwen3.8-27B",
    "revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    "inventory_fingerprint": "d" * 64,
    "bundle_manifest_sha256": _BUNDLE["manifest_sha256"],
    "layout_fingerprint": _BUNDLE["layout_fingerprint"],
    "graph_revision": _BUNDLE["graph_revision"],
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _document(body: dict) -> dict:
    return {
        "schema": Q4_BANK_SCHEMA,
        "body": body,
        "sha256": hashlib.sha256(_canonical(body)).hexdigest(),
    }


def _quantized(kernel: Q4NativeKernel, value: torch.Tensor, fmt: str) -> np.ndarray:
    output = np.empty(
        value.shape[0] * kernel.row_bytes(fmt, value.shape[1]),
        dtype=np.uint8,
    )
    kernel.quantize(value.float().contiguous(), output, fmt=fmt, threads=2)
    return output


def _dequantized(
    kernel: Q4NativeKernel,
    packed: np.ndarray,
    *,
    fmt: str,
    rows: int,
    cols: int,
) -> torch.Tensor:
    ids = torch.arange(rows, dtype=torch.int64)
    output = torch.empty((rows, cols), dtype=torch.float32)
    code = kernel.library.immer_q4_dequantize_rows_f32(
        kernel._pointer(packed),
        4 if fmt == Q4_0 else 8,
        rows,
        cols,
        kernel._pointer(ids),
        rows,
        kernel._pointer(output),
        2,
    )
    if code:
        raise AssertionError(code)
    return output


class _Source:
    repo_id = _SOURCE["repo_id"]
    revision = _SOURCE["revision"]

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.tensors = tensors

    def inventory(self):
        return {
            "tensors": [
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": "BF16",
                }
                for name, value in self.tensors.items()
            ]
        }

    def metrics(self):
        return {"inventory_source_fingerprint": _SOURCE["inventory_fingerprint"]}


class _Pager:
    torch = torch

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self.source = _Source(tensors)
        self._lock = threading.RLock()

    def _read_rows(self, name, start, count, *, dtype, device):
        return self.source.tensors[name][start : start + count].to(
            dtype=dtype,
            device=device,
        )


class _RawSource(_Source):
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        super().__init__(tensors)
        self.meta = {}
        payload = bytearray()
        for name, value in tensors.items():
            words = (
                value.float().contiguous().numpy().view(np.uint32) >> np.uint32(16)
            ).astype("<u2")
            begin = len(payload)
            payload.extend(words.tobytes())
            self.meta[name] = {
                "name": name,
                "dtype": "BF16",
                "shape": list(value.shape),
                "shard": "weights.bin",
                "data_start": 0,
                "offset_in_shard": [begin, len(payload)],
            }
        self.payload = bytes(payload)
        self.raw_calls = []

    def find(self, name):
        return dict(self.meta[name])

    def raw_bytes(self, shard, offset, length):
        self.raw_calls.append((shard, offset, length))
        return self.payload[offset : offset + length]


class Q4NativeKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kernel = Q4NativeKernel.load()

    def test_wire_sizes_and_row_roundtrip(self) -> None:
        values = torch.linspace(-4.0, 3.0, 128).reshape(2, 64)
        for fmt, row_bytes in ((Q4_0, 36), (Q8_0, 68)):
            with self.subTest(fmt=fmt):
                packed = _quantized(self.kernel, values, fmt)
                actual = _dequantized(
                    self.kernel,
                    packed,
                    fmt=fmt,
                    rows=2,
                    cols=64,
                )
                self.assertEqual(packed.size, 2 * row_bytes)
                self.assertTrue(torch.isfinite(actual).all())
                tolerance = 0.55 if fmt == Q4_0 else 0.04
                self.assertLessEqual(float((actual - values).abs().max()), tolerance)

    def test_native_linear_matches_its_quantized_operands(self) -> None:
        generator = torch.Generator().manual_seed(17)
        weight = torch.randn((13, 64), generator=generator)
        value = torch.randn((3, 64), generator=generator)
        qinput = _dequantized(
            self.kernel,
            _quantized(self.kernel, value, Q8_0),
            fmt=Q8_0,
            rows=3,
            cols=64,
        )
        for fmt in (Q4_0, Q8_0):
            with self.subTest(fmt=fmt):
                packed = _quantized(self.kernel, weight, fmt)
                qweight = _dequantized(
                    self.kernel,
                    packed,
                    fmt=fmt,
                    rows=13,
                    cols=64,
                )
                output = torch.empty((3, 13), dtype=torch.float32)
                code = self.kernel.library.immer_q4_linear_f32(
                    self.kernel._pointer(value),
                    3,
                    64,
                    self.kernel._pointer(packed),
                    4 if fmt == Q4_0 else 8,
                    13,
                    self.kernel._pointer(output),
                    2,
                )
                self.assertEqual(code, 0)
                torch.testing.assert_close(
                    output,
                    qinput @ qweight.T,
                    rtol=2e-5,
                    atol=2e-5,
                )

    def test_subnormal_block_scales_remain_finite(self) -> None:
        values = torch.stack(
            (
                torch.linspace(-0.005, 0.005, 64),
                torch.linspace(-0.00002, 0.00002, 64),
            )
        )
        for fmt in (Q4_0, Q8_0):
            packed = _quantized(self.kernel, values, fmt)
            actual = _dequantized(
                self.kernel,
                packed,
                fmt=fmt,
                rows=2,
                cols=64,
            )
            self.assertTrue(torch.isfinite(actual).all(), fmt)
            self.assertGreater(float(actual[0].abs().max()), 0.0, fmt)


class Q4BankTests(unittest.TestCase):
    def _build(self, root: Path) -> tuple[dict[str, torch.Tensor], dict]:
        tensors = {
            "model.language_model.layers.0.mlp.gate_proj.weight": torch.linspace(
                -1, 1, 5 * 64
            ).reshape(5, 64),
            "model.language_model.embed_tokens.weight": torch.linspace(
                -2, 2, 7 * 64
            ).reshape(7, 64),
            "lm_head.weight": torch.linspace(-3, 3, 9 * 64).reshape(9, 64),
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": (
                torch.linspace(-1.5, 1.5, 6 * 64).reshape(6, 64)
            ),
            "model.language_model.layers.0.mlp.down_proj.weight": torch.linspace(
                -0.5, 0.5, 4 * 64
            ).reshape(4, 64),
            "model.visual.blocks.0.attn.qkv.weight": torch.ones((4, 64)),
        }
        manifest = Q4BankBuilder(
            root,
            pager=_Pager(tensors),
            bundle_receipt=_BUNDLE,
            row_chunk=2,
            threads=2,
        ).build()
        return tensors, manifest

    def test_builder_mounts_mmap_and_executes_rows_and_linears(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "q4"
            tensors, manifest = self._build(root)
            formats = {
                row["name"]: row["format"] for row in manifest["body"]["tensors"]
            }
            self.assertEqual(len(formats), 5)
            self.assertEqual(formats["lm_head.weight"], Q8_0)
            self.assertEqual(
                formats["model.language_model.embed_tokens.weight"], Q8_0
            )
            self.assertEqual(
                formats[
                    "model.language_model.layers.0.mlp.gate_proj.weight"
                ],
                Q4_0,
            )
            self.assertEqual(
                formats["model.language_model.layers.0.mlp.down_proj.weight"],
                Q4_0,
            )
            bank = Q4Bank.load(
                root,
                bundle_receipt=_BUNDLE,
                repo_id=_SOURCE["repo_id"],
                revision=_SOURCE["revision"],
                inventory_fingerprint=_SOURCE["inventory_fingerprint"],
                threads=2,
            )
            try:
                name = "model.language_model.layers.0.mlp.gate_proj.weight"
                rows = bank.rows(name, (3, 1, 3), dtype=torch.float32)
                self.assertEqual(tuple(rows.shape), (3, 64))
                torch.testing.assert_close(rows[0], rows[2])
                self.assertEqual(bank.metrics()["logical_weight_bytes"], 2 * 36)
                value = torch.ones((2, 64), dtype=torch.bfloat16)
                output = bank.linear(value, name)
                self.assertEqual(tuple(output.shape), (2, 5))
                self.assertEqual(output.dtype, torch.bfloat16)
                self.assertTrue(torch.isfinite(output).all())
                metrics = bank.metrics()
                self.assertEqual(metrics["mapped_tensors"], 1)
                self.assertEqual(metrics["linear_calls"], 1)
                self.assertEqual(metrics["linear_input_rows"], 2)
            finally:
                bank.close()

    def test_builder_is_tensor_resumable_and_manifest_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "q4"
            _tensors, first = self._build(root)
            files = {
                path.name: path.stat().st_ino for path in (root / "weights").iterdir()
            }
            _tensors, second = self._build(root)
            self.assertEqual(first, second)
            self.assertEqual(
                files,
                {
                    path.name: path.stat().st_ino
                    for path in (root / "weights").iterdir()
                },
            )
            with self.assertRaisesRegex(Q4BankError, "another build"):
                Q4BankBuilder(
                    root,
                    pager=_Pager(_tensors),
                    bundle_receipt=_BUNDLE,
                    row_chunk=2,
                    threads=2,
                    format_policy=Q4_BALANCED_POLICY,
                ).build()

    def test_balanced_bank_hardlinks_stable_q4_and_rebuilds_sensitive_q8(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "base"
            balanced = Path(temporary) / "balanced"
            tensors, base_manifest = self._build(base)
            pager = _Pager(tensors)
            builder = Q4BankBuilder(
                balanced,
                pager=pager,
                bundle_receipt=_BUNDLE,
                row_chunk=2,
                threads=2,
                format_policy=Q4_BALANCED_POLICY,
                reuse_root=base,
            )
            plan = builder.plan()
            self.assertGreater(plan["reused_payload_bytes"], 0)
            self.assertGreater(plan["new_payload_bytes"], 0)
            manifest = builder.build()
            base_rows = {row["name"]: row for row in base_manifest["body"]["tensors"]}
            rows = {row["name"]: row for row in manifest["body"]["tensors"]}
            gate = "model.language_model.layers.0.mlp.gate_proj.weight"
            linear = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
            down = "model.language_model.layers.0.mlp.down_proj.weight"
            self.assertEqual(rows[gate]["format"], Q4_0)
            self.assertEqual(rows[linear]["format"], Q8_0)
            self.assertEqual(rows[down]["format"], Q8_0)
            self.assertEqual(
                (base / "weights" / base_rows[gate]["file"]).stat().st_ino,
                (balanced / "weights" / rows[gate]["file"]).stat().st_ino,
            )
            self.assertNotEqual(
                (base / "weights" / base_rows[linear]["file"]).stat().st_ino,
                (balanced / "weights" / rows[linear]["file"]).stat().st_ino,
            )

    def test_mount_rejects_foreign_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "q4"
            _tensors, manifest = self._build(root)
            with self.assertRaisesRegex(Q4BankError, "different causal bundle"):
                Q4Bank.load(
                    root,
                    bundle_receipt={**_BUNDLE, "manifest_sha256": "f" * 64},
                    repo_id=_SOURCE["repo_id"],
                    revision=_SOURCE["revision"],
                    inventory_fingerprint=_SOURCE["inventory_fingerprint"],
                )

    def test_verify_cache_rehashes_corruption_and_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "q4"
            _tensors, manifest = self._build(root)
            with mock.patch(
                "immer.runtimes.qwen3_8.q4._file_sha256",
                side_effect=AssertionError("cache miss"),
            ):
                bank = Q4Bank.load(
                    root,
                    bundle_receipt=_BUNDLE,
                    repo_id=_SOURCE["repo_id"],
                    revision=_SOURCE["revision"],
                    inventory_fingerprint=_SOURCE["inventory_fingerprint"],
                )
                bank.close()
            payload = root / "weights" / manifest["body"]["tensors"][0]["file"]
            with payload.open("r+b", buffering=0) as handle:
                first = handle.read(1)
                handle.seek(0)
                handle.write(bytes((first[0] ^ 0x01,)))
                handle.flush()
            with self.assertRaisesRegex(Q4BankError, "digest differs"):
                Q4Bank.load(
                    root,
                    bundle_receipt=_BUNDLE,
                    repo_id=_SOURCE["repo_id"],
                    revision=_SOURCE["revision"],
                    inventory_fingerprint=_SOURCE["inventory_fingerprint"],
                )
            payload = root / "weights" / manifest["body"]["tensors"][0]["file"]
            payload.write_bytes(payload.read_bytes()[:-1])
            with self.assertRaisesRegex(Q4BankError, "size differs"):
                Q4Bank.load(
                    root,
                    bundle_receipt=_BUNDLE,
                    repo_id=_SOURCE["repo_id"],
                    revision=_SOURCE["revision"],
                    inventory_fingerprint=_SOURCE["inventory_fingerprint"],
                )

    def test_weight_pager_routes_linears_embeddings_and_head_without_bf16_reads(self) -> None:
        from immer.runtimes.qwen3_8.pager import Qwen38PagerError, Qwen38WeightPager

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "q4"
            tensors, _manifest = self._build(root)
            source = _RawSource(tensors)
            bank = Q4Bank.load(
                root,
                bundle_receipt=_BUNDLE,
                repo_id=_SOURCE["repo_id"],
                revision=_SOURCE["revision"],
                inventory_fingerprint=_SOURCE["inventory_fingerprint"],
                threads=2,
            )
            pager = Qwen38WeightPager(
                source,
                device="cpu",
                compute_dtype="bfloat16",
                q4_bank=bank,
            )
            try:
                output = pager.linear(
                    torch.ones((1, 64), dtype=torch.bfloat16),
                    "model.language_model.layers.0.mlp.gate_proj",
                )
                self.assertEqual(tuple(output.shape), (1, 5))
                embedding = pager.embedding((6, 2, 6))
                self.assertEqual(tuple(embedding.shape), (3, 64))
                values, token_ids = pager.topk_logits(
                    torch.ones((1, 64), dtype=torch.bfloat16),
                    k=2,
                )
                self.assertEqual(tuple(values.shape), (1, 2))
                self.assertEqual(tuple(token_ids.shape), (1, 2))
                self.assertEqual(source.raw_calls, [])
                metrics = pager.metrics()
                self.assertTrue(metrics["q4_bank_attached"])
                self.assertEqual(metrics["logical_weight_bytes"], 0)
                self.assertGreater(metrics["q4_logical_weight_bytes"], 0)
                with mock.patch("immer.runtimes.qwen3_8.pager.gc.collect") as collect:
                    pager.release()
                collect.assert_not_called()
                self.assertEqual(pager.metrics()["gc_q4_mmap_skips"], 1)
            finally:
                pager.close()
            low_bank = Q4Bank.load(
                root,
                bundle_receipt=_BUNDLE,
                repo_id=_SOURCE["repo_id"],
                revision=_SOURCE["revision"],
                inventory_fingerprint=_SOURCE["inventory_fingerprint"],
                threads=2,
            )
            low_pager = Qwen38WeightPager(
                source,
                device="cpu",
                compute_dtype="bfloat16",
                max_resident_bytes=64,
                q4_bank=low_bank,
            )
            try:
                with self.assertRaisesRegex(Qwen38PagerError, "transient Q4"):
                    low_pager.linear(
                        torch.ones((1, 64), dtype=torch.bfloat16),
                        "model.language_model.layers.0.mlp.gate_proj",
                    )
                self.assertEqual(low_bank.metrics()["linear_calls"], 0)
            finally:
                low_pager.close()


if __name__ == "__main__":
    unittest.main()
