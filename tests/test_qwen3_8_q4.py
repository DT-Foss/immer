from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from immer.runtimes.ooe.mlp_pilot_residual import MlpPilotAffineLayer
from immer.runtimes.ooe.mlp_pilot_router import MlpPilotLayerModel

from immer.runtimes.qwen3_8.q4 import (
    Q4_0,
    Q8_0,
    Q4_BANK_SCHEMA,
    Q4_BALANCED_POLICY,
    Q4_RECURRENT_POLICY,
    Q4Bank,
    Q4BankBuilder,
    Q4BankError,
    Q4NativeKernel,
)
from immer.runtimes.qwen3_8.q4_fast_mlp import (
    PackedFastMlpMarkovController,
    Qwen38PackedFastMlpExecutor,
)
from immer.runtimes.qwen3_8.kernels import (
    causal_depthwise_conv,
    DeltaNetState,
    gated_delta_net_core,
    gated_delta_net_postconv_core,
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


class _ExactPackedBank:
    def __init__(self, weights: dict[str, torch.Tensor]) -> None:
        self.weights = weights
        self.entries = {
            name: SimpleNamespace(
                payload_bytes=value.shape[0] * (value.shape[1] // 32) * 18
            )
            for name, value in weights.items()
        }
        self.logical_weight_bytes = 0

    def metrics(self) -> dict[str, int]:
        return {"logical_weight_bytes": self.logical_weight_bytes}

    def linear_rows(
        self,
        values: torch.Tensor,
        name: str,
        row_ids: tuple[int, ...],
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        weight = self.weights[name].index_select(0, torch.tensor(row_ids))
        self.logical_weight_bytes += len(row_ids) * (weight.shape[1] // 32) * 18
        return torch.nn.functional.linear(values, weight).to(output_dtype)

    def linear_rows_pair(
        self,
        values: torch.Tensor,
        names: tuple[str, str],
        row_ids: tuple[int, ...],
        *,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.linear_rows(values, names[0], row_ids, output_dtype=output_dtype),
            self.linear_rows(values, names[1], row_ids, output_dtype=output_dtype),
        )

    def linear_selected_blocks(
        self,
        block_values: torch.Tensor,
        block_ids: torch.Tensor,
        name: str,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        dense = torch.zeros(
            (len(block_values), self.weights[name].shape[1]), dtype=torch.float32
        )
        for row in range(len(block_values)):
            for offset, block in enumerate(block_ids[row].tolist()):
                start = block * 32
                dense[row, start : start + 32] = block_values[row, offset]
        self.logical_weight_bytes += (
            len(block_values) * block_values.shape[1] * self.weights[name].shape[0] * 18
        )
        return torch.nn.functional.linear(dense, self.weights[name]).to(output_dtype)

    def linear_routed(
        self,
        full_block_values: torch.Tensor,
        full_block_ids: torch.Tensor,
        sparse_values: torch.Tensor,
        sparse_coords: torch.Tensor,
        name: str,
        *,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        dense = torch.zeros(
            (len(full_block_values), self.weights[name].shape[1]),
            dtype=torch.float32,
        )
        for row in range(len(full_block_values)):
            for offset, block in enumerate(full_block_ids[row].tolist()):
                start = block * 32
                dense[row, start : start + 32] = full_block_values[row, offset]
            dense[row, sparse_coords[row]] = sparse_values[row]
        touched = full_block_values.shape[1] + len(
            {coordinate // 32 for coordinate in sparse_coords[0].tolist()}
        )
        self.logical_weight_bytes += touched * self.weights[name].shape[0] * 18
        return torch.nn.functional.linear(dense, self.weights[name]).to(output_dtype)

    def sparse_mlp(
        self,
        values: torch.Tensor,
        names: tuple[str, str, str],
        *,
        pilot_ids: np.ndarray,
        coefficients: np.ndarray,
        block_size: int,
        selected_block_count: int,
        affine_scale: np.ndarray,
        affine_bias: np.ndarray,
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = torch.nn.functional.linear(values, self.weights[names[0]])
        up = torch.nn.functional.linear(values, self.weights[names[1]])
        pilots = torch.tensor(pilot_ids.reshape(-1), dtype=torch.long)
        pilot_gate = gate.index_select(1, pilots).reshape(
            len(values), len(pilot_ids), -1
        )
        pilot_up = up.index_select(1, pilots).reshape_as(pilot_gate)
        features = torch.square(torch.nn.functional.silu(pilot_gate) * pilot_up)
        coeff = torch.tensor(coefficients, dtype=torch.float64)
        scores = coeff[None, :, 0] + (
            features.to(torch.float64) * coeff[None, :, 1:]
        ).sum(dim=2)
        selected = torch.tensor(
            np.argsort(-scores.numpy(), axis=1, kind="stable")[
                :, :selected_block_count
            ],
            dtype=torch.int64,
        )
        activation = torch.nn.functional.silu(gate) * up
        masked = torch.zeros_like(activation)
        for row, blocks in enumerate(selected.tolist()):
            for block in blocks:
                start = block * block_size
                masked[row, start : start + block_size] = activation[
                    row, start : start + block_size
                ]
        output = torch.nn.functional.linear(masked, self.weights[names[2]])
        output = output * torch.tensor(affine_scale) + torch.tensor(affine_bias)
        full = sum(self.entries[name].payload_bytes for name in names)
        self.logical_weight_bytes += full // 2
        return output.to(output_dtype), selected


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


class PackedQ4FastMlpTests(unittest.TestCase):
    def test_markov_width_controller_learns_route_continuity_and_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "route-markov.json"
            identity = {"plan": "a" * 64}
            controller = PackedFastMlpMarkovController(
                path,
                identity=identity,
                layers=(0,),
                maximum_width=64,
            )
            route = tuple(range(64))
            self.assertEqual(controller.width(0), 64)
            controller.observe(0, route)
            self.assertEqual(controller.width(0), 64)
            controller.observe(0, route)
            self.assertEqual(controller.width(0), 32)
            controller.flush()

            restored = PackedFastMlpMarkovController(
                path,
                identity=identity,
                layers=(0,),
                maximum_width=64,
            )
            self.assertEqual(restored.width(0), 32)
            self.assertEqual(restored.metrics()["markov_transitions"], 1)
            changed = PackedFastMlpMarkovController(
                path,
                identity={"plan": "b" * 64},
                layers=(0,),
                maximum_width=64,
            )
            self.assertNotEqual(changed.path, restored.path)
            self.assertEqual(changed.metrics()["markov_transitions"], 0)

    def test_pilot_route_executes_only_the_selected_q4_block(self) -> None:
        generator = torch.Generator().manual_seed(71)
        gate = 0.01 * torch.randn((64, 32), generator=generator)
        up = 0.01 * torch.randn((64, 32), generator=generator)
        down = 0.01 * torch.randn((32, 64), generator=generator)
        gate[0] = 1.0
        up[0] = 1.0
        gate[32] = 0.01
        up[32] = 0.01
        base = "model.language_model.layers.0.mlp"
        bank = _ExactPackedBank(
            {
                f"{base}.gate_proj.weight": gate,
                f"{base}.up_proj.weight": up,
                f"{base}.down_proj.weight": down,
            }
        )
        model = MlpPilotLayerModel(
            layer=0,
            intermediate_dimension=64,
            block_size=32,
            selected_block_count=1,
            pilot_offsets=((0,), (0,)),
            coefficients=np.ones((2, 2), dtype=np.float64),
            random_pilot_offsets=((1,), (1,)),
            random_coefficients=np.ones((2, 2), dtype=np.float64),
            marginal_block_scores=np.ones(2, dtype=np.float64),
            training_group_sha256s=("a" * 64,),
        )
        affine = MlpPilotAffineLayer(
            layer=0,
            output_dimension=32,
            scale=np.ones(32, dtype=np.float64),
            bias=np.zeros(32, dtype=np.float64),
            training_group_sha256s=("b" * 64,),
        )
        executor = Qwen38PackedFastMlpExecutor(
            SimpleNamespace(models=(model,)),  # type: ignore[arg-type]
            SimpleNamespace(models=(affine,)),  # type: ignore[arg-type]
            bank,  # type: ignore[arg-type]
            active_layers=(0,),
            output_dtype=torch.float32,
        )
        hidden = torch.ones((1, 32), dtype=torch.float32)

        actual, trace = executor.execute(hidden, layer=0)

        full_gate = torch.nn.functional.linear(hidden, gate)
        full_up = torch.nn.functional.linear(hidden, up)
        activation = torch.nn.functional.silu(full_gate) * full_up
        activation[:, 32:] = 0.0
        expected = torch.nn.functional.linear(activation, down)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(trace.selected_blocks, ((0,),))
        self.assertEqual(trace.selected_neuron_count, 32)
        metrics = executor.metrics()
        self.assertEqual(metrics["packed_sparse_calls"], 1)
        self.assertEqual(metrics["packed_sparse_rows"], 1)
        self.assertGreater(metrics["q4_weight_bytes_saved"], 0)


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
            self.assertEqual(formats["model.language_model.embed_tokens.weight"], Q8_0)
            self.assertEqual(
                formats["model.language_model.layers.0.mlp.gate_proj.weight"],
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
                grouped = bank.linear_group(
                    value,
                    (name, "lm_head.weight"),
                )
                separate = (
                    bank.linear(value, name),
                    bank.linear(value, "lm_head.weight"),
                )
                for actual, expected in zip(grouped, separate, strict=True):
                    torch.testing.assert_close(actual, expected)
                selected_ids = (4, 1, 3)
                selected = bank.linear_rows(
                    value,
                    name,
                    selected_ids,
                    output_dtype=torch.float32,
                )
                full = bank.linear(value, name, output_dtype=torch.float32)
                torch.testing.assert_close(
                    selected,
                    full.index_select(1, torch.tensor(selected_ids)),
                    rtol=0.0,
                    atol=0.0,
                )
                paired = bank.linear_rows_pair(
                    value,
                    (name, "lm_head.weight"),
                    selected_ids,
                    output_dtype=torch.float32,
                )
                paired_full = (
                    bank.linear(value, name, output_dtype=torch.float32),
                    bank.linear(value, "lm_head.weight", output_dtype=torch.float32),
                )
                for actual, complete in zip(paired, paired_full, strict=True):
                    torch.testing.assert_close(
                        actual,
                        complete.index_select(1, torch.tensor(selected_ids)),
                        rtol=0.0,
                        atol=0.0,
                    )

                block_values = torch.stack(
                    (
                        torch.linspace(-1.0, 1.0, 32),
                        torch.linspace(0.5, -0.5, 32),
                    )
                ).reshape(2, 1, 32)
                block_ids = torch.tensor([[0], [1]], dtype=torch.int64)
                sparse = bank.linear_selected_blocks(
                    block_values,
                    block_ids,
                    "model.language_model.layers.0.mlp.down_proj.weight",
                    output_dtype=torch.float32,
                )
                dense = torch.zeros((2, 64), dtype=torch.float32)
                dense[0, :32] = block_values[0, 0]
                dense[1, 32:] = block_values[1, 0]
                expected_sparse = bank.linear(
                    dense,
                    "model.language_model.layers.0.mlp.down_proj.weight",
                    output_dtype=torch.float32,
                )
                torch.testing.assert_close(sparse, expected_sparse, rtol=0.0, atol=0.0)

                routed_sparse_values = torch.tensor(
                    [[0.25, -0.75], [0.5, 0.125]], dtype=torch.float32
                )
                routed_sparse_ids = torch.tensor([[40, 50], [3, 7]], dtype=torch.int64)
                routed = bank.linear_routed(
                    block_values,
                    block_ids,
                    routed_sparse_values,
                    routed_sparse_ids,
                    "model.language_model.layers.0.mlp.down_proj.weight",
                    output_dtype=torch.float32,
                )
                routed_dense = dense.clone()
                routed_dense.scatter_(1, routed_sparse_ids, routed_sparse_values)
                expected_routed = bank.linear(
                    routed_dense,
                    "model.language_model.layers.0.mlp.down_proj.weight",
                    output_dtype=torch.float32,
                )
                torch.testing.assert_close(
                    routed,
                    expected_routed,
                    rtol=2e-3,
                    atol=2e-3,
                )
                metrics = bank.metrics()
                self.assertEqual(metrics["linear_group_calls"], 1)
                self.assertEqual(metrics["linear_row_calls"], 3)
                self.assertEqual(metrics["sparse_block_calls"], 1)
                self.assertEqual(metrics["sparse_coordinate_calls"], 1)
                self.assertEqual(metrics["selected_output_rows"], 9)
                self.assertEqual(metrics["selected_input_blocks"], 4)
                self.assertEqual(metrics["selected_input_coordinates"], 4)
                before_release = bank.linear(
                    value,
                    name,
                    output_dtype=torch.float32,
                )
                bank.release_touched()
                after_release = bank.linear(
                    value,
                    name,
                    output_dtype=torch.float32,
                )
                torch.testing.assert_close(
                    after_release,
                    before_release,
                    rtol=0.0,
                    atol=0.0,
                )
                self.assertGreater(
                    bank.metrics()["mapping_discard_calls"],
                    0,
                )
            finally:
                bank.close()

    def test_native_topk_matches_full_q8_scan_across_discard_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "q8-head"
            generator = torch.Generator().manual_seed(8181)
            head = torch.randn((9001, 64), generator=generator)
            Q4BankBuilder(
                root,
                pager=_Pager({"lm_head.weight": head}),
                bundle_receipt=_BUNDLE,
                row_chunk=1024,
                threads=2,
            ).build()
            bank = Q4Bank.load(
                root,
                bundle_receipt=_BUNDLE,
                repo_id=_SOURCE["repo_id"],
                revision=_SOURCE["revision"],
                inventory_fingerprint=_SOURCE["inventory_fingerprint"],
                threads=2,
            )
            try:
                hidden = torch.randn((2, 64), generator=generator).to(torch.bfloat16)
                complete = bank.linear(
                    hidden,
                    "lm_head.weight",
                    output_dtype=torch.bfloat16,
                )
                order = torch.argsort(
                    complete,
                    dim=-1,
                    descending=True,
                    stable=True,
                )[:, :5]
                expected = torch.gather(complete, -1, order)
                bank.release_touched()

                values, ids = bank.topk(
                    hidden,
                    "lm_head.weight",
                    k=5,
                    block_rows=257,
                    output_dtype=torch.bfloat16,
                )

                torch.testing.assert_close(values, expected, rtol=0.0, atol=0.0)
                torch.testing.assert_close(ids, order, rtol=0.0, atol=0.0)
                metrics = bank.metrics()
                self.assertEqual(metrics["native_topk_calls"], 1)
                self.assertEqual(metrics["native_topk_rows"], 2 * len(head))
                self.assertGreaterEqual(metrics["mapping_discard_calls"], 2)
                self.assertGreater(metrics["native_topk_discard_bytes"], 0)
            finally:
                bank.close()

    def test_builder_includes_embedded_mtp_matrices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "q4-mtp"
            tensors = {
                "mtp.fc.weight": torch.linspace(-1, 1, 64 * 128).reshape(64, 128),
                "mtp.pre_fc_norm_hidden.weight": torch.zeros(64),
            }
            manifest = Q4BankBuilder(
                root,
                pager=_Pager(tensors),
                bundle_receipt=_BUNDLE,
                row_chunk=16,
                threads=2,
            ).build()

            rows = {row["name"]: row for row in manifest["body"]["tensors"]}
            self.assertEqual(set(rows), {"mtp.fc.weight"})
            self.assertEqual(rows["mtp.fc.weight"]["format"], Q4_0)

    def test_fused_sparse_mlp_matches_the_selected_dense_q4_route(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "q4-fused-mlp"
            generator = torch.Generator().manual_seed(991)
            base = "model.language_model.layers.0.mlp"
            tensors = {
                f"{base}.gate_proj.weight": torch.randn((64, 64), generator=generator),
                f"{base}.up_proj.weight": torch.randn((64, 64), generator=generator),
                f"{base}.down_proj.weight": torch.randn((32, 64), generator=generator),
            }
            Q4BankBuilder(
                root,
                pager=_Pager(tensors),
                bundle_receipt=_BUNDLE,
                row_chunk=16,
                threads=2,
            ).build()
            bank = Q4Bank.load(
                root,
                bundle_receipt=_BUNDLE,
                repo_id=_SOURCE["repo_id"],
                revision=_SOURCE["revision"],
                inventory_fingerprint=_SOURCE["inventory_fingerprint"],
                threads=2,
            )
            try:
                values = torch.randn((2, 64), generator=generator)
                pilots = np.asarray(((0, 16), (32, 48)), dtype=np.int64)
                coefficients = np.asarray(
                    ((0.1, 0.7, 0.3), (0.2, 0.4, 0.6)), dtype=np.float64
                )
                scale = np.linspace(0.8, 1.2, 32, dtype=np.float32)
                bias = np.linspace(-0.1, 0.1, 32, dtype=np.float32)

                actual, selected = bank.sparse_mlp(
                    values,
                    (
                        f"{base}.gate_proj.weight",
                        f"{base}.up_proj.weight",
                        f"{base}.down_proj.weight",
                    ),
                    pilot_ids=pilots,
                    coefficients=coefficients,
                    block_size=32,
                    selected_block_count=1,
                    affine_scale=scale,
                    affine_bias=bias,
                    output_dtype=torch.float32,
                )

                gate = bank.linear(
                    values, f"{base}.gate_proj.weight", output_dtype=torch.float32
                )
                up = bank.linear(
                    values, f"{base}.up_proj.weight", output_dtype=torch.float32
                )
                pilot_indices = torch.tensor(pilots.reshape(-1), dtype=torch.long)
                pilot_gate = gate.index_select(1, pilot_indices).reshape(2, 2, 2)
                pilot_up = up.index_select(1, pilot_indices).reshape(2, 2, 2)
                features = torch.square(
                    torch.nn.functional.silu(pilot_gate) * pilot_up
                ).to(torch.float64)
                coeff = torch.tensor(coefficients, dtype=torch.float64)
                scores = torch.clamp_min(
                    coeff[None, :, 0] + (features * coeff[None, :, 1:]).sum(dim=2),
                    0.0,
                )
                expected_selected = torch.tensor(
                    np.argsort(-scores.numpy(), axis=1, kind="stable")[:, :1],
                    dtype=torch.int64,
                )
                torch.testing.assert_close(selected, expected_selected)
                activated = torch.nn.functional.silu(gate) * up
                masked = torch.zeros_like(activated)
                for row, (block,) in enumerate(expected_selected.tolist()):
                    start = block * 32
                    masked[row, start : start + 32] = activated[row, start : start + 32]
                expected = bank.linear(
                    masked,
                    f"{base}.down_proj.weight",
                    output_dtype=torch.float32,
                )
                expected = expected * torch.tensor(scale) + torch.tensor(bias)
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
                metrics = bank.metrics()
                self.assertEqual(metrics["fused_mlp_calls"], 1)
                self.assertEqual(metrics["fused_mlp_rows"], 2)
            finally:
                bank.close()

    def test_fused_deltanet_step_matches_packed_projection_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "q4-fused-deltanet"
            generator = torch.Generator().manual_seed(992)
            base = "model.language_model.layers.0.linear_attn"
            hidden_dim = 64
            key_heads, value_heads = 1, 2
            key_dim = value_dim = 16
            qkv_rows = 2 * key_heads * key_dim + value_heads * value_dim
            z_rows = value_heads * value_dim
            names = (
                f"{base}.in_proj_qkv.weight",
                f"{base}.in_proj_z.weight",
                f"{base}.in_proj_b.weight",
                f"{base}.in_proj_a.weight",
            )
            tensors = {
                names[0]: torch.randn((qkv_rows, hidden_dim), generator=generator),
                names[1]: torch.randn((z_rows, hidden_dim), generator=generator),
                names[2]: torch.randn((value_heads, hidden_dim), generator=generator),
                names[3]: torch.randn((value_heads, hidden_dim), generator=generator),
            }
            Q4BankBuilder(
                root,
                pager=_Pager(tensors),
                bundle_receipt=_BUNDLE,
                row_chunk=16,
                threads=2,
            ).build()
            bank = Q4Bank.load(
                root,
                bundle_receipt=_BUNDLE,
                repo_id=_SOURCE["repo_id"],
                revision=_SOURCE["revision"],
                inventory_fingerprint=_SOURCE["inventory_fingerprint"],
                threads=2,
            )
            try:
                hidden = torch.randn(
                    (1, 1, hidden_dim), generator=generator, dtype=torch.bfloat16
                )
                conv_weight = (
                    torch.randn(
                        (qkv_rows, 1, 4), generator=generator, dtype=torch.bfloat16
                    )
                    * 0.05
                )
                a_log = torch.randn(value_heads, generator=generator) * 0.1
                dt_bias = torch.randn(value_heads, generator=generator) * 0.1
                norm_weight = (
                    torch.randn(value_dim, generator=generator, dtype=torch.bfloat16)
                    * 0.1
                    + 1.0
                )
                state = DeltaNetState(
                    conv=torch.randn(
                        (1, qkv_rows, 4), generator=generator, dtype=torch.bfloat16
                    ),
                    recurrent=torch.randn(
                        (1, value_heads, key_dim, value_dim), generator=generator
                    ),
                )
                projections = bank.linear_group(
                    hidden,
                    names,
                    output_dtype=torch.bfloat16,
                )
                expected, expected_state = gated_delta_net_core(
                    *projections,
                    conv1d_weight=conv_weight,
                    A_log=a_log,
                    dt_bias=dt_bias,
                    norm_weight=norm_weight,
                    num_key_heads=key_heads,
                    num_value_heads=value_heads,
                    key_head_dim=key_dim,
                    value_head_dim=value_dim,
                    state=state,
                    native_recurrence=True,
                )
                convolved, _ = causal_depthwise_conv(
                    projections[0],
                    conv_weight,
                    conv_state=state.conv,
                )

                actual_qkv, actual_z, actual_b, actual_a, actual_conv = (
                    bank.deltanet_step(
                        hidden,
                        names,
                        conv_weight=conv_weight,
                        A_log=a_log,
                        dt_bias=dt_bias,
                        norm_weight=norm_weight,
                        conv_state=state.conv,
                        recurrent_state=state.recurrent,
                        key_heads=key_heads,
                        value_heads=value_heads,
                        key_dim=key_dim,
                        value_dim=value_dim,
                        rms_eps=1e-6,
                        output_dtype=torch.bfloat16,
                    )
                )
                actual, actual_recurrent = gated_delta_net_postconv_core(
                    actual_qkv,
                    actual_z,
                    actual_b,
                    actual_a,
                    A_log=a_log,
                    dt_bias=dt_bias,
                    norm_weight=norm_weight,
                    num_key_heads=key_heads,
                    num_value_heads=value_heads,
                    key_head_dim=key_dim,
                    value_head_dim=value_dim,
                    recurrent_state=state.recurrent,
                    rms_norm_eps=1e-6,
                    native_recurrence=True,
                )

                torch.testing.assert_close(actual_qkv, convolved)
                torch.testing.assert_close(actual_z, projections[1])
                torch.testing.assert_close(actual_b, projections[2])
                torch.testing.assert_close(actual_a, projections[3])
                torch.testing.assert_close(
                    actual_recurrent,
                    expected_state.recurrent,
                    rtol=2e-5,
                    atol=2e-5,
                )
                torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
                torch.testing.assert_close(actual_conv, expected_state.conv)
                metrics = bank.metrics()
                self.assertEqual(metrics["fused_deltanet_calls"], 1)
                self.assertEqual(metrics["fused_deltanet_rows"], 1)
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

            recurrent = Path(temporary) / "recurrent"
            recurrent_manifest = Q4BankBuilder(
                recurrent,
                pager=_Pager(tensors),
                bundle_receipt=_BUNDLE,
                row_chunk=2,
                threads=2,
                format_policy=Q4_RECURRENT_POLICY,
                reuse_root=balanced,
            ).build()
            recurrent_rows = {
                row["name"]: row for row in recurrent_manifest["body"]["tensors"]
            }
            self.assertEqual(recurrent_rows[linear]["format"], Q8_0)
            self.assertEqual(recurrent_rows[down]["format"], Q4_0)
            self.assertEqual(
                (balanced / "weights" / rows[linear]["file"]).stat().st_ino,
                (recurrent / "weights" / recurrent_rows[linear]["file"]).stat().st_ino,
            )
            self.assertNotEqual(
                (balanced / "weights" / rows[down]["file"]).stat().st_ino,
                (recurrent / "weights" / recurrent_rows[down]["file"]).stat().st_ino,
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

    def test_weight_pager_routes_linears_embeddings_and_head_without_bf16_reads(
        self,
    ) -> None:
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
                grouped = pager.linear_group(
                    torch.ones((1, 64), dtype=torch.bfloat16),
                    (
                        "model.language_model.layers.0.mlp.gate_proj",
                        "model.language_model.layers.0.mlp.down_proj",
                    ),
                )
                self.assertEqual(tuple(grouped[0].shape), (1, 5))
                self.assertEqual(tuple(grouped[1].shape), (1, 4))
                embedding = pager.embedding((6, 2, 6))
                self.assertEqual(tuple(embedding.shape), (3, 64))
                head_hidden = torch.ones((1, 64), dtype=torch.bfloat16)
                complete_logits = bank.linear(
                    head_hidden,
                    "lm_head.weight",
                    output_dtype=torch.bfloat16,
                )
                complete_ids = torch.arange(9, dtype=torch.long).reshape(1, -1)
                expected_values, expected_ids = pager._stable_topk(
                    complete_logits,
                    complete_ids,
                    2,
                )
                bank.release_touched()
                values, token_ids = pager.topk_logits(
                    head_hidden,
                    k=2,
                    block_rows=3,
                )
                self.assertEqual(tuple(values.shape), (1, 2))
                self.assertEqual(tuple(token_ids.shape), (1, 2))
                torch.testing.assert_close(values, expected_values, rtol=0.0, atol=0.0)
                torch.testing.assert_close(token_ids, expected_ids, rtol=0.0, atol=0.0)
                self.assertEqual(source.raw_calls, [])
                metrics = pager.metrics()
                self.assertTrue(metrics["q4_bank_attached"])
                self.assertEqual(metrics["logical_weight_bytes"], 0)
                self.assertGreater(metrics["q4_logical_weight_bytes"], 0)
                self.assertEqual(metrics["grouped_linear_calls"], 1)
                self.assertEqual(metrics["q4_native_topk_calls"], 1)
                self.assertEqual(metrics["q4_native_topk_rows"], 9)
                self.assertGreater(metrics["q4_mapping_discard_calls"], 0)
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
