from __future__ import annotations

import threading
import unittest
from unittest import mock

import numpy as np


class _Source:
    def __init__(self) -> None:
        self.data = {
            "dense.weight": np.asarray([[1, 2], [3, 4]], dtype=np.float32),
            "group.weight": np.asarray(
                [[1, 0], [0, 1], [2, 0], [0, 3]], dtype=np.float32
            ),
            "embed.weight": np.arange(20, dtype=np.float32).reshape(10, 2),
            "head.weight": np.asarray(
                [[1, 0], [0, 1], [1, 1], [-1, 0]], dtype=np.float32
            ),
        }
        self.dtypes = {name: "F32" for name in self.data}
        self.row_calls: list[tuple[str, int, int]] = []

    def find(self, name: str) -> dict:
        value = self.data[name]
        itemsize = {
            "F8_E4M3": 1,
            "F8_E8M0": 1,
            "I8": 1,
        }.get(self.dtypes[name], value.dtype.itemsize)
        return {
            "name": name,
            "dtype": self.dtypes[name],
            "shape": list(value.shape),
            "offset_in_shard": [0, value.size * itemsize],
        }

    def tensor(self, name: str) -> np.ndarray:
        return self.data[name].copy()

    def rows(self, name: str, start_row: int = 0, n_rows: int = 8, **_) -> np.ndarray:
        self.row_calls.append((name, start_row, n_rows))
        return self.data[name][start_row : start_row + n_rows].copy()

    def metrics(self) -> dict:
        return {"network_or_source_body_bytes": 0}


class _EncodedExpertSource:
    """Synthetic safetensors source with controllable expert adjacency."""

    def __init__(self, base: str, encoding: str, *, adjacent: bool) -> None:
        self.base = base
        self.tensor_calls: list[str] = []
        self.raw_calls: list[tuple[str, int, int]] = []
        self.data_start = 8
        self.shard_name = "expert.safetensors"
        self.data: dict[str, np.ndarray] = {}
        self.dtypes: dict[str, str] = {}
        encoded: dict[str, bytes] = {}
        for role in ("w1", "w2", "w3"):
            weight_name = f"{base}.{role}.weight"
            scale_name = f"{base}.{role}.scale"
            if encoding == "fp4":
                self.data[weight_name] = np.full((128, 64), 0x22, dtype=np.int8)
                self.data[scale_name] = np.ones((128, 4), dtype=np.float32)
                self.dtypes[weight_name] = "I8"
                encoded[weight_name] = self.data[weight_name].tobytes()
            elif encoding == "fp8":
                self.data[weight_name] = np.ones((128, 128), dtype=np.float32)
                self.data[scale_name] = np.ones((1, 1), dtype=np.float32)
                self.dtypes[weight_name] = "F8_E4M3"
                encoded[weight_name] = bytes([0x38]) * (128 * 128)
            else:  # pragma: no cover - test fixture invariant
                raise ValueError(encoding)
            self.dtypes[scale_name] = "F8_E8M0"
            encoded[scale_name] = bytes([0x7F]) * self.data[scale_name].size

        self.meta: dict[str, dict] = {}
        payload = bytearray()
        for group in ("scale", "weight"):
            if payload:
                payload.extend(b"gap-between-groups")
            for index, role in enumerate(("w1", "w2", "w3")):
                if index and not adjacent:
                    payload.extend(b"!")
                name = f"{base}.{role}.{group}"
                begin = len(payload)
                payload.extend(encoded[name])
                end = len(payload)
                self.meta[name] = {
                    "name": name,
                    "dtype": self.dtypes[name],
                    "shape": list(self.data[name].shape),
                    "shard": self.shard_name,
                    "data_start": self.data_start,
                    "offset_in_shard": [begin, end],
                }
        self.payload = bytes(payload)

    def find(self, name: str) -> dict:
        return dict(self.meta[name])

    def tensor(self, name: str) -> np.ndarray:
        self.tensor_calls.append(name)
        return self.data[name].copy()

    def raw_bytes(self, shard: str, offset: int, length: int) -> bytes:
        self.raw_calls.append((shard, offset, length))
        start = offset - self.data_start
        return self.payload[start : start + length]

    def metrics(self) -> dict:
        return {
            "network_or_source_body_bytes": sum(
                length for _, _, length in self.raw_calls
            )
        }


class _MultiEncodedExpertSource:
    """Combine independently adjacent experts behind one tensor source."""

    def __init__(
        self, bases: list[str], *, adjacent: bool = True, encoding: str = "fp4"
    ) -> None:
        self.children: dict[str, _EncodedExpertSource] = {}
        self.meta: dict[str, dict] = {}
        self.data: dict[str, np.ndarray] = {}
        self.raw_calls: list[tuple[str, int, int]] = []
        self.tensor_calls: list[str] = []
        for index, base in enumerate(bases):
            child = _EncodedExpertSource(base, encoding, adjacent=adjacent)
            shard = f"expert-{index}.safetensors"
            child.shard_name = shard
            for name, record in child.meta.items():
                updated = dict(record)
                updated["shard"] = shard
                self.meta[name] = updated
            self.children[shard] = child
            self.data.update(child.data)

    def find(self, name: str) -> dict:
        return dict(self.meta[name])

    def tensor(self, name: str) -> np.ndarray:
        self.tensor_calls.append(name)
        return self.data[name].copy()

    def raw_bytes(self, shard: str, offset: int, length: int) -> bytes:
        self.raw_calls.append((shard, offset, length))
        child = self.children[shard]
        start = offset - child.data_start
        return child.payload[start : start + length]

    def metrics(self) -> dict:
        return {
            "network_or_source_body_bytes": sum(
                length for _, _, length in self.raw_calls
            )
        }


class _BlockingExpertSource(_MultiEncodedExpertSource):
    def __init__(self, bases: list[str]) -> None:
        super().__init__(bases)
        self.blocked_shard: str | None = None
        self.started = threading.Event()
        self.release_read = threading.Event()
        self.completed = threading.Event()
        self.fail = False

    def raw_bytes(self, shard: str, offset: int, length: int) -> bytes:
        if shard == self.blocked_shard:
            self.started.set()
            if not self.release_read.wait(timeout=5):
                raise RuntimeError("test did not release blocked expert read")
            if self.fail:
                raise RuntimeError("simulated expert prefetch failure")
        result = super().raw_bytes(shard, offset, length)
        if shard == self.blocked_shard:
            blocked_calls = sum(
                observed_shard == shard for observed_shard, _, _ in self.raw_calls
            )
            if blocked_calls == 2:
                self.completed.set()
        return result


class DeepSeekV4PagerTests(unittest.TestCase):
    @staticmethod
    def _manual_expert(pager, x, base: str, route_weight=None):
        import torch

        gate = pager.linear(x, f"{base}.w1").float()
        up = pager.linear(x, f"{base}.w3").float()
        hidden = torch.nn.functional.silu(torch.clamp(gate, max=10.0)) * torch.clamp(
            up, min=-10.0, max=10.0
        )
        if route_weight is not None:
            hidden = hidden * route_weight
        return pager.linear(hidden.to(x.dtype), f"{base}.w2")

    def test_fp4_routed_expert_coalesces_to_two_ranges_with_exact_parity(
        self,
    ) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager
        from immer.runtimes.deepseek_v4.quantization import (
            quantize_fp8_e4m3_parts,
        )

        base = "layers.0.ffn.experts.7"
        x = torch.linspace(-0.01, 0.01, 128, dtype=torch.bfloat16)[None, :]
        route_weight = torch.asarray([[0.25]], dtype=torch.float32)

        reference = DeepSeekWeightPager(
            _EncodedExpertSource(base, "fp4", adjacent=False),
            device="cpu",
            compute_dtype="bfloat16",
        )
        expected = self._manual_expert(reference, x, base, route_weight)
        fallback_source = _EncodedExpertSource(base, "fp4", adjacent=False)
        fallback = DeepSeekWeightPager(
            fallback_source, device="cpu", compute_dtype="bfloat16"
        )
        fallback_actual = fallback.expert(
            x, base, route_weight=route_weight, swiglu_limit=10.0
        )

        source = _EncodedExpertSource(base, "fp4", adjacent=True)
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        with mock.patch(
            "immer.runtimes.deepseek_v4.pager.quantize_fp8_e4m3_parts",
            wraps=quantize_fp8_e4m3_parts,
        ) as activation_quantize:
            actual = pager.expert(x, base, route_weight=route_weight, swiglu_limit=10.0)

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(fallback_actual, expected, atol=0, rtol=0)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertEqual(activation_quantize.call_count, 2)
        self.assertEqual(len(source.raw_calls), 2)
        self.assertEqual(source.tensor_calls, [])
        self.assertEqual(len(fallback_source.tensor_calls), 6)
        self.assertEqual(fallback_source.raw_calls, [])
        metrics = pager.metrics()
        self.assertEqual(metrics["linear_calls"], 3)
        self.assertEqual(metrics["expert_calls"], 1)
        self.assertEqual(metrics["coalesced_expert_calls"], 1)
        self.assertEqual(metrics["expert_source_ranges"], 2)
        self.assertEqual(metrics["peak_single_weight_bytes"], 128 * 128 * 4)
        self.assertEqual(metrics["materialized_float_bytes"], 3 * 128 * 128 * 4)
        self.assertEqual(metrics["materialized_scale_bytes"], 3 * 128 * 4 * 4)
        self.assertEqual(
            metrics["logical_weight_bytes"],
            sum(
                end - begin
                for meta in source.meta.values()
                for begin, end in [meta["offset_in_shard"]]
            ),
        )

    def test_fp8_shared_expert_coalesces_to_two_ranges_with_exact_parity(
        self,
    ) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        base = "layers.0.ffn.shared_experts"
        x = torch.linspace(-0.01, 0.01, 128, dtype=torch.float32)[None, :]

        reference = DeepSeekWeightPager(
            _EncodedExpertSource(base, "fp8", adjacent=False),
            device="cpu",
            compute_dtype="float32",
        )
        expected = self._manual_expert(reference, x, base)
        fallback_source = _EncodedExpertSource(base, "fp8", adjacent=False)
        fallback = DeepSeekWeightPager(
            fallback_source, device="cpu", compute_dtype="float32"
        )
        fallback_actual = fallback.expert(x, base, swiglu_limit=10.0)

        source = _EncodedExpertSource(base, "fp8", adjacent=True)
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        actual = pager.expert(x, base, swiglu_limit=10.0)

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(fallback_actual, expected, atol=0, rtol=0)
        self.assertEqual(len(source.raw_calls), 2)
        self.assertEqual(source.tensor_calls, [])
        self.assertEqual(len(fallback_source.tensor_calls), 6)
        self.assertEqual(fallback_source.raw_calls, [])
        self.assertEqual(pager.metrics()["linear_calls"], 3)

    def test_exact_prefetch_is_async_bounded_and_avoids_second_read(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager
        from immer.runtimes.deepseek_v4.pager import DeepSeekPagerError

        base = "layers.0.ffn.experts.7"
        other = "layers.0.ffn.experts.8"
        source = _BlockingExpertSource([base, other])
        source.blocked_shard = "expert-0.safetensors"
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")

        ticket = pager.prefetch_expert(base)
        self.assertIsNotNone(ticket)
        assert ticket is not None
        self.assertTrue(source.started.wait(timeout=2))
        self.assertFalse(ticket.future.done())
        with self.assertRaisesRegex(DeepSeekPagerError, "outstanding"):
            pager.prefetch_expert(other)
        source.release_read.set()

        payload = pager.consume_expert_prefetch(ticket, base)
        x = torch.zeros((1, 128), dtype=torch.bfloat16)
        actual = pager.expert(x, base, prefetched_payload=payload)
        self.assertTrue(torch.isfinite(actual).all())
        self.assertEqual(len(source.raw_calls), 2)
        with self.assertRaisesRegex(DeepSeekPagerError, "stale or consumed"):
            pager.consume_expert_prefetch(ticket, base)
        metrics = pager.metrics()
        self.assertEqual(metrics["expert_prefetch_submitted"], 1)
        self.assertEqual(metrics["expert_prefetch_consumed"], 1)
        self.assertEqual(metrics["expert_prefetch_max_outstanding"], 1)
        self.assertLessEqual(
            metrics["expert_prefetch_peak_bytes"],
            2 * pager.EXPERT_PREFETCH_PAYLOAD_LIMIT_BYTES,
        )
        pager.release()

    def test_next_expert_read_starts_before_current_expert_compute(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        first = "layers.0.ffn.experts.7"
        second = "layers.0.ffn.experts.8"
        source = _BlockingExpertSource([first, second])
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        first_ticket = pager.prefetch_expert(first)
        assert first_ticket is not None
        first_payload = pager.consume_expert_prefetch(first_ticket, first)

        source.blocked_shard = "expert-1.safetensors"
        source.started.clear()
        source.release_read.clear()
        second_ticket = pager.prefetch_expert(second)
        assert second_ticket is not None
        linear = pager._linear_materialized
        observed = threading.Event()

        def assert_prefetch_started(*args, **kwargs):
            self.assertTrue(source.started.wait(timeout=2))
            observed.set()
            source.release_read.set()
            return linear(*args, **kwargs)

        x = torch.zeros((1, 128), dtype=torch.bfloat16)
        with mock.patch.object(
            pager, "_linear_materialized", side_effect=assert_prefetch_started
        ):
            first_output = pager.expert(
                x, first, prefetched_payload=first_payload
            )
        self.assertTrue(observed.is_set())
        second_payload = pager.consume_expert_prefetch(second_ticket, second)
        second_output = pager.expert(x, second, prefetched_payload=second_payload)
        self.assertTrue(torch.equal(first_output, second_output))
        self.assertEqual(len(source.raw_calls), 4)
        self.assertEqual(pager.metrics()["expert_prefetch_peak_bytes"], 2 * 26112)
        pager.release()

    def test_prefetch_failure_clears_slot_and_fallback_stays_synchronous(self) -> None:
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        base = "layers.0.ffn.experts.7"
        source = _BlockingExpertSource([base])
        source.blocked_shard = "expert-0.safetensors"
        source.fail = True
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        ticket = pager.prefetch_expert(base)
        assert ticket is not None
        source.release_read.set()
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            pager.consume_expert_prefetch(ticket, base)

        source.fail = False
        source.blocked_shard = None
        replacement = pager.prefetch_expert(base)
        assert replacement is not None
        payload = pager.consume_expert_prefetch(replacement, base)
        pager.discard_expert_payload(payload)
        self.assertEqual(pager.metrics()["expert_prefetch_failures"], 1)
        pager.release()

        fallback = DeepSeekWeightPager(
            _MultiEncodedExpertSource([base], adjacent=False),
            device="cpu",
            compute_dtype="bfloat16",
        )
        self.assertIsNone(fallback.prefetch_expert(base))
        self.assertEqual(fallback.metrics()["expert_prefetch_sync_fallbacks"], 1)
        fallback.release()

    def test_current_compute_failure_does_not_wait_for_blocked_next_read(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        first = "layers.0.ffn.experts.7"
        second = "layers.0.ffn.experts.8"
        source = _BlockingExpertSource([first, second])
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        first_ticket = pager.prefetch_expert(first)
        assert first_ticket is not None
        first_payload = pager.consume_expert_prefetch(first_ticket, first)

        source.blocked_shard = "expert-1.safetensors"
        source.started.clear()
        source.release_read.clear()
        next_ticket = pager.prefetch_expert(second)
        assert next_ticket is not None
        self.assertTrue(source.started.wait(timeout=2))
        with (
            mock.patch.object(
                pager,
                "_expert_with_payload",
                side_effect=RuntimeError("simulated current compute failure"),
            ),
            self.assertRaisesRegex(RuntimeError, "current compute"),
        ):
            pager.expert(
                torch.zeros((1, 128), dtype=torch.bfloat16),
                first,
                prefetched_payload=first_payload,
            )

        cancellation_done = threading.Event()

        def cancel() -> None:
            pager.cancel_expert_prefetch(next_ticket)
            cancellation_done.set()

        cancellation = threading.Thread(target=cancel)
        cancellation.start()
        self.assertTrue(cancellation_done.wait(timeout=1))
        cancellation.join(timeout=1)
        self.assertFalse(cancellation.is_alive())
        self.assertTrue(pager.metrics()["expert_prefetch_draining"])
        self.assertIsNone(pager.prefetch_expert(first))

        source.release_read.set()
        self.assertTrue(source.completed.wait(timeout=2))
        draining = pager._draining_prefetch
        if draining is not None:
            draining.result(timeout=2)
        pager.release()
        self.assertFalse(pager.metrics()["expert_prefetch_draining"])

    def test_coalesced_expert_rejects_reserved_float8_weight_and_scale_codes(
        self,
    ) -> None:
        import torch

        from immer.knowledge import TensorEncodingError
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        base = "layers.0.ffn.shared_experts"
        for tensor_name, reserved in (
            (f"{base}.w1.weight", 0x7F),
            (f"{base}.w1.scale", 0xFF),
        ):
            with self.subTest(tensor=tensor_name):
                source = _EncodedExpertSource(base, "fp8", adjacent=True)
                payload = bytearray(source.payload)
                begin = int(source.meta[tensor_name]["offset_in_shard"][0])
                payload[begin] = reserved
                source.payload = bytes(payload)
                pager = DeepSeekWeightPager(
                    source, device="cpu", compute_dtype="float32"
                )
                with self.assertRaisesRegex(
                    TensorEncodingError, r"Reservierte/nicht-endliche"
                ):
                    pager.expert(torch.zeros((1, 128)), base)

    def test_quantized_fp8_and_packed_fp4_execute_end_to_end(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _Source()
        source.data.update(
            {
                "fp8.weight": np.ones((2, 128), dtype=np.float32),
                "fp8.scale": np.full((1, 1), 2.0, dtype=np.float32),
                "fp4.weight": np.full((2, 64), 0x22, dtype=np.int8),
                "fp4.scale": np.full((2, 4), 2.0, dtype=np.float32),
            }
        )
        source.dtypes.update(
            {
                "fp8.weight": "F8_E4M3",
                "fp8.scale": "F8_E8M0",
                "fp4.weight": "I8",
                "fp4.scale": "F8_E8M0",
            }
        )
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        x = torch.ones((1, 128), dtype=torch.float32)
        torch.testing.assert_close(pager.linear(x, "fp8"), torch.full((1, 2), 256.0))
        torch.testing.assert_close(pager.linear(x, "fp4"), torch.full((1, 2), 256.0))
        metrics = pager.metrics()
        self.assertEqual(metrics["linear_calls"], 2)
        self.assertEqual(metrics["peak_single_weight_bytes"], 2 * 128 * 4)
        self.assertEqual(metrics["materialized_scale_bytes"], 4 + 2 * 4 * 4)

    def test_fp8_linear_matches_official_k128_accumulator_seed_130(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        torch.manual_seed(130)
        n, k = 128, 4096
        x = torch.randn(1, k).to(torch.bfloat16)
        qw = torch.randn(n, k).to(torch.float8_e4m3fn).float()
        sw = torch.pow(2.0, torch.randint(-16, 0, (k // 128,)).float())

        grouped = x.float().numpy().reshape(1, -1, 128)
        sx = np.exp2(
            np.ceil(
                np.log2(
                    np.maximum(np.max(np.abs(grouped), axis=-1), np.float32(1e-4))
                    / np.float32(448.0)
                )
            )
        ).astype(np.float32)
        qx = (
            torch.from_numpy(grouped / sx[..., None])
            .to(torch.float8_e4m3fn)
            .float()
            .reshape(1, k)
        )
        expected = torch.zeros(1, n, dtype=torch.float32)
        for block in range(k // 128):
            start = block * 128
            stop = start + 128
            expected += (
                (qx[:, start:stop] @ qw[:, start:stop].T)
                * float(sx[0, block])
                * sw[block]
            )
        expected = expected.to(torch.bfloat16)
        self.assertEqual(expected[0, 127].float().item(), -21.375)

        source = _Source()
        source.data["p.weight"] = qw.numpy()
        source.data["p.scale"] = sw.reshape(1, -1).numpy()
        source.dtypes["p.weight"] = "F8_E4M3"
        source.dtypes["p.scale"] = "F8_E8M0"
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        actual = pager.linear(x, "p")

        self.assertTrue(torch.equal(actual, expected))
        metrics = pager.metrics()
        self.assertEqual(metrics["block_scaled_linear_calls"], 1)
        self.assertEqual(metrics["fp8_k128_tiles"], k // 128)

    def test_fp8_linear_preserves_leading_dims_and_partial_n_tile(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import (
            DeepSeekWeightPager,
            quantize_fp8_e4m3_parts,
        )

        torch.manual_seed(207)
        n, k = 130, 256
        x = torch.randn(2, 3, k).to(torch.bfloat16)
        qw = torch.randn(n, k).to(torch.float8_e4m3fn).float()
        sw = torch.pow(
            2.0,
            torch.asarray([[-8, -4], [-2, -10]], dtype=torch.float32),
        )
        qx_values, sx_values = quantize_fp8_e4m3_parts(x.float().numpy())
        qx = torch.from_numpy(qx_values.reshape(-1, k))
        sx = torch.from_numpy(sx_values.reshape(-1, k // 128))
        row_scales = sw.repeat_interleave(128, dim=0)[:n]
        expected = torch.zeros(qx.shape[0], n, dtype=torch.float32)
        for block in range(k // 128):
            start = block * 128
            stop = start + 128
            expected += (
                (qx[:, start:stop] @ qw[:, start:stop].T)
                * sx[:, block, None]
                * row_scales[None, :, block]
            )
        expected = expected.reshape(2, 3, n).to(torch.bfloat16)

        source = _Source()
        source.data["partial.weight"] = qw.numpy()
        source.data["partial.scale"] = sw.numpy()
        source.dtypes["partial.weight"] = "F8_E4M3"
        source.dtypes["partial.scale"] = "F8_E8M0"
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        actual = pager.linear(x, "partial")
        self.assertTrue(torch.equal(actual, expected))

    def test_fp8_block_accumulator_matches_cpu_on_mps(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        if not torch.backends.mps.is_available():
            self.skipTest("MPS is unavailable")
        torch.manual_seed(431)
        n, k = 128, 512
        x = torch.randn(2, k).to(torch.bfloat16)
        qw = torch.randn(n, k).to(torch.float8_e4m3fn).float().numpy()
        sw = torch.pow(2.0, torch.randint(-12, 1, (1, k // 128)).float()).numpy()

        def source() -> _Source:
            result = _Source()
            result.data["mps.weight"] = qw
            result.data["mps.scale"] = sw
            result.dtypes["mps.weight"] = "F8_E4M3"
            result.dtypes["mps.scale"] = "F8_E8M0"
            return result

        cpu = DeepSeekWeightPager(
            source(), device="cpu", compute_dtype="bfloat16"
        ).linear(x, "mps")
        mps = DeepSeekWeightPager(
            source(), device="mps", compute_dtype="bfloat16"
        ).linear(x, "mps")
        self.assertTrue(torch.equal(cpu, mps.to("cpu")))

    def test_fp4_linear_matches_official_k32_weight_scale_rule(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import (
            DeepSeekWeightPager,
            quantize_fp8_e4m3_parts,
        )

        rng = np.random.default_rng(19)
        n, k = 5, 256
        low = rng.integers(0, 16, size=(n, k // 2), dtype=np.uint8)
        high = rng.integers(0, 16, size=(n, k // 2), dtype=np.uint8)
        packed = np.ascontiguousarray(low | (high << np.uint8(4))).view(np.int8)
        table = np.asarray(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=np.float32,
        )
        bits = packed.view(np.uint8)
        qw_values = np.empty((n, k), dtype=np.float32)
        qw_values[:, 0::2] = table[bits & np.uint8(0x0F)]
        qw_values[:, 1::2] = table[(bits >> np.uint8(4)) & np.uint8(0x0F)]
        sw_values = np.exp2(rng.integers(-12, 1, size=(n, k // 32)).astype(np.float32))
        torch.manual_seed(19)
        x = torch.randn(2, k).to(torch.bfloat16)
        qx_values, sx_values = quantize_fp8_e4m3_parts(x.float().numpy())
        qx = torch.from_numpy(qx_values)
        sx = torch.from_numpy(sx_values)
        qw = torch.from_numpy(qw_values)
        sw = torch.from_numpy(sw_values)
        expected = torch.zeros(2, n, dtype=torch.float32)
        for block in range(k // 32):
            start = block * 32
            stop = start + 32
            expected += (
                (qx[:, start:stop] @ qw[:, start:stop].T)
                * sx[:, block // 4, None]
                * sw[None, :, block]
            )
        expected = expected.to(torch.bfloat16)

        source = _Source()
        source.data["fp4_exact.weight"] = packed
        source.data["fp4_exact.scale"] = sw_values
        source.dtypes["fp4_exact.weight"] = "I8"
        source.dtypes["fp4_exact.scale"] = "F8_E8M0"
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="bfloat16")
        actual = pager.linear(x, "fp4_exact")
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(pager.metrics()["fp4_k32_tiles"], k // 32)

    def test_dense_linear_and_metrics(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _Source()
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        actual = pager.linear(torch.asarray([[5.0, 6.0]]), "dense")
        torch.testing.assert_close(actual, torch.asarray([[17.0, 39.0]]))
        metrics = pager.metrics()
        self.assertEqual(metrics["linear_calls"], 1)
        self.assertEqual(metrics["logical_weight_bytes"], 16)
        self.assertEqual(metrics["peak_single_weight_bytes"], 16)

    def test_auto_compute_dtype_preserves_official_bfloat16_on_cpu(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        pager = DeepSeekWeightPager(_Source(), device="cpu", compute_dtype="auto")
        self.assertEqual(pager.compute_dtype, torch.bfloat16)

    def test_embedding_coalesces_consecutive_unique_rows_and_restores_order(
        self,
    ) -> None:
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _Source()
        pager = DeepSeekWeightPager(source, device="cpu")
        actual = pager.embedding([5, 2, 3, 5]).float().numpy()
        np.testing.assert_array_equal(actual, source.data["embed.weight"][[5, 2, 3, 5]])
        self.assertEqual(
            source.row_calls,
            [("embed.weight", 2, 2), ("embed.weight", 5, 1)],
        )

    def test_grouped_linear_does_not_mix_groups(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        pager = DeepSeekWeightPager(_Source(), device="cpu", compute_dtype="float32")
        x = torch.asarray([[[5.0, 7.0], [11.0, 13.0]]])
        actual = pager.grouped_linear(x, "group", groups=2)
        torch.testing.assert_close(actual, torch.asarray([[[5.0, 7.0], [22.0, 39.0]]]))

    def test_candidate_head_coalesces_rows_and_restores_requested_order(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _Source()
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        hidden = torch.asarray([[2.0, 3.0]])
        actual = pager.candidate_logits(hidden, [3, 1, 2, 3])
        expected = torch.nn.functional.linear(
            hidden, torch.from_numpy(source.data["head.weight"][[3, 1, 2, 3]])
        )
        torch.testing.assert_close(actual, expected)
        self.assertEqual(source.row_calls, [("head.weight", 1, 3)])

    def test_projection_releases_do_not_purge_until_explicit_boundary(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _Source()
        base = "layers.0.ffn.shared_experts"
        source.data.update(
            {
                f"{base}.w1.weight": np.eye(2, dtype=np.float32),
                f"{base}.w2.weight": np.eye(2, dtype=np.float32),
                f"{base}.w3.weight": np.asarray(
                    [[2.0, 0.0], [0.0, 3.0]], dtype=np.float32
                ),
            }
        )
        source.dtypes.update(
            {name: "F32" for name in source.data if name.startswith(base)}
        )
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        x = torch.asarray([[1.0, 2.0]])
        fake_mps = mock.Mock()

        with (
            mock.patch.object(pager.torch, "mps", fake_mps, create=True),
            mock.patch.object(pager, "_uses_mps_allocator", return_value=True),
        ):
            dense = pager.linear(x, "dense")
            expert = pager.expert(x, base)
            grouped = pager.grouped_linear(
                torch.asarray([[[5.0, 7.0], [11.0, 13.0]]]),
                "group",
                groups=2,
            )

            torch.testing.assert_close(dense, torch.asarray([[5.0, 11.0]]))
            expected_expert = torch.nn.functional.silu(x) * torch.asarray([[2.0, 6.0]])
            torch.testing.assert_close(expert, expected_expert)
            torch.testing.assert_close(
                grouped, torch.asarray([[[5.0, 7.0], [22.0, 39.0]]])
            )
            fake_mps.empty_cache.assert_not_called()
            metrics = pager.metrics()
            self.assertEqual(metrics["linear_calls"], 5)
            self.assertEqual(metrics["expert_calls"], 1)
            self.assertEqual(metrics["materialized_weight_releases"], 5)
            self.assertEqual(metrics["release_boundaries"], 0)
            self.assertEqual(metrics["mps_cache_purges"], 0)

            pager.release()

        fake_mps.empty_cache.assert_called_once_with()
        metrics = pager.metrics()
        self.assertEqual(metrics["release_boundaries"], 1)
        self.assertEqual(metrics["mps_cache_purges"], 1)

    def test_cpu_release_and_missing_mps_cache_api_are_safe(self) -> None:
        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        pager = DeepSeekWeightPager(_Source(), device="cpu", compute_dtype="float32")
        fake_mps = mock.Mock()
        with mock.patch.object(pager.torch, "mps", fake_mps, create=True):
            pager.release()
        fake_mps.empty_cache.assert_not_called()
        self.assertEqual(pager.metrics()["release_boundaries"], 1)
        self.assertEqual(pager.metrics()["mps_cache_purges"], 0)

        with (
            mock.patch.object(pager, "_uses_mps_allocator", return_value=True),
            mock.patch.object(pager.torch, "mps", object(), create=True),
        ):
            self.assertFalse(pager._purge_mps_cache())
        self.assertEqual(pager.metrics()["mps_cache_purges"], 0)

    def test_blockwise_head_topk_matches_full_linear(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _Source()
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        hidden = torch.asarray([[2.0, 3.0]])
        values, ids = pager.topk_logits(hidden, k=2, block_rows=2)
        full = torch.nn.functional.linear(
            hidden, torch.from_numpy(source.data["head.weight"])
        )
        expected_values, expected_ids = torch.topk(full, 2, dim=-1)
        torch.testing.assert_close(values, expected_values)
        torch.testing.assert_close(ids, expected_ids)
        self.assertEqual(pager.metrics()["head_rows"], 4)

    def test_blockwise_head_topk_breaks_exact_ties_by_lower_token_id(self) -> None:
        import torch

        from immer.runtimes.deepseek_v4 import DeepSeekWeightPager

        source = _Source()
        source.data["head.weight"] = np.ones((7, 2), dtype=np.float32)
        pager = DeepSeekWeightPager(source, device="cpu", compute_dtype="float32")
        values, ids = pager.topk_logits(torch.asarray([[1.0, 1.0]]), k=4, block_rows=3)
        torch.testing.assert_close(values, torch.full((1, 4), 2.0))
        torch.testing.assert_close(ids, torch.asarray([[0, 1, 2, 3]]))


if __name__ == "__main__":
    unittest.main()
