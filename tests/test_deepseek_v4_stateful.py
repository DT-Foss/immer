from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from immer.runtimes.deepseek_v4.kernels import (
    compressed_indices,
    precompute_freqs_cis,
    window_indices,
)
from immer.runtimes.deepseek_v4.stateful import (
    CircularKVCache,
    DeepSeekV4Compressor,
    DeepSeekV4Indexer,
    NativeAttentionState,
)


def _direct_rms(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    dtype = x.dtype
    values = x.float()
    values = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return (values * weight.float()).to(dtype)


def _direct_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    values = x.float().unflatten(-1, (-1, 2))
    real, imag = values.unbind(-1)
    shape = (1, x.shape[1], *((1,) * (x.ndim - 3)), x.shape[-1] // 2)
    cos = freqs.real.to(x.device).reshape(shape)
    sin = freqs.imag.to(x.device).reshape(shape)
    return (
        torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1)
        .flatten(-2)
        .to(x.dtype)
    )


def _hadamard_qat(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode != "indexer":
        return x
    if x.shape[-1] != 4:
        raise AssertionError("synthetic Hadamard expects four features")
    matrix = (
        x.new_tensor(((1, 1, 1, 1), (1, -1, 1, -1), (1, 1, -1, -1), (1, -1, -1, 1)))
        / 2.0
    )
    return torch.matmul(x, matrix.T)


class _Callbacks:
    def __init__(self, weights: dict[str, torch.Tensor]) -> None:
        self.weights = weights
        self.qat_modes: list[str] = []

    def linear(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return F.linear(x, self.weights[name].to(x.device))

    def rms(self, x: torch.Tensor, name: str) -> torch.Tensor:
        return _direct_rms(x, self.weights[name].to(x.device))

    def identity_qat(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        self.qat_modes.append(mode)
        return x

    def hadamard_qat(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        self.qat_modes.append(mode)
        return _hadamard_qat(x, mode)


def _direct_prefill(
    x: torch.Tensor,
    *,
    ratio: int,
    head_dim: int,
    rope_dim: int,
    ape: torch.Tensor,
    freqs: torch.Tensor,
    weights: dict[str, torch.Tensor],
    prefix: str,
    qat,
) -> torch.Tensor | None:
    overlap = ratio == 4
    kv = F.linear(x.float(), weights[f"{prefix}.wkv"])
    score = F.linear(x.float(), weights[f"{prefix}.wgate"])
    cutoff = x.shape[1] - x.shape[1] % ratio
    if not cutoff:
        return None
    kv = kv[:, :cutoff].unflatten(1, (-1, ratio))
    score = score[:, :cutoff].unflatten(1, (-1, ratio)) + ape
    if overlap:
        blocks = cutoff // ratio
        transformed_kv = kv.new_zeros((x.shape[0], blocks, 2 * ratio, head_dim))
        transformed_score = score.new_full(
            (x.shape[0], blocks, 2 * ratio, head_dim), float("-inf")
        )
        transformed_kv[:, :, ratio:] = kv[..., head_dim:]
        transformed_score[:, :, ratio:] = score[..., head_dim:]
        transformed_kv[:, 1:, :ratio] = kv[:, :-1, :, :head_dim]
        transformed_score[:, 1:, :ratio] = score[:, :-1, :, :head_dim]
        kv, score = transformed_kv, transformed_score
    pooled = (kv * score.softmax(dim=2)).sum(dim=2)
    pooled = _direct_rms(pooled.to(x.dtype), weights[f"{prefix}.norm.weight"])
    rope = _direct_rope(pooled[..., -rope_dim:], freqs[:cutoff:ratio])
    pooled = torch.cat((pooled[..., :-rope_dim], rope), dim=-1)
    return qat(
        pooled, "indexer" if overlap and "indexer" in prefix else "compressed-kv"
    )


def _compressor(
    *,
    callbacks: _Callbacks,
    prefix: str,
    ratio: int,
    head_dim: int,
    rope_dim: int,
    max_seq_len: int,
    ape: torch.Tensor,
    freqs: torch.Tensor,
    rotate: bool = False,
) -> DeepSeekV4Compressor:
    return DeepSeekV4Compressor(
        prefix=prefix,
        compress_ratio=ratio,
        head_dim=head_dim,
        rope_head_dim=rope_dim,
        max_batch_size=2,
        max_seq_len=max_seq_len,
        ape=ape,
        freqs_cis=freqs,
        linear=callbacks.linear,
        rms=callbacks.rms,
        qat=callbacks.hadamard_qat if rotate else callbacks.identity_qat,
        rotate=rotate,
    )


class CircularKVCacheTests(unittest.TestCase):
    def test_prefill_and_decode_preserve_official_physical_ring(self) -> None:
        values = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
        cache = CircularKVCache(max_batch_size=1, window_size=4, head_dim=2)
        cache.prefill(values)
        expected = torch.cat((values[:, 4:6], values[:, 2:4]), dim=1)
        torch.testing.assert_close(cache.physical(), expected)
        torch.testing.assert_close(cache.ordered(), values[:, 2:6])

        token = torch.tensor([[[100.0, 101.0]]])
        cache.append(token, start_pos=6)
        expected[:, 2] = token[:, 0]
        torch.testing.assert_close(cache.physical(), expected)
        torch.testing.assert_close(
            cache.ordered(), torch.cat((values[:, 3:6], token), dim=1)
        )
        self.assertEqual(cache.next_position, 7)
        self.assertEqual(cache.nbytes, 4 * 2 * 4)

    def test_reset_reuses_or_releases_storage_and_rejects_gaps(self) -> None:
        cache = CircularKVCache(max_batch_size=2, window_size=3, head_dim=2)
        cache.prefill(torch.ones(1, 1, 2))
        with self.assertRaisesRegex(ValueError, "non-contiguous"):
            cache.append(torch.ones(1, 1, 2), start_pos=2)
        cache.reset()
        self.assertEqual(cache.next_position, 0)
        self.assertTrue(torch.count_nonzero(cache.physical()) == 0)
        cache.prefill(torch.ones(2, 1, 2))
        self.assertEqual(tuple(cache.physical().shape), (2, 3, 2))
        cache.reset(release=True)
        self.assertEqual(cache.nbytes, 0)
        with self.assertRaisesRegex(RuntimeError, "not been initialized"):
            cache.physical()


class CompressorTests(unittest.TestCase):
    def test_ratio128_matches_direct_official_formula(self) -> None:
        torch.manual_seed(11)
        ratio, dim, rope_dim, input_dim, seqlen = 128, 4, 2, 3, 130
        prefix = "compressor"
        weights = {
            f"{prefix}.wkv": torch.randn(dim, input_dim),
            f"{prefix}.wgate": torch.randn(dim, input_dim),
            f"{prefix}.norm.weight": torch.randn(dim),
        }
        ape = torch.randn(ratio, dim)
        freqs = precompute_freqs_cis(rope_dim, 256, base=40_000.0)
        callbacks = _Callbacks(weights)
        compressor = _compressor(
            callbacks=callbacks,
            prefix=prefix,
            ratio=ratio,
            head_dim=dim,
            rope_dim=rope_dim,
            max_seq_len=256,
            ape=ape,
            freqs=freqs,
        )
        x = torch.randn(2, seqlen, input_dim)

        actual = compressor.forward(x, 0)
        expected = _direct_prefill(
            x,
            ratio=ratio,
            head_dim=dim,
            rope_dim=rope_dim,
            ape=ape,
            freqs=freqs,
            weights=weights,
            prefix=prefix,
            qat=lambda value, _mode: value,
        )
        assert actual is not None and expected is not None
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(compressor.cache.active(), expected)
        self.assertEqual(compressor.next_position, seqlen)
        self.assertEqual(compressor.cache.length, 1)
        self.assertEqual(callbacks.qat_modes, ["compressed-kv"])

    def test_ratio4_overlap_chunking_equals_full_prefill(self) -> None:
        torch.manual_seed(17)
        ratio, dim, rope_dim, input_dim, seqlen = 4, 4, 2, 3, 10
        prefix = "compressor"
        weights = {
            f"{prefix}.wkv": torch.randn(2 * dim, input_dim),
            f"{prefix}.wgate": torch.randn(2 * dim, input_dim),
            f"{prefix}.norm.weight": torch.randn(dim),
        }
        ape = torch.randn(ratio, 2 * dim)
        freqs = precompute_freqs_cis(rope_dim, 16, base=40_000.0)
        x = torch.randn(1, seqlen, input_dim)

        full_callbacks = _Callbacks(weights)
        full = _compressor(
            callbacks=full_callbacks,
            prefix=prefix,
            ratio=ratio,
            head_dim=dim,
            rope_dim=rope_dim,
            max_seq_len=16,
            ape=ape,
            freqs=freqs,
        )
        full_output = full.forward(x, 0)
        expected = _direct_prefill(
            x,
            ratio=ratio,
            head_dim=dim,
            rope_dim=rope_dim,
            ape=ape,
            freqs=freqs,
            weights=weights,
            prefix=prefix,
            qat=lambda value, _mode: value,
        )
        assert full_output is not None and expected is not None
        torch.testing.assert_close(full_output, expected, rtol=1e-5, atol=1e-6)

        chunk_callbacks = _Callbacks(weights)
        chunked = _compressor(
            callbacks=chunk_callbacks,
            prefix=prefix,
            ratio=ratio,
            head_dim=dim,
            rope_dim=rope_dim,
            max_seq_len=16,
            ape=ape,
            freqs=freqs,
        )
        self.assertIsNone(chunked.forward(x[:, :3], 0))
        first = chunked.forward(x[:, 3:6], 3)
        second = chunked.forward(x[:, 6:], 6)
        assert first is not None and second is not None
        tokenwise_output = torch.cat((first, second), dim=1)
        torch.testing.assert_close(tokenwise_output, full_output, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(chunked.cache.active(), full.cache.active())
        self.assertEqual(chunked.next_position, seqlen)
        self.assertLessEqual(chunked.cache.capacity, 2)

        chunked.reset()
        self.assertEqual(chunked.next_position, 0)
        self.assertEqual(chunked.cache.length, 0)
        self.assertIsNone(chunked.forward(x[:, :2], 0))


class IndexerTests(unittest.TestCase):
    def _fixture(self):
        torch.manual_seed(23)
        dim, rope_dim, input_dim, qrank, heads = 4, 2, 3, 3, 2
        prefix = "indexer"
        compressor_prefix = f"{prefix}.compressor"
        weights = {
            f"{compressor_prefix}.wkv": torch.randn(2 * dim, input_dim),
            f"{compressor_prefix}.wgate": torch.randn(2 * dim, input_dim),
            f"{compressor_prefix}.norm.weight": torch.randn(dim),
            f"{prefix}.wq_b": torch.randn(heads * dim, qrank),
            f"{prefix}.weights_proj": torch.randn(heads, input_dim),
        }
        ape = torch.randn(4, 2 * dim)
        freqs = precompute_freqs_cis(rope_dim, 16, base=40_000.0)
        x = torch.randn(1, 8, input_dim)
        qr = torch.randn(1, 8, qrank)
        return dim, rope_dim, heads, prefix, weights, ape, freqs, x, qr

    def _build(self, fixture):
        dim, rope_dim, heads, prefix, weights, ape, freqs, _x, _qr = fixture
        callbacks = _Callbacks(weights)
        compressor = _compressor(
            callbacks=callbacks,
            prefix=f"{prefix}.compressor",
            ratio=4,
            head_dim=dim,
            rope_dim=rope_dim,
            max_seq_len=16,
            ape=ape,
            freqs=freqs,
            rotate=True,
        )
        indexer = DeepSeekV4Indexer(
            prefix=prefix,
            compressor=compressor,
            n_heads=heads,
            head_dim=dim,
            rope_head_dim=rope_dim,
            index_topk=2,
            freqs_cis=freqs,
            linear=callbacks.linear,
            qat=callbacks.hadamard_qat,
        )
        return callbacks, indexer

    def test_prefill_topk_matches_direct_official_scoring(self) -> None:
        fixture = self._fixture()
        dim, rope_dim, heads, prefix, weights, ape, freqs, x, qr = fixture
        callbacks, indexer = self._build(fixture)
        actual = indexer.forward(x, qr, start_pos=0, offset=x.shape[1])

        compressed = _direct_prefill(
            x,
            ratio=4,
            head_dim=dim,
            rope_dim=rope_dim,
            ape=ape,
            freqs=freqs,
            weights=weights,
            prefix=f"{prefix}.compressor",
            qat=_hadamard_qat,
        )
        assert compressed is not None
        q = F.linear(qr, weights[f"{prefix}.wq_b"]).unflatten(-1, (heads, dim))
        q_rope = _direct_rope(q[..., -rope_dim:], freqs[: x.shape[1]])
        q = torch.cat((q[..., :-rope_dim], q_rope), dim=-1)
        q = _hadamard_qat(q, "indexer")
        projected_weights = F.linear(x, weights[f"{prefix}.weights_proj"])
        projected_weights *= dim**-0.5 * heads**-0.5
        scores = torch.einsum("bshd,btd->bsht", q, compressed)
        scores = (scores.relu() * projected_weights.unsqueeze(-1)).sum(dim=2)
        blocks = torch.arange(2).repeat(8, 1)
        complete = torch.arange(1, 9).unsqueeze(1) // 4
        scores = scores.masked_fill(blocks >= complete, float("-inf"))
        expected = scores.topk(2, dim=-1).indices
        expected = torch.where(expected >= complete, -1, expected + 8).to(torch.int32)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(indexer.cache.length, 2)
        self.assertEqual(callbacks.qat_modes.count("indexer"), 2)

    def test_token_decode_final_selection_equals_prefill_final_row(self) -> None:
        fixture = self._fixture()
        *_prefix_data, x, qr = fixture
        _callbacks, full = self._build(fixture)
        expected = full.forward(x, qr, start_pos=0, offset=11)[:, -1]

        _callbacks, incremental = self._build(fixture)
        empty = incremental.forward(x[:, :3], qr[:, :3], start_pos=0, offset=11)
        self.assertEqual(tuple(empty.shape), (1, 3, 0))
        final = None
        for position in range(3, 8):
            final = incremental.forward(
                x[:, position : position + 1],
                qr[:, position : position + 1],
                start_pos=position,
                offset=11,
            )
        assert final is not None
        torch.testing.assert_close(final[:, 0], expected)
        torch.testing.assert_close(incremental.cache.active(), full.cache.active())


class NativeAttentionStateTests(unittest.TestCase):
    def test_prefill_decode_assembly_and_bounded_reset(self) -> None:
        torch.manual_seed(31)
        ratio, dim, rope_dim, input_dim = 2, 4, 2, 3
        prefix = "compressor"
        weights = {
            f"{prefix}.wkv": torch.randn(dim, input_dim),
            f"{prefix}.wgate": torch.randn(dim, input_dim),
            f"{prefix}.norm.weight": torch.randn(dim),
        }
        callbacks = _Callbacks(weights)
        ape = torch.randn(ratio, dim)
        freqs = precompute_freqs_cis(rope_dim, 10, base=40_000.0)
        compressor = _compressor(
            callbacks=callbacks,
            prefix=prefix,
            ratio=ratio,
            head_dim=dim,
            rope_dim=rope_dim,
            max_seq_len=10,
            ape=ape,
            freqs=freqs,
        )
        state = NativeAttentionState(
            max_batch_size=1,
            max_seq_len=10,
            window_size=3,
            head_dim=dim,
            compress_ratio=ratio,
            compressor=compressor,
        )
        x = torch.randn(1, 5, input_dim)
        kv = torch.randn(1, 5, dim)
        prefill = state.assemble(kv, start_pos=0, x=x)
        expected_indices = torch.cat(
            (
                window_indices(3, 1, 5, 0),
                compressed_indices(2, 1, 5, 0, offset=5),
            ),
            dim=-1,
        )
        torch.testing.assert_close(prefill.indices, expected_indices)
        torch.testing.assert_close(prefill.kv[:, :5], kv)
        torch.testing.assert_close(prefill.kv[:, 5:], compressor.cache.active())
        self.assertEqual(prefill.local_cache_entries, 5)
        self.assertEqual(prefill.compressed_cache_entries, 2)

        x_next = torch.randn(1, 1, input_dim)
        kv_next = torch.randn(1, 1, dim)
        decode = state.assemble(kv_next, start_pos=5, x=x_next)
        expected_indices = torch.cat(
            (
                window_indices(3, 1, 1, 5),
                compressed_indices(2, 1, 1, 5, offset=3),
            ),
            dim=-1,
        )
        torch.testing.assert_close(decode.indices, expected_indices)
        torch.testing.assert_close(decode.kv[:, :3], state.local.physical())
        torch.testing.assert_close(decode.kv[:, 3:], compressor.cache.active())
        self.assertEqual(decode.compressed_cache_entries, 3)
        self.assertGreater(state.state_nbytes, 0)

        q = torch.randn(1, 1, 2, dim)
        sink = torch.randn(2)
        output = decode.attend(q, sink)
        self.assertEqual(tuple(output.shape), tuple(q.shape))
        self.assertTrue(torch.isfinite(output).all())

        state.reset(release=True)
        self.assertEqual(state.next_position, 0)
        self.assertEqual(state.state_nbytes, 0)

    def test_uncompressed_contract_needs_no_branch_callbacks(self) -> None:
        state = NativeAttentionState(
            max_batch_size=1,
            max_seq_len=8,
            window_size=4,
            head_dim=2,
        )
        kv = torch.randn(1, 3, 2)
        assembly = state.assemble(kv, start_pos=0)
        torch.testing.assert_close(assembly.kv, kv)
        torch.testing.assert_close(assembly.indices, window_indices(4, 1, 3, 0))
        self.assertEqual(assembly.compressed_cache_entries, 0)

    @unittest.skipUnless(torch.backends.mps.is_available(), "Apple MPS unavailable")
    def test_compressed_prefill_and_decode_run_on_mps(self) -> None:
        torch.manual_seed(37)
        weights = {
            "compressor.wkv": torch.randn(4, 3),
            "compressor.wgate": torch.randn(4, 3),
            "compressor.norm.weight": torch.ones(4),
        }
        callbacks = _Callbacks(weights)
        compressor = _compressor(
            callbacks=callbacks,
            prefix="compressor",
            ratio=2,
            head_dim=4,
            rope_dim=2,
            max_seq_len=8,
            ape=torch.randn(2, 4),
            freqs=precompute_freqs_cis(2, 8),
        )
        state = NativeAttentionState(
            max_batch_size=1,
            max_seq_len=8,
            window_size=3,
            head_dim=4,
            compress_ratio=2,
            compressor=compressor,
        )
        prefill = state.assemble(
            torch.randn(1, 5, 4, device="mps"),
            start_pos=0,
            x=torch.randn(1, 5, 3, device="mps"),
        )
        decode = state.assemble(
            torch.randn(1, 1, 4, device="mps"),
            start_pos=5,
            x=torch.randn(1, 1, 3, device="mps"),
        )
        torch.mps.synchronize()
        self.assertEqual(prefill.kv.device.type, "mps")
        self.assertEqual(decode.kv.device.type, "mps")
        self.assertEqual(tuple(decode.kv.shape), (1, 6, 4))


if __name__ == "__main__":
    unittest.main()
