from __future__ import annotations

import math
import unittest

try:
    import torch
except ImportError:  # pragma: no cover - neural extra is optional
    torch = None


@unittest.skipIf(torch is None, "DeepSeek-V4 kernels require the neural extra")
class DeepSeekV4KernelTests(unittest.TestCase):
    def test_normalized_hadamard_matches_dense_walsh_matrix(self) -> None:
        from immer.runtimes.deepseek_v4.kernels import hadamard_transform

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        matrix = (
            torch.tensor(
                [
                    [1.0, 1.0, 1.0, 1.0],
                    [1.0, -1.0, 1.0, -1.0],
                    [1.0, 1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0, 1.0],
                ]
            )
            / 2.0
        )
        torch.testing.assert_close(hadamard_transform(x), x @ matrix.T)

    def test_rms_norm_matches_direct_construction(self) -> None:
        from immer.runtimes.deepseek_v4.kernels import rms_norm

        x = torch.tensor([[[1.0, -2.0, 3.0], [4.0, 0.5, -1.0]]])
        weight = torch.tensor([0.5, 1.5, -2.0])
        expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)
        expected = expected * weight
        torch.testing.assert_close(rms_norm(x, weight), expected)

    def test_rope_table_and_rotation_match_dense_real_math(self) -> None:
        from immer.runtimes.deepseek_v4.kernels import (
            apply_rotary_emb,
            precompute_freqs_cis,
        )

        freqs = precompute_freqs_cis(4, 3, 0, 10_000.0, 1.0, 32, 1)
        inverse_freq = 1.0 / (
            10_000.0 ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4)
        )
        angles = torch.outer(torch.arange(3, dtype=torch.float32), inverse_freq)
        expected_freqs = torch.polar(torch.ones_like(angles), angles)
        torch.testing.assert_close(freqs, expected_freqs)

        x = torch.arange(1, 1 + 1 * 3 * 2 * 4, dtype=torch.float32).view(1, 3, 2, 4)
        original = x.clone()
        pairs = original.unflatten(-1, (2, 2))
        real, imag = pairs.unbind(-1)
        cos = expected_freqs.real.view(1, 3, 1, 2)
        sin = expected_freqs.imag.view(1, 3, 1, 2)
        expected = torch.stack(
            (real * cos - imag * sin, real * sin + imag * cos), dim=-1
        ).flatten(-2)
        returned = apply_rotary_emb(x, freqs)
        self.assertIs(returned, x)
        torch.testing.assert_close(x, expected)
        apply_rotary_emb(x, freqs, inverse=True)
        torch.testing.assert_close(x, original, atol=2e-5, rtol=2e-5)

    def test_rope_real_pair_table_executes_on_every_available_device(self) -> None:
        from immer.runtimes.deepseek_v4.kernels import (
            apply_rotary_emb,
            precompute_freqs_cis,
        )

        freqs = precompute_freqs_cis(4, 2)
        pair_table = torch.stack((freqs.real, freqs.imag), dim=-1)
        devices = ["cpu"]
        if torch.backends.mps.is_available():
            devices.append("mps")
        expected = None
        for device in devices:
            with self.subTest(device=device):
                value = torch.tensor(
                    [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]],
                    device=device,
                )
                apply_rotary_emb(value, pair_table)
                actual = value.cpu()
                if expected is None:
                    expected = actual
                else:
                    torch.testing.assert_close(actual, expected)

    def test_window_indices_are_causal_and_decode_circularly(self) -> None:
        from immer.runtimes.deepseek_v4.kernels import window_indices

        actual = window_indices(3, 2, 5, 0)
        expected = torch.tensor(
            [
                [0, -1, -1],
                [0, 1, -1],
                [0, 1, 2],
                [1, 2, 3],
                [2, 3, 4],
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual[0], expected)
        torch.testing.assert_close(actual[1], expected)
        for query, row in enumerate(actual[0]):
            self.assertTrue(torch.all(row[row >= 0] <= query))

        decoded = window_indices(3, 1, 1, 4)
        torch.testing.assert_close(
            decoded, torch.tensor([[[2, 0, 1]]], dtype=torch.int32)
        )

    def test_compressed_indices_only_expose_complete_blocks(self) -> None:
        from immer.runtimes.deepseek_v4.kernels import compressed_indices

        actual = compressed_indices(2, 1, 6, 0, 6)
        expected = torch.tensor(
            [
                [
                    [-1, -1, -1],
                    [6, -1, -1],
                    [6, -1, -1],
                    [6, 7, -1],
                    [6, 7, -1],
                    [6, 7, 8],
                ]
            ],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected)
        for query, row in enumerate(actual[0]):
            blocks = row[row >= 0] - 6
            self.assertTrue(torch.all(blocks < (query + 1) // 2))
        torch.testing.assert_close(
            compressed_indices(2, 1, 1, 5, 6),
            torch.tensor([[[6, 7, 8]]], dtype=torch.int32),
        )

    @staticmethod
    def _scalar_sparse_reference(q, kv, sink, indices, scale):
        output = torch.zeros_like(q, dtype=torch.float32)
        for batch in range(q.shape[0]):
            for sequence in range(q.shape[1]):
                for head in range(q.shape[2]):
                    valid = [
                        int(index) for index in indices[batch, sequence] if index >= 0
                    ]
                    logits = [
                        float(torch.dot(q[batch, sequence, head], kv[batch, index]))
                        * scale
                        for index in valid
                    ]
                    logits.append(float(sink[head]))
                    maximum = max(logits)
                    exponentials = [math.exp(value - maximum) for value in logits]
                    denominator = sum(exponentials)
                    for probability, index in zip(exponentials[:-1], valid):
                        output[batch, sequence, head] += (
                            probability / denominator * kv[batch, index]
                        )
        return output

    def test_sparse_attention_matches_scalar_sink_reference(self) -> None:
        from immer.runtimes.deepseek_v4.kernels import sparse_attention

        q = torch.tensor([[[[1.0, 0.5], [-0.5, 1.0]], [[0.25, 2.0], [1.0, -1.0]]]])
        kv = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, -1.0]]])
        sink = torch.tensor([-0.25, 0.75])
        indices = torch.tensor([[[0, -1, -1], [0, 2, 1]]], dtype=torch.int32)
        scale = 0.5
        expected = self._scalar_sparse_reference(q, kv, sink, indices, scale)
        actual = sparse_attention(q, kv, sink, indices, scale)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

        all_sink = sparse_attention(
            q[:, :1], kv, sink, torch.full((1, 1, 3), -1, dtype=torch.int32), scale
        )
        torch.testing.assert_close(all_sink, torch.zeros_like(all_sink))

    def test_sparse_attention_matches_official_blockwise_bfloat16_cast(self) -> None:
        from immer.runtimes.deepseek_v4.kernels import sparse_attention

        # This 65-entry fixture crosses the official 64-wide tile boundary;
        # a global softmax differs from the literal online loop on this seed.
        torch.manual_seed(274)
        q = torch.randn(1, 1, 4, 128, dtype=torch.bfloat16) * 3
        kv = torch.randn(1, 65, 128, dtype=torch.bfloat16) * 3
        sink = torch.randn(4, dtype=torch.float32) * 3
        indices = torch.arange(65, dtype=torch.int32).reshape(1, 1, 65)
        scale = 128**-0.5

        running_max = torch.full((1, 1, 4), float("-inf"), dtype=torch.float32)
        denominator = torch.zeros_like(running_max)
        numerator = torch.zeros(1, 1, 4, 128, dtype=torch.float32)
        q32 = q.float()
        for start in range(0, 65, 64):
            values = kv[:, start : start + 64].float()
            scores = torch.einsum("bshd,bkd->bshk", q32, values) * scale
            next_max = torch.maximum(running_max, scores.amax(dim=-1))
            rescale = torch.exp(running_max - next_max)
            exponentials = torch.exp(scores - next_max.unsqueeze(-1))
            denominator = denominator * rescale + exponentials.sum(dim=-1)
            numerator = numerator * rescale.unsqueeze(-1) + torch.einsum(
                "bshk,bkd->bshd", exponentials.to(torch.bfloat16).float(), values
            )
            running_max = next_max
        denominator += torch.exp(sink.view(1, 1, -1) - running_max)
        expected = (numerator / denominator.unsqueeze(-1)).to(torch.bfloat16)

        actual = sparse_attention(q, kv, sink, indices, scale)
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_sparse_attention_window_causality_blocks_future_values(self) -> None:
        from immer.runtimes.deepseek_v4.kernels import sparse_attention, window_indices

        torch.manual_seed(4)
        q = torch.randn(1, 5, 2, 4)
        kv = torch.randn(1, 5, 4)
        sink = torch.randn(2)
        indices = window_indices(3, 1, 5, 0)
        baseline = sparse_attention(q, kv, sink, indices, 0.5)
        for query in range(5):
            changed = kv.clone()
            changed[:, query + 1 :] += 10_000.0
            perturbed = sparse_attention(q, changed, sink, indices, 0.5)
            torch.testing.assert_close(perturbed[:, query], baseline[:, query])

    @unittest.skipUnless(
        torch is not None and torch.backends.mps.is_available(),
        "MPS is unavailable",
    )
    def test_norm_sparse_attention_and_sinkhorn_match_mps(self) -> None:
        from immer.runtimes.deepseek_v4.kernels import (
            hc_split_sinkhorn,
            rms_norm,
            sparse_attention,
        )

        torch.manual_seed(21)
        x = torch.randn(2, 3, 8)
        weight = torch.randn(8)
        torch.testing.assert_close(
            rms_norm(x.to("mps"), weight.to("mps")).cpu(),
            rms_norm(x, weight),
            atol=2e-5,
            rtol=2e-5,
        )

        q = torch.randn(1, 3, 2, 4)
        kv = torch.randn(1, 4, 4)
        sink = torch.randn(2)
        indices = torch.tensor([[[0, -1], [0, 1], [1, 2]]], dtype=torch.int32)
        torch.testing.assert_close(
            sparse_attention(
                q.to("mps"), kv.to("mps"), sink.to("mps"), indices.to("mps")
            ).cpu(),
            sparse_attention(q, kv, sink, indices),
            atol=2e-5,
            rtol=2e-5,
        )

        hc = 2
        mixes = torch.randn(1, 2, (2 + hc) * hc)
        scale = torch.randn(3)
        base = torch.randn((2 + hc) * hc)
        cpu = hc_split_sinkhorn(mixes, scale, base, hc, 4)
        mps = hc_split_sinkhorn(mixes.to("mps"), scale.to("mps"), base.to("mps"), hc, 4)
        for actual, expected in zip(mps, cpu):
            torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-5)

    @staticmethod
    def _scalar_hc_reference(mixes, scale, base, hc, iterations, eps):
        rows = mixes.reshape(-1, mixes.shape[-1])
        all_pre, all_post, all_comb = [], [], []
        for controls in rows:
            pre = [
                torch.sigmoid(controls[j] * scale[0] + base[j]) + eps for j in range(hc)
            ]
            post = [
                2 * torch.sigmoid(controls[hc + j] * scale[1] + base[hc + j])
                for j in range(hc)
            ]
            comb = torch.empty(hc, hc)
            for source in range(hc):
                for destination in range(hc):
                    index = 2 * hc + source * hc + destination
                    comb[source, destination] = controls[index] * scale[2] + base[index]
            comb = torch.softmax(comb, -1) + eps
            for destination in range(hc):
                comb[:, destination] /= comb[:, destination].sum() + eps
            for _ in range(iterations - 1):
                for source in range(hc):
                    comb[source] /= comb[source].sum() + eps
                for destination in range(hc):
                    comb[:, destination] /= comb[:, destination].sum() + eps
            all_pre.append(torch.stack(pre))
            all_post.append(torch.stack(post))
            all_comb.append(comb)
        leading = mixes.shape[:-1]
        return (
            torch.stack(all_pre).view(*leading, hc),
            torch.stack(all_post).view(*leading, hc),
            torch.stack(all_comb).view(*leading, hc, hc),
        )

    def test_hc_split_sinkhorn_matches_scalar_iterations(self) -> None:
        from immer.runtimes.deepseek_v4.kernels import hc_split_sinkhorn

        torch.manual_seed(8)
        hc = 3
        mixes = torch.randn(1, 2, (2 + hc) * hc)
        scale = torch.tensor([0.7, -0.3, 1.1])
        base = torch.linspace(-0.2, 0.4, mixes.shape[-1])
        expected = self._scalar_hc_reference(mixes, scale, base, hc, 7, 1e-6)
        actual = hc_split_sinkhorn(mixes, scale, base, hc, 7, 1e-6)
        for got, want in zip(actual, expected):
            torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-6)
        self.assertTrue(
            torch.allclose(actual[2].sum(-2), torch.ones(1, 2, hc), atol=1e-5)
        )

    def test_hc_pre_post_and_head_match_dense_constructions(self) -> None:
        from immer.runtimes.deepseek_v4.kernels import (
            hc_head,
            hc_post,
            hc_pre,
        )

        torch.manual_seed(12)
        x = torch.randn(1, 2, 2, 3)
        fn = torch.randn(8, 6)
        scale = torch.tensor([0.4, 0.6, -0.2])
        base = torch.randn(8)
        reduced, post, comb = hc_pre(x, fn, scale, base, 2, 5)
        flat = x.flatten(2).float()
        inverse_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
        mixes = flat @ fn.T * inverse_rms
        expected_pre, expected_post, expected_comb = self._scalar_hc_reference(
            mixes, scale, base, 2, 5, 1e-6
        )
        expected_reduced = (expected_pre.unsqueeze(-1) * x).sum(-2)
        torch.testing.assert_close(reduced, expected_reduced, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(post, expected_post, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(comb, expected_comb, atol=2e-6, rtol=2e-6)

        branch = torch.randn(1, 2, 3)
        expanded = hc_post(branch, x, post, comb)
        expected_expanded = torch.empty_like(expanded)
        for sequence in range(2):
            for destination in range(2):
                expected_expanded[0, sequence, destination] = (
                    post[0, sequence, destination] * branch[0, sequence]
                )
                for source in range(2):
                    expected_expanded[0, sequence, destination] += (
                        comb[0, sequence, source, destination] * x[0, sequence, source]
                    )
        torch.testing.assert_close(expanded, expected_expanded)

        head_fn = torch.randn(2, 6)
        head_scale = torch.tensor([0.35])
        head_base = torch.tensor([-0.1, 0.2])
        actual_head = hc_head(x, head_fn, head_scale, head_base)
        head_mixes = flat @ head_fn.T * inverse_rms
        head_pre = torch.sigmoid(head_mixes * head_scale + head_base) + 1e-6
        expected_head = (head_pre.unsqueeze(-1) * x).sum(-2)
        torch.testing.assert_close(actual_head, expected_head)

    def test_invalid_sparse_index_fails_closed(self) -> None:
        from immer.runtimes.deepseek_v4.kernels import sparse_attention

        q = torch.zeros(1, 1, 1, 2)
        kv = torch.zeros(1, 2, 2)
        with self.assertRaisesRegex(IndexError, "outside"):
            sparse_attention(q, kv, torch.zeros(1), torch.tensor([[[2]]]))


if __name__ == "__main__":
    unittest.main()
