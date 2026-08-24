from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class CrsaTests(unittest.TestCase):
    def test_streaming_prefix_log_is_public(self) -> None:
        from immer.attention.crsa import streaming_prefix_log
        from immer.attention.crsa.operators import (
            streaming_prefix_log as implementation,
        )

        self.assertIs(streaming_prefix_log, implementation)

    def test_prefix_log_preserves_exact_causal_support(self) -> None:
        from immer.attention.crsa.operators import AttentionSpec, prefix_log

        torch.manual_seed(17)
        logits = torch.randn(3, 4, 32, 32, dtype=torch.float64)
        weights = prefix_log(logits, AttentionSpec(kind="prefix_log", alpha=2.0))
        future = torch.triu(torch.ones(32, 32, dtype=torch.bool), diagonal=1)
        self.assertEqual(float(weights[..., future].abs().sum()), 0.0)
        self.assertTrue(
            torch.allclose(
                weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-12, rtol=0
            )
        )

    def test_prefix_log_has_zero_future_row_gradient(self) -> None:
        from immer.attention.crsa.operators import AttentionSpec, prefix_log

        torch.manual_seed(23)
        logits = torch.randn(1, 1, 12, 12, dtype=torch.float64, requires_grad=True)
        weights = prefix_log(logits, AttentionSpec(kind="prefix_log", alpha=1.0))
        row = 5
        values = torch.arange(12, dtype=torch.float64)
        loss = (weights[0, 0, row] * values).sum()
        grad = torch.autograd.grad(loss, logits)[0]
        self.assertEqual(float(grad[:, :, row + 1 :, :].abs().sum()), 0.0)

    def test_full_support_sinkhorn_negative_control_leaks_future_rows(self) -> None:
        from immer.attention.crsa.operators import leaky_masked_sinkhorn

        torch.manual_seed(23)
        logits = torch.randn(1, 1, 12, 12, dtype=torch.float64, requires_grad=True)
        weights = leaky_masked_sinkhorn(logits)
        row = 5
        values = torch.arange(12, dtype=torch.float64)
        loss = (weights[0, 0, row] * values).sum()
        grad = torch.autograd.grad(loss, logits)[0]
        self.assertGreater(float(grad[:, :, row + 1 :, :].abs().sum()), 0.0)

    def test_geometric_lambda_one_is_prefix(self) -> None:
        from immer.attention.crsa.operators import (
            AttentionSpec,
            geometric_prefix_log,
            prefix_log,
        )

        torch.manual_seed(19)
        logits = torch.randn(2, 4, 24, 24)
        spec = AttentionSpec(
            kind="geometric_prefix", alpha=1.0, usage_decay=1.0, diagonal_debit=3.0
        )
        got = geometric_prefix_log(logits, spec)
        expected = prefix_log(
            logits, AttentionSpec(kind="prefix_log", alpha=1.0, diagonal_debit=3.0)
        )
        self.assertTrue(torch.equal(got, expected))

    def test_streaming_prefix_log_matches_square_chunked_and_tokenwise(self) -> None:
        from immer.attention.crsa.operators import (
            AttentionSpec,
            prefix_log,
            streaming_prefix_log,
        )

        torch.manual_seed(71)
        logits = torch.randn(2, 4, 9, 9, dtype=torch.float64)
        allowed = torch.ones(9, 9, dtype=torch.bool).tril()
        spec = AttentionSpec(
            kind="prefix_log",
            alpha=1.0,
            diagonal_debit=3.0,
        )
        expected = prefix_log(logits, spec)
        whole, whole_usage = streaming_prefix_log(
            logits,
            spec,
            query_start=0,
            prior_log_usage=None,
            allowed=allowed,
        )
        self.assertTrue(torch.equal(whole, expected))

        prefix, usage = streaming_prefix_log(
            logits[:, :, :4, :4],
            spec,
            query_start=0,
            prior_log_usage=None,
            allowed=allowed[:4, :4],
        )
        suffix, split_usage = streaming_prefix_log(
            logits[:, :, 4:, :],
            spec,
            query_start=4,
            prior_log_usage=usage,
            allowed=allowed[4:, :],
        )
        split = torch.cat((torch.nn.functional.pad(prefix, (0, 5)), suffix), dim=2)
        torch.testing.assert_close(split, expected, rtol=1e-15, atol=1e-15)
        torch.testing.assert_close(split_usage, whole_usage, rtol=1e-15, atol=1e-15)

        rows = []
        usage = None
        for query_start in range(9):
            row, usage = streaming_prefix_log(
                logits[:, :, query_start : query_start + 1, : query_start + 1],
                spec,
                query_start=query_start,
                prior_log_usage=usage,
                allowed=torch.ones(1, query_start + 1, dtype=torch.bool),
            )
            rows.append(torch.nn.functional.pad(row, (0, 8 - query_start)))
        tokenwise = torch.cat(rows, dim=2)
        torch.testing.assert_close(tokenwise, expected, rtol=1e-15, atol=1e-15)
        torch.testing.assert_close(usage, whole_usage, rtol=1e-15, atol=1e-15)

    def test_streaming_prefix_mask_has_zero_mass_and_gradient(self) -> None:
        from immer.attention.crsa.operators import (
            AttentionSpec,
            streaming_prefix_log,
        )

        torch.manual_seed(73)
        logits = torch.randn(1, 2, 6, 6, dtype=torch.float64, requires_grad=True)
        causal = torch.ones(6, 6, dtype=torch.bool).tril()
        key_valid = torch.tensor([True, True, True, True, False, False])
        allowed = causal & key_valid
        weights, usage = streaming_prefix_log(
            logits,
            AttentionSpec(
                kind="prefix_log",
                alpha=1.0,
                diagonal_debit=3.0,
            ),
            query_start=0,
            prior_log_usage=None,
            allowed=allowed,
        )
        self.assertEqual(
            float(weights.detach().masked_select(~allowed).abs().sum()), 0.0
        )
        self.assertTrue(torch.isneginf(usage[..., 4:]).all())
        loss = (weights[:, :, 3] * torch.arange(6, dtype=torch.float64)).sum()
        gradient = torch.autograd.grad(loss, logits)[0]
        self.assertEqual(float(gradient[:, :, 4:].abs().sum()), 0.0)
        self.assertEqual(float(gradient[..., 4:].abs().sum()), 0.0)

    def test_role_complete_free_heads_are_exact_causal_softmax(self) -> None:
        from immer.attention.crsa.operators import (
            AttentionSpec,
            apply_attention,
            role_complete_attention,
        )

        torch.manual_seed(29)
        logits = torch.randn(2, 4, 20, 20, dtype=torch.float64)
        spec = AttentionSpec(
            kind="role_complete",
            local_heads=2,
            balanced_heads=1,
            free_heads=1,
            diagonal_debit=3.0,
            slope=0.08,
        )
        routed = role_complete_attention(logits, spec)
        free = apply_attention(logits[:, -1:], AttentionSpec(kind="softmax"))
        self.assertTrue(torch.equal(routed[:, -1:], free))


if __name__ == "__main__":
    unittest.main()
