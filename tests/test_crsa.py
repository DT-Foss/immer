from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class CrsaTests(unittest.TestCase):
    def test_prefix_log_preserves_exact_causal_support(self) -> None:
        from immer.attention.crsa.operators import AttentionSpec, prefix_log

        torch.manual_seed(17)
        logits = torch.randn(3, 4, 32, 32, dtype=torch.float64)
        weights = prefix_log(logits, AttentionSpec(kind="prefix_log", alpha=2.0))
        future = torch.triu(torch.ones(32, 32, dtype=torch.bool), diagonal=1)
        self.assertEqual(float(weights[..., future].abs().sum()), 0.0)
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-12, rtol=0))

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
        from immer.attention.crsa.operators import AttentionSpec, geometric_prefix_log, prefix_log

        torch.manual_seed(19)
        logits = torch.randn(2, 4, 24, 24)
        spec = AttentionSpec(kind="geometric_prefix", alpha=1.0, usage_decay=1.0, diagonal_debit=3.0)
        got = geometric_prefix_log(logits, spec)
        expected = prefix_log(logits, AttentionSpec(kind="prefix_log", alpha=1.0, diagonal_debit=3.0))
        self.assertTrue(torch.equal(got, expected))

    def test_role_complete_free_heads_are_exact_causal_softmax(self) -> None:
        from immer.attention.crsa.operators import AttentionSpec, apply_attention, role_complete_attention

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
