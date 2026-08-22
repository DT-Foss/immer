from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class CrsaRouterTests(unittest.TestCase):
    def test_fixed_context_shape_and_exact_causal_support(self) -> None:
        from immer.attention.router import RoleCompleteContext

        torch.manual_seed(41)
        context = RoleCompleteContext(d_model=128)
        states = torch.randn(3, 9, 128, dtype=torch.float64)
        weights = context.attention_weights(states)
        routed = context(states)
        features = context.route_features(states)

        self.assertEqual(weights.shape, (3, 4, 9, 9))
        self.assertEqual(routed.shape, states.shape)
        self.assertEqual(features.shape, (3, 256))
        future = torch.triu(torch.ones(9, 9, dtype=torch.bool), diagonal=1)
        self.assertEqual(float(weights[..., future].abs().sum()), 0.0)
        self.assertTrue(
            torch.allclose(
                weights.sum(-1),
                torch.ones_like(weights.sum(-1)),
                atol=1e-12,
                rtol=0.0,
            )
        )

    def test_free_head_is_bit_exact_causal_softmax(self) -> None:
        from immer.attention.crsa.operators import AttentionSpec, apply_attention
        from immer.attention.router import RoleCompleteContext

        torch.manual_seed(43)
        context = RoleCompleteContext(d_model=128)
        states = torch.randn(2, 11, 128, dtype=torch.float64)
        logits = context.attention_logits(states)
        routed = context.attention_weights(states)
        reference = apply_attention(logits[:, -1:], AttentionSpec(kind="softmax"))
        self.assertTrue(torch.equal(routed[:, -1:], reference))

    def test_ridge_decision_and_json_state_are_deterministic(self) -> None:
        from immer.attention.router import RidgeRouteHead

        features = torch.tensor(
            [
                [-2.0, -1.0],
                [-1.0, -2.0],
                [-1.5, -1.5],
                [1.0, 2.0],
                [2.0, 1.0],
                [1.5, 1.5],
            ],
            dtype=torch.float64,
        )
        labels = torch.tensor([0, 0, 0, 1, 1, 1])
        first = RidgeRouteHead.fit(features, labels, ridge=10.0, metadata={"seed": 7})
        second = RidgeRouteHead.fit(features, labels, ridge=10.0, metadata={"seed": 7})

        self.assertTrue(torch.equal(first.coefficient, second.coefficient))
        self.assertTrue(torch.equal(first.predict(features), labels))
        self.assertEqual(first.digest(), second.digest())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "router.json"
            first.save(path)
            restored = RidgeRouteHead.load(path)
        self.assertTrue(torch.equal(first.scores(features), restored.scores(features)))
        self.assertEqual(first.digest(), restored.digest())
        self.assertEqual(
            [item.label for item in restored.decide(features)],
            ["text"] * 3 + ["arithmetic"] * 3,
        )

    def test_measured_default_head_is_self_describing_and_loadable(self) -> None:
        from immer.attention.router import FrozenA1CrsaRouter

        router = FrozenA1CrsaRouter.load()
        measurement = router.head.metadata["measurement"]
        self.assertEqual(
            measurement["verdict"],
            "POSITIVE_CONTEXT_ROUTER__CRSA_NOT_UNIQUE_VS_SOFTMAX",
        )
        self.assertEqual(
            measurement["crsa_specific_advantage_over_softmax"],
            "NOT_SHOWN",
        )
        self.assertEqual(router.head.metadata["arithmetic_overlap_removed"], 48)
        self.assertTrue(
            measurement["positive_router_criteria"]["stateless_deployment_182_of_182"]
        )
        self.assertEqual(
            measurement["interface_checks"]["stateless_deployment_feature_max_abs"],
            0.0,
        )
        self.assertEqual(router.head.feature_dim, 256)

    def test_context_capture_accepts_stateless_and_streaming_a1_scans(self) -> None:
        from torch import nn

        from immer.attention.router import capture_a1_scan_states

        class StatelessScan(nn.Module):
            def forward(self, hidden):
                return hidden.cumsum(dim=1)

        class StreamingScan(nn.Module):
            def forward(self, hidden, state=None):
                sequence = hidden.cumsum(dim=1)
                return sequence, sequence[:, -1]

        class Block(nn.Module):
            def __init__(self, scan):
                super().__init__()
                self.scan = scan
                self.ln1 = nn.Identity()
                self.ln2 = nn.Identity()
                self.ffn = nn.Identity()

        class Host(nn.Module):
            def __init__(self, scan):
                super().__init__()
                self.embed = nn.Embedding(16, 8)
                self.layers = nn.ModuleList([Block(scan)])
                for parameter in self.parameters():
                    parameter.requires_grad_(False)

        ids = torch.tensor([[1, 2, 3, 4]])
        torch.manual_seed(47)
        stateless = Host(StatelessScan())
        torch.manual_seed(47)
        streaming = Host(StreamingScan())
        expected = stateless.embed(ids).cumsum(dim=1)
        got_stateless = capture_a1_scan_states(stateless, ids)
        got_streaming = capture_a1_scan_states(streaming, ids)
        self.assertTrue(torch.equal(got_stateless, expected))
        self.assertTrue(torch.equal(got_streaming, expected))

    def test_noncanonical_role_program_is_rejected(self) -> None:
        from immer.attention.router import RoleCompleteContext

        with self.assertRaisesRegex(ValueError, "exactly 2 Local"):
            RoleCompleteContext(local_heads=3, balanced_heads=0, free_heads=1)


if __name__ == "__main__":
    unittest.main()
