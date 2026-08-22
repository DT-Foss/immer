from __future__ import annotations

import json
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class DeepSeekV4CrsaGraftTests(unittest.TestCase):
    def test_off_and_zero_alpha_are_true_object_identity(self) -> None:
        from immer.runtimes.deepseek_v4.graft import DeepSeekV4CrsaGraft

        hidden = torch.tensor([[[float("nan"), 1.0, 2.0]]])
        off, off_evidence = DeepSeekV4CrsaGraft(mode="off", alpha=1.0)(
            hidden, return_evidence=True
        )
        zero, zero_evidence = DeepSeekV4CrsaGraft(mode="crsa", alpha=0.0)(
            hidden, return_evidence=True
        )

        self.assertIs(off, hidden)
        self.assertIs(zero, hidden)
        self.assertTrue(off_evidence.identity)
        self.assertTrue(zero_evidence.identity)
        self.assertEqual(off_evidence.estimated_peak_bytes, 0)
        self.assertEqual(zero_evidence.operator, "identity(alpha=0)")

    def test_plain_history_matches_repository_role_complete_context(self) -> None:
        from immer.attention.router import RoleCompleteContext
        from immer.runtimes.deepseek_v4.graft import DeepSeekV4CrsaGraft

        torch.manual_seed(73)
        hidden = torch.randn(2, 9, 16, dtype=torch.float64)
        alpha = 0.125
        expected_context = RoleCompleteContext(d_model=16)(hidden)
        got, evidence = DeepSeekV4CrsaGraft(mode="crsa", alpha=alpha)(
            hidden, return_evidence=True
        )

        self.assertTrue(torch.equal(got, hidden + alpha * expected_context))
        self.assertEqual(evidence.operator, "role_complete(2local+1balanced+1free)")
        self.assertEqual(evidence.learned_parameters, 0)
        self.assertFalse(evidence.uses_a1_ridge)
        self.assertEqual(
            list(DeepSeekV4CrsaGraft(mode="crsa", alpha=alpha).parameters()), []
        )
        json.dumps(evidence.to_dict())

    def test_all_live_modes_are_strictly_causal_under_future_perturbation(self) -> None:
        from immer.runtimes.deepseek_v4.graft import DeepSeekV4CrsaGraft

        torch.manual_seed(79)
        hidden = torch.randn(2, 10, 32, dtype=torch.float64)
        cutoff = 5
        changed = hidden.clone()
        changed[:, cutoff + 1 :] = torch.randn_like(changed[:, cutoff + 1 :]) * 1000.0

        for mode in ("crsa", "softmax", "shuffle"):
            with self.subTest(mode=mode):
                graft = DeepSeekV4CrsaGraft(mode=mode, alpha=0.2, shuffle_seed=91)
                baseline, evidence = graft(hidden, return_evidence=True)
                perturbed = graft(changed)
                self.assertTrue(
                    torch.equal(baseline[:, : cutoff + 1], perturbed[:, : cutoff + 1])
                )
                self.assertTrue(evidence.strict_causal)
                self.assertEqual(evidence.future_weight_max_abs, 0.0)
                self.assertLess(evidence.row_sum_max_error, 1e-12)

    def test_hyper_connection_histories_remain_independent_and_shape_stable(
        self,
    ) -> None:
        from immer.runtimes.deepseek_v4.graft import DeepSeekV4CrsaGraft

        torch.manual_seed(83)
        hidden = torch.randn(2, 7, 4, 16)
        changed = hidden.clone()
        changed[:, :, 2] += 500.0
        graft = DeepSeekV4CrsaGraft(mode="softmax", alpha=0.3)
        baseline, evidence = graft(hidden, return_evidence=True)
        perturbed = graft(changed)

        self.assertEqual(baseline.shape, hidden.shape)
        self.assertTrue(torch.equal(baseline[:, :, :2], perturbed[:, :, :2]))
        self.assertTrue(torch.equal(baseline[:, :, 3:], perturbed[:, :, 3:]))
        self.assertEqual(evidence.independent_histories, 8)
        weights = graft.attention_weights(hidden)
        self.assertEqual(weights.shape, (2, 4, 4, 7, 7))

    def test_shuffle_is_reproducible_causal_row_marginal_placebo(self) -> None:
        from immer.runtimes.deepseek_v4.graft import DeepSeekV4CrsaGraft

        torch.manual_seed(89)
        hidden = torch.randn(1, 12, 24, dtype=torch.float64)
        graft = DeepSeekV4CrsaGraft(mode="crsa", alpha=0.1, shuffle_seed=101)
        crsa = graft.attention_weights(hidden, mode="crsa")
        shuffled_a = graft.attention_weights(hidden, mode="shuffle")
        shuffled_b = graft.attention_weights(hidden, mode="shuffle")

        self.assertTrue(torch.equal(shuffled_a, shuffled_b))
        self.assertFalse(torch.equal(crsa[..., 5:, :], shuffled_a[..., 5:, :]))
        future = torch.ones(12, 12, dtype=torch.bool).triu(1)
        self.assertEqual(float(shuffled_a[..., future].abs().sum()), 0.0)
        for query in range(12):
            expected = crsa[..., query, : query + 1].sort(dim=-1).values
            got = shuffled_a[..., query, : query + 1].sort(dim=-1).values
            self.assertTrue(torch.equal(expected, got))

    def test_history_guard_reports_uncapped_estimate(self) -> None:
        from immer.runtimes.deepseek_v4.graft import DeepSeekV4CrsaGraft

        hidden = torch.randn(1, 9, 16)
        graft = DeepSeekV4CrsaGraft(mode="crsa", alpha=0.1, max_history=8)
        with self.assertRaisesRegex(ValueError, "estimated peak"):
            graft(hidden)
        estimate = graft.estimate_peak_bytes(tuple(hidden.shape), element_size=4)
        matrix = 1 * 4 * 9 * 9 * 4
        history = 1 * 9 * 16 * 4
        self.assertEqual(estimate, 3 * matrix + 3 * history)

    @unittest.skipUnless(
        torch is not None and torch.backends.mps.is_available(), "MPS unavailable"
    )
    def test_live_graft_and_evidence_execute_on_mps_bfloat16(self) -> None:
        from immer.runtimes.deepseek_v4.graft import DeepSeekV4CrsaGraft

        hidden = torch.randn(1, 2, 4, 128, device="mps", dtype=torch.bfloat16)
        output, evidence = DeepSeekV4CrsaGraft(mode="crsa", alpha=0.05)(
            hidden, return_evidence=True
        )
        self.assertEqual(output.device.type, "mps")
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(evidence.strict_causal)
        self.assertEqual(len(evidence.head_entropy), 4)


if __name__ == "__main__":
    unittest.main()
