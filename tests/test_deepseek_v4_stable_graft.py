from __future__ import annotations

import json
import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch not installed")
class DeepSeekV4StableCrsaGraftTests(unittest.TestCase):
    def test_off_and_zero_alpha_are_exact_object_identity(self) -> None:
        from immer.runtimes.deepseek_v4.stable_graft import (
            DeepSeekV4StableCrsaGraft,
            STABLE_GRAFT_POLICY,
        )

        hidden = torch.tensor([[[float("nan"), 1.0, 2.0]]])
        off, off_evidence = DeepSeekV4StableCrsaGraft(mode="off", alpha=1.0)(
            hidden, return_evidence=True
        )
        zero, zero_evidence = DeepSeekV4StableCrsaGraft(mode="crsa", alpha=0.0)(
            hidden, return_evidence=True
        )

        self.assertIs(off, hidden)
        self.assertIs(zero, hidden)
        self.assertEqual(off_evidence.policy, STABLE_GRAFT_POLICY)
        self.assertTrue(zero_evidence.identity)
        json.dumps(off_evidence.to_dict())

    def test_per_token_head_rms_is_preserved_under_extreme_prefix_norm(self) -> None:
        from immer.runtimes.deepseek_v4.stable_graft import (
            DeepSeekV4StableCrsaGraft,
        )

        torch.manual_seed(701)
        hidden = torch.randn(2, 11, 4, 32, dtype=torch.float64)
        hidden[:, 0] *= 2000.0
        output, evidence = DeepSeekV4StableCrsaGraft(
            mode="crsa", alpha=0.1
        )(hidden, return_evidence=True)

        expected = hidden.reshape(2, 11, 4, 4, 8).float().square().mean(-1).sqrt()
        got = output.reshape(2, 11, 4, 4, 8).float().square().mean(-1).sqrt()
        self.assertTrue(torch.allclose(got, expected, rtol=2e-6, atol=2e-6))
        self.assertLess(evidence.max_head_rms_relative_error, 2e-6)
        self.assertLess(float(output[:, -1].norm() / hidden[:, -1].norm()), 1.01)
        self.assertGreater(float((output - hidden).abs().max()), 0.0)

    def test_attention_weights_are_invariant_to_positive_headwise_rescaling(
        self,
    ) -> None:
        from immer.runtimes.deepseek_v4.stable_graft import (
            DeepSeekV4StableCrsaGraft,
        )

        torch.manual_seed(709)
        hidden = torch.randn(1, 9, 32, dtype=torch.float64)
        scales = torch.exp(torch.randn(1, 9, 4, 1, dtype=torch.float64) * 3.0)
        changed = (hidden.reshape(1, 9, 4, 8) * scales).reshape_as(hidden)
        graft = DeepSeekV4StableCrsaGraft(mode="crsa", alpha=0.1)

        original = graft.attention_weights(hidden)
        rescaled = graft.attention_weights(changed)

        self.assertTrue(torch.allclose(original, rescaled, rtol=2e-6, atol=2e-6))

    def test_all_live_modes_remain_strictly_causal(self) -> None:
        from immer.runtimes.deepseek_v4.stable_graft import (
            DeepSeekV4StableCrsaGraft,
        )

        torch.manual_seed(719)
        hidden = torch.randn(2, 10, 32, dtype=torch.float64)
        changed = hidden.clone()
        changed[:, 6:] = torch.randn_like(changed[:, 6:]) * 1000.0
        for mode in ("crsa", "softmax", "shuffle"):
            with self.subTest(mode=mode):
                graft = DeepSeekV4StableCrsaGraft(
                    mode=mode, alpha=0.1, shuffle_seed=727
                )
                baseline, evidence = graft(hidden, return_evidence=True)
                perturbed = graft(changed)
                self.assertTrue(torch.equal(baseline[:, :6], perturbed[:, :6]))
                self.assertTrue(evidence.strict_causal)
                self.assertEqual(evidence.future_weight_max_abs, 0.0)
                self.assertLess(evidence.row_sum_max_error, 2e-6)

    def test_hyper_connections_stay_independent(self) -> None:
        from immer.runtimes.deepseek_v4.stable_graft import (
            DeepSeekV4StableCrsaGraft,
        )

        torch.manual_seed(733)
        hidden = torch.randn(2, 7, 4, 16)
        changed = hidden.clone()
        changed[:, :, 2] += 500.0
        graft = DeepSeekV4StableCrsaGraft(mode="softmax", alpha=0.1)

        baseline = graft(hidden)
        perturbed = graft(changed)

        self.assertTrue(torch.equal(baseline[:, :, :2], perturbed[:, :, :2]))
        self.assertTrue(torch.equal(baseline[:, :, 3:], perturbed[:, :, 3:]))

    def test_alpha_and_history_bounds_fail_closed(self) -> None:
        from immer.runtimes.deepseek_v4.stable_graft import (
            DeepSeekV4StableCrsaGraft,
        )

        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            DeepSeekV4StableCrsaGraft(mode="crsa", alpha=1.01)
        graft = DeepSeekV4StableCrsaGraft(
            mode="crsa", alpha=0.1, max_history=4
        )
        with self.assertRaisesRegex(ValueError, "estimated peak"):
            graft(torch.randn(1, 5, 16))


if __name__ == "__main__":
    unittest.main()
