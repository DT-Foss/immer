from __future__ import annotations

from dataclasses import replace
import unittest

from immer.runtimes.qwen3_8.draft_protocol import (
    RollingDraftProposal,
    RoundWindowPolicy,
)


class RollingDraftProtocolTests(unittest.TestCase):
    def _proposal(
        self,
        *,
        ceiling: int,
        confidence: float,
        phrase_support: int = 0,
        phrase_confidence: float = 0.0,
        phrase_width: int = 0,
    ) -> RollingDraftProposal:
        width = ceiling - 1
        return RollingDraftProposal.build(
            tuple(range(width)),
            (confidence,) * width,
            (0.0,) * width,
            request_window_ceiling=ceiling,
            provider_abi="test-provider/v1",
            phrase_source=None if phrase_width == 0 else "global",
            phrase_support=phrase_support,
            phrase_confidence=phrase_confidence,
            phrase_width=phrase_width,
        )

    def test_low_confidence_abstains_to_k1_under_k16_ceiling(self) -> None:
        proposal = self._proposal(ceiling=16, confidence=0.10)

        self.assertEqual(proposal.recommended_window, 1)
        self.assertEqual(
            proposal.select_window(
                request_window_ceiling=16,
                remaining_tokens=16,
            ).chosen_window,
            1,
        )

    def test_high_confidence_prefers_k16(self) -> None:
        proposal = self._proposal(ceiling=16, confidence=0.99)

        self.assertEqual(proposal.recommended_window, 16)
        utilities = [row.utility for row in proposal.horizons]
        self.assertEqual(utilities, sorted(utilities))

    def test_compute_bound_row_cost_requires_near_certain_speculation(self) -> None:
        marginal = self._proposal(ceiling=4, confidence=0.50)
        certain = self._proposal(ceiling=4, confidence=0.99)
        costs = {1: 1.0, 4: 3.7}

        marginal_policy = marginal.select_window(
            request_window_ceiling=4,
            remaining_tokens=4,
            window_work_costs=costs,
        )
        certain_policy = certain.select_window(
            request_window_ceiling=4,
            remaining_tokens=4,
            window_work_costs=costs,
        )

        self.assertEqual(marginal_policy.chosen_window, 1)
        self.assertEqual(certain_policy.chosen_window, 4)
        self.assertEqual(
            {row.window: row.work_proxy for row in marginal_policy.horizons},
            costs,
        )

    def test_repeated_long_phrase_can_open_k8(self) -> None:
        proposal = self._proposal(
            ceiling=8,
            confidence=0.10,
            phrase_support=3,
            phrase_confidence=1.0,
            phrase_width=7,
        )

        self.assertEqual(proposal.recommended_window, 8)
        self.assertEqual(proposal.horizons[-1].phrase_covered_tokens, 7)
        policy = proposal.select_window(
            request_window_ceiling=8,
            remaining_tokens=8,
        )
        self.assertTrue(policy.matches_provider_tokens(proposal.token_ids))
        self.assertFalse(
            policy.matches_provider_tokens((*proposal.token_ids[:-1], 999))
        )

    def test_remaining_budget_restricts_the_round_without_new_proposal(self) -> None:
        proposal = self._proposal(ceiling=16, confidence=0.99)

        k8 = proposal.select_window(
            request_window_ceiling=16,
            remaining_tokens=10,
        )
        k3 = proposal.select_window(
            request_window_ceiling=16,
            remaining_tokens=3,
        )

        self.assertEqual(k8.chosen_window, 8)
        self.assertEqual(k8.eligible_windows, (1, 4, 8, 16))
        self.assertEqual(k3.chosen_window, 4)
        self.assertEqual(k3.selector, "markov-prefix-utility/v2")

    def test_policy_rejects_a_window_outside_its_eligible_set(self) -> None:
        proposal = self._proposal(ceiling=8, confidence=0.99)
        policy = proposal.select_window(
            request_window_ceiling=8,
            remaining_tokens=8,
        )

        with self.assertRaises(ValueError):
            replace(policy, chosen_window=16)
        self.assertIsInstance(policy, RoundWindowPolicy)


if __name__ == "__main__":
    unittest.main()
