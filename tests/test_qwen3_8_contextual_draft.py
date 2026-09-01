from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from immer.runtimes.qwen3_8.contextual_continuation import (
    ContextualContinuationBank,
    ContextualContinuationIdentity,
)
from immer.runtimes.qwen3_8.markov_draft import (
    FingerprintRollingK4DraftProvider,
)
from immer.runtimes.qwen3_8.hybrid_draft import Qwen38MarkovMtpDraftProvider


_DIGEST = "a" * 64


def _identity() -> ContextualContinuationIdentity:
    return ContextualContinuationIdentity(
        runtime_sha256=_DIGEST,
        model_sha256="b" * 64,
        q4_sha256="c" * 64,
        tokenizer_sha256="d" * 64,
        hidden_width=8,
    )


class ContextualContinuationDraftTests(unittest.TestCase):
    def test_confirmed_hidden_tail_reopens_as_target_verified_k4_option(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "context.json"
            bank = ContextualContinuationBank(state, _identity(), max_cells=16)
            prompt = (1, 2, 3)
            hidden = torch.tensor(
                [[[0.25, -0.5, 0.75, 1.0, -1.25, 1.5, -1.75, 2.0]]],
                dtype=torch.float32,
            )

            first = FingerprintRollingK4DraftProvider(
                vocab_size=256,
                proposal_width=3,
                contextual_continuation_bank=bank,
            )
            first.begin_request(prompt)
            first.propose_round_state(prompt, 10, hidden)
            first.observe_verification(0, 1)
            first.reconcile_prefix((*prompt, 10))
            first.observe_final((*prompt, 10, 20, 21, 22, 23))
            first.close()

            reopened = ContextualContinuationBank(
                state,
                _identity(),
                max_cells=16,
            )
            second_prompt = (4, 5, 6)
            second = FingerprintRollingK4DraftProvider(
                vocab_size=256,
                proposal_width=3,
                contextual_continuation_bank=reopened,
            )
            second.begin_request(second_prompt)
            proposal = second.propose_round_state(second_prompt, 10, hidden.clone())

            self.assertEqual(proposal.token_ids, (20, 21, 22))
            self.assertEqual(proposal.phrase_source, "crystal")
            self.assertEqual(proposal.phrase_width, 3)
            self.assertEqual(proposal.recommended_window, 4)
            self.assertTrue(all(value > 0.99 for value in proposal.token_confidences))

            second.observe_verification(3, 3)
            second.reconcile_prefix((*second_prompt, 10, 20, 21, 22))
            second.observe_final((*second_prompt, 10, 20, 21, 22, 23))
            metrics = second.metrics()
            second.close()

            self.assertEqual(metrics.crystal_queries, 1)
            self.assertEqual(metrics.crystal_query_hits, 1)
            self.assertEqual(metrics.crystal_option_calls, 1)
            self.assertEqual(metrics.crystal_proposed_tokens, 3)
            self.assertEqual(metrics.crystal_verified_tokens, 3)
            self.assertEqual(metrics.crystal_accepted_tokens, 3)
            self.assertEqual(metrics.crystal_mismatches, 0)
            self.assertEqual(metrics.crystal_captures, 1)
            self.assertEqual(metrics.crystal_bank_cells, 1)
            self.assertEqual(metrics.crystal_bank_support, 2)

    def test_wrong_crystal_tail_only_trains_a_miss(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "context.json"
            bank = ContextualContinuationBank(state, _identity(), max_cells=16)
            hidden = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8) + 1
            key = bank.project(hidden, 10)
            bank.settle(
                captures=(
                    bank.make_capture(hidden, 10, 3, (20, 21, 22)),
                )
            )
            provider = FingerprintRollingK4DraftProvider(
                vocab_size=256,
                proposal_width=3,
                contextual_continuation_bank=bank,
            )
            prompt = (7, 8, 9)
            provider.begin_request(prompt)
            proposal = provider.propose_round_state(prompt, 10, hidden)
            self.assertEqual(proposal.token_ids, (20, 21, 22))

            provider.observe_verification(0, 1)
            provider.reconcile_prefix((*prompt, 10))
            provider.observe_final((*prompt, 10, 99, 98))
            metrics = provider.metrics()
            provider.close()

            candidate = bank.query_key(key)[0]
            self.assertEqual(candidate.position_verified[0], 1)
            self.assertEqual(candidate.position_hits[0], 0)
            self.assertEqual(metrics.crystal_accepted_tokens, 0)
            self.assertEqual(metrics.crystal_mismatches, 1)

    def test_hybrid_gives_hidden_crystal_first_refusal_without_loading_mtp(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = ContextualContinuationBank(
                Path(temporary) / "context.json",
                _identity(),
                max_cells=16,
            )
            boundary = torch.tensor(
                [[[1.0, 0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 1.5]]],
                dtype=torch.float32,
            )
            bank.settle(
                captures=(
                    bank.make_capture(boundary, 10, 3, (20, 21, 22)),
                )
            )
            markov = FingerprintRollingK4DraftProvider(
                vocab_size=256,
                proposal_width=3,
                contextual_continuation_bank=bank,
            )
            mtp_factory = Mock(side_effect=AssertionError("MTP must stay cold"))
            hybrid = Qwen38MarkovMtpDraftProvider(markov, mtp_factory)
            prompt = (1, 2, 3)
            prompt_hidden = boundary.expand(1, len(prompt), 8).clone()
            hybrid.begin_request_state(prompt, prompt_hidden)

            proposal = hybrid.propose_round_state(prompt, 10, boundary.clone())

            self.assertEqual(proposal.token_ids, (20, 21, 22))
            self.assertEqual(proposal.phrase_source, "crystal")
            self.assertEqual(hybrid.metrics().selected_provider, "markov")
            mtp_factory.assert_not_called()
            hybrid.close()

    def test_crystal_settlement_failure_is_visible_without_losing_qwen_result(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = ContextualContinuationBank(
                Path(temporary) / "context.json",
                _identity(),
                max_cells=16,
            )
            provider = FingerprintRollingK4DraftProvider(
                vocab_size=256,
                proposal_width=3,
                contextual_continuation_bank=bank,
            )
            prompt = (1, 2, 3)
            hidden = torch.ones((1, 1, 8), dtype=torch.float32)
            provider.begin_request(prompt)
            provider.propose_round_state(prompt, 10, hidden)
            provider.observe_verification(0, 1)
            provider.reconcile_prefix((*prompt, 10))

            with patch.object(bank, "settle", side_effect=OSError("disk full")):
                provider.observe_final((*prompt, 10, 20, 21))

            metrics = provider.metrics()
            provider.close()
            self.assertEqual(metrics.crystal_failures, 1)
            self.assertEqual(metrics.crystal_captures, 0)
            self.assertEqual(metrics.crystal_bank_cells, 0)


if __name__ == "__main__":
    unittest.main()
