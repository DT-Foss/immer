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
from immer.runtimes.qwen3_8.layer_contextual_continuation import (
    LayerContextualContinuationBank,
    LayerContextualContinuationIdentity,
    LayerContextualContinuationTransaction,
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


def _layer_identity(
    layers: tuple[int, ...] = (1,),
) -> LayerContextualContinuationIdentity:
    return LayerContextualContinuationIdentity(
        runtime_sha256="1" * 64,
        model_sha256="2" * 64,
        q4_sha256="3" * 64,
        tokenizer_sha256="4" * 64,
        hidden_dim=8,
        layers=layers,
        projection_seed=17,
    )


class _LayerTransactions:
    def __init__(
        self,
        current: LayerContextualContinuationTransaction | None,
        transactions: tuple[LayerContextualContinuationTransaction, ...],
    ) -> None:
        self.current_transaction = current
        self.transactions = transactions

    def current(self) -> LayerContextualContinuationTransaction | None:
        return self.current_transaction

    def since(
        self,
        boundary: int,
    ) -> tuple[LayerContextualContinuationTransaction, ...]:
        return tuple(
            transaction
            for transaction in self.transactions
            if transaction.boundary_index > boundary
        )


def _layer_transaction(
    bank: LayerContextualContinuationBank,
    *,
    boundary: int,
    known_token: int,
    values: dict[int, float] | None = None,
) -> LayerContextualContinuationTransaction:
    selected = values or {layer: 1.0 for layer in bank.identity.layers}
    return bank.capture_boundaries(
        {
            layer: torch.full((1, 1, 8), selected[layer], dtype=torch.float32)
            for layer in bank.identity.layers
        },
        known_token,
        boundary,
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


class LayerContextualContinuationDraftTests(unittest.TestCase):
    def _provider(
        self,
        bank: LayerContextualContinuationBank,
        source: _LayerTransactions,
        *,
        contextual_bank: ContextualContinuationBank | None = None,
    ) -> FingerprintRollingK4DraftProvider:
        return FingerprintRollingK4DraftProvider(
            vocab_size=256,
            proposal_width=3,
            contextual_continuation_bank=contextual_bank,
            layer_contextual_continuation_bank=bank,
            layer_contextual_current_transaction=source.current,
            layer_contextual_transactions_since=source.since,
        )

    def test_layer_hit_strips_known_token_and_settles_only_at_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = LayerContextualContinuationBank(
                Path(temporary) / "layers.json",
                _layer_identity(),
                max_cells=16,
            )
            seed = _layer_transaction(bank, boundary=0, known_token=3)
            bank.settle_verified_prefix(seed, (10, 20, 21, 22))
            prompt = (1, 2, 3)
            current = _layer_transaction(bank, boundary=2, known_token=3)
            source = _LayerTransactions(current, (current,))
            provider = self._provider(bank, source)
            provider.begin_request(prompt)

            proposal = provider.propose_round_state(
                prompt,
                10,
                torch.ones((1, 1, 8)),
            )

            self.assertEqual(proposal.token_ids, (20, 21, 22))
            self.assertEqual(proposal.phrase_source, "crystal")
            self.assertEqual(bank.metrics().settlements, 1)
            provider.observe_verification(3, 3)
            provider.reconcile_prefix((*prompt, 10, 20, 21, 22))
            self.assertEqual(bank.metrics().settlements, 1)
            provider.observe_final((*prompt, 10, 20, 21, 22))
            metrics = provider.metrics()

            self.assertEqual(metrics.crystal_queries, 1)
            self.assertEqual(metrics.crystal_proposed_tokens, 3)
            self.assertEqual(metrics.crystal_verified_tokens, 3)
            self.assertEqual(metrics.crystal_accepted_tokens, 3)
            self.assertEqual(metrics.crystal_captures, 1)
            layer = metrics.layer_context_crystal
            assert layer is not None
            self.assertEqual(layer["crystal_proposed_tokens"], 3)
            self.assertEqual(layer["last_query"][0]["target_tail"], [10, 20, 21, 22])
            self.assertEqual(layer["last_query"][0]["planner_token_ids"], [20, 21, 22])
            provider.close()

    def test_layer_and_token_buckets_are_isolated_and_misses_stay_cold(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = LayerContextualContinuationBank(
                Path(temporary) / "layers.json",
                _layer_identity((1, 3)),
                max_cells=32,
            )
            first = _layer_transaction(
                bank,
                boundary=0,
                known_token=3,
                values={1: 1.0, 3: 1.0},
            )
            second = _layer_transaction(
                bank,
                boundary=1,
                known_token=3,
                values={1: -1.0, 3: -1.0},
            )
            bank.settle_verified_prefix(first, (10, 20, 21))
            bank.settle_verified_prefix(second, (10, 30, 31))
            current = _layer_transaction(
                bank,
                boundary=2,
                known_token=3,
                values={1: 1.0, 3: -1.0},
            )
            provider = self._provider(
                bank,
                _LayerTransactions(current, (current,)),
            )
            provider.begin_request((1, 2, 3))
            proposal = provider.propose_round_state(
                (1, 2, 3), 10, torch.ones((1, 1, 8))
            )
            evidence = provider.metrics().layer_context_crystal
            assert evidence is not None
            self.assertEqual(proposal.token_ids[:2], (20, 21))
            self.assertEqual(
                {row["layer"] for row in evidence["last_query"]},
                {1, 3},
            )
            provider.close()

            miss = _layer_transaction(bank, boundary=2, known_token=4)
            miss_provider = self._provider(
                bank,
                _LayerTransactions(miss, (miss,)),
            )
            miss_provider.begin_request((1, 2, 4))
            miss_provider.propose_round_state(
                (1, 2, 4), 10, torch.ones((1, 1, 8))
            )
            miss_metrics = miss_provider.metrics().layer_context_crystal
            assert miss_metrics is not None
            self.assertEqual(miss_metrics["crystal_queries"], 2)
            self.assertEqual(miss_metrics["crystal_query_hits"], 0)
            self.assertEqual(miss_metrics["crystal_proposed_tokens"], 0)
            miss_provider.close()

    def test_final_captures_unqueried_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = LayerContextualContinuationBank(
                Path(temporary) / "layers.json",
                _layer_identity(),
                max_cells=16,
            )
            prompt_boundary = _layer_transaction(bank, boundary=2, known_token=3)
            generated_boundary = _layer_transaction(
                bank,
                boundary=3,
                known_token=10,
            )
            source = _LayerTransactions(
                prompt_boundary,
                (prompt_boundary, generated_boundary),
            )
            provider = self._provider(bank, source)
            provider.begin_request((1, 2, 3))
            provider.observe_final((1, 2, 3, 10, 20))

            self.assertEqual(bank.metrics().settlements, 2)
            layer = provider.metrics().layer_context_crystal
            assert layer is not None
            self.assertEqual(layer["crystal_queries"], 0)
            self.assertEqual(layer["crystal_captures"], 2)
            provider.close()

    def test_feedback_is_bound_to_the_exact_queried_transaction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = LayerContextualContinuationBank(
                Path(temporary) / "layers.json",
                _layer_identity(),
                max_cells=16,
            )
            seed = _layer_transaction(bank, boundary=0, known_token=3)
            bank.settle_verified_prefix(seed, (10, 20, 21, 22))
            queried = _layer_transaction(bank, boundary=2, known_token=3)
            settled = _layer_transaction(bank, boundary=2, known_token=3)
            source = _LayerTransactions(queried, (settled,))
            provider = self._provider(bank, source)
            provider.begin_request((1, 2, 3))
            provider.propose_round_state(
                (1, 2, 3), 10, torch.ones((1, 1, 8))
            )
            provider.observe_verification(0, 1)
            provider.reconcile_prefix((1, 2, 3, 10))
            provider.observe_final((1, 2, 3, 10, 99))

            candidate = bank.query_options(queried)[0]
            self.assertEqual(candidate.position_verified, (0, 0, 0, 0))
            layer = provider.metrics().layer_context_crystal
            assert layer is not None
            self.assertEqual(layer["crystal_verified_tokens"], 1)
            provider.close()

    def test_abort_and_auxiliary_source_failure_never_settle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = LayerContextualContinuationBank(
                Path(temporary) / "layers.json",
                _layer_identity(),
                max_cells=16,
            )
            current = _layer_transaction(bank, boundary=2, known_token=3)
            source = _LayerTransactions(current, (current,))
            aborted = self._provider(bank, source)
            aborted.begin_request((1, 2, 3))
            aborted.propose_round_state(
                (1, 2, 3), 10, torch.ones((1, 1, 8))
            )
            aborted.close()
            self.assertEqual(bank.metrics().settlements, 0)

            failed_source = _LayerTransactions(None, (current,))
            failed = self._provider(bank, failed_source)
            legacy = FingerprintRollingK4DraftProvider(
                vocab_size=256,
                proposal_width=3,
            )
            failed.begin_request((1, 2, 3))
            legacy.begin_request((1, 2, 3))
            hidden = torch.ones((1, 1, 8))
            self.assertEqual(
                failed.propose_round_state((1, 2, 3), 10, hidden),
                legacy.propose_round_state((1, 2, 3), 10, hidden),
            )
            failed.observe_verification(0, 1)
            legacy.observe_verification(0, 1)
            failed.reconcile_prefix((1, 2, 3, 10))
            legacy.reconcile_prefix((1, 2, 3, 10))
            failed.observe_final((1, 2, 3, 10, 11))
            legacy.observe_final((1, 2, 3, 10, 11))
            self.assertEqual(bank.metrics().settlements, 0)
            self.assertGreater(failed.metrics().crystal_failures, 0)
            failed.close()
            legacy.close()

    def test_v1_and_layer_v2_coexist_with_deduped_phrase_and_summed_work(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy_bank = ContextualContinuationBank(
                root / "legacy.json",
                _identity(),
                max_cells=16,
            )
            hidden = torch.ones((1, 1, 8))
            legacy_bank.settle(
                captures=(legacy_bank.make_capture(hidden, 10, 3, (20, 21, 22)),)
            )
            layer_bank = LayerContextualContinuationBank(
                root / "layers.json",
                _layer_identity(),
                max_cells=16,
            )
            seed = _layer_transaction(layer_bank, boundary=0, known_token=3)
            layer_bank.settle_verified_prefix(seed, (10, 20, 21, 22))
            current = _layer_transaction(layer_bank, boundary=2, known_token=3)
            provider = self._provider(
                layer_bank,
                _LayerTransactions(current, (current,)),
                contextual_bank=legacy_bank,
            )
            provider.begin_request((1, 2, 3))
            proposal = provider.propose_round_state((1, 2, 3), 10, hidden)

            self.assertEqual(proposal.token_ids, (20, 21, 22))
            self.assertIsNone(provider._pending_phrase_option.crystal_layer)
            provider.observe_verification(3, 3)
            provider.reconcile_prefix((1, 2, 3, 10, 20, 21, 22))
            provider.observe_final((1, 2, 3, 10, 20, 21, 22))
            metrics = provider.metrics()
            self.assertEqual(metrics.crystal_queries, 2)
            self.assertEqual(metrics.crystal_proposed_tokens, 3)
            self.assertEqual(metrics.crystal_verified_tokens, 3)
            self.assertEqual(metrics.crystal_accepted_tokens, 3)
            self.assertEqual(metrics.crystal_bank_cells, 2)
            provider.close()

    def test_duplicate_layer_tails_count_only_the_selected_planner_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = LayerContextualContinuationBank(
                Path(temporary) / "layers.json",
                _layer_identity((1, 3)),
                max_cells=16,
            )
            seed = _layer_transaction(bank, boundary=0, known_token=3)
            bank.settle_verified_prefix(seed, (10, 20, 21, 22))
            current = _layer_transaction(bank, boundary=2, known_token=3)
            provider = self._provider(
                bank,
                _LayerTransactions(current, (current,)),
            )
            provider.begin_request((1, 2, 3))

            proposal = provider.propose_round_state(
                (1, 2, 3),
                10,
                torch.ones((1, 1, 8)),
            )
            self.assertEqual(proposal.token_ids, (20, 21, 22))
            provider.observe_verification(3, 3)
            provider.reconcile_prefix((1, 2, 3, 10, 20, 21, 22))
            provider.observe_final((1, 2, 3, 10, 20, 21, 22))

            metrics = provider.metrics()
            layer = metrics.layer_context_crystal
            assert layer is not None
            self.assertEqual(metrics.crystal_queries, 2)
            self.assertEqual(metrics.crystal_query_hits, 2)
            self.assertEqual(metrics.crystal_proposed_tokens, 3)
            self.assertEqual(metrics.crystal_verified_tokens, 3)
            self.assertEqual(metrics.crystal_accepted_tokens, 3)
            self.assertEqual(
                sum(
                    row["crystal_proposed_tokens"]
                    for row in layer["layers"].values()
                ),
                3,
            )
            self.assertEqual(
                sum(
                    row["crystal_accepted_tokens"]
                    for row in layer["layers"].values()
                ),
                3,
            )
            provider.close()

    def test_hybrid_forwards_layer_callbacks_without_loading_mtp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = LayerContextualContinuationBank(
                Path(temporary) / "layers.json",
                _layer_identity(),
                max_cells=16,
            )
            seed = _layer_transaction(bank, boundary=0, known_token=3)
            bank.settle_verified_prefix(seed, (10, 20, 21, 22))
            current = _layer_transaction(bank, boundary=2, known_token=3)
            markov = self._provider(
                bank,
                _LayerTransactions(current, (current,)),
            )
            mtp_factory = Mock(side_effect=AssertionError("MTP must stay cold"))
            hybrid = Qwen38MarkovMtpDraftProvider(markov, mtp_factory)
            prompt = (1, 2, 3)
            hidden = torch.ones((1, len(prompt), 8))
            hybrid.begin_request_state(prompt, hidden)

            proposal = hybrid.propose_round_state(
                prompt,
                10,
                hidden[:, -1:],
            )

            self.assertEqual(proposal.token_ids, (20, 21, 22))
            self.assertIsNotNone(hybrid.metrics().markov["layer_context_crystal"])
            mtp_factory.assert_not_called()
            hybrid.close()

    def test_no_bank_explicit_none_is_exact_legacy_metrics_and_trace(self) -> None:
        implicit = FingerprintRollingK4DraftProvider(
            vocab_size=256,
            proposal_width=3,
        )
        explicit = FingerprintRollingK4DraftProvider(
            vocab_size=256,
            proposal_width=3,
            layer_contextual_continuation_bank=None,
            layer_contextual_current_transaction=None,
            layer_contextual_transactions_since=None,
        )
        prompt = (1, 2, 3)
        hidden = torch.ones((1, 1, 8))
        implicit.begin_request(prompt)
        explicit.begin_request(prompt)

        self.assertEqual(
            implicit.propose_round_state(prompt, 10, hidden),
            explicit.propose_round_state(prompt, 10, hidden),
        )
        self.assertEqual(implicit.metrics().to_dict(), explicit.metrics().to_dict())
        self.assertNotIn("layer_context_crystal", implicit.metrics().to_dict())
        implicit.close()
        explicit.close()


if __name__ == "__main__":
    unittest.main()
