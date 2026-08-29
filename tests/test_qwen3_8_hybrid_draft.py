from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

import torch

from immer.runtimes.qwen3_8.draft_protocol import RollingDraftProposal
from immer.runtimes.qwen3_8.hybrid_draft import (
    MARKOV_MTP_WINDOW_WORK_COSTS,
    Qwen38MarkovMtpDraftError,
    Qwen38MarkovMtpDraftProvider,
)
from immer.runtimes.qwen3_8.markov_draft import (
    FingerprintRollingK4DraftProvider,
)


def _proposal(*, confidence: float, tokens=(3, 4, 5)) -> RollingDraftProposal:
    return RollingDraftProposal.build(
        tokens,
        (confidence,) * len(tokens),
        (0.0,) * len(tokens),
        request_window_ceiling=len(tokens) + 1,
        provider_abi="test-draft/v1",
    )


@dataclass(frozen=True)
class _MarkovMetrics:
    source_body_bytes: int = 0
    linear_calls: int = 0
    last_confidence: float = 0.9
    last_disagreement: float = 0.1
    last_phrase_source: str | None = "global"
    last_phrase_support: int = 3
    last_phrase_confidence: float = 0.8
    last_phrase_width: int = 3
    effective_experts: float = 2.5

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class _Markov:
    def __init__(self, confidence: float | list[float]) -> None:
        self.confidences = (
            [float(confidence)]
            if isinstance(confidence, (int, float))
            else [float(value) for value in confidence]
        )
        self.confidence = self.confidences[0]
        self._proposal_index = 0
        self.begin_calls = []
        self.propose_calls = []
        self.discard_calls = 0
        self.verification_calls = []
        self.reconcile_calls = []
        self.advance_calls = []
        self.external_reconcile_calls = []
        self.final_calls = []
        self.pending = False
        self.closed = False

    def begin_request(self, history):
        self.begin_calls.append(history)

    def propose_round(self, history, known_token):
        self.propose_calls.append((history, known_token, "round"))
        self.pending = True
        index = min(self._proposal_index, len(self.confidences) - 1)
        self.confidence = self.confidences[index]
        self._proposal_index += 1
        return _proposal(confidence=self.confidence)

    def propose_after(self, history, known_token):
        self.propose_calls.append((history, known_token, "plain"))
        self.pending = True
        return (3, 4, 5)

    def discard_pending_proposal(self):
        if not self.pending:
            raise AssertionError("discard without proposal")
        self.pending = False
        self.discard_calls += 1

    def observe_verification(self, accepted, verified):
        self.verification_calls.append((accepted, verified))

    def reconcile_prefix(self, history):
        if not self.pending:
            raise AssertionError("reconcile without proposal")
        self.pending = False
        self.reconcile_calls.append(history)

    def advance_confirmed_prefix(self, history):
        if self.pending:
            raise AssertionError("advance with pending proposal")
        self.advance_calls.append(history)

    def reconcile_external_prefix(self, history):
        if not self.pending:
            raise AssertionError("external reconcile without proposal")
        self.pending = False
        self.external_reconcile_calls.append(history)

    def observe_final(self, history):
        if self.pending:
            raise AssertionError("final with pending proposal")
        self.final_calls.append(history)

    def metrics(self):
        return _MarkovMetrics(last_confidence=self.confidence)

    def close(self):
        self.closed = True


@dataclass(frozen=True)
class _MtpMetrics:
    source_body_bytes: int = 4096
    linear_calls: int = 9
    pending: bool = False
    closed: bool = False

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class _Mtp:
    target_state_isolation = "hidden-argument+shared-pager-only/v1"

    def __init__(self) -> None:
        self.begin_calls = []
        self.propose_calls = []
        self.verification_calls = []
        self.reconcile_calls = []
        self.advance_calls = []
        self.final_calls = []
        self.pending = False
        self.closed = False

    def begin_request_state(self, history, hidden):
        self.begin_calls.append((history, hidden.detach().clone()))

    def propose_round_state(self, history, known_token, hidden):
        self.propose_calls.append(
            (history, known_token, hidden.detach().clone(), "round")
        )
        self.pending = True
        return _proposal(confidence=0.75, tokens=(7, 8, 9))

    def propose_after_state(self, history, known_token, hidden):
        self.propose_calls.append(
            (history, known_token, hidden.detach().clone(), "plain")
        )
        self.pending = True
        return (7, 8, 9)

    def observe_verification(self, accepted, verified):
        self.verification_calls.append((accepted, verified))

    def reconcile_prefix(self, history):
        if not self.pending:
            raise AssertionError("reconcile without MTP proposal")
        self.pending = False
        self.reconcile_calls.append(history)

    def advance_confirmed_prefix_state(
        self,
        history,
        previous_target_hidden,
        committed_hidden,
    ):
        if self.pending:
            raise AssertionError("advance with pending MTP proposal")
        self.advance_calls.append(
            (
                history,
                previous_target_hidden.detach().clone(),
                committed_hidden.detach().clone(),
            )
        )

    def observe_final(self, history):
        self.final_calls.append(history)

    def metrics(self):
        return _MtpMetrics(pending=self.pending, closed=self.closed)

    def close(self):
        self.closed = True


class Qwen38HybridDraftTests(unittest.TestCase):
    def test_high_value_markov_proposal_locks_without_loading_mtp(self) -> None:
        markov = _Markov(0.99)
        factory_calls = []
        provider = Qwen38MarkovMtpDraftProvider(
            markov,
            lambda: factory_calls.append(True),
        )
        prompt = (11, 12, 13)
        hidden = torch.randn((1, len(prompt), 8), dtype=torch.bfloat16)
        provider.begin_request_state(prompt, hidden)

        proposal = provider.propose_round_state(
            prompt,
            14,
            hidden[:, -1:],
        )

        self.assertGreater(
            proposal.select_window(
                request_window_ceiling=4,
                remaining_tokens=4,
                window_work_costs=MARKOV_MTP_WINDOW_WORK_COSTS,
            ).chosen_window,
            1,
        )
        self.assertEqual(provider.selected_provider, "markov")
        self.assertEqual(factory_calls, [])
        provider.observe_verification(2, 3)
        provider.reconcile_prefix((*prompt, 14, 3, 4))
        final = (*prompt, 14, 3, 4, 21)
        provider.observe_final(final)
        self.assertEqual(markov.verification_calls, [(2, 3)])
        self.assertEqual(markov.final_calls, [final])

        metrics = provider.metrics()
        self.assertEqual(metrics.selected_provider, "markov")
        self.assertEqual(metrics.selection_calls, 1)
        self.assertEqual(metrics.markov_selections, 1)
        self.assertEqual(metrics.mtp_selections, 0)
        self.assertEqual(metrics.source_body_bytes, 0)
        self.assertIsNone(metrics.mtp)
        self.assertEqual(metrics.effective_experts, 2.5)
        provider.close()
        self.assertTrue(markov.closed)

    def test_k1_lazily_loads_mtp_and_rechecks_markov_each_round(self) -> None:
        markov = _Markov(0.01)
        mtp = _Mtp()
        factory_calls = []

        def factory():
            factory_calls.append("mtp")
            return mtp

        provider = Qwen38MarkovMtpDraftProvider(markov, factory)
        prompt = (1, 2, 3, 4)
        hidden = torch.arange(32, dtype=torch.float32).reshape(1, 4, 8)
        original = hidden.clone()
        provider.begin_request_state(prompt, hidden)
        hidden.add_(1000)

        first = provider.propose_round_state(
            prompt,
            5,
            torch.zeros((1, 1, 8)),
        )

        self.assertEqual(first.token_ids, (7, 8, 9))
        self.assertEqual(provider.selected_provider, "mtp")
        self.assertEqual(factory_calls, ["mtp"])
        self.assertEqual(markov.discard_calls, 0)
        self.assertEqual(len(markov.propose_calls), 1)
        self.assertTrue(torch.equal(mtp.begin_calls[0][1], original))
        provider.observe_verification(1, 2)
        provider.reconcile_prefix((*prompt, 5, 7))

        next_history = (*prompt, 5, 7)
        provider.propose_round_state(
            next_history,
            6,
            torch.ones((1, 1, 8)),
        )
        provider.observe_verification(0, 1)
        provider.reconcile_prefix((*next_history, 6))
        final = (*next_history, 6, 22, 23)
        provider.observe_final(final)

        self.assertEqual(factory_calls, ["mtp"])
        self.assertEqual(len(markov.propose_calls), 2)
        self.assertEqual(markov.discard_calls, 0)
        self.assertEqual(
            markov.external_reconcile_calls,
            [(*prompt, 5, 7), (*next_history, 6)],
        )
        self.assertEqual(mtp.verification_calls, [(1, 2), (0, 1)])
        self.assertEqual(mtp.final_calls, [final])
        self.assertEqual(markov.final_calls, [final])
        metrics = provider.metrics()
        self.assertEqual(metrics.selected_provider, "mtp")
        self.assertEqual(metrics.source_body_bytes, 4096)
        self.assertEqual(metrics.linear_calls, 9)
        self.assertEqual(metrics.mtp_selections, 2)
        self.assertEqual(metrics.markov_external_feedback_rounds, 2)
        self.assertEqual(metrics.to_dict()["mtp"]["linear_calls"], 9)
        close_order = []

        def close_mtp():
            close_order.append("mtp")
            mtp.closed = True

        def close_markov():
            close_order.append("markov")
            markov.closed = True

        mtp.close = close_mtp
        markov.close = close_markov
        provider.close()
        provider.close()
        self.assertEqual(close_order, ["mtp", "markov"])
        self.assertTrue(markov.closed)
        self.assertTrue(mtp.closed)

    def test_mtp_initialization_failure_keeps_the_pending_markov_fallback(
        self,
    ) -> None:
        class FailingBeginMtp(_Mtp):
            def begin_request_state(self, history, hidden):
                super().begin_request_state(history, hidden)
                raise RuntimeError("broken MTP init")

        def broken_factory():
            raise RuntimeError("broken MTP factory")

        failing_mtp = FailingBeginMtp()
        for factory, expected_closed in (
            (broken_factory, None),
            (lambda: failing_mtp, failing_mtp),
        ):
            with self.subTest(factory=factory):
                markov = _Markov(0.01)
                provider = Qwen38MarkovMtpDraftProvider(markov, factory)
                prompt = (1, 2, 3)
                hidden = torch.zeros((1, len(prompt), 8))
                provider.begin_request_state(prompt, hidden)

                proposal = provider.propose_round_state(
                    prompt,
                    4,
                    hidden[:, -1:],
                )

                self.assertEqual(proposal.token_ids, (3, 4, 5))
                self.assertEqual(provider.selected_provider, "markov")
                self.assertTrue(markov.pending)
                metrics = provider.metrics()
                self.assertEqual(metrics.mtp_init_failures, 1)
                self.assertEqual(metrics.markov_rounds, 1)
                self.assertIsNone(metrics.mtp)
                if expected_closed is not None:
                    self.assertTrue(expected_closed.closed)
                provider.observe_verification(0, 1)
                provider.reconcile_prefix((*prompt, 4))
                provider.observe_final((*prompt, 4, 9))
                provider.close()

    def test_mtp_contract_violation_is_not_hidden_by_markov_fallback(self) -> None:
        class WrongIsolationMtp(_Mtp):
            target_state_isolation = "wrong/v1"

        markov = _Markov(0.01)
        mtp = WrongIsolationMtp()
        provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
        prompt = (1, 2, 3)
        hidden = torch.zeros((1, len(prompt), 8))
        provider.begin_request_state(prompt, hidden)

        with self.assertRaisesRegex(
            Qwen38MarkovMtpDraftError,
            "does not preserve target-state isolation",
        ):
            provider.propose_round_state(prompt, 4, hidden[:, -1:])

        self.assertTrue(mtp.closed)
        self.assertTrue(markov.pending)
        self.assertEqual(provider.metrics().mtp_init_failures, 0)
        provider.close()

    def test_round_council_switches_back_to_markov_and_keeps_mtp_synced(self) -> None:
        markov = _Markov([0.99, 0.99, 0.01, 0.99])
        mtp = _Mtp()
        factory_calls = []

        def factory():
            factory_calls.append("mtp")
            return mtp

        provider = Qwen38MarkovMtpDraftProvider(markov, factory)
        prompt = (1, 2)
        prompt_hidden = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
        expected_prompt = prompt_hidden.clone()
        provider.begin_request_state(prompt, prompt_hidden)
        prompt_hidden.add_(1000)

        provider.propose_round_state(prompt, 10, torch.full((1, 1, 4), 10.0))
        provider.observe_verification(1, 2)
        first_history = (*prompt, 10, 3)
        first_hidden = torch.tensor(
            [[[10.0, 11.0, 12.0, 13.0], [30.0, 31.0, 32.0, 33.0]]]
        )
        first_snapshot = first_hidden.clone()
        provider.reconcile_prefix_state(first_history, first_hidden)
        self.assertTrue(torch.equal(first_hidden, first_snapshot))

        provider.propose_round_state(
            first_history,
            11,
            torch.full((1, 1, 4), 11.0),
        )
        provider.observe_verification(0, 1)
        second_history = (*first_history, 11)
        second_hidden = torch.tensor([[[40.0, 41.0, 42.0, 43.0]]])
        provider.reconcile_prefix_state(second_history, second_hidden)

        switched = provider.propose_round_state(
            second_history,
            12,
            torch.full((1, 1, 4), 12.0),
        )

        expected_history_hidden = torch.cat(
            (expected_prompt, first_snapshot, second_hidden),
            dim=1,
        )
        self.assertEqual(switched.token_ids, (7, 8, 9))
        self.assertEqual(provider.selected_provider, "mtp")
        self.assertEqual(factory_calls, ["mtp"])
        self.assertEqual(markov.discard_calls, 0)
        self.assertEqual(len(markov.propose_calls), 3)
        self.assertEqual(mtp.begin_calls[0][0], second_history)
        self.assertTrue(torch.equal(mtp.begin_calls[0][1], expected_history_hidden))
        provider.observe_verification(1, 1)
        mtp_history = (*second_history, 12, 7)
        provider.reconcile_prefix_state(
            mtp_history,
            torch.tensor([[[50.0, 51.0, 52.0, 53.0], [60.0, 61.0, 62.0, 63.0]]]),
        )

        returned = provider.propose_round_state(
            mtp_history,
            13,
            torch.full((1, 1, 4), 13.0),
        )
        self.assertEqual(factory_calls, ["mtp"])
        self.assertEqual(returned.token_ids, (3, 4, 5))
        self.assertEqual(provider.selected_provider, "markov")
        self.assertEqual(len(markov.propose_calls), 4)
        self.assertEqual(len(mtp.propose_calls), 1)
        provider.observe_verification(1, 1)
        final_hidden = torch.full((1, 2, 4), 70.0)
        provider.reconcile_prefix_state(
            (*mtp_history, 13, 3),
            final_hidden,
        )
        final = (*mtp_history, 13, 3, 14)
        provider.observe_final(final)
        self.assertEqual(markov.final_calls, [final])
        self.assertEqual(markov.external_reconcile_calls, [mtp_history])
        self.assertEqual(len(mtp.advance_calls), 1)
        self.assertEqual(mtp.advance_calls[0][0], (*mtp_history, 13, 3))
        self.assertTrue(
            torch.equal(
                mtp.advance_calls[0][1],
                torch.full((1, 1, 4), 13.0),
            )
        )
        self.assertTrue(torch.equal(mtp.advance_calls[0][2], final_hidden))

        metrics = provider.metrics()
        self.assertEqual(metrics.schema, "immer.qwen3.8-markov-mtp-hybrid-provider/v8")
        self.assertEqual(metrics.selected_provider, "markov")
        self.assertEqual(metrics.selection_calls, 4)
        self.assertEqual(metrics.markov_rounds, 3)
        self.assertEqual(metrics.mtp_rounds, 1)
        self.assertEqual(metrics.markov_external_feedback_rounds, 1)
        self.assertEqual(metrics.provider_switches, 2)
        self.assertEqual(metrics.hidden_history_rows, len(second_history))
        self.assertEqual(
            metrics.hidden_history_bytes,
            expected_history_hidden.numel() * expected_history_hidden.element_size(),
        )
        self.assertFalse(metrics.switch_available)
        provider.close()

    def test_regular_reconcile_permanently_falls_back_to_markov(self) -> None:
        markov = _Markov([0.99, 0.01])
        factory_calls = []
        provider = Qwen38MarkovMtpDraftProvider(
            markov,
            lambda: factory_calls.append("mtp"),
        )
        prompt = (1, 2)
        hidden = torch.zeros((1, 2, 4))
        provider.begin_request_state(prompt, hidden)
        provider.propose_round_state(prompt, 3, hidden[:, -1:])
        provider.observe_verification(1, 1)
        history = (*prompt, 3, 3)
        provider.reconcile_prefix(history)

        proposal = provider.propose_round_state(
            history,
            4,
            hidden[:, -1:],
        )

        self.assertEqual(provider.selected_provider, "markov")
        self.assertEqual(factory_calls, [])
        self.assertEqual(markov.discard_calls, 0)
        self.assertEqual(proposal.token_ids, (3, 4, 5))
        metrics = provider.metrics()
        self.assertEqual(metrics.markov_rounds, 2)
        self.assertEqual(metrics.provider_switches, 0)
        self.assertFalse(metrics.switch_available)
        provider.observe_verification(0, 1)
        provider.reconcile_prefix((*history, 4))
        provider.observe_final((*history, 4, 8))
        provider.close()

    def test_hidden_callbacks_are_isolated_and_shape_bound(self) -> None:
        class MutatingMtp(_Mtp):
            def begin_request_state(self, history, hidden):
                super().begin_request_state(history, hidden)
                hidden.add_(500)

            def propose_round_state(self, history, known_token, hidden):
                result = super().propose_round_state(history, known_token, hidden)
                hidden.mul_(0)
                return result

        markov = _Markov(0.01)
        mtp = MutatingMtp()
        provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
        prompt = (1, 2)
        prompt_hidden = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
        prompt_snapshot = prompt_hidden.clone()
        provider.begin_request_state(prompt, prompt_hidden)
        proposal_hidden = torch.full((1, 1, 4), 7.0)
        proposal_snapshot = proposal_hidden.clone()

        provider.propose_round_state(prompt, 3, proposal_hidden)

        self.assertTrue(torch.equal(prompt_hidden, prompt_snapshot))
        self.assertTrue(torch.equal(proposal_hidden, proposal_snapshot))
        provider.observe_verification(0, 1)
        committed = torch.full((1, 1, 4), 9.0)
        committed_snapshot = committed.clone()
        provider.reconcile_prefix_state((*prompt, 3), committed)
        self.assertTrue(torch.equal(committed, committed_snapshot))
        provider.observe_final((*prompt, 3, 8))
        provider.close()

        bad = Qwen38MarkovMtpDraftProvider(_Markov(0.99), _Mtp)
        bad.begin_request_state(prompt, prompt_snapshot)
        bad.propose_round_state(prompt, 3, proposal_snapshot)
        bad.observe_verification(0, 1)
        with self.assertRaisesRegex(
            Qwen38MarkovMtpDraftError,
            "committed hidden rows",
        ):
            bad.reconcile_prefix_state(
                (*prompt, 3),
                torch.zeros((1, 2, 4)),
            )
        bad.close()

    def test_mtp_request_persists_full_target_episode_into_real_markov(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "markov.bin"
            markov = FingerprintRollingK4DraftProvider(
                vocab_size=64,
                state_path=state,
                max_history_tokens=128,
                proposal_width=3,
            )
            mtp = _Mtp()
            provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
            prompt = (40, 41, 42)
            hidden = torch.zeros((1, len(prompt), 8), dtype=torch.bfloat16)
            provider.begin_request_state(prompt, hidden)
            provider.propose_round_state(prompt, 43, hidden[:, -1:])
            self.assertEqual(provider.selected_provider, "mtp")
            provider.observe_verification(0, 1)
            provider.reconcile_prefix((*prompt, 43))
            final = (*prompt, 43, 44, 45, 46)
            provider.observe_final(final)
            live_metrics = provider.metrics()
            self.assertEqual(live_metrics.markov_external_feedback_rounds, 1)
            self.assertEqual(live_metrics.markov["external_reconcile_calls"], 1)
            self.assertEqual(live_metrics.markov["council_feedback"], 1)
            provider.close()

            restored = FingerprintRollingK4DraftProvider(
                vocab_size=64,
                state_path=state,
                max_history_tokens=128,
                proposal_width=3,
            )
            self.assertEqual(restored.confirmed_episodes(), (final,))
            self.assertEqual(restored.metrics().updates, 1)
            self.assertTrue(
                all(value == 1 for value in restored._state.expert_observations)
            )
            restored.close()

    def test_fixed_short_proposal_locks_directly_to_markov(self) -> None:
        markov = _Markov(0.01)
        factory_calls = []
        provider = Qwen38MarkovMtpDraftProvider(
            markov,
            lambda: factory_calls.append(True),
        )
        prompt = (1, 2)
        hidden = torch.zeros((1, 2, 4))
        provider.begin_request_state(prompt, hidden)

        proposal = provider.propose_after_state(prompt, 3, hidden[:, -1:])

        self.assertEqual(proposal, (3, 4, 5))
        self.assertEqual(provider.selected_provider, "markov")
        self.assertEqual(factory_calls, [])
        self.assertEqual(markov.propose_calls, [(prompt, 3, "plain")])
        self.assertEqual(provider.metrics().selection_calls, 1)
        provider.reconcile_prefix((*prompt, 3))
        provider.observe_final((*prompt, 3, 9))
        provider.close()

    def test_callback_state_rejects_unreconciled_or_reused_request(self) -> None:
        provider = Qwen38MarkovMtpDraftProvider(_Markov(0.99), _Mtp)
        self.assertTrue(callable(provider))
        with self.assertRaisesRegex(
            Qwen38MarkovMtpDraftError,
            "target-hidden rolling callbacks",
        ):
            provider((1, 2))
        prompt = (1, 2)
        hidden = torch.zeros((1, 2, 4))
        with self.assertRaisesRegex(Qwen38MarkovMtpDraftError, "not active"):
            provider.propose_round_state(prompt, 3, hidden[:, -1:])
        provider.begin_request_state(prompt, hidden)
        with self.assertRaisesRegex(Qwen38MarkovMtpDraftError, "exactly one"):
            provider.begin_request_state(prompt, hidden)
        provider.propose_round_state(prompt, 3, hidden[:, -1:])
        with self.assertRaisesRegex(Qwen38MarkovMtpDraftError, "unreconciled"):
            provider.observe_final((*prompt, 3))
        provider.reconcile_prefix((*prompt, 3))
        provider.observe_final((*prompt, 3, 9))
        with self.assertRaisesRegex(Qwen38MarkovMtpDraftError, "not active"):
            provider.propose_round_state((*prompt, 3), 9, hidden[:, -1:])
        provider.close()


if __name__ == "__main__":
    unittest.main()
