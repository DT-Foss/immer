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
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence
        self.begin_calls = []
        self.propose_calls = []
        self.discard_calls = 0
        self.verification_calls = []
        self.reconcile_calls = []
        self.final_calls = []
        self.pending = False
        self.closed = False

    def begin_request(self, history):
        self.begin_calls.append(history)

    def propose_round(self, history, known_token):
        self.propose_calls.append((history, known_token, "round"))
        self.pending = True
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

    def test_k1_discards_markov_lazily_loads_mtp_and_never_switches(self) -> None:
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
        self.assertEqual(markov.discard_calls, 1)
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
        self.assertEqual(len(markov.propose_calls), 1)
        self.assertEqual(mtp.verification_calls, [(1, 2), (0, 1)])
        self.assertEqual(mtp.final_calls, [final])
        self.assertEqual(markov.final_calls, [final])
        metrics = provider.metrics()
        self.assertEqual(metrics.selected_provider, "mtp")
        self.assertEqual(metrics.source_body_bytes, 4096)
        self.assertEqual(metrics.linear_calls, 9)
        self.assertEqual(metrics.mtp_selections, 1)
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
            provider.close()

            restored = FingerprintRollingK4DraftProvider(
                vocab_size=64,
                state_path=state,
                max_history_tokens=128,
                proposal_width=3,
            )
            self.assertEqual(restored.confirmed_episodes(), (final,))
            self.assertEqual(restored.metrics().updates, 1)
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
