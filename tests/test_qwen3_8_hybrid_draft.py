from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from immer.runtimes.qwen3_8.draft_protocol import RollingDraftProposal
from immer.runtimes.qwen3_8.hybrid_draft import (
    MARKOV_MTP_WINDOW_WORK_COSTS,
    Qwen38MarkovMtpDraftError,
    Qwen38MarkovMtpDraftProvider,
)
from immer.runtimes.qwen3_8.markov_draft import (
    FingerprintRollingK4DraftProvider,
    MarkovLanguageTokenEvidence,
)
from immer.runtimes.qwen3_8.markov_atlas import AtlasTokenEvidence
from immer.runtimes.qwen3_8.mtp_draft import Qwen35MtpCarry


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
    def __init__(
        self,
        confidence: float | list[float],
        *,
        tokens: tuple[int, ...] = (3, 4, 5),
        atlas_scores: tuple[float, ...] = (),
        online_scores: tuple[float, ...] = (),
        provider_policy: dict[str, tuple[tuple[float, bool], ...]] | None = None,
    ) -> None:
        self.confidences = (
            [float(confidence)]
            if isinstance(confidence, (int, float))
            else [float(value) for value in confidence]
        )
        self.confidence = self.confidences[0]
        self.tokens = tokens
        self.atlas_scores = atlas_scores
        self.online_scores = online_scores
        self.provider_policy = provider_policy or {}
        self.provider_feedback = []
        self._provider_observations = {
            "markov": [0] * 16,
            "mtp": [0] * 16,
        }
        self._provider_hits = {
            "markov": [0] * 16,
            "mtp": [0] * 16,
        }
        self._proposal_index = 0
        self.begin_calls = []
        self.propose_calls = []
        self.discard_calls = 0
        self.verification_calls = []
        self.virtual_verification_calls = []
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
        return _proposal(confidence=self.confidence, tokens=self.tokens)

    def propose_after(self, history, known_token):
        self.propose_calls.append((history, known_token, "plain"))
        self.pending = True
        return self.tokens

    def discard_pending_proposal(self):
        if not self.pending:
            raise AssertionError("discard without proposal")
        self.pending = False
        self.discard_calls += 1

    def atlas_evidence_for_pending(self, token_ids):
        if not self.atlas_scores:
            return ()
        return tuple(
            AtlasTokenEvidence(
                token_id=token,
                context_order=2 if score > 0.0 else 0,
                support=8 if score > 0.0 else 0,
                total=16 if score > 0.0 else 0,
                probability=0.5 if score > 0.0 else 0.0,
                score=score,
            )
            for token, score in zip(token_ids, self.atlas_scores, strict=True)
        )

    def language_evidence_for_pending(self, token_ids):
        if not self.atlas_scores and not self.online_scores:
            return ()
        atlas = self.atlas_scores or (0.0,) * len(token_ids)
        online = self.online_scores or (0.0,) * len(token_ids)
        return tuple(
            MarkovLanguageTokenEvidence(
                token_id=token,
                atlas_score=atlas_score,
                online_score=online_score,
                score=1.0 - (1.0 - atlas_score) * (1.0 - online_score),
                atlas_support=8 if atlas_score > 0.0 else 0,
                online_support=8 if online_score > 0.0 else 0,
            )
            for token, atlas_score, online_score in zip(
                token_ids,
                atlas,
                online,
                strict=True,
            )
        )

    def provider_policy_score(self, provider, position):
        configured = self.provider_policy.get(provider, ())
        if position < len(configured):
            return configured[position]
        observations = self._provider_observations[provider][position]
        if observations == 0:
            return 0.5, False
        return self._provider_hits[provider][position] / observations, True

    def observe_provider_policy_feedback(self, provider, position, hit):
        self.provider_feedback.append((provider, position, hit))
        self._provider_observations[provider][position] += 1
        self._provider_hits[provider][position] += int(hit)

    def observe_verification(self, accepted, verified):
        self.verification_calls.append((accepted, verified))

    def observe_virtual_verification(self, accepted, verified):
        self.virtual_verification_calls.append((accepted, verified))

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
        self.virtual_verification_calls = []
        self.teacher_verification_calls = []
        self.reconcile_calls = []
        self.reconcile_state_calls = []
        self.advance_calls = []
        self.final_calls = []
        self.export_calls = []
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

    def observe_virtual_verification(self, accepted, verified):
        self.virtual_verification_calls.append((accepted, verified))

    def observe_teacher_verification(self, position, outcome):
        self.teacher_verification_calls.append((position, outcome))

    def reconcile_prefix(self, history):
        if not self.pending:
            raise AssertionError("reconcile without MTP proposal")
        self.pending = False
        self.reconcile_calls.append(history)

    def reconcile_prefix_state(self, history, hidden):
        if not self.pending:
            raise AssertionError("stateful reconcile without MTP proposal")
        self.pending = False
        self.reconcile_state_calls.append((history, hidden.detach().clone()))

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

    def export_carry(self, history):
        self.export_calls.append(history)
        return ("carry", history)

    def metrics(self):
        return _MtpMetrics(pending=self.pending, closed=self.closed)

    def close(self):
        self.closed = True


class Qwen38HybridDraftTests(unittest.TestCase):
    def test_teacher_verification_delegates_only_for_pending_mtp(self) -> None:
        markov = _Markov([0.01, 0.99])
        mtp = _Mtp()
        provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
        prompt = (1, 2, 3)
        hidden = torch.zeros((1, len(prompt), 8))
        provider.begin_request_state(prompt, hidden)

        provider.observe_teacher_verification(1, False)
        provider.propose_round_state(prompt, 4, hidden[:, -1:])
        self.assertEqual(provider.selected_provider, "mtp")
        provider.observe_teacher_verification(1, True)
        self.assertEqual(mtp.teacher_verification_calls, [(1, True)])

        provider.observe_verification(1, 1)
        mtp_history = (*prompt, 4, 7)
        provider.reconcile_prefix(mtp_history)
        provider.observe_teacher_verification(2, False)

        provider.propose_round_state(
            mtp_history,
            5,
            torch.ones((1, 1, 8)),
        )
        self.assertEqual(provider.selected_provider, "markov")
        provider.observe_teacher_verification(1, False)
        self.assertEqual(mtp.teacher_verification_calls, [(1, True)])

        provider.observe_verification(1, 1)
        final_prefix = (*mtp_history, 5, 3)
        provider.reconcile_prefix(final_prefix)
        provider.observe_final((*final_prefix, 10))
        provider.close()

    def test_restored_hybrid_initializes_mtp_from_carried_prefix_suffix(self) -> None:
        markov = _Markov(0.01)
        mtp = _Mtp()
        provider = Qwen38MarkovMtpDraftProvider(
            markov,
            lambda: mtp,
            restored_prefix_length=2,
        )
        prompt = (11, 12, 13, 14)
        suffix_hidden = torch.randn((1, 2, 8), dtype=torch.bfloat16)

        provider.begin_request_state(prompt, suffix_hidden)
        proposal = provider.propose_round_state(
            prompt,
            15,
            suffix_hidden[:, -1:],
        )

        self.assertEqual(proposal.token_ids, (7, 8, 9))
        self.assertEqual(markov.begin_calls, [prompt])
        self.assertEqual(mtp.begin_calls[0][0], prompt)
        self.assertTrue(torch.equal(mtp.begin_calls[0][1], suffix_hidden))
        provider.close()

    def test_terminal_uncommitted_tail_exports_last_target_boundary(self) -> None:
        markov = _Markov(0.01)
        mtp = _Mtp()
        provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
        prompt = (11, 12)
        hidden = torch.randn((1, 2, 8), dtype=torch.bfloat16)
        provider.begin_request_state(prompt, hidden)
        proposal = provider.propose_round_state(prompt, 13, hidden[:, -1:])
        provider.observe_verification(1, 2)
        emitted_history = (*prompt, 13, proposal.token_ids[0])
        provider.reconcile_prefix(emitted_history)
        provider.observe_final(emitted_history)

        carry = provider.export_mtp_carry(prompt)

        self.assertEqual(carry, ("carry", prompt))
        self.assertEqual(mtp.export_calls, [prompt])
        with self.assertRaisesRegex(
            Qwen38MarkovMtpDraftError,
            "target boundary",
        ):
            provider.export_mtp_carry(emitted_history)
        provider.close()

    def test_mtp_carry_export_rejects_wrong_target_boundary_hidden(self) -> None:
        class WrongHiddenMtp(_Mtp):
            def export_carry(self, history):
                return Qwen35MtpCarry(
                    schema="fixture",
                    identity=(),
                    history=history,
                    next_position=len(history) - 1,
                    state=None,
                    last_target_hidden=torch.zeros((1, 1, 8)),
                )

        markov = _Markov(0.01)
        mtp = WrongHiddenMtp()
        provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
        prompt = (11, 12)
        hidden = torch.ones((1, 2, 8), dtype=torch.bfloat16)
        provider.begin_request_state(prompt, hidden)
        proposal = provider.propose_round_state(prompt, 13, hidden[:, -1:])
        provider.observe_verification(1, 2)
        emitted = (*prompt, 13, proposal.token_ids[0])
        provider.reconcile_prefix(emitted)
        provider.observe_final(emitted)

        with self.assertRaisesRegex(
            Qwen38MarkovMtpDraftError,
            "target hidden differs",
        ):
            provider.export_mtp_carry(prompt)
        provider.close()

    def test_markov_only_request_materializes_mtp_carry_at_export(self) -> None:
        markov = _Markov(0.99)
        mtp = _Mtp()
        factory_calls = []

        def factory():
            factory_calls.append(True)
            return mtp

        provider = Qwen38MarkovMtpDraftProvider(markov, factory)
        prompt = (11, 12)
        hidden = torch.randn((1, 2, 8), dtype=torch.bfloat16)
        provider.begin_request_state(prompt, hidden)
        provider.propose_round_state(prompt, 13, hidden[:, -1:])
        provider.observe_verification(1, 2)
        committed = (*prompt, 13, 3)
        extension_hidden = torch.randn((1, 2, 8), dtype=torch.bfloat16)
        provider.reconcile_prefix_state(committed, extension_hidden)
        provider.observe_final(committed)

        carry = provider.export_mtp_carry(committed)

        self.assertEqual(carry, ("carry", committed))
        self.assertEqual(factory_calls, [True])
        self.assertEqual(mtp.begin_calls[0][0], committed)
        self.assertEqual(tuple(mtp.begin_calls[0][1].shape), (1, 4, 8))
        provider.close()

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

    def test_real_atlas_online_beam_locks_without_loading_mtp(self) -> None:
        class PlanningExpert:
            @staticmethod
            def distribution(context):
                if not context:
                    return {"07": 0.55, "08": 0.45}
                if context[-1] == "07":
                    return {"09": 0.51, "10": 0.49}
                if context[-1] == "08":
                    return {"09": 0.99, "10": 0.01}
                return {"11": 0.95, "12": 0.05}

        class Atlas:
            max_branches = 8
            context_count = 2
            token_count = 10

            @staticmethod
            def continuation(*_args, **_kwargs):
                return None

            @staticmethod
            def token_options(history, *, limit=None):
                if history[-1] != 2:
                    return ()
                return (
                    AtlasTokenEvidence(7, 1, 8, 10, 0.8, 0.6),
                    AtlasTokenEvidence(8, 1, 6, 10, 0.6, 0.5),
                )

        markov = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            proposal_width=3,
        )
        markov.atlas = Atlas()
        markov._expert_models = lambda _history: tuple(
            (PlanningExpert(), []) for _ in markov._experts
        )
        factory_calls = []
        provider = Qwen38MarkovMtpDraftProvider(
            markov,
            lambda: factory_calls.append("mtp"),
        )
        prompt = (20, 21)
        hidden = torch.zeros((1, len(prompt), 8))
        provider.begin_request_state(prompt, hidden)

        proposal = provider.propose_round_state(prompt, 2, hidden[:, -1:])

        self.assertEqual(proposal.token_ids[:2], (8, 9))
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
        self.assertEqual(markov._beam_verified_tokens, 0)
        self.assertEqual(markov._beam_accepted_tokens, 0)
        committed = (*prompt, 2, *proposal.token_ids[:2])
        provider.reconcile_prefix(committed)
        self.assertEqual(markov._beam_verified_tokens, 3)
        self.assertEqual(markov._beam_accepted_tokens, 2)
        provider.observe_final((*committed, 13))
        provider.close()

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
        self.assertEqual(first.provider_abi, "test-draft/v1")
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
        self.assertEqual(metrics.provider_tournament_calls, 2)
        self.assertEqual(metrics.provider_tournament_markov_selections, 0)
        self.assertEqual(metrics.provider_tournament_mtp_selections, 2)
        self.assertEqual(metrics.consensus_rounds, 0)
        self.assertEqual(metrics.consensus_agreement_tokens, 0)
        self.assertEqual(metrics.consensus_confidence_gain, 0.0)
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

    def test_learned_markov_skill_beats_raw_mtp_confidence_and_reconciles_shadow(
        self,
    ) -> None:
        learned = {
            "markov": ((1.0, True),) * 3,
            "mtp": ((0.05, True),) * 3,
        }
        markov = _Markov(
            0.10,
            tokens=(3, 4, 5),
            provider_policy=learned,
        )
        mtp = _Mtp()
        provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
        provider._select_markov = lambda _proposal: False
        prompt = (1, 2, 3)
        hidden = torch.zeros((1, len(prompt), 8))
        provider.begin_request_state(prompt, hidden)

        proposal = provider.propose_round_state(prompt, 4, hidden[:, -1:])

        self.assertEqual(proposal.token_ids, (3, 4, 5))
        self.assertEqual(proposal.token_confidences, (0.10, 0.10, 0.10))
        self.assertEqual(provider.selected_provider, "markov")
        self.assertTrue(mtp.pending)
        provider.observe_verification(1, 1)
        committed = (*prompt, 4, 3)
        committed_hidden = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8)
        provider.reconcile_prefix_state(committed, committed_hidden)

        self.assertFalse(mtp.pending)
        self.assertEqual(len(mtp.reconcile_state_calls), 1)
        self.assertEqual(mtp.reconcile_state_calls[0][0], committed)
        self.assertTrue(
            torch.equal(mtp.reconcile_state_calls[0][1], committed_hidden)
        )
        metrics = provider.metrics()
        self.assertEqual(metrics.provider_tournament_calls, 1)
        self.assertEqual(metrics.provider_tournament_markov_selections, 1)
        self.assertEqual(metrics.provider_tournament_mtp_selections, 0)
        self.assertEqual(metrics.markov_rounds, 1)
        self.assertEqual(metrics.mtp_rounds, 0)
        provider.observe_final((*committed, 10))
        provider.close()

    def test_provider_candidates_stop_feedback_at_their_own_first_mismatch(
        self,
    ) -> None:
        markov = _Markov(0.10, tokens=(7, 31, 9))
        mtp = _Mtp()
        provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
        provider._select_markov = lambda _proposal: False
        prompt = (1, 2, 3)
        hidden = torch.zeros((1, len(prompt), 8))
        provider.begin_request_state(prompt, hidden)

        provider.propose_round_state(prompt, 4, hidden[:, -1:])
        self.assertEqual(provider.selected_provider, "mtp")
        provider.observe_verification(0, 1)
        committed = (*prompt, 4, 7, 31)
        provider.reconcile_prefix(committed)
        provider.observe_final((*committed, 10))

        self.assertEqual(
            markov.provider_feedback,
            [
                ("markov", 0, True),
                ("mtp", 0, True),
                ("markov", 1, True),
                ("mtp", 1, False),
                ("markov", 2, False),
            ],
        )
        metrics = provider.metrics()
        self.assertEqual(metrics.provider_trace_created, 1)
        self.assertEqual(metrics.provider_trace_active, 0)
        self.assertEqual(metrics.provider_trace_feedback_tokens, 5)
        provider.close()

    def test_same_request_provider_feedback_switches_mtp_to_markov(self) -> None:
        markov = _Markov(0.10, tokens=(3, 4, 5))
        mtp = _Mtp()
        provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
        provider._select_markov = lambda _proposal: False
        prompt = (1, 2, 3)
        hidden = torch.zeros((1, len(prompt), 8))
        provider.begin_request_state(prompt, hidden)

        first = provider.propose_round_state(prompt, 4, hidden[:, -1:])
        self.assertEqual(first.token_ids, (7, 8, 9))
        self.assertEqual(provider.selected_provider, "mtp")
        provider.observe_verification(0, 1)
        first_history = (*prompt, 4, 3, 4, 5)
        provider.reconcile_prefix(first_history)
        self.assertEqual(markov.provider_policy_score("markov", 0), (1.0, True))
        self.assertEqual(markov.provider_policy_score("mtp", 0), (0.0, True))

        second = provider.propose_round_state(
            first_history,
            6,
            torch.ones((1, 1, 8)),
        )
        self.assertEqual(second.token_ids, (3, 4, 5))
        self.assertEqual(provider.selected_provider, "markov")
        provider.observe_verification(1, 1)
        second_history = (*first_history, 6, 3)
        provider.reconcile_prefix(second_history)
        self.assertEqual(mtp.reconcile_calls, [first_history, second_history])
        provider.observe_final((*second_history, 10))

        metrics = provider.metrics()
        self.assertEqual(metrics.provider_tournament_calls, 2)
        self.assertEqual(metrics.provider_tournament_mtp_selections, 1)
        self.assertEqual(metrics.provider_tournament_markov_selections, 1)
        self.assertEqual(metrics.provider_switches, 1)
        provider.close()

    def test_overlapping_provider_traces_persist_global_and_dialect_rows(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "provider-traces.bin"
            markov = FingerprintRollingK4DraftProvider(
                vocab_size=64,
                state_path=state_path,
                max_history_tokens=128,
                proposal_width=3,
            )
            original_propose = markov.propose_round
            candidates = (
                (7, 8, 9, 10, 13, 14, 15),
                (9, 10, 11, 12, 13, 14, 15),
            )
            markov_round = 0

            def deterministic_propose(history, known_token):
                nonlocal markov_round
                original_propose(history, known_token)
                tokens = candidates[min(markov_round, len(candidates) - 1)]
                markov_round += 1
                return _proposal(confidence=0.10, tokens=tokens)

            class SequencedMtp(_Mtp):
                def propose_round_state(self, history, known_token, hidden):
                    self.propose_calls.append(
                        (history, known_token, hidden.detach().clone(), "round")
                    )
                    self.pending = True
                    index = min(len(self.propose_calls) - 1, len(candidates) - 1)
                    return _proposal(confidence=0.75, tokens=candidates[index])

            mtp = SequencedMtp()
            provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
            provider._select_markov = lambda _proposal: False
            prompt = (40, 41)
            hidden = torch.zeros((1, len(prompt), 8), dtype=torch.bfloat16)
            provider.begin_request_state(prompt, hidden)
            with (
                mock.patch.object(
                    markov,
                    "propose_round",
                    side_effect=deterministic_propose,
                ),
                mock.patch.object(
                    markov,
                    "language_evidence_for_pending",
                    return_value=(),
                ),
            ):
                provider.propose_round_state(prompt, 42, hidden[:, -1:])
                provider.observe_verification(1, 1)
                first_history = (*prompt, 42, 7)
                provider.reconcile_prefix(first_history)

                provider.propose_round_state(
                    first_history,
                    8,
                    torch.ones((1, 1, 8), dtype=torch.bfloat16),
                )
                provider.observe_verification(1, 1)
                second_history = (*first_history, 8, 9)
                provider.reconcile_prefix(second_history)

            interim = provider.metrics()
            self.assertEqual(interim.provider_trace_created, 2)
            self.assertEqual(interim.provider_trace_active, 2)
            self.assertEqual(interim.provider_trace_feedback_tokens, 8)

            final_history = (*second_history, 10)
            provider.observe_final(final_history)
            final = provider.metrics()
            self.assertEqual(final.provider_trace_active, 0)
            self.assertEqual(final.provider_trace_feedback_tokens, 12)
            expected_observations = (2, 2, 1)
            expected_hits = (2, 2, 1)
            self.assertEqual(
                tuple(final.markov["planner_observations"][3][:3]),
                expected_observations,
            )
            self.assertEqual(
                tuple(final.markov["planner_hits"][3][:3]),
                expected_hits,
            )
            self.assertEqual(
                tuple(final.markov["planner_observations"][4][:3]),
                expected_observations,
            )
            self.assertEqual(
                tuple(final.markov["planner_hits"][4][:3]),
                expected_hits,
            )
            self.assertEqual(
                tuple(final.markov["active_dialect_planner_observations"][3][:3]),
                expected_observations,
            )
            self.assertEqual(
                tuple(final.markov["active_dialect_planner_hits"][3][:3]),
                expected_hits,
            )
            self.assertEqual(
                tuple(final.markov["active_dialect_planner_observations"][4][:3]),
                expected_observations,
            )
            self.assertEqual(
                tuple(final.markov["active_dialect_planner_hits"][4][:3]),
                expected_hits,
            )
            provider.close()

            restored = FingerprintRollingK4DraftProvider(
                vocab_size=64,
                state_path=state_path,
                max_history_tokens=128,
                proposal_width=3,
            )
            restored_metrics = restored.metrics()
            self.assertEqual(
                restored_metrics.planner_observations[3][:3],
                expected_observations,
            )
            self.assertEqual(restored_metrics.planner_hits[3][:3], expected_hits)
            self.assertEqual(
                restored_metrics.planner_observations[4][:3],
                expected_observations,
            )
            self.assertEqual(restored_metrics.planner_hits[4][:3], expected_hits)
            restored._activate_dialect(prompt)
            restored_metrics = restored.metrics()
            self.assertEqual(
                restored_metrics.active_dialect_planner_observations[3][:3],
                expected_observations,
            )
            self.assertEqual(
                restored_metrics.active_dialect_planner_hits[3][:3],
                expected_hits,
            )
            self.assertEqual(
                restored_metrics.active_dialect_planner_observations[4][:3],
                expected_observations,
            )
            self.assertEqual(
                restored_metrics.active_dialect_planner_hits[4][:3],
                expected_hits,
            )
            restored.close()

    def test_final_failure_rolls_back_provider_feedback_and_close_cleans_up(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "provider-final-failure.bin"
            markov = FingerprintRollingK4DraftProvider(
                vocab_size=64,
                state_path=state_path,
                max_history_tokens=128,
                proposal_width=3,
            )
            original_propose = markov.propose_round

            def deterministic_propose(history, known_token):
                original_propose(history, known_token)
                return _proposal(confidence=0.10, tokens=(7, 8, 9))

            mtp = _Mtp()
            provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
            provider._select_markov = lambda _proposal: False
            prompt = (40, 41)
            hidden = torch.zeros((1, len(prompt), 8), dtype=torch.bfloat16)
            provider.begin_request_state(prompt, hidden)
            with mock.patch.object(
                markov,
                "propose_round",
                side_effect=deterministic_propose,
            ):
                provider.propose_round_state(prompt, 42, hidden[:, -1:])
                provider.observe_verification(1, 1)
                committed = (*prompt, 42, 7)
                provider.reconcile_prefix(committed)

            before_traces = list(provider._provider_traces)
            before_feedback_tokens = provider.metrics().provider_trace_feedback_tokens
            before_observations = [
                list(row) for row in markov._request_planner_observations
            ]
            before_hits = [list(row) for row in markov._request_planner_hits]
            before_feedback = list(markov._planner_feedback)
            with (
                mock.patch.object(markov, "_persist", side_effect=OSError("disk full")),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                provider.observe_final((*committed, 8))

            self.assertEqual(provider._provider_traces, before_traces)
            self.assertEqual(
                provider.metrics().provider_trace_feedback_tokens,
                before_feedback_tokens,
            )
            self.assertEqual(markov._request_planner_observations, before_observations)
            self.assertEqual(markov._request_planner_hits, before_hits)
            self.assertEqual(markov._planner_feedback, before_feedback)

            provider.close()
            self.assertEqual(provider._provider_traces, [])
            self.assertIsNone(provider._pending_provider_candidates)
            self.assertIsNone(markov._request_planner_observations)
            self.assertIsNone(markov._request_planner_hits)
            self.assertEqual(provider.metrics().provider_trace_active, 0)
            self.assertTrue(provider.metrics().closed)
            self.assertTrue(markov._closed)

    def test_virtual_k1_verification_routes_only_to_the_selected_mtp(self) -> None:
        markov = _Markov(0.01)
        mtp = _Mtp()
        provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
        prompt = (1, 2, 3, 4)
        hidden = torch.zeros((1, len(prompt), 8))
        provider.begin_request_state(prompt, hidden)

        provider.propose_round_state(prompt, 5, hidden[:, -1:])
        provider.observe_virtual_verification(1, 1)
        provider.reconcile_prefix((*prompt, 5))

        self.assertEqual(mtp.virtual_verification_calls, [(1, 1)])
        self.assertEqual(markov.virtual_verification_calls, [])
        provider.observe_final((*prompt, 5, 9))
        provider.close()
        self.assertTrue(markov.closed)
        self.assertTrue(mtp.closed)

    def test_mtp_proposal_gains_discounted_confidence_from_markov_agreement(
        self,
    ) -> None:
        markov = _Markov(0.55, tokens=(7, 8, 9))
        mtp = _Mtp()
        provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
        provider._select_markov = lambda _proposal: False
        prompt = (1, 2, 3)
        hidden = torch.zeros((1, len(prompt), 8))
        provider.begin_request_state(prompt, hidden)

        proposal = provider.propose_round_state(
            prompt,
            4,
            hidden[:, -1:],
        )

        self.assertEqual(provider.selected_provider, "mtp")
        self.assertEqual(proposal.token_ids, (7, 8, 9))
        self.assertEqual(
            proposal.provider_abi,
            "immer.qwen3.8-markov-mtp-hybrid-provider/v27",
        )
        self.assertTrue(
            all(abs(value - 0.7625) < 1e-12 for value in proposal.token_confidences)
        )
        metrics = provider.metrics()
        self.assertEqual(metrics.consensus_rounds, 1)
        self.assertEqual(metrics.consensus_agreement_tokens, 3)
        self.assertEqual(metrics.last_consensus_agreement_tokens, 3)
        self.assertAlmostEqual(metrics.consensus_confidence_gain, 0.0375)
        self.assertAlmostEqual(metrics.last_consensus_confidence_gain, 0.0375)
        provider.observe_verification(1, 1)
        provider.reconcile_prefix((*prompt, 4, 7))
        provider.observe_final((*prompt, 4, 7, 10))
        provider.close()

    def test_atlas_votes_for_mtp_tokens_without_requiring_markov_top1(self) -> None:
        markov = _Markov(
            0.01,
            tokens=(31, 32, 33),
            atlas_scores=(0.8, 0.4, 0.0),
        )
        mtp = _Mtp()
        provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
        prompt = (1, 2, 3)
        hidden = torch.zeros((1, len(prompt), 8))
        provider.begin_request_state(prompt, hidden)

        proposal = provider.propose_round_state(prompt, 4, hidden[:, -1:])

        self.assertEqual(provider.selected_provider, "mtp")
        self.assertEqual(proposal.token_ids, (7, 8, 9))
        self.assertEqual(proposal.token_confidences, (0.8, 0.775, 0.75))
        metrics = provider.metrics()
        self.assertEqual(metrics.consensus_agreement_tokens, 0)
        self.assertEqual(metrics.atlas_consensus_rounds, 1)
        self.assertEqual(metrics.atlas_consensus_tokens, 2)
        self.assertAlmostEqual(metrics.atlas_consensus_confidence_gain, 0.075)
        self.assertEqual(metrics.last_atlas_consensus_tokens, 2)
        self.assertAlmostEqual(
            metrics.last_atlas_consensus_confidence_gain,
            0.075,
        )
        provider.observe_verification(1, 2)
        provider.reconcile_prefix((*prompt, 4, 7))
        provider.observe_final((*prompt, 4, 7, 10))
        provider.close()

    def test_online_memory_votes_for_mtp_tokens_without_static_atlas(self) -> None:
        markov = _Markov(
            0.01,
            tokens=(31, 32, 33),
            online_scores=(0.6, 0.2, 0.0),
        )
        mtp = _Mtp()
        provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
        prompt = (1, 2, 3)
        hidden = torch.zeros((1, len(prompt), 8))
        provider.begin_request_state(prompt, hidden)

        proposal = provider.propose_round_state(prompt, 4, hidden[:, -1:])

        self.assertEqual(proposal.token_confidences, (0.7875, 0.7625, 0.75))
        metrics = provider.metrics()
        self.assertEqual(metrics.atlas_consensus_tokens, 0)
        self.assertEqual(metrics.online_consensus_rounds, 1)
        self.assertEqual(metrics.online_consensus_tokens, 2)
        self.assertAlmostEqual(metrics.online_consensus_confidence_gain, 0.05)
        provider.observe_verification(1, 2)
        provider.reconcile_prefix((*prompt, 4, 7))
        provider.observe_final((*prompt, 4, 7, 10))
        provider.close()

    def test_consensus_stops_at_first_divergence(self) -> None:
        markov = _Markov(0.55, tokens=(7, 31, 9))
        mtp = _Mtp()
        provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
        provider._select_markov = lambda _proposal: False
        prompt = (1, 2, 3)
        hidden = torch.zeros((1, len(prompt), 8))
        provider.begin_request_state(prompt, hidden)

        proposal = provider.propose_round_state(prompt, 4, hidden[:, -1:])

        self.assertEqual(proposal.token_ids, (7, 8, 9))
        self.assertAlmostEqual(proposal.token_confidences[0], 0.7625)
        self.assertEqual(proposal.token_confidences[1:], (0.75, 0.75))
        metrics = provider.metrics()
        self.assertEqual(metrics.consensus_agreement_tokens, 1)
        self.assertAlmostEqual(metrics.consensus_confidence_gain, 0.0125)
        provider.observe_verification(0, 1)
        provider.reconcile_prefix((*prompt, 4))
        provider.observe_final((*prompt, 4, 10))
        provider.close()

    def test_consensus_never_revives_mtp_zero_confidence_padding(self) -> None:
        class PaddedMtp(_Mtp):
            def propose_round_state(self, history, known_token, hidden):
                self.propose_calls.append(
                    (history, known_token, hidden.detach().clone(), "round")
                )
                self.pending = True
                return RollingDraftProposal.build(
                    (7, 8, 9),
                    (0.75, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    request_window_ceiling=4,
                    provider_abi="test-mtp-padded/v1",
                )

        equal_policy = {
            "markov": ((0.8, True),) * 3,
            "mtp": ((0.8, True),) * 3,
        }
        markov = _Markov(
            0.55,
            tokens=(7, 8, 9),
            provider_policy=equal_policy,
        )
        mtp = PaddedMtp()
        provider = Qwen38MarkovMtpDraftProvider(markov, lambda: mtp)
        provider._select_markov = lambda _proposal: False
        prompt = (1, 2, 3)
        hidden = torch.zeros((1, len(prompt), 8))
        provider.begin_request_state(prompt, hidden)

        proposal = provider.propose_round_state(prompt, 4, hidden[:, -1:])

        self.assertEqual(provider.selected_provider, "mtp")
        self.assertAlmostEqual(proposal.token_confidences[0], 0.7625)
        self.assertEqual(proposal.token_confidences[1:], (0.0, 0.0))
        metrics = provider.metrics()
        self.assertEqual(metrics.consensus_agreement_tokens, 1)
        self.assertAlmostEqual(metrics.consensus_confidence_gain, 0.0125)
        provider.observe_verification(0, 1)
        provider.reconcile_prefix((*prompt, 4))
        provider.observe_final((*prompt, 4, 10))
        provider.close()

    def test_consensus_gain_is_monotone_at_confidence_cap(self) -> None:
        provider = Qwen38MarkovMtpDraftProvider(_Markov(0.55), _Mtp)
        provider._shadow_markov_proposal = _proposal(
            confidence=0.55,
            tokens=(7, 8, 9),
        )
        mtp = _proposal(confidence=0.999, tokens=(7, 8, 9))

        fused = provider._fuse_mtp_consensus(mtp)

        self.assertIs(fused, mtp)
        metrics = provider.metrics()
        self.assertEqual(metrics.consensus_confidence_gain, 0.0)
        self.assertEqual(metrics.last_consensus_confidence_gain, 0.0)
        provider.close()

    def test_saturated_mtp_confidence_is_not_counted_as_atlas_help(self) -> None:
        markov = _Markov(
            0.01,
            tokens=(31, 32, 33),
            atlas_scores=(0.8, 0.8, 0.8),
        )
        provider = Qwen38MarkovMtpDraftProvider(markov, _Mtp)
        provider._shadow_markov_proposal = _proposal(
            confidence=0.01,
            tokens=(31, 32, 33),
        )
        mtp = _proposal(confidence=0.999, tokens=(7, 8, 9))

        fused = provider._fuse_mtp_consensus(mtp)

        self.assertIs(fused, mtp)
        metrics = provider.metrics()
        self.assertEqual(metrics.atlas_consensus_rounds, 0)
        self.assertEqual(metrics.atlas_consensus_tokens, 0)
        self.assertEqual(metrics.atlas_consensus_confidence_gain, 0.0)
        provider.close()

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
        self.assertEqual(metrics.schema, "immer.qwen3.8-markov-mtp-hybrid-provider/v27")
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
            provider._select_markov = lambda _proposal: False
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
