from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from safetensors.torch import save_file
import torch

from immer.knowledge import Streamer
from immer.runtimes.qwen3_8 import (
    FingerprintRollingK4DraftProvider,
    QWEN38_K2_SPECULATIVE_SCHEMA,
    QWEN38_K4_SPECULATIVE_ROUND_SCHEMA,
    QWEN38_K4_SPECULATIVE_SCHEMA,
    Qwen38K2SpeculativeDecoder,
    Qwen38K4SpeculativeDecoder,
    Qwen38NativeHeadCrsa,
    Qwen38SpeculativeError,
    Qwen38WeightPager,
    StreamedQwen38,
    RollingDraftProposal,
)

from test_qwen3_8_model import (
    _assert_layer_states_equal,
    _native_tiny_config,
    _tiny_config,
    _tiny_weights,
)


class _Pairs:
    def __init__(self, *pairs: tuple[int, int]) -> None:
        self.pairs = list(pairs)
        self.histories: list[tuple[int, ...]] = []

    def __call__(self, history: tuple[int, ...]) -> tuple[int, int]:
        self.histories.append(history)
        if not self.pairs:
            raise AssertionError("draft provider was called too many times")
        return self.pairs.pop(0)


class _Quads:
    def __init__(self, *blocks: tuple[int, int, int, int]) -> None:
        self.blocks = list(blocks)
        self.histories: list[tuple[int, ...]] = []

    def __call__(self, history: tuple[int, ...]) -> tuple[int, int, int, int]:
        self.histories.append(history)
        if not self.blocks:
            raise AssertionError("K=4 draft provider was called too many times")
        return self.blocks.pop(0)


class _RollingFromTokens:
    def __init__(
        self,
        prompt: tuple[int, ...],
        tokens: tuple[int, ...],
        *,
        accepted_per_wave: int,
        vocab_size: int,
        proposal_width: int = 3,
    ) -> None:
        self.prompt = prompt
        self.tokens = tokens
        self.accepted_per_wave = accepted_per_wave
        self.vocab_size = vocab_size
        self.proposal_width = proposal_width
        self.proposals: list[tuple[int, tuple[int, ...]]] = []
        self.reconciled: list[tuple[int, ...]] = []

    def __call__(self, _history):
        raise AssertionError("rolling provider used legacy __call__")

    def propose_after(
        self, history: tuple[int, ...], known_token: int
    ) -> tuple[int, ...]:
        position = len(history) - len(self.prompt)
        if known_token != self.tokens[position]:
            raise AssertionError("rolling known token differs from target")
        proposal = list(self.tokens[position + 1 : position + 1 + self.proposal_width])
        while len(proposal) < self.proposal_width:
            proposal.append((proposal[-1] + 1) % self.vocab_size)
        if self.accepted_per_wave < self.proposal_width:
            index = self.accepted_per_wave
            proposal[index] = (proposal[index] + 1) % self.vocab_size
        block = tuple(proposal)
        self.proposals.append((known_token, block))
        return block

    def reconcile_prefix(self, history: tuple[int, ...]) -> None:
        self.reconciled.append(history)


class _AdaptiveRollingFromTokens(_RollingFromTokens):
    def __init__(self, *args, confidence: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.confidence = confidence

    def propose_round(
        self,
        history: tuple[int, ...],
        known_token: int,
    ) -> RollingDraftProposal:
        proposal = self.propose_after(history, known_token)
        return RollingDraftProposal.build(
            proposal,
            (self.confidence,) * len(proposal),
            (0.0,) * len(proposal),
            request_window_ceiling=self.proposal_width + 1,
            provider_abi="test-adaptive-provider/v1",
        )


class _StateRollingFromTokens(_RollingFromTokens):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.begin_hidden: torch.Tensor | None = None
        self.proposal_hidden: list[torch.Tensor] = []
        self.reconcile_hidden: list[torch.Tensor] = []

    def begin_request_state(
        self,
        history: tuple[int, ...],
        target_hidden: torch.Tensor,
    ) -> None:
        if history != self.prompt:
            raise AssertionError("state provider received another prompt")
        self.begin_hidden = target_hidden.detach().clone()

    def propose_after_state(
        self,
        history: tuple[int, ...],
        known_token: int,
        target_hidden: torch.Tensor,
    ) -> tuple[int, ...]:
        self.proposal_hidden.append(target_hidden.detach().clone())
        return super().propose_after(history, known_token)

    def reconcile_prefix_state(
        self,
        history: tuple[int, ...],
        target_hidden: torch.Tensor,
    ) -> None:
        self.reconcile_hidden.append(target_hidden.detach().clone())
        super().reconcile_prefix(history)


class _MutatingStateRollingFromTokens(_StateRollingFromTokens):
    def __init__(self, *args, mutate_at: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mutate_at = mutate_at

    def begin_request_state(self, history, target_hidden) -> None:
        super().begin_request_state(history, target_hidden)
        if self.mutate_at == "begin":
            target_hidden.data.zero_()

    def propose_after_state(self, history, known_token, target_hidden):
        proposal = super().propose_after_state(history, known_token, target_hidden)
        if self.mutate_at == "proposal":
            target_hidden.data.zero_()
        return proposal

    def reconcile_prefix_state(self, history, target_hidden) -> None:
        super().reconcile_prefix_state(history, target_hidden)
        if self.mutate_at == "reconcile":
            target_hidden.data.zero_()


class _MutatingPair(Sequence[int]):
    def __init__(self, model: StreamedQwen38, pair: tuple[int, int]) -> None:
        self.model = model
        self.pair = pair

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> int:
        if not 0 <= index < 2:
            raise IndexError(index)
        if index == 0:
            self.model._layer_states[0].recurrent.zero_()
        return self.pair[index]


class _MutatingQuad(Sequence[int]):
    def __init__(self, model: StreamedQwen38, block: tuple[int, ...]) -> None:
        self.model = model
        self.block = block

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> int:
        if not 0 <= index < 4:
            raise IndexError(index)
        if index == 2:
            self.model._layer_states[0].recurrent.data.zero_()
        return self.block[index]


class Qwen38SpeculativeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = _tiny_config()
        save_file(_tiny_weights(self.config), self.root / "model.safetensors")
        self.source = Streamer.from_local(self.root, budget_mb=1000, use_cache=False)
        self.pagers: list[Qwen38WeightPager] = []

    def tearDown(self) -> None:
        for pager in self.pagers:
            pager.close()
        self.source.close()
        self.temporary.cleanup()

    def _model(self, **kwargs) -> StreamedQwen38:
        max_seq_len = kwargs.pop("max_seq_len", 16)
        pager = Qwen38WeightPager(
            self.source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=2 * 1024**2,
        )
        self.pagers.append(pager)
        return StreamedQwen38(
            self.config,
            pager,
            max_batch_size=1,
            max_seq_len=max_seq_len,
            **kwargs,
        )

    def _baseline(
        self, *, count: int, eos: tuple[int, ...] = ()
    ) -> tuple[StreamedQwen38, tuple[int, ...], object]:
        model = self._model()
        tokens, evidence = model.generate_greedy(
            [[1, 4]],
            max_new_tokens=count,
            eos_token_ids=eos,
            head_block_rows=7,
        )
        return model, tokens, evidence

    def _assert_state_equal(self, left: StreamedQwen38, right: StreamedQwen38) -> None:
        self.assertEqual(left.next_position, right.next_position)
        self.assertEqual(left.state_batch_size, right.state_batch_size)
        self.assertEqual(left.state_bytes, right.state_bytes)
        _assert_layer_states_equal(self, left._layer_states, right._layer_states)
        self.assertIsNone(left._graft_history)
        self.assertIsNone(right._graft_history)

    def test_full_acceptance_is_exact_and_saves_source_reads(self) -> None:
        baseline, tokens, baseline_evidence = self._baseline(count=2)
        provider = _Pairs((tokens[0], tokens[1]))
        candidate = self._model()

        result = Qwen38K2SpeculativeDecoder(candidate, provider).generate(
            [[1, 4]], max_new_tokens=2, head_block_rows=7
        )

        self.assertEqual(result.token_ids, tokens)
        self._assert_state_equal(candidate, baseline)
        self.assertEqual(result.evidence.schema, QWEN38_K2_SPECULATIVE_SCHEMA)
        self.assertEqual(result.evidence.prefill_forward_passes, 1)
        self.assertEqual(result.evidence.forward_passes, 2)
        self.assertEqual(result.evidence.head_scans, 1)
        self.assertEqual(result.evidence.accepted_draft_tokens, 2)
        self.assertEqual(len(result.evidence.rounds), 1)
        row = result.evidence.rounds[0]
        self.assertEqual(row.replay_kind, "commit-k2")
        self.assertEqual(row.accepted_prefix_length, 2)
        self.assertEqual(row.forward_passes, 1)
        self.assertEqual(row.head_scans, 1)
        self.assertGreater(row.provider_guard_bytes, 0)
        self.assertGreater(row.provider_guard_seconds, 0.0)
        self.assertLessEqual(row.provider_guard_seconds, row.seconds)
        self.assertEqual(
            result.evidence.provider_guard_bytes,
            row.provider_guard_bytes,
        )
        self.assertEqual(
            result.evidence.provider_guard_seconds,
            row.provider_guard_seconds,
        )
        self.assertIn("provider_guard_seconds", result.evidence.to_dict())
        self.assertEqual(provider.histories, [(1, 4)])
        self.assertLess(
            result.evidence.source_body_bytes,
            baseline_evidence.source_body_bytes,
        )
        self.assertEqual(result.evidence.linear_calls, baseline_evidence.linear_calls)
        with self.assertRaises(FrozenInstanceError):
            row.forward_passes = 4  # type: ignore[misc]

    def test_mismatch_zero_falls_back_then_finishes_single(self) -> None:
        baseline, tokens, _evidence = self._baseline(count=2)
        wrong = (tokens[0] + 1) % self.config.vocab_size
        provider = _Pairs((wrong, tokens[1]))
        candidate = self._model()

        result = Qwen38K2SpeculativeDecoder(candidate, provider).generate(
            [1, 4], max_new_tokens=2, head_block_rows=7
        )

        self.assertEqual(result.token_ids, tokens)
        self._assert_state_equal(candidate, baseline)
        self.assertEqual(
            [row.replay_kind for row in result.evidence.rounds],
            ["mismatch0-decode", "remaining-single"],
        )
        self.assertEqual(
            [row.accepted_prefix_length for row in result.evidence.rounds],
            [0, 0],
        )
        self.assertEqual(result.evidence.forward_passes, 4)
        self.assertEqual(result.evidence.head_scans, 2)

    def test_mismatch_one_restages_the_verified_prefix(self) -> None:
        baseline, tokens, _evidence = self._baseline(count=2)
        wrong = (tokens[1] + 1) % self.config.vocab_size
        provider = _Pairs((tokens[0], wrong))
        candidate = self._model()

        result = Qwen38K2SpeculativeDecoder(candidate, provider).generate(
            [[1, 4]], max_new_tokens=2, head_block_rows=7
        )

        self.assertEqual(result.token_ids, tokens)
        self._assert_state_equal(candidate, baseline)
        row = result.evidence.rounds[0]
        self.assertEqual(row.replay_kind, "mismatch1-restage")
        self.assertEqual(row.accepted_prefix_length, 1)
        self.assertEqual(row.emitted_token_ids, tokens)
        self.assertEqual(result.evidence.forward_passes, 3)
        self.assertEqual(result.evidence.head_scans, 1)

    def test_eos_at_zero_never_commits_the_post_eos_draft(self) -> None:
        probe, first_two, _probe_evidence = self._baseline(count=2)
        eos = (first_two[0],)
        baseline, stopped, _evidence = self._baseline(count=2, eos=eos)
        provider = _Pairs((first_two[0], first_two[1]))
        candidate = self._model()

        result = Qwen38K2SpeculativeDecoder(candidate, provider).generate(
            [[1, 4]],
            max_new_tokens=2,
            eos_token_ids=eos,
            head_block_rows=7,
        )

        self.assertEqual(stopped, first_two[:1])
        self.assertEqual(result.token_ids, stopped)
        self._assert_state_equal(candidate, baseline)
        row = result.evidence.rounds[0]
        self.assertEqual(row.replay_kind, "eos0-decode")
        self.assertEqual(row.accepted_prefix_length, 1)
        self.assertEqual(row.emitted_token_ids, first_two[:1])
        self.assertTrue(row.stopped_on_eos)
        self.assertEqual(candidate.next_position, 3)
        self.assertEqual(probe.next_position, 4)

    def test_eos_at_one_commits_the_exact_pair(self) -> None:
        _probe, tokens, _probe_evidence = self._baseline(count=2)
        self.assertNotEqual(tokens[0], tokens[1])
        eos = (tokens[1],)
        baseline, stopped, _evidence = self._baseline(count=2, eos=eos)
        provider = _Pairs((tokens[0], tokens[1]))
        candidate = self._model()

        result = Qwen38K2SpeculativeDecoder(candidate, provider).generate(
            [[1, 4]],
            max_new_tokens=2,
            eos_token_ids=eos,
            head_block_rows=7,
        )

        self.assertEqual(stopped, tokens)
        self.assertEqual(result.token_ids, stopped)
        self._assert_state_equal(candidate, baseline)
        row = result.evidence.rounds[0]
        self.assertEqual(row.replay_kind, "commit-k2")
        self.assertEqual(row.accepted_prefix_length, 2)
        self.assertTrue(row.stopped_on_eos)

    def test_odd_limit_uses_one_final_normal_greedy_step(self) -> None:
        baseline, tokens, _evidence = self._baseline(count=3)
        provider = _Pairs((tokens[0], tokens[1]))
        candidate = self._model()

        result = Qwen38K2SpeculativeDecoder(candidate, provider).generate(
            [[1, 4]], max_new_tokens=3, head_block_rows=7
        )

        self.assertEqual(result.token_ids, tokens)
        self._assert_state_equal(candidate, baseline)
        self.assertEqual(
            [row.replay_kind for row in result.evidence.rounds],
            ["commit-k2", "remaining-single"],
        )
        self.assertEqual(result.evidence.forward_passes, 3)
        self.assertEqual(result.evidence.head_scans, 2)
        self.assertEqual(len(provider.histories), 1)
        self.assertEqual(result.evidence.rounds[-1].provider_guard_bytes, 0)
        self.assertEqual(result.evidence.rounds[-1].provider_guard_seconds, 0.0)

    def test_two_accepted_rounds_receive_only_committed_history(self) -> None:
        baseline, tokens, baseline_evidence = self._baseline(count=4)
        provider = _Pairs(
            (tokens[0], tokens[1]),
            (tokens[2], tokens[3]),
        )
        candidate = self._model()

        result = Qwen38K2SpeculativeDecoder(candidate, provider).generate(
            [[1, 4]], max_new_tokens=4, head_block_rows=7
        )

        self.assertEqual(result.token_ids, tokens)
        self._assert_state_equal(candidate, baseline)
        self.assertEqual(
            provider.histories,
            [(1, 4), (1, 4, tokens[0], tokens[1])],
        )
        self.assertEqual(
            [row.replay_kind for row in result.evidence.rounds],
            ["commit-k2", "commit-k2"],
        )
        self.assertEqual(result.evidence.forward_passes, 3)
        self.assertEqual(result.evidence.head_scans, 2)
        self.assertLess(
            result.evidence.source_body_bytes,
            baseline_evidence.source_body_bytes,
        )

    def test_invalid_provider_output_fails_before_staging(self) -> None:
        with self.assertRaisesRegex(TypeError, "draft_provider"):
            Qwen38K2SpeculativeDecoder(self._model(), None)  # type: ignore[arg-type]

        invalid = (
            (1,),
            (1, 2, 3),
            "12",
            (True, 2),
            (-1, 2),
            (self.config.vocab_size, 2),
        )
        for proposal in invalid:
            with self.subTest(proposal=proposal):
                model = self._model()
                decoder = Qwen38K2SpeculativeDecoder(
                    model, lambda _history, value=proposal: value
                )
                with self.assertRaises(Qwen38SpeculativeError):
                    decoder.generate([[1, 4]], max_new_tokens=2, head_block_rows=7)
                self.assertEqual(model.next_position, 2)
                self.assertFalse(model.state_poisoned)
                stage = model.stage_continuation_block([[9, 7]])
                model.discard_continuation_block(stage)

        model = self._model()

        def fail(_history):
            raise LookupError("provider boom")

        with self.assertRaisesRegex(Qwen38SpeculativeError, "provider boom"):
            Qwen38K2SpeculativeDecoder(model, fail).generate(
                [[1, 4]], max_new_tokens=2, head_block_rows=7
            )
        self.assertEqual(model.next_position, 2)

    def test_head_failure_discards_stage_and_preserves_prefill_state(self) -> None:
        _baseline, tokens, _evidence = self._baseline(count=2)
        model = self._model()
        decoder = Qwen38K2SpeculativeDecoder(model, _Pairs((tokens[0], tokens[1])))

        with mock.patch.object(
            model.pager, "topk_logits", side_effect=RuntimeError("head failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "head failure"):
                decoder.generate([[1, 4]], max_new_tokens=2, head_block_rows=7)

        self.assertEqual(model.next_position, 2)
        self.assertFalse(model.state_poisoned)
        stage = model.stage_continuation_block([[tokens[0], tokens[1]]])
        model.discard_continuation_block(stage)

    def test_provider_state_mutation_fails_closed_before_staging(self) -> None:
        _baseline, tokens, _evidence = self._baseline(count=2)
        model = self._model()

        def mutate(_history):
            model._layer_states[0].recurrent.zero_()
            return tokens

        with self.assertRaisesRegex(
            Qwen38SpeculativeError, "changed target model state"
        ):
            Qwen38K2SpeculativeDecoder(model, mutate).generate(
                [[1, 4]], max_new_tokens=2, head_block_rows=7
            )
        self.assertEqual(model.next_position, 0)
        self.assertEqual(model.state_bytes, 0)
        self.assertFalse(model.state_poisoned)

        lazy_model = self._model()

        def lazy(_history):
            return _MutatingPair(lazy_model, (tokens[0], tokens[1]))

        with self.assertRaisesRegex(
            Qwen38SpeculativeError, "changed target model state"
        ):
            Qwen38K2SpeculativeDecoder(lazy_model, lazy).generate(
                [[1, 4]], max_new_tokens=2, head_block_rows=7
            )
        self.assertEqual(lazy_model.next_position, 0)
        self.assertEqual(lazy_model.state_bytes, 0)

        data_model = self._model()

        def mutate_without_version(_history):
            data_model._layer_states[0].recurrent.data.zero_()
            return tokens

        with self.assertRaisesRegex(
            Qwen38SpeculativeError, "changed target model state"
        ):
            Qwen38K2SpeculativeDecoder(data_model, mutate_without_version).generate(
                [[1, 4]], max_new_tokens=2, head_block_rows=7
            )
        self.assertEqual(data_model.next_position, 0)
        self.assertEqual(data_model.state_bytes, 0)

    def test_speculative_decoder_exposes_no_live_progress_hooks(self) -> None:
        _baseline, tokens, _evidence = self._baseline(count=2)
        model = self._model()
        decoder = Qwen38K2SpeculativeDecoder(model, _Pairs((tokens[0], tokens[1])))
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
            decoder.generate(
                [[1, 4]],
                max_new_tokens=2,
                head_block_rows=7,
                progress=lambda _event: None,  # type: ignore[call-arg]
            )
        self.assertEqual(model.next_position, 0)
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
            decoder.generate(
                [[1, 4]],
                max_new_tokens=2,
                head_block_rows=7,
                head_progress=lambda _event: None,  # type: ignore[call-arg]
            )
        self.assertEqual(model.next_position, 0)

    def test_native_full_acceptance_matches_greedy_state(self) -> None:
        native_root = self.root / "native"
        native_root.mkdir()
        config = _native_tiny_config()
        save_file(_tiny_weights(config), native_root / "model.safetensors")
        source = Streamer.from_local(native_root, budget_mb=1000, use_cache=False)
        pagers = [
            Qwen38WeightPager(
                source,
                device="cpu",
                compute_dtype="bfloat16",
                max_resident_bytes=2 * 1024**2,
            )
            for _ in range(2)
        ]
        observed = [[], []]
        models = [
            StreamedQwen38(
                config,
                pager,
                native_head_crsa=Qwen38NativeHeadCrsa(alpha=0.1),
                native_head_crsa_observer=rows.append,
                max_batch_size=1,
                max_seq_len=16,
            )
            for pager, rows in zip(pagers, observed, strict=True)
        ]
        baseline, candidate = models
        try:
            tokens, _baseline_evidence = baseline.generate_greedy(
                [[1, 4]], max_new_tokens=2, head_block_rows=7
            )
            result = Qwen38K2SpeculativeDecoder(
                candidate, _Pairs((tokens[0], tokens[1]))
            ).generate([[1, 4]], max_new_tokens=2, head_block_rows=7)

            self.assertEqual(result.token_ids, tokens)
            self.assertEqual(candidate.next_position, baseline.next_position)
            _assert_layer_states_equal(
                self, candidate._layer_states, baseline._layer_states
            )
            self.assertEqual(
                [
                    (row.query_start, row.query_length, row.history_length_after)
                    for row in observed[1]
                ],
                [(0, 2, 2), (2, 1, 3), (3, 1, 4)],
            )
        finally:
            for pager in pagers:
                pager.close()
            source.close()

    def test_k4_prefix_zero_through_four_replay_is_exact(self) -> None:
        expected_kinds = (
            "mismatch0-decode",
            "mismatch1-restage",
            "mismatch2-restage",
            "mismatch3-restage",
            "commit-k4",
        )
        for accepted in range(5):
            with self.subTest(accepted=accepted):
                baseline, tokens, _baseline_evidence = self._baseline(count=4)
                proposal = list(tokens)
                if accepted < 4:
                    proposal[accepted] = (
                        proposal[accepted] + 1
                    ) % self.config.vocab_size
                provider = _Quads(tuple(proposal))  # type: ignore[arg-type]
                candidate = self._model()

                result = Qwen38K4SpeculativeDecoder(candidate, provider).generate(
                    [[1, 4]], max_new_tokens=4, head_block_rows=7
                )

                self.assertEqual(result.token_ids, tokens)
                self._assert_state_equal(candidate, baseline)
                first = result.evidence.rounds[0]
                self.assertEqual(first.replay_kind, expected_kinds[accepted])
                self.assertEqual(first.accepted_prefix_length, accepted)
                self.assertEqual(first.head_scans, 1)
                self.assertEqual(first.forward_passes, 1 if accepted == 4 else 2)
                self.assertEqual(provider.histories, [(1, 4)])

    def test_k4_full_acceptance_uses_one_stage_and_one_head_scan(self) -> None:
        baseline, tokens, baseline_evidence = self._baseline(count=4)
        candidate = self._model()
        provider = _Quads(tokens)  # type: ignore[arg-type]

        with (
            mock.patch.object(
                candidate,
                "stage_continuation_block",
                wraps=candidate.stage_continuation_block,
            ) as stage_call,
            mock.patch.object(
                candidate.pager,
                "topk_logits",
                wraps=candidate.pager.topk_logits,
            ) as head_call,
        ):
            result = Qwen38K4SpeculativeDecoder(candidate, provider).generate(
                [[1, 4]], max_new_tokens=4, head_block_rows=7
            )

        self.assertEqual(result.token_ids, tokens)
        self._assert_state_equal(candidate, baseline)
        self.assertEqual(stage_call.call_count, 1)
        self.assertEqual(head_call.call_count, 1)
        self.assertEqual(result.evidence.schema, QWEN38_K4_SPECULATIVE_SCHEMA)
        self.assertEqual(
            result.evidence.rounds[0].schema,
            QWEN38_K4_SPECULATIVE_ROUND_SCHEMA,
        )
        self.assertEqual(result.evidence.forward_passes, 2)
        self.assertEqual(result.evidence.head_scans, 1)
        self.assertEqual(result.evidence.accepted_draft_tokens, 4)
        self.assertLess(
            result.evidence.source_body_bytes,
            baseline_evidence.source_body_bytes,
        )

    def test_k4_eos_at_every_position_commits_no_post_eos_state(self) -> None:
        _probe, tokens, _probe_evidence = self._baseline(count=4)
        self.assertEqual(len(set(tokens)), 4)
        expected_kinds = (
            "eos0-decode",
            "eos1-restage",
            "eos2-restage",
            "eos3-commit-k4",
        )
        for eos_index in range(4):
            with self.subTest(eos_index=eos_index):
                eos = (tokens[eos_index],)
                baseline, stopped, _evidence = self._baseline(count=4, eos=eos)
                candidate = self._model()
                result = Qwen38K4SpeculativeDecoder(
                    candidate,
                    _Quads(tokens),  # type: ignore[arg-type]
                ).generate(
                    [[1, 4]],
                    max_new_tokens=4,
                    eos_token_ids=eos,
                    head_block_rows=7,
                )

                self.assertEqual(result.token_ids, tokens[: eos_index + 1])
                self.assertEqual(result.token_ids, stopped)
                self._assert_state_equal(candidate, baseline)
                row = result.evidence.rounds[0]
                self.assertEqual(row.replay_kind, expected_kinds[eos_index])
                self.assertEqual(row.accepted_prefix_length, eos_index + 1)
                self.assertEqual(candidate.next_position, 3 + eos_index)
                self.assertTrue(row.stopped_on_eos)

    def test_k4_mismatched_fourth_eos_is_restaged(self) -> None:
        baseline, tokens, _evidence = self._baseline(
            count=4,
            eos=(),
        )
        wrong = (tokens[3] + 1) % self.config.vocab_size
        eos_baseline, stopped, _evidence = self._baseline(
            count=4,
            eos=(tokens[3],),
        )
        candidate = self._model()
        result = Qwen38K4SpeculativeDecoder(
            candidate,
            _Quads((tokens[0], tokens[1], tokens[2], wrong)),
        ).generate(
            [[1, 4]],
            max_new_tokens=4,
            eos_token_ids=(tokens[3],),
            head_block_rows=7,
        )

        self.assertEqual(result.token_ids, stopped)
        self.assertEqual(result.token_ids, tokens)
        self._assert_state_equal(candidate, eos_baseline)
        self.assertEqual(result.evidence.rounds[0].replay_kind, "eos3-restage")
        self.assertEqual(result.evidence.rounds[0].accepted_prefix_length, 3)
        self.assertEqual(baseline.next_position, eos_baseline.next_position)

    def test_k4_one_shot_discards_terminal_eos_without_replay(self) -> None:
        _baseline, tokens, _evidence = self._baseline(count=4)
        candidate = self._model()
        result = Qwen38K4SpeculativeDecoder(
            candidate,
            _Quads(tokens),  # type: ignore[arg-type]
        ).generate(
            [[1, 4]],
            max_new_tokens=4,
            eos_token_ids=(tokens[2],),
            head_block_rows=7,
            retain_final_state=False,
        )

        self.assertEqual(result.token_ids, tokens[:3])
        self.assertEqual(result.evidence.forward_passes, 2)
        self.assertEqual(result.evidence.rounds[0].replay_kind, "eos-uncommitted")
        self.assertFalse(result.evidence.final_state_committed)
        self.assertEqual(candidate.next_position, 2)
        self.assertIsNone(candidate._pending_block_stage)

    def test_k4_one_shot_mismatch3_returns_correction_without_replay(self) -> None:
        _baseline, tokens, _evidence = self._baseline(count=4)
        proposal = (*tokens[:3], (tokens[3] + 1) % self.config.vocab_size)
        candidate = self._model()
        result = Qwen38K4SpeculativeDecoder(
            candidate,
            _Quads(proposal),
        ).generate(
            [[1, 4]],
            max_new_tokens=4,
            head_block_rows=7,
            retain_final_state=False,
        )

        self.assertEqual(result.token_ids, tokens)
        self.assertEqual(result.evidence.forward_passes, 2)
        self.assertEqual(
            result.evidence.rounds[0].replay_kind,
            "mismatch3-uncommitted",
        )
        self.assertFalse(result.evidence.final_state_committed)
        self.assertEqual(candidate.next_position, 2)
        self.assertIsNone(candidate._pending_block_stage)

    def test_k4_terminal_tails_one_through_three_never_call_provider(self) -> None:
        for tail in range(1, 4):
            with self.subTest(tail=tail):
                baseline, tokens, _evidence = self._baseline(count=tail)
                candidate = self._model()

                def forbidden(_history):
                    raise AssertionError("terminal tail invoked the provider")

                result = Qwen38K4SpeculativeDecoder(candidate, forbidden).generate(
                    [[1, 4]], max_new_tokens=tail, head_block_rows=7
                )

                self.assertEqual(result.token_ids, tokens)
                self._assert_state_equal(candidate, baseline)
                self.assertEqual(
                    [row.replay_kind for row in result.evidence.rounds],
                    ["terminal-single"] * tail,
                )
                self.assertEqual(result.evidence.forward_passes, 1 + tail)
                self.assertEqual(result.evidence.head_scans, tail)
                self.assertEqual(result.evidence.provider_guard_bytes, 0)

    def test_k4_one_shot_tail_skips_only_the_unused_final_decode(self) -> None:
        _baseline, tokens, _evidence = self._baseline(count=3)
        candidate = self._model()

        def forbidden(_history):
            raise AssertionError("terminal tail invoked the provider")

        result = Qwen38K4SpeculativeDecoder(candidate, forbidden).generate(
            [[1, 4]],
            max_new_tokens=3,
            head_block_rows=7,
            retain_final_state=False,
        )

        self.assertEqual(result.token_ids, tokens)
        self.assertEqual(result.evidence.forward_passes, 3)
        self.assertEqual(
            [row.replay_kind for row in result.evidence.rounds],
            ["terminal-single", "terminal-single", "terminal-single-uncommitted"],
        )
        self.assertFalse(result.evidence.final_state_committed)
        self.assertEqual(candidate.next_position, 4)

    def test_k4_on_tokens_reports_cumulative_output_after_each_round(self) -> None:
        baseline, tokens, _evidence = self._baseline(count=5)
        candidate = self._model()
        updates: list[tuple[int, ...]] = []

        result = Qwen38K4SpeculativeDecoder(
            candidate,
            _Quads(tokens[:4]),  # type: ignore[arg-type]
        ).generate(
            [[1, 4]],
            max_new_tokens=5,
            head_block_rows=7,
            on_tokens=updates.append,
        )

        self.assertEqual(result.token_ids, tokens)
        self._assert_state_equal(candidate, baseline)
        self.assertEqual(updates, [tokens[:4], tokens])
        self.assertTrue(all(isinstance(update, tuple) for update in updates))

    def test_rolling_on_tokens_reports_once_per_multi_token_round(self) -> None:
        prompt = (1, 4)
        baseline = self._model()
        tokens, _evidence = baseline.generate_greedy(
            [prompt], max_new_tokens=8, head_block_rows=7
        )
        provider = _RollingFromTokens(
            prompt,
            tokens,
            accepted_per_wave=3,
            vocab_size=self.config.vocab_size,
        )
        candidate = self._model()
        updates: list[tuple[int, ...]] = []

        result = Qwen38K4SpeculativeDecoder(candidate, provider).generate_rolling(
            [prompt],
            max_new_tokens=8,
            head_block_rows=7,
            on_tokens=updates.append,
        )

        self.assertEqual(result.token_ids, tokens)
        self._assert_state_equal(candidate, baseline)
        self.assertEqual(updates, [tokens[:4], tokens])

    def test_rolling_state_provider_receives_target_hidden_rows(self) -> None:
        prompt = (1, 4)
        baseline = self._model()
        tokens, _evidence = baseline.generate_greedy(
            [prompt], max_new_tokens=8, head_block_rows=7
        )
        provider = _StateRollingFromTokens(
            prompt,
            tokens,
            accepted_per_wave=3,
            vocab_size=self.config.vocab_size,
        )
        candidate = self._model()

        result = Qwen38K4SpeculativeDecoder(
            candidate,
            provider,
        ).generate_rolling(
            [prompt],
            max_new_tokens=8,
            head_block_rows=7,
        )

        self.assertEqual(result.token_ids, tokens)
        self._assert_state_equal(candidate, baseline)
        self.assertIsNotNone(provider.begin_hidden)
        assert provider.begin_hidden is not None
        self.assertEqual(
            tuple(provider.begin_hidden.shape), (1, len(prompt), self.config.dim)
        )
        self.assertEqual(len(provider.proposal_hidden), 2)
        self.assertEqual(
            tuple(provider.proposal_hidden[0].shape), (1, 1, self.config.dim)
        )
        self.assertEqual(len(provider.reconcile_hidden), 2)
        self.assertTrue(
            all(
                tuple(value.shape) == (1, 4, self.config.dim)
                for value in provider.reconcile_hidden
            )
        )

    def test_rolling_markov_provider_continues_from_restored_prefix(self) -> None:
        prompt = (1, 4)
        baseline = self._model()
        tokens, _evidence = baseline.generate_greedy(
            [prompt], max_new_tokens=4, head_block_rows=7
        )
        candidate = self._model()
        candidate.prefill([[prompt[0]]], reset=True)
        provider = _RollingFromTokens(
            prompt,
            tokens,
            accepted_per_wave=3,
            vocab_size=self.config.vocab_size,
        )

        result = Qwen38K4SpeculativeDecoder(
            candidate,
            provider,
        ).generate_rolling(
            [prompt],
            max_new_tokens=4,
            restored_prefix_length=1,
            head_block_rows=7,
        )

        self.assertEqual(result.token_ids, tokens)
        self._assert_state_equal(candidate, baseline)

    def test_real_markov_continues_from_an_exact_anchor_seed(self) -> None:
        prompt = (1, 4)
        reference = self._model()
        expected, _evidence = reference.generate_greedy(
            [prompt], max_new_tokens=4, head_block_rows=7
        )
        candidate = self._model()
        prompt_hidden, _ = candidate.prefill([prompt], reset=True)
        restored_seed = prompt_hidden[:, -1:].detach().clone()

        class ExactRestoredMarkov(FingerprintRollingK4DraftProvider):
            def propose_round(self, history, known_token):
                proposal = super().propose_round(history, known_token)
                position = len(history) - len(prompt)
                tokens = (expected[position + 1], *proposal.token_ids[1:])
                self._pending_planner = "beam"
                return RollingDraftProposal.build(
                    tokens,
                    (0.10,) * len(tokens),
                    (0.0,) * len(tokens),
                    request_window_ceiling=len(tokens) + 1,
                    provider_abi=proposal.provider_abi,
                )

        provider = ExactRestoredMarkov(
            vocab_size=self.config.vocab_size,
            proposal_width=3,
        )
        result = Qwen38K4SpeculativeDecoder(
            candidate,
            provider,
            window_size=4,
            adaptive_round_windows=True,
        ).generate_rolling(
            [prompt],
            max_new_tokens=4,
            restored_prefix_length=len(prompt),
            restored_seed_hidden=restored_seed,
            head_block_rows=7,
        )

        self.assertEqual(result.token_ids, expected)
        self.assertEqual(result.evidence.prefill_forward_passes, 0)
        self.assertEqual(provider._beam_verified_tokens, 3)
        self.assertEqual(provider._beam_accepted_tokens, 3)
        self._assert_state_equal(candidate, reference)
        provider.close()

    def test_rolling_state_provider_cannot_mutate_target_hidden(self) -> None:
        prompt = (1, 4)
        baseline = self._model()
        tokens, _evidence = baseline.generate_greedy(
            [prompt], max_new_tokens=4, head_block_rows=7
        )
        for mutate_at in ("begin", "proposal", "reconcile"):
            with self.subTest(mutate_at=mutate_at):
                provider = _MutatingStateRollingFromTokens(
                    prompt,
                    tokens,
                    accepted_per_wave=3,
                    vocab_size=self.config.vocab_size,
                    mutate_at=mutate_at,
                )
                candidate = self._model()
                with self.assertRaisesRegex(
                    Qwen38SpeculativeError,
                    "mutated its hidden argument",
                ):
                    Qwen38K4SpeculativeDecoder(
                        candidate,
                        provider,
                    ).generate_rolling(
                        [prompt],
                        max_new_tokens=8,
                        head_block_rows=7,
                    )
                self.assertEqual(candidate.next_position, 0)

    def test_k4_on_tokens_reports_terminal_eos_round(self) -> None:
        _baseline, tokens, _evidence = self._baseline(count=4)
        eos_baseline, stopped, _evidence = self._baseline(
            count=4,
            eos=(tokens[2],),
        )
        candidate = self._model()
        updates: list[tuple[int, ...]] = []

        result = Qwen38K4SpeculativeDecoder(
            candidate,
            _Quads(tokens),  # type: ignore[arg-type]
        ).generate(
            [[1, 4]],
            max_new_tokens=4,
            eos_token_ids=(tokens[2],),
            head_block_rows=7,
            on_tokens=updates.append,
        )

        self.assertEqual(result.token_ids, stopped)
        self._assert_state_equal(candidate, eos_baseline)
        self.assertEqual(updates, [stopped])

    def test_k4_on_tokens_rejects_non_callable_for_both_modes(self) -> None:
        for method_name in ("generate", "generate_rolling"):
            with self.subTest(method=method_name):
                candidate = self._model()
                decoder = Qwen38K4SpeculativeDecoder(
                    candidate,
                    lambda _history: (0, 0, 0, 0),
                )
                method = getattr(decoder, method_name)
                with self.assertRaisesRegex(
                    TypeError, "on_tokens must be callable or None"
                ):
                    method(
                        [[1, 4]],
                        max_new_tokens=1,
                        head_block_rows=7,
                        on_tokens=object(),
                    )
                self.assertEqual(candidate.next_position, 0)
                self.assertEqual(candidate.state_bytes, 0)

    def test_k4_on_tokens_failures_propagate_for_both_modes(self) -> None:
        class CallbackFailure(RuntimeError):
            pass

        failure = CallbackFailure("stream consumer failed")

        def fail(_tokens: tuple[int, ...]) -> None:
            raise failure

        for method_name in ("generate", "generate_rolling"):
            with self.subTest(method=method_name):
                candidate = self._model()
                decoder = Qwen38K4SpeculativeDecoder(
                    candidate,
                    lambda _history: (0, 0, 0, 0),
                )
                method = getattr(decoder, method_name)
                with self.assertRaises(CallbackFailure) as raised:
                    method(
                        [[1, 4]],
                        max_new_tokens=1,
                        head_block_rows=7,
                        on_tokens=fail,
                    )
                self.assertIs(raised.exception, failure)

    def test_rolling_k4_matches_greedy_for_every_acceptance_prefix(self) -> None:
        prompt = (1, 4)
        reference = self._model()
        expected_long, _ = reference.generate_greedy(
            [prompt], max_new_tokens=12, head_block_rows=7
        )
        expected = expected_long[:8]
        for accepted in range(4):
            with self.subTest(accepted=accepted):
                baseline = self._model()
                baseline_tokens, _ = baseline.generate_greedy(
                    [prompt], max_new_tokens=8, head_block_rows=7
                )
                provider = _RollingFromTokens(
                    prompt,
                    expected_long,
                    accepted_per_wave=accepted,
                    vocab_size=self.config.vocab_size,
                )
                candidate = self._model()

                result = Qwen38K4SpeculativeDecoder(
                    candidate, provider
                ).generate_rolling(
                    [prompt],
                    max_new_tokens=8,
                    head_block_rows=7,
                )

                self.assertEqual(result.token_ids, expected)
                self.assertEqual(result.token_ids, baseline_tokens)
                self._assert_state_equal(candidate, baseline)
                self.assertTrue(
                    all(row.forward_passes <= 1 for row in result.evidence.rounds)
                )
                self.assertTrue(
                    all(row.head_scans <= 1 for row in result.evidence.rounds)
                )
                self.assertEqual(
                    result.evidence.accepted_draft_tokens,
                    sum(row.accepted_prefix_length for row in result.evidence.rounds),
                )

    def test_rolling_k8_matches_greedy_across_conv_boundary(self) -> None:
        prompt = (1, 4)
        reference = self._model()
        expected, _ = reference.generate_greedy(
            [prompt], max_new_tokens=8, head_block_rows=7
        )
        for accepted in (0, 3, 7):
            with self.subTest(accepted=accepted):
                baseline = self._model()
                baseline.generate_greedy([prompt], max_new_tokens=8, head_block_rows=7)
                provider = _RollingFromTokens(
                    prompt,
                    expected,
                    accepted_per_wave=accepted,
                    vocab_size=self.config.vocab_size,
                    proposal_width=7,
                )
                candidate = self._model()
                result = Qwen38K4SpeculativeDecoder(
                    candidate,
                    provider,
                    window_size=8,
                ).generate_rolling(
                    [prompt],
                    max_new_tokens=8,
                    head_block_rows=7,
                )
                self.assertEqual(result.token_ids, expected)
                self._assert_state_equal(candidate, baseline)
                self.assertEqual(result.evidence.window_size, 8)
                self.assertTrue(
                    all(row.window_size == 8 for row in result.evidence.rounds)
                )

    def test_adaptive_k8_ceiling_uses_round_local_k4_or_k8_without_extra_scan(
        self,
    ) -> None:
        prompt = (1, 4)
        reference = self._model()
        expected, _ = reference.generate_greedy(
            [prompt], max_new_tokens=8, head_block_rows=7
        )
        rows = []
        for confidence, expected_windows in (
            (0.10, (1, 1, 1, 1, 1, 1, 1)),
            (0.99, (8,)),
        ):
            baseline = self._model()
            baseline.generate_greedy([prompt], max_new_tokens=8, head_block_rows=7)
            provider = _AdaptiveRollingFromTokens(
                prompt,
                expected,
                accepted_per_wave=7,
                vocab_size=self.config.vocab_size,
                proposal_width=7,
                confidence=confidence,
            )
            candidate = self._model()

            result = Qwen38K4SpeculativeDecoder(
                candidate,
                provider,
                window_size=8,
                adaptive_round_windows=True,
            ).generate_rolling(
                [prompt],
                max_new_tokens=8,
                head_block_rows=7,
            )

            self.assertEqual(result.token_ids, expected)
            self._assert_state_equal(candidate, baseline)
            self.assertTrue(result.evidence.adaptive_windows)
            self.assertEqual(result.evidence.used_window_sizes, expected_windows)
            self.assertEqual(
                tuple(
                    row.window_size
                    for row in result.evidence.rounds
                    if row.target_token_ids
                ),
                expected_windows,
            )
            self.assertTrue(
                all(
                    len(row.provider_proposed_token_ids) == 7
                    and len(row.proposed_token_ids) == row.window_size - 1
                    and row.round_policy is not None
                    and row.head_scans == 1
                    and row.forward_passes == 1
                    for row in result.evidence.rounds
                    if row.target_token_ids
                )
            )
            first = result.evidence.rounds[0]
            with self.assertRaises(ValueError):
                replace(
                    first,
                    provider_proposed_token_ids=(99,) * 7,
                )
            with self.assertRaises(ValueError):
                replace(result.evidence, adaptive_windows=False)
            rows.append(result.evidence.forward_passes)
        self.assertGreater(rows[0], rows[1])

    def test_adaptive_terminal_budget_stages_only_the_remaining_three_tokens(
        self,
    ) -> None:
        prompt = (1, 4)
        baseline = self._model()
        expected, _ = baseline.generate_greedy(
            [prompt], max_new_tokens=3, head_block_rows=7
        )
        provider = _AdaptiveRollingFromTokens(
            prompt,
            expected,
            accepted_per_wave=7,
            vocab_size=self.config.vocab_size,
            proposal_width=7,
            confidence=0.99,
        )
        candidate = self._model()

        result = Qwen38K4SpeculativeDecoder(
            candidate,
            provider,
            window_size=8,
            adaptive_round_windows=True,
        ).generate_rolling(
            [prompt],
            max_new_tokens=3,
            head_block_rows=7,
        )

        self.assertEqual(result.token_ids, expected)
        self._assert_state_equal(candidate, baseline)
        self.assertEqual(result.evidence.used_window_sizes, (4,))
        row = result.evidence.rounds[0]
        self.assertEqual(row.window_size, 4)
        self.assertEqual(len(row.proposed_token_ids), 3)
        self.assertEqual(len(row.provider_proposed_token_ids), 7)
        self.assertEqual(row.round_policy.selector, "markov-prefix-utility/v2")

    def test_real_markov_verification_matches_the_budget_committed_prefix(
        self,
    ) -> None:
        prompt = (1, 4)
        teacher = self._model()
        teacher_tokens, _ = teacher.generate_greedy(
            [prompt], max_new_tokens=4, head_block_rows=7
        )
        reference = self._model()
        expected, baseline = reference.generate_greedy(
            [prompt], max_new_tokens=3, head_block_rows=7
        )

        class ExactWideMarkov(FingerprintRollingK4DraftProvider):
            def propose_round(self, history, known_token):
                proposal = super().propose_round(history, known_token)
                position = len(history) - len(prompt)
                tokens = tuple(teacher_tokens[position + 1 : position + 4])
                self._pending_proposal = tokens
                self._pending_planner = "beam"
                return RollingDraftProposal.build(
                    tokens,
                    (0.99,) * len(tokens),
                    (0.0,) * len(tokens),
                    request_window_ceiling=len(tokens) + 1,
                    provider_abi=proposal.provider_abi,
                )

        provider = ExactWideMarkov(
            vocab_size=self.config.vocab_size,
            proposal_width=3,
        )
        candidate = self._model()

        result = Qwen38K4SpeculativeDecoder(
            candidate,
            provider,
            window_size=4,
            adaptive_round_windows=True,
        ).generate_rolling(
            [prompt],
            max_new_tokens=3,
            head_block_rows=7,
        )

        self.assertEqual(result.token_ids, expected)
        self.assertEqual(result.evidence.used_window_sizes, (4,))
        self.assertEqual(result.evidence.rounds[0].accepted_prefix_length, 2)
        self.assertEqual(provider._beam_verified_tokens, 2)
        self.assertEqual(provider._beam_accepted_tokens, 2)
        self.assertEqual(provider.metrics().reconcile_calls, 1)
        self._assert_state_equal(candidate, reference)
        provider.close()

    def test_zero_weight_markov_proposal_uses_target_only_window_costs(self) -> None:
        class MarkovAdaptive(_AdaptiveRollingFromTokens):
            def propose_round(self, history, known_token):
                return replace(
                    super().propose_round(history, known_token),
                    provider_abi="immer.qwen3.8-markov-draft-provider/v27",
                )

        prompt = (1, 4)
        baseline = self._model()
        expected, _evidence = baseline.generate_greedy(
            [prompt], max_new_tokens=4, head_block_rows=7
        )
        provider = MarkovAdaptive(
            prompt,
            expected,
            accepted_per_wave=3,
            vocab_size=self.config.vocab_size,
            proposal_width=3,
            confidence=0.30,
        )
        candidate = self._model()

        result = Qwen38K4SpeculativeDecoder(
            candidate,
            provider,
            window_size=4,
            adaptive_round_windows=True,
            round_window_work_costs={1: 1.0, 2: 1.6, 4: 2.8},
        ).generate_rolling(
            [prompt],
            max_new_tokens=4,
            head_block_rows=7,
        )

        self.assertEqual(result.token_ids, expected)
        self.assertEqual(result.evidence.used_window_sizes, (4,))
        self._assert_state_equal(candidate, baseline)

    def test_low_confidence_k1_uses_direct_decode_without_transactional_stage(
        self,
    ) -> None:
        prompt = (1, 4)
        reference = self._model()
        expected, baseline = reference.generate_greedy(
            [prompt], max_new_tokens=4, head_block_rows=7
        )
        provider = _AdaptiveRollingFromTokens(
            prompt,
            expected,
            accepted_per_wave=3,
            vocab_size=self.config.vocab_size,
            proposal_width=3,
            confidence=0.10,
        )
        candidate = self._model()

        with mock.patch.object(
            candidate,
            "stage_continuation_block",
            side_effect=AssertionError("K1 must not stage"),
        ):
            result = Qwen38K4SpeculativeDecoder(
                candidate,
                provider,
                window_size=4,
                adaptive_round_windows=True,
            ).generate_rolling(
                [prompt],
                max_new_tokens=4,
                head_block_rows=7,
            )

        self.assertEqual(result.token_ids, expected)
        self.assertEqual(result.evidence.used_window_sizes, (1, 1, 1))
        self.assertEqual(result.evidence.forward_passes, baseline.forward_passes)
        self._assert_state_equal(candidate, reference)

    def test_real_markov_k1_commits_a_correct_virtual_target_match(self) -> None:
        prompt = (1, 4)
        reference = self._model()
        expected, baseline = reference.generate_greedy(
            [prompt], max_new_tokens=4, head_block_rows=7
        )

        class ExactK1Markov(FingerprintRollingK4DraftProvider):
            def propose_round(self, history, known_token):
                proposal = super().propose_round(history, known_token)
                position = len(history) - len(prompt)
                tokens = (expected[position + 1], *proposal.token_ids[1:])
                self._pending_planner = "beam"
                return RollingDraftProposal.build(
                    tokens,
                    (0.10,) * len(tokens),
                    (0.0,) * len(tokens),
                    request_window_ceiling=len(tokens) + 1,
                    provider_abi=proposal.provider_abi,
                )

        provider = ExactK1Markov(
            vocab_size=self.config.vocab_size,
            proposal_width=3,
        )
        candidate = self._model()

        with mock.patch.object(
            candidate,
            "stage_continuation_block",
            side_effect=AssertionError("real Markov K1 must not stage"),
        ):
            result = Qwen38K4SpeculativeDecoder(
                candidate,
                provider,
                window_size=4,
                adaptive_round_windows=True,
            ).generate_rolling(
                [prompt],
                max_new_tokens=4,
                head_block_rows=7,
            )

        self.assertEqual(result.token_ids, expected)
        self.assertEqual(result.evidence.used_window_sizes, (1, 1, 1))
        self.assertEqual(result.evidence.forward_passes, baseline.forward_passes)
        self._assert_state_equal(candidate, reference)
        self.assertEqual(provider._beam_verified_tokens, 3)
        self.assertEqual(provider._beam_accepted_tokens, 3)
        metrics = provider.metrics()
        self.assertEqual(metrics.reconcile_calls, 3)
        self.assertEqual(metrics.updates, 1)
        provider.close()

    def test_rolling_k16_full_acceptance_uses_one_target_wave(self) -> None:
        prompt = (1, 4)
        baseline = self._model(max_seq_len=32)
        expected, baseline_evidence = baseline.generate_greedy(
            [prompt], max_new_tokens=16, head_block_rows=7
        )
        provider = _RollingFromTokens(
            prompt,
            expected,
            accepted_per_wave=15,
            vocab_size=self.config.vocab_size,
            proposal_width=15,
        )
        candidate = self._model(max_seq_len=32)
        result = Qwen38K4SpeculativeDecoder(
            candidate,
            provider,
            window_size=16,
        ).generate_rolling(
            [prompt],
            max_new_tokens=16,
            head_block_rows=7,
        )
        self.assertEqual(result.token_ids, expected)
        self._assert_state_equal(candidate, baseline)
        self.assertEqual(len(result.evidence.rounds), 1)
        self.assertEqual(result.evidence.forward_passes, 2)
        self.assertLess(
            result.evidence.forward_passes,
            baseline_evidence.forward_passes,
        )

        adaptive_provider = _AdaptiveRollingFromTokens(
            prompt,
            expected,
            accepted_per_wave=15,
            vocab_size=self.config.vocab_size,
            proposal_width=15,
            confidence=0.99,
        )
        adaptive = self._model(max_seq_len=32)
        adaptive_result = Qwen38K4SpeculativeDecoder(
            adaptive,
            adaptive_provider,
            window_size=16,
            adaptive_round_windows=True,
        ).generate_rolling(
            [prompt],
            max_new_tokens=16,
            head_block_rows=7,
        )
        self.assertEqual(adaptive_result.token_ids, expected)
        self._assert_state_equal(adaptive, baseline)
        self.assertEqual(adaptive_result.evidence.used_window_sizes, (16,))
        self.assertEqual(adaptive_result.evidence.forward_passes, 2)

    def test_rolling_k4_one_shot_discards_only_terminal_state(self) -> None:
        prompt = (1, 4)
        reference = self._model()
        expected, _ = reference.generate_greedy(
            [prompt], max_new_tokens=8, head_block_rows=7
        )
        provider = _RollingFromTokens(
            prompt,
            (*expected, 3, 4, 5),
            accepted_per_wave=3,
            vocab_size=self.config.vocab_size,
        )
        candidate = self._model()

        result = Qwen38K4SpeculativeDecoder(candidate, provider).generate_rolling(
            [prompt],
            max_new_tokens=8,
            head_block_rows=7,
            retain_final_state=False,
        )

        self.assertEqual(result.token_ids, expected)
        self.assertFalse(result.evidence.final_state_committed)
        self.assertLess(candidate.next_position, len(prompt) + len(expected))
        self.assertTrue(all(row.forward_passes == 1 for row in result.evidence.rounds))

    def test_k4_receipt_digests_reject_tampering(self) -> None:
        _baseline, tokens, _evidence = self._baseline(count=4)
        result = Qwen38K4SpeculativeDecoder(
            self._model(),
            _Quads(tokens),  # type: ignore[arg-type]
        ).generate([[1, 4]], max_new_tokens=4, head_block_rows=7)
        row = result.evidence.rounds[0]

        self.assertEqual(len(row.evidence_sha256), 64)
        self.assertEqual(len(result.evidence.evidence_sha256), 64)
        self.assertEqual(
            result.evidence.to_dict()["rounds"][0]["evidence_sha256"],
            row.evidence_sha256,
        )
        with self.assertRaisesRegex(ValueError, "digest is invalid"):
            replace(row, evidence_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "digest is invalid"):
            replace(result.evidence, evidence_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "target transition"):
            replace(row, replay_kind="mismatch3-restage")  # type: ignore[arg-type]

    def test_k4_untrusted_proposal_and_reconcile_cannot_mutate_target(self) -> None:
        _baseline, tokens, _evidence = self._baseline(count=4)
        invalid_values = (
            tokens[:3],
            (*tokens, tokens[0]),
            (tokens[0], tokens[1], True, tokens[3]),
            (tokens[0], tokens[1], tokens[2], self.config.vocab_size),
        )
        for proposal in invalid_values:
            with self.subTest(proposal=proposal):
                model = self._model()
                with self.assertRaises(Qwen38SpeculativeError):
                    Qwen38K4SpeculativeDecoder(
                        model,
                        lambda _history, value=proposal: value,
                    ).generate([[1, 4]], max_new_tokens=4, head_block_rows=7)
                self.assertEqual(model.next_position, 2)
                self.assertIsNone(model._pending_block_stage)

        lazy_model = self._model()
        with self.assertRaisesRegex(
            Qwen38SpeculativeError, "changed target model state"
        ):
            Qwen38K4SpeculativeDecoder(
                lazy_model,
                lambda _history: _MutatingQuad(lazy_model, tokens),
            ).generate([[1, 4]], max_new_tokens=4, head_block_rows=7)
        self.assertEqual(lazy_model.next_position, 0)
        self.assertEqual(lazy_model.state_bytes, 0)

        target = self._model()

        class MutatingReconciler:
            def __call__(self, _history):
                return tokens

            def reconcile(self, _history):
                target._layer_states[0].recurrent.data.add_(1.0)

        with self.assertRaisesRegex(
            Qwen38SpeculativeError, "changed target model state while reconciling"
        ):
            Qwen38K4SpeculativeDecoder(target, MutatingReconciler()).generate(
                [[1, 4]], max_new_tokens=4, head_block_rows=7
            )
        self.assertEqual(target.next_position, 0)
        self.assertEqual(target.state_bytes, 0)

    def test_k4_head_failure_discards_pending_target_stage(self) -> None:
        _baseline, tokens, _evidence = self._baseline(count=4)
        target = self._model()
        with mock.patch.object(
            target.pager, "topk_logits", side_effect=RuntimeError("head failure")
        ):
            with self.assertRaisesRegex(RuntimeError, "head failure"):
                Qwen38K4SpeculativeDecoder(
                    target,
                    _Quads(tokens),  # type: ignore[arg-type]
                ).generate([[1, 4]], max_new_tokens=4, head_block_rows=7)
        self.assertEqual(target.next_position, 2)
        self.assertIsNone(target._pending_block_stage)
        stage = target.stage_continuation_block([tokens])
        target.discard_continuation_block(stage)


if __name__ == "__main__":
    unittest.main()
