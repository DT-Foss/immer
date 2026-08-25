from __future__ import annotations

from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from safetensors.torch import save_file

from immer.knowledge import Streamer
from immer.runtimes.qwen3_8 import (
    QWEN38_K2_SPECULATIVE_SCHEMA,
    Qwen38K2SpeculativeDecoder,
    Qwen38NativeHeadCrsa,
    Qwen38SpeculativeError,
    Qwen38WeightPager,
    StreamedQwen38,
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
            max_seq_len=16,
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


if __name__ == "__main__":
    unittest.main()
