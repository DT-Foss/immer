from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file

from immer.knowledge import Streamer
from immer.runtimes.qwen3_8 import (
    QWEN35_K2_DRAFT_PROVIDER_SCHEMA,
    QWEN35_K4_DRAFT_PROVIDER_SCHEMA,
    Qwen35K2DraftProvider,
    Qwen35K2DraftProviderError,
    Qwen35K4DraftProvider,
    Qwen35K4DraftProviderError,
    Qwen38Config,
    Qwen38K2SpeculativeDecoder,
    Qwen38K4SpeculativeDecoder,
    Qwen38SpeculativeError,
    Qwen38WeightPager,
    StreamedQwen38,
)

from test_qwen3_8_model import (
    _assert_layer_states_equal,
    _tiny_config,
    _tiny_config_mapping,
    _tiny_weights,
)


class Qwen35LocalDraftTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = _tiny_config()
        save_file(_tiny_weights(self.config), self.root / "model.safetensors")
        self.source = Streamer.from_local(self.root, budget_mb=1000, use_cache=False)
        self.sources = [self.source]
        self.pagers: list[Qwen38WeightPager] = []

    def tearDown(self) -> None:
        for pager in self.pagers:
            pager.close()
        for source in self.sources:
            source.close()
        self.temporary.cleanup()

    def _model(
        self,
        *,
        config: Qwen38Config | None = None,
        source: Streamer | None = None,
    ) -> StreamedQwen38:
        config = self.config if config is None else config
        source = self.source if source is None else source
        pager = Qwen38WeightPager(
            source,
            device="cpu",
            compute_dtype="bfloat16",
            max_resident_bytes=2 * 1024**2,
        )
        self.pagers.append(pager)
        return StreamedQwen38(
            config,
            pager,
            max_batch_size=1,
            max_seq_len=16,
        )

    def _assert_model_state_equal(
        self, left: StreamedQwen38, right: StreamedQwen38
    ) -> None:
        self.assertEqual(left.next_position, right.next_position)
        self.assertEqual(left.state_batch_size, right.state_batch_size)
        self.assertEqual(left.state_bytes, right.state_bytes)
        _assert_layer_states_equal(self, left._layer_states, right._layer_states)

    def test_identical_local_model_reconciles_every_fully_accepted_round(self) -> None:
        baseline = self._model()
        expected, _evidence = baseline.generate_greedy(
            [[1, 4]], max_new_tokens=4, head_block_rows=7
        )
        target = self._model()
        draft = self._model()
        provider = Qwen35K2DraftProvider(draft, head_block_rows=7)

        result = Qwen38K2SpeculativeDecoder(target, provider).generate(
            [[1, 4]], max_new_tokens=4, head_block_rows=7
        )

        self.assertEqual(result.token_ids, expected)
        self.assertEqual(provider.committed_history, (1, 4, *expected))
        self.assertIsNone(provider.pending_proposal)
        self._assert_model_state_equal(target, baseline)
        self._assert_model_state_equal(draft, baseline)
        metrics = provider.metrics()
        self.assertEqual(metrics.schema, QWEN35_K2_DRAFT_PROVIDER_SCHEMA)
        self.assertEqual(metrics.prefill_calls, 1)
        self.assertEqual(metrics.draft_calls, 2)
        self.assertEqual(metrics.reconcile_calls, 2)
        self.assertEqual(metrics.accepted_prefix_2, 2)
        self.assertEqual(metrics.committed_tokens, 4)
        self.assertFalse(metrics.pending)
        self.assertFalse(metrics.poisoned)
        self.assertGreater(metrics.source_body_bytes, 0)
        self.assertGreater(metrics.linear_calls, 0)

    def test_final_single_round_does_not_run_unneeded_draft_reconciliation(
        self,
    ) -> None:
        baseline = self._model()
        expected, _evidence = baseline.generate_greedy(
            [[1, 4]], max_new_tokens=3, head_block_rows=7
        )
        draft_reference = self._model()
        first_pair, _draft_evidence = draft_reference.generate_greedy(
            [[1, 4]], max_new_tokens=2, head_block_rows=7
        )
        target = self._model()
        draft = self._model()
        provider = Qwen35K2DraftProvider(draft, head_block_rows=7)

        result = Qwen38K2SpeculativeDecoder(target, provider).generate(
            [[1, 4]], max_new_tokens=3, head_block_rows=7
        )

        self.assertEqual(result.token_ids, expected)
        self.assertEqual(first_pair, expected[:2])
        self.assertEqual(provider.committed_history, (1, 4, *expected[:2]))
        self.assertEqual(provider.metrics().accepted_prefix_2, 1)
        self.assertEqual(provider.metrics().reconcile_calls, 1)
        self.assertEqual(provider.metrics().committed_tokens, 2)
        self._assert_model_state_equal(draft, draft_reference)

    def test_single_token_request_never_invokes_local_drafter(self) -> None:
        target = self._model()
        draft = self._model()
        provider = Qwen35K2DraftProvider(draft, head_block_rows=7)

        result = Qwen38K2SpeculativeDecoder(target, provider).generate(
            [[1, 4]], max_new_tokens=1, head_block_rows=7
        )

        self.assertEqual(len(result.token_ids), 1)
        self.assertIsNone(provider.committed_history)
        self.assertEqual(draft.next_position, 0)
        self.assertIsNone(provider.pending_proposal)
        metrics = provider.metrics()
        self.assertEqual(metrics.draft_calls, 0)
        self.assertEqual(metrics.prefill_calls, 0)
        self.assertEqual(metrics.reconcile_calls, 0)

    def test_tied_embedding_checkpoint_uses_embedding_as_output_head(self) -> None:
        mapping = _tiny_config_mapping()
        mapping["tie_word_embeddings"] = True
        config = Qwen38Config.from_mapping(mapping, require_official=False)
        tied_root = self.root / "tied"
        tied_root.mkdir()
        weights = _tiny_weights(config)
        weights.pop("lm_head.weight", None)
        save_file(weights, tied_root / "model.safetensors")
        source = Streamer.from_local(tied_root, budget_mb=1000, use_cache=False)
        self.sources.append(source)
        provider = Qwen35K2DraftProvider(
            self._model(config=config, source=source),
            head_block_rows=7,
        )

        proposal = provider((1, 4))
        provider.reconcile((1, 4, *proposal))

        self.assertEqual(provider.model.output_head_name, provider.model.EMBED_NAME)
        self.assertEqual(provider.committed_history, (1, 4, *proposal))
        self.assertEqual(provider.metrics().accepted_prefix_2, 1)

    def test_one_token_acceptance_restages_target_correction_exactly(self) -> None:
        base = (1, 4)
        draft = self._model()
        provider = Qwen35K2DraftProvider(draft, head_block_rows=7)
        proposal = provider(base)
        correction = (proposal[1] + 1) % self.config.vocab_size
        committed = (*base, proposal[0], correction)

        provider.reconcile(committed)

        reference = self._model()
        reference.prefill([base])
        stage = reference.stage_continuation_block([[proposal[0], correction]])
        reference.commit_continuation_block(stage)
        self._assert_model_state_equal(draft, reference)
        self.assertEqual(provider.committed_history, committed)
        metrics = provider.metrics()
        self.assertEqual(metrics.accepted_prefix_1, 1)
        self.assertEqual(metrics.restaged_pairs, 1)
        self.assertEqual(metrics.committed_tokens, 2)

    def test_zero_acceptance_discards_pair_and_decodes_only_target_token(self) -> None:
        base = (1, 4)
        draft = self._model()
        provider = Qwen35K2DraftProvider(draft, head_block_rows=7)
        proposal = provider(base)
        correction = (proposal[0] + 1) % self.config.vocab_size

        provider.reconcile((*base, correction))

        reference = self._model()
        reference.prefill([base])
        reference.decode([[correction]])
        self._assert_model_state_equal(draft, reference)
        self.assertEqual(provider.committed_history, (*base, correction))
        metrics = provider.metrics()
        self.assertEqual(metrics.accepted_prefix_0, 1)
        self.assertEqual(metrics.committed_tokens, 1)
        self.assertEqual(draft.next_position, len(base) + 1)

    def test_eos_at_first_token_commits_no_post_eos_draft_state(self) -> None:
        probe = self._model()
        tokens, _evidence = probe.generate_greedy(
            [[1, 4]], max_new_tokens=2, head_block_rows=7
        )
        target = self._model()
        draft = self._model()
        provider = Qwen35K2DraftProvider(
            draft,
            eos_token_ids=(tokens[0],),
            head_block_rows=7,
        )

        result = Qwen38K2SpeculativeDecoder(target, provider).generate(
            [[1, 4]],
            max_new_tokens=2,
            eos_token_ids=(tokens[0],),
            head_block_rows=7,
        )

        self.assertEqual(result.token_ids, tokens[:1])
        self.assertEqual(provider.committed_history, (1, 4, tokens[0]))
        self.assertEqual(draft.next_position, 3)
        self.assertIsNone(provider.pending_proposal)
        self._assert_model_state_equal(draft, target)
        metrics = provider.metrics()
        self.assertEqual(metrics.accepted_prefix_1, 1)
        self.assertEqual(metrics.committed_tokens, 1)

    def test_committed_and_pending_tensor_drift_fail_closed_and_reset_recovers(
        self,
    ) -> None:
        provider = Qwen35K2DraftProvider(self._model(), head_block_rows=7)
        proposal = provider((1, 4))
        pending = provider.model._pending_block_stage
        self.assertIsNotNone(pending)
        assert pending is not None
        pending.layer_states[0].recurrent.data.add_(1.0)

        with self.assertRaisesRegex(
            Qwen35K2DraftProviderError, "cursor or continuation state drifted"
        ):
            provider.reconcile((1, 4, proposal[0], proposal[1]))
        self.assertTrue(provider.poisoned)
        self.assertEqual(provider.model.next_position, 0)
        self.assertEqual(provider.model.state_bytes, 0)

        provider.reset()
        self.assertFalse(provider.poisoned)
        next_proposal = provider((1, 4))
        provider.model._next_position += 1
        with self.assertRaisesRegex(
            Qwen35K2DraftProviderError, "cursor or continuation state drifted"
        ):
            provider.reconcile((1, 4, *next_proposal))
        self.assertTrue(provider.poisoned)

    def test_history_drift_and_unreconciled_reuse_fail_closed(self) -> None:
        provider = Qwen35K2DraftProvider(self._model(), head_block_rows=7)
        provider((1, 4))
        with self.assertRaisesRegex(Qwen35K2DraftProviderError, "not reconciled"):
            provider((1, 4))
        self.assertTrue(provider.poisoned)

        provider.reset()
        proposal = provider((1, 4))
        with self.assertRaisesRegex(
            Qwen35K2DraftProviderError, "changed the committed draft prefix"
        ):
            provider.reconcile((1, 5, *proposal))
        self.assertTrue(provider.poisoned)

    def test_reconcile_callback_cannot_mutate_target_model(self) -> None:
        baseline = self._model()
        tokens, _evidence = baseline.generate_greedy(
            [[1, 4]], max_new_tokens=2, head_block_rows=7
        )
        target = self._model()

        class MutatingProvider:
            def __call__(self, _history):
                return tokens

            def reconcile(self, _history):
                target._layer_states[0].recurrent.data.add_(1.0)

        with self.assertRaisesRegex(
            Qwen38SpeculativeError,
            "changed target model state while reconciling",
        ):
            Qwen38K2SpeculativeDecoder(target, MutatingProvider()).generate(
                [[1, 4]], max_new_tokens=2, head_block_rows=7
            )
        self.assertEqual(target.next_position, 0)
        self.assertEqual(target.state_bytes, 0)

        descriptor_target = self._model()

        class DescriptorMutatingProvider:
            def __call__(self, _history):
                return tokens

            @property
            def reconcile(self):
                descriptor_target._layer_states[0].recurrent.data.add_(1.0)
                return lambda _history: None

        with self.assertRaisesRegex(
            Qwen38SpeculativeError,
            "changed target model state while reconciling",
        ):
            Qwen38K2SpeculativeDecoder(
                descriptor_target, DescriptorMutatingProvider()
            ).generate([[1, 4]], max_new_tokens=2, head_block_rows=7)
        self.assertEqual(descriptor_target.next_position, 0)
        self.assertEqual(descriptor_target.state_bytes, 0)

    def test_close_discards_pending_stage_and_is_terminal(self) -> None:
        provider = Qwen35K2DraftProvider(self._model(), head_block_rows=7)
        provider((1, 4))
        provider.close()
        provider.close()
        self.assertTrue(provider.closed)
        self.assertEqual(provider.model.next_position, 0)
        self.assertEqual(provider.model.state_bytes, 0)
        with self.assertRaisesRegex(Qwen35K2DraftProviderError, "closed"):
            provider((1, 4))
        with self.assertRaisesRegex(Qwen35K2DraftProviderError, "closed"):
            provider.reset()

    def test_k4_identical_local_model_accepts_two_complete_blocks(self) -> None:
        baseline = self._model()
        expected, _evidence = baseline.generate_greedy(
            [[1, 4]], max_new_tokens=8, head_block_rows=7
        )
        target = self._model()
        draft = self._model()
        provider = Qwen35K4DraftProvider(draft, head_block_rows=7)

        result = Qwen38K4SpeculativeDecoder(target, provider).generate(
            [[1, 4]], max_new_tokens=8, head_block_rows=7
        )

        self.assertEqual(result.token_ids, expected)
        self.assertEqual(provider.committed_history, (1, 4, *expected))
        self.assertIsNone(provider.pending_proposal)
        self._assert_model_state_equal(target, baseline)
        self._assert_model_state_equal(draft, baseline)
        metrics = provider.metrics()
        self.assertEqual(metrics.schema, QWEN35_K4_DRAFT_PROVIDER_SCHEMA)
        self.assertEqual(metrics.prefill_calls, 1)
        self.assertEqual(metrics.draft_calls, 2)
        self.assertEqual(metrics.extension_calls, 6)
        self.assertEqual(metrics.reconcile_calls, 2)
        self.assertEqual(metrics.accepted_prefix_4, 2)
        self.assertEqual(metrics.committed_tokens, 8)
        self.assertFalse(metrics.pending)
        self.assertFalse(metrics.poisoned)

    def test_k4_rolling_identical_model_emits_eight_tokens_in_two_waves(self) -> None:
        baseline = self._model()
        expected, baseline_evidence = baseline.generate_greedy(
            [[1, 4]], max_new_tokens=8, head_block_rows=7
        )
        target = self._model()
        draft = self._model()
        provider = Qwen35K4DraftProvider(draft, head_block_rows=7)

        result = Qwen38K4SpeculativeDecoder(target, provider).generate_rolling(
            [[1, 4]],
            max_new_tokens=8,
            head_block_rows=7,
        )

        self.assertEqual(result.token_ids, expected)
        self.assertEqual(len(result.evidence.rounds), 2)
        self.assertEqual(result.evidence.forward_passes, 3)
        self.assertLess(
            result.evidence.forward_passes,
            baseline_evidence.forward_passes,
        )
        self.assertEqual(provider.committed_history, (1, 4, *expected))
        self._assert_model_state_equal(target, baseline)
        self._assert_model_state_equal(draft, baseline)
        metrics = provider.metrics()
        self.assertEqual(metrics.accepted_prefix_3, 2)
        self.assertEqual(metrics.reconcile_calls, 2)
        self.assertEqual(metrics.restaged_blocks, 0)
        self.assertEqual(metrics.committed_tokens, 8)

    def test_k4_proposal_is_opaque_and_does_not_move_committed_cursor(self) -> None:
        draft = self._model()
        provider = Qwen35K4DraftProvider(draft, head_block_rows=7)

        proposal = provider((1, 4))

        self.assertEqual(len(proposal), 4)
        self.assertEqual(draft.next_position, 2)
        self.assertEqual(provider.committed_history, (1, 4))
        self.assertEqual(provider.pending_proposal, proposal)
        self.assertIsNotNone(draft._pending_block_stage)
        assert draft._pending_block_stage is not None
        self.assertEqual(
            draft._pending_block_stage.evidence.input_token_ids,
            (proposal,),
        )
        self.assertEqual(provider.metrics().extension_calls, 3)
        provider.close()
        self.assertEqual(draft.next_position, 0)
        self.assertEqual(draft.state_bytes, 0)

    def test_k4_reconcile_prefix_zero_through_four_is_exact(self) -> None:
        base = (1, 4)
        for accepted in range(5):
            with self.subTest(accepted=accepted):
                draft = self._model()
                provider = Qwen35K4DraftProvider(draft, head_block_rows=7)
                proposal = provider(base)
                if accepted == 4:
                    delta = proposal
                else:
                    correction = (proposal[accepted] + 1) % self.config.vocab_size
                    delta = (*proposal[:accepted], correction)

                provider.reconcile((*base, *delta))

                reference = self._model()
                reference.prefill([base])
                if len(delta) == 1:
                    reference.decode([[delta[0]]])
                else:
                    replay = reference.stage_continuation_block([delta])
                    reference.commit_continuation_block(replay)
                self._assert_model_state_equal(draft, reference)
                self.assertEqual(provider.committed_history, (*base, *delta))
                self.assertIsNone(provider.pending_proposal)
                metrics = provider.metrics()
                self.assertEqual(
                    getattr(metrics, f"accepted_prefix_{accepted}"),
                    1,
                )
                self.assertEqual(metrics.committed_tokens, len(delta))
                self.assertEqual(
                    metrics.restaged_blocks,
                    int(1 < len(delta) < 4 or (len(delta) == 4 and accepted < 4)),
                )

    def test_k4_rolling_proposal_commits_zero_to_three_without_weight_replay(
        self,
    ) -> None:
        base = (1, 4)
        reference_tokens_model = self._model()
        expected, _ = reference_tokens_model.generate_greedy(
            [base], max_new_tokens=4, head_block_rows=7
        )
        for accepted in range(4):
            with self.subTest(accepted=accepted):
                draft = self._model()
                provider = Qwen35K4DraftProvider(draft, head_block_rows=7)
                proposal = provider.propose_after(base, expected[0])
                self.assertEqual(proposal, expected[1:4])
                self.assertEqual(draft.next_position, len(base) + 1)
                self.assertEqual(provider.pending_proposal, proposal)
                committed = (*base, expected[0], *proposal[:accepted])

                before = draft.pager.metrics()
                provider.reconcile_prefix(committed)
                after = draft.pager.metrics()

                reference = self._model()
                reference.prefill([base])
                for token in committed[len(base) :]:
                    reference.decode([[token]])
                self._assert_model_state_equal(draft, reference)
                self.assertEqual(provider.committed_history, committed)
                self.assertIsNone(provider.pending_proposal)
                for key in (
                    "tensor_reads",
                    "linear_calls",
                    "network_or_source_body_bytes",
                ):
                    self.assertEqual(after[key], before[key])
                self.assertEqual(
                    getattr(provider.metrics(), f"accepted_prefix_{accepted}"),
                    1,
                )

    def test_k4_rolling_next_wave_starts_after_target_known_correction(self) -> None:
        base = (1, 4)
        baseline = self._model()
        expected, _ = baseline.generate_greedy(
            [base], max_new_tokens=6, head_block_rows=7
        )
        provider = Qwen35K4DraftProvider(self._model(), head_block_rows=7)
        first = provider.propose_after(base, expected[0])
        provider.reconcile_prefix((*base, expected[0], first[0]))
        history = (*base, expected[0], first[0])

        second = provider.propose_after(history, expected[2])

        self.assertEqual(second, expected[3:6])
        self.assertEqual(provider.committed_history, (*history, expected[2]))
        self.assertEqual(provider.model.next_position, len(history) + 1)
        provider.reconcile_prefix((*history, expected[2]))
        self.assertIsNone(provider.pending_proposal)

    def test_k4_rejected_suffix_never_reaches_the_next_proposal(self) -> None:
        base = (1, 4)
        provider = Qwen35K4DraftProvider(self._model(), head_block_rows=7)
        first = provider(base)
        correction = (first[2] + 1) % self.config.vocab_size
        committed = (*base, first[0], first[1], correction)
        provider.reconcile(committed)

        reference = self._model()
        reference.prefill([base])
        replay = reference.stage_continuation_block([[first[0], first[1], correction]])
        hidden, _evidence = reference.commit_continuation_block(replay)
        values, selected = reference.pager.topk_logits(
            hidden[:, -1],
            k=1,
            name=reference.output_head_name,
            block_rows=7,
        )
        expected_next = int(selected[0, 0].item())
        del values, selected

        second = provider(committed)

        self.assertEqual(second[0], expected_next)
        self.assertNotEqual(provider.model.next_position, len(base) + 4)
        self.assertEqual(provider.model.next_position, len(committed))
        provider.close()

    def test_k4_eos_at_every_position_replays_only_through_eos(self) -> None:
        probe = self._model()
        tokens, _evidence = probe.generate_greedy(
            [[1, 4]], max_new_tokens=4, head_block_rows=7
        )
        self.assertEqual(len(set(tokens)), 4)
        for eos_index in range(4):
            with self.subTest(eos_index=eos_index):
                eos = tokens[eos_index]
                draft = self._model()
                provider = Qwen35K4DraftProvider(
                    draft,
                    eos_token_ids=(eos,),
                    head_block_rows=7,
                )
                proposal = provider((1, 4))
                self.assertEqual(proposal[: eos_index + 1], tokens[: eos_index + 1])
                self.assertTrue(all(token == eos for token in proposal[eos_index:]))

                committed = (1, 4, *proposal[: eos_index + 1])
                provider.reconcile(committed)

                reference = self._model()
                reference.prefill([[1, 4]])
                delta = proposal[: eos_index + 1]
                if len(delta) == 1:
                    reference.decode([[delta[0]]])
                else:
                    stage = reference.stage_continuation_block([delta])
                    reference.commit_continuation_block(stage)
                self._assert_model_state_equal(draft, reference)
                self.assertEqual(draft.next_position, len(committed))
                self.assertIsNone(provider.pending_proposal)

    def test_k4_drift_invalid_reconcile_and_extension_failure_fail_closed(self) -> None:
        provider = Qwen35K4DraftProvider(self._model(), head_block_rows=7)
        proposal = provider((1, 4))
        pending = provider.model._pending_block_stage
        self.assertIsNotNone(pending)
        assert pending is not None
        pending.layer_states[0].recurrent.data.add_(1.0)
        with self.assertRaisesRegex(
            Qwen35K4DraftProviderError, "cursor or continuation state drifted"
        ):
            provider.reconcile((1, 4, *proposal))
        self.assertTrue(provider.poisoned)
        self.assertEqual(provider.model.next_position, 0)

        provider.reset()
        proposal = provider((1, 4))
        bad_second = (proposal[1] + 1) % self.config.vocab_size
        with self.assertRaisesRegex(
            Qwen35K4DraftProviderError, "before its final item"
        ):
            provider.reconcile((1, 4, proposal[0], bad_second, proposal[2]))
        self.assertTrue(provider.poisoned)

        provider.reset()
        proposal = provider((1, 4))
        with self.assertRaisesRegex(
            Qwen35K4DraftProviderError, "accepted block without EOS"
        ):
            provider.reconcile((1, 4, proposal[0], proposal[1]))
        self.assertTrue(provider.poisoned)

        broken_model = self._model()
        broken = Qwen35K4DraftProvider(broken_model, head_block_rows=7)
        original = broken_model.extend_continuation_block
        calls = 0

        def fail_second(stage, input_ids):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("extension failed")
            return original(stage, input_ids)

        broken_model.extend_continuation_block = (  # type: ignore[method-assign]
            fail_second
        )
        with self.assertRaisesRegex(Qwen35K4DraftProviderError, "extension failed"):
            broken((1, 4))
        self.assertTrue(broken.poisoned)
        self.assertEqual(broken_model.next_position, 0)
        self.assertEqual(broken_model.state_bytes, 0)
        self.assertIsNone(broken_model._pending_block_stage)


if __name__ == "__main__":
    unittest.main()
