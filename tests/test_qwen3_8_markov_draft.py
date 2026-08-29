from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import json
import os
import tempfile
import unittest
from unittest import mock
import zlib

from safetensors.torch import save_file

from immer.knowledge import Streamer
from immer.runtimes.qwen3_8.markov_draft import (
    FingerprintRollingK4DraftProvider,
    MarkovDialectState,
    MarkovDraftError,
    MarkovDraftState,
)
from immer.runtimes.qwen3_8.draft_protocol import RollingDraftProposal
import immer.runtimes.qwen3_8.markov_draft as markov_module
from immer.runtimes.qwen3_8.model import StreamedQwen38
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager
from immer.runtimes.qwen3_8.speculative import Qwen38K4SpeculativeDecoder

from test_qwen3_8_model import (
    _assert_layer_states_equal,
    _tiny_config,
    _tiny_weights,
)


class Qwen38MarkovDraftTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = _tiny_config()
        save_file(_tiny_weights(self.config), self.root / "model.safetensors")
        self.source = Streamer.from_local(
            self.root, budget_mb=1000, use_cache=False
        )
        self.pagers: list[Qwen38WeightPager] = []

    def tearDown(self) -> None:
        for pager in self.pagers:
            pager.close()
        self.source.close()
        self.temporary.cleanup()

    def _model(self) -> StreamedQwen38:
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
            max_seq_len=32,
        )

    def test_state_roundtrip_and_corruption_rejection(self) -> None:
        state = MarkovDraftState(
            vocab_size=32,
            max_history_tokens=64,
            token_ids=(1, 2, 3, 1, 2, 3),
            updates=4,
        )
        encoded = state.to_bytes()

        self.assertEqual(MarkovDraftState.from_bytes(encoded), state)
        damaged = bytearray(encoded)
        damaged[-1] ^= 1
        with self.assertRaises(MarkovDraftError):
            MarkovDraftState.from_bytes(bytes(damaged))

    def test_v1_state_migrates_into_one_episode_and_initializes_council(self) -> None:
        legacy = {
            "max_history_tokens": 64,
            "schema": "immer.qwen3.8-markov-draft-state/v1",
            "token_ids": [1, 2, 3],
            "updates": 1,
            "vocab_size": 32,
        }
        raw = json.dumps(
            legacy,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        path = self.root / "legacy-state.bin"
        path.write_bytes(b"IMMD\x01" + zlib.compress(raw, level=9))

        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=path,
            max_history_tokens=64,
        )

        self.assertEqual(provider._state.episode_lengths, (3,))
        self.assertEqual(len(provider.metrics().expert_weights), 8)
        provider.close()

    def test_v2_council_state_migrates_to_v4_phrase_memory(self) -> None:
        seed = FingerprintRollingK4DraftProvider(vocab_size=32)
        state = seed._state
        seed.close()
        v2 = {
            "episode_lengths": [3],
            "expert_hits": list(state.expert_hits),
            "expert_log_weights": [value.hex() for value in state.expert_log_weights],
            "expert_names": list(state.expert_names),
            "expert_observations": list(state.expert_observations),
            "feedback_count": 0,
            "leader_changes": 0,
            "max_history_tokens": 4096,
            "regime_generation": 0,
            "schema": "immer.qwen3.8-markov-draft-state/v2",
            "surprise_cusum": float(0.0).hex(),
            "surprise_deviation": float(1.0).hex(),
            "surprise_mean": float(0.0).hex(),
            "token_ids": [1, 2, 3],
            "updates": 1,
            "vocab_size": 32,
        }
        raw = json.dumps(
            v2,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        path = self.root / "v2-state.bin"
        path.write_bytes(b"IMMD\x02" + zlib.compress(raw, level=9))

        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=path,
        )
        self.assertEqual(provider.metrics().dialect_count, 0)
        provider.observe_final((1, 2, 3, 4))
        provider.close()

        self.assertTrue(path.read_bytes().startswith(b"IMMD\x04"))
        migrated = MarkovDraftState.from_bytes(path.read_bytes())
        self.assertEqual(len(migrated.dialects), 1)

    def test_v3_dialect_state_migrates_episode_bindings_to_v4(self) -> None:
        encoded = MarkovDraftState(
            vocab_size=32,
            max_history_tokens=64,
            token_ids=(1, 2, 3),
            episode_lengths=(3,),
        ).to_bytes()
        document = json.loads(zlib.decompress(encoded[5:]))
        document.pop("episode_dialects")
        document["schema"] = "immer.qwen3.8-markov-draft-state/v3"
        raw = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        path = self.root / "v3-state.bin"
        path.write_bytes(b"IMMD\x03" + zlib.compress(raw, level=9))

        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=path,
            max_history_tokens=64,
        )

        self.assertEqual(provider._state.episode_dialects, (None,))
        provider.observe_final((1, 2, 3, 4))
        provider.close()
        self.assertTrue(path.read_bytes().startswith(b"IMMD\x04"))

    def test_variable_order_provider_predicts_and_learns_confirmed_prefix(self) -> None:
        state_path = self.root / "markov-state.bin"
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=state_path,
            max_history_tokens=64,
        )

        proposal = provider.propose_after((1, 2), 3)
        provider.reconcile_prefix((1, 2, 3, proposal[0], proposal[1]))
        provider.observe_final((1, 2, 3, proposal[0], proposal[1], 7))
        provider.close()

        self.assertEqual(proposal, (1, 2, 3))
        restored = MarkovDraftState.from_bytes(state_path.read_bytes())
        self.assertEqual(restored.token_ids[-4:], (3, 1, 2, 7))
        self.assertEqual(restored.updates, 1)
        self.assertEqual(restored.episode_lengths, (6,))
        self.assertEqual(len(restored.expert_names), 8)
        self.assertEqual(restored.feedback_count, 3)

    def test_variable_window_markov_proposes_and_learns_seven_tokens(self) -> None:
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            max_history_tokens=64,
            proposal_width=7,
        )

        proposal = provider.propose_after((1, 2), 3)
        self.assertEqual(len(proposal), 7)
        provider.reconcile_prefix((1, 2, 3, *proposal[:5]))
        provider.observe_final((1, 2, 3, *proposal[:5], 7))

        metrics = provider.metrics()
        self.assertEqual(metrics.proposal_width, 7)
        self.assertEqual(metrics.predictions, 8)
        self.assertEqual(metrics.council_feedback, 6)
        provider.close()

    def test_persisted_markov_memory_drafts_native_qwen_ids_without_model(self) -> None:
        prompt = (1, 4)
        baseline_long = self._model()
        expected_long, _ = baseline_long.generate_greedy(
            [prompt], max_new_tokens=12, head_block_rows=7
        )
        baseline = self._model()
        expected, baseline_evidence = baseline.generate_greedy(
            [prompt], max_new_tokens=8, head_block_rows=7
        )
        training_episode = (*prompt, *expected_long)
        state_path = self.root / "trained-markov.bin"
        state_path.write_bytes(
            MarkovDraftState(
                vocab_size=self.config.vocab_size,
                max_history_tokens=4096,
                token_ids=training_episode * 32,
                episode_lengths=(len(training_episode),) * 32,
                updates=32,
            ).to_bytes()
        )
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=self.config.vocab_size,
            state_path=state_path,
        )
        target = self._model()

        result = Qwen38K4SpeculativeDecoder(
            target, provider
        ).generate_rolling(
            [prompt],
            max_new_tokens=8,
            head_block_rows=7,
        )

        self.assertEqual(result.token_ids, expected)
        self.assertLess(result.evidence.forward_passes, baseline_evidence.forward_passes)
        self.assertGreater(result.evidence.accepted_draft_tokens, 0)
        _assert_layer_states_equal(self, target._layer_states, baseline._layer_states)
        metrics = provider.metrics()
        self.assertEqual(metrics.source_body_bytes, 0)
        self.assertEqual(metrics.linear_calls, 0)
        self.assertGreater(metrics.learned_tokens, 0)
        self.assertEqual(len(metrics.expert_weights), 8)
        self.assertEqual(len(metrics.expert_accuracy), 8)
        self.assertGreaterEqual(metrics.effective_experts, 1.0)
        self.assertLessEqual(metrics.effective_experts, 8.0)
        self.assertGreater(metrics.last_confidence, 0.0)
        self.assertGreaterEqual(metrics.last_disagreement, 0.0)
        self.assertGreater(metrics.phrase_option_calls, 0)
        self.assertGreater(metrics.phrase_accepted_tokens, 0)
        self.assertEqual(metrics.last_phrase_source, "global")
        provider.close()
        restored = MarkovDraftState.from_bytes(state_path.read_bytes())
        self.assertEqual(restored.updates, 33)
        self.assertEqual(restored.episode_lengths[-1], len(prompt) + len(expected))

    def test_real_markov_council_drives_adaptive_k8_target_wave(self) -> None:
        prompt = (1, 4)
        reference = self._model()
        expected, _ = reference.generate_greedy(
            [prompt], max_new_tokens=8, head_block_rows=7
        )
        episode = (*prompt, *expected)
        state_path = self.root / "adaptive-markov.bin"
        state_path.write_bytes(
            MarkovDraftState(
                vocab_size=self.config.vocab_size,
                max_history_tokens=4096,
                token_ids=episode * 8,
                episode_lengths=(len(episode),) * 8,
                updates=8,
            ).to_bytes()
        )
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=self.config.vocab_size,
            state_path=state_path,
            proposal_width=7,
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
        _assert_layer_states_equal(self, candidate._layer_states, reference._layer_states)
        self.assertEqual(result.evidence.used_window_sizes, (8,))
        self.assertEqual(result.evidence.rounds[0].round_policy.chosen_window, 8)
        self.assertEqual(len(result.evidence.rounds[0].provider_proposed_token_ids), 7)
        metrics = provider.metrics()
        self.assertEqual(metrics.adaptive_proposal_calls, 1)
        self.assertEqual(metrics.last_recommended_window, 8)
        provider.close()

    def test_one_token_rolling_request_persists_its_terminal_token(self) -> None:
        state_path = self.root / "one-token-markov.bin"
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=self.config.vocab_size,
            state_path=state_path,
        )
        target = self._model()

        result = Qwen38K4SpeculativeDecoder(
            target, provider
        ).generate_rolling(
            [[1, 4]],
            max_new_tokens=1,
            head_block_rows=7,
            retain_final_state=False,
        )
        provider.close()

        restored = MarkovDraftState.from_bytes(state_path.read_bytes())
        self.assertEqual(restored.token_ids[-1], result.token_ids[0])
        self.assertEqual(restored.updates, 1)

    def test_council_reweights_and_recovers_after_regime_flip(self) -> None:
        provider = FingerprintRollingK4DraftProvider(vocab_size=32)
        history = tuple((1, 2, 1, 3) * 20) + (2, 1)
        proposal, feedback, _confidence, _disagreement = (
            provider._predict_council(history, 1)
        )
        self.assertEqual(proposal, (3,))

        for _ in range(30):
            provider._apply_council_feedback(feedback[0], 3)
        structured = dict(provider.metrics().expert_weights)
        self.assertGreater(structured["recent-o2-w256"], 0.20)
        self.assertLess(structured["global-o0-w4096"], 0.01)

        for _ in range(80):
            provider._apply_council_feedback(feedback[0], 1)
        shifted = dict(provider.metrics().expert_weights)
        self.assertGreater(shifted["global-o0-w4096"], 0.45)
        self.assertLess(shifted["recent-o2-w256"], 0.01)
        self.assertTrue(all(weight >= 0.05 / 8 for weight in shifted.values()))
        self.assertGreaterEqual(provider.metrics().leader_changes, 1)
        self.assertGreaterEqual(provider.metrics().regime_generation, 1)
        self.assertGreater(provider.metrics().surprise_mean, 0.0)

    def test_persistent_history_keeps_explicit_episode_boundaries(self) -> None:
        state_path = self.root / "episodes.bin"
        state_path.write_bytes(
            MarkovDraftState(
                vocab_size=32,
                max_history_tokens=64,
                token_ids=(1, 2, 3, 4),
                episode_lengths=(2, 2),
            ).to_bytes()
        )
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=state_path,
            max_history_tokens=64,
        )

        symbols = provider._persistent_symbols()

        self.assertEqual(symbols, ("01", "02", "<episode>", "03", "04"))
        provider.close()

    def test_persistent_state_lock_is_held_until_provider_close(self) -> None:
        if markov_module.fcntl is None:
            self.skipTest("POSIX flock is unavailable")
        state_path = self.root / "locked-state.bin"
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=state_path,
            max_history_tokens=64,
        )
        lock_path = self.root / ".locked-state.bin.lock"
        descriptor = os.open(lock_path, os.O_RDWR)
        try:
            with self.assertRaises(BlockingIOError):
                markov_module.fcntl.flock(
                    descriptor,
                    markov_module.fcntl.LOCK_EX | markov_module.fcntl.LOCK_NB,
                )
            provider.close()
            markov_module.fcntl.flock(
                descriptor,
                markov_module.fcntl.LOCK_EX | markov_module.fcntl.LOCK_NB,
            )
        finally:
            os.close(descriptor)

    def test_aborted_half_episode_persists_no_feedback_or_tokens(self) -> None:
        state_path = self.root / "atomic-episode.bin"
        first = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=state_path,
            max_history_tokens=64,
        )
        proposal = first.propose_after((1, 2), 3)
        first.reconcile_prefix((1, 2, 3, proposal[0]))
        first.close()

        aborted = MarkovDraftState.from_bytes(state_path.read_bytes())
        self.assertEqual(aborted.feedback_count, 0)
        self.assertEqual(aborted.updates, 0)
        self.assertEqual(aborted.token_ids, ())
        self.assertEqual(aborted.dialects, ())

        replay = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=state_path,
            max_history_tokens=64,
        )
        replay_proposal = replay.propose_after((1, 2), 3)
        self.assertEqual(replay_proposal, proposal)
        replay.reconcile_prefix((1, 2, 3, replay_proposal[0]))
        replay.observe_final((1, 2, 3, replay_proposal[0], 7))
        replay.close()

        completed = MarkovDraftState.from_bytes(state_path.read_bytes())
        self.assertEqual(completed.feedback_count, 2)
        self.assertEqual(completed.updates, 1)
        self.assertEqual(completed.episode_lengths, (5,))

    def test_state_replacement_during_descriptor_read_fails_closed(self) -> None:
        state_path = self.root / "replace-state.bin"
        state_path.write_bytes(
            MarkovDraftState(
                vocab_size=32,
                max_history_tokens=64,
                token_ids=(1, 2, 3),
            ).to_bytes()
        )
        replacement = self.root / "replacement.bin"
        replacement.write_bytes(
            MarkovDraftState(
                vocab_size=32,
                max_history_tokens=64,
                token_ids=(7, 8, 9),
            ).to_bytes()
        )
        original_read = markov_module.os.read
        replaced = False

        def replace_then_read(descriptor, length):
            nonlocal replaced
            if not replaced:
                replaced = True
                replacement.replace(state_path)
            return original_read(descriptor, length)

        with mock.patch.object(
            markov_module.os,
            "read",
            side_effect=replace_then_read,
        ):
            with self.assertRaisesRegex(MarkovDraftError, "changed while read"):
                FingerprintRollingK4DraftProvider(
                    vocab_size=32,
                    state_path=state_path,
                    max_history_tokens=64,
                )

    def test_similar_context_reuses_dialect_and_distinct_context_creates_one(self) -> None:
        state_path = self.root / "dialects.bin"
        first_context = tuple((1, 2, 3, 4) * 20)
        similar_context = (*first_context[:-1], 5)
        second_context = tuple((40, 41, 42, 43) * 20)
        for context, expected_count, minimum_similarity in (
            (first_context, 1, 0.0),
            (similar_context, 1, 0.80),
            (second_context, 2, 0.0),
        ):
            provider = FingerprintRollingK4DraftProvider(
                vocab_size=64,
                state_path=state_path,
                max_history_tokens=256,
            )
            provider.observe_final(context)
            metrics = provider.metrics()
            self.assertEqual(metrics.dialect_count, expected_count)
            self.assertGreaterEqual(
                metrics.active_dialect_similarity,
                minimum_similarity,
            )
            provider.close()

        state = MarkovDraftState.from_bytes(state_path.read_bytes())
        visits = sorted(row.visits for row in state.dialects)
        self.assertEqual(visits, [1, 2])

    def test_ricci_retention_evicts_oldest_low_visit_dialect(self) -> None:
        provider = FingerprintRollingK4DraftProvider(vocab_size=128)
        width = len(provider._experts)
        dialects = tuple(
            MarkovDialectState(
                dialect_id=f"{index:064x}",
                signature=(index,),
                visits=1,
                last_seen=index,
                rapidities=(0.0,) * width,
                observations=(0,) * width,
                hits=(0,) * width,
            )
            for index in range(64)
        )
        provider._state = replace(provider._state, clock=100, dialects=dialects)
        provider._active_dialect = MarkovDialectState(
            dialect_id="f" * 64,
            signature=(999,),
            visits=1,
            last_seen=101,
            rapidities=(0.0,) * width,
            observations=(0,) * width,
            hits=(0,) * width,
        )
        provider._active_dialect_is_new = True
        provider._request_signature = (999,)

        provider._commit_active_dialect()

        self.assertEqual(len(provider._state.dialects), 64)
        self.assertNotIn("0" * 64, {row.dialect_id for row in provider._state.dialects})
        self.assertIn("f" * 64, {row.dialect_id for row in provider._state.dialects})
        self.assertEqual(provider.metrics().dialect_evictions, 1)

    def test_dialect_profiles_recall_opposite_expert_regimes(self) -> None:
        state_path = self.root / "opposite-dialects.bin"
        structured_context = tuple((1, 2, 3, 4) * 20)
        frequency_context = tuple((40, 41, 42, 43) * 20)

        def feedback(target: int, *, structured: bool):
            rows = []
            other = 1 if target != 1 else 3
            for index in range(8):
                correct = (index >= 4) == structured
                probability = 0.9 if correct else 0.1
                rows.append(
                    (
                        {
                            f"{target:03d}": probability,
                            f"{other:03d}": 1.0 - probability,
                            "<unknown>": 0.0,
                        },
                        target if correct else other,
                    )
                )
            return tuple(rows)

        first = FingerprintRollingK4DraftProvider(
            vocab_size=128,
            state_path=state_path,
            max_history_tokens=256,
        )
        first._activate_dialect(structured_context)
        for _ in range(30):
            first._apply_council_feedback(feedback(3, structured=True), 3)
        first.observe_final(structured_context)
        first.close()

        second = FingerprintRollingK4DraftProvider(
            vocab_size=128,
            state_path=state_path,
            max_history_tokens=256,
        )
        second._activate_dialect(frequency_context)
        for _ in range(80):
            second._apply_council_feedback(feedback(1, structured=False), 1)
        second.observe_final(frequency_context)
        second.close()

        recalled = []
        for context in (structured_context, frequency_context):
            provider = FingerprintRollingK4DraftProvider(
                vocab_size=128,
                state_path=state_path,
                max_history_tokens=256,
            )
            provider._activate_dialect(context)
            recalled.append(dict(provider.metrics().expert_weights))
            self.assertEqual(provider.metrics().active_dialect_similarity, 1.0)
            provider.close()

        self.assertGreater(recalled[0]["recent-o2-w256"], 0.20)
        self.assertLess(recalled[0]["global-o0-w4096"], 0.01)
        self.assertGreater(recalled[1]["global-o0-w4096"], 0.20)
        self.assertLess(recalled[1]["recent-o2-w256"], 0.01)

    def test_one_provider_cannot_leak_dialect_into_a_second_request(self) -> None:
        provider = FingerprintRollingK4DraftProvider(vocab_size=64)
        provider.begin_request((1, 2, 3, 4))
        provider.observe_final((1, 2, 3, 4, 5))

        with self.assertRaisesRegex(MarkovDraftError, "exactly one request"):
            provider.begin_request((40, 41, 42, 43))

    def test_dialect_influence_scales_with_context_similarity(self) -> None:
        provider = FingerprintRollingK4DraftProvider(vocab_size=64)
        width = len(provider._experts)
        provider._active_dialect = MarkovDialectState(
            dialect_id="a" * 64,
            signature=(1, 2, 3),
            visits=4,
            last_seen=0,
            rapidities=(-2.0, -2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 2.0),
            observations=(10,) * width,
            hits=(5,) * width,
        )
        provider._active_dialect_similarity = 0.20
        low = provider._weights()
        provider._active_dialect_similarity = 1.0
        exact = provider._weights()

        self.assertGreater(sum(exact[4:]), sum(low[4:]))
        self.assertLess(sum(exact[:4]), sum(low[:4]))

    def test_global_phrase_agent_emits_only_repeated_three_token_option(self) -> None:
        episode = (1, 2, 3, 4, 5, 6)
        state_path = self.root / "global-phrases.bin"
        state_path.write_bytes(
            MarkovDraftState(
                vocab_size=32,
                max_history_tokens=128,
                token_ids=episode * 3,
                episode_lengths=(len(episode),) * 3,
            ).to_bytes()
        )
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=state_path,
            max_history_tokens=128,
        )
        provider.begin_request((9, 1))

        proposal = provider.propose_after((9, 1), 2)

        self.assertEqual(proposal, (3, 4, 5))
        metrics = provider.metrics()
        self.assertEqual(metrics.last_phrase_source, "global")
        self.assertEqual(metrics.last_phrase_support, 3)
        self.assertEqual(metrics.last_phrase_confidence, 1.0)
        provider.reconcile_prefix((9, 1, 2, 3, 4))
        provider.observe_final((9, 1, 2, 3, 4, 7))
        self.assertEqual(provider.metrics().phrase_accepted_tokens, 2)
        provider.close()

    def test_phrase_agent_fills_a_seven_token_window_from_repeated_episode(self) -> None:
        episode = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        state_path = self.root / "wide-global-phrases.bin"
        state_path.write_bytes(
            MarkovDraftState(
                vocab_size=32,
                max_history_tokens=128,
                token_ids=episode * 3,
                episode_lengths=(len(episode),) * 3,
            ).to_bytes()
        )
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=state_path,
            max_history_tokens=128,
            proposal_width=7,
        )
        provider.begin_request((20, 1))

        proposal = provider.propose_after((20, 1), 2)

        self.assertEqual(proposal, (3, 4, 5, 6, 7, 8, 9))
        metrics = provider.metrics()
        self.assertEqual(metrics.last_phrase_width, 7)
        self.assertEqual(metrics.phrase_draft_tokens, 7)
        provider.reconcile_prefix((20, 1, 2, *proposal))
        provider.observe_final((20, 1, 2, *proposal))
        self.assertEqual(provider.metrics().phrase_accepted_tokens, 7)
        provider.close()

    def test_adaptive_proposal_exposes_prefix_local_horizon_evidence(self) -> None:
        episode = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            max_history_tokens=128,
            proposal_width=7,
        )
        provider._state = replace(
            provider._state,
            token_ids=episode * 3,
            episode_lengths=(len(episode),) * 3,
            episode_dialects=(None,) * 3,
        )
        provider.begin_request((20, 1))

        proposal = provider.propose_round((20, 1), 2)

        self.assertIsInstance(proposal, RollingDraftProposal)
        self.assertEqual(proposal.token_ids, (3, 4, 5, 6, 7, 8, 9))
        self.assertEqual(tuple(row.window for row in proposal.horizons), (4, 8))
        self.assertEqual(proposal.recommended_window, 8)
        self.assertEqual(proposal.phrase_width, 7)
        policy = proposal.select_window(
            request_window_ceiling=8,
            remaining_tokens=8,
        )
        self.assertEqual(policy.chosen_window, 8)
        provider.reconcile_prefix((20, 1, 2, *proposal.token_ids))
        provider.observe_final((20, 1, 2, *proposal.token_ids))
        metrics = provider.metrics()
        self.assertEqual(metrics.adaptive_proposal_calls, 1)
        self.assertEqual(dict(metrics.recommended_windows)[8], 1)
        self.assertEqual(metrics.last_recommended_window, 8)
        provider.close()

    def test_phrase_agent_uses_longest_prefix_with_repeated_support(self) -> None:
        episodes = (
            (1, 2, 3, 4, 5, 6, 7, 8),
            (1, 2, 3, 4, 5, 6, 9, 10),
            (1, 2, 3, 4, 5, 6, 11, 12),
        )
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            max_history_tokens=128,
            proposal_width=7,
        )
        provider._state = replace(
            provider._state,
            token_ids=tuple(token for episode in episodes for token in episode),
            episode_lengths=tuple(len(episode) for episode in episodes),
            episode_dialects=(None,) * len(episodes),
        )
        provider.begin_request((20, 1))

        proposal = provider.propose_after((20, 1), 2)

        self.assertEqual(proposal[:4], (3, 4, 5, 6))
        self.assertEqual(provider.metrics().last_phrase_width, 4)
        self.assertEqual(provider.metrics().last_phrase_support, 3)
        provider.reconcile_prefix((20, 1, 2, *proposal[:4]))
        provider.observe_final((20, 1, 2, *proposal[:4]))
        provider.close()

    def test_dialect_phrase_agent_activates_before_global_support_threshold(self) -> None:
        state_path = self.root / "dialect-phrases.bin"
        prompt = tuple((1, 2) * 20)
        episode = (*prompt, 3, 4, 5, 6)
        for _ in range(2):
            writer = FingerprintRollingK4DraftProvider(
                vocab_size=32,
                state_path=state_path,
                max_history_tokens=256,
            )
            writer.observe_final(episode)
            writer.close()
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=state_path,
            max_history_tokens=256,
        )
        provider.begin_request(prompt)

        proposal = provider.propose_after(prompt, 3)

        self.assertEqual(proposal, (4, 5, 6))
        self.assertEqual(provider.metrics().last_phrase_source, "dialect")
        self.assertEqual(provider.metrics().last_phrase_support, 2)
        provider.reconcile_prefix((*prompt, 3, 4, 5, 6))
        provider.observe_final((*prompt, 3, 4, 5, 6))
        provider.close()

    def test_stronger_global_phrase_beats_weak_dialect_phrase(self) -> None:
        prompt = (9, 1, 2)
        local_episode = (1, 2, 3, 4, 5)
        global_episode = (1, 2, 7, 8, 9)
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            max_history_tokens=64,
        )
        dialect_id = "a" * 64
        expert_count = len(provider._state.expert_names)
        dialect = MarkovDialectState(
            dialect_id=dialect_id,
            signature=provider._context_signature(prompt),
            visits=2,
            last_seen=1,
            rapidities=(0.0,) * expert_count,
            observations=(0,) * expert_count,
            hits=(0,) * expert_count,
        )
        episodes = (local_episode,) * 2 + (global_episode,) * 6
        provider._state = replace(
            provider._state,
            token_ids=tuple(token for episode in episodes for token in episode),
            episode_lengths=(5,) * len(episodes),
            episode_dialects=(dialect_id, dialect_id) + (None,) * 6,
            clock=1,
            dialects=(dialect,),
        )
        provider.begin_request(prompt)

        proposal = provider.propose_after((9, 1), 2)

        self.assertEqual(proposal, (7, 8, 9))
        self.assertEqual(provider.metrics().last_phrase_source, "global")
        self.assertEqual(provider.metrics().last_phrase_support, 6)
        provider.reconcile_prefix((9, 1, 2, 7, 8, 9))
        provider.observe_final((9, 1, 2, 7, 8, 9))
        provider.close()


if __name__ == "__main__":
    unittest.main()
