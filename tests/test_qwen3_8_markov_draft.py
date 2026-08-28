from __future__ import annotations

from pathlib import Path
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
    MarkovDraftError,
    MarkovDraftState,
)
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
        provider.close()
        restored = MarkovDraftState.from_bytes(state_path.read_bytes())
        self.assertEqual(restored.updates, 33)
        self.assertEqual(restored.episode_lengths[-1], len(prompt) + len(expected))

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
        proposal, feedback = provider._predict_council(history, 1)
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


if __name__ == "__main__":
    unittest.main()
