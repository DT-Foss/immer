from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file

from immer.knowledge import Streamer
from immer.runtimes.qwen3_8.markov_draft import (
    FingerprintRollingK4DraftProvider,
    MarkovDraftError,
    MarkovDraftState,
)
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
        self.assertEqual(restored.updates, 2)

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


if __name__ == "__main__":
    unittest.main()
