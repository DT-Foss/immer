from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from immer.runtimes.qwen3_8.markov_atlas import (
    MarkovAtlasError,
    MarkovTokenAtlas,
)
from immer.runtimes.qwen3_8.markov_draft import FingerprintRollingK4DraftProvider


TOKENIZER_SHA = "a" * 64


class Qwen38MarkovAtlasTests(unittest.TestCase):
    def _atlas(self) -> MarkovTokenAtlas:
        return MarkovTokenAtlas.build(
            (
                (9, 2, 3, 4, 5, 6),
                (8, 2, 3, 4, 5, 7),
                (7, 2, 3, 4, 5, 8),
                (6, 2, 3, 4, 5, 9),
            ),
            vocab_size=32,
            tokenizer_sha256=TOKENIZER_SHA,
            max_order=4,
            min_context_count=2,
            max_branches=4,
        )

    def test_longest_context_continuation_crosses_no_document_boundary(self) -> None:
        atlas = self._atlas()

        continuation = atlas.continuation(
            (31, 2),
            max_tokens=4,
            min_support=2,
            min_confidence=0.5,
        )

        self.assertIsNotNone(continuation)
        assert continuation is not None
        self.assertEqual(continuation.token_ids, (3, 4, 5))
        self.assertEqual(continuation.context_order, 1)
        self.assertEqual(continuation.support, 4)
        self.assertGreaterEqual(continuation.minimum_confidence, 0.5)
        self.assertIsNone(
            atlas.continuation(
                (6,),
                max_tokens=2,
                min_support=2,
                min_confidence=0.5,
            )
        )

    def test_roundtrip_identity_and_corruption_rejection(self) -> None:
        atlas = self._atlas()
        encoded = atlas.to_bytes()
        restored = MarkovTokenAtlas.from_bytes(
            encoded,
            expected_vocab_size=32,
            expected_tokenizer_sha256=TOKENIZER_SHA,
        )

        self.assertEqual(restored.metrics(), atlas.metrics())
        self.assertEqual(restored.sha256, atlas.sha256)
        with self.assertRaisesRegex(MarkovAtlasError, "vocabulary"):
            MarkovTokenAtlas.from_bytes(encoded, expected_vocab_size=33)
        damaged = bytearray(encoded)
        damaged[-1] ^= 1
        with self.assertRaises(MarkovAtlasError):
            MarkovTokenAtlas.from_bytes(bytes(damaged))

    def test_atomic_file_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "atlas.bin"
            atlas = self._atlas()
            atlas.write(path)

            restored = MarkovTokenAtlas.load(
                path,
                expected_vocab_size=32,
                expected_tokenizer_sha256=TOKENIZER_SHA,
            )

        self.assertEqual(restored.sha256, atlas.sha256)

    def test_provider_uses_atlas_phrase_and_records_target_acceptance(self) -> None:
        atlas = self._atlas()
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            proposal_width=3,
            atlas=atlas,
        )
        prompt = (20, 21)
        provider.begin_request(prompt)

        proposal = provider.propose_round(prompt, 2)

        self.assertEqual(proposal.token_ids, (3, 4, 5))
        self.assertEqual(proposal.phrase_source, "atlas")
        self.assertEqual(proposal.phrase_width, 3)
        provider.reconcile_prefix((*prompt, 2, 3, 4, 5))
        provider.observe_final((*prompt, 2, 3, 4, 5))
        metrics = provider.metrics()
        self.assertEqual(metrics.atlas_contexts, atlas.context_count)
        self.assertEqual(metrics.atlas_corpus_tokens, atlas.token_count)
        self.assertEqual(metrics.atlas_option_calls, 1)
        self.assertEqual(metrics.atlas_draft_tokens, 3)
        self.assertEqual(metrics.atlas_accepted_tokens, 3)
        provider.close()

    def test_provider_leaves_low_confidence_atlas_routes_to_the_council(self) -> None:
        atlas = MarkovTokenAtlas.build(
            (
                (9, 2, 3),
                (8, 2, 3),
                (7, 2, 4),
                (6, 2, 4),
            ),
            vocab_size=32,
            tokenizer_sha256=TOKENIZER_SHA,
            max_order=2,
        )
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            proposal_width=3,
            atlas=atlas,
        )
        prompt = (20, 21)
        provider.begin_request(prompt)

        proposal = provider.propose_round(prompt, 2)

        self.assertNotEqual(proposal.phrase_source, "atlas")
        self.assertEqual(provider.metrics().atlas_option_calls, 0)
        provider.discard_pending_proposal()
        provider.observe_final((*prompt, 2, 3))
        provider.close()

    def test_context_cap_keeps_the_highest_support_rows(self) -> None:
        atlas = MarkovTokenAtlas.build(
            (
                (1, 2, 1, 2, 1, 2, 1, 2),
                (3, 4, 3, 4),
            ),
            vocab_size=8,
            tokenizer_sha256=TOKENIZER_SHA,
            max_order=2,
            min_context_count=2,
            max_branches=2,
            max_contexts=1,
        )

        self.assertEqual(atlas.context_count, 1)
        continuation = atlas.continuation((1,), max_tokens=1)
        self.assertIsNotNone(continuation)
        assert continuation is not None
        self.assertEqual(continuation.token_ids, (2,))


if __name__ == "__main__":
    unittest.main()
