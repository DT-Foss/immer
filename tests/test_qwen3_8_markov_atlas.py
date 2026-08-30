from __future__ import annotations

from pathlib import Path
import hashlib
import json
import tempfile
import unittest
import zlib

from immer.runtimes.qwen3_8.markov_atlas import (
    LEGACY_MARKOV_ATLAS_PREFIX,
    LEGACY_MARKOV_ATLAS_SCHEMA,
    MARKOV_ATLAS_PREFIX,
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
        with self.assertRaises(MarkovAtlasError):
            MarkovTokenAtlas.from_bytes(encoded + b"junk")

    def test_external_candidate_receives_backoff_evidence_without_being_top1(
        self,
    ) -> None:
        atlas = self._atlas()

        evidence = atlas.token_evidence((31, 2, 3, 4, 5), 8)
        sequence = atlas.sequence_evidence((31, 2), (3, 4, 5))

        self.assertEqual(evidence.token_id, 8)
        self.assertEqual(evidence.context_order, 4)
        self.assertEqual(evidence.support, 1)
        self.assertEqual(evidence.total, 4)
        self.assertGreater(evidence.score, 0.0)
        self.assertEqual(tuple(row.token_id for row in sequence), (3, 4, 5))
        self.assertTrue(all(row.support == 4 for row in sequence))
        absent = atlas.token_evidence((31, 2, 3, 4, 5), 30)
        self.assertEqual(absent.support, 0)
        self.assertEqual(absent.score, 0.0)
        root = atlas.token_evidence((30,), 3)
        self.assertEqual(root.context_order, 0)
        self.assertEqual(root.support, 4)
        self.assertGreater(root.score, 0.0)

    def test_token_options_include_non_top1_retained_branches(self) -> None:
        atlas = self._atlas()
        history = (31, 2, 3, 4, 5)

        options = atlas.token_options(history)

        self.assertEqual(tuple(row.token_id for row in options), (6, 7, 8, 9))
        external = next(row for row in options if row.token_id == 8)
        self.assertEqual(external, atlas.token_evidence(history, 8))
        continuation = atlas.continuation(
            history,
            max_tokens=1,
            min_support=1,
            min_confidence=0.0,
        )
        self.assertIsNotNone(continuation)
        assert continuation is not None
        self.assertEqual(continuation.token_ids, (6,))

    def test_token_options_deduplicate_multi_order_evidence_to_strongest(self) -> None:
        atlas = self._atlas()
        history = (31, 2)

        options = atlas.token_options(history)

        self.assertEqual(tuple(row.token_id for row in options), (3, 2, 4, 5))
        self.assertEqual(sum(row.token_id == 3 for row in options), 1)
        strongest = options[0]
        self.assertEqual(strongest, atlas.token_evidence(history, 3))
        self.assertEqual(strongest.context_order, 1)
        self.assertTrue(all(row.context_order == 0 for row in options[1:]))

    def test_token_options_stably_order_ties_and_apply_limit(self) -> None:
        atlas = self._atlas()
        history = (31, 2, 3, 4, 5)

        expected = atlas.token_options(history)

        self.assertEqual(tuple(row.token_id for row in expected), (6, 7, 8, 9))
        self.assertEqual(atlas.token_options(history), expected)
        self.assertEqual(
            tuple(row.token_id for row in atlas.token_options(history, limit=3)),
            (6, 7, 8),
        )
        self.assertEqual(len(expected), atlas.max_branches)

    def test_token_options_reject_invalid_history_and_limit(self) -> None:
        atlas = self._atlas()

        for history in ((), (-1,), (32,), (True,), (2, 1.0)):
            with self.subTest(history=history):
                with self.assertRaisesRegex(ValueError, "history"):
                    atlas.token_options(history)
        for history in ("2", b"2", bytearray(b"2")):
            with self.subTest(history=history):
                with self.assertRaisesRegex(TypeError, "history"):
                    atlas.token_options(history)
        for limit in (0, -1, True, 1.0, atlas.max_branches + 1):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "limit"):
                    atlas.token_options((2,), limit=limit)

    def test_token_options_match_after_compact_v2_reload(self) -> None:
        atlas = self._atlas()
        encoded = atlas.to_bytes()
        restored = MarkovTokenAtlas.from_bytes(encoded)
        history = (31, 2, 3, 4, 5)

        expected = atlas.token_options(history)
        actual = restored.token_options(history)

        self.assertTrue(encoded.startswith(MARKOV_ATLAS_PREFIX))
        self.assertEqual(actual, expected)
        self.assertEqual(restored.to_bytes(), encoded)

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

    def test_legacy_json_atlas_migrates_to_compact_binary_codec(self) -> None:
        body = {
            "document_count": 2,
            "max_branches": 2,
            "max_order": 2,
            "min_context_count": 2,
            "rows": [
                [[2], 2, [[3, 2]]],
                [[2, 3], 2, [[4, 2]]],
            ],
            "token_count": 6,
            "tokenizer_sha256": TOKENIZER_SHA,
            "vocab_size": 32,
        }
        canonical_body = json.dumps(
            body,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        document = {
            "body": body,
            "schema": LEGACY_MARKOV_ATLAS_SCHEMA,
            "sha256": hashlib.sha256(canonical_body).hexdigest(),
        }
        raw = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()

        atlas = MarkovTokenAtlas.from_bytes(
            LEGACY_MARKOV_ATLAS_PREFIX + zlib.compress(raw)
        )

        continuation = atlas.continuation((9, 2), max_tokens=2)
        self.assertIsNotNone(continuation)
        assert continuation is not None
        self.assertEqual(continuation.token_ids, (3, 4))
        self.assertTrue(atlas.to_bytes().startswith(MARKOV_ATLAS_PREFIX))

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

    def test_position_zero_beam_failure_also_gates_the_atlas_phrase_floor(
        self,
    ) -> None:
        atlas = self._atlas()
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            proposal_width=3,
            atlas=atlas,
        )
        prompt = (20, 21)
        provider.begin_request(prompt)
        provider._beam_position_verified[0] = 1
        provider._beam_position_hits[0] = 0

        proposal = provider.propose_round(prompt, 2)

        self.assertEqual(proposal.phrase_source, "atlas")
        self.assertLess(proposal.phrase_confidence, 0.17)
        self.assertEqual(proposal.recommended_window, 1)
        provider.discard_pending_proposal()
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

    def test_pending_provider_scores_external_tokens_as_atlas_votes(self) -> None:
        atlas = self._atlas()
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            proposal_width=3,
            atlas=atlas,
        )
        prompt = (20, 21)
        provider.begin_request(prompt)
        provider.propose_round(prompt, 2)

        rows = provider.atlas_evidence_for_pending((3, 4, 5))

        self.assertEqual(tuple(row.token_id for row in rows), (3, 4, 5))
        self.assertTrue(all(row.score > 0.0 for row in rows))
        metrics = provider.metrics()
        self.assertEqual(metrics.atlas_vote_calls, 1)
        self.assertEqual(metrics.atlas_vote_tokens, 3)
        self.assertEqual(metrics.atlas_vote_supported_tokens, 3)
        self.assertGreater(metrics.atlas_vote_score_sum, 0.0)
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
            max_contexts=2,
        )

        self.assertEqual(atlas.context_count, 2)
        continuation = atlas.continuation((1,), max_tokens=1)
        self.assertIsNotNone(continuation)
        assert continuation is not None
        self.assertEqual(continuation.token_ids, (2,))


if __name__ == "__main__":
    unittest.main()
