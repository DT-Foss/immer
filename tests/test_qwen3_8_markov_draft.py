from __future__ import annotations

from pathlib import Path
from dataclasses import replace
import json
import math
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
from immer.runtimes.qwen3_8.markov_atlas import AtlasTokenEvidence
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
        self.source = Streamer.from_local(self.root, budget_mb=1000, use_cache=False)
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
            imported_episode_sha256s=("a" * 64,),
        )
        encoded = state.to_bytes()

        self.assertEqual(MarkovDraftState.from_bytes(encoded), state)
        damaged = bytearray(encoded)
        damaged[-1] ^= 1
        with self.assertRaises(MarkovDraftError):
            MarkovDraftState.from_bytes(bytes(damaged))

    def test_default_provider_expands_online_history_without_expanding_ppm_window(
        self,
    ) -> None:
        path = self.root / "growing-history.bin"
        legacy = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=path,
            max_history_tokens=4096,
        )
        legacy.observe_final((1, 2, 3, 4))
        legacy.close()

        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=path,
        )

        self.assertEqual(provider._state.max_history_tokens, 65_536)
        self.assertEqual(provider.metrics().history_capacity_tokens, 65_536)
        self.assertEqual(max(row.window for row in provider._experts), 4096)
        provider.observe_final((5, 6, 7, 8))
        provider.close()
        self.assertEqual(
            MarkovDraftState.from_bytes(path.read_bytes()).max_history_tokens,
            65_536,
        )

    def test_persistent_ppm_models_are_reused_across_one_request(self) -> None:
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=64,
            max_history_tokens=256,
        )
        provider.observe_final((1, 2, 3, 4, 1, 2, 3, 5))
        first = provider._expert_models((9, 10, 11))
        second = provider._expert_models((9, 10, 11, 12))

        for spec, first_row, second_row in zip(
            provider._experts,
            first,
            second,
            strict=True,
        ):
            if spec.local_only:
                self.assertIsNot(first_row[0], second_row[0])
            else:
                self.assertIs(first_row[0], second_row[0])
        provider.close()

    def test_live_answer_memory_scores_external_mtp_candidates(self) -> None:
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=64,
            max_history_tokens=256,
            proposal_width=3,
        )
        episodes = (
            (2, 3, 4, 5),
            (2, 3, 4, 6),
            (2, 3, 4, 7),
            (2, 3, 4, 8),
        )
        provider._state = replace(
            provider._state,
            token_ids=tuple(token for row in episodes for token in row),
            episode_lengths=tuple(len(row) for row in episodes),
            episode_dialects=(None,) * len(episodes),
            episode_prompt_lengths=(None,) * len(episodes),
            expert_observations=(16,) * len(provider._experts),
            expert_hits=(12,) * len(provider._experts),
        )
        prompt = (20, 21)
        provider.begin_request(prompt)
        provider.propose_round(prompt, 2)

        rows = provider.language_evidence_for_pending((3, 4, 5))

        self.assertEqual(tuple(row.token_id for row in rows), (3, 4, 5))
        self.assertTrue(all(row.atlas_score == 0.0 for row in rows))
        self.assertGreater(rows[0].online_score, 0.0)
        self.assertGreater(rows[1].online_score, 0.0)
        metrics = provider.metrics()
        self.assertEqual(metrics.online_vote_calls, 1)
        self.assertEqual(metrics.online_vote_tokens, 3)
        self.assertGreaterEqual(metrics.online_vote_supported_tokens, 2)
        provider.discard_pending_proposal()
        provider.observe_final((*prompt, 2, 3))
        provider.close()

    def test_uncalibrated_live_memory_cannot_boost_mtp_confidence(self) -> None:
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=64,
            max_history_tokens=128,
            proposal_width=3,
        )
        episodes = ((2, 3, 4), (2, 3, 5), (2, 3, 6))
        provider._state = replace(
            provider._state,
            token_ids=tuple(token for row in episodes for token in row),
            episode_lengths=(3, 3, 3),
            episode_dialects=(None, None, None),
            episode_prompt_lengths=(None, None, None),
        )
        prompt = (20, 21)
        provider.begin_request(prompt)
        provider.propose_round(prompt, 2)

        rows = provider.language_evidence_for_pending((3, 4, 5))

        self.assertGreater(rows[0].online_support, 0)
        self.assertTrue(all(row.online_score == 0.0 for row in rows))
        provider.discard_pending_proposal()
        provider.observe_final((*prompt, 2, 3))
        provider.close()

    def test_episode_scorer_receives_only_the_confirmed_answer(self) -> None:
        scored = []
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=64,
            max_history_tokens=128,
            episode_scorer=lambda tokens: scored.append(tokens) or 7.5,
            episode_priority=lambda _tokens: 7.5,
        )
        prompt = (20, 21, 22)
        answer = (3, 4, 5)
        provider.begin_request(prompt)

        provider.observe_final((*prompt, *answer))

        self.assertEqual(scored, [answer])
        metrics = provider.metrics()
        self.assertEqual(metrics.retention_scored_episodes, 1)
        self.assertEqual(metrics.retention_failures, 0)
        self.assertEqual(metrics.last_retention_priority, 7.5)
        provider.close()

    def test_o1_priority_retains_valuable_old_episode_over_low_value_newer_one(
        self,
    ) -> None:
        valuable = (1, 1, 1, 1)
        expendable = (2, 2, 2, 2)
        incoming = (3, 3, 3, 3, 3, 3)
        priorities = {valuable: 10.0, expendable: 0.1}
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=64,
            max_history_tokens=12,
            max_order=8,
            episode_priority=lambda tokens: priorities.get(tokens, 1.0),
        )
        provider._state = replace(
            provider._state,
            token_ids=(*valuable, *expendable),
            episode_lengths=(4, 4),
            episode_dialects=(None, None),
            episode_prompt_lengths=(None, None),
        )
        provider._activate_dialect((20, 21))

        provider._learn_episode(
            incoming,
            prompt_length=None,
            priority=5.0,
        )

        self.assertEqual(provider.confirmed_episodes(), (valuable, incoming))
        self.assertEqual(provider.metrics().retention_priority_evictions, 1)
        provider.close()

    def test_oversized_answer_suffix_keeps_its_o1_priority_alias(self) -> None:
        aliases = {}

        def remember(tokens, priority):
            aliases[tokens] = priority

        provider = FingerprintRollingK4DraftProvider(
            vocab_size=64,
            max_order=8,
            max_history_tokens=12,
            episode_priority_store=remember,
        )
        provider._activate_dialect((30, 31))
        episode = tuple(range(1, 21))

        provider._learn_episode(
            episode,
            prompt_length=5,
            priority=9.0,
        )

        retained = episode[-12:]
        self.assertEqual(provider.confirmed_episodes(), (retained,))
        self.assertEqual(provider._state.episode_prompt_lengths, (None,))
        self.assertEqual(aliases, {retained: 9.0})
        provider.close()

    def test_confirmed_request_persists_its_prompt_boundary(self) -> None:
        provider = FingerprintRollingK4DraftProvider(vocab_size=32)
        prompt = (1, 4, 7)
        provider.begin_request(prompt)
        provider.observe_final((*prompt, 9, 10))

        self.assertEqual(provider._state.episode_lengths, (5,))
        self.assertEqual(provider._state.episode_prompt_lengths, (3,))
        provider.close()

    def test_draft_corpus_contains_generated_answers_not_chat_prompts(self) -> None:
        provider = FingerprintRollingK4DraftProvider(vocab_size=64)
        episodes = (
            (1, 2, 3, 40, 41, 42),
            (7, 8, 9, 50, 51),
            (60, 61),
        )
        provider._state = replace(
            provider._state,
            token_ids=tuple(token for episode in episodes for token in episode),
            episode_lengths=tuple(map(len, episodes)),
            episode_dialects=(None, None, None),
            episode_prompt_lengths=(3, 3, None),
        )

        self.assertEqual(
            provider._generation_episodes(),
            ((40, 41, 42), (50, 51), (60, 61)),
        )
        self.assertEqual(
            provider._persistent_symbols(),
            ("40", "41", "42", "<episode>", "50", "51", "<episode>", "60", "61"),
        )
        provider.close()

    def test_council_can_resume_after_another_provider_commits_prefix(self) -> None:
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=64,
            proposal_width=3,
        )
        prompt = (1, 2, 3)
        provider.begin_request(prompt)
        external = (*prompt, 40, 41)

        provider.advance_confirmed_prefix(external)
        proposal = provider.propose_round(external, 42)

        self.assertEqual(len(proposal.token_ids), 3)
        provider.discard_pending_proposal()
        provider.advance_confirmed_prefix((*external, 42))
        provider.observe_final((*external, 42, 43))
        self.assertEqual(provider.confirmed_transitions(), ((prompt, (40, 41, 42, 43)),))
        provider.close()

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

    def test_v2_council_state_migrates_to_v9_planner_memory(self) -> None:
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

        self.assertTrue(path.read_bytes().startswith(b"IMMD\x09"))
        migrated = MarkovDraftState.from_bytes(path.read_bytes())
        self.assertEqual(len(migrated.dialects), 1)

    def test_v3_dialect_state_migrates_episode_bindings_to_v9(self) -> None:
        encoded = MarkovDraftState(
            vocab_size=32,
            max_history_tokens=64,
            token_ids=(1, 2, 3),
            episode_lengths=(3,),
        ).to_bytes()
        document = json.loads(zlib.decompress(encoded[5:]))
        document.pop("episode_dialects")
        document.pop("episode_prompt_lengths")
        document.pop("imported_episode_sha256s")
        document.pop("horizon_expert_hits")
        document.pop("horizon_expert_observations")
        document.pop("lookahead_greedy_hits")
        document.pop("lookahead_hits")
        document.pop("lookahead_observations")
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
        self.assertTrue(path.read_bytes().startswith(b"IMMD\x09"))

    def test_v4_state_migrates_empty_import_inventory_to_v9(self) -> None:
        encoded = MarkovDraftState(
            vocab_size=32,
            max_history_tokens=64,
            token_ids=(1, 2, 3),
            episode_lengths=(3,),
        ).to_bytes()
        document = json.loads(zlib.decompress(encoded[5:]))
        document.pop("imported_episode_sha256s")
        document.pop("episode_prompt_lengths")
        document.pop("horizon_expert_hits")
        document.pop("horizon_expert_observations")
        document.pop("lookahead_greedy_hits")
        document.pop("lookahead_hits")
        document.pop("lookahead_observations")
        document["schema"] = "immer.qwen3.8-markov-draft-state/v4"
        raw = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        path = self.root / "v4-state.bin"
        path.write_bytes(b"IMMD\x04" + zlib.compress(raw, level=9))

        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=path,
            max_history_tokens=64,
        )

        self.assertEqual(provider.imported_episode_sha256s(), ())
        provider.close()
        self.assertTrue(path.read_bytes().startswith(b"IMMD\x09"))

    def test_v5_state_migrates_unknown_prompt_boundaries_to_v9(self) -> None:
        encoded = MarkovDraftState(
            vocab_size=32,
            max_history_tokens=64,
            token_ids=(1, 2, 3),
            episode_lengths=(3,),
        ).to_bytes()
        document = json.loads(zlib.decompress(encoded[5:]))
        document.pop("episode_prompt_lengths")
        document.pop("horizon_expert_hits")
        document.pop("horizon_expert_observations")
        document.pop("lookahead_greedy_hits")
        document.pop("lookahead_hits")
        document.pop("lookahead_observations")
        document["schema"] = "immer.qwen3.8-markov-draft-state/v5"
        raw = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        path = self.root / "v5-state.bin"
        path.write_bytes(b"IMMD\x05" + zlib.compress(raw, level=9))

        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=path,
            max_history_tokens=64,
        )

        self.assertEqual(provider._state.episode_prompt_lengths, (None,))
        provider.close()
        self.assertTrue(path.read_bytes().startswith(b"IMMD\x09"))

    def test_v6_state_migrates_zeroed_position_expert_memory_to_v9(self) -> None:
        encoded = MarkovDraftState(
            vocab_size=32,
            max_history_tokens=64,
            token_ids=(1, 2, 3),
            episode_lengths=(3,),
        ).to_bytes()
        document = json.loads(zlib.decompress(encoded[5:]))
        document.pop("horizon_expert_hits")
        document.pop("horizon_expert_observations")
        document.pop("lookahead_greedy_hits")
        document.pop("lookahead_hits")
        document.pop("lookahead_observations")
        document["schema"] = "immer.qwen3.8-markov-draft-state/v6"
        raw = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        path = self.root / "v6-state.bin"
        path.write_bytes(b"IMMD\x06" + zlib.compress(raw, level=9))

        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=path,
            max_history_tokens=64,
        )

        self.assertEqual(len(provider._state.horizon_expert_observations), 16)
        self.assertTrue(
            all(not any(row) for row in provider._state.horizon_expert_observations)
        )
        provider.close()
        self.assertTrue(path.read_bytes().startswith(b"IMMD\x09"))

    def test_v7_state_migrates_dialect_position_memory_to_v9(self) -> None:
        seed = FingerprintRollingK4DraftProvider(vocab_size=32)
        seed.observe_final((1, 2, 3))
        document = json.loads(zlib.decompress(seed._state.to_bytes()[5:]))
        seed.close()
        for dialect in document["dialects"]:
            dialect.pop("horizon_hits")
            dialect.pop("horizon_observations")
        document.pop("lookahead_greedy_hits")
        document.pop("lookahead_hits")
        document.pop("lookahead_observations")
        document["schema"] = "immer.qwen3.8-markov-draft-state/v7"
        raw = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        path = self.root / "v7-state.bin"
        path.write_bytes(b"IMMD\x07" + zlib.compress(raw, level=9))

        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=path,
        )

        self.assertEqual(len(provider._state.dialects), 1)
        dialect = provider._state.dialects[0]
        self.assertEqual(len(dialect.horizon_observations), 16)
        self.assertTrue(all(not any(row) for row in dialect.horizon_observations))
        provider.close()
        self.assertTrue(path.read_bytes().startswith(b"IMMD\x09"))

    def test_v8_state_migrates_zeroed_lookahead_outcomes_to_v9(self) -> None:
        provider = FingerprintRollingK4DraftProvider(vocab_size=32)
        document = json.loads(zlib.decompress(provider._state.to_bytes()[5:]))
        provider.close()
        document.pop("lookahead_greedy_hits")
        document.pop("lookahead_hits")
        document.pop("lookahead_observations")
        document["schema"] = "immer.qwen3.8-markov-draft-state/v8"
        raw = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        path = self.root / "v8-state.bin"
        path.write_bytes(b"IMMD\x08" + zlib.compress(raw, level=9))

        migrated = MarkovDraftState.from_bytes(path.read_bytes())
        self.assertEqual(migrated.lookahead_observations, (0,) * 16)
        self.assertEqual(migrated.lookahead_hits, (0,) * 16)
        self.assertEqual(migrated.lookahead_greedy_hits, (0,) * 16)
        self.assertEqual(
            MarkovDraftState.from_bytes(migrated.to_bytes()),
            migrated,
        )

        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            state_path=path,
        )

        self.assertEqual(provider._state.lookahead_observations, (0,) * 16)
        self.assertEqual(provider._state.lookahead_hits, (0,) * 16)
        self.assertEqual(provider._state.lookahead_greedy_hits, (0,) * 16)
        provider.close()
        self.assertTrue(path.read_bytes().startswith(b"IMMD\x09"))

    def test_import_digest_survives_episode_eviction_and_prevents_replay(self) -> None:
        state_path = self.root / "imported-markov.bin"
        first_episode = tuple(range(1, 41))
        second_episode = tuple(range(41, 101))
        first_digest = "1" * 64
        second_digest = "2" * 64

        provider = FingerprintRollingK4DraftProvider(
            vocab_size=128,
            state_path=state_path,
            max_history_tokens=64,
        )
        self.assertTrue(provider.import_confirmed_episode(first_episode, first_digest))
        provider.close()

        provider = FingerprintRollingK4DraftProvider(
            vocab_size=128,
            state_path=state_path,
            max_history_tokens=64,
        )
        self.assertFalse(provider.import_confirmed_episode(first_episode, first_digest))
        self.assertTrue(
            provider.import_confirmed_episode(second_episode, second_digest)
        )
        provider.close()

        restored = MarkovDraftState.from_bytes(state_path.read_bytes())
        self.assertNotIn(1, restored.token_ids)
        self.assertEqual(restored.updates, 2)
        self.assertEqual(
            restored.imported_episode_sha256s,
            (first_digest, second_digest),
        )

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

        result = Qwen38K4SpeculativeDecoder(target, provider).generate_rolling(
            [prompt],
            max_new_tokens=8,
            head_block_rows=7,
        )

        self.assertEqual(result.token_ids, expected)
        self.assertLess(
            result.evidence.forward_passes, baseline_evidence.forward_passes
        )
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
        _assert_layer_states_equal(
            self, candidate._layer_states, reference._layer_states
        )
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

        result = Qwen38K4SpeculativeDecoder(target, provider).generate_rolling(
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
        proposal, feedback, _confidence, _disagreement = provider._predict_council(
            history, 1
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

    def test_target_confirmed_expert_evidence_calibrates_draft_confidence(
        self,
    ) -> None:
        class FixedExpert:
            @staticmethod
            def distribution(_context):
                return {
                    "07": 0.9,
                    "08": 0.05,
                    markov_module._UNKNOWN_TOKEN: 0.05,
                }

        def fixed_models(provider):
            return tuple((FixedExpert(), []) for _ in provider._experts)

        cold = FingerprintRollingK4DraftProvider(vocab_size=32)
        cold._expert_models = lambda _history: fixed_models(cold)
        _tokens, _feedback, cold_confidence, _disagreement = cold._predict_council(
            (1,), 1
        )
        self.assertAlmostEqual(cold_confidence[0], 0.9)
        self.assertEqual(cold.metrics().last_empirical_evidence, 0.0)
        cold.close()

        trained = FingerprintRollingK4DraftProvider(vocab_size=32)
        trained._state = replace(
            trained._state,
            expert_observations=(100,) * len(trained._experts),
            expert_hits=(70,) * len(trained._experts),
        )
        trained._expert_models = lambda _history: fixed_models(trained)

        tokens, _feedback, confidence, disagreement = trained._predict_council(
            (1,), 3
        )
        proposal = RollingDraftProposal.build(
            tokens,
            confidence,
            disagreement,
            request_window_ceiling=4,
            provider_abi=markov_module.MARKOV_DRAFT_PROVIDER_ABI,
        )

        expected_evidence = (100 / 108) * (71 / 102) * (0.85 / 0.95)
        expected_confidence = 1.0 - (1.0 - 0.9) * (1.0 - expected_evidence)
        self.assertTrue(all(token == 7 for token in tokens))
        self.assertTrue(
            all(abs(value - expected_confidence) < 1e-12 for value in confidence)
        )
        metrics = trained.metrics()
        self.assertAlmostEqual(metrics.last_raw_confidence, 0.9)
        self.assertAlmostEqual(metrics.last_empirical_evidence, expected_evidence)
        self.assertEqual(
            proposal.select_window(
                request_window_ceiling=4,
                remaining_tokens=4,
                window_work_costs={1: 1.0, 2: 1.6, 4: 2.8},
            ).chosen_window,
            4,
        )

        class NearTieExpert:
            @staticmethod
            def distribution(_context):
                return {
                    "07": 0.101,
                    "08": 0.1,
                    markov_module._UNKNOWN_TOKEN: 0.799,
                }

        trained._expert_models = lambda _history: tuple(
            (NearTieExpert(), []) for _ in trained._experts
        )
        _tokens, _feedback, near_tie, _disagreement = trained._predict_council(
            (1,), 1
        )
        self.assertLess(near_tie[0], 0.12)

        class UnknownDominantExpert:
            @staticmethod
            def distribution(_context):
                return {
                    "07": 0.11,
                    markov_module._UNKNOWN_TOKEN: 0.89,
                }

        trained._expert_models = lambda _history: tuple(
            (UnknownDominantExpert(), []) for _ in trained._experts
        )
        _tokens, _feedback, unknown_dominant, _disagreement = (
            trained._predict_council((1,), 1)
        )
        self.assertAlmostEqual(unknown_dominant[0], 0.11)
        self.assertEqual(trained.metrics().last_empirical_evidence, 0.0)

        trained._expert_models = lambda _history: fixed_models(trained)
        _tokens, _feedback, forced_confidence, _disagreement = (
            trained._predict_council((1,), 1, forced_prefix=(7,))
        )
        self.assertAlmostEqual(forced_confidence[0], 0.9)
        self.assertEqual(trained.metrics().last_empirical_evidence, 0.0)
        trained.close()

    def test_horizon_position_uses_its_own_target_confirmed_posterior(self) -> None:
        class DecisiveExpert:
            @staticmethod
            def distribution(_context):
                return {
                    "07": 0.4,
                    "08": 0.1,
                    markov_module._UNKNOWN_TOKEN: 0.1,
                }

        provider = FingerprintRollingK4DraftProvider(vocab_size=32)
        width = len(provider._experts)
        observations = [[0] * width for _ in range(16)]
        hits = [[0] * width for _ in range(16)]
        observations[0] = [100] * width
        hits[0] = [90] * width
        observations[1] = [100] * width
        hits[1] = [10] * width
        provider._state = replace(
            provider._state,
            expert_observations=(100,) * width,
            expert_hits=(70,) * width,
            horizon_expert_observations=tuple(tuple(row) for row in observations),
            horizon_expert_hits=tuple(tuple(row) for row in hits),
        )
        provider._expert_models = lambda _history: tuple(
            (DecisiveExpert(), []) for _ in provider._experts
        )

        tokens, feedback, confidence, _disagreement = provider._predict_council(
            (1,), 3
        )

        self.assertEqual(tokens, (7, 7, 7))
        self.assertGreater(confidence[0], confidence[2])
        self.assertGreater(confidence[2], confidence[1])
        self.assertGreater(confidence[1], 0.4)
        provider._apply_council_feedback(feedback[2], 7, 2)
        metrics = provider.metrics()
        self.assertEqual(metrics.horizon_observations[:3], (100, 100, 1))
        self.assertGreater(metrics.horizon_weighted_accuracy[0], 0.8)
        self.assertLess(metrics.horizon_weighted_accuracy[1], 0.2)
        restored = MarkovDraftState.from_bytes(provider._state.to_bytes())
        self.assertEqual(restored.horizon_expert_observations[2], (1,) * width)
        self.assertEqual(restored.horizon_expert_hits[2], (1,) * width)
        provider.close()

    def test_position_specialists_can_change_the_recursive_token_choice(self) -> None:
        class TokenExpert:
            def __init__(self, token: int) -> None:
                self.token = token

            def distribution(self, _context):
                other = 8 if self.token == 7 else 7
                return {
                    f"{self.token:02d}": 0.9,
                    f"{other:02d}": 0.05,
                    markov_module._UNKNOWN_TOKEN: 0.05,
                }

        provider = FingerprintRollingK4DraftProvider(vocab_size=32)
        width = len(provider._experts)
        observations = [[0] * width for _ in range(16)]
        hits = [[0] * width for _ in range(16)]
        observations[1] = [20] * width
        hits[1] = [0, 0, 0, 0, 0, 20, 20, 20]
        provider._state = replace(
            provider._state,
            expert_log_weights=(-2.0, -2.0, -2.0, -2.0, 4.0, -2.0, -2.0, -2.0),
            horizon_expert_observations=tuple(tuple(row) for row in observations),
            horizon_expert_hits=tuple(tuple(row) for row in hits),
        )
        provider._expert_models = lambda _history: tuple(
            (TokenExpert(8 if index >= 5 else 7), [])
            for index in range(width)
        )
        global_weights = provider._weights()
        position_weights, maturity, dialect_maturity = provider._position_weighting(
            1,
            global_weights,
        )

        tokens, _feedback, _confidence, _disagreement = provider._predict_council(
            (1,), 2
        )

        self.assertEqual(tokens, (7, 8))
        self.assertAlmostEqual(maturity, 20 / 28)
        self.assertEqual(dialect_maturity, 0.0)
        self.assertLess(position_weights[4], global_weights[4])
        self.assertGreater(sum(position_weights[5:]), sum(global_weights[5:]))
        self.assertAlmostEqual(sum(position_weights), 1.0)
        self.assertTrue(
            all(value >= provider.FIXED_SHARE / width for value in position_weights)
        )
        self.assertEqual(provider.metrics().last_position, 1)
        self.assertAlmostEqual(provider.metrics().last_position_maturity, 20 / 28)
        self.assertEqual(provider.metrics().position_specialist_predictions, 1)
        self.assertAlmostEqual(provider.metrics().max_position_maturity, 20 / 28)
        provider.close()

    def test_dialect_beta_skill_changes_the_contextual_specialist_choice(self) -> None:
        class TokenExpert:
            def __init__(self, token: int) -> None:
                self.token = token

            def distribution(self, _context):
                other = 8 if self.token == 7 else 7
                return {
                    f"{self.token:02d}": 0.9,
                    f"{other:02d}": 0.05,
                    markov_module._UNKNOWN_TOKEN: 0.05,
                }

        provider = FingerprintRollingK4DraftProvider(vocab_size=32)
        width = len(provider._experts)
        provider._state = replace(
            provider._state,
            expert_log_weights=(-2.0, -2.0, -2.0, -2.0, 4.0, -2.0, -2.0, -2.0),
        )
        provider._expert_models = lambda _history: tuple(
            (TokenExpert(8 if index >= 5 else 7), [])
            for index in range(width)
        )
        global_token = provider._predict_council((1,), 1)[0]
        dialect_observations = [[0] * width for _ in range(16)]
        dialect_hits = [[0] * width for _ in range(16)]
        dialect_observations[0] = [20] * width
        dialect_hits[0] = [0, 0, 0, 0, 0, 20, 20, 20]
        provider._active_dialect = MarkovDialectState(
            dialect_id="d" * 64,
            signature=(1,),
            visits=1,
            last_seen=0,
            rapidities=(0.0,) * width,
            observations=(20,) * width,
            hits=(0, 0, 0, 0, 0, 20, 20, 20),
            horizon_observations=tuple(
                tuple(row) for row in dialect_observations
            ),
            horizon_hits=tuple(tuple(row) for row in dialect_hits),
        )
        provider._active_dialect_similarity = 1.0

        dialect_token = provider._predict_council((1,), 1)[0]
        dialect_metrics = provider.metrics()
        deep_unobserved = provider._predict_council(
            (1,),
            1,
            position_offset=7,
        )[0]

        self.assertEqual(global_token, (7,))
        self.assertEqual(dialect_token, (8,))
        self.assertAlmostEqual(
            dialect_metrics.last_dialect_skill_maturity,
            20 / 28,
        )
        self.assertEqual(deep_unobserved, (7,))
        metrics = provider.metrics()
        self.assertEqual(metrics.last_position_maturity, 0.0)
        self.assertEqual(metrics.last_dialect_skill_maturity, 0.0)
        self.assertEqual(metrics.dialect_specialist_predictions, 1)
        self.assertAlmostEqual(metrics.max_dialect_skill_maturity, 20 / 28)
        provider.close()

    def test_dialect_council_uses_similarity_visits_and_ricci_age(self) -> None:
        provider = FingerprintRollingK4DraftProvider(vocab_size=64)
        prompt = tuple((1, 2, 3, 4) * 16)
        signature = provider._context_signature(prompt)
        shared = signature[: max(1, len(signature) // 2)]
        extras = tuple(
            value
            for value in range(1_000, 2_000)
            if value not in set(signature)
        )[: len(signature) - len(shared)]
        partial_signature = tuple(sorted((*shared, *extras)))
        width = len(provider._experts)
        zeros = tuple((0,) * width for _ in range(16))

        def dialect(digest: str, dialect_signature, *, visits: int, last_seen: int):
            return MarkovDialectState(
                dialect_id=digest * 64,
                signature=dialect_signature,
                visits=visits,
                last_seen=last_seen,
                rapidities=(0.0,) * width,
                observations=(0,) * width,
                hits=(0,) * width,
                horizon_observations=zeros,
                horizon_hits=zeros,
            )

        exact = dialect("a", signature, visits=1, last_seen=0)
        popular = dialect("b", partial_signature, visits=10, last_seen=100)
        weak = tuple(
            dialect(
                digest,
                tuple(sorted((*signature[:-1], 10_000 + index))),
                visits=1,
                last_seen=0,
            )
            for index, digest in enumerate(("c", "d", "e"))
        )
        provider._state = replace(
            provider._state,
            clock=100,
            dialects=tuple(
                sorted((exact, popular, *weak), key=lambda row: row.dialect_id)
            ),
        )

        provider.begin_request(prompt)

        neighbors = provider._dialect_neighbors
        self.assertEqual(provider._active_dialect.dialect_id, exact.dialect_id)
        self.assertEqual(len(neighbors), 4)
        self.assertAlmostEqual(sum(row[1] for row in neighbors), 1.0)
        weights = {row[2].dialect_id: row[1] for row in neighbors}
        self.assertIn(popular.dialect_id, weights)
        self.assertGreater(weights[popular.dialect_id], weights[exact.dialect_id])
        expected_ratio = (
            provider._dialect_similarity(signature, partial_signature) * 10
        ) / math.exp(-provider.RICCI_AGE_ALPHA * 100)
        self.assertAlmostEqual(
            weights[popular.dialect_id] / weights[exact.dialect_id],
            expected_ratio,
        )
        metrics = provider.metrics()
        self.assertEqual(metrics.dialect_neighbor_count, 4)
        self.assertGreater(metrics.dialect_neighbor_effective, 1.0)
        self.assertEqual(metrics.dialect_neighbor_max_similarity, 1.0)
        self.assertIn("a" * 64, metrics.dialect_neighbor_ids)
        self.assertIn("b" * 64, metrics.dialect_neighbor_ids)
        provider.close()

    def test_one_step_lookahead_can_choose_a_better_markov_sequence(self) -> None:
        calls = []

        class PlanningExpert:
            @staticmethod
            def distribution(context):
                calls.append(tuple(context))
                if not context:
                    return {"07": 0.55, "08": 0.45}
                if context[-1] == "07":
                    return {"09": 0.51, "10": 0.49}
                return {"09": 0.99, "10": 0.01}

        provider = FingerprintRollingK4DraftProvider(vocab_size=32)
        provider._expert_models = lambda history: tuple(
            (
                PlanningExpert(),
                [provider._symbol(token) for token in history if 7 <= token <= 12],
            )
            for _ in provider._experts
        )

        planned = provider._predict_council((1,), 1)[0]
        planned_metrics = provider.metrics()
        planned_call_count = len(calls)
        forced = provider._predict_council((1,), 1, forced_prefix=(7,))[0]
        terminal = provider._predict_council((1,), 1, position_offset=15)[0]

        self.assertEqual(planned, (8,))
        self.assertEqual(planned_metrics.lookahead_calls, 1)
        self.assertEqual(planned_metrics.lookahead_candidates, 2)
        self.assertEqual(planned_metrics.lookahead_token_changes, 1)
        self.assertGreater(planned_metrics.last_lookahead_gain, 0.05)
        self.assertEqual(planned_call_count, len(provider._experts) * 3)
        self.assertEqual(forced, (7,))
        self.assertEqual(terminal, (7,))
        self.assertEqual(provider.metrics().lookahead_calls, 1)
        self.assertEqual(provider.metrics().last_lookahead_gain, 0.0)
        self.assertEqual(len(calls), len(provider._experts) * 5)
        provider.close()

    def test_atlas_online_beam_can_choose_non_top1_for_a_stronger_path(self) -> None:
        class PlanningExpert:
            @staticmethod
            def distribution(context):
                if not context:
                    return {"07": 0.55, "08": 0.45}
                if context[-1] == "07":
                    return {"09": 0.51, "10": 0.49}
                return {"09": 0.99, "10": 0.01}

        class Atlas:
            max_branches = 8
            context_count = 2
            token_count = 10

            @staticmethod
            def token_options(history, *, limit=None):
                if len(history) > 1:
                    return ()
                return (
                    AtlasTokenEvidence(7, 1, 8, 10, 0.8, 0.6),
                    AtlasTokenEvidence(8, 1, 6, 10, 0.6, 0.5),
                )

        provider = FingerprintRollingK4DraftProvider(vocab_size=32)
        provider.atlas = Atlas()
        provider._expert_models = lambda history: tuple(
            (
                PlanningExpert(),
                [provider._symbol(token) for token in history if 7 <= token <= 12],
            )
            for _ in provider._experts
        )

        result = provider._predict_beam((1,), 2)

        self.assertIsNotNone(result)
        assert result is not None
        tokens, feedback, confidences, disagreements = result
        self.assertEqual(tokens, (8, 9))
        self.assertEqual(len(feedback), 2)
        self.assertTrue(all(row[0]["09"] == 0.99 for row in feedback[1]))
        self.assertEqual(len(confidences), 2)
        self.assertTrue(all(0.0 < value < 1.0 for value in confidences))
        self.assertEqual(len(disagreements), 2)
        self.assertEqual(provider._last_plan_trace[0][0], 8)
        self.assertEqual(provider._last_plan_trace[0][1], 7)
        provider._beam_verified_tokens = 3
        provider._beam_accepted_tokens = 0
        discounted = provider._predict_beam((1,), 2)
        self.assertIsNotNone(discounted)
        assert discounted is not None
        reliability = 0.5 / 5
        for before, after in zip(confidences, discounted[2], strict=True):
            self.assertAlmostEqual(after, before * reliability)
        provider._beam_verified_tokens = 0
        provider._beam_accepted_tokens = 0
        wide = provider._predict_beam((1,), 4)
        self.assertIsNotNone(wide)
        assert wide is not None
        initial_round = RollingDraftProposal.build(
            wide[0][:3],
            wide[2][:3],
            wide[3][:3],
            request_window_ceiling=4,
            provider_abi=markov_module.MARKOV_DRAFT_PROVIDER_ABI,
        )
        provider._beam_verified_tokens = 1
        provider._beam_accepted_tokens = 0
        cooled = provider._predict_beam((1,), 4)
        self.assertIsNotNone(cooled)
        assert cooled is not None
        cooled_round = RollingDraftProposal.build(
            cooled[0][:3],
            cooled[2][:3],
            cooled[3][:3],
            request_window_ceiling=4,
            provider_abi=markov_module.MARKOV_DRAFT_PROVIDER_ABI,
        )
        self.assertGreater(initial_round.recommended_window, 1)
        self.assertEqual(cooled_round.recommended_window, 1)
        provider.close()

    def test_beam_proposal_keeps_feedback_reconciliation_exact(self) -> None:
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

        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            proposal_width=3,
        )
        provider.atlas = Atlas()
        provider._expert_models = lambda history: tuple(
            (
                PlanningExpert(),
                [provider._symbol(token) for token in history if 7 <= token <= 12],
            )
            for _ in provider._experts
        )
        prompt = (20, 21)
        provider.begin_request(prompt)

        proposal = provider.propose_round(prompt, 2)
        committed = (*prompt, 2, *proposal.token_ids[:2])
        provider.reconcile_prefix(committed)
        provider.observe_final((*committed, 13))

        self.assertEqual(proposal.token_ids[:2], (8, 9))
        self.assertEqual(provider.metrics().council_feedback, 3)
        self.assertEqual(provider._state.feedback_count, 3)
        provider.close()

    def test_beam_verification_is_committed_only_for_the_reconciled_prefix(
        self,
    ) -> None:
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            proposal_width=3,
        )
        prompt = (1, 2)
        provider.begin_request(prompt)
        proposal = provider.propose_round(prompt, 3)
        provider._pending_planner = "beam"

        provider.observe_verification(3, 3)
        self.assertEqual(provider._beam_verified_tokens, 0)
        self.assertEqual(provider._beam_accepted_tokens, 0)
        with self.assertRaisesRegex(
            MarkovDraftError,
            "verification acceptance differs from reconciled prefix",
        ):
            provider.reconcile_prefix((*prompt, 3, *proposal.token_ids[:2]))

        self.assertEqual(provider._beam_verified_tokens, 0)
        self.assertEqual(provider._beam_accepted_tokens, 0)
        provider.discard_pending_proposal()
        self.assertIsNone(provider._pending_accepted_prefix_length)
        self.assertIsNone(provider._pending_verified_proposals)
        provider.close()

    def test_atlas_ranking_score_is_not_served_as_acceptance_probability(self) -> None:
        class Expert:
            @staticmethod
            def distribution(_context):
                return {"07": 0.55, "08": 0.45}

        class Atlas:
            max_branches = 8

            @staticmethod
            def token_options(_history, *, limit=None):
                return (AtlasTokenEvidence(8, 6, 1, 100, 0.01, 0.99),)

        provider = FingerprintRollingK4DraftProvider(vocab_size=32)
        provider.atlas = Atlas()
        provider._expert_models = lambda history: tuple(
            (
                Expert(),
                [provider._symbol(token) for token in history if 7 <= token <= 12],
            )
            for _ in provider._experts
        )

        result = provider._predict_beam((1,), 1)

        self.assertIsNotNone(result)
        assert result is not None
        tokens, _feedback, confidences, _disagreements = result
        self.assertEqual(tokens, (7,))
        self.assertLess(confidences[0], 0.60)
        provider.close()

    def test_external_mismatch_replays_the_same_beam_planner(self) -> None:
        class Expert:
            @staticmethod
            def distribution(context):
                if not context:
                    return {"07": 0.55, "08": 0.45}
                return {"09": 0.9, "10": 0.1}

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

        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            proposal_width=3,
        )
        provider.atlas = Atlas()
        provider._expert_models = lambda history: tuple(
            (
                Expert(),
                [provider._symbol(token) for token in history if 7 <= token <= 12],
            )
            for _ in provider._experts
        )
        prompt = (20, 21)
        provider.begin_request(prompt)
        with mock.patch.object(
            provider,
            "_predict_beam",
            wraps=provider._predict_beam,
        ) as beam:
            proposal = provider.propose_round(prompt, 2)
            mismatch = 7 if proposal.token_ids[0] != 7 else 8
            provider.reconcile_external_prefix((*prompt, 2, mismatch, 9))

        self.assertGreaterEqual(beam.call_count, 3)
        self.assertIsNone(provider._pending_planner)
        self.assertGreaterEqual(provider.metrics().teacher_forced_predictions, 2)
        provider.observe_final((*prompt, 2, mismatch, 9, 13))
        provider.close()

    def test_k16_beam_retains_only_compact_paths(self) -> None:
        class WideExpert:
            @staticmethod
            def distribution(context):
                offset = len(context) % 16
                weights = {
                    str(token): float(256 - ((token - offset) % 256))
                    for token in range(256)
                }
                total = sum(weights.values())
                return {token: value / total for token, value in weights.items()}

        class Atlas:
            max_branches = 8

            @staticmethod
            def token_options(_history, *, limit=None):
                return tuple(
                    AtlasTokenEvidence(
                        token,
                        1,
                        4,
                        16,
                        0.25,
                        0.10,
                    )
                    for token in range(8)
                )

        provider = FingerprintRollingK4DraftProvider(
            vocab_size=512,
            proposal_width=15,
        )
        provider.atlas = Atlas()
        provider._expert_models = lambda _history: tuple(
            (WideExpert(), []) for _ in provider._experts
        )

        result = provider._predict_beam((511,), 16)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(len(result[0]), 16)
        self.assertLessEqual(provider._last_beam_path_count, 8)
        self.assertNotIn(
            "feedback",
            markov_module._BeamStep.__dataclass_fields__,
        )
        provider.close()

    def test_target_feedback_can_disable_a_harmful_lookahead_override(self) -> None:
        class PlanningExpert:
            @staticmethod
            def distribution(context):
                if not context:
                    return {"07": 0.55, "08": 0.45}
                if context[-1] == "07":
                    return {"09": 0.51, "10": 0.49}
                return {"09": 0.99, "10": 0.01}

        provider = FingerprintRollingK4DraftProvider(vocab_size=32)
        provider._expert_models = lambda _history: tuple(
            (PlanningExpert(), []) for _ in provider._experts
        )
        planned, feedback, _confidence, _disagreement = provider._predict_council(
            (1,), 1
        )
        self.assertEqual(planned, (8,))
        for _ in range(16):
            provider._apply_council_feedback(
                feedback[0],
                7,
                0,
                8,
                7,
            )

        learned, _feedback, _confidence, _disagreement = provider._predict_council(
            (1,), 1
        )

        self.assertEqual(learned, (7,))
        self.assertEqual(provider._state.lookahead_observations[0], 16)
        self.assertEqual(provider._state.lookahead_hits[0], 0)
        self.assertEqual(provider._state.lookahead_greedy_hits[0], 16)
        restored = MarkovDraftState.from_bytes(provider._state.to_bytes())
        self.assertEqual(restored.lookahead_observations[0], 16)
        self.assertEqual(restored.lookahead_hits[0], 0)
        self.assertEqual(restored.lookahead_greedy_hits[0], 16)
        provider.close()

    def test_runtime_carry_records_greedy_winner_against_planned_token(self) -> None:
        class PlanningExpert:
            @staticmethod
            def distribution(context):
                if not context or context[-1] == "02":
                    return {"07": 0.55, "08": 0.45}
                if context[-1] == "07":
                    return {"09": 0.51, "10": 0.49}
                return {"09": 0.99, "10": 0.01}

        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            proposal_width=3,
        )
        provider._expert_models = lambda _history: tuple(
            (PlanningExpert(), []) for _ in provider._experts
        )
        prompt = (20, 1)
        provider.begin_request(prompt)
        proposal = provider.propose_round(prompt, 2)
        self.assertEqual(proposal.token_ids[0], 8)
        history = (*prompt, 2)
        provider.reconcile_prefix(history)

        provider.propose_round(history, 7)
        provider.discard_pending_proposal()
        provider.observe_final((*history, 7))

        self.assertEqual(provider._state.lookahead_observations[0], 1)
        self.assertEqual(provider._state.lookahead_hits[0], 0)
        self.assertEqual(provider._state.lookahead_greedy_hits[0], 1)
        provider.close()

    def test_below_threshold_prompt_starts_global_without_neighbor_leakage(
        self,
    ) -> None:
        provider = FingerprintRollingK4DraftProvider(vocab_size=64)
        prompt = tuple((1, 2, 3, 4) * 16)
        signature = provider._context_signature(prompt)
        shared = signature[: max(2, (len(signature) + 5) // 6)]
        extras = tuple(
            value for value in range(20_000, 21_000) if value not in set(signature)
        )[: len(signature) - len(shared)]
        neighbor_signature = tuple(sorted((*shared, *extras)))
        similarity = provider._dialect_similarity(signature, neighbor_signature)
        self.assertGreaterEqual(similarity, provider.MIN_DIALECT_NEIGHBOR_SIMILARITY)
        self.assertLess(similarity, provider.DIALECT_SIMILARITY_THRESHOLD)
        width = len(provider._experts)
        zeros = tuple((0,) * width for _ in range(16))
        old = MarkovDialectState(
            dialect_id="f" * 64,
            signature=neighbor_signature,
            visits=100,
            last_seen=0,
            rapidities=(10.0,) + (0.0,) * (width - 1),
            observations=(100,) * width,
            hits=(100,) + (0,) * (width - 1),
            horizon_observations=zeros,
            horizon_hits=zeros,
        )
        provider._state = replace(provider._state, dialects=(old,))
        global_weights = provider._weights()

        provider.begin_request(prompt)

        self.assertTrue(provider._active_dialect_is_new)
        self.assertEqual(provider._active_dialect_similarity, 0.0)
        self.assertEqual(provider._dialect_neighbors, ())
        self.assertEqual(provider._weights(), global_weights)
        self.assertEqual(provider.metrics().dialect_neighbor_count, 0)
        provider.close()

    def test_active_dialect_always_keeps_one_council_seat(self) -> None:
        provider = FingerprintRollingK4DraftProvider(vocab_size=64)
        prompt = tuple((1, 2, 3, 4) * 16)
        signature = provider._context_signature(prompt)
        width = len(provider._experts)
        zeros = tuple((0,) * width for _ in range(16))

        def dialect(digest: str, dialect_signature, *, visits: int):
            return MarkovDialectState(
                dialect_id=digest * 64,
                signature=dialect_signature,
                visits=visits,
                last_seen=100,
                rapidities=(0.0,) * width,
                observations=(0,) * width,
                hits=(0,) * width,
                horizon_observations=zeros,
                horizon_hits=zeros,
            )

        active = dialect("a", signature, visits=1)
        heavy = tuple(
            dialect(
                digest,
                tuple(sorted((*signature[:-1], 30_000 + index))),
                visits=100,
            )
            for index, digest in enumerate(("b", "c", "d", "e"))
        )
        provider._state = replace(
            provider._state,
            clock=100,
            dialects=tuple(sorted((active, *heavy), key=lambda row: row.dialect_id)),
        )

        provider.begin_request(prompt)

        neighbor_ids = {row[2].dialect_id for row in provider._dialect_neighbors}
        self.assertEqual(provider._active_dialect.dialect_id, active.dialect_id)
        self.assertEqual(len(neighbor_ids), 4)
        self.assertIn(active.dialect_id, neighbor_ids)
        self.assertEqual(len(neighbor_ids & {row.dialect_id for row in heavy}), 3)
        provider.close()

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

    def test_similar_context_reuses_dialect_and_distinct_context_creates_one(
        self,
    ) -> None:
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

    def test_single_dialect_normalization_does_not_square_similarity(self) -> None:
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

        self.assertEqual(exact, low)

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

    def test_phrase_agent_fills_a_seven_token_window_from_repeated_episode(
        self,
    ) -> None:
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
        self.assertEqual(tuple(row.window for row in proposal.horizons), (1, 2, 4, 8))
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

    def test_discarded_proposal_still_learns_the_confirmed_request(self) -> None:
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            max_history_tokens=128,
            proposal_width=7,
        )
        prompt = (20, 1)
        provider.begin_request(prompt)
        provider.propose_round(prompt, 2)

        provider.discard_pending_proposal()
        provider.observe_final((*prompt, 2, 7, 11))

        metrics = provider.metrics()
        self.assertEqual(metrics.episode_count, 1)
        self.assertEqual(metrics.updates, 1)
        self.assertEqual(
            provider.confirmed_episodes(),
            ((*prompt, 2, 7, 11),),
        )
        provider.close()

    def test_external_mismatch_teacher_forces_the_confirmed_suffix(
        self,
    ) -> None:
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            max_history_tokens=128,
            proposal_width=3,
        )
        prompt = (20, 1)
        provider.begin_request(prompt)
        proposal = provider.propose_round(prompt, 2)
        first_predictions = {
            prediction for _distribution, prediction in provider._pending_feedback[0]
        }
        external = next(
            token
            for token in range(provider.vocab_size)
            if token not in first_predictions and token != proposal.token_ids[0]
        )
        before = provider._state
        served = provider.metrics()

        provider.reconcile_external_prefix((*prompt, 2, external, 17))

        interim = provider.metrics()
        self.assertEqual(interim.external_reconcile_calls, 1)
        self.assertEqual(interim.external_feedback_tokens, 2)
        self.assertEqual(interim.teacher_forced_predictions, 2)
        self.assertEqual(interim.teacher_forced_feedback_tokens, 1)
        self.assertEqual(interim.teacher_forced_failures, 0)
        self.assertTrue(provider._carry_feedback_teacher_forced)
        self.assertEqual(provider._carry_feedback_position, 2)
        self.assertEqual(interim.predictions, served.predictions)
        self.assertEqual(interim.council_predictions, served.council_predictions)
        self.assertEqual(interim.last_raw_confidence, served.last_raw_confidence)
        self.assertEqual(
            interim.last_empirical_evidence,
            served.last_empirical_evidence,
        )
        self.assertEqual(interim.last_confidence, served.last_confidence)
        self.assertEqual(interim.last_disagreement, served.last_disagreement)
        self.assertEqual(interim.council_feedback, 0)
        self.assertEqual(interim.phrase_accepted_tokens, 0)
        provider.observe_final((*prompt, 2, external, 17))
        after = provider.metrics()
        self.assertEqual(after.council_feedback, 2)
        self.assertEqual(after.horizon_observations[:2], (1, 1))
        self.assertEqual(
            provider._state.expert_observations,
            tuple(value + 2 for value in before.expert_observations),
        )
        self.assertTrue(
            all(
                old <= new <= old + 1
                for old, new in zip(
                    before.expert_hits,
                    provider._state.expert_hits,
                    strict=True,
                )
            )
        )
        provider.close()

    def test_mismatch_teacher_carry_trains_next_recursive_position(self) -> None:
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            max_history_tokens=128,
            proposal_width=3,
        )
        prompt = (20, 1)
        provider.begin_request(prompt)
        proposal = provider.propose_round(prompt, 2)
        mismatch = (proposal.token_ids[0] + 1) % provider.vocab_size
        history = (*prompt, 2, mismatch)

        provider.reconcile_external_prefix(history)

        self.assertTrue(provider._carry_feedback_teacher_forced)
        self.assertEqual(provider._carry_feedback_position, 1)
        self.assertEqual(provider.metrics().teacher_forced_predictions, 1)
        provider.propose_round(history, 7)
        metrics = provider.metrics()
        self.assertEqual(metrics.teacher_forced_feedback_tokens, 1)
        self.assertEqual(metrics.external_feedback_tokens, 2)
        provider.discard_pending_proposal()
        provider.observe_final((*history, 7))
        metrics = provider.metrics()
        self.assertEqual(metrics.council_feedback, 2)
        self.assertEqual(metrics.horizon_observations[:2], (1, 1))
        provider.close()

    def test_teacher_carry_finalization_failure_rolls_back_for_retry(self) -> None:
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            max_history_tokens=128,
            proposal_width=3,
        )
        prompt = (20, 1)
        provider.begin_request(prompt)
        proposal = provider.propose_round(prompt, 2)
        mismatch = (proposal.token_ids[0] + 1) % provider.vocab_size
        history = (*prompt, 2, mismatch)
        provider.reconcile_external_prefix(history)
        before_metrics = provider.metrics()
        before_state = provider._state
        before_feedback = list(provider._episode_feedback)
        before_carry = provider._carry_feedback

        with (
            mock.patch.object(
                provider,
                "_persist",
                side_effect=OSError("disk unavailable"),
            ),
            self.assertRaisesRegex(OSError, "disk unavailable"),
        ):
            provider.observe_final((*history, 7))

        after_failure = provider.metrics()
        self.assertEqual(provider._state, before_state)
        self.assertEqual(provider._episode_feedback, before_feedback)
        self.assertIs(provider._carry_feedback, before_carry)
        self.assertEqual(provider._carry_feedback_position, 1)
        self.assertTrue(provider._carry_feedback_teacher_forced)
        self.assertEqual(
            after_failure.council_feedback,
            before_metrics.council_feedback,
        )
        self.assertEqual(
            after_failure.external_feedback_tokens,
            before_metrics.external_feedback_tokens,
        )
        self.assertEqual(
            after_failure.teacher_forced_feedback_tokens,
            before_metrics.teacher_forced_feedback_tokens,
        )

        provider.observe_final((*history, 7))
        after_retry = provider.metrics()
        self.assertEqual(after_retry.council_feedback, 2)
        self.assertEqual(after_retry.external_feedback_tokens, 2)
        self.assertEqual(after_retry.teacher_forced_feedback_tokens, 1)
        provider.close()

    def test_external_k1_reconciliation_carries_feedback_to_next_known_token(
        self,
    ) -> None:
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            max_history_tokens=128,
            proposal_width=3,
        )
        prompt = (20, 1)
        provider.begin_request(prompt)
        provider.propose_round(prompt, 2)

        history = (*prompt, 2)
        provider.reconcile_external_prefix(history)

        self.assertIsNotNone(provider._carry_feedback)
        self.assertEqual(provider.metrics().external_feedback_tokens, 0)
        provider.propose_round(history, 7)
        self.assertEqual(len(provider._episode_feedback), 1)
        provider.discard_pending_proposal()
        provider.observe_final((*history, 7))
        self.assertEqual(provider.metrics().council_feedback, 1)
        provider.close()

    def test_teacher_forced_suffix_failure_keeps_verified_request_usable(
        self,
    ) -> None:
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            max_history_tokens=128,
            proposal_width=3,
        )
        prompt = (20, 1)
        provider.begin_request(prompt)
        proposal = provider.propose_round(prompt, 2)
        mismatch = (proposal.token_ids[0] + 1) % provider.vocab_size
        with mock.patch.object(
            provider,
            "_predict_council",
            side_effect=MarkovDraftError("teacher unavailable"),
        ):
            provider.reconcile_external_prefix((*prompt, 2, mismatch, 17))

        metrics = provider.metrics()
        self.assertEqual(metrics.external_feedback_tokens, 1)
        self.assertEqual(metrics.teacher_forced_feedback_tokens, 0)
        self.assertEqual(metrics.teacher_forced_failures, 1)
        provider.observe_final((*prompt, 2, mismatch, 17))
        self.assertEqual(provider.metrics().council_feedback, 1)
        provider.close()

    def test_external_matching_prefix_scores_each_conditionally_valid_row(
        self,
    ) -> None:
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            max_history_tokens=128,
            proposal_width=3,
        )
        prompt = (20, 1)
        provider.begin_request(prompt)
        proposal = provider.propose_round(prompt, 2)
        matched = proposal.token_ids[:2]

        provider.reconcile_external_prefix((*prompt, 2, *matched))

        self.assertEqual(provider.metrics().external_feedback_tokens, 2)
        self.assertIsNotNone(provider._carry_feedback)
        provider.observe_final((*prompt, 2, *matched, 9))
        self.assertEqual(provider.metrics().council_feedback, 3)
        self.assertEqual(provider.metrics().horizon_observations[:3], (1, 1, 1))
        self.assertEqual(provider.metrics().phrase_accepted_tokens, 0)
        provider.close()

    def test_k16_external_prefix_trains_position_fifteen_carry(self) -> None:
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            max_history_tokens=128,
            proposal_width=15,
        )
        prompt = (20, 1)
        provider.begin_request(prompt)
        proposal = provider.propose_round(prompt, 2)
        committed = (*prompt, 2, *proposal.token_ids)

        provider.reconcile_external_prefix(committed)
        self.assertEqual(provider._carry_feedback_position, 15)
        provider.observe_final((*committed, 9))

        metrics = provider.metrics()
        self.assertEqual(metrics.external_feedback_tokens, 15)
        self.assertEqual(metrics.council_feedback, 16)
        self.assertEqual(metrics.horizon_observations, (1,) * 16)
        provider.close()

    def test_external_reconciliation_rejects_missing_or_noncontiguous_state(
        self,
    ) -> None:
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            max_history_tokens=128,
            proposal_width=3,
        )
        prompt = (20, 1)
        provider.begin_request(prompt)
        with self.assertRaisesRegex(MarkovDraftError, "pending proposal"):
            provider.reconcile_external_prefix((*prompt, 2))
        provider.propose_round(prompt, 2)
        with self.assertRaisesRegex(MarkovDraftError, "known base"):
            provider.reconcile_external_prefix((20, 1, 3))
        with self.assertRaisesRegex(MarkovDraftError, "proposal width"):
            provider.reconcile_external_prefix((*prompt, 2, 3, 4, 5, 6))
        provider.discard_pending_proposal()
        provider.observe_final((*prompt, 2))
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

    def test_dialect_phrase_agent_activates_before_global_support_threshold(
        self,
    ) -> None:
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

    def test_exact_dialect_phrase_is_not_damped_twice(self) -> None:
        prompt = tuple((1, 2) * 20)
        episode = (*prompt, 3, 4, 5, 6)
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=32,
            max_history_tokens=256,
        )
        dialect_id = "a" * 64
        signature = provider._context_signature(prompt)
        extra_signature = tuple(
            value for value in range(10_000, 11_000) if value not in set(signature)
        )[: max(0, 32 - len(signature))]
        dialect_signature = tuple(sorted((*signature, *extra_signature)))
        expert_count = len(provider._state.expert_names)
        provider._state = replace(
            provider._state,
            token_ids=episode * 2,
            episode_lengths=(len(episode),) * 2,
            episode_dialects=(dialect_id,) * 2,
            dialects=(
                MarkovDialectState(
                    dialect_id=dialect_id,
                    signature=dialect_signature,
                    visits=2,
                    last_seen=1,
                    rapidities=(0.0,) * expert_count,
                    observations=(0,) * expert_count,
                    hits=(0,) * expert_count,
                ),
            ),
            clock=1,
        )
        provider.begin_request(prompt)
        self.assertGreaterEqual(provider.metrics().active_dialect_similarity, 0.20)
        self.assertLess(provider.metrics().active_dialect_similarity, 1.0)

        proposal = provider.propose_round(prompt, 3)
        policy = proposal.select_window(
            request_window_ceiling=4,
            remaining_tokens=4,
            window_work_costs={1: 1.0, 2: 1.6, 4: 2.8},
        )

        self.assertEqual(proposal.token_ids, (4, 5, 6))
        self.assertEqual(proposal.phrase_source, "dialect")
        self.assertEqual(proposal.phrase_support, 2)
        self.assertEqual(proposal.phrase_confidence, 1.0)
        self.assertEqual(policy.chosen_window, 2)
        provider.discard_pending_proposal()
        provider.close()

    def test_composition_copies_unseen_slots_into_a_literal_program(self) -> None:
        first_prompt = (1, 20, 2, 30)
        second_prompt = (1, 21, 2, 31)
        first_output = (50, 20, 9, 30, 51)
        second_output = (50, 21, 9, 31, 51)
        episodes = (
            (*first_prompt, *first_output),
            (*second_prompt, *second_output),
        )
        provider = FingerprintRollingK4DraftProvider(
            vocab_size=64,
            max_history_tokens=128,
            proposal_width=7,
        )
        provider._state = replace(
            provider._state,
            token_ids=tuple(token for episode in episodes for token in episode),
            episode_lengths=tuple(len(episode) for episode in episodes),
            episode_dialects=(None, None),
            episode_prompt_lengths=(len(first_prompt), len(second_prompt)),
        )
        prompt = (1, 22, 2, 32)
        expected_output = (50, 22, 9, 32, 51)
        provider.begin_request(prompt)

        proposal = provider.propose_round(prompt, expected_output[0])

        self.assertEqual(proposal.token_ids[:4], expected_output[1:])
        self.assertEqual(proposal.phrase_source, "global")
        self.assertEqual(proposal.phrase_support, 2)
        self.assertEqual(proposal.phrase_confidence, 1.0)
        metrics = provider.metrics()
        self.assertGreaterEqual(metrics.composition_programs, 1)
        self.assertEqual(metrics.composition_option_calls, 1)
        self.assertEqual(metrics.composition_draft_tokens, 4)
        self.assertEqual(metrics.last_composition_support, 2)
        self.assertEqual(metrics.last_composition_copy_tokens, 2)

        provider.reconcile_prefix((*prompt, *expected_output[:3]))
        self.assertEqual(provider.metrics().composition_accepted_tokens, 2)
        provider.observe_final((*prompt, *expected_output))
        self.assertEqual(provider._state.episode_prompt_lengths[-1], len(prompt))
        self.assertGreaterEqual(provider.metrics().composition_programs, 1)
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
