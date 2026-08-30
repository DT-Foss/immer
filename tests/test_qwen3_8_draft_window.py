from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import hashlib
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zlib

from immer.cli import main
from immer.contracts import ExecutionStatus, Request
from immer.runtimes.qwen3_8.draft_window import (
    DRAFT_WINDOW_ACTIONS,
    DRAFT_WINDOW_FEEDBACK_SCHEMA,
    DRAFT_WINDOW_STATE_SCHEMA,
    LEGACY_DRAFT_WINDOW_STATE_SCHEMA,
    V1_DRAFT_WINDOW_STATE_SCHEMA,
    DraftWindowController,
    DraftWindowError,
    DraftWindowFeedback,
    DraftWindowNestedHorizon,
    DraftWindowState,
    contextual_bottom_k_signature,
)
from immer.runtimes.qwen3_8.draft_protocol import RollingDraftProposal

from test_qwen3_8_adapter import _Model, _Runtime, _Tokenizer, _chat


def _feedback(
    selection,
    *,
    accepted: int = 4,
    emitted: int = 8,
    outcome: str = "ok",
    receipt: str = "a" * 64,
) -> DraftWindowFeedback:
    return DraftWindowFeedback(
        proposed_window=selection.proposed_window,
        accepted_draft_tokens=accepted,
        emitted_tokens=emitted,
        target_source_body_bytes=2 * 1024**3,
        draft_source_body_bytes=128 * 1024**2,
        aux_source_body_bytes=32 * 1024**2,
        target_forwards=3,
        seconds=1.25,
        outcome=outcome,
        target_receipt_sha256=receipt,
    )


def _rolling_result(
    window: int = 8,
    *,
    accepted: int = 4,
    token_ids: tuple[int, ...] = (7, 8, 9, 10),
    executed_window: int | None = None,
    round_policy=None,
):
    active_window = window if executed_window is None else executed_window
    round_evidence = SimpleNamespace(
        accepted_prefix_length=accepted,
        provider_proposed_token_ids=tuple(range(window - 1)),
        proposed_token_ids=tuple(range(active_window - 1)),
        round_index=0,
        round_policy=round_policy,
        window_size=active_window,
    )
    evidence = SimpleNamespace(
        accepted_draft_tokens=accepted,
        source_body_bytes=100,
        linear_calls=10,
        seconds=1.0,
        state_bytes=456,
        stopped_on_eos=False,
        prompt_token_ids=(11, 12),
        generated_token_ids=token_ids,
        forward_passes=3,
        rounds=(round_evidence,),
        schema="immer.qwen3.8-rolling-speculative-generation/v2",
        final_state_committed=False,
        window_size=window,
        adaptive_windows=round_policy is not None,
        used_window_sizes=(active_window,),
    )
    return SimpleNamespace(token_ids=token_ids, evidence=evidence)


class DraftWindowControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state_path = self.root / "window.bin"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_cold_choice_is_k8_and_ceiling_and_budget_are_hard_bounds(self) -> None:
        controller = DraftWindowController(self.state_path)

        cold = controller.choose((1, 2, 3), max_window=16, max_new_tokens=64)
        ceiling = controller.choose((1, 2, 3), max_window=6, max_new_tokens=64)
        budget = controller.choose((1, 2, 3), max_window=16, max_new_tokens=6)
        unavailable = controller.choose((1, 2, 3), max_window=16, max_new_tokens=3)

        self.assertEqual(cold.proposed_window, 8)
        self.assertEqual(cold.eligible_windows, DRAFT_WINDOW_ACTIONS)
        self.assertEqual(ceiling.proposed_window, 4)
        self.assertEqual(budget.proposed_window, 4)
        self.assertIsNone(unavailable)

    def test_selection_is_read_only_until_a_target_receipt_is_settled(self) -> None:
        controller = DraftWindowController(self.state_path)
        before = controller.metrics()
        selection = controller.choose((9, 8, 7, 6), max_window=16, max_new_tokens=16)

        self.assertFalse(self.state_path.exists())
        self.assertEqual(before.updates, 0)
        self.assertEqual(controller.metrics().updates, 0)

        settled = controller.settle(selection, _feedback(selection))
        self.assertTrue(self.state_path.is_file())
        self.assertEqual(settled.updates, 1)

    def test_policy_bootstraps_each_allowed_arm_then_samples_fixed_share(self) -> None:
        controller = DraftWindowController(self.state_path)
        first = controller.choose((1, 2, 3), max_window=16, max_new_tokens=16)
        controller.settle(first, _feedback(first, receipt="1" * 64))
        second = controller.choose((1, 2, 3), max_window=16, max_new_tokens=16)
        controller.settle(second, _feedback(second, receipt="2" * 64))
        third = controller.choose((1, 2, 3), max_window=16, max_new_tokens=16)
        controller.settle(third, _feedback(third, receipt="3" * 64))
        with patch(
            "immer.runtimes.qwen3_8.draft_window.secrets.randbelow",
            return_value=0,
        ):
            sampled = controller.choose(
                (1, 2, 3), max_window=16, max_new_tokens=16
            )

        self.assertEqual(
            (first.proposed_window, second.proposed_window, third.proposed_window),
            (8, 4, 16),
        )
        self.assertEqual(
            (first.selection_mode, second.selection_mode, third.selection_mode),
            ("cold", "bootstrap", "bootstrap"),
        )
        self.assertEqual(sampled.selection_mode, "fixed-share")
        self.assertEqual(sampled.proposed_window, 4)
        self.assertEqual(
            sampled.selection_probability,
            dict(sampled.policy_weights)[sampled.proposed_window],
        )

    def test_default_k8_ceiling_can_never_bootstrap_or_sample_k16(self) -> None:
        controller = DraftWindowController(self.state_path)
        selected = []
        with patch(
            "immer.runtimes.qwen3_8.draft_window.secrets.randbelow",
            return_value=(1 << 53) - 1,
        ):
            for index in range(8):
                selection = controller.choose(
                    (1, 2, 3), max_window=8, max_new_tokens=16
                )
                selected.append(selection.proposed_window)
                controller.settle(
                    selection,
                    _feedback(selection, receipt=f"{index + 1:064x}"),
                )

        self.assertEqual(selected[:2], [8, 4])
        self.assertNotIn(16, selected)

    def test_reward_favors_confirmed_tokens_and_penalizes_failures(self) -> None:
        controller = DraftWindowController(self.state_path)
        selection = controller.choose((1, 2), max_window=16, max_new_tokens=16)
        useful = _feedback(selection, accepted=6, emitted=8)
        zero = _feedback(selection, accepted=0, emitted=8)
        timeout = _feedback(
            selection,
            accepted=0,
            emitted=0,
            outcome="timeout",
        )

        self.assertGreater(useful.reward, 0.0)
        self.assertLess(zero.reward, -6.0)
        self.assertLess(timeout.reward, zero.reward)
        self.assertEqual(
            useful.total_source_body_bytes,
            useful.target_source_body_bytes
            + useful.draft_source_body_bytes
            + useful.aux_source_body_bytes,
        )

    def test_feedback_v3_uses_exact_joint_runtime_reward_and_page_context(
        self,
    ) -> None:
        controller = DraftWindowController(self.state_path)
        selection = controller.choose((1, 2), max_window=16, max_new_tokens=16)
        feedback = replace(
            _feedback(
                selection,
                accepted=0,
                emitted=0,
                outcome="timeout",
            ),
            o1_priority=12.5,
            page_actions=96,
            page_actions_saved=64,
            runtime_reward=3.125,
        )

        self.assertEqual(
            DRAFT_WINDOW_FEEDBACK_SCHEMA,
            "immer.qwen3.8-draft-window-feedback/v3",
        )
        self.assertEqual(feedback.reward, 3.125)
        record = feedback.to_dict()
        self.assertEqual(
            {
                key: record[key]
                for key in (
                    "o1_priority",
                    "page_actions",
                    "page_actions_saved",
                    "reward",
                    "runtime_reward",
                    "schema",
                )
            },
            {
                "o1_priority": 12.5,
                "page_actions": 96,
                "page_actions_saved": 64,
                "reward": 3.125,
                "runtime_reward": 3.125,
                "schema": DRAFT_WINDOW_FEEDBACK_SCHEMA,
            },
        )
        metrics = controller.settle(selection, feedback)
        self.assertEqual(metrics.updates, 1)
        state = DraftWindowState.from_bytes(self.state_path.read_bytes())
        self.assertAlmostEqual(
            state.agents[1].reward_sum,
            3.125,
        )

        for changes in (
            {"page_actions": -1},
            {"page_actions_saved": True},
            {"o1_priority": -0.1},
            {"o1_priority": float("nan")},
            {"runtime_reward": float("nan")},
            {"runtime_reward": -16.01},
            {"runtime_reward": 16.01},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    replace(feedback, **changes)

    def test_wider_wave_teaches_exact_shorter_prefix_reliability(self) -> None:
        controller = DraftWindowController(self.state_path)
        selection = controller.choose((1, 2, 3), max_window=8, max_new_tokens=16)
        nested = DraftWindowNestedHorizon(
            candidate_window=4,
            observed_window=8,
            wave_count=2,
            accepted_draft_tokens=5,
            proposed_draft_tokens=6,
        )

        metrics = controller.settle(
            selection,
            replace(_feedback(selection), nested_horizons=(nested,)),
        )

        short = metrics.agents[0]
        self.assertEqual(short.observations, 0)
        self.assertEqual(short.nested_observations, 1)
        self.assertEqual(short.nested_accepted_draft_tokens, 5)
        self.assertEqual(short.nested_proposed_draft_tokens, 6)
        self.assertAlmostEqual(short.to_dict()["nested_reliability"], 5 / 6)
        self.assertGreater(short.nested_acceptance_score_sum, 0.0)
        with patch(
            "immer.runtimes.qwen3_8.draft_window.secrets.randbelow",
            return_value=0,
        ):
            next_selection = controller.choose(
                (1, 2, 3), max_window=8, max_new_tokens=16
            )
        self.assertEqual(next_selection.selection_mode, "fixed-share")

    def test_nested_acceptance_never_rewrites_executed_work_reward(self) -> None:
        plain_path = self.root / "plain-window.bin"
        nested_path = self.root / "nested-window.bin"
        plain = DraftWindowController(plain_path)
        informed = DraftWindowController(nested_path)
        plain_selection = plain.choose(
            (1, 2, 3), max_window=8, max_new_tokens=16
        )
        informed_selection = informed.choose(
            (1, 2, 3), max_window=8, max_new_tokens=16
        )
        nested = DraftWindowNestedHorizon(
            candidate_window=4,
            observed_window=8,
            wave_count=1,
            accepted_draft_tokens=3,
            proposed_draft_tokens=3,
        )

        plain_metrics = plain.settle(
            plain_selection,
            _feedback(plain_selection),
        )
        informed_metrics = informed.settle(
            informed_selection,
            replace(
                _feedback(informed_selection),
                nested_horizons=(nested,),
            ),
        )
        plain_state = DraftWindowState.from_bytes(plain_path.read_bytes())
        informed_state = DraftWindowState.from_bytes(nested_path.read_bytes())

        self.assertEqual(plain_state.rapidities, informed_state.rapidities)
        self.assertEqual(
            plain_state.dialects[0].rapidities,
            informed_state.dialects[0].rapidities,
        )
        self.assertGreater(
            dict(informed_metrics.policy_weights)[4],
            dict(plain_metrics.policy_weights)[4],
        )

    def test_council_signal_formula_moves_horizon_with_predictive_strength(self) -> None:
        weak = DraftWindowState(
            signal_observations=1,
            council_confidence_ema=0.20,
            council_disagreement_ema=0.50,
        )
        strong = DraftWindowState(
            signal_observations=1,
            council_confidence_ema=0.99,
            council_disagreement_ema=0.0,
            phrase_confidence_ema=1.0,
            phrase_support_ema=3.0,
            phrase_width_ema=7.0,
        )

        weak_signal = DraftWindowController._signal_rapidities(weak)
        strong_signal = DraftWindowController._signal_rapidities(strong)

        self.assertGreater(weak_signal[0], weak_signal[2])
        self.assertGreater(strong_signal[2], strong_signal[0])

    def test_target_receipt_persists_council_and_phrase_signal_emas(self) -> None:
        controller = DraftWindowController(self.state_path)
        selection = controller.choose((1, 2, 3), max_window=8, max_new_tokens=16)
        feedback = replace(
            _feedback(selection),
            council_confidence=0.90,
            council_disagreement=0.10,
            effective_experts=2.5,
            phrase_confidence=0.80,
            phrase_support=4,
            phrase_width=7,
        )

        metrics = controller.settle(selection, feedback)

        self.assertEqual(metrics.signal_observations, 1)
        self.assertEqual(metrics.council_confidence_ema, 0.90)
        self.assertEqual(metrics.council_disagreement_ema, 0.10)
        self.assertEqual(metrics.phrase_confidence_ema, 0.80)
        self.assertEqual(metrics.phrase_support_ema, 4.0)
        self.assertEqual(metrics.phrase_width_ema, 7.0)

    def test_settlement_persists_all_cumulative_agent_work_metrics(self) -> None:
        controller = DraftWindowController(self.state_path)
        selection = controller.choose((3, 1, 4), max_window=16, max_new_tokens=16)
        feedback = _feedback(selection, accepted=5, emitted=7)

        metrics = controller.settle(selection, feedback)
        agent = metrics.agents[DRAFT_WINDOW_ACTIONS.index(8)]
        restored = DraftWindowState.from_bytes(self.state_path.read_bytes())

        self.assertEqual(agent.observations, 1)
        self.assertEqual(agent.accepted_draft_tokens, 5)
        self.assertEqual(agent.emitted_tokens, 7)
        self.assertEqual(agent.target_source_body_bytes, 2 * 1024**3)
        self.assertEqual(agent.draft_source_body_bytes, 128 * 1024**2)
        self.assertEqual(agent.aux_source_body_bytes, 32 * 1024**2)
        self.assertEqual(agent.target_forwards, 3)
        self.assertEqual(agent.seconds, 1.25)
        self.assertEqual(restored, controller._state)
        self.assertLess(len(self.state_path.read_bytes()), 1024 * 1024)

    def test_zero_acceptance_moves_the_policy_away_from_k8(self) -> None:
        controller = DraftWindowController(self.state_path)
        first = controller.choose((5, 5, 5), max_window=16, max_new_tokens=16)
        controller.settle(first, _feedback(first, accepted=0, emitted=8))

        second = controller.choose((5, 5, 5), max_window=16, max_new_tokens=16)

        self.assertEqual(first.proposed_window, 8)
        self.assertEqual(second.proposed_window, 4)
        self.assertGreater(
            dict(second.policy_weights)[4], dict(second.policy_weights)[8]
        )

    def test_confirmed_surprise_cusum_shrinks_policy_on_regime_change(self) -> None:
        controller = DraftWindowController(self.state_path)
        with patch(
            "immer.runtimes.qwen3_8.draft_window.secrets.randbelow",
            return_value=0,
        ):
            for index in range(1, 9):
                selection = controller.choose(
                    (5, 5, 5), max_window=16, max_new_tokens=16
                )
                controller.settle(
                    selection,
                    _feedback(
                        selection,
                        accepted=8,
                        receipt=f"{index:064x}",
                    ),
                )
            for index in range(9, 12):
                selection = controller.choose(
                    (5, 5, 5), max_window=16, max_new_tokens=16
                )
                metrics = controller.settle(
                    selection,
                    _feedback(
                        selection,
                        accepted=0,
                        emitted=0,
                        outcome="timeout",
                        receipt=f"{index:064x}",
                    ),
                )

        self.assertEqual(metrics.regime_generation, 1)
        self.assertEqual(metrics.surprise_cusum, 0.0)

    def test_settlement_is_idempotent_for_the_same_selection_and_receipt(self) -> None:
        controller = DraftWindowController(self.state_path)
        selection = controller.choose((1, 3, 3, 7), max_window=16, max_new_tokens=16)
        feedback = _feedback(selection)

        first = controller.settle(selection, feedback)
        second = controller.settle(selection, feedback)

        self.assertEqual(first.state_sha256, second.state_sha256)
        self.assertEqual(second.updates, 1)

    def test_failed_atomic_write_rolls_back_the_entire_settlement(self) -> None:
        controller = DraftWindowController(self.state_path)
        selection = controller.choose((1, 3, 5), max_window=16, max_new_tokens=16)
        with (
            patch(
                "immer.runtimes.qwen3_8.draft_window._persist_state",
                side_effect=DraftWindowError("disk failed"),
            ),
            self.assertRaises(DraftWindowError),
        ):
            controller.settle(selection, _feedback(selection))

        self.assertFalse(self.state_path.exists())
        self.assertEqual(controller.metrics().updates, 0)

    def test_context_sketch_is_bounded_and_raw_tokens_are_not_persisted(self) -> None:
        prompt = tuple(range(700))
        signature = contextual_bottom_k_signature(prompt)
        controller = DraftWindowController(self.state_path)
        selection = controller.choose(prompt, max_window=16, max_new_tokens=16)
        controller.settle(selection, _feedback(selection))

        state = DraftWindowState.from_bytes(self.state_path.read_bytes())
        self.assertEqual(len(signature), 32)
        self.assertEqual(state.dialects[0].signature, signature)
        self.assertNotIn(
            json.dumps(list(prompt)).encode("utf-8"),
            self.state_path.read_bytes(),
        )

    def test_ricci_retention_bounds_contextual_dialects(self) -> None:
        controller = DraftWindowController(self.state_path, max_dialects=2)
        for index in range(3):
            prompt = tuple(range(index * 100 + 1, index * 100 + 20))
            selection = controller.choose(
                prompt,
                max_window=16,
                max_new_tokens=16,
            )
            controller.settle(
                selection,
                _feedback(selection, receipt=f"{index + 1:064x}"),
            )

        metrics = controller.metrics()
        self.assertEqual(metrics.dialect_count, 2)
        self.assertEqual(metrics.dialect_evictions, 1)

    def test_concurrent_terminal_settlements_remain_atomic(self) -> None:
        controller = DraftWindowController(self.state_path)
        selections = [
            controller.choose((index + 1, 2, 3), max_window=16, max_new_tokens=16)
            for index in range(8)
        ]

        def settle(index: int) -> None:
            controller.settle(
                selections[index],
                _feedback(selections[index], receipt=f"{index + 1:064x}"),
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(settle, range(8)))

        restored = DraftWindowState.from_bytes(self.state_path.read_bytes())
        self.assertEqual(restored.updates, 8)
        self.assertEqual(sum(agent.observations for agent in restored.agents), 8)

    def test_corrupt_and_symlink_states_fail_closed(self) -> None:
        self.state_path.write_bytes(b"not-a-controller-state")
        with self.assertRaises(DraftWindowError):
            DraftWindowController(self.state_path)

        self.state_path.unlink()
        target = self.root / "target.bin"
        target.write_bytes(DraftWindowState().to_bytes())
        self.state_path.symlink_to(target)
        with self.assertRaises(DraftWindowError):
            DraftWindowController(self.state_path)

    def test_v0_state_migrates_on_the_first_terminal_receipt(self) -> None:
        empty = DraftWindowState()
        legacy = {
            "agents": [agent.to_record() for agent in empty.agents],
            "clock": 0,
            "rapidities": [value.hex() for value in empty.rapidities],
            "regime_generation": 0,
            "schema": LEGACY_DRAFT_WINDOW_STATE_SCHEMA,
            "surprise_cusum": float(0.0).hex(),
            "surprise_deviation": float(1.0).hex(),
            "surprise_mean": float(0.0).hex(),
            "updates": 0,
        }
        raw = json.dumps(
            legacy,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.state_path.write_bytes(b"IMDW\x00" + zlib.compress(raw, level=9))
        controller = DraftWindowController(self.state_path)
        selection = controller.choose((2, 4, 6), max_window=16, max_new_tokens=16)
        controller.settle(selection, _feedback(selection))

        restored = DraftWindowState.from_bytes(self.state_path.read_bytes())
        self.assertEqual(restored.updates, 1)
        self.assertEqual(restored.to_record()["schema"], DRAFT_WINDOW_STATE_SCHEMA)
        self.assertTrue(self.state_path.read_bytes().startswith(b"IMDW\x02"))

    def test_v1_state_binds_to_one_runtime_identity_and_migrates(self) -> None:
        body = DraftWindowState().to_record()
        body.pop("policy_identity_sha256")
        body["schema"] = V1_DRAFT_WINDOW_STATE_SCHEMA
        encoded_body = json.dumps(
            body,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        envelope = json.dumps(
            {
                "body": body,
                "body_sha256": hashlib.sha256(encoded_body).hexdigest(),
            },
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self.state_path.write_bytes(b"IMDW\x01" + zlib.compress(envelope, level=9))
        controller = DraftWindowController(self.state_path)

        metrics = controller.bind_policy_identity("a" * 64)

        self.assertEqual(metrics.policy_identity_sha256, "a" * 64)
        self.assertTrue(self.state_path.read_bytes().startswith(b"IMDW\x02"))
        rebound = DraftWindowController(self.state_path)
        with self.assertRaisesRegex(DraftWindowError, "different runtime identity"):
            rebound.bind_policy_identity("b" * 64)

    def test_feedback_requires_a_target_receipt_and_matching_window(self) -> None:
        controller = DraftWindowController(self.state_path)
        selection = controller.choose((1, 2), max_window=16, max_new_tokens=16)
        with self.assertRaises(ValueError):
            replace(_feedback(selection), target_receipt_sha256="bad")
        with self.assertRaises(DraftWindowError):
            controller.settle(
                selection,
                replace(_feedback(selection), proposed_window=4),
            )
        self.assertEqual(controller.metrics().updates, 0)


class DraftWindowAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state_path = self.root / "window.bin"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _adaptive_chat(self, runtime: _Runtime, **overrides):
        runtime.model.max_seq_len = 32
        options = {
            "draft_mode": "markov",
            "draft_window": 16,
            "draft_window_state_path": self.state_path,
            "max_new_tokens": 16,
            "max_context_tokens": 32,
        }
        options.update(overrides)
        return _chat(runtime, **options)

    def test_adapter_uses_k8_then_settles_public_receipt_metrics(self) -> None:
        runtime = _Runtime()

        class Q4Metrics:
            def __init__(self) -> None:
                self.calls = 0

            def metrics(self):
                selected = 10 if self.calls == 0 else 30
                self.calls += 1
                return {"page_mlp_selected_pages": selected}

        class PageRouter:
            page_count = 272
            route_width = 192

            def __init__(self) -> None:
                self.events = []
                self.metric_calls = 0

            def begin_runtime_reward(self) -> None:
                self.events.append(("begin",))

            @staticmethod
            def flush() -> None:
                pass

            def metrics(self):
                saved = 5 if self.metric_calls == 0 else 17
                self.metric_calls += 1
                return {
                    "adaptive_width_pages_saved": saved,
                    "runtime_reward_updates": 0,
                }

            def settle_runtime_reward(self, receipt: str, reward: float):
                self.events.append(("settle", receipt, reward))
                return {"runtime_reward_updates": 1}

        q4 = Q4Metrics()
        router = PageRouter()
        runtime.model.pager.q4_bank = q4
        runtime.mlp_page_router = router
        chat = self._adaptive_chat(runtime)
        generated = _rolling_result()
        decoder = SimpleNamespace(generate_rolling=lambda *args, **kwargs: generated)

        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
            return_value=decoder,
        ) as constructor:
            result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        constructor.assert_called_once()
        self.assertEqual(constructor.call_args.kwargs["window_size"], 8)
        receipt = result.evidence["draft_window"]
        self.assertTrue(receipt["settled"])
        self.assertEqual(receipt["selection"]["proposed_window"], 8)
        self.assertEqual(receipt["feedback"]["accepted_draft_tokens"], 4)
        self.assertEqual(receipt["feedback"]["emitted_tokens"], 4)
        self.assertEqual(receipt["feedback"]["target_source_body_bytes"], 100)
        self.assertEqual(receipt["feedback"]["draft_source_body_bytes"], 0)
        self.assertEqual(receipt["feedback"]["aux_source_body_bytes"], 0)
        self.assertEqual(receipt["feedback"]["target_forwards"], 3)
        self.assertEqual(receipt["feedback"]["seconds"], 1.0)
        self.assertEqual(receipt["feedback"]["page_actions"], 20)
        self.assertEqual(receipt["feedback"]["page_actions_saved"], 12)
        self.assertEqual(receipt["feedback"]["o1_priority"], 0.0)
        self.assertEqual(
            receipt["feedback"]["runtime_reward"],
            result.evidence["runtime_reward"]["reward"],
        )
        expected_reward = (
            3.0 * (12 / (20 + 12))
            + 2.0 * (4 / 4)
            - math.log1p(3) / 2.0
        )
        self.assertAlmostEqual(
            receipt["feedback"]["runtime_reward"],
            expected_reward,
        )
        self.assertEqual(
            receipt["feedback"]["reward"],
            result.evidence["runtime_reward"]["reward"],
        )
        self.assertEqual(
            receipt["feedback"]["schema"],
            DRAFT_WINDOW_FEEDBACK_SCHEMA,
        )
        self.assertEqual(result.evidence["runtime_reward"]["accepted_draft_tokens"], 4)
        self.assertEqual(result.evidence["runtime_reward"]["page_actions"], 20)
        self.assertEqual(result.evidence["runtime_reward"]["page_actions_saved"], 12)
        self.assertEqual(result.evidence["runtime_reward"]["router_updates"], 1)
        self.assertEqual(router.events[0], ("begin",))
        self.assertEqual(router.events[1][0], "settle")
        self.assertEqual(
            router.events[1][1],
            result.evidence["runtime_reward"]["receipt_sha256"],
        )
        self.assertEqual(receipt["metrics"]["agents"]["8"]["observations"], 1)
        self.assertEqual(
            receipt["metrics"]["agents"]["4"]["nested_observations"],
            1,
        )
        nested = receipt["feedback"]["nested_horizons"]
        self.assertEqual(nested[0]["candidate_window"], 4)
        self.assertEqual(nested[0]["accepted_draft_tokens"], 3)
        self.assertEqual(nested[0]["proposed_draft_tokens"], 3)
        self.assertEqual(result.evidence["draft"]["window_size"], 8)
        chat.close()

    def test_adapter_records_round_local_k1_abstention_under_k8_ceiling(self) -> None:
        proposal = RollingDraftProposal.build(
            tuple(range(7)),
            (0.10,) * 7,
            (0.0,) * 7,
            request_window_ceiling=8,
            provider_abi="test-provider/v1",
        )
        policy = proposal.select_window(
            request_window_ceiling=8,
            remaining_tokens=8,
        )
        runtime = _Runtime()
        runtime.model.pager = SimpleNamespace(q4_bank=object())
        chat = self._adaptive_chat(runtime)
        generated = _rolling_result(
            window=8,
            executed_window=1,
            accepted=0,
            round_policy=policy,
        )
        decoder = SimpleNamespace(generate_rolling=lambda *args, **kwargs: generated)

        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
            return_value=decoder,
        ) as constructor:
            result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertTrue(constructor.call_args.kwargs["adaptive_round_windows"])
        self.assertEqual(
            constructor.call_args.kwargs["round_window_work_costs"],
            {1: 1.0, 2: 1.6, 4: 2.8, 8: 5.2, 16: 10.0},
        )
        draft = result.evidence["draft"]
        self.assertTrue(draft["adaptive_windows"])
        self.assertEqual(draft["window_size"], 8)
        self.assertEqual(draft["used_window_sizes"], [1])
        self.assertEqual(
            draft["round_window_policies"][0]["round_policy"]["chosen_window"],
            1,
        )
        self.assertEqual(
            result.evidence["draft_window"]["feedback"]["nested_horizons"],
            [],
        )
        chat.close()

    def test_markov_provider_abi_change_cannot_reuse_window_policy(self) -> None:
        decoder = SimpleNamespace(
            generate_rolling=lambda *args, **kwargs: _rolling_result()
        )
        first = self._adaptive_chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
            return_value=decoder,
        ):
            self.assertTrue(first.handle(Request("chat", "hello")).ok)
        first.close()

        second = self._adaptive_chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.MARKOV_DRAFT_PROVIDER_ABI",
            "immer.qwen3.8-markov-draft-provider/v999",
        ):
            result = second.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertIn("different runtime identity", result.reason)
        second.close()

    def test_post_generation_decode_error_teaches_terminal_error_reward(self) -> None:
        class BrokenDecode(_Tokenizer):
            def decode(self, token_ids):
                raise RuntimeError("decode failed")

        runtime = _Runtime(tokenizer=BrokenDecode())
        chat = self._adaptive_chat(runtime)
        decoder = SimpleNamespace(
            generate_rolling=lambda *args, **kwargs: _rolling_result()
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
            return_value=decoder,
        ):
            result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        receipt = result.evidence["draft_window"]
        self.assertTrue(receipt["settled"])
        self.assertEqual(receipt["feedback"]["outcome"], "error")
        self.assertIsNone(receipt["feedback"]["runtime_reward"])
        self.assertLess(receipt["feedback"]["reward"], -9.0)
        self.assertEqual(receipt["post_generation_outcome"], "error")
        restored = DraftWindowState.from_bytes(self.state_path.read_bytes())
        self.assertEqual(restored.updates, 1)
        self.assertEqual(restored.agents[1].errors, 1)
        self.assertLess(restored.agents[1].reward_sum, 0.0)
        chat.close()

    def test_successful_generation_cleanup_failure_aborts_page_reward_and_teaches_error(
        self,
    ) -> None:
        runtime = _Runtime(
            model=_Model(cleanup_error=RuntimeError("cannot reset")),
        )

        class PageRouter:
            page_count = 272
            route_width = 192

            def __init__(self) -> None:
                self.events = []

            def abort_runtime_reward(self) -> None:
                self.events.append("abort")

            def begin_runtime_reward(self) -> None:
                self.events.append("begin")

            @staticmethod
            def flush() -> None:
                pass

            @staticmethod
            def metrics():
                return {
                    "adaptive_width_pages_saved": 0,
                    "runtime_reward_updates": 0,
                }

            def settle_runtime_reward(self, _receipt: str, _reward: float):
                self.events.append("settle")
                return {"runtime_reward_updates": 1}

        router = PageRouter()
        runtime.mlp_page_router = router
        chat = self._adaptive_chat(runtime)
        decoder = SimpleNamespace(
            generate_rolling=lambda *args, **kwargs: _rolling_result()
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
            return_value=decoder,
        ):
            result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertEqual(result.reason, "Qwen3.8 state cleanup failed")
        self.assertEqual(router.events, ["begin", "abort"])
        self.assertIsNone(chat._pending_page_runtime_reward)
        feedback = result.evidence["draft_window"]["feedback"]
        self.assertEqual(feedback["outcome"], "error")
        self.assertIsNone(feedback["runtime_reward"])
        self.assertLess(feedback["reward"], -9.0)
        restored = DraftWindowState.from_bytes(self.state_path.read_bytes())
        self.assertEqual(restored.updates, 1)
        self.assertEqual(restored.agents[1].errors, 1)
        self.assertLess(restored.agents[1].reward_sum, 0.0)
        self.assertEqual(runtime.close_calls, 1)
        chat.close()

    def test_draft_settlement_failure_preserves_independent_page_reward(
        self,
    ) -> None:
        runtime = _Runtime()

        class PageRouter:
            page_count = 272
            route_width = 192

            def __init__(self) -> None:
                self.active = False
                self.events = []

            def abort_runtime_reward(self) -> None:
                self.events.append("abort")
                self.active = False

            def begin_runtime_reward(self) -> None:
                self.events.append("begin")
                self.active = True

            @staticmethod
            def flush() -> None:
                pass

            @staticmethod
            def metrics():
                return {
                    "adaptive_width_pages_saved": 0,
                    "runtime_reward_updates": 0,
                }

            def settle_runtime_reward(self, _receipt: str, _reward: float):
                self.events.append("settle")
                self.active = False
                return {"runtime_reward_updates": 1}

        router = PageRouter()
        runtime.mlp_page_router = router
        chat = self._adaptive_chat(runtime)
        decoder = SimpleNamespace(
            generate_rolling=lambda *args, **kwargs: _rolling_result()
        )
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
                return_value=decoder,
            ),
            patch.object(
                chat._draft_window_controller,
                "settle",
                side_effect=OSError("window disk full"),
            ),
        ):
            result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertEqual(result.reason, "Qwen3.8 draft-window settlement failed")
        self.assertEqual(router.events, ["begin"])
        self.assertTrue(router.active)
        self.assertIsNotNone(chat._pending_page_runtime_reward)
        self.assertNotIn("settlement", result.evidence["runtime_reward"])
        chat.close()

    def test_broken_auxiliary_metrics_teach_terminal_error_reward(
        self,
    ) -> None:
        runtime = _Runtime()
        chat = self._adaptive_chat(runtime)
        provider = SimpleNamespace(
            metrics=lambda: (_ for _ in ()).throw(RuntimeError("metrics failed")),
            close=lambda: None,
        )
        decoder = SimpleNamespace(
            generate_rolling=lambda *args, **kwargs: _rolling_result()
        )
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.FingerprintRollingK4DraftProvider",
                return_value=provider,
            ),
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
                return_value=decoder,
            ),
        ):
            result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        feedback = result.evidence["draft_window"]["feedback"]
        self.assertEqual(feedback["outcome"], "error")
        self.assertIsNone(feedback["runtime_reward"])
        self.assertLess(feedback["reward"], -9.0)
        self.assertEqual(feedback["target_source_body_bytes"], 100)
        self.assertEqual(feedback["draft_source_body_bytes"], 0)
        restored = DraftWindowState.from_bytes(self.state_path.read_bytes())
        self.assertEqual(restored.updates, 1)
        self.assertEqual(restored.agents[1].errors, 1)
        chat.close()

    def test_target_timeout_teaches_the_selected_window_once(self) -> None:
        runtime = _Runtime()
        chat = self._adaptive_chat(runtime)
        decoder = SimpleNamespace(
            generate_rolling=lambda *args, **kwargs: (_ for _ in ()).throw(
                TimeoutError("target timed out")
            )
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
            return_value=decoder,
        ):
            result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        receipt = result.evidence["draft_window"]
        self.assertTrue(receipt["settled"])
        self.assertEqual(receipt["feedback"]["outcome"], "timeout")
        self.assertLess(receipt["feedback"]["reward"], -12.0)
        restored = DraftWindowState.from_bytes(self.state_path.read_bytes())
        self.assertEqual(restored.updates, 1)
        self.assertEqual(restored.agents[1].timeouts, 1)
        chat.close()

    def test_provider_setup_timeout_does_not_blame_target_window(self) -> None:
        runtime = _Runtime()
        chat = self._adaptive_chat(runtime)
        with patch(
            "immer.runtimes.qwen3_8.adapter.FingerprintRollingK4DraftProvider",
            side_effect=TimeoutError("provider setup timed out"),
        ):
            result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertFalse(result.evidence["draft_window"]["settled"])
        self.assertEqual(DraftWindowController(self.state_path).metrics().updates, 0)
        chat.close()

    def test_non_timeout_failure_without_target_result_does_not_teach(self) -> None:
        runtime = _Runtime()
        chat = self._adaptive_chat(runtime)
        decoder = SimpleNamespace(
            generate_rolling=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("decoder failed")
            )
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
            return_value=decoder,
        ):
            result = chat.handle(Request("chat", "hello"))

        self.assertIs(result.status, ExecutionStatus.ERROR)
        self.assertFalse(result.evidence["draft_window"]["settled"])
        self.assertEqual(DraftWindowController(self.state_path).metrics().updates, 0)
        chat.close()

    def test_abort_after_target_receipt_is_settled_then_re_raised(self) -> None:
        class AbortedDecode(_Tokenizer):
            def decode(self, token_ids):
                raise KeyboardInterrupt("cancelled")

        runtime = _Runtime(tokenizer=AbortedDecode())
        chat = self._adaptive_chat(runtime)
        decoder = SimpleNamespace(
            generate_rolling=lambda *args, **kwargs: _rolling_result()
        )
        with (
            patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
                return_value=decoder,
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            chat.handle(Request("chat", "hello"))

        restored = DraftWindowState.from_bytes(self.state_path.read_bytes())
        self.assertEqual(restored.updates, 1)
        self.assertEqual(restored.agents[1].aborted, 1)
        self.assertLess(restored.agents[1].reward_sum, 0.0)
        chat.close()

    def test_output_budget_below_k8_selects_k4(self) -> None:
        runtime = _Runtime()
        chat = self._adaptive_chat(runtime, max_new_tokens=6)
        generated = _rolling_result(window=4)
        decoder = SimpleNamespace(generate_rolling=lambda *args, **kwargs: generated)
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
            return_value=decoder,
        ) as constructor:
            result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(constructor.call_args.kwargs["window_size"], 4)
        self.assertEqual(
            result.evidence["draft_window"]["selection"]["eligible_windows"],
            [4],
        )
        chat.close()

    def test_short_output_budget_keeps_rolling_draft_without_policy_update(self) -> None:
        runtime = _Runtime()
        chat = self._adaptive_chat(runtime, max_new_tokens=3)
        generated = _rolling_result(
            window=3,
            accepted=2,
            token_ids=(7, 8, 9),
        )
        decoder = SimpleNamespace(generate_rolling=lambda *args, **kwargs: generated)
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
            return_value=decoder,
        ) as constructor:
            result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(constructor.call_args.kwargs["window_size"], 3)
        self.assertNotIn("draft_window", result.evidence)
        self.assertEqual(DraftWindowController(self.state_path).metrics().updates, 0)
        chat.close()

    def test_policy_exposes_choice_and_pre_request_cumulative_metrics(self) -> None:
        runtime = _Runtime()
        chat = self._adaptive_chat(runtime)
        decoder = SimpleNamespace(
            generate_rolling=lambda *args, **kwargs: _rolling_result()
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
            return_value=decoder,
        ):
            result = chat.handle(Request("chat", "hello"))
        self.assertTrue(result.ok)

        with patch(
            "immer.runtimes.qwen3_8.adapter._digest",
            side_effect=lambda value: value,
        ):
            policy = chat._result_cell_generation_policy_sha256()

        controller = policy["draft_window_controller"]
        self.assertEqual(policy["draft_window"], 8)
        self.assertEqual(controller["selection"]["proposed_window"], 8)
        self.assertEqual(controller["metrics"]["agents"]["8"]["observations"], 0)
        self.assertEqual(
            result.evidence["draft_window"]["metrics"]["agents"]["8"]["observations"],
            1,
        )
        chat.close()

    def test_adaptive_result_cell_keeps_the_exact_generation_policy_binding(
        self,
    ) -> None:
        runtime = _Runtime()
        chat = self._adaptive_chat(
            runtime,
            result_cell_code_revision="f" * 40,
        )
        decoder = SimpleNamespace(
            generate_rolling=lambda *args, **kwargs: _rolling_result()
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
            return_value=decoder,
        ):
            result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        bound = result.evidence["result_cell_binding_receipt"]["body"]
        self.assertEqual(
            bound["generation_policy_sha256"],
            chat._result_cell_generation_policy_sha256(),
        )
        chat.close()

    def test_cli_wires_draft_window_state_as_an_explicit_opt_in(self) -> None:
        qwen = _chat(_Runtime())
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
            return_value=qwen,
        ) as constructor:
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--draft-mode",
                        "markov",
                        "--draft-window",
                        "12",
                        "--draft-window-state",
                        "/state/qwen-window.bin",
                        "--range-markov-state",
                        "/state/qwen-ranges.bin",
                        "--range-prefetch-max-mb",
                        "32",
                        "--range-prefetch-min-support",
                        "3",
                        "--range-prefetch-min-confidence",
                        "0.8",
                        "--range-prefetch-beam-horizon",
                        "4",
                        "--range-prefetch-beam-width",
                        "6",
                        "--range-prefetch-hint-cooldown",
                        "3",
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["draft_window"], 12)
        self.assertEqual(
            options["draft_window_state_path"],
            "/state/qwen-window.bin",
        )
        self.assertEqual(options["range_markov_state_path"], "/state/qwen-ranges.bin")
        self.assertEqual(options["range_prefetch_max_bytes"], 32 * 1024**2)
        self.assertEqual(options["range_prefetch_min_support"], 3)
        self.assertEqual(options["range_prefetch_min_confidence"], 0.8)
        self.assertEqual(options["range_prefetch_beam_horizon"], 4)
        self.assertEqual(options["range_prefetch_beam_width"], 6)
        self.assertEqual(options["range_prefetch_hint_cooldown"], 3)

    def test_cli_canonical_deployment_auto_wires_draft_window_state(self) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            markov_state = Path(temporary) / "qwen-markov.bin"
            markov_state.write_bytes(b"fixture")
            draft_window_state = Path(temporary) / "qwen-draft-window.bin"
            with (
                patch.dict("os.environ", {}, clear=True),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE",
                    markov_state,
                ),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_DRAFT_WINDOW_STATE",
                    draft_window_state,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(["chat", "hello", "--raw-qwen"])

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["draft_mode"], "markov")
        self.assertEqual(options["draft_window"], 8)
        self.assertEqual(
            options["draft_window_state_path"],
            str(draft_window_state),
        )

    def test_cli_can_opt_out_of_default_or_environment_draft_window_state(
        self,
    ) -> None:
        qwen = _chat(_Runtime())
        with tempfile.TemporaryDirectory() as temporary:
            deployed = Path(temporary) / "deployed"
            (deployed / "causal" / "q4-base-v2").mkdir(parents=True)
            markov_state = Path(temporary) / "qwen-markov.bin"
            markov_state.write_bytes(b"fixture")
            with (
                patch.dict(
                    "os.environ",
                    {
                        "IMMER_QWEN38_DRAFT_WINDOW_STATE": "/env/window.bin",
                    },
                    clear=True,
                ),
                patch("immer.cli._QWEN38_DEPLOYMENT_ROOT", deployed),
                patch(
                    "immer.cli._QWEN38_DEPLOYMENT_MARKOV_DRAFT_STATE",
                    markov_state,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--no-draft-window-controller",
                    ]
                )

        self.assertEqual(code, 0)
        self.assertIsNone(
            constructor.call_args.kwargs["draft_window_state_path"]
        )

        error = io.StringIO()
        with redirect_stderr(error):
            conflict = main(
                [
                    "chat",
                    "hello",
                    "--qwen38-causal-bundle",
                    "/models/qwen.causal",
                    "--qwen38-tokenizer",
                    "/models/tokenizer.json",
                    "--draft-window-state",
                    "/state/window.bin",
                    "--no-draft-window-controller",
                ]
            )
        self.assertEqual(conflict, 2)
        self.assertIn("mutually exclusive", error.getvalue())

    def test_no_state_keeps_the_existing_fixed_window_contract(self) -> None:
        runtime = _Runtime()
        runtime.model.max_seq_len = 32
        chat = _chat(
            runtime,
            draft_mode="markov",
            draft_window=12,
            max_new_tokens=16,
            max_context_tokens=32,
        )
        decoder = SimpleNamespace(
            generate_rolling=lambda *args, **kwargs: _rolling_result(window=12)
        )
        with patch(
            "immer.runtimes.qwen3_8.adapter.Qwen38K4SpeculativeDecoder",
            return_value=decoder,
        ) as constructor:
            result = chat.handle(Request("chat", "hello"))

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(constructor.call_args.kwargs["window_size"], 12)
        self.assertNotIn("draft_window", result.evidence)
        chat.close()


if __name__ == "__main__":
    unittest.main()
