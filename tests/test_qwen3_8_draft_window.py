from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
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
    DRAFT_WINDOW_STATE_SCHEMA,
    LEGACY_DRAFT_WINDOW_STATE_SCHEMA,
    DraftWindowController,
    DraftWindowError,
    DraftWindowFeedback,
    DraftWindowState,
    contextual_bottom_k_signature,
)

from test_qwen3_8_adapter import _Runtime, _Tokenizer, _chat


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


def _rolling_result(window: int = 8, *, accepted: int = 4):
    evidence = SimpleNamespace(
        accepted_draft_tokens=accepted,
        source_body_bytes=100,
        linear_calls=10,
        seconds=1.0,
        state_bytes=456,
        stopped_on_eos=False,
        prompt_token_ids=(11, 12),
        generated_token_ids=(7, 8, 9, 10),
        forward_passes=3,
        rounds=(object(),),
        schema="immer.qwen3.8-rolling-speculative-generation/v2",
        final_state_committed=False,
        window_size=window,
    )
    return SimpleNamespace(token_ids=(7, 8, 9, 10), evidence=evidence)


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
        for index in range(1, 9):
            selection = controller.choose((5, 5, 5), max_window=16, max_new_tokens=16)
            controller.settle(
                selection,
                _feedback(
                    selection,
                    accepted=8,
                    receipt=f"{index:064x}",
                ),
            )
        for index in range(9, 11):
            selection = controller.choose((5, 5, 5), max_window=16, max_new_tokens=16)
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
        self.assertTrue(self.state_path.read_bytes().startswith(b"IMDW\x01"))

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
        self.assertEqual(receipt["metrics"]["agents"]["8"]["observations"], 1)
        self.assertEqual(result.evidence["draft"]["window_size"], 8)
        chat.close()

    def test_error_after_target_receipt_is_an_atomic_negative_outcome(self) -> None:
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
        self.assertLess(receipt["feedback"]["reward"], -9.0)
        restored = DraftWindowState.from_bytes(self.state_path.read_bytes())
        self.assertEqual(restored.updates, 1)
        self.assertEqual(restored.agents[1].errors, 1)
        chat.close()

    def test_broken_auxiliary_metrics_still_settle_target_receipt_negative(
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
        self.assertEqual(feedback["target_source_body_bytes"], 100)
        self.assertEqual(feedback["draft_source_body_bytes"], 0)
        restored = DraftWindowState.from_bytes(self.state_path.read_bytes())
        self.assertEqual(restored.updates, 1)
        self.assertEqual(restored.agents[1].errors, 1)
        chat.close()

    def test_timeout_without_target_receipt_does_not_mutate_controller(self) -> None:
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
        self.assertFalse(result.evidence["draft_window"]["settled"])
        self.assertEqual(
            result.evidence["draft_window"]["reason"],
            "no-verified-target-receipt",
        )
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
        self.assertLess(restored.agents[1].reward_sum, -10.0)
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
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["draft_window"], 12)
        self.assertEqual(
            options["draft_window_state_path"],
            "/state/qwen-window.bin",
        )

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
