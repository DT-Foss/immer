from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from immer.contracts import ExecutionStatus, Result
from immer.runtimes.qwen3_8.action_bank import (
    INFERENCE_ACTION_BANK_SCHEMA,
    InferenceActionBank,
    InferenceActionBankError,
    InferenceActionDirective,
    InferenceActionReceipt,
    executed_actions_from_result,
)
from immer.runtimes.qwen3_8.inference_economics import InferenceEconomicsReceipt


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _economics(
    request: str,
    *,
    warm: bool = False,
    fertig: bool = False,
    draft: bool = True,
    pages: bool = True,
    target: int = 3,
    saved: int = 1,
) -> InferenceEconomicsReceipt:
    return InferenceEconomicsReceipt(
        request_sha256=_sha(request),
        question_sha256=_sha(f"question:{request}"),
        runtime_profile_sha256=_sha("profile"),
        result_sha256=_sha(f"result:{request}"),
        output_sha256=_sha(f"output:{request}"),
        status="ok",
        component="qwen3.8.fertig-chat",
        route="ooe_verification_abstained" if warm else "qwen_verification_abstained",
        target_forwards=target,
        saved_qwen_forwards=saved,
        generated_tokens=target + saved,
        accepted_draft_tokens=saved if draft else 0,
        proposed_draft_tokens=saved + 1 if draft else 0,
        source_body_bytes=100,
        target_source_body_bytes=90,
        draft_source_body_bytes=10 if draft else 0,
        logical_weight_bytes=500,
        page_mlp_weight_bytes=200 if pages else 0,
        prefetch_bytes=50 if pages else 0,
        physical_read_bytes=25,
        linear_calls=10,
        process_peak_rss_bytes=1024,
        generation_seconds=2.0,
        request_wall_seconds=2.25,
        time_to_first_token_seconds=0.5,
        selected_pages=12 if pages else 0,
        saved_pages=2 if pages else 0,
        o1_priority=1.5,
        runtime_reward=0.25,
        warm_hit=warm,
        fertig_exact=fertig,
        draft_active=draft,
        page_active=pages,
        avoidable_work_bytes={"target_fallback": 500},
    )


class InferenceActionReceiptTests(unittest.TestCase):
    def test_normal_cold_request_becomes_one_content_addressed_action_vector(self) -> None:
        receipt = InferenceActionReceipt.from_economics(_economics("one"))
        self.assertEqual(
            receipt.actions,
            ("dynamic_mlp_pages", "qwen_target", "target_verified_draft"),
        )
        self.assertEqual(receipt.target_forwards, 3)
        self.assertEqual(receipt.saved_qwen_forwards, 1)
        self.assertEqual(
            InferenceActionReceipt.from_document(receipt.to_document()),
            receipt,
        )
        self.assertNotIn("question:one", json.dumps(receipt.to_document()))

    def test_legacy_draft_savings_flag_cannot_impersonate_a_zero_forward_hit(
        self,
    ) -> None:
        economics = _economics("legacy-cold", warm=True, target=3, saved=1)
        receipt = InferenceActionReceipt.from_economics(economics)
        self.assertEqual(
            receipt.actions,
            ("dynamic_mlp_pages", "qwen_target", "target_verified_draft"),
        )
        self.assertEqual(receipt.target_forwards, 3)
        self.assertEqual(receipt.accepted_draft_tokens, 1)
        self.assertEqual(receipt.source_body_bytes, 100)

    def test_parametric_program_is_distinct_from_an_exact_result_cell(self) -> None:
        economics = _economics(
            "program",
            warm=True,
            draft=False,
            pages=False,
            target=0,
            saved=4,
        )
        result = Result(
            ExecutionStatus.OK,
            "qwen3.8.fertig-chat",
            output="OMEGA",
            evidence={
                "receipt": {
                    "qwen": {"component": "immer.markov-parametric-template"}
                }
            },
        )
        actions = executed_actions_from_result(result, economics)
        self.assertEqual(actions, ("parametric_program",))
        receipt = InferenceActionReceipt.from_economics(
            economics,
            executed_actions=actions,
        )
        self.assertEqual(receipt.actions, ("parametric_program",))
        self.assertEqual(receipt.target_forwards, 0)

    def test_exact_anchor_restore_is_a_continuation_battery_action(self) -> None:
        economics = _economics("battery")
        result = Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output="answer",
            evidence={
                "anchor_cache": {
                    "forward_passes_saved": 1,
                    "prompt_token_layer_evaluations_saved": 640,
                    "status": "hit",
                }
            },
        )
        actions = executed_actions_from_result(result, economics)
        self.assertEqual(
            actions,
            (
                "continuation_battery",
                "dynamic_mlp_pages",
                "qwen_target",
                "target_verified_draft",
            ),
        )

    def test_native_prefix_sinkhorn_is_recorded_as_executed_attention(self) -> None:
        economics = _economics("prefix-sinkhorn")
        action_identity = _sha("physical prefix action")
        result = Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output="answer",
            evidence={
                "prefix_sinkhorn": {
                    "action_identity_sha256": action_identity,
                    "active": True,
                    "available": True,
                    "configuration": {
                        "alpha": 1.0,
                        "layer": 27,
                        "replace_base_softmax": True,
                    },
                    "request": {
                        "base_softmax_head_rows_skipped": 12,
                        "base_softmax_probability_elements_skipped": 144,
                    },
                    "schema": (
                        "immer.qwen3.8-prefix-sinkhorn-action-evidence/v1"
                    ),
                }
            },
        )
        actions = executed_actions_from_result(result, economics)
        self.assertEqual(
            actions,
            (
                "dynamic_mlp_pages",
                "prefix_sinkhorn",
                "qwen_target",
                "target_verified_draft",
            ),
        )

        static_only = Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output="answer",
            evidence={
                "prefix_sinkhorn": {
                    "action_identity_sha256": action_identity,
                    "active": True,
                    "available": True,
                    "configuration": {
                        "alpha": 1.0,
                        "replace_base_softmax": True,
                    },
                    "schema": (
                        "immer.qwen3.8-prefix-sinkhorn-action-evidence/v1"
                    ),
                }
            },
        )
        self.assertNotIn(
            "prefix_sinkhorn",
            executed_actions_from_result(static_only, economics),
        )

    def test_executed_delta_head_work_is_recorded_as_coordinate_action(self) -> None:
        economics = _economics("delta-head")
        result = Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output="answer",
            evidence={
                "delta_head_router": {
                    "schema": "immer.qwen3.8-packed-delta-head-router/v1",
                    "request": {
                        "calls": 2,
                        "logical_bytes_saved": 4_096,
                        "rows": 3,
                    },
                }
            },
        )
        actions = executed_actions_from_result(result, economics)
        self.assertEqual(
            actions,
            (
                "dynamic_mlp_pages",
                "mlp_head_coordinate",
                "qwen_target",
                "target_verified_draft",
            ),
        )

    def test_exact_q8_head_pruning_is_recorded_as_lm_head_coordinate(self) -> None:
        economics = _economics(
            "q8-head-coordinate",
            draft=False,
            pages=False,
            saved=0,
        )
        result = Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output="answer",
            evidence={
                "exact_head": {
                    "manifest_sha256": "a" * 64,
                    "tensor_sha256": "b" * 64,
                    "request": {
                        "applicable_calls": 3,
                        "logical_head_bytes_avoided": 4096,
                        "manifest_sha256": "a" * 64,
                        "packed_rows_avoided": 32,
                        "packed_weight_bytes_avoided": 4096,
                        "rows_pruned": 32,
                        "schema": "immer.qwen3.8-exact-head-request/v1",
                    },
                }
            },
        )
        self.assertEqual(
            executed_actions_from_result(result, economics),
            ("lm_head_coordinate", "qwen_target"),
        )
        no_savings = Result(
            result.status,
            result.component,
            output=result.output,
            evidence={
                "exact_head": {
                    **dict(result.evidence["exact_head"]),
                    "request": {
                        **dict(result.evidence["exact_head"]["request"]),
                        "packed_weight_bytes_avoided": 0,
                    },
                }
            },
        )
        self.assertEqual(
            executed_actions_from_result(no_savings, economics),
            ("qwen_target",),
        )

    def test_delta_head_identity_without_saved_request_work_is_not_an_action(
        self,
    ) -> None:
        economics = _economics("delta-head-inactive")
        for missing_work in ("calls", "rows", "logical_bytes_saved"):
            with self.subTest(missing_work=missing_work):
                request = {
                    "calls": 2,
                    "logical_bytes_saved": 4_096,
                    "rows": 3,
                }
                request[missing_work] = 0
                result = Result(
                    ExecutionStatus.OK,
                    "qwen3.8.causal-chat",
                    output="answer",
                    evidence={
                        "delta_head_router": {
                            "schema": (
                                "immer.qwen3.8-packed-delta-head-router/v1"
                            ),
                            "request": request,
                        }
                    },
                )
                self.assertNotIn(
                    "mlp_head_coordinate",
                    executed_actions_from_result(result, economics),
                )

    def test_accepted_context_crystal_is_recorded_as_compute_crystal(self) -> None:
        economics = _economics("context-crystal")
        result = Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output="answer",
            evidence={
                "draft": {
                    "context_crystal": {
                        "crystal_accepted_tokens": 7,
                        "crystal_proposed_tokens": 15,
                    }
                }
            },
        )

        actions = executed_actions_from_result(result, economics)

        self.assertIn("compute_crystal", actions)
        self.assertIn("target_verified_draft", actions)

    def test_unaccepted_context_crystal_is_not_an_executed_compute_action(
        self,
    ) -> None:
        economics = _economics("context-crystal-miss")
        result = Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output="answer",
            evidence={
                "draft": {
                    "context_crystal": {
                        "crystal_accepted_tokens": 0,
                        "crystal_proposed_tokens": 15,
                    }
                }
            },
        )

        self.assertNotIn(
            "compute_crystal",
            executed_actions_from_result(result, economics),
        )

    def test_exact_attention_output_crystal_records_physical_replay(self) -> None:
        economics = _economics(
            "attention-output-crystal",
            draft=False,
            pages=False,
            saved=0,
        )
        result = Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output="answer",
            evidence={
                "receipt": {
                    "qwen": {
                        "attention_output_crystal": {
                            "schema": (
                                "immer.qwen3.8-attention-output-crystal-evidence/v1"
                            ),
                            "request": {
                                "hits": 2,
                                "logical_projection_bytes_saved": 8_192,
                                "skipped_projection_calls": 8,
                            },
                        }
                    }
                }
            },
        )

        actions = executed_actions_from_result(result, economics)

        self.assertEqual(actions, ("attention_output_crystal", "qwen_target"))
        receipt = InferenceActionReceipt.from_economics(
            economics,
            executed_actions=actions,
        )
        self.assertEqual(receipt.actions, actions)
        self.assertEqual(receipt.saved_qwen_forwards, 0)

    def test_attention_output_crystal_requires_exact_positive_physical_work(
        self,
    ) -> None:
        economics = _economics(
            "attention-output-crystal-invalid",
            draft=False,
            pages=False,
            saved=0,
        )
        invalid_values = (
            ("hits", 0),
            ("hits", True),
            ("skipped_projection_calls", 0),
            ("logical_projection_bytes_saved", 0),
        )
        for field, invalid in invalid_values:
            with self.subTest(field=field, invalid=invalid):
                request = {
                    "hits": 2,
                    "logical_projection_bytes_saved": 8_192,
                    "skipped_projection_calls": 8,
                }
                request[field] = invalid
                result = Result(
                    ExecutionStatus.OK,
                    "qwen3.8.causal-chat",
                    output="answer",
                    evidence={
                        "attention_output_crystal": {
                            "schema": (
                                "immer.qwen3.8-attention-output-crystal-evidence/v1"
                            ),
                            "request": request,
                        }
                    },
                )
                self.assertNotIn(
                    "attention_output_crystal",
                    executed_actions_from_result(result, economics),
                )

        near_evidence = Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output="answer",
            evidence={
                "attention_output_crystal": {
                    "schema": "immer.qwen3.8-attention-output-crystal-near/v1",
                    "request": {
                        "hits": 2,
                        "logical_projection_bytes_saved": 8_192,
                        "skipped_projection_calls": 8,
                    },
                }
            },
        )
        self.assertNotIn(
            "attention_output_crystal",
            executed_actions_from_result(near_evidence, economics),
        )

        zero_target = _economics(
            "attention-output-crystal-zero-target",
            draft=False,
            pages=False,
            target=0,
            saved=0,
        )
        self.assertNotIn(
            "attention_output_crystal",
            executed_actions_from_result(
                Result(
                    ExecutionStatus.OK,
                    "qwen3.8.causal-chat",
                    output="answer",
                    evidence={
                        "attention_output_crystal": {
                            "schema": (
                                "immer.qwen3.8-attention-output-crystal-evidence/v1"
                            ),
                            "request": {
                                "hits": 2,
                                "logical_projection_bytes_saved": 8_192,
                                "skipped_projection_calls": 8,
                            },
                        }
                    },
                ),
                zero_target,
            ),
        )

    def test_exact_mlp_page_coordinate_records_physical_page_work(self) -> None:
        economics = _economics(
            "mlp-page-coordinate",
            draft=False,
            pages=False,
            saved=0,
        )
        result = Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output="answer",
            evidence={
                "mlp_page_coordinate": {
                    "schema": "immer.qwen3.8-mlp-page-coordinate-evidence/v1",
                    "request": {
                        "hits": 4,
                        "logical_page_weight_bytes_saved": 8192,
                        "physical_pages_saved": 320,
                    },
                }
            },
        )
        self.assertEqual(
            executed_actions_from_result(result, economics),
            ("mlp_page_coordinate", "qwen_target"),
        )


class InferenceActionBankTests(unittest.TestCase):
    def test_explicit_prefix_disable_blocks_warm_replay_but_omission_does_not(
        self,
    ) -> None:
        from immer.runtimes.ooe.qwen_warm_growth import (
            prefix_sinkhorn_warm_allowed,
        )

        base = dict(
            question_sha256=_sha("warm prefix policy"),
            runtime_profile_sha256=_sha("profile"),
            primary_actions=("qwen_target",),
            fallback_actions=("qwen_target",),
            draft_enabled=None,
            source_signature_sha256s=("6" * 64,),
            support=1,
            saved_qwen_forwards=0,
        )
        omitted = InferenceActionDirective(**base)
        disabled = InferenceActionDirective(
            **base,
            disabled_actions=("prefix_sinkhorn",),
        )
        self.assertTrue(prefix_sinkhorn_warm_allowed({}))
        self.assertTrue(
            prefix_sinkhorn_warm_allowed(
                {"qwen_inference_action_directive": omitted.to_document()}
            )
        )
        self.assertFalse(
            prefix_sinkhorn_warm_allowed(
                {"qwen_inference_action_directive": disabled.to_document()}
            )
        )
        self.assertFalse(
            prefix_sinkhorn_warm_allowed(
                {"qwen_inference_action_directive": {"tampered": True}}
            )
        )

    def test_observe_restart_duplicate_and_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "actions"
            bank = InferenceActionBank(root)
            first = bank.observe(_economics("cold"))
            duplicate = bank.observe(_economics("cold"))
            warm = bank.observe(
                _economics(
                    "warm",
                    warm=True,
                    target=0,
                    saved=4,
                )
            )
            program = bank.observe(
                _economics(
                    "program",
                    warm=True,
                    draft=False,
                    pages=False,
                    target=0,
                    saved=2,
                ),
                executed_actions=("parametric_program",),
            )
            snapshot = bank.snapshot()
            reconciled = bank.reconcile(
                (
                    _economics("cold"),
                    _economics(
                        "program",
                        warm=True,
                        draft=False,
                        pages=False,
                        target=0,
                        saved=2,
                    ),
                )
            )
            directive = bank.recommend(
                question_sha256=_sha("question:warm"),
                runtime_profile_sha256=_sha("profile"),
            )
            runtime_directive = bank.recommend(
                question_sha256=_sha("new question"),
                runtime_profile_sha256=_sha("profile"),
            )
            transferred_directive = bank.recommend(
                question_sha256=_sha("new question"),
                runtime_profile_sha256=_sha("other profile"),
            )
            restarted = InferenceActionBank(root)
            restarted_snapshot = restarted.snapshot()

        self.assertFalse(first.duplicate)
        self.assertTrue(duplicate.duplicate)
        self.assertFalse(warm.duplicate)
        self.assertFalse(program.duplicate)
        self.assertEqual(snapshot["schema"], INFERENCE_ACTION_BANK_SCHEMA)
        self.assertEqual(snapshot["requests"], 3)
        self.assertEqual(len(snapshot["signatures"]), 3)
        self.assertEqual(snapshot["signatures"][0]["actions"], ["stored_result"])
        self.assertEqual(snapshot["signatures"][0]["saved_qwen_forwards"], 4)
        self.assertEqual(snapshot["signatures"][0]["accepted_draft_tokens"], 0)
        self.assertEqual(snapshot["signatures"][0]["source_body_bytes"], 0)
        self.assertEqual(restarted_snapshot, snapshot)
        self.assertEqual(reconciled, snapshot)
        assert directive is not None
        self.assertEqual(directive.primary_actions, ("stored_result",))
        self.assertEqual(
            directive.fallback_actions,
            ("dynamic_mlp_pages", "qwen_target", "target_verified_draft"),
        )
        self.assertTrue(directive.draft_enabled)
        self.assertIsNone(directive.draft_window_ceiling)
        self.assertEqual(directive.support, 3)
        self.assertEqual(directive.saved_qwen_forwards, 7)
        self.assertEqual(
            InferenceActionDirective.from_document(directive.to_document()),
            directive,
        )
        assert runtime_directive is not None
        self.assertEqual(runtime_directive.primary_actions, ("parametric_program",))
        self.assertEqual(runtime_directive.fallback_actions, directive.fallback_actions)
        self.assertTrue(runtime_directive.draft_enabled)
        assert transferred_directive is not None
        self.assertEqual(
            transferred_directive.primary_actions,
            ("parametric_program",),
        )
        self.assertEqual(
            transferred_directive.fallback_actions,
            directive.fallback_actions,
        )
        self.assertTrue(transferred_directive.draft_enabled)

    def test_positive_compute_crystal_authorizes_k16_target_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = InferenceActionBank(Path(temporary) / "actions")
            bank.observe(
                _economics("crystal", target=29, saved=11),
                executed_actions=(
                    "compute_crystal",
                    "continuation_battery",
                    "dynamic_mlp_pages",
                    "qwen_target",
                    "target_verified_draft",
                ),
            )

            directive = bank.recommend(
                question_sha256=_sha("new question"),
                runtime_profile_sha256=_sha("other profile"),
            )

        assert directive is not None
        self.assertEqual(directive.draft_window_ceiling, 16)
        self.assertIn("compute_crystal", directive.primary_actions)
        self.assertIn("compute_crystal", directive.fallback_actions)
        self.assertEqual(
            InferenceActionDirective.from_document(directive.to_document()),
            directive,
        )

        for changes in (
            {"draft_enabled": False},
            {"saved_qwen_forwards": 0},
            {
                "primary_actions": ("qwen_target",),
                "fallback_actions": ("qwen_target",),
            },
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                InferenceActionDirective(
                    question_sha256=_sha("question"),
                    runtime_profile_sha256=_sha("profile"),
                    primary_actions=changes.get(
                        "primary_actions", ("compute_crystal", "qwen_target")
                    ),
                    fallback_actions=changes.get(
                        "fallback_actions", ("compute_crystal", "qwen_target")
                    ),
                    draft_enabled=changes.get("draft_enabled", True),
                    source_signature_sha256s=("3" * 64,),
                    support=1,
                    saved_qwen_forwards=changes.get("saved_qwen_forwards", 1),
                    draft_window_ceiling=16,
                )

    def test_attention_output_crystal_recommends_without_draft_savings(self) -> None:
        economics = _economics(
            "attention-runtime",
            draft=False,
            pages=False,
            saved=0,
        )
        result = Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output="answer",
            evidence={
                "attention_output_crystal": {
                    "schema": (
                        "immer.qwen3.8-attention-output-crystal-evidence/v1"
                    ),
                    "request": {
                        "hits": 1,
                        "logical_projection_bytes_saved": 4_096,
                        "skipped_projection_calls": 4,
                    },
                }
            },
        )
        actions = executed_actions_from_result(result, economics)
        with tempfile.TemporaryDirectory() as temporary:
            bank = InferenceActionBank(Path(temporary) / "actions")
            observation = bank.observe(economics, executed_actions=actions)
            directive = bank.recommend(
                question_sha256=_sha("new attention question"),
                runtime_profile_sha256=_sha("other profile"),
            )

        self.assertEqual(
            observation.snapshot["action_catalog"][0],
            {"action": "attention_output_crystal", "observed_requests": 1},
        )
        assert directive is not None
        self.assertEqual(
            directive.primary_actions,
            ("attention_output_crystal", "qwen_target"),
        )
        self.assertEqual(directive.fallback_actions, directive.primary_actions)
        self.assertIsNone(directive.draft_enabled)
        self.assertEqual(directive.saved_qwen_forwards, 0)
        self.assertIsNone(directive.draft_window_ceiling)
        self.assertEqual(
            InferenceActionDirective.from_document(directive.to_document()),
            directive,
        )

        with self.assertRaises(ValueError):
            InferenceActionDirective(
                question_sha256=_sha("question"),
                runtime_profile_sha256=_sha("profile"),
                primary_actions=("attention_output_crystal",),
                fallback_actions=("qwen_target",),
                draft_enabled=False,
                source_signature_sha256s=("3" * 64,),
                support=1,
                saved_qwen_forwards=0,
            )
        disabled = InferenceActionDirective(
            question_sha256=_sha("explicit prefix disable"),
            runtime_profile_sha256=_sha("profile"),
            primary_actions=("qwen_target",),
            fallback_actions=("qwen_target",),
            draft_enabled=None,
            source_signature_sha256s=("4" * 64,),
            support=1,
            saved_qwen_forwards=0,
            disabled_actions=("prefix_sinkhorn",),
        )
        self.assertEqual(
            InferenceActionDirective.from_document(disabled.to_document()),
            disabled,
        )
        with self.assertRaisesRegex(ValueError, "conflict"):
            InferenceActionDirective(
                question_sha256=_sha("conflicting prefix disable"),
                runtime_profile_sha256=_sha("profile"),
                primary_actions=("prefix_sinkhorn", "qwen_target"),
                fallback_actions=("qwen_target",),
                draft_enabled=None,
                source_signature_sha256s=("5" * 64,),
                support=1,
                saved_qwen_forwards=0,
                disabled_actions=("prefix_sinkhorn",),
            )

    def test_attention_replay_stays_additive_to_higher_ranked_drafting(self) -> None:
        attention = _economics(
            "attention-additive",
            draft=False,
            pages=False,
            saved=0,
        )
        attention_actions = (
            "attention_output_crystal",
            "qwen_target",
        )
        draft = _economics("higher-ranked-draft", saved=5)
        with tempfile.TemporaryDirectory() as temporary:
            bank = InferenceActionBank(Path(temporary) / "actions")
            bank.observe(attention, executed_actions=attention_actions)
            bank.observe(draft)
            directive = bank.recommend(
                question_sha256=_sha("additive question"),
                runtime_profile_sha256=_sha("other profile"),
            )

        assert directive is not None
        self.assertIn("attention_output_crystal", directive.primary_actions)
        self.assertIn("target_verified_draft", directive.primary_actions)
        self.assertTrue(directive.draft_enabled)

    def test_lm_head_coordinate_is_safe_additive_across_runtime_profiles(self) -> None:
        economics = _economics(
            "lm-head-coordinate-runtime",
            draft=False,
            pages=False,
            saved=0,
        )
        actions = ("lm_head_coordinate", "qwen_target")
        with tempfile.TemporaryDirectory() as temporary:
            bank = InferenceActionBank(Path(temporary) / "actions")
            bank.observe(economics, executed_actions=actions)
            directive = bank.recommend(
                question_sha256=_sha("other lm-head question"),
                runtime_profile_sha256=_sha("other profile"),
            )

        assert directive is not None
        self.assertEqual(directive.primary_actions, actions)
        self.assertEqual(directive.fallback_actions, actions)
        with self.assertRaises(ValueError):
            InferenceActionDirective(
                question_sha256=_sha("invalid lm-head coordinate"),
                runtime_profile_sha256=_sha("profile"),
                primary_actions=("lm_head_coordinate",),
                fallback_actions=("qwen_target",),
                draft_enabled=None,
                source_signature_sha256s=("7" * 64,),
                support=1,
                saved_qwen_forwards=0,
            )

    def test_delta_coordinate_is_recommended_only_for_the_same_runtime(self) -> None:
        economics = _economics(
            "delta-coordinate",
            draft=False,
            pages=False,
            saved=0,
        )
        actions = ("mlp_head_coordinate", "qwen_target")
        with tempfile.TemporaryDirectory() as temporary:
            bank = InferenceActionBank(Path(temporary) / "actions")
            bank.observe(economics, executed_actions=actions)
            same = bank.recommend(
                question_sha256=_sha("same-runtime question"),
                runtime_profile_sha256=_sha("profile"),
            )
            other = bank.recommend(
                question_sha256=_sha("other-runtime question"),
                runtime_profile_sha256=_sha("other profile"),
            )

        assert same is not None
        self.assertEqual(same.primary_actions, actions)
        self.assertEqual(same.fallback_actions, actions)
        self.assertIsNone(other)
        with self.assertRaises(ValueError):
            InferenceActionDirective(
                question_sha256=_sha("invalid coordinate"),
                runtime_profile_sha256=_sha("profile"),
                primary_actions=("mlp_head_coordinate",),
                fallback_actions=("qwen_target",),
                draft_enabled=None,
                source_signature_sha256s=("3" * 64,),
                support=1,
                saved_qwen_forwards=0,
            )

    def test_physical_prefix_sinkhorn_is_recommended_only_for_same_runtime(
        self,
    ) -> None:
        economics = _economics(
            "physical-prefix-runtime",
            draft=False,
            pages=False,
            saved=0,
        )
        actions = ("prefix_sinkhorn", "qwen_target")
        with tempfile.TemporaryDirectory() as temporary:
            bank = InferenceActionBank(Path(temporary) / "actions")
            bank.observe(economics, executed_actions=actions)
            same = bank.recommend(
                question_sha256=_sha("same-prefix-runtime"),
                runtime_profile_sha256=_sha("profile"),
            )
            other = bank.recommend(
                question_sha256=_sha("other-prefix-runtime"),
                runtime_profile_sha256=_sha("other profile"),
            )

        assert same is not None
        self.assertEqual(same.primary_actions, actions)
        self.assertEqual(same.fallback_actions, actions)
        self.assertIsNone(other)
        with self.assertRaises(ValueError):
            InferenceActionDirective(
                question_sha256=_sha("invalid prefix action"),
                runtime_profile_sha256=_sha("profile"),
                primary_actions=("prefix_sinkhorn",),
                fallback_actions=("qwen_target",),
                draft_enabled=None,
                source_signature_sha256s=("3" * 64,),
                support=1,
                saved_qwen_forwards=0,
            )

    def test_mlp_page_coordinate_is_safe_additive_runtime_replay(self) -> None:
        economics = _economics(
            "mlp-page-coordinate-runtime",
            draft=False,
            pages=False,
            saved=0,
        )
        actions = ("mlp_page_coordinate", "qwen_target")
        with tempfile.TemporaryDirectory() as temporary:
            bank = InferenceActionBank(Path(temporary) / "actions")
            bank.observe(economics, executed_actions=actions)
            directive = bank.recommend(
                question_sha256=_sha("page-coordinate question"),
                runtime_profile_sha256=_sha("other profile"),
            )

        assert directive is not None
        self.assertEqual(directive.primary_actions, actions)
        self.assertEqual(directive.fallback_actions, actions)
        with self.assertRaises(ValueError):
            InferenceActionDirective(
                question_sha256=_sha("invalid page coordinate"),
                runtime_profile_sha256=_sha("profile"),
                primary_actions=("mlp_page_coordinate",),
                fallback_actions=("qwen_target",),
                draft_enabled=None,
                source_signature_sha256s=("3" * 64,),
                support=1,
                saved_qwen_forwards=0,
            )

    def test_reconcile_recovers_a_missed_derived_event_and_tamper_is_hard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = InferenceActionBank(Path(temporary) / "actions")
            snapshot = bank.reconcile((_economics("one"), _economics("two")))
            self.assertEqual(snapshot["requests"], 2)
            event = next(bank.events.iterdir())
            payload = bytearray(event.read_bytes())
            payload[-1] ^= 1
            event.chmod(0o600)
            event.write_bytes(payload)
            with self.assertRaises(InferenceActionBankError):
                bank.snapshot()

    def test_reconcile_never_invents_a_missing_warm_subtype(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = InferenceActionBank(Path(temporary) / "actions")
            snapshot = bank.reconcile(
                (
                    _economics(
                        "unknown-warm",
                        warm=True,
                        draft=False,
                        pages=False,
                        target=0,
                        saved=4,
                    ),
                )
            )
        self.assertEqual(snapshot["requests"], 0)


if __name__ == "__main__":
    unittest.main()
