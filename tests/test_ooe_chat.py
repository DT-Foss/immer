from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from immer.composition import CompositionRoot
from immer.cognition.fertig.adapter import CandidateVerificationStatus
from immer.cognition.qwen_fertig_chat import QwenFertigChat
from immer.contracts import ExecutionStatus, Request, Result
from immer.runtimes.ooe.chat import (
    ControllerPromptFeatureProvider,
    OoeChatHook,
    result_from_document,
    result_to_document,
)
from immer.runtimes.ooe.controller import (
    ActionExecution,
    ControllerConfig,
    OoeController,
)
from immer.runtimes.ooe.crystal import CrystalStore

from test_ooe_qwen_bridge_controller import (
    _ATLAS_GRAPH,
    _EXECUTOR,
    _PIN,
    _VERIFIER,
    _WEIGHT_GRAPH,
    _feature,
    _hash,
    _transition,
)
from test_qwen_fertig_chat import (
    MATH_QUESTION,
    _Qwen,
    _certificate,
    _patched_solver,
    _receipt,
    _verification,
)


class OoeChatHookTests(unittest.TestCase):
    def _controller(
        self,
        root: Path,
        *,
        result_document: dict | None = None,
    ) -> tuple[OoeController, object]:
        def executor(receipt):
            document = result_document or result_to_document(
                Result(
                    ExecutionStatus.OK,
                    "ooe.fixture",
                    output="500",
                    evidence={"verified": True},
                )
            )
            return ActionExecution(
                feature_receipt_sha256=receipt.sha256,
                action="probe_coordinate",
                executor_sha256=_EXECUTOR,
                verifier_sha256=_VERIFIER,
                evidence_sha256=receipt.evidence_sha256s[0],
                quality_sha256=_hash(f"chat-quality:{receipt.sha256}"),
                result=document,
                quality_verified=True,
                qwen_forwards=0,
                teacher_baseline_qwen_forwards=1,
            )

        controller = OoeController(
            model_pin_sha256=_PIN.sha256,
            weight_graph_revision_sha256=_WEIGHT_GRAPH.sha256,
            atlas_graph_revision=_ATLAS_GRAPH,
            crystal_store=CrystalStore(root),
            config=ControllerConfig(
                replicas=4,
                replica_fanout=4,
                min_coverage_per_source=1,
                min_promoted_sources=1,
                router_radius=1.0,
                router_min_margin=0.0,
                token_min_confidence=1e-9,
                reservoir_size=8,
            ),
            action_executors={"probe_coordinate": executor},
        )
        training, _ = _feature(0, site=0, source=4)
        controller.ingest_teacher(
            training,
            _transition(training, "qwen_fallback", "probe_coordinate"),
        )
        coverage = controller.coverage_receipt(training.site_identity.sha256)
        controller.promote(
            training.site_identity.sha256,
            coverage_sha256=coverage.sha256,
            verifier_sha256s=coverage.verifier_sha256s,
            expected_store_generation=controller.crystal_store.manifest().generation,
        )
        warm, _ = _feature(1, site=0, source=4)
        return controller, warm

    def test_result_document_roundtrip_and_tamper_rejection(self) -> None:
        result = Result(
            ExecutionStatus.OK,
            "fixture",
            output="answer",
            evidence={"nested": {"value": 1}},
        )
        document = result_to_document(result)
        self.assertEqual(result_from_document(document), result)
        tampered = {**document, "sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            result_from_document(tampered)

    def test_controller_prompt_provider_uses_sealed_question_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller, warm = self._controller(Path(tmp))
            provider = ControllerPromptFeatureProvider(controller)
            resolved = provider("question-0-4", {})
            self.assertIsNotNone(resolved)
            assert resolved is not None
            self.assertEqual(resolved.site_identity, warm.site_identity)
            self.assertEqual(resolved.probe, warm.probe)
            self.assertIsNone(provider("unknown question", {}))
            self.assertEqual(
                provider(
                    "question-0-4",
                    {"qwen_token_sha256": warm.probe.token_sha256},
                ),
                resolved,
            )

    def test_verified_warm_result_bypasses_qwen_and_still_passes_fertig(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller, warm = self._controller(Path(tmp))
            hook = OoeChatHook(
                controller=controller,
                feature_provider=lambda _question, _metadata: warm,
                quality_verifier=lambda _receipt, _execution: True,
            )
            qwen = _Qwen(RuntimeError("Qwen must not run on a warm hit"))
            with _patched_solver(
                None,
                _verification(
                    CandidateVerificationStatus.VERIFIED,
                    candidate="500",
                    expected="500",
                ),
            ) as (solver, _certify, _verify):
                result = QwenFertigChat(qwen, solver, ooe_hook=hook).handle(
                    Request("chat", MATH_QUESTION)
                )

            self.assertTrue(result.ok)
            self.assertEqual(result.output, "500")
            self.assertEqual(qwen.requests, [])
            receipt = _receipt(result)
            self.assertEqual(receipt["route"], "ooe_verified")
            self.assertEqual(receipt["ooe"]["warm"]["status"], "hit")
            self.assertEqual(controller.metrics.saved_qwen_forwards, 1)
            self.assertEqual(controller.metrics.teacher_calls, 0)

    def test_miss_calls_qwen_once_and_observes_final_result(self) -> None:
        observations = []
        with tempfile.TemporaryDirectory() as tmp:
            controller, _warm = self._controller(Path(tmp))
            hook = OoeChatHook(
                controller=controller,
                feature_provider=lambda _question, _metadata: None,
                quality_verifier=lambda _receipt, _execution: True,
                cold_observer=lambda question, metadata, qwen, final: (
                    observations.append((question, dict(metadata), qwen, final))
                ),
            )
            qwen = _Qwen(Result(ExecutionStatus.OK, "qwen.fixture", output="500"))
            with _patched_solver(
                None,
                _verification(
                    CandidateVerificationStatus.VERIFIED,
                    candidate="500",
                    expected="500",
                ),
            ) as (solver, _certify, _verify):
                result = QwenFertigChat(qwen, solver, ooe_hook=hook).handle(
                    Request("chat", MATH_QUESTION, {"conversation_id": "test"})
                )

            self.assertEqual(len(qwen.requests), 1)
            self.assertEqual(len(observations), 1)
            self.assertEqual(observations[0][0], MATH_QUESTION)
            receipt = _receipt(result)
            self.assertEqual(receipt["route"], "qwen_verified")
            self.assertEqual(receipt["ooe"]["warm"]["status"], "no-feature")
            self.assertEqual(receipt["ooe"]["cold_observer"]["status"], "observed")

    def test_invalid_warm_payload_falls_through_without_false_savings(self) -> None:
        valid = result_to_document(
            Result(ExecutionStatus.OK, "ooe.fixture", output="500")
        )
        tampered = {**valid, "sha256": "f" * 64}
        with tempfile.TemporaryDirectory() as tmp:
            controller, warm = self._controller(
                Path(tmp),
                result_document=tampered,
            )
            hook = OoeChatHook(
                controller=controller,
                feature_provider=lambda _question, _metadata: warm,
                quality_verifier=lambda _receipt, _execution: True,
            )
            qwen = _Qwen(Result(ExecutionStatus.OK, "qwen.fixture", output="500"))
            with _patched_solver(
                None,
                _verification(
                    CandidateVerificationStatus.VERIFIED,
                    candidate="500",
                    expected="500",
                ),
            ) as (solver, _certify, _verify):
                result = QwenFertigChat(qwen, solver, ooe_hook=hook).handle(
                    Request("chat", MATH_QUESTION)
                )

            self.assertTrue(result.ok)
            self.assertEqual(len(qwen.requests), 1)
            self.assertEqual(controller.metrics.saved_qwen_forwards, 0)
            self.assertEqual(controller.metrics.quality_failures, 1)

    def test_fertig_mismatch_rejects_pending_warm_savings(self) -> None:
        wrong = result_to_document(
            Result(ExecutionStatus.OK, "ooe.fixture", output="999")
        )
        with tempfile.TemporaryDirectory() as tmp:
            controller, warm = self._controller(Path(tmp), result_document=wrong)
            hook = OoeChatHook(
                controller=controller,
                feature_provider=lambda _question, _metadata: warm,
                quality_verifier=lambda _receipt, _execution: True,
            )
            qwen = _Qwen(RuntimeError("warm result must reach FERTIG directly"))
            with _patched_solver(
                None,
                _verification(
                    CandidateVerificationStatus.MISMATCH,
                    candidate="999",
                    expected="500",
                ),
            ) as (solver, _certify, _verify):
                result = QwenFertigChat(qwen, solver, ooe_hook=hook).handle(
                    Request("chat", MATH_QUESTION)
                )

            self.assertEqual(result.output, "500")
            self.assertEqual(qwen.requests, [])
            receipt = _receipt(result)
            self.assertEqual(receipt["route"], "ooe_fertig_mismatch_override")
            self.assertEqual(receipt["ooe"]["accounting"]["disposition"], "rejected")
            self.assertEqual(controller.metrics.saved_qwen_forwards, 0)
            self.assertEqual(controller.metrics.crystal_executions, 0)
            self.assertEqual(controller.metrics.verified_results, 0)
            self.assertEqual(controller.metrics.quality_failures, 1)

    def test_tampered_active_crystal_is_hard_error_not_qwen_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            controller, warm = self._controller(root)
            entry = controller.crystal_store.manifest().entries[0]
            payload_path = root / "objects" / f"{entry.payload_sha256}.crystal"
            payload = bytearray(payload_path.read_bytes())
            payload[-1] ^= 1
            payload_path.chmod(0o600)
            payload_path.write_bytes(payload)

            hook = OoeChatHook(
                controller=controller,
                feature_provider=lambda _question, _metadata: warm,
                quality_verifier=lambda _receipt, _execution: True,
            )
            qwen = _Qwen(RuntimeError("tamper must not fall through to Qwen"))
            with _patched_solver(
                None,
                _verification(
                    CandidateVerificationStatus.ABSTAINED,
                    candidate=None,
                    expected=None,
                ),
            ) as (solver, _certify, _verify):
                result = QwenFertigChat(qwen, solver, ooe_hook=hook).handle(
                    Request("chat", MATH_QUESTION)
                )

            self.assertEqual(result.status, ExecutionStatus.ERROR)
            self.assertEqual(qwen.requests, [])
            receipt = _receipt(result)
            self.assertEqual(receipt["route"], "ooe_integrity_error")
            self.assertEqual(receipt["ooe"]["warm"]["status"], "integrity-error")

    def test_fertig_exact_first_refusal_runs_before_ooe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller, _warm = self._controller(Path(tmp))
            hook = OoeChatHook(
                controller=controller,
                feature_provider=lambda _question, _metadata: (_ for _ in ()).throw(
                    AssertionError("OoE must not run before an exact certificate")
                ),
                quality_verifier=lambda _receipt, _execution: True,
            )
            qwen = _Qwen(RuntimeError("Qwen must not run"))
            with _patched_solver(
                _certificate(),
                _verification(
                    CandidateVerificationStatus.ABSTAINED,
                    candidate=None,
                    expected=None,
                ),
            ) as (solver, _certify, _verify):
                result = QwenFertigChat(qwen, solver, ooe_hook=hook).handle(
                    Request("chat", MATH_QUESTION)
                )

            self.assertEqual(result.output, "500")
            self.assertEqual(qwen.requests, [])
            self.assertEqual(_receipt(result)["route"], "fertig_exact_short_circuit")

    def test_composition_root_wires_hook_only_through_qwen_fertig(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            controller, _warm = self._controller(Path(tmp))
            hook = OoeChatHook(
                controller=controller,
                feature_provider=lambda _question, _metadata: None,
                quality_verifier=lambda _receipt, _execution: True,
            )
            raw_qwen = _Qwen(Result(ExecutionStatus.OK, "qwen.fixture", output="x"))
            anchor_root = Path(tmp) / "anchors"
            with mock.patch(
                "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                return_value=raw_qwen,
            ) as qwen_factory:
                root = CompositionRoot.build(
                    qwen38_causal_bundle="local.causal",
                    qwen38_tokenizer="tokenizer.json",
                    qwen38_ooe_hook=hook,
                    qwen38_anchor_cache=anchor_root,
                )
            self.assertIs(root.ooe_chat, hook)
            self.assertIs(
                root.qwen_anchor_cache,
                qwen_factory.call_args.kwargs["anchor_cache"],
            )
            self.assertEqual(root.qwen_anchor_cache.root, anchor_root.absolute())
            self.assertIsInstance(root.general_chat, QwenFertigChat)
            assert isinstance(root.general_chat, QwenFertigChat)
            self.assertIs(root.general_chat.ooe_hook, hook)

            with self.assertRaisesRegex(ValueError, "requires the local Qwen"):
                CompositionRoot.build(qwen38_ooe_hook=hook)
            with self.assertRaisesRegex(ValueError, "requires the Qwen-FERTIG"):
                CompositionRoot.build(
                    qwen38_causal_bundle="local.causal",
                    qwen38_tokenizer="tokenizer.json",
                    qwen38_raw_chat=True,
                    qwen38_ooe_hook=hook,
                )


if __name__ == "__main__":
    unittest.main()
