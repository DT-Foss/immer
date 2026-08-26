from __future__ import annotations

from dataclasses import dataclass
import hashlib
import threading
import unittest

from immer.contracts import ExecutionStatus, Request, Result
from immer.capabilities.s3_runtime import S3Arithmetic
from immer.runtimes.ooe.chat import result_from_document
from immer.runtimes.ooe.qwen_bridge import (
    ACTION_SCHEMA_SHA256,
    QwenOoeFeatureReceipt,
    feature_schema_sha256,
)
from immer.runtimes.ooe.s3_executor import (
    RegisteringFeatureProvider,
    S3ArithmeticExecutor,
    S3ExecutorIntegrityError,
    TransientPromptRegistry,
)
from immer.runtimes.qwen3_8.semantic_atlas import GraphRevision, ProbeIdentity


PROVENANCE = "9" * 64


def _feature(question: str, *, temporal: int = 0) -> QwenOoeFeatureReceipt:
    question_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
    return QwenOoeFeatureReceipt(
        temporal_index=temporal,
        measurement_sha256=hashlib.sha256(f"measurement:{temporal}".encode()).hexdigest(),
        model_pin_sha256="1" * 64,
        weight_coordinate_sha256="2" * 64,
        weight_graph_revision=GraphRevision(3, "3" * 64),
        atlas_graph_revision=GraphRevision(4, "4" * 64),
        probe=ProbeIdentity(
            question_sha256=question_sha256,
            token_sha256="5" * 64,
            family_sha256="6" * 64,
            label_source_sha256="7" * 64,
        ),
        feature_schema_sha256=feature_schema_sha256(8),
        action_schema_sha256=ACTION_SCHEMA_SHA256,
        verifier_sha256s=(PROVENANCE,),
        evidence_sha256s=(PROVENANCE,),
        o1_surprise=0.1,
        o1_learning_progress=0.2,
        feature_sketch=(0.1,) * 8,
    )


@dataclass
class _FakeS3:
    status: ExecutionStatus = ExecutionStatus.OK
    leak_prompt: bool = False

    name = "fake.s3"
    capabilities = frozenset({"exact_math"})

    def handle(self, request: Request) -> Result:
        evidence = {
            "artifact_sha256": "a" * 64,
            "crystal_verified": True,
            "no_training": True,
            "route": "ARITH",
        }
        if self.leak_prompt:
            evidence["raw"] = request.payload
        return Result(
            self.status,
            self.name,
            output="four" if self.status is ExecutionStatus.OK else None,
            reason=None if self.status is ExecutionStatus.OK else "outside grammar",
            evidence=evidence,
        )


class S3ArithmeticExecutorTests(unittest.TestCase):
    def test_production_s3_organ_executes_through_ooe_adapter(self) -> None:
        question = "two plus two is"
        receipt = _feature(question)
        registry = TransientPromptRegistry()
        registry.bind(receipt, question)
        execution = S3ArithmeticExecutor(
            S3Arithmetic(),
            registry,
            provenance_sha256=PROVENANCE,
        )(receipt)
        result = result_from_document(execution.result)
        self.assertTrue(execution.quality_verified)
        self.assertEqual(result.output, "four")
        self.assertEqual(result.component, "s3.ship-v6.arithmetic")
        self.assertTrue(result.evidence["crystal_verified"])

    def test_real_organ_result_replaces_cached_qwen_document(self) -> None:
        question = "two plus two is"
        receipt = _feature(question)
        registry = TransientPromptRegistry()
        registry.bind(receipt, question)
        execution = S3ArithmeticExecutor(
            _FakeS3(),
            registry,
            provenance_sha256=PROVENANCE,
        )(receipt)
        execution.assert_bound(receipt, "mount_organ")
        result = result_from_document(execution.result)
        self.assertTrue(result.ok)
        self.assertEqual(result.output, "four")
        self.assertEqual(result.component, "fake.s3")
        self.assertTrue(result.evidence["crystal_verified"])
        self.assertEqual(execution.qwen_forwards, 0)
        self.assertEqual(execution.teacher_baseline_qwen_forwards, 0)
        self.assertEqual(execution.saved_qwen_forwards, 0)
        self.assertNotIn(question, str(execution.result))

    def test_registering_provider_keeps_raw_question_transient(self) -> None:
        question = "three plus four is"
        receipt = _feature(question)
        registry = TransientPromptRegistry()

        def provider(value: str, _metadata: object) -> QwenOoeFeatureReceipt:
            self.assertEqual(value, question)
            return receipt

        wrapped = RegisteringFeatureProvider(provider, registry)
        self.assertIs(wrapped(question, {}), receipt)
        self.assertEqual(registry.consume(receipt), question)
        with self.assertRaisesRegex(S3ExecutorIntegrityError, "no matching"):
            registry.consume(receipt)

    def test_prompt_hash_mismatch_and_stale_receipt_fail_closed(self) -> None:
        registry = TransientPromptRegistry()
        receipt = _feature("two plus two is")
        with self.assertRaisesRegex(S3ExecutorIntegrityError, "does not match"):
            registry.bind(receipt, "two plus three is")
        registry.bind(receipt, "two plus two is")
        other = _feature("two plus two is", temporal=1)
        with self.assertRaisesRegex(S3ExecutorIntegrityError, "no matching"):
            registry.consume(other)

    def test_missing_joint_provenance_fails_before_component_call(self) -> None:
        question = "two plus two is"
        receipt = _feature(question)
        registry = TransientPromptRegistry()
        registry.bind(receipt, question)
        executor = S3ArithmeticExecutor(
            _FakeS3(),
            registry,
            provenance_sha256="8" * 64,
        )
        with self.assertRaisesRegex(S3ExecutorIntegrityError, "provenance"):
            executor(receipt)

    def test_abstention_is_an_executed_but_unverified_warm_result(self) -> None:
        question = "not arithmetic"
        receipt = _feature(question)
        registry = TransientPromptRegistry()
        registry.bind(receipt, question)
        execution = S3ArithmeticExecutor(
            _FakeS3(status=ExecutionStatus.ABSTAINED),
            registry,
            provenance_sha256=PROVENANCE,
        )(receipt)
        self.assertFalse(execution.quality_verified)
        self.assertFalse(result_from_document(execution.result).ok)
        self.assertEqual(execution.saved_qwen_forwards, 0)

    def test_component_cannot_leak_transient_prompt_into_receipt(self) -> None:
        question = "two plus two is"
        receipt = _feature(question)
        registry = TransientPromptRegistry()
        registry.bind(receipt, question)
        executor = S3ArithmeticExecutor(
            _FakeS3(leak_prompt=True),
            registry,
            provenance_sha256=PROVENANCE,
        )
        with self.assertRaisesRegex(S3ExecutorIntegrityError, "leaked"):
            executor(receipt)

    def test_thread_local_registry_does_not_cross_prompts(self) -> None:
        registry = TransientPromptRegistry()
        failures: list[BaseException] = []

        def worker(index: int) -> None:
            try:
                question = f"question {index}"
                receipt = _feature(question, temporal=index)
                registry.bind(receipt, question)
                self.assertEqual(registry.consume(receipt), question)
            except BaseException as exc:  # pragma: no cover - surfaced below
                failures.append(exc)

        threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failures:
            raise failures[0]

    def test_same_thread_interleaving_is_receipt_addressed(self) -> None:
        registry = TransientPromptRegistry()
        first_question = "question one"
        second_question = "question two"
        first = _feature(first_question, temporal=1)
        second = _feature(second_question, temporal=2)
        registry.bind(first, first_question)
        registry.bind(second, second_question)
        self.assertEqual(registry.consume(first), first_question)
        self.assertEqual(registry.consume(second), second_question)

    def test_s3_cannot_receive_caller_supplied_qwen_savings_credit(self) -> None:
        registry = TransientPromptRegistry()
        with self.assertRaises(TypeError):
            S3ArithmeticExecutor(
                _FakeS3(),
                registry,
                provenance_sha256=PROVENANCE,
                teacher_baseline_qwen_forwards=1_000_000,
            )


if __name__ == "__main__":
    unittest.main()
