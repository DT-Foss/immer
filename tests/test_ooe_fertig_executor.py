from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import unittest
from unittest.mock import patch

from immer.cognition.fertig import FertigSolver
from immer.runtimes.ooe.fertig_executor import (
    FERTIG_EXECUTION_AUTHORITY_SHA256,
    FERTIG_EXECUTION_RESULT_SCHEMA,
    FERTIG_EXECUTOR_SHA256,
    FertigExactExecutor,
    FertigExecutorIntegrityError,
)
from immer.runtimes.ooe.qwen_bridge import (
    ACTION_SCHEMA_SHA256,
    QwenOoeFeatureReceipt,
    feature_schema_sha256,
)
from immer.runtimes.ooe.s3_executor import TransientPromptRegistry
from immer.runtimes.qwen3_8.semantic_atlas import GraphRevision, ProbeIdentity


EXACT_QUESTION = (
    "A phone tree is used to contact families and relatives of Ali's deceased "
    "coworker. Ali decided to call 3 families. Then each family calls 3 other "
    "families, and so on. How many families will be notified during the fourth "
    "round of calls?"
)
ABSTAIN_QUESTION = (
    "Dijana and Anis live near a lake, and every weekend they go out rowing "
    "into the lake. Calculate their friendship."
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _feature(question: str, *, authority: bool = True) -> QwenOoeFeatureReceipt:
    hashes = (FERTIG_EXECUTION_AUTHORITY_SHA256,) if authority else (_sha("other"),)
    return QwenOoeFeatureReceipt(
        temporal_index=0,
        measurement_sha256=_sha("fertig-measurement"),
        model_pin_sha256=_sha("fertig-model"),
        weight_coordinate_sha256=_sha("fertig-coordinate"),
        weight_graph_revision=GraphRevision(3, _sha("fertig-weight-event")),
        atlas_graph_revision=GraphRevision(4, _sha("fertig-atlas-event")),
        probe=ProbeIdentity(
            question_sha256=_sha(question),
            token_sha256=_sha("fertig-tokens"),
            family_sha256=_sha("fertig-family"),
            label_source_sha256=_sha("fertig-label-source"),
        ),
        feature_schema_sha256=feature_schema_sha256(8),
        action_schema_sha256=ACTION_SCHEMA_SHA256,
        verifier_sha256s=hashes,
        evidence_sha256s=hashes,
        o1_surprise=0.5,
        o1_learning_progress=0.25,
        feature_sketch=(0.1,) * 8,
    )


class FertigExactExecutorTests(unittest.TestCase):
    def test_exact_certificate_executes_and_reverifies_without_prompt_leak(
        self,
    ) -> None:
        feature = _feature(EXACT_QUESTION)
        registry = TransientPromptRegistry()
        registry.bind(feature, EXACT_QUESTION)
        executor = FertigExactExecutor(FertigSolver(), registry)
        execution = executor(feature)
        execution.assert_bound(feature, "execute_fertig")
        self.assertEqual(execution.executor_sha256, FERTIG_EXECUTOR_SHA256)
        self.assertTrue(execution.quality_verified)
        self.assertEqual(execution.result["format"], FERTIG_EXECUTION_RESULT_SCHEMA)
        self.assertEqual(execution.result["certificate"]["answer"], "81")
        self.assertNotIn(EXACT_QUESTION, json.dumps(execution.result, sort_keys=True))
        self.assertTrue(executor.verify(feature, execution))
        self.assertFalse(executor.verify(feature, execution))

    def test_abstention_is_real_execution_but_not_positive_quality(self) -> None:
        feature = _feature(ABSTAIN_QUESTION)
        registry = TransientPromptRegistry()
        registry.bind(feature, ABSTAIN_QUESTION)
        executor = FertigExactExecutor(FertigSolver(), registry)
        execution = executor(feature)
        self.assertFalse(execution.quality_verified)
        self.assertEqual(execution.result["status"], "abstained")
        self.assertIsNone(execution.result["certificate"])
        self.assertFalse(executor.verify(feature, execution))

    def test_missing_authority_and_tamper_fail_before_learning(self) -> None:
        missing = _feature(EXACT_QUESTION, authority=False)
        registry = TransientPromptRegistry()
        registry.bind(missing, EXACT_QUESTION)
        executor = FertigExactExecutor(FertigSolver(), registry)
        with self.assertRaisesRegex(FertigExecutorIntegrityError, "authority"):
            executor(missing)

        feature = _feature(EXACT_QUESTION)
        registry.bind(feature, EXACT_QUESTION)
        execution = executor(feature)
        forged = replace(
            execution,
            result={**execution.result, "status": "abstained"},
            quality_verified=False,
        )
        self.assertFalse(executor.verify(feature, forged))
        executor.discard(execution)

    def test_same_question_charges_executor_and_verifier_only_once(self) -> None:
        feature = _feature(EXACT_QUESTION)
        registry = TransientPromptRegistry()
        solver = FertigSolver()
        executor = FertigExactExecutor(solver, registry)
        with patch.object(solver, "certify", wraps=solver.certify) as certify:
            for _ in range(2):
                registry.bind(feature, EXACT_QUESTION)
                execution = executor(feature)
                self.assertTrue(executor.verify(feature, execution))
        self.assertEqual(certify.call_count, 2)


if __name__ == "__main__":
    unittest.main()
