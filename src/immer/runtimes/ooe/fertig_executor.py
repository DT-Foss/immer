"""Proof-preserving FERTIG execution for verified OoE action learning."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import threading

from immer.cognition.fertig import FertigSolver

from .controller import ActionExecution
from .identity import canonical_json_bytes
from .qwen_bridge import QwenOoeFeatureReceipt
from .s3_executor import TransientPromptRegistry


FERTIG_EXECUTOR_SCHEMA = "immer-ooe-fertig-exact-executor/v1"
FERTIG_EXECUTION_RESULT_SCHEMA = "immer-ooe-fertig-exact-result/v1"
FERTIG_EXECUTOR_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "action": "execute_fertig",
            "format": FERTIG_EXECUTOR_SCHEMA,
            "solver_surface": "FertigSolver.certify",
        }
    )
).hexdigest()
FERTIG_QUALITY_VERIFIER_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "format": "immer-ooe-fertig-exact-quality-verifier/v1",
            "policy": "rerun-certify-and-compare-canonical-proof",
        }
    )
).hexdigest()
FERTIG_EXECUTION_AUTHORITY_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "action": "execute_fertig",
            "executor_sha256": FERTIG_EXECUTOR_SHA256,
            "format": "immer-ooe-fertig-execution-authority/v1",
            "quality_verifier_sha256": FERTIG_QUALITY_VERIFIER_SHA256,
        }
    )
).hexdigest()
MAX_PENDING_FERTIG_VERIFICATIONS = 1_024
MAX_CACHED_FERTIG_CERTIFICATES = 1_024


class FertigExecutorError(RuntimeError):
    """The exact FERTIG action could not satisfy its execution contract."""


class FertigExecutorIntegrityError(FertigExecutorError):
    """FERTIG execution input, proof, or replay binding changed."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _result_document(
    receipt: QwenOoeFeatureReceipt,
    certificate_bytes: bytes | None,
) -> dict[str, object]:
    certificate = None if certificate_bytes is None else json.loads(certificate_bytes)
    return {
        "certificate": certificate,
        "feature_receipt_sha256": receipt.sha256,
        "format": FERTIG_EXECUTION_RESULT_SCHEMA,
        "question_sha256": receipt.probe.question_sha256,
        "status": "certified" if certificate is not None else "abstained",
    }


def _quality_sha256(
    receipt: QwenOoeFeatureReceipt,
    result: Mapping[str, object],
) -> str:
    return _digest(
        {
            "executor_sha256": FERTIG_EXECUTOR_SHA256,
            "feature_receipt_sha256": receipt.sha256,
            "format": "immer-ooe-fertig-execution-quality/v1",
            "quality_verified": result.get("status") == "certified",
            "result_sha256": _digest(result),
            "verifier_sha256": FERTIG_QUALITY_VERIFIER_SHA256,
        }
    )


class FertigExactExecutor:
    """Execute and independently replay ``FertigSolver.certify`` one-shot."""

    action = "execute_fertig"
    executor_sha256 = FERTIG_EXECUTOR_SHA256
    authority_sha256 = FERTIG_EXECUTION_AUTHORITY_SHA256
    quality_verifier_sha256 = FERTIG_QUALITY_VERIFIER_SHA256

    def __init__(
        self,
        solver: FertigSolver,
        registry: TransientPromptRegistry,
    ) -> None:
        if type(solver) is not FertigSolver:
            raise TypeError("solver must be the production FertigSolver")
        if not isinstance(registry, TransientPromptRegistry):
            raise TypeError("registry must be a TransientPromptRegistry")
        self.solver = solver
        self.registry = registry
        self._lock = threading.RLock()
        self._verification_questions: dict[str, str] = {}
        self._execution_certificates: dict[str, bytes | None] = {}
        self._verification_certificates: dict[str, bytes | None] = {}

    def _certify_once(
        self,
        question_sha256: str,
        question: str,
        cache: dict[str, bytes | None],
    ) -> bytes | None:
        with self._lock:
            if question_sha256 in cache:
                return cache[question_sha256]
            if len(cache) >= MAX_CACHED_FERTIG_CERTIFICATES:
                raise FertigExecutorError("FERTIG certificate cache is full")
            certified = self.solver.certify(question)
            value = (
                None
                if certified is None
                else canonical_json_bytes(certified.to_dict())
            )
            cache[question_sha256] = value
            return value

    def __call__(self, receipt: QwenOoeFeatureReceipt) -> ActionExecution:
        if not isinstance(receipt, QwenOoeFeatureReceipt):
            raise TypeError("receipt must be a QwenOoeFeatureReceipt")
        if (
            self.authority_sha256 not in receipt.verifier_sha256s
            or self.authority_sha256 not in receipt.evidence_sha256s
        ):
            raise FertigExecutorIntegrityError(
                "feature receipt lacks the exact FERTIG execution authority"
            )
        question = self.registry.consume(receipt)
        certificate_bytes = self._certify_once(
            receipt.probe.question_sha256,
            question,
            self._execution_certificates,
        )
        result = _result_document(receipt, certificate_bytes)
        if question in json.dumps(result, allow_nan=False, sort_keys=True):
            raise FertigExecutorIntegrityError("FERTIG result leaked the raw question")
        execution = ActionExecution(
            feature_receipt_sha256=receipt.sha256,
            action=self.action,
            executor_sha256=self.executor_sha256,
            verifier_sha256=self.authority_sha256,
            evidence_sha256=self.authority_sha256,
            quality_sha256=_quality_sha256(receipt, result),
            result=result,
            quality_verified=certificate_bytes is not None,
            qwen_forwards=0,
            teacher_baseline_qwen_forwards=0,
        )
        with self._lock:
            if execution.sha256 in self._verification_questions:
                raise FertigExecutorIntegrityError(
                    "FERTIG execution is already pending verification"
                )
            if len(self._verification_questions) >= (MAX_PENDING_FERTIG_VERIFICATIONS):
                raise FertigExecutorError("FERTIG verification inventory is full")
            self._verification_questions[execution.sha256] = question
        return execution

    def verify(
        self,
        receipt: QwenOoeFeatureReceipt,
        execution: ActionExecution,
    ) -> bool:
        """Rerun the exact proof surface and consume its transient question."""

        if not isinstance(receipt, QwenOoeFeatureReceipt) or not isinstance(
            execution, ActionExecution
        ):
            return False
        try:
            execution.assert_bound(receipt, self.action)
        except (TypeError, ValueError, RuntimeError):
            return False
        with self._lock:
            question = self._verification_questions.pop(execution.sha256, None)
        if question is None:
            return False
        try:
            certificate_bytes = self._certify_once(
                receipt.probe.question_sha256,
                question,
                self._verification_certificates,
            )
            result = _result_document(receipt, certificate_bytes)
            expected = ActionExecution(
                feature_receipt_sha256=receipt.sha256,
                action=self.action,
                executor_sha256=self.executor_sha256,
                verifier_sha256=self.authority_sha256,
                evidence_sha256=self.authority_sha256,
                quality_sha256=_quality_sha256(receipt, result),
                result=result,
                quality_verified=certificate_bytes is not None,
                qwen_forwards=0,
                teacher_baseline_qwen_forwards=0,
            )
        except (TypeError, ValueError, RuntimeError):
            return False
        return execution == expected and execution.quality_verified

    def discard(self, execution: ActionExecution) -> None:
        if not isinstance(execution, ActionExecution):
            raise TypeError("execution must be an ActionExecution")
        with self._lock:
            self._verification_questions.pop(execution.sha256, None)


__all__ = [
    "FERTIG_EXECUTION_AUTHORITY_SHA256",
    "FERTIG_EXECUTION_RESULT_SCHEMA",
    "FERTIG_EXECUTOR_SCHEMA",
    "FERTIG_EXECUTOR_SHA256",
    "FERTIG_QUALITY_VERIFIER_SHA256",
    "FertigExactExecutor",
    "FertigExecutorError",
    "FertigExecutorIntegrityError",
]
