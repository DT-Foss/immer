"""Transient prompt bridge from OoE routing to the exact S3 OrganBank."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import threading
from typing import Any

from immer.contracts import Component, Request, Result

from .chat import FeatureProvider, result_to_document
from .controller import ActionExecution
from .identity import canonical_json_bytes, require_sha256
from .qwen_bridge import QwenOoeFeatureReceipt


S3_EXECUTOR_SCHEMA = "immer-ooe-s3-arithmetic-executor/v1"
S3_EXECUTOR_SHA256 = hashlib.sha256(S3_EXECUTOR_SCHEMA.encode("utf-8")).hexdigest()
MAX_TRANSIENT_PROMPTS = 64


class S3ExecutorIntegrityError(RuntimeError):
    pass


class TransientPromptRegistry:
    """Thread-local one-shot raw prompt binding; nothing is persisted."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _bindings(self) -> dict[str, str]:
        bindings = getattr(self._local, "bindings", None)
        if bindings is None:
            bindings = {}
            self._local.bindings = bindings
        return bindings

    def bind(self, receipt: QwenOoeFeatureReceipt, question: str) -> None:
        if not isinstance(receipt, QwenOoeFeatureReceipt):
            raise TypeError("receipt must be a QwenOoeFeatureReceipt")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty text")
        question_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
        if question_sha256 != receipt.probe.question_sha256:
            raise S3ExecutorIntegrityError(
                "raw question does not match the feature receipt"
            )
        bindings = self._bindings()
        previous = bindings.get(receipt.sha256)
        if previous is not None and previous != question:
            raise S3ExecutorIntegrityError(
                "feature receipt is already bound to another transient prompt"
            )
        if previous is None and len(bindings) >= MAX_TRANSIENT_PROMPTS:
            raise S3ExecutorIntegrityError("transient prompt registry is full")
        bindings[receipt.sha256] = question

    def consume(self, receipt: QwenOoeFeatureReceipt) -> str:
        if not isinstance(receipt, QwenOoeFeatureReceipt):
            raise TypeError("receipt must be a QwenOoeFeatureReceipt")
        question = self._bindings().pop(receipt.sha256, None)
        if not isinstance(question, str):
            raise S3ExecutorIntegrityError(
                "no matching transient prompt is bound to this feature receipt"
            )
        if hashlib.sha256(question.encode("utf-8")).hexdigest() != receipt.probe.question_sha256:
            raise S3ExecutorIntegrityError("transient prompt binding changed")
        return question

    def discard(self, receipt: QwenOoeFeatureReceipt) -> None:
        if not isinstance(receipt, QwenOoeFeatureReceipt):
            raise TypeError("receipt must be a QwenOoeFeatureReceipt")
        self._bindings().pop(receipt.sha256, None)

    def clear(self) -> None:
        self._bindings().clear()


class RegisteringFeatureProvider:
    """Wrap a hash-only feature provider and retain its raw prompt for one call."""

    def __init__(
        self,
        provider: FeatureProvider,
        registry: TransientPromptRegistry,
    ) -> None:
        if not callable(provider):
            raise TypeError("provider must be callable")
        if not isinstance(registry, TransientPromptRegistry):
            raise TypeError("registry must be a TransientPromptRegistry")
        self.provider = provider
        self.registry = registry

    def __call__(
        self,
        question: str,
        metadata: Mapping[str, Any],
    ) -> QwenOoeFeatureReceipt | None:
        receipt = self.provider(question, metadata)
        if receipt is None:
            return None
        if not isinstance(receipt, QwenOoeFeatureReceipt):
            raise TypeError(
                "wrapped provider must return QwenOoeFeatureReceipt or None"
            )
        self.registry.bind(receipt, question)
        return receipt


class S3ArithmeticExecutor:
    """Execute an actual cold-loaded S3 organ after a warm Markov route."""

    def __init__(
        self,
        component: Component,
        registry: TransientPromptRegistry,
        *,
        provenance_sha256: str,
    ) -> None:
        if not isinstance(component, Component):
            raise TypeError("component must implement the Component protocol")
        if "exact_math" not in component.capabilities:
            raise ValueError("component must advertise exact_math")
        if not isinstance(registry, TransientPromptRegistry):
            raise TypeError("registry must be a TransientPromptRegistry")
        self.component = component
        self.registry = registry
        self.provenance_sha256 = require_sha256(
            provenance_sha256,
            field="provenance_sha256",
        )

    @staticmethod
    def _quality_verified(result: Result) -> bool:
        evidence = result.evidence
        return bool(
            result.ok
            and isinstance(result.output, str)
            and result.output.strip()
            and isinstance(evidence, Mapping)
            and evidence.get("crystal_verified") is True
            and evidence.get("no_training") is True
            and isinstance(evidence.get("artifact_sha256"), str)
        )

    def __call__(self, receipt: QwenOoeFeatureReceipt) -> ActionExecution:
        if not isinstance(receipt, QwenOoeFeatureReceipt):
            raise TypeError("receipt must be a QwenOoeFeatureReceipt")
        if (
            self.provenance_sha256 not in receipt.verifier_sha256s
            or self.provenance_sha256 not in receipt.evidence_sha256s
        ):
            raise S3ExecutorIntegrityError(
                "feature receipt lacks the exact S3 provenance binding"
            )
        question = self.registry.consume(receipt)
        result = self.component.handle(
            Request(
                "exact_math",
                question,
                metadata={
                    "ooe_feature_receipt_sha256": receipt.sha256,
                    "question_sha256": receipt.probe.question_sha256,
                },
            )
        )
        if not isinstance(result, Result):
            raise S3ExecutorIntegrityError("S3 component returned an invalid result")
        document = result_to_document(result)
        quality = self._quality_verified(result)
        quality_sha256 = hashlib.sha256(
            canonical_json_bytes(
                {
                    "executor": S3_EXECUTOR_SCHEMA,
                    "feature_receipt_sha256": receipt.sha256,
                    "provenance_sha256": self.provenance_sha256,
                    "question_sha256": receipt.probe.question_sha256,
                    "result_sha256": hashlib.sha256(
                        canonical_json_bytes(document)
                    ).hexdigest(),
                    "quality_verified": quality,
                }
            )
        ).hexdigest()
        # Prove the persisted document is prompt-free.  S3 evidence is allowed
        # to carry hashes and route metadata, never the transient input text.
        encoded = json.dumps(document, sort_keys=True, ensure_ascii=False)
        if question in encoded:
            raise S3ExecutorIntegrityError("S3 result document leaked the raw prompt")
        return ActionExecution(
            feature_receipt_sha256=receipt.sha256,
            action="mount_organ",
            executor_sha256=S3_EXECUTOR_SHA256,
            verifier_sha256=self.provenance_sha256,
            evidence_sha256=self.provenance_sha256,
            quality_sha256=quality_sha256,
            result=document,
            quality_verified=quality,
            qwen_forwards=0,
            # S3 has no cold Qwen-generation receipt.  It executes zero Qwen
            # forwards and receives zero savings credit until such a baseline
            # is authenticated by a separate teacher contract.
            teacher_baseline_qwen_forwards=0,
        )


__all__ = [
    "RegisteringFeatureProvider",
    "S3ArithmeticExecutor",
    "S3_EXECUTOR_SCHEMA",
    "S3_EXECUTOR_SHA256",
    "S3ExecutorIntegrityError",
    "TransientPromptRegistry",
    "MAX_TRANSIENT_PROMPTS",
]
