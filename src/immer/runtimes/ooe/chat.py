"""Verified warm-OoE hook for the existing Qwen/FERTIG chat wrapper."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

from immer.contracts import ExecutionStatus, Result

from .controller import (
    ActionExecution,
    ExecutionQualityVerifier,
    OoeController,
    OoeControllerIntegrityError,
    OoeControllerStaleError,
    OoeDecision,
    WarmAccountingReceipt,
)
from .crystal import CrystalStoreError
from .identity import canonical_json_bytes, require_sha256
from .qwen_bridge import (
    OoeAction,
    QwenOoeBridgeIntegrityError,
    QwenOoeBridgeStaleError,
    QwenOoeFeatureReceipt,
    validate_action,
)


CHAT_RESULT_SCHEMA = "immer-ooe-chat-result/v1"
CHAT_OBSERVATION_SCHEMA = "immer-ooe-chat-observation/v1"


FeatureProvider = Callable[[str, Mapping[str, Any]], QwenOoeFeatureReceipt | None]
ColdObserver = Callable[
    [str, Mapping[str, Any], Result, Result], Mapping[str, Any] | None
]
ControllerRestorer = Callable[[], OoeController]


class OoeChatIntegrityError(RuntimeError):
    """A warm Crystal, execution receipt, or feature binding failed integrity."""


class ControllerPromptFeatureProvider:
    """Resolve an exact known prompt from the controller's sealed O1 history."""

    def __init__(self, controller: OoeController) -> None:
        if not isinstance(controller, OoeController):
            raise TypeError("controller must be an OoeController")
        self.controller = controller

    def __call__(
        self,
        question: str,
        metadata: Mapping[str, Any],
    ) -> QwenOoeFeatureReceipt | None:
        question_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
        raw_token = metadata.get("qwen_token_sha256")
        token_sha256 = (
            None
            if raw_token is None
            else require_sha256(raw_token, field="qwen_token_sha256")
        )
        matches = self.controller.feature_receipts_for_prompt(
            question_sha256,
            token_sha256,
        )
        raw_site = metadata.get("ooe_site_identity_sha256")
        if raw_site is not None:
            site_sha256 = require_sha256(
                raw_site,
                field="ooe_site_identity_sha256",
            )
            matches = tuple(
                receipt
                for receipt in matches
                if receipt.site_identity.sha256 == site_sha256
            )
        if not matches:
            return None
        if len({receipt.site_identity.sha256 for receipt in matches}) != 1:
            # O1/Atlas measured more than one weight house for this prompt.  A
            # caller must name the intended house; never pick one silently.
            return None
        return matches[-1]


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def result_to_document(result: Result) -> dict[str, Any]:
    """Seal one canonical runtime result for an OoE action executor."""

    if not isinstance(result, Result):
        raise TypeError("result must be a Result")
    if not isinstance(result.status, ExecutionStatus):
        raise TypeError("result status must be an ExecutionStatus")
    if not isinstance(result.component, str) or not result.component.strip():
        raise ValueError("result component must be non-empty text")
    if result.reason is not None and not isinstance(result.reason, str):
        raise TypeError("result reason must be text or None")
    if not isinstance(result.evidence, Mapping):
        raise TypeError("result evidence must be a mapping")
    body = {
        "component": result.component,
        "evidence": dict(result.evidence),
        "output": result.output,
        "reason": result.reason,
        "status": result.status.value,
    }
    # Validate and normalize before sealing so the ActionExecution's canonical
    # result hash and the reconstructed Result see exactly the same value.
    normalized = json.loads(canonical_json_bytes(body))
    return {
        "body": normalized,
        "schema": CHAT_RESULT_SCHEMA,
        "sha256": _digest(normalized),
    }


def result_from_document(document: object) -> Result:
    """Authenticate and reconstruct one warm executor result."""

    if not isinstance(document, Mapping) or set(document) != {
        "body",
        "schema",
        "sha256",
    }:
        raise ValueError("warm chat result envelope is invalid")
    if document.get("schema") != CHAT_RESULT_SCHEMA:
        raise ValueError("warm chat result schema is invalid")
    body = document.get("body")
    if not isinstance(body, Mapping) or set(body) != {
        "component",
        "evidence",
        "output",
        "reason",
        "status",
    }:
        raise ValueError("warm chat result body is invalid")
    claimed = require_sha256(document.get("sha256"), field="sha256")
    if claimed != _digest(body):
        raise ValueError("warm chat result SHA-256 mismatch")
    try:
        status = ExecutionStatus(body["status"])
    except (TypeError, ValueError) as exc:
        raise ValueError("warm chat result status is invalid") from exc
    component = body["component"]
    reason = body["reason"]
    evidence = body["evidence"]
    if not isinstance(component, str) or not component.strip():
        raise ValueError("warm chat result component is invalid")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("warm chat result reason is invalid")
    if not isinstance(evidence, Mapping):
        raise ValueError("warm chat result evidence is invalid")
    result = Result(
        status,
        component,
        output=body["output"],
        reason=reason,
        evidence=dict(evidence),
    )
    if result_to_document(result) != dict(document):
        raise ValueError("warm chat result failed canonical reconstruction")
    return result


@dataclass(frozen=True, slots=True)
class OoeChatAttempt:
    result: Result | None
    evidence: Mapping[str, Any]
    decision: OoeDecision | None = None
    _owner: Any | None = field(default=None, repr=False, compare=False)
    _settler: Any | None = field(default=None, repr=False, compare=False)
    _abstention_authorized: bool = field(
        default=False,
        repr=False,
        compare=False,
    )

    @property
    def hit(self) -> bool:
        return self.result is not None


class OoeChatHook:
    """Connect a promoted OoE Crystal to Qwen/FERTIG without guessing."""

    def __init__(
        self,
        *,
        controller: OoeController,
        feature_provider: FeatureProvider,
        quality_verifier: ExecutionQualityVerifier,
        source_action: OoeAction | str = "qwen_fallback",
        cold_observer: ColdObserver | None = None,
        direct_provider: Any | None = None,
        snapshot_name: str | None = None,
        snapshot_restorer: ControllerRestorer | None = None,
        commit_on_fertig_abstention: bool = False,
    ) -> None:
        if not isinstance(controller, OoeController):
            raise TypeError("controller must be an OoeController")
        if not callable(feature_provider):
            raise TypeError("feature_provider must be callable")
        if not callable(quality_verifier):
            raise TypeError("quality_verifier must be callable")
        if cold_observer is not None and not callable(cold_observer):
            raise TypeError("cold_observer must be callable or None")
        if direct_provider is not None and not callable(direct_provider):
            raise TypeError("direct_provider must be callable or None")
        if snapshot_name is not None and (
            not isinstance(snapshot_name, str)
            or not snapshot_name
            or "\x00" in snapshot_name
        ):
            raise ValueError("snapshot_name must be non-empty text or None")
        if snapshot_restorer is not None and not callable(snapshot_restorer):
            raise TypeError("snapshot_restorer must be callable or None")
        if snapshot_restorer is not None and snapshot_name is None:
            raise ValueError("snapshot_restorer requires snapshot_name")
        if not isinstance(commit_on_fertig_abstention, bool):
            raise TypeError("commit_on_fertig_abstention must be boolean")
        self.controller = controller
        self.feature_provider = feature_provider
        self.quality_verifier = quality_verifier
        self.source_action = validate_action(source_action)
        self.cold_observer = cold_observer
        self.direct_provider = direct_provider
        self.snapshot_name = snapshot_name
        self.snapshot_restorer = snapshot_restorer
        self.commit_on_fertig_abstention = commit_on_fertig_abstention
        self._snapshot_sha256 = (
            None
            if snapshot_name is None
            else hashlib.sha256(
                controller.crystal_store.restore_state(snapshot_name)
            ).hexdigest()
        )
        self._lock = threading.RLock()

    @classmethod
    def for_controller(
        cls,
        controller: OoeController,
        *,
        quality_verifier: ExecutionQualityVerifier,
        source_action: OoeAction | str = "qwen_fallback",
        cold_observer: ColdObserver | None = None,
        snapshot_name: str | None = None,
        snapshot_restorer: ControllerRestorer | None = None,
        commit_on_fertig_abstention: bool = False,
    ) -> "OoeChatHook":
        """Build the direct exact-prompt hook over persisted controller history."""

        return cls(
            controller=controller,
            feature_provider=ControllerPromptFeatureProvider(controller),
            quality_verifier=quality_verifier,
            source_action=source_action,
            cold_observer=cold_observer,
            snapshot_name=snapshot_name,
            snapshot_restorer=snapshot_restorer,
            commit_on_fertig_abstention=commit_on_fertig_abstention,
        )

    @staticmethod
    def _stream_id(question: str, metadata: Mapping[str, Any]) -> str:
        conversation = metadata.get("conversation_id")
        identity = (
            conversation
            if isinstance(conversation, str) and conversation.strip()
            else hashlib.sha256(question.encode("utf-8")).hexdigest()
        )
        return f"chat:{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _decision_record(decision: Any) -> dict[str, Any]:
        return {
            "action": decision.action,
            "confidence": decision.confidence,
            "crystal_sha256": decision.crystal_sha256,
            "execution_receipt_sha256": decision.execution_receipt_sha256,
            "feature_receipt_sha256": decision.feature_receipt_sha256,
            "origin": decision.origin,
            "quality_verified": decision.quality_verified,
            "reason": decision.reason,
            "site_identity_sha256": decision.site_identity_sha256,
        }

    def try_warm(
        self,
        question: str,
        metadata: Mapping[str, Any],
    ) -> OoeChatAttempt:
        """Return a verified warm Result, or a sealed miss that falls through."""

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be non-empty text")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        with self._lock:
            try:
                if self.direct_provider is not None:
                    direct = self.direct_provider(question, metadata)
                    if direct is not None:
                        if (
                            not isinstance(direct, OoeChatAttempt)
                            or not direct.hit
                            or not callable(direct._settler)
                        ):
                            raise TypeError(
                                "direct_provider must return a settled warm attempt or None"
                            )
                        return direct
                feature = self.feature_provider(question, metadata)
                if feature is None:
                    return OoeChatAttempt(None, {"status": "no-feature"})
                if not isinstance(feature, QwenOoeFeatureReceipt):
                    raise TypeError(
                        "feature_provider must return QwenOoeFeatureReceipt or None"
                    )

                def verify(
                    receipt: QwenOoeFeatureReceipt,
                    execution: ActionExecution,
                ) -> bool:
                    if not bool(self.quality_verifier(receipt, execution)):
                        return False
                    try:
                        candidate = result_from_document(execution.result)
                    except (TypeError, ValueError):
                        return False
                    return (
                        candidate.ok
                        and isinstance(candidate.output, str)
                        and bool(candidate.output.strip())
                    )

                decision = self.controller.try_warm(
                    feature,
                    self.source_action,
                    quality_verifier=verify,
                    stream_id=self._stream_id(question, metadata),
                )
                evidence = {
                    "decision": self._decision_record(decision),
                    "feature_receipt_sha256": feature.sha256,
                    "status": "hit" if decision.quality_verified else "miss",
                }
                if (
                    decision.origin != "crystal"
                    or not decision.quality_verified
                    or decision.execution_result is None
                ):
                    return OoeChatAttempt(None, evidence, decision)
                result = result_from_document(decision.execution_result)
                return OoeChatAttempt(result, evidence, decision)
            except (OoeControllerStaleError, QwenOoeBridgeStaleError) as exc:
                return OoeChatAttempt(
                    None,
                    {
                        "error": f"{type(exc).__module__}.{type(exc).__qualname__}",
                        "status": "stale",
                    },
                )
            except OoeChatIntegrityError:
                raise
            except (
                OoeControllerIntegrityError,
                QwenOoeBridgeIntegrityError,
                CrystalStoreError,
            ) as exc:
                raise OoeChatIntegrityError(
                    f"warm OoE integrity failure: {type(exc).__qualname__}"
                ) from exc
            except Exception as exc:
                return OoeChatAttempt(
                    None,
                    {
                        "error": f"{type(exc).__module__}.{type(exc).__qualname__}",
                        "status": "error",
                    },
                )

    def _restore_persisted_controller(self) -> None:
        if self.snapshot_name is None:
            return
        current = self.controller
        if self.snapshot_restorer is None:
            restored = OoeController.restore(
                crystal_store=current.crystal_store,
                name=self.snapshot_name,
                action_executors=current._executors,
                atlas_revision_verifier=current._atlas_revision_verifier,
                expected_model_pin_sha256=current.model_pin_sha256,
                expected_weight_graph_revision_sha256=(
                    current.weight_graph_revision_sha256
                ),
            )
        else:
            restored = self.snapshot_restorer()
        if (
            not isinstance(restored, OoeController)
            or restored.model_pin_sha256 != current.model_pin_sha256
            or restored.weight_graph_revision_sha256
            != current.weight_graph_revision_sha256
            or Path(restored.crystal_store.root)
            != Path(current.crystal_store.root)
        ):
            raise OoeChatIntegrityError(
                "snapshot restorer returned another controller authority"
            )
        raw = restored.crystal_store.restore_state(self.snapshot_name)
        self.controller = restored
        if isinstance(self.feature_provider, ControllerPromptFeatureProvider):
            self.feature_provider.controller = restored
        self._snapshot_sha256 = hashlib.sha256(raw).hexdigest()

    def _settle_warm(
        self,
        attempt: OoeChatAttempt,
        *,
        accept: bool,
    ) -> WarmAccountingReceipt:
        if not isinstance(attempt, OoeChatAttempt) or not attempt.hit:
            raise ValueError("only a warm hit can be settled")
        if attempt._settler is not None:
            if not callable(attempt._settler):
                raise OoeChatIntegrityError("direct warm settlement is invalid")
            receipt = attempt._settler(accept)
            if not isinstance(receipt, WarmAccountingReceipt):
                raise OoeChatIntegrityError(
                    "direct warm settlement returned an invalid receipt"
                )
            return receipt
        decision = attempt.decision
        if not isinstance(decision, OoeDecision):
            raise OoeChatIntegrityError("warm hit lost its controller decision")
        try:
            receipt = (
                self.controller.commit_warm(decision)
                if accept
                else self.controller.reject_warm(decision)
            )
            if self.snapshot_name is not None:
                publication = self.controller.save_snapshot(
                    name=self.snapshot_name,
                    expected_sha256=self._snapshot_sha256,
                )
                self._snapshot_sha256 = publication.payload_sha256
            return receipt
        except (OoeControllerIntegrityError, CrystalStoreError) as exc:
            if self.snapshot_name is not None:
                try:
                    self._restore_persisted_controller()
                except Exception as restore_exc:
                    raise OoeChatIntegrityError(
                        "warm accounting failed and controller recovery failed"
                    ) from restore_exc
            raise OoeChatIntegrityError("warm accounting integrity failure") from exc

    def commit_warm(self, attempt: OoeChatAttempt) -> WarmAccountingReceipt:
        """Commit savings only after the outer verifier accepts the final answer."""

        return self._settle_warm(attempt, accept=True)

    def reject_warm(self, attempt: OoeChatAttempt) -> WarmAccountingReceipt:
        """Reject pending savings after mismatch, abstention, or adjudicator failure."""

        return self._settle_warm(attempt, accept=False)

    def abstention_commit_authorized(self, _attempt: OoeChatAttempt) -> bool:
        return (
            _attempt._abstention_authorized
            or self.commit_on_fertig_abstention
        )

    def observe_cold(
        self,
        question: str,
        metadata: Mapping[str, Any],
        qwen_result: Result,
        final_result: Result,
    ) -> dict[str, Any]:
        """Send one completed cold path to O1/Atlas without affecting its answer."""

        if self.cold_observer is None:
            return {"status": "not-configured"}
        try:
            observation = self.cold_observer(
                question,
                metadata,
                qwen_result,
                final_result,
            )
            normalized = {} if observation is None else dict(observation)
            receipt = {
                "final_result_sha256": result_to_document(final_result)["sha256"],
                "observation_sha256": _digest(normalized),
                "qwen_result_sha256": result_to_document(qwen_result)["sha256"],
                "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "schema": CHAT_OBSERVATION_SCHEMA,
            }
            return {
                "observation": normalized,
                "receipt": receipt,
                "status": "observed",
            }
        except Exception as exc:
            return {
                "error": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "status": "error",
            }


class ChainedOoeChatHook(OoeChatHook):
    """Try independent verified Markov authorities in deterministic order."""

    def __init__(self, hooks: tuple[OoeChatHook, ...]) -> None:
        if (
            not isinstance(hooks, tuple)
            or len(hooks) < 2
            or any(type(hook) is not OoeChatHook for hook in hooks)
        ):
            raise TypeError("hooks must contain at least two production OoE hooks")
        self.hooks = hooks
        self.controller = hooks[0].controller
        self.commit_on_fertig_abstention = any(
            hook.commit_on_fertig_abstention for hook in hooks
        )
        self.cold_observer = self.observe_cold
        self._lock = threading.RLock()

    def try_warm(
        self,
        question: str,
        metadata: Mapping[str, Any],
    ) -> OoeChatAttempt:
        attempts = []
        with self._lock:
            for index, hook in enumerate(self.hooks):
                attempt = hook.try_warm(question, metadata)
                attempts.append({"authority": index, **dict(attempt.evidence)})
                if attempt.hit:
                    self.controller = hook.controller
                    return OoeChatAttempt(
                        attempt.result,
                        {
                            "attempts": attempts,
                            "authority": index,
                            "status": "hit",
                        },
                        attempt.decision,
                        _owner=hook,
                        _settler=attempt._settler,
                        _abstention_authorized=(
                            attempt._abstention_authorized
                        ),
                    )
        return OoeChatAttempt(
            None,
            {"attempts": attempts, "status": "miss"},
        )

    @staticmethod
    def _owner(attempt: OoeChatAttempt) -> OoeChatHook:
        owner = attempt._owner
        if type(owner) is not OoeChatHook:
            raise OoeChatIntegrityError("chained warm attempt lost its authority")
        return owner

    def commit_warm(self, attempt: OoeChatAttempt) -> WarmAccountingReceipt:
        owner = self._owner(attempt)
        receipt = owner.commit_warm(attempt)
        self.controller = owner.controller
        return receipt

    def reject_warm(self, attempt: OoeChatAttempt) -> WarmAccountingReceipt:
        owner = self._owner(attempt)
        receipt = owner.reject_warm(attempt)
        self.controller = owner.controller
        return receipt

    def abstention_commit_authorized(self, attempt: OoeChatAttempt) -> bool:
        return self._owner(attempt).abstention_commit_authorized(attempt)

    def observe_cold(
        self,
        question: str,
        metadata: Mapping[str, Any],
        qwen_result: Result,
        final_result: Result,
    ) -> dict[str, Any]:
        observations = []
        for index, hook in enumerate(self.hooks):
            if hook.cold_observer is None:
                continue
            observations.append(
                {
                    "authority": index,
                    **hook.observe_cold(
                        question,
                        metadata,
                        qwen_result,
                        final_result,
                    ),
                }
            )
            self.controller = hook.controller
        return {"authorities": observations, "status": "observed"}


__all__ = [
    "CHAT_OBSERVATION_SCHEMA",
    "CHAT_RESULT_SCHEMA",
    "ChainedOoeChatHook",
    "ColdObserver",
    "ControllerPromptFeatureProvider",
    "FeatureProvider",
    "OoeChatAttempt",
    "OoeChatHook",
    "OoeChatIntegrityError",
    "result_from_document",
    "result_to_document",
]
