"""Passive content-addressed action economics from normal inference receipts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import threading
from typing import Any

from ..ooe.identity import canonical_json_bytes, require_sha256
from .inference_economics import InferenceEconomicsReceipt


INFERENCE_ACTION_RECEIPT_SCHEMA = "immer.qwen3.8-inference-action-receipt/v1"
INFERENCE_ACTION_BANK_SCHEMA = "immer.qwen3.8-inference-action-bank/v1"
INFERENCE_ACTION_DIRECTIVE_SCHEMA = "immer.qwen3.8-inference-action-directive/v1"
MAX_ACTION_RECEIPT_BYTES = 32 * 1024
ACTION_CATALOG = (
    "compute_crystal",
    "continuation_battery",
    "dynamic_mlp_pages",
    "external_drafter",
    "fertig_exact",
    "mlp_head_coordinate",
    "parametric_program",
    "prefix_sinkhorn",
    "qwen_suffix",
    "qwen_target",
    "stored_result",
    "target_verified_draft",
)
_EVENT = re.compile(r"([0-9a-f]{64})-([0-9a-f]{64})\.json")


class InferenceActionBankError(RuntimeError):
    """The passive action bank or one of its content addresses is invalid."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _uint(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _nonnegative_float(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or float(value) < 0.0
        or float(value) != float(value)
        or float(value) in {float("inf"), float("-inf")}
    ):
        raise ValueError(f"{label} must be finite and non-negative")
    return float(value)


def _actions(receipt: InferenceEconomicsReceipt) -> tuple[str, ...]:
    if receipt.fertig_exact:
        return ("fertig_exact",)
    if receipt.warm_hit and receipt.target_forwards == 0:
        return ("stored_result",)
    actions = []
    if receipt.draft_active:
        actions.append("target_verified_draft")
    if receipt.page_active:
        actions.append("dynamic_mlp_pages")
    if receipt.target_forwards > 0:
        actions.append("qwen_target")
    if not actions:
        actions.append("qwen_suffix")
    return tuple(sorted(actions))


@dataclass(frozen=True, slots=True)
class InferenceActionReceipt:
    request_sha256: str
    economics_receipt_sha256: str
    question_sha256: str
    runtime_profile_sha256: str
    result_sha256: str
    input_contract_sha256: str
    action_signature_sha256: str
    quality_authority_sha256: str
    actions: tuple[str, ...]
    status: str
    target_forwards: int
    saved_qwen_forwards: int
    generated_tokens: int
    accepted_draft_tokens: int
    proposed_draft_tokens: int
    source_body_bytes: int
    request_wall_seconds: float
    process_peak_rss_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "request_sha256",
            "economics_receipt_sha256",
            "question_sha256",
            "runtime_profile_sha256",
            "result_sha256",
            "input_contract_sha256",
            "action_signature_sha256",
            "quality_authority_sha256",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), field=name),
            )
        if (
            not self.actions
            or tuple(sorted(set(self.actions))) != self.actions
            or any(action not in ACTION_CATALOG for action in self.actions)
        ):
            raise ValueError("actions must be sorted known action classes")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must be non-empty text")
        for name in (
            "target_forwards",
            "saved_qwen_forwards",
            "generated_tokens",
            "accepted_draft_tokens",
            "proposed_draft_tokens",
            "source_body_bytes",
            "process_peak_rss_bytes",
        ):
            _uint(getattr(self, name), name)
        _nonnegative_float(self.request_wall_seconds, "request_wall_seconds")

    @classmethod
    def from_economics(
        cls,
        receipt: InferenceEconomicsReceipt,
    ) -> "InferenceActionReceipt":
        if not isinstance(receipt, InferenceEconomicsReceipt):
            raise TypeError("receipt must be InferenceEconomicsReceipt")
        actions = _actions(receipt)
        signature = _digest(
            {
                "actions": actions,
                "runtime_profile_sha256": receipt.runtime_profile_sha256,
                "schema": "immer.qwen3.8-action-signature/v1",
            }
        )
        input_contract = _digest(
            {
                "action_signature_sha256": signature,
                "question_sha256": receipt.question_sha256,
                "runtime_profile_sha256": receipt.runtime_profile_sha256,
                "schema": "immer.qwen3.8-action-input-contract/v1",
            }
        )
        zero_forward_action = (
            receipt.warm_hit and receipt.target_forwards == 0
        ) or receipt.fertig_exact
        return cls(
            request_sha256=receipt.request_sha256,
            economics_receipt_sha256=receipt.sha256,
            question_sha256=receipt.question_sha256,
            runtime_profile_sha256=receipt.runtime_profile_sha256,
            result_sha256=receipt.result_sha256,
            input_contract_sha256=input_contract,
            action_signature_sha256=signature,
            quality_authority_sha256=receipt.result_sha256,
            actions=actions,
            status=receipt.status,
            target_forwards=receipt.target_forwards,
            saved_qwen_forwards=receipt.saved_qwen_forwards,
            generated_tokens=receipt.generated_tokens,
            accepted_draft_tokens=(
                0 if zero_forward_action else receipt.accepted_draft_tokens
            ),
            proposed_draft_tokens=(
                0 if zero_forward_action else receipt.proposed_draft_tokens
            ),
            source_body_bytes=(
                0
                if zero_forward_action
                else max(
                    receipt.source_body_bytes,
                    receipt.target_source_body_bytes + receipt.draft_source_body_bytes,
                )
            ),
            request_wall_seconds=(
                0.0 if zero_forward_action else receipt.request_wall_seconds
            ),
            process_peak_rss_bytes=(
                0 if zero_forward_action else receipt.process_peak_rss_bytes
            ),
        )

    @property
    def sha256(self) -> str:
        return _digest(self.body())

    def body(self) -> dict[str, object]:
        return {
            name: list(value) if name == "actions" else value
            for name, value in (
                (field, getattr(self, field)) for field in self.__dataclass_fields__
            )
        }

    def to_document(self) -> dict[str, object]:
        body = self.body()
        document = {
            "body": body,
            "schema": INFERENCE_ACTION_RECEIPT_SCHEMA,
            "sha256": _digest(body),
        }
        if len(canonical_json_bytes(document)) > MAX_ACTION_RECEIPT_BYTES:
            raise InferenceActionBankError("action receipt exceeds 32 KiB")
        return document

    @classmethod
    def from_document(cls, value: object) -> "InferenceActionReceipt":
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "schema", "sha256"}
            or value.get("schema") != INFERENCE_ACTION_RECEIPT_SCHEMA
            or not isinstance(value.get("body"), Mapping)
            or value.get("sha256") != _digest(value["body"])
        ):
            raise InferenceActionBankError("action receipt envelope is invalid")
        body = dict(value["body"])
        actions = body.get("actions")
        if isinstance(actions, list):
            body["actions"] = tuple(actions)
        try:
            receipt = cls(**body)
        except (TypeError, ValueError) as exc:
            raise InferenceActionBankError("action receipt is invalid") from exc
        if receipt.to_document() != dict(value):
            raise InferenceActionBankError("action receipt is not canonical")
        return receipt


@dataclass(frozen=True, slots=True)
class InferenceActionObservation:
    receipt: InferenceActionReceipt
    duplicate: bool
    snapshot: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class InferenceActionDirective:
    question_sha256: str
    runtime_profile_sha256: str
    primary_actions: tuple[str, ...]
    fallback_actions: tuple[str, ...]
    draft_enabled: bool | None
    source_signature_sha256s: tuple[str, ...]
    support: int
    saved_qwen_forwards: int

    def __post_init__(self) -> None:
        for name in ("question_sha256", "runtime_profile_sha256"):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), field=name),
            )
        for name in ("primary_actions", "fallback_actions"):
            actions = getattr(self, name)
            if (
                not actions
                or tuple(sorted(set(actions))) != actions
                or any(action not in ACTION_CATALOG for action in actions)
            ):
                raise ValueError(f"{name} must be sorted known action classes")
        if self.draft_enabled is not None and not isinstance(
            self.draft_enabled,
            bool,
        ):
            raise TypeError("draft_enabled must be boolean or null")
        if (
            tuple(sorted(set(self.source_signature_sha256s)))
            != self.source_signature_sha256s
            or any(
                require_sha256(value, field="source signature") != value
                for value in self.source_signature_sha256s
            )
        ):
            raise ValueError("source signatures must be sorted unique SHA-256 values")
        if self.support <= 0 or isinstance(self.support, bool):
            raise ValueError("directive support must be positive")
        _uint(self.saved_qwen_forwards, "saved_qwen_forwards")

    @property
    def sha256(self) -> str:
        return _digest(self.body())

    def body(self) -> dict[str, object]:
        return {
            "draft_enabled": self.draft_enabled,
            "fallback_actions": list(self.fallback_actions),
            "primary_actions": list(self.primary_actions),
            "question_sha256": self.question_sha256,
            "runtime_profile_sha256": self.runtime_profile_sha256,
            "saved_qwen_forwards": self.saved_qwen_forwards,
            "source_signature_sha256s": list(self.source_signature_sha256s),
            "support": self.support,
        }

    def to_document(self) -> dict[str, object]:
        body = self.body()
        return {
            "body": body,
            "schema": INFERENCE_ACTION_DIRECTIVE_SCHEMA,
            "sha256": _digest(body),
        }

    @classmethod
    def from_document(cls, value: object) -> "InferenceActionDirective":
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "schema", "sha256"}
            or value.get("schema") != INFERENCE_ACTION_DIRECTIVE_SCHEMA
            or not isinstance(value.get("body"), Mapping)
            or value.get("sha256") != _digest(value["body"])
        ):
            raise InferenceActionBankError("action directive envelope is invalid")
        body = dict(value["body"])
        for name in (
            "fallback_actions",
            "primary_actions",
            "source_signature_sha256s",
        ):
            if isinstance(body.get(name), list):
                body[name] = tuple(body[name])
        try:
            directive = cls(**body)
        except (TypeError, ValueError) as exc:
            raise InferenceActionBankError("action directive is invalid") from exc
        if directive.to_document() != dict(value):
            raise InferenceActionBankError("action directive is not canonical")
        return directive


class InferenceActionBank:
    """Crash-safe passive action index derived from authoritative economics."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().absolute()
        self.events = self.root / "events"
        self.staging = self.root / "staging"
        self._lock = threading.RLock()
        for index, path in enumerate((self.root, self.events, self.staging)):
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                path.mkdir(parents=index == 0, exist_ok=False, mode=0o700)
                metadata = path.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise InferenceActionBankError(
                    "action-bank storage path is not a plain directory"
                )
            path.chmod(0o700)

    @staticmethod
    def _write_all(descriptor: int, payload: bytes) -> None:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short action-bank write")
            view = view[written:]

    def _publish(self, destination: Path, payload: bytes) -> bool:
        temporary = self.staging / (
            f"{destination.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
        )
        descriptor = os.open(
            temporary,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
            0o600,
        )
        try:
            self._write_all(descriptor, payload)
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, destination)
            directory = os.open(self.events, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return True
        except FileExistsError:
            return False
        finally:
            temporary.unlink(missing_ok=True)

    def _receipts(self) -> tuple[InferenceActionReceipt, ...]:
        receipts = []
        requests: dict[str, str] = {}
        for path in sorted(self.events.iterdir(), key=lambda item: item.name):
            if not stat.S_ISREG(path.lstat().st_mode) or _EVENT.fullmatch(path.name) is None:
                raise InferenceActionBankError("action-bank event inventory is invalid")
            raw = path.read_bytes()
            if len(raw) > MAX_ACTION_RECEIPT_BYTES:
                raise InferenceActionBankError("action-bank event exceeds 32 KiB")
            try:
                document = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise InferenceActionBankError("action-bank event is invalid JSON") from exc
            if canonical_json_bytes(document) != raw:
                raise InferenceActionBankError("action-bank event is not canonical")
            receipt = InferenceActionReceipt.from_document(document)
            match = _EVENT.fullmatch(path.name)
            assert match is not None
            if (match.group(1), match.group(2)) != (
                receipt.request_sha256,
                receipt.sha256,
            ):
                raise InferenceActionBankError("action-bank event name is invalid")
            prior = requests.get(receipt.request_sha256)
            if prior is not None and prior != receipt.sha256:
                raise InferenceActionBankError(
                    "one action-bank request has conflicting receipts"
                )
            requests[receipt.request_sha256] = receipt.sha256
            receipts.append(receipt)
        return tuple(receipts)

    @staticmethod
    def _snapshot(receipts: Iterable[InferenceActionReceipt]) -> dict[str, Any]:
        rows: dict[str, dict[str, Any]] = {}
        catalog = {action: 0 for action in ACTION_CATALOG}
        requests = 0
        for receipt in receipts:
            requests += 1
            for action in receipt.actions:
                catalog[action] += 1
            row = rows.setdefault(
                receipt.action_signature_sha256,
                {
                    "accepted_draft_tokens": 0,
                    "actions": list(receipt.actions),
                    "generated_tokens": 0,
                    "max_process_peak_rss_bytes": 0,
                    "ok_results": 0,
                    "positive_net_requests": 0,
                    "proposed_draft_tokens": 0,
                    "request_wall_seconds": 0.0,
                    "runtime_profile_sha256": receipt.runtime_profile_sha256,
                    "saved_qwen_forwards": 0,
                    "signature_sha256": receipt.action_signature_sha256,
                    "source_body_bytes": 0,
                    "support": 0,
                    "target_forwards": 0,
                },
            )
            row["support"] += 1
            row["ok_results"] += int(receipt.status == "ok")
            row["positive_net_requests"] += int(receipt.saved_qwen_forwards > 0)
            row["target_forwards"] += receipt.target_forwards
            row["saved_qwen_forwards"] += receipt.saved_qwen_forwards
            row["generated_tokens"] += receipt.generated_tokens
            row["accepted_draft_tokens"] += receipt.accepted_draft_tokens
            row["proposed_draft_tokens"] += receipt.proposed_draft_tokens
            row["source_body_bytes"] += receipt.source_body_bytes
            row["request_wall_seconds"] += receipt.request_wall_seconds
            row["max_process_peak_rss_bytes"] = max(
                row["max_process_peak_rss_bytes"],
                receipt.process_peak_rss_bytes,
            )
        ranked = sorted(
            rows.values(),
            key=lambda row: (
                -row["saved_qwen_forwards"],
                -row["ok_results"],
                row["target_forwards"],
                row["request_wall_seconds"],
                row["signature_sha256"],
            ),
        )
        return {
            "action_catalog": [
                {"action": action, "observed_requests": catalog[action]}
                for action in ACTION_CATALOG
            ],
            "leader_signature_sha256": (
                None if not ranked else ranked[0]["signature_sha256"]
            ),
            "requests": requests,
            "schema": INFERENCE_ACTION_BANK_SCHEMA,
            "signatures": ranked,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot(self._receipts())

    def observe(
        self,
        economics: InferenceEconomicsReceipt,
    ) -> InferenceActionObservation:
        receipt = InferenceActionReceipt.from_economics(economics)
        payload = canonical_json_bytes(receipt.to_document())
        destination = self.events / (
            f"{receipt.request_sha256}-{receipt.sha256}.json"
        )
        with self._lock:
            created = self._publish(destination, payload)
            receipts = self._receipts()
            matches = [
                row for row in receipts if row.request_sha256 == receipt.request_sha256
            ]
            if len(matches) != 1 or matches[0].sha256 != receipt.sha256:
                raise InferenceActionBankError(
                    "action-bank request conflicts with existing evidence"
                )
            return InferenceActionObservation(
                receipt=matches[0],
                duplicate=not created,
                snapshot=self._snapshot(receipts),
            )

    def reconcile(
        self,
        receipts: Iterable[InferenceEconomicsReceipt],
    ) -> dict[str, Any]:
        for receipt in receipts:
            self.observe(receipt)
        return self.snapshot()

    def recommend(
        self,
        *,
        question_sha256: str,
        runtime_profile_sha256: str,
    ) -> InferenceActionDirective | None:
        question = require_sha256(question_sha256, field="question_sha256")
        profile = require_sha256(
            runtime_profile_sha256,
            field="runtime_profile_sha256",
        )
        with self._lock:
            receipts = self._receipts()
        exact = [
            receipt
            for receipt in receipts
            if receipt.question_sha256 == question
            and receipt.status == "ok"
            and any(
                action in {"fertig_exact", "stored_result"}
                for action in receipt.actions
            )
        ]
        exact.sort(
            key=lambda receipt: (
                -receipt.saved_qwen_forwards,
                receipt.target_forwards,
                receipt.action_signature_sha256,
            )
        )
        runtime = [
            receipt
            for receipt in receipts
            if receipt.runtime_profile_sha256 == profile
            and receipt.status == "ok"
            and "qwen_target" in receipt.actions
            and "target_verified_draft" in receipt.actions
            and receipt.saved_qwen_forwards > 0
        ]
        runtime.sort(
            key=lambda receipt: (
                -receipt.saved_qwen_forwards,
                receipt.target_forwards,
                receipt.request_wall_seconds,
                receipt.action_signature_sha256,
            )
        )
        if not exact and not runtime:
            return None
        primary = exact[0].actions if exact else runtime[0].actions
        fallback = runtime[0].actions if runtime else ("qwen_target",)
        sources = tuple(
            sorted(
                {
                    receipt.action_signature_sha256
                    for receipt in (*exact, *runtime)
                }
            )
        )
        return InferenceActionDirective(
            question_sha256=question,
            runtime_profile_sha256=profile,
            primary_actions=primary,
            fallback_actions=fallback,
            draft_enabled=True if runtime else None,
            source_signature_sha256s=sources,
            support=len(exact) + len(runtime),
            saved_qwen_forwards=sum(
                receipt.saved_qwen_forwards for receipt in (*exact, *runtime)
            ),
        )


__all__ = [
    "ACTION_CATALOG",
    "INFERENCE_ACTION_BANK_SCHEMA",
    "INFERENCE_ACTION_DIRECTIVE_SCHEMA",
    "INFERENCE_ACTION_RECEIPT_SCHEMA",
    "InferenceActionBank",
    "InferenceActionBankError",
    "InferenceActionDirective",
    "InferenceActionObservation",
    "InferenceActionReceipt",
]
