"""Stable output authority for cross-profile zero-forward Qwen replay."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

from .mlp_page_markov import MLP_PAGE_OUTPUT_ABI as QWEN_MLP_PAGE_OUTPUT_ABI


QWEN_OUTPUT_SEMANTICS_SCHEMA = "immer.qwen3.8-output-semantics/v1"
QWEN_SEMANTIC_REPLAY_KEY_SCHEMA = "immer.qwen3.8-semantic-replay-key/v1"
QWEN_SEMANTIC_REPLAY_RECEIPT_SCHEMA = (
    "immer.qwen3.8-semantic-replay-receipt/v1"
)
QWEN_SEMANTIC_REPLAY_EVIDENCE_KEY = "qwen_semantic_replay_receipt"
QWEN_SEMANTIC_REPLAY_METADATA_KEY = "qwen_semantic_replay_key"
QWEN_TARGET_OUTPUT_ABI = "greedy-no-thinking-target-authority/v1"
_HEX = frozenset("0123456789abcdef")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or bool(set(value) - _HEX)
    ):
        raise ValueError(f"{field} must be lowercase SHA-256")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class QwenOutputSemantics:
    repo_id: str
    revision: str
    q4_manifest_file_sha256: str
    q4_native_abi: int
    q4_bank_codec_abi: int
    tokenizer_sha256: str
    compute_dtype: str
    max_context_tokens: int
    max_prompt_tokens: int
    max_new_tokens: int
    eos_token_ids: tuple[int, ...]
    mlp_page_route_width: int | None = None
    mlp_page_width_actions: tuple[int, ...] = ()
    mlp_page_energy_coverage: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.repo_id, str) or not self.repo_id:
            raise ValueError("repo_id must be non-empty")
        if not isinstance(self.revision, str) or not self.revision:
            raise ValueError("revision must be non-empty")
        for field in (
            "q4_manifest_file_sha256",
            "tokenizer_sha256",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
        for field in (
            "q4_native_abi",
            "q4_bank_codec_abi",
            "max_context_tokens",
            "max_prompt_tokens",
            "max_new_tokens",
        ):
            _positive_int(getattr(self, field), field)
        if self.max_prompt_tokens + self.max_new_tokens > self.max_context_tokens:
            raise ValueError("prompt plus output exceeds semantic context bound")
        dtype = "bfloat16" if self.compute_dtype == "auto" else self.compute_dtype
        if dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("compute_dtype is invalid")
        object.__setattr__(self, "compute_dtype", dtype)
        eos = tuple(self.eos_token_ids)
        if (
            not eos
            or len(set(eos)) != len(eos)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in eos
            )
        ):
            raise ValueError("eos_token_ids are invalid")
        object.__setattr__(self, "eos_token_ids", eos)
        width = self.mlp_page_route_width
        actions = tuple(self.mlp_page_width_actions)
        coverage = self.mlp_page_energy_coverage
        if width is None:
            if actions or coverage is not None:
                raise ValueError("disabled MLP page semantics retain route fields")
            return
        _positive_int(width, "mlp_page_route_width")
        if (
            not actions
            or actions != tuple(sorted(set(actions)))
            or actions[-1] != width
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= width
                for value in actions
            )
        ):
            raise ValueError("MLP page width actions are invalid")
        if (
            isinstance(coverage, bool)
            or not isinstance(coverage, (int, float))
            or not math.isfinite(float(coverage))
            or not 0.0 < float(coverage) <= 1.0
        ):
            raise ValueError("MLP page energy coverage is invalid")
        object.__setattr__(self, "mlp_page_width_actions", actions)
        object.__setattr__(self, "mlp_page_energy_coverage", float(coverage))

    def to_document(self) -> dict[str, object]:
        body = {
            "compute_dtype": self.compute_dtype,
            "eos_token_ids": list(self.eos_token_ids),
            "max_context_tokens": self.max_context_tokens,
            "max_new_tokens": self.max_new_tokens,
            "max_prompt_tokens": self.max_prompt_tokens,
            "mlp_page": (
                None
                if self.mlp_page_route_width is None
                else {
                    "energy_coverage": self.mlp_page_energy_coverage.hex(),
                    "output_abi": QWEN_MLP_PAGE_OUTPUT_ABI,
                    "route_width": self.mlp_page_route_width,
                    "width_actions": list(self.mlp_page_width_actions),
                }
            ),
            "q4_bank_codec_abi": self.q4_bank_codec_abi,
            "q4_manifest_file_sha256": self.q4_manifest_file_sha256,
            "q4_native_abi": self.q4_native_abi,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "target_output_abi": QWEN_TARGET_OUTPUT_ABI,
            "tokenizer_sha256": self.tokenizer_sha256,
        }
        return {
            "body": body,
            "schema": QWEN_OUTPUT_SEMANTICS_SCHEMA,
            "sha256": _digest(body),
        }

    @property
    def sha256(self) -> str:
        return str(self.to_document()["sha256"])

    @classmethod
    def from_document(cls, value: object) -> "QwenOutputSemantics":
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "schema", "sha256"}
            or value.get("schema") != QWEN_OUTPUT_SEMANTICS_SCHEMA
            or not isinstance(value.get("body"), Mapping)
            or value.get("sha256") != _digest(value["body"])
        ):
            raise ValueError("Qwen output-semantics document is invalid")
        body = value["body"]
        expected = {
            "compute_dtype",
            "eos_token_ids",
            "max_context_tokens",
            "max_new_tokens",
            "max_prompt_tokens",
            "mlp_page",
            "q4_bank_codec_abi",
            "q4_manifest_file_sha256",
            "q4_native_abi",
            "repo_id",
            "revision",
            "target_output_abi",
            "tokenizer_sha256",
        }
        if set(body) != expected or body["target_output_abi"] != QWEN_TARGET_OUTPUT_ABI:
            raise ValueError("Qwen output-semantics body is invalid")
        page = body["mlp_page"]
        if page is None:
            page_values = (None, (), None)
        elif (
            isinstance(page, Mapping)
            and set(page)
            == {"energy_coverage", "output_abi", "route_width", "width_actions"}
            and page.get("output_abi") == QWEN_MLP_PAGE_OUTPUT_ABI
            and isinstance(page.get("energy_coverage"), str)
        ):
            page_values = (
                page["route_width"],
                tuple(page["width_actions"]),
                float.fromhex(page["energy_coverage"]),
            )
        else:
            raise ValueError("Qwen MLP output semantics are invalid")
        result = cls(
            repo_id=body["repo_id"],
            revision=body["revision"],
            q4_manifest_file_sha256=body["q4_manifest_file_sha256"],
            q4_native_abi=body["q4_native_abi"],
            q4_bank_codec_abi=body["q4_bank_codec_abi"],
            tokenizer_sha256=body["tokenizer_sha256"],
            compute_dtype=body["compute_dtype"],
            max_context_tokens=body["max_context_tokens"],
            max_prompt_tokens=body["max_prompt_tokens"],
            max_new_tokens=body["max_new_tokens"],
            eos_token_ids=tuple(body["eos_token_ids"]),
            mlp_page_route_width=page_values[0],
            mlp_page_width_actions=page_values[1],
            mlp_page_energy_coverage=page_values[2],
        )
        if result.to_document() != dict(value):
            raise ValueError("Qwen output semantics failed canonical reconstruction")
        return result


@dataclass(frozen=True, slots=True)
class QwenSemanticReplayKey:
    output_semantics_sha256: str
    question_sha256: str
    rendered_prompt_sha256: str
    rendered_prompt_token_sha256: str
    system_prompt_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "output_semantics_sha256",
            "question_sha256",
            "rendered_prompt_sha256",
            "rendered_prompt_token_sha256",
            "system_prompt_sha256",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))

    def to_document(self) -> dict[str, object]:
        body = {
            "output_semantics_sha256": self.output_semantics_sha256,
            "question_sha256": self.question_sha256,
            "rendered_prompt_sha256": self.rendered_prompt_sha256,
            "rendered_prompt_token_sha256": self.rendered_prompt_token_sha256,
            "system_prompt_sha256": self.system_prompt_sha256,
        }
        return {
            "body": body,
            "schema": QWEN_SEMANTIC_REPLAY_KEY_SCHEMA,
            "sha256": _digest(body),
        }

    @property
    def sha256(self) -> str:
        return str(self.to_document()["sha256"])

    @classmethod
    def from_document(cls, value: object) -> "QwenSemanticReplayKey":
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "schema", "sha256"}
            or value.get("schema") != QWEN_SEMANTIC_REPLAY_KEY_SCHEMA
            or not isinstance(value.get("body"), Mapping)
            or value.get("sha256") != _digest(value["body"])
        ):
            raise ValueError("Qwen semantic replay key is invalid")
        body = value["body"]
        if set(body) != {
            "output_semantics_sha256",
            "question_sha256",
            "rendered_prompt_sha256",
            "rendered_prompt_token_sha256",
            "system_prompt_sha256",
        }:
            raise ValueError("Qwen semantic replay key body is invalid")
        result = cls(**body)
        if result.to_document() != dict(value):
            raise ValueError("Qwen semantic replay key failed reconstruction")
        return result


def semantic_replay_receipt(
    semantics: QwenOutputSemantics,
    key: QwenSemanticReplayKey,
) -> dict[str, object]:
    if key.output_semantics_sha256 != semantics.sha256:
        raise ValueError("semantic replay key belongs to another output authority")
    body = {"key": key.to_document(), "semantics": semantics.to_document()}
    return {
        "body": body,
        "schema": QWEN_SEMANTIC_REPLAY_RECEIPT_SCHEMA,
        "sha256": _digest(body),
    }


def semantic_replay_key_for_prompt(
    semantics: QwenOutputSemantics,
    *,
    question: str,
    rendered_prompt: str,
    rendered_prompt_token_sha256: str,
    system_prompt: str,
) -> QwenSemanticReplayKey:
    if not isinstance(semantics, QwenOutputSemantics):
        raise TypeError("semantics must be QwenOutputSemantics")
    for value, field in (
        (question, "question"),
        (rendered_prompt, "rendered_prompt"),
        (system_prompt, "system_prompt"),
    ):
        if not isinstance(value, str):
            raise TypeError(f"{field} must be text")
    return QwenSemanticReplayKey(
        output_semantics_sha256=semantics.sha256,
        question_sha256=hashlib.sha256(question.encode("utf-8")).hexdigest(),
        rendered_prompt_sha256=hashlib.sha256(
            rendered_prompt.encode("utf-8")
        ).hexdigest(),
        rendered_prompt_token_sha256=rendered_prompt_token_sha256,
        system_prompt_sha256=hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
    )


def parse_semantic_replay_receipt(
    value: object,
) -> tuple[QwenOutputSemantics, QwenSemanticReplayKey]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"body", "schema", "sha256"}
        or value.get("schema") != QWEN_SEMANTIC_REPLAY_RECEIPT_SCHEMA
        or not isinstance(value.get("body"), Mapping)
        or set(value["body"]) != {"key", "semantics"}
        or value.get("sha256") != _digest(value["body"])
    ):
        raise ValueError("Qwen semantic replay receipt is invalid")
    semantics = QwenOutputSemantics.from_document(value["body"]["semantics"])
    key = QwenSemanticReplayKey.from_document(value["body"]["key"])
    if semantic_replay_receipt(semantics, key) != dict(value):
        raise ValueError("Qwen semantic replay receipt failed reconstruction")
    return semantics, key


__all__ = [
    "QWEN_MLP_PAGE_OUTPUT_ABI",
    "QWEN_OUTPUT_SEMANTICS_SCHEMA",
    "QWEN_SEMANTIC_REPLAY_EVIDENCE_KEY",
    "QWEN_SEMANTIC_REPLAY_KEY_SCHEMA",
    "QWEN_SEMANTIC_REPLAY_METADATA_KEY",
    "QWEN_SEMANTIC_REPLAY_RECEIPT_SCHEMA",
    "QWEN_TARGET_OUTPUT_ABI",
    "QwenOutputSemantics",
    "QwenSemanticReplayKey",
    "parse_semantic_replay_receipt",
    "semantic_replay_key_for_prompt",
    "semantic_replay_receipt",
]
