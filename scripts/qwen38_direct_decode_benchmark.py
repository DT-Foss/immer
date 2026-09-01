#!/usr/bin/env python3
"""Benchmark Qwen3.8 transport parity and sealed free-generation branches.

The original v1 commands benchmark one teacher-forced decode token.  The
separate branch commands below select a gold-free cohort, run genuine greedy
autoregressive generation, compare sealed off/CRSA arms, and only then admit a
separate label-bearing source for transition evaluation.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time
from typing import Any

import torch

from immer.cognition.fertig import FertigSolver
from immer.knowledge import AccessTrace, AccessTraceRecorder, Streamer
from immer.runtimes.deepseek_v4.benchmark import extract_gsm8k_answer
from immer.runtimes.qwen3_8 import (
    END_OF_TEXT_TOKEN_ID,
    IM_END_TOKEN_ID,
    NATIVE_HEAD_CRSA_EVIDENCE_SCHEMA,
    NATIVE_HEAD_CRSA_FREE_HEADS,
    NATIVE_HEAD_CRSA_KV_HEADS,
    NATIVE_HEAD_CRSA_LAYER,
    NATIVE_HEAD_CRSA_QUERY_HEADS,
    NATIVE_HEAD_CRSA_ROW_SUM_TOLERANCE,
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    CausalWeightMount,
    DELTANET_PROBE_SCHEMA,
    DeltaNetProbeRecorder,
    LogicalModelIdentity,
    NativeHeadCrsaEvidence,
    Qwen38BundleError,
    Qwen38Config,
    Qwen38NativeHeadCrsa,
    Qwen38NativeFork,
    Qwen38StableCrsaGraft,
    Qwen38Tokenizer,
    Qwen38WeightPager,
    StreamedQwen38,
    build_probe_document,
    verify_qwen38_causal_mount,
    verify_probe_document,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "artifacts" / "private" / "qwen3.8-cache"
INPUT_SCHEMA = "immer.qwen3.8-direct-decode-input/v1"
PREFIX_SCHEMA = "immer.qwen3.8-direct-prefix/v1"
DECODE_SCHEMA = "immer.qwen3.8-direct-decode/v1"
COMPARISON_SCHEMA = "immer.qwen3.8-direct-decode-comparison/v1"
FERTIG_INPUT_SCHEMA = "immer.qwen3.8-fertig-draft-inputs/v1"
BRANCH_INPUT_SCHEMA = "immer.qwen3.8-generation-branch-input/v2"
BRANCH_INPUT_SCHEMA_V3 = "immer.qwen3.8-generation-branch-input/v3"
BRANCH_RESULT_SCHEMA = "immer.qwen3.8-generation-branch-arm/v2"
BRANCH_COMPARISON_SCHEMA = "immer.qwen3.8-generation-branch-comparison/v2"
BRANCH_EVALUATION_SCHEMA = "immer.qwen3.8-generation-branch-evaluation/v2"
NATIVE_BRANCH_RESULT_SCHEMA = "immer.qwen3.8-generation-branch-arm/v3"
TRIAD_COMPARISON_SCHEMA = "immer.qwen3.8-generation-branch-comparison/v3"
TRIAD_EVALUATION_SCHEMA = "immer.qwen3.8-generation-branch-evaluation/v3"
NATIVE_FORK_PAIR_SCHEMA = "immer.qwen3.8-generation-native-fork-pair/v1"
NATIVE_FORK_REFERENCE_COMPARISON_SCHEMA = (
    "immer.qwen3.8-generation-native-fork-reference-comparison/v1"
)
BRANCH_QUESTION_SCHEMA = "immer.qwen3.8-generation-branch-questions/v1"
BRANCH_ADJUDICATION_SCHEMA = "immer.qwen3.8-generation-branch-adjudication/v1"
BRANCH_ADJUDICATION_EVALUATION_SCHEMA = (
    "immer.qwen3.8-generation-branch-adjudication-evaluation/v1"
)
TOKEN_CHAIN_SCHEMA = "immer.qwen3.8-autoregressive-token-chain/v1"
EXTERNAL_RAW_SEAL_KIND = "external-raw-file-sha256/v1"

# Explicit aliases make the additive schema generation unambiguous while the
# historic BRANCH_* constants remain byte-compatible v2 identities.
BRANCH_RESULT_SCHEMA_V3 = NATIVE_BRANCH_RESULT_SCHEMA
BRANCH_RESULT_V3_SCHEMA = NATIVE_BRANCH_RESULT_SCHEMA
BRANCH_TRIAD_COMPARISON_SCHEMA = TRIAD_COMPARISON_SCHEMA
BRANCH_TRIAD_EVALUATION_SCHEMA = TRIAD_EVALUATION_SCHEMA

NATIVE_CRSA_ALPHA = 0.01
NATIVE_CRSA_BALANCE_ALPHA = 1.0
NATIVE_CRSA_DIAGONAL_DEBIT = 3.0

ANSWER_GENERATION_PREFIX_LITERAL = "#### "
ANSWER_GENERATION_PREFIX_KIND = "fixed-answer-value-prefix/v1"
OFFICIAL_ANSWER_GENERATION_PREFIX_TOKEN_IDS = (794, 220)
ANSWER_OUTPUT_INSTRUCTION = (
    "Solve the math problem internally. Return only #### followed by the numeric "
    "answer. Do not show work."
)

_SEAL_FIELDS = ("sha256", "report_sha256", "document_sha256")
_FORBIDDEN_BRANCH_INPUT_KEYS = frozenset(
    {
        "answer",
        "candidate_correct",
        "correct",
        "correctness",
        "decode_token_id",
        "draft_token_ids",
        "gold",
        "predicted",
    }
)


class QwenDirectDecodeError(RuntimeError):
    """The shared-prefix Qwen decode contract cannot be satisfied."""


@dataclass(slots=True)
class RuntimeSource:
    source: Streamer
    mount: CausalWeightMount | None = None
    verification: dict[str, Any] | None = None

    def close(self) -> None:
        if self.mount is not None:
            self.mount.close()
        else:
            self.source.close()


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QwenDirectDecodeError("value is not canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _strict_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()

    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in entries:
            if key in result:
                raise QwenDirectDecodeError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        document = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=pairs
        )
    except QwenDirectDecodeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QwenDirectDecodeError(f"cannot read JSON: {source}") from exc
    if not isinstance(document, dict):
        raise QwenDirectDecodeError("JSON root must be an object")
    return document


def _externally_sealed_json(
    path: str | os.PathLike[str], expected_sha256: object, label: str
) -> tuple[dict[str, Any], str]:
    """Read one file once, authenticate its raw bytes, then parse strict JSON."""

    expected = _digest_string(expected_sha256, f"{label} expected raw file")
    source = Path(path).expanduser().resolve()

    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in entries:
            if key in result:
                raise QwenDirectDecodeError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise QwenDirectDecodeError(f"cannot read {label}: {source}") from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise QwenDirectDecodeError(f"{label} raw file SHA-256 mismatch")
    try:
        document = json.loads(raw, object_pairs_hook=pairs)
    except QwenDirectDecodeError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QwenDirectDecodeError(f"cannot read JSON: {source}") from exc
    if not isinstance(document, dict):
        raise QwenDirectDecodeError("JSON root must be an object")
    return document, actual


def _atomic_bytes(path: str | os.PathLike[str], value: bytes) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".pending", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def _write_json(path: str | os.PathLike[str], document: Mapping[str, Any]) -> Path:
    return _atomic_bytes(path, _canonical(dict(document)) + b"\n")


def _result(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {**dict(identity), "sha256": _sha256(identity)}


def _verify_document_seal(document: Mapping[str, Any], label: str) -> tuple[str, str]:
    present = [name for name in _SEAL_FIELDS if name in document]
    if len(present) != 1:
        raise QwenDirectDecodeError(f"{label} must have one canonical document seal")
    field = present[0]
    claimed = document.get(field)
    if not isinstance(claimed, str) or not re.fullmatch(r"[0-9a-f]{64}", claimed):
        raise QwenDirectDecodeError(f"{label} document seal is invalid")
    unsealed = dict(document)
    del unsealed[field]
    if claimed != _sha256(unsealed):
        raise QwenDirectDecodeError(f"{label} document seal mismatch")
    return field, claimed


def _contains_forbidden_branch_input_key(value: object) -> bool:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower()
            if key in _FORBIDDEN_BRANCH_INPUT_KEYS:
                return True
            if _contains_forbidden_branch_input_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_branch_input_key(child) for child in value)
    return False


def _nonnegative_count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QwenDirectDecodeError(f"{label} must be a non-negative integer")
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QwenDirectDecodeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise QwenDirectDecodeError(f"{label} must be finite and non-negative")
    return result


def _digest_string(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise QwenDirectDecodeError(f"{label} must be a SHA-256 digest")
    return value


def _token_chain_sha256(
    prompt_token_ids: Sequence[int], generated_token_ids: Sequence[int]
) -> str:
    return _sha256(
        {
            "generated_token_ids": list(generated_token_ids),
            "prompt_length": len(prompt_token_ids),
            "prompt_token_ids": list(prompt_token_ids),
            "schema": TOKEN_CHAIN_SCHEMA,
        }
    )


def _hidden_sha256(hidden: torch.Tensor) -> str:
    cpu = hidden.detach().to(device="cpu").contiguous()
    return hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()


def _token_rows(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise QwenDirectDecodeError(f"{label} must be a non-empty token list")
    tokens: list[int] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise QwenDirectDecodeError(f"{label} contains an invalid token")
        tokens.append(raw)
    return tuple(tokens)


def _source_rows(document: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if document.get("schema") != FERTIG_INPUT_SCHEMA:
        raise QwenDirectDecodeError("branch source input schema is invalid")
    source = document.get("source")
    protocol = document.get("protocol")
    raw_rows = document.get("items")
    if (
        not isinstance(source, Mapping)
        or not isinstance(protocol, Mapping)
        or not isinstance(raw_rows, list)
        or not raw_rows
    ):
        raise QwenDirectDecodeError("branch source input structure is invalid")
    checkpoint = source.get("checkpoint")
    revision = source.get("revision")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise QwenDirectDecodeError("branch source checkpoint is invalid")
    if not isinstance(revision, str) or not revision:
        raise QwenDirectDecodeError("branch source revision is invalid")
    rows: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise QwenDirectDecodeError(f"branch source row {index} is not an object")
        item_id = raw.get("item_id")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise QwenDirectDecodeError(
                "branch source item IDs are invalid or duplicated"
            )
        _token_rows(raw.get("prompt_token_ids"), f"source prompt {item_id}")
        seen.add(item_id)
        rows.append(raw)
    batch_size = protocol.get("batch_size")
    if batch_size is not None and batch_size != len(rows):
        raise QwenDirectDecodeError("branch source batch size is inconsistent")
    protocol_ids = protocol.get("item_ids")
    if protocol_ids is not None and protocol_ids != [row["item_id"] for row in rows]:
        raise QwenDirectDecodeError("branch source item order is inconsistent")
    return tuple(rows)


def _source_contract_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    """Project only generation inputs; labels cannot influence this digest."""

    rows = _source_rows(document)
    source = document["source"]
    protocol = document["protocol"]
    eos = _token_rows(protocol.get("accepted_eos_token_ids"), "accepted_eos_token_ids")
    if len(set(eos)) != len(eos):
        raise QwenDirectDecodeError("accepted EOS token IDs are duplicated")
    return {
        "items": [
            {
                "item_id": str(row["item_id"]),
                "prompt_token_ids": list(
                    _token_rows(row["prompt_token_ids"], "prompt_token_ids")
                ),
            }
            for row in rows
        ],
        "protocol": {
            "accepted_eos_token_ids": list(eos),
            "system_prompt": protocol.get("system_prompt"),
            "thinking": protocol.get("thinking"),
        },
        "schema": FERTIG_INPUT_SCHEMA,
        "source": {
            "checkpoint": source["checkpoint"],
            "revision": source["revision"],
        },
    }


def _tokenizer_record(
    path: str | os.PathLike[str],
    *,
    require_official: bool,
    tokenizer_factory: Callable[..., Qwen38Tokenizer] = Qwen38Tokenizer,
) -> tuple[Qwen38Tokenizer, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise QwenDirectDecodeError("tokenizer JSON is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise QwenDirectDecodeError("tokenizer JSON must be a plain regular file")
    try:
        tokenizer = tokenizer_factory(source, require_official=require_official)
        vocab_size = int(tokenizer.backend.get_vocab_size(with_added_tokens=True))
        end_of_text = tokenizer.backend.token_to_id("<|endoftext|>")
        im_end = tokenizer.backend.token_to_id("<|im_end|>")
    except Exception as exc:
        raise QwenDirectDecodeError("tokenizer identity cannot be verified") from exc
    if (
        vocab_size < 1
        or isinstance(end_of_text, bool)
        or not isinstance(end_of_text, int)
        or isinstance(im_end, bool)
        or not isinstance(im_end, int)
    ):
        raise QwenDirectDecodeError("tokenizer control-token identity is invalid")
    if require_official and (
        end_of_text != END_OF_TEXT_TOKEN_ID or im_end != IM_END_TOKEN_ID
    ):
        raise QwenDirectDecodeError("tokenizer control-token IDs are not official")
    record = {
        "control_token_ids": {
            "end_of_text": end_of_text,
            "im_end": im_end,
        },
        "kind": "tokenizers-json/v1",
        "require_official": require_official,
        "sha256": _sha256_file(source),
        "size_bytes": metadata.st_size,
        "vocab_size": vocab_size,
    }
    return tokenizer, record


def select_branch_cohort(
    args: argparse.Namespace,
    *,
    tokenizer_factory: Callable[..., Qwen38Tokenizer] = Qwen38Tokenizer,
) -> dict[str, Any]:
    """Seal an ordered, label-free generation cohort from one sealed Dev input."""

    document, raw_file_sha256 = _externally_sealed_json(
        args.inputs,
        args.inputs_sha256,
        "branch source input",
    )
    projection = _source_contract_projection(document)
    rows = projection["items"]
    offset = int(args.offset)
    limit = int(args.limit)
    if offset + limit > len(rows):
        raise QwenDirectDecodeError("branch cohort slice exceeds source rows")
    selected = rows[offset : offset + limit]
    tokenizer, tokenizer_identity = _tokenizer_record(
        args.tokenizer_json,
        require_official=getattr(args, "_require_official", True),
        tokenizer_factory=tokenizer_factory,
    )
    del tokenizer
    eos = projection["protocol"]["accepted_eos_token_ids"]
    controls = tokenizer_identity["control_token_ids"]
    if set(eos) != {controls["im_end"], controls["end_of_text"]}:
        raise QwenDirectDecodeError("source EOS set differs from tokenizer identity")
    vocab_size = int(tokenizer_identity["vocab_size"])
    for row in selected:
        if max(row["prompt_token_ids"]) >= vocab_size:
            raise QwenDirectDecodeError("branch prompt exceeds tokenizer vocabulary")
    identity = {
        "items": selected,
        "protocol": {
            "accepted_eos_token_ids": eos,
            "execution_mode": "serial_items/shared_weight_pager",
            "independent_prefills": True,
            "teacher_forced_tokens_after_prompt": 0,
        },
        "schema": BRANCH_INPUT_SCHEMA,
        "selection": {
            "item_ids": [row["item_id"] for row in selected],
            "kind": "ordered-slice/v1",
            "limit": limit,
            "offset": offset,
            "source_items": len(rows),
        },
        "source": {
            "checkpoint": projection["source"]["checkpoint"],
            "contract_sha256": _sha256(projection),
            "raw_file_sha256": raw_file_sha256,
            "revision": projection["source"]["revision"],
            "schema": FERTIG_INPUT_SCHEMA,
            "seal_kind": EXTERNAL_RAW_SEAL_KIND,
        },
        "status": "sealed",
        "tokenizer": tokenizer_identity,
    }
    if _contains_forbidden_branch_input_key(identity):  # pragma: no cover
        raise QwenDirectDecodeError("branch cohort contains forbidden label fields")
    return _result(identity)


def _validate_answer_source_protocol(projection: Mapping[str, Any]) -> None:
    protocol = projection.get("protocol")
    system_prompt = (
        protocol.get("system_prompt") if isinstance(protocol, Mapping) else None
    )
    instruction = system_prompt.lower() if isinstance(system_prompt, str) else ""
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("thinking") is not False
        or not system_prompt
        or "return" not in instruction
        or "####" not in system_prompt
        or "answer" not in instruction
    ):
        raise QwenDirectDecodeError(
            "answer branch source must disable thinking and use the explicit #### "
            "output instruction"
        )


def _answer_generation_prefix(
    tokenizer: Qwen38Tokenizer,
    tokenizer_identity: Mapping[str, Any],
) -> tuple[int, ...]:
    try:
        token_ids = tokenizer.encode(ANSWER_GENERATION_PREFIX_LITERAL)
    except Exception as exc:
        raise QwenDirectDecodeError(
            "answer generation prefix cannot be encoded"
        ) from exc
    if not token_ids:
        raise QwenDirectDecodeError("answer generation prefix encoded to no tokens")
    vocab_size = int(tokenizer_identity["vocab_size"])
    if max(token_ids) >= vocab_size:
        raise QwenDirectDecodeError(
            "answer generation prefix exceeds tokenizer vocabulary"
        )
    try:
        decoded = tokenizer.decode(token_ids)
    except Exception as exc:
        raise QwenDirectDecodeError(
            "answer generation prefix cannot be decoded"
        ) from exc
    if decoded != ANSWER_GENERATION_PREFIX_LITERAL:
        raise QwenDirectDecodeError(
            "answer generation prefix does not round-trip through pinned tokenizer"
        )
    if tokenizer_identity.get("require_official") is True and token_ids != (
        OFFICIAL_ANSWER_GENERATION_PREFIX_TOKEN_IDS
    ):
        raise QwenDirectDecodeError(
            "official answer generation prefix must encode to [794, 220]"
        )
    return token_ids


def _answer_generation_prefix_record(token_ids: Sequence[int]) -> dict[str, Any]:
    return {
        "kind": ANSWER_GENERATION_PREFIX_KIND,
        "label_free": True,
        "literal": ANSWER_GENERATION_PREFIX_LITERAL,
        "prefix_is_part_of_sealed_prompt": True,
        "teacher_forced_tokens_after_prompt": 0,
        "token_ids": list(token_ids),
    }


def select_answer_branch_cohort(
    args: argparse.Namespace,
    *,
    tokenizer_factory: Callable[..., Qwen38Tokenizer] = Qwen38Tokenizer,
) -> dict[str, Any]:
    """Seal value-generation prompts with the one fixed label-free prefix."""

    document, raw_file_sha256 = _externally_sealed_json(
        args.inputs,
        args.inputs_sha256,
        "answer branch source input",
    )
    projection = _source_contract_projection(document)
    _validate_answer_source_protocol(projection)
    rows = projection["items"]
    offset = int(args.offset)
    limit = int(args.limit)
    if offset + limit > len(rows):
        raise QwenDirectDecodeError("answer branch cohort slice exceeds source rows")
    selected = rows[offset : offset + limit]
    tokenizer, tokenizer_identity = _tokenizer_record(
        args.tokenizer_json,
        require_official=getattr(args, "_require_official", True),
        tokenizer_factory=tokenizer_factory,
    )
    prefix = _answer_generation_prefix(tokenizer, tokenizer_identity)
    eos = projection["protocol"]["accepted_eos_token_ids"]
    controls = tokenizer_identity["control_token_ids"]
    if set(eos) != {controls["im_end"], controls["end_of_text"]}:
        raise QwenDirectDecodeError("source EOS set differs from tokenizer identity")
    vocab_size = int(tokenizer_identity["vocab_size"])
    items: list[dict[str, Any]] = []
    for row in selected:
        source_prompt = list(row["prompt_token_ids"])
        effective_prompt = [*source_prompt, *prefix]
        if max(effective_prompt) >= vocab_size:
            raise QwenDirectDecodeError(
                "answer branch prompt exceeds tokenizer vocabulary"
            )
        items.append(
            {
                "effective_prompt_token_ids": effective_prompt,
                "item_id": row["item_id"],
                "source_prompt_token_ids": source_prompt,
            }
        )
    identity = {
        "items": items,
        "protocol": {
            "accepted_eos_token_ids": eos,
            "execution_mode": "serial_items/shared_weight_pager",
            "generation_prefix": _answer_generation_prefix_record(prefix),
            "independent_prefills": True,
            "teacher_forced_tokens_after_prompt": 0,
        },
        "schema": BRANCH_INPUT_SCHEMA_V3,
        "selection": {
            "item_ids": [row["item_id"] for row in selected],
            "kind": "ordered-slice/v1",
            "limit": limit,
            "offset": offset,
            "source_items": len(rows),
        },
        "source": {
            "checkpoint": projection["source"]["checkpoint"],
            "contract_sha256": _sha256(projection),
            "raw_file_sha256": raw_file_sha256,
            "revision": projection["source"]["revision"],
            "schema": FERTIG_INPUT_SCHEMA,
            "seal_kind": EXTERNAL_RAW_SEAL_KIND,
        },
        "status": "sealed",
        "tokenizer": tokenizer_identity,
    }
    if _contains_forbidden_branch_input_key(identity):  # pragma: no cover
        raise QwenDirectDecodeError("answer branch cohort contains forbidden fields")
    return _result(identity)


def _selected_question_rows(
    source: Mapping[str, Any], branch_input: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    """Bind source questions to one sealed answer-prefix selection without labels."""

    if branch_input.get("schema") != BRANCH_INPUT_SCHEMA_V3:
        raise QwenDirectDecodeError(
            "question projection requires a sealed answer-prefix branch input"
        )
    projection = _source_contract_projection(source)
    if _sha256(projection) != branch_input["source"]["contract_sha256"]:
        raise QwenDirectDecodeError("question source generation contract differs")
    _validate_answer_source_protocol(projection)
    selection = branch_input["selection"]
    expected_items = projection["items"][
        selection["offset"] : selection["offset"] + selection["limit"]
    ]
    prefix = tuple(branch_input["protocol"]["generation_prefix"]["token_ids"])
    sealed_items = branch_input["items"]
    if selection["source_items"] != len(projection["items"]) or len(
        sealed_items
    ) != len(expected_items):
        raise QwenDirectDecodeError(
            "sealed answer branch selection differs from question source"
        )
    for sealed, expected in zip(sealed_items, expected_items, strict=True):
        source_prompt = list(expected["prompt_token_ids"])
        if (
            sealed["item_id"] != expected["item_id"]
            or sealed["source_prompt_token_ids"] != source_prompt
            or sealed["effective_prompt_token_ids"] != [*source_prompt, *prefix]
        ):
            raise QwenDirectDecodeError(
                "sealed answer prompts differ from question source"
            )
    by_id = {row["item_id"]: row for row in _source_rows(source)}
    rows: list[Mapping[str, Any]] = []
    for item_id in selection["item_ids"]:
        row = by_id.get(item_id)
        if row is None:
            raise QwenDirectDecodeError("question source is missing a selected item")
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise QwenDirectDecodeError(
                "question source contains an invalid original question"
            )
        rows.append(row)
    return rows


def select_answer_branch_questions(args: argparse.Namespace) -> dict[str, Any]:
    """Project original questions into a label-free exact-adjudication artifact."""

    branch_input = _load_branch_input(args.input)
    if branch_input["schema"] != BRANCH_INPUT_SCHEMA_V3:
        raise QwenDirectDecodeError(
            "question projection requires a sealed answer-prefix branch input"
        )
    expected_source_sha256 = _digest_string(
        args.inputs_sha256, "question source externally pinned raw file"
    )
    if expected_source_sha256 != branch_input["source"]["raw_file_sha256"]:
        raise QwenDirectDecodeError(
            "question source external SHA-256 differs from sealed branch input"
        )
    source, source_raw_sha256 = _externally_sealed_json(
        args.inputs,
        expected_source_sha256,
        "question source",
    )
    rows = _selected_question_rows(source, branch_input)
    projection = _source_contract_projection(source)
    items = [
        {
            "item_id": row["item_id"],
            "question": row["question"],
            "question_sha256": hashlib.sha256(
                row["question"].encode("utf-8")
            ).hexdigest(),
        }
        for row in rows
    ]
    identity = {
        "branch_input_sha256": branch_input["sha256"],
        "items": items,
        "protocol": {
            "projection_contains_label_fields": False,
            "rendering": "qwen3.8-no-thinking/v1",
            "role": "pre-gold-exact-semantic-adjudication/v1",
            "source_file_may_contain_labels": True,
            "system_prompt": projection["protocol"]["system_prompt"],
            "thinking": False,
        },
        "schema": BRANCH_QUESTION_SCHEMA,
        "selection": dict(branch_input["selection"]),
        "source": {
            "contract_sha256": branch_input["source"]["contract_sha256"],
            "raw_file_sha256": source_raw_sha256,
            "schema": source["schema"],
            "seal_kind": EXTERNAL_RAW_SEAL_KIND,
        },
        "status": "sealed",
    }
    if _contains_forbidden_branch_input_key(identity):  # pragma: no cover
        raise QwenDirectDecodeError("question projection contains forbidden labels")
    return _validate_branch_question_document(_result(identity))


def _validate_branch_input_document(document: dict[str, Any]) -> dict[str, Any]:
    required = {
        "items",
        "protocol",
        "schema",
        "selection",
        "sha256",
        "source",
        "status",
        "tokenizer",
    }
    if set(document) != required or document.get("schema") != BRANCH_INPUT_SCHEMA:
        raise QwenDirectDecodeError("branch input schema is invalid")
    _verify_document_seal(document, "branch input")
    if document.get("status") != "sealed":
        raise QwenDirectDecodeError("branch input is not sealed")
    if _contains_forbidden_branch_input_key(document):
        raise QwenDirectDecodeError("branch input contains label or draft fields")
    source = document.get("source")
    selection = document.get("selection")
    protocol = document.get("protocol")
    tokenizer = document.get("tokenizer")
    rows = document.get("items")
    if not all(
        isinstance(value, Mapping) for value in (source, selection, protocol, tokenizer)
    ) or not isinstance(rows, list):
        raise QwenDirectDecodeError("branch input structure is invalid")
    if set(source) != {
        "checkpoint",
        "contract_sha256",
        "raw_file_sha256",
        "revision",
        "schema",
        "seal_kind",
    }:
        raise QwenDirectDecodeError("branch input source identity is invalid")
    if (
        source.get("schema") != FERTIG_INPUT_SCHEMA
        or source.get("seal_kind") != EXTERNAL_RAW_SEAL_KIND
        or not isinstance(source.get("checkpoint"), str)
        or not source["checkpoint"]
        or not isinstance(source.get("revision"), str)
        or not source["revision"]
    ):
        raise QwenDirectDecodeError("branch input source contract is invalid")
    _digest_string(source.get("contract_sha256"), "source contract")
    _digest_string(source.get("raw_file_sha256"), "source raw file")
    if (
        set(selection)
        != {
            "item_ids",
            "kind",
            "limit",
            "offset",
            "source_items",
        }
        or selection.get("kind") != "ordered-slice/v1"
    ):
        raise QwenDirectDecodeError("branch selection contract is invalid")
    offset = _nonnegative_count(selection.get("offset"), "selection offset")
    limit = _nonnegative_count(selection.get("limit"), "selection limit")
    source_items = _nonnegative_count(
        selection.get("source_items"), "source item count"
    )
    if limit < 1 or len(rows) != limit or offset + limit > source_items:
        raise QwenDirectDecodeError("branch selection bounds are invalid")
    item_ids = selection.get("item_ids")
    if not isinstance(item_ids, list) or len(item_ids) != limit:
        raise QwenDirectDecodeError("branch selection item IDs are invalid")
    if set(protocol) != {
        "accepted_eos_token_ids",
        "execution_mode",
        "independent_prefills",
        "teacher_forced_tokens_after_prompt",
    }:
        raise QwenDirectDecodeError("branch generation protocol is invalid")
    eos = _token_rows(protocol.get("accepted_eos_token_ids"), "accepted EOS IDs")
    if (
        len(set(eos)) != len(eos)
        or protocol.get("execution_mode") != "serial_items/shared_weight_pager"
        or protocol.get("independent_prefills") is not True
        or protocol.get("teacher_forced_tokens_after_prompt") != 0
    ):
        raise QwenDirectDecodeError("branch generation protocol is inconsistent")
    tokenizer_required = {
        "control_token_ids",
        "kind",
        "require_official",
        "sha256",
        "size_bytes",
        "vocab_size",
    }
    if (
        set(tokenizer) != tokenizer_required
        or tokenizer.get("kind") != "tokenizers-json/v1"
    ):
        raise QwenDirectDecodeError("branch tokenizer identity is invalid")
    _digest_string(tokenizer.get("sha256"), "tokenizer")
    vocab_size = _nonnegative_count(tokenizer.get("vocab_size"), "tokenizer vocabulary")
    _nonnegative_count(tokenizer.get("size_bytes"), "tokenizer size")
    controls = tokenizer.get("control_token_ids")
    if (
        vocab_size < 1
        or not isinstance(tokenizer.get("require_official"), bool)
        or not isinstance(controls, Mapping)
        or set(controls) != {"end_of_text", "im_end"}
        or set(eos) != {controls.get("end_of_text"), controls.get("im_end")}
    ):
        raise QwenDirectDecodeError("branch tokenizer control tokens are invalid")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {"item_id", "prompt_token_ids"}:
            raise QwenDirectDecodeError("branch item schema is invalid")
        item_id = row.get("item_id")
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in seen
            or item_id != item_ids[index]
        ):
            raise QwenDirectDecodeError("branch item order or identity is invalid")
        tokens = _token_rows(row.get("prompt_token_ids"), f"prompt {item_id}")
        if max(tokens) >= vocab_size:
            raise QwenDirectDecodeError("branch prompt exceeds tokenizer vocabulary")
        seen.add(item_id)
    return document


def _answer_branch_v2_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    projected = {key: value for key, value in document.items() if key != "sha256"}
    projected["items"] = [
        {
            "item_id": row["item_id"],
            "prompt_token_ids": row["effective_prompt_token_ids"],
        }
        for row in document["items"]
    ]
    projected["protocol"] = {
        key: value
        for key, value in document["protocol"].items()
        if key != "generation_prefix"
    }
    projected["schema"] = BRANCH_INPUT_SCHEMA
    return _result(projected)


def _validate_answer_branch_input_document(
    document: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "items",
        "protocol",
        "schema",
        "selection",
        "sha256",
        "source",
        "status",
        "tokenizer",
    }
    if set(document) != required or document.get("schema") != BRANCH_INPUT_SCHEMA_V3:
        raise QwenDirectDecodeError("answer branch input schema is invalid")
    _verify_document_seal(document, "answer branch input")
    if document.get("status") != "sealed":
        raise QwenDirectDecodeError("answer branch input is not sealed")
    if _contains_forbidden_branch_input_key(document):
        raise QwenDirectDecodeError(
            "answer branch input contains label or draft fields"
        )
    protocol = document.get("protocol")
    tokenizer = document.get("tokenizer")
    rows = document.get("items")
    if (
        not isinstance(protocol, Mapping)
        or not isinstance(tokenizer, Mapping)
        or not isinstance(rows, list)
    ):
        raise QwenDirectDecodeError("answer branch input structure is invalid")
    if set(protocol) != {
        "accepted_eos_token_ids",
        "execution_mode",
        "generation_prefix",
        "independent_prefills",
        "teacher_forced_tokens_after_prompt",
    }:
        raise QwenDirectDecodeError("answer branch generation protocol is invalid")
    prefix = protocol.get("generation_prefix")
    if not isinstance(prefix, Mapping) or set(prefix) != {
        "kind",
        "label_free",
        "literal",
        "prefix_is_part_of_sealed_prompt",
        "teacher_forced_tokens_after_prompt",
        "token_ids",
    }:
        raise QwenDirectDecodeError("answer generation prefix record is invalid")
    prefix_tokens = _token_rows(
        prefix.get("token_ids"), "answer generation prefix tokens"
    )
    if (
        prefix.get("kind") != ANSWER_GENERATION_PREFIX_KIND
        or prefix.get("literal") != ANSWER_GENERATION_PREFIX_LITERAL
        or prefix.get("label_free") is not True
        or prefix.get("prefix_is_part_of_sealed_prompt") is not True
        or prefix.get("teacher_forced_tokens_after_prompt") != 0
        or protocol.get("teacher_forced_tokens_after_prompt") != 0
    ):
        raise QwenDirectDecodeError("answer generation prefix contract is invalid")
    if tokenizer.get("require_official") is True and prefix_tokens != (
        OFFICIAL_ANSWER_GENERATION_PREFIX_TOKEN_IDS
    ):
        raise QwenDirectDecodeError("official answer generation prefix IDs are invalid")
    selection = document.get("selection")
    item_ids = selection.get("item_ids") if isinstance(selection, Mapping) else None
    vocab_size = tokenizer.get("vocab_size")
    if (
        not isinstance(item_ids, list)
        or not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size < 1
        or len(rows) != len(item_ids)
    ):
        raise QwenDirectDecodeError("answer branch item identity is invalid")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "effective_prompt_token_ids",
            "item_id",
            "source_prompt_token_ids",
        }:
            raise QwenDirectDecodeError("answer branch item schema is invalid")
        item_id = row.get("item_id")
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in seen
            or item_id != item_ids[index]
        ):
            raise QwenDirectDecodeError(
                "answer branch item order or identity is invalid"
            )
        source_prompt = _token_rows(
            row.get("source_prompt_token_ids"), f"source prompt {item_id}"
        )
        effective_prompt = _token_rows(
            row.get("effective_prompt_token_ids"), f"effective prompt {item_id}"
        )
        if effective_prompt != (*source_prompt, *prefix_tokens):
            raise QwenDirectDecodeError(
                "answer branch effective prompt does not equal source prompt plus prefix"
            )
        if max(effective_prompt) >= vocab_size:
            raise QwenDirectDecodeError(
                "answer branch prompt exceeds tokenizer vocabulary"
            )
        seen.add(item_id)
    _validate_branch_input_document(_answer_branch_v2_projection(document))
    return document


def _validate_branch_input(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema") == BRANCH_INPUT_SCHEMA:
        return _validate_branch_input_document(document)
    if document.get("schema") == BRANCH_INPUT_SCHEMA_V3:
        return _validate_answer_branch_input_document(document)
    raise QwenDirectDecodeError("branch input schema is invalid")


def _load_branch_input(path: str | os.PathLike[str]) -> dict[str, Any]:
    return _validate_branch_input(_strict_json(path))


def _validate_branch_question_document(
    document: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "branch_input_sha256",
        "items",
        "protocol",
        "schema",
        "selection",
        "sha256",
        "source",
        "status",
    }
    if (
        set(document) != required
        or document.get("schema") != BRANCH_QUESTION_SCHEMA
        or document.get("status") != "sealed"
    ):
        raise QwenDirectDecodeError("branch question document schema is invalid")
    _verify_document_seal(document, "branch question document")
    _digest_string(document.get("branch_input_sha256"), "branch question input")
    if _contains_forbidden_branch_input_key(document):
        raise QwenDirectDecodeError("branch question document contains label fields")
    protocol = document.get("protocol")
    source = document.get("source")
    selection = document.get("selection")
    items = document.get("items")
    if (
        not isinstance(protocol, Mapping)
        or not isinstance(source, Mapping)
        or not isinstance(selection, Mapping)
        or not isinstance(items, list)
    ):
        raise QwenDirectDecodeError("branch question document structure is invalid")
    if set(protocol) != {
        "projection_contains_label_fields",
        "rendering",
        "role",
        "source_file_may_contain_labels",
        "system_prompt",
        "thinking",
    } or (
        protocol.get("projection_contains_label_fields") is not False
        or protocol.get("rendering") != "qwen3.8-no-thinking/v1"
        or protocol.get("role") != "pre-gold-exact-semantic-adjudication/v1"
        or protocol.get("source_file_may_contain_labels") is not True
        or not isinstance(protocol.get("system_prompt"), str)
        or protocol.get("thinking") is not False
    ):
        raise QwenDirectDecodeError("branch question protocol is invalid")
    if set(source) != {
        "contract_sha256",
        "raw_file_sha256",
        "schema",
        "seal_kind",
    } or (
        source.get("schema") != FERTIG_INPUT_SCHEMA
        or source.get("seal_kind") != EXTERNAL_RAW_SEAL_KIND
    ):
        raise QwenDirectDecodeError("branch question source identity is invalid")
    _digest_string(source.get("contract_sha256"), "branch question source contract")
    _digest_string(source.get("raw_file_sha256"), "branch question raw source")
    if (
        set(selection)
        != {
            "item_ids",
            "kind",
            "limit",
            "offset",
            "source_items",
        }
        or selection.get("kind") != "ordered-slice/v1"
    ):
        raise QwenDirectDecodeError("branch question selection is invalid")
    offset = _nonnegative_count(selection.get("offset"), "question selection offset")
    limit = _nonnegative_count(selection.get("limit"), "question selection limit")
    source_items = _nonnegative_count(
        selection.get("source_items"), "question source item count"
    )
    item_ids = selection.get("item_ids")
    if (
        limit < 1
        or offset + limit > source_items
        or len(items) != limit
        or not isinstance(item_ids, list)
        or len(item_ids) != limit
    ):
        raise QwenDirectDecodeError("branch question selection bounds are invalid")
    seen: set[str] = set()
    for index, row in enumerate(items):
        if not isinstance(row, Mapping) or set(row) != {
            "item_id",
            "question",
            "question_sha256",
        }:
            raise QwenDirectDecodeError("branch question item schema is invalid")
        item_id = row.get("item_id")
        question = row.get("question")
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in seen
            or item_id != item_ids[index]
            or not isinstance(question, str)
            or not question.strip()
        ):
            raise QwenDirectDecodeError("branch question item identity is invalid")
        question_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
        if row.get("question_sha256") != question_sha256:
            raise QwenDirectDecodeError("branch question digest mismatch")
        seen.add(item_id)
    return document


def _load_branch_question_document(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    return _validate_branch_question_document(_strict_json(path))


def _verify_question_prompt_bindings(
    questions: Mapping[str, Any],
    branch_input: Mapping[str, Any],
    tokenizer_path: str | os.PathLike[str],
) -> None:
    """Prove every projected question reproduces its sealed source prompt."""

    require_official = branch_input["tokenizer"]["require_official"]
    tokenizer, identity = _tokenizer_record(
        tokenizer_path,
        require_official=require_official,
    )
    if identity != branch_input["tokenizer"]:
        raise QwenDirectDecodeError(
            "question tokenizer differs from the sealed branch tokenizer"
        )
    system_prompt = questions["protocol"]["system_prompt"]
    question_rows = questions["items"]
    prompt_rows = branch_input["items"]
    if len(question_rows) != len(prompt_rows):
        raise QwenDirectDecodeError("question and sealed prompt counts differ")
    for question_row, prompt_row in zip(question_rows, prompt_rows, strict=True):
        if question_row["item_id"] != prompt_row["item_id"]:
            raise QwenDirectDecodeError("question and sealed prompt order differs")
        rendered = tokenizer.render_no_thinking_prompt(
            system_prompt,
            question_row["question"],
        )
        if list(tokenizer.encode(rendered)) != prompt_row["source_prompt_token_ids"]:
            raise QwenDirectDecodeError(
                "question text does not reproduce the sealed source prompt"
            )


def _load_input(path: str | os.PathLike[str]) -> dict[str, Any]:
    document = _strict_json(path)
    if document.get("schema") != INPUT_SCHEMA or set(document) != {
        "decode_token_id",
        "item_id",
        "prefix_token_ids",
        "schema",
        "sha256",
    }:
        raise QwenDirectDecodeError("direct-decode input schema is invalid")
    identity = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != _sha256(identity):
        raise QwenDirectDecodeError("direct-decode input SHA-256 mismatch")
    if not isinstance(document.get("item_id"), str) or not document["item_id"]:
        raise QwenDirectDecodeError("direct-decode item_id is invalid")
    prefix = _token_rows(document.get("prefix_token_ids"), "prefix_token_ids")
    decode = document.get("decode_token_id")
    if isinstance(decode, bool) or not isinstance(decode, int) or decode < 0:
        raise QwenDirectDecodeError("decode_token_id is invalid")
    return {**document, "prefix_token_ids": list(prefix)}


def select_input(args: argparse.Namespace) -> dict[str, Any]:
    document = _strict_json(args.inputs)
    if document.get("schema") != FERTIG_INPUT_SCHEMA:
        raise QwenDirectDecodeError("Qwen FERTIG input schema is invalid")
    rows = document.get("items")
    if not isinstance(rows, list):
        raise QwenDirectDecodeError("Qwen FERTIG input rows are invalid")
    matching = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("item_id") == args.item_id
    ]
    if len(matching) != 1:
        raise QwenDirectDecodeError("item_id does not select exactly one input")
    row = matching[0]
    prefix = _token_rows(row.get("prompt_token_ids"), "prompt_token_ids")
    draft = _token_rows(row.get("draft_token_ids"), "draft_token_ids")
    identity = {
        "decode_token_id": draft[0],
        "item_id": args.item_id,
        "prefix_token_ids": list(prefix),
        "schema": INPUT_SCHEMA,
    }
    return _result(identity)


def _local_path(value: str) -> Path | None:
    path = Path(value).expanduser()
    return path.resolve() if path.exists() else None


def _pinned_inventory(
    path: str | None,
    *,
    repo_id: str,
    revision: str,
) -> tuple[Mapping[str, Any] | None, str | None]:
    if path is None:
        return None, None
    document = _strict_json(path)
    if (
        document.get("schema") != "immer.tensor-inventory-cache/v1"
        or document.get("repo_id") != repo_id
        or document.get("revision") != revision
        or not isinstance(document.get("inventory"), Mapping)
        or not isinstance(document.get("source_fingerprint"), str)
    ):
        raise QwenDirectDecodeError("pinned inventory schema is invalid")
    inventory = document["inventory"]
    if (
        inventory.get("repo") != repo_id
        or inventory.get("revision") != revision
        or document.get("inventory_sha256") != _sha256(inventory)
    ):
        raise QwenDirectDecodeError("pinned inventory identity is invalid")
    return inventory, document["source_fingerprint"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_shard_digest(shard: Mapping[str, Any]) -> str:
    values: set[str] = set()
    for raw in (shard.get("payload_sha256"), shard.get("linked_etag")):
        if raw is None:
            continue
        value = str(raw).strip().strip('"').lower()
        if re.fullmatch(r"[0-9a-f]{64}", value):
            values.add(value)
    if len(values) != 1:
        raise QwenDirectDecodeError("local shard lacks one unambiguous payload SHA-256")
    return next(iter(values))


def _verify_local_payload(
    root: Path,
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        root_metadata = root.lstat()
    except OSError as exc:
        raise QwenDirectDecodeError("local weight root is missing") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise QwenDirectDecodeError("local weight root must be a plain directory")
    receipts: list[dict[str, Any]] = []
    total = 0
    for shard in inventory.get("shards", ()):
        if not isinstance(shard, Mapping):
            raise QwenDirectDecodeError("pinned shard table is invalid")
        name = shard.get("file")
        size = shard.get("size")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
        ):
            raise QwenDirectDecodeError("pinned shard coordinate is invalid")
        path = root / name
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise QwenDirectDecodeError(f"local shard is missing: {name}") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != size
        ):
            raise QwenDirectDecodeError(f"local shard shape is invalid: {name}")
        expected = _expected_shard_digest(shard)
        actual = _sha256_file(path)
        if actual != expected:
            raise QwenDirectDecodeError(f"local shard SHA-256 mismatch: {name}")
        receipts.append({"file": name, "sha256": actual, "size": size})
        total += size
    if not receipts:
        raise QwenDirectDecodeError("pinned inventory contains no shards")
    return {
        "checkpoint_bytes": total,
        "kind": "complete-local-shards/v1",
        "shards": len(receipts),
        "shards_sha256": _sha256(receipts),
    }


def _verify_causal_mount(
    mount: CausalWeightMount, *, require_official_config: bool = True
) -> dict[str, Any]:
    try:
        return verify_qwen38_causal_mount(
            mount, require_official_config=require_official_config
        )
    except Qwen38BundleError as exc:
        raise QwenDirectDecodeError(str(exc)) from exc


def _build_source(
    args: argparse.Namespace, recorder: AccessTraceRecorder
) -> RuntimeSource:
    if args.causal_bundle is not None:
        mount = CausalWeightMount(
            args.causal_bundle,
            LogicalModelIdentity(args.logical_repo_id, args.revision),
            budget_mb=args.source_budget_mb,
        )
        try:
            verification = _verify_causal_mount(
                mount,
                require_official_config=getattr(args, "_require_official", True),
            )
            mount.source.set_access_observer(recorder)
            return RuntimeSource(mount.source, mount, verification)
        except Exception:
            mount.close()
            raise
    local = _local_path(args.source)
    common = {
        "revision": args.revision,
        "budget_mb": args.source_budget_mb,
        "cache_dir": Path(args.cache_dir).expanduser().resolve(),
        "max_cache_bytes": int(args.max_cache_gb * 1024**3),
        "access_observer": recorder,
        "verbose": False,
    }
    if local is not None:
        inventory, fingerprint = _pinned_inventory(
            args.pinned_inventory,
            repo_id=args.logical_repo_id,
            revision=args.revision,
        )
        if inventory is None or fingerprint is None:
            raise QwenDirectDecodeError(
                "local source requires a pinned inventory with shard hashes"
            )
        verification = _verify_local_payload(local, inventory)
        return RuntimeSource(
            Streamer.from_local(
                local,
                repo_id=args.logical_repo_id,
                pinned_inventory=inventory,
                pinned_fingerprint=fingerprint,
                use_cache=False,
                **common,
            ),
            verification=verification,
        )
    return RuntimeSource(
        Streamer(args.source, **common),
        verification={"kind": "remote-pinned-range-source/v1"},
    )


def _runtime(
    args: argparse.Namespace,
    recorder: AccessTraceRecorder,
    delta_probe: DeltaNetProbeRecorder | None = None,
    *,
    graft: Qwen38StableCrsaGraft | None = None,
    graft_layer: int | None = None,
    native_head_crsa: Qwen38NativeHeadCrsa | None = None,
    native_head_crsa_observer: Callable[[NativeHeadCrsaEvidence], None] | None = None,
    max_batch_size: int = 1,
) -> tuple[RuntimeSource, StreamedQwen38]:
    runtime = _build_source(args, recorder)
    pager: Qwen38WeightPager | None = None
    try:
        raw_config = runtime.source.reader.fetch_file("config.json")
        config_document = json.loads(raw_config)
        if not isinstance(config_document, Mapping):
            raise QwenDirectDecodeError("checkpoint config root is invalid")
        config = Qwen38Config.from_mapping(
            config_document,
            require_official=getattr(args, "_require_official", True),
        )
        pager = Qwen38WeightPager(
            runtime.source,
            device=args.device,
            compute_dtype=args.dtype,
            max_resident_bytes=args.max_resident_mb * 1024**2,
            require_source_identity=True,
            causal_tensor_reader=(
                None if runtime.mount is None else runtime.mount.tensor_reader
            ),
        )
        model = StreamedQwen38(
            config,
            pager,
            graft=graft,
            graft_layer=graft_layer,
            delta_probe=delta_probe,
            native_head_crsa=native_head_crsa,
            native_head_crsa_observer=native_head_crsa_observer,
            max_batch_size=max_batch_size,
            max_seq_len=args.max_seq_len,
        )
        model.checkpoint_preflight()
        return runtime, model
    except Exception:
        if pager is not None:
            pager.close()
        runtime.close()
        raise


def _progress(event: Mapping[str, Any]) -> None:
    sys.stderr.write(json.dumps(dict(event), sort_keys=True) + "\n")
    sys.stderr.flush()


def _trace_receipt(
    recorder: AccessTraceRecorder, path: str | os.PathLike[str]
) -> dict[str, Any]:
    metrics = recorder.metrics()
    if metrics["dropped_capacity"] or metrics["dropped_identity"]:
        raise QwenDirectDecodeError("access trace recorder dropped operations")
    trace = recorder.snapshot()
    trace.verify()
    output = _atomic_bytes(path, trace.to_bytes())
    return {
        "inventory_fingerprint": trace.inventory_fingerprint,
        "leaves": metrics["leaves"],
        "operations": metrics["operations"],
        "path": str(output),
        "sha256": trace.sha256,
    }


def _checkpoint(model: StreamedQwen38) -> dict[str, str]:
    source = model.pager.source
    source.inventory()
    metrics = source.metrics()
    return {
        "inventory_fingerprint": str(metrics["inventory_source_fingerprint"]),
        "repo_id": str(metrics["repo_id"]),
        "revision": str(metrics["revision"]),
    }


def _delta_probe_receipt(
    args: argparse.Namespace,
    recorder: DeltaNetProbeRecorder | None,
    *,
    checkpoint: Mapping[str, Any],
    context_mode: str,
    start_pos: int,
    end_pos: int,
    inputs: Mapping[str, Any],
    hidden_sha256: str,
) -> dict[str, Any] | None:
    path = getattr(args, "delta_probe", None)
    if path is None:
        if recorder is not None:  # pragma: no cover - internal contract.
            raise QwenDirectDecodeError("unused DeltaNet probe recorder")
        return None
    if recorder is None:  # pragma: no cover - internal contract.
        raise QwenDirectDecodeError("DeltaNet probe recorder is missing")
    document = build_probe_document(
        recorder,
        checkpoint=checkpoint,
        context_mode=context_mode,
        start_pos=start_pos,
        end_pos=end_pos,
        item_id=str(inputs["item_id"]),
        input_sha256=str(inputs["sha256"]),
        hidden_sha256=hidden_sha256,
    )
    output = _atomic_bytes(path, _canonical(document) + b"\n")
    return {
        "path": str(output),
        "records": len(document["body"]["records"]),
        "schema": DELTANET_PROBE_SCHEMA,
        "sha256": document["sha256"],
    }


def _cleanup(runtime: RuntimeSource, model: StreamedQwen38) -> None:
    active_error = sys.exc_info()[1]
    cleanup_error: Exception | None = None
    for action in (
        lambda: model.reset_state(release=True),
        model.pager.close,
        runtime.close,
    ):
        try:
            action()
        except Exception as exc:
            if cleanup_error is None:
                cleanup_error = exc
    if active_error is None and cleanup_error is not None:
        raise cleanup_error


def prepare_prefix(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _load_input(args.input)
    prefix = inputs["prefix_token_ids"]
    if len(prefix) >= args.max_seq_len:
        raise QwenDirectDecodeError("prefix leaves no room for decode")
    recorder = AccessTraceRecorder()
    delta_probe = DeltaNetProbeRecorder() if args.delta_probe is not None else None
    runtime, model = _runtime(args, recorder, delta_probe)
    try:
        if max((*prefix, int(inputs["decode_token_id"]))) >= model.config.vocab_size:
            raise QwenDirectDecodeError("input token exceeds checkpoint vocabulary")
        hidden, evidence = model.prefill(
            [prefix], tokenwise=False, reset=True, progress=_progress
        )
        snapshot = model.save_state(args.snapshot, transport_neutral=True)
        trace = _trace_receipt(recorder, args.access_trace)
        checkpoint = _checkpoint(model)
        hidden_sha256 = _hidden_sha256(hidden)
        probe_receipt = _delta_probe_receipt(
            args,
            delta_probe,
            checkpoint=checkpoint,
            context_mode="prefill",
            start_pos=0,
            end_pos=len(prefix),
            inputs=inputs,
            hidden_sha256=hidden_sha256,
        )
        identity = {
            "checkpoint": checkpoint,
            "evidence": [asdict(row) for row in evidence],
            "hidden_dtype": str(hidden.dtype).removeprefix("torch."),
            "hidden_shape": list(hidden.shape),
            "hidden_sha256": hidden_sha256,
            "input_sha256": inputs["sha256"],
            "pager": model.pager.metrics(),
            "schema": PREFIX_SCHEMA,
            "snapshot": snapshot,
            "source_verification": runtime.verification,
            "trace": trace,
        }
        if probe_receipt is not None:
            identity["delta_probe"] = probe_receipt
        return _result(identity)
    finally:
        _cleanup(runtime, model)


def decode_arm(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _load_input(args.input)
    prefix_result = _load_result(args.prefix_result, PREFIX_SCHEMA)
    if prefix_result["input_sha256"] != inputs["sha256"]:
        raise QwenDirectDecodeError("prefix result belongs to another input")
    snapshot_path = Path(args.snapshot).expanduser().resolve()
    if Path(prefix_result["snapshot"]["manifest"]).resolve() != snapshot_path:
        raise QwenDirectDecodeError("prefix result belongs to another snapshot")
    prefix = inputs["prefix_token_ids"]
    recorder = AccessTraceRecorder()
    delta_probe = DeltaNetProbeRecorder() if args.delta_probe is not None else None
    runtime, model = _runtime(args, recorder, delta_probe)
    try:
        restore_started = time.perf_counter()
        restored = model.load_state(args.snapshot, transport_neutral=True)
        if restored["payload_sha256"] != prefix_result["snapshot"]["payload_sha256"]:
            raise QwenDirectDecodeError("restored snapshot payload differs from prefix")
        restore_seconds = time.perf_counter() - restore_started
        if model.next_position != len(prefix):
            raise QwenDirectDecodeError("snapshot cursor differs from input prefix")
        checkpoint = _checkpoint(model)
        if checkpoint != prefix_result["checkpoint"]:
            raise QwenDirectDecodeError("prefix/decode checkpoint identity differs")
        hidden, evidence = model.decode(
            [[int(inputs["decode_token_id"])]], progress=_progress
        )
        trace = _trace_receipt(recorder, args.access_trace)
        hidden_sha256 = _hidden_sha256(hidden)
        probe_receipt = _delta_probe_receipt(
            args,
            delta_probe,
            checkpoint=checkpoint,
            context_mode="decode",
            start_pos=len(prefix),
            end_pos=len(prefix) + 1,
            inputs=inputs,
            hidden_sha256=hidden_sha256,
        )
        identity = {
            "checkpoint": checkpoint,
            "evidence": asdict(evidence),
            "hidden_dtype": str(hidden.dtype).removeprefix("torch."),
            "hidden_shape": list(hidden.shape),
            "hidden_sha256": hidden_sha256,
            "input_sha256": inputs["sha256"],
            "pager": model.pager.metrics(),
            "restore_seconds": restore_seconds,
            "schema": DECODE_SCHEMA,
            "snapshot": restored,
            "source_verification": runtime.verification,
            "trace": trace,
        }
        if probe_receipt is not None:
            identity["delta_probe"] = probe_receipt
        return _result(identity)
    finally:
        _cleanup(runtime, model)


def _load_result(path: str, schema: str) -> dict[str, Any]:
    document = _strict_json(path)
    if document.get("schema") != schema or not isinstance(document.get("sha256"), str):
        raise QwenDirectDecodeError("benchmark result schema is invalid")
    identity = {key: value for key, value in document.items() if key != "sha256"}
    if document["sha256"] != _sha256(identity):
        raise QwenDirectDecodeError("benchmark result SHA-256 mismatch")
    return document


def _verified_delta_probe(
    result: Mapping[str, Any],
) -> dict[str, Any] | None:
    receipt = result.get("delta_probe")
    if receipt is None:
        return None
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "path",
        "records",
        "schema",
        "sha256",
    }:
        raise QwenDirectDecodeError("DeltaNet probe receipt is invalid")
    if receipt.get("schema") != DELTANET_PROBE_SCHEMA:
        raise QwenDirectDecodeError("DeltaNet probe receipt schema is invalid")
    try:
        document = verify_probe_document(_strict_json(str(receipt["path"])))
    except Exception as exc:
        raise QwenDirectDecodeError("DeltaNet probe artifact is invalid") from exc
    body = document["body"]
    if (
        document["sha256"] != receipt.get("sha256")
        or len(body["records"]) != receipt.get("records")
        or body["checkpoint"] != result.get("checkpoint")
        or body["input_sha256"] != result.get("input_sha256")
        or body["hidden_sha256"] != result.get("hidden_sha256")
        or body["context_mode"] != "decode"
    ):
        raise QwenDirectDecodeError("DeltaNet probe/result identity differs")
    return document


def compare(args: argparse.Namespace) -> dict[str, Any]:
    remote = _load_result(args.remote, DECODE_SCHEMA)
    local = _load_result(args.local, DECODE_SCHEMA)
    for key in ("hidden_dtype", "hidden_shape", "hidden_sha256", "input_sha256"):
        if remote[key] != local[key]:
            raise QwenDirectDecodeError(f"remote/local {key} differs")
    for key in ("inventory_fingerprint", "repo_id", "revision"):
        if remote["checkpoint"][key] != local["checkpoint"][key]:
            raise QwenDirectDecodeError(f"remote/local checkpoint {key} differs")
    if remote["snapshot"]["payload_sha256"] != local["snapshot"]["payload_sha256"]:
        raise QwenDirectDecodeError("remote/local snapshot payload differs")
    remote_probe = _verified_delta_probe(remote)
    local_probe = _verified_delta_probe(local)
    if (remote_probe is None) != (local_probe is None):
        raise QwenDirectDecodeError("remote/local DeltaNet instrumentation differs")
    probe_sha256 = None
    if remote_probe is not None and local_probe is not None:
        if remote_probe["sha256"] != local_probe["sha256"]:
            raise QwenDirectDecodeError("remote/local DeltaNet probes differ")
        probe_sha256 = remote_probe["sha256"]
    remote_seconds = float(remote["evidence"]["seconds"])
    local_seconds = float(local["evidence"]["seconds"])
    if (
        not math.isfinite(remote_seconds)
        or not math.isfinite(local_seconds)
        or remote_seconds <= 0.0
        or local_seconds <= 0.0
    ):
        raise QwenDirectDecodeError("remote/local decode seconds are invalid")
    identity = {
        "hidden_sha256": remote["hidden_sha256"],
        "hidden_dtype": remote["hidden_dtype"],
        "hidden_shape": remote["hidden_shape"],
        "input_sha256": remote["input_sha256"],
        "delta_probe_sha256": probe_sha256,
        "local": {
            "result_sha256": local["sha256"],
            "seconds": local_seconds,
            "source_body_bytes": local["evidence"]["source_body_bytes"],
        },
        "remote": {
            "result_sha256": remote["sha256"],
            "seconds": remote_seconds,
            "source_body_bytes": remote["evidence"]["source_body_bytes"],
        },
        "schema": COMPARISON_SCHEMA,
        "seconds_ratio_local_over_remote": local_seconds / remote_seconds,
        "speedup_remote_over_local": remote_seconds / local_seconds,
    }
    return _result(identity)


def _build_graft(
    args: argparse.Namespace,
) -> tuple[Qwen38StableCrsaGraft | None, int | None, dict[str, Any]]:
    mode = str(args.mode)
    if mode == "off":
        return (
            None,
            None,
            {
                "alpha": 0.0,
                "arm_mode": "off",
                "attention_spec": None,
                "evidence_schema": None,
                "heads": None,
                "implementation": "none",
                "layer": None,
                "learned_parameters": 0,
                "max_history": None,
                "operator_mode": "off",
                "policy": "no-graft/v1",
                "rms_eps": None,
                "shuffle_seed": None,
                "stateful_history": False,
                "strict_causal": True,
                "uses_a1_ridge": False,
            },
        )
    if mode != "stable-crsa":
        raise QwenDirectDecodeError("branch mode must be off or stable-crsa")
    try:
        graft = Qwen38StableCrsaGraft(
            mode="crsa",
            alpha=args.graft_alpha,
            max_history=args.graft_max_history,
            rms_eps=args.graft_rms_eps,
        )
    except (TypeError, ValueError) as exc:
        raise QwenDirectDecodeError(
            "stable CRSA graft configuration is invalid"
        ) from exc
    layer = int(args.graft_layer)
    identity = {
        "alpha": graft.alpha,
        "arm_mode": "stable-crsa",
        "attention_spec": asdict(graft.spec),
        "evidence_schema": graft.evidence_schema,
        "heads": graft.heads,
        "implementation": ("immer.runtimes.qwen3_8.Qwen38StableCrsaGraft"),
        "layer": layer,
        "learned_parameters": sum(
            int(parameter.numel()) for parameter in graft.parameters()
        ),
        "max_history": graft.max_history,
        "operator_mode": graft.mode,
        "policy": graft.policy,
        "rms_eps": graft.rms_eps,
        "shuffle_seed": graft.shuffle_seed,
        "stateful_history": True,
        "strict_causal": True,
        "uses_a1_ridge": False,
    }
    return graft, layer, identity


def _native_intervention_identity(
    intervention: Qwen38NativeHeadCrsa,
) -> dict[str, Any]:
    return {
        "config": {
            "alpha": intervention.alpha,
            "balance_alpha": intervention.balance_alpha,
            "diagonal_debit": intervention.diagonal_debit,
            "head_indices": list(intervention.head_indices),
            "layer": intervention.layer,
        },
        "evidence": {
            "free_heads": list(NATIVE_HEAD_CRSA_FREE_HEADS),
            "row_sum_max_error_tolerance": NATIVE_HEAD_CRSA_ROW_SUM_TOLERANCE,
            "schema": intervention.evidence_schema,
            "selected_kv_heads": list(intervention.selected_kv_heads),
        },
        "implementation": "immer.runtimes.qwen3_8.Qwen38NativeHeadCrsa",
        "kind": "native-head-crsa",
        "learned_parameters": 0,
        "policy": "selected-head-probability-residual/prefix-log/v1",
        "stateful_history": True,
        "strict_causal": True,
        "usage_state": "per-selected-head-prefix-log/v1",
    }


def _build_native_intervention(
    args: argparse.Namespace,
) -> tuple[Qwen38NativeHeadCrsa, dict[str, Any]]:
    try:
        values = (
            float(getattr(args, "native_alpha", NATIVE_CRSA_ALPHA)),
            float(getattr(args, "native_balance_alpha", NATIVE_CRSA_BALANCE_ALPHA)),
            float(getattr(args, "native_diagonal_debit", NATIVE_CRSA_DIAGONAL_DEBIT)),
        )
    except (TypeError, ValueError) as exc:
        raise QwenDirectDecodeError(
            "native CRSA intervention configuration is invalid"
        ) from exc
    if values != (
        NATIVE_CRSA_ALPHA,
        NATIVE_CRSA_BALANCE_ALPHA,
        NATIVE_CRSA_DIAGONAL_DEBIT,
    ):
        raise QwenDirectDecodeError(
            "native CRSA arm requires alpha=0.01, balance_alpha=1, diagonal_debit=3"
        )
    try:
        intervention = Qwen38NativeHeadCrsa(
            alpha=values[0],
            balance_alpha=values[1],
            diagonal_debit=values[2],
        )
    except (TypeError, ValueError) as exc:  # pragma: no cover - constants above.
        raise QwenDirectDecodeError(
            "native CRSA intervention configuration is invalid"
        ) from exc
    if (
        intervention.layer != NATIVE_HEAD_CRSA_LAYER
        or intervention.head_indices != NATIVE_HEAD_CRSA_QUERY_HEADS
        or intervention.selected_kv_heads != NATIVE_HEAD_CRSA_KV_HEADS
    ):
        raise QwenDirectDecodeError("native CRSA fixed head mapping is invalid")
    return intervention, _native_intervention_identity(intervention)


def _source_metric(source: object, name: str) -> int:
    metrics_method = getattr(source, "metrics", None)
    metrics = dict(metrics_method()) if callable(metrics_method) else {}
    value = metrics.get(name, 0)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _branch_input_identity(inputs: Mapping[str, Any]) -> dict[str, Any]:
    # The complete projection is small and contains no labels or draft tokens.
    # Keeping it here lets later stages recompute its seal and bind every prompt
    # rather than trusting an opaque input digest supplied by the arm producer.
    return dict(inputs)


def _branch_source_budget_preflight(
    args: argparse.Namespace,
    verification: Mapping[str, Any] | None,
    *,
    items: int,
) -> None:
    """Reject a cumulative source budget that cannot cover the sealed bound."""

    if verification is None or "checkpoint_bytes" not in verification:
        return
    checkpoint_bytes = verification["checkpoint_bytes"]
    if (
        isinstance(checkpoint_bytes, bool)
        or not isinstance(checkpoint_bytes, int)
        or checkpoint_bytes <= 0
    ):
        raise QwenDirectDecodeError("verified checkpoint byte count is invalid")
    passes = items * (1 + int(args.max_new_tokens))
    required = checkpoint_bytes * passes
    available = int(args.source_budget_mb) * 1024**2
    if available < required:
        raise QwenDirectDecodeError(
            "branch source budget is below the sealed worst-case bound: "
            f"{available}/{required} bytes for {passes} full passes"
        )


def generate_arm(
    args: argparse.Namespace,
    *,
    runtime_factory: Callable[..., tuple[RuntimeSource, StreamedQwen38]] | None = None,
    tokenizer_factory: Callable[..., Qwen38Tokenizer] = Qwen38Tokenizer,
) -> dict[str, Any]:
    """Generate every selected item exactly once with one shared weight pager."""

    inputs = _load_branch_input(args.input)
    if (
        args.logical_repo_id != inputs["source"]["checkpoint"]
        or args.revision != inputs["source"]["revision"]
    ):
        raise QwenDirectDecodeError(
            "requested checkpoint/revision differs from branch input"
        )
    _tokenizer, tokenizer_identity = _tokenizer_record(
        args.tokenizer_json,
        require_official=getattr(args, "_require_official", True),
        tokenizer_factory=tokenizer_factory,
    )
    if tokenizer_identity != inputs["tokenizer"]:
        raise QwenDirectDecodeError("generation tokenizer differs from branch input")
    tokenizer = _tokenizer
    answer_input = inputs["schema"] == BRANCH_INPUT_SCHEMA_V3
    if answer_input:
        encoded_prefix = _answer_generation_prefix(tokenizer, tokenizer_identity)
        sealed_prefix = tuple(inputs["protocol"]["generation_prefix"]["token_ids"])
        if encoded_prefix != sealed_prefix:
            raise QwenDirectDecodeError(
                "answer generation prefix differs from pinned tokenizer"
            )
    prompts = [
        _token_rows(
            (
                row["effective_prompt_token_ids"]
                if answer_input
                else row["prompt_token_ids"]
            ),
            f"prompt {row['item_id']}",
        )
        for row in inputs["items"]
    ]
    max_prompt = max(len(prompt) for prompt in prompts)
    if max_prompt + args.max_new_tokens > args.max_seq_len:
        raise QwenDirectDecodeError("generation bound exceeds max_seq_len")
    native_mode = str(args.mode) == "native-crsa"
    native_evidence: list[NativeHeadCrsaEvidence] = []
    native_observer = native_evidence.append if native_mode else None
    intervention: Qwen38NativeHeadCrsa | None = None
    intervention_identity: dict[str, Any] | None = None
    if native_mode:
        intervention, intervention_identity = _build_native_intervention(args)
        graft = None
        graft_layer = None
        graft_identity = None
    else:
        graft, graft_layer, graft_identity = _build_graft(args)
    if graft is not None and args.graft_max_history < max_prompt + args.max_new_tokens:
        raise QwenDirectDecodeError("graft max_history is below the generation bound")
    recorder = AccessTraceRecorder()
    factory = _runtime if runtime_factory is None else runtime_factory
    if native_mode:
        runtime, model = factory(
            args,
            recorder,
            graft=None,
            graft_layer=None,
            native_head_crsa=intervention,
            native_head_crsa_observer=native_observer,
            max_batch_size=1,
        )
    else:
        runtime, model = factory(
            args,
            recorder,
            graft=graft,
            graft_layer=graft_layer,
            max_batch_size=1,
        )
    attachment_valid = (
        model.graft is graft
        and model.graft_layer == graft_layer
        and model.max_batch_size == 1
        and model.max_seq_len == args.max_seq_len
    )
    if native_mode:
        attachment_valid = (
            attachment_valid
            and getattr(model, "native_head_crsa", None) is intervention
            and getattr(model, "native_head_crsa_observer", None) is native_observer
        )
    else:
        attachment_valid = (
            attachment_valid
            and getattr(model, "native_head_crsa", None) is None
            and getattr(model, "native_head_crsa_observer", None) is None
        )
    if not attachment_valid:
        _cleanup(runtime, model)
        raise QwenDirectDecodeError("branch runtime attachment differs from contract")
    try:
        _branch_source_budget_preflight(
            args,
            runtime.verification,
            items=len(prompts),
        )
    except Exception:
        _cleanup(runtime, model)
        raise
    started = time.perf_counter()
    source_start = _source_metric(model.pager.source, "network_or_source_body_bytes")
    items: list[dict[str, Any]] = []
    try:
        eos = tuple(
            int(value) for value in inputs["protocol"]["accepted_eos_token_ids"]
        )
        if (
            max((*eos, *(token for prompt in prompts for token in prompt)))
            >= model.config.vocab_size
        ):
            raise QwenDirectDecodeError("branch token exceeds checkpoint vocabulary")
        for source_row, prompt in zip(inputs["items"], prompts, strict=True):
            native_evidence_start = len(native_evidence)
            head_blocks = 0
            completed_scans = 0

            def head_progress(event: Mapping[str, int]) -> None:
                nonlocal completed_scans, head_blocks
                head_blocks += 1
                if event.get("rows_done") == event.get("vocab_rows"):
                    completed_scans += 1

            try:
                generated, evidence = model.generate_greedy(
                    [prompt],
                    max_new_tokens=args.max_new_tokens,
                    prefill_tokenwise=False,
                    eos_token_ids=eos,
                    head_block_rows=args.head_block_rows,
                    progress=_progress,
                    head_progress=head_progress,
                )
            except Exception as exc:
                raise QwenDirectDecodeError(
                    f"generation failed for {source_row['item_id']}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            generated_ids = tuple(int(token) for token in generated)
            if not generated_ids:
                raise QwenDirectDecodeError("greedy generation emitted no token")
            if tuple(evidence.prompt_token_ids) != prompt:
                raise QwenDirectDecodeError("runtime changed the selected prompt")
            if tuple(evidence.generated_token_ids) != generated_ids:
                raise QwenDirectDecodeError("runtime generation evidence differs")
            if (
                evidence.context_mode != "stateful_autoregressive"
                or evidence.stateful_cache is not True
                or evidence.general_generation is not True
            ):
                raise QwenDirectDecodeError(
                    "runtime did not perform general generation"
                )
            if completed_scans != len(generated_ids):
                raise QwenDirectDecodeError("LM-head scan accounting is incomplete")
            text = tokenizer.decode(generated_ids)
            stopped = bool(evidence.stopped_on_eos)
            finish_reason = "stop" if stopped else "length"
            eos_token_id = generated_ids[-1] if stopped else None
            if stopped != (generated_ids[-1] in eos):
                raise QwenDirectDecodeError("runtime EOS evidence is inconsistent")
            if answer_input and "####" in text:
                raise QwenDirectDecodeError(
                    "answer branch generated text re-emits the sealed #### prefix"
                )
            parsed_text = (
                f"{ANSWER_GENERATION_PREFIX_LITERAL}{text}" if answer_input else text
            )
            item = {
                "context_mode": evidence.context_mode,
                "eos_token_id": eos_token_id,
                "finish_reason": finish_reason,
                "forward_passes": int(evidence.forward_passes),
                "general_generation": bool(evidence.general_generation),
                "generated_text": text,
                "generated_token_ids": list(generated_ids),
                "head_scan_blocks": head_blocks,
                "head_scans": completed_scans,
                "item_id": str(source_row["item_id"]),
                "linear_calls": int(evidence.linear_calls),
                "parsed_numeric_answer": extract_gsm8k_answer(parsed_text),
                "prefill_mode": evidence.prefill_mode,
                "prompt_token_ids": list(prompt),
                "seconds": float(evidence.seconds),
                "source_body_bytes": int(evidence.source_body_bytes),
                "state_bytes": int(evidence.state_bytes),
                "stateful_autoregressive": bool(evidence.stateful_cache),
                "stopped_on_eos": stopped,
                "token_chain_sha256": _token_chain_sha256(prompt, generated_ids),
            }
            if native_mode:
                committed = native_evidence[native_evidence_start:]
                if len(committed) != int(evidence.forward_passes):
                    raise QwenDirectDecodeError(
                        "native CRSA evidence count differs from committed forwards"
                    )
                evidence_rows = [row.to_dict() for row in committed]
                item["intervention_evidence"] = evidence_rows
                item["intervention_evidence_sha256"] = _sha256(evidence_rows)
            items.append(item)
        checkpoint = _checkpoint(model)
        if (
            checkpoint["repo_id"] != inputs["source"]["checkpoint"]
            or checkpoint["revision"] != inputs["source"]["revision"]
        ):
            raise QwenDirectDecodeError("runtime checkpoint differs from branch input")
        trace = _trace_receipt(recorder, args.access_trace)
        wall_seconds = time.perf_counter() - started
        generation = {
            "algorithm": "exact-greedy/v1",
            "eos_token_ids": list(eos),
            "general_generation": True,
            "head_block_rows": args.head_block_rows,
            "max_new_tokens": args.max_new_tokens,
            "max_seq_len": args.max_seq_len,
            "prefill_mode": "batched-per-item",
            "stateful_autoregressive": True,
            "teacher_forced_tokens_after_prompt": 0,
            "temperature": 0.0,
        }
        execution = {
            "checkpoint_layers": int(model.config.n_layers),
            "cohort_size": len(items),
            "device": str(model.pager.device),
            "dtype": str(model.pager.compute_dtype).removeprefix("torch."),
            "execution_mode": "serial_items/shared_weight_pager",
            "independent_prefills": len(items),
            "max_batch_size": 1,
            "runtime_instances": 1,
            "shared_weight_pager": True,
        }
        traffic = {
            "access_trace": trace,
            "access_trace_sha256": trace["sha256"],
            "forward_passes": sum(row["forward_passes"] for row in items),
            "generation_seconds": sum(row["seconds"] for row in items),
            "head_scan_blocks": sum(row["head_scan_blocks"] for row in items),
            "head_scans": sum(row["head_scans"] for row in items),
            "linear_calls": sum(row["linear_calls"] for row in items),
            "runtime_source_body_bytes": (
                _source_metric(model.pager.source, "network_or_source_body_bytes")
                - source_start
            ),
            "source_body_bytes": sum(row["source_body_bytes"] for row in items),
            "wall_seconds": wall_seconds,
        }
        if native_mode:
            identity = {
                "arm": "native-crsa",
                "bundle": dict(runtime.verification or {}),
                "checkpoint": checkpoint,
                "execution": execution,
                "generation": generation,
                "input": _branch_input_identity(inputs),
                "intervention": intervention_identity,
                "items": items,
                "schema": NATIVE_BRANCH_RESULT_SCHEMA,
                "status": "sealed",
                "tokenizer": tokenizer_identity,
                "traffic": traffic,
            }
            result = _result(identity)
            _validate_native_branch_result_document(result)
            return result
        identity = {
            "arm": str(args.mode),
            "bundle": dict(runtime.verification or {}),
            "checkpoint": checkpoint,
            "execution": execution,
            "generation": generation,
            "graft": graft_identity,
            "input": _branch_input_identity(inputs),
            "items": items,
            "schema": BRANCH_RESULT_SCHEMA,
            "status": "sealed",
            "tokenizer": tokenizer_identity,
            "traffic": traffic,
        }
        return _result(identity)
    finally:
        _cleanup(runtime, model)


def _verify_access_trace_receipt(
    receipt: object,
    *,
    artifact_path: str | os.PathLike[str] | None = None,
) -> None:
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "inventory_fingerprint",
        "leaves",
        "operations",
        "path",
        "sha256",
    }:
        raise QwenDirectDecodeError("branch access-trace receipt is invalid")
    _digest_string(receipt.get("sha256"), "access trace")
    if not isinstance(receipt.get("inventory_fingerprint"), str):
        raise QwenDirectDecodeError("branch access-trace inventory is invalid")
    leaves = _nonnegative_count(receipt.get("leaves"), "access-trace leaves")
    operations = _nonnegative_count(
        receipt.get("operations"), "access-trace operations"
    )
    trace_path = (
        Path(str(receipt["path"])) if artifact_path is None else Path(artifact_path)
    )
    try:
        trace = AccessTrace.from_bytes(trace_path.read_bytes())
        trace.verify()
    except Exception as exc:
        raise QwenDirectDecodeError("branch access-trace artifact is invalid") from exc
    if (
        trace.sha256 != receipt["sha256"]
        or trace.inventory_fingerprint != receipt["inventory_fingerprint"]
        or len(trace.operations) != operations
        or sum(len(operation.leaves) for operation in trace.operations) != leaves
    ):
        raise QwenDirectDecodeError("branch access-trace receipt differs from artifact")


def _validate_graft_identity(graft: object, arm: str) -> Mapping[str, Any]:
    required = {
        "alpha",
        "arm_mode",
        "attention_spec",
        "evidence_schema",
        "heads",
        "implementation",
        "layer",
        "learned_parameters",
        "max_history",
        "operator_mode",
        "policy",
        "rms_eps",
        "shuffle_seed",
        "stateful_history",
        "strict_causal",
        "uses_a1_ridge",
    }
    if not isinstance(graft, Mapping) or set(graft) != required:
        raise QwenDirectDecodeError("branch graft identity is invalid")
    if graft.get("arm_mode") != arm:
        raise QwenDirectDecodeError("branch arm/graft mode differs")
    if arm == "off":
        if graft != _build_graft(argparse.Namespace(mode="off"))[2]:
            raise QwenDirectDecodeError("off arm graft identity is not inert")
        return graft
    if arm != "stable-crsa":
        raise QwenDirectDecodeError("branch arm mode is invalid")
    if (
        graft.get("implementation") != "immer.runtimes.qwen3_8.Qwen38StableCrsaGraft"
        or graft.get("operator_mode") != "crsa"
        or graft.get("stateful_history") is not True
        or graft.get("strict_causal") is not True
        or graft.get("uses_a1_ridge") is not False
        or graft.get("learned_parameters") != 0
        or not isinstance(graft.get("attention_spec"), Mapping)
        or graft["attention_spec"].get("kind") != "role_complete"
        or graft.get("heads") != 4
    ):
        raise QwenDirectDecodeError("stable CRSA graft identity is incomplete")
    for name in ("layer", "max_history"):
        if isinstance(graft.get(name), bool) or not isinstance(graft.get(name), int):
            raise QwenDirectDecodeError(f"stable CRSA {name} is invalid")
    if graft["layer"] < 0 or graft["max_history"] < 1:
        raise QwenDirectDecodeError("stable CRSA layer/history bound is invalid")
    alpha = _finite_nonnegative(graft.get("alpha"), "stable CRSA alpha")
    if alpha > 1.0:
        raise QwenDirectDecodeError("stable CRSA alpha must be in [0, 1]")
    rms_eps = _finite_nonnegative(graft.get("rms_eps"), "stable CRSA RMS epsilon")
    if rms_eps <= 0.0:
        raise QwenDirectDecodeError("stable CRSA RMS epsilon must be positive")
    expected = _build_graft(
        argparse.Namespace(
            graft_alpha=alpha,
            graft_layer=graft["layer"],
            graft_max_history=graft["max_history"],
            graft_rms_eps=rms_eps,
            mode="stable-crsa",
        )
    )[2]
    if graft != expected:
        raise QwenDirectDecodeError("stable CRSA graft identity is not canonical")
    return graft


def _relocated_access_trace_path(
    result_path: str | os.PathLike[str], document: Mapping[str, Any]
) -> Path:
    """Resolve one transported result's adjacent trace without changing its seal."""

    receipt = document.get("traffic")
    if not isinstance(receipt, Mapping):
        return Path("")
    access_trace = receipt.get("access_trace")
    if not isinstance(access_trace, Mapping):
        return Path("")
    recorded = Path(str(access_trace.get("path", "")))
    if recorded.is_file():
        return recorded
    source = Path(result_path)
    if "result-" not in source.name:
        return recorded
    adjacent = source.with_name(source.name.replace("result-", "access-", 1))
    return adjacent if adjacent.is_file() else recorded


def _load_branch_result(path: str | os.PathLike[str]) -> dict[str, Any]:
    document = _strict_json(path)
    return _validate_branch_result_document(
        document,
        access_trace_path=_relocated_access_trace_path(path, document),
    )


def _validate_branch_result_document(
    document: dict[str, Any],
    *,
    access_trace_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    required = {
        "arm",
        "bundle",
        "checkpoint",
        "execution",
        "generation",
        "graft",
        "input",
        "items",
        "schema",
        "sha256",
        "status",
        "tokenizer",
        "traffic",
    }
    if set(document) != required or document.get("schema") != BRANCH_RESULT_SCHEMA:
        raise QwenDirectDecodeError("branch result schema is invalid")
    _verify_document_seal(document, "branch result")
    if document.get("status") != "sealed":
        raise QwenDirectDecodeError("branch result is not sealed")
    arm = document.get("arm")
    if not isinstance(arm, str):
        raise QwenDirectDecodeError("branch result arm is invalid")
    _validate_graft_identity(document.get("graft"), arm)
    checkpoint = document.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "inventory_fingerprint",
        "repo_id",
        "revision",
    }:
        raise QwenDirectDecodeError("branch checkpoint identity is invalid")
    for name in checkpoint:
        if not isinstance(checkpoint[name], str) or not checkpoint[name]:
            raise QwenDirectDecodeError("branch checkpoint identity is invalid")
    if not isinstance(document.get("bundle"), Mapping):
        raise QwenDirectDecodeError("branch bundle identity is invalid")
    if not isinstance(document["bundle"].get("kind"), str):
        raise QwenDirectDecodeError("branch bundle kind is invalid")
    input_identity = document.get("input")
    if not isinstance(input_identity, dict):
        raise QwenDirectDecodeError("branch input identity is invalid")
    _validate_branch_input(input_identity)
    answer_input = input_identity["schema"] == BRANCH_INPUT_SCHEMA_V3
    if (
        checkpoint["repo_id"] != input_identity["source"]["checkpoint"]
        or checkpoint["revision"] != input_identity["source"]["revision"]
    ):
        raise QwenDirectDecodeError("branch checkpoint differs from sealed input")
    generation = document.get("generation")
    generation_required = {
        "algorithm",
        "eos_token_ids",
        "general_generation",
        "head_block_rows",
        "max_new_tokens",
        "max_seq_len",
        "prefill_mode",
        "stateful_autoregressive",
        "teacher_forced_tokens_after_prompt",
        "temperature",
    }
    if not isinstance(generation, Mapping) or set(generation) != generation_required:
        raise QwenDirectDecodeError("branch generation identity is invalid")
    eos = _token_rows(generation.get("eos_token_ids"), "generation EOS IDs")
    max_new_tokens = _nonnegative_count(
        generation.get("max_new_tokens"), "max_new_tokens"
    )
    if (
        max_new_tokens < 1
        or generation.get("algorithm") != "exact-greedy/v1"
        or generation.get("general_generation") is not True
        or generation.get("stateful_autoregressive") is not True
        or generation.get("teacher_forced_tokens_after_prompt") != 0
        or generation.get("temperature") != 0.0
        or generation.get("prefill_mode") != "batched-per-item"
    ):
        raise QwenDirectDecodeError("branch generation contract is inconsistent")
    if list(eos) != input_identity["protocol"]["accepted_eos_token_ids"]:
        raise QwenDirectDecodeError("branch generation EOS set differs from input")
    if (
        _nonnegative_count(generation.get("head_block_rows"), "head block rows") < 1
        or _nonnegative_count(generation.get("max_seq_len"), "max sequence length") < 1
    ):
        raise QwenDirectDecodeError("branch generation bounds must be positive")
    execution = document.get("execution")
    if not isinstance(execution, Mapping) or set(execution) != {
        "checkpoint_layers",
        "cohort_size",
        "device",
        "dtype",
        "execution_mode",
        "independent_prefills",
        "max_batch_size",
        "runtime_instances",
        "shared_weight_pager",
    }:
        raise QwenDirectDecodeError("branch execution identity is invalid")
    rows = document.get("items")
    if not isinstance(rows, list) or not rows:
        raise QwenDirectDecodeError("branch result contains no items")
    if (
        isinstance(execution.get("checkpoint_layers"), bool)
        or not isinstance(execution.get("checkpoint_layers"), int)
        or execution["checkpoint_layers"] < 1
        or execution.get("cohort_size") != len(rows)
        or execution.get("independent_prefills") != len(rows)
        or execution.get("max_batch_size") != 1
        or execution.get("runtime_instances") != 1
        or execution.get("shared_weight_pager") is not True
        or execution.get("execution_mode") != "serial_items/shared_weight_pager"
        or not isinstance(execution.get("device"), str)
        or not execution["device"]
        or not isinstance(execution.get("dtype"), str)
        or not execution["dtype"]
    ):
        raise QwenDirectDecodeError("branch execution accounting is inconsistent")
    if (
        arm == "stable-crsa"
        and document["graft"]["layer"] >= execution["checkpoint_layers"]
    ):
        raise QwenDirectDecodeError("stable CRSA layer exceeds checkpoint depth")
    item_ids = input_identity["selection"]["item_ids"]
    if (
        not isinstance(item_ids, list)
        or len(item_ids) != len(rows)
        or any(not isinstance(item_id, str) or not item_id for item_id in item_ids)
        or len(set(item_ids)) != len(item_ids)
    ):
        raise QwenDirectDecodeError("branch result item identity is invalid")
    tokenizer = document.get("tokenizer")
    if not isinstance(tokenizer, Mapping) or set(tokenizer) != {
        "control_token_ids",
        "kind",
        "require_official",
        "sha256",
        "size_bytes",
        "vocab_size",
    }:
        raise QwenDirectDecodeError("branch result tokenizer identity is invalid")
    _digest_string(tokenizer.get("sha256"), "branch result tokenizer")
    tokenizer_vocab = _nonnegative_count(
        tokenizer.get("vocab_size"), "branch result tokenizer vocabulary"
    )
    _nonnegative_count(tokenizer.get("size_bytes"), "branch result tokenizer size")
    controls = tokenizer.get("control_token_ids")
    if (
        tokenizer.get("kind") != "tokenizers-json/v1"
        or not isinstance(tokenizer.get("require_official"), bool)
        or tokenizer_vocab < 1
        or not isinstance(controls, Mapping)
        or set(controls) != {"end_of_text", "im_end"}
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < tokenizer_vocab
            for value in controls.values()
        )
        or set(eos) != set(controls.values())
    ):
        raise QwenDirectDecodeError("branch result tokenizer contract is invalid")
    if tokenizer != input_identity["tokenizer"]:
        raise QwenDirectDecodeError("branch result tokenizer differs from sealed input")
    row_required = {
        "context_mode",
        "eos_token_id",
        "finish_reason",
        "forward_passes",
        "general_generation",
        "generated_text",
        "generated_token_ids",
        "head_scan_blocks",
        "head_scans",
        "item_id",
        "linear_calls",
        "parsed_numeric_answer",
        "prefill_mode",
        "prompt_token_ids",
        "seconds",
        "source_body_bytes",
        "state_bytes",
        "stateful_autoregressive",
        "stopped_on_eos",
        "token_chain_sha256",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != row_required:
            raise QwenDirectDecodeError("branch generated item schema is invalid")
        if row.get("item_id") != item_ids[index]:
            raise QwenDirectDecodeError("branch generated item order differs")
        prompt = _token_rows(row.get("prompt_token_ids"), "generated prompt")
        sealed_item = input_identity["items"][index]
        sealed_prompt = (
            sealed_item["effective_prompt_token_ids"]
            if answer_input
            else sealed_item["prompt_token_ids"]
        )
        if row["item_id"] != sealed_item["item_id"] or list(prompt) != sealed_prompt:
            raise QwenDirectDecodeError(
                "branch result prompt differs from sealed input"
            )
        generated = _token_rows(row.get("generated_token_ids"), "generated tokens")
        if (
            len(generated) > max_new_tokens
            or max((*prompt, *generated)) >= tokenizer_vocab
            or len(prompt) + len(generated) > generation["max_seq_len"]
        ):
            raise QwenDirectDecodeError("branch generated token bound was exceeded")
        if row.get("token_chain_sha256") != _token_chain_sha256(prompt, generated):
            raise QwenDirectDecodeError("branch token-chain digest mismatch")
        text = row.get("generated_text")
        if answer_input and isinstance(text, str) and "####" in text:
            raise QwenDirectDecodeError(
                "answer branch generated text re-emits the sealed #### prefix"
            )
        parsed_text = (
            f"{ANSWER_GENERATION_PREFIX_LITERAL}{text}"
            if answer_input and isinstance(text, str)
            else text
        )
        if not isinstance(text, str) or row.get(
            "parsed_numeric_answer"
        ) != extract_gsm8k_answer(parsed_text):
            raise QwenDirectDecodeError("branch parsed numeric answer is inconsistent")
        stopped = row.get("stopped_on_eos")
        if not isinstance(stopped, bool):
            raise QwenDirectDecodeError("branch EOS state is invalid")
        if stopped:
            if (
                row.get("finish_reason") != "stop"
                or row.get("eos_token_id") != generated[-1]
                or generated[-1] not in eos
            ):
                raise QwenDirectDecodeError("branch EOS finish evidence is invalid")
        elif (
            row.get("finish_reason") != "length"
            or row.get("eos_token_id") is not None
            or len(generated) != max_new_tokens
        ):
            raise QwenDirectDecodeError("branch length finish evidence is invalid")
        if (
            row.get("context_mode") != "stateful_autoregressive"
            or row.get("general_generation") is not True
            or row.get("stateful_autoregressive") is not True
            or row.get("prefill_mode") != "batched"
            or row.get("forward_passes") != len(generated) + 1
            or row.get("head_scans") != len(generated)
        ):
            raise QwenDirectDecodeError("branch autoregressive evidence is invalid")
        for name in (
            "forward_passes",
            "head_scan_blocks",
            "head_scans",
            "linear_calls",
            "source_body_bytes",
            "state_bytes",
        ):
            _nonnegative_count(row.get(name), f"branch item {name}")
        _finite_nonnegative(row.get("seconds"), "branch item seconds")
    if arm == "stable-crsa" and document["graft"]["max_history"] < max(
        len(row["prompt_token_ids"]) + max_new_tokens for row in rows
    ):
        raise QwenDirectDecodeError("stable CRSA history bound is too small")
    traffic = document.get("traffic")
    traffic_required = {
        "access_trace",
        "access_trace_sha256",
        "forward_passes",
        "generation_seconds",
        "head_scan_blocks",
        "head_scans",
        "linear_calls",
        "runtime_source_body_bytes",
        "source_body_bytes",
        "wall_seconds",
    }
    if not isinstance(traffic, Mapping) or set(traffic) != traffic_required:
        raise QwenDirectDecodeError("branch traffic evidence is invalid")
    _verify_access_trace_receipt(
        traffic.get("access_trace"), artifact_path=access_trace_path
    )
    if traffic.get("access_trace_sha256") != traffic["access_trace"]["sha256"]:
        raise QwenDirectDecodeError("branch access-trace digest is inconsistent")
    sums = {
        "forward_passes": sum(row["forward_passes"] for row in rows),
        "generation_seconds": sum(row["seconds"] for row in rows),
        "head_scan_blocks": sum(row["head_scan_blocks"] for row in rows),
        "head_scans": sum(row["head_scans"] for row in rows),
        "linear_calls": sum(row["linear_calls"] for row in rows),
        "source_body_bytes": sum(row["source_body_bytes"] for row in rows),
    }
    for name, expected in sums.items():
        if traffic.get(name) != expected:
            raise QwenDirectDecodeError(f"branch traffic field {name} is inconsistent")
    _nonnegative_count(traffic.get("runtime_source_body_bytes"), "runtime source bytes")
    _finite_nonnegative(traffic.get("wall_seconds"), "branch wall seconds")
    if traffic["wall_seconds"] < traffic["generation_seconds"]:
        raise QwenDirectDecodeError("branch wall timing is below model timing")
    return document


def _validate_native_intervention_identity(
    value: object,
) -> Mapping[str, Any]:
    expected = _build_native_intervention(
        argparse.Namespace(
            native_alpha=NATIVE_CRSA_ALPHA,
            native_balance_alpha=NATIVE_CRSA_BALANCE_ALPHA,
            native_diagonal_debit=NATIVE_CRSA_DIAGONAL_DEBIT,
        )
    )[1]
    if not isinstance(value, Mapping) or _canonical(dict(value)) != _canonical(
        expected
    ):
        raise QwenDirectDecodeError("native CRSA intervention identity is invalid")
    return value


def _native_evidence_from_mapping(
    value: object,
    *,
    item_id: str,
) -> NativeHeadCrsaEvidence:
    required = {
        "alpha_per_head",
        "argmax_changed_queries_per_head",
        "base_softmax_head_rows_skipped",
        "base_softmax_probability_elements_skipped",
        "batch_size",
        "execution_mode",
        "free_head_max_abs_error",
        "free_heads",
        "future_weight_max_abs",
        "history_length_after",
        "history_length_before",
        "identity",
        "key_length",
        "layer",
        "mean_l1_probability_delta_per_head",
        "query_length",
        "query_start",
        "row_sum_max_error",
        "schema",
        "selected_kv_heads",
        "selected_query_heads",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise QwenDirectDecodeError(
            f"native CRSA evidence schema is invalid for {item_id}"
        )
    payload = dict(value)
    for name in (
        "alpha_per_head",
        "argmax_changed_queries_per_head",
        "free_heads",
        "mean_l1_probability_delta_per_head",
        "selected_kv_heads",
        "selected_query_heads",
    ):
        raw = payload[name]
        if not isinstance(raw, list):
            raise QwenDirectDecodeError(
                f"native CRSA evidence tuple field {name} is invalid for {item_id}"
            )
        payload[name] = tuple(raw)
    try:
        evidence = NativeHeadCrsaEvidence(**payload)
    except (TypeError, ValueError) as exc:
        raise QwenDirectDecodeError(
            f"native CRSA evidence is invalid for {item_id}: {exc}"
        ) from exc
    if _canonical(evidence.to_dict()) != _canonical(dict(value)):
        raise QwenDirectDecodeError(
            f"native CRSA evidence is not canonical for {item_id}"
        )
    return evidence


def _validate_native_evidence_chain(row: Mapping[str, Any]) -> None:
    item_id = str(row["item_id"])
    raw_rows = row.get("intervention_evidence")
    evidence_sha256 = _digest_string(
        row.get("intervention_evidence_sha256"),
        f"native CRSA evidence for {item_id}",
    )
    if not isinstance(raw_rows, list) or len(raw_rows) != row["forward_passes"]:
        raise QwenDirectDecodeError(
            f"native CRSA evidence count differs from forward passes for {item_id}"
        )
    prompt_length = len(row["prompt_token_ids"])
    for index, raw in enumerate(raw_rows):
        evidence = _native_evidence_from_mapping(raw, item_id=item_id)
        query_start = 0 if index == 0 else prompt_length + index - 1
        query_length = prompt_length if index == 0 else 1
        key_length = query_start + query_length
        if (
            evidence.schema != NATIVE_HEAD_CRSA_EVIDENCE_SCHEMA
            or evidence.layer != NATIVE_HEAD_CRSA_LAYER
            or evidence.selected_query_heads != NATIVE_HEAD_CRSA_QUERY_HEADS
            or evidence.selected_kv_heads != NATIVE_HEAD_CRSA_KV_HEADS
            or evidence.free_heads != NATIVE_HEAD_CRSA_FREE_HEADS
            or evidence.alpha_per_head
            != (NATIVE_CRSA_ALPHA,) * len(NATIVE_HEAD_CRSA_QUERY_HEADS)
            or evidence.free_head_max_abs_error != 0.0
            or evidence.future_weight_max_abs != 0.0
            or evidence.row_sum_max_error > NATIVE_HEAD_CRSA_ROW_SUM_TOLERANCE
            or evidence.identity is not False
        ):
            raise QwenDirectDecodeError(
                f"native CRSA evidence identity differs for {item_id}"
            )
        if (
            evidence.query_start != query_start
            or evidence.query_length != query_length
            or evidence.key_length != key_length
            or evidence.history_length_before != query_start
            or evidence.history_length_after != key_length
        ):
            raise QwenDirectDecodeError(
                f"native CRSA evidence history chain is invalid for {item_id}"
            )
        if any(
            changed > query_length
            for changed in evidence.argmax_changed_queries_per_head
        ):
            raise QwenDirectDecodeError(
                f"native CRSA argmax evidence exceeds query count for {item_id}"
            )
    if evidence_sha256 != _sha256(raw_rows):
        raise QwenDirectDecodeError(
            f"native CRSA evidence digest mismatch for {item_id}"
        )


def _native_v2_common_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    projected = {key: value for key, value in document.items() if key != "sha256"}
    projected.pop("intervention", None)
    projected["arm"] = "off"
    projected["graft"] = _build_graft(argparse.Namespace(mode="off"))[2]
    projected["items"] = [
        {
            key: value
            for key, value in row.items()
            if key not in {"intervention_evidence", "intervention_evidence_sha256"}
        }
        for row in document["items"]
    ]
    projected["schema"] = BRANCH_RESULT_SCHEMA
    return _result(projected)


def _validate_native_branch_result_document(
    document: dict[str, Any],
    *,
    access_trace_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    required = {
        "arm",
        "bundle",
        "checkpoint",
        "execution",
        "generation",
        "input",
        "intervention",
        "items",
        "schema",
        "sha256",
        "status",
        "tokenizer",
        "traffic",
    }
    if (
        set(document) != required
        or document.get("schema") != NATIVE_BRANCH_RESULT_SCHEMA
        or document.get("arm") != "native-crsa"
        or document.get("status") != "sealed"
        or not isinstance(document.get("items"), list)
        or any(not isinstance(row, Mapping) for row in document.get("items", ()))
    ):
        raise QwenDirectDecodeError("native branch result schema is invalid")
    _verify_document_seal(document, "native branch result")
    _validate_native_intervention_identity(document.get("intervention"))
    _validate_branch_result_document(
        _native_v2_common_projection(document),
        access_trace_path=access_trace_path,
    )
    layers = document["execution"]["checkpoint_layers"]
    if layers <= NATIVE_HEAD_CRSA_LAYER:
        raise QwenDirectDecodeError("native CRSA layer exceeds checkpoint depth")
    for row in document["items"]:
        _validate_native_evidence_chain(row)
    return document


def _load_native_branch_result(path: str | os.PathLike[str]) -> dict[str, Any]:
    document = _strict_json(path)
    return _validate_native_branch_result_document(
        document,
        access_trace_path=_relocated_access_trace_path(path, document),
    )


def _native_fork_source_budget_preflight(
    args: argparse.Namespace,
    verification: Mapping[str, Any] | None,
    *,
    items: int,
) -> None:
    """Reserve the independent two-arm upper bound for one forked run."""

    _branch_source_budget_preflight(args, verification, items=2 * items)


def _native_fork_runtime(
    args: argparse.Namespace,
    recorder: AccessTraceRecorder,
    *,
    native_head_crsa: Qwen38NativeHeadCrsa,
) -> tuple[RuntimeSource, Qwen38NativeFork]:
    """Build one authenticated source, one pager, and one native fork."""

    runtime = _build_source(args, recorder)
    pager: Qwen38WeightPager | None = None
    try:
        raw_config = runtime.source.reader.fetch_file("config.json")
        config_document = json.loads(raw_config)
        if not isinstance(config_document, Mapping):
            raise QwenDirectDecodeError("checkpoint config root is invalid")
        config = Qwen38Config.from_mapping(
            config_document,
            require_official=getattr(args, "_require_official", True),
        )
        pager = Qwen38WeightPager(
            runtime.source,
            device=args.device,
            compute_dtype=args.dtype,
            max_resident_bytes=args.max_resident_mb * 1024**2,
            require_source_identity=True,
            causal_tensor_reader=(
                None if runtime.mount is None else runtime.mount.tensor_reader
            ),
        )
        fork = Qwen38NativeFork(
            config,
            pager,
            native_head_crsa=native_head_crsa,
            max_seq_len=args.max_seq_len,
        )
        fork.checkpoint_preflight()
        return runtime, fork
    except Exception:
        if pager is not None:
            pager.close()
        runtime.close()
        raise


def _cleanup_native_fork(runtime: RuntimeSource, fork: Qwen38NativeFork) -> None:
    active_error = sys.exc_info()[1]
    cleanup_error: Exception | None = None
    for action in (fork.pager.close, runtime.close):
        try:
            action()
        except Exception as exc:
            if cleanup_error is None:
                cleanup_error = exc
    if active_error is None and cleanup_error is not None:
        raise cleanup_error


def _checkpoint_from_pager(pager: Qwen38WeightPager) -> dict[str, str]:
    source = pager.source
    source.inventory()
    metrics = source.metrics()
    return {
        "inventory_fingerprint": str(metrics["inventory_source_fingerprint"]),
        "repo_id": str(metrics["repo_id"]),
        "revision": str(metrics["revision"]),
    }


_FORK_TRAFFIC_BASE_FIELDS = (
    "shared_source_body_bytes",
    "off_source_body_bytes",
    "native_source_body_bytes",
    "shared_linear_calls",
    "off_linear_calls",
    "native_linear_calls",
    "shared_layers",
    "off_layers",
    "native_layers",
    "independent_complete_layers",
    "complete_layers_saved",
)


def _native_fork_traffic_record(
    value: object,
    *,
    off_forward_passes: int,
    native_forward_passes: int,
) -> dict[str, int]:
    try:
        record = {name: int(getattr(value, name)) for name in _FORK_TRAFFIC_BASE_FIELDS}
        joined_weight_passes = int(getattr(value, "fork_layer_weight_passes"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise QwenDirectDecodeError("native fork traffic receipt is invalid") from exc
    independent_weight_passes = off_forward_passes + native_forward_passes
    actual_weight_passes = independent_weight_passes - joined_weight_passes
    record.update(
        {
            "actual_fork_layer_weight_passes": actual_weight_passes,
            "complete_layers_executed": (
                record["independent_complete_layers"] - record["complete_layers_saved"]
            ),
            "fork_layer_weight_passes_saved": joined_weight_passes,
            "independent_fork_layer_weight_passes": independent_weight_passes,
            "joined_fork_layer_weight_passes": joined_weight_passes,
            "layers": (
                record["shared_layers"] + record["off_layers"] + record["native_layers"]
            ),
            "linear_calls": (
                record["shared_linear_calls"]
                + record["off_linear_calls"]
                + record["native_linear_calls"]
            ),
            "source_body_bytes": (
                record["shared_source_body_bytes"]
                + record["off_source_body_bytes"]
                + record["native_source_body_bytes"]
            ),
        }
    )
    return record


def _native_fork_arm_record(
    *,
    arm: str,
    prompt: Sequence[int],
    generated: Sequence[int],
    text: str,
    stopped_on_eos: bool,
    eos: Sequence[int],
    forward_passes: int,
    head_block_rows: int,
    vocab_size: int,
    intervention_evidence: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    generated_ids = tuple(int(token) for token in generated)
    finish_reason = "stop" if stopped_on_eos else "length"
    eos_token_id = generated_ids[-1] if stopped_on_eos else None
    record: dict[str, Any] = {
        "context_mode": "stateful_autoregressive",
        "eos_token_id": eos_token_id,
        "finish_reason": finish_reason,
        "forward_passes": forward_passes,
        "general_generation": True,
        "generated_text": text,
        "generated_token_ids": list(generated_ids),
        "head_scan_blocks": len(generated_ids)
        * math.ceil(vocab_size / head_block_rows),
        "head_scans": len(generated_ids),
        "parsed_numeric_answer": extract_gsm8k_answer(
            f"{ANSWER_GENERATION_PREFIX_LITERAL}{text}"
        ),
        "prefill_mode": "batched",
        "stateful_autoregressive": True,
        "stopped_on_eos": stopped_on_eos,
        "token_chain_sha256": _token_chain_sha256(prompt, generated_ids),
    }
    if arm == "native":
        rows = [dict(row) for row in (intervention_evidence or ())]
        record["intervention_evidence"] = rows
        record["intervention_evidence_sha256"] = _sha256(rows)
    return record


def _native_fork_identity(depth: int) -> dict[str, Any]:
    saved, independent = Qwen38NativeFork.complete_layer_savings(depth)
    return {
        "common_complete_layers": NATIVE_HEAD_CRSA_LAYER,
        "complete_layers_saved_per_joined_forward": saved,
        "fork_layer": NATIVE_HEAD_CRSA_LAYER,
        "independent_complete_layers_per_joined_forward": independent,
        "independent_fork_layer_weight_passes_per_joined_forward": 2,
        "joined_fork_layer_weight_passes_per_joined_forward": 1,
        "kind": "native-layer-27-lockstep-fork/v1",
        "permanent_split_after_token_divergence": True,
        "rejoin_after_split": False,
        "shared_checkpoint_source": True,
        "shared_weight_pager": True,
    }


def _validate_complete_causal_bundle_receipt(
    value: object,
    *,
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
    required = {
        "checkpoint_bytes",
        "graph_revision",
        "kind",
        "layout_fingerprint",
        "manifest_sha256",
        "shards",
        "shards_sha256",
        "tensor_bindings",
        "weights_layout",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("kind") != "complete-causal-bundle/v1"
        or value.get("weights_layout") not in {"flat/v1", "nested/v1"}
        or value.get("layout_fingerprint") != checkpoint.get("inventory_fingerprint")
    ):
        raise QwenDirectDecodeError("native fork requires a complete causal bundle")
    for name in ("layout_fingerprint", "manifest_sha256", "shards_sha256"):
        _digest_string(value.get(name), f"native fork bundle {name}")
    for name in ("checkpoint_bytes", "shards", "tensor_bindings"):
        if _nonnegative_count(value.get(name), f"native fork bundle {name}") < 1:
            raise QwenDirectDecodeError(
                f"native fork bundle {name} must be positive"
            )
    revision = value.get("graph_revision")
    if (
        not isinstance(revision, list)
        or len(revision) != 2
        or isinstance(revision[0], bool)
        or not isinstance(revision[0], int)
        or revision[0] < 0
    ):
        raise QwenDirectDecodeError("native fork bundle graph revision is invalid")
    _digest_string(revision[1], "native fork bundle graph revision")
    return value


def generate_native_fork(
    args: argparse.Namespace,
    *,
    runtime_factory: Callable[..., tuple[RuntimeSource, Qwen38NativeFork]]
    | None = None,
    tokenizer_factory: Callable[..., Qwen38Tokenizer] = Qwen38Tokenizer,
) -> dict[str, Any]:
    """Generate off/native arms through one real layer-27 runtime fork."""

    inputs = _load_branch_input(args.input)
    if inputs["schema"] != BRANCH_INPUT_SCHEMA_V3:
        raise QwenDirectDecodeError(
            "native fork requires a sealed answer-prefix v3 input"
        )
    if (
        getattr(args, "_require_official", True)
        and getattr(args, "causal_bundle", None) is None
    ):
        raise QwenDirectDecodeError(
            "official native fork generation requires one local complete causal bundle"
        )
    if (
        args.logical_repo_id != inputs["source"]["checkpoint"]
        or args.revision != inputs["source"]["revision"]
    ):
        raise QwenDirectDecodeError(
            "requested checkpoint/revision differs from branch input"
        )
    tokenizer, tokenizer_identity = _tokenizer_record(
        args.tokenizer_json,
        require_official=getattr(args, "_require_official", True),
        tokenizer_factory=tokenizer_factory,
    )
    if tokenizer_identity != inputs["tokenizer"]:
        raise QwenDirectDecodeError("generation tokenizer differs from branch input")
    prefix = _answer_generation_prefix(tokenizer, tokenizer_identity)
    if tuple(inputs["protocol"]["generation_prefix"]["token_ids"]) != prefix:
        raise QwenDirectDecodeError(
            "answer generation prefix differs from pinned tokenizer"
        )
    prompts = [
        _token_rows(row["effective_prompt_token_ids"], f"prompt {row['item_id']}")
        for row in inputs["items"]
    ]
    if max(len(prompt) for prompt in prompts) + args.max_new_tokens > args.max_seq_len:
        raise QwenDirectDecodeError("generation bound exceeds max_seq_len")
    intervention, intervention_identity = _build_native_intervention(args)
    recorder = AccessTraceRecorder()
    factory = _native_fork_runtime if runtime_factory is None else runtime_factory
    runtime, fork = factory(
        args,
        recorder,
        native_head_crsa=intervention,
    )
    if (
        fork.native_head_crsa is not intervention
        or fork.max_seq_len != args.max_seq_len
        or fork.pager.source is not runtime.source
    ):
        _cleanup_native_fork(runtime, fork)
        raise QwenDirectDecodeError(
            "native fork runtime attachment differs from contract"
        )
    try:
        _native_fork_source_budget_preflight(
            args, runtime.verification, items=len(prompts)
        )
    except Exception:
        _cleanup_native_fork(runtime, fork)
        raise
    started = time.perf_counter()
    source_start = _source_metric(fork.pager.source, "network_or_source_body_bytes")
    items: list[dict[str, Any]] = []
    try:
        eos = tuple(
            int(value) for value in inputs["protocol"]["accepted_eos_token_ids"]
        )
        if (
            max((*eos, *(token for prompt in prompts for token in prompt)))
            >= fork.config.vocab_size
        ):
            raise QwenDirectDecodeError(
                "native fork token exceeds checkpoint vocabulary"
            )
        for source_row, prompt in zip(inputs["items"], prompts, strict=True):
            head_blocks = 0
            head_scans = 0

            def head_progress(event: Mapping[str, int]) -> None:
                nonlocal head_blocks, head_scans
                head_blocks += 1
                if event.get("rows_done") == event.get("vocab_rows"):
                    head_scans += 1

            try:
                generated = fork.generate_greedy(
                    [prompt],
                    max_new_tokens=args.max_new_tokens,
                    eos_token_ids=eos,
                    head_block_rows=args.head_block_rows,
                    progress=_progress,
                    head_progress=head_progress,
                )
            except Exception as exc:
                raise QwenDirectDecodeError(
                    f"native fork generation failed for {source_row['item_id']}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            evidence = generated.evidence
            off_ids = tuple(int(token) for token in generated.off_token_ids)
            native_ids = tuple(int(token) for token in generated.native_token_ids)
            if (
                not off_ids
                or not native_ids
                or tuple(evidence.prompt_token_ids) != prompt
                or tuple(evidence.off_generated_token_ids) != off_ids
                or tuple(evidence.native_generated_token_ids) != native_ids
            ):
                raise QwenDirectDecodeError("native fork generation evidence differs")
            if head_scans != len(off_ids) + len(native_ids):
                raise QwenDirectDecodeError(
                    "native fork LM-head scan count is incomplete"
                )
            expected_head_blocks = head_scans * math.ceil(
                fork.config.vocab_size / args.head_block_rows
            )
            if head_blocks != expected_head_blocks:
                raise QwenDirectDecodeError(
                    "native fork LM-head block accounting differs"
                )
            off_text = tokenizer.decode(off_ids)
            native_text = tokenizer.decode(native_ids)
            if "####" in off_text or "####" in native_text:
                raise QwenDirectDecodeError(
                    "answer fork generated text re-emits the sealed #### prefix"
                )
            evidence_rows = [
                row.to_dict() for row in evidence.traffic.native_head_crsa_evidence
            ]
            off = _native_fork_arm_record(
                arm="off",
                prompt=prompt,
                generated=off_ids,
                text=off_text,
                stopped_on_eos=bool(evidence.off_stopped_on_eos),
                eos=eos,
                forward_passes=int(evidence.off_forward_passes),
                head_block_rows=args.head_block_rows,
                vocab_size=fork.config.vocab_size,
            )
            native = _native_fork_arm_record(
                arm="native",
                prompt=prompt,
                generated=native_ids,
                text=native_text,
                stopped_on_eos=bool(evidence.native_stopped_on_eos),
                eos=eos,
                forward_passes=int(evidence.native_forward_passes),
                head_block_rows=args.head_block_rows,
                vocab_size=fork.config.vocab_size,
                intervention_evidence=evidence_rows,
            )
            first = _first_divergence(off_ids, native_ids)
            if first != evidence.first_divergence_step:
                raise QwenDirectDecodeError("native fork divergence evidence differs")
            traffic = _native_fork_traffic_record(
                evidence.traffic,
                off_forward_passes=off["forward_passes"],
                native_forward_passes=native["forward_passes"],
            )
            item_fork = {
                "final_joined": bool(evidence.final_joined),
                "first_divergence_index": first,
                "first_divergence_position": generated.state.first_divergence_position,
                "initially_joined": True,
                "joined_pair_forward_passes": traffic[
                    "joined_fork_layer_weight_passes"
                ],
                "permanently_split": not bool(evidence.final_joined),
                "rejoined_after_split": False,
            }
            items.append(
                {
                    "fork": item_fork,
                    "item_id": str(source_row["item_id"]),
                    "native": native,
                    "off": off,
                    "prompt_token_ids": list(prompt),
                    "seconds": float(evidence.seconds),
                    "traffic": traffic,
                }
            )
        checkpoint = _checkpoint_from_pager(fork.pager)
        if (
            checkpoint["repo_id"] != inputs["source"]["checkpoint"]
            or checkpoint["revision"] != inputs["source"]["revision"]
        ):
            raise QwenDirectDecodeError("runtime checkpoint differs from branch input")
        trace = _trace_receipt(recorder, args.access_trace)
        generation = {
            "algorithm": "exact-greedy/v1",
            "eos_token_ids": list(eos),
            "general_generation": True,
            "head_block_rows": args.head_block_rows,
            "max_new_tokens": args.max_new_tokens,
            "max_seq_len": args.max_seq_len,
            "prefill_mode": "batched-per-item",
            "stateful_autoregressive": True,
            "teacher_forced_tokens_after_prompt": 0,
            "temperature": 0.0,
        }
        execution = {
            "checkpoint_layers": int(fork.config.n_layers),
            "cohort_size": len(items),
            "device": str(fork.pager.device),
            "dtype": str(fork.pager.compute_dtype).removeprefix("torch."),
            "execution_mode": "serial_items/native-lockstep-fork",
            "max_batch_size": 1,
            "model_views": 2,
            "runtime_instances": 1,
            "shared_pair_prefills": len(items),
            "source_instances": 1,
            "weight_pagers": 1,
        }
        sum_fields = (
            *_FORK_TRAFFIC_BASE_FIELDS,
            "actual_fork_layer_weight_passes",
            "complete_layers_executed",
            "fork_layer_weight_passes_saved",
            "independent_fork_layer_weight_passes",
            "joined_fork_layer_weight_passes",
            "layers",
            "linear_calls",
            "source_body_bytes",
        )
        traffic: dict[str, Any] = {
            name: sum(row["traffic"][name] for row in items) for name in sum_fields
        }
        traffic.update(
            {
                "access_trace": trace,
                "access_trace_sha256": trace["sha256"],
                "forward_passes": sum(
                    row[arm]["forward_passes"]
                    for row in items
                    for arm in ("off", "native")
                ),
                "generation_seconds": sum(row["seconds"] for row in items),
                "head_scan_blocks": sum(
                    row[arm]["head_scan_blocks"]
                    for row in items
                    for arm in ("off", "native")
                ),
                "head_scans": sum(
                    row[arm]["head_scans"] for row in items for arm in ("off", "native")
                ),
                "native_forward_passes": sum(
                    row["native"]["forward_passes"] for row in items
                ),
                "off_forward_passes": sum(
                    row["off"]["forward_passes"] for row in items
                ),
                "runtime_source_body_bytes": (
                    _source_metric(fork.pager.source, "network_or_source_body_bytes")
                    - source_start
                ),
                "wall_seconds": time.perf_counter() - started,
            }
        )
        identity = {
            "bundle": dict(runtime.verification or {}),
            "checkpoint": checkpoint,
            "execution": execution,
            "fork": _native_fork_identity(fork.config.n_layers),
            "generation": generation,
            "input": _branch_input_identity(inputs),
            "intervention": intervention_identity,
            "items": items,
            "schema": NATIVE_FORK_PAIR_SCHEMA,
            "status": "sealed",
            "tokenizer": tokenizer_identity,
            "traffic": traffic,
        }
        result = _result(identity)
        return _validate_native_fork_pair_document(result)
    finally:
        _cleanup_native_fork(runtime, fork)


_NATIVE_FORK_ITEM_TRAFFIC_FIELDS = frozenset(
    {
        *_FORK_TRAFFIC_BASE_FIELDS,
        "actual_fork_layer_weight_passes",
        "complete_layers_executed",
        "fork_layer_weight_passes_saved",
        "independent_fork_layer_weight_passes",
        "joined_fork_layer_weight_passes",
        "layers",
        "linear_calls",
        "source_body_bytes",
    }
)
_NATIVE_FORK_ARM_FIELDS = frozenset(
    {
        "context_mode",
        "eos_token_id",
        "finish_reason",
        "forward_passes",
        "general_generation",
        "generated_text",
        "generated_token_ids",
        "head_scan_blocks",
        "head_scans",
        "parsed_numeric_answer",
        "prefill_mode",
        "stateful_autoregressive",
        "stopped_on_eos",
        "token_chain_sha256",
    }
)


def _validate_native_fork_arm_record(
    value: object,
    *,
    arm: str,
    item_id: str,
    prompt: tuple[int, ...],
    eos: tuple[int, ...],
    generation: Mapping[str, Any],
    tokenizer_vocab: int,
) -> Mapping[str, Any]:
    required = set(_NATIVE_FORK_ARM_FIELDS)
    if arm == "native":
        required.update({"intervention_evidence", "intervention_evidence_sha256"})
    if not isinstance(value, Mapping) or set(value) != required:
        raise QwenDirectDecodeError(f"native fork {arm} token record is invalid")
    generated = _token_rows(value.get("generated_token_ids"), f"fork {arm} tokens")
    max_new_tokens = int(generation["max_new_tokens"])
    if (
        len(generated) > max_new_tokens
        or max((*prompt, *generated)) >= tokenizer_vocab
        or len(prompt) + len(generated) > generation["max_seq_len"]
    ):
        raise QwenDirectDecodeError(f"native fork {arm} token bound was exceeded")
    text = value.get("generated_text")
    if not isinstance(text, str) or "####" in text:
        raise QwenDirectDecodeError(
            f"native fork {arm} text violates answer-prefix semantics"
        )
    if value.get("parsed_numeric_answer") != extract_gsm8k_answer(
        f"{ANSWER_GENERATION_PREFIX_LITERAL}{text}"
    ):
        raise QwenDirectDecodeError(
            f"native fork {arm} parsed numeric answer is inconsistent"
        )
    if value.get("token_chain_sha256") != _token_chain_sha256(prompt, generated):
        raise QwenDirectDecodeError(f"native fork {arm} token chain differs")
    stopped = value.get("stopped_on_eos")
    if not isinstance(stopped, bool):
        raise QwenDirectDecodeError(f"native fork {arm} EOS state is invalid")
    if stopped:
        if (
            value.get("finish_reason") != "stop"
            or value.get("eos_token_id") != generated[-1]
            or generated[-1] not in eos
        ):
            raise QwenDirectDecodeError(f"native fork {arm} EOS evidence differs")
    elif (
        value.get("finish_reason") != "length"
        or value.get("eos_token_id") is not None
        or len(generated) != max_new_tokens
    ):
        raise QwenDirectDecodeError(f"native fork {arm} length evidence differs")
    if (
        value.get("context_mode") != "stateful_autoregressive"
        or value.get("general_generation") is not True
        or value.get("stateful_autoregressive") is not True
        or value.get("prefill_mode") != "batched"
        or value.get("forward_passes") != len(generated) + 1
        or value.get("head_scans") != len(generated)
        or value.get("head_scan_blocks")
        != len(generated) * math.ceil(tokenizer_vocab / generation["head_block_rows"])
    ):
        raise QwenDirectDecodeError(
            f"native fork {arm} autoregressive evidence differs"
        )
    for name in ("forward_passes", "head_scan_blocks", "head_scans"):
        _nonnegative_count(value.get(name), f"native fork {arm} {name}")
    if arm == "native":
        _validate_native_evidence_chain(
            {
                "forward_passes": value["forward_passes"],
                "intervention_evidence": value["intervention_evidence"],
                "intervention_evidence_sha256": value["intervention_evidence_sha256"],
                "item_id": item_id,
                "prompt_token_ids": list(prompt),
            }
        )
    return value


def _validate_native_fork_item_traffic(
    value: object,
    *,
    depth: int,
    off_forward_passes: int,
    native_forward_passes: int,
    joined_forward_passes: int,
) -> Mapping[str, int]:
    if not isinstance(value, Mapping) or set(value) != _NATIVE_FORK_ITEM_TRAFFIC_FIELDS:
        raise QwenDirectDecodeError("native fork item traffic schema is invalid")
    for name in value:
        _nonnegative_count(value.get(name), f"native fork item traffic {name}")
    expected = {
        "independent_complete_layers": (off_forward_passes + native_forward_passes)
        * depth,
        "complete_layers_saved": joined_forward_passes * NATIVE_HEAD_CRSA_LAYER,
        "independent_fork_layer_weight_passes": (
            off_forward_passes + native_forward_passes
        ),
        "joined_fork_layer_weight_passes": joined_forward_passes,
        "fork_layer_weight_passes_saved": joined_forward_passes,
        "actual_fork_layer_weight_passes": (
            off_forward_passes + native_forward_passes - joined_forward_passes
        ),
        "shared_layers": joined_forward_passes * NATIVE_HEAD_CRSA_LAYER,
        "off_layers": (
            joined_forward_passes * (depth - NATIVE_HEAD_CRSA_LAYER)
            + (off_forward_passes - joined_forward_passes) * depth
        ),
        "native_layers": (
            joined_forward_passes * (depth - NATIVE_HEAD_CRSA_LAYER)
            + (native_forward_passes - joined_forward_passes) * depth
        ),
    }
    expected["complete_layers_executed"] = (
        expected["independent_complete_layers"] - expected["complete_layers_saved"]
    )
    expected["layers"] = expected["complete_layers_executed"]
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise QwenDirectDecodeError(
                f"native fork item traffic {name} is inconsistent"
            )
    for total, fields in {
        "source_body_bytes": (
            "shared_source_body_bytes",
            "off_source_body_bytes",
            "native_source_body_bytes",
        ),
        "linear_calls": (
            "shared_linear_calls",
            "off_linear_calls",
            "native_linear_calls",
        ),
        "layers": ("shared_layers", "off_layers", "native_layers"),
    }.items():
        if value[total] != sum(value[name] for name in fields):
            raise QwenDirectDecodeError(
                f"native fork item traffic {total} sum is inconsistent"
            )
    return value


def _validate_native_fork_pair_document(
    document: dict[str, Any],
    *,
    access_trace_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    required = {
        "bundle",
        "checkpoint",
        "execution",
        "fork",
        "generation",
        "input",
        "intervention",
        "items",
        "schema",
        "sha256",
        "status",
        "tokenizer",
        "traffic",
    }
    if (
        set(document) != required
        or document.get("schema") != NATIVE_FORK_PAIR_SCHEMA
        or document.get("status") != "sealed"
    ):
        raise QwenDirectDecodeError("native fork pair schema is invalid")
    _verify_document_seal(document, "native fork pair")
    _validate_native_intervention_identity(document.get("intervention"))
    inputs = document.get("input")
    if not isinstance(inputs, dict):
        raise QwenDirectDecodeError("native fork pair input is invalid")
    _validate_branch_input(inputs)
    if inputs["schema"] != BRANCH_INPUT_SCHEMA_V3:
        raise QwenDirectDecodeError("native fork pair input is not answer-prefix v3")
    checkpoint = document.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "inventory_fingerprint",
        "repo_id",
        "revision",
    }:
        raise QwenDirectDecodeError("native fork checkpoint identity is invalid")
    if any(not isinstance(value, str) or not value for value in checkpoint.values()):
        raise QwenDirectDecodeError("native fork checkpoint identity is invalid")
    if (
        checkpoint["repo_id"] != inputs["source"]["checkpoint"]
        or checkpoint["revision"] != inputs["source"]["revision"]
    ):
        raise QwenDirectDecodeError("native fork checkpoint differs from input")
    bundle = document.get("bundle")
    _validate_complete_causal_bundle_receipt(bundle, checkpoint=checkpoint)
    tokenizer = document.get("tokenizer")
    if tokenizer != inputs["tokenizer"]:
        raise QwenDirectDecodeError("native fork tokenizer differs from input")
    if not isinstance(tokenizer, Mapping):  # pragma: no cover - implied above.
        raise QwenDirectDecodeError("native fork tokenizer identity is invalid")
    tokenizer_vocab = _nonnegative_count(
        tokenizer.get("vocab_size"), "native fork tokenizer vocabulary"
    )
    generation = document.get("generation")
    generation_required = {
        "algorithm",
        "eos_token_ids",
        "general_generation",
        "head_block_rows",
        "max_new_tokens",
        "max_seq_len",
        "prefill_mode",
        "stateful_autoregressive",
        "teacher_forced_tokens_after_prompt",
        "temperature",
    }
    if not isinstance(generation, Mapping) or set(generation) != generation_required:
        raise QwenDirectDecodeError("native fork generation identity is invalid")
    eos = _token_rows(generation.get("eos_token_ids"), "native fork EOS IDs")
    if (
        list(eos) != inputs["protocol"]["accepted_eos_token_ids"]
        or generation.get("algorithm") != "exact-greedy/v1"
        or generation.get("general_generation") is not True
        or generation.get("stateful_autoregressive") is not True
        or generation.get("teacher_forced_tokens_after_prompt") != 0
        or generation.get("temperature") != 0.0
        or generation.get("prefill_mode") != "batched-per-item"
        or _nonnegative_count(generation.get("max_new_tokens"), "max_new_tokens") < 1
        or _nonnegative_count(generation.get("max_seq_len"), "max_seq_len") < 1
        or _nonnegative_count(generation.get("head_block_rows"), "head_block_rows") < 1
    ):
        raise QwenDirectDecodeError("native fork generation contract differs")
    execution = document.get("execution")
    execution_fields = {
        "checkpoint_layers",
        "cohort_size",
        "device",
        "dtype",
        "execution_mode",
        "max_batch_size",
        "model_views",
        "runtime_instances",
        "shared_pair_prefills",
        "source_instances",
        "weight_pagers",
    }
    rows = document.get("items")
    if (
        not isinstance(execution, Mapping)
        or set(execution) != execution_fields
        or not isinstance(rows, list)
        or not rows
    ):
        raise QwenDirectDecodeError("native fork execution structure is invalid")
    depth = _nonnegative_count(execution.get("checkpoint_layers"), "fork depth")
    if (
        depth <= NATIVE_HEAD_CRSA_LAYER
        or execution.get("cohort_size") != len(rows)
        or execution.get("shared_pair_prefills") != len(rows)
        or execution.get("execution_mode") != "serial_items/native-lockstep-fork"
        or execution.get("max_batch_size") != 1
        or execution.get("model_views") != 2
        or execution.get("runtime_instances") != 1
        or execution.get("source_instances") != 1
        or execution.get("weight_pagers") != 1
        or not isinstance(execution.get("device"), str)
        or not execution["device"]
        or not isinstance(execution.get("dtype"), str)
        or not execution["dtype"]
    ):
        raise QwenDirectDecodeError("native fork execution accounting differs")
    if document.get("fork") != _native_fork_identity(depth):
        raise QwenDirectDecodeError("native fork identity is not canonical")
    item_ids = inputs["selection"]["item_ids"]
    item_traffic: list[Mapping[str, int]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "fork",
            "item_id",
            "native",
            "off",
            "prompt_token_ids",
            "seconds",
            "traffic",
        }:
            raise QwenDirectDecodeError("native fork generated item schema is invalid")
        if row.get("item_id") != item_ids[index]:
            raise QwenDirectDecodeError("native fork generated item order differs")
        prompt = _token_rows(row.get("prompt_token_ids"), "native fork prompt")
        sealed = inputs["items"][index]
        if (
            row["item_id"] != sealed["item_id"]
            or list(prompt) != sealed["effective_prompt_token_ids"]
        ):
            raise QwenDirectDecodeError("native fork prompt differs from input")
        off = _validate_native_fork_arm_record(
            row.get("off"),
            arm="off",
            item_id=row["item_id"],
            prompt=prompt,
            eos=eos,
            generation=generation,
            tokenizer_vocab=tokenizer_vocab,
        )
        native = _validate_native_fork_arm_record(
            row.get("native"),
            arm="native",
            item_id=row["item_id"],
            prompt=prompt,
            eos=eos,
            generation=generation,
            tokenizer_vocab=tokenizer_vocab,
        )
        first = _first_divergence(
            off["generated_token_ids"], native["generated_token_ids"]
        )
        fork_receipt = row.get("fork")
        joined = first + 1 if first is not None else 1 + len(off["generated_token_ids"])
        expected_fork = {
            "final_joined": first is None,
            "first_divergence_index": first,
            "first_divergence_position": (
                None if first is None else len(prompt) + first
            ),
            "initially_joined": True,
            "joined_pair_forward_passes": joined,
            "permanently_split": first is not None,
            "rejoined_after_split": False,
        }
        if fork_receipt != expected_fork:
            raise QwenDirectDecodeError("native fork split evidence is inconsistent")
        traffic = _validate_native_fork_item_traffic(
            row.get("traffic"),
            depth=depth,
            off_forward_passes=off["forward_passes"],
            native_forward_passes=native["forward_passes"],
            joined_forward_passes=joined,
        )
        item_traffic.append(traffic)
        _finite_nonnegative(row.get("seconds"), "native fork item seconds")
    traffic = document.get("traffic")
    aggregate_fields = {
        *_NATIVE_FORK_ITEM_TRAFFIC_FIELDS,
        "access_trace",
        "access_trace_sha256",
        "forward_passes",
        "generation_seconds",
        "head_scan_blocks",
        "head_scans",
        "native_forward_passes",
        "off_forward_passes",
        "runtime_source_body_bytes",
        "wall_seconds",
    }
    if not isinstance(traffic, Mapping) or set(traffic) != aggregate_fields:
        raise QwenDirectDecodeError("native fork aggregate traffic schema is invalid")
    _verify_access_trace_receipt(
        traffic.get("access_trace"), artifact_path=access_trace_path
    )
    if traffic.get("access_trace_sha256") != traffic["access_trace"]["sha256"]:
        raise QwenDirectDecodeError("native fork trace digest differs")
    expected_sums: dict[str, int | float] = {
        name: sum(row[name] for row in item_traffic)
        for name in _NATIVE_FORK_ITEM_TRAFFIC_FIELDS
    }
    expected_sums.update(
        {
            "forward_passes": sum(
                row[arm]["forward_passes"] for row in rows for arm in ("off", "native")
            ),
            "generation_seconds": sum(row["seconds"] for row in rows),
            "head_scan_blocks": sum(
                row[arm]["head_scan_blocks"]
                for row in rows
                for arm in ("off", "native")
            ),
            "head_scans": sum(
                row[arm]["head_scans"] for row in rows for arm in ("off", "native")
            ),
            "native_forward_passes": sum(
                row["native"]["forward_passes"] for row in rows
            ),
            "off_forward_passes": sum(row["off"]["forward_passes"] for row in rows),
        }
    )
    for name, expected in expected_sums.items():
        if traffic.get(name) != expected:
            raise QwenDirectDecodeError(
                f"native fork aggregate traffic {name} is inconsistent"
            )
    for name in (
        "runtime_source_body_bytes",
        "wall_seconds",
    ):
        if name == "wall_seconds":
            _finite_nonnegative(traffic.get(name), f"native fork {name}")
        else:
            _nonnegative_count(traffic.get(name), f"native fork {name}")
    if traffic["runtime_source_body_bytes"] != traffic["source_body_bytes"]:
        raise QwenDirectDecodeError("native fork runtime/source bytes differ")
    if traffic["wall_seconds"] < traffic["generation_seconds"]:
        raise QwenDirectDecodeError("native fork wall timing is below model timing")
    return document


def _load_native_fork_pair(path: str | os.PathLike[str]) -> dict[str, Any]:
    document = _strict_json(path)
    return _validate_native_fork_pair_document(
        document,
        access_trace_path=_relocated_access_trace_path(path, document),
    )


_REFERENCE_TOKEN_FIELDS = (
    "eos_token_id",
    "finish_reason",
    "forward_passes",
    "generated_text",
    "generated_token_ids",
    "parsed_numeric_answer",
    "stopped_on_eos",
    "token_chain_sha256",
)


def _reference_token_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value[name] for name in _REFERENCE_TOKEN_FIELDS}


def _saving_record(
    reference: int,
    fork: int,
    *,
    measurement: str,
) -> dict[str, Any]:
    return {
        "fork": fork,
        "measurement": measurement,
        "reference": reference,
        "saved": reference - fork,
    }


def compare_native_fork_references(args: argparse.Namespace) -> dict[str, Any]:
    """Prove fork parity against independently sealed off-v2/native-v3 arms."""

    pair = _load_native_fork_pair(args.pair)
    off = _load_branch_result(args.off)
    native = _load_native_branch_result(args.native)
    if off.get("arm") != "off" or native.get("arm") != "native-crsa":
        raise QwenDirectDecodeError(
            "native fork comparison requires off-v2 and native-v3 references"
        )
    if off["input"]["schema"] != BRANCH_INPUT_SCHEMA_V3:
        raise QwenDirectDecodeError("native fork off reference is not answer-prefix v3")
    for name in ("bundle", "checkpoint", "input", "tokenizer", "generation"):
        if off[name] != native[name] or pair[name] != off[name]:
            raise QwenDirectDecodeError(
                f"native fork/reference {name} identity differs"
            )
    if pair["intervention"] != native["intervention"]:
        raise QwenDirectDecodeError("native fork/reference intervention differs")
    if (
        off["execution"]["checkpoint_layers"] != pair["execution"]["checkpoint_layers"]
        or native["execution"]["checkpoint_layers"]
        != pair["execution"]["checkpoint_layers"]
    ):
        raise QwenDirectDecodeError("native fork/reference checkpoint depth differs")
    if not (len(pair["items"]) == len(off["items"]) == len(native["items"])):
        raise QwenDirectDecodeError("native fork/reference item counts differ")
    items: list[dict[str, Any]] = []
    for pair_row, off_row, native_row in zip(
        pair["items"], off["items"], native["items"], strict=True
    ):
        if (
            pair_row["item_id"] != off_row["item_id"]
            or pair_row["item_id"] != native_row["item_id"]
            or pair_row["prompt_token_ids"] != off_row["prompt_token_ids"]
            or pair_row["prompt_token_ids"] != native_row["prompt_token_ids"]
        ):
            raise QwenDirectDecodeError("native fork/reference item identity differs")
        pair_off = _reference_token_projection(pair_row["off"])
        pair_native = _reference_token_projection(pair_row["native"])
        reference_off = _reference_token_projection(off_row)
        reference_native = _reference_token_projection(native_row)
        if pair_off != reference_off:
            raise QwenDirectDecodeError(
                f"native fork off parity differs for {pair_row['item_id']}"
            )
        if pair_native != reference_native:
            raise QwenDirectDecodeError(
                f"native fork native parity differs for {pair_row['item_id']}"
            )
        items.append(
            {
                "fork": pair_row["fork"],
                "item_id": pair_row["item_id"],
                "native": pair_native,
                "off": pair_off,
                "prompt_token_ids": pair_row["prompt_token_ids"],
                "reference_parity": True,
            }
        )
    reference_forward_passes = (
        off["traffic"]["forward_passes"] + native["traffic"]["forward_passes"]
    )
    depth = pair["execution"]["checkpoint_layers"]
    reference_complete_layers = reference_forward_passes * depth
    savings = {
        "complete_layers": _saving_record(
            reference_complete_layers,
            pair["traffic"]["complete_layers_executed"],
            measurement="sealed-forward-count-times-checkpoint-depth/v1",
        ),
        "layer_weight_passes": _saving_record(
            reference_complete_layers,
            pair["traffic"]["complete_layers_executed"]
            - pair["traffic"]["joined_fork_layer_weight_passes"],
            measurement="complete-layers-minus-shared-fork-pass/v1",
        ),
        "fork_layer_weight_passes": _saving_record(
            reference_forward_passes,
            pair["traffic"]["actual_fork_layer_weight_passes"],
            measurement="sealed-fork-layer-pass-accounting/v1",
        ),
        "linear_calls": _saving_record(
            off["traffic"]["linear_calls"] + native["traffic"]["linear_calls"],
            pair["traffic"]["linear_calls"],
            measurement="runtime-linear-call-counters/v1",
        ),
        "source_body_bytes": _saving_record(
            off["traffic"]["source_body_bytes"]
            + native["traffic"]["source_body_bytes"],
            pair["traffic"]["source_body_bytes"],
            measurement="runtime-source-body-byte-counters/v1",
        ),
    }
    identity = {
        "identity": {
            "bundle": pair["bundle"],
            "checkpoint": pair["checkpoint"],
            "fork": pair["fork"],
            "generation": pair["generation"],
            "input_sha256": pair["input"]["sha256"],
            "intervention": pair["intervention"],
            "tokenizer": pair["tokenizer"],
        },
        "items": items,
        "pair": {
            "result_sha256": pair["sha256"],
            "traffic": pair["traffic"],
            "traffic_sha256": _sha256(pair["traffic"]),
        },
        "references": {
            "native": {
                "result_sha256": native["sha256"],
                "traffic": native["traffic"],
                "traffic_sha256": _sha256(native["traffic"]),
            },
            "off": {
                "result_sha256": off["sha256"],
                "traffic": off["traffic"],
                "traffic_sha256": _sha256(off["traffic"]),
            },
        },
        "savings": savings,
        "schema": NATIVE_FORK_REFERENCE_COMPARISON_SCHEMA,
        "status": "sealed",
    }
    result = _result(identity)
    return _validate_native_fork_reference_comparison_document(result)


def _validate_native_fork_reference_comparison_document(
    document: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "identity",
        "items",
        "pair",
        "references",
        "savings",
        "schema",
        "sha256",
        "status",
    }
    if (
        set(document) != required
        or document.get("schema") != NATIVE_FORK_REFERENCE_COMPARISON_SCHEMA
        or document.get("status") != "sealed"
    ):
        raise QwenDirectDecodeError(
            "native fork reference comparison schema is invalid"
        )
    _verify_document_seal(document, "native fork reference comparison")
    identity = document.get("identity")
    if not isinstance(identity, Mapping) or set(identity) != {
        "bundle",
        "checkpoint",
        "fork",
        "generation",
        "input_sha256",
        "intervention",
        "tokenizer",
    }:
        raise QwenDirectDecodeError("native fork comparison identity is invalid")
    _digest_string(identity.get("input_sha256"), "native fork comparison input")
    _validate_native_intervention_identity(identity.get("intervention"))
    checkpoint = identity.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise QwenDirectDecodeError("native fork comparison checkpoint is invalid")
    _validate_complete_causal_bundle_receipt(
        identity.get("bundle"), checkpoint=checkpoint
    )
    fork_identity = identity.get("fork")
    independent = (
        fork_identity.get("independent_complete_layers_per_joined_forward")
        if isinstance(fork_identity, Mapping)
        else None
    )
    if isinstance(independent, bool) or not isinstance(independent, int):
        raise QwenDirectDecodeError("native fork comparison depth is invalid")
    depth = independent // 2
    if independent != 2 * depth or fork_identity != _native_fork_identity(depth):
        raise QwenDirectDecodeError("native fork comparison identity is inconsistent")
    pair = document.get("pair")
    references = document.get("references")
    if (
        not isinstance(pair, Mapping)
        or set(pair) != {"result_sha256", "traffic", "traffic_sha256"}
        or not isinstance(references, Mapping)
        or set(references) != {"off", "native"}
    ):
        raise QwenDirectDecodeError("native fork comparison receipts are invalid")
    receipts: dict[str, Mapping[str, Any]] = {}
    for name, receipt in (("pair", pair), *references.items()):
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != {"result_sha256", "traffic", "traffic_sha256"}
            or not isinstance(receipt.get("traffic"), Mapping)
        ):
            raise QwenDirectDecodeError(f"native fork {name} receipt is invalid")
        _digest_string(receipt.get("result_sha256"), f"native fork {name} result")
        if receipt.get("traffic_sha256") != _sha256(receipt["traffic"]):
            raise QwenDirectDecodeError(f"native fork {name} traffic digest differs")
        receipts[name] = receipt["traffic"]
    rows = document.get("items")
    if (
        not isinstance(rows, list)
        or not rows
        or any(
            not isinstance(row, Mapping) or row.get("reference_parity") is not True
            for row in rows
        )
    ):
        raise QwenDirectDecodeError("native fork comparison parity is invalid")
    forwards = {
        arm: sum(row[arm]["forward_passes"] for row in rows)
        for arm in ("off", "native")
    }
    for arm in ("off", "native"):
        if receipts[arm].get("forward_passes") != forwards[arm]:
            raise QwenDirectDecodeError(f"native fork {arm} traffic is inconsistent")
    total = forwards["off"] + forwards["native"]
    pair_traffic = receipts["pair"]
    if (
        pair_traffic.get("off_forward_passes") != forwards["off"]
        or pair_traffic.get("native_forward_passes") != forwards["native"]
        or pair_traffic.get("forward_passes") != total
        or pair_traffic.get("independent_complete_layers") != total * depth
    ):
        raise QwenDirectDecodeError("native fork pair traffic is inconsistent")
    expected = {
        "complete_layers": _saving_record(
            total * depth,
            _nonnegative_count(
                pair_traffic.get("complete_layers_executed"), "pair complete layers"
            ),
            measurement="sealed-forward-count-times-checkpoint-depth/v1",
        ),
        "layer_weight_passes": _saving_record(
            total * depth,
            _nonnegative_count(
                pair_traffic.get("complete_layers_executed"), "pair complete layers"
            )
            - _nonnegative_count(
                pair_traffic.get("joined_fork_layer_weight_passes"),
                "pair joined fork-layer passes",
            ),
            measurement="complete-layers-minus-shared-fork-pass/v1",
        ),
        "fork_layer_weight_passes": _saving_record(
            total,
            _nonnegative_count(
                pair_traffic.get("actual_fork_layer_weight_passes"),
                "pair fork-layer passes",
            ),
            measurement="sealed-fork-layer-pass-accounting/v1",
        ),
        "linear_calls": _saving_record(
            sum(
                _nonnegative_count(receipts[arm].get("linear_calls"), arm)
                for arm in ("off", "native")
            ),
            _nonnegative_count(pair_traffic.get("linear_calls"), "pair linears"),
            measurement="runtime-linear-call-counters/v1",
        ),
        "source_body_bytes": _saving_record(
            sum(
                _nonnegative_count(receipts[arm].get("source_body_bytes"), arm)
                for arm in ("off", "native")
            ),
            _nonnegative_count(pair_traffic.get("source_body_bytes"), "pair bytes"),
            measurement="runtime-source-body-byte-counters/v1",
        ),
    }
    if document.get("savings") != expected:
        raise QwenDirectDecodeError("native fork measured savings are inconsistent")
    return document


def _load_native_fork_reference_comparison(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    return _validate_native_fork_reference_comparison_document(_strict_json(path))


def _paired_identity(
    off: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    fields = ("checkpoint", "bundle", "input", "tokenizer", "generation", "execution")
    identity = {name: off[name] for name in fields}
    for name in fields:
        if candidate[name] != identity[name]:
            raise QwenDirectDecodeError(f"generated arm {name} identity differs")
    return identity


def _first_divergence(left: Sequence[int], right: Sequence[int]) -> int | None:
    common = min(len(left), len(right))
    for index in range(common):
        if left[index] != right[index]:
            return index
    return None if len(left) == len(right) else common


def _compare_branch_documents(
    off: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    if off.get("arm") != "off" or candidate.get("arm") != "stable-crsa":
        raise QwenDirectDecodeError("comparison requires off and stable-crsa arms")
    identity = _paired_identity(off, candidate)
    off_rows = off["items"]
    candidate_rows = candidate["items"]
    if len(off_rows) != len(candidate_rows):  # identity normally catches this.
        raise QwenDirectDecodeError("generated arm item counts differ")
    items: list[dict[str, Any]] = []
    divergences = 0
    parsed_answer_changes = 0
    for off_row, candidate_row in zip(off_rows, candidate_rows, strict=True):
        if (
            off_row["item_id"] != candidate_row["item_id"]
            or off_row["prompt_token_ids"] != candidate_row["prompt_token_ids"]
        ):
            raise QwenDirectDecodeError("generated arm item or prompt identity differs")
        off_tokens = off_row["generated_token_ids"]
        candidate_tokens = candidate_row["generated_token_ids"]
        first = _first_divergence(off_tokens, candidate_tokens)
        diverged = first is not None
        divergences += diverged
        answer_changed = (
            off_row["parsed_numeric_answer"] != candidate_row["parsed_numeric_answer"]
        )
        parsed_answer_changes += answer_changed
        items.append(
            {
                "candidate_finish_reason": candidate_row["finish_reason"],
                "candidate_generated_text": candidate_row["generated_text"],
                "candidate_generated_token_ids": candidate_tokens,
                "candidate_parsed_numeric_answer": candidate_row[
                    "parsed_numeric_answer"
                ],
                "candidate_token_chain_sha256": candidate_row["token_chain_sha256"],
                "diverged": diverged,
                "first_divergence_index": first,
                "item_id": off_row["item_id"],
                "off_finish_reason": off_row["finish_reason"],
                "off_generated_text": off_row["generated_text"],
                "off_generated_token_ids": off_tokens,
                "off_parsed_numeric_answer": off_row["parsed_numeric_answer"],
                "off_token_chain_sha256": off_row["token_chain_sha256"],
                "parsed_answer_changed": answer_changed,
            }
        )
    identity_document = {
        "candidate": {
            "graft": candidate["graft"],
            "result_sha256": candidate["sha256"],
            "traffic": candidate["traffic"],
        },
        "comparison": {
            "branch_effect_observed": divergences > 0,
            "diverged_items": divergences,
            "items": len(items),
            "parsed_answer_changes": parsed_answer_changes,
            "verdict": "diverged" if divergences else "no_divergence",
        },
        "identity": identity,
        "items": items,
        "off": {
            "graft": off["graft"],
            "result_sha256": off["sha256"],
            "traffic": off["traffic"],
        },
        "schema": BRANCH_COMPARISON_SCHEMA,
        "status": "sealed",
    }
    return _result(identity_document)


def compare_generated_arms(args: argparse.Namespace) -> dict[str, Any]:
    return _compare_branch_documents(
        _load_branch_result(args.off),
        _load_branch_result(args.candidate),
    )


def _load_branch_comparison(path: str | os.PathLike[str]) -> dict[str, Any]:
    document = _strict_json(path)
    if (
        document.get("schema") != BRANCH_COMPARISON_SCHEMA
        or document.get("status") != "sealed"
        or set(document)
        != {
            "candidate",
            "comparison",
            "identity",
            "items",
            "off",
            "schema",
            "sha256",
            "status",
        }
    ):
        raise QwenDirectDecodeError("branch comparison schema is invalid")
    _verify_document_seal(document, "branch comparison")
    return document


def _triad_pair_report(
    off: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    arm: str,
) -> dict[str, Any]:
    if len(off["items"]) != len(candidate["items"]):
        raise QwenDirectDecodeError("generated triad item counts differ")
    items: list[dict[str, Any]] = []
    divergences = 0
    answer_changes = 0
    for off_row, candidate_row in zip(off["items"], candidate["items"], strict=True):
        if (
            off_row["item_id"] != candidate_row["item_id"]
            or off_row["prompt_token_ids"] != candidate_row["prompt_token_ids"]
        ):
            raise QwenDirectDecodeError(
                "generated triad item or prompt identity differs"
            )
        first = _first_divergence(
            off_row["generated_token_ids"], candidate_row["generated_token_ids"]
        )
        diverged = first is not None
        answer_changed = (
            off_row["parsed_numeric_answer"] != candidate_row["parsed_numeric_answer"]
        )
        divergences += diverged
        answer_changes += answer_changed
        items.append(
            {
                "candidate_parsed_numeric_answer": candidate_row[
                    "parsed_numeric_answer"
                ],
                "candidate_token_chain_sha256": candidate_row["token_chain_sha256"],
                "diverged": diverged,
                "first_divergence_index": first,
                "item_id": off_row["item_id"],
                "off_parsed_numeric_answer": off_row["parsed_numeric_answer"],
                "off_token_chain_sha256": off_row["token_chain_sha256"],
                "parsed_answer_changed": answer_changed,
            }
        )
    return {
        "arm": arm,
        "branch_effect_observed": divergences > 0,
        "diverged_items": divergences,
        "items": items,
        "parsed_answer_changes": answer_changes,
        "total": len(items),
        "verdict": "diverged" if divergences else "no_divergence",
    }


def _compare_generated_triad_documents(
    off: Mapping[str, Any],
    stable: Mapping[str, Any],
    native: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        off.get("arm") != "off"
        or stable.get("arm") != "stable-crsa"
        or native.get("arm") != "native-crsa"
    ):
        raise QwenDirectDecodeError(
            "triad comparison requires off, stable-crsa, and native-crsa arms"
        )
    identity = _paired_identity(off, stable)
    for name, value in _paired_identity(off, native).items():
        if identity[name] != value:  # pragma: no cover - pair checks already bind.
            raise QwenDirectDecodeError(f"generated triad {name} identity differs")
    document = {
        "arm_seals": {
            "native_crsa": native["sha256"],
            "off": off["sha256"],
            "stable_crsa": stable["sha256"],
        },
        "arms": {
            "native_crsa": {
                "intervention": native["intervention"],
                "traffic": native["traffic"],
            },
            "off": {"graft": off["graft"], "traffic": off["traffic"]},
            "stable_crsa": {
                "graft": stable["graft"],
                "traffic": stable["traffic"],
            },
        },
        "comparisons": {
            "off_vs_hidden": _triad_pair_report(off, stable, arm="stable-crsa"),
            "off_vs_native": _triad_pair_report(off, native, arm="native-crsa"),
        },
        "identity": identity,
        "schema": TRIAD_COMPARISON_SCHEMA,
        "status": "sealed",
    }
    return _result(document)


def compare_generated_triad(args: argparse.Namespace) -> dict[str, Any]:
    return _compare_generated_triad_documents(
        _load_branch_result(args.off),
        _load_branch_result(args.stable),
        _load_native_branch_result(args.native),
    )


def _validate_triad_pair_report(
    value: object,
    *,
    arm: str,
    item_ids: Sequence[str],
) -> None:
    required = {
        "arm",
        "branch_effect_observed",
        "diverged_items",
        "items",
        "parsed_answer_changes",
        "total",
        "verdict",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise QwenDirectDecodeError("generated triad pair report is invalid")
    rows = value.get("items")
    total = _nonnegative_count(value.get("total"), "generated triad pair total")
    divergences = _nonnegative_count(
        value.get("diverged_items"), "generated triad divergences"
    )
    answer_changes = _nonnegative_count(
        value.get("parsed_answer_changes"), "generated triad answer changes"
    )
    if (
        value.get("arm") != arm
        or not isinstance(rows, list)
        or total != len(rows)
        or total != len(item_ids)
        or divergences > total
        or answer_changes > total
        or value.get("branch_effect_observed") is not (divergences > 0)
        or value.get("verdict") != ("diverged" if divergences else "no_divergence")
    ):
        raise QwenDirectDecodeError("generated triad pair accounting is invalid")
    counted_divergences = 0
    counted_changes = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "candidate_parsed_numeric_answer",
            "candidate_token_chain_sha256",
            "diverged",
            "first_divergence_index",
            "item_id",
            "off_parsed_numeric_answer",
            "off_token_chain_sha256",
            "parsed_answer_changed",
        }:
            raise QwenDirectDecodeError("generated triad pair item is invalid")
        first = row.get("first_divergence_index")
        if first is not None:
            _nonnegative_count(first, "generated triad first divergence")
        if (
            row.get("item_id") != item_ids[index]
            or not isinstance(row.get("diverged"), bool)
            or not isinstance(row.get("parsed_answer_changed"), bool)
            or row["diverged"] is not (first is not None)
        ):
            raise QwenDirectDecodeError("generated triad pair item is inconsistent")
        _digest_string(row.get("candidate_token_chain_sha256"), "candidate chain")
        _digest_string(row.get("off_token_chain_sha256"), "off chain")
        counted_divergences += row["diverged"]
        counted_changes += row["parsed_answer_changed"]
    if counted_divergences != divergences or counted_changes != answer_changes:
        raise QwenDirectDecodeError("generated triad pair counts are inconsistent")


def _load_generated_triad_comparison(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    document = _strict_json(path)
    if (
        set(document)
        != {
            "arm_seals",
            "arms",
            "comparisons",
            "identity",
            "schema",
            "sha256",
            "status",
        }
        or document.get("schema") != TRIAD_COMPARISON_SCHEMA
        or document.get("status") != "sealed"
    ):
        raise QwenDirectDecodeError("generated triad comparison schema is invalid")
    _verify_document_seal(document, "generated triad comparison")
    seals = document.get("arm_seals")
    if not isinstance(seals, Mapping) or set(seals) != {
        "native_crsa",
        "off",
        "stable_crsa",
    }:
        raise QwenDirectDecodeError("generated triad arm seals are invalid")
    for name, value in seals.items():
        _digest_string(value, f"generated triad {name} arm")
    arms = document.get("arms")
    comparisons = document.get("comparisons")
    identity = document.get("identity")
    if (
        not isinstance(arms, Mapping)
        or set(arms) != {"native_crsa", "off", "stable_crsa"}
        or not isinstance(comparisons, Mapping)
        or set(comparisons) != {"off_vs_hidden", "off_vs_native"}
        or not isinstance(identity, Mapping)
        or set(identity)
        != {"bundle", "checkpoint", "execution", "generation", "input", "tokenizer"}
    ):
        raise QwenDirectDecodeError("generated triad identity is invalid")
    if (
        not isinstance(arms["off"], Mapping)
        or set(arms["off"]) != {"graft", "traffic"}
        or not isinstance(arms["stable_crsa"], Mapping)
        or set(arms["stable_crsa"]) != {"graft", "traffic"}
        or not isinstance(arms["native_crsa"], Mapping)
        or set(arms["native_crsa"]) != {"intervention", "traffic"}
    ):
        raise QwenDirectDecodeError("generated triad arm identity is invalid")
    _validate_graft_identity(arms["off"].get("graft"), "off")
    _validate_graft_identity(arms["stable_crsa"].get("graft"), "stable-crsa")
    _validate_native_intervention_identity(arms["native_crsa"].get("intervention"))
    input_identity = identity.get("input")
    if not isinstance(input_identity, Mapping):
        raise QwenDirectDecodeError("generated triad input identity is invalid")
    selection = input_identity.get("selection")
    if not isinstance(selection, Mapping):
        raise QwenDirectDecodeError("generated triad selection identity is invalid")
    item_ids = selection.get("item_ids")
    if not isinstance(item_ids, list) or any(
        not isinstance(item_id, str) or not item_id for item_id in item_ids
    ):
        raise QwenDirectDecodeError("generated triad item identity is invalid")
    _validate_triad_pair_report(
        comparisons["off_vs_hidden"], arm="stable-crsa", item_ids=item_ids
    )
    _validate_triad_pair_report(
        comparisons["off_vs_native"], arm="native-crsa", item_ids=item_ids
    )
    return document


def _build_adjudication_item(
    *,
    item_id: str,
    question: str,
    candidates: Mapping[str, object],
    solver: FertigSolver | None = None,
) -> dict[str, Any]:
    required_arms = ("off", "stable_crsa", "native_crsa")
    if set(candidates) != set(required_arms):
        raise QwenDirectDecodeError("adjudication candidate arms are invalid")
    exact_solver = solver or FertigSolver()
    try:
        results = {
            arm: exact_solver.verify_candidate(question, candidates[arm])
            for arm in required_arms
        }
    except Exception as exc:
        raise QwenDirectDecodeError(
            f"FERTIG candidate verification failed: {type(exc).__name__}: {exc}"
        ) from exc
    verifications = {arm: result.to_dict() for arm, result in results.items()}
    statuses = {result.status.value for result in results.values()}
    if statuses == {"abstained"}:
        decision = "abstained"
        exact_solution = None
        selected_answer = None
    else:
        if "abstained" in statuses or not statuses <= {"verified", "mismatch"}:
            raise QwenDirectDecodeError(
                "FERTIG verification support differs across identical questions"
            )
        expected = {result.expected for result in results.values()}
        exact_evidence = {
            _canonical(result.evidence["exact_solution"]) for result in results.values()
        }
        if len(expected) != 1 or None in expected or len(exact_evidence) != 1:
            raise QwenDirectDecodeError(
                "FERTIG exact evidence differs across candidate arms"
            )
        selected_answer = expected.pop()
        exact_solution = next(iter(results.values())).evidence["exact_solution"]
        decision = (
            "certificate_override" if "mismatch" in statuses else "certificate_verified"
        )
    return {
        "candidates": {arm: verifications[arm]["candidate"] for arm in required_arms},
        "decision": decision,
        "exact_solution": exact_solution,
        "item_id": item_id,
        "question": question,
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        "selected_answer": selected_answer,
        "verifications": verifications,
    }


def _adjudication_summary(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    overrides = sum(row.get("decision") == "certificate_override" for row in items)
    verified = sum(row.get("decision") == "certificate_verified" for row in items)
    abstained = sum(row.get("decision") == "abstained" for row in items)
    return {
        "abstained": abstained,
        "certificate_overrides": overrides,
        "certificate_verified": verified,
        "certified": overrides + verified,
        "total": len(items),
    }


def _validate_branch_adjudication_document(
    document: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "arm_seals",
        "branch_input_sha256",
        "items",
        "protocol",
        "question_source_sha256",
        "schema",
        "sha256",
        "status",
        "summary",
        "triad_sha256",
    }
    if (
        set(document) != required
        or document.get("schema") != BRANCH_ADJUDICATION_SCHEMA
        or document.get("status") != "sealed"
    ):
        raise QwenDirectDecodeError("branch adjudication schema is invalid")
    _verify_document_seal(document, "branch adjudication")
    _digest_string(document.get("branch_input_sha256"), "adjudication branch input")
    _digest_string(document.get("question_source_sha256"), "adjudication questions")
    _digest_string(document.get("triad_sha256"), "adjudication triad")
    seals = document.get("arm_seals")
    protocol = document.get("protocol")
    items = document.get("items")
    summary = document.get("summary")
    if not isinstance(seals, Mapping) or set(seals) != {
        "native_crsa",
        "off",
        "stable_crsa",
    }:
        raise QwenDirectDecodeError("adjudication arm seals are invalid")
    for arm, seal in seals.items():
        _digest_string(seal, f"adjudication {arm} arm")
    if protocol != {
        "all_arm_seals_admitted_before_questions": True,
        "decision_rule": "exact-fertig-certificate-overrides-candidates/v1",
        "generation_was_label_free": True,
        "gold_accessed_by_adjudicator": False,
        "question_prompt_binding": "official-tokenizer-reencode/v1",
        "triad_seal_admitted_before_questions": True,
        "unsupported_policy": "abstain",
    }:
        raise QwenDirectDecodeError("branch adjudication protocol is invalid")
    if not isinstance(items, list) or not items:
        raise QwenDirectDecodeError("branch adjudication contains no items")
    recomputed: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in items:
        if not isinstance(row, Mapping) or set(row) != {
            "candidates",
            "decision",
            "exact_solution",
            "item_id",
            "question",
            "question_sha256",
            "selected_answer",
            "verifications",
        }:
            raise QwenDirectDecodeError("branch adjudication item schema is invalid")
        item_id = row.get("item_id")
        question = row.get("question")
        candidates = row.get("candidates")
        if (
            not isinstance(item_id, str)
            or not item_id
            or item_id in seen
            or not isinstance(question, str)
            or not question.strip()
            or not isinstance(candidates, Mapping)
        ):
            raise QwenDirectDecodeError("branch adjudication item identity is invalid")
        expected = _build_adjudication_item(
            item_id=item_id,
            question=question,
            candidates=candidates,
        )
        if _canonical(row) != _canonical(expected):
            raise QwenDirectDecodeError("branch adjudication evidence mismatch")
        recomputed.append(expected)
        seen.add(item_id)
    expected_summary = _adjudication_summary(recomputed)
    if not isinstance(summary, Mapping) or dict(summary) != expected_summary:
        raise QwenDirectDecodeError("branch adjudication summary is inconsistent")
    return document


def _load_branch_adjudication_document(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    return _validate_branch_adjudication_document(_strict_json(path))


def adjudicate_generated_triad(args: argparse.Namespace) -> dict[str, Any]:
    """Apply exact FERTIG certificates after arms seal and before any gold access."""

    off = _load_branch_result(args.off)
    stable = _load_branch_result(args.stable)
    native = _load_native_branch_result(args.native)
    expected_triad = _compare_generated_triad_documents(off, stable, native)
    triad = _load_generated_triad_comparison(args.triad)
    if _canonical(triad) != _canonical(expected_triad):
        raise QwenDirectDecodeError("triad comparison does not bind admitted arms")
    if off["input"]["schema"] != BRANCH_INPUT_SCHEMA_V3:
        raise QwenDirectDecodeError(
            "exact adjudication requires sealed answer-prefix branch arms"
        )

    # Questions are admitted only after every arm and the recomputed triad seal.
    questions = _load_branch_question_document(args.questions)
    if (
        questions["branch_input_sha256"] != off["input"]["sha256"]
        or questions["selection"] != off["input"]["selection"]
        or questions["source"]["contract_sha256"]
        != off["input"]["source"]["contract_sha256"]
        or questions["source"]["raw_file_sha256"]
        != off["input"]["source"]["raw_file_sha256"]
    ):
        raise QwenDirectDecodeError(
            "branch question document does not bind the admitted answer input"
        )
    _verify_question_prompt_bindings(
        questions,
        off["input"],
        args.tokenizer_json,
    )
    question_rows = questions["items"]
    if not (
        len(question_rows)
        == len(off["items"])
        == len(stable["items"])
        == len(native["items"])
    ):
        raise QwenDirectDecodeError("adjudication arm and question counts differ")
    items: list[dict[str, Any]] = []
    solver = FertigSolver()
    for question_row, off_row, stable_row, native_row in zip(
        question_rows,
        off["items"],
        stable["items"],
        native["items"],
        strict=True,
    ):
        item_id = question_row["item_id"]
        if item_id not in {
            off_row["item_id"],
            stable_row["item_id"],
            native_row["item_id"],
        } or not (
            item_id
            == off_row["item_id"]
            == stable_row["item_id"]
            == native_row["item_id"]
        ):
            raise QwenDirectDecodeError("adjudication item order differs across arms")
        items.append(
            _build_adjudication_item(
                item_id=item_id,
                question=question_row["question"],
                candidates={
                    "native_crsa": native_row["parsed_numeric_answer"],
                    "off": off_row["parsed_numeric_answer"],
                    "stable_crsa": stable_row["parsed_numeric_answer"],
                },
                solver=solver,
            )
        )
    identity = {
        "arm_seals": dict(triad["arm_seals"]),
        "branch_input_sha256": off["input"]["sha256"],
        "items": items,
        "protocol": {
            "all_arm_seals_admitted_before_questions": True,
            "decision_rule": "exact-fertig-certificate-overrides-candidates/v1",
            "generation_was_label_free": True,
            "gold_accessed_by_adjudicator": False,
            "question_prompt_binding": "official-tokenizer-reencode/v1",
            "triad_seal_admitted_before_questions": True,
            "unsupported_policy": "abstain",
        },
        "question_source_sha256": questions["sha256"],
        "schema": BRANCH_ADJUDICATION_SCHEMA,
        "status": "sealed",
        "summary": _adjudication_summary(items),
        "triad_sha256": triad["sha256"],
    }
    return _validate_branch_adjudication_document(_result(identity))


def evaluate_adjudicated_triad(args: argparse.Namespace) -> dict[str, Any]:
    """Measure the sealed adjudicated system only after every pre-gold seal binds."""

    # Re-admit the complete generation side before opening any label-bearing bytes.
    off = _load_branch_result(args.off)
    stable = _load_branch_result(args.stable)
    native = _load_native_branch_result(args.native)
    expected_triad = _compare_generated_triad_documents(off, stable, native)
    triad = _load_generated_triad_comparison(args.triad)
    if _canonical(triad) != _canonical(expected_triad):
        raise QwenDirectDecodeError("triad comparison does not bind admitted arms")
    questions = _load_branch_question_document(args.questions)
    expected_adjudication = adjudicate_generated_triad(args)
    if (
        expected_adjudication["arm_seals"] != triad["arm_seals"]
        or expected_adjudication["triad_sha256"] != triad["sha256"]
        or expected_adjudication["question_source_sha256"] != questions["sha256"]
        or expected_adjudication["branch_input_sha256"] != off["input"]["sha256"]
    ):
        raise QwenDirectDecodeError(
            "adjudication inputs changed while their seals were being admitted"
        )
    adjudication = _load_branch_adjudication_document(args.adjudication)
    if _canonical(adjudication) != _canonical(expected_adjudication):
        raise QwenDirectDecodeError(
            "adjudication does not bind admitted arms, triad, and questions"
        )

    expected_source_sha256 = _digest_string(
        args.gold_source_sha256, "label source externally pinned raw file"
    )
    if expected_source_sha256 != off["input"]["source"]["raw_file_sha256"]:
        raise QwenDirectDecodeError(
            "label source external SHA-256 differs from sealed branch input"
        )
    source, source_raw_sha256 = _externally_sealed_json(
        args.gold_source,
        expected_source_sha256,
        "label source",
    )
    _source_rows(source)
    _rows, targets = _gold_rows_for_branch(source, off)

    transitions = {
        "correct_to_abstained": 0,
        "correct_to_correct": 0,
        "correct_to_wrong": 0,
        "wrong_to_abstained": 0,
        "wrong_to_correct": 0,
        "wrong_to_wrong": 0,
    }
    items: list[dict[str, Any]] = []
    off_correct_count = 0
    system_correct_count = 0
    covered = 0
    for off_row, adjudicated in zip(off["items"], adjudication["items"], strict=True):
        item_id = off_row["item_id"]
        if adjudicated["item_id"] != item_id:
            raise QwenDirectDecodeError("adjudication evaluation item order differs")
        target = targets[item_id]
        selected = adjudicated["selected_answer"]
        off_correct = off_row["parsed_numeric_answer"] == target
        system_correct = selected is not None and selected == target
        off_correct_count += off_correct
        system_correct_count += system_correct
        covered += selected is not None
        if selected is None:
            transition = "correct_to_abstained" if off_correct else "wrong_to_abstained"
        else:
            transition = _transition_against_off(
                off_correct=off_correct,
                candidate_correct=system_correct,
            )
        transitions[transition] += 1
        items.append(
            {
                "decision": adjudicated["decision"],
                "gold": target,
                "item_id": item_id,
                "off_correct": off_correct,
                "off_parsed_numeric_answer": off_row["parsed_numeric_answer"],
                "selected_answer": selected,
                "system_correct": system_correct,
                "transition": transition,
                "unsafe": transition in {"correct_to_abstained", "correct_to_wrong"},
            }
        )
    total = len(items)
    net = system_correct_count - off_correct_count
    unsafe = transitions["correct_to_wrong"] + transitions["correct_to_abstained"]
    if unsafe:
        verdict = "unsafe"
    elif net > 0:
        verdict = "improved"
    elif net < 0:
        verdict = "regressed"
    else:
        verdict = "neutral"
    return _result(
        {
            "adjudication_sha256": adjudication["sha256"],
            "arm_seals": dict(triad["arm_seals"]),
            "items": items,
            "protocol": {
                "adjudicator_did_not_access_label_source": True,
                "adjudication_recomputed_before_label_source": True,
                "all_seals_admitted_before_evaluator_opened_label_source": True,
                "generation_was_label_free": True,
                "transition_rule": "canonical-numeric-exact-match-with-abstention/v1",
            },
            "question_source_sha256": questions["sha256"],
            "schema": BRANCH_ADJUDICATION_EVALUATION_SCHEMA,
            "source": {
                "contract_sha256": off["input"]["source"]["contract_sha256"],
                "raw_file_sha256": source_raw_sha256,
                "schema": source["schema"],
                "seal_kind": EXTERNAL_RAW_SEAL_KIND,
            },
            "status": "sealed",
            "summary": {
                **transitions,
                "coverage": covered / total,
                "covered": covered,
                "net_accuracy_delta": net / total,
                "net_correct_delta": net,
                "off_accuracy": off_correct_count / total,
                "off_correct": off_correct_count,
                "quality_success": (
                    transitions["wrong_to_correct"] > 0 and net > 0 and unsafe == 0
                ),
                "system_accuracy": system_correct_count / total,
                "system_correct": system_correct_count,
                "total": total,
                "unsafe_correct_losses": unsafe,
                "verdict": verdict,
            },
            "triad_sha256": triad["sha256"],
        }
    )


def _gold_rows_for_branch(
    source: Mapping[str, Any],
    off: Mapping[str, Any],
) -> tuple[list[Mapping[str, Any]], dict[str, str]]:
    projection = _source_contract_projection(source)
    if _sha256(projection) != off["input"]["source"]["contract_sha256"]:
        raise QwenDirectDecodeError("label source generation contract differs")
    selection = off["input"]["selection"]
    expected_items = projection["items"][
        selection["offset"] : selection["offset"] + selection["limit"]
    ]
    if off["input"]["schema"] == BRANCH_INPUT_SCHEMA:
        if (
            selection["source_items"] != len(projection["items"])
            or off["input"]["items"] != expected_items
        ):
            raise QwenDirectDecodeError(
                "sealed branch selection differs from label source"
            )
    else:
        _validate_answer_source_protocol(projection)
        prefix = tuple(off["input"]["protocol"]["generation_prefix"]["token_ids"])
        sealed_items = off["input"]["items"]
        if selection["source_items"] != len(projection["items"]) or len(
            sealed_items
        ) != len(expected_items):
            raise QwenDirectDecodeError(
                "sealed answer branch selection differs from label source"
            )
        for sealed, expected in zip(sealed_items, expected_items, strict=True):
            source_prompt = list(expected["prompt_token_ids"])
            if (
                sealed["item_id"] != expected["item_id"]
                or sealed["source_prompt_token_ids"] != source_prompt
                or sealed["effective_prompt_token_ids"] != [*source_prompt, *prefix]
            ):
                raise QwenDirectDecodeError(
                    "sealed answer prompts differ from label source"
                )
    selection_ids = selection["item_ids"]
    by_id = {row["item_id"]: row for row in _source_rows(source)}
    rows: list[Mapping[str, Any]] = []
    targets: dict[str, str] = {}
    for item_id in selection_ids:
        raw = by_id.get(item_id)
        if raw is None:
            raise QwenDirectDecodeError("label source is missing a selected item")
        target = extract_gsm8k_answer(raw.get("gold"))
        if target is None:
            raise QwenDirectDecodeError(
                "label source contains an invalid numeric target"
            )
        rows.append(raw)
        targets[item_id] = target
    return rows, targets


def evaluate_generated_arms(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate only after both arm and optional comparison seals are admitted."""

    # Ordering is part of the protocol: do not open the label source above here.
    off = _load_branch_result(args.off)
    candidate = _load_branch_result(args.candidate)
    expected_comparison = _compare_branch_documents(off, candidate)
    comparison_path = getattr(args, "comparison", None)
    if comparison_path is not None:
        comparison = _load_branch_comparison(comparison_path)
        if comparison != expected_comparison:
            raise QwenDirectDecodeError("comparison does not bind the admitted arms")
    else:
        comparison = expected_comparison

    expected_source_sha256 = _digest_string(
        args.gold_source_sha256, "label source externally pinned raw file"
    )
    if expected_source_sha256 != off["input"]["source"]["raw_file_sha256"]:
        raise QwenDirectDecodeError(
            "label source external SHA-256 differs from sealed branch input"
        )
    source, source_raw_sha256 = _externally_sealed_json(
        args.gold_source,
        expected_source_sha256,
        "label source",
    )
    _source_rows(source)
    _rows, targets = _gold_rows_for_branch(source, off)
    items: list[dict[str, Any]] = []
    counts = {
        "correct_to_correct": 0,
        "correct_to_wrong": 0,
        "wrong_to_correct": 0,
        "wrong_to_wrong": 0,
    }
    for off_row, candidate_row in zip(off["items"], candidate["items"], strict=True):
        item_id = off_row["item_id"]
        target = targets[item_id]
        off_correct = off_row["parsed_numeric_answer"] == target
        candidate_correct = candidate_row["parsed_numeric_answer"] == target
        if off_correct and candidate_correct:
            transition = "correct_to_correct"
        elif off_correct:
            transition = "correct_to_wrong"
        elif candidate_correct:
            transition = "wrong_to_correct"
        else:
            transition = "wrong_to_wrong"
        counts[transition] += 1
        items.append(
            {
                "candidate_correct": candidate_correct,
                "candidate_parsed_numeric_answer": candidate_row[
                    "parsed_numeric_answer"
                ],
                "gold": target,
                "item_id": item_id,
                "off_correct": off_correct,
                "off_parsed_numeric_answer": off_row["parsed_numeric_answer"],
                "sequence_diverged": (
                    off_row["generated_token_ids"]
                    != candidate_row["generated_token_ids"]
                ),
                "transition": transition,
                "unsafe": transition == "correct_to_wrong",
            }
        )
    total = len(items)
    off_correct_count = counts["correct_to_correct"] + counts["correct_to_wrong"]
    candidate_correct_count = counts["correct_to_correct"] + counts["wrong_to_correct"]
    net = candidate_correct_count - off_correct_count
    if counts["correct_to_wrong"]:
        verdict = "unsafe"
    elif net > 0:
        verdict = "improved"
    elif net < 0:
        verdict = "regressed"
    else:
        verdict = "neutral"
    identity = {
        "comparison_sha256": comparison["sha256"],
        "items": items,
        "protocol": {
            "arm_seals_admitted_before_label_source": True,
            "comparison_admitted_before_label_source": True,
            "generation_was_label_free": True,
            "transition_rule": "canonical-numeric-exact-match/v1",
        },
        "schema": BRANCH_EVALUATION_SCHEMA,
        "source": {
            "contract_sha256": off["input"]["source"]["contract_sha256"],
            "raw_file_sha256": source_raw_sha256,
            "schema": source["schema"],
            "seal_kind": EXTERNAL_RAW_SEAL_KIND,
        },
        "status": "sealed",
        "summary": {
            **counts,
            "candidate_accuracy": candidate_correct_count / total,
            "candidate_correct": candidate_correct_count,
            "net_accuracy_delta": net / total,
            "net_correct_delta": net,
            "off_accuracy": off_correct_count / total,
            "off_correct": off_correct_count,
            "quality_success": (
                counts["wrong_to_correct"] > 0
                and net > 0
                and counts["correct_to_wrong"] == 0
            ),
            "total": total,
            "unsafe_correct_to_wrong": counts["correct_to_wrong"],
            "verdict": verdict,
        },
    }
    return _result(identity)


def _transition_against_off(*, off_correct: bool, candidate_correct: bool) -> str:
    if off_correct and candidate_correct:
        return "correct_to_correct"
    if off_correct:
        return "correct_to_wrong"
    if candidate_correct:
        return "wrong_to_correct"
    return "wrong_to_wrong"


def _triad_transition_summary(
    counts: Mapping[str, int],
    *,
    total: int,
    off_correct: int,
) -> dict[str, Any]:
    candidate_correct = counts["correct_to_correct"] + counts["wrong_to_correct"]
    net = candidate_correct - off_correct
    if counts["correct_to_wrong"]:
        verdict = "unsafe"
    elif net > 0:
        verdict = "improved"
    elif net < 0:
        verdict = "regressed"
    else:
        verdict = "neutral"
    return {
        **dict(counts),
        "accuracy": candidate_correct / total,
        "correct": candidate_correct,
        "net_accuracy_delta": net / total,
        "net_correct_delta": net,
        "quality_success": (
            counts["wrong_to_correct"] > 0
            and net > 0
            and counts["correct_to_wrong"] == 0
        ),
        "unsafe_correct_to_wrong": counts["correct_to_wrong"],
        "verdict": verdict,
    }


def evaluate_generated_triad(args: argparse.Namespace) -> dict[str, Any]:
    """Admit three arm seals and the exact triad seal before opening gold."""

    # The order is a security property: no label-bearing bytes are opened until
    # all result documents and the recomputed triad comparison are admitted.
    off = _load_branch_result(args.off)
    stable = _load_branch_result(args.stable)
    native = _load_native_branch_result(args.native)
    expected_triad = _compare_generated_triad_documents(off, stable, native)
    triad_path = getattr(args, "triad", None)
    if triad_path is None:
        raise QwenDirectDecodeError("sealed generated triad comparison is required")
    triad = _load_generated_triad_comparison(triad_path)
    if _canonical(triad) != _canonical(expected_triad):
        raise QwenDirectDecodeError("triad comparison does not bind admitted arms")

    expected_source_sha256 = _digest_string(
        args.gold_source_sha256, "label source externally pinned raw file"
    )
    if expected_source_sha256 != off["input"]["source"]["raw_file_sha256"]:
        raise QwenDirectDecodeError(
            "label source external SHA-256 differs from sealed branch input"
        )
    source, source_raw_sha256 = _externally_sealed_json(
        args.gold_source,
        expected_source_sha256,
        "label source",
    )
    _source_rows(source)
    _rows, targets = _gold_rows_for_branch(source, off)
    transitions = (
        "correct_to_correct",
        "correct_to_wrong",
        "wrong_to_correct",
        "wrong_to_wrong",
    )
    hidden_counts = {name: 0 for name in transitions}
    native_counts = {name: 0 for name in transitions}
    items: list[dict[str, Any]] = []
    off_correct_count = 0
    for off_row, hidden_row, native_row in zip(
        off["items"], stable["items"], native["items"], strict=True
    ):
        item_id = off_row["item_id"]
        target = targets[item_id]
        off_correct = off_row["parsed_numeric_answer"] == target
        hidden_correct = hidden_row["parsed_numeric_answer"] == target
        native_correct = native_row["parsed_numeric_answer"] == target
        off_correct_count += off_correct
        hidden_transition = _transition_against_off(
            off_correct=off_correct, candidate_correct=hidden_correct
        )
        native_transition = _transition_against_off(
            off_correct=off_correct, candidate_correct=native_correct
        )
        hidden_counts[hidden_transition] += 1
        native_counts[native_transition] += 1
        items.append(
            {
                "gold": target,
                "hidden_correct": hidden_correct,
                "hidden_parsed_numeric_answer": hidden_row["parsed_numeric_answer"],
                "hidden_sequence_diverged": (
                    off_row["generated_token_ids"] != hidden_row["generated_token_ids"]
                ),
                "hidden_transition": hidden_transition,
                "hidden_unsafe": hidden_transition == "correct_to_wrong",
                "item_id": item_id,
                "native_correct": native_correct,
                "native_parsed_numeric_answer": native_row["parsed_numeric_answer"],
                "native_sequence_diverged": (
                    off_row["generated_token_ids"] != native_row["generated_token_ids"]
                ),
                "native_transition": native_transition,
                "native_unsafe": native_transition == "correct_to_wrong",
                "off_correct": off_correct,
                "off_parsed_numeric_answer": off_row["parsed_numeric_answer"],
            }
        )
    total = len(items)
    hidden_summary = _triad_transition_summary(
        hidden_counts, total=total, off_correct=off_correct_count
    )
    native_summary = _triad_transition_summary(
        native_counts, total=total, off_correct=off_correct_count
    )
    return _result(
        {
            "arm_seals": dict(triad["arm_seals"]),
            "items": items,
            "protocol": {
                "all_arm_seals_admitted_before_label_source": True,
                "generation_was_label_free": True,
                "transition_rule": "canonical-numeric-exact-match/v1",
                "triad_seal_admitted_before_label_source": True,
            },
            "schema": TRIAD_EVALUATION_SCHEMA,
            "source": {
                "contract_sha256": off["input"]["source"]["contract_sha256"],
                "raw_file_sha256": source_raw_sha256,
                "schema": source["schema"],
                "seal_kind": EXTERNAL_RAW_SEAL_KIND,
            },
            "status": "sealed",
            "summary": {
                "hidden": hidden_summary,
                "native": native_summary,
                "off_accuracy": off_correct_count / total,
                "off_correct": off_correct_count,
                "total": total,
            },
            "triad_sha256": triad["sha256"],
        }
    )


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _unit_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return value


def _runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--access-trace", required=True)
    parser.add_argument("--delta-probe")
    parser.add_argument("--source", default=OFFICIAL_REPO_ID)
    parser.add_argument("--causal-bundle")
    parser.add_argument("--logical-repo-id", default=OFFICIAL_REPO_ID)
    parser.add_argument("--revision", default=OFFICIAL_REVISION)
    parser.add_argument("--pinned-inventory")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--max-cache-gb", type=_positive_float, default=1.0)
    parser.add_argument("--source-budget-mb", type=_positive_int, default=65536)
    parser.add_argument("--max-resident-mb", type=_positive_int, default=384)
    parser.add_argument("--max-seq-len", type=_positive_int, default=256)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )


def _branch_runtime_arguments(
    parser: argparse.ArgumentParser, *, include_mode: bool = True
) -> None:
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--access-trace", required=True)
    parser.add_argument("--source", default=OFFICIAL_REPO_ID)
    parser.add_argument("--causal-bundle")
    parser.add_argument("--logical-repo-id", default=OFFICIAL_REPO_ID)
    parser.add_argument("--revision", default=OFFICIAL_REVISION)
    parser.add_argument("--pinned-inventory")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--max-cache-gb", type=_positive_float, default=1.0)
    parser.add_argument("--source-budget-mb", type=_positive_int, default=65536)
    parser.add_argument("--max-resident-mb", type=_positive_int, default=384)
    parser.add_argument("--max-seq-len", type=_positive_int, default=256)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=32)
    parser.add_argument("--head-block-rows", type=_positive_int, default=8192)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    if include_mode:
        parser.add_argument(
            "--mode", choices=("off", "stable-crsa", "native-crsa"), required=True
        )
        parser.add_argument("--graft-layer", type=_nonnegative_int, default=27)
        parser.add_argument("--graft-alpha", type=_unit_float, default=0.01)
        parser.add_argument("--graft-max-history", type=_positive_int, default=256)
        parser.add_argument("--graft-rms-eps", type=_positive_float, default=1e-6)
    parser.add_argument("--native-alpha", type=_unit_float, default=NATIVE_CRSA_ALPHA)
    parser.add_argument(
        "--native-balance-alpha",
        type=_nonnegative_float,
        default=NATIVE_CRSA_BALANCE_ALPHA,
    )
    parser.add_argument(
        "--native-diagonal-debit",
        type=_nonnegative_float,
        default=NATIVE_CRSA_DIAGONAL_DEBIT,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select-input")
    select.add_argument("--inputs", required=True)
    select.add_argument("--item-id", required=True)
    select.add_argument("--output", required=True)
    select.set_defaults(handler=select_input)
    prepare = subparsers.add_parser("prepare-prefix")
    _runtime_arguments(prepare)
    prepare.set_defaults(handler=prepare_prefix)
    decode = subparsers.add_parser("decode")
    _runtime_arguments(decode)
    decode.add_argument("--prefix-result", required=True)
    decode.set_defaults(handler=decode_arm)
    comparison = subparsers.add_parser("compare")
    comparison.add_argument("--remote", required=True)
    comparison.add_argument("--local", required=True)
    comparison.add_argument("--output", required=True)
    comparison.set_defaults(handler=compare)
    branch_select = subparsers.add_parser(
        "select-branch-cohort", aliases=("select-branch-input",)
    )
    branch_select.add_argument("--inputs", "--source-input", required=True)
    branch_select.add_argument("--inputs-sha256", required=True)
    branch_select.add_argument("--tokenizer-json", required=True)
    branch_select.add_argument("--offset", type=_nonnegative_int, default=0)
    branch_select.add_argument("--limit", type=_positive_int, default=8)
    branch_select.add_argument("--output", required=True)
    branch_select.set_defaults(handler=select_branch_cohort)
    answer_branch_select = subparsers.add_parser(
        "select-answer-branch-cohort", aliases=("select-answer-branch-input",)
    )
    answer_branch_select.add_argument("--inputs", "--source-input", required=True)
    answer_branch_select.add_argument("--inputs-sha256", required=True)
    answer_branch_select.add_argument("--tokenizer-json", required=True)
    answer_branch_select.add_argument("--offset", type=_nonnegative_int, default=0)
    answer_branch_select.add_argument("--limit", type=_positive_int, default=8)
    answer_branch_select.add_argument("--output", required=True)
    answer_branch_select.set_defaults(handler=select_answer_branch_cohort)
    question_select = subparsers.add_parser(
        "select-answer-branch-questions",
        aliases=("select-branch-questions",),
    )
    question_select.add_argument("--input", required=True)
    question_select.add_argument("--inputs", "--source-input", required=True)
    question_select.add_argument("--inputs-sha256", required=True)
    question_select.add_argument("--output", required=True)
    question_select.set_defaults(handler=select_answer_branch_questions)
    branch_generate = subparsers.add_parser("generate-arm")
    _branch_runtime_arguments(branch_generate)
    branch_generate.set_defaults(handler=generate_arm)
    fork_generate = subparsers.add_parser("generate-native-fork")
    _branch_runtime_arguments(fork_generate, include_mode=False)
    fork_generate.set_defaults(handler=generate_native_fork)
    fork_compare = subparsers.add_parser("compare-native-fork-references")
    fork_compare.add_argument("--pair", required=True)
    fork_compare.add_argument("--off", required=True)
    fork_compare.add_argument("--native", required=True)
    fork_compare.add_argument("--output", required=True)
    fork_compare.set_defaults(handler=compare_native_fork_references)
    branch_compare = subparsers.add_parser(
        "compare-generated-arms", aliases=("compare-branches",)
    )
    branch_compare.add_argument("--off", required=True)
    branch_compare.add_argument("--candidate", required=True)
    branch_compare.add_argument("--output", required=True)
    branch_compare.set_defaults(handler=compare_generated_arms)
    branch_evaluate = subparsers.add_parser(
        "evaluate-generated-arms", aliases=("evaluate-branches",)
    )
    branch_evaluate.add_argument("--off", required=True)
    branch_evaluate.add_argument("--candidate", required=True)
    branch_evaluate.add_argument("--comparison")
    branch_evaluate.add_argument("--gold-source", "--source-input", required=True)
    branch_evaluate.add_argument("--gold-source-sha256", required=True)
    branch_evaluate.add_argument("--output", required=True)
    branch_evaluate.set_defaults(handler=evaluate_generated_arms)
    triad_compare = subparsers.add_parser("compare-generated-triad")
    triad_compare.add_argument("--off", required=True)
    triad_compare.add_argument(
        "--stable",
        "--hidden",
        "--stable-crsa",
        dest="stable",
        required=True,
    )
    triad_compare.add_argument(
        "--native", "--native-crsa", dest="native", required=True
    )
    triad_compare.add_argument("--output", required=True)
    triad_compare.set_defaults(handler=compare_generated_triad)
    triad_adjudicate = subparsers.add_parser("adjudicate-generated-triad")
    triad_adjudicate.add_argument("--off", required=True)
    triad_adjudicate.add_argument(
        "--stable",
        "--hidden",
        "--stable-crsa",
        dest="stable",
        required=True,
    )
    triad_adjudicate.add_argument(
        "--native", "--native-crsa", dest="native", required=True
    )
    triad_adjudicate.add_argument(
        "--triad", "--comparison", dest="triad", required=True
    )
    triad_adjudicate.add_argument("--questions", required=True)
    triad_adjudicate.add_argument("--tokenizer-json", required=True)
    triad_adjudicate.add_argument("--output", required=True)
    triad_adjudicate.set_defaults(handler=adjudicate_generated_triad)
    adjudication_evaluate = subparsers.add_parser("evaluate-adjudicated-triad")
    adjudication_evaluate.add_argument("--off", required=True)
    adjudication_evaluate.add_argument(
        "--stable",
        "--hidden",
        "--stable-crsa",
        dest="stable",
        required=True,
    )
    adjudication_evaluate.add_argument(
        "--native", "--native-crsa", dest="native", required=True
    )
    adjudication_evaluate.add_argument(
        "--triad", "--comparison", dest="triad", required=True
    )
    adjudication_evaluate.add_argument("--questions", required=True)
    adjudication_evaluate.add_argument("--tokenizer-json", required=True)
    adjudication_evaluate.add_argument("--adjudication", required=True)
    adjudication_evaluate.add_argument("--gold-source", "--source-input", required=True)
    adjudication_evaluate.add_argument("--gold-source-sha256", required=True)
    adjudication_evaluate.add_argument("--output", required=True)
    adjudication_evaluate.set_defaults(handler=evaluate_adjudicated_triad)
    triad_evaluate = subparsers.add_parser("evaluate-generated-triad")
    triad_evaluate.add_argument("--off", required=True)
    triad_evaluate.add_argument(
        "--stable",
        "--hidden",
        "--stable-crsa",
        dest="stable",
        required=True,
    )
    triad_evaluate.add_argument(
        "--native", "--native-crsa", dest="native", required=True
    )
    triad_evaluate.add_argument("--triad", "--comparison", dest="triad", required=True)
    triad_evaluate.add_argument("--gold-source", "--source-input", required=True)
    triad_evaluate.add_argument("--gold-source-sha256", required=True)
    triad_evaluate.add_argument("--output", required=True)
    triad_evaluate.set_defaults(handler=evaluate_generated_triad)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = args.handler(args)
        _write_json(args.output, document)
    except QwenDirectDecodeError as exc:
        raise SystemExit(f"Qwen direct decode failed: {exc}") from exc
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "schema": document["schema"],
                "sha256": document["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
