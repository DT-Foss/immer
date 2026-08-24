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
BRANCH_RESULT_SCHEMA = "immer.qwen3.8-generation-branch-arm/v2"
BRANCH_COMPARISON_SCHEMA = "immer.qwen3.8-generation-branch-comparison/v2"
BRANCH_EVALUATION_SCHEMA = "immer.qwen3.8-generation-branch-evaluation/v2"
NATIVE_BRANCH_RESULT_SCHEMA = "immer.qwen3.8-generation-branch-arm/v3"
TRIAD_COMPARISON_SCHEMA = "immer.qwen3.8-generation-branch-comparison/v3"
TRIAD_EVALUATION_SCHEMA = "immer.qwen3.8-generation-branch-evaluation/v3"
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


def _load_branch_input(path: str | os.PathLike[str]) -> dict[str, Any]:
    return _validate_branch_input_document(_strict_json(path))


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
    prompts = [
        _token_rows(row["prompt_token_ids"], f"prompt {row['item_id']}")
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
                "parsed_numeric_answer": extract_gsm8k_answer(text),
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


def _verify_access_trace_receipt(receipt: object) -> None:
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
    try:
        trace = AccessTrace.from_bytes(Path(str(receipt["path"])).read_bytes())
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


def _load_branch_result(path: str | os.PathLike[str]) -> dict[str, Any]:
    return _validate_branch_result_document(_strict_json(path))


def _validate_branch_result_document(document: dict[str, Any]) -> dict[str, Any]:
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
    _validate_branch_input_document(input_identity)
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
        if (
            row["item_id"] != input_identity["items"][index]["item_id"]
            or list(prompt) != input_identity["items"][index]["prompt_token_ids"]
        ):
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
        if not isinstance(text, str) or row.get(
            "parsed_numeric_answer"
        ) != extract_gsm8k_answer(text):
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
    _verify_access_trace_receipt(traffic.get("access_trace"))
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
    _validate_branch_result_document(_native_v2_common_projection(document))
    layers = document["execution"]["checkpoint_layers"]
    if layers <= NATIVE_HEAD_CRSA_LAYER:
        raise QwenDirectDecodeError("native CRSA layer exceeds checkpoint depth")
    for row in document["items"]:
        _validate_native_evidence_chain(row)
    return document


def _load_native_branch_result(path: str | os.PathLike[str]) -> dict[str, Any]:
    return _validate_native_branch_result_document(_strict_json(path))


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
    if (
        selection["source_items"] != len(projection["items"])
        or off["input"]["items"] != expected_items
    ):
        raise QwenDirectDecodeError("sealed branch selection differs from label source")
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


def _branch_runtime_arguments(parser: argparse.ArgumentParser) -> None:
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
    branch_generate = subparsers.add_parser("generate-arm")
    _branch_runtime_arguments(branch_generate)
    branch_generate.set_defaults(handler=generate_arm)
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
