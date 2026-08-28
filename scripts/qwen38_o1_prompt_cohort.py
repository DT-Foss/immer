#!/usr/bin/env python3
"""Build a deterministic label-free unseen-prompt registry for Qwen O1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Mapping, Sequence, cast

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.qwen3_8 import Qwen38Tokenizer, prompt_token_sha256


COHORT_SCHEMA = "immer.qwen3.8-o1-unseen-prompt-cohort/v1"
SYSTEM_PROMPT = (
    "Solve the math problem internally. Return only #### followed by the "
    "numeric answer. Do not show work."
)
_MAX_INPUT_BYTES = 64 * 1024 * 1024


class CohortError(RuntimeError):
    pass


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a SHA-256") from exc
    if value != value.lower():
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_read(path: Path, maximum: int = _MAX_INPUT_BYTES) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise CohortError(f"invalid bounded input: {path}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise CohortError(f"input exceeds its bound: {path}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise CohortError(f"input changed while read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _json(path: Path) -> tuple[object, bytes]:
    raw = _stable_read(path)

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise ValueError(key)
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CohortError(f"input is not strict JSON: {path}") from exc
    return value, raw


def _persist_exact(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or _stable_read(path, max(1, len(data))) != data:
            raise CohortError(f"sealed output changed: {path}")
        return
    temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError:
        if _stable_read(path, max(1, len(data))) != data:
            raise CohortError(f"sealed output collided: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _existing_prompt_hashes(paths: Sequence[Path]) -> tuple[str, ...]:
    hashes = set()
    for path in paths:
        value, _raw = _json(path)
        if not isinstance(value, Mapping):
            raise CohortError(f"existing prompt manifest is invalid: {path}")
        body = value.get("body", value)
        if not isinstance(body, Mapping):
            raise CohortError(f"existing prompt manifest body is invalid: {path}")
        prompts = body.get("prompts")
        if not isinstance(prompts, list):
            raise CohortError(f"existing prompt registry is absent: {path}")
        for row in prompts:
            if not isinstance(row, Mapping):
                raise CohortError(f"existing prompt row is invalid: {path}")
            digest = row.get("sha256")
            try:
                hashes.add(_require_sha256(digest, field="existing prompt SHA"))
            except ValueError as exc:
                raise CohortError(f"existing prompt SHA is invalid: {path}") from exc
    return tuple(sorted(hashes))


def build_cohort(
    *,
    benchmark_path: Path,
    tokenizer_path: Path,
    existing_manifest_paths: Sequence[Path],
    existing_prompt_sha256s: Sequence[str],
    count: int,
    seed_sha256: str,
    max_prompt_tokens: int,
    require_official_tokenizer: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be positive")
    if (
        isinstance(max_prompt_tokens, bool)
        or not isinstance(max_prompt_tokens, int)
        or max_prompt_tokens < 1
    ):
        raise ValueError("max_prompt_tokens must be positive")
    seed_sha256 = _require_sha256(seed_sha256, field="seed_sha256")
    benchmark, benchmark_raw = _json(benchmark_path)
    if not isinstance(benchmark, Mapping) or not isinstance(
        benchmark.get("items"), list
    ):
        raise CohortError("benchmark must contain an item list")
    tokenizer_raw = _stable_read(tokenizer_path)
    tokenizer = Qwen38Tokenizer(
        tokenizer_path, require_official=require_official_tokenizer
    )
    existing = set(_existing_prompt_hashes(existing_manifest_paths))
    for digest in existing_prompt_sha256s:
        existing.add(_require_sha256(digest, field="existing_prompt_sha256s"))
    if not existing:
        raise ValueError("at least one existing prompt identity is required")
    items = cast(list[object], benchmark["items"])
    question_inventory = []
    inventory_ids = set()
    for row in items:
        if not isinstance(row, Mapping):
            raise CohortError("benchmark item is invalid")
        item_id = row.get("item_id")
        question = row.get("question")
        if (
            not isinstance(item_id, str)
            or not item_id
            or not isinstance(question, str)
            or not question.strip()
            or item_id in inventory_ids
        ):
            raise CohortError("benchmark item identity/question is invalid")
        inventory_ids.add(item_id)
        question_inventory.append(
            {
                "item_id": item_id,
                "question_sha256": _sha256_bytes(question.strip().encode("utf-8")),
            }
        )
    question_inventory_sha256 = _digest(
        {
            "items": sorted(question_inventory, key=lambda row: row["item_id"]),
            "schema": "immer.qwen3.8-label-free-question-inventory/v1",
        }
    )
    candidates = []
    seen_item_ids = set()
    seen_questions = set()
    seen_prompts = set(existing)
    for row in items:
        if not isinstance(row, Mapping):
            raise CohortError("benchmark item is invalid")
        item_id = row.get("item_id")
        question = row.get("question")
        if (
            not isinstance(item_id, str)
            or not item_id
            or not isinstance(question, str)
            or not question.strip()
            or item_id in seen_item_ids
        ):
            raise CohortError("benchmark item identity/question is invalid")
        seen_item_ids.add(item_id)
        normalized_question = question.strip()
        question_sha256 = _sha256_bytes(normalized_question.encode("utf-8"))
        if question_sha256 in seen_questions:
            continue
        seen_questions.add(question_sha256)
        rendered = Qwen38Tokenizer.render_no_thinking_prompt(
            SYSTEM_PROMPT, normalized_question
        )
        tokens = tokenizer.encode(rendered)
        if not tokens or len(tokens) > max_prompt_tokens:
            continue
        prompt_sha256 = prompt_token_sha256(tokens)
        if prompt_sha256 in seen_prompts:
            continue
        seen_prompts.add(prompt_sha256)
        rank_sha256 = _digest(
            {
                "item_id": item_id,
                "question_inventory_sha256": question_inventory_sha256,
                "question_sha256": question_sha256,
                "schema": "immer.qwen3.8-unseen-prompt-rank/v1",
                "seed_sha256": seed_sha256,
            }
        )
        candidates.append(
            {
                "item_id": item_id,
                "prompt": {
                    "sha256": prompt_sha256,
                    "spec_defaults": {"question_sha256": question_sha256},
                    "token_ids": list(tokens),
                },
                "question_sha256": question_sha256,
                "rank_sha256": rank_sha256,
            }
        )
    ranked = sorted(
        candidates,
        key=lambda row: (row["rank_sha256"], row["item_id"]),
    )
    if len(ranked) < count:
        raise CohortError("not enough unique bounded unseen prompts")
    selected = ranked[:count]
    prompts = sorted((row["prompt"] for row in selected), key=lambda row: row["sha256"])
    prompt_document = {"prompts": prompts}
    body = {
        "benchmark_sha256": _sha256_bytes(benchmark_raw),
        "count": count,
        "existing_prompt_sha256s": sorted(existing),
        "max_prompt_tokens": max_prompt_tokens,
        "prompt_registry_sha256": _digest(prompt_document),
        "question_inventory_sha256": question_inventory_sha256,
        "seed_sha256": seed_sha256,
        "selected": [
            {
                "item_id_sha256": _sha256_bytes(row["item_id"].encode("utf-8")),
                "prompt_sha256": row["prompt"]["sha256"],
                "question_sha256": row["question_sha256"],
                "rank_sha256": row["rank_sha256"],
                "token_count": len(row["prompt"]["token_ids"]),
            }
            for row in selected
        ],
        "system_prompt_sha256": _sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
        "tokenizer_sha256": _sha256_bytes(tokenizer_raw),
    }
    cohort_document = {
        "body": body,
        "body_sha256": _digest(body),
        "schema": COHORT_SCHEMA,
    }
    return prompt_document, cohort_document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--existing-manifest", action="append", default=[])
    parser.add_argument("--existing-prompt-sha256", action="append", default=[])
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument(
        "--seed-sha256",
        default=hashlib.sha256(b"immer:qwen-unseen-mlp-prompts/v1").hexdigest(),
    )
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--allow-nonofficial-tokenizer", action="store_true")
    parser.add_argument("--output-prompts", required=True)
    parser.add_argument("--output-manifest", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    prompts, cohort = build_cohort(
        benchmark_path=Path(args.benchmark).expanduser().absolute(),
        tokenizer_path=Path(args.tokenizer).expanduser().absolute(),
        existing_manifest_paths=tuple(
            Path(path).expanduser().absolute() for path in args.existing_manifest
        ),
        existing_prompt_sha256s=tuple(args.existing_prompt_sha256),
        count=args.count,
        seed_sha256=args.seed_sha256,
        max_prompt_tokens=args.max_prompt_tokens,
        require_official_tokenizer=not args.allow_nonofficial_tokenizer,
    )
    prompt_bytes = canonical_json_bytes(prompts) + b"\n"
    manifest_bytes = canonical_json_bytes(cohort) + b"\n"
    _persist_exact(Path(args.output_prompts).expanduser().absolute(), prompt_bytes)
    _persist_exact(Path(args.output_manifest).expanduser().absolute(), manifest_bytes)
    print(
        json.dumps(
            {
                "cohort_body_sha256": cohort["body_sha256"],
                "count": len(prompts["prompts"]),
                "manifest_sha256": _sha256_bytes(manifest_bytes),
                "prompt_registry_sha256": _sha256_bytes(prompt_bytes),
                "schema": COHORT_SCHEMA,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
