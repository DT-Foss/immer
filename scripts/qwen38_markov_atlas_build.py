#!/usr/bin/env python3
"""Compile local text into a zero-model-byte Qwen token-transition atlas."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator
import json
from pathlib import Path
import stat
import sys
from typing import Any

from immer.runtimes.qwen3_8.encoding import Qwen38Tokenizer
from immer.runtimes.qwen3_8.markov_atlas import (
    MarkovAtlasError,
    MarkovTokenAtlas,
    tokenizer_file_sha256,
)


DEFAULT_TEXT_FIELDS = (
    "answer",
    "completion",
    "content",
    "generated_text",
    "output",
    "response",
    "text",
)


class AtlasBuildError(RuntimeError):
    pass


def _regular(path: Path, *, max_file_bytes: int) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise AtlasBuildError(f"cannot inspect corpus file: {path}") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size > max_file_bytes
    ):
        raise AtlasBuildError(f"corpus file is not an admissible regular file: {path}")


def _json_texts(value: object, fields: frozenset[str]) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, child in value.items():
            if key in fields and isinstance(child, str):
                yield child
            elif isinstance(child, (dict, list)):
                yield from _json_texts(child, fields)
    elif isinstance(value, list):
        for child in value:
            yield from _json_texts(child, fields)


def _plain_texts(path: Path) -> Iterator[str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            paragraph: list[str] = []
            for line in handle:
                stripped = line.strip()
                if stripped:
                    paragraph.append(stripped)
                    continue
                if paragraph:
                    yield "\n".join(paragraph)
                    paragraph.clear()
            if paragraph:
                yield "\n".join(paragraph)
    except (OSError, UnicodeError) as exc:
        raise AtlasBuildError(f"cannot read text corpus: {path}") from exc


def _jsonl_texts(path: Path, fields: frozenset[str]) -> Iterator[str]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AtlasBuildError(
                        f"invalid JSONL at {path}:{line_number}"
                    ) from exc
                yield from _json_texts(value, fields)
    except (OSError, UnicodeError) as exc:
        raise AtlasBuildError(f"cannot read JSONL corpus: {path}") from exc


def _json_file_texts(path: Path, fields: frozenset[str]) -> Iterator[str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AtlasBuildError(f"cannot read JSON corpus: {path}") from exc
    yield from _json_texts(value, fields)


def iter_corpus_texts(
    paths: Iterable[Path],
    *,
    fields: frozenset[str],
    max_file_bytes: int,
    min_characters: int,
) -> Iterator[str]:
    seen: set[str] = set()
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        _regular(path, max_file_bytes=max_file_bytes)
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            rows = _plain_texts(path)
        elif suffix == ".jsonl":
            rows = _jsonl_texts(path, fields)
        elif suffix == ".json":
            rows = _json_file_texts(path, fields)
        else:
            raise AtlasBuildError(f"unsupported corpus suffix: {path}")
        for text in rows:
            normalized = text.strip()
            if len(normalized) < min_characters or normalized in seen:
                continue
            seen.add(normalized)
            yield normalized


def token_documents(
    texts: Iterable[str],
    tokenizer: Qwen38Tokenizer,
    *,
    max_tokens: int,
    max_document_tokens: int,
    counters: dict[str, int],
) -> Iterator[tuple[int, ...]]:
    remaining = max_tokens
    for text in texts:
        if remaining <= 0:
            break
        tokens = tokenizer.encode(text)
        if len(tokens) < 2:
            continue
        tokens = tokens[: min(max_document_tokens, remaining)]
        if len(tokens) < 2:
            break
        counters["documents"] += 1
        counters["characters"] += len(text)
        counters["tokens"] += len(tokens)
        remaining -= len(tokens)
        yield tokens


def run(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer_path = args.tokenizer.expanduser().resolve()
    tokenizer = Qwen38Tokenizer(tokenizer_path, require_official=not args.allow_test_tokenizer)
    fields = frozenset(args.field or DEFAULT_TEXT_FIELDS)
    if not fields or any(not field for field in fields):
        raise ValueError("at least one non-empty text field is required")
    counters = {"characters": 0, "documents": 0, "tokens": 0}
    texts = iter_corpus_texts(
        args.input,
        fields=fields,
        max_file_bytes=int(args.max_file_mb * 1024**2),
        min_characters=args.min_characters,
    )
    documents = token_documents(
        texts,
        tokenizer,
        max_tokens=args.max_tokens,
        max_document_tokens=args.max_document_tokens,
        counters=counters,
    )
    atlas = MarkovTokenAtlas.build(
        documents,
        vocab_size=args.vocab_size,
        tokenizer_sha256=tokenizer_file_sha256(tokenizer_path),
        max_order=args.max_order,
        min_context_count=args.min_context_count,
        max_branches=args.max_branches,
        max_contexts=args.max_contexts,
    )
    atlas.write(args.output)
    return {
        "atlas": atlas.metrics(),
        "characters_consumed": counters["characters"],
        "documents_consumed": counters["documents"],
        "output": str(args.output.expanduser().absolute()),
        "schema": "immer.qwen3.8-markov-atlas-build-result/v1",
        "status": "complete",
        "tokens_consumed": counters["tokens"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile local text/JSONL into a compact Qwen token Markov atlas"
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field", action="append")
    parser.add_argument("--vocab-size", type=int, default=248_320)
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--min-context-count", type=int, default=2)
    parser.add_argument("--max-branches", type=int, default=8)
    parser.add_argument("--max-contexts", type=int, default=250_000)
    parser.add_argument("--max-tokens", type=int, default=1_000_000)
    parser.add_argument("--max-document-tokens", type=int, default=2048)
    parser.add_argument("--min-characters", type=int, default=20)
    parser.add_argument("--max-file-mb", type=float, default=256.0)
    parser.add_argument("--allow-test-tokenizer", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        positive = (
            args.vocab_size,
            args.max_order,
            args.min_context_count,
            args.max_branches,
            args.max_contexts,
            args.max_tokens,
            args.max_document_tokens,
            args.min_characters,
        )
        if any(value <= 0 for value in positive) or args.max_file_mb <= 0:
            raise ValueError("atlas build bounds must be positive")
        result = run(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (AtlasBuildError, MarkovAtlasError, OSError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {"reason": f"{type(exc).__name__}: {exc}", "status": "error"},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
