#!/usr/bin/env python3
"""Import existing target-confirmed Qwen receipts into Markov memory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import sys
from typing import Any

from immer.runtimes.qwen3_8.markov_draft import (
    FingerprintRollingK4DraftProvider,
    MarkovDraftError,
)


IMPORT_JOURNAL_SCHEMA = "immer.qwen3.8-markov-receipt-imports/v1"
Episode = tuple[tuple[int, ...], tuple[int, ...]]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _episode_tokens(episode: Episode) -> tuple[int, ...]:
    prompt, generated = episode
    return (*prompt, *generated)


def _episode_digest(episode: Episode) -> str:
    prompt, generated = episode
    if not prompt:
        return _digest(list(generated))
    return _digest(
        {
            "generated_token_ids": list(generated),
            "prompt_token_ids": list(prompt),
            "schema": "immer.qwen3.8-structured-episode/v1",
        }
    )


def _token_tuple(value: object, *, vocab_size: int) -> tuple[int, ...] | None:
    if not isinstance(value, list) or any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or not 0 <= token < vocab_size
        for token in value
    ):
        return None
    return tuple(value)


def _extract_document(
    value: object,
    *,
    vocab_size: int,
    min_tokens: int,
    source: Path,
    episodes: dict[Episode, str],
) -> None:
    if isinstance(value, dict):
        generated = _token_tuple(
            value.get("generated_token_ids"), vocab_size=vocab_size
        )
        if generated:
            prompt = None
            for key in (
                "effective_prompt_token_ids",
                "prompt_token_ids",
                "source_prompt_token_ids",
            ):
                prompt = _token_tuple(value.get(key), vocab_size=vocab_size)
                if prompt is not None:
                    break
            episode = (prompt or (), generated)
            if len(_episode_tokens(episode)) >= min_tokens:
                episodes.setdefault(episode, str(source))
        for child in value.values():
            _extract_document(
                child,
                vocab_size=vocab_size,
                min_tokens=min_tokens,
                source=source,
                episodes=episodes,
            )
    elif isinstance(value, list):
        for child in value:
            _extract_document(
                child,
                vocab_size=vocab_size,
                min_tokens=min_tokens,
                source=source,
                episodes=episodes,
            )


def scan_receipts(
    roots: tuple[Path, ...],
    *,
    vocab_size: int,
    min_tokens: int,
    max_file_bytes: int,
) -> tuple[dict[Episode, str], int, int]:
    episodes: dict[Episode, str] = {}
    scanned = 0
    rejected = 0
    files: set[Path] = set()
    for root in roots:
        source = root.expanduser().resolve()
        if source.is_file() and not source.is_symlink():
            files.add(source)
        elif source.is_dir() and not source.is_symlink():
            files.update(
                path
                for path in source.rglob("*.json")
                if path.is_file() and not path.is_symlink()
            )
        else:
            raise FileNotFoundError(
                f"receipt root is not a regular file/directory: {root}"
            )
    for path in sorted(files):
        try:
            metadata = path.stat()
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_file_bytes:
                rejected += 1
                continue
            document = json.loads(path.read_text(encoding="utf-8"))
            scanned += 1
            _extract_document(
                document,
                vocab_size=vocab_size,
                min_tokens=min_tokens,
                source=path,
                episodes=episodes,
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            rejected += 1
    return episodes, scanned, rejected


def _journal_path(state_path: Path) -> Path:
    return state_path.parent / f"{state_path.name}.imports.json"


def _load_journal(path: Path, *, vocab_size: int) -> set[str]:
    if not path.exists():
        return set()
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MarkovDraftError("cannot read Markov import journal") from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"body", "schema", "sha256"}
        or document.get("schema") != IMPORT_JOURNAL_SCHEMA
        or not isinstance(document.get("body"), dict)
        or document.get("sha256") != _digest(document["body"])
        or _canonical(document) != raw
        or document["body"].get("vocab_size") != vocab_size
        or not isinstance(document["body"].get("episode_sha256"), list)
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in document["body"]["episode_sha256"]
        )
    ):
        raise MarkovDraftError("Markov import journal is invalid")
    return set(document["body"]["episode_sha256"])


def _provider(
    state_path: Path,
    *,
    vocab_size: int,
    proposal_width: int,
) -> FingerprintRollingK4DraftProvider:
    return FingerprintRollingK4DraftProvider(
        vocab_size=vocab_size,
        state_path=state_path,
        proposal_width=proposal_width,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    state_path = args.state.expanduser().resolve()
    episodes, scanned, rejected = scan_receipts(
        tuple(args.root),
        vocab_size=args.vocab_size,
        min_tokens=args.min_tokens,
        max_file_bytes=int(args.max_file_mb * 1024**2),
    )
    journal_path = _journal_path(state_path)
    if journal_path.exists() and not state_path.exists():
        raise MarkovDraftError("Markov import journal exists without its state")
    legacy_imported = _load_journal(journal_path, vocab_size=args.vocab_size)
    imported = set(legacy_imported)
    if state_path.exists():
        provider = _provider(
            state_path,
            vocab_size=args.vocab_size,
            proposal_width=args.proposal_width,
        )
        try:
            retained_transitions = provider.confirmed_transitions()
            imported.update(provider.imported_episode_sha256s())
            retained_digests = {
                _episode_digest((prompt or (), generated))
                for prompt, generated in retained_transitions
            }
            imported.update(retained_digests)
            if not args.dry_run:
                provider.register_imported_episode_sha256s(
                    tuple(sorted(legacy_imported | retained_digests))
                )
                imported.update(provider.imported_episode_sha256s())
        finally:
            provider.close()
    ordered = sorted(
        episodes,
        key=lambda row: (episodes[row], _episode_digest(row)),
    )
    pending = [row for row in ordered if _episode_digest(row) not in imported]
    if args.limit is not None:
        pending = pending[: args.limit]
    imported_now = 0
    imported_tokens = 0
    if not args.dry_run:
        for episode in pending:
            provider = _provider(
                state_path,
                vocab_size=args.vocab_size,
                proposal_width=args.proposal_width,
            )
            try:
                prompt, generated = episode
                added = (
                    provider.import_confirmed_transition(
                        prompt,
                        generated,
                        _episode_digest(episode),
                    )
                    if prompt
                    else provider.import_confirmed_episode(
                        generated,
                        _episode_digest(episode),
                    )
                )
            finally:
                provider.close()
            if not added:
                continue
            imported.add(_episode_digest(episode))
            imported_now += 1
            imported_tokens += len(_episode_tokens(episode))
    metrics: dict[str, Any] = {}
    if state_path.exists():
        provider = _provider(
            state_path,
            vocab_size=args.vocab_size,
            proposal_width=args.proposal_width,
        )
        try:
            metrics = provider.metrics().to_dict()
        finally:
            provider.close()
    return {
        "schema": "immer.qwen3.8-markov-receipt-bootstrap/v1",
        "status": "dry-run" if args.dry_run else "complete",
        "files_scanned": scanned,
        "files_rejected": rejected,
        "episodes_extracted": len(episodes),
        "episodes_pending": len(pending),
        "episodes_imported": imported_now,
        "tokens_imported": imported_tokens,
        "imported_episode_digests": len(imported),
        "state": str(state_path),
        "metrics": metrics,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import existing target-confirmed Qwen JSON receipts into Markov memory"
    )
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--vocab-size", type=int, default=248_320)
    parser.add_argument("--proposal-width", type=int, default=3)
    parser.add_argument("--min-tokens", type=int, default=4)
    parser.add_argument("--max-file-mb", type=float, default=64.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.vocab_size <= 1 or not 1 <= args.proposal_width <= 15:
            raise ValueError("vocab/proposal width is invalid")
        if args.min_tokens <= 0 or args.max_file_mb <= 0:
            raise ValueError("receipt bounds must be positive")
        if args.limit is not None and args.limit <= 0:
            raise ValueError("limit must be positive")
        result = run(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except (MarkovDraftError, OSError, TypeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "reason": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
