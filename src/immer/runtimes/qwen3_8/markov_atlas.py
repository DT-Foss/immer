"""Compact corpus-scale token transitions for zero-weight Qwen drafting.

The ordinary online Markov state deliberately retains only recent verified
answers.  This module is the long-term language rail: it compiles any local
text corpus into a bounded variable-order transition map and serves
continuations without opening model weights.  Documents remain separated, so
the atlas never invents transitions across source boundaries.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Any
import zlib


MARKOV_ATLAS_SCHEMA = "immer.qwen3.8-markov-token-atlas/v1"
MARKOV_ATLAS_PREFIX = b"IMMA\x01"
MAX_MARKOV_ATLAS_BYTES = 256 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")


class MarkovAtlasError(RuntimeError):
    """A corpus atlas is malformed, mismatched, or cannot be persisted."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MarkovAtlasError("Markov atlas metadata is not canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _validate_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _stable_read(path: Path) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or not len(MARKOV_ATLAS_PREFIX) < before.st_size <= MAX_MARKOV_ATLAS_BYTES
        ):
            raise MarkovAtlasError("Markov atlas file size is invalid")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(4 * 1024**2, remaining))
            if not chunk:
                raise MarkovAtlasError("Markov atlas returned a short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        linked = path.lstat()
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or (after.st_dev, after.st_ino) != (linked.st_dev, linked.st_ino)
        ):
            raise MarkovAtlasError("Markov atlas changed while read")
        return b"".join(chunks)
    except MarkovAtlasError:
        raise
    except OSError as exc:
        raise MarkovAtlasError(f"cannot read Markov atlas: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class AtlasContinuation:
    token_ids: tuple[int, ...]
    context_order: int
    support: int
    total: int
    minimum_confidence: float

    def __post_init__(self) -> None:
        if (
            not self.token_ids
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in self.token_ids
            )
            or isinstance(self.context_order, bool)
            or not isinstance(self.context_order, int)
            or self.context_order < 1
            or isinstance(self.support, bool)
            or not isinstance(self.support, int)
            or self.support < 1
            or isinstance(self.total, bool)
            or not isinstance(self.total, int)
            or self.total < self.support
            or not math.isfinite(self.minimum_confidence)
            or not 0.0 <= self.minimum_confidence <= 1.0
        ):
            raise ValueError("Markov atlas continuation is invalid")


@dataclass(frozen=True, slots=True)
class _AtlasRow:
    total: int
    branches: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.total, bool)
            or not isinstance(self.total, int)
            or self.total < 1
            or not self.branches
            or any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
                for token, count in self.branches
            )
            or len({token for token, _count in self.branches}) != len(self.branches)
            or sum(count for _token, count in self.branches) > self.total
            or tuple(
                sorted(self.branches, key=lambda row: (-row[1], row[0]))
            )
            != self.branches
        ):
            raise ValueError("Markov atlas transition row is invalid")


class MarkovTokenAtlas:
    """Immutable variable-order token map compiled from document boundaries."""

    def __init__(
        self,
        *,
        vocab_size: int,
        tokenizer_sha256: str,
        max_order: int,
        min_context_count: int,
        max_branches: int,
        document_count: int,
        token_count: int,
        rows: Mapping[tuple[int, ...], _AtlasRow],
    ) -> None:
        self.vocab_size = _positive_int(vocab_size, label="vocab_size")
        if self.vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        self.tokenizer_sha256 = _validate_sha256(
            tokenizer_sha256,
            label="tokenizer_sha256",
        )
        self.max_order = _positive_int(max_order, label="max_order")
        if self.max_order > 16:
            raise ValueError("max_order exceeds 16")
        self.min_context_count = _positive_int(
            min_context_count,
            label="min_context_count",
        )
        self.max_branches = _positive_int(max_branches, label="max_branches")
        if self.max_branches > 64:
            raise ValueError("max_branches exceeds 64")
        self.document_count = _positive_int(document_count, label="document_count")
        self.token_count = _positive_int(token_count, label="token_count")
        normalized = dict(rows)
        if not normalized:
            raise ValueError("Markov atlas contains no transitions")
        for context, row in normalized.items():
            if (
                not isinstance(context, tuple)
                or not 1 <= len(context) <= self.max_order
                or any(
                    isinstance(token, bool)
                    or not isinstance(token, int)
                    or not 0 <= token < self.vocab_size
                    for token in context
                )
                or not isinstance(row, _AtlasRow)
                or row.total < self.min_context_count
                or len(row.branches) > self.max_branches
                or any(token >= self.vocab_size for token, _count in row.branches)
            ):
                raise ValueError("Markov atlas context is invalid")
        self._rows = normalized
        self._artifact_sha256: str | None = None

    @property
    def context_count(self) -> int:
        return len(self._rows)

    @property
    def sha256(self) -> str:
        if self._artifact_sha256 is None:
            self._artifact_sha256 = _sha256(self.to_bytes())
        return self._artifact_sha256

    @classmethod
    def build(
        cls,
        documents: Iterable[Sequence[int]],
        *,
        vocab_size: int,
        tokenizer_sha256: str,
        max_order: int = 6,
        min_context_count: int = 2,
        max_branches: int = 8,
        max_contexts: int = 250_000,
    ) -> "MarkovTokenAtlas":
        """Compile exact counts while allocating full counters only on reuse."""

        vocab_size = _positive_int(vocab_size, label="vocab_size")
        max_order = _positive_int(max_order, label="max_order")
        min_context_count = _positive_int(
            min_context_count,
            label="min_context_count",
        )
        max_branches = _positive_int(max_branches, label="max_branches")
        max_contexts = _positive_int(max_contexts, label="max_contexts")
        _validate_sha256(tokenizer_sha256, label="tokenizer_sha256")
        if vocab_size <= 1 or max_order > 16 or max_branches > 64:
            raise ValueError("Markov atlas build topology is invalid")

        first_observation: dict[tuple[int, ...], int] = {}
        repeated: dict[tuple[int, ...], Counter[int]] = {}
        document_count = 0
        token_count = 0
        for raw_document in documents:
            if isinstance(raw_document, (str, bytes, bytearray)):
                raise TypeError("Markov atlas documents must contain token IDs")
            tokens = tuple(raw_document)
            if not tokens:
                continue
            if any(
                isinstance(token, bool)
                or not isinstance(token, int)
                or not 0 <= token < vocab_size
                for token in tokens
            ):
                raise ValueError("Markov atlas document contains an invalid token")
            document_count += 1
            token_count += len(tokens)
            for index in range(1, len(tokens)):
                target = tokens[index]
                for order in range(1, min(max_order, index) + 1):
                    context = tokens[index - order : index]
                    counter = repeated.get(context)
                    if counter is not None:
                        counter[target] += 1
                        continue
                    previous = first_observation.pop(context, None)
                    if previous is None:
                        first_observation[context] = target
                    else:
                        repeated[context] = Counter((previous, target))
        if document_count == 0 or token_count == 0:
            raise ValueError("cannot build a Markov atlas from an empty corpus")

        candidates = []
        for context, counter in repeated.items():
            total = sum(counter.values())
            if total < min_context_count:
                continue
            branches = tuple(
                sorted(counter.items(), key=lambda row: (-row[1], row[0]))[
                    :max_branches
                ]
            )
            candidates.append((context, _AtlasRow(total=total, branches=branches)))
        if len(candidates) > max_contexts:
            candidates = sorted(
                candidates,
                key=lambda item: (
                    item[1].total,
                    len(item[0]),
                    item[1].branches[0][1],
                    tuple(-token for token in item[0]),
                ),
                reverse=True,
            )[:max_contexts]
        rows = dict(candidates)
        if not rows:
            raise ValueError("corpus has no repeated Markov context")
        return cls(
            vocab_size=vocab_size,
            tokenizer_sha256=tokenizer_sha256,
            max_order=max_order,
            min_context_count=min_context_count,
            max_branches=max_branches,
            document_count=document_count,
            token_count=token_count,
            rows=rows,
        )

    def continuation(
        self,
        history: Sequence[int],
        *,
        max_tokens: int,
        min_support: int = 2,
        min_confidence: float = 0.30,
    ) -> AtlasContinuation | None:
        if isinstance(history, (str, bytes, bytearray)):
            raise TypeError("history must contain token IDs")
        prefix = tuple(history)
        if not prefix or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocab_size
            for token in prefix
        ):
            raise ValueError("history contains an invalid token")
        max_tokens = _positive_int(max_tokens, label="max_tokens")
        min_support = _positive_int(min_support, label="min_support")
        if (
            isinstance(min_confidence, bool)
            or not isinstance(min_confidence, (int, float))
            or not math.isfinite(float(min_confidence))
            or not 0.0 <= float(min_confidence) <= 1.0
        ):
            raise ValueError("min_confidence must lie in [0, 1]")
        min_confidence = float(min_confidence)

        generated: list[int] = []
        first_order = 0
        support = 0
        minimum = 1.0
        total_bound = 0
        context = list(prefix)
        for _index in range(max_tokens):
            match: tuple[int, _AtlasRow] | None = None
            for order in range(min(self.max_order, len(context)), 0, -1):
                row = self._rows.get(tuple(context[-order:]))
                if row is not None:
                    match = (order, row)
                    break
            if match is None:
                break
            order, row = match
            token, count = row.branches[0]
            confidence = count / row.total
            if count < min_support or confidence < min_confidence:
                break
            if not generated:
                first_order = order
                support = count
            else:
                support = min(support, count)
            total_bound = max(total_bound, row.total)
            minimum = min(minimum, confidence)
            generated.append(token)
            context.append(token)
        if not generated:
            return None
        # MarkovPhraseOption represents one support/total pair.  Preserve the
        # path's minimum conditional probability without overstating support.
        total = max(support, math.ceil(support / max(minimum, 1e-12)))
        total = max(total, total_bound if support == total_bound else total)
        return AtlasContinuation(
            token_ids=tuple(generated),
            context_order=first_order,
            support=support,
            total=total,
            minimum_confidence=minimum,
        )

    def to_bytes(self) -> bytes:
        rows = [
            [
                list(context),
                row.total,
                [[token, count] for token, count in row.branches],
            ]
            for context, row in sorted(
                self._rows.items(),
                key=lambda item: (len(item[0]), item[0]),
            )
        ]
        body = {
            "document_count": self.document_count,
            "max_branches": self.max_branches,
            "max_order": self.max_order,
            "min_context_count": self.min_context_count,
            "rows": rows,
            "token_count": self.token_count,
            "tokenizer_sha256": self.tokenizer_sha256,
            "vocab_size": self.vocab_size,
        }
        document = {
            "body": body,
            "schema": MARKOV_ATLAS_SCHEMA,
            "sha256": _digest(body),
        }
        encoded = MARKOV_ATLAS_PREFIX + zlib.compress(_canonical(document), level=9)
        if len(encoded) > MAX_MARKOV_ATLAS_BYTES:
            raise MarkovAtlasError("Markov atlas exceeds its file-size bound")
        self._artifact_sha256 = _sha256(encoded)
        return encoded

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        expected_vocab_size: int | None = None,
        expected_tokenizer_sha256: str | None = None,
    ) -> "MarkovTokenAtlas":
        if not isinstance(data, bytes) or not data.startswith(MARKOV_ATLAS_PREFIX):
            raise MarkovAtlasError("Markov atlas prefix is invalid")
        if len(data) > MAX_MARKOV_ATLAS_BYTES:
            raise MarkovAtlasError("Markov atlas exceeds its file-size bound")
        try:
            raw = zlib.decompress(data[len(MARKOV_ATLAS_PREFIX) :])
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError, zlib.error) as exc:
            raise MarkovAtlasError("cannot decode Markov atlas") from exc
        if (
            not isinstance(document, dict)
            or set(document) != {"body", "schema", "sha256"}
            or document.get("schema") != MARKOV_ATLAS_SCHEMA
            or not isinstance(document.get("body"), dict)
            or document.get("sha256") != _digest(document["body"])
            or _canonical(document) != raw
        ):
            raise MarkovAtlasError("Markov atlas document is invalid")
        body = document["body"]
        expected = {
            "document_count",
            "max_branches",
            "max_order",
            "min_context_count",
            "rows",
            "token_count",
            "tokenizer_sha256",
            "vocab_size",
        }
        if set(body) != expected or not isinstance(body["rows"], list):
            raise MarkovAtlasError("Markov atlas body is invalid")
        rows: dict[tuple[int, ...], _AtlasRow] = {}
        try:
            for value in body["rows"]:
                if (
                    not isinstance(value, list)
                    or len(value) != 3
                    or not isinstance(value[0], list)
                    or not isinstance(value[2], list)
                ):
                    raise ValueError("invalid row")
                context = tuple(value[0])
                if context in rows:
                    raise ValueError("duplicate context")
                rows[context] = _AtlasRow(
                    total=value[1],
                    branches=tuple(tuple(branch) for branch in value[2]),
                )
            atlas = cls(
                vocab_size=body["vocab_size"],
                tokenizer_sha256=body["tokenizer_sha256"],
                max_order=body["max_order"],
                min_context_count=body["min_context_count"],
                max_branches=body["max_branches"],
                document_count=body["document_count"],
                token_count=body["token_count"],
                rows=rows,
            )
            atlas._artifact_sha256 = _sha256(data)
        except (TypeError, ValueError) as exc:
            raise MarkovAtlasError("Markov atlas values are invalid") from exc
        if expected_vocab_size is not None and atlas.vocab_size != expected_vocab_size:
            raise MarkovAtlasError("Markov atlas vocabulary differs from Qwen")
        if (
            expected_tokenizer_sha256 is not None
            and atlas.tokenizer_sha256 != expected_tokenizer_sha256
        ):
            raise MarkovAtlasError("Markov atlas tokenizer differs from Qwen")
        return atlas

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_vocab_size: int | None = None,
        expected_tokenizer_sha256: str | None = None,
    ) -> "MarkovTokenAtlas":
        source = Path(path).expanduser().absolute()
        return cls.from_bytes(
            _stable_read(source),
            expected_vocab_size=expected_vocab_size,
            expected_tokenizer_sha256=expected_tokenizer_sha256,
        )

    def write(self, path: str | Path) -> None:
        target = Path(path).expanduser().absolute()
        target.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_bytes()
        temporary = target.parent / f".{target.name}.{secrets.token_hex(8)}.tmp"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | int(getattr(os, "O_CLOEXEC", 0))
                | int(getattr(os, "O_NOFOLLOW", 0)),
                0o600,
            )
            view = memoryview(data)
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written <= 0:
                    raise OSError("short Markov atlas write")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, target)
        except OSError as exc:
            raise MarkovAtlasError(f"cannot persist Markov atlas: {target}") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def metrics(self) -> dict[str, Any]:
        orders = Counter(len(context) for context in self._rows)
        return {
            "context_count": self.context_count,
            "contexts_by_order": {
                str(order): count for order, count in sorted(orders.items())
            },
            "document_count": self.document_count,
            "max_branches": self.max_branches,
            "max_order": self.max_order,
            "min_context_count": self.min_context_count,
            "schema": MARKOV_ATLAS_SCHEMA,
            "sha256": self.sha256,
            "token_count": self.token_count,
            "tokenizer_sha256": self.tokenizer_sha256,
            "vocab_size": self.vocab_size,
        }


def tokenizer_file_sha256(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    return _sha256(source.read_bytes())


__all__ = [
    "AtlasContinuation",
    "MARKOV_ATLAS_PREFIX",
    "MARKOV_ATLAS_SCHEMA",
    "MarkovAtlasError",
    "MarkovTokenAtlas",
    "tokenizer_file_sha256",
]
