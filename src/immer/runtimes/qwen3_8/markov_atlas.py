"""Compact corpus-scale token transitions for zero-weight Qwen drafting.

The ordinary online Markov state deliberately retains only recent verified
answers.  This module is the long-term language rail: it compiles any local
text corpus into a bounded variable-order transition map and serves
continuations without opening model weights.  Documents remain separated, so
the atlas never invents transitions across source boundaries.
"""

from __future__ import annotations

from array import array
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
import struct
import sys
from typing import Any
import zlib


MARKOV_ATLAS_SCHEMA = "immer.qwen3.8-markov-token-atlas/v2"
LEGACY_MARKOV_ATLAS_SCHEMA = "immer.qwen3.8-markov-token-atlas/v1"
MARKOV_ATLAS_PREFIX = b"IMMA\x02"
LEGACY_MARKOV_ATLAS_PREFIX = b"IMMA\x01"
MAX_MARKOV_ATLAS_BYTES = 256 * 1024 * 1024
MAX_MARKOV_ATLAS_EXPANDED_BYTES = 512 * 1024 * 1024
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


def _context_hash(context: Sequence[int]) -> int:
    # FNV-1a over native token integers is enough for the in-memory lookup:
    # every hit is still checked against the exact flat context bytes.  A
    # cryptographic digest here cost several seconds across 500k rows at each
    # process start without adding integrity (the compact payload already owns
    # a SHA-256).
    fingerprint = (1_469_598_103_934_665_603 ^ len(context)) & ((1 << 64) - 1)
    for token in context:
        fingerprint ^= int(token)
        fingerprint = (fingerprint * 1_099_511_628_211) & ((1 << 64) - 1)
    return fingerprint


def _little_array_bytes(values: array) -> bytes:
    if sys.byteorder == "little" or values.itemsize == 1:
        return values.tobytes()
    copied = array(values.typecode, values)
    copied.byteswap()
    return copied.tobytes()


def _array_from_little(typecode: str, value: memoryview) -> array:
    result = array(typecode)
    result.frombytes(value.tobytes())
    if sys.byteorder != "little" and result.itemsize > 1:
        result.byteswap()
    return result


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
class AtlasTokenEvidence:
    token_id: int
    context_order: int
    support: int
    total: int
    probability: float
    score: float

    def __post_init__(self) -> None:
        absent = self.context_order == self.support == self.total == 0
        if (
            isinstance(self.token_id, bool)
            or not isinstance(self.token_id, int)
            or self.token_id < 0
            or isinstance(self.context_order, bool)
            or not isinstance(self.context_order, int)
            or self.context_order < 0
            or isinstance(self.support, bool)
            or not isinstance(self.support, int)
            or self.support < 0
            or isinstance(self.total, bool)
            or not isinstance(self.total, int)
            or self.total < self.support
            or absent != (self.support == 0)
            or not math.isfinite(self.probability)
            or not 0.0 <= self.probability <= 1.0
            or not math.isfinite(self.score)
            or not 0.0 <= self.score <= 1.0
            or (absent and (self.probability != 0.0 or self.score != 0.0))
        ):
            raise ValueError("Markov atlas token evidence is invalid")


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
    """Immutable variable-order token map with a compact flat-array index."""

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
        self._initialize_metadata(
            vocab_size=vocab_size,
            tokenizer_sha256=tokenizer_sha256,
            max_order=max_order,
            min_context_count=min_context_count,
            max_branches=max_branches,
            document_count=document_count,
            token_count=token_count,
        )
        self._install_rows(rows.items(), presorted=False)

    def _initialize_metadata(
        self,
        *,
        vocab_size: int,
        tokenizer_sha256: str,
        max_order: int,
        min_context_count: int,
        max_branches: int,
        document_count: int,
        token_count: int,
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
        self._artifact_sha256: str | None = None

    def _install_rows(
        self,
        rows: Iterable[tuple[tuple[int, ...], _AtlasRow]],
        *,
        presorted: bool,
    ) -> None:
        materialized = rows if presorted else sorted(rows, key=lambda item: (len(item[0]), item[0]))
        self._orders = array("B")
        self._context_offsets = array("I", (0,))
        self._context_tokens = array("I")
        self._totals = array("I")
        self._branch_offsets = array("I", (0,))
        self._branch_tokens = array("I")
        self._branch_counts = array("I")
        self._hash_index: dict[int, int | tuple[int, ...]] = {}
        previous: tuple[int, ...] | None = None
        for context, row in materialized:
            if (
                not isinstance(context, tuple)
                or not 0 <= len(context) <= self.max_order
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
            if previous is not None and (len(context), context) <= (
                len(previous),
                previous,
            ):
                raise ValueError("Markov atlas contexts are duplicated or unsorted")
            index = len(self._totals)
            self._orders.append(len(context))
            self._context_tokens.extend(context)
            self._context_offsets.append(len(self._context_tokens))
            self._totals.append(row.total)
            for token, count in row.branches:
                self._branch_tokens.append(token)
                self._branch_counts.append(count)
            self._branch_offsets.append(len(self._branch_tokens))
            fingerprint = _context_hash(context)
            existing = self._hash_index.get(fingerprint)
            if existing is None:
                self._hash_index[fingerprint] = index
            elif isinstance(existing, int):
                self._hash_index[fingerprint] = (existing, index)
            else:
                self._hash_index[fingerprint] = (*existing, index)
            previous = context
        if not self._totals:
            raise ValueError("Markov atlas contains no transitions")

    def _context_matches(self, index: int, context: tuple[int, ...]) -> bool:
        if self._orders[index] != len(context):
            return False
        start = self._context_offsets[index]
        return all(
            self._context_tokens[start + offset] == token
            for offset, token in enumerate(context)
        )

    def _row_at(self, index: int) -> _AtlasRow:
        start = self._branch_offsets[index]
        stop = self._branch_offsets[index + 1]
        return _AtlasRow(
            total=self._totals[index],
            branches=tuple(
                (self._branch_tokens[offset], self._branch_counts[offset])
                for offset in range(start, stop)
            ),
        )

    def _lookup_index(self, context: tuple[int, ...]) -> int | None:
        candidates = self._hash_index.get(_context_hash(context))
        if candidates is None:
            return None
        indexes = (candidates,) if isinstance(candidates, int) else candidates
        for index in indexes:
            if self._context_matches(index, context):
                return index
        return None

    def _lookup(self, context: tuple[int, ...]) -> _AtlasRow | None:
        index = self._lookup_index(context)
        return None if index is None else self._row_at(index)

    def _branch_evidence(
        self,
        *,
        token_id: int,
        context_order: int,
        support: int,
        total: int,
    ) -> AtlasTokenEvidence:
        probability = support / total
        support_strength = 1.0 - math.exp(-support / 4.0)
        order_strength = 0.5 + 0.5 * context_order / self.max_order
        return AtlasTokenEvidence(
            token_id=token_id,
            context_order=context_order,
            support=support,
            total=total,
            probability=probability,
            score=probability * support_strength * order_strength,
        )

    @staticmethod
    def _evidence_strength(
        evidence: AtlasTokenEvidence,
    ) -> tuple[float, int, int, int]:
        return (
            evidence.score,
            evidence.context_order,
            evidence.support,
            -evidence.total,
        )

    @property
    def context_count(self) -> int:
        return len(self._totals)

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
            for index in range(len(tokens)):
                target = tokens[index]
                for order in range(min(max_order, index) + 1):
                    context = () if order == 0 else tokens[index - order : index]
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
                row = self._lookup(tuple(context[-order:]))
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

    def token_evidence(
        self,
        history: Sequence[int],
        token_id: int,
    ) -> AtlasTokenEvidence:
        """Score one supplied candidate across every available PPM order.

        Unlike :meth:`continuation`, this never substitutes the Atlas Top-1.
        It answers the Council question "how much corpus mass supports this
        exact external token?".  Support saturation prevents a single rare
        phrase from looking authoritative, while the order term rewards more
        specific contexts without suppressing useful unigram evidence.
        """

        if isinstance(history, (str, bytes, bytearray)):
            raise TypeError("history must contain token IDs")
        context = tuple(history)
        if not context or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocab_size
            for token in context
        ):
            raise ValueError("history contains an invalid token")
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or not 0 <= token_id < self.vocab_size
        ):
            raise ValueError("candidate token is outside the Atlas vocabulary")
        candidates: list[AtlasTokenEvidence] = []
        for order in range(min(self.max_order, len(context)) + 1):
            key = () if order == 0 else context[-order:]
            row = self._lookup(key)
            if row is None:
                continue
            count = next(
                (
                    branch_count
                    for branch_token, branch_count in row.branches
                    if branch_token == token_id
                ),
                0,
            )
            if count <= 0:
                continue
            candidates.append(
                self._branch_evidence(
                    token_id=token_id,
                    context_order=order,
                    support=count,
                    total=row.total,
                )
            )
        if not candidates:
            return AtlasTokenEvidence(
                token_id=token_id,
                context_order=0,
                support=0,
                total=0,
                probability=0.0,
                score=0.0,
            )
        return max(candidates, key=self._evidence_strength)

    def token_options(
        self,
        history: Sequence[int],
        *,
        limit: int | None = None,
    ) -> tuple[AtlasTokenEvidence, ...]:
        """Return the strongest retained evidence for each available token.

        Every matching PPM order, including the order-zero corpus row, is
        scanned directly in the flat branch arrays.  The result is bounded by
        ``max_branches`` and ordered by descending evidence strength, with the
        token ID providing a stable ascending tie-breaker.
        """

        if isinstance(history, (str, bytes, bytearray)):
            raise TypeError("history must contain token IDs")
        context = tuple(history)
        if not context or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocab_size
            for token in context
        ):
            raise ValueError("history contains an invalid token")
        option_limit = (
            self.max_branches
            if limit is None
            else _positive_int(limit, label="limit")
        )
        if option_limit > self.max_branches:
            raise ValueError("limit must not exceed max_branches")

        strongest: dict[int, AtlasTokenEvidence] = {}
        for order in range(min(self.max_order, len(context)) + 1):
            key = () if order == 0 else context[-order:]
            index = self._lookup_index(key)
            if index is None:
                continue
            total = self._totals[index]
            start = self._branch_offsets[index]
            stop = self._branch_offsets[index + 1]
            for offset in range(start, stop):
                token_id = self._branch_tokens[offset]
                evidence = self._branch_evidence(
                    token_id=token_id,
                    context_order=order,
                    support=self._branch_counts[offset],
                    total=total,
                )
                previous = strongest.get(token_id)
                if previous is None or self._evidence_strength(
                    evidence
                ) > self._evidence_strength(previous):
                    strongest[token_id] = evidence

        ranked = sorted(
            strongest.values(),
            key=lambda row: (
                -row.score,
                -row.context_order,
                -row.support,
                -row.probability,
                row.token_id,
            ),
        )
        return tuple(ranked[:option_limit])

    def sequence_evidence(
        self,
        history: Sequence[int],
        token_ids: Sequence[int],
    ) -> tuple[AtlasTokenEvidence, ...]:
        if isinstance(token_ids, (str, bytes, bytearray)):
            raise TypeError("token_ids must contain integers")
        proposed = tuple(token_ids)
        if not proposed:
            raise ValueError("token_ids must not be empty")
        context = list(history)
        rows = []
        for token_id in proposed:
            evidence = self.token_evidence(context, token_id)
            rows.append(evidence)
            context.append(token_id)
        return tuple(rows)

    def to_bytes(self) -> bytes:
        arrays = (
            ("orders", "B", self._orders),
            ("context_offsets", "I", self._context_offsets),
            ("context_tokens", "I", self._context_tokens),
            ("totals", "I", self._totals),
            ("branch_offsets", "I", self._branch_offsets),
            ("branch_tokens", "I", self._branch_tokens),
            ("branch_counts", "I", self._branch_counts),
        )
        sections = []
        payload = bytearray()
        for name, typecode, values in arrays:
            encoded = _little_array_bytes(values)
            sections.append([name, typecode, len(values), len(encoded)])
            payload.extend(encoded)
        body = {
            "context_count": self.context_count,
            "document_count": self.document_count,
            "layout": "flat-arrays-le/v1",
            "max_branches": self.max_branches,
            "max_order": self.max_order,
            "min_context_count": self.min_context_count,
            "payload_bytes": len(payload),
            "payload_sha256": _sha256(payload),
            "sections": sections,
            "token_count": self.token_count,
            "tokenizer_sha256": self.tokenizer_sha256,
            "vocab_size": self.vocab_size,
        }
        document = {
            "body": body,
            "schema": MARKOV_ATLAS_SCHEMA,
            "sha256": _digest(body),
        }
        header = _canonical(document)
        raw = struct.pack("<I", len(header)) + header + payload
        encoded = MARKOV_ATLAS_PREFIX + zlib.compress(raw, level=9)
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
        if not isinstance(data, bytes) or not data.startswith(
            (MARKOV_ATLAS_PREFIX, LEGACY_MARKOV_ATLAS_PREFIX)
        ):
            raise MarkovAtlasError("Markov atlas prefix is invalid")
        if len(data) > MAX_MARKOV_ATLAS_BYTES:
            raise MarkovAtlasError("Markov atlas exceeds its file-size bound")
        try:
            prefix = (
                MARKOV_ATLAS_PREFIX
                if data.startswith(MARKOV_ATLAS_PREFIX)
                else LEGACY_MARKOV_ATLAS_PREFIX
            )
            decoder = zlib.decompressobj()
            raw = decoder.decompress(
                data[len(prefix) :],
                MAX_MARKOV_ATLAS_EXPANDED_BYTES + 1,
            )
            if (
                len(raw) > MAX_MARKOV_ATLAS_EXPANDED_BYTES
                or decoder.unconsumed_tail
                or decoder.unused_data
                or not decoder.eof
            ):
                raise zlib.error("expanded Markov atlas exceeds its bound")
        except zlib.error as exc:
            raise MarkovAtlasError("cannot decode Markov atlas") from exc
        if prefix == LEGACY_MARKOV_ATLAS_PREFIX:
            atlas = cls._from_legacy_bytes(raw)
        else:
            atlas = cls._from_compact_bytes(raw)
        atlas._artifact_sha256 = _sha256(data)
        if expected_vocab_size is not None and atlas.vocab_size != expected_vocab_size:
            raise MarkovAtlasError("Markov atlas vocabulary differs from Qwen")
        if (
            expected_tokenizer_sha256 is not None
            and atlas.tokenizer_sha256 != expected_tokenizer_sha256
        ):
            raise MarkovAtlasError("Markov atlas tokenizer differs from Qwen")
        return atlas

    @classmethod
    def _from_compact_bytes(cls, raw: bytes) -> "MarkovTokenAtlas":
        if len(raw) < 4:
            raise MarkovAtlasError("Markov atlas compact header is truncated")
        header_bytes = struct.unpack_from("<I", raw, 0)[0]
        if not 0 < header_bytes <= 64 * 1024 or 4 + header_bytes > len(raw):
            raise MarkovAtlasError("Markov atlas compact header size is invalid")
        header_raw = raw[4 : 4 + header_bytes]
        payload = memoryview(raw)[4 + header_bytes :]
        try:
            document = json.loads(header_raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise MarkovAtlasError("cannot decode Markov atlas header") from exc
        if (
            not isinstance(document, dict)
            or set(document) != {"body", "schema", "sha256"}
            or document.get("schema") != MARKOV_ATLAS_SCHEMA
            or not isinstance(document.get("body"), dict)
            or document.get("sha256") != _digest(document["body"])
            or _canonical(document) != header_raw
        ):
            raise MarkovAtlasError("Markov atlas document is invalid")
        body = document["body"]
        if body.get("layout") == "flat-arrays-le/v1":
            return cls._from_array_payload(body, payload)
        expected = {
            "context_count",
            "document_count",
            "max_branches",
            "max_order",
            "min_context_count",
            "payload_bytes",
            "payload_sha256",
            "token_count",
            "tokenizer_sha256",
            "vocab_size",
        }
        if (
            set(body) != expected
            or body.get("payload_bytes") != len(payload)
            or body.get("payload_sha256") != _sha256(payload)
        ):
            raise MarkovAtlasError("Markov atlas body is invalid")

        context_count = body.get("context_count")
        if (
            isinstance(context_count, bool)
            or not isinstance(context_count, int)
            or context_count <= 0
        ):
            raise MarkovAtlasError("Markov atlas context count is invalid")
        cursor = 0

        def rows() -> Iterable[tuple[tuple[int, ...], _AtlasRow]]:
            nonlocal cursor
            for _index in range(context_count):
                if cursor >= len(payload):
                    raise ValueError("truncated compact row")
                order = payload[cursor]
                cursor += 1
                context_bytes = order * 4
                if cursor + context_bytes + 5 > len(payload):
                    raise ValueError("truncated compact context")
                context = tuple(
                    struct.unpack_from(f"<{order}I", payload, cursor)
                ) if order else ()
                cursor += context_bytes
                total = struct.unpack_from("<I", payload, cursor)[0]
                cursor += 4
                branch_count = payload[cursor]
                cursor += 1
                branch_bytes = branch_count * 8
                if cursor + branch_bytes > len(payload):
                    raise ValueError("truncated compact branches")
                branches = tuple(
                    struct.unpack_from("<II", payload, cursor + 8 * branch)
                    for branch in range(branch_count)
                )
                cursor += branch_bytes
                yield context, _AtlasRow(total=total, branches=branches)

        try:
            atlas = cls.__new__(cls)
            atlas._initialize_metadata(
                vocab_size=body["vocab_size"],
                tokenizer_sha256=body["tokenizer_sha256"],
                max_order=body["max_order"],
                min_context_count=body["min_context_count"],
                max_branches=body["max_branches"],
                document_count=body["document_count"],
                token_count=body["token_count"],
            )
            atlas._install_rows(rows(), presorted=True)
            if cursor != len(payload):
                raise ValueError("trailing compact row bytes")
        except (TypeError, ValueError, struct.error) as exc:
            raise MarkovAtlasError("Markov atlas values are invalid") from exc
        return atlas

    @classmethod
    def _from_array_payload(
        cls,
        body: Mapping[str, Any],
        payload: memoryview,
    ) -> "MarkovTokenAtlas":
        expected = {
            "context_count",
            "document_count",
            "layout",
            "max_branches",
            "max_order",
            "min_context_count",
            "payload_bytes",
            "payload_sha256",
            "sections",
            "token_count",
            "tokenizer_sha256",
            "vocab_size",
        }
        section_specs = (
            ("orders", "B"),
            ("context_offsets", "I"),
            ("context_tokens", "I"),
            ("totals", "I"),
            ("branch_offsets", "I"),
            ("branch_tokens", "I"),
            ("branch_counts", "I"),
        )
        if (
            set(body) != expected
            or body.get("layout") != "flat-arrays-le/v1"
            or body.get("payload_bytes") != len(payload)
            or body.get("payload_sha256") != _sha256(payload)
            or not isinstance(body.get("sections"), list)
            or len(body["sections"]) != len(section_specs)
        ):
            raise MarkovAtlasError("Markov atlas array body is invalid")
        context_count = body.get("context_count")
        if (
            isinstance(context_count, bool)
            or not isinstance(context_count, int)
            or context_count <= 0
        ):
            raise MarkovAtlasError("Markov atlas context count is invalid")

        arrays: dict[str, array] = {}
        cursor = 0
        try:
            for descriptor, (expected_name, expected_type) in zip(
                body["sections"],
                section_specs,
                strict=True,
            ):
                if (
                    not isinstance(descriptor, list)
                    or len(descriptor) != 4
                    or descriptor[:2] != [expected_name, expected_type]
                    or isinstance(descriptor[2], bool)
                    or not isinstance(descriptor[2], int)
                    or descriptor[2] < 0
                    or isinstance(descriptor[3], bool)
                    or not isinstance(descriptor[3], int)
                    or descriptor[3] < 0
                ):
                    raise ValueError("invalid array section")
                items, byte_count = descriptor[2], descriptor[3]
                item_size = array(expected_type).itemsize
                if byte_count != items * item_size or cursor + byte_count > len(
                    payload
                ):
                    raise ValueError("array section size mismatch")
                arrays[expected_name] = _array_from_little(
                    expected_type,
                    payload[cursor : cursor + byte_count],
                )
                cursor += byte_count
            if cursor != len(payload):
                raise ValueError("trailing array payload")

            atlas = cls.__new__(cls)
            atlas._initialize_metadata(
                vocab_size=body["vocab_size"],
                tokenizer_sha256=body["tokenizer_sha256"],
                max_order=body["max_order"],
                min_context_count=body["min_context_count"],
                max_branches=body["max_branches"],
                document_count=body["document_count"],
                token_count=body["token_count"],
            )
            atlas._orders = arrays["orders"]
            atlas._context_offsets = arrays["context_offsets"]
            atlas._context_tokens = arrays["context_tokens"]
            atlas._totals = arrays["totals"]
            atlas._branch_offsets = arrays["branch_offsets"]
            atlas._branch_tokens = arrays["branch_tokens"]
            atlas._branch_counts = arrays["branch_counts"]
            if (
                len(atlas._orders) != context_count
                or len(atlas._totals) != context_count
                or len(atlas._context_offsets) != context_count + 1
                or len(atlas._branch_offsets) != context_count + 1
                or len(atlas._branch_tokens) != len(atlas._branch_counts)
                or atlas._context_offsets[0] != 0
                or atlas._context_offsets[-1] != len(atlas._context_tokens)
                or atlas._branch_offsets[0] != 0
                or atlas._branch_offsets[-1] != len(atlas._branch_tokens)
                or any(order > atlas.max_order for order in atlas._orders)
                or any(total < atlas.min_context_count for total in atlas._totals)
                or any(token >= atlas.vocab_size for token in atlas._context_tokens)
            ):
                raise ValueError("array topology mismatch")

            atlas._hash_index = {}
            for index in range(context_count):
                context_start = atlas._context_offsets[index]
                context_stop = atlas._context_offsets[index + 1]
                if context_stop - context_start != atlas._orders[index]:
                    raise ValueError("context offset mismatch")
                if context_stop < context_start:
                    raise ValueError("context offsets move backwards")
                branch_start = atlas._branch_offsets[index]
                branch_stop = atlas._branch_offsets[index + 1]
                branch_count = branch_stop - branch_start
                if (
                    branch_stop < branch_start
                    or not 1 <= branch_count <= atlas.max_branches
                ):
                    raise ValueError("branch offset mismatch")
                total = 0
                previous_count: int | None = None
                previous_token: int | None = None
                for offset in range(branch_start, branch_stop):
                    token = atlas._branch_tokens[offset]
                    count = atlas._branch_counts[offset]
                    if (
                        token >= atlas.vocab_size
                        or count <= 0
                        or (
                            previous_count is not None
                            and (
                                count > previous_count
                                or (
                                    count == previous_count
                                    and previous_token is not None
                                    and token <= previous_token
                                )
                            )
                        )
                    ):
                        raise ValueError("branch values are invalid")
                    total += count
                    previous_count = count
                    previous_token = token
                if total > atlas._totals[index]:
                    raise ValueError("branch counts exceed total")
                context = tuple(atlas._context_tokens[context_start:context_stop])
                fingerprint = _context_hash(context)
                existing = atlas._hash_index.get(fingerprint)
                existing_indexes = (
                    ()
                    if existing is None
                    else (existing,)
                    if isinstance(existing, int)
                    else existing
                )
                if any(
                    atlas._context_matches(candidate, context)
                    for candidate in existing_indexes
                ):
                    raise ValueError("duplicate array context")
                if existing is None:
                    atlas._hash_index[fingerprint] = index
                elif isinstance(existing, int):
                    atlas._hash_index[fingerprint] = (existing, index)
                else:
                    atlas._hash_index[fingerprint] = (*existing, index)
        except (IndexError, KeyError, TypeError, ValueError) as exc:
            raise MarkovAtlasError("Markov atlas array values are invalid") from exc
        return atlas

    @classmethod
    def _from_legacy_bytes(cls, raw: bytes) -> "MarkovTokenAtlas":
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise MarkovAtlasError("cannot decode legacy Markov atlas") from exc
        if (
            not isinstance(document, dict)
            or set(document) != {"body", "schema", "sha256"}
            or document.get("schema") != LEGACY_MARKOV_ATLAS_SCHEMA
            or not isinstance(document.get("body"), dict)
            or document.get("sha256") != _digest(document["body"])
            or _canonical(document) != raw
        ):
            raise MarkovAtlasError("legacy Markov atlas document is invalid")
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
            raise MarkovAtlasError("legacy Markov atlas body is invalid")
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
        except (TypeError, ValueError) as exc:
            raise MarkovAtlasError("legacy Markov atlas values are invalid") from exc
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
        orders = Counter(self._orders)
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
    "AtlasTokenEvidence",
    "LEGACY_MARKOV_ATLAS_PREFIX",
    "LEGACY_MARKOV_ATLAS_SCHEMA",
    "MARKOV_ATLAS_PREFIX",
    "MARKOV_ATLAS_SCHEMA",
    "MarkovAtlasError",
    "MarkovTokenAtlas",
    "tokenizer_file_sha256",
]
