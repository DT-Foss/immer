"""Sparse variable-order Markov drafting over native Qwen token IDs.

This is the runtime-native transfer of FERTIG's ``TransitionFingerprint``
equation: suffix counts, interpolation, and deterministic backoff operate on
fixed-width Qwen token symbols, while the target remains the sole authority.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Sequence
import zlib

MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v1"
MARKOV_DRAFT_METRICS_SCHEMA = "immer.qwen3.8-markov-draft-metrics/v1"
_STATE_PREFIX = b"IMMD\x01"
_MAX_STATE_BYTES = 16 * 1024 * 1024
_UNKNOWN_TOKEN = "<unknown>"


class MarkovDraftError(RuntimeError):
    pass


class _TransitionFingerprint:
    """Sparse interpolated variable-order counts over token-ID symbols."""

    def __init__(
        self,
        *,
        max_order: int,
        alpha: float,
        backoff_strength: float,
        min_count: int,
        vocabulary: tuple[str, ...],
        counts: Mapping[tuple[str, ...], Counter[str]],
    ) -> None:
        self.max_order = max_order
        self.alpha = alpha
        self.backoff_strength = backoff_strength
        self.min_count = min_count
        self.vocabulary = tuple(sorted(set(vocabulary) | {_UNKNOWN_TOKEN}))
        self.counts = dict(counts)

    @classmethod
    def fit(
        cls,
        sequence: Sequence[str],
        *,
        max_order: int,
        alpha: float,
        backoff_strength: float,
        min_count: int,
    ) -> "_TransitionFingerprint":
        tokens = tuple(sequence)
        if not tokens:
            raise ValueError("cannot fit an empty Markov token sequence")
        counts: dict[tuple[str, ...], Counter[str]] = {}
        for index, target in enumerate(tokens):
            for order in range(min(max_order, index) + 1):
                context = tokens[index - order : index] if order else ()
                counts.setdefault(context, Counter())[target] += 1
        return cls(
            max_order=max_order,
            alpha=alpha,
            backoff_strength=backoff_strength,
            min_count=min_count,
            vocabulary=tokens,
            counts=counts,
        )

    def distribution(self, context: Sequence[str]) -> dict[str, float]:
        root = self.counts[()]
        denominator = sum(root.values()) + self.alpha * len(self.vocabulary)
        probabilities = {
            token: (root.get(token, 0) + self.alpha) / denominator
            for token in self.vocabulary
        }
        known = tuple(
            token if token in self.vocabulary else _UNKNOWN_TOKEN
            for token in context
        )
        for order in range(1, min(self.max_order, len(known)) + 1):
            counter = self.counts.get(known[-order:])
            if not counter:
                continue
            total = sum(counter.values())
            if total < self.min_count:
                continue
            weight = total / (total + self.backoff_strength)
            for token in self.vocabulary:
                empirical = counter.get(token, 0) / total
                probabilities[token] = (
                    (1.0 - weight) * probabilities[token] + weight * empirical
                )
        return probabilities


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class MarkovDraftState:
    vocab_size: int
    max_history_tokens: int
    token_ids: tuple[int, ...]
    updates: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.vocab_size, bool)
            or not isinstance(self.vocab_size, int)
            or self.vocab_size <= 1
        ):
            raise ValueError("vocab_size must be greater than one")
        if (
            isinstance(self.max_history_tokens, bool)
            or not isinstance(self.max_history_tokens, int)
            or self.max_history_tokens < 8
        ):
            raise ValueError("max_history_tokens must be at least eight")
        if len(self.token_ids) > self.max_history_tokens or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocab_size
            for token in self.token_ids
        ):
            raise ValueError("Markov draft token history is invalid")
        if isinstance(self.updates, bool) or not isinstance(self.updates, int) or self.updates < 0:
            raise ValueError("updates must be a non-negative integer")

    def to_bytes(self) -> bytes:
        raw = _canonical(
            {
                "max_history_tokens": self.max_history_tokens,
                "schema": MARKOV_DRAFT_STATE_SCHEMA,
                "token_ids": list(self.token_ids),
                "updates": self.updates,
                "vocab_size": self.vocab_size,
            }
        )
        encoded = _STATE_PREFIX + zlib.compress(raw, level=9)
        if len(encoded) > _MAX_STATE_BYTES:
            raise MarkovDraftError("Markov draft state exceeds its byte bound")
        return encoded

    @classmethod
    def from_bytes(cls, data: bytes) -> "MarkovDraftState":
        if (
            not isinstance(data, bytes)
            or len(data) > _MAX_STATE_BYTES
            or not data.startswith(_STATE_PREFIX)
        ):
            raise MarkovDraftError("Markov draft state envelope is invalid")
        try:
            decoder = zlib.decompressobj()
            raw = decoder.decompress(
                data[len(_STATE_PREFIX) :], _MAX_STATE_BYTES + 1
            )
            if (
                len(raw) > _MAX_STATE_BYTES
                or decoder.unconsumed_tail
                or not decoder.eof
            ):
                raise zlib.error("expanded Markov state exceeds its bound")
            value = json.loads(raw)
        except (zlib.error, UnicodeError, json.JSONDecodeError) as exc:
            raise MarkovDraftError("Markov draft state is corrupt") from exc
        if (
            not isinstance(value, dict)
            or set(value)
            != {
                "max_history_tokens",
                "schema",
                "token_ids",
                "updates",
                "vocab_size",
            }
            or value.get("schema") != MARKOV_DRAFT_STATE_SCHEMA
            or not isinstance(value.get("token_ids"), list)
            or _canonical(value) != raw
        ):
            raise MarkovDraftError("Markov draft state schema is invalid")
        try:
            return cls(
                vocab_size=value["vocab_size"],
                max_history_tokens=value["max_history_tokens"],
                token_ids=tuple(value["token_ids"]),
                updates=value["updates"],
            )
        except (TypeError, ValueError) as exc:
            raise MarkovDraftError("Markov draft state values are invalid") from exc


@dataclass(frozen=True, slots=True)
class MarkovDraftMetrics:
    schema: str
    draft_calls: int
    reconcile_calls: int
    predictions: int
    learned_tokens: int
    updates: int
    state_bytes: int
    source_body_bytes: int = 0
    linear_calls: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FingerprintRollingK4DraftProvider:
    """Draft with sparse PPM counts and learn only target-confirmed tokens."""

    def __init__(
        self,
        *,
        vocab_size: int,
        state_path: str | Path | None = None,
        max_order: int = 8,
        alpha: float = 0.5,
        backoff_strength: float = 3.0,
        min_count: int = 1,
        max_history_tokens: int = 4096,
    ) -> None:
        if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 1:
            raise ValueError("vocab_size must be greater than one")
        if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 0:
            raise ValueError("max_order must be a non-negative integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(backoff_strength) or backoff_strength <= 0:
            raise ValueError("backoff_strength must be finite and positive")
        if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
            raise ValueError("min_count must be a positive integer")
        if (
            isinstance(max_history_tokens, bool)
            or not isinstance(max_history_tokens, int)
            or max_history_tokens < max(8, max_order + 1)
        ):
            raise ValueError("max_history_tokens is too small")
        if state_path is not None and not isinstance(state_path, (str, Path)):
            raise TypeError("state_path must be a filesystem path or None")
        self.vocab_size = vocab_size
        self.width = len(str(vocab_size - 1))
        self.state_path = (
            None if state_path is None else Path(state_path).expanduser().absolute()
        )
        self.max_order = max_order
        self.alpha = float(alpha)
        self.backoff_strength = float(backoff_strength)
        self.min_count = min_count
        self.max_history_tokens = max_history_tokens
        self._state = self._load_state()
        self._pending_base: tuple[int, ...] | None = None
        self._pending_proposal: tuple[int, ...] | None = None
        self._last_confirmed_length: int | None = None
        self._draft_calls = 0
        self._reconcile_calls = 0
        self._predictions = 0
        self._closed = False

    def _load_state(self) -> MarkovDraftState:
        empty = MarkovDraftState(
            vocab_size=self.vocab_size,
            max_history_tokens=self.max_history_tokens,
            token_ids=(),
        )
        path = self.state_path
        if path is None or (not path.exists() and not path.is_symlink()):
            return empty
        try:
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise MarkovDraftError("Markov draft state must be a regular file")
            if not 0 < metadata.st_size <= _MAX_STATE_BYTES:
                raise MarkovDraftError("Markov draft state size is invalid")
            state = MarkovDraftState.from_bytes(path.read_bytes())
        except OSError as exc:
            raise MarkovDraftError("cannot read Markov draft state") from exc
        if (
            state.vocab_size != self.vocab_size
            or state.max_history_tokens != self.max_history_tokens
        ):
            raise MarkovDraftError("Markov draft state configuration changed")
        return state

    def _token_tuple(self, value: object, *, label: str) -> tuple[int, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{label} must be an integer sequence")
        result = tuple(value)
        if not result or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocab_size
            for token in result
        ):
            raise ValueError(f"{label} contains invalid token IDs")
        return result

    def _symbol(self, token: int) -> str:
        return f"{token:0{self.width}d}"

    def _predict(self, history: tuple[int, ...], count: int) -> tuple[int, ...]:
        learned = self._state.token_ids
        combined = (*learned, *history)[-self.max_history_tokens :]
        symbols = tuple(self._symbol(token) for token in combined)
        fingerprint = _TransitionFingerprint.fit(
            symbols,
            max_order=min(self.max_order, len(symbols) - 1),
            alpha=self.alpha,
            backoff_strength=self.backoff_strength,
            min_count=self.min_count,
        )
        context = list(symbols)
        proposal: list[int] = []
        for _ in range(count):
            distribution = fingerprint.distribution(context)
            candidates = [
                (probability, -int(symbol), int(symbol))
                for symbol, probability in distribution.items()
                if symbol != _UNKNOWN_TOKEN
                and symbol.isdecimal()
                and 0 <= int(symbol) < self.vocab_size
            ]
            if not candidates:
                raise MarkovDraftError("Markov fingerprint has no Qwen token")
            token = max(candidates)[2]
            proposal.append(token)
            context.append(self._symbol(token))
        self._predictions += count
        return tuple(proposal)

    def __call__(self, history: tuple[int, ...], /) -> tuple[int, int, int, int]:
        committed = self._token_tuple(history, label="Markov draft history")
        proposal = self._predict(committed, 4)
        return proposal[0], proposal[1], proposal[2], proposal[3]

    def propose_after(
        self,
        history: tuple[int, ...],
        known_token: int,
        /,
    ) -> tuple[int, int, int]:
        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        committed = self._token_tuple(history, label="rolling Markov history")
        if isinstance(known_token, bool) or not isinstance(known_token, int) or not 0 <= known_token < self.vocab_size:
            raise ValueError("known token is outside the Qwen vocabulary")
        if self._pending_base is not None:
            raise MarkovDraftError("previous Markov proposal was not reconciled")
        if self._last_confirmed_length is None:
            self._last_confirmed_length = len(committed)
        elif len(committed) != self._last_confirmed_length:
            raise MarkovDraftError("rolling Markov history length is discontinuous")
        base = (*committed, known_token)
        proposal = self._predict(base, 3)
        self._pending_base = base
        self._pending_proposal = proposal
        self._draft_calls += 1
        return proposal[0], proposal[1], proposal[2]

    def _learn(self, tokens: Sequence[int]) -> None:
        if not tokens:
            return
        combined = (*self._state.token_ids, *tokens)[-self.max_history_tokens :]
        self._state = MarkovDraftState(
            vocab_size=self.vocab_size,
            max_history_tokens=self.max_history_tokens,
            token_ids=combined,
            updates=self._state.updates + 1,
        )

    def reconcile_prefix(self, history: tuple[int, ...], /) -> None:
        committed = self._token_tuple(history, label="reconciled Markov history")
        base = self._pending_base
        proposal = self._pending_proposal
        if base is None or proposal is None:
            raise MarkovDraftError("reconcile_prefix requires a pending proposal")
        if committed[: len(base)] != base:
            raise MarkovDraftError("Markov reconciliation changed its known base")
        delta = committed[len(base) :]
        if len(delta) > 3 or delta != proposal[: len(delta)]:
            raise MarkovDraftError("Markov reconciliation is not a proposal prefix")
        assert self._last_confirmed_length is not None
        self._learn(committed[self._last_confirmed_length :])
        self._last_confirmed_length = len(committed)
        self._pending_base = None
        self._pending_proposal = None
        self._reconcile_calls += 1
        self._persist()

    def observe_final(self, history: tuple[int, ...], /) -> None:
        committed = self._token_tuple(history, label="final Markov history")
        if self._pending_base is not None:
            raise MarkovDraftError("cannot finalize an unreconciled proposal")
        if self._last_confirmed_length is None:
            self._learn(committed)
            self._last_confirmed_length = len(committed)
            self._persist()
            return
        if len(committed) < self._last_confirmed_length:
            raise MarkovDraftError("final Markov history moved backwards")
        self._learn(committed[self._last_confirmed_length :])
        self._last_confirmed_length = len(committed)
        self._persist()

    def _persist(self) -> None:
        path = self.state_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self._state.to_bytes()
        temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_CREAT
                | os.O_EXCL
                | os.O_WRONLY
                | int(getattr(os, "O_NOFOLLOW", 0)),
                0o600,
            )
            offset = 0
            while offset < len(data):
                written = os.write(descriptor, data[offset:])
                if written <= 0:
                    raise OSError("short Markov draft state write")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, path)
        except OSError as exc:
            raise MarkovDraftError("cannot persist Markov draft state") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def metrics(self) -> MarkovDraftMetrics:
        return MarkovDraftMetrics(
            schema=MARKOV_DRAFT_METRICS_SCHEMA,
            draft_calls=self._draft_calls,
            reconcile_calls=self._reconcile_calls,
            predictions=self._predictions,
            learned_tokens=len(self._state.token_ids),
            updates=self._state.updates,
            state_bytes=len(self._state.to_bytes()),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._pending_base = None
        self._pending_proposal = None
        self._persist()
        self._closed = True


__all__ = [
    "MARKOV_DRAFT_METRICS_SCHEMA",
    "MARKOV_DRAFT_STATE_SCHEMA",
    "FingerprintRollingK4DraftProvider",
    "MarkovDraftError",
    "MarkovDraftMetrics",
    "MarkovDraftState",
]
