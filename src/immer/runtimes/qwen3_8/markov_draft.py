"""Sparse variable-order Markov drafting over native Qwen token IDs.

This is the runtime-native transfer of FERTIG's ``TransitionFingerprint``
equation: suffix counts, interpolation, and deterministic backoff operate on
fixed-width Qwen token symbols, while the target remains the sole authority.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Sequence
import zlib

from .draft_protocol import RollingDraftProposal
from .markov_composition import (
    CompositionBounds,
    ConfirmedTokenEpisode,
    MarkovCompositionProgram,
    derive_programs,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - production targets are POSIX.
    fcntl = None  # type: ignore[assignment]

MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v7"
MARKOV_DRAFT_PROVIDER_ABI = "immer.qwen3.8-markov-draft-provider/v14"
V6_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v6"
V5_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v5"
V4_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v4"
V3_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v3"
V2_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v2"
LEGACY_MARKOV_DRAFT_STATE_SCHEMA = "immer.qwen3.8-markov-draft-state/v1"
MARKOV_DRAFT_METRICS_SCHEMA = "immer.qwen3.8-markov-draft-metrics/v11"
_STATE_PREFIX = b"IMMD\x07"
_V6_STATE_PREFIX = b"IMMD\x06"
_V5_STATE_PREFIX = b"IMMD\x05"
_V4_STATE_PREFIX = b"IMMD\x04"
_V3_STATE_PREFIX = b"IMMD\x03"
_V2_STATE_PREFIX = b"IMMD\x02"
_LEGACY_STATE_PREFIX = b"IMMD\x01"
_MAX_STATE_BYTES = 16 * 1024 * 1024
_UNKNOWN_TOKEN = "<unknown>"
_EPISODE_TOKEN = "<episode>"
_HEX = frozenset("0123456789abcdef")
_MAX_IMPORTED_EPISODE_DIGESTS = 65_536
_MAX_PROPOSAL_POSITIONS = 16


class MarkovDraftError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class MarkovExpertSpec:
    name: str
    max_order: int
    window: int
    local_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Markov expert name must not be empty")
        if (
            isinstance(self.max_order, bool)
            or not isinstance(self.max_order, int)
            or self.max_order < 0
        ):
            raise ValueError("Markov expert order must be non-negative")
        if (
            isinstance(self.window, bool)
            or not isinstance(self.window, int)
            or self.window < max(8, self.max_order + 1)
        ):
            raise ValueError("Markov expert window is too small")
        if not isinstance(self.local_only, bool):
            raise TypeError("local_only must be boolean")


def _expert_specs(
    max_order: int, max_history_tokens: int
) -> tuple[MarkovExpertSpec, ...]:
    rows = (
        ("local-o0-w128", 0, min(128, max_history_tokens), True),
        ("global-o0-w4096", 0, max_history_tokens, False),
        ("local-o1-w128", min(1, max_order), min(128, max_history_tokens), True),
        ("global-o1-w4096", min(1, max_order), max_history_tokens, False),
        (
            "recent-o2-w256",
            min(2, max_order),
            min(256, max_history_tokens),
            False,
        ),
        (
            "medium-o4-w1024",
            min(4, max_order),
            min(1024, max_history_tokens),
            False,
        ),
        ("deep-o8-w4096", min(8, max_order), max_history_tokens, False),
        (f"max-o{max_order}-w4096", max_order, max_history_tokens, False),
    )
    return tuple(
        MarkovExpertSpec(name, order, max(8, window), local)
        for name, order, window, local in rows
    )


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
            token if token in self.vocabulary else _UNKNOWN_TOKEN for token in context
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
                probabilities[token] = (1.0 - weight) * probabilities[
                    token
                ] + weight * empirical
        return probabilities


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _read_state_bytes(path: Path) -> bytes:
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
            or not 0 < before.st_size <= _MAX_STATE_BYTES
        ):
            raise MarkovDraftError("Markov draft state size is invalid")
        chunks = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise MarkovDraftError("Markov draft state returned a short read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        linked = path.lstat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (after.st_dev, after.st_ino) != (linked.st_dev, linked.st_ino):
            raise MarkovDraftError("Markov draft state changed while read")
        return b"".join(chunks)
    except MarkovDraftError:
        raise
    except OSError as exc:
        raise MarkovDraftError("cannot read Markov draft state") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class MarkovDialectState:
    dialect_id: str
    signature: tuple[int, ...]
    visits: int
    last_seen: int
    rapidities: tuple[float, ...]
    observations: tuple[int, ...]
    hits: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.dialect_id, str)
            or len(self.dialect_id) != 64
            or set(self.dialect_id) - _HEX
        ):
            raise ValueError("dialect_id must be a SHA-256 digest")
        signature = tuple(self.signature)
        if (
            not 1 <= len(signature) <= 32
            or signature != tuple(sorted(set(signature)))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < 1 << 64
                for value in signature
            )
        ):
            raise ValueError("dialect signature is invalid")
        if (
            isinstance(self.visits, bool)
            or not isinstance(self.visits, int)
            or self.visits <= 0
            or isinstance(self.last_seen, bool)
            or not isinstance(self.last_seen, int)
            or self.last_seen < 0
        ):
            raise ValueError("dialect visit state is invalid")
        rapidities = tuple(float(value) for value in self.rapidities)
        observations = tuple(self.observations)
        hits = tuple(self.hits)
        if (
            not rapidities
            or len({len(rapidities), len(observations), len(hits)}) != 1
            or any(not math.isfinite(value) for value in rapidities)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (*observations, *hits)
            )
            or any(hit > seen for hit, seen in zip(hits, observations, strict=True))
        ):
            raise ValueError("dialect expert state is invalid")
        object.__setattr__(self, "signature", signature)
        object.__setattr__(self, "rapidities", rapidities)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "hits", hits)

    def to_record(self) -> dict[str, object]:
        return {
            "dialect_id": self.dialect_id,
            "hits": list(self.hits),
            "last_seen": self.last_seen,
            "observations": list(self.observations),
            "rapidities": [value.hex() for value in self.rapidities],
            "signature": [f"{value:016x}" for value in self.signature],
            "visits": self.visits,
        }

    @classmethod
    def from_record(cls, value: object) -> "MarkovDialectState":
        if not isinstance(value, Mapping) or set(value) != {
            "dialect_id",
            "hits",
            "last_seen",
            "observations",
            "rapidities",
            "signature",
            "visits",
        }:
            raise ValueError("dialect record is invalid")
        try:
            return cls(
                dialect_id=value["dialect_id"],
                signature=tuple(int(item, 16) for item in value["signature"]),
                visits=value["visits"],
                last_seen=value["last_seen"],
                rapidities=tuple(float.fromhex(item) for item in value["rapidities"]),
                observations=tuple(value["observations"]),
                hits=tuple(value["hits"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("dialect record values are invalid") from exc


@dataclass(frozen=True, slots=True)
class MarkovPhraseOption:
    token_ids: tuple[int, ...]
    source: str
    context_order: int
    support: int
    total: int
    kind: str = "literal"

    @property
    def confidence(self) -> float:
        return self.support / self.total

    def __post_init__(self) -> None:
        if self.source not in {"dialect", "global"}:
            raise ValueError("phrase option source is invalid")
        if self.kind not in {"composition", "literal"}:
            raise ValueError("phrase option kind is invalid")
        if (
            not 1 <= len(self.token_ids) <= 15
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in self.token_ids
            )
            or self.context_order < 1
            or self.support < 1
            or self.total < self.support
        ):
            raise ValueError("phrase option evidence is invalid")


@dataclass(frozen=True, slots=True)
class MarkovDraftState:
    vocab_size: int
    max_history_tokens: int
    token_ids: tuple[int, ...]
    updates: int = 0
    expert_names: tuple[str, ...] = ()
    expert_log_weights: tuple[float, ...] = ()
    expert_observations: tuple[int, ...] = ()
    expert_hits: tuple[int, ...] = ()
    horizon_expert_observations: tuple[tuple[int, ...], ...] = ()
    horizon_expert_hits: tuple[tuple[int, ...], ...] = ()
    leader_changes: int = 0
    episode_lengths: tuple[int, ...] = ()
    feedback_count: int = 0
    surprise_mean: float = 0.0
    surprise_deviation: float = 1.0
    surprise_cusum: float = 0.0
    regime_generation: int = 0
    clock: int = 0
    dialects: tuple[MarkovDialectState, ...] = ()
    episode_dialects: tuple[str | None, ...] = ()
    episode_prompt_lengths: tuple[int | None, ...] = ()
    imported_episode_sha256s: tuple[str, ...] = ()

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
        episode_lengths = tuple(self.episode_lengths)
        if self.token_ids and not episode_lengths:
            episode_lengths = (len(self.token_ids),)
        if any(
            isinstance(length, bool) or not isinstance(length, int) or length <= 0
            for length in episode_lengths
        ) or sum(episode_lengths) != len(self.token_ids):
            raise ValueError("Markov episode boundaries are invalid")
        episode_dialects = tuple(self.episode_dialects)
        if episode_lengths and not episode_dialects:
            episode_dialects = (None,) * len(episode_lengths)
        if len(episode_dialects) != len(episode_lengths) or any(
            value is not None
            and (not isinstance(value, str) or len(value) != 64 or set(value) - _HEX)
            for value in episode_dialects
        ):
            raise ValueError("Markov episode dialect bindings are invalid")
        episode_prompt_lengths = tuple(self.episode_prompt_lengths)
        if episode_lengths and not episode_prompt_lengths:
            episode_prompt_lengths = (None,) * len(episode_lengths)
        if len(episode_prompt_lengths) != len(episode_lengths) or any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value < episode_length
            )
            for value, episode_length in zip(
                episode_prompt_lengths,
                episode_lengths,
                strict=True,
            )
        ):
            raise ValueError("Markov episode prompt boundaries are invalid")
        if (
            isinstance(self.updates, bool)
            or not isinstance(self.updates, int)
            or self.updates < 0
        ):
            raise ValueError("updates must be a non-negative integer")
        names = tuple(self.expert_names)
        log_weights = tuple(float(value) for value in self.expert_log_weights)
        observations = tuple(self.expert_observations)
        hits = tuple(self.expert_hits)
        lengths = {len(names), len(log_weights), len(observations), len(hits)}
        if lengths not in ({0}, {len(names)}) or (
            names
            and (
                len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name for name in names)
                or any(not math.isfinite(value) for value in log_weights)
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 0
                    for value in (*observations, *hits)
                )
                or any(hit > seen for hit, seen in zip(hits, observations, strict=True))
            )
        ):
            raise ValueError("Markov expert state is invalid")
        horizon_observations = tuple(
            tuple(row) for row in self.horizon_expert_observations
        )
        horizon_hits = tuple(tuple(row) for row in self.horizon_expert_hits)
        if bool(horizon_observations) != bool(horizon_hits) or (
            horizon_observations
            and (
                len(horizon_observations) != _MAX_PROPOSAL_POSITIONS
                or len(horizon_hits) != _MAX_PROPOSAL_POSITIONS
                or any(
                    len(observed) != len(names) or len(hit) != len(names)
                    for observed, hit in zip(
                        horizon_observations,
                        horizon_hits,
                        strict=True,
                    )
                )
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for row in (*horizon_observations, *horizon_hits)
                    for value in row
                )
                or any(
                    hit > observed
                    for observed_row, hit_row in zip(
                        horizon_observations,
                        horizon_hits,
                        strict=True,
                    )
                    for observed, hit in zip(
                        observed_row,
                        hit_row,
                        strict=True,
                    )
                )
            )
        ):
            raise ValueError("Markov horizon expert state is invalid")
        if (
            isinstance(self.leader_changes, bool)
            or not isinstance(self.leader_changes, int)
            or self.leader_changes < 0
        ):
            raise ValueError("leader_changes must be non-negative")
        if (
            isinstance(self.feedback_count, bool)
            or not isinstance(self.feedback_count, int)
            or self.feedback_count < 0
            or isinstance(self.regime_generation, bool)
            or not isinstance(self.regime_generation, int)
            or self.regime_generation < 0
        ):
            raise ValueError("Markov regime counters are invalid")
        for value, label in (
            (self.surprise_mean, "surprise_mean"),
            (self.surprise_deviation, "surprise_deviation"),
            (self.surprise_cusum, "surprise_cusum"),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{label} must be finite and non-negative")
        if (
            isinstance(self.clock, bool)
            or not isinstance(self.clock, int)
            or self.clock < 0
        ):
            raise ValueError("clock must be a non-negative integer")
        imported = tuple(self.imported_episode_sha256s)
        if (
            len(imported) > _MAX_IMPORTED_EPISODE_DIGESTS
            or imported != tuple(sorted(set(imported)))
            or any(
                not isinstance(value, str) or len(value) != 64 or set(value) - _HEX
                for value in imported
            )
        ):
            raise ValueError("imported episode digest inventory is invalid")
        dialects = tuple(self.dialects)
        if (
            len(dialects) > 64
            or any(not isinstance(row, MarkovDialectState) for row in dialects)
            or tuple(sorted(dialects, key=lambda row: row.dialect_id)) != dialects
            or len({row.dialect_id for row in dialects}) != len(dialects)
            or any(
                len(row.rapidities) != len(names) or row.last_seen > self.clock
                for row in dialects
            )
        ):
            raise ValueError("Markov dialect inventory is invalid")
        object.__setattr__(self, "expert_names", names)
        object.__setattr__(self, "expert_log_weights", log_weights)
        object.__setattr__(self, "expert_observations", observations)
        object.__setattr__(self, "expert_hits", hits)
        object.__setattr__(
            self,
            "horizon_expert_observations",
            horizon_observations,
        )
        object.__setattr__(self, "horizon_expert_hits", horizon_hits)
        object.__setattr__(self, "episode_lengths", episode_lengths)
        object.__setattr__(self, "dialects", dialects)
        object.__setattr__(self, "episode_dialects", episode_dialects)
        object.__setattr__(
            self,
            "episode_prompt_lengths",
            episode_prompt_lengths,
        )
        object.__setattr__(self, "imported_episode_sha256s", imported)

    def to_bytes(self) -> bytes:
        raw = _canonical(
            {
                "max_history_tokens": self.max_history_tokens,
                "expert_hits": list(self.expert_hits),
                "expert_log_weights": [
                    value.hex() for value in self.expert_log_weights
                ],
                "expert_names": list(self.expert_names),
                "expert_observations": list(self.expert_observations),
                "horizon_expert_hits": [
                    list(row) for row in self.horizon_expert_hits
                ],
                "horizon_expert_observations": [
                    list(row) for row in self.horizon_expert_observations
                ],
                "leader_changes": self.leader_changes,
                "episode_lengths": list(self.episode_lengths),
                "episode_dialects": list(self.episode_dialects),
                "episode_prompt_lengths": list(self.episode_prompt_lengths),
                "imported_episode_sha256s": list(self.imported_episode_sha256s),
                "clock": self.clock,
                "dialects": [row.to_record() for row in self.dialects],
                "feedback_count": self.feedback_count,
                "regime_generation": self.regime_generation,
                "schema": MARKOV_DRAFT_STATE_SCHEMA,
                "surprise_cusum": self.surprise_cusum.hex(),
                "surprise_deviation": self.surprise_deviation.hex(),
                "surprise_mean": self.surprise_mean.hex(),
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
            or not data.startswith(
                (
                    _STATE_PREFIX,
                    _V6_STATE_PREFIX,
                    _V5_STATE_PREFIX,
                    _V4_STATE_PREFIX,
                    _V3_STATE_PREFIX,
                    _V2_STATE_PREFIX,
                    _LEGACY_STATE_PREFIX,
                )
            )
        ):
            raise MarkovDraftError("Markov draft state envelope is invalid")
        try:
            decoder = zlib.decompressobj()
            prefix = (
                _STATE_PREFIX
                if data.startswith(_STATE_PREFIX)
                else (
                    _V6_STATE_PREFIX
                    if data.startswith(_V6_STATE_PREFIX)
                    else (
                        _V5_STATE_PREFIX
                        if data.startswith(_V5_STATE_PREFIX)
                        else (
                            _V4_STATE_PREFIX
                            if data.startswith(_V4_STATE_PREFIX)
                            else (
                                _V2_STATE_PREFIX
                                if data.startswith(_V2_STATE_PREFIX)
                                else (
                                    _V3_STATE_PREFIX
                                    if data.startswith(_V3_STATE_PREFIX)
                                    else _LEGACY_STATE_PREFIX
                                )
                            )
                        )
                    )
                )
            )
            raw = decoder.decompress(data[len(prefix) :], _MAX_STATE_BYTES + 1)
            if (
                len(raw) > _MAX_STATE_BYTES
                or decoder.unconsumed_tail
                or not decoder.eof
            ):
                raise zlib.error("expanded Markov state exceeds its bound")
            value = json.loads(raw)
        except (zlib.error, UnicodeError, json.JSONDecodeError) as exc:
            raise MarkovDraftError("Markov draft state is corrupt") from exc
        legacy = (
            isinstance(value, dict)
            and value.get("schema") == LEGACY_MARKOV_DRAFT_STATE_SCHEMA
        )
        v2 = (
            isinstance(value, dict)
            and value.get("schema") == V2_MARKOV_DRAFT_STATE_SCHEMA
        )
        v3 = (
            isinstance(value, dict)
            and value.get("schema") == V3_MARKOV_DRAFT_STATE_SCHEMA
        )
        v4 = (
            isinstance(value, dict)
            and value.get("schema") == V4_MARKOV_DRAFT_STATE_SCHEMA
        )
        v5 = (
            isinstance(value, dict)
            and value.get("schema") == V5_MARKOV_DRAFT_STATE_SCHEMA
        )
        v6 = (
            isinstance(value, dict)
            and value.get("schema") == V6_MARKOV_DRAFT_STATE_SCHEMA
        )
        v2_fields = {
            "expert_hits",
            "expert_log_weights",
            "expert_names",
            "expert_observations",
            "leader_changes",
            "episode_lengths",
            "feedback_count",
            "max_history_tokens",
            "regime_generation",
            "schema",
            "surprise_cusum",
            "surprise_deviation",
            "surprise_mean",
            "token_ids",
            "updates",
            "vocab_size",
        }
        expected = (
            {
                "max_history_tokens",
                "schema",
                "token_ids",
                "updates",
                "vocab_size",
            }
            if legacy
            else v2_fields
            if v2
            else {
                "expert_hits",
                "expert_log_weights",
                "expert_names",
                "expert_observations",
                "leader_changes",
                "episode_lengths",
                "clock",
                "dialects",
                "feedback_count",
                "max_history_tokens",
                "regime_generation",
                "schema",
                "surprise_cusum",
                "surprise_deviation",
                "surprise_mean",
                "token_ids",
                "updates",
                "vocab_size",
            }
            if v3
            else {
                "expert_hits",
                "expert_log_weights",
                "expert_names",
                "expert_observations",
                "leader_changes",
                "episode_lengths",
                "episode_dialects",
                "clock",
                "dialects",
                "feedback_count",
                "max_history_tokens",
                "regime_generation",
                "schema",
                "surprise_cusum",
                "surprise_deviation",
                "surprise_mean",
                "token_ids",
                "updates",
                "vocab_size",
            }
            if v4
            else {
                "expert_hits",
                "expert_log_weights",
                "expert_names",
                "expert_observations",
                "leader_changes",
                "episode_lengths",
                "episode_dialects",
                "imported_episode_sha256s",
                "clock",
                "dialects",
                "feedback_count",
                "max_history_tokens",
                "regime_generation",
                "schema",
                "surprise_cusum",
                "surprise_deviation",
                "surprise_mean",
                "token_ids",
                "updates",
                "vocab_size",
            }
            if v5
            else {
                "expert_hits",
                "expert_log_weights",
                "expert_names",
                "expert_observations",
                "leader_changes",
                "episode_lengths",
                "episode_dialects",
                "episode_prompt_lengths",
                "imported_episode_sha256s",
                "clock",
                "dialects",
                "feedback_count",
                "max_history_tokens",
                "regime_generation",
                "schema",
                "surprise_cusum",
                "surprise_deviation",
                "surprise_mean",
                "token_ids",
                "updates",
                "vocab_size",
            }
            if v6
            else {
                "expert_hits",
                "expert_log_weights",
                "expert_names",
                "expert_observations",
                "horizon_expert_hits",
                "horizon_expert_observations",
                "leader_changes",
                "episode_lengths",
                "episode_dialects",
                "episode_prompt_lengths",
                "imported_episode_sha256s",
                "clock",
                "dialects",
                "feedback_count",
                "max_history_tokens",
                "regime_generation",
                "schema",
                "surprise_cusum",
                "surprise_deviation",
                "surprise_mean",
                "token_ids",
                "updates",
                "vocab_size",
            }
        )
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or value.get("schema")
            not in {
                MARKOV_DRAFT_STATE_SCHEMA,
                V6_MARKOV_DRAFT_STATE_SCHEMA,
                V5_MARKOV_DRAFT_STATE_SCHEMA,
                V4_MARKOV_DRAFT_STATE_SCHEMA,
                V3_MARKOV_DRAFT_STATE_SCHEMA,
                V2_MARKOV_DRAFT_STATE_SCHEMA,
                LEGACY_MARKOV_DRAFT_STATE_SCHEMA,
            }
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
                expert_names=tuple(value.get("expert_names", ())),
                expert_log_weights=tuple(
                    float.fromhex(item) for item in value.get("expert_log_weights", ())
                ),
                expert_observations=tuple(value.get("expert_observations", ())),
                expert_hits=tuple(value.get("expert_hits", ())),
                horizon_expert_observations=tuple(
                    tuple(row)
                    for row in value.get("horizon_expert_observations", ())
                ),
                horizon_expert_hits=tuple(
                    tuple(row) for row in value.get("horizon_expert_hits", ())
                ),
                leader_changes=value.get("leader_changes", 0),
                episode_lengths=tuple(value.get("episode_lengths", ())),
                episode_dialects=tuple(value.get("episode_dialects", ())),
                episode_prompt_lengths=tuple(value.get("episode_prompt_lengths", ())),
                imported_episode_sha256s=tuple(
                    value.get("imported_episode_sha256s", ())
                ),
                feedback_count=value.get("feedback_count", 0),
                surprise_mean=float.fromhex(value.get("surprise_mean", "0x0.0p+0")),
                surprise_deviation=float.fromhex(
                    value.get("surprise_deviation", "0x1.0p+0")
                ),
                surprise_cusum=float.fromhex(value.get("surprise_cusum", "0x0.0p+0")),
                regime_generation=value.get("regime_generation", 0),
                clock=value.get("clock", 0),
                dialects=tuple(
                    MarkovDialectState.from_record(row)
                    for row in value.get("dialects", ())
                ),
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
    episode_count: int
    imported_episode_count: int
    updates: int
    state_bytes: int
    council_predictions: int
    council_feedback: int
    external_reconcile_calls: int
    external_feedback_tokens: int
    teacher_forced_predictions: int
    teacher_forced_feedback_tokens: int
    teacher_forced_failures: int
    leader_changes: int
    expert_weights: tuple[tuple[str, float], ...]
    expert_accuracy: tuple[tuple[str, float], ...]
    effective_experts: float
    last_confidence: float
    last_raw_confidence: float
    last_empirical_evidence: float
    last_disagreement: float
    horizon_observations: tuple[int, ...]
    horizon_mean_accuracy: tuple[float, ...]
    regime_generation: int
    surprise_mean: float
    surprise_cusum: float
    dialect_count: int
    active_dialect_id: str | None
    active_dialect_similarity: float
    dialect_evictions: int
    phrase_option_calls: int
    phrase_draft_tokens: int
    phrase_accepted_tokens: int
    composition_programs: int
    composition_option_calls: int
    composition_draft_tokens: int
    composition_accepted_tokens: int
    last_composition_support: int
    last_composition_copy_tokens: int
    last_phrase_source: str | None
    last_phrase_support: int
    last_phrase_confidence: float
    last_phrase_width: int
    adaptive_proposal_calls: int
    recommended_windows: tuple[tuple[int, int], ...]
    last_recommended_window: int | None
    last_horizon_utilities: tuple[tuple[int, float], ...]
    source_body_bytes: int = 0
    linear_calls: int = 0
    proposal_width: int = 3

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["expert_weights"] = dict(self.expert_weights)
        value["expert_accuracy"] = dict(self.expert_accuracy)
        value["recommended_windows"] = dict(self.recommended_windows)
        value["last_horizon_utilities"] = dict(self.last_horizon_utilities)
        return value


class FingerprintRollingK4DraftProvider:
    """Draft with sparse PPM counts and learn only target-confirmed tokens."""

    EXPERT_LEARNING_RATE = 0.5
    RAPIDITY_DECAY = 0.95
    EXPERT_TEMPERATURE = 0.5
    FIXED_SHARE = 0.05
    SURPRISE_RATE = 0.05
    CUSUM_DECAY = 0.90
    CUSUM_DRIFT = 0.50
    CUSUM_THRESHOLD = 8.0
    REGIME_WARMUP = 16
    REGIME_RAPIDITY_SHRINK = 0.25
    MAX_DIALECTS = 64
    DIALECT_SKETCH_SIZE = 32
    DIALECT_SIMILARITY_THRESHOLD = 0.20
    DIALECT_STRENGTH = 2.0
    RICCI_AGE_ALPHA = 0.001
    PHRASE_MAX_CONTEXT = 8
    PHRASE_MAX_WIDTH = 15
    DIALECT_PHRASE_MIN_SUPPORT = 2
    GLOBAL_PHRASE_MIN_SUPPORT = 3
    COMPOSITION_BOUNDS = CompositionBounds()
    EMPIRICAL_EVIDENCE_SATURATION = 8.0

    def __init__(
        self,
        *,
        vocab_size: int,
        state_path: str | Path | None = None,
        max_order: int = 16,
        alpha: float = 0.5,
        backoff_strength: float = 3.0,
        min_count: int = 1,
        max_history_tokens: int = 4096,
        proposal_width: int = 3,
    ) -> None:
        if (
            isinstance(vocab_size, bool)
            or not isinstance(vocab_size, int)
            or vocab_size <= 1
        ):
            raise ValueError("vocab_size must be greater than one")
        if (
            isinstance(max_order, bool)
            or not isinstance(max_order, int)
            or max_order < 0
        ):
            raise ValueError("max_order must be a non-negative integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(backoff_strength) or backoff_strength <= 0:
            raise ValueError("backoff_strength must be finite and positive")
        if (
            isinstance(min_count, bool)
            or not isinstance(min_count, int)
            or min_count < 1
        ):
            raise ValueError("min_count must be a positive integer")
        if (
            isinstance(max_history_tokens, bool)
            or not isinstance(max_history_tokens, int)
            or max_history_tokens < max(8, max_order + 1)
        ):
            raise ValueError("max_history_tokens is too small")
        if state_path is not None and not isinstance(state_path, (str, Path)):
            raise TypeError("state_path must be a filesystem path or None")
        if (
            isinstance(proposal_width, bool)
            or not isinstance(proposal_width, int)
            or not 1 <= proposal_width <= 15
        ):
            raise ValueError("proposal_width must lie in [1, 15]")
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
        self.proposal_width = proposal_width
        self._experts = _expert_specs(max_order, max_history_tokens)
        self._state_lock_descriptor: int | None = None
        self._acquire_state_lock()
        try:
            self._state = self._load_state()
        except Exception:
            self._release_state_lock()
            raise
        names = tuple(row.name for row in self._experts)
        if not self._state.expert_names:
            uniform = -math.log(len(names))
            horizon_zeros = tuple(
                (0,) * len(names) for _ in range(_MAX_PROPOSAL_POSITIONS)
            )
            self._state = replace(
                self._state,
                expert_names=names,
                expert_log_weights=(uniform,) * len(names),
                expert_observations=(0,) * len(names),
                expert_hits=(0,) * len(names),
                horizon_expert_observations=horizon_zeros,
                horizon_expert_hits=horizon_zeros,
            )
        elif self._state.expert_names != names:
            self._release_state_lock()
            raise MarkovDraftError("Markov council topology changed")
        elif not self._state.horizon_expert_observations:
            horizon_zeros = tuple(
                (0,) * len(names) for _ in range(_MAX_PROPOSAL_POSITIONS)
            )
            self._state = replace(
                self._state,
                horizon_expert_observations=horizon_zeros,
                horizon_expert_hits=horizon_zeros,
            )
        self._pending_base: tuple[int, ...] | None = None
        self._pending_proposal: tuple[int, ...] | None = None
        self._pending_feedback: tuple[
            tuple[tuple[dict[str, float], int], ...], ...
        ] = ()
        self._carry_feedback: tuple[tuple[dict[str, float], int], ...] | None = None
        self._carry_feedback_position: int | None = None
        self._episode_feedback: list[
            tuple[tuple[tuple[dict[str, float], int], ...], int, int]
        ] = []
        self._active_dialect: MarkovDialectState | None = None
        self._active_dialect_is_new = False
        self._request_signature: tuple[int, ...] = ()
        self._active_dialect_similarity = 0.0
        self._dialect_evictions = 0
        self._request_started = False
        self._request_completed = False
        self._request_prompt: tuple[int, ...] | None = None
        self._request_prompt_length: int | None = None
        self._pending_phrase_option: MarkovPhraseOption | None = None
        self._phrase_option_calls = 0
        self._phrase_draft_tokens = 0
        self._phrase_accepted_tokens = 0
        self._last_phrase_option: MarkovPhraseOption | None = None
        self._composition_cache: dict[
            str | None,
            tuple[MarkovCompositionProgram, ...],
        ] = {}
        self._composition_program_count = 0
        self._composition_option_calls = 0
        self._composition_draft_tokens = 0
        self._composition_accepted_tokens = 0
        self._last_composition_program: MarkovCompositionProgram | None = None
        self._pending_composition_program: MarkovCompositionProgram | None = None
        self._last_confirmed_length: int | None = None
        self._draft_calls = 0
        self._reconcile_calls = 0
        self._predictions = 0
        self._council_predictions = 0
        self._council_feedback = 0
        self._external_reconcile_calls = 0
        self._external_feedback_tokens = 0
        self._teacher_forced_predictions = 0
        self._teacher_forced_feedback_tokens = 0
        self._teacher_forced_failures = 0
        self._last_confidence = 0.0
        self._last_raw_confidence = 0.0
        self._last_empirical_evidence = 0.0
        self._last_disagreement = 0.0
        self._adaptive_proposal_calls = 0
        self._recommended_window_counts = {1: 0, 4: 0, 8: 0, 16: 0}
        self._last_round_proposal: RollingDraftProposal | None = None
        self._pending_import_digest: str | None = None
        self._closed = False

    def _acquire_state_lock(self) -> None:
        path = self.state_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.parent / f".{path.name}.lock"
        descriptor = os.open(
            lock,
            os.O_CREAT
            | os.O_RDWR
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
            0o600,
        )
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_EX)
        except Exception:
            os.close(descriptor)
            raise
        self._state_lock_descriptor = descriptor

    def _release_state_lock(self) -> None:
        descriptor = self._state_lock_descriptor
        self._state_lock_descriptor = None
        if descriptor is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

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
            state = MarkovDraftState.from_bytes(_read_state_bytes(path))
        except OSError as exc:  # pragma: no cover - normalized by helper.
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

    def _context_signature(self, history: Sequence[int]) -> tuple[int, ...]:
        tokens = tuple(history)
        features: set[int] = set()
        for order in (1, 2, 3):
            for start in range(max(0, len(tokens) - 512), len(tokens) - order + 1):
                material = bytearray((order,))
                for token in tokens[start : start + order]:
                    material.extend(int(token).to_bytes(4, "little", signed=False))
                digest = hashlib.blake2b(
                    material,
                    digest_size=8,
                    person=b"IMMRDIA",
                ).digest()
                features.add(int.from_bytes(digest, "big"))
        length_bucket = min(255, len(tokens).bit_length())
        features.add(
            int.from_bytes(
                hashlib.blake2b(
                    bytes((255, length_bucket)),
                    digest_size=8,
                    person=b"IMMRDIA",
                ).digest(),
                "big",
            )
        )
        return tuple(sorted(features)[: self.DIALECT_SKETCH_SIZE])

    @staticmethod
    def _dialect_similarity(left: Sequence[int], right: Sequence[int]) -> float:
        first = set(left)
        second = set(right)
        union = first | second
        return 0.0 if not union else len(first & second) / len(union)

    def _activate_dialect(self, history: tuple[int, ...]) -> None:
        if self._active_dialect is not None:
            return
        signature = self._context_signature(history)
        self._request_signature = signature
        candidates = [
            (
                self._dialect_similarity(signature, row.signature),
                row.visits,
                row.dialect_id,
                row,
            )
            for row in self._state.dialects
        ]
        selected = max(candidates, default=None)
        if selected is not None and selected[0] >= self.DIALECT_SIMILARITY_THRESHOLD:
            self._active_dialect_similarity = selected[0]
            self._active_dialect = selected[3]
            self._active_dialect_is_new = False
            return
        dialect_id = hashlib.sha256(
            _canonical([f"{value:016x}" for value in signature])
        ).hexdigest()
        self._active_dialect = MarkovDialectState(
            dialect_id=dialect_id,
            signature=signature,
            visits=1,
            last_seen=self._state.clock + 1,
            rapidities=(0.0,) * len(self._experts),
            observations=(0,) * len(self._experts),
            hits=(0,) * len(self._experts),
        )
        self._active_dialect_similarity = 0.0
        self._active_dialect_is_new = True

    def _weights(self) -> tuple[float, ...]:
        rapidities = self._state.expert_log_weights
        if self._active_dialect is not None:
            rapidities = tuple(
                global_value
                + self.DIALECT_STRENGTH * self._active_dialect_similarity * local_value
                for global_value, local_value in zip(
                    rapidities,
                    self._active_dialect.rapidities,
                    strict=True,
                )
            )
        scaled = tuple(value / self.EXPERT_TEMPERATURE for value in rapidities)
        maximum = max(scaled)
        raw = tuple(math.exp(value - maximum) for value in scaled)
        total = sum(raw)
        count = len(raw)
        return tuple(
            (1.0 - self.FIXED_SHARE) * value / total + self.FIXED_SHARE / count
            for value in raw
        )

    def begin_request(self, history: tuple[int, ...], /) -> None:
        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        if self._request_started or self._request_completed:
            raise MarkovDraftError("Markov provider accepts exactly one request")
        committed = self._token_tuple(history, label="Markov request history")
        self._activate_dialect(committed)
        self._request_prompt = committed
        self._request_prompt_length = len(committed)
        self._request_started = True

    def _persistent_symbols(self) -> tuple[str, ...]:
        rows: list[str] = []
        for index, episode in enumerate(self._generation_episodes()):
            if index:
                rows.append(_EPISODE_TOKEN)
            rows.extend(self._symbol(token) for token in episode)
        return tuple(rows)

    def _episodes(self, dialect_id: str | None = None) -> tuple[tuple[int, ...], ...]:
        rows = []
        offset = 0
        for length, bound_dialect in zip(
            self._state.episode_lengths,
            self._state.episode_dialects,
            strict=True,
        ):
            episode = self._state.token_ids[offset : offset + length]
            offset += length
            if dialect_id is None or bound_dialect == dialect_id:
                rows.append(episode)
        return tuple(rows)

    def _generation_episodes(
        self,
        dialect_id: str | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Return the Qwen-generated side of every confirmed transition.

        The Council is a draft language model, so its persistent PPM corpus must
        contain Qwen's answers rather than repeated chat templates and arbitrary
        user prompts.  Legacy promptless episodes remain usable as answer-only
        rows; structured episodes start exactly at their stored prompt boundary.
        """

        rows = []
        offset = 0
        for length, bound_dialect, prompt_length in zip(
            self._state.episode_lengths,
            self._state.episode_dialects,
            self._state.episode_prompt_lengths,
            strict=True,
        ):
            episode = self._state.token_ids[offset : offset + length]
            offset += length
            if dialect_id is not None and bound_dialect != dialect_id:
                continue
            generated = episode if prompt_length is None else episode[prompt_length:]
            if generated:
                rows.append(generated)
        return tuple(rows)

    def _structured_episodes(
        self,
        dialect_id: str | None = None,
    ) -> tuple[ConfirmedTokenEpisode, ...]:
        rows = []
        offset = 0
        for length, bound_dialect, prompt_length in zip(
            self._state.episode_lengths,
            self._state.episode_dialects,
            self._state.episode_prompt_lengths,
            strict=True,
        ):
            episode = self._state.token_ids[offset : offset + length]
            offset += length
            if prompt_length is None or (
                dialect_id is not None and bound_dialect != dialect_id
            ):
                continue
            rows.append(
                ConfirmedTokenEpisode(
                    prompt=episode[:prompt_length],
                    output=episode[prompt_length:],
                )
            )
        return tuple(rows)

    def _composition_programs(
        self,
        dialect_id: str | None,
    ) -> tuple[MarkovCompositionProgram, ...]:
        if dialect_id not in self._composition_cache:
            self._composition_cache[dialect_id] = derive_programs(
                self._structured_episodes(dialect_id),
                self.COMPOSITION_BOUNDS,
            )
            self._composition_program_count = len(
                {
                    row.canonical_key
                    for programs in self._composition_cache.values()
                    for row in programs
                }
            )
        return self._composition_cache[dialect_id]

    def _composition_option_from(
        self,
        history: tuple[int, ...],
        *,
        source: str,
        dialect_id: str | None,
    ) -> tuple[MarkovPhraseOption, MarkovCompositionProgram] | None:
        prompt = self._request_prompt
        prompt_length = self._request_prompt_length
        if (
            prompt is None
            or prompt_length is None
            or history[:prompt_length] != prompt
            or len(history) <= prompt_length
        ):
            return None
        confirmed_output = history[prompt_length:]
        matches: list[tuple[tuple[int, ...], MarkovCompositionProgram]] = []
        for program in self._composition_programs(dialect_id):
            continuation = program.match(
                prompt,
                confirmed_output,
                limit=self.proposal_width,
            )
            if continuation:
                matches.append((continuation, program))
        continuations = {row[0] for row in matches}
        if len(continuations) != 1:
            return None
        continuation = continuations.pop()
        program = max(
            (row[1] for row in matches if row[0] == continuation),
            key=lambda row: (
                row.support,
                row.distinct_bindings,
                row.copied_tokens,
                row.context_width,
                tuple(-len(atom.canonical_key) for atom in row.atoms),
                row.canonical_key,
            ),
        )
        return (
            MarkovPhraseOption(
                token_ids=continuation,
                source=source,
                context_order=program.context_width,
                support=program.support,
                total=program.total,
                kind="composition",
            ),
            program,
        )

    def confirmed_episodes(self) -> tuple[tuple[int, ...], ...]:
        """Return immutable target-confirmed episodes for idempotent import."""

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        return self._episodes()

    def confirmed_transitions(
        self,
    ) -> tuple[tuple[tuple[int, ...] | None, tuple[int, ...]], ...]:
        """Return retained prompt/output splits, including unknown legacy ones."""

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        rows = []
        offset = 0
        for length, prompt_length in zip(
            self._state.episode_lengths,
            self._state.episode_prompt_lengths,
            strict=True,
        ):
            episode = self._state.token_ids[offset : offset + length]
            offset += length
            rows.append(
                (
                    None if prompt_length is None else episode[:prompt_length],
                    episode if prompt_length is None else episode[prompt_length:],
                )
            )
        return tuple(rows)

    @staticmethod
    def _import_digest(value: object) -> str:
        if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
            raise ValueError("import digest must be a SHA-256 value")
        return value

    def imported_episode_sha256s(self) -> tuple[str, ...]:
        """Return durable receipt identities retained beyond token eviction."""

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        return self._state.imported_episode_sha256s

    def register_imported_episode_sha256s(self, values: Sequence[str], /) -> int:
        """Atomically migrate legacy import identities into provider state."""

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        if self._request_started or self._request_completed:
            raise MarkovDraftError("cannot migrate imports during a request")
        digests = tuple(self._import_digest(value) for value in values)
        merged = tuple(sorted(set(self._state.imported_episode_sha256s) | set(digests)))
        added = len(merged) - len(self._state.imported_episode_sha256s)
        if not added:
            return 0
        original = self._state
        try:
            self._state = replace(
                self._state,
                imported_episode_sha256s=merged,
            )
            self._persist()
        except Exception:
            self._state = original
            raise
        return added

    def import_confirmed_episode(
        self,
        history: tuple[int, ...],
        receipt_sha256: str,
        /,
    ) -> bool:
        """Learn one receipt exactly once in the same atomic state commit."""

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        digest = self._import_digest(receipt_sha256)
        if digest in self._state.imported_episode_sha256s:
            return False
        if self._pending_import_digest is not None:
            raise MarkovDraftError("another receipt import is pending")
        self._pending_import_digest = digest
        try:
            self.observe_final(history)
        finally:
            self._pending_import_digest = None
        return True

    def import_confirmed_transition(
        self,
        prompt: tuple[int, ...],
        generated: tuple[int, ...],
        receipt_sha256: str,
        /,
    ) -> bool:
        """Atomically learn one receipt with its prompt/output boundary."""

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        prompt_tokens = self._token_tuple(prompt, label="imported prompt")
        generated_tokens = self._token_tuple(
            generated,
            label="imported generation",
        )
        digest = self._import_digest(receipt_sha256)
        if digest in self._state.imported_episode_sha256s:
            return False
        if self._pending_import_digest is not None:
            raise MarkovDraftError("another receipt import is pending")
        self._pending_import_digest = digest
        try:
            self.begin_request(prompt_tokens)
            self.observe_final((*prompt_tokens, *generated_tokens))
        finally:
            self._pending_import_digest = None
        return True

    def _phrase_option_from(
        self,
        history: tuple[int, ...],
        *,
        source: str,
        dialect_id: str | None,
        minimum_support: int,
    ) -> MarkovPhraseOption | None:
        episodes = self._generation_episodes(dialect_id)
        candidates: list[MarkovPhraseOption] = []
        max_width = min(self.proposal_width, self.PHRASE_MAX_WIDTH)
        if max_width < 2:
            return None
        for order in range(min(self.PHRASE_MAX_CONTEXT, len(history)), 0, -1):
            suffix = history[-order:]
            continuations: list[tuple[int, ...]] = []
            for episode in episodes:
                for position in range(order, len(episode) - 1):
                    if episode[position - order : position] == suffix:
                        continuation = episode[position : position + max_width]
                        if len(continuation) >= 2:
                            continuations.append(continuation)
            if not continuations:
                continue
            for width in range(2, max_width + 1):
                phrases = Counter(
                    continuation[:width]
                    for continuation in continuations
                    if len(continuation) >= width
                )
                if not phrases:
                    break
                phrase, support = max(
                    phrases.items(),
                    key=lambda row: (
                        row[1],
                        tuple(-token for token in row[0]),
                    ),
                )
                if support >= minimum_support:
                    candidates.append(
                        MarkovPhraseOption(
                            token_ids=phrase,
                            source=source,
                            context_order=order,
                            support=support,
                            total=sum(phrases.values()),
                        )
                    )
        if not candidates:
            return None
        return max(candidates, key=self._phrase_option_score)

    def _phrase_option_score(
        self,
        option: MarkovPhraseOption,
    ) -> tuple[float, int, int, int, tuple[int, ...], str]:
        scope = self._active_dialect_similarity if option.source == "dialect" else 1.0
        quality = (
            option.confidence
            * math.log1p(option.support)
            * math.sqrt(len(option.token_ids))
            * (1.0 + option.context_order / self.PHRASE_MAX_CONTEXT)
            * scope
        )
        return (
            quality,
            option.context_order,
            option.support,
            len(option.token_ids),
            tuple(-token for token in option.token_ids),
            option.source,
        )

    def _phrase_option(self, history: tuple[int, ...]) -> MarkovPhraseOption | None:
        dialect = self._active_dialect
        candidates: list[MarkovPhraseOption] = []
        composition_rows: list[tuple[MarkovPhraseOption, MarkovCompositionProgram]] = []
        if dialect is not None and not self._active_dialect_is_new:
            local = self._phrase_option_from(
                history,
                source="dialect",
                dialect_id=dialect.dialect_id,
                minimum_support=self.DIALECT_PHRASE_MIN_SUPPORT,
            )
            if local is not None:
                candidates.append(local)
            local_composition = self._composition_option_from(
                history,
                source="dialect",
                dialect_id=dialect.dialect_id,
            )
            if local_composition is not None:
                composition_rows.append(local_composition)
        global_option = self._phrase_option_from(
            history,
            source="global",
            dialect_id=None,
            minimum_support=self.GLOBAL_PHRASE_MIN_SUPPORT,
        )
        if global_option is not None:
            candidates.append(global_option)
        global_composition = self._composition_option_from(
            history,
            source="global",
            dialect_id=None,
        )
        if global_composition is not None:
            composition_rows.append(global_composition)
        composition_outputs = {row[0].token_ids for row in composition_rows}
        if len(composition_outputs) == 1:
            candidates.extend(row[0] for row in composition_rows)
        if not candidates:
            self._pending_composition_program = None
            return None
        selected = max(candidates, key=self._phrase_option_score)
        self._pending_composition_program = next(
            (program for option, program in composition_rows if option == selected),
            None,
        )
        return selected

    def _expert_models(
        self, history: tuple[int, ...]
    ) -> tuple[tuple[_TransitionFingerprint, list[str]], ...]:
        rows = []
        persistent = self._persistent_symbols()
        current = tuple(self._symbol(token) for token in history)
        for spec in self._experts:
            source = (
                current
                if spec.local_only or not persistent
                else (*persistent, _EPISODE_TOKEN, *current)
            )
            selected = source[-spec.window :]
            model = _TransitionFingerprint.fit(
                selected,
                max_order=min(spec.max_order, len(selected) - 1),
                alpha=self.alpha,
                backoff_strength=self.backoff_strength,
                min_count=self.min_count,
            )
            rows.append((model, list(selected)))
        return tuple(rows)

    def _calibrated_confidence(
        self,
        token: int,
        raw_confidence: float,
        expert_row: Sequence[tuple[Mapping[str, float], int]],
        weights: Sequence[float],
        position: int,
        *,
        allow_empirical: bool = True,
    ) -> tuple[float, float]:
        """Fuse PPM mass with target-confirmed Fixed-Share expert evidence.

        Each agreeing expert contributes its Beta(1,1) posterior correctness,
        discounted by ``n / (n + tau)`` and by its current normalized Top-1
        margin ``(p1-p2)/(p1+p2)``.  An unobserved or undecided Council therefore
        cannot create confidence from its prior alone.  The weighted evidence
        is joined with the current PPM probability by the complement product
        ``1 - (1-p_raw)(1-e)``.  Target disagreement remains a separate prefix
        penalty in :class:`RollingDraftProposal`.  Forced phrase tokens disable
        this path because phrase support is already scored independently.
        """

        if len(expert_row) != len(self._experts) or len(weights) != len(self._experts):
            raise MarkovDraftError("Markov empirical confidence width changed")
        if not 0 <= position < _MAX_PROPOSAL_POSITIONS:
            raise MarkovDraftError("Markov confidence position is invalid")
        evidence = 0.0
        token_symbol = self._symbol(token)
        for index, ((_distribution, predicted), weight) in enumerate(
            zip(expert_row, weights, strict=True)
        ):
            if not allow_empirical or predicted != token:
                continue
            token_mass = float(_distribution.get(token_symbol, 0.0))
            runner_up = max(
                (
                    float(probability)
                    for symbol, probability in _distribution.items()
                    if symbol != token_symbol
                    and (
                        symbol == _UNKNOWN_TOKEN
                        or (
                            symbol.isdecimal()
                            and 0 <= int(symbol) < self.vocab_size
                        )
                    )
                ),
                default=0.0,
            )
            denominator = token_mass + runner_up
            decisiveness = (
                1.0
                if denominator <= 0.0 and token_mass > 0.0
                else 0.0
                if denominator <= 0.0
                else max(0.0, min(1.0, (token_mass - runner_up) / denominator))
            )
            if decisiveness <= 0.0:
                continue
            observations = self._state.horizon_expert_observations[position][index]
            hits = self._state.horizon_expert_hits[position][index]
            if observations <= 0:
                observations = self._state.expert_observations[index]
                hits = self._state.expert_hits[index]
            if observations <= 0:
                continue
            posterior = (hits + 1.0) / (observations + 2.0)
            maturity = observations / (
                observations + self.EMPIRICAL_EVIDENCE_SATURATION
            )
            evidence += float(weight) * maturity * posterior * decisiveness
        evidence = max(0.0, min(0.999, evidence))
        raw = max(0.0, min(0.999, float(raw_confidence)))
        calibrated = 1.0 - (1.0 - raw) * (1.0 - evidence)
        return max(raw, min(0.999, calibrated)), evidence

    def _predict_council(
        self,
        history: tuple[int, ...],
        count: int,
        *,
        forced_prefix: Sequence[int] = (),
        position_offset: int = 0,
    ) -> tuple[
        tuple[int, ...],
        tuple[tuple[tuple[dict[str, float], int], ...], ...],
        tuple[float, ...],
        tuple[float, ...],
    ]:
        experts = self._expert_models(history)
        weights = self._weights()
        proposal: list[int] = []
        feedback_rows = []
        confidences = []
        disagreements = []
        forced = tuple(forced_prefix)
        if (
            isinstance(position_offset, bool)
            or not isinstance(position_offset, int)
            or position_offset < 0
            or position_offset + count > _MAX_PROPOSAL_POSITIONS
        ):
            raise ValueError("Council position range is invalid")
        if len(forced) > count or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or not 0 <= token < self.vocab_size
            for token in forced
        ):
            raise ValueError("forced Council prefix is invalid")
        for position in range(count):
            horizon_position = position_offset + position
            distributions = tuple(
                model.distribution(context) for model, context in experts
            )
            numeric_symbols = sorted(
                {
                    symbol
                    for distribution in distributions
                    for symbol in distribution
                    if symbol != _UNKNOWN_TOKEN
                    and symbol.isdecimal()
                    and 0 <= int(symbol) < self.vocab_size
                }
            )
            if not numeric_symbols:
                raise MarkovDraftError("Markov council has no Qwen token")
            mixture = {
                symbol: sum(
                    weight * distribution.get(symbol, 0.0)
                    for weight, distribution in zip(weights, distributions, strict=True)
                )
                for symbol in numeric_symbols
            }
            token = (
                forced[position]
                if position < len(forced)
                else max(
                    (probability, -int(symbol), int(symbol))
                    for symbol, probability in mixture.items()
                )[2]
            )
            proposal.append(token)
            universe = set().union(
                *(distribution.keys() for distribution in distributions)
            )
            pooled = {
                symbol: sum(
                    weight * distribution.get(symbol, 0.0)
                    for weight, distribution in zip(weights, distributions, strict=True)
                )
                for symbol in universe
            }

            def entropy(probabilities: Mapping[str, float]) -> float:
                return -sum(
                    value * math.log(max(value, 1e-12))
                    for value in probabilities.values()
                    if value > 0.0
                )

            self._last_disagreement = max(
                0.0,
                entropy(pooled)
                - sum(
                    weight * entropy(distribution)
                    for weight, distribution in zip(weights, distributions, strict=True)
                ),
            )
            expert_row = []
            for distribution in distributions:
                predicted = max(
                    (
                        probability,
                        -int(symbol),
                        int(symbol),
                    )
                    for symbol, probability in distribution.items()
                    if symbol != _UNKNOWN_TOKEN
                    and symbol.isdecimal()
                    and 0 <= int(symbol) < self.vocab_size
                )[2]
                expert_row.append((distribution, predicted))
            self._last_raw_confidence = mixture.get(self._symbol(token), 0.0)
            (
                self._last_confidence,
                self._last_empirical_evidence,
            ) = self._calibrated_confidence(
                token,
                self._last_raw_confidence,
                expert_row,
                weights,
                horizon_position,
                allow_empirical=position >= len(forced),
            )
            confidences.append(self._last_confidence)
            disagreements.append(self._last_disagreement)
            feedback_rows.append(tuple(expert_row))
            symbol = self._symbol(token)
            for _model, context in experts:
                context.append(symbol)
        self._predictions += count
        self._council_predictions += count
        return (
            tuple(proposal),
            tuple(feedback_rows),
            tuple(confidences),
            tuple(disagreements),
        )

    def _teacher_forced_prediction(
        self,
        history: tuple[int, ...],
        *,
        position: int,
    ) -> tuple[tuple[tuple[dict[str, float], int], ...], int]:
        """Predict one retrospective row without changing served-draft metrics."""

        snapshot = (
            self._predictions,
            self._council_predictions,
            self._last_raw_confidence,
            self._last_empirical_evidence,
            self._last_confidence,
            self._last_disagreement,
        )
        try:
            tokens, feedback, _confidence, _disagreement = self._predict_council(
                history,
                1,
                position_offset=position,
            )
            return feedback[0], tokens[0]
        finally:
            (
                self._predictions,
                self._council_predictions,
                self._last_raw_confidence,
                self._last_empirical_evidence,
                self._last_confidence,
                self._last_disagreement,
            ) = snapshot

    def _apply_council_feedback(
        self,
        feedback: tuple[tuple[dict[str, float], int], ...],
        token: int,
        position: int = 0,
    ) -> None:
        if len(feedback) != len(self._experts):
            raise MarkovDraftError("Markov council feedback width changed")
        if not 0 <= position < _MAX_PROPOSAL_POSITIONS:
            raise MarkovDraftError("Markov feedback position is invalid")
        symbol = self._symbol(token)
        before_leader = max(
            range(len(self._experts)),
            key=lambda index: self._state.expert_log_weights[index],
        )
        logs = list(self._state.expert_log_weights)
        observations = list(self._state.expert_observations)
        hits = list(self._state.expert_hits)
        horizon_observations = [
            list(row) for row in self._state.horizon_expert_observations
        ]
        horizon_hits = [list(row) for row in self._state.horizon_expert_hits]
        weights = self._weights()
        probabilities = []
        for distribution, _prediction in feedback:
            if symbol in distribution:
                probabilities.append(distribution[symbol])
                continue
            seen = sum(
                candidate.isdecimal() and 0 <= int(candidate) < self.vocab_size
                for candidate in distribution
            )
            probabilities.append(
                distribution.get(_UNKNOWN_TOKEN, 1e-12) / max(1, self.vocab_size - seen)
            )
        mixture_probability = sum(
            weight * probability
            for weight, probability in zip(weights, probabilities, strict=True)
        )
        surprise = -math.log(max(mixture_probability, 1e-12))
        previous_mean = self._state.surprise_mean
        previous_deviation = self._state.surprise_deviation
        if self._state.feedback_count == 0:
            z_score = 0.0
        else:
            z_score = (surprise - previous_mean) / max(previous_deviation, 1e-6)
        next_mean = (
            1.0 - self.SURPRISE_RATE
        ) * previous_mean + self.SURPRISE_RATE * surprise
        next_deviation = (
            1.0 - self.SURPRISE_RATE
        ) * previous_deviation + self.SURPRISE_RATE * abs(surprise - previous_mean)
        next_cusum = max(
            0.0,
            self.CUSUM_DECAY * self._state.surprise_cusum + z_score - self.CUSUM_DRIFT,
        )
        for index, (distribution, prediction) in enumerate(feedback):
            probability = probabilities[index]
            advantage = math.log(max(probability, 1e-12)) - math.log(
                max(mixture_probability, 1e-12)
            )
            logs[index] = (
                self.RAPIDITY_DECAY * logs[index]
                + self.EXPERT_LEARNING_RATE * advantage
            )
            observations[index] += 1
            hits[index] += int(prediction == token)
            horizon_observations[position][index] += 1
            horizon_hits[position][index] += int(prediction == token)
        center = sum(logs) / len(logs)
        logs = [value - center for value in logs]
        regime_change = (
            self._state.feedback_count + 1 >= self.REGIME_WARMUP
            and next_cusum > self.CUSUM_THRESHOLD
        )
        if regime_change:
            logs = [self.REGIME_RAPIDITY_SHRINK * value for value in logs]
            next_cusum = 0.0
        dialect = self._active_dialect
        if dialect is not None:
            local_logs = list(dialect.rapidities)
            local_observations = list(dialect.observations)
            local_hits = list(dialect.hits)
            for index, (_distribution, prediction) in enumerate(feedback):
                advantage = math.log(max(probabilities[index], 1e-12)) - math.log(
                    max(mixture_probability, 1e-12)
                )
                local_logs[index] = (
                    self.RAPIDITY_DECAY * local_logs[index]
                    + self.EXPERT_LEARNING_RATE * advantage
                )
                local_observations[index] += 1
                local_hits[index] += int(prediction == token)
            local_center = sum(local_logs) / len(local_logs)
            local_logs = [value - local_center for value in local_logs]
            if regime_change:
                local_logs = [
                    self.REGIME_RAPIDITY_SHRINK * value for value in local_logs
                ]
            self._active_dialect = replace(
                dialect,
                rapidities=tuple(local_logs),
                observations=tuple(local_observations),
                hits=tuple(local_hits),
            )
        after_leader = max(range(len(logs)), key=logs.__getitem__)
        self._state = replace(
            self._state,
            expert_log_weights=tuple(logs),
            expert_observations=tuple(observations),
            expert_hits=tuple(hits),
            horizon_expert_observations=tuple(
                tuple(row) for row in horizon_observations
            ),
            horizon_expert_hits=tuple(tuple(row) for row in horizon_hits),
            leader_changes=(
                self._state.leader_changes + int(before_leader != after_leader)
            ),
            feedback_count=self._state.feedback_count + 1,
            surprise_mean=next_mean,
            surprise_deviation=next_deviation,
            surprise_cusum=next_cusum,
            regime_generation=(self._state.regime_generation + int(regime_change)),
        )
        self._council_feedback += 1

    def __call__(self, history: tuple[int, ...], /) -> tuple[int, int, int, int]:
        committed = self._token_tuple(history, label="Markov draft history")
        proposal, _feedback, _confidence, _disagreement = self._predict_council(
            committed,
            4,
        )
        return proposal[0], proposal[1], proposal[2], proposal[3]

    def _prepare_rolling_proposal(
        self,
        history: tuple[int, ...],
        known_token: int,
    ) -> tuple[
        tuple[int, ...],
        tuple[float, ...],
        tuple[float, ...],
        MarkovPhraseOption | None,
    ]:
        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        if self._request_completed:
            raise MarkovDraftError("Markov provider request is complete")
        committed = self._token_tuple(history, label="rolling Markov history")
        if (
            isinstance(known_token, bool)
            or not isinstance(known_token, int)
            or not 0 <= known_token < self.vocab_size
        ):
            raise ValueError("known token is outside the Qwen vocabulary")
        if self._pending_base is not None:
            raise MarkovDraftError("previous Markov proposal was not reconciled")
        if self._last_confirmed_length is None:
            if not self._request_started:
                self._activate_dialect(committed)
                self._request_started = True
            self._last_confirmed_length = len(committed)
        elif len(committed) != self._last_confirmed_length:
            raise MarkovDraftError("rolling Markov history length is discontinuous")
        if self._carry_feedback is not None:
            if self._carry_feedback_position is None:
                raise MarkovDraftError("Markov carry feedback position is missing")
            self._episode_feedback.append(
                (
                    self._carry_feedback,
                    known_token,
                    self._carry_feedback_position,
                )
            )
            self._carry_feedback = None
            self._carry_feedback_position = None
        base = (*committed, known_token)
        option = self._phrase_option(base)
        complete, feedback, confidences, disagreements = self._predict_council(
            base,
            self.proposal_width + 1,
            forced_prefix=(
                () if option is None else option.token_ids[: self.proposal_width]
            ),
        )
        proposal = complete[: self.proposal_width]
        self._pending_base = base
        self._pending_proposal = proposal
        self._pending_feedback = feedback
        self._pending_phrase_option = option
        if option is not None:
            self._phrase_option_calls += 1
            self._phrase_draft_tokens += min(
                len(option.token_ids),
                self.proposal_width,
            )
            self._last_phrase_option = option
            if option.kind == "composition":
                self._composition_option_calls += 1
                self._composition_draft_tokens += min(
                    len(option.token_ids),
                    self.proposal_width,
                )
                self._last_composition_program = self._pending_composition_program
        self._draft_calls += 1
        return (
            proposal,
            confidences[: self.proposal_width],
            disagreements[: self.proposal_width],
            option,
        )

    def propose_after(
        self,
        history: tuple[int, ...],
        known_token: int,
        /,
    ) -> tuple[int, ...]:
        proposal, _confidence, _disagreement, _option = self._prepare_rolling_proposal(
            history, known_token
        )
        return proposal

    def propose_round(
        self,
        history: tuple[int, ...],
        known_token: int,
        /,
    ) -> RollingDraftProposal:
        if self.proposal_width + 1 not in {4, 8, 16}:
            raise MarkovDraftError(
                "adaptive Markov proposal requires a K4/K8/K16 ceiling"
            )
        proposal, confidences, disagreements, option = self._prepare_rolling_proposal(
            history, known_token
        )
        result = RollingDraftProposal.build(
            proposal,
            confidences,
            disagreements,
            request_window_ceiling=self.proposal_width + 1,
            provider_abi=MARKOV_DRAFT_PROVIDER_ABI,
            phrase_source=None if option is None else option.source,
            phrase_support=0 if option is None else option.support,
            phrase_confidence=(0.0 if option is None else option.confidence),
            phrase_width=(
                0 if option is None else min(len(option.token_ids), self.proposal_width)
            ),
        )
        self._adaptive_proposal_calls += 1
        self._recommended_window_counts[result.recommended_window] += 1
        self._last_round_proposal = result
        return result

    def _learn_episode(
        self,
        tokens: Sequence[int],
        *,
        prompt_length: int | None,
    ) -> None:
        episode = tuple(tokens)
        if not episode:
            return
        if prompt_length is not None and (
            isinstance(prompt_length, bool)
            or not isinstance(prompt_length, int)
            or not 1 <= prompt_length < len(episode)
        ):
            raise MarkovDraftError("confirmed episode prompt boundary is invalid")
        if self._active_dialect is None:
            raise MarkovDraftError("confirmed episode has no dialect binding")
        episodes = []
        dialect_ids = list(self._state.episode_dialects)
        prompt_lengths = list(self._state.episode_prompt_lengths)
        offset = 0
        for length in self._state.episode_lengths:
            episodes.append(self._state.token_ids[offset : offset + length])
            offset += length
        episodes.append(episode)
        dialect_ids.append(self._active_dialect.dialect_id)
        prompt_lengths.append(prompt_length)
        total = sum(len(row) for row in episodes)
        while len(episodes) > 1 and total > self.max_history_tokens:
            total -= len(episodes.pop(0))
            dialect_ids.pop(0)
            prompt_lengths.pop(0)
        if total > self.max_history_tokens:
            episodes[0] = episodes[0][-self.max_history_tokens :]
            prompt_lengths[0] = None
        combined = tuple(token for row in episodes for token in row)
        self._state = replace(
            self._state,
            token_ids=combined,
            episode_lengths=tuple(len(row) for row in episodes),
            episode_dialects=tuple(dialect_ids),
            episode_prompt_lengths=tuple(prompt_lengths),
            updates=self._state.updates + 1,
        )
        self._composition_cache.clear()

    def _commit_active_dialect(self) -> None:
        dialect = self._active_dialect
        if dialect is None:
            raise MarkovDraftError("completed request has no active dialect")
        clock = self._state.clock + 1
        signature = tuple(
            sorted(set(dialect.signature) | set(self._request_signature))[
                : self.DIALECT_SKETCH_SIZE
            ]
        )
        committed = replace(
            dialect,
            signature=signature,
            visits=dialect.visits
            if self._active_dialect_is_new
            else dialect.visits + 1,
            last_seen=clock,
        )
        profiles = {row.dialect_id: row for row in self._state.dialects}
        profiles[committed.dialect_id] = committed
        if len(profiles) > self.MAX_DIALECTS:
            candidates = [
                row
                for row in profiles.values()
                if row.dialect_id != committed.dialect_id
            ]
            evicted = min(
                candidates,
                key=lambda row: (
                    row.visits
                    * math.exp(-self.RICCI_AGE_ALPHA * (clock - row.last_seen)),
                    row.last_seen,
                    row.dialect_id,
                ),
            )
            del profiles[evicted.dialect_id]
            self._dialect_evictions += 1
        self._state = replace(
            self._state,
            clock=clock,
            dialects=tuple(sorted(profiles.values(), key=lambda row: row.dialect_id)),
        )

    def reconcile_prefix(self, history: tuple[int, ...], /) -> None:
        committed = self._token_tuple(history, label="reconciled Markov history")
        base = self._pending_base
        proposal = self._pending_proposal
        if base is None or proposal is None:
            raise MarkovDraftError("reconcile_prefix requires a pending proposal")
        if len(self._pending_feedback) != self.proposal_width + 1:
            raise MarkovDraftError("Markov council proposal feedback is missing")
        if committed[: len(base)] != base:
            raise MarkovDraftError("Markov reconciliation changed its known base")
        delta = committed[len(base) :]
        if len(delta) > self.proposal_width or delta != proposal[: len(delta)]:
            raise MarkovDraftError("Markov reconciliation is not a proposal prefix")
        assert self._last_confirmed_length is not None
        for index, token in enumerate(delta):
            self._episode_feedback.append(
                (self._pending_feedback[index], token, index)
            )
        if self._pending_phrase_option is not None:
            self._phrase_accepted_tokens += min(
                len(delta),
                len(self._pending_phrase_option.token_ids),
            )
            if self._pending_phrase_option.kind == "composition":
                self._composition_accepted_tokens += min(
                    len(delta),
                    len(self._pending_phrase_option.token_ids),
                )
        self._carry_feedback = self._pending_feedback[len(delta)]
        self._carry_feedback_position = len(delta)
        self._last_confirmed_length = len(committed)
        self._pending_base = None
        self._pending_proposal = None
        self._pending_feedback = ()
        self._pending_phrase_option = None
        self._pending_composition_program = None
        self._reconcile_calls += 1

    def reconcile_external_prefix(self, history: tuple[int, ...], /) -> None:
        """Train one unused Council proposal from another verified provider.

        Feedback remains valid through the first mismatch: later Council rows
        were conditioned on its rejected token rather than the actual target
        prefix.  A fully matching observed prefix retains the next-token carry;
        a mismatch drops that counterfactual tail and the next round rebuilds
        directly from the confirmed history.
        """

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        if not self._request_started or self._request_completed:
            raise MarkovDraftError("Markov request is not active")
        committed = self._token_tuple(
            history,
            label="externally reconciled Markov history",
        )
        base = self._pending_base
        proposal = self._pending_proposal
        if base is None or proposal is None:
            raise MarkovDraftError(
                "reconcile_external_prefix requires a pending proposal"
            )
        if len(self._pending_feedback) != self.proposal_width + 1:
            raise MarkovDraftError("Markov council proposal feedback is missing")
        if committed[: len(base)] != base:
            raise MarkovDraftError(
                "external Markov reconciliation changed its known base"
            )
        delta = committed[len(base) :]
        if len(delta) > self.proposal_width:
            raise MarkovDraftError(
                "external Markov reconciliation exceeds the proposal width"
            )
        assert self._last_confirmed_length is not None
        verified = 0
        prefix_matches = True
        mismatch_index: int | None = None
        for index, token in enumerate(delta):
            self._episode_feedback.append(
                (self._pending_feedback[index], token, index)
            )
            verified += 1
            if token != proposal[index]:
                prefix_matches = False
                mismatch_index = index
                break
        if mismatch_index is not None:
            for index in range(mismatch_index + 1, len(delta)):
                actual_context = (*base, *delta[:index])
                try:
                    teacher_feedback, _teacher_token = (
                        self._teacher_forced_prediction(
                            actual_context,
                            position=index,
                        )
                    )
                except (MarkovDraftError, ValueError):
                    self._teacher_forced_failures += 1
                    break
                self._episode_feedback.append(
                    (teacher_feedback, delta[index], index)
                )
                verified += 1
                self._teacher_forced_predictions += 1
                self._teacher_forced_feedback_tokens += 1
        self._carry_feedback = (
            self._pending_feedback[len(delta)] if prefix_matches else None
        )
        self._carry_feedback_position = len(delta) if prefix_matches else None
        self._last_confirmed_length = len(committed)
        self._pending_base = None
        self._pending_proposal = None
        self._pending_feedback = ()
        self._pending_phrase_option = None
        self._pending_composition_program = None
        self._reconcile_calls += 1
        self._external_reconcile_calls += 1
        self._external_feedback_tokens += verified

    def discard_pending_proposal(self) -> None:
        """Drop one unused proposal while retaining request-level learning.

        A provider cascade may ask the council whether it already owns a
        useful continuation, then delegate the request to another drafter.
        The council's speculative feedback is unavailable in that case, but
        the final target-confirmed episode is still valuable and remains
        learnable through :meth:`observe_final`.
        """

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        if self._pending_base is None or self._pending_proposal is None:
            raise MarkovDraftError("no pending Markov proposal to discard")
        self._pending_base = None
        self._pending_proposal = None
        self._pending_feedback = ()
        self._pending_phrase_option = None
        self._pending_composition_program = None
        self._last_round_proposal = None

    def advance_confirmed_prefix(self, history: tuple[int, ...], /) -> None:
        """Follow target-confirmed tokens emitted by another draft expert.

        This keeps the cheap Council eligible on every later decode round while
        another provider owns the current proposal.  No synthetic expert
        feedback is invented; the complete target episode is learned once at
        finalization as usual.
        """

        if self._closed:
            raise MarkovDraftError("Markov draft provider is closed")
        if not self._request_started or self._request_completed:
            raise MarkovDraftError("Markov request is not active")
        if self._pending_base is not None or self._pending_proposal is not None:
            raise MarkovDraftError("cannot advance an unreconciled Markov proposal")
        committed = self._token_tuple(history, label="advanced Markov history")
        prompt = self._request_prompt
        if prompt is None or committed[: len(prompt)] != prompt:
            raise MarkovDraftError("advanced Markov history changed its request prompt")
        previous = self._last_confirmed_length
        if previous is None:
            previous = len(prompt)
        if len(committed) < previous:
            raise MarkovDraftError("advanced Markov history moved backwards")
        if self._carry_feedback is not None and len(committed) > previous:
            if self._carry_feedback_position is None:
                raise MarkovDraftError("Markov carry feedback position is missing")
            self._episode_feedback.append(
                (
                    self._carry_feedback,
                    committed[previous],
                    self._carry_feedback_position,
                )
            )
            self._carry_feedback = None
            self._carry_feedback_position = None
        self._last_confirmed_length = len(committed)

    def observe_final(self, history: tuple[int, ...], /) -> None:
        committed = self._token_tuple(history, label="final Markov history")
        if self._request_completed:
            raise MarkovDraftError("Markov provider request is already complete")
        if self._active_dialect is None:
            self._activate_dialect(committed)
            self._request_started = True
        if self._pending_base is not None:
            raise MarkovDraftError("cannot finalize an unreconciled proposal")
        original_state = self._state
        original_dialect = self._active_dialect
        original_carry = self._carry_feedback
        original_carry_position = self._carry_feedback_position
        original_feedback = list(self._episode_feedback)
        original_confirmed_length = self._last_confirmed_length
        original_evictions = self._dialect_evictions
        try:
            if (
                self._last_confirmed_length is not None
                and len(committed) < self._last_confirmed_length
            ):
                raise MarkovDraftError("final Markov history moved backwards")
            if (
                self._carry_feedback is not None
                and self._last_confirmed_length is not None
                and len(committed) > self._last_confirmed_length
            ):
                if self._carry_feedback_position is None:
                    raise MarkovDraftError("Markov carry feedback position is missing")
                self._episode_feedback.append(
                    (
                        self._carry_feedback,
                        committed[self._last_confirmed_length],
                        self._carry_feedback_position,
                    )
                )
            self._carry_feedback = None
            self._carry_feedback_position = None
            for feedback, token, position in self._episode_feedback:
                self._apply_council_feedback(feedback, token, position)
            self._episode_feedback.clear()
            self._learn_episode(
                committed,
                prompt_length=self._request_prompt_length,
            )
            if self._pending_import_digest is not None:
                self._state = replace(
                    self._state,
                    imported_episode_sha256s=tuple(
                        sorted(
                            {
                                *self._state.imported_episode_sha256s,
                                self._pending_import_digest,
                            }
                        )
                    ),
                )
            self._commit_active_dialect()
            self._last_confirmed_length = len(committed)
            self._persist()
            self._request_completed = True
        except Exception:
            self._state = original_state
            self._active_dialect = original_dialect
            self._carry_feedback = original_carry
            self._carry_feedback_position = original_carry_position
            self._episode_feedback = original_feedback
            self._last_confirmed_length = original_confirmed_length
            self._dialect_evictions = original_evictions
            raise

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
            directory = os.open(
                path.parent,
                os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as exc:
            raise MarkovDraftError("cannot persist Markov draft state") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def metrics(self) -> MarkovDraftMetrics:
        weights = self._weights()
        accuracy = tuple(
            0.0 if seen == 0 else hit / seen
            for hit, seen in zip(
                self._state.expert_hits,
                self._state.expert_observations,
                strict=True,
            )
        )
        return MarkovDraftMetrics(
            schema=MARKOV_DRAFT_METRICS_SCHEMA,
            draft_calls=self._draft_calls,
            reconcile_calls=self._reconcile_calls,
            predictions=self._predictions,
            learned_tokens=len(self._state.token_ids),
            episode_count=len(self._state.episode_lengths),
            imported_episode_count=len(self._state.imported_episode_sha256s),
            updates=self._state.updates,
            state_bytes=len(self._state.to_bytes()),
            council_predictions=self._council_predictions,
            council_feedback=self._council_feedback,
            external_reconcile_calls=self._external_reconcile_calls,
            external_feedback_tokens=self._external_feedback_tokens,
            teacher_forced_predictions=self._teacher_forced_predictions,
            teacher_forced_feedback_tokens=self._teacher_forced_feedback_tokens,
            teacher_forced_failures=self._teacher_forced_failures,
            leader_changes=self._state.leader_changes,
            expert_weights=tuple(zip(self._state.expert_names, weights, strict=True)),
            expert_accuracy=tuple(zip(self._state.expert_names, accuracy, strict=True)),
            effective_experts=1.0 / sum(value * value for value in weights),
            last_confidence=self._last_confidence,
            last_raw_confidence=self._last_raw_confidence,
            last_empirical_evidence=self._last_empirical_evidence,
            last_disagreement=self._last_disagreement,
            horizon_observations=tuple(
                max(row, default=0)
                for row in self._state.horizon_expert_observations
            ),
            horizon_mean_accuracy=tuple(
                sum(
                    weight * (hit / observed if observed else 0.0)
                    for weight, observed, hit in zip(
                        weights,
                        observed_row,
                        hit_row,
                        strict=True,
                    )
                )
                for observed_row, hit_row in zip(
                    self._state.horizon_expert_observations,
                    self._state.horizon_expert_hits,
                    strict=True,
                )
            ),
            regime_generation=self._state.regime_generation,
            surprise_mean=self._state.surprise_mean,
            surprise_cusum=self._state.surprise_cusum,
            dialect_count=len(self._state.dialects),
            active_dialect_id=(
                None
                if self._active_dialect is None
                else self._active_dialect.dialect_id
            ),
            active_dialect_similarity=self._active_dialect_similarity,
            dialect_evictions=self._dialect_evictions,
            phrase_option_calls=self._phrase_option_calls,
            phrase_draft_tokens=self._phrase_draft_tokens,
            phrase_accepted_tokens=self._phrase_accepted_tokens,
            composition_programs=self._composition_program_count,
            composition_option_calls=self._composition_option_calls,
            composition_draft_tokens=self._composition_draft_tokens,
            composition_accepted_tokens=self._composition_accepted_tokens,
            last_composition_support=(
                0
                if self._last_composition_program is None
                else self._last_composition_program.support
            ),
            last_composition_copy_tokens=(
                0
                if self._last_composition_program is None
                else self._last_composition_program.copied_tokens
            ),
            last_phrase_source=(
                None
                if self._last_phrase_option is None
                else self._last_phrase_option.source
            ),
            last_phrase_support=(
                0
                if self._last_phrase_option is None
                else self._last_phrase_option.support
            ),
            last_phrase_confidence=(
                0.0
                if self._last_phrase_option is None
                else self._last_phrase_option.confidence
            ),
            last_phrase_width=(
                0
                if self._last_phrase_option is None
                else len(self._last_phrase_option.token_ids)
            ),
            adaptive_proposal_calls=self._adaptive_proposal_calls,
            recommended_windows=tuple(
                (window, self._recommended_window_counts[window])
                for window in (1, 4, 8, 16)
            ),
            last_recommended_window=(
                None
                if self._last_round_proposal is None
                else self._last_round_proposal.recommended_window
            ),
            last_horizon_utilities=(
                ()
                if self._last_round_proposal is None
                else tuple(
                    (row.window, row.utility)
                    for row in self._last_round_proposal.horizons
                )
            ),
            proposal_width=self.proposal_width,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._pending_base = None
        self._pending_proposal = None
        self._pending_feedback = ()
        self._carry_feedback = None
        self._carry_feedback_position = None
        self._pending_phrase_option = None
        self._pending_composition_program = None
        self._pending_import_digest = None
        self._episode_feedback.clear()
        try:
            self._persist()
        finally:
            self._closed = True
            self._release_state_lock()


__all__ = [
    "LEGACY_MARKOV_DRAFT_STATE_SCHEMA",
    "V2_MARKOV_DRAFT_STATE_SCHEMA",
    "V3_MARKOV_DRAFT_STATE_SCHEMA",
    "V4_MARKOV_DRAFT_STATE_SCHEMA",
    "V5_MARKOV_DRAFT_STATE_SCHEMA",
    "MARKOV_DRAFT_METRICS_SCHEMA",
    "MARKOV_DRAFT_PROVIDER_ABI",
    "MARKOV_DRAFT_STATE_SCHEMA",
    "FingerprintRollingK4DraftProvider",
    "MarkovDraftError",
    "MarkovDialectState",
    "MarkovDraftMetrics",
    "MarkovDraftState",
    "MarkovExpertSpec",
    "MarkovPhraseOption",
]
