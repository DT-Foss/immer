"""Demand-driven compute-battery control for native Qwen state anchors.

This module is deliberately a *control plane* for
``SemanticStateAnchorCache``.  It learns reusable token prefixes, accounts for
their measured economics, and chooses which native anchors should be charged
during idle time.  It never executes a model and never serialises tensors.

The persistent sidecar contains token IDs and numeric measurements, but no
prompt text.  Token IDs are reversible with the tokenizer and therefore remain
prompt-equivalent sensitive data: the sidecar is local-only and mode ``0600``.
Its canonical SHA-256 seal detects accidental or unsealed modification.  The
native anchor cache remains responsible for authenticating snapshot manifests
and blobs.  Verifying a snapshot blob with SHA-256 is O(blob bytes), not O(1);
only comparison of two already-computed digests is O(1).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Iterator, TYPE_CHECKING

from .semantic_state_cache import token_prefix_sha256

if TYPE_CHECKING:
    from .semantic_state_cache import AnchorReceipt, SemanticStateAnchorCache


ANCHOR_BATTERY_STATE_SCHEMA = "immer.qwen3.8-anchor-battery/v1"
RADIX_DEMAND_SCHEMA = "immer.qwen3.8-anchor-demand-radix/v1"
PROFIT_LEDGER_SCHEMA = "immer.qwen3.8-anchor-profit-ledger/v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_TOKEN_ID = 2**63 - 1
_MAX_PREFIX_TOKENS = 1_048_576
_MAX_STATE_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024


class AnchorBatteryError(ValueError):
    """Base error for invalid battery observations or policy inputs."""


class AnchorBatteryIntegrityError(AnchorBatteryError):
    """The persisted control-plane state is malformed or has changed."""


class AnchorBatteryIdentityError(AnchorBatteryError):
    """A state sidecar belongs to another immutable model pin."""


class AnchorBatteryConflictError(AnchorBatteryIntegrityError):
    """A newer writer committed after this controller loaded its state."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AnchorBatteryIntegrityError(
            "battery state is not canonical JSON"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_document(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AnchorBatteryIntegrityError(f"{label} must be a lowercase SHA-256")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AnchorBatteryIntegrityError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise AnchorBatteryIntegrityError(f"{label} must be positive")
    return result


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnchorBatteryIntegrityError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise AnchorBatteryIntegrityError(f"{label} must be finite and non-negative")
    return result


def _tokens(token_ids: Sequence[int]) -> tuple[int, ...]:
    if isinstance(token_ids, (str, bytes, bytearray)) or not isinstance(
        token_ids, Sequence
    ):
        raise TypeError("token_ids must be a sequence of integers")
    if len(token_ids) > _MAX_PREFIX_TOKENS:
        raise AnchorBatteryError(f"token sequence exceeds {_MAX_PREFIX_TOKENS} tokens")
    result: list[int] = []
    for token_id in token_ids:
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            or token_id > _MAX_TOKEN_ID
        ):
            raise AnchorBatteryError(
                "token IDs must be non-negative signed 64-bit integers"
            )
        result.append(token_id)
    return tuple(result)


def _json_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AnchorBatteryIntegrityError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _stable_signature(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


@contextmanager
def _exclusive_state_lock(destination: Path) -> Iterator[None]:
    """Serialize state CAS transactions across processes without symlinks."""

    parent = destination.parent
    try:
        parent_before = os.lstat(parent)
    except OSError as exc:
        raise AnchorBatteryIntegrityError(
            "battery state parent is unavailable"
        ) from exc
    if stat.S_ISLNK(parent_before.st_mode) or not stat.S_ISDIR(parent_before.st_mode):
        raise AnchorBatteryIntegrityError("battery state parent must be a directory")
    lock_path = parent / f".{destination.name}.lock"
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise AnchorBatteryIntegrityError("battery state lock is unavailable") from exc
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        opened_before = os.fstat(descriptor)
        linked_before = os.lstat(lock_path)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or not stat.S_ISREG(linked_before.st_mode)
            or not _same_inode(opened_before, linked_before)
        ):
            raise AnchorBatteryIntegrityError("battery state lock is not stable")
        yield
        opened_after = os.fstat(descriptor)
        linked_after = os.lstat(lock_path)
        parent_after = os.lstat(parent)
        if (
            _stable_signature(opened_before) != _stable_signature(opened_after)
            or not _same_inode(opened_after, linked_after)
            or not _same_inode(parent_before, parent_after)
        ):
            raise AnchorBatteryIntegrityError("battery state lock changed")
    except OSError as exc:
        raise AnchorBatteryIntegrityError("battery state lock failed") from exc
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@dataclass(slots=True)
class _RadixNode:
    demand_count: int = 0
    boundary_count: int = 0
    last_seen_sequence: int = 0
    children: dict[int, "_RadixNode"] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PrefixDemand:
    """One exact token prefix learned from request demand."""

    token_ids: tuple[int, ...]
    prefix_sha256: str
    demand_count: int
    boundary_count: int
    last_seen_sequence: int

    def __post_init__(self) -> None:
        tokens = _tokens(self.token_ids)
        if not tokens:
            raise AnchorBatteryError("prefix demand cannot be empty")
        if self.prefix_sha256 != token_prefix_sha256(tokens):
            raise AnchorBatteryIntegrityError("prefix demand SHA-256 mismatch")
        _positive_int(self.demand_count, "prefix demand count")
        boundary = _nonnegative_int(self.boundary_count, "prefix boundary count")
        if boundary > self.demand_count:
            raise AnchorBatteryIntegrityError("boundary count exceeds demand count")
        _positive_int(self.last_seen_sequence, "prefix last-seen sequence")

    @property
    def prefix_length(self) -> int:
        return len(self.token_ids)


class RadixDemandMiner:
    """Incrementally count exact token prefixes in a compact radix trie.

    Every request contributes one count to each prefix along its trie path.
    Consequently an unseen suffix can reuse any prefix learned from earlier
    requests.  Optional semantic boundary lengths are counted separately and
    can be used as a deterministic policy preference; they never replace exact
    token-prefix matching.
    """

    def __init__(self, *, max_prefix_tokens: int = 4096) -> None:
        if (
            isinstance(max_prefix_tokens, bool)
            or not isinstance(max_prefix_tokens, int)
            or max_prefix_tokens <= 0
            or max_prefix_tokens > _MAX_PREFIX_TOKENS
        ):
            raise ValueError(f"max_prefix_tokens must be in [1, {_MAX_PREFIX_TOKENS}]")
        self.max_prefix_tokens = max_prefix_tokens
        self.observation_count = 0
        self._root = _RadixNode()

    def observe(
        self,
        token_ids: Sequence[int],
        *,
        semantic_boundary_lengths: Iterable[int] = (),
    ) -> tuple[PrefixDemand, ...]:
        """Add one exact token sequence and return its updated prefix path."""

        tokens = _tokens(token_ids)
        if not tokens:
            raise AnchorBatteryError("cannot learn from an empty token sequence")
        limit = min(len(tokens), self.max_prefix_tokens)
        boundaries: set[int] = set()
        for raw_length in semantic_boundary_lengths:
            if isinstance(raw_length, bool) or not isinstance(raw_length, int):
                raise AnchorBatteryError("semantic boundary lengths must be integers")
            if raw_length <= 0 or raw_length > limit:
                raise AnchorBatteryError(
                    "semantic boundary length lies outside the observed prefix"
                )
            boundaries.add(raw_length)
        self.observation_count += 1
        sequence = self.observation_count
        self._root.demand_count += 1
        self._root.last_seen_sequence = sequence
        node = self._root
        path: list[PrefixDemand] = []
        prefix: list[int] = []
        for depth, token_id in enumerate(tokens[:limit], start=1):
            prefix.append(token_id)
            node = node.children.setdefault(token_id, _RadixNode())
            node.demand_count += 1
            node.last_seen_sequence = sequence
            if depth in boundaries:
                node.boundary_count += 1
            exact = tuple(prefix)
            path.append(
                PrefixDemand(
                    token_ids=exact,
                    prefix_sha256=token_prefix_sha256(exact),
                    demand_count=node.demand_count,
                    boundary_count=node.boundary_count,
                    last_seen_sequence=sequence,
                )
            )
        return tuple(path)

    def candidates(
        self,
        *,
        min_demand: int = 2,
        min_prefix_tokens: int = 1,
        semantic_boundaries_only: bool = False,
        collapse_dominated: bool = True,
    ) -> tuple[PrefixDemand, ...]:
        """Return recurring prefixes in deterministic longest-first order.

        By default, a shorter node is omitted when one child has the exact
        same demand: every observed request already shares the longer prefix,
        so the short cell would save less work without covering more demand.
        Semantic boundaries are always retained.
        """

        if (
            isinstance(min_demand, bool)
            or not isinstance(min_demand, int)
            or min_demand <= 0
        ):
            raise ValueError("min_demand must be a positive integer")
        if (
            isinstance(min_prefix_tokens, bool)
            or not isinstance(min_prefix_tokens, int)
            or min_prefix_tokens <= 0
        ):
            raise ValueError("min_prefix_tokens must be a positive integer")
        if not isinstance(semantic_boundaries_only, bool) or not isinstance(
            collapse_dominated, bool
        ):
            raise TypeError("candidate filters must be boolean")
        rows: list[PrefixDemand] = []
        stack: list[tuple[_RadixNode, tuple[int, ...]]] = [(self._root, ())]
        while stack:
            node, prefix = stack.pop()
            if prefix and node.demand_count >= min_demand:
                dominated = collapse_dominated and any(
                    child.demand_count == node.demand_count
                    for child in node.children.values()
                )
                if (
                    len(prefix) >= min_prefix_tokens
                    and (not semantic_boundaries_only or node.boundary_count > 0)
                    and (node.boundary_count > 0 or not dominated)
                ):
                    rows.append(
                        PrefixDemand(
                            token_ids=prefix,
                            prefix_sha256=token_prefix_sha256(prefix),
                            demand_count=node.demand_count,
                            boundary_count=node.boundary_count,
                            last_seen_sequence=node.last_seen_sequence,
                        )
                    )
            for token_id in sorted(node.children, reverse=True):
                stack.append((node.children[token_id], (*prefix, token_id)))
        rows.sort(
            key=lambda row: (
                -row.prefix_length,
                -row.demand_count,
                -row.boundary_count,
                row.prefix_sha256,
            )
        )
        return tuple(rows)

    def longest_recurring_prefix(
        self, token_ids: Sequence[int], *, min_demand: int = 2
    ) -> PrefixDemand | None:
        """Find the deepest recurring prefix for a possibly unseen suffix."""

        if (
            isinstance(min_demand, bool)
            or not isinstance(min_demand, int)
            or min_demand <= 0
        ):
            raise ValueError("min_demand must be a positive integer")
        tokens = _tokens(token_ids)
        node = self._root
        deepest: PrefixDemand | None = None
        prefix: list[int] = []
        for token_id in tokens[: self.max_prefix_tokens]:
            node = node.children.get(token_id)  # type: ignore[assignment]
            if node is None:
                break
            prefix.append(token_id)
            if node.demand_count >= min_demand:
                exact = tuple(prefix)
                deepest = PrefixDemand(
                    token_ids=exact,
                    prefix_sha256=token_prefix_sha256(exact),
                    demand_count=node.demand_count,
                    boundary_count=node.boundary_count,
                    last_seen_sequence=node.last_seen_sequence,
                )
        return deepest

    def to_document(self) -> dict[str, Any]:
        # Breadth-first rows avoid Python recursion limits for long system
        # prompts while storing each radix edge exactly once.
        pending: list[tuple[_RadixNode, int, int | None]] = [(self._root, -1, None)]
        rows: list[dict[str, Any]] = []
        cursor = 0
        while cursor < len(pending):
            node, parent, token_id = pending[cursor]
            rows.append(
                {
                    "boundary_count": node.boundary_count,
                    "demand_count": node.demand_count,
                    "last_seen_sequence": node.last_seen_sequence,
                    "parent": parent,
                    "token_id": token_id,
                }
            )
            for child_token in sorted(node.children):
                pending.append((node.children[child_token], cursor, child_token))
            cursor += 1

        return {
            "max_prefix_tokens": self.max_prefix_tokens,
            "nodes": rows,
            "observation_count": self.observation_count,
            "schema": RADIX_DEMAND_SCHEMA,
        }

    @classmethod
    def from_document(cls, raw: Any) -> "RadixDemandMiner":
        if not isinstance(raw, Mapping) or set(raw) != {
            "max_prefix_tokens",
            "nodes",
            "observation_count",
            "schema",
        }:
            raise AnchorBatteryIntegrityError("radix demand document is invalid")
        if raw.get("schema") != RADIX_DEMAND_SCHEMA:
            raise AnchorBatteryIntegrityError("radix demand schema mismatch")
        maximum = _positive_int(raw.get("max_prefix_tokens"), "maximum prefix tokens")
        if maximum > _MAX_PREFIX_TOKENS:
            raise AnchorBatteryIntegrityError("maximum prefix token count is excessive")
        observation_count = _nonnegative_int(
            raw.get("observation_count"), "observation count"
        )
        raw_nodes = raw.get("nodes")
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise AnchorBatteryIntegrityError("radix demand nodes are invalid")
        nodes: list[_RadixNode] = []
        parent_rows: list[int] = []
        token_rows: list[int | None] = []
        previous_token_by_parent: dict[int, int] = {}
        for index, value in enumerate(raw_nodes):
            if not isinstance(value, Mapping) or set(value) != {
                "boundary_count",
                "demand_count",
                "last_seen_sequence",
                "parent",
                "token_id",
            }:
                raise AnchorBatteryIntegrityError("radix node is invalid")
            demand = _nonnegative_int(value.get("demand_count"), "node demand count")
            boundary = _nonnegative_int(
                value.get("boundary_count"), "node boundary count"
            )
            seen = _nonnegative_int(
                value.get("last_seen_sequence"), "node last-seen sequence"
            )
            if (
                boundary > demand
                or demand > observation_count
                or seen > observation_count
            ):
                raise AnchorBatteryIntegrityError(
                    "radix node counters are inconsistent"
                )
            if demand == 0 and seen != 0:
                raise AnchorBatteryIntegrityError("unobserved radix node has recency")
            if demand > 0 and seen == 0:
                raise AnchorBatteryIntegrityError("observed radix node lacks recency")
            parent = value.get("parent")
            token_id = value.get("token_id")
            if index == 0:
                if parent != -1 or token_id is not None:
                    raise AnchorBatteryIntegrityError("radix root row is invalid")
            else:
                if (
                    isinstance(parent, bool)
                    or not isinstance(parent, int)
                    or parent < 0
                    or parent >= index
                    or isinstance(token_id, bool)
                    or not isinstance(token_id, int)
                    or token_id < 0
                    or token_id > _MAX_TOKEN_ID
                ):
                    raise AnchorBatteryIntegrityError("radix edge is invalid")
                if demand > nodes[parent].demand_count:
                    raise AnchorBatteryIntegrityError(
                        "radix child exceeds parent demand"
                    )
                previous = previous_token_by_parent.get(parent, -1)
                if token_id <= previous:
                    raise AnchorBatteryIntegrityError("radix child order is invalid")
                previous_token_by_parent[parent] = token_id
            nodes.append(
                _RadixNode(
                    demand_count=demand,
                    boundary_count=boundary,
                    last_seen_sequence=seen,
                )
            )
            parent_rows.append(parent)
            token_rows.append(token_id)

        for index in range(1, len(nodes)):
            parent = parent_rows[index]
            token_id = token_rows[index]
            assert token_id is not None
            if token_id in nodes[parent].children:
                raise AnchorBatteryIntegrityError("radix child token is duplicated")
            nodes[parent].children[token_id] = nodes[index]
        root = nodes[0]
        if root.demand_count != observation_count:
            raise AnchorBatteryIntegrityError(
                "radix root differs from observation count"
            )
        miner = cls(max_prefix_tokens=maximum)
        miner.observation_count = observation_count
        miner._root = root
        if miner.to_document() != dict(raw):
            raise AnchorBatteryIntegrityError("radix node layout is not canonical")
        return miner


@dataclass(frozen=True, slots=True)
class PrefixProfit:
    """Measured lifetime economics for one exact token prefix."""

    prefix_sha256: str
    prefix_length: int
    charge_seconds: float = 0.0
    active_charge_seconds: float = 0.0
    restore_seconds: float = 0.0
    verify_seconds: float = 0.0
    bytes_stored: int = 0
    demand_count: int = 0
    hits: int = 0
    misses: int = 0
    live_seconds: float = 0.0
    saved_live_seconds: float = 0.0
    invalidated_charge_seconds: float = 0.0
    wasted_charge_seconds: float = 0.0
    charges: int = 0
    invalidations: int = 0
    active_hits: int = 0
    active: bool = False

    def __post_init__(self) -> None:
        _digest(self.prefix_sha256, "profit prefix SHA-256")
        _positive_int(self.prefix_length, "profit prefix length")
        for label, value in (
            ("charge seconds", self.charge_seconds),
            ("active charge seconds", self.active_charge_seconds),
            ("restore seconds", self.restore_seconds),
            ("verify seconds", self.verify_seconds),
            ("live seconds", self.live_seconds),
            ("saved live seconds", self.saved_live_seconds),
            ("invalidated charge seconds", self.invalidated_charge_seconds),
            ("wasted charge seconds", self.wasted_charge_seconds),
        ):
            _finite_nonnegative(value, label)
        for label, value in (
            ("stored bytes", self.bytes_stored),
            ("demand count", self.demand_count),
            ("hits", self.hits),
            ("misses", self.misses),
            ("charges", self.charges),
            ("invalidations", self.invalidations),
            ("active hits", self.active_hits),
        ):
            _nonnegative_int(value, label)
        if self.hits + self.misses != self.demand_count:
            raise AnchorBatteryIntegrityError("profit demand count is inconsistent")
        if self.saved_live_seconds > self.live_seconds:
            raise AnchorBatteryIntegrityError("saved live seconds exceed live baseline")
        if self.invalidated_charge_seconds > self.charge_seconds:
            raise AnchorBatteryIntegrityError("invalidated charge exceeds total charge")
        if self.wasted_charge_seconds > self.invalidated_charge_seconds:
            raise AnchorBatteryIntegrityError(
                "wasted charge exceeds invalidated charge"
            )
        if self.active:
            if self.bytes_stored <= 0 or self.active_charge_seconds <= 0.0:
                raise AnchorBatteryIntegrityError("active profit row lacks a charge")
        elif self.bytes_stored != 0 or self.active_charge_seconds != 0.0:
            raise AnchorBatteryIntegrityError("inactive profit row retains a charge")
        if not self.active and self.active_hits != 0:
            raise AnchorBatteryIntegrityError("inactive profit row retains active hits")
        if self.active_hits > self.hits:
            raise AnchorBatteryIntegrityError("active hits exceed lifetime hits")

    @property
    def soc(self) -> float:
        """Discharged utility divided by all idle charge investment."""

        if self.charge_seconds == 0.0:
            return 0.0
        return min(1.0, self.saved_live_seconds / self.charge_seconds)

    @property
    def self_discharge(self) -> float:
        """Fraction of charge invalidated before its first useful discharge."""

        if self.charge_seconds == 0.0:
            return 0.0
        return min(1.0, self.wasted_charge_seconds / self.charge_seconds)

    @property
    def turnover_fraction(self) -> float:
        """Fraction of all charge retired by model/cache turnover."""

        if self.charge_seconds == 0.0:
            return 0.0
        return min(1.0, self.invalidated_charge_seconds / self.charge_seconds)

    @property
    def waste_fraction(self) -> float:
        if self.charge_seconds == 0.0:
            return 0.0
        return min(1.0, self.wasted_charge_seconds / self.charge_seconds)

    def to_document(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "active_charge_seconds": self.active_charge_seconds,
            "active_hits": self.active_hits,
            "bytes_stored": self.bytes_stored,
            "charge_seconds": self.charge_seconds,
            "charges": self.charges,
            "demand_count": self.demand_count,
            "hits": self.hits,
            "invalidated_charge_seconds": self.invalidated_charge_seconds,
            "invalidations": self.invalidations,
            "live_seconds": self.live_seconds,
            "misses": self.misses,
            "prefix_length": self.prefix_length,
            "prefix_sha256": self.prefix_sha256,
            "restore_seconds": self.restore_seconds,
            "saved_live_seconds": self.saved_live_seconds,
            "verify_seconds": self.verify_seconds,
            "wasted_charge_seconds": self.wasted_charge_seconds,
        }

    @classmethod
    def from_document(cls, raw: Any) -> "PrefixProfit":
        fields = {
            "active",
            "active_charge_seconds",
            "active_hits",
            "bytes_stored",
            "charge_seconds",
            "charges",
            "demand_count",
            "hits",
            "invalidated_charge_seconds",
            "invalidations",
            "live_seconds",
            "misses",
            "prefix_length",
            "prefix_sha256",
            "restore_seconds",
            "saved_live_seconds",
            "verify_seconds",
            "wasted_charge_seconds",
        }
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise AnchorBatteryIntegrityError("profit row is invalid")
        if not isinstance(raw.get("active"), bool):
            raise AnchorBatteryIntegrityError("profit active flag must be boolean")
        return cls(**{key: raw[key] for key in fields})


class ProfitLedger:
    """Per-prefix measured cost, utility, and invalidation accounting."""

    def __init__(self) -> None:
        self._rows: dict[str, PrefixProfit] = {}

    def get(self, prefix_sha256: str) -> PrefixProfit | None:
        return self._rows.get(_digest(prefix_sha256, "prefix SHA-256"))

    def rows(self) -> tuple[PrefixProfit, ...]:
        return tuple(self._rows[key] for key in sorted(self._rows))

    def _row(self, token_ids: Sequence[int]) -> tuple[tuple[int, ...], PrefixProfit]:
        tokens = _tokens(token_ids)
        if not tokens:
            raise AnchorBatteryError("profit prefix cannot be empty")
        digest = token_prefix_sha256(tokens)
        row = self._rows.get(digest)
        if row is None:
            row = PrefixProfit(prefix_sha256=digest, prefix_length=len(tokens))
        elif row.prefix_length != len(tokens):
            raise AnchorBatteryIntegrityError("profit prefix length conflict")
        return tokens, row

    def record_charge(
        self,
        token_ids: Sequence[int],
        *,
        charge_seconds: float,
        bytes_stored: int,
    ) -> PrefixProfit:
        """Record a completed native-cache ``store``; no tensors are handled."""

        _tokens_value, row = self._row(token_ids)
        charge = _finite_nonnegative(charge_seconds, "charge seconds")
        if charge <= 0.0:
            raise AnchorBatteryError("charge_seconds must be positive")
        stored = _positive_int(bytes_stored, "stored bytes")
        if row.active:
            raise AnchorBatteryError("prefix already has an active anchor charge")
        updated = replace(
            row,
            charge_seconds=row.charge_seconds + charge,
            active_charge_seconds=charge,
            bytes_stored=stored,
            charges=row.charges + 1,
            active_hits=0,
            active=True,
        )
        self._rows[updated.prefix_sha256] = updated
        return updated

    def record_anchor_charge(
        self,
        token_ids: Sequence[int],
        anchor: "AnchorReceipt",
        *,
        charge_seconds: float,
    ) -> PrefixProfit:
        """Bind accounting to an actual ``SemanticStateAnchorCache`` receipt."""

        tokens = _tokens(token_ids)
        if anchor.prefix_sha256 != token_prefix_sha256(tokens):
            raise AnchorBatteryIdentityError("anchor receipt belongs to another prefix")
        if anchor.prefix_length != len(tokens):
            raise AnchorBatteryIdentityError("anchor receipt length mismatch")
        return self.record_charge(
            tokens,
            charge_seconds=charge_seconds,
            bytes_stored=anchor.cache_bytes,
        )

    def record_hit(
        self,
        token_ids: Sequence[int],
        *,
        live_seconds: float,
        restore_seconds: float,
        verify_seconds: float,
    ) -> PrefixProfit:
        """Record one discharge against the equivalent live-prefill baseline."""

        _tokens_value, row = self._row(token_ids)
        live = _finite_nonnegative(live_seconds, "live seconds")
        restore = _finite_nonnegative(restore_seconds, "restore seconds")
        verify = _finite_nonnegative(verify_seconds, "verify seconds")
        if not row.active:
            raise AnchorBatteryError("cannot hit an inactive anchor")
        saved = max(0.0, live - restore - verify)
        updated = replace(
            row,
            demand_count=row.demand_count + 1,
            hits=row.hits + 1,
            active_hits=row.active_hits + 1,
            live_seconds=row.live_seconds + live,
            restore_seconds=row.restore_seconds + restore,
            verify_seconds=row.verify_seconds + verify,
            saved_live_seconds=row.saved_live_seconds + saved,
        )
        self._rows[updated.prefix_sha256] = updated
        return updated

    def record_miss(
        self, token_ids: Sequence[int], *, live_seconds: float
    ) -> PrefixProfit:
        _tokens_value, row = self._row(token_ids)
        live = _finite_nonnegative(live_seconds, "live seconds")
        updated = replace(
            row,
            demand_count=row.demand_count + 1,
            misses=row.misses + 1,
            live_seconds=row.live_seconds + live,
        )
        self._rows[updated.prefix_sha256] = updated
        return updated

    def invalidate(self, prefix_sha256: str) -> PrefixProfit:
        digest = _digest(prefix_sha256, "prefix SHA-256")
        row = self._rows.get(digest)
        if row is None:
            raise AnchorBatteryError("cannot invalidate an unknown prefix")
        if not row.active:
            return row
        wasted = row.active_charge_seconds if row.active_hits == 0 else 0.0
        updated = replace(
            row,
            active=False,
            active_charge_seconds=0.0,
            active_hits=0,
            bytes_stored=0,
            invalidated_charge_seconds=(
                row.invalidated_charge_seconds + row.active_charge_seconds
            ),
            wasted_charge_seconds=row.wasted_charge_seconds + wasted,
            invalidations=row.invalidations + 1,
        )
        self._rows[digest] = updated
        return updated

    def reconcile_cache(self, cache: "SemanticStateAnchorCache") -> tuple[str, ...]:
        """Mark active ledger rows absent from the native anchor index invalid.

        This reads only cache receipts.  Snapshot verification remains the
        native cache's job and requires O(total blob bytes) when performed.
        """

        active = {receipt.prefix_sha256: receipt for receipt in cache.receipts()}
        invalidated: list[str] = []
        for digest, row in tuple(self._rows.items()):
            if not row.active:
                continue
            receipt = active.get(digest)
            if receipt is None:
                self.invalidate(digest)
                invalidated.append(digest)
                continue
            if receipt.prefix_length != row.prefix_length:
                raise AnchorBatteryIdentityError(
                    "cache receipt length conflicts with ledger"
                )
            if receipt.cache_bytes != row.bytes_stored:
                raise AnchorBatteryIdentityError(
                    "cache receipt bytes conflict with ledger"
                )
        return tuple(sorted(invalidated))

    @property
    def charge_seconds(self) -> float:
        return sum(row.charge_seconds for row in self._rows.values())

    @property
    def saved_live_seconds(self) -> float:
        return sum(row.saved_live_seconds for row in self._rows.values())

    @property
    def invalidated_charge_seconds(self) -> float:
        return sum(row.invalidated_charge_seconds for row in self._rows.values())

    @property
    def wasted_charge_seconds(self) -> float:
        return sum(row.wasted_charge_seconds for row in self._rows.values())

    @property
    def soc(self) -> float:
        charged = self.charge_seconds
        return 0.0 if charged == 0.0 else min(1.0, self.saved_live_seconds / charged)

    @property
    def self_discharge(self) -> float:
        charged = self.charge_seconds
        return 0.0 if charged == 0.0 else min(1.0, self.wasted_charge_seconds / charged)

    @property
    def turnover_fraction(self) -> float:
        charged = self.charge_seconds
        return (
            0.0
            if charged == 0.0
            else min(1.0, self.invalidated_charge_seconds / charged)
        )

    def to_document(self) -> dict[str, Any]:
        return {
            "rows": [row.to_document() for row in self.rows()],
            "schema": PROFIT_LEDGER_SCHEMA,
        }

    @classmethod
    def from_document(cls, raw: Any) -> "ProfitLedger":
        if not isinstance(raw, Mapping) or set(raw) != {"rows", "schema"}:
            raise AnchorBatteryIntegrityError("profit ledger document is invalid")
        if raw.get("schema") != PROFIT_LEDGER_SCHEMA:
            raise AnchorBatteryIntegrityError("profit ledger schema mismatch")
        raw_rows = raw.get("rows")
        if not isinstance(raw_rows, list):
            raise AnchorBatteryIntegrityError("profit ledger rows are invalid")
        rows = tuple(PrefixProfit.from_document(row) for row in raw_rows)
        digests = tuple(row.prefix_sha256 for row in rows)
        if digests != tuple(sorted(digests)) or len(digests) != len(set(digests)):
            raise AnchorBatteryIntegrityError("profit ledger order is not canonical")
        ledger = cls()
        ledger._rows = {row.prefix_sha256: row for row in rows}
        return ledger


@dataclass(frozen=True, slots=True)
class AnchorChargeCandidate:
    """Measured forecast for charging one native semantic anchor."""

    token_ids: tuple[int, ...]
    expected_live_cost: float
    idle_charge_cost: float
    store_cost: float
    verify_cost: float
    invalidation_cost: float
    bytes_stored: int
    charge_seconds: float
    demand_count: int
    o1_surprise: float = 0.0
    o1_learning_progress: float = 0.0

    def __post_init__(self) -> None:
        if not _tokens(self.token_ids):
            raise AnchorBatteryError("charge candidate prefix cannot be empty")
        for label, value in (
            ("expected live cost", self.expected_live_cost),
            ("idle charge cost", self.idle_charge_cost),
            ("store cost", self.store_cost),
            ("verify cost", self.verify_cost),
            ("invalidation cost", self.invalidation_cost),
            ("charge seconds", self.charge_seconds),
        ):
            _finite_nonnegative(value, label)
        _positive_int(self.bytes_stored, "candidate stored bytes")
        _positive_int(self.demand_count, "candidate demand count")
        _finite_nonnegative(self.o1_surprise, "O1 surprise")
        _finite_nonnegative(self.o1_learning_progress, "O1 learning progress")

    @property
    def prefix_sha256(self) -> str:
        return token_prefix_sha256(self.token_ids)

    @property
    def net_value(self) -> float:
        """David's complete compute-battery profitability inequality."""

        return (
            self.expected_live_cost
            - self.idle_charge_cost
            - self.store_cost
            - self.verify_cost
            - self.invalidation_cost
        )

    @property
    def value_density(self) -> float:
        return self.net_value / self.bytes_stored

    @property
    def o1_progress_signal(self) -> float:
        """Bound raw O1 values; surprise matters only with learning progress."""

        progress = self.o1_learning_progress / (1.0 + self.o1_learning_progress)
        surprise = self.o1_surprise / (1.0 + self.o1_surprise)
        return progress * (1.0 + surprise)


@dataclass(frozen=True, slots=True)
class AnchorChargePlan:
    selected: tuple[AnchorChargeCandidate, ...]
    rejected_unprofitable: tuple[AnchorChargeCandidate, ...]
    skipped_budget: tuple[AnchorChargeCandidate, ...]
    charge_bytes: int
    charge_seconds: float


class AnchorChargePolicy:
    """Deterministic profitability and value-density charge controller.

    Economics is the hard gate and primary ordering.  O1 learning progress,
    then surprise gated by that progress, breaks economic ties.  Therefore a
    raw high-surprise prefix can never replace a more profitable prefix or make
    a loss-making charge eligible.
    """

    @staticmethod
    def _priority(candidate: AnchorChargeCandidate) -> tuple[Any, ...]:
        return (
            -candidate.value_density,
            -candidate.o1_progress_signal,
            -len(candidate.token_ids),
            -candidate.demand_count,
            candidate.prefix_sha256,
        )

    def rank(
        self, candidates: Iterable[AnchorChargeCandidate]
    ) -> tuple[AnchorChargeCandidate, ...]:
        rows = tuple(candidates)
        digests = [row.prefix_sha256 for row in rows]
        if len(digests) != len(set(digests)):
            raise AnchorBatteryError("charge candidates contain duplicate prefixes")
        return tuple(sorted(rows, key=self._priority))

    def select(
        self,
        candidates: Iterable[AnchorChargeCandidate],
        *,
        max_charge_bytes: int,
        max_charge_seconds: float,
    ) -> AnchorChargePlan:
        byte_budget = _nonnegative_int(max_charge_bytes, "charge byte budget")
        time_budget = _finite_nonnegative(max_charge_seconds, "charge time budget")
        ranked = self.rank(candidates)
        selected: list[AnchorChargeCandidate] = []
        rejected: list[AnchorChargeCandidate] = []
        skipped: list[AnchorChargeCandidate] = []
        used_bytes = 0
        used_seconds = 0.0
        for candidate in ranked:
            if candidate.net_value <= 0.0:
                rejected.append(candidate)
                continue
            if (
                used_bytes + candidate.bytes_stored > byte_budget
                or used_seconds + candidate.charge_seconds > time_budget
            ):
                skipped.append(candidate)
                continue
            selected.append(candidate)
            used_bytes += candidate.bytes_stored
            used_seconds += candidate.charge_seconds
        return AnchorChargePlan(
            selected=tuple(selected),
            rejected_unprofitable=tuple(rejected),
            skipped_budget=tuple(skipped),
            charge_bytes=used_bytes,
            charge_seconds=used_seconds,
        )

    def eviction_order(
        self, candidates: Iterable[AnchorChargeCandidate]
    ) -> tuple[AnchorChargeCandidate, ...]:
        """Return the exact inverse charge preference: weakest cells first."""

        return tuple(reversed(self.rank(candidates)))


class AnchorBatteryController:
    """Sealed persistent radix miner plus profit ledger.

    ``save``/``open`` persist only control metadata.  Actual state tensors stay
    exclusively inside ``SemanticStateAnchorCache``.  The radix rows contain
    reversible token IDs and are confidential despite containing no raw text.
    """

    def __init__(
        self,
        *,
        model_pin_sha256: str,
        miner: RadixDemandMiner | None = None,
        ledger: ProfitLedger | None = None,
        generation: int = 0,
    ) -> None:
        self.model_pin_sha256 = _digest(model_pin_sha256, "model pin SHA-256")
        self.miner = miner or RadixDemandMiner()
        self.ledger = ledger or ProfitLedger()
        self.generation = _nonnegative_int(generation, "battery generation")
        self._loaded_path: Path | None = None
        self._loaded_file_sha256: str | None = None
        self._loaded_generation: int | None = None

    def observe(
        self,
        token_ids: Sequence[int],
        *,
        semantic_boundary_lengths: Iterable[int] = (),
    ) -> tuple[PrefixDemand, ...]:
        return self.miner.observe(
            token_ids,
            semantic_boundary_lengths=semantic_boundary_lengths,
        )

    def _document_at_generation(self, generation: int) -> dict[str, Any]:
        body = {
            "generation": _nonnegative_int(generation, "battery generation"),
            "ledger": self.ledger.to_document(),
            "miner": self.miner.to_document(),
            "model_pin_sha256": self.model_pin_sha256,
            "schema": ANCHOR_BATTERY_STATE_SCHEMA,
        }
        return {"body": body, "body_sha256": _sha256_document(body)}

    def to_document(self) -> dict[str, Any]:
        return self._document_at_generation(self.generation)

    def save(self, path: str | os.PathLike[str]) -> str:
        """Atomically save one canonical, hash-sealed control-plane snapshot."""

        destination = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
        destination.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        with _exclusive_state_lock(destination):
            try:
                linked = os.lstat(destination)
            except FileNotFoundError:
                linked = None
            except OSError as exc:
                raise AnchorBatteryIntegrityError(
                    "cannot inspect battery state"
                ) from exc
            if linked is not None and (
                stat.S_ISLNK(linked.st_mode) or not stat.S_ISREG(linked.st_mode)
            ):
                raise AnchorBatteryIntegrityError(
                    "battery state must be a regular file"
                )
            if self._loaded_path is None:
                if linked is not None:
                    raise AnchorBatteryConflictError(
                        "refusing to replace state without a loaded CAS base"
                    )
            else:
                if destination != self._loaded_path:
                    raise AnchorBatteryConflictError(
                        "loaded controller cannot save to another state path"
                    )
                if linked is None:
                    raise AnchorBatteryConflictError(
                        "battery state disappeared after it was loaded"
                    )
                if (
                    self._loaded_file_sha256 is None
                    or self._loaded_generation is None
                    or self.generation != self._loaded_generation
                ):
                    raise AnchorBatteryConflictError(
                        "controller generation differs from its loaded CAS base"
                    )
                current_raw = _read_stable_state(destination)
                if _sha256_bytes(current_raw) != self._loaded_file_sha256:
                    raise AnchorBatteryConflictError(
                        "battery state advanced after this controller loaded"
                    )
            next_generation = self.generation + 1
            document = self._document_at_generation(next_generation)
            raw = _canonical_json(document) + b"\n"
            if len(raw) > _MAX_STATE_BYTES:
                raise AnchorBatteryIntegrityError(
                    "battery state exceeds its byte limit"
                )
            descriptor, temporary_name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".pending",
            )
            temporary = Path(temporary_name)
            try:
                os.fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, destination)
                directory_fd = os.open(
                    destination.parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                temporary.unlink(missing_ok=True)
            committed = _read_stable_state(destination)
            if committed != raw:
                raise AnchorBatteryIntegrityError("battery state commit mismatch")
            committed_sha256 = _sha256_bytes(raw)
            self.generation = next_generation
            self._loaded_path = destination
            self._loaded_file_sha256 = committed_sha256
            self._loaded_generation = next_generation
            return committed_sha256

    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str],
        *,
        expected_model_pin_sha256: str,
    ) -> "AnchorBatteryController":
        expected = _digest(expected_model_pin_sha256, "expected model pin SHA-256")
        source = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
        raw = _read_stable_state(source)
        if not raw.endswith(b"\n"):
            raise AnchorBatteryIntegrityError("battery state is not canonical JSONL")
        try:
            document = json.loads(
                raw,
                object_pairs_hook=_json_no_duplicates,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    AnchorBatteryIntegrityError(f"invalid JSON constant: {value}")
                ),
            )
        except AnchorBatteryIntegrityError:
            raise
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise AnchorBatteryIntegrityError("battery state is invalid JSON") from exc
        if not isinstance(document, Mapping) or set(document) != {
            "body",
            "body_sha256",
        }:
            raise AnchorBatteryIntegrityError("battery state envelope is invalid")
        if raw != _canonical_json(document) + b"\n":
            raise AnchorBatteryIntegrityError("battery state is not canonical")
        body = document.get("body")
        if not isinstance(body, Mapping) or set(body) != {
            "generation",
            "ledger",
            "miner",
            "model_pin_sha256",
            "schema",
        }:
            raise AnchorBatteryIntegrityError("battery state body is invalid")
        if body.get("schema") != ANCHOR_BATTERY_STATE_SCHEMA:
            raise AnchorBatteryIntegrityError("battery state schema mismatch")
        claimed = _digest(document.get("body_sha256"), "battery body SHA-256")
        if claimed != _sha256_document(body):
            raise AnchorBatteryIntegrityError("battery state SHA-256 mismatch")
        model_pin = _digest(body.get("model_pin_sha256"), "model pin SHA-256")
        if model_pin != expected:
            raise AnchorBatteryIdentityError(
                "battery state belongs to another model pin"
            )
        controller = cls(
            model_pin_sha256=model_pin,
            miner=RadixDemandMiner.from_document(body.get("miner")),
            ledger=ProfitLedger.from_document(body.get("ledger")),
            generation=_nonnegative_int(body.get("generation"), "battery generation"),
        )
        controller._loaded_path = source
        controller._loaded_file_sha256 = _sha256_bytes(raw)
        controller._loaded_generation = controller.generation
        return controller


def _read_stable_state(path: Path) -> bytes:
    """Read one bounded regular inode without following path replacements."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AnchorBatteryIntegrityError("cannot open battery state") from exc
    try:
        before = os.fstat(descriptor)
        linked_before = os.lstat(path)
        if (
            not stat.S_ISREG(before.st_mode)
            or not stat.S_ISREG(linked_before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or stat.S_IMODE(linked_before.st_mode) != 0o600
            or (before.st_dev, before.st_ino)
            != (
                linked_before.st_dev,
                linked_before.st_ino,
            )
        ):
            raise AnchorBatteryIntegrityError(
                "battery state is not a stable regular file"
            )
        if before.st_size < 0 or before.st_size > _MAX_STATE_BYTES:
            raise AnchorBatteryIntegrityError("battery state exceeds its byte limit")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise AnchorBatteryIntegrityError("battery state was truncated")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise AnchorBatteryIntegrityError("battery state grew while reading")
        after = os.fstat(descriptor)
        linked_after = os.lstat(path)
        signatures = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_signature = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            signatures != after_signature
            or (after.st_dev, after.st_ino)
            != (linked_after.st_dev, linked_after.st_ino)
            or stat.S_IMODE(after.st_mode) != 0o600
            or stat.S_IMODE(linked_after.st_mode) != 0o600
        ):
            raise AnchorBatteryIntegrityError("battery state changed while reading")
        return b"".join(chunks)
    except OSError as exc:
        raise AnchorBatteryIntegrityError("cannot read stable battery state") from exc
    finally:
        os.close(descriptor)


__all__ = [
    "ANCHOR_BATTERY_STATE_SCHEMA",
    "AnchorBatteryConflictError",
    "AnchorBatteryController",
    "AnchorBatteryError",
    "AnchorBatteryIdentityError",
    "AnchorBatteryIntegrityError",
    "AnchorChargeCandidate",
    "AnchorChargePlan",
    "AnchorChargePolicy",
    "PrefixDemand",
    "PrefixProfit",
    "ProfitLedger",
    "RADIX_DEMAND_SCHEMA",
    "RadixDemandMiner",
    "PROFIT_LEDGER_SCHEMA",
]
