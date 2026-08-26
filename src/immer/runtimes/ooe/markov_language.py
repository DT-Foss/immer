"""Consequence-grounded, context-aware executable Markov language.

The receiver sees an opaque word, an authenticated context identifier, its own
chosen action, and verifier feedback.  It never receives the sender's intent or
the target action.  A shared word/action table learns stable meaning while a
context residual can override that meaning when the same word has different
consequences in different states.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import re
from typing import cast

import numpy as np
from numpy.typing import NDArray

from .identity import canonical_json_bytes, require_sha256


ACTION_FRONTIER_SCHEMA = "immer-ooe-action-frontier/v1"
CONSEQUENCE_FEEDBACK_SCHEMA = "immer-ooe-consequence-feedback/v1"
LANGUAGE_STATE_SCHEMA = "immer-ooe-markov-language-state/v1"
LANGUAGE_SNAPSHOT_SCHEMA = "immer-ooe-markov-language-snapshot/v1"
LANGUAGE_EPISODE_SCHEMA = "immer-ooe-markov-language-episode/v1"
FACTORIZED_GRAMMAR_SCHEMA = "immer-ooe-factorized-language/v1"

ACTION_ARTIFACT_KINDS = frozenset(
    {
        "causal-site",
        "crystal",
        "macro-option",
        "materialized-route",
        "organ",
        "program",
        "qwen-path",
        "residual-plan",
    }
)

MAX_ACTIONS = 1_024
MAX_WORDS = 4_096
MAX_CONTEXTS = 4_096
MAX_PENDING_DECISIONS = 65_536
MAX_SLOTS = 16
MAX_SLOT_VALUES = 1_024
MAX_STATE_BYTES = 256 * 1024 * 1024

_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,191}")


class MarkovLanguageError(RuntimeError):
    """Base error for the consequence-language runtime."""


class MarkovLanguageIntegrityError(MarkovLanguageError):
    """A sealed language object failed reconstruction or contract binding."""


class MarkovLanguageConflictError(MarkovLanguageError):
    """A pending decision was stale, replayed, or bound to another episode."""


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical identifier")
    return value


def _hash_items(
    value: Mapping[str, str] | Sequence[tuple[str, str]],
    *,
    field: str,
) -> tuple[tuple[str, str], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    normalized = tuple(
        sorted(
            (
                _identifier(name, field=f"{field} name"),
                require_sha256(digest, field=f"{field}[{name!r}]"),
            )
            for name, digest in items
        )
    )
    if len(normalized) > 1_024 or len({name for name, _ in normalized}) != len(
        normalized
    ):
        raise ValueError(f"{field} must be bounded and uniquely named")
    return normalized


def _strict_json(data: bytes, *, label: str) -> object:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if len(data) > MAX_STATE_BYTES:
        raise MarkovLanguageIntegrityError(f"{label} exceeds its byte bound")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise MarkovLanguageIntegrityError(f"{label} is invalid JSON") from exc
    if canonical_json_bytes(value) != data:
        raise MarkovLanguageIntegrityError(f"{label} is not canonical JSON")
    return value


def _body_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _finite(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{field} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _probability(value: object, *, field: str) -> float:
    result = _finite(value, field=field)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} must lie in [0, 1]")
    return result


def _positive_int(value: object, *, field: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 1 <= int(value) <= maximum
    ):
        raise ValueError(f"{field} must lie in [1, {maximum}]")
    return int(value)


def _argmax_random(values: NDArray[np.float64], rng: np.random.Generator) -> int:
    maximum = float(np.max(values))
    candidates = np.flatnonzero(np.abs(values - maximum) <= 1e-12)
    return int(candidates[int(rng.integers(len(candidates)))])


def _float_matrix_record(value: NDArray[np.float64]) -> list[list[str]]:
    return [[float(item).hex() for item in row] for row in value]


def _float_matrix_from_record(
    value: object,
    *,
    shape: tuple[int, int],
    field: str,
) -> NDArray[np.float64]:
    if not isinstance(value, list) or len(value) != shape[0]:
        raise MarkovLanguageIntegrityError(f"{field} row count is invalid")
    result = np.empty(shape, dtype=np.float64)
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != shape[1]:
            raise MarkovLanguageIntegrityError(f"{field} column count is invalid")
        for column_index, item in enumerate(row):
            if not isinstance(item, str):
                raise MarkovLanguageIntegrityError(f"{field} value is invalid")
            try:
                decoded = float.fromhex(item)
            except ValueError as exc:
                raise MarkovLanguageIntegrityError(
                    f"{field} contains an invalid float"
                ) from exc
            if not math.isfinite(decoded) or decoded.hex() != item:
                raise MarkovLanguageIntegrityError(
                    f"{field} contains a non-canonical float"
                )
            result[row_index, column_index] = decoded
    return result


def _int_matrix_record(value: NDArray[np.int64]) -> list[list[int]]:
    return [[int(item) for item in row] for row in value]


def _int_matrix_from_record(
    value: object,
    *,
    shape: tuple[int, int],
    field: str,
) -> NDArray[np.int64]:
    if not isinstance(value, list) or len(value) != shape[0]:
        raise MarkovLanguageIntegrityError(f"{field} row count is invalid")
    result = np.empty(shape, dtype=np.int64)
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != shape[1]:
            raise MarkovLanguageIntegrityError(f"{field} column count is invalid")
        for column_index, item in enumerate(row):
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise MarkovLanguageIntegrityError(f"{field} value is invalid")
            result[row_index, column_index] = item
    return result


@dataclass(frozen=True, slots=True)
class ActionBinding:
    action_id: str
    artifact_kind: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "action_id", _identifier(self.action_id, field="action_id")
        )
        if self.artifact_kind not in ACTION_ARTIFACT_KINDS:
            raise ValueError("unsupported action artifact kind")
        object.__setattr__(
            self,
            "artifact_sha256",
            require_sha256(self.artifact_sha256, field="artifact_sha256"),
        )

    def to_record(self) -> dict[str, str]:
        return {
            "action_id": self.action_id,
            "artifact_kind": self.artifact_kind,
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_record(cls, value: object) -> "ActionBinding":
        if not isinstance(value, Mapping) or set(value) != {
            "action_id",
            "artifact_kind",
            "artifact_sha256",
        }:
            raise MarkovLanguageIntegrityError("action binding is invalid")
        try:
            return cls(
                action_id=cast(str, value.get("action_id")),
                artifact_kind=cast(str, value.get("artifact_kind")),
                artifact_sha256=cast(str, value.get("artifact_sha256")),
            )
        except (TypeError, ValueError) as exc:
            raise MarkovLanguageIntegrityError("action binding is invalid") from exc


@dataclass(frozen=True, slots=True)
class ActionFrontier:
    actions: tuple[ActionBinding, ...]
    action_schema_sha256: str
    context_schema_sha256: str
    authority_hashes: tuple[tuple[str, str], ...] = ()
    allow_duplicate_artifacts: bool = False

    FORMAT = ACTION_FRONTIER_SCHEMA

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        if not 1 <= len(actions) <= MAX_ACTIONS or any(
            not isinstance(item, ActionBinding) for item in actions
        ):
            raise ValueError("action frontier is invalid")
        if tuple(sorted(actions, key=lambda item: item.action_id)) != actions:
            raise ValueError("action frontier must be sorted by action_id")
        if len({item.action_id for item in actions}) != len(actions):
            raise ValueError("action frontier contains duplicate action IDs")
        if not isinstance(self.allow_duplicate_artifacts, bool):
            raise TypeError("allow_duplicate_artifacts must be bool")
        artifacts = tuple(
            (item.artifact_kind, item.artifact_sha256) for item in actions
        )
        if not self.allow_duplicate_artifacts and len(set(artifacts)) != len(artifacts):
            raise ValueError("action frontier contains duplicate executable artifacts")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(
            self,
            "action_schema_sha256",
            require_sha256(self.action_schema_sha256, field="action_schema_sha256"),
        )
        object.__setattr__(
            self,
            "context_schema_sha256",
            require_sha256(self.context_schema_sha256, field="context_schema_sha256"),
        )
        object.__setattr__(
            self,
            "authority_hashes",
            _hash_items(self.authority_hashes, field="authority_hashes"),
        )

    @classmethod
    def create(
        cls,
        actions: Sequence[ActionBinding],
        *,
        action_schema_sha256: str,
        context_schema_sha256: str,
        authority_hashes: Mapping[str, str] | None = None,
        allow_duplicate_artifacts: bool = False,
    ) -> "ActionFrontier":
        return cls(
            actions=tuple(sorted(actions, key=lambda item: item.action_id)),
            action_schema_sha256=action_schema_sha256,
            context_schema_sha256=context_schema_sha256,
            authority_hashes=_hash_items(
                {} if authority_hashes is None else authority_hashes,
                field="authority_hashes",
            ),
            allow_duplicate_artifacts=allow_duplicate_artifacts,
        )

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(item.action_id for item in self.actions)

    def binding(self, action_id: str) -> ActionBinding:
        wanted = _identifier(action_id, field="action_id")
        for item in self.actions:
            if item.action_id == wanted:
                return item
        raise KeyError(f"action is outside the frontier: {wanted}")

    def to_record(self) -> dict[str, object]:
        return {
            "actions": [item.to_record() for item in self.actions],
            "action_schema_sha256": self.action_schema_sha256,
            "allow_duplicate_artifacts": self.allow_duplicate_artifacts,
            "authority_hashes": dict(self.authority_hashes),
            "context_schema_sha256": self.context_schema_sha256,
            "format": self.FORMAT,
        }

    @classmethod
    def from_record(cls, value: object) -> "ActionFrontier":
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "actions",
                "action_schema_sha256",
                "allow_duplicate_artifacts",
                "authority_hashes",
                "context_schema_sha256",
                "format",
            }
            or value.get("format") != cls.FORMAT
        ):
            raise MarkovLanguageIntegrityError("action frontier is invalid")
        raw_actions = value.get("actions")
        raw_authorities = value.get("authority_hashes")
        if not isinstance(raw_actions, list) or not isinstance(
            raw_authorities, Mapping
        ):
            raise MarkovLanguageIntegrityError("action frontier inventory is invalid")
        try:
            return cls(
                actions=tuple(ActionBinding.from_record(item) for item in raw_actions),
                action_schema_sha256=cast(str, value.get("action_schema_sha256")),
                context_schema_sha256=cast(str, value.get("context_schema_sha256")),
                authority_hashes=_hash_items(
                    cast(Mapping[str, str], raw_authorities),
                    field="authority_hashes",
                ),
                allow_duplicate_artifacts=cast(
                    bool, value.get("allow_duplicate_artifacts")
                ),
            )
        except (TypeError, ValueError) as exc:
            raise MarkovLanguageIntegrityError("action frontier is invalid") from exc

    @property
    def sha256(self) -> str:
        return _body_sha256(self.to_record())


@dataclass(frozen=True, slots=True)
class SenderEmission:
    frontier_sha256: str
    intent_action_id: str
    word_id: str
    sequence: int

    FORMAT = "immer-ooe-sender-emission/v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frontier_sha256",
            require_sha256(self.frontier_sha256, field="frontier_sha256"),
        )
        object.__setattr__(
            self,
            "intent_action_id",
            _identifier(self.intent_action_id, field="intent_action_id"),
        )
        object.__setattr__(self, "word_id", _identifier(self.word_id, field="word_id"))
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a non-negative integer")

    def to_record(self) -> dict[str, object]:
        return {
            "format": self.FORMAT,
            "frontier_sha256": self.frontier_sha256,
            "intent_action_id": self.intent_action_id,
            "sequence": self.sequence,
            "word_id": self.word_id,
        }

    @property
    def sha256(self) -> str:
        return _body_sha256(self.to_record())


@dataclass(frozen=True, slots=True)
class ReceiverDecision:
    frontier_sha256: str
    word_id: str
    context_id: str
    action_id: str
    sequence: int

    FORMAT = "immer-ooe-receiver-decision/v1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frontier_sha256",
            require_sha256(self.frontier_sha256, field="frontier_sha256"),
        )
        for field_name in ("word_id", "context_id", "action_id"):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), field=field_name),
            )
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a non-negative integer")

    def to_record(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "context_id": self.context_id,
            "format": self.FORMAT,
            "frontier_sha256": self.frontier_sha256,
            "sequence": self.sequence,
            "word_id": self.word_id,
        }

    @classmethod
    def from_record(cls, value: object) -> "ReceiverDecision":
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "action_id",
                "context_id",
                "format",
                "frontier_sha256",
                "sequence",
                "word_id",
            }
            or value.get("format") != cls.FORMAT
        ):
            raise MarkovLanguageIntegrityError("receiver decision is invalid")
        try:
            return cls(
                frontier_sha256=cast(str, value.get("frontier_sha256")),
                word_id=cast(str, value.get("word_id")),
                context_id=cast(str, value.get("context_id")),
                action_id=cast(str, value.get("action_id")),
                sequence=cast(int, value.get("sequence")),
            )
        except (TypeError, ValueError) as exc:
            raise MarkovLanguageIntegrityError("receiver decision is invalid") from exc

    @property
    def sha256(self) -> str:
        return _body_sha256(self.to_record())


@dataclass(frozen=True, slots=True)
class SenderDecision:
    emission: SenderEmission
    receiver_decision_sha256: str

    FORMAT = "immer-ooe-sender-decision/v1"

    def __post_init__(self) -> None:
        if not isinstance(self.emission, SenderEmission):
            raise TypeError("emission must be a SenderEmission")
        object.__setattr__(
            self,
            "receiver_decision_sha256",
            require_sha256(
                self.receiver_decision_sha256,
                field="receiver_decision_sha256",
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "emission": self.emission.to_record(),
            "format": self.FORMAT,
            "receiver_decision_sha256": self.receiver_decision_sha256,
        }

    @property
    def sha256(self) -> str:
        return _body_sha256(self.to_record())


@dataclass(frozen=True, slots=True)
class ConsequenceFeedback:
    receiver_decision_sha256: str
    frontier_sha256: str
    outcome_receipt_sha256: str
    reward: float
    accepted: bool

    FORMAT = CONSEQUENCE_FEEDBACK_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "receiver_decision_sha256",
            "frontier_sha256",
            "outcome_receipt_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        object.__setattr__(self, "reward", _finite(self.reward, field="reward"))
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be bool")

    @classmethod
    def for_decision(
        cls,
        decision: ReceiverDecision,
        *,
        reward: float,
        accepted: bool,
        outcome_receipt_sha256: str,
    ) -> "ConsequenceFeedback":
        if not isinstance(decision, ReceiverDecision):
            raise TypeError("decision must be a ReceiverDecision")
        return cls(
            receiver_decision_sha256=decision.sha256,
            frontier_sha256=decision.frontier_sha256,
            outcome_receipt_sha256=outcome_receipt_sha256,
            reward=reward,
            accepted=accepted,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "format": self.FORMAT,
            "frontier_sha256": self.frontier_sha256,
            "outcome_receipt_sha256": self.outcome_receipt_sha256,
            "receiver_decision_sha256": self.receiver_decision_sha256,
            "reward_hex": self.reward.hex(),
        }

    @classmethod
    def from_record(cls, value: object) -> "ConsequenceFeedback":
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "accepted",
                "format",
                "frontier_sha256",
                "outcome_receipt_sha256",
                "receiver_decision_sha256",
                "reward_hex",
            }
            or value.get("format") != cls.FORMAT
        ):
            raise MarkovLanguageIntegrityError("consequence feedback is invalid")
        reward_hex = value.get("reward_hex")
        if not isinstance(reward_hex, str):
            raise MarkovLanguageIntegrityError("feedback reward is invalid")
        try:
            reward = float.fromhex(reward_hex)
            if reward.hex() != reward_hex:
                raise ValueError("non-canonical float")
            return cls(
                receiver_decision_sha256=cast(
                    str, value.get("receiver_decision_sha256")
                ),
                frontier_sha256=cast(str, value.get("frontier_sha256")),
                outcome_receipt_sha256=cast(str, value.get("outcome_receipt_sha256")),
                reward=reward,
                accepted=cast(bool, value.get("accepted")),
            )
        except (TypeError, ValueError) as exc:
            raise MarkovLanguageIntegrityError(
                "consequence feedback is invalid"
            ) from exc

    @property
    def sha256(self) -> str:
        return _body_sha256(self.to_record())


@dataclass(frozen=True, slots=True)
class LanguageEpisodeReceipt:
    sender_decision_sha256: str
    receiver_decision_sha256: str
    feedback_sha256: str
    state_before_sha256: str
    state_after_sha256: str

    FORMAT = LANGUAGE_EPISODE_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "sender_decision_sha256",
            "receiver_decision_sha256",
            "feedback_sha256",
            "state_before_sha256",
            "state_after_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )

    def to_record(self) -> dict[str, str]:
        return {
            "feedback_sha256": self.feedback_sha256,
            "format": self.FORMAT,
            "receiver_decision_sha256": self.receiver_decision_sha256,
            "sender_decision_sha256": self.sender_decision_sha256,
            "state_after_sha256": self.state_after_sha256,
            "state_before_sha256": self.state_before_sha256,
        }

    @property
    def sha256(self) -> str:
        return _body_sha256(self.to_record())


class ConsequenceMarkovLanguage:
    """Separate sender/receiver policies joined only by consequence feedback."""

    def __init__(
        self,
        frontier: ActionFrontier,
        vocabulary: Sequence[str],
        *,
        seed: int = 0,
        minimum_visits: int = 8,
        minimum_value: float = 0.25,
        minimum_margin: float = 0.08,
        context_full_weight_visits: int = 16,
    ) -> None:
        if not isinstance(frontier, ActionFrontier):
            raise TypeError("frontier must be an ActionFrontier")
        words = tuple(_identifier(value, field="word_id") for value in vocabulary)
        if not len(frontier.actions) <= len(words) <= MAX_WORDS:
            raise ValueError("vocabulary must cover the bounded action frontier")
        if len(set(words)) != len(words):
            raise ValueError("vocabulary words must be unique")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self.frontier = frontier
        self.vocabulary = words
        self.seed = seed
        self.minimum_visits = _positive_int(
            minimum_visits, field="minimum_visits", maximum=1_000_000
        )
        self.minimum_value = _finite(minimum_value, field="minimum_value")
        self.minimum_margin = _finite(minimum_margin, field="minimum_margin")
        if self.minimum_margin < 0.0:
            raise ValueError("minimum_margin must be non-negative")
        self.context_full_weight_visits = _positive_int(
            context_full_weight_visits,
            field="context_full_weight_visits",
            maximum=1_000_000,
        )
        self.rng = np.random.default_rng(seed)
        action_count = len(frontier.actions)
        word_count = len(words)
        self.sender_q = self.rng.normal(0.0, 0.01, size=(action_count, word_count))
        self.receiver_q = self.rng.normal(0.0, 0.01, size=(word_count, action_count))
        self.receiver_visits = np.zeros((word_count, action_count), dtype=np.int64)
        self.context_q: dict[str, NDArray[np.float64]] = {}
        self.context_visits: dict[str, NDArray[np.int64]] = {}
        self._pending_emissions: dict[str, SenderEmission] = {}
        self._pending_receivers: dict[str, ReceiverDecision] = {}
        self._pending_senders: dict[str, SenderDecision] = {}
        self._sequence = 0
        self.training_episodes = 0
        self._transition_head_sha256 = _body_sha256(
            {
                "format": "immer-ooe-markov-language-transition-root/v1",
                "frontier_sha256": frontier.sha256,
                "seed": seed,
                "vocabulary": list(words),
            }
        )

    @property
    def action_ids(self) -> tuple[str, ...]:
        return self.frontier.action_ids

    def _action_index(self, action_id: str) -> int:
        wanted = _identifier(action_id, field="action_id")
        try:
            return self.action_ids.index(wanted)
        except ValueError as exc:
            raise KeyError(f"action is outside the frontier: {wanted}") from exc

    def _word_index(self, word_id: str) -> int:
        wanted = _identifier(word_id, field="word_id")
        try:
            return self.vocabulary.index(wanted)
        except ValueError as exc:
            raise KeyError(f"word is outside the vocabulary: {wanted}") from exc

    def _next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

    def _ensure_context(
        self, context_id: str
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        context = _identifier(context_id, field="context_id")
        if context not in self.context_q:
            if len(self.context_q) >= MAX_CONTEXTS:
                raise MarkovLanguageError("context table is full")
            self.context_q[context] = np.zeros_like(self.receiver_q)
            self.context_visits[context] = np.zeros_like(self.receiver_visits)
        return self.context_q[context], self.context_visits[context]

    def _receiver_scores(self, word_index: int, context_id: str) -> NDArray[np.float64]:
        context = _identifier(context_id, field="context_id")
        global_scores = self.receiver_q[word_index]
        local_q = self.context_q.get(context)
        local_visits = self.context_visits.get(context)
        if local_q is None or local_visits is None:
            return global_scores.copy()
        visits = int(np.sum(local_visits[word_index]))
        weight = min(1.0, visits / self.context_full_weight_visits)
        return (1.0 - weight) * global_scores + weight * local_q[word_index]

    def emit(self, intent_action_id: str, *, epsilon: float = 0.0) -> SenderEmission:
        exploration = _probability(epsilon, field="epsilon")
        action_index = self._action_index(intent_action_id)
        if self.rng.random() < exploration:
            word_index = int(self.rng.integers(len(self.vocabulary)))
        else:
            word_index = _argmax_random(self.sender_q[action_index], self.rng)
        if len(self._pending_emissions) >= MAX_PENDING_DECISIONS:
            raise MarkovLanguageError("pending emission table is full")
        word_id = self.vocabulary[word_index]
        if any(
            pending.frontier_sha256 == self.frontier.sha256
            and pending.word_id == word_id
            for pending in self._pending_emissions.values()
        ):
            raise MarkovLanguageConflictError(
                "an opaque word already has an uncommitted sender emission"
            )
        emission = SenderEmission(
            frontier_sha256=self.frontier.sha256,
            intent_action_id=self.action_ids[action_index],
            word_id=word_id,
            sequence=self._next_sequence(),
        )
        self._pending_emissions[emission.sha256] = emission
        return emission

    def receiver_decide(
        self,
        word_id: str,
        context_id: str,
        *,
        epsilon: float = 0.0,
    ) -> ReceiverDecision:
        """Choose from receiver-visible information only.

        This method intentionally has no intent, target action, target state, or
        semantic-label parameter.
        """

        exploration = _probability(epsilon, field="epsilon")
        word_index = self._word_index(word_id)
        context = _identifier(context_id, field="context_id")
        if self.rng.random() < exploration:
            action_index = int(self.rng.integers(len(self.action_ids)))
        else:
            action_index = _argmax_random(
                self._receiver_scores(word_index, context), self.rng
            )
        decision = ReceiverDecision(
            frontier_sha256=self.frontier.sha256,
            word_id=self.vocabulary[word_index],
            context_id=context,
            action_id=self.action_ids[action_index],
            sequence=self._next_sequence(),
        )
        if len(self._pending_receivers) >= MAX_PENDING_DECISIONS:
            raise MarkovLanguageError("pending receiver table is full")
        self._pending_receivers[decision.sha256] = decision
        return decision

    def bind_sender(
        self,
        emission: SenderEmission,
        decision: ReceiverDecision,
    ) -> SenderDecision:
        if not isinstance(emission, SenderEmission) or not isinstance(
            decision, ReceiverDecision
        ):
            raise TypeError("emission and decision have invalid types")
        if (
            self._pending_emissions.get(emission.sha256) != emission
            or emission.frontier_sha256 != self.frontier.sha256
            or decision.frontier_sha256 != self.frontier.sha256
            or emission.word_id != decision.word_id
            or decision.sha256 not in self._pending_receivers
        ):
            raise MarkovLanguageConflictError(
                "sender emission and receiver decision do not share an episode"
            )
        bound = SenderDecision(emission, decision.sha256)
        if len(self._pending_senders) >= MAX_PENDING_DECISIONS:
            raise MarkovLanguageError("pending sender table is full")
        self._pending_senders[bound.sha256] = bound
        del self._pending_emissions[emission.sha256]
        return bound

    def abort_pending_episode(
        self,
        emission: SenderEmission,
        decision: ReceiverDecision | None = None,
    ) -> None:
        """Discard an uncommitted episode without turning it into feedback."""

        if not isinstance(emission, SenderEmission):
            raise TypeError("emission must be a SenderEmission")
        pending_emission = self._pending_emissions.get(emission.sha256)
        if pending_emission == emission:
            del self._pending_emissions[emission.sha256]
        if decision is not None:
            if not isinstance(decision, ReceiverDecision):
                raise TypeError("decision must be a ReceiverDecision")
            pending_receiver = self._pending_receivers.get(decision.sha256)
            if pending_receiver == decision:
                del self._pending_receivers[decision.sha256]

    def observe_receiver(
        self,
        decision: ReceiverDecision,
        feedback: ConsequenceFeedback,
        *,
        learning_rate: float = 0.15,
    ) -> None:
        rate = _finite(learning_rate, field="learning_rate")
        if not 0.0 < rate <= 1.0:
            raise ValueError("learning_rate must lie in (0, 1]")
        if not isinstance(decision, ReceiverDecision) or not isinstance(
            feedback, ConsequenceFeedback
        ):
            raise TypeError("decision and feedback have invalid types")
        pending = self._pending_receivers.get(decision.sha256)
        if (
            pending != decision
            or feedback.receiver_decision_sha256 != decision.sha256
            or feedback.frontier_sha256 != self.frontier.sha256
        ):
            raise MarkovLanguageConflictError("receiver feedback is stale or replayed")
        word_index = self._word_index(decision.word_id)
        action_index = self._action_index(decision.action_id)
        context_q, context_visits = self._ensure_context(decision.context_id)
        old = self.receiver_q[word_index, action_index]
        self.receiver_q[word_index, action_index] = old + rate * (feedback.reward - old)
        old_context = context_q[word_index, action_index]
        context_q[word_index, action_index] = old_context + rate * (
            feedback.reward - old_context
        )
        self.receiver_visits[word_index, action_index] += 1
        context_visits[word_index, action_index] += 1
        del self._pending_receivers[decision.sha256]

    def observe_sender(
        self,
        decision: SenderDecision,
        feedback: ConsequenceFeedback,
        *,
        learning_rate: float = 0.15,
    ) -> None:
        rate = _finite(learning_rate, field="learning_rate")
        if not 0.0 < rate <= 1.0:
            raise ValueError("learning_rate must lie in (0, 1]")
        if not isinstance(decision, SenderDecision) or not isinstance(
            feedback, ConsequenceFeedback
        ):
            raise TypeError("decision and feedback have invalid types")
        pending = self._pending_senders.get(decision.sha256)
        if (
            pending != decision
            or feedback.receiver_decision_sha256 != decision.receiver_decision_sha256
            or feedback.frontier_sha256 != self.frontier.sha256
        ):
            raise MarkovLanguageConflictError("sender feedback is stale or replayed")
        action_index = self._action_index(decision.emission.intent_action_id)
        word_index = self._word_index(decision.emission.word_id)
        old = self.sender_q[action_index, word_index]
        self.sender_q[action_index, word_index] = old + rate * (feedback.reward - old)
        del self._pending_senders[decision.sha256]
        self.training_episodes += 1

    def run_episode(
        self,
        intent_action_id: str,
        context_id: str,
        consequence: Callable[[str], tuple[float, bool, str]],
        *,
        epsilon: float,
        learning_rate: float = 0.15,
    ) -> LanguageEpisodeReceipt:
        """Run one episode; ``consequence`` receives only the chosen action ID."""

        if not callable(consequence):
            raise TypeError("consequence must be callable")
        before = self._transition_head_sha256
        emission = self.emit(intent_action_id, epsilon=epsilon)
        receiver = self.receiver_decide(emission.word_id, context_id, epsilon=epsilon)
        sender = self.bind_sender(emission, receiver)
        reward, accepted, outcome_sha256 = consequence(receiver.action_id)
        feedback = ConsequenceFeedback.for_decision(
            receiver,
            reward=reward,
            accepted=accepted,
            outcome_receipt_sha256=outcome_sha256,
        )
        self.observe_receiver(receiver, feedback, learning_rate=learning_rate)
        self.observe_sender(sender, feedback, learning_rate=learning_rate)
        self._transition_head_sha256 = _body_sha256(
            {
                "feedback_sha256": feedback.sha256,
                "format": "immer-ooe-markov-language-transition/v1",
                "previous_sha256": before,
                "receiver_decision_sha256": receiver.sha256,
                "sender_decision_sha256": sender.sha256,
            }
        )
        return LanguageEpisodeReceipt(
            sender_decision_sha256=sender.sha256,
            receiver_decision_sha256=receiver.sha256,
            feedback_sha256=feedback.sha256,
            state_before_sha256=before,
            state_after_sha256=self._transition_head_sha256,
        )

    def decode_word(self, word_id: str, context_id: str) -> str | None:
        try:
            word_index = self._word_index(word_id)
        except KeyError:
            return None
        context = _identifier(context_id, field="context_id")
        scores = self._receiver_scores(word_index, context)
        action_index = int(np.argmax(scores))
        local_visits = self.context_visits.get(context)
        context_total = (
            0 if local_visits is None else int(np.sum(local_visits[word_index]))
        )
        global_total = int(np.sum(self.receiver_visits[word_index]))
        visits = context_total if context_total > 0 else global_total
        ordered = np.sort(scores)
        margin = (
            float(ordered[-1] - ordered[-2]) if len(ordered) > 1 else float(ordered[-1])
        )
        if (
            visits < self.minimum_visits
            or float(scores[action_index]) < self.minimum_value
            or margin < self.minimum_margin
        ):
            return None
        return self.action_ids[action_index]

    def encode_action(self, action_id: str, context_id: str) -> str | None:
        action_index = self._action_index(action_id)
        ordered = np.argsort(-self.sender_q[action_index], kind="stable")
        for word_index in ordered:
            word = self.vocabulary[int(word_index)]
            if self.decode_word(word, context_id) == self.action_ids[action_index]:
                return word
        return None

    def accuracy(self, context_id: str) -> float:
        correct = sum(
            self.encode_action(action_id, context_id) is not None
            for action_id in self.action_ids
        )
        return correct / len(self.action_ids)

    def induce_sender_from_receiver(self, context_id: str) -> None:
        context = _identifier(context_id, field="context_id")
        self.sender_q.fill(-1.0)
        claimed: set[int] = set()
        for action_index, action_id in enumerate(self.action_ids):
            candidates = [
                word_index
                for word_index, word in enumerate(self.vocabulary)
                if word_index not in claimed
                and self.decode_word(word, context) == action_id
            ]
            if not candidates:
                continue
            word_index = max(
                candidates,
                key=lambda index: float(self.receiver_q[index, action_index]),
            )
            self.sender_q[action_index, word_index] = 1.0
            claimed.add(word_index)

    def learn_from_teacher_episode(
        self,
        teacher: "LanguageSnapshot",
        intent_action_id: str,
        context_id: str,
        consequence: Callable[[str], tuple[float, bool, str]],
        *,
        epsilon: float,
        learning_rate: float = 0.15,
    ) -> ConsequenceFeedback:
        if not isinstance(teacher, LanguageSnapshot):
            raise TypeError("teacher must be a LanguageSnapshot")
        if teacher.frontier_sha256 != self.frontier.sha256:
            raise MarkovLanguageConflictError("teacher frontier is stale")
        word = teacher.encode_action(intent_action_id, context_id)
        if word is None:
            raise MarkovLanguageError("teacher cannot encode the requested action")
        receiver = self.receiver_decide(word, context_id, epsilon=epsilon)
        reward, accepted, outcome_sha = consequence(receiver.action_id)
        feedback = ConsequenceFeedback.for_decision(
            receiver,
            reward=reward,
            accepted=accepted,
            outcome_receipt_sha256=outcome_sha,
        )
        self.observe_receiver(receiver, feedback, learning_rate=learning_rate)
        self.training_episodes += 1
        self._transition_head_sha256 = _body_sha256(
            {
                "feedback_sha256": feedback.sha256,
                "format": "immer-ooe-markov-language-cultural-transition/v1",
                "previous_sha256": self._transition_head_sha256,
                "receiver_decision_sha256": receiver.sha256,
                "teacher_snapshot_sha256": teacher.sha256,
            }
        )
        return feedback

    def to_document(self) -> dict[str, object]:
        body = {
            "context_full_weight_visits": self.context_full_weight_visits,
            "contexts": [
                {
                    "context_id": context,
                    "q": _float_matrix_record(self.context_q[context]),
                    "visits": _int_matrix_record(self.context_visits[context]),
                }
                for context in sorted(self.context_q)
            ],
            "frontier": self.frontier.to_record(),
            "minimum_margin_hex": self.minimum_margin.hex(),
            "minimum_value_hex": self.minimum_value.hex(),
            "minimum_visits": self.minimum_visits,
            "pending_receivers": [
                item.to_record() for _, item in sorted(self._pending_receivers.items())
            ],
            "receiver_q": _float_matrix_record(self.receiver_q),
            "receiver_visits": _int_matrix_record(self.receiver_visits),
            "rng_state": self.rng.bit_generator.state,
            "seed": self.seed,
            "sender_q": _float_matrix_record(self.sender_q),
            "sequence": self._sequence,
            "training_episodes": self.training_episodes,
            "transition_head_sha256": self._transition_head_sha256,
            "vocabulary": list(self.vocabulary),
        }
        return {
            "body": body,
            "body_sha256": _body_sha256(body),
            "schema": LANGUAGE_STATE_SCHEMA,
        }

    def to_bytes(self) -> bytes:
        if self._pending_emissions or self._pending_receivers or self._pending_senders:
            raise MarkovLanguageConflictError(
                "serialize only after the complete episode commits or aborts"
            )
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_STATE_BYTES:
            raise MarkovLanguageError("language state exceeds its byte bound")
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        expected_frontier: ActionFrontier | None = None,
    ) -> "ConsequenceMarkovLanguage":
        value = _strict_json(data, label="Markov language state")
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "body",
                "body_sha256",
                "schema",
            }
            or value.get("schema") != LANGUAGE_STATE_SCHEMA
        ):
            raise MarkovLanguageIntegrityError("language state envelope is invalid")
        body = value.get("body")
        if not isinstance(body, Mapping) or value.get("body_sha256") != _body_sha256(
            cast(Mapping[str, object], body)
        ):
            raise MarkovLanguageIntegrityError("language state hash is invalid")
        expected_fields = {
            "context_full_weight_visits",
            "contexts",
            "frontier",
            "minimum_margin_hex",
            "minimum_value_hex",
            "minimum_visits",
            "pending_receivers",
            "receiver_q",
            "receiver_visits",
            "rng_state",
            "seed",
            "sender_q",
            "sequence",
            "training_episodes",
            "transition_head_sha256",
            "vocabulary",
        }
        if set(body) != expected_fields:
            raise MarkovLanguageIntegrityError("language state fields are invalid")
        try:
            frontier = ActionFrontier.from_record(body.get("frontier"))
            if expected_frontier is not None and frontier != expected_frontier:
                raise MarkovLanguageConflictError("language frontier is stale")
            vocabulary = body.get("vocabulary")
            if not isinstance(vocabulary, list):
                raise MarkovLanguageIntegrityError("vocabulary is invalid")
            minimum_value_hex = body.get("minimum_value_hex")
            minimum_margin_hex = body.get("minimum_margin_hex")
            if not isinstance(minimum_value_hex, str) or not isinstance(
                minimum_margin_hex, str
            ):
                raise MarkovLanguageIntegrityError("threshold encoding is invalid")
            learner = cls(
                frontier,
                tuple(cast(list[str], vocabulary)),
                seed=cast(int, body.get("seed")),
                minimum_visits=cast(int, body.get("minimum_visits")),
                minimum_value=float.fromhex(minimum_value_hex),
                minimum_margin=float.fromhex(minimum_margin_hex),
                context_full_weight_visits=cast(
                    int, body.get("context_full_weight_visits")
                ),
            )
            shape = (len(learner.vocabulary), len(learner.action_ids))
            learner.sender_q = _float_matrix_from_record(
                body.get("sender_q"),
                shape=(shape[1], shape[0]),
                field="sender_q",
            )
            learner.receiver_q = _float_matrix_from_record(
                body.get("receiver_q"), shape=shape, field="receiver_q"
            )
            learner.receiver_visits = _int_matrix_from_record(
                body.get("receiver_visits"),
                shape=shape,
                field="receiver_visits",
            )
            contexts = body.get("contexts")
            if not isinstance(contexts, list) or len(contexts) > MAX_CONTEXTS:
                raise MarkovLanguageIntegrityError("context inventory is invalid")
            learner.context_q.clear()
            learner.context_visits.clear()
            for raw in contexts:
                if not isinstance(raw, Mapping) or set(raw) != {
                    "context_id",
                    "q",
                    "visits",
                }:
                    raise MarkovLanguageIntegrityError("context record is invalid")
                context = _identifier(raw.get("context_id"), field="context_id")
                if context in learner.context_q:
                    raise MarkovLanguageIntegrityError("duplicate context record")
                learner.context_q[context] = _float_matrix_from_record(
                    raw.get("q"), shape=shape, field="context q"
                )
                learner.context_visits[context] = _int_matrix_from_record(
                    raw.get("visits"), shape=shape, field="context visits"
                )
            pending = body.get("pending_receivers")
            if not isinstance(pending, list) or len(pending) > MAX_PENDING_DECISIONS:
                raise MarkovLanguageIntegrityError("pending decisions are invalid")
            if pending:
                raise MarkovLanguageIntegrityError(
                    "serialized language contains a partial episode"
                )
            learner._pending_emissions = {}
            learner._pending_receivers = {}
            learner._pending_senders = {}
            sequence = body.get("sequence")
            episodes = body.get("training_episodes")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
                or isinstance(episodes, bool)
                or not isinstance(episodes, int)
                or episodes < 0
            ):
                raise MarkovLanguageIntegrityError("language counters are invalid")
            learner._sequence = sequence
            learner.training_episodes = episodes
            learner._transition_head_sha256 = require_sha256(
                cast(str, body.get("transition_head_sha256")),
                field="transition_head_sha256",
            )
            rng_state = body.get("rng_state")
            if not isinstance(rng_state, Mapping):
                raise MarkovLanguageIntegrityError("RNG state is invalid")
            learner.rng.bit_generator.state = dict(rng_state)
        except MarkovLanguageError:
            raise
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise MarkovLanguageIntegrityError(
                "language state reconstruction failed"
            ) from exc
        if learner.to_bytes() != data:
            raise MarkovLanguageIntegrityError(
                "language state failed canonical reconstruction"
            )
        return learner


@dataclass(frozen=True, slots=True)
class LanguageSnapshot:
    frontier_sha256: str
    learner_state_sha256: str
    vocabulary: tuple[str, ...]
    sender_action_words: tuple[tuple[str, str], ...]
    global_word_actions: tuple[tuple[str, str], ...]
    context_word_actions: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    context_action_words: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]

    FORMAT = LANGUAGE_SNAPSHOT_SCHEMA

    def __post_init__(self) -> None:
        for field_name in ("frontier_sha256", "learner_state_sha256"):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        vocabulary = tuple(
            _identifier(item, field="word_id") for item in self.vocabulary
        )
        if not vocabulary or len(set(vocabulary)) != len(vocabulary):
            raise ValueError("snapshot vocabulary is invalid")
        object.__setattr__(self, "vocabulary", vocabulary)
        sender = tuple(
            sorted(
                (
                    _identifier(action, field="action_id"),
                    _identifier(word, field="word_id"),
                )
                for action, word in self.sender_action_words
            )
        )
        global_map = tuple(
            sorted(
                (
                    _identifier(word, field="word_id"),
                    _identifier(action, field="action_id"),
                )
                for word, action in self.global_word_actions
            )
        )
        contexts = tuple(
            sorted(
                (
                    _identifier(context, field="context_id"),
                    tuple(
                        sorted(
                            (
                                _identifier(word, field="word_id"),
                                _identifier(action, field="action_id"),
                            )
                            for word, action in rows
                        )
                    ),
                )
                for context, rows in self.context_word_actions
            )
        )
        context_senders = tuple(
            sorted(
                (
                    _identifier(context, field="context_id"),
                    tuple(
                        sorted(
                            (
                                _identifier(action, field="action_id"),
                                _identifier(word, field="word_id"),
                            )
                            for action, word in rows
                        )
                    ),
                )
                for context, rows in self.context_action_words
            )
        )
        for rows, field in ((sender, "sender"), (global_map, "global")):
            if len({row[0] for row in rows}) != len(rows):
                raise ValueError(f"snapshot {field} map contains duplicate keys")
        if len({context for context, _ in contexts}) != len(contexts):
            raise ValueError("snapshot contexts contain duplicates")
        if len({context for context, _ in context_senders}) != len(context_senders) or {
            context for context, _ in context_senders
        } != {context for context, _ in contexts}:
            raise ValueError("snapshot sender contexts are invalid")
        if (
            any(word not in vocabulary for _, word in sender)
            or any(word not in vocabulary for word, _ in global_map)
            or any(word not in vocabulary for _, rows in contexts for word, _ in rows)
            or any(
                word not in vocabulary
                for _, rows in context_senders
                for _, word in rows
            )
        ):
            raise ValueError("snapshot mapping references an unknown word")
        if any(len({word for word, _ in rows}) != len(rows) for _, rows in contexts):
            raise ValueError("snapshot context map contains duplicate words")
        if any(
            len({action for action, _ in rows}) != len(rows)
            for _, rows in context_senders
        ):
            raise ValueError("snapshot context sender map contains duplicate actions")
        object.__setattr__(self, "sender_action_words", sender)
        object.__setattr__(self, "global_word_actions", global_map)
        object.__setattr__(self, "context_word_actions", contexts)
        object.__setattr__(self, "context_action_words", context_senders)

    @classmethod
    def freeze(
        cls,
        learner: ConsequenceMarkovLanguage,
        *,
        contexts: Sequence[str],
        require_complete: bool = True,
    ) -> "LanguageSnapshot":
        if not isinstance(learner, ConsequenceMarkovLanguage):
            raise TypeError("learner must be a ConsequenceMarkovLanguage")
        context_ids = tuple(_identifier(item, field="context_id") for item in contexts)
        if not context_ids:
            raise ValueError("snapshot needs at least one declared context")
        context_maps = tuple(
            (
                context,
                tuple(
                    (word, action)
                    for word in learner.vocabulary
                    if (action := learner.decode_word(word, context)) is not None
                ),
            )
            for context in context_ids
        )
        context_senders = tuple(
            (
                context,
                tuple(
                    (action, word)
                    for action in learner.action_ids
                    if (word := learner.encode_action(action, context)) is not None
                ),
            )
            for context in context_ids
        )
        if require_complete and any(
            len(rows) != len(learner.action_ids) for _, rows in context_senders
        ):
            raise MarkovLanguageError("language is not complete enough to freeze")
        global_map_list: list[tuple[str, str]] = []
        for word in learner.vocabulary:
            actions = tuple(
                learner.decode_word(word, context) for context in context_ids
            )
            if actions[0] is not None and len(set(actions)) == 1:
                global_map_list.append((word, cast(str, actions[0])))
        sender = []
        sender_maps = tuple(dict(rows) for _, rows in context_senders)
        for action in learner.action_ids:
            words = tuple(mapping.get(action) for mapping in sender_maps)
            if words[0] is not None and len(set(words)) == 1:
                sender.append((action, cast(str, words[0])))
        return cls(
            frontier_sha256=learner.frontier.sha256,
            learner_state_sha256=learner.sha256,
            vocabulary=learner.vocabulary,
            sender_action_words=tuple(sender),
            global_word_actions=tuple(global_map_list),
            context_word_actions=context_maps,
            context_action_words=context_senders,
        )

    def decode_word(self, word_id: str, context_id: str) -> str | None:
        word = _identifier(word_id, field="word_id")
        context = _identifier(context_id, field="context_id")
        for candidate_context, rows in self.context_word_actions:
            if candidate_context == context:
                return dict(rows).get(word)
        return dict(self.global_word_actions).get(word)

    def encode_action(self, action_id: str, context_id: str) -> str | None:
        action = _identifier(action_id, field="action_id")
        context = _identifier(context_id, field="context_id")
        for candidate_context, rows in self.context_action_words:
            if candidate_context == context:
                word = dict(rows).get(action)
                if word is not None and self.decode_word(word, context) == action:
                    return word
        for candidate_action, word in self.sender_action_words:
            if candidate_action == action and self.decode_word(word, context) == action:
                return word
        return None

    def primitive_artifacts(
        self,
        frontier: ActionFrontier,
        *,
        context_id: str | None = None,
    ) -> dict[str, tuple[str, str]]:
        if not isinstance(frontier, ActionFrontier) or frontier.sha256 != (
            self.frontier_sha256
        ):
            raise MarkovLanguageConflictError("snapshot frontier is stale")
        rows = self.global_word_actions
        if context_id is not None:
            context = _identifier(context_id, field="context_id")
            rows = next(
                (
                    mapping
                    for candidate, mapping in self.context_word_actions
                    if candidate == context
                ),
                self.global_word_actions,
            )
        result: dict[str, tuple[str, str]] = {}
        for word, action_id in rows:
            binding = frontier.binding(action_id)
            result[word] = (binding.artifact_kind, binding.artifact_sha256)
        return result

    def to_document(self) -> dict[str, object]:
        body = {
            "context_action_words": [
                {
                    "action_words": dict(rows),
                    "context_id": context,
                }
                for context, rows in self.context_action_words
            ],
            "context_word_actions": [
                {
                    "context_id": context,
                    "word_actions": dict(rows),
                }
                for context, rows in self.context_word_actions
            ],
            "frontier_sha256": self.frontier_sha256,
            "global_word_actions": dict(self.global_word_actions),
            "learner_state_sha256": self.learner_state_sha256,
            "sender_action_words": dict(self.sender_action_words),
            "vocabulary": list(self.vocabulary),
        }
        return {
            "body": body,
            "body_sha256": _body_sha256(body),
            "schema": self.FORMAT,
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_document())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "LanguageSnapshot":
        value = _strict_json(data, label="language snapshot")
        if (
            not isinstance(value, Mapping)
            or set(value)
            != {
                "body",
                "body_sha256",
                "schema",
            }
            or value.get("schema") != cls.FORMAT
        ):
            raise MarkovLanguageIntegrityError("snapshot envelope is invalid")
        body = value.get("body")
        if (
            not isinstance(body, Mapping)
            or value.get("body_sha256")
            != _body_sha256(cast(Mapping[str, object], body))
            or set(body)
            != {
                "context_word_actions",
                "context_action_words",
                "frontier_sha256",
                "global_word_actions",
                "learner_state_sha256",
                "sender_action_words",
                "vocabulary",
            }
        ):
            raise MarkovLanguageIntegrityError("snapshot body is invalid")
        contexts = body.get("context_word_actions")
        context_senders = body.get("context_action_words")
        global_map = body.get("global_word_actions")
        sender = body.get("sender_action_words")
        vocabulary = body.get("vocabulary")
        if (
            not isinstance(contexts, list)
            or not isinstance(context_senders, list)
            or not isinstance(global_map, Mapping)
            or not isinstance(sender, Mapping)
            or not isinstance(vocabulary, list)
        ):
            raise MarkovLanguageIntegrityError("snapshot inventory is invalid")
        try:
            context_rows = []
            for row in contexts:
                if (
                    not isinstance(row, Mapping)
                    or set(row)
                    != {
                        "context_id",
                        "word_actions",
                    }
                    or not isinstance(row.get("word_actions"), Mapping)
                ):
                    raise MarkovLanguageIntegrityError("snapshot context is invalid")
                context_rows.append(
                    (
                        cast(str, row.get("context_id")),
                        tuple(cast(Mapping[str, str], row.get("word_actions")).items()),
                    )
                )
            context_sender_rows = []
            for row in context_senders:
                if (
                    not isinstance(row, Mapping)
                    or set(row) != {"action_words", "context_id"}
                    or not isinstance(row.get("action_words"), Mapping)
                ):
                    raise MarkovLanguageIntegrityError(
                        "snapshot context sender is invalid"
                    )
                context_sender_rows.append(
                    (
                        cast(str, row.get("context_id")),
                        tuple(cast(Mapping[str, str], row.get("action_words")).items()),
                    )
                )
            snapshot = cls(
                frontier_sha256=cast(str, body.get("frontier_sha256")),
                learner_state_sha256=cast(str, body.get("learner_state_sha256")),
                vocabulary=tuple(cast(list[str], vocabulary)),
                sender_action_words=tuple(cast(Mapping[str, str], sender).items()),
                global_word_actions=tuple(cast(Mapping[str, str], global_map).items()),
                context_word_actions=tuple(context_rows),
                context_action_words=tuple(context_sender_rows),
            )
        except (TypeError, ValueError) as exc:
            raise MarkovLanguageIntegrityError(
                "snapshot reconstruction failed"
            ) from exc
        if snapshot.to_bytes() != data:
            raise MarkovLanguageIntegrityError(
                "snapshot failed canonical reconstruction"
            )
        return snapshot


class FactorizedConsequenceGrammar:
    """Opaque multi-slot grammar updated only by whole-action consequence."""

    def __init__(
        self,
        frontier: ActionFrontier,
        *,
        slot_names: Sequence[str],
        action_factors: Mapping[str, Sequence[str]],
        slot_vocabularies: Sequence[Sequence[str]],
        seed: int = 0,
        minimum_visits: int = 8,
        minimum_value: float = 0.25,
        minimum_margin: float = 0.08,
        context_full_weight_visits: int = 16,
    ) -> None:
        if not isinstance(frontier, ActionFrontier):
            raise TypeError("frontier must be an ActionFrontier")
        slots = tuple(_identifier(value, field="slot_name") for value in slot_names)
        if not 2 <= len(slots) <= MAX_SLOTS or len(set(slots)) != len(slots):
            raise ValueError("grammar needs 2..16 unique slots")
        factors = {
            _identifier(action, field="action_id"): tuple(
                _identifier(value, field="factor_value") for value in values
            )
            for action, values in action_factors.items()
        }
        if set(factors) != set(frontier.action_ids) or any(
            len(values) != len(slots) for values in factors.values()
        ):
            raise ValueError("action factors must cover the complete frontier")
        if len(set(factors.values())) != len(factors):
            raise ValueError("action factor tuples must be unique")
        vocabularies = tuple(
            tuple(_identifier(word, field="slot_word") for word in vocabulary)
            for vocabulary in slot_vocabularies
        )
        if len(vocabularies) != len(slots) or any(
            not vocabulary or len(set(vocabulary)) != len(vocabulary)
            for vocabulary in vocabularies
        ):
            raise ValueError("slot vocabularies are invalid")
        values = tuple(
            tuple(sorted({row[index] for row in factors.values()}))
            for index in range(len(slots))
        )
        if math.prod(len(row) for row in values) != len(factors):
            raise ValueError(
                "factorized grammar requires a complete Cartesian action set"
            )
        if any(
            not 1 <= len(row) <= MAX_SLOT_VALUES or len(vocabularies[index]) < len(row)
            for index, row in enumerate(values)
        ):
            raise ValueError("slot vocabulary does not cover its factor values")
        self.frontier = frontier
        self.slot_names = slots
        self.action_factors = factors
        self.factor_actions = {value: key for key, value in factors.items()}
        self.slot_values = values
        self.slot_vocabularies = vocabularies
        self.rng = np.random.default_rng(seed)
        self.minimum_visits = _positive_int(
            minimum_visits, field="minimum_visits", maximum=1_000_000
        )
        self.minimum_value = _finite(minimum_value, field="minimum_value")
        self.minimum_margin = _finite(minimum_margin, field="minimum_margin")
        self.context_full_weight_visits = _positive_int(
            context_full_weight_visits,
            field="context_full_weight_visits",
            maximum=1_000_000,
        )
        self.sender_q = tuple(
            self.rng.normal(0.0, 0.01, size=(len(values[index]), len(vocabulary)))
            for index, vocabulary in enumerate(vocabularies)
        )
        self.receiver_q = tuple(
            self.rng.normal(0.0, 0.01, size=(len(vocabulary), len(values[index])))
            for index, vocabulary in enumerate(vocabularies)
        )
        self.receiver_visits = tuple(
            np.zeros_like(table, dtype=np.int64) for table in self.receiver_q
        )
        self.context_q: dict[str, tuple[NDArray[np.float64], ...]] = {}
        self.context_visits: dict[str, tuple[NDArray[np.int64], ...]] = {}

    def _ensure_context(self, context_id: str) -> None:
        context = _identifier(context_id, field="context_id")
        if context not in self.context_q:
            if len(self.context_q) >= MAX_CONTEXTS:
                raise MarkovLanguageError("grammar context table is full")
            self.context_q[context] = tuple(
                np.zeros_like(table) for table in self.receiver_q
            )
            self.context_visits[context] = tuple(
                np.zeros_like(table, dtype=np.int64) for table in self.receiver_visits
            )

    def _scores(self, slot: int, word: int, context: str) -> NDArray[np.float64]:
        global_scores = self.receiver_q[slot][word]
        if context not in self.context_q:
            return global_scores.copy()
        visits = int(np.sum(self.context_visits[context][slot][word]))
        weight = min(1.0, visits / self.context_full_weight_visits)
        return (1.0 - weight) * global_scores + weight * self.context_q[context][slot][
            word
        ]

    def run_episode(
        self,
        intent_action_id: str,
        context_id: str,
        consequence: Callable[[str], tuple[float, bool, str]],
        *,
        epsilon: float,
        learning_rate: float = 0.15,
    ) -> tuple[tuple[str, ...], str, ConsequenceFeedback]:
        exploration = _probability(epsilon, field="epsilon")
        rate = _finite(learning_rate, field="learning_rate")
        if not 0.0 < rate <= 1.0:
            raise ValueError("learning_rate must lie in (0, 1]")
        intent = _identifier(intent_action_id, field="intent_action_id")
        try:
            factors = self.action_factors[intent]
        except KeyError as exc:
            raise KeyError("intent action is outside the grammar") from exc
        context = _identifier(context_id, field="context_id")
        self._ensure_context(context)
        word_indices = []
        predicted_indices = []
        for slot, factor in enumerate(factors):
            value_index = self.slot_values[slot].index(factor)
            word_index = (
                int(self.rng.integers(len(self.slot_vocabularies[slot])))
                if self.rng.random() < exploration
                else _argmax_random(self.sender_q[slot][value_index], self.rng)
            )
            predicted_index = (
                int(self.rng.integers(len(self.slot_values[slot])))
                if self.rng.random() < exploration
                else _argmax_random(self._scores(slot, word_index, context), self.rng)
            )
            word_indices.append(word_index)
            predicted_indices.append(predicted_index)
        predicted_factors = tuple(
            self.slot_values[slot][index]
            for slot, index in enumerate(predicted_indices)
        )
        predicted_action = self.factor_actions[predicted_factors]
        reward, accepted, outcome_sha = consequence(predicted_action)
        synthetic_decision = ReceiverDecision(
            frontier_sha256=self.frontier.sha256,
            word_id=self.slot_vocabularies[0][word_indices[0]],
            context_id=context,
            action_id=predicted_action,
            sequence=0,
        )
        feedback = ConsequenceFeedback.for_decision(
            synthetic_decision,
            reward=reward,
            accepted=accepted,
            outcome_receipt_sha256=outcome_sha,
        )
        for slot, factor in enumerate(factors):
            value_index = self.slot_values[slot].index(factor)
            word_index = word_indices[slot]
            predicted_index = predicted_indices[slot]
            for table, row, column in (
                (self.sender_q[slot], value_index, word_index),
                (self.receiver_q[slot], word_index, predicted_index),
                (self.context_q[context][slot], word_index, predicted_index),
            ):
                old = table[row, column]
                table[row, column] = old + rate * (reward - old)
            self.receiver_visits[slot][word_index, predicted_index] += 1
            self.context_visits[context][slot][word_index, predicted_index] += 1
        return (
            tuple(
                self.slot_vocabularies[slot][index]
                for slot, index in enumerate(word_indices)
            ),
            predicted_action,
            feedback,
        )

    def encode_action(self, action_id: str) -> tuple[str, ...]:
        factors = self.action_factors[_identifier(action_id, field="action_id")]
        return tuple(
            self.slot_vocabularies[slot][
                int(
                    np.argmax(self.sender_q[slot][self.slot_values[slot].index(factor)])
                )
            ]
            for slot, factor in enumerate(factors)
        )

    def decode_words(self, words: Sequence[str], context_id: str) -> str | None:
        if len(words) != len(self.slot_names):
            return None
        context = _identifier(context_id, field="context_id")
        factors = []
        for slot, word_id in enumerate(words):
            try:
                word = self.slot_vocabularies[slot].index(
                    _identifier(word_id, field="slot_word")
                )
            except ValueError:
                return None
            scores = self._scores(slot, word, context)
            choice = int(np.argmax(scores))
            visits = int(np.sum(self.receiver_visits[slot][word]))
            ordered = np.sort(scores)
            margin = (
                float(ordered[-1] - ordered[-2])
                if len(ordered) > 1
                else float(ordered[-1])
            )
            if (
                visits < self.minimum_visits
                or float(scores[choice]) < self.minimum_value
                or margin < self.minimum_margin
            ):
                return None
            factors.append(self.slot_values[slot][choice])
        return self.factor_actions.get(tuple(factors))

    @property
    def sha256(self) -> str:
        body = {
            "action_factors": {
                action: list(values)
                for action, values in sorted(self.action_factors.items())
            },
            "context_schema_sha256": self.frontier.context_schema_sha256,
            "format": FACTORIZED_GRAMMAR_SCHEMA,
            "frontier_sha256": self.frontier.sha256,
            "slot_names": list(self.slot_names),
            "slot_vocabularies": [list(row) for row in self.slot_vocabularies],
        }
        return _body_sha256(body)


@dataclass(frozen=True, slots=True)
class HolisticConsequenceTable:
    """Seen-combination baseline that must abstain on an unseen action tuple."""

    seen_action_ids: frozenset[str]

    def __init__(self, seen_action_ids: Sequence[str] | set[str]) -> None:
        object.__setattr__(
            self,
            "seen_action_ids",
            frozenset(
                _identifier(action, field="action_id") for action in seen_action_ids
            ),
        )

    def encode(self, action_id: str) -> str | None:
        action = _identifier(action_id, field="action_id")
        return f"holistic:{action}" if action in self.seen_action_ids else None


def assert_receiver_information_boundary() -> None:
    """Executable API guard used by tests and release checks."""

    parameters = set(
        inspect.signature(ConsequenceMarkovLanguage.receiver_decide).parameters
    )
    forbidden = {"intent", "intent_action_id", "target", "target_action", "label"}
    if parameters & forbidden:
        raise AssertionError("receiver API exposes latent semantics")


__all__ = [
    "ACTION_ARTIFACT_KINDS",
    "ACTION_FRONTIER_SCHEMA",
    "ActionBinding",
    "ActionFrontier",
    "ConsequenceFeedback",
    "ConsequenceMarkovLanguage",
    "FactorizedConsequenceGrammar",
    "HolisticConsequenceTable",
    "LanguageEpisodeReceipt",
    "LanguageSnapshot",
    "MarkovLanguageConflictError",
    "MarkovLanguageError",
    "MarkovLanguageIntegrityError",
    "ReceiverDecision",
    "SenderDecision",
    "SenderEmission",
    "assert_receiver_information_boundary",
]
