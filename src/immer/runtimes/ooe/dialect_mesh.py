"""Cross-dialect translation and portable executable word programs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import cast

from .executable_lexicon import ExecutableWordDefinition
from .identity import canonical_json_bytes, require_sha256
from .language_bridge import LanguageMacroDiscoveryReceipt
from .markov_language import ActionBinding, ActionFrontier, LanguageSnapshot


DIALECT_TRANSLATION_SCHEMA = "immer-ooe-dialect-translation/v1"
PORTABLE_WORD_PROGRAM_SCHEMA = "immer-ooe-portable-word-program/v1"
PORTABLE_WORD_LOCALIZATION_SCHEMA = "immer-ooe-portable-word-localization/v1"

MAX_PROGRAM_ACTIONS = 4_096
MAX_TRANSLATIONS = 4_096
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


class DialectMeshError(RuntimeError):
    """Base error for cross-dialect executable language."""


class DialectMeshIntegrityError(DialectMeshError):
    """A dialect, action binding, or portable program failed verification."""


def _identifier(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > 256
    ):
        raise ValueError(f"{field} must be canonical bounded text")
    return value


def _hash_items(
    values: Mapping[str, str] | Sequence[tuple[str, str]],
    *,
    field: str,
) -> tuple[tuple[str, str], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    result = tuple(
        sorted(
            (
                _identifier(name, field=f"{field} name"),
                require_sha256(digest, field=f"{field}[{name!r}]"),
            )
            for name, digest in items
        )
    )
    if len(result) > 1_024 or len({name for name, _ in result}) != len(result):
        raise ValueError(f"{field} must be bounded and uniquely named")
    return result


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_json(data: bytes, *, label: str) -> object:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if len(data) > MAX_PAYLOAD_BYTES:
        raise DialectMeshIntegrityError(f"{label} exceeds its byte bound")

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
        raise DialectMeshIntegrityError(f"{label} is invalid JSON") from exc
    if canonical_json_bytes(value) != data:
        raise DialectMeshIntegrityError(f"{label} is not canonical JSON")
    return value


def _seal(schema: str, body: Mapping[str, object]) -> bytes:
    exact = dict(body)
    return canonical_json_bytes(
        {"body": exact, "body_sha256": _sha256(exact), "schema": schema}
    )


def _open(data: bytes, *, schema: str, label: str) -> Mapping[str, object]:
    value = _strict_json(data, label=label)
    if (
        not isinstance(value, Mapping)
        or set(value) != {"body", "body_sha256", "schema"}
        or value.get("schema") != schema
        or not isinstance(value.get("body"), Mapping)
    ):
        raise DialectMeshIntegrityError(f"{label} envelope is invalid")
    body = cast(Mapping[str, object], value.get("body"))
    if value.get("body_sha256") != _sha256(body):
        raise DialectMeshIntegrityError(f"{label} body hash mismatch")
    return body


@dataclass(frozen=True, slots=True)
class DialectTranslationReceipt:
    source_snapshot_sha256: str
    target_snapshot_sha256: str
    source_frontier_sha256: str
    target_frontier_sha256: str
    source_context_id: str
    target_context_id: str
    word_translations: tuple[tuple[str, str, str], ...]
    unmapped_action_ids: tuple[str, ...]

    FORMAT = DIALECT_TRANSLATION_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "source_snapshot_sha256",
            "target_snapshot_sha256",
            "source_frontier_sha256",
            "target_frontier_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        for field_name in ("source_context_id", "target_context_id"):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), field=field_name),
            )
        translations = tuple(
            sorted(
                (
                    _identifier(source, field="source_word"),
                    _identifier(target, field="target_word"),
                    _identifier(action, field="action_id"),
                )
                for source, target, action in self.word_translations
            )
        )
        if len(translations) > MAX_TRANSLATIONS or len(
            {source for source, _, _ in translations}
        ) != len(translations):
            raise ValueError("dialect translation source words are invalid")
        if len({action for _, _, action in translations}) != len(translations):
            raise ValueError("dialect translation actions must be unique")
        object.__setattr__(self, "word_translations", translations)
        unmapped = tuple(
            sorted(
                _identifier(action, field="unmapped_action_id")
                for action in self.unmapped_action_ids
            )
        )
        if len(set(unmapped)) != len(unmapped) or set(unmapped) & {
            action for _, _, action in translations
        }:
            raise ValueError("unmapped dialect actions are invalid")
        object.__setattr__(self, "unmapped_action_ids", unmapped)

    def to_record(self) -> dict[str, object]:
        return {
            "format": self.FORMAT,
            "source_context_id": self.source_context_id,
            "source_frontier_sha256": self.source_frontier_sha256,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "target_context_id": self.target_context_id,
            "target_frontier_sha256": self.target_frontier_sha256,
            "target_snapshot_sha256": self.target_snapshot_sha256,
            "unmapped_action_ids": list(self.unmapped_action_ids),
            "word_translations": [
                {
                    "action_id": action,
                    "source_word": source,
                    "target_word": target,
                }
                for source, target, action in self.word_translations
            ],
        }

    def to_bytes(self) -> bytes:
        return _seal(self.FORMAT, self.to_record())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "DialectTranslationReceipt":
        body = _open(data, schema=cls.FORMAT, label="dialect translation")
        expected = {
            "format",
            "source_context_id",
            "source_frontier_sha256",
            "source_snapshot_sha256",
            "target_context_id",
            "target_frontier_sha256",
            "target_snapshot_sha256",
            "unmapped_action_ids",
            "word_translations",
        }
        if set(body) != expected or body.get("format") != cls.FORMAT:
            raise DialectMeshIntegrityError("dialect translation body is invalid")
        raw_translations = body.get("word_translations")
        unmapped = body.get("unmapped_action_ids")
        if not isinstance(raw_translations, list) or not isinstance(unmapped, list):
            raise DialectMeshIntegrityError("dialect translation inventory is invalid")
        try:
            translations = []
            for row in raw_translations:
                if not isinstance(row, Mapping) or set(row) != {
                    "action_id",
                    "source_word",
                    "target_word",
                }:
                    raise ValueError("invalid translation row")
                translations.append(
                    (
                        cast(str, row.get("source_word")),
                        cast(str, row.get("target_word")),
                        cast(str, row.get("action_id")),
                    )
                )
            result = cls(
                source_snapshot_sha256=cast(str, body.get("source_snapshot_sha256")),
                target_snapshot_sha256=cast(str, body.get("target_snapshot_sha256")),
                source_frontier_sha256=cast(str, body.get("source_frontier_sha256")),
                target_frontier_sha256=cast(str, body.get("target_frontier_sha256")),
                source_context_id=cast(str, body.get("source_context_id")),
                target_context_id=cast(str, body.get("target_context_id")),
                word_translations=tuple(translations),
                unmapped_action_ids=tuple(unmapped),
            )
        except (TypeError, ValueError) as exc:
            raise DialectMeshIntegrityError(
                "dialect translation reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise DialectMeshIntegrityError(
                "dialect translation failed canonical reconstruction"
            )
        return result


def align_dialects(
    source_snapshot: LanguageSnapshot,
    target_snapshot: LanguageSnapshot,
    source_frontier: ActionFrontier,
    target_frontier: ActionFrontier,
    *,
    source_context_id: str,
    target_context_id: str,
) -> DialectTranslationReceipt:
    """Align two independently grounded dialects through exact action bindings."""

    for value, name in (
        (source_snapshot, "source_snapshot"),
        (target_snapshot, "target_snapshot"),
    ):
        if not isinstance(value, LanguageSnapshot):
            raise TypeError(f"{name} must be a LanguageSnapshot")
    for value, name in (
        (source_frontier, "source_frontier"),
        (target_frontier, "target_frontier"),
    ):
        if not isinstance(value, ActionFrontier):
            raise TypeError(f"{name} must be an ActionFrontier")
    if source_snapshot.frontier_sha256 != source_frontier.sha256:
        raise DialectMeshIntegrityError("source snapshot frontier is stale")
    if target_snapshot.frontier_sha256 != target_frontier.sha256:
        raise DialectMeshIntegrityError("target snapshot frontier is stale")
    source_context = _identifier(source_context_id, field="source_context_id")
    target_context = _identifier(target_context_id, field="target_context_id")
    source_bindings = {item.action_id: item for item in source_frontier.actions}
    target_bindings = {item.action_id: item for item in target_frontier.actions}
    translations = []
    unmapped = []
    for action in sorted(set(source_bindings) | set(target_bindings)):
        if source_bindings.get(action) != target_bindings.get(action):
            unmapped.append(action)
            continue
        source_word = source_snapshot.encode_action(action, source_context)
        target_word = target_snapshot.encode_action(action, target_context)
        if source_word is None or target_word is None:
            unmapped.append(action)
            continue
        translations.append((source_word, target_word, action))
    return DialectTranslationReceipt(
        source_snapshot_sha256=source_snapshot.sha256,
        target_snapshot_sha256=target_snapshot.sha256,
        source_frontier_sha256=source_frontier.sha256,
        target_frontier_sha256=target_frontier.sha256,
        source_context_id=source_context,
        target_context_id=target_context,
        word_translations=tuple(translations),
        unmapped_action_ids=tuple(unmapped),
    )


def translate_words(
    word_ids: Sequence[str],
    receipt: DialectTranslationReceipt,
    *,
    source_snapshot: LanguageSnapshot,
    target_snapshot: LanguageSnapshot,
    source_frontier: ActionFrontier,
    target_frontier: ActionFrontier,
) -> tuple[str, ...] | None:
    if not isinstance(receipt, DialectTranslationReceipt):
        raise TypeError("receipt must be a DialectTranslationReceipt")
    expected = align_dialects(
        source_snapshot,
        target_snapshot,
        source_frontier,
        target_frontier,
        source_context_id=receipt.source_context_id,
        target_context_id=receipt.target_context_id,
    )
    if expected != receipt:
        raise DialectMeshIntegrityError(
            "dialect translation differs from the bound snapshots"
        )
    mapping = {source: target for source, target, _ in receipt.word_translations}
    output = []
    for word in word_ids:
        source = _identifier(word, field="word_id")
        target = mapping.get(source)
        if target is None:
            return None
        output.append(target)
    return tuple(output)


@dataclass(frozen=True, slots=True)
class PortableWordProgram:
    source_snapshot_sha256: str
    source_frontier_sha256: str
    source_definition_sha256: str
    discovery_sha256: str
    authority_hashes: tuple[tuple[str, str], ...]
    action_bindings: tuple[ActionBinding, ...]
    action_sequence: tuple[str, ...]
    source_word_sequence: tuple[str, ...]
    support_trajectory_sha256s: tuple[str, ...]

    FORMAT = PORTABLE_WORD_PROGRAM_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "source_snapshot_sha256",
            "source_frontier_sha256",
            "source_definition_sha256",
            "discovery_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        object.__setattr__(
            self,
            "authority_hashes",
            _hash_items(self.authority_hashes, field="authority_hashes"),
        )
        bindings = tuple(sorted(self.action_bindings, key=lambda item: item.action_id))
        if (
            not bindings
            or len(bindings) > MAX_PROGRAM_ACTIONS
            or any(not isinstance(item, ActionBinding) for item in bindings)
        ):
            raise ValueError("portable action bindings are invalid")
        if len({item.action_id for item in bindings}) != len(bindings):
            raise ValueError("portable action bindings contain duplicates")
        object.__setattr__(self, "action_bindings", bindings)
        actions = tuple(
            _identifier(value, field="action_id") for value in self.action_sequence
        )
        words = tuple(
            _identifier(value, field="source_word")
            for value in self.source_word_sequence
        )
        if not 2 <= len(actions) <= MAX_PROGRAM_ACTIONS or len(actions) != len(words):
            raise ValueError("portable program sequence lengths are invalid")
        binding_map = {item.action_id: item for item in bindings}
        if any(action not in binding_map for action in actions):
            raise ValueError("portable program references an unbound action")
        object.__setattr__(self, "action_sequence", actions)
        object.__setattr__(self, "source_word_sequence", words)
        supports = tuple(
            sorted(
                require_sha256(value, field="support_trajectory_sha256")
                for value in self.support_trajectory_sha256s
            )
        )
        if not supports or len(set(supports)) != len(supports):
            raise ValueError("portable support trajectories are invalid")
        object.__setattr__(self, "support_trajectory_sha256s", supports)

    def to_record(self) -> dict[str, object]:
        return {
            "action_bindings": [item.to_record() for item in self.action_bindings],
            "action_sequence": list(self.action_sequence),
            "authority_hashes": dict(self.authority_hashes),
            "discovery_sha256": self.discovery_sha256,
            "format": self.FORMAT,
            "source_definition_sha256": self.source_definition_sha256,
            "source_frontier_sha256": self.source_frontier_sha256,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "source_word_sequence": list(self.source_word_sequence),
            "support_trajectory_sha256s": list(self.support_trajectory_sha256s),
        }

    def to_bytes(self) -> bytes:
        return _seal(self.FORMAT, self.to_record())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "PortableWordProgram":
        body = _open(data, schema=cls.FORMAT, label="portable word program")
        expected = {
            "action_bindings",
            "action_sequence",
            "authority_hashes",
            "discovery_sha256",
            "format",
            "source_definition_sha256",
            "source_frontier_sha256",
            "source_snapshot_sha256",
            "source_word_sequence",
            "support_trajectory_sha256s",
        }
        if set(body) != expected or body.get("format") != cls.FORMAT:
            raise DialectMeshIntegrityError("portable word body is invalid")
        raw_bindings = body.get("action_bindings")
        authorities = body.get("authority_hashes")
        collections = {
            name: body.get(name)
            for name in (
                "action_sequence",
                "source_word_sequence",
                "support_trajectory_sha256s",
            )
        }
        if (
            not isinstance(raw_bindings, list)
            or not isinstance(authorities, Mapping)
            or any(not isinstance(value, list) for value in collections.values())
        ):
            raise DialectMeshIntegrityError("portable word inventory is invalid")
        try:
            result = cls(
                source_snapshot_sha256=cast(str, body.get("source_snapshot_sha256")),
                source_frontier_sha256=cast(str, body.get("source_frontier_sha256")),
                source_definition_sha256=cast(
                    str, body.get("source_definition_sha256")
                ),
                discovery_sha256=cast(str, body.get("discovery_sha256")),
                authority_hashes=_hash_items(
                    cast(Mapping[str, str], authorities), field="authority_hashes"
                ),
                action_bindings=tuple(
                    ActionBinding.from_record(row) for row in raw_bindings
                ),
                action_sequence=tuple(cast(list[str], collections["action_sequence"])),
                source_word_sequence=tuple(
                    cast(list[str], collections["source_word_sequence"])
                ),
                support_trajectory_sha256s=tuple(
                    cast(list[str], collections["support_trajectory_sha256s"])
                ),
            )
        except (TypeError, ValueError) as exc:
            raise DialectMeshIntegrityError(
                "portable word reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise DialectMeshIntegrityError(
                "portable word failed canonical reconstruction"
            )
        return result


def portable_program_from_discovery(
    discovery: LanguageMacroDiscoveryReceipt,
    definition: ExecutableWordDefinition,
    source_snapshot: LanguageSnapshot,
    source_frontier: ActionFrontier,
) -> PortableWordProgram:
    if not isinstance(discovery, LanguageMacroDiscoveryReceipt):
        raise TypeError("discovery must be a LanguageMacroDiscoveryReceipt")
    if not isinstance(definition, ExecutableWordDefinition):
        raise TypeError("definition must be an ExecutableWordDefinition")
    if not isinstance(source_snapshot, LanguageSnapshot):
        raise TypeError("source_snapshot must be a LanguageSnapshot")
    if not isinstance(source_frontier, ActionFrontier):
        raise TypeError("source_frontier must be an ActionFrontier")
    if (
        definition not in discovery.definitions
        or definition.context_id is not None
        or discovery.language_snapshot_sha256 != source_snapshot.sha256
        or discovery.frontier_sha256 != source_frontier.sha256
        or source_snapshot.frontier_sha256 != source_frontier.sha256
        or definition.language_snapshot_sha256 != source_snapshot.sha256
        or definition.frontier_sha256 != source_frontier.sha256
    ):
        raise DialectMeshIntegrityError(
            "portable source objects belong to another language contract"
        )
    stable = dict(source_snapshot.global_word_actions)
    try:
        actions = tuple(stable[word] for word in definition.child_word_ids)
        binding_map = {
            action: source_frontier.binding(action) for action in set(actions)
        }
        supports = dict(discovery.definition_supports)[definition.sha256]
    except KeyError as exc:
        raise DialectMeshIntegrityError(
            "portable definition contains context-specific or unsupported words"
        ) from exc
    return PortableWordProgram(
        source_snapshot_sha256=source_snapshot.sha256,
        source_frontier_sha256=source_frontier.sha256,
        source_definition_sha256=definition.sha256,
        discovery_sha256=discovery.sha256,
        authority_hashes=source_frontier.authority_hashes,
        action_bindings=tuple(binding_map.values()),
        action_sequence=actions,
        source_word_sequence=definition.child_word_ids,
        support_trajectory_sha256s=supports,
    )


@dataclass(frozen=True, slots=True)
class PortableWordLocalizationReceipt:
    portable_program_sha256: str
    target_snapshot_sha256: str
    target_frontier_sha256: str
    target_context_id: str
    localized_definition_sha256: str
    action_word_pairs: tuple[tuple[str, str], ...]
    target_binding_sha256s: tuple[str, ...]

    FORMAT = PORTABLE_WORD_LOCALIZATION_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "portable_program_sha256",
            "target_snapshot_sha256",
            "target_frontier_sha256",
            "localized_definition_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        object.__setattr__(
            self,
            "target_context_id",
            _identifier(self.target_context_id, field="target_context_id"),
        )
        pairs = tuple(
            (
                _identifier(action, field="action_id"),
                _identifier(word, field="target_word"),
            )
            for action, word in self.action_word_pairs
        )
        if not pairs or len(pairs) > MAX_PROGRAM_ACTIONS:
            raise ValueError("localized action/word sequence is invalid")
        object.__setattr__(self, "action_word_pairs", pairs)
        bindings = tuple(
            require_sha256(value, field="target_binding_sha256")
            for value in self.target_binding_sha256s
        )
        if len(bindings) != len(pairs):
            raise ValueError("localized binding sequence length changed")
        object.__setattr__(self, "target_binding_sha256s", bindings)

    def to_record(self) -> dict[str, object]:
        return {
            "action_word_pairs": [
                {"action_id": action, "word_id": word}
                for action, word in self.action_word_pairs
            ],
            "format": self.FORMAT,
            "localized_definition_sha256": self.localized_definition_sha256,
            "portable_program_sha256": self.portable_program_sha256,
            "target_binding_sha256s": list(self.target_binding_sha256s),
            "target_context_id": self.target_context_id,
            "target_frontier_sha256": self.target_frontier_sha256,
            "target_snapshot_sha256": self.target_snapshot_sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(self.FORMAT, self.to_record())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "PortableWordLocalizationReceipt":
        body = _open(data, schema=cls.FORMAT, label="portable word localization")
        expected = {
            "action_word_pairs",
            "format",
            "localized_definition_sha256",
            "portable_program_sha256",
            "target_binding_sha256s",
            "target_context_id",
            "target_frontier_sha256",
            "target_snapshot_sha256",
        }
        if set(body) != expected or body.get("format") != cls.FORMAT:
            raise DialectMeshIntegrityError("localization body is invalid")
        raw_pairs = body.get("action_word_pairs")
        bindings = body.get("target_binding_sha256s")
        if not isinstance(raw_pairs, list) or not isinstance(bindings, list):
            raise DialectMeshIntegrityError("localization inventory is invalid")
        try:
            pairs = []
            for row in raw_pairs:
                if not isinstance(row, Mapping) or set(row) != {
                    "action_id",
                    "word_id",
                }:
                    raise ValueError("invalid localization pair")
                pairs.append(
                    (
                        cast(str, row.get("action_id")),
                        cast(str, row.get("word_id")),
                    )
                )
            result = cls(
                portable_program_sha256=cast(str, body.get("portable_program_sha256")),
                target_snapshot_sha256=cast(str, body.get("target_snapshot_sha256")),
                target_frontier_sha256=cast(str, body.get("target_frontier_sha256")),
                target_context_id=cast(str, body.get("target_context_id")),
                localized_definition_sha256=cast(
                    str, body.get("localized_definition_sha256")
                ),
                action_word_pairs=tuple(pairs),
                target_binding_sha256s=tuple(bindings),
            )
        except (TypeError, ValueError) as exc:
            raise DialectMeshIntegrityError(
                "localization reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise DialectMeshIntegrityError(
                "localization failed canonical reconstruction"
            )
        return result


def localize_portable_program(
    portable: PortableWordProgram,
    target_snapshot: LanguageSnapshot,
    target_frontier: ActionFrontier,
    *,
    source_discovery: LanguageMacroDiscoveryReceipt,
    source_definition: ExecutableWordDefinition,
    source_snapshot: LanguageSnapshot,
    source_frontier: ActionFrontier,
    target_context_id: str,
) -> tuple[ExecutableWordDefinition, PortableWordLocalizationReceipt]:
    if not isinstance(portable, PortableWordProgram):
        raise TypeError("portable must be a PortableWordProgram")
    if not isinstance(target_snapshot, LanguageSnapshot):
        raise TypeError("target_snapshot must be a LanguageSnapshot")
    if not isinstance(target_frontier, ActionFrontier):
        raise TypeError("target_frontier must be an ActionFrontier")
    expected_portable = portable_program_from_discovery(
        source_discovery,
        source_definition,
        source_snapshot,
        source_frontier,
    )
    if expected_portable != portable:
        raise DialectMeshIntegrityError(
            "portable program differs from its source discovery evidence"
        )
    if target_snapshot.frontier_sha256 != target_frontier.sha256:
        raise DialectMeshIntegrityError("target snapshot frontier is stale")
    if target_frontier.authority_hashes != portable.authority_hashes:
        raise DialectMeshIntegrityError(
            "target verifier authorities differ from the portable program"
        )
    source_bindings = {item.action_id: item for item in portable.action_bindings}
    words = []
    binding_hashes = []
    pairs = []
    for action in portable.action_sequence:
        try:
            target_binding = target_frontier.binding(action)
        except KeyError as exc:
            raise DialectMeshIntegrityError(
                f"target dialect lacks portable action {action!r}"
            ) from exc
        if target_binding != source_bindings[action]:
            raise DialectMeshIntegrityError(
                f"target action binding changed for {action!r}"
            )
        word = target_snapshot.encode_action(action, target_context_id)
        if word is None:
            raise DialectMeshError(
                f"target dialect cannot express portable action {action!r}"
            )
        words.append(word)
        pairs.append((action, word))
        binding_hashes.append(_sha256(target_binding.to_record()))
    localized_word_sha256 = _sha256(
        {
            "portable_program_sha256": portable.sha256,
            "target_context_id": target_context_id,
            "target_snapshot_sha256": target_snapshot.sha256,
        }
    )
    definition = ExecutableWordDefinition.create(
        f"portable-{localized_word_sha256[:32]}",
        tuple(words),
        language_snapshot_sha256=target_snapshot.sha256,
        frontier_sha256=target_frontier.sha256,
        authority_hashes=target_frontier.authority_hashes,
        context_id=target_context_id,
    )
    receipt = PortableWordLocalizationReceipt(
        portable_program_sha256=portable.sha256,
        target_snapshot_sha256=target_snapshot.sha256,
        target_frontier_sha256=target_frontier.sha256,
        target_context_id=target_context_id,
        localized_definition_sha256=definition.sha256,
        action_word_pairs=tuple(pairs),
        target_binding_sha256s=tuple(binding_hashes),
    )
    return definition, receipt


__all__ = [
    "DIALECT_TRANSLATION_SCHEMA",
    "PORTABLE_WORD_LOCALIZATION_SCHEMA",
    "PORTABLE_WORD_PROGRAM_SCHEMA",
    "DialectMeshError",
    "DialectMeshIntegrityError",
    "DialectTranslationReceipt",
    "PortableWordLocalizationReceipt",
    "PortableWordProgram",
    "align_dialects",
    "localize_portable_program",
    "portable_program_from_discovery",
    "translate_words",
]
