"""Self-hosting executable word DAGs over authenticated compute artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Iterator, Literal, cast

from .compute_crystals import (
    MAX_PROGRAM_STEPS,
    ComputeChargeReceipt,
    ComputeCrystal,
    ComputeCrystalBank,
    ComputeCrystalError,
    ComputeCrystalFusionError,
    ComputeCrystalVM,
    ComputeProgram,
    fuse_compatible_chain_with_provenance,
)
from .crystal import (
    CrystalStore,
    CrystalStoreError,
    ManifestConflictError,
)
from .identity import canonical_json_bytes, require_sha256
from .markov_language import ACTION_ARTIFACT_KINDS


WORD_DEFINITION_SCHEMA = "immer-ooe-executable-word-definition/v1"
LEXICON_STATE_SCHEMA = "immer-ooe-executable-lexicon-state/v1"
COMPILED_WORD_SCHEMA = "immer-ooe-compiled-word/v1"
LEXICON_COMMIT_SCHEMA = "immer-ooe-executable-lexicon-commit/v1"

MAX_DEFINITIONS = 65_536
MAX_DIRECT_REFERENCES = 4_096
MAX_DEFINITION_DEPTH = 2_048
MAX_EXPANDED_ACTIONS = (1 << 63) - 1
MAX_LEXICON_BYTES = 256 * 1024 * 1024
MAX_HISTORY_STATES = 200_000

_HEAD_STATE = "ooe-executable-lexicon-head/v1"
_HISTORY_PREFIX = "ooe-executable-lexicon-history/v1:"
_COMMIT_PREFIX = "ooe-executable-lexicon-commit/v1:"
_BANK_LOCK = ".executable-lexicon.lock"
_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,191}")
_STATE_NAME_RE = re.compile(
    rb'^\{"format":"immer-ooe-controller-state/v1","generation":[1-9][0-9]*,"name":"([^"\\]*)","payload_base64":"'
)


class ExecutableLexiconError(RuntimeError):
    """Base error for executable word storage and compilation."""


class ExecutableLexiconIntegrityError(ExecutableLexiconError):
    """A word, state, history object, or compiled receipt failed validation."""


class ExecutableLexiconConflictError(ExecutableLexiconError):
    """A definition, CAS head, or execution contract conflicts."""


class ExecutableLexiconCompileError(ExecutableLexiconError):
    """A word DAG cannot compile under the requested numerical runtime."""


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


def _body_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(dict(value))).hexdigest()


def _strict_json(data: bytes, *, label: str) -> object:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if len(data) > MAX_LEXICON_BYTES:
        raise ExecutableLexiconIntegrityError(f"{label} exceeds its byte bound")

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
        raise ExecutableLexiconIntegrityError(f"{label} is invalid JSON") from exc
    if canonical_json_bytes(value) != data:
        raise ExecutableLexiconIntegrityError(f"{label} is not canonical JSON")
    return value


def _bounded_count(
    value: object,
    *,
    field: str,
    maximum: int,
    allow_zero: bool = False,
) -> int:
    minimum = 0 if allow_zero else 1
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{field} must lie in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class PrimitiveWordBinding:
    word_id: str
    artifact_kind: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "word_id", _identifier(self.word_id, field="word_id"))
        if self.artifact_kind not in ACTION_ARTIFACT_KINDS:
            raise ValueError("unsupported primitive artifact kind")
        object.__setattr__(
            self,
            "artifact_sha256",
            require_sha256(self.artifact_sha256, field="artifact_sha256"),
        )

    def to_record(self) -> dict[str, str]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_sha256": self.artifact_sha256,
            "word_id": self.word_id,
        }

    @classmethod
    def from_record(cls, value: object) -> "PrimitiveWordBinding":
        if not isinstance(value, Mapping) or set(value) != {
            "artifact_kind",
            "artifact_sha256",
            "word_id",
        }:
            raise ExecutableLexiconIntegrityError("primitive binding is invalid")
        try:
            return cls(
                word_id=cast(str, value.get("word_id")),
                artifact_kind=cast(str, value.get("artifact_kind")),
                artifact_sha256=cast(str, value.get("artifact_sha256")),
            )
        except (TypeError, ValueError) as exc:
            raise ExecutableLexiconIntegrityError(
                "primitive binding is invalid"
            ) from exc


@dataclass(frozen=True, slots=True)
class ExecutableWordDefinition:
    new_word_id: str
    child_word_ids: tuple[str, ...]
    language_snapshot_sha256: str
    frontier_sha256: str
    authority_hashes: tuple[tuple[str, str], ...]

    FORMAT = WORD_DEFINITION_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "new_word_id", _identifier(self.new_word_id, field="new_word_id")
        )
        children = tuple(
            _identifier(value, field="child_word_id") for value in self.child_word_ids
        )
        if not 1 <= len(children) <= MAX_DIRECT_REFERENCES:
            raise ValueError("definition has an invalid direct-reference count")
        object.__setattr__(self, "child_word_ids", children)
        for field_name in ("language_snapshot_sha256", "frontier_sha256"):
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

    @classmethod
    def create(
        cls,
        new_word_id: str,
        child_word_ids: Sequence[str],
        *,
        language_snapshot_sha256: str,
        frontier_sha256: str,
        authority_hashes: Mapping[str, str] | Sequence[tuple[str, str]],
    ) -> "ExecutableWordDefinition":
        return cls(
            new_word_id=new_word_id,
            child_word_ids=tuple(child_word_ids),
            language_snapshot_sha256=language_snapshot_sha256,
            frontier_sha256=frontier_sha256,
            authority_hashes=_hash_items(authority_hashes, field="authority_hashes"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "authority_hashes": dict(self.authority_hashes),
            "child_word_ids": list(self.child_word_ids),
            "format": self.FORMAT,
            "frontier_sha256": self.frontier_sha256,
            "language_snapshot_sha256": self.language_snapshot_sha256,
            "new_word_id": self.new_word_id,
        }

    def to_bytes(self) -> bytes:
        body = self.to_record()
        return canonical_json_bytes(
            {
                "body": body,
                "body_sha256": _body_sha256(body),
                "schema": self.FORMAT,
            }
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ExecutableWordDefinition":
        value = _strict_json(data, label="word definition")
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
            raise ExecutableLexiconIntegrityError("definition envelope is invalid")
        body = value.get("body")
        if (
            not isinstance(body, Mapping)
            or value.get("body_sha256")
            != _body_sha256(cast(Mapping[str, object], body))
            or set(body)
            != {
                "authority_hashes",
                "child_word_ids",
                "format",
                "frontier_sha256",
                "language_snapshot_sha256",
                "new_word_id",
            }
            or body.get("format") != cls.FORMAT
        ):
            raise ExecutableLexiconIntegrityError("definition body is invalid")
        children = body.get("child_word_ids")
        authorities = body.get("authority_hashes")
        if not isinstance(children, list) or not isinstance(authorities, Mapping):
            raise ExecutableLexiconIntegrityError("definition inventory is invalid")
        try:
            result = cls(
                new_word_id=cast(str, body.get("new_word_id")),
                child_word_ids=tuple(children),
                language_snapshot_sha256=cast(
                    str, body.get("language_snapshot_sha256")
                ),
                frontier_sha256=cast(str, body.get("frontier_sha256")),
                authority_hashes=_hash_items(
                    cast(Mapping[str, str], authorities),
                    field="authority_hashes",
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ExecutableLexiconIntegrityError(
                "definition reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise ExecutableLexiconIntegrityError(
                "definition failed canonical reconstruction"
            )
        return result


@dataclass(frozen=True, slots=True)
class CompiledWordReceipt:
    word_id: str
    definition_sha256: str
    source_lexicon_state_sha256: str
    language_snapshot_sha256: str
    frontier_sha256: str
    authority_hashes: tuple[tuple[str, str], ...]
    artifact_kind: Literal["crystal", "program"] | str
    artifact_sha256: str
    program_sha256: str
    charge_sha256: str | None
    expanded_primitive_actions: int
    stored_definition_references: int
    constant_discharge: bool
    compute_bank_anchor_sha256: str
    compiler_verifier_sha256: str
    verification_receipt_sha256: str

    FORMAT = COMPILED_WORD_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "word_id", _identifier(self.word_id, field="word_id"))
        for field_name in (
            "definition_sha256",
            "source_lexicon_state_sha256",
            "language_snapshot_sha256",
            "frontier_sha256",
            "artifact_sha256",
            "program_sha256",
            "compute_bank_anchor_sha256",
            "compiler_verifier_sha256",
            "verification_receipt_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        if self.artifact_kind not in ("crystal", "program"):
            raise ValueError("compiled artifact kind is invalid")
        if self.compiler_verifier_sha256 != (EXECUTABLE_WORD_COMPILER_VERIFIER_SHA256):
            raise ValueError("compiled word uses another compiler verifier")
        charge = self.charge_sha256
        if charge is not None:
            charge = require_sha256(charge, field="charge_sha256")
        if not isinstance(self.constant_discharge, bool):
            raise TypeError("constant_discharge must be bool")
        if self.constant_discharge != (
            self.artifact_kind == "crystal" and charge is not None
        ):
            raise ValueError("constant_discharge disagrees with the artifact")
        object.__setattr__(self, "charge_sha256", charge)
        object.__setattr__(
            self,
            "expanded_primitive_actions",
            _bounded_count(
                self.expanded_primitive_actions,
                field="expanded_primitive_actions",
                maximum=MAX_EXPANDED_ACTIONS,
            ),
        )
        object.__setattr__(
            self,
            "stored_definition_references",
            _bounded_count(
                self.stored_definition_references,
                field="stored_definition_references",
                maximum=MAX_EXPANDED_ACTIONS,
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "authority_hashes",
            _hash_items(self.authority_hashes, field="authority_hashes"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_sha256": self.artifact_sha256,
            "authority_hashes": dict(self.authority_hashes),
            "charge_sha256": self.charge_sha256,
            "compiler_verifier_sha256": self.compiler_verifier_sha256,
            "compute_bank_anchor_sha256": self.compute_bank_anchor_sha256,
            "constant_discharge": self.constant_discharge,
            "definition_sha256": self.definition_sha256,
            "expanded_primitive_actions": self.expanded_primitive_actions,
            "format": self.FORMAT,
            "frontier_sha256": self.frontier_sha256,
            "language_snapshot_sha256": self.language_snapshot_sha256,
            "program_sha256": self.program_sha256,
            "source_lexicon_state_sha256": self.source_lexicon_state_sha256,
            "stored_definition_references": self.stored_definition_references,
            "verification_receipt_sha256": self.verification_receipt_sha256,
            "word_id": self.word_id,
        }

    def to_bytes(self) -> bytes:
        body = self.to_record()
        return canonical_json_bytes(
            {
                "body": body,
                "body_sha256": _body_sha256(body),
                "schema": self.FORMAT,
            }
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "CompiledWordReceipt":
        value = _strict_json(data, label="compiled word receipt")
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
            raise ExecutableLexiconIntegrityError(
                "compiled receipt envelope is invalid"
            )
        body = value.get("body")
        expected = {
            "artifact_kind",
            "artifact_sha256",
            "authority_hashes",
            "charge_sha256",
            "compiler_verifier_sha256",
            "compute_bank_anchor_sha256",
            "constant_discharge",
            "definition_sha256",
            "expanded_primitive_actions",
            "format",
            "frontier_sha256",
            "language_snapshot_sha256",
            "program_sha256",
            "source_lexicon_state_sha256",
            "stored_definition_references",
            "verification_receipt_sha256",
            "word_id",
        }
        if (
            not isinstance(body, Mapping)
            or set(body) != expected
            or body.get("format") != cls.FORMAT
            or value.get("body_sha256")
            != _body_sha256(cast(Mapping[str, object], body))
        ):
            raise ExecutableLexiconIntegrityError("compiled receipt body is invalid")
        authorities = body.get("authority_hashes")
        if not isinstance(authorities, Mapping):
            raise ExecutableLexiconIntegrityError("compiled authorities are invalid")
        try:
            result = cls(
                word_id=cast(str, body.get("word_id")),
                definition_sha256=cast(str, body.get("definition_sha256")),
                source_lexicon_state_sha256=cast(
                    str, body.get("source_lexicon_state_sha256")
                ),
                language_snapshot_sha256=cast(
                    str, body.get("language_snapshot_sha256")
                ),
                frontier_sha256=cast(str, body.get("frontier_sha256")),
                authority_hashes=_hash_items(
                    cast(Mapping[str, str], authorities), field="authority_hashes"
                ),
                artifact_kind=cast(str, body.get("artifact_kind")),
                artifact_sha256=cast(str, body.get("artifact_sha256")),
                program_sha256=cast(str, body.get("program_sha256")),
                charge_sha256=cast(str | None, body.get("charge_sha256")),
                expanded_primitive_actions=cast(
                    int, body.get("expanded_primitive_actions")
                ),
                stored_definition_references=cast(
                    int, body.get("stored_definition_references")
                ),
                constant_discharge=cast(bool, body.get("constant_discharge")),
                compute_bank_anchor_sha256=cast(
                    str, body.get("compute_bank_anchor_sha256")
                ),
                compiler_verifier_sha256=cast(
                    str, body.get("compiler_verifier_sha256")
                ),
                verification_receipt_sha256=cast(
                    str, body.get("verification_receipt_sha256")
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ExecutableLexiconIntegrityError(
                "compiled receipt reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise ExecutableLexiconIntegrityError(
                "compiled receipt failed canonical reconstruction"
            )
        return result


@dataclass(frozen=True, slots=True)
class ExecutableLexiconState:
    generation: int
    previous_state_sha256: str | None
    language_snapshot_sha256: str
    frontier_sha256: str
    authority_hashes: tuple[tuple[str, str], ...]
    primitive_bindings: tuple[PrimitiveWordBinding, ...]
    definitions: tuple[ExecutableWordDefinition, ...]
    compiled_receipts: tuple[CompiledWordReceipt, ...] = ()

    FORMAT = LEXICON_STATE_SCHEMA

    def __post_init__(self) -> None:
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 1
        ):
            raise ValueError("lexicon generation must be positive")
        previous = self.previous_state_sha256
        if previous is not None:
            previous = require_sha256(previous, field="previous_state_sha256")
        if self.generation == 1 and previous is not None:
            raise ValueError("initial lexicon state cannot have a predecessor")
        if self.generation > 1 and previous is None:
            raise ValueError("non-initial lexicon state needs a predecessor")
        object.__setattr__(self, "previous_state_sha256", previous)
        for field_name in ("language_snapshot_sha256", "frontier_sha256"):
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
        primitives = tuple(self.primitive_bindings)
        if not primitives or len(primitives) > MAX_DEFINITIONS:
            raise ValueError("primitive binding inventory is invalid")
        if any(not isinstance(item, PrimitiveWordBinding) for item in primitives):
            raise TypeError("primitive binding inventory is invalid")
        if tuple(sorted(primitives, key=lambda item: item.word_id)) != primitives:
            raise ValueError("primitive bindings must be sorted")
        if len({item.word_id for item in primitives}) != len(primitives):
            raise ValueError("primitive words must be unique")
        definitions = tuple(self.definitions)
        if len(definitions) > MAX_DEFINITIONS or any(
            not isinstance(item, ExecutableWordDefinition) for item in definitions
        ):
            raise ValueError("definition inventory is invalid")
        if tuple(sorted(definitions, key=lambda item: item.new_word_id)) != definitions:
            raise ValueError("definitions must be sorted")
        if len({item.new_word_id for item in definitions}) != len(definitions):
            raise ValueError("definition words must be unique")
        primitive_words = {item.word_id for item in primitives}
        if primitive_words & {item.new_word_id for item in definitions}:
            raise ValueError("a definition cannot replace a primitive word")
        for definition in definitions:
            if (
                definition.language_snapshot_sha256 != self.language_snapshot_sha256
                or definition.frontier_sha256 != self.frontier_sha256
                or definition.authority_hashes != self.authority_hashes
            ):
                raise ValueError("definition execution contract is stale")
        receipts = tuple(self.compiled_receipts)
        if len(receipts) > MAX_DEFINITIONS or any(
            not isinstance(item, CompiledWordReceipt) for item in receipts
        ):
            raise ValueError("compiled receipt inventory is invalid")
        if tuple(sorted(receipts, key=lambda item: item.word_id)) != receipts:
            raise ValueError("compiled receipts must be sorted")
        if len({item.word_id for item in receipts}) != len(receipts):
            raise ValueError("compiled receipt words must be unique")
        definition_map = {item.new_word_id: item for item in definitions}
        for receipt in receipts:
            definition = definition_map.get(receipt.word_id)
            if (
                definition is None
                or receipt.definition_sha256 != definition.sha256
                or receipt.language_snapshot_sha256 != self.language_snapshot_sha256
                or receipt.frontier_sha256 != self.frontier_sha256
                or receipt.authority_hashes != self.authority_hashes
            ):
                raise ValueError("compiled receipt execution contract is stale")
        object.__setattr__(self, "primitive_bindings", primitives)
        object.__setattr__(self, "definitions", definitions)
        object.__setattr__(self, "compiled_receipts", receipts)
        self._validate_graph()

    @classmethod
    def initial(
        cls,
        primitive_bindings: Sequence[PrimitiveWordBinding],
        *,
        language_snapshot_sha256: str,
        frontier_sha256: str,
        authority_hashes: Mapping[str, str] | Sequence[tuple[str, str]],
    ) -> "ExecutableLexiconState":
        return cls(
            generation=1,
            previous_state_sha256=None,
            language_snapshot_sha256=language_snapshot_sha256,
            frontier_sha256=frontier_sha256,
            authority_hashes=_hash_items(authority_hashes, field="authority_hashes"),
            primitive_bindings=tuple(
                sorted(primitive_bindings, key=lambda item: item.word_id)
            ),
            definitions=(),
            compiled_receipts=(),
        )

    @property
    def primitive_map(self) -> dict[str, PrimitiveWordBinding]:
        return {item.word_id: item for item in self.primitive_bindings}

    @property
    def definition_map(self) -> dict[str, ExecutableWordDefinition]:
        return {item.new_word_id: item for item in self.definitions}

    def knows(self, word_id: str) -> bool:
        word = _identifier(word_id, field="word_id")
        return word in self.primitive_map or word in self.definition_map

    def _validate_graph(self) -> None:
        primitives = set(self.primitive_map)
        definitions = self.definition_map
        state: dict[str, int] = {}

        def visit(word: str, depth: int) -> None:
            if depth > MAX_DEFINITION_DEPTH:
                raise ValueError("definition graph exceeds its depth bound")
            marker = state.get(word, 0)
            if marker == 1:
                raise ValueError("definition graph contains a cycle")
            if marker == 2 or word in primitives:
                return
            definition = definitions.get(word)
            if definition is None:
                raise ValueError(f"definition references unknown word {word!r}")
            state[word] = 1
            for child in definition.child_word_ids:
                visit(child, depth + 1)
            state[word] = 2

        for word in definitions:
            visit(word, 1)

    def expanded_length(self, word_id: str) -> int:
        word = _identifier(word_id, field="word_id")
        primitives = self.primitive_map
        definitions = self.definition_map
        memo: dict[str, int] = {}

        def length(current: str, depth: int) -> int:
            if current in primitives:
                return 1
            if current in memo:
                return memo[current]
            if depth > MAX_DEFINITION_DEPTH:
                raise ExecutableLexiconIntegrityError(
                    "definition graph exceeds its depth bound"
                )
            try:
                children = definitions[current].child_word_ids
            except KeyError as exc:
                raise KeyError(f"unknown word: {current}") from exc
            total = 0
            for child in children:
                total += length(child, depth + 1)
                if total > MAX_EXPANDED_ACTIONS:
                    raise ExecutableLexiconError(
                        "expanded word exceeds its action bound"
                    )
            memo[current] = total
            return total

        return length(word, 1)

    def reachable_reference_count(self, word_id: str) -> int:
        word = _identifier(word_id, field="word_id")
        definitions = self.definition_map
        seen: set[str] = set()

        def visit(current: str) -> int:
            definition = definitions.get(current)
            if definition is None or current in seen:
                return 0
            seen.add(current)
            return len(definition.child_word_ids) + sum(
                visit(child) for child in definition.child_word_ids
            )

        if not self.knows(word):
            raise KeyError(f"unknown word: {word}")
        return visit(word)

    def expand_word(self, word_id: str, *, max_actions: int) -> tuple[str, ...]:
        bound = _bounded_count(
            max_actions,
            field="max_actions",
            maximum=MAX_EXPANDED_ACTIONS,
        )
        expected = self.expanded_length(word_id)
        if expected > bound:
            raise ExecutableLexiconError("expanded word exceeds caller bound")
        primitives = self.primitive_map
        definitions = self.definition_map
        output: list[str] = []

        def visit(word: str) -> None:
            if word in primitives:
                output.append(word)
                return
            for child in definitions[word].child_word_ids:
                visit(child)

        visit(_identifier(word_id, field="word_id"))
        return tuple(output)

    def with_definition(
        self, definition: ExecutableWordDefinition
    ) -> "ExecutableLexiconState":
        if not isinstance(definition, ExecutableWordDefinition):
            raise TypeError("definition must be an ExecutableWordDefinition")
        if (
            definition.language_snapshot_sha256 != self.language_snapshot_sha256
            or definition.frontier_sha256 != self.frontier_sha256
            or definition.authority_hashes != self.authority_hashes
        ):
            raise ExecutableLexiconConflictError("definition contract is stale")
        if definition.new_word_id in self.primitive_map:
            raise ExecutableLexiconConflictError("cannot redefine a primitive word")
        existing = self.definition_map.get(definition.new_word_id)
        if existing is not None:
            if existing == definition:
                return self
            raise ExecutableLexiconConflictError("conflicting word redefinition")
        if len(self.definitions) >= MAX_DEFINITIONS:
            raise ExecutableLexiconError("definition inventory is full")
        if definition.new_word_id in definition.child_word_ids:
            raise ExecutableLexiconConflictError("cyclic word definition")
        missing = [
            child for child in definition.child_word_ids if not self.knows(child)
        ]
        if missing:
            raise ExecutableLexiconConflictError(
                f"definition has unknown dependency {missing[0]!r}"
            )
        try:
            return ExecutableLexiconState(
                generation=self.generation + 1,
                previous_state_sha256=self.sha256,
                language_snapshot_sha256=self.language_snapshot_sha256,
                frontier_sha256=self.frontier_sha256,
                authority_hashes=self.authority_hashes,
                primitive_bindings=self.primitive_bindings,
                definitions=tuple(
                    sorted(
                        (*self.definitions, definition),
                        key=lambda item: item.new_word_id,
                    )
                ),
                compiled_receipts=self.compiled_receipts,
            )
        except ValueError as exc:
            raise ExecutableLexiconConflictError(
                "definition would invalidate the word graph"
            ) from exc

    def with_compiled_receipt(
        self, receipt: CompiledWordReceipt
    ) -> "ExecutableLexiconState":
        if not isinstance(receipt, CompiledWordReceipt):
            raise TypeError("receipt must be a CompiledWordReceipt")
        if receipt.source_lexicon_state_sha256 != self.sha256:
            raise ExecutableLexiconConflictError(
                "compiled receipt was built from another lexicon state"
            )
        existing = {item.word_id: item for item in self.compiled_receipts}.get(
            receipt.word_id
        )
        if existing is not None:
            if existing == receipt:
                return self
            raise ExecutableLexiconConflictError("conflicting compiled word receipt")
        return ExecutableLexiconState(
            generation=self.generation + 1,
            previous_state_sha256=self.sha256,
            language_snapshot_sha256=self.language_snapshot_sha256,
            frontier_sha256=self.frontier_sha256,
            authority_hashes=self.authority_hashes,
            primitive_bindings=self.primitive_bindings,
            definitions=self.definitions,
            compiled_receipts=tuple(
                sorted(
                    (*self.compiled_receipts, receipt),
                    key=lambda item: item.word_id,
                )
            ),
        )

    def to_document(self) -> dict[str, object]:
        body = {
            "authority_hashes": dict(self.authority_hashes),
            "compiled_receipts": [item.to_record() for item in self.compiled_receipts],
            "definitions": [item.to_record() for item in self.definitions],
            "frontier_sha256": self.frontier_sha256,
            "generation": self.generation,
            "language_snapshot_sha256": self.language_snapshot_sha256,
            "previous_state_sha256": self.previous_state_sha256,
            "primitive_bindings": [
                item.to_record() for item in self.primitive_bindings
            ],
        }
        return {
            "body": body,
            "body_sha256": _body_sha256(body),
            "schema": self.FORMAT,
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_LEXICON_BYTES:
            raise ExecutableLexiconError("lexicon state exceeds its byte bound")
        return data

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ExecutableLexiconState":
        value = _strict_json(data, label="lexicon state")
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
            raise ExecutableLexiconIntegrityError("lexicon state envelope is invalid")
        body = value.get("body")
        expected = {
            "authority_hashes",
            "compiled_receipts",
            "definitions",
            "frontier_sha256",
            "generation",
            "language_snapshot_sha256",
            "previous_state_sha256",
            "primitive_bindings",
        }
        if (
            not isinstance(body, Mapping)
            or set(body) != expected
            or value.get("body_sha256")
            != _body_sha256(cast(Mapping[str, object], body))
        ):
            raise ExecutableLexiconIntegrityError("lexicon state body is invalid")
        primitives = body.get("primitive_bindings")
        definitions = body.get("definitions")
        compiled = body.get("compiled_receipts")
        authorities = body.get("authority_hashes")
        if (
            not isinstance(primitives, list)
            or not isinstance(definitions, list)
            or not isinstance(compiled, list)
            or not isinstance(authorities, Mapping)
        ):
            raise ExecutableLexiconIntegrityError("lexicon inventory is invalid")
        try:
            definition_objects = tuple(
                ExecutableWordDefinition.from_bytes(
                    canonical_json_bytes(
                        {
                            "body": row,
                            "body_sha256": _body_sha256(
                                cast(Mapping[str, object], row)
                            ),
                            "schema": WORD_DEFINITION_SCHEMA,
                        }
                    )
                )
                for row in definitions
            )
            compiled_objects = tuple(
                CompiledWordReceipt.from_bytes(
                    canonical_json_bytes(
                        {
                            "body": row,
                            "body_sha256": _body_sha256(
                                cast(Mapping[str, object], row)
                            ),
                            "schema": COMPILED_WORD_SCHEMA,
                        }
                    )
                )
                for row in compiled
            )
            result = cls(
                generation=cast(int, body.get("generation")),
                previous_state_sha256=cast(
                    str | None, body.get("previous_state_sha256")
                ),
                language_snapshot_sha256=cast(
                    str, body.get("language_snapshot_sha256")
                ),
                frontier_sha256=cast(str, body.get("frontier_sha256")),
                authority_hashes=_hash_items(
                    cast(Mapping[str, str], authorities), field="authority_hashes"
                ),
                primitive_bindings=tuple(
                    PrimitiveWordBinding.from_record(row) for row in primitives
                ),
                definitions=definition_objects,
                compiled_receipts=compiled_objects,
            )
        except ExecutableLexiconError:
            raise
        except (TypeError, ValueError) as exc:
            raise ExecutableLexiconIntegrityError(
                "lexicon state reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise ExecutableLexiconIntegrityError(
                "lexicon state failed canonical reconstruction"
            )
        return result


class ExecutableLexicon:
    """Mutable protocol facade over immutable lexicon states."""

    def __init__(self, state: ExecutableLexiconState) -> None:
        if not isinstance(state, ExecutableLexiconState):
            raise TypeError("state must be an ExecutableLexiconState")
        self.state = state

    def export_definition(self, word_id: str) -> ExecutableWordDefinition | None:
        return self.state.definition_map.get(_identifier(word_id, field="word_id"))

    def install(self, definition: ExecutableWordDefinition) -> None:
        self.state = self.state.with_definition(definition)

    def coin(
        self,
        new_word_id: str,
        child_word_ids: Sequence[str],
    ) -> ExecutableWordDefinition:
        definition = ExecutableWordDefinition.create(
            new_word_id,
            child_word_ids,
            language_snapshot_sha256=self.state.language_snapshot_sha256,
            frontier_sha256=self.state.frontier_sha256,
            authority_hashes=self.state.authority_hashes,
        )
        self.install(definition)
        return definition

    def decode_or_request(
        self,
        word_ids: Sequence[str],
        sender: "ExecutableLexicon",
    ) -> tuple[tuple[str, ...] | None, int]:
        if not isinstance(sender, ExecutableLexicon):
            raise TypeError("sender must be an ExecutableLexicon")
        repairs = 0

        def ensure(word: str, stack: set[str]) -> bool:
            nonlocal repairs
            if self.state.knows(word):
                return True
            if word in stack:
                raise ExecutableLexiconConflictError("remote definition cycle")
            definition = sender.export_definition(word)
            if definition is None:
                return False
            stack.add(word)
            for child in definition.child_word_ids:
                if not ensure(child, stack):
                    stack.remove(word)
                    return False
            stack.remove(word)
            self.install(definition)
            repairs += 1
            return True

        normalized = tuple(_identifier(word, field="word_id") for word in word_ids)
        for word in normalized:
            if not ensure(word, set()):
                return None, repairs
        return normalized, repairs


EXECUTABLE_WORD_COMPILER_VERIFIER_SHA256 = _body_sha256(
    {
        "composition": "execution-order exact ComputeCrystal fusion",
        "fallback": "bounded ordered ComputeProgram",
        "format": "immer-ooe-executable-word-compiler-verifier/v1",
        "transitive_work": "immer-ooe-fusion-work-provenance/v1",
    }
)


@dataclass(frozen=True, slots=True)
class _CompiledNode:
    artifact_kind: Literal["crystal", "program"]
    artifact_sha256: str
    program_sha256: str
    charge_sha256: str | None
    crystals: tuple[ComputeCrystal, ...]
    expanded_primitive_actions: int


class ExecutableWordCompiler:
    """Bottom-up compiler from a definition DAG into ComputeCrystal artifacts."""

    def __init__(
        self,
        state: ExecutableLexiconState,
        compute_bank: ComputeCrystalBank,
    ) -> None:
        if not isinstance(state, ExecutableLexiconState):
            raise TypeError("state must be an ExecutableLexiconState")
        if not isinstance(compute_bank, ComputeCrystalBank):
            raise TypeError("compute_bank must be a ComputeCrystalBank")
        self.state = state
        self.compute_bank = compute_bank
        self._memo: dict[str, _CompiledNode] = {}

    def _primitive(self, binding: PrimitiveWordBinding) -> _CompiledNode:
        if binding.artifact_kind == "crystal":
            try:
                crystal = self.compute_bank.restore_crystal(binding.artifact_sha256)
            except ComputeCrystalError as exc:
                raise ExecutableLexiconCompileError(
                    f"primitive crystal is unavailable for {binding.word_id!r}"
                ) from exc
            program = ComputeProgram.compose((crystal,))
            self.compute_bank.publish_program(program)
            return _CompiledNode(
                artifact_kind="crystal",
                artifact_sha256=crystal.sha256,
                program_sha256=program.sha256,
                charge_sha256=None,
                crystals=(crystal,),
                expanded_primitive_actions=1,
            )
        if binding.artifact_kind == "program":
            try:
                program, crystals = self.compute_bank.resolve_program(
                    binding.artifact_sha256
                )
            except ComputeCrystalError as exc:
                raise ExecutableLexiconCompileError(
                    f"primitive program is unavailable for {binding.word_id!r}"
                ) from exc
            return _CompiledNode(
                artifact_kind="program",
                artifact_sha256=program.sha256,
                program_sha256=program.sha256,
                charge_sha256=None,
                crystals=crystals,
                expanded_primitive_actions=1,
            )
        raise ExecutableLexiconCompileError(
            f"artifact kind {binding.artifact_kind!r} can be named but not compiled"
        )

    def _compile_node(self, word_id: str, depth: int) -> _CompiledNode:
        word = _identifier(word_id, field="word_id")
        cached = self._memo.get(word)
        if cached is not None:
            return cached
        if depth > MAX_DEFINITION_DEPTH:
            raise ExecutableLexiconCompileError(
                "word compilation exceeds its depth bound"
            )
        primitive = self.state.primitive_map.get(word)
        if primitive is not None:
            result = self._primitive(primitive)
            self._memo[word] = result
            return result
        try:
            definition = self.state.definition_map[word]
        except KeyError as exc:
            raise ExecutableLexiconCompileError(f"unknown word: {word}") from exc
        children = tuple(
            self._compile_node(child, depth + 1) for child in definition.child_word_ids
        )
        expanded = 0
        for child in children:
            expanded += child.expanded_primitive_actions
            if expanded > MAX_EXPANDED_ACTIONS:
                raise ExecutableLexiconCompileError(
                    "compiled word exceeds its expanded-action bound"
                )

        direct = tuple(
            child.crystals[0] for child in children if len(child.crystals) == 1
        )
        if len(direct) == len(children) and len(direct) >= 2:
            try:
                source_program = ComputeProgram.compose(direct)
                self.compute_bank.publish_program(source_program)
                fused = fuse_compatible_chain_with_provenance(
                    direct,
                    extensions={
                        "executable_word": {
                            "definition_sha256": definition.sha256,
                            "frontier_sha256": self.state.frontier_sha256,
                            "language_snapshot_sha256": (
                                self.state.language_snapshot_sha256
                            ),
                            "word_id": word,
                        }
                    },
                )
            except ComputeCrystalFusionError:
                fused = None
            except ComputeCrystalError as exc:
                raise ExecutableLexiconCompileError(
                    f"word {word!r} violates its compute ABI"
                ) from exc
            if fused is not None:
                self.compute_bank.publish_crystal(fused)
                verification_receipt_sha256 = _body_sha256(
                    {
                        "definition_sha256": definition.sha256,
                        "format": "immer-ooe-executable-word-exact-composition/v1",
                        "fused_crystal_sha256": fused.sha256,
                        "source_program_sha256": source_program.sha256,
                    }
                )
                charge = ComputeChargeReceipt.create(
                    source_program=source_program,
                    source_crystals=direct,
                    fused_crystal=fused,
                    charge_verifier_sha256=(EXECUTABLE_WORD_COMPILER_VERIFIER_SHA256),
                    verification_receipt_sha256=verification_receipt_sha256,
                )
                self.compute_bank.publish_charge(charge)
                program = ComputeProgram.compose((fused,))
                self.compute_bank.publish_program(program)
                result = _CompiledNode(
                    artifact_kind="crystal",
                    artifact_sha256=fused.sha256,
                    program_sha256=program.sha256,
                    charge_sha256=charge.sha256,
                    crystals=(fused,),
                    expanded_primitive_actions=expanded,
                )
                self._memo[word] = result
                return result

        flattened = tuple(crystal for child in children for crystal in child.crystals)
        if not flattened or len(flattened) > MAX_PROGRAM_STEPS:
            raise ExecutableLexiconCompileError(
                "non-fusible word exceeds the bounded program representation"
            )
        try:
            program = ComputeProgram.compose(flattened)
            self.compute_bank.publish_program(program)
        except ComputeCrystalError as exc:
            raise ExecutableLexiconCompileError(
                f"word {word!r} has an incompatible ordered program"
            ) from exc
        result = _CompiledNode(
            artifact_kind="program",
            artifact_sha256=program.sha256,
            program_sha256=program.sha256,
            charge_sha256=None,
            crystals=flattened,
            expanded_primitive_actions=expanded,
        )
        self._memo[word] = result
        return result

    def compile(self, word_id: str) -> CompiledWordReceipt:
        word = _identifier(word_id, field="word_id")
        try:
            definition = self.state.definition_map[word]
        except KeyError as exc:
            raise ExecutableLexiconCompileError(
                "compile a defined word, not a primitive binding"
            ) from exc
        node = self._compile_node(word, 1)
        verification = _body_sha256(
            {
                "artifact_kind": node.artifact_kind,
                "artifact_sha256": node.artifact_sha256,
                "definition_sha256": definition.sha256,
                "expanded_primitive_actions": node.expanded_primitive_actions,
                "format": "immer-ooe-executable-word-compilation/v1",
                "program_sha256": node.program_sha256,
            }
        )
        return CompiledWordReceipt(
            word_id=word,
            definition_sha256=definition.sha256,
            source_lexicon_state_sha256=self.state.sha256,
            language_snapshot_sha256=self.state.language_snapshot_sha256,
            frontier_sha256=self.state.frontier_sha256,
            authority_hashes=self.state.authority_hashes,
            artifact_kind=node.artifact_kind,
            artifact_sha256=node.artifact_sha256,
            program_sha256=node.program_sha256,
            charge_sha256=node.charge_sha256,
            expanded_primitive_actions=node.expanded_primitive_actions,
            stored_definition_references=self.state.reachable_reference_count(word),
            constant_discharge=(
                node.artifact_kind == "crystal" and node.charge_sha256 is not None
            ),
            compute_bank_anchor_sha256=self.compute_bank.current_anchor_sha256(),
            compiler_verifier_sha256=EXECUTABLE_WORD_COMPILER_VERIFIER_SHA256,
            verification_receipt_sha256=verification,
        )

    def execute(self, receipt: CompiledWordReceipt, value: object):
        if not isinstance(receipt, CompiledWordReceipt):
            raise TypeError("receipt must be a CompiledWordReceipt")
        definition = self.state.definition_map.get(receipt.word_id)
        if (
            definition is None
            or receipt.definition_sha256 != definition.sha256
            or receipt.frontier_sha256 != self.state.frontier_sha256
            or receipt.language_snapshot_sha256 != self.state.language_snapshot_sha256
            or receipt.authority_hashes != self.state.authority_hashes
        ):
            raise ExecutableLexiconConflictError(
                "compiled word belongs to another lexicon contract"
            )
        try:
            self.compute_bank.assert_descends_from(receipt.compute_bank_anchor_sha256)
        except ComputeCrystalError as exc:
            raise ExecutableLexiconIntegrityError(
                "compiled word compute-bank anchor is not an ancestor"
            ) from exc
        node = self._compile_node(receipt.word_id, 1)
        expected_verification = _body_sha256(
            {
                "artifact_kind": node.artifact_kind,
                "artifact_sha256": node.artifact_sha256,
                "definition_sha256": definition.sha256,
                "expanded_primitive_actions": node.expanded_primitive_actions,
                "format": "immer-ooe-executable-word-compilation/v1",
                "program_sha256": node.program_sha256,
            }
        )
        if (
            receipt.artifact_kind != node.artifact_kind
            or receipt.artifact_sha256 != node.artifact_sha256
            or receipt.program_sha256 != node.program_sha256
            or receipt.charge_sha256 != node.charge_sha256
            or receipt.expanded_primitive_actions != node.expanded_primitive_actions
            or receipt.stored_definition_references
            != self.state.reachable_reference_count(receipt.word_id)
            or receipt.verification_receipt_sha256 != expected_verification
        ):
            raise ExecutableLexiconIntegrityError(
                "compiled receipt disagrees with the executable word DAG"
            )
        return ComputeCrystalVM(self.compute_bank).execute(
            receipt.program_sha256,
            value,
            charge_basis_sha256=receipt.charge_sha256,
        )


def _commit_bytes(state_sha256: str) -> bytes:
    address = require_sha256(state_sha256, field="state_sha256")
    return canonical_json_bytes(
        {
            "schema": LEXICON_COMMIT_SCHEMA,
            "state_sha256": address,
        }
    )


def _decode_commit(data: bytes) -> str:
    value = _strict_json(data, label="lexicon commit marker")
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "schema",
            "state_sha256",
        }
        or value.get("schema") != LEXICON_COMMIT_SCHEMA
    ):
        raise ExecutableLexiconIntegrityError("lexicon commit marker is invalid")
    try:
        return require_sha256(
            cast(str, value.get("state_sha256")), field="state_sha256"
        )
    except ValueError as exc:
        raise ExecutableLexiconIntegrityError(
            "lexicon commit marker address is invalid"
        ) from exc


class ExecutableLexiconBank:
    """Crash-safe append-only history for definitions and compiled words."""

    def __init__(
        self,
        store: CrystalStore | str | os.PathLike[str],
        *,
        initial_state: ExecutableLexiconState | None = None,
        trusted_head_sha256: str | None = None,
    ) -> None:
        self.store = store if isinstance(store, CrystalStore) else CrystalStore(store)
        self.root = Path(self.store.root)
        self.trusted_head_sha256 = (
            None
            if trusted_head_sha256 is None
            else require_sha256(trusted_head_sha256, field="trusted_head_sha256")
        )
        if initial_state is not None:
            if not isinstance(initial_state, ExecutableLexiconState):
                raise TypeError("initial_state must be an ExecutableLexiconState")
            if initial_state.generation != 1:
                raise ValueError("initial_state must have generation 1")
            self.initialize(initial_state)

    @staticmethod
    def history_state_name(state_sha256: str) -> str:
        return _HISTORY_PREFIX + require_sha256(state_sha256, field="state_sha256")

    @staticmethod
    def commit_state_name(state_sha256: str) -> str:
        return _COMMIT_PREFIX + require_sha256(state_sha256, field="state_sha256")

    @staticmethod
    def head_state_name() -> str:
        return _HEAD_STATE

    @contextmanager
    def _locked(self) -> Iterator[None]:
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.root / _BANK_LOCK, flags, 0o600)
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ExecutableLexiconIntegrityError(
                    "lexicon lock is not a regular file"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            opened = os.fstat(descriptor)
            linked = (self.root / _BANK_LOCK).lstat()
            if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
                raise ExecutableLexiconIntegrityError(
                    "lexicon lock changed while acquiring it"
                )
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _restore_optional(self, name: str) -> bytes | None:
        try:
            return self.store.restore_state(name)
        except KeyError:
            return None
        except CrystalStoreError as exc:
            raise ExecutableLexiconIntegrityError(
                "lexicon state failed storage integrity"
            ) from exc

    def _publish_immutable(self, name: str, payload: bytes) -> None:
        existing = self._restore_optional(name)
        if existing is not None:
            if existing != payload:
                raise ExecutableLexiconIntegrityError(
                    "immutable lexicon object contains different bytes"
                )
            return
        try:
            self.store.publish_state(name, payload)
        except CrystalStoreError as exc:
            raise ExecutableLexiconIntegrityError(
                "immutable lexicon publication failed"
            ) from exc
        restored = self._restore_optional(name)
        if restored != payload:
            raise ExecutableLexiconIntegrityError(
                "immutable lexicon publication did not roundtrip"
            )

    def _history_state_names(self) -> tuple[str, ...]:
        state_path = self.root / "state"
        names: list[str] = []
        for path in sorted(state_path.iterdir()):
            if not path.name.endswith(".state"):
                continue
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(path, flags)
            except OSError:
                continue
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    continue
                prefix = os.read(descriptor, 4096)
                after = os.fstat(descriptor)
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
                ):
                    raise ExecutableLexiconIntegrityError(
                        "lexicon state changed during history scan"
                    )
            finally:
                os.close(descriptor)
            match = _STATE_NAME_RE.match(prefix)
            if match is None:
                continue
            try:
                name = match.group(1).decode("ascii")
            except UnicodeDecodeError:
                continue
            if name.startswith((_HISTORY_PREFIX, _COMMIT_PREFIX)):
                expected = self.store._state_filename(name)
                if path.name != expected:
                    raise ExecutableLexiconIntegrityError(
                        "lexicon history filename is not name-bound"
                    )
                names.append(name)
        if len(names) > MAX_HISTORY_STATES:
            raise ExecutableLexiconIntegrityError(
                "lexicon history exceeds its state bound"
            )
        return tuple(names)

    def _inventory(
        self,
    ) -> tuple[dict[str, ExecutableLexiconState], set[str]]:
        histories: dict[str, ExecutableLexiconState] = {}
        commits: set[str] = set()
        for name in self._history_state_names():
            payload = self._restore_optional(name)
            if payload is None:
                raise ExecutableLexiconIntegrityError(
                    "lexicon history disappeared during scan"
                )
            if name.startswith(_HISTORY_PREFIX):
                address = require_sha256(
                    name[len(_HISTORY_PREFIX) :], field="history address"
                )
                state = ExecutableLexiconState.from_bytes(payload)
                if state.sha256 != address:
                    raise ExecutableLexiconIntegrityError(
                        "lexicon history is stored under another address"
                    )
                histories[address] = state
            else:
                address = require_sha256(
                    name[len(_COMMIT_PREFIX) :], field="commit address"
                )
                if _decode_commit(payload) != address:
                    raise ExecutableLexiconIntegrityError(
                        "lexicon commit is stored under another address"
                    )
                commits.add(address)
        return histories, commits

    @staticmethod
    def _committed_chain(
        histories: Mapping[str, ExecutableLexiconState],
        commits: set[str],
    ) -> tuple[ExecutableLexiconState, ...]:
        committed = {
            digest: state for digest, state in histories.items() if digest in commits
        }
        if not committed:
            return ()
        roots = [state for state in committed.values() if state.generation == 1]
        if len(roots) != 1:
            raise ExecutableLexiconIntegrityError(
                "committed lexicon history has no unique root"
            )
        chain = [roots[0]]
        while True:
            children = [
                state
                for state in committed.values()
                if state.previous_state_sha256 == chain[-1].sha256
                and state.generation == chain[-1].generation + 1
            ]
            if not children:
                break
            if len(children) != 1:
                raise ExecutableLexiconIntegrityError(
                    "committed lexicon history contains a fork"
                )
            chain.append(children[0])
        if {state.sha256 for state in chain} != set(committed):
            raise ExecutableLexiconIntegrityError(
                "committed lexicon history is disconnected"
            )
        for previous, current in zip(chain, chain[1:], strict=False):
            ExecutableLexiconBank._assert_extension(previous, current)
        return tuple(chain)

    @staticmethod
    def _assert_extension(
        previous: ExecutableLexiconState,
        current: ExecutableLexiconState,
    ) -> None:
        added_definitions = set(current.definitions) - set(previous.definitions)
        added_receipts = set(current.compiled_receipts) - set(
            previous.compiled_receipts
        )
        if (
            current.generation != previous.generation + 1
            or current.previous_state_sha256 != previous.sha256
            or current.language_snapshot_sha256 != previous.language_snapshot_sha256
            or current.frontier_sha256 != previous.frontier_sha256
            or current.authority_hashes != previous.authority_hashes
            or current.primitive_bindings != previous.primitive_bindings
            or not set(previous.definitions) <= set(current.definitions)
            or not set(previous.compiled_receipts) <= set(current.compiled_receipts)
            or len(current.definitions) + len(current.compiled_receipts)
            != len(previous.definitions) + len(previous.compiled_receipts) + 1
            or (
                added_receipts
                and next(iter(added_receipts)).source_lexicon_state_sha256
                != previous.sha256
            )
            or (added_definitions and added_receipts)
        ):
            raise ExecutableLexiconIntegrityError(
                "lexicon state is not an append-only extension"
            )

    def _assert_trusted(self, chain: Sequence[ExecutableLexiconState]) -> None:
        if self.trusted_head_sha256 is None:
            return
        if self.trusted_head_sha256 not in {state.sha256 for state in chain}:
            raise ExecutableLexiconIntegrityError(
                "lexicon history does not descend from its trusted head"
            )

    def _head_unlocked(self) -> ExecutableLexiconState:
        head_payload = self._restore_optional(_HEAD_STATE)
        histories, commits = self._inventory()
        chain = list(self._committed_chain(histories, commits))
        if head_payload is None:
            if chain:
                raise ExecutableLexiconIntegrityError(
                    "lexicon head was deleted or rolled back"
                )
            raise KeyError("executable lexicon is not initialized")
        head = ExecutableLexiconState.from_bytes(head_payload)
        historical = histories.get(head.sha256)
        if historical is None or historical.to_bytes() != head.to_bytes():
            raise ExecutableLexiconIntegrityError(
                "lexicon head lacks immutable history"
            )
        if head.sha256 not in commits:
            if chain:
                self._assert_extension(chain[-1], head)
            elif head.generation != 1:
                raise ExecutableLexiconIntegrityError(
                    "uncommitted lexicon head lacks a root"
                )
            self._publish_immutable(
                self.commit_state_name(head.sha256),
                _commit_bytes(head.sha256),
            )
            commits.add(head.sha256)
            chain = list(self._committed_chain(histories, commits))
        if not chain or chain[-1].sha256 != head.sha256:
            raise ExecutableLexiconIntegrityError(
                "lexicon head is a fully resealed rollback"
            )
        self._assert_trusted(chain)
        return head

    def initialize(
        self, initial_state: ExecutableLexiconState
    ) -> ExecutableLexiconState:
        with self._locked():
            try:
                current = self._head_unlocked()
            except KeyError:
                current = None
            if current is not None:
                if (
                    current.language_snapshot_sha256
                    != initial_state.language_snapshot_sha256
                    or current.frontier_sha256 != initial_state.frontier_sha256
                    or current.authority_hashes != initial_state.authority_hashes
                    or current.primitive_bindings != initial_state.primitive_bindings
                ):
                    raise ExecutableLexiconConflictError(
                        "existing lexicon has another initial contract"
                    )
                return current
            self._publish_immutable(
                self.history_state_name(initial_state.sha256),
                initial_state.to_bytes(),
            )
            try:
                self.store.publish_state(_HEAD_STATE, initial_state.to_bytes())
            except CrystalStoreError as exc:
                raise ExecutableLexiconIntegrityError(
                    "lexicon head publication failed"
                ) from exc
            self._publish_immutable(
                self.commit_state_name(initial_state.sha256),
                _commit_bytes(initial_state.sha256),
            )
            return self._head_unlocked()

    def head(self) -> ExecutableLexiconState:
        with self._locked():
            return self._head_unlocked()

    def current_anchor_sha256(self) -> str:
        return self.head().sha256

    def _append_unlocked(
        self,
        updated: ExecutableLexiconState,
        *,
        expected_head_sha256: str | None,
    ) -> ExecutableLexiconState:
        current = self._head_unlocked()
        if expected_head_sha256 is not None and current.sha256 != require_sha256(
            expected_head_sha256, field="expected_head_sha256"
        ):
            raise ExecutableLexiconConflictError("lexicon head changed before append")
        self._assert_extension(current, updated)
        self._publish_immutable(
            self.history_state_name(updated.sha256), updated.to_bytes()
        )
        try:
            self.store.publish_state(
                _HEAD_STATE,
                updated.to_bytes(),
                expected_sha256=current.sha256,
            )
        except ManifestConflictError as exc:
            raise ExecutableLexiconConflictError("lexicon head CAS conflicted") from exc
        except CrystalStoreError as exc:
            raise ExecutableLexiconIntegrityError(
                "lexicon head publication failed"
            ) from exc
        self._publish_immutable(
            self.commit_state_name(updated.sha256),
            _commit_bytes(updated.sha256),
        )
        return self._head_unlocked()

    def append_definition(
        self,
        definition: ExecutableWordDefinition,
        *,
        expected_head_sha256: str | None = None,
    ) -> ExecutableLexiconState:
        with self._locked():
            current = self._head_unlocked()
            updated = current.with_definition(definition)
            if updated is current:
                return current
            return self._append_unlocked(
                updated, expected_head_sha256=expected_head_sha256
            )

    def append_compiled_receipt(
        self,
        receipt: CompiledWordReceipt,
        *,
        expected_head_sha256: str | None = None,
    ) -> ExecutableLexiconState:
        with self._locked():
            current = self._head_unlocked()
            updated = current.with_compiled_receipt(receipt)
            if updated is current:
                return current
            return self._append_unlocked(
                updated, expected_head_sha256=expected_head_sha256
            )


__all__ = [
    "COMPILED_WORD_SCHEMA",
    "EXECUTABLE_WORD_COMPILER_VERIFIER_SHA256",
    "LEXICON_STATE_SCHEMA",
    "WORD_DEFINITION_SCHEMA",
    "CompiledWordReceipt",
    "ExecutableLexicon",
    "ExecutableLexiconBank",
    "ExecutableLexiconCompileError",
    "ExecutableLexiconConflictError",
    "ExecutableLexiconError",
    "ExecutableLexiconIntegrityError",
    "ExecutableLexiconState",
    "ExecutableWordCompiler",
    "ExecutableWordDefinition",
    "PrimitiveWordBinding",
]
