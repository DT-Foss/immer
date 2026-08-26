"""End-to-end executable language over promoted OoE controller Crystals.

This module turns a verified :class:`ControllerCrystalExportReceipt` into a
small consequence-trained language, proves a repeated four-step program on
held-out inputs, and compiles that program to one charged ComputeCrystal.  All
training rewards come from exact controller-Crystal execution receipts; no
semantic action label is exposed to the receiver.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from typing import Any, cast

import numpy as np

from .compute_crystals import (
    ComputeCrystalBank,
    ComputeCrystalError,
    ComputeCrystalVM,
    tensor_sha256,
)
from .controller_crystal_bridge import (
    ControllerCrystalExport,
    ControllerCrystalExportReceipt,
)
from .crystal import CrystalStore, CrystalStoreError, ManifestConflictError
from .executable_lexicon import (
    CompiledWordReceipt,
    ExecutableLexiconBank,
    ExecutableLexiconError,
    ExecutableLexiconState,
    ExecutableWordCompiler,
    ExecutableWordDefinition,
)
from .identity import canonical_json_bytes, require_sha256
from .language_bridge import (
    ConsequenceLanguageBridge,
    ConsequenceLanguageStateBank,
    ControllerCrystalLanguageExecutor,
    ControllerCrystalOutcomeReceipt,
    LanguageOutcomeCommitReceipt,
    SnapshotComputeResolutionReceipt,
    resolve_snapshot_compute_bindings,
)
from .markov_language import (
    ConsequenceMarkovLanguage,
    LanguageSnapshot,
    MarkovLanguageError,
)
from .qwen_bridge import OOE_ACTIONS


CONTROLLER_PROGRAM_SUPPORT_SCHEMA = "immer-ooe-controller-program-support/v1"
CONTROLLER_LANGUAGE_BOOTSTRAP_REPORT_SCHEMA = (
    "immer-ooe-controller-language-bootstrap-report/v1"
)
CONTROLLER_LANGUAGE_BOOTSTRAP_CONFIG_SCHEMA = (
    "immer-ooe-controller-language-bootstrap-config/v1"
)
CONTROLLER_LANGUAGE_BOOTSTRAP_ALGORITHM_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "format": "immer-ooe-controller-language-bootstrap-algorithm/v1",
            "language": "consequence-markov-v1",
            "program_steps": 4,
            "support_occurrences": 3,
            "training_reward": "verified-controller-crystal-parity",
            "word_compiler": "executable-lexicon-v1",
        }
    )
).hexdigest()
CONTROLLER_PROGRAM_SUPPORT_VERIFIER_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "format": "immer-ooe-controller-program-support-verifier/v1",
            "numerical_replay": "exact-float64-controller-crystal",
            "program_steps": 4,
            "support_occurrences": 3,
        }
    )
).hexdigest()

_STATE_PREFIX = "ooe-controller-language-bootstrap/v1"
_MAX_DOCUMENT_BYTES = 128 * 1024 * 1024
_PROGRAM_STEPS = 4
_SUPPORT_OCCURRENCES = 3


class ControllerLanguageBootstrapError(RuntimeError):
    """The controller language could not be trained or proven."""


class ControllerLanguageBootstrapIntegrityError(ControllerLanguageBootstrapError):
    """A sealed artifact, source binding, or numerical proof changed."""


class ControllerLanguageBootstrapConflictError(ControllerLanguageBootstrapError):
    """A persistent bootstrap address contains another run."""


class ControllerLanguageConvergenceError(ControllerLanguageBootstrapError):
    """The bounded learner did not reach the executable language contract."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: object) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _identifier(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value.encode("utf-8")) > 512
    ):
        raise ValueError(f"{field} must be canonical bounded text")
    return value


def _strict_json(data: bytes, *, label: str) -> object:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if len(data) > _MAX_DOCUMENT_BYTES:
        raise ControllerLanguageBootstrapIntegrityError(
            f"{label} exceeds its hard byte bound"
        )

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
        raise ControllerLanguageBootstrapIntegrityError(
            f"{label} is invalid canonical JSON"
        ) from exc
    if canonical_json_bytes(value) != data:
        raise ControllerLanguageBootstrapIntegrityError(
            f"{label} is not canonical JSON"
        )
    return value


def _seal(schema: str, body: Mapping[str, object]) -> bytes:
    exact = dict(body)
    return canonical_json_bytes(
        {"body": exact, "body_sha256": _digest(exact), "schema": schema}
    )


def _open(data: bytes, *, schema: str, label: str) -> Mapping[str, object]:
    value = _strict_json(data, label=label)
    if (
        not isinstance(value, Mapping)
        or set(value) != {"body", "body_sha256", "schema"}
        or value.get("schema") != schema
        or not isinstance(value.get("body"), Mapping)
    ):
        raise ControllerLanguageBootstrapIntegrityError(f"{label} envelope is invalid")
    body = cast(Mapping[str, object], value.get("body"))
    if value.get("body_sha256") != _digest(body):
        raise ControllerLanguageBootstrapIntegrityError(f"{label} body hash mismatch")
    return body


def _bound_bytes(data: bytes) -> dict[str, object]:
    return {
        "bytes": len(data),
        "data_base64": base64.b64encode(data).decode("ascii"),
        "sha256": _sha256_bytes(data),
    }


def _decode_bound_bytes(value: object, *, label: str) -> bytes:
    if not isinstance(value, Mapping) or set(value) != {
        "bytes",
        "data_base64",
        "sha256",
    }:
        raise ControllerLanguageBootstrapIntegrityError(
            f"{label} descriptor is invalid"
        )
    count = value.get("bytes")
    encoded = value.get("data_base64")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 1 <= count <= _MAX_DOCUMENT_BYTES
        or not isinstance(encoded, str)
        or not encoded.isascii()
        or len(encoded) != 4 * ((count + 2) // 3)
    ):
        raise ControllerLanguageBootstrapIntegrityError(
            f"{label} descriptor is invalid"
        )
    try:
        data = base64.b64decode(encoded, validate=True)
        claimed = require_sha256(value.get("sha256"), field=f"{label}.sha256")
    except (TypeError, ValueError) as exc:
        raise ControllerLanguageBootstrapIntegrityError(
            f"{label} descriptor is invalid"
        ) from exc
    if (
        len(data) != count
        or base64.b64encode(data).decode("ascii") != encoded
        or _sha256_bytes(data) != claimed
    ):
        raise ControllerLanguageBootstrapIntegrityError(
            f"{label} bytes do not match their descriptor"
        )
    return data


def _finite_hex(value: float, *, field: str) -> str:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result.hex()


def _parse_finite_hex(value: object, *, field: str) -> float:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a canonical float hex string")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a canonical float hex string") from exc
    if not math.isfinite(result) or result.hex() != value:
        raise ValueError(f"{field} must be a canonical float hex string")
    return result


@dataclass(frozen=True, slots=True)
class ControllerLanguageBootstrapConfig:
    """Bounded deterministic training and proof parameters."""

    seed: int = 17
    max_training_episodes: int = 4_000
    exploration_start: float = 0.45
    exploration_end: float = 0.0
    learning_rate: float = 0.2
    minimum_visits: int = 8
    minimum_value: float = 0.25
    minimum_margin: float = 0.08
    context_full_weight_visits: int = 16
    stable_greedy_cycles: int = 3

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        for field_name, lower, upper in (
            ("max_training_episodes", 1, 1_000_000),
            ("minimum_visits", 1, 1_000_000),
            ("context_full_weight_visits", 1, 1_000_000),
            ("stable_greedy_cycles", 1, 1_024),
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not lower <= value <= upper
            ):
                raise ValueError(f"{field_name} is outside its bound")
        for field_name in (
            "exploration_start",
            "exploration_end",
            "learning_rate",
            "minimum_value",
            "minimum_margin",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)
        if not 0.0 <= self.exploration_end <= self.exploration_start <= 1.0:
            raise ValueError("exploration bounds are invalid")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in (0, 1]")
        if self.minimum_margin < 0.0:
            raise ValueError("minimum_margin must be non-negative")

    def to_record(self) -> dict[str, object]:
        return {
            "context_full_weight_visits": self.context_full_weight_visits,
            "exploration_end_hex": _finite_hex(
                self.exploration_end, field="exploration_end"
            ),
            "exploration_start_hex": _finite_hex(
                self.exploration_start, field="exploration_start"
            ),
            "format": CONTROLLER_LANGUAGE_BOOTSTRAP_CONFIG_SCHEMA,
            "learning_rate_hex": _finite_hex(self.learning_rate, field="learning_rate"),
            "max_training_episodes": self.max_training_episodes,
            "minimum_margin_hex": _finite_hex(
                self.minimum_margin, field="minimum_margin"
            ),
            "minimum_value_hex": _finite_hex(self.minimum_value, field="minimum_value"),
            "minimum_visits": self.minimum_visits,
            "seed": self.seed,
            "stable_greedy_cycles": self.stable_greedy_cycles,
        }

    @property
    def sha256(self) -> str:
        return _digest(self.to_record())

    @classmethod
    def from_record(cls, value: object) -> "ControllerLanguageBootstrapConfig":
        expected = {
            "context_full_weight_visits",
            "exploration_end_hex",
            "exploration_start_hex",
            "format",
            "learning_rate_hex",
            "max_training_episodes",
            "minimum_margin_hex",
            "minimum_value_hex",
            "minimum_visits",
            "seed",
            "stable_greedy_cycles",
        }
        if (
            not isinstance(value, Mapping)
            or set(value) != expected
            or value.get("format") != CONTROLLER_LANGUAGE_BOOTSTRAP_CONFIG_SCHEMA
        ):
            raise ControllerLanguageBootstrapIntegrityError(
                "bootstrap config record is invalid"
            )
        try:
            result = cls(
                seed=cast(int, value.get("seed")),
                max_training_episodes=cast(int, value.get("max_training_episodes")),
                exploration_start=_parse_finite_hex(
                    value.get("exploration_start_hex"), field="exploration_start"
                ),
                exploration_end=_parse_finite_hex(
                    value.get("exploration_end_hex"), field="exploration_end"
                ),
                learning_rate=_parse_finite_hex(
                    value.get("learning_rate_hex"), field="learning_rate"
                ),
                minimum_visits=cast(int, value.get("minimum_visits")),
                minimum_value=_parse_finite_hex(
                    value.get("minimum_value_hex"), field="minimum_value"
                ),
                minimum_margin=_parse_finite_hex(
                    value.get("minimum_margin_hex"), field="minimum_margin"
                ),
                context_full_weight_visits=cast(
                    int, value.get("context_full_weight_visits")
                ),
                stable_greedy_cycles=cast(int, value.get("stable_greedy_cycles")),
            )
        except (TypeError, ValueError) as exc:
            raise ControllerLanguageBootstrapIntegrityError(
                "bootstrap config reconstruction failed"
            ) from exc
        if result.to_record() != value:
            raise ControllerLanguageBootstrapIntegrityError(
                "bootstrap config failed canonical reconstruction"
            )
        return result


@dataclass(frozen=True, slots=True)
class ControllerProgramOccurrence:
    """One exact held-out execution of the four-action program."""

    ordinal: int
    context_id: str
    commits: tuple[LanguageOutcomeCommitReceipt, ...]
    outcomes: tuple[ControllerCrystalOutcomeReceipt, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or not 0 <= self.ordinal < _SUPPORT_OCCURRENCES
        ):
            raise ValueError("support occurrence ordinal is invalid")
        object.__setattr__(
            self, "context_id", _identifier(self.context_id, field="context_id")
        )
        commits = tuple(self.commits)
        outcomes = tuple(self.outcomes)
        if len(commits) != _PROGRAM_STEPS or any(
            not isinstance(item, LanguageOutcomeCommitReceipt) for item in commits
        ):
            raise ValueError("support occurrence needs four language commits")
        if len(outcomes) != _PROGRAM_STEPS or any(
            not isinstance(item, ControllerCrystalOutcomeReceipt) for item in outcomes
        ):
            raise ValueError("support occurrence needs four controller outcomes")
        if len({item.sha256 for item in commits}) != len(commits) or len(
            {item.sha256 for item in outcomes}
        ) != len(outcomes):
            raise ValueError("support occurrence repeats a receipt")
        object.__setattr__(self, "commits", commits)
        object.__setattr__(self, "outcomes", outcomes)

    def to_record(self) -> dict[str, object]:
        return {
            "commits": [_bound_bytes(item.to_bytes()) for item in self.commits],
            "context_id": self.context_id,
            "ordinal": self.ordinal,
            "outcomes": [_bound_bytes(item.to_bytes()) for item in self.outcomes],
        }

    @classmethod
    def from_record(cls, value: object) -> "ControllerProgramOccurrence":
        if not isinstance(value, Mapping) or set(value) != {
            "commits",
            "context_id",
            "ordinal",
            "outcomes",
        }:
            raise ControllerLanguageBootstrapIntegrityError(
                "support occurrence record is invalid"
            )
        commits = value.get("commits")
        outcomes = value.get("outcomes")
        if not isinstance(commits, list) or not isinstance(outcomes, list):
            raise ControllerLanguageBootstrapIntegrityError(
                "support occurrence receipt inventory is invalid"
            )
        try:
            result = cls(
                ordinal=cast(int, value.get("ordinal")),
                context_id=cast(str, value.get("context_id")),
                commits=tuple(
                    LanguageOutcomeCommitReceipt.from_bytes(
                        _decode_bound_bytes(item, label="language commit")
                    )
                    for item in commits
                ),
                outcomes=tuple(
                    ControllerCrystalOutcomeReceipt.from_bytes(
                        _decode_bound_bytes(item, label="controller outcome")
                    )
                    for item in outcomes
                ),
            )
        except ControllerLanguageBootstrapIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise ControllerLanguageBootstrapIntegrityError(
                "support occurrence reconstruction failed"
            ) from exc
        if result.to_record() != value:
            raise ControllerLanguageBootstrapIntegrityError(
                "support occurrence failed canonical reconstruction"
            )
        return result


@dataclass(frozen=True, slots=True)
class ControllerProgramSupportReceipt:
    """Three fully joined held-out proofs for one four-action word program."""

    export_receipt_sha256: str
    frontier_sha256: str
    language_snapshot: LanguageSnapshot
    action_ids: tuple[str, ...]
    word_ids: tuple[str, ...]
    occurrences: tuple[ControllerProgramOccurrence, ...]
    verifier_sha256: str

    FORMAT = CONTROLLER_PROGRAM_SUPPORT_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "export_receipt_sha256",
            "frontier_sha256",
            "verifier_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        if not isinstance(self.language_snapshot, LanguageSnapshot):
            raise TypeError("language_snapshot must be a LanguageSnapshot")
        if self.language_snapshot.frontier_sha256 != self.frontier_sha256:
            raise ValueError("support snapshot belongs to another frontier")
        if self.verifier_sha256 != CONTROLLER_PROGRAM_SUPPORT_VERIFIER_SHA256:
            raise ValueError("support receipt uses another verifier")
        actions = tuple(
            _identifier(item, field="action_id") for item in self.action_ids
        )
        words = tuple(_identifier(item, field="word_id") for item in self.word_ids)
        if len(actions) != _PROGRAM_STEPS or len(words) != _PROGRAM_STEPS:
            raise ValueError("supported program must contain exactly four steps")
        object.__setattr__(self, "action_ids", actions)
        object.__setattr__(self, "word_ids", words)
        occurrences = tuple(self.occurrences)
        if (
            len(occurrences) != _SUPPORT_OCCURRENCES
            or any(
                not isinstance(item, ControllerProgramOccurrence)
                for item in occurrences
            )
            or tuple(item.ordinal for item in occurrences)
            != tuple(range(_SUPPORT_OCCURRENCES))
            or len({item.context_id for item in occurrences}) != len(occurrences)
        ):
            raise ValueError("support needs three ordered distinct occurrences")
        all_commits: set[str] = set()
        all_outcomes: set[str] = set()
        for occurrence in occurrences:
            if (
                occurrence.commits[0].language_state_before_sha256
                != self.language_snapshot.learner_state_sha256
            ):
                raise ValueError(
                    "held-out occurrence did not start from the frozen learner"
                )
            if any(
                current.language_state_before_sha256
                != previous.language_state_after_sha256
                for previous, current in zip(
                    occurrence.commits,
                    occurrence.commits[1:],
                    strict=False,
                )
            ):
                raise ValueError("held-out occurrence language chain is discontinuous")
            for step, (commit, outcome) in enumerate(
                zip(occurrence.commits, occurrence.outcomes, strict=True)
            ):
                expected_action = actions[step]
                expected_word = words[step]
                if (
                    not commit.accepted
                    or commit.reward != 1.0
                    or commit.external_outcome_kind != "controller-crystal-choice"
                    or commit.external_outcome_sha256 != outcome.sha256
                    or commit.routing_decision_sha256 != outcome.routing_decision_sha256
                    or commit.frontier_sha256 != self.frontier_sha256
                    or commit.action_id != expected_action
                    or commit.word_id != expected_word
                    or commit.context_id != occurrence.context_id
                    or not outcome.accepted
                    or outcome.frontier_sha256 != self.frontier_sha256
                    or outcome.intended_action_id != expected_action
                    or outcome.selected_action_id != expected_action
                    or self.language_snapshot.decode_word(
                        expected_word, occurrence.context_id
                    )
                    != expected_action
                    or dict(self.language_snapshot.global_word_actions).get(
                        expected_word
                    )
                    != expected_action
                ):
                    raise ValueError(
                        "support occurrence is rejected or semantically mismatched"
                    )
                all_commits.add(commit.sha256)
                all_outcomes.add(outcome.sha256)
        if len(all_commits) != _SUPPORT_OCCURRENCES * _PROGRAM_STEPS or len(
            all_outcomes
        ) != (_SUPPORT_OCCURRENCES * _PROGRAM_STEPS):
            raise ValueError("support repeats evidence across occurrences")
        object.__setattr__(self, "occurrences", occurrences)

    def to_record(self) -> dict[str, object]:
        return {
            "action_ids": list(self.action_ids),
            "export_receipt_sha256": self.export_receipt_sha256,
            "format": self.FORMAT,
            "frontier_sha256": self.frontier_sha256,
            "language_snapshot": _bound_bytes(self.language_snapshot.to_bytes()),
            "minimum_support": _SUPPORT_OCCURRENCES,
            "occurrences": [item.to_record() for item in self.occurrences],
            "program_steps": _PROGRAM_STEPS,
            "verifier_sha256": self.verifier_sha256,
            "word_ids": list(self.word_ids),
        }

    def to_bytes(self) -> bytes:
        return _seal(self.FORMAT, self.to_record())

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> "ControllerProgramSupportReceipt":
        body = _open(data, schema=cls.FORMAT, label="controller program support")
        expected = {
            "action_ids",
            "export_receipt_sha256",
            "format",
            "frontier_sha256",
            "language_snapshot",
            "minimum_support",
            "occurrences",
            "program_steps",
            "verifier_sha256",
            "word_ids",
        }
        actions = body.get("action_ids")
        words = body.get("word_ids")
        occurrences = body.get("occurrences")
        if (
            set(body) != expected
            or body.get("format") != cls.FORMAT
            or body.get("minimum_support") != _SUPPORT_OCCURRENCES
            or body.get("program_steps") != _PROGRAM_STEPS
            or not isinstance(actions, list)
            or not isinstance(words, list)
            or not isinstance(occurrences, list)
        ):
            raise ControllerLanguageBootstrapIntegrityError(
                "controller program support body is invalid"
            )
        try:
            result = cls(
                export_receipt_sha256=cast(str, body.get("export_receipt_sha256")),
                frontier_sha256=cast(str, body.get("frontier_sha256")),
                language_snapshot=LanguageSnapshot.from_bytes(
                    _decode_bound_bytes(
                        body.get("language_snapshot"), label="language snapshot"
                    )
                ),
                action_ids=tuple(actions),
                word_ids=tuple(words),
                occurrences=tuple(
                    ControllerProgramOccurrence.from_record(item)
                    for item in occurrences
                ),
                verifier_sha256=cast(str, body.get("verifier_sha256")),
            )
        except ControllerLanguageBootstrapIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise ControllerLanguageBootstrapIntegrityError(
                "controller program support reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise ControllerLanguageBootstrapIntegrityError(
                "controller program support failed canonical reconstruction"
            )
        return result

    def verify(
        self,
        export_receipt: ControllerCrystalExportReceipt,
        compute_bank: ComputeCrystalBank,
    ) -> None:
        """Replay every held-out numerical proof against the exported bank."""

        if not isinstance(export_receipt, ControllerCrystalExportReceipt):
            raise TypeError("export_receipt must be a ControllerCrystalExportReceipt")
        if not isinstance(compute_bank, ComputeCrystalBank):
            raise TypeError("compute_bank must be a ComputeCrystalBank")
        if (
            export_receipt.sha256 != self.export_receipt_sha256
            or export_receipt.frontier.sha256 != self.frontier_sha256
        ):
            raise ControllerLanguageBootstrapIntegrityError(
                "support belongs to another controller export"
            )
        try:
            crystals = export_receipt.restore_crystals(compute_bank)
        except Exception as exc:
            raise ControllerLanguageBootstrapIntegrityError(
                "support controller export failed compute restore"
            ) from exc
        by_action = {
            site.action_id: crystal
            for site, crystal in zip(export_receipt.sites, crystals, strict=True)
        }
        for occurrence in self.occurrences:
            for action_id, outcome in zip(
                self.action_ids, occurrence.outcomes, strict=True
            ):
                crystal = by_action.get(action_id)
                binding = export_receipt.frontier.binding(action_id)
                if crystal is None:
                    raise ControllerLanguageBootstrapIntegrityError(
                        "support action is absent from the controller export"
                    )
                value = outcome.input_array
                try:
                    expected_output = crystal.apply(value)
                    input_sha = tensor_sha256(value, crystal.input_abi)
                    output_sha = tensor_sha256(expected_output, crystal.output_abi)
                except (TypeError, ValueError, ComputeCrystalError) as exc:
                    raise ControllerLanguageBootstrapIntegrityError(
                        "support numerical replay failed"
                    ) from exc
                if (
                    outcome.authorized_bank_anchor_sha256
                    != export_receipt.final_bank_anchor_sha256
                    or outcome.intended_crystal_sha256 != crystal.sha256
                    or outcome.selected_crystal_sha256 != crystal.sha256
                    or binding.artifact_kind != "crystal"
                    or binding.artifact_sha256 != crystal.sha256
                    or outcome.input_sha256 != input_sha
                    or outcome.intended_output_sha256 != output_sha
                    or outcome.selected_output_sha256 != output_sha
                ):
                    raise ControllerLanguageBootstrapIntegrityError(
                        "support numerical proof differs from exact replay"
                    )


@dataclass(frozen=True, slots=True)
class ControllerLanguageBootstrapReport:
    """Canonical result seal for one complete controller-language bootstrap."""

    run_sha256: str
    config: ControllerLanguageBootstrapConfig
    export_receipt_sha256: str
    controller_snapshot_sha256: str
    model_pin_sha256: str
    weight_graph_revision_sha256: str
    atlas_graph_revision_sha256: str
    frontier_sha256: str
    authorized_compute_bank_anchor_sha256: str
    final_compute_bank_anchor_sha256: str
    training_episodes: int
    accepted_training_episodes: int
    stable_greedy_cycles: int
    heldout_executions: int
    heldout_accepted: int
    language_state_sha256: str
    language_snapshot_sha256: str
    support_receipt_sha256: str
    primitive_resolution_sha256: str
    definition_sha256: str
    compiled_receipt_sha256: str
    compiler_source_lexicon_state_sha256: str
    final_lexicon_state_sha256: str
    compiled_crystal_sha256: str
    compiled_program_sha256: str
    charge_receipt_sha256: str
    flat_output_sha256: str
    compiled_output_sha256: str
    parity_verification_sha256: str
    max_abs_error_hex: str
    parity_tolerance_hex: str
    flat_operator_count: int
    compiled_operator_count: int
    equivalent_source_work: int
    live_discharge_work: int
    historical_work_released: int

    FORMAT = CONTROLLER_LANGUAGE_BOOTSTRAP_REPORT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.config, ControllerLanguageBootstrapConfig):
            raise TypeError("config must be a ControllerLanguageBootstrapConfig")
        for field_name in (
            "run_sha256",
            "export_receipt_sha256",
            "controller_snapshot_sha256",
            "model_pin_sha256",
            "weight_graph_revision_sha256",
            "atlas_graph_revision_sha256",
            "frontier_sha256",
            "authorized_compute_bank_anchor_sha256",
            "final_compute_bank_anchor_sha256",
            "language_state_sha256",
            "language_snapshot_sha256",
            "support_receipt_sha256",
            "primitive_resolution_sha256",
            "definition_sha256",
            "compiled_receipt_sha256",
            "compiler_source_lexicon_state_sha256",
            "final_lexicon_state_sha256",
            "compiled_crystal_sha256",
            "compiled_program_sha256",
            "charge_receipt_sha256",
            "flat_output_sha256",
            "compiled_output_sha256",
            "parity_verification_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        for field_name in (
            "training_episodes",
            "accepted_training_episodes",
            "stable_greedy_cycles",
            "heldout_executions",
            "heldout_accepted",
            "flat_operator_count",
            "compiled_operator_count",
            "equivalent_source_work",
            "live_discharge_work",
            "historical_work_released",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if (
            self.accepted_training_episodes > self.training_episodes
            or self.stable_greedy_cycles < self.config.stable_greedy_cycles
            or self.heldout_executions != _PROGRAM_STEPS * _SUPPORT_OCCURRENCES
            or self.heldout_accepted != self.heldout_executions
            or self.flat_operator_count != _PROGRAM_STEPS
            or self.compiled_operator_count != 1
            or self.equivalent_source_work <= self.live_discharge_work
            or self.historical_work_released
            != self.equivalent_source_work - self.live_discharge_work
        ):
            raise ValueError("bootstrap report proof accounting is inconsistent")
        error = _parse_finite_hex(self.max_abs_error_hex, field="max_abs_error")
        tolerance = _parse_finite_hex(
            self.parity_tolerance_hex, field="parity_tolerance"
        )
        expected_parity = _digest(
            {
                "compiled_output_sha256": self.compiled_output_sha256,
                "flat_output_sha256": self.flat_output_sha256,
                "format": "immer-ooe-controller-language-numerical-parity/v1",
                "max_abs_error_hex": self.max_abs_error_hex,
                "parity_tolerance_hex": self.parity_tolerance_hex,
            }
        )
        if error < 0.0 or tolerance <= 0.0 or error > tolerance:
            raise ValueError("bootstrap numerical parity bound is not satisfied")
        if self.parity_verification_sha256 != expected_parity:
            raise ValueError("bootstrap numerical parity seal is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "accepted_training_episodes": self.accepted_training_episodes,
            "algorithm_sha256": CONTROLLER_LANGUAGE_BOOTSTRAP_ALGORITHM_SHA256,
            "atlas_graph_revision_sha256": self.atlas_graph_revision_sha256,
            "authorized_compute_bank_anchor_sha256": (
                self.authorized_compute_bank_anchor_sha256
            ),
            "charge_receipt_sha256": self.charge_receipt_sha256,
            "compiled_crystal_sha256": self.compiled_crystal_sha256,
            "compiled_operator_count": self.compiled_operator_count,
            "compiled_output_sha256": self.compiled_output_sha256,
            "parity_verification_sha256": self.parity_verification_sha256,
            "max_abs_error_hex": self.max_abs_error_hex,
            "parity_tolerance_hex": self.parity_tolerance_hex,
            "compiled_program_sha256": self.compiled_program_sha256,
            "compiled_receipt_sha256": self.compiled_receipt_sha256,
            "compiler_source_lexicon_state_sha256": (
                self.compiler_source_lexicon_state_sha256
            ),
            "config": self.config.to_record(),
            "controller_snapshot_sha256": self.controller_snapshot_sha256,
            "definition_sha256": self.definition_sha256,
            "equivalent_source_work": self.equivalent_source_work,
            "export_receipt_sha256": self.export_receipt_sha256,
            "final_compute_bank_anchor_sha256": (self.final_compute_bank_anchor_sha256),
            "final_lexicon_state_sha256": self.final_lexicon_state_sha256,
            "flat_operator_count": self.flat_operator_count,
            "flat_output_sha256": self.flat_output_sha256,
            "format": self.FORMAT,
            "frontier_sha256": self.frontier_sha256,
            "heldout_accepted": self.heldout_accepted,
            "heldout_executions": self.heldout_executions,
            "historical_work_released": self.historical_work_released,
            "language_snapshot_sha256": self.language_snapshot_sha256,
            "language_state_sha256": self.language_state_sha256,
            "live_discharge_work": self.live_discharge_work,
            "model_pin_sha256": self.model_pin_sha256,
            "primitive_resolution_sha256": self.primitive_resolution_sha256,
            "run_sha256": self.run_sha256,
            "stable_greedy_cycles": self.stable_greedy_cycles,
            "support_receipt_sha256": self.support_receipt_sha256,
            "training_episodes": self.training_episodes,
            "weight_graph_revision_sha256": self.weight_graph_revision_sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(self.FORMAT, self.to_record())

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> "ControllerLanguageBootstrapReport":
        body = _open(data, schema=cls.FORMAT, label="controller language report")
        expected = set(cls.__dataclass_fields__) | {"algorithm_sha256", "format"}
        if (
            set(body) != expected
            or body.get("format") != cls.FORMAT
            or body.get("algorithm_sha256")
            != CONTROLLER_LANGUAGE_BOOTSTRAP_ALGORITHM_SHA256
        ):
            raise ControllerLanguageBootstrapIntegrityError(
                "controller language report body is invalid"
            )
        try:
            result = cls(
                **{
                    field_name: (
                        ControllerLanguageBootstrapConfig.from_record(
                            body.get("config")
                        )
                        if field_name == "config"
                        else body.get(field_name)
                    )
                    for field_name in cls.__dataclass_fields__
                }
            )
        except (TypeError, ValueError) as exc:
            raise ControllerLanguageBootstrapIntegrityError(
                "controller language report reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise ControllerLanguageBootstrapIntegrityError(
                "controller language report failed canonical reconstruction"
            )
        return result


@dataclass(frozen=True, slots=True)
class ControllerLanguageBootstrapResult:
    report: ControllerLanguageBootstrapReport
    language: ConsequenceMarkovLanguage
    snapshot: LanguageSnapshot
    support: ControllerProgramSupportReceipt
    primitive_resolution: SnapshotComputeResolutionReceipt
    definition: ExecutableWordDefinition
    compiled: CompiledWordReceipt
    lexicon_state: ExecutableLexiconState


def _run_sha256(
    export_receipt: ControllerCrystalExportReceipt,
    config: ControllerLanguageBootstrapConfig,
) -> str:
    return _digest(
        {
            "algorithm_sha256": CONTROLLER_LANGUAGE_BOOTSTRAP_ALGORITHM_SHA256,
            "config_sha256": config.sha256,
            "export_receipt_sha256": export_receipt.sha256,
            "format": "immer-ooe-controller-language-bootstrap-run/v1",
        }
    )


def _state_name(run_sha256: str, kind: str) -> str:
    return f"{_STATE_PREFIX}:{kind}:{require_sha256(run_sha256, field='run_sha256')}"


def _restore_optional(store: CrystalStore, name: str) -> bytes | None:
    try:
        return store.restore_state(name)
    except KeyError:
        return None
    except (OSError, CrystalStoreError) as exc:
        raise ControllerLanguageBootstrapIntegrityError(
            f"persistent {name!r} failed storage integrity"
        ) from exc


def _restore_required(store: CrystalStore, name: str) -> bytes:
    payload = _restore_optional(store, name)
    if payload is None:
        raise ControllerLanguageBootstrapIntegrityError(
            f"persistent {name!r} is missing"
        )
    return payload


def _publish_immutable(store: CrystalStore, name: str, data: bytes) -> None:
    current = _restore_optional(store, name)
    if current is not None:
        if current != data:
            raise ControllerLanguageBootstrapConflictError(
                f"persistent {name!r} contains another bootstrap artifact"
            )
        return
    try:
        store.publish_state(name, data)
    except ManifestConflictError as exc:
        raise ControllerLanguageBootstrapConflictError(
            f"persistent {name!r} publication conflicted"
        ) from exc
    except (OSError, CrystalStoreError) as exc:
        raise ControllerLanguageBootstrapIntegrityError(
            f"persistent {name!r} publication failed"
        ) from exc
    if _restore_optional(store, name) != data:
        raise ControllerLanguageBootstrapIntegrityError(
            f"persistent {name!r} did not roundtrip"
        )


def _training_input(episode: int) -> np.ndarray:
    return np.asarray(
        [
            ((episode * 3 + 1) % 23 - 11) / 7.0,
            ((episode * 5 + 2) % 29 - 14) / 9.0,
            ((episode * 7 + 3) % 31 - 15) / 11.0,
            ((episode * 11 + 4) % 37 - 18) / 13.0,
            ((episode * 13 + 5) % 41 - 20) / 15.0,
        ],
        dtype=np.float64,
    )


def _support_input(occurrence: int, step: int) -> np.ndarray:
    offset = 10_000 + occurrence * _PROGRAM_STEPS + step
    return _training_input(offset)


def _verification_input() -> np.ndarray:
    return np.asarray([-1.25, -0.5, 0.125, 0.875, 1.75], dtype=np.float64)


def _complete_snapshot(
    language: ConsequenceMarkovLanguage,
    contexts: Sequence[str],
) -> LanguageSnapshot | None:
    try:
        return LanguageSnapshot.freeze(
            language,
            contexts=tuple(contexts),
            require_complete=True,
        )
    except MarkovLanguageError:
        return None


def _train_language(
    export_receipt: ControllerCrystalExportReceipt,
    compute_bank: ComputeCrystalBank,
    config: ControllerLanguageBootstrapConfig,
) -> tuple[
    ConsequenceMarkovLanguage,
    ConsequenceLanguageBridge,
    ControllerCrystalLanguageExecutor,
    int,
    int,
]:
    frontier = export_receipt.frontier
    word_prefix = _digest(
        {
            "export_receipt_sha256": export_receipt.sha256,
            "format": "immer-ooe-controller-language-vocabulary/v1",
        }
    )[:16]
    vocabulary = tuple(
        f"cc-{word_prefix}-{index:04d}" for index in range(len(frontier.actions))
    )
    language = ConsequenceMarkovLanguage(
        frontier,
        vocabulary,
        seed=config.seed,
        minimum_visits=config.minimum_visits,
        minimum_value=config.minimum_value,
        minimum_margin=config.minimum_margin,
        context_full_weight_visits=config.context_full_weight_visits,
    )
    bridge = ConsequenceLanguageBridge(language)
    executor = ControllerCrystalLanguageExecutor(export_receipt, compute_bank)
    actions = language.action_ids
    train_context = "controller-language-train"
    probe_context = "controller-language-heldout-probe"
    minimum_exploration = max(128, len(actions) * len(actions) * 32)
    exploration_span = min(
        max(1, config.max_training_episodes // 2), minimum_exploration
    )
    accepted = 0
    stable = 0
    cycle: list[tuple[str, str, bool]] = []
    for episode in range(config.max_training_episodes):
        if episode < exploration_span:
            fraction = episode / max(1, exploration_span - 1)
            epsilon = config.exploration_start + fraction * (
                config.exploration_end - config.exploration_start
            )
        else:
            epsilon = 0.0
        intent = actions[episode % len(actions)]
        decision = bridge.begin(intent, train_context, epsilon=epsilon)
        outcome = executor.execute(decision, _training_input(episode))
        commit = bridge.settle_controller_crystal(
            decision,
            outcome,
            export_receipt,
            compute_bank,
            learning_rate=config.learning_rate,
            executor=executor,
        )
        accepted += int(commit.accepted)
        if epsilon != 0.0:
            stable = 0
            cycle.clear()
            continue
        cycle.append((intent, commit.word_id, commit.accepted))
        if len(cycle) != len(actions):
            continue
        snapshot = _complete_snapshot(language, (train_context, probe_context))
        canonical = snapshot is not None and all(
            accepted_step
            and snapshot.encode_action(action_id, train_context) == word_id
            and snapshot.decode_word(word_id, probe_context) == action_id
            for action_id, word_id, accepted_step in cycle
        )
        stable = stable + 1 if canonical else 0
        cycle.clear()
        if stable >= config.stable_greedy_cycles:
            return language, bridge, executor, accepted, stable
    raise ControllerLanguageConvergenceError(
        "controller language did not converge within max_training_episodes"
    )


def _collect_support(
    *,
    bridge: ConsequenceLanguageBridge,
    executor: ControllerCrystalLanguageExecutor,
    export_receipt: ControllerCrystalExportReceipt,
    compute_bank: ComputeCrystalBank,
    config: ControllerLanguageBootstrapConfig,
) -> tuple[LanguageSnapshot, ControllerProgramSupportReceipt]:
    train_context = "controller-language-train"
    frozen_language = bridge.language.to_bytes()
    snapshot = LanguageSnapshot.freeze(
        bridge.language,
        contexts=(train_context, "controller-language-heldout-freeze"),
        require_complete=True,
    )
    actions = bridge.language.action_ids
    program_actions = tuple(actions[index % len(actions)] for index in range(4))
    program_words = tuple(
        cast(str, snapshot.encode_action(action_id, train_context))
        for action_id in program_actions
    )
    if any(word is None for word in program_words):
        raise ControllerLanguageConvergenceError(
            "converged language cannot encode the four-action program"
        )
    occurrences: list[ControllerProgramOccurrence] = []
    max_attempts = _SUPPORT_OCCURRENCES * 8
    for attempt in range(max_attempts):
        context = f"controller-language-heldout-{attempt:04d}"
        evaluator = ConsequenceMarkovLanguage.from_bytes(
            frozen_language, expected_frontier=export_receipt.frontier
        )
        evaluation_bridge = ConsequenceLanguageBridge(evaluator)
        commits: list[LanguageOutcomeCommitReceipt] = []
        outcomes: list[ControllerCrystalOutcomeReceipt] = []
        for step, action_id in enumerate(program_actions):
            decision = evaluation_bridge.begin(action_id, context, epsilon=0.0)
            outcome = executor.execute(decision, _support_input(len(occurrences), step))
            commit = evaluation_bridge.settle_controller_crystal(
                decision,
                outcome,
                export_receipt,
                compute_bank,
                learning_rate=config.learning_rate,
                executor=executor,
            )
            commits.append(commit)
            outcomes.append(outcome)
        if (
            all(commit.accepted for commit in commits)
            and tuple(commit.word_id for commit in commits) == program_words
        ):
            occurrences.append(
                ControllerProgramOccurrence(
                    ordinal=len(occurrences),
                    context_id=context,
                    commits=tuple(commits),
                    outcomes=tuple(outcomes),
                )
            )
            if len(occurrences) == _SUPPORT_OCCURRENCES:
                break
    if len(occurrences) != _SUPPORT_OCCURRENCES:
        raise ControllerLanguageConvergenceError(
            "greedy held-out execution did not produce three identical supports"
        )
    if bridge.language.to_bytes() != frozen_language:
        raise ControllerLanguageBootstrapIntegrityError(
            "held-out evaluation mutated the persisted learner"
        )
    support = ControllerProgramSupportReceipt(
        export_receipt_sha256=export_receipt.sha256,
        frontier_sha256=export_receipt.frontier.sha256,
        language_snapshot=snapshot,
        action_ids=program_actions,
        word_ids=program_words,
        occurrences=tuple(occurrences),
        verifier_sha256=CONTROLLER_PROGRAM_SUPPORT_VERIFIER_SHA256,
    )
    support.verify(export_receipt, compute_bank)
    return snapshot, support


def _compile_supported_program(
    *,
    snapshot: LanguageSnapshot,
    support: ControllerProgramSupportReceipt,
    export_receipt: ControllerCrystalExportReceipt,
    compute_bank: ComputeCrystalBank,
    lexicon_store: CrystalStore,
) -> tuple[
    SnapshotComputeResolutionReceipt,
    ExecutableWordDefinition,
    CompiledWordReceipt,
    ExecutableLexiconState,
    Any,
    Any,
]:
    primitive_bindings, resolution = resolve_snapshot_compute_bindings(
        snapshot, export_receipt.frontier
    )
    initial = ExecutableLexiconState.initial(
        primitive_bindings,
        language_snapshot_sha256=snapshot.sha256,
        frontier_sha256=export_receipt.frontier.sha256,
        authority_hashes=export_receipt.frontier.authority_hashes,
    )
    lexicon_bank = ExecutableLexiconBank(lexicon_store, initial_state=initial)
    definition = ExecutableWordDefinition.create(
        f"controller-program-{support.sha256[:32]}",
        support.word_ids,
        language_snapshot_sha256=snapshot.sha256,
        frontier_sha256=export_receipt.frontier.sha256,
        authority_hashes=export_receipt.frontier.authority_hashes,
    )
    head = lexicon_bank.head()
    existing_definition = head.definition_map.get(definition.new_word_id)
    if existing_definition is not None and existing_definition != definition:
        raise ControllerLanguageBootstrapConflictError(
            "lexicon contains another supported-program definition"
        )
    if existing_definition is None:
        head = lexicon_bank.append_definition(
            definition, expected_head_sha256=head.sha256
        )
    existing_compiled = next(
        (
            item
            for item in head.compiled_receipts
            if item.word_id == definition.new_word_id
        ),
        None,
    )
    if existing_compiled is None:
        compiler_state = head
        compiler = ExecutableWordCompiler(compiler_state, compute_bank)
        compiled = compiler.compile(definition.new_word_id)
        final_state = lexicon_bank.append_compiled_receipt(
            compiled, expected_head_sha256=compiler_state.sha256
        )
    else:
        compiled = existing_compiled
        try:
            compiler_payload = lexicon_store.restore_state(
                ExecutableLexiconBank.history_state_name(
                    compiled.source_lexicon_state_sha256
                )
            )
            compiler_state = ExecutableLexiconState.from_bytes(compiler_payload)
        except (KeyError, CrystalStoreError, ValueError) as exc:
            raise ControllerLanguageBootstrapIntegrityError(
                "compiled word lost its exact source lexicon state"
            ) from exc
        compiler = ExecutableWordCompiler(compiler_state, compute_bank)
        final_state = head
    if (
        compiled.artifact_kind != "crystal"
        or not compiled.constant_discharge
        or compiled.charge_sha256 is None
        or compiled.expanded_primitive_actions != _PROGRAM_STEPS
    ):
        raise ControllerLanguageBootstrapIntegrityError(
            "supported four-action word did not compile to one charged Crystal"
        )
    value = _verification_input()
    try:
        charge = compute_bank.restore_charge(compiled.charge_sha256)
        flat = ComputeCrystalVM(compute_bank).execute(
            charge.source_program_sha256, value
        )
        fused = compiler.execute(compiled, value)
    except ComputeCrystalError as exc:
        raise ControllerLanguageBootstrapIntegrityError(
            "compiled supported-program verification failed"
        ) from exc
    if (
        flat.receipt.executed_operator_count != _PROGRAM_STEPS
        or fused.receipt.executed_operator_count != 1
        or fused.receipt.equivalent_unfused_source_work
        != flat.receipt.live_discharge_work
        or fused.receipt.historical_work_released <= 0
    ):
        raise ControllerLanguageBootstrapIntegrityError(
            "compiled Crystal differs from flat execution or work accounting"
        )
    scale = max(
        1.0,
        float(np.max(np.abs(flat.output))),
        float(np.max(np.abs(fused.output))),
    )
    tolerance = (
        np.finfo(np.float64).eps * 64.0 * _PROGRAM_STEPS * len(OOE_ACTIONS) * scale
    )
    error = float(np.max(np.abs(flat.output - fused.output)))
    if not math.isfinite(error) or error > tolerance:
        raise ControllerLanguageBootstrapIntegrityError(
            "compiled Crystal exceeds the deterministic float64 parity bound"
        )
    return resolution, definition, compiled, final_state, flat, fused


def _restore_result(
    *,
    report: ControllerLanguageBootstrapReport,
    export_receipt: ControllerCrystalExportReceipt,
    compute_bank: ComputeCrystalBank,
    state_store: CrystalStore,
    lexicon_store: CrystalStore,
) -> ControllerLanguageBootstrapResult:
    if (
        report.export_receipt_sha256 != export_receipt.sha256
        or report.frontier_sha256 != export_receipt.frontier.sha256
        or report.run_sha256 != _run_sha256(export_receipt, report.config)
    ):
        raise ControllerLanguageBootstrapConflictError(
            "persisted report belongs to another controller-language run"
        )
    language_bank = ConsequenceLanguageStateBank(
        state_store, state_name=_state_name(report.run_sha256, "language")
    )
    language = language_bank.restore(
        expected_frontier=export_receipt.frontier,
        trusted_state_sha256=report.language_state_sha256,
    )
    snapshot = LanguageSnapshot.from_bytes(
        _restore_required(state_store, _state_name(report.run_sha256, "snapshot"))
    )
    support = ControllerProgramSupportReceipt.from_bytes(
        _restore_required(state_store, _state_name(report.run_sha256, "support"))
    )
    resolution = SnapshotComputeResolutionReceipt.from_bytes(
        _restore_required(state_store, _state_name(report.run_sha256, "resolution"))
    )
    try:
        lexicon_payload = lexicon_store.restore_state(
            ExecutableLexiconBank.history_state_name(report.final_lexicon_state_sha256)
        )
        lexicon_state = ExecutableLexiconState.from_bytes(lexicon_payload)
    except (KeyError, OSError, CrystalStoreError, ValueError) as exc:
        raise ControllerLanguageBootstrapIntegrityError(
            "persisted report lost its lexicon state"
        ) from exc
    definition = next(
        (
            item
            for item in lexicon_state.definitions
            if item.sha256 == report.definition_sha256
        ),
        None,
    )
    compiled = next(
        (
            item
            for item in lexicon_state.compiled_receipts
            if item.sha256 == report.compiled_receipt_sha256
        ),
        None,
    )
    if (
        snapshot.sha256 != report.language_snapshot_sha256
        or support.sha256 != report.support_receipt_sha256
        or support.language_snapshot != snapshot
        or resolution.sha256 != report.primitive_resolution_sha256
        or resolution.language_snapshot_sha256 != snapshot.sha256
        or resolution.frontier_sha256 != export_receipt.frontier.sha256
        or lexicon_state.sha256 != report.final_lexicon_state_sha256
        or definition is None
        or compiled is None
    ):
        raise ControllerLanguageBootstrapIntegrityError(
            "persisted report artifact hashes changed"
        )
    if (
        language.sha256 != snapshot.learner_state_sha256
        or language.training_episodes != report.training_episodes
        or definition.child_word_ids != support.word_ids
        or compiled.definition_sha256 != definition.sha256
        or compiled.artifact_kind != "crystal"
        or compiled.artifact_sha256 != report.compiled_crystal_sha256
        or compiled.program_sha256 != report.compiled_program_sha256
        or compiled.charge_sha256 != report.charge_receipt_sha256
        or compiled.source_lexicon_state_sha256
        != report.compiler_source_lexicon_state_sha256
        or not compiled.constant_discharge
        or compiled.expanded_primitive_actions != _PROGRAM_STEPS
    ):
        raise ControllerLanguageBootstrapIntegrityError(
            "persisted language or compiled word differs from its report"
        )
    if (
        report.controller_snapshot_sha256 != export_receipt.controller_snapshot_sha256
        or report.model_pin_sha256 != export_receipt.model_pin_sha256
        or report.weight_graph_revision_sha256
        != export_receipt.weight_graph_revision_sha256
        or report.atlas_graph_revision_sha256
        != export_receipt.atlas_graph_revision.sha256
        or report.authorized_compute_bank_anchor_sha256
        != export_receipt.final_bank_anchor_sha256
    ):
        raise ControllerLanguageBootstrapIntegrityError(
            "persisted report controller authorities changed"
        )
    try:
        compiler_state = ExecutableLexiconState.from_bytes(
            lexicon_store.restore_state(
                ExecutableLexiconBank.history_state_name(
                    compiled.source_lexicon_state_sha256
                )
            )
        )
        compiler = ExecutableWordCompiler(compiler_state, compute_bank)
        try:
            compute_bank.assert_descends_from(report.final_compute_bank_anchor_sha256)
        except ComputeCrystalError:
            rebuilt = compiler.compile(definition.new_word_id)
            if rebuilt != compiled:
                raise ControllerLanguageBootstrapConflictError(
                    "equivalent compute bank rebuilt another compiled word"
                )
            compute_bank.assert_descends_from(report.final_compute_bank_anchor_sha256)
        charge = compute_bank.restore_charge(cast(str, compiled.charge_sha256))
        value = _verification_input()
        flat = ComputeCrystalVM(compute_bank).execute(
            charge.source_program_sha256, value
        )
        fused = compiler.execute(compiled, value)
    except ControllerLanguageBootstrapConflictError:
        raise
    except (
        KeyError,
        OSError,
        CrystalStoreError,
        ComputeCrystalError,
        ExecutableLexiconError,
        ValueError,
    ) as exc:
        raise ControllerLanguageBootstrapIntegrityError(
            "persisted compiled word failed exact replay"
        ) from exc
    error = float(np.max(np.abs(flat.output - fused.output)))
    tolerance = (
        np.finfo(np.float64).eps
        * 64.0
        * _PROGRAM_STEPS
        * len(OOE_ACTIONS)
        * max(
            1.0,
            float(np.max(np.abs(flat.output))),
            float(np.max(np.abs(fused.output))),
        )
    )
    if (
        flat.receipt.output_sha256 != report.flat_output_sha256
        or fused.receipt.output_sha256 != report.compiled_output_sha256
        or error.hex() != report.max_abs_error_hex
        or tolerance.hex() != report.parity_tolerance_hex
        or flat.receipt.executed_operator_count != report.flat_operator_count
        or fused.receipt.executed_operator_count != report.compiled_operator_count
        or fused.receipt.equivalent_unfused_source_work != report.equivalent_source_work
        or fused.receipt.live_discharge_work != report.live_discharge_work
        or fused.receipt.historical_work_released != report.historical_work_released
    ):
        raise ControllerLanguageBootstrapIntegrityError(
            "persisted compilation parity or work accounting changed"
        )
    support.verify(export_receipt, compute_bank)
    return ControllerLanguageBootstrapResult(
        report=report,
        language=language,
        snapshot=snapshot,
        support=support,
        primitive_resolution=resolution,
        definition=definition,
        compiled=compiled,
        lexicon_state=lexicon_state,
    )


def bootstrap_controller_crystal_language(
    export: ControllerCrystalExport | ControllerCrystalExportReceipt,
    compute_bank: ComputeCrystalBank,
    *,
    state_store: CrystalStore | str | os.PathLike[str],
    lexicon_store: CrystalStore | str | os.PathLike[str],
    config: ControllerLanguageBootstrapConfig | None = None,
) -> ControllerLanguageBootstrapResult:
    """Train, prove, persist, and compile one controller-Crystal language."""

    if isinstance(export, ControllerCrystalExport):
        export_receipt = export.receipt
    elif isinstance(export, ControllerCrystalExportReceipt):
        export_receipt = export
    else:
        raise TypeError("export must be a ControllerCrystalExport or receipt")
    if not isinstance(compute_bank, ComputeCrystalBank):
        raise TypeError("compute_bank must be a ComputeCrystalBank")
    exact_config = config or ControllerLanguageBootstrapConfig()
    if not isinstance(exact_config, ControllerLanguageBootstrapConfig):
        raise TypeError("config must be a ControllerLanguageBootstrapConfig")
    state = (
        state_store
        if isinstance(state_store, CrystalStore)
        else CrystalStore(state_store)
    )
    lexicon = (
        lexicon_store
        if isinstance(lexicon_store, CrystalStore)
        else CrystalStore(lexicon_store)
    )
    try:
        export_receipt.restore_crystals(compute_bank)
    except Exception as exc:
        raise ControllerLanguageBootstrapIntegrityError(
            "controller export cannot mount its compute bank"
        ) from exc
    run_sha = _run_sha256(export_receipt, exact_config)
    report_name = _state_name(run_sha, "report")
    prior_report = _restore_optional(state, report_name)
    if prior_report is not None:
        return _restore_result(
            report=ControllerLanguageBootstrapReport.from_bytes(prior_report),
            export_receipt=export_receipt,
            compute_bank=compute_bank,
            state_store=state,
            lexicon_store=lexicon,
        )

    language, bridge, executor, accepted, stable = _train_language(
        export_receipt, compute_bank, exact_config
    )
    snapshot, support = _collect_support(
        bridge=bridge,
        executor=executor,
        export_receipt=export_receipt,
        compute_bank=compute_bank,
        config=exact_config,
    )
    final_language = bridge.language
    language_bank = ConsequenceLanguageStateBank(
        state, state_name=_state_name(run_sha, "language")
    )
    language_bank.initialize(final_language)
    _publish_immutable(state, _state_name(run_sha, "snapshot"), snapshot.to_bytes())
    _publish_immutable(state, _state_name(run_sha, "support"), support.to_bytes())

    resolution, definition, compiled, final_lexicon, flat, fused = (
        _compile_supported_program(
            snapshot=snapshot,
            support=support,
            export_receipt=export_receipt,
            compute_bank=compute_bank,
            lexicon_store=lexicon,
        )
    )
    _publish_immutable(state, _state_name(run_sha, "resolution"), resolution.to_bytes())
    if fused.receipt.charge_basis_sha256 != compiled.charge_sha256:
        raise ControllerLanguageBootstrapIntegrityError(
            "compiled execution lost its exact charge basis"
        )
    report = ControllerLanguageBootstrapReport(
        run_sha256=run_sha,
        config=exact_config,
        export_receipt_sha256=export_receipt.sha256,
        controller_snapshot_sha256=export_receipt.controller_snapshot_sha256,
        model_pin_sha256=export_receipt.model_pin_sha256,
        weight_graph_revision_sha256=export_receipt.weight_graph_revision_sha256,
        atlas_graph_revision_sha256=export_receipt.atlas_graph_revision.sha256,
        frontier_sha256=export_receipt.frontier.sha256,
        authorized_compute_bank_anchor_sha256=(export_receipt.final_bank_anchor_sha256),
        final_compute_bank_anchor_sha256=compute_bank.current_anchor_sha256(),
        training_episodes=final_language.training_episodes,
        accepted_training_episodes=accepted,
        stable_greedy_cycles=stable,
        heldout_executions=_PROGRAM_STEPS * _SUPPORT_OCCURRENCES,
        heldout_accepted=_PROGRAM_STEPS * _SUPPORT_OCCURRENCES,
        language_state_sha256=final_language.sha256,
        language_snapshot_sha256=snapshot.sha256,
        support_receipt_sha256=support.sha256,
        primitive_resolution_sha256=resolution.sha256,
        definition_sha256=definition.sha256,
        compiled_receipt_sha256=compiled.sha256,
        compiler_source_lexicon_state_sha256=(compiled.source_lexicon_state_sha256),
        final_lexicon_state_sha256=final_lexicon.sha256,
        compiled_crystal_sha256=compiled.artifact_sha256,
        compiled_program_sha256=compiled.program_sha256,
        charge_receipt_sha256=cast(str, compiled.charge_sha256),
        flat_output_sha256=flat.receipt.output_sha256,
        compiled_output_sha256=fused.receipt.output_sha256,
        parity_verification_sha256=_digest(
            {
                "compiled_output_sha256": fused.receipt.output_sha256,
                "flat_output_sha256": flat.receipt.output_sha256,
                "format": "immer-ooe-controller-language-numerical-parity/v1",
                "max_abs_error_hex": float(
                    np.max(np.abs(flat.output - fused.output))
                ).hex(),
                "parity_tolerance_hex": (
                    np.finfo(np.float64).eps
                    * 64.0
                    * _PROGRAM_STEPS
                    * len(OOE_ACTIONS)
                    * max(
                        1.0,
                        float(np.max(np.abs(flat.output))),
                        float(np.max(np.abs(fused.output))),
                    )
                ).hex(),
            }
        ),
        max_abs_error_hex=float(np.max(np.abs(flat.output - fused.output))).hex(),
        parity_tolerance_hex=(
            np.finfo(np.float64).eps
            * 64.0
            * _PROGRAM_STEPS
            * len(OOE_ACTIONS)
            * max(
                1.0,
                float(np.max(np.abs(flat.output))),
                float(np.max(np.abs(fused.output))),
            )
        ).hex(),
        flat_operator_count=flat.receipt.executed_operator_count,
        compiled_operator_count=fused.receipt.executed_operator_count,
        equivalent_source_work=fused.receipt.equivalent_unfused_source_work,
        live_discharge_work=fused.receipt.live_discharge_work,
        historical_work_released=fused.receipt.historical_work_released,
    )
    _publish_immutable(state, report_name, report.to_bytes())
    return ControllerLanguageBootstrapResult(
        report=report,
        language=final_language,
        snapshot=snapshot,
        support=support,
        primitive_resolution=resolution,
        definition=definition,
        compiled=compiled,
        lexicon_state=final_lexicon,
    )


__all__ = [
    "CONTROLLER_LANGUAGE_BOOTSTRAP_ALGORITHM_SHA256",
    "CONTROLLER_LANGUAGE_BOOTSTRAP_CONFIG_SCHEMA",
    "CONTROLLER_LANGUAGE_BOOTSTRAP_REPORT_SCHEMA",
    "CONTROLLER_PROGRAM_SUPPORT_SCHEMA",
    "CONTROLLER_PROGRAM_SUPPORT_VERIFIER_SHA256",
    "ControllerLanguageBootstrapConfig",
    "ControllerLanguageBootstrapConflictError",
    "ControllerLanguageBootstrapError",
    "ControllerLanguageBootstrapIntegrityError",
    "ControllerLanguageBootstrapReport",
    "ControllerLanguageBootstrapResult",
    "ControllerLanguageConvergenceError",
    "ControllerProgramOccurrence",
    "ControllerProgramSupportReceipt",
    "bootstrap_controller_crystal_language",
]
