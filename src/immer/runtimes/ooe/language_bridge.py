"""Production bridge from authenticated OoE outcomes to executable language."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Iterator, cast

import numpy as np

from .algebra_agents import (
    AlgebraRouterState,
    AlgebraSelectionReceipt,
    OperatorAlgebraCandidate,
    VerifierBoundOutcome,
)
from .compute_graph import ComputeOperatorGraph, ComputeOperatorGraphState
from .compute_crystals import ComputeCrystalBank, ComputeCrystalError, tensor_sha256
from .controller_crystal_bridge import (
    CONTROLLER_CRYSTAL_LANGUAGE_REWARD_POLICY_SHA256,
    ControllerCrystalExportReceipt,
)
from .crystal import CrystalStore, CrystalStoreError, ManifestConflictError
from .demand_execution import DemandRoutedExecutionReceipt
from .executable_lexicon import (
    CompiledWordReceipt,
    ExecutableWordDefinition,
    PrimitiveWordBinding,
)
from .identity import canonical_json_bytes, require_sha256
from .markov_language import (
    ActionBinding,
    ActionFrontier,
    ConsequenceFeedback,
    ConsequenceMarkovLanguage,
    LanguageSnapshot,
    MarkovLanguageConflictError,
    ReceiverDecision,
    SenderDecision,
    SenderEmission,
)
from .options import MacroOption


LANGUAGE_ROUTING_DECISION_SCHEMA = "immer-ooe-language-routing-decision/v1"
LANGUAGE_OUTCOME_COMMIT_SCHEMA = "immer-ooe-language-outcome-commit/v1"
CONTROLLER_CRYSTAL_OUTCOME_SCHEMA = "immer-ooe-controller-crystal-outcome/v1"
FRONTIER_MIGRATION_SCHEMA = "immer-ooe-language-frontier-migration/v1"
VERIFIED_WORD_TRAJECTORY_SCHEMA = "immer-ooe-verified-word-trajectory/v1"
LANGUAGE_MACRO_DISCOVERY_SCHEMA = "immer-ooe-language-macro-discovery/v1"
ROUTE_WORD_RESOLUTION_SCHEMA = "immer-ooe-route-word-resolution/v1"
MACRO_ACTION_PROMOTION_SCHEMA = "immer-ooe-language-macro-action-promotion/v1"
LANGUAGE_REVISION_PROMOTION_SCHEMA = "immer-ooe-language-revision-promotion/v1"
ROUTE_FRONTIER_TRANSITION_SCHEMA = "immer-ooe-route-frontier-transition/v1"
SNAPSHOT_COMPUTE_RESOLUTION_SCHEMA = "immer-ooe-snapshot-compute-resolution/v1"
LANGUAGE_STATE_NAME = "ooe-production-markov-language/v1"

MAX_TRAJECTORIES = 100_000
MAX_TRAJECTORY_LENGTH = 256
MAX_DISCOVERED_DEFINITIONS = 16_384
MAX_MACRO_LENGTH = 64
MAX_RECEIPT_BYTES = 64 * 1024 * 1024


class LanguageBridgeError(RuntimeError):
    """Base error for the production consequence-language bridge."""


class LanguageBridgeIntegrityError(LanguageBridgeError):
    """An action, outcome, revision, or persistence binding failed."""


class LanguageBridgeConflictError(LanguageBridgeError):
    """A state CAS, episode, or frontier transition conflicted."""


def _sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


DEMAND_LANGUAGE_REWARD_POLICY_SHA256 = _sha256(
    {
        "failure": "use-sealed-demand-outcome-reward",
        "format": "immer-ooe-demand-language-reward/v1",
        "success": "use-sealed-demand-outcome-reward",
    }
)
ALGEBRA_LANGUAGE_REWARD_POLICY_SHA256 = _sha256(
    {
        "failure": -1,
        "format": "immer-ooe-algebra-language-reward/v1",
        "success": 1,
    }
)
CONTROLLER_CRYSTAL_OUTCOME_VERIFIER_SHA256 = _sha256(
    {
        "accepted": "intended-action-and-output-sha256-equal-selected",
        "execution": "restore-both-compute-crystals-and-recompute",
        "format": "immer-ooe-controller-crystal-outcome-verifier/v1",
        "input": "canonical-float64-vector",
    }
)


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


def _strict_json(data: bytes, *, label: str) -> object:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if len(data) > MAX_RECEIPT_BYTES:
        raise LanguageBridgeIntegrityError(f"{label} exceeds its byte bound")

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
        raise LanguageBridgeIntegrityError(f"{label} is invalid JSON") from exc
    if canonical_json_bytes(value) != data:
        raise LanguageBridgeIntegrityError(f"{label} is not canonical JSON")
    return value


def _seal(schema: str, body: Mapping[str, object]) -> bytes:
    exact = dict(body)
    return canonical_json_bytes(
        {
            "body": exact,
            "body_sha256": _sha256(exact),
            "schema": schema,
        }
    )


def _open(data: bytes, *, schema: str, label: str) -> Mapping[str, object]:
    value = _strict_json(data, label=label)
    if (
        not isinstance(value, Mapping)
        or set(value) != {"body", "body_sha256", "schema"}
        or value.get("schema") != schema
        or not isinstance(value.get("body"), Mapping)
    ):
        raise LanguageBridgeIntegrityError(f"{label} envelope is invalid")
    body = cast(Mapping[str, object], value.get("body"))
    if value.get("body_sha256") != _sha256(body):
        raise LanguageBridgeIntegrityError(f"{label} body hash mismatch")
    return body


def route_action_id(route_sha256: str) -> str:
    return "route:" + require_sha256(route_sha256, field="route_sha256")


def candidate_action_id(candidate_sha256: str) -> str:
    return "candidate:" + require_sha256(candidate_sha256, field="candidate_sha256")


def _merge_authorities(
    required: Mapping[str, str],
    extra: Mapping[str, str] | None,
) -> dict[str, str]:
    result = dict(required)
    if extra is not None:
        for name, digest in extra.items():
            exact_name = _identifier(name, field="authority name")
            exact_digest = require_sha256(digest, field=f"authority[{name!r}]")
            prior = result.get(exact_name)
            if prior is not None and prior != exact_digest:
                raise ValueError(f"authority {exact_name!r} conflicts")
            result[exact_name] = exact_digest
    return result


def build_route_action_frontier(
    graph_state: ComputeOperatorGraphState,
    *,
    context_schema_sha256: str,
    route_sha256s: Sequence[str] | None = None,
    extra_authorities: Mapping[str, str] | None = None,
) -> ActionFrontier:
    """Build an exact language frontier over materialized graph routes."""

    if not isinstance(graph_state, ComputeOperatorGraphState):
        raise TypeError("graph_state must be a ComputeOperatorGraphState")
    available = {route.sha256: route for route in graph_state.materialized_routes}
    if route_sha256s is None:
        routes = tuple(available.values())
    else:
        wanted = tuple(
            require_sha256(value, field="route_sha256") for value in route_sha256s
        )
        if len(set(wanted)) != len(wanted):
            raise ValueError("route frontier contains duplicate addresses")
        try:
            routes = tuple(available[address] for address in wanted)
        except KeyError as exc:
            raise LanguageBridgeIntegrityError(
                "route frontier references another graph revision"
            ) from exc
    if not routes:
        raise ValueError("route frontier must not be empty")
    route_contracts = [
        {
            "charge_basis_sha256": route.charge_basis_sha256,
            "evidence_sha256s": list(route.evidence_sha256s),
            "executable_program_sha256": route.executable_program_sha256,
            "input_abi_sha256": route.input_abi_sha256,
            "output_abi_sha256": route.output_abi_sha256,
            "route_sha256": route.sha256,
            "verifier_sha256s": list(route.verifier_sha256s),
        }
        for route in sorted(routes, key=lambda item: item.sha256)
    ]
    contract_sha256 = _sha256(
        {
            "format": "immer-ooe-language-route-action-schema/v1",
            "graph_state_sha256": graph_state.sha256,
            "routes": route_contracts,
        }
    )
    authorities = _merge_authorities(
        {
            "graph-state": graph_state.sha256,
            "language-reward-policy": DEMAND_LANGUAGE_REWARD_POLICY_SHA256,
            "route-contract-set": contract_sha256,
        },
        extra_authorities,
    )
    return ActionFrontier.create(
        tuple(
            ActionBinding(
                action_id=route_action_id(route.sha256),
                artifact_kind="materialized-route",
                artifact_sha256=route.sha256,
            )
            for route in routes
        ),
        action_schema_sha256=contract_sha256,
        context_schema_sha256=context_schema_sha256,
        authority_hashes=authorities,
    )


@dataclass(frozen=True, slots=True)
class RouteFrontierTransitionReceipt:
    prior_frontier_sha256: str
    next_frontier_sha256: str
    prior_graph_state_sha256: str
    next_graph_state_sha256: str
    graph_history_sha256s: tuple[str, ...]
    graph_state_payloads: tuple[bytes, ...]
    retained_route_sha256s: tuple[str, ...]
    added_route_sha256s: tuple[str, ...]
    changed_authority_names: tuple[str, ...]

    FORMAT = ROUTE_FRONTIER_TRANSITION_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "prior_frontier_sha256",
            "next_frontier_sha256",
            "prior_graph_state_sha256",
            "next_graph_state_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        for field_name in (
            "graph_history_sha256s",
            "retained_route_sha256s",
            "added_route_sha256s",
        ):
            values = tuple(
                require_sha256(value, field=field_name)
                for value in getattr(self, field_name)
            )
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} contains duplicates")
            object.__setattr__(self, field_name, values)
        if (
            not self.graph_history_sha256s
            or self.graph_history_sha256s[0] != self.prior_graph_state_sha256
            or self.graph_history_sha256s[-1] != self.next_graph_state_sha256
            or set(self.retained_route_sha256s) & set(self.added_route_sha256s)
        ):
            raise ValueError("route frontier transition inventories are invalid")
        payloads = tuple(self.graph_state_payloads)
        if len(payloads) != len(self.graph_history_sha256s) or any(
            not isinstance(payload, bytes) for payload in payloads
        ):
            raise ValueError("route frontier graph payload inventory is invalid")
        try:
            states = tuple(
                ComputeOperatorGraphState.from_bytes(payload) for payload in payloads
            )
        except Exception as exc:
            raise ValueError("route frontier graph payload is invalid") from exc
        if tuple(state.sha256 for state in states) != self.graph_history_sha256s:
            raise ValueError("route frontier graph payload hashes changed")
        for previous, current in zip(states, states[1:], strict=False):
            try:
                ComputeOperatorGraph._validate_extension(previous, current)
            except Exception as exc:
                raise ValueError(
                    "route frontier graph payloads are not append-only"
                ) from exc
        first_routes = {route.sha256 for route in states[0].materialized_routes}
        final_routes = {route.sha256 for route in states[-1].materialized_routes}
        if (
            not set(self.retained_route_sha256s) <= first_routes
            or not set(self.retained_route_sha256s) <= final_routes
            or not set(self.added_route_sha256s) <= final_routes
        ):
            raise ValueError(
                "route frontier transition routes are absent from graph payloads"
            )
        object.__setattr__(self, "graph_state_payloads", payloads)
        names = tuple(
            sorted(
                _identifier(value, field="changed_authority_name")
                for value in self.changed_authority_names
            )
        )
        if len(set(names)) != len(names):
            raise ValueError("changed authority names contain duplicates")
        object.__setattr__(self, "changed_authority_names", names)

    def to_record(self) -> dict[str, object]:
        return {
            "added_route_sha256s": list(self.added_route_sha256s),
            "changed_authority_names": list(self.changed_authority_names),
            "format": self.FORMAT,
            "graph_history_sha256s": list(self.graph_history_sha256s),
            "graph_state_payloads_base64": [
                base64.b64encode(payload).decode("ascii")
                for payload in self.graph_state_payloads
            ],
            "next_frontier_sha256": self.next_frontier_sha256,
            "next_graph_state_sha256": self.next_graph_state_sha256,
            "prior_frontier_sha256": self.prior_frontier_sha256,
            "prior_graph_state_sha256": self.prior_graph_state_sha256,
            "retained_route_sha256s": list(self.retained_route_sha256s),
        }

    def to_bytes(self) -> bytes:
        return _seal(self.FORMAT, self.to_record())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "RouteFrontierTransitionReceipt":
        body = _open(data, schema=cls.FORMAT, label="route frontier transition")
        expected = {
            "added_route_sha256s",
            "changed_authority_names",
            "format",
            "graph_history_sha256s",
            "graph_state_payloads_base64",
            "next_frontier_sha256",
            "next_graph_state_sha256",
            "prior_frontier_sha256",
            "prior_graph_state_sha256",
            "retained_route_sha256s",
        }
        if set(body) != expected or body.get("format") != cls.FORMAT:
            raise LanguageBridgeIntegrityError(
                "route frontier transition body is invalid"
            )
        collections = {
            name: body.get(name)
            for name in (
                "added_route_sha256s",
                "changed_authority_names",
                "graph_history_sha256s",
                "graph_state_payloads_base64",
                "retained_route_sha256s",
            )
        }
        if any(not isinstance(value, list) for value in collections.values()):
            raise LanguageBridgeIntegrityError(
                "route frontier transition inventory is invalid"
            )
        try:
            encoded_payloads = collections["graph_state_payloads_base64"]
            assert isinstance(encoded_payloads, list)
            payloads = tuple(
                base64.b64decode(value, validate=True) for value in encoded_payloads
            )
            result = cls(
                prior_frontier_sha256=cast(str, body.get("prior_frontier_sha256")),
                next_frontier_sha256=cast(str, body.get("next_frontier_sha256")),
                prior_graph_state_sha256=cast(
                    str, body.get("prior_graph_state_sha256")
                ),
                next_graph_state_sha256=cast(str, body.get("next_graph_state_sha256")),
                graph_history_sha256s=tuple(
                    cast(list[str], collections["graph_history_sha256s"])
                ),
                graph_state_payloads=payloads,
                retained_route_sha256s=tuple(
                    cast(list[str], collections["retained_route_sha256s"])
                ),
                added_route_sha256s=tuple(
                    cast(list[str], collections["added_route_sha256s"])
                ),
                changed_authority_names=tuple(
                    cast(list[str], collections["changed_authority_names"])
                ),
            )
        except (TypeError, ValueError) as exc:
            raise LanguageBridgeIntegrityError(
                "route frontier transition reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise LanguageBridgeIntegrityError(
                "route frontier transition failed canonical reconstruction"
            )
        return result


def prove_route_frontier_transition(
    prior_frontier: ActionFrontier,
    next_frontier: ActionFrontier,
    graph_history: Sequence[ComputeOperatorGraphState],
) -> tuple[RouteFrontierTransitionReceipt, dict[str, str]]:
    """Prove append-only graph growth for retained materialized-route actions."""

    if not isinstance(prior_frontier, ActionFrontier) or not isinstance(
        next_frontier, ActionFrontier
    ):
        raise TypeError("frontiers must be ActionFrontier values")
    states = tuple(graph_history)
    if len(states) < 2 or any(
        not isinstance(state, ComputeOperatorGraphState) for state in states
    ):
        raise ValueError("graph_history needs at least two graph states")
    prior_authorities = dict(prior_frontier.authority_hashes)
    next_authorities = dict(next_frontier.authority_hashes)
    if (
        prior_authorities.get("graph-state") != states[0].sha256
        or next_authorities.get("graph-state") != states[-1].sha256
    ):
        raise LanguageBridgeIntegrityError(
            "frontier graph authorities do not bracket the supplied history"
        )
    for previous, current in zip(states, states[1:], strict=False):
        try:
            ComputeOperatorGraph._validate_extension(previous, current)
        except Exception as exc:
            raise LanguageBridgeIntegrityError(
                "graph history is not an append-only revision chain"
            ) from exc
    prior_routes = {
        item.artifact_sha256
        for item in prior_frontier.actions
        if item.artifact_kind == "materialized-route"
    }
    next_routes = {
        item.artifact_sha256
        for item in next_frontier.actions
        if item.artifact_kind == "materialized-route"
    }
    if not prior_routes <= next_routes:
        raise LanguageBridgeIntegrityError(
            "next route frontier removed an existing route action"
        )
    prior_bindings = {item.action_id: item for item in prior_frontier.actions}
    next_bindings = {item.action_id: item for item in next_frontier.actions}
    if any(
        next_bindings.get(action) != binding
        for action, binding in prior_bindings.items()
    ):
        raise LanguageBridgeIntegrityError(
            "next route frontier rebound an existing action"
        )
    changed_names = tuple(
        sorted(
            name
            for name in set(prior_authorities) | set(next_authorities)
            if prior_authorities.get(name) != next_authorities.get(name)
        )
    )
    unsupported = set(changed_names) - {"graph-state", "route-contract-set"}
    if unsupported:
        raise LanguageBridgeIntegrityError(
            "route transition changed a non-graph authority"
        )
    receipt = RouteFrontierTransitionReceipt(
        prior_frontier_sha256=prior_frontier.sha256,
        next_frontier_sha256=next_frontier.sha256,
        prior_graph_state_sha256=states[0].sha256,
        next_graph_state_sha256=states[-1].sha256,
        graph_history_sha256s=tuple(state.sha256 for state in states),
        graph_state_payloads=tuple(state.to_bytes() for state in states),
        retained_route_sha256s=tuple(sorted(prior_routes)),
        added_route_sha256s=tuple(sorted(next_routes - prior_routes)),
        changed_authority_names=changed_names,
    )
    return receipt, {name: receipt.sha256 for name in changed_names}


def build_algebra_action_frontier(
    router_state: AlgebraRouterState,
    *,
    context_schema_sha256: str,
    extra_authorities: Mapping[str, str] | None = None,
) -> ActionFrontier:
    """Build a language frontier over the active algebra candidate catalog."""

    if not isinstance(router_state, AlgebraRouterState):
        raise TypeError("router_state must be an AlgebraRouterState")
    candidates = tuple(router_state.candidates)
    if not candidates:
        raise ValueError("algebra frontier must not be empty")
    schema_sha256 = _sha256(
        {
            "candidates": [
                {
                    "candidate_sha256": candidate.sha256,
                    "family": candidate.family,
                    "program_sha256": candidate.program_sha256,
                    "schema_sha256": candidate.schema_contract_sha256,
                    "verifier_sha256": candidate.verifier_sha256,
                }
                for candidate in candidates
            ],
            "format": "immer-ooe-language-algebra-action-schema/v1",
            "router_state_sha256": router_state.sha256,
        }
    )
    authorities = _merge_authorities(
        {
            "algebra-archive": router_state.archive_sha256,
            "algebra-router-state": router_state.sha256,
            "language-reward-policy": ALGEBRA_LANGUAGE_REWARD_POLICY_SHA256,
        },
        extra_authorities,
    )
    return ActionFrontier.create(
        tuple(
            ActionBinding(
                candidate_action_id(candidate.sha256),
                "program",
                candidate.program_sha256,
            )
            for candidate in candidates
        ),
        action_schema_sha256=schema_sha256,
        context_schema_sha256=context_schema_sha256,
        authority_hashes=authorities,
    )


def _emission_from_record(value: object) -> SenderEmission:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "format",
            "frontier_sha256",
            "intent_action_id",
            "sequence",
            "word_id",
        }
        or value.get("format") != SenderEmission.FORMAT
    ):
        raise LanguageBridgeIntegrityError("sender emission is invalid")
    try:
        return SenderEmission(
            frontier_sha256=cast(str, value.get("frontier_sha256")),
            intent_action_id=cast(str, value.get("intent_action_id")),
            word_id=cast(str, value.get("word_id")),
            sequence=cast(int, value.get("sequence")),
        )
    except (TypeError, ValueError) as exc:
        raise LanguageBridgeIntegrityError("sender emission is invalid") from exc


def _sender_from_record(value: object) -> SenderDecision:
    if (
        not isinstance(value, Mapping)
        or set(value)
        != {
            "emission",
            "format",
            "receiver_decision_sha256",
        }
        or value.get("format") != SenderDecision.FORMAT
    ):
        raise LanguageBridgeIntegrityError("sender decision is invalid")
    try:
        return SenderDecision(
            emission=_emission_from_record(value.get("emission")),
            receiver_decision_sha256=cast(str, value.get("receiver_decision_sha256")),
        )
    except (TypeError, ValueError) as exc:
        raise LanguageBridgeIntegrityError("sender decision is invalid") from exc


@dataclass(frozen=True, slots=True)
class LanguageRoutingDecision:
    language_state_before_sha256: str
    frontier_sha256: str
    context_id: str
    emission: SenderEmission
    receiver: ReceiverDecision
    sender: SenderDecision
    selected_artifact_kind: str
    selected_artifact_sha256: str

    FORMAT = LANGUAGE_ROUTING_DECISION_SCHEMA

    def __post_init__(self) -> None:
        for field_name in ("language_state_before_sha256", "frontier_sha256"):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        object.__setattr__(
            self, "context_id", _identifier(self.context_id, field="context_id")
        )
        if not isinstance(self.emission, SenderEmission):
            raise TypeError("emission must be a SenderEmission")
        if not isinstance(self.receiver, ReceiverDecision):
            raise TypeError("receiver must be a ReceiverDecision")
        if not isinstance(self.sender, SenderDecision):
            raise TypeError("sender must be a SenderDecision")
        object.__setattr__(
            self,
            "selected_artifact_kind",
            _identifier(self.selected_artifact_kind, field="selected_artifact_kind"),
        )
        object.__setattr__(
            self,
            "selected_artifact_sha256",
            require_sha256(
                self.selected_artifact_sha256,
                field="selected_artifact_sha256",
            ),
        )
        if (
            self.emission.frontier_sha256 != self.frontier_sha256
            or self.receiver.frontier_sha256 != self.frontier_sha256
            or self.emission.word_id != self.receiver.word_id
            or self.receiver.context_id != self.context_id
            or self.sender.emission != self.emission
            or self.sender.receiver_decision_sha256 != self.receiver.sha256
        ):
            raise LanguageBridgeIntegrityError(
                "language routing episode bindings disagree"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "context_id": self.context_id,
            "emission": self.emission.to_record(),
            "format": self.FORMAT,
            "frontier_sha256": self.frontier_sha256,
            "language_state_before_sha256": self.language_state_before_sha256,
            "receiver": self.receiver.to_record(),
            "selected_artifact_kind": self.selected_artifact_kind,
            "selected_artifact_sha256": self.selected_artifact_sha256,
            "sender": self.sender.to_record(),
        }

    def to_bytes(self) -> bytes:
        return _seal(self.FORMAT, self.to_record())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "LanguageRoutingDecision":
        body = _open(data, schema=cls.FORMAT, label="language routing decision")
        expected = {
            "context_id",
            "emission",
            "format",
            "frontier_sha256",
            "language_state_before_sha256",
            "receiver",
            "selected_artifact_kind",
            "selected_artifact_sha256",
            "sender",
        }
        if set(body) != expected or body.get("format") != cls.FORMAT:
            raise LanguageBridgeIntegrityError("routing decision body is invalid")
        try:
            result = cls(
                language_state_before_sha256=cast(
                    str, body.get("language_state_before_sha256")
                ),
                frontier_sha256=cast(str, body.get("frontier_sha256")),
                context_id=cast(str, body.get("context_id")),
                emission=_emission_from_record(body.get("emission")),
                receiver=ReceiverDecision.from_record(body.get("receiver")),
                sender=_sender_from_record(body.get("sender")),
                selected_artifact_kind=cast(str, body.get("selected_artifact_kind")),
                selected_artifact_sha256=cast(
                    str, body.get("selected_artifact_sha256")
                ),
            )
        except (TypeError, ValueError) as exc:
            raise LanguageBridgeIntegrityError(
                "routing decision reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise LanguageBridgeIntegrityError(
                "routing decision failed canonical reconstruction"
            )
        return result


@dataclass(frozen=True, slots=True)
class LanguageOutcomeCommitReceipt:
    routing_decision_sha256: str
    frontier_sha256: str
    external_outcome_kind: str
    external_outcome_sha256: str
    feedback_sha256: str
    language_state_before_sha256: str
    language_state_after_sha256: str
    action_id: str
    word_id: str
    context_id: str
    accepted: bool
    reward: float

    FORMAT = LANGUAGE_OUTCOME_COMMIT_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "routing_decision_sha256",
            "frontier_sha256",
            "external_outcome_sha256",
            "feedback_sha256",
            "language_state_before_sha256",
            "language_state_after_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        for field_name in (
            "external_outcome_kind",
            "action_id",
            "word_id",
            "context_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), field=field_name),
            )
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be bool")
        reward = float(self.reward)
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        object.__setattr__(self, "reward", reward)
        if self.language_state_before_sha256 == self.language_state_after_sha256:
            raise ValueError("committed language outcome must change learner state")

    def to_record(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "action_id": self.action_id,
            "context_id": self.context_id,
            "external_outcome_kind": self.external_outcome_kind,
            "external_outcome_sha256": self.external_outcome_sha256,
            "feedback_sha256": self.feedback_sha256,
            "format": self.FORMAT,
            "frontier_sha256": self.frontier_sha256,
            "language_state_after_sha256": self.language_state_after_sha256,
            "language_state_before_sha256": self.language_state_before_sha256,
            "reward_hex": self.reward.hex(),
            "routing_decision_sha256": self.routing_decision_sha256,
            "word_id": self.word_id,
        }

    def to_bytes(self) -> bytes:
        return _seal(self.FORMAT, self.to_record())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "LanguageOutcomeCommitReceipt":
        body = _open(data, schema=cls.FORMAT, label="language outcome commit")
        expected = {
            "accepted",
            "action_id",
            "context_id",
            "external_outcome_kind",
            "external_outcome_sha256",
            "feedback_sha256",
            "format",
            "frontier_sha256",
            "language_state_after_sha256",
            "language_state_before_sha256",
            "reward_hex",
            "routing_decision_sha256",
            "word_id",
        }
        if set(body) != expected or body.get("format") != cls.FORMAT:
            raise LanguageBridgeIntegrityError("outcome commit body is invalid")
        reward_hex = body.get("reward_hex")
        if not isinstance(reward_hex, str):
            raise LanguageBridgeIntegrityError("outcome reward is invalid")
        try:
            reward = float.fromhex(reward_hex)
            if reward.hex() != reward_hex:
                raise ValueError("non-canonical reward")
            result = cls(
                routing_decision_sha256=cast(str, body.get("routing_decision_sha256")),
                frontier_sha256=cast(str, body.get("frontier_sha256")),
                external_outcome_kind=cast(str, body.get("external_outcome_kind")),
                external_outcome_sha256=cast(str, body.get("external_outcome_sha256")),
                feedback_sha256=cast(str, body.get("feedback_sha256")),
                language_state_before_sha256=cast(
                    str, body.get("language_state_before_sha256")
                ),
                language_state_after_sha256=cast(
                    str, body.get("language_state_after_sha256")
                ),
                action_id=cast(str, body.get("action_id")),
                word_id=cast(str, body.get("word_id")),
                context_id=cast(str, body.get("context_id")),
                accepted=cast(bool, body.get("accepted")),
                reward=reward,
            )
        except (TypeError, ValueError) as exc:
            raise LanguageBridgeIntegrityError(
                "outcome commit reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise LanguageBridgeIntegrityError(
                "outcome commit failed canonical reconstruction"
            )
        return result


@dataclass(frozen=True, slots=True)
class ControllerCrystalOutcomeReceipt:
    routing_decision_sha256: str
    frontier_sha256: str
    authorized_bank_anchor_sha256: str
    intended_action_id: str
    selected_action_id: str
    intended_crystal_sha256: str
    selected_crystal_sha256: str
    input_values_hex: tuple[str, ...]
    input_sha256: str
    intended_output_sha256: str
    selected_output_sha256: str
    accepted: bool
    verifier_sha256: str = CONTROLLER_CRYSTAL_OUTCOME_VERIFIER_SHA256

    FORMAT = CONTROLLER_CRYSTAL_OUTCOME_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "routing_decision_sha256",
            "frontier_sha256",
            "authorized_bank_anchor_sha256",
            "intended_crystal_sha256",
            "selected_crystal_sha256",
            "input_sha256",
            "intended_output_sha256",
            "selected_output_sha256",
            "verifier_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        for field_name in ("intended_action_id", "selected_action_id"):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), field=field_name),
            )
        values = tuple(self.input_values_hex)
        if not 1 <= len(values) <= 4_096:
            raise ValueError("controller-Crystal input vector length is invalid")
        for value in values:
            if not isinstance(value, str):
                raise TypeError("controller-Crystal input values must be hex strings")
            decoded = float.fromhex(value)
            if not math.isfinite(decoded) or decoded.hex() != value:
                raise ValueError("controller-Crystal input value is not canonical")
        object.__setattr__(self, "input_values_hex", values)
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be bool")
        if self.verifier_sha256 != CONTROLLER_CRYSTAL_OUTCOME_VERIFIER_SHA256:
            raise ValueError("controller-Crystal outcome uses another verifier")
        expected_acceptance = (
            self.intended_action_id == self.selected_action_id
            and self.intended_crystal_sha256 == self.selected_crystal_sha256
            and self.intended_output_sha256 == self.selected_output_sha256
        )
        if self.accepted != expected_acceptance:
            raise ValueError("controller-Crystal acceptance differs from exact parity")

    @property
    def input_array(self) -> np.ndarray:
        return np.asarray(
            [float.fromhex(value) for value in self.input_values_hex],
            dtype=np.float64,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "authorized_bank_anchor_sha256": self.authorized_bank_anchor_sha256,
            "format": self.FORMAT,
            "frontier_sha256": self.frontier_sha256,
            "input_sha256": self.input_sha256,
            "input_values_hex": list(self.input_values_hex),
            "intended_action_id": self.intended_action_id,
            "intended_crystal_sha256": self.intended_crystal_sha256,
            "intended_output_sha256": self.intended_output_sha256,
            "routing_decision_sha256": self.routing_decision_sha256,
            "selected_action_id": self.selected_action_id,
            "selected_crystal_sha256": self.selected_crystal_sha256,
            "selected_output_sha256": self.selected_output_sha256,
            "verifier_sha256": self.verifier_sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(self.FORMAT, self.to_record())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "ControllerCrystalOutcomeReceipt":
        body = _open(data, schema=cls.FORMAT, label="controller-Crystal outcome")
        expected = {
            "accepted",
            "authorized_bank_anchor_sha256",
            "format",
            "frontier_sha256",
            "input_sha256",
            "input_values_hex",
            "intended_action_id",
            "intended_crystal_sha256",
            "intended_output_sha256",
            "routing_decision_sha256",
            "selected_action_id",
            "selected_crystal_sha256",
            "selected_output_sha256",
            "verifier_sha256",
        }
        values = body.get("input_values_hex")
        if (
            set(body) != expected
            or body.get("format") != cls.FORMAT
            or not isinstance(values, list)
        ):
            raise LanguageBridgeIntegrityError(
                "controller-Crystal outcome body is invalid"
            )
        try:
            result = cls(
                routing_decision_sha256=cast(str, body.get("routing_decision_sha256")),
                frontier_sha256=cast(str, body.get("frontier_sha256")),
                authorized_bank_anchor_sha256=cast(
                    str, body.get("authorized_bank_anchor_sha256")
                ),
                intended_action_id=cast(str, body.get("intended_action_id")),
                selected_action_id=cast(str, body.get("selected_action_id")),
                intended_crystal_sha256=cast(str, body.get("intended_crystal_sha256")),
                selected_crystal_sha256=cast(str, body.get("selected_crystal_sha256")),
                input_values_hex=tuple(values),
                input_sha256=cast(str, body.get("input_sha256")),
                intended_output_sha256=cast(str, body.get("intended_output_sha256")),
                selected_output_sha256=cast(str, body.get("selected_output_sha256")),
                accepted=cast(bool, body.get("accepted")),
                verifier_sha256=cast(str, body.get("verifier_sha256")),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise LanguageBridgeIntegrityError(
                "controller-Crystal outcome reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise LanguageBridgeIntegrityError(
                "controller-Crystal outcome failed canonical reconstruction"
            )
        return result


class ControllerCrystalLanguageExecutor:
    """One verified export mounted once for repeated consequence episodes."""

    def __init__(
        self,
        export_receipt: ControllerCrystalExportReceipt,
        compute_bank: ComputeCrystalBank,
    ) -> None:
        if not isinstance(export_receipt, ControllerCrystalExportReceipt):
            raise TypeError("export_receipt must be a ControllerCrystalExportReceipt")
        if not isinstance(compute_bank, ComputeCrystalBank):
            raise TypeError("compute_bank must be a ComputeCrystalBank")
        try:
            crystals = export_receipt.restore_crystals(compute_bank)
        except Exception as exc:
            raise LanguageBridgeIntegrityError(
                "controller-Crystal export provenance failed restore"
            ) from exc
        self.export_receipt = export_receipt
        self.compute_bank = compute_bank
        self._crystals = crystals
        self._issued: dict[str, ControllerCrystalOutcomeReceipt] = {}

    def execute(
        self,
        decision: LanguageRoutingDecision,
        input_value: object,
    ) -> ControllerCrystalOutcomeReceipt:
        outcome = _execute_controller_crystal_choice(
            decision,
            self.export_receipt,
            self.compute_bank,
            input_value,
            self._crystals,
        )
        if outcome.sha256 in self._issued:
            raise LanguageBridgeConflictError(
                "controller-Crystal outcome is already pending"
            )
        if len(self._issued) >= 100_000:
            raise LanguageBridgeError(
                "controller-Crystal pending outcome inventory is full"
            )
        self._issued[outcome.sha256] = outcome
        return outcome

    def verify_and_consume(
        self,
        decision: LanguageRoutingDecision,
        outcome: ControllerCrystalOutcomeReceipt,
    ) -> None:
        if not isinstance(decision, LanguageRoutingDecision):
            raise TypeError("decision must be a LanguageRoutingDecision")
        if not isinstance(outcome, ControllerCrystalOutcomeReceipt):
            raise TypeError("outcome must be a ControllerCrystalOutcomeReceipt")
        if (
            outcome.routing_decision_sha256 != decision.sha256
            or self._issued.get(outcome.sha256) != outcome
        ):
            raise LanguageBridgeIntegrityError(
                "controller-Crystal outcome was not issued by this executor"
            )
        try:
            self.compute_bank.assert_descends_from(
                outcome.authorized_bank_anchor_sha256
            )
        except ComputeCrystalError as exc:
            raise LanguageBridgeIntegrityError(
                "controller-Crystal outcome bank anchor is not an ancestor"
            ) from exc
        if outcome.authorized_bank_anchor_sha256 != (
            self.export_receipt.final_bank_anchor_sha256
        ):
            raise LanguageBridgeIntegrityError(
                "controller-Crystal outcome names another authorized bank anchor"
            )
        del self._issued[outcome.sha256]


def execute_controller_crystal_choice(
    decision: LanguageRoutingDecision,
    export_receipt: ControllerCrystalExportReceipt,
    compute_bank: ComputeCrystalBank,
    input_value: object,
) -> ControllerCrystalOutcomeReceipt:
    """Replay-safe one-shot execution over an authenticated controller export."""

    return ControllerCrystalLanguageExecutor(export_receipt, compute_bank).execute(
        decision, input_value
    )


def _execute_controller_crystal_choice(
    decision: LanguageRoutingDecision,
    export_receipt: ControllerCrystalExportReceipt,
    compute_bank: ComputeCrystalBank,
    input_value: object,
    exported_crystals: Sequence[object],
) -> ControllerCrystalOutcomeReceipt:
    """Execute intended and receiver-selected live controller policies exactly."""

    if not isinstance(decision, LanguageRoutingDecision):
        raise TypeError("decision must be a LanguageRoutingDecision")
    if not isinstance(export_receipt, ControllerCrystalExportReceipt):
        raise TypeError("export_receipt must be a ControllerCrystalExportReceipt")
    if not isinstance(compute_bank, ComputeCrystalBank):
        raise TypeError("compute_bank must be a ComputeCrystalBank")
    frontier = export_receipt.frontier
    if decision.frontier_sha256 != frontier.sha256:
        raise LanguageBridgeIntegrityError("routing decision uses another frontier")
    if type(input_value) is not np.ndarray:
        raise TypeError("controller-Crystal input must be an exact numpy.ndarray")
    value = cast(np.ndarray, input_value)
    if value.dtype != np.dtype(np.float64) or value.ndim != 1:
        raise ValueError("controller-Crystal input must be one float64 vector")
    if not np.all(np.isfinite(value)):
        raise ValueError("controller-Crystal input contains non-finite values")

    authorities = dict(frontier.authority_hashes)
    required_authorities = {
        "controller-atlas-graph",
        "controller-calibration",
        "controller-crystal-manifest",
        "controller-model-pin",
        "controller-state",
        "controller-weight-graph",
        "compute-bank-anchor",
        "language-reward-policy",
    }
    if not required_authorities <= set(authorities):
        raise LanguageBridgeIntegrityError(
            "controller-Crystal frontier lacks exact authorities"
        )
    frontier_bank_anchor = authorities["compute-bank-anchor"]
    if frontier_bank_anchor != export_receipt.final_bank_anchor_sha256:
        raise LanguageBridgeIntegrityError(
            "controller-Crystal frontier differs from its export receipt"
        )
    crystals = tuple(exported_crystals)
    if len(crystals) != len(export_receipt.sites) or tuple(
        getattr(crystal, "sha256", None) for crystal in crystals
    ) != tuple(site.compute_crystal_sha256 for site in export_receipt.sites):
        raise LanguageBridgeIntegrityError(
            "controller-Crystal runtime artifacts differ from their export receipt"
        )
    if authorities["language-reward-policy"] != (
        CONTROLLER_CRYSTAL_LANGUAGE_REWARD_POLICY_SHA256
    ):
        raise LanguageBridgeIntegrityError(
            "controller-Crystal reward authority changed"
        )

    intended_action = decision.emission.intent_action_id
    selected_action = decision.receiver.action_id
    try:
        intended_binding = frontier.binding(intended_action)
        selected_binding = frontier.binding(selected_action)
    except KeyError as exc:
        raise LanguageBridgeIntegrityError(
            "controller-Crystal decision names an unknown action"
        ) from exc
    if (
        intended_binding.artifact_kind != "crystal"
        or selected_binding.artifact_kind != "crystal"
        or decision.selected_artifact_kind != selected_binding.artifact_kind
        or decision.selected_artifact_sha256 != selected_binding.artifact_sha256
    ):
        raise LanguageBridgeIntegrityError(
            "controller-Crystal decision artifact binding changed"
        )
    crystals_by_action = {
        site.action_id: crystal
        for site, crystal in zip(export_receipt.sites, crystals, strict=True)
    }
    try:
        intended = crystals_by_action[intended_action]
        selected = crystals_by_action[selected_action]
    except KeyError as exc:
        raise LanguageBridgeIntegrityError(
            "controller-Crystal action is absent from its export receipt"
        ) from exc
    if (
        intended.sha256 != intended_binding.artifact_sha256
        or selected.sha256 != selected_binding.artifact_sha256
    ):
        raise LanguageBridgeIntegrityError(
            "controller-Crystal binding differs from its exported artifact"
        )
    if (
        intended.input_abi != selected.input_abi
        or intended.output_abi != selected.output_abi
        or intended.input_abi != intended.output_abi
    ):
        raise LanguageBridgeIntegrityError(
            "controller-Crystal policies have incompatible numerical ABIs"
        )
    canonical_input = intended.input_abi.validate(
        value, field="controller-Crystal input"
    )
    selected.input_abi.validate(canonical_input, field="controller-Crystal input")
    intended_output = intended.apply(canonical_input)
    selected_output = selected.apply(canonical_input)
    input_sha = tensor_sha256(canonical_input, intended.input_abi)
    intended_output_sha = tensor_sha256(intended_output, intended.output_abi)
    selected_output_sha = tensor_sha256(selected_output, selected.output_abi)
    return ControllerCrystalOutcomeReceipt(
        routing_decision_sha256=decision.sha256,
        frontier_sha256=frontier.sha256,
        authorized_bank_anchor_sha256=frontier_bank_anchor,
        intended_action_id=intended_action,
        selected_action_id=selected_action,
        intended_crystal_sha256=intended.sha256,
        selected_crystal_sha256=selected.sha256,
        input_values_hex=tuple(float(item).hex() for item in canonical_input),
        input_sha256=input_sha,
        intended_output_sha256=intended_output_sha,
        selected_output_sha256=selected_output_sha,
        accepted=(
            intended_action == selected_action
            and intended.sha256 == selected.sha256
            and intended_output_sha == selected_output_sha
        ),
    )


class ConsequenceLanguageStateBank:
    """Atomic language-state pointer with caller-persisted rollback anchor."""

    def __init__(
        self,
        store: CrystalStore | str | os.PathLike[str],
        *,
        state_name: str = LANGUAGE_STATE_NAME,
    ) -> None:
        self.store = store if isinstance(store, CrystalStore) else CrystalStore(store)
        self.state_name = _identifier(state_name, field="state_name")
        self.root = Path(self.store.root)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        path = self.root / ".production-language.lock"
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise LanguageBridgeIntegrityError(
                    "language bank lock is not a regular file"
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            linked = path.lstat()
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
                raise LanguageBridgeIntegrityError(
                    "language bank lock changed while acquiring it"
                )
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _restore_optional(self) -> bytes | None:
        try:
            return self.store.restore_state(self.state_name)
        except KeyError:
            return None
        except CrystalStoreError as exc:
            raise LanguageBridgeIntegrityError(
                "language bank state failed storage integrity"
            ) from exc

    def initialize(self, language: ConsequenceMarkovLanguage) -> str:
        if not isinstance(language, ConsequenceMarkovLanguage):
            raise TypeError("language must be a ConsequenceMarkovLanguage")
        payload = language.to_bytes()
        with self._locked():
            current = self._restore_optional()
            if current is not None:
                if current != payload:
                    raise LanguageBridgeConflictError(
                        "language bank already contains another state"
                    )
                return hashlib.sha256(current).hexdigest()
            try:
                publication = self.store.publish_state(self.state_name, payload)
            except CrystalStoreError as exc:
                raise LanguageBridgeIntegrityError(
                    "language bank initialization failed"
                ) from exc
            return publication.payload_sha256

    def publish(
        self,
        language: ConsequenceMarkovLanguage,
        *,
        expected_state_sha256: str,
    ) -> str:
        if not isinstance(language, ConsequenceMarkovLanguage):
            raise TypeError("language must be a ConsequenceMarkovLanguage")
        expected = require_sha256(expected_state_sha256, field="expected_state_sha256")
        with self._locked():
            return self._publish_unlocked(language, expected)

    def _publish_unlocked(
        self,
        language: ConsequenceMarkovLanguage,
        expected_state_sha256: str,
    ) -> str:
        try:
            publication = self.store.publish_state(
                self.state_name,
                language.to_bytes(),
                expected_sha256=expected_state_sha256,
            )
        except ManifestConflictError as exc:
            raise LanguageBridgeConflictError("language bank CAS conflicted") from exc
        except CrystalStoreError as exc:
            raise LanguageBridgeIntegrityError(
                "language bank publication failed"
            ) from exc
        return publication.payload_sha256

    def revision_state_name(self, language_state_sha256: str) -> str:
        return (
            self.state_name
            + ":promotion:"
            + require_sha256(language_state_sha256, field="language_state_sha256")
        )

    def publish_with_revision(
        self,
        language: ConsequenceMarkovLanguage,
        revision: "LanguageRevisionPromotionReceipt",
        *,
        expected_state_sha256: str,
    ) -> str:
        if not isinstance(language, ConsequenceMarkovLanguage):
            raise TypeError("language must be a ConsequenceMarkovLanguage")
        if not isinstance(revision, LanguageRevisionPromotionReceipt):
            raise TypeError("revision must be a LanguageRevisionPromotionReceipt")
        expected = require_sha256(expected_state_sha256, field="expected_state_sha256")
        if (
            revision.prior_language_state_sha256 != expected
            or revision.next_language_state_sha256 != language.sha256
            or revision.next_frontier_sha256 != language.frontier.sha256
        ):
            raise LanguageBridgeIntegrityError(
                "revision metadata differs from the promoted language"
            )
        name = self.revision_state_name(language.sha256)
        payload = revision.to_bytes()
        with self._locked():
            try:
                existing = self.store.restore_state(name)
            except KeyError:
                existing = None
            except CrystalStoreError as exc:
                raise LanguageBridgeIntegrityError(
                    "language revision metadata failed storage integrity"
                ) from exc
            if existing is None:
                try:
                    self.store.publish_state(name, payload)
                except CrystalStoreError as exc:
                    raise LanguageBridgeIntegrityError(
                        "language revision metadata publication failed"
                    ) from exc
            elif existing != payload:
                raise LanguageBridgeIntegrityError(
                    "language revision address contains different metadata"
                )
            return self._publish_unlocked(language, expected)

    def restore_revision(
        self, language_state_sha256: str
    ) -> "LanguageRevisionPromotionReceipt":
        state_sha = require_sha256(language_state_sha256, field="language_state_sha256")
        try:
            payload = self.store.restore_state(self.revision_state_name(state_sha))
        except KeyError as exc:
            raise KeyError("language revision metadata is absent") from exc
        except CrystalStoreError as exc:
            raise LanguageBridgeIntegrityError(
                "language revision metadata failed storage integrity"
            ) from exc
        revision = LanguageRevisionPromotionReceipt.from_bytes(payload)
        if revision.next_language_state_sha256 != state_sha:
            raise LanguageBridgeIntegrityError(
                "language revision metadata is stored under another state"
            )
        return revision

    def restore(
        self,
        *,
        expected_frontier: ActionFrontier | None = None,
        trusted_state_sha256: str | None = None,
    ) -> ConsequenceMarkovLanguage:
        payload = self._restore_optional()
        if payload is None:
            raise KeyError("production language state is absent")
        digest = hashlib.sha256(payload).hexdigest()
        if trusted_state_sha256 is not None and digest != require_sha256(
            trusted_state_sha256, field="trusted_state_sha256"
        ):
            raise LanguageBridgeIntegrityError(
                "language state does not equal its external rollback anchor"
            )
        try:
            return ConsequenceMarkovLanguage.from_bytes(
                payload, expected_frontier=expected_frontier
            )
        except MarkovLanguageConflictError as exc:
            raise LanguageBridgeConflictError(
                "stored language belongs to another frontier"
            ) from exc

    def current_state_sha256(self) -> str:
        payload = self._restore_optional()
        if payload is None:
            raise KeyError("production language state is absent")
        return hashlib.sha256(payload).hexdigest()


class ConsequenceLanguageBridge:
    """Join receiver choices to fully verified production outcome receipts."""

    def __init__(
        self,
        language: ConsequenceMarkovLanguage,
        *,
        state_bank: ConsequenceLanguageStateBank | None = None,
    ) -> None:
        if not isinstance(language, ConsequenceMarkovLanguage):
            raise TypeError("language must be a ConsequenceMarkovLanguage")
        if state_bank is not None and not isinstance(
            state_bank, ConsequenceLanguageStateBank
        ):
            raise TypeError("state_bank must be a ConsequenceLanguageStateBank")
        self.language = language
        self.state_bank = state_bank
        self._before_payloads: dict[str, bytes] = {}
        if state_bank is not None:
            state_bank.initialize(language)

    def begin(
        self,
        intent_action_id: str,
        context_id: str,
        *,
        epsilon: float,
    ) -> LanguageRoutingDecision:
        before_payload = self.language.to_bytes()
        before_sha256 = hashlib.sha256(before_payload).hexdigest()
        try:
            emission = self.language.emit(intent_action_id, epsilon=epsilon)
            receiver = self.language.receiver_decide(
                emission.word_id, context_id, epsilon=epsilon
            )
            sender = self.language.bind_sender(emission, receiver)
            binding = self.language.frontier.binding(receiver.action_id)
            decision = LanguageRoutingDecision(
                language_state_before_sha256=before_sha256,
                frontier_sha256=self.language.frontier.sha256,
                context_id=receiver.context_id,
                emission=emission,
                receiver=receiver,
                sender=sender,
                selected_artifact_kind=binding.artifact_kind,
                selected_artifact_sha256=binding.artifact_sha256,
            )
        except Exception:
            self.language = ConsequenceMarkovLanguage.from_bytes(
                before_payload,
                expected_frontier=self.language.frontier,
            )
            raise
        self._before_payloads[decision.sha256] = before_payload
        return decision

    def _rollback(self, decision: LanguageRoutingDecision) -> None:
        payload = self._before_payloads.pop(decision.sha256, None)
        if payload is None:
            raise LanguageBridgeConflictError(
                "routing decision is unknown or already settled"
            )
        self.language = ConsequenceMarkovLanguage.from_bytes(
            payload, expected_frontier=self.language.frontier
        )

    def abort(self, decision: LanguageRoutingDecision) -> str:
        if not isinstance(decision, LanguageRoutingDecision):
            raise TypeError("decision must be a LanguageRoutingDecision")
        before = self._before_payloads.get(decision.sha256)
        if before is None:
            raise LanguageBridgeConflictError(
                "routing decision is unknown or already settled"
            )
        self.language.abort_bound_episode(decision.sender, decision.receiver)
        try:
            after_sha = self.language.sha256
            if self.state_bank is not None:
                self.state_bank.publish(
                    self.language,
                    expected_state_sha256=decision.language_state_before_sha256,
                )
        except Exception:
            self._rollback(decision)
            raise
        del self._before_payloads[decision.sha256]
        return after_sha

    def _commit_feedback(
        self,
        decision: LanguageRoutingDecision,
        feedback: ConsequenceFeedback,
        *,
        outcome_kind: str,
        outcome_sha256: str,
        learning_rate: float,
    ) -> LanguageOutcomeCommitReceipt:
        before_payload = self._before_payloads.get(decision.sha256)
        if before_payload is None:
            raise LanguageBridgeConflictError(
                "routing decision is unknown or already settled"
            )
        if (
            decision.frontier_sha256 != self.language.frontier.sha256
            or feedback.frontier_sha256 != self.language.frontier.sha256
            or feedback.receiver_decision_sha256 != decision.receiver.sha256
        ):
            raise LanguageBridgeIntegrityError(
                "feedback names another language decision or frontier"
            )
        try:
            self.language.observe_receiver(
                decision.receiver, feedback, learning_rate=learning_rate
            )
            self.language.observe_sender(
                decision.sender, feedback, learning_rate=learning_rate
            )
            prior_language = ConsequenceMarkovLanguage.from_bytes(
                before_payload,
                expected_frontier=self.language.frontier,
            )
            self.language._transition_head_sha256 = _sha256(
                {
                    "external_outcome_kind": outcome_kind,
                    "external_outcome_sha256": outcome_sha256,
                    "feedback_sha256": feedback.sha256,
                    "format": "immer-ooe-production-language-transition/v1",
                    "previous_sha256": prior_language._transition_head_sha256,
                    "routing_decision_sha256": decision.sha256,
                }
            )
            after_sha256 = self.language.sha256
            if self.state_bank is not None:
                published = self.state_bank.publish(
                    self.language,
                    expected_state_sha256=decision.language_state_before_sha256,
                )
                if published != after_sha256:
                    raise LanguageBridgeIntegrityError(
                        "published language state hash changed"
                    )
        except Exception:
            self._rollback(decision)
            raise
        del self._before_payloads[decision.sha256]
        return LanguageOutcomeCommitReceipt(
            routing_decision_sha256=decision.sha256,
            frontier_sha256=decision.frontier_sha256,
            external_outcome_kind=outcome_kind,
            external_outcome_sha256=outcome_sha256,
            feedback_sha256=feedback.sha256,
            language_state_before_sha256=decision.language_state_before_sha256,
            language_state_after_sha256=after_sha256,
            action_id=decision.receiver.action_id,
            word_id=decision.receiver.word_id,
            context_id=decision.context_id,
            accepted=feedback.accepted,
            reward=feedback.reward,
        )

    def settle_demand(
        self,
        decision: LanguageRoutingDecision,
        routed: DemandRoutedExecutionReceipt,
        graph_state: ComputeOperatorGraphState,
        *,
        learning_rate: float = 0.15,
    ) -> LanguageOutcomeCommitReceipt:
        if not isinstance(decision, LanguageRoutingDecision):
            raise TypeError("decision must be a LanguageRoutingDecision")
        if not isinstance(routed, DemandRoutedExecutionReceipt):
            raise TypeError("routed must be a DemandRoutedExecutionReceipt")
        if not isinstance(graph_state, ComputeOperatorGraphState):
            raise TypeError("graph_state must be a ComputeOperatorGraphState")
        authorities = dict(self.language.frontier.authority_hashes)
        outcome = routed.outcome
        if (
            authorities.get("graph-state") != graph_state.sha256
            or authorities.get("language-reward-policy")
            != DEMAND_LANGUAGE_REWARD_POLICY_SHA256
        ):
            raise LanguageBridgeIntegrityError(
                "language frontier has another graph or reward policy"
            )
        routes = {route.sha256: route for route in graph_state.materialized_routes}
        route = routes.get(decision.selected_artifact_sha256)
        if (
            decision.selected_artifact_kind != "materialized-route"
            or route is None
            or decision.receiver.action_id != route_action_id(route.sha256)
            or outcome.route_sha256 != route.sha256
            or outcome.graph_generation != graph_state.generation
            or outcome.graph_state_sha256 != graph_state.sha256
            or outcome.input_abi_sha256 != route.input_abi_sha256
            or outcome.output_abi_sha256 != route.output_abi_sha256
            or outcome.route_verifier_sha256s != route.verifier_sha256s
            or outcome.route_evidence_sha256s != route.evidence_sha256s
            or routed.graph_generation != graph_state.generation
            or routed.graph_state_sha256 != graph_state.sha256
            or routed.selected_prefix_route_sha256 != route.sha256
        ):
            raise LanguageBridgeIntegrityError(
                "demand outcome differs from the selected exact route"
            )
        feedback = ConsequenceFeedback.for_decision(
            decision.receiver,
            reward=outcome.reward,
            accepted=outcome.success,
            outcome_receipt_sha256=routed.sha256,
        )
        return self._commit_feedback(
            decision,
            feedback,
            outcome_kind="demand-routed-execution",
            outcome_sha256=routed.sha256,
            learning_rate=learning_rate,
        )

    def settle_algebra(
        self,
        decision: LanguageRoutingDecision,
        selection: AlgebraSelectionReceipt,
        candidate: OperatorAlgebraCandidate,
        outcome: VerifierBoundOutcome,
        *,
        learning_rate: float = 0.15,
    ) -> LanguageOutcomeCommitReceipt:
        if not isinstance(selection, AlgebraSelectionReceipt):
            raise TypeError("selection must be an AlgebraSelectionReceipt")
        if not isinstance(candidate, OperatorAlgebraCandidate):
            raise TypeError("candidate must be an OperatorAlgebraCandidate")
        if not isinstance(outcome, VerifierBoundOutcome):
            raise TypeError("outcome must be a VerifierBoundOutcome")
        authorities = dict(self.language.frontier.authority_hashes)
        if (
            authorities.get("algebra-router-state") != selection.router_parent_sha256
            or authorities.get("language-reward-policy")
            != ALGEBRA_LANGUAGE_REWARD_POLICY_SHA256
            or decision.receiver.action_id != candidate_action_id(candidate.sha256)
            or decision.selected_artifact_kind != "program"
            or decision.selected_artifact_sha256 != candidate.program_sha256
            or selection.candidate_sha256 != candidate.sha256
            or selection.program_sha256 != candidate.program_sha256
            or selection.verifier_sha256 != candidate.verifier_sha256
            or outcome.selection_sha256 != selection.sha256
            or outcome.candidate_sha256 != candidate.sha256
            or outcome.verifier_sha256 != candidate.verifier_sha256
        ):
            raise LanguageBridgeIntegrityError(
                "algebra outcome differs from the selected exact candidate"
            )
        reward = 1.0 if outcome.success else -1.0
        feedback = ConsequenceFeedback.for_decision(
            decision.receiver,
            reward=reward,
            accepted=outcome.success,
            outcome_receipt_sha256=outcome.sha256,
        )
        return self._commit_feedback(
            decision,
            feedback,
            outcome_kind="algebra-outcome",
            outcome_sha256=outcome.sha256,
            learning_rate=learning_rate,
        )

    def settle_controller_crystal(
        self,
        decision: LanguageRoutingDecision,
        outcome: ControllerCrystalOutcomeReceipt,
        export_receipt: ControllerCrystalExportReceipt,
        compute_bank: ComputeCrystalBank,
        *,
        learning_rate: float = 0.15,
        executor: ControllerCrystalLanguageExecutor | None = None,
    ) -> LanguageOutcomeCommitReceipt:
        """Commit reward only after replaying an exact promoted-policy choice."""

        if not isinstance(outcome, ControllerCrystalOutcomeReceipt):
            raise TypeError("outcome must be a ControllerCrystalOutcomeReceipt")
        if not isinstance(export_receipt, ControllerCrystalExportReceipt):
            raise TypeError("export_receipt must be a ControllerCrystalExportReceipt")
        frontier = export_receipt.frontier
        if frontier != self.language.frontier:
            raise LanguageBridgeIntegrityError(
                "controller-Crystal settlement uses another language frontier"
            )
        if executor is not None:
            if not isinstance(executor, ControllerCrystalLanguageExecutor):
                raise TypeError("executor must be a ControllerCrystalLanguageExecutor")
            if (
                executor.export_receipt != export_receipt
                or executor.compute_bank is not compute_bank
            ):
                raise LanguageBridgeIntegrityError(
                    "controller-Crystal executor uses another export or bank"
                )
            executor.verify_and_consume(decision, outcome)
        else:
            try:
                compute_bank.assert_descends_from(outcome.authorized_bank_anchor_sha256)
            except ComputeCrystalError as exc:
                raise LanguageBridgeIntegrityError(
                    "controller-Crystal outcome bank anchor is not an ancestor"
                ) from exc
            if outcome.authorized_bank_anchor_sha256 != (
                export_receipt.final_bank_anchor_sha256
            ):
                raise LanguageBridgeIntegrityError(
                    "controller-Crystal outcome names another authorized bank anchor"
                )
            expected = execute_controller_crystal_choice(
                decision, export_receipt, compute_bank, outcome.input_array
            )
            if expected != outcome:
                raise LanguageBridgeIntegrityError(
                    "controller-Crystal outcome differs from exact replay"
                )
        feedback = ConsequenceFeedback.for_decision(
            decision.receiver,
            reward=1.0 if outcome.accepted else -1.0,
            accepted=outcome.accepted,
            outcome_receipt_sha256=outcome.sha256,
        )
        return self._commit_feedback(
            decision,
            feedback,
            outcome_kind="controller-crystal-choice",
            outcome_sha256=outcome.sha256,
            learning_rate=learning_rate,
        )


@dataclass(frozen=True, slots=True)
class FrontierMigrationReceipt:
    prior_frontier_sha256: str
    next_frontier_sha256: str
    prior_language_state_sha256: str
    next_language_state_sha256: str
    retained_action_ids: tuple[str, ...]
    added_action_ids: tuple[str, ...]
    removed_action_ids: tuple[str, ...]
    reset_action_ids: tuple[str, ...]
    added_word_ids: tuple[str, ...]
    authority_transition_proofs: tuple[tuple[str, str], ...]
    authority_evidence_retained: bool
    context_evidence_retained: bool
    reward_policy_retained: bool
    seed: int

    FORMAT = FRONTIER_MIGRATION_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "prior_frontier_sha256",
            "next_frontier_sha256",
            "prior_language_state_sha256",
            "next_language_state_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        for field_name in (
            "retained_action_ids",
            "added_action_ids",
            "removed_action_ids",
            "reset_action_ids",
            "added_word_ids",
        ):
            values = tuple(
                sorted(
                    _identifier(value, field=field_name)
                    for value in getattr(self, field_name)
                )
            )
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} contains duplicates")
            object.__setattr__(self, field_name, values)
        inventories = (
            set(self.retained_action_ids),
            set(self.added_action_ids),
            set(self.removed_action_ids),
            set(self.reset_action_ids),
        )
        if any(
            left & right
            for index, left in enumerate(inventories)
            for right in inventories[index + 1 :]
        ):
            raise ValueError("migration action inventories overlap")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        object.__setattr__(
            self,
            "authority_transition_proofs",
            _hash_items(
                self.authority_transition_proofs,
                field="authority_transition_proofs",
            ),
        )
        if (
            not isinstance(self.authority_evidence_retained, bool)
            or not isinstance(self.context_evidence_retained, bool)
            or not isinstance(self.reward_policy_retained, bool)
        ):
            raise TypeError("migration retention flags must be bool")

    def to_record(self) -> dict[str, object]:
        return {
            "added_action_ids": list(self.added_action_ids),
            "added_word_ids": list(self.added_word_ids),
            "authority_evidence_retained": self.authority_evidence_retained,
            "authority_transition_proofs": dict(self.authority_transition_proofs),
            "context_evidence_retained": self.context_evidence_retained,
            "format": self.FORMAT,
            "next_frontier_sha256": self.next_frontier_sha256,
            "next_language_state_sha256": self.next_language_state_sha256,
            "prior_frontier_sha256": self.prior_frontier_sha256,
            "prior_language_state_sha256": self.prior_language_state_sha256,
            "removed_action_ids": list(self.removed_action_ids),
            "reset_action_ids": list(self.reset_action_ids),
            "reward_policy_retained": self.reward_policy_retained,
            "retained_action_ids": list(self.retained_action_ids),
            "seed": self.seed,
        }

    def to_bytes(self) -> bytes:
        return _seal(self.FORMAT, self.to_record())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "FrontierMigrationReceipt":
        body = _open(data, schema=cls.FORMAT, label="frontier migration")
        expected = {
            "added_action_ids",
            "added_word_ids",
            "authority_evidence_retained",
            "authority_transition_proofs",
            "context_evidence_retained",
            "format",
            "next_frontier_sha256",
            "next_language_state_sha256",
            "prior_frontier_sha256",
            "prior_language_state_sha256",
            "removed_action_ids",
            "reset_action_ids",
            "reward_policy_retained",
            "retained_action_ids",
            "seed",
        }
        if set(body) != expected or body.get("format") != cls.FORMAT:
            raise LanguageBridgeIntegrityError("migration body is invalid")
        inventories = {}
        for field in (
            "added_action_ids",
            "added_word_ids",
            "removed_action_ids",
            "reset_action_ids",
            "retained_action_ids",
        ):
            value = body.get(field)
            if not isinstance(value, list):
                raise LanguageBridgeIntegrityError("migration inventory is invalid")
            inventories[field] = tuple(value)
        raw_proofs = body.get("authority_transition_proofs")
        if not isinstance(raw_proofs, Mapping):
            raise LanguageBridgeIntegrityError("migration authority proofs are invalid")
        try:
            result = cls(
                prior_frontier_sha256=cast(str, body.get("prior_frontier_sha256")),
                next_frontier_sha256=cast(str, body.get("next_frontier_sha256")),
                prior_language_state_sha256=cast(
                    str, body.get("prior_language_state_sha256")
                ),
                next_language_state_sha256=cast(
                    str, body.get("next_language_state_sha256")
                ),
                retained_action_ids=inventories["retained_action_ids"],
                added_action_ids=inventories["added_action_ids"],
                removed_action_ids=inventories["removed_action_ids"],
                reset_action_ids=inventories["reset_action_ids"],
                added_word_ids=inventories["added_word_ids"],
                authority_transition_proofs=_hash_items(
                    cast(Mapping[str, str], raw_proofs),
                    field="authority_transition_proofs",
                ),
                authority_evidence_retained=cast(
                    bool, body.get("authority_evidence_retained")
                ),
                context_evidence_retained=cast(
                    bool, body.get("context_evidence_retained")
                ),
                reward_policy_retained=cast(bool, body.get("reward_policy_retained")),
                seed=cast(int, body.get("seed")),
            )
        except (TypeError, ValueError) as exc:
            raise LanguageBridgeIntegrityError(
                "migration reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise LanguageBridgeIntegrityError(
                "migration failed canonical reconstruction"
            )
        return result


def migrate_language_frontier(
    language: ConsequenceMarkovLanguage,
    next_frontier: ActionFrontier,
    *,
    seed: int,
    authority_transition_receipts: Sequence[object] = (),
    additional_word_ids: Sequence[str] = (),
) -> tuple[ConsequenceMarkovLanguage, FrontierMigrationReceipt]:
    """Carry evidence only for action bindings unchanged across revisions."""

    if not isinstance(language, ConsequenceMarkovLanguage):
        raise TypeError("language must be a ConsequenceMarkovLanguage")
    if not isinstance(next_frontier, ActionFrontier):
        raise TypeError("next_frontier must be an ActionFrontier")
    prior_state_sha256 = language.sha256
    if language.frontier.sha256 == next_frontier.sha256:
        raise ValueError("frontier migration requires a changed frontier")
    old_bindings = {item.action_id: item for item in language.frontier.actions}
    new_bindings = {item.action_id: item for item in next_frontier.actions}
    prior_authorities = dict(language.frontier.authority_hashes)
    next_authorities = dict(next_frontier.authority_hashes)
    proof_map: dict[str, str] = {}
    for transition in authority_transition_receipts:
        if isinstance(transition, RouteFrontierTransitionReceipt):
            if (
                transition.prior_frontier_sha256 != language.frontier.sha256
                or transition.next_frontier_sha256 != next_frontier.sha256
            ):
                raise LanguageBridgeIntegrityError(
                    "route transition proof names another frontier pair"
                )
            prior_route_sha256s = {
                item.artifact_sha256
                for item in language.frontier.actions
                if item.artifact_kind == "materialized-route"
            }
            next_route_sha256s = {
                item.artifact_sha256
                for item in next_frontier.actions
                if item.artifact_kind == "materialized-route"
            }
            actual_changed_names = {
                name
                for name in set(prior_authorities) | set(next_authorities)
                if prior_authorities.get(name) != next_authorities.get(name)
            }
            if (
                transition.prior_graph_state_sha256
                != prior_authorities.get("graph-state")
                or transition.next_graph_state_sha256
                != next_authorities.get("graph-state")
                or set(transition.retained_route_sha256s) != prior_route_sha256s
                or set(transition.added_route_sha256s)
                != next_route_sha256s - prior_route_sha256s
                or set(transition.changed_authority_names) != actual_changed_names
            ):
                raise LanguageBridgeIntegrityError(
                    "route transition proof does not justify authority retention"
                )
            names = transition.changed_authority_names
        elif isinstance(transition, MacroActionPromotionReceipt):
            if (
                transition.parent_frontier_sha256 != language.frontier.sha256
                or transition.next_frontier_sha256 != next_frontier.sha256
            ):
                raise LanguageBridgeIntegrityError(
                    "macro promotion proof names another frontier pair"
                )
            try:
                promoted_binding = next_frontier.binding(transition.promoted_action_id)
            except KeyError as exc:
                raise LanguageBridgeIntegrityError(
                    "macro promotion action is absent from the next frontier"
                ) from exc
            expected_lineage = _sha256(
                {
                    "parent_frontier_sha256": language.frontier.sha256,
                    "prior_lineage_sha256": prior_authorities.get(
                        "language-frontier-lineage"
                    ),
                    "promotion_evidence_sha256": (transition.promotion_evidence_sha256),
                }
            )
            if (
                promoted_binding.artifact_kind != transition.artifact_kind
                or promoted_binding.artifact_sha256 != transition.artifact_sha256
                or next_authorities.get("language-frontier-lineage") != expected_lineage
            ):
                raise LanguageBridgeIntegrityError(
                    "macro promotion proof does not justify authority retention"
                )
            names = ("language-frontier-lineage",)
        else:
            raise TypeError("unsupported authority transition receipt")
        for name in names:
            prior = proof_map.get(name)
            if prior is not None and prior != transition.sha256:
                raise LanguageBridgeIntegrityError(
                    "authority has conflicting transition proofs"
                )
            proof_map[name] = transition.sha256
    proofs = _hash_items(proof_map, field="authority_transition_proofs")
    proof_names = {name for name, _ in proofs}
    changed_authorities = {
        name
        for name in set(prior_authorities) | set(next_authorities)
        if prior_authorities.get(name) != next_authorities.get(name)
    }
    reward_policy_retained = prior_authorities.get(
        "language-reward-policy"
    ) == next_authorities.get("language-reward-policy")
    authority_evidence_retained = (
        changed_authorities - {"language-reward-policy"}
    ) <= proof_names
    context_evidence_retained = (
        language.frontier.context_schema_sha256 == next_frontier.context_schema_sha256
    )
    shared = set(old_bindings) & set(new_bindings)
    retained = tuple(
        sorted(
            action
            for action in shared
            if old_bindings[action] == new_bindings[action]
            and reward_policy_retained
            and authority_evidence_retained
        )
    )
    reset = tuple(sorted(action for action in shared if action not in set(retained)))
    added = tuple(sorted(set(new_bindings) - set(old_bindings)))
    removed = tuple(sorted(set(old_bindings) - set(new_bindings)))
    supplied_words = tuple(
        _identifier(value, field="additional_word_id") for value in additional_word_ids
    )
    if len(set(supplied_words)) != len(supplied_words) or set(supplied_words) & set(
        language.vocabulary
    ):
        raise ValueError("additional_word_ids must be new and unique")
    vocabulary = list(language.vocabulary)
    vocabulary.extend(supplied_words)
    generated_words = []
    while len(vocabulary) < len(next_frontier.actions):
        ordinal = len(generated_words)
        candidate = (
            "frontier-word-"
            + _sha256(
                {
                    "frontier_sha256": next_frontier.sha256,
                    "ordinal": ordinal,
                    "seed": seed,
                }
            )[:32]
        )
        if candidate not in set(vocabulary):
            vocabulary.append(candidate)
            generated_words.append(candidate)
    added_words = tuple((*supplied_words, *generated_words))
    migrated = ConsequenceMarkovLanguage(
        next_frontier,
        tuple(vocabulary),
        seed=seed,
        minimum_visits=language.minimum_visits,
        minimum_value=language.minimum_value,
        minimum_margin=language.minimum_margin,
        context_full_weight_visits=language.context_full_weight_visits,
    )
    old_indices = {action: index for index, action in enumerate(language.action_ids)}
    new_indices = {action: index for index, action in enumerate(migrated.action_ids)}
    old_word_count = len(language.vocabulary)
    for action in retained:
        old_index = old_indices[action]
        new_index = new_indices[action]
        migrated.sender_q[new_index, :old_word_count] = language.sender_q[old_index]
        migrated.receiver_q[:old_word_count, new_index] = language.receiver_q[
            :, old_index
        ]
        migrated.receiver_visits[:old_word_count, new_index] = language.receiver_visits[
            :, old_index
        ]
    if context_evidence_retained:
        for context in sorted(language.context_q):
            new_q, new_visits = migrated._ensure_context(context)
            old_q = language.context_q[context]
            old_visits = language.context_visits[context]
            for action in retained:
                old_index = old_indices[action]
                new_index = new_indices[action]
                new_q[:old_word_count, new_index] = old_q[:, old_index]
                new_visits[:old_word_count, new_index] = old_visits[:, old_index]
    migrated.training_episodes = language.training_episodes
    migration_basis = {
        "added_action_ids": list(added),
        "added_word_ids": list(added_words),
        "authority_evidence_retained": authority_evidence_retained,
        "authority_transition_proofs": dict(proofs),
        "context_evidence_retained": context_evidence_retained,
        "format": FRONTIER_MIGRATION_SCHEMA,
        "next_frontier_sha256": next_frontier.sha256,
        "prior_frontier_sha256": language.frontier.sha256,
        "prior_language_state_sha256": prior_state_sha256,
        "removed_action_ids": list(removed),
        "reset_action_ids": list(reset),
        "reward_policy_retained": reward_policy_retained,
        "retained_action_ids": list(retained),
        "seed": seed,
    }
    migrated._transition_head_sha256 = _sha256(migration_basis)
    next_state_sha256 = migrated.sha256
    receipt = FrontierMigrationReceipt(
        prior_frontier_sha256=language.frontier.sha256,
        next_frontier_sha256=next_frontier.sha256,
        prior_language_state_sha256=prior_state_sha256,
        next_language_state_sha256=next_state_sha256,
        retained_action_ids=retained,
        added_action_ids=added,
        removed_action_ids=removed,
        reset_action_ids=reset,
        added_word_ids=added_words,
        authority_transition_proofs=proofs,
        authority_evidence_retained=authority_evidence_retained,
        context_evidence_retained=context_evidence_retained,
        reward_policy_retained=reward_policy_retained,
        seed=seed,
    )
    return migrated, receipt


@dataclass(frozen=True, slots=True)
class VerifiedWordTrajectory:
    episode_sha256: str
    terminal_verifier_sha256: str
    terminal_verification_sha256: str
    expected_steps: int
    language_snapshot_sha256: str
    frontier_sha256: str
    authority_hashes: tuple[tuple[str, str], ...]
    word_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    context_ids: tuple[str, ...]
    outcome_commit_sha256s: tuple[str, ...]

    FORMAT = VERIFIED_WORD_TRAJECTORY_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "episode_sha256",
            "terminal_verifier_sha256",
            "terminal_verification_sha256",
            "language_snapshot_sha256",
            "frontier_sha256",
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
        words = tuple(_identifier(value, field="word_id") for value in self.word_ids)
        actions = tuple(
            _identifier(value, field="action_id") for value in self.action_ids
        )
        contexts = tuple(
            _identifier(value, field="context_id") for value in self.context_ids
        )
        outcomes = tuple(
            require_sha256(value, field="outcome_commit_sha256")
            for value in self.outcome_commit_sha256s
        )
        if not 2 <= len(words) <= MAX_TRAJECTORY_LENGTH or not (
            len(words) == len(actions) == len(contexts) == len(outcomes)
        ):
            raise ValueError("verified word trajectory has invalid lengths")
        if (
            isinstance(self.expected_steps, bool)
            or not isinstance(self.expected_steps, int)
            or self.expected_steps != len(words)
        ):
            raise ValueError("trajectory expected_steps does not equal its length")
        if len(set(outcomes)) != len(outcomes):
            raise ValueError("trajectory outcome commits must be unique")
        if dict(self.authority_hashes).get("episode-terminal-verifier") != (
            self.terminal_verifier_sha256
        ):
            raise ValueError("trajectory terminal verifier is not frontier-authorized")
        object.__setattr__(self, "word_ids", words)
        object.__setattr__(self, "action_ids", actions)
        object.__setattr__(self, "context_ids", contexts)
        object.__setattr__(self, "outcome_commit_sha256s", outcomes)

    @classmethod
    def create(
        cls,
        commits: Sequence[LanguageOutcomeCommitReceipt],
        snapshot: LanguageSnapshot,
        *,
        episode_sha256: str,
        terminal_verifier_sha256: str,
        terminal_verification_sha256: str,
        expected_steps: int,
        frontier: ActionFrontier,
    ) -> "VerifiedWordTrajectory":
        rows = tuple(commits)
        if not rows or any(
            not isinstance(item, LanguageOutcomeCommitReceipt) for item in rows
        ):
            raise TypeError("commits must contain language outcome receipts")
        if not isinstance(snapshot, LanguageSnapshot):
            raise TypeError("snapshot must be a LanguageSnapshot")
        if not isinstance(frontier, ActionFrontier):
            raise TypeError("frontier must be an ActionFrontier")
        if snapshot.frontier_sha256 != frontier.sha256:
            raise LanguageBridgeIntegrityError("snapshot frontier is stale")
        if any(
            not item.accepted
            or item.frontier_sha256 != frontier.sha256
            or snapshot.decode_word(item.word_id, item.context_id) != item.action_id
            for item in rows
        ):
            raise LanguageBridgeIntegrityError(
                "trajectory contains rejected or semantically mismatched outcomes"
            )
        return cls(
            episode_sha256=episode_sha256,
            terminal_verifier_sha256=terminal_verifier_sha256,
            terminal_verification_sha256=terminal_verification_sha256,
            expected_steps=expected_steps,
            language_snapshot_sha256=snapshot.sha256,
            frontier_sha256=frontier.sha256,
            authority_hashes=frontier.authority_hashes,
            word_ids=tuple(item.word_id for item in rows),
            action_ids=tuple(item.action_id for item in rows),
            context_ids=tuple(item.context_id for item in rows),
            outcome_commit_sha256s=tuple(item.sha256 for item in rows),
        )

    @classmethod
    def from_demand_episode(
        cls,
        rows: Sequence[
            tuple[LanguageOutcomeCommitReceipt, DemandRoutedExecutionReceipt]
        ],
        snapshot: LanguageSnapshot,
        *,
        episode_sha256: str,
        terminal_verifier_sha256: str,
        terminal_verification_sha256: str,
        expected_steps: int,
        frontier: ActionFrontier,
    ) -> "VerifiedWordTrajectory":
        episode = require_sha256(episode_sha256, field="episode_sha256")
        paired = tuple(rows)
        if not paired or any(
            not isinstance(commit, LanguageOutcomeCommitReceipt)
            or not isinstance(routed, DemandRoutedExecutionReceipt)
            for commit, routed in paired
        ):
            raise TypeError("rows must contain language/demand receipt pairs")
        ordered = tuple(
            sorted(
                paired,
                key=lambda item: item[1].selection.event.logical_time,
            )
        )
        logical_times = tuple(
            routed.selection.event.logical_time for _, routed in ordered
        )
        if len(set(logical_times)) != len(logical_times):
            raise LanguageBridgeIntegrityError(
                "demand episode contains duplicate selection times"
            )
        if any(
            routed.episode_sha256 != episode
            or not routed.outcome.success
            or not commit.accepted
            or commit.external_outcome_kind != "demand-routed-execution"
            or commit.external_outcome_sha256 != routed.sha256
            or commit.action_id != route_action_id(routed.outcome.route_sha256)
            for commit, routed in ordered
        ):
            raise LanguageBridgeIntegrityError(
                "demand episode contains incomplete, rejected, or mismatched steps"
            )
        return cls.create(
            tuple(commit for commit, _ in ordered),
            snapshot,
            episode_sha256=episode,
            terminal_verifier_sha256=terminal_verifier_sha256,
            terminal_verification_sha256=terminal_verification_sha256,
            expected_steps=expected_steps,
            frontier=frontier,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "action_ids": list(self.action_ids),
            "authority_hashes": dict(self.authority_hashes),
            "context_ids": list(self.context_ids),
            "episode_sha256": self.episode_sha256,
            "expected_steps": self.expected_steps,
            "format": self.FORMAT,
            "frontier_sha256": self.frontier_sha256,
            "language_snapshot_sha256": self.language_snapshot_sha256,
            "outcome_commit_sha256s": list(self.outcome_commit_sha256s),
            "terminal_verifier_sha256": self.terminal_verifier_sha256,
            "terminal_verification_sha256": self.terminal_verification_sha256,
            "word_ids": list(self.word_ids),
        }

    def to_bytes(self) -> bytes:
        return _seal(self.FORMAT, self.to_record())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "VerifiedWordTrajectory":
        body = _open(data, schema=cls.FORMAT, label="verified word trajectory")
        expected = {
            "action_ids",
            "authority_hashes",
            "context_ids",
            "episode_sha256",
            "expected_steps",
            "format",
            "frontier_sha256",
            "language_snapshot_sha256",
            "outcome_commit_sha256s",
            "terminal_verifier_sha256",
            "terminal_verification_sha256",
            "word_ids",
        }
        if set(body) != expected or body.get("format") != cls.FORMAT:
            raise LanguageBridgeIntegrityError("trajectory body is invalid")
        authorities = body.get("authority_hashes")
        collections = {
            field: body.get(field)
            for field in (
                "action_ids",
                "context_ids",
                "outcome_commit_sha256s",
                "word_ids",
            )
        }
        if not isinstance(authorities, Mapping) or any(
            not isinstance(value, list) for value in collections.values()
        ):
            raise LanguageBridgeIntegrityError("trajectory inventory is invalid")
        try:
            result = cls(
                episode_sha256=cast(str, body.get("episode_sha256")),
                terminal_verifier_sha256=cast(
                    str, body.get("terminal_verifier_sha256")
                ),
                terminal_verification_sha256=cast(
                    str, body.get("terminal_verification_sha256")
                ),
                expected_steps=cast(int, body.get("expected_steps")),
                language_snapshot_sha256=cast(
                    str, body.get("language_snapshot_sha256")
                ),
                frontier_sha256=cast(str, body.get("frontier_sha256")),
                authority_hashes=_hash_items(
                    cast(Mapping[str, str], authorities), field="authority_hashes"
                ),
                word_ids=tuple(cast(list[str], collections["word_ids"])),
                action_ids=tuple(cast(list[str], collections["action_ids"])),
                context_ids=tuple(cast(list[str], collections["context_ids"])),
                outcome_commit_sha256s=tuple(
                    cast(list[str], collections["outcome_commit_sha256s"])
                ),
            )
        except (TypeError, ValueError) as exc:
            raise LanguageBridgeIntegrityError(
                "trajectory reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise LanguageBridgeIntegrityError(
                "trajectory failed canonical reconstruction"
            )
        return result


@dataclass(frozen=True, slots=True)
class LanguageMacroDiscoveryReceipt:
    language_snapshot_sha256: str
    frontier_sha256: str
    authority_hashes: tuple[tuple[str, str], ...]
    trajectory_sha256s: tuple[str, ...]
    min_support: int
    min_macro_length: int
    max_macro_length: int
    candidate_count: int
    definitions: tuple[ExecutableWordDefinition, ...]
    definition_supports: tuple[tuple[str, tuple[str, ...]], ...]

    FORMAT = LANGUAGE_MACRO_DISCOVERY_SCHEMA

    def __post_init__(self) -> None:
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
        trajectories = tuple(
            sorted(
                require_sha256(value, field="trajectory_sha256")
                for value in self.trajectory_sha256s
            )
        )
        if not trajectories or len(set(trajectories)) != len(trajectories):
            raise ValueError("macro discovery trajectory inventory is invalid")
        object.__setattr__(self, "trajectory_sha256s", trajectories)
        for field_name, lower, upper in (
            ("min_support", 1, MAX_TRAJECTORIES),
            ("min_macro_length", 2, MAX_MACRO_LENGTH),
            ("max_macro_length", 2, MAX_MACRO_LENGTH),
            ("candidate_count", 0, MAX_TRAJECTORIES * MAX_TRAJECTORY_LENGTH),
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not (lower <= value <= upper)
            ):
                raise ValueError(f"{field_name} is outside its bound")
        if self.max_macro_length < self.min_macro_length:
            raise ValueError("macro length bounds are inverted")
        definitions = tuple(sorted(self.definitions, key=lambda item: item.new_word_id))
        if len(definitions) > MAX_DISCOVERED_DEFINITIONS or any(
            not isinstance(item, ExecutableWordDefinition) for item in definitions
        ):
            raise ValueError("macro definition inventory is invalid")
        if len({item.new_word_id for item in definitions}) != len(definitions):
            raise ValueError("macro definitions contain duplicate words")
        if any(
            item.language_snapshot_sha256 != self.language_snapshot_sha256
            or item.frontier_sha256 != self.frontier_sha256
            or item.authority_hashes != self.authority_hashes
            for item in definitions
        ):
            raise ValueError("macro definition execution contract is stale")
        object.__setattr__(self, "definitions", definitions)
        supports = tuple(
            sorted(
                (
                    require_sha256(definition_sha, field="definition_sha256"),
                    tuple(
                        sorted(
                            require_sha256(value, field="support_trajectory_sha256")
                            for value in sources
                        )
                    ),
                )
                for definition_sha, sources in self.definition_supports
            )
        )
        if {item.sha256 for item in definitions} != {
            definition_sha for definition_sha, _ in supports
        } or any(
            len(sources) < self.min_support
            or len(set(sources)) != len(sources)
            or not set(sources) <= set(trajectories)
            for _, sources in supports
        ):
            raise ValueError("macro support inventory disagrees with definitions")
        object.__setattr__(self, "definition_supports", supports)

    def to_record(self) -> dict[str, object]:
        return {
            "authority_hashes": dict(self.authority_hashes),
            "candidate_count": self.candidate_count,
            "definition_supports": {
                definition: list(sources)
                for definition, sources in self.definition_supports
            },
            "definitions": [item.to_record() for item in self.definitions],
            "format": self.FORMAT,
            "frontier_sha256": self.frontier_sha256,
            "language_snapshot_sha256": self.language_snapshot_sha256,
            "max_macro_length": self.max_macro_length,
            "min_macro_length": self.min_macro_length,
            "min_support": self.min_support,
            "trajectory_sha256s": list(self.trajectory_sha256s),
        }

    def to_bytes(self) -> bytes:
        return _seal(self.FORMAT, self.to_record())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "LanguageMacroDiscoveryReceipt":
        body = _open(data, schema=cls.FORMAT, label="language macro discovery")
        expected = {
            "authority_hashes",
            "candidate_count",
            "definition_supports",
            "definitions",
            "format",
            "frontier_sha256",
            "language_snapshot_sha256",
            "max_macro_length",
            "min_macro_length",
            "min_support",
            "trajectory_sha256s",
        }
        if set(body) != expected or body.get("format") != cls.FORMAT:
            raise LanguageBridgeIntegrityError("macro discovery body is invalid")
        authorities = body.get("authority_hashes")
        raw_definitions = body.get("definitions")
        raw_supports = body.get("definition_supports")
        trajectories = body.get("trajectory_sha256s")
        if (
            not isinstance(authorities, Mapping)
            or not isinstance(raw_definitions, list)
            or not isinstance(raw_supports, Mapping)
            or not isinstance(trajectories, list)
        ):
            raise LanguageBridgeIntegrityError("macro discovery inventory is invalid")
        try:
            definitions = tuple(
                ExecutableWordDefinition.from_bytes(
                    _seal(
                        ExecutableWordDefinition.FORMAT,
                        cast(Mapping[str, object], row),
                    )
                )
                for row in raw_definitions
            )
            supports = tuple(
                (definition, tuple(cast(list[str], sources)))
                for definition, sources in raw_supports.items()
                if isinstance(sources, list)
            )
            if len(supports) != len(raw_supports):
                raise ValueError("support values must be lists")
            result = cls(
                language_snapshot_sha256=cast(
                    str, body.get("language_snapshot_sha256")
                ),
                frontier_sha256=cast(str, body.get("frontier_sha256")),
                authority_hashes=_hash_items(
                    cast(Mapping[str, str], authorities), field="authority_hashes"
                ),
                trajectory_sha256s=tuple(trajectories),
                min_support=cast(int, body.get("min_support")),
                min_macro_length=cast(int, body.get("min_macro_length")),
                max_macro_length=cast(int, body.get("max_macro_length")),
                candidate_count=cast(int, body.get("candidate_count")),
                definitions=definitions,
                definition_supports=supports,
            )
        except (TypeError, ValueError) as exc:
            raise LanguageBridgeIntegrityError(
                "macro discovery reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise LanguageBridgeIntegrityError(
                "macro discovery failed canonical reconstruction"
            )
        return result


def discover_language_macros(
    trajectories: Sequence[VerifiedWordTrajectory],
    snapshot: LanguageSnapshot,
    frontier: ActionFrontier,
    *,
    min_support: int = 2,
    min_macro_length: int = 2,
    max_macro_length: int = 8,
    max_definitions: int = 1_024,
) -> LanguageMacroDiscoveryReceipt:
    """Mine repeated globally stable word programs from verified trajectories."""

    rows = tuple(trajectories)
    if not 1 <= len(rows) <= MAX_TRAJECTORIES or any(
        not isinstance(item, VerifiedWordTrajectory) for item in rows
    ):
        raise ValueError("trajectory inventory is invalid")
    if not isinstance(snapshot, LanguageSnapshot):
        raise TypeError("snapshot must be a LanguageSnapshot")
    if not isinstance(frontier, ActionFrontier):
        raise TypeError("frontier must be an ActionFrontier")
    if snapshot.frontier_sha256 != frontier.sha256:
        raise LanguageBridgeIntegrityError("snapshot frontier is stale")
    support_floor = int(min_support)
    lower = int(min_macro_length)
    upper = int(max_macro_length)
    limit = int(max_definitions)
    if not 1 <= support_floor <= len(rows):
        raise ValueError("min_support is outside the trajectory inventory")
    if not 2 <= lower <= upper <= MAX_MACRO_LENGTH:
        raise ValueError("macro length bounds are invalid")
    if not 1 <= limit <= MAX_DISCOVERED_DEFINITIONS:
        raise ValueError("max_definitions is outside its bound")
    contract = (
        snapshot.sha256,
        frontier.sha256,
        frontier.authority_hashes,
    )
    if any(
        (
            row.language_snapshot_sha256,
            row.frontier_sha256,
            row.authority_hashes,
        )
        != contract
        for row in rows
    ):
        raise LanguageBridgeIntegrityError(
            "trajectories do not share the discovery contract"
        )
    trajectory_hashes = tuple(item.sha256 for item in rows)
    if len(set(trajectory_hashes)) != len(trajectory_hashes):
        raise ValueError("macro discovery trajectories must be unique")
    stable_words = dict(snapshot.global_word_actions)
    candidates: dict[tuple[str, ...], set[str]] = defaultdict(set)
    semantics: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    for row in rows:
        seen: set[tuple[str, ...]] = set()
        maximum = min(upper, len(row.word_ids))
        for length in range(lower, maximum + 1):
            for start in range(len(row.word_ids) - length + 1):
                words = row.word_ids[start : start + length]
                actions = row.action_ids[start : start + length]
                if any(
                    stable_words.get(word) != action
                    for word, action in zip(words, actions, strict=True)
                ):
                    continue
                seen.add(words)
                semantics[words].add(actions)
        for words in seen:
            candidates[words].add(row.sha256)
    selected = sorted(
        (
            (words, tuple(sorted(sources)))
            for words, sources in candidates.items()
            if len(sources) >= support_floor and len(semantics[words]) == 1
        ),
        key=lambda item: (-len(item[1]), -len(item[0]), item[0]),
    )[:limit]
    definitions = []
    supports = []
    for words, sources in selected:
        word_digest = _sha256(
            {
                "frontier_sha256": frontier.sha256,
                "snapshot_sha256": snapshot.sha256,
                "word_ids": list(words),
            }
        )
        definition = ExecutableWordDefinition.create(
            f"macro-{word_digest[:32]}",
            words,
            language_snapshot_sha256=snapshot.sha256,
            frontier_sha256=frontier.sha256,
            authority_hashes=frontier.authority_hashes,
        )
        definitions.append(definition)
        supports.append((definition.sha256, sources))
    return LanguageMacroDiscoveryReceipt(
        language_snapshot_sha256=snapshot.sha256,
        frontier_sha256=frontier.sha256,
        authority_hashes=frontier.authority_hashes,
        trajectory_sha256s=trajectory_hashes,
        min_support=support_floor,
        min_macro_length=lower,
        max_macro_length=upper,
        candidate_count=len(candidates),
        definitions=tuple(definitions),
        definition_supports=tuple(supports),
    )


@dataclass(frozen=True, slots=True)
class MacroActionPromotionReceipt:
    parent_frontier_sha256: str
    next_frontier_sha256: str
    definition_sha256: str
    discovery_sha256: str
    compiled_word_sha256: str
    promoted_action_id: str
    artifact_kind: str
    artifact_sha256: str
    promotion_evidence_sha256: str

    FORMAT = MACRO_ACTION_PROMOTION_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "parent_frontier_sha256",
            "next_frontier_sha256",
            "definition_sha256",
            "discovery_sha256",
            "compiled_word_sha256",
            "artifact_sha256",
            "promotion_evidence_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        object.__setattr__(
            self,
            "promoted_action_id",
            _identifier(self.promoted_action_id, field="promoted_action_id"),
        )
        if self.artifact_kind not in ("crystal", "program"):
            raise ValueError("promoted macro artifact kind is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "artifact_kind": self.artifact_kind,
            "artifact_sha256": self.artifact_sha256,
            "compiled_word_sha256": self.compiled_word_sha256,
            "definition_sha256": self.definition_sha256,
            "discovery_sha256": self.discovery_sha256,
            "format": self.FORMAT,
            "next_frontier_sha256": self.next_frontier_sha256,
            "parent_frontier_sha256": self.parent_frontier_sha256,
            "promoted_action_id": self.promoted_action_id,
            "promotion_evidence_sha256": self.promotion_evidence_sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(self.FORMAT, self.to_record())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "MacroActionPromotionReceipt":
        body = _open(data, schema=cls.FORMAT, label="macro action promotion")
        expected = {
            "artifact_kind",
            "artifact_sha256",
            "compiled_word_sha256",
            "definition_sha256",
            "discovery_sha256",
            "format",
            "next_frontier_sha256",
            "parent_frontier_sha256",
            "promoted_action_id",
            "promotion_evidence_sha256",
        }
        if set(body) != expected or body.get("format") != cls.FORMAT:
            raise LanguageBridgeIntegrityError("macro promotion body is invalid")
        try:
            result = cls(
                parent_frontier_sha256=cast(str, body.get("parent_frontier_sha256")),
                next_frontier_sha256=cast(str, body.get("next_frontier_sha256")),
                definition_sha256=cast(str, body.get("definition_sha256")),
                discovery_sha256=cast(str, body.get("discovery_sha256")),
                compiled_word_sha256=cast(str, body.get("compiled_word_sha256")),
                promoted_action_id=cast(str, body.get("promoted_action_id")),
                artifact_kind=cast(str, body.get("artifact_kind")),
                artifact_sha256=cast(str, body.get("artifact_sha256")),
                promotion_evidence_sha256=cast(
                    str, body.get("promotion_evidence_sha256")
                ),
            )
        except (TypeError, ValueError) as exc:
            raise LanguageBridgeIntegrityError(
                "macro promotion reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise LanguageBridgeIntegrityError(
                "macro promotion failed canonical reconstruction"
            )
        return result


def promote_compiled_word_action(
    frontier: ActionFrontier,
    discovery: LanguageMacroDiscoveryReceipt,
    definition: ExecutableWordDefinition,
    compiled: CompiledWordReceipt,
    *,
    action_id: str | None = None,
) -> tuple[ActionFrontier, MacroActionPromotionReceipt]:
    """Append one verified compiled word as a new language action revision."""

    if not isinstance(frontier, ActionFrontier):
        raise TypeError("frontier must be an ActionFrontier")
    if not isinstance(discovery, LanguageMacroDiscoveryReceipt):
        raise TypeError("discovery must be a LanguageMacroDiscoveryReceipt")
    if not isinstance(definition, ExecutableWordDefinition):
        raise TypeError("definition must be an ExecutableWordDefinition")
    if not isinstance(compiled, CompiledWordReceipt):
        raise TypeError("compiled must be a CompiledWordReceipt")
    if (
        discovery.frontier_sha256 != frontier.sha256
        or discovery.authority_hashes != frontier.authority_hashes
        or definition not in discovery.definitions
        or compiled.definition_sha256 != definition.sha256
        or compiled.frontier_sha256 != frontier.sha256
        or compiled.language_snapshot_sha256 != discovery.language_snapshot_sha256
        or compiled.authority_hashes != frontier.authority_hashes
    ):
        raise LanguageBridgeIntegrityError(
            "macro promotion evidence belongs to another contract"
        )
    promoted_id = (
        f"word-action:{definition.sha256[:32]}"
        if action_id is None
        else _identifier(action_id, field="action_id")
    )
    if promoted_id in set(frontier.action_ids):
        raise LanguageBridgeConflictError("promoted action ID already exists")
    evidence_sha256 = _sha256(
        {
            "compiled_word_sha256": compiled.sha256,
            "definition_sha256": definition.sha256,
            "discovery_sha256": discovery.sha256,
            "format": MACRO_ACTION_PROMOTION_SCHEMA,
            "parent_frontier_sha256": frontier.sha256,
        }
    )
    action_schema_sha256 = _sha256(
        {
            "format": "immer-ooe-promoted-language-action-schema/v1",
            "new_action": {
                "action_id": promoted_id,
                "artifact_kind": compiled.artifact_kind,
                "artifact_sha256": compiled.artifact_sha256,
                "promotion_evidence_sha256": evidence_sha256,
            },
            "parent_action_schema_sha256": frontier.action_schema_sha256,
            "parent_frontier_sha256": frontier.sha256,
        }
    )
    authorities = dict(frontier.authority_hashes)
    authorities["language-frontier-lineage"] = _sha256(
        {
            "parent_frontier_sha256": frontier.sha256,
            "prior_lineage_sha256": authorities.get("language-frontier-lineage"),
            "promotion_evidence_sha256": evidence_sha256,
        }
    )
    next_frontier = ActionFrontier.create(
        (
            *frontier.actions,
            ActionBinding(
                promoted_id,
                compiled.artifact_kind,
                compiled.artifact_sha256,
            ),
        ),
        action_schema_sha256=action_schema_sha256,
        context_schema_sha256=frontier.context_schema_sha256,
        authority_hashes=authorities,
    )
    receipt = MacroActionPromotionReceipt(
        parent_frontier_sha256=frontier.sha256,
        next_frontier_sha256=next_frontier.sha256,
        definition_sha256=definition.sha256,
        discovery_sha256=discovery.sha256,
        compiled_word_sha256=compiled.sha256,
        promoted_action_id=promoted_id,
        artifact_kind=compiled.artifact_kind,
        artifact_sha256=compiled.artifact_sha256,
        promotion_evidence_sha256=evidence_sha256,
    )
    return next_frontier, receipt


@dataclass(frozen=True, slots=True)
class LanguageRevisionPromotionReceipt:
    macro_promotion_sha256: str
    frontier_migration_sha256: str
    parent_frontier_sha256: str
    next_frontier_sha256: str
    prior_language_state_sha256: str
    next_language_state_sha256: str
    persisted_state_sha256: str

    FORMAT = LANGUAGE_REVISION_PROMOTION_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "macro_promotion_sha256",
            "frontier_migration_sha256",
            "parent_frontier_sha256",
            "next_frontier_sha256",
            "prior_language_state_sha256",
            "next_language_state_sha256",
            "persisted_state_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        if self.next_language_state_sha256 != self.persisted_state_sha256:
            raise ValueError("promoted language state was not persisted exactly")

    def to_record(self) -> dict[str, object]:
        return {
            "format": self.FORMAT,
            "frontier_migration_sha256": self.frontier_migration_sha256,
            "macro_promotion_sha256": self.macro_promotion_sha256,
            "next_frontier_sha256": self.next_frontier_sha256,
            "next_language_state_sha256": self.next_language_state_sha256,
            "parent_frontier_sha256": self.parent_frontier_sha256,
            "persisted_state_sha256": self.persisted_state_sha256,
            "prior_language_state_sha256": self.prior_language_state_sha256,
        }

    def to_bytes(self) -> bytes:
        return _seal(self.FORMAT, self.to_record())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "LanguageRevisionPromotionReceipt":
        body = _open(data, schema=cls.FORMAT, label="language revision promotion")
        expected = {
            "format",
            "frontier_migration_sha256",
            "macro_promotion_sha256",
            "next_frontier_sha256",
            "next_language_state_sha256",
            "parent_frontier_sha256",
            "persisted_state_sha256",
            "prior_language_state_sha256",
        }
        if set(body) != expected or body.get("format") != cls.FORMAT:
            raise LanguageBridgeIntegrityError(
                "language revision promotion body is invalid"
            )
        try:
            result = cls(
                macro_promotion_sha256=cast(str, body.get("macro_promotion_sha256")),
                frontier_migration_sha256=cast(
                    str, body.get("frontier_migration_sha256")
                ),
                parent_frontier_sha256=cast(str, body.get("parent_frontier_sha256")),
                next_frontier_sha256=cast(str, body.get("next_frontier_sha256")),
                prior_language_state_sha256=cast(
                    str, body.get("prior_language_state_sha256")
                ),
                next_language_state_sha256=cast(
                    str, body.get("next_language_state_sha256")
                ),
                persisted_state_sha256=cast(str, body.get("persisted_state_sha256")),
            )
        except (TypeError, ValueError) as exc:
            raise LanguageBridgeIntegrityError(
                "language revision promotion reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise LanguageBridgeIntegrityError(
                "language revision promotion failed canonical reconstruction"
            )
        return result


def promote_and_migrate_language_revision(
    language: ConsequenceMarkovLanguage,
    state_bank: ConsequenceLanguageStateBank,
    discovery: LanguageMacroDiscoveryReceipt,
    definition: ExecutableWordDefinition,
    compiled: CompiledWordReceipt,
    *,
    seed: int,
    action_id: str | None = None,
) -> tuple[
    ConsequenceMarkovLanguage,
    ActionFrontier,
    MacroActionPromotionReceipt,
    FrontierMigrationReceipt,
    LanguageRevisionPromotionReceipt,
]:
    """Atomically publish a compiled word as a new learned action revision."""

    if not isinstance(language, ConsequenceMarkovLanguage):
        raise TypeError("language must be a ConsequenceMarkovLanguage")
    if not isinstance(state_bank, ConsequenceLanguageStateBank):
        raise TypeError("state_bank must be a ConsequenceLanguageStateBank")
    prior_state_sha256 = language.sha256
    if state_bank.current_state_sha256() != prior_state_sha256:
        raise LanguageBridgeConflictError(
            "language bank head differs from the promotion source state"
        )
    next_frontier, promotion = promote_compiled_word_action(
        language.frontier,
        discovery,
        definition,
        compiled,
        action_id=action_id,
    )
    migrated, migration = migrate_language_frontier(
        language,
        next_frontier,
        seed=seed,
        authority_transition_receipts=(promotion,),
    )
    receipt = LanguageRevisionPromotionReceipt(
        macro_promotion_sha256=promotion.sha256,
        frontier_migration_sha256=migration.sha256,
        parent_frontier_sha256=language.frontier.sha256,
        next_frontier_sha256=next_frontier.sha256,
        prior_language_state_sha256=prior_state_sha256,
        next_language_state_sha256=migrated.sha256,
        persisted_state_sha256=migrated.sha256,
    )
    persisted = state_bank.publish_with_revision(
        migrated,
        receipt,
        expected_state_sha256=prior_state_sha256,
    )
    if persisted != migrated.sha256:
        raise LanguageBridgeIntegrityError(
            "promoted language publication changed its state address"
        )
    return migrated, next_frontier, promotion, migration, receipt


@dataclass(frozen=True, slots=True)
class RouteWordResolutionReceipt:
    language_snapshot_sha256: str
    frontier_sha256: str
    graph_state_sha256: str
    context_id: str | None
    primitive_bindings: tuple[PrimitiveWordBinding, ...]
    route_program_pairs: tuple[tuple[str, str], ...]

    FORMAT = ROUTE_WORD_RESOLUTION_SCHEMA

    def __post_init__(self) -> None:
        for field_name in (
            "language_snapshot_sha256",
            "frontier_sha256",
            "graph_state_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        context = self.context_id
        if context is not None:
            context = _identifier(context, field="context_id")
        object.__setattr__(self, "context_id", context)
        bindings = tuple(sorted(self.primitive_bindings, key=lambda item: item.word_id))
        if (
            not bindings
            or any(not isinstance(item, PrimitiveWordBinding) for item in bindings)
            or any(item.artifact_kind != "program" for item in bindings)
        ):
            raise ValueError("resolved primitive bindings are invalid")
        if len({item.word_id for item in bindings}) != len(bindings):
            raise ValueError("resolved word bindings contain duplicates")
        object.__setattr__(self, "primitive_bindings", bindings)
        pairs = tuple(
            sorted(
                (
                    require_sha256(route, field="route_sha256"),
                    require_sha256(program, field="program_sha256"),
                )
                for route, program in self.route_program_pairs
            )
        )
        if len({route for route, _ in pairs}) != len(pairs):
            raise ValueError("route program inventory contains duplicates")
        if {item.artifact_sha256 for item in bindings} - {
            program for _, program in pairs
        }:
            raise ValueError("primitive binding lacks its route program evidence")
        object.__setattr__(self, "route_program_pairs", pairs)

    def to_record(self) -> dict[str, object]:
        return {
            "context_id": self.context_id,
            "format": self.FORMAT,
            "frontier_sha256": self.frontier_sha256,
            "graph_state_sha256": self.graph_state_sha256,
            "language_snapshot_sha256": self.language_snapshot_sha256,
            "primitive_bindings": [
                item.to_record() for item in self.primitive_bindings
            ],
            "route_program_pairs": [
                {"program_sha256": program, "route_sha256": route}
                for route, program in self.route_program_pairs
            ],
        }

    def to_bytes(self) -> bytes:
        return _seal(self.FORMAT, self.to_record())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "RouteWordResolutionReceipt":
        body = _open(data, schema=cls.FORMAT, label="route word resolution")
        expected = {
            "context_id",
            "format",
            "frontier_sha256",
            "graph_state_sha256",
            "language_snapshot_sha256",
            "primitive_bindings",
            "route_program_pairs",
        }
        if set(body) != expected or body.get("format") != cls.FORMAT:
            raise LanguageBridgeIntegrityError("route resolution body is invalid")
        raw_bindings = body.get("primitive_bindings")
        raw_pairs = body.get("route_program_pairs")
        if not isinstance(raw_bindings, list) or not isinstance(raw_pairs, list):
            raise LanguageBridgeIntegrityError("route resolution inventory is invalid")
        try:
            pairs = []
            for row in raw_pairs:
                if not isinstance(row, Mapping) or set(row) != {
                    "program_sha256",
                    "route_sha256",
                }:
                    raise ValueError("invalid route/program pair")
                pairs.append(
                    (
                        cast(str, row.get("route_sha256")),
                        cast(str, row.get("program_sha256")),
                    )
                )
            result = cls(
                language_snapshot_sha256=cast(
                    str, body.get("language_snapshot_sha256")
                ),
                frontier_sha256=cast(str, body.get("frontier_sha256")),
                graph_state_sha256=cast(str, body.get("graph_state_sha256")),
                context_id=cast(str | None, body.get("context_id")),
                primitive_bindings=tuple(
                    PrimitiveWordBinding.from_record(row) for row in raw_bindings
                ),
                route_program_pairs=tuple(pairs),
            )
        except (TypeError, ValueError) as exc:
            raise LanguageBridgeIntegrityError(
                "route resolution reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise LanguageBridgeIntegrityError(
                "route resolution failed canonical reconstruction"
            )
        return result


def resolve_route_words_to_programs(
    snapshot: LanguageSnapshot,
    frontier: ActionFrontier,
    graph_state: ComputeOperatorGraphState,
    *,
    context_id: str | None = None,
) -> tuple[tuple[PrimitiveWordBinding, ...], RouteWordResolutionReceipt]:
    """Resolve learned route words into locally compilable program bindings."""

    if not isinstance(snapshot, LanguageSnapshot):
        raise TypeError("snapshot must be a LanguageSnapshot")
    if not isinstance(frontier, ActionFrontier):
        raise TypeError("frontier must be an ActionFrontier")
    if not isinstance(graph_state, ComputeOperatorGraphState):
        raise TypeError("graph_state must be a ComputeOperatorGraphState")
    if (
        snapshot.frontier_sha256 != frontier.sha256
        or dict(frontier.authority_hashes).get("graph-state") != graph_state.sha256
    ):
        raise LanguageBridgeIntegrityError("route-word resolution contract is stale")
    artifacts = snapshot.primitive_artifacts(
        frontier,
        context_id=context_id,
    )
    routes = {route.sha256: route for route in graph_state.materialized_routes}
    bindings = []
    route_program_pairs = set()
    for word, (kind, route_sha256) in sorted(artifacts.items()):
        if kind != "materialized-route":
            raise LanguageBridgeIntegrityError(
                "route frontier exported a non-route artifact"
            )
        route = routes.get(route_sha256)
        if route is None:
            raise LanguageBridgeIntegrityError(
                "learned route is absent from the bound graph state"
            )
        bindings.append(
            PrimitiveWordBinding(
                word,
                "program",
                route.executable_program_sha256,
            )
        )
        route_program_pairs.add((route.sha256, route.executable_program_sha256))
    if not bindings:
        raise LanguageBridgeError("snapshot has no executable stable route words")
    receipt = RouteWordResolutionReceipt(
        language_snapshot_sha256=snapshot.sha256,
        frontier_sha256=frontier.sha256,
        graph_state_sha256=graph_state.sha256,
        context_id=context_id,
        primitive_bindings=tuple(bindings),
        route_program_pairs=tuple(route_program_pairs),
    )
    return receipt.primitive_bindings, receipt


@dataclass(frozen=True, slots=True)
class SnapshotComputeResolutionReceipt:
    language_snapshot_sha256: str
    frontier_sha256: str
    context_id: str | None
    graph_state_sha256: str | None
    primitive_bindings: tuple[PrimitiveWordBinding, ...]
    resolutions: tuple[tuple[str, str, str, str, str], ...]

    FORMAT = SNAPSHOT_COMPUTE_RESOLUTION_SCHEMA

    def __post_init__(self) -> None:
        for field_name in ("language_snapshot_sha256", "frontier_sha256"):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        context = self.context_id
        if context is not None:
            context = _identifier(context, field="context_id")
        object.__setattr__(self, "context_id", context)
        graph = self.graph_state_sha256
        if graph is not None:
            graph = require_sha256(graph, field="graph_state_sha256")
        object.__setattr__(self, "graph_state_sha256", graph)
        bindings = tuple(sorted(self.primitive_bindings, key=lambda item: item.word_id))
        if (
            not bindings
            or any(not isinstance(item, PrimitiveWordBinding) for item in bindings)
            or any(
                item.artifact_kind not in ("crystal", "program") for item in bindings
            )
        ):
            raise ValueError("snapshot compute bindings are invalid")
        if len({item.word_id for item in bindings}) != len(bindings):
            raise ValueError("snapshot compute bindings contain duplicate words")
        object.__setattr__(self, "primitive_bindings", bindings)
        resolutions = tuple(
            sorted(
                (
                    _identifier(word, field="word_id"),
                    _identifier(source_kind, field="source_kind"),
                    require_sha256(source_sha, field="source_sha256"),
                    _identifier(resolved_kind, field="resolved_kind"),
                    require_sha256(resolved_sha, field="resolved_sha256"),
                )
                for word, source_kind, source_sha, resolved_kind, resolved_sha in self.resolutions
            )
        )
        if len({word for word, *_ in resolutions}) != len(resolutions):
            raise ValueError("snapshot compute resolutions contain duplicate words")
        by_word = {item.word_id: item for item in bindings}
        if any(
            word not in by_word
            or by_word[word].artifact_kind != resolved_kind
            or by_word[word].artifact_sha256 != resolved_sha
            for word, _source_kind, _source_sha, resolved_kind, resolved_sha in resolutions
        ) or set(by_word) != {word for word, *_ in resolutions}:
            raise ValueError("snapshot resolutions disagree with primitive bindings")
        object.__setattr__(self, "resolutions", resolutions)

    def to_record(self) -> dict[str, object]:
        return {
            "context_id": self.context_id,
            "format": self.FORMAT,
            "frontier_sha256": self.frontier_sha256,
            "graph_state_sha256": self.graph_state_sha256,
            "language_snapshot_sha256": self.language_snapshot_sha256,
            "primitive_bindings": [
                item.to_record() for item in self.primitive_bindings
            ],
            "resolutions": [
                {
                    "resolved_kind": resolved_kind,
                    "resolved_sha256": resolved_sha,
                    "source_kind": source_kind,
                    "source_sha256": source_sha,
                    "word_id": word,
                }
                for word, source_kind, source_sha, resolved_kind, resolved_sha in self.resolutions
            ],
        }

    def to_bytes(self) -> bytes:
        return _seal(self.FORMAT, self.to_record())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "SnapshotComputeResolutionReceipt":
        body = _open(data, schema=cls.FORMAT, label="snapshot compute resolution")
        expected = {
            "context_id",
            "format",
            "frontier_sha256",
            "graph_state_sha256",
            "language_snapshot_sha256",
            "primitive_bindings",
            "resolutions",
        }
        if set(body) != expected or body.get("format") != cls.FORMAT:
            raise LanguageBridgeIntegrityError(
                "snapshot compute resolution body is invalid"
            )
        raw_bindings = body.get("primitive_bindings")
        raw_resolutions = body.get("resolutions")
        if not isinstance(raw_bindings, list) or not isinstance(raw_resolutions, list):
            raise LanguageBridgeIntegrityError(
                "snapshot compute resolution inventory is invalid"
            )
        try:
            resolutions = []
            for row in raw_resolutions:
                if not isinstance(row, Mapping) or set(row) != {
                    "resolved_kind",
                    "resolved_sha256",
                    "source_kind",
                    "source_sha256",
                    "word_id",
                }:
                    raise ValueError("invalid snapshot resolution row")
                resolutions.append(
                    (
                        cast(str, row.get("word_id")),
                        cast(str, row.get("source_kind")),
                        cast(str, row.get("source_sha256")),
                        cast(str, row.get("resolved_kind")),
                        cast(str, row.get("resolved_sha256")),
                    )
                )
            result = cls(
                language_snapshot_sha256=cast(
                    str, body.get("language_snapshot_sha256")
                ),
                frontier_sha256=cast(str, body.get("frontier_sha256")),
                context_id=cast(str | None, body.get("context_id")),
                graph_state_sha256=cast(str | None, body.get("graph_state_sha256")),
                primitive_bindings=tuple(
                    PrimitiveWordBinding.from_record(row) for row in raw_bindings
                ),
                resolutions=tuple(resolutions),
            )
        except (TypeError, ValueError) as exc:
            raise LanguageBridgeIntegrityError(
                "snapshot compute resolution reconstruction failed"
            ) from exc
        if result.to_bytes() != data:
            raise LanguageBridgeIntegrityError(
                "snapshot compute resolution failed canonical reconstruction"
            )
        return result


def resolve_snapshot_compute_bindings(
    snapshot: LanguageSnapshot,
    frontier: ActionFrontier,
    *,
    context_id: str | None = None,
    graph_state: ComputeOperatorGraphState | None = None,
) -> tuple[tuple[PrimitiveWordBinding, ...], SnapshotComputeResolutionReceipt]:
    """Resolve every stable snapshot word to a local Crystal or program."""

    if not isinstance(snapshot, LanguageSnapshot):
        raise TypeError("snapshot must be a LanguageSnapshot")
    if not isinstance(frontier, ActionFrontier):
        raise TypeError("frontier must be an ActionFrontier")
    if snapshot.frontier_sha256 != frontier.sha256:
        raise LanguageBridgeIntegrityError("snapshot frontier is stale")
    if graph_state is not None and not isinstance(
        graph_state, ComputeOperatorGraphState
    ):
        raise TypeError("graph_state must be a ComputeOperatorGraphState")
    artifacts = snapshot.primitive_artifacts(frontier, context_id=context_id)
    needs_graph = any(
        source_kind == "materialized-route"
        for source_kind, _source_sha in artifacts.values()
    )
    graph_authority = dict(frontier.authority_hashes).get("graph-state")
    if graph_state is not None and graph_authority != graph_state.sha256:
        raise LanguageBridgeIntegrityError(
            "supplied graph state is not authorized by the snapshot frontier"
        )
    if needs_graph and graph_state is None:
        raise LanguageBridgeIntegrityError(
            "materialized-route words require their authorized graph state"
        )
    routes = (
        {}
        if graph_state is None
        else {route.sha256: route for route in graph_state.materialized_routes}
    )
    bindings = []
    resolutions = []
    for word, (source_kind, source_sha) in sorted(artifacts.items()):
        if source_kind in ("crystal", "program"):
            resolved_kind = source_kind
            resolved_sha = source_sha
        elif source_kind == "materialized-route":
            if graph_state is None or source_sha not in routes:
                raise LanguageBridgeIntegrityError(
                    "route word cannot resolve under the supplied graph state"
                )
            resolved_kind = "program"
            resolved_sha = routes[source_sha].executable_program_sha256
        else:
            raise LanguageBridgeError(
                f"snapshot word {word!r} names non-compute kind {source_kind!r}"
            )
        bindings.append(PrimitiveWordBinding(word, resolved_kind, resolved_sha))
        resolutions.append((word, source_kind, source_sha, resolved_kind, resolved_sha))
    if not bindings:
        raise LanguageBridgeError("snapshot has no stable compute words")
    receipt = SnapshotComputeResolutionReceipt(
        language_snapshot_sha256=snapshot.sha256,
        frontier_sha256=frontier.sha256,
        context_id=context_id,
        graph_state_sha256=None if graph_state is None else graph_state.sha256,
        primitive_bindings=tuple(bindings),
        resolutions=tuple(resolutions),
    )
    return receipt.primitive_bindings, receipt


def definition_from_macro_option(
    option: MacroOption,
    snapshot: LanguageSnapshot,
    frontier: ActionFrontier,
    *,
    context_id: str,
    graph_authority_name: str,
) -> ExecutableWordDefinition:
    """Translate one verified MacroOption action program into known words."""

    if not isinstance(option, MacroOption):
        raise TypeError("option must be a MacroOption")
    if not isinstance(snapshot, LanguageSnapshot):
        raise TypeError("snapshot must be a LanguageSnapshot")
    if not isinstance(frontier, ActionFrontier):
        raise TypeError("frontier must be an ActionFrontier")
    authority_name = _identifier(graph_authority_name, field="graph_authority_name")
    if (
        snapshot.frontier_sha256 != frontier.sha256
        or dict(frontier.authority_hashes).get(authority_name)
        != option.identity.graph_revision_sha256
    ):
        raise LanguageBridgeIntegrityError(
            "macro option graph authority differs from the language frontier"
        )
    words = []
    for action_id in option.identity.action_sequence:
        word = snapshot.encode_action(action_id, context_id)
        if word is None:
            raise LanguageBridgeError(
                f"snapshot cannot encode option action {action_id!r}"
            )
        words.append(word)
    return ExecutableWordDefinition.create(
        f"option-{option.sha256[:32]}",
        tuple(words),
        language_snapshot_sha256=snapshot.sha256,
        frontier_sha256=frontier.sha256,
        authority_hashes=frontier.authority_hashes,
    )


__all__ = [
    "ALGEBRA_LANGUAGE_REWARD_POLICY_SHA256",
    "CONTROLLER_CRYSTAL_LANGUAGE_REWARD_POLICY_SHA256",
    "CONTROLLER_CRYSTAL_OUTCOME_SCHEMA",
    "CONTROLLER_CRYSTAL_OUTCOME_VERIFIER_SHA256",
    "DEMAND_LANGUAGE_REWARD_POLICY_SHA256",
    "FRONTIER_MIGRATION_SCHEMA",
    "LANGUAGE_MACRO_DISCOVERY_SCHEMA",
    "LANGUAGE_OUTCOME_COMMIT_SCHEMA",
    "LANGUAGE_ROUTING_DECISION_SCHEMA",
    "LANGUAGE_REVISION_PROMOTION_SCHEMA",
    "LANGUAGE_STATE_NAME",
    "MACRO_ACTION_PROMOTION_SCHEMA",
    "ROUTE_FRONTIER_TRANSITION_SCHEMA",
    "SNAPSHOT_COMPUTE_RESOLUTION_SCHEMA",
    "ROUTE_WORD_RESOLUTION_SCHEMA",
    "VERIFIED_WORD_TRAJECTORY_SCHEMA",
    "ConsequenceLanguageBridge",
    "ConsequenceLanguageStateBank",
    "ControllerCrystalLanguageExecutor",
    "ControllerCrystalOutcomeReceipt",
    "FrontierMigrationReceipt",
    "LanguageBridgeConflictError",
    "LanguageBridgeError",
    "LanguageBridgeIntegrityError",
    "LanguageMacroDiscoveryReceipt",
    "LanguageOutcomeCommitReceipt",
    "LanguageRoutingDecision",
    "LanguageRevisionPromotionReceipt",
    "MacroActionPromotionReceipt",
    "RouteFrontierTransitionReceipt",
    "RouteWordResolutionReceipt",
    "SnapshotComputeResolutionReceipt",
    "VerifiedWordTrajectory",
    "build_algebra_action_frontier",
    "build_route_action_frontier",
    "candidate_action_id",
    "definition_from_macro_option",
    "discover_language_macros",
    "execute_controller_crystal_choice",
    "migrate_language_frontier",
    "promote_compiled_word_action",
    "promote_and_migrate_language_revision",
    "prove_route_frontier_transition",
    "resolve_route_words_to_programs",
    "resolve_snapshot_compute_bindings",
    "route_action_id",
]
