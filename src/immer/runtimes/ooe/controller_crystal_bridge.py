"""Exact publication of promoted controller kernels into the compute substrate.

The bridge does not retrain, smooth, or reinterpret a promoted controller
kernel.  It restores every active :class:`CrystalPayload`, converts its exact
quantized 5x5 transition matrix into a generic ``ComputeCrystal.markov``
artifact, publishes the artifacts sequentially, and seals the resulting
``ActionFrontier`` against both the source controller state and the final
``ComputeCrystalBank`` anchor.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from typing import Any, cast

import numpy as np

from immer.runtimes.qwen3_8.semantic_atlas import GraphRevision

from .compute_crystals import (
    ComputeBankManifest,
    ComputeCrystal,
    ComputeCrystalBank,
    ComputeCrystalError,
)
from .controller import (
    CONTROLLER_STATE_NAME,
    CONTROLLER_STATE_SCHEMA,
    OoeController,
    OoeControllerIntegrityError,
)
from .crystal import (
    CrystalManifest,
    CrystalPayload,
    CrystalStoreError,
    ManifestConflictError,
)
from .identity import OoeSiteIdentity, canonical_json_bytes, require_sha256
from .markov_language import ActionBinding, ActionFrontier
from .qwen_bridge import ACTION_SCHEMA_SHA256, OOE_ACTIONS


CONTROLLER_CRYSTAL_SOURCE_EXTENSION = "immer.controller-crystal-source"
CONTROLLER_CRYSTAL_SOURCE_SCHEMA = "immer-ooe-controller-crystal-source/v1"
CONTROLLER_CRYSTAL_EXPORT_SCHEMA = "immer-ooe-controller-crystal-export/v1"
CONTROLLER_CRYSTAL_ACTION_SCHEMA = "immer-ooe-controller-crystal-actions/v1"
CONTROLLER_CRYSTAL_CONTEXT_SCHEMA = {
    "action_vector": list(OOE_ACTIONS),
    "dtype": "float64",
    "format": "immer-ooe-controller-crystal-language-context/v1",
    "operator": "row-markov-transition",
    "source_action_schema_sha256": ACTION_SCHEMA_SHA256,
}
CONTROLLER_CRYSTAL_CONTEXT_SCHEMA_SHA256 = hashlib.sha256(
    canonical_json_bytes(CONTROLLER_CRYSTAL_CONTEXT_SCHEMA)
).hexdigest()
CONTROLLER_CRYSTAL_LANGUAGE_REWARD_POLICY = {
    "failure": -1,
    "format": "immer-ooe-controller-crystal-language-reward/v1",
    "success": 1,
}
CONTROLLER_CRYSTAL_LANGUAGE_REWARD_POLICY_SHA256 = hashlib.sha256(
    canonical_json_bytes(CONTROLLER_CRYSTAL_LANGUAGE_REWARD_POLICY)
).hexdigest()
CONTROLLER_CRYSTAL_EXPORT_STATE_PREFIX = "ooe-controller-crystal-export/v1:"
CONTROLLER_CRYSTAL_TRANSACTION_SCHEMA = (
    "immer-ooe-controller-crystal-export-transaction/v1"
)
CONTROLLER_CRYSTAL_TRANSACTION_STATE_PREFIX = (
    "ooe-controller-crystal-export-transaction/v1:"
)
MAX_CONTROLLER_CRYSTAL_EXPORT_BYTES = 256 * 1024 * 1024

_AUTHORITY_NAMES = frozenset(
    {
        "compute-bank-anchor",
        "controller-atlas-graph",
        "controller-calibration",
        "controller-crystal-manifest",
        "controller-model-pin",
        "controller-state",
        "controller-weight-graph",
        "language-reward-policy",
    }
)


class ControllerCrystalBridgeError(RuntimeError):
    """The controller-to-compute publication contract cannot be satisfied."""


class ControllerCrystalBridgeIntegrityError(ControllerCrystalBridgeError):
    """A source, publication, frontier, or receipt failed exact reconstruction."""


class ControllerCrystalBridgeStaleError(ControllerCrystalBridgeIntegrityError):
    """A caller pin or a concurrently observed source state is stale."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: object) -> str:
    return _sha256_bytes(canonical_json_bytes(value))


def _strict_json(data: bytes, *, label: str, maximum: int) -> object:
    if not isinstance(data, bytes):
        raise TypeError(f"{label} must be immutable bytes")
    if len(data) > maximum:
        raise ControllerCrystalBridgeIntegrityError(
            f"{label} exceeds its hard byte limit"
        )

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ControllerCrystalBridgeIntegrityError(
            f"{label} is invalid canonical JSON"
        ) from exc
    if canonical_json_bytes(value) != data:
        raise ControllerCrystalBridgeIntegrityError(f"{label} is not canonical JSON")
    return value


def _decode_bound_bytes(
    value: object,
    *,
    label: str,
    maximum: int,
) -> bytes:
    if not isinstance(value, Mapping) or set(value) != {
        "bytes",
        "data_base64",
        "sha256",
    }:
        raise ControllerCrystalBridgeIntegrityError(
            f"{label} byte descriptor is invalid"
        )
    byte_count = value.get("bytes")
    encoded = value.get("data_base64")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 1
        or byte_count > maximum
        or not isinstance(encoded, str)
        or not encoded.isascii()
        or len(encoded) != 4 * ((byte_count + 2) // 3)
    ):
        raise ControllerCrystalBridgeIntegrityError(
            f"{label} byte descriptor is invalid"
        )
    try:
        data = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise ControllerCrystalBridgeIntegrityError(
            f"{label} base64 payload is invalid"
        ) from exc
    try:
        claimed = require_sha256(value.get("sha256"), field=f"{label}.sha256")
    except ValueError as exc:
        raise ControllerCrystalBridgeIntegrityError(
            f"{label} SHA-256 is invalid"
        ) from exc
    if (
        len(data) != byte_count
        or base64.b64encode(data).decode("ascii") != encoded
        or _sha256_bytes(data) != claimed
    ):
        raise ControllerCrystalBridgeIntegrityError(
            f"{label} bytes do not match their descriptor"
        )
    return data


def _bound_bytes(data: bytes) -> dict[str, object]:
    return {
        "bytes": len(data),
        "data_base64": base64.b64encode(data).decode("ascii"),
        "sha256": _sha256_bytes(data),
    }


def _snapshot_body(snapshot: bytes) -> dict[str, Any]:
    document = _strict_json(
        snapshot,
        label="controller snapshot",
        maximum=MAX_CONTROLLER_CRYSTAL_EXPORT_BYTES,
    )
    if (
        not isinstance(document, Mapping)
        or set(document) != {"body", "schema", "sha256"}
        or document.get("schema") != CONTROLLER_STATE_SCHEMA
    ):
        raise ControllerCrystalBridgeIntegrityError(
            "controller snapshot envelope is invalid"
        )
    body = document.get("body")
    if not isinstance(body, Mapping):
        raise ControllerCrystalBridgeIntegrityError(
            "controller snapshot body is invalid"
        )
    try:
        claimed = require_sha256(document.get("sha256"), field="snapshot sha256")
    except ValueError as exc:
        raise ControllerCrystalBridgeIntegrityError(
            "controller snapshot seal is invalid"
        ) from exc
    if claimed != _digest(body):
        raise ControllerCrystalBridgeIntegrityError(
            "controller snapshot seal does not match its body"
        )
    return cast(dict[str, Any], dict(body))


def controller_site_action_id(site_identity_sha256: str) -> str:
    """Return the one canonical language action identifier for a site."""

    return "controller-site:" + require_sha256(
        site_identity_sha256,
        field="site_identity_sha256",
    )


def _hash_pairs(
    value: Mapping[str, str] | Sequence[tuple[str, str]],
    *,
    field: str,
) -> tuple[tuple[str, str], ...]:
    try:
        items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    except TypeError as exc:
        raise ValueError(f"{field} must contain named SHA-256 values") from exc
    normalized: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{field} entries must be name/digest pairs")
        name, digest = item
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or "\x00" in name
        ):
            raise ValueError(f"{field} names must be canonical non-empty text")
        normalized.append((name, require_sha256(digest, field=f"{field}[{name!r}]")))
    result = tuple(sorted(normalized))
    if not result or len({name for name, _ in result}) != len(result):
        raise ValueError(f"{field} must be non-empty with unique names")
    return result


@dataclass(frozen=True, slots=True)
class ControllerCrystalSiteReceipt:
    """One exact source payload and its sequential compute-bank publication."""

    ordinal: int
    action_id: str
    site_identity: OoeSiteIdentity
    source_payload_sha256: str
    coverage_sha256: str
    calibration_sha256: str
    verifier_hashes: tuple[tuple[str, str], ...]
    evidence_hashes: tuple[tuple[str, str], ...]
    consensus_receipt_sha256: str
    quantized_kernel_sha256: str
    compute_crystal_sha256: str
    bank_generation_before: int
    bank_generation_after: int
    bank_anchor_before: str
    bank_anchor_after: str
    manifest_changed: bool
    object_created: bool | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
        ):
            raise ValueError("site receipt ordinal must be non-negative")
        if not isinstance(self.site_identity, OoeSiteIdentity):
            raise TypeError("site_identity must be an OoeSiteIdentity")
        expected_action = controller_site_action_id(self.site_identity.sha256)
        if self.action_id != expected_action:
            raise ValueError("site receipt action ID does not match its identity")
        for field_name in (
            "source_payload_sha256",
            "coverage_sha256",
            "calibration_sha256",
            "consensus_receipt_sha256",
            "quantized_kernel_sha256",
            "compute_crystal_sha256",
            "bank_anchor_before",
            "bank_anchor_after",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        object.__setattr__(
            self,
            "verifier_hashes",
            _hash_pairs(self.verifier_hashes, field="verifier_hashes"),
        )
        object.__setattr__(
            self,
            "evidence_hashes",
            _hash_pairs(self.evidence_hashes, field="evidence_hashes"),
        )
        for field_name in ("bank_generation_before", "bank_generation_after"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if not isinstance(self.manifest_changed, bool):
            raise TypeError("manifest_changed must be bool")
        if self.object_created is not None and not isinstance(
            self.object_created, bool
        ):
            raise TypeError("object_created must be bool or None")
        expected_after = self.bank_generation_before + int(self.manifest_changed)
        if self.bank_generation_after != expected_after:
            raise ValueError("site receipt bank generation is not sequential")
        if self.manifest_changed == (self.bank_anchor_before == self.bank_anchor_after):
            raise ValueError("site receipt bank anchor change is inconsistent")

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "action_id": self.action_id,
            "bank_anchor_after": self.bank_anchor_after,
            "bank_anchor_before": self.bank_anchor_before,
            "bank_generation_after": self.bank_generation_after,
            "bank_generation_before": self.bank_generation_before,
            "calibration_sha256": self.calibration_sha256,
            "compute_crystal_sha256": self.compute_crystal_sha256,
            "consensus_receipt_sha256": self.consensus_receipt_sha256,
            "coverage_sha256": self.coverage_sha256,
            "evidence_hashes": dict(self.evidence_hashes),
            "manifest_changed": self.manifest_changed,
            "ordinal": self.ordinal,
            "quantized_kernel_sha256": self.quantized_kernel_sha256,
            "site_identity": self.site_identity.to_dict(),
            "source_payload_sha256": self.source_payload_sha256,
            "verifier_hashes": dict(self.verifier_hashes),
        }
        if self.object_created is not None:
            record["object_created"] = self.object_created
        return record

    @classmethod
    def from_record(cls, value: object) -> "ControllerCrystalSiteReceipt":
        fields = {
            "action_id",
            "bank_anchor_after",
            "bank_anchor_before",
            "bank_generation_after",
            "bank_generation_before",
            "calibration_sha256",
            "compute_crystal_sha256",
            "consensus_receipt_sha256",
            "coverage_sha256",
            "evidence_hashes",
            "manifest_changed",
            "ordinal",
            "quantized_kernel_sha256",
            "site_identity",
            "source_payload_sha256",
            "verifier_hashes",
        }
        actual_fields = set(value) if isinstance(value, Mapping) else set()
        if not isinstance(value, Mapping) or actual_fields not in (
            fields,
            fields | {"object_created"},
        ):
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal site receipt is invalid"
            )
        verifier_hashes = value.get("verifier_hashes")
        evidence_hashes = value.get("evidence_hashes")
        if not isinstance(verifier_hashes, Mapping) or not isinstance(
            evidence_hashes, Mapping
        ):
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal site hashes are invalid"
            )
        try:
            return cls(
                ordinal=cast(int, value.get("ordinal")),
                action_id=cast(str, value.get("action_id")),
                site_identity=OoeSiteIdentity.from_dict(value.get("site_identity")),
                source_payload_sha256=cast(str, value.get("source_payload_sha256")),
                coverage_sha256=cast(str, value.get("coverage_sha256")),
                calibration_sha256=cast(str, value.get("calibration_sha256")),
                verifier_hashes=_hash_pairs(
                    cast(Mapping[str, str], verifier_hashes),
                    field="verifier_hashes",
                ),
                evidence_hashes=_hash_pairs(
                    cast(Mapping[str, str], evidence_hashes),
                    field="evidence_hashes",
                ),
                consensus_receipt_sha256=cast(
                    str, value.get("consensus_receipt_sha256")
                ),
                quantized_kernel_sha256=cast(str, value.get("quantized_kernel_sha256")),
                compute_crystal_sha256=cast(str, value.get("compute_crystal_sha256")),
                bank_generation_before=cast(int, value.get("bank_generation_before")),
                bank_generation_after=cast(int, value.get("bank_generation_after")),
                bank_anchor_before=cast(str, value.get("bank_anchor_before")),
                bank_anchor_after=cast(str, value.get("bank_anchor_after")),
                manifest_changed=cast(bool, value.get("manifest_changed")),
                object_created=(
                    cast(bool, value.get("object_created"))
                    if "object_created" in value
                    else None
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal site receipt failed validation"
            ) from exc


@dataclass(frozen=True, slots=True)
class ControllerCrystalExportTransaction:
    """Crash-recoverable publication intent and its committed site prefix."""

    controller_state_name: str
    controller_snapshot_sha256: str
    source_manifest_sha256: str
    source_payload_sha256s: tuple[str, ...]
    compute_crystal_sha256s: tuple[str, ...]
    start_bank_manifest: bytes
    completed_sites: tuple[ControllerCrystalSiteReceipt, ...] = ()
    committed_receipt_sha256: str | None = None

    FORMAT = CONTROLLER_CRYSTAL_TRANSACTION_SCHEMA

    def __post_init__(self) -> None:
        if (
            not isinstance(self.controller_state_name, str)
            or not self.controller_state_name
            or self.controller_state_name != self.controller_state_name.strip()
            or "\x00" in self.controller_state_name
        ):
            raise ValueError("transaction controller state name is invalid")
        for field_name in (
            "controller_snapshot_sha256",
            "source_manifest_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        sources = tuple(
            require_sha256(value, field="source_payload_sha256s")
            for value in self.source_payload_sha256s
        )
        crystals = tuple(
            require_sha256(value, field="compute_crystal_sha256s")
            for value in self.compute_crystal_sha256s
        )
        if not sources or len(sources) != len(crystals):
            raise ValueError("transaction plan inventories must be equal and non-empty")
        if len(set(sources)) != len(sources) or len(set(crystals)) != len(crystals):
            raise ValueError("transaction plan inventories must be unique")
        object.__setattr__(self, "source_payload_sha256s", sources)
        object.__setattr__(self, "compute_crystal_sha256s", crystals)
        if not isinstance(self.start_bank_manifest, bytes):
            raise TypeError("start_bank_manifest must be immutable bytes")
        try:
            start = ComputeBankManifest.from_bytes(self.start_bank_manifest)
        except (TypeError, ValueError, ComputeCrystalError) as exc:
            raise ControllerCrystalBridgeIntegrityError(
                "transaction start bank manifest is invalid"
            ) from exc
        completed = tuple(self.completed_sites)
        if len(completed) > len(sources) or any(
            not isinstance(site, ControllerCrystalSiteReceipt) for site in completed
        ):
            raise ValueError("transaction completed-site prefix is invalid")
        previous_generation = start.generation
        previous_anchor = start.sha256
        for ordinal, site in enumerate(completed):
            if (
                site.ordinal != ordinal
                or site.source_payload_sha256 != sources[ordinal]
                or site.compute_crystal_sha256 != crystals[ordinal]
                or site.bank_generation_before != previous_generation
                or site.bank_anchor_before != previous_anchor
            ):
                raise ControllerCrystalBridgeIntegrityError(
                    "transaction completed-site prefix is discontinuous"
                )
            previous_generation = site.bank_generation_after
            previous_anchor = site.bank_anchor_after
        object.__setattr__(self, "completed_sites", completed)
        committed = self.committed_receipt_sha256
        if committed is not None:
            committed = require_sha256(committed, field="committed_receipt_sha256")
            if len(completed) != len(sources):
                raise ValueError("committed transaction has an incomplete site prefix")
        object.__setattr__(self, "committed_receipt_sha256", committed)

    @property
    def plan_record(self) -> dict[str, object]:
        return {
            "compute_crystal_sha256s": list(self.compute_crystal_sha256s),
            "controller_snapshot_sha256": self.controller_snapshot_sha256,
            "controller_state_name": self.controller_state_name,
            "format": "immer-ooe-controller-crystal-export-plan/v1",
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_payload_sha256s": list(self.source_payload_sha256s),
        }

    @property
    def plan_sha256(self) -> str:
        return _digest(self.plan_record)

    @property
    def state_name(self) -> str:
        return CONTROLLER_CRYSTAL_TRANSACTION_STATE_PREFIX + self.plan_sha256

    @property
    def start_manifest(self) -> ComputeBankManifest:
        return ComputeBankManifest.from_bytes(self.start_bank_manifest)

    @property
    def status(self) -> str:
        return "committed" if self.committed_receipt_sha256 is not None else "prepared"

    def to_document(self) -> dict[str, object]:
        body = {
            "committed_receipt_sha256": self.committed_receipt_sha256,
            "completed_sites": [site.to_record() for site in self.completed_sites],
            "plan": self.plan_record,
            "plan_sha256": self.plan_sha256,
            "start_bank_manifest": _bound_bytes(self.start_bank_manifest),
            "status": self.status,
        }
        return {
            "body": body,
            "body_sha256": _digest(body),
            "schema": self.FORMAT,
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_CONTROLLER_CRYSTAL_EXPORT_BYTES:
            raise ValueError("controller-crystal transaction exceeds its hard limit")
        return data

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.to_bytes())

    def append(
        self, site: ControllerCrystalSiteReceipt
    ) -> "ControllerCrystalExportTransaction":
        if self.committed_receipt_sha256 is not None:
            raise ControllerCrystalBridgeIntegrityError(
                "cannot append to a committed export transaction"
            )
        return ControllerCrystalExportTransaction(
            controller_state_name=self.controller_state_name,
            controller_snapshot_sha256=self.controller_snapshot_sha256,
            source_manifest_sha256=self.source_manifest_sha256,
            source_payload_sha256s=self.source_payload_sha256s,
            compute_crystal_sha256s=self.compute_crystal_sha256s,
            start_bank_manifest=self.start_bank_manifest,
            completed_sites=(*self.completed_sites, site),
        )

    def commit(self, receipt_sha256: str) -> "ControllerCrystalExportTransaction":
        if len(self.completed_sites) != len(self.source_payload_sha256s):
            raise ControllerCrystalBridgeIntegrityError(
                "cannot commit an incomplete export transaction"
            )
        return ControllerCrystalExportTransaction(
            controller_state_name=self.controller_state_name,
            controller_snapshot_sha256=self.controller_snapshot_sha256,
            source_manifest_sha256=self.source_manifest_sha256,
            source_payload_sha256s=self.source_payload_sha256s,
            compute_crystal_sha256s=self.compute_crystal_sha256s,
            start_bank_manifest=self.start_bank_manifest,
            completed_sites=self.completed_sites,
            committed_receipt_sha256=require_sha256(
                receipt_sha256, field="receipt_sha256"
            ),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "ControllerCrystalExportTransaction":
        document = _strict_json(
            data,
            label="controller-crystal export transaction",
            maximum=MAX_CONTROLLER_CRYSTAL_EXPORT_BYTES,
        )
        if (
            not isinstance(document, Mapping)
            or set(document) != {"body", "body_sha256", "schema"}
            or document.get("schema") != cls.FORMAT
        ):
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal transaction envelope is invalid"
            )
        body = document.get("body")
        if not isinstance(body, Mapping) or set(body) != {
            "committed_receipt_sha256",
            "completed_sites",
            "plan",
            "plan_sha256",
            "start_bank_manifest",
            "status",
        }:
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal transaction body is invalid"
            )
        if document.get("body_sha256") != _digest(body):
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal transaction body SHA-256 mismatch"
            )
        plan = body.get("plan")
        completed = body.get("completed_sites")
        if (
            not isinstance(plan, Mapping)
            or set(plan)
            != {
                "compute_crystal_sha256s",
                "controller_snapshot_sha256",
                "controller_state_name",
                "format",
                "source_manifest_sha256",
                "source_payload_sha256s",
            }
            or plan.get("format") != "immer-ooe-controller-crystal-export-plan/v1"
        ):
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal transaction plan is invalid"
            )
        sources = plan.get("source_payload_sha256s")
        crystals = plan.get("compute_crystal_sha256s")
        if (
            not isinstance(sources, list)
            or not isinstance(crystals, list)
            or not isinstance(completed, list)
        ):
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal transaction inventories are invalid"
            )
        try:
            transaction = cls(
                controller_state_name=cast(str, plan.get("controller_state_name")),
                controller_snapshot_sha256=cast(
                    str, plan.get("controller_snapshot_sha256")
                ),
                source_manifest_sha256=cast(str, plan.get("source_manifest_sha256")),
                source_payload_sha256s=tuple(sources),
                compute_crystal_sha256s=tuple(crystals),
                start_bank_manifest=_decode_bound_bytes(
                    body.get("start_bank_manifest"),
                    label="transaction start bank manifest",
                    maximum=MAX_CONTROLLER_CRYSTAL_EXPORT_BYTES,
                ),
                completed_sites=tuple(
                    ControllerCrystalSiteReceipt.from_record(site) for site in completed
                ),
                committed_receipt_sha256=cast(
                    str | None, body.get("committed_receipt_sha256")
                ),
            )
        except ControllerCrystalBridgeIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal transaction failed reconstruction"
            ) from exc
        if (
            body.get("plan_sha256") != transaction.plan_sha256
            or body.get("status") != transaction.status
            or transaction.to_bytes() != data
        ):
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal transaction failed canonical reconstruction"
            )
        return transaction


def _action_schema_sha256(
    sites: Sequence[ControllerCrystalSiteReceipt],
) -> str:
    return _digest(
        {
            "actions": [
                {
                    "action_id": site.action_id,
                    "compute_crystal_sha256": site.compute_crystal_sha256,
                    "site_identity_sha256": site.site_identity.sha256,
                    "source_payload_sha256": site.source_payload_sha256,
                }
                for site in sites
            ],
            "format": CONTROLLER_CRYSTAL_ACTION_SCHEMA,
            "source_action_schema_sha256": ACTION_SCHEMA_SHA256,
        }
    )


def _authorities(
    *,
    controller_snapshot_sha256: str,
    model_pin_sha256: str,
    weight_graph_revision_sha256: str,
    atlas_graph_revision_sha256: str,
    calibration_sha256: str,
    source_manifest_sha256: str,
    final_bank_anchor_sha256: str,
) -> dict[str, str]:
    return {
        "compute-bank-anchor": final_bank_anchor_sha256,
        "controller-atlas-graph": atlas_graph_revision_sha256,
        "controller-calibration": calibration_sha256,
        "controller-crystal-manifest": source_manifest_sha256,
        "controller-model-pin": model_pin_sha256,
        "controller-state": controller_snapshot_sha256,
        "controller-weight-graph": weight_graph_revision_sha256,
        "language-reward-policy": (CONTROLLER_CRYSTAL_LANGUAGE_REWARD_POLICY_SHA256),
    }


def _frontier(
    sites: Sequence[ControllerCrystalSiteReceipt],
    *,
    authorities: Mapping[str, str],
) -> ActionFrontier:
    return ActionFrontier.create(
        tuple(
            ActionBinding(
                action_id=site.action_id,
                artifact_kind="crystal",
                artifact_sha256=site.compute_crystal_sha256,
            )
            for site in sites
        ),
        action_schema_sha256=_action_schema_sha256(sites),
        context_schema_sha256=CONTROLLER_CRYSTAL_CONTEXT_SCHEMA_SHA256,
        authority_hashes=authorities,
    )


def _source_extension(
    *,
    controller_state_name: str,
    controller_snapshot_sha256: str,
    model_pin_sha256: str,
    weight_graph_revision_sha256: str,
    atlas_graph_revision: GraphRevision,
    calibration_sha256: str,
    source_manifest: CrystalManifest,
    site: ControllerCrystalSiteReceipt | None,
    payload: CrystalPayload,
    action_id: str,
) -> dict[str, object]:
    site_identity = payload.identity if site is None else site.site_identity
    source_payload_sha256 = (
        payload.sha256 if site is None else site.source_payload_sha256
    )
    coverage_sha256 = payload.coverage_sha256 if site is None else site.coverage_sha256
    consensus_sha256 = (
        payload.consensus_receipt_sha256
        if site is None
        else site.consensus_receipt_sha256
    )
    quantized_sha256 = (
        _sha256_bytes(payload.quantized_kernel)
        if site is None
        else site.quantized_kernel_sha256
    )
    verifier_hashes = payload.verifier_hashes if site is None else site.verifier_hashes
    evidence_hashes = payload.evidence_hashes if site is None else site.evidence_hashes
    return {
        CONTROLLER_CRYSTAL_SOURCE_EXTENSION: {
            "action_id": action_id,
            "atlas_graph_revision": atlas_graph_revision.to_document(),
            "atlas_graph_revision_sha256": atlas_graph_revision.sha256,
            "calibration_sha256": calibration_sha256,
            "consensus_receipt_sha256": consensus_sha256,
            "context_schema_sha256": CONTROLLER_CRYSTAL_CONTEXT_SCHEMA_SHA256,
            "controller_snapshot_sha256": controller_snapshot_sha256,
            "controller_state_name": controller_state_name,
            "coverage_sha256": coverage_sha256,
            "evidence_hashes": dict(evidence_hashes),
            "evidence_set_sha256": _digest(dict(evidence_hashes)),
            "format": CONTROLLER_CRYSTAL_SOURCE_SCHEMA,
            "language_reward_policy_sha256": (
                CONTROLLER_CRYSTAL_LANGUAGE_REWARD_POLICY_SHA256
            ),
            "model_pin_sha256": model_pin_sha256,
            "quantized_kernel": {
                "columns": payload.kernel_columns,
                "levels": payload.quantization_levels,
                "rows": payload.kernel_rows,
                "sha256": quantized_sha256,
                "storage": "uint16-le",
            },
            "site_identity": site_identity.to_dict(),
            "site_identity_sha256": site_identity.sha256,
            "source_manifest_generation": source_manifest.generation,
            "source_manifest_sha256": source_manifest.sha256,
            "source_payload_sha256": source_payload_sha256,
            "verifier_hashes": dict(verifier_hashes),
            "verifier_set_sha256": _digest(dict(verifier_hashes)),
            "weight_graph_revision_sha256": weight_graph_revision_sha256,
        }
    }


@dataclass(frozen=True, slots=True)
class ControllerCrystalExportReceipt:
    """Self-contained controller snapshot, source manifest, and frontier seal."""

    controller_state_name: str
    controller_snapshot: bytes
    model_pin_sha256: str
    weight_graph_revision_sha256: str
    atlas_graph_revision: GraphRevision
    calibration_sha256: str
    source_manifest: bytes
    start_bank_generation: int
    start_bank_anchor_sha256: str
    final_bank_generation: int
    final_bank_anchor_sha256: str
    sites: tuple[ControllerCrystalSiteReceipt, ...]
    frontier: ActionFrontier
    transaction_plan_sha256: str | None = None

    FORMAT = CONTROLLER_CRYSTAL_EXPORT_SCHEMA

    def __post_init__(self) -> None:
        if (
            not isinstance(self.controller_state_name, str)
            or not self.controller_state_name
            or self.controller_state_name != self.controller_state_name.strip()
            or "\x00" in self.controller_state_name
        ):
            raise ValueError("controller_state_name must be canonical non-empty text")
        if not isinstance(self.controller_snapshot, bytes) or not isinstance(
            self.source_manifest, bytes
        ):
            raise TypeError("bound source payloads must be immutable bytes")
        body = _snapshot_body(self.controller_snapshot)
        try:
            manifest = CrystalManifest.from_bytes(self.source_manifest)
        except (TypeError, ValueError, CrystalStoreError) as exc:
            raise ControllerCrystalBridgeIntegrityError(
                "bound CrystalStore manifest is invalid"
            ) from exc
        for field_name in (
            "model_pin_sha256",
            "weight_graph_revision_sha256",
            "calibration_sha256",
            "start_bank_anchor_sha256",
            "final_bank_anchor_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                require_sha256(getattr(self, field_name), field=field_name),
            )
        if not isinstance(self.atlas_graph_revision, GraphRevision):
            raise TypeError("atlas_graph_revision must be a GraphRevision")
        for field_name in ("start_bank_generation", "final_bank_generation"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        sites = tuple(self.sites)
        if not sites or any(
            not isinstance(site, ControllerCrystalSiteReceipt) for site in sites
        ):
            raise ValueError("export receipt needs at least one valid site")
        if tuple(sorted(sites, key=lambda item: item.action_id)) != sites:
            raise ValueError("export receipt sites must be sorted by action ID")
        if tuple(site.ordinal for site in sites) != tuple(range(len(sites))):
            raise ValueError("export receipt site ordinals are not contiguous")
        if len({site.action_id for site in sites}) != len(sites):
            raise ValueError("export receipt contains duplicate sites")
        previous_anchor = self.start_bank_anchor_sha256
        previous_generation = self.start_bank_generation
        for site in sites:
            if (
                site.bank_anchor_before != previous_anchor
                or site.bank_generation_before != previous_generation
            ):
                raise ValueError("export publication sequence is discontinuous")
            previous_anchor = site.bank_anchor_after
            previous_generation = site.bank_generation_after
        if (
            previous_anchor != self.final_bank_anchor_sha256
            or previous_generation != self.final_bank_generation
        ):
            raise ValueError("export final bank anchor is not the sequence head")
        object.__setattr__(self, "sites", sites)
        if not isinstance(self.frontier, ActionFrontier):
            raise TypeError("frontier must be an ActionFrontier")

        snapshot_sha256 = _sha256_bytes(self.controller_snapshot)
        if body.get("model_pin_sha256") != self.model_pin_sha256:
            raise ControllerCrystalBridgeIntegrityError(
                "receipt model pin differs from its controller snapshot"
            )
        if (
            body.get("weight_graph_revision_sha256")
            != self.weight_graph_revision_sha256
        ):
            raise ControllerCrystalBridgeIntegrityError(
                "receipt weight graph differs from its controller snapshot"
            )
        try:
            snapshot_atlas = GraphRevision.from_document(
                body.get("atlas_current_revision")
            )
        except (TypeError, ValueError) as exc:
            raise ControllerCrystalBridgeIntegrityError(
                "controller snapshot Atlas revision is invalid"
            ) from exc
        if snapshot_atlas != self.atlas_graph_revision:
            raise ControllerCrystalBridgeIntegrityError(
                "receipt Atlas graph differs from its controller snapshot"
            )
        router = body.get("router")
        if (
            not isinstance(router, Mapping)
            or router.get("calibration_sha256") != self.calibration_sha256
        ):
            raise ControllerCrystalBridgeIntegrityError(
                "receipt calibration differs from its controller snapshot"
            )
        if (
            body.get("crystal_manifest_sha256") != manifest.sha256
            or body.get("crystal_manifest_generation") != manifest.generation
            or canonical_json_bytes(body.get("crystal_manifest")) != manifest.to_bytes()
        ):
            raise ControllerCrystalBridgeIntegrityError(
                "receipt manifest differs from its controller snapshot"
            )
        snapshot_sites = body.get("sites")
        if not isinstance(snapshot_sites, list):
            raise ControllerCrystalBridgeIntegrityError(
                "controller snapshot site inventory is invalid"
            )
        expected_sources: dict[str, str] = {}
        for row in snapshot_sites:
            if not isinstance(row, Mapping):
                raise ControllerCrystalBridgeIntegrityError(
                    "controller snapshot site inventory is invalid"
                )
            site_sha256 = row.get("site_identity_sha256")
            payload_sha256 = row.get("crystal_sha256")
            if not isinstance(site_sha256, str) or not isinstance(payload_sha256, str):
                raise ControllerCrystalBridgeIntegrityError(
                    "controller snapshot contains an unpromoted site"
                )
            expected_sources[site_sha256] = payload_sha256
        observed_sources = {
            site.site_identity.sha256: site.source_payload_sha256 for site in sites
        }
        if observed_sources != expected_sources:
            raise ControllerCrystalBridgeIntegrityError(
                "export sites differ from the controller snapshot"
            )
        authorities = _authorities(
            controller_snapshot_sha256=snapshot_sha256,
            model_pin_sha256=self.model_pin_sha256,
            weight_graph_revision_sha256=self.weight_graph_revision_sha256,
            atlas_graph_revision_sha256=self.atlas_graph_revision.sha256,
            calibration_sha256=self.calibration_sha256,
            source_manifest_sha256=manifest.sha256,
            final_bank_anchor_sha256=self.final_bank_anchor_sha256,
        )
        if set(authorities) != _AUTHORITY_NAMES:
            raise AssertionError("controller authority inventory changed")
        expected_frontier = _frontier(sites, authorities=authorities)
        if expected_frontier != self.frontier:
            raise ControllerCrystalBridgeIntegrityError(
                "frontier does not match its controller export"
            )
        derived_plan_sha256 = _digest(
            {
                "compute_crystal_sha256s": [
                    site.compute_crystal_sha256 for site in sites
                ],
                "controller_snapshot_sha256": snapshot_sha256,
                "controller_state_name": self.controller_state_name,
                "format": "immer-ooe-controller-crystal-export-plan/v1",
                "source_manifest_sha256": manifest.sha256,
                "source_payload_sha256s": [
                    site.source_payload_sha256 for site in sites
                ],
            }
        )
        legacy_sites = tuple(site.object_created is not None for site in sites)
        if self.transaction_plan_sha256 is None:
            if not all(legacy_sites):
                raise ControllerCrystalBridgeIntegrityError(
                    "new export receipt requires transaction metadata"
                )
        else:
            plan_sha256 = require_sha256(
                self.transaction_plan_sha256,
                field="transaction_plan_sha256",
            )
            if plan_sha256 != derived_plan_sha256 or any(legacy_sites):
                raise ControllerCrystalBridgeIntegrityError(
                    "transaction metadata does not match the new export receipt"
                )
            object.__setattr__(self, "transaction_plan_sha256", plan_sha256)

    @property
    def controller_snapshot_sha256(self) -> str:
        return _sha256_bytes(self.controller_snapshot)

    @property
    def source_manifest_sha256(self) -> str:
        return _sha256_bytes(self.source_manifest)

    @property
    def derived_transaction_plan_sha256(self) -> str:
        return _digest(
            {
                "compute_crystal_sha256s": [
                    site.compute_crystal_sha256 for site in self.sites
                ],
                "controller_snapshot_sha256": self.controller_snapshot_sha256,
                "controller_state_name": self.controller_state_name,
                "format": "immer-ooe-controller-crystal-export-plan/v1",
                "source_manifest_sha256": self.source_manifest_sha256,
                "source_payload_sha256s": [
                    site.source_payload_sha256 for site in self.sites
                ],
            }
        )

    @property
    def transaction_state_name(self) -> str | None:
        if self.transaction_plan_sha256 is None:
            return None
        return (
            CONTROLLER_CRYSTAL_TRANSACTION_STATE_PREFIX + self.transaction_plan_sha256
        )

    @property
    def is_legacy(self) -> bool:
        return self.transaction_plan_sha256 is None

    def to_document(self) -> dict[str, object]:
        body = {
            "atlas_graph_revision": self.atlas_graph_revision.to_document(),
            "calibration_sha256": self.calibration_sha256,
            "controller_snapshot": _bound_bytes(self.controller_snapshot),
            "controller_state_name": self.controller_state_name,
            "final_bank_anchor_sha256": self.final_bank_anchor_sha256,
            "final_bank_generation": self.final_bank_generation,
            "frontier": self.frontier.to_record(),
            "model_pin_sha256": self.model_pin_sha256,
            "reward_policy": CONTROLLER_CRYSTAL_LANGUAGE_REWARD_POLICY,
            "sites": [site.to_record() for site in self.sites],
            "source_manifest": _bound_bytes(self.source_manifest),
            "start_bank_anchor_sha256": self.start_bank_anchor_sha256,
            "start_bank_generation": self.start_bank_generation,
            "weight_graph_revision_sha256": self.weight_graph_revision_sha256,
        }
        if self.transaction_plan_sha256 is not None:
            body["transaction_plan_sha256"] = self.transaction_plan_sha256
        return {
            "body": body,
            "body_sha256": _digest(body),
            "schema": self.FORMAT,
        }

    def to_bytes(self) -> bytes:
        data = canonical_json_bytes(self.to_document())
        if len(data) > MAX_CONTROLLER_CRYSTAL_EXPORT_BYTES:
            raise ValueError("controller-crystal export exceeds its hard byte limit")
        return data

    @property
    def sha256(self) -> str:
        return _sha256_bytes(self.to_bytes())

    @property
    def state_name(self) -> str:
        return CONTROLLER_CRYSTAL_EXPORT_STATE_PREFIX + self.sha256

    @classmethod
    def from_bytes(cls, data: bytes) -> "ControllerCrystalExportReceipt":
        document = _strict_json(
            data,
            label="controller-crystal export receipt",
            maximum=MAX_CONTROLLER_CRYSTAL_EXPORT_BYTES,
        )
        if (
            not isinstance(document, Mapping)
            or set(document) != {"body", "body_sha256", "schema"}
            or document.get("schema") != cls.FORMAT
        ):
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal export envelope is invalid"
            )
        body = document.get("body")
        fields = {
            "atlas_graph_revision",
            "calibration_sha256",
            "controller_snapshot",
            "controller_state_name",
            "final_bank_anchor_sha256",
            "final_bank_generation",
            "frontier",
            "model_pin_sha256",
            "reward_policy",
            "sites",
            "source_manifest",
            "start_bank_anchor_sha256",
            "start_bank_generation",
            "weight_graph_revision_sha256",
        }
        body_fields = set(body) if isinstance(body, Mapping) else set()
        if not isinstance(body, Mapping) or body_fields not in (
            fields,
            fields | {"transaction_plan_sha256"},
        ):
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal export body is invalid"
            )
        try:
            claimed = require_sha256(document.get("body_sha256"), field="body_sha256")
        except ValueError as exc:
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal export body SHA-256 is invalid"
            ) from exc
        if claimed != _digest(body):
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal export body SHA-256 mismatch"
            )
        if body.get("reward_policy") != CONTROLLER_CRYSTAL_LANGUAGE_REWARD_POLICY:
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal reward policy is invalid"
            )
        raw_sites = body.get("sites")
        if not isinstance(raw_sites, list):
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal site inventory is invalid"
            )
        try:
            receipt = cls(
                controller_state_name=cast(str, body.get("controller_state_name")),
                controller_snapshot=_decode_bound_bytes(
                    body.get("controller_snapshot"),
                    label="controller snapshot",
                    maximum=MAX_CONTROLLER_CRYSTAL_EXPORT_BYTES,
                ),
                model_pin_sha256=cast(str, body.get("model_pin_sha256")),
                weight_graph_revision_sha256=cast(
                    str, body.get("weight_graph_revision_sha256")
                ),
                atlas_graph_revision=GraphRevision.from_document(
                    body.get("atlas_graph_revision")
                ),
                calibration_sha256=cast(str, body.get("calibration_sha256")),
                source_manifest=_decode_bound_bytes(
                    body.get("source_manifest"),
                    label="source CrystalStore manifest",
                    maximum=MAX_CONTROLLER_CRYSTAL_EXPORT_BYTES,
                ),
                start_bank_generation=cast(int, body.get("start_bank_generation")),
                start_bank_anchor_sha256=cast(
                    str, body.get("start_bank_anchor_sha256")
                ),
                final_bank_generation=cast(int, body.get("final_bank_generation")),
                final_bank_anchor_sha256=cast(
                    str, body.get("final_bank_anchor_sha256")
                ),
                sites=tuple(
                    ControllerCrystalSiteReceipt.from_record(row) for row in raw_sites
                ),
                frontier=ActionFrontier.from_record(body.get("frontier")),
                transaction_plan_sha256=(
                    cast(str, body.get("transaction_plan_sha256"))
                    if "transaction_plan_sha256" in body
                    else None
                ),
            )
        except ControllerCrystalBridgeIntegrityError:
            raise
        except (TypeError, ValueError) as exc:
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal export failed reconstruction"
            ) from exc
        if receipt.to_bytes() != data:
            raise ControllerCrystalBridgeIntegrityError(
                "controller-crystal export failed canonical roundtrip"
            )
        return receipt

    def restore_crystals(
        self,
        bank: ComputeCrystalBank,
    ) -> tuple[ComputeCrystal, ...]:
        """Restore and audit every exported compute artifact from the bank."""

        if not isinstance(bank, ComputeCrystalBank):
            raise TypeError("bank must be a ComputeCrystalBank")
        try:
            bank.assert_descends_from(self.final_bank_anchor_sha256)
            crystals = tuple(
                bank.restore_crystal(site.compute_crystal_sha256) for site in self.sites
            )
        except ComputeCrystalError as exc:
            raise ControllerCrystalBridgeIntegrityError(
                "exported compute bank failed restore"
            ) from exc
        manifest = CrystalManifest.from_bytes(self.source_manifest)
        for site, crystal in zip(self.sites, crystals, strict=True):
            if crystal.sha256 != site.compute_crystal_sha256:
                raise ControllerCrystalBridgeIntegrityError(
                    "restored compute crystal has another content address"
                )
            extension = crystal.extensions.get(CONTROLLER_CRYSTAL_SOURCE_EXTENSION)
            if not isinstance(extension, Mapping):
                raise ControllerCrystalBridgeIntegrityError(
                    "restored compute crystal lacks its source extension"
                )
            quantized = extension.get("quantized_kernel")
            if (
                extension.get("format") != CONTROLLER_CRYSTAL_SOURCE_SCHEMA
                or extension.get("action_id") != site.action_id
                or extension.get("atlas_graph_revision")
                != self.atlas_graph_revision.to_document()
                or extension.get("atlas_graph_revision_sha256")
                != self.atlas_graph_revision.sha256
                or extension.get("calibration_sha256") != self.calibration_sha256
                or extension.get("consensus_receipt_sha256")
                != site.consensus_receipt_sha256
                or extension.get("context_schema_sha256")
                != CONTROLLER_CRYSTAL_CONTEXT_SCHEMA_SHA256
                or extension.get("controller_snapshot_sha256")
                != self.controller_snapshot_sha256
                or extension.get("controller_state_name") != self.controller_state_name
                or extension.get("coverage_sha256") != site.coverage_sha256
                or extension.get("evidence_hashes") != dict(site.evidence_hashes)
                or extension.get("evidence_set_sha256")
                != _digest(dict(site.evidence_hashes))
                or extension.get("language_reward_policy_sha256")
                != CONTROLLER_CRYSTAL_LANGUAGE_REWARD_POLICY_SHA256
                or extension.get("model_pin_sha256") != self.model_pin_sha256
                or not isinstance(quantized, Mapping)
                or quantized.get("columns") != len(OOE_ACTIONS)
                or not isinstance(quantized.get("levels"), int)
                or isinstance(quantized.get("levels"), bool)
                or not 2 <= cast(int, quantized.get("levels")) <= 65535
                or quantized.get("rows") != len(OOE_ACTIONS)
                or quantized.get("sha256") != site.quantized_kernel_sha256
                or quantized.get("storage") != "uint16-le"
                or extension.get("site_identity") != site.site_identity.to_dict()
                or extension.get("source_manifest_sha256") != manifest.sha256
                or extension.get("source_manifest_generation") != manifest.generation
                or extension.get("source_payload_sha256") != site.source_payload_sha256
                or extension.get("site_identity_sha256") != site.site_identity.sha256
                or extension.get("verifier_hashes") != dict(site.verifier_hashes)
                or extension.get("verifier_set_sha256")
                != _digest(dict(site.verifier_hashes))
                or extension.get("weight_graph_revision_sha256")
                != self.weight_graph_revision_sha256
            ):
                raise ControllerCrystalBridgeIntegrityError(
                    "restored compute crystal source binding is stale"
                )
        return crystals

    def assert_current(
        self,
        controller: OoeController,
        bank: ComputeCrystalBank,
        *,
        require_persisted_receipt: bool = True,
    ) -> None:
        """Re-audit source state, all source payloads, and the target bank."""

        if not isinstance(controller, OoeController):
            raise TypeError("controller must be an OoeController")
        if not isinstance(bank, ComputeCrystalBank):
            raise TypeError("bank must be a ComputeCrystalBank")
        current_snapshot = controller.snapshot_bytes()
        if current_snapshot != self.controller_snapshot:
            raise ControllerCrystalBridgeStaleError(
                "controller snapshot changed after crystal export"
            )
        try:
            persisted = controller.crystal_store.restore_state(
                self.controller_state_name
            )
        except (KeyError, CrystalStoreError) as exc:
            raise ControllerCrystalBridgeIntegrityError(
                "bound controller snapshot is not persistently restorable"
            ) from exc
        if persisted != self.controller_snapshot:
            raise ControllerCrystalBridgeStaleError(
                "persisted controller snapshot differs from the export"
            )
        current_manifest = controller.crystal_store.manifest()
        if current_manifest.to_bytes() != self.source_manifest:
            raise ControllerCrystalBridgeStaleError(
                "source CrystalStore manifest changed after export"
            )
        audit = controller.crystal_store.audit()
        if (
            not audit.clean
            or audit.manifest_sha256 != self.source_manifest_sha256
            or audit.generation != current_manifest.generation
        ):
            raise ControllerCrystalBridgeIntegrityError(
                "source CrystalStore failed its complete object audit"
            )
        source_payloads: list[CrystalPayload] = []
        try:
            for site in self.sites:
                payload = controller.crystal_store.restore(site.source_payload_sha256)
                _assert_source_payload(
                    controller=controller,
                    manifest=current_manifest,
                    payload=payload,
                    site=site,
                    calibration_sha256=self.calibration_sha256,
                )
                source_payloads.append(payload)
        except (KeyError, CrystalStoreError, OoeControllerIntegrityError) as exc:
            raise ControllerCrystalBridgeIntegrityError(
                "source controller crystal failed restore or audit"
            ) from exc
        crystals = self.restore_crystals(bank)
        for site, payload, crystal in zip(
            self.sites, source_payloads, crystals, strict=True
        ):
            expected_extension = _source_extension(
                controller_state_name=self.controller_state_name,
                controller_snapshot_sha256=self.controller_snapshot_sha256,
                model_pin_sha256=self.model_pin_sha256,
                weight_graph_revision_sha256=self.weight_graph_revision_sha256,
                atlas_graph_revision=self.atlas_graph_revision,
                calibration_sha256=self.calibration_sha256,
                source_manifest=current_manifest,
                site=site,
                payload=payload,
                action_id=site.action_id,
            )
            if crystal.extensions != expected_extension or not np.array_equal(
                crystal.apply(np.eye(len(OOE_ACTIONS), dtype=np.float64)),
                payload.restore_kernel(),
            ):
                raise ControllerCrystalBridgeIntegrityError(
                    "compute crystal does not preserve its exact source kernel"
                )
        if require_persisted_receipt:
            try:
                stored_receipt = bank.store.restore_state(self.state_name)
            except (KeyError, CrystalStoreError) as exc:
                raise ControllerCrystalBridgeIntegrityError(
                    "controller-crystal export receipt is not persisted"
                ) from exc
            if stored_receipt != self.to_bytes():
                raise ControllerCrystalBridgeIntegrityError(
                    "persisted controller-crystal export receipt changed"
                )


@dataclass(frozen=True, slots=True)
class ControllerCrystalExport:
    """Runtime result of one fully verified controller export."""

    receipt: ControllerCrystalExportReceipt
    crystals: tuple[ComputeCrystal, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, ControllerCrystalExportReceipt):
            raise TypeError("receipt must be a ControllerCrystalExportReceipt")
        crystals = tuple(self.crystals)
        if len(crystals) != len(self.receipt.sites) or any(
            not isinstance(crystal, ComputeCrystal) for crystal in crystals
        ):
            raise ValueError("export crystals do not match the receipt inventory")
        if tuple(crystal.sha256 for crystal in crystals) != tuple(
            site.compute_crystal_sha256 for site in self.receipt.sites
        ):
            raise ControllerCrystalBridgeIntegrityError(
                "export crystal addresses differ from the receipt"
            )
        object.__setattr__(self, "crystals", crystals)

    @property
    def frontier(self) -> ActionFrontier:
        return self.receipt.frontier


def _assert_source_payload(
    *,
    controller: OoeController,
    manifest: CrystalManifest,
    payload: CrystalPayload,
    site: ControllerCrystalSiteReceipt | None,
    calibration_sha256: str,
) -> None:
    site_sha256 = payload.identity.sha256 if site is None else site.site_identity.sha256
    payload_sha256 = payload.sha256 if site is None else site.source_payload_sha256
    try:
        entry = manifest.resolve(site_sha256)
        named = controller.crystal_store.restore_named(site_sha256)
        coverage = controller.coverage_receipt(site_sha256)
    except (KeyError, CrystalStoreError) as exc:
        raise ControllerCrystalBridgeIntegrityError(
            "source CrystalStore cannot resolve the promoted site"
        ) from exc
    if (
        entry.payload_sha256 != payload_sha256
        or entry.identity_sha256 != site_sha256
        or named.to_bytes() != payload.to_bytes()
        or payload.name != site_sha256
        or payload.identity.sha256 != site_sha256
        or payload.identity.model_pin_sha256 != controller.model_pin_sha256
        or payload.identity.graph_revision_sha256
        != controller.weight_graph_revision_sha256
        or payload.identity.action_schema_sha256 != ACTION_SCHEMA_SHA256
        or payload.kernel_rows != len(OOE_ACTIONS)
        or payload.kernel_columns != len(OOE_ACTIONS)
        or payload.coverage_sha256 != coverage.sha256
        or payload.calibration_sha256 != calibration_sha256
        or tuple(sorted(set(dict(payload.verifier_hashes).values())))
        != coverage.verifier_sha256s
        or tuple(sorted(set(dict(payload.evidence_hashes).values())))
        != coverage.evidence_sha256s
    ):
        raise ControllerCrystalBridgeIntegrityError(
            "source controller crystal binding is stale or incomplete"
        )
    if site is not None and (
        site.site_identity != payload.identity
        or site.coverage_sha256 != payload.coverage_sha256
        or site.calibration_sha256 != payload.calibration_sha256
        or site.verifier_hashes != payload.verifier_hashes
        or site.evidence_hashes != payload.evidence_hashes
        or site.consensus_receipt_sha256 != payload.consensus_receipt_sha256
        or site.quantized_kernel_sha256 != _sha256_bytes(payload.quantized_kernel)
    ):
        raise ControllerCrystalBridgeIntegrityError(
            "site receipt differs from its source CrystalPayload"
        )


def _check_expected(
    actual: str,
    expected: str | None,
    *,
    field: str,
) -> None:
    if expected is None:
        return
    if require_sha256(expected, field=field) != actual:
        raise ControllerCrystalBridgeStaleError(f"{field} is stale")


def _assert_source_state_unchanged(
    controller: OoeController,
    *,
    controller_state_name: str,
    snapshot: bytes,
    manifest: CrystalManifest,
    stage: str,
) -> None:
    try:
        live_snapshot = controller.snapshot_bytes()
        persisted_snapshot = controller.crystal_store.restore_state(
            controller_state_name
        )
        live_manifest = controller.crystal_store.manifest()
        audit = controller.crystal_store.audit()
    except (KeyError, CrystalStoreError) as exc:
        raise ControllerCrystalBridgeIntegrityError(
            f"controller or source store became unreadable {stage}"
        ) from exc
    if (
        live_snapshot != snapshot
        or persisted_snapshot != snapshot
        or live_manifest.to_bytes() != manifest.to_bytes()
    ):
        raise ControllerCrystalBridgeStaleError(
            f"controller or source store changed {stage}"
        )
    if (
        not audit.clean
        or audit.generation != manifest.generation
        or audit.manifest_sha256 != manifest.sha256
    ):
        raise ControllerCrystalBridgeIntegrityError(
            f"source CrystalStore failed its complete object audit {stage}"
        )


def _manifest_after_crystal(
    manifest: ComputeBankManifest,
    crystal_sha256: str,
) -> ComputeBankManifest:
    digest = require_sha256(crystal_sha256, field="crystal_sha256")
    if digest in manifest.crystal_sha256s:
        return manifest
    return ComputeBankManifest(
        generation=manifest.generation + 1,
        crystal_sha256s=tuple(sorted((*manifest.crystal_sha256s, digest))),
        program_sha256s=manifest.program_sha256s,
        charge_sha256s=manifest.charge_sha256s,
        previous_manifest_sha256=manifest.sha256,
    )


def _planned_site_receipt(
    *,
    ordinal: int,
    payload: CrystalPayload,
    crystal: ComputeCrystal,
    bank_manifest_before: ComputeBankManifest,
    bank_manifest_after: ComputeBankManifest,
) -> ControllerCrystalSiteReceipt:
    return ControllerCrystalSiteReceipt(
        ordinal=ordinal,
        action_id=controller_site_action_id(payload.identity.sha256),
        site_identity=payload.identity,
        source_payload_sha256=payload.sha256,
        coverage_sha256=payload.coverage_sha256,
        calibration_sha256=payload.calibration_sha256,
        verifier_hashes=payload.verifier_hashes,
        evidence_hashes=payload.evidence_hashes,
        consensus_receipt_sha256=payload.consensus_receipt_sha256,
        quantized_kernel_sha256=_sha256_bytes(payload.quantized_kernel),
        compute_crystal_sha256=crystal.sha256,
        bank_generation_before=bank_manifest_before.generation,
        bank_generation_after=bank_manifest_after.generation,
        bank_anchor_before=bank_manifest_before.sha256,
        bank_anchor_after=bank_manifest_after.sha256,
        manifest_changed=bank_manifest_after != bank_manifest_before,
    )


def _transaction_plan(
    *,
    controller_state_name: str,
    controller_snapshot_sha256: str,
    source_manifest_sha256: str,
    payloads: Sequence[CrystalPayload],
    crystals: Sequence[ComputeCrystal],
    start_bank_manifest: ComputeBankManifest,
) -> ControllerCrystalExportTransaction:
    return ControllerCrystalExportTransaction(
        controller_state_name=controller_state_name,
        controller_snapshot_sha256=controller_snapshot_sha256,
        source_manifest_sha256=source_manifest_sha256,
        source_payload_sha256s=tuple(payload.sha256 for payload in payloads),
        compute_crystal_sha256s=tuple(crystal.sha256 for crystal in crystals),
        start_bank_manifest=start_bank_manifest.to_bytes(),
    )


def _restore_transaction(
    bank: ComputeCrystalBank,
    state_name: str,
) -> ControllerCrystalExportTransaction | None:
    try:
        data = bank.store.restore_state(state_name)
    except KeyError:
        return None
    except CrystalStoreError as exc:
        raise ControllerCrystalBridgeIntegrityError(
            "prepared controller-crystal transaction failed restore"
        ) from exc
    return ControllerCrystalExportTransaction.from_bytes(data)


def _persist_transaction(
    bank: ComputeCrystalBank,
    transaction: ControllerCrystalExportTransaction,
    *,
    previous: ControllerCrystalExportTransaction | None,
) -> None:
    data = transaction.to_bytes()
    if len(data) > bank.store.max_state_bytes:
        raise ControllerCrystalBridgeIntegrityError(
            "controller-crystal transaction exceeds target state capacity"
        )
    try:
        bank.store.publish_state(
            transaction.state_name,
            data,
            expected_sha256=None if previous is None else previous.sha256,
        )
        restored = bank.store.restore_state(transaction.state_name)
    except (CrystalStoreError, ManifestConflictError, OSError, ValueError) as exc:
        raise ControllerCrystalBridgeIntegrityError(
            "controller-crystal transaction persistence failed; retry resumes it"
        ) from exc
    if restored != data:
        raise ControllerCrystalBridgeIntegrityError(
            "controller-crystal transaction changed after publication"
        )


def export_controller_crystals(
    controller: OoeController,
    bank: ComputeCrystalBank,
    *,
    controller_state_name: str = CONTROLLER_STATE_NAME,
    expected_controller_snapshot_sha256: str | None = None,
    expected_model_pin_sha256: str | None = None,
    expected_weight_graph_revision_sha256: str | None = None,
    expected_atlas_graph_revision_sha256: str | None = None,
    expected_source_manifest_sha256: str | None = None,
    expected_compute_bank_anchor_sha256: str | None = None,
) -> ControllerCrystalExport:
    """Publish every promoted controller site and persist its exact frontier seal."""

    if not isinstance(controller, OoeController):
        raise TypeError("controller must be an OoeController")
    if not isinstance(bank, ComputeCrystalBank):
        raise TypeError("bank must be a ComputeCrystalBank")
    if (
        not isinstance(controller_state_name, str)
        or not controller_state_name
        or controller_state_name != controller_state_name.strip()
        or "\x00" in controller_state_name
    ):
        raise ValueError("controller_state_name must be canonical non-empty text")

    snapshot_before = controller.snapshot_bytes()
    snapshot_sha256 = _sha256_bytes(snapshot_before)
    body = _snapshot_body(snapshot_before)
    try:
        persisted_before = controller.crystal_store.restore_state(controller_state_name)
    except (KeyError, CrystalStoreError) as exc:
        raise ControllerCrystalBridgeIntegrityError(
            "controller must have an exact persisted snapshot before export"
        ) from exc
    if persisted_before != snapshot_before:
        raise ControllerCrystalBridgeStaleError(
            "live controller differs from its persisted snapshot"
        )
    model_pin_sha256 = require_sha256(
        body.get("model_pin_sha256"), field="model_pin_sha256"
    )
    weight_graph_revision_sha256 = require_sha256(
        body.get("weight_graph_revision_sha256"),
        field="weight_graph_revision_sha256",
    )
    try:
        atlas_graph_revision = GraphRevision.from_document(
            body.get("atlas_current_revision")
        )
    except (TypeError, ValueError) as exc:
        raise ControllerCrystalBridgeIntegrityError(
            "controller Atlas head is invalid"
        ) from exc
    router = body.get("router")
    if not isinstance(router, Mapping) or router.get("calibration_sha256") is None:
        raise ControllerCrystalBridgeIntegrityError(
            "controller router is not calibrated by promoted evidence"
        )
    calibration_sha256 = require_sha256(
        router.get("calibration_sha256"), field="calibration_sha256"
    )
    _check_expected(
        snapshot_sha256,
        expected_controller_snapshot_sha256,
        field="expected_controller_snapshot_sha256",
    )
    _check_expected(
        model_pin_sha256,
        expected_model_pin_sha256,
        field="expected_model_pin_sha256",
    )
    _check_expected(
        weight_graph_revision_sha256,
        expected_weight_graph_revision_sha256,
        field="expected_weight_graph_revision_sha256",
    )
    _check_expected(
        atlas_graph_revision.sha256,
        expected_atlas_graph_revision_sha256,
        field="expected_atlas_graph_revision_sha256",
    )

    source_manifest = controller.crystal_store.manifest()
    source_manifest_bytes = source_manifest.to_bytes()
    if (
        body.get("crystal_manifest_sha256") != source_manifest.sha256
        or body.get("crystal_manifest_generation") != source_manifest.generation
        or canonical_json_bytes(body.get("crystal_manifest")) != source_manifest_bytes
    ):
        raise ControllerCrystalBridgeStaleError(
            "controller snapshot does not bind the current CrystalStore manifest"
        )
    _check_expected(
        source_manifest.sha256,
        expected_source_manifest_sha256,
        field="expected_source_manifest_sha256",
    )
    audit = controller.crystal_store.audit()
    if (
        not audit.clean
        or audit.manifest_sha256 != source_manifest.sha256
        or audit.generation != source_manifest.generation
    ):
        raise ControllerCrystalBridgeIntegrityError(
            "source CrystalStore failed its complete object audit"
        )

    raw_sites = body.get("sites")
    if not isinstance(raw_sites, list) or not raw_sites:
        raise ControllerCrystalBridgeIntegrityError(
            "controller has no promoted sites to export"
        )
    source_addresses: list[tuple[str, str]] = []
    for raw_site in raw_sites:
        if not isinstance(raw_site, Mapping):
            raise ControllerCrystalBridgeIntegrityError(
                "controller site inventory is invalid"
            )
        site_sha256 = require_sha256(
            raw_site.get("site_identity_sha256"), field="site_identity_sha256"
        )
        source = raw_site.get("crystal_sha256")
        if source is None:
            raise ControllerCrystalBridgeIntegrityError(
                f"controller site {site_sha256} is not promoted"
            )
        source_addresses.append(
            (site_sha256, require_sha256(source, field="crystal_sha256"))
        )
    source_addresses.sort()
    if len({site for site, _ in source_addresses}) != len(source_addresses):
        raise ControllerCrystalBridgeIntegrityError(
            "controller contains duplicate site identities"
        )

    payloads: list[CrystalPayload] = []
    crystals: list[ComputeCrystal] = []
    for site_sha256, source_sha256 in source_addresses:
        try:
            payload = controller.crystal_store.restore(source_sha256)
            _assert_source_payload(
                controller=controller,
                manifest=source_manifest,
                payload=payload,
                site=None,
                calibration_sha256=calibration_sha256,
            )
        except (KeyError, CrystalStoreError, OoeControllerIntegrityError) as exc:
            raise ControllerCrystalBridgeIntegrityError(
                f"promoted controller site {site_sha256} failed source audit"
            ) from exc
        if payload.identity.sha256 != site_sha256:
            raise ControllerCrystalBridgeIntegrityError(
                "promoted payload is bound to another controller site"
            )
        action_id = controller_site_action_id(site_sha256)
        extension = _source_extension(
            controller_state_name=controller_state_name,
            controller_snapshot_sha256=snapshot_sha256,
            model_pin_sha256=model_pin_sha256,
            weight_graph_revision_sha256=weight_graph_revision_sha256,
            atlas_graph_revision=atlas_graph_revision,
            calibration_sha256=calibration_sha256,
            source_manifest=source_manifest,
            site=None,
            payload=payload,
            action_id=action_id,
        )
        try:
            crystal = ComputeCrystal.markov(
                payload.restore_kernel(),
                extensions=extension,
            )
        except (TypeError, ValueError, ComputeCrystalError) as exc:
            raise ControllerCrystalBridgeIntegrityError(
                "promoted controller kernel cannot form an exact ComputeCrystal"
            ) from exc
        if not np.array_equal(
            crystal.apply(np.eye(len(OOE_ACTIONS), dtype=np.float64)),
            payload.restore_kernel(),
        ):
            raise ControllerCrystalBridgeIntegrityError(
                "ComputeCrystal changed the promoted controller kernel"
            )
        payloads.append(payload)
        crystals.append(crystal)

    _assert_source_state_unchanged(
        controller,
        controller_state_name=controller_state_name,
        snapshot=snapshot_before,
        manifest=source_manifest,
        stage="while preparing the export",
    )

    current_start = bank.manifest()
    proposed = _transaction_plan(
        controller_state_name=controller_state_name,
        controller_snapshot_sha256=snapshot_sha256,
        source_manifest_sha256=source_manifest.sha256,
        payloads=payloads,
        crystals=crystals,
        start_bank_manifest=current_start,
    )
    transaction = _restore_transaction(bank, proposed.state_name)
    if transaction is None:
        transaction = proposed
    elif transaction.plan_record != proposed.plan_record:
        raise ControllerCrystalBridgeIntegrityError(
            "prepared transaction plan conflicts with this controller export"
        )
    start_manifest = transaction.start_manifest
    start_anchor = start_manifest.sha256
    _check_expected(
        start_anchor,
        expected_compute_bank_anchor_sha256,
        field="expected_compute_bank_anchor_sha256",
    )

    manifest_sequence = [start_manifest]
    planned_sites: list[ControllerCrystalSiteReceipt] = []
    for ordinal, (payload, crystal) in enumerate(zip(payloads, crystals, strict=True)):
        before = manifest_sequence[-1]
        after = _manifest_after_crystal(before, crystal.sha256)
        planned_sites.append(
            _planned_site_receipt(
                ordinal=ordinal,
                payload=payload,
                crystal=crystal,
                bank_manifest_before=before,
                bank_manifest_after=after,
            )
        )
        manifest_sequence.append(after)

    sites = tuple(planned_sites)
    final_manifest = manifest_sequence[-1]
    anchor = final_manifest.sha256
    authorities = _authorities(
        controller_snapshot_sha256=snapshot_sha256,
        model_pin_sha256=model_pin_sha256,
        weight_graph_revision_sha256=weight_graph_revision_sha256,
        atlas_graph_revision_sha256=atlas_graph_revision.sha256,
        calibration_sha256=calibration_sha256,
        source_manifest_sha256=source_manifest.sha256,
        final_bank_anchor_sha256=anchor,
    )
    frontier = _frontier(sites, authorities=authorities)
    receipt = ControllerCrystalExportReceipt(
        controller_state_name=controller_state_name,
        controller_snapshot=snapshot_before,
        model_pin_sha256=model_pin_sha256,
        weight_graph_revision_sha256=weight_graph_revision_sha256,
        atlas_graph_revision=atlas_graph_revision,
        calibration_sha256=calibration_sha256,
        source_manifest=source_manifest_bytes,
        start_bank_generation=start_manifest.generation,
        start_bank_anchor_sha256=start_anchor,
        final_bank_generation=final_manifest.generation,
        final_bank_anchor_sha256=anchor,
        sites=sites,
        frontier=frontier,
        transaction_plan_sha256=transaction.plan_sha256,
    )
    final_transaction = ControllerCrystalExportTransaction(
        controller_state_name=transaction.controller_state_name,
        controller_snapshot_sha256=transaction.controller_snapshot_sha256,
        source_manifest_sha256=transaction.source_manifest_sha256,
        source_payload_sha256s=transaction.source_payload_sha256s,
        compute_crystal_sha256s=transaction.compute_crystal_sha256s,
        start_bank_manifest=transaction.start_bank_manifest,
        completed_sites=sites,
        committed_receipt_sha256=receipt.sha256,
    )
    if (
        len(receipt.to_bytes()) > bank.store.max_state_bytes
        or len(final_transaction.to_bytes()) > bank.store.max_state_bytes
        or any(
            len(crystal.to_bytes()) > bank.store.max_state_bytes for crystal in crystals
        )
        or any(
            len(manifest.to_bytes()) > bank.store.max_state_bytes
            for manifest in manifest_sequence
        )
    ):
        raise ControllerCrystalBridgeIntegrityError(
            "export receipt exceeds target state capacity before publication"
        )

    if transaction.committed_receipt_sha256 is not None:
        if transaction.committed_receipt_sha256 != receipt.sha256:
            raise ControllerCrystalBridgeIntegrityError(
                "committed transaction references another export receipt"
            )
        restored = restore_controller_crystal_export(bank, receipt.sha256)
        restored.receipt.assert_current(controller, bank)
        return restored
    if not transaction.completed_sites:
        existing = _restore_transaction(bank, transaction.state_name)
        if existing is None:
            _persist_transaction(bank, transaction, previous=None)
        else:
            transaction = existing
    if transaction.completed_sites != sites[: len(transaction.completed_sites)]:
        raise ControllerCrystalBridgeIntegrityError(
            "prepared transaction progress differs from the deterministic plan"
        )

    while len(transaction.completed_sites) < len(sites):
        ordinal = len(transaction.completed_sites)
        before = manifest_sequence[ordinal]
        after = manifest_sequence[ordinal + 1]
        current = bank.manifest()
        if current == after and after != before:
            # The process stopped after the append-only bank commit but before
            # its transaction checkpoint.  The one-generation exact delta is
            # sufficient to recover that completed publication.
            pass
        elif current == before:
            if after != before:
                try:
                    publication = bank.publish_crystal(
                        crystals[ordinal],
                        expected_generation=before.generation,
                    )
                except ComputeCrystalError as exc:
                    raise ControllerCrystalBridgeIntegrityError(
                        "sequential publication failed; prepared transaction remains"
                    ) from exc
                if (
                    publication.artifact_kind != "crystal"
                    or publication.payload_sha256 != crystals[ordinal].sha256
                    or publication.generation != after.generation
                    or publication.current_anchor_sha256 != after.sha256
                    or publication.manifest_changed is not True
                    or bank.manifest() != after
                ):
                    raise ControllerCrystalBridgeIntegrityError(
                        "compute bank publication differs from its prepared delta"
                    )
            else:
                restored = bank.restore_crystal(crystals[ordinal].sha256)
                if restored.to_bytes() != crystals[ordinal].to_bytes():
                    raise ControllerCrystalBridgeIntegrityError(
                        "prepared existing compute crystal failed exact restore"
                    )
        else:
            raise ControllerCrystalBridgeStaleError(
                "compute bank diverged from the recoverable prepared transaction"
            )
        try:
            restored = bank.restore_crystal(crystals[ordinal].sha256)
        except ComputeCrystalError as exc:
            raise ControllerCrystalBridgeIntegrityError(
                "published transaction crystal failed restore"
            ) from exc
        if restored.to_bytes() != crystals[ordinal].to_bytes():
            raise ControllerCrystalBridgeIntegrityError(
                "published transaction crystal differs from its plan"
            )
        updated = transaction.append(sites[ordinal])
        _persist_transaction(bank, updated, previous=transaction)
        transaction = updated

    try:
        bank.assert_descends_from(anchor)
    except ComputeCrystalError as exc:
        raise ControllerCrystalBridgeIntegrityError(
            "compute bank lost the prepared export anchor"
        ) from exc
    _assert_source_state_unchanged(
        controller,
        controller_state_name=controller_state_name,
        snapshot=snapshot_before,
        manifest=source_manifest,
        stage="during compute publication",
    )
    try:
        bank.store.publish_state(receipt.state_name, receipt.to_bytes())
        stored = bank.store.restore_state(receipt.state_name)
    except (CrystalStoreError, ManifestConflictError, OSError, ValueError) as exc:
        raise ControllerCrystalBridgeIntegrityError(
            "export receipt persistence failed; prepared transaction resumes safely"
        ) from exc
    if stored != receipt.to_bytes():
        raise ControllerCrystalBridgeIntegrityError(
            "persisted export receipt changed after publication"
        )
    committed = transaction.commit(receipt.sha256)
    _persist_transaction(bank, committed, previous=transaction)
    receipt.assert_current(controller, bank)
    return ControllerCrystalExport(receipt=receipt, crystals=tuple(crystals))


def restore_controller_crystal_export(
    bank: ComputeCrystalBank,
    receipt_sha256: str,
) -> ControllerCrystalExport:
    """Restore a persisted export receipt and all of its compute crystals."""

    if not isinstance(bank, ComputeCrystalBank):
        raise TypeError("bank must be a ComputeCrystalBank")
    digest = require_sha256(receipt_sha256, field="receipt_sha256")
    state_name = CONTROLLER_CRYSTAL_EXPORT_STATE_PREFIX + digest
    try:
        data = bank.store.restore_state(state_name)
    except (KeyError, CrystalStoreError) as exc:
        raise ControllerCrystalBridgeIntegrityError(
            "controller-crystal export receipt is missing or invalid"
        ) from exc
    if _sha256_bytes(data) != digest:
        raise ControllerCrystalBridgeIntegrityError(
            "controller-crystal export receipt address mismatches its bytes"
        )
    receipt = ControllerCrystalExportReceipt.from_bytes(data)
    if not receipt.is_legacy:
        assert receipt.transaction_plan_sha256 is not None
        transaction = restore_controller_crystal_transaction(
            bank,
            receipt.transaction_plan_sha256,
        )
        if (
            transaction.committed_receipt_sha256 != receipt.sha256
            or transaction.completed_sites != receipt.sites
            or transaction.start_manifest.sha256 != receipt.start_bank_anchor_sha256
            or transaction.start_manifest.generation != receipt.start_bank_generation
        ):
            raise ControllerCrystalBridgeIntegrityError(
                "export receipt lacks its exact committed transaction"
            )
    crystals = receipt.restore_crystals(bank)
    return ControllerCrystalExport(receipt=receipt, crystals=crystals)


def restore_controller_crystal_transaction(
    bank: ComputeCrystalBank,
    plan_sha256: str,
) -> ControllerCrystalExportTransaction:
    """Restore one prepared or committed export transaction by plan address."""

    if not isinstance(bank, ComputeCrystalBank):
        raise TypeError("bank must be a ComputeCrystalBank")
    digest = require_sha256(plan_sha256, field="plan_sha256")
    transaction = _restore_transaction(
        bank,
        CONTROLLER_CRYSTAL_TRANSACTION_STATE_PREFIX + digest,
    )
    if transaction is None:
        raise ControllerCrystalBridgeIntegrityError(
            "controller-crystal export transaction is missing"
        )
    if transaction.plan_sha256 != digest:
        raise ControllerCrystalBridgeIntegrityError(
            "controller-crystal transaction address mismatches its plan"
        )
    return transaction


__all__ = [
    "CONTROLLER_CRYSTAL_ACTION_SCHEMA",
    "CONTROLLER_CRYSTAL_CONTEXT_SCHEMA",
    "CONTROLLER_CRYSTAL_CONTEXT_SCHEMA_SHA256",
    "CONTROLLER_CRYSTAL_EXPORT_SCHEMA",
    "CONTROLLER_CRYSTAL_EXPORT_STATE_PREFIX",
    "CONTROLLER_CRYSTAL_LANGUAGE_REWARD_POLICY",
    "CONTROLLER_CRYSTAL_LANGUAGE_REWARD_POLICY_SHA256",
    "CONTROLLER_CRYSTAL_SOURCE_EXTENSION",
    "CONTROLLER_CRYSTAL_SOURCE_SCHEMA",
    "CONTROLLER_CRYSTAL_TRANSACTION_SCHEMA",
    "CONTROLLER_CRYSTAL_TRANSACTION_STATE_PREFIX",
    "ControllerCrystalBridgeError",
    "ControllerCrystalBridgeIntegrityError",
    "ControllerCrystalBridgeStaleError",
    "ControllerCrystalExport",
    "ControllerCrystalExportReceipt",
    "ControllerCrystalExportTransaction",
    "ControllerCrystalSiteReceipt",
    "controller_site_action_id",
    "export_controller_crystals",
    "restore_controller_crystal_export",
    "restore_controller_crystal_transaction",
]
