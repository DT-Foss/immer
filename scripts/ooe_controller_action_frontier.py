#!/usr/bin/env python3
"""Promote a learned controller fork and export its executable action frontier."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from immer.runtimes.ooe.compute_crystals import ComputeCrystalBank
from immer.runtimes.ooe.controller import (
    CONTROLLER_STATE_NAME,
    OoeController,
    OoeControllerIntegrityError,
)
from immer.runtimes.ooe.controller_crystal_bridge import (
    export_controller_crystals,
    restore_controller_crystal_export,
)
from immer.runtimes.ooe.crystal import (
    CrystalManifest,
    CrystalManifestEntry,
    CrystalStore,
)
from immer.runtimes.ooe.identity import canonical_json_bytes, require_sha256
from immer.runtimes.ooe.qwen_bridge import OOE_ACTIONS

from ooe_controller_language_bootstrap import (
    _LiveAtlasRevisionMembership,
    _existing_directory,
    _open_atlas,
    _output_directory,
    _stable_regular_bytes,
    _write_atomic_immutable,
)
from ooe_fertig_action_learning import (
    _assert_disjoint_store_roots,
    _assert_output_outside_stores,
    _assert_planned_output_outside_stores,
    _assert_planned_store_roots,
    _commit_fork_provenance,
    _open_or_create_controller_fork,
    _target_store_path,
)


REPORT_SCHEMA = "immer-ooe-controller-action-frontier-report/v1"
_REPORT_BODY_FIELDS = frozenset(
    {
        "atlas_graph_revision_sha256",
        "compute_bank_anchor_sha256",
        "controller_snapshot_sha256",
        "execute_fertig_action_index",
        "export_receipt_sha256",
        "format",
        "frontier_sha256",
        "learned_controller_snapshot_sha256",
        "model_pin_sha256",
        "rows",
        "site_count",
        "source_manifest_sha256",
        "weight_graph_revision_sha256",
    }
)


class ControllerActionFrontierCliError(RuntimeError):
    """The promoted controller fork or frontier export is not exact."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _optional_pin(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    try:
        return require_sha256(value, field=field)
    except ValueError as exc:
        raise ControllerActionFrontierCliError(f"{field} is not a SHA-256") from exc


def _assert_optional(actual: str, expected: str | None, *, field: str) -> None:
    if expected is not None and actual != expected:
        raise ControllerActionFrontierCliError(f"{field} is stale")


def _snapshot_body(data: bytes) -> dict[str, Any]:
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ControllerActionFrontierCliError(
            "controller snapshot is invalid JSON"
        ) from exc
    if canonical_json_bytes(document) != data:
        raise ControllerActionFrontierCliError(
            "controller snapshot is not canonical JSON"
        )
    if not isinstance(document, Mapping) or not isinstance(
        document.get("body"), dict
    ):
        raise ControllerActionFrontierCliError("controller snapshot is invalid")
    return document["body"]


def _unpromoted_sites(controller: OoeController) -> tuple[str, ...]:
    body = _snapshot_body(controller.snapshot_bytes())
    rows = body.get("sites")
    if not isinstance(rows, list):
        raise ControllerActionFrontierCliError(
            "controller snapshot has no site inventory"
        )
    pending = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ControllerActionFrontierCliError(
                "controller snapshot site is invalid"
            )
        site = require_sha256(row.get("site_identity_sha256"), field="site")
        crystal = row.get("crystal_sha256")
        if crystal is None:
            pending.append(site)
        else:
            require_sha256(crystal, field="crystal_sha256")
    return tuple(sorted(pending))


def _promotion_lineage_sha256(snapshot: bytes) -> str:
    body = json.loads(canonical_json_bytes(_snapshot_body(snapshot)))
    for field in (
        "crystal_manifest",
        "crystal_manifest_generation",
        "crystal_manifest_sha256",
    ):
        body[field] = None
    router = body.get("router")
    metrics = body.get("metrics")
    sites = body.get("sites")
    if (
        not isinstance(router, dict)
        or not isinstance(metrics, dict)
        or not isinstance(sites, list)
    ):
        raise ControllerActionFrontierCliError(
            "controller promotion lineage fields are invalid"
        )
    for field in ("calibration_sha256", "radius", "min_margin"):
        router[field] = None
    metrics["promotions"] = 0
    for site in sites:
        if not isinstance(site, dict) or "crystal_sha256" not in site:
            raise ControllerActionFrontierCliError(
                "controller promotion lineage site is invalid"
            )
        site["crystal_sha256"] = None
    return _digest(
        {
            "normalized_controller": body,
            "schema": "immer-ooe-promotion-lineage/v1",
        }
    )


def _expected_promotion_manifest(
    learned_manifest: CrystalManifest,
    promoted: OoeController,
) -> CrystalManifest:
    body = _snapshot_body(promoted.snapshot_bytes())
    raw_sites = body.get("sites")
    if not isinstance(raw_sites, list):
        raise ControllerActionFrontierCliError(
            "promoted controller has no site inventory"
        )
    active = []
    for row in raw_sites:
        if not isinstance(row, Mapping):
            raise ControllerActionFrontierCliError(
                "promoted controller site inventory is invalid"
            )
        site = require_sha256(row.get("site_identity_sha256"), field="site")
        crystal = require_sha256(row.get("crystal_sha256"), field="crystal_sha256")
        active.append((site, crystal))
    manifest = learned_manifest
    for site, crystal in sorted(active):
        entries = {entry.name: entry for entry in manifest.entries}
        prior = entries.get(site)
        if prior is not None and prior.payload_sha256 == crystal:
            continue
        entries[site] = CrystalManifestEntry(
            name=site,
            payload_sha256=crystal,
            identity_sha256=site,
        )
        manifest = CrystalManifest(
            generation=manifest.generation + 1,
            entries=tuple(sorted(entries.values(), key=lambda row: row.name)),
            objects=tuple(sorted(set(manifest.objects) | {crystal})),
            previous_manifest_sha256=manifest.sha256,
        )
    return manifest


def _load_report(path: Path) -> dict[str, Any]:
    try:
        data = _stable_regular_bytes(path, maximum=128 * 1024 * 1024)
        document = json.loads(data)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ControllerActionFrontierCliError(
            "existing frontier report is invalid JSON"
        ) from exc
    if canonical_json_bytes(document) != data:
        raise ControllerActionFrontierCliError(
            "existing frontier report is not canonical JSON"
        )
    if not isinstance(document, dict) or set(document) != {
        "body",
        "schema",
        "sha256",
    }:
        raise ControllerActionFrontierCliError(
            "existing frontier report envelope is invalid"
        )
    body = document.get("body")
    if (
        document.get("schema") != REPORT_SCHEMA
        or not isinstance(body, dict)
        or set(body) != _REPORT_BODY_FIELDS
        or body.get("format") != REPORT_SCHEMA
        or document.get("sha256") != _digest(body)
    ):
        raise ControllerActionFrontierCliError(
            "existing frontier report seal is invalid"
        )
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Clone an execution-learned OoE controller, promote all exact "
            "sites, and export their Markov kernels as ComputeCrystals."
        )
    )
    parser.add_argument("--learned-controller-store", required=True)
    parser.add_argument("--promoted-controller-store", required=True)
    parser.add_argument("--atlas-root", required=True)
    parser.add_argument("--compute-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--controller-state-name", default=CONTROLLER_STATE_NAME)
    parser.add_argument("--expected-learned-snapshot-sha256")
    parser.add_argument("--expected-model-pin-sha256")
    parser.add_argument("--expected-weight-graph-revision-sha256")
    parser.add_argument("--expected-compute-bank-anchor-sha256")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    learned_root = _existing_directory(
        args.learned_controller_store,
        label="learned controller store",
    )
    atlas_root = _existing_directory(args.atlas_root, label="Atlas root")
    promoted_root = Path(args.promoted_controller_store).expanduser().absolute()
    compute_root = Path(args.compute_root).expanduser().absolute()
    output = Path(args.output).expanduser().absolute()
    _assert_planned_store_roots(
        learned_root,
        promoted_root,
        compute_root,
        atlas_root,
    )
    _assert_planned_output_outside_stores(
        output,
        learned_root,
        promoted_root,
        compute_root,
        atlas_root,
    )
    promoted_root = _target_store_path(
        str(promoted_root),
        label="promoted controller store",
    )
    compute_root = _output_directory(str(compute_root), label="compute root")
    _assert_disjoint_store_roots(
        learned_root,
        promoted_root,
        compute_root,
        atlas_root,
    )
    _output_directory(str(output.parent), label="output parent")
    expected_learned = _optional_pin(
        args.expected_learned_snapshot_sha256,
        field="expected_learned_snapshot_sha256",
    )
    expected_model = _optional_pin(
        args.expected_model_pin_sha256,
        field="expected_model_pin_sha256",
    )
    expected_weight = _optional_pin(
        args.expected_weight_graph_revision_sha256,
        field="expected_weight_graph_revision_sha256",
    )
    expected_bank = _optional_pin(
        args.expected_compute_bank_anchor_sha256,
        field="expected_compute_bank_anchor_sha256",
    )

    atlas = _open_atlas(atlas_root)
    membership = _LiveAtlasRevisionMembership(atlas)
    learned_store = CrystalStore(learned_root)
    learned_store.clean_staging(
        expected_generation=learned_store.manifest().generation
    )
    learned_state = learned_store.restore_state(args.controller_state_name)
    learned_sha256 = hashlib.sha256(learned_state).hexdigest()
    _assert_optional(learned_sha256, expected_learned, field="learned snapshot")
    learned = OoeController.restore(
        crystal_store=learned_store,
        name=args.controller_state_name,
        atlas_revision_verifier=membership,
        expected_model_pin_sha256=expected_model,
        expected_weight_graph_revision_sha256=expected_weight,
    )
    _assert_optional(learned.model_pin_sha256, expected_model, field="model pin")
    _assert_optional(
        learned.weight_graph_revision_sha256,
        expected_weight,
        field="weight graph revision",
    )

    bank = ComputeCrystalBank(compute_root)
    prior_report = None
    prior_export = None
    if output.exists() or output.is_symlink():
        prior_report = _load_report(output)
        prior_body = prior_report["body"]
        if (
            prior_body.get("learned_controller_snapshot_sha256") != learned_sha256
            or prior_body.get("model_pin_sha256") != learned.model_pin_sha256
            or prior_body.get("weight_graph_revision_sha256")
            != learned.weight_graph_revision_sha256
        ):
            raise ControllerActionFrontierCliError(
                "existing frontier report belongs to another learned controller"
            )
        receipt_sha256 = require_sha256(
            prior_body.get("export_receipt_sha256"),
            field="export_receipt_sha256",
        )
        prior_export = restore_controller_crystal_export(bank, receipt_sha256)
        _assert_optional(
            prior_export.receipt.start_bank_anchor_sha256,
            expected_bank,
            field="compute bank start anchor",
        )

    promoted_store, fork_provenance, provenance_present = (
        _open_or_create_controller_fork(
            learned_store,
            promoted_root,
            state_name=args.controller_state_name,
        )
    )
    _assert_disjoint_store_roots(
        learned_root,
        promoted_root,
        compute_root,
        atlas_root,
    )
    _assert_output_outside_stores(
        output,
        learned_root,
        promoted_root,
        compute_root,
        atlas_root,
    )
    promoted_store.clean_staging(
        expected_generation=promoted_store.manifest().generation
    )
    try:
        promoted = OoeController.restore(
            crystal_store=promoted_store,
            name=args.controller_state_name,
            atlas_revision_verifier=membership,
            expected_model_pin_sha256=learned.model_pin_sha256,
            expected_weight_graph_revision_sha256=(
                learned.weight_graph_revision_sha256
            ),
        )
    except OoeControllerIntegrityError:
        learned_body = _snapshot_body(learned_state)
        promoted, _recovery = OoeController.restore_recoverable(
            crystal_store=promoted_store,
            name=args.controller_state_name,
            atlas_revision_verifier=membership,
            expected_model_pin_sha256=learned.model_pin_sha256,
            expected_weight_graph_revision_sha256=(
                learned.weight_graph_revision_sha256
            ),
            expected_old_manifest_generation=learned_body[
                "crystal_manifest_generation"
            ],
            expected_old_manifest_sha256=learned_body["crystal_manifest_sha256"],
            expected_old_state_sha256=learned_sha256,
        )
    if _promotion_lineage_sha256(promoted.snapshot_bytes()) != (
        _promotion_lineage_sha256(learned_state)
    ):
        raise ControllerActionFrontierCliError(
            "promoted controller is not a promotion-only learned descendant"
        )
    persisted_before = promoted_store.restore_state(args.controller_state_name)
    pending = _unpromoted_sites(promoted)
    for site in pending:
        coverage = promoted.coverage_receipt(site)
        promoted.promote(
            site,
            coverage_sha256=coverage.sha256,
            verifier_sha256s=coverage.verifier_sha256s,
            expected_store_generation=promoted_store.manifest().generation,
        )
    if _unpromoted_sites(promoted):
        raise ControllerActionFrontierCliError(
            "controller still contains unpromoted sites"
        )
    expected_promoted_manifest = _expected_promotion_manifest(
        learned_store.manifest(),
        promoted,
    )
    if promoted_store.manifest().to_bytes() != expected_promoted_manifest.to_bytes():
        raise ControllerActionFrontierCliError(
            "promoted manifest is not the exact deterministic learned delta"
        )
    if not provenance_present:
        _commit_fork_provenance(
            promoted_store,
            fork_provenance,
            state_name=args.controller_state_name,
        )
    if promoted.snapshot_bytes() != persisted_before:
        publication = promoted.save_snapshot(
            name=args.controller_state_name,
            expected_sha256=hashlib.sha256(persisted_before).hexdigest(),
        )
        if publication.payload_sha256 != hashlib.sha256(
            promoted.snapshot_bytes()
        ).hexdigest():
            raise ControllerActionFrontierCliError(
                "promoted controller snapshot publication changed bytes"
            )
    if promoted_store.restore_state(args.controller_state_name) != (
        promoted.snapshot_bytes()
    ):
        raise ControllerActionFrontierCliError(
            "promoted controller differs from persistent state"
        )

    if prior_report is not None:
        if prior_export is None:
            raise AssertionError("prior report lost its restored export")
        export = prior_export
        export.receipt.assert_current(promoted, bank)
    else:
        export = export_controller_crystals(
            promoted,
            bank,
            controller_state_name=args.controller_state_name,
            expected_controller_snapshot_sha256=hashlib.sha256(
                promoted.snapshot_bytes()
            ).hexdigest(),
            expected_model_pin_sha256=promoted.model_pin_sha256,
            expected_weight_graph_revision_sha256=(
                promoted.weight_graph_revision_sha256
            ),
            expected_atlas_graph_revision_sha256=(
                promoted.atlas_graph_revision_sha256
            ),
            expected_source_manifest_sha256=promoted_store.manifest().sha256,
            expected_compute_bank_anchor_sha256=expected_bank,
        )

    execute_index = OOE_ACTIONS.index("execute_fertig")
    rows = []
    for site, crystal in zip(export.receipt.sites, export.crystals, strict=True):
        kernel = crystal.apply(np.eye(len(OOE_ACTIONS), dtype=np.float64))
        execute_row = kernel[execute_index]
        rows.append(
            {
                "compute_crystal_sha256": crystal.sha256,
                "execute_fertig_choice": OOE_ACTIONS[int(np.argmax(execute_row))],
                "execute_fertig_row": execute_row.tolist(),
                "site_identity_sha256": site.site_identity.sha256,
                "source_payload_sha256": site.source_payload_sha256,
            }
        )
    body = {
        "atlas_graph_revision_sha256": (
            export.receipt.atlas_graph_revision.sha256
        ),
        "compute_bank_anchor_sha256": export.receipt.final_bank_anchor_sha256,
        "controller_snapshot_sha256": export.receipt.controller_snapshot_sha256,
        "execute_fertig_action_index": execute_index,
        "export_receipt_sha256": export.receipt.sha256,
        "format": REPORT_SCHEMA,
        "frontier_sha256": export.frontier.sha256,
        "learned_controller_snapshot_sha256": learned_sha256,
        "model_pin_sha256": export.receipt.model_pin_sha256,
        "rows": rows,
        "site_count": len(rows),
        "source_manifest_sha256": export.receipt.source_manifest_sha256,
        "weight_graph_revision_sha256": (
            export.receipt.weight_graph_revision_sha256
        ),
    }
    report = {"body": body, "schema": REPORT_SCHEMA, "sha256": _digest(body)}
    _write_atomic_immutable(output, canonical_json_bytes(report))
    return report


def main() -> int:
    args = _parser().parse_args()
    try:
        report = run(args)
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"ooe_controller_action_frontier: {exc}") from exc
    print(report["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
