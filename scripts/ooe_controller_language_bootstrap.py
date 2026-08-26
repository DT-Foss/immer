#!/usr/bin/env python3
"""Bootstrap executable Markov language from a promoted OoE controller."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
import stat
import sys

from immer.knowledge.livecausal import LiveGraph
from immer.runtimes.deepseek_v4.causal_weights import TensorRangePlan
from immer.runtimes.ooe.compute_crystals import ComputeCrystalBank
from immer.runtimes.ooe.controller import CONTROLLER_STATE_NAME, OoeController
from immer.runtimes.ooe.controller_crystal_bridge import (
    ControllerCrystalBridgeError,
    export_controller_crystals,
    restore_controller_crystal_export,
)
from immer.runtimes.ooe.controller_language_intelligence import (
    ControllerLanguageBootstrapConfig,
    ControllerLanguageBootstrapReport,
    bootstrap_controller_crystal_language,
)
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.identity import require_sha256
from immer.runtimes.qwen3_8.semantic_atlas import (
    ATLAS_EDGE_SCHEMA,
    MEASUREMENT_RECEIPT_SCHEMA,
    GraphRevision,
    MeasurementReceipt,
    SemanticWeightAtlas,
)


class ControllerLanguageBootstrapCliError(RuntimeError):
    """The CLI input or persistent runtime cannot be authenticated."""


def _existing_directory(value: str, *, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ControllerLanguageBootstrapCliError(
            f"{label} does not exist: {path}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise ControllerLanguageBootstrapCliError(
            f"{label} must be a real directory: {path}"
        )
    return path


def _output_directory(value: str, *, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    path.mkdir(parents=True, exist_ok=True)
    return _existing_directory(str(path), label=label)


def _stable_regular_bytes(path: Path, *, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ControllerLanguageBootstrapCliError(
            f"cannot open regular file: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
            raise ControllerLanguageBootstrapCliError(
                f"file is not bounded regular data: {path}"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ControllerLanguageBootstrapCliError(
                    f"file ended during stable read: {path}"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)

        def identity(row: os.stat_result) -> tuple[int, int, int, int, int]:
            return (
                row.st_dev,
                row.st_ino,
                row.st_size,
                row.st_mtime_ns,
                row.st_ctime_ns,
            )

        if identity(before) != identity(after):
            raise ControllerLanguageBootstrapCliError(
                f"file changed during stable read: {path}"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _write_atomic_immutable(path: Path, data: bytes) -> None:
    """Install one report without replacing an existing different report."""

    if path.exists() or path.is_symlink():
        if _stable_regular_bytes(path, maximum=128 * 1024 * 1024) != data:
            raise ControllerLanguageBootstrapCliError(
                "output report already exists with different bytes"
            )
        return
    parent = _existing_directory(str(path.parent), label="output parent")
    token = hashlib.sha256(data).hexdigest()[:16]
    temporary = parent / f".{path.name}.{os.getpid()}.{token}.tmp"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        try:
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short report write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if _stable_regular_bytes(path, maximum=128 * 1024 * 1024) != data:
                raise ControllerLanguageBootstrapCliError(
                    "concurrent output report contains different bytes"
                )
        directory = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _plan_from_measurement(measurement: MeasurementReceipt) -> TensorRangePlan:
    coordinate = measurement.coordinate
    plan = TensorRangePlan(
        name=coordinate.tensor,
        dtype=coordinate.dtype,
        shape=coordinate.shape,
        shard=coordinate.shard,
        absolute_offset=coordinate.tensor_absolute_offset,
        length=coordinate.tensor_length,
    )
    if not coordinate.matches_plan(plan):
        raise ControllerLanguageBootstrapCliError(
            "Atlas measurement failed tensor-plan reconstruction"
        )
    return plan


def _open_atlas(root: Path) -> SemanticWeightAtlas:
    try:
        graph = LiveGraph(root)
        if not graph.store.verify(include_tombstones=True):
            raise ControllerLanguageBootstrapCliError(
                "Atlas LiveGraph segment verification failed"
            )
        measurements: dict[str, MeasurementReceipt] = {}
        for segment_sha256 in graph.store.segments():
            records = tuple(
                record
                for _sha256, _index, record in graph.store.iter_records(segment_sha256)
            )
            for record in records:
                if (
                    record.get("schema") == ATLAS_EDGE_SCHEMA
                    and record.get("edge_kind") == "primary"
                ):
                    document = record.get("document")
                    if (
                        isinstance(document, Mapping)
                        and document.get("schema") == MEASUREMENT_RECEIPT_SCHEMA
                    ):
                        measurement = MeasurementReceipt.from_document(document)
                        if measurement.sha256 != record.get("document_sha256"):
                            raise ControllerLanguageBootstrapCliError(
                                "Atlas primary measurement identity mismatch"
                            )
                        measurements[measurement.sha256] = measurement
        if not measurements:
            raise ControllerLanguageBootstrapCliError(
                "Atlas contains no primary MeasurementReceipts"
            )
        pins = {item.model_pin.sha256: item.model_pin for item in measurements.values()}
        if len(pins) != 1:
            raise ControllerLanguageBootstrapCliError(
                "Atlas primary measurements span multiple model pins"
            )
        plans: dict[str, TensorRangePlan] = {}
        for measurement in measurements.values():
            plan = _plan_from_measurement(measurement)
            prior = plans.get(plan.name)
            if prior is not None and prior != plan:
                raise ControllerLanguageBootstrapCliError(
                    "Atlas contains conflicting tensor plans"
                )
            plans[plan.name] = plan
        atlas = SemanticWeightAtlas(
            graph,
            model_pin=next(iter(pins.values())),
            tensor_plans=tuple(plans.values()),
        )
        atlas.verify_or_raise()
        return atlas
    except ControllerLanguageBootstrapCliError:
        raise
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise ControllerLanguageBootstrapCliError(
            "cannot reconstruct and authenticate the SemanticWeightAtlas"
        ) from exc


class _LiveAtlasRevisionMembership:
    """Force-rehashed exact revision membership over the live Atlas journal."""

    def __init__(self, atlas: SemanticWeightAtlas) -> None:
        if not isinstance(atlas, SemanticWeightAtlas):
            raise TypeError("atlas must be a SemanticWeightAtlas")
        self.atlas = atlas
        atlas.verify_or_raise()

    def __call__(self, revision: GraphRevision) -> bool:
        if not isinstance(revision, GraphRevision):
            return False
        self.atlas.verify_or_raise()
        before = self.atlas.revision()
        if revision.sequence > before.sequence:
            return False
        present = self.atlas.contains_revision(revision)
        self.atlas.verify_or_raise()
        after = self.atlas.revision()
        if after != before:
            raise ControllerLanguageBootstrapCliError(
                "Atlas changed during revision-membership verification"
            )
        return bool(present)


def _pin(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    try:
        return require_sha256(value, field=field)
    except ValueError as exc:
        raise ControllerLanguageBootstrapCliError(
            f"{field} must be a lowercase SHA-256"
        ) from exc


def _assert_equal(actual: str, expected: str | None, *, field: str) -> None:
    if expected is not None and actual != expected:
        raise ControllerLanguageBootstrapCliError(f"{field} is stale")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and compile executable Markov language from promoted OoE "
            "controller Crystals."
        )
    )
    parser.add_argument("--controller-store", required=True)
    parser.add_argument("--atlas-root", required=True)
    parser.add_argument("--compute-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--lexicon-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--controller-state-name", default=CONTROLLER_STATE_NAME)
    parser.add_argument("--export-receipt-sha256")
    parser.add_argument("--expected-controller-snapshot-sha256")
    parser.add_argument("--expected-model-pin-sha256")
    parser.add_argument("--expected-weight-graph-revision-sha256")
    parser.add_argument("--expected-atlas-graph-revision-sha256")
    parser.add_argument("--expected-source-manifest-sha256")
    parser.add_argument(
        "--expected-compute-bank-anchor-sha256",
        help="required pre-export bank anchor when publishing a new export",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-training-episodes", type=int, default=4_000)
    parser.add_argument("--exploration-start", type=float, default=0.45)
    parser.add_argument("--exploration-end", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.2)
    parser.add_argument("--minimum-visits", type=int, default=8)
    parser.add_argument("--minimum-value", type=float, default=0.25)
    parser.add_argument("--minimum-margin", type=float, default=0.08)
    parser.add_argument("--context-full-weight-visits", type=int, default=16)
    parser.add_argument("--stable-greedy-cycles", type=int, default=3)
    return parser


def run(args: argparse.Namespace) -> ControllerLanguageBootstrapReport:
    controller_root = _existing_directory(
        args.controller_store, label="controller store"
    )
    atlas_root = _existing_directory(args.atlas_root, label="Atlas root")
    compute_root = _output_directory(args.compute_root, label="compute root")
    state_root = _output_directory(args.state_root, label="state root")
    lexicon_root = _output_directory(args.lexicon_root, label="lexicon root")
    output = Path(args.output).expanduser().absolute()
    _output_directory(str(output.parent), label="output parent")

    expected_snapshot = _pin(
        args.expected_controller_snapshot_sha256,
        field="expected_controller_snapshot_sha256",
    )
    expected_model = _pin(
        args.expected_model_pin_sha256, field="expected_model_pin_sha256"
    )
    expected_weight = _pin(
        args.expected_weight_graph_revision_sha256,
        field="expected_weight_graph_revision_sha256",
    )
    expected_atlas = _pin(
        args.expected_atlas_graph_revision_sha256,
        field="expected_atlas_graph_revision_sha256",
    )
    expected_manifest = _pin(
        args.expected_source_manifest_sha256,
        field="expected_source_manifest_sha256",
    )
    expected_bank = _pin(
        args.expected_compute_bank_anchor_sha256,
        field="expected_compute_bank_anchor_sha256",
    )
    explicit_export = _pin(args.export_receipt_sha256, field="export_receipt_sha256")

    prior_report: ControllerLanguageBootstrapReport | None = None
    if output.exists() or output.is_symlink():
        prior_report = ControllerLanguageBootstrapReport.from_bytes(
            _stable_regular_bytes(output, maximum=128 * 1024 * 1024)
        )
        if explicit_export is not None and (
            explicit_export != prior_report.export_receipt_sha256
        ):
            raise ControllerLanguageBootstrapCliError(
                "output report and --export-receipt-sha256 conflict"
            )
        explicit_export = prior_report.export_receipt_sha256

    atlas = _open_atlas(atlas_root)
    membership = _LiveAtlasRevisionMembership(atlas)
    controller_store = CrystalStore(controller_root)
    controller = OoeController.restore(
        crystal_store=controller_store,
        name=args.controller_state_name,
        atlas_revision_verifier=membership,
        expected_model_pin_sha256=expected_model,
        expected_weight_graph_revision_sha256=expected_weight,
    )
    snapshot_sha = hashlib.sha256(controller.snapshot_bytes()).hexdigest()
    _assert_equal(
        snapshot_sha, expected_snapshot, field="expected_controller_snapshot_sha256"
    )
    _assert_equal(
        controller.atlas_graph_revision_sha256,
        expected_atlas,
        field="expected_atlas_graph_revision_sha256",
    )
    _assert_equal(
        controller.crystal_store.manifest().sha256,
        expected_manifest,
        field="expected_source_manifest_sha256",
    )

    bank = ComputeCrystalBank(compute_root)
    if explicit_export is None:
        export = export_controller_crystals(
            controller,
            bank,
            controller_state_name=args.controller_state_name,
            expected_controller_snapshot_sha256=expected_snapshot,
            expected_model_pin_sha256=expected_model,
            expected_weight_graph_revision_sha256=expected_weight,
            expected_atlas_graph_revision_sha256=expected_atlas,
            expected_source_manifest_sha256=expected_manifest,
            expected_compute_bank_anchor_sha256=expected_bank,
        )
    else:
        try:
            export = restore_controller_crystal_export(bank, explicit_export)
        except ControllerCrystalBridgeError:
            export = export_controller_crystals(
                controller,
                bank,
                controller_state_name=args.controller_state_name,
                expected_controller_snapshot_sha256=expected_snapshot,
                expected_model_pin_sha256=expected_model,
                expected_weight_graph_revision_sha256=expected_weight,
                expected_atlas_graph_revision_sha256=expected_atlas,
                expected_source_manifest_sha256=expected_manifest,
                expected_compute_bank_anchor_sha256=expected_bank,
            )
            if export.receipt.sha256 != explicit_export:
                raise ControllerLanguageBootstrapCliError(
                    "compute bank cannot reconstruct the output report export"
                )
        else:
            if (
                expected_bank is not None
                and export.receipt.start_bank_anchor_sha256 != expected_bank
            ):
                raise ControllerLanguageBootstrapCliError(
                    "expected_compute_bank_anchor_sha256 is stale"
                )
            export.receipt.assert_current(controller, bank)
    _assert_equal(
        export.receipt.controller_snapshot_sha256,
        expected_snapshot,
        field="expected_controller_snapshot_sha256",
    )
    _assert_equal(
        export.receipt.model_pin_sha256,
        expected_model,
        field="expected_model_pin_sha256",
    )
    _assert_equal(
        export.receipt.weight_graph_revision_sha256,
        expected_weight,
        field="expected_weight_graph_revision_sha256",
    )
    _assert_equal(
        export.receipt.atlas_graph_revision.sha256,
        expected_atlas,
        field="expected_atlas_graph_revision_sha256",
    )
    _assert_equal(
        export.receipt.source_manifest_sha256,
        expected_manifest,
        field="expected_source_manifest_sha256",
    )

    config = ControllerLanguageBootstrapConfig(
        seed=args.seed,
        max_training_episodes=args.max_training_episodes,
        exploration_start=args.exploration_start,
        exploration_end=args.exploration_end,
        learning_rate=args.learning_rate,
        minimum_visits=args.minimum_visits,
        minimum_value=args.minimum_value,
        minimum_margin=args.minimum_margin,
        context_full_weight_visits=args.context_full_weight_visits,
        stable_greedy_cycles=args.stable_greedy_cycles,
    )
    if prior_report is not None and prior_report.config != config:
        raise ControllerLanguageBootstrapCliError(
            "output report was produced with another bootstrap configuration"
        )
    result = bootstrap_controller_crystal_language(
        export,
        bank,
        state_store=CrystalStore(state_root),
        lexicon_store=CrystalStore(lexicon_root),
        config=config,
    )
    _write_atomic_immutable(output, result.report.to_bytes())
    return result.report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    sys.stdout.buffer.write(report.to_bytes() + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
