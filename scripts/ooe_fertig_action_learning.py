#!/usr/bin/env python3
"""Learn real FERTIG executions on an immutable fork of a Qwen/O1 controller."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import stat
from typing import Any

from immer.cognition.fertig import FertigSolver
from immer.runtimes.ooe.controller import CONTROLLER_STATE_NAME, OoeController
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.execution_learning import (
    ActionAuthorityReceipt,
    ExecutionLearningBank,
    ExecutionLearningBridge,
)
from immer.runtimes.ooe.fertig_executor import (
    FERTIG_EXECUTION_AUTHORITY_SHA256,
    FERTIG_EXECUTOR_SHA256,
    FERTIG_QUALITY_VERIFIER_SHA256,
    FertigExactExecutor,
)
from immer.runtimes.ooe.identity import canonical_json_bytes, require_sha256
from immer.runtimes.ooe.s3_executor import TransientPromptRegistry

from ooe_controller_language_bootstrap import (
    _existing_directory,
    _open_atlas,
    _output_directory,
    _stable_regular_bytes,
    _write_atomic_immutable,
)


REPORT_SCHEMA = "immer-ooe-fertig-action-learning-report/v1"
CONTROLLER_FORK_PROVENANCE_SCHEMA = "immer-ooe-controller-fork-provenance/v1"
MAX_INPUT_BYTES = 16 * 1024 * 1024
_REPORT_BODY_FIELDS = frozenset(
    {
        "action_authority_sha256",
        "atlas_head_sha256",
        "certified_question_count",
        "controller_snapshot_sha256",
        "execution_count",
        "format",
        "initial_source_action",
        "learning_head_sha256",
        "learning_stream_head_sha256",
        "model_pin_sha256",
        "qwen_forwards",
        "question_input_sha256",
        "rows",
        "source_controller_snapshot_sha256",
        "weight_graph_revision_sha256",
    }
)
_REPORT_ROW_FIELDS = frozenset(
    {
        "answer",
        "execution_sha256",
        "feature_receipt_sha256",
        "learning_receipt_sha256",
        "measurement_sha256",
        "question_sha256",
        "site_identity_sha256",
        "source_action",
        "target_action",
        "temporal_index",
        "trace_sha256",
    }
)


class FertigActionLearningCliError(RuntimeError):
    """The live FERTIG action-learning transaction cannot be authenticated."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _load_questions(path: Path) -> tuple[str, dict[str, str]]:
    try:
        data = _stable_regular_bytes(path, maximum=MAX_INPUT_BYTES)
        document = json.loads(data)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FertigActionLearningCliError("question input is invalid JSON") from exc
    if not isinstance(document, Mapping) or set(document) != {
        "body",
        "schema",
        "sha256",
    }:
        raise FertigActionLearningCliError("question input envelope is invalid")
    body = document.get("body")
    if not isinstance(body, Mapping) or _digest(body) != document.get("sha256"):
        raise FertigActionLearningCliError("question input seal is invalid")
    items = body.get("items")
    if not isinstance(items, list) or not items:
        raise FertigActionLearningCliError("question input has no items")
    questions: dict[str, str] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise FertigActionLearningCliError("question item is invalid")
        question = item.get("question")
        if (
            not isinstance(question, str)
            or not question
            or question != question.strip()
            or "\x00" in question
            or len(question.encode("utf-8")) > 64 * 1024
        ):
            raise FertigActionLearningCliError("question text is not canonical")
        question_sha256 = hashlib.sha256(question.encode("utf-8")).hexdigest()
        prior = questions.get(question_sha256)
        if prior is not None and prior != question:
            raise FertigActionLearningCliError("question SHA-256 collision")
        questions[question_sha256] = question
    return require_sha256(document["sha256"], field="question input sha256"), questions


def _load_report(path: Path) -> dict[str, Any]:
    try:
        data = _stable_regular_bytes(path, maximum=128 * 1024 * 1024)
        document = json.loads(data)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise FertigActionLearningCliError("existing report is invalid JSON") from exc
    if canonical_json_bytes(document) != data:
        raise FertigActionLearningCliError("existing report is not canonical JSON")
    if not isinstance(document, dict) or set(document) != {
        "body",
        "schema",
        "sha256",
    }:
        raise FertigActionLearningCliError("existing report envelope is invalid")
    body = document.get("body")
    if (
        document.get("schema") != REPORT_SCHEMA
        or not isinstance(body, dict)
        or set(body) != _REPORT_BODY_FIELDS
        or body.get("format") != REPORT_SCHEMA
        or document.get("sha256") != _digest(body)
    ):
        raise FertigActionLearningCliError("existing report seal is invalid")
    return document


def _target_store_path(value: str, *, label: str) -> Path:
    path = Path(value).expanduser().absolute()
    _output_directory(str(path.parent), label=f"{label} parent")
    try:
        path.lstat()
    except FileNotFoundError:
        return path
    return _existing_directory(str(path), label=label)


def _assert_planned_store_roots(*paths: Path) -> None:
    resolved = tuple(path.resolve(strict=False) for path in paths)
    for index, left in enumerate(resolved):
        for right in resolved[index + 1 :]:
            if (
                left == right
                or left in right.parents
                or right in left.parents
            ):
                raise FertigActionLearningCliError(
                    "planned store roots must be disjoint before directory creation"
                )


def _assert_disjoint_store_roots(*paths: Path) -> None:
    resolved_rows = []
    identities = []
    for path in paths:
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            resolved_rows.append(path.parent.resolve(strict=True) / path.name)
            identities.append(None)
        else:
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise FertigActionLearningCliError(
                    "managed store root must be a real directory"
                )
            resolved_rows.append(path.resolve(strict=True))
            identities.append((metadata.st_dev, metadata.st_ino))
    resolved = tuple(resolved_rows)
    for index, left in enumerate(resolved):
        for other in range(index + 1, len(resolved)):
            right = resolved[other]
            if (
                (
                    identities[index] is not None
                    and identities[index] == identities[other]
                )
                or left == right
                or left in right.parents
                or right in left.parents
            ):
                raise FertigActionLearningCliError(
                    "managed store roots must be physically disjoint"
                )


def _assert_output_outside_stores(output: Path, *roots: Path) -> None:
    output_real = output.parent.resolve(strict=True) / output.name
    for root in roots:
        root_real = root.resolve(strict=True)
        if output_real == root_real or root_real in output_real.parents:
            raise FertigActionLearningCliError(
                "output must not be stored inside a managed CrystalStore"
            )


def _assert_planned_output_outside_stores(output: Path, *roots: Path) -> None:
    output_real = output.resolve(strict=False)
    for root in roots:
        root_real = root.resolve(strict=False)
        if output_real == root_real or root_real in output_real.parents:
            raise FertigActionLearningCliError(
                "planned output must not be inside a managed store"
            )


def _fork_provenance_state_name(state_name: str) -> str:
    return (
        CONTROLLER_FORK_PROVENANCE_SCHEMA
        + ":"
        + hashlib.sha256(state_name.encode("utf-8")).hexdigest()
    )


def _fork_provenance(
    source: CrystalStore,
    *,
    state_name: str,
) -> bytes:
    source_state = source.restore_state(state_name)
    source_manifest = source.manifest()
    body = {
        "controller_state_name": state_name,
        "format": CONTROLLER_FORK_PROVENANCE_SCHEMA,
        "source_manifest_generation": source_manifest.generation,
        "source_manifest_sha256": source_manifest.sha256,
        "source_snapshot_sha256": hashlib.sha256(source_state).hexdigest(),
    }
    return canonical_json_bytes(
        {
            "body": body,
            "schema": CONTROLLER_FORK_PROVENANCE_SCHEMA,
            "sha256": _digest(body),
        }
    )


def _commit_fork_provenance(
    target: CrystalStore,
    provenance: bytes,
    *,
    state_name: str,
) -> None:
    provenance_name = _fork_provenance_state_name(state_name)
    try:
        prior = target.restore_state(provenance_name)
    except KeyError:
        publication = target.publish_state(provenance_name, provenance)
        if publication.payload_sha256 != hashlib.sha256(provenance).hexdigest():
            raise FertigActionLearningCliError(
                "controller fork provenance changed during publication"
            )
    else:
        if prior != provenance:
            raise FertigActionLearningCliError(
                "controller fork provenance belongs to another source"
            )
    if target.restore_state(provenance_name) != provenance:
        raise FertigActionLearningCliError(
            "controller fork provenance failed exact roundtrip"
        )


def _open_or_create_controller_fork(
    source: CrystalStore,
    target_root: Path,
    *,
    state_name: str,
) -> tuple[CrystalStore, bytes, bool]:
    provenance = _fork_provenance(source, state_name=state_name)
    provenance_body = json.loads(provenance)["body"]
    try:
        target_root.lstat()
    except FileNotFoundError:
        resume_exact_fork = True
    else:
        entries = set(path.name for path in target_root.iterdir())
        resume_exact_fork = (
            not entries
            or "IMMER-EXACT-FORK-INTENT" in entries
            or any(
                name.startswith(".IMMER-EXACT-FORK-INTENT.")
                and name.endswith(".tmp")
                for name in entries
            )
        )
    if resume_exact_fork:
        target = source.fork_exact(
            target_root,
            state_names=(state_name,),
            expected_manifest_sha256=provenance_body["source_manifest_sha256"],
            expected_state_sha256s={
                state_name: provenance_body["source_snapshot_sha256"]
            },
        )
    else:
        target = CrystalStore(target_root)
    provenance_name = _fork_provenance_state_name(state_name)
    try:
        prior = target.restore_state(provenance_name)
    except KeyError:
        source_state = source.restore_state(state_name)
        if (
            target.restore_state(state_name) == source_state
            and target.manifest().to_bytes() == source.manifest().to_bytes()
        ):
            _commit_fork_provenance(
                target,
                provenance,
                state_name=state_name,
            )
            return target, provenance, True
        return target, provenance, False
    if prior != provenance:
        raise FertigActionLearningCliError(
            "controller fork provenance belongs to another source"
        )
    return target, provenance, True


def _active_measurement(atlas, feature):
    result = atlas.query_by_prompt_signature(feature.probe.prompt_signature)
    matches = tuple(
        measurement
        for measurement in result.measurements
        if measurement.sha256 == feature.measurement_sha256
        and measurement.probe == feature.probe
        and measurement.coordinate.sha256 == feature.weight_coordinate_sha256
    )
    if len(matches) != 1:
        raise FertigActionLearningCliError(
            "source feature does not resolve to one active Atlas measurement"
        )
    return matches[0]


def _committed_controller_endpoint(head, source_snapshot_sha256: str) -> str:
    endpoint = require_sha256(
        source_snapshot_sha256,
        field="source_snapshot_sha256",
    )
    for ordinal, trace in enumerate(head.traces):
        if trace.ordinal != ordinal or trace.controller_snapshot_before_sha256 != (
            endpoint
        ):
            raise FertigActionLearningCliError(
                "execution-learning trace is not a descendant of the source fork"
            )
        endpoint = trace.controller_snapshot_after_sha256
    return endpoint


def _replay_controller_endpoint(
    source_store: CrystalStore,
    *,
    state_name: str,
    atlas_revision_verifier,
    model_pin_sha256: str,
    weight_graph_revision_sha256: str,
    head,
) -> OoeController:
    replay = OoeController.restore(
        crystal_store=source_store,
        name=state_name,
        atlas_revision_verifier=atlas_revision_verifier,
        expected_model_pin_sha256=model_pin_sha256,
        expected_weight_graph_revision_sha256=weight_graph_revision_sha256,
    )
    for receipt, trace in zip(head.receipts, head.traces, strict=True):
        if hashlib.sha256(replay.snapshot_bytes()).hexdigest() != (
            trace.controller_snapshot_before_sha256
        ):
            raise FertigActionLearningCliError(
                "learning trace before-snapshot failed controller replay"
            )
        replay.ingest_teacher(receipt.feature, receipt.transition)
        if hashlib.sha256(replay.snapshot_bytes()).hexdigest() != (
            trace.controller_snapshot_after_sha256
        ):
            raise FertigActionLearningCliError(
                "learning trace after-snapshot failed controller replay"
            )
    return replay


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fork a live Qwen/O1 controller and learn only exact, replayed "
            "FERTIG ActionExecution outcomes."
        )
    )
    parser.add_argument("--source-controller-store", required=True)
    parser.add_argument("--target-controller-store", required=True)
    parser.add_argument("--learning-store", required=True)
    parser.add_argument("--atlas-root", required=True)
    parser.add_argument("--question-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--controller-state-name", default=CONTROLLER_STATE_NAME)
    parser.add_argument("--expected-source-snapshot-sha256")
    parser.add_argument("--expected-model-pin-sha256")
    parser.add_argument("--expected-weight-graph-revision-sha256")
    return parser


def _optional_pin(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    try:
        return require_sha256(value, field=field)
    except ValueError as exc:
        raise FertigActionLearningCliError(f"{field} is not a SHA-256") from exc


def _assert_optional(actual: str, expected: str | None, *, field: str) -> None:
    if expected is not None and actual != expected:
        raise FertigActionLearningCliError(f"{field} is stale")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = _existing_directory(
        args.source_controller_store,
        label="source controller store",
    )
    atlas_root = _existing_directory(args.atlas_root, label="Atlas root")
    target_root = Path(args.target_controller_store).expanduser().absolute()
    learning_root = Path(args.learning_store).expanduser().absolute()
    output = Path(args.output).expanduser().absolute()
    _assert_planned_store_roots(
        source_root,
        target_root,
        learning_root,
        atlas_root,
    )
    _assert_planned_output_outside_stores(
        output,
        source_root,
        target_root,
        learning_root,
        atlas_root,
    )
    target_root = _target_store_path(
        str(target_root),
        label="target controller store",
    )
    learning_root = _output_directory(str(learning_root), label="learning store")
    _assert_disjoint_store_roots(
        source_root,
        target_root,
        learning_root,
        atlas_root,
    )
    question_path = Path(args.question_input).expanduser().absolute()
    _output_directory(str(output.parent), label="output parent")

    expected_source = _optional_pin(
        args.expected_source_snapshot_sha256,
        field="expected_source_snapshot_sha256",
    )
    expected_model = _optional_pin(
        args.expected_model_pin_sha256,
        field="expected_model_pin_sha256",
    )
    expected_weight = _optional_pin(
        args.expected_weight_graph_revision_sha256,
        field="expected_weight_graph_revision_sha256",
    )
    input_sha256, questions = _load_questions(question_path)
    atlas = _open_atlas(atlas_root)
    _assert_optional(atlas.model_pin.sha256, expected_model, field="model pin")
    membership = atlas.contains_revision
    source_store = CrystalStore(source_root)
    source_state = source_store.restore_state(args.controller_state_name)
    source_state_sha256 = hashlib.sha256(source_state).hexdigest()
    _assert_optional(source_state_sha256, expected_source, field="source snapshot")
    source_controller = OoeController.restore(
        crystal_store=source_store,
        name=args.controller_state_name,
        atlas_revision_verifier=membership,
        expected_model_pin_sha256=atlas.model_pin.sha256,
        expected_weight_graph_revision_sha256=expected_weight,
    )
    _assert_optional(
        source_controller.weight_graph_revision_sha256,
        expected_weight,
        field="weight graph revision",
    )
    initial_source = source_controller.last_teacher_action
    if initial_source is None:
        raise FertigActionLearningCliError(
            "source controller has no authenticated teacher history"
        )
    authority = ActionAuthorityReceipt(
        action="execute_fertig",
        model_pin_sha256=source_controller.model_pin_sha256,
        weight_graph_revision_sha256=(
            source_controller.weight_graph_revision_sha256
        ),
        executor_sha256=FERTIG_EXECUTOR_SHA256,
        action_verifier_name="fertig-exact-certificate",
        action_verifier_sha256=FERTIG_EXECUTION_AUTHORITY_SHA256,
        quality_verifier_name="fertig-exact-replay",
        quality_verifier_sha256=FERTIG_QUALITY_VERIFIER_SHA256,
        evidence_sha256=FERTIG_EXECUTION_AUTHORITY_SHA256,
    )
    target_store, fork_provenance, provenance_present = (
        _open_or_create_controller_fork(
            source_store,
            target_root,
            state_name=args.controller_state_name,
        )
    )
    _assert_disjoint_store_roots(
        source_root,
        target_root,
        learning_root,
        atlas_root,
    )
    _assert_output_outside_stores(
        output,
        source_root,
        target_root,
        learning_root,
        atlas_root,
    )
    target_store.clean_staging(expected_generation=target_store.manifest().generation)
    learning_store = CrystalStore(learning_root)
    learning_store.clean_staging(
        expected_generation=learning_store.manifest().generation
    )
    learning_bank = ExecutionLearningBank(learning_store)
    solver = FertigSolver()
    certified_questions = {
        sha256: (question, certificate)
        for sha256, question in sorted(questions.items())
        if (certificate := solver.certify(question)) is not None
    }
    if not certified_questions:
        raise FertigActionLearningCliError(
            "FERTIG did not certify any supplied question"
        )
    selected: list[tuple[Any, Any]] = []
    for question_sha256 in sorted(certified_questions):
        features = source_controller.feature_receipts_for_prompt(question_sha256)
        if not features:
            raise FertigActionLearningCliError(
                "certified question has no source-controller features"
            )
        for feature in features:
            measurement = _active_measurement(atlas, feature)
            selected.append((feature, measurement))
    selected.sort(key=lambda row: row[0].temporal_index)

    if output.exists() or output.is_symlink():
        report = _load_report(output)
        target_controller = OoeController.restore(
            crystal_store=target_store,
            name=args.controller_state_name,
            atlas_revision_verifier=membership,
            expected_model_pin_sha256=source_controller.model_pin_sha256,
            expected_weight_graph_revision_sha256=(
                source_controller.weight_graph_revision_sha256
            ),
        )
        persisted_state = target_store.restore_state(args.controller_state_name)
        head = learning_bank.head()
        if head.generation != len(selected):
            raise FertigActionLearningCliError(
                "stored learning head is not the complete certified frontier"
            )
        for receipt, (source_feature, measurement) in zip(
            head.receipts,
            selected,
            strict=True,
        ):
            if (
                receipt.measurement != measurement
                or receipt.action_authority != authority
                or receipt.feature.site_identity != source_feature.site_identity
                or receipt.feature.o1_surprise != source_feature.o1_surprise
                or receipt.feature.o1_learning_progress
                != source_feature.o1_learning_progress
                or receipt.feature.feature_dimensions
                != source_feature.feature_dimensions
                or receipt.execution.action != "execute_fertig"
            ):
                raise FertigActionLearningCliError(
                    "stored learning head differs from certified frontier"
                )
        replay = _replay_controller_endpoint(
            source_store,
            state_name=args.controller_state_name,
            atlas_revision_verifier=membership,
            model_pin_sha256=source_controller.model_pin_sha256,
            weight_graph_revision_sha256=(
                source_controller.weight_graph_revision_sha256
            ),
            head=head,
        )
        endpoint = _committed_controller_endpoint(head, source_state_sha256)
        if (
            hashlib.sha256(replay.snapshot_bytes()).hexdigest() != endpoint
            or hashlib.sha256(persisted_state).hexdigest() != endpoint
        ):
            raise FertigActionLearningCliError(
                "target controller is not the committed learning endpoint"
            )
        if not provenance_present:
            _commit_fork_provenance(
                target_store,
                fork_provenance,
                state_name=args.controller_state_name,
            )
        body = report["body"]
        rows = body.get("rows")
        if not isinstance(rows, list) or len(rows) != head.generation:
            raise FertigActionLearningCliError(
                "existing report row count differs from learning head"
            )
        if (
            body.get("action_authority_sha256") != authority.sha256
            or body.get("atlas_head_sha256") != atlas.revision().sha256
            or body.get("controller_snapshot_sha256")
            != hashlib.sha256(persisted_state).hexdigest()
            or body.get("execution_count") != head.generation
            or body.get("initial_source_action") != initial_source
            or body.get("learning_head_sha256") != head.sha256
            or body.get("learning_stream_head_sha256") != head.stream_head_sha256
            or body.get("model_pin_sha256") != target_controller.model_pin_sha256
            or body.get("question_input_sha256") != input_sha256
            or body.get("source_controller_snapshot_sha256")
            != source_state_sha256
            or body.get("weight_graph_revision_sha256")
            != target_controller.weight_graph_revision_sha256
            or target_controller.snapshot_bytes() != persisted_state
        ):
            raise FertigActionLearningCliError(
                "existing report differs from authenticated runtime state"
            )
        for ordinal, (row, receipt, trace) in enumerate(
            zip(rows, head.receipts, head.traces, strict=True)
        ):
            if not isinstance(row, Mapping) or set(row) != _REPORT_ROW_FIELDS:
                raise FertigActionLearningCliError(
                    "existing report contains an invalid row"
                )
            certificate = receipt.execution.result.get("certificate")
            if (
                not isinstance(certificate, Mapping)
                or receipt.action_authority != authority
                or row.get("answer") != certificate.get("answer")
                or row.get("execution_sha256") != receipt.execution.sha256
                or row.get("feature_receipt_sha256") != receipt.feature.sha256
                or row.get("learning_receipt_sha256") != receipt.sha256
                or row.get("measurement_sha256") != receipt.measurement.sha256
                or row.get("question_sha256")
                != receipt.feature.probe.question_sha256
                or row.get("site_identity_sha256")
                != receipt.feature.site_identity.sha256
                or row.get("source_action") != receipt.transition.source_action
                or row.get("target_action") != receipt.transition.target_action
                or row.get("temporal_index") != receipt.feature.temporal_index
                or row.get("trace_sha256") != trace.sha256
                or trace.ordinal != ordinal
            ):
                raise FertigActionLearningCliError(
                    "existing report row differs from learning evidence"
                )
            if row.get("question_sha256") not in questions:
                raise FertigActionLearningCliError(
                    "existing report references an unknown question identity"
                )
        if body.get("qwen_forwards") != sum(
            receipt.execution.qwen_forwards for receipt in head.receipts
        ) or body.get("certified_question_count") != len(certified_questions):
            raise FertigActionLearningCliError(
                "existing report Qwen accounting differs"
            )
        return report

    target_controller = OoeController.restore(
        crystal_store=target_store,
        name=args.controller_state_name,
        atlas_revision_verifier=membership,
        expected_model_pin_sha256=atlas.model_pin.sha256,
        expected_weight_graph_revision_sha256=(
            source_controller.weight_graph_revision_sha256
        ),
    )
    registry = TransientPromptRegistry()
    executor = FertigExactExecutor(solver, registry)

    def execute(feature):
        try:
            question = certified_questions[feature.probe.question_sha256][0]
        except KeyError as exc:
            raise FertigActionLearningCliError(
                "executor received an unregistered question identity"
            ) from exc
        registry.bind(feature, question)
        return executor(feature)

    bridge = ExecutionLearningBridge(
        controller=target_controller,
        atlas=atlas,
        bank=learning_bank,
        action_authorities={"execute_fertig": authority},
        action_executors={
            "execute_fertig": (FERTIG_EXECUTOR_SHA256, execute),
        },
        quality_verifiers={
            "fertig-exact-replay": (
                FERTIG_QUALITY_VERIFIER_SHA256,
                executor.verify,
            ),
        },
        initial_source_action=initial_source,
        controller_state_name=args.controller_state_name,
    )
    head_before = learning_bank.head()
    if head_before.generation > len(selected):
        raise FertigActionLearningCliError(
            "learning history is longer than the selected execution frontier"
        )
    if head_before.traces and head_before.traces[0].source_action != initial_source:
        raise FertigActionLearningCliError(
            "learning history starts from another source action"
        )
    for receipt, (source_feature, measurement) in zip(
        head_before.receipts,
        selected,
        strict=False,
    ):
        if (
            receipt.measurement != measurement
            or receipt.action_authority != authority
            or receipt.feature.site_identity != source_feature.site_identity
            or receipt.feature.o1_surprise != source_feature.o1_surprise
            or receipt.feature.o1_learning_progress
            != source_feature.o1_learning_progress
            or receipt.feature.feature_dimensions != source_feature.feature_dimensions
            or receipt.execution.action != "execute_fertig"
        ):
            raise FertigActionLearningCliError(
                "learning history is not the selected execution prefix"
            )
    committed_endpoint = _committed_controller_endpoint(
        head_before,
        source_state_sha256,
    )
    replay = _replay_controller_endpoint(
        source_store,
        state_name=args.controller_state_name,
        atlas_revision_verifier=membership,
        model_pin_sha256=source_controller.model_pin_sha256,
        weight_graph_revision_sha256=(
            source_controller.weight_graph_revision_sha256
        ),
        head=head_before,
    )
    if hashlib.sha256(replay.snapshot_bytes()).hexdigest() != committed_endpoint:
        raise FertigActionLearningCliError(
            "committed controller endpoint differs from exact replay"
        )
    target_endpoint = hashlib.sha256(
        target_store.restore_state(args.controller_state_name)
    ).hexdigest()
    if target_endpoint != committed_endpoint:
        next_index = head_before.generation
        if next_index >= len(selected):
            raise FertigActionLearningCliError(
                "target controller is ahead of a complete learning history"
            )
        source_feature, measurement = selected[next_index]
        next_source = (
            initial_source
            if not head_before.traces
            else head_before.traces[-1].target_action
        )
        next_temporal = (
            source_controller.last_temporal_index + 1
            if not head_before.traces
            else head_before.traces[-1].temporal_index + 1
        )
        transaction_sha256 = bridge._transaction_sha256(
            measurement=measurement,
            authority=authority,
            temporal_index=next_temporal,
            source_action=next_source,
            o1_surprise=source_feature.o1_surprise,
            o1_learning_progress=source_feature.o1_learning_progress,
            dimensions=source_feature.feature_dimensions,
            weight=1.0,
        )
        pending = learning_bank.intent(transaction_sha256)
        if (
            pending is None
            or pending.bank_head_before_sha256 != head_before.sha256
            or pending.trace.controller_snapshot_before_sha256
            != committed_endpoint
            or pending.trace.controller_snapshot_after_sha256 != target_endpoint
            or pending.receipt.measurement != measurement
            or pending.receipt.action_authority != authority
        ):
            raise FertigActionLearningCliError(
                "target controller is not one recoverable intent ahead"
            )
        replay.ingest_teacher(pending.receipt.feature, pending.receipt.transition)
        if hashlib.sha256(replay.snapshot_bytes()).hexdigest() != target_endpoint:
            raise FertigActionLearningCliError(
                "recoverable intent does not derive the target controller"
            )
    if not provenance_present:
        _commit_fork_provenance(
            target_store,
            fork_provenance,
            state_name=args.controller_state_name,
        )
    learned = []
    for source_feature, measurement in selected:
        receipt = bridge.learn_from_execution(
            measurement,
            action="execute_fertig",
            o1_surprise=source_feature.o1_surprise,
            o1_learning_progress=source_feature.o1_learning_progress,
            dimensions=source_feature.feature_dimensions,
        )
        if receipt is None:
            raise FertigActionLearningCliError(
                "pre-certified FERTIG execution failed exact replay"
            )
        if receipt.feature.site_identity != source_feature.site_identity:
            raise FertigActionLearningCliError(
                "authority enrichment changed the source weight site"
            )
        learned.append(receipt)

    final_controller = bridge.controller
    final_state = final_controller.snapshot_bytes()
    persisted_state = target_store.restore_state(args.controller_state_name)
    if final_state != persisted_state:
        raise FertigActionLearningCliError(
            "final controller differs from its persistent snapshot"
        )
    head = learning_bank.head()
    if head.generation != len(learned):
        raise FertigActionLearningCliError(
            "learning trace generation differs from selected executions"
        )
    if _committed_controller_endpoint(head, source_state_sha256) != hashlib.sha256(
        persisted_state
    ).hexdigest():
        raise FertigActionLearningCliError(
            "final controller does not equal the committed trace endpoint"
        )
    rows = []
    for receipt in learned:
        certificate = receipt.execution.result.get("certificate")
        if not isinstance(certificate, Mapping):
            raise FertigActionLearningCliError(
                "learned FERTIG execution lost its certificate"
            )
        rows.append(
            {
                "answer": certificate.get("answer"),
                "execution_sha256": receipt.execution.sha256,
                "feature_receipt_sha256": receipt.feature.sha256,
                "learning_receipt_sha256": receipt.sha256,
                "measurement_sha256": receipt.measurement.sha256,
                "question_sha256": receipt.feature.probe.question_sha256,
                "site_identity_sha256": receipt.feature.site_identity.sha256,
                "source_action": receipt.transition.source_action,
                "target_action": receipt.transition.target_action,
                "temporal_index": receipt.feature.temporal_index,
                "trace_sha256": head.traces[len(rows)].sha256,
            }
        )
    body = {
        "action_authority_sha256": authority.sha256,
        "atlas_head_sha256": atlas.revision().sha256,
        "certified_question_count": len(certified_questions),
        "controller_snapshot_sha256": hashlib.sha256(final_state).hexdigest(),
        "execution_count": len(rows),
        "format": REPORT_SCHEMA,
        "initial_source_action": initial_source,
        "learning_head_sha256": head.sha256,
        "learning_stream_head_sha256": head.stream_head_sha256,
        "model_pin_sha256": final_controller.model_pin_sha256,
        "qwen_forwards": sum(row.execution.qwen_forwards for row in learned),
        "question_input_sha256": input_sha256,
        "rows": rows,
        "source_controller_snapshot_sha256": source_state_sha256,
        "weight_graph_revision_sha256": (
            final_controller.weight_graph_revision_sha256
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
        raise SystemExit(f"ooe_fertig_action_learning: {exc}") from exc
    print(report["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
