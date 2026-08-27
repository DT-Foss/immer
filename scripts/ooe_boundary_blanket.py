#!/usr/bin/env python3
"""Discover and evaluate the real Qwen layer-boundary Markov blanket."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
from typing import Sequence, cast

from immer.runtimes.ooe.boundary_blanket import (
    BOUNDARY_CANDIDATE_SCHEMA_SHA256,
    BoundaryBlanketFit,
    BoundaryCorpus,
    BoundaryFitConfig,
    evaluate_boundary_blanket_holdout,
    evaluate_boundary_blanket_loqo,
    fit_boundary_blanket,
    verify_boundary_blanket_holdout,
    verify_boundary_blanket_loqo,
)
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.operator_harvester import (
    HARVESTER_STATE_PREFIX,
    HarvesterState,
)


REPORT_SCHEMA = "immer-ooe-qwen-boundary-blanket-report/v4"
REPORT_VERIFICATION_SCHEMA = "immer-ooe-qwen-boundary-report-verification/v1"
MAX_STATE_ENVELOPE_BYTES = 96 * 1024 * 1024


class CliError(RuntimeError):
    """The requested persisted-state evaluation is invalid."""


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return result


def _strict_state_envelope(path: Path) -> dict[str, object]:
    metadata = path.lstat()
    if path.is_symlink() or not path.is_file():
        raise CliError(f"state entry is not a regular file: {path}")
    if not 0 < metadata.st_size <= MAX_STATE_ENVELOPE_BYTES:
        raise CliError(f"state entry exceeds its byte bound: {path}")
    before = (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
    data = path.read_bytes()
    after_metadata = path.lstat()
    after = (
        after_metadata.st_dev,
        after_metadata.st_ino,
        after_metadata.st_size,
        after_metadata.st_mtime_ns,
    )
    if before != after or len(data) != metadata.st_size:
        raise CliError(f"state entry changed while being read: {path}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise CliError(f"state entry is not strict JSON: {path}") from exc
    if not isinstance(value, dict) or set(value) != {
        "format",
        "generation",
        "name",
        "payload_base64",
        "payload_sha256",
    }:
        raise CliError(f"state entry envelope is invalid: {path}")
    name = value.get("name")
    if not isinstance(name, str):
        raise CliError(f"state entry name is invalid: {path}")
    expected_filename = f"{hashlib.sha256(name.encode('utf-8')).hexdigest()}.state"
    if path.name != expected_filename:
        raise CliError(f"state entry filename does not bind its name: {path}")
    return cast(dict[str, object], value)


def discover_harvester_state_name(compute_root: Path) -> str:
    """Strictly discover the one default harvester state in a CrystalStore."""

    state_root = compute_root / "state"
    if state_root.is_symlink() or not state_root.is_dir():
        raise CliError("compute root has no real state directory")
    names = []
    for path in sorted(state_root.glob("*.state")):
        envelope = _strict_state_envelope(path)
        name = envelope["name"]
        if isinstance(name, str) and name.startswith(HARVESTER_STATE_PREFIX):
            names.append(name)
    if len(names) != 1:
        raise CliError(
            "automatic discovery requires exactly one operator harvester state; "
            "pass --harvester-state-name"
        )
    return names[0]


def load_corpus(
    compute_root: Path,
    *,
    harvester_state_name: str | None,
    expected_harvester_identity_sha256: str | None,
    expected_model_pin_sha256: str | None,
    expected_latest_atlas_revision_sha256: str | None,
) -> tuple[str, BoundaryCorpus]:
    store = CrystalStore(compute_root)
    name = (
        discover_harvester_state_name(compute_root)
        if harvester_state_name is None
        else harvester_state_name
    )
    payload = store.restore_state(name)
    state = HarvesterState.from_bytes(payload)
    if name != f"{HARVESTER_STATE_PREFIX}{state.identity_sha256}":
        raise CliError("harvester state name differs from its authenticated identity")
    corpus = BoundaryCorpus.from_harvester_state(
        state,
        expected_harvester_identity_sha256=expected_harvester_identity_sha256,
        expected_model_pin_sha256=expected_model_pin_sha256,
        expected_latest_atlas_revision_sha256=expected_latest_atlas_revision_sha256,
    )
    return name, corpus


def _build_report_unverified(
    corpus: BoundaryCorpus,
    *,
    harvester_state_name: str,
    train_end: int = 25,
    calibration_end: int = 35,
    holdout_end: int = 40,
    config: BoundaryFitConfig | None = None,
    loqo: bool = False,
) -> dict[str, object]:
    """Fit the chronological split and evaluate its untouched final holdout."""

    if not isinstance(corpus, BoundaryCorpus):
        raise TypeError("corpus must be a BoundaryCorpus")
    if not isinstance(harvester_state_name, str) or not harvester_state_name:
        raise ValueError("harvester_state_name must be non-empty")
    if not 0 < train_end < calibration_end < holdout_end:
        raise ValueError("split boundaries must be strictly increasing")
    if holdout_end != len(corpus.macros):
        raise ValueError(
            "holdout_end must consume the complete authenticated macro chronology"
        )
    selected_config = BoundaryFitConfig() if config is None else config
    fit = fit_boundary_blanket(
        corpus,
        train_indices=tuple(range(train_end)),
        calibration_indices=tuple(range(train_end, calibration_end)),
        config=selected_config,
    )
    holdout = evaluate_boundary_blanket_holdout(
        fit,
        corpus,
        holdout_indices=tuple(range(calibration_end, holdout_end)),
    )
    loqo_report = (
        evaluate_boundary_blanket_loqo(corpus, config=selected_config) if loqo else None
    )
    evaluation_kinds = ["strict_topology_holdout"]
    if loqo_report is not None:
        evaluation_kinds.append("loqo_prompt_audit")
    body = {
        "candidate_schema_sha256": BOUNDARY_CANDIDATE_SCHEMA_SHA256,
        "corpus_sha256": corpus.sha256,
        "fit": fit.to_document(),
        "fit_sha256": fit.sha256,
        "harvester_identity_sha256": corpus.harvester_identity_sha256,
        "harvester_state_name": harvester_state_name,
        "harvester_state_sha256": corpus.harvester_state_sha256,
        "evaluation_kinds": evaluation_kinds,
        "strict_topology_holdout": holdout,
        "strict_topology_holdout_sha256": holdout["body_sha256"],
        "loqo_prompt_audit": loqo_report,
        "loqo_prompt_audit_sha256": (
            None if loqo_report is None else loqo_report["body_sha256"]
        ),
        "macro_count": len(corpus.macros),
        "model_pin_sha256": corpus.model_pin_sha256,
        "split": {
            "calibration": [train_end, calibration_end],
            "holdout": [calibration_end, holdout_end],
            "train": [0, train_end],
        },
        "verification": {
            "full_recomputation": True,
            "schema": REPORT_VERIFICATION_SCHEMA,
            "verified_sections": evaluation_kinds,
        },
    }
    return {
        "body": body,
        "body_sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
        "schema": REPORT_SCHEMA,
    }


def verify_report(
    corpus: BoundaryCorpus,
    document: dict[str, object],
) -> None:
    """Recompute the fit and every labeled evaluation in a CLI report."""

    if not isinstance(corpus, BoundaryCorpus):
        raise TypeError("corpus must be a BoundaryCorpus")
    if (
        not isinstance(document, dict)
        or set(document) != {"body", "body_sha256", "schema"}
        or document.get("schema") != REPORT_SCHEMA
        or not isinstance(document.get("body"), dict)
        or document.get("body_sha256")
        != hashlib.sha256(canonical_json_bytes(document.get("body"))).hexdigest()
    ):
        raise CliError("boundary report envelope is invalid")
    body = cast(dict[str, object], document["body"])
    expected_fields = {
        "candidate_schema_sha256",
        "corpus_sha256",
        "evaluation_kinds",
        "fit",
        "fit_sha256",
        "harvester_identity_sha256",
        "harvester_state_name",
        "harvester_state_sha256",
        "loqo_prompt_audit",
        "loqo_prompt_audit_sha256",
        "macro_count",
        "model_pin_sha256",
        "split",
        "strict_topology_holdout",
        "strict_topology_holdout_sha256",
        "verification",
    }
    if set(body) != expected_fields:
        raise CliError("boundary report body is invalid")
    split = body.get("split")
    if not isinstance(split, dict) or set(split) != {
        "calibration",
        "holdout",
        "train",
    }:
        raise CliError("boundary report split is invalid")
    train = split.get("train")
    calibration = split.get("calibration")
    holdout_range = split.get("holdout")
    if not all(
        isinstance(row, list)
        and len(row) == 2
        and all(isinstance(value, int) and not isinstance(value, bool) for value in row)
        for row in (train, calibration, holdout_range)
    ):
        raise CliError("boundary report split boundaries are invalid")
    assert isinstance(train, list)
    assert isinstance(calibration, list)
    assert isinstance(holdout_range, list)
    train_end = train[1]
    calibration_end = calibration[1]
    holdout_end = holdout_range[1]
    if (
        train != [0, train_end]
        or calibration != [train_end, calibration_end]
        or holdout_range != [calibration_end, holdout_end]
    ):
        raise CliError("boundary report split is not contiguous chronology")
    fit_document = body.get("fit")
    if not isinstance(fit_document, dict):
        raise CliError("boundary report fit is invalid")
    fit = BoundaryBlanketFit.from_bytes(
        canonical_json_bytes(fit_document), corpus=corpus
    )
    if body.get("fit_sha256") != fit.sha256:
        raise CliError("boundary report fit hash is invalid")
    strict_holdout = body.get("strict_topology_holdout")
    if not isinstance(strict_holdout, dict):
        raise CliError("strict topology holdout is invalid")
    verify_boundary_blanket_holdout(strict_holdout, fit=fit, corpus=corpus)
    strict_body = strict_holdout.get("body")
    if (
        not isinstance(strict_body, dict)
        or strict_body.get("holdout_kind") != "temporal_chronological"
        or strict_body.get("topology_holdout") is not True
    ):
        raise CliError("strict topology holdout label is not proven")
    if body.get("strict_topology_holdout_sha256") != strict_holdout.get("body_sha256"):
        raise CliError("strict topology holdout hash is invalid")
    loqo_document = body.get("loqo_prompt_audit")
    loqo_enabled = loqo_document is not None
    if loqo_enabled:
        if not isinstance(loqo_document, dict):
            raise CliError("LOQO prompt audit is invalid")
        verify_boundary_blanket_loqo(corpus, loqo_document, fit.config)
        if body.get("loqo_prompt_audit_sha256") != loqo_document.get("body_sha256"):
            raise CliError("LOQO prompt audit hash is invalid")
    elif body.get("loqo_prompt_audit_sha256") is not None:
        raise CliError("absent LOQO audit carries a hash")
    expected_kinds = ["strict_topology_holdout"] + (
        ["loqo_prompt_audit"] if loqo_enabled else []
    )
    if body.get("evaluation_kinds") != expected_kinds or body.get("verification") != {
        "full_recomputation": True,
        "schema": REPORT_VERIFICATION_SCHEMA,
        "verified_sections": expected_kinds,
    }:
        raise CliError("boundary report verification inventory is invalid")
    expected = _build_report_unverified(
        corpus,
        harvester_state_name=cast(str, body.get("harvester_state_name")),
        train_end=train_end,
        calibration_end=calibration_end,
        holdout_end=holdout_end,
        config=fit.config,
        loqo=loqo_enabled,
    )
    if canonical_json_bytes(expected) != canonical_json_bytes(document):
        raise CliError("boundary report differs from complete recomputation")


def build_report(
    corpus: BoundaryCorpus,
    *,
    harvester_state_name: str,
    train_end: int = 25,
    calibration_end: int = 35,
    holdout_end: int = 40,
    config: BoundaryFitConfig | None = None,
    loqo: bool = False,
) -> dict[str, object]:
    """Build and fully recompute-verify the complete labeled report."""

    report = _build_report_unverified(
        corpus,
        harvester_state_name=harvester_state_name,
        train_end=train_end,
        calibration_end=calibration_end,
        holdout_end=holdout_end,
        config=config,
        loqo=loqo,
    )
    verify_report(corpus, report)
    return report


def write_new_atomic(path: Path, payload: bytes) -> None:
    """Publish bytes atomically while refusing to replace any existing target."""

    if not isinstance(payload, bytes):
        raise TypeError("payload must be immutable bytes")
    path = path.absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise CliError("output parent must be a real directory")
    temporary = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(16)}.tmp"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(fd, view[offset:])
            if written <= 0:
                raise OSError("short write while publishing report")
            offset += written
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute-root", required=True)
    parser.add_argument("--harvester-state-name")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-end", type=_positive_int, default=25)
    parser.add_argument("--calibration-end", type=_positive_int, default=35)
    parser.add_argument("--holdout-end", type=_positive_int, default=40)
    parser.add_argument(
        "--selection-tolerance",
        type=_nonnegative_float,
        default=1e-10,
    )
    parser.add_argument(
        "--ridge",
        type=_nonnegative_float,
        action="append",
        dest="ridge_grid",
    )
    parser.add_argument("--normalization-floor", type=_positive_float, default=1e-12)
    parser.add_argument("--max-condition-number", type=_positive_float, default=1e12)
    parser.add_argument("--max-working-mb", type=_positive_float, default=1024.0)
    parser.add_argument("--expected-harvester-identity-sha256")
    parser.add_argument("--expected-model-pin-sha256")
    parser.add_argument("--expected-latest-atlas-revision-sha256")
    parser.add_argument("--loqo", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.compute_root).expanduser().absolute()
    state_name, corpus = load_corpus(
        root,
        harvester_state_name=args.harvester_state_name,
        expected_harvester_identity_sha256=args.expected_harvester_identity_sha256,
        expected_model_pin_sha256=args.expected_model_pin_sha256,
        expected_latest_atlas_revision_sha256=(
            args.expected_latest_atlas_revision_sha256
        ),
    )
    ridge_grid = (
        (0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
        if args.ridge_grid is None
        else tuple(sorted(set(args.ridge_grid)))
    )
    config = BoundaryFitConfig(
        selection_tolerance=args.selection_tolerance,
        ridge_grid=ridge_grid,
        normalization_floor=args.normalization_floor,
        max_condition_number=args.max_condition_number,
        max_working_bytes=int(args.max_working_mb * 1024**2),
    )
    report = build_report(
        corpus,
        harvester_state_name=state_name,
        train_end=args.train_end,
        calibration_end=args.calibration_end,
        holdout_end=args.holdout_end,
        config=config,
        loqo=args.loqo,
    )
    payload = canonical_json_bytes(report)
    write_new_atomic(Path(args.output).expanduser(), payload)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
