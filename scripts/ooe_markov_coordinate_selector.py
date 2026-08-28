#!/usr/bin/env python3
"""Fit and evaluate the sealed Markov coordinate agent on one MLP bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Mapping, Sequence

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.markov_coordinate_selector import (
    MARKOV_SELECTOR_VERIFIER_SHA256,
    MarkovCoordinateSelectorConfig,
    MarkovCoordinateSelectorEvaluation,
    MarkovCoordinateSelectorFit,
    exact_output_reuse_ceiling,
    evaluate_markov_coordinate_selector,
    fit_markov_coordinate_selector,
)
from immer.runtimes.ooe.prompt_row_roles import derive_prompt_row_roles
from immer.runtimes.ooe.qwen_mlp_evidence import QwenMlpEvidenceBank


REPORT_SCHEMA = "immer.qwen-markov-coordinate-live-report/v1"
FIT_NAME = "fit.json"
HOLDOUT_NAME = "holdout.json"
REPORT_NAME = "report.json"
FIT_ROW_ROLES_NAME = "fit-row-roles.json"
HOLDOUT_ROW_ROLES_NAME = "holdout-row-roles.json"
REUSE_CEILING_NAME = "reuse-ceiling.json"


class CliError(RuntimeError):
    pass


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _stable_read(path: Path, maximum: int = 64 * 1024 * 1024) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise CliError(f"invalid bounded artifact: {path}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > maximum:
                raise CliError(f"artifact exceeds its bound: {path}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise CliError(f"artifact changed while read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _persist_exact(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or _stable_read(path, max(len(data), 1)) != data:
            raise CliError(f"sealed artifact changed: {path}")
        return
    temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
    descriptor = os.open(
        temporary,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short write")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError:
        if _stable_read(path, max(len(data), 1)) != data:
            raise CliError(f"sealed artifact collided: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def _config(args: argparse.Namespace) -> MarkovCoordinateSelectorConfig:
    return MarkovCoordinateSelectorConfig(
        quant_bits=args.quant_bits,
        max_depth=args.max_depth,
        beam_width=args.beam_width,
        internal_validation_groups=args.internal_validation_groups,
        energy_candidates=args.energy_candidates,
        fisher_candidates=args.fisher_candidates,
        variance_candidates=args.variance_candidates,
        random_candidates=args.random_candidates,
        max_candidate_pool=args.max_candidate_pool,
        max_evaluated_states=args.max_evaluated_states,
        minimum_raw_key_bits=args.minimum_raw_key_bits,
        max_working_bytes=int(args.max_working_gb * 1024**3),
    )


def _prompt_tokens(path: Path, prompts: set[str]) -> dict[str, tuple[int, ...]]:
    try:
        document = json.loads(_stable_read(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError(f"prompt registry is not JSON: {path}") from exc
    if not isinstance(document, dict):
        raise CliError(f"prompt registry root is invalid: {path}")
    body = document.get("body", document)
    if not isinstance(body, dict) or not isinstance(body.get("prompts"), list):
        raise CliError(f"prompt registry is absent: {path}")
    rows = {}
    for row in body["prompts"]:
        if not isinstance(row, dict) or row.get("sha256") not in prompts:
            continue
        tokens = row.get("token_ids")
        if not isinstance(tokens, list):
            raise CliError(f"prompt token row is invalid: {path}")
        rows[row["sha256"]] = tuple(tokens)
    if set(rows) != prompts:
        raise CliError(f"prompt registry does not cover the selected bank: {path}")
    return rows


def _authority(
    fit_bank: QwenMlpEvidenceBank,
    holdout_bank: QwenMlpEvidenceBank,
    holdout_groups: Sequence[object],
) -> str:
    fit_state = fit_bank.state()
    holdout_state = holdout_bank.state()
    return _digest(
        {
            "fit_bank_state_sha256": fit_state.sha256,
            "holdout_group_sha256s": [
                getattr(group, "group_sha256") for group in holdout_groups
            ],
            "holdout_bank_state_sha256": holdout_state.sha256,
            "schema": "immer.qwen-markov-coordinate-holdout-authority/v1",
            "holdout_split_counts": list(holdout_state.split_counts),
        }
    )


def _report(
    *,
    fit_bank: QwenMlpEvidenceBank,
    holdout_bank: QwenMlpEvidenceBank,
    fit: MarkovCoordinateSelectorFit,
    evaluation: MarkovCoordinateSelectorEvaluation,
    calibration_reuse_ceiling: Mapping[str, object],
    fit_row_role_sha256: str | None,
    holdout_row_role_sha256: str | None,
) -> dict[str, object]:
    by_name = {result.name: result for result in evaluation.results}
    locked = fit.locked_model
    body = {
        "adaptive_metrics": {
            name: result.adaptive_metrics.to_record()
            for name, result in sorted(by_name.items())
        },
        "external_holdout": fit_bank.root != holdout_bank.root,
        "fit_bank_state_sha256": fit_bank.state().sha256,
        "beam_state_count": len(fit.beam_states),
        "calibration_metrics": locked.calibration_metrics.to_record(),
        "calibration_reuse_ceiling": dict(calibration_reuse_ceiling),
        "candidate_pool_size": len(fit.nomination.candidate_pool),
        "config_sha256": fit.config.sha256,
        "evaluated_state_count": fit.evaluated_state_count,
        "fit_sha256": fit.sha256,
        "fit_row_role_sha256": fit_row_role_sha256,
        "frozen_metrics": {
            name: result.frozen_metrics.to_record()
            for name, result in sorted(by_name.items())
        },
        "holdout_authority_sha256": evaluation.holdout_authority_sha256,
        "holdout_bank_state_sha256": holdout_bank.state().sha256,
        "holdout_sha256": evaluation.sha256,
        "holdout_row_role_sha256": holdout_row_role_sha256,
        "locked_basis_indices": list(locked.basis_indices),
        "locked_k": locked.k,
        "locked_quant_bits": locked.quant_bits,
        "locked_raw_key_bits": 2 * locked.k * locked.quant_bits,
        "model_pin_sha256": fit.model_pin_sha256,
        "promoted": by_name["markov"].promoted,
        "selector_verifier_sha256": MARKOV_SELECTOR_VERIFIER_SHA256,
    }
    return {
        "body": body,
        "body_sha256": _digest(body),
        "schema": REPORT_SCHEMA,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    fit_bank = QwenMlpEvidenceBank(
        Path(args.bank_root).expanduser().absolute(),
        deferred_tensor_splits=("holdout",),
    )
    if fit_bank.state().split_counts != (25, 10, 5):
        raise CliError("selector requires one complete 25/10/5 MLP bank")
    fit_roles = None
    if args.row_mode == "content":
        if args.fit_prompt_registry is None:
            raise CliError("content row mode requires --fit-prompt-registry")
        fit_prompts = {
            receipt.entry.prompt_sha256
            for receipt, _verification in fit_bank.committed_pairs()
        }
        fit_roles = derive_prompt_row_roles(
            _prompt_tokens(
                Path(args.fit_prompt_registry).expanduser().absolute(), fit_prompts
            )
        )
    fit_corpus = fit_bank.build_subspace_corpus(
        allowed_splits=("train", "calibration"),
        row_indices_by_prompt=(
            None if fit_roles is None else fit_roles.row_indices_by_prompt
        ),
    )
    train_indices = tuple(range(25))
    calibration_indices = tuple(range(25, 35))
    config = _config(args)
    train = fit_corpus.groups[:25]
    calibration = fit_corpus.groups[25:35]
    ceiling = exact_output_reuse_ceiling(train, calibration)
    output = Path(args.output_root).expanduser().absolute()
    if fit_roles is not None:
        _persist_exact(output / FIT_ROW_ROLES_NAME, fit_roles.to_bytes())
    _persist_exact(
        output / REUSE_CEILING_NAME,
        canonical_json_bytes(ceiling.to_record()) + b"\n",
    )
    if ceiling.adaptive_exact_hits == 0:
        body = {
            "calibration_reuse_ceiling": ceiling.to_record(),
            "fit_bank_state_sha256": fit_bank.state().sha256,
            "fit_row_role_sha256": (None if fit_roles is None else fit_roles.sha256),
            "model_pin_sha256": fit_corpus.model_pin_sha256,
            "selector_verifier_sha256": MARKOV_SELECTOR_VERIFIER_SHA256,
            "status": "no_exact_output_reuse_signal",
        }
        report = {
            "body": body,
            "body_sha256": _digest(body),
            "schema": REPORT_SCHEMA,
        }
        _persist_exact(output / REPORT_NAME, canonical_json_bytes(report) + b"\n")
        return report
    fit = fit_markov_coordinate_selector(
        fit_corpus,
        train_group_indices=train_indices,
        calibration_group_indices=calibration_indices,
        config=config,
    )
    _persist_exact(output / FIT_NAME, fit.to_bytes())
    # Only after the fit bytes are durable may either holdout bank be opened.
    if not fit_bank.audit().clean:
        raise CliError("selector fit bank failed its post-lock full payload audit")
    holdout_bank = (
        fit_bank
        if args.external_holdout_bank_root is None
        else QwenMlpEvidenceBank(
            Path(args.external_holdout_bank_root).expanduser().absolute()
        )
    )
    if (
        holdout_bank.state().split_counts != (25, 10, 5)
        or not holdout_bank.audit().clean
    ):
        raise CliError("external holdout bank is not a clean complete 25/10/5 bank")
    holdout_roles = None
    if args.row_mode == "content":
        registry_path = (
            args.fit_prompt_registry
            if args.external_holdout_bank_root is None
            else args.holdout_prompt_registry
        )
        if registry_path is None:
            raise CliError(
                "external content holdout requires --holdout-prompt-registry"
            )
        holdout_prompts = {
            receipt.entry.prompt_sha256
            for receipt, _verification in holdout_bank.committed_pairs()
        }
        holdout_roles = derive_prompt_row_roles(
            _prompt_tokens(Path(registry_path).expanduser().absolute(), holdout_prompts)
        )
        _persist_exact(output / HOLDOUT_ROW_ROLES_NAME, holdout_roles.to_bytes())
    holdout_corpus = holdout_bank.build_subspace_corpus(
        allowed_splits=("holdout",),
        row_indices_by_prompt=(
            None if holdout_roles is None else holdout_roles.row_indices_by_prompt
        ),
    )
    authority = _authority(fit_bank, holdout_bank, holdout_corpus.groups)
    evaluation = evaluate_markov_coordinate_selector(
        fit,
        fit_corpus=fit_corpus,
        holdout_corpus=holdout_corpus,
        holdout_authority_sha256=authority,
    )
    _persist_exact(output / HOLDOUT_NAME, evaluation.to_bytes())
    report = _report(
        fit_bank=fit_bank,
        holdout_bank=holdout_bank,
        fit=fit,
        evaluation=evaluation,
        calibration_reuse_ceiling=ceiling.to_record(),
        fit_row_role_sha256=None if fit_roles is None else fit_roles.sha256,
        holdout_row_role_sha256=(
            None if holdout_roles is None else holdout_roles.sha256
        ),
    )
    _persist_exact(output / REPORT_NAME, canonical_json_bytes(report) + b"\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--external-holdout-bank-root")
    parser.add_argument("--row-mode", choices=("all", "content"), default="all")
    parser.add_argument("--fit-prompt-registry")
    parser.add_argument("--holdout-prompt-registry")
    parser.add_argument("--quant-bits", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--internal-validation-groups", type=int, default=5)
    parser.add_argument("--energy-candidates", type=int, default=96)
    parser.add_argument("--fisher-candidates", type=int, default=96)
    parser.add_argument("--variance-candidates", type=int, default=64)
    parser.add_argument("--random-candidates", type=int, default=64)
    parser.add_argument("--max-candidate-pool", type=int, default=512)
    parser.add_argument("--max-evaluated-states", type=int, default=100_000)
    parser.add_argument("--minimum-raw-key-bits", type=int, default=64)
    parser.add_argument("--max-working-gb", type=float, default=8.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not math.isfinite(args.max_working_gb) or args.max_working_gb <= 0.0:
        raise CliError("--max-working-gb must be finite and positive")
    report = run(args)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
