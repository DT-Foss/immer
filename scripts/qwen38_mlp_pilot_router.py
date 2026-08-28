#!/usr/bin/env python3
"""Fit a layer-local Qwen Gate×Up pilot router and open a later generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import stat
from typing import Sequence

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.mlp_pilot_router import (
    DEFAULT_RANDOM_SEED_SHA256,
    PILOT_ROUTER_VERIFIER_SHA256,
    MlpPilotRouterConfig,
    evaluate_mlp_pilot_router,
    fit_mlp_pilot_router,
)
from immer.runtimes.ooe.prompt_row_roles import derive_prompt_row_roles
from immer.runtimes.ooe.qwen_mlp_evidence import QwenMlpEvidenceBank


REPORT_SCHEMA = "immer.qwen3.8-mlp-pilot-live-report/v1"
FIT_NAME = "fit.json"
EVALUATION_NAME = "evaluation.json"
REPORT_NAME = "report.json"
FIT_ROW_ROLES_NAME = "fit-row-roles.json"
HOLDOUT_ROW_ROLES_NAME = "holdout-row-roles.json"


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
        if path.is_symlink() or _stable_read(path, max(1, len(data))) != data:
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
            count = os.write(descriptor, view[offset:])
            if count <= 0:
                raise OSError("short write")
            offset += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError:
        if _stable_read(path, max(1, len(data))) != data:
            raise CliError(f"sealed artifact collided: {path}")
    finally:
        temporary.unlink(missing_ok=True)


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
    result = {}
    for row in body["prompts"]:
        if not isinstance(row, dict) or row.get("sha256") not in prompts:
            continue
        if not isinstance(row.get("token_ids"), list):
            raise CliError(f"prompt token row is invalid: {path}")
        result[row["sha256"]] = tuple(row["token_ids"])
    if set(result) != prompts:
        raise CliError(f"prompt registry does not cover the selected bank: {path}")
    return result


def _bank_prompts(bank: QwenMlpEvidenceBank) -> set[str]:
    return {
        receipt.entry.prompt_sha256 for receipt, _verification in bank.committed_pairs()
    }


def _config(args: argparse.Namespace) -> MlpPilotRouterConfig:
    return MlpPilotRouterConfig(
        block_size=args.block_size,
        pilot_count=args.pilot_count,
        selected_block_count=args.selected_block_count,
        ridge=args.ridge,
        random_seed_sha256=args.random_seed_sha256,
        max_working_bytes=int(args.max_working_gb * 1024**3),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    output = Path(args.output_root).expanduser().absolute()
    fit_bank = QwenMlpEvidenceBank(Path(args.fit_bank_root).expanduser().absolute())
    if fit_bank.state().split_counts != (25, 10, 5) or not fit_bank.audit().clean:
        raise CliError("fit bank is not a clean complete 25/10/5 bank")
    fit_prompts = _bank_prompts(fit_bank)
    if not 1 <= args.train_prompt_count < len(fit_prompts):
        raise CliError("--train-prompt-count must leave calibration prompts")
    fit_roles = derive_prompt_row_roles(
        _prompt_tokens(
            Path(args.fit_prompt_registry).expanduser().absolute(), fit_prompts
        )
    )
    fit_corpus = fit_bank.build_subspace_corpus(
        row_indices_by_prompt=fit_roles.row_indices_by_prompt
    )
    prompts = tuple(sorted(fit_prompts))
    fit = fit_mlp_pilot_router(
        fit_corpus,
        train_prompt_sha256s=prompts[: args.train_prompt_count],
        calibration_prompt_sha256s=prompts[args.train_prompt_count :],
        row_role_sha256=fit_roles.sha256,
        config=_config(args),
    )
    _persist_exact(output / FIT_ROW_ROLES_NAME, fit_roles.to_bytes())
    _persist_exact(output / FIT_NAME, fit.to_bytes())

    body: dict[str, object] = {
        "calibration_metrics": fit.calibration_metrics.to_record(),
        "config_sha256": fit.config.sha256,
        "fit_bank_state_sha256": fit_bank.state().sha256,
        "fit_row_role_sha256": fit_roles.sha256,
        "fit_sha256": fit.sha256,
        "model_pin_sha256": fit.model_pin_sha256,
        "status": "calibration_candidate",
        "verifier_sha256": PILOT_ROUTER_VERIFIER_SHA256,
    }
    if args.external_holdout_bank_root is not None:
        if args.holdout_prompt_registry is None:
            raise CliError("external holdout requires --holdout-prompt-registry")
        holdout_bank = QwenMlpEvidenceBank(
            Path(args.external_holdout_bank_root).expanduser().absolute()
        )
        if (
            holdout_bank.state().split_counts != (25, 10, 5)
            or not holdout_bank.audit().clean
        ):
            raise CliError("holdout bank is not a clean complete 25/10/5 bank")
        holdout_prompts = _bank_prompts(holdout_bank)
        holdout_roles = derive_prompt_row_roles(
            _prompt_tokens(
                Path(args.holdout_prompt_registry).expanduser().absolute(),
                holdout_prompts,
            )
        )
        holdout_corpus = holdout_bank.build_subspace_corpus(
            row_indices_by_prompt=holdout_roles.row_indices_by_prompt
        )
        authority = _digest(
            {
                "fit_sha256": fit.sha256,
                "holdout_bank_state_sha256": holdout_bank.state().sha256,
                "holdout_group_sha256s": [
                    group.sha256 for group in holdout_corpus.groups
                ],
                "schema": "immer.qwen3.8-mlp-pilot-holdout-authority/v1",
            }
        )
        evaluation = evaluate_mlp_pilot_router(
            fit,
            holdout_corpus,
            holdout_authority_sha256=authority,
            holdout_row_role_sha256=holdout_roles.sha256,
        )
        _persist_exact(output / HOLDOUT_ROW_ROLES_NAME, holdout_roles.to_bytes())
        _persist_exact(output / EVALUATION_NAME, evaluation.to_bytes())
        body.update(
            {
                "evaluation_sha256": evaluation.sha256,
                "external_metrics": evaluation.metrics.to_record(),
                "holdout_authority_sha256": authority,
                "holdout_bank_state_sha256": holdout_bank.state().sha256,
                "holdout_row_role_sha256": holdout_roles.sha256,
                "status": "external_generation_evaluated",
            }
        )
    report = {
        "body": body,
        "body_sha256": _digest(body),
        "schema": REPORT_SCHEMA,
    }
    _persist_exact(output / REPORT_NAME, canonical_json_bytes(report) + b"\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-bank-root", required=True)
    parser.add_argument("--fit-prompt-registry", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--external-holdout-bank-root")
    parser.add_argument("--holdout-prompt-registry")
    parser.add_argument("--train-prompt-count", type=int, default=3)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--pilot-count", type=int, default=4)
    parser.add_argument("--selected-block-count", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--random-seed-sha256", default=DEFAULT_RANDOM_SEED_SHA256)
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
