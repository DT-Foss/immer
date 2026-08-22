#!/usr/bin/env python3
"""Measure honest COLD/WARM/HOT Streamer cache states in isolated processes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from immer.runtimes.deepseek_v4.cache_profile import (
    PHASE_SCHEMA,
    CacheProfileError,
    CacheWorkload,
    CacheWorkloadOperation,
    atomic_write_json,
    build_report,
    build_streamer,
    create_owned_run,
    execute_workload,
    load_phase,
    phase_result_path,
    seal_phase_envelope,
    validate_cold_phase,
    validate_owned_run,
)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _row_operation(raw: str) -> CacheWorkloadOperation:
    try:
        tensor, start, count = raw.rsplit(",", 2)
        return CacheWorkloadOperation(tensor, int(start), int(count))
    except (ValueError, CacheProfileError) as exc:
        raise argparse.ArgumentTypeError(
            "row operation must be TENSOR,START,COUNT with COUNT > 0"
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", help="parent for a new run-owned cache")
    parser.add_argument("--resume", help="resume after a completed COLD phase")
    parser.add_argument("--source", help="local checkpoint directory or HF repo id")
    parser.add_argument("--revision")
    parser.add_argument("--dataset-sha256")
    parser.add_argument("--prompt-sha256")
    parser.add_argument("--mode")
    parser.add_argument("--seed", type=_nonnegative_int)
    parser.add_argument("--tensor", action="append", default=[])
    parser.add_argument(
        "--rows",
        action="append",
        default=[],
        type=_row_operation,
        metavar="TENSOR,START,COUNT",
    )
    parser.add_argument("--source-budget-mb", type=_positive_int)
    parser.add_argument("--cache-budget-mb", type=_positive_int)
    parser.add_argument("--report-name", default="report.json")
    parser.add_argument(
        "--_worker-phase", choices=("cold", "warm-hot"), help=argparse.SUPPRESS
    )
    parser.add_argument("--_run-root", help=argparse.SUPPRESS)
    parser.add_argument("--_owner-token", help=argparse.SUPPRESS)
    parser.add_argument(
        "--_source-budget-bytes", type=_positive_int, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--_cache-budget-bytes", type=_positive_int, help=argparse.SUPPRESS
    )
    return parser


def _fresh_workload(args: argparse.Namespace) -> CacheWorkload:
    if not args.output_root:
        raise CacheProfileError("--output-root is required for a fresh run")
    if not args.source:
        raise CacheProfileError("--source is required for a fresh run")
    operations = [CacheWorkloadOperation(name) for name in args.tensor]
    operations.extend(args.rows)
    source_path = Path(args.source).expanduser()
    if source_path.exists():
        if not source_path.is_dir():
            raise CacheProfileError("local --source must be a checkpoint directory")
        source = str(source_path.resolve())
        source_kind = "local"
    else:
        if source_path.is_absolute() or args.source.startswith((".", "~")):
            raise CacheProfileError(f"local --source does not exist: {source_path}")
        source = args.source
        source_kind = "remote"
    return CacheWorkload(
        source=source,
        source_kind=source_kind,
        revision=args.revision or "local",
        dataset_sha256=args.dataset_sha256,
        prompt_sha256=args.prompt_sha256,
        mode=args.mode or "off",
        seed=0 if args.seed is None else args.seed,
        operations=tuple(operations),
    )


def _cache_snapshot(cache_dir: Path) -> tuple[int, int]:
    files = 0
    size = 0
    for path in cache_dir.rglob("*"):
        if path.is_symlink():
            raise CacheProfileError(f"owned cache contains a symlink: {path}")
        if path.is_file():
            files += 1
            size += path.stat().st_size
        elif not path.is_dir():
            raise CacheProfileError(f"owned cache contains a special file: {path}")
    return files, size


def _phase_envelope(
    *,
    phase: str,
    owner_token: str,
    workload: CacheWorkload,
    source_budget_bytes: int,
    cache_budget_bytes: int,
    cache_dir: Path,
) -> dict[str, Any]:
    if phase not in {"cold", "warm-hot"}:
        raise CacheProfileError(f"unknown worker phase {phase!r}")
    if phase == "cold":
        if cache_dir.exists() and any(cache_dir.iterdir()):
            raise CacheProfileError("COLD requires an empty run-owned cache")
        cache_dir.mkdir(mode=0o700, exist_ok=True)
    elif not cache_dir.is_dir() or not any(cache_dir.iterdir()):
        raise CacheProfileError("WARM requires the populated COLD disk cache")
    cache_files_before, cache_bytes_before = _cache_snapshot(cache_dir)
    source = build_streamer(
        workload,
        cache_dir,
        source_budget_bytes=source_budget_bytes,
        cache_budget_bytes=cache_budget_bytes,
    )
    reader_instance = f"pid:{os.getpid()}:reader:{id(source.reader)}"
    if phase == "cold":
        cold = execute_workload(
            workload,
            source,
            state="cold",
            refresh_inventory=True,
            reader_instance=reader_instance,
            cache_files_before=cache_files_before,
            cache_bytes_on_disk_before=cache_bytes_before,
        )
        validate_cold_phase(cold, workload, fresh_cache=True)
        return seal_phase_envelope(
            {
                "schema": PHASE_SCHEMA,
                "phase": "cold",
                "owner_token_sha256": hashlib.sha256(owner_token.encode()).hexdigest(),
                "cold": cold,
            }
        )
    warm = execute_workload(
        workload,
        source,
        state="warm",
        refresh_inventory=False,
        reader_instance=reader_instance,
        cache_files_before=cache_files_before,
        cache_bytes_on_disk_before=cache_bytes_before,
    )
    cache_files_before, cache_bytes_before = _cache_snapshot(cache_dir)
    hot = execute_workload(
        workload,
        source,
        state="hot",
        refresh_inventory=False,
        reader_instance=reader_instance,
        cache_files_before=cache_files_before,
        cache_bytes_on_disk_before=cache_bytes_before,
    )
    return seal_phase_envelope(
        {
            "schema": PHASE_SCHEMA,
            "phase": "warm-hot",
            "owner_token_sha256": hashlib.sha256(owner_token.encode()).hexdigest(),
            "warm": warm,
            "hot": hot,
        }
    )


def _worker(args: argparse.Namespace) -> int:
    if not args._run_root or not args._owner_token:
        raise CacheProfileError("worker requires owned run coordinates")
    if args._source_budget_bytes is None or args._cache_budget_bytes is None:
        raise CacheProfileError("worker requires exact stored byte budgets")
    run_root = Path(args._run_root).expanduser().resolve()
    owned = validate_owned_run(run_root, args._owner_token)
    if (
        args._source_budget_bytes != owned["marker"]["source_budget_bytes"]
        or args._cache_budget_bytes != owned["marker"]["cache_budget_bytes"]
    ):
        raise CacheProfileError("worker budgets do not match the immutable owned run")
    workload = owned["workload"]
    envelope = _phase_envelope(
        phase=args._worker_phase,
        owner_token=args._owner_token,
        workload=workload,
        source_budget_bytes=args._source_budget_bytes,
        cache_budget_bytes=args._cache_budget_bytes,
        cache_dir=owned["cache_dir"],
    )
    result_path = phase_result_path(run_root, args._worker_phase)
    if result_path.exists() or result_path.is_symlink():
        raise CacheProfileError(f"phase result already exists: {result_path}")
    atomic_write_json(result_path, envelope)
    return 0


def _child_command(
    args: argparse.Namespace, run_root: Path, owner_token: str, phase: str
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_worker-phase",
        phase,
        "--_run-root",
        str(run_root),
        "--_owner-token",
        owner_token,
        "--_source-budget-bytes",
        str(args._source_budget_bytes),
        "--_cache-budget-bytes",
        str(args._cache_budget_bytes),
    ]


def _run_child(
    args: argparse.Namespace, run_root: Path, owner_token: str, phase: str
) -> None:
    result = subprocess.run(
        _child_command(args, run_root, owner_token, phase),
        cwd=Path(__file__).resolve().parent.parent,
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode:
        message = (
            result.stderr.strip().splitlines()[-1] if result.stderr.strip() else ""
        )
        raise CacheProfileError(
            f"{phase} subprocess failed with {result.returncode}: {message}"
        )


def _validated_envelope(path: Path, phase: str, owner_token: str) -> dict[str, Any]:
    envelope = load_phase(path, phase)
    if envelope.get("phase") != phase:
        raise CacheProfileError(f"{phase} phase label mismatch")
    expected = hashlib.sha256(owner_token.encode()).hexdigest()
    if envelope.get("owner_token_sha256") != expected:
        raise CacheProfileError(f"{phase} phase belongs to another run owner")
    return envelope


@contextmanager
def _exclusive_run(run_root: Path) -> Iterator[None]:
    lock_path = run_root / ".cache-profile.lock"
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise CacheProfileError(f"cannot open run lock: {lock_path}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CacheProfileError("run lock is not a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CacheProfileError(
                "another cache-profile process owns this run"
            ) from exc
        yield
    finally:
        os.close(descriptor)


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.resume:
        fresh_inputs = (
            args.output_root,
            args.source,
            args.revision,
            args.dataset_sha256,
            args.prompt_sha256,
            args.mode,
            args.seed,
            args.tensor,
            args.rows,
            args.source_budget_mb,
            args.cache_budget_mb,
        )
        if any(value is not None and value != [] for value in fresh_inputs):
            raise CacheProfileError(
                "--resume uses the immutable stored workload; omit fresh-run inputs"
            )
        run_root = Path(args.resume).expanduser().resolve()
        owned = validate_owned_run(run_root)
        owner_token = owned["marker"]["owner_token"]
        workload = owned["workload"]
        args._source_budget_bytes = owned["marker"]["source_budget_bytes"]
        args._cache_budget_bytes = owned["marker"]["cache_budget_bytes"]
        resume = True
    else:
        if args.source_budget_mb is None:
            args.source_budget_mb = 16_384
        if args.cache_budget_mb is None:
            args.cache_budget_mb = 24_576
        args._source_budget_bytes = args.source_budget_mb * 1024**2
        args._cache_budget_bytes = args.cache_budget_mb * 1024**2
        workload = _fresh_workload(args)
        run_root, owner_token = create_owned_run(
            Path(args.output_root),
            workload,
            source_budget_bytes=args._source_budget_bytes,
            cache_budget_bytes=args._cache_budget_bytes,
        )
        resume = False
    with _exclusive_run(run_root):
        if resume:
            cold_path = phase_result_path(run_root, "cold")
            if not cold_path.exists():
                raise CacheProfileError(
                    "partial COLD cannot be resumed safely; start a new owned run"
                )
            cold_envelope = _validated_envelope(cold_path, "cold", owner_token)
            validate_cold_phase(
                cold_envelope.get("cold", {}), workload, fresh_cache=True
            )
        else:
            _run_child(args, run_root, owner_token, "cold")
            cold_envelope = _validated_envelope(
                phase_result_path(run_root, "cold"), "cold", owner_token
            )
        warm_hot_path = phase_result_path(run_root, "warm-hot")
        if not warm_hot_path.exists():
            _run_child(args, run_root, owner_token, "warm-hot")
        warm_hot_envelope = _validated_envelope(warm_hot_path, "warm-hot", owner_token)
        script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report = build_report(
            run_root=run_root,
            workload=workload,
            cold_envelope=cold_envelope,
            warm_hot_envelope=warm_hot_envelope,
            source_budget_bytes=args._source_budget_bytes,
            cache_budget_bytes=args._cache_budget_bytes,
            command=tuple(sys.argv),
            harness_sha256=script_sha,
        )
        report_path = run_root / args.report_name
        if report_path.parent != run_root or report_path.name in {
            ".cache-profile.lock",
            "OWNER.json",
            "workload.json",
            "cold.json",
            "warm-hot.json",
        }:
            raise CacheProfileError(
                "--report-name must be a safe filename in the run root"
            )
        if report_path.is_symlink():
            raise CacheProfileError("--report-name must not replace a symlink")
        atomic_write_json(report_path, report)
    return {
        "status": "complete",
        "run_root": str(run_root),
        "report": str(report_path),
        "workload_signature": workload.signature,
        "hot_claimed": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args._worker_phase:
            return _worker(args)
        receipt = run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
