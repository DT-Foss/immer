#!/usr/bin/env python3
"""Calibrate the all-layer Fast-MLP state from existing exact O1 evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from immer.runtimes.deepseek_v4.causal_weights import (
    CausalWeightMount,
    LogicalModelIdentity,
)
from immer.runtimes.ooe.qwen_mlp_evidence import QwenMlpEvidenceBank
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.qwen_mlp_evidence import (
    EVIDENCE_BUDGET_SCHEMA,
    MlpEvidenceBudget,
)
from immer.runtimes.qwen3_8.bundle import verify_qwen38_causal_mount
from immer.runtimes.qwen3_8.config import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    Qwen38Config,
)
from immer.runtimes.qwen3_8.fast_mlp import (
    Qwen38FastMlpPaths,
    open_qwen38_fast_mlp,
)
from immer.runtimes.qwen3_8.kernels import swiglu
from immer.runtimes.qwen3_8.pager import Qwen38WeightPager


STATUS_SCHEMA = "immer.qwen3.8-fast-mlp-existing-evidence-status/v1"
_STATE_MAX_BYTES = 16 * 1024 * 1024
_SPLITS = ("train", "calibration", "holdout")


class CliError(RuntimeError):
    pass


def _stable_read(path: Path, maximum: int) -> bytes:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
        )
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise CliError(f"input is not a bounded regular file: {path}")
        data = bytearray(before.st_size)
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            chunk = os.read(descriptor, len(view) - offset)
            if not chunk:
                raise CliError(f"short input read: {path}")
            view[offset : offset + len(chunk)] = chunk
            offset += len(chunk)
        after = os.fstat(descriptor)
        linked = path.lstat()
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or (after.st_dev, after.st_ino) != (linked.st_dev, linked.st_ino):
            raise CliError(f"input changed while read: {path}")
        return bytes(data)
    except CliError:
        raise
    except OSError as exc:
        raise CliError(f"cannot read input: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise CliError(f"output state already exists: {path}")
    temporary = path.parent / f".{path.name}.{secrets.token_hex(12)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_CREAT
            | os.O_EXCL
            | os.O_WRONLY
            | int(getattr(os, "O_CLOEXEC", 0))
            | int(getattr(os, "O_NOFOLLOW", 0)),
            0o600,
        )
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short output-state write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(
            path.parent,
            os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0)),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as exc:
        raise CliError(f"output state collided: {path}") from exc
    except OSError as exc:
        raise CliError(f"cannot publish output state: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _clone_state(input_state: Path | None, output_state: Path) -> None:
    if input_state is None:
        if output_state.exists() or output_state.is_symlink():
            raise CliError(f"output state already exists: {output_state}")
        return
    if input_state.absolute() == output_state.absolute():
        raise CliError("input and output state must be different files")
    _publish_new(output_state, _stable_read(input_state, _STATE_MAX_BYTES))


def _evidence_budget(root: Path) -> MlpEvidenceBudget:
    data = _stable_read(root / "BUDGET.json", 64 * 1024)
    try:
        document = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError("evidence budget is not JSON") from exc
    if (
        not isinstance(document, dict)
        or set(document) != {"body", "body_sha256", "schema"}
        or document.get("schema") != EVIDENCE_BUDGET_SCHEMA
        or not isinstance(document.get("body"), dict)
        or canonical_json_bytes(document) != data
    ):
        raise CliError("evidence budget envelope is invalid")
    body = document["body"]
    limits = body.get("limits")
    expected = set(MlpEvidenceBudget.__dataclass_fields__)
    if (
        set(body) != {"budget_sha256", "limits"}
        or not isinstance(limits, dict)
        or set(limits) != expected
        or document.get("body_sha256")
        != hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    ):
        raise CliError("evidence budget body is invalid")
    try:
        budget = MlpEvidenceBudget(**limits)
    except (TypeError, ValueError) as exc:
        raise CliError("evidence budget limits are invalid") from exc
    if body.get("budget_sha256") != budget.sha256:
        raise CliError("evidence budget digest changed")
    return budget


def _pairs_by_layer(
    pairs: Iterable[tuple[Any, Any]],
    *,
    splits: Sequence[str],
) -> dict[int, tuple[tuple[Any, Any], ...]]:
    selected = frozenset(splits)
    if not selected or not selected <= frozenset(_SPLITS):
        raise CliError("selected evidence splits are invalid")
    rows: dict[int, list[tuple[Any, Any]]] = defaultdict(list)
    identities: set[str] = set()
    for receipt, verification in pairs:
        entry = receipt.entry
        if entry.split not in selected:
            continue
        rows[entry.layer].append((receipt, verification))
        identities.add(receipt.identity_sha256)
    if not rows or sum(map(len, rows.values())) != len(identities):
        raise CliError("evidence groups are empty or duplicated")
    return {
        layer: tuple(
            sorted(
                values,
                key=lambda pair: (
                    pair[0].entry.ordinal,
                    pair[0].entry.prompt_sha256,
                ),
            )
        )
        for layer, values in sorted(rows.items())
    }


def _tensor(bank: Any, receipt: Any, stage: str) -> torch.Tensor:
    references = {reference.stage: reference for reference in receipt.tensors}
    reference = references.get(stage)
    if reference is None:
        raise CliError(f"evidence group lacks {stage}")
    restored = bank.restore_tensor(reference)
    array = np.asarray(restored)
    if not np.issubdtype(array.dtype, np.floating) or not bool(
        np.isfinite(array).all()
    ):
        raise CliError(f"restored {stage} tensor is not finite floating point")
    return torch.from_numpy(np.array(array, copy=True)).reshape(
        -1,
        reference.shape[-1],
    ).to(torch.bfloat16)


def _calibrate_layer(
    *,
    layer: int,
    pairs: Sequence[tuple[Any, Any]],
    bank: Any,
    pager: Any,
    executor: Any,
    config: Qwen38Config,
) -> int:
    if not pairs:
        raise CliError(f"layer {layer} has no evidence groups")
    down_name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    down = pager.tensor_torch(
        down_name,
        dtype=torch.bfloat16,
        device="cpu",
        zero_copy_cpu=True,
    )
    expected = (config.dim, config.intermediate_size)
    if tuple(down.shape) != expected:
        raise CliError(f"layer {layer} Down projection changed shape")
    observed = 0
    try:
        for receipt, _verification in pairs:
            gate = _tensor(bank, receipt, "mlp.gate")
            up = _tensor(bank, receipt, "mlp.up")
            output = _tensor(bank, receipt, "mlp.output")
            if gate.shape != up.shape or gate.shape[0] != output.shape[0]:
                raise CliError(f"layer {layer} evidence row ABI changed")
            activated = swiglu(gate, up)
            observation = executor.observe_full(
                layer=layer,
                gate=gate,
                up=up,
                activated=activated,
                output=output,
                down_weight=down,
            )
            if observation is None:
                raise CliError("weight-only executor returned no observation")
            observed += len(observation.shadow_row_indices)
    finally:
        del down
        pager.release(force_gc=True)
    return observed


def run(args: argparse.Namespace) -> dict[str, object]:
    bundle_root = Path(args.bundle).expanduser().absolute()
    evidence_root = Path(args.evidence_root).expanduser().absolute()
    fast_root = Path(args.fast_mlp_root).expanduser().absolute()
    input_state = (
        None
        if args.input_state is None
        else Path(args.input_state).expanduser().absolute()
    )
    output_state = Path(args.output_state).expanduser().absolute()
    _clone_state(input_state, output_state)
    try:
        bank = QwenMlpEvidenceBank(
            evidence_root,
            budget=_evidence_budget(evidence_root),
        )
        audit = bank.audit()
        if not audit.clean:
            raise CliError("existing MLP evidence bank audit is not clean")
        pairs = _pairs_by_layer(bank.committed_pairs(), splits=args.splits)
    except Exception:
        output_state.unlink(missing_ok=True)
        raise

    identity = LogicalModelIdentity(OFFICIAL_REPO_ID, OFFICIAL_REVISION)
    fast = None
    pager = None
    completed = False
    try:
        with CausalWeightMount(
            bundle_root,
            identity,
            budget_mb=args.source_budget_mb,
        ) as mount:
            bundle = verify_qwen38_causal_mount(
                mount,
                require_official_config=True,
            )
            config = Qwen38Config.from_file(
                mount.weights_root / "config.json",
                require_official=True,
            )
            if set(pairs) != set(range(config.n_layers)):
                raise CliError("evidence bank does not cover every model layer")
            pager = Qwen38WeightPager(
                mount.source,
                device="cpu",
                compute_dtype="bfloat16",
                max_resident_bytes=args.max_resident_mb * 1024**2,
                close_source=False,
                require_source_identity=True,
                causal_tensor_reader=mount.tensor_reader,
            )
            fast = open_qwen38_fast_mlp(
                paths=Qwen38FastMlpPaths.from_root(fast_root),
                target_mount=mount,
                target_pager=pager,
                config=config,
                source_budget_mb=args.fast_source_budget_mb,
                max_resident_bytes=args.fast_max_resident_mb * 1024**2,
                online_state_path=output_state,
            )
            observed = 0
            for layer, layer_pairs in pairs.items():
                observed += _calibrate_layer(
                    layer=layer,
                    pairs=layer_pairs,
                    bank=bank,
                    pager=pager,
                    executor=fast.executor,
                    config=config,
                )
            metrics = fast.executor.online_metrics()
            if metrics is None:
                raise CliError("weight-only online metrics are absent")
            decisions = [
                fast.executor.decision(layer=layer, row_count=8)
                for layer in range(config.n_layers)
            ]
            fast.metrics()  # One atomic output-state flush.
            completed = True
            return {
                "bank_audit_receipts": audit.receipt_count,
                "bundle": bundle,
                "decision_reasons": dict(Counter(row.reason for row in decisions)),
                "evidence_groups": sum(map(len, pairs.values())),
                "layers": len(pairs),
                "online_metrics": metrics,
                "output_state": str(output_state),
                "schema": STATUS_SCHEMA,
                "shadow_rows": observed,
                "source_pager": pager.metrics(),
                "sparse_layers": [row.layer for row in decisions if row.use_sparse],
                "splits": list(args.splits),
            }
    finally:
        if fast is not None:
            fast.close()
        if pager is not None:
            pager.close()
        if not completed:
            # A failed candidate is never mistaken for a complete calibration.
            try:
                output_state.unlink(missing_ok=True)
            except OSError:
                pass


def _split_list(value: str) -> tuple[str, ...]:
    result = tuple(part.strip() for part in value.split(",") if part.strip())
    if not result or len(set(result)) != len(result) or any(
        part not in _SPLITS for part in result
    ):
        raise argparse.ArgumentTypeError(
            "splits must be a unique comma list of train,calibration,holdout"
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("evidence_root", type=Path)
    parser.add_argument("fast_mlp_root", type=Path)
    parser.add_argument("output_state", type=Path)
    parser.add_argument("--input-state", type=Path)
    parser.add_argument("--splits", type=_split_list, default=_SPLITS)
    parser.add_argument("--source-budget-mb", type=float, default=32768.0)
    parser.add_argument("--fast-source-budget-mb", type=float, default=32768.0)
    parser.add_argument("--max-resident-mb", type=int, default=192)
    parser.add_argument("--fast-max-resident-mb", type=int, default=192)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in ("source_budget_mb", "fast_source_budget_mb"):
        value = getattr(args, name)
        if not isinstance(value, (int, float)) or value <= 0:
            raise CliError(f"--{name.replace('_', '-')} must be positive")
    for name in ("max_resident_mb", "fast_max_resident_mb"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CliError(f"--{name.replace('_', '-')} must be a positive integer")
    print(json.dumps(run(args), allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
