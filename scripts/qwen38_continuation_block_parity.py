#!/usr/bin/env python3
"""Benchmark exact Qwen3.8 K=2 continuation-block parity.

The benchmark replays one independently generated, sealed off-arm token chain
through two fresh runtimes over the same authenticated causal bundle.  Both
runtimes prefill once.  The reference path then commits two one-token decodes;
the candidate path stages and atomically commits the same two-token
continuation block.  No labels or gold answers enter the report.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import importlib.util
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
from types import ModuleType
from typing import Any

import torch

from immer.knowledge import AccessTraceRecorder
from immer.runtimes.qwen3_8 import AttentionState, DeltaNetState, StreamedQwen38


ROOT = Path(__file__).resolve().parent.parent
DIRECT_SCRIPT = ROOT / "scripts" / "qwen38_direct_decode_benchmark.py"
RESULT_SCHEMA = "immer.qwen3.8-continuation-block-parity/v1"
EXECUTION_SCHEMA = "immer.qwen3.8-continuation-block-execution/v1"
STATE_SCHEMA = "immer.qwen3.8-continuation-state-manifest/v1"
CONTRACT_KIND = "sealed-off-chain-k2-replay/v1"
CONTINUATION_LENGTH = 2


class ContinuationParityError(RuntimeError):
    """The sealed continuation-parity contract cannot be satisfied."""


def _load_direct_protocol() -> ModuleType:
    """Load the canonical branch validators without duplicating their rules."""

    module_name = "qwen38_direct_decode_benchmark"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, DIRECT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import Qwen direct-decode protocol")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


direct = _load_direct_protocol()


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContinuationParityError(f"{label} must be a non-negative integer")
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContinuationParityError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ContinuationParityError(f"{label} must be finite and non-negative")
    return result


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ContinuationParityError(f"{label} must be a SHA-256 digest")
    return value


def _seal(identity: Mapping[str, Any]) -> dict[str, Any]:
    return direct._result(dict(identity))


def _verify_seal(document: Mapping[str, Any], label: str) -> None:
    try:
        direct._verify_document_seal(document, label)
    except Exception as exc:
        raise ContinuationParityError(str(exc)) from exc


def _tensor_record(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    if not isinstance(name, str) or not name:
        raise ContinuationParityError("state tensor name is invalid")
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
        raise ContinuationParityError(f"state tensor {name!r} must be floating point")
    detached = tensor.detach()
    return {
        "dtype": str(detached.dtype).removeprefix("torch."),
        "name": name,
        "nbytes": int(detached.numel() * detached.element_size()),
        "sha256": direct._hidden_sha256(detached),
        "shape": [int(value) for value in detached.shape],
    }


def _state_manifest(model: StreamedQwen38) -> dict[str, Any]:
    raw_states = getattr(model, "_layer_states", None)
    if not isinstance(raw_states, list) or len(raw_states) != int(
        model.config.n_layers
    ):
        raise ContinuationParityError("runtime layer-state container is invalid")
    records: list[dict[str, Any]] = []
    for layer, state in enumerate(raw_states):
        prefix = f"layer.{layer:03d}"
        if isinstance(state, AttentionState):
            records.append(_tensor_record(f"{prefix}.attention.key", state.key))
            records.append(_tensor_record(f"{prefix}.attention.value", state.value))
            if state.crsa_log_usage is not None:
                records.append(
                    _tensor_record(
                        f"{prefix}.attention.crsa_log_usage",
                        state.crsa_log_usage,
                    )
                )
        elif isinstance(state, DeltaNetState):
            records.append(_tensor_record(f"{prefix}.deltanet.conv", state.conv))
            records.append(
                _tensor_record(f"{prefix}.deltanet.recurrent", state.recurrent)
            )
        else:
            raise ContinuationParityError(
                f"runtime layer {layer} has no committed continuation state"
            )
    graft_history = getattr(model, "_graft_history", None)
    if graft_history is not None:
        records.append(_tensor_record("graft.history", graft_history))
    identity = {
        "cursor": int(model.next_position),
        "schema": STATE_SCHEMA,
        "tensor_count": len(records),
        "tensors": records,
        "total_bytes": sum(record["nbytes"] for record in records),
    }
    return _seal(identity)


_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8}


def _validate_state_manifest(document: object) -> Mapping[str, Any]:
    required = {
        "cursor",
        "schema",
        "sha256",
        "tensor_count",
        "tensors",
        "total_bytes",
    }
    if (
        not isinstance(document, Mapping)
        or set(document) != required
        or document.get("schema") != STATE_SCHEMA
    ):
        raise ContinuationParityError("continuation state manifest is invalid")
    _verify_seal(document, "continuation state manifest")
    cursor = _nonnegative_int(document.get("cursor"), "state cursor")
    if cursor < 1:
        raise ContinuationParityError("continuation state cursor must be positive")
    records = document.get("tensors")
    count = _nonnegative_int(document.get("tensor_count"), "state tensor count")
    total = _nonnegative_int(document.get("total_bytes"), "state total bytes")
    if not isinstance(records, list) or count != len(records) or not records:
        raise ContinuationParityError("continuation state tensor list is invalid")
    names: set[str] = set()
    observed_total = 0
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "dtype",
            "name",
            "nbytes",
            "sha256",
            "shape",
        }:
            raise ContinuationParityError("continuation state tensor record is invalid")
        name = record.get("name")
        dtype = record.get("dtype")
        shape = record.get("shape")
        if not isinstance(name, str) or not name or name in names:
            raise ContinuationParityError("continuation state tensor names are invalid")
        if dtype not in _DTYPE_BYTES:
            raise ContinuationParityError("continuation state tensor dtype is invalid")
        if (
            not isinstance(shape, list)
            or not shape
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in shape
            )
        ):
            raise ContinuationParityError("continuation state tensor shape is invalid")
        nbytes = _nonnegative_int(record.get("nbytes"), "state tensor bytes")
        expected = math.prod(shape) * _DTYPE_BYTES[str(dtype)]
        if nbytes != expected:
            raise ContinuationParityError("continuation state tensor bytes disagree")
        _digest(record.get("sha256"), "state tensor")
        names.add(name)
        observed_total += nbytes
    if observed_total != total:
        raise ContinuationParityError("continuation state total bytes disagree")
    return document


def _state_diff(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> list[dict[str, Any]]:
    left_rows = {row["name"]: row for row in left["tensors"]}
    right_rows = {row["name"]: row for row in right["tensors"]}
    differences: list[dict[str, Any]] = []
    for name in sorted(set(left_rows) | set(right_rows)):
        left_row = left_rows.get(name)
        right_row = right_rows.get(name)
        if left_row != right_row:
            differences.append(
                {
                    "block": None if right_row is None else right_row.get("sha256"),
                    "name": name,
                    "tokenwise": None if left_row is None else left_row.get("sha256"),
                }
            )
    return differences


def _metric(owner: object, name: str) -> int:
    return direct._source_metric(owner, name)


def _evidence_sum(rows: Sequence[Any], field: str) -> int | float:
    values = [getattr(row, field) for row in rows]
    if field == "seconds":
        return math.fsum(float(value) for value in values)
    return sum(int(value) for value in values)


def _phase_record(
    *,
    start_pos: int,
    end_pos: int,
    hidden: torch.Tensor,
    model: StreamedQwen38,
    source_body_bytes: int,
    linear_calls: int,
    model_seconds: float,
    wall_seconds: float,
    hash_started: float,
    public_stage_hidden_sha256: str | None,
) -> dict[str, Any]:
    hidden_sha256 = direct._hidden_sha256(hidden)
    state = _state_manifest(model)
    return {
        "end_pos": end_pos,
        "hash_seconds": time.perf_counter() - hash_started,
        "hidden_sha256": hidden_sha256,
        "linear_calls": linear_calls,
        "model_seconds": model_seconds,
        "public_stage_hidden_sha256": public_stage_hidden_sha256,
        "source_body_bytes": source_body_bytes,
        "start_pos": start_pos,
        "state": state,
        "wall_seconds": wall_seconds,
    }


def _run_execution(
    args: argparse.Namespace,
    *,
    mode: str,
    prompt: tuple[int, ...],
    continuation: tuple[int, ...],
    trace_path: str | os.PathLike[str],
    expected_checkpoint: Mapping[str, Any],
    expected_bundle: Mapping[str, Any],
    expected_bundle_root: Path,
    runtime_factory: Callable[..., tuple[Any, StreamedQwen38]],
) -> dict[str, Any]:
    recorder = AccessTraceRecorder()
    runtime, model = runtime_factory(args, recorder, max_batch_size=1)
    try:
        if (
            model.graft is not None
            or model.graft_layer is not None
            or model.native_head_crsa is not None
            or model.delta_probe is not None
        ):
            raise ContinuationParityError("continuation parity requires an off runtime")
        checkpoint = direct._checkpoint(model)
        bundle = dict(runtime.verification or {})
        mount = getattr(runtime, "mount", None)
        mounted_root = getattr(mount, "root", None)
        if mounted_root is None or Path(mounted_root).resolve() != expected_bundle_root:
            raise ContinuationParityError(
                "runtime did not open the requested causal bundle root"
            )
        if checkpoint != dict(expected_checkpoint):
            raise ContinuationParityError("runtime checkpoint differs from reference")
        if bundle != dict(expected_bundle):
            raise ContinuationParityError(
                "runtime causal bundle differs from reference"
            )
        if int(model.config.n_layers) != int(args._reference_checkpoint_layers):
            raise ContinuationParityError("runtime layer count differs from reference")
        if len(prompt) + len(continuation) > model.max_seq_len:
            raise ContinuationParityError("continuation replay exceeds runtime context")

        source = model.pager.source
        prefix_start_bytes = _metric(source, "network_or_source_body_bytes")
        prefix_start_linears = _metric(model.pager, "linear_calls")
        prefix_started = time.perf_counter()
        prefix_hidden, prefix_evidence = model.prefill(
            [prompt], tokenwise=False, reset=True
        )
        prefix_wall = time.perf_counter() - prefix_started
        prefix_bytes = (
            _metric(source, "network_or_source_body_bytes") - prefix_start_bytes
        )
        prefix_linears = _metric(model.pager, "linear_calls") - prefix_start_linears
        reported_prefix_bytes = int(_evidence_sum(prefix_evidence, "source_body_bytes"))
        reported_prefix_linears = int(_evidence_sum(prefix_evidence, "linear_calls"))
        reported_prefix_seconds = float(_evidence_sum(prefix_evidence, "seconds"))
        if (prefix_bytes, prefix_linears) != (
            reported_prefix_bytes,
            reported_prefix_linears,
        ):
            raise ContinuationParityError(
                "prefill counters differ from runtime evidence"
            )
        if model.next_position != len(prompt):
            raise ContinuationParityError("prefill cursor differs from sealed prompt")
        prefix_hash_started = time.perf_counter()
        prefix = _phase_record(
            start_pos=0,
            end_pos=len(prompt),
            hidden=prefix_hidden,
            model=model,
            source_body_bytes=prefix_bytes,
            linear_calls=prefix_linears,
            model_seconds=reported_prefix_seconds,
            wall_seconds=prefix_wall,
            hash_started=prefix_hash_started,
            public_stage_hidden_sha256=None,
        )

        continuation_start_bytes = _metric(source, "network_or_source_body_bytes")
        continuation_start_linears = _metric(model.pager, "linear_calls")
        continuation_started = time.perf_counter()
        public_stage_sha: str | None = None
        stage_base_state: dict[str, Any] | None = None
        stage_base_state_equal: bool | None = None
        if mode == "tokenwise":
            hidden_rows: list[torch.Tensor] = []
            continuation_evidence: list[Any] = []
            for token in continuation:
                hidden, evidence = model.decode([[token]])
                hidden_rows.append(hidden)
                continuation_evidence.append(evidence)
            continuation_hidden = torch.cat(hidden_rows, dim=1)
            reported_continuation_bytes = int(
                _evidence_sum(continuation_evidence, "source_body_bytes")
            )
            reported_continuation_linears = int(
                _evidence_sum(continuation_evidence, "linear_calls")
            )
            reported_continuation_seconds = float(
                _evidence_sum(continuation_evidence, "seconds")
            )
            continuation_mode = "two-single-token-commits/v1"
        elif mode == "block":
            stage = model.stage_continuation_block([continuation])
            public_stage_sha = direct._hidden_sha256(stage.hidden)
            stage_base_state = _state_manifest(model)
            stage_base_state_equal = bool(
                model.next_position == len(prompt)
                and stage_base_state == prefix["state"]
            )
            continuation_hidden, evidence = model.commit_continuation_block(stage)
            # The public stage and committed evidence use related but
            # deliberately different dataclasses.  Bind every shared field.
            shared = (
                "start_pos",
                "end_pos",
                "input_token_ids",
                "layers_executed",
                "checkpoint_layers",
                "complete_layer_stack",
                "stateful_cache",
                "source_body_bytes",
                "linear_calls",
                "seconds",
                "graft_mode",
                "graft_history_tokens",
            )
            if any(
                getattr(stage.evidence, name) != getattr(evidence, name)
                for name in shared
            ):
                raise ContinuationParityError(
                    "block stage evidence differs from committed evidence"
                )
            reported_continuation_bytes = int(evidence.source_body_bytes)
            reported_continuation_linears = int(evidence.linear_calls)
            reported_continuation_seconds = float(evidence.seconds)
            continuation_mode = "stage-then-atomic-commit-k2/v1"
        else:  # pragma: no cover - internal caller fixes both modes.
            raise AssertionError(mode)
        continuation_wall = time.perf_counter() - continuation_started
        continuation_bytes = (
            _metric(source, "network_or_source_body_bytes") - continuation_start_bytes
        )
        continuation_linears = (
            _metric(model.pager, "linear_calls") - continuation_start_linears
        )
        if (continuation_bytes, continuation_linears) != (
            reported_continuation_bytes,
            reported_continuation_linears,
        ):
            raise ContinuationParityError(
                "continuation counters differ from runtime evidence"
            )
        if public_stage_sha is not None and public_stage_sha != direct._hidden_sha256(
            continuation_hidden
        ):
            raise ContinuationParityError("public stage hidden differs after commit")
        expected_end = len(prompt) + len(continuation)
        if model.next_position != expected_end:
            raise ContinuationParityError("continuation cursor differs after commit")
        continuation_hash_started = time.perf_counter()
        continuation_record = _phase_record(
            start_pos=len(prompt),
            end_pos=expected_end,
            hidden=continuation_hidden,
            model=model,
            source_body_bytes=continuation_bytes,
            linear_calls=continuation_linears,
            model_seconds=reported_continuation_seconds,
            wall_seconds=continuation_wall,
            hash_started=continuation_hash_started,
            public_stage_hidden_sha256=public_stage_sha,
        )
        trace = direct._trace_receipt(recorder, trace_path)
        identity = {
            "bundle": bundle,
            "checkpoint": checkpoint,
            "continuation": continuation_record,
            "continuation_mode": continuation_mode,
            "final_cursor": int(model.next_position),
            "final_state_bytes": int(model.state_bytes),
            "mode": mode,
            "prefill": prefix,
            "prefill_mode": "batched",
            "schema": EXECUTION_SCHEMA,
            "stage_base_state": stage_base_state,
            "stage_base_state_equal": stage_base_state_equal,
            "trace": trace,
        }
        return _seal(identity)
    finally:
        direct._cleanup(runtime, model)


def _validate_phase(value: object, *, allow_stage: bool) -> Mapping[str, Any]:
    required = {
        "end_pos",
        "hash_seconds",
        "hidden_sha256",
        "linear_calls",
        "model_seconds",
        "public_stage_hidden_sha256",
        "source_body_bytes",
        "start_pos",
        "state",
        "wall_seconds",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ContinuationParityError("continuation execution phase is invalid")
    start = _nonnegative_int(value.get("start_pos"), "phase start")
    end = _nonnegative_int(value.get("end_pos"), "phase end")
    if end <= start:
        raise ContinuationParityError("continuation execution phase is empty")
    _digest(value.get("hidden_sha256"), "phase hidden")
    stage_sha = value.get("public_stage_hidden_sha256")
    if allow_stage:
        _digest(stage_sha, "public stage hidden")
        if stage_sha != value["hidden_sha256"]:
            raise ContinuationParityError("public and committed block hidden differ")
    elif stage_sha is not None:
        raise ContinuationParityError("unexpected public stage hidden receipt")
    _nonnegative_int(value.get("source_body_bytes"), "phase source bytes")
    _nonnegative_int(value.get("linear_calls"), "phase linear calls")
    model_seconds = _finite_nonnegative(value.get("model_seconds"), "model seconds")
    wall_seconds = _finite_nonnegative(value.get("wall_seconds"), "wall seconds")
    _finite_nonnegative(value.get("hash_seconds"), "hash seconds")
    if wall_seconds < model_seconds:
        raise ContinuationParityError("phase wall time is below model time")
    state = _validate_state_manifest(value.get("state"))
    if state["cursor"] != end:
        raise ContinuationParityError("phase state cursor differs from phase end")
    return value


def _validate_execution(document: object, *, mode: str) -> Mapping[str, Any]:
    required = {
        "bundle",
        "checkpoint",
        "continuation",
        "continuation_mode",
        "final_cursor",
        "final_state_bytes",
        "mode",
        "prefill",
        "prefill_mode",
        "schema",
        "sha256",
        "stage_base_state",
        "stage_base_state_equal",
        "trace",
    }
    if (
        not isinstance(document, Mapping)
        or set(document) != required
        or document.get("schema") != EXECUTION_SCHEMA
        or document.get("mode") != mode
    ):
        raise ContinuationParityError("continuation execution receipt is invalid")
    _verify_seal(document, "continuation execution receipt")
    if not isinstance(document.get("bundle"), Mapping) or not isinstance(
        document.get("checkpoint"), Mapping
    ):
        raise ContinuationParityError("continuation execution identity is invalid")
    prefill = _validate_phase(document.get("prefill"), allow_stage=False)
    continuation = _validate_phase(
        document.get("continuation"), allow_stage=mode == "block"
    )
    if continuation["start_pos"] != prefill["end_pos"]:
        raise ContinuationParityError("continuation does not follow the prefill")
    if continuation["end_pos"] - continuation["start_pos"] != CONTINUATION_LENGTH:
        raise ContinuationParityError("continuation receipt is not K=2")
    if document.get("final_cursor") != continuation["end_pos"]:
        raise ContinuationParityError("execution cursor differs from continuation")
    final_state_bytes = _nonnegative_int(
        document.get("final_state_bytes"), "final state bytes"
    )
    if final_state_bytes != continuation["state"]["total_bytes"]:
        raise ContinuationParityError("final state bytes differ from tensor manifest")
    expected_continuation = (
        "two-single-token-commits/v1"
        if mode == "tokenwise"
        else "stage-then-atomic-commit-k2/v1"
    )
    if (
        document.get("prefill_mode") != "batched"
        or document.get("continuation_mode") != expected_continuation
    ):
        raise ContinuationParityError("continuation execution mode is invalid")
    stage_base_state = document.get("stage_base_state")
    stage_base_state_equal = document.get("stage_base_state_equal")
    if mode == "block":
        validated_base = _validate_state_manifest(stage_base_state)
        expected_equal = bool(
            validated_base == prefill["state"]
            and validated_base["cursor"] == prefill["end_pos"]
        )
        if (
            not isinstance(stage_base_state_equal, bool)
            or stage_base_state_equal is not expected_equal
        ):
            raise ContinuationParityError(
                "block staging base-state accounting is inconsistent"
            )
    elif stage_base_state is not None or stage_base_state_equal is not None:
        raise ContinuationParityError("tokenwise execution contains block-stage state")
    try:
        direct._verify_access_trace_receipt(document.get("trace"))
    except Exception as exc:
        raise ContinuationParityError(str(exc)) from exc
    return document


def _comparison(
    tokenwise: Mapping[str, Any], block: Mapping[str, Any]
) -> dict[str, Any]:
    prefix_state_differences = _state_diff(
        tokenwise["prefill"]["state"], block["prefill"]["state"]
    )
    continuation_state_differences = _state_diff(
        tokenwise["continuation"]["state"], block["continuation"]["state"]
    )
    block_stage_differences = _state_diff(
        block["prefill"]["state"], block["stage_base_state"]
    )
    result = {
        "block_stage_base_state_differences": block_stage_differences,
        "block_stage_base_state_equal": (
            block["stage_base_state_equal"] is True and not block_stage_differences
        ),
        "bundle_equal": tokenwise["bundle"] == block["bundle"],
        "checkpoint_equal": tokenwise["checkpoint"] == block["checkpoint"],
        "continuation_hidden_equal": (
            tokenwise["continuation"]["hidden_sha256"]
            == block["continuation"]["hidden_sha256"]
        ),
        "continuation_state_differences": continuation_state_differences,
        "continuation_state_equal": not continuation_state_differences,
        "final_cursor_equal": tokenwise["final_cursor"] == block["final_cursor"],
        "prefix_hidden_equal": (
            tokenwise["prefill"]["hidden_sha256"] == block["prefill"]["hidden_sha256"]
        ),
        "prefix_state_differences": prefix_state_differences,
        "prefix_state_equal": not prefix_state_differences,
    }
    result["positive"] = all(
        value for key, value in result.items() if key.endswith("_equal")
    )
    return result


def _contains_gold_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).strip().lower() == "gold" or _contains_gold_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_gold_key(child) for child in value)
    return False


def _validate_result_document(document: dict[str, Any]) -> dict[str, Any]:
    required = {
        "comparison",
        "contract",
        "executions",
        "input",
        "reference",
        "schema",
        "sha256",
        "status",
    }
    if (
        set(document) != required
        or document.get("schema") != RESULT_SCHEMA
        or document.get("status") not in {"positive", "mismatch"}
    ):
        raise ContinuationParityError("continuation parity result schema is invalid")
    _verify_seal(document, "continuation parity result")
    if _contains_gold_key(document):
        raise ContinuationParityError("continuation parity result contains gold data")
    contract = document.get("contract")
    if contract != {
        "continuation_length": CONTINUATION_LENGTH,
        "gold_free": True,
        "kind": CONTRACT_KIND,
        "reference_arm": "off",
        "runtime_instances": 2,
        "same_causal_bundle": True,
    }:
        raise ContinuationParityError("continuation parity contract is invalid")
    input_identity = document.get("input")
    if not isinstance(input_identity, Mapping) or set(input_identity) != {
        "continuation_token_ids",
        "effective_prompt_token_ids",
        "input_sha256",
        "item_id",
        "token_chain_sha256",
    }:
        raise ContinuationParityError("continuation parity input identity is invalid")
    prompt = direct._token_rows(
        input_identity.get("effective_prompt_token_ids"), "parity prompt"
    )
    continuation = direct._token_rows(
        input_identity.get("continuation_token_ids"), "parity continuation"
    )
    if len(continuation) != CONTINUATION_LENGTH:
        raise ContinuationParityError("continuation parity token chain is not K=2")
    _digest(input_identity.get("input_sha256"), "parity input")
    if input_identity.get("token_chain_sha256") != direct._token_chain_sha256(
        prompt, continuation
    ):
        raise ContinuationParityError("continuation parity token-chain seal differs")
    if (
        not isinstance(input_identity.get("item_id"), str)
        or not input_identity["item_id"]
    ):
        raise ContinuationParityError("continuation parity item identity is invalid")
    reference = document.get("reference")
    if not isinstance(reference, Mapping) or set(reference) != {
        "bundle",
        "checkpoint",
        "result_sha256",
        "token_chain_sha256",
    }:
        raise ContinuationParityError("continuation parity reference is invalid")
    _digest(reference.get("result_sha256"), "reference result")
    _digest(reference.get("token_chain_sha256"), "reference token chain")
    if reference["token_chain_sha256"] != input_identity["token_chain_sha256"]:
        raise ContinuationParityError("reference token chain differs from replay")
    executions = document.get("executions")
    if not isinstance(executions, Mapping) or set(executions) != {
        "block",
        "tokenwise",
    }:
        raise ContinuationParityError("continuation parity executions are invalid")
    tokenwise = _validate_execution(executions["tokenwise"], mode="tokenwise")
    block = _validate_execution(executions["block"], mode="block")
    if (
        tokenwise["checkpoint"] != reference["checkpoint"]
        or block["checkpoint"] != reference["checkpoint"]
        or tokenwise["bundle"] != reference["bundle"]
        or block["bundle"] != reference["bundle"]
    ):
        raise ContinuationParityError(
            "execution source identity differs from reference"
        )
    expected_comparison = _comparison(tokenwise, block)
    if document.get("comparison") != expected_comparison:
        raise ContinuationParityError("continuation parity comparison is inconsistent")
    expected_status = "positive" if expected_comparison["positive"] else "mismatch"
    if document["status"] != expected_status:
        raise ContinuationParityError("continuation parity status is inconsistent")
    return document


def _load_inputs_reference(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], tuple[int, ...], tuple[int, ...]]:
    inputs = direct._load_branch_input(args.input)
    reference = direct._load_branch_result(args.reference_off)
    if inputs.get("schema") != direct.BRANCH_INPUT_SCHEMA_V3:
        raise ContinuationParityError("continuation parity requires answer-branch v3")
    if (
        reference.get("arm") != "off"
        or reference.get("schema") != direct.BRANCH_RESULT_SCHEMA
    ):
        raise ContinuationParityError(
            "continuation parity requires an off-v2 reference"
        )
    if reference.get("input") != inputs:
        raise ContinuationParityError(
            "reference result differs from sealed branch input"
        )
    if len(inputs.get("items", ())) != 1 or len(reference.get("items", ())) != 1:
        raise ContinuationParityError("continuation parity requires a one-item cohort")
    source_row = inputs["items"][0]
    reference_row = reference["items"][0]
    prompt = direct._token_rows(
        source_row.get("effective_prompt_token_ids"), "effective parity prompt"
    )
    continuation = direct._token_rows(
        reference_row.get("generated_token_ids"), "reference continuation"
    )
    eos = tuple(int(value) for value in inputs["protocol"]["accepted_eos_token_ids"])
    if (
        len(continuation) != CONTINUATION_LENGTH
        or continuation[0] in eos
        or continuation[-1] not in eos
        or reference_row.get("stopped_on_eos") is not True
        or reference_row.get("eos_token_id") != continuation[-1]
    ):
        raise ContinuationParityError(
            "reference chain must contain one token followed by one EOS token"
        )
    if reference_row.get("token_chain_sha256") != direct._token_chain_sha256(
        prompt, continuation
    ):
        raise ContinuationParityError("reference token-chain seal is inconsistent")
    if getattr(args, "_require_official", True):
        expected = (direct.OFFICIAL_REPO_ID, direct.OFFICIAL_REVISION)
        source_identity = (inputs["source"]["checkpoint"], inputs["source"]["revision"])
        checkpoint_identity = (
            reference["checkpoint"]["repo_id"],
            reference["checkpoint"]["revision"],
        )
        requested_identity = (args.logical_repo_id, args.revision)
        if source_identity != expected or checkpoint_identity != expected:
            raise ContinuationParityError(
                "continuation parity source is not official Qwen"
            )
        if requested_identity != expected:
            raise ContinuationParityError("requested runtime is not official Qwen")
    args._reference_checkpoint_layers = reference["execution"]["checkpoint_layers"]
    return inputs, reference, prompt, continuation


def run(
    args: argparse.Namespace,
    *,
    runtime_factory: Callable[..., tuple[Any, StreamedQwen38]] | None = None,
) -> dict[str, Any]:
    bundle_path = Path(args.causal_bundle).expanduser().resolve()
    try:
        bundle_metadata = bundle_path.lstat()
    except OSError as exc:
        raise ContinuationParityError("causal bundle root is missing") from exc
    if stat.S_ISLNK(bundle_metadata.st_mode) or not stat.S_ISDIR(
        bundle_metadata.st_mode
    ):
        raise ContinuationParityError("causal bundle root must be a plain directory")
    tokenwise_trace = Path(args.access_trace_tokenwise).expanduser().resolve()
    block_trace = Path(args.access_trace_block).expanduser().resolve()
    if tokenwise_trace == block_trace:
        raise ContinuationParityError("continuation access-trace paths must differ")
    inputs, reference, prompt, continuation = _load_inputs_reference(args)
    if len(prompt) + len(continuation) > int(args.max_seq_len):
        raise ContinuationParityError("continuation parity exceeds max_seq_len")
    factory = direct._runtime if runtime_factory is None else runtime_factory
    tokenwise = _run_execution(
        args,
        mode="tokenwise",
        prompt=prompt,
        continuation=continuation,
        trace_path=tokenwise_trace,
        expected_checkpoint=reference["checkpoint"],
        expected_bundle=reference["bundle"],
        expected_bundle_root=bundle_path,
        runtime_factory=factory,
    )
    block = _run_execution(
        args,
        mode="block",
        prompt=prompt,
        continuation=continuation,
        trace_path=block_trace,
        expected_checkpoint=reference["checkpoint"],
        expected_bundle=reference["bundle"],
        expected_bundle_root=bundle_path,
        runtime_factory=factory,
    )
    comparison = _comparison(tokenwise, block)
    identity = {
        "comparison": comparison,
        "contract": {
            "continuation_length": CONTINUATION_LENGTH,
            "gold_free": True,
            "kind": CONTRACT_KIND,
            "reference_arm": "off",
            "runtime_instances": 2,
            "same_causal_bundle": True,
        },
        "executions": {"block": block, "tokenwise": tokenwise},
        "input": {
            "continuation_token_ids": list(continuation),
            "effective_prompt_token_ids": list(prompt),
            "input_sha256": inputs["sha256"],
            "item_id": inputs["items"][0]["item_id"],
            "token_chain_sha256": direct._token_chain_sha256(prompt, continuation),
        },
        "reference": {
            "bundle": reference["bundle"],
            "checkpoint": reference["checkpoint"],
            "result_sha256": reference["sha256"],
            "token_chain_sha256": reference["items"][0]["token_chain_sha256"],
        },
        "schema": RESULT_SCHEMA,
        "status": "positive" if comparison["positive"] else "mismatch",
    }
    return _validate_result_document(_seal(identity))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference-off", required=True)
    parser.add_argument("--causal-bundle", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--access-trace-tokenwise", required=True)
    parser.add_argument("--access-trace-block", required=True)
    parser.add_argument("--source", default=direct.OFFICIAL_REPO_ID)
    parser.add_argument("--logical-repo-id", default=direct.OFFICIAL_REPO_ID)
    parser.add_argument("--revision", default=direct.OFFICIAL_REVISION)
    parser.add_argument("--pinned-inventory")
    parser.add_argument("--cache-dir", default=str(direct.DEFAULT_CACHE))
    parser.add_argument("--max-cache-gb", type=_positive_float, default=1.0)
    parser.add_argument("--source-budget-mb", type=_positive_int, default=262144)
    parser.add_argument("--max-resident-mb", type=_positive_int, default=384)
    parser.add_argument("--max-seq-len", type=_positive_int, default=256)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = run(args)
        output = direct._write_json(args.output, document)
    except (ContinuationParityError, direct.QwenDirectDecodeError) as exc:
        raise SystemExit(f"Qwen continuation parity failed: {exc}") from exc
    print(
        direct.json.dumps(
            {
                "output": str(output),
                "schema": document["schema"],
                "sha256": document["sha256"],
                "status": document["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
