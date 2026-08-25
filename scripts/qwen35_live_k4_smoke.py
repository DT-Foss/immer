#!/usr/bin/env python3
"""Run authenticated local Qwen3.5 K=4 drafting against causal Qwen3.8.

Both checkpoints and the tokenizer remain local and immutable.  Qwen3.5-0.8B
may only propose; Qwen3.8-27B verifies four positions with one transactional
continuation stage and one vocabulary scan.  The default target attention is
the native Prefix-Sinkhorn Head-CRSA path.  An optional fresh tokenwise target
replay proves every committed hidden row and the terminal continuation state.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any

import torch

from immer.runtimes.qwen3_8 import (
    K4SpeculativeGenerationEvidence,
    K4SpeculativeRoundEvidence,
    LogicalModelIdentity,
    NativeHeadCrsaEvidence,
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    QWEN35_K4_DRAFT_PROVIDER_SCHEMA,
    QWEN35_DRAFTER_REPO_ID,
    QWEN35_DRAFTER_REVISION,
    Qwen35K4DraftProvider,
    Qwen38K4SpeculativeDecoder,
    Qwen38Tokenizer,
    Qwen38WeightPager,
    StreamedQwen38,
)


ROOT = Path(__file__).resolve().parent.parent
BASE_SCRIPT = ROOT / "scripts" / "qwen35_live_k2_smoke.py"
DEFAULT_TARGET_BUNDLE = Path("/app/models/Qwen3.8-27B")
DEFAULT_DRAFT_BUNDLE = Path("/app/models/Qwen3.5-0.8B")
DEFAULT_OUTPUT = ROOT / "results" / "qwen35-live-k4-smoke.json"
RESULT_SCHEMA = "immer.qwen3.5-live-k4-smoke/v1"
TRANSACTION_SCHEMA = "immer.qwen3.8-k4-transaction-trace/v1"
PARITY_SCHEMA = "immer.qwen3.8-k4-tokenwise-parity/v1"
MIB = 1024**2


def _load_base() -> ModuleType:
    name = "qwen35_live_k2_smoke"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import live K2 authentication protocol")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()


class LiveK4SmokeError(RuntimeError):
    """The authenticated local K=4 execution contract cannot be completed."""


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def _token_ids(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "token IDs must be comma-separated integers"
        ) from exc
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("token IDs must be non-negative")
    return values


def _canonical_eos_ids(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise LiveK4SmokeError("--eos-token-ids must be an integer sequence")
    try:
        raw = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise LiveK4SmokeError("--eos-token-ids must be an integer sequence") from exc
    if any(isinstance(token, bool) or not isinstance(token, int) for token in raw):
        raise LiveK4SmokeError("--eos-token-ids must contain integers")
    return tuple(sorted(set(raw)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-bundle", default=str(DEFAULT_TARGET_BUNDLE))
    parser.add_argument("--draft-bundle", default=str(DEFAULT_DRAFT_BUNDLE))
    parser.add_argument("--tokenizer-json")
    parser.add_argument("--prompt", default="What is 17 + 25?")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--max-new-tokens", type=_positive_int, default=4)
    parser.add_argument("--expected-token-ids", type=_token_ids)
    parser.add_argument("--eos-token-ids", type=_token_ids, default=())
    parser.add_argument(
        "--attention-mode", choices=("native-crsa", "off"), default="native-crsa"
    )
    parser.add_argument(
        "--parity-control",
        choices=("tokenwise", "none"),
        default="tokenwise",
        help="fresh authenticated target replay used for hidden/state proof",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--target-source-budget-mb", type=_positive_float, default=1_048_576.0
    )
    parser.add_argument(
        "--draft-source-budget-mb", type=_positive_float, default=262_144.0
    )
    parser.add_argument("--max-resident-mb", type=_positive_float, default=384.0)
    parser.add_argument("--max-context-tokens", type=_positive_int, default=2048)
    parser.add_argument(
        "--head-block-rows",
        type=_positive_int,
        default=Qwen38WeightPager.DEFAULT_HEAD_BLOCK_ROWS,
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return parser


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LiveK4SmokeError("result is not canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "sha256" in result:
        raise LiveK4SmokeError("document already contains a seal")
    result["sha256"] = _sha256(result)
    return result


def _state_parts_receipt(
    *,
    cursor: int,
    batch_size: int | None,
    layers: Sequence[Any],
    graft_history: torch.Tensor | None,
    pending: bool,
    poisoned: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any] | None] = []
    total = 0
    for index, state in enumerate(layers):
        if state is None:
            rows.append(None)
            continue
        tensors: dict[str, Any] = {}
        for name in ("key", "value", "crsa_log_usage", "conv", "recurrent"):
            tensor = getattr(state, name, None)
            if tensor is not None:
                receipt = base._tensor_receipt(tensor)
                assert receipt is not None
                total += int(receipt["nbytes"])
                tensors[name] = receipt
        rows.append(
            {"kind": type(state).__qualname__, "layer": index, "tensors": tensors}
        )
    graft = base._tensor_receipt(graft_history)
    if graft is not None:
        total += int(graft["nbytes"])
    body = {
        "batch_size": batch_size,
        "cursor": cursor,
        "graft_history": graft,
        "layers": rows,
        "pending_block": pending,
        "poisoned": poisoned,
        "state_bytes": total,
    }
    return {**body, "sha256": _sha256(body)}


def _validate_tensor_receipt(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "dtype",
        "nbytes",
        "sha256",
        "shape",
    }:
        raise LiveK4SmokeError(f"{label} tensor receipt is invalid")
    dtype_bytes = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8}
    dtype = value.get("dtype")
    shape = value.get("shape")
    if (
        dtype not in dtype_bytes
        or not isinstance(shape, list)
        or not shape
        or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in shape
        )
        or not base._is_sha256(value.get("sha256"))
    ):
        raise LiveK4SmokeError(f"{label} tensor receipt is invalid")
    expected = math.prod(shape) * dtype_bytes[str(dtype)]
    if value.get("nbytes") != expected:
        raise LiveK4SmokeError(f"{label} tensor byte count is invalid")
    return value


def _validate_state_receipt(
    value: object,
    label: str,
    *,
    cursor: int | None = None,
    pending: bool | None = None,
) -> Mapping[str, Any]:
    required = {
        "batch_size",
        "cursor",
        "graft_history",
        "layers",
        "pending_block",
        "poisoned",
        "sha256",
        "state_bytes",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise LiveK4SmokeError(f"{label} state receipt is invalid")
    unsigned = {key: child for key, child in value.items() if key != "sha256"}
    if value.get("sha256") != _sha256(unsigned):
        raise LiveK4SmokeError(f"{label} state receipt seal is invalid")
    if (
        isinstance(value.get("cursor"), bool)
        or not isinstance(value.get("cursor"), int)
        or int(value["cursor"]) < 0
        or not isinstance(value.get("pending_block"), bool)
        or not isinstance(value.get("poisoned"), bool)
        or (value.get("batch_size") is not None and value.get("batch_size") != 1)
    ):
        raise LiveK4SmokeError(f"{label} state metadata is invalid")
    if cursor is not None and value.get("cursor") != cursor:
        raise LiveK4SmokeError(f"{label} state cursor is invalid")
    if pending is not None and value.get("pending_block") is not pending:
        raise LiveK4SmokeError(f"{label} pending-state flag is invalid")
    layers = value.get("layers")
    if not isinstance(layers, list) or not layers:
        raise LiveK4SmokeError(f"{label} state layers are invalid")
    total = 0
    for index, row in enumerate(layers):
        if row is None:
            continue
        if (
            not isinstance(row, Mapping)
            or set(row) != {"kind", "layer", "tensors"}
            or row.get("layer") != index
            or not isinstance(row.get("kind"), str)
            or not isinstance(row.get("tensors"), Mapping)
        ):
            raise LiveK4SmokeError(f"{label} state layer is invalid")
        for name, tensor in row["tensors"].items():
            if name not in {"key", "value", "crsa_log_usage", "conv", "recurrent"}:
                raise LiveK4SmokeError(f"{label} state tensor name is invalid")
            total += int(_validate_tensor_receipt(tensor, f"{label}.{name}")["nbytes"])
    graft = value.get("graft_history")
    if graft is not None:
        total += int(_validate_tensor_receipt(graft, f"{label}.graft")["nbytes"])
    if value.get("state_bytes") != total:
        raise LiveK4SmokeError(f"{label} state byte total is invalid")
    return value


def _validate_native_event(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LiveK4SmokeError(f"{label} native CRSA event is invalid")
    typed = dict(value)
    for name in (
        "selected_query_heads",
        "selected_kv_heads",
        "alpha_per_head",
        "argmax_changed_queries_per_head",
        "mean_l1_probability_delta_per_head",
        "free_heads",
    ):
        if isinstance(typed.get(name), list):
            typed[name] = tuple(typed[name])
    try:
        NativeHeadCrsaEvidence(**typed)
    except (TypeError, ValueError) as exc:
        raise LiveK4SmokeError(f"{label} native CRSA event is invalid") from exc
    if (
        value.get("alpha_per_head") != [0.01, 0.01, 0.01, 0.01]
        or value.get("identity") is not False
    ):
        raise LiveK4SmokeError(f"{label} native CRSA deployment profile changed")
    return value


def _committed_state(model: StreamedQwen38) -> dict[str, Any]:
    return _state_parts_receipt(
        cursor=model.next_position,
        batch_size=model.state_batch_size,
        layers=model._layer_states,
        graft_history=model._graft_history,
        pending=model._pending_block_stage is not None,
        poisoned=model.state_poisoned,
    )


def _native_rows(rows: Sequence[Any]) -> list[dict[str, Any]]:
    return [row.to_dict() for row in rows]


class _TargetTransactionRecorder:
    """Record target stages without exposing a commit handle to the drafter."""

    def __init__(self, model: StreamedQwen38) -> None:
        self.model = model
        self.transactions: list[dict[str, Any]] = []
        self._by_handle: dict[int, dict[str, Any]] = {}
        self._stage = model.stage_continuation_block
        self._commit = model.commit_continuation_block
        self._discard = model.discard_continuation_block
        self._decode = model.decode
        model.stage_continuation_block = self.stage  # type: ignore[method-assign]
        model.commit_continuation_block = self.commit  # type: ignore[method-assign]
        model.discard_continuation_block = self.discard  # type: ignore[method-assign]
        model.decode = self.decode  # type: ignore[method-assign]

    def stage(self, token_ids: Any, **kwargs: Any):
        stage = self._stage(token_ids, **kwargs)
        pending = self.model._pending_block_stage
        if pending is None or pending.handle is not stage:
            raise LiveK4SmokeError("target stage has no private transaction")
        evidence = asdict(stage.evidence)
        inputs = list(evidence["input_token_ids"][0])
        start = int(evidence["start_pos"])
        hidden_positions = [
            {
                "hidden": base._tensor_receipt(stage.hidden[:, offset : offset + 1]),
                "input_token_id": int(token),
                "position": start + offset,
            }
            for offset, token in enumerate(inputs)
        ]
        native = _native_rows(pending.native_head_crsa_evidence)
        row = {
            "committed_hidden": None,
            "committed_hidden_positions": None,
            "committed_state": None,
            "evidence": evidence,
            "hidden_positions": hidden_positions,
            "index": len(self.transactions),
            "native_crsa": {
                "events": native,
                "sha256": _sha256(native),
            },
            "staged_state": _state_parts_receipt(
                cursor=int(evidence["end_pos"]),
                batch_size=1,
                layers=pending.layer_states,
                graft_history=pending.graft_history,
                pending=True,
                poisoned=False,
            ),
            "transition": "staged",
        }
        self.transactions.append(row)
        self._by_handle[id(stage)] = row
        return stage

    def commit(self, stage: Any):
        row = self._by_handle.get(id(stage))
        if row is None:
            raise LiveK4SmokeError("target committed an unrecorded stage")
        hidden, evidence = self._commit(stage)
        committed_evidence = asdict(evidence)
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
        if any(committed_evidence[name] != row["evidence"][name] for name in shared):
            raise LiveK4SmokeError("committed stage evidence changed")
        committed_hidden = base._tensor_receipt(hidden)
        staged_hidden = base._tensor_receipt(stage.hidden)
        if committed_hidden != staged_hidden:
            raise LiveK4SmokeError("committed stage hidden changed")
        committed_state = _committed_state(self.model)
        staged_state = row["staged_state"]
        for name in ("batch_size", "cursor", "graft_history", "layers", "state_bytes"):
            if committed_state[name] != staged_state[name]:
                raise LiveK4SmokeError("committed stage state changed")
        row["transition"] = "committed"
        row["committed_hidden"] = committed_hidden
        row["committed_hidden_positions"] = [
            {
                "hidden": base._tensor_receipt(hidden[:, offset : offset + 1]),
                "input_token_id": position["input_token_id"],
                "position": position["position"],
            }
            for offset, position in enumerate(row["hidden_positions"])
        ]
        if row["committed_hidden_positions"] != row["hidden_positions"]:
            raise LiveK4SmokeError("committed stage position hidden changed")
        row["committed_state"] = committed_state
        return hidden, evidence

    def discard(self, stage: Any) -> None:
        row = self._by_handle.get(id(stage))
        if row is None:
            raise LiveK4SmokeError("target discarded an unrecorded stage")
        self._discard(stage)
        row["transition"] = "discarded"

    def decode(self, token_ids: Any, **kwargs: Any):
        start = self.model.next_position
        hidden, evidence = self._decode(token_ids, **kwargs)
        ids = list(asdict(evidence)["input_token_ids"][0])
        row = {
            "committed_hidden": base._tensor_receipt(hidden),
            "committed_hidden_positions": [
                {
                    "hidden": base._tensor_receipt(hidden[:, offset : offset + 1]),
                    "input_token_id": int(token),
                    "position": start + offset,
                }
                for offset, token in enumerate(ids)
            ],
            "committed_state": _committed_state(self.model),
            "evidence": asdict(evidence),
            "hidden_positions": [
                {
                    "hidden": base._tensor_receipt(hidden[:, offset : offset + 1]),
                    "input_token_id": int(token),
                    "position": start + offset,
                }
                for offset, token in enumerate(ids)
            ],
            "index": len(self.transactions),
            "native_crsa": {"events": [], "sha256": _sha256([])},
            "staged_state": None,
            "transition": "decoded",
        }
        self.transactions.append(row)
        return hidden, evidence

    def close(self) -> None:
        self.model.stage_continuation_block = self._stage  # type: ignore[method-assign]
        self.model.commit_continuation_block = self._commit  # type: ignore[method-assign]
        self.model.discard_continuation_block = self._discard  # type: ignore[method-assign]
        self.model.decode = self._decode  # type: ignore[method-assign]

    def receipt(self) -> dict[str, Any]:
        body = {
            "schema": TRANSACTION_SCHEMA,
            "transactions": self.transactions,
        }
        return {**body, "sha256": _sha256(body)}


class _TimedDraftProvider:
    def __init__(self, provider: Qwen35K4DraftProvider) -> None:
        self.provider = provider
        self.proposal_seconds = 0.0
        self.reconcile_seconds = 0.0
        self.proposals: list[tuple[int, int, int, int]] = []
        self.reconciled_lengths: list[int] = []

    def __call__(self, history: tuple[int, ...], /) -> tuple[int, int, int, int]:
        started = time.perf_counter()
        try:
            proposal = self.provider(history)
        finally:
            self.proposal_seconds += time.perf_counter() - started
        self.proposals.append(proposal)
        return proposal

    def reconcile(self, history: tuple[int, ...], /) -> None:
        started = time.perf_counter()
        try:
            self.provider.reconcile(history)
        finally:
            self.reconcile_seconds += time.perf_counter() - started
        self.reconciled_lengths.append(len(history))


def _acceptance(rounds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    proposed_rows = [row for row in rounds if row["proposed_token_ids"]]
    proposed = sum(len(row["proposed_token_ids"]) for row in proposed_rows)
    accepted = sum(int(row["accepted_prefix_length"]) for row in proposed_rows)
    prefixes = Counter(int(row["accepted_prefix_length"]) for row in proposed_rows)
    replays = Counter(str(row["replay_kind"]) for row in rounds)
    return {
        "accepted_draft_tokens": accepted,
        "accepted_prefix_histogram": {
            str(index): int(prefixes.get(index, 0)) for index in range(5)
        },
        "draft_tokens_proposed": proposed,
        "draft_verification_rounds": len(proposed_rows),
        "rate": 0.0 if proposed == 0 else accepted / proposed,
        "replay_histogram": dict(sorted(replays.items())),
        "target_verification_rounds": len(rounds),
        "terminal_single_rounds": len(rounds) - len(proposed_rows),
    }


def _round_object(row: Mapping[str, Any]) -> K4SpeculativeRoundEvidence:
    values = dict(row)
    for name in (
        "eos_token_ids",
        "proposed_token_ids",
        "target_token_ids",
        "emitted_token_ids",
    ):
        values[name] = tuple(values[name])
    return K4SpeculativeRoundEvidence(**values)


def _committed_position_rows(
    transactions: Mapping[str, Any], native_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    positions: list[dict[str, Any]] = []
    for transaction in transactions["transactions"]:
        if transaction["transition"] not in {"committed", "decoded"}:
            continue
        for hidden in transaction["hidden_positions"]:
            positions.append(
                {
                    **hidden,
                    "state_after_position": (
                        transaction["committed_state"]
                        if hidden["position"] == transaction["evidence"]["end_pos"] - 1
                        else None
                    ),
                    "transaction_index": transaction["index"],
                }
            )
    by_position = {int(row["query_start"]): row for row in native_rows}
    for row in positions:
        row["native_crsa"] = by_position.get(int(row["position"]))
    return sorted(positions, key=lambda row: int(row["position"]))


def _tokenwise_parity(
    args: argparse.Namespace,
    *,
    target_path: Path,
    prompt: tuple[int, ...],
    generated: tuple[int, ...],
    live_positions: Sequence[Mapping[str, Any]],
    live_state: Mapping[str, Any],
    live_native: Sequence[Mapping[str, Any]],
    require_production_profile: bool,
    require_official_target: bool,
) -> dict[str, Any]:
    native_evidence: list[Any] = []
    owned = base._open_model(
        role="target",
        bundle=target_path,
        identity=LogicalModelIdentity(OFFICIAL_REPO_ID, OFFICIAL_REVISION),
        source_budget_mb=args.target_source_budget_mb,
        device=args.device,
        compute_dtype=args.compute_dtype,
        max_resident_bytes=int(args.max_resident_mb * MIB),
        max_context_tokens=args.max_context_tokens,
        attention_mode=args.attention_mode,
        native_evidence=native_evidence,
        require_production_profile=require_production_profile,
        require_official_target=require_official_target,
    )
    try:
        pager_before = base._pager_counters(owned.pager)
        started = time.perf_counter()
        owned.model.prefill([prompt], reset=True, tokenwise=False)
        positions: list[dict[str, Any]] = []
        for token in generated:
            hidden, _evidence = owned.model.decode([[token]])
            position = owned.model.next_position - 1
            current_native = _native_rows(native_evidence)
            positions.append(
                {
                    "hidden": base._tensor_receipt(hidden),
                    "input_token_id": token,
                    "native_crsa": next(
                        (
                            row
                            for row in reversed(current_native)
                            if row["query_start"] == position
                        ),
                        None,
                    ),
                    "position": position,
                    "state_after_position": _committed_state(owned.model),
                }
            )
        wall_seconds = time.perf_counter() - started
        state = _committed_state(owned.model)
        native = _native_rows(native_evidence)
        hidden_equal = [
            left["hidden"] == right["hidden"]
            and left["position"] == right["position"]
            and left["input_token_id"] == right["input_token_id"]
            for left, right in zip(live_positions, positions, strict=True)
        ]
        body = {
            "bundle": owned.bundle_receipt,
            "comparison": {
                "crsa_equal": list(live_native) == native,
                "final_state_equal": dict(live_state) == state,
                "hidden_equal_by_position": hidden_equal,
                "hidden_equal": len(hidden_equal) == len(generated)
                and all(hidden_equal),
                "positive": False,
            },
            "input": {
                "generated_token_ids": list(generated),
                "prompt_token_ids": list(prompt),
                "token_chain_sha256": _sha256(
                    {"generated": list(generated), "prompt": list(prompt)}
                ),
            },
            "mode": "four-tokenwise-commits/v1",
            "native_crsa": {"events": native, "sha256": _sha256(native)},
            "pager": {
                "counters": base._counter_delta(
                    pager_before, base._pager_counters(owned.pager)
                ),
                "device": owned.pager.resolved_device,
                "dtype": owned.pager.resolved_dtype,
            },
            "positions": positions,
            "schema": PARITY_SCHEMA,
            "state": state,
            "timing": {
                "bundle_verify_seconds": owned.verify_seconds,
                "preflight_seconds": owned.preflight_seconds,
                "wall_seconds": wall_seconds,
            },
        }
        comparison = body["comparison"]
        comparison["positive"] = bool(
            comparison["crsa_equal"]
            and comparison["final_state_equal"]
            and comparison["hidden_equal"]
        )
        return {**body, "sha256": _sha256(body)}
    finally:
        owned.close()


def _validate_generation_evidence(value: object) -> K4SpeculativeGenerationEvidence:
    if not isinstance(value, Mapping):
        raise LiveK4SmokeError("target speculative evidence is absent")
    values = dict(value)
    try:
        values["prompt_token_ids"] = tuple(values["prompt_token_ids"])
        values["eos_token_ids"] = tuple(values["eos_token_ids"])
        values["generated_token_ids"] = tuple(values["generated_token_ids"])
        values["rounds"] = tuple(_round_object(row) for row in values["rounds"])
        evidence = K4SpeculativeGenerationEvidence(**values)
    except (TypeError, ValueError, KeyError) as exc:
        raise LiveK4SmokeError("target speculative evidence is invalid") from exc
    if evidence.to_dict() != dict(value):
        raise LiveK4SmokeError("target speculative evidence changed on reconstruction")
    return evidence


def _validate_transaction_state_pair(
    staged: Mapping[str, Any], committed: Mapping[str, Any], label: str
) -> None:
    for name in ("batch_size", "cursor", "graft_history", "layers", "state_bytes"):
        if staged[name] != committed[name]:
            raise LiveK4SmokeError(f"{label} staged/committed state differs")


def _validate_transaction_row(
    value: object,
    *,
    index: int,
    transition: str,
    input_ids: Sequence[int],
    start_pos: int,
    attention_mode: str,
) -> Mapping[str, Any]:
    required = {
        "committed_hidden",
        "committed_hidden_positions",
        "committed_state",
        "evidence",
        "hidden_positions",
        "index",
        "native_crsa",
        "staged_state",
        "transition",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise LiveK4SmokeError("target transaction row is invalid")
    if value.get("index") != index or value.get("transition") != transition:
        raise LiveK4SmokeError("target transaction order is invalid")
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping):
        raise LiveK4SmokeError("target transaction evidence is invalid")
    end_pos = start_pos + len(input_ids)
    raw_inputs = evidence.get("input_token_ids")
    if (
        evidence.get("start_pos") != start_pos
        or evidence.get("end_pos") != end_pos
        or raw_inputs != [list(input_ids)]
        and raw_inputs != (tuple(input_ids),)
        or evidence.get("complete_layer_stack") is not True
        or evidence.get("stateful_cache") is not True
        or evidence.get("layers_executed") != evidence.get("checkpoint_layers")
        or not isinstance(evidence.get("source_body_bytes"), int)
        or int(evidence["source_body_bytes"]) < 0
        or not isinstance(evidence.get("linear_calls"), int)
        or int(evidence["linear_calls"]) < 0
        or not isinstance(evidence.get("seconds"), (int, float))
        or not math.isfinite(float(evidence["seconds"]))
        or float(evidence["seconds"]) < 0.0
    ):
        raise LiveK4SmokeError("target transaction evidence is inconsistent")
    hidden_positions = value.get("hidden_positions")
    if not isinstance(hidden_positions, list) or len(hidden_positions) != len(
        input_ids
    ):
        raise LiveK4SmokeError("target transaction hidden trace is incomplete")
    for offset, (row, token) in enumerate(
        zip(hidden_positions, input_ids, strict=True)
    ):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"hidden", "input_token_id", "position"}
            or row.get("position") != start_pos + offset
            or row.get("input_token_id") != token
        ):
            raise LiveK4SmokeError("target transaction position is inconsistent")
        _validate_tensor_receipt(row.get("hidden"), "target transaction hidden")
    native = value.get("native_crsa")
    if (
        not isinstance(native, Mapping)
        or set(native) != {"events", "sha256"}
        or not isinstance(native.get("events"), list)
        or native.get("sha256") != _sha256(native.get("events"))
    ):
        raise LiveK4SmokeError("target transaction native receipt is invalid")
    for event in native["events"]:
        _validate_native_event(event, "target transaction")
    if transition in {"staged", "committed", "discarded"}:
        staged = _validate_state_receipt(
            value.get("staged_state"),
            "target staged",
            cursor=end_pos,
            pending=True,
        )
        if staged.get("poisoned") is not False:
            raise LiveK4SmokeError("target staged state is poisoned")
        if staged.get("state_bytes") != evidence.get("staged_state_bytes"):
            raise LiveK4SmokeError("target staged state bytes differ from evidence")
        expected_events = len(input_ids) if attention_mode == "native-crsa" else 0
        if len(native["events"]) != expected_events:
            raise LiveK4SmokeError("target staged CRSA position trace is incomplete")
        if attention_mode == "native-crsa" and [
            row["query_start"] for row in native["events"]
        ] != list(range(start_pos, end_pos)):
            raise LiveK4SmokeError("target staged CRSA positions are inconsistent")
    elif value.get("staged_state") is not None or native["events"]:
        raise LiveK4SmokeError("target decode contains private staged state")
    if transition == "committed":
        committed = _validate_state_receipt(
            value.get("committed_state"),
            "target committed",
            cursor=end_pos,
            pending=False,
        )
        if committed.get("poisoned") is not False:
            raise LiveK4SmokeError("target committed state is poisoned")
        _validate_transaction_state_pair(staged, committed, "target transaction")
        hidden = _validate_tensor_receipt(
            value.get("committed_hidden"), "target committed hidden"
        )
        if hidden["shape"][0] != 1 or hidden["shape"][1] != len(input_ids):
            raise LiveK4SmokeError("target committed hidden width is invalid")
        if value.get("committed_hidden_positions") != hidden_positions:
            raise LiveK4SmokeError("target committed position hidden differs")
    elif transition == "decoded":
        committed = _validate_state_receipt(
            value.get("committed_state"),
            "target decoded",
            cursor=end_pos,
            pending=False,
        )
        if committed.get("poisoned") is not False:
            raise LiveK4SmokeError("target decoded state is poisoned")
        hidden = _validate_tensor_receipt(
            value.get("committed_hidden"), "target decoded hidden"
        )
        if hidden["shape"][0] != 1 or hidden["shape"][1] != len(input_ids):
            raise LiveK4SmokeError("target decoded hidden width is invalid")
        if value.get("committed_hidden_positions") != hidden_positions:
            raise LiveK4SmokeError("target decoded position hidden differs")
    elif (
        value.get("committed_hidden") is not None
        or value.get("committed_hidden_positions") is not None
        or value.get("committed_state") is not None
    ):
        raise LiveK4SmokeError("uncommitted target stage contains committed state")
    return value


def _expected_transaction_plan(
    rounds: Sequence[Mapping[str, Any]],
) -> list[tuple[str, list[int], int]]:
    plan: list[tuple[str, list[int], int]] = []
    for row in rounds:
        start = int(row["start_pos"])
        replay = str(row["replay_kind"])
        proposal = list(row["proposed_token_ids"])
        emitted = list(row["emitted_token_ids"])
        if replay == "terminal-single":
            plan.append(("decoded", emitted, start))
        elif replay in {"commit-k4", "eos3-commit-k4"}:
            plan.append(("committed", proposal, start))
        elif replay in {"mismatch0-decode", "eos0-decode"}:
            plan.extend((("discarded", proposal, start), ("decoded", emitted, start)))
        elif replay in {
            "mismatch1-restage",
            "mismatch2-restage",
            "mismatch3-restage",
            "eos1-restage",
            "eos2-restage",
            "eos3-restage",
        }:
            plan.extend((("discarded", proposal, start), ("committed", emitted, start)))
        else:
            raise LiveK4SmokeError("target replay kind has no transaction grammar")
    return plan


def _validate_transactions(
    value: object,
    *,
    rounds: Sequence[Mapping[str, Any]],
    native_rows: Sequence[Mapping[str, Any]],
    attention_mode: str,
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "sha256",
        "transactions",
    }:
        raise LiveK4SmokeError("target transaction trace is invalid")
    unsigned = {key: child for key, child in value.items() if key != "sha256"}
    if (
        value.get("schema") != TRANSACTION_SCHEMA
        or value.get("sha256") != _sha256(unsigned)
        or not isinstance(value.get("transactions"), list)
    ):
        raise LiveK4SmokeError("target transaction receipt is invalid")
    plan = _expected_transaction_plan(rounds)
    rows = value["transactions"]
    if len(rows) != len(plan):
        raise LiveK4SmokeError("target transaction count differs from replay grammar")
    for index, (row, expected) in enumerate(zip(rows, plan, strict=True)):
        transition, inputs, start = expected
        _validate_transaction_row(
            row,
            index=index,
            transition=transition,
            input_ids=inputs,
            start_pos=start,
            attention_mode=attention_mode,
        )
    positions = _committed_position_rows(value, native_rows)
    position_set = {position["position"] for position in positions}
    continuation_native = [
        row for row in native_rows if row["query_start"] in position_set
    ]
    if attention_mode == "native-crsa":
        by_position = {row["query_start"]: row for row in continuation_native}
        transaction_native: list[Mapping[str, Any]] = []
        for row in rows:
            if row["transition"] == "committed":
                transaction_native.extend(row["native_crsa"]["events"])
            elif row["transition"] == "decoded":
                transaction_native.extend(
                    by_position[position["position"]]
                    for position in row["hidden_positions"]
                )
        if transaction_native != continuation_native:
            raise LiveK4SmokeError(
                "committed transaction CRSA trace differs from target"
            )
    elif continuation_native or any(row["native_crsa"]["events"] for row in rows):
        raise LiveK4SmokeError("off transaction emitted native CRSA evidence")
    return value, positions


def _validate_tokenwise_parity(
    value: object,
    *,
    prompt_ids: Sequence[int],
    generated: Sequence[int],
    target_bundle: Mapping[str, Any],
    target_positions: Sequence[Mapping[str, Any]],
    target_state: Mapping[str, Any],
    target_native: Sequence[Mapping[str, Any]],
    attention_mode: str,
) -> Mapping[str, Any]:
    required = {
        "bundle",
        "comparison",
        "input",
        "mode",
        "native_crsa",
        "pager",
        "positions",
        "schema",
        "sha256",
        "state",
        "timing",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise LiveK4SmokeError("tokenwise parity receipt is invalid")
    unsigned = {key: child for key, child in value.items() if key != "sha256"}
    if (
        value.get("schema") != PARITY_SCHEMA
        or value.get("mode") != "four-tokenwise-commits/v1"
        or value.get("sha256") != _sha256(unsigned)
        or value.get("bundle") != target_bundle
    ):
        raise LiveK4SmokeError("tokenwise parity identity is invalid")
    expected_input = {
        "generated_token_ids": list(generated),
        "prompt_token_ids": list(prompt_ids),
        "token_chain_sha256": _sha256(
            {"generated": list(generated), "prompt": list(prompt_ids)}
        ),
    }
    if value.get("input") != expected_input:
        raise LiveK4SmokeError("tokenwise parity input differs from live execution")
    state = _validate_state_receipt(
        value.get("state"),
        "tokenwise final",
        cursor=len(prompt_ids) + len(generated),
        pending=False,
    )
    positions = value.get("positions")
    if not isinstance(positions, list) or len(positions) != len(generated):
        raise LiveK4SmokeError("tokenwise position trace is incomplete")
    for offset, (row, token) in enumerate(zip(positions, generated, strict=True)):
        position = len(prompt_ids) + offset
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "hidden",
                "input_token_id",
                "native_crsa",
                "position",
                "state_after_position",
            }
            or row.get("position") != position
            or row.get("input_token_id") != token
        ):
            raise LiveK4SmokeError("tokenwise position trace is inconsistent")
        _validate_tensor_receipt(row.get("hidden"), "tokenwise hidden")
        _validate_state_receipt(
            row.get("state_after_position"),
            "tokenwise position",
            cursor=position + 1,
            pending=False,
        )
        if attention_mode == "native-crsa":
            _validate_native_event(row.get("native_crsa"), "tokenwise position")
        elif row.get("native_crsa") is not None:
            raise LiveK4SmokeError("off tokenwise position emitted native CRSA")
    native = value.get("native_crsa")
    if (
        not isinstance(native, Mapping)
        or set(native) != {"events", "sha256"}
        or not isinstance(native.get("events"), list)
        or native.get("sha256") != _sha256(native.get("events"))
    ):
        raise LiveK4SmokeError("tokenwise native CRSA receipt is invalid")
    for event in native["events"]:
        _validate_native_event(event, "tokenwise")
    control_native_by_position = {
        row["query_start"]: row
        for row in native["events"]
        if row["query_start"] >= len(prompt_ids)
    }
    if attention_mode == "native-crsa" and any(
        row["native_crsa"] != control_native_by_position.get(row["position"])
        for row in positions
    ):
        raise LiveK4SmokeError("tokenwise position CRSA copy differs from global trace")
    hidden_equal = [
        left["hidden"] == right["hidden"]
        and left["position"] == right["position"]
        and left["input_token_id"] == right["input_token_id"]
        for left, right in zip(target_positions, positions, strict=True)
    ]
    expected_comparison = {
        "crsa_equal": list(target_native) == native["events"],
        "final_state_equal": dict(target_state) == state,
        "hidden_equal_by_position": hidden_equal,
        "hidden_equal": len(hidden_equal) == len(generated) and all(hidden_equal),
        "positive": False,
    }
    expected_comparison["positive"] = bool(
        expected_comparison["crsa_equal"]
        and expected_comparison["final_state_equal"]
        and expected_comparison["hidden_equal"]
    )
    if (
        value.get("comparison") != expected_comparison
        or not expected_comparison["positive"]
    ):
        raise LiveK4SmokeError("tokenwise parity comparison is inconsistent")
    return value


def _validate_bundle_receipt(value: object, label: str) -> Mapping[str, Any]:
    required = {
        "checkpoint_bytes",
        "graph_revision",
        "kind",
        "layout_fingerprint",
        "manifest_sha256",
        "repo_id",
        "revision",
        "shards",
        "shards_sha256",
        "tensor_bindings",
        "weights_layout",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("kind") != "complete-causal-bundle/v1"
        or value.get("weights_layout") not in {"flat/v1", "nested/v1"}
        or not isinstance(value.get("graph_revision"), list)
        or len(value["graph_revision"]) != 2
        or not isinstance(value["graph_revision"][0], int)
        or int(value["graph_revision"][0]) < 0
        or not base._is_sha256(value["graph_revision"][1])
        or any(
            not base._is_sha256(value.get(name))
            for name in ("layout_fingerprint", "manifest_sha256", "shards_sha256")
        )
        or any(
            isinstance(value.get(name), bool)
            or not isinstance(value.get(name), int)
            or int(value[name]) < 1
            for name in ("checkpoint_bytes", "shards", "tensor_bindings")
        )
    ):
        raise LiveK4SmokeError(f"{label} bundle receipt is invalid")
    return value


def _validate_result(
    document: Mapping[str, Any],
    *,
    require_production_profile: bool = True,
    require_official_target: bool = True,
) -> dict[str, Any]:
    required = {
        "acceptance",
        "contract",
        "drafter",
        "parity",
        "prompt",
        "schema",
        "sha256",
        "status",
        "target",
        "tokens",
    }
    if set(document) != required or document.get("schema") != RESULT_SCHEMA:
        raise LiveK4SmokeError("result schema is invalid")
    unsealed = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != _sha256(unsealed):
        raise LiveK4SmokeError("result seal is invalid")
    contract = document.get("contract")
    prompt = document.get("prompt")
    tokens = document.get("tokens")
    target = document.get("target")
    drafter = document.get("drafter")
    parity = document.get("parity")
    if not all(
        isinstance(value, Mapping)
        for value in (contract, prompt, tokens, target, drafter, parity)
    ):
        raise LiveK4SmokeError("result sections must be objects")
    assert isinstance(contract, Mapping)
    assert isinstance(prompt, Mapping)
    assert isinstance(tokens, Mapping)
    assert isinstance(target, Mapping)
    assert isinstance(drafter, Mapping)
    assert isinstance(parity, Mapping)
    generated = tokens.get("generated_token_ids")
    prompt_ids = prompt.get("token_ids")
    rounds = tokens.get("rounds")
    if (
        contract.get("k") != 4
        or contract.get("bundle_authentication")
        != "pinned-causal-manifest+range-identity/v1"
        or contract.get("receipt_security") != "self-sealed-integrity/v1"
        or contract.get("target_k4_stage_per_full_round") != 1
        or contract.get("target_head_scans_per_round") != 1
        or contract.get("local_only") is not True
        or contract.get("remote_io") is not False
        or contract.get("checkpoint_mutation") is not False
        or contract.get("target_commits_only") is not True
        or not isinstance(contract.get("max_new_tokens"), int)
        or int(contract["max_new_tokens"]) < 4
        or not isinstance(generated, list)
        or not isinstance(prompt_ids, list)
        or not prompt_ids
        or not isinstance(rounds, list)
        or not rounds
    ):
        raise LiveK4SmokeError("K4 execution contract is inconsistent")
    try:
        for row in rounds:
            _round_object(row)
    except (TypeError, ValueError, KeyError) as exc:
        raise LiveK4SmokeError("K4 round evidence is invalid") from exc
    emitted = [token for row in rounds for token in row["emitted_token_ids"]]
    if emitted != generated:
        raise LiveK4SmokeError("round token trace is inconsistent")
    if document.get("acceptance") != _acceptance(rounds):
        raise LiveK4SmokeError("acceptance accounting is inconsistent")
    if len(generated) > int(contract["max_new_tokens"]):
        raise LiveK4SmokeError("generated output exceeds the declared limit")
    target_bundle = _validate_bundle_receipt(target.get("bundle"), "target")
    draft_bundle = _validate_bundle_receipt(drafter.get("bundle"), "draft")
    rendered = Qwen38Tokenizer.render_no_thinking_prompt(
        str(prompt.get("system", "")), str(prompt.get("user", ""))
    )
    tokenizer = prompt.get("tokenizer")
    if (
        prompt.get("token_count") != len(prompt_ids)
        or prompt.get("rendered_sha256")
        != hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        or not isinstance(tokenizer, Mapping)
        or tokenizer.get("kind") != "local-tokenizers-json/v1"
        or not base._is_sha256(tokenizer.get("sha256"))
        or not isinstance(tokenizer.get("size_bytes"), int)
        or int(tokenizer["size_bytes"]) < 1
    ):
        raise LiveK4SmokeError("prompt/tokenizer receipt is invalid")
    final_cursor = len(prompt_ids) + len(generated)
    target_state = _validate_state_receipt(
        target.get("state"), "target final", cursor=final_cursor, pending=False
    )
    if target_state.get("poisoned") is not False:
        raise LiveK4SmokeError("target final state is poisoned")
    alignment = drafter.get("alignment")
    if not isinstance(alignment, Mapping):
        raise LiveK4SmokeError("drafter alignment is absent")
    draft_cursor = int(alignment.get("committed_cursor", -1))
    draft_state = _validate_state_receipt(
        drafter.get("state"), "draft final", cursor=draft_cursor, pending=False
    )
    if draft_state.get("poisoned") is not False:
        raise LiveK4SmokeError("draft final state is poisoned")
    suffix = alignment.get("target_only_suffix_ids")
    if (
        drafter["state"].get("cursor") != draft_cursor
        or alignment.get("target_cursor") != final_cursor
        or not isinstance(suffix, list)
        or suffix != [*prompt_ids, *generated][draft_cursor:]
        or len(suffix) > 3
    ):
        raise LiveK4SmokeError("drafter/target cursor alignment is inconsistent")
    provider = drafter.get("provider")
    histogram = document["acceptance"]["accepted_prefix_histogram"]
    proposal_rows = [row for row in rounds if row["proposed_token_ids"]]
    restaged = sum(
        row["replay_kind"]
        in {
            "mismatch1-restage",
            "mismatch2-restage",
            "mismatch3-restage",
            "eos1-restage",
            "eos2-restage",
            "eos3-restage",
        }
        for row in proposal_rows
    )
    provider_expected = {
        "prefill_calls": 1,
        "draft_calls": len(proposal_rows),
        "extension_calls": 3 * len(proposal_rows),
        "reconcile_calls": len(proposal_rows),
        "restaged_blocks": restaged,
        "committed_tokens": sum(len(row["emitted_token_ids"]) for row in proposal_rows),
        "state_bytes": draft_state["state_bytes"],
        "pending": False,
        "poisoned": False,
    }
    if (
        not isinstance(provider, Mapping)
        or provider.get("schema") != QWEN35_K4_DRAFT_PROVIDER_SCHEMA
        or any(
            provider.get(f"accepted_prefix_{index}") != histogram[str(index)]
            for index in range(5)
        )
        or any(
            provider.get(name) != expected
            for name, expected in provider_expected.items()
        )
    ):
        raise LiveK4SmokeError("drafter prefix accounting is inconsistent")
    trace = drafter.get("trace")
    expected_reconciled: list[int] = []
    cursor = len(prompt_ids)
    for row in proposal_rows:
        cursor += len(row["emitted_token_ids"])
        expected_reconciled.append(cursor)
    if trace != {
        "proposals": [row["proposed_token_ids"] for row in proposal_rows],
        "reconciled_history_lengths": expected_reconciled,
    }:
        raise LiveK4SmokeError("drafter proposal/reconcile trace is inconsistent")
    native = target.get("native_crsa")
    if (
        not isinstance(native, Mapping)
        or set(native) != {"events", "sha256"}
        or not isinstance(native.get("events"), list)
        or native.get("sha256") != _sha256(native.get("events"))
    ):
        raise LiveK4SmokeError("native CRSA receipt is invalid")
    native_rows = native["events"]
    for event in native_rows:
        _validate_native_event(event, "target")
    attention_mode = contract.get("attention_mode")
    if attention_mode == "native-crsa":
        expected_attention = {
            "alpha": 0.01,
            "balance_alpha": 1.0,
            "diagonal_debit": 3.0,
            "free_softmax_heads": 20,
            "layer": 27,
            "selected_query_heads": [2, 8, 14, 20],
            "selected_sinkhorn_heads": 4,
        }
        if contract.get("attention") != expected_attention:
            raise LiveK4SmokeError("native CRSA contract profile changed")
        if len(native_rows) != 1 + len(generated):
            raise LiveK4SmokeError("native CRSA target history is incomplete")
        expected_spans = [(0, len(prompt_ids))] + [
            (len(prompt_ids) + offset, 1) for offset in range(len(generated))
        ]
        if (
            any(
                row.get("query_start") != start
                or row.get("query_length") != width
                or row.get("history_length_before") != start
                or row.get("history_length_after") != start + width
                or row.get("key_length") != start + width
                for row, (start, width) in zip(native_rows, expected_spans, strict=True)
            )
            or native_rows[-1].get("history_length_after") != final_cursor
        ):
            raise LiveK4SmokeError("native CRSA target history is inconsistent")
    elif attention_mode == "off":
        target_config = target.get("config")
        heads = (
            target_config.get("heads") if isinstance(target_config, Mapping) else None
        )
        if contract.get("attention") != {
            "free_softmax_heads": heads,
            "selected_sinkhorn_heads": 0,
        } or native.get("events"):
            raise LiveK4SmokeError("off mode emitted native CRSA evidence")
    else:
        raise LiveK4SmokeError("attention mode is invalid")
    _transaction, derived_positions = _validate_transactions(
        target.get("transactions"),
        rounds=rounds,
        native_rows=native_rows,
        attention_mode=str(attention_mode),
    )
    positions = target.get("positions")
    if positions != derived_positions or [
        row.get("position") for row in derived_positions
    ] != list(range(len(prompt_ids), final_cursor)):
        raise LiveK4SmokeError("per-position target receipt is incomplete")
    if (
        not derived_positions
        or derived_positions[-1].get("state_after_position") != target_state
    ):
        raise LiveK4SmokeError("final transaction state differs from target state")
    for row in derived_positions:
        _validate_tensor_receipt(row.get("hidden"), "target position hidden")
        state = row.get("state_after_position")
        if state is not None:
            _validate_state_receipt(
                state,
                "target position",
                cursor=int(row["position"]) + 1,
                pending=False,
            )
        if attention_mode == "native-crsa":
            _validate_native_event(row.get("native_crsa"), "target position")
        elif row.get("native_crsa") is not None:
            raise LiveK4SmokeError("off target position emitted native CRSA")
    speculative = _validate_generation_evidence(target.get("speculative_evidence"))
    if (
        list(speculative.prompt_token_ids) != prompt_ids
        or list(speculative.generated_token_ids) != generated
        or list(speculative.eos_token_ids) != contract.get("eos_token_ids")
        or speculative.to_dict()["rounds"] != rounds
        or speculative.accepted_draft_tokens
        != document["acceptance"]["accepted_draft_tokens"]
        or speculative.head_scans != len(rounds)
        or speculative.state_bytes != target_state["state_bytes"]
    ):
        raise LiveK4SmokeError("target speculative evidence is not cross-linked")
    eos_ids = set(contract.get("eos_token_ids", ()))
    short = len(generated) < int(contract["max_new_tokens"])
    if short and (
        speculative.stopped_on_eos is not True
        or not generated
        or generated[-1] not in eos_ids
    ):
        raise LiveK4SmokeError("short output did not terminate on a declared EOS")
    if speculative.stopped_on_eos and (not generated or generated[-1] not in eos_ids):
        raise LiveK4SmokeError("EOS stop does not end in a declared EOS token")
    tokenwise = parity.get("tokenwise_control")
    if contract.get("parity_control") == "tokenwise":
        _validate_tokenwise_parity(
            tokenwise,
            prompt_ids=prompt_ids,
            generated=generated,
            target_bundle=target_bundle,
            target_positions=derived_positions,
            target_state=target_state,
            target_native=native_rows,
            attention_mode=str(attention_mode),
        )
    elif tokenwise is not None:
        raise LiveK4SmokeError("disabled parity control emitted a receipt")
    expected = parity.get("expected_token_ids")
    exact = parity.get("exact")
    if parity.get("provided") is True:
        if not isinstance(expected, list) or exact is not (expected == generated):
            raise LiveK4SmokeError("independent parity receipt is inconsistent")
    elif expected is not None or exact is not None:
        raise LiveK4SmokeError("absent independent parity must be null")
    expected_status = "mismatch" if exact is False else "positive"
    if document.get("status") != expected_status:
        raise LiveK4SmokeError("result status disagrees with independent parity")
    if tokens.get("token_chain_sha256") != _sha256(
        {"generated": generated, "prompt": prompt_ids}
    ) or tokens.get("evidence_chain_sha256") != _sha256(
        [row["evidence_sha256"] for row in rounds]
    ):
        raise LiveK4SmokeError("token/evidence chain receipt is inconsistent")
    pinned = contract.get("pinned_production_profiles")
    official = contract.get("official_target_config")
    if not isinstance(pinned, bool) or not isinstance(official, bool):
        raise LiveK4SmokeError("production verification policy is invalid")
    if require_production_profile and pinned is not True:
        raise LiveK4SmokeError("production profile verification cannot be downgraded")
    if require_official_target and official is not True:
        raise LiveK4SmokeError("official target verification cannot be downgraded")
    if pinned:
        base._validate_profile(target_bundle, base._TARGET_PROFILE, "target")
        base._validate_profile(draft_bundle, base._DRAFT_PROFILE, "draft")
        if (
            target_bundle.get("repo_id") != OFFICIAL_REPO_ID
            or target_bundle.get("revision") != OFFICIAL_REVISION
            or draft_bundle.get("repo_id") != QWEN35_DRAFTER_REPO_ID
            or draft_bundle.get("revision") != QWEN35_DRAFTER_REVISION
        ):
            raise LiveK4SmokeError("production bundle identity is invalid")
    return dict(document)


def _guard_output_path(
    output: str | os.PathLike[str],
    *,
    target_bundle: Path,
    draft_bundle: Path,
    tokenizer: Path,
) -> Path:
    destination = Path(output).expanduser().resolve()
    protected = {
        "target bundle": target_bundle.resolve(),
        "draft bundle": draft_bundle.resolve(),
    }
    for label, root in protected.items():
        if destination == root or destination.is_relative_to(root):
            raise LiveK4SmokeError(f"--output must not modify the {label}")
    if destination == tokenizer.resolve():
        raise LiveK4SmokeError("--output must not replace the tokenizer")
    return destination


def run(
    args: argparse.Namespace,
    *,
    require_production_profile: bool = True,
    require_official_target: bool = True,
) -> tuple[dict[str, Any], Path]:
    if not isinstance(args.prompt, str) or not args.prompt.strip():
        raise LiveK4SmokeError("--prompt must contain text")
    if not isinstance(args.system_prompt, str):
        raise LiveK4SmokeError("--system-prompt must be text")
    if (
        isinstance(args.max_new_tokens, bool)
        or not isinstance(args.max_new_tokens, int)
        or args.max_new_tokens < 4
    ):
        raise LiveK4SmokeError("--max-new-tokens must be at least four")
    eos_token_ids = _canonical_eos_ids(args.eos_token_ids)
    if args.expected_token_ids is not None and len(args.expected_token_ids) != (
        args.max_new_tokens
    ):
        raise LiveK4SmokeError(
            "--expected-token-ids must contain exactly --max-new-tokens IDs"
        )
    target_path = Path(args.target_bundle).expanduser().resolve()
    draft_path = Path(args.draft_bundle).expanduser().resolve()
    if target_path == draft_path:
        raise LiveK4SmokeError("target and draft bundles must be independent")
    tokenizer_path = (
        Path(args.tokenizer_json).expanduser().resolve()
        if args.tokenizer_json
        else target_path / "tokenizer.json"
    )
    output_path = _guard_output_path(
        args.output,
        target_bundle=target_path,
        draft_bundle=draft_path,
        tokenizer=tokenizer_path,
    )
    tokenizer_sha, tokenizer_size = base._file_sha256(tokenizer_path, "local tokenizer")
    tokenizer = Qwen38Tokenizer(
        tokenizer_path, require_official=require_official_target
    )
    if base._file_sha256(tokenizer_path, "local tokenizer")[0] != tokenizer_sha:
        raise LiveK4SmokeError("local tokenizer changed while it was loaded")
    rendered = tokenizer.render_no_thinking_prompt(args.system_prompt, args.prompt)
    prompt = tuple(tokenizer.encode(rendered))
    if not prompt:
        raise LiveK4SmokeError("local tokenizer produced an empty prompt")
    if len(prompt) + args.max_new_tokens > args.max_context_tokens:
        raise LiveK4SmokeError("prompt plus output exceeds --max-context-tokens")

    target = draft = None
    provider = None
    recorder = None
    primary: BaseException | None = None
    result: tuple[dict[str, Any], Path] | None = None
    native_evidence: list[Any] = []
    try:
        target = base._open_model(
            role="target",
            bundle=target_path,
            identity=LogicalModelIdentity(OFFICIAL_REPO_ID, OFFICIAL_REVISION),
            source_budget_mb=args.target_source_budget_mb,
            device=args.device,
            compute_dtype=args.compute_dtype,
            max_resident_bytes=int(args.max_resident_mb * MIB),
            max_context_tokens=args.max_context_tokens,
            attention_mode=args.attention_mode,
            native_evidence=native_evidence,
            require_production_profile=require_production_profile,
            require_official_target=require_official_target,
        )
        draft = base._open_model(
            role="draft",
            bundle=draft_path,
            identity=LogicalModelIdentity(
                QWEN35_DRAFTER_REPO_ID, QWEN35_DRAFTER_REVISION
            ),
            source_budget_mb=args.draft_source_budget_mb,
            device=args.device,
            compute_dtype=args.compute_dtype,
            max_resident_bytes=int(args.max_resident_mb * MIB),
            max_context_tokens=args.max_context_tokens,
            attention_mode="off",
            native_evidence=[],
            require_production_profile=require_production_profile,
            require_official_target=False,
        )
        if target.model.config.vocab_size != draft.model.config.vocab_size:
            raise LiveK4SmokeError("target and drafter vocabularies differ")
        vocabulary = target.model.config.vocab_size
        checked = (*prompt, *eos_token_ids)
        if args.expected_token_ids is not None:
            checked += tuple(args.expected_token_ids)
        if any(token < 0 or token >= vocabulary for token in checked):
            raise LiveK4SmokeError("token lies outside checkpoint vocabulary")
        provider = Qwen35K4DraftProvider(
            draft.model,
            eos_token_ids=eos_token_ids,
            head_block_rows=args.head_block_rows,
        )
        timed = _TimedDraftProvider(provider)
        recorder = _TargetTransactionRecorder(target.model)
        target_before = base._pager_counters(target.pager)
        draft_before = base._pager_counters(draft.pager)
        wall_started = time.perf_counter()
        generated = Qwen38K4SpeculativeDecoder(target.model, timed).generate(
            [prompt],
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            head_block_rows=args.head_block_rows,
        )
        wall_seconds = time.perf_counter() - wall_started
        target_after = base._pager_counters(target.pager)
        draft_after = base._pager_counters(draft.pager)
        if base._file_sha256(tokenizer_path, "local tokenizer")[0] != tokenizer_sha:
            raise LiveK4SmokeError("local tokenizer changed during execution")
        rounds = [row.to_dict() for row in generated.evidence.rounds]
        native_rows = _native_rows(native_evidence)
        transaction_receipt = recorder.receipt()
        positions = _committed_position_rows(transaction_receipt, native_rows)
        target_state = _committed_state(target.model)
        draft_state = _committed_state(draft.model)
        full_history = (*prompt, *generated.token_ids)
        draft_history = provider.committed_history
        if draft_history is None or full_history[: len(draft_history)] != draft_history:
            raise LiveK4SmokeError("drafter history is not target committed")
        suffix = full_history[len(draft_history) :]
        if len(suffix) > 3:
            raise LiveK4SmokeError("drafter terminal suffix exceeds K4 tail")
        tokenwise = None
        if args.parity_control == "tokenwise":
            tokenwise = _tokenwise_parity(
                args,
                target_path=target_path,
                prompt=prompt,
                generated=generated.token_ids,
                live_positions=positions,
                live_state=target_state,
                live_native=native_rows,
                require_production_profile=require_production_profile,
                require_official_target=require_official_target,
            )
        independent = base._parity(
            args.expected_token_ids,
            generated.token_ids,
            required_length=args.max_new_tokens,
        )
        parity = {**independent, "tokenwise_control": tokenwise}
        provider_metrics = provider.metrics().to_dict()
        report = _seal(
            {
                "acceptance": _acceptance(rounds),
                "contract": {
                    "attention": (
                        {
                            "alpha": 0.01,
                            "balance_alpha": 1.0,
                            "diagonal_debit": 3.0,
                            "free_softmax_heads": 20,
                            "layer": 27,
                            "selected_query_heads": [2, 8, 14, 20],
                            "selected_sinkhorn_heads": 4,
                        }
                        if args.attention_mode == "native-crsa"
                        else {
                            "free_softmax_heads": target.model.config.n_heads,
                            "selected_sinkhorn_heads": 0,
                        }
                    ),
                    "attention_mode": args.attention_mode,
                    "batch_size": 1,
                    "bundle_authentication": (
                        "pinned-causal-manifest+range-identity/v1"
                    ),
                    "checkpoint_mutation": False,
                    "eos_token_ids": list(eos_token_ids),
                    "k": 4,
                    "local_only": True,
                    "max_new_tokens": args.max_new_tokens,
                    "official_target_config": require_official_target,
                    "parity_control": args.parity_control,
                    "pinned_production_profiles": require_production_profile,
                    "remote_io": False,
                    "receipt_security": "self-sealed-integrity/v1",
                    "target_commits_only": True,
                    "target_head_scans_per_round": 1,
                    "target_k4_stage_per_full_round": 1,
                    "terminal_tail": "exact-single-target-commits/v1",
                },
                "drafter": {
                    "alignment": {
                        "committed_cursor": len(draft_history),
                        "target_cursor": len(full_history),
                        "target_only_suffix_ids": list(suffix),
                    },
                    "bundle": draft.bundle_receipt,
                    "config": {
                        "layers": draft.model.config.n_layers,
                        "tied_embeddings": draft.model.config.tie_word_embeddings,
                        "vocab_size": draft.model.config.vocab_size,
                    },
                    "pager": {
                        "counters": base._counter_delta(draft_before, draft_after),
                        "device": draft.pager.resolved_device,
                        "dtype": draft.pager.resolved_dtype,
                    },
                    "preflight": draft.preflight_receipt,
                    "provider": provider_metrics,
                    "state": draft_state,
                    "timing": {
                        "bundle_verify_seconds": draft.verify_seconds,
                        "preflight_seconds": draft.preflight_seconds,
                        "proposal_seconds": timed.proposal_seconds,
                        "reconcile_seconds": timed.reconcile_seconds,
                        "total_provider_seconds": (
                            timed.proposal_seconds + timed.reconcile_seconds
                        ),
                    },
                    "trace": {
                        "proposals": [list(row) for row in timed.proposals],
                        "reconciled_history_lengths": timed.reconciled_lengths,
                    },
                },
                "parity": parity,
                "prompt": {
                    "rendered_sha256": hashlib.sha256(
                        rendered.encode("utf-8")
                    ).hexdigest(),
                    "system": args.system_prompt.strip(),
                    "token_count": len(prompt),
                    "token_ids": list(prompt),
                    "tokenizer": {
                        "kind": "local-tokenizers-json/v1",
                        "sha256": tokenizer_sha,
                        "size_bytes": tokenizer_size,
                    },
                    "user": args.prompt.strip(),
                },
                "schema": RESULT_SCHEMA,
                "status": "mismatch" if independent["exact"] is False else "positive",
                "target": {
                    "bundle": target.bundle_receipt,
                    "config": {
                        "heads": target.model.config.n_heads,
                        "layers": target.model.config.n_layers,
                        "vocab_size": target.model.config.vocab_size,
                    },
                    "native_crsa": {
                        "events": native_rows,
                        "sha256": _sha256(native_rows),
                    },
                    "pager": {
                        "counters": base._counter_delta(target_before, target_after),
                        "device": target.pager.resolved_device,
                        "dtype": target.pager.resolved_dtype,
                    },
                    "positions": positions,
                    "preflight": target.preflight_receipt,
                    "speculative_evidence": generated.evidence.to_dict(),
                    "state": target_state,
                    "timing": {
                        "bundle_verify_seconds": target.verify_seconds,
                        "generation_seconds": generated.evidence.seconds,
                        "preflight_seconds": target.preflight_seconds,
                        "provider_guard_seconds": (
                            generated.evidence.provider_guard_seconds
                        ),
                        "wall_seconds": wall_seconds,
                    },
                    "transactions": transaction_receipt,
                },
                "tokens": {
                    "evidence_chain_sha256": _sha256(
                        [row["evidence_sha256"] for row in rounds]
                    ),
                    "generated_text": tokenizer.decode(generated.token_ids),
                    "generated_token_ids": list(generated.token_ids),
                    "pieces": list(tokenizer.token_pieces(generated.token_ids)),
                    "rounds": rounds,
                    "token_chain_sha256": _sha256(
                        {"generated": list(generated.token_ids), "prompt": list(prompt)}
                    ),
                },
            }
        )
        report = _validate_result(
            report,
            require_production_profile=require_production_profile,
            require_official_target=require_official_target,
        )
        output = base._atomic_json(output_path, report)
        result = report, output
    except BaseException as exc:
        primary = exc
    finally:
        cleanup: list[Exception] = []
        if recorder is not None:
            try:
                recorder.close()
            except Exception as exc:
                cleanup.append(exc)
        if provider is not None:
            try:
                provider.close()
            except Exception as exc:
                cleanup.append(exc)
        for owned in (target, draft):
            if owned is not None:
                try:
                    owned.close()
                except Exception as exc:
                    cleanup.append(exc)
        if cleanup:
            failure = cleanup[0]
            if primary is None:
                primary = LiveK4SmokeError(
                    f"runtime cleanup failed: {type(failure).__name__}: {failure}"
                )
            else:
                primary.add_note(
                    f"runtime cleanup also failed: {type(failure).__name__}: {failure}"
                )
    if primary is not None:
        raise primary
    if result is None:
        raise AssertionError("live K4 execution returned no result")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report, output = run(args)
    except (LiveK4SmokeError, OSError, TypeError, ValueError, KeyError) as exc:
        sys.stderr.write(f"qwen35_live_k4_smoke: error: {exc}\n")
        return 2
    print(
        json.dumps(
            {
                "acceptance_rate": report["acceptance"]["rate"],
                "attention_mode": report["contract"]["attention_mode"],
                "generated_token_ids": report["tokens"]["generated_token_ids"],
                "output": str(output),
                "parity_positive": (
                    report["parity"]["tokenwise_control"] is None
                    or report["parity"]["tokenwise_control"]["comparison"]["positive"]
                ),
                "sha256": report["sha256"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 1 if report["status"] == "mismatch" else 0


if __name__ == "__main__":
    raise SystemExit(main())
