#!/usr/bin/env python3
"""Seal an official exact-greedy versus fixed-draft K=2 Qwen benchmark.

Selection is a separate gold-free projection from the historical Qwen draft
input.  Runtime decisions consume only that projection, one authenticated
causal bundle, and a fixed two-token provider identity.
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
from immer.runtimes.qwen3_8 import (
    K2SpeculativeGenerationEvidence,
    K2SpeculativeRoundEvidence,
    QWEN38_K2_SPECULATIVE_SCHEMA,
    Qwen38K2SpeculativeDecoder,
    StreamedQwen38,
)


ROOT = Path(__file__).resolve().parent.parent
DIRECT_SCRIPT = ROOT / "scripts" / "qwen38_direct_decode_benchmark.py"
PARITY_SCRIPT = ROOT / "scripts" / "qwen38_continuation_block_parity.py"
INPUT_SCHEMA = "immer.qwen3.8-speculative-k2-input/v1"
EXECUTION_SCHEMA = "immer.qwen3.8-speculative-k2-execution/v1"
HIDDEN_SCHEMA = "immer.qwen3.8-speculative-hidden-manifest/v1"
RESULT_SCHEMA = "immer.qwen3.8-speculative-k2-benchmark/v1"
PROVIDER_KIND = "fixed-q3-draft-pair/v1"


class SpeculativeBenchmarkError(RuntimeError):
    """The sealed speculative benchmark contract cannot be satisfied."""


def _load_protocol(name: str, path: Path) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import protocol {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


direct = _load_protocol("qwen38_direct_decode_benchmark", DIRECT_SCRIPT)
parity = _load_protocol("qwen38_continuation_block_parity", PARITY_SCRIPT)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    return direct._result(dict(value))


def _verify_seal(value: Mapping[str, Any], label: str) -> None:
    try:
        direct._verify_document_seal(value, label)
    except Exception as exc:
        raise SpeculativeBenchmarkError(str(exc)) from exc


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SpeculativeBenchmarkError(f"{label} must be a SHA-256 digest")
    return value


def _count(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SpeculativeBenchmarkError(f"{label} must be a non-negative integer")
    return value


def _seconds(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpeculativeBenchmarkError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise SpeculativeBenchmarkError(f"{label} must be finite and non-negative")
    return result


_FORBIDDEN = frozenset(
    {"answer", "candidate_correct", "correct", "gold", "question", "text"}
)


def _contains_forbidden(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).strip().lower() in _FORBIDDEN or _contains_forbidden(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden(child) for child in value)
    return False


def _provider(source: Mapping[str, Any]) -> dict[str, Any]:
    identity = source.get("drafts")
    if not isinstance(identity, str) or not identity:
        raise SpeculativeBenchmarkError("draft source provider identity is invalid")
    return {
        "identity": identity,
        "kind": PROVIDER_KIND,
        "selection": "first-two-token-ids/v1",
    }


def _source_k2_projection(document: Mapping[str, Any]) -> dict[str, Any]:
    rows = direct._source_rows(document)
    source = document["source"]
    return {
        "items": [
            {
                "draft_token_ids": list(
                    direct._token_rows(row.get("draft_token_ids"), "source draft")[:2]
                ),
                "item_id": row["item_id"],
                "prompt_token_ids": list(
                    direct._token_rows(row.get("prompt_token_ids"), "source prompt")
                ),
            }
            for row in rows
        ],
        "provider": _provider(source),
        "schema": direct.FERTIG_INPUT_SCHEMA,
        "source": {
            "checkpoint": source["checkpoint"],
            "revision": source["revision"],
        },
    }


def _validate_input(document: dict[str, Any]) -> dict[str, Any]:
    required = {
        "draft_token_ids",
        "item_id",
        "prompt_token_ids",
        "provider",
        "schema",
        "selected_input_sha256",
        "selection_index",
        "sha256",
        "source",
        "status",
    }
    if (
        set(document) != required
        or document.get("schema") != INPUT_SCHEMA
        or document.get("status") != "sealed"
        or _contains_forbidden(document)
    ):
        raise SpeculativeBenchmarkError("speculative input schema is invalid")
    _verify_seal(document, "speculative input")
    item_id = document.get("item_id")
    index = document.get("selection_index")
    if not isinstance(item_id, str) or not item_id:
        raise SpeculativeBenchmarkError("speculative item ID is invalid")
    _count(index, "selection index")
    prompt = direct._token_rows(document.get("prompt_token_ids"), "speculative prompt")
    draft = direct._token_rows(document.get("draft_token_ids"), "speculative draft")
    if len(draft) != 2:
        raise SpeculativeBenchmarkError("speculative draft must contain exactly K=2")
    provider = document.get("provider")
    if not isinstance(provider, Mapping) or set(provider) != {
        "identity",
        "kind",
        "selection",
    }:
        raise SpeculativeBenchmarkError("speculative provider identity is invalid")
    if (
        provider.get("kind") != PROVIDER_KIND
        or provider.get("selection") != "first-two-token-ids/v1"
        or not isinstance(provider.get("identity"), str)
        or not provider["identity"]
    ):
        raise SpeculativeBenchmarkError("speculative provider contract is invalid")
    source = document.get("source")
    if not isinstance(source, Mapping) or set(source) != {
        "checkpoint",
        "contract_sha256",
        "raw_file_sha256",
        "revision",
        "schema",
    }:
        raise SpeculativeBenchmarkError("speculative source identity is invalid")
    if (
        source.get("schema") != direct.FERTIG_INPUT_SCHEMA
        or not isinstance(source.get("checkpoint"), str)
        or not source["checkpoint"]
        or not isinstance(source.get("revision"), str)
        or not source["revision"]
    ):
        raise SpeculativeBenchmarkError("speculative source contract is invalid")
    for name in ("raw_file_sha256", "contract_sha256"):
        _digest(source.get(name), f"source {name}")
    selected = {
        "draft_token_ids": list(draft),
        "item_id": item_id,
        "prompt_token_ids": list(prompt),
        "provider": dict(provider),
        "selection_index": index,
        "source_contract_sha256": source["contract_sha256"],
    }
    if document.get("selected_input_sha256") != direct._sha256(selected):
        raise SpeculativeBenchmarkError("selected input SHA-256 is inconsistent")
    return document


def select_input(args: argparse.Namespace) -> dict[str, Any]:
    source, raw_sha256 = direct._externally_sealed_json(
        args.inputs, args.inputs_sha256, "Qwen draft input"
    )
    rows = direct._source_rows(source)
    index = args.index
    if index >= len(rows) or rows[index]["item_id"] != args.item_id:
        raise SpeculativeBenchmarkError("item_id/index does not select one fixed row")
    projection = _source_k2_projection(source)
    contract_sha256 = direct._sha256(projection)
    row = projection["items"][index]
    if len(row["draft_token_ids"]) != 2:
        raise SpeculativeBenchmarkError("selected source row has fewer than two drafts")
    selected = {
        "draft_token_ids": row["draft_token_ids"],
        "item_id": row["item_id"],
        "prompt_token_ids": row["prompt_token_ids"],
        "provider": projection["provider"],
        "selection_index": index,
        "source_contract_sha256": contract_sha256,
    }
    identity = {
        "draft_token_ids": row["draft_token_ids"],
        "item_id": row["item_id"],
        "prompt_token_ids": row["prompt_token_ids"],
        "provider": projection["provider"],
        "schema": INPUT_SCHEMA,
        "selected_input_sha256": direct._sha256(selected),
        "selection_index": index,
        "source": {
            "checkpoint": projection["source"]["checkpoint"],
            "contract_sha256": contract_sha256,
            "raw_file_sha256": raw_sha256,
            "revision": projection["source"]["revision"],
            "schema": source["schema"],
        },
        "status": "sealed",
    }
    if getattr(args, "_require_official", True) and (
        identity["source"]["checkpoint"],
        identity["source"]["revision"],
    ) != (direct.OFFICIAL_REPO_ID, direct.OFFICIAL_REVISION):
        raise SpeculativeBenchmarkError("speculative source is not official Qwen")
    return _validate_input(_seal(identity))


def _hidden_manifest(
    records: Mapping[str, list[tuple[int, torch.Tensor]]],
) -> dict[str, Any]:
    body: dict[str, Any] = {"schema": HIDDEN_SCHEMA}
    for phase in ("prefill", "committed", "staged"):
        rows = []
        for ordinal, (position, tensor) in enumerate(records.get(phase, [])):
            record = parity._tensor_record(
                f"{phase}.{ordinal:06d}.pos.{position:06d}", tensor
            )
            rows.append({"position": position, **record})
        body[phase] = rows
    return _seal(body)


def _validate_hidden_manifest(document: object) -> Mapping[str, Any]:
    if (
        not isinstance(document, Mapping)
        or set(document) != {"committed", "prefill", "schema", "sha256", "staged"}
        or document.get("schema") != HIDDEN_SCHEMA
    ):
        raise SpeculativeBenchmarkError("hidden manifest schema is invalid")
    _verify_seal(document, "hidden manifest")
    for phase in ("prefill", "committed", "staged"):
        rows = document.get(phase)
        if not isinstance(rows, list):
            raise SpeculativeBenchmarkError("hidden manifest rows are invalid")
        positions: set[int] = set()
        for ordinal, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != {
                "dtype",
                "name",
                "nbytes",
                "position",
                "sha256",
                "shape",
            }:
                raise SpeculativeBenchmarkError("hidden tensor record is invalid")
            position = _count(row.get("position"), "hidden position")
            if phase != "staged" and position in positions:
                raise SpeculativeBenchmarkError("hidden positions are duplicated")
            positions.add(position)
            _digest(row.get("sha256"), "hidden tensor")
            if row.get("name") != f"{phase}.{ordinal:06d}.pos.{position:06d}":
                raise SpeculativeBenchmarkError("hidden tensor name is inconsistent")
            dtype = row.get("dtype")
            if dtype not in parity._DTYPE_BYTES:
                raise SpeculativeBenchmarkError("hidden tensor dtype is invalid")
            shape = row.get("shape")
            if (
                not isinstance(shape, list)
                or not shape
                or any(
                    isinstance(value, bool) or not isinstance(value, int) or value < 1
                    for value in shape
                )
            ):
                raise SpeculativeBenchmarkError("hidden tensor shape is invalid")
            nbytes = _count(row.get("nbytes"), "hidden tensor bytes")
            if nbytes != math.prod(shape) * parity._DTYPE_BYTES[dtype]:
                raise SpeculativeBenchmarkError("hidden tensor bytes disagree")
    return document


class _HiddenRecorder:
    def __init__(self, model: StreamedQwen38) -> None:
        self.model = model
        self.records: dict[str, list[tuple[int, torch.Tensor]]] = {
            "prefill": [],
            "committed": [],
            "staged": [],
        }
        self._wrap()

    def _save(self, phase: str, start: int, hidden: torch.Tensor) -> None:
        for offset in range(hidden.shape[1]):
            self.records[phase].append(
                (start + offset, hidden[:, offset : offset + 1].detach().clone())
            )

    def _wrap(self) -> None:
        prefill = self.model.prefill
        decode = self.model.decode
        stage = self.model.stage_continuation_block
        commit = self.model.commit_continuation_block

        def recorded_prefill(*args, **kwargs):
            hidden, evidence = prefill(*args, **kwargs)
            self._save("prefill", 0, hidden)
            return hidden, evidence

        def recorded_decode(*args, **kwargs):
            start = self.model.next_position
            hidden, evidence = decode(*args, **kwargs)
            self._save("committed", start, hidden)
            return hidden, evidence

        def recorded_stage(*args, **kwargs):
            result = stage(*args, **kwargs)
            self._save("staged", result.evidence.start_pos, result.hidden)
            return result

        def recorded_commit(*args, **kwargs):
            start = self.model.next_position
            hidden, evidence = commit(*args, **kwargs)
            self._save("committed", start, hidden)
            return hidden, evidence

        self.model.prefill = recorded_prefill
        self.model.decode = recorded_decode
        self.model.stage_continuation_block = recorded_stage
        self.model.commit_continuation_block = recorded_commit


def _run_execution(
    args: argparse.Namespace,
    *,
    mode: str,
    selected: Mapping[str, Any],
    trace_path: Path,
    runtime_factory: Callable[..., tuple[Any, StreamedQwen38]],
    expected_checkpoint: Mapping[str, Any] | None,
    expected_bundle: Mapping[str, Any] | None,
    expected_bundle_root: Path,
) -> dict[str, Any]:
    recorder = AccessTraceRecorder()
    runtime, model = runtime_factory(args, recorder, max_batch_size=1)
    try:
        if any(
            value is not None
            for value in (
                model.graft,
                model.graft_layer,
                model.native_head_crsa,
                model.delta_probe,
            )
        ):
            raise SpeculativeBenchmarkError(
                "speculative benchmark requires an off runtime"
            )
        mount = getattr(runtime, "mount", None)
        if (
            getattr(mount, "root", None) is None
            or Path(mount.root).resolve() != expected_bundle_root
        ):
            raise SpeculativeBenchmarkError(
                "runtime opened a different causal bundle root"
            )
        checkpoint = direct._checkpoint(model)
        bundle = dict(runtime.verification or {})
        selected_identity = (
            selected["source"]["checkpoint"],
            selected["source"]["revision"],
        )
        if (checkpoint.get("repo_id"), checkpoint.get("revision")) != selected_identity:
            raise SpeculativeBenchmarkError(
                "runtime checkpoint differs from sealed input"
            )
        if expected_checkpoint is not None and checkpoint != dict(expected_checkpoint):
            raise SpeculativeBenchmarkError("runtime checkpoint identity differs")
        if expected_bundle is not None and bundle != dict(expected_bundle):
            raise SpeculativeBenchmarkError("runtime bundle identity differs")
        if (
            model.next_position != 0
            or model.state_bytes != 0
            or any(state is not None for state in model._layer_states)
        ):
            raise SpeculativeBenchmarkError("runtime did not start with empty state")
        prompt = tuple(selected["prompt_token_ids"])
        draft = tuple(selected["draft_token_ids"])
        if len(prompt) + 2 > model.max_seq_len:
            raise SpeculativeBenchmarkError("speculative request exceeds max_seq_len")
        if max((*prompt, *draft)) >= model.config.vocab_size:
            raise SpeculativeBenchmarkError("speculative token exceeds vocabulary")
        hidden = _HiddenRecorder(model)
        source_start = direct._source_metric(
            model.pager.source, "network_or_source_body_bytes"
        )
        linears_start = direct._source_metric(model.pager, "linear_calls")
        started = time.perf_counter()
        eos = (direct.IM_END_TOKEN_ID, direct.END_OF_TEXT_TOKEN_ID)
        reference_stage = {
            "linear_calls": 0,
            "seconds": 0.0,
            "source_body_bytes": 0,
        }
        if mode == "greedy":
            recorded_prefill = model.prefill

            def audited_prefill(*prefill_args, **prefill_kwargs):
                result = recorded_prefill(*prefill_args, **prefill_kwargs)
                audit_source = direct._source_metric(
                    model.pager.source, "network_or_source_body_bytes"
                )
                audit_linears = direct._source_metric(model.pager, "linear_calls")
                audit_started = time.perf_counter()
                stage = model.stage_continuation_block([draft])
                model.discard_continuation_block(stage)
                reference_stage.update(
                    {
                        "linear_calls": direct._source_metric(
                            model.pager, "linear_calls"
                        )
                        - audit_linears,
                        "seconds": time.perf_counter() - audit_started,
                        "source_body_bytes": direct._source_metric(
                            model.pager.source, "network_or_source_body_bytes"
                        )
                        - audit_source,
                    }
                )
                return result

            model.prefill = audited_prefill
            tokens, evidence = model.generate_greedy(
                [prompt],
                max_new_tokens=2,
                eos_token_ids=eos,
                head_block_rows=args.head_block_rows,
            )
            provider_guard_seconds = 0.0
            provider_guard_bytes = 0
            head_scans = len(tokens)
            evidence_document = None
        elif mode == "speculative":

            def provider(_history: tuple[int, ...]) -> tuple[int, ...]:
                return draft

            result = Qwen38K2SpeculativeDecoder(model, provider).generate(
                [prompt],
                max_new_tokens=2,
                eos_token_ids=eos,
                head_block_rows=args.head_block_rows,
            )
            tokens = result.token_ids
            evidence = result.evidence
            provider_guard_seconds = evidence.provider_guard_seconds
            provider_guard_bytes = evidence.provider_guard_bytes
            head_scans = evidence.head_scans
            evidence_document = evidence.to_dict()
        else:  # pragma: no cover
            raise AssertionError(mode)
        raw_wall_seconds = time.perf_counter() - started
        raw_source_bytes = (
            direct._source_metric(model.pager.source, "network_or_source_body_bytes")
            - source_start
        )
        raw_linear_calls = (
            direct._source_metric(model.pager, "linear_calls") - linears_start
        )
        if (
            raw_source_bytes != evidence.source_body_bytes
            or raw_linear_calls != evidence.linear_calls
        ):
            raise SpeculativeBenchmarkError(
                "execution counters differ from runtime evidence"
            )
        source_bytes = raw_source_bytes - int(reference_stage["source_body_bytes"])
        linear_calls = raw_linear_calls - int(reference_stage["linear_calls"])
        model_seconds = max(
            0.0, float(evidence.seconds) - float(reference_stage["seconds"])
        )
        wall_seconds = max(0.0, raw_wall_seconds - float(reference_stage["seconds"]))
        manifest_started = time.perf_counter()
        hidden_manifest = _hidden_manifest(hidden.records)
        try:
            state_manifest = parity._state_manifest(model)
        except Exception as exc:
            raise SpeculativeBenchmarkError(str(exc)) from exc
        manifest_seconds = time.perf_counter() - manifest_started
        trace = direct._trace_receipt(recorder, trace_path)
        execution = _seal(
            {
                "bundle": bundle,
                "checkpoint": checkpoint,
                "final_cursor": int(model.next_position),
                "final_state_bytes": int(model.state_bytes),
                "hidden": hidden_manifest,
                "input_sha256": selected["sha256"],
                "mode": mode,
                "provider": None if mode == "greedy" else dict(selected["provider"]),
                "schema": EXECUTION_SCHEMA,
                "speculative_evidence": evidence_document,
                "state": state_manifest,
                "status": "sealed",
                "token_chain_sha256": direct._token_chain_sha256(prompt, tokens),
                "token_ids": list(tokens),
                "traffic": {
                    "forward_passes": int(evidence.forward_passes),
                    "head_scans": head_scans,
                    "linear_calls": linear_calls,
                    "manifest_seconds": manifest_seconds,
                    "model_seconds": model_seconds,
                    "model_seconds_excluding_provider_guard": max(
                        0.0, model_seconds - provider_guard_seconds
                    ),
                    "provider_guard_bytes": provider_guard_bytes,
                    "provider_guard_seconds": provider_guard_seconds,
                    "raw": {
                        "linear_calls": raw_linear_calls,
                        "model_seconds": float(evidence.seconds),
                        "source_body_bytes": raw_source_bytes,
                        "wall_seconds": raw_wall_seconds,
                    },
                    "reference_stage": reference_stage,
                    "source_body_bytes": source_bytes,
                    "trace": trace,
                    "trace_sha256": trace["sha256"],
                    "wall_seconds": wall_seconds,
                },
            }
        )
        return _validate_execution(execution, access_trace_path=trace_path)
    finally:
        direct._cleanup(runtime, model)


def _validate_execution(
    document: dict[str, Any], *, access_trace_path: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    required = {
        "bundle",
        "checkpoint",
        "final_cursor",
        "final_state_bytes",
        "hidden",
        "input_sha256",
        "mode",
        "provider",
        "schema",
        "sha256",
        "speculative_evidence",
        "state",
        "status",
        "token_chain_sha256",
        "token_ids",
        "traffic",
    }
    if (
        set(document) != required
        or document.get("schema") != EXECUTION_SCHEMA
        or document.get("status") != "sealed"
        or document.get("mode") not in {"greedy", "speculative"}
    ):
        raise SpeculativeBenchmarkError("speculative execution schema is invalid")
    _verify_seal(document, "speculative execution")
    _digest(document.get("input_sha256"), "execution input")
    checkpoint = document.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "inventory_fingerprint",
        "repo_id",
        "revision",
    }:
        raise SpeculativeBenchmarkError("execution checkpoint identity is invalid")
    if any(not isinstance(value, str) or not value for value in checkpoint.values()):
        raise SpeculativeBenchmarkError("execution checkpoint identity is invalid")
    _digest(checkpoint.get("inventory_fingerprint"), "checkpoint inventory")
    try:
        direct._validate_complete_causal_bundle_receipt(
            document.get("bundle"), checkpoint=checkpoint
        )
    except Exception as exc:
        raise SpeculativeBenchmarkError(
            "execution complete causal-bundle receipt is invalid"
        ) from exc
    tokens = direct._token_rows(document.get("token_ids"), "execution tokens")
    if len(tokens) not in (1, 2):
        raise SpeculativeBenchmarkError("execution emitted outside the K2 bound")
    _digest(document.get("token_chain_sha256"), "execution token chain")
    try:
        state = parity._validate_state_manifest(document.get("state"))
    except Exception as exc:
        raise SpeculativeBenchmarkError(str(exc)) from exc
    hidden = _validate_hidden_manifest(document.get("hidden"))
    cursor = _count(document.get("final_cursor"), "final cursor")
    final_bytes = _count(document.get("final_state_bytes"), "final state bytes")
    if cursor != state["cursor"] or final_bytes != state["total_bytes"]:
        raise SpeculativeBenchmarkError("execution final state receipt is inconsistent")
    if len(hidden["committed"]) != len(tokens):
        raise SpeculativeBenchmarkError("committed hidden count differs from tokens")
    mode = document["mode"]
    provider = document.get("provider")
    evidence = document.get("speculative_evidence")
    if mode == "greedy":
        if provider is not None or evidence is not None:
            raise SpeculativeBenchmarkError(
                "greedy execution contains provider evidence"
            )
    elif not isinstance(provider, Mapping) or not isinstance(evidence, Mapping):
        raise SpeculativeBenchmarkError("speculative execution lacks provider evidence")
    traffic = document.get("traffic")
    fields = {
        "forward_passes",
        "head_scans",
        "linear_calls",
        "manifest_seconds",
        "model_seconds",
        "model_seconds_excluding_provider_guard",
        "provider_guard_bytes",
        "provider_guard_seconds",
        "raw",
        "reference_stage",
        "source_body_bytes",
        "trace",
        "trace_sha256",
        "wall_seconds",
    }
    if not isinstance(traffic, Mapping) or set(traffic) != fields:
        raise SpeculativeBenchmarkError("speculative traffic schema is invalid")
    raw = traffic.get("raw")
    if not isinstance(raw, Mapping) or set(raw) != {
        "linear_calls",
        "model_seconds",
        "source_body_bytes",
        "wall_seconds",
    }:
        raise SpeculativeBenchmarkError("raw execution traffic is invalid")
    _count(raw.get("linear_calls"), "raw linear calls")
    _count(raw.get("source_body_bytes"), "raw source bytes")
    _seconds(raw.get("model_seconds"), "raw model seconds")
    _seconds(raw.get("wall_seconds"), "raw wall seconds")
    reference_stage = traffic.get("reference_stage")
    if not isinstance(reference_stage, Mapping) or set(reference_stage) != {
        "linear_calls",
        "seconds",
        "source_body_bytes",
    }:
        raise SpeculativeBenchmarkError("reference-stage traffic is invalid")
    _count(reference_stage.get("linear_calls"), "reference-stage linear calls")
    _count(reference_stage.get("source_body_bytes"), "reference-stage source bytes")
    _seconds(reference_stage.get("seconds"), "reference-stage seconds")
    if (
        raw["linear_calls"] - reference_stage["linear_calls"]
        != traffic.get("linear_calls")
        or raw["source_body_bytes"] - reference_stage["source_body_bytes"]
        != traffic.get("source_body_bytes")
        or not math.isclose(
            raw["model_seconds"] - reference_stage["seconds"],
            traffic.get("model_seconds"),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            raw["wall_seconds"] - reference_stage["seconds"],
            traffic.get("wall_seconds"),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise SpeculativeBenchmarkError("adjusted execution traffic is inconsistent")
    for name in (
        "forward_passes",
        "head_scans",
        "linear_calls",
        "provider_guard_bytes",
        "source_body_bytes",
    ):
        _count(traffic.get(name), f"traffic {name}")
    for name in (
        "manifest_seconds",
        "model_seconds",
        "model_seconds_excluding_provider_guard",
        "provider_guard_seconds",
        "wall_seconds",
    ):
        _seconds(traffic.get(name), f"traffic {name}")
    if traffic.get("trace_sha256") != traffic.get("trace", {}).get("sha256"):
        raise SpeculativeBenchmarkError("execution trace digest is inconsistent")
    if not math.isclose(
        traffic["model_seconds_excluding_provider_guard"]
        + traffic["provider_guard_seconds"],
        traffic["model_seconds"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SpeculativeBenchmarkError(
            "execution model-time accounting is inconsistent"
        )
    if mode == "greedy" and (
        traffic.get("provider_guard_bytes") != 0
        or traffic.get("provider_guard_seconds") != 0.0
        or traffic.get("head_scans") != len(tokens)
        or reference_stage.get("linear_calls") == 0
    ):
        raise SpeculativeBenchmarkError("greedy traffic accounting is inconsistent")
    if mode == "speculative" and any(reference_stage.values()):
        raise SpeculativeBenchmarkError(
            "speculative execution contains a reference-stage charge"
        )
    try:
        direct._verify_access_trace_receipt(
            traffic.get("trace"), artifact_path=access_trace_path
        )
    except Exception as exc:
        raise SpeculativeBenchmarkError(str(exc)) from exc
    return document


def _hidden_diff(
    left: Mapping[str, Any], right: Mapping[str, Any], phase: str
) -> list[str]:
    left_rows = {row["position"]: row["sha256"] for row in left[phase]}
    right_rows = {row["position"]: row["sha256"] for row in right[phase]}
    return [
        str(position)
        for position in sorted(set(left_rows) | set(right_rows))
        if left_rows.get(position) != right_rows.get(position)
    ]


def _hidden_sequence_diff(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> list[str]:
    comparable = ("dtype", "nbytes", "position", "sha256", "shape")
    return [
        str(index)
        for index in range(max(len(left), len(right)))
        if index >= len(left)
        or index >= len(right)
        or {name: left[index][name] for name in comparable}
        != {name: right[index][name] for name in comparable}
    ]


def _speculative_evidence(value: object) -> K2SpeculativeGenerationEvidence:
    if not isinstance(value, Mapping):
        raise SpeculativeBenchmarkError("speculative generation evidence is invalid")
    try:
        body = dict(value)
        raw_rounds = body.get("rounds")
        if not isinstance(raw_rounds, list):
            raise TypeError("rounds")
        rounds = []
        for raw in raw_rounds:
            if not isinstance(raw, Mapping):
                raise TypeError("round")
            row = dict(raw)
            for field in (
                "proposed_token_ids",
                "target_token_ids",
                "emitted_token_ids",
            ):
                if not isinstance(row.get(field), list):
                    raise TypeError(field)
                row[field] = tuple(row[field])
            rounds.append(K2SpeculativeRoundEvidence(**row))
        for field in ("prompt_token_ids", "generated_token_ids"):
            if not isinstance(body.get(field), list):
                raise TypeError(field)
            body[field] = tuple(body[field])
        body["rounds"] = tuple(rounds)
        evidence = K2SpeculativeGenerationEvidence(**body)
    except (TypeError, ValueError) as exc:
        raise SpeculativeBenchmarkError(
            "speculative generation evidence is invalid"
        ) from exc
    if evidence.to_dict() != dict(value):
        raise SpeculativeBenchmarkError(
            "speculative generation evidence is non-canonical"
        )
    return evidence


def _validate_bound_execution(
    execution: Mapping[str, Any], selected: Mapping[str, Any]
) -> None:
    prompt = tuple(selected["prompt_token_ids"])
    tokens = tuple(execution["token_ids"])
    checkpoint = execution["checkpoint"]
    if (
        checkpoint.get("repo_id") != selected["source"]["checkpoint"]
        or checkpoint.get("revision") != selected["source"]["revision"]
    ):
        raise SpeculativeBenchmarkError(
            "execution checkpoint differs from sealed input"
        )
    if execution["token_chain_sha256"] != direct._token_chain_sha256(prompt, tokens):
        raise SpeculativeBenchmarkError(
            "execution token chain differs from sealed input"
        )
    if execution["final_cursor"] != len(prompt) + len(tokens):
        raise SpeculativeBenchmarkError("execution cursor differs from token chain")
    hidden = execution["hidden"]
    if [row["position"] for row in hidden["prefill"]] != list(range(len(prompt))):
        raise SpeculativeBenchmarkError("prefill hidden positions differ from prompt")
    if [row["position"] for row in hidden["committed"]] != list(
        range(len(prompt), len(prompt) + len(tokens))
    ):
        raise SpeculativeBenchmarkError("committed hidden positions differ from tokens")
    trace = execution["traffic"]["trace"]
    if trace["inventory_fingerprint"] != checkpoint.get("inventory_fingerprint"):
        raise SpeculativeBenchmarkError("execution trace checkpoint differs")
    if execution["mode"] == "greedy":
        if [row["position"] for row in hidden["staged"]] != [
            len(prompt),
            len(prompt) + 1,
        ]:
            raise SpeculativeBenchmarkError("greedy draft-reference stage is invalid")
        return
    evidence = _speculative_evidence(execution["speculative_evidence"])
    if (
        evidence.schema != QWEN38_K2_SPECULATIVE_SCHEMA
        or evidence.prompt_token_ids != prompt
        or evidence.generated_token_ids != tokens
        or evidence.rounds[0].proposed_token_ids != tuple(selected["draft_token_ids"])
        or evidence.source_body_bytes != execution["traffic"]["source_body_bytes"]
        or evidence.linear_calls != execution["traffic"]["linear_calls"]
        or evidence.forward_passes != execution["traffic"]["forward_passes"]
        or evidence.head_scans != execution["traffic"]["head_scans"]
        or evidence.provider_guard_bytes != execution["traffic"]["provider_guard_bytes"]
        or evidence.provider_guard_seconds
        != execution["traffic"]["provider_guard_seconds"]
        or evidence.seconds != execution["traffic"]["model_seconds"]
        or evidence.state_bytes != execution["final_state_bytes"]
    ):
        raise SpeculativeBenchmarkError("speculative evidence differs from execution")
    first_round = evidence.rounds[0]
    attempts = 2 if first_round.replay_kind == "mismatch1-restage" else 1
    staged = hidden["staged"]
    expected_positions = [
        position
        for _attempt in range(attempts)
        for position in (first_round.start_pos, first_round.start_pos + 1)
    ]
    if [row["position"] for row in staged] != expected_positions:
        raise SpeculativeBenchmarkError(
            "staged hidden rows differ from replay evidence"
        )
    if first_round.replay_kind in {"commit-k2", "mismatch1-restage"}:
        comparable = ("dtype", "nbytes", "position", "sha256", "shape")
        accepted = [{name: row[name] for name in comparable} for row in staged[-2:]]
        committed = [
            {name: row[name] for name in comparable} for row in hidden["committed"]
        ]
        if accepted != committed:
            raise SpeculativeBenchmarkError(
                "committed hidden rows differ from accepted K2 stage"
            )


def _comparison(
    greedy: Mapping[str, Any], speculative: Mapping[str, Any]
) -> dict[str, Any]:
    state_differences = [
        {
            "greedy": row["tokenwise"],
            "name": row["name"],
            "speculative": row["block"],
        }
        for row in parity._state_diff(greedy["state"], speculative["state"])
    ]
    prefill_differences = _hidden_diff(
        greedy["hidden"], speculative["hidden"], "prefill"
    )
    committed_differences = _hidden_diff(
        greedy["hidden"], speculative["hidden"], "committed"
    )
    initial_stage_differences = _hidden_sequence_diff(
        greedy["hidden"]["staged"], speculative["hidden"]["staged"][:2]
    )
    exact = {
        "bundle_equal": greedy["bundle"] == speculative["bundle"],
        "checkpoint_equal": greedy["checkpoint"] == speculative["checkpoint"],
        "committed_hidden_equal": not committed_differences,
        "cursor_equal": greedy["final_cursor"] == speculative["final_cursor"],
        "initial_stage_hidden_equal": not initial_stage_differences,
        "prefill_hidden_equal": not prefill_differences,
        "state_equal": not state_differences,
        "tokens_equal": greedy["token_ids"] == speculative["token_ids"],
    }
    return {
        **exact,
        "committed_hidden_differences": committed_differences,
        "costs": {
            "greedy": dict(greedy["traffic"]),
            "forward_passes_saved": (
                greedy["traffic"]["forward_passes"]
                - speculative["traffic"]["forward_passes"]
            ),
            "head_scans_saved": (
                greedy["traffic"]["head_scans"] - speculative["traffic"]["head_scans"]
            ),
            "linear_calls_saved": (
                greedy["traffic"]["linear_calls"]
                - speculative["traffic"]["linear_calls"]
            ),
            "speculative": dict(speculative["traffic"]),
            "source_body_bytes_saved": (
                greedy["traffic"]["source_body_bytes"]
                - speculative["traffic"]["source_body_bytes"]
            ),
            "wall_seconds_saved": (
                greedy["traffic"]["wall_seconds"]
                - speculative["traffic"]["wall_seconds"]
            ),
        },
        "initial_stage_hidden_differences": initial_stage_differences,
        "positive": all(exact.values()),
        "prefill_hidden_differences": prefill_differences,
        "state_differences": state_differences,
    }


def _validate_result(
    document: dict[str, Any],
    *,
    trace_paths: Mapping[str, str | os.PathLike[str]] | None = None,
) -> dict[str, Any]:
    required = {
        "comparison",
        "contract",
        "executions",
        "input",
        "schema",
        "sha256",
        "status",
    }
    if (
        set(document) != required
        or document.get("schema") != RESULT_SCHEMA
        or document.get("status") not in {"positive", "mismatch"}
        or _contains_forbidden(document)
    ):
        raise SpeculativeBenchmarkError("speculative result schema is invalid")
    _verify_seal(document, "speculative result")
    selected = document.get("input")
    if not isinstance(selected, dict):
        raise SpeculativeBenchmarkError("speculative result input is invalid")
    _validate_input(selected)
    contract = document.get("contract")
    if contract != {
        "gold_free": True,
        "greedy_max_new_tokens": 2,
        "provider": selected["provider"],
        "runtime_instances": 2,
        "same_causal_bundle": True,
    }:
        raise SpeculativeBenchmarkError("speculative result contract is invalid")
    executions = document.get("executions")
    if not isinstance(executions, Mapping) or set(executions) != {
        "greedy",
        "speculative",
    }:
        raise SpeculativeBenchmarkError("speculative result executions are invalid")
    paths = {} if trace_paths is None else dict(trace_paths)
    greedy = _validate_execution(
        executions["greedy"], access_trace_path=paths.get("greedy")
    )
    speculative = _validate_execution(
        executions["speculative"], access_trace_path=paths.get("speculative")
    )
    _validate_bound_execution(greedy, selected)
    _validate_bound_execution(speculative, selected)
    if (
        greedy["input_sha256"] != selected["sha256"]
        or speculative["input_sha256"] != selected["sha256"]
        or greedy["provider"] is not None
        or speculative["provider"] != selected["provider"]
    ):
        raise SpeculativeBenchmarkError("execution input/provider identity differs")
    expected = _comparison(greedy, speculative)
    if document.get("comparison") != expected:
        raise SpeculativeBenchmarkError("speculative comparison is inconsistent")
    status = "positive" if expected["positive"] else "mismatch"
    if document["status"] != status:
        raise SpeculativeBenchmarkError("speculative result status is inconsistent")
    return document


def _load_result(path: str | os.PathLike[str]) -> dict[str, Any]:
    result_path = Path(path).expanduser().resolve()
    document = direct._strict_json(result_path)
    trace_paths: dict[str, Path] = {}
    executions = document.get("executions")
    if isinstance(executions, Mapping):
        for mode in ("greedy", "speculative"):
            execution = executions.get(mode)
            traffic = (
                execution.get("traffic") if isinstance(execution, Mapping) else None
            )
            receipt = traffic.get("trace") if isinstance(traffic, Mapping) else None
            recorded = (
                Path(str(receipt.get("path"))) if isinstance(receipt, Mapping) else None
            )
            if recorded is not None and recorded.is_file():
                trace_paths[mode] = recorded
            elif recorded is not None:
                adjacent = result_path.parent / recorded.name
                if adjacent.is_file():
                    trace_paths[mode] = adjacent
    return _validate_result(document, trace_paths=trace_paths)


def run(
    args: argparse.Namespace,
    *,
    runtime_factory: Callable[..., tuple[Any, StreamedQwen38]] | None = None,
) -> dict[str, Any]:
    selected = _validate_input(direct._strict_json(args.input))
    bundle_root = Path(args.causal_bundle).expanduser().resolve()
    try:
        metadata = bundle_root.lstat()
    except OSError as exc:
        raise SpeculativeBenchmarkError("causal bundle root is missing") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SpeculativeBenchmarkError("causal bundle root must be a plain directory")
    greedy_trace = Path(args.access_trace_greedy).expanduser().resolve()
    speculative_trace = Path(args.access_trace_speculative).expanduser().resolve()
    if greedy_trace == speculative_trace:
        raise SpeculativeBenchmarkError("access trace paths must differ")
    source_identity = (selected["source"]["checkpoint"], selected["source"]["revision"])
    requested = (args.logical_repo_id, args.revision)
    if getattr(args, "_require_official", True) and (
        source_identity != (direct.OFFICIAL_REPO_ID, direct.OFFICIAL_REVISION)
        or requested != (direct.OFFICIAL_REPO_ID, direct.OFFICIAL_REVISION)
    ):
        raise SpeculativeBenchmarkError("speculative runtime is not official Qwen")
    factory = direct._runtime if runtime_factory is None else runtime_factory
    greedy = _run_execution(
        args,
        mode="greedy",
        selected=selected,
        trace_path=greedy_trace,
        runtime_factory=factory,
        expected_checkpoint=None,
        expected_bundle=None,
        expected_bundle_root=bundle_root,
    )
    speculative = _run_execution(
        args,
        mode="speculative",
        selected=selected,
        trace_path=speculative_trace,
        runtime_factory=factory,
        expected_checkpoint=greedy["checkpoint"],
        expected_bundle=greedy["bundle"],
        expected_bundle_root=bundle_root,
    )
    comparison = _comparison(greedy, speculative)
    result = _seal(
        {
            "comparison": comparison,
            "contract": {
                "gold_free": True,
                "greedy_max_new_tokens": 2,
                "provider": selected["provider"],
                "runtime_instances": 2,
                "same_causal_bundle": True,
            },
            "executions": {"greedy": greedy, "speculative": speculative},
            "input": selected,
            "schema": RESULT_SCHEMA,
            "status": "positive" if comparison["positive"] else "mismatch",
        }
    )
    return _validate_result(result)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select-input")
    select.add_argument("--inputs", required=True)
    select.add_argument("--inputs-sha256", required=True)
    select.add_argument("--item-id", required=True)
    select.add_argument("--index", type=_nonnegative_int, required=True)
    select.add_argument("--output", required=True)
    select.set_defaults(handler=select_input)
    execute = subparsers.add_parser("run")
    execute.add_argument("--input", required=True)
    execute.add_argument("--causal-bundle", required=True)
    execute.add_argument("--output", required=True)
    execute.add_argument("--access-trace-greedy", required=True)
    execute.add_argument("--access-trace-speculative", required=True)
    execute.add_argument("--source", default=direct.OFFICIAL_REPO_ID)
    execute.add_argument("--logical-repo-id", default=direct.OFFICIAL_REPO_ID)
    execute.add_argument("--revision", default=direct.OFFICIAL_REVISION)
    execute.add_argument("--pinned-inventory")
    execute.add_argument("--cache-dir", default=str(direct.DEFAULT_CACHE))
    execute.add_argument("--max-cache-gb", type=_positive_float, default=1.0)
    execute.add_argument("--source-budget-mb", type=_positive_int, default=262144)
    execute.add_argument("--max-resident-mb", type=_positive_int, default=384)
    execute.add_argument("--max-seq-len", type=_positive_int, default=256)
    execute.add_argument("--head-block-rows", type=_positive_int, default=8192)
    execute.add_argument("--device", choices=("cpu", "mps"), default="mps")
    execute.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    execute.set_defaults(handler=run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        document = args.handler(args)
        output = direct._write_json(args.output, document)
    except (SpeculativeBenchmarkError, direct.QwenDirectDecodeError) as exc:
        raise SystemExit(f"Qwen speculative benchmark failed: {exc}") from exc
    print(
        direct.json.dumps(
            {
                "output": str(output),
                "schema": document["schema"],
                "sha256": document["sha256"],
                "status": document.get("status", "sealed"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
