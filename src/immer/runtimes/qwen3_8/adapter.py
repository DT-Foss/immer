"""Lazy local chat facade for the authenticated Qwen3.8 causal bundle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from numbers import Integral
import os
from pathlib import Path
import stat
import threading
import time
from typing import Any

import torch

from ...knowledge.range_markov import MarkovRangePrefetcher
from ...contracts import ExecutionStatus, Request, Result
from ..deepseek_v4.causal_weights import CausalWeightMount, LogicalModelIdentity
from .bundle import verify_qwen38_causal_mount
from .config import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    QWEN35_DRAFTER_REPO_ID,
    QWEN35_DRAFTER_REVISION,
    Qwen38Config,
)
from .encoding import END_OF_TEXT_TOKEN_ID, IM_END_TOKEN_ID, Qwen38Tokenizer
from .draft_window import (
    DRAFT_WINDOW_ACTIONS,
    DraftWindowController,
    DraftWindowFeedback,
    DraftWindowNestedHorizon,
    DraftWindowSelection,
)
from .fast_mlp import (
    Qwen38FastMlpMount,
    Qwen38FastMlpPaths,
    open_qwen38_fast_mlp,
)
from .exact_head import ExactHeadIndex, ExactHeadNotApplicable
from .model import StreamedQwen38
from .local_draft import Qwen35K4DraftProvider
from .markov_draft import (
    MARKOV_DRAFT_PROVIDER_ABI,
    MARKOV_DRAFT_STATE_SCHEMA,
    FingerprintRollingK4DraftProvider,
)
from .pager import Qwen38WeightPager
from .speculative import Qwen38K4SpeculativeDecoder
from .semantic_state_cache import (
    AnchorReceipt,
    RestoredAnchor,
    SEMANTIC_ANCHOR_SEED_SCHEMA,
    SemanticStateAnchorCache,
    token_prefix_sha256,
)


_SHA256 = frozenset("0123456789abcdef")
_BUNDLE_RECEIPT_FIELDS = (
    "checkpoint_bytes",
    "graph_revision",
    "kind",
    "layout_fingerprint",
    "manifest_sha256",
    "shards",
    "shards_sha256",
    "tensor_bindings",
    "weights_layout",
)
_GENERATION_RECEIPT_FIELDS = (
    "context_mode",
    "stateful_cache",
    "general_generation",
    "prefill_mode",
    "forward_passes",
    "source_body_bytes",
    "linear_calls",
    "seconds",
    "state_bytes",
    "stopped_on_eos",
)
RESULT_CELL_GENERATION_POLICY_SCHEMA = (
    "immer.qwen3.8-result-cell-generation-policy/v1"
)
_RESULT_CELL_CODE_REVISION_LENGTHS = frozenset((40, 64))


class Qwen38ChatError(RuntimeError):
    """A local Qwen3.8 chat runtime cannot be opened or executed safely."""


class _RequestRejected(ValueError):
    pass


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    descriptor: int | None = None
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise Qwen38ChatError(
                f"local tokenizer must be a non-symlink regular file: {path}"
            )
        flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
        flags |= int(getattr(os, "O_NOFOLLOW", 0))
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise Qwen38ChatError("local tokenizer changed while it was opened")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 4 * 1024**2):
            digest.update(chunk)
        after = os.fstat(descriptor)
        linked = path.lstat()
    except Qwen38ChatError:
        raise
    except OSError as exc:
        raise Qwen38ChatError(f"cannot authenticate local tokenizer: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
        or stat.S_ISLNK(linked.st_mode)
        or not stat.S_ISREG(linked.st_mode)
        or (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    ):
        raise Qwen38ChatError("local tokenizer changed while it was authenticated")
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and not (set(value) - _SHA256)


def _token_ids(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence of integer token IDs")
    result: list[int] = []
    for raw in value:
        if isinstance(raw, bool) or not isinstance(raw, Integral):
            raise TypeError(f"{label} must contain only integer token IDs")
        token_id = int(raw)
        if token_id < 0:
            raise ValueError(f"{label} contains a negative token ID")
        result.append(token_id)
    return tuple(result)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{label} must be a string-keyed mapping or dataclass")
    return dict(value)


def _compact_bundle_receipt(value: object) -> dict[str, Any]:
    receipt = _mapping(value, "bundle receipt")
    if receipt.get("kind") != "complete-causal-bundle/v1":
        raise Qwen38ChatError("runtime lacks a verified complete Qwen causal bundle")
    if any(
        not _is_sha256(receipt.get(key))
        for key in (
            "layout_fingerprint",
            "manifest_sha256",
            "shards_sha256",
        )
    ):
        raise Qwen38ChatError("bundle receipt digest is invalid")
    for key in ("checkpoint_bytes", "shards", "tensor_bindings"):
        _positive_int(receipt.get(key), f"bundle receipt {key}")
    graph_revision = receipt.get("graph_revision")
    if (
        not isinstance(graph_revision, list)
        or len(graph_revision) != 2
        or isinstance(graph_revision[0], bool)
        or not isinstance(graph_revision[0], int)
        or graph_revision[0] < 0
        or not _is_sha256(graph_revision[1])
    ):
        raise Qwen38ChatError("bundle receipt graph revision is invalid")
    if receipt.get("weights_layout") not in {"flat/v1", "nested/v1"}:
        raise Qwen38ChatError("bundle receipt weights layout is invalid")
    return {key: receipt[key] for key in _BUNDLE_RECEIPT_FIELDS}


def _compact_generation_receipt(
    value: object,
    *,
    prompt_ids: tuple[int, ...],
    generated_ids: tuple[int, ...],
) -> dict[str, Any]:
    evidence = _mapping(value, "generation evidence")
    if (
        "prompt_token_ids" in evidence
        and _token_ids(evidence["prompt_token_ids"], "generation prompt token IDs")
        != prompt_ids
    ):
        raise Qwen38ChatError("generation evidence prompt differs from execution")
    if (
        "generated_token_ids" in evidence
        and _token_ids(evidence["generated_token_ids"], "generation output token IDs")
        != generated_ids
    ):
        raise Qwen38ChatError("generation evidence output differs from execution")
    missing = [key for key in _GENERATION_RECEIPT_FIELDS if key not in evidence]
    if missing:
        raise Qwen38ChatError(
            "generation evidence is incomplete: " + ", ".join(missing)
        )
    compact = {key: evidence[key] for key in _GENERATION_RECEIPT_FIELDS}
    for key in ("forward_passes", "source_body_bytes", "linear_calls", "state_bytes"):
        item = compact[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise Qwen38ChatError(f"generation evidence {key} is invalid")
    seconds = compact["seconds"]
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(float(seconds))
        or float(seconds) < 0
    ):
        raise Qwen38ChatError("generation evidence seconds is invalid")
    for key in ("stateful_cache", "general_generation", "stopped_on_eos"):
        if not isinstance(compact[key], bool):
            raise Qwen38ChatError(f"generation evidence {key} is invalid")
    if (
        compact["stateful_cache"] is not True
        or compact["general_generation"] is not True
    ):
        raise Qwen38ChatError("generation did not use the stateful general path")
    if compact["context_mode"] != "stateful_autoregressive":
        raise Qwen38ChatError("generation evidence context mode is invalid")
    if compact["prefill_mode"] != "batched":
        raise Qwen38ChatError("generation evidence prefill mode is invalid")
    # Canonical JSON is both a public-value check and a finite-float check.
    try:
        json.dumps(compact, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise Qwen38ChatError("generation evidence is not canonical JSON") from exc
    compact.update(
        {
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(generated_ids),
            "token_trace_sha256": _digest(
                {"generated": generated_ids, "prompt": prompt_ids}
            ),
        }
    )
    return compact


def _nested_draft_horizons(value: object) -> tuple[DraftWindowNestedHorizon, ...]:
    """Recover exact shorter-prefix reliability from an already verified wave."""

    observed_window = getattr(value, "window_size", None)
    rounds = getattr(value, "rounds", None)
    if observed_window not in DRAFT_WINDOW_ACTIONS or not isinstance(rounds, tuple):
        return ()
    horizons = []
    for candidate in DRAFT_WINDOW_ACTIONS:
        if candidate >= observed_window:
            continue
        wave_count = 0
        accepted = 0
        proposed = 0
        for row in rounds:
            row_proposal = getattr(row, "proposed_token_ids", ())
            row_accepted = getattr(row, "accepted_prefix_length", None)
            row_window = getattr(row, "window_size", observed_window)
            if not isinstance(row_proposal, tuple) or not row_proposal:
                continue
            if (
                isinstance(row_window, bool)
                or not isinstance(row_window, int)
                or not 2 <= row_window <= observed_window
            ):
                raise Qwen38ChatError(
                    "rolling evidence has an invalid executed window"
                )
            if candidate >= row_window:
                continue
            if (
                isinstance(row_accepted, bool)
                or not isinstance(row_accepted, int)
                or row_accepted < 0
            ):
                raise Qwen38ChatError(
                    "rolling evidence has invalid nested-prefix acceptance"
                )
            visible = candidate - 1
            if len(row_proposal) < visible:
                raise Qwen38ChatError(
                    "rolling evidence lacks its staged nested prefix"
                )
            if visible <= 0:
                continue
            wave_count += 1
            proposed += visible
            accepted += min(row_accepted, visible)
        if proposed:
            horizons.append(
                DraftWindowNestedHorizon(
                    candidate_window=candidate,
                    observed_window=observed_window,
                    wave_count=wave_count,
                    accepted_draft_tokens=accepted,
                    proposed_draft_tokens=proposed,
                )
            )
    return tuple(horizons)


def _runtime_source_body_bytes(runtime: object) -> int:
    model = getattr(runtime, "model", None)
    pager = getattr(model, "pager", None)
    source = getattr(pager, "source", None)
    metrics = getattr(source, "metrics", None)
    if not callable(metrics):
        return 0
    try:
        value = metrics().get("network_or_source_body_bytes", 0)
    except Exception:
        return 0
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _anchor_model_state(model: object) -> tuple[int, bool, int, int | None]:
    """Return the native state fields that make cache miss/hit mutation explicit."""

    next_position = getattr(model, "next_position", None)
    state_poisoned = getattr(model, "state_poisoned", None)
    state_bytes = getattr(model, "state_bytes", None)
    state_batch_size = getattr(model, "state_batch_size", None)
    if (
        isinstance(next_position, bool)
        or not isinstance(next_position, int)
        or next_position < 0
        or not isinstance(state_poisoned, bool)
        or isinstance(state_bytes, bool)
        or not isinstance(state_bytes, int)
        or state_bytes < 0
        or (
            state_batch_size is not None
            and (
                isinstance(state_batch_size, bool)
                or not isinstance(state_batch_size, int)
                or state_batch_size <= 0
            )
        )
    ):
        raise Qwen38ChatError("runtime model lacks the native anchor-state contract")
    return next_position, state_poisoned, state_bytes, state_batch_size


def _anchor_document(value: object) -> dict[str, Any]:
    if type(value) is not AnchorReceipt:
        raise Qwen38ChatError("anchor cache returned an unsealed receipt")
    document = value.to_document()
    try:
        reconstructed = AnchorReceipt.from_document(document)
    except Exception as exc:
        raise Qwen38ChatError("anchor receipt authentication failed") from exc
    if reconstructed != value:
        raise Qwen38ChatError("anchor receipt reconstruction differs")
    for key in ("prefix_sha256", "receipt_sha256"):
        if not _is_sha256(document.get(key)):
            raise Qwen38ChatError(f"anchor receipt {key} is invalid")
    _positive_int(document.get("prefix_length"), "anchor receipt prefix_length")
    for key in (
        "hit_count",
        "seed_hidden_bytes",
        "snapshot_manifest_bytes",
        "snapshot_payload_bytes",
        "state_bytes",
    ):
        item = document.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise Qwen38ChatError(f"anchor receipt {key} is invalid")
    try:
        json.dumps(document, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise Qwen38ChatError("anchor receipt is not canonical JSON") from exc
    return document


def _anchor_seed_sha256(value: torch.Tensor) -> str:
    if not isinstance(value, torch.Tensor):
        raise Qwen38ChatError("anchor seed is not a tensor")
    cpu = value.detach().to(device="cpu").contiguous()
    raw = cpu.view(torch.uint8).numpy().reshape(-1).tobytes()
    header = json.dumps(
        {
            "dtype": str(cpu.dtype).removeprefix("torch."),
            "schema": SEMANTIC_ANCHOR_SEED_SCHEMA,
            "shape": list(cpu.shape),
        },
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + raw).hexdigest()


def _anchor_hit_evidence(
    restored: object,
    *,
    prompt_tokens: int,
    generation: Mapping[str, Any],
    restore_seconds: float,
    n_layers: int,
    final_state_committed: bool = True,
) -> dict[str, Any]:
    if type(restored) is not RestoredAnchor:
        raise Qwen38ChatError("anchor cache returned an unsealed restore result")
    anchor = restored.anchor
    receipt = _anchor_document(anchor)
    prefix_tokens = receipt["prefix_length"]
    exact_prefix = getattr(restored, "exact_prefix", None)
    query_length = getattr(restored, "query_length", None)
    suffix_start = getattr(restored, "suffix_start", None)
    seed_hidden = getattr(restored, "seed_hidden", None)
    if (
        not isinstance(exact_prefix, bool)
        or query_length != prompt_tokens
        or suffix_start != prefix_tokens
        or exact_prefix != (prefix_tokens == prompt_tokens)
        or (exact_prefix and seed_hidden is None)
        or (not exact_prefix and seed_hidden is not None)
    ):
        raise Qwen38ChatError("restored anchor differs from the prompt contract")
    _positive_int(n_layers, "runtime model decoder depth")
    if not isinstance(final_state_committed, bool):
        raise TypeError("final_state_committed must be a boolean")
    final_commit = int(final_state_committed)
    forward_baseline = generation["generated_tokens"] + final_commit
    forward_executed = generation["forward_passes"]
    suffix_tokens = prompt_tokens - prefix_tokens
    prefill_sweeps_executed = int(suffix_tokens > 0)
    expected_forwards = (
        generation["generated_tokens"]
        + prefill_sweeps_executed
        - (1 - final_commit)
    )
    if forward_executed != expected_forwards:
        raise Qwen38ChatError("anchor generation forward count is inconsistent")
    snapshot_artifact_bytes = (
        receipt["snapshot_manifest_bytes"]
        + receipt["snapshot_payload_bytes"]
        + (receipt["seed_hidden_bytes"] if exact_prefix else 0)
    )
    return {
        "schema": "immer.qwen3.8-anchor-execution/v1",
        "status": "hit",
        "anchor": receipt,
        "exact_prefix": exact_prefix,
        "prompt_tokens": prompt_tokens,
        "prefix_tokens": prefix_tokens,
        "suffix_tokens": suffix_tokens,
        "restore_seconds": restore_seconds,
        # The cache first hashes every selected artifact, then the native
        # loader reads it again to reconstruct state.  Exact-prefix seeds are
        # likewise authenticated once and decoded once.
        "snapshot_bytes_read": 2 * snapshot_artifact_bytes,
        "forward_passes_baseline": forward_baseline,
        "forward_passes_executed": forward_executed,
        "forward_passes_saved": forward_baseline - forward_executed,
        "prefill_weight_sweeps_baseline": 1,
        "prefill_weight_sweeps_executed": prefill_sweeps_executed,
        "prefill_weight_sweeps_saved": 1 - prefill_sweeps_executed,
        "checkpoint_read_sweeps_baseline": 1,
        "checkpoint_read_sweeps_executed": prefill_sweeps_executed,
        "checkpoint_read_sweeps_saved": 1 - prefill_sweeps_executed,
        "prompt_token_layer_evaluations_baseline": prompt_tokens * n_layers,
        "prompt_token_layer_evaluations_executed": suffix_tokens * n_layers,
        "prompt_token_layer_evaluations_saved": prefix_tokens * n_layers,
        "checkpoint_source_body_bytes_read": generation["source_body_bytes"],
        "checkpoint_linear_calls_executed": generation["linear_calls"],
    }


class _OwnedRuntime:
    """One fully verified local runtime and its ordered resource teardown."""

    def __init__(
        self,
        *,
        mount: CausalWeightMount,
        pager: Qwen38WeightPager,
        model: StreamedQwen38,
        tokenizer: Qwen38Tokenizer,
        tokenizer_sha256: str,
        bundle_receipt: Mapping[str, Any],
        preflight_receipt: Mapping[str, Any],
        fast_mlp_mount: Qwen38FastMlpMount | None = None,
        exact_head_index: ExactHeadIndex | None = None,
        range_prefetcher: MarkovRangePrefetcher | None = None,
    ) -> None:
        self.mount = mount
        self.pager = pager
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_sha256 = tokenizer_sha256
        self.bundle_receipt = dict(bundle_receipt)
        self.preflight_receipt = dict(preflight_receipt)
        self.fast_mlp_mount = fast_mlp_mount
        self.exact_head_index = exact_head_index
        self.range_prefetcher = range_prefetcher
        self.exact_head_receipt = (
            None
            if exact_head_index is None
            else exact_head_index.receipt.to_record()
        )
        self.fast_mlp_receipt = (
            None if fast_mlp_mount is None else fast_mlp_mount.receipt.to_record()
        )
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        failures: list[Exception] = []
        try:
            self.model.reset_state(release=True)
        except Exception as exc:  # release the remaining owners regardless
            failures.append(exc)
        self.model.mlp_sparse_executor = None
        try:
            self.pager.attach_exact_head_index(None)
        except Exception as exc:
            failures.append(exc)
        if self.exact_head_index is not None:
            try:
                self.exact_head_index.close()
            except Exception as exc:
                failures.append(exc)
        if self.fast_mlp_mount is not None:
            try:
                self.fast_mlp_mount.close()
            except Exception as exc:
                failures.append(exc)
        if self.range_prefetcher is not None:
            try:
                self.mount.source.set_access_observer(
                    None,
                    prepare_identity=False,
                )
                self.range_prefetcher.close()
            except Exception as exc:
                failures.append(exc)
        try:
            self.pager.close()
        except Exception as exc:
            failures.append(exc)
        try:
            self.mount.close()
        except Exception as exc:
            failures.append(exc)
        self._closed = True
        if failures:
            raise Qwen38ChatError(
                "Qwen3.8 runtime cleanup failed: "
                f"{type(failures[0]).__name__}: {failures[0]}"
            ) from failures[0]


def _open_local_runtime(
    *,
    bundle_path: Path,
    tokenizer_path: Path,
    identity: LogicalModelIdentity,
    require_official_config: bool,
    device: str,
    compute_dtype: str,
    source_budget_mb: float,
    max_resident_bytes: int,
    max_context_tokens: int,
    fast_mlp_paths: Qwen38FastMlpPaths | None = None,
    fast_mlp_source_budget_mb: float | None = None,
    fast_mlp_max_resident_bytes: int | None = None,
    fast_mlp_active_layers: Sequence[int] | None = None,
    fast_mlp_online_state_path: Path | None = None,
    exact_head_root: Path | None = None,
    exact_head_block_rows: int | None = None,
    exact_head_max_bytes: int = 128 * 1024**2,
    range_markov_state_path: Path | None = None,
    range_prefetch_max_bytes: int = 64 * 1024**2,
    range_prefetch_min_support: int = 2,
    range_prefetch_min_confidence: float = 0.65,
    range_prefetch_beam_horizon: int = 3,
    range_prefetch_beam_width: int = 4,
    range_prefetch_hint_cooldown: int = 2,
) -> _OwnedRuntime:
    """Open one pinned local causal model; no remote source exists here."""

    if not bundle_path.is_dir():
        raise FileNotFoundError(f"causal bundle directory is missing: {bundle_path}")
    tokenizer_sha256 = _file_sha256(tokenizer_path)

    mount: CausalWeightMount | None = None
    pager: Qwen38WeightPager | None = None
    model: StreamedQwen38 | None = None
    fast_mlp_mount: Qwen38FastMlpMount | None = None
    exact_head_index: ExactHeadIndex | None = None
    range_prefetcher: MarkovRangePrefetcher | None = None
    try:
        mount = CausalWeightMount(
            bundle_path,
            identity,
            budget_mb=source_budget_mb,
        )
        bundle_receipt = verify_qwen38_causal_mount(
            mount,
            require_official_config=require_official_config,
        )
        config = Qwen38Config.from_file(
            mount.weights_root / "config.json",
            require_official=require_official_config,
        )
        pager = Qwen38WeightPager(
            mount.source,
            device=device,
            compute_dtype=compute_dtype,
            max_resident_bytes=max_resident_bytes,
            close_source=False,
            require_source_identity=True,
            causal_tensor_reader=mount.tensor_reader,
        )
        if exact_head_root is not None:
            if exact_head_block_rows is None:
                raise Qwen38ChatError(
                    "exact head index requires its configured head block rows"
                )
            exact_head_index = ExactHeadIndex.load(
                exact_head_root,
                max_payload_bytes=exact_head_max_bytes,
            )
            output_head_name = (
                "model.language_model.embed_tokens.weight"
                if config.tie_word_embeddings
                else "lm_head.weight"
            )
            try:
                exact_head_index.validate_mount(
                    pager,
                    name=output_head_name,
                    block_rows=exact_head_block_rows,
                )
            except ExactHeadNotApplicable:
                exact_head_index.close()
                exact_head_index = None
            else:
                pager.attach_exact_head_index(exact_head_index)
        if fast_mlp_paths is not None:
            fast_mlp_mount = open_qwen38_fast_mlp(
                paths=fast_mlp_paths,
                target_mount=mount,
                target_pager=pager,
                config=config,
                source_budget_mb=(
                    source_budget_mb
                    if fast_mlp_source_budget_mb is None
                    else fast_mlp_source_budget_mb
                ),
                max_resident_bytes=(
                    max_resident_bytes
                    if fast_mlp_max_resident_bytes is None
                    else fast_mlp_max_resident_bytes
                ),
                active_layers=fast_mlp_active_layers,
                online_state_path=fast_mlp_online_state_path,
            )
        model = StreamedQwen38(
            config,
            pager,
            max_batch_size=1,
            max_seq_len=max_context_tokens,
            mlp_sparse_executor=(
                None if fast_mlp_mount is None else fast_mlp_mount.executor
            ),
            packed_continuation_gemm=fast_mlp_mount is not None,
        )
        preflight_receipt = model.checkpoint_preflight()
        tokenizer = Qwen38Tokenizer(tokenizer_path, require_official=True)
        if _file_sha256(tokenizer_path) != tokenizer_sha256:
            raise Qwen38ChatError("local tokenizer changed while it was loaded")
        if range_markov_state_path is not None:
            range_prefetcher = MarkovRangePrefetcher(
                range_markov_state_path,
                prefetch_range=mount.source.prefetch_range,
                min_support=range_prefetch_min_support,
                min_confidence=range_prefetch_min_confidence,
                max_prefetch_bytes=range_prefetch_max_bytes,
                beam_horizon=range_prefetch_beam_horizon,
                beam_width=range_prefetch_beam_width,
                hint_cooldown_operations=range_prefetch_hint_cooldown,
            )
            source_metrics = mount.source.metrics()
            range_prefetcher.bind_source_identity(
                mount.source.repo_id,
                mount.source.revision,
                source_metrics.get("inventory_source_fingerprint"),
            )
            mount.source.set_access_observer(
                range_prefetcher,
                prepare_identity=False,
            )
        return _OwnedRuntime(
            mount=mount,
            pager=pager,
            model=model,
            tokenizer=tokenizer,
            tokenizer_sha256=tokenizer_sha256,
            bundle_receipt=bundle_receipt,
            preflight_receipt=preflight_receipt,
            fast_mlp_mount=fast_mlp_mount,
            exact_head_index=exact_head_index,
            range_prefetcher=range_prefetcher,
        )
    except Exception:
        if model is not None:
            try:
                model.reset_state(release=True)
            except Exception:
                pass
        if range_prefetcher is not None:
            try:
                if mount is not None:
                    mount.source.set_access_observer(None, prepare_identity=False)
                range_prefetcher.close()
            except Exception:
                pass
        if pager is not None:
            try:
                pager.attach_exact_head_index(None)
            except Exception:
                pass
            if exact_head_index is not None:
                try:
                    exact_head_index.close()
                except Exception:
                    pass
            if fast_mlp_mount is not None:
                try:
                    fast_mlp_mount.close()
                except Exception:
                    pass
            try:
                pager.close()
            except Exception:
                pass
        if mount is not None:
            mount.close()
        raise


def _open_official_runtime(
    *,
    bundle_path: Path,
    tokenizer_path: Path,
    device: str,
    compute_dtype: str,
    source_budget_mb: float,
    max_resident_bytes: int,
    max_context_tokens: int,
    fast_mlp_paths: Qwen38FastMlpPaths | None = None,
    fast_mlp_source_budget_mb: float | None = None,
    fast_mlp_max_resident_bytes: int | None = None,
    fast_mlp_active_layers: Sequence[int] | None = None,
    fast_mlp_online_state_path: Path | None = None,
    exact_head_root: Path | None = None,
    exact_head_block_rows: int | None = None,
    exact_head_max_bytes: int = 128 * 1024**2,
    range_markov_state_path: Path | None = None,
    range_prefetch_max_bytes: int = 64 * 1024**2,
    range_prefetch_min_support: int = 2,
    range_prefetch_min_confidence: float = 0.65,
    range_prefetch_beam_horizon: int = 3,
    range_prefetch_beam_width: int = 4,
    range_prefetch_hint_cooldown: int = 2,
) -> _OwnedRuntime:
    return _open_local_runtime(
        bundle_path=bundle_path,
        tokenizer_path=tokenizer_path,
        identity=LogicalModelIdentity(OFFICIAL_REPO_ID, OFFICIAL_REVISION),
        require_official_config=True,
        device=device,
        compute_dtype=compute_dtype,
        source_budget_mb=source_budget_mb,
        max_resident_bytes=max_resident_bytes,
        max_context_tokens=max_context_tokens,
        fast_mlp_paths=fast_mlp_paths,
        fast_mlp_source_budget_mb=fast_mlp_source_budget_mb,
        fast_mlp_max_resident_bytes=fast_mlp_max_resident_bytes,
        fast_mlp_active_layers=fast_mlp_active_layers,
        fast_mlp_online_state_path=fast_mlp_online_state_path,
        exact_head_root=exact_head_root,
        exact_head_block_rows=exact_head_block_rows,
        exact_head_max_bytes=exact_head_max_bytes,
        range_markov_state_path=range_markov_state_path,
        range_prefetch_max_bytes=range_prefetch_max_bytes,
        range_prefetch_min_support=range_prefetch_min_support,
        range_prefetch_min_confidence=range_prefetch_min_confidence,
        range_prefetch_beam_horizon=range_prefetch_beam_horizon,
        range_prefetch_beam_width=range_prefetch_beam_width,
        range_prefetch_hint_cooldown=range_prefetch_hint_cooldown,
    )


class Qwen38CausalChat:
    """Lazy ``chat`` component backed only by an authenticated local bundle."""

    name = "qwen3.8.causal-chat"
    capabilities = frozenset({"chat"})

    def __init__(
        self,
        bundle_path: str | Path,
        tokenizer_path: str | Path,
        *,
        system_prompt: str = "",
        device: str = "auto",
        compute_dtype: str = "auto",
        source_budget_mb: float = 4_194_304,
        max_resident_bytes: int = 192 * 1024**2,
        max_prompt_tokens: int = 1024,
        max_new_tokens: int = 64,
        max_context_tokens: int = 2048,
        head_block_rows: int = Qwen38WeightPager.DEFAULT_HEAD_BLOCK_ROWS,
        exact_head_root: str | Path | None = None,
        exact_head_max_bytes: int = 128 * 1024**2,
        anchor_cache: SemanticStateAnchorCache | None = None,
        result_cell_code_revision: str | None = None,
        draft_bundle_path: str | Path | None = None,
        draft_mode: str | None = None,
        draft_window: int = 8,
        draft_source_budget_mb: float = 1_048_576,
        draft_max_resident_bytes: int | None = None,
        markov_draft_state_path: str | Path | None = None,
        draft_window_state_path: str | Path | None = None,
        fast_mlp_root: str | Path | None = None,
        fast_mlp_source_budget_mb: float | None = None,
        fast_mlp_max_resident_bytes: int | None = None,
        fast_mlp_active_layers: Sequence[int] | None = None,
        fast_mlp_online_state_path: str | Path | None = None,
        range_markov_state_path: str | Path | None = None,
        range_prefetch_max_bytes: int = 64 * 1024**2,
        range_prefetch_min_support: int = 2,
        range_prefetch_min_confidence: float = 0.65,
        range_prefetch_beam_horizon: int = 3,
        range_prefetch_beam_width: int = 4,
        range_prefetch_hint_cooldown: int = 2,
    ) -> None:
        if not isinstance(bundle_path, (str, Path)):
            raise TypeError("bundle_path must be a local filesystem path")
        if not isinstance(tokenizer_path, (str, Path)):
            raise TypeError("tokenizer_path must be a local filesystem path")
        if not isinstance(system_prompt, str):
            raise TypeError("system_prompt must be text")
        if device not in {"auto", "cpu", "mps"}:
            raise ValueError("device must be auto, cpu, or mps")
        if compute_dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError(
                "compute_dtype must be auto, float16, bfloat16, or float32"
            )
        source_budget_mb = _positive_number(source_budget_mb, "source_budget_mb")
        draft_source_budget_mb = _positive_number(
            draft_source_budget_mb, "draft_source_budget_mb"
        )
        max_resident_bytes = _positive_int(max_resident_bytes, "max_resident_bytes")
        if draft_max_resident_bytes is None:
            draft_max_resident_bytes = 64 * 1024**2
        draft_max_resident_bytes = _positive_int(
            draft_max_resident_bytes, "draft_max_resident_bytes"
        )
        if fast_mlp_source_budget_mb is not None:
            fast_mlp_source_budget_mb = _positive_number(
                fast_mlp_source_budget_mb, "fast_mlp_source_budget_mb"
            )
        if fast_mlp_max_resident_bytes is not None:
            fast_mlp_max_resident_bytes = _positive_int(
                fast_mlp_max_resident_bytes, "fast_mlp_max_resident_bytes"
            )
        max_prompt_tokens = _positive_int(max_prompt_tokens, "max_prompt_tokens")
        max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        max_context_tokens = _positive_int(max_context_tokens, "max_context_tokens")
        head_block_rows = _positive_int(head_block_rows, "head_block_rows")
        if exact_head_root is not None and not isinstance(
            exact_head_root, (str, Path)
        ):
            raise TypeError("exact_head_root must be a local path or None")
        exact_head_max_bytes = _positive_int(
            exact_head_max_bytes, "exact_head_max_bytes"
        )
        if range_markov_state_path is not None and not isinstance(
            range_markov_state_path,
            (str, Path),
        ):
            raise TypeError("range_markov_state_path must be a local path or None")
        range_prefetch_max_bytes = _positive_int(
            range_prefetch_max_bytes,
            "range_prefetch_max_bytes",
        )
        range_prefetch_min_support = _positive_int(
            range_prefetch_min_support,
            "range_prefetch_min_support",
        )
        if (
            isinstance(range_prefetch_min_confidence, bool)
            or not isinstance(range_prefetch_min_confidence, (int, float))
            or not math.isfinite(float(range_prefetch_min_confidence))
            or not 0.0 <= float(range_prefetch_min_confidence) <= 1.0
        ):
            raise ValueError("range_prefetch_min_confidence must lie in [0, 1]")
        range_prefetch_min_confidence = float(range_prefetch_min_confidence)
        range_prefetch_beam_horizon = _positive_int(
            range_prefetch_beam_horizon,
            "range_prefetch_beam_horizon",
        )
        range_prefetch_beam_width = _positive_int(
            range_prefetch_beam_width,
            "range_prefetch_beam_width",
        )
        range_prefetch_hint_cooldown = _positive_int(
            range_prefetch_hint_cooldown,
            "range_prefetch_hint_cooldown",
        )
        if range_prefetch_beam_horizon > 8 or range_prefetch_beam_width > 16:
            raise ValueError("range prefetch beam exceeds its bounded topology")
        if (
            anchor_cache is not None
            and type(anchor_cache) is not SemanticStateAnchorCache
        ):
            raise TypeError("anchor_cache must be a SemanticStateAnchorCache or None")
        if draft_bundle_path is not None and not isinstance(
            draft_bundle_path, (str, Path)
        ):
            raise TypeError("draft_bundle_path must be a local path or None")
        if draft_mode is not None and draft_mode not in {"qwen35", "markov"}:
            raise ValueError("draft_mode must be qwen35, markov, or None")
        if (
            isinstance(draft_window, bool)
            or not isinstance(draft_window, int)
            or not 2 <= draft_window <= StreamedQwen38.MAX_CONTINUATION_BLOCK_WIDTH
        ):
            raise ValueError("draft_window must lie in [2, 16]")
        if markov_draft_state_path is not None and not isinstance(
            markov_draft_state_path, (str, Path)
        ):
            raise TypeError("markov_draft_state_path must be a local path or None")
        if draft_window_state_path is not None and not isinstance(
            draft_window_state_path, (str, Path)
        ):
            raise TypeError("draft_window_state_path must be a local path or None")
        if draft_mode is None:
            if draft_bundle_path is not None:
                draft_mode = "qwen35"
            elif markov_draft_state_path is not None:
                draft_mode = "markov"
        if draft_mode == "qwen35" and draft_bundle_path is None:
            raise ValueError("qwen35 draft mode requires draft_bundle_path")
        if draft_mode == "markov" and draft_bundle_path is not None:
            raise ValueError("markov draft mode does not use a draft bundle")
        if draft_mode != "markov" and markov_draft_state_path is not None:
            raise ValueError("markov_draft_state_path requires markov draft mode")
        if draft_window_state_path is not None and draft_mode is None:
            raise ValueError("draft_window_state_path requires a rolling draft mode")
        if draft_window_state_path is not None and draft_window < min(
            DRAFT_WINDOW_ACTIONS
        ):
            raise ValueError("adaptive draft-window ceiling must admit at least K=4")
        if draft_mode is not None and anchor_cache is not None:
            raise ValueError("rolling drafting and anchor restore cannot share a request")
        if fast_mlp_root is not None and not isinstance(fast_mlp_root, (str, Path)):
            raise TypeError("fast_mlp_root must be a local path or None")
        if fast_mlp_online_state_path is not None and not isinstance(
            fast_mlp_online_state_path,
            (str, Path),
        ):
            raise TypeError("fast_mlp_online_state_path must be a local path or None")
        if fast_mlp_active_layers is not None:
            try:
                fast_mlp_active_layers = tuple(fast_mlp_active_layers)
            except TypeError as exc:
                raise TypeError(
                    "fast_mlp_active_layers must be an integer sequence"
                ) from exc
            if (
                not fast_mlp_active_layers
                or fast_mlp_active_layers
                != tuple(sorted(set(fast_mlp_active_layers)))
                or any(
                    isinstance(layer, bool)
                    or not isinstance(layer, int)
                    or layer < 0
                    for layer in fast_mlp_active_layers
                )
            ):
                raise ValueError(
                    "fast_mlp_active_layers must be sorted unique non-negative integers"
                )
        if fast_mlp_root is None and any(
            value is not None
            for value in (
                fast_mlp_source_budget_mb,
                fast_mlp_max_resident_bytes,
                fast_mlp_active_layers,
                fast_mlp_online_state_path,
            )
        ):
            raise ValueError("fast-MLP options require fast_mlp_root")
        if fast_mlp_root is not None and fast_mlp_max_resident_bytes is None:
            fast_mlp_max_resident_bytes = 64 * 1024**2
        if result_cell_code_revision is not None and (
            not isinstance(result_cell_code_revision, str)
            or len(result_cell_code_revision)
            not in _RESULT_CELL_CODE_REVISION_LENGTHS
            or set(result_cell_code_revision) - _SHA256
        ):
            raise ValueError(
                "result_cell_code_revision must be a full lowercase Git/SHA revision"
            )
        if max_prompt_tokens + max_new_tokens > max_context_tokens:
            raise ValueError(
                "max_prompt_tokens plus max_new_tokens exceeds max_context_tokens"
            )
        self._bundle_path = Path(bundle_path).expanduser().absolute()
        self._tokenizer_path = Path(tokenizer_path).expanduser().absolute()
        self._system_prompt = system_prompt.strip()
        self._device = device
        self._compute_dtype = compute_dtype
        self._source_budget_mb = source_budget_mb
        self._max_resident_bytes = max_resident_bytes
        self._max_prompt_tokens = max_prompt_tokens
        self._max_new_tokens = max_new_tokens
        self._max_context_tokens = max_context_tokens
        self._head_block_rows = head_block_rows
        self._exact_head_root = (
            None
            if exact_head_root is None
            else Path(exact_head_root).expanduser().absolute()
        )
        self._exact_head_max_bytes = exact_head_max_bytes
        self._anchor_cache = anchor_cache
        self._draft_bundle_path = (
            None
            if draft_bundle_path is None
            else Path(draft_bundle_path).expanduser().absolute()
        )
        self._draft_mode = draft_mode
        self._draft_window = draft_window
        self._draft_source_budget_mb = draft_source_budget_mb
        self._draft_max_resident_bytes = draft_max_resident_bytes
        self._markov_draft_state_path = (
            None
            if markov_draft_state_path is None
            else Path(markov_draft_state_path).expanduser().absolute()
        )
        self._draft_window_state_path = (
            None
            if draft_window_state_path is None
            else Path(draft_window_state_path).expanduser().absolute()
        )
        self._draft_window_controller = (
            None
            if self._draft_window_state_path is None
            else DraftWindowController(self._draft_window_state_path)
        )
        self._fast_mlp_paths = (
            None
            if fast_mlp_root is None
            else Qwen38FastMlpPaths.from_root(fast_mlp_root)
        )
        self._fast_mlp_source_budget_mb = fast_mlp_source_budget_mb
        self._fast_mlp_max_resident_bytes = fast_mlp_max_resident_bytes
        self._fast_mlp_active_layers = fast_mlp_active_layers
        self._fast_mlp_online_state_path = (
            None
            if fast_mlp_online_state_path is None
            else Path(fast_mlp_online_state_path).expanduser().absolute()
        )
        self._range_markov_state_path = (
            None
            if range_markov_state_path is None
            else Path(range_markov_state_path).expanduser().absolute()
        )
        self._range_prefetch_max_bytes = range_prefetch_max_bytes
        self._range_prefetch_min_support = range_prefetch_min_support
        self._range_prefetch_min_confidence = range_prefetch_min_confidence
        self._range_prefetch_beam_horizon = range_prefetch_beam_horizon
        self._range_prefetch_beam_width = range_prefetch_beam_width
        self._range_prefetch_hint_cooldown = range_prefetch_hint_cooldown
        self._result_cell_code_revision = result_cell_code_revision
        self._runtime: Any | None = None
        self._draft_runtime: Any | None = None
        self._last_draft_evidence: dict[str, Any] | None = None
        self._last_fast_mlp_evidence: dict[str, Any] | None = None
        self._last_exact_head_evidence: dict[str, Any] | None = None
        self._draft_window_selection: DraftWindowSelection | None = None
        self._draft_window_policy_metrics: dict[str, Any] | None = None
        self._pending_draft_window_feedback: dict[str, Any] | None = None
        self._bundle_receipt: dict[str, Any] | None = None
        self._tokenizer_sha256: str | None = None
        self._load_error: str | None = None
        self._close_error: str | None = None
        self._closed = False
        self._lock = threading.RLock()

    def _result_cell_generation_policy_sha256(self) -> str:
        policy: dict[str, Any] = {
            "anchor_cache_enabled": self._anchor_cache is not None,
            "compute_dtype": self._compute_dtype,
            "decoding": "greedy",
            "device": self._device,
            "eos_token_ids": [IM_END_TOKEN_ID, END_OF_TEXT_TOKEN_ID],
            "head_block_rows": self._head_block_rows,
            "max_context_tokens": self._max_context_tokens,
            "max_new_tokens": self._max_new_tokens,
            "max_prompt_tokens": self._max_prompt_tokens,
            "max_resident_bytes": self._max_resident_bytes,
            "prefill_tokenwise": False,
            "retain_final_state": False,
            "schema": RESULT_CELL_GENERATION_POLICY_SCHEMA,
            "source_budget_mb": self._source_budget_mb,
            "thinking": False,
        }
        adaptive_eligible = self._draft_window_controller is not None and any(
            window <= self._draft_window and window <= self._max_new_tokens
            for window in DRAFT_WINDOW_ACTIONS
        )
        fixed_eligible = (
            self._draft_window_controller is None and self._max_new_tokens >= 2
        )
        short_fixed_eligible = (
            self._draft_window_controller is not None
            and 2 <= self._max_new_tokens < min(DRAFT_WINDOW_ACTIONS)
        )
        if self._draft_mode is not None and (
            adaptive_eligible or fixed_eligible or short_fixed_eligible
        ):
            policy["decoding"] = "greedy-rolling-window-draft-verify"
            policy["draft_mode"] = self._draft_mode
            selection = self._draft_window_selection
            policy["draft_window"] = (
                selection.proposed_window
                if selection is not None
                else min(self._draft_window, self._max_new_tokens)
                if short_fixed_eligible
                else (
                    8
                    if self._draft_window_controller is not None
                    and 8 <= self._draft_window
                    and 8 <= self._max_new_tokens
                    else (
                        4
                        if self._draft_window_controller is not None
                        else min(self._draft_window, self._max_new_tokens)
                    )
                )
            )
            if self._draft_window_controller is not None:
                metrics = self._draft_window_policy_metrics
                if metrics is None:
                    metrics = self._draft_window_controller.metrics().to_dict()
                controller_policy: dict[str, Any] = {
                    "actions": list(DRAFT_WINDOW_ACTIONS),
                    "configured_ceiling": self._draft_window,
                    "cold_choice": 8,
                    "context": "bottom-k-token-ngram-dialect",
                    "fixed_share": self._draft_window_controller.FIXED_SHARE,
                    "learning_source": "terminal-target-confirmed-receipts",
                    "metrics": metrics,
                    "nested_horizon_learning": "exact-shorter-prefix/v1",
                    "persistent": True,
                    "policy": "k8-cold-bootstrap-then-sampled-fixed-share",
                    "provider_signals": [
                        "council-confidence",
                        "council-disagreement",
                        "phrase-confidence",
                        "phrase-width",
                    ],
                    "round_window_selector": (
                        "markov-prefix-utility/v1"
                        if self._draft_mode == "markov"
                        else "fixed-request-window"
                    ),
                    "short_window_fallback": short_fixed_eligible,
                    "updates_require_target_receipt": True,
                }
                if selection is not None:
                    controller_policy["selection"] = selection.to_dict()
                policy["draft_window_controller"] = controller_policy
            if self._draft_mode == "qwen35":
                policy["draft_model"] = {
                    "repo_id": QWEN35_DRAFTER_REPO_ID,
                    "revision": QWEN35_DRAFTER_REVISION,
                }
            else:
                policy["markov_draft"] = {
                    "max_history_tokens": 4096,
                    "max_order": 16,
                    "experts": 8,
                    "fixed_share": 0.05,
                    "dialect_profiles": 64,
                    "dialect_sketch_size": 32,
                    "dialect_similarity_threshold": 0.20,
                    "dialect_phrase_min_support": 2,
                    "global_phrase_min_support": 3,
                    "phrase_max_width": min(
                        15,
                        max(0, int(policy["draft_window"]) - 1),
                    ),
                    "provider_abi": MARKOV_DRAFT_PROVIDER_ABI,
                    "round_window_selector": (
                        "markov-prefix-utility/v1"
                        if self._draft_window_controller is not None
                        else "fixed-request-window"
                    ),
                    "persistent": self._markov_draft_state_path is not None,
                }
        elif self._draft_mode is not None:
            policy["draft_fallback"] = {
                "configured_mode": self._draft_mode,
                "reason": (
                    "no-adaptive-window-below-ceiling-and-output-budget"
                    if self._draft_window_controller is not None
                    else "max-new-tokens-below-2"
                ),
            }
            if self._draft_window_controller is not None:
                metrics = self._draft_window_policy_metrics
                if metrics is None:
                    metrics = self._draft_window_controller.metrics().to_dict()
                policy["draft_window_controller"] = {
                    "actions": list(DRAFT_WINDOW_ACTIONS),
                    "configured_ceiling": self._draft_window,
                    "cold_choice": 8,
                    "learning_source": "terminal-target-confirmed-receipts",
                    "metrics": metrics,
                    "persistent": True,
                    "updates_require_target_receipt": True,
                }
        if self._range_markov_state_path is not None:
            policy["range_markov"] = {
                "enabled": True,
                "max_prefetch_bytes": self._range_prefetch_max_bytes,
                "beam_horizon": self._range_prefetch_beam_horizon,
                "beam_width": self._range_prefetch_beam_width,
                "hint_cooldown_operations": (
                    self._range_prefetch_hint_cooldown
                ),
                "min_confidence": self._range_prefetch_min_confidence,
                "min_support": self._range_prefetch_min_support,
                "prefetch": "local-posix-fadvise-willneed/v1",
                "state": "operation-order1-order2-ricci/v1",
            }
        if self._fast_mlp_paths is not None:
            policy["fast_mlp"] = {
                "active_layers": (
                    "all-fitted"
                    if self._fast_mlp_active_layers is None
                    else list(self._fast_mlp_active_layers)
                ),
                "enabled": True,
                "online_state": self._fast_mlp_online_state_path is not None,
            }
            runtime = self._runtime
            receipt = (
                None
                if runtime is None
                else getattr(runtime, "fast_mlp_receipt", None)
            )
            if receipt is not None:
                policy["fast_mlp"]["artifacts"] = {
                    key: receipt[key]
                    for key in (
                        "affine_fit_sha256",
                        "model_pin_sha256",
                        "pilot_manifest_body_sha256",
                        "router_fit_sha256",
                        "transpose_manifest_sha256",
                        "weights_index_sha256",
                    )
                }
                for key in (
                    "adaptive_width_policy_sha256",
                    "initialization",
                    "maximum_selected_block_count_by_layer",
                    "maximum_transport_row_fraction_by_layer",
                    "online_config_sha256",
                    "online_state_persistent",
                ):
                    if key in receipt:
                        policy["fast_mlp"]["artifacts"][key] = receipt[key]
        if self._exact_head_root is not None:
            policy["exact_head"] = {
                "enabled": True,
                "max_bytes": self._exact_head_max_bytes,
            }
            runtime = self._runtime
            receipt = (
                None
                if runtime is None
                else getattr(runtime, "exact_head_receipt", None)
            )
            if receipt is not None:
                policy["exact_head"]["artifact"] = dict(receipt)
        return _digest(policy)

    def _draft_window_runtime_identity(self) -> str:
        bundle = self._bundle_receipt
        tokenizer_sha256 = self._tokenizer_sha256
        if bundle is None or not _is_sha256(tokenizer_sha256):
            raise Qwen38ChatError(
                "draft-window identity requires a loaded target runtime"
            )
        if self._draft_mode == "markov":
            provider: dict[str, Any] = {
                "abi": MARKOV_DRAFT_PROVIDER_ABI,
                "alpha": 0.5,
                "backoff_strength": 3.0,
                "experts": 8,
                "fixed_share": 0.05,
                "kind": "markov-council",
                "max_history_tokens": 4096,
                "max_order": 16,
                "min_count": 1,
                "phrase_max_context": 8,
                "phrase_max_width": 15,
                "schema": MARKOV_DRAFT_STATE_SCHEMA,
                "state_path": (
                    None
                    if self._markov_draft_state_path is None
                    else str(self._markov_draft_state_path)
                ),
            }
        elif self._draft_mode == "qwen35":
            provider = {
                "bundle_path": (
                    None
                    if self._draft_bundle_path is None
                    else str(self._draft_bundle_path)
                ),
                "kind": "qwen35",
                "repo_id": QWEN35_DRAFTER_REPO_ID,
                "revision": QWEN35_DRAFTER_REVISION,
            }
        else:
            raise Qwen38ChatError("draft-window identity lacks a draft provider")
        return _digest(
            {
                "provider": provider,
                "schema": "immer.qwen3.8-draft-window-runtime-identity/v1",
                "target_bundle": bundle,
                "target_repo_id": OFFICIAL_REPO_ID,
                "target_revision": OFFICIAL_REVISION,
                "tokenizer_sha256": tokenizer_sha256,
            }
        )

    def _result_cell_binding_receipt(
        self,
        *,
        question: str,
        rendered_prompt: str,
        prompt_ids: tuple[int, ...],
    ) -> dict[str, Any] | None:
        code_revision = self._result_cell_code_revision
        if code_revision is None:
            return None
        # Imported only for the explicitly enabled OoE path.  The default Qwen
        # facade remains dependency- and evidence-compatible with prior runs.
        from ..ooe.result_cells import (
            ResultCellBinding,
            qwen_result_binding_evidence,
        )
        from .cartography_probe import prompt_token_sha256
        from .semantic_atlas import ModelPin

        bundle = self._bundle_receipt
        tokenizer_sha256 = self._tokenizer_sha256
        if bundle is None:
            raise Qwen38ChatError("authenticated runtime bundle receipt is missing")
        if not _is_sha256(tokenizer_sha256):
            raise Qwen38ChatError("runtime tokenizer receipt is invalid")
        pin = ModelPin(
            repo_id=OFFICIAL_REPO_ID,
            revision=OFFICIAL_REVISION,
            bundle_fingerprint=bundle["layout_fingerprint"],
            bundle_manifest_sha256=bundle["manifest_sha256"],
            code_revision=code_revision,
        )
        binding = ResultCellBinding(
            model_pin=pin,
            tokenizer_sha256=tokenizer_sha256,
            question_sha256=hashlib.sha256(question.encode("utf-8")).hexdigest(),
            rendered_prompt_sha256=hashlib.sha256(
                rendered_prompt.encode("utf-8")
            ).hexdigest(),
            rendered_prompt_token_sha256=prompt_token_sha256(prompt_ids),
            system_prompt_sha256=hashlib.sha256(
                self._system_prompt.encode("utf-8")
            ).hexdigest(),
            generation_policy_sha256=(
                self._result_cell_generation_policy_sha256()
            ),
        )
        return qwen_result_binding_evidence(binding)

    @property
    def model_id(self) -> str:
        return OFFICIAL_REPO_ID

    @property
    def revision(self) -> str:
        return OFFICIAL_REVISION

    @property
    def loaded(self) -> bool:
        with self._lock:
            return self._runtime is not None

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def _open_runtime(self) -> _OwnedRuntime:
        return _open_official_runtime(
            bundle_path=self._bundle_path,
            tokenizer_path=self._tokenizer_path,
            device=self._device,
            compute_dtype=self._compute_dtype,
            source_budget_mb=self._source_budget_mb,
            max_resident_bytes=self._max_resident_bytes,
            max_context_tokens=self._max_context_tokens,
            fast_mlp_paths=self._fast_mlp_paths,
            fast_mlp_source_budget_mb=self._fast_mlp_source_budget_mb,
            fast_mlp_max_resident_bytes=self._fast_mlp_max_resident_bytes,
            fast_mlp_active_layers=self._fast_mlp_active_layers,
            fast_mlp_online_state_path=self._fast_mlp_online_state_path,
            exact_head_root=self._exact_head_root,
            exact_head_block_rows=self._head_block_rows,
            exact_head_max_bytes=self._exact_head_max_bytes,
            range_markov_state_path=self._range_markov_state_path,
            range_prefetch_max_bytes=self._range_prefetch_max_bytes,
            range_prefetch_min_support=self._range_prefetch_min_support,
            range_prefetch_min_confidence=self._range_prefetch_min_confidence,
            range_prefetch_beam_horizon=self._range_prefetch_beam_horizon,
            range_prefetch_beam_width=self._range_prefetch_beam_width,
            range_prefetch_hint_cooldown=self._range_prefetch_hint_cooldown,
        )

    def _open_draft_runtime(self) -> _OwnedRuntime:
        if self._draft_bundle_path is None:
            raise Qwen38ChatError("rolling draft bundle is not configured")
        return _open_local_runtime(
            bundle_path=self._draft_bundle_path,
            tokenizer_path=self._tokenizer_path,
            identity=LogicalModelIdentity(
                QWEN35_DRAFTER_REPO_ID,
                QWEN35_DRAFTER_REVISION,
            ),
            require_official_config=False,
            device=self._device,
            compute_dtype=self._compute_dtype,
            source_budget_mb=self._draft_source_budget_mb,
            max_resident_bytes=self._draft_max_resident_bytes,
            max_context_tokens=self._max_context_tokens,
        )

    def _load_draft_locked(self, target: _OwnedRuntime) -> _OwnedRuntime:
        if self._draft_runtime is None:
            self._draft_runtime = self._open_draft_runtime()
        draft = self._draft_runtime
        if (
            draft.tokenizer_sha256 != target.tokenizer_sha256
            or draft.model.config.vocab_size != target.model.config.vocab_size
        ):
            raise Qwen38ChatError("target and rolling drafter vocabularies differ")
        return draft

    def _generate_locked(
        self,
        runtime: _OwnedRuntime,
        prompt_ids: tuple[int, ...],
        generation_options: Mapping[str, Any],
    ) -> tuple[tuple[int, ...], Mapping[str, Any]]:
        self._last_draft_evidence = None
        self._last_fast_mlp_evidence = None
        self._last_exact_head_evidence = None
        fast_mount = getattr(runtime, "fast_mlp_mount", None)
        fast_before = None if fast_mount is None else fast_mount.metrics()
        exact_head = getattr(runtime, "exact_head_index", None)
        exact_before = None if exact_head is None else exact_head.metrics()
        adaptive_selection = self._draft_window_selection
        draft_enabled = self._draft_mode is not None and (
            (self._draft_window_controller is None and self._max_new_tokens >= 2)
            or adaptive_selection is not None
            or (
                self._draft_window_controller is not None
                and 2 <= self._max_new_tokens < min(DRAFT_WINDOW_ACTIONS)
            )
        )
        if not draft_enabled:
            generated, evidence = runtime.model.generate_greedy(
                [list(prompt_ids)],
                retain_final_state=False,
                **generation_options,
            )
            mapped = _mapping(evidence, "generation evidence")
            self._record_fast_mlp_request(
                runtime,
                fast_before,
                target_source_body_bytes=int(mapped["source_body_bytes"]),
                draft_source_body_bytes=0,
            )
            self._record_exact_head_request(runtime, exact_before)
            return generated, evidence
        eos = tuple(generation_options["eos_token_ids"])
        draft_window = (
            adaptive_selection.proposed_window
            if adaptive_selection is not None
            else min(self._draft_window, self._max_new_tokens)
        )
        if self._draft_mode == "qwen35":
            draft = self._load_draft_locked(runtime)
            provider: Any = Qwen35K4DraftProvider(
                draft.model,
                eos_token_ids=eos,
                head_block_rows=self._head_block_rows,
                window_size=draft_window,
            )
        else:
            provider = FingerprintRollingK4DraftProvider(
                vocab_size=runtime.model.config.vocab_size,
                state_path=self._markov_draft_state_path,
                proposal_width=draft_window - 1,
            )
        rolling_started = time.perf_counter()
        rolling_source_start = _runtime_source_body_bytes(runtime)
        try:
            generated = Qwen38K4SpeculativeDecoder(
                runtime.model,
                provider,
                window_size=draft_window,
                adaptive_round_windows=(
                    adaptive_selection is not None
                    and self._draft_mode == "markov"
                ),
            ).generate_rolling(
                [prompt_ids],
                max_new_tokens=self._max_new_tokens,
                eos_token_ids=eos,
                head_block_rows=self._head_block_rows,
                retain_final_state=False,
            )
            evidence = generated.evidence
            if (
                adaptive_selection is not None
                and getattr(evidence, "window_size", None)
                != adaptive_selection.proposed_window
            ):
                raise Qwen38ChatError(
                    "rolling execution window differs from its Markov selection"
                )
            mapped_evidence = {
                "prompt_token_ids": evidence.prompt_token_ids,
                "generated_token_ids": evidence.generated_token_ids,
                "context_mode": "stateful_autoregressive",
                "stateful_cache": True,
                "general_generation": True,
                "prefill_mode": "batched",
                "forward_passes": evidence.forward_passes,
                "source_body_bytes": evidence.source_body_bytes,
                "linear_calls": evidence.linear_calls,
                "seconds": evidence.seconds,
                "state_bytes": runtime.model.state_bytes,
                "stopped_on_eos": evidence.stopped_on_eos,
                "final_state_committed": evidence.final_state_committed,
            }
            self._stage_draft_window_feedback(
                prompt_ids=prompt_ids,
                generated_ids=tuple(generated.token_ids),
                generation_evidence=mapped_evidence,
                accepted_draft_tokens=evidence.accepted_draft_tokens,
                draft_source_body_bytes=0,
                aux_source_body_bytes=0,
                nested_horizons=_nested_draft_horizons(evidence),
            )
            provider_metrics = provider.metrics()
            fast_request = self._record_fast_mlp_request(
                runtime,
                fast_before,
                target_source_body_bytes=evidence.source_body_bytes,
                draft_source_body_bytes=provider_metrics.source_body_bytes,
            )
            aux_source_body_bytes = (
                0
                if fast_request is None
                else int(fast_request["aux_source_body_bytes"])
            )
            self._last_draft_evidence = {
                "accepted_draft_tokens": evidence.accepted_draft_tokens,
                "mode": self._draft_mode,
                "draft_source_body_bytes": provider_metrics.source_body_bytes,
                "aux_source_body_bytes": aux_source_body_bytes,
                "draft_linear_calls": provider_metrics.linear_calls,
                "target_source_body_bytes": evidence.source_body_bytes,
                "target_linear_calls": evidence.linear_calls,
                "total_source_body_bytes": (
                    evidence.source_body_bytes
                    + provider_metrics.source_body_bytes
                    + aux_source_body_bytes
                ),
                "total_linear_calls": (
                    evidence.linear_calls + provider_metrics.linear_calls
                ),
                "final_state_committed": evidence.final_state_committed,
                "rounds": len(evidence.rounds),
                "round_window_policies": [
                    {
                        "accepted_prefix_length": row.accepted_prefix_length,
                        "provider_proposal_width": len(
                            getattr(row, "provider_proposed_token_ids", ())
                        ),
                        "round_index": row.round_index,
                        "round_policy": (
                            None
                            if getattr(row, "round_policy", None) is None
                            else row.round_policy.to_dict()
                        ),
                        "staged_proposal_width": len(row.proposed_token_ids),
                        "window_size": row.window_size,
                    }
                    for row in evidence.rounds
                    if getattr(row, "proposed_token_ids", ())
                ],
                "adaptive_windows": getattr(evidence, "adaptive_windows", False),
                "used_window_sizes": list(
                    getattr(evidence, "used_window_sizes", ())
                ),
                "window_size": getattr(evidence, "window_size", draft_window),
                "schema": evidence.schema,
                "nested_horizons": [
                    row.to_dict() for row in _nested_draft_horizons(evidence)
                ],
            }
            if adaptive_selection is not None:
                self._last_draft_evidence["window_selection"] = (
                    adaptive_selection.to_dict()
                )
            provider_record = getattr(provider_metrics, "to_dict", None)
            if callable(provider_record):
                self._last_draft_evidence["provider"] = provider_record()
            self._stage_draft_window_feedback(
                prompt_ids=prompt_ids,
                generated_ids=tuple(generated.token_ids),
                generation_evidence=mapped_evidence,
                council_confidence=getattr(
                    provider_metrics,
                    "last_confidence",
                    None,
                ),
                council_disagreement=getattr(
                    provider_metrics,
                    "last_disagreement",
                    None,
                ),
                effective_experts=getattr(
                    provider_metrics,
                    "effective_experts",
                    None,
                ),
                phrase_confidence=getattr(
                    provider_metrics,
                    "last_phrase_confidence",
                    None,
                ),
                phrase_support=getattr(
                    provider_metrics,
                    "last_phrase_support",
                    None,
                ),
                phrase_width=getattr(
                    provider_metrics,
                    "last_phrase_width",
                    None,
                ),
            )
            self._record_exact_head_request(runtime, exact_before)
            return generated.token_ids, mapped_evidence
        except TimeoutError:
            if adaptive_selection is not None and self._pending_draft_window_feedback is None:
                elapsed = time.perf_counter() - rolling_started
                source_bytes = max(
                    0,
                    _runtime_source_body_bytes(runtime) - rolling_source_start,
                )
                timeout_receipt = {
                    "elapsed_seconds": elapsed,
                    "schema": "immer.qwen3.8-draft-window-timeout/v1",
                    "selection_id": adaptive_selection.selection_id,
                    "target_source_body_bytes": source_bytes,
                }
                self._pending_draft_window_feedback = {
                    "_terminal_outcome": "timeout",
                    "accepted_draft_tokens": 0,
                    "aux_source_body_bytes": 0,
                    "council_confidence": None,
                    "council_disagreement": None,
                    "draft_source_body_bytes": 0,
                    "effective_experts": None,
                    "emitted_tokens": 0,
                    "nested_horizons": (),
                    "phrase_confidence": None,
                    "phrase_support": 0,
                    "phrase_width": 0,
                    "proposed_window": adaptive_selection.proposed_window,
                    "seconds": elapsed,
                    "target_forwards": 1,
                    "target_receipt_sha256": _digest(timeout_receipt),
                    "target_source_body_bytes": source_bytes,
                }
            raise
        finally:
            provider.close()

    def _stage_draft_window_feedback(
        self,
        *,
        prompt_ids: tuple[int, ...],
        generated_ids: tuple[int, ...],
        generation_evidence: Mapping[str, Any],
        accepted_draft_tokens: int | None = None,
        draft_source_body_bytes: int | None = None,
        aux_source_body_bytes: int | None = None,
        nested_horizons: tuple[DraftWindowNestedHorizon, ...] | None = None,
        council_confidence: float | None = None,
        council_disagreement: float | None = None,
        effective_experts: float | None = None,
        phrase_confidence: float | None = None,
        phrase_support: int | None = None,
        phrase_width: int | None = None,
    ) -> None:
        selection = self._draft_window_selection
        if selection is None or self._draft_window_controller is None:
            return
        draft = self._last_draft_evidence
        if draft is None and any(
            value is None
            for value in (
                accepted_draft_tokens,
                draft_source_body_bytes,
                aux_source_body_bytes,
            )
        ):
            raise Qwen38ChatError(
                "adaptive draft-window request lacks rolling draft evidence"
            )
        if accepted_draft_tokens is None:
            accepted_draft_tokens = int(draft["accepted_draft_tokens"])
        if draft_source_body_bytes is None:
            draft_source_body_bytes = int(draft["draft_source_body_bytes"])
        if aux_source_body_bytes is None:
            aux_source_body_bytes = int(draft["aux_source_body_bytes"])
        if nested_horizons is None:
            nested_horizons = (
                ()
                if self._pending_draft_window_feedback is None
                else tuple(
                    self._pending_draft_window_feedback.get(
                        "nested_horizons",
                        (),
                    )
                )
            )
        pending = self._pending_draft_window_feedback or {}
        if council_confidence is None:
            council_confidence = pending.get("council_confidence")
        if council_disagreement is None:
            council_disagreement = pending.get("council_disagreement")
        if effective_experts is None:
            effective_experts = pending.get("effective_experts")
        if phrase_confidence is None:
            phrase_confidence = pending.get("phrase_confidence")
        if phrase_support is None:
            phrase_support = int(pending.get("phrase_support", 0))
        if phrase_width is None:
            phrase_width = int(pending.get("phrase_width", 0))
        receipt = _compact_generation_receipt(
            generation_evidence,
            prompt_ids=prompt_ids,
            generated_ids=generated_ids,
        )
        self._pending_draft_window_feedback = {
            "accepted_draft_tokens": accepted_draft_tokens,
            "aux_source_body_bytes": aux_source_body_bytes,
            "council_confidence": council_confidence,
            "council_disagreement": council_disagreement,
            "draft_source_body_bytes": draft_source_body_bytes,
            "emitted_tokens": len(generated_ids),
            "effective_experts": effective_experts,
            "nested_horizons": nested_horizons,
            "phrase_confidence": phrase_confidence,
            "phrase_support": phrase_support,
            "phrase_width": phrase_width,
            "proposed_window": selection.proposed_window,
            "seconds": float(receipt["seconds"]),
            "target_forwards": int(receipt["forward_passes"]),
            "target_receipt_sha256": _digest(receipt),
            "target_source_body_bytes": int(receipt["source_body_bytes"]),
        }

    def _record_exact_head_request(
        self,
        runtime: _OwnedRuntime,
        before: Mapping[str, object] | None,
    ) -> dict[str, Any] | None:
        index = getattr(runtime, "exact_head_index", None)
        if index is None or before is None:
            return None
        after = index.metrics()
        counters = {}
        for key, value in after.items():
            previous = before.get(key)
            if (
                isinstance(value, int)
                and not isinstance(value, bool)
                and isinstance(previous, int)
                and not isinstance(previous, bool)
            ):
                delta = value - previous
                if delta < 0:
                    raise Qwen38ChatError(
                        "exact-head request counters moved backwards"
                    )
                counters[key] = delta
        request = {
            **counters,
            "last_fallback_reason": after.get("last_fallback_reason", ""),
            "manifest_sha256": after.get("manifest_sha256"),
            "schema": "immer.qwen3.8-exact-head-request/v1",
        }
        self._last_exact_head_evidence = request
        return request

    def _record_fast_mlp_request(
        self,
        runtime: _OwnedRuntime,
        before: Mapping[str, int] | None,
        *,
        target_source_body_bytes: int,
        draft_source_body_bytes: int,
    ) -> dict[str, Any] | None:
        mount = getattr(runtime, "fast_mlp_mount", None)
        if mount is None or before is None:
            return None
        after = mount.metrics()
        delta = {
            key: int(after.get(key, 0)) - int(before.get(key, 0))
            for key in after
        }
        if any(value < 0 for value in delta.values()):
            raise Qwen38ChatError("fast-MLP request counters moved backwards")
        aux = delta["source_body_bytes"]
        request = {
            "aux_logical_weight_bytes": delta["logical_weight_bytes"],
            "aux_source_body_bytes": aux,
            "draft_source_body_bytes": draft_source_body_bytes,
            "pilot_source_body_bytes": delta["pilot_source_body_bytes"],
            "schema": "immer.qwen3.8-fast-mlp-request/v1",
            "target_source_body_bytes": target_source_body_bytes,
            "total_source_body_bytes": (
                target_source_body_bytes + draft_source_body_bytes + aux
            ),
            "transpose_source_body_bytes": delta[
                "transpose_source_body_bytes"
            ],
        }
        for field in (
            "online_confirmed_rows",
            "online_exact_waves",
            "online_output_confirmed_rows",
            "online_output_shadow_waves",
            "online_sparse_rows",
            "online_sparse_waves",
            "online_surprises",
            "online_width_updates",
        ):
            if field in delta:
                request[field] = delta[field]
        self._last_fast_mlp_evidence = request
        return request

    def _base_evidence(self) -> dict[str, Any]:
        evidence: dict[str, Any] = {
            "execution": "local-authenticated-causal-bundle/v1",
            "model": OFFICIAL_REPO_ID,
            "revision": OFFICIAL_REVISION,
            "thinking": False,
        }
        if self._bundle_receipt is not None:
            evidence["bundle"] = dict(self._bundle_receipt)
        if self._tokenizer_sha256 is not None:
            evidence["tokenizer_sha256"] = self._tokenizer_sha256
        if self._draft_runtime is not None:
            evidence["draft_bundle"] = dict(self._draft_runtime.bundle_receipt)
        fast_mlp_receipt = (
            None
            if self._runtime is None
            else getattr(self._runtime, "fast_mlp_receipt", None)
        )
        if fast_mlp_receipt is not None:
            evidence["fast_mlp"] = dict(fast_mlp_receipt)
        exact_head_receipt = (
            None
            if self._runtime is None
            else getattr(self._runtime, "exact_head_receipt", None)
        )
        if exact_head_receipt is not None:
            evidence["exact_head"] = dict(exact_head_receipt)
        range_prefetcher = (
            None
            if self._runtime is None
            else getattr(self._runtime, "range_prefetcher", None)
        )
        if range_prefetcher is not None:
            evidence["range_markov"] = range_prefetcher.metrics()
        return evidence

    def _load_locked(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        if self._closed:
            raise Qwen38ChatError("Qwen3.8 chat component is closed")
        if self._load_error is not None:
            raise Qwen38ChatError(self._load_error)
        runtime = None
        try:
            runtime = self._open_runtime()
            model = getattr(runtime, "model", None)
            tokenizer = getattr(runtime, "tokenizer", None)
            if not callable(getattr(model, "generate_greedy", None)) or not callable(
                getattr(model, "reset_state", None)
            ):
                raise TypeError(
                    "runtime model must provide generate_greedy/reset_state"
                )
            if not callable(getattr(tokenizer, "encode", None)) or not callable(
                getattr(tokenizer, "decode", None)
            ):
                raise TypeError("runtime tokenizer must provide encode/decode")
            if not callable(getattr(runtime, "close", None)):
                raise TypeError("runtime must provide close")
            bundle_receipt = _compact_bundle_receipt(
                getattr(runtime, "bundle_receipt", None)
            )
            tokenizer_sha256 = getattr(runtime, "tokenizer_sha256", None)
            if not _is_sha256(tokenizer_sha256):
                raise Qwen38ChatError("runtime tokenizer receipt is invalid")
            model_context = _positive_int(
                getattr(model, "max_seq_len", None), "runtime model max_seq_len"
            )
            if model_context < self._max_context_tokens:
                raise Qwen38ChatError(
                    "runtime model context is smaller than the facade contract"
                )
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            close = getattr(runtime, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as close_exc:
                    self._load_error += (
                        f"; cleanup: {type(close_exc).__name__}: {close_exc}"
                    )
            raise Qwen38ChatError(self._load_error) from exc
        self._runtime = runtime
        self._bundle_receipt = bundle_receipt
        self._tokenizer_sha256 = str(tokenizer_sha256)
        return runtime

    def _execute_locked(self, runtime: Any, text: str) -> Result:
        prompt = Qwen38Tokenizer.render_no_thinking_prompt(
            self._system_prompt,
            text,
        )
        raw_prompt = runtime.tokenizer.encode(prompt)
        prompt_ids = _token_ids(
            getattr(raw_prompt, "ids", raw_prompt), "encoded prompt"
        )
        if not prompt_ids:
            raise _RequestRejected("official no-thinking prompt encoded to no tokens")
        if len(prompt_ids) > self._max_prompt_tokens:
            raise _RequestRejected(
                f"prompt has {len(prompt_ids)} tokens, limit is "
                f"{self._max_prompt_tokens}"
            )
        if len(prompt_ids) + self._max_new_tokens > self._max_context_tokens:
            raise _RequestRejected("prompt plus output budget exceeds model context")
        config = getattr(runtime.model, "config", None)
        vocab_size = _positive_int(
            getattr(config, "vocab_size", None), "runtime model vocabulary"
        )
        if any(token_id >= vocab_size for token_id in prompt_ids):
            raise _RequestRejected("prompt token is outside the checkpoint vocabulary")

        if self._draft_window_controller is not None:
            self._draft_window_controller.bind_policy_identity(
                self._draft_window_runtime_identity()
            )
            self._draft_window_selection = self._draft_window_controller.choose(
                prompt_ids,
                max_window=self._draft_window,
                max_new_tokens=self._max_new_tokens,
            )
            self._draft_window_policy_metrics = (
                self._draft_window_controller.metrics().to_dict()
                if self._draft_window_selection is None
                else self._draft_window_controller.metrics_for_selection(
                    self._draft_window_selection
                ).to_dict()
            )

        restored = None
        restore_seconds = 0.0
        anchor_miss: dict[str, Any] | None = None
        generation_options: dict[str, Any] = {
            "max_new_tokens": self._max_new_tokens,
            "prefill_tokenwise": False,
            "eos_token_ids": (IM_END_TOKEN_ID, END_OF_TEXT_TOKEN_ID),
            "head_block_rows": self._head_block_rows,
        }
        if self._anchor_cache is not None:
            before_restore = _anchor_model_state(runtime.model)
            if before_restore != (0, False, 0, None):
                raise Qwen38ChatError("anchor restore requires an empty released model")
            restore_started = time.perf_counter()
            restored = self._anchor_cache.restore_deepest(
                runtime.model,
                prompt_ids,
            )
            restore_seconds = time.perf_counter() - restore_started
            if restored is None:
                if _anchor_model_state(runtime.model) != before_restore:
                    raise Qwen38ChatError("anchor-cache miss mutated model state")
                anchor_miss = {
                    "schema": "immer.qwen3.8-anchor-execution/v1",
                    "status": "miss",
                    "prompt_tokens": len(prompt_ids),
                    "restore_seconds": restore_seconds,
                }
            else:
                if type(restored) is not RestoredAnchor:
                    raise Qwen38ChatError(
                        "anchor cache returned an unsealed restore result"
                    )
                anchor = restored.anchor
                anchor_document = _anchor_document(anchor)
                prefix_length = anchor_document["prefix_length"]
                exact_prefix = restored.exact_prefix
                if (
                    restored.query_length != len(prompt_ids)
                    or restored.suffix_start != prefix_length
                    or not isinstance(exact_prefix, bool)
                    or exact_prefix != (prefix_length == len(prompt_ids))
                    or anchor_document["prefix_sha256"]
                    != token_prefix_sha256(prompt_ids[:prefix_length])
                    or exact_prefix != (restored.seed_hidden is not None)
                ):
                    raise Qwen38ChatError(
                        "restored anchor differs from the tokenized prompt"
                    )
                if exact_prefix and (
                    anchor.seed_hidden_tensor_sha256 is None
                    or _anchor_seed_sha256(restored.seed_hidden)
                    != anchor.seed_hidden_tensor_sha256
                ):
                    raise Qwen38ChatError(
                        "restored seed differs from the sealed anchor receipt"
                    )
                restored_state = _anchor_model_state(runtime.model)
                if restored_state != (
                    prefix_length,
                    False,
                    anchor_document["state_bytes"],
                    1,
                ):
                    raise Qwen38ChatError(
                        "restored model state differs from anchor receipt"
                    )
                generation_options.update(
                    {
                        "restored_prefix_length": prefix_length,
                        "restored_seed_hidden": restored.seed_hidden,
                    }
                )

        raw_generated, raw_evidence = self._generate_locked(
            runtime,
            prompt_ids,
            generation_options,
        )
        generated_ids = _token_ids(raw_generated, "generated output")
        if len(generated_ids) > self._max_new_tokens:
            raise Qwen38ChatError("generated output exceeds its token budget")
        if any(token_id >= vocab_size for token_id in generated_ids):
            raise Qwen38ChatError(
                "generated token is outside the checkpoint vocabulary"
            )
        eos_positions = [
            index
            for index, token_id in enumerate(generated_ids)
            if token_id in {IM_END_TOKEN_ID, END_OF_TEXT_TOKEN_ID}
        ]
        if eos_positions and eos_positions[0] != len(generated_ids) - 1:
            raise Qwen38ChatError("generated output contains tokens after EOS")
        receipt = _compact_generation_receipt(
            raw_evidence,
            prompt_ids=prompt_ids,
            generated_ids=generated_ids,
        )
        self._stage_draft_window_feedback(
            prompt_ids=prompt_ids,
            generated_ids=generated_ids,
            generation_evidence=raw_evidence,
        )
        if bool(receipt["stopped_on_eos"]) != bool(eos_positions):
            raise Qwen38ChatError("generation EOS receipt differs from output")
        decoded = runtime.tokenizer.decode(generated_ids)
        if not isinstance(decoded, str):
            raise Qwen38ChatError("runtime tokenizer returned a non-text response")
        output = decoded.strip()
        evidence = {
            **self._base_evidence(),
            "generation": receipt,
            "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        }
        if self._last_draft_evidence is not None:
            evidence["draft"] = dict(self._last_draft_evidence)
        if self._last_fast_mlp_evidence is not None:
            evidence["fast_mlp"] = {
                **dict(evidence["fast_mlp"]),
                "request": dict(self._last_fast_mlp_evidence),
            }
        if self._last_exact_head_evidence is not None:
            evidence["exact_head"] = {
                **dict(evidence["exact_head"]),
                "request": dict(self._last_exact_head_evidence),
            }
        result_cell_binding = self._result_cell_binding_receipt(
            question=text,
            rendered_prompt=prompt,
            prompt_ids=prompt_ids,
        )
        if result_cell_binding is not None:
            evidence["result_cell_binding_receipt"] = result_cell_binding
        if restored is not None:
            evidence["anchor_cache"] = _anchor_hit_evidence(
                restored,
                prompt_tokens=len(prompt_ids),
                generation=receipt,
                restore_seconds=restore_seconds,
                n_layers=_positive_int(
                    getattr(config, "n_layers", None),
                    "runtime model decoder depth",
                ),
                final_state_committed=False,
            )
        elif anchor_miss is not None:
            evidence["anchor_cache"] = anchor_miss
        if not output:
            return Result(
                ExecutionStatus.ABSTAINED,
                self.name,
                reason="Qwen3.8 decoded an empty response",
                evidence=evidence,
            )
        return Result(
            ExecutionStatus.OK,
            self.name,
            output=output,
            evidence=evidence,
        )

    def _retire_runtime_locked(self, error: Exception) -> str:
        detail = f"{type(error).__name__}: {error}"
        runtime = self._runtime
        draft_runtime = self._draft_runtime
        self._runtime = None
        self._draft_runtime = None
        self._load_error = f"runtime retired after cleanup failure: {detail}"
        if runtime is not None:
            try:
                runtime.close()
            except Exception as close_exc:
                detail += f"; close: {type(close_exc).__name__}: {close_exc}"
        if draft_runtime is not None:
            try:
                draft_runtime.close()
            except Exception as close_exc:
                detail += f"; draft close: {type(close_exc).__name__}: {close_exc}"
        return detail

    def _finalize_draft_window_result(
        self,
        result: Result,
        *,
        failure_outcome: str | None = None,
    ) -> Result:
        selection = self._draft_window_selection
        controller = self._draft_window_controller
        if selection is None or controller is None:
            return result
        record: dict[str, Any] = {
            "schema": "immer.qwen3.8-draft-window-request/v2",
            "selection": selection.to_dict(),
            "settled": False,
        }
        pending = self._pending_draft_window_feedback
        if pending is None:
            record["reason"] = "no-verified-target-receipt"
            metrics = self._draft_window_policy_metrics
            if metrics is None:
                metrics = controller.metrics().to_dict()
            record["metrics"] = metrics
            return Result(
                result.status,
                result.component,
                output=result.output,
                reason=result.reason,
                evidence={**dict(result.evidence), "draft_window": record},
            )
        feedback_values = dict(pending)
        outcome = str(feedback_values.pop("_terminal_outcome", "ok"))
        record["request_status"] = result.status.value
        if failure_outcome is not None:
            record["post_generation_outcome"] = failure_outcome
        try:
            feedback = DraftWindowFeedback(**feedback_values, outcome=outcome)
            metrics = controller.settle(selection, feedback)
        except Exception as exc:
            record["settlement"] = {
                "detail": f"{type(exc).__name__}: {exc}",
                "status": "error",
            }
            return Result(
                ExecutionStatus.ERROR,
                self.name,
                reason="Qwen3.8 draft-window settlement failed",
                evidence={**dict(result.evidence), "draft_window": record},
            )
        record.update(
            {
                "feedback": feedback.to_dict(),
                "metrics": metrics.to_dict(),
                "settled": True,
            }
        )
        evidence = {**dict(result.evidence), "draft_window": record}
        draft = evidence.get("draft")
        if isinstance(draft, Mapping):
            evidence["draft"] = {
                **dict(draft),
                "window_controller": {
                    "proposed_window": selection.proposed_window,
                    "reward": feedback.reward,
                    "settled": True,
                },
            }
        return Result(
            result.status,
            result.component,
            output=result.output,
            reason=result.reason,
            evidence=evidence,
        )

    def handle(self, request: Request) -> Result:
        if request.capability not in self.capabilities:
            return Result(
                ExecutionStatus.REJECTED,
                self.name,
                reason="unsupported capability",
            )
        if not isinstance(request.payload, str) or not request.payload.strip():
            return Result(
                ExecutionStatus.REJECTED,
                self.name,
                reason="chat payload must be non-empty text",
            )

        with self._lock:
            self._draft_window_selection = None
            self._draft_window_policy_metrics = None
            self._pending_draft_window_feedback = None
            if self._closed:
                return Result(
                    ExecutionStatus.UNAVAILABLE,
                    self.name,
                    reason="Qwen3.8 chat component is closed",
                    evidence=self._base_evidence(),
                )
            try:
                runtime = self._load_locked()
            except Qwen38ChatError as exc:
                return Result(
                    ExecutionStatus.UNAVAILABLE,
                    self.name,
                    reason=f"local Qwen3.8 runtime unavailable: {exc}",
                    evidence=self._base_evidence(),
                )

            failure_outcome: str | None = None
            abort: BaseException | None = None
            try:
                result = self._execute_locked(runtime, request.payload.strip())
            except _RequestRejected as exc:
                result = Result(
                    ExecutionStatus.REJECTED,
                    self.name,
                    reason=str(exc),
                    evidence=self._base_evidence(),
                )
            except Exception as exc:
                failure_outcome = (
                    "timeout" if isinstance(exc, TimeoutError) else "error"
                )
                result = Result(
                    ExecutionStatus.ERROR,
                    self.name,
                    reason=f"Qwen3.8 generation failed: {type(exc).__name__}: {exc}",
                    evidence=self._base_evidence(),
                )
            except BaseException as exc:
                abort = exc
                failure_outcome = "aborted"
                result = Result(
                    ExecutionStatus.ERROR,
                    self.name,
                    reason=f"Qwen3.8 generation aborted: {type(exc).__name__}: {exc}",
                    evidence=self._base_evidence(),
                )
            try:
                runtime.model.reset_state(release=True)
            except Exception as exc:
                cleanup = self._retire_runtime_locked(exc)
                failed = Result(
                    ExecutionStatus.ERROR,
                    self.name,
                    reason="Qwen3.8 state cleanup failed",
                    evidence={
                        **self._base_evidence(),
                        "cleanup": {"status": "error", "detail": cleanup},
                    },
                )
                finalized = self._finalize_draft_window_result(
                    failed,
                    failure_outcome=("aborted" if abort is not None else "error"),
                )
                if abort is not None:
                    raise abort
                return finalized
            finalized = self._finalize_draft_window_result(
                result,
                failure_outcome=failure_outcome,
            )
            if abort is not None:
                raise abort
            return finalized

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            runtime = self._runtime
            draft_runtime = self._draft_runtime
            self._runtime = None
            self._draft_runtime = None
            self._closed = True
            if runtime is not None:
                try:
                    runtime.close()
                except Exception as exc:
                    self._close_error = f"{type(exc).__name__}: {exc}"
            if draft_runtime is not None:
                try:
                    draft_runtime.close()
                except Exception as exc:
                    self._close_error = f"{type(exc).__name__}: {exc}"

    def __enter__(self) -> "Qwen38CausalChat":
        with self._lock:
            if self._closed:
                raise Qwen38ChatError("Qwen3.8 chat component is closed")
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


Qwen38Chat = Qwen38CausalChat


__all__ = [
    "RESULT_CELL_GENERATION_POLICY_SCHEMA",
    "Qwen38CausalChat",
    "Qwen38Chat",
    "Qwen38ChatError",
]
