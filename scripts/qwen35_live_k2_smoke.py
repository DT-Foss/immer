#!/usr/bin/env python3
"""Run the pinned local Qwen3.5 drafter against the causal Qwen3.8 target.

This is the end-to-end K=2 integration path: both local checkpoint bundles are
fully authenticated, Qwen3.5-0.8B drafts transactionally through an independent
causal pager, and Qwen3.8-27B alone commits output tokens.  No remote source,
download path, or checkpoint mutation exists in this command.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from typing import Any

import torch

from immer.runtimes.qwen3_8 import (
    CausalWeightMount,
    LogicalModelIdentity,
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    QWEN35_DRAFTER_REPO_ID,
    QWEN35_DRAFTER_REVISION,
    NativeHeadCrsaEvidence,
    Qwen35K2DraftProvider,
    Qwen38Config,
    Qwen38K2SpeculativeDecoder,
    Qwen38NativeHeadCrsa,
    Qwen38Tokenizer,
    Qwen38WeightPager,
    StreamedQwen38,
    verify_qwen38_causal_mount,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_BUNDLE = Path("/app/models/Qwen3.8-27B")
DEFAULT_DRAFT_BUNDLE = Path("/app/models/Qwen3.5-0.8B")
DEFAULT_OUTPUT = ROOT / "results" / "qwen35-live-k2-smoke.json"
RESULT_SCHEMA = "immer.qwen3.5-live-k2-smoke/v1"
MIB = 1024**2

# These are immutable checkpoint identities.  The causal graph revision and
# bundle-manifest seal remain live and are therefore verified, not hard-coded.
_TARGET_PROFILE = {
    "checkpoint_bytes": 55_563_006_776,
    "layout_fingerprint": (
        "8446f49a8ab8b696ede33a072f03be0dd253baf4f624e688e25b27ae843e022d"
    ),
    "shards": 18,
    "tensor_bindings": 1_199,
}
_DRAFT_PROFILE = {
    "checkpoint_bytes": 1_746_942_600,
    "layout_fingerprint": (
        "4e36807736d1109201d3537265a083d544215cc7c026d0b068029dc3271705b4"
    ),
    "shards": 1,
    "tensor_bindings": 488,
}


class LiveK2SmokeError(RuntimeError):
    """The authenticated local target/drafter contract cannot be completed."""


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _multiple_k2_tokens(raw: str) -> int:
    value = _positive_int(raw)
    if value < 2 or value % 2:
        raise argparse.ArgumentTypeError("must be a positive even integer")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def _token_ids(raw: str) -> tuple[int, ...]:
    try:
        values = tuple(int(part.strip()) for part in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "token IDs must be comma-separated integers"
        ) from exc
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("token IDs must be non-negative")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-bundle",
        default=str(DEFAULT_TARGET_BUNDLE),
        help="pinned in-place causal Qwen3.8-27B bundle",
    )
    parser.add_argument(
        "--draft-bundle",
        default=str(DEFAULT_DRAFT_BUNDLE),
        help="pinned in-place causal Qwen3.5-0.8B bundle",
    )
    parser.add_argument(
        "--tokenizer-json",
        help="local Qwen tokenizer.json (default: TARGET_BUNDLE/tokenizer.json)",
    )
    parser.add_argument(
        "--prompt",
        default="What is 17 + 25?",
        help="user message encoded by the local no-thinking chat template",
    )
    parser.add_argument("--system-prompt", default="")
    parser.add_argument(
        "--max-new-tokens",
        type=_multiple_k2_tokens,
        default=2,
        help=(
            "positive even output length; no K=1 round is planned, while a "
            "mismatch-0 may require one exact terminal fallback"
        ),
    )
    parser.add_argument(
        "--expected-token-ids",
        type=_token_ids,
        help="independent comma-separated exact-greedy reference",
    )
    parser.add_argument(
        "--attention-mode",
        choices=("native-crsa", "off"),
        default="native-crsa",
        help="target attention: Prefix-Sinkhorn Head-CRSA or all-softmax",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument(
        "--compute-dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--target-source-budget-mb",
        type=_positive_float,
        default=1_048_576.0,
        help="cumulative local target-range read budget",
    )
    parser.add_argument(
        "--draft-source-budget-mb",
        type=_positive_float,
        default=262_144.0,
        help="cumulative local drafter-range read budget",
    )
    parser.add_argument(
        "--max-resident-mb",
        type=_positive_float,
        default=384.0,
        help="one-pager source+tensor residency bound",
    )
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
        raise LiveK2SmokeError("result is not canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _seal(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    if "sha256" in result:
        raise LiveK2SmokeError("result already contains a seal")
    result["sha256"] = _sha256(result)
    return result


def _atomic_json(path: str | os.PathLike[str], document: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    body = _canonical(dict(document)) + b"\n"
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".pending", dir=destination.parent
        )
    except OSError as exc:
        raise LiveK2SmokeError(f"cannot prepare output: {destination}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        raise LiveK2SmokeError(f"cannot write output: {destination}") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _plain_file(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LiveK2SmokeError(f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise LiveK2SmokeError(f"{label} must be a non-symlink regular file")
    return metadata


def _file_sha256(path: Path, label: str) -> tuple[str, int]:
    opened = _plain_file(path, label)
    flags = os.O_RDONLY | int(getattr(os, "O_CLOEXEC", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    digest = hashlib.sha256()
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise LiveK2SmokeError(f"cannot open {label}: {path}") from exc
    try:
        actual = os.fstat(descriptor)
        if (actual.st_dev, actual.st_ino) != (opened.st_dev, opened.st_ino):
            raise LiveK2SmokeError(f"{label} changed while opening")
        while chunk := os.read(descriptor, 1024**2):
            digest.update(chunk)
        current = os.fstat(descriptor)
        linked = _plain_file(path, label)
        if (
            current.st_size != actual.st_size
            or current.st_mtime_ns != actual.st_mtime_ns
            or current.st_ctime_ns != actual.st_ctime_ns
            or (linked.st_dev, linked.st_ino) != (actual.st_dev, actual.st_ino)
        ):
            raise LiveK2SmokeError(f"{label} changed while hashing")
    finally:
        os.close(descriptor)
    return digest.hexdigest(), actual.st_size


def _metric(owner: object, name: str) -> int:
    metrics = getattr(owner, "metrics", None)
    values = dict(metrics()) if callable(metrics) else {}
    value = values.get(name, 0)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _pager_counters(pager: Qwen38WeightPager) -> dict[str, int]:
    values = pager.metrics()
    names = (
        "tensor_reads",
        "row_reads",
        "linear_calls",
        "embedding_rows",
        "head_rows",
        "logical_weight_bytes",
        "materialized_tensor_bytes",
        "materialized_weight_releases",
        "release_boundaries",
        "causal_tensor_read_calls",
        "causal_tensor_requested_bytes",
        "causal_tensor_source_requests",
        "causal_tensor_source_bytes",
    )
    return {
        name: int(values.get(name, 0))
        for name in names
        if isinstance(values.get(name, 0), int)
        and not isinstance(values.get(name, 0), bool)
    }


def _counter_delta(
    before: Mapping[str, int], after: Mapping[str, int]
) -> dict[str, int]:
    names = sorted(set(before) | set(after))
    result = {
        name: int(after.get(name, 0)) - int(before.get(name, 0)) for name in names
    }
    if any(value < 0 for value in result.values()):
        raise LiveK2SmokeError("pager counters moved backwards")
    return result


def _tensor_receipt(value: torch.Tensor | None) -> dict[str, Any] | None:
    if value is None:
        return None
    detached = value.detach().contiguous().to(device="cpu")
    raw = detached.view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(value.dtype).removeprefix("torch."),
        "nbytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(value.shape),
    }


def _state_receipt(model: StreamedQwen38) -> dict[str, Any]:
    layers: list[dict[str, Any] | None] = []
    for index, state in enumerate(model._layer_states):
        if state is None:
            layers.append(None)
            continue
        layers.append(
            {
                "kind": type(state).__qualname__,
                "layer": index,
                "tensors": {
                    name: _tensor_receipt(getattr(state, name, None))
                    for name in (
                        "key",
                        "value",
                        "crsa_log_usage",
                        "conv",
                        "recurrent",
                    )
                    if getattr(state, name, None) is not None
                },
            }
        )
    body = {
        "batch_size": model.state_batch_size,
        "cursor": model.next_position,
        "graft_history": _tensor_receipt(model._graft_history),
        "layers": layers,
        "pending_block": model._pending_block_stage is not None,
        "poisoned": model.state_poisoned,
        "state_bytes": model.state_bytes,
    }
    return {**body, "sha256": _sha256(body)}


def _validate_draft_config(config: Qwen38Config) -> None:
    expected = {
        "vocab_size": 248_320,
        "dim": 1_024,
        "intermediate_size": 3_584,
        "n_layers": 24,
        "n_heads": 8,
        "n_kv_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "tie_word_embeddings": True,
    }
    for name, wanted in expected.items():
        actual = getattr(config, name)
        if actual != wanted:
            raise LiveK2SmokeError(
                f"pinned Qwen3.5-0.8B requires {name}={wanted!r}, got {actual!r}"
            )


def _validate_profile(
    receipt: Mapping[str, Any], profile: Mapping[str, Any], label: str
) -> None:
    for name, expected in profile.items():
        if receipt.get(name) != expected:
            raise LiveK2SmokeError(
                f"{label} checkpoint profile changed at {name}: "
                f"{receipt.get(name)!r} != {expected!r}"
            )


@dataclass(slots=True)
class _OwnedModel:
    role: str
    mount: CausalWeightMount
    pager: Qwen38WeightPager
    model: StreamedQwen38
    bundle_receipt: dict[str, Any]
    preflight_receipt: dict[str, Any]
    verify_seconds: float
    preflight_seconds: float
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        failures: list[Exception] = []
        try:
            self.model.reset_state(release=True)
        except Exception as exc:  # pragma: no cover - exercised through mocks.
            failures.append(exc)
        try:
            self.pager.close()
        except Exception as exc:  # pragma: no cover
            failures.append(exc)
        try:
            self.mount.close()
        except Exception as exc:  # pragma: no cover
            failures.append(exc)
        self._closed = True
        if failures:
            raise LiveK2SmokeError(
                f"{self.role} cleanup failed: {type(failures[0]).__name__}: "
                f"{failures[0]}"
            ) from failures[0]


def _open_model(
    *,
    role: str,
    bundle: Path,
    identity: LogicalModelIdentity,
    source_budget_mb: float,
    device: str,
    compute_dtype: str,
    max_resident_bytes: int,
    max_context_tokens: int,
    attention_mode: str,
    native_evidence: list[Any],
    require_production_profile: bool,
    require_official_target: bool,
) -> _OwnedModel:
    mount: CausalWeightMount | None = None
    pager: Qwen38WeightPager | None = None
    model: StreamedQwen38 | None = None
    try:
        mount = CausalWeightMount(bundle, identity, budget_mb=source_budget_mb)
        verify_started = time.perf_counter()
        bundle_receipt = verify_qwen38_causal_mount(
            mount,
            require_official_config=(role == "target" and require_official_target),
        )
        verify_seconds = time.perf_counter() - verify_started
        bundle_receipt = {
            **bundle_receipt,
            "repo_id": identity.repo_id,
            "revision": identity.revision,
        }
        if require_production_profile:
            _validate_profile(
                bundle_receipt,
                _TARGET_PROFILE if role == "target" else _DRAFT_PROFILE,
                role,
            )
        config = Qwen38Config.from_file(
            mount.weights_root / "config.json",
            require_official=(role == "target" and require_official_target),
        )
        if role == "draft" and require_production_profile:
            _validate_draft_config(config)
        intervention = None
        observer: Callable[[Any], None] | None = None
        if role == "target" and attention_mode == "native-crsa":
            intervention = Qwen38NativeHeadCrsa(
                alpha=0.01,
                balance_alpha=1.0,
                diagonal_debit=3.0,
            )
            observer = native_evidence.append
        pager = Qwen38WeightPager(
            mount.source,
            device=device,
            compute_dtype=compute_dtype,
            max_resident_bytes=max_resident_bytes,
            close_source=False,
            require_source_identity=True,
            causal_tensor_reader=mount.tensor_reader,
        )
        model = StreamedQwen38(
            config,
            pager,
            native_head_crsa=intervention,
            native_head_crsa_observer=observer,
            max_batch_size=1,
            max_seq_len=max_context_tokens,
        )
        preflight_started = time.perf_counter()
        preflight = model.checkpoint_preflight()
        preflight_seconds = time.perf_counter() - preflight_started
        return _OwnedModel(
            role=role,
            mount=mount,
            pager=pager,
            model=model,
            bundle_receipt=bundle_receipt,
            preflight_receipt=preflight,
            verify_seconds=verify_seconds,
            preflight_seconds=preflight_seconds,
        )
    except Exception:
        if model is not None:
            try:
                model.reset_state(release=True)
            except Exception:
                pass
        if pager is not None:
            try:
                pager.close()
            except Exception:
                pass
        if mount is not None:
            mount.close()
        raise


class _TimedDraftProvider:
    """Observe drafter time and proposals without gaining target authority."""

    def __init__(self, provider: Qwen35K2DraftProvider) -> None:
        self.provider = provider
        self.proposal_seconds = 0.0
        self.reconcile_seconds = 0.0
        self.proposals: list[tuple[int, int]] = []
        self.reconciled_lengths: list[int] = []

    def __call__(self, history: tuple[int, ...], /) -> tuple[int, int]:
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


def _parity(
    expected: tuple[int, ...] | None,
    generated: tuple[int, ...],
    *,
    required_length: int,
) -> dict[str, Any]:
    if expected is not None and len(expected) != required_length:
        raise LiveK2SmokeError(
            "--expected-token-ids must contain exactly --max-new-tokens IDs"
        )
    exact = None if expected is None else expected == generated
    return {
        "exact": exact,
        "expected_token_ids": None if expected is None else list(expected),
        "provided": expected is not None,
    }


def _acceptance(rounds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    proposal_rows = [row for row in rounds if row["proposed_token_ids"]]
    proposed = sum(len(row["proposed_token_ids"]) for row in proposal_rows)
    accepted = sum(int(row["accepted_prefix_length"]) for row in proposal_rows)
    prefixes = Counter(int(row["accepted_prefix_length"]) for row in proposal_rows)
    replay = Counter(str(row["replay_kind"]) for row in rounds)
    return {
        "accepted_draft_tokens": accepted,
        "accepted_prefix_histogram": {
            str(value): int(prefixes.get(value, 0)) for value in (0, 1, 2)
        },
        "draft_tokens_proposed": proposed,
        "draft_verification_rounds": len(proposal_rows),
        "rate": 0.0 if proposed == 0 else accepted / proposed,
        "replay_histogram": dict(sorted(replay.items())),
        "target_verification_rounds": len(rounds),
        "terminal_single_rounds": len(rounds) - len(proposal_rows),
    }


def _validate_result(document: Mapping[str, Any]) -> dict[str, Any]:
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
        raise LiveK2SmokeError("result schema is invalid")
    unsealed = {key: value for key, value in document.items() if key != "sha256"}
    if document.get("sha256") != _sha256(unsealed):
        raise LiveK2SmokeError("result seal is invalid")
    contract = document.get("contract")
    prompt = document.get("prompt")
    tokens = document.get("tokens")
    target = document.get("target")
    drafter = document.get("drafter")
    parity = document.get("parity")
    acceptance = document.get("acceptance")
    if not all(
        isinstance(value, Mapping)
        for value in (contract, prompt, tokens, target, drafter, parity, acceptance)
    ):
        raise LiveK2SmokeError("result sections must be objects")
    assert isinstance(contract, Mapping)
    assert isinstance(prompt, Mapping)
    assert isinstance(tokens, Mapping)
    assert isinstance(target, Mapping)
    assert isinstance(drafter, Mapping)
    assert isinstance(parity, Mapping)
    assert isinstance(acceptance, Mapping)
    generated = tokens.get("generated_token_ids")
    prompt_ids = prompt.get("token_ids")
    rounds = tokens.get("rounds")
    requested = contract.get("max_new_tokens")
    if (
        contract.get("k") != 2
        or contract.get("terminal_k1_fallback") != "only-after-mismatch0"
        or not isinstance(requested, int)
        or requested < 2
        or requested % 2
        or not isinstance(generated, list)
        or len(generated) != requested
        or not isinstance(prompt_ids, list)
        or not prompt_ids
        or not isinstance(rounds, list)
        or len(rounds) < 1
    ):
        raise LiveK2SmokeError("K2 execution contract is inconsistent")
    emitted = [token for row in rounds for token in row.get("emitted_token_ids", ())]
    proposal_rows = [
        row for row in rounds if row.get("replay_kind") != "remaining-single"
    ]
    single_rows = [
        row for row in rounds if row.get("replay_kind") == "remaining-single"
    ]
    if (
        emitted != generated
        or not proposal_rows
        or any(len(row.get("proposed_token_ids", ())) != 2 for row in proposal_rows)
        or any(row.get("proposed_token_ids") != [] for row in single_rows)
        or len(single_rows) > 1
        or (single_rows and rounds[-1] is not single_rows[0])
    ):
        raise LiveK2SmokeError("round token trace is inconsistent")
    expected_acceptance = _acceptance(rounds)
    if dict(acceptance) != expected_acceptance:
        raise LiveK2SmokeError("acceptance accounting is inconsistent")
    final_cursor = len(prompt_ids) + len(generated)
    target_state = target.get("state")
    draft_state = drafter.get("state")
    provider = drafter.get("provider")
    if (
        not isinstance(target_state, Mapping)
        or not isinstance(draft_state, Mapping)
        or not isinstance(provider, Mapping)
        or target_state.get("cursor") != final_cursor
        or target_state.get("pending_block") is not False
        or draft_state.get("pending_block") is not False
        or target_state.get("poisoned") is not False
        or draft_state.get("poisoned") is not False
        or provider.get("pending") is not False
        or provider.get("poisoned") is not False
    ):
        raise LiveK2SmokeError("final transactional state is inconsistent")
    alignment = drafter.get("alignment")
    full_history = [*prompt_ids, *generated]
    if (
        not isinstance(alignment, Mapping)
        or draft_state.get("cursor") != alignment.get("committed_cursor")
        or alignment.get("target_cursor") != final_cursor
        or not isinstance(alignment.get("target_only_suffix_ids"), list)
        or len(alignment["target_only_suffix_ids"])
        != final_cursor - int(alignment.get("committed_cursor", -1))
        or len(alignment["target_only_suffix_ids"]) not in (0, 1)
        or bool(alignment["target_only_suffix_ids"]) != bool(single_rows)
        or alignment["target_only_suffix_ids"]
        != full_history[int(alignment.get("committed_cursor", -1)) :]
    ):
        raise LiveK2SmokeError("drafter/target cursor alignment is inconsistent")
    for state in (target_state, draft_state):
        state_body = {key: value for key, value in state.items() if key != "sha256"}
        if state.get("sha256") != _sha256(state_body):
            raise LiveK2SmokeError("continuation-state receipt seal is invalid")
    speculative = target.get("speculative_evidence")
    if (
        not isinstance(speculative, Mapping)
        or speculative.get("prompt_token_ids") != prompt_ids
        or speculative.get("generated_token_ids") != generated
        or speculative.get("rounds") != rounds
        or speculative.get("accepted_draft_tokens")
        != acceptance.get("accepted_draft_tokens")
        or speculative.get("state_bytes") != target_state.get("state_bytes")
    ):
        raise LiveK2SmokeError("target speculative receipt is inconsistent")
    prefix_histogram = acceptance.get("accepted_prefix_histogram")
    if (
        not isinstance(prefix_histogram, Mapping)
        or provider.get("accepted_prefix_0") != prefix_histogram.get("0")
        or provider.get("accepted_prefix_1") != prefix_histogram.get("1")
        or provider.get("accepted_prefix_2") != prefix_histogram.get("2")
        or provider.get("draft_calls") != acceptance.get("draft_verification_rounds")
        or provider.get("state_bytes") != draft_state.get("state_bytes")
    ):
        raise LiveK2SmokeError("drafter provider receipt is inconsistent")
    expected = parity.get("expected_token_ids")
    exact = parity.get("exact")
    if parity.get("provided") is True:
        if not isinstance(expected, list) or exact is not (expected == generated):
            raise LiveK2SmokeError("independent parity receipt is inconsistent")
    elif expected is not None or exact is not None:
        raise LiveK2SmokeError("absent independent parity must be null")
    expected_status = "mismatch" if exact is False else "positive"
    if document.get("status") != expected_status:
        raise LiveK2SmokeError("result status disagrees with parity")
    mode = contract.get("attention_mode")
    evidence = target.get("native_crsa")
    expected_native_attention = {
        "alpha": 0.01,
        "balance_alpha": 1.0,
        "diagonal_debit": 3.0,
        "free_softmax_heads": 20,
        "layer": 27,
        "selected_query_heads": [2, 8, 14, 20],
        "selected_sinkhorn_heads": 4,
    }
    if mode == "native-crsa":
        if (
            contract.get("attention") != expected_native_attention
            or not isinstance(evidence, Mapping)
            or int(evidence.get("count", 0)) < 1
        ):
            raise LiveK2SmokeError("native CRSA produced no committed evidence")
        events = evidence.get("events")
        expected_free_heads = [head for head in range(24) if head not in (2, 8, 14, 20)]
        if not isinstance(events, list):
            raise LiveK2SmokeError("native CRSA events are invalid")
        previous = 0
        for event in events:
            typed_event = dict(event) if isinstance(event, Mapping) else {}
            for name in (
                "selected_query_heads",
                "selected_kv_heads",
                "alpha_per_head",
                "argmax_changed_queries_per_head",
                "mean_l1_probability_delta_per_head",
                "free_heads",
            ):
                if isinstance(typed_event.get(name), list):
                    typed_event[name] = tuple(typed_event[name])
            try:
                NativeHeadCrsaEvidence(**typed_event)
            except (TypeError, ValueError) as exc:
                raise LiveK2SmokeError("native CRSA event receipt is invalid") from exc
            if (
                not isinstance(event, Mapping)
                or event.get("schema") != "immer.qwen3.8-native-head-crsa/v1"
                or event.get("layer") != 27
                or event.get("selected_query_heads") != [2, 8, 14, 20]
                or event.get("selected_kv_heads") != [0, 1, 2, 3]
                or event.get("free_heads") != expected_free_heads
                or event.get("alpha_per_head") != [0.01, 0.01, 0.01, 0.01]
                or event.get("identity") is not False
                or event.get("history_length_before") != previous
                or event.get("query_start") != previous
                or event.get("history_length_after") != event.get("key_length")
                or event.get("history_length_after")
                != previous + int(event.get("query_length", -1))
            ):
                raise LiveK2SmokeError(
                    "native CRSA event differs from the declared intervention"
                )
            previous = int(event["history_length_after"])
        if previous != final_cursor:
            raise LiveK2SmokeError("native CRSA event history is incomplete")
        target_layers = target_state.get("layers")
        if not isinstance(target_layers, list) or len(target_layers) <= 27:
            raise LiveK2SmokeError("native CRSA state layer is absent")
        layer = target_layers[27]
        tensors = layer.get("tensors") if isinstance(layer, Mapping) else None
        if not isinstance(tensors, Mapping) or "crsa_log_usage" not in tensors:
            raise LiveK2SmokeError("native CRSA usage-state receipt is absent")
    elif mode == "off":
        target_config = target.get("config")
        target_heads = (
            target_config.get("heads") if isinstance(target_config, Mapping) else None
        )
        if contract.get("attention") != {
            "free_softmax_heads": target_heads,
            "selected_sinkhorn_heads": 0,
        } or evidence != {"count": 0, "events": [], "sha256": _sha256([])}:
            raise LiveK2SmokeError("off mode emitted native CRSA evidence")
    else:
        raise LiveK2SmokeError("attention mode is invalid")
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("count") != len(evidence.get("events", ()))
        or evidence.get("sha256") != _sha256(evidence.get("events"))
    ):
        raise LiveK2SmokeError("native CRSA receipt is inconsistent")
    if tokens.get("token_chain_sha256") != _sha256(
        {"generated": generated, "prompt": prompt_ids}
    ):
        raise LiveK2SmokeError("token-chain receipt is inconsistent")
    rendered = Qwen38Tokenizer.render_no_thinking_prompt(
        str(prompt.get("system", "")), str(prompt.get("user", ""))
    )
    if (
        prompt.get("rendered_sha256")
        != hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    ):
        raise LiveK2SmokeError("prompt rendering receipt is inconsistent")
    tokenizer_receipt = prompt.get("tokenizer")
    if (
        not isinstance(tokenizer_receipt, Mapping)
        or tokenizer_receipt.get("kind") != "local-tokenizers-json/v1"
        or not _is_sha256(tokenizer_receipt.get("sha256"))
        or not isinstance(tokenizer_receipt.get("size_bytes"), int)
        or int(tokenizer_receipt["size_bytes"]) < 1
    ):
        raise LiveK2SmokeError("local tokenizer receipt is invalid")
    for section in (target, drafter):
        bundle = section.get("bundle")
        if (
            not isinstance(bundle, Mapping)
            or set(bundle)
            != {
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
            or bundle.get("kind") != "complete-causal-bundle/v1"
            or bundle.get("weights_layout") not in {"flat/v1", "nested/v1"}
            or not isinstance(bundle.get("graph_revision"), list)
            or len(bundle["graph_revision"]) != 2
            or not isinstance(bundle["graph_revision"][0], int)
            or bundle["graph_revision"][0] < 0
            or not _is_sha256(bundle["graph_revision"][1])
            or any(
                not _is_sha256(bundle.get(name))
                for name in (
                    "layout_fingerprint",
                    "manifest_sha256",
                    "shards_sha256",
                )
            )
            or any(
                not isinstance(bundle.get(name), int) or int(bundle[name]) < 1
                for name in ("checkpoint_bytes", "shards", "tensor_bindings")
            )
        ):
            raise LiveK2SmokeError("execution lacks a complete bundle receipt")
    pinned = contract.get("pinned_production_profiles")
    official = contract.get("official_target_config")
    if not isinstance(pinned, bool) or not isinstance(official, bool):
        raise LiveK2SmokeError("bundle profile contract is invalid")
    if pinned:
        _validate_profile(target["bundle"], _TARGET_PROFILE, "target")
        _validate_profile(drafter["bundle"], _DRAFT_PROFILE, "draft")
        if (
            target["bundle"].get("repo_id") != OFFICIAL_REPO_ID
            or target["bundle"].get("revision") != OFFICIAL_REVISION
            or drafter["bundle"].get("repo_id") != QWEN35_DRAFTER_REPO_ID
            or drafter["bundle"].get("revision") != QWEN35_DRAFTER_REVISION
            or official is not True
            or target.get("config")
            != {"heads": 24, "layers": 64, "vocab_size": 248_320}
            or drafter.get("config")
            != {"layers": 24, "tied_embeddings": True, "vocab_size": 248_320}
        ):
            raise LiveK2SmokeError("production bundle identity is invalid")
    return dict(document)


def run(
    args: argparse.Namespace,
    *,
    require_production_profile: bool = True,
    require_official_target: bool = True,
) -> tuple[dict[str, Any], Path]:
    """Authenticate both bundles, execute the live K2 path, and write one seal."""

    if not isinstance(args.prompt, str) or not args.prompt.strip():
        raise LiveK2SmokeError("--prompt must contain text")
    if not isinstance(args.system_prompt, str):
        raise LiveK2SmokeError("--system-prompt must be text")
    if (
        isinstance(args.max_new_tokens, bool)
        or not isinstance(args.max_new_tokens, int)
        or args.max_new_tokens < 2
        or args.max_new_tokens % 2
    ):
        raise LiveK2SmokeError("--max-new-tokens must be a positive even integer")
    if args.expected_token_ids is not None and len(args.expected_token_ids) != (
        args.max_new_tokens
    ):
        raise LiveK2SmokeError(
            "--expected-token-ids must contain exactly --max-new-tokens IDs"
        )
    target_path = Path(args.target_bundle).expanduser().absolute()
    draft_path = Path(args.draft_bundle).expanduser().absolute()
    if target_path == draft_path:
        raise LiveK2SmokeError("target and draft bundles must be independent")
    tokenizer_path = (
        Path(args.tokenizer_json).expanduser().absolute()
        if args.tokenizer_json
        else target_path / "tokenizer.json"
    )
    tokenizer_sha256, tokenizer_size = _file_sha256(tokenizer_path, "local tokenizer")
    tokenizer = Qwen38Tokenizer(
        tokenizer_path,
        require_official=require_official_target,
    )
    if _file_sha256(tokenizer_path, "local tokenizer")[0] != tokenizer_sha256:
        raise LiveK2SmokeError("local tokenizer changed while it was loaded")
    rendered = tokenizer.render_no_thinking_prompt(args.system_prompt, args.prompt)
    prompt = tokenizer.encode(rendered)
    if not prompt:
        raise LiveK2SmokeError("local tokenizer produced an empty prompt")
    if len(prompt) + args.max_new_tokens > args.max_context_tokens:
        raise LiveK2SmokeError("prompt plus output exceeds --max-context-tokens")

    target: _OwnedModel | None = None
    draft: _OwnedModel | None = None
    provider: Qwen35K2DraftProvider | None = None
    primary: BaseException | None = None
    result: tuple[dict[str, Any], Path] | None = None
    native_evidence: list[Any] = []
    try:
        target = _open_model(
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
        draft = _open_model(
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
            raise LiveK2SmokeError("target and drafter vocabularies differ")
        if any(token >= target.model.config.vocab_size for token in prompt):
            raise LiveK2SmokeError("prompt token lies outside checkpoint vocabulary")
        if args.expected_token_ids is not None and any(
            token >= target.model.config.vocab_size for token in args.expected_token_ids
        ):
            raise LiveK2SmokeError(
                "independent reference token lies outside checkpoint vocabulary"
            )

        provider = Qwen35K2DraftProvider(
            draft.model,
            eos_token_ids=(),
            head_block_rows=args.head_block_rows,
        )
        timed_provider = _TimedDraftProvider(provider)
        target_pager_before = _pager_counters(target.pager)
        draft_pager_before = _pager_counters(draft.pager)
        wall_started = time.perf_counter()
        generated = Qwen38K2SpeculativeDecoder(target.model, timed_provider).generate(
            [prompt],
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=(),
            head_block_rows=args.head_block_rows,
        )
        wall_seconds = time.perf_counter() - wall_started
        target_pager_after = _pager_counters(target.pager)
        draft_pager_after = _pager_counters(draft.pager)
        if len(generated.token_ids) != args.max_new_tokens:
            raise LiveK2SmokeError("speculative decoder did not complete K2 output")
        if _file_sha256(tokenizer_path, "local tokenizer")[0] != tokenizer_sha256:
            raise LiveK2SmokeError("local tokenizer changed during execution")

        round_rows = [row.to_dict() for row in generated.evidence.rounds]
        proposal_rows = [
            row for row in round_rows if row["replay_kind"] != "remaining-single"
        ]
        if len(timed_provider.proposals) != len(proposal_rows):
            raise LiveK2SmokeError("drafter proposal trace is incomplete")
        full_history = (*prompt, *generated.token_ids)
        draft_history = provider.committed_history
        if draft_history is None or full_history[: len(draft_history)] != draft_history:
            raise LiveK2SmokeError("drafter history is not a target-committed prefix")
        target_only_suffix = full_history[len(draft_history) :]
        if len(target_only_suffix) not in (0, 1) or bool(target_only_suffix) != bool(
            round_rows[-1]["replay_kind"] == "remaining-single"
        ):
            raise LiveK2SmokeError("drafter/target terminal alignment is invalid")
        target_state = _state_receipt(target.model)
        draft_state = _state_receipt(draft.model)
        native_rows = [row.to_dict() for row in native_evidence]
        if args.attention_mode == "native-crsa":
            if not native_rows:
                raise LiveK2SmokeError("native CRSA emitted no committed evidence")
            if native_rows[-1]["history_length_after"] != target.model.next_position:
                raise LiveK2SmokeError("native CRSA history differs from target cursor")
            layer = target_state["layers"][Qwen38NativeHeadCrsa().layer]
            tensors = {} if layer is None else layer.get("tensors", {})
            if "crsa_log_usage" not in tensors:
                raise LiveK2SmokeError("native CRSA usage state is absent")
        elif native_rows:
            raise LiveK2SmokeError("off mode emitted native CRSA evidence")

        parity = _parity(
            args.expected_token_ids,
            generated.token_ids,
            required_length=args.max_new_tokens,
        )
        provider_metrics = provider.metrics().to_dict()
        attention_identity = (
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
        )
        report = _seal(
            {
                "acceptance": _acceptance(round_rows),
                "contract": {
                    "attention": attention_identity,
                    "attention_mode": args.attention_mode,
                    "batch_size": 1,
                    "checkpoint_mutation": False,
                    "eos_stopping": False,
                    "k": 2,
                    "local_only": True,
                    "max_new_tokens": args.max_new_tokens,
                    "official_target_config": require_official_target,
                    "pinned_production_profiles": require_production_profile,
                    "remote_io": False,
                    "terminal_k1_fallback": "only-after-mismatch0",
                    "target_commits_only": True,
                },
                "drafter": {
                    "alignment": {
                        "committed_cursor": len(draft_history),
                        "target_cursor": len(full_history),
                        "target_only_suffix_ids": list(target_only_suffix),
                    },
                    "bundle": draft.bundle_receipt,
                    "config": {
                        "layers": draft.model.config.n_layers,
                        "tied_embeddings": draft.model.config.tie_word_embeddings,
                        "vocab_size": draft.model.config.vocab_size,
                    },
                    "pager": {
                        "counters": _counter_delta(
                            draft_pager_before, draft_pager_after
                        ),
                        "device": draft.pager.resolved_device,
                        "dtype": draft.pager.resolved_dtype,
                    },
                    "preflight": draft.preflight_receipt,
                    "provider": provider_metrics,
                    "state": draft_state,
                    "timing": {
                        "bundle_verify_seconds": draft.verify_seconds,
                        "preflight_seconds": draft.preflight_seconds,
                        "proposal_seconds": timed_provider.proposal_seconds,
                        "reconcile_seconds": timed_provider.reconcile_seconds,
                        "total_provider_seconds": (
                            timed_provider.proposal_seconds
                            + timed_provider.reconcile_seconds
                        ),
                    },
                    "trace": {
                        "proposals": [list(row) for row in timed_provider.proposals],
                        "reconciled_history_lengths": (
                            timed_provider.reconciled_lengths
                        ),
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
                        "sha256": tokenizer_sha256,
                        "size_bytes": tokenizer_size,
                    },
                    "user": args.prompt.strip(),
                },
                "schema": RESULT_SCHEMA,
                "status": "mismatch" if parity["exact"] is False else "positive",
                "target": {
                    "bundle": target.bundle_receipt,
                    "config": {
                        "heads": target.model.config.n_heads,
                        "layers": target.model.config.n_layers,
                        "vocab_size": target.model.config.vocab_size,
                    },
                    "native_crsa": {
                        "count": len(native_rows),
                        "events": native_rows,
                        "sha256": _sha256(native_rows),
                    },
                    "pager": {
                        "counters": _counter_delta(
                            target_pager_before, target_pager_after
                        ),
                        "device": target.pager.resolved_device,
                        "dtype": target.pager.resolved_dtype,
                    },
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
                },
                "tokens": {
                    "generated_text": tokenizer.decode(generated.token_ids),
                    "generated_token_ids": list(generated.token_ids),
                    "pieces": list(tokenizer.token_pieces(generated.token_ids)),
                    "rounds": round_rows,
                    "token_chain_sha256": _sha256(
                        {
                            "generated": list(generated.token_ids),
                            "prompt": list(prompt),
                        }
                    ),
                },
            }
        )
        report = _validate_result(report)
        output = _atomic_json(args.output, report)
        result = report, output
    except BaseException as exc:
        primary = exc
    finally:
        cleanup_failures: list[Exception] = []
        if provider is not None:
            try:
                provider.close()
            except Exception as exc:
                cleanup_failures.append(exc)
        for owned in (target, draft):
            if owned is not None:
                try:
                    owned.close()
                except Exception as exc:
                    cleanup_failures.append(exc)
        if cleanup_failures:
            cleanup = cleanup_failures[0]
            if primary is None:
                primary = LiveK2SmokeError(
                    f"runtime cleanup failed: {type(cleanup).__name__}: {cleanup}"
                )
            else:
                primary.add_note(
                    f"runtime cleanup also failed: {type(cleanup).__name__}: {cleanup}"
                )
    if primary is not None:
        raise primary
    if result is None:  # pragma: no cover - all branches above assign or fail.
        raise AssertionError("live K2 execution returned no result")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report, output = run(args)
    except (LiveK2SmokeError, OSError, TypeError, ValueError, KeyError) as exc:
        sys.stderr.write(f"qwen35_live_k2_smoke: error: {exc}\n")
        return 2
    print(
        json.dumps(
            {
                "acceptance_rate": report["acceptance"]["rate"],
                "attention_mode": report["contract"]["attention_mode"],
                "generated_token_ids": report["tokens"]["generated_token_ids"],
                "output": str(output),
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 1 if report["status"] == "mismatch" else 0


if __name__ == "__main__":
    raise SystemExit(main())
