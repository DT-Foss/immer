"""Native one-layer Qwen3.5 MTP drafter over the local causal Q4 bank."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import numbers
import os
from pathlib import Path
import secrets
import time
from typing import Any, Iterable, Sequence

import torch

from .config import Qwen38Config
from .draft_protocol import RollingDraftProposal
from .kernels import AttentionState, full_attention_core, rms_norm
from .pager import Qwen38WeightPager


QWEN35_MTP_DRAFT_PROVIDER_SCHEMA = "immer.qwen3.5-mtp-draft-provider/v5"
_QWEN35_MTP_DRAFT_PROVIDER_V4_SCHEMA = "immer.qwen3.5-mtp-draft-provider/v4"
_QWEN35_MTP_DRAFT_PROVIDER_V3_SCHEMA = "immer.qwen3.5-mtp-draft-provider/v3"
_QWEN35_MTP_DRAFT_PROVIDER_V2_SCHEMA = "immer.qwen3.5-mtp-draft-provider/v2"
_QWEN35_MTP_DRAFT_PROVIDER_LEGACY_SCHEMA = (
    "immer.qwen3.5-mtp-draft-provider/v1"
)
QWEN35_MTP_CALIBRATION_SCHEMA = "immer.qwen3.5-mtp-markov-calibration/v2"
_QWEN35_MTP_CALIBRATION_V1_SCHEMA = "immer.qwen3.5-mtp-markov-calibration/v1"
QWEN35_MTP_CARRY_SCHEMA = "immer.qwen3.5-mtp-attention-carry/v1"
_AGGREGATE_GAP_BUCKET = -1
_MIN_EXACT_BUCKET_OBSERVATIONS = 2
MTP_MATRIX_NAMES = (
    "mtp.fc.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
)
MTP_CONTROL_NAMES = (
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.self_attn.q_norm.weight",
    "mtp.layers.0.self_attn.k_norm.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.norm.weight",
)


class Qwen35MtpDraftError(RuntimeError):
    """The embedded MTP branch cannot produce a history-aligned proposal."""


def _owner_metric(owner: object, name: str) -> int:
    metrics = getattr(owner, "metrics", None)
    values = dict(metrics()) if callable(metrics) else {}
    value = values.get(name, 0)
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _state_bytes(state: AttentionState | None) -> int:
    if state is None:
        return 0
    return sum(
        value.numel() * value.element_size()
        for value in (state.key, state.value, state.crsa_log_usage)
        if value is not None
    )


def _clone_attention_state(state: AttentionState | None) -> AttentionState | None:
    if state is None:
        return None
    return AttentionState(
        key=state.key.detach().clone().contiguous(),
        value=state.value.detach().clone().contiguous(),
        crsa_log_usage=(
            None
            if state.crsa_log_usage is None
            else state.crsa_log_usage.detach().clone().contiguous()
        ),
    )


@dataclass(frozen=True, slots=True)
class Qwen35MtpCarry:
    """One content-bound in-memory MTP cache aligned to a target token prefix."""

    schema: str
    identity: tuple[object, ...]
    history: tuple[int, ...]
    next_position: int
    state: AttentionState | None
    last_target_hidden: torch.Tensor

    @property
    def state_bytes(self) -> int:
        return _state_bytes(self.state) + (
            self.last_target_hidden.numel()
            * self.last_target_hidden.element_size()
        )


@dataclass(frozen=True, slots=True)
class Qwen35MtpDraftMetrics:
    schema: str
    begin_calls: int
    advance_calls: int
    advanced_tokens: int
    proposal_calls: int
    reconcile_calls: int
    draft_steps: int
    head_scans: int
    proposed_tokens: int
    computed_proposal_tokens: int
    padded_proposal_tokens: int
    verified_proposal_tokens: int
    accepted_tokens: int
    rejected_tokens: int
    source_body_bytes: int
    logical_weight_bytes: int
    linear_calls: int
    seconds: float
    state_bytes: int
    pending: bool
    closed: bool
    proposal_width: int
    last_mean_confidence: float
    last_minimum_confidence: float
    calibration_states: int
    calibration_updates: int
    calibration_persistent: bool
    carried_context: bool
    cold_calibration_states: int
    carried_calibration_states: int
    cold_calibration_updates: int
    carried_calibration_updates: int
    teacher_verifications: int
    teacher_hits: int
    teacher_misses: int
    teacher_max_position: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Qwen35MtpDraftProvider:
    """Use Qwen's embedded MTP layer as the rolling speculative provider.

    The committed MTP cache contains shifted pairs ``embedding(token[t+1])``
    and target/draft hidden ``h[t]``. Each proposal owns explicit prefix states;
    reconciliation publishes only the target-accepted prefix.
    """

    target_state_isolation = "hidden-argument+shared-pager-only/v1"

    def __init__(
        self,
        config: Qwen38Config,
        pager: Qwen38WeightPager,
        *,
        proposal_width: int,
        eos_token_ids: Iterable[int] = (),
        head_block_rows: int = Qwen38WeightPager.DEFAULT_HEAD_BLOCK_ROWS,
        state_path: str | Path | None = None,
        initial_carry: Qwen35MtpCarry | None = None,
    ) -> None:
        if not isinstance(config, Qwen38Config):
            raise TypeError("config must be Qwen38Config")
        if not isinstance(pager, Qwen38WeightPager):
            raise TypeError("pager must be Qwen38WeightPager")
        if config.mtp_num_hidden_layers != 1 or config.mtp_use_dedicated_embeddings:
            raise ValueError("Qwen MTP must be one-layer with shared embeddings")
        if (
            isinstance(proposal_width, bool)
            or not isinstance(proposal_width, int)
            or not 1 <= proposal_width < 16
        ):
            raise ValueError("proposal_width must lie in [1, 15]")
        if (
            isinstance(head_block_rows, bool)
            or not isinstance(head_block_rows, int)
            or head_block_rows <= 0
        ):
            raise ValueError("head_block_rows must be positive")
        bank = pager.q4_bank
        if bank is None or any(not bank.has(name) for name in MTP_MATRIX_NAMES):
            raise ValueError("local Q4 bank does not contain the embedded MTP branch")
        if isinstance(eos_token_ids, (str, bytes)):
            raise TypeError("eos_token_ids must be an integer iterable")
        eos: set[int] = set()
        for raw in eos_token_ids:
            if isinstance(raw, bool) or not isinstance(raw, numbers.Integral):
                raise TypeError("eos_token_ids must contain integers")
            token = int(raw)
            if not 0 <= token < config.vocab_size:
                raise ValueError("EOS token is outside the MTP vocabulary")
            eos.add(token)

        self.config = config
        self.pager = pager
        self.proposal_width = proposal_width
        self.eos_token_ids = frozenset(eos)
        self.head_block_rows = head_block_rows
        self.state_path = (
            None if state_path is None else Path(state_path).expanduser().absolute()
        )
        self._source_start = _owner_metric(pager.source, "network_or_source_body_bytes")
        self._controls = {
            name: pager.tensor_torch(name, dtype=torch.float32, device=pager.device)
            for name in MTP_CONTROL_NAMES
        }
        self._committed_history: tuple[int, ...] | None = None
        self._committed_state: AttentionState | None = None
        self._last_target_hidden: torch.Tensor | None = None
        self._last_target_hidden_history_length = 0
        self._carry_imported = False
        self._next_position = 0
        self._pending_base: tuple[int, ...] | None = None
        self._pending_proposal: tuple[int, ...] | None = None
        self._pending_states: tuple[AttentionState, ...] = ()
        self._last_confidences: tuple[float, ...] = ()
        self._pending_gap_buckets: tuple[int, ...] = ()
        self._pending_computed_width = 0
        self._pending_teacher_positions: set[int] = set()
        self._carried_context = initial_carry is not None
        self._reliability: dict[tuple[bool, int, int, int], list[int]] = {}
        self._previous_outcomes = {False: -1, True: -1}
        self._calibration_updates = 0
        self._calibration_updates_by_context = {False: 0, True: 0}
        self._adaptive_round_call = False
        self._load_calibration()
        self._closed = False
        self._begin_calls = 0
        self._advance_calls = 0
        self._advanced_tokens = 0
        self._proposal_calls = 0
        self._reconcile_calls = 0
        self._draft_steps = 0
        self._head_scans = 0
        self._proposed_tokens = 0
        self._computed_proposal_tokens = 0
        self._padded_proposal_tokens = 0
        self._verified_proposal_tokens = 0
        self._accepted_tokens = 0
        self._rejected_tokens = 0
        self._teacher_verifications = 0
        self._teacher_hits = 0
        self._teacher_misses = 0
        self._teacher_max_position = 0
        self._linear_calls = 0
        self._source_body_bytes = max(
            0,
            _owner_metric(pager.source, "network_or_source_body_bytes")
            - self._source_start,
        )
        self._logical_weight_bytes = 0
        self._seconds = 0.0
        if initial_carry is not None:
            self._restore_carry(initial_carry)

    def _calibration_identity(self) -> dict[str, object]:
        return {
            "dim": self.config.dim,
            "provider": QWEN35_MTP_DRAFT_PROVIDER_SCHEMA,
            "q4_manifest_sha256": self.pager.q4_bank.identity["manifest_sha256"],
            "vocab_size": self.config.vocab_size,
        }

    @property
    def _previous_outcome(self) -> int:
        return self._previous_outcomes[self._carried_context]

    @_previous_outcome.setter
    def _previous_outcome(self, value: int) -> None:
        self._previous_outcomes[self._carried_context] = value

    @staticmethod
    def _ensure_carried_aggregates(
        rows: dict[tuple[bool, int, int, int], list[int]],
    ) -> None:
        grouped: dict[tuple[int, int], list[int]] = {}
        for (carried, position, bucket, previous), counts in rows.items():
            if not carried or bucket == _AGGREGATE_GAP_BUCKET:
                continue
            aggregate = grouped.setdefault((position, previous), [1, 1])
            aggregate[0] += counts[0] - 1
            aggregate[1] += counts[1] - 1
        for (position, previous), counts in grouped.items():
            rows.setdefault(
                (True, position, _AGGREGATE_GAP_BUCKET, previous),
                counts,
            )

    def _reliability_counts(
        self,
        position: int,
        bucket: int,
        previous: int,
    ) -> list[int] | None:
        exact = self._reliability.get(
            (self._carried_context, position, bucket, previous)
        )
        if not self._carried_context:
            if exact is None and position > 0:
                exact = self._reliability.get((False, 0, bucket, previous))
            return exact
        if exact is not None and sum(exact) - 2 >= _MIN_EXACT_BUCKET_OBSERVATIONS:
            return exact
        aggregate = self._reliability.get(
            (True, position, _AGGREGATE_GAP_BUCKET, previous)
        )
        if aggregate is not None:
            return aggregate
        if position > 0:
            base_exact = self._reliability.get((True, 0, bucket, previous))
            if (
                base_exact is not None
                and sum(base_exact) - 2 >= _MIN_EXACT_BUCKET_OBSERVATIONS
            ):
                return base_exact
            base_aggregate = self._reliability.get(
                (True, 0, _AGGREGATE_GAP_BUCKET, previous)
            )
            if base_aggregate is not None:
                return base_aggregate
            if base_exact is not None:
                return base_exact
        return exact

    def _carry_identity(self) -> tuple[object, ...]:
        source_metrics_callback = getattr(self.pager.source, "metrics", None)
        source_metrics = (
            dict(source_metrics_callback())
            if callable(source_metrics_callback)
            else {}
        )
        return (
            QWEN35_MTP_DRAFT_PROVIDER_SCHEMA,
            self.config.dim,
            self.config.intermediate_size,
            self.config.vocab_size,
            self.config.n_heads,
            self.config.n_kv_heads,
            self.config.head_dim,
            self.config.rotary_dim,
            self.config.partial_rotary_factor,
            self.config.rope_theta,
            self.config.mrope_interleaved,
            self.config.mrope_section,
            self.config.rms_norm_eps,
            self.config.hidden_act,
            self.pager.q4_bank.identity["manifest_sha256"],
            getattr(self.pager.source, "repo_id", None),
            getattr(self.pager.source, "revision", None),
            source_metrics.get("inventory_source_fingerprint"),
            str(self.pager.device),
            str(self.pager.compute_dtype),
        )

    def _restore_carry(self, carry: Qwen35MtpCarry) -> None:
        if not isinstance(carry, Qwen35MtpCarry):
            raise TypeError("initial_carry must be a Qwen35MtpCarry")
        if (
            carry.schema != QWEN35_MTP_CARRY_SCHEMA
            or carry.identity != self._carry_identity()
        ):
            raise ValueError("MTP carry identity or cursor is invalid")
        history = self._history(carry.history, label="MTP carry history")
        if carry.next_position != len(history) - 1:
            raise ValueError("MTP carry identity or cursor is invalid")
        hidden = self._hidden(
            carry.last_target_hidden,
            rows=1,
            label="MTP carry target hidden",
        )
        state = _clone_attention_state(carry.state)
        if len(history) > 1 and state is None:
            raise ValueError("MTP carry lacks its committed attention state")
        if state is not None and (
            tuple(state.key.shape)
            != (
                1,
                self.config.n_kv_heads,
                len(history) - 1,
                self.config.head_dim,
            )
            or state.key.device != self.pager.device
            or state.key.dtype != self.pager.compute_dtype
            or state.crsa_log_usage is not None
            or not bool(torch.isfinite(state.key).all().item())
            or not bool(torch.isfinite(state.value).all().item())
        ):
            raise ValueError("MTP carry attention tensor contract is invalid")
        self._committed_history = history
        self._committed_state = state
        self._last_target_hidden = hidden.detach().clone().contiguous()
        self._last_target_hidden_history_length = len(history)
        self._next_position = carry.next_position
        self._carry_imported = True

    def _load_calibration(self) -> None:
        path = self.state_path
        if path is None or not path.is_file() or path.is_symlink():
            return
        try:
            document = json.loads(path.read_bytes())
            identity = self._calibration_identity()
            v4_identity = {
                **identity,
                "provider": _QWEN35_MTP_DRAFT_PROVIDER_V4_SCHEMA,
            }
            v3_identity = {
                **identity,
                "provider": _QWEN35_MTP_DRAFT_PROVIDER_V3_SCHEMA,
            }
            legacy_identity = {
                **identity,
                "provider": _QWEN35_MTP_DRAFT_PROVIDER_LEGACY_SCHEMA,
            }
            v2_identity = {
                **identity,
                "provider": _QWEN35_MTP_DRAFT_PROVIDER_V2_SCHEMA,
            }
            if (
                not isinstance(document, dict)
                or document.get("schema")
                not in (
                    QWEN35_MTP_CALIBRATION_SCHEMA,
                    _QWEN35_MTP_CALIBRATION_V1_SCHEMA,
                )
                or document.get("identity")
                not in (
                    identity,
                    v4_identity,
                    v3_identity,
                    v2_identity,
                    legacy_identity,
                )
                or not isinstance(document.get("rows"), list)
                or isinstance(document.get("updates"), bool)
                or not isinstance(document.get("updates"), int)
                or document["updates"] < 0
            ):
                return
            schema = document["schema"]
            if schema == _QWEN35_MTP_CALIBRATION_V1_SCHEMA:
                if document.get("previous_outcome") not in {-1, 0, 1}:
                    return
                previous_outcomes = {
                    False: document["previous_outcome"],
                    True: -1,
                }
                updates_by_context = {False: document["updates"], True: 0}
            else:
                raw_previous = document.get("previous_outcomes")
                raw_updates = document.get("updates_by_context")
                if (
                    not isinstance(raw_previous, dict)
                    or set(raw_previous) != {"carried", "cold"}
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value not in {-1, 0, 1}
                        for value in raw_previous.values()
                    )
                    or not isinstance(raw_updates, dict)
                    or set(raw_updates) != {"carried", "cold"}
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        for value in raw_updates.values()
                    )
                    or sum(raw_updates.values()) != document["updates"]
                ):
                    return
                previous_outcomes = {
                    False: raw_previous["cold"],
                    True: raw_previous["carried"],
                }
                updates_by_context = {
                    False: raw_updates["cold"],
                    True: raw_updates["carried"],
                }
            restored: dict[tuple[bool, int, int, int], list[int]] = {}
            for row in document["rows"]:
                expected_keys = {
                    "alpha",
                    "beta",
                    "bucket",
                    "position",
                    "previous",
                }
                if schema == QWEN35_MTP_CALIBRATION_SCHEMA:
                    expected_keys.add("carried")
                if not isinstance(row, dict) or set(row) != expected_keys:
                    return
                carried = row.get("carried", False)
                if not isinstance(carried, bool):
                    return
                values = tuple(
                    row[key]
                    for key in (
                        "position",
                        "bucket",
                        "previous",
                        "alpha",
                        "beta",
                    )
                )
                if any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in values
                ):
                    return
                position, bucket, previous, alpha, beta = values
                if (
                    not 0 <= position < 15
                    or not (
                        0 <= bucket <= 6
                        or (
                            schema == QWEN35_MTP_CALIBRATION_SCHEMA
                            and carried
                            and bucket == _AGGREGATE_GAP_BUCKET
                        )
                    )
                    or previous not in {-1, 0, 1}
                    or alpha < 1
                    or beta < 1
                ):
                    return
                restored[(carried, position, bucket, previous)] = [alpha, beta]
            self._ensure_carried_aggregates(restored)
            self._reliability = restored
            self._previous_outcomes = previous_outcomes
            self._calibration_updates = document["updates"]
            self._calibration_updates_by_context = updates_by_context
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return

    def _save_calibration(self) -> None:
        path = self.state_path
        if path is None:
            return
        document = {
            "identity": self._calibration_identity(),
            "previous_outcomes": {
                "carried": self._previous_outcomes[True],
                "cold": self._previous_outcomes[False],
            },
            "rows": [
                {
                    "alpha": counts[0],
                    "beta": counts[1],
                    "bucket": key[2],
                    "carried": key[0],
                    "position": key[1],
                    "previous": key[3],
                }
                for key, counts in sorted(self._reliability.items())
            ],
            "schema": QWEN35_MTP_CALIBRATION_SCHEMA,
            "updates": self._calibration_updates,
            "updates_by_context": {
                "carried": self._calibration_updates_by_context[True],
                "cold": self._calibration_updates_by_context[False],
            },
        }
        data = json.dumps(
            document,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.{secrets.token_hex(8)}.tmp"
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | int(getattr(os, "O_CLOEXEC", 0)),
                0o600,
            )
            offset = 0
            while offset < len(data):
                written = os.write(descriptor, data[offset:])
                if written <= 0:
                    raise OSError("short MTP calibration write")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = None
            os.replace(temporary, path)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def _history(self, value: object, *, label: str) -> tuple[int, ...]:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"{label} must be an integer sequence")
        result = tuple(value)
        if not result or any(
            isinstance(token, bool)
            or not isinstance(token, numbers.Integral)
            or not 0 <= int(token) < self.config.vocab_size
            for token in result
        ):
            raise ValueError(f"{label} contains an invalid token")
        return tuple(int(token) for token in result)

    def _hidden(
        self,
        value: object,
        *,
        rows: int,
        label: str,
    ) -> torch.Tensor:
        if (
            not isinstance(value, torch.Tensor)
            or not value.is_floating_point()
            or tuple(value.shape) != (1, rows, self.config.dim)
            or value.device != self.pager.device
            or value.dtype != self.pager.compute_dtype
            or not bool(torch.isfinite(value).all().item())
        ):
            raise ValueError(f"{label} differs from target hidden-state execution")
        return value.detach()

    def _norm(self, value: torch.Tensor, name: str) -> torch.Tensor:
        return rms_norm(
            value,
            self._controls[name],
            eps=self.config.rms_norm_eps,
        )

    def _step(
        self,
        token_ids: tuple[int, ...],
        conditioning_hidden: torch.Tensor,
        *,
        state: AttentionState | None,
        start_pos: int,
    ) -> tuple[torch.Tensor, AttentionState]:
        if not token_ids:
            raise ValueError("MTP step requires at least one token")
        rows = len(token_ids)
        hidden = self._hidden(
            conditioning_hidden,
            rows=rows,
            label="MTP conditioning hidden",
        )
        started = time.perf_counter()
        linears_before = _owner_metric(self.pager, "linear_calls")
        source_before = _owner_metric(self.pager.source, "network_or_source_body_bytes")
        logical_before = _owner_metric(self.pager.q4_bank, "logical_weight_bytes")
        try:
            embedding = self.pager.embedding(
                token_ids,
                name="model.language_model.embed_tokens.weight",
            ).reshape(1, rows, self.config.dim)
            embedding = self._norm(embedding, "mtp.pre_fc_norm_embedding.weight")
            conditioned = self._norm(hidden, "mtp.pre_fc_norm_hidden.weight")
            fused = self.pager.linear(
                torch.cat((embedding, conditioned), dim=-1),
                "mtp.fc.weight",
            )

            residual = fused
            normalized = self._norm(
                fused,
                "mtp.layers.0.input_layernorm.weight",
            )
            base = "mtp.layers.0.self_attn"
            projected_query_gate, projected_key, projected_value = (
                self.pager.linear_group(
                    normalized,
                    (f"{base}.q_proj", f"{base}.k_proj", f"{base}.v_proj"),
                )
            )
            positions = torch.arange(
                start_pos,
                start_pos + rows,
                device=self.pager.device,
                dtype=torch.long,
            ).reshape(1, -1)
            mixed, next_state = full_attention_core(
                projected_query_gate,
                projected_key,
                projected_value,
                q_norm_weight=self._controls[f"{base}.q_norm.weight"],
                k_norm_weight=self._controls[f"{base}.k_norm.weight"],
                num_attention_heads=self.config.n_heads,
                num_key_value_heads=self.config.n_kv_heads,
                head_dim=self.config.head_dim,
                position_ids=positions,
                state=state,
                attention_mask=None,
                rope_theta=self.config.rope_theta,
                rotary_dim=self.config.rotary_dim,
                partial_rotary_factor=self.config.partial_rotary_factor,
                mrope_section=self.config.mrope_section,
                mrope_interleaved=self.config.mrope_interleaved,
                rms_norm_eps=self.config.rms_norm_eps,
            )
            hidden = residual + self.pager.linear(mixed, f"{base}.o_proj")

            residual = hidden
            normalized = self._norm(
                hidden,
                "mtp.layers.0.post_attention_layernorm.weight",
            )
            mlp = "mtp.layers.0.mlp"
            hidden = residual + self.pager.mlp(
                normalized,
                (
                    f"{mlp}.gate_proj",
                    f"{mlp}.up_proj",
                    f"{mlp}.down_proj",
                ),
            )
            hidden = self._norm(hidden, "mtp.norm.weight")
            self._draft_steps += rows
            return hidden, next_state
        finally:
            self._linear_calls += max(
                0,
                _owner_metric(self.pager, "linear_calls") - linears_before,
            )
            self._source_body_bytes += max(
                0,
                _owner_metric(
                    self.pager.source,
                    "network_or_source_body_bytes",
                )
                - source_before,
            )
            self._logical_weight_bytes += max(
                0,
                _owner_metric(self.pager.q4_bank, "logical_weight_bytes")
                - logical_before,
            )
            self._seconds += time.perf_counter() - started
            self.pager.release()

    @staticmethod
    def _gap_bucket(gap: float) -> int:
        for index, ceiling in enumerate((0.5, 1.0, 2.0, 4.0, 8.0, 16.0)):
            if gap < ceiling:
                return index
        return 6

    def _scan(
        self,
        hidden: torch.Tensor,
        *,
        proposal_index: int,
    ) -> tuple[int, float, int]:
        started = time.perf_counter()
        logical_before = _owner_metric(self.pager.q4_bank, "logical_weight_bytes")
        values, selected = self.pager.topk_logits(
            hidden[:, -1],
            k=2,
            name="lm_head.weight",
            block_rows=self.head_block_rows,
        )
        self._seconds += time.perf_counter() - started
        self._logical_weight_bytes += max(
            0,
            _owner_metric(self.pager.q4_bank, "logical_weight_bytes") - logical_before,
        )
        if tuple(selected.shape) != (1, 2) or tuple(values.shape) != (1, 2):
            raise Qwen35MtpDraftError("MTP LM head returned an invalid token")
        token = int(selected[0, 0].item())
        gap = max(0.0, float(values[0, 0].item()) - float(values[0, 1].item()))
        raw_confidence = max(0.0, min(0.999, 1.0 - math.exp(-gap)))
        bucket = self._gap_bucket(gap)
        previous = self._previous_outcome if proposal_index == 0 else 1
        exact = self._reliability_counts(proposal_index, bucket, previous)
        alpha, beta = [1, 1] if exact is None else exact
        confidence = min(raw_confidence, alpha / (alpha + beta))
        del values, selected
        if not 0 <= token < self.config.vocab_size:
            raise Qwen35MtpDraftError("MTP selected a token outside the vocabulary")
        self._head_scans += 1
        return token, confidence, bucket

    def begin_request_state(
        self,
        history: tuple[int, ...],
        target_hidden: torch.Tensor,
        /,
    ) -> None:
        if self._closed:
            raise Qwen35MtpDraftError("MTP provider is closed")
        committed = self._history(history, label="MTP request history")
        if self._carry_imported:
            base = self._committed_history
            previous = self._last_target_hidden
            if (
                base is None
                or previous is None
                or len(committed) <= len(base)
                or committed[: len(base)] != base
            ):
                raise Qwen35MtpDraftError(
                    "MTP carry history is not a strict request prefix"
                )
            added = len(committed) - len(base)
            hidden = self._hidden(
                target_hidden,
                rows=added,
                label="MTP restored request hidden",
            )
            self._carry_imported = False
            self.advance_confirmed_prefix_state(
                committed,
                previous.detach().clone(),
                hidden,
            )
            self._begin_calls += 1
            return
        hidden = self._hidden(
            target_hidden,
            rows=len(committed),
            label="MTP request hidden",
        )
        if self._committed_history is not None or self._pending_base is not None:
            raise Qwen35MtpDraftError("MTP request state was already initialized")
        if len(committed) > 1:
            _output, state = self._step(
                committed[1:],
                hidden[:, :-1],
                state=None,
                start_pos=0,
            )
            self._committed_state = state
        self._committed_history = committed
        self._last_target_hidden = hidden[:, -1:].detach().clone().contiguous()
        self._last_target_hidden_history_length = len(committed)
        self._next_position = len(committed) - 1
        self._begin_calls += 1

    def __call__(self, _history: tuple[int, ...], /) -> tuple[int, ...]:
        raise Qwen35MtpDraftError(
            "embedded MTP requires target-hidden rolling callbacks"
        )

    def advance_confirmed_prefix_state(
        self,
        history: tuple[int, ...],
        previous_target_hidden: torch.Tensor,
        committed_hidden: torch.Tensor,
        /,
    ) -> None:
        """Append a target-confirmed prefix without replaying the MTP cache.

        For an extension ``x[0:m]``, the exact shifted MTP pairs are
        ``(x[0], h[-1])`` followed by ``(x[i], h[i-1])``.  The caller supplies
        the previous committed target row ``h[-1]`` and the target rows for the
        extension; the final row remains the conditioner for the next token.
        """

        if self._closed:
            raise Qwen35MtpDraftError("MTP provider is closed")
        base = self._committed_history
        if base is None:
            raise Qwen35MtpDraftError("MTP request state is not initialized")
        if self._pending_base is not None:
            raise Qwen35MtpDraftError(
                "cannot advance MTP state with a pending proposal"
            )
        committed = self._history(history, label="MTP confirmed history")
        if len(committed) <= len(base) or committed[: len(base)] != base:
            raise Qwen35MtpDraftError(
                "MTP confirmed history is not a contiguous extension"
            )
        added = len(committed) - len(base)
        previous = self._hidden(
            previous_target_hidden,
            rows=1,
            label="MTP previous target hidden",
        )
        fragment = self._hidden(
            committed_hidden,
            rows=added,
            label="MTP committed target hidden",
        )
        conditioning = torch.cat((previous, fragment[:, :-1]), dim=1)
        _output, state = self._step(
            committed[len(base) :],
            conditioning,
            state=self._committed_state,
            start_pos=self._next_position,
        )
        self._committed_history = committed
        self._committed_state = state
        self._last_target_hidden = fragment[:, -1:].detach().clone().contiguous()
        self._last_target_hidden_history_length = len(committed)
        self._next_position = len(committed) - 1
        self._advance_calls += 1
        self._advanced_tokens += added

    def propose_after_state(
        self,
        history: tuple[int, ...],
        known_token: int,
        target_hidden: torch.Tensor,
        /,
    ) -> tuple[int, ...]:
        if self._closed:
            raise Qwen35MtpDraftError("MTP provider is closed")
        committed = self._history(history, label="MTP proposal history")
        if committed != self._committed_history:
            raise Qwen35MtpDraftError("MTP proposal history is not committed")
        if self._pending_base is not None:
            raise Qwen35MtpDraftError("previous MTP proposal was not reconciled")
        if isinstance(known_token, bool) or not isinstance(
            known_token, numbers.Integral
        ):
            raise TypeError("known MTP token must be an integer")
        known = int(known_token)
        if not 0 <= known < self.config.vocab_size:
            raise ValueError("known MTP token is outside the vocabulary")
        if known in self.eos_token_ids:
            raise ValueError("cannot propose after MTP EOS")
        conditioned = self._hidden(
            target_hidden,
            rows=1,
            label="MTP proposal hidden",
        )

        output, state = self._step(
            (known,),
            conditioned,
            state=self._committed_state,
            start_pos=self._next_position,
        )
        states = [state]
        proposal: list[int] = []
        confidences: list[float] = []
        gap_buckets: list[int] = []
        computed_width = 0
        for proposal_index in range(self.proposal_width):
            token, confidence, bucket = self._scan(
                output,
                proposal_index=proposal_index,
            )
            self._computed_proposal_tokens += 1
            computed_width += 1
            proposal.append(token)
            confidences.append(confidence)
            gap_buckets.append(bucket)
            if self._adaptive_round_call and confidence < 0.6:
                missing = self.proposal_width - len(proposal)
                self._padded_proposal_tokens += missing
                proposal.extend([token] * missing)
                confidences.extend([0.0] * missing)
                gap_buckets.extend([bucket] * missing)
                states.extend([state] * (self.proposal_width + 1 - len(states)))
                break
            if token in self.eos_token_ids:
                missing = self.proposal_width - len(proposal)
                self._padded_proposal_tokens += missing
                proposal.extend([token] * missing)
                confidences.extend(
                    [confidence] * (self.proposal_width - len(confidences))
                )
                gap_buckets.extend([bucket] * (self.proposal_width - len(gap_buckets)))
                states.extend([state] * (self.proposal_width + 1 - len(states)))
                break
            output, state = self._step(
                (token,),
                output[:, -1:],
                state=state,
                start_pos=self._next_position + len(proposal),
            )
            states.append(state)

        result = tuple(proposal)
        if len(result) != self.proposal_width or len(states) != self.proposal_width + 1:
            raise Qwen35MtpDraftError("MTP proposal/state width is incomplete")
        self._pending_base = (*committed, known)
        self._pending_proposal = result
        self._pending_states = tuple(states)
        self._last_confidences = tuple(confidences)
        self._pending_gap_buckets = tuple(gap_buckets)
        self._pending_computed_width = computed_width
        self._pending_teacher_positions.clear()
        self._proposal_calls += 1
        self._proposed_tokens += len(result)
        return result

    def propose_round_state(
        self,
        history: tuple[int, ...],
        known_token: int,
        target_hidden: torch.Tensor,
        /,
    ) -> RollingDraftProposal:
        self._adaptive_round_call = True
        try:
            proposal = self.propose_after_state(history, known_token, target_hidden)
        finally:
            self._adaptive_round_call = False
        return RollingDraftProposal.build(
            proposal,
            self._last_confidences,
            (0.0,) * len(proposal),
            request_window_ceiling=self.proposal_width + 1,
            provider_abi=QWEN35_MTP_DRAFT_PROVIDER_SCHEMA,
        )

    def _update_calibration(
        self,
        *,
        position: int,
        bucket: int,
        previous: int,
        outcome: bool,
    ) -> None:
        key = (self._carried_context, position, bucket, previous)
        alpha, beta = self._reliability.setdefault(key, [1, 1])
        if outcome:
            alpha += 1
        else:
            beta += 1
        self._reliability[key] = [alpha, beta]
        if self._carried_context:
            aggregate_key = (
                True,
                position,
                _AGGREGATE_GAP_BUCKET,
                previous,
            )
            aggregate_alpha, aggregate_beta = self._reliability.setdefault(
                aggregate_key,
                [1, 1],
            )
            if outcome:
                aggregate_alpha += 1
            else:
                aggregate_beta += 1
            self._reliability[aggregate_key] = [
                aggregate_alpha,
                aggregate_beta,
            ]
        self._calibration_updates += 1
        self._calibration_updates_by_context[self._carried_context] += 1

    def observe_verification(
        self,
        accepted_prefix_length: int,
        verified_proposals: int,
        /,
    ) -> None:
        proposal = self._pending_proposal
        buckets = self._pending_gap_buckets
        if proposal is None or len(buckets) != len(proposal):
            raise Qwen35MtpDraftError("MTP verification has no pending proposal")
        if (
            isinstance(accepted_prefix_length, bool)
            or not isinstance(accepted_prefix_length, int)
            or isinstance(verified_proposals, bool)
            or not isinstance(verified_proposals, int)
            or not 0 <= accepted_prefix_length <= verified_proposals <= len(proposal)
        ):
            raise ValueError("MTP verification prefix is invalid")
        self._verified_proposal_tokens += verified_proposals
        self._rejected_tokens += verified_proposals - accepted_prefix_length
        previous = self._previous_outcome
        for index in range(verified_proposals):
            outcome = index < accepted_prefix_length
            self._update_calibration(
                position=index,
                bucket=buckets[index],
                previous=previous,
                outcome=outcome,
            )
            previous = int(outcome)
            if not outcome:
                break
        self._previous_outcome = previous
        try:
            self._save_calibration()
        except OSError:
            pass

    def observe_teacher_verification(
        self,
        position: int,
        outcome: bool,
        /,
    ) -> None:
        if self._closed:
            raise Qwen35MtpDraftError("MTP provider is closed")
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position < 1
            or not isinstance(outcome, bool)
        ):
            raise ValueError("MTP teacher verification is invalid")
        if not self._pending_gap_buckets:
            raise Qwen35MtpDraftError("MTP teacher verification has no proposal")
        if position >= len(self._pending_gap_buckets):
            raise ValueError("MTP teacher verification is invalid")
        if position >= self._pending_computed_width:
            return
        if position in self._pending_teacher_positions:
            raise Qwen35MtpDraftError(
                "MTP teacher verification position was already observed"
            )
        self._update_calibration(
            position=position,
            bucket=self._pending_gap_buckets[position],
            previous=1,
            outcome=outcome,
        )
        self._teacher_verifications += 1
        self._teacher_hits += int(outcome)
        self._teacher_misses += int(not outcome)
        self._teacher_max_position = max(self._teacher_max_position, position)
        self._pending_teacher_positions.add(position)
        self._previous_outcome = int(outcome)
        try:
            self._save_calibration()
        except OSError:
            pass

    def observe_virtual_verification(
        self,
        accepted_prefix_length: int,
        verified_proposals: int,
        /,
    ) -> None:
        """Calibrate one target-checked K1 draft that remains uncommitted."""

        if verified_proposals > 1:
            raise ValueError("virtual MTP verification covers at most one proposal")
        self.observe_verification(accepted_prefix_length, verified_proposals)

    def reconcile_prefix(self, history: tuple[int, ...], /) -> None:
        committed = self._history(history, label="MTP reconciled history")
        base = self._pending_base
        proposal = self._pending_proposal
        if base is None or proposal is None or not self._pending_states:
            raise Qwen35MtpDraftError("MTP reconciliation has no pending proposal")
        if committed[: len(base)] != base:
            raise Qwen35MtpDraftError("MTP reconciliation changed its known base")
        delta = committed[len(base) :]
        if len(delta) > len(proposal) or delta != proposal[: len(delta)]:
            raise Qwen35MtpDraftError("MTP reconciliation is not a proposal prefix")
        accepted = len(delta)
        self._committed_history = committed
        self._committed_state = self._pending_states[accepted]
        self._next_position = len(committed) - 1
        self._accepted_tokens += accepted
        self._pending_base = None
        self._pending_proposal = None
        self._pending_states = ()
        self._pending_gap_buckets = ()
        self._pending_computed_width = 0
        self._pending_teacher_positions.clear()
        self._reconcile_calls += 1

    def reconcile_prefix_state(
        self,
        history: tuple[int, ...],
        committed_hidden: torch.Tensor,
        /,
    ) -> None:
        base = self._committed_history
        if base is None:
            raise Qwen35MtpDraftError("MTP request state is not initialized")
        committed = self._history(history, label="MTP reconciled history")
        added = len(committed) - len(base)
        fragment = self._hidden(
            committed_hidden,
            rows=added,
            label="MTP reconciled target hidden",
        )
        self.reconcile_prefix(committed)
        self._last_target_hidden = fragment[:, -1:].detach().clone().contiguous()
        self._last_target_hidden_history_length = len(committed)

    def export_carry(
        self,
        history: tuple[int, ...],
        /,
    ) -> Qwen35MtpCarry:
        if self._closed:
            raise Qwen35MtpDraftError("cannot export carry from a closed provider")
        if (
            self._pending_base is not None
            or self._pending_proposal is not None
            or self._pending_states
            or self._pending_gap_buckets
            or self._pending_teacher_positions
            or self._adaptive_round_call
        ):
            raise Qwen35MtpDraftError("cannot export carry with a pending proposal")
        committed = self._committed_history
        state = self._committed_state
        last_target_hidden = self._last_target_hidden
        expected = self._history(history, label="MTP carry history")
        if (
            committed is None
            or len(expected) > len(committed)
            or committed[: len(expected)] != expected
            or last_target_hidden is None
            or self._last_target_hidden_history_length != len(expected)
        ):
            raise Qwen35MtpDraftError(
                "MTP carry history and target-hidden cursor are not aligned"
            )
        hidden = self._hidden(
            last_target_hidden,
            rows=1,
            label="MTP carry target hidden",
        )
        state_length = len(expected) - 1
        if state_length == 0:
            exported_state = None
        else:
            if state is None or state.length < state_length:
                raise Qwen35MtpDraftError("MTP carry attention state is too short")
            exported_state = AttentionState(
                key=state.key[:, :, :state_length].detach().clone().contiguous(),
                value=state.value[:, :, :state_length].detach().clone().contiguous(),
                crsa_log_usage=(
                    None
                    if state.crsa_log_usage is None
                    else state.crsa_log_usage[:, :, :state_length]
                    .detach()
                    .clone()
                    .contiguous()
                ),
            )
        return Qwen35MtpCarry(
            schema=QWEN35_MTP_CARRY_SCHEMA,
            identity=self._carry_identity(),
            history=expected,
            next_position=len(expected) - 1,
            state=exported_state,
            last_target_hidden=hidden.detach().clone().contiguous(),
        )

    def observe_final(self, _history: tuple[int, ...], /) -> None:
        return None

    def metrics(self) -> Qwen35MtpDraftMetrics:
        return Qwen35MtpDraftMetrics(
            schema=QWEN35_MTP_DRAFT_PROVIDER_SCHEMA,
            begin_calls=self._begin_calls,
            advance_calls=self._advance_calls,
            advanced_tokens=self._advanced_tokens,
            proposal_calls=self._proposal_calls,
            reconcile_calls=self._reconcile_calls,
            draft_steps=self._draft_steps,
            head_scans=self._head_scans,
            proposed_tokens=self._proposed_tokens,
            computed_proposal_tokens=self._computed_proposal_tokens,
            padded_proposal_tokens=self._padded_proposal_tokens,
            verified_proposal_tokens=self._verified_proposal_tokens,
            accepted_tokens=self._accepted_tokens,
            rejected_tokens=self._rejected_tokens,
            source_body_bytes=self._source_body_bytes,
            logical_weight_bytes=self._logical_weight_bytes,
            linear_calls=self._linear_calls,
            seconds=self._seconds,
            state_bytes=_state_bytes(self._committed_state),
            pending=self._pending_base is not None,
            closed=self._closed,
            proposal_width=self.proposal_width,
            last_mean_confidence=(
                0.0
                if not self._last_confidences
                else sum(self._last_confidences) / len(self._last_confidences)
            ),
            last_minimum_confidence=(
                0.0 if not self._last_confidences else min(self._last_confidences)
            ),
            calibration_states=len(self._reliability),
            calibration_updates=self._calibration_updates,
            calibration_persistent=self.state_path is not None,
            carried_context=self._carried_context,
            cold_calibration_states=sum(
                1 for key in self._reliability if key[0] is False
            ),
            carried_calibration_states=sum(
                1 for key in self._reliability if key[0] is True
            ),
            cold_calibration_updates=self._calibration_updates_by_context[False],
            carried_calibration_updates=self._calibration_updates_by_context[True],
            teacher_verifications=self._teacher_verifications,
            teacher_hits=self._teacher_hits,
            teacher_misses=self._teacher_misses,
            teacher_max_position=self._teacher_max_position,
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._save_calibration()
        except OSError:
            pass
        self._controls.clear()
        self._committed_history = None
        self._committed_state = None
        self._last_target_hidden = None
        self._last_target_hidden_history_length = 0
        self._carry_imported = False
        self._pending_base = None
        self._pending_proposal = None
        self._pending_states = ()
        self._last_confidences = ()
        self._pending_gap_buckets = ()
        self._pending_computed_width = 0
        self._pending_teacher_positions.clear()
        self._reliability.clear()
        self._closed = True
        self.pager.release()


__all__ = [
    "MTP_CONTROL_NAMES",
    "MTP_MATRIX_NAMES",
    "QWEN35_MTP_CARRY_SCHEMA",
    "QWEN35_MTP_DRAFT_PROVIDER_SCHEMA",
    "Qwen35MtpCarry",
    "Qwen35MtpDraftError",
    "Qwen35MtpDraftMetrics",
    "Qwen35MtpDraftProvider",
]
