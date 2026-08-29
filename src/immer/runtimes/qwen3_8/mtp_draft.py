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


QWEN35_MTP_DRAFT_PROVIDER_SCHEMA = "immer.qwen3.5-mtp-draft-provider/v1"
QWEN35_MTP_CALIBRATION_SCHEMA = "immer.qwen3.5-mtp-markov-calibration/v1"
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


@dataclass(frozen=True, slots=True)
class Qwen35MtpDraftMetrics:
    schema: str
    begin_calls: int
    proposal_calls: int
    reconcile_calls: int
    draft_steps: int
    head_scans: int
    proposed_tokens: int
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
        self._next_position = 0
        self._pending_base: tuple[int, ...] | None = None
        self._pending_proposal: tuple[int, ...] | None = None
        self._pending_states: tuple[AttentionState, ...] = ()
        self._last_confidences: tuple[float, ...] = ()
        self._pending_gap_buckets: tuple[int, ...] = ()
        self._reliability: dict[tuple[int, int, int], list[int]] = {}
        self._previous_outcome = -1
        self._calibration_updates = 0
        self._adaptive_round_call = False
        self._load_calibration()
        self._closed = False
        self._begin_calls = 0
        self._proposal_calls = 0
        self._reconcile_calls = 0
        self._draft_steps = 0
        self._head_scans = 0
        self._proposed_tokens = 0
        self._accepted_tokens = 0
        self._rejected_tokens = 0
        self._linear_calls = 0
        self._source_body_bytes = max(
            0,
            _owner_metric(pager.source, "network_or_source_body_bytes")
            - self._source_start,
        )
        self._logical_weight_bytes = 0
        self._seconds = 0.0

    def _calibration_identity(self) -> dict[str, object]:
        return {
            "dim": self.config.dim,
            "provider": QWEN35_MTP_DRAFT_PROVIDER_SCHEMA,
            "q4_manifest_sha256": self.pager.q4_bank.identity["manifest_sha256"],
            "vocab_size": self.config.vocab_size,
        }

    def _load_calibration(self) -> None:
        path = self.state_path
        if path is None or not path.is_file() or path.is_symlink():
            return
        try:
            document = json.loads(path.read_bytes())
            if (
                not isinstance(document, dict)
                or document.get("schema") != QWEN35_MTP_CALIBRATION_SCHEMA
                or document.get("identity") != self._calibration_identity()
                or not isinstance(document.get("rows"), list)
                or isinstance(document.get("updates"), bool)
                or not isinstance(document.get("updates"), int)
                or document["updates"] < 0
                or document.get("previous_outcome") not in {-1, 0, 1}
            ):
                return
            restored: dict[tuple[int, int, int], list[int]] = {}
            for row in document["rows"]:
                if not isinstance(row, dict) or set(row) != {
                    "alpha",
                    "beta",
                    "bucket",
                    "position",
                    "previous",
                }:
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
                    or not 0 <= bucket <= 6
                    or previous not in {-1, 0, 1}
                    or alpha < 1
                    or beta < 1
                ):
                    return
                restored[(position, bucket, previous)] = [alpha, beta]
            self._reliability = restored
            self._previous_outcome = document["previous_outcome"]
            self._calibration_updates = document["updates"]
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return

    def _save_calibration(self) -> None:
        path = self.state_path
        if path is None:
            return
        document = {
            "identity": self._calibration_identity(),
            "previous_outcome": self._previous_outcome,
            "rows": [
                {
                    "alpha": counts[0],
                    "beta": counts[1],
                    "bucket": key[1],
                    "position": key[0],
                    "previous": key[2],
                }
                for key, counts in sorted(self._reliability.items())
            ],
            "schema": QWEN35_MTP_CALIBRATION_SCHEMA,
            "updates": self._calibration_updates,
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
        alpha, beta = self._reliability.get(
            (proposal_index, bucket, self._previous_outcome),
            [1, 1],
        )
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
        self._next_position = len(committed) - 1
        self._begin_calls += 1

    def __call__(self, _history: tuple[int, ...], /) -> tuple[int, ...]:
        raise Qwen35MtpDraftError(
            "embedded MTP requires target-hidden rolling callbacks"
        )

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
        for proposal_index in range(self.proposal_width):
            token, confidence, bucket = self._scan(
                output,
                proposal_index=proposal_index,
            )
            proposal.append(token)
            confidences.append(confidence)
            gap_buckets.append(bucket)
            if self._adaptive_round_call and proposal_index == 0 and confidence < 0.6:
                missing = self.proposal_width - len(proposal)
                proposal.extend([token] * missing)
                confidences.extend([0.0] * missing)
                gap_buckets.extend([bucket] * missing)
                states.extend([state] * (self.proposal_width + 1 - len(states)))
                break
            if token in self.eos_token_ids:
                proposal.extend([token] * (self.proposal_width - len(proposal)))
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
        previous = self._previous_outcome
        for index in range(verified_proposals):
            outcome = index < accepted_prefix_length
            key = (index, buckets[index], previous)
            alpha, beta = self._reliability.setdefault(key, [1, 1])
            if outcome:
                alpha += 1
            else:
                beta += 1
            self._reliability[key] = [alpha, beta]
            self._calibration_updates += 1
            previous = int(outcome)
            if not outcome:
                break
        self._previous_outcome = previous
        try:
            self._save_calibration()
        except OSError:
            pass

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
        self._rejected_tokens += len(proposal) - accepted
        self._pending_base = None
        self._pending_proposal = None
        self._pending_states = ()
        self._pending_gap_buckets = ()
        self._reconcile_calls += 1

    def observe_final(self, _history: tuple[int, ...], /) -> None:
        return None

    def metrics(self) -> Qwen35MtpDraftMetrics:
        return Qwen35MtpDraftMetrics(
            schema=QWEN35_MTP_DRAFT_PROVIDER_SCHEMA,
            begin_calls=self._begin_calls,
            proposal_calls=self._proposal_calls,
            reconcile_calls=self._reconcile_calls,
            draft_steps=self._draft_steps,
            head_scans=self._head_scans,
            proposed_tokens=self._proposed_tokens,
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
        self._pending_base = None
        self._pending_proposal = None
        self._pending_states = ()
        self._last_confidences = ()
        self._pending_gap_buckets = ()
        self._reliability.clear()
        self._closed = True
        self.pager.release()


__all__ = [
    "MTP_CONTROL_NAMES",
    "MTP_MATRIX_NAMES",
    "QWEN35_MTP_DRAFT_PROVIDER_SCHEMA",
    "Qwen35MtpDraftError",
    "Qwen35MtpDraftMetrics",
    "Qwen35MtpDraftProvider",
]
