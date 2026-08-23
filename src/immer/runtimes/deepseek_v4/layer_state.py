"""Owned in-memory attention handles for layer-major V4 generation.

The main decoder keeps one mutable attention object per layer.  Layer-major
generation cannot leave all of those objects resident, so this module captures
one layer after each native prefill/decode call and restores it just before the
next call for that layer.  Handles own detached CPU storage and are bound to
the exact runtime identity.  Their large KV tensors are deliberately not
content-hashed on every token.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any

import torch

from .snapshot import SnapshotTensor


def _positive_index(value: int, name: str, *, allow_zero: bool = True) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _json_value(value: Any) -> Any:
    """Return a mutable, canonical-JSON-compatible copy."""

    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("attention metadata keys must be strings")
            result[key] = _json_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        f"attention metadata contains unsupported value {type(value).__name__}"
    )


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            _json_value(value),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("layer attention identity is not canonical JSON") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _owned_tensors(
    tensors: Mapping[str, torch.Tensor | SnapshotTensor],
) -> Mapping[str, torch.Tensor]:
    if not isinstance(tensors, Mapping):
        raise TypeError("attention tensors must be a mapping")
    result: dict[str, torch.Tensor] = {}
    for name in sorted(tensors):
        if not isinstance(name, str) or not name:
            raise ValueError("attention tensor names must be non-empty strings")
        record = tensors[name]
        value = record.value if isinstance(record, SnapshotTensor) else record
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"attention tensor {name!r} is not a torch tensor")
        if value.layout != torch.strided:
            raise ValueError(f"attention tensor {name!r} must use strided storage")
        result[name] = value.detach().to(device="cpu", copy=True).contiguous()
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class LayerAttentionState:
    """One layer's portable, content-bound native attention continuation."""

    layer: int
    next_position: int
    batch_size: int
    runtime_identity_sha256: str = field(repr=False)
    metadata: Mapping[str, Any]
    tensors: Mapping[str, torch.Tensor]

    def __post_init__(self) -> None:
        _positive_index(self.layer, "layer")
        _positive_index(self.next_position, "next_position", allow_zero=False)
        _positive_index(self.batch_size, "batch_size", allow_zero=False)
        runtime_identity = self.runtime_identity_sha256
        if (
            not isinstance(runtime_identity, str)
            or len(runtime_identity) != 64
            or any(character not in "0123456789abcdef" for character in runtime_identity)
        ):
            raise ValueError("runtime_identity_sha256 must be a lowercase SHA-256")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("attention metadata must be a mapping")
        metadata = _freeze_json(_json_value(self.metadata))
        tensors = _owned_tensors(self.tensors)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "tensors", tensors)


class LayerStateRunner:
    """Execute one native decoder layer while keeping its cache on CPU."""

    def __init__(self, model: Any) -> None:
        required = (
            "_attention_state",
            "_block",
            "_snapshot_identity",
            "_token_tensor",
            "release_layer_state",
        )
        if any(not callable(getattr(model, name, None)) for name in required):
            raise TypeError("model lacks the streamed V4 layer-state contract")
        self.model = model
        self.runtime_identity_sha256 = _sha256(model._snapshot_identity())

    def _validate_handle(
        self,
        state: LayerAttentionState,
        *,
        layer: int,
        batch_size: int,
    ) -> None:
        if not isinstance(state, LayerAttentionState):
            raise TypeError("attention_state must be a LayerAttentionState")
        if state.layer != layer:
            raise ValueError(
                f"attention state belongs to layer {state.layer}, not layer {layer}"
            )
        if state.batch_size != batch_size:
            raise ValueError(
                f"attention state batch {state.batch_size} does not match {batch_size}"
            )
        if state.runtime_identity_sha256 != self.runtime_identity_sha256:
            raise ValueError("attention state belongs to a different V4 runtime")
        metadata_position = state.metadata.get("next_position")
        if metadata_position != state.next_position:
            raise ValueError("attention state cursor disagrees with its metadata")
        metadata_local = state.metadata.get("local")
        if not isinstance(metadata_local, Mapping):
            raise ValueError("attention state omits its local cache metadata")
        if metadata_local.get("batch_size") != state.batch_size:
            raise ValueError("attention state batch disagrees with its metadata")
    def _token_mask(
        self,
        value: Any | None,
        *,
        ids: torch.Tensor,
        prefill: bool,
    ) -> torch.Tensor | None:
        if value is None:
            return None
        mask = self.model.torch.as_tensor(value, device=self.model.pager.device)
        if mask.dtype != self.model.torch.bool:
            raise TypeError("token_mask must be boolean")
        if tuple(mask.shape) != tuple(ids.shape):
            raise ValueError("token_mask must match token_ids shape")
        if prefill:
            if bool((~mask[:, 0]).any().item()):
                raise ValueError("every prefill row must contain a non-empty prefix")
            if ids.shape[1] > 1 and bool(
                (mask[:, 1:] & ~mask[:, :-1]).any().item()
            ):
                raise ValueError("prefill token_mask must describe exact prefixes")
        return mask

    def forward_layer_stateful(
        self,
        hidden: Any,
        token_ids: Any,
        *,
        layer: int,
        attention_state: LayerAttentionState | None = None,
        token_mask: Any | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[tuple[int, ...], ...],
        LayerAttentionState,
    ]:
        """Run a position-zero prefill or one contiguous layer decode step.

        A decode mask may be false for finished rows.  Their fixed-row
        attention cache still advances with the supplied filler token, while
        the mask suppresses routed and shared MoE work for those rows.
        """

        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError("layer must be an integer")
        if not 0 <= layer < self.model.config.n_layers:
            raise ValueError("layer outside decoder depth")

        try:
            ids = self.model._token_tensor(token_ids)
            batch_size, sequence = (int(value) for value in ids.shape)
            prefill = attention_state is None
            if prefill:
                start_pos = 0
            else:
                self._validate_handle(
                    attention_state, layer=layer, batch_size=batch_size
                )
                start_pos = attention_state.next_position
                if sequence != 1:
                    raise ValueError("layer decode accepts exactly one token")
            end_pos = start_pos + sequence
            if end_pos > self.model.max_seq_len:
                raise ValueError(
                    f"context end {end_pos} exceeds max_seq_len={self.model.max_seq_len}"
                )

            if not isinstance(hidden, torch.Tensor) or not hidden.is_floating_point():
                raise TypeError("hidden must be a floating-point torch tensor")
            expected_shape = (
                batch_size,
                sequence,
                self.model.config.hc_mult,
                self.model.config.dim,
            )
            if tuple(hidden.shape) != expected_shape:
                raise ValueError(
                    f"hidden shape {tuple(hidden.shape)} does not match {expected_shape}"
                )
            mask = self._token_mask(token_mask, ids=ids, prefill=prefill)
            hidden = hidden.to(
                device=self.model.pager.device,
                dtype=self.model.pager.compute_dtype,
            )

            # Never restore into an object left over by another microbatch.
            self.model.release_layer_state(layer)
            resident, _freqs = self.model._attention_state(layer)
            if attention_state is not None:
                restored = {
                    name: value.detach()
                    .to(device=self.model.pager.device, copy=True)
                    .contiguous()
                    for name, value in attention_state.tensors.items()
                }
                resident._restore_snapshot(attention_state.metadata, restored)
                if resident.next_position != start_pos:
                    raise ValueError("restored attention cursor is not contiguous")

            output, selected = self.model._block(
                hidden,
                layer,
                ids,
                start_pos,
                mask,
            )
            if resident.next_position != end_pos:
                raise RuntimeError("layer attention cursor did not advance exactly")
            records: dict[str, SnapshotTensor] = {}
            metadata = resident._snapshot_state(
                f"attention.layer_{layer:03d}", records
            )
            result_state = LayerAttentionState(
                layer=layer,
                next_position=end_pos,
                batch_size=batch_size,
                runtime_identity_sha256=self.runtime_identity_sha256,
                metadata=metadata,
                tensors=records,
            )
            return output, selected, result_state
        finally:
            self.model.release_layer_state(layer)


__all__ = ["LayerAttentionState", "LayerStateRunner"]
