"""Synchronous chat component for an already constructed DeepSeek-V4 runtime.

Construction of the streamed model remains outside the composition root.  The
adapter only owns request framing, bounds, serialization, and cleanup around a
prebuilt model/tokenizer pair.
"""

from __future__ import annotations

import inspect
import math
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from numbers import Integral
from typing import Any

from ...contracts import ExecutionStatus, Request, Result
from .encoding import encode_user_prompt


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _token_ids(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{label} must be a sequence of integer token IDs")
    result: list[int] = []
    for token_id in value:
        if isinstance(token_id, bool) or not isinstance(token_id, Integral):
            raise TypeError(f"{label} must contain only integer token IDs")
        result.append(int(token_id))
    return tuple(result)


def _public_value(value: Any) -> Any:
    """Copy generation evidence into a small JSON-compatible value tree."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("generation evidence contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("generation evidence keys must be strings")
            copied[key] = _public_value(item)
        return copied
    if isinstance(value, tuple):
        return tuple(_public_value(item) for item in value)
    if isinstance(value, list):
        return [_public_value(item) for item in value]
    raise TypeError(
        f"generation evidence contains unsupported {type(value).__name__} value"
    )


def _generation_evidence(value: object) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise TypeError("generation evidence must be a dataclass or mapping")
    copied = _public_value(value)
    if not isinstance(copied, dict):  # Mapping conversion above is deliberately exact.
        raise AssertionError("generation evidence did not produce a dictionary")
    return copied


class DeepSeekV4Chat:
    """Expose one prebuilt streamed DeepSeek-V4 decoder as ``chat``."""

    name = "deepseek-v4.chat"
    capabilities = frozenset({"chat"})

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_id: str,
        revision: str,
        tokenizer_sha256: str | None = None,
        thinking_mode: str = "chat",
        reasoning_effort: str = "low",
        assistant_prefix: str = "",
        max_prompt_tokens: int = 512,
        max_new_tokens: int = 96,
        max_context_tokens: int | None = None,
        eos_token_ids: Sequence[int] = (),
        prefill_tokenwise: bool = True,
        head_block_rows: int = 1024,
    ) -> None:
        if not callable(getattr(model, "generate_greedy", None)):
            raise TypeError("model must provide generate_greedy")
        if not callable(getattr(model, "reset_state", None)):
            raise TypeError("model must provide reset_state")
        if not callable(getattr(tokenizer, "encode", None)) or not callable(
            getattr(tokenizer, "decode", None)
        ):
            raise TypeError("tokenizer must provide encode and decode")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be non-empty text")
        if not isinstance(revision, str) or not revision.strip():
            raise ValueError("revision must be non-empty text")
        if tokenizer_sha256 is None:
            tokenizer_sha256 = getattr(tokenizer, "sha256", None)
        if not isinstance(tokenizer_sha256, str) or _SHA256.fullmatch(
            tokenizer_sha256.lower()
        ) is None:
            raise ValueError("tokenizer_sha256 must be a 64-character hex digest")
        if thinking_mode not in {"chat", "thinking"}:
            raise ValueError("thinking_mode must be 'chat' or 'thinking'")
        if reasoning_effort not in {"low", "high", "max"}:
            raise ValueError("reasoning_effort must be low, high, or max")
        if not isinstance(assistant_prefix, str):
            raise TypeError("assistant_prefix must be text")
        max_prompt_tokens = _positive_int(max_prompt_tokens, "max_prompt_tokens")
        max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        head_block_rows = _positive_int(head_block_rows, "head_block_rows")
        if not isinstance(prefill_tokenwise, bool):
            raise TypeError("prefill_tokenwise must be a boolean")

        model_context = _positive_int(
            getattr(model, "max_seq_len", None), "model.max_seq_len"
        )
        configured_context = (
            model_context
            if max_context_tokens is None
            else _positive_int(max_context_tokens, "max_context_tokens")
        )
        config = getattr(model, "config", None)
        checkpoint_context = getattr(config, "max_position_embeddings", model_context)
        checkpoint_context = _positive_int(
            checkpoint_context, "model.config.max_position_embeddings"
        )
        if configured_context > min(model_context, checkpoint_context):
            raise ValueError("max_context_tokens exceeds the prebuilt model context")

        vocab_size = _positive_int(
            getattr(config, "vocab_size", None), "model.config.vocab_size"
        )
        eos = _token_ids(eos_token_ids, "eos_token_ids")
        if len(eos) != len(set(eos)):
            raise ValueError("eos_token_ids must be distinct")
        if any(token_id < 0 or token_id >= vocab_size for token_id in eos):
            raise ValueError("EOS token outside checkpoint vocabulary")

        graft = getattr(model, "graft", None)
        graft_mode = "off" if graft is None else getattr(graft, "mode", None)
        if not isinstance(graft_mode, str) or not graft_mode:
            raise ValueError("configured graft must expose a non-empty mode")
        graft_alpha = 0.0 if graft is None else getattr(graft, "alpha", None)
        if (
            isinstance(graft_alpha, bool)
            or not isinstance(graft_alpha, (int, float))
            or not math.isfinite(float(graft_alpha))
            or float(graft_alpha) < 0.0
        ):
            raise ValueError("configured graft must expose a finite non-negative alpha")
        graft_layer = getattr(model, "graft_layer", None)
        if graft_layer is not None and (
            isinstance(graft_layer, bool) or not isinstance(graft_layer, int)
        ):
            raise ValueError("model.graft_layer must be an integer or None")

        self._model = model
        self._tokenizer = tokenizer
        self._model_id = model_id.strip()
        self._revision = revision.strip()
        self._tokenizer_sha256 = tokenizer_sha256.lower()
        self._thinking_mode = thinking_mode
        self._reasoning_effort = reasoning_effort
        self._assistant_prefix = assistant_prefix
        self._max_prompt_tokens = max_prompt_tokens
        self._max_new_tokens = max_new_tokens
        self._max_context_tokens = configured_context
        self._vocab_size = vocab_size
        self._eos_token_ids = eos
        self._prefill_tokenwise = prefill_tokenwise
        self._head_block_rows = head_block_rows
        self._graft_mode = graft_mode
        self._graft_alpha = float(graft_alpha)
        self._graft_layer = graft_layer
        self._lock = threading.Lock()
        self._cleanup_error: str | None = None

    def _base_evidence(self) -> dict[str, Any]:
        return {
            "model": self._model_id,
            "revision": self._revision,
            "tokenizer_sha256": self._tokenizer_sha256,
            "thinking_mode": self._thinking_mode,
            "reasoning_effort": self._reasoning_effort,
            "assistant_prefix": self._assistant_prefix,
            "graft_mode": self._graft_mode,
            "graft_alpha": self._graft_alpha,
            "graft_layer": self._graft_layer,
            "max_prompt_tokens": self._max_prompt_tokens,
            "max_new_tokens": self._max_new_tokens,
            "max_context_tokens": self._max_context_tokens,
            "eos_token_ids": self._eos_token_ids,
        }

    def _reset(self, *, release: bool) -> None:
        reset = self._model.reset_state
        try:
            parameters = inspect.signature(reset).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        supports_release = any(
            parameter.name == "release"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
        if supports_release:
            reset(release=release)
        else:
            reset()

    def _run(self, text: str) -> Result:
        envelope = encode_user_prompt(
            text,
            thinking_mode=self._thinking_mode,
            reasoning_effort=self._reasoning_effort,
        )
        encoded = self._tokenizer.encode(envelope + self._assistant_prefix)
        raw_prompt_ids = getattr(encoded, "ids", encoded)
        prompt_ids = _token_ids(raw_prompt_ids, "encoded prompt")
        if not prompt_ids:
            raise ValueError("official prompt encoding produced no tokens")
        if len(prompt_ids) > self._max_prompt_tokens:
            raise ValueError(
                f"prompt has {len(prompt_ids)} tokens, limit is "
                f"{self._max_prompt_tokens}"
            )
        if any(token_id < 0 or token_id >= self._vocab_size for token_id in prompt_ids):
            raise ValueError("prompt token outside checkpoint vocabulary")
        if len(prompt_ids) + self._max_new_tokens > self._max_context_tokens:
            raise ValueError("prompt plus output budget exceeds model context")

        raw_generated, raw_evidence = self._model.generate_greedy(
            [list(prompt_ids)],
            max_new_tokens=self._max_new_tokens,
            prefill_tokenwise=self._prefill_tokenwise,
            eos_token_ids=self._eos_token_ids,
            head_block_rows=self._head_block_rows,
        )
        generated_ids = _token_ids(raw_generated, "generated output")
        if len(generated_ids) > self._max_new_tokens:
            raise ValueError("generated output exceeds max_new_tokens")
        if len(prompt_ids) + len(generated_ids) > self._max_context_tokens:
            raise ValueError("generated output exceeds model context")
        if any(
            token_id < 0 or token_id >= self._vocab_size
            for token_id in generated_ids
        ):
            raise ValueError("generated token outside checkpoint vocabulary")
        eos_positions = [
            index
            for index, token_id in enumerate(generated_ids)
            if token_id in self._eos_token_ids
        ]
        if eos_positions and eos_positions[0] != len(generated_ids) - 1:
            raise ValueError("generated output contains tokens after EOS")

        generation = _generation_evidence(raw_evidence)
        evidence_ids = generation.get("generated_token_ids")
        if evidence_ids is not None and _token_ids(
            evidence_ids, "generation evidence token IDs"
        ) != generated_ids:
            raise ValueError("generation evidence does not match generated output")
        decoded = self._tokenizer.decode(list(generated_ids))
        if not isinstance(decoded, str):
            raise TypeError("tokenizer decode must return text")
        evidence = {
            **self._base_evidence(),
            "prompt_token_count": len(prompt_ids),
            "generated_token_ids": generated_ids,
            "generation": generation,
        }
        output = decoded.strip()
        if not output:
            return Result(
                ExecutionStatus.ABSTAINED,
                self.name,
                reason="DeepSeek-V4 decoded an empty response",
                evidence=evidence,
            )
        return Result(
            ExecutionStatus.OK,
            self.name,
            output=output,
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
            if self._cleanup_error is not None:
                previous_error = self._cleanup_error
                try:
                    self._reset(release=True)
                except Exception as recovery_exc:
                    return Result(
                        ExecutionStatus.ERROR,
                        self.name,
                        reason="DeepSeek-V4 state cleanup remains unavailable",
                        evidence={
                            **self._base_evidence(),
                            "cleanup": {
                                "status": "error",
                                "previous_error": previous_error,
                                "recovery_error": (
                                    f"{type(recovery_exc).__name__}: {recovery_exc}"
                                ),
                            },
                        },
                    )
                self._cleanup_error = None
            failed = False
            try:
                result = self._run(request.payload)
            except Exception as exc:  # one failed decoder request must not escape runtime
                failed = True
                result = Result(
                    ExecutionStatus.ERROR,
                    self.name,
                    reason=f"DeepSeek-V4 generation failed: {type(exc).__name__}: {exc}",
                    evidence=self._base_evidence(),
                )
            try:
                self._reset(release=failed)
            except Exception as cleanup_exc:  # preserve the primary request outcome
                self._cleanup_error = (
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
                evidence = dict(result.evidence)
                evidence["cleanup"] = {
                    "status": "error",
                    "release_requested": failed,
                    "error": self._cleanup_error,
                }
                result = Result(
                    result.status,
                    result.component,
                    output=result.output,
                    reason=result.reason,
                    evidence=evidence,
                )
            return result


__all__ = ["DeepSeekV4Chat"]
