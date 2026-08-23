"""One-file rolling resume checkpoints for Qwen3.8 draft verification.

The checkpoint is deliberately small in scope: one decoder-boundary hidden
state plus the counters required by :class:`Qwen38DraftVerifier`.  The caller
binds it to the exact token batch and execution/graft contract through a
deterministic identity; no model weights or range-cache objects are copied.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any

import torch

from .draft_verification import DraftVerificationResumeState


RESUME_SCHEMA = "immer.qwen3.8-draft-resume/v1"
_MAX_HEADER_BYTES = 64 * 1024
_IDENTITY_PATTERN = re.compile(r"[0-9a-f]{64}")
_ELEMENT_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "float32": 4,
}


class Qwen38ResumeError(RuntimeError):
    """A rolling resume file or its contract is invalid."""


def _absolute(path: str | os.PathLike[str]) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _dtype_name(dtype: str | torch.dtype) -> str:
    name = str(dtype).removeprefix("torch.")
    if name not in _ELEMENT_BYTES:
        raise Qwen38ResumeError(f"unsupported rolling resume dtype: {name}")
    return name


def _shape(value: Sequence[int]) -> tuple[int, int, int]:
    try:
        shape = tuple(value)
    except TypeError as exc:
        raise Qwen38ResumeError("resume hidden shape must be a 3D sequence") from exc
    if len(shape) != 3 or any(
        isinstance(size, bool)
        or not isinstance(size, numbers.Integral)
        or int(size) <= 0
        for size in shape
    ):
        raise Qwen38ResumeError(
            "resume hidden shape must be [batch, sequence, dimension]"
        )
    return tuple(int(size) for size in shape)  # type: ignore[return-value]


def _positive_layers(n_layers: int) -> int:
    if (
        isinstance(n_layers, bool)
        or not isinstance(n_layers, numbers.Integral)
        or int(n_layers) <= 0
    ):
        raise Qwen38ResumeError("n_layers must be a positive integer")
    return int(n_layers)


def _graft_layer(value: int | None, n_layers: int) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or not 0 <= int(value) < n_layers
    ):
        raise Qwen38ResumeError("active_graft_layer is outside decoder depth")
    return int(value)


def _json_value(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        converted = float(value)
        if not math.isfinite(converted):
            raise Qwen38ResumeError(f"{name} contains a non-finite number")
        return converted
    if isinstance(value, Mapping):
        converted_mapping: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise Qwen38ResumeError(f"{name} mapping keys must be strings")
            converted_mapping[key] = _json_value(nested, name)
        return converted_mapping
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(nested, name) for nested in value]
    raise Qwen38ResumeError(f"{name} is not deterministically JSON serializable")


def _token_batch(value: Any, name: str, *, allow_empty_rows: bool) -> list[list[int]]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise Qwen38ResumeError(f"{name} must be a batch of token rows")
    batch: list[list[int]] = []
    for row in value:
        if isinstance(row, (str, bytes, bytearray)) or not isinstance(row, Sequence):
            raise Qwen38ResumeError(f"{name} must be a batch of token rows")
        converted: list[int] = []
        for token in row:
            if (
                isinstance(token, bool)
                or not isinstance(token, numbers.Integral)
                or int(token) < 0
            ):
                raise Qwen38ResumeError(f"{name} contains an invalid token id")
            converted.append(int(token))
        if not converted and not allow_empty_rows:
            raise Qwen38ResumeError(f"{name} rows must not be empty")
        batch.append(converted)
    if not batch:
        raise Qwen38ResumeError(f"{name} must not be empty")
    return batch


def build_resume_identity(
    *,
    source_id: str,
    source_revision: str,
    prompt_token_ids: Sequence[Sequence[int]],
    draft_token_ids: Sequence[Sequence[int]],
    execution_contract: Mapping[str, Any],
    graft_contract: Mapping[str, Any] | None = None,
) -> str:
    """Bind a checkpoint to its immutable source, inputs, and math contract."""

    if not isinstance(source_id, str) or not source_id.strip():
        raise Qwen38ResumeError("source_id must be a non-empty string")
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise Qwen38ResumeError("source_revision must be a non-empty string")
    if not isinstance(execution_contract, Mapping):
        raise Qwen38ResumeError("execution_contract must be a mapping")
    if graft_contract is not None and not isinstance(graft_contract, Mapping):
        raise Qwen38ResumeError("graft_contract must be a mapping or None")
    prompts = _token_batch(
        prompt_token_ids,
        "prompt_token_ids",
        allow_empty_rows=False,
    )
    drafts = _token_batch(
        draft_token_ids,
        "draft_token_ids",
        allow_empty_rows=True,
    )
    if len(prompts) != len(drafts):
        raise Qwen38ResumeError("prompt and draft batches must have the same size")
    payload = {
        "schema": RESUME_SCHEMA,
        "source_id": source_id,
        "source_revision": source_revision,
        "prompt_token_ids": prompts,
        "draft_token_ids": drafts,
        "execution_contract": _json_value(execution_contract, "execution_contract"),
        "graft_contract": _json_value(graft_contract, "graft_contract"),
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:  # Defensive boundary around json internals.
        raise Qwen38ResumeError("resume identity contract is invalid") from exc
    return hashlib.sha256(encoded).hexdigest()


def hidden_state_bytes(
    shape: Sequence[int],
    dtype: str | torch.dtype,
) -> int:
    """Return the exact tensor payload bytes for native Qwen ``[B, S, D]``."""

    return math.prod(_shape(shape)) * _ELEMENT_BYTES[_dtype_name(dtype)]


def _max_file_bytes(shape: Sequence[int], dtype: str | torch.dtype) -> int:
    # Safetensors stores an eight-byte header length, a bounded JSON header,
    # then the raw tensor payload.
    return hidden_state_bytes(shape, dtype) + 8 + _MAX_HEADER_BYTES


def _lstat(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise Qwen38ResumeError(f"cannot inspect rolling resume path: {path}") from exc


def _reject_nonregular(path: Path, *, missing_ok: bool) -> os.stat_result | None:
    info = _lstat(path)
    if info is None:
        if missing_ok:
            return None
        raise Qwen38ResumeError(f"rolling resume file does not exist: {path}")
    if stat.S_ISLNK(info.st_mode):
        raise Qwen38ResumeError(f"rolling resume path must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise Qwen38ResumeError(f"rolling resume path must be a regular file: {path}")
    return info


def preflight_resume_disk(
    path: str | os.PathLike[str],
    *,
    expected_shape: Sequence[int],
    expected_dtype: str | torch.dtype,
) -> dict[str, int]:
    """Check only the space an atomic replacement can actually require."""

    destination = _absolute(path)
    _reject_nonregular(destination, missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    hidden_bytes = hidden_state_bytes(expected_shape, expected_dtype)
    required_bytes = _max_file_bytes(expected_shape, expected_dtype)
    try:
        free_bytes = int(shutil.disk_usage(destination.parent).free)
    except OSError as exc:
        raise Qwen38ResumeError("cannot inspect rolling resume disk space") from exc
    if free_bytes < required_bytes:
        raise Qwen38ResumeError(
            "insufficient free disk for the next atomic rolling checkpoint: "
            f"need {required_bytes} bytes, have {free_bytes}"
        )
    return {
        "free_bytes": free_bytes,
        "required_bytes": required_bytes,
        "hidden_bytes": hidden_bytes,
        "max_file_bytes": required_bytes,
    }


def _metadata_int(metadata: Mapping[str, str], name: str) -> int:
    raw = metadata.get(name)
    try:
        value = int(raw) if raw is not None else -1
    except (TypeError, ValueError) as exc:
        raise Qwen38ResumeError(f"resume metadata {name} is invalid") from exc
    if value < 0 or str(value) != raw:
        raise Qwen38ResumeError(f"resume metadata {name} is invalid")
    return value


def _validate_identity(identity: str) -> str:
    if not isinstance(identity, str) or _IDENTITY_PATTERN.fullmatch(identity) is None:
        raise Qwen38ResumeError("resume identity must be a lowercase SHA-256 digest")
    return identity


def _validate_state(
    state: DraftVerificationResumeState,
    *,
    expected_shape: Sequence[int],
    expected_dtype: str | torch.dtype,
    n_layers: int,
    active_graft_layer: int | None,
) -> tuple[tuple[int, int, int], str, int, int | None]:
    if not isinstance(state, DraftVerificationResumeState):
        raise Qwen38ResumeError("state has the wrong resume type")
    shape = _shape(expected_shape)
    dtype = _dtype_name(expected_dtype)
    layers = _positive_layers(n_layers)
    graft_layer = _graft_layer(active_graft_layer, layers)
    if not isinstance(state.hidden, torch.Tensor):
        raise Qwen38ResumeError("resume hidden state must be a torch tensor")
    if tuple(state.hidden.shape) != shape:
        raise Qwen38ResumeError("rolling resume hidden shape is invalid")
    observed_dtype = str(state.hidden.dtype).removeprefix("torch.")
    if observed_dtype != dtype:
        raise Qwen38ResumeError("rolling resume hidden dtype is invalid")
    for name, value in (
        ("next_layer", state.next_layer),
        ("layer_calls", state.layer_calls),
        ("layer_retry_count", state.layer_retry_count),
        ("source_body_bytes", state.source_body_bytes),
        ("linear_calls", state.linear_calls),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise Qwen38ResumeError(f"resume {name} must be a non-negative integer")
    if not 1 <= state.next_layer <= layers or state.layer_calls != state.next_layer:
        raise Qwen38ResumeError("rolling resume layer boundary is invalid")
    if (
        isinstance(state.seconds, bool)
        or not isinstance(state.seconds, (int, float))
        or not math.isfinite(float(state.seconds))
        or float(state.seconds) < 0.0
    ):
        raise Qwen38ResumeError("rolling resume seconds are invalid")
    expected_graft = graft_layer is not None and state.next_layer > graft_layer
    if (
        not isinstance(state.graft_applied, bool)
        or state.graft_applied != expected_graft
    ):
        raise Qwen38ResumeError(
            "rolling resume graft state does not match its layer boundary"
        )
    return shape, dtype, layers, graft_layer


def write_resume(
    path: str | os.PathLike[str],
    identity: str,
    state: DraftVerificationResumeState,
    *,
    expected_shape: Sequence[int],
    expected_dtype: str | torch.dtype,
    n_layers: int,
    active_graft_layer: int | None = None,
) -> None:
    """Atomically replace the one rolling safetensors checkpoint."""

    identity = _validate_identity(identity)
    shape, dtype, _layers, _graft = _validate_state(
        state,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        n_layers=n_layers,
        active_graft_layer=active_graft_layer,
    )
    destination = _absolute(path)
    _reject_nonregular(destination, missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    hidden = state.hidden.detach().to(device="cpu").contiguous()
    metadata = {
        "schema": RESUME_SCHEMA,
        "identity": identity,
        "next_layer": str(state.next_layer),
        "layer_calls": str(state.layer_calls),
        "layer_retry_count": str(state.layer_retry_count),
        "source_body_bytes": str(state.source_body_bytes),
        "linear_calls": str(state.linear_calls),
        "seconds": repr(float(state.seconds)),
        "graft_applied": "1" if state.graft_applied else "0",
        "dtype": dtype,
        "shape": ",".join(str(size) for size in shape),
    }
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise Qwen38ResumeError("rolling resume requires safetensors") from exc

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".pending",
        dir=destination.parent,
    )
    os.close(descriptor)
    try:
        save_file({"hidden": hidden}, temporary_name, metadata=metadata)
        temporary = Path(temporary_name)
        if temporary.stat().st_size > _max_file_bytes(shape, dtype):
            raise Qwen38ResumeError("generated rolling resume exceeds its file bound")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Qwen38ResumeError:
        raise
    except Exception as exc:
        raise Qwen38ResumeError("cannot write rolling resume file") from exc
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def load_resume(
    path: str | os.PathLike[str],
    identity: str,
    *,
    expected_shape: Sequence[int],
    expected_dtype: str | torch.dtype,
    n_layers: int,
    active_graft_layer: int | None = None,
) -> DraftVerificationResumeState | None:
    """Load and strictly validate one decoder-boundary checkpoint."""

    identity = _validate_identity(identity)
    shape = _shape(expected_shape)
    dtype = _dtype_name(expected_dtype)
    layers = _positive_layers(n_layers)
    graft_layer = _graft_layer(active_graft_layer, layers)
    source = _absolute(path)
    info = _reject_nonregular(source, missing_ok=True)
    if info is None:
        return None
    maximum = _max_file_bytes(shape, dtype)
    minimum = hidden_state_bytes(shape, dtype) + 8
    if info.st_size < minimum or info.st_size > maximum:
        raise Qwen38ResumeError("rolling resume file exceeds its hidden-state bound")
    try:
        from safetensors import safe_open

        with safe_open(str(source), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {"hidden"}:
                raise Qwen38ResumeError("rolling resume has unexpected tensors")
            metadata = handle.metadata()
            hidden = handle.get_tensor("hidden")
    except Qwen38ResumeError:
        raise
    except Exception as exc:
        raise Qwen38ResumeError("cannot decode rolling resume file") from exc
    required_metadata = {
        "schema",
        "identity",
        "next_layer",
        "layer_calls",
        "layer_retry_count",
        "source_body_bytes",
        "linear_calls",
        "seconds",
        "graft_applied",
        "dtype",
        "shape",
    }
    if (
        not isinstance(metadata, dict)
        or set(metadata) != required_metadata
        or metadata.get("schema") != RESUME_SCHEMA
    ):
        raise Qwen38ResumeError("rolling resume metadata schema is invalid")
    if metadata.get("identity") != identity:
        raise Qwen38ResumeError("rolling resume belongs to another run")
    if metadata.get("shape") != ",".join(str(size) for size in shape):
        raise Qwen38ResumeError("rolling resume hidden shape metadata is invalid")
    observed_dtype = str(hidden.dtype).removeprefix("torch.")
    if (
        tuple(hidden.shape) != shape
        or metadata.get("dtype") != observed_dtype
        or observed_dtype != dtype
    ):
        raise Qwen38ResumeError("rolling resume hidden shape or dtype is invalid")
    try:
        seconds = float(metadata["seconds"])
    except (TypeError, ValueError) as exc:
        raise Qwen38ResumeError("rolling resume seconds are invalid") from exc
    graft_raw = metadata.get("graft_applied")
    if graft_raw not in {"0", "1"}:
        raise Qwen38ResumeError("rolling resume graft state is invalid")
    state = DraftVerificationResumeState(
        next_layer=_metadata_int(metadata, "next_layer"),
        hidden=hidden,
        layer_calls=_metadata_int(metadata, "layer_calls"),
        layer_retry_count=_metadata_int(metadata, "layer_retry_count"),
        source_body_bytes=_metadata_int(metadata, "source_body_bytes"),
        linear_calls=_metadata_int(metadata, "linear_calls"),
        seconds=seconds,
        graft_applied=graft_raw == "1",
    )
    _validate_state(
        state,
        expected_shape=shape,
        expected_dtype=dtype,
        n_layers=layers,
        active_graft_layer=graft_layer,
    )
    return state


def delete_resume(path: str | os.PathLike[str]) -> bool:
    """Explicitly delete a regular resume file; never follow a symlink."""

    destination = _absolute(path)
    info = _reject_nonregular(destination, missing_ok=True)
    if info is None:
        return False
    try:
        destination.unlink()
    except OSError as exc:
        raise Qwen38ResumeError("cannot delete rolling resume file") from exc
    return True


__all__ = [
    "RESUME_SCHEMA",
    "Qwen38ResumeError",
    "build_resume_identity",
    "delete_resume",
    "hidden_state_bytes",
    "load_resume",
    "preflight_resume_disk",
    "write_resume",
]
