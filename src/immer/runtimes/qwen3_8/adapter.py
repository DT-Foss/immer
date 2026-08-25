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
from typing import Any

from ...contracts import ExecutionStatus, Request, Result
from ..deepseek_v4.causal_weights import CausalWeightMount, LogicalModelIdentity
from .bundle import verify_qwen38_causal_mount
from .config import OFFICIAL_REPO_ID, OFFICIAL_REVISION, Qwen38Config
from .encoding import END_OF_TEXT_TOKEN_ID, IM_END_TOKEN_ID, Qwen38Tokenizer
from .model import StreamedQwen38
from .pager import Qwen38WeightPager


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
    ) -> None:
        self.mount = mount
        self.pager = pager
        self.model = model
        self.tokenizer = tokenizer
        self.tokenizer_sha256 = tokenizer_sha256
        self.bundle_receipt = dict(bundle_receipt)
        self.preflight_receipt = dict(preflight_receipt)
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        failures: list[Exception] = []
        try:
            self.model.reset_state(release=True)
        except Exception as exc:  # release the remaining owners regardless
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


def _open_official_runtime(
    *,
    bundle_path: Path,
    tokenizer_path: Path,
    device: str,
    compute_dtype: str,
    source_budget_mb: float,
    max_resident_bytes: int,
    max_context_tokens: int,
) -> _OwnedRuntime:
    """Open only the fixed official local model; no remote source exists here."""

    if not bundle_path.is_dir():
        raise FileNotFoundError(f"causal bundle directory is missing: {bundle_path}")
    tokenizer_sha256 = _file_sha256(tokenizer_path)

    mount: CausalWeightMount | None = None
    pager: Qwen38WeightPager | None = None
    model: StreamedQwen38 | None = None
    try:
        mount = CausalWeightMount(
            bundle_path,
            LogicalModelIdentity(OFFICIAL_REPO_ID, OFFICIAL_REVISION),
            budget_mb=source_budget_mb,
        )
        bundle_receipt = verify_qwen38_causal_mount(
            mount,
            require_official_config=True,
        )
        config = Qwen38Config.from_file(
            mount.weights_root / "config.json",
            require_official=True,
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
        model = StreamedQwen38(
            config,
            pager,
            max_batch_size=1,
            max_seq_len=max_context_tokens,
        )
        preflight_receipt = model.checkpoint_preflight()
        tokenizer = Qwen38Tokenizer(tokenizer_path, require_official=True)
        if _file_sha256(tokenizer_path) != tokenizer_sha256:
            raise Qwen38ChatError("local tokenizer changed while it was loaded")
        return _OwnedRuntime(
            mount=mount,
            pager=pager,
            model=model,
            tokenizer=tokenizer,
            tokenizer_sha256=tokenizer_sha256,
            bundle_receipt=bundle_receipt,
            preflight_receipt=preflight_receipt,
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
        source_budget_mb: float = 65536,
        max_resident_bytes: int = Qwen38WeightPager.DEFAULT_MAX_RESIDENT_BYTES,
        max_prompt_tokens: int = 1024,
        max_new_tokens: int = 64,
        max_context_tokens: int = 2048,
        head_block_rows: int = Qwen38WeightPager.DEFAULT_HEAD_BLOCK_ROWS,
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
        max_resident_bytes = _positive_int(max_resident_bytes, "max_resident_bytes")
        max_prompt_tokens = _positive_int(max_prompt_tokens, "max_prompt_tokens")
        max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        max_context_tokens = _positive_int(max_context_tokens, "max_context_tokens")
        head_block_rows = _positive_int(head_block_rows, "head_block_rows")
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
        self._runtime: Any | None = None
        self._bundle_receipt: dict[str, Any] | None = None
        self._tokenizer_sha256: str | None = None
        self._load_error: str | None = None
        self._close_error: str | None = None
        self._closed = False
        self._lock = threading.RLock()

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
        )

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

        raw_generated, raw_evidence = runtime.model.generate_greedy(
            [list(prompt_ids)],
            max_new_tokens=self._max_new_tokens,
            prefill_tokenwise=False,
            eos_token_ids=(IM_END_TOKEN_ID, END_OF_TEXT_TOKEN_ID),
            head_block_rows=self._head_block_rows,
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
        self._runtime = None
        self._load_error = f"runtime retired after cleanup failure: {detail}"
        if runtime is not None:
            try:
                runtime.close()
            except Exception as close_exc:
                detail += f"; close: {type(close_exc).__name__}: {close_exc}"
        return detail

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
                result = Result(
                    ExecutionStatus.ERROR,
                    self.name,
                    reason=f"Qwen3.8 generation failed: {type(exc).__name__}: {exc}",
                    evidence=self._base_evidence(),
                )
            try:
                runtime.model.reset_state(release=True)
            except Exception as exc:
                cleanup = self._retire_runtime_locked(exc)
                return Result(
                    ExecutionStatus.ERROR,
                    self.name,
                    reason="Qwen3.8 state cleanup failed",
                    evidence={
                        **self._base_evidence(),
                        "cleanup": {"status": "error", "detail": cleanup},
                    },
                )
            return result

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            runtime = self._runtime
            self._runtime = None
            self._closed = True
            if runtime is not None:
                try:
                    runtime.close()
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


__all__ = ["Qwen38CausalChat", "Qwen38Chat", "Qwen38ChatError"]
