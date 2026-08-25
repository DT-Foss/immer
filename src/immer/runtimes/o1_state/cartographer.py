"""Persistent, bounded O1-guided probe scheduling.

The cartographer decides *what to measure next*.  Probe execution and semantic
atlas promotion stay outside this module: an O1 stream contributes only a
rolling surprise signal, while a successful measurement becomes promotable
only after an external atlas receipt is attached.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import tempfile
import threading
import time
import unicodedata
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Literal, Protocol


PROBE_JOB_SCHEMA = "immer.o1-cartography-probe-job/v1"
PROBE_OUTCOME_SCHEMA = "immer.o1-cartography-probe-outcome/v1"
COVERAGE_SCHEMA = "immer.o1-cartography-coverage/v1"
CARTOGRAPHER_STATE_SCHEMA = "immer.o1-cartography-state/v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}")
_FORBIDDEN_RAW_KEYS = frozenset(
    {
        "completion",
        "decoded",
        "input",
        "input_text",
        "output",
        "output_text",
        "prompt",
        "prompt_text",
        "raw_prompt",
    }
)
_MAX_JSON_DEPTH = 16
_MAX_JSON_NODES = 8192
_MAX_STATE_BYTES = 64 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_PATH_LOCKS: dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


class CartographyError(ValueError):
    """Base error for malformed cartography state or operations."""


class CartographyIntegrityError(CartographyError):
    """A persisted scheduler sidecar is malformed or has been modified."""


class CartographyIdentityError(CartographyError):
    """A sidecar belongs to different code, model, jobs, or scheduler policy."""


def _thread_path_lock(path: Path) -> threading.RLock:
    key = os.path.normcase(os.path.abspath(os.fspath(path)))
    with _PATH_LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, threading.RLock())


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _stable_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        stat.S_IFMT(value.st_mode),
    )


def _validate_open_path_identity(
    descriptor_stat: os.stat_result,
    path_stat: os.stat_result,
    *,
    label: str,
) -> None:
    if not stat.S_ISREG(descriptor_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise CartographyIntegrityError(f"{label} is not a regular file")
    if not _same_inode(descriptor_stat, path_stat):
        raise CartographyIntegrityError(f"{label} changed during descriptor open")


@contextmanager
def _exclusive_state_lock(path: Path) -> Iterator[None]:
    """Serialize one state-file transaction across threads and processes."""

    thread_lock = _thread_path_lock(path)
    with thread_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(f".{path.name}.lock")
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise CartographyIntegrityError(
                "cartography state lock is unavailable"
            ) from exc
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            opened_before = os.fstat(descriptor)
            try:
                path_before = os.lstat(lock_path)
            except OSError as exc:
                raise CartographyIntegrityError(
                    "cartography state lock path disappeared"
                ) from exc
            _validate_open_path_identity(
                opened_before, path_before, label="cartography state lock"
            )
            yield
            opened_after = os.fstat(descriptor)
            try:
                path_after = os.lstat(lock_path)
            except OSError as exc:
                raise CartographyIntegrityError(
                    "cartography state lock path disappeared"
                ) from exc
            _validate_open_path_identity(
                opened_after, path_after, label="cartography state lock"
            )
            if _stable_signature(opened_before) != _stable_signature(opened_after):
                raise CartographyIntegrityError(
                    "cartography state lock changed during transaction"
                )
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


class SurpriseStream(Protocol):
    """Small interface implemented by ``O1StateStream`` and ``LearningStream``."""

    loss_ema: float | None

    def observe(self, text: str) -> None: ...

    def snapshot(self) -> Mapping[str, Any]: ...

    def restore(self, state: Mapping[str, Any]) -> None: ...


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _read_stable_regular_file(path: Path) -> bytes:
    """Read one bounded inode without following or racing a path replacement."""

    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened_before = os.fstat(descriptor)
        path_before = os.lstat(path)
        _validate_open_path_identity(
            opened_before, path_before, label="cartography sidecar"
        )
        if opened_before.st_size < 0 or opened_before.st_size > _MAX_STATE_BYTES:
            raise CartographyIntegrityError(
                f"cartography sidecar exceeds {_MAX_STATE_BYTES} bytes"
            )
        chunks: list[bytes] = []
        remaining = opened_before.st_size
        while remaining:
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise CartographyIntegrityError(
                    "cartography sidecar was truncated while reading"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise CartographyIntegrityError("cartography sidecar grew while reading")
        opened_after = os.fstat(descriptor)
        path_after = os.lstat(path)
        _validate_open_path_identity(
            opened_after, path_after, label="cartography sidecar"
        )
        if (
            _stable_signature(opened_before) != _stable_signature(opened_after)
            or _stable_signature(path_before) != _stable_signature(path_after)
            or not _same_inode(opened_after, path_after)
        ):
            raise CartographyIntegrityError("cartography sidecar changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_document(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise CartographyError(f"{label} is not a canonical identifier")
    return value


def _pin(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > 256
        or "\x00" in value
    ):
        raise CartographyError(f"{label} must be a non-empty bounded string")
    normalized = unicodedata.normalize("NFC", value)
    if normalized != value:
        raise CartographyError(f"{label} must be NFC-normalized")
    return value


def _sha256(value: Any, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CartographyError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CartographyError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: Any, label: str) -> int:
    value = _nonnegative_int(value, label)
    if value == 0:
        raise CartographyError(f"{label} must be positive")
    return value


def _nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CartographyError(f"{label} must be a non-negative finite number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise CartographyError(f"{label} must be a non-negative finite number")
    return result


def _optional_positive_int(value: Any, label: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, label)


def _optional_positive_float(value: Any, label: str) -> float | None:
    if value is None:
        return None
    result = _nonnegative_float(value, label)
    if result == 0.0:
        raise CartographyError(f"{label} must be positive")
    return result


def _normalize_json(value: Any) -> Any:
    nodes = [0]

    def visit(item: Any, depth: int) -> Any:
        nodes[0] += 1
        if nodes[0] > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise CartographyError("structured observation exceeds JSON bounds")
        if item is None or isinstance(item, bool):
            return item
        if isinstance(item, int) and not isinstance(item, bool):
            if not -(2**63) <= item < 2**63:
                raise CartographyError("structured observation integer is out of range")
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise CartographyError(
                    "structured observation contains NaN or infinity"
                )
            return item
        if isinstance(item, str):
            normalized = unicodedata.normalize("NFC", item)
            if normalized != item:
                raise CartographyError(
                    "structured observation strings must be NFC-normalized"
                )
            return item
        if isinstance(item, Mapping):
            normalized_mapping: dict[str, Any] = {}
            for raw_key, raw_value in item.items():
                if not isinstance(raw_key, str):
                    raise CartographyError(
                        "structured observation keys must be strings"
                    )
                key = unicodedata.normalize("NFC", raw_key)
                if key != raw_key or key in normalized_mapping:
                    raise CartographyError(
                        "structured observation keys are not canonical"
                    )
                normalized_mapping[key] = visit(raw_value, depth + 1)
            return normalized_mapping
        if isinstance(item, (list, tuple)):
            return [visit(member, depth + 1) for member in item]
        raise CartographyError("structured observation is not canonical JSON")

    return visit(value, 0)


def canonical_observation_bytes(observation: Mapping[str, Any]) -> bytes:
    """Return the exact UTF-8 bytes presented to the O1 surprise stream."""

    if not isinstance(observation, Mapping):
        raise CartographyError("a probe observation must be a mapping")
    return _canonical_json(_normalize_json(observation))


def _contains_forbidden_raw_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, member in value.items():
            if key.casefold() in _FORBIDDEN_RAW_KEYS:
                return True
            if _contains_forbidden_raw_key(member):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_raw_key(member) for member in value)
    return False


@dataclass(frozen=True, slots=True, order=True)
class ProbeTarget:
    """One module-local head, block, or unpartitioned module target."""

    module: str
    unit_kind: Literal["head", "block", "module"]
    unit_index: int | None

    def __post_init__(self) -> None:
        _identifier(self.module, "probe module")
        if self.unit_kind not in {"head", "block", "module"}:
            raise CartographyError("probe unit kind must be head, block, or module")
        if self.unit_kind == "module":
            if self.unit_index is not None:
                raise CartographyError(
                    "an unpartitioned module cannot have a unit index"
                )
        else:
            _nonnegative_int(self.unit_index, "probe unit index")

    def to_document(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "unit_index": self.unit_index,
            "unit_kind": self.unit_kind,
        }

    @classmethod
    def from_document(cls, raw: Any) -> ProbeTarget:
        if not isinstance(raw, Mapping) or set(raw) != {
            "module",
            "unit_index",
            "unit_kind",
        }:
            raise CartographyIntegrityError(
                "probe target has unknown or missing fields"
            )
        return cls(
            module=raw["module"],
            unit_kind=raw["unit_kind"],
            unit_index=raw["unit_index"],
        )


@dataclass(frozen=True, slots=True)
class ProbeJob:
    """One immutable cell in the finite cartography frontier."""

    job_id: str
    layer: int
    target: ProbeTarget
    probe_family: str
    intervention: str
    code_pin: str
    model_pin: str
    seed: int
    prompt_sha256: str | None = None
    prompt: str | None = None
    read_budget_bytes: int | None = None
    model_budget_seconds: float | None = None

    def __post_init__(self) -> None:
        _sha256(self.job_id, "probe job id")
        _nonnegative_int(self.layer, "probe layer")
        if not isinstance(self.target, ProbeTarget):
            raise CartographyError("probe target is invalid")
        _identifier(self.probe_family, "probe family")
        _identifier(self.intervention, "probe intervention")
        _pin(self.code_pin, "code pin")
        _pin(self.model_pin, "model pin")
        seed = _nonnegative_int(self.seed, "probe seed")
        if seed >= 2**63:
            raise CartographyError("probe seed must fit signed 64-bit range")
        prompt_digest = _sha256(
            self.prompt_sha256, "probe prompt SHA-256", optional=True
        )
        if self.prompt is not None:
            if not isinstance(self.prompt, str) or "\x00" in self.prompt:
                raise CartographyError("raw probe prompt is invalid")
            if hashlib.sha256(self.prompt.encode("utf-8")).hexdigest() != prompt_digest:
                raise CartographyError("raw probe prompt does not match its SHA-256")
        _optional_positive_int(self.read_budget_bytes, "probe read budget")
        _optional_positive_float(self.model_budget_seconds, "probe model budget")
        if self.job_id != _sha256_document(self._identity_document()):
            raise CartographyError("probe job id does not match its canonical identity")

    def _identity_document(self) -> dict[str, Any]:
        return {
            "code_pin": self.code_pin,
            "intervention": self.intervention,
            "layer": self.layer,
            "model_budget_seconds": self.model_budget_seconds,
            "model_pin": self.model_pin,
            "probe_family": self.probe_family,
            "prompt_sha256": self.prompt_sha256,
            "read_budget_bytes": self.read_budget_bytes,
            "seed": self.seed,
            "target": self.target.to_document(),
        }

    @classmethod
    def create(
        cls,
        *,
        layer: int,
        target: ProbeTarget,
        probe_family: str,
        intervention: str,
        code_pin: str,
        model_pin: str,
        seed: int,
        prompt_sha256: str | None = None,
        prompt: str | None = None,
        read_budget_bytes: int | None = None,
        model_budget_seconds: float | None = None,
    ) -> ProbeJob:
        if prompt is not None and prompt_sha256 is None:
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        body = {
            "code_pin": code_pin,
            "intervention": intervention,
            "layer": layer,
            "model_budget_seconds": model_budget_seconds,
            "model_pin": model_pin,
            "probe_family": probe_family,
            "prompt_sha256": prompt_sha256,
            "read_budget_bytes": read_budget_bytes,
            "seed": seed,
            "target": target.to_document(),
        }
        return cls(
            job_id=_sha256_document(body),
            layer=layer,
            target=target,
            probe_family=probe_family,
            intervention=intervention,
            code_pin=code_pin,
            model_pin=model_pin,
            seed=seed,
            prompt_sha256=prompt_sha256,
            prompt=prompt,
            read_budget_bytes=read_budget_bytes,
            model_budget_seconds=model_budget_seconds,
        )

    def to_document(self, *, include_raw_prompt: bool = False) -> dict[str, Any]:
        document = {
            "code_pin": self.code_pin,
            "intervention": self.intervention,
            "job_id": self.job_id,
            "layer": self.layer,
            "model_budget_seconds": self.model_budget_seconds,
            "model_pin": self.model_pin,
            "probe_family": self.probe_family,
            "prompt_sha256": self.prompt_sha256,
            "read_budget_bytes": self.read_budget_bytes,
            "schema": PROBE_JOB_SCHEMA,
            "seed": self.seed,
            "target": self.target.to_document(),
        }
        if include_raw_prompt:
            document["prompt"] = self.prompt
        return document

    @classmethod
    def from_document(cls, raw: Any, *, allow_raw_prompt: bool = False) -> ProbeJob:
        required = {
            "code_pin",
            "intervention",
            "job_id",
            "layer",
            "model_budget_seconds",
            "model_pin",
            "probe_family",
            "prompt_sha256",
            "read_budget_bytes",
            "schema",
            "seed",
            "target",
        }
        allowed = required | ({"prompt"} if allow_raw_prompt else set())
        if not isinstance(raw, Mapping) or set(raw) != allowed:
            raise CartographyIntegrityError("probe job has unknown or missing fields")
        if raw["schema"] != PROBE_JOB_SCHEMA:
            raise CartographyIntegrityError("probe job schema mismatch")
        return cls(
            job_id=raw["job_id"],
            layer=raw["layer"],
            target=ProbeTarget.from_document(raw["target"]),
            probe_family=raw["probe_family"],
            intervention=raw["intervention"],
            code_pin=raw["code_pin"],
            model_pin=raw["model_pin"],
            seed=raw["seed"],
            prompt_sha256=raw["prompt_sha256"],
            prompt=raw.get("prompt"),
            read_budget_bytes=raw["read_budget_bytes"],
            model_budget_seconds=raw["model_budget_seconds"],
        )


def build_probe_frontier(
    *,
    layers: Iterable[int],
    targets: Iterable[ProbeTarget],
    probe_families: Iterable[str],
    interventions: Iterable[str],
    code_pin: str,
    model_pin: str,
    seed: int,
    prompt_hashes_by_family: Mapping[str, str] | None = None,
    raw_prompts_by_family: Mapping[str, str] | None = None,
    read_budget_bytes: int | None = None,
    model_budget_seconds: float | None = None,
) -> tuple[ProbeJob, ...]:
    """Materialize the finite layer x target x family x intervention frontier."""

    layer_axis = tuple(layers)
    target_axis = tuple(targets)
    family_axis = tuple(probe_families)
    intervention_axis = tuple(interventions)
    for label, axis in (
        ("layer", layer_axis),
        ("target", target_axis),
        ("probe family", family_axis),
        ("intervention", intervention_axis),
    ):
        if len(set(axis)) != len(axis):
            raise CartographyError(f"duplicate {label} in probe frontier")
    for layer in layer_axis:
        _nonnegative_int(layer, "probe layer")
    for target in target_axis:
        if not isinstance(target, ProbeTarget):
            raise CartographyError("probe frontier target is invalid")
    for family in family_axis:
        _identifier(family, "probe family")
    for intervention in intervention_axis:
        _identifier(intervention, "probe intervention")
    _pin(code_pin, "code pin")
    _pin(model_pin, "model pin")
    base_seed = _nonnegative_int(seed, "cartographer seed")
    if base_seed >= 2**63:
        raise CartographyError("cartographer seed must fit signed 64-bit range")
    prompt_hashes = dict(prompt_hashes_by_family or {})
    raw_prompts = dict(raw_prompts_by_family or {})
    unknown = (set(prompt_hashes) | set(raw_prompts)) - set(family_axis)
    if unknown:
        raise CartographyError("prompt mapping contains an unknown probe family")

    jobs: list[ProbeJob] = []
    for layer in sorted(layer_axis):
        for target in sorted(target_axis):
            for family in sorted(family_axis):
                for intervention in sorted(intervention_axis):
                    identity = {
                        "base_seed": base_seed,
                        "intervention": intervention,
                        "layer": layer,
                        "probe_family": family,
                        "target": target.to_document(),
                    }
                    job_seed = int(_sha256_document(identity)[:16], 16) % (2**63)
                    prompt = raw_prompts.get(family)
                    prompt_sha256 = prompt_hashes.get(family)
                    if prompt is not None:
                        calculated = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                        if prompt_sha256 is not None and prompt_sha256 != calculated:
                            raise CartographyError(
                                f"raw prompt for {family!r} does not match its hash"
                            )
                        prompt_sha256 = calculated
                    jobs.append(
                        ProbeJob.create(
                            layer=layer,
                            target=target,
                            probe_family=family,
                            intervention=intervention,
                            code_pin=code_pin,
                            model_pin=model_pin,
                            seed=job_seed,
                            prompt_sha256=prompt_sha256,
                            prompt=prompt,
                            read_budget_bytes=read_budget_bytes,
                            model_budget_seconds=model_budget_seconds,
                        )
                    )
    return tuple(jobs)


Status = Literal["succeeded", "failed", "aborted"]


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    """One immutable attempt result; observations remain measurements, not truth."""

    job_id: str
    attempt: int
    attempt_id: str
    status: Status
    observation_bytes: bytes | None = None
    error_code: str | None = None
    error_sha256: str | None = None
    error_message: str | None = None
    read_bytes: int = 0
    model_seconds: float = 0.0
    wall_seconds: float = 0.0
    atlas_receipt_sha256: str | None = None
    o1_loss_before: float | None = None
    o1_loss_after: float | None = None
    surprise: float = 0.0
    learning_progress: float = 0.0

    def __post_init__(self) -> None:
        _sha256(self.job_id, "outcome job id")
        _positive_int(self.attempt, "outcome attempt")
        _sha256(self.attempt_id, "outcome attempt id")
        expected_attempt_id = _sha256_document(
            {"attempt": self.attempt, "job_id": self.job_id}
        )
        if self.attempt_id != expected_attempt_id:
            raise CartographyError("outcome attempt id does not match job and attempt")
        if self.status not in {"succeeded", "failed", "aborted"}:
            raise CartographyError("probe outcome status is invalid")
        if self.observation_bytes is not None:
            if not isinstance(self.observation_bytes, bytes):
                raise CartographyError("outcome observation must be canonical bytes")
            try:
                observation = json.loads(self.observation_bytes)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CartographyError("outcome observation is invalid JSON") from exc
            if canonical_observation_bytes(observation) != self.observation_bytes:
                raise CartographyError("outcome observation bytes are not canonical")
        if self.status == "succeeded" and self.observation_bytes is None:
            raise CartographyError("a successful probe requires an observation")
        if self.error_code is not None:
            _identifier(self.error_code, "outcome error code")
        _sha256(self.error_sha256, "outcome error SHA-256", optional=True)
        if self.error_message is not None and not isinstance(self.error_message, str):
            raise CartographyError("outcome error message must be text")
        _nonnegative_int(self.read_bytes, "outcome read bytes")
        _nonnegative_float(self.model_seconds, "outcome model seconds")
        _nonnegative_float(self.wall_seconds, "outcome wall seconds")
        receipt = _sha256(
            self.atlas_receipt_sha256,
            "outcome atlas receipt SHA-256",
            optional=True,
        )
        if receipt is not None and self.status != "succeeded":
            raise CartographyError("only a successful probe can carry an atlas receipt")
        for value, label in (
            (self.o1_loss_before, "O1 loss before"),
            (self.o1_loss_after, "O1 loss after"),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise CartographyError(f"{label} must be finite")
        _nonnegative_float(self.surprise, "outcome surprise")
        _nonnegative_float(self.learning_progress, "outcome learning progress")

    @staticmethod
    def attempt_identity(job_id: str, attempt: int) -> str:
        return _sha256_document({"attempt": attempt, "job_id": job_id})

    @classmethod
    def succeeded(
        cls,
        job: ProbeJob,
        attempt: int,
        observation: Mapping[str, Any],
        *,
        read_bytes: int = 0,
        model_seconds: float = 0.0,
        wall_seconds: float = 0.0,
        atlas_receipt_sha256: str | None = None,
    ) -> ProbeOutcome:
        return cls(
            job_id=job.job_id,
            attempt=attempt,
            attempt_id=cls.attempt_identity(job.job_id, attempt),
            status="succeeded",
            observation_bytes=canonical_observation_bytes(observation),
            read_bytes=read_bytes,
            model_seconds=model_seconds,
            wall_seconds=wall_seconds,
            atlas_receipt_sha256=atlas_receipt_sha256,
        )

    @classmethod
    def failed(
        cls,
        job: ProbeJob,
        attempt: int,
        *,
        error_code: str,
        error: str = "",
        observation: Mapping[str, Any] | None = None,
        read_bytes: int = 0,
        model_seconds: float = 0.0,
        wall_seconds: float = 0.0,
    ) -> ProbeOutcome:
        return cls(
            job_id=job.job_id,
            attempt=attempt,
            attempt_id=cls.attempt_identity(job.job_id, attempt),
            status="failed",
            observation_bytes=(
                canonical_observation_bytes(observation)
                if observation is not None
                else None
            ),
            error_code=error_code,
            error_sha256=_sha256_bytes(error.encode("utf-8")),
            error_message=error or None,
            read_bytes=read_bytes,
            model_seconds=model_seconds,
            wall_seconds=wall_seconds,
        )

    @classmethod
    def aborted(
        cls,
        job: ProbeJob,
        attempt: int,
        *,
        error_code: str,
        error: str = "",
        read_bytes: int = 0,
        model_seconds: float = 0.0,
        wall_seconds: float = 0.0,
    ) -> ProbeOutcome:
        return cls(
            job_id=job.job_id,
            attempt=attempt,
            attempt_id=cls.attempt_identity(job.job_id, attempt),
            status="aborted",
            error_code=error_code,
            error_sha256=_sha256_bytes(error.encode("utf-8")),
            error_message=error or None,
            read_bytes=read_bytes,
            model_seconds=model_seconds,
            wall_seconds=wall_seconds,
        )

    @property
    def semantically_promoted(self) -> bool:
        return self.status == "succeeded" and self.atlas_receipt_sha256 is not None

    def observation_document(self) -> Mapping[str, Any] | None:
        if self.observation_bytes is None:
            return None
        value = json.loads(self.observation_bytes)
        if not isinstance(value, Mapping):  # protected by __post_init__
            raise CartographyIntegrityError("outcome observation is not a mapping")
        return value

    def to_document(self, *, include_error_message: bool = False) -> dict[str, Any]:
        return {
            "atlas_receipt_sha256": self.atlas_receipt_sha256,
            "attempt": self.attempt,
            "attempt_id": self.attempt_id,
            "error_code": self.error_code,
            "error_message": self.error_message if include_error_message else None,
            "error_sha256": self.error_sha256,
            "job_id": self.job_id,
            "learning_progress": self.learning_progress,
            "model_seconds": self.model_seconds,
            "o1_loss_after": self.o1_loss_after,
            "o1_loss_before": self.o1_loss_before,
            "observation": self.observation_document(),
            "read_bytes": self.read_bytes,
            "schema": PROBE_OUTCOME_SCHEMA,
            "status": self.status,
            "surprise": self.surprise,
            "wall_seconds": self.wall_seconds,
        }

    @classmethod
    def from_document(cls, raw: Any) -> ProbeOutcome:
        fields = {
            "atlas_receipt_sha256",
            "attempt",
            "attempt_id",
            "error_code",
            "error_message",
            "error_sha256",
            "job_id",
            "learning_progress",
            "model_seconds",
            "o1_loss_after",
            "o1_loss_before",
            "observation",
            "read_bytes",
            "schema",
            "status",
            "surprise",
            "wall_seconds",
        }
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise CartographyIntegrityError(
                "probe outcome has unknown or missing fields"
            )
        if raw["schema"] != PROBE_OUTCOME_SCHEMA:
            raise CartographyIntegrityError("probe outcome schema mismatch")
        observation = raw["observation"]
        return cls(
            job_id=raw["job_id"],
            attempt=raw["attempt"],
            attempt_id=raw["attempt_id"],
            status=raw["status"],
            observation_bytes=(
                canonical_observation_bytes(observation)
                if observation is not None
                else None
            ),
            error_code=raw["error_code"],
            error_sha256=raw["error_sha256"],
            error_message=raw["error_message"],
            read_bytes=raw["read_bytes"],
            model_seconds=raw["model_seconds"],
            wall_seconds=raw["wall_seconds"],
            atlas_receipt_sha256=raw["atlas_receipt_sha256"],
            o1_loss_before=raw["o1_loss_before"],
            o1_loss_after=raw["o1_loss_after"],
            surprise=raw["surprise"],
            learning_progress=raw["learning_progress"],
        )


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 2
    retry_failed: bool = True
    retry_aborted: bool = True

    def __post_init__(self) -> None:
        _positive_int(self.max_attempts, "maximum probe attempts")
        if not isinstance(self.retry_failed, bool) or not isinstance(
            self.retry_aborted, bool
        ):
            raise CartographyError("retry policy flags must be booleans")

    def to_document(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "retry_aborted": self.retry_aborted,
            "retry_failed": self.retry_failed,
        }

    @classmethod
    def from_document(cls, raw: Any) -> RetryPolicy:
        if not isinstance(raw, Mapping) or set(raw) != {
            "max_attempts",
            "retry_aborted",
            "retry_failed",
        }:
            raise CartographyIntegrityError("retry policy is malformed")
        return cls(
            max_attempts=raw["max_attempts"],
            retry_failed=raw["retry_failed"],
            retry_aborted=raw["retry_aborted"],
        )


@dataclass(frozen=True, slots=True)
class CartographyBudget:
    """Persistent aggregate and per-observation scheduler limits."""

    max_total_attempts: int | None = None
    max_total_read_bytes: int | None = None
    max_total_model_seconds: float | None = None
    max_observation_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        _optional_positive_int(self.max_total_attempts, "total attempt budget")
        _optional_positive_int(self.max_total_read_bytes, "total read budget")
        _optional_positive_float(
            self.max_total_model_seconds, "total model-time budget"
        )
        _positive_int(self.max_observation_bytes, "observation byte budget")

    def to_document(self) -> dict[str, Any]:
        return {
            "max_observation_bytes": self.max_observation_bytes,
            "max_total_attempts": self.max_total_attempts,
            "max_total_model_seconds": self.max_total_model_seconds,
            "max_total_read_bytes": self.max_total_read_bytes,
        }

    @classmethod
    def from_document(cls, raw: Any) -> CartographyBudget:
        if not isinstance(raw, Mapping) or set(raw) != {
            "max_observation_bytes",
            "max_total_attempts",
            "max_total_model_seconds",
            "max_total_read_bytes",
        }:
            raise CartographyIntegrityError("cartography budget is malformed")
        return cls(
            max_total_attempts=raw["max_total_attempts"],
            max_total_read_bytes=raw["max_total_read_bytes"],
            max_total_model_seconds=raw["max_total_model_seconds"],
            max_observation_bytes=raw["max_observation_bytes"],
        )


@dataclass(frozen=True, slots=True)
class Coverage:
    total_jobs: int
    succeeded_jobs: int
    uncovered_jobs: int
    retryable_jobs: int
    failed_terminal_jobs: int
    aborted_terminal_jobs: int
    in_flight_jobs: int
    attempts: int
    promoted_jobs: int

    @property
    def terminal_jobs(self) -> int:
        return (
            self.succeeded_jobs + self.failed_terminal_jobs + self.aborted_terminal_jobs
        )

    @property
    def complete(self) -> bool:
        return self.terminal_jobs == self.total_jobs and self.in_flight_jobs == 0

    @property
    def fraction(self) -> float:
        return 1.0 if self.total_jobs == 0 else self.terminal_jobs / self.total_jobs

    def to_document(self) -> dict[str, Any]:
        return {
            "aborted_terminal_jobs": self.aborted_terminal_jobs,
            "attempts": self.attempts,
            "complete": self.complete,
            "failed_terminal_jobs": self.failed_terminal_jobs,
            "fraction": self.fraction,
            "in_flight_jobs": self.in_flight_jobs,
            "promoted_jobs": self.promoted_jobs,
            "retryable_jobs": self.retryable_jobs,
            "schema": COVERAGE_SCHEMA,
            "succeeded_jobs": self.succeeded_jobs,
            "terminal_jobs": self.terminal_jobs,
            "total_jobs": self.total_jobs,
            "uncovered_jobs": self.uncovered_jobs,
        }


@dataclass(frozen=True, slots=True)
class RunResult:
    outcomes: tuple[ProbeOutcome, ...]
    elapsed_seconds: float
    stop_reason: str
    coverage: Coverage


ProbeExecutor = Callable[[ProbeJob, int], ProbeOutcome | Mapping[str, Any]]


class O1Cartographer:
    """Deterministic finite scheduler with exact, tamper-evident resume state."""

    def __init__(
        self,
        *,
        jobs: Sequence[ProbeJob],
        code_pin: str,
        model_pin: str,
        state_path: str | Path,
        stream: SurpriseStream | None = None,
        seed: int = 0,
        retry_policy: RetryPolicy | None = None,
        budget: CartographyBudget | None = None,
        rolling_window: int = 8,
        replay_items: int = 64,
        replay_bytes: int = 8 * 1024 * 1024,
        allow_raw_prompts: bool = False,
    ) -> None:
        self.code_pin = _pin(code_pin, "code pin")
        self.model_pin = _pin(model_pin, "model pin")
        self.seed = _nonnegative_int(seed, "cartographer seed")
        if self.seed >= 2**63:
            raise CartographyError("cartographer seed must fit signed 64-bit range")
        self.retry_policy = retry_policy or RetryPolicy()
        self.budget = budget or CartographyBudget()
        self.rolling_window = _positive_int(rolling_window, "rolling window")
        self.replay_items = _nonnegative_int(replay_items, "replay item limit")
        self.replay_bytes = _nonnegative_int(replay_bytes, "replay byte limit")
        if not isinstance(allow_raw_prompts, bool):
            raise CartographyError("allow_raw_prompts must be a boolean")
        self.allow_raw_prompts = allow_raw_prompts
        self.state_path = Path(state_path).expanduser()
        if self.state_path.exists() and self.state_path.is_dir():
            raise CartographyIntegrityError("cartography state path is a directory")
        normalized_jobs = tuple(jobs)
        if any(not isinstance(job, ProbeJob) for job in normalized_jobs):
            raise CartographyError("cartography jobs must be ProbeJob instances")
        ids = [job.job_id for job in normalized_jobs]
        if len(set(ids)) != len(ids):
            raise CartographyError("cartography frontier contains duplicate jobs")
        if any(
            job.code_pin != self.code_pin or job.model_pin != self.model_pin
            for job in normalized_jobs
        ):
            raise CartographyIdentityError(
                "probe frontier pins do not match scheduler pins"
            )
        if self.budget.max_total_read_bytes is not None and any(
            job.read_budget_bytes is None for job in normalized_jobs
        ):
            raise CartographyError(
                "a total read budget requires a read cap on every probe job"
            )
        if self.budget.max_total_model_seconds is not None and any(
            job.model_budget_seconds is None for job in normalized_jobs
        ):
            raise CartographyError(
                "a total model-time budget requires a model-time cap on every probe job"
            )
        if not self.allow_raw_prompts and any(
            job.prompt is not None for job in normalized_jobs
        ):
            raise CartographyError(
                "raw probe prompts are disabled; provide only hashes"
            )
        self.jobs = tuple(sorted(normalized_jobs, key=lambda job: job.job_id))
        self._jobs_by_id = {job.job_id: job for job in self.jobs}
        self._outcomes: list[ProbeOutcome] = []
        self._in_flight: tuple[str, int, str] | None = None
        self._histories: dict[tuple[str, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=self.rolling_window)
        )
        self._replay: deque[bytes] = deque()
        self._replay_size = 0
        self._stream_state: Mapping[str, Any] | None = None
        self._generation = 0
        self._file_sha256: str | None = None
        self._last_stop_reason = "ready"
        self._lock = threading.RLock()
        if stream is None:
            from .adapter import O1StateStream

            stream = O1StateStream(seed=self.seed)
        self.stream = stream

        if self.state_path.exists() or self.state_path.is_symlink():
            self._restore_from_disk()
        else:
            self.checkpoint()

    @classmethod
    def restore(
        cls,
        state_path: str | Path,
        *,
        code_pin: str,
        model_pin: str,
        stream: SurpriseStream | None = None,
    ) -> O1Cartographer:
        path = Path(state_path).expanduser()
        with _exclusive_state_lock(path):
            raw, document = cls._read_document(path)
        config = document.get("config")
        if not isinstance(config, Mapping):
            raise CartographyIntegrityError("cartography config is missing")
        if config.get("code_pin") != code_pin or config.get("model_pin") != model_pin:
            raise CartographyIdentityError("cartography code/model pin is stale")
        allow_raw = config.get("allow_raw_prompts")
        if not isinstance(allow_raw, bool):
            raise CartographyIntegrityError("raw-prompt policy is malformed")
        jobs_raw = document.get("jobs")
        if not isinstance(jobs_raw, list):
            raise CartographyIntegrityError("cartography jobs are malformed")
        jobs = tuple(
            ProbeJob.from_document(job, allow_raw_prompt=allow_raw) for job in jobs_raw
        )
        instance = cls(
            jobs=jobs,
            code_pin=code_pin,
            model_pin=model_pin,
            state_path=path,
            stream=stream,
            seed=config.get("seed"),
            retry_policy=RetryPolicy.from_document(config.get("retry_policy")),
            budget=CartographyBudget.from_document(config.get("budget")),
            rolling_window=config.get("rolling_window"),
            replay_items=config.get("replay_items"),
            replay_bytes=config.get("replay_bytes"),
            allow_raw_prompts=allow_raw,
        )
        # The constructor authenticated the same bytes. Keep the read alive as
        # an explicit guard against a path swap between the two operations.
        if document.get("in_flight") is None and instance._file_sha256 != _sha256_bytes(
            raw
        ):
            raise CartographyIntegrityError(
                "cartography sidecar changed during restore"
            )
        return instance

    @property
    def outcomes(self) -> tuple[ProbeOutcome, ...]:
        with self._lock:
            return tuple(self._outcomes)

    @property
    def replay_count(self) -> int:
        with self._lock:
            return len(self._replay)

    def _config_document(self) -> dict[str, Any]:
        return {
            "allow_raw_prompts": self.allow_raw_prompts,
            "budget": self.budget.to_document(),
            "code_pin": self.code_pin,
            "model_pin": self.model_pin,
            "replay_bytes": self.replay_bytes,
            "replay_items": self.replay_items,
            "retry_policy": self.retry_policy.to_document(),
            "rolling_window": self.rolling_window,
            "seed": self.seed,
        }

    def _features(self, job: ProbeJob) -> tuple[tuple[str, str], ...]:
        target = (
            f"{job.layer}:{job.target.module}:{job.target.unit_kind}:"
            f"{job.target.unit_index}"
        )
        return (
            ("probe_family", job.probe_family),
            ("intervention", job.intervention),
            ("module", job.target.module),
            ("target", target),
        )

    def _priority(self, job: ProbeJob) -> float:
        weighted = 0.0
        total_weight = 0.0
        for feature in self._features(job):
            weight = 2.0 if feature[0] == "probe_family" else 1.0
            values = self._histories.get(feature)
            if values:
                weighted += weight * sum(values) / len(values)
            total_weight += weight
        return weighted / total_weight

    def _tie_break(self, job: ProbeJob) -> str:
        return _sha256_document({"job_id": job.job_id, "seed": self.seed})

    def _outcomes_by_job(self) -> dict[str, list[ProbeOutcome]]:
        grouped: dict[str, list[ProbeOutcome]] = defaultdict(list)
        for outcome in self._outcomes:
            grouped[outcome.job_id].append(outcome)
        return grouped

    def _retryable(self, attempts: Sequence[ProbeOutcome]) -> bool:
        if not attempts:
            return True
        if len(attempts) >= self.retry_policy.max_attempts:
            return False
        latest = attempts[-1]
        if latest.status == "succeeded":
            return False
        if latest.status == "failed":
            return self.retry_policy.retry_failed
        return self.retry_policy.retry_aborted

    def coverage(self) -> Coverage:
        with self._lock:
            grouped = self._outcomes_by_job()
            succeeded = uncovered = retryable = failed = aborted = promoted = 0
            for job in self.jobs:
                attempts = grouped.get(job.job_id, ())
                if not attempts:
                    uncovered += 1
                    continue
                latest = attempts[-1]
                if latest.status == "succeeded":
                    succeeded += 1
                    promoted += int(latest.semantically_promoted)
                elif self._retryable(attempts):
                    retryable += 1
                elif latest.status == "failed":
                    failed += 1
                else:
                    aborted += 1
            return Coverage(
                total_jobs=len(self.jobs),
                succeeded_jobs=succeeded,
                uncovered_jobs=uncovered,
                retryable_jobs=retryable,
                failed_terminal_jobs=failed,
                aborted_terminal_jobs=aborted,
                in_flight_jobs=int(self._in_flight is not None),
                attempts=len(self._outcomes) + int(self._in_flight is not None),
                promoted_jobs=promoted,
            )

    def _totals(self) -> tuple[int, int, float]:
        return (
            len(self._outcomes) + int(self._in_flight is not None),
            sum(outcome.read_bytes for outcome in self._outcomes),
            sum(outcome.model_seconds for outcome in self._outcomes),
        )

    def _fits_budget(self, job: ProbeJob) -> bool:
        attempts, read_bytes, model_seconds = self._totals()
        if (
            self.budget.max_total_attempts is not None
            and attempts >= self.budget.max_total_attempts
        ):
            return False
        if (
            self.budget.max_total_read_bytes is not None
            and job.read_budget_bytes is not None
            and read_bytes + job.read_budget_bytes > self.budget.max_total_read_bytes
        ):
            return False
        if (
            self.budget.max_total_model_seconds is not None
            and job.model_budget_seconds is not None
            and model_seconds + job.model_budget_seconds
            > self.budget.max_total_model_seconds
        ):
            return False
        if (
            self.budget.max_total_read_bytes is not None
            and read_bytes >= self.budget.max_total_read_bytes
        ):
            return False
        if (
            self.budget.max_total_model_seconds is not None
            and model_seconds >= self.budget.max_total_model_seconds
        ):
            return False
        return True

    def next_job(self) -> ProbeJob | None:
        with self._lock:
            if self._in_flight is not None:
                raise CartographyError("a probe attempt is already in flight")
            grouped = self._outcomes_by_job()
            candidates: list[tuple[int, float, str, str, ProbeJob]] = []
            had_schedulable = False
            for job in self.jobs:
                attempts = grouped.get(job.job_id, ())
                if not self._retryable(attempts):
                    continue
                had_schedulable = True
                if not self._fits_budget(job):
                    continue
                uncovered_rank = 0 if not attempts else 1
                candidates.append(
                    (
                        uncovered_rank,
                        -self._priority(job),
                        self._tie_break(job),
                        job.job_id,
                        job,
                    )
                )
            if not candidates:
                self._last_stop_reason = (
                    "budget" if had_schedulable else "coverage-complete"
                )
                return None
            self._last_stop_reason = "ready"
            return min(candidates)[-1]

    def _measure_o1(
        self, observation: bytes
    ) -> tuple[float | None, float | None, float, float]:
        before_raw = getattr(self.stream, "loss_ema", None)
        before = float(before_raw) if before_raw is not None else None
        surprises_before = int(getattr(self.stream, "surprises", 0))
        updates_before = int(getattr(self.stream, "updates", 0))
        self.stream.observe(observation.decode("utf-8"))
        after_raw = getattr(self.stream, "loss_ema", None)
        after = float(after_raw) if after_raw is not None else None
        surprises_after = int(getattr(self.stream, "surprises", surprises_before))
        updates_after = int(getattr(self.stream, "updates", updates_before))
        if before is None or after is None:
            delta = 0.0
        else:
            delta = after - before
        surprise = max(0.0, delta) + max(0, surprises_after - surprises_before)
        learning_progress = max(0.0, -delta) + max(0, updates_after - updates_before)
        return before, after, surprise, learning_progress

    def _append_replay(self, observation: bytes) -> None:
        if self.replay_items == 0 or self.replay_bytes == 0:
            return
        if len(observation) > self.replay_bytes:
            return
        self._replay.append(observation)
        self._replay_size += len(observation)
        while (
            len(self._replay) > self.replay_items
            or self._replay_size > self.replay_bytes
        ):
            self._replay_size -= len(self._replay.popleft())

    def _sanitize_outcome(self, job: ProbeJob, outcome: ProbeOutcome) -> ProbeOutcome:
        if outcome.job_id != job.job_id:
            raise CartographyError("executor returned an outcome for another job")
        expected_attempt = (
            len([item for item in self._outcomes if item.job_id == job.job_id]) + 1
        )
        if outcome.attempt != expected_attempt:
            raise CartographyError("executor returned the wrong probe attempt number")
        if outcome.atlas_receipt_sha256 is not None:
            raise CartographyError(
                "atlas receipts must be attached after probe execution"
            )
        if not self.allow_raw_prompts:
            outcome = replace(outcome, error_message=None)
        if outcome.observation_bytes is not None:
            observation = outcome.observation_document()
            if not self.allow_raw_prompts and _contains_forbidden_raw_key(observation):
                return ProbeOutcome.failed(
                    job,
                    outcome.attempt,
                    error_code="raw-observation-rejected",
                    error="structured observation contains a raw-text field",
                    read_bytes=outcome.read_bytes,
                    model_seconds=outcome.model_seconds,
                    wall_seconds=outcome.wall_seconds,
                )
            if len(outcome.observation_bytes) > self.budget.max_observation_bytes:
                return ProbeOutcome.failed(
                    job,
                    outcome.attempt,
                    error_code="observation-budget-exceeded",
                    error="structured observation exceeds its byte budget",
                    read_bytes=outcome.read_bytes,
                    model_seconds=outcome.model_seconds,
                    wall_seconds=outcome.wall_seconds,
                )
        if (
            job.read_budget_bytes is not None
            and outcome.read_bytes > job.read_budget_bytes
        ) or (
            job.model_budget_seconds is not None
            and outcome.model_seconds > job.model_budget_seconds
        ):
            return ProbeOutcome.failed(
                job,
                outcome.attempt,
                error_code="job-budget-exceeded",
                error="executor exceeded the job's declared budget",
                read_bytes=outcome.read_bytes,
                model_seconds=outcome.model_seconds,
                wall_seconds=outcome.wall_seconds,
            )
        return outcome

    def step(self, executor: ProbeExecutor) -> ProbeOutcome | None:
        """Execute and durably commit at most one probe attempt."""

        with self._lock:
            job = self.next_job()
            if job is None:
                return None
            attempt = sum(item.job_id == job.job_id for item in self._outcomes) + 1
            attempt_id = ProbeOutcome.attempt_identity(job.job_id, attempt)
            self._in_flight = (job.job_id, attempt, attempt_id)
            self._generation += 1
            self._checkpoint(capture_stream=False)

            started = time.monotonic()
            try:
                result = executor(job, attempt)
                if isinstance(result, ProbeOutcome):
                    outcome = result
                elif isinstance(result, Mapping):
                    outcome = ProbeOutcome.succeeded(job, attempt, result)
                else:
                    raise CartographyError(
                        "probe executor must return ProbeOutcome or an observation mapping"
                    )
            except Exception as exc:
                outcome = ProbeOutcome.failed(
                    job,
                    attempt,
                    error_code="executor-exception",
                    error=f"{type(exc).__name__}:{exc}",
                    wall_seconds=time.monotonic() - started,
                )
            outcome = self._sanitize_outcome(job, outcome)
            if outcome.observation_bytes is not None:
                before, after, surprise, progress = self._measure_o1(
                    outcome.observation_bytes
                )
                outcome = replace(
                    outcome,
                    o1_loss_before=before,
                    o1_loss_after=after,
                    surprise=surprise,
                    learning_progress=progress,
                )
                signal = surprise + progress
                for feature in self._features(job):
                    self._histories[feature].append(signal)
                self._append_replay(outcome.observation_bytes)
            self._outcomes.append(outcome)
            self._in_flight = None
            self._last_stop_reason = "ready"
            self._generation += 1
            self._checkpoint(capture_stream=True)
            return outcome

    def run(
        self,
        executor: ProbeExecutor,
        *,
        max_jobs: int,
        max_seconds: float,
    ) -> RunResult:
        """Run a bounded number of one-shot steps; this never starts a daemon."""

        _nonnegative_int(max_jobs, "run job limit")
        seconds = _nonnegative_float(max_seconds, "run time limit")
        started = time.monotonic()
        outcomes: list[ProbeOutcome] = []
        stop_reason = "max-jobs"
        while len(outcomes) < max_jobs:
            if time.monotonic() - started >= seconds:
                stop_reason = "max-seconds"
                break
            outcome = self.step(executor)
            if outcome is None:
                stop_reason = self._last_stop_reason
                break
            outcomes.append(outcome)
        elapsed = time.monotonic() - started
        return RunResult(
            outcomes=tuple(outcomes),
            elapsed_seconds=elapsed,
            stop_reason=stop_reason,
            coverage=self.coverage(),
        )

    def replay(self, *, max_items: int | None = None) -> int:
        """Replay bounded canonical observations into O1 without changing truth state."""

        if max_items is not None:
            _nonnegative_int(max_items, "replay limit")
        with self._lock:
            items = tuple(self._replay)
            if max_items is not None:
                items = items[-max_items:] if max_items else ()
            for observation in items:
                self.stream.observe(observation.decode("utf-8"))
            return len(items)

    def attach_atlas_receipt(
        self, *, job_id: str, attempt: int, receipt_sha256: str
    ) -> ProbeOutcome:
        """Bind an externally produced atlas receipt to a successful outcome."""

        _sha256(job_id, "atlas receipt job id")
        _positive_int(attempt, "atlas receipt attempt")
        receipt = _sha256(receipt_sha256, "atlas receipt SHA-256")
        with self._lock:
            for index, outcome in enumerate(self._outcomes):
                if outcome.job_id != job_id or outcome.attempt != attempt:
                    continue
                if outcome.status != "succeeded":
                    raise CartographyError(
                        "a failed probe cannot receive an atlas receipt"
                    )
                if outcome.atlas_receipt_sha256 not in {None, receipt}:
                    raise CartographyIntegrityError("atlas receipt cannot be replaced")
                updated = replace(outcome, atlas_receipt_sha256=receipt)
                self._outcomes[index] = updated
                self._generation += 1
                self._checkpoint(capture_stream=False)
                return updated
        raise CartographyError("atlas receipt target outcome does not exist")

    def checkpoint(self) -> str:
        with self._lock:
            self._generation += 1
            return self._checkpoint(capture_stream=True)

    def _state_body(self) -> dict[str, Any]:
        histories = [
            {"kind": kind, "name": name, "values": list(values)}
            for (kind, name), values in sorted(self._histories.items())
        ]
        replay = [
            {
                "bytes_b64": base64.b64encode(value).decode("ascii"),
                "sha256": _sha256_bytes(value),
            }
            for value in self._replay
        ]
        in_flight = None
        if self._in_flight is not None:
            in_flight = {
                "attempt": self._in_flight[1],
                "attempt_id": self._in_flight[2],
                "job_id": self._in_flight[0],
            }
        return {
            "config": self._config_document(),
            "generation": self._generation,
            "histories": histories,
            "in_flight": in_flight,
            "jobs": [
                job.to_document(include_raw_prompt=self.allow_raw_prompts)
                for job in self.jobs
            ],
            "last_stop_reason": self._last_stop_reason,
            "outcomes": [
                outcome.to_document(include_error_message=self.allow_raw_prompts)
                for outcome in self._outcomes
            ],
            "replay": replay,
            "schema": CARTOGRAPHER_STATE_SCHEMA,
            "stream_state": self._stream_state,
        }

    def _checkpoint(self, *, capture_stream: bool) -> str:
        with _exclusive_state_lock(self.state_path):
            return self._checkpoint_locked(capture_stream=capture_stream)

    def _checkpoint_locked(self, *, capture_stream: bool) -> str:
        if capture_stream:
            snapshot = self.stream.snapshot()
            if not isinstance(snapshot, Mapping):
                raise CartographyError("O1 stream snapshot must be a mapping")
            self._stream_state = _normalize_json(snapshot)
            if not self.allow_raw_prompts and _contains_forbidden_raw_key(
                self._stream_state
            ):
                raise CartographyError("O1 stream snapshot contains a raw-text field")
        body = self._state_body()
        document = {**body, "state_sha256": _sha256_document(body)}
        encoded = _canonical_json(document) + b"\n"
        if len(encoded) > _MAX_STATE_BYTES:
            raise CartographyError(
                f"cartography state exceeds {_MAX_STATE_BYTES} bytes"
            )
        existing_document = self._read_document_if_present(self.state_path)
        if existing_document is not None:
            existing, _ = existing_document
            if (
                self._file_sha256 is not None
                and _sha256_bytes(existing) != self._file_sha256
            ):
                raise CartographyIntegrityError(
                    "cartography sidecar changed since the last checkpoint"
                )
            if self._file_sha256 is None:
                if existing == encoded:
                    self._file_sha256 = _sha256_bytes(existing)
                    return document["state_sha256"]
                raise CartographyIntegrityError(
                    "cartography sidecar appeared during initialization"
                )
        elif self._file_sha256 is not None:
            raise CartographyIntegrityError(
                "cartography sidecar disappeared since the last checkpoint"
            )
        self._atomic_write(self.state_path, encoded)
        committed, _ = self._read_document(self.state_path)
        if committed != encoded:
            raise CartographyIntegrityError(
                "cartography sidecar commit does not match staged state"
            )
        self._file_sha256 = _sha256_bytes(committed)
        return document["state_sha256"]

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            metadata = None
        except OSError as exc:
            raise CartographyIntegrityError(
                "cartography sidecar target is unavailable"
            ) from exc
        if metadata is not None:
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise CartographyIntegrityError(
                    "cartography sidecar is not a regular file"
                )
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".pending", dir=path.parent
        )
        temporary_path = Path(temporary)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _read_document(path: Path) -> tuple[bytes, Mapping[str, Any]]:
        try:
            raw = _read_stable_regular_file(path)
        except OSError as exc:
            raise CartographyIntegrityError(
                "cartography sidecar is unavailable"
            ) from exc
        if not raw.endswith(b"\n"):
            raise CartographyIntegrityError(
                "cartography sidecar is not canonical JSONL"
            )
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CartographyIntegrityError(
                "cartography sidecar is invalid JSON"
            ) from exc
        if not isinstance(document, Mapping):
            raise CartographyIntegrityError("cartography sidecar root is not a mapping")
        if raw != _canonical_json(document) + b"\n":
            raise CartographyIntegrityError("cartography sidecar is not canonical")
        if document.get("schema") != CARTOGRAPHER_STATE_SCHEMA:
            raise CartographyIntegrityError("cartography sidecar schema mismatch")
        claimed = document.get("state_sha256")
        _sha256(claimed, "cartography state SHA-256")
        body = dict(document)
        del body["state_sha256"]
        if _sha256_document(body) != claimed:
            raise CartographyIntegrityError("cartography sidecar digest mismatch")
        return raw, document

    @staticmethod
    def _read_document_if_present(
        path: Path,
    ) -> tuple[bytes, Mapping[str, Any]] | None:
        try:
            return O1Cartographer._read_document(path)
        except CartographyIntegrityError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                return None
            raise

    def _restore_from_disk(self) -> None:
        with _exclusive_state_lock(self.state_path):
            self._restore_from_disk_locked()

    def _restore_from_disk_locked(self) -> None:
        raw, document = self._read_document(self.state_path)
        expected_fields = {
            "config",
            "generation",
            "histories",
            "in_flight",
            "jobs",
            "last_stop_reason",
            "outcomes",
            "replay",
            "schema",
            "state_sha256",
            "stream_state",
        }
        if set(document) != expected_fields:
            raise CartographyIntegrityError(
                "cartography sidecar has unknown or missing fields"
            )
        if document["config"] != self._config_document():
            raise CartographyIdentityError("cartography scheduler policy changed")
        disk_jobs = tuple(
            ProbeJob.from_document(job, allow_raw_prompt=self.allow_raw_prompts)
            for job in document["jobs"]
        )
        if disk_jobs != self.jobs:
            raise CartographyIdentityError("cartography frontier changed")
        self._generation = _nonnegative_int(document["generation"], "state generation")
        outcomes_raw = document["outcomes"]
        if not isinstance(outcomes_raw, list):
            raise CartographyIntegrityError("cartography outcomes are malformed")
        self._outcomes = [
            ProbeOutcome.from_document(raw_outcome) for raw_outcome in outcomes_raw
        ]
        grouped: dict[str, list[ProbeOutcome]] = defaultdict(list)
        attempt_ids: set[str] = set()
        for outcome in self._outcomes:
            if outcome.job_id not in self._jobs_by_id:
                raise CartographyIntegrityError(
                    "outcome references an unknown probe job"
                )
            if not self.allow_raw_prompts and outcome.error_message is not None:
                raise CartographyIntegrityError(
                    "raw executor error text is present while raw text is disabled"
                )
            if (
                not self.allow_raw_prompts
                and outcome.observation_bytes is not None
                and _contains_forbidden_raw_key(outcome.observation_document())
            ):
                raise CartographyIntegrityError(
                    "raw observation text is present while raw text is disabled"
                )
            if outcome.attempt_id in attempt_ids:
                raise CartographyIntegrityError("duplicate probe attempt in sidecar")
            attempt_ids.add(outcome.attempt_id)
            grouped[outcome.job_id].append(outcome)
        for attempts in grouped.values():
            if [item.attempt for item in attempts] != list(range(1, len(attempts) + 1)):
                raise CartographyIntegrityError("probe attempts are not contiguous")
            if any(item.status == "succeeded" for item in attempts[:-1]):
                raise CartographyIntegrityError("probe was repeated after success")
            if len(attempts) > self.retry_policy.max_attempts:
                raise CartographyIntegrityError("probe exceeded its retry policy")

        histories_raw = document["histories"]
        if not isinstance(histories_raw, list):
            raise CartographyIntegrityError("cartography histories are malformed")
        self._histories.clear()
        for row in histories_raw:
            if not isinstance(row, Mapping) or set(row) != {"kind", "name", "values"}:
                raise CartographyIntegrityError("cartography history row is malformed")
            feature = (_identifier(row["kind"], "history kind"), str(row["name"]))
            if feature in self._histories:
                raise CartographyIntegrityError("duplicate cartography history feature")
            values = row["values"]
            if not isinstance(values, list) or len(values) > self.rolling_window:
                raise CartographyIntegrityError(
                    "cartography history values are malformed"
                )
            history = deque(maxlen=self.rolling_window)
            history.extend(
                _nonnegative_float(value, "history signal") for value in values
            )
            self._histories[feature] = history

        replay_raw = document["replay"]
        if not isinstance(replay_raw, list) or len(replay_raw) > self.replay_items:
            raise CartographyIntegrityError("cartography replay buffer is malformed")
        self._replay.clear()
        self._replay_size = 0
        for row in replay_raw:
            if not isinstance(row, Mapping) or set(row) != {"bytes_b64", "sha256"}:
                raise CartographyIntegrityError("cartography replay row is malformed")
            try:
                value = base64.b64decode(row["bytes_b64"], validate=True)
            except (TypeError, ValueError) as exc:
                raise CartographyIntegrityError(
                    "cartography replay bytes are invalid"
                ) from exc
            if _sha256_bytes(value) != row["sha256"]:
                raise CartographyIntegrityError("cartography replay digest mismatch")
            try:
                observation = json.loads(value)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CartographyIntegrityError(
                    "cartography replay JSON is invalid"
                ) from exc
            if canonical_observation_bytes(observation) != value:
                raise CartographyIntegrityError(
                    "cartography replay bytes are not canonical"
                )
            self._replay.append(value)
            self._replay_size += len(value)
        if self._replay_size > self.replay_bytes:
            raise CartographyIntegrityError(
                "cartography replay byte budget is exceeded"
            )

        in_flight = document["in_flight"]
        self._in_flight = None
        if in_flight is not None:
            if not isinstance(in_flight, Mapping) or set(in_flight) != {
                "attempt",
                "attempt_id",
                "job_id",
            }:
                raise CartographyIntegrityError("in-flight probe state is malformed")
            job_id = in_flight["job_id"]
            attempt = in_flight["attempt"]
            attempt_id = in_flight["attempt_id"]
            if job_id not in self._jobs_by_id:
                raise CartographyIntegrityError(
                    "in-flight probe references an unknown job"
                )
            if attempt != len(grouped.get(job_id, ())) + 1:
                raise CartographyIntegrityError(
                    "in-flight probe attempt is not contiguous"
                )
            if attempt_id != ProbeOutcome.attempt_identity(job_id, attempt):
                raise CartographyIntegrityError("in-flight probe attempt id is invalid")
            self._in_flight = (job_id, attempt, attempt_id)

        stream_state = document["stream_state"]
        if stream_state is not None and not isinstance(stream_state, Mapping):
            raise CartographyIntegrityError("O1 stream state is malformed")
        self._stream_state = stream_state
        if self._stream_state is not None:
            try:
                self.stream.restore(self._stream_state)
            except Exception as exc:
                raise CartographyIntegrityError(
                    "O1 stream state restore failed"
                ) from exc
        last_stop_reason = document["last_stop_reason"]
        if not isinstance(last_stop_reason, str):
            raise CartographyIntegrityError("cartography stop reason is malformed")
        self._last_stop_reason = last_stop_reason
        self._file_sha256 = _sha256_bytes(raw)

        if self._in_flight is not None:
            job_id, attempt, _ = self._in_flight
            job = self._jobs_by_id[job_id]
            self._outcomes.append(
                ProbeOutcome.aborted(
                    job,
                    attempt,
                    error_code="scheduler-restart",
                    error="scheduler restarted before the attempt outcome committed",
                )
            )
            self._in_flight = None
            self._generation += 1
            self._checkpoint_locked(capture_stream=False)


__all__ = [
    "CARTOGRAPHER_STATE_SCHEMA",
    "COVERAGE_SCHEMA",
    "PROBE_JOB_SCHEMA",
    "PROBE_OUTCOME_SCHEMA",
    "CartographyBudget",
    "CartographyError",
    "CartographyIdentityError",
    "CartographyIntegrityError",
    "Coverage",
    "O1Cartographer",
    "ProbeJob",
    "ProbeOutcome",
    "ProbeTarget",
    "RetryPolicy",
    "RunResult",
    "build_probe_frontier",
    "canonical_observation_bytes",
]
