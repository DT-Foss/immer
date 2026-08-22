"""Fail-closed cache-temperature evidence for fixed Streamer workloads.

This module profiles transport/cache behaviour, not model quality.  A profile
starts in a newly-created run-owned cache, executes COLD in one process, then
executes WARM and HOT in a second process.  HOT is only admissible when it is
the immediate repeat on the same ``Streamer`` reader and all three executions
have identical ordered workload keys and decoded content digests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from immer.knowledge import Streamer
from immer.runtimes.deepseek_v4.benchmark import canonical_digest


CACHE_PROFILE_SCHEMA = "immer.deepseek-v4-cache-profile/v1"
WORKLOAD_SCHEMA = "immer.deepseek-v4-cache-workload/v1"
OWNER_SCHEMA = "immer.deepseek-v4-cache-profile-owner/v1"
PHASE_SCHEMA = "immer.deepseek-v4-cache-profile-phase/v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_PINNED_REVISION = re.compile(r"[0-9a-fA-F]{40,64}")
_WORKLOAD_FIELDS = frozenset(
    {
        "schema",
        "source",
        "source_kind",
        "revision",
        "dataset_sha256",
        "prompt_sha256",
        "mode",
        "seed",
        "operations",
        "workload_keys",
        "signature",
    }
)
_OWNER_FIELDS = frozenset(
    {
        "schema",
        "owner_token",
        "run_root",
        "workload_signature",
        "source_budget_bytes",
        "cache_budget_bytes",
        "created_ns",
        "marker_sha256",
    }
)
_PHASE_FIELDS = {
    "cold": frozenset(
        {"schema", "phase", "owner_token_sha256", "cold", "phase_sha256"}
    ),
    "warm-hot": frozenset(
        {
            "schema",
            "phase",
            "owner_token_sha256",
            "warm",
            "hot",
            "phase_sha256",
        }
    ),
}
_STATE_FIELDS = frozenset(
    {
        "schema",
        "state_attempted",
        "claimed_state",
        "workload_signature",
        "workload_keys",
        "pid",
        "reader_instance",
        "elapsed_ns",
        "inventory_sha256",
        "source_identity",
        "observations",
        "counters",
        "source_budget_limit_bytes",
        "cache_limit_bytes",
        "cache_files_before",
        "cache_bytes_on_disk_before",
        "cache_bytes_after",
    }
)
_REQUIRED_COUNTERS = (
    ("network_or_source_body_bytes",),
    ("cache_bytes_reused",),
    ("cache_bytes_written",),
    ("cache_hits",),
    ("cache_misses",),
    ("cache_writes",),
    ("cache_evictions",),
    ("cache_evicted_bytes",),
    ("cache_write_skips_oversize",),
    ("failed_requests",),
    ("budget", "http_requests"),
    ("budget", "bytes_body"),
    ("budget", "bytes_overhead_approx"),
)


class CacheProfileError(RuntimeError):
    """The cache experiment cannot support the requested state labels."""


def _require_digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise CacheProfileError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CacheProfileError(f"{label} must be a non-empty string")
    return value


def _require_nonnegative_int(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CacheProfileError(f"{label} must be a non-negative integer")
    return value


def _json_object(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CacheProfileError(f"{label} must be a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CacheProfileError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CacheProfileError(f"{label} must contain a JSON object")
    return value


def _sealed_digest(document: Mapping[str, Any], field: str) -> str:
    return canonical_digest(
        {key: value for key, value in document.items() if key != field}
    )


def _seal(document: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(document)
    result[field] = _sealed_digest(result, field)
    return result


def _validate_seal(document: Mapping[str, Any], field: str, label: str) -> None:
    if document.get(field) != _sealed_digest(document, field):
        raise CacheProfileError(f"{label} digest mismatch")


def atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    """Durably replace one run-owned JSON commit point."""

    requested = path.expanduser()
    requested.parent.mkdir(parents=True, exist_ok=True)
    target = requested.parent.resolve() / requested.name
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        metadata = None
    if metadata is not None and stat.S_ISLNK(metadata.st_mode):
        raise CacheProfileError(f"refusing to replace symlink: {target}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                document,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass


@dataclass(frozen=True, slots=True)
class CacheWorkloadOperation:
    """One deterministic tensor read in the cache experiment."""

    tensor_name: str
    start_row: int | None = None
    n_rows: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.tensor_name, "tensor_name")
        if (self.start_row is None) != (self.n_rows is None):
            raise CacheProfileError("start_row and n_rows must be supplied together")
        if self.start_row is not None:
            _require_nonnegative_int(self.start_row, "start_row")
            _require_nonnegative_int(self.n_rows, "n_rows")
            if self.n_rows == 0:
                raise CacheProfileError("n_rows must be positive")

    @property
    def kind(self) -> str:
        return "tensor" if self.start_row is None else "rows"

    @property
    def key(self) -> str:
        return canonical_digest(asdict(self))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CacheWorkloadOperation":
        allowed = {"tensor_name", "start_row", "n_rows"}
        if set(raw) != allowed:
            raise CacheProfileError("workload operation fields do not match schema")
        return cls(
            tensor_name=raw.get("tensor_name"),
            start_row=raw.get("start_row"),
            n_rows=raw.get("n_rows"),
        )


@dataclass(frozen=True, slots=True)
class CacheWorkload:
    """The immutable, paired workload and model-input provenance."""

    source: str
    source_kind: str
    revision: str
    dataset_sha256: str
    prompt_sha256: str
    mode: str
    seed: int
    operations: tuple[CacheWorkloadOperation, ...]

    def __post_init__(self) -> None:
        _require_text(self.source, "source")
        if self.source_kind not in {"local", "remote"}:
            raise CacheProfileError("source_kind must be 'local' or 'remote'")
        _require_text(self.revision, "revision")
        _require_digest(self.dataset_sha256, "dataset_sha256")
        _require_digest(self.prompt_sha256, "prompt_sha256")
        _require_text(self.mode, "mode")
        _require_nonnegative_int(self.seed, "seed")
        object.__setattr__(self, "operations", tuple(self.operations))
        if not self.operations:
            raise CacheProfileError("cache workload must contain an operation")
        if not all(isinstance(op, CacheWorkloadOperation) for op in self.operations):
            raise CacheProfileError("operations must be CacheWorkloadOperation values")
        keys = self.workload_keys
        if len(keys) != len(set(keys)):
            raise CacheProfileError("cache workload contains duplicate operations")

    @property
    def source_is_local(self) -> bool:
        return self.source_kind == "local"

    @property
    def workload_keys(self) -> tuple[str, ...]:
        return ("inventory", *(operation.key for operation in self.operations))

    @property
    def signature(self) -> str:
        return canonical_digest(self.to_dict(include_signature=False))

    @property
    def revision_is_pinned(self) -> bool:
        return (
            self.source_is_local
            or _PINNED_REVISION.fullmatch(self.revision) is not None
        )

    def to_dict(self, *, include_signature: bool = True) -> dict[str, Any]:
        result = {
            "schema": WORKLOAD_SCHEMA,
            "source": self.source,
            "source_kind": self.source_kind,
            "revision": self.revision,
            "dataset_sha256": self.dataset_sha256,
            "prompt_sha256": self.prompt_sha256,
            "mode": self.mode,
            "seed": self.seed,
            "operations": [asdict(operation) for operation in self.operations],
            "workload_keys": list(self.workload_keys),
        }
        if include_signature:
            result["signature"] = self.signature
        return result

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CacheWorkload":
        if set(raw) != _WORKLOAD_FIELDS:
            raise CacheProfileError("cache workload fields do not match schema")
        if raw.get("schema") != WORKLOAD_SCHEMA:
            raise CacheProfileError("unknown cache workload schema")
        operations = raw.get("operations")
        if not isinstance(operations, list) or not all(
            isinstance(operation, Mapping) for operation in operations
        ):
            raise CacheProfileError("workload operations must be a list of objects")
        value = cls(
            source=raw.get("source"),
            source_kind=raw.get("source_kind"),
            revision=raw.get("revision"),
            dataset_sha256=raw.get("dataset_sha256"),
            prompt_sha256=raw.get("prompt_sha256"),
            mode=raw.get("mode"),
            seed=raw.get("seed"),
            operations=tuple(
                CacheWorkloadOperation.from_dict(operation) for operation in operations
            ),
        )
        if raw.get("workload_keys") != list(value.workload_keys):
            raise CacheProfileError("workload keys do not match operations")
        if raw.get("signature") != value.signature:
            raise CacheProfileError("cache workload signature mismatch")
        return value


def _counter(metrics: Mapping[str, Any], path: Sequence[str]) -> int:
    current: Any = metrics
    try:
        for part in path:
            current = current[part]
    except (KeyError, TypeError) as exc:
        raise CacheProfileError(f"missing required counter {'.'.join(path)}") from exc
    if isinstance(current, bool) or not isinstance(current, int) or current < 0:
        raise CacheProfileError(f"counter {'.'.join(path)} is not non-negative int")
    return current


def counter_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, int]:
    """Return all required counter deltas, rejecting omissions or resets."""

    result: dict[str, int] = {}
    for path in _REQUIRED_COUNTERS:
        old = _counter(before, path)
        new = _counter(after, path)
        if new < old:
            raise CacheProfileError(f"counter {'.'.join(path)} decreased")
        result["_".join(path)] = new - old
    return result


def _content_digest(value: Any) -> tuple[str, int]:
    try:
        import numpy as np

        array = np.ascontiguousarray(value)
        body = memoryview(array).cast("B")
        digest = hashlib.sha256(body).hexdigest()
        return digest, int(body.nbytes)
    except (ImportError, TypeError, ValueError) as exc:
        raise CacheProfileError("tensor result is not a contiguous array") from exc


def _observation(
    *,
    key: str,
    action: Any,
    source: Streamer,
) -> dict[str, Any]:
    before = source.metrics()
    started = time.perf_counter_ns()
    result = action()
    elapsed_ns = time.perf_counter_ns() - started
    after = source.metrics()
    digest, decoded_bytes = _content_digest(result)
    return {
        "workload_key": key,
        "elapsed_ns": elapsed_ns,
        "decoded_sha256": digest,
        "decoded_bytes": decoded_bytes,
        "counters": counter_delta(before, after),
    }


def execute_workload(
    workload: CacheWorkload,
    source: Streamer,
    *,
    state: str,
    refresh_inventory: bool,
    reader_instance: str,
    cache_files_before: int,
    cache_bytes_on_disk_before: int,
) -> dict[str, Any]:
    """Execute one state without swallowing any transport or integrity error."""

    if state not in {"cold", "warm", "hot"}:
        raise CacheProfileError(f"unknown cache state {state!r}")
    _require_nonnegative_int(cache_files_before, "cache_files_before")
    _require_nonnegative_int(cache_bytes_on_disk_before, "cache_bytes_on_disk_before")
    before = source.metrics()
    started = time.perf_counter_ns()
    inventory_started = time.perf_counter_ns()
    inventory = source.inventory(refresh=refresh_inventory)
    inventory_elapsed = time.perf_counter_ns() - inventory_started
    inventory_after = source.metrics()
    inventory_digest = canonical_digest(inventory)
    observations = [
        {
            "workload_key": "inventory",
            "elapsed_ns": inventory_elapsed,
            "decoded_sha256": inventory_digest,
            "decoded_bytes": len(
                json.dumps(
                    inventory,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ),
            "counters": counter_delta(before, inventory_after),
        }
    ]
    for operation in workload.operations:
        if operation.kind == "tensor":

            def action(operation: CacheWorkloadOperation = operation) -> Any:
                return source.tensor(operation.tensor_name)
        else:

            def action(operation: CacheWorkloadOperation = operation) -> Any:
                return source.rows(
                    operation.tensor_name,
                    start_row=operation.start_row,
                    n_rows=operation.n_rows,
                )

        observations.append(
            _observation(key=operation.key, action=action, source=source)
        )
    after = source.metrics()
    elapsed_ns = time.perf_counter_ns() - started
    return {
        "schema": PHASE_SCHEMA,
        "state_attempted": state,
        "claimed_state": state,
        "workload_signature": workload.signature,
        "workload_keys": list(workload.workload_keys),
        "pid": os.getpid(),
        "reader_instance": reader_instance,
        "elapsed_ns": elapsed_ns,
        "inventory_sha256": inventory_digest,
        "source_identity": {
            "repo_id": after.get("repo_id"),
            "revision": after.get("revision"),
            "inventory_source_fingerprint": after.get("inventory_source_fingerprint"),
        },
        "observations": observations,
        "counters": counter_delta(before, after),
        "source_budget_limit_bytes": _counter(after, ("budget", "limit_bytes")),
        "cache_limit_bytes": _counter(after, ("cache_limit_bytes",)),
        "cache_files_before": cache_files_before,
        "cache_bytes_on_disk_before": cache_bytes_on_disk_before,
        "cache_bytes_after": _counter(after, ("cache_bytes",)),
    }


def _operation_evidence(phase: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    observations = phase.get("observations")
    if not isinstance(observations, list):
        raise CacheProfileError("phase observations are missing")
    evidence: list[tuple[str, str]] = []
    for observation in observations:
        if not isinstance(observation, Mapping):
            raise CacheProfileError("phase observation is not an object")
        key = observation.get("workload_key")
        digest = observation.get("decoded_sha256")
        _require_text(key, "observation workload_key")
        _require_digest(digest, "observation decoded_sha256")
        evidence.append((key, digest))
    return tuple(evidence)


def _validate_phase_shape(
    phase: Mapping[str, Any], workload: CacheWorkload, label: str
) -> None:
    if set(phase) != _STATE_FIELDS:
        raise CacheProfileError(f"{label} phase fields do not match schema")
    if phase.get("schema") != PHASE_SCHEMA:
        raise CacheProfileError(f"{label} phase schema mismatch")
    if phase.get("state_attempted") != label or phase.get("claimed_state") != label:
        raise CacheProfileError(f"{label} phase was not admissibly labelled")
    if phase.get("workload_signature") != workload.signature:
        raise CacheProfileError(f"{label} workload signature mismatch")
    if phase.get("workload_keys") != list(workload.workload_keys):
        raise CacheProfileError(f"{label} workload keys mismatch")
    pid = phase.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise CacheProfileError(f"{label} phase pid is invalid")
    _require_text(phase.get("reader_instance"), f"{label} reader_instance")
    _require_nonnegative_int(phase.get("elapsed_ns"), f"{label} elapsed_ns")
    _require_digest(phase.get("inventory_sha256"), f"{label} inventory_sha256")
    cache_bytes = phase.get("cache_bytes_after")
    _require_nonnegative_int(cache_bytes, f"{label} cache_bytes_after")
    for field in (
        "source_budget_limit_bytes",
        "cache_limit_bytes",
        "cache_files_before",
        "cache_bytes_on_disk_before",
    ):
        _require_nonnegative_int(phase.get(field), f"{label} {field}")
    if phase.get("source_budget_limit_bytes", 0) == 0:
        raise CacheProfileError(f"{label} source budget must be positive")
    if phase.get("cache_limit_bytes", 0) == 0:
        raise CacheProfileError(f"{label} cache limit must be positive")

    identity = phase.get("source_identity")
    if not isinstance(identity, Mapping) or set(identity) != {
        "repo_id",
        "revision",
        "inventory_source_fingerprint",
    }:
        raise CacheProfileError(f"{label} source identity is incomplete")
    expected_repo = (
        f"local:{Path(workload.source).expanduser().resolve()}"
        if workload.source_is_local
        else workload.source
    )
    if identity.get("repo_id") != expected_repo:
        raise CacheProfileError(f"{label} source repo identity mismatch")
    if identity.get("revision") != workload.revision:
        raise CacheProfileError(f"{label} source revision mismatch")
    _require_digest(
        identity.get("inventory_source_fingerprint"),
        f"{label} inventory_source_fingerprint",
    )

    counters = phase.get("counters")
    if not isinstance(counters, Mapping):
        raise CacheProfileError(f"{label} phase has no complete counters")
    required_counters = {"_".join(path) for path in _REQUIRED_COUNTERS}
    if set(counters) != required_counters:
        raise CacheProfileError(f"{label} phase counter fields do not match schema")
    for counter_name in sorted(required_counters):
        _require_nonnegative_int(
            counters.get(counter_name), f"{label} counter {counter_name}"
        )

    observations = phase.get("observations")
    if not isinstance(observations, list) or len(observations) != len(
        workload.workload_keys
    ):
        raise CacheProfileError(f"{label} phase omitted workload observations")
    observation_fields = {
        "workload_key",
        "elapsed_ns",
        "decoded_sha256",
        "decoded_bytes",
        "counters",
    }
    totals = {name: 0 for name in required_counters}
    for expected_key, observation in zip(workload.workload_keys, observations):
        if (
            not isinstance(observation, Mapping)
            or set(observation) != observation_fields
        ):
            raise CacheProfileError(
                f"{label} phase observation fields do not match schema"
            )
        if observation.get("workload_key") != expected_key:
            raise CacheProfileError(f"{label} phase observation order mismatch")
        _require_nonnegative_int(
            observation.get("elapsed_ns"), f"{label} observation elapsed_ns"
        )
        _require_digest(
            observation.get("decoded_sha256"),
            f"{label} observation decoded_sha256",
        )
        _require_nonnegative_int(
            observation.get("decoded_bytes"),
            f"{label} observation decoded_bytes",
        )
        observation_counters = observation.get("counters")
        if (
            not isinstance(observation_counters, Mapping)
            or set(observation_counters) != required_counters
        ):
            raise CacheProfileError(
                f"{label} observation counter fields do not match schema"
            )
        for counter_name in required_counters:
            value = observation_counters.get(counter_name)
            _require_nonnegative_int(
                value, f"{label} observation counter {counter_name}"
            )
            totals[counter_name] += value
    if observations[0].get("decoded_sha256") != phase.get("inventory_sha256"):
        raise CacheProfileError(f"{label} inventory evidence mismatch")
    if totals != dict(counters):
        raise CacheProfileError(f"{label} aggregate counters do not match observations")


def validate_cold_phase(
    phase: Mapping[str, Any], workload: CacheWorkload, *, fresh_cache: bool
) -> None:
    _validate_phase_shape(phase, workload, "cold")
    if not fresh_cache:
        raise CacheProfileError("cold phase did not start with an empty owned cache")
    if (
        phase.get("cache_files_before") != 0
        or phase.get("cache_bytes_on_disk_before") != 0
    ):
        raise CacheProfileError("cold phase contains pre-existing cache evidence")
    counters = phase.get("counters")
    if counters.get("cache_evictions") or counters.get("cache_write_skips_oversize"):
        raise CacheProfileError("cold workload did not remain fully resident")
    if counters.get("failed_requests"):
        raise CacheProfileError("cold workload contains failed source requests")
    if (
        counters.get("network_or_source_body_bytes", 0) <= 0
        or counters.get("budget_bytes_body", 0) <= 0
        or counters.get("cache_misses", 0) <= 0
        or counters.get("cache_writes", 0) <= 0
        or counters.get("cache_bytes_written", 0) <= 0
    ):
        raise CacheProfileError("cold phase did not read and populate from source")
    if phase.get("cache_bytes_after", 0) <= 0:
        raise CacheProfileError("cold phase did not populate the disk cache")


def _zero_source_complete_hit(phase: Mapping[str, Any]) -> bool:
    counters = phase.get("counters")
    if not isinstance(counters, Mapping):
        return False
    observations = phase.get("observations")
    payload_count = (
        max(0, len(observations) - 1) if isinstance(observations, list) else 0
    )
    return (
        counters.get("network_or_source_body_bytes") == 0
        and counters.get("budget_bytes_body") == 0
        and counters.get("cache_misses") == 0
        and counters.get("cache_bytes_written") == 0
        and counters.get("cache_writes") == 0
        and counters.get("cache_evictions") == 0
        and counters.get("cache_write_skips_oversize") == 0
        and counters.get("failed_requests") == 0
        and counters.get("cache_hits") == payload_count
        and payload_count > 0
    )


def validate_paired_phases(
    workload: CacheWorkload,
    cold: Mapping[str, Any],
    warm: Mapping[str, Any],
    hot: Mapping[str, Any],
) -> None:
    """Validate state isolation and exact paired-workload equivalence."""

    validate_cold_phase(cold, workload, fresh_cache=True)
    for label, phase in (("warm", warm), ("hot", hot)):
        _validate_phase_shape(phase, workload, label)
        if (
            phase.get("cache_files_before", 0) <= 0
            or phase.get("cache_bytes_on_disk_before", 0) <= 0
        ):
            raise CacheProfileError(f"{label} did not start from populated disk cache")
        if not _zero_source_complete_hit(phase):
            raise CacheProfileError(f"{label} was not a complete source-free cache hit")
    if cold.get("pid") == warm.get("pid"):
        raise CacheProfileError("COLD and WARM must execute in different processes")
    if warm.get("pid") != hot.get("pid"):
        raise CacheProfileError("WARM and HOT must execute in the same process")
    if warm.get("reader_instance") != hot.get("reader_instance"):
        raise CacheProfileError("HOT did not reuse the WARM reader instance")
    if not (
        cold.get("cache_bytes_after")
        == warm.get("cache_bytes_after")
        == hot.get("cache_bytes_after")
    ):
        raise CacheProfileError("cache residency changed across paired states")
    if not (
        warm.get("cache_files_before") == hot.get("cache_files_before")
        and warm.get("cache_bytes_on_disk_before")
        == hot.get("cache_bytes_on_disk_before")
    ):
        raise CacheProfileError("disk cache changed between WARM and HOT")
    limits = {
        (
            phase.get("source_budget_limit_bytes"),
            phase.get("cache_limit_bytes"),
        )
        for phase in (cold, warm, hot)
    }
    if len(limits) != 1:
        raise CacheProfileError("cache profile budgets changed across states")
    baseline = _operation_evidence(cold)
    if _operation_evidence(warm) != baseline or _operation_evidence(hot) != baseline:
        raise CacheProfileError("decoded workload evidence differs across cache states")
    identities = {
        canonical_digest(phase.get("source_identity")) for phase in (cold, warm, hot)
    }
    if len(identities) != 1:
        raise CacheProfileError("source identity changed across cache states")


def create_owned_run(
    output_root: Path,
    workload: CacheWorkload,
    *,
    source_budget_bytes: int,
    cache_budget_bytes: int,
) -> tuple[Path, str]:
    """Create a unique run root without inspecting or removing sibling data."""

    for value, label in (
        (source_budget_bytes, "source_budget_bytes"),
        (cache_budget_bytes, "cache_budget_bytes"),
    ):
        _require_nonnegative_int(value, label)
        if value == 0:
            raise CacheProfileError(f"{label} must be positive")

    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise CacheProfileError("output root must be a regular directory")
    run_root = root / f"cache-profile-{uuid.uuid4().hex}"
    run_root.mkdir(mode=0o700)
    token = uuid.uuid4().hex
    marker = _seal(
        {
            "schema": OWNER_SCHEMA,
            "owner_token": token,
            "run_root": str(run_root),
            "workload_signature": workload.signature,
            "source_budget_bytes": source_budget_bytes,
            "cache_budget_bytes": cache_budget_bytes,
            "created_ns": time.time_ns(),
        },
        "marker_sha256",
    )
    atomic_write_json(run_root / "OWNER.json", marker)
    atomic_write_json(run_root / "workload.json", workload.to_dict())
    return run_root, token


def validate_owned_run(
    run_root: Path, owner_token: str | None = None
) -> dict[str, Any]:
    root = run_root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise CacheProfileError("run root must be a regular directory")
    marker = _json_object(root / "OWNER.json", "ownership marker")
    if set(marker) != _OWNER_FIELDS:
        raise CacheProfileError("ownership marker fields do not match schema")
    _validate_seal(marker, "marker_sha256", "ownership marker")
    if marker.get("schema") != OWNER_SCHEMA or marker.get("run_root") != str(root):
        raise CacheProfileError("ownership marker does not bind this run root")
    token = marker.get("owner_token")
    if not isinstance(token, str) or re.fullmatch(r"[0-9a-f]{32}", token) is None:
        raise CacheProfileError("ownership marker token is invalid")
    if owner_token is not None and token != owner_token:
        raise CacheProfileError("ownership token mismatch")
    workload = CacheWorkload.from_dict(
        _json_object(root / "workload.json", "workload manifest")
    )
    if marker.get("workload_signature") != workload.signature:
        raise CacheProfileError("ownership marker/workload signature mismatch")
    for field in ("source_budget_bytes", "cache_budget_bytes"):
        value = marker.get(field)
        _require_nonnegative_int(value, f"ownership marker {field}")
        if value == 0:
            raise CacheProfileError(f"ownership marker {field} must be positive")
    cache_dir = root / "cache"
    if cache_dir.exists() and (cache_dir.is_symlink() or not cache_dir.is_dir()):
        raise CacheProfileError("owned cache path is not a regular directory")
    return {"marker": marker, "workload": workload, "cache_dir": cache_dir}


def build_streamer(
    workload: CacheWorkload,
    cache_dir: Path,
    *,
    source_budget_bytes: int,
    cache_budget_bytes: int,
) -> Streamer:
    for value, label in (
        (source_budget_bytes, "source_budget_bytes"),
        (cache_budget_bytes, "cache_budget_bytes"),
    ):
        _require_nonnegative_int(value, label)
    if source_budget_bytes == 0:
        raise CacheProfileError("source budget must be positive")
    common = {
        "revision": workload.revision,
        "budget_mb": source_budget_bytes / 1024**2,
        "cache_dir": cache_dir,
        "use_cache": True,
        "max_cache_bytes": cache_budget_bytes,
    }
    if workload.source_is_local:
        return Streamer.from_local(workload.source, **common)
    if not workload.revision_is_pinned:
        raise CacheProfileError("remote cache profiles require a pinned revision")
    return Streamer(workload.source, **common)


def phase_result_path(run_root: Path, phase: str) -> Path:
    if phase not in {"cold", "warm-hot"}:
        raise CacheProfileError("invalid cache profile phase")
    return run_root / f"{phase}.json"


def load_phase(path: Path, expected_phase: str) -> dict[str, Any]:
    if expected_phase not in _PHASE_FIELDS:
        raise CacheProfileError("invalid expected cache profile phase")
    result = _json_object(path, f"{expected_phase} phase result")
    if set(result) != _PHASE_FIELDS[expected_phase]:
        raise CacheProfileError(f"{expected_phase} phase fields do not match schema")
    if result.get("schema") != PHASE_SCHEMA or result.get("phase") != expected_phase:
        raise CacheProfileError(f"{expected_phase} phase result schema mismatch")
    _validate_seal(result, "phase_sha256", f"{expected_phase} phase")
    return result


def seal_phase_envelope(document: Mapping[str, Any]) -> dict[str, Any]:
    """Seal a worker envelope after validating its phase-level shape."""

    phase = document.get("phase")
    if phase not in _PHASE_FIELDS:
        raise CacheProfileError("invalid phase envelope label")
    expected_without_seal = _PHASE_FIELDS[phase] - {"phase_sha256"}
    if set(document) != expected_without_seal:
        raise CacheProfileError("phase envelope fields do not match schema")
    return _seal(document, "phase_sha256")


def build_report(
    *,
    run_root: Path,
    workload: CacheWorkload,
    cold_envelope: Mapping[str, Any],
    warm_hot_envelope: Mapping[str, Any],
    source_budget_bytes: int,
    cache_budget_bytes: int,
    command: Sequence[str],
    harness_sha256: str,
) -> dict[str, Any]:
    owned = validate_owned_run(run_root)
    if owned["workload"] != workload:
        raise CacheProfileError("report workload does not match owned run")
    marker = owned["marker"]
    if (
        marker.get("source_budget_bytes") != source_budget_bytes
        or marker.get("cache_budget_bytes") != cache_budget_bytes
    ):
        raise CacheProfileError("report budgets do not match owned run")
    for envelope, phase in (
        (cold_envelope, "cold"),
        (warm_hot_envelope, "warm-hot"),
    ):
        if set(envelope) != _PHASE_FIELDS[phase] or envelope.get("phase") != phase:
            raise CacheProfileError(f"{phase} phase envelope is invalid")
        _validate_seal(envelope, "phase_sha256", f"{phase} phase")
    expected_owner = hashlib.sha256(marker["owner_token"].encode()).hexdigest()
    if (
        cold_envelope.get("owner_token_sha256") != expected_owner
        or warm_hot_envelope.get("owner_token_sha256") != expected_owner
    ):
        raise CacheProfileError("phase envelope owner does not match owned run")
    cold = cold_envelope.get("cold")
    warm = warm_hot_envelope.get("warm")
    hot = warm_hot_envelope.get("hot")
    if not all(isinstance(phase, Mapping) for phase in (cold, warm, hot)):
        raise CacheProfileError("phase envelope is incomplete")
    validate_paired_phases(workload, cold, warm, hot)
    for label, phase in (("cold", cold), ("warm", warm), ("hot", hot)):
        if phase.get("source_budget_limit_bytes") != source_budget_bytes:
            raise CacheProfileError(f"{label} source budget evidence mismatch")
        if phase.get("cache_limit_bytes") != cache_budget_bytes:
            raise CacheProfileError(f"{label} cache budget evidence mismatch")
    report = {
        "schema": CACHE_PROFILE_SCHEMA,
        "status": "complete",
        "scope": "streamer_transport_cache_only",
        "quality_or_end_to_end_performance_claim": False,
        "run_root": str(run_root.expanduser().resolve()),
        "workload": workload.to_dict(),
        "paired_workload_keys_exact": True,
        "states": {"cold": cold, "warm": warm, "hot": hot},
        "state_protocol": {
            "cold": "fresh run-owned empty cache; isolated process",
            "warm": "new process; completely populated identical disk cache",
            "hot": "immediate identical repeat; same process and Streamer reader",
        },
        "budgets": {
            "source_bytes_per_process": source_budget_bytes,
            "cache_bytes": cache_budget_bytes,
        },
        "provenance": {
            "harness_sha256": _require_digest(harness_sha256, "harness_sha256"),
            "cache_profile_module_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "command": list(command),
            "pid_orchestration": {
                "cold_pid": cold.get("pid"),
                "warm_pid": warm.get("pid"),
                "hot_pid": hot.get("pid"),
            },
        },
    }
    report["report_sha256"] = canonical_digest(report)
    return report


def finite_seconds(nanoseconds: Any) -> float:
    value = float(nanoseconds) / 1_000_000_000.0
    if not math.isfinite(value) or value < 0:
        raise CacheProfileError("elapsed time is not finite and non-negative")
    return value
