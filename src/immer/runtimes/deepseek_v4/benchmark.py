"""Deterministic benchmark contracts for the streamed DeepSeek-V4 runtime.

This module deliberately does no model or dataset I/O.  It turns observations
made by a runner into canonical, digestible records and applies the preregistered
frontier stop/go gates.  A runtime exception is an item outcome, not a dropped
row: errors stay in every denominator.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass, field, is_dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum, StrEnum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "immer.deepseek_v4.benchmark/v1"
JOURNAL_GENESIS_SHA256 = "0" * 64
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PINNED_REVISION = re.compile(r"[0-9a-fA-F]{40,64}")
_GSM8K_LITERAL = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?"
_GSM8K_NUMBER = re.compile(rf"(?<![A-Za-z0-9.]){_GSM8K_LITERAL}(?![A-Za-z0-9]|\.\d)")
_GSM8K_CURRENCY = re.compile(rf"[$€£]\s*({_GSM8K_LITERAL})(?![A-Za-z0-9]|\.\d)")
_GSM8K_BOXED = re.compile(r"\\boxed\s*\{\s*([^{}]+?)\s*\}")
_MARKDOWN_STRONG = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)


class BenchmarkContractError(ValueError):
    """A measurement is incomplete, ambiguous, or not reproducible."""


class BenchmarkTask(StrEnum):
    DECODER_PARITY = "decoder_parity"
    MMLU = "mmlu"
    GSM8K = "gsm8k"


class OutcomeStatus(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    ABSTAINED = "abstained"
    ERROR = "error"


class CacheState(StrEnum):
    COLD = "cold"
    WARM = "warm"
    HOT = "hot"
    UNCONTROLLED = "uncontrolled"


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonical_value(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise BenchmarkContractError("canonical mappings require string keys")
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BenchmarkContractError("canonical metrics must be finite")
        return value
    raise BenchmarkContractError(
        f"value is not canonical JSON data: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Encode JSON data with stable key order and no platform whitespace."""

    canonical = _canonical_value(value)
    return json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def strict_json_loads(raw: str | bytes | bytearray) -> Any:
    """Decode JSON while rejecting duplicate keys at every object depth."""

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BenchmarkContractError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(raw, object_pairs_hook=reject_duplicate_keys)


def journal_record_digest(record: Mapping[str, Any]) -> str:
    """Hash a journal record without its self-referential digest field."""

    if not isinstance(record, Mapping):
        raise BenchmarkContractError("journal record must be a mapping")
    payload = dict(record)
    payload.pop("record_sha256", None)
    return canonical_digest(payload)


def seal_journal_record(
    record: Mapping[str, Any],
    *,
    sequence: int,
    previous_sha256: str = JOURNAL_GENESIS_SHA256,
) -> dict[str, Any]:
    """Bind a canonical record to its exact position in an append-only chain."""

    _require_count(sequence, "journal sequence")
    _require_digest(previous_sha256, "journal previous sha256")
    collisions = {"sequence", "previous_sha256", "record_sha256"}.intersection(record)
    if collisions:
        names = ", ".join(sorted(collisions))
        raise BenchmarkContractError(
            f"journal record already has integrity fields: {names}"
        )
    sealed = dict(record)
    sealed["sequence"] = sequence
    sealed["previous_sha256"] = previous_sha256
    sealed["record_sha256"] = journal_record_digest(sealed)
    return sealed


def verify_journal_record(
    record: Mapping[str, Any],
    *,
    sequence: int,
    previous_sha256: str = JOURNAL_GENESIS_SHA256,
) -> str:
    """Verify chain position and content, returning the authenticated record hash."""

    _require_count(sequence, "expected journal sequence")
    _require_digest(previous_sha256, "expected journal previous sha256")
    if record.get("sequence") != sequence:
        raise BenchmarkContractError(
            f"journal sequence mismatch: expected {sequence}, got {record.get('sequence')!r}"
        )
    if record.get("previous_sha256") != previous_sha256:
        raise BenchmarkContractError(f"journal chain mismatch at sequence {sequence}")
    claimed = record.get("record_sha256")
    _require_digest(claimed, "journal record sha256")
    actual = journal_record_digest(record)
    if not hmac.compare_digest(claimed, actual):
        raise BenchmarkContractError(
            f"journal record hash mismatch at sequence {sequence}"
        )
    return claimed


def dataset_digest(records: Iterable[Mapping[str, Any]]) -> str:
    """Digest an ordered dataset split with unambiguous per-record framing."""

    digest = hashlib.sha256()
    count = 0
    for record in records:
        raw = canonical_json_bytes(record)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        count += 1
    digest.update(count.to_bytes(8, "big"))
    return digest.hexdigest()


def _require_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkContractError(f"{label} must be a non-empty string")
    return value


def _require_digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BenchmarkContractError(f"{label} must be lowercase SHA-256")
    return value


def _require_count(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkContractError(f"{label} must be a non-negative integer")
    return value


def _finite_nonnegative(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise BenchmarkContractError(f"{label} must be finite and non-negative")
    return result


@dataclass(frozen=True, slots=True)
class DatasetProvenance:
    name: str
    split: str
    sha256: str
    num_items: int
    revision: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.name, "dataset name")
        _require_text(self.split, "dataset split")
        _require_digest(self.sha256, "dataset sha256")
        _require_count(self.num_items, "dataset num_items")
        if self.revision is not None:
            _require_text(self.revision, "dataset revision")

    @classmethod
    def from_records(
        cls,
        name: str,
        split: str,
        records: Sequence[Mapping[str, Any]],
        *,
        revision: str | None = None,
    ) -> "DatasetProvenance":
        return cls(name, split, dataset_digest(records), len(records), revision)


@dataclass(frozen=True, slots=True)
class BenchmarkProvenance:
    model_id: str
    model_revision: str
    inventory_sha256: str
    config_sha256: str
    tokenizer_sha256: str
    dataset: DatasetProvenance
    harness_revision: str
    graft: str = "none"
    command: tuple[str, ...] = ()
    protocol: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.model_id, "model id")
        _require_text(self.model_revision, "model revision")
        _require_digest(self.inventory_sha256, "inventory sha256")
        _require_digest(self.config_sha256, "config sha256")
        _require_digest(self.tokenizer_sha256, "tokenizer sha256")
        if not isinstance(self.dataset, DatasetProvenance):
            raise BenchmarkContractError("dataset provenance is required")
        _require_text(self.harness_revision, "harness revision")
        _require_text(self.graft, "graft")
        if any(not isinstance(part, str) for part in self.command):
            raise BenchmarkContractError("command must contain only strings")
        object.__setattr__(self, "command", tuple(self.command))
        canonical_json_bytes(self.protocol)

    @property
    def revision_is_pinned(self) -> bool:
        return _PINNED_REVISION.fullmatch(self.model_revision) is not None


@dataclass(frozen=True, slots=True)
class ItemMetric:
    item_id: str
    task: BenchmarkTask
    seed: int
    status: OutcomeStatus
    expected: Any = None
    predicted: Any = None
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.item_id, "item id")
        object.__setattr__(self, "task", BenchmarkTask(self.task))
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise BenchmarkContractError("seed must be a non-negative integer")
        object.__setattr__(self, "status", OutcomeStatus(self.status))
        if self.status is OutcomeStatus.ERROR:
            _require_text(self.error or "", "error")
        elif self.error is not None:
            raise BenchmarkContractError(
                "only error outcomes may carry an error message"
            )
        canonical_json_bytes(self.expected)
        canonical_json_bytes(self.predicted)
        canonical_json_bytes(self.metadata)

    @property
    def correct(self) -> bool:
        return self.status is OutcomeStatus.CORRECT

    @property
    def failed(self) -> bool:
        return self.status is OutcomeStatus.ERROR

    @property
    def abstained(self) -> bool:
        return self.status is OutcomeStatus.ABSTAINED


@dataclass(frozen=True, slots=True)
class OutcomeSummary:
    total: int
    correct: int
    incorrect: int
    abstained: int
    errors: int
    accuracy: float
    accuracy_attempted: float
    coverage: float
    wrong_rate: float
    error_rate: float

    @classmethod
    def from_items(cls, items: Iterable[ItemMetric]) -> "OutcomeSummary":
        rows = tuple(items)
        total = len(rows)
        correct = sum(row.status is OutcomeStatus.CORRECT for row in rows)
        incorrect = sum(row.status is OutcomeStatus.INCORRECT for row in rows)
        abstained = sum(row.status is OutcomeStatus.ABSTAINED for row in rows)
        errors = sum(row.status is OutcomeStatus.ERROR for row in rows)
        if correct + incorrect + abstained + errors != total:
            raise AssertionError("outcome partition is incomplete")
        denominator = total or 1
        attempted = correct + incorrect
        return cls(
            total=total,
            correct=correct,
            incorrect=incorrect,
            abstained=abstained,
            errors=errors,
            accuracy=correct / denominator if total else 0.0,
            accuracy_attempted=correct / attempted if attempted else 0.0,
            coverage=attempted / denominator if total else 0.0,
            wrong_rate=incorrect / denominator if total else 0.0,
            error_rate=errors / denominator if total else 0.0,
        )


def summarize_outcomes(
    items: Iterable[ItemMetric], *, task: BenchmarkTask | str | None = None
) -> OutcomeSummary:
    selected = tuple(items)
    if task is not None:
        wanted = BenchmarkTask(task)
        selected = tuple(item for item in selected if item.task is wanted)
    return OutcomeSummary.from_items(selected)


def _normalize_choice(value: Any, num_choices: int) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if 0 <= value < num_choices else None
    if not isinstance(value, str):
        return None
    stripped = value.strip().upper()
    if len(stripped) == 1 and "A" <= stripped <= "Z":
        choice = ord(stripped) - ord("A")
        return choice if choice < num_choices else None
    if stripped.isdigit():
        choice = int(stripped)
        return choice if 0 <= choice < num_choices else None
    match = re.search(r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*\(?([A-Z])\)?", stripped)
    if match:
        choice = ord(match.group(1)) - ord("A")
        return choice if choice < num_choices else None
    return None


def evaluate_mmlu(
    item_id: str,
    expected: int | str,
    predicted: int | str | None,
    *,
    seed: int,
    num_choices: int = 4,
    error: str | None = None,
    abstained: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> ItemMetric:
    if (
        isinstance(num_choices, bool)
        or not isinstance(num_choices, int)
        or num_choices < 2
    ):
        raise BenchmarkContractError("num_choices must be an integer >= 2")
    gold = _normalize_choice(expected, num_choices)
    if gold is None:
        raise BenchmarkContractError("MMLU expected choice is invalid")
    guess = _normalize_choice(predicted, num_choices)
    if error is not None and abstained:
        raise BenchmarkContractError("an item cannot be both error and abstained")
    if error is not None:
        status = OutcomeStatus.ERROR
    elif abstained:
        status = OutcomeStatus.ABSTAINED
    else:
        status = OutcomeStatus.CORRECT if guess == gold else OutcomeStatus.INCORRECT
    return ItemMetric(
        item_id=item_id,
        task=BenchmarkTask.MMLU,
        seed=seed,
        status=status,
        expected=gold,
        predicted=guess,
        error=error,
        metadata=metadata or {},
    )


def _canonical_gsm8k_number(literal: str) -> str | None:
    literal = literal.replace(",", "")
    try:
        value = Decimal(literal)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if value == 0:
        return "0"
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def _gsm8k_number(region: str, *, first: bool = False) -> str | None:
    matches = _GSM8K_NUMBER.findall(region)
    if not matches:
        return None
    return _canonical_gsm8k_number(matches[0] if first else matches[-1])


def _terminal_answer_number(region: str) -> str | None:
    currency = _GSM8K_CURRENCY.findall(region)
    if currency:
        return _canonical_gsm8k_number(currency[-1])
    return _gsm8k_number(region, first="=" not in region)


def extract_gsm8k_answer(text: Any) -> str | None:
    """Return the canonical final decimal answer used by GSM8K exact match.

    Explicit dataset markers and terminal model answer spans outrank the last
    number in free text.  This avoids reading a trailing duration or unit count
    as the answer in conclusions such as ``**earns $120 in 2 weeks**``.
    """

    if text is None:
        return None
    raw = str(text).strip()
    if "####" in raw:
        return _gsm8k_number(raw.rsplit("####", 1)[-1])

    boxed = _GSM8K_BOXED.findall(raw)
    if boxed:
        answer = _gsm8k_number(boxed[-1])
        if answer is not None:
            return answer

    strong = tuple(_MARKDOWN_STRONG.finditer(raw))
    if strong and not raw[strong[-1].end() :].strip():
        answer = _terminal_answer_number(strong[-1].group(1))
        if answer is not None:
            return answer

    return _gsm8k_number(raw)


def evaluate_gsm8k(
    item_id: str,
    expected: Any,
    predicted: Any,
    *,
    seed: int,
    error: str | None = None,
    abstained: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> ItemMetric:
    gold = extract_gsm8k_answer(expected)
    if gold is None:
        raise BenchmarkContractError("GSM8K expected answer has no numeric target")
    guess = extract_gsm8k_answer(predicted)
    if error is not None and abstained:
        raise BenchmarkContractError("an item cannot be both error and abstained")
    if error is not None:
        status = OutcomeStatus.ERROR
    elif abstained:
        status = OutcomeStatus.ABSTAINED
    else:
        status = OutcomeStatus.CORRECT if guess == gold else OutcomeStatus.INCORRECT
    return ItemMetric(
        item_id=item_id,
        task=BenchmarkTask.GSM8K,
        seed=seed,
        status=status,
        expected=gold,
        predicted=guess,
        error=error,
        metadata=metadata or {},
    )


@dataclass(frozen=True, slots=True)
class PerformanceMetric:
    item_id: str
    seed: int
    cache_state: CacheState
    latency_ms: float
    source_bytes: int
    cache_bytes: int
    requests: int
    output_tokens: int = 0
    error: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.item_id, "performance item id")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise BenchmarkContractError("seed must be a non-negative integer")
        object.__setattr__(self, "cache_state", CacheState(self.cache_state))
        object.__setattr__(
            self, "latency_ms", _finite_nonnegative(self.latency_ms, "latency_ms")
        )
        for label in ("source_bytes", "cache_bytes", "requests", "output_tokens"):
            _require_count(getattr(self, label), label)
        if self.error is not None:
            _require_text(self.error, "performance error")

    @property
    def failed(self) -> bool:
        return self.error is not None

    @classmethod
    def from_snapshots(
        cls,
        item_id: str,
        seed: int,
        cache_state: CacheState | str,
        latency_ms: float,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        *,
        output_tokens: int = 0,
        error: str | None = None,
    ) -> "PerformanceMetric":
        def delta(label: str, *paths: tuple[str, ...]) -> int:
            for path in paths:
                try:
                    old: Any = before
                    new: Any = after
                    for part in path:
                        old = old[part]
                        new = new[part]
                    difference = int(new) - int(old)
                except KeyError:
                    continue
                except (TypeError, ValueError) as exc:
                    raise BenchmarkContractError(
                        f"counter {'.'.join(path)} is not an integer"
                    ) from exc
                if difference < 0:
                    raise BenchmarkContractError(
                        f"counter {'.'.join(path)} decreased between snapshots"
                    )
                return difference
            alternatives = " or ".join(".".join(path) for path in paths)
            raise BenchmarkContractError(
                f"missing {label} counter in before/after snapshots: {alternatives}"
            )

        source_bytes = delta(
            "source bytes", ("network_or_source_body_bytes",), ("budget", "body")
        )
        cache_bytes = delta("cache bytes", ("cache_bytes_reused",))
        requests = delta("requests", ("budget", "requests"), ("range_requests",))
        return cls(
            item_id,
            seed,
            CacheState(cache_state),
            latency_ms,
            source_bytes,
            cache_bytes,
            requests,
            output_tokens,
            error,
        )


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    point = (len(ordered) - 1) * fraction
    lower = math.floor(point)
    upper = math.ceil(point)
    if lower == upper:
        return ordered[lower]
    weight = point - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


@dataclass(frozen=True, slots=True)
class PerformanceSummary:
    cache_state: CacheState
    total: int
    errors: int
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_source_bytes: int
    mean_source_bytes: float
    total_cache_bytes: int
    total_requests: int
    output_tokens: int
    tokens_per_second: float

    @classmethod
    def from_metrics(
        cls, cache_state: CacheState | str, metrics: Iterable[PerformanceMetric]
    ) -> "PerformanceSummary":
        state = CacheState(cache_state)
        rows = tuple(metric for metric in metrics if metric.cache_state is state)
        total = len(rows)
        latencies = [row.latency_ms for row in rows]
        elapsed_seconds = sum(latencies) / 1000.0
        output_tokens = sum(row.output_tokens for row in rows)
        return cls(
            state,
            total,
            sum(row.failed for row in rows),
            statistics.fmean(latencies) if latencies else 0.0,
            _percentile(latencies, 0.50),
            _percentile(latencies, 0.95),
            _percentile(latencies, 0.99),
            sum(row.source_bytes for row in rows),
            statistics.fmean(row.source_bytes for row in rows) if rows else 0.0,
            sum(row.cache_bytes for row in rows),
            sum(row.requests for row in rows),
            output_tokens,
            output_tokens / elapsed_seconds if elapsed_seconds else 0.0,
        )


@dataclass(frozen=True, slots=True)
class CacheProfile:
    cold: PerformanceSummary | None
    warm: PerformanceSummary | None
    hot: PerformanceSummary | None
    workload_aligned: bool = False

    @classmethod
    def from_metrics(cls, metrics: Iterable[PerformanceMetric]) -> "CacheProfile":
        rows = tuple(metrics)

        keys_by_state: dict[CacheState, set[tuple[str, int]]] = {}
        for state in (CacheState.COLD, CacheState.WARM, CacheState.HOT):
            keys = [(row.item_id, row.seed) for row in rows if row.cache_state is state]
            if len(keys) != len(set(keys)):
                raise BenchmarkContractError(
                    f"duplicate {state.value} cache workload item"
                )
            keys_by_state[state] = set(keys)

        def summary(state: CacheState) -> PerformanceSummary | None:
            selected = tuple(row for row in rows if row.cache_state is state)
            return (
                PerformanceSummary.from_metrics(state, selected) if selected else None
            )

        state_keys = tuple(keys_by_state.values())
        aligned = bool(state_keys[0]) and all(
            keys == state_keys[0] for keys in state_keys[1:]
        )
        return cls(
            summary(CacheState.COLD),
            summary(CacheState.WARM),
            summary(CacheState.HOT),
            aligned,
        )

    @property
    def complete(self) -> bool:
        return self.workload_aligned and all(
            summary is not None and summary.total > 0 for summary in self.summaries
        )

    @property
    def summaries(self) -> tuple[PerformanceSummary | None, ...]:
        return (self.cold, self.warm, self.hot)


def summarize_cache_metrics(metrics: Iterable[PerformanceMetric]) -> CacheProfile:
    return CacheProfile.from_metrics(metrics)


@dataclass(frozen=True, slots=True)
class LogitTolerance:
    atol: float
    rtol: float

    def __post_init__(self) -> None:
        _finite_nonnegative(self.atol, "logit atol")
        _finite_nonnegative(self.rtol, "logit rtol")


LOGIT_TOLERANCES: Mapping[str, LogitTolerance] = {
    "float64": LogitTolerance(1e-8, 1e-7),
    "float32": LogitTolerance(1e-5, 1e-5),
    "bfloat16": LogitTolerance(5e-2, 5e-2),
    "float16": LogitTolerance(2e-2, 2e-2),
    "fp8_e4m3": LogitTolerance(5e-1, 5e-2),
    "fp4_e2m1": LogitTolerance(1.0, 1e-1),
}
_DTYPE_ALIASES = {
    "f64": "float64",
    "f32": "float32",
    "bf16": "bfloat16",
    "f16": "float16",
    "fp16": "float16",
    "f8_e4m3": "fp8_e4m3",
    "float8_e4m3fn": "fp8_e4m3",
    "fp4": "fp4_e2m1",
}


def logit_tolerance(dtype: str) -> LogitTolerance:
    normalized = str(dtype).lower().replace("torch.", "")
    normalized = _DTYPE_ALIASES.get(normalized, normalized)
    try:
        return LOGIT_TOLERANCES[normalized]
    except KeyError as exc:
        raise BenchmarkContractError(
            f"no preregistered tolerance for dtype {dtype!r}"
        ) from exc


def arrays_bit_identical(reference: Any, candidate: Any) -> bool:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is an optional project extra
        torch = None
    if torch is not None and (
        isinstance(reference, torch.Tensor) or isinstance(candidate, torch.Tensor)
    ):
        if not isinstance(reference, torch.Tensor) or not isinstance(
            candidate, torch.Tensor
        ):
            return False
        left_tensor = reference.detach().to("cpu").contiguous()
        right_tensor = candidate.detach().to("cpu").contiguous()
        if (
            left_tensor.dtype != right_tensor.dtype
            or left_tensor.shape != right_tensor.shape
        ):
            return False
        left_bytes = left_tensor.view(torch.uint8).numpy().tobytes()
        right_bytes = right_tensor.view(torch.uint8).numpy().tobytes()
        return left_bytes == right_bytes
    left = np.asarray(reference)
    right = np.asarray(candidate)
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def _numpy_float64(value: Any) -> np.ndarray:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is an optional project extra
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float64).numpy()
    return np.asarray(value, dtype=np.float64)


def maximum_future_attention_mass(
    attention: Any, *, query_start: int = 0, key_start: int = 0
) -> float:
    """Largest absolute attention value assigned to a causally future key."""

    weights = _numpy_float64(attention)
    if weights.ndim < 2:
        raise BenchmarkContractError("attention must have query and key dimensions")
    q_len, k_len = weights.shape[-2:]
    query_positions = np.arange(query_start, query_start + q_len)[:, None]
    key_positions = np.arange(key_start, key_start + k_len)[None, :]
    future = key_positions > query_positions
    if not np.any(future):
        return 0.0
    expanded = np.broadcast_to(future, weights.shape)
    selected = np.abs(weights[expanded])
    if not np.all(np.isfinite(selected)):
        raise BenchmarkContractError("future attention contains non-finite values")
    return float(selected.max(initial=0.0))


@dataclass(frozen=True, slots=True)
class DecoderParityMetric:
    item_id: str
    seed: int
    dtype: str
    max_abs_logit_error: float
    max_allowed_logit_error: float
    top1_match: bool
    sequence_match: bool
    sequence_checked: bool
    within_logit_tolerance: bool
    missing_tensors: tuple[str, ...] = ()
    fallbacks: tuple[str, ...] = ()
    alpha0_bit_identical: bool | None = None
    future_attention_mass: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.item_id, "parity item id")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise BenchmarkContractError("seed must be a non-negative integer")
        logit_tolerance(self.dtype)
        object.__setattr__(
            self,
            "max_abs_logit_error",
            _finite_nonnegative(self.max_abs_logit_error, "max_abs_logit_error"),
        )
        object.__setattr__(
            self,
            "max_allowed_logit_error",
            _finite_nonnegative(
                self.max_allowed_logit_error, "max_allowed_logit_error"
            ),
        )
        object.__setattr__(self, "missing_tensors", tuple(self.missing_tensors))
        object.__setattr__(self, "fallbacks", tuple(self.fallbacks))
        if any(
            not isinstance(value, str) or not value for value in self.missing_tensors
        ):
            raise BenchmarkContractError("missing_tensors must contain names")
        if any(not isinstance(value, str) or not value for value in self.fallbacks):
            raise BenchmarkContractError("fallbacks must contain names")
        if self.future_attention_mass is not None:
            _finite_nonnegative(self.future_attention_mass, "future_attention_mass")
        if self.error is not None:
            _require_text(self.error, "parity error")

    @property
    def passed(self) -> bool:
        return (
            self.error is None
            and not self.missing_tensors
            and not self.fallbacks
            and self.top1_match
            and self.sequence_match
            and self.within_logit_tolerance
        )


def evaluate_decoder_parity(
    item_id: str,
    reference_logits: Any,
    candidate_logits: Any,
    *,
    seed: int,
    dtype: str,
    reference_sequence: Sequence[int] | None = None,
    candidate_sequence: Sequence[int] | None = None,
    tolerance: LogitTolerance | None = None,
    missing_tensors: Sequence[str] = (),
    fallbacks: Sequence[str] = (),
    alpha0_reference: Any | None = None,
    alpha0_candidate: Any | None = None,
    attention: Any | None = None,
    attention_query_start: int = 0,
    attention_key_start: int = 0,
    error: str | None = None,
) -> DecoderParityMetric:
    reference = _numpy_float64(reference_logits)
    candidate = _numpy_float64(candidate_logits)
    if reference.shape != candidate.shape or reference.size == 0:
        raise BenchmarkContractError(
            "parity logits must be non-empty and shape-identical"
        )
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise BenchmarkContractError("parity logits must be finite")
    selected_tolerance = tolerance or logit_tolerance(dtype)
    difference = np.abs(reference - candidate)
    allowed = selected_tolerance.atol + selected_tolerance.rtol * np.abs(reference)
    max_abs = float(difference.max())
    max_allowed = float(allowed.max())
    within = bool(np.all(difference <= allowed))
    top1 = bool(
        np.array_equal(np.argmax(reference, axis=-1), np.argmax(candidate, axis=-1))
    )
    if (reference_sequence is None) != (candidate_sequence is None):
        raise BenchmarkContractError("both parity sequences must be supplied together")
    sequence_checked = reference_sequence is not None
    if reference_sequence is None:
        sequence_match = top1
    else:
        sequence_match = tuple(reference_sequence) == tuple(candidate_sequence or ())
    if (alpha0_reference is None) != (alpha0_candidate is None):
        raise BenchmarkContractError("both alpha=0 outputs must be supplied together")
    alpha0_equal = (
        None
        if alpha0_reference is None
        else arrays_bit_identical(alpha0_reference, alpha0_candidate)
    )
    future_mass = (
        None
        if attention is None
        else maximum_future_attention_mass(
            attention,
            query_start=attention_query_start,
            key_start=attention_key_start,
        )
    )
    return DecoderParityMetric(
        item_id=item_id,
        seed=seed,
        dtype=dtype,
        max_abs_logit_error=max_abs,
        max_allowed_logit_error=max_allowed,
        top1_match=top1,
        sequence_match=sequence_match,
        sequence_checked=sequence_checked,
        within_logit_tolerance=within,
        missing_tensors=tuple(missing_tensors),
        fallbacks=tuple(fallbacks),
        alpha0_bit_identical=alpha0_equal,
        future_attention_mass=future_mass,
        error=error,
    )


@dataclass(frozen=True, slots=True)
class DecoderParitySummary:
    total: int
    passed: int
    errors: int
    top1_matches: int
    sequence_checks: int
    sequence_matches: int
    within_tolerance: int
    missing_tensors: int
    fallbacks: int
    alpha0_checks: int
    alpha0_matches: int
    future_mass_checks: int
    max_future_attention_mass: float | None
    max_abs_logit_error: float

    @classmethod
    def from_metrics(
        cls, metrics: Iterable[DecoderParityMetric]
    ) -> "DecoderParitySummary":
        rows = tuple(metrics)
        future = tuple(
            row.future_attention_mass
            for row in rows
            if row.future_attention_mass is not None
        )
        return cls(
            total=len(rows),
            passed=sum(row.passed for row in rows),
            errors=sum(row.error is not None for row in rows),
            top1_matches=sum(row.top1_match for row in rows),
            sequence_checks=sum(row.sequence_checked for row in rows),
            sequence_matches=sum(row.sequence_match for row in rows),
            within_tolerance=sum(row.within_logit_tolerance for row in rows),
            missing_tensors=sum(len(row.missing_tensors) for row in rows),
            fallbacks=sum(len(row.fallbacks) for row in rows),
            alpha0_checks=sum(row.alpha0_bit_identical is not None for row in rows),
            alpha0_matches=sum(row.alpha0_bit_identical is True for row in rows),
            future_mass_checks=len(future),
            max_future_attention_mass=max(future) if future else None,
            max_abs_logit_error=max(
                (row.max_abs_logit_error for row in rows), default=0.0
            ),
        )


def summarize_decoder_parity(
    metrics: Iterable[DecoderParityMetric],
) -> DecoderParitySummary:
    return DecoderParitySummary.from_metrics(metrics)


@dataclass(frozen=True, slots=True)
class PairedAblationSummary:
    observations: int
    pairs: int
    seeds: tuple[int, ...]
    candidate_accuracy: float
    baseline_accuracy: float
    accuracy_delta: float
    seed_delta_mean: float
    seed_delta_standard_error: float
    z_score: float | None
    wins: int
    ties: int
    losses: int
    discordant_pairs: int
    exact_one_sided_p_value: float
    inference_method: str
    candidate_wrong_rate: float
    baseline_wrong_rate: float
    candidate_coverage: float
    baseline_coverage: float
    candidate_mean_source_bytes: float | None
    baseline_mean_source_bytes: float | None
    source_byte_cache_state: CacheState | None
    source_byte_comparison_valid: bool

    def exceeds_sigma(self, sigma: float) -> bool:
        """Legacy diagnostic only; zero-variance repeats are never evidence."""

        threshold = _finite_nonnegative(sigma, "sigma")
        if self.seed_delta_mean <= 0:
            return False
        if self.seed_delta_standard_error == 0:
            return False
        assert self.z_score is not None
        return self.z_score > threshold

    def passes_exact_paired_gate(self, maximum_p_value: float) -> bool:
        threshold = _finite_nonnegative(maximum_p_value, "maximum_p_value")
        return (
            self.accuracy_delta > 0
            and self.wins > self.losses
            and self.discordant_pairs > 0
            and self.exact_one_sided_p_value < threshold
        )


def _unique_items(
    items: Iterable[ItemMetric], label: str
) -> dict[tuple[str, str, int], ItemMetric]:
    result: dict[tuple[str, str, int], ItemMetric] = {}
    for item in items:
        key = (item.task.value, item.item_id, item.seed)
        if key in result:
            raise BenchmarkContractError(f"duplicate {label} item: {key}")
        result[key] = item
    return result


def _mean_source_bytes(
    metrics: Iterable[PerformanceMetric] | None,
    keys: set[tuple[str, str, int]],
    *,
    cache_state: CacheState,
) -> float | None:
    if metrics is None:
        return None
    by_key: dict[tuple[str, int], PerformanceMetric] = {}
    for metric in metrics:
        if metric.cache_state is not cache_state:
            continue
        key = (metric.item_id, metric.seed)
        if key in by_key:
            raise BenchmarkContractError(
                f"duplicate performance metric: {key + (cache_state.value,)}"
            )
        by_key[key] = metric
    wanted = {(item_id, seed) for _, item_id, seed in keys}
    if set(by_key) != wanted:
        raise BenchmarkContractError(
            "performance metrics do not match paired item coverage"
        )
    return (
        statistics.fmean(metric.source_bytes for metric in by_key.values())
        if by_key
        else 0.0
    )


def summarize_paired_ablation(
    candidate: Iterable[ItemMetric],
    baseline: Iterable[ItemMetric],
    *,
    candidate_performance: Iterable[PerformanceMetric] | None = None,
    baseline_performance: Iterable[PerformanceMetric] | None = None,
    cache_state: CacheState | str = CacheState.COLD,
) -> PairedAblationSummary:
    candidate_by_key = _unique_items(candidate, "candidate")
    baseline_by_key = _unique_items(baseline, "baseline")
    if set(candidate_by_key) != set(baseline_by_key):
        raise BenchmarkContractError(
            "paired variants must cover exactly the same item/seed keys"
        )
    if not candidate_by_key:
        raise BenchmarkContractError("paired summary requires at least one pair")
    keys = set(candidate_by_key)
    item_keys = tuple(sorted({(task, item_id) for task, item_id, _ in keys}))

    def grouped_rate(
        rows: Mapping[tuple[str, str, int], ItemMetric],
        item_key: tuple[str, str],
        predicate: Any,
    ) -> float:
        selected = [
            row
            for (task, item_id, _), row in rows.items()
            if (task, item_id) == item_key
        ]
        return sum(bool(predicate(row)) for row in selected) / len(selected)

    candidate_accuracy_by_item = [
        grouped_rate(candidate_by_key, item_key, lambda row: row.correct)
        for item_key in item_keys
    ]
    baseline_accuracy_by_item = [
        grouped_rate(baseline_by_key, item_key, lambda row: row.correct)
        for item_key in item_keys
    ]
    candidate_wrong_by_item = [
        grouped_rate(
            candidate_by_key,
            item_key,
            lambda row: row.status is OutcomeStatus.INCORRECT,
        )
        for item_key in item_keys
    ]
    baseline_wrong_by_item = [
        grouped_rate(
            baseline_by_key,
            item_key,
            lambda row: row.status is OutcomeStatus.INCORRECT,
        )
        for item_key in item_keys
    ]
    candidate_coverage_by_item = [
        grouped_rate(
            candidate_by_key,
            item_key,
            lambda row: row.status in {OutcomeStatus.CORRECT, OutcomeStatus.INCORRECT},
        )
        for item_key in item_keys
    ]
    baseline_coverage_by_item = [
        grouped_rate(
            baseline_by_key,
            item_key,
            lambda row: row.status in {OutcomeStatus.CORRECT, OutcomeStatus.INCORRECT},
        )
        for item_key in item_keys
    ]
    wins = ties = losses = 0
    for candidate_rate, baseline_rate in zip(
        candidate_accuracy_by_item, baseline_accuracy_by_item, strict=True
    ):
        delta = candidate_rate - baseline_rate
        wins += delta > 0
        ties += delta == 0
        losses += delta < 0
    discordant = wins + losses
    exact_p = (
        sum(
            math.comb(discordant, successes)
            for successes in range(wins, discordant + 1)
        )
        / (2**discordant)
        if discordant
        else 1.0
    )
    seeds = tuple(sorted({key[2] for key in keys}))
    seed_deltas: list[float] = []
    for seed in seeds:
        seed_keys = tuple(key for key in keys if key[2] == seed)
        candidate_acc = sum(candidate_by_key[key].correct for key in seed_keys) / len(
            seed_keys
        )
        baseline_acc = sum(baseline_by_key[key].correct for key in seed_keys) / len(
            seed_keys
        )
        seed_deltas.append(candidate_acc - baseline_acc)
    mean_delta = statistics.fmean(seed_deltas)
    standard_error = (
        statistics.stdev(seed_deltas) / math.sqrt(len(seed_deltas))
        if len(seed_deltas) > 1
        else 0.0
    )
    z_score = mean_delta / standard_error if standard_error else None
    state = CacheState(cache_state)
    candidate_accuracy = statistics.fmean(candidate_accuracy_by_item)
    baseline_accuracy = statistics.fmean(baseline_accuracy_by_item)
    candidate_source_bytes = _mean_source_bytes(
        candidate_performance, keys, cache_state=state
    )
    baseline_source_bytes = _mean_source_bytes(
        baseline_performance, keys, cache_state=state
    )
    has_paired_source_bytes = (
        candidate_source_bytes is not None and baseline_source_bytes is not None
    )
    return PairedAblationSummary(
        observations=len(keys),
        pairs=len(item_keys),
        seeds=seeds,
        candidate_accuracy=candidate_accuracy,
        baseline_accuracy=baseline_accuracy,
        accuracy_delta=candidate_accuracy - baseline_accuracy,
        seed_delta_mean=mean_delta,
        seed_delta_standard_error=standard_error,
        z_score=z_score,
        wins=wins,
        ties=ties,
        losses=losses,
        discordant_pairs=discordant,
        exact_one_sided_p_value=exact_p,
        inference_method="exact_paired_sign_on_unique_items",
        candidate_wrong_rate=statistics.fmean(candidate_wrong_by_item),
        baseline_wrong_rate=statistics.fmean(baseline_wrong_by_item),
        candidate_coverage=statistics.fmean(candidate_coverage_by_item),
        baseline_coverage=statistics.fmean(baseline_coverage_by_item),
        candidate_mean_source_bytes=candidate_source_bytes,
        baseline_mean_source_bytes=baseline_source_bytes,
        source_byte_cache_state=state if has_paired_source_bytes else None,
        source_byte_comparison_valid=(
            has_paired_source_bytes and state is not CacheState.UNCONTROLLED
        ),
    )


@dataclass(frozen=True, slots=True)
class StopGoPolicy:
    selection_max_exact_p_value: float = 0.02275
    selection_min_variable_seeds: int = 5
    min_reference_similarity: float = 0.95
    max_accuracy_gap: float = 0.05
    min_coverage: float = 0.95
    max_wrong_rate_delta: float = 0.02
    future_mass_tolerance: float = 0.0
    equality_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        for label in (
            "selection_max_exact_p_value",
            "min_reference_similarity",
            "max_accuracy_gap",
            "min_coverage",
            "max_wrong_rate_delta",
            "future_mass_tolerance",
            "equality_tolerance",
        ):
            _finite_nonnegative(getattr(self, label), label)
        if self.min_reference_similarity > 1 or self.min_coverage > 1:
            raise BenchmarkContractError("ratio thresholds cannot exceed one")
        if self.selection_max_exact_p_value > 1:
            raise BenchmarkContractError(
                "selection p-value threshold cannot exceed one"
            )
        if (
            isinstance(self.selection_min_variable_seeds, bool)
            or not isinstance(self.selection_min_variable_seeds, int)
            or self.selection_min_variable_seeds < 1
        ):
            raise BenchmarkContractError(
                "selection_min_variable_seeds must be a positive integer"
            )


@dataclass(frozen=True, slots=True)
class GateDecision:
    name: str
    passed: bool
    reasons: tuple[str, ...]
    evidence: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class StopGoDecision:
    gates: tuple[GateDecision, ...]

    @property
    def go(self) -> bool:
        return bool(self.gates) and all(gate.passed for gate in self.gates)

    @property
    def verdict(self) -> str:
        return "go" if self.go else "stop"

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value({"verdict": self.verdict, "gates": self.gates})


def _gate(
    name: str, checks: Sequence[tuple[bool, str]], evidence: Mapping[str, Any]
) -> GateDecision:
    reasons = tuple(reason for passed, reason in checks if not passed)
    return GateDecision(name, not reasons, reasons, evidence)


def calculate_stop_go(
    *,
    provenance: BenchmarkProvenance,
    decoder: DecoderParitySummary | None,
    selection: PairedAblationSummary | None,
    graft: PairedAblationSummary | None,
    shuffled: PairedAblationSummary | None,
    candidate: OutcomeSummary | None,
    reference: OutcomeSummary | None,
    cache_profile: CacheProfile | None,
    policy: StopGoPolicy | None = None,
) -> StopGoDecision:
    """Apply the preregistered gates; absent evidence is always a stop."""

    rules = policy or StopGoPolicy()
    provenance_gate = _gate(
        "provenance",
        [(provenance.revision_is_pinned, "model revision is not an immutable digest")],
        {"model_id": provenance.model_id, "revision": provenance.model_revision},
    )

    if decoder is None:
        decoder_gate = _gate(
            "decoder_parity", [(False, "decoder parity was not measured")], {}
        )
    else:
        decoder_gate = _gate(
            "decoder_parity",
            [
                (decoder.total > 0, "decoder parity has no items"),
                (decoder.passed == decoder.total, "not every decoder item passed"),
                (decoder.errors == 0, "decoder parity contains runtime errors"),
                (decoder.top1_matches == decoder.total, "greedy top-1 mismatch"),
                (
                    decoder.sequence_checks == decoder.total,
                    "greedy sequence parity was not explicitly measured",
                ),
                (decoder.sequence_matches == decoder.total, "greedy sequence mismatch"),
                (
                    decoder.within_tolerance == decoder.total,
                    "logits exceed dtype tolerance",
                ),
                (decoder.missing_tensors == 0, "checkpoint tensors are missing"),
                (decoder.fallbacks == 0, "decoder used a fallback path"),
                (
                    decoder.alpha0_checks == decoder.total
                    and decoder.alpha0_matches == decoder.total,
                    "CRSA alpha=0 is not bit-identical",
                ),
                (
                    decoder.future_mass_checks == decoder.total
                    and decoder.max_future_attention_mass is not None
                    and decoder.max_future_attention_mass
                    <= rules.future_mass_tolerance,
                    "causal attention assigns non-zero future mass",
                ),
            ],
            asdict(decoder),
        )

    if selection is None:
        selection_gate = _gate(
            "selection", [(False, "selection/placebo pairs are missing")], {}
        )
    else:
        selection_gate = _gate(
            "selection",
            [
                (
                    selection.inference_method == "exact_paired_sign_on_unique_items",
                    "selection did not use independent item-level paired evidence",
                ),
                (
                    len(selection.seeds) >= rules.selection_min_variable_seeds,
                    "selection has fewer than the preregistered variable seeds",
                ),
                (
                    selection.passes_exact_paired_gate(
                        rules.selection_max_exact_p_value
                    ),
                    "selection fails the exact one-sided paired sign gate",
                ),
            ],
            asdict(selection),
        )

    if graft is None or shuffled is None:
        graft_gate = _gate(
            "graft", [(False, "graft or shuffled ablation pairs are missing")], {}
        )
    else:
        quality_improved = graft.accuracy_delta > rules.equality_tolerance
        quality_equal = abs(graft.accuracy_delta) <= rules.equality_tolerance
        bytes_lower = (
            graft.source_byte_comparison_valid
            and graft.candidate_mean_source_bytes is not None
            and graft.baseline_mean_source_bytes is not None
            and graft.candidate_mean_source_bytes < graft.baseline_mean_source_bytes
        )
        graft_gate = _gate(
            "graft",
            [
                (
                    quality_improved or (quality_equal and bytes_lower),
                    "graft neither improves paired quality nor preserves it with fewer bytes",
                ),
                (
                    graft.candidate_wrong_rate
                    <= graft.baseline_wrong_rate + rules.equality_tolerance,
                    "graft increases wrong-answer rate",
                ),
                (
                    shuffled.accuracy_delta > rules.equality_tolerance,
                    "shuffled graft is not worse than the real graft",
                ),
            ],
            {"real_vs_baseline": asdict(graft), "real_vs_shuffled": asdict(shuffled)},
        )

    if (
        candidate is None
        or reference is None
        or not candidate.total
        or not reference.total
    ):
        poc_gate = _gate(
            "frontier_similarity",
            [(False, "candidate/reference outcomes are missing")],
            {},
        )
    else:
        gap = reference.accuracy - candidate.accuracy
        ratio_pass = (
            candidate.accuracy >= rules.min_reference_similarity * reference.accuracy
        )
        gap_pass = gap <= rules.max_accuracy_gap
        poc_gate = _gate(
            "frontier_similarity",
            [
                (
                    ratio_pass or gap_pass,
                    "candidate is below both the accuracy-ratio and percentage-point limits",
                ),
                (
                    candidate.coverage >= rules.min_coverage,
                    "candidate coverage is too low",
                ),
                (
                    candidate.wrong_rate
                    <= reference.wrong_rate + rules.max_wrong_rate_delta,
                    "candidate wrong-answer rate exceeds the reference allowance",
                ),
            ],
            {
                "candidate": asdict(candidate),
                "reference": asdict(reference),
                "accuracy_gap": gap,
                "reference_ratio": (
                    candidate.accuracy / reference.accuracy
                    if reference.accuracy
                    else 1.0
                ),
            },
        )

    if cache_profile is None:
        cache_gate = _gate(
            "cache_profile", [(False, "cold/warm/hot profile is missing")], {}
        )
    else:
        cache_errors = sum(
            summary.errors for summary in cache_profile.summaries if summary is not None
        )
        cache_gate = _gate(
            "cache_profile",
            [
                (
                    cache_profile.complete,
                    "cold, warm, and hot measurements on the same workload are required",
                ),
                (cache_errors == 0, "cache profile contains failed measurements"),
            ],
            {
                "workload_aligned": cache_profile.workload_aligned,
                "states": {
                    state.value: (None if summary is None else asdict(summary))
                    for state, summary in zip(
                        (CacheState.COLD, CacheState.WARM, CacheState.HOT),
                        cache_profile.summaries,
                        strict=True,
                    )
                },
            },
        )
    return StopGoDecision(
        (
            provenance_gate,
            decoder_gate,
            selection_gate,
            graft_gate,
            poc_gate,
            cache_gate,
        )
    )


@dataclass(frozen=True, slots=True)
class RunMetrics:
    outcomes: OutcomeSummary
    cache: CacheProfile
    decoder: DecoderParitySummary


@dataclass(frozen=True, slots=True)
class BenchmarkRun:
    provenance: BenchmarkProvenance
    items: tuple[ItemMetric, ...]
    performance: tuple[PerformanceMetric, ...] = ()
    decoder_parity: tuple[DecoderParityMetric, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise BenchmarkContractError(
                f"unsupported benchmark schema: {self.schema_version!r}"
            )
        object.__setattr__(self, "items", tuple(self.items))
        object.__setattr__(self, "performance", tuple(self.performance))
        object.__setattr__(self, "decoder_parity", tuple(self.decoder_parity))
        _unique_items(self.items, "run")
        performance_keys: set[tuple[str, int, str]] = set()
        for metric in self.performance:
            key = (metric.item_id, metric.seed, metric.cache_state.value)
            if key in performance_keys:
                raise BenchmarkContractError(f"duplicate run performance metric: {key}")
            performance_keys.add(key)
        parity_keys: set[tuple[str, int]] = set()
        for metric in self.decoder_parity:
            key = (metric.item_id, metric.seed)
            if key in parity_keys:
                raise BenchmarkContractError(
                    f"duplicate run decoder parity metric: {key}"
                )
            parity_keys.add(key)
        canonical_json_bytes(self.metadata)
        if self.provenance.dataset.num_items:
            observed_ids = {item.item_id for item in self.items}
            if len(observed_ids) != self.provenance.dataset.num_items:
                raise BenchmarkContractError(
                    "run item coverage does not equal the pinned dataset"
                )
        if self.performance:
            item_keys = {(item.item_id, item.seed) for item in self.items}
            states = {metric.cache_state for metric in self.performance}
            for state in states:
                state_keys = {
                    (metric.item_id, metric.seed)
                    for metric in self.performance
                    if metric.cache_state is state
                }
                if state_keys != item_keys:
                    raise BenchmarkContractError(
                        f"{state.value} performance coverage does not equal run items"
                    )

    def summarize(self) -> RunMetrics:
        return RunMetrics(
            OutcomeSummary.from_items(self.items),
            CacheProfile.from_metrics(self.performance),
            DecoderParitySummary.from_metrics(self.decoder_parity),
        )

    def to_dict(self) -> dict[str, Any]:
        return _canonical_value(self)

    def digest(self) -> str:
        return canonical_digest(self)

    @property
    def run_id(self) -> str:
        return self.digest()[:16]


__all__ = [
    "SCHEMA_VERSION",
    "BenchmarkContractError",
    "BenchmarkTask",
    "OutcomeStatus",
    "CacheState",
    "DatasetProvenance",
    "BenchmarkProvenance",
    "ItemMetric",
    "OutcomeSummary",
    "PerformanceMetric",
    "PerformanceSummary",
    "CacheProfile",
    "LogitTolerance",
    "DecoderParityMetric",
    "DecoderParitySummary",
    "PairedAblationSummary",
    "StopGoPolicy",
    "GateDecision",
    "StopGoDecision",
    "RunMetrics",
    "BenchmarkRun",
    "canonical_json_bytes",
    "canonical_digest",
    "dataset_digest",
    "summarize_outcomes",
    "evaluate_mmlu",
    "extract_gsm8k_answer",
    "evaluate_gsm8k",
    "summarize_cache_metrics",
    "logit_tolerance",
    "arrays_bit_identical",
    "maximum_future_attention_mass",
    "evaluate_decoder_parity",
    "summarize_decoder_parity",
    "summarize_paired_ablation",
    "calculate_stop_go",
]
