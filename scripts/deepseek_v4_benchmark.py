#!/usr/bin/env python3
"""Resumable, resource-bounded benchmark runner for streamed DeepSeek-V4.

The runner performs inference only.  It does not package or upload anything.
Datasets may be JSON or JSONL and should preferably carry pre-tokenized
``prompt_token_ids`` plus one-token ``candidate_token_ids``.  This makes the
offline fixture path dependency-free and lets MMLU use exact candidate-head
scores without scanning the one-gibibyte vocabulary head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from immer.knowledge import Streamer
from immer.runtimes.deepseek_v4 import (
    DeepSeekV4Config,
    DeepSeekWeightPager,
    StreamedDeepSeekV4,
)
from immer.runtimes.deepseek_v4.benchmark import (
    SCHEMA_VERSION,
    BenchmarkProvenance,
    BenchmarkRun,
    BenchmarkTask,
    CacheState,
    DatasetProvenance,
    ItemMetric,
    OutcomeStatus,
    PerformanceMetric,
    PerformanceSummary,
    canonical_digest,
    canonical_json_bytes,
    evaluate_gsm8k,
    evaluate_mmlu,
    extract_gsm8k_answer,
    summarize_paired_ablation,
)
from immer.runtimes.deepseek_v4.encoding import encode_user_prompt
from immer.runtimes.deepseek_v4.graft import DeepSeekV4CrsaGraft, GRAFT_MODES


ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_SOURCE = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
DEFAULT_CACHE = ROOT / "artifacts" / "private" / "deepseek-v4-benchmark-cache"
JOURNAL_SCHEMA = "immer.deepseek-v4-benchmark-journal/v1"
REPORT_SCHEMA = "immer.deepseek-v4-benchmark-report/v1"
_DIGEST = re.compile(r"[0-9a-f]{64}")
ASSISTANT_GENERATION_PREFIX = "Answer:"
TEXT_PROMPT_PROTOCOL = "official_single_user_envelope_then_assistant_answer_prefix/v1"


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    item_id: str
    prompt_token_ids: tuple[int, ...]
    candidate_token_ids: tuple[int, ...] | None
    candidate_values: tuple[Any, ...]
    benchmark_protocol: str
    prompt_encoding: str
    prompt_protocol: str
    assistant_generation_prefix: str | None
    candidate_surfaces: tuple[str, ...]
    candidate_tokenization: str
    rendered_prompt_sha256: str | None


class RunnerError(RuntimeError):
    """The requested run cannot produce comparable benchmark evidence."""


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def _positive_float(raw: str) -> float:
    value = _nonnegative_float(raw)
    if value == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _json_line(document: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(document) + b"\n"


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(
                (
                    json.dumps(
                        document,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        indent=2,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class JsonlJournal:
    """Append-only item ledger with safe recovery of one torn final write."""

    def __init__(
        self,
        path: Path,
        *,
        header: Mapping[str, Any],
        resume: bool,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if not resume:
                raise RunnerError(f"journal exists; pass --resume: {self.path}")
            records = self._read_repairable()
            if not records or records[0].get("type") != "header":
                raise RunnerError("journal has no valid header")
            existing = records[0]
            if existing.get("schema") != JOURNAL_SCHEMA:
                raise RunnerError("journal header uses an unknown schema")
            if existing.get("signature") != header.get("signature"):
                raise RunnerError("resume signature differs from the existing journal")
            self.records = records
        else:
            self.records = [dict(header)]
            self._append_record(header, create=True)
        self.completed: dict[tuple[str, str, int], dict[str, Any]] = {}
        for record in self.records[1:]:
            if record.get("type") != "item":
                raise RunnerError("journal contains an unknown record type")
            if record.get("schema") != JOURNAL_SCHEMA:
                raise RunnerError("journal item uses an unknown schema")
            if record.get("signature") != header.get("signature"):
                raise RunnerError("journal item signature differs from its header")
            key = _record_key(record)
            if key in self.completed:
                raise RunnerError(f"journal contains duplicate item key: {key}")
            self.completed[key] = record

    def _read_repairable(self) -> list[dict[str, Any]]:
        raw = self.path.read_bytes()
        records: list[dict[str, Any]] = []
        valid_bytes = 0
        lines = raw.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not line.strip():
                valid_bytes += len(line)
                continue
            try:
                value = json.loads(line)
            except (UnicodeError, json.JSONDecodeError) as exc:
                torn_tail = index == len(lines) - 1 and not line.endswith(b"\n")
                if not torn_tail:
                    raise RunnerError(f"corrupt journal record {index + 1}") from exc
                with self.path.open("r+b") as handle:
                    handle.truncate(valid_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                break
            if not isinstance(value, dict):
                raise RunnerError(f"journal record {index + 1} is not an object")
            records.append(value)
            valid_bytes += len(line)
        return records

    def _append_record(
        self, record: Mapping[str, Any], *, create: bool = False
    ) -> None:
        flags = os.O_WRONLY | os.O_APPEND
        if create:
            flags |= os.O_CREAT | os.O_EXCL
        descriptor = os.open(self.path, flags, 0o600)
        try:
            body = _json_line(record)
            written = os.write(descriptor, body)
            if written != len(body):
                raise OSError("short JSONL write")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def append_item(self, record: Mapping[str, Any]) -> None:
        key = _record_key(record)
        if key in self.completed:
            raise RunnerError(f"attempted to append completed item: {key}")
        self._append_record(record)
        copied = dict(record)
        self.records.append(copied)
        self.completed[key] = copied


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, int]:
    try:
        mode = record["mode"]
        item_id = record["item_id"]
        seed = record["seed"]
        if not isinstance(mode, str) or not mode:
            raise TypeError
        if not isinstance(item_id, str) or not item_id:
            raise TypeError
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise TypeError
        return mode, item_id, seed
    except (KeyError, TypeError, ValueError) as exc:
        raise RunnerError("journal item has an invalid key") from exc


def _read_dataset(path: Path, *, maximum_bytes: int) -> list[dict[str, Any]]:
    source = path.expanduser().resolve()
    size = source.stat().st_size
    if size > maximum_bytes:
        raise RunnerError(f"dataset exceeds --max-dataset-mb ({size} bytes)")
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RunnerError(
                "Parquet input requires pyarrow; install the local analysis dependency"
            ) from exc
        try:
            rows = parquet.read_table(source).to_pylist()
        except Exception as exc:
            raise RunnerError(f"cannot read dataset Parquet: {source}") from exc
    else:
        raw = source.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        rows: list[Any] = []
        for line_number, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RunnerError(f"invalid dataset JSONL line {line_number}") from exc
    elif suffix != ".parquet":
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RunnerError("invalid dataset JSON") from exc
        rows = document.get("items") if isinstance(document, dict) else document
    if not isinstance(rows, list) or not rows:
        raise RunnerError("dataset must contain a non-empty item list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RunnerError(f"dataset item {index} is not an object")
        raw_id = row.get("id", row.get("item_id"))
        item_id = f"row-{index:06d}" if raw_id is None else str(raw_id).strip()
        if not item_id:
            raise RunnerError(f"dataset item {index} has an empty id")
        if item_id in seen:
            raise RunnerError(f"duplicate dataset item id: {item_id}")
        seen.add(item_id)
        copied = dict(row)
        copied["id"] = item_id
        result.append(copied)
    return result


class LocalTokenizer:
    def __init__(self, path: Path) -> None:
        source = path.expanduser().resolve()
        self.location = str(source)
        self._initialize(source.read_bytes())

    @classmethod
    def from_bytes(cls, raw: bytes, *, location: str) -> "LocalTokenizer":
        instance = cls.__new__(cls)
        instance.location = location
        instance._initialize(raw)
        return instance

    def _initialize(self, raw: bytes) -> None:
        try:
            from transformers import PreTrainedTokenizerFast
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RunnerError(
                "text datasets require transformers or pre-tokenized IDs"
            ) from exc
        try:
            backend = Tokenizer.from_str(raw.decode("utf-8"))
        except Exception as exc:
            raise RunnerError(f"invalid tokenizer JSON at {self.location}") from exc
        self.raw = bytes(raw)
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()

    def encode(self, text: str) -> list[int]:
        return [
            int(value)
            for value in self.tokenizer.encode(text, add_special_tokens=False)
        ]

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=True))


def _needs_tokenizer(rows: Sequence[Mapping[str, Any]], task: BenchmarkTask) -> bool:
    for row in rows:
        if "prompt_token_ids" not in row:
            return True
        if "candidate_token_ids" not in row:
            # MMLU needs contextual choice-token encoding; canonical GSM8K
            # needs decoding after free generation.
            return True
    return False


def _load_tokenizer(
    args: argparse.Namespace,
    source: Streamer,
    rows: Sequence[Mapping[str, Any]],
    task: BenchmarkTask,
) -> LocalTokenizer | None:
    if args.tokenizer_json is not None:
        return LocalTokenizer(Path(args.tokenizer_json))
    if not _needs_tokenizer(rows, task):
        return None

    bundled = (
        ROOT / "artifacts" / "private" / "deepseek-v4-reference" / "tokenizer.json"
    )
    if bundled.is_file():
        return LocalTokenizer(bundled)

    reader = getattr(source, "reader", None)
    fetch_file = getattr(reader, "fetch_file", None)
    if not callable(fetch_file):
        raise RunnerError(
            "text dataset needs an official tokenizer; pass --tokenizer-json"
        )
    try:
        raw = fetch_file("tokenizer.json")
    except Exception as exc:
        raise RunnerError(
            "cannot obtain the official tokenizer; pass --tokenizer-json"
        ) from exc
    if not isinstance(raw, bytes) or not raw:
        raise RunnerError("source tokenizer.json is empty or not bytes")
    return LocalTokenizer.from_bytes(raw, location="source:tokenizer.json")


def _local_source(value: str) -> Path | None:
    raw = value.removeprefix("local:") if value.startswith("local:") else value
    candidate = Path(raw).expanduser()
    explicit_path = (
        value.startswith("local:")
        or candidate.is_absolute()
        or raw.startswith(("./", "../"))
    )
    if explicit_path:
        return candidate.resolve()
    return candidate.resolve() if candidate.is_dir() else None


def _build_source(args: argparse.Namespace) -> tuple[Streamer, str]:
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    common = {
        "revision": args.revision,
        "budget_mb": args.source_budget_mb,
        "cache_dir": cache_dir,
        "use_cache": not args.no_cache,
        "max_cache_bytes": int(args.cache_budget_mb * 1024**2),
        "verbose": False,
    }
    local = _local_source(args.source)
    if local is not None:
        if not local.is_dir():
            raise RunnerError(f"local checkpoint does not exist: {local}")
        return Streamer.from_local(local, **common), f"local:{local}"
    source = Streamer(args.source, **common)
    if re.fullmatch(r"[0-9a-fA-F]{40,64}", args.revision) is None:
        raise RunnerError("remote checkpoints require an immutable revision digest")
    return source, args.source


def _load_config(
    args: argparse.Namespace, source: Streamer
) -> tuple[DeepSeekV4Config, dict[str, Any]]:
    if args.config is None:
        raw = source.reader.fetch_file("config.json")
        location = "source:config.json"
    else:
        config_path = Path(args.config).expanduser().resolve()
        raw = config_path.read_bytes()
        location = str(config_path)
    try:
        document = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RunnerError(f"invalid config JSON at {location}") from exc
    if not isinstance(document, dict):
        raise RunnerError("config root must be an object")
    return DeepSeekV4Config.from_mapping(document), {
        "location": location,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _digest_or_canonical(value: Any, fallback: Any) -> str:
    text = str(value or "").lower()
    return text if _DIGEST.fullmatch(text) else canonical_digest(fallback)


def _parse_modes(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    for raw in values:
        for mode in raw.split(","):
            normalized = mode.strip().lower()
            if normalized not in GRAFT_MODES:
                raise RunnerError(f"unknown graft mode: {normalized!r}")
            if normalized in result:
                raise RunnerError(f"duplicate graft mode: {normalized}")
            result.append(normalized)
    if not result:
        raise RunnerError("at least one graft mode is required")
    return tuple(result)


def _parse_seeds(values: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for seed in values:
        if seed < 0:
            raise RunnerError("seeds must be non-negative")
        if seed in result:
            raise RunnerError(f"duplicate seed: {seed}")
        result.append(seed)
    if not result:
        raise RunnerError("at least one seed is required")
    return tuple(result)


def _mode_seeds(mode: str, seeds: Sequence[int]) -> tuple[int, ...]:
    """Only the shuffled routing placebo has a seed-dependent forward."""

    return tuple(seeds) if mode == "shuffle" else (int(seeds[0]),)


def _integer_ids(value: Any, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise RunnerError(f"{label} must be a non-empty integer list")
    ids: list[int] = []
    for token_id in value:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise RunnerError(f"{label} contains an invalid token ID")
        ids.append(token_id)
    return ids


def _user_content(row: Mapping[str, Any], task: BenchmarkTask) -> str:
    question = str(row.get("question", "")).strip()
    if not question:
        raise RunnerError(f"item {row['id']} has no question or prompt_token_ids")
    if task is BenchmarkTask.MMLU:
        choices = row.get("choices")
        if not isinstance(choices, list) or len(choices) < 2:
            raise RunnerError(f"MMLU item {row['id']} has no choices")
        rendered = "\n".join(
            f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices)
        )
        return f"{question}\n{rendered}"
    return question


def _prompt_ids(
    row: Mapping[str, Any],
    task: BenchmarkTask,
    tokenizer: LocalTokenizer | None,
    args: argparse.Namespace,
) -> tuple[list[int], str | None]:
    if "prompt_token_ids" in row:
        return _integer_ids(row["prompt_token_ids"], "prompt_token_ids"), None
    if tokenizer is None:
        raise RunnerError(f"item {row['id']} needs --tokenizer-json")
    envelope = encode_user_prompt(
        _user_content(row, task),
        thinking_mode=args.thinking_mode,
        reasoning_effort=args.reasoning_effort,
    )
    rendered = envelope + ASSISTANT_GENERATION_PREFIX
    ids = tokenizer.encode(rendered)
    if not ids:
        raise RunnerError(f"item {row['id']} tokenizes to an empty prompt")
    return ids, rendered


def _contextual_candidate_tokens(
    tokenizer: LocalTokenizer,
    rendered_prompt: str,
    prompt_ids: Sequence[int],
    surfaces: Sequence[str],
    *,
    item_id: str,
) -> list[int]:
    """Tokenize candidates in context and require an exact one-token suffix."""

    tokens: list[int] = []
    prefix = list(prompt_ids)
    for surface in surfaces:
        combined = tokenizer.encode(rendered_prompt + surface)
        if combined[: len(prefix)] != prefix:
            raise RunnerError(
                f"item {item_id} candidate {surface!r} retokenizes the prompt prefix"
            )
        suffix = combined[len(prefix) :]
        if len(suffix) != 1:
            raise RunnerError(
                f"item {item_id} candidate {surface!r} is not one token in context"
            )
        tokens.append(suffix[0])
    if len(tokens) != len(set(tokens)):
        raise RunnerError(f"item {item_id} candidate surfaces map to duplicate tokens")
    return tokens


def _candidate_tokens(
    row: Mapping[str, Any],
    task: BenchmarkTask,
    tokenizer: LocalTokenizer | None,
    *,
    prompt_ids: Sequence[int],
    rendered_prompt: str | None,
) -> tuple[list[int], list[Any], list[str], str] | None:
    raw = row.get("candidate_token_ids")
    values: list[Any]
    if raw is not None:
        if not isinstance(raw, list) or len(raw) < 2:
            raise RunnerError(
                "candidate_token_ids must contain at least two candidates"
            )
        tokens: list[int] = []
        for candidate in raw:
            if (
                isinstance(candidate, int)
                and not isinstance(candidate, bool)
                and candidate >= 0
            ):
                tokens.append(candidate)
            elif isinstance(candidate, list) and len(candidate) == 1:
                tokens.extend(_integer_ids(candidate, "candidate token"))
            else:
                raise RunnerError(
                    "candidate scoring currently requires exactly one token per candidate"
                )
        if task is BenchmarkTask.MMLU:
            values = list(range(len(tokens)))
            surfaces = [f" {chr(65 + index)}" for index in range(len(tokens))]
        else:
            answer_values = row.get("candidate_answers")
            if not isinstance(answer_values, list) or len(answer_values) != len(tokens):
                raise RunnerError(
                    "GSM8K candidate tokens require aligned candidate_answers"
                )
            values = list(answer_values)
            surfaces = [f" {value}" for value in answer_values]
        if len(tokens) != len(set(tokens)):
            raise RunnerError("candidate token IDs must be distinct")
        validation = "external_pretokenized_candidate_ids"
        if rendered_prompt is not None:
            if tokenizer is None:  # defensive: rendered prompts require a tokenizer
                raise AssertionError("rendered prompt has no tokenizer")
            contextual = _contextual_candidate_tokens(
                tokenizer,
                rendered_prompt,
                prompt_ids,
                surfaces,
                item_id=str(row["id"]),
            )
            if contextual != tokens:
                raise RunnerError(
                    f"item {row['id']} candidate_token_ids do not match "
                    "the recorded prompt and suffix surfaces"
                )
            validation = "full_prompt_exact_prefix_one_token_suffix_verified"
        return tokens, values, surfaces, validation
    if task is BenchmarkTask.MMLU:
        choices = row.get("choices")
        if not isinstance(choices, list) or len(choices) < 2:
            raise RunnerError(
                "MMLU choices are required when candidates are not pre-tokenized"
            )
        if tokenizer is None:
            raise RunnerError("MMLU item needs candidate_token_ids or --tokenizer-json")
        if rendered_prompt is None:
            raise AssertionError("text MMLU prompt was not rendered")
        surfaces = [f" {chr(65 + index)}" for index in range(len(choices))]
        tokens = _contextual_candidate_tokens(
            tokenizer,
            rendered_prompt,
            prompt_ids,
            surfaces,
            item_id=str(row["id"]),
        )
        return (
            tokens,
            list(range(len(tokens))),
            surfaces,
            "full_prompt_exact_prefix_one_token_suffix_verified",
        )
    return None


def _prepare_rows(
    rows: Sequence[Mapping[str, Any]],
    task: BenchmarkTask,
    tokenizer: LocalTokenizer | None,
    config: DeepSeekV4Config,
    args: argparse.Namespace,
) -> dict[str, EvaluationPlan]:
    if any(
        token_id < 0 or token_id >= config.vocab_size for token_id in args.eos_token_ids
    ):
        raise RunnerError("EOS token outside checkpoint vocabulary")
    plans: dict[str, EvaluationPlan] = {}
    for row in rows:
        if task is BenchmarkTask.MMLU:
            candidates = row.get("candidate_token_ids")
            choices = row.get("choices")
            count = (
                len(candidates) if isinstance(candidates, list) else len(choices or [])
            )
            if count < 2:
                raise RunnerError(
                    f"MMLU item {row['id']} needs at least two candidates"
                )
            try:
                evaluate_mmlu(
                    str(row["id"]), row.get("answer"), None, seed=0, num_choices=count
                )
            except ValueError as exc:
                raise RunnerError(f"invalid MMLU answer at {row['id']}") from exc
        elif extract_gsm8k_answer(row.get("answer")) is None:
            raise RunnerError(f"invalid GSM8K answer at {row['id']}")
        prompt, rendered_prompt = _prompt_ids(row, task, tokenizer, args)
        if len(prompt) > args.max_prompt_tokens:
            raise RunnerError(
                f"item {row['id']} prompt has {len(prompt)} tokens, "
                f"limit is {args.max_prompt_tokens}"
            )
        if any(token_id >= config.vocab_size for token_id in prompt):
            raise RunnerError(
                f"item {row['id']} prompt token outside checkpoint vocabulary"
            )
        candidates = _candidate_tokens(
            row,
            task,
            tokenizer,
            prompt_ids=prompt,
            rendered_prompt=rendered_prompt,
        )
        if candidates is None:
            if task is not BenchmarkTask.GSM8K or tokenizer is None:
                raise RunnerError(
                    f"item {row['id']} has no executable scoring protocol"
                )
            candidate_ids: tuple[int, ...] | None = None
            candidate_values: tuple[Any, ...] = ()
            candidate_surfaces: tuple[str, ...] = ()
            candidate_tokenization = "full_vocabulary_greedy_generation"
            protocol = "gsm8k_canonical_free_generation_exact_match"
        else:
            (
                raw_candidate_ids,
                raw_candidate_values,
                raw_candidate_surfaces,
                candidate_tokenization,
            ) = candidates
            if any(token_id >= config.vocab_size for token_id in raw_candidate_ids):
                raise RunnerError(
                    f"item {row['id']} candidate token outside checkpoint vocabulary"
                )
            candidate_ids = tuple(raw_candidate_ids)
            candidate_values = tuple(raw_candidate_values)
            candidate_surfaces = tuple(raw_candidate_surfaces)
            if task is BenchmarkTask.GSM8K:
                if not args.allow_closed_set_gsm8k:
                    raise RunnerError(
                        "GSM8K candidate_answers are closed-set diagnostics; "
                        "pass --allow-closed-set-gsm8k explicitly"
                    )
                normalized_answers = tuple(
                    extract_gsm8k_answer(value) for value in raw_candidate_values
                )
                if any(value is None for value in normalized_answers):
                    raise RunnerError(
                        f"item {row['id']} has a non-numeric GSM8K candidate answer"
                    )
                if len(normalized_answers) != len(set(normalized_answers)):
                    raise RunnerError(
                        f"item {row['id']} has duplicate GSM8K candidate answers"
                    )
                gold = extract_gsm8k_answer(row.get("answer"))
                if gold not in normalized_answers:
                    raise RunnerError(
                        f"item {row['id']} closed set does not contain the gold answer"
                    )
                candidate_values = normalized_answers
                protocol = "gsm8k_closed_set_candidate_token_diagnostic"
            else:
                protocol = "mmlu_constrained_single_token_choice"
        prompt_encoding = (
            "external_pretokenized_ids"
            if "prompt_token_ids" in row
            else (
                "official_single_user_message/"
                f"{args.thinking_mode}/{args.reasoning_effort}"
            )
        )
        prompt_protocol = (
            "external_pretokenized_prompt/v1"
            if rendered_prompt is None
            else TEXT_PROMPT_PROTOCOL
        )
        item_id = str(row["id"])
        plans[item_id] = EvaluationPlan(
            item_id=item_id,
            prompt_token_ids=tuple(prompt),
            candidate_token_ids=candidate_ids,
            candidate_values=candidate_values,
            benchmark_protocol=protocol,
            prompt_encoding=prompt_encoding,
            prompt_protocol=prompt_protocol,
            assistant_generation_prefix=(
                None if rendered_prompt is None else ASSISTANT_GENERATION_PREFIX
            ),
            candidate_surfaces=candidate_surfaces,
            candidate_tokenization=candidate_tokenization,
            rendered_prompt_sha256=(
                None
                if rendered_prompt is None
                else hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()
            ),
        )
    return plans


def _model_for_mode(
    config: DeepSeekV4Config,
    pager: DeepSeekWeightPager,
    args: argparse.Namespace,
    mode: str,
    seed: int,
) -> StreamedDeepSeekV4:
    graft = None
    graft_layer = None
    if mode != "off":
        graft = DeepSeekV4CrsaGraft(
            mode=mode,
            alpha=args.graft_alpha,
            max_history=args.max_prompt_tokens + args.max_new_tokens,
            shuffle_seed=args.graft_seed ^ seed,
        )
        graft_layer = (
            args.graft_layer if args.graft_layer is not None else config.n_layers // 2
        )
    return StreamedDeepSeekV4(
        config,
        pager,
        graft=graft,
        graft_layer=graft_layer,
        max_batch_size=1,
        max_seq_len=args.max_prompt_tokens + args.max_new_tokens,
    )


def _compact_prefill_evidence(rows: Sequence[Any]) -> dict[str, Any]:
    documents = [asdict(row) for row in rows]
    return {
        "forward_passes": len(rows),
        "source_body_bytes": sum(int(row.source_body_bytes) for row in rows),
        "linear_calls": sum(int(row.linear_calls) for row in rows),
        "seconds": sum(float(row.seconds) for row in rows),
        "attention_state_bytes": max(
            (int(row.attention_state_bytes) for row in rows), default=0
        ),
        "evidence_sha256": canonical_digest(documents),
    }


def _coalesced_candidate_logits(
    pager: DeepSeekWeightPager, hidden: Any, token_ids: Sequence[int]
) -> tuple[Any, list[list[int]]]:
    unique = sorted(set(int(token_id) for token_id in token_ids))
    runs: list[tuple[int, int]] = []
    start = previous = unique[0]
    for token_id in unique[1:]:
        if token_id != previous + 1:
            runs.append((start, previous + 1))
            start = token_id
        previous = token_id
    runs.append((start, previous + 1))
    # ParallelHead is an explicit FP32 path in the published runtime.  Reuse
    # the pager implementation so the benchmark cannot silently score in the
    # decoder's BF16 compute dtype.
    values = pager.candidate_logits(hidden, token_ids)
    return values, [[start, stop] for start, stop in runs]


def _score_item(
    model: StreamedDeepSeekV4,
    pager: DeepSeekWeightPager,
    plan: EvaluationPlan,
    tokenizer: LocalTokenizer | None,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    prompt = list(plan.prompt_token_ids)
    if plan.candidate_token_ids is not None:
        token_ids = list(plan.candidate_token_ids)
        hidden, forwards = model.prefill(
            [prompt], tokenwise=not args.batched_prefill, reset=True
        )
        logits, head_row_ranges = _coalesced_candidate_logits(
            pager, hidden[:, -1], token_ids
        )
        scores = [
            float(value)
            for value in logits[0].detach().to("cpu", dtype=model.torch.float32)
        ]
        if not all(math.isfinite(value) for value in scores):
            raise RunnerError("candidate logits are non-finite")
        winner = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        return plan.candidate_values[winner], {
            "engine": "candidate_token_head",
            "benchmark_protocol": plan.benchmark_protocol,
            "prompt_encoding": plan.prompt_encoding,
            "prompt_protocol": plan.prompt_protocol,
            "assistant_generation_prefix": plan.assistant_generation_prefix,
            "rendered_prompt_sha256": plan.rendered_prompt_sha256,
            "prompt_tokens": len(prompt),
            "candidate_token_ids": token_ids,
            "candidate_surfaces": list(plan.candidate_surfaces),
            "candidate_tokenization": plan.candidate_tokenization,
            "candidate_scores": scores,
            "selected_candidate": winner,
            "head_row_ranges": head_row_ranges,
            "prefill": _compact_prefill_evidence(forwards),
        }
    if tokenizer is None:
        raise RunnerError("free GSM8K generation requires --tokenizer-json")
    generated, evidence = model.generate_greedy(
        [prompt],
        max_new_tokens=args.max_new_tokens,
        prefill_tokenwise=not args.batched_prefill,
        eos_token_ids=args.eos_token_ids,
        head_block_rows=args.head_block_rows,
    )
    return tokenizer.decode(generated), {
        "engine": "greedy_full_head",
        "benchmark_protocol": plan.benchmark_protocol,
        "prompt_encoding": plan.prompt_encoding,
        "prompt_protocol": plan.prompt_protocol,
        "assistant_generation_prefix": plan.assistant_generation_prefix,
        "rendered_prompt_sha256": plan.rendered_prompt_sha256,
        "prompt_tokens": len(prompt),
        "candidate_tokenization": plan.candidate_tokenization,
        "generation": asdict(evidence),
    }


def _counter_delta(
    before: Mapping[str, Any], after: Mapping[str, Any], key: str
) -> int:
    try:
        return max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
    except (TypeError, ValueError):
        return 0


def _extended_metrics(
    source_before: Mapping[str, Any],
    source_after: Mapping[str, Any],
    pager_before: Mapping[str, Any],
    pager_after: Mapping[str, Any],
) -> dict[str, Any]:
    before_budget = source_before.get("budget", {})
    after_budget = source_after.get("budget", {})
    return {
        "source_body_bytes": _counter_delta(
            source_before, source_after, "network_or_source_body_bytes"
        ),
        "charged_overhead_bytes": _counter_delta(
            before_budget, after_budget, "overhead"
        ),
        "requests": _counter_delta(before_budget, after_budget, "requests"),
        "cache_bytes_reused": _counter_delta(
            source_before, source_after, "cache_bytes_reused"
        ),
        "cache_bytes_written": _counter_delta(
            source_before, source_after, "cache_bytes_written"
        ),
        "logical_weight_bytes": _counter_delta(
            pager_before, pager_after, "logical_weight_bytes"
        ),
        "materialized_float_bytes": _counter_delta(
            pager_before, pager_after, "materialized_float_bytes"
        ),
        "materialized_scale_bytes": _counter_delta(
            pager_before, pager_after, "materialized_scale_bytes"
        ),
        "linear_calls": _counter_delta(pager_before, pager_after, "linear_calls"),
        "head_rows": _counter_delta(pager_before, pager_after, "head_rows"),
    }


def _item_record(
    *,
    mode: str,
    seed: int,
    row: Mapping[str, Any],
    plan: EvaluationPlan,
    task: BenchmarkTask,
    model: StreamedDeepSeekV4,
    pager: DeepSeekWeightPager,
    source: Streamer,
    tokenizer: LocalTokenizer | None,
    args: argparse.Namespace,
    signature: str,
) -> dict[str, Any]:
    source_before = source.metrics()
    pager_before = pager.metrics()
    started = time.perf_counter()
    predicted: Any = None
    runtime: dict[str, Any] = {}
    error: str | None = None
    try:
        predicted, runtime = _score_item(model, pager, plan, tokenizer, args)
    except Exception as exc:  # item failures are benchmark outcomes
        error = f"{type(exc).__name__}: {exc}"[:1000]
        model.reset_state(release=True)
    latency_ms = (time.perf_counter() - started) * 1000.0
    source_after = source.metrics()
    pager_after = pager.metrics()
    if task is BenchmarkTask.MMLU:
        candidate_count = len(plan.candidate_token_ids or ())
        metric = evaluate_mmlu(
            str(row["id"]),
            row.get("answer"),
            predicted,
            seed=seed,
            num_choices=candidate_count,
            error=error,
            metadata={
                "mode": mode,
                "benchmark_protocol": plan.benchmark_protocol,
                "prompt_encoding": plan.prompt_encoding,
                "prompt_protocol": plan.prompt_protocol,
                "assistant_generation_prefix": plan.assistant_generation_prefix,
                "candidate_tokenization": plan.candidate_tokenization,
                "runtime": runtime,
            },
        )
    else:
        metric = evaluate_gsm8k(
            str(row["id"]),
            row.get("answer"),
            predicted,
            seed=seed,
            error=error,
            metadata={
                "mode": mode,
                "benchmark_protocol": plan.benchmark_protocol,
                "prompt_encoding": plan.prompt_encoding,
                "prompt_protocol": plan.prompt_protocol,
                "assistant_generation_prefix": plan.assistant_generation_prefix,
                "candidate_tokenization": plan.candidate_tokenization,
                "runtime": runtime,
            },
        )
    performance = PerformanceMetric.from_snapshots(
        str(row["id"]),
        seed,
        CacheState.UNCONTROLLED,
        latency_ms,
        source_before,
        source_after,
        output_tokens=(
            len(runtime.get("generation", {}).get("generated_token_ids", ()))
            if error is None
            else 0
        ),
        error=error,
    )
    measurements = _extended_metrics(
        source_before, source_after, pager_before, pager_after
    )
    return {
        "schema": JOURNAL_SCHEMA,
        "type": "item",
        "signature": signature,
        "mode": mode,
        "seed": seed,
        "item_id": str(row["id"]),
        "item": asdict(metric),
        "performance": asdict(performance),
        "measurements": measurements,
    }


def _restore_item(document: Mapping[str, Any]) -> ItemMetric:
    return ItemMetric(
        item_id=str(document["item_id"]),
        task=BenchmarkTask(document["task"]),
        seed=int(document["seed"]),
        status=OutcomeStatus(document["status"]),
        expected=document.get("expected"),
        predicted=document.get("predicted"),
        error=document.get("error"),
        metadata=document.get("metadata", {}),
    )


def _restore_performance(document: Mapping[str, Any]) -> PerformanceMetric:
    return PerformanceMetric(
        item_id=str(document["item_id"]),
        seed=int(document["seed"]),
        cache_state=CacheState(document["cache_state"]),
        latency_ms=float(document["latency_ms"]),
        source_bytes=int(document["source_bytes"]),
        cache_bytes=int(document["cache_bytes"]),
        requests=int(document["requests"]),
        output_tokens=int(document.get("output_tokens", 0)),
        error=document.get("error"),
    )


def _provenance_dict(value: BenchmarkProvenance) -> dict[str, Any]:
    return asdict(value)


def _provenance_from_dict(document: Mapping[str, Any]) -> BenchmarkProvenance:
    dataset = DatasetProvenance(**document["dataset"])
    return BenchmarkProvenance(
        model_id=document["model_id"],
        model_revision=document["model_revision"],
        inventory_sha256=document["inventory_sha256"],
        config_sha256=document["config_sha256"],
        tokenizer_sha256=document["tokenizer_sha256"],
        dataset=dataset,
        harness_revision=document["harness_revision"],
        graft=document.get("graft", "none"),
        command=tuple(document.get("command", ())),
        protocol=document.get("protocol", {}),
    )


def _build_report(
    journal: JsonlJournal,
    *,
    source: Streamer,
    pager: DeepSeekWeightPager,
    args: argparse.Namespace,
    modes: Sequence[str],
    seeds: Sequence[int],
    selected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    header = journal.records[0]
    base_provenance = _provenance_from_dict(header["provenance"])
    records = list(journal.completed.values())
    by_mode: dict[str, dict[str, Any]] = {}
    mode_items: dict[str, tuple[ItemMetric, ...]] = {}
    mode_performance: dict[str, tuple[PerformanceMetric, ...]] = {}
    for mode in modes:
        selected = [record for record in records if record["mode"] == mode]
        items = tuple(_restore_item(record["item"]) for record in selected)
        performance = tuple(
            _restore_performance(record["performance"]) for record in selected
        )
        measured_cache = PerformanceSummary.from_metrics(
            CacheState.UNCONTROLLED, performance
        )
        provenance = replace(base_provenance, graft=mode)
        run = BenchmarkRun(provenance, items, performance)
        run_metrics = run.summarize()
        outcome = run_metrics.outcomes
        attempted = outcome.correct + outcome.incorrect
        protocols = sorted(
            {str(item.metadata.get("benchmark_protocol", "unknown")) for item in items}
        )
        protocol_summaries = {
            protocol: asdict(
                run_metrics.outcomes.from_items(
                    item
                    for item in items
                    if item.metadata.get("benchmark_protocol") == protocol
                )
            )
            for protocol in protocols
        }
        has_closed_set_gsm8k = (
            "gsm8k_closed_set_candidate_token_diagnostic" in protocols
        )
        summary_document: dict[str, Any] = {
            **asdict(outcome),
            "accuracy_total": outcome.accuracy,
            "accuracy_attempted": outcome.correct / attempted if attempted else 0.0,
            "accuracy_is_canonical": not has_closed_set_gsm8k,
        }
        if has_closed_set_gsm8k:
            summary_document["accuracy_total"] = None
            summary_document["accuracy_attempted"] = None
            summary_document["accuracy"] = None
            summary_document["scope"] = (
                "counts_only; closed-set GSM8K accuracy is diagnostic and appears "
                "only under protocol_summaries"
            )
        mode_items[mode] = items
        mode_performance[mode] = performance
        by_mode[mode] = {
            "run": run.to_dict(),
            "run_id": run.run_id,
            "summary": summary_document,
            "protocol_summaries": protocol_summaries,
            "cache_profile": asdict(run_metrics.cache),
            "cache_profile_complete": run_metrics.cache.complete,
            "measured_cache_state": asdict(measured_cache),
            "byte_comparison_across_modes_valid": False,
            "measurement_totals": {
                key: sum(int(record["measurements"].get(key, 0)) for record in selected)
                for key in (
                    "source_body_bytes",
                    "charged_overhead_bytes",
                    "requests",
                    "cache_bytes_reused",
                    "cache_bytes_written",
                    "logical_weight_bytes",
                    "materialized_float_bytes",
                    "materialized_scale_bytes",
                    "linear_calls",
                    "head_rows",
                )
            },
        }
    paired: dict[str, Any] = {}
    if "off" in mode_items:
        for mode in modes:
            if mode in {"off", "shuffle"}:
                continue
            paired[f"{mode}_vs_off"] = asdict(
                summarize_paired_ablation(
                    mode_items[mode],
                    mode_items["off"],
                    candidate_performance=mode_performance[mode],
                    baseline_performance=mode_performance["off"],
                    cache_state=CacheState.UNCONTROLLED,
                )
            )
        if "shuffle" in mode_items:
            off_by_item = {
                (item.task, item.item_id): item for item in mode_items["off"]
            }
            expanded_off = tuple(
                replace(off_by_item[(item.task, item.item_id)], seed=item.seed)
                for item in mode_items["shuffle"]
            )
            off_perf_by_item = {
                metric.item_id: metric for metric in mode_performance["off"]
            }
            expanded_off_performance = tuple(
                replace(off_perf_by_item[metric.item_id], seed=metric.seed)
                for metric in mode_performance["shuffle"]
            )
            paired["shuffle_mean_over_seeds_vs_off"] = asdict(
                summarize_paired_ablation(
                    mode_items["shuffle"],
                    expanded_off,
                    candidate_performance=mode_performance["shuffle"],
                    baseline_performance=expanded_off_performance,
                    cache_state=CacheState.UNCONTROLLED,
                )
            )
            for seed in _mode_seeds("shuffle", seeds):
                shuffle_seed_items = tuple(
                    item for item in mode_items["shuffle"] if item.seed == seed
                )
                off_for_seed = tuple(
                    replace(off_by_item[(item.task, item.item_id)], seed=seed)
                    for item in shuffle_seed_items
                )
                shuffle_seed_performance = tuple(
                    metric
                    for metric in mode_performance["shuffle"]
                    if metric.seed == seed
                )
                off_performance_for_seed = tuple(
                    replace(off_perf_by_item[metric.item_id], seed=seed)
                    for metric in shuffle_seed_performance
                )
                paired[f"shuffle_seed_{seed}_vs_off"] = asdict(
                    summarize_paired_ablation(
                        shuffle_seed_items,
                        off_for_seed,
                        candidate_performance=shuffle_seed_performance,
                        baseline_performance=off_performance_for_seed,
                        cache_state=CacheState.UNCONTROLLED,
                    )
                )
    if "crsa" in mode_items and "shuffle" in mode_items:
        crsa_by_item = {(item.task, item.item_id): item for item in mode_items["crsa"]}
        expanded_crsa = tuple(
            replace(crsa_by_item[(item.task, item.item_id)], seed=item.seed)
            for item in mode_items["shuffle"]
        )
        crsa_perf_by_item = {
            metric.item_id: metric for metric in mode_performance["crsa"]
        }
        expanded_crsa_performance = tuple(
            replace(crsa_perf_by_item[metric.item_id], seed=metric.seed)
            for metric in mode_performance["shuffle"]
        )
        paired["crsa_vs_shuffle_mean_over_seeds"] = asdict(
            summarize_paired_ablation(
                expanded_crsa,
                mode_items["shuffle"],
                candidate_performance=expanded_crsa_performance,
                baseline_performance=mode_performance["shuffle"],
                cache_state=CacheState.UNCONTROLLED,
            )
        )
    expected = len(selected_rows) * sum(len(_mode_seeds(mode, seeds)) for mode in modes)
    return {
        "schema": REPORT_SCHEMA,
        "contract_schema": SCHEMA_VERSION,
        "signature": header["signature"],
        "status": "complete" if len(records) == expected else "incomplete",
        "journal": str(journal.path),
        "expected_item_runs": expected,
        "completed_item_runs": len(records),
        "modes": by_mode,
        "paired": paired,
        "seed_protocol": {
            "deterministic_modes": [
                mode for mode in modes if mode in {"off", "crsa", "softmax"}
            ],
            "deterministic_seed": int(seeds[0]),
            "variable_mode": "shuffle" if "shuffle" in modes else None,
            "shuffle_seeds": list(_mode_seeds("shuffle", seeds))
            if "shuffle" in modes
            else [],
            "repeated_deterministic_seeds_are_not_executed_or_counted": True,
        },
        "cache_measurement_protocol": {
            "state": CacheState.UNCONTROLLED.value,
            "profile_complete": False,
            "reason": (
                "single-pass workload shares a bounded cache; item order and mode "
                "order confound source/cache bytes, so no cold/warm/hot label is claimed"
            ),
            "cross_mode_byte_comparison_valid": False,
        },
        "benchmark_protocols": header["benchmark_protocols"],
        "prompt_protocols": header["prompt_protocols"],
        "assistant_generation_prefixes": header["assistant_generation_prefixes"],
        "candidate_tokenizations": header["candidate_tokenizations"],
        "quantized_accumulation_policy": header["quantized_accumulation_policy"],
        "attention_qat_policy": header["attention_qat_policy"],
        "budgets": {
            "source_limit_bytes_per_process": int(source.budget.limit),
            "source_used_bytes_this_process": int(source.bytes_moved()),
            "cache_limit_bytes": int(args.cache_budget_mb * 1024**2),
            "cache_enabled": not args.no_cache,
        },
        "source_metrics_this_process": source.metrics(),
        "pager_metrics_this_process": pager.metrics(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded MMLU/GSM8K evidence through streamed DeepSeek-V4"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="read-only local JSON, JSONL, or Parquet dataset",
    )
    parser.add_argument("--task", required=True, choices=("mmlu", "gsm8k"))
    parser.add_argument("--source", default=OFFICIAL_SOURCE)
    parser.add_argument("--revision", default=OFFICIAL_REVISION)
    parser.add_argument("--config", default=None)
    parser.add_argument("--tokenizer-json", default=None)
    parser.add_argument("--thinking-mode", choices=("chat", "thinking"), default="chat")
    parser.add_argument(
        "--reasoning-effort", choices=("low", "high", "max"), default="low"
    )
    parser.add_argument(
        "--allow-closed-set-gsm8k",
        action="store_true",
        help="allow explicitly non-canonical candidate-answer GSM8K diagnostics",
    )
    parser.add_argument(
        "--modes", nargs="+", default=["off", "crsa", "softmax", "shuffle"]
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="shuffle seeds; deterministic modes execute only the first seed",
    )
    parser.add_argument("--limit", type=_positive_int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--journal", required=True, help="resumable append-only JSONL")
    parser.add_argument(
        "--output", required=True, help="atomically replaced JSON report"
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--source-budget-mb", type=_positive_float, default=16_384.0)
    parser.add_argument("--cache-budget-mb", type=_nonnegative_float, default=24_576.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--max-dataset-mb", type=_positive_float, default=64.0)
    parser.add_argument(
        "--max-prompt-tokens",
        type=_positive_int,
        default=256,
        help="hard prompt bound (256 covers the bundled canonical MMLU/GSM8K splits)",
    )
    parser.add_argument("--max-new-tokens", type=_positive_int, default=32)
    parser.add_argument("--head-block-rows", type=_positive_int, default=1024)
    parser.add_argument("--eos-token-ids", nargs="*", type=int, default=[])
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    parser.add_argument("--no-activation-quantization", action="store_true")
    parser.add_argument("--batched-prefill", action="store_true")
    parser.add_argument("--graft-alpha", type=_nonnegative_float, default=0.05)
    parser.add_argument("--graft-layer", type=int, default=None)
    parser.add_argument("--graft-seed", type=int, default=17)
    parser.add_argument(
        "--preflight", choices=("none", "sample", "exhaustive"), default="exhaustive"
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    task = BenchmarkTask(args.task)
    modes = _parse_modes(args.modes)
    seeds = _parse_seeds(args.seeds)
    rows = _read_dataset(
        Path(args.dataset), maximum_bytes=int(args.max_dataset_mb * 1024**2)
    )
    selected_rows = rows[: args.limit] if args.limit is not None else rows
    source, source_label = _build_source(args)
    config, config_meta = _load_config(args, source)
    tokenizer = _load_tokenizer(args, source, selected_rows, task)
    if args.graft_layer is not None and not 0 <= args.graft_layer < config.n_layers:
        raise RunnerError("--graft-layer is outside decoder depth")
    inventory = source.inventory()
    plans = _prepare_rows(selected_rows, task, tokenizer, config, args)
    benchmark_protocols = sorted({plan.benchmark_protocol for plan in plans.values()})
    prompt_encodings = sorted({plan.prompt_encoding for plan in plans.values()})
    prompt_protocols = sorted({plan.prompt_protocol for plan in plans.values()})
    candidate_tokenizations = sorted(
        {plan.candidate_tokenization for plan in plans.values()}
    )
    assistant_generation_prefixes = sorted(
        {
            plan.assistant_generation_prefix
            for plan in plans.values()
            if plan.assistant_generation_prefix is not None
        }
    )
    inventory_sha = _digest_or_canonical(
        source.metrics().get("inventory_source_fingerprint"), inventory
    )
    tokenizer_sha = (
        tokenizer.sha256
        if tokenizer is not None
        else hashlib.sha256(b"pretokenized-token-ids/v1").hexdigest()
    )
    dataset_provenance = DatasetProvenance.from_records(
        Path(args.dataset).name,
        "selected",
        selected_rows,
        revision=canonical_digest(rows),
    )
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance = BenchmarkProvenance(
        model_id=source_label,
        model_revision=args.revision,
        inventory_sha256=inventory_sha,
        config_sha256=config_meta["sha256"],
        tokenizer_sha256=tokenizer_sha,
        dataset=dataset_provenance,
        harness_revision=f"script-sha256:{script_sha}",
        command=tuple(sys.argv),
        protocol={
            "benchmark_protocols": benchmark_protocols,
            "prompt_protocols": prompt_protocols,
            "prompt_encodings": prompt_encodings,
            "assistant_generation_prefixes": assistant_generation_prefixes,
            "candidate_tokenizations": candidate_tokenizations,
            "candidate_suffixes_are_scored_after_recorded_prompt": True,
            "activation_quantization": not args.no_activation_quantization,
            "quantized_accumulation_policy": (
                DeepSeekWeightPager.QUANTIZED_ACCUMULATION_POLICY
            ),
            "attention_qat_policy": StreamedDeepSeekV4.ATTENTION_QAT_POLICY,
            "tokenizer_location": (
                "pretokenized-token-ids/v1" if tokenizer is None else tokenizer.location
            ),
        },
    )
    signature_payload = {
        "harness_sha256": script_sha,
        "source": source_label,
        "revision": args.revision,
        "config_sha256": config_meta["sha256"],
        "inventory_sha256": inventory_sha,
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_location": (
            "pretokenized-token-ids/v1" if tokenizer is None else tokenizer.location
        ),
        "dataset": asdict(dataset_provenance),
        "task": task.value,
        "benchmark_protocols": benchmark_protocols,
        "prompt_encodings": prompt_encodings,
        "prompt_protocols": prompt_protocols,
        "assistant_generation_prefixes": assistant_generation_prefixes,
        "candidate_tokenizations": candidate_tokenizations,
        "modes": modes,
        "seeds": seeds,
        "seed_protocol": {mode: list(_mode_seeds(mode, seeds)) for mode in modes},
        "graft_alpha": args.graft_alpha,
        "graft_layer": args.graft_layer,
        "graft_seed": args.graft_seed,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "batched_prefill": args.batched_prefill,
        "activation_quantization": not args.no_activation_quantization,
        "quantized_accumulation_policy": (
            DeepSeekWeightPager.QUANTIZED_ACCUMULATION_POLICY
        ),
        "attention_qat_policy": StreamedDeepSeekV4.ATTENTION_QAT_POLICY,
        "device": args.device,
        "dtype": args.dtype,
        "head_block_rows": args.head_block_rows,
        "eos_token_ids": tuple(args.eos_token_ids),
        "source_budget_mb": args.source_budget_mb,
        "cache_budget_mb": args.cache_budget_mb,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()),
        "cache_enabled": not args.no_cache,
        "preflight": args.preflight,
        "thinking_mode": args.thinking_mode,
        "reasoning_effort": args.reasoning_effort,
        "allow_closed_set_gsm8k": args.allow_closed_set_gsm8k,
    }
    signature = canonical_digest(signature_payload)
    pager = DeepSeekWeightPager(
        source,
        device=args.device,
        compute_dtype=args.dtype,
        simulate_activation_quantization=not args.no_activation_quantization,
    )
    if args.preflight != "none":
        model = _model_for_mode(config, pager, args, "off", seeds[0])
        model.checkpoint_preflight(exhaustive_experts=args.preflight == "exhaustive")
        del model
    header = {
        "schema": JOURNAL_SCHEMA,
        "type": "header",
        "signature": signature,
        "provenance": _provenance_dict(provenance),
        "task": task.value,
        "modes": list(modes),
        "seeds": list(seeds),
        "mode_seeds": {mode: list(_mode_seeds(mode, seeds)) for mode in modes},
        "benchmark_protocols": benchmark_protocols,
        "prompt_encodings": prompt_encodings,
        "prompt_protocols": prompt_protocols,
        "assistant_generation_prefixes": assistant_generation_prefixes,
        "candidate_tokenizations": candidate_tokenizations,
        "quantized_accumulation_policy": (
            DeepSeekWeightPager.QUANTIZED_ACCUMULATION_POLICY
        ),
        "attention_qat_policy": StreamedDeepSeekV4.ATTENTION_QAT_POLICY,
        "closed_set_gsm8k_is_noncanonical": (
            "gsm8k_closed_set_candidate_token_diagnostic" in benchmark_protocols
        ),
        "selected_item_ids": [str(row["id"]) for row in selected_rows],
        "budgets": {
            "source_bytes": int(args.source_budget_mb * 1024**2),
            "cache_bytes": int(args.cache_budget_mb * 1024**2),
        },
    }
    journal = JsonlJournal(Path(args.journal), header=header, resume=args.resume)

    new_records = 0
    for mode in modes:
        for seed in _mode_seeds(mode, seeds):
            pending = [
                row
                for row in selected_rows
                if (mode, str(row["id"]), seed) not in journal.completed
            ]
            if not pending:
                continue
            model = _model_for_mode(config, pager, args, mode, seed)
            try:
                for row in pending:
                    record = _item_record(
                        mode=mode,
                        seed=seed,
                        row=row,
                        plan=plans[str(row["id"])],
                        task=task,
                        model=model,
                        pager=pager,
                        source=source,
                        tokenizer=tokenizer,
                        args=args,
                        signature=signature,
                    )
                    journal.append_item(record)
                    new_records += 1
            finally:
                model.reset_state(release=True)
                del model
                pager.release()
    report = _build_report(
        journal,
        source=source,
        pager=pager,
        args=args,
        modes=modes,
        seeds=seeds,
        selected_rows=selected_rows,
    )
    _atomic_write_json(Path(args.output), report)
    return {
        "status": report["status"],
        "signature": signature,
        "new_item_runs": new_records,
        "completed_item_runs": report["completed_item_runs"],
        "expected_item_runs": report["expected_item_runs"],
        "journal": str(Path(args.journal).expanduser().resolve()),
        "report": str(Path(args.output).expanduser().resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        receipt = run(args)
    except Exception as exc:
        error = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        sys.stderr.write(json.dumps(error, sort_keys=True) + "\n")
        return 2
    sys.stdout.write(json.dumps(receipt, sort_keys=True) + "\n")
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
