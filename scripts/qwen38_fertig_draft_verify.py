#!/usr/bin/env python3
"""Verify a historical FERTIG-abstention cohort with one Qwen3.8 pass.

The committed local Q3 baseline supplies eight short answer candidates.  The
official pinned BF16 checkpoint then teacher-forces all candidates together,
streaming each decoder matrix once in layer-major order.  One rolling hidden
state makes the long pass resumable without retaining model weights.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request

from immer.knowledge import AccessTrace, AccessTraceRecorder, Streamer
from immer.runtimes.deepseek_v4.benchmark import extract_gsm8k_answer
from immer.runtimes.qwen3_8 import (
    OFFICIAL_REPO_ID,
    OFFICIAL_REVISION,
    CausalWeightMount,
    LogicalModelIdentity,
    Qwen38BundleError,
    Qwen38Config,
    Qwen38DraftVerifier,
    Qwen38StableCrsaGraft,
    Qwen38Tokenizer,
    Qwen38WeightPager,
    StreamedQwen38,
    verify_qwen38_causal_mount,
)
from immer.runtimes.qwen3_8.encoding import (
    END_OF_TEXT_TOKEN_ID,
    IM_END_TOKEN_ID,
)
from immer.runtimes.qwen3_8.draft_verification import DraftVerificationResumeState
from immer.runtimes.qwen3_8.resume import (
    build_resume_identity,
    delete_resume,
    load_resume,
    preflight_resume_disk,
    write_resume,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK = ROOT / "results" / "bench_gsm8k.json"
DEFAULT_DRAFTS = ROOT / "results" / "qwen38_reference_baseline.json"
DEFAULT_RUN_DIR = ROOT / "artifacts" / "private" / "qwen3.8-fertig-draft-verify"
DEFAULT_CACHE = ROOT / "artifacts" / "private" / "qwen3.8-cache"
DEFAULT_TOKENIZER = (
    ROOT / "artifacts" / "private" / "qwen3.8-reference" / "tokenizer.json"
)
TOKENIZER_CEILING = 32 * 1024**2
OFFICIAL_TOKENIZER_SHA256 = (
    "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
)
RESULT_SCHEMA = "immer.qwen3.8-fertig-draft-verification/v1"
TRACE_CHECKPOINT_SCHEMA = "immer.qwen3.8-trace-resume-pair/v1"
INPUT_SCHEMA = "immer.qwen3.8-fertig-draft-inputs/v1"
BASELINE_SCHEMA = "immer.qwen-local-fertig-baseline/v1"
BASELINE_MODEL_REVISION = "Qwen3.8-27B-Q3_K_M-no-think"
BASELINE_SYSTEM_PROMPT = (
    "Solve the math problem internally. Return only #### followed by the numeric "
    "answer. Do not show work."
)
FIXED_ITEM_IDS = (
    "gsm8k-test-0737-b673ac26d1268186",
    "gsm8k-test-0815-13fae6ff992c2157",
    "gsm8k-test-0857-fbb13bf06cc00a16",
    "gsm8k-test-0901-5fbbe0d05836802d",
    "gsm8k-test-1032-e5873b4dd50866b6",
    "gsm8k-test-1042-d9ceeb071728fb0e",
    "gsm8k-test-1175-3598411cca7939c2",
    "gsm8k-test-1240-a316335b6f4ebc29",
)


class CliError(RuntimeError):
    """The bounded Qwen verification contract cannot be satisfied."""


@dataclass(frozen=True, slots=True)
class PreparedDraft:
    item_id: str
    question: str
    gold: str
    text: str
    answer: str
    candidate_correct: bool
    prompt_token_ids: tuple[int, ...]
    draft_token_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "question": self.question,
            "gold": self.gold,
            "text": self.text,
            "answer": self.answer,
            "candidate_correct": self.candidate_correct,
            "prompt_token_ids": list(self.prompt_token_ids),
            "draft_token_ids": list(self.draft_token_ids),
        }


@dataclass(frozen=True, slots=True)
class TraceResumeCheckpoint:
    manifest: Path
    resume_path: Path
    trace_path: Path
    trace: AccessTrace
    next_layer: int
    source_body_bytes: int


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def _positive_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def _nonnegative_float(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def _unit_float(raw: str) -> float:
    value = _nonnegative_float(raw)
    if value > 1:
        raise argparse.ArgumentTypeError("must be at most 1")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--drafts-json", default=str(DEFAULT_DRAFTS))
    parser.add_argument(
        "--dynamic-cohort",
        action="store_true",
        help="use the explicitly ordered cohort sealed by the baseline report",
    )
    parser.add_argument("--tokenizer-json")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--result-json")
    parser.add_argument("--access-trace-json")
    parser.add_argument("--resume-file")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--causal-bundle")
    parser.add_argument("--cache-budget-gb", type=_nonnegative_float, default=1.0)
    parser.add_argument("--source-budget-mb", type=_positive_int, default=65536)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="mps")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--max-draft-tokens", type=_positive_int, default=32)
    parser.add_argument("--head-block-rows", type=_positive_int, default=8192)
    parser.add_argument("--layer-retries", type=_nonnegative_int, default=2)
    parser.add_argument("--head-retries", type=_nonnegative_int, default=2)
    parser.add_argument("--mode", choices=("off", "stable-crsa"), default="off")
    parser.add_argument("--graft-layer", type=_nonnegative_int, default=27)
    parser.add_argument("--graft-alpha", type=_unit_float, default=0.01)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="validate and tokenize the fixed drafts without opening model weights",
    )
    parser.add_argument(
        "--offline-tokenizer",
        action="store_true",
        help="fail instead of fetching the pinned tokenizer when it is not cached",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="explicitly delete this run's rolling hidden-state resume first",
    )
    return parser


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CliError(f"cannot read {label}: {source}") from exc
    if not isinstance(document, dict):
        raise CliError(f"{label} root must be an object")
    return document


def _atomic_write_bytes(path: Path, body: bytes) -> Path:
    destination = path.expanduser().resolve()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
    except OSError as exc:
        raise CliError(f"cannot prepare output: {destination}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        raise CliError(f"cannot write output: {destination}") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _atomic_write_json(path: str | Path, document: Mapping[str, Any]) -> Path:
    try:
        body = (
            json.dumps(
                document,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CliError("result contains non-serializable data") from exc
    return _atomic_write_bytes(Path(path), body)


def _access_trace_path(args: argparse.Namespace) -> Path | None:
    raw = getattr(args, "access_trace_json", None)
    return None if raw is None else Path(raw).expanduser().resolve()


def _discard_access_trace(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise CliError(f"cannot inspect access trace: {path}") from exc
    if path.is_symlink() or not path.is_file() or metadata.st_nlink < 1:
        raise CliError("access trace must be a non-symlink regular file")
    try:
        path.unlink()
    except OSError as exc:
        raise CliError(f"cannot discard access trace: {path}") from exc
    return True


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CliError("trace checkpoint metadata is not canonical JSON") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _trace_checkpoint_manifest(trace_path: Path) -> Path:
    return trace_path.with_name(f".{trace_path.name}.checkpoint.json")


def _checkpoint_content_path(
    trace_path: Path, kind: str, digest: str, suffix: str
) -> Path:
    return trace_path.with_name(f".{trace_path.stem}.{kind}.{digest}{suffix}")


def _publish_content(path: Path, value: bytes, digest: str) -> Path:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != digest:
            raise CliError(f"trace checkpoint content collision: {path}")
        return path
    written = _atomic_write_bytes(path, value)
    if _sha256_file(written) != digest:
        raise CliError("published trace checkpoint content SHA-256 mismatch")
    return written


def _strict_checkpoint_document(path: Path) -> dict[str, Any]:
    document = _read_json(path, "trace checkpoint manifest")
    if set(document) != {"body", "schema", "sha256"}:
        raise CliError("trace checkpoint manifest schema is invalid")
    if document.get("schema") != TRACE_CHECKPOINT_SCHEMA:
        raise CliError("trace checkpoint manifest type is invalid")
    body = document.get("body")
    if (
        not isinstance(body, dict)
        or document.get("sha256") != hashlib.sha256(_canonical_json(body)).hexdigest()
    ):
        raise CliError("trace checkpoint manifest SHA-256 mismatch")
    return document


def _checkpoint_member(parent: Path, raw: object, suffix: str) -> Path:
    if (
        not isinstance(raw, str)
        or not raw
        or Path(raw).name != raw
        or not raw.endswith(suffix)
    ):
        raise CliError("trace checkpoint member path is unsafe")
    path = parent / raw
    if path.is_symlink() or not path.is_file():
        raise CliError("trace checkpoint member is missing or non-regular")
    return path


def _load_trace_checkpoint(
    trace_path: Path,
    identity: str,
) -> TraceResumeCheckpoint | None:
    manifest = _trace_checkpoint_manifest(trace_path)
    if not manifest.exists() and not manifest.is_symlink():
        return None
    if manifest.is_symlink() or not manifest.is_file():
        raise CliError("trace checkpoint manifest must be a regular file")
    document = _strict_checkpoint_document(manifest)
    body = document["body"]
    if body.get("schema") != TRACE_CHECKPOINT_SCHEMA:
        raise CliError("trace checkpoint authenticated schema is invalid")
    if body.get("resume_identity") != identity:
        raise CliError("trace checkpoint belongs to another run identity")
    next_layer = body.get("next_layer")
    source_body_bytes = body.get("source_body_bytes")
    if (
        isinstance(next_layer, bool)
        or not isinstance(next_layer, int)
        or next_layer < 1
        or isinstance(source_body_bytes, bool)
        or not isinstance(source_body_bytes, int)
        or source_body_bytes < 0
    ):
        raise CliError("trace checkpoint counters are invalid")
    resume = body.get("resume")
    trace_record = body.get("trace")
    if not isinstance(resume, Mapping) or set(resume) != {"file", "sha256"}:
        raise CliError("trace checkpoint resume descriptor is invalid")
    if not isinstance(trace_record, Mapping) or set(trace_record) != {
        "file",
        "file_sha256",
        "trace_sha256",
    }:
        raise CliError("trace checkpoint trace descriptor is invalid")
    resume_path = _checkpoint_member(
        manifest.parent, resume.get("file"), ".safetensors"
    )
    content_trace = _checkpoint_member(
        manifest.parent, trace_record.get("file"), ".json"
    )
    if _sha256_file(resume_path) != resume.get("sha256"):
        raise CliError("trace checkpoint resume SHA-256 mismatch")
    encoded_trace = content_trace.read_bytes()
    if hashlib.sha256(encoded_trace).hexdigest() != trace_record.get("file_sha256"):
        raise CliError("trace checkpoint trace SHA-256 mismatch")
    trace = AccessTrace.from_bytes(encoded_trace)
    if trace.sha256 != trace_record.get("trace_sha256"):
        raise CliError("trace checkpoint canonical trace identity mismatch")
    return TraceResumeCheckpoint(
        manifest=manifest,
        resume_path=resume_path,
        trace_path=content_trace,
        trace=trace,
        next_layer=next_layer,
        source_body_bytes=source_body_bytes,
    )


def _write_trace_checkpoint(
    recorder: AccessTraceRecorder,
    trace_path: Path,
    identity: str,
    state: DraftVerificationResumeState,
    *,
    expected_shape: Sequence[int],
    expected_dtype: str,
    n_layers: int,
    active_graft_layer: int | None,
) -> dict[str, Any]:
    prior = _load_trace_checkpoint(trace_path, identity)
    metrics = recorder.metrics()
    if metrics["dropped_capacity"] or metrics["dropped_identity"]:
        raise CliError("access trace recorder dropped operations")
    trace = recorder.snapshot()
    trace.verify()
    trace_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{trace_path.stem}.resume.",
        suffix=".pending",
        dir=trace_path.parent,
    )
    os.close(descriptor)
    os.unlink(temporary)
    temporary_path = Path(temporary)
    try:
        write_resume(
            temporary_path,
            identity,
            state,
            expected_shape=expected_shape,
            expected_dtype=expected_dtype,
            n_layers=n_layers,
            active_graft_layer=active_graft_layer,
        )
        resume_bytes = temporary_path.read_bytes()
        resume_sha = hashlib.sha256(resume_bytes).hexdigest()
        resume_path = _checkpoint_content_path(
            trace_path, "resume", resume_sha, ".safetensors"
        )
        _publish_content(resume_path, resume_bytes, resume_sha)
    finally:
        temporary_path.unlink(missing_ok=True)

    trace_encoded = trace.to_bytes()
    trace_file_sha = hashlib.sha256(trace_encoded).hexdigest()
    trace_path_content = _checkpoint_content_path(
        trace_path, "trace", trace_file_sha, ".json"
    )
    _publish_content(trace_path_content, trace_encoded, trace_file_sha)
    _fsync_directory(trace_path.parent)
    body = {
        "next_layer": int(state.next_layer),
        "resume": {"file": resume_path.name, "sha256": resume_sha},
        "resume_identity": identity,
        "schema": TRACE_CHECKPOINT_SCHEMA,
        "source_body_bytes": int(state.source_body_bytes),
        "trace": {
            "file": trace_path_content.name,
            "file_sha256": trace_file_sha,
            "trace_sha256": trace.sha256,
        },
    }
    document = {
        "body": body,
        "schema": TRACE_CHECKPOINT_SCHEMA,
        "sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
    }
    manifest = _trace_checkpoint_manifest(trace_path)
    _atomic_write_bytes(manifest, _canonical_json(document) + b"\n")
    _fsync_directory(trace_path.parent)
    _atomic_write_bytes(trace_path, trace_encoded)
    _fsync_directory(trace_path.parent)
    if prior is not None:
        for obsolete in (prior.resume_path, prior.trace_path):
            if obsolete not in {resume_path, trace_path_content}:
                try:
                    obsolete.unlink()
                except OSError:
                    pass
    return {
        "inventory_fingerprint": trace.inventory_fingerprint,
        "leaves": metrics["leaves"],
        "next_layer": int(state.next_layer),
        "operations": metrics["operations"],
        "path": str(trace_path),
        "resume_sha256": resume_sha,
        "sha256": trace.sha256,
        "source_body_bytes": int(state.source_body_bytes),
    }


def _discard_trace_checkpoint(trace_path: Path, *, keep_public: bool = False) -> bool:
    manifest = _trace_checkpoint_manifest(trace_path)
    if not manifest.exists() and not manifest.is_symlink():
        return False if keep_public else _discard_access_trace(trace_path)
    if manifest.is_symlink() or not manifest.is_file():
        raise CliError("trace checkpoint manifest must be a regular file")
    document = _strict_checkpoint_document(manifest)
    body = document["body"]
    members: list[Path] = []
    for key, suffix in (("resume", ".safetensors"), ("trace", ".json")):
        descriptor = body.get(key)
        if not isinstance(descriptor, Mapping):
            raise CliError("trace checkpoint descriptor is invalid")
        members.append(
            _checkpoint_member(manifest.parent, descriptor.get("file"), suffix)
        )
    if not keep_public:
        _discard_access_trace(trace_path)
    for member in members:
        member.unlink()
    manifest.unlink()
    return True


def _progress(event: str, **fields: Any) -> None:
    sys.stderr.write(json.dumps({"event": event, **fields}, sort_keys=True) + "\n")
    sys.stderr.flush()


def _fetch_tokenizer(destination: Path) -> Path:
    url = (
        f"https://huggingface.co/{OFFICIAL_REPO_ID}/resolve/"
        f"{OFFICIAL_REVISION}/tokenizer.json"
    )
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            raw_length = response.headers.get("Content-Length")
            if raw_length is not None and int(raw_length) > TOKENIZER_CEILING:
                raise CliError("pinned tokenizer exceeds its 32 MiB bound")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(min(1024**2, TOKENIZER_CEILING + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > TOKENIZER_CEILING:
                    raise CliError("pinned tokenizer exceeds its 32 MiB bound")
    except CliError:
        raise
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
        raise CliError("cannot fetch the pinned Qwen3.8 tokenizer") from exc
    body = b"".join(chunks)
    if hashlib.sha256(body).hexdigest() != OFFICIAL_TOKENIZER_SHA256:
        raise CliError("pinned tokenizer SHA-256 mismatch")
    return _atomic_write_bytes(destination, body)


def _verify_tokenizer_digest(path: Path) -> Path:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024**2), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CliError(f"cannot read tokenizer JSON: {path}") from exc
    if digest.hexdigest() != OFFICIAL_TOKENIZER_SHA256:
        raise CliError("tokenizer JSON does not match the pinned Qwen3.8 revision")
    return path


def _resolve_tokenizer_path(args: argparse.Namespace) -> Path:
    if args.tokenizer_json:
        explicit = Path(args.tokenizer_json).expanduser().resolve()
        if not explicit.is_file():
            raise CliError(f"tokenizer JSON is not a file: {explicit}")
        return _verify_tokenizer_digest(explicit)
    if DEFAULT_TOKENIZER.is_file():
        return _verify_tokenizer_digest(DEFAULT_TOKENIZER.resolve())
    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(
            repo_id=OFFICIAL_REPO_ID,
            filename="tokenizer.json",
            revision=OFFICIAL_REVISION,
            local_files_only=True,
        )
    except Exception:
        cached = None
    if cached is not None and Path(cached).is_file():
        return _verify_tokenizer_digest(Path(cached).resolve())
    if args.offline_tokenizer:
        raise CliError("pinned Qwen3.8 tokenizer is not cached locally")
    return _fetch_tokenizer(DEFAULT_TOKENIZER)


def _benchmark_rows(path: str | Path) -> dict[str, Mapping[str, Any]]:
    document = _read_json(path, "FERTIG GSM8K report")
    rows = document.get("items")
    if not isinstance(rows, list):
        raise CliError("FERTIG GSM8K report has no item list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            continue
        item_id = raw.get("item_id")
        if isinstance(item_id, str):
            if item_id in indexed:
                raise CliError(f"duplicate benchmark item: {item_id}")
            indexed[item_id] = raw
    return indexed


def _cohort_item_ids(
    drafts_json: str | Path,
    *,
    dynamic: bool,
) -> tuple[str, ...]:
    if not dynamic:
        return FIXED_ITEM_IDS
    document = _read_json(drafts_json, "Qwen local baseline")
    if document.get("schema") != BASELINE_SCHEMA:
        raise CliError("Qwen baseline schema mismatch")
    claimed = document.get("report_sha256")
    unsealed = dict(document)
    unsealed.pop("report_sha256", None)
    try:
        actual = hashlib.sha256(
            json.dumps(
                unsealed,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise CliError("dynamic baseline report is not canonical JSON") from exc
    if not isinstance(claimed, str) or claimed != actual:
        raise CliError("dynamic baseline report seal is invalid")
    benchmark = document.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise CliError("dynamic baseline benchmark identity is missing")
    raw_ids = benchmark.get("item_ids")
    selection = benchmark.get("selection")
    if (
        not isinstance(raw_ids, list)
        or not 1 <= len(raw_ids) <= 64
        or not isinstance(selection, Mapping)
        or selection.get("cohort") not in {"abstained", "eligible"}
        or selection.get("limit") != len(raw_ids)
    ):
        raise CliError("dynamic baseline cohort contract is invalid")
    item_ids = tuple(raw_ids)
    if any(not isinstance(item_id, str) or not item_id for item_id in item_ids):
        raise CliError("dynamic baseline item ID is invalid")
    if len(set(item_ids)) != len(item_ids):
        raise CliError("dynamic baseline item IDs are duplicated")
    rows = document.get("items")
    if not isinstance(rows, list) or [
        row.get("item_id") if isinstance(row, Mapping) else None for row in rows
    ] != list(item_ids):
        raise CliError("dynamic baseline rows do not match their cohort order")
    return item_ids


def _prepare_drafts(
    benchmark: str | Path,
    drafts_json: str | Path,
    tokenizer: Qwen38Tokenizer,
    *,
    max_draft_tokens: int,
    item_ids: Sequence[str] = FIXED_ITEM_IDS,
) -> tuple[PreparedDraft, ...]:
    benchmark_rows = _benchmark_rows(benchmark)
    document = _read_json(drafts_json, "Qwen local baseline")
    if document.get("schema") != BASELINE_SCHEMA:
        raise CliError("Qwen baseline schema mismatch")
    model = document.get("model")
    if (
        not isinstance(model, Mapping)
        or model.get("revision") != BASELINE_MODEL_REVISION
        or model.get("backend") != "openai-compatible"
    ):
        raise CliError("Qwen baseline model identity mismatch")
    protocol = document.get("protocol")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("system_prompt") != BASELINE_SYSTEM_PROMPT
        or protocol.get("temperature") != 0.0
        or protocol.get("seed") != 0
        or protocol.get("max_tokens") != 32
    ):
        raise CliError("Qwen baseline prompt contract mismatch")
    summary = document.get("summary")
    if not isinstance(summary, Mapping) or any(
        summary.get(name) != 0 for name in ("truncated", "unparseable", "error")
    ):
        raise CliError("Qwen baseline contains incomplete candidates")
    raw_rows = document.get("items")
    if not isinstance(raw_rows, list) or len(raw_rows) != len(item_ids):
        raise CliError("Qwen baseline item count mismatch")

    prepared: list[PreparedDraft] = []
    for index, (item_id, raw) in enumerate(zip(item_ids, raw_rows, strict=True)):
        if not isinstance(raw, Mapping) or raw.get("item_id") != item_id:
            raise CliError(f"Qwen baseline row {index} is out of order")
        benchmark_row = benchmark_rows.get(item_id)
        if benchmark_row is None or benchmark_row.get("status") not in {
            "abstained",
            "correct",
        }:
            raise CliError(f"fixed historical cohort item is unavailable: {item_id}")
        question = benchmark_row.get("question")
        if not isinstance(question, str) or raw.get("question") != question:
            raise CliError(f"stale baseline question: {item_id}")
        gold = extract_gsm8k_answer(benchmark_row.get("gold"))
        if gold is None or raw.get("gold") != gold:
            raise CliError(f"stale baseline gold answer: {item_id}")
        text = raw.get("text")
        if not isinstance(text, str) or not text:
            raise CliError(f"baseline candidate has no text: {item_id}")
        answer = extract_gsm8k_answer(text)
        if answer is None or raw.get("predicted") != answer:
            raise CliError(f"baseline candidate answer mismatch: {item_id}")
        correct = answer == gold
        expected_status = "correct" if correct else "incorrect"
        if (
            raw.get("status") != expected_status
            or raw.get("correct") is not correct
            or raw.get("finish_reason") != "stop"
        ):
            raise CliError(f"baseline candidate status mismatch: {item_id}")

        prompt_text = tokenizer.render_no_thinking_prompt(
            BASELINE_SYSTEM_PROMPT,
            question,
        )
        prompt_ids = tokenizer.encode(prompt_text)
        combined_ids = tokenizer.encode(prompt_text + text)
        if combined_ids[: len(prompt_ids)] != prompt_ids:
            raise CliError(f"candidate changes the tokenized prompt prefix: {item_id}")
        draft_ids = combined_ids[len(prompt_ids) :]
        if not draft_ids or len(draft_ids) > max_draft_tokens:
            raise CliError(f"baseline candidate violates the draft bound: {item_id}")
        if IM_END_TOKEN_ID in draft_ids or END_OF_TEXT_TOKEN_ID in draft_ids:
            raise CliError(f"baseline candidate contains a stop token: {item_id}")
        if raw.get("prompt_tokens") != len(prompt_ids):
            raise CliError(f"baseline/server prompt token mismatch: {item_id}")
        if raw.get("completion_tokens") != len(draft_ids):
            raise CliError(f"baseline/server completion token mismatch: {item_id}")
        if tokenizer.decode(draft_ids) != text:
            raise CliError(f"baseline candidate tokenizer roundtrip failed: {item_id}")
        prepared.append(
            PreparedDraft(
                item_id=item_id,
                question=question,
                gold=gold,
                text=text,
                answer=answer,
                candidate_correct=correct,
                prompt_token_ids=prompt_ids,
                draft_token_ids=draft_ids,
            )
        )
    correct_count = sum(row.candidate_correct for row in prepared)
    if (
        summary.get("total") != len(prepared)
        or summary.get("correct") != correct_count
        or summary.get("incorrect") != len(prepared) - correct_count
        or summary.get("accuracy") != correct_count / len(prepared)
    ):
        raise CliError("Qwen baseline summary does not match its candidate rows")
    return tuple(prepared)


def _input_document(rows: Sequence[PreparedDraft]) -> dict[str, Any]:
    lengths = [len(row.prompt_token_ids) + len(row.draft_token_ids) for row in rows]
    candidate_correct = sum(row.candidate_correct for row in rows)
    return {
        "schema": INPUT_SCHEMA,
        "source": {
            "checkpoint": OFFICIAL_REPO_ID,
            "revision": OFFICIAL_REVISION,
            "drafts": "Qwen3.8-27B-Q3_K_M local no-thinking baseline",
        },
        "protocol": {
            "system_prompt": BASELINE_SYSTEM_PROMPT,
            "thinking": False,
            "batch_size": len(rows),
            "padding": "right",
            "padding_token_id": END_OF_TEXT_TOKEN_ID,
            "eos_token_id": IM_END_TOKEN_ID,
            "accepted_eos_token_ids": [IM_END_TOKEN_ID, END_OF_TEXT_TOKEN_ID],
            "weight_order": "layer-major, one BF16 matrix at a time",
            "dynamic_cohort": tuple(row.item_id for row in rows) != FIXED_ITEM_IDS,
            "item_ids": [row.item_id for row in rows],
        },
        "summary": {
            "items": len(rows),
            "candidate_correct": candidate_correct,
            "candidate_accuracy": candidate_correct / len(rows),
            "max_combined_tokens": max(lengths),
            "verification_rows": sum(len(row.draft_token_ids) + 1 for row in rows),
        },
        "items": [row.to_dict() for row in rows],
    }


def _resume_path(args: argparse.Namespace) -> Path:
    if args.resume_file:
        return Path(args.resume_file).expanduser().resolve()
    return Path(args.run_dir).expanduser().resolve() / f"resume-{args.mode}.safetensors"


@contextmanager
def _model_runtime(
    args: argparse.Namespace,
    *,
    max_batch_size: int,
    max_seq_len: int,
) -> Iterator[StreamedQwen38]:
    mount: CausalWeightMount | None = None
    if args.causal_bundle is None:
        source = Streamer(
            OFFICIAL_REPO_ID,
            revision=OFFICIAL_REVISION,
            budget_mb=args.source_budget_mb,
            cache_dir=Path(args.cache_dir).expanduser().resolve(),
            max_cache_bytes=int(args.cache_budget_gb * 1024**3),
            verbose=False,
        )
        source_verification: dict[str, Any] = {"kind": "remote-pinned-range-source/v1"}
    else:
        mount = CausalWeightMount(
            Path(args.causal_bundle).expanduser().resolve(),
            LogicalModelIdentity(OFFICIAL_REPO_ID, OFFICIAL_REVISION),
            budget_mb=args.source_budget_mb,
        )
        source = mount.source
        verification_started = time.perf_counter()
        try:
            source_verification = verify_qwen38_causal_mount(
                mount,
                require_official_config=getattr(args, "_require_official", True),
            )
        except Qwen38BundleError as exc:
            mount.close()
            raise CliError(f"causal bundle verification failed: {exc}") from exc
        except Exception:
            mount.close()
            raise
        source_verification = {
            **source_verification,
            "seconds": time.perf_counter() - verification_started,
        }
    setattr(args, "_source_verification", source_verification)
    pager: Qwen38WeightPager | None = None
    model: StreamedQwen38 | None = None
    try:
        try:
            raw_config = source.reader.fetch_file("config.json")
            document = json.loads(raw_config)
        except Exception as exc:
            raise CliError("cannot load pinned Qwen3.8 config") from exc
        if not isinstance(document, Mapping):
            raise CliError("pinned Qwen3.8 config root is not an object")
        config = Qwen38Config.from_mapping(document)
        pager = Qwen38WeightPager(
            source,
            device=args.device,
            compute_dtype=args.dtype,
            require_source_identity=True,
            causal_tensor_reader=None if mount is None else mount.tensor_reader,
        )
        graft = None
        graft_layer = None
        if args.mode == "stable-crsa":
            graft = Qwen38StableCrsaGraft(
                mode="crsa",
                alpha=args.graft_alpha,
                max_history=max_seq_len,
            )
            graft_layer = args.graft_layer
        model = StreamedQwen38(
            config,
            pager,
            graft=graft,
            graft_layer=graft_layer,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
        model.checkpoint_preflight()
        yield model
    finally:
        active_error = sys.exc_info()[1]
        cleanup_error: Exception | None = None
        if model is not None:
            try:
                model.reset_state(release=True)
            except Exception as exc:
                cleanup_error = exc
        if pager is not None:
            try:
                pager.close()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        try:
            if mount is None:
                source.close()
            else:
                mount.close()
        except Exception as exc:
            if cleanup_error is None:
                cleanup_error = exc
        if active_error is None and cleanup_error is not None:
            raise cleanup_error


def _cache_disk_preflight(args: argparse.Namespace) -> dict[str, int]:
    if args.causal_bundle is not None:
        bundle = Path(args.causal_bundle).expanduser().resolve()
        free = shutil.disk_usage(bundle).free
        return {"cache_bytes": 0, "cache_growth_bytes": 0, "free_bytes": free}
    cache = Path(args.cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    current = sum(
        path.stat().st_size
        for path in cache.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    limit = int(args.cache_budget_gb * 1024**3)
    growth = max(0, limit - current)
    free = shutil.disk_usage(cache).free
    if free < growth:
        raise CliError(
            f"insufficient free disk for bounded Qwen cache: need {growth}, have {free}"
        )
    return {"cache_bytes": current, "cache_growth_bytes": growth, "free_bytes": free}


def _apply_cumulative_source_budget(
    args: argparse.Namespace,
    model: StreamedQwen38,
    *,
    prior_source_bytes: int,
) -> dict[str, int]:
    total_limit = int(args.source_budget_mb * 1024**2)
    if prior_source_bytes < 0 or prior_source_bytes > total_limit:
        raise CliError("rolling resume has exhausted the cumulative source budget")
    budget = model.pager.source.budget
    remaining_limit = total_limit - prior_source_bytes
    already_used = int(budget.total)
    if already_used > remaining_limit:
        raise CliError("current preflight exhausted the cumulative source budget")
    budget.limit = remaining_limit
    return {
        "total_limit_bytes": total_limit,
        "prior_source_bytes": prior_source_bytes,
        "current_process_bytes": already_used,
        "remaining_process_limit_bytes": remaining_limit,
    }


def _verify(
    args: argparse.Namespace,
    rows: Sequence[PreparedDraft],
    *,
    runtime_factory: Callable[..., Any] | None = None,
    verifier_factory: Callable[..., Any] | None = None,
) -> Any:
    prompts = tuple(row.prompt_token_ids for row in rows)
    drafts = tuple(row.draft_token_ids for row in rows)
    max_seq_len = max(
        len(prompt) + len(draft) for prompt, draft in zip(prompts, drafts, strict=True)
    )
    resume_path = _resume_path(args)
    trace_path = _access_trace_path(args)
    if args.restart:
        if trace_path is not None and _discard_trace_checkpoint(trace_path):
            _progress("access_trace_checkpoint_discarded", path=str(trace_path))
        if delete_resume(resume_path):
            _progress("resume_discarded", path=str(resume_path))
    runtime = _model_runtime if runtime_factory is None else runtime_factory
    verifier_type = (
        Qwen38DraftVerifier if verifier_factory is None else verifier_factory
    )

    def progress(event: Mapping[str, Any]) -> None:
        fields = dict(event)
        name = str(fields.pop("event", "progress"))
        _progress(f"local_{name}", **fields)

    def head_progress(event: Mapping[str, Any]) -> None:
        fields = dict(event)
        fields.pop("event", None)
        _progress("head_progress", **fields)

    with runtime(
        args,
        max_batch_size=len(rows),
        max_seq_len=max_seq_len,
    ) as model:
        expected_shape = tuple(model.prefill_hidden_shape(len(rows), max_seq_len))
        expected_dtype = str(model.pager.compute_dtype).removeprefix("torch.")
        execution_contract = {
            "device": str(model.pager.device),
            "dtype": expected_dtype,
            "padding_token_id": END_OF_TEXT_TOKEN_ID,
            "eos_token_id": IM_END_TOKEN_ID,
            "accepted_eos_token_ids": [IM_END_TOKEN_ID, END_OF_TEXT_TOKEN_ID],
        }
        graft_contract = (
            None
            if args.mode == "off"
            else {
                "mode": args.mode,
                "layer": args.graft_layer,
                "alpha": args.graft_alpha,
            }
        )
        identity = build_resume_identity(
            source_id=OFFICIAL_REPO_ID,
            source_revision=OFFICIAL_REVISION,
            prompt_token_ids=prompts,
            draft_token_ids=drafts,
            execution_contract=execution_contract,
            graft_contract=graft_contract,
        )
        pair = (
            None if trace_path is None else _load_trace_checkpoint(trace_path, identity)
        )
        if (
            trace_path is not None
            and pair is not None
            and (resume_path.exists() or resume_path.is_symlink())
        ):
            raise CliError("paired trace and legacy resume both exist; use --restart")
        if (
            trace_path is not None
            and pair is None
            and (
                resume_path.exists()
                or resume_path.is_symlink()
                or trace_path.exists()
                or trace_path.is_symlink()
            )
        ):
            raise CliError("unpaired access trace/resume state exists; use --restart")
        active_resume_path = resume_path if pair is None else pair.resume_path
        recorder = (
            None
            if trace_path is None
            else AccessTraceRecorder(initial_trace=None if pair is None else pair.trace)
        )
        if recorder is not None:
            set_observer = getattr(model.pager.source, "set_access_observer", None)
            if not callable(set_observer):
                raise CliError("runtime source cannot attach an access recorder")
            set_observer(recorder)
        _progress("cache_disk_preflight", **_cache_disk_preflight(args))
        _progress(
            "resume_disk_preflight",
            **preflight_resume_disk(
                active_resume_path,
                expected_shape=expected_shape,
                expected_dtype=expected_dtype,
            ),
        )
        resume = load_resume(
            active_resume_path,
            identity,
            expected_shape=expected_shape,
            expected_dtype=expected_dtype,
            n_layers=int(model.config.n_layers),
            active_graft_layer=(args.graft_layer if args.mode != "off" else None),
        )
        if pair is not None:
            if resume is None:
                raise CliError("trace checkpoint resume payload is missing")
            if (
                int(resume.next_layer) != pair.next_layer
                or int(resume.source_body_bytes) != pair.source_body_bytes
            ):
                raise CliError("trace checkpoint and resume counters disagree")
        if resume is not None:
            _progress(
                "resume_reused",
                path=str(active_resume_path),
                next_layer=resume.next_layer,
                layers=int(model.config.n_layers),
            )
        _progress(
            "source_budget_preflight",
            **_apply_cumulative_source_budget(
                args,
                model,
                prior_source_bytes=(0 if resume is None else resume.source_body_bytes),
            ),
        )

        last_checkpoint_state: DraftVerificationResumeState | None = resume

        def checkpoint(state: Any) -> None:
            nonlocal last_checkpoint_state
            if not isinstance(state, DraftVerificationResumeState):
                raise CliError("verifier checkpoint state has the wrong type")
            last_checkpoint_state = state
            if trace_path is None:
                write_resume(
                    resume_path,
                    identity,
                    state,
                    expected_shape=expected_shape,
                    expected_dtype=expected_dtype,
                    n_layers=int(model.config.n_layers),
                    active_graft_layer=(
                        args.graft_layer if args.mode != "off" else None
                    ),
                )
            else:
                assert isinstance(recorder, AccessTraceRecorder)
                setattr(
                    args,
                    "_access_trace_receipt",
                    _write_trace_checkpoint(
                        recorder,
                        trace_path,
                        identity,
                        state,
                        expected_shape=expected_shape,
                        expected_dtype=expected_dtype,
                        n_layers=int(model.config.n_layers),
                        active_graft_layer=(
                            args.graft_layer if args.mode != "off" else None
                        ),
                    ),
                )

        verifier = verifier_type(
            model,
            layer_retries=args.layer_retries,
            head_retries=args.head_retries,
        )
        report = verifier.verify(
            prompts,
            drafts,
            eos_token_id=IM_END_TOKEN_ID,
            eos_token_ids=(IM_END_TOKEN_ID, END_OF_TEXT_TOKEN_ID),
            padding_token_id=END_OF_TEXT_TOKEN_ID,
            max_draft_tokens=args.max_draft_tokens,
            head_block_rows=args.head_block_rows,
            progress=progress,
            head_progress=head_progress,
            resume_state=resume,
            checkpoint=checkpoint,
        )
        if trace_path is not None:
            if last_checkpoint_state is None:
                raise CliError("verifier produced no paired resume checkpoint")
            assert isinstance(recorder, AccessTraceRecorder)
            setattr(
                args,
                "_access_trace_receipt",
                _write_trace_checkpoint(
                    recorder,
                    trace_path,
                    identity,
                    last_checkpoint_state,
                    expected_shape=expected_shape,
                    expected_dtype=expected_dtype,
                    n_layers=int(model.config.n_layers),
                    active_graft_layer=(
                        args.graft_layer if args.mode != "off" else None
                    ),
                ),
            )
        return report


def _result_document(
    args: argparse.Namespace,
    rows: Sequence[PreparedDraft],
    tokenizer: Qwen38Tokenizer,
    report: Any,
) -> dict[str, Any]:
    if len(report.rows) != len(rows):
        raise CliError("Qwen verifier returned the wrong row count")
    items: list[dict[str, Any]] = []
    content_verified = eos_verified = fully_verified = verified_correct = 0
    for index, (prepared, verified) in enumerate(zip(rows, report.rows, strict=True)):
        if int(verified.row) != index:
            raise CliError("Qwen verifier returned rows out of order")
        content_verified += bool(verified.draft_verified)
        eos_verified += verified.eos_verified is True
        fully_verified += bool(verified.fully_verified)
        verified_correct += bool(verified.fully_verified and prepared.candidate_correct)
        items.append(
            {
                **prepared.to_dict(),
                "teacher_forced_greedy_target_ids": list(verified.target_token_ids),
                "teacher_forced_greedy_target_pieces": list(
                    tokenizer.token_pieces(verified.target_token_ids)
                ),
                "verification": verified.to_dict(),
            }
        )
    total = len(rows)
    candidate_correct = sum(row.candidate_correct for row in rows)
    evidence = report.evidence.to_dict()
    return {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "source": {
            "checkpoint": OFFICIAL_REPO_ID,
            "revision": OFFICIAL_REVISION,
            "drafts": str(Path(args.drafts_json).expanduser().resolve()),
            "verification": getattr(args, "_source_verification", None),
        },
        "protocol": {
            "batch_size": total,
            "padding": "right",
            "padding_token_id": END_OF_TEXT_TOKEN_ID,
            "eos_token_id": IM_END_TOKEN_ID,
            "accepted_eos_token_ids": [IM_END_TOKEN_ID, END_OF_TEXT_TOKEN_ID],
            "mode": args.mode,
            "graft_layer": args.graft_layer if args.mode != "off" else None,
            "graft_alpha": args.graft_alpha if args.mode != "off" else None,
            "weight_order": "layer-major, one BF16 matrix at a time",
            "dynamic_cohort": bool(args.dynamic_cohort),
            "item_ids": [row.item_id for row in rows],
        },
        "summary": {
            "candidate_correct": candidate_correct,
            "candidate_accuracy": candidate_correct / total,
            "content_verified": content_verified,
            "eos_verified": eos_verified,
            "fully_verified": fully_verified,
            "total": total,
            "content_verification_rate": content_verified / total,
            "full_verification_rate": fully_verified / total,
            "verified_correct": verified_correct,
            "verified_end_to_end_accuracy": verified_correct / total,
        },
        "traffic": {
            "access_trace": getattr(args, "_access_trace_receipt", None),
            "source_body_bytes": evidence.get("source_body_bytes"),
            "linear_calls": evidence.get("linear_calls"),
            "layer_calls": evidence.get("layer_calls"),
            "layer_retry_count": evidence.get("layer_retry_count"),
            "head_scans": evidence.get("head_scans"),
            "head_retry_count": evidence.get("head_retry_count"),
        },
        "evidence": evidence,
        "items": items,
    }


def run(
    args: argparse.Namespace,
    *,
    tokenizer_factory: Callable[..., Qwen38Tokenizer] = Qwen38Tokenizer,
    runtime_factory: Callable[..., Any] | None = None,
    verifier_factory: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    tokenizer_path = _resolve_tokenizer_path(args)
    tokenizer = tokenizer_factory(tokenizer_path)
    item_ids = _cohort_item_ids(
        args.drafts_json,
        dynamic=bool(args.dynamic_cohort),
    )
    rows = _prepare_drafts(
        args.benchmark,
        args.drafts_json,
        tokenizer,
        max_draft_tokens=args.max_draft_tokens,
        item_ids=item_ids,
    )
    run_dir = Path(args.run_dir).expanduser().resolve()
    result_path = (
        Path(args.result_json).expanduser().resolve()
        if args.result_json
        else run_dir
        / ("inputs.json" if args.prepare_only else f"result-{args.mode}.json")
    )
    if args.prepare_only:
        result = _input_document(rows)
    else:
        _progress("local_verification_start", rows=len(rows), mode=args.mode)
        report = _verify(
            args,
            rows,
            runtime_factory=runtime_factory,
            verifier_factory=verifier_factory,
        )
        result = _result_document(args, rows, tokenizer, report)
    output = _atomic_write_json(result_path, result)
    if not args.prepare_only:
        try:
            trace_path = _access_trace_path(args)
            removed = (
                delete_resume(_resume_path(args))
                if trace_path is None
                else _discard_trace_checkpoint(trace_path, keep_public=True)
            )
        except Exception as exc:
            _progress("resume_cleanup_warning", error=f"{type(exc).__name__}: {exc}")
        else:
            if removed:
                _progress(
                    "resume_removed",
                    path=(
                        str(_resume_path(args))
                        if trace_path is None
                        else str(_trace_checkpoint_manifest(trace_path))
                    ),
                )
        _progress("local_verification_complete", **dict(result["summary"]))
    return result, output


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result, output = run(args)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        sys.stderr.write(
            json.dumps(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            )
            + "\n"
        )
        return 2
    print(
        json.dumps(
            {
                "status": result.get("status", "inputs_ready"),
                "result_json": str(output),
                "summary": result["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
