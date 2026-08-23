#!/usr/bin/env python3
"""Verify a DeepSeek API draft cohort with one streamed V4 layer pass.

The API supplies complete GSM8K continuations cheaply.  The local checkpoint
then checks every drafted next token (and the following EOS) in one right-
padded layer-major pass.  The historical eight-item cohort remains the default;
``--item-ids-json`` selects a larger ordered cohort.  This is a PoC runner, not
a packaging or publishing command.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request

from immer.knowledge import Streamer
from immer.runtimes.deepseek_v4 import (
    DeepSeekV4Config,
    DeepSeekWeightPager,
    StreamedDeepSeekV4,
)
from immer.runtimes.deepseek_v4.benchmark import extract_gsm8k_answer
from immer.runtimes.deepseek_v4.draft_verification import (
    DraftVerificationResumeState,
    LayerwiseDraftVerifier,
)
from immer.runtimes.deepseek_v4.encoding import encode_user_prompt
from immer.runtimes.deepseek_v4.stable_graft import DeepSeekV4StableCrsaGraft


ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_SOURCE = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
API_MODEL = "deepseek-v4-flash"
API_BASE_URL = "https://api.deepseek.com/beta"
ASSISTANT_PREFIX = "Answer:"
EOS_TOKEN_ID = 1
DEFAULT_BENCHMARK = ROOT / "results" / "bench_gsm8k.json"
DEFAULT_TOKENIZER = (
    ROOT / "artifacts" / "private" / "deepseek-v4-reference" / "tokenizer.json"
)
DEFAULT_RUN_DIR = ROOT / "artifacts" / "private" / "deepseek-v4-fertig-draft-verify"
DEFAULT_CACHE = ROOT / "artifacts" / "private" / "deepseek-v4-cache"
DRAFT_SCHEMA = "immer.deepseek-v4-fertig-drafts/v1"
RESULT_SCHEMA = "immer.deepseek-v4-fertig-draft-verification/v1"
RESUME_SCHEMA = "immer.deepseek-v4-fertig-draft-resume/v1"
COHORT_SCHEMA = "immer.deepseek-v4-fertig-cohort/v1"
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
_PINNED_REVISION = re.compile(r"[0-9a-fA-F]{40,64}")
_TRANSIENT_HTTP = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class CliError(RuntimeError):
    """The draft or streamed verification contract was not satisfied."""


@dataclass(frozen=True, slots=True)
class SelectedItem:
    item_id: str
    question: str
    gold: str
    prompt_text: str
    prompt_token_ids: tuple[int, ...]


class LocalTokenizer:
    """The published tokenizer JSON without Transformers-side defaults."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise CliError("local tokenizer execution requires tokenizers") from exc
        try:
            self.backend = Tokenizer.from_file(str(self.path))
        except Exception as exc:
            raise CliError(f"cannot load tokenizer JSON: {self.path}") from exc

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(
            int(token)
            for token in self.backend.encode(text, add_special_tokens=False).ids
        )

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.backend.decode(list(token_ids), skip_special_tokens=True)

    def token_pieces(self, token_ids: Sequence[int]) -> tuple[str, ...]:
        return tuple(
            self.backend.decode([int(token)], skip_special_tokens=False)
            for token in token_ids
        )


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
    parser.add_argument(
        "--item-ids-json",
        help=(
            "JSON array of unique benchmark item IDs in verification order; "
            "defaults to the historical fixed eight-item cohort"
        ),
    )
    parser.add_argument("--tokenizer-json", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--drafts-json")
    parser.add_argument("--result-json")
    parser.add_argument("--resume-file")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="discard the one rolling layer resume file before local verification",
    )
    parser.add_argument("--refresh-drafts", action="store_true")
    parser.add_argument(
        "--draft-only",
        action="store_true",
        help="fetch or reuse API drafts without loading checkpoint weights",
    )
    parser.add_argument("--api-base-url", default=API_BASE_URL)
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--api-timeout", type=_positive_float, default=120.0)
    parser.add_argument("--api-retries", type=_nonnegative_int, default=4)
    parser.add_argument("--max-draft-tokens", type=_positive_int, default=128)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--cache-budget-gb", type=_nonnegative_float, default=6.0)
    parser.add_argument("--source-budget-mb", type=_positive_int, default=196608)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="mps")
    parser.add_argument("--dtype", choices=("auto", "bfloat16"), default="bfloat16")
    parser.add_argument("--no-activation-quantization", action="store_true")
    parser.add_argument("--no-expert-prefetch", action="store_true")
    parser.add_argument("--head-block-rows", type=_positive_int, default=1024)
    parser.add_argument("--layer-retries", type=_nonnegative_int, default=2)
    parser.add_argument("--mode", choices=("off", "stable-crsa"), default="off")
    parser.add_argument("--graft-layer", type=_nonnegative_int, default=21)
    parser.add_argument("--graft-alpha", type=_unit_float, default=0.01)
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


def _read_item_ids(path: str | Path | None) -> tuple[str, ...]:
    if path is None:
        return FIXED_ITEM_IDS
    source = Path(path).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CliError(f"cannot read item-ID manifest: {source}") from exc
    if not isinstance(document, list):
        raise CliError("item-ID manifest root must be an array")
    if not document:
        raise CliError("item-ID manifest must not be empty")
    item_ids: list[str] = []
    seen: set[str] = set()
    for index, raw in enumerate(document):
        if not isinstance(raw, str) or not raw.strip() or raw != raw.strip():
            raise CliError(
                f"item-ID manifest entry {index} must be a non-empty trimmed string"
            )
        if raw in seen:
            raise CliError(f"duplicate item ID in manifest: {raw}")
        seen.add(raw)
        item_ids.append(raw)
    return tuple(item_ids)


def _cohort_document(items: Sequence[SelectedItem]) -> dict[str, Any]:
    rows = [
        {
            "item_id": item.item_id,
            "question": item.question,
            "gold": item.gold,
            "prompt_token_ids": list(item.prompt_token_ids),
        }
        for item in items
    ]
    encoded = json.dumps(
        {"schema": COHORT_SCHEMA, "items": rows},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return {
        "schema": COHORT_SCHEMA,
        "identity": hashlib.sha256(encoded).hexdigest(),
        "item_ids": [item.item_id for item in items],
    }


def _atomic_write_json(path: str | Path, document: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
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
        raise CliError("JSON output contains a non-serializable value") from exc
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resume_path(args: argparse.Namespace) -> Path:
    if args.resume_file:
        return Path(args.resume_file).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    return run_dir / f"resume-{args.mode}.safetensors"


def _resume_identity(
    args: argparse.Namespace,
    prompts: Sequence[Sequence[int]],
    drafts: Sequence[Sequence[int]],
    *,
    resolved_device: str,
    resolved_dtype: str,
    cohort_identity: str | None = None,
) -> str:
    payload = {
        "schema": RESUME_SCHEMA,
        "checkpoint": OFFICIAL_SOURCE,
        "revision": OFFICIAL_REVISION,
        "prompt_token_ids": [list(row) for row in prompts],
        "draft_token_ids": [list(row) for row in drafts],
        "eos_token_id": EOS_TOKEN_ID,
        "padding_token_id": EOS_TOKEN_ID,
        "device": resolved_device,
        "dtype": resolved_dtype,
        "activation_quantization": not args.no_activation_quantization,
        "mode": args.mode,
        "graft_layer": args.graft_layer if args.mode == "stable-crsa" else None,
        "graft_alpha": args.graft_alpha if args.mode == "stable-crsa" else None,
    }
    if cohort_identity is not None:
        payload["cohort_identity"] = cohort_identity
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clear_resume(path: Path) -> bool:
    if path.is_symlink():
        raise CliError(f"resume path must not be a symlink: {path}")
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise CliError(f"cannot remove rolling resume file: {path}") from exc
    return True


def _write_resume(
    path: Path,
    identity: str,
    state: DraftVerificationResumeState,
) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise CliError("rolling resume requires safetensors") from exc

    destination = path.expanduser().resolve()
    if destination.is_symlink():
        raise CliError(f"resume path must not be a symlink: {destination}")
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
        "dtype": str(hidden.dtype).removeprefix("torch."),
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".pending",
        dir=destination.parent,
    )
    os.close(descriptor)
    try:
        save_file({"hidden": hidden}, temporary, metadata=metadata)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception as exc:
        raise CliError("cannot write rolling layer resume file") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _resume_int(metadata: Mapping[str, str], name: str) -> int:
    raw = metadata.get(name)
    try:
        value = int(raw) if raw is not None else -1
    except ValueError as exc:
        raise CliError(f"resume metadata {name} is invalid") from exc
    if value < 0:
        raise CliError(f"resume metadata {name} is invalid")
    return value


def _resume_element_size(dtype: str) -> int:
    element_size = {"bfloat16": 2, "float16": 2, "float32": 4}.get(dtype)
    if element_size is None:
        raise CliError(f"unsupported rolling resume dtype: {dtype}")
    return element_size


def _preflight_resume_disk(
    args: argparse.Namespace,
    resume_path: Path,
    *,
    hidden_bytes: int,
    current_cache_bytes: int,
) -> dict[str, int]:
    resume_parent = resume_path.expanduser().resolve().parent
    cache_root = Path(args.cache_dir).expanduser().resolve()
    resume_parent.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    checkpoint_reserve = 2 * (hidden_bytes + 1024**2) + 512 * 1024**2
    cache_limit = 0 if args.no_cache else int(args.cache_budget_gb * 1024**3)
    cache_growth = max(0, cache_limit - current_cache_bytes)
    resume_free = shutil.disk_usage(resume_parent).free
    same_volume = os.stat(resume_parent).st_dev == os.stat(cache_root).st_dev
    required = checkpoint_reserve + (cache_growth if same_volume else 0)
    if resume_free < required:
        raise CliError(
            "insufficient free disk for rolling resume and bounded cache: "
            f"need {required} bytes, have {resume_free}"
        )
    if not same_volume and cache_growth:
        cache_free = shutil.disk_usage(cache_root).free
        if cache_free < cache_growth:
            raise CliError(
                "insufficient free disk for bounded range cache: "
                f"need {cache_growth} bytes, have {cache_free}"
            )
    return {
        "free_bytes": resume_free,
        "required_bytes": required,
        "hidden_bytes": hidden_bytes,
        "cache_growth_bytes": cache_growth,
    }


def _load_resume(
    path: Path,
    identity: str,
    *,
    legacy_identity: str | None = None,
    expected_shape: Sequence[int],
    expected_dtype: str,
    n_layers: int,
) -> DraftVerificationResumeState | None:
    source = path.expanduser().resolve()
    if not source.exists():
        return None
    if source.is_symlink() or not source.is_file():
        raise CliError(f"resume path must be a regular file: {source}")
    element_size = _resume_element_size(expected_dtype)
    expected_bytes = math.prod(int(value) for value in expected_shape) * element_size
    if source.stat().st_size > expected_bytes + 1024**2:
        raise CliError("rolling resume file exceeds its hidden-state bound")
    try:
        from safetensors import safe_open

        with safe_open(source, framework="pt", device="cpu") as handle:
            if list(handle.keys()) != ["hidden"]:
                raise CliError("rolling resume has unexpected tensors")
            metadata = handle.metadata()
            hidden = handle.get_tensor("hidden")
    except CliError:
        raise
    except Exception as exc:
        raise CliError("cannot decode rolling resume file; use --restart") from exc
    required = {
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
    }
    if (
        not isinstance(metadata, dict)
        or set(metadata) != required
        or metadata.get("schema") != RESUME_SCHEMA
    ):
        raise CliError("rolling resume metadata schema is invalid")
    if metadata.get("identity") not in {identity, legacy_identity}:
        raise CliError("rolling resume belongs to another run; use --restart")
    if tuple(hidden.shape) != tuple(int(value) for value in expected_shape):
        raise CliError("rolling resume hidden shape is invalid")
    observed_dtype = str(hidden.dtype).removeprefix("torch.")
    if metadata.get("dtype") != observed_dtype or observed_dtype != expected_dtype:
        raise CliError("rolling resume hidden dtype is invalid")
    next_layer = _resume_int(metadata, "next_layer")
    layer_calls = _resume_int(metadata, "layer_calls")
    if not 1 <= next_layer <= n_layers or layer_calls != next_layer:
        raise CliError("rolling resume layer boundary is invalid")
    try:
        seconds = float(metadata["seconds"])
    except ValueError as exc:
        raise CliError("rolling resume seconds are invalid") from exc
    if not math.isfinite(seconds) or seconds < 0.0:
        raise CliError("rolling resume seconds are invalid")
    graft_raw = metadata.get("graft_applied")
    if graft_raw not in {"0", "1"}:
        raise CliError("rolling resume graft state is invalid")
    return DraftVerificationResumeState(
        next_layer=next_layer,
        hidden=hidden,
        layer_calls=layer_calls,
        layer_retry_count=_resume_int(metadata, "layer_retry_count"),
        source_body_bytes=_resume_int(metadata, "source_body_bytes"),
        linear_calls=_resume_int(metadata, "linear_calls"),
        seconds=seconds,
        graft_applied=graft_raw == "1",
    )


def _selected_items(
    benchmark: str | Path,
    tokenizer: LocalTokenizer,
    item_ids: Sequence[str] = FIXED_ITEM_IDS,
) -> tuple[SelectedItem, ...]:
    if not item_ids:
        raise CliError("selected item cohort must not be empty")
    if len(set(item_ids)) != len(item_ids):
        raise CliError("selected item cohort contains duplicate IDs")
    document = _read_json(benchmark, "FERTIG GSM8K report")
    raw_rows = document.get("items")
    if not isinstance(raw_rows, list):
        raise CliError("FERTIG GSM8K report has no item list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            continue
        item_id = raw.get("item_id")
        if isinstance(item_id, str):
            if item_id in by_id:
                raise CliError(f"duplicate benchmark item: {item_id}")
            by_id[item_id] = raw

    selected: list[SelectedItem] = []
    for item_id in item_ids:
        try:
            raw = by_id[item_id]
        except KeyError as exc:
            raise CliError(f"selected benchmark item is missing: {item_id}") from exc
        if raw.get("status") not in {"abstained", "correct"}:
            raise CliError(f"selected benchmark item is unusable: {item_id}")
        question = raw.get("question")
        if not isinstance(question, str) or not question.strip():
            raise CliError(f"selected item has no question: {item_id}")
        gold = extract_gsm8k_answer(raw.get("gold"))
        if gold is None:
            raise CliError(f"selected item has no numeric gold answer: {item_id}")
        prompt = (
            encode_user_prompt(
                question,
                thinking_mode="chat",
                reasoning_effort="low",
            )
            + ASSISTANT_PREFIX
        )
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            raise CliError(f"selected item encoded to an empty prompt: {item_id}")
        selected.append(SelectedItem(item_id, question, gold, prompt, prompt_ids))
    return tuple(selected)


def _request_payload(item: SelectedItem, *, max_tokens: int) -> dict[str, Any]:
    return {
        "model": API_MODEL,
        "messages": [
            {"role": "user", "content": item.question},
            {
                "role": "assistant",
                "content": ASSISTANT_PREFIX,
                "prefix": True,
            },
        ],
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": max_tokens,
        "logprobs": True,
        "stream": False,
    }


def _retry_delay(error: BaseException, attempt: int) -> float:
    if isinstance(error, urllib.error.HTTPError):
        raw = error.headers.get("Retry-After") if error.headers is not None else None
        if raw is not None:
            try:
                return min(30.0, max(0.0, float(raw)))
            except ValueError:
                pass
    return min(8.0, 0.5 * (2**attempt))


def _post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    api_key: str,
    timeout: float,
    retries: int,
    opener: Callable[..., Any] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "immer-deepseek-v4-draft-verify/1",
        },
        method="POST",
    )
    open_url = urllib.request.urlopen if opener is None else opener
    for attempt in range(retries + 1):
        try:
            with open_url(request, timeout=timeout) as response:
                status = int(getattr(response, "status", 200))
                if not 200 <= status < 300:
                    raise CliError(f"DeepSeek API returned HTTP {status}")
                body = response.read()
            decoded = json.loads(body.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise CliError("DeepSeek API response root is not an object")
            return decoded
        except urllib.error.HTTPError as exc:
            transient = exc.code in _TRANSIENT_HTTP
            if not transient or attempt == retries:
                raise CliError(f"DeepSeek API returned HTTP {exc.code}") from exc
            sleeper(_retry_delay(exc, attempt))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                raise CliError(f"DeepSeek API transport failed: {exc}") from exc
            sleeper(_retry_delay(exc, attempt))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CliError("DeepSeek API returned invalid JSON") from exc
    raise AssertionError("unreachable retry loop")


def _required_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CliError(f"DeepSeek API {label} is missing or invalid")
    return value


def _parse_api_draft(
    item: SelectedItem,
    tokenizer: LocalTokenizer,
    response: Mapping[str, Any],
    *,
    max_draft_tokens: int,
) -> dict[str, Any]:
    choices = response.get("choices")
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], Mapping)
    ):
        raise CliError(f"{item.item_id}: DeepSeek response has no first choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
        raise CliError(f"{item.item_id}: DeepSeek response has no text content")
    content = str(message["content"])
    logprobs = choice.get("logprobs")
    logprob_rows = logprobs.get("content") if isinstance(logprobs, Mapping) else None
    if not isinstance(logprob_rows, list) or any(
        not isinstance(row, Mapping) or not isinstance(row.get("token"), str)
        for row in logprob_rows
    ):
        raise CliError(f"{item.item_id}: DeepSeek response has no token logprobs")
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        raise CliError(f"{item.item_id}: DeepSeek response has no usage")
    prompt_tokens = _required_count(usage.get("prompt_tokens"), "prompt_tokens")
    completion_tokens = _required_count(
        usage.get("completion_tokens"), "completion_tokens"
    )
    total_tokens = _required_count(usage.get("total_tokens"), "total_tokens")
    if prompt_tokens != len(item.prompt_token_ids):
        raise CliError(
            f"{item.item_id}: API/local prompt token mismatch "
            f"({prompt_tokens} != {len(item.prompt_token_ids)})"
        )
    token_ids = tokenizer.encode(content)
    api_token_pieces = tuple(str(row["token"]) for row in logprob_rows)
    if len(token_ids) != len(api_token_pieces):
        raise CliError(
            f"{item.item_id}: API/local completion token mismatch "
            f"({len(api_token_pieces)} != {len(token_ids)})"
        )
    local_token_pieces = tokenizer.token_pieces(token_ids)
    if local_token_pieces != api_token_pieces or "".join(api_token_pieces) != content:
        raise CliError(f"{item.item_id}: API/local completion token pieces differ")
    if completion_tokens != len(api_token_pieces):
        raise CliError(
            f"{item.item_id}: API usage/logprob completion token mismatch "
            f"({completion_tokens} != {len(api_token_pieces)})"
        )
    if total_tokens != prompt_tokens + completion_tokens:
        raise CliError(f"{item.item_id}: API total token accounting mismatch")
    if len(token_ids) > max_draft_tokens:
        raise CliError(f"{item.item_id}: completion exceeds local draft bound")
    if EOS_TOKEN_ID in token_ids:
        raise CliError(f"{item.item_id}: API content unexpectedly contains EOS")
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop":
        raise CliError(
            f"{item.item_id}: API draft did not reach EOS "
            f"(finish_reason={finish_reason!r})"
        )
    response_model = response.get("model")
    if response_model != API_MODEL:
        raise CliError(
            f"{item.item_id}: API response model mismatch "
            f"({response_model!r} != {API_MODEL!r})"
        )
    system_fingerprint = response.get("system_fingerprint")
    if not isinstance(system_fingerprint, str) or not system_fingerprint.strip():
        raise CliError(f"{item.item_id}: API response has no system_fingerprint")
    return {
        "item_id": item.item_id,
        "response_model": response_model,
        "system_fingerprint": system_fingerprint,
        "content": content,
        "token_ids": list(token_ids),
        "logprob_tokens": list(api_token_pieces),
        "logprob_token_count": len(api_token_pieces),
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def _fetch_api_draft(
    item: SelectedItem,
    tokenizer: LocalTokenizer,
    args: argparse.Namespace,
    api_key: str,
) -> dict[str, Any]:
    response = _post_json(
        f"{args.api_base_url.rstrip('/')}/chat/completions",
        _request_payload(item, max_tokens=args.max_draft_tokens),
        api_key=api_key,
        timeout=args.api_timeout,
        retries=args.api_retries,
    )
    return _parse_api_draft(
        item,
        tokenizer,
        response,
        max_draft_tokens=args.max_draft_tokens,
    )


def _draft_document(
    items: Sequence[SelectedItem],
    drafts: Sequence[Mapping[str, Any]],
    *,
    max_draft_tokens: int,
) -> dict[str, Any]:
    return {
        "schema": DRAFT_SCHEMA,
        "cohort": _cohort_document(items),
        "protocol": {
            "model": API_MODEL,
            "assistant_prefix": ASSISTANT_PREFIX,
            "thinking": {"type": "disabled"},
            "temperature": 0,
            "max_tokens": max_draft_tokens,
            "logprobs": True,
        },
        "items": [
            {
                "item_id": item.item_id,
                "question": item.question,
                "gold": item.gold,
                "prompt_token_ids": list(item.prompt_token_ids),
                "response_model": draft["response_model"],
                "system_fingerprint": draft["system_fingerprint"],
                "content": draft["content"],
                "token_ids": list(draft["token_ids"]),
                "logprob_tokens": list(draft["logprob_tokens"]),
                "logprob_token_count": draft["logprob_token_count"],
                "finish_reason": draft["finish_reason"],
                "usage": dict(draft["usage"]),
            }
            for item, draft in zip(items, drafts, strict=True)
        ],
    }


def _validate_draft_document(
    document: Mapping[str, Any],
    items: Sequence[SelectedItem],
    tokenizer: LocalTokenizer,
    *,
    max_draft_tokens: int,
) -> tuple[dict[str, Any], ...]:
    if document.get("schema") != DRAFT_SCHEMA:
        raise CliError("draft cache schema mismatch; use --refresh-drafts")
    expected_cohort = _cohort_document(items)
    cached_cohort = document.get("cohort")
    is_legacy_fixed_cache = (
        cached_cohort is None
        and tuple(item.item_id for item in items) == FIXED_ITEM_IDS
    )
    if not is_legacy_fixed_cache and cached_cohort != expected_cohort:
        raise CliError("draft cache cohort mismatch; use --refresh-drafts")
    protocol = document.get("protocol")
    expected_protocol = _draft_document((), (), max_draft_tokens=max_draft_tokens)[
        "protocol"
    ]
    if protocol != expected_protocol:
        raise CliError("draft cache protocol mismatch; use --refresh-drafts")
    rows = document.get("items")
    if not isinstance(rows, list) or len(rows) != len(items):
        raise CliError("draft cache item count mismatch; use --refresh-drafts")
    validated: list[dict[str, Any]] = []
    for item, raw in zip(items, rows, strict=True):
        if not isinstance(raw, Mapping) or raw.get("item_id") != item.item_id:
            raise CliError("draft cache item order mismatch; use --refresh-drafts")
        if raw.get("question") != item.question or raw.get("gold") != item.gold:
            raise CliError(f"{item.item_id}: stale cached benchmark row")
        if raw.get("prompt_token_ids") != list(item.prompt_token_ids):
            raise CliError(f"{item.item_id}: stale cached prompt tokens")
        content = raw.get("content")
        token_ids = raw.get("token_ids")
        if not isinstance(content, str) or not isinstance(token_ids, list):
            raise CliError(f"{item.item_id}: invalid cached draft")
        encoded = tokenizer.encode(content)
        if token_ids != list(encoded):
            raise CliError(f"{item.item_id}: stale cached completion tokens")
        if len(encoded) > max_draft_tokens or EOS_TOKEN_ID in encoded:
            raise CliError(f"{item.item_id}: cached completion violates token bound")
        token_pieces = raw.get("logprob_tokens")
        if (
            not isinstance(token_pieces, list)
            or any(not isinstance(piece, str) for piece in token_pieces)
            or tuple(token_pieces) != tokenizer.token_pieces(encoded)
            or "".join(token_pieces) != content
        ):
            raise CliError(f"{item.item_id}: cached API/local token pieces differ")
        if raw.get("logprob_token_count") != len(encoded):
            raise CliError(f"{item.item_id}: cached logprob count mismatch")
        if raw.get("finish_reason") != "stop":
            raise CliError(f"{item.item_id}: cached API draft did not reach EOS")
        if raw.get("response_model") != API_MODEL:
            raise CliError(f"{item.item_id}: cached API response model mismatch")
        fingerprint = raw.get("system_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise CliError(f"{item.item_id}: cached system_fingerprint is invalid")
        usage = raw.get("usage")
        if not isinstance(usage, Mapping):
            raise CliError(f"{item.item_id}: cached usage is invalid")
        if _required_count(usage.get("prompt_tokens"), "prompt_tokens") != len(
            item.prompt_token_ids
        ):
            raise CliError(f"{item.item_id}: cached API prompt count mismatch")
        completion_tokens = _required_count(
            usage.get("completion_tokens"), "completion_tokens"
        )
        if completion_tokens != len(encoded):
            raise CliError(f"{item.item_id}: cached completion usage mismatch")
        total_tokens = _required_count(usage.get("total_tokens"), "total_tokens")
        if total_tokens != len(item.prompt_token_ids) + completion_tokens:
            raise CliError(f"{item.item_id}: cached total token accounting mismatch")
        validated.append(dict(raw))
    return tuple(validated)


def _prepare_drafts(
    args: argparse.Namespace,
    items: Sequence[SelectedItem],
    tokenizer: LocalTokenizer,
    *,
    fetcher: Callable[
        [SelectedItem, LocalTokenizer, argparse.Namespace, str], dict[str, Any]
    ]
    | None = None,
) -> tuple[dict[str, Any], ...]:
    run_dir = Path(args.run_dir).expanduser().resolve()
    draft_path = (
        Path(args.drafts_json).expanduser().resolve()
        if args.drafts_json
        else run_dir / "drafts.json"
    )
    if draft_path.is_file() and not args.refresh_drafts:
        _progress("drafts_reused", path=str(draft_path))
        return _validate_draft_document(
            _read_json(draft_path, "draft cache"),
            items,
            tokenizer,
            max_draft_tokens=args.max_draft_tokens,
        )

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise CliError(
            f"{args.api_key_env} is required when no reusable draft cache exists"
        )
    call = _fetch_api_draft if fetcher is None else fetcher
    drafts: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        _progress("api_draft_start", item_id=item.item_id, row=index)
        draft = call(item, tokenizer, args, api_key)
        drafts.append(draft)
        _progress(
            "api_draft_complete",
            item_id=item.item_id,
            row=index,
            tokens=len(draft["token_ids"]),
        )
    document = _draft_document(
        items,
        drafts,
        max_draft_tokens=args.max_draft_tokens,
    )
    validated = _validate_draft_document(
        document,
        items,
        tokenizer,
        max_draft_tokens=args.max_draft_tokens,
    )
    _atomic_write_json(draft_path, document)
    return validated


def _progress(event: str, **fields: Any) -> None:
    sys.stderr.write(json.dumps({"event": event, **fields}, sort_keys=True) + "\n")
    sys.stderr.flush()


@contextmanager
def _model_runtime(
    args: argparse.Namespace,
    *,
    max_seq_len: int,
    max_batch_size: int,
):
    if _PINNED_REVISION.fullmatch(OFFICIAL_REVISION) is None:
        raise CliError("official checkpoint revision is not immutable")
    source = Streamer(
        OFFICIAL_SOURCE,
        revision=OFFICIAL_REVISION,
        budget_mb=args.source_budget_mb,
        cache_dir=Path(args.cache_dir).expanduser().resolve(),
        use_cache=not args.no_cache,
        max_cache_bytes=int(args.cache_budget_gb * 1024**3),
        verbose=False,
    )
    pager: DeepSeekWeightPager | None = None
    model: StreamedDeepSeekV4 | None = None
    try:
        try:
            raw_config = source.reader.fetch_file("config.json")
            config_document = json.loads(raw_config)
        except Exception as exc:
            raise CliError("cannot load DeepSeek-V4 config JSON") from exc
        if not isinstance(config_document, Mapping):
            raise CliError("DeepSeek-V4 config root must be an object")
        config = DeepSeekV4Config.from_mapping(config_document)
        pager = DeepSeekWeightPager(
            source,
            device=args.device,
            compute_dtype=args.dtype,
            simulate_activation_quantization=not args.no_activation_quantization,
            expert_prefetch=not args.no_expert_prefetch,
        )
        graft = None
        graft_layer = None
        if args.mode == "stable-crsa":
            graft = DeepSeekV4StableCrsaGraft(
                mode="crsa",
                alpha=args.graft_alpha,
                max_history=max_seq_len,
            )
            graft_layer = args.graft_layer
        model = StreamedDeepSeekV4(
            config,
            pager,
            graft=graft,
            graft_layer=graft_layer,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )
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
            source.close()
        except Exception as exc:
            if cleanup_error is None:
                cleanup_error = exc
        if active_error is None and cleanup_error is not None:
            raise cleanup_error


def _verify_locally(
    args: argparse.Namespace,
    items: Sequence[SelectedItem],
    drafts: Sequence[Mapping[str, Any]],
    *,
    runtime_factory: Callable[..., Any] | None = None,
    verifier_factory: Callable[..., Any] | None = None,
) -> Any:
    prompts = tuple(item.prompt_token_ids for item in items)
    draft_ids = tuple(tuple(int(token) for token in row["token_ids"]) for row in drafts)
    max_seq_len = max(
        len(prompt) + len(draft)
        for prompt, draft in zip(prompts, draft_ids, strict=True)
    )
    resume_path = _resume_path(args)
    if args.restart and _clear_resume(resume_path):
        _progress("resume_discarded", path=str(resume_path))
    runtime = _model_runtime if runtime_factory is None else runtime_factory
    verifier_type = (
        LayerwiseDraftVerifier if verifier_factory is None else verifier_factory
    )

    def progress(row: Mapping[str, Any]) -> None:
        fields = dict(row)
        phase = str(fields.pop("event", "progress"))
        _progress(f"local_{phase}", **fields)

    def head_progress(row: Mapping[str, Any]) -> None:
        fields = dict(row)
        fields.pop("event", None)
        _progress("head_progress", **fields)

    with runtime(
        args,
        max_seq_len=max_seq_len,
        max_batch_size=len(items),
    ) as model:
        expected_shape = (
            len(prompts),
            max_seq_len,
            int(model.config.hc_mult),
            int(model.config.dim),
        )
        resolved_device = str(model.pager.device)
        expected_dtype = str(model.pager.compute_dtype).removeprefix("torch.")
        resume_identity = _resume_identity(
            args,
            prompts,
            draft_ids,
            resolved_device=resolved_device,
            resolved_dtype=expected_dtype,
            cohort_identity=_cohort_document(items)["identity"],
        )
        legacy_resume_identity = None
        if tuple(item.item_id for item in items) == FIXED_ITEM_IDS:
            legacy_resume_identity = _resume_identity(
                args,
                prompts,
                draft_ids,
                resolved_device=resolved_device,
                resolved_dtype=expected_dtype,
            )
        raw_cache_bytes = model.pager.source.metrics().get("cache_bytes", 0)
        current_cache_bytes = (
            int(raw_cache_bytes)
            if isinstance(raw_cache_bytes, (int, float))
            and not isinstance(raw_cache_bytes, bool)
            and raw_cache_bytes >= 0
            else 0
        )
        hidden_bytes = math.prod(expected_shape) * _resume_element_size(expected_dtype)
        disk = _preflight_resume_disk(
            args,
            resume_path,
            hidden_bytes=hidden_bytes,
            current_cache_bytes=current_cache_bytes,
        )
        _progress("resume_disk_preflight", **disk)
        resume_state = _load_resume(
            resume_path,
            resume_identity,
            legacy_identity=legacy_resume_identity,
            expected_shape=expected_shape,
            expected_dtype=expected_dtype,
            n_layers=int(model.config.n_layers),
        )
        if resume_state is not None:
            _progress(
                "resume_reused",
                path=str(resume_path),
                next_layer=resume_state.next_layer,
                layers=int(model.config.n_layers),
            )

        def checkpoint(state: DraftVerificationResumeState) -> None:
            _write_resume(resume_path, resume_identity, state)

        verifier = verifier_type(model, layer_retries=args.layer_retries)
        return verifier.verify(
            prompts,
            draft_ids,
            eos_token_id=EOS_TOKEN_ID,
            padding_token_id=EOS_TOKEN_ID,
            max_draft_tokens=args.max_draft_tokens,
            head_block_rows=args.head_block_rows,
            progress=progress,
            head_progress=head_progress,
            resume_state=resume_state,
            checkpoint=checkpoint,
        )


def _api_summary(
    items: Sequence[SelectedItem], drafts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    correct = 0
    prompt_tokens = completion_tokens = total_tokens = 0
    for item, draft in zip(items, drafts, strict=True):
        answer = extract_gsm8k_answer(ASSISTANT_PREFIX + str(draft["content"]))
        correct += answer == item.gold
        usage = draft["usage"]
        prompt_tokens += int(usage["prompt_tokens"])
        completion_tokens += int(usage["completion_tokens"])
        total_tokens += int(usage["total_tokens"])
    return {
        "correct": correct,
        "total": len(items),
        "accuracy": correct / len(items),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def _verification_result(
    args: argparse.Namespace,
    items: Sequence[SelectedItem],
    drafts: Sequence[Mapping[str, Any]],
    tokenizer: LocalTokenizer,
    report: Any,
) -> dict[str, Any]:
    if len(report.rows) != len(items):
        raise CliError("local verifier returned the wrong row count")
    rows: list[dict[str, Any]] = []
    content_verified = eos_verified = fully_verified = local_correct = 0
    for index, (item, draft, verified) in enumerate(
        zip(items, drafts, report.rows, strict=True)
    ):
        if int(verified.row) != index:
            raise CliError("local verifier returned rows out of order")
        api_content = ASSISTANT_PREFIX + str(draft["content"])
        api_answer = extract_gsm8k_answer(api_content)
        content_is_verified = bool(verified.draft_verified)
        effective_eos_verified = verified.eos_verified if content_is_verified else None
        local_content = None
        local_answer = None
        if bool(verified.fully_verified):
            local_content = ASSISTANT_PREFIX + tokenizer.decode(
                verified.draft_token_ids
            )
            local_answer = extract_gsm8k_answer(local_content)
        is_local_correct = bool(verified.fully_verified) and local_answer == item.gold
        content_verified += content_is_verified
        eos_verified += effective_eos_verified is True
        fully_verified += bool(verified.fully_verified)
        local_correct += is_local_correct
        verification = verified.to_dict()
        rows.append(
            {
                "item_id": item.item_id,
                "gold": item.gold,
                "api_content": api_content,
                "api_response_model": draft["response_model"],
                "api_system_fingerprint": draft["system_fingerprint"],
                "api_answer": api_answer,
                "api_correct": api_answer == item.gold,
                "locally_fully_verified_content": local_content,
                "local_answer": local_answer,
                "local_correct": is_local_correct,
                "content_verified": content_is_verified,
                "eos_verified": effective_eos_verified,
                "fully_verified": bool(verified.fully_verified),
                "verification": verification,
            }
        )
    evidence = report.evidence.to_dict()
    api = _api_summary(items, drafts)
    total = len(items)
    return {
        "schema": RESULT_SCHEMA,
        "status": "complete",
        "protocol": {
            "checkpoint": OFFICIAL_SOURCE,
            "revision": OFFICIAL_REVISION,
            "api_model": API_MODEL,
            "prompt": "official-chat/low + Answer:",
            "batch_size": total,
            "padding": "right",
            "eos_token_id": EOS_TOKEN_ID,
            "padding_token_id": EOS_TOKEN_ID,
            "mode": args.mode,
            "graft_layer": args.graft_layer if args.mode == "stable-crsa" else None,
            "graft_alpha": args.graft_alpha if args.mode == "stable-crsa" else None,
        },
        "summary": {
            "all_verified": bool(report.all_verified),
            "content_verified": content_verified,
            "eos_verified": eos_verified,
            "fully_verified": fully_verified,
            "total": total,
            "content_verification_rate": content_verified / total,
            "eos_verification_rate": eos_verified / total,
            "api_accuracy": api["accuracy"],
            "local_correct": local_correct,
            "local_verified_total": fully_verified,
            "local_verified_subset_accuracy": (
                local_correct / fully_verified if fully_verified else None
            ),
            "local_end_to_end_accuracy": local_correct / total,
        },
        "traffic": {
            "api": api["usage"],
            "local": {
                "source_body_bytes": evidence.get("source_body_bytes"),
                "linear_calls": evidence.get("linear_calls"),
                "layer_calls": evidence.get("layer_calls"),
                "layer_retry_count": evidence.get("layer_retry_count"),
                "head_scans": evidence.get("head_scans"),
                "head_retry_count": evidence.get("head_retry_count"),
            },
        },
        "evidence": evidence,
        "items": rows,
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    item_ids = _read_item_ids(args.item_ids_json)
    tokenizer = LocalTokenizer(args.tokenizer_json)
    items = _selected_items(args.benchmark, tokenizer, item_ids)
    drafts = _prepare_drafts(args, items, tokenizer)
    run_dir = Path(args.run_dir).expanduser().resolve()
    default_result_name = (
        "result-drafts.json" if args.draft_only else f"result-{args.mode}.json"
    )
    result_path = (
        Path(args.result_json).expanduser().resolve()
        if args.result_json
        else run_dir / default_result_name
    )
    if args.draft_only:
        api = _api_summary(items, drafts)
        result = {
            "schema": RESULT_SCHEMA,
            "status": "drafts_ready",
            "summary": {
                "items": len(items),
                "api_correct": api["correct"],
                "api_accuracy": api["accuracy"],
            },
            "traffic": {"api": api["usage"], "local": None},
        }
    else:
        _progress("local_verification_start", rows=len(items), mode=args.mode)
        report = _verify_locally(args, items, drafts)
        result = _verification_result(args, items, drafts, tokenizer, report)
        _progress("local_verification_complete", **dict(result["summary"]))
    _atomic_write_json(result_path, result)
    if not args.draft_only:
        resume_path = _resume_path(args)
        try:
            removed = _clear_resume(resume_path)
        except CliError as exc:
            _progress(
                "resume_cleanup_warning",
                path=str(resume_path),
                error=str(exc),
            )
        else:
            if removed:
                _progress("resume_removed", path=str(resume_path))
    return result, result_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result, result_path = run(args)
    except Exception as exc:
        sys.stderr.write(
            json.dumps(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
                sort_keys=True,
            )
            + "\n"
        )
        return 2
    sys.stdout.write(
        json.dumps(
            {
                "status": result["status"],
                "result_json": str(result_path),
                "summary": result["summary"],
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
