#!/usr/bin/env python3
"""Run a local Qwen model on a fixed historical FERTIG-abstention cohort.

The command is deliberately local-only: it resolves an already cached Hugging
Face snapshot (or an explicit checkpoint directory) and never downloads model
files.  MLX is imported only after the benchmark and checkpoint contracts have
been validated, so ``--help`` and unit tests do not require an MLX runtime.

    PYTHONPATH=src python3 scripts/qwen_local_baseline.py
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
import time
from typing import Any, Protocol
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
DEFAULT_BENCHMARK = ROOT / "results" / "bench_gsm8k.json"
DEFAULT_OUTPUT = ROOT / "results" / "qwen_local_baseline.json"
DEFAULT_OPENAI_URL = "http://127.0.0.1:8780/v1"
RESULT_SCHEMA = "immer.qwen-local-fertig-baseline/v1"
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
SYSTEM_PROMPT = (
    "Solve the math problem internally. Return only #### followed by the numeric "
    "answer. Do not show work."
)
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z0-9]|\.\d)"
)
_COMMIT = re.compile(r"[0-9a-fA-F]{40,64}")


class CliError(RuntimeError):
    """The local benchmark contract could not be satisfied."""


def _canonical_digest(document: Any) -> str:
    payload = json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _seal_report(document: Mapping[str, Any]) -> dict[str, Any]:
    if "report_sha256" in document:
        raise CliError("report is already sealed")
    sealed = dict(document)
    sealed["report_sha256"] = _canonical_digest(sealed)
    return sealed


def _public_path(value: str | Path) -> str:
    resolved = Path(value).expanduser().resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return "<external>"


def _public_backend_location(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
            return "<loopback-openai-compatible>"
        return "<external-openai-compatible>"
    location = _public_path(value)
    return (
        "<external-local-checkpoint>" if location == "<external>" else location
    )


@dataclass(frozen=True, slots=True)
class BenchmarkItem:
    item_id: str
    question: str
    gold: str


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    peak_memory_gb: float | None = None


class GenerationBackend(Protocol):
    model_id: str
    model_path: str
    model_revision: str | None
    backend_name: str
    load_seconds: float

    def generate(self, question: str, *, max_tokens: int) -> Generation: ...


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
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="cached Hugging Face model id or local MLX checkpoint directory",
    )
    parser.add_argument(
        "--backend",
        choices=("mlx", "openai"),
        default="mlx",
        help="local MLX checkpoint or an OpenAI-compatible Qwen server",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_OPENAI_URL,
        help="OpenAI-compatible /v1 endpoint (used by --backend openai)",
    )
    parser.add_argument(
        "--openai-prompt-mode",
        choices=("chat", "qwen3.8-no-thinking"),
        default="chat",
        help=(
            "server-side chat templating or the pinned Qwen3.8 no-thinking "
            "prompt sent through /completions"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=300.0,
        help="per-question server timeout in seconds",
    )
    parser.add_argument(
        "--revision",
        help="optional cached Hugging Face commit/ref (never fetched remotely)",
    )
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-tokens", type=_positive_int, default=32)
    parser.add_argument("--seed", type=_nonnegative_int, default=0)
    return parser


def extract_numeric_answer(value: Any) -> str | None:
    """Return GSM8K's canonical final decimal, accepting a marker or last number."""

    if value is None:
        return None
    raw = str(value).strip()
    final = raw.rsplit("####", 1)[-1] if "####" in raw else raw
    matches = _NUMBER.findall(final)
    if not matches:
        return None
    try:
        number = Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    if number == 0:
        return "0"
    normalized = format(number.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CliError(f"cannot read {label}: {source}") from exc
    if not isinstance(document, dict):
        raise CliError(f"{label} root must be an object")
    return document


def select_fixed_items(benchmark: str | Path) -> tuple[BenchmarkItem, ...]:
    """Select the same eight historical rows as FERTIG improves around them."""

    document = _read_json(benchmark, "FERTIG GSM8K report")
    raw_items = document.get("items")
    if not isinstance(raw_items, list):
        raise CliError("FERTIG GSM8K report has no item list")
    by_id: dict[str, Mapping[str, Any]] = {}
    for raw in raw_items:
        if not isinstance(raw, Mapping):
            continue
        item_id = raw.get("item_id")
        if isinstance(item_id, str):
            if item_id in by_id:
                raise CliError(f"duplicate benchmark item: {item_id}")
            by_id[item_id] = raw

    selected: list[BenchmarkItem] = []
    for item_id in FIXED_ITEM_IDS:
        raw = by_id.get(item_id)
        if raw is None:
            raise CliError(f"fixed benchmark item is missing: {item_id}")
        if raw.get("status") not in {"abstained", "correct"}:
            raise CliError(f"fixed historical cohort item is unusable: {item_id}")
        question = raw.get("question")
        if not isinstance(question, str) or not question.strip():
            raise CliError(f"fixed item has no question: {item_id}")
        gold = extract_numeric_answer(raw.get("gold"))
        if gold is None:
            raise CliError(f"fixed item has no numeric gold answer: {item_id}")
        selected.append(BenchmarkItem(item_id, question, gold))
    return tuple(selected)


def _validate_checkpoint(path: Path) -> Path:
    checkpoint = path.expanduser().resolve()
    if not checkpoint.is_dir():
        raise CliError(f"local MLX checkpoint directory does not exist: {checkpoint}")
    if not (checkpoint / "config.json").is_file():
        raise CliError(f"local MLX checkpoint has no config.json: {checkpoint}")
    if not any(
        (checkpoint / name).is_file() for name in ("tokenizer.json", "vocab.json")
    ):
        raise CliError(f"local MLX checkpoint has no tokenizer files: {checkpoint}")
    if not any(checkpoint.glob("*.safetensors")) and not any(checkpoint.glob("*.npz")):
        raise CliError(f"local MLX checkpoint has no model weights: {checkpoint}")
    return checkpoint


def resolve_local_model(
    model: str, revision: str | None = None
) -> tuple[Path, str | None]:
    """Resolve a path or cached Hub snapshot while explicitly forbidding download."""

    explicit = Path(model).expanduser()
    if explicit.exists() or explicit.is_absolute() or model.startswith("."):
        checkpoint = _validate_checkpoint(explicit)
        inferred = checkpoint.name if _COMMIT.fullmatch(checkpoint.name) else None
        return checkpoint, inferred
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise CliError(
            "huggingface_hub is required to resolve a cached model id; "
            "pass an explicit local checkpoint path"
        ) from exc
    try:
        cached = snapshot_download(
            repo_id=model,
            revision=revision,
            local_files_only=True,
        )
    except Exception as exc:
        suffix = f" at revision {revision}" if revision else ""
        raise CliError(
            f"local MLX checkpoint is unavailable for {model}{suffix}; this "
            "runner never downloads models, so cache it first or pass a local path"
        ) from exc
    checkpoint = _validate_checkpoint(Path(cached))
    inferred = checkpoint.name if _COMMIT.fullmatch(checkpoint.name) else revision
    return checkpoint, inferred


class MlxBackend:
    """Lazy-imported deterministic ``mlx_lm`` generation backend."""

    backend_name = "mlx_lm"

    def __init__(self, model_id: str, revision: str | None, *, seed: int) -> None:
        checkpoint, resolved_revision = resolve_local_model(model_id, revision)
        try:
            import mlx.core as mx
            from mlx_lm import load, stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise CliError("local Qwen execution requires mlx and mlx_lm") from exc

        started = time.perf_counter()
        try:
            mx.random.seed(seed)
            model, tokenizer = load(str(checkpoint), lazy=True)
            if hasattr(model, "eval"):
                model.eval()
        except Exception as exc:
            raise CliError(f"cannot load local MLX checkpoint: {checkpoint}") from exc
        self.model_id = model_id
        self.model_path = str(checkpoint)
        self.model_revision = resolved_revision
        self.load_seconds = time.perf_counter() - started
        self._mx = mx
        self._model = model
        self._tokenizer = tokenizer
        self._stream_generate = stream_generate
        self._sampler = make_sampler(temp=0.0)

    def generate(self, question: str, *, max_tokens: int) -> Generation:
        messages = (
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        )
        try:
            prompt = self._tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )
        except Exception as exc:
            raise CliError("Qwen tokenizer cannot render its chat template") from exc

        chunks: list[str] = []
        last: Any = None
        for response in self._stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=self._sampler,
        ):
            chunks.append(str(response.text))
            last = response
        self._mx.synchronize()
        if last is None:
            raise CliError("MLX generation returned no response")
        return Generation(
            text="".join(chunks),
            prompt_tokens=_optional_count(getattr(last, "prompt_tokens", None)),
            completion_tokens=_optional_count(getattr(last, "generation_tokens", None)),
            finish_reason=_optional_text(getattr(last, "finish_reason", None)),
            peak_memory_gb=_optional_nonnegative(getattr(last, "peak_memory", None)),
        )


class OpenAIBackend:
    """Deterministic Qwen generation through a local SSH tunnel or LAN server."""

    backend_name = "openai-compatible"

    def __init__(
        self,
        model_id: str,
        revision: str | None,
        *,
        base_url: str,
        timeout: float,
        seed: int,
        prompt_mode: str = "chat",
    ) -> None:
        endpoint = str(base_url).rstrip("/")
        if not endpoint.startswith(("http://", "https://")):
            raise CliError("OpenAI base URL must use http:// or https://")
        self.model_id = model_id
        self.model_path = endpoint
        self.model_revision = revision
        self.base_url = endpoint
        self.timeout = float(timeout)
        self.seed = seed
        if prompt_mode not in {"chat", "qwen3.8-no-thinking"}:
            raise CliError(f"unsupported OpenAI prompt mode: {prompt_mode}")
        self.prompt_mode = prompt_mode
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/models", timeout=min(self.timeout, 5.0)
            ) as response:
                if response.status != 200:
                    raise CliError(
                        f"Qwen server liveness returned HTTP {response.status}"
                    )
        except CliError:
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise CliError(f"Qwen server is unavailable: {self.base_url}") from exc
        self.load_seconds = time.perf_counter() - started

    def generate(self, question: str, *, max_tokens: int) -> Generation:
        if self.prompt_mode == "qwen3.8-no-thinking":
            from immer.runtimes.qwen3_8.encoding import Qwen38Tokenizer

            payload = {
                "model": self.model_id,
                "prompt": Qwen38Tokenizer.render_no_thinking_prompt(
                    SYSTEM_PROMPT, question
                ),
                "max_tokens": max_tokens,
                "temperature": 0,
                "seed": self.seed,
                "stream": False,
                "stop": ["<|im_end|>", "<|endoftext|>"],
            }
            route = "completions"
        else:
            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "max_tokens": max_tokens,
                "temperature": 0,
                "seed": self.seed,
                "stream": False,
            }
            route = "chat/completions"
        request = urllib.request.Request(
            f"{self.base_url}/{route}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                document = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read(1024).decode("utf-8", errors="replace")
            except OSError:
                detail = ""
            raise CliError(
                f"Qwen server returned HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise CliError("Qwen server request failed") from exc
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise CliError("Qwen server returned invalid JSON") from exc
        if not isinstance(document, Mapping):
            raise CliError("Qwen server response root is not an object")
        choices = document.get("choices")
        if not isinstance(choices, list) or not choices:
            raise CliError("Qwen server response has no choices")
        choice = choices[0]
        if not isinstance(choice, Mapping):
            raise CliError("Qwen server returned an invalid first choice")
        if self.prompt_mode == "qwen3.8-no-thinking":
            content = choice.get("text")
        else:
            message = choice.get("message")
            content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise CliError("Qwen server response has no text content")
        if "</think>" in content:
            content = content.split("</think>", 1)[1].strip()
        usage = document.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        return Generation(
            text=content,
            prompt_tokens=_optional_count(usage.get("prompt_tokens")),
            completion_tokens=_optional_count(usage.get("completion_tokens")),
            finish_reason=_optional_text(choice.get("finish_reason")),
        )


def _optional_count(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def _optional_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _optional_nonnegative(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0 else None


def _generation_was_truncated(generation: Generation, max_tokens: int) -> bool:
    if generation.finish_reason in {"length", "max_length", "max_tokens"}:
        return True
    return (
        generation.finish_reason != "stop"
        and generation.completion_tokens is not None
        and generation.completion_tokens >= max_tokens
    )


def _atomic_write_json(path: str | Path, document: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(document, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CliError("result contains non-serializable data") from exc
    except OSError as exc:
        raise CliError(f"cannot prepare result path: {destination}") from exc
    try:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
    except OSError as exc:
        raise CliError(f"cannot create result file: {destination}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except OSError as exc:
        raise CliError(f"cannot write result file: {destination}") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


def _progress(event: str, **fields: Any) -> None:
    sys.stderr.write(json.dumps({"event": event, **fields}, sort_keys=True) + "\n")
    sys.stderr.flush()


def run(
    args: argparse.Namespace,
    *,
    backend_factory: Callable[[argparse.Namespace], GenerationBackend] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[dict[str, Any], Path]:
    items = select_fixed_items(args.benchmark)
    if backend_factory is None:
        if args.backend == "mlx":

            def factory(parsed: argparse.Namespace) -> GenerationBackend:
                return MlxBackend(parsed.model, parsed.revision, seed=parsed.seed)

        elif args.backend == "openai":

            def factory(parsed: argparse.Namespace) -> GenerationBackend:
                return OpenAIBackend(
                    parsed.model,
                    parsed.revision,
                    base_url=parsed.base_url,
                    timeout=parsed.timeout,
                    seed=parsed.seed,
                    prompt_mode=parsed.openai_prompt_mode,
                )

        else:  # pragma: no cover - argparse owns the public boundary
            raise CliError(f"unsupported backend: {args.backend}")
    else:
        factory = backend_factory
    backend = factory(args)

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        _progress("qwen_item_start", item_id=item.item_id, row=index)
        started = clock()
        try:
            generated = backend.generate(item.question, max_tokens=args.max_tokens)
            latency = max(0.0, clock() - started)
            predicted = extract_numeric_answer(generated.text)
            if _generation_was_truncated(generated, args.max_tokens):
                status = "truncated"
                correct = None
            elif predicted is None:
                status = "unparseable"
                correct: bool | None = False
            else:
                status = "correct" if predicted == item.gold else "incorrect"
                correct = status == "correct"
            row = {
                "item_id": item.item_id,
                "question": item.question,
                "gold": item.gold,
                "predicted": predicted,
                "correct": correct,
                "status": status,
                "latency_seconds": latency,
                "prompt_tokens": generated.prompt_tokens,
                "completion_tokens": generated.completion_tokens,
                "finish_reason": generated.finish_reason,
                "peak_memory_gb": generated.peak_memory_gb,
                "text": generated.text,
                "error": None,
            }
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            latency = max(0.0, clock() - started)
            row = {
                "item_id": item.item_id,
                "question": item.question,
                "gold": item.gold,
                "predicted": None,
                "correct": None,
                "status": "error",
                "latency_seconds": latency,
                "prompt_tokens": None,
                "completion_tokens": None,
                "finish_reason": None,
                "peak_memory_gb": None,
                "text": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        _progress(
            "qwen_item_complete",
            item_id=item.item_id,
            row=index,
            status=row["status"],
            seconds=row["latency_seconds"],
        )

    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("correct", "incorrect", "truncated", "unparseable", "error")
    }
    latencies = [float(row["latency_seconds"]) for row in rows]
    parsed = counts["correct"] + counts["incorrect"]
    total = len(rows)
    report = _seal_report({
        "schema": RESULT_SCHEMA,
        "model": {
            "id": backend.model_id,
            "revision": backend.model_revision,
            "path": _public_backend_location(backend.model_path),
            "backend": backend.backend_name,
            "load_seconds": backend.load_seconds,
        },
        "benchmark": {
            "name": "GSM8K historical FERTIG-abstention cohort",
            "source": _public_path(args.benchmark),
            "item_ids": list(FIXED_ITEM_IDS),
        },
        "protocol": {
            "system_prompt": SYSTEM_PROMPT,
            "backend": args.backend,
            "openai_prompt_mode": (
                args.openai_prompt_mode if args.backend == "openai" else None
            ),
            "temperature": 0.0,
            "seed": args.seed,
            "max_tokens": args.max_tokens,
            "answer_extraction": "last numeric value after final #### marker",
        },
        "items": rows,
        "summary": {
            "total": total,
            **counts,
            "accuracy": counts["correct"] / total,
            "parsed_accuracy": counts["correct"] / parsed if parsed else None,
            "total_latency_seconds": sum(latencies),
            "mean_latency_seconds": statistics.fmean(latencies),
            "median_latency_seconds": statistics.median(latencies),
        },
    })
    output = _atomic_write_json(args.output, report)
    return report, output


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report, output = run(args)
    except CliError as exc:
        sys.stderr.write(f"qwen_local_baseline: error: {exc}\n")
        return 2
    summary = {"output": str(output), **report["summary"]}
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
