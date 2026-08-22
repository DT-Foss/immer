#!/usr/bin/env python3
"""Reproducible local runner for the range-streamed DeepSeek-V4 runtime.

The runner deliberately stops at execution.  It neither builds nor uploads a
Hugging Face artifact: a remote repository is only a range-addressable tensor
source.  The useful first proof is an isolated position-zero diagnostic, for
which the published compressed/index attention branches have no keys and the
layer stack can be executed with bounded weight residency.  This is explicitly
not stateful prefill/decode or a claim of general text generation.

Examples::

    PYTHONPATH=src python3 scripts/deepseek_v4_poc.py preflight \
      --budget-mb 64 --progress-jsonl results/v4-preflight.jsonl

    PYTHONPATH=src python3 scripts/deepseek_v4_poc.py one-token \
      --token-id 0 --top-k 5 --budget-mb 16384 \
      --progress-jsonl results/v4-one-token.jsonl \
      --output-json results/v4-one-token.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence, TextIO

from immer.knowledge import Streamer
from immer.runtimes.deepseek_v4 import (
    DeepSeekV4Config,
    DeepSeekWeightPager,
    StreamedDeepSeekV4,
)
from immer.runtimes.deepseek_v4.graft import DeepSeekV4CrsaGraft


ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_SOURCE = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
DEFAULT_CACHE = ROOT / "artifacts" / "private" / "deepseek-v4-cache"
RESULT_SCHEMA = "immer.deepseek-v4-poc/v1"
PROGRESS_SCHEMA = "immer.deepseek-v4-poc-progress/v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(document: Any, *, pretty: bool = False) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, document: Any) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(document, pretty=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class ProgressLog:
    """Write one JSON object per event without contaminating result stdout."""

    def __init__(self, destination: str, run_id: str, started: float) -> None:
        self.run_id = run_id
        self.started = started
        self._owns_handle = destination != "-"
        if self._owns_handle:
            path = Path(destination).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.handle: TextIO = path.open("w", encoding="utf-8")
        else:
            self.handle = sys.stderr

    def emit(self, event: str, **fields: Any) -> None:
        record = {
            "schema": PROGRESS_SCHEMA,
            "run_id": self.run_id,
            "event": event,
            "timestamp": _utc_now(),
            "elapsed_seconds": round(time.perf_counter() - self.started, 6),
            **fields,
        }
        self.handle.write(_json_bytes(record).decode("utf-8"))
        self.handle.flush()

    def close(self) -> None:
        if self._owns_handle:
            self.handle.close()

    def __enter__(self) -> "ProgressLog":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _run_command(arguments: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            list(arguments),
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value or None


def _git_metadata() -> dict[str, Any]:
    commit = _run_command(("git", "rev-parse", "HEAD"))
    status = _run_command(("git", "status", "--porcelain", "--untracked-files=normal"))
    return {
        "commit": commit,
        "dirty": status is not None,
        "status": status.splitlines() if status else [],
    }


def _physical_memory_bytes() -> int | None:
    if sys.platform == "darwin":
        value = _run_command(("sysctl", "-n", "hw.memsize"))
        if value is not None and value.isdigit():
            return int(value)
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _hardware_metadata() -> dict[str, Any]:
    try:
        import torch

        torch_version: str | None = str(torch.__version__)
        mps_available: bool | None = bool(torch.backends.mps.is_available())
    except ImportError:
        torch_version = None
        mps_available = None
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "physical_memory_bytes": _physical_memory_bytes(),
        "python": platform.python_version(),
        "torch": torch_version,
        "mps_available": mps_available,
    }


def _local_source_path(value: str) -> Path | None:
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
    local = _local_source_path(args.source)
    common = {
        "revision": args.revision,
        "budget_mb": args.budget_mb,
        "cache_dir": args.cache_dir,
        "use_cache": not args.no_cache,
        "max_cache_bytes": int(args.max_cache_gb * 1024**3),
        "verbose": False,
    }
    if local is not None:
        if not local.is_dir():
            raise FileNotFoundError(f"local source directory does not exist: {local}")
        return Streamer.from_local(local, **common), f"local:{local}"
    return Streamer(args.source, **common), args.source


def _load_config(
    args: argparse.Namespace, source: Streamer
) -> tuple[DeepSeekV4Config, dict[str, Any]]:
    if args.config is not None:
        path = Path(args.config).expanduser().resolve()
        raw = path.read_bytes()
        location = str(path)
    else:
        raw = source.reader.fetch_file("config.json")
        location = "source:config.json"
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid DeepSeek-V4 config JSON at {location}") from exc
    if not isinstance(document, dict):
        raise ValueError("DeepSeek-V4 config root must be an object")
    config = DeepSeekV4Config.from_mapping(document)
    return config, {
        "location": location,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "architecture": document.get("architectures"),
    }


def _arguments_metadata(args: argparse.Namespace) -> dict[str, Any]:
    result = vars(args).copy()
    result.pop("handler", None)
    for key in ("cache_dir", "config", "output_json"):
        if result.get(key) is not None:
            result[key] = str(Path(result[key]).expanduser().resolve())
    return result


def _base_report(
    args: argparse.Namespace, run_id: str, started_at: str
) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "run_id": run_id,
        "mode": args.command,
        "started_at": started_at,
        "arguments": _arguments_metadata(args),
        "git": _git_metadata(),
        "hardware": _hardware_metadata(),
    }


def _provenance(
    args: argparse.Namespace,
    source_label: str,
    source: Streamer,
    config_meta: dict[str, Any],
    config: DeepSeekV4Config,
) -> dict[str, Any]:
    metrics = source.metrics()
    stateful = args.command == "generate"
    return {
        "source": source_label,
        "revision": args.revision,
        "revision_is_pinned": bool(metrics.get("revision_is_pinned")),
        "revision_is_mutable": bool(metrics.get("revision_is_mutable")),
        "config": config_meta,
        "cache_dir": None if args.no_cache else str(Path(args.cache_dir).resolve()),
        "cache_limit_bytes": (
            0 if args.no_cache else int(metrics.get("cache_limit_bytes") or 0)
        ),
        "budget_limit_bytes": int(source.budget.limit),
        "inventory_fingerprint": metrics.get("inventory_source_fingerprint"),
        "decoder": {
            "architecture": "DeepseekV4ForCausalLM",
            "execution_contract": (
                "stateful_autoregressive" if stateful else "isolated_position_zero"
            ),
            "stateful_kv_cache": stateful,
            "layers": config.n_layers,
            "dimension": config.dim,
            "routed_experts": config.n_routed_experts,
            "active_experts": config.n_activated_experts,
            "expert_storage": config.expert_dtype,
        },
    }


def _runtime(
    args: argparse.Namespace,
    progress: ProgressLog,
) -> tuple[
    Streamer,
    DeepSeekV4Config,
    DeepSeekWeightPager,
    StreamedDeepSeekV4,
    dict[str, Any],
    str,
]:
    source, source_label = _build_source(args)
    progress.emit(
        "source_ready",
        source=source_label,
        revision=args.revision,
        budget_mb=args.budget_mb,
    )
    config, config_meta = _load_config(args, source)
    progress.emit(
        "config_validated",
        config_sha256=config_meta["sha256"],
        layers=config.n_layers,
        dimension=config.dim,
    )
    inventory = source.inventory()
    progress.emit(
        "inventory_ready",
        tensors=len(inventory.get("tensors", [])),
        shards=len(inventory.get("shards", [])),
        model_payload_bytes=int(inventory.get("model_payload_bytes", 0)),
        inventory_fingerprint=source.metrics().get("inventory_source_fingerprint"),
    )
    pager = DeepSeekWeightPager(
        source,
        device=args.device,
        compute_dtype=args.dtype,
        simulate_activation_quantization=not args.no_activation_quantization,
    )
    graft_mode = str(getattr(args, "graft_mode", "off"))
    graft = None
    graft_layer = None
    if graft_mode != "off":
        graft = DeepSeekV4CrsaGraft(
            mode=graft_mode,
            alpha=float(args.graft_alpha),
            max_history=int(args.context_limit),
            shuffle_seed=int(args.graft_seed),
        )
        graft_layer = (
            int(args.graft_layer)
            if args.graft_layer is not None
            else config.n_layers // 2
        )
    model = StreamedDeepSeekV4(
        config,
        pager,
        graft=graft,
        graft_layer=graft_layer,
        max_batch_size=1,
        max_seq_len=int(getattr(args, "context_limit", 512)),
    )
    return source, config, pager, model, config_meta, source_label


def _preflight(
    args: argparse.Namespace,
    progress: ProgressLog,
    base: dict[str, Any],
) -> dict[str, Any]:
    source, config, pager, model, config_meta, source_label = _runtime(args, progress)
    checks = model.checkpoint_preflight(exhaustive_experts=not args.sampled_experts)
    progress.emit("preflight_complete", **checks)
    return {
        **base,
        "status": "ok",
        "scope": "deepseek-v4-isolated-position-zero-layer-stack",
        "preflight": checks,
        "provenance": _provenance(args, source_label, source, config_meta, config),
        "pager": pager.metrics(),
        "source_metrics": source.metrics(),
    }


def _flatten_torch(value: Any, cast: Any) -> list[Any]:
    return [cast(item) for item in value.detach().to("cpu").reshape(-1).tolist()]


def _complete_layer_stack(evidence: Any) -> bool:
    """Read the explicit diagnostic-completeness contract from runtime evidence."""

    return bool(evidence.complete_layer_stack)


def _one_token(
    args: argparse.Namespace,
    progress: ProgressLog,
    base: dict[str, Any],
) -> dict[str, Any]:
    source, config, pager, model, config_meta, source_label = _runtime(args, progress)
    checks = model.checkpoint_preflight(exhaustive_experts=not args.sampled_experts)
    progress.emit("preflight_complete", **checks)

    def layer_progress(event: dict[str, Any]) -> None:
        progress.emit("layer_complete", **event)

    def head_progress(event: dict[str, int]) -> None:
        progress.emit("head_block_complete", **event)

    progress.emit(
        "decoder_start",
        token_id=args.token_id,
        stop_after_layer=args.stop_after_layer,
    )
    if args.stop_after_layer is None:
        values, token_ids, evidence = model.next_token_from_one(
            args.token_id,
            topk=args.top_k,
            head_block_rows=args.head_block_rows,
            progress=layer_progress,
            head_progress=head_progress,
        )
    else:
        hidden, evidence = model.hidden_one_token(
            args.token_id,
            stop_after_layer=args.stop_after_layer,
            progress=layer_progress,
        )
        if _complete_layer_stack(evidence):
            values, token_ids = pager.topk_logits(
                hidden[:, -1],
                k=args.top_k,
                block_rows=args.head_block_rows,
                progress=head_progress,
            )
        else:
            values = token_ids = None

    logits = None
    if values is not None and token_ids is not None:
        logits = {
            "top_k": args.top_k,
            "head_block_rows": args.head_block_rows,
            "token_ids": _flatten_torch(token_ids, int),
            "values": _flatten_torch(values, float),
        }
    evidence_dict = model.evidence_dict(evidence)
    progress.emit(
        "decoder_complete",
        complete_layer_stack=_complete_layer_stack(evidence),
        context_mode=evidence.context_mode,
        stateful_kv_cache=bool(evidence.stateful_kv_cache),
        layers_executed=evidence.layers_executed,
        source_body_bytes=evidence.source_body_bytes,
        seconds=evidence.seconds,
    )
    return {
        **base,
        "status": "ok",
        "scope": "deepseek-v4-isolated-position-zero-layer-stack",
        "exactness": {
            "complete_layer_stack": _complete_layer_stack(evidence),
            "context_mode": evidence.context_mode,
            "stateful_kv_cache": bool(evidence.stateful_kv_cache),
            "position": 0,
            "compressed_and_index_key_sets": "empty",
            "dspark_speculative_module": "not_executed",
            "general_generation": False,
        },
        "preflight": checks,
        "one_token": {
            "evidence": evidence_dict,
            "logits": logits,
        },
        "provenance": _provenance(args, source_label, source, config_meta, config),
        "pager": pager.metrics(),
        "source_metrics": source.metrics(),
    }


def _parse_token_ids(value: str) -> list[int]:
    try:
        ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--token-ids must be comma-separated integers") from exc
    if not ids:
        raise ValueError("--token-ids must contain at least one integer")
    return ids


def _tokenize_prompt(
    source: Streamer, args: argparse.Namespace
) -> tuple[list[int], dict[str, Any]]:
    if args.token_ids is not None:
        return _parse_token_ids(args.token_ids), {
            "mode": "explicit_token_ids",
            "tokenizer_sha256": None,
            "encoded_prompt": None,
        }
    if args.prompt is None:
        raise ValueError("generate requires either --prompt or --token-ids")
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("text prompts require the tokenizers package") from exc
    from immer.runtimes.deepseek_v4.encoding import encode_user_prompt

    raw = source.reader.fetch_file("tokenizer.json")
    tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    encoded = encode_user_prompt(
        args.prompt,
        thinking_mode=args.thinking_mode,
        reasoning_effort=args.reasoning_effort,
    )
    token_ids = tokenizer.encode(encoded, add_special_tokens=False).ids
    if not token_ids:
        raise RuntimeError("official prompt encoding produced no tokens")
    return token_ids, {
        "mode": "official_single_user_message",
        "tokenizer_sha256": hashlib.sha256(raw).hexdigest(),
        "encoded_prompt": encoded,
        "thinking_mode": args.thinking_mode,
        "reasoning_effort": args.reasoning_effort,
    }


def _decode_tokens(
    source: Streamer, token_ids: Sequence[int]
) -> tuple[str | None, str | None]:
    try:
        from tokenizers import Tokenizer
    except ImportError:
        return None, None
    try:
        raw = source.reader.fetch_file("tokenizer.json")
    except Exception:  # explicit IDs may use a tensor-only local fixture
        return None, None
    tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
    return (
        tokenizer.decode(list(token_ids), skip_special_tokens=False),
        hashlib.sha256(raw).hexdigest(),
    )


def _exact_cascade(prompt: str) -> dict[str, Any]:
    from immer.composition import compose_runtime

    result = compose_runtime().dispatch("exact_math", prompt)
    return {
        "status": result.status.value,
        "component": result.component,
        "output": result.output,
        "reason": result.reason,
        "evidence": dict(result.evidence),
    }


def _generate(
    args: argparse.Namespace,
    progress: ProgressLog,
    base: dict[str, Any],
) -> dict[str, Any]:
    if args.prompt is not None and args.token_ids is not None:
        raise ValueError("pass either --prompt or --token-ids, not both")
    source, config, pager, model, config_meta, source_label = _runtime(args, progress)
    checks = model.checkpoint_preflight(exhaustive_experts=not args.sampled_experts)
    progress.emit("preflight_complete", **checks)
    prompt_ids, encoding = _tokenize_prompt(source, args)
    if len(prompt_ids) + args.max_new_tokens > args.context_limit:
        raise ValueError(
            "prompt plus requested output exceeds the local --context-limit"
        )

    def model_progress(event: dict[str, Any]) -> None:
        payload = dict(event)
        event_name = str(payload.pop("event", "layer_complete"))
        progress.emit(event_name, **payload)

    def head_progress(event: dict[str, int]) -> None:
        progress.emit("head_block_complete", **event)

    progress.emit(
        "generation_start",
        prompt_tokens=len(prompt_ids),
        max_new_tokens=args.max_new_tokens,
        prefill_mode=args.prefill_mode,
        graft_mode=args.graft_mode,
        graft_layer=(
            args.graft_layer
            if args.graft_layer is not None
            else (config.n_layers // 2 if args.graft_mode != "off" else None)
        ),
    )
    generated, evidence = model.generate_greedy(
        [prompt_ids],
        max_new_tokens=args.max_new_tokens,
        prefill_tokenwise=args.prefill_mode == "tokenwise",
        eos_token_ids=(args.eos_token_id,),
        head_block_rows=args.head_block_rows,
        progress=model_progress,
        head_progress=head_progress,
    )
    completion, decode_sha = _decode_tokens(source, generated)
    cascade = None
    if args.exact_cascade:
        if args.prompt is None:
            raise ValueError("--exact-cascade requires a textual --prompt")
        cascade = _exact_cascade(args.prompt)
    evidence_dict = model.generation_evidence_dict(evidence)
    progress.emit(
        "generation_complete",
        generated_tokens=len(generated),
        stopped_on_eos=evidence.stopped_on_eos,
        source_body_bytes=evidence.source_body_bytes,
        seconds=evidence.seconds,
    )
    return {
        **base,
        "status": "ok",
        "scope": "deepseek-v4-stateful-autoregressive-main-decoder",
        "exactness": {
            "complete_layer_stack": True,
            "context_mode": evidence.context_mode,
            "stateful_kv_cache": True,
            "native_sparse_attention": True,
            "general_generation": True,
            "dspark_speculative_module": "not_executed",
            "sampling": "global_greedy_argmax",
            "prefill_mode": evidence.prefill_mode,
        },
        "preflight": checks,
        "generation": {
            "encoding": encoding,
            "prompt_token_ids": prompt_ids,
            "generated_token_ids": list(generated),
            "completion": completion,
            "decode_tokenizer_sha256": decode_sha,
            "evidence": evidence_dict,
            "graft": {
                "mode": args.graft_mode,
                "alpha": args.graft_alpha,
                "layer": (
                    args.graft_layer
                    if args.graft_layer is not None
                    else (config.n_layers // 2 if args.graft_mode != "off" else None)
                ),
            },
            "exact_cascade": cascade,
        },
        "provenance": _provenance(args, source_label, source, config_meta, config),
        "pager": pager.metrics(),
        "source_metrics": source.metrics(),
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not parsed > 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative number")
    return parsed


def _common_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--source",
        default=OFFICIAL_SOURCE,
        help="HF repository ID, local directory, or local:/absolute/path",
    )
    common.add_argument(
        "--revision",
        default=OFFICIAL_REVISION,
        help="immutable source commit (use an explicit local fixture label locally)",
    )
    common.add_argument(
        "--config",
        help="config.json path; default reads config.json through the tensor source",
    )
    common.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE),
        help="verified resumable range cache",
    )
    common.add_argument(
        "--no-cache",
        action="store_true",
        help="disable inventory and range cache writes/reuse",
    )
    common.add_argument(
        "--max-cache-gb",
        type=_positive_float,
        default=12.0,
        help="hard verified-cache disk cap with LRU eviction (default: 12 GiB)",
    )
    common.add_argument(
        "--budget-mb",
        type=_positive_float,
        default=16384.0,
        help="hard source-transfer budget in MiB (cache hits cost zero)",
    )
    common.add_argument(
        "--device",
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="torch execution device",
    )
    common.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="auto",
        help="activation/matmul compute dtype",
    )
    common.add_argument(
        "--no-activation-quantization",
        action="store_true",
        help="diagnostic ablation: skip simulated published activation FP8 Q/DQ",
    )
    common.add_argument(
        "--sampled-experts",
        action="store_true",
        help="preflight only expert 0 per layer instead of every routed expert",
    )
    common.add_argument(
        "--progress-jsonl",
        default="-",
        help="progress JSONL path; '-' writes progress to stderr",
    )
    common.add_argument(
        "--output-json",
        help="optional atomic copy of the final JSON report (stdout is always emitted)",
    )
    common.add_argument(
        "--debug",
        action="store_true",
        help="include a Python traceback in a failed JSON report",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight or execute the local range-streamed DeepSeek-V4 PoC.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()
    preflight = subparsers.add_parser(
        "preflight",
        parents=[common],
        help="validate the isolated position-zero tensor contract without payload reads",
    )
    preflight.set_defaults(handler=_preflight)
    one = subparsers.add_parser(
        "one-token",
        parents=[common],
        help="execute an isolated token and optionally scan the complete LM head",
    )
    one.add_argument("--token-id", type=_nonnegative_int, default=0)
    one.add_argument("--top-k", type=_positive_int, default=5)
    one.add_argument("--head-block-rows", type=_positive_int, default=1024)
    one.add_argument(
        "--stop-after-layer",
        type=_positive_int,
        help="diagnostic decoder prefix; incomplete prefixes intentionally return no logits",
    )
    one.set_defaults(handler=_one_token)
    generate = subparsers.add_parser(
        "generate",
        parents=[common],
        help="run native stateful prefill and exact greedy autoregressive decode",
    )
    generate.add_argument("--prompt", help="single user message in official V4 format")
    generate.add_argument(
        "--token-ids",
        help="comma-separated prompt IDs (diagnostic alternative to --prompt)",
    )
    generate.add_argument(
        "--thinking-mode", choices=("chat", "thinking"), default="chat"
    )
    generate.add_argument(
        "--reasoning-effort", choices=("low", "high", "max"), default="low"
    )
    generate.add_argument("--max-new-tokens", type=_positive_int, default=1)
    generate.add_argument(
        "--prefill-mode",
        choices=("batched", "tokenwise"),
        default="batched",
        help="batched is the native causal prefill; tokenwise is the slower parity reference",
    )
    generate.add_argument("--context-limit", type=_positive_int, default=256)
    generate.add_argument("--head-block-rows", type=_positive_int, default=1024)
    generate.add_argument("--eos-token-id", type=_nonnegative_int, default=1)
    generate.add_argument(
        "--graft-mode",
        choices=("off", "crsa", "softmax", "shuffle"),
        default="off",
    )
    generate.add_argument("--graft-alpha", type=_nonnegative_float, default=0.05)
    generate.add_argument("--graft-layer", type=_nonnegative_int)
    generate.add_argument("--graft-seed", type=_nonnegative_int, default=17)
    generate.add_argument(
        "--exact-cascade",
        action="store_true",
        help="also run the existing guarded S3-to-FERTIG exact path on the prompt",
    )
    generate.set_defaults(handler=_generate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    started_at = _utc_now()
    run_id = uuid.uuid4().hex
    base = _base_report(args, run_id, started_at)
    exit_code = 0
    progress: ProgressLog | None = None
    try:
        progress = ProgressLog(args.progress_jsonl, run_id, started)
        progress.emit("run_start", mode=args.command)
        report = args.handler(args, progress, base)
        report["finished_at"] = _utc_now()
        report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        progress.emit("run_complete", status="ok")
    except KeyboardInterrupt as exc:
        exit_code = 130
        report = {
            **base,
            "status": "error",
            "error": {"type": type(exc).__name__, "message": "interrupted"},
            "finished_at": _utc_now(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
        if progress is not None:
            progress.emit("run_failed", **report["error"])
    except Exception as exc:  # noqa: BLE001 - CLI must serialize hard failures
        exit_code = 1
        error: dict[str, Any] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        if args.debug:
            error["traceback"] = traceback.format_exc()
        report = {
            **base,
            "status": "error",
            "error": error,
            "finished_at": _utc_now(),
            "elapsed_seconds": round(time.perf_counter() - started, 6),
        }
        if progress is not None:
            progress.emit("run_failed", **error)
    finally:
        if progress is not None:
            progress.close()

    if args.output_json:
        try:
            _atomic_write_json(Path(args.output_json), report)
        except Exception as exc:  # noqa: BLE001 - result persistence is part of success
            exit_code = 1
            report = {
                **report,
                "status": "error",
                "output_error": {"type": type(exc).__name__, "message": str(exc)},
            }
    sys.stdout.buffer.write(_json_bytes(report, pretty=True))
    sys.stdout.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
