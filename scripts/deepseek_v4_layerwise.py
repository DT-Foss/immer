#!/usr/bin/env python3
"""Run a fixed MMLU split through the out-of-core V4 layer-major engine.

This command executes and scores locally.  It neither packages nor uploads a
model.  ``--dry-run`` performs metadata/tokenizer/dataset/disk admission and
prints the exact activation/source budget before creating the run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow.parquet as parquet

from immer.knowledge import Streamer
from immer.runtimes.deepseek_v4 import (
    DeepSeekRuntimeSource,
    DeepSeekRuntimeSourceError,
    DeepSeekV4Config,
    DeepSeekWeightPager,
    LayerwiseItem,
    LayerwiseScorer,
    StreamedDeepSeekV4,
    open_deepseek_runtime_source,
    shareable_runtime_evidence,
)
from immer.runtimes.deepseek_v4.encoding import encode_user_prompt


ROOT = Path(__file__).resolve().parent.parent
OFFICIAL_SOURCE = "deepseek-ai/DeepSeek-V4-Flash-0731"
OFFICIAL_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
DEFAULT_DATASET = ROOT / "evals" / "mmlu_high_school_geography_test.parquet"
DEFAULT_TOKENIZER = (
    ROOT / "artifacts" / "private" / "deepseek-v4-reference" / "tokenizer.json"
)
DEFAULT_RUN_DIR = ROOT / "artifacts" / "private" / "deepseek-v4-layerwise"
DEFAULT_CACHE = ROOT / "artifacts" / "private" / "deepseek-v4-cache"
ASSISTANT_PREFIX = "Answer:"


class CliError(RuntimeError):
    """The requested layerwise run is not reproducible or executable."""


class LocalTokenizer:
    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise CliError("local tokenizer execution requires tokenizers") from exc
        try:
            self.raw = self.path.read_bytes()
            self.backend = Tokenizer.from_str(self.raw.decode("utf-8"))
        except Exception as exc:
            raise CliError(f"cannot load tokenizer JSON: {self.path}") from exc

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.raw).hexdigest()

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(int(value) for value in self.backend.encode(text).ids)


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--tokenizer-json", default=str(DEFAULT_TOKENIZER))
    parser.add_argument("--source", default=OFFICIAL_SOURCE)
    parser.add_argument(
        "--causal-bundle",
        help="local bundle with authenticated complete dense-weight coverage",
    )
    parser.add_argument("--logical-repo-id", default=OFFICIAL_SOURCE)
    parser.add_argument("--revision", default=OFFICIAL_REVISION)
    parser.add_argument("--config")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--source-budget-mb", type=_positive_int, default=196608)
    parser.add_argument("--cache-budget-gb", type=_nonnegative_float, default=12.0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16"), default="auto")
    parser.add_argument("--no-activation-quantization", action="store_true")
    parser.add_argument("--no-expert-prefetch", action="store_true")
    parser.add_argument("--microbatch-size", type=_positive_int, default=32)
    parser.add_argument(
        "--padding", choices=("right", "exact-length"), default="right"
    )
    parser.add_argument(
        "--modes", default="off,crsa,softmax,shuffle", help="comma-separated ablations"
    )
    parser.add_argument("--graft-layer", type=int)
    parser.add_argument("--graft-alpha", type=_nonnegative_float, default=0.05)
    parser.add_argument("--graft-seed", type=int, default=17)
    parser.add_argument("--max-prompt-tokens", type=_positive_int, default=256)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--thinking-mode", choices=("chat", "thinking"), default="chat")
    parser.add_argument(
        "--reasoning-effort", choices=("low", "high", "max"), default="low"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-full-source-disk", action="store_true")
    parser.add_argument("--disk-margin-gb", type=_nonnegative_float, default=1.0)
    parser.add_argument(
        "--preflight", choices=("none", "representative", "exhaustive"), default="none"
    )
    return parser


def _read_rows(path: Path) -> list[dict[str, Any]]:
    source = path.expanduser().resolve()
    if source.suffix.lower() == ".parquet":
        try:
            values = parquet.read_table(source).to_pylist()
        except Exception as exc:
            raise CliError(f"cannot read MMLU Parquet: {source}") from exc
    else:
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CliError(f"cannot read MMLU JSON: {source}") from exc
        values = document.get("items") if isinstance(document, dict) else document
    if not isinstance(values, list) or not values:
        raise CliError("MMLU dataset must contain a non-empty item list")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise CliError(f"MMLU row {index} is not an object")
        row = dict(value)
        item_id = str(row.get("id", row.get("item_id", f"row-{index:06d}"))).strip()
        if not item_id or item_id in seen:
            raise CliError(f"MMLU row {index} has an empty or duplicate ID")
        seen.add(item_id)
        row["id"] = item_id
        rows.append(row)
    return rows


def _source(args: argparse.Namespace) -> DeepSeekRuntimeSource:
    try:
        return open_deepseek_runtime_source(
            source=args.source,
            revision=args.revision,
            logical_repo_id=args.logical_repo_id,
            causal_bundle=args.causal_bundle,
            budget_mb=args.source_budget_mb,
            cache_dir=args.cache_dir,
            use_cache=not args.no_cache,
            max_cache_bytes=int(args.cache_budget_gb * 1024**3),
            require_remote_pinned_revision=True,
        )
    except (DeepSeekRuntimeSourceError, FileNotFoundError) as exc:
        raise CliError(str(exc)) from exc


def _config(args: argparse.Namespace, source: Streamer) -> tuple[DeepSeekV4Config, str]:
    try:
        raw = (
            Path(args.config).expanduser().resolve().read_bytes()
            if args.config
            else source.reader.fetch_file("config.json")
        )
        document = json.loads(raw)
    except Exception as exc:
        raise CliError("cannot load DeepSeek-V4 config JSON") from exc
    if not isinstance(document, Mapping):
        raise CliError("DeepSeek-V4 config root must be an object")
    return DeepSeekV4Config.from_mapping(document), hashlib.sha256(raw).hexdigest()


def _contextual_candidate(
    tokenizer: LocalTokenizer,
    rendered: str,
    prompt_ids: tuple[int, ...],
    surface: str,
    item_id: str,
) -> int:
    combined = tokenizer.encode(rendered + surface)
    if combined[: len(prompt_ids)] != prompt_ids or len(combined) != len(prompt_ids) + 1:
        raise CliError(
            f"MMLU item {item_id}: candidate {surface!r} is not an exact one-token suffix"
        )
    return combined[-1]


def _items(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: LocalTokenizer,
    config: DeepSeekV4Config,
    args: argparse.Namespace,
) -> tuple[LayerwiseItem, ...]:
    result: list[LayerwiseItem] = []
    for row in rows:
        item_id = str(row["id"])
        question = row.get("question")
        choices = row.get("choices")
        answer = row.get("answer")
        if not isinstance(question, str) or not question.strip():
            raise CliError(f"MMLU item {item_id} has no question")
        if not isinstance(choices, list) or not 2 <= len(choices) <= 26:
            raise CliError(f"MMLU item {item_id} has invalid choices")
        if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(choices):
            raise CliError(f"MMLU item {item_id} has an invalid answer")
        choice_text = "\n".join(
            f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices)
        )
        user = f"{question.strip()}\n{choice_text}"
        rendered = (
            encode_user_prompt(
                user,
                thinking_mode=args.thinking_mode,
                reasoning_effort=args.reasoning_effort,
            )
            + ASSISTANT_PREFIX
        )
        prompt_ids = tokenizer.encode(rendered)
        if not prompt_ids or len(prompt_ids) > args.max_prompt_tokens:
            raise CliError(
                f"MMLU item {item_id} prompt length {len(prompt_ids)} is outside bound"
            )
        surfaces = tuple(f" {chr(65 + index)}" for index in range(len(choices)))
        candidate_ids = tuple(
            _contextual_candidate(tokenizer, rendered, prompt_ids, surface, item_id)
            for surface in surfaces
        )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise CliError(f"MMLU item {item_id} candidates tokenize to duplicate IDs")
        if any(token >= config.vocab_size for token in (*prompt_ids, *candidate_ids)):
            raise CliError(f"MMLU item {item_id} has token outside model vocabulary")
        result.append(
            LayerwiseItem(
                item_id=item_id,
                prompt_token_ids=prompt_ids,
                candidate_token_ids=candidate_ids,
                candidate_values=tuple(range(len(choices))),
                expected=answer,
            )
        )
    return tuple(result)


def _modes(raw: str) -> tuple[str, ...]:
    result = tuple(value.strip().lower() for value in raw.split(",") if value.strip())
    if not result or len(result) != len(set(result)):
        raise CliError("--modes must contain distinct mode names")
    return result


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _accuracy(result: Mapping[str, Any]) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}
    modes = result.get("modes", {})
    if not isinstance(modes, Mapping):
        return summaries
    for name, mode in modes.items():
        if not isinstance(name, str) or not isinstance(mode, Mapping):
            continue
        rows = mode.get("items", [])
        if not isinstance(rows, list):
            continue
        correct = sum(row.get("predicted") == row.get("expected") for row in rows)
        summaries[name] = {
            "total": len(rows),
            "correct": correct,
            "accuracy": correct / len(rows) if rows else 0.0,
        }
    return summaries


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = Path(args.dataset).expanduser().resolve()
    rows = _read_rows(dataset)
    if args.limit is not None:
        rows = rows[: args.limit]
    tokenizer = LocalTokenizer(Path(args.tokenizer_json))
    runtime_source = _source(args)
    try:
        return _run_with_source(
            args,
            dataset=dataset,
            rows=rows,
            tokenizer=tokenizer,
            source=runtime_source.source,
            source_label=runtime_source.label,
            causal_weight_reader=runtime_source.causal_weight_reader,
            causal_tensor_reader=runtime_source.causal_tensor_reader,
        )
    finally:
        runtime_source.close()


def _run_with_source(
    args: argparse.Namespace,
    *,
    dataset: Path,
    rows: list[dict[str, Any]],
    tokenizer: LocalTokenizer,
    source: Streamer,
    source_label: str,
    causal_weight_reader: Any | None = None,
    causal_tensor_reader: Any | None = None,
) -> dict[str, Any]:
    config, config_sha256 = _config(args, source)
    items = _items(rows, tokenizer, config, args)
    pager = DeepSeekWeightPager(
        source,
        device=args.device,
        compute_dtype=args.dtype,
        simulate_activation_quantization=not args.no_activation_quantization,
        expert_prefetch=not args.no_expert_prefetch,
        causal_weight_reader=causal_weight_reader,
        causal_tensor_reader=causal_tensor_reader,
        causal_missing_fallback=False,
    )
    try:
        model = StreamedDeepSeekV4(
            config,
            pager,
            max_batch_size=args.microbatch_size,
            max_seq_len=args.max_prompt_tokens,
        )
        if args.preflight != "none":
            model.checkpoint_preflight(
                exhaustive_experts=args.preflight == "exhaustive"
            )
        scorer = LayerwiseScorer(
            model,
            items,
            run_dir=args.run_dir,
            modes=_modes(args.modes),
            microbatch_size=args.microbatch_size,
            padding=args.padding,
            graft_layer=args.graft_layer,
            graft_alpha=args.graft_alpha,
            graft_seed=args.graft_seed,
            tokenizer_sha256=tokenizer.sha256,
            dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
            config_sha256=config_sha256,
            require_full_source_disk=args.require_full_source_disk,
            source_cache_reserve_bytes=(
                0
                if args.causal_bundle is not None
                else int(args.cache_budget_gb * 1024**3)
            ),
            disk_margin_bytes=int(args.disk_margin_gb * 1024**3),
        )
    except BaseException:
        pager.release()
        raise
    source_transfer_budget = int(args.source_budget_mb * 1024**2)
    if source_transfer_budget < scorer.plan.official_source_safe_bytes:
        pager.release()
        raise CliError(
            "source transfer budget is below the checkpoint safe cap: "
            f"{source_transfer_budget} < {scorer.plan.official_source_safe_bytes} bytes"
        )
    if args.dry_run:
        receipt = {
            "status": "planned",
            "source": source_label,
            "source_transfer_budget_bytes": source_transfer_budget,
            "run_dir": shareable_runtime_evidence(
                str(Path(args.run_dir).expanduser().absolute())
            ),
            **scorer.plan_dict(),
        }
        pager.release()
        return receipt

    def progress(row: Mapping[str, Any]) -> None:
        if row.get("event") == "layer_complete":
            sys.stderr.write(json.dumps(dict(row), sort_keys=True) + "\n")
            sys.stderr.flush()

    try:
        result = scorer.run(resume=args.resume, progress=progress)
    finally:
        pager.release()
    return {
        "status": "complete",
        "source": source_label,
        "run_dir": shareable_runtime_evidence(
            str(Path(args.run_dir).expanduser().absolute())
        ),
        "result_body_sha256": _canonical_digest(result),
        "summaries": _accuracy(result),
        "result": result,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = run(args)
    except Exception as exc:
        sys.stderr.write(
            json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True)
            + "\n"
        )
        return 2
    sys.stdout.write(json.dumps(receipt, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
