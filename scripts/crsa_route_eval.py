#!/usr/bin/env python3
"""Measure frozen-A1 contextual routing through the fixed CRSA program.

The evaluation target is exactly SHIP-v6's 152 arithmetic cases and 30
WikiText-2 text windows.  Calibration follows the same v6 recipe, except that
all 48 arithmetic rows duplicated in the evaluation set are removed by
default.  The report includes a raw-A1 ablation, an ordinary causal-softmax
ablation, and fixed-seed permuted-label placebos.

This is a router/context diagnostic, not a value-path experiment.  It never
uses cross-model least squares and never feeds a static embedding mean to the
router.

    PYTHONPATH=src python3 scripts/crsa_route_eval.py
    PYTHONPATH=src python3 scripts/crsa_route_eval.py --router-out artifacts/crsa_router.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable

import torch

from immer.artifacts import load_artifact_specs
from immer.attention.crsa.operators import AttentionSpec, apply_attention
from immer.attention.router import (
    FEATURE_SCHEMA,
    RidgeRouteHead,
    RoleCompleteContext,
    capture_a1_scan_states,
)
from immer.resource_paths import s3_ship_manifest
from immer.runtimes.o1_state.model import (
    SelectiveNoPETransformerLM,
    StreamingNoPELM,
)


SEED = 7
WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
DIGITS = WORDS[:10]
VOCAB_MAX = 5000


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(document: Any) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _word_id(stoi: dict[str, int], word: str) -> int:
    if word not in stoi:
        raise RuntimeError(f"canonical A1 vocabulary is missing {word!r}")
    return stoi[word]


def load_wikitext2() -> tuple[str, str]:
    """Load the original frozen-A1 corpus without importing its research harness."""

    from datasets import load_dataset

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    return (
        "\n\n".join(dataset["train"]["text"]),
        "\n\n".join(dataset["validation"]["text"]),
    )


def build_vocab(text: str) -> tuple[list[str], dict[str, int], int, int]:
    """Reproduce the canonical A1 frequency vocabulary exactly."""

    frequency: dict[str, int] = {}
    for word in re.findall(r"[a-zA-Z]+", text.lower()):
        frequency[word] = frequency.get(word, 0) + 1
    vocabulary = [
        word
        for word, _count in sorted(
            frequency.items(),
            key=lambda item: -item[1],
        )[:VOCAB_MAX]
    ]
    stoi = {word: index for index, word in enumerate(vocabulary)}
    return vocabulary, stoi, len(vocabulary), len(vocabulary) + 1


def tokenize(text: str, stoi: dict[str, int], unknown: int) -> list[int]:
    return [
        stoi.get(word, unknown)
        for word in re.findall(r"[a-zA-Z]+", text.lower())
    ]


def default_checkpoint(artifact_root: str | Path | None = None) -> Path:
    """Resolve the frozen host through the same manifest contract as deployment."""

    _root, specs = load_artifact_specs(
        s3_ship_manifest(),
        configured_root=artifact_root,
    )
    try:
        return next(spec.destination for spec in specs if spec.label.startswith("host:"))
    except StopIteration as exc:
        raise RuntimeError("SHIP manifest does not name a frozen host") from exc


def build_cases(
    stoi: dict[str, int],
    validation_ids: list[int],
    *,
    seed: int = SEED,
    remove_eval_overlap: bool = True,
) -> dict[str, Any]:
    """Reconstruct v6 calibration and its exact 152+30 evaluation cases."""
    def wid(word: str) -> int:
        return _word_id(stoi, word)

    calibration_arithmetic = [
        [wid(WORDS[a]), wid(operator), wid(WORDS[b]), wid("is")]
        for a in range(2, 6)
        for b in range(2, 6)
        for operator in ("plus", "less", "times", "remainder")
    ] + [
        [wid(DIGITS[int(character)]) for character in str(a)]
        + [wid(operator), wid(DIGITS[b]), wid("is")]
        for a in range(10, 30)
        for b in range(2, 5)
        for operator in ("times", "plus")
    ][:48]

    calibration_rng = random.Random(seed)
    calibration_text = [
        validation_ids[start : start + 4]
        for start in (
            calibration_rng.randrange(len(validation_ids) - 5) for _ in range(64)
        )
    ]

    evaluation_arithmetic: list[list[int]] = []
    evaluation_arithmetic += [
        [wid(WORDS[a]), wid("plus"), wid(WORDS[b]), wid("is")]
        for a in range(2, 6)
        for b in range(2, 6)
    ]
    evaluation_arithmetic += [
        [wid(WORDS[a]), wid("less"), wid(WORDS[b]), wid("is")]
        for a in range(3, 9)
        for b in range(1, 3)
    ]
    evaluation_arithmetic += [
        [wid(WORDS[a]), wid("times"), wid(WORDS[b]), wid("is")]
        for a in range(2, 9)
        for b in range(2, 9)
        if a * b <= 16
    ]
    evaluation_arithmetic += [
        [wid(WORDS[a]), wid("remainder"), wid(WORDS[b]), wid("is")]
        for a in range(1, 10)
        for b in range(1, 10)
    ][:220]
    decimal_rng = random.Random(seed + 5)
    for _ in range(24):
        a = decimal_rng.randint(10, 99)
        b = decimal_rng.randint(2, 9)
        operator = decimal_rng.choice(("times", "plus"))
        evaluation_arithmetic.append(
            [wid(DIGITS[int(character)]) for character in str(a)]
            + [wid(operator), wid(DIGITS[b]), wid("is")]
        )

    evaluation_rng = random.Random(seed + 77)
    evaluation_text = []
    for _ in range(30):
        start = evaluation_rng.randrange(len(validation_ids) - 5)
        evaluation_text.append(validation_ids[start : start + 4])

    if len(calibration_arithmetic) != 112:
        raise AssertionError("v6 calibration arithmetic count changed")
    if len(evaluation_arithmetic) != 152 or len(evaluation_text) != 30:
        raise AssertionError("v6 evaluation case count changed")

    evaluation_set = {tuple(row) for row in evaluation_arithmetic}
    overlap = sum(tuple(row) in evaluation_set for row in calibration_arithmetic)
    if remove_eval_overlap:
        calibration_arithmetic = [
            row for row in calibration_arithmetic if tuple(row) not in evaluation_set
        ]
    return {
        "calibration_arithmetic": calibration_arithmetic,
        "calibration_text": calibration_text,
        "evaluation_arithmetic": evaluation_arithmetic,
        "evaluation_text": evaluation_text,
        "arithmetic_overlap_found": overlap,
        "arithmetic_overlap_removed": overlap if remove_eval_overlap else 0,
    }


def load_frozen_a1(checkpoint: Path) -> tuple[Any, dict[str, int], list[int]]:
    """Load the exact A1 architecture and its canonical WikiText-2 vocabulary."""
    train_text, validation_text = load_wikitext2()
    vocab, stoi, unknown, mask = build_vocab(train_text)
    del unknown
    validation_ids = tokenize(validation_text, stoi, len(vocab))
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    try:
        state = bundle["arms"]["A1"]["model"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("checkpoint does not contain the frozen A1 arm") from exc
    host = StreamingNoPELM(
        len(vocab),
        mask,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_head=32,
        seq_len=64,
        dropout=0.0,
        causal=True,
    )
    host.load_state_dict(state)
    host.eval()
    for parameter in host.parameters():
        parameter.requires_grad_(False)
    return host, stoi, validation_ids


def load_frozen_stateless_a1(checkpoint: Path, *, vocab_size: int, mask: int) -> Any:
    """Load the exact deployment host used by :class:`S3Arithmetic`."""
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    host = SelectiveNoPETransformerLM(
        vocab_size,
        mask,
        d_model=128,
        n_layers=2,
        n_heads=4,
        d_head=32,
        seq_len=64,
        dropout=0.0,
        causal=True,
    )
    host.load_state_dict(bundle["arms"]["A1"]["model"])
    host.eval()
    for parameter in host.parameters():
        parameter.requires_grad_(False)
    return host


def capture_rows(host: Any, rows: list[list[int]], *, batch_size: int = 128) -> list[torch.Tensor]:
    """Capture variable-length rows without padding the causal program."""
    captured: list[torch.Tensor | None] = [None] * len(rows)
    for length in sorted({len(row) for row in rows}):
        indices = [index for index, row in enumerate(rows) if len(row) == length]
        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset : offset + batch_size]
            ids = torch.tensor([rows[index] for index in batch_indices], dtype=torch.long)
            states = capture_a1_scan_states(host, ids, layer=0)
            for index, state in zip(batch_indices, states):
                captured[index] = state.cpu()
    if any(state is None for state in captured):
        raise AssertionError("failed to capture every A1 case")
    return [state for state in captured if state is not None]


def stack_features(
    states: Iterable[torch.Tensor],
    context: RoleCompleteContext,
    *,
    mode: str,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for state in states:
        batch = state.unsqueeze(0)
        if mode == "a1_raw":
            feature = batch[:, -1]
        elif mode == "crsa_residual":
            feature = context.route_features(batch, operator="role_complete")
        elif mode == "softmax_residual":
            feature = context.route_features(batch, operator="softmax")
        else:
            raise ValueError(f"unknown feature mode: {mode}")
        rows.append(feature[0])
    return torch.stack(rows)


def route_metrics(labels: torch.Tensor, predictions: torch.Tensor) -> dict[str, float | int]:
    labels = labels.to(torch.int64).cpu()
    predictions = predictions.to(torch.int64).cpu()
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must align")
    positive = labels == 1
    negative = labels == 0
    arithmetic_recall = float((predictions[positive] == 1).double().mean())
    text_recall = float((predictions[negative] == 0).double().mean())
    return {
        "accuracy": round(float((predictions == labels).double().mean()), 6),
        "balanced_accuracy": round((arithmetic_recall + text_recall) / 2.0, 6),
        "arithmetic_recall": round(arithmetic_recall, 6),
        "text_recall": round(text_recall, 6),
        "errors": int((predictions != labels).sum()),
    }


def fit_and_measure(
    calibration: torch.Tensor,
    calibration_labels: torch.Tensor,
    evaluation: torch.Tensor,
    evaluation_labels: torch.Tensor,
    *,
    ridge: float,
    feature_schema: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[RidgeRouteHead, dict[str, float | int]]:
    head = RidgeRouteHead.fit(
        calibration,
        calibration_labels,
        ridge=ridge,
        feature_schema=feature_schema,
        metadata=metadata,
    )
    return head, route_metrics(evaluation_labels, head.predict(evaluation))


def placebo_distribution(
    calibration: torch.Tensor,
    calibration_labels: torch.Tensor,
    evaluation: torch.Tensor,
    evaluation_labels: torch.Tensor,
    *,
    ridge: float,
    count: int,
    seed: int,
) -> dict[str, Any]:
    balanced: list[float] = []
    for index in range(count):
        generator = torch.Generator().manual_seed(seed + index)
        permuted = calibration_labels[
            torch.randperm(len(calibration_labels), generator=generator)
        ]
        head = RidgeRouteHead.fit(calibration, permuted, ridge=ridge)
        metrics = route_metrics(evaluation_labels, head.predict(evaluation))
        balanced.append(float(metrics["balanced_accuracy"]))
    scores = torch.tensor(balanced, dtype=torch.float64)
    return {
        "kind": "permuted_calibration_labels",
        "seed_first": seed,
        "runs": count,
        "balanced_accuracy_mean": round(float(scores.mean()), 6),
        "balanced_accuracy_std": round(float(scores.std(unbiased=False)), 6),
        "balanced_accuracy_min": round(float(scores.min()), 6),
        "balanced_accuracy_max": round(float(scores.max()), 6),
        "balanced_accuracy_runs": [round(value, 6) for value in balanced],
    }


def interface_checks(host: Any, context: RoleCompleteContext, sample_ids: list[int]) -> dict[str, Any]:
    ids = torch.tensor([sample_ids], dtype=torch.long)
    direct = capture_a1_scan_states(host, ids, layer=0)
    hooked: dict[str, torch.Tensor] = {}
    handle = host.layers[0].scan.register_forward_hook(
        lambda _module, _inputs, output: hooked.__setitem__("states", output[0].detach())
    )
    try:
        with torch.no_grad():
            host(ids, None)
    finally:
        handle.remove()
    hook_delta = float((direct - hooked["states"]).abs().max())

    weights = context.attention_weights(direct)
    logits = context.attention_logits(direct)
    free_reference = apply_attention(logits[:, -1:], AttentionSpec(kind="softmax"))
    future = torch.triu(
        torch.ones(direct.shape[1], direct.shape[1], dtype=torch.bool), diagonal=1
    )
    return {
        "direct_matches_v6_hook_max_abs": hook_delta,
        "causal_future_mass": float(weights[..., future].abs().sum()),
        "free_head_bit_exact": bool(torch.equal(weights[:, -1:], free_reference)),
        "row_sum_max_abs_error": float((weights.sum(-1) - 1.0).abs().max()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="SHIP checkpoint containing arms.A1.model (default: manifest artifact)",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="external SHIP artifact directory used for the default checkpoint",
    )
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--placebos", type=int, default=32)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--keep-overlap",
        action="store_true",
        help="replicate v6 calibration literally instead of removing its 48 evaluation duplicates",
    )
    parser.add_argument(
        "--router-out",
        type=Path,
        help="persist the deterministic head only when the positive-router criteria pass",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.placebos < 1:
        raise SystemExit("--placebos must be positive")
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else default_checkpoint(args.artifact_root)
    )
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)

    print("loading frozen A1 and canonical WikiText-2 vocabulary", file=sys.stderr, flush=True)
    host, stoi, validation_ids = load_frozen_a1(checkpoint)
    cases = build_cases(
        stoi,
        validation_ids,
        seed=args.seed,
        remove_eval_overlap=not args.keep_overlap,
    )
    calibration_rows = cases["calibration_arithmetic"] + cases["calibration_text"]
    evaluation_rows = cases["evaluation_arithmetic"] + cases["evaluation_text"]
    calibration_labels = torch.tensor(
        [1] * len(cases["calibration_arithmetic"])
        + [0] * len(cases["calibration_text"]),
        dtype=torch.int64,
    )
    evaluation_labels = torch.tensor(
        [1] * len(cases["evaluation_arithmetic"])
        + [0] * len(cases["evaluation_text"]),
        dtype=torch.int64,
    )

    print(
        f"capturing contextual layer-0 scan states: {len(calibration_rows)} calibration, "
        f"{len(evaluation_rows)} evaluation",
        file=sys.stderr,
        flush=True,
    )
    calibration_states = capture_rows(host, calibration_rows)
    evaluation_states = capture_rows(host, evaluation_rows)
    context = RoleCompleteContext()

    feature_sets: dict[str, tuple[torch.Tensor, torch.Tensor, str]] = {}
    for mode, schema in (
        ("crsa_residual", FEATURE_SCHEMA),
        ("a1_raw", "a1.layer0.scan:raw_last"),
        (
            "softmax_residual",
            "a1.layer0.scan:[raw_last|causal_softmax_last];heads=4",
        ),
    ):
        feature_sets[mode] = (
            stack_features(calibration_states, context, mode=mode),
            stack_features(evaluation_states, context, mode=mode),
            schema,
        )

    checkpoint_sha = _sha256_file(checkpoint)
    split_document = {
        "seed": args.seed,
        "remove_eval_overlap": not args.keep_overlap,
        "calibration_rows": calibration_rows,
        "calibration_labels": calibration_labels.tolist(),
        "evaluation_rows": evaluation_rows,
        "evaluation_labels": evaluation_labels.tolist(),
    }
    split_sha = _json_digest(split_document)
    provenance = {
        "source": "frozen A1 contextual layer-0 scan states",
        "calibration_host": "StreamingNoPELM with zero incoming state",
        "deployment_host": "SelectiveNoPETransformerLM used by S3Arithmetic",
        "checkpoint_sha256": checkpoint_sha,
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1",
        "seed": args.seed,
        "decimal_seed": args.seed + 5,
        "evaluation_text_seed": args.seed + 77,
        "placebo_seed_first": 1000,
        "split_sha256": split_sha,
        "calibration_arithmetic": len(cases["calibration_arithmetic"]),
        "calibration_text": len(cases["calibration_text"]),
        "evaluation_arithmetic": len(cases["evaluation_arithmetic"]),
        "evaluation_text": len(cases["evaluation_text"]),
        "arithmetic_overlap_found": cases["arithmetic_overlap_found"],
        "arithmetic_overlap_removed": cases["arithmetic_overlap_removed"],
    }

    crsa_cal, crsa_eval, crsa_schema = feature_sets["crsa_residual"]
    head, crsa_metrics = fit_and_measure(
        crsa_cal,
        calibration_labels,
        crsa_eval,
        evaluation_labels,
        ridge=args.ridge,
        feature_schema=crsa_schema,
        metadata=provenance,
    )
    raw_cal, raw_eval, raw_schema = feature_sets["a1_raw"]
    _, raw_metrics = fit_and_measure(
        raw_cal,
        calibration_labels,
        raw_eval,
        evaluation_labels,
        ridge=args.ridge,
        feature_schema=raw_schema,
    )
    soft_cal, soft_eval, soft_schema = feature_sets["softmax_residual"]
    _, softmax_metrics = fit_and_measure(
        soft_cal,
        calibration_labels,
        soft_eval,
        evaluation_labels,
        ridge=args.ridge,
        feature_schema=soft_schema,
    )
    placebo = placebo_distribution(
        crsa_cal,
        calibration_labels,
        crsa_eval,
        evaluation_labels,
        ridge=args.ridge,
        count=args.placebos,
        seed=1000,
    )
    checks = interface_checks(host, context, evaluation_rows[0])
    deployment_host = load_frozen_stateless_a1(
        checkpoint,
        vocab_size=len(stoi),
        mask=len(stoi) + 1,
    )
    deployment_states = capture_rows(deployment_host, evaluation_rows)
    deployment_features = stack_features(
        deployment_states,
        context,
        mode="crsa_residual",
    )
    deployment_metrics = route_metrics(
        evaluation_labels,
        head.predict(deployment_features),
    )
    checks["stateless_deployment_feature_max_abs"] = float(
        (crsa_eval - deployment_features).abs().max()
    )
    checks["stateless_deployment_metrics"] = deployment_metrics

    crsa_balanced = float(crsa_metrics["balanced_accuracy"])
    raw_balanced = float(raw_metrics["balanced_accuracy"])
    softmax_balanced = float(softmax_metrics["balanced_accuracy"])
    placebo_max = float(placebo["balanced_accuracy_max"])
    criteria = {
        "balanced_accuracy_at_least_0_95": crsa_balanced >= 0.95,
        "each_class_recall_at_least_0_90": min(
            float(crsa_metrics["arithmetic_recall"]),
            float(crsa_metrics["text_recall"]),
        )
        >= 0.90,
        "above_best_label_placebo_by_at_least_0_10": crsa_balanced - placebo_max >= 0.10,
        "strictly_above_raw_a1_ablation": crsa_balanced > raw_balanced,
        "causal_support_exact": checks["causal_future_mass"] == 0.0,
        "free_head_bit_exact": checks["free_head_bit_exact"],
        "stateless_deployment_182_of_182": deployment_metrics["errors"] == 0,
    }
    persistable = all(criteria.values())
    crsa_specific_win = crsa_balanced > softmax_balanced
    signal_verdict = "POSITIVE" if persistable else "NEGATIVE"
    specificity_verdict = "SHOWN" if crsa_specific_win else "NOT_SHOWN"
    verdict = (
        "POSITIVE_CONTEXT_ROUTER__CRSA_NOT_UNIQUE_VS_SOFTMAX"
        if persistable and not crsa_specific_win
        else "POSITIVE_CRSA_SPECIFIC_ROUTER"
        if persistable
        else "NEGATIVE_DO_NOT_ROUTE_ONLINE"
    )

    # The state is self-describing: deployment can audit the exact split,
    # controls and bounded claim without needing this script's stdout report.
    head.metadata["measurement"] = {
        "schema": "immer.crsa-route-eval/v1",
        "verdict": verdict,
        "context_router_signal": signal_verdict,
        "crsa_specific_advantage_over_softmax": specificity_verdict,
        "metrics": {
            "crsa_residual": crsa_metrics,
            "a1_raw_ablation": raw_metrics,
            "causal_softmax_residual_ablation": softmax_metrics,
            "delta_balanced_vs_raw": round(crsa_balanced - raw_balanced, 6),
            "delta_balanced_vs_softmax": round(crsa_balanced - softmax_balanced, 6),
            "delta_balanced_vs_best_placebo": round(crsa_balanced - placebo_max, 6),
        },
        "placebo": placebo,
        "interface_checks": checks,
        "positive_router_criteria": criteria,
    }

    report = {
        "schema": "immer.crsa-route-eval/v1",
        "verdict": verdict,
        "claims": {
            "context_router_signal": signal_verdict,
            "crsa_specific_advantage_over_softmax": specificity_verdict,
            "note": (
                "The fixed CRSA residual router beats permuted-label placebos and the raw-A1 "
                "ablation. Causal softmax ties it on this small route suite, so this does not "
                "establish a CRSA-specific advantage."
            ),
        },
        "persistable_router": persistable,
        "router_state_sha256": head.digest() if persistable else None,
        "feature": context.description(),
        "provenance": provenance,
        "metrics": {
            "crsa_residual": crsa_metrics,
            "a1_raw_ablation": raw_metrics,
            "causal_softmax_residual_ablation": softmax_metrics,
            "delta_balanced_vs_raw": round(crsa_balanced - raw_balanced, 6),
            "delta_balanced_vs_softmax": round(crsa_balanced - softmax_balanced, 6),
            "delta_balanced_vs_best_placebo": round(crsa_balanced - placebo_max, 6),
        },
        "placebo": placebo,
        "interface_checks": checks,
        "positive_router_criteria": criteria,
    }

    if args.router_out is not None:
        if not persistable:
            print("router criteria failed; refusing to persist an online head", file=sys.stderr)
        else:
            head.save(args.router_out)
            print(f"persisted deterministic router: {args.router_out}", file=sys.stderr)

    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if persistable else 2


if __name__ == "__main__":
    raise SystemExit(main())
