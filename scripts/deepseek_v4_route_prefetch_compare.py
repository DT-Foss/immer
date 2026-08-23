#!/usr/bin/env python3
"""Compare bit-identical demand-only, real-Markov, and placebo prefetch runs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

RESULT_SCHEMA = "immer.deepseek-v4-fertig-draft-verification/v1"
REPORT_SCHEMA = "immer.deepseek-v4-route-prefetch-comparison/v1"


class CompareError(RuntimeError):
    """The three runtime arms do not form one valid controlled comparison."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CompareError("comparison value is not canonical JSON") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()

    def pairs(entries: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in entries:
            if key in result:
                raise CompareError(f"duplicate JSON key in {source}: {key!r}")
            result[key] = value
        return result

    try:
        document = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=pairs
        )
    except CompareError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CompareError(f"cannot read runtime result: {source}") from exc
    if not isinstance(document, dict):
        raise CompareError(f"runtime result root is not an object: {source}")
    if document.get("schema") != RESULT_SCHEMA or document.get("status") != "complete":
        raise CompareError(f"runtime result is not complete: {source}")
    return document


def _invariant_view(document: Mapping[str, Any]) -> dict[str, Any]:
    evidence = document.get("evidence")
    traffic = document.get("traffic")
    if not isinstance(evidence, Mapping) or not isinstance(traffic, Mapping):
        raise CompareError("runtime result lacks evidence or traffic")
    local = traffic.get("local")
    if not isinstance(local, Mapping):
        raise CompareError("runtime result lacks local traffic evidence")
    return {
        "evidence": {
            key: value
            for key, value in evidence.items()
            if key not in {"seconds", "source_body_bytes"}
        },
        "items": document.get("items"),
        "protocol": document.get("protocol"),
        "schema": document.get("schema"),
        "status": document.get("status"),
        "summary": document.get("summary"),
        "traffic": {
            "api": traffic.get("api"),
            "local": {
                key: value for key, value in local.items() if key != "source_body_bytes"
            },
        },
    }


def _finite_number(value: object, label: str, *, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise CompareError(f"{label} must be finite and >= {minimum}")
    return float(value)


def _counter(metrics: Mapping[str, Any], name: str) -> int:
    value = metrics.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CompareError(f"pager metric {name} must be a non-negative integer")
    return value


def _arm_record(
    document: Mapping[str, Any],
    *,
    expected_enabled: bool,
    expected_role: str | None,
) -> dict[str, Any]:
    instrumentation = document.get("instrumentation")
    route = (
        instrumentation.get("route_prefetch")
        if isinstance(instrumentation, Mapping)
        else None
    )
    legacy_baseline = route is None and not expected_enabled
    if legacy_baseline:
        route = {
            "enabled": False,
            "model_role": None,
            "model_snapshot_sha256": None,
            "pager": {},
        }
    if not isinstance(route, Mapping) or route.get("enabled") is not expected_enabled:
        raise CompareError("route-prefetch arm enablement does not match")
    if route.get("model_role") != expected_role:
        raise CompareError("route-prefetch arm role does not match")
    pager = route.get("pager")
    if not isinstance(pager, Mapping):
        raise CompareError("route-prefetch arm lacks pager counters")
    evidence = document["evidence"]
    seconds = _finite_number(evidence.get("seconds"), "runtime seconds")
    source_bytes = int(
        _finite_number(evidence.get("source_body_bytes"), "source bytes")
    )
    submitted = _counter(pager, "expert_reservoir_submitted")
    prediction_hits = _counter(pager, "expert_reservoir_prediction_hits")
    usable_hits = _counter(pager, "expert_reservoir_usable_hits")
    misses = _counter(pager, "expert_reservoir_misses")
    wasted = _counter(pager, "expert_reservoir_wasted")
    if usable_hits > prediction_hits or prediction_hits > submitted:
        raise CompareError("route reservoir hit counters are inconsistent")
    return {
        "legacy_baseline_instrumentation": legacy_baseline,
        "model_role": expected_role,
        "model_snapshot_sha256": route.get("model_snapshot_sha256"),
        "seconds": seconds,
        "source_body_bytes": source_bytes,
        "pager": dict(pager),
        "reservoir": {
            "submitted": submitted,
            "prediction_hits": prediction_hits,
            "usable_hits": usable_hits,
            "misses": misses,
            "wasted": wasted,
            "prediction_precision": (
                prediction_hits / submitted if submitted else None
            ),
            "usable_precision": usable_hits / submitted if submitted else None,
            "demand_coverage": (
                usable_hits / (usable_hits + misses) if usable_hits + misses else None
            ),
        },
    }


def build_report(
    baseline_path: str | os.PathLike[str],
    real_path: str | os.PathLike[str],
    placebo_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Prove output identity, then compare only transport and timing evidence."""

    documents = {
        "baseline": _read(baseline_path),
        "real_markov": _read(real_path),
        "placebo_markov": _read(placebo_path),
    }
    invariant = _invariant_view(documents["baseline"])
    for name in ("real_markov", "placebo_markov"):
        if _invariant_view(documents[name]) != invariant:
            raise CompareError(f"{name} changed model outputs or execution math")
    arms = {
        "baseline": _arm_record(
            documents["baseline"], expected_enabled=False, expected_role=None
        ),
        "real_markov": _arm_record(
            documents["real_markov"],
            expected_enabled=True,
            expected_role="real_markov",
        ),
        "placebo_markov": _arm_record(
            documents["placebo_markov"],
            expected_enabled=True,
            expected_role="placebo_markov",
        ),
    }
    baseline = arms["baseline"]

    def contrast(arm: Mapping[str, Any]) -> dict[str, float | int | None]:
        seconds = float(arm["seconds"])
        source_bytes = int(arm["source_body_bytes"])
        baseline_seconds = float(baseline["seconds"])
        baseline_bytes = int(baseline["source_body_bytes"])
        return {
            "seconds_delta": seconds - baseline_seconds,
            "seconds_ratio": seconds / baseline_seconds if baseline_seconds else None,
            "source_body_bytes_delta": source_bytes - baseline_bytes,
            "source_body_bytes_ratio": (
                source_bytes / baseline_bytes if baseline_bytes else None
            ),
        }

    identity = {
        "arms": arms,
        "contrasts": {
            "placebo_vs_baseline": contrast(arms["placebo_markov"]),
            "real_vs_baseline": contrast(arms["real_markov"]),
            "real_vs_placebo": {
                "seconds_delta": (
                    float(arms["real_markov"]["seconds"])
                    - float(arms["placebo_markov"]["seconds"])
                ),
                "source_body_bytes_delta": (
                    int(arms["real_markov"]["source_body_bytes"])
                    - int(arms["placebo_markov"]["source_body_bytes"])
                ),
            },
        },
        "invariant_sha256": _sha256(invariant),
        "schema": REPORT_SCHEMA,
    }
    return {**identity, "sha256": _sha256(identity)}


def write_report(path: str | os.PathLike[str], report: Mapping[str, Any]) -> None:
    identity = {key: value for key, value in report.items() if key != "sha256"}
    if report.get("schema") != REPORT_SCHEMA or report.get("sha256") != _sha256(
        identity
    ):
        raise CompareError("comparison report digest is invalid")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical(dict(report)))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--real", required=True)
    parser.add_argument("--placebo", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build_report(args.baseline, args.real, args.placebo)
        write_report(args.output, report)
    except CompareError as exc:
        raise SystemExit(f"route-prefetch comparison failed: {exc}") from exc
    print(
        json.dumps(
            {
                "contrasts": report["contrasts"],
                "output": str(Path(args.output).expanduser().resolve()),
                "sha256": report["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
