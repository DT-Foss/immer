#!/usr/bin/env python3
"""Capture/resume exact Qwen MLP evidence through a typed runner.

This cut ships a bounded fixture runner for CI.  A live 27B runner must
implement ``ExactMlpCaptureRunner``; fixture receipts are permanently marked
and cannot be adapted into a production SubspaceCorpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from immer.runtimes.ooe.qwen_mlp_evidence import (
    CAPTURE_STAGES,
    CaptureManifest,
    CapturePlanEntry,
    ExactMlpBoundaryCapture,
    MlpEvidenceBudget,
    MlpEvidenceReceipt,
    MlpProjectionVerificationReceipt,
    QwenMlpEvidenceBank,
    run_capture_manifest,
)
from immer.runtimes.ooe.subspace_battery import graph_revision_sha256


STATUS_SCHEMA = "immer.qwen3.8-mlp-evidence-status/v1"
FIXTURE_SCHEMA = "immer.qwen3.8-mlp-evidence-fixture/v1"


class CliError(RuntimeError):
    pass


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class FixtureRunner:
    capture_mode = "fixture"

    def __init__(self, manifest: CaptureManifest, fixture: dict[str, object]) -> None:
        if fixture.get("schema") != FIXTURE_SCHEMA or set(fixture) != {
            "hidden_dimension",
            "intermediate_dimension",
            "rows",
            "schema",
            "seed_sha256",
        }:
            raise CliError("fixture schema is invalid")
        for field in ("rows", "hidden_dimension", "intermediate_dimension"):
            value = fixture.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise CliError(f"fixture {field} must be positive")
        seed = fixture.get("seed_sha256")
        if not isinstance(seed, str) or len(seed) != 64:
            raise CliError("fixture seed_sha256 is invalid")
        self.manifest = manifest
        self.rows = int(fixture["rows"])
        self.hidden = int(fixture["hidden_dimension"])
        self.intermediate = int(fixture["intermediate_dimension"])
        self.seed = seed

    def capture(
        self, entry: CapturePlanEntry, sink: ExactMlpBoundaryCapture
    ) -> tuple[MlpEvidenceReceipt, MlpProjectionVerificationReceipt]:
        generator = np.random.default_rng(
            int(
                hashlib.sha256(f"{self.seed}:{entry.sha256}".encode()).hexdigest()[:16],
                16,
            )
        )
        tensors = {
            "mlp.input": generator.normal(size=(1, self.rows, self.hidden)).astype(
                np.float32
            ),
            "mlp.gate": generator.normal(size=(1, self.rows, self.intermediate)).astype(
                np.float32
            ),
            "mlp.up": generator.normal(size=(1, self.rows, self.intermediate)).astype(
                np.float32
            ),
            "mlp.output": generator.normal(size=(1, self.rows, self.hidden)).astype(
                np.float32
            ),
        }
        sink.begin_group(entry)
        for stage in CAPTURE_STAGES:
            sink(entry.layer, stage, tensors[stage])
        event = _hash(f"fixture-atlas-event:{entry.ordinal}")
        sketch = _hash(f"fixture-input-sketch:{entry.ordinal}")
        receipt = sink.finalize(
            capture_mode=self.capture_mode,
            manifest_sha256=self.manifest.sha256,
            model_pin_sha256=self.manifest.model_pin_sha256,
            token_sha256=_hash(f"fixture-token:{entry.prompt_sha256}"),
            atlas_sequence=entry.ordinal,
            atlas_event_sha256=event,
            atlas_revision_sha256=graph_revision_sha256(entry.ordinal, event),
            measurement_sha256=_hash(f"fixture-measurement:{entry.ordinal}"),
            weight_revision_sha256=_hash("fixture-weight-revision"),
            access_trace_sha256=_hash(f"fixture-access:{entry.ordinal}"),
            source_receipt_sha256s=tuple(
                sorted(
                    (
                        _hash(f"fixture-gate-range:{entry.ordinal}"),
                        _hash(f"fixture-up-range:{entry.ordinal}"),
                        _hash(f"fixture-down-range:{entry.ordinal}"),
                    )
                )
            ),
            cartography_input_sketch_sha256=sketch,
            recomputed_input_sketch_sha256=sketch,
        )
        verification = MlpProjectionVerificationReceipt(
            evidence_receipt_sha256=receipt.sha256,
            verifier_sha256=_hash("fixture-verifier-not-live"),
            verifier_evidence_sha256=_hash(
                f"fixture-verifier-evidence:{receipt.sha256}"
            ),
            replay_access_trace_sha256=_hash(f"fixture-replay:{receipt.sha256}"),
            storage_exact=True,
            input_sketch_exact=True,
            gate_exact=True,
            up_exact=True,
            output_exact=True,
        )
        return receipt, verification


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--fixture",
        required=True,
        help="Bounded CI fixture JSON; live runner is supplied by the separate observer cut",
    )
    parser.add_argument("--max-total-mb", type=float, default=4096.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest_path = Path(args.manifest).expanduser().absolute()
    fixture_path = Path(args.fixture).expanduser().absolute()
    manifest = CaptureManifest.from_bytes(manifest_path.read_bytes())
    try:
        fixture = json.loads(fixture_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CliError("fixture is not JSON") from exc
    if not isinstance(fixture, dict):
        raise CliError("fixture must be an object")
    maximum = float(args.max_total_mb)
    if not np.isfinite(maximum) or maximum <= 0:
        raise CliError("--max-total-mb must be finite and positive")
    bank = QwenMlpEvidenceBank(
        Path(args.root).expanduser().absolute(),
        budget=MlpEvidenceBudget(max_total_referenced_bytes=int(maximum * 1024**2)),
    )
    publications = run_capture_manifest(
        manifest, bank, FixtureRunner(manifest, fixture)
    )
    state = bank.state()
    audit = bank.audit()
    status = {
        "audit_clean": audit.clean,
        "fixture": True,
        "manifest_sha256": manifest.sha256,
        "new_publications": len(publications),
        "receipt_count": state.receipt_count,
        "referenced_tensor_bytes": state.referenced_tensor_bytes,
        "schema": STATUS_SCHEMA,
        "split_counts": list(state.split_counts),
        "state_sha256": state.sha256,
    }
    print(json.dumps(status, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
