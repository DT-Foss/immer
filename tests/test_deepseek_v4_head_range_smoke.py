from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deepseek_v4_head_range_smoke.py"
REPORT = ROOT / "results" / "deepseek-v4-head-range-network-smoke.json"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_head_range_smoke", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = _load_script()


def _source_delta(*, requests: int, retries: int = 0) -> dict[str, int]:
    return {
        "network_or_source_body_bytes": 100,
        "range_source_bytes": 100,
        "range_source_requests": requests,
        "failed_requests": 0,
        "transport_retries": retries,
    }


def _arm(*, name: str, requests: int, seconds: float) -> dict:
    return {
        "arm": name,
        "seconds": seconds,
        "head": {"sha256": "a" * 64},
        "logit_stream_sha256": "b" * 64,
        "logit_rows": 8,
        "logit_blocks": 4,
        "topk_values_sha256": "c" * 64,
        "topk_token_ids_sha256": "d" * 64,
        "topk_values": [[4.0, 3.0]],
        "topk_token_ids": [[7, 2]],
        "source_metrics_delta": _source_delta(requests=requests),
        "pager_metrics_delta": {"head_transport_fallbacks": 0},
        "source_identity": {
            "revision_is_pinned": True,
            "revision_is_mutable": False,
            "transport_active_leases": 0,
            "transport_closed": True,
        },
    }


class DeepSeekV4HeadRangeSmokeTests(unittest.TestCase):
    def test_sealed_report_binds_current_runtime_and_repeated_exact_pairs(self) -> None:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        seal = report.pop("report_sha256")
        self.assertEqual(smoke._canonical_digest(report), seal)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["protocol"]["pairs"], 4)
        self.assertEqual(report["summary"]["candidate_wins"], 4)
        self.assertGreater(
            report["summary"]["median_speedup_baseline_over_candidate"], 1.0
        )

        provenance = report["provenance"]
        sources = smoke.runtime_source_manifest(
            project_files=("scripts/deepseek_v4_head_range_smoke.py",)
        )
        dependencies = smoke.runtime_dependency_versions()
        self.assertEqual(provenance["runtime_sources"], sources)
        self.assertEqual(
            provenance["runtime_source_sha256"], smoke._canonical_digest(sources)
        )
        self.assertEqual(provenance["runtime_dependencies"], dependencies)
        self.assertEqual(
            provenance["runtime_dependency_sha256"],
            smoke._canonical_digest(dependencies),
        )
        self.assertEqual(provenance["harness_sha256"], smoke._sha256_file(SCRIPT))

        reference_stream = None
        for trial in report["trials"]:
            baseline = trial["baseline"]
            candidate = trial["candidate"]
            comparison = trial["comparison"]
            self.assertTrue(comparison["bit_identical"])
            self.assertTrue(comparison["source_bytes_equal"])
            self.assertEqual(comparison["baseline_source_requests"], 127)
            self.assertEqual(comparison["candidate_source_requests"], 16)
            self.assertEqual(comparison["source_requests_avoided"], 111)
            self.assertEqual(
                baseline["source_metrics_delta"]["range_source_bytes"],
                candidate["source_metrics_delta"]["range_source_bytes"],
            )
            self.assertEqual(
                baseline["source_metrics_delta"]["transport_retries"], 0
            )
            self.assertEqual(
                candidate["source_metrics_delta"]["transport_retries"], 0
            )
            stream = baseline["logit_stream_sha256"]
            self.assertEqual(candidate["logit_stream_sha256"], stream)
            if reference_stream is None:
                reference_stream = stream
            self.assertEqual(stream, reference_stream)

    def test_source_builder_rejects_unsealable_local_or_alternate_source(self) -> None:
        args = SimpleNamespace(source="local:/tmp/checkpoint", revision="a" * 40)
        with self.assertRaisesRegex(
            smoke.HeadSmokeError, "bound to the official checkpoint source"
        ):
            smoke._build_source(
                args,
                cache_dir=Path("/tmp/not-created"),
                budget_mb=1.0,
                max_cache_bytes=1,
            )

    def test_schedule_alternates_and_rejects_invalid_pairs(self) -> None:
        self.assertEqual(
            smoke._trial_schedule(3),
            (
                ("baseline", "candidate"),
                ("candidate", "baseline"),
                ("baseline", "candidate"),
            ),
        )
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                smoke._trial_schedule(value)

    def test_pair_gate_requires_exact_outputs_equal_bytes_and_fewer_requests(
        self,
    ) -> None:
        baseline = _arm(name="baseline", requests=127, seconds=12.0)
        candidate = _arm(name="candidate", requests=16, seconds=8.0)
        comparison = smoke._validate_pair(baseline, candidate)
        self.assertTrue(comparison["bit_identical"])
        self.assertEqual(comparison["source_requests_avoided"], 111)
        self.assertEqual(comparison["speedup_baseline_over_candidate"], 1.5)

        mismatched = dict(candidate, logit_stream_sha256="e" * 64)
        with self.assertRaisesRegex(smoke.HeadSmokeError, "exactness mismatch"):
            smoke._validate_pair(baseline, mismatched)

        no_reduction = dict(candidate)
        no_reduction["source_metrics_delta"] = _source_delta(requests=127)
        with self.assertRaisesRegex(smoke.HeadSmokeError, "did not reduce"):
            smoke._validate_pair(baseline, no_reduction)

        retried = dict(candidate)
        retried["source_metrics_delta"] = _source_delta(requests=16, retries=1)
        with self.assertRaisesRegex(smoke.HeadSmokeError, "transport retries"):
            smoke._validate_pair(baseline, retried)

    def test_full_logit_stream_digest_is_order_and_value_bound(self) -> None:
        import torch

        blocks = [
            (0, torch.asarray([[1.0, 2.0]], dtype=torch.float32)),
            (2, torch.asarray([[3.0]], dtype=torch.float32)),
        ]
        first, rows = smoke._logit_stream_digest(blocks)
        second, _ = smoke._logit_stream_digest(blocks)
        changed, _ = smoke._logit_stream_digest(
            [blocks[0], (2, torch.asarray([[4.0]], dtype=torch.float32))]
        )
        self.assertEqual(rows, 3)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        with self.assertRaisesRegex(smoke.HeadSmokeError, "out of order"):
            smoke._logit_stream_digest(list(reversed(blocks)))

    def test_summary_is_paired_and_does_not_overclaim_one_smoke(self) -> None:
        trials = [
            {
                "baseline": {"seconds": 12.0},
                "candidate": {"seconds": 8.0},
                "comparison": {"speedup_baseline_over_candidate": 1.5},
            },
            {
                "baseline": {"seconds": 10.0},
                "candidate": {"seconds": 10.0},
                "comparison": {"speedup_baseline_over_candidate": 1.0},
            },
        ]
        summary = smoke._summarize(trials)
        self.assertEqual(summary["pairs"], 2)
        self.assertEqual(summary["candidate_wins"], 1)
        self.assertEqual(summary["median_speedup_baseline_over_candidate"], 1.25)
        self.assertIn("mechanism_smoke_only", summary["performance_claim"])

    def test_manifest_selection_is_sealed_and_bound_to_the_final_layer(self) -> None:
        import torch
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as raw:
            run = Path(raw)
            objects = run / "objects"
            objects.mkdir()
            temporary = objects / "activation.safetensors"
            hidden = torch.arange(24, dtype=torch.float32).reshape(1, 3, 4, 2)
            hidden = hidden.to(dtype=torch.bfloat16)
            save_file({"hidden": hidden}, temporary)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            object_path = objects / f"{digest}.safetensors"
            temporary.rename(object_path)
            config_sha256 = "f" * 64
            checkpoint = {
                "bucket": 0,
                "dtype": "bfloat16",
                "file": object_path.name,
                "file_bytes": object_path.stat().st_size,
                "generation_layer": 42,
                "schema": "immer.deepseek-v4-layerwise-activation/v4",
                "sha256": digest,
                "shape": [1, 3, 4, 2],
                "tensor_bytes": hidden.numel() * hidden.element_size(),
                "variant": "off",
                "version": 4,
            }
            body = {
                "identity": {
                    "model": {
                        "repo_id": "repo",
                        "revision": "1" * 40,
                        "config_sha256": config_sha256,
                    }
                },
                "plan": {
                    "buckets": [
                        {
                            "index": 0,
                            "item_indices": [7],
                            "valid_lengths": [2],
                        }
                    ]
                },
                "state": {"checkpoints": [checkpoint]},
            }
            manifest = {
                "schema": "immer.deepseek-v4-layerwise/v4",
                "version": 4,
                "kind": "manifest",
                "body": body,
                "body_sha256": smoke._canonical_digest(body),
            }
            path = run / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            config = SimpleNamespace(n_layers=43, hc_mult=4, dim=2)

            actual, evidence = smoke._validate_manifest(
                path,
                config=config,
                config_sha256=config_sha256,
                source="repo",
                revision="1" * 40,
                variant="off",
                bucket=0,
                row=0,
            )
            torch.testing.assert_close(actual, hidden)
            self.assertEqual(evidence["selected_item_index"], 7)
            self.assertEqual(evidence["selected_sequence_position"], 1)

            broken = json.loads(path.read_text(encoding="utf-8"))
            broken["body_sha256"] = "0" * 64
            path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(smoke.HeadSmokeError, "seal is invalid"):
                smoke._validate_manifest(
                    path,
                    config=config,
                    config_sha256=config_sha256,
                    source="repo",
                    revision="1" * 40,
                    variant="off",
                    bucket=0,
                    row=0,
                )

    def test_atomic_report_seal_and_symlink_refusal(self) -> None:
        document = smoke._seal_report({"schema": smoke.RESULT_SCHEMA, "value": 1})
        unsealed = dict(document)
        digest = unsealed.pop("report_sha256")
        self.assertEqual(digest, smoke._canonical_digest(unsealed))

        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "nested" / "report.json"
            smoke._atomic_write_json(output, document)
            self.assertEqual(json.loads(output.read_text()), document)

            foreign = root / "foreign.json"
            foreign.write_text("sentinel", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(foreign)
            with self.assertRaisesRegex(smoke.HeadSmokeError, "symlink"):
                smoke._atomic_write_json(link, document)
            self.assertEqual(foreign.read_text(encoding="utf-8"), "sentinel")

    def test_cli_defaults_bind_official_checkpoint_and_existing_activation(self) -> None:
        args = smoke._parser().parse_args([])
        self.assertEqual(args.source, smoke.OFFICIAL_SOURCE)
        self.assertEqual(args.revision, smoke.OFFICIAL_REVISION)
        self.assertEqual(args.block_rows, 1024)
        self.assertEqual(args.candidate_batch_blocks, 8)
        self.assertEqual(args.pairs, 1)
        self.assertEqual(args.output_json, smoke.DEFAULT_OUTPUT)


if __name__ == "__main__":
    unittest.main()
