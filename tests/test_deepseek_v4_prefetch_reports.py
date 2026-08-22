from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest

from immer.runtimes.deepseek_v4.benchmark import canonical_digest
from immer.runtimes.deepseek_v4.provenance import (
    runtime_dependency_versions,
    runtime_source_manifest,
)


ROOT = Path(__file__).resolve().parent.parent
Q3_REPORTS = (
    ROOT / "results" / "deepseek-v4-exact-prefetch-window-smoke.json",
    ROOT / "results" / "deepseek-v4-exact-prefetch-window-network-smoke.json",
)
ADJACENT_REPORTS = (
    ROOT / "results" / "deepseek-v4-adjacent-range-warm-smoke.json",
    ROOT / "results" / "deepseek-v4-adjacent-range-network-smoke.json",
)
REPORTS = Q3_REPORTS + ADJACENT_REPORTS
HARNESS = ROOT / "scripts" / "deepseek_v4_prefetch_smoke.py"
EXPERT_PAYLOAD_BYTES = 40_108_032


class DeepSeekV4PrefetchReportsTests(unittest.TestCase):
    def test_reports_bind_current_runtime_and_closed_bounded_transport(self) -> None:
        sources = runtime_source_manifest()
        dependencies = runtime_dependency_versions()
        harness_sha256 = hashlib.sha256(HARNESS.read_bytes()).hexdigest()
        for path in REPORTS:
            with self.subTest(path=path.name):
                report = json.loads(path.read_text(encoding="utf-8"))
                seal = report.pop("report_sha256")
                self.assertEqual(canonical_digest(report), seal)
                provenance = report["provenance"]
                self.assertEqual(provenance["runtime_sources"], sources)
                self.assertEqual(
                    provenance["runtime_source_sha256"], canonical_digest(sources)
                )
                self.assertEqual(provenance["runtime_dependencies"], dependencies)
                self.assertEqual(
                    provenance["runtime_dependency_sha256"],
                    canonical_digest(dependencies),
                )
                self.assertEqual(provenance["harness_sha256"], harness_sha256)
                execution = provenance["execution"]
                self.assertEqual(execution["expert_prefetch_active_read_limit"], 2)
                self.assertEqual(execution["expert_prefetch_max_outstanding"], 3)
                self.assertEqual(
                    execution["source_transport_policy"],
                    "requests-session-pool-2/v1",
                )
                self.assertEqual(execution["source_transport_connection_limit"], 2)
                source = report["metrics"]["source"]["after"]
                self.assertLessEqual(source["transport_peak_leases"], 2)
                self.assertEqual(source["transport_active_leases"], 0)
                self.assertTrue(source["transport_closed"])
                self.assertEqual(source["transport_retries"], 0)
                self.assertEqual(source["failed_requests"], 0)
                self.assertTrue(report["exactness"]["all_outputs_finite"])
                self.assertTrue(report["exactness"]["all_outputs_torch_equal"])
                self.assertTrue(report["exactness"]["all_outputs_bit_equal"])
                expected_cache_mode = (
                    "disabled_cold_source_each_execution"
                    if report["arguments"]["no_cache"]
                    else "shared_exact_leaf_cache_after_symmetric_warmup"
                )
                self.assertEqual(report["protocol"]["cache_mode"], expected_cache_mode)

    def test_q3_reports_bind_disabled_baseline_and_production_policy(self) -> None:
        for path in Q3_REPORTS:
            with self.subTest(path=path.name):
                report = json.loads(path.read_text(encoding="utf-8"))
                execution = report["provenance"]["execution"]
                self.assertEqual(report["protocol"]["contrast"], "off-vs-prefetch")
                self.assertEqual(execution["baseline_mode"], "off")
                self.assertEqual(execution["candidate_mode"], "on")
                self.assertEqual(
                    execution["baseline_expert_prefetch_policy"], "disabled"
                )
                self.assertEqual(
                    execution["baseline_expert_prefetch_transport_policy"],
                    "disabled",
                )
                self.assertEqual(
                    execution["expert_prefetch_policy"],
                    "exact-router-window-q3-a2/v2",
                )
                self.assertEqual(
                    execution["expert_prefetch_transport_policy"],
                    "streamer-exact-range/v1",
                )
                self.assertEqual(execution["expert_range_coalesce_max_experts"], 1)
                self.assertEqual(execution["expert_range_coalesce_max_gap_bytes"], 0)

    def test_adjacent_reports_bind_candidate_and_physical_receipts(self) -> None:
        for path in ADJACENT_REPORTS:
            with self.subTest(path=path.name):
                report = json.loads(path.read_text(encoding="utf-8"))
                execution = report["provenance"]["execution"]
                protocol = report["protocol"]
                self.assertEqual(protocol["contrast"], "q3-vs-adjacent-pairs")
                self.assertEqual(execution["baseline_mode"], "q3")
                self.assertEqual(execution["candidate_mode"], "adjacent_pairs")
                self.assertEqual(
                    execution["baseline_expert_prefetch_policy"],
                    "exact-router-window-q3-a2/v2",
                )
                self.assertEqual(
                    execution["baseline_expert_prefetch_transport_policy"],
                    "streamer-exact-range/v1",
                )
                self.assertEqual(
                    execution["expert_prefetch_policy"],
                    "exact-router-window-q3-a2-adjacent-pairs/v3",
                )
                self.assertEqual(
                    execution["expert_prefetch_transport_policy"],
                    "streamer-exact-leaf-adjacent-envelope/v2",
                )
                self.assertEqual(execution["expert_range_coalesce_max_experts"], 2)
                self.assertEqual(execution["expert_range_coalesce_max_gap_bytes"], 0)
                self.assertEqual(
                    protocol["arm_contracts"],
                    {
                        "q3": {
                            "submitted": 3,
                            "batches": 3,
                            "range_requests_avoided": 0,
                            "minimum_source_envelopes": 6,
                        },
                        "adjacent_pairs": {
                            "submitted": 3,
                            "batches": 2,
                            "range_requests_avoided": 2,
                            "minimum_source_envelopes": 4,
                        },
                    },
                )
                cold = bool(report["arguments"]["no_cache"])
                for trial in report["trials"]:
                    delta = trial["pager_delta"]
                    self.assertEqual(delta["expert_range_gap_bytes"], 0)
                    self.assertEqual(
                        delta["expert_range_requests_avoided"],
                        2 if trial["mode"] == "adjacent_pairs" else 0,
                    )
                    expected_envelopes = (
                        (4 if trial["mode"] == "adjacent_pairs" else 6)
                        if cold
                        else 0
                    )
                    expected_bytes = EXPERT_PAYLOAD_BYTES if cold else 0
                    self.assertEqual(
                        delta["expert_transport_envelopes"], expected_envelopes
                    )
                    self.assertEqual(
                        delta["expert_transport_source_bytes"], expected_bytes
                    )
                    self.assertEqual(trial["source_body_bytes_delta"], expected_bytes)


if __name__ == "__main__":
    unittest.main()
