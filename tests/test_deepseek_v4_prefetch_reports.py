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
REPORTS = (
    ROOT / "results" / "deepseek-v4-exact-prefetch-window-smoke.json",
    ROOT / "results" / "deepseek-v4-exact-prefetch-window-network-smoke.json",
)
HARNESS = ROOT / "scripts" / "deepseek_v4_prefetch_smoke.py"


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
                self.assertEqual(
                    execution["expert_prefetch_policy"],
                    "exact-router-window-q3-a2/v2",
                )
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
                self.assertTrue(report["exactness"]["all_outputs_bit_equal"])


if __name__ == "__main__":
    unittest.main()
