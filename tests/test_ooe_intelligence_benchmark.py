from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.intelligence import run_intelligence_benchmark


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ooe_intelligence_benchmark.py"


def _script_module():
    specification = importlib.util.spec_from_file_location(
        "ooe_intelligence_benchmark",
        SCRIPT,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class OoeIntelligenceBenchmarkTests(unittest.TestCase):
    def test_full_matrix_learns_plans_options_novelty_and_regime_change(self) -> None:
        report = run_intelligence_benchmark(task_count=50)
        headline = report["headline"]

        self.assertEqual(headline["ps_lifted_success_rate"], 1.0)
        self.assertEqual(headline["central_success_rate"], 1.0)
        self.assertGreater(headline["ps_minus_local"], 0.8)
        self.assertGreater(headline["ps_minus_placebo"], 0.5)
        self.assertGreaterEqual(headline["option_depth_reduction"], 1)
        self.assertLess(headline["mpo_compression_ratio"], 0.5)
        self.assertEqual(headline["ood_false_accepts"], 0)
        self.assertGreater(headline["regime_adaptation_gain"], 0.5)
        self.assertEqual(headline["topology_shift_success_rate"], 1.0)
        self.assertLess(headline["no_memory_success_rate"], 0.5)
        self.assertGreater(headline["reservoir_fused_accuracy"], 0.88)
        self.assertGreater(headline["reservoir_fused_minus_no_memory"], 0.30)
        self.assertGreater(headline["reservoir_fused_minus_placebo"], 0.30)
        self.assertEqual(headline["compute_crystal_route_length"], 4)
        self.assertEqual(headline["compute_crystal_live_operators"], 1)
        self.assertGreater(
            headline["compute_crystal_historical_work_released"],
            0,
        )
        self.assertTrue(headline["algebraic_crystals_length_12_exact"])
        self.assertEqual(report["body"]["task_contract"]["multi_step_teacher_labels"], 0)
        self.assertEqual(report["body"]["fusion"]["evidence_events"], 68)
        self.assertEqual(report["body"]["local_replica"]["replica_count"], 12)
        local = report["body"]["local_replica"]
        self.assertEqual(
            local["success"],
            sum(replica["success"] for replica in local["replicas"]),
        )
        self.assertEqual(local["tasks"], 12 * 50)
        self.assertAlmostEqual(
            headline["local_success_rate"],
            local["success"] / local["tasks"],
        )
        self.assertLess(
            report["body"]["topology_shift"]["complete_rounds"],
            report["body"]["topology_shift"]["barbell_rounds"],
        )
        json.dumps(report, allow_nan=False, sort_keys=True)

    def test_report_is_seed_deterministic_and_self_sealed(self) -> None:
        first = run_intelligence_benchmark(seed=77, task_count=25)
        second = run_intelligence_benchmark(seed=77, task_count=25)
        self.assertEqual(first, second)
        body = {"body": first["body"], "headline": first["headline"]}
        import hashlib

        self.assertEqual(
            first["sha256"],
            hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
        )

    def test_cli_writes_atomic_canonical_report(self) -> None:
        module = _script_module()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "report.json"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                status = module.main(
                    [
                        "--seed",
                        "99",
                        "--tasks",
                        "25",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(status, 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(output.read_bytes(), canonical_json_bytes(report) + b"\n")
            self.assertFalse(any(output.parent.glob("*.pending")))


if __name__ == "__main__":
    unittest.main()
