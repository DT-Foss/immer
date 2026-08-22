from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "deepseek_v4_prefetch_smoke.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("deepseek_v4_prefetch_smoke", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import prefetch smoke script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


smoke = _load_script()


class DeepSeekV4PrefetchSmokeScriptTests(unittest.TestCase):
    def test_defaults_are_official_pinned_and_counterbalanced(self) -> None:
        args = smoke._parser().parse_args([])
        self.assertEqual(args.source, smoke.OFFICIAL_SOURCE)
        self.assertEqual(args.revision, smoke.OFFICIAL_REVISION)
        self.assertRegex(args.revision, r"^[0-9a-f]{40,64}$")
        self.assertEqual(args.layer, 3)
        self.assertEqual(args.experts, (0, 1, 2))
        self.assertEqual(args.trials, 20)
        self.assertEqual(args.route_weight, 0.25)
        self.assertEqual(smoke._trial_schedule(4), ("off", "on", "on", "off"))
        with self.assertRaisesRegex(smoke.SmokeError, "even"):
            smoke._trial_schedule(3)

    def test_expert_argument_rejects_duplicates_and_wrong_arity(self) -> None:
        self.assertEqual(smoke._parse_experts("7,2,9"), (7, 2, 9))
        for raw in ("1", "1,1", "-1,2", "1,2,1", "x,2"):
            with self.subTest(raw=raw):
                with self.assertRaises(Exception):
                    smoke._parse_experts(raw)

    def test_pure_summary_computes_balanced_speedup(self) -> None:
        summary = smoke._summarize_trials(
            (
                {"mode": "off", "seconds": 0.8},
                {"mode": "on", "seconds": 0.4},
                {"mode": "on", "seconds": 0.6},
                {"mode": "off", "seconds": 1.2},
            )
        )
        self.assertEqual(summary["modes"]["off"]["count"], 2)
        self.assertEqual(summary["modes"]["on"]["count"], 2)
        self.assertAlmostEqual(summary["modes"]["off"]["mean_seconds"], 1.0)
        self.assertAlmostEqual(summary["modes"]["on"]["mean_seconds"], 0.5)
        self.assertAlmostEqual(summary["mean_speedup_off_over_on"], 2.0)
        self.assertAlmostEqual(summary["mean_latency_reduction_fraction"], 0.5)
        with self.assertRaisesRegex(smoke.SmokeError, "equal"):
            smoke._summarize_trials(({"mode": "off", "seconds": 1.0},))

    def test_atomic_schema_report_is_sealed_and_does_not_follow_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            report = smoke._seal_report(
                {
                    "schema": smoke.RESULT_SCHEMA,
                    "status": "ok",
                    "quality_or_end_to_end_performance_claim": False,
                }
            )
            expected = smoke._canonical_digest(
                {key: value for key, value in report.items() if key != "report_sha256"}
            )
            self.assertEqual(report["report_sha256"], expected)

            output = root / "report.json"
            smoke._atomic_write_json(output, report)
            self.assertTrue(output.read_bytes().endswith(b"\n"))
            self.assertEqual(json.loads(output.read_text()), report)

            foreign = root / "foreign.json"
            foreign.write_text("sentinel", encoding="utf-8")
            link = root / "link.json"
            link.symlink_to(foreign)
            with self.assertRaisesRegex(smoke.SmokeError, "symlink"):
                smoke._atomic_write_json(link, report)
            self.assertEqual(foreign.read_text(encoding="utf-8"), "sentinel")


if __name__ == "__main__":
    unittest.main()
