from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import warnings

from immer.runtimes.ooe.identity import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark():
    path = ROOT / "scripts" / "ooe_demand_blanket_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "_ooe_demand_blanket_benchmark", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DemandBlanketBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_benchmark()
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            cls.report = cls.module.build_report(seeds=1)

    def test_real_receipts_recover_noncontiguous_memory_and_execute_priority(self) -> None:
        module = self.module
        report = self.report
        self.assertTrue(module.verify_report(report))
        self.assertEqual(report["schema"], module.REPORT_SCHEMA)
        self.assertEqual(
            report["body_sha256"], module._digest(report["body"])
        )
        aggregate = report["body"]["aggregate"]
        exact_one = {"numerator": 1, "denominator": 1}
        for field in (
            "selected_lags_1_3_ratio",
            "validation_acceptance_ratio",
            "unseen_history_exact_ratio",
            "blanket_priority_ratio",
            "ppm_priority_ratio",
            "full_ucb_priority_ratio",
            "incompatible_never_executed_ratio",
            "exact_output_ratio",
            "past_compute_release_ratio",
        ):
            self.assertEqual(aggregate[field], exact_one, field)
        self.assertEqual(
            aggregate["selected_holdout_accuracy"],
            aggregate["full_holdout_accuracy"],
        )
        self.assertGreater(
            module._from_ratio(
                aggregate["selected_holdout_accuracy"], field="selected"
            ),
            module._from_ratio(
                aggregate["contiguous_ppm_holdout_accuracy"], field="ppm"
            ),
        )
        self.assertGreater(
            module._from_ratio(
                aggregate["selected_holdout_accuracy"], field="selected"
            ),
            module._from_ratio(
                aggregate["random_holdout_accuracy"], field="random"
            ),
        )
        self.assertGreater(aggregate["historical_work_released"], 0)
        self.assertEqual(
            aggregate["lag_feature_reduction"],
            {"numerator": 1, "denominator": 3},
        )

        seed = report["body"]["seed_reports"][0]["body"]
        self.assertEqual(seed["selected_lags"], [1, 3])
        self.assertEqual(len(seed["outcome_receipt_sha256s"]), 30)
        self.assertEqual(seed["train_episode_count"], 1)
        self.assertEqual(seed["calibration_episode_count"], 1)
        self.assertEqual(seed["holdout_episode_count"], 1)
        for field in (
            "receipt_sha256",
            "model_sha256",
            "validation_sha256",
            "prediction_sha256",
        ):
            self.assertEqual(len(seed[field]), 64, field)
        self.assertEqual(
            set(seed["execution_sha256s"]),
            {"blanket", "ppm_fallback", "full_ucb_fallback", "incompatible_guard"},
        )
        executions = seed["executions"]
        self.assertIsNone(executions["blanket"]["ppm_prediction_sha256"])
        self.assertIsNone(
            executions["ppm_fallback"]["blanket_candidate_route_sha256"]
        )
        self.assertIsNotNone(executions["ppm_fallback"]["ppm_prediction_sha256"])
        self.assertEqual(executions["full_ucb_fallback"]["selection_score_count"], 2)
        self.assertNotEqual(
            executions["incompatible_guard"]["blanket_candidate_route_sha256"],
            executions["incompatible_guard"]["selected_prefix_route_sha256"],
        )
        self.assertTrue(all(seed["checks"].values()))
        self.assertTrue(
            all(
                execution["verified"]
                and execution["historical_work_released"] > 0
                for execution in executions.values()
            )
        )

    def test_deterministic_rerun_tamper_and_atomic_no_replace(self) -> None:
        module = self.module
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            rerun = module.build_report(seeds=1)
        self.assertEqual(
            canonical_json_bytes(rerun), canonical_json_bytes(self.report)
        )

        tampered = copy.deepcopy(self.report)
        tampered["body"]["seed_reports"][0]["body"]["selected_lags"] = [1, 2, 3]
        with self.assertRaises(module.DemandBlanketBenchmarkError):
            module.verify_report(tampered)

        def reseal(value):
            seed_report = value["body"]["seed_reports"][0]
            seed_report["body_sha256"] = module._digest(seed_report["body"])
            value["body"]["aggregate"] = module._aggregate(
                value["body"]["seed_reports"]
            )
            value["body_sha256"] = module._digest(value["body"])

        forged_score = copy.deepcopy(self.report)
        forged_score["body"]["seed_reports"][0]["body"]["executions"][
            "full_ucb_fallback"
        ]["selection_score_count"] = 1
        reseal(forged_score)
        with self.assertRaises(module.DemandBlanketBenchmarkError):
            module.verify_report(forged_score)

        forged_check = copy.deepcopy(self.report)
        forged_check["body"]["seed_reports"][0]["body"]["checks"][
            "blanket_precedes_ppm"
        ] = False
        reseal(forged_check)
        with self.assertRaises(module.DemandBlanketBenchmarkError):
            module.verify_report(forged_check)

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "report.json"
            module.write_report_no_replace(output, self.report)
            first = output.read_bytes()
            self.assertEqual(first, canonical_json_bytes(self.report))
            self.assertEqual(json.loads(first), self.report)
            with self.assertRaises(FileExistsError):
                module.write_report_no_replace(output, rerun)
            self.assertEqual(output.read_bytes(), first)


if __name__ == "__main__":
    unittest.main()
