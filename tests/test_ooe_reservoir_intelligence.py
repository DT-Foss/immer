from __future__ import annotations

import hashlib
import json
import warnings
import unittest

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.reservoir_intelligence import (
    run_reservoir_intelligence_benchmark,
)


class ReservoirIntelligenceBenchmarkTests(unittest.TestCase):
    def test_distributed_temporal_memory_beats_all_controls(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            report = run_reservoir_intelligence_benchmark()
        headline = report["headline"]
        self.assertGreater(headline["fused_accuracy"], 0.88)
        self.assertGreater(headline["fused_minus_local"], 0.05)
        self.assertGreater(headline["fused_minus_no_memory"], 0.30)
        self.assertGreater(headline["fused_minus_placebo"], 0.30)
        self.assertGreater(headline["topology_shift_accuracy"], 0.88)
        self.assertLess(
            report["body"]["topology_shift"]["complete_rounds"],
            report["body"]["topology_shift"]["barbell_rounds"],
        )
        self.assertLess(
            report["body"]["topology_shift"]["readout_max_absolute_delta"],
            1e-5,
        )
        json.dumps(report, allow_nan=False, sort_keys=True)

    def test_report_is_deterministic_and_self_sealed(self) -> None:
        first = run_reservoir_intelligence_benchmark(
            seed=91,
            train_steps_per_replica=200,
            test_steps=300,
            delay=4,
        )
        second = run_reservoir_intelligence_benchmark(
            seed=91,
            train_steps_per_replica=200,
            test_steps=300,
            delay=4,
        )
        self.assertEqual(first, second)
        body = {"body": first["body"], "headline": first["headline"]}
        self.assertEqual(
            first["sha256"],
            hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
        )


if __name__ == "__main__":
    unittest.main()
