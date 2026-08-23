from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deepseek_v4_route_prefetch_compare.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_route_prefetch_compare", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import route-prefetch comparison script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare = _load_script()


def _result(*, enabled: bool, role: str | None, seconds: float, body: int) -> dict:
    pager = {
        "expert_reservoir_submitted": 0 if not enabled else 12,
        "expert_reservoir_prediction_hits": 0 if not enabled else 8,
        "expert_reservoir_usable_hits": 0 if not enabled else 8,
        "expert_reservoir_misses": 0 if not enabled else 4,
        "expert_reservoir_wasted": 0 if not enabled else 4,
        "expert_reservoir_wait_ns": 0 if not enabled else 100,
    }
    return {
        "schema": compare.RESULT_SCHEMA,
        "status": "complete",
        "protocol": {"checkpoint": "deepseek-ai/DeepSeek-V4-Flash-0731"},
        "summary": {"fully_verified": 3, "local_correct": 3},
        "traffic": {
            "api": {"total_tokens": 10},
            "local": {"source_body_bytes": body, "linear_calls": 100},
        },
        "evidence": {
            "seconds": seconds,
            "source_body_bytes": body,
            "linear_calls": 100,
            "layer_calls": 43,
        },
        "instrumentation": {
            "route_prefetch": {
                "enabled": enabled,
                "model_role": role,
                "model_snapshot_sha256": None if role is None else role + "-sha",
                "pager": pager,
            }
        },
        "items": [{"item_id": "same", "local_answer": "42"}],
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class RoutePrefetchCompareTests(unittest.TestCase):
    def test_identical_outputs_produce_controlled_transport_contrasts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            baseline = directory / "baseline.json"
            real = directory / "real.json"
            placebo = directory / "placebo.json"
            output = directory / "comparison.json"
            legacy_baseline = _result(enabled=False, role=None, seconds=10, body=1000)
            legacy_baseline.pop("instrumentation")
            _write(baseline, legacy_baseline)
            _write(
                real,
                _result(enabled=True, role="real_markov", seconds=7, body=800),
            )
            _write(
                placebo,
                _result(enabled=True, role="placebo_markov", seconds=9, body=950),
            )

            report = compare.build_report(baseline, real, placebo)
            compare.write_report(output, report)

            self.assertEqual(
                report["contrasts"]["real_vs_baseline"]["seconds_ratio"], 0.7
            )
            self.assertEqual(
                report["contrasts"]["real_vs_placebo"]["source_body_bytes_delta"],
                -150,
            )
            self.assertEqual(
                report["arms"]["real_markov"]["reservoir"]["demand_coverage"],
                2 / 3,
            )
            self.assertTrue(
                report["arms"]["baseline"]["legacy_baseline_instrumentation"]
            )
            self.assertEqual(
                output.read_bytes(),
                json.dumps(
                    report,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8"),
            )

    def test_changed_outputs_or_wrong_arm_role_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            baseline = directory / "baseline.json"
            real = directory / "real.json"
            placebo = directory / "placebo.json"
            _write(baseline, _result(enabled=False, role=None, seconds=10, body=1000))
            changed = _result(enabled=True, role="real_markov", seconds=7, body=800)
            changed["items"][0]["local_answer"] = "41"
            _write(real, changed)
            _write(
                placebo,
                _result(enabled=True, role="placebo_markov", seconds=9, body=950),
            )
            with self.assertRaisesRegex(compare.CompareError, "changed model outputs"):
                compare.build_report(baseline, real, placebo)

            fixed = copy.deepcopy(changed)
            fixed["items"][0]["local_answer"] = "42"
            fixed["instrumentation"]["route_prefetch"]["model_role"] = "placebo_markov"
            _write(real, fixed)
            with self.assertRaisesRegex(compare.CompareError, "role"):
                compare.build_report(baseline, real, placebo)


if __name__ == "__main__":
    unittest.main()
