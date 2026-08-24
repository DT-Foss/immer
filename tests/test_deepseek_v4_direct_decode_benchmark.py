from __future__ import annotations

import copy
from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from dataclasses import replace

from test_deepseek_v4_model import (
    _TwoLayerPrefetchQuantizedTinyCheckpoint,
    _config,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "deepseek_v4_direct_decode_benchmark.py"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "deepseek_v4_direct_decode_benchmark", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import direct decode benchmark script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = _load_script()


def _arm(name: str, *, seconds: float, body: int, hidden: str = "h" * 64):
    enabled = name != "baseline"
    identity = {
        "arm": name,
        "checkpoint": {
            "inventory_fingerprint": "f" * 64,
            "repo_id": benchmark.OFFICIAL_SOURCE,
            "revision": benchmark.OFFICIAL_REVISION,
        },
        "evidence": {
            "context_mode": "decode",
            "input_token_ids": [[11]],
            "seconds": seconds,
            "selected_experts": [[[0, 1]], [[0, 1]]],
            "source_body_bytes": body,
        },
        "hidden_sha256": hidden,
        "input_sha256": "i" * 64,
        "layers": [
            {
                "layer": layer,
                "seconds": seconds / 43,
                "source_body_bytes": body // 43,
            }
            for layer in range(43)
        ],
        "pager": {
            "expert_prefetch_wait_ns": int(seconds * 100),
            "expert_reservoir_failures": 0,
            "expert_reservoir_misses": 1 if enabled else 0,
            "expert_reservoir_ready_hits": 1 if enabled else 0,
            "expert_reservoir_submitted": 1 if enabled else 0,
            "expert_reservoir_usable_hits": 1 if enabled else 0,
            "expert_reservoir_wait_ns": 7 if enabled else 0,
            "expert_reservoir_wasted": 0,
        },
        "route_model_artifact_sha256": None if not enabled else name + "-artifact",
        "route_prefetch": {
            "alpha": 1.0,
            "direct_max_rows": 8,
            "enabled": enabled,
            "k": 3,
            "min_confidence": 0.125,
            "window_rows": 2,
        },
        "schema": benchmark.RESULT_SCHEMA,
        "seconds": seconds,
        "snapshot": {"manifest": "/private/shared.json"},
        "source_body_bytes": body,
        "trace": {"sha256": name + "-trace"},
    }
    return benchmark._result(identity)


class DirectDecodeBenchmarkTests(unittest.TestCase):
    def test_input_round_trip_is_digest_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            document = benchmark.build_input_document([2, 3, 5, 7], 11)
            benchmark._write_json(path, document)
            self.assertEqual(benchmark.load_input_document(path), document)

            tampered = dict(document)
            tampered["decode_token_id"] = 13
            benchmark._write_json(path, tampered)
            with self.assertRaisesRegex(benchmark.DirectDecodeError, "digest"):
                benchmark.load_input_document(path)

    def test_comparison_requires_bit_identity_and_matched_route_settings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            paths = {
                "baseline": directory / "baseline.json",
                "real": directory / "real.json",
                "placebo": directory / "placebo.json",
            }
            documents = {
                "baseline": _arm("baseline", seconds=10.0, body=1000),
                "real": _arm("real_markov", seconds=7.0, body=800),
                "placebo": _arm("placebo_markov", seconds=8.0, body=900),
            }
            for name, path in paths.items():
                benchmark._write_json(path, documents[name])
            args = type(
                "Args",
                (),
                {
                    "baseline": str(paths["baseline"]),
                    "real": str(paths["real"]),
                    "placebo": str(paths["placebo"]),
                },
            )()

            report = benchmark.compare_arms(args)
            self.assertEqual(
                report["contrasts"]["real_vs_baseline"]["seconds_ratio"], 0.7
            )
            self.assertEqual(
                report["contrasts"]["real_vs_placebo"]["seconds_ratio"],
                7 / 8,
            )
            self.assertAlmostEqual(
                report["contrasts"]["real_vs_baseline"]["causal_layer_seconds_ratio"],
                0.7,
            )
            self.assertEqual(report["arms"]["real_markov"]["reservoir_hits"], 1)

            changed = copy.deepcopy(documents["real"])
            changed["hidden_sha256"] = "x" * 64
            changed = benchmark._result(
                {key: value for key, value in changed.items() if key != "sha256"}
            )
            benchmark._write_json(paths["real"], changed)
            with self.assertRaisesRegex(benchmark.DirectDecodeError, "changed hidden"):
                benchmark.compare_arms(args)

            changed = copy.deepcopy(documents["real"])
            changed["route_prefetch"]["k"] = 6
            changed = benchmark._result(
                {key: value for key, value in changed.items() if key != "sha256"}
            )
            benchmark._write_json(paths["real"], changed)
            with self.assertRaisesRegex(benchmark.DirectDecodeError, "settings differ"):
                benchmark.compare_arms(args)

    def test_prepare_and_baseline_decode_share_real_transport_neutral_state(
        self,
    ) -> None:
        class Source(_TwoLayerPrefetchQuantizedTinyCheckpoint):
            def metrics(self):
                result = super().metrics()
                result["inventory_source_fingerprint"] = "a" * 64
                return result

            def close(self):
                return None

        config = replace(
            _config(0, n_layers=2),
            n_activated_experts=2,
            n_hash_layers=2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            input_path = directory / "input.json"
            snapshot = directory / "prefix.json"
            benchmark._write_json(
                input_path,
                benchmark.build_input_document([7, 8, 9, 10], 11),
            )
            prepare_args = benchmark._parser().parse_args(
                [
                    "prepare-prefix",
                    "--input",
                    str(input_path),
                    "--snapshot",
                    str(snapshot),
                    "--output",
                    str(directory / "prepare.json"),
                    "--device",
                    "cpu",
                    "--max-seq-len",
                    "8",
                    "--no-access-trace",
                ]
            )
            with (
                mock.patch.object(
                    benchmark,
                    "_build_source",
                    return_value=benchmark.RuntimeSource(Source()),
                ),
                mock.patch.object(benchmark, "_load_config", return_value=config),
                redirect_stderr(io.StringIO()),
            ):
                prepared = benchmark.prepare_prefix(prepare_args)
            self.assertEqual(prepared["arm"], "prepare")
            self.assertTrue(prepared["snapshot"]["transport_neutral"])

            decode_args = benchmark._parser().parse_args(
                [
                    "decode-arm",
                    "--input",
                    str(input_path),
                    "--snapshot",
                    str(snapshot),
                    "--output",
                    str(directory / "baseline.json"),
                    "--device",
                    "cpu",
                    "--max-seq-len",
                    "8",
                    "--no-access-trace",
                ]
            )
            with (
                mock.patch.object(
                    benchmark,
                    "_build_source",
                    return_value=benchmark.RuntimeSource(Source()),
                ),
                mock.patch.object(benchmark, "_load_config", return_value=config),
                redirect_stderr(io.StringIO()),
            ):
                decoded = benchmark.decode_arm(decode_args)
            self.assertEqual(decoded["arm"], "baseline")
            self.assertEqual(decoded["evidence"]["context_mode"], "decode")
            self.assertTrue(decoded["snapshot"]["restore"]["transport_neutral"])
            self.assertRegex(decoded["hidden_sha256"], r"^[0-9a-f]{64}$")

    def test_replicated_comparison_pairs_opposite_execution_orders(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cycles = (
                {
                    "baseline": _arm("baseline", seconds=10.0, body=1000),
                    "real": _arm("real_markov", seconds=7.0, body=800),
                    "placebo": _arm("placebo_markov", seconds=8.0, body=900),
                },
                {
                    "baseline": _arm("baseline", seconds=12.0, body=1000),
                    "real": _arm("real_markov", seconds=9.0, body=800),
                    "placebo": _arm("placebo_markov", seconds=8.0, body=900),
                },
            )
            paths = {arm: [] for arm in ("baseline", "real", "placebo")}
            for index, cycle in enumerate(cycles):
                for arm, document in cycle.items():
                    path = directory / f"{arm}-{index}.json"
                    benchmark._write_json(path, document)
                    paths[arm].append(str(path))
            args = type(
                "Args",
                (),
                {
                    "baseline": paths["baseline"],
                    "real": paths["real"],
                    "placebo": paths["placebo"],
                },
            )()

            report = benchmark.compare_replicates(args)
            self.assertEqual(report["cycle_count"], 2)
            self.assertEqual(
                report["paired"]["real_vs_placebo"]["seconds_deltas"],
                [-1.0, 1.0],
            )
            self.assertEqual(
                report["paired"]["real_vs_placebo"]["seconds_mean_delta"],
                0.0,
            )

    def test_snapshot_rebind_allows_only_audited_streamer_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            directory = Path(temporary)
            source_path = directory / "source.json"
            target_path = directory / "rebound.json"
            model = benchmark.StreamedDeepSeekV4(
                _config(0),
                benchmark.DeepSeekWeightPager(
                    _TwoLayerPrefetchQuantizedTinyCheckpoint(),
                    device="cpu",
                    compute_dtype="float32",
                ),
                max_seq_len=8,
            )
            model.prefill([[2, 3, 5, 7]], tokenwise=False)
            model.save_state(source_path, transport_neutral=True)
            document = json.loads(source_path.read_text(encoding="utf-8"))
            current = copy.deepcopy(document["body"]["identity"])
            streamer = next(
                row
                for row in current["runtime"]["sources"]
                if row["path"] == "immer/knowledge/streamer.py"
            )
            streamer["sha256"] = "f" * 64
            current["runtime"]["source_sha256"] = benchmark._sha256(
                current["runtime"]["sources"]
            )

            receipt = benchmark.rebind_transport_snapshot(
                source_path,
                target_path,
                current_identity=current,
            )
            self.assertEqual(
                receipt["changed_source_paths"], ["immer/knowledge/streamer.py"]
            )
            benchmark.read_snapshot(target_path, expected_identity=current)

            forbidden = copy.deepcopy(current)
            model_source = next(
                row
                for row in forbidden["runtime"]["sources"]
                if row["path"] == "immer/runtimes/deepseek_v4/model.py"
            )
            model_source["sha256"] = "e" * 64
            forbidden["runtime"]["source_sha256"] = benchmark._sha256(
                forbidden["runtime"]["sources"]
            )
            with self.assertRaisesRegex(
                benchmark.DirectDecodeError, "outside audited|transport-only"
            ):
                benchmark.rebind_transport_snapshot(
                    source_path,
                    directory / "forbidden.json",
                    current_identity=forbidden,
                )
            model.reset_state(release=True)
            model.pager.close()

    def test_parser_requires_explicit_subcommand(self) -> None:
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            benchmark._parser().parse_args([])
        args = benchmark._parser().parse_args(
            [
                "decode-arm",
                "--input",
                "input.json",
                "--snapshot",
                "prefix.json",
                "--output",
                "result.json",
            ]
        )
        self.assertEqual(args.route_prefetch_k, 3)
        self.assertEqual(args.route_prefetch_min_confidence, 0.125)


if __name__ == "__main__":
    unittest.main()
