from __future__ import annotations

import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from test_qwen3_8_model import _tiny_config, _tiny_config_mapping, _tiny_weights


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "qwen38_stream_smoke.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen38_stream_smoke", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import Qwen3.8 stream smoke")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = _load_script()


class Qwen38StreamSmokeTests(unittest.TestCase):
    def test_help_does_not_touch_network_or_weights(self) -> None:
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                smoke._parser().parse_args(["--help"])
        self.assertEqual(caught.exception.code, 0)

    def test_token_id_parser_is_strict(self) -> None:
        self.assertEqual(smoke._token_ids("1, 2,3"), (1, 2, 3))
        for invalid in ("", "1,-2", "1,nope"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(Exception):
                    smoke._token_ids(invalid)

    def test_activation_summary_is_json_ready_and_detects_nonfinite(self) -> None:
        finite = smoke._activation_summary(
            torch.tensor([[[1.0, -1.0], [3.0, -3.0]]])
        )
        self.assertEqual(finite["shape"], [1, 2, 2])
        self.assertTrue(finite["all_finite"])
        self.assertAlmostEqual(finite["rms"], 5**0.5)
        bad = smoke._activation_summary(torch.tensor([[[float("nan")]]]))
        self.assertFalse(bad["all_finite"])

    def test_local_fixture_executes_one_layer_and_atomically_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            config = root / "config.json"
            output = root / "result.json"
            config.write_text(json.dumps(_tiny_config_mapping()), encoding="utf-8")
            save_file(_tiny_weights(_tiny_config()), source / "model.safetensors")
            args = smoke._parser().parse_args(
                [
                    "--source-dir",
                    str(source),
                    "--token-ids",
                    "1",
                    "--config",
                    str(config),
                    "--cache-dir",
                    str(root / "cache"),
                    "--output",
                    str(output),
                    "--device",
                    "cpu",
                    "--compute-dtype",
                    "float32",
                    "--budget-mb",
                    "20",
                    "--cache-mb",
                    "20",
                ]
            )

            report, written = smoke.run(args)

            self.assertEqual(written, output.resolve())
            self.assertEqual(report["execution"]["completed_layers"], 1)
            self.assertTrue(report["execution"]["activation"]["all_finite"])
            self.assertEqual(report["execution"]["activation"]["shape"], [1, 1, 12])
            self.assertGreater(report["execution"]["source_body_bytes"], 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), report)


if __name__ == "__main__":
    unittest.main()
