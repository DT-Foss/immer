from __future__ import annotations

import io
import importlib.util
import json
import struct
import builtins
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np

from immer.cli import main


ROOT = Path(__file__).resolve().parents[1]
HAS_SHIP = importlib.util.find_spec("torch") is not None and all(
    path.is_file()
    for path in (
        ROOT / "vendor/o1state/results/pos_ckpt.pt",
        ROOT / "vendor/o1state/s3_ship/organ_dual_donor.pt",
        ROOT / "vendor/o1state/s3_ship/organ_mul_donor.pt",
        ROOT / "vendor/o1state/s3_ship/organ_mod_kreis.pt",
        ROOT / "vendor/o1state/s3_ship/organ_dezimal.pt",
    )
)


def _write_safetensors_fixture(root: Path) -> np.ndarray:
    values = np.array([[1.25, -2.5], [3.0, 4.5], [8.0, -0.125]], dtype="<f4")
    body = values.tobytes()
    header = {
        "fixture.weight": {
            "dtype": "F32",
            "shape": list(values.shape),
            "data_offsets": [0, len(body)],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + body
    )
    return values


class CliTests(unittest.TestCase):
    def test_components_command_reports_integrated_code(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["components"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("FERTIG", text)
        self.assertIn("CRSA", text)
        self.assertIn("LiveCausal", text)
        self.assertIn("CausalWeights", text)
        self.assertIn("DeepSeekV4", text)
        self.assertIn("MarkovRouter", text)
        self.assertIn("OrganBank", text)
        self.assertNotIn("FLCA", text)
        self.assertNotIn("QAD", text)

    def test_doctor_worldstream_does_not_depend_on_requests(self) -> None:
        original_import = builtins.__import__

        def import_without_requests(name, *args, **kwargs):
            if name == "requests" or name.startswith("requests."):
                raise AssertionError("WorldStream must not import requests")
            return original_import(name, *args, **kwargs)

        out = io.StringIO()
        with patch("builtins.__import__", side_effect=import_without_requests), redirect_stdout(out):
            main(["doctor"])
        self.assertIn("✓ WorldStream", out.getvalue())

    def test_organs_uses_the_bundled_manifest_by_default(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["organs", "list"])
        self.assertEqual(code, 0)
        self.assertIn("arith-dual", out.getvalue())
        self.assertIn("z3-circle", out.getvalue())

    def test_stream_cli_is_offline_budgeted_and_reads_exact_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            root.mkdir()
            values = _write_safetensors_fixture(root)

            inventory_out = io.StringIO()
            with redirect_stdout(inventory_out):
                inventory_code = main([
                    "stream",
                    str(root),
                    "--local",
                    "--cache-dir",
                    str(cache),
                    "--budget-mb",
                    "1",
                ])
            inventory = json.loads(inventory_out.getvalue())
            self.assertEqual(inventory_code, 0)
            self.assertEqual(inventory["tensor_count"], 1)
            self.assertEqual(inventory["tensors"], ["fixture.weight"])

            rows_out = io.StringIO()
            with redirect_stdout(rows_out):
                rows_code = main([
                    "stream",
                    str(root),
                    "--local",
                    "--cache-dir",
                    str(cache),
                    "--budget-mb",
                    "1",
                    "--tensor",
                    "fixture.weight",
                    "--start-row",
                    "1",
                    "--rows",
                    "2",
                ])
            rows = json.loads(rows_out.getvalue())
            self.assertEqual(rows_code, 0)
            self.assertEqual(rows["shape"], [2, 2])
            np.testing.assert_array_equal(rows["preview"], values[1:3])
            self.assertEqual(rows["metrics"]["inventory_cache_hits"], 1)

    @unittest.skipUnless(HAS_SHIP, "local ignored SHIP-v6 artifacts or torch are unavailable")
    def test_solve_and_eval_reach_the_cold_runtime(self) -> None:
        solve_out = io.StringIO()
        with redirect_stdout(solve_out):
            solve_code = main(["solve", "Was ist drei plus fünf?"])
        payload = json.loads(solve_out.getvalue().splitlines()[-1])
        self.assertEqual(solve_code, 0)
        self.assertEqual(payload["output"], "eight")
        self.assertEqual(payload["evidence"]["route"], "s3_crystal")

        eval_out = io.StringIO()
        with redirect_stdout(eval_out):
            eval_code = main(["eval"])
        report = json.loads(eval_out.getvalue().splitlines()[-1])
        self.assertEqual(eval_code, 0)
        self.assertTrue(report["passed"])
        self.assertEqual(report["correct"], 152)

    @unittest.skipUnless(HAS_SHIP, "local ignored SHIP-v6 artifacts or torch are unavailable")
    def test_serve_answers_exact_math_and_persists_the_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "life.json"
            out = io.StringIO()
            turns = (
                "Was ist drei plus fünf?\n"
                "help\n"
                "do monthly report\n"
                "/quit\n"
            )
            with patch("sys.stdin", io.StringIO(turns)), redirect_stdout(out):
                code = main([
                    "serve",
                    "--state",
                    str(state),
                    "--no-dashboard",
                ])
            self.assertEqual(code, 0)
            self.assertIn("[exact] eight", out.getvalue())
            self.assertIn("[fertig:help]", out.getvalue())
            self.assertIn("explicitly injected desktop backend", out.getvalue())
            persisted = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(persisted["state"]["turns"], 3)


if __name__ == "__main__":
    unittest.main()
