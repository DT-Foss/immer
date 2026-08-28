from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from immer.runtimes.ooe.qwen_mlp_evidence import (
    CaptureManifest,
    QwenMlpEvidenceBank,
    canonical_capture_plan,
)

from test_ooe_qwen_mlp_evidence import _capture_pair, _verifier
from test_qwen38_o1_mlp_evidence import _hash


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_SCRIPT = ROOT / "scripts" / "ooe_markov_coordinate_selector.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MarkovCoordinateSelectorScriptTests(unittest.TestCase):
    def test_live_bank_fit_holdout_and_no_replace_report(self) -> None:
        selector = _load(SELECTOR_SCRIPT, "ooe_markov_coordinate_selector_test")
        prompts = tuple(_hash(f"selector-prompt:{index}") for index in range(5))
        manifest = CaptureManifest(
            model_pin_sha256=_hash("selector-model"),
            input_manifest_sha256=_hash("selector-inputs"),
            prompt_sha256s=prompts,
            entries=canonical_capture_plan(prompts),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank_root = root / "bank"
            bank = QwenMlpEvidenceBank(bank_root)
            for index in range(40):
                receipt, verification = _capture_pair(bank, manifest, index=index)
                bank.append_verified(receipt, _verifier(verification))
            output = root / "selector"
            arguments = [
                "--bank-root",
                str(bank_root),
                "--output-root",
                str(output),
                "--max-depth",
                "2",
                "--beam-width",
                "2",
                "--internal-validation-groups",
                "5",
                "--energy-candidates",
                "6",
                "--fisher-candidates",
                "6",
                "--variance-candidates",
                "6",
                "--random-candidates",
                "6",
                "--max-candidate-pool",
                "6",
                "--max-evaluated-states",
                "32",
                "--max-working-gb",
                "0.1",
            ]
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(selector.main(arguments), 0)
            report = json.loads(stream.getvalue())
            self.assertEqual(report, json.loads((output / "report.json").read_bytes()))
            self.assertEqual(report["schema"], selector.REPORT_SCHEMA)
            self.assertEqual(report["body"]["locked_raw_key_bits"], 64)
            self.assertTrue((output / "fit.json").is_file())
            self.assertTrue((output / "holdout.json").is_file())
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(selector.main(arguments), 0)


if __name__ == "__main__":
    unittest.main()
