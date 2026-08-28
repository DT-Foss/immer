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
    run_capture_manifest,
)
from immer.runtimes.qwen3_8 import prompt_token_sha256

from test_ooe_qwen_mlp_evidence import _capture_pair, _verifier
from test_qwen38_o1_mlp_evidence import _hash


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_SCRIPT = ROOT / "scripts" / "ooe_markov_coordinate_selector.py"
MLP_SCRIPT = ROOT / "scripts" / "qwen38_o1_mlp_evidence.py"


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

            external_prompts = tuple(
                _hash(f"external-selector-prompt:{index}") for index in range(5)
            )
            external_manifest = CaptureManifest(
                model_pin_sha256=manifest.model_pin_sha256,
                input_manifest_sha256=_hash("external-selector-inputs"),
                prompt_sha256s=external_prompts,
                entries=canonical_capture_plan(external_prompts),
            )
            external_bank_root = root / "external-bank"
            external_bank = QwenMlpEvidenceBank(external_bank_root)
            for index in range(40):
                receipt, verification = _capture_pair(
                    external_bank, external_manifest, index=index
                )
                external_bank.append_verified(receipt, _verifier(verification))
            external_output = root / "external-selector"
            external_arguments = [
                *arguments,
                "--external-holdout-bank-root",
                str(external_bank_root),
            ]
            external_arguments[external_arguments.index(str(output))] = str(
                external_output
            )
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(selector.main(external_arguments), 0)
            external_report = json.loads(stream.getvalue())
            self.assertTrue(external_report["body"]["external_holdout"])
            self.assertNotEqual(
                external_report["body"]["fit_bank_state_sha256"],
                external_report["body"]["holdout_bank_state_sha256"],
            )

    def test_content_rows_abstain_when_exact_reuse_ceiling_is_zero(self) -> None:
        selector = _load(SELECTOR_SCRIPT, "ooe_markov_coordinate_selector_content_test")
        mlp = _load(MLP_SCRIPT, "qwen38_o1_mlp_content_fixture")
        token_rows = tuple((1, 2, 10 + index, 8, 9) for index in range(5))
        prompts = tuple(prompt_token_sha256(tokens) for tokens in token_rows)
        manifest = CaptureManifest(
            model_pin_sha256=_hash("content-model"),
            input_manifest_sha256=_hash("content-inputs"),
            prompt_sha256s=prompts,
            entries=canonical_capture_plan(prompts),
        )

        class LiveFixtureRunner(mlp.FixtureRunner):
            capture_mode = "live-exact"

        fixture = {
            "hidden_dimension": 4,
            "intermediate_dimension": 6,
            "rows": 5,
            "schema": mlp.FIXTURE_SCHEMA,
            "seed_sha256": _hash("content-fixture"),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank_root = root / "bank"
            bank = QwenMlpEvidenceBank(bank_root)
            run_capture_manifest(manifest, bank, LiveFixtureRunner(manifest, fixture))
            registry = root / "prompts.json"
            registry.write_text(
                json.dumps(
                    {
                        "prompts": [
                            {"sha256": prompt, "token_ids": list(tokens)}
                            for prompt, tokens in zip(prompts, token_rows, strict=True)
                        ]
                    }
                )
            )
            output = root / "selector"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(
                    selector.main(
                        [
                            "--bank-root",
                            str(bank_root),
                            "--output-root",
                            str(output),
                            "--row-mode",
                            "content",
                            "--fit-prompt-registry",
                            str(registry),
                            "--max-depth",
                            "2",
                            "--beam-width",
                            "2",
                            "--max-candidate-pool",
                            "6",
                            "--max-evaluated-states",
                            "32",
                            "--max-working-gb",
                            "0.1",
                        ]
                    ),
                    0,
                )
            report = json.loads(stream.getvalue())
            self.assertEqual(report["body"]["status"], "no_exact_output_reuse_signal")
            self.assertEqual(
                report["body"]["calibration_reuse_ceiling"]["adaptive_exact_hits"],
                0,
            )
            self.assertFalse((output / "fit.json").exists())
            self.assertFalse((output / "holdout.json").exists())
            self.assertTrue((output / "fit-row-roles.json").is_file())
            self.assertTrue((output / "reuse-ceiling.json").is_file())


if __name__ == "__main__":
    unittest.main()
