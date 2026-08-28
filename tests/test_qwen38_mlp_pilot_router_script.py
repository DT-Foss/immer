from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from immer.runtimes.ooe.qwen_mlp_evidence import (
    CaptureManifest,
    QwenMlpEvidenceBank,
    canonical_capture_plan,
    run_capture_manifest,
)
from immer.runtimes.qwen3_8 import prompt_token_sha256

from test_qwen38_o1_mlp_evidence import _hash


ROOT = Path(__file__).resolve().parents[1]
ROUTER_SCRIPT = ROOT / "scripts" / "qwen38_mlp_pilot_router.py"
MLP_SCRIPT = ROOT / "scripts" / "qwen38_o1_mlp_evidence.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Qwen38MlpPilotRouterScriptTests(unittest.TestCase):
    def test_fit_is_durable_before_external_generation_is_opened(self) -> None:
        router = _load(ROUTER_SCRIPT, "qwen38_mlp_pilot_router_test")
        mlp = _load(MLP_SCRIPT, "qwen38_mlp_pilot_fixture")

        class LiveFixtureRunner(mlp.FixtureRunner):
            capture_mode = "live-exact"

        fixture = {
            "hidden_dimension": 4,
            "intermediate_dimension": 6,
            "rows": 5,
            "schema": mlp.FIXTURE_SCHEMA,
            "seed_sha256": _hash("pilot-fixture"),
        }

        def build(root: Path, prefix: str) -> tuple[Path, Path]:
            token_rows = tuple((1, 2, 1000 + index, 8, 9) for index in range(5))
            if prefix == "holdout":
                token_rows = tuple((1, 2, 2000 + index, 8, 9) for index in range(5))
            prompts = tuple(prompt_token_sha256(tokens) for tokens in token_rows)
            manifest = CaptureManifest(
                model_pin_sha256=_hash("pilot-model"),
                input_manifest_sha256=_hash(f"pilot-input:{prefix}"),
                prompt_sha256s=prompts,
                entries=canonical_capture_plan(prompts),
            )
            bank_root = root / f"{prefix}-bank"
            bank = QwenMlpEvidenceBank(bank_root)
            run_capture_manifest(manifest, bank, LiveFixtureRunner(manifest, fixture))
            registry = root / f"{prefix}-prompts.json"
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
            return bank_root, registry

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fit_bank, fit_registry = build(root, "fit")
            holdout_bank, holdout_registry = build(root, "holdout")
            output = root / "router"
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                self.assertEqual(
                    router.main(
                        [
                            "--fit-bank-root",
                            str(fit_bank),
                            "--fit-prompt-registry",
                            str(fit_registry),
                            "--output-root",
                            str(output),
                            "--external-holdout-bank-root",
                            str(holdout_bank),
                            "--holdout-prompt-registry",
                            str(holdout_registry),
                            "--block-size",
                            "2",
                            "--pilot-count",
                            "1",
                            "--selected-block-count",
                            "1",
                            "--max-working-gb",
                            "0.1",
                        ]
                    ),
                    0,
                )
            report = json.loads(stream.getvalue())
            self.assertEqual(report["body"]["status"], "external_generation_evaluated")
            self.assertTrue((output / "fit.json").is_file())
            self.assertTrue((output / "fit-row-roles.json").is_file())
            self.assertTrue((output / "evaluation.json").is_file())
            self.assertTrue((output / "holdout-row-roles.json").is_file())
            self.assertTrue((output / "report.json").is_file())


if __name__ == "__main__":
    unittest.main()
