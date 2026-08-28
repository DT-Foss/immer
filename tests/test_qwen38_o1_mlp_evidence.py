from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from immer.runtimes.ooe.qwen_mlp_evidence import (
    CaptureManifest,
    QwenMlpEvidenceBank,
    QwenMlpEvidenceIntegrityError,
    canonical_capture_plan,
    run_capture_manifest,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qwen38_o1_mlp_evidence.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("qwen38_o1_mlp_evidence", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _hash(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


class Qwen38O1MlpEvidenceScriptTests(unittest.TestCase):
    def test_manifest_is_exact_25_10_5_and_fixture_resume_is_status_only(self) -> None:
        module = _load_script()
        prompts = tuple(_hash(f"prompt:{index}") for index in range(5))
        manifest = CaptureManifest(
            model_pin_sha256=_hash("model-pin"),
            input_manifest_sha256=_hash("input-manifest"),
            prompt_sha256s=prompts,
            entries=canonical_capture_plan(prompts),
        )
        self.assertEqual(len(manifest.entries), 40)
        self.assertEqual(
            tuple(
                sum(row.split == split for row in manifest.entries)
                for split in ("train", "calibration", "holdout")
            ),
            (25, 10, 5),
        )
        self.assertEqual(
            tuple(
                dict.fromkeys(
                    row.layer for row in manifest.entries if row.split == "train"
                )
            ),
            (45, 18, 36, 63, 27),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            fixture_path = root / "fixture.json"
            manifest_path.write_bytes(manifest.to_bytes())
            fixture_path.write_text(
                json.dumps(
                    {
                        "hidden_dimension": 4,
                        "intermediate_dimension": 6,
                        "rows": 2,
                        "schema": module.FIXTURE_SCHEMA,
                        "seed_sha256": _hash("fixture-seed"),
                    }
                )
            )
            bank_root = root / "bank"
            status_path = root / "first-status.json"
            outputs = []
            for index in range(2):
                arguments = [
                    "--root",
                    str(bank_root),
                    "--manifest",
                    str(manifest_path),
                    "--fixture",
                    str(fixture_path),
                ]
                if index == 0:
                    arguments.extend(("--output", str(status_path)))
                stream = io.StringIO()
                with contextlib.redirect_stdout(stream):
                    self.assertEqual(
                        module.main(arguments),
                        0,
                    )
                outputs.append(json.loads(stream.getvalue()))
            self.assertEqual(json.loads(status_path.read_bytes()), outputs[0])
            self.assertEqual(outputs[0]["new_publications"], 40)
            self.assertEqual(outputs[1]["new_publications"], 0)
            self.assertEqual(outputs[1]["split_counts"], [25, 10, 5])
            self.assertEqual(outputs[1]["receipt_count"], 40)
            self.assertTrue(outputs[1]["audit_clean"])
            rendered = json.dumps(outputs)
            self.assertNotIn("data_base64", rendered)
            self.assertEqual(
                set(outputs[1]),
                {
                    "audit_clean",
                    "budget_sha256",
                    "fixture",
                    "manifest_sha256",
                    "new_publications",
                    "receipt_count",
                    "referenced_tensor_bytes",
                    "schema",
                    "split_counts",
                    "state_sha256",
                },
            )
            bank = QwenMlpEvidenceBank(bank_root)

            class LiveFixtureRunner(module.FixtureRunner):
                capture_mode = "live-exact"

            fixture = json.loads(fixture_path.read_text())
            holdout_bank = QwenMlpEvidenceBank(root / "holdout-first-bank")
            with self.assertRaisesRegex(
                QwenMlpEvidenceIntegrityError, "25/10/5 state machine"
            ):
                run_capture_manifest(
                    manifest,
                    holdout_bank,
                    module.FixtureRunner(manifest, fixture),
                    allowed_splits=("holdout",),
                )
            self.assertEqual(holdout_bank.state().receipt_count, 0)
            self.assertTrue(holdout_bank.audit().clean)
            with self.assertRaisesRegex(
                QwenMlpEvidenceIntegrityError, "25/10/5 state machine"
            ):
                run_capture_manifest(
                    manifest,
                    bank,
                    LiveFixtureRunner(manifest, fixture),
                )
            self.assertEqual(bank.state().receipt_count, 40)
            self.assertTrue(bank.audit().clean)
            live_bank = QwenMlpEvidenceBank(root / "live-bank")
            live_publications = run_capture_manifest(
                manifest,
                live_bank,
                LiveFixtureRunner(manifest, fixture),
            )
            self.assertEqual(len(live_publications), 40)
            self.assertEqual(live_bank.state().split_counts, (25, 10, 5))
            self.assertEqual(len(live_bank.build_subspace_corpus().groups), 40)
            self.assertEqual(
                len(
                    live_bank.build_subspace_corpus(
                        allowed_splits=("train", "calibration")
                    ).groups
                ),
                35,
            )
            self.assertEqual(
                len(
                    live_bank.build_subspace_corpus(allowed_splits=("holdout",)).groups
                ),
                5,
            )
            holdout_receipt = next(
                receipt
                for receipt, _verification in live_bank.committed_pairs()
                if receipt.entry.split == "holdout"
            )
            holdout_ref = holdout_receipt.tensors[0]
            holdout_path = (
                root
                / "live-bank"
                / "objects"
                / holdout_ref.object_sha256[:2]
                / f"{holdout_ref.object_sha256}.tensor"
            )
            tampered = bytearray(holdout_path.read_bytes())
            tampered[-1] ^= 1
            holdout_path.chmod(0o600)
            holdout_path.write_bytes(tampered)
            deferred = QwenMlpEvidenceBank(
                root / "live-bank", deferred_tensor_splits=("holdout",)
            )
            self.assertEqual(
                len(
                    deferred.build_subspace_corpus(
                        allowed_splits=("train", "calibration")
                    ).groups
                ),
                35,
            )
            with self.assertRaises(QwenMlpEvidenceIntegrityError):
                deferred.audit()
            with self.assertRaisesRegex(
                QwenMlpEvidenceIntegrityError,
                "one live manifest/model/verifier pin",
            ):
                bank.build_subspace_corpus()

            orphan = bank.publish_tensor(
                "mlp.input", 63, np.ones((1, 1, 3), dtype=np.float32)
            )
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                module.main(
                    [
                        "--root",
                        str(bank_root),
                        "--manifest",
                        str(manifest_path),
                        "--fixture",
                        str(fixture_path),
                    ]
                )
            orphan_status = json.loads(stream.getvalue())
            self.assertFalse(orphan_status["audit_clean"])
            self.assertEqual(orphan_status["new_publications"], 0)
            self.assertIn(orphan.object_sha256, bank.audit().orphan_objects)

    def test_manifest_roundtrip_and_tampered_plan_rejected(self) -> None:
        prompts = tuple(_hash(f"prompt:{index}") for index in range(5))
        manifest = CaptureManifest(
            _hash("model"),
            _hash("inputs"),
            prompts,
            canonical_capture_plan(prompts),
        )
        self.assertEqual(
            CaptureManifest.from_bytes(manifest.to_bytes()).to_bytes(),
            manifest.to_bytes(),
        )
        document = json.loads(manifest.to_bytes())
        document["body"]["entries"][0]["split"] = "holdout"
        document["body_sha256"] = _hash_from_body(document["body"])
        with self.assertRaises((ValueError, QwenMlpEvidenceIntegrityError)):
            CaptureManifest.from_bytes(_canonical(document))

    def test_exact_prompt_cohort_selects_five_hashes(self) -> None:
        module = _load_script()
        prompts = tuple(_hash(f"cohort-prompt:{index}") for index in range(5))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cohort = root / "prompts.json"
            cohort.write_text(
                json.dumps(
                    {
                        "prompts": [
                            {
                                "sha256": prompt,
                                "spec_defaults": {},
                                "token_ids": [index + 1],
                            }
                            for index, prompt in enumerate(reversed(prompts))
                        ]
                    }
                )
            )
            args = module._parser().parse_args(
                [
                    "--root",
                    str(root / "bank"),
                    "--cartography-root",
                    str(root / "cartography"),
                    "--prompt-cohort",
                    str(cohort),
                    "--authority-only",
                ]
            )
            self.assertEqual(
                module._requested_prompts(args, root / "bank"), tuple(sorted(prompts))
            )


def _canonical(value: object) -> bytes:
    from immer.runtimes.ooe.identity import canonical_json_bytes

    return canonical_json_bytes(value)


def _hash_from_body(value: object) -> str:
    import hashlib

    return hashlib.sha256(_canonical(value)).hexdigest()


if __name__ == "__main__":
    unittest.main()
