from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from immer.hf_export import HfExportError, export_hf_poc


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "s3_ship_v6.json"
SOURCE_CHECKPOINT = ROOT / "vendor" / "o1state" / "results" / "pos_ckpt.pt"
SOURCE_ORGANS = (
    ROOT / "vendor" / "o1state" / "s3_ship" / "organ_dual_donor.pt",
    ROOT / "vendor" / "o1state" / "s3_ship" / "organ_mul_donor.pt",
    ROOT / "vendor" / "o1state" / "s3_ship" / "organ_mod_kreis.pt",
    ROOT / "vendor" / "o1state" / "s3_ship" / "organ_dezimal.pt",
)
HAS_EXPORT_INPUTS = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("safetensors") is not None
    and SOURCE_CHECKPOINT.is_file()
    and all(path.is_file() for path in SOURCE_ORGANS)
)


def _isolated(script: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name == "PYTHONPATH" or name.startswith("IMMER_"):
            environment.pop(name, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, "-I", "-B", str(script), *arguments],
        cwd=script.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


@unittest.skipUnless(
    HAS_EXPORT_INPUTS,
    "local ignored SHIP-v6 artifacts, torch, or safetensors are unavailable",
)
class OfflineHfExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.bundle = cls.root / "bundle"
        cls.report = export_hf_poc(cls.bundle, manifest=MANIFEST)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_bundle_shape_a1_only_state_and_unchanged_organs(self) -> None:
        required = {
            "README.md",
            "checksums.json",
            "config.json",
            "crsa_router_v1.json",
            "model.safetensors",
            "requirements.txt",
            "s3_ship_v6.json",
            "solve.py",
            "verify.py",
        }
        self.assertTrue(required.issubset({path.name for path in self.bundle.iterdir()}))
        self.assertLess(
            (self.bundle / "model.safetensors").stat().st_size,
            SOURCE_CHECKPOINT.stat().st_size // 2,
        )

        from safetensors.torch import load_file

        state = load_file(str(self.bundle / "model.safetensors"), device="cpu")
        self.assertEqual(len(state), 29)
        self.assertIn("embed.weight", state)
        self.assertIn("head.weight", state)
        self.assertFalse(any(name.startswith("arms.") for name in state))
        self.assertFalse(any("optimizer" in name for name in state))

        rebased = json.loads((self.bundle / "s3_ship_v6.json").read_text(encoding="utf-8"))
        host = rebased["s3"]["host"]
        self.assertEqual(host["artifact"], "model.safetensors")
        self.assertEqual(host["format"], "safetensors-state-dict")
        self.assertEqual(
            {row["artifact"] for row in rebased["organs"]},
            {f"organs/{path.name}" for path in SOURCE_ORGANS},
        )
        for source in SOURCE_ORGANS:
            self.assertEqual((self.bundle / "organs" / source.name).read_bytes(), source.read_bytes())

        runtime_files = tuple((self.bundle / "runtime" / "immer").rglob("*"))
        self.assertFalse(any("__pycache__" in path.parts for path in runtime_files))
        self.assertFalse(any(path.suffix in {".pyc", ".pyo"} for path in runtime_files))
        self.assertFalse(any(path.name == ".DS_Store" for path in runtime_files))
        card = (self.bundle / "README.md").read_text(encoding="utf-8")
        self.assertIn("license: other", card)
        self.assertIn("LICENSE STATUS: NOT CLEARED", card)
        self.assertFalse(self.report["public_upload_allowed"])

    def test_generated_verifier_and_exact_cascade_are_checkout_isolated(self) -> None:
        verification = _isolated(self.bundle / "verify.py")
        self.assertEqual(verification.returncode, 0, verification.stderr)
        report = json.loads(verification.stdout)
        self.assertEqual(report["arithmetic"], "152/152")
        self.assertEqual(report["routes"], "152/152")
        self.assertEqual(
            report["router_state_digest"],
            "561db8fc50ea029288f318eb9f7c206bd5c007757dbdd60843f1a03990efd819",
        )
        self.assertEqual(report["runtime_origin"], "runtime/immer/__init__.py")

        solved = _isolated(self.bundle / "solve.py", "three plus five is")
        self.assertEqual(solved.returncode, 0, solved.stderr)
        answer = json.loads(solved.stdout)
        self.assertEqual(answer["status"], "ok")
        self.assertEqual(answer["component"], "immer.exact-cascade")
        self.assertEqual(answer["output"], "eight")
        self.assertEqual(tuple(answer["evidence"]["order"]), ("s3", "fertig"))

    def test_checksum_payload_is_deterministic_and_tampering_is_rejected(self) -> None:
        second = self.root / "deterministic-second"
        export_hf_poc(second, manifest=MANIFEST)
        self.assertEqual(
            (self.bundle / "checksums.json").read_bytes(),
            (second / "checksums.json").read_bytes(),
        )
        self.assertEqual(
            (self.bundle / "model.safetensors").read_bytes(),
            (second / "model.safetensors").read_bytes(),
        )

        tampered = self.root / "tampered"
        shutil.copytree(self.bundle, tampered)
        victim = tampered / "organs" / SOURCE_ORGANS[0].name
        victim.write_bytes(victim.read_bytes() + b"tamper")
        rejected = _isolated(tampered / "verify.py")
        self.assertNotEqual(rejected.returncode, 0)
        error = json.loads(rejected.stderr)
        self.assertEqual(error["status"], "error")
        self.assertIn("SHA-256 mismatch", error["reason"])

        extra = self.root / "extra-file"
        shutil.copytree(self.bundle, extra)
        (extra / "untracked.txt").write_text("not in exact file set", encoding="utf-8")
        rejected_extra = _isolated(extra / "verify.py")
        self.assertNotEqual(rejected_extra.returncode, 0)
        self.assertIn("file set mismatch", rejected_extra.stderr)

    def test_existing_target_refusal_and_recoverable_replace(self) -> None:
        target = self.root / "existing"
        target.mkdir()
        marker = target / "user-data.txt"
        marker.write_text("preserve me", encoding="utf-8")
        with self.assertRaisesRegex(HfExportError, "was not touched"):
            export_hf_poc(target, manifest=MANIFEST)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me")

        report = export_hf_poc(target, manifest=MANIFEST, replace=True)
        backup = Path(report["backup"])
        self.assertTrue((target / "verify.py").is_file())
        self.assertEqual((backup / "user-data.txt").read_text(encoding="utf-8"), "preserve me")
        self.assertFalse(report["public_upload_allowed"])

    def test_unsafe_or_recursive_targets_are_rejected_before_staging(self) -> None:
        with self.assertRaisesRegex(HfExportError, "unsafe export target"):
            export_hf_poc(Path.cwd() / "new-directory" / "..", manifest=MANIFEST)

        package_target = ROOT / "src" / "immer" / "recursive-export"
        with self.assertRaisesRegex(HfExportError, "inside the IMMER source package"):
            export_hf_poc(package_target, manifest=MANIFEST)
        self.assertFalse(package_target.exists())


if __name__ == "__main__":
    unittest.main()
