from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from immer.resource_paths import (
    CRSA_ROUTER_V1,
    MANIFEST_NAMES,
    S3_SHIP_V6,
    crsa_router_manifest,
    manifest_path,
    s3_ship_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
SHIP_ARTIFACTS = (
    ROOT / "vendor" / "o1state" / "results" / "pos_ckpt.pt",
    ROOT / "vendor" / "o1state" / "s3_ship" / "organ_dual_donor.pt",
    ROOT / "vendor" / "o1state" / "s3_ship" / "organ_mul_donor.pt",
    ROOT / "vendor" / "o1state" / "s3_ship" / "organ_mod_kreis.pt",
    ROOT / "vendor" / "o1state" / "s3_ship" / "organ_dezimal.pt",
)


def _build_python() -> str | None:
    candidates = tuple(
        dict.fromkeys(
            candidate
            for candidate in (sys.executable, shutil.which("python"), shutil.which("python3"))
            if candidate
        )
    )
    probe = "import build, setuptools, wheel"
    for candidate in candidates:
        result = subprocess.run(
            [candidate, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return candidate
    return None


def _copy_build_source(destination: Path) -> None:
    for name in ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md"):
        shutil.copy2(ROOT / name, destination / name)
    shutil.copytree(ROOT / "src", destination / "src")
    manifest_dir = destination / "manifests"
    manifest_dir.mkdir()
    for name in MANIFEST_NAMES:
        shutil.copy2(ROOT / "manifests" / name, manifest_dir / name)


class ResourcePathTests(unittest.TestCase):
    def test_editable_checkout_prefers_canonical_root_manifests(self) -> None:
        self.assertEqual(s3_ship_manifest(), ROOT / "manifests" / S3_SHIP_V6)
        self.assertEqual(
            crsa_router_manifest(), ROOT / "manifests" / CRSA_ROUTER_V1
        )
        self.assertEqual(
            json.loads(s3_ship_manifest().read_text(encoding="utf-8"))["schema"],
            "immer.s3-ship-v6/v1",
        )
        self.assertEqual(
            json.loads(crsa_router_manifest().read_text(encoding="utf-8"))["schema"],
            "immer.crsa-a1-router/v1",
        )

    def test_locator_rejects_unknown_names_instead_of_traversing(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown IMMER manifest"):
            manifest_path("../pyproject.toml")

    def test_sdist_and_installed_wheel_contain_readable_manifest_copies(self) -> None:
        python = _build_python()
        if python is None:
            self.skipTest("no interpreter with build, setuptools and wheel is available")

        with tempfile.TemporaryDirectory() as tmp:
            temporary = Path(tmp)
            source = temporary / "source"
            source.mkdir()
            _copy_build_source(source)
            distributions = temporary / "dist"
            distributions.mkdir()

            subprocess.run(
                [
                    python,
                    "-m",
                    "build",
                    "--sdist",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    str(distributions),
                ],
                cwd=source,
                check=True,
                capture_output=True,
                text=True,
            )
            wheel_path = next(distributions.glob("immer-*.whl"))
            sdist_path = next(distributions.glob("immer-*.tar.gz"))

            with zipfile.ZipFile(wheel_path) as wheel:
                names = set(wheel.namelist())
                for name in MANIFEST_NAMES:
                    member = f"immer/resources/{name}"
                    self.assertIn(member, names)
                    packaged = wheel.read(member)
                    self.assertEqual(packaged, (ROOT / "manifests" / name).read_bytes())
                    document = json.loads(packaged.decode("utf-8"))
                    self.assertIn("schema", document)

            with tarfile.open(sdist_path, "r:gz") as sdist:
                names = set(sdist.getnames())
                prefix = sdist_path.name.removesuffix(".tar.gz")
                for name in MANIFEST_NAMES:
                    self.assertIn(f"{prefix}/manifests/{name}", names)

            target = temporary / "target"
            subprocess.run(
                [
                    python,
                    "-m",
                    "pip",
                    "install",
                    "--quiet",
                    "--no-deps",
                    "--target",
                    str(target),
                    str(wheel_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            probe = temporary / "probe"
            probe.mkdir()
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(target)
            environment["IMMER_TEST_TARGET"] = str(target)
            verification = subprocess.run(
                [
                    python,
                    "-c",
                    (
                        "import json, os, pathlib, immer; "
                        "from immer.resource_paths import MANIFEST_NAMES, manifest_path; "
                        "root=pathlib.Path(immer.__file__).resolve().parent; "
                        "expected=pathlib.Path(os.environ['IMMER_TEST_TARGET']).resolve(); "
                        "assert root.is_relative_to(expected); "
                        "paths=[manifest_path(name) for name in MANIFEST_NAMES]; "
                        "assert all(path.parent == root / 'resources' for path in paths); "
                        "assert all(json.loads(path.read_text(encoding='utf-8'))"
                        "['schema'] for path in paths); "
                        "print('\\n'.join(map(str, paths)))"
                    ),
                ],
                cwd=probe,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            for name in MANIFEST_NAMES:
                self.assertIn(f"immer/resources/{name}", verification.stdout)

            components = subprocess.run(
                [python, "-m", "immer", "components"],
                cwd=probe,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(components.returncode, 0, components.stderr)
            self.assertIn("WorldStream", components.stdout)

            if all(path.is_file() for path in SHIP_ARTIFACTS):
                artifact_root = temporary / "ship-artifacts"
                artifact_root.mkdir()
                for source_artifact in SHIP_ARTIFACTS:
                    destination = artifact_root / source_artifact.name
                    try:
                        os.link(source_artifact, destination)
                    except OSError:
                        shutil.copy2(source_artifact, destination)
                evaluation = subprocess.run(
                    [
                        python,
                        "-m",
                        "immer",
                        "eval",
                        "--artifact-root",
                        str(artifact_root),
                    ],
                    cwd=probe,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(evaluation.returncode, 0, evaluation.stderr)
                report = json.loads(evaluation.stdout.splitlines()[-1])
                self.assertTrue(report["passed"])
                self.assertEqual(report["correct"], 152)
                self.assertTrue(report["no_training"])


if __name__ == "__main__":
    unittest.main()
