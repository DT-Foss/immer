from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from immer.artifacts import ArtifactBootstrapError, import_artifacts, load_artifact_specs


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


class ArtifactBootstrapTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, bytes, bytes]:
        target = root / "target"
        source = root / "read-only-source"
        (target / "manifests").mkdir(parents=True)
        (target / "src" / "immer").mkdir(parents=True)
        (target / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
        (source / "results").mkdir(parents=True)
        (source / "s3_ship").mkdir(parents=True)
        host_body = b"frozen-host-fixture"
        organ_body = b"cold-organ-fixture"
        (source / "results" / "host.pt").write_bytes(host_body)
        (source / "s3_ship" / "organ.pt").write_bytes(organ_body)
        manifest = target / "manifests" / "ship.json"
        manifest.write_text(
            json.dumps(
                {
                    "organs": [
                        {
                            "name": "fixture",
                            "capability": "fixture",
                            "group": "fixture",
                            "artifact": "../vendor/o1state/s3_ship/organ.pt",
                            "sha256": _digest(organ_body),
                        }
                    ],
                    "s3": {
                        "host": {
                            "name": "host",
                            "artifact": "../vendor/o1state/results/host.pt",
                            "sha256": _digest(host_body),
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest, source, host_body, organ_body

    def test_dry_run_then_atomic_import_and_zero_copy_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, source, host_body, organ_body = self._fixture(Path(tmp))
            dry = import_artifacts(manifest, source, dry_run=True)
            self.assertEqual(
                {row["status"] for row in dry["artifacts"]},
                {"would_install"},
            )
            _, specs = load_artifact_specs(manifest)
            self.assertFalse(any(spec.destination.exists() for spec in specs))

            installed = import_artifacts(manifest, source)
            self.assertEqual(
                {row["status"] for row in installed["artifacts"]},
                {"installed"},
            )
            bodies = {spec.destination.name: spec.destination.read_bytes() for spec in specs}
            self.assertEqual(bodies, {"host.pt": host_body, "organ.pt": organ_body})

            resumed = import_artifacts(manifest, source)
            self.assertEqual(
                {row["status"] for row in resumed["artifacts"]},
                {"verified"},
            )

    def test_wrong_existing_destination_is_preserved_unless_replace_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, source, _, _ = self._fixture(Path(tmp))
            import_artifacts(manifest, source)
            _, specs = load_artifact_specs(manifest)
            victim = next(spec.destination for spec in specs if spec.destination.name == "organ.pt")
            victim.write_bytes(b"do-not-silently-overwrite")

            with self.assertRaisesRegex(ArtifactBootstrapError, "was not touched"):
                import_artifacts(manifest, source)
            self.assertEqual(victim.read_bytes(), b"do-not-silently-overwrite")

            report = import_artifacts(manifest, source, replace=True)
            self.assertIn("replaced", {row["status"] for row in report["artifacts"]})

    def test_manifest_cannot_write_outside_its_target_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest, _, _, _ = self._fixture(Path(tmp))
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["organs"][0]["artifact"] = "../../escape.pt"
            manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ArtifactBootstrapError, "leaves target root"):
                load_artifact_specs(manifest)

    def test_explicit_artifact_root_rebases_legacy_paths_by_logical_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, source, _, _ = self._fixture(root)
            external = root / "external-cache"
            report = import_artifacts(manifest, source, configured_root=external)
            self.assertEqual(Path(report["target_root"]), external.resolve())
            self.assertEqual(
                {path.name for path in external.iterdir()},
                {"host.pt", "organ.pt"},
            )

    def test_explicit_artifact_root_preserves_safe_manifest_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, source, _, organ_body = self._fixture(root)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["organs"][0]["artifact"] = "organs/organ.pt"
            manifest.write_text(json.dumps(data), encoding="utf-8")

            external = root / "external-cache"
            import_artifacts(manifest, source, configured_root=external)
            self.assertEqual((external / "organs" / "organ.pt").read_bytes(), organ_body)

    def test_explicit_artifact_root_rejects_unrecognized_parent_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, source, _, _ = self._fixture(root)
            data = json.loads(manifest.read_text(encoding="utf-8"))
            data["organs"][0]["artifact"] = "../outside/organ.pt"
            manifest.write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaisesRegex(ArtifactBootstrapError, "traverses outside"):
                import_artifacts(
                    manifest,
                    source,
                    configured_root=root / "external-cache",
                )


if __name__ == "__main__":
    unittest.main()
