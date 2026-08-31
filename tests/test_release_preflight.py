from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "release_preflight",
    ROOT / "scripts" / "release_preflight.py",
)
assert SPEC is not None and SPEC.loader is not None
release_preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_preflight)


def _add_tar(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


class ReleasePreflightTests(unittest.TestCase):
    def test_clean_v1_wheel_and_sdist_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = root / "immer-1.0.0-py3-none-any.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("immer/__init__.py", '__version__ = "1.0.0"\n')
                archive.writestr(
                    "immer-1.0.0.dist-info/METADATA",
                    "Name: immer\nVersion: 1.0.0\n",
                )
            sdist = root / "immer-1.0.0.tar.gz"
            with tarfile.open(sdist, "w:gz") as archive:
                _add_tar(archive, "immer-1.0.0/README.md", b"# IMMER\n")
                _add_tar(
                    archive,
                    "immer-1.0.0/src/immer/__init__.py",
                    b'__version__ = "1.0.0"\n',
                )

            wheel_report = release_preflight.inspect_distribution(wheel)
            sdist_report = release_preflight.inspect_distribution(sdist)

        self.assertEqual(wheel_report["kind"], "wheel")
        self.assertEqual(sdist_report["kind"], "sdist")

    def test_model_payload_and_machine_path_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload_wheel = root / "payload.whl"
            with zipfile.ZipFile(payload_wheel, "w") as archive:
                archive.writestr("immer/private/model.safetensors", b"weights")
            with self.assertRaisesRegex(
                release_preflight.ReleasePreflightError,
                "payload release member",
            ):
                release_preflight.inspect_distribution(payload_wheel)

            path_wheel = root / "path.whl"
            private_path = b"/" + b"Users" + b"/operator/private"
            with zipfile.ZipFile(path_wheel, "w") as archive:
                archive.writestr("immer/config.py", private_path)
            with self.assertRaisesRegex(
                release_preflight.ReleasePreflightError,
                "machine path",
            ):
                release_preflight.inspect_distribution(path_wheel)


if __name__ == "__main__":
    unittest.main()
