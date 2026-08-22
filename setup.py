"""Setuptools build hooks for IMMER's generated package resources."""

from __future__ import annotations

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


_ROOT = Path(__file__).resolve().parent
_MANIFEST_NAMES = ("s3_ship_v6.json", "crsa_router_v1.json")


class BuildPyWithManifests(_build_py):
    """Copy canonical root manifests into the installed Python package."""

    def _manifest_outputs(self) -> tuple[Path, ...]:
        destination = Path(self.build_lib) / "immer" / "resources"
        return tuple(destination / name for name in _MANIFEST_NAMES)

    def run(self) -> None:
        super().run()
        outputs = self._manifest_outputs()
        self.mkpath(str(outputs[0].parent))
        for name, destination in zip(_MANIFEST_NAMES, outputs, strict=True):
            source = _ROOT / "manifests" / name
            if not source.is_file():
                raise FileNotFoundError(f"required canonical manifest is missing: {source}")
            self.copy_file(str(source), str(destination))

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        outputs = list(super().get_outputs(include_bytecode=include_bytecode))
        outputs.extend(str(path) for path in self._manifest_outputs())
        return outputs


setup(cmdclass={"build_py": BuildPyWithManifests})
