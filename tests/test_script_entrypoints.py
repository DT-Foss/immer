from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "crsa_route_eval.py",
    "router_v2.py",
    "router_v2_stage2.py",
    "poc_mmlu_stream.py",
    "bench_gsm8k.py",
    "qwen38_causal_bundle.py",
    "qwen38_direct_decode_benchmark.py",
    "qwen38_fertig_arm_compare.py",
)
FORBIDDEN_RESEARCH_IMPORTS = (
    "vendor/mitglm",
    "hf_organ_reader",
    "casi_tensor_map",
    "sys.path.insert",
    "length_extrap_v2",
    "streaming_train",
)


class ScriptEntrypointTests(unittest.TestCase):
    @staticmethod
    def _load_script(name: str):
        path = ROOT / "scripts" / name
        spec = importlib.util.spec_from_file_location(f"_test_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _assert_help(self, script: Path, pythonpath: Path, cwd: Path) -> None:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(pythonpath)
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("usage:", result.stdout.lower())

    def test_help_works_from_checkout_with_public_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            for name in SCRIPTS:
                with self.subTest(script=name):
                    self._assert_help(ROOT / "scripts" / name, ROOT / "src", cwd)

    def test_help_works_without_repository_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            site = root / "site"
            bin_dir = root / "bin"
            shutil.copytree(ROOT / "src" / "immer", site / "immer")
            resources = site / "immer" / "resources"
            resources.mkdir()
            for manifest in ("s3_ship_v6.json", "crsa_router_v1.json"):
                shutil.copy2(ROOT / "manifests" / manifest, resources / manifest)
            bin_dir.mkdir()
            for name in SCRIPTS:
                copied = shutil.copy2(ROOT / "scripts" / name, bin_dir / name)
                with self.subTest(script=name):
                    self._assert_help(Path(copied), site, root)

    def test_scripts_do_not_import_repository_research_helpers(self) -> None:
        for name in SCRIPTS:
            source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            for forbidden in FORBIDDEN_RESEARCH_IMPORTS:
                with self.subTest(script=name, forbidden=forbidden):
                    self.assertNotIn(forbidden, source)

    def test_selected_row_helpers_preserve_order_and_exact_ranges(self) -> None:
        matrix = np.arange(80, dtype=np.float32).reshape(10, 8)

        class FakeSource:
            def __init__(self) -> None:
                self.calls: list[tuple[int, int]] = []

            def find(self, _name: str) -> dict[str, list[int]]:
                return {"shape": list(matrix.shape)}

            def rows(self, _name: str, start: int, count: int) -> np.ndarray:
                self.calls.append((start, count))
                return matrix[start : start + count].copy()

        for name in ("router_v2.py", "router_v2_stage2.py", "poc_mmlu_stream.py"):
            source = FakeSource()
            helper = self._load_script(name)._fetch_rows
            with self.subTest(script=name):
                actual = helper(source, "fixture", [4, 1, 2, 2, 7])
                np.testing.assert_array_equal(actual, matrix[[4, 1, 2, 2, 7]])
                self.assertEqual(sorted(source.calls), [(1, 2), (4, 1), (7, 1)])

    def test_stage2_is_still_explicitly_falsified(self) -> None:
        source = (ROOT / "scripts" / "router_v2_stage2.py").read_text(encoding="utf-8")
        self.assertIn('METHOD_VERDICT = "NEGATIVE_METHOD_FALSIFIED"', source)
        self.assertIn('"eligible_as_runtime_router": False', source)


if __name__ == "__main__":
    unittest.main()
