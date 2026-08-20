from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class ScriptSmokeTests(unittest.TestCase):
    def test_bootstrap_script(self) -> None:
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "bootstrap.py")], capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("bootstrap_answer: 27", result.stdout)


if __name__ == "__main__":
    unittest.main()
