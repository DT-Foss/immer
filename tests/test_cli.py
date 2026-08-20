from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from immer.cli import main


class CliTests(unittest.TestCase):
    def test_components_command_reports_integrated_code(self) -> None:
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(["components"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("FERTIG", text)
        self.assertIn("CRSA", text)
        self.assertIn("OrganBank", text)


if __name__ == "__main__":
    unittest.main()
