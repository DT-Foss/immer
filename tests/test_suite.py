from __future__ import annotations

import json
import tempfile
import unittest
import urllib.request
from pathlib import Path

from immer.suite import Metrics, start_dashboard


class MetricsTests(unittest.TestCase):
    def test_emit_writes_status_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / "status.json"
            jsonl = Path(tmp) / "metrics.jsonl"
            metrics = Metrics(status_path=status, jsonl_path=jsonl)
            metrics.bump("chat")
            document = metrics.emit({"tokens": 42})
            self.assertEqual(document["counters"]["chat"], 1)
            self.assertEqual(document["gauges"]["tokens"], 42)
            self.assertTrue(status.is_file())
            lines = jsonl.read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["gauges"]["tokens"], 42)

    def test_status_file_is_fresh_after_second_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Metrics(status_path=Path(tmp) / "status.json")
            metrics.bump("turns", 1)
            metrics.emit()
            metrics.bump("turns", 1)
            fresh = metrics.emit()
            on_disk = json.loads((Path(tmp) / "status.json").read_text())
            self.assertEqual(on_disk["counters"]["turns"], fresh["counters"]["turns"])


class DashboardTests(unittest.TestCase):
    def test_status_and_html_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = Metrics(status_path=Path(tmp) / "status.json")
            server = start_dashboard(metrics, lambda: {"tokens": 7}, 0)
            port = server.server_address[1]
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/status") as response:
                    payload = json.loads(response.read())
                self.assertEqual(payload["gauges"]["tokens"], 7)
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
                    html = response.read().decode("utf-8")
                self.assertIn("Lebenszeichen", html)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
