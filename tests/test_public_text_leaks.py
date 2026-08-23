from __future__ import annotations

import ipaddress
import json
from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _tracked_text() -> dict[str, str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    result: dict[str, str] = {}
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        relative = raw.decode("utf-8")
        try:
            result[relative] = (ROOT / relative).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
    return result


class PublicTextLeakTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.files = _tracked_text()

    def test_no_personal_home_or_server_root_paths(self) -> None:
        home = re.compile(r"/(?:Users|home)/[^/\s\"'`]+/")
        server_root = "/" + "root/"
        leaks = {
            path: sorted(set(home.findall(text)) | ({server_root} if server_root in text else set()))
            for path, text in self.files.items()
            if home.search(text) or server_root in text
        }
        self.assertEqual(leaks, {})

    def test_no_machine_ip_addresses(self) -> None:
        candidate = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
        leaks: dict[str, list[str]] = {}
        for path, text in self.files.items():
            for raw in candidate.findall(text):
                try:
                    address = ipaddress.ip_address(raw)
                except ValueError:
                    continue
                allowed = (
                    address.is_loopback
                    or address.is_unspecified
                    or address in ipaddress.ip_network("192.0.2.0/24")
                    or address in ipaddress.ip_network("198.51.100.0/24")
                    or address in ipaddress.ip_network("203.0.113.0/24")
                )
                if not allowed:
                    leaks.setdefault(path, []).append(raw)
        self.assertEqual(leaks, {})

    def test_no_credential_filenames_outside_ignore_rules(self) -> None:
        forbidden = (
            "CREDENTIALS" + ".md",
            "id_" + "rsa",
            "id_" + "ed25519",
        )
        leaks = {
            path: [name for name in forbidden if name in text]
            for path, text in self.files.items()
            if path != ".gitignore" and any(name in text for name in forbidden)
        }
        self.assertEqual(leaks, {})

    def test_tracked_result_json_has_no_absolute_machine_paths(self) -> None:
        leaks: dict[str, list[str]] = {}

        def walk(value):
            if isinstance(value, dict):
                for nested in value.values():
                    yield from walk(nested)
            elif isinstance(value, list):
                for nested in value:
                    yield from walk(nested)
            elif isinstance(value, str):
                yield value

        for path in sorted(self.files):
            if not path.startswith("results/") or not path.endswith(".json"):
                continue
            document = json.loads(self.files[path])
            absolute = sorted(
                {
                    value
                    for value in walk(document)
                    if value.startswith("/") and not value.startswith("/v1/")
                }
            )
            if absolute:
                leaks[path] = absolute
        self.assertEqual(leaks, {})


if __name__ == "__main__":
    unittest.main()
