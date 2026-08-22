from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parent.parent
RECEIPT = (
    ROOT / "results" / "deepseek-v4-mmlu-off-4-exact-v4-window2x3-receipt.json"
)


def _canonical(value) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class DeepSeekV4LayerwiseReceiptTests(unittest.TestCase):
    def test_receipt_is_self_sealed_and_scoped_as_a_four_item_smoke(self) -> None:
        document = json.loads(RECEIPT.read_text(encoding="utf-8"))
        seal = document.pop("report_sha256")
        self.assertEqual(hashlib.sha256(_canonical(document)).hexdigest(), seal)
        self.assertEqual(document["schema"], "immer.deepseek-v4-layerwise-receipt/v1")
        self.assertEqual(document["status"], "complete")
        self.assertFalse(document["timing_observation"]["performance_claim"])
        self.assertIn("not-frontier-parity", document["claim"])
        outcomes = document["outcomes"]
        self.assertEqual(len(outcomes), 4)
        self.assertEqual(sum(bool(row["correct"]) for row in outcomes), 3)
        self.assertEqual(document["scope"]["accuracy"], 0.75)

    def test_bound_runtime_commit_exists(self) -> None:
        document = json.loads(RECEIPT.read_text(encoding="utf-8"))
        commit = document["bindings"]["runtime_git_commit"]
        result = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8"))

    def test_private_artifact_matches_receipt_when_present(self) -> None:
        document = json.loads(RECEIPT.read_text(encoding="utf-8"))
        run = ROOT / document["artifact"]["run"]
        if not run.is_dir():
            self.skipTest("private layerwise artifact is not present")
        manifest_path = run / "manifest.json"
        result_path = run / "result.json"
        self.assertEqual(
            _sha256_file(manifest_path),
            document["artifact"]["manifest_file_sha256"],
        )
        self.assertEqual(
            _sha256_file(result_path),
            document["artifact"]["result_file_sha256"],
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertEqual(
            hashlib.sha256(_canonical(manifest["body"])).hexdigest(),
            document["artifact"]["manifest_body_sha256"],
        )
        self.assertEqual(
            hashlib.sha256(_canonical(result["body"])).hexdigest(),
            document["artifact"]["result_body_sha256"],
        )
        self.assertEqual(
            manifest["body"]["state"]["result_body_sha256"],
            result["body_sha256"],
        )
        checkpoint = document["final_checkpoint"]
        checkpoint_path = run / "objects" / f"{checkpoint['sha256']}.safetensors"
        self.assertEqual(_sha256_file(checkpoint_path), checkpoint["sha256"])
        self.assertEqual(checkpoint_path.stat().st_size, checkpoint["file_bytes"])


if __name__ == "__main__":
    unittest.main()
