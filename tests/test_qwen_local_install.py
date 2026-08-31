from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.qwen3_8.bundle import QWEN38_BUNDLE_SCHEMA
from immer.runtimes.qwen3_8.config import OFFICIAL_REPO_ID, OFFICIAL_REVISION
from immer.runtimes.qwen3_8.local_install import (
    QwenLocalInstallError,
    inspect_local_qwen,
)
from immer.runtimes.qwen3_8.q4 import Q4_BANK_SCHEMA


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_document(path: Path, schema: str, body: dict) -> None:
    path.write_bytes(
        canonical_json_bytes(
            {
                "body": body,
                "schema": schema,
                "sha256": hashlib.sha256(canonical_json_bytes(body)).hexdigest(),
            }
        )
    )


def _fixture(root: Path) -> tuple[Path, Path]:
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE"}}),
        encoding="utf-8",
    )
    (root / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    (root / "model-00001.safetensors").write_bytes(b"source")
    causal = root / "causal"
    causal.mkdir()
    (causal / "manifest.head.json").write_text("{}", encoding="utf-8")
    (causal / "manifest.jsonl").write_text("{}\n", encoding="utf-8")
    packed = causal / "q4-base-v3-mtp"
    packed.mkdir()
    weights = packed / "weights"
    weights.mkdir()
    (weights / "tensor.q4_0.bin").write_bytes(b"packed")
    _write_document(
        root / "bundle.json",
        QWEN38_BUNDLE_SCHEMA,
        {
            "checkpoint_bytes": 6,
            "checkpoint_complete": True,
            "config_sha256": _sha(root / "config.json"),
            "graph_revision": [1, "a" * 64],
            "index_sha256": _sha(root / "model.safetensors.index.json"),
            "inventory_sha256": "b" * 64,
            "layout_fingerprint": "c" * 64,
            "logical_model": {
                "repo_id": OFFICIAL_REPO_ID,
                "revision": OFFICIAL_REVISION,
            },
            "shards": [
                {
                    "file": "model-00001.safetensors",
                    "sha256": "d" * 64,
                    "size": 6,
                }
            ],
            "tensor_bindings": 1,
            "weights_layout": "flat/v1",
        },
    )
    _write_document(
        packed / "manifest.json",
        Q4_BANK_SCHEMA,
        {
            "block_size": 32,
            "format_policy": "q4_0-text-matrices+q8_0-embedding-head/v1",
            "native_abi": 2,
            "payload_bytes": 6,
            "source": {
                "repo_id": OFFICIAL_REPO_ID,
                "revision": OFFICIAL_REVISION,
            },
            "source_bf16_bytes": 6,
            "tensor_count": 1,
            "tensors": [
                {
                    "file": "tensor.q4_0.bin",
                    "format": "q4_0",
                    "name": "model.layer.weight",
                    "payload_bytes": 6,
                    "payload_sha256": "e" * 64,
                    "row_bytes": 6,
                    "shape": [1, 1],
                    "source_dtype": "BF16",
                }
            ],
        },
    )
    return root / "tokenizer.json", packed


class QwenLocalInstallTests(unittest.TestCase):
    def test_complete_install_is_recognized_without_payload_hashing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer, packed = _fixture(root)
            with patch(
                "immer.runtimes.qwen3_8.local_install.Qwen38Config.from_file"
            ) as config:
                install = inspect_local_qwen(
                    root,
                    tokenizer_path=tokenizer,
                    q4_root=packed,
                )

        config.assert_called_once_with(root / "config.json", require_official=True)
        self.assertEqual(install.checkpoint_bytes, 6)
        self.assertEqual(install.checkpoint_shards, 1)
        self.assertEqual(install.q4_payload_bytes, 6)
        self.assertEqual(install.q4_tensors, 1)
        self.assertIn("Q4/Q8 tensors", install.summary())

    def test_payload_size_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer, packed = _fixture(root)
            (packed / "weights" / "tensor.q4_0.bin").write_bytes(b"short")
            with (
                patch(
                    "immer.runtimes.qwen3_8.local_install.Qwen38Config.from_file"
                ),
                self.assertRaisesRegex(QwenLocalInstallError, "payload size differs"),
            ):
                inspect_local_qwen(
                    root,
                    tokenizer_path=tokenizer,
                    q4_root=packed,
                )

    def test_manifest_digest_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tokenizer, packed = _fixture(root)
            document = json.loads((packed / "manifest.json").read_text())
            document["body"]["payload_bytes"] = 5
            (packed / "manifest.json").write_text(json.dumps(document))
            with (
                patch(
                    "immer.runtimes.qwen3_8.local_install.Qwen38Config.from_file"
                ),
                self.assertRaisesRegex(QwenLocalInstallError, "envelope is invalid"),
            ):
                inspect_local_qwen(
                    root,
                    tokenizer_path=tokenizer,
                    q4_root=packed,
                )


if __name__ == "__main__":
    unittest.main()
