from __future__ import annotations

import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from immer.knowledge.streamer import (
    ByteBudgetExceeded,
    CacheIntegrityError,
    RangeValidationError,
    Streamer,
    TensorSource,
)
from immer.knowledge._hf_source import HFRangeReader, SourceRangeError


class _FakeResponse:
    def __init__(
        self,
        body: bytes,
        *,
        status: int,
        headers: dict[str, str],
        url: str,
    ) -> None:
        self._body = io.BytesIO(body)
        self.status = status
        self.headers = headers
        self._url = url

    def read(self, amount: int = -1) -> bytes:
        return self._body.read(amount)

    def close(self) -> None:
        self._body.close()

    def geturl(self) -> str:
        return self._url

    def getcode(self) -> int:
        return self.status


class _FakeOpener:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = responses
        self.requests = []

    def open(self, request, timeout):
        self.requests.append((request, timeout))
        return self.responses.pop(0)


def _write_fixture(root: Path) -> tuple[np.ndarray, np.ndarray]:
    f32 = np.array(
        [[1.25, -2.5], [3.0, 4.5], [8.0, -0.125], [16.0, 32.0]],
        dtype="<f4",
    )
    bf32 = np.array([[0.5, -1.0], [2.0, 7.5], [-4.0, 9.0]], dtype="<f4")
    bf16 = (bf32.view("<u4") >> 16).astype("<u2")
    f32_bytes = f32.tobytes()
    bf16_bytes = bf16.tobytes()
    header = {
        "float.weight": {
            "dtype": "F32",
            "shape": list(f32.shape),
            "data_offsets": [0, len(f32_bytes)],
        },
        "brain.weight": {
            "dtype": "BF16",
            "shape": list(bf32.shape),
            "data_offsets": [len(f32_bytes), len(f32_bytes) + len(bf16_bytes)],
        },
        "__metadata__": {"fixture": "streamer-contract"},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + f32_bytes + bf16_bytes
    )
    return f32, bf32


class StreamerContractTests(unittest.TestCase):
    def test_packaged_hf_reader_enforces_content_range_and_bounded_files(self) -> None:
        digest = "a" * 64
        opener = _FakeOpener(
            [
                _FakeResponse(
                    b"cdef",
                    status=206,
                    headers={"Content-Range": "bytes 2-5/10", "ETag": "fixture-etag"},
                    url=f"https://cdn.example/{digest}?Expires=4102444800",
                ),
                _FakeResponse(
                    b"{}",
                    status=200,
                    headers={"Content-Length": "2"},
                    url="https://huggingface.co/org/repo/resolve/pinned/config.json",
                ),
            ]
        )
        budget = Streamer("fixture", budget_mb=1, use_cache=False).budget
        reader = HFRangeReader(
            "org/repo", revision="pinned", budget=budget, opener=opener
        )

        self.assertEqual(reader.get_range("model.safetensors", 2, 5), b"cdef")
        request, _timeout = opener.requests[0]
        self.assertEqual(request.get_header("Range"), "bytes=2-5")
        self.assertEqual(reader.file_info["model.safetensors"]["size"], 10)
        self.assertEqual(reader.file_info["model.safetensors"]["cas_url_hash"], digest)
        self.assertEqual(reader.fetch_file_bounded("config.json", 16), b"{}")
        self.assertEqual(budget.body, 6)

    def test_packaged_hf_reader_rejects_mismatched_content_range(self) -> None:
        opener = _FakeOpener(
            [
                _FakeResponse(
                    b"abcd",
                    status=206,
                    headers={"Content-Range": "bytes 1-4/10"},
                    url="https://cdn.example/file",
                )
            ]
        )
        budget = Streamer("fixture", budget_mb=1, use_cache=False).budget
        reader = HFRangeReader("org/repo", budget=budget, opener=opener)
        with self.assertRaises(SourceRangeError):
            reader.get_range("model.safetensors", 2, 5)
        self.assertEqual(budget.body, 4)

    def test_wheel_style_import_has_no_repository_vendor_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            root.mkdir()
            _write_fixture(root)
            source_root = Path(__file__).resolve().parents[1] / "src"
            wheel_root = Path(tmp) / "wheel-site"
            shutil.copytree(source_root / "immer", wheel_root / "immer")
            script = r"""
import os
import sys

class BlockResearchImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"hf_organ_reader", "casi_tensor_map", "live_casiv2"}:
            raise ImportError(f"repository research import blocked: {fullname}")
        return None

sys.meta_path.insert(0, BlockResearchImports())
from immer.knowledge.streamer import Streamer

source = Streamer.from_local(os.environ["IMMER_FIXTURE"], use_cache=False)
rows = source.rows("float.weight", 1, 1)
assert rows.tolist() == [[3.0, 4.5]]
assert not ({"hf_organ_reader", "casi_tensor_map", "live_casiv2"} & set(sys.modules))
print("wheel-safe")
"""
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(wheel_root)
            environment["IMMER_FIXTURE"] = str(root)
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=tmp,
                env=environment,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "wheel-safe")

    def test_local_source_inventory_exact_rows_and_torch_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            root.mkdir()
            f32, bf32 = _write_fixture(root)

            source = Streamer.from_local(root, cache_dir=cache, budget_mb=1.0)
            self.assertIsInstance(source, TensorSource)
            inventory = source.inventory()
            self.assertEqual(len(inventory["tensors"]), 2)

            selected = source.rows("float.weight", start_row=1, n_rows=2)
            np.testing.assert_array_equal(selected, f32[1:3])
            self.assertTrue(selected.flags.writeable)
            decoded = source.rows("brain.weight", start_row=0, n_rows=3)
            np.testing.assert_array_equal(decoded, bf32)

            try:
                import torch
            except ImportError:
                torch = None
            if torch is not None:
                bridged = source.rows_torch("float.weight", 2, 1)
                self.assertEqual(tuple(bridged.shape), (1, 2))
                self.assertEqual(bridged.device.type, "cpu")
                self.assertFalse(bridged.requires_grad)
                np.testing.assert_array_equal(bridged.numpy(), f32[2:3])

            metrics = source.metrics()
            self.assertGreater(metrics["network_or_source_body_bytes"], 0)
            self.assertEqual(metrics["failed_requests"], 0)
            self.assertEqual(metrics["optional_misses"], 1)
            self.assertTrue(metrics["revision_is_mutable"])
            # Header + selected rows, never the complete source/model payload.
            self.assertLess(metrics["network_or_source_body_bytes"], (root / "model.safetensors").stat().st_size + f32.nbytes + bf32.nbytes)
            self.assertIsNotNone(metrics["inventory_source_fingerprint"])

    def test_refresh_bypasses_resume_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            root.mkdir()
            _write_fixture(root)
            Streamer.from_local(root, cache_dir=cache).inventory()

            resumed = Streamer.from_local(root, cache_dir=cache)
            resumed.inventory()
            self.assertEqual(resumed.bytes_moved(), 0)
            resumed.inventory(refresh=True)
            self.assertGreater(resumed.bytes_moved(), 0)
            self.assertGreaterEqual(resumed.metrics()["cache_writes"], 2)

    def test_local_resume_cache_rejects_changed_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            root.mkdir()
            _write_fixture(root)
            Streamer.from_local(root, cache_dir=cache).inventory()

            shard = root / "model.safetensors"
            stat = shard.stat()
            os.utime(
                shard,
                ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
            )
            resumed = Streamer.from_local(root, cache_dir=cache)
            with self.assertRaisesRegex(CacheIntegrityError, "Quelle hat sich"):
                resumed.inventory()

    def test_resume_cache_is_zero_transfer_and_sha_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            root.mkdir()
            f32, _ = _write_fixture(root)

            first = Streamer.from_local(root, cache_dir=cache, budget_mb=1.0)
            np.testing.assert_array_equal(first.rows("float.weight", 1, 2), f32[1:3])
            self.assertGreater(first.bytes_moved(), 0)

            resumed = Streamer.from_local(root, cache_dir=cache, budget_mb=0.0)
            np.testing.assert_array_equal(resumed.rows("float.weight", 1, 2), f32[1:3])
            self.assertEqual(resumed.bytes_moved(), 0)
            metrics = resumed.metrics()
            self.assertEqual(metrics["inventory_cache_hits"], 1)
            self.assertEqual(metrics["cache_hits"], 1)
            self.assertEqual(metrics["cache_integrity_checks"], 1)
            self.assertGreater(metrics["cache_bytes_reused"], 0)

            # Damage precisely the row payload, not an arbitrary cache file.
            row_blob = None
            for meta_path in (cache / "ranges").glob("*.json"):
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                contract = meta["contract"]
                if contract["start"] is not None and meta["size"] == 2 * f32.shape[1] * f32.itemsize:
                    row_blob = meta_path.with_suffix(".bin")
                    break
            self.assertIsNotNone(row_blob)
            row_blob.write_bytes(b"corrupt")

            corrupted = Streamer.from_local(root, cache_dir=cache, budget_mb=1.0)
            with self.assertRaises(CacheIntegrityError):
                corrupted.rows("float.weight", 1, 2)

    def test_budget_is_preflighted_and_never_overshoots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture(root)
            source = Streamer.from_local(root, use_cache=False, budget_mb=1.0)
            source.inventory()
            used = source.budget.total
            source.budget.limit = used + 3

            with self.assertRaises(ByteBudgetExceeded):
                source.raw_bytes("model.safetensors", 0, 4)
            self.assertEqual(source.budget.total, used)
            self.assertLessEqual(source.budget.total, source.budget.limit)
            self.assertGreaterEqual(source.metrics()["budget"]["rejected_charges"], 1)

    def test_ranges_and_tensor_bounds_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_fixture(root)
            source = Streamer.from_local(root, use_cache=False)

            self.assertEqual(source.raw_bytes("model.safetensors", 0, 0), b"")
            self.assertEqual(len(source.raw_bytes("model.safetensors", 0, 8)), 8)
            with self.assertRaises(RangeValidationError):
                source.raw_bytes("model.safetensors", -1, 1)
            with self.assertRaises(RangeValidationError):
                source.raw_bytes("../outside", 0, 1)
            with self.assertRaises(RangeValidationError):
                source.reader.fetch_file("model.safetensors")
            with self.assertRaises(RangeValidationError):
                source.rows("float.weight", 3, 2)
            empty = source.rows("float.weight", 4, 0)
            self.assertEqual(empty.shape, (0, 2))

    def test_inventory_cache_integrity_is_not_silently_refetched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            root.mkdir()
            _write_fixture(root)
            first = Streamer.from_local(root, cache_dir=cache)
            first.inventory()
            inventory_cache = next((cache / "inventories").glob("*.json"))
            envelope = json.loads(inventory_cache.read_text(encoding="utf-8"))
            envelope["inventory"]["tensors"][0]["shape"][0] = 999
            inventory_cache.write_text(json.dumps(envelope), encoding="utf-8")

            second = Streamer.from_local(root, cache_dir=cache)
            with self.assertRaises(CacheIntegrityError):
                second.inventory()
            self.assertEqual(second.bytes_moved(), 0)


if __name__ == "__main__":
    unittest.main()
