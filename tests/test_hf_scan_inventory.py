from __future__ import annotations

import json
import unittest

from immer.knowledge._hf_source import (
    Budget,
    SourceError,
    SourceNotFound,
    scan_inventory,
)
from immer.knowledge.streamer import Streamer


class _InventoryReader:
    repo = "fixture/model"
    rev = "7" * 40

    def __init__(self, *, size: object) -> None:
        self.size = size

    def fetch_optional_file(self, filename: str) -> bytes:
        if filename != "model.safetensors.index.json":
            raise AssertionError(filename)
        return json.dumps(
            {
                "metadata": {"total_size": 12},
                "weight_map": {
                    "a": "model.safetensors",
                    "b": "model.safetensors",
                },
            },
            separators=(",", ":"),
        ).encode("utf-8")

    def fetch_st_header(self, filename: str):
        if filename != "model.safetensors":
            raise AssertionError(filename)
        return (
            {
                "a": {"data_offsets": [0, 4], "dtype": "F32", "shape": [1]},
                "b": {"data_offsets": [4, 12], "dtype": "F32", "shape": [2]},
            },
            128,
            {
                "cas_url_hash": "a" * 64,
                "etag": '"' + "b" * 64 + '"',
                "header_len": 120,
                "size": self.size,
            },
        )


class ScanInventorySizeTests(unittest.TestCase):
    def test_cached_header_without_http_size_derives_exact_file_size(self) -> None:
        inventory = scan_inventory(_InventoryReader(size=None), Budget(1))
        self.assertEqual(inventory["shards"][0]["size"], 140)
        self.assertEqual(inventory["model_payload_bytes"], 12)

    def test_reported_file_size_must_equal_header_derived_size(self) -> None:
        with self.assertRaisesRegex(SourceError, "140"):
            scan_inventory(_InventoryReader(size=141), Budget(1))

    def test_reported_file_size_rejects_lossy_numeric_coercion(self) -> None:
        with self.assertRaisesRegex(SourceError, "ungueltig"):
            scan_inventory(_InventoryReader(size=140.9), Budget(1))

    def test_empty_safetensors_shard_uses_header_end_as_file_size(self) -> None:
        class EmptyReader(_InventoryReader):
            def fetch_optional_file(self, filename: str) -> bytes:
                raise SourceNotFound(filename)

            def fetch_st_header(self, filename: str):
                return {}, 128, {"header_len": 120, "size": None}

        inventory = scan_inventory(EmptyReader(size=None), Budget(1))
        self.assertEqual(inventory["shards"][0]["size"], 128)
        self.assertEqual(inventory["shards"][0]["n_tensors"], 0)
        self.assertEqual(inventory["model_payload_bytes"], 0)

    def test_tensor_offsets_must_be_contiguous_and_nonoverlapping(self) -> None:
        class InvalidOffsetsReader(_InventoryReader):
            def __init__(self, intervals):
                super().__init__(size=None)
                self.intervals = intervals

            def fetch_st_header(self, filename: str):
                header = {
                    name: {"data_offsets": offsets, "dtype": "F32", "shape": [1]}
                    for name, offsets in self.intervals
                }
                return header, 128, {"header_len": 120, "size": None}

        for label, intervals in (
            ("gap", (("a", [0, 4]), ("b", [8, 12]))),
            ("overlap", (("a", [0, 8]), ("b", [4, 12]))),
        ):
            with self.subTest(label=label), self.assertRaisesRegex(
                SourceError, "lueckenlos"
            ):
                scan_inventory(InvalidOffsetsReader(intervals), Budget(1))

    def test_live_and_cached_header_scans_have_identical_source_identity(self) -> None:
        live = scan_inventory(_InventoryReader(size=140), Budget(1))
        cached = scan_inventory(_InventoryReader(size=None), Budget(1))
        self.assertEqual(
            Streamer._inventory_layout_projection(live),
            Streamer._inventory_layout_projection(cached),
        )
        self.assertEqual(
            Streamer._source_fingerprint(live),
            Streamer._source_fingerprint(cached),
        )


if __name__ == "__main__":
    unittest.main()
