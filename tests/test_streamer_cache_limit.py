from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from immer.knowledge import streamer as streamer_module
from immer.knowledge.streamer import Streamer


def _write_source(root: Path, size: int = 16 * 1024) -> None:
    root.mkdir(parents=True)
    root.joinpath("payload.dat").write_bytes(bytes(range(256)) * ((size + 255) // 256))


def _complete_pairs(cache: Path) -> list[tuple[Path, Path, dict]]:
    pairs: list[tuple[Path, Path, dict]] = []
    for leaf in ("ranges", "files"):
        directory = cache / leaf
        if not directory.is_dir():
            continue
        for meta_path in directory.iterdir():
            if meta_path.suffix != ".json":
                continue
            blob_path = meta_path.with_suffix(".bin")
            if blob_path.is_file():
                pairs.append(
                    (
                        blob_path,
                        meta_path,
                        json.loads(meta_path.read_text(encoding="utf-8")),
                    )
                )
    return pairs


def _pair_bytes(pair: tuple[Path, Path, dict]) -> int:
    return pair[0].stat().st_size + pair[1].stat().st_size


class StreamerCacheLimitTests(unittest.TestCase):
    def test_multi_range_admission_keeps_leaf_keys_under_hard_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            _write_source(root)
            limit = 700
            source = Streamer.from_local(
                root,
                cache_dir=cache,
                budget_mb=1.0,
                max_cache_bytes=limit,
            )

            result = source.raw_bytes_many(
                "payload.dat",
                [(0, 256), (256, 256), (512, 256), (768, 256)],
                resident_limit_bytes=1024,
            )

            self.assertEqual(result.source_requests, 1)
            self.assertEqual(result.source_bytes, 1024)
            self.assertEqual(result.resident_bytes, 1024)
            self.assertEqual(
                b"".join(bytes(part) for part in result.parts),
                bytes(range(256)) * 4,
            )
            pairs = _complete_pairs(cache)
            self.assertGreaterEqual(len(pairs), 1)
            self.assertTrue(
                all(
                    pair[2]["contract"]["end"]
                    - pair[2]["contract"]["start"]
                    + 1
                    == 256
                    for pair in pairs
                )
            )
            metrics = source.metrics()
            self.assertLessEqual(metrics["cache_bytes"], limit)
            self.assertGreater(metrics["cache_write_skips_oversize"], 0)

    def test_default_is_unlimited_and_capped_cache_evicts_true_lru(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            _write_source(root)
            source = Streamer.from_local(root, cache_dir=cache, budget_mb=1.0)

            for offset in (0, 512, 1024):
                self.assertEqual(len(source.raw_bytes("payload.dat", offset, 256)), 256)
            pairs = _complete_pairs(cache)
            self.assertEqual(len(pairs), 3)
            by_start = {pair[2]["contract"]["start"]: pair for pair in pairs}

            now = time.time_ns()
            os.utime(by_start[0][1], ns=(now - 3_000_000_000, now - 3_000_000_000))
            os.utime(by_start[512][1], ns=(now - 2_000_000_000, now - 2_000_000_000))
            os.utime(by_start[1024][1], ns=(now - 1_000_000_000, now - 1_000_000_000))

            # A verified hit makes range 0 the newest entry. Range 512 must
            # therefore be evicted even though it was written second.
            source.raw_bytes("payload.dat", 0, 256)
            pairs = _complete_pairs(cache)
            by_start = {pair[2]["contract"]["start"]: pair for pair in pairs}
            total = sum(_pair_bytes(pair) for pair in pairs)
            limit = total - _pair_bytes(by_start[512])

            inventory_dir = cache / "inventories"
            inventory_dir.mkdir(parents=True, exist_ok=True)
            sentinel = inventory_dir / "must-survive.json"
            sentinel.write_text("{}", encoding="utf-8")

            capped = Streamer.from_local(
                root,
                cache_dir=cache,
                budget_mb=1.0,
                max_cache_bytes=limit,
            )
            metrics = capped.metrics()
            remaining = {
                pair[2]["contract"]["start"] for pair in _complete_pairs(cache)
            }
            self.assertEqual(remaining, {0, 1024})
            self.assertTrue(sentinel.is_file())
            self.assertLessEqual(metrics["cache_bytes"], limit)
            self.assertEqual(metrics["cache_evictions"], 1)
            self.assertGreater(metrics["cache_evicted_bytes"], 0)
            self.assertIsNone(source.metrics()["cache_limit_bytes"])

    def test_oversize_entry_is_returned_but_never_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            _write_source(root)
            source = Streamer.from_local(
                root,
                cache_dir=cache,
                budget_mb=1.0,
                max_cache_bytes=64,
            )

            self.assertEqual(len(source.raw_bytes("payload.dat", 0, 512)), 512)
            metrics = source.metrics()
            self.assertEqual(_complete_pairs(cache), [])
            self.assertEqual(metrics["cache_bytes"], 0)
            self.assertEqual(metrics["cache_writes"], 0)
            self.assertEqual(metrics["cache_write_skips_oversize"], 1)

    def test_parallel_writes_leave_complete_pairs_below_hard_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            _write_source(root)
            limit = 1200
            source = Streamer.from_local(
                root,
                cache_dir=cache,
                budget_mb=2.0,
                max_cache_bytes=limit,
            )

            with ThreadPoolExecutor(max_workers=8) as pool:
                bodies = list(
                    pool.map(
                        lambda offset: source.raw_bytes("payload.dat", offset, 256),
                        range(0, 4096, 256),
                    )
                )
            self.assertTrue(all(len(body) == 256 for body in bodies))

            metrics = source.metrics()
            self.assertLessEqual(metrics["cache_bytes"], limit)
            self.assertGreater(metrics["cache_evictions"], 0)
            for leaf in ("ranges", "files"):
                directory = cache / leaf
                if not directory.is_dir():
                    continue
                names = {path.name for path in directory.iterdir()}
                for name in names:
                    if name.endswith(".bin"):
                        self.assertIn(f"{name[:-4]}.json", names)
                    elif name.endswith(".json"):
                        self.assertIn(f"{name[:-5]}.bin", names)

    def test_eviction_ignores_inventory_and_incomplete_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            _write_source(root)
            source = Streamer.from_local(root, cache_dir=cache, budget_mb=1.0)
            source.raw_bytes("payload.dat", 0, 128)

            orphan = cache / "ranges" / ("f" * 64 + ".bin")
            orphan.write_bytes(b"orphan")
            inventory = cache / "inventories" / "inventory.json"
            inventory.parent.mkdir(parents=True, exist_ok=True)
            inventory.write_text("inventory", encoding="utf-8")

            capped = Streamer.from_local(
                root,
                cache_dir=cache,
                budget_mb=1.0,
                max_cache_bytes=0,
            )
            self.assertEqual(capped.metrics()["cache_bytes"], 0)
            self.assertEqual(_complete_pairs(cache), [])
            self.assertEqual(orphan.read_bytes(), b"orphan")
            self.assertEqual(inventory.read_text(encoding="utf-8"), "inventory")

    def test_invalid_limits_fail_before_io(self) -> None:
        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    Streamer("fixture", max_cache_bytes=invalid)  # type: ignore[arg-type]

    def test_request_priority_evicts_experts_before_hot_core_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            _write_source(root)
            initial = Streamer.from_local(root, cache_dir=cache, budget_mb=1.0)
            initial.raw_bytes("payload.dat", 0, 256)
            initial.raw_bytes("payload.dat", 512, 256)
            pairs = _complete_pairs(cache)
            # Leave metadata-size slack so replacing one range is sufficient
            # even though decimal offsets have different serialized lengths.
            limit = sum(_pair_bytes(pair) for pair in pairs) + 256

            capped = Streamer.from_local(
                root,
                cache_dir=cache,
                budget_mb=1.0,
                max_cache_bytes=limit,
            )
            # Core range 0 is protected for this model pass. Range 512 stays
            # ordinary priority zero even though both are valid cache hits.
            with capped.cache_priority(1):
                capped.raw_bytes("payload.dat", 0, 256)
            capped.raw_bytes("payload.dat", 1024, 256)

            remaining = {
                pair[2]["contract"]["start"] for pair in _complete_pairs(cache)
            }
            self.assertEqual(remaining, {0, 1024})
            metrics = capped.metrics()
            self.assertGreaterEqual(metrics["cache_priority_promotions"], 1)
            self.assertEqual(metrics["cache_evictions"], 1)

    def test_same_key_refresh_never_stages_bytes_above_hard_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            _write_source(root)
            initial = Streamer.from_local(root, cache_dir=cache, budget_mb=1.0)
            initial.raw_bytes("payload.dat", 0, 256)
            limit = sum(_pair_bytes(pair) for pair in _complete_pairs(cache))
            capped = Streamer.from_local(
                root,
                cache_dir=cache,
                budget_mb=1.0,
                max_cache_bytes=limit,
            )

            original_atomic_write = streamer_module._atomic_write
            projected_peaks: list[int] = []

            def observed_atomic_write(path: Path, data: bytes) -> None:
                present = sum(
                    child.stat().st_size
                    for leaf in ("ranges", "files")
                    for child in (cache / leaf).glob("*")
                    if child.is_file()
                )
                # mkstemp writes the replacement before os.replace(). This is
                # the maximum size reached by this individual atomic write.
                projected_peaks.append(present + len(data))
                original_atomic_write(path, data)

            with mock.patch.object(
                streamer_module, "_atomic_write", side_effect=observed_atomic_write
            ):
                with capped.reader.uncached():
                    self.assertEqual(
                        capped.raw_bytes("payload.dat", 0, 256),
                        bytes(range(256)),
                    )

            self.assertEqual(len(projected_peaks), 3)
            self.assertLessEqual(max(projected_peaks), limit)
            self.assertLessEqual(
                sum(_pair_bytes(pair) for pair in _complete_pairs(cache)), limit
            )

    def test_failed_or_interrupted_refresh_recovers_as_cache_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            _write_source(root)
            initial = Streamer.from_local(root, cache_dir=cache, budget_mb=1.0)
            initial.raw_bytes("payload.dat", 0, 256)
            limit = sum(_pair_bytes(pair) for pair in _complete_pairs(cache))
            capped = Streamer.from_local(
                root,
                cache_dir=cache,
                budget_mb=1.0,
                max_cache_bytes=limit,
            )
            original_atomic_write = streamer_module._atomic_write

            def fail_metadata(path: Path, data: bytes) -> None:
                if path.suffix == ".json":
                    raise OSError("injected metadata write failure")
                original_atomic_write(path, data)

            with mock.patch.object(
                streamer_module, "_atomic_write", side_effect=fail_metadata
            ):
                with capped.reader.uncached(), self.assertRaises(OSError):
                    capped.raw_bytes("payload.dat", 0, 256)
            self.assertEqual(list((cache / "ranges").glob("*")), [])

            # Reproduce a hard process stop after the payload write: the
            # transaction marker makes the exact partial key recoverable.
            key, _contract = capped.reader._cache_key("range", "payload.dat", 0, 255)
            directory = cache / "ranges"
            directory.mkdir(parents=True, exist_ok=True)
            directory.joinpath(f"{key}.bin").write_bytes(b"partial")
            directory.joinpath(f"{key}.pending").write_bytes(b"")
            recovered = Streamer.from_local(
                root,
                cache_dir=cache,
                budget_mb=1.0,
                max_cache_bytes=limit,
            )
            self.assertEqual(
                recovered.raw_bytes("payload.dat", 0, 256), bytes(range(256))
            )
            self.assertEqual(recovered.metrics()["cache_recoveries"], 1)
            self.assertEqual(len(_complete_pairs(cache)), 1)

    def test_priorities_do_not_leak_between_readers_or_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "source"
            cache = Path(tmp) / "cache"
            _write_source(root)
            initial = Streamer.from_local(root, cache_dir=cache, budget_mb=1.0)
            for offset in (0, 512):
                initial.raw_bytes("payload.dat", offset, 256)
            limit = sum(_pair_bytes(pair) for pair in _complete_pairs(cache)) + 256

            first = Streamer.from_local(
                root,
                cache_dir=cache,
                budget_mb=1.0,
                max_cache_bytes=limit,
            )
            with first.cache_priority(1):
                first.raw_bytes("payload.dat", 0, 256)

            pairs = _complete_pairs(cache)
            by_start = {pair[2]["contract"]["start"]: pair for pair in pairs}
            now = time.time_ns()
            os.utime(by_start[0][1], ns=(now - 2_000_000_000,) * 2)
            os.utime(by_start[512][1], ns=(now - 1_000_000_000,) * 2)

            second = Streamer.from_local(
                root,
                cache_dir=cache,
                budget_mb=1.0,
                max_cache_bytes=limit,
            )
            second.raw_bytes("payload.dat", 512, 256)
            second.raw_bytes("payload.dat", 1024, 256)
            remaining = {
                pair[2]["contract"]["start"] for pair in _complete_pairs(cache)
            }
            self.assertEqual(remaining, {512, 1024})

            with second.cache_priority(1):
                second.raw_bytes("payload.dat", 512, 256)
            second.clear_cache_priorities()
            self.assertEqual(second.reader._cache_priorities, {})


if __name__ == "__main__":
    unittest.main()
