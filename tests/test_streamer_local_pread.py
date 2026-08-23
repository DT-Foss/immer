from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from immer.knowledge.streamer import (
    ByteBudgetExceeded,
    HardByteBudget,
    LocalRangeReader,
    RangeValidationError,
    Streamer,
)


class LocalPreadRangeReaderTests(unittest.TestCase):
    def test_configured_root_symlink_is_rejected_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            actual = base / "actual"
            actual.mkdir()
            link = base / "configured-root"
            os.symlink(actual, link)

            with self.assertRaises(RangeValidationError):
                LocalRangeReader(link)
            with self.assertRaises(RangeValidationError):
                Streamer.from_local(link, use_cache=False)

    def test_concurrent_overlapping_reads_share_fd_without_cursor_races(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = bytes(range(256)) * 4096
            root.joinpath("weights.bin").write_bytes(payload)
            reader = LocalRangeReader(root, max_open_files=4)
            workers = 24
            start = threading.Barrier(workers)

            def read(index: int) -> bytes:
                offset = (index * 7919) % (len(payload) - 32_768)
                start.wait()
                return reader.get_range("weights.bin", offset, offset + 32_767)

            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results = tuple(pool.map(read, range(workers)))
                for index, result in enumerate(results):
                    offset = (index * 7919) % (len(payload) - 32_768)
                    self.assertEqual(result, payload[offset : offset + 32_768])
                metrics = reader.transport_metrics()
                self.assertEqual(metrics["transport_fd_opens"], 1)
                self.assertEqual(metrics["transport_fd_reuses"], workers - 1)
                self.assertGreaterEqual(metrics["transport_peak_leases"], 2)
                self.assertEqual(metrics["transport_fd_open_files"], 1)
            finally:
                reader.close()

    def test_fd_reuse_preserves_budget_and_exact_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = b"0123456789abcdef"
            root.joinpath("weights.bin").write_bytes(payload)
            budget = HardByteBudget(1.0)
            with LocalRangeReader(root, budget=budget) as reader:
                self.assertEqual(reader.get_range("weights.bin", 2, 7), b"234567")
                self.assertEqual(reader.get_range("weights.bin", 5, 10), b"56789a")
                metrics = reader.transport_metrics()
                self.assertEqual(metrics["transport_fd_opens"], 1)
                self.assertEqual(metrics["transport_fd_reuses"], 1)
                self.assertEqual(budget.body, 12)
                self.assertEqual(budget.requests, 2)

    def test_direct_over_budget_calls_fail_before_any_pread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.joinpath("weights.bin").write_bytes(b"0123456789abcdef")
            for method, arguments in (
                ("get_range", ("weights.bin", 0, 7)),
                ("fetch_file", ("weights.bin",)),
            ):
                with self.subTest(method=method):
                    reader = LocalRangeReader(root, budget=HardByteBudget(0.0))
                    try:
                        with mock.patch("os.pread", wraps=os.pread) as pread:
                            with self.assertRaises(ByteBudgetExceeded):
                                getattr(reader, method)(*arguments)
                        pread.assert_not_called()
                        self.assertEqual(
                            reader.transport_metrics()["transport_fd_opens"], 0
                        )
                        self.assertEqual(reader.budget.body, 0)
                        self.assertEqual(reader.budget.requests, 0)
                        self.assertEqual(reader.budget.rejected_charges, 1)
                    finally:
                        reader.close()

    def test_streamer_outer_reservation_is_reused_and_charged_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.joinpath("weights.bin").write_bytes(b"0123456789abcdef")
            source = Streamer.from_local(root, use_cache=False, budget_mb=1.0)
            try:
                with mock.patch("os.pread", wraps=os.pread) as pread:
                    self.assertEqual(source.raw_bytes("weights.bin", 3, 8), b"3456789a")
                pread.assert_called_once()
                metrics = source.metrics()
                self.assertEqual(metrics["network_or_source_body_bytes"], 8)
                self.assertEqual(metrics["budget"]["bytes_body"], 8)
                self.assertEqual(metrics["budget"]["http_requests"], 1)
                self.assertEqual(metrics["budget"]["rejected_charges"], 0)
                self.assertEqual(source.budget._reserved_bytes, 0)
            finally:
                source.close()

    def test_direct_fetch_re_reserves_after_growth_before_pread(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "metadata.json"
            path.write_bytes(b"old!")
            reader = LocalRangeReader(root, budget=HardByteBudget(1.0))
            original_lease = reader._lease_entry
            replaced = False

            def replace_then_lease(key: str, parts: tuple[str, ...]):
                nonlocal replaced
                if not replaced:
                    replacement = root / "replacement.tmp"
                    replacement.write_bytes(b"new-data")
                    os.replace(replacement, path)
                    replaced = True
                return original_lease(key, parts)

            try:
                with (
                    mock.patch.object(
                        reader, "_lease_entry", side_effect=replace_then_lease
                    ),
                    mock.patch("os.pread", wraps=os.pread) as pread,
                ):
                    self.assertEqual(reader.fetch_file("metadata.json"), b"new-data")
                self.assertEqual([call.args[1] for call in pread.call_args_list], [8])
                self.assertEqual(reader.budget.body, 8)
                self.assertEqual(reader.budget.requests, 1)
                self.assertEqual(reader.budget.rejected_charges, 0)
                self.assertEqual(reader.budget._reserved_bytes, 0)
            finally:
                reader.close()

    def test_configurable_lru_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, body in (("a", b"a"), ("b", b"b"), ("c", b"c")):
                root.joinpath(name).write_bytes(body)
            with LocalRangeReader(root, max_open_files=2) as reader:
                self.assertEqual(reader.get_range("a", 0, 0), b"a")
                self.assertEqual(reader.get_range("b", 0, 0), b"b")
                self.assertEqual(reader.get_range("a", 0, 0), b"a")
                self.assertEqual(reader.get_range("c", 0, 0), b"c")
                self.assertEqual(tuple(reader._fd_cache), ("a", "c"))
                self.assertEqual(reader.get_range("b", 0, 0), b"b")
                self.assertEqual(tuple(reader._fd_cache), ("c", "b"))
                metrics = reader.transport_metrics()
                self.assertEqual(metrics["transport_fd_max_open_files"], 2)
                self.assertEqual(metrics["transport_fd_peak_open_files"], 2)
                self.assertEqual(metrics["transport_fd_open_files"], 2)
                self.assertEqual(metrics["transport_fd_opens"], 4)
                self.assertEqual(metrics["transport_fd_reuses"], 1)
                self.assertEqual(metrics["transport_fd_evictions"], 2)
                self.assertEqual(metrics["transport_fd_closes"], 2)

    def test_none_allows_unbounded_fd_residency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            count = LocalRangeReader.DEFAULT_MAX_OPEN_FILES + 6
            for index in range(count):
                root.joinpath(f"part-{index:03d}").write_bytes(bytes([index]))
            reader = LocalRangeReader(root, max_open_files=None)
            try:
                for index in range(count):
                    self.assertEqual(
                        reader.get_range(f"part-{index:03d}", 0, 0), bytes([index])
                    )
                metrics = reader.transport_metrics()
                self.assertIsNone(metrics["transport_fd_max_open_files"])
                self.assertEqual(metrics["transport_fd_open_files"], count)
                self.assertEqual(metrics["transport_fd_evictions"], 0)
            finally:
                reader.close()
            metrics = reader.transport_metrics()
            self.assertEqual(metrics["transport_fd_open_files"], 0)
            self.assertEqual(metrics["transport_fd_opens"], count)
            self.assertEqual(metrics["transport_fd_closes"], count)

    def test_max_open_files_validation_and_streamer_forwarding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for invalid in (0, -1, True, 1.5, "4"):
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        LocalRangeReader(root, max_open_files=invalid)  # type: ignore[arg-type]

            source = Streamer.from_local(root, use_cache=False, max_open_files=None)
            try:
                self.assertIsNone(source.metrics()["transport_fd_max_open_files"])
            finally:
                source.close()

    def test_close_waits_for_active_read_and_is_idempotent_without_fd_leaks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.joinpath("weights.bin").write_bytes(b"abcdefgh")
            reader = LocalRangeReader(root)
            entered = threading.Event()
            release = threading.Event()
            closed = threading.Event()
            original_pread = os.pread

            def held_pread(fd: int, length: int, offset: int) -> bytes:
                entered.set()
                self.assertTrue(release.wait(timeout=5))
                return original_pread(fd, length, offset)

            with mock.patch("os.pread", side_effect=held_pread):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    read_future = pool.submit(reader.get_range, "weights.bin", 1, 6)
                    self.assertTrue(entered.wait(timeout=5))
                    data_fds = tuple(entry.fd for entry in reader._fd_cache.values())
                    root_fd = reader._root_fd

                    def close_reader() -> None:
                        reader.close()
                        closed.set()

                    close_future = pool.submit(close_reader)
                    time.sleep(0.03)
                    self.assertFalse(closed.is_set())
                    release.set()
                    self.assertEqual(read_future.result(timeout=5), b"bcdefg")
                    close_future.result(timeout=5)

            reader.close()
            metrics = reader.transport_metrics()
            self.assertTrue(metrics["transport_closed"])
            self.assertEqual(metrics["transport_fd_open_files"], 0)
            self.assertEqual(metrics["transport_fd_opens"], 1)
            self.assertEqual(metrics["transport_fd_closes"], 1)
            for descriptor in (*data_fds, root_fd):
                with self.assertRaises(OSError):
                    os.fstat(descriptor)
            with self.assertRaises(RuntimeError):
                reader.get_range("weights.bin", 0, 0)

    def test_atomic_replacement_reopens_instead_of_reading_stale_inode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "weights.bin"
            path.write_bytes(b"old-value")
            reader = LocalRangeReader(root)
            try:
                self.assertEqual(reader.get_range("weights.bin", 0, 8), b"old-value")
                old_stat = path.stat()
                replacement = root / "replacement.tmp"
                replacement.write_bytes(b"new-value")
                os.utime(
                    replacement,
                    ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns),
                )
                os.replace(replacement, path)

                self.assertEqual(reader.get_range("weights.bin", 0, 8), b"new-value")
                metrics = reader.transport_metrics()
                self.assertEqual(metrics["transport_fd_opens"], 2)
                self.assertEqual(metrics["transport_fd_reopens"], 1)
                self.assertEqual(metrics["transport_fd_closes"], 1)
                self.assertEqual(metrics["transport_fd_open_files"], 1)
            finally:
                reader.close()

    def test_traversal_and_symlink_escape_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "root"
            root.mkdir()
            outside = base / "outside.bin"
            outside.write_bytes(b"secret")
            (root / "inside.bin").write_bytes(b"inside")
            os.symlink(outside, root / "leaf-link")
            outside_dir = base / "outside-dir"
            outside_dir.mkdir()
            (outside_dir / "payload").write_bytes(b"secret")
            os.symlink(outside_dir, root / "dir-link")

            with LocalRangeReader(root) as reader:
                for filename in (
                    "../outside.bin",
                    str(outside),
                    "leaf-link",
                    "dir-link/payload",
                ):
                    with self.subTest(filename=filename):
                        with self.assertRaises(RangeValidationError):
                            reader.get_range(filename, 0, 0)
                self.assertEqual(reader.get_range("inside.bin", 0, 5), b"inside")
                self.assertEqual(reader.budget.requests, 1)


if __name__ == "__main__":
    unittest.main()
