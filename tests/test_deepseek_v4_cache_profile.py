from __future__ import annotations

import fcntl
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from immer.runtimes.deepseek_v4.benchmark import canonical_digest
from immer.runtimes.deepseek_v4.cache_profile import (
    CACHE_PROFILE_SCHEMA,
    CacheProfileError,
    CacheWorkload,
    CacheWorkloadOperation,
    atomic_write_json,
    build_report,
    create_owned_run,
    load_phase,
    validate_owned_run,
)


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "deepseek_v4_cache_profile.py"
MIB = 1024**2
DATASET_SHA256 = "a" * 64
PROMPT_SHA256 = "b" * 64


def _write_fixture(root: Path) -> None:
    root.mkdir(parents=True)
    tensors = {
        "left.weight": np.arange(12, dtype="<f4").reshape(4, 3),
        "right.weight": np.linspace(-2.0, 2.0, 15, dtype="<f4").reshape(5, 3),
    }
    header: dict[str, object] = {}
    payloads: list[bytes] = []
    offset = 0
    for name, array in tensors.items():
        payload = np.ascontiguousarray(array).tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    header["__metadata__"] = {"fixture": "deepseek-v4-cache-profile"}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"".join(payloads)
    )


def _workload(source: Path) -> CacheWorkload:
    return CacheWorkload(
        source=str(source.resolve()),
        source_kind="local",
        revision="local",
        dataset_sha256=DATASET_SHA256,
        prompt_sha256=PROMPT_SHA256,
        mode="off",
        seed=17,
        operations=(
            CacheWorkloadOperation("left.weight"),
            CacheWorkloadOperation("right.weight", start_row=1, n_rows=2),
        ),
    )


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    pythonpath = str(ROOT / "src")
    if environment.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{environment['PYTHONPATH']}"
    environment["PYTHONPATH"] = pythonpath
    return environment


def _run_script(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        cwd=ROOT,
        env=_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _run_cold_worker(
    run_root: Path, owner_token: str
) -> subprocess.CompletedProcess[str]:
    return _run_script(
        "--_worker-phase",
        "cold",
        "--_run-root",
        str(run_root),
        "--_owner-token",
        owner_token,
        "--_source-budget-bytes",
        str(2 * MIB),
        "--_cache-budget-bytes",
        str(2 * MIB),
    )


def _fresh_run(source: Path, output_root: Path) -> subprocess.CompletedProcess[str]:
    return _run_script(
        "--output-root",
        str(output_root),
        "--source",
        str(source),
        "--revision",
        "local",
        "--dataset-sha256",
        DATASET_SHA256,
        "--prompt-sha256",
        PROMPT_SHA256,
        "--mode",
        "off",
        "--seed",
        "17",
        "--tensor",
        "left.weight",
        "--rows",
        "right.weight,1,2",
        "--source-budget-mb",
        "2",
        "--cache-budget-mb",
        "2",
    )


def _receipt(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    if result.returncode != 0:
        raise AssertionError(
            f"command failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1])


class CacheProfileWorkloadTests(unittest.TestCase):
    def test_workload_identity_rejects_unknown_or_changed_fields(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            source = Path(raw) / "source"
            _write_fixture(source)
            workload = _workload(source)
            document = workload.to_dict()
            self.assertEqual(CacheWorkload.from_dict(document), workload)

            with self.subTest("unknown field"):
                changed = dict(document)
                changed["note"] = "not signed"
                with self.assertRaisesRegex(CacheProfileError, "fields"):
                    CacheWorkload.from_dict(changed)

            with self.subTest("signed input changed"):
                changed = dict(document)
                changed["mode"] = "crsa"
                with self.assertRaisesRegex(CacheProfileError, "signature"):
                    CacheWorkload.from_dict(changed)

            with self.subTest("operation order changed"):
                changed = dict(document)
                changed["operations"] = list(reversed(changed["operations"]))
                with self.assertRaisesRegex(CacheProfileError, "workload keys"):
                    CacheWorkload.from_dict(changed)

    def test_atomic_json_refuses_to_follow_a_foreign_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            foreign = root / "foreign.json"
            foreign.write_text("sentinel", encoding="utf-8")
            link = root / "report.json"
            link.symlink_to(foreign)
            with self.assertRaisesRegex(CacheProfileError, "symlink"):
                atomic_write_json(link, {"would": "overwrite"})
            self.assertEqual(foreign.read_text(encoding="utf-8"), "sentinel")


class CacheProfileProcessTests(unittest.TestCase):
    def test_real_cold_warm_hot_protocol_and_transport_counters(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            output = root / "runs"
            _write_fixture(source)
            output.mkdir()
            sibling = output / "foreign-sibling.txt"
            sibling.write_text("keep", encoding="utf-8")

            receipt = _receipt(_fresh_run(source, output))
            run_root = Path(receipt["run_root"])
            report = json.loads(Path(receipt["report"]).read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], CACHE_PROFILE_SCHEMA)
            self.assertEqual(report["scope"], "streamer_transport_cache_only")
            self.assertFalse(report["quality_or_end_to_end_performance_claim"])
            self.assertEqual(
                report["report_sha256"],
                canonical_digest(
                    {
                        key: value
                        for key, value in report.items()
                        if key != "report_sha256"
                    }
                ),
            )
            self.assertEqual(sibling.read_text(encoding="utf-8"), "keep")
            self.assertEqual(len(tuple(output.glob("cache-profile-*"))), 1)
            self.assertEqual(
                report["provenance"]["cache_profile_module_sha256"],
                hashlib.sha256(
                    (
                        ROOT / "src/immer/runtimes/deepseek_v4/cache_profile.py"
                    ).read_bytes()
                ).hexdigest(),
            )

            cold = report["states"]["cold"]
            warm = report["states"]["warm"]
            hot = report["states"]["hot"]
            self.assertNotEqual(cold["pid"], warm["pid"])
            self.assertEqual(warm["pid"], hot["pid"])
            self.assertEqual(warm["reader_instance"], hot["reader_instance"])
            self.assertEqual(cold["workload_keys"], warm["workload_keys"])
            self.assertEqual(warm["workload_keys"], hot["workload_keys"])

            cold_counters = cold["counters"]
            self.assertEqual(cold["cache_files_before"], 0)
            self.assertEqual(cold["cache_bytes_on_disk_before"], 0)
            self.assertEqual(cold["source_budget_limit_bytes"], 2 * MIB)
            self.assertEqual(cold["cache_limit_bytes"], 2 * MIB)
            self.assertGreater(cold_counters["network_or_source_body_bytes"], 0)
            self.assertEqual(
                cold_counters["network_or_source_body_bytes"],
                cold_counters["budget_bytes_body"],
            )
            self.assertGreater(cold_counters["cache_misses"], 0)
            self.assertGreater(cold_counters["cache_writes"], 0)
            self.assertGreater(cold_counters["cache_bytes_written"], 0)

            for state in (warm, hot):
                counters = state["counters"]
                self.assertGreater(state["cache_files_before"], 0)
                self.assertGreater(state["cache_bytes_on_disk_before"], 0)
                self.assertEqual(state["source_budget_limit_bytes"], 2 * MIB)
                self.assertEqual(state["cache_limit_bytes"], 2 * MIB)
                self.assertEqual(counters["network_or_source_body_bytes"], 0)
                self.assertEqual(counters["budget_bytes_body"], 0)
                self.assertEqual(counters["cache_misses"], 0)
                self.assertEqual(counters["cache_writes"], 0)
                self.assertEqual(counters["cache_hits"], 2)
                self.assertGreater(counters["cache_bytes_reused"], 0)
            self.assertEqual(
                cold["cache_bytes_after"],
                warm["cache_bytes_after"],
            )
            self.assertEqual(warm["cache_bytes_after"], hot["cache_bytes_after"])
            self.assertEqual(warm["cache_files_before"], hot["cache_files_before"])
            self.assertEqual(
                warm["cache_bytes_on_disk_before"],
                hot["cache_bytes_on_disk_before"],
            )
            self.assertTrue((run_root / "cache").is_dir())
            self.assertEqual(
                load_phase(run_root / "cold.json", "cold")["phase"], "cold"
            )
            self.assertEqual(
                load_phase(run_root / "warm-hot.json", "warm-hot")["phase"],
                "warm-hot",
            )

    def test_resume_uses_stored_workload_and_exact_budgets(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            output = root / "runs"
            _write_fixture(source)
            workload = _workload(source)
            run_root, owner_token = create_owned_run(
                output,
                workload,
                source_budget_bytes=2 * MIB,
                cache_budget_bytes=2 * MIB,
            )
            cold = _run_cold_worker(run_root, owner_token)
            self.assertEqual(cold.returncode, 0, cold.stderr)
            self.assertFalse((run_root / "warm-hot.json").exists())

            lock_descriptor = os.open(
                run_root / ".cache-profile.lock", os.O_CREAT | os.O_RDWR, 0o600
            )
            try:
                fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                concurrent = _run_script("--resume", str(run_root))
                self.assertEqual(concurrent.returncode, 2)
                self.assertIn("another cache-profile process", concurrent.stderr)
                self.assertFalse((run_root / "warm-hot.json").exists())
            finally:
                os.close(lock_descriptor)

            changed = _run_script("--resume", str(run_root), "--mode", "crsa")
            self.assertEqual(changed.returncode, 2)
            self.assertIn("immutable stored workload", changed.stderr)
            self.assertFalse((run_root / "warm-hot.json").exists())

            receipt = _receipt(_run_script("--resume", str(run_root)))
            report = json.loads(Path(receipt["report"]).read_text(encoding="utf-8"))
            self.assertEqual(report["workload"], workload.to_dict())
            self.assertEqual(report["budgets"]["source_bytes_per_process"], 2 * MIB)
            self.assertEqual(report["budgets"]["cache_bytes"], 2 * MIB)

            phase_stat = (run_root / "warm-hot.json").stat()
            _receipt(_run_script("--resume", str(run_root)))
            self.assertEqual(
                (run_root / "warm-hot.json").stat().st_ino, phase_stat.st_ino
            )
            self.assertEqual(
                (run_root / "warm-hot.json").stat().st_mtime_ns,
                phase_stat.st_mtime_ns,
            )

    def test_worker_rejects_budget_substitution_before_cache_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            _write_fixture(source)
            run_root, owner_token = create_owned_run(
                root / "runs",
                _workload(source),
                source_budget_bytes=2 * MIB,
                cache_budget_bytes=2 * MIB,
            )
            result = _run_script(
                "--_worker-phase",
                "cold",
                "--_run-root",
                str(run_root),
                "--_owner-token",
                owner_token,
                "--_source-budget-bytes",
                str(MIB),
                "--_cache-budget-bytes",
                str(2 * MIB),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("budgets do not match", result.stderr)
            self.assertFalse((run_root / "cache").exists())
            self.assertFalse((run_root / "cold.json").exists())

    def test_missing_absolute_local_source_never_falls_back_to_remote(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            output = root / "runs"
            missing = root / "missing-checkpoint"
            result = _run_script(
                "--output-root",
                str(output),
                "--source",
                str(missing),
                "--dataset-sha256",
                DATASET_SHA256,
                "--prompt-sha256",
                PROMPT_SHA256,
                "--tensor",
                "left.weight",
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("local --source does not exist", result.stderr)
            self.assertFalse(output.exists())

    def test_cold_rejects_prefilled_owned_cache_without_deleting_it(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            _write_fixture(source)
            run_root, owner_token = create_owned_run(
                root / "runs",
                _workload(source),
                source_budget_bytes=2 * MIB,
                cache_budget_bytes=2 * MIB,
            )
            cache_dir = run_root / "cache"
            cache_dir.mkdir()
            foreign = cache_dir / "foreign-cache-entry.bin"
            foreign.write_bytes(b"must-not-be-evicted")

            result = _run_cold_worker(run_root, owner_token)
            self.assertEqual(result.returncode, 2)
            self.assertIn("COLD requires an empty", result.stderr)
            self.assertEqual(foreign.read_bytes(), b"must-not-be-evicted")
            self.assertFalse((run_root / "cold.json").exists())

    def test_corrupted_owner_and_workload_manifests_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            _write_fixture(source)
            run_root, _ = create_owned_run(
                root / "runs",
                _workload(source),
                source_budget_bytes=2 * MIB,
                cache_budget_bytes=2 * MIB,
            )
            marker_path = run_root / "OWNER.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["created_ns"] += 1
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(CacheProfileError, "digest mismatch"):
                validate_owned_run(run_root)

            marker["marker_sha256"] = canonical_digest(
                {key: value for key, value in marker.items() if key != "marker_sha256"}
            )
            marker_path.write_text(json.dumps(marker), encoding="utf-8")
            workload_path = run_root / "workload.json"
            workload = json.loads(workload_path.read_text(encoding="utf-8"))
            workload["untrusted_note"] = "ignored by old readers"
            workload_path.write_text(json.dumps(workload), encoding="utf-8")
            with self.assertRaisesRegex(CacheProfileError, "fields"):
                validate_owned_run(run_root)

    def test_corrupted_or_semantically_relabelled_phase_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            _write_fixture(source)
            run_root, owner_token = create_owned_run(
                root / "runs",
                _workload(source),
                source_budget_bytes=2 * MIB,
                cache_budget_bytes=2 * MIB,
            )
            cold_result = _run_cold_worker(run_root, owner_token)
            self.assertEqual(cold_result.returncode, 0, cold_result.stderr)
            cold_path = run_root / "cold.json"
            original = json.loads(cold_path.read_text(encoding="utf-8"))

            corrupted = json.loads(json.dumps(original))
            corrupted["cold"]["counters"]["cache_hits"] += 1
            cold_path.write_text(json.dumps(corrupted), encoding="utf-8")
            result = _run_script("--resume", str(run_root))
            self.assertEqual(result.returncode, 2)
            self.assertIn("digest mismatch", result.stderr)
            self.assertFalse((run_root / "warm-hot.json").exists())

            relabelled = json.loads(json.dumps(original))
            relabelled["cold"]["claimed_state"] = "warm"
            relabelled["phase_sha256"] = canonical_digest(
                {
                    key: value
                    for key, value in relabelled.items()
                    if key != "phase_sha256"
                }
            )
            cold_path.write_text(json.dumps(relabelled), encoding="utf-8")
            result = _run_script("--resume", str(run_root))
            self.assertEqual(result.returncode, 2)
            self.assertIn("not admissibly labelled", result.stderr)
            self.assertFalse((run_root / "warm-hot.json").exists())

    def test_report_symlink_is_rejected_without_touching_foreign_target(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            _write_fixture(source)
            run_root, owner_token = create_owned_run(
                root / "runs",
                _workload(source),
                source_budget_bytes=2 * MIB,
                cache_budget_bytes=2 * MIB,
            )
            cold_result = _run_cold_worker(run_root, owner_token)
            self.assertEqual(cold_result.returncode, 0, cold_result.stderr)
            foreign = root / "foreign-report.json"
            foreign.write_text("unchanged", encoding="utf-8")
            (run_root / "report.json").symlink_to(foreign)

            result = _run_script("--resume", str(run_root))
            self.assertEqual(result.returncode, 2)
            self.assertIn("must not replace a symlink", result.stderr)
            self.assertEqual(foreign.read_text(encoding="utf-8"), "unchanged")

    def test_paired_validator_rejects_process_or_counter_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "source"
            output = root / "runs"
            _write_fixture(source)
            receipt = _receipt(_fresh_run(source, output))
            run_root = Path(receipt["run_root"])
            cold_envelope = load_phase(run_root / "cold.json", "cold")
            warm_hot_envelope = load_phase(run_root / "warm-hot.json", "warm-hot")
            harness_sha = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()

            forged = json.loads(json.dumps(warm_hot_envelope))
            forged["warm"]["pid"] = cold_envelope["cold"]["pid"]
            forged["phase_sha256"] = canonical_digest(
                {key: value for key, value in forged.items() if key != "phase_sha256"}
            )
            with self.assertRaisesRegex(CacheProfileError, "different processes"):
                build_report(
                    run_root=run_root,
                    workload=_workload(source),
                    cold_envelope=cold_envelope,
                    warm_hot_envelope=forged,
                    source_budget_bytes=2 * MIB,
                    cache_budget_bytes=2 * MIB,
                    command=("fixture",),
                    harness_sha256=harness_sha,
                )

            forged = json.loads(json.dumps(warm_hot_envelope))
            forged["hot"]["counters"]["network_or_source_body_bytes"] = 1
            forged["phase_sha256"] = canonical_digest(
                {key: value for key, value in forged.items() if key != "phase_sha256"}
            )
            with self.assertRaisesRegex(CacheProfileError, "aggregate counters"):
                build_report(
                    run_root=run_root,
                    workload=_workload(source),
                    cold_envelope=cold_envelope,
                    warm_hot_envelope=forged,
                    source_budget_bytes=2 * MIB,
                    cache_budget_bytes=2 * MIB,
                    command=("fixture",),
                    harness_sha256=harness_sha,
                )


if __name__ == "__main__":
    unittest.main()
