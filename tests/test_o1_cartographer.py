from __future__ import annotations

import hashlib
import multiprocessing
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import immer.runtimes.o1_state.cartographer as cartographer_module
from immer.runtimes.o1_state.cartographer import (
    CartographyBudget,
    CartographyError,
    CartographyIdentityError,
    CartographyIntegrityError,
    O1Cartographer,
    ProbeJob,
    ProbeOutcome,
    ProbeTarget,
    RetryPolicy,
    build_probe_frontier,
    canonical_observation_bytes,
)


CODE_PIN = "code-0123456789abcdef"
MODEL_PIN = "qwen-local-causal-0123456789abcdef"


class FakeO1Stream:
    def __init__(self, losses: list[float] | None = None) -> None:
        self.loss_ema: float | None = 0.0
        self.losses = list(losses or [])
        self.observed: list[str] = []
        self.tokens = 0

    def observe(self, text: str) -> None:
        self.observed.append(text)
        self.tokens += len(text.encode("utf-8"))
        if self.losses:
            self.loss_ema = self.losses.pop(0)

    def snapshot(self):
        return {"loss_ema": self.loss_ema, "tokens": self.tokens}

    def restore(self, state):
        value = state.get("loss_ema")
        self.loss_ema = float(value) if value is not None else None
        self.tokens = int(state.get("tokens", 0))


def frontier(
    *,
    layers=(0, 1),
    families=("arithmetic", "coreference"),
    interventions=("baseline",),
    read_budget_bytes=None,
    model_budget_seconds=None,
):
    return build_probe_frontier(
        layers=layers,
        targets=(ProbeTarget("attention", "head", 0),),
        probe_families=families,
        interventions=interventions,
        code_pin=CODE_PIN,
        model_pin=MODEL_PIN,
        seed=17,
        read_budget_bytes=read_budget_bytes,
        model_budget_seconds=model_budget_seconds,
    )


def _concurrent_step_worker(state_path, ready, start, results) -> None:
    try:
        scheduler = O1Cartographer.restore(
            state_path,
            code_pin=CODE_PIN,
            model_pin=MODEL_PIN,
            stream=FakeO1Stream(),
        )
        ready.put("ready")
        if not start.wait(15):
            raise RuntimeError("concurrent writer start timed out")
        outcome = scheduler.step(
            lambda job, attempt: {
                "attempt": attempt,
                "layer": job.layer,
                "writer_pid": os.getpid(),
            }
        )
        results.put(("ok", outcome.job_id, outcome.attempt))
    except BaseException as exc:
        results.put(("error", type(exc).__name__, str(exc)))


def _concurrent_receipt_worker(
    state_path,
    job_id,
    attempt,
    receipt_sha256,
    ready,
    start,
    results,
) -> None:
    try:
        scheduler = O1Cartographer.restore(
            state_path,
            code_pin=CODE_PIN,
            model_pin=MODEL_PIN,
            stream=FakeO1Stream(),
        )
        ready.put("ready")
        if not start.wait(15):
            raise RuntimeError("concurrent receipt start timed out")
        scheduler.attach_atlas_receipt(
            job_id=job_id,
            attempt=attempt,
            receipt_sha256=receipt_sha256,
        )
        results.put(("ok", job_id, receipt_sha256))
    except BaseException as exc:
        results.put(("error", type(exc).__name__, str(exc)))


class O1CartographerTests(unittest.TestCase):
    def make_scheduler(self, root: Path, **kwargs) -> O1Cartographer:
        return O1Cartographer(
            jobs=kwargs.pop("jobs", frontier()),
            code_pin=CODE_PIN,
            model_pin=MODEL_PIN,
            state_path=root / "cartography.json",
            stream=kwargs.pop("stream", FakeO1Stream()),
            seed=17,
            **kwargs,
        )

    def test_selection_is_deterministic_for_explicit_seed(self) -> None:
        with (
            tempfile.TemporaryDirectory() as first_tmp,
            tempfile.TemporaryDirectory() as second_tmp,
        ):
            first = self.make_scheduler(Path(first_tmp))
            second = self.make_scheduler(Path(second_tmp))
            first_order = []
            second_order = []
            for scheduler, order in ((first, first_order), (second, second_order)):
                while (job := scheduler.next_job()) is not None:
                    order.append(job.job_id)
                    scheduler.step(lambda selected, attempt: {"value": selected.layer})
            self.assertEqual(first_order, second_order)
            self.assertEqual(len(first_order), len(set(first_order)))

    def test_full_coverage_terminates_without_duplicate_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = self.make_scheduler(Path(tmp))
            calls = []

            def execute(job, attempt):
                calls.append((job.job_id, attempt))
                return {"activation_norm": job.layer + 0.5}

            report = scheduler.run(execute, max_jobs=100, max_seconds=10)
            self.assertEqual(len(report.outcomes), len(frontier()))
            self.assertTrue(report.coverage.complete)
            self.assertEqual(report.stop_reason, "coverage-complete")
            self.assertEqual(len(calls), len(set(calls)))
            self.assertIsNone(scheduler.step(execute))

    def test_completed_frontier_extends_additively_and_resumes_exactly(self) -> None:
        initial = frontier(layers=(0,), families=("a",))
        extension = frontier(layers=(7, 8), families=("b",))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scheduler = self.make_scheduler(root, jobs=initial)
            initial_generation = scheduler.generation
            first_report = scheduler.run(
                lambda job, _attempt: {"layer": job.layer},
                max_jobs=10,
                max_seconds=10,
            )
            self.assertTrue(first_report.coverage.complete)
            old_outcomes = scheduler.outcomes
            completed_generation = scheduler.generation
            self.assertGreater(completed_generation, initial_generation)

            added = scheduler.extend_jobs(extension)
            self.assertEqual(
                added, tuple(sorted(extension, key=lambda job: job.job_id))
            )
            self.assertEqual(scheduler.outcomes, old_outcomes)
            self.assertEqual(scheduler.coverage().total_jobs, 3)
            self.assertEqual(scheduler.coverage().succeeded_jobs, 1)
            self.assertEqual(scheduler.coverage().uncovered_jobs, 2)
            self.assertFalse(scheduler.coverage().complete)
            self.assertEqual(scheduler.generation, completed_generation + 1)
            with self.assertRaises(AttributeError):
                setattr(scheduler, "generation", 0)

            resumed = O1Cartographer.restore(
                scheduler.state_path,
                code_pin=CODE_PIN,
                model_pin=MODEL_PIN,
                stream=FakeO1Stream(),
            )
            self.assertEqual(resumed.jobs, scheduler.jobs)
            self.assertEqual(resumed.outcomes, old_outcomes)
            self.assertEqual(resumed.generation, scheduler.generation)
            second_report = resumed.run(
                lambda job, _attempt: {"layer": job.layer},
                max_jobs=10,
                max_seconds=10,
            )
            self.assertEqual(len(second_report.outcomes), 2)
            self.assertTrue(second_report.coverage.complete)
            self.assertEqual(
                {row.job_id for row in resumed.outcomes},
                {row.job_id for row in (*initial, *extension)},
            )

    def test_frontier_extension_is_idempotent_and_collapses_exact_duplicates(
        self,
    ) -> None:
        initial = frontier(layers=(0,), families=("a",))
        extension = frontier(layers=(9,), families=("b",))
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = self.make_scheduler(Path(tmp), jobs=initial)
            before = scheduler.state_path.read_bytes()
            self.assertEqual(scheduler.extend_jobs((*initial, *initial)), ())
            self.assertEqual(scheduler.state_path.read_bytes(), before)

            added = scheduler.extend_jobs((*extension, *extension))
            self.assertEqual(added, extension)
            committed = scheduler.state_path.read_bytes()
            self.assertEqual(scheduler.extend_jobs(extension), ())
            self.assertEqual(scheduler.state_path.read_bytes(), committed)
            self.assertEqual(len(scheduler.jobs), 2)

    def test_frontier_extension_rejects_wrong_pins_payload_collisions_and_policy_breaks(
        self,
    ) -> None:
        initial = frontier(
            layers=(0,),
            families=("a",),
            read_budget_bytes=10,
            model_budget_seconds=1.0,
        )
        wrong_pin = ProbeJob.create(
            layer=1,
            target=ProbeTarget("attention", "head", 0),
            probe_family="b",
            intervention="baseline",
            code_pin="other-code-pin",
            model_pin=MODEL_PIN,
            seed=19,
            read_budget_bytes=10,
            model_budget_seconds=1.0,
        )
        without_caps = frontier(layers=(2,), families=("c",))[0]
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = self.make_scheduler(
                Path(tmp),
                jobs=initial,
                budget=CartographyBudget(
                    max_total_read_bytes=100,
                    max_total_model_seconds=10.0,
                ),
            )
            original = scheduler.state_path.read_bytes()
            with self.assertRaises(CartographyIdentityError):
                scheduler.extend_jobs((wrong_pin,))
            with self.assertRaisesRegex(CartographyError, "read cap"):
                scheduler.extend_jobs((without_caps,))
            self.assertEqual(scheduler.state_path.read_bytes(), original)

        model_capped_initial = frontier(
            layers=(0,), families=("a",), model_budget_seconds=1.0
        )
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = self.make_scheduler(
                Path(tmp),
                jobs=model_capped_initial,
                budget=CartographyBudget(max_total_model_seconds=10.0),
            )
            with self.assertRaisesRegex(CartographyError, "model-time cap"):
                scheduler.extend_jobs((without_caps,))

        prompt = "same identity, different persisted carrier"
        prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
        identity = dict(
            layer=3,
            target=ProbeTarget("mlp", "block", 0),
            probe_family="collision",
            intervention="baseline",
            code_pin=CODE_PIN,
            model_pin=MODEL_PIN,
            seed=23,
            prompt_sha256=prompt_sha256,
        )
        hash_only = ProbeJob.create(**identity)
        with_prompt = ProbeJob.create(**identity, prompt=prompt)
        self.assertEqual(hash_only.job_id, with_prompt.job_id)
        self.assertNotEqual(hash_only, with_prompt)
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = self.make_scheduler(
                Path(tmp), jobs=(hash_only,), allow_raw_prompts=True
            )
            with self.assertRaisesRegex(
                CartographyIntegrityError, "collides with an existing job id"
            ):
                scheduler.extend_jobs((with_prompt,))

    def test_frontier_extension_preserves_histories_replay_and_stream_snapshot(
        self,
    ) -> None:
        initial = frontier(layers=(0,), families=("a",))
        extension = frontier(layers=(4,), families=("b",))
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = self.make_scheduler(
                Path(tmp), jobs=initial, stream=FakeO1Stream([6.0])
            )
            scheduler.step(lambda job, _attempt: {"layer": job.layer, "norm": 3.0})
            _, before = O1Cartographer._read_document(scheduler.state_path)

            scheduler.extend_jobs(extension)
            _, after = O1Cartographer._read_document(scheduler.state_path)
            for field in ("histories", "outcomes", "replay", "stream_state"):
                self.assertEqual(after[field], before[field], field)
            self.assertEqual(after["generation"], before["generation"] + 1)
            self.assertEqual(scheduler.replay_count, 1)
            self.assertEqual(scheduler.outcomes[0].job_id, initial[0].job_id)

    def test_frontier_extension_is_crash_safe_before_and_after_atomic_replace(
        self,
    ) -> None:
        initial = frontier(layers=(0,), families=("a",))
        extension = frontier(layers=(5,), families=("b",))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scheduler = self.make_scheduler(root, jobs=initial)
            original = scheduler.state_path.read_bytes()
            with mock.patch.object(
                cartographer_module.os,
                "replace",
                side_effect=OSError("simulated pre-replace power loss"),
            ):
                with self.assertRaises(OSError):
                    scheduler.extend_jobs(extension)
            self.assertEqual(scheduler.jobs, initial)
            self.assertEqual(scheduler.state_path.read_bytes(), original)
            self.assertFalse(tuple(root.glob("*.pending")))
            restored_old = O1Cartographer.restore(
                scheduler.state_path,
                code_pin=CODE_PIN,
                model_pin=MODEL_PIN,
                stream=FakeO1Stream(),
            )
            self.assertEqual(restored_old.jobs, initial)

            original_atomic_write = O1Cartographer._atomic_write

            def commit_then_interrupt(path, data):
                original_atomic_write(path, data)
                raise OSError("simulated post-replace durability interruption")

            with mock.patch.object(
                O1Cartographer,
                "_atomic_write",
                staticmethod(commit_then_interrupt),
            ):
                with self.assertRaises(OSError):
                    scheduler.extend_jobs(extension)
            self.assertEqual(set(scheduler.jobs), set((*initial, *extension)))
            restored_new = O1Cartographer.restore(
                scheduler.state_path,
                code_pin=CODE_PIN,
                model_pin=MODEL_PIN,
                stream=FakeO1Stream(),
            )
            self.assertEqual(restored_new.jobs, scheduler.jobs)
            committed = scheduler.state_path.read_bytes()
            self.assertEqual(scheduler.extend_jobs(extension), ())
            self.assertEqual(scheduler.state_path.read_bytes(), committed)

    def test_stale_frontier_extender_cannot_overwrite_a_committed_extension(
        self,
    ) -> None:
        initial = frontier(layers=(0,), families=("a",))
        first_extension = frontier(layers=(1,), families=("b",))
        stale_extension = frontier(layers=(2,), families=("c",))
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = self.make_scheduler(Path(tmp), jobs=initial)
            stale = O1Cartographer.restore(
                scheduler.state_path,
                code_pin=CODE_PIN,
                model_pin=MODEL_PIN,
                stream=FakeO1Stream(),
            )
            scheduler.extend_jobs(first_extension)
            with self.assertRaisesRegex(
                CartographyIntegrityError, "changed since the last checkpoint"
            ):
                stale.extend_jobs(stale_extension)

            final = O1Cartographer.restore(
                scheduler.state_path,
                code_pin=CODE_PIN,
                model_pin=MODEL_PIN,
                stream=FakeO1Stream(),
            )
            self.assertEqual(set(final.jobs), set((*initial, *first_extension)))

    def test_frontier_extension_has_no_job_count_cap_below_sidecar_limit(self) -> None:
        initial = frontier(layers=(0,), families=("a",))
        extension = frontier(layers=range(1, 1026), families=("bulk",))
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = self.make_scheduler(Path(tmp), jobs=initial)
            self.assertEqual(len(scheduler.extend_jobs(extension)), 1025)
            self.assertEqual(scheduler.coverage().total_jobs, 1026)
            self.assertLess(
                len(scheduler.state_path.read_bytes()),
                cartographer_module._MAX_STATE_BYTES,
            )

    def test_frontier_extension_rejects_only_when_sidecar_resource_limit_is_hit(
        self,
    ) -> None:
        initial = frontier(layers=(0,), families=("a",))
        extension = frontier(layers=(1,), families=("b",))
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = self.make_scheduler(Path(tmp), jobs=initial)
            original = scheduler.state_path.read_bytes()
            with mock.patch.object(
                cartographer_module, "_MAX_STATE_BYTES", len(original) + 1
            ):
                with self.assertRaisesRegex(CartographyError, "state exceeds"):
                    scheduler.extend_jobs(extension)
            self.assertEqual(scheduler.jobs, initial)
            self.assertEqual(scheduler.state_path.read_bytes(), original)

    def test_surprise_reprioritizes_uncovered_sibling_family(self) -> None:
        jobs = frontier(layers=(0, 1), families=("a", "b"))
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = self.make_scheduler(
                Path(tmp), jobs=jobs, stream=FakeO1Stream([10.0])
            )
            first = scheduler.next_job()
            scheduler.step(lambda _job, _attempt: {"activation_norm": 123.0})
            second = scheduler.next_job()
            self.assertNotEqual(first.job_id, second.job_id)
            self.assertEqual(second.probe_family, first.probe_family)

    def test_surprise_reprioritizes_uncovered_sibling_prompt(self) -> None:
        prompt_a = hashlib.sha256(b"prompt-a").hexdigest()
        prompt_b = hashlib.sha256(b"prompt-b").hexdigest()
        jobs = tuple(
            ProbeJob.create(
                layer=index,
                target=ProbeTarget(f"module-{prompt_index}-{index}", "module", None),
                probe_family="shared-family",
                intervention="baseline",
                code_pin=CODE_PIN,
                model_pin=MODEL_PIN,
                seed=prompt_index * 10 + index,
                prompt_sha256=prompt,
            )
            for prompt_index, prompt in enumerate((prompt_a, prompt_b))
            for index in (0, 1)
        )
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = self.make_scheduler(
                Path(tmp), jobs=jobs, stream=FakeO1Stream([10.0])
            )
            first = scheduler.next_job()
            scheduler.step(lambda _job, _attempt: {"activation_norm": 123.0})
            second = scheduler.next_job()
            self.assertNotEqual(first.job_id, second.job_id)
            self.assertEqual(second.prompt_sha256, first.prompt_sha256)

    def test_crash_resume_preserves_aborted_attempt_then_retries(self) -> None:
        jobs = frontier(layers=(0,), families=("a",))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scheduler = self.make_scheduler(root, jobs=jobs)
            original = O1Cartographer._atomic_write
            writes = 0

            def crash_second_write(path, data):
                nonlocal writes
                writes += 1
                if writes == 2:
                    raise OSError("simulated power loss")
                return original(path, data)

            with mock.patch.object(
                O1Cartographer, "_atomic_write", staticmethod(crash_second_write)
            ):
                with self.assertRaises(OSError):
                    scheduler.step(lambda _job, _attempt: {"value": 1})

            resumed = O1Cartographer.restore(
                root / "cartography.json",
                code_pin=CODE_PIN,
                model_pin=MODEL_PIN,
                stream=FakeO1Stream(),
            )
            self.assertEqual(resumed.outcomes[0].status, "aborted")
            self.assertEqual(resumed.outcomes[0].attempt, 1)
            retried = resumed.step(lambda job, attempt: {"value": attempt})
            self.assertEqual(retried.attempt, 2)
            self.assertTrue(resumed.coverage().complete)

    def test_restore_rejects_stale_code_or_model_pin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_scheduler(root)
            with self.assertRaises(CartographyIdentityError):
                O1Cartographer.restore(
                    root / "cartography.json",
                    code_pin="stale-code",
                    model_pin=MODEL_PIN,
                    stream=FakeO1Stream(),
                )
            with self.assertRaises(CartographyIdentityError):
                O1Cartographer.restore(
                    root / "cartography.json",
                    code_pin=CODE_PIN,
                    model_pin="stale-model",
                    stream=FakeO1Stream(),
                )

    def test_persistent_budget_stops_before_unaffordable_job(self) -> None:
        jobs = frontier(layers=(0, 1), families=("a",), read_budget_bytes=100)
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = self.make_scheduler(
                Path(tmp),
                jobs=jobs,
                budget=CartographyBudget(max_total_read_bytes=150),
            )
            calls = 0

            def execute(job, attempt):
                nonlocal calls
                calls += 1
                return ProbeOutcome.succeeded(
                    job, attempt, {"value": 1}, read_bytes=100
                )

            report = scheduler.run(execute, max_jobs=10, max_seconds=10)
            self.assertEqual(calls, 1)
            self.assertEqual(report.stop_reason, "budget")
            self.assertFalse(report.coverage.complete)

    def test_failed_jobs_follow_explicit_retry_policy_and_remain_auditable(
        self,
    ) -> None:
        jobs = frontier(layers=(0,), families=("a",))
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = self.make_scheduler(
                Path(tmp),
                jobs=jobs,
                retry_policy=RetryPolicy(max_attempts=2, retry_failed=True),
            )
            report = scheduler.run(
                lambda job, attempt: ProbeOutcome.failed(
                    job, attempt, error_code="probe-failed", error=f"attempt-{attempt}"
                ),
                max_jobs=10,
                max_seconds=10,
            )
            self.assertEqual([item.attempt for item in report.outcomes], [1, 2])
            self.assertEqual(
                [item.status for item in report.outcomes], ["failed", "failed"]
            )
            self.assertEqual(report.coverage.failed_terminal_jobs, 1)
            self.assertTrue(report.coverage.complete)

        with tempfile.TemporaryDirectory() as tmp:
            no_retry = self.make_scheduler(
                Path(tmp),
                jobs=jobs,
                retry_policy=RetryPolicy(max_attempts=3, retry_failed=False),
            )
            report = no_retry.run(
                lambda job, attempt: ProbeOutcome.failed(
                    job, attempt, error_code="probe-failed", error="once"
                ),
                max_jobs=10,
                max_seconds=10,
            )
            self.assertEqual(len(report.outcomes), 1)
            self.assertTrue(report.coverage.complete)

    def test_checkpoint_restore_keeps_exact_frontier_and_bounded_replay(
        self,
    ) -> None:
        jobs = frontier(layers=(0, 1, 2), families=("a",))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scheduler = self.make_scheduler(
                root,
                jobs=jobs,
                replay_items=2,
                replay_bytes=1024,
            )
            scheduler.step(lambda job, _attempt: {"layer": job.layer})
            expected_next = scheduler.next_job().job_id
            expected_outcomes = scheduler.outcomes

            restored_stream = FakeO1Stream()
            restored = O1Cartographer.restore(
                root / "cartography.json",
                code_pin=CODE_PIN,
                model_pin=MODEL_PIN,
                stream=restored_stream,
            )
            self.assertEqual(restored.outcomes, expected_outcomes)
            self.assertEqual(restored.next_job().job_id, expected_next)
            restored.step(lambda job, _attempt: {"layer": job.layer})
            restored.step(lambda job, _attempt: {"layer": job.layer})
            self.assertEqual(restored.replay_count, 2)
            before = len(restored_stream.observed)
            self.assertEqual(restored.replay(), 2)
            self.assertEqual(len(restored_stream.observed) - before, 2)

    def test_concurrent_process_writers_never_silently_lose_state(self) -> None:
        jobs = frontier(layers=(0, 1, 2), families=("a",))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scheduler = self.make_scheduler(root, jobs=jobs)
            first = scheduler.step(lambda job, _attempt: {"layer": job.layer})
            receipt = hashlib.sha256(b"preexisting-atlas-receipt").hexdigest()
            scheduler.attach_atlas_receipt(
                job_id=first.job_id,
                attempt=first.attempt,
                receipt_sha256=receipt,
            )

            context = multiprocessing.get_context("spawn")
            ready = context.Queue()
            start = context.Event()
            results = context.Queue()
            processes = [
                context.Process(
                    target=_concurrent_step_worker,
                    args=(scheduler.state_path, ready, start, results),
                )
                for _ in range(2)
            ]
            for process in processes:
                process.start()
            self.assertEqual(
                [ready.get(timeout=15) for _ in processes], ["ready", "ready"]
            )
            start.set()
            rows = [results.get(timeout=20) for _ in processes]
            for process in processes:
                process.join(timeout=20)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                self.assertEqual(process.exitcode, 0)

            self.assertEqual(sum(row[0] == "ok" for row in rows), 1)
            errors = [row for row in rows if row[0] == "error"]
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0][1], "CartographyIntegrityError")

            restored = O1Cartographer.restore(
                scheduler.state_path,
                code_pin=CODE_PIN,
                model_pin=MODEL_PIN,
                stream=FakeO1Stream(),
            )
            self.assertEqual(len(restored.outcomes), 2)
            self.assertEqual(restored.outcomes[0].atlas_receipt_sha256, receipt)
            self.assertEqual(len({row.attempt_id for row in restored.outcomes}), 2)
            _, document = O1Cartographer._read_document(restored.state_path)
            family_history = next(
                row
                for row in document["histories"]
                if row["kind"] == "probe_family" and row["name"] == "a"
            )
            self.assertEqual(len(family_history["values"]), 2)

            restored.step(lambda job, _attempt: {"layer": job.layer})
            unreceipted = [
                row for row in restored.outcomes if row.atlas_receipt_sha256 is None
            ]
            self.assertEqual(len(unreceipted), 2)
            receipt_by_job = {
                row.job_id: hashlib.sha256(f"receipt:{row.job_id}".encode()).hexdigest()
                for row in unreceipted
            }
            ready = context.Queue()
            start = context.Event()
            results = context.Queue()
            processes = [
                context.Process(
                    target=_concurrent_receipt_worker,
                    args=(
                        restored.state_path,
                        row.job_id,
                        row.attempt,
                        receipt_by_job[row.job_id],
                        ready,
                        start,
                        results,
                    ),
                )
                for row in unreceipted
            ]
            for process in processes:
                process.start()
            self.assertEqual(
                [ready.get(timeout=15) for _ in processes], ["ready", "ready"]
            )
            start.set()
            rows = [results.get(timeout=20) for _ in processes]
            for process in processes:
                process.join(timeout=20)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(sum(row[0] == "ok" for row in rows), 1)
            self.assertEqual(
                [row[1] for row in rows if row[0] == "error"],
                ["CartographyIntegrityError"],
            )

            final = O1Cartographer.restore(
                restored.state_path,
                code_pin=CODE_PIN,
                model_pin=MODEL_PIN,
                stream=FakeO1Stream(),
            )
            self.assertEqual(len(final.outcomes), 3)
            self.assertEqual(
                sum(row.atlas_receipt_sha256 is not None for row in final.outcomes), 2
            )
            missing = next(
                row for row in final.outcomes if row.atlas_receipt_sha256 is None
            )
            final.attach_atlas_receipt(
                job_id=missing.job_id,
                attempt=missing.attempt,
                receipt_sha256=receipt_by_job[missing.job_id],
            )
            self.assertEqual(final.coverage().promoted_jobs, 3)
            _, final_document = O1Cartographer._read_document(final.state_path)
            final_family_history = next(
                row
                for row in final_document["histories"]
                if row["kind"] == "probe_family" and row["name"] == "a"
            )
            self.assertEqual(len(final_family_history["values"]), 3)

    def test_descriptor_read_rejects_symlink_and_path_swap_toctou(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scheduler = self.make_scheduler(root)
            state_path = scheduler.state_path
            original = state_path.read_bytes()

            replacement = root / "replacement.json"
            replacement.write_bytes(original)
            original_lstat = cartographer_module.os.lstat
            swapped = False

            def swap_before_identity_check(path):
                nonlocal swapped
                if Path(path) == state_path and not swapped:
                    swapped = True
                    os.replace(replacement, state_path)
                return original_lstat(path)

            with mock.patch.object(
                cartographer_module.os, "lstat", side_effect=swap_before_identity_check
            ):
                with self.assertRaisesRegex(
                    CartographyIntegrityError, "changed during descriptor open"
                ):
                    O1Cartographer._read_document(state_path)

            real_state = root / "real-state.json"
            real_state.write_bytes(original)
            state_path.unlink()
            state_path.symlink_to(real_state)
            with self.assertRaises(CartographyIntegrityError):
                O1Cartographer._read_document(state_path)

    def test_descriptor_read_rejects_oversized_state_before_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            oversized = Path(tmp) / "oversized.json"
            with oversized.open("wb") as handle:
                handle.truncate(64 * 1024 * 1024 + 1)
            with self.assertRaisesRegex(CartographyIntegrityError, "exceeds"):
                O1Cartographer._read_document(oversized)

    def test_sidecar_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scheduler = self.make_scheduler(root)
            path = scheduler.state_path
            raw = bytearray(path.read_bytes())
            raw[raw.index(b'"generation"') + 2] ^= 1
            path.write_bytes(raw)
            with self.assertRaises(CartographyIntegrityError):
                O1Cartographer.restore(
                    path,
                    code_pin=CODE_PIN,
                    model_pin=MODEL_PIN,
                    stream=FakeO1Stream(),
                )

    def test_observation_bytes_are_canonical_and_raw_prompt_fields_are_blocked(
        self,
    ) -> None:
        jobs = frontier(layers=(0,), families=("a",))
        with tempfile.TemporaryDirectory() as tmp:
            stream = FakeO1Stream()
            scheduler = self.make_scheduler(Path(tmp), jobs=jobs, stream=stream)
            outcome = scheduler.step(
                lambda _job, _attempt: {"z": [2, 1], "a": {"norm": 3.0}}
            )
            expected = canonical_observation_bytes({"z": [2, 1], "a": {"norm": 3.0}})
            self.assertEqual(outcome.observation_bytes, expected)
            self.assertEqual(stream.observed[-1].encode(), expected)

        with tempfile.TemporaryDirectory() as tmp:
            scheduler = self.make_scheduler(Path(tmp), jobs=jobs)
            outcome = scheduler.step(
                lambda _job, _attempt: {"raw_prompt": "PRIVATE PROMPT"}
            )
            self.assertEqual(outcome.status, "failed")
            self.assertEqual(outcome.error_code, "raw-observation-rejected")
            self.assertNotIn(b"PRIVATE PROMPT", scheduler.state_path.read_bytes())

    def test_semantic_promotion_requires_external_atlas_receipt(self) -> None:
        jobs = frontier(layers=(0,), families=("a",))
        with tempfile.TemporaryDirectory() as tmp:
            scheduler = self.make_scheduler(Path(tmp), jobs=jobs)
            outcome = scheduler.step(lambda _job, _attempt: {"value": 7})
            self.assertFalse(outcome.semantically_promoted)
            receipt = hashlib.sha256(b"external-atlas-receipt").hexdigest()
            promoted = scheduler.attach_atlas_receipt(
                job_id=outcome.job_id,
                attempt=outcome.attempt,
                receipt_sha256=receipt,
            )
            self.assertTrue(promoted.semantically_promoted)
            self.assertEqual(scheduler.coverage().promoted_jobs, 1)
            with self.assertRaises(CartographyIntegrityError):
                scheduler.attach_atlas_receipt(
                    job_id=outcome.job_id,
                    attempt=outcome.attempt,
                    receipt_sha256=hashlib.sha256(b"replacement").hexdigest(),
                )

    def test_frontier_rejects_duplicates_and_raw_prompt_defaults_to_hash_only(
        self,
    ) -> None:
        with self.assertRaises(CartographyError):
            build_probe_frontier(
                layers=(0, 0),
                targets=(ProbeTarget("attention", "head", 0),),
                probe_families=("a",),
                interventions=("baseline",),
                code_pin=CODE_PIN,
                model_pin=MODEL_PIN,
                seed=1,
            )
        prompt = "do not persist me"
        jobs = build_probe_frontier(
            layers=(0,),
            targets=(ProbeTarget("attention", "head", 0),),
            probe_families=("a",),
            interventions=("baseline",),
            code_pin=CODE_PIN,
            model_pin=MODEL_PIN,
            seed=1,
            prompt_hashes_by_family={"a": hashlib.sha256(prompt.encode()).hexdigest()},
        )
        self.assertIsNone(jobs[0].prompt)
        self.assertEqual(
            jobs[0].prompt_sha256, hashlib.sha256(prompt.encode()).hexdigest()
        )


if __name__ == "__main__":
    unittest.main()
