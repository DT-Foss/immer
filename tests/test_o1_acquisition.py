from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from immer.runtimes.o1_state.acquisition import (
    AcquisitionError,
    AcquisitionIntegrityError,
    AcquisitionLeakageError,
    AcquisitionLimits,
    DOptimalAcquisitionReceipt,
    encode_probe_job_structural_design,
    plan_d_optimal_acquisition,
    plan_probe_job_acquisition,
    replay_d_optimal_acquisition,
    replay_probe_job_acquisition,
    selected_probe_jobs,
)
from immer.runtimes.o1_state.cartographer import (
    O1Cartographer,
    ProbeJob,
    ProbeTarget,
    build_probe_frontier,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _frontier() -> tuple[ProbeJob, ...]:
    return build_probe_frontier(
        layers=(0, 9, 18, 27),
        targets=(
            ProbeTarget("mlp", "block", 0),
            ProbeTarget("mlp", "block", 1),
        ),
        probe_families=("generation-a", "generation-b"),
        interventions=("passive", "placebo"),
        code_pin="code-v1",
        model_pin="qwen-local-v1",
        seed=17,
        prompt_hashes_by_family={
            "generation-a": _hash("prompt-a"),
            "generation-b": _hash("prompt-b"),
        },
    )


class _Stream:
    loss_ema: float | None = None

    def observe(self, text: str) -> None:
        del text

    def snapshot(self) -> dict[str, object]:
        return {"schema": "test-stream/v1"}

    def restore(self, state: dict[str, object]) -> None:
        if state != {"schema": "test-stream/v1"}:
            raise ValueError("wrong stream state")


class DOptimalAcquisitionTests(unittest.TestCase):
    def test_structural_plan_roundtrips_and_feeds_o1_cartographer(self) -> None:
        jobs = _frontier()
        receipt = plan_probe_job_acquisition(
            reversed(jobs), budget=9, normalization="l2"
        )
        restored = DOptimalAcquisitionReceipt.from_bytes(receipt.to_bytes())
        self.assertEqual(restored, receipt)
        selected = replay_probe_job_acquisition(restored, jobs)
        self.assertEqual(
            tuple(job.job_id for job in selected), receipt.selected_job_ids
        )
        self.assertEqual(len(selected), 9)
        self.assertTrue(all(gain >= 0.0 for gain in receipt.step_logdet_gains))

        with tempfile.TemporaryDirectory() as temporary:
            scheduler = O1Cartographer(
                jobs=selected,
                code_pin="code-v1",
                model_pin="qwen-local-v1",
                state_path=Path(temporary) / "o1.json",
                stream=_Stream(),
            )
            self.assertEqual(scheduler.coverage().total_jobs, 9)
            self.assertIn(scheduler.next_job(), selected)

    def test_structural_encoder_binds_prompt_identity(self) -> None:
        target = ProbeTarget("mlp", "block", 0)
        first = ProbeJob.create(
            layer=18,
            target=target,
            probe_family="shared-family",
            intervention="passive",
            code_pin="code-v1",
            model_pin="qwen-local-v1",
            seed=1,
            prompt_sha256=_hash("first prompt"),
        )
        second = ProbeJob.create(
            layer=18,
            target=target,
            probe_family="shared-family",
            intervention="passive",
            code_pin="code-v1",
            model_pin="qwen-local-v1",
            seed=2,
            prompt_sha256=_hash("second prompt"),
        )
        rows, feature_pin = encode_probe_job_structural_design((first, second))
        self.assertNotEqual(rows[first.job_id], rows[second.job_id])
        self.assertEqual(len(feature_pin), 64)

    def test_rank_deficient_design_is_deterministic(self) -> None:
        jobs = _frontier()[:6]
        rows = {job.job_id: (0.0, 0.0, 0.0) for job in jobs}
        kwargs = dict(
            design_abi="rank-deficient/v1",
            feature_pin_sha256=_hash("rank-deficient-schema"),
            budget=4,
        )
        first = plan_d_optimal_acquisition(jobs, rows, **kwargs)
        second = plan_d_optimal_acquisition(reversed(jobs), rows, **kwargs)
        self.assertEqual(first.to_bytes(), second.to_bytes())
        self.assertEqual(first.selected_job_ids, tuple(sorted(rows))[:4])
        self.assertEqual(first.step_logdet_gains, (0.0, 0.0, 0.0, 0.0))

    def test_targets_and_holdout_values_are_never_inputs(self) -> None:
        jobs = _frontier()[:4]
        rows = {job.job_id: (1.0, float(index)) for index, job in enumerate(jobs)}
        common = dict(
            design_abi="leakage-test/v1",
            feature_pin_sha256=_hash("leakage-schema"),
            budget=2,
        )
        with self.assertRaises(AcquisitionLeakageError):
            plan_d_optimal_acquisition(jobs, rows, target_values=(1, 2), **common)
        with self.assertRaises(AcquisitionLeakageError):
            plan_probe_job_acquisition(jobs, budget=2, holdout_values={"y": 1})
        polluted = {job.job_id: {"feature": [1.0], "target": 1} for job in jobs}
        with self.assertRaises(AcquisitionLeakageError):
            plan_d_optimal_acquisition(jobs, polluted, **common)

    def test_receipt_tamper_and_noncanonical_json_are_rejected(self) -> None:
        receipt = plan_probe_job_acquisition(_frontier(), budget=4)
        document = json.loads(receipt.to_bytes())
        document["body"]["budget"] = 3
        tampered = json.dumps(
            document, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        with self.assertRaisesRegex(AcquisitionIntegrityError, "SHA-256"):
            DOptimalAcquisitionReceipt.from_bytes(tampered)
        with self.assertRaisesRegex(AcquisitionIntegrityError, "canonical"):
            DOptimalAcquisitionReceipt.from_bytes(
                json.dumps(json.loads(receipt.to_bytes()), indent=2).encode("utf-8")
            )
        with self.assertRaises(AcquisitionLeakageError):
            replace(receipt, target_values_used=True)

    def test_replay_rejects_changed_design_and_frontier(self) -> None:
        jobs = _frontier()[:8]
        rows = {
            job.job_id: (1.0, float(job.layer), float(job.target.unit_index or 0))
            for job in jobs
        }
        receipt = plan_d_optimal_acquisition(
            jobs,
            rows,
            design_abi="replay/v1",
            feature_pin_sha256=_hash("replay-schema"),
            budget=4,
        )
        changed = dict(rows)
        changed[jobs[0].job_id] = (1.0, 999.0, 0.0)
        with self.assertRaisesRegex(AcquisitionIntegrityError, "differs"):
            replay_d_optimal_acquisition(receipt, jobs, changed)
        with self.assertRaisesRegex(AcquisitionIntegrityError, "frontier"):
            selected_probe_jobs(receipt, jobs[:-1])

    def test_capacity_and_design_shape_are_enforced(self) -> None:
        jobs = _frontier()[:4]
        with self.assertRaises(AcquisitionError):
            plan_probe_job_acquisition(
                jobs,
                budget=3,
                limits=AcquisitionLimits(candidate_capacity=3),
            )
        rows = {job.job_id: (1.0, 2.0) for job in jobs}
        rows[jobs[0].job_id] = (1.0,)
        with self.assertRaisesRegex(AcquisitionError, "inconsistent"):
            plan_d_optimal_acquisition(
                jobs,
                rows,
                design_abi="shape/v1",
                feature_pin_sha256=_hash("shape-schema"),
                budget=2,
            )


if __name__ == "__main__":
    unittest.main()
