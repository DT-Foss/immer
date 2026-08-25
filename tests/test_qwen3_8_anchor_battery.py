from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import stat
import tempfile
import unittest

from immer.runtimes.qwen3_8.anchor_battery import (
    AnchorBatteryConflictError,
    AnchorBatteryController,
    AnchorBatteryError,
    AnchorBatteryIdentityError,
    AnchorBatteryIntegrityError,
    AnchorChargeCandidate,
    AnchorChargePolicy,
    ProfitLedger,
    RadixDemandMiner,
)


PIN = "7" * 64


def candidate(
    tokens: tuple[int, ...],
    *,
    live: float,
    charge: float = 1.0,
    store: float = 0.1,
    verify: float = 0.1,
    invalidation: float = 0.1,
    size: int = 100,
    demand: int = 2,
    surprise: float = 0.0,
    progress: float = 0.0,
) -> AnchorChargeCandidate:
    return AnchorChargeCandidate(
        token_ids=tokens,
        expected_live_cost=live,
        idle_charge_cost=charge,
        store_cost=store,
        verify_cost=verify,
        invalidation_cost=invalidation,
        bytes_stored=size,
        charge_seconds=charge,
        demand_count=demand,
        o1_surprise=surprise,
        o1_learning_progress=progress,
    )


class RadixDemandMinerTests(unittest.TestCase):
    def test_recurring_prefix_matches_unseen_question_suffix(self) -> None:
        miner = RadixDemandMiner(max_prefix_tokens=16)
        miner.observe((151644, 8948, 198, 11, 12), semantic_boundary_lengths=(3,))
        miner.observe((151644, 8948, 198, 31, 32), semantic_boundary_lengths=(3,))

        match = miner.longest_recurring_prefix(
            (151644, 8948, 198, 999, 1000), min_demand=2
        )
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.token_ids, (151644, 8948, 198))
        self.assertEqual(match.demand_count, 2)
        self.assertEqual(match.boundary_count, 2)

        recurring = miner.candidates(min_demand=2)
        self.assertEqual(recurring[0].token_ids, (151644, 8948, 198))
        self.assertNotIn("prompt", json.dumps(miner.to_document()))

    def test_exact_tokens_only_and_semantic_boundaries_are_validated(self) -> None:
        miner = RadixDemandMiner()
        with self.assertRaises(TypeError):
            miner.observe("raw prompt")  # type: ignore[arg-type]
        with self.assertRaises(AnchorBatteryError):
            miner.observe((1, 2), semantic_boundary_lengths=(3,))
        with self.assertRaises(AnchorBatteryError):
            miner.observe((1, True))  # type: ignore[list-item]

    def test_long_prefix_persistence_does_not_depend_on_python_recursion(self) -> None:
        tokens = tuple(range(1_500))
        miner = RadixDemandMiner(max_prefix_tokens=2_000)
        miner.observe(tokens)
        miner.observe(tokens)
        document = miner.to_document()
        reopened = RadixDemandMiner.from_document(document)
        rows = reopened.candidates(min_demand=2)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].token_ids, tokens)


class ChargePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = AnchorChargePolicy()

    def test_longest_useful_prefix_wins_value_density(self) -> None:
        short = candidate((1, 2), live=4.0, size=100)
        longest = candidate((1, 2, 3, 4), live=9.0, size=100)
        ranked = self.policy.rank((short, longest))
        self.assertEqual(ranked[0].token_ids, longest.token_ids)
        plan = self.policy.select(
            (short, longest), max_charge_bytes=100, max_charge_seconds=10.0
        )
        self.assertEqual(plan.selected, (longest,))
        self.assertEqual(plan.skipped_budget, (short,))

    def test_complete_profitability_inequality_rejects_loss(self) -> None:
        loss = candidate(
            (9,),
            live=4.0,
            charge=1.0,
            store=1.0,
            verify=1.0,
            invalidation=1.0,
        )
        gain = replace(loss, token_ids=(10,), expected_live_cost=4.0001)
        plan = self.policy.select(
            (loss, gain), max_charge_bytes=1_000, max_charge_seconds=10.0
        )
        self.assertEqual(plan.rejected_unprofitable, (loss,))
        self.assertEqual(plan.selected, (gain,))
        self.assertEqual(loss.net_value, 0.0)

    def test_value_density_controls_eviction_order(self) -> None:
        dense = candidate((1,), live=10.0, size=100)
        medium = candidate((2,), live=10.0, size=200)
        weak = candidate((3,), live=3.0, size=400)
        self.assertEqual(
            [
                row.token_ids
                for row in self.policy.eviction_order((dense, weak, medium))
            ],
            [weak.token_ids, medium.token_ids, dense.token_ids],
        )

    def test_o1_progress_is_explicit_tiebreak_not_surprise_replacement(self) -> None:
        stagnant_surprise = candidate((20,), live=7.0, surprise=1.0, progress=0.0)
        learning = candidate((21,), live=7.0, surprise=0.25, progress=0.5)
        ranked = self.policy.rank((stagnant_surprise, learning))
        self.assertEqual(ranked[0], learning)

        unprofitable_surprise = candidate(
            (22,),
            live=0.1,
            surprise=1.0,
            progress=1.0,
        )
        plan = self.policy.select(
            (unprofitable_surprise,),
            max_charge_bytes=10_000,
            max_charge_seconds=10.0,
        )
        self.assertFalse(plan.selected)
        self.assertEqual(plan.rejected_unprofitable, (unprofitable_surprise,))

        production_scale = candidate((23,), live=7.0, surprise=5.0, progress=5.0)
        self.assertGreater(production_scale.o1_progress_signal, 0.0)
        self.assertLess(production_scale.o1_progress_signal, 2.0)

    def test_selection_respects_both_idle_budgets_deterministically(self) -> None:
        first = candidate((31,), live=12.0, size=100, charge=2.0)
        second = candidate((32,), live=11.0, size=100, charge=2.0)
        third = candidate((33,), live=10.0, size=100, charge=2.0)
        one = self.policy.select(
            (third, first, second), max_charge_bytes=250, max_charge_seconds=4.0
        )
        two = self.policy.select(
            (second, third, first), max_charge_bytes=250, max_charge_seconds=4.0
        )
        self.assertEqual(one, two)
        self.assertEqual(one.selected, (first, second))
        self.assertEqual(one.charge_bytes, 200)
        self.assertEqual(one.charge_seconds, 4.0)


class ProfitLedgerTests(unittest.TestCase):
    def test_soc_hits_misses_self_discharge_and_wasted_recharge(self) -> None:
        ledger = ProfitLedger()
        tokens = (100, 200, 300)
        row = ledger.record_miss(tokens, live_seconds=5.0)
        self.assertEqual((row.demand_count, row.hits, row.misses), (1, 0, 1))

        row = ledger.record_charge(tokens, charge_seconds=2.0, bytes_stored=4096)
        row = ledger.record_hit(
            tokens,
            live_seconds=5.0,
            restore_seconds=0.5,
            verify_seconds=0.25,
        )
        self.assertEqual((row.demand_count, row.hits, row.misses), (2, 1, 1))
        self.assertAlmostEqual(row.saved_live_seconds, 4.25)
        self.assertEqual(row.soc, 1.0)

        row = ledger.invalidate(row.prefix_sha256)
        self.assertFalse(row.active)
        self.assertEqual(row.invalidated_charge_seconds, 2.0)
        self.assertEqual(row.wasted_charge_seconds, 0.0)

        row = ledger.record_charge(tokens, charge_seconds=1.0, bytes_stored=4096)
        row = ledger.invalidate(row.prefix_sha256)
        self.assertEqual(row.charge_seconds, 3.0)
        self.assertEqual(row.invalidated_charge_seconds, 3.0)
        self.assertEqual(row.wasted_charge_seconds, 1.0)
        self.assertAlmostEqual(row.self_discharge, 1.0 / 3.0)
        self.assertEqual(row.turnover_fraction, 1.0)
        self.assertAlmostEqual(row.waste_fraction, 1.0 / 3.0)
        self.assertAlmostEqual(ledger.self_discharge, 1.0 / 3.0)
        self.assertEqual(ledger.turnover_fraction, 1.0)
        self.assertAlmostEqual(ledger.wasted_charge_seconds, 1.0)

    def test_duplicate_active_charge_and_inactive_hit_are_rejected(self) -> None:
        ledger = ProfitLedger()
        tokens = (4, 5)
        with self.assertRaises(AnchorBatteryError):
            ledger.record_hit(
                tokens,
                live_seconds=1.0,
                restore_seconds=0.1,
                verify_seconds=0.1,
            )
        ledger.record_charge(tokens, charge_seconds=1.0, bytes_stored=10)
        with self.assertRaises(AnchorBatteryError):
            ledger.record_charge(tokens, charge_seconds=1.0, bytes_stored=10)


class ControllerPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="anchor-battery-")
        self.path = Path(self.temporary.name) / "battery.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _controller(self) -> AnchorBatteryController:
        controller = AnchorBatteryController(
            model_pin_sha256=PIN,
            miner=RadixDemandMiner(max_prefix_tokens=32),
        )
        controller.observe((10, 11, 12, 90), semantic_boundary_lengths=(3,))
        controller.observe((10, 11, 12, 91), semantic_boundary_lengths=(3,))
        controller.ledger.record_charge(
            (10, 11, 12), charge_seconds=2.5, bytes_stored=1024
        )
        controller.ledger.record_hit(
            (10, 11, 12),
            live_seconds=4.0,
            restore_seconds=0.4,
            verify_seconds=0.1,
        )
        return controller

    def test_canonical_reopen_and_resume_are_deterministic(self) -> None:
        original = self._controller()
        seal = original.save(self.path)
        self.assertEqual(len(seal), 64)
        raw = self.path.read_bytes()
        self.assertTrue(raw.endswith(b"\n"))
        self.assertNotIn(b"prompt", raw.lower())
        self.assertIn(b'"token_id":10', raw)
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)

        left = AnchorBatteryController.open(self.path, expected_model_pin_sha256=PIN)
        right = AnchorBatteryController.open(self.path, expected_model_pin_sha256=PIN)
        self.assertEqual(left.to_document(), right.to_document())
        self.assertEqual(
            left.miner.longest_recurring_prefix((10, 11, 12, 999)).token_ids,
            (10, 11, 12),
        )

        left.observe((10, 11, 12, 92), semantic_boundary_lengths=(3,))
        right.observe((10, 11, 12, 92), semantic_boundary_lengths=(3,))
        self.assertEqual(left.to_document(), right.to_document())

    def test_stale_reopened_writer_cannot_clobber_newer_generation(self) -> None:
        original = self._controller()
        original.save(self.path)
        first = AnchorBatteryController.open(self.path, expected_model_pin_sha256=PIN)
        stale = AnchorBatteryController.open(self.path, expected_model_pin_sha256=PIN)

        first.observe((10, 11, 12, 92), semantic_boundary_lengths=(3,))
        first.save(self.path)
        stale.observe((10, 11, 12, 93), semantic_boundary_lengths=(3,))
        with self.assertRaises(AnchorBatteryConflictError):
            stale.save(self.path)

        reopened = AnchorBatteryController.open(
            self.path, expected_model_pin_sha256=PIN
        )
        self.assertEqual(reopened.generation, first.generation)
        self.assertEqual(reopened.miner.observation_count, 3)

    def test_new_controller_cannot_overwrite_existing_state_without_cas(self) -> None:
        self._controller().save(self.path)
        replacement = self._controller()
        with self.assertRaises(AnchorBatteryConflictError):
            replacement.save(self.path)

    def test_tamper_and_wrong_model_pin_are_rejected(self) -> None:
        controller = self._controller()
        controller.save(self.path)
        with self.assertRaises(AnchorBatteryIdentityError):
            AnchorBatteryController.open(self.path, expected_model_pin_sha256="8" * 64)

        document = json.loads(self.path.read_text(encoding="utf-8"))
        document["body"]["miner"]["observation_count"] += 1
        self.path.write_text(
            json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(AnchorBatteryIntegrityError):
            AnchorBatteryController.open(self.path, expected_model_pin_sha256=PIN)

    def test_noncanonical_but_resealed_document_is_rejected(self) -> None:
        controller = self._controller()
        controller.save(self.path)
        document = json.loads(self.path.read_text(encoding="utf-8"))
        self.path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        with self.assertRaises(AnchorBatteryIntegrityError):
            AnchorBatteryController.open(self.path, expected_model_pin_sha256=PIN)

    def test_sensitive_token_sidecar_rejects_permissive_mode(self) -> None:
        self._controller().save(self.path)
        self.path.chmod(0o644)
        with self.assertRaises(AnchorBatteryIntegrityError):
            AnchorBatteryController.open(self.path, expected_model_pin_sha256=PIN)


if __name__ == "__main__":
    unittest.main()
