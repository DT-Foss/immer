from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

from immer.runtimes.ooe.compute_crystals import (
    CAUSAL_MIX_FLOAT64,
    ComputeCrystal,
    ComputeCrystalABIError,
    ComputeCrystalBank,
)
from immer.runtimes.ooe.compute_graph import (
    CAUSAL_MIX_FUSION_VERIFIER_SHA256,
    OPERATOR_GRAPH_COMMIT_PREFIX,
    OPERATOR_GRAPH_STATE_NAME,
    ComputeOperatorGraph,
    ComputeOperatorGraphConflictError,
    ComputeOperatorGraphIntegrityError,
    ComputeRoutePlan,
    ComputeRouteUnavailableError,
    MaterializedRoute,
    OperatorEdge,
    RouteChargeReceipt,
    RouteDischargeReceipt,
)
from immer.runtimes.ooe.crystal import ManifestConflictError
from immer.runtimes.ooe.identity import canonical_json_bytes


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _edge(
    source: str,
    target: str,
    crystal: ComputeCrystal,
    ordinal: int,
) -> OperatorEdge:
    return OperatorEdge(
        source_state=source,
        target_state=target,
        crystal_sha256=crystal.sha256,
        verifier_sha256=_digest(f"verifier-{ordinal}"),
        evidence_sha256=_digest(f"evidence-{ordinal}"),
    )


class ComputeOperatorGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bank = ComputeCrystalBank(self.temporary.name)
        self.graph = ComputeOperatorGraph(self.bank)

    def _publish_edges(
        self,
        crystals: tuple[ComputeCrystal, ...],
        states: tuple[str, ...],
        *,
        offset: int = 0,
    ) -> tuple[OperatorEdge, ...]:
        for crystal in crystals:
            self.bank.publish_crystal(crystal)
        edges = tuple(
            _edge(states[index], states[index + 1], crystal, index + offset)
            for index, crystal in enumerate(crystals)
        )
        self.graph.append_edges(edges)
        return edges

    def test_operator_edge_is_canonical_generic_and_immutable(self) -> None:
        crystal = ComputeCrystal.affine([[2.0]], [1.0])
        edge = _edge("raw", "normalised", crystal, 0)
        restored = OperatorEdge.from_bytes(edge.to_bytes())

        self.assertEqual(restored, edge)
        self.assertEqual(edge.action, edge.sha256)
        text = edge.to_bytes().decode("utf-8")
        self.assertNotIn("qwen", text.lower())
        self.assertNotIn("model_pin", text)
        with self.assertRaises(ComputeOperatorGraphIntegrityError):
            OperatorEdge.from_bytes(edge.to_bytes() + b"\n")

    def test_unseen_multiedge_task_is_planned_only_from_one_step_evidence(self) -> None:
        crystals = tuple(
            ComputeCrystal.affine([[float(scale)]], [float(offset)])
            for scale, offset in ((2, 1), (3, -2), (-1, 4), (0.5, 8))
        )
        edges = self._publish_edges(crystals, ("s0", "s1", "s2", "s3", "s4"))

        decision = self.graph.plan_route("s0", "s4")

        self.assertFalse(decision.abstained)
        self.assertIsNotNone(decision.plan)
        plan = decision.plan
        assert plan is not None
        self.assertEqual(
            plan.primitive_edge_sha256s, tuple(edge.sha256 for edge in edges)
        )
        self.assertEqual(
            plan.finite_plan.expected_states, ("s0", "s1", "s2", "s3", "s4")
        )
        self.assertEqual(plan.finite_plan.expected_actions, plan.primitive_edge_sha256s)
        self.assertEqual(ComputeRoutePlan.from_bytes(plan.to_bytes()), plan)
        # No multi-step evidence can exist: the world contains exactly four events.
        self.assertEqual(self.graph.build_world_model().total_events, 4)

    def test_causal_mix_route_fuses_and_charges_without_markov_ledger(self) -> None:
        kernels = (
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.7, 0.3, 0.0, 0.0],
                    [0.2, 0.5, 0.3, 0.0],
                    [0.1, 0.2, 0.3, 0.4],
                ],
                dtype=np.float64,
            ),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.4, 0.6, 0.0, 0.0],
                    [0.1, 0.2, 0.7, 0.0],
                    [0.25, 0.25, 0.25, 0.25],
                ],
                dtype=np.float64,
            ),
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.8, 0.2, 0.0, 0.0],
                    [0.4, 0.2, 0.4, 0.0],
                    [0.1, 0.3, 0.2, 0.4],
                ],
                dtype=np.float64,
            ),
        )
        crystals = tuple(ComputeCrystal.causal_mix(kernel) for kernel in kernels)
        self._publish_edges(crystals, ("p0", "p1", "p2", "p3"))
        decision = self.graph.plan_route("p0", "p3")
        assert decision.plan is not None
        charged = self.graph.charge_route(decision.plan)

        self.assertEqual(
            charged.route.charge_verifier_sha256,
            CAUSAL_MIX_FUSION_VERIFIER_SHA256,
        )
        self.assertIsNone(charged.route.contraction_ledger)
        assert charged.route.fused_crystal_sha256 is not None
        fused = self.bank.restore_crystal(charged.route.fused_crystal_sha256)
        self.assertEqual(fused.operator_kind, CAUSAL_MIX_FLOAT64)
        values = (
            np.random.default_rng(20260826).normal(size=(2, 3, 4)).astype(np.float64)
        )
        expected = values
        for crystal in crystals:
            expected = crystal.apply(expected)
        discharged = self.graph.discharge_exact(charged.route.sha256, values)
        np.testing.assert_allclose(
            discharged.output,
            expected,
            rtol=1e-15,
            atol=1e-15,
        )
        self.assertGreater(discharged.receipt.historical_work_released, 0)
        reopened = ComputeOperatorGraph(ComputeCrystalBank(self.temporary.name))
        self.assertEqual(reopened.state(), self.graph.state())
        np.testing.assert_allclose(
            reopened.discharge_exact(charged.route.sha256, values).output,
            expected,
            rtol=1e-15,
            atol=1e-15,
        )

    def test_exact_path_selects_requested_parallel_edge_and_charges_roundtrip(
        self,
    ) -> None:
        alternatives = (
            ComputeCrystal.affine([[2.0]], [1.0]),
            ComputeCrystal.affine([[-5.0]], [7.0]),
        )
        tail = ComputeCrystal.affine([[3.0]], [-2.0])
        for crystal in (*alternatives, tail):
            self.bank.publish_crystal(crystal)
        parallel = (
            _edge("raw", "hidden", alternatives[0], 200),
            _edge("raw", "hidden", alternatives[1], 201),
        )
        tail_edge = _edge("hidden", "answer", tail, 202)
        self.graph.append_edges((*parallel, tail_edge))

        canonical = self.graph.plan_route("raw", "answer")
        assert canonical.plan is not None
        canonical_first = canonical.plan.primitive_edge_sha256s[0]
        requested = next(edge for edge in parallel if edge.sha256 != canonical_first)
        plan = self.graph.plan_exact_path((requested.sha256, tail_edge.sha256))

        self.assertEqual(
            plan.primitive_edge_sha256s,
            (requested.sha256, tail_edge.sha256),
        )
        self.assertEqual(plan.finite_plan.expected_states, ("raw", "hidden", "answer"))
        self.assertEqual(
            tuple(entry.action for entry in plan.finite_plan.policy),
            plan.primitive_edge_sha256s,
        )
        self.assertEqual(ComputeRoutePlan.from_bytes(plan.to_bytes()), plan)
        reopened = ComputeOperatorGraph(ComputeCrystalBank(self.temporary.name))
        self.assertEqual(
            reopened.plan_exact_path(plan.primitive_edge_sha256s),
            plan,
        )

        charged = self.graph.charge_route(plan)
        self.assertEqual(
            charged.route.primitive_edge_sha256s,
            plan.primitive_edge_sha256s,
        )
        self.assertIsNotNone(charged.route.charge_basis_sha256)
        self.assertTrue(charged.charge_basis_created)
        value = np.array([[-4.0], [0.25], [19.0]], dtype=np.float64)
        requested_crystal = self.bank.restore_crystal(requested.crystal_sha256)
        expected = tail.apply(requested_crystal.apply(value))
        discharged = self.graph.discharge("raw", "answer", value)
        np.testing.assert_array_equal(discharged.output, expected)
        self.assertEqual(
            discharged.route.primitive_edge_sha256s,
            plan.primitive_edge_sha256s,
        )
        self.assertGreater(discharged.receipt.historical_work_released, 0)

        canonical_charge = self.graph.charge_route(canonical.plan)
        endpoint_routes = tuple(
            route
            for route in self.graph.state().materialized_routes
            if route.source_state == "raw" and route.goal_state == "answer"
        )
        self.assertEqual(len(endpoint_routes), 2)
        exact_discharge = self.graph.discharge_exact(charged.route.sha256, value)
        np.testing.assert_array_equal(exact_discharge.output, expected)
        self.assertEqual(exact_discharge.route.sha256, charged.route.sha256)
        self.assertEqual(
            exact_discharge.receipt.route_sha256,
            charged.route.sha256,
        )
        canonical_edge = next(
            edge for edge in parallel if edge.sha256 == canonical_first
        )
        canonical_crystal = self.bank.restore_crystal(canonical_edge.crystal_sha256)
        canonical_expected = tail.apply(canonical_crystal.apply(value))
        canonical_discharge = self.graph.query_exact(
            canonical_charge.route.sha256,
            value,
        )
        np.testing.assert_array_equal(canonical_discharge.output, canonical_expected)
        self.assertEqual(
            canonical_discharge.route.sha256,
            canonical_charge.route.sha256,
        )
        self.assertNotEqual(
            canonical_discharge.route.sha256,
            exact_discharge.route.sha256,
        )
        with self.assertRaisesRegex(ComputeRouteUnavailableError, "absent"):
            self.graph.discharge_exact(_digest("missing-route"), value)

        single = self.graph.plan_exact_path((requested.sha256,))
        single_charge = self.graph.charge_route(single)
        self.assertIsNone(single_charge.route.charge_basis_sha256)
        self.assertFalse(single_charge.charge_basis_created)
        self.assertEqual(
            single_charge.route.equivalent_source_work_units,
            single_charge.route.live_work_units,
        )

    def test_exact_path_rejects_missing_disconnected_tampered_and_stale_head(
        self,
    ) -> None:
        crystals = (
            ComputeCrystal.affine([[2.0]], [1.0]),
            ComputeCrystal.affine([[3.0]], [2.0]),
            ComputeCrystal.affine([[5.0]], [-4.0]),
        )
        for crystal in crystals:
            self.bank.publish_crystal(crystal)
        connected = (
            _edge("a", "b", crystals[0], 210),
            _edge("b", "c", crystals[1], 211),
        )
        detached = _edge("x", "y", crystals[2], 212)
        head, _changed = self.graph.append_edges((*connected, detached))

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.graph.plan_exact_path(())
        with self.assertRaisesRegex(ComputeRouteUnavailableError, "absent"):
            self.graph.plan_exact_path((_digest("missing-edge"),))
        with self.assertRaisesRegex(ComputeRouteUnavailableError, "disconnected"):
            self.graph.plan_exact_path((connected[0].sha256, detached.sha256))

        plan = self.graph.plan_exact_path(tuple(edge.sha256 for edge in connected))
        tampered_finite = replace(
            plan.finite_plan,
            evidence_hashes=(_digest("forged-evidence"),),
        )
        tampered = ComputeRoutePlan(
            graph_generation=plan.graph_generation,
            graph_state_sha256=plan.graph_state_sha256,
            finite_plan=tampered_finite,
            primitive_edge_sha256s=plan.primitive_edge_sha256s,
        )
        with self.assertRaisesRegex(
            ComputeOperatorGraphIntegrityError, "not reproducible"
        ):
            self.graph.charge_route(tampered)

        anchored = ComputeOperatorGraph(
            self.bank,
            trusted_graph_state_sha256=head.sha256,
        )
        extension = ComputeCrystal.affine([[7.0]], [0.0])
        self.bank.publish_crystal(extension)
        self.graph.append_edge(_edge("c", "d", extension, 213))
        with self.assertRaisesRegex(
            ComputeOperatorGraphIntegrityError, "trusted anchor"
        ):
            anchored.plan_exact_path(tuple(edge.sha256 for edge in connected))

    def test_exact_path_rejects_restored_crystal_abi_and_weak_evidence(self) -> None:
        wide = ComputeCrystal.affine(np.eye(2), np.zeros(2))
        narrow = ComputeCrystal.affine([[1.0]], [0.0])
        weak = ComputeCrystal.affine([[2.0]], [1.0])
        for crystal in (wide, narrow, weak):
            self.bank.publish_crystal(crystal)
        incompatible = (
            _edge("a", "b", wide, 220),
            _edge("b", "c", narrow, 221),
        )
        weak_edge = replace(_edge("u", "v", weak, 222), weight=0.5)
        self.graph.append_edges((*incompatible, weak_edge))

        with self.assertRaisesRegex(ComputeCrystalABIError, "ABI mismatch"):
            self.graph.plan_exact_path(tuple(edge.sha256 for edge in incompatible))
        with self.assertRaisesRegex(
            ComputeOperatorGraphIntegrityError, "one-step evidence"
        ):
            self.graph.plan_exact_path((weak_edge.sha256,))
        self.assertEqual(self.graph.state().materialized_routes, ())

    def test_affine_segment_monoid_runs_future_inputs_as_one_live_operator(
        self,
    ) -> None:
        crystals = (
            ComputeCrystal.affine([[2.0]], [1.0]),
            ComputeCrystal.affine([[-3.0]], [4.0]),
            ComputeCrystal.affine([[0.5]], [-2.0]),
        )
        self._publish_edges(crystals, ("raw", "scaled", "centred", "answer"))
        decision = self.graph.plan_route("raw", "answer")
        assert decision.plan is not None
        charged = self.graph.charge_route(decision.plan)

        self.assertIsNotNone(charged.route.fused_crystal_sha256)
        self.assertIsNotNone(charged.route.charge_basis_sha256)
        self.assertTrue(charged.charge_basis_created)
        charge = self.bank.restore_charge(charged.route.charge_basis_sha256 or "")
        self.assertEqual(
            charge.source_program_sha256, charged.route.primitive_program_sha256
        )
        self.assertEqual(
            charge.fused_crystal_sha256, charged.route.fused_crystal_sha256
        )
        self.assertGreater(charged.composition_work_units, 0)
        self.assertEqual(
            MaterializedRoute.from_bytes(charged.route.to_bytes()), charged.route
        )
        self.assertEqual(RouteChargeReceipt.from_bytes(charged.to_bytes()), charged)
        future = np.array([[-7.25], [0.125], [91.0]], dtype=np.float64)
        expected = future
        for crystal in crystals:
            expected = crystal.apply(expected)
        discharged = self.graph.discharge("raw", "answer", future)

        np.testing.assert_array_equal(discharged.output, expected)
        self.assertEqual(discharged.receipt.vm_receipt.executed_operator_count, 1)
        self.assertEqual(
            discharged.receipt.vm_receipt.charge_basis_sha256,
            charged.route.charge_basis_sha256,
        )
        self.assertEqual(discharged.receipt.application_count, 3)
        self.assertGreater(discharged.receipt.historical_work_released, 0)
        self.assertEqual(
            discharged.receipt.graph_state_sha256, self.graph.state().sha256
        )
        self.assertEqual(
            discharged.receipt.world_model_sha256,
            decision.plan.world_model_sha256,
        )
        self.assertEqual(
            RouteDischargeReceipt.from_bytes(discharged.receipt.to_bytes()),
            discharged.receipt,
        )

        warm = self.graph.charge_route(decision.plan)
        self.assertFalse(warm.graph_changed)
        self.assertEqual(warm.composition_work_units, 0)
        second = self.graph.query(
            "raw", "answer", np.array([[1234.5]], dtype=np.float64)
        )
        self.assertEqual(second.receipt.vm_receipt.executed_operator_count, 1)

    def test_unreachable_and_topology_action_placebos_abstain(self) -> None:
        identity = ComputeCrystal.affine([[1.0]], [0.0])
        self.bank.publish_crystal(identity)
        real = _edge("a", "b", identity, 1)
        disconnected = _edge("c", "d", identity, 2)
        self.graph.append_edges((real, disconnected))

        unreachable = self.graph.plan_route("a", "d")
        self.assertTrue(unreachable.abstained)
        self.assertEqual(unreachable.reason, "unreachable-or-uncovered")

        # The edge SHA is the action: changing topology changes the action identity.
        placebo = replace(real, target_state="d", evidence_sha256=_digest("placebo"))
        self.assertNotEqual(placebo.action, real.action)
        other_dir = tempfile.TemporaryDirectory()
        self.addCleanup(other_dir.cleanup)
        other_bank = ComputeCrystalBank(other_dir.name)
        other_bank.publish_crystal(identity)
        other_graph = ComputeOperatorGraph(other_bank)
        other_graph.append_edge(placebo)
        placebo_plan = other_graph.plan_route("a", "d")
        self.assertFalse(placebo_plan.abstained)
        assert placebo_plan.plan is not None
        self.assertEqual(placebo_plan.plan.primitive_edge_sha256s, (placebo.sha256,))

        with self.assertRaises(ComputeRouteUnavailableError):
            self.graph.discharge("a", "d", np.array([1.0], dtype=np.float64))

    def test_mixed_abi_route_fails_before_graph_materialization(self) -> None:
        affine = ComputeCrystal.affine(np.eye(2), np.zeros(2))
        lookup = ComputeCrystal.lookup([[1.0, 2.0], [3.0, 4.0]])
        self._publish_edges((affine, lookup), ("a", "b", "c"))
        decision = self.graph.plan_route("a", "c")
        assert decision.plan is not None
        generation = self.graph.state().generation

        with self.assertRaises(ComputeCrystalABIError):
            self.graph.charge_route(decision.plan)
        self.assertEqual(self.graph.state().generation, generation)
        self.assertEqual(self.graph.state().materialized_routes, ())

    def test_forged_but_self_consistent_plan_cannot_be_charged(self) -> None:
        crystals = (
            ComputeCrystal.affine([[2.0]], [1.0]),
            ComputeCrystal.affine([[3.0]], [2.0]),
        )
        self._publish_edges(crystals, ("a", "b", "c"))
        decision = self.graph.plan_route("a", "c")
        assert decision.plan is not None
        authentic = decision.plan
        # Raising only the horizon leaves a constructible, hash-consistent
        # FiniteHorizonPlan object, but it is not the deterministic planner's
        # output for that graph/world revision.
        forged_finite = replace(
            authentic.finite_plan, horizon=authentic.finite_plan.horizon + 1
        )
        forged = ComputeRoutePlan(
            graph_generation=authentic.graph_generation,
            graph_state_sha256=authentic.graph_state_sha256,
            finite_plan=forged_finite,
            primitive_edge_sha256s=authentic.primitive_edge_sha256s,
        )
        generation = self.graph.state().generation

        with self.assertRaisesRegex(
            ComputeOperatorGraphIntegrityError, "not reproducible"
        ):
            self.graph.charge_route(forged)
        self.assertEqual(self.graph.state().generation, generation)
        self.assertEqual(self.graph.state().materialized_routes, ())

    def test_markov_route_binds_and_verifies_contraction_ledger(self) -> None:
        first_kernel = np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float64)
        second_kernel = np.array([[0.6, 0.4], [0.1, 0.9]], dtype=np.float64)
        crystals = (
            ComputeCrystal.markov(first_kernel),
            ComputeCrystal.markov(second_kernel),
        )
        self._publish_edges(crystals, ("prior", "filtered", "posterior"))
        decision = self.graph.plan_route("prior", "posterior")
        assert decision.plan is not None
        charged = self.graph.charge_route(decision.plan)
        ledger = charged.route.contraction_ledger

        self.assertIsNotNone(ledger)
        assert ledger is not None
        self.assertEqual(ledger.depth, 2)
        fused = self.bank.restore_crystal(charged.route.fused_crystal_sha256 or "")
        composed = fused.apply(np.eye(2, dtype=np.float64))
        self.assertTrue(ledger.verifies(composed))
        self.assertLessEqual(ledger.tau_upper_bound, 1.0)
        result = self.graph.discharge(
            "prior", "posterior", np.array([0.25, 0.75], dtype=np.float64)
        )
        np.testing.assert_allclose(
            result.output,
            np.array([0.25, 0.75], dtype=np.float64) @ first_kernel @ second_kernel,
            rtol=0.0,
            atol=1e-15,
        )
        self.assertEqual(result.receipt.contraction_ledger_sha256, ledger.sha256)

    def test_persistence_reopen_and_append_idempotence(self) -> None:
        crystals = (
            ComputeCrystal.affine([[2.0]], [1.0]),
            ComputeCrystal.affine([[3.0]], [-4.0]),
        )
        edges = self._publish_edges(crystals, ("a", "b", "c"))
        decision = self.graph.plan_route("a", "c")
        assert decision.plan is not None
        charged = self.graph.charge_route(decision.plan)
        before = self.graph.state()

        reopened = ComputeOperatorGraph(ComputeCrystalBank(self.temporary.name))
        self.assertEqual(reopened.state(), before)
        state, changed = reopened.append_edge(edges[0])
        self.assertFalse(changed)
        self.assertEqual(state, before)
        warm = reopened.charge_route(decision.plan)
        self.assertFalse(warm.graph_changed)
        self.assertEqual(warm.route.sha256, charged.route.sha256)
        value = np.array([7.0], dtype=np.float64)
        np.testing.assert_array_equal(
            reopened.discharge("a", "c", value).output,
            crystals[1].apply(crystals[0].apply(value)),
        )

    def test_crash_after_artifact_publication_recovers_idempotently(self) -> None:
        crystals = (
            ComputeCrystal.affine([[2.0]], [1.0]),
            ComputeCrystal.affine([[4.0]], [3.0]),
        )
        self._publish_edges(crystals, ("a", "b", "c"))
        decision = self.graph.plan_route("a", "c")
        assert decision.plan is not None
        generation = self.graph.state().generation
        original = self.graph.append_materialized_route

        with mock.patch.object(
            self.graph,
            "append_materialized_route",
            side_effect=RuntimeError("simulated crash before graph append"),
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                self.graph.charge_route(decision.plan)
        self.assertEqual(self.graph.state().generation, generation)
        self.assertEqual(self.graph.state().materialized_routes, ())

        with mock.patch.object(self.graph, "append_materialized_route", wraps=original):
            recovered = self.graph.charge_route(decision.plan)
        self.assertTrue(recovered.graph_changed)
        self.assertFalse(recovered.primitive_program_created)
        self.assertFalse(recovered.fused_crystal_created)
        self.assertFalse(recovered.executable_program_created)
        self.assertFalse(recovered.charge_basis_created)
        self.assertEqual(len(self.graph.state().materialized_routes), 1)

    def test_stale_cas_retry_concurrent_append_and_tamper_detection(self) -> None:
        crystals = tuple(
            ComputeCrystal.affine([[float(index + 1)]], [0.0]) for index in range(5)
        )
        for crystal in crystals:
            self.bank.publish_crystal(crystal)
        edges = tuple(
            _edge(f"s{index}", f"s{index + 1}", crystal, 20 + index)
            for index, crystal in enumerate(crystals)
        )
        initial = self.graph.state()
        self.graph.append_edge(edges[0], expected_generation=initial.generation)
        with self.assertRaises(ComputeOperatorGraphConflictError):
            self.graph.append_edge(edges[1], expected_generation=initial.generation)

        real_publish = self.bank.store.publish_state
        conflicts = {"remaining": 1}

        def conflict_once(name: str, payload: bytes, **kwargs: object):
            if name == OPERATOR_GRAPH_STATE_NAME and conflicts["remaining"]:
                conflicts["remaining"] -= 1
                raise ManifestConflictError("injected graph CAS race")
            return real_publish(name, payload, **kwargs)

        with mock.patch.object(
            self.bank.store, "publish_state", side_effect=conflict_once
        ):
            state, changed = self.graph.append_edge(edges[1])
        self.assertTrue(changed)
        self.assertEqual(conflicts["remaining"], 0)
        self.assertIn(edges[1], state.edges)

        errors: list[BaseException] = []

        def append(edge: OperatorEdge) -> None:
            try:
                ComputeOperatorGraph(
                    ComputeCrystalBank(self.temporary.name)
                ).append_edge(edge)
            except BaseException as exc:  # pragma: no cover - diagnostic collection
                errors.append(exc)

        threads = [threading.Thread(target=append, args=(edge,)) for edge in edges[2:]]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)
        self.assertFalse(errors)
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        concurrent = self.graph.state()
        self.assertEqual(
            {edge.sha256 for edge in concurrent.edges}, {edge.sha256 for edge in edges}
        )

        # A fully resealed pointer still fails because its content-addressed
        # predecessor/history chain cannot be forged by changing the pointer.
        document = json.loads(concurrent.to_bytes())
        document["body"]["edges"][0]["target_state"] = "forged-target"
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        forged = canonical_json_bytes(document)
        self.bank.store.publish_state(
            OPERATOR_GRAPH_STATE_NAME,
            forged,
            expected_sha256=concurrent.sha256,
        )
        with self.assertRaises(ComputeOperatorGraphIntegrityError):
            self.graph.state()

    def test_resealed_pointer_rollback_is_rejected_by_committed_history(self) -> None:
        crystals = (
            ComputeCrystal.affine([[2.0]], [1.0]),
            ComputeCrystal.affine([[3.0]], [2.0]),
        )
        for crystal in crystals:
            self.bank.publish_crystal(crystal)
        first, _changed = self.graph.append_edge(_edge("a", "b", crystals[0], 80))
        second, _changed = self.graph.append_edge(_edge("b", "c", crystals[1], 81))
        self.assertEqual(second.generation, 2)

        self.bank.store.publish_state(
            OPERATOR_GRAPH_STATE_NAME,
            first.to_bytes(),
            expected_sha256=second.sha256,
        )

        with self.assertRaisesRegex(
            ComputeOperatorGraphIntegrityError, "resealed rollback"
        ):
            ComputeOperatorGraph(self.bank).state()

    def test_history_before_pointer_crash_is_ignored_and_retry_recovers(self) -> None:
        crystals = (
            ComputeCrystal.affine([[2.0]], [1.0]),
            ComputeCrystal.affine([[3.0]], [2.0]),
        )
        for crystal in crystals:
            self.bank.publish_crystal(crystal)
        edge_one = _edge("a", "b", crystals[0], 90)
        edge_two = _edge("b", "c", crystals[1], 91)
        stable, _changed = self.graph.append_edge(edge_one)
        original_publish = self.bank.store.publish_state

        def crash_at_pointer(name: str, payload: bytes, **kwargs: object):
            if name == OPERATOR_GRAPH_STATE_NAME:
                raise OSError("simulated history-before-pointer crash")
            return original_publish(name, payload, **kwargs)

        with mock.patch.object(
            self.bank.store, "publish_state", side_effect=crash_at_pointer
        ):
            with self.assertRaisesRegex(OSError, "history-before-pointer"):
                self.graph.append_edge(edge_two)

        self.assertEqual(ComputeOperatorGraph(self.bank).state(), stable)
        recovered, changed = ComputeOperatorGraph(self.bank).append_edge(edge_two)
        self.assertTrue(changed)
        self.assertEqual(recovered.generation, 2)

    def test_pointer_before_commit_crash_recovers_commit_marker(self) -> None:
        crystal = ComputeCrystal.affine([[2.0]], [1.0])
        self.bank.publish_crystal(crystal)
        edge = _edge("a", "b", crystal, 100)
        original_publish = self.bank.store.publish_state

        def crash_at_commit(name: str, payload: bytes, **kwargs: object):
            if name.startswith(OPERATOR_GRAPH_COMMIT_PREFIX):
                raise OSError("simulated pointer-before-commit crash")
            return original_publish(name, payload, **kwargs)

        with mock.patch.object(
            self.bank.store, "publish_state", side_effect=crash_at_commit
        ):
            with self.assertRaisesRegex(OSError, "pointer-before-commit"):
                self.graph.append_edge(edge)

        recovered = ComputeOperatorGraph(self.bank).state()
        self.assertEqual(recovered.generation, 1)
        self.assertEqual(recovered.edges, (edge,))
        self.assertEqual(
            self.bank.store.restore_state(
                ComputeOperatorGraph.commit_state_name(recovered.sha256)
            ),
            canonical_json_bytes(
                {
                    "schema": "immer-ooe-compute-operator-graph-commit/v1",
                    "body": {"graph_state_sha256": recovered.sha256},
                    "body_sha256": hashlib.sha256(
                        canonical_json_bytes({"graph_state_sha256": recovered.sha256})
                    ).hexdigest(),
                }
            ),
        )

    def test_external_graph_anchor_rejects_complete_valid_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            victim_root = root / "victim"
            replacement_root = root / "replacement"

            victim_bank = ComputeCrystalBank(victim_root)
            victim_crystal = ComputeCrystal.affine([[2.0]], [1.0])
            victim_bank.publish_crystal(victim_crystal)
            victim_graph = ComputeOperatorGraph(victim_bank)
            victim_graph.append_edge(_edge("victim-a", "victim-b", victim_crystal, 110))
            trusted_anchor = victim_graph.current_anchor_sha256()

            replacement_bank = ComputeCrystalBank(replacement_root)
            replacement_crystal = ComputeCrystal.affine([[7.0]], [-3.0])
            replacement_bank.publish_crystal(replacement_crystal)
            replacement_graph = ComputeOperatorGraph(replacement_bank)
            replacement_graph.append_edge(
                _edge("replacement-a", "replacement-b", replacement_crystal, 111)
            )
            replacement_head = replacement_graph.state()

            shutil.rmtree(victim_root / "state")
            shutil.copytree(replacement_root / "state", victim_root / "state")

            unanchored = ComputeOperatorGraph(ComputeCrystalBank(victim_root))
            self.assertEqual(unanchored.state(), replacement_head)
            anchored = ComputeOperatorGraph(
                ComputeCrystalBank(victim_root),
                trusted_graph_state_sha256=trusted_anchor,
            )
            with self.assertRaisesRegex(
                ComputeOperatorGraphIntegrityError, "trusted anchor"
            ):
                anchored.state()
            resolver_anchored = ComputeOperatorGraph(
                ComputeCrystalBank(victim_root),
                trusted_head_resolver=lambda: trusted_anchor,
            )
            with self.assertRaisesRegex(
                ComputeOperatorGraphIntegrityError, "trusted anchor"
            ):
                resolver_anchored.state()

    def test_exact_anchor_rejects_grafted_attacker_continuation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            victim_root = root / "victim"
            graft_root = root / "graft"

            victim_bank = ComputeCrystalBank(victim_root)
            first = ComputeCrystal.affine([[2.0]], [1.0])
            victim_bank.publish_crystal(first)
            victim_graph = ComputeOperatorGraph(victim_bank)
            victim_graph.append_edge(_edge("a", "b", first, 130))
            trusted_exact_head = victim_graph.current_anchor_sha256()

            shutil.copytree(victim_root, graft_root)
            graft_bank = ComputeCrystalBank(graft_root)
            attacker = ComputeCrystal.affine([[11.0]], [-7.0])
            graft_bank.publish_crystal(attacker)
            graft_graph = ComputeOperatorGraph(graft_bank)
            grafted_head, _changed = graft_graph.append_edge(
                _edge("b", "attacker-goal", attacker, 131)
            )
            self.assertEqual(grafted_head.previous_state_sha256, trusted_exact_head)

            shutil.rmtree(victim_root / "state")
            shutil.copytree(graft_root / "state", victim_root / "state")
            self.assertEqual(
                ComputeOperatorGraph(ComputeCrystalBank(victim_root)).state(),
                grafted_head,
            )
            exact = ComputeOperatorGraph(
                ComputeCrystalBank(victim_root),
                trusted_graph_state_sha256=trusted_exact_head,
            )
            with self.assertRaisesRegex(
                ComputeOperatorGraphIntegrityError, "does not equal the trusted anchor"
            ):
                exact.state()

    def test_explicit_session_anchor_advances_but_resolver_stays_authoritative(
        self,
    ) -> None:
        crystals = (
            ComputeCrystal.affine([[2.0]], [1.0]),
            ComputeCrystal.affine([[3.0]], [2.0]),
            ComputeCrystal.affine([[5.0]], [-1.0]),
        )
        for crystal in crystals:
            self.bank.publish_crystal(crystal)
        first, _changed = self.graph.append_edge(_edge("a", "b", crystals[0], 140))

        explicit = ComputeOperatorGraph(
            self.bank,
            trusted_graph_state_sha256=first.sha256,
        )
        second, changed = explicit.append_edge(_edge("b", "c", crystals[1], 141))
        self.assertTrue(changed)
        self.assertEqual(explicit.trusted_graph_state_sha256, second.sha256)
        self.assertEqual(explicit.current_anchor_sha256(), second.sha256)

        authority = {"head": second.sha256}
        resolved = ComputeOperatorGraph(
            self.bank,
            trusted_head_resolver=lambda: authority["head"],
        )
        third, changed = resolved.append_edge(_edge("c", "d", crystals[2], 142))
        self.assertTrue(changed)
        self.assertEqual(resolved.current_anchor_sha256(), third.sha256)
        with self.assertRaisesRegex(
            ComputeOperatorGraphIntegrityError, "does not equal the trusted anchor"
        ):
            resolved.state()
        authority["head"] = third.sha256
        self.assertEqual(resolved.state(), third)

    def test_existing_route_fast_path_rejects_forged_graph_generation(self) -> None:
        crystals = (
            ComputeCrystal.affine([[2.0]], [1.0]),
            ComputeCrystal.affine([[3.0]], [2.0]),
        )
        self._publish_edges(crystals, ("a", "b", "c"), offset=120)
        decision = self.graph.plan_route("a", "c")
        assert decision.plan is not None
        self.graph.charge_route(decision.plan)
        forged = replace(decision.plan, graph_generation=999)

        with self.assertRaisesRegex(
            ComputeOperatorGraphIntegrityError, "planning generation"
        ):
            self.graph.charge_route(forged)


if __name__ == "__main__":
    unittest.main()
