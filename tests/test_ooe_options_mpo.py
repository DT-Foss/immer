from __future__ import annotations

from dataclasses import replace
import base64
import hashlib
import json
import unittest
from unittest.mock import patch
import warnings

import numpy as np

import immer.runtimes.ooe.mpo as mpo_module
import immer.runtimes.ooe.options as options_module
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.math_core import array_sha256
from immer.runtimes.ooe.mpo import (
    ActionConditionedMPO,
    MAX_MPO_ACTIONS,
    MPOIntegrityError,
    factorize_action_transitions,
)
from immer.runtimes.ooe.options import (
    BalancedOptionAllocation,
    MacroOption,
    OptionIntegrityError,
    OptionKernelCatalog,
    VerifiedTrajectory,
    allocate_options_balanced,
    compose_stochastic_kernels,
    discover_macro_options,
)


MODEL_HASH = "1" * 64
GRAPH_HASH = "2" * 64
VERIFIERS = {"exact": "3" * 64, "fertig": "4" * 64}


def _digest(number: int) -> str:
    return f"{number:064x}"


def _deterministic_kernel(targets: list[int]) -> np.ndarray:
    result = np.zeros((len(targets), len(targets)), dtype=np.float64)
    result[np.arange(len(targets)), targets] = 1.0
    return result


def _world() -> dict[str, np.ndarray]:
    # Three isomorphic corridors.  The first and third are observed; the
    # middle corridor is deliberately reserved for unseen option planning.
    return {
        "advance": _deterministic_kernel([1, 1, 2, 4, 4, 5, 7, 7, 8]),
        "finish": _deterministic_kernel([0, 2, 2, 3, 5, 5, 6, 8, 8]),
    }


def _trajectory(start: int, receipt: int) -> VerifiedTrajectory:
    return VerifiedTrajectory.create(
        states=(start, start + 1, start + 2),
        actions=("advance", "finish"),
        source_world_model_sha256=MODEL_HASH,
        graph_revision_sha256=GRAPH_HASH,
        verifier_hashes=VERIFIERS,
        verification_receipt_sha256=_digest(receipt),
        outcome_sha256=_digest(receipt + 100),
    )


def _discover() -> tuple[dict[str, np.ndarray], MacroOption]:
    world = _world()
    result = discover_macro_options(
        (_trajectory(0, 10), _trajectory(6, 11)),
        world,
        min_support=2,
        min_option_length=2,
        max_option_length=2,
    )
    if len(result.options) != 1:
        raise AssertionError("fixture must discover exactly one option")
    return world, result.options[0]


def _kron_all(matrices: list[np.ndarray]) -> np.ndarray:
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def _structured_tensor() -> np.ndarray:
    identity = np.eye(2, dtype=np.float64)
    flip = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    return np.stack(
        (
            _kron_all([identity, identity, identity, identity]),
            _kron_all([flip, flip, flip, flip]),
        )
    )


def _factorize(
    tensor: np.ndarray,
    *,
    max_rank: int,
    tolerance: float = 1e-12,
) -> ActionConditionedMPO:
    return factorize_action_transitions(
        tensor,
        action_ids=("stay", "flip"),
        state_shape=(2, 2, 2, 2),
        source_world_model_sha256=MODEL_HASH,
        graph_revision_sha256=GRAPH_HASH,
        verifier_hashes=VERIFIERS,
        max_rank=max_rank,
        relative_tolerance=tolerance,
    )


class MacroOptionTests(unittest.TestCase):
    def test_composition_is_exact_ordered_matrix_product(self) -> None:
        left = np.asarray(
            [[0.7, 0.3], [0.2, 0.8]],
            dtype=np.float64,
        )
        middle = np.asarray(
            [[0.9, 0.1], [0.4, 0.6]],
            dtype=np.float64,
        )
        right = np.asarray(
            [[0.55, 0.45], [0.05, 0.95]],
            dtype=np.float64,
        )
        expected = (left @ middle) @ right
        actual = compose_stochastic_kernels((left, middle, right))
        self.assertTrue(np.array_equal(actual, expected))
        self.assertFalse(
            np.allclose(actual, compose_stochastic_kernels((right, middle, left)))
        )

    def test_deterministic_16x16_composition_is_runtimewarning_clean(self) -> None:
        left = _deterministic_kernel(list(range(1, 16)) + [0])
        right = _deterministic_kernel(list(range(2, 16)) + [0, 1])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            actual = compose_stochastic_kernels((left, right))
        expected_targets = [
            int(np.argmax(right[int(np.argmax(left[state]))]))
            for state in range(16)
        ]
        self.assertTrue(np.array_equal(actual, _deterministic_kernel(expected_targets)))

    def test_discovery_uses_distinct_verified_successes_and_binds_contract(self) -> None:
        world = _world()
        result = discover_macro_options(
            (_trajectory(0, 10), _trajectory(6, 11)),
            world,
            min_support=2,
            max_option_length=2,
        )
        self.assertEqual(len(result.options), 1)
        option = result.options[0]
        self.assertEqual(option.identity.action_sequence, ("advance", "finish"))
        self.assertEqual(option.support, 2)
        self.assertEqual(option.identity.source_world_model_sha256, MODEL_HASH)
        self.assertEqual(option.identity.graph_revision_sha256, GRAPH_HASH)
        self.assertEqual(dict(option.identity.verifier_hashes), VERIFIERS)
        self.assertEqual(
            result.receipt.discovered_option_sha256s,
            (option.sha256,),
        )
        repeated = discover_macro_options(
            (_trajectory(0, 10), _trajectory(6, 11)),
            world,
            min_support=2,
            max_option_length=2,
        )
        self.assertEqual(result.receipt.sha256, repeated.receipt.sha256)
        self.assertEqual(option.to_bytes(), repeated.options[0].to_bytes())

    def test_only_successful_world_consistent_trajectories_qualify(self) -> None:
        trajectory = _trajectory(0, 10)
        with self.assertRaisesRegex(ValueError, "successful"):
            replace(trajectory, verified_success=False)
        duplicate = _trajectory(0, 10)
        with self.assertRaisesRegex(ValueError, "unique"):
            discover_macro_options(
                (trajectory, duplicate),
                _world(),
                min_support=2,
            )
        inconsistent = replace(trajectory, states=(0, 2, 2))
        with self.assertRaisesRegex(ValueError, "contradicts"):
            discover_macro_options((inconsistent,), _world(), min_support=1)

    def test_mixed_graph_or_verifier_contract_cannot_form_an_option(self) -> None:
        changed = replace(_trajectory(6, 11), graph_revision_sha256="a" * 64)
        with self.assertRaisesRegex(ValueError, "contracts"):
            discover_macro_options(
                (_trajectory(0, 10), changed),
                _world(),
                min_support=2,
            )

    def test_option_roundtrip_and_primitive_rebinding(self) -> None:
        world, option = _discover()
        restored = MacroOption.from_bytes(option.to_bytes())
        self.assertEqual(restored.sha256, option.sha256)
        self.assertTrue(np.array_equal(restored.restore_kernel(), option.restore_kernel()))
        changed = dict(world)
        changed["finish"] = np.eye(9, dtype=np.float64)
        with self.assertRaisesRegex(OptionIntegrityError, "changed"):
            restored.verify_against(changed)

    def test_option_payload_tamper_fails_even_with_rehashed_envelope(self) -> None:
        _, option = _discover()
        document = json.loads(option.to_bytes())
        raw = base64.b64decode(document["body"]["kernel"]["data_base64"])
        damaged = bytes([raw[0] ^ 1]) + raw[1:]
        document["body"]["kernel"]["data_base64"] = base64.b64encode(
            damaged
        ).decode("ascii")
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaises(OptionIntegrityError):
            MacroOption.from_bytes(canonical_json_bytes(document))

    def test_option_shortens_unseen_planning_without_hiding_primitive_cost(self) -> None:
        world, option = _discover()
        catalog = OptionKernelCatalog(world, (option,))
        primitive = catalog.plan(3, 5, include_options=False)
        hierarchical = catalog.plan(3, 5, include_options=True)
        assert primitive is not None
        assert hierarchical is not None
        self.assertEqual(primitive.operator_depth, 2)
        self.assertEqual(hierarchical.operator_depth, 1)
        self.assertEqual(hierarchical.primitive_length, 2)
        self.assertEqual(hierarchical.primitive_actions, ("advance", "finish"))
        self.assertEqual(hierarchical.success_probability, 1.0)
        self.assertLess(hierarchical.expanded_states, primitive.expanded_states)
        contraction = catalog.contraction_ledger(hierarchical)
        self.assertEqual(contraction.depth, hierarchical.primitive_length)
        self.assertEqual(len(contraction.sha256), 64)
        self.assertTrue(
            contraction.verifies(
                compose_stochastic_kernels(
                    tuple(world[action] for action in hierarchical.primitive_actions)
                )
            )
        )

    def test_planner_maximizes_path_probability_instead_of_first_argmax_goal(
        self,
    ) -> None:
        direct = np.eye(4, dtype=np.float64)
        direct[0] = [0.0, 0.0, 0.6, 0.4]
        setup = np.eye(4, dtype=np.float64)
        setup[0] = [0.0, 0.9, 0.0, 0.1]
        finish = np.eye(4, dtype=np.float64)
        finish[1] = [0.0, 0.0, 0.9, 0.1]
        catalog = OptionKernelCatalog(
            {"direct": direct, "finish": finish, "setup": setup}
        )
        plan = catalog.plan(0, 2, max_operator_depth=2)
        assert plan is not None
        self.assertEqual(plan.operator_ids, ("action:setup", "action:finish"))
        self.assertEqual(plan.primitive_actions, ("setup", "finish"))
        self.assertAlmostEqual(plan.success_probability, 0.81)
        self.assertTrue(plan.optimal_within_depth_limit)
        self.assertEqual(plan.sha256, catalog.plan(0, 2, max_operator_depth=2).sha256)


class OptionAllocationTests(unittest.TestCase):
    def test_seeded_gumbel_allocation_is_deterministic_and_birkhoff_balanced(
        self,
    ) -> None:
        option_ids = (_digest(501), _digest(502), _digest(503))
        scores = np.asarray(
            [[5.0, 1.0, 0.0], [0.0, 5.0, 1.0], [1.0, 0.0, 5.0]],
            dtype=np.float64,
        )
        first = allocate_options_balanced(
            scores,
            agent_ids=("alice", "bob", "carol"),
            option_sha256s=option_ids,
            gumbel_seed="allocation-round-7",
            gumbel_scale=0.15,
        )
        second = allocate_options_balanced(
            scores,
            agent_ids=("alice", "bob", "carol"),
            option_sha256s=option_ids,
            gumbel_seed="allocation-round-7",
            gumbel_scale=0.15,
        )
        self.assertTrue(np.array_equal(first.allocation, second.allocation))
        self.assertEqual(first.receipt.sha256, second.receipt.sha256)
        self.assertTrue(np.allclose(first.padded_allocation.sum(0), 1.0, atol=1e-9))
        self.assertTrue(np.allclose(first.padded_allocation.sum(1), 1.0, atol=1e-9))
        self.assertEqual(first.assigned_option("alice"), option_ids[0])
        self.assertEqual(first.assigned_option("bob"), option_ids[1])
        self.assertEqual(first.assigned_option("carol"), option_ids[2])
        self.assertFalse(first.allocation.flags.writeable)

    def test_rectangular_allocation_uses_explicit_birkhoff_padding(self) -> None:
        result = allocate_options_balanced(
            [[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]],
            agent_ids=("left", "right"),
            option_sha256s=(_digest(601), _digest(602), _digest(603)),
        )
        self.assertEqual(result.allocation.shape, (2, 3))
        self.assertEqual(result.padded_allocation.shape, (3, 3))
        self.assertEqual(result.receipt.padded_size, 3)
        self.assertTrue(np.allclose(result.padded_allocation.sum(0), 1.0, atol=1e-9))
        self.assertTrue(np.allclose(result.padded_allocation.sum(1), 1.0, atol=1e-9))

    def test_allocation_tamper_is_rejected_by_receipt(self) -> None:
        result = allocate_options_balanced(
            [[3.0, 1.0], [1.0, 3.0]],
            agent_ids=("left", "right"),
            option_sha256s=(_digest(701), _digest(702)),
        )
        damaged = result.allocation.copy()
        damaged[0, 0] += 0.01
        with self.assertRaisesRegex(OptionIntegrityError, "hash"):
            BalancedOptionAllocation(
                allocation=damaged,
                padded_allocation=result.padded_allocation,
                receipt=result.receipt,
            )

    def test_gumbel_noise_requires_a_bound_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "seed"):
            allocate_options_balanced(
                [[1.0]],
                agent_ids=("agent",),
                option_sha256s=(_digest(801),),
                gumbel_scale=0.1,
            )

    def test_discrete_projection_prevents_duplicate_rowwise_assignments(self) -> None:
        option_ids = (_digest(901), _digest(902), _digest(903))
        result = allocate_options_balanced(
            np.zeros((3, 3), dtype=np.float64),
            agent_ids=("a", "b", "c"),
            option_sha256s=option_ids,
        )
        # Every rowwise argmax is column zero on the uniform Birkhoff matrix;
        # the discrete projection must still produce a one-to-one optimum.
        self.assertEqual(
            tuple(int(np.argmax(row)) for row in result.allocation),
            (0, 0, 0),
        )
        assigned = tuple(result.assigned_option(agent) for agent in ("a", "b", "c"))
        self.assertEqual(len(set(assigned)), 3)
        self.assertEqual(set(assigned), set(option_ids))
        repeated = allocate_options_balanced(
            np.zeros((3, 3), dtype=np.float64),
            agent_ids=("a", "b", "c"),
            option_sha256s=option_ids,
        )
        self.assertEqual(
            result.receipt.assigned_option_sha256s,
            repeated.receipt.assigned_option_sha256s,
        )
        self.assertEqual(result.receipt.sha256, repeated.receipt.sha256)

    def test_rectangular_discrete_assignment_marks_only_surplus_agents_unassigned(
        self,
    ) -> None:
        result = allocate_options_balanced(
            np.zeros((3, 2), dtype=np.float64),
            agent_ids=("a", "b", "c"),
            option_sha256s=(_digest(911), _digest(912)),
        )
        assigned = tuple(result.assigned_option(agent) for agent in ("a", "b", "c"))
        self.assertEqual(sum(value is None for value in assigned), 1)
        real = [value for value in assigned if value is not None]
        self.assertEqual(len(real), len(set(real)))

    def test_discrete_assignment_is_receipt_bound_not_rederived_by_row_argmax(
        self,
    ) -> None:
        options = (_digest(921), _digest(922), _digest(923))
        result = allocate_options_balanced(
            np.zeros((3, 3), dtype=np.float64),
            agent_ids=("a", "b", "c"),
            option_sha256s=options,
        )
        canonical = result.receipt.assigned_option_sha256s
        forged = (canonical[1], canonical[0], canonical[2])
        forged_columns = tuple(options.index(value) for value in forged)
        forged_receipt = replace(
            result.receipt,
            assigned_option_sha256s=forged,
            assignment_sha256=options_module._assignment_sha256(
                result.receipt.agent_ids,
                options,
                forged_columns,
            ),
        )
        with self.assertRaisesRegex(OptionIntegrityError, "maximum-weight"):
            BalancedOptionAllocation(
                allocation=result.allocation,
                padded_allocation=result.padded_allocation,
                receipt=forged_receipt,
            )


class ActionConditionedMPOTests(unittest.TestCase):
    def test_structured_world_model_gets_real_compact_mpo(self) -> None:
        tensor = _structured_tensor()
        result = _factorize(tensor, max_rank=4)
        self.assertTrue(result.compressed)
        self.assertEqual(result.receipt.mode, "mpo")
        self.assertEqual(
            result.receipt.selection_reason,
            "tolerance_met_and_compact",
        )
        self.assertLess(
            result.receipt.stored_numeric_bytes,
            result.receipt.dense_numeric_bytes,
        )
        self.assertLess(float.fromhex(result.receipt.compression_ratio_hex), 0.2)
        self.assertLessEqual(max(result.receipt.attempted_bond_ranks), 4)
        self.assertLessEqual(
            float.fromhex(result.receipt.reconstruction_relative_error_hex),
            1e-12,
        )
        self.assertTrue(np.allclose(result.reconstruct(), tensor, atol=1e-12))
        cores = result.restore_cores()
        self.assertEqual(cores[0].shape[:2], (1, 2))
        self.assertEqual(len(cores), 5)
        self.assertTrue(all(core.ndim == 4 for core in cores[1:]))

    def test_mpo_exposes_clean_action_kernel_api_for_option_planning(self) -> None:
        result = _factorize(_structured_tensor(), max_rank=4)
        kernels = result.as_action_kernels()
        self.assertEqual(set(kernels), {"stay", "flip"})
        catalog = OptionKernelCatalog(kernels)
        plan = catalog.plan(0, 15, max_operator_depth=1)
        assert plan is not None
        self.assertEqual(plan.primitive_actions, ("flip",))
        self.assertAlmostEqual(plan.success_probability, 1.0, places=14)

    def test_selected_mpo_action_executes_without_full_reconstruction(self) -> None:
        tensor = _structured_tensor()
        result = _factorize(tensor, max_rank=4)
        expected_selected = result.reconstruct()[1].copy()
        baseline_catalog = OptionKernelCatalog(
            {"stay": tensor[0], "flip": tensor[1]}
        )
        baseline_plan = baseline_catalog.plan(0, 15, max_operator_depth=1)
        with patch.object(
            ActionConditionedMPO,
            "reconstruct",
            side_effect=AssertionError("full reconstruction forbidden"),
        ):
            selected = result.transition_kernel("flip")
            direct_kernels = result.as_action_kernels()
            direct_plan = OptionKernelCatalog(direct_kernels).plan(
                0,
                15,
                max_operator_depth=1,
            )
        self.assertTrue(np.array_equal(selected, expected_selected))
        self.assertTrue(np.allclose(selected, tensor[1], atol=1e-12))
        assert baseline_plan is not None
        assert direct_plan is not None
        self.assertEqual(direct_plan.primitive_actions, baseline_plan.primitive_actions)
        self.assertEqual(direct_plan.operator_depth, baseline_plan.operator_depth)
        self.assertAlmostEqual(
            direct_plan.success_probability,
            baseline_plan.success_probability,
            places=14,
        )

    def test_selected_dense_fallback_action_slices_without_reconstruction(self) -> None:
        identity = np.eye(2, dtype=np.float64)
        flip = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        tensor = np.stack((identity, flip))
        result = factorize_action_transitions(
            tensor,
            action_ids=("stay", "flip"),
            state_shape=(2,),
            source_world_model_sha256=MODEL_HASH,
            graph_revision_sha256=GRAPH_HASH,
            verifier_hashes=VERIFIERS,
            max_rank=2,
            relative_tolerance=1e-12,
        )
        self.assertTrue(result.receipt.exact_fallback)
        baseline_plan = OptionKernelCatalog(
            {"stay": identity, "flip": flip}
        ).plan(0, 1, max_operator_depth=1)
        with patch.object(
            ActionConditionedMPO,
            "reconstruct",
            side_effect=AssertionError("full reconstruction forbidden"),
        ):
            selected = result.transition_kernel(1)
            direct_plan = OptionKernelCatalog(result.as_action_kernels()).plan(
                0,
                1,
                max_operator_depth=1,
            )
        self.assertTrue(np.array_equal(selected, flip))
        assert baseline_plan is not None
        assert direct_plan is not None
        self.assertEqual(direct_plan.primitive_actions, baseline_plan.primitive_actions)
        self.assertEqual(direct_plan.operator_depth, baseline_plan.operator_depth)
        self.assertEqual(
            direct_plan.success_probability,
            baseline_plan.success_probability,
        )

    def test_mpo_payload_roundtrip_is_canonical(self) -> None:
        tensor = _structured_tensor()
        result = _factorize(tensor, max_rank=4)
        encoded = result.to_bytes()
        restored = ActionConditionedMPO.from_bytes(encoded, source_tensor=tensor)
        self.assertEqual(restored.receipt.sha256, result.receipt.sha256)
        self.assertEqual(restored.to_bytes(), encoded)
        self.assertTrue(np.array_equal(restored.reconstruct(), result.reconstruct()))
        self.assertTrue(
            np.array_equal(restored.transition_kernel("flip"), result.reconstruct()[1])
        )

    def test_compressed_load_requires_external_source_and_parse_only_cannot_execute(
        self,
    ) -> None:
        tensor = _structured_tensor()
        result = _factorize(tensor, max_rank=4)
        encoded = result.to_bytes()
        with self.assertRaisesRegex(MPOIntegrityError, "external source"):
            ActionConditionedMPO.from_bytes(encoded)
        structural = ActionConditionedMPO.parse_structure_only(encoded)
        self.assertFalse(structural.source_verified)
        with self.assertRaisesRegex(MPOIntegrityError, "cannot execute"):
            structural.reconstruct()
        with self.assertRaisesRegex(MPOIntegrityError, "cannot execute"):
            structural.transition_kernel("flip")
        restored = ActionConditionedMPO.from_bytes(
            encoded,
            source_resolver=lambda receipt: (
                tensor if receipt.source_world_model_sha256 == MODEL_HASH else None
            ),
        )
        self.assertTrue(restored.source_verified)
        self.assertTrue(np.array_equal(restored.reconstruct(), result.reconstruct()))

    def test_rehashed_forged_factors_fail_actual_external_error_check(self) -> None:
        tensor = _structured_tensor()
        original = _factorize(tensor, max_rank=4)
        cores = original.restore_cores()
        forged_action = np.zeros_like(cores[0])
        forged_action_bytes = np.asarray(
            forged_action,
            dtype="<f8",
            order="C",
        ).tobytes(order="C")
        site_bytes = original.site_core_bytes
        payload_sha = mpo_module._factor_payload_sha256(
            action_core_shape=original.action_core_shape,
            action_core_bytes=forged_action_bytes,
            site_core_shapes=original.site_core_shapes,
            site_core_bytes=site_bytes,
            dense_fallback_bytes=None,
        )
        physical_cores = [forged_action]
        physical_cores.extend(
            core.reshape(core.shape[0], core.shape[1] * core.shape[2], core.shape[3])
            for core in cores[1:]
        )
        physical = mpo_module._reconstruct_tt(physical_cores)
        forged_tensor = mpo_module._physical_to_action_tensor(
            physical,
            action_count=original.receipt.action_count,
            state_shape=original.receipt.state_shape,
        )
        forged_receipt = replace(
            original.receipt,
            factor_payload_sha256=payload_sha,
            reconstruction_sha256=array_sha256(forged_tensor),
            # The attacker deliberately retains the original tiny claimed
            # reconstruction error and recomputes every content hash.
        )
        forged = ActionConditionedMPO(
            receipt=forged_receipt,
            action_core_shape=original.action_core_shape,
            action_core_bytes=forged_action_bytes,
            site_core_shapes=original.site_core_shapes,
            site_core_bytes=site_bytes,
            dense_fallback_bytes=None,
            _source_verified=False,
        )
        encoded = forged.to_bytes()
        structural = ActionConditionedMPO.parse_structure_only(encoded)
        self.assertFalse(structural.source_verified)
        with self.assertRaisesRegex(MPOIntegrityError, "exceeds receipt tolerance"):
            ActionConditionedMPO.from_bytes(encoded, source_tensor=tensor)

    def test_unstructured_model_rejects_low_rank_overclaim_and_falls_back_exact(
        self,
    ) -> None:
        rng = np.random.default_rng(20260826)
        tensor = rng.random((2, 16, 16), dtype=np.float64)
        tensor /= tensor.sum(axis=2, keepdims=True)
        result = _factorize(tensor, max_rank=1, tolerance=1e-12)
        self.assertFalse(result.compressed)
        self.assertTrue(result.receipt.exact_fallback)
        self.assertEqual(result.receipt.selection_reason, "rank_budget_exceeded")
        self.assertFalse(result.receipt.attempted_mpo_met_tolerance)
        self.assertGreater(
            float.fromhex(result.receipt.attempted_relative_error_hex),
            1e-2,
        )
        self.assertEqual(
            result.receipt.stored_numeric_bytes,
            result.receipt.dense_numeric_bytes,
        )
        self.assertEqual(
            float.fromhex(result.receipt.reconstruction_absolute_error_hex),
            0.0,
        )
        self.assertTrue(np.array_equal(result.reconstruct(), tensor))

    def test_small_noncompact_factorization_falls_back_without_false_compression(
        self,
    ) -> None:
        tensor = np.eye(2, dtype=np.float64)[None, :, :]
        result = factorize_action_transitions(
            tensor,
            action_ids=("stay",),
            state_shape=(2,),
            source_world_model_sha256=MODEL_HASH,
            graph_revision_sha256=GRAPH_HASH,
            verifier_hashes=VERIFIERS,
            max_rank=2,
            relative_tolerance=1e-12,
        )
        self.assertTrue(result.receipt.exact_fallback)
        self.assertEqual(result.receipt.selection_reason, "mpo_not_compact")
        self.assertTrue(result.receipt.attempted_mpo_met_tolerance)
        self.assertTrue(np.array_equal(result.reconstruct(), tensor))

    def test_mpo_core_tamper_fails_even_with_rehashed_envelope(self) -> None:
        result = _factorize(_structured_tensor(), max_rank=4)
        document = json.loads(result.to_bytes())
        encoded = document["body"]["factors"]["action_core"]["data_base64"]
        raw = base64.b64decode(encoded)
        damaged = bytes([raw[0] ^ 1]) + raw[1:]
        document["body"]["factors"]["action_core"]["data_base64"] = (
            base64.b64encode(damaged).decode("ascii")
        )
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaises(MPOIntegrityError):
            ActionConditionedMPO.from_bytes(canonical_json_bytes(document))

    def test_mpo_contract_and_bounds_fail_closed(self) -> None:
        invalid = _structured_tensor()
        invalid[0, 0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "row-stochastic"):
            _factorize(invalid, max_rank=2)
        with self.assertRaisesRegex(ValueError, "action_ids"):
            factorize_action_transitions(
                _structured_tensor(),
                action_ids=("only-one",),
                state_shape=(2, 2, 2, 2),
                source_world_model_sha256=MODEL_HASH,
                graph_revision_sha256=GRAPH_HASH,
                verifier_hashes=VERIFIERS,
                max_rank=2,
            )
        with self.assertRaisesRegex(ValueError, "state_shape"):
            factorize_action_transitions(
                _structured_tensor(),
                action_ids=("stay", "flip"),
                state_shape=(2, 2),
                source_world_model_sha256=MODEL_HASH,
                graph_revision_sha256=GRAPH_HASH,
                verifier_hashes=VERIFIERS,
                max_rank=2,
            )
        too_many_actions = np.repeat(np.eye(2)[None, :, :], MAX_MPO_ACTIONS + 1, axis=0)
        with self.assertRaisesRegex(ValueError, "bounds"):
            factorize_action_transitions(
                too_many_actions,
                action_ids=tuple(f"a{index}" for index in range(MAX_MPO_ACTIONS + 1)),
                state_shape=(2,),
                source_world_model_sha256=MODEL_HASH,
                graph_revision_sha256=GRAPH_HASH,
                verifier_hashes=VERIFIERS,
                max_rank=2,
            )

    def test_mpo_working_set_cap_is_enforced_before_unbounded_svd(self) -> None:
        tensor = _structured_tensor()
        with self.assertRaisesRegex(ValueError, "working set"):
            factorize_action_transitions(
                tensor,
                action_ids=("stay", "flip"),
                state_shape=(2, 2, 2, 2),
                source_world_model_sha256=MODEL_HASH,
                graph_revision_sha256=GRAPH_HASH,
                verifier_hashes=VERIFIERS,
                max_rank=4,
                max_work_bytes=tensor.nbytes,
            )

    def test_source_model_graph_and_verifiers_change_receipt_identity(self) -> None:
        tensor = _structured_tensor()
        first = _factorize(tensor, max_rank=4)
        second = factorize_action_transitions(
            tensor,
            action_ids=("stay", "flip"),
            state_shape=(2, 2, 2, 2),
            source_world_model_sha256="a" * 64,
            graph_revision_sha256=GRAPH_HASH,
            verifier_hashes=VERIFIERS,
            max_rank=4,
            relative_tolerance=1e-12,
        )
        self.assertNotEqual(first.receipt.sha256, second.receipt.sha256)
        self.assertEqual(first.receipt.source_tensor_sha256, second.receipt.source_tensor_sha256)


if __name__ == "__main__":
    unittest.main()
