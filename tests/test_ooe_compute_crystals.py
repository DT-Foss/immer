from __future__ import annotations

import base64
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest import mock
import warnings

import numpy as np

from immer.runtimes.ooe.compute_crystals import (
    AFFINE_FLOAT64,
    COMPUTE_BANK_MANIFEST_STATE,
    ComputeBankManifest,
    ComputeChargeReceipt,
    ComputeCrystal,
    ComputeCrystalABIError,
    ComputeCrystalBank,
    ComputeCrystalConflictError,
    ComputeCrystalFusionError,
    ComputeCrystalIntegrityError,
    ComputeCrystalMissError,
    ComputeCrystalVM,
    ComputeExecutionReceipt,
    ComputeProgram,
    NumericalABI,
    fuse_affine_chain,
    fuse_compatible_chain,
    fuse_markov_chain,
    fuse_permutation_chain,
    tensor_sha256,
)
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.identity import canonical_json_bytes


def _state_path(store: CrystalStore, state_name: str) -> Path:
    filename = hashlib.sha256(state_name.encode("utf-8")).hexdigest() + ".state"
    return Path(store.root) / "state" / filename


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _reseal_state_payload(path: Path, *, name: str, payload: bytes) -> None:
    envelope = json.loads(path.read_bytes())
    envelope["name"] = name
    envelope["payload_base64"] = base64.b64encode(payload).decode("ascii")
    envelope["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    os.chmod(path, 0o644)
    path.write_bytes(canonical_json_bytes(envelope))


def _affine_chain() -> tuple[ComputeCrystal, ComputeCrystal, ComputeCrystal]:
    return (
        ComputeCrystal.affine(
            np.array(
                [
                    [1.0, 2.0, -1.0, 0.5],
                    [-2.0, 0.25, 3.0, 1.0],
                    [0.0, 1.5, 2.0, -0.5],
                    [4.0, -1.0, 0.5, 2.0],
                    [0.25, 0.75, -2.0, 3.0],
                    [1.25, -0.5, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            np.arange(6, dtype=np.float64) / 10.0,
        ),
        ComputeCrystal.affine(
            np.array(
                [
                    [1.0, 0.0, 2.0, -1.0, 0.5, 1.0],
                    [-0.5, 2.0, 0.0, 1.0, -1.0, 3.0],
                    [2.0, 1.0, -0.25, 0.0, 1.5, -1.0],
                    [0.25, -1.0, 1.0, 2.0, 0.0, 0.5],
                    [1.5, 0.5, -2.0, 1.0, 0.25, 0.0],
                ],
                dtype=np.float64,
            ),
            np.array([0.5, -1.0, 0.0, 2.0, -0.25], dtype=np.float64),
        ),
        ComputeCrystal.affine(
            np.array(
                [
                    [1.0, 2.0, 0.0, -1.0, 0.5],
                    [-2.0, 0.0, 1.0, 0.25, 1.5],
                    [0.5, -1.0, 2.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
            np.array([1.0, -2.0, 0.75], dtype=np.float64),
        ),
    )


class ComputeCrystalArtifactTests(unittest.TestCase):
    def test_crystal_roundtrip_is_canonical_task_and_model_agnostic(self) -> None:
        crystal = ComputeCrystal.affine(
            [[1.0, 2.0], [3.0, 4.0]],
            [0.5, -0.5],
            extensions={"calibration_sha256": "a" * 64, "note": "optional"},
        )
        encoded = crystal.to_bytes()
        restored = ComputeCrystal.from_bytes(encoded)

        self.assertEqual(restored, crystal)
        self.assertEqual(restored.to_bytes(), encoded)
        self.assertEqual(restored.sha256, hashlib.sha256(encoded).hexdigest())
        self.assertEqual(restored.operator_kind, AFFINE_FLOAT64)
        document = json.loads(encoded)
        text = encoded.decode("utf-8")
        self.assertEqual(document["schema"], "immer-ooe-compute-crystal/v1")
        self.assertNotIn("model_pin", text)
        self.assertNotIn("question", text)
        self.assertNotIn("prompt", text)

        with self.assertRaises(ComputeCrystalIntegrityError):
            ComputeCrystal.from_bytes(encoded + b"\n")
        with self.assertRaises(ComputeCrystalIntegrityError):
            ComputeCrystal.from_bytes(b'{"schema":"x","schema":"y"}')

    def test_resealed_payload_tamper_and_nonfinite_values_are_rejected(self) -> None:
        crystal = ComputeCrystal.affine([[1.0, 2.0]], [3.0])
        document = json.loads(crystal.to_bytes())
        document["body"]["payload"]["matrix"][
            "data_base64"
        ] = "AAAAAAAA8H8AAAAAAAAAQA=="
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaises(ComputeCrystalIntegrityError):
            ComputeCrystal.from_bytes(canonical_json_bytes(document))

        document = json.loads(crystal.to_bytes())
        document["body"]["discharge_work_units"] += 1
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        with self.assertRaises(ComputeCrystalIntegrityError):
            ComputeCrystal.from_bytes(canonical_json_bytes(document))

    def test_affine_accepts_unseen_values_and_arbitrary_leading_dimensions(
        self,
    ) -> None:
        crystal = ComputeCrystal.affine(
            [[2.0, -1.0, 0.5], [0.0, 3.0, -2.0]], [1.0, -4.0]
        )
        unseen = np.array(
            [
                [[1.25, -3.0, 7.0], [9.0, 0.5, -2.25]],
                [[-5.0, 2.0, 0.0], [4.0, 8.0, 3.0]],
            ],
            dtype=np.float64,
        )
        expected = unseen @ np.array(
            [[2.0, -1.0, 0.5], [0.0, 3.0, -2.0]], dtype=np.float64
        ).T + np.array([1.0, -4.0], dtype=np.float64)
        np.testing.assert_array_equal(crystal.apply(unseen), expected)
        with self.assertRaises(ComputeCrystalABIError):
            crystal.apply(np.array([1, 2, 3], dtype=np.int64))
        with self.assertRaises(ComputeCrystalABIError):
            crystal.apply(np.ones(4, dtype=np.float64))

    def test_runtime_abi_rejects_hostile_array_protocol_without_invoking_it(
        self,
    ) -> None:
        class HostileArray:
            called = False
            interface_called = False

            def __array__(self, *args: object, **kwargs: object) -> np.ndarray:
                self.called = True
                raise AssertionError("hostile array protocol executed")

            @property
            def __array_interface__(self) -> dict[str, object]:
                self.interface_called = True
                raise AssertionError("hostile array interface executed")

        hostile = HostileArray()
        crystal = ComputeCrystal.affine([[1.0]], [0.0])
        with self.assertRaisesRegex(ComputeCrystalABIError, "exact numpy.ndarray"):
            crystal.apply(hostile)
        self.assertFalse(hostile.called)
        self.assertFalse(hostile.interface_called)

    def test_permutation_supports_float64_and_int64_without_code_payloads(self) -> None:
        float_permutation = ComputeCrystal.permutation([2, 0, 3, 1])
        integer_permutation = ComputeCrystal.permutation([1, 2, 0], dtype="int64")
        np.testing.assert_array_equal(
            float_permutation.apply(np.array([10.0, 20.0, 30.0, 40.0])),
            np.array([30.0, 10.0, 40.0, 20.0]),
        )
        np.testing.assert_array_equal(
            integer_permutation.apply(np.array([[4, 5, 6]], dtype=np.int64)),
            np.array([[5, 6, 4]], dtype=np.int64),
        )
        with self.assertRaises(ValueError):
            ComputeCrystal.permutation([0, 0, 1])

    def test_lookup_maps_indices_supplied_only_at_discharge(self) -> None:
        table = np.array([[1.0, 2.0], [10.0, 20.0], [-4.0, 8.0]], dtype=np.float64)
        crystal = ComputeCrystal.lookup(table)
        indices = np.array([[2, 0], [1, 2]], dtype=np.int64)
        np.testing.assert_array_equal(crystal.apply(indices), table[indices])
        self.assertEqual(crystal.input_abi, NumericalABI("int64", ()))
        self.assertEqual(crystal.output_abi, NumericalABI("float64", (2,)))
        with self.assertRaises(ComputeCrystalABIError):
            crystal.apply(np.array([3], dtype=np.int64))
        with self.assertRaises(ComputeCrystalABIError):
            crystal.apply(np.array([-1], dtype=np.int64))

    def test_markov_kernel_is_row_stochastic_and_executable(self) -> None:
        kernel = np.array(
            [[0.75, 0.20, 0.05], [0.10, 0.60, 0.30], [0.0, 0.25, 0.75]],
            dtype=np.float64,
        )
        crystal = ComputeCrystal.markov(kernel)
        distributions = np.array(
            [[1.0, 0.0, 0.0], [0.25, 0.50, 0.25]], dtype=np.float64
        )
        np.testing.assert_allclose(
            crystal.apply(distributions), distributions @ kernel, rtol=0.0, atol=0.0
        )
        with self.assertRaises(ValueError):
            ComputeCrystal.markov([[0.9, 0.2], [0.1, 0.9]])
        with self.assertRaises(ValueError):
            ComputeCrystal.markov([[1.1, -0.1], [0.0, 1.0]])


class ComputeProgramAndFusionTests(unittest.TestCase):
    def test_program_composes_by_abi_and_rejects_mixed_invalid_edges(self) -> None:
        affine = ComputeCrystal.affine(np.eye(3), np.zeros(3))
        permutation = ComputeCrystal.permutation([2, 0, 1])
        compatible = ComputeProgram.compose((affine, permutation))
        self.assertEqual(ComputeProgram.from_bytes(compatible.to_bytes()), compatible)

        wrong_dimension = ComputeCrystal.permutation([1, 0])
        with self.assertRaises(ComputeCrystalABIError):
            ComputeProgram.compose((affine, wrong_dimension))
        lookup = ComputeCrystal.lookup([[1.0, 2.0], [3.0, 4.0]])
        with self.assertRaises(ComputeCrystalABIError):
            ComputeProgram.compose((affine, lookup))
        with self.assertRaises(ComputeCrystalFusionError):
            fuse_compatible_chain((affine, permutation))

    def test_affine_fusion_matches_unfused_on_unseen_inputs_and_saves_live_work(
        self,
    ) -> None:
        chain = _affine_chain()
        fused = fuse_affine_chain(chain)
        self.assertEqual(fused.parent_sha256s, tuple(item.sha256 for item in chain))
        unseen = np.random.default_rng(20260826).normal(size=(19, 4)).astype(np.float64)
        unfused = unseen
        for crystal in chain:
            unfused = crystal.apply(unfused)
        np.testing.assert_allclose(fused.apply(unseen), unfused, rtol=1e-13, atol=1e-13)
        self.assertNotIn("equivalent_source_work_units", fused.as_record())

    def test_permutation_fusion_is_bit_exact(self) -> None:
        chain = (
            ComputeCrystal.permutation([2, 0, 4, 1, 3]),
            ComputeCrystal.permutation([4, 3, 2, 0, 1]),
            ComputeCrystal.permutation([1, 0, 3, 4, 2]),
        )
        fused = fuse_permutation_chain(chain)
        unseen = np.arange(35, dtype=np.float64).reshape(7, 5)
        expected = unseen
        for crystal in chain:
            expected = crystal.apply(expected)
        np.testing.assert_array_equal(fused.apply(unseen), expected)

    def test_markov_fusion_matches_unfused_transition_distribution(self) -> None:
        chain = (
            ComputeCrystal.markov([[0.8, 0.2], [0.1, 0.9]]),
            ComputeCrystal.markov([[0.5, 0.5], [0.25, 0.75]]),
            ComputeCrystal.markov([[0.9, 0.1], [0.4, 0.6]]),
        )
        fused = fuse_markov_chain(chain)
        unseen = np.array([[0.37, 0.63], [0.91, 0.09]], dtype=np.float64)
        expected = unseen
        for crystal in chain:
            expected = crystal.apply(expected)
        np.testing.assert_allclose(
            fused.apply(unseen), expected, rtol=1e-15, atol=1e-15
        )
        self.assertGreater(
            sum(item.discharge_work_units for item in chain),
            fused.discharge_work_units,
        )

    def test_markov_fusion_is_warning_free_for_32_by_32_chain_of_50(self) -> None:
        random = np.random.default_rng(512)
        chain = []
        for _ in range(50):
            kernel = random.random((32, 32))
            kernel /= kernel.sum(axis=1, keepdims=True)
            chain.append(ComputeCrystal.markov(kernel))
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fused = fuse_markov_chain(chain)
            result = fused.apply(np.eye(32, dtype=np.float64))
        self.assertEqual(result.shape, (32, 32))
        np.testing.assert_allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)


class ComputeCrystalBankAndVMTests(unittest.TestCase):
    def _publish_chain(
        self, bank: ComputeCrystalBank, chain: tuple[ComputeCrystal, ...]
    ) -> None:
        for crystal in chain:
            bank.publish_crystal(crystal)

    def test_bank_reopen_idempotence_and_vm_receipt_for_unseen_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bank"
            bank = ComputeCrystalBank(root)
            chain = _affine_chain()
            self._publish_chain(bank, chain)
            program = ComputeProgram.compose(chain)
            program_publication = bank.publish_program(program)
            generation = program_publication.generation

            same = bank.publish_program(program)
            self.assertFalse(same.object_created)
            self.assertFalse(same.manifest_changed)
            self.assertEqual(same.generation, generation)

            reopened = ComputeCrystalBank(root)
            self.assertEqual(reopened.restore_program(program.sha256), program)
            unseen = np.random.default_rng(99).normal(size=(11, 4)).astype(np.float64)
            execution = ComputeCrystalVM(reopened).execute(program.sha256, unseen)
            expected = unseen
            for crystal in chain:
                expected = crystal.apply(expected)
            np.testing.assert_array_equal(execution.output, expected)
            self.assertEqual(execution.receipt.executed_operator_count, 3)
            self.assertEqual(
                execution.receipt.input_sha256,
                tensor_sha256(unseen, program.input_abi),
            )
            self.assertEqual(
                execution.receipt.output_sha256,
                tensor_sha256(expected, program.output_abi),
            )
            self.assertEqual(
                ComputeExecutionReceipt.from_bytes(execution.receipt.to_bytes()),
                execution.receipt,
            )
            empty = ComputeCrystalVM(reopened).execute(
                program.sha256, np.empty((0, 4), dtype=np.float64)
            )
            self.assertEqual(empty.output.shape, (0, 3))
            self.assertEqual(empty.receipt.equivalent_unfused_source_work, 0)
            self.assertEqual(empty.receipt.live_discharge_work, 0)

    def test_vm_rejects_unpublished_in_memory_program_bypass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = ComputeCrystalBank(Path(temporary) / "bank")
            crystal = ComputeCrystal.affine([[2.0]], [1.0])
            bank.publish_crystal(crystal)
            program = ComputeProgram.compose((crystal,))
            vm = ComputeCrystalVM(bank)
            with self.assertRaises(ComputeCrystalMissError):
                vm.execute(program, np.array([4.0], dtype=np.float64))
            bank.publish_program(program)
            executed = vm.execute(program, np.array([4.0], dtype=np.float64))
            np.testing.assert_array_equal(executed.output, np.array([9.0]))

    def test_fused_vm_requires_published_charge_for_positive_work_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = ComputeCrystalBank(Path(temporary) / "bank")
            chain = _affine_chain()
            self._publish_chain(bank, chain)
            fused = fuse_affine_chain(chain)
            bank.publish_crystal(fused)
            unfused_program = ComputeProgram.compose(chain)
            fused_program = ComputeProgram.compose((fused,))
            bank.publish_program(unfused_program)
            bank.publish_program(fused_program)
            charge = ComputeChargeReceipt.create(
                source_program=unfused_program,
                source_crystals=chain,
                fused_crystal=fused,
                charge_verifier_sha256=_sha("affine-charge-verifier"),
                verification_receipt_sha256=_sha("affine-verification-receipt"),
            )
            charge_publication = bank.publish_charge(charge)
            self.assertEqual(charge_publication.artifact_kind, "charge")
            self.assertEqual(bank.restore_charge(charge.sha256), charge)
            unseen = np.random.default_rng(7).normal(size=(23, 4)).astype(np.float64)

            vm = ComputeCrystalVM(bank)
            unfused = vm.execute(unfused_program.sha256, unseen)
            generic = vm.execute(fused_program.sha256, unseen)
            discharged = vm.execute(
                fused_program.sha256,
                unseen,
                charge_basis_sha256=charge.sha256,
            )
            np.testing.assert_allclose(
                discharged.output, unfused.output, rtol=1e-13, atol=1e-13
            )
            self.assertEqual(unfused.receipt.executed_operator_count, 3)
            self.assertEqual(discharged.receipt.executed_operator_count, 1)
            self.assertEqual(unfused.receipt.historical_work_released, 0)
            self.assertEqual(generic.receipt.historical_work_released, 0)
            self.assertIsNone(generic.receipt.charge_basis_sha256)
            self.assertGreater(discharged.receipt.historical_work_released, 0)
            self.assertEqual(discharged.receipt.charge_basis_sha256, charge.sha256)
            self.assertEqual(
                discharged.receipt.equivalent_unfused_source_work,
                23 * sum(item.discharge_work_units for item in chain),
            )
            self.assertEqual(
                discharged.receipt.live_discharge_work,
                23 * fused.discharge_work_units,
            )

    def test_identity_and_permutation_padding_cannot_mint_saved_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = ComputeCrystalBank(Path(temporary) / "bank")
            identity = ComputeCrystal.permutation(np.arange(16, dtype=np.int64))
            swap = ComputeCrystal.permutation(
                np.array([1, 0, *range(2, 16)], dtype=np.int64)
            )
            bank.publish_crystal(identity)
            bank.publish_crystal(swap)
            padded_chain = (identity,) * 64 + (swap,) + (identity,) * 64
            fused = fuse_permutation_chain(padded_chain)
            bank.publish_crystal(fused)
            program = ComputeProgram.compose((fused,))
            bank.publish_program(program)
            execution = ComputeCrystalVM(bank).execute(
                program.sha256, np.arange(16, dtype=np.float64)
            )
            self.assertEqual(execution.receipt.historical_work_released, 0)
            self.assertIsNone(execution.receipt.charge_basis_sha256)
            self.assertEqual(
                execution.receipt.equivalent_unfused_source_work,
                execution.receipt.live_discharge_work,
            )

            affine_identity = ComputeCrystal.affine(
                np.eye(4, dtype=np.float64), np.zeros(4, dtype=np.float64)
            )
            bank.publish_crystal(affine_identity)
            affine_fused = fuse_affine_chain((affine_identity,) * 32)
            bank.publish_crystal(affine_fused)
            affine_program = ComputeProgram.compose((affine_fused,))
            bank.publish_program(affine_program)
            affine_execution = ComputeCrystalVM(bank).execute(
                affine_program.sha256,
                np.ones(4, dtype=np.float64),
            )
            self.assertEqual(affine_execution.receipt.historical_work_released, 0)

    def test_canceling_affine_padding_is_named_only_by_explicit_charge_basis(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = ComputeCrystalBank(Path(temporary) / "bank")
            scale_two = ComputeCrystal.affine([[2.0]], [0.0])
            scale_half = ComputeCrystal.affine([[0.5]], [0.0])
            translate = ComputeCrystal.affine([[1.0]], [3.0])
            for crystal in (scale_two, scale_half, translate):
                bank.publish_crystal(crystal)
            padded_chain = (scale_two, scale_half) * 32 + (translate,)
            fused = fuse_affine_chain(padded_chain)
            bank.publish_crystal(fused)
            source_program = ComputeProgram.compose(padded_chain)
            fused_program = ComputeProgram.compose((fused,))
            bank.publish_program(source_program)
            bank.publish_program(fused_program)
            charge = ComputeChargeReceipt.create(
                source_program=source_program,
                source_crystals=padded_chain,
                fused_crystal=fused,
                charge_verifier_sha256=_sha("padded-affine-verifier"),
                verification_receipt_sha256=_sha("padded-affine-receipt"),
            )
            bank.publish_charge(charge)
            value = np.array([7.0], dtype=np.float64)
            generic = ComputeCrystalVM(bank).execute(fused_program.sha256, value)
            charged = ComputeCrystalVM(bank).execute(
                fused_program.sha256,
                value,
                charge_basis_sha256=charge.sha256,
            )
            np.testing.assert_array_equal(charged.output, np.array([10.0]))
            self.assertEqual(generic.receipt.historical_work_released, 0)
            self.assertEqual(
                charged.receipt.equivalent_unfused_source_work,
                sum(crystal.discharge_work_units for crystal in padded_chain),
            )
            self.assertEqual(charged.receipt.charge_basis_sha256, charge.sha256)
            self.assertGreater(charged.receipt.historical_work_released, 0)
            self.assertEqual(charge.source_program_sha256, source_program.sha256)
            self.assertEqual(
                charge.source_crystal_sha256s,
                tuple(crystal.sha256 for crystal in padded_chain),
            )

    def test_legitimate_markov_charge_releases_exact_source_program_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = ComputeCrystalBank(Path(temporary) / "bank")
            chain = (
                ComputeCrystal.markov([[0.8, 0.2], [0.1, 0.9]]),
                ComputeCrystal.markov([[0.5, 0.5], [0.25, 0.75]]),
                ComputeCrystal.markov([[0.9, 0.1], [0.4, 0.6]]),
            )
            self._publish_chain(bank, chain)
            fused = fuse_markov_chain(chain)
            bank.publish_crystal(fused)
            source_program = ComputeProgram.compose(chain)
            fused_program = ComputeProgram.compose((fused,))
            bank.publish_program(source_program)
            bank.publish_program(fused_program)
            charge = ComputeChargeReceipt.create(
                source_program=source_program,
                source_crystals=chain,
                fused_crystal=fused,
                charge_verifier_sha256=_sha("markov-charge-verifier"),
                verification_receipt_sha256=_sha("markov-charge-receipt"),
            )
            bank.publish_charge(charge)
            execution = ComputeCrystalVM(bank).execute(
                fused_program.sha256,
                np.array([[0.37, 0.63]], dtype=np.float64),
                charge_basis_sha256=charge.sha256,
            )
            self.assertEqual(execution.receipt.charge_basis_sha256, charge.sha256)
            self.assertEqual(
                execution.receipt.equivalent_unfused_source_work,
                sum(crystal.discharge_work_units for crystal in chain),
            )
            self.assertGreater(execution.receipt.historical_work_released, 0)

    def test_manifest_stale_cas_and_concurrent_append_are_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "bank"
            first_bank = ComputeCrystalBank(root)
            first = ComputeCrystal.permutation([1, 0])
            first_bank.publish_crystal(first, expected_generation=0)
            with self.assertRaises(ComputeCrystalConflictError):
                first_bank.publish_crystal(
                    ComputeCrystal.permutation([2, 0, 1]),
                    expected_generation=0,
                )

            crystals = tuple(
                ComputeCrystal.affine(
                    [[float(index + 1)]],
                    [float(index)],
                    extensions={"index": index},
                )
                for index in range(8)
            )
            errors: list[BaseException] = []

            def publish(crystal: ComputeCrystal) -> None:
                try:
                    ComputeCrystalBank(root).publish_crystal(crystal)
                except BaseException as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            threads = [
                threading.Thread(target=publish, args=(item,)) for item in crystals
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(errors, [])
            manifest = ComputeCrystalBank(root).manifest()
            self.assertEqual(
                set(manifest.crystal_sha256s),
                {first.sha256, *(item.sha256 for item in crystals)},
            )
            self.assertEqual(manifest.generation, 1 + len(crystals))

    def test_transient_manifest_cas_conflict_is_retried_without_rewriting_object(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "bank")
            bank = ComputeCrystalBank(store, manifest_retry_limit=3)
            crystal = ComputeCrystal.permutation([1, 2, 0])
            original_publish = store.publish_state
            conflict_count = 0

            def conflict_once(name: str, payload: bytes, **kwargs: object):
                nonlocal conflict_count
                if name == COMPUTE_BANK_MANIFEST_STATE and conflict_count == 0:
                    conflict_count += 1
                    from immer.runtimes.ooe.crystal import ManifestConflictError

                    raise ManifestConflictError("simulated competing append")
                return original_publish(name, payload, **kwargs)

            with mock.patch.object(store, "publish_state", side_effect=conflict_once):
                publication = bank.publish_crystal(crystal)

            self.assertEqual(conflict_count, 1)
            self.assertTrue(publication.manifest_changed)
            self.assertFalse(publication.object_created)
            self.assertEqual(bank.restore_crystal(crystal.sha256), crystal)

    def test_crash_between_object_and_manifest_is_idempotently_recoverable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "bank")
            bank = ComputeCrystalBank(store)
            crystal = ComputeCrystal.permutation([2, 1, 0])
            original_publish = store.publish_state

            def crash_once(name: str, payload: bytes, **kwargs: object):
                if name == COMPUTE_BANK_MANIFEST_STATE:
                    raise OSError("simulated power loss")
                return original_publish(name, payload, **kwargs)

            with mock.patch.object(store, "publish_state", side_effect=crash_once):
                with self.assertRaisesRegex(OSError, "simulated power loss"):
                    bank.publish_crystal(crystal)

            self.assertEqual(ComputeCrystalBank(store).manifest().generation, 0)
            recovered = ComputeCrystalBank(store).publish_crystal(crystal)
            self.assertFalse(recovered.object_created)
            self.assertTrue(recovered.manifest_changed)
            self.assertEqual(recovered.generation, 1)
            self.assertEqual(
                ComputeCrystalBank(store).restore_crystal(crystal.sha256), crystal
            )

    def test_crash_after_head_cas_recovers_missing_immutable_commit_marker(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "bank")
            bank = ComputeCrystalBank(store)
            crystal = ComputeCrystal.permutation([1, 0])
            original_publish = store.publish_state

            def crash_at_commit(name: str, payload: bytes, **kwargs: object):
                if name.startswith("ooe-compute-crystal-bank-commit/v1:"):
                    raise OSError("simulated commit-marker power loss")
                return original_publish(name, payload, **kwargs)

            with mock.patch.object(store, "publish_state", side_effect=crash_at_commit):
                with self.assertRaisesRegex(OSError, "commit-marker power loss"):
                    bank.publish_crystal(crystal)

            recovered = ComputeCrystalBank(store).manifest()
            self.assertEqual(recovered.generation, 1)
            self.assertIn(crystal.sha256, recovered.crystal_sha256s)

    def test_fully_resealed_manifest_rollback_is_rejected_by_history_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "bank")
            bank = ComputeCrystalBank(store)
            first = ComputeCrystal.permutation([1, 0])
            second = ComputeCrystal.permutation([2, 0, 1])
            bank.publish_crystal(first)
            old_head = bank.manifest()
            bank.publish_crystal(second)
            self.assertEqual(bank.manifest().generation, 2)

            head_path = _state_path(store, COMPUTE_BANK_MANIFEST_STATE)
            _reseal_state_payload(
                head_path,
                name=COMPUTE_BANK_MANIFEST_STATE,
                payload=old_head.to_bytes(),
            )
            self.assertEqual(
                store.restore_state(COMPUTE_BANK_MANIFEST_STATE), old_head.to_bytes()
            )
            with self.assertRaisesRegex(
                ComputeCrystalIntegrityError, "resealed rollback"
            ):
                ComputeCrystalBank(store).manifest()

    def test_fully_resealed_alternate_inventory_without_history_is_rejected(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "bank")
            bank = ComputeCrystalBank(store)
            first = ComputeCrystal.permutation([1, 0])
            second = ComputeCrystal.permutation([2, 0, 1])
            bank.publish_crystal(first)
            predecessor = bank.manifest()
            bank.publish_crystal(second)
            alternate = hashlib.sha256(b"alternate inventory").hexdigest()
            forged = ComputeBankManifest(
                generation=2,
                crystal_sha256s=tuple(
                    sorted((*predecessor.crystal_sha256s, alternate))
                ),
                program_sha256s=(),
                charge_sha256s=(),
                previous_manifest_sha256=predecessor.sha256,
            )
            head_path = _state_path(store, COMPUTE_BANK_MANIFEST_STATE)
            _reseal_state_payload(
                head_path,
                name=COMPUTE_BANK_MANIFEST_STATE,
                payload=forged.to_bytes(),
            )
            self.assertEqual(
                store.restore_state(COMPUTE_BANK_MANIFEST_STATE), forged.to_bytes()
            )
            with self.assertRaisesRegex(
                ComputeCrystalIntegrityError, "immutable history"
            ):
                ComputeCrystalBank(store).manifest()

    def test_external_anchor_rejects_complete_valid_bank_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            victim_root = root / "victim"
            attacker_root = root / "replacement"

            victim = ComputeCrystalBank(victim_root)
            victim_crystal = ComputeCrystal.permutation([1, 0])
            victim.publish_crystal(victim_crystal)
            victim_program = ComputeProgram.compose((victim_crystal,))
            anchor_publication = victim.publish_program(victim_program)
            trusted_anchor = anchor_publication.current_anchor_sha256
            self.assertEqual(trusted_anchor, victim.current_anchor_sha256())

            anchored_descendant = ComputeCrystalBank(
                victim_root,
                trusted_manifest_sha256=trusted_anchor,
            )
            descendant = anchored_descendant.publish_crystal(
                ComputeCrystal.permutation([2, 0, 1])
            )
            self.assertNotEqual(descendant.current_anchor_sha256, trusted_anchor)
            self.assertEqual(anchored_descendant.manifest().generation, 3)

            replacement = ComputeCrystalBank(attacker_root)
            replacement_crystal = ComputeCrystal.affine([[3.0]], [2.0])
            replacement.publish_crystal(replacement_crystal)
            replacement_program = ComputeProgram.compose((replacement_crystal,))
            replacement.publish_program(replacement_program)
            replacement_head = replacement.manifest()

            shutil.rmtree(victim_root / "state")
            shutil.copytree(attacker_root / "state", victim_root / "state")

            unanchored = ComputeCrystalBank(victim_root)
            self.assertEqual(unanchored.manifest(), replacement_head)
            np.testing.assert_array_equal(
                ComputeCrystalVM(unanchored)
                .execute(
                    replacement_program.sha256,
                    np.array([4.0], dtype=np.float64),
                )
                .output,
                np.array([14.0], dtype=np.float64),
            )

            anchored = ComputeCrystalBank(
                victim_root,
                trusted_manifest_sha256=trusted_anchor,
            )
            with self.assertRaisesRegex(ComputeCrystalIntegrityError, "trusted anchor"):
                anchored.manifest()
            resolver_anchored = ComputeCrystalBank(
                victim_root,
                trusted_head_resolver=lambda: trusted_anchor,
            )
            with self.assertRaisesRegex(ComputeCrystalIntegrityError, "trusted anchor"):
                ComputeCrystalVM(resolver_anchored).execute(
                    replacement_program.sha256,
                    np.array([4.0], dtype=np.float64),
                )

    def test_payload_and_manifest_tamper_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "bank")
            bank = ComputeCrystalBank(store)
            crystal = ComputeCrystal.permutation([1, 0, 2])
            bank.publish_crystal(crystal)

            object_path = _state_path(
                store, ComputeCrystalBank.crystal_state_name(crystal.sha256)
            )
            os.chmod(object_path, 0o644)
            object_data = bytearray(object_path.read_bytes())
            object_data[len(object_data) // 2] ^= 1
            object_path.write_bytes(object_data)
            with self.assertRaises(ComputeCrystalIntegrityError):
                bank.restore_crystal(crystal.sha256)

        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "bank")
            bank = ComputeCrystalBank(store)
            bank.publish_crystal(ComputeCrystal.permutation([1, 0]))
            manifest_path = _state_path(store, COMPUTE_BANK_MANIFEST_STATE)
            os.chmod(manifest_path, 0o644)
            data = bytearray(manifest_path.read_bytes())
            data[len(data) // 3] ^= 1
            manifest_path.write_bytes(data)
            with self.assertRaises(ComputeCrystalIntegrityError):
                bank.manifest()

    def test_forged_charge_accounting_is_rejected_against_source_program(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = ComputeCrystalBank(Path(temporary) / "bank")
            chain = _affine_chain()[:2]
            self._publish_chain(bank, chain)
            fused = fuse_affine_chain(chain)
            bank.publish_crystal(fused)
            source_program = ComputeProgram.compose(chain)
            bank.publish_program(source_program)
            charge = ComputeChargeReceipt.create(
                source_program=source_program,
                source_crystals=chain,
                fused_crystal=fused,
                charge_verifier_sha256=_sha("forgery-verifier"),
                verification_receipt_sha256=_sha("forgery-receipt"),
            )
            self.assertEqual(ComputeChargeReceipt.from_bytes(charge.to_bytes()), charge)
            forged = replace(
                charge,
                source_work_units=charge.source_work_units + 1,
            )
            with self.assertRaises(ComputeCrystalIntegrityError):
                bank.publish_charge(forged)

    def test_program_and_manifest_canonical_roundtrips_reject_resealed_tamper(
        self,
    ) -> None:
        crystal = ComputeCrystal.permutation([1, 0])
        program = ComputeProgram.compose((crystal,))
        document = json.loads(program.to_bytes())
        document["body"]["input_abi"]["dtype"] = "int64"
        document["body_sha256"] = hashlib.sha256(
            canonical_json_bytes(document["body"])
        ).hexdigest()
        forged = ComputeProgram.from_bytes(canonical_json_bytes(document))
        with self.assertRaises(ComputeCrystalIntegrityError):
            forged.validate_crystals((crystal,))

        empty = ComputeBankManifest.empty()
        self.assertEqual(ComputeBankManifest.from_bytes(empty.to_bytes()), empty)
        manifest_document = json.loads(
            ComputeBankManifest(
                generation=1,
                crystal_sha256s=(crystal.sha256,),
                program_sha256s=(),
                charge_sha256s=(),
                previous_manifest_sha256=empty.sha256,
            ).to_bytes()
        )
        manifest_document["body"]["generation"] = 2
        with self.assertRaises(ComputeCrystalIntegrityError):
            ComputeBankManifest.from_bytes(canonical_json_bytes(manifest_document))


if __name__ == "__main__":
    unittest.main()
