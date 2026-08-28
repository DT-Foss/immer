from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import unittest

from immer.runtimes.ooe.affine_monoid import (
    AffineActionAtom,
    FieldBound,
    RingSpec,
    VectorStateSchema,
)
from immer.runtimes.ooe.identity import canonical_json_bytes
from immer.runtimes.ooe.modular_affine_fit import (
    MODULAR_AFFINE_FIT_ALGORITHM_SHA256,
    ModularAffineCalibrationError,
    ModularAffineFitError,
    ModularAffineFitReceipt,
    ModularAffineIdentificationError,
    ModularAffineIntegrityError,
    ModularAffineObservation,
    fit_modular_affine_action,
    verify_modular_affine_fit,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _schema(
    dimension: int,
    *,
    modulus: int = 17,
    dead_index: int | None = None,
    name: str = "FitState",
) -> VectorStateSchema:
    dead = dimension if dead_index is None else dead_index
    fields = [f"x{index}" for index in range(dimension)]
    fields.insert(dead, "dead")
    rings = [RingSpec("modular", modulus) for _ in range(dimension)]
    rings.insert(dead, RingSpec())
    bounds = [FieldBound(0, modulus - 1) for _ in range(dimension)]
    bounds.insert(dead, FieldBound(0, 1))
    initial = [0] * (dimension + 1)
    dead_state = [0] * (dimension + 1)
    dead_state[dead] = 1
    return VectorStateSchema(
        name=name,
        fields=tuple(fields),
        rings=tuple(rings),
        bounds=tuple(bounds),
        initial=tuple(initial),
        dead=tuple(dead_state),
        logical_capacity=min(modulus**dimension, 1_000_000),
        max_abs_bits=64,
        dead_index=dead,
    )


def _apply_formula(
    values: tuple[int, ...],
    matrix: tuple[tuple[int, ...], ...],
    bias: tuple[int, ...],
    modulus: int,
) -> tuple[int, ...]:
    return tuple(
        (offset + sum(coefficient * value for coefficient, value in zip(row, values)))
        % modulus
        for row, offset in zip(matrix, bias)
    )


def _observation(
    schema: VectorStateSchema,
    *,
    values: tuple[int, ...],
    matrix: tuple[tuple[int, ...], ...],
    bias: tuple[int, ...],
    split: str,
    temporal_index: int,
    identity: str,
    modulus: int = 17,
    output: tuple[int, ...] | None = None,
) -> ModularAffineObservation:
    return ModularAffineObservation(
        split=split,
        temporal_index=temporal_index,
        input_values=values,
        output_values=(
            _apply_formula(values, matrix, bias, modulus) if output is None else output
        ),
        modulus=modulus,
        schema_sha256=schema.sha256,
        execution_identity_sha256=identity,
        source_receipt_sha256=_hash(f"source:{split}:{temporal_index}"),
    )


def _full_state(schema: VectorStateSchema, values: tuple[int, ...]) -> tuple[int, ...]:
    result: list[int] = []
    cursor = 0
    for index in range(schema.dimension):
        if index == schema.dead_index:
            result.append(0)
        else:
            result.append(values[cursor])
            cursor += 1
    return tuple(result)


def _active_state(
    schema: VectorStateSchema, values: tuple[int, ...]
) -> tuple[int, ...]:
    return tuple(
        value for index, value in enumerate(values) if index != schema.dead_index
    )


class ExactModularAffineFitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.modulus = 17
        self.schema = _schema(2, dead_index=1)
        self.identity = _hash("qwen-model-graph-feature-action-contract")
        self.verifier = _hash("exact-held-out-transition-verifier")
        self.matrix = ((2, 3), (7, 4))
        self.bias = (5, 11)
        self.train = tuple(
            _observation(
                self.schema,
                values=values,
                matrix=self.matrix,
                bias=self.bias,
                split="train",
                temporal_index=index,
                identity=self.identity,
            )
            for index, values in enumerate(((0, 0), (1, 0), (0, 1), (4, 9)))
        )
        self.calibration = tuple(
            _observation(
                self.schema,
                values=values,
                matrix=self.matrix,
                bias=self.bias,
                split="calibration",
                temporal_index=100 + index,
                identity=self.identity,
            )
            for index, values in enumerate(((5, 8), (13, 16), (9, 2)))
        )

    def fit(self):
        return fit_modular_affine_action(
            self.schema,
            name="LearnedTransition",
            modulus=self.modulus,
            execution_identity_sha256=self.identity,
            calibration_verifier_sha256=self.verifier,
            train_observations=self.train,
            calibration_observations=self.calibration,
        )

    def test_exact_action_uses_existing_atom_and_ring_on_unseen_states(self) -> None:
        fit = self.fit()
        self.assertIsInstance(fit.action, AffineActionAtom)
        self.assertEqual(fit.ring_spec, RingSpec("modular", self.modulus))
        self.assertEqual(fit.receipt.dimension, 2)
        self.assertEqual(fit.receipt.rank, 3)
        self.assertEqual(len(fit.receipt.pivot_observation_sha256s), 3)
        self.assertEqual(
            fit.receipt.algorithm_sha256,
            MODULAR_AFFINE_FIT_ALGORITHM_SHA256,
        )

        # These states were absent from both training and calibration.
        for values in ((2, 6), (8, 15), (16, 16), (12, 1)):
            state = self.schema.state(_full_state(self.schema, values))
            result = fit.action.apply(self.schema, state)
            self.assertEqual(
                _active_state(self.schema, result.values),
                _apply_formula(values, self.matrix, self.bias, self.modulus),
            )

        # Existing AffineActionAtom dead-state semantics remain absorbing.
        dead = self.schema.dead_state()
        self.assertEqual(fit.action.apply(self.schema, dead), dead)

    def test_receipt_roundtrip_restores_atom_and_replays_all_evidence(self) -> None:
        fit = self.fit()
        data = fit.receipt.to_bytes()
        restored = ModularAffineFitReceipt.from_bytes(data, schema=self.schema)
        self.assertEqual(restored, fit.receipt)
        self.assertEqual(restored.restore_action(self.schema), fit.action)
        self.assertEqual(
            verify_modular_affine_fit(
                restored,
                self.schema,
                train_observations=self.train,
                calibration_observations=self.calibration,
            ),
            restored.sha256,
        )
        self.assertEqual(
            ModularAffineObservation.from_bytes(self.train[0].to_bytes()),
            self.train[0],
        )

    def test_row_order_is_content_addressed_and_deterministic(self) -> None:
        forward = self.fit()
        reverse = fit_modular_affine_action(
            self.schema,
            name="LearnedTransition",
            modulus=self.modulus,
            execution_identity_sha256=self.identity,
            calibration_verifier_sha256=self.verifier,
            train_observations=tuple(reversed(self.train)),
            calibration_observations=tuple(reversed(self.calibration)),
        )
        self.assertEqual(reverse.action.to_bytes(), forward.action.to_bytes())
        self.assertEqual(reverse.receipt.to_bytes(), forward.receipt.to_bytes())

    def test_general_dimension_d_is_recovered_exactly(self) -> None:
        dimension = 4
        modulus = 101
        schema = _schema(dimension, modulus=modulus, dead_index=2, name="FitStateD4")
        matrix = tuple(
            tuple((3 * row + 5 * column + 2) % modulus for column in range(dimension))
            for row in range(dimension)
        )
        bias = (7, 19, 41, 83)
        basis = [(0,) * dimension]
        basis.extend(
            tuple(1 if column == row else 0 for column in range(dimension))
            for row in range(dimension)
        )
        train = tuple(
            _observation(
                schema,
                values=values,
                matrix=matrix,
                bias=bias,
                split="train",
                temporal_index=index,
                identity=self.identity,
                modulus=modulus,
            )
            for index, values in enumerate(basis)
        )
        calibration_values = ((11, 17, 23, 29), (97, 89, 71, 53))
        calibration = tuple(
            _observation(
                schema,
                values=values,
                matrix=matrix,
                bias=bias,
                split="calibration",
                temporal_index=100 + index,
                identity=self.identity,
                modulus=modulus,
            )
            for index, values in enumerate(calibration_values)
        )
        fit = fit_modular_affine_action(
            schema,
            name="LearnedD4",
            modulus=modulus,
            execution_identity_sha256=self.identity,
            calibration_verifier_sha256=self.verifier,
            train_observations=train,
            calibration_observations=calibration,
        )
        unseen = (37, 43, 47, 59)
        result = fit.action.apply(schema, schema.state(_full_state(schema, unseen)))
        self.assertEqual(
            _active_state(schema, result.values),
            _apply_formula(unseen, matrix, bias, modulus),
        )
        self.assertEqual(fit.receipt.rank, dimension + 1)

    def test_underdetermined_and_singular_training_are_rejected(self) -> None:
        with self.assertRaises(ModularAffineIdentificationError):
            fit_modular_affine_action(
                self.schema,
                name="TooFewRows",
                modulus=self.modulus,
                execution_identity_sha256=self.identity,
                calibration_verifier_sha256=self.verifier,
                train_observations=self.train[:2],
                calibration_observations=self.calibration,
            )

        singular = tuple(
            _observation(
                self.schema,
                values=values,
                matrix=self.matrix,
                bias=self.bias,
                split="train",
                temporal_index=200 + index,
                identity=self.identity,
            )
            for index, values in enumerate(((0, 0), (1, 1), (2, 2), (3, 3)))
        )
        with self.assertRaises(ModularAffineIdentificationError):
            fit_modular_affine_action(
                self.schema,
                name="SingularRows",
                modulus=self.modulus,
                execution_identity_sha256=self.identity,
                calibration_verifier_sha256=self.verifier,
                train_observations=singular,
                calibration_observations=self.calibration,
            )

    def test_composite_modulus_and_wrong_calibration_are_rejected(self) -> None:
        with self.assertRaisesRegex(ModularAffineFitError, "prime"):
            fit_modular_affine_action(
                self.schema,
                name="CompositeRing",
                modulus=15,
                execution_identity_sha256=self.identity,
                calibration_verifier_sha256=self.verifier,
                train_observations=self.train,
                calibration_observations=self.calibration,
            )

        wrong = replace(
            self.calibration[0],
            output_values=(
                (self.calibration[0].output_values[0] + 1) % self.modulus,
                self.calibration[0].output_values[1],
            ),
        )
        with self.assertRaises(ModularAffineCalibrationError):
            fit_modular_affine_action(
                self.schema,
                name="WrongCalibration",
                modulus=self.modulus,
                execution_identity_sha256=self.identity,
                calibration_verifier_sha256=self.verifier,
                train_observations=self.train,
                calibration_observations=(wrong, *self.calibration[1:]),
            )

    def test_identity_mismatch_and_cross_split_receipt_reuse_are_rejected(self) -> None:
        wrong_identity = replace(
            self.train[0], execution_identity_sha256=_hash("other-model-contract")
        )
        with self.assertRaisesRegex(ModularAffineIntegrityError, "identity mismatch"):
            fit_modular_affine_action(
                self.schema,
                name="WrongIdentity",
                modulus=self.modulus,
                execution_identity_sha256=self.identity,
                calibration_verifier_sha256=self.verifier,
                train_observations=(wrong_identity, *self.train[1:]),
                calibration_observations=self.calibration,
            )

        reused = replace(
            self.calibration[0],
            source_receipt_sha256=self.train[0].source_receipt_sha256,
        )
        with self.assertRaisesRegex(ModularAffineIntegrityError, "receipts overlap"):
            fit_modular_affine_action(
                self.schema,
                name="LeakedCalibration",
                modulus=self.modulus,
                execution_identity_sha256=self.identity,
                calibration_verifier_sha256=self.verifier,
                train_observations=self.train,
                calibration_observations=(reused, *self.calibration[1:]),
            )

    def test_observation_receipt_and_action_tampering_fail_closed(self) -> None:
        observation_document = json.loads(self.train[0].to_bytes())
        observation_document["body"]["input_values"][0] += 1
        with self.assertRaises(ModularAffineIntegrityError):
            ModularAffineObservation.from_bytes(
                canonical_json_bytes(observation_document)
            )

        fit = self.fit()
        receipt_document = json.loads(fit.receipt.to_bytes())
        receipt_document["body"]["rank"] -= 1
        with self.assertRaises(ModularAffineIntegrityError):
            ModularAffineFitReceipt.from_bytes(
                canonical_json_bytes(receipt_document), schema=self.schema
            )

        forged_action_identity = replace(
            fit.receipt, action_sha256=_hash("different-action")
        )
        with self.assertRaisesRegex(ModularAffineIntegrityError, "action identity"):
            forged_action_identity.restore_action(self.schema)

        other_schema = _schema(2, dead_index=1, name="OtherFitState")
        with self.assertRaisesRegex(ModularAffineIntegrityError, "schema identity"):
            fit.receipt.restore_action(other_schema)


if __name__ == "__main__":
    unittest.main()
