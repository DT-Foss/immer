from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from immer.runtimes.ooe.compute_crystals import (
    ComputeCrystal,
    ComputeCrystalBank,
)
from immer.runtimes.ooe.crystal import CrystalStore
from immer.runtimes.ooe.executable_lexicon import (
    CompiledWordReceipt,
    ExecutableLexicon,
    ExecutableLexiconBank,
    ExecutableLexiconCompileError,
    ExecutableLexiconConflictError,
    ExecutableLexiconIntegrityError,
    ExecutableLexiconState,
    ExecutableWordCompiler,
    ExecutableWordDefinition,
    PrimitiveWordBinding,
)
from immer.runtimes.ooe.identity import canonical_json_bytes


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _contract() -> tuple[str, str, dict[str, str]]:
    return _sha("snapshot"), _sha("frontier"), {"verifier": _sha("verifier")}


def _initial(
    primitives: tuple[PrimitiveWordBinding, ...] | None = None,
) -> ExecutableLexiconState:
    snapshot, frontier, authorities = _contract()
    return ExecutableLexiconState.initial(
        primitives
        or (
            PrimitiveWordBinding("tau-0", "crystal", _sha("crystal-0")),
            PrimitiveWordBinding("tau-1", "crystal", _sha("crystal-1")),
        ),
        language_snapshot_sha256=snapshot,
        frontier_sha256=frontier,
        authority_hashes=authorities,
    )


def _definition(word: str, children: tuple[str, ...]) -> ExecutableWordDefinition:
    snapshot, frontier, authorities = _contract()
    return ExecutableWordDefinition.create(
        word,
        children,
        language_snapshot_sha256=snapshot,
        frontier_sha256=frontier,
        authority_hashes=authorities,
    )


def _state_path(store: CrystalStore, state_name: str) -> Path:
    filename = hashlib.sha256(state_name.encode("utf-8")).hexdigest() + ".state"
    return Path(store.root) / "state" / filename


def _reseal_state_payload(path: Path, *, name: str, payload: bytes) -> None:
    envelope = json.loads(path.read_bytes())
    envelope["name"] = name
    envelope["payload_base64"] = base64.b64encode(payload).decode("ascii")
    envelope["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    os.chmod(path, 0o644)
    path.write_bytes(canonical_json_bytes(envelope))


class DefinitionProtocolTests(unittest.TestCase):
    def test_definition_roundtrip_tamper_conflict_and_stale_contract(self) -> None:
        definition = _definition("tau-macro", ("tau-0", "tau-1"))
        self.assertEqual(
            ExecutableWordDefinition.from_bytes(definition.to_bytes()), definition
        )
        document = json.loads(definition.to_bytes())
        document["body"]["child_word_ids"].append("tau-0")
        with self.assertRaisesRegex(ExecutableLexiconIntegrityError, "body"):
            ExecutableWordDefinition.from_bytes(canonical_json_bytes(document))

        state = _initial().with_definition(definition)
        self.assertIs(state.with_definition(definition), state)
        with self.assertRaisesRegex(ExecutableLexiconConflictError, "conflicting"):
            state.with_definition(_definition("tau-macro", ("tau-1",)))
        with self.assertRaisesRegex(ExecutableLexiconConflictError, "primitive"):
            state.with_definition(_definition("tau-0", ("tau-1",)))
        with self.assertRaisesRegex(ExecutableLexiconConflictError, "unknown"):
            state.with_definition(_definition("tau-unknown", ("tau-missing",)))
        with self.assertRaisesRegex(ExecutableLexiconConflictError, "cyclic"):
            state.with_definition(_definition("tau-self", ("tau-self",)))

        snapshot, frontier, _ = _contract()
        stale = ExecutableWordDefinition.create(
            "tau-stale",
            ("tau-0",),
            language_snapshot_sha256=snapshot,
            frontier_sha256=frontier,
            authority_hashes={"verifier": _sha("changed-verifier")},
        )
        with self.assertRaisesRegex(ExecutableLexiconConflictError, "stale"):
            state.with_definition(stale)

    def test_contextual_definition_requires_its_exact_contextual_lexicon(self) -> None:
        snapshot, frontier, authorities = _contract()
        contextual = ExecutableWordDefinition.create(
            "tau-contextual",
            ("tau-0", "tau-1"),
            language_snapshot_sha256=snapshot,
            frontier_sha256=frontier,
            authority_hashes=authorities,
            context_id="context-red",
        )
        self.assertEqual(
            ExecutableWordDefinition.from_bytes(contextual.to_bytes()), contextual
        )
        contextual_state = ExecutableLexiconState.initial(
            _initial().primitive_bindings,
            language_snapshot_sha256=snapshot,
            frontier_sha256=frontier,
            authority_hashes=authorities,
            context_id="context-red",
        ).with_definition(contextual)
        self.assertEqual(contextual_state.context_id, "context-red")
        self.assertEqual(
            ExecutableLexiconState.from_bytes(contextual_state.to_bytes()),
            contextual_state,
        )
        with self.assertRaisesRegex(ExecutableLexiconConflictError, "stale"):
            _initial().with_definition(contextual)

    def test_nested_definition_repair_happens_only_on_first_use(self) -> None:
        sender = ExecutableLexicon(_initial())
        sender.coin("tau-inner", ("tau-0", "tau-1", "tau-0"))
        sender.coin("tau-outer", ("tau-inner", "tau-1", "tau-inner"))
        receiver = ExecutableLexicon(_initial())

        decoded, repairs = receiver.decode_or_request(("tau-outer",), sender)
        self.assertEqual(decoded, ("tau-outer",))
        self.assertEqual(repairs, 2)
        self.assertEqual(
            receiver.state.expand_word("tau-outer", max_actions=7),
            ("tau-0", "tau-1", "tau-0", "tau-1", "tau-0", "tau-1", "tau-0"),
        )
        decoded_again, repairs_again = receiver.decode_or_request(
            ("tau-outer",), sender
        )
        self.assertEqual(decoded_again, ("tau-outer",))
        self.assertEqual(repairs_again, 0)
        missing, missing_repairs = receiver.decode_or_request(("tau-never",), sender)
        self.assertIsNone(missing)
        self.assertEqual(missing_repairs, 0)

    def test_state_rejects_a_forged_multi_definition_cycle(self) -> None:
        first = _definition("tau-a", ("tau-b",))
        second = _definition("tau-b", ("tau-a",))
        with self.assertRaisesRegex(ValueError, "cycle"):
            ExecutableLexiconState(
                generation=2,
                previous_state_sha256=_initial().sha256,
                language_snapshot_sha256=_sha("snapshot"),
                frontier_sha256=_sha("frontier"),
                authority_hashes=(("verifier", _sha("verifier")),),
                primitive_bindings=_initial().primitive_bindings,
                definitions=(first, second),
            )


class RecursiveCompilerTests(unittest.TestCase):
    def test_depth_twelve_word_is_one_crystal_with_transitive_charge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compute_root = Path(temporary) / "compute"
            lexicon_root = Path(temporary) / "lexicon"
            compute_bank = ComputeCrystalBank(compute_root)
            primitive = ComputeCrystal.affine([[1.0]], [1.0])
            compute_bank.publish_crystal(primitive)
            initial = _initial(
                (PrimitiveWordBinding("tau-p", "crystal", primitive.sha256),)
            )
            lexicon_bank = ExecutableLexiconBank(lexicon_root, initial_state=initial)
            current = "tau-p"
            for depth in range(1, 13):
                definition = _definition(
                    f"tau-depth-{depth}",
                    (current, "tau-p", current),
                )
                lexicon_bank.append_definition(definition)
                current = definition.new_word_id

            source_state = lexicon_bank.head()
            self.assertEqual(source_state.expanded_length(current), 8_191)
            self.assertEqual(source_state.reachable_reference_count(current), 36)
            compiler = ExecutableWordCompiler(source_state, compute_bank)
            receipt = compiler.compile(current)
            self.assertEqual(receipt.expanded_primitive_actions, 8_191)
            self.assertEqual(receipt.stored_definition_references, 36)
            self.assertEqual(receipt.artifact_kind, "crystal")
            self.assertTrue(receipt.constant_discharge)
            self.assertIsNotNone(receipt.charge_sha256)
            self.assertEqual(
                CompiledWordReceipt.from_bytes(receipt.to_bytes()), receipt
            )

            fused = compute_bank.restore_crystal(receipt.artifact_sha256)
            self.assertEqual(len(fused.parent_sha256s), 3)
            self.assertEqual(fused.parent_sha256s[0], fused.parent_sha256s[2])
            self.assertEqual(len(compute_bank.manifest().crystal_sha256s), 13)
            execution = compiler.execute(receipt, np.array([3.0], dtype=np.float64))
            np.testing.assert_array_equal(execution.output, np.array([8_194.0]))
            self.assertEqual(
                execution.receipt.equivalent_unfused_source_work,
                8_191 * primitive.discharge_work_units,
            )
            self.assertEqual(
                execution.receipt.historical_work_released,
                8_191 * primitive.discharge_work_units - fused.discharge_work_units,
            )

            stored = lexicon_bank.append_compiled_receipt(
                receipt, expected_head_sha256=source_state.sha256
            )
            self.assertEqual(stored.compiled_receipts, (receipt,))
            reopened_state = ExecutableLexiconBank(lexicon_root).head()
            reopened_compute = ComputeCrystalBank(compute_root)
            reopened_compiler = ExecutableWordCompiler(reopened_state, reopened_compute)
            reopened_execution = reopened_compiler.execute(
                reopened_state.compiled_receipts[0],
                np.array([11.0], dtype=np.float64),
            )
            np.testing.assert_array_equal(
                reopened_execution.output, np.array([8_202.0])
            )

    def test_mixed_operator_family_falls_back_to_bounded_program(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compute_bank = ComputeCrystalBank(Path(temporary) / "compute")
            affine = ComputeCrystal.affine([[2.0]], [1.0])
            permutation = ComputeCrystal.permutation([0])
            compute_bank.publish_crystal(affine)
            compute_bank.publish_crystal(permutation)
            state = _initial(
                (
                    PrimitiveWordBinding("tau-a", "crystal", affine.sha256),
                    PrimitiveWordBinding("tau-p", "crystal", permutation.sha256),
                )
            ).with_definition(_definition("tau-mixed", ("tau-a", "tau-p")))
            compiler = ExecutableWordCompiler(state, compute_bank)
            receipt = compiler.compile("tau-mixed")
            self.assertEqual(receipt.artifact_kind, "program")
            self.assertFalse(receipt.constant_discharge)
            self.assertIsNone(receipt.charge_sha256)
            self.assertEqual(
                len(
                    compute_bank.restore_program(receipt.program_sha256).crystal_sha256s
                ),
                2,
            )
            execution = compiler.execute(receipt, np.array([4.0], dtype=np.float64))
            np.testing.assert_array_equal(execution.output, np.array([9.0]))

    def test_nonfusible_expansion_over_program_bound_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            compute_bank = ComputeCrystalBank(Path(temporary) / "compute")
            affine = ComputeCrystal.affine([[1.0]], [1.0])
            permutation = ComputeCrystal.permutation([0])
            compute_bank.publish_crystal(affine)
            compute_bank.publish_crystal(permutation)
            state = _initial(
                (
                    PrimitiveWordBinding("tau-a", "crystal", affine.sha256),
                    PrimitiveWordBinding("tau-p", "crystal", permutation.sha256),
                )
            )
            state = state.with_definition(_definition("tau-mix-1", ("tau-a", "tau-p")))
            state = state.with_definition(
                _definition("tau-mix-2", ("tau-mix-1", "tau-mix-1"))
            )
            state = state.with_definition(
                _definition("tau-mix-3", ("tau-mix-2", "tau-mix-2"))
            )
            with mock.patch(
                "immer.runtimes.ooe.executable_lexicon.MAX_PROGRAM_STEPS", 4
            ):
                with self.assertRaisesRegex(
                    ExecutableLexiconCompileError, "bounded program"
                ):
                    ExecutableWordCompiler(state, compute_bank).compile("tau-mix-3")

    def test_non_compute_word_can_be_named_but_not_compiled(self) -> None:
        state = _initial(
            (PrimitiveWordBinding("tau-organ", "organ", _sha("organ")),)
        ).with_definition(_definition("tau-wrapper", ("tau-organ", "tau-organ")))
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                ExecutableLexiconCompileError, "named but not compiled"
            ):
                ExecutableWordCompiler(
                    state, ComputeCrystalBank(Path(temporary) / "compute")
                ).compile("tau-wrapper")

    def test_compiled_receipt_rejects_a_complete_compute_bank_replacement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primitive = ComputeCrystal.affine([[1.0]], [1.0])
            original_bank = ComputeCrystalBank(root / "original")
            original_bank.publish_crystal(primitive)
            state = _initial(
                (PrimitiveWordBinding("tau-p", "crystal", primitive.sha256),)
            ).with_definition(_definition("tau-m", ("tau-p", "tau-p")))
            original_receipt = ExecutableWordCompiler(state, original_bank).compile(
                "tau-m"
            )

            replacement_bank = ComputeCrystalBank(root / "replacement")
            replacement_bank.publish_crystal(ComputeCrystal.affine([[3.0]], [7.0]))
            replacement_bank.publish_crystal(primitive)
            replacement_compiler = ExecutableWordCompiler(state, replacement_bank)
            replacement_receipt = replacement_compiler.compile("tau-m")
            self.assertEqual(
                replacement_receipt.artifact_sha256,
                original_receipt.artifact_sha256,
            )
            self.assertNotEqual(
                replacement_receipt.compute_bank_anchor_sha256,
                original_receipt.compute_bank_anchor_sha256,
            )
            with self.assertRaisesRegex(
                ExecutableLexiconIntegrityError,
                "anchor is not an ancestor",
            ):
                replacement_compiler.execute(
                    original_receipt,
                    np.array([2.0], dtype=np.float64),
                )


class LexiconBankTests(unittest.TestCase):
    def test_atomic_append_reopen_and_stale_cas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = ExecutableLexiconBank(temporary, initial_state=_initial())
            root = bank.head()
            first = bank.append_definition(
                _definition("tau-a", ("tau-0", "tau-1")),
                expected_head_sha256=root.sha256,
            )
            self.assertEqual(first.generation, 2)
            self.assertEqual(ExecutableLexiconBank(temporary).head(), first)
            with self.assertRaisesRegex(ExecutableLexiconConflictError, "changed"):
                bank.append_definition(
                    _definition("tau-b", ("tau-1", "tau-0")),
                    expected_head_sha256=root.sha256,
                )

    def test_crash_before_head_is_retryable_and_after_head_recovers_commit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(temporary)
            bank = ExecutableLexiconBank(store, initial_state=_initial())
            definition = _definition("tau-a", ("tau-0", "tau-1"))
            original_publish = store.publish_state

            def crash_before_head(name: str, payload: bytes, **kwargs: object):
                if name == bank.head_state_name():
                    raise OSError("power loss before head")
                return original_publish(name, payload, **kwargs)

            with mock.patch.object(
                store, "publish_state", side_effect=crash_before_head
            ):
                with self.assertRaisesRegex(OSError, "before head"):
                    bank.append_definition(definition)
            self.assertEqual(ExecutableLexiconBank(store).head().generation, 1)
            recovered = ExecutableLexiconBank(store).append_definition(definition)
            self.assertEqual(recovered.generation, 2)

        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(temporary)
            bank = ExecutableLexiconBank(store, initial_state=_initial())
            definition = _definition("tau-a", ("tau-0", "tau-1"))
            original_publish = store.publish_state

            def crash_after_head(name: str, payload: bytes, **kwargs: object):
                if name.startswith("ooe-executable-lexicon-commit/v1:"):
                    raise OSError("power loss after head")
                return original_publish(name, payload, **kwargs)

            with mock.patch.object(
                store, "publish_state", side_effect=crash_after_head
            ):
                with self.assertRaisesRegex(OSError, "after head"):
                    bank.append_definition(definition)
            recovered = ExecutableLexiconBank(store).head()
            self.assertEqual(recovered.generation, 2)
            self.assertIn("tau-a", recovered.definition_map)

    def test_resealed_rollback_and_wrong_trusted_head_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(temporary)
            bank = ExecutableLexiconBank(store, initial_state=_initial())
            old = bank.append_definition(_definition("tau-a", ("tau-0",)))
            latest = bank.append_definition(_definition("tau-b", ("tau-a",)))
            self.assertEqual(latest.generation, 3)
            head_path = _state_path(store, bank.head_state_name())
            _reseal_state_payload(
                head_path,
                name=bank.head_state_name(),
                payload=old.to_bytes(),
            )
            with self.assertRaisesRegex(ExecutableLexiconIntegrityError, "rollback"):
                ExecutableLexiconBank(store).head()

        with tempfile.TemporaryDirectory() as temporary:
            bank = ExecutableLexiconBank(temporary, initial_state=_initial())
            with self.assertRaisesRegex(
                ExecutableLexiconIntegrityError, "trusted head"
            ):
                ExecutableLexiconBank(
                    temporary, trusted_head_sha256=_sha("unrelated-head")
                ).head()

    def test_concurrent_expected_head_allows_exactly_one_append(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bank = ExecutableLexiconBank(root, initial_state=_initial())
            anchor = bank.current_anchor_sha256()
            successes: list[str] = []
            errors: list[BaseException] = []

            def append(word: str) -> None:
                try:
                    state = ExecutableLexiconBank(root).append_definition(
                        _definition(word, ("tau-0", "tau-1")),
                        expected_head_sha256=anchor,
                    )
                    successes.append(state.sha256)
                except BaseException as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            threads = [
                threading.Thread(target=append, args=("tau-left",)),
                threading.Thread(target=append, args=("tau-right",)),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len(successes), 1)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], ExecutableLexiconConflictError)
            self.assertEqual(ExecutableLexiconBank(root).head().generation, 2)

    def test_committed_generation_cannot_smuggle_two_appends(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(temporary)
            bank = ExecutableLexiconBank(store, initial_state=_initial())
            initial = bank.head()
            forged = ExecutableLexiconState(
                generation=2,
                previous_state_sha256=initial.sha256,
                language_snapshot_sha256=initial.language_snapshot_sha256,
                frontier_sha256=initial.frontier_sha256,
                authority_hashes=initial.authority_hashes,
                primitive_bindings=initial.primitive_bindings,
                definitions=tuple(
                    sorted(
                        (
                            _definition("tau-a", ("tau-0",)),
                            _definition("tau-b", ("tau-1",)),
                        ),
                        key=lambda item: item.new_word_id,
                    )
                ),
            )
            store.publish_state(
                bank.history_state_name(forged.sha256), forged.to_bytes()
            )
            store.publish_state(
                bank.head_state_name(),
                forged.to_bytes(),
                expected_sha256=initial.sha256,
            )
            store.publish_state(
                bank.commit_state_name(forged.sha256),
                canonical_json_bytes(
                    {
                        "schema": "immer-ooe-executable-lexicon-commit/v1",
                        "state_sha256": forged.sha256,
                    }
                ),
            )
            with self.assertRaisesRegex(
                ExecutableLexiconIntegrityError,
                "append-only extension",
            ):
                ExecutableLexiconBank(store).head()

    def test_lock_inode_replacement_is_rejected_after_flock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bank = ExecutableLexiconBank(temporary, initial_state=_initial())
            original_lstat = Path.lstat

            def replaced_inode(path: Path):
                metadata = original_lstat(path)
                if path.name == ".executable-lexicon.lock":
                    return SimpleNamespace(
                        st_dev=metadata.st_dev,
                        st_ino=metadata.st_ino + 1,
                    )
                return metadata

            with mock.patch.object(Path, "lstat", replaced_inode):
                with self.assertRaisesRegex(
                    ExecutableLexiconIntegrityError,
                    "lock changed",
                ):
                    bank.head()


if __name__ == "__main__":
    unittest.main()
