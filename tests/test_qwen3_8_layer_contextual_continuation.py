from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from immer.runtimes.qwen3_8 import layer_contextual_continuation as layer_bank
from immer.runtimes.qwen3_8.layer_contextual_continuation import (
    DEFAULT_LAYER_CONTEXTUAL_MAX_RECEIPTS,
    DEFAULT_LAYER_CONTEXTUAL_SKETCH_DIM,
    LAYER_CONTEXTUAL_STAGE,
    LayerContextualContinuationBank,
    LayerContextualContinuationCapacityError,
    LayerContextualContinuationConflictError,
    LayerContextualContinuationIdentity,
    LayerContextualContinuationIdentityError,
    LayerContextualContinuationIntegrityError,
    LayerContextualContinuationKey,
    LayerContextualContinuationTransaction,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _identity(
    *,
    runtime: str = "runtime",
    layers: tuple[int, ...] = (1, 3),
    seed: int = 17,
) -> LayerContextualContinuationIdentity:
    return LayerContextualContinuationIdentity(
        runtime_sha256=_hash(runtime),
        model_sha256=_hash("model"),
        q4_sha256=_hash("q4"),
        tokenizer_sha256=_hash("tokenizer"),
        hidden_dim=12,
        layers=layers,
        projection_seed=seed,
    )


def _hidden(offset: float = 0.0) -> torch.Tensor:
    return torch.arange(1, 13, dtype=torch.float32).reshape(1, 1, 12) + offset


def _axis_key(layer: int, token: int, axis: int) -> LayerContextualContinuationKey:
    values = torch.zeros(DEFAULT_LAYER_CONTEXTUAL_SKETCH_DIM, dtype=torch.int8)
    values[axis] = 127
    return LayerContextualContinuationKey(layer, token, values)


def _tie_key(layer: int, token: int, left: int, right: int):
    values = torch.zeros(DEFAULT_LAYER_CONTEXTUAL_SKETCH_DIM, dtype=torch.int8)
    values[left] = 90
    values[right] = 90
    return LayerContextualContinuationKey(layer, token, values)


def _transaction(
    identity: LayerContextualContinuationIdentity,
    axes: dict[int, int],
    *,
    token: int,
    boundary: int,
    nonce: int,
) -> LayerContextualContinuationTransaction:
    return LayerContextualContinuationTransaction.create(
        identity_sha256=identity.identity_sha256,
        boundary_index=boundary,
        known_token=token,
        keys=tuple(_axis_key(layer, token, axes[layer]) for layer in identity.layers),
        transaction_nonce=f"{nonce:032x}",
    )


class LayerContextualContinuationBankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / "layer-continuations.json"
        self.identity = _identity(layers=(3, 1))

    def test_identity_projection_and_transaction_are_fully_bound(self) -> None:
        self.assertEqual(self.identity.layers, (1, 3))
        self.assertEqual(self.identity.stage, LAYER_CONTEXTUAL_STAGE)
        self.assertEqual(self.identity.sketch_dim, DEFAULT_LAYER_CONTEXTUAL_SKETCH_DIM)
        bank = LayerContextualContinuationBank(self.path, self.identity, max_cells=8)
        transaction = bank.capture_boundaries(
            {3: _hidden(3.0), 1: _hidden(1.0)},
            known_token=41,
            boundary_index=1234,
            transaction_nonce="a" * 32,
        )

        self.assertEqual(transaction.layers, (1, 3))
        self.assertEqual(transaction.known_token, 41)
        self.assertEqual(transaction.boundary_index, 1234)
        for key in transaction.keys:
            self.assertEqual(set(key.to_record()), {"layer", "known_token", "q8"})
            self.assertEqual(key.q8.dtype, torch.int8)
            self.assertEqual(key.q8.device.type, "cpu")
            self.assertTrue(key.q8.is_contiguous())
            self.assertEqual(key.q8.numel(), DEFAULT_LAYER_CONTEXTUAL_SKETCH_DIM)

        changed = LayerContextualContinuationBank(
            self.root / "changed.json",
            _identity(runtime="changed", layers=(1, 3)),
            max_cells=8,
        )
        with self.assertRaises(LayerContextualContinuationIdentityError):
            changed.query_options(transaction)

    def test_queries_are_isolated_by_layer_and_known_token(self) -> None:
        bank = LayerContextualContinuationBank(self.path, self.identity, max_cells=16)
        first = _transaction(
            self.identity,
            {1: 0, 3: 1},
            token=7,
            boundary=0,
            nonce=1,
        )
        second = _transaction(
            self.identity,
            {1: 1, 3: 0},
            token=7,
            boundary=1,
            nonce=2,
        )
        bank.settle_verified_prefix(first, (200,))
        bank.settle_verified_prefix(second, (100,))

        query = _transaction(
            self.identity,
            {1: 0, 3: 0},
            token=7,
            boundary=2,
            nonce=3,
        )
        options = bank.query_options(query)
        self.assertEqual(
            [(option.layer, option.target_tail) for option in options],
            [(1, (200,)), (3, (100,))],
        )

        wrong_token = _transaction(
            self.identity,
            {1: 0, 3: 0},
            token=8,
            boundary=2,
            nonce=4,
        )
        self.assertEqual(bank.query_options(wrong_token), ())

    def test_query_options_is_file_read_only_with_process_local_metrics(self) -> None:
        bank = LayerContextualContinuationBank(self.path, self.identity, max_cells=8)
        captured = _transaction(
            self.identity,
            {1: 0, 3: 1},
            token=7,
            boundary=0,
            nonce=5,
        )
        bank.settle_verified_prefix(captured, (20, 21))
        query = _transaction(
            self.identity,
            {1: 0, 3: 1},
            token=7,
            boundary=1,
            nonce=6,
        )
        before_bytes = self.path.read_bytes()
        before_mtime = self.path.stat().st_mtime_ns
        before_metrics = bank.metrics()

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(
                executor.map(lambda _index: bank.query_options(query), range(16))
            )

        self.assertTrue(all(len(options) == 2 for options in results))
        self.assertEqual(self.path.read_bytes(), before_bytes)
        self.assertEqual(self.path.stat().st_mtime_ns, before_mtime)
        metrics = bank.metrics()
        self.assertEqual(metrics.state_sha256, before_metrics.state_sha256)
        self.assertEqual(metrics.clock, before_metrics.clock)
        self.assertEqual(metrics.crystal_queries, 32)
        self.assertEqual(metrics.persisted_crystal_queries, 0)
        self.assertEqual(metrics.process_crystal_queries_delta, 32)
        self.assertEqual(metrics.layers["1"].crystal_queries, 16)
        self.assertEqual(
            metrics.query_metrics_scope,
            "persisted_plus_process_local_delta",
        )

        reopened = LayerContextualContinuationBank(
            self.path,
            self.identity,
            max_cells=8,
        )
        self.assertEqual(reopened.metrics().crystal_queries, 0)
        with self.assertRaises(ValueError):
            bank.query_options(query, limit=0)
        self.assertEqual(self.path.read_bytes(), before_bytes)
        self.assertEqual(self.path.stat().st_mtime_ns, before_mtime)

    def test_exact_nearest_and_cosine_ties_have_canonical_order(self) -> None:
        identity = _identity(layers=(2,))
        bank = LayerContextualContinuationBank(self.path, identity, max_cells=8)
        first = _transaction(
            identity,
            {2: 0},
            token=9,
            boundary=0,
            nonce=10,
        )
        second = _transaction(
            identity,
            {2: 1},
            token=9,
            boundary=1,
            nonce=11,
        )
        bank.settle_verified_prefix(first, (20,))
        bank.settle_verified_prefix(second, (10,))

        exact = bank.query_options(
            _transaction(
                identity,
                {2: 0},
                token=9,
                boundary=2,
                nonce=12,
            ),
            limit=2,
        )
        self.assertEqual(exact[0].target_tail, (20,))
        self.assertAlmostEqual(exact[0].cosine, 1.0)

        tie_key = _tie_key(2, 9, 0, 1)
        tie = LayerContextualContinuationTransaction.create(
            identity_sha256=identity.identity_sha256,
            boundary_index=3,
            known_token=9,
            keys=(tie_key,),
            transaction_nonce="d" * 32,
        )
        tied = bank.query_options(tie, limit=2)
        self.assertEqual([option.target_tail for option in tied], [(10,), (20,)])
        self.assertAlmostEqual(tied[0].cosine, tied[1].cosine)
        self.assertAlmostEqual(tied[0].margin or 0.0, 0.0)

    def test_capture_and_feedback_publish_atomically_and_settle_idempotently(
        self,
    ) -> None:
        bank = LayerContextualContinuationBank(self.path, self.identity, max_cells=8)
        initial = _transaction(
            self.identity,
            {1: 0, 3: 1},
            token=7,
            boundary=0,
            nonce=20,
        )
        bank.settle_verified_prefix(initial, (11, 12, 13))
        current = _transaction(
            self.identity,
            {1: 0, 3: 1},
            token=7,
            boundary=1,
            nonce=21,
        )
        options = bank.query_options(current)
        before_bytes = self.path.read_bytes()
        before_metrics = bank.metrics()

        with (
            patch.object(
                layer_bank,
                "_atomic_write",
                side_effect=LayerContextualContinuationIntegrityError(
                    "publication failed"
                ),
            ),
            self.assertRaisesRegex(
                LayerContextualContinuationIntegrityError, "publication failed"
            ),
        ):
            bank.settle_verified_prefix(
                current,
                (11, 99, 100),
                options=options,
            )
        self.assertEqual(self.path.read_bytes(), before_bytes)
        self.assertEqual(bank.metrics(), before_metrics)

        metrics = bank.settle_verified_prefix(
            current,
            (11, 99, 100),
            options=options,
        )
        settled_bytes = self.path.read_bytes()
        repeated = bank.settle_verified_prefix(
            current,
            (11, 99, 100),
            options=options,
        )
        self.assertEqual(self.path.read_bytes(), settled_bytes)
        self.assertEqual(repeated, metrics)
        self.assertEqual(metrics.settlements, 2)
        self.assertEqual(metrics.crystal_bank_cells, 4)
        self.assertEqual(metrics.crystal_bank_support, 4)
        self.assertEqual(metrics.crystal_captures, 4)
        self.assertEqual(metrics.crystal_verified_tokens, 6)
        self.assertEqual(metrics.crystal_accepted_tokens, 2)
        self.assertEqual(metrics.crystal_mismatches, 2)
        self.assertEqual(metrics.layers["1"].crystal_captures, 2)
        self.assertEqual(metrics.layers["3"].crystal_verified_tokens, 3)

        next_query = _transaction(
            self.identity,
            {1: 0, 3: 1},
            token=7,
            boundary=2,
            nonce=22,
        )
        learned = bank.query_options(next_query, limit=2)
        original = [option for option in learned if option.target_tail == (11, 12, 13)]
        self.assertEqual(len(original), 2)
        for option in original:
            self.assertEqual(option.position_verified, (1, 1, 1))
            self.assertEqual(option.position_hits, (1, 0, 0))

        with self.assertRaises(LayerContextualContinuationConflictError):
            bank.settle_verified_prefix(current, (11, 12, 13), options=options)

    def test_reopen_hash_tamper_identity_symlink_and_size_fail_closed(self) -> None:
        bank = LayerContextualContinuationBank(self.path, self.identity, max_cells=8)
        transaction = _transaction(
            self.identity,
            {1: 0, 3: 1},
            token=7,
            boundary=0,
            nonce=30,
        )
        bank.settle_verified_prefix(transaction, (2, 3))
        original_bytes = self.path.read_bytes()
        restored = LayerContextualContinuationBank(
            self.path,
            self.identity,
            max_cells=8,
        )
        self.assertEqual(restored.metrics().crystal_bank_cells, 2)
        self.assertEqual(
            LayerContextualContinuationBank.read_identity(self.path), self.identity
        )
        with self.assertRaises(LayerContextualContinuationIdentityError):
            LayerContextualContinuationBank(
                self.path,
                _identity(runtime="foreign", layers=(1, 3)),
                max_cells=8,
            )

        document = json.loads(self.path.read_bytes())
        document["body"]["clock"] += 1
        self.path.write_bytes(_canonical(document))
        with self.assertRaisesRegex(
            LayerContextualContinuationIntegrityError,
            "SHA-256 mismatch",
        ):
            LayerContextualContinuationBank(self.path, self.identity, max_cells=8)

        document = json.loads(original_bytes)
        document["body"]["cells"][0]["key"]["q8"] = [
            0
        ] * DEFAULT_LAYER_CONTEXTUAL_SKETCH_DIM
        document["body_sha256"] = hashlib.sha256(
            _canonical(document["body"])
        ).hexdigest()
        self.path.write_bytes(_canonical(document))
        with self.assertRaisesRegex(
            LayerContextualContinuationIntegrityError,
            "key values",
        ):
            LayerContextualContinuationBank(self.path, self.identity, max_cells=8)

        linked_path = self.root / "linked.json"
        target = self.root / "target.json"
        target.write_bytes(b"{}")
        linked_path.symlink_to(target)
        with self.assertRaises(LayerContextualContinuationIntegrityError):
            LayerContextualContinuationBank(linked_path, self.identity, max_cells=8)

        oversized = self.root / "oversized.json"
        with oversized.open("wb") as stream:
            stream.truncate(64 * 1024 * 1024 + 1)
        with self.assertRaisesRegex(
            LayerContextualContinuationIntegrityError,
            "byte limit",
        ):
            LayerContextualContinuationBank(oversized, self.identity, max_cells=8)

        locked_path = self.root / "locked.json"
        lock_target = self.root / "lock-target"
        lock_target.write_bytes(b"")
        (self.root / ".locked.json.lock").symlink_to(lock_target)
        locked_bank = LayerContextualContinuationBank(
            locked_path, self.identity, max_cells=8
        )
        with self.assertRaises(LayerContextualContinuationIntegrityError):
            locked_bank.settle_verified_prefix(transaction, (2, 3))

    def test_cell_capacity_does_not_evict_independent_receipts(self) -> None:
        bank = LayerContextualContinuationBank(
            self.path,
            self.identity,
            max_cells=3,
            max_receipts=16,
        )
        latest = None
        for index in range(10):
            latest = _transaction(
                self.identity,
                {1: index, 3: index + 16},
                token=7,
                boundary=index,
                nonce=200 + index,
            )
            bank.settle_verified_prefix(latest, (100 + index,))
        assert latest is not None
        before = self.path.read_bytes()
        metrics = bank.settle_verified_prefix(latest, (109,))

        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(metrics.crystal_bank_cells, 3)
        self.assertEqual(metrics.receipt_count, 10)
        self.assertEqual(metrics.max_receipts, 16)
        self.assertEqual(metrics.settlements, 10)
        self.assertEqual(metrics.crystal_captures, 20)
        self.assertEqual(metrics.evictions, 17)

    def test_receipt_capacity_rejects_new_work_but_preserves_old_retries(
        self,
    ) -> None:
        self.assertEqual(DEFAULT_LAYER_CONTEXTUAL_MAX_RECEIPTS, 65_536)
        bank = LayerContextualContinuationBank(
            self.path,
            self.identity,
            max_cells=1,
            max_receipts=2,
        )
        first = _transaction(
            self.identity,
            {1: 0, 3: 1},
            token=7,
            boundary=0,
            nonce=300,
        )
        second = _transaction(
            self.identity,
            {1: 2, 3: 3},
            token=7,
            boundary=1,
            nonce=301,
        )
        third = _transaction(
            self.identity,
            {1: 4, 3: 5},
            token=7,
            boundary=2,
            nonce=302,
        )
        bank.settle_verified_prefix(first, (10,))
        bank.settle_verified_prefix(second, (20,))
        before_bytes = self.path.read_bytes()
        before_metrics = bank.metrics()

        retried = bank.settle_verified_prefix(first, (10,))
        self.assertEqual(retried, before_metrics)
        self.assertEqual(self.path.read_bytes(), before_bytes)
        with self.assertRaisesRegex(
            LayerContextualContinuationCapacityError,
            "receipt capacity",
        ):
            bank.settle_verified_prefix(third, (30,))
        self.assertEqual(bank.metrics(), before_metrics)
        self.assertEqual(self.path.read_bytes(), before_bytes)
        self.assertEqual(bank.settle_verified_prefix(second, (20,)), before_metrics)

        reopened = LayerContextualContinuationBank(
            self.path,
            self.identity,
            max_cells=1,
            max_receipts=2,
        )
        self.assertEqual(reopened.metrics().receipt_count, 2)
        self.assertEqual(reopened.settle_verified_prefix(first, (10,)).settlements, 2)

    def test_concurrent_instances_lose_no_updates_and_deduplicate_one_receipt(
        self,
    ) -> None:
        banks = [
            LayerContextualContinuationBank(self.path, self.identity, max_cells=32)
            for _ in range(8)
        ]
        transactions = [
            _transaction(
                self.identity,
                {1: index, 3: index + 16},
                token=7,
                boundary=index,
                nonce=100 + index,
            )
            for index in range(8)
        ]

        def settle(index: int) -> None:
            banks[index].settle_verified_prefix(
                transactions[index],
                (1000 + index,),
            )

        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(settle, range(8)))

        restored = LayerContextualContinuationBank(
            self.path, self.identity, max_cells=32
        )
        metrics = restored.metrics()
        self.assertEqual(metrics.settlements, 8)
        self.assertEqual(metrics.crystal_captures, 16)
        self.assertEqual(metrics.crystal_bank_cells, 16)
        self.assertEqual(metrics.layers["1"].crystal_bank_cells, 8)
        self.assertEqual(metrics.layers["3"].crystal_bank_cells, 8)

        shared = _transaction(
            self.identity,
            {1: 24, 3: 25},
            token=7,
            boundary=99,
            nonce=999,
        )
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(
                executor.map(
                    lambda bank_instance: bank_instance.settle_verified_prefix(
                        shared, (9999,)
                    ),
                    banks,
                )
            )
        final = restored.refresh()
        self.assertEqual(final.settlements, 9)
        self.assertEqual(final.crystal_captures, 18)
        self.assertEqual(final.crystal_bank_cells, 18)

    def test_persistence_contains_no_hidden_state_prompt_or_boundary_payload(
        self,
    ) -> None:
        bank = LayerContextualContinuationBank(self.path, self.identity, max_cells=8)
        hidden_by_layer = {1: _hidden(0.125), 3: _hidden(8.5)}
        transaction = bank.capture_boundaries(
            hidden_by_layer,
            known_token=7,
            boundary_index=987654321,
            transaction_nonce="f" * 32,
        )
        bank.settle_verified_prefix(transaction, (11, 12, 13))

        encoded = self.path.read_bytes()
        for hidden in hidden_by_layer.values():
            self.assertNotIn(hidden.numpy().tobytes(), encoded)
        self.assertNotIn(b"prompt", encoded.lower())
        self.assertNotIn(b"raw_hidden", encoded.lower())
        self.assertNotIn(b"hidden_state", encoded.lower())
        self.assertNotIn(b"987654321", encoded)
        self.assertNotIn(transaction.transaction_nonce.encode("ascii"), encoded)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

        document = json.loads(encoded)
        key_records = [cell["key"] for cell in document["body"]["cells"]]
        self.assertTrue(key_records)
        self.assertTrue(
            all(set(record) == {"layer", "known_token", "q8"} for record in key_records)
        )


if __name__ == "__main__":
    unittest.main()
