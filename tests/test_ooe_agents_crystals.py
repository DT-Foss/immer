from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np

from immer.runtimes.ooe.agents import (
    AttractorRouter,
    MarkovPDAgent,
    MobileMarkovToken,
)
from immer.runtimes.ooe.crystal import (
    CrystalCollisionError,
    CrystalIdentityError,
    CrystalPayload,
    CrystalStore,
    CrystalStoreError,
    CrystalTamperError,
    ManifestConflictError,
    MAX_KERNEL_DIMENSION,
)
from immer.runtimes.ooe.identity import OoeSiteIdentity, canonical_json_bytes
from immer.runtimes.ooe.math_core import RapidityLedger


def _sha(character: str) -> str:
    return character * 64


def _identity(*, coordinate: str = "2") -> OoeSiteIdentity:
    return OoeSiteIdentity(
        model_pin_sha256=_sha("1"),
        weight_coordinate_sha256=_sha(coordinate),
        graph_revision_sha256=_sha("3"),
        feature_schema_sha256=_sha("4"),
        action_schema_sha256=_sha("5"),
    )


def _payload(
    *,
    name: str = "layer.12/router/action-7",
    identity: OoeSiteIdentity | None = None,
    evidence: str = "9",
    kernel: np.ndarray | None = None,
) -> CrystalPayload:
    if kernel is None:
        kernel = np.asarray(
            [
                [0.05, 0.90, 0.05],
                [0.10, 0.15, 0.75],
                [0.80, 0.10, 0.10],
            ],
            dtype=np.float64,
        )
    return CrystalPayload.from_kernel(
        name=name,
        identity=_identity() if identity is None else identity,
        kernel=kernel,
        coverage_sha256=_sha("6"),
        calibration_sha256=_sha("7"),
        verifier_hashes={"atlas": _sha("8"), "fertig": _sha("a")},
        evidence_hashes={"holdout": _sha(evidence)},
        consensus_receipt={
            "format": "test-consensus/v1",
            "quorum_sha256": _sha("b"),
            "rounds": 14,
        },
    )


class SiteIdentityTests(unittest.TestCase):
    def test_identity_roundtrip_and_every_binding_changes_address(self) -> None:
        identity = _identity()
        self.assertEqual(OoeSiteIdentity.from_dict(identity.to_dict()), identity)
        self.assertEqual(len(identity.sha256), 64)
        for field, character in (
            ("model_pin_sha256", "c"),
            ("weight_coordinate_sha256", "d"),
            ("graph_revision_sha256", "e"),
            ("feature_schema_sha256", "f"),
            ("action_schema_sha256", "0"),
        ):
            changed = identity.to_dict()
            changed[field] = _sha(character)
            self.assertNotEqual(
                OoeSiteIdentity.from_dict(changed).sha256, identity.sha256
            )

    def test_identity_rejects_unpinned_or_noncanonical_hashes(self) -> None:
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            OoeSiteIdentity(
                model_pin_sha256="A" * 64,
                weight_coordinate_sha256=_sha("2"),
                graph_revision_sha256=_sha("3"),
                feature_schema_sha256=_sha("4"),
                action_schema_sha256=_sha("5"),
            )
        value = _identity().to_dict()
        value["surprise"] = _sha("6")
        with self.assertRaisesRegex(ValueError, "unknown or missing"):
            OoeSiteIdentity.from_dict(value)


class AgentTests(unittest.TestCase):
    def test_weighted_general_kernel_and_active_state_tracking(self) -> None:
        agent = MarkovPDAgent(state_size=4, action_size=3, prior=0.1)
        agent.observe(1, 2, weight=3.0)
        agent.observe_distribution(2, np.array([0.25, 0.25, 0.50]), total_weight=2.0)
        self.assertEqual(agent.active_states(), (1, 2))
        self.assertEqual(agent.active_states(min_mass=3.0), (1,))
        self.assertEqual(agent.observations, 2)
        self.assertAlmostEqual(agent.observation_mass, 5.0)
        prediction = agent.predict(np.array([0.0, 1.0, 0.0, 0.0]))
        self.assertEqual(prediction.shape, (3,))
        self.assertEqual(int(prediction.argmax()), 2)
        self.assertAlmostEqual(float(prediction.sum()), 1.0)

    def test_weight_and_state_validation_fail_closed(self) -> None:
        agent = MarkovPDAgent(3)
        for weight in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                agent.observe(0, 1, weight=weight)
        with self.assertRaises(IndexError):
            agent.observe(3, 0)
        with self.assertRaises(IndexError):
            agent.observe(0, 3)
        with self.assertRaises(TypeError):
            agent.observe(True, 0)
        with self.assertRaises(ValueError):
            MarkovPDAgent(3.5)

    def test_merge_moves_evidence_not_the_other_agents_prior(self) -> None:
        left = MarkovPDAgent(3, prior=0.2)
        right = MarkovPDAgent(3, prior=0.7)
        right.observe(0, 2, weight=4.0)
        left.merge_from(right, weight=0.5)
        self.assertAlmostEqual(left.decision_counts[0, 2], 2.2)
        self.assertAlmostEqual(left.observation_mass, 2.0)
        self.assertEqual(left.observations, 1)
        self.assertEqual(left.event_count, 1)
        self.assertAlmostEqual(left.evidence_mass, 2.0)
        self.assertEqual(left.active_states(), (0,))
        self.assertTrue(np.allclose(left.decision_kernel().sum(axis=1), 1.0))

    def test_router_requires_radius_and_nearest_margin(self) -> None:
        router = AttractorRouter(radius=2.0, min_margin=0.20)
        for sketch in (np.array([-1.0, 0.0]), np.array([-0.9, 0.0])):
            router.observe("left", sketch)
        for sketch in (np.array([1.0, 0.0]), np.array([0.9, 0.0])):
            router.observe("right", sketch)

        self.assertEqual(router.classify(np.array([-0.95, 0.01])), "left")
        ambiguous = router.decision(np.array([0.0, 0.0]))
        self.assertFalse(ambiguous.accepted)
        self.assertEqual(ambiguous.reason, "ambiguous-margin")

        strict = AttractorRouter(radius=0.1, min_margin=0.01)
        strict.observe("left", np.array([-1.0, 0.0]))
        strict.observe("right", np.array([1.0, 0.0]))
        outside = strict.decision(np.array([-0.5, 0.0]))
        self.assertFalse(outside.accepted)
        self.assertEqual(outside.reason, "outside-radius")

    def test_router_calibration_is_evidence_hashed(self) -> None:
        router = AttractorRouter(radius=5.0)
        router.observe("a", np.array([-1.0, 0.0]))
        router.observe("b", np.array([1.0, 0.0]))
        digest = router.calibrate(
            (
                ("a", np.array([-1.02, 0.01])),
                ("a", np.array([-0.98, -0.01])),
                ("b", np.array([1.02, 0.01])),
                ("b", np.array([0.98, -0.01])),
            ),
            radius_quantile=1.0,
            margin_quantile=0.0,
        )
        self.assertEqual(router.calibration_sha256, digest)
        self.assertEqual(router.classify(np.array([-1.0, 0.0])), "a")
        self.assertIsNone(router.classify(np.array([0.0, 0.0])))
        router.observe("a", np.array([-1.0, 0.01]))
        self.assertIsNone(router.calibration_sha256)

    def test_calibration_hash_binds_complete_centroid_state(self) -> None:
        samples = (
            ("a", np.array([-0.9, 0.0])),
            ("b", np.array([0.9, 0.0])),
        )
        near = AttractorRouter(radius=5.0)
        near.observe("a", np.array([-1.0, 0.0]))
        near.observe("b", np.array([1.0, 0.0]))
        far = AttractorRouter(radius=5.0)
        far.observe("a", np.array([-2.0, 0.0]))
        far.observe("b", np.array([2.0, 0.0]))
        reverse_insertion = AttractorRouter(radius=5.0)
        reverse_insertion.observe("b", np.array([1.0, 0.0]))
        reverse_insertion.observe("a", np.array([-1.0, 0.0]))
        near_digest = near.calibrate(samples)
        far_digest = far.calibrate(samples)
        reverse_digest = reverse_insertion.calibrate(samples)
        self.assertNotEqual(near_digest, far_digest)
        self.assertEqual(near_digest, reverse_digest)

    def test_mobile_token_memory_and_rapidity_change_executable_gate(self) -> None:
        token = MobileMarkovToken.from_state(0, state_size=3, reservoir_size=12)
        confidence = token.update(
            np.array([0.01, 0.98, 0.01]),
            np.linspace(-1.0, 1.0, 24),
            "site-a",
        )
        self.assertGreater(confidence, 0.0)
        self.assertTrue(token.gate(min_confidence=confidence * 0.99))

        no_memory = MobileMarkovToken(
            belief=token.belief.copy(),
            reservoir=np.zeros_like(token.reservoir),
            ledger=RapidityLedger(token.ledger.xi, token.ledger.limit),
            _reservoir_coherence=token._reservoir_coherence,
        )
        no_rapidity = MobileMarkovToken(
            belief=token.belief.copy(),
            reservoir=token.reservoir.copy(),
            ledger=RapidityLedger(),
            _reservoir_coherence=token._reservoir_coherence,
        )
        self.assertEqual(no_memory.execution_confidence, 0.0)
        self.assertEqual(no_rapidity.execution_confidence, 0.0)
        self.assertFalse(no_memory.gate(min_confidence=0.01))
        self.assertEqual(no_memory.fallbacks, 1)

    def test_mobile_token_branch_mass_and_state_are_preserved(self) -> None:
        token = MobileMarkovToken.from_state(0, state_size=4)
        token.update(np.array([0.05, 0.55, 0.30, 0.10]), np.arange(12), "x")
        children = token.spawn(top_k=2)
        self.assertEqual(len(children), 2)
        self.assertAlmostEqual(sum(child.branch_mass for child in children), 0.85)
        self.assertEqual({int(child.belief.argmax()) for child in children}, {1, 2})
        self.assertTrue(all(child.route == ["x"] for child in children))
        self.assertTrue(
            all(np.array_equal(child.reservoir, token.reservoir) for child in children)
        )


class CrystalPayloadTests(unittest.TestCase):
    def test_quantized_payload_binds_every_execution_artifact(self) -> None:
        payload = _payload()
        restored = payload.restore_kernel()
        self.assertTrue(np.all(restored.sum(axis=1) == 1.0))
        self.assertListEqual(restored.argmax(axis=1).tolist(), [1, 2, 0])
        self.assertEqual(payload.identity, _identity())
        self.assertEqual(payload.coverage_sha256, _sha("6"))
        self.assertEqual(payload.calibration_sha256, _sha("7"))
        self.assertEqual(set(dict(payload.verifier_hashes)), {"atlas", "fertig"})
        self.assertEqual(payload.consensus_receipt["rounds"], 14)
        self.assertEqual(len(payload.consensus_receipt_sha256), 64)
        self.assertEqual(len(payload.verifier_sha256), 64)
        self.assertEqual(len(payload.evidence_sha256), 64)

    def test_payload_is_canonical_and_roundtrips_exactly(self) -> None:
        payload = _payload()
        encoded = payload.to_bytes()
        restored = CrystalPayload.from_bytes(encoded)
        self.assertEqual(restored, payload)
        self.assertEqual(restored.sha256, hashlib.sha256(encoded).hexdigest())
        with self.assertRaises(CrystalTamperError):
            CrystalPayload.from_bytes(encoded + b"\n")

        value = json.loads(encoded)
        value["identity"]["model_pin_sha256"] = _sha("c")
        rebound = CrystalPayload.from_bytes(canonical_json_bytes(value))
        self.assertNotEqual(rebound.sha256, payload.sha256)
        self.assertNotEqual(rebound.identity.sha256, payload.identity.sha256)

    def test_payload_rejects_missing_verifier_or_invalid_kernel(self) -> None:
        kwargs = {
            "name": "bad",
            "identity": _identity(),
            "kernel": np.eye(2),
            "coverage_sha256": _sha("6"),
            "calibration_sha256": _sha("7"),
            "evidence_hashes": {"e": _sha("8")},
            "consensus_receipt": {"ok": True},
        }
        with self.assertRaises(ValueError):
            CrystalPayload.from_kernel(verifier_hashes={}, **kwargs)
        with self.assertRaises(ValueError):
            CrystalPayload.from_kernel(
                verifier_hashes={"v": _sha("9")},
                **{**kwargs, "kernel": np.array([[0.0, 0.0], [0.5, 0.5]])},
            )


class CrystalStoreTests(unittest.TestCase):
    def test_content_addressed_roundtrip_permissions_and_manifest_cas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "bank")
            payload = _payload()
            publication = store.publish(payload, expected_generation=0)
            self.assertTrue(publication.object_created)
            self.assertTrue(publication.manifest_changed)
            self.assertEqual(publication.generation, 1)
            self.assertEqual(store.restore(publication.payload_sha256), payload)
            self.assertEqual(store.restore_named(payload.name), payload)

            object_path = store.root / "objects" / f"{payload.sha256}.crystal"
            self.assertEqual(stat.S_IMODE(object_path.stat().st_mode), 0o444)
            same = store.publish(payload, expected_generation=1)
            self.assertFalse(same.object_created)
            self.assertFalse(same.manifest_changed)
            self.assertEqual(same.generation, 1)
            self.assertTrue(store.audit().clean)

            with self.assertRaises(ManifestConflictError):
                store.publish(payload, expected_generation=0)

    def test_same_site_can_promote_but_name_cannot_rebind_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "bank")
            first = _payload()
            store.publish(first, expected_generation=0)
            promoted = _payload(evidence="c")
            publication = store.publish(promoted, expected_generation=1)
            self.assertEqual(publication.generation, 2)
            self.assertEqual(store.restore_named(first.name), promoted)
            self.assertEqual(len(store.manifest().objects), 2)

            rebound = _payload(identity=_identity(coordinate="d"), evidence="e")
            with self.assertRaises(CrystalIdentityError):
                store.publish(rebound, expected_generation=2)
            self.assertTrue(store.audit().clean)

    def test_named_restore_is_one_locked_manifest_object_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "bank")
            first = _payload()
            promoted = _payload(evidence="c")
            store.publish(first, expected_generation=0)
            manifest_read = threading.Event()
            release_reader = threading.Event()
            writer_started = threading.Event()
            writer_done = threading.Event()
            results: dict[str, object] = {}
            original_read = CrystalStore._stable_read

            def blocking_read(
                dir_fd: int, name: str, *, max_bytes: int | None = None
            ) -> bytes:
                data = original_read(dir_fd, name, max_bytes=max_bytes)
                if (
                    name == "manifest.json"
                    and threading.current_thread().name == "named-reader"
                ):
                    manifest_read.set()
                    if not release_reader.wait(2.0):
                        raise TimeoutError("test reader was not released")
                return data

            def read_named() -> None:
                try:
                    results["read"] = store.restore_named(first.name)
                except BaseException as exc:  # pragma: no cover - assertion carrier
                    results["read_error"] = exc

            def publish_next() -> None:
                writer_started.set()
                try:
                    results["write"] = store.publish(promoted, expected_generation=1)
                except BaseException as exc:  # pragma: no cover - assertion carrier
                    results["write_error"] = exc
                finally:
                    writer_done.set()

            with mock.patch.object(
                CrystalStore, "_stable_read", staticmethod(blocking_read)
            ):
                reader = threading.Thread(target=read_named, name="named-reader")
                reader.start()
                self.assertTrue(manifest_read.wait(1.0))
                writer = threading.Thread(target=publish_next, name="publisher")
                writer.start()
                self.assertTrue(writer_started.wait(1.0))
                time.sleep(0.05)
                self.assertFalse(writer_done.is_set())
                release_reader.set()
                reader.join(2.0)
                writer.join(2.0)

            self.assertFalse(reader.is_alive())
            self.assertFalse(writer.is_alive())
            self.assertNotIn("read_error", results)
            self.assertNotIn("write_error", results)
            self.assertEqual(results["read"], first)
            self.assertEqual(store.manifest().generation, 2)
            self.assertEqual(store.restore_named(first.name), promoted)

    def test_restore_detects_payload_tamper_and_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "bank")
            payload = _payload()
            publication = store.publish(payload)
            object_path = store.root / "objects" / f"{payload.sha256}.crystal"
            os.chmod(object_path, 0o644)
            corrupted = bytearray(object_path.read_bytes())
            corrupted[len(corrupted) // 2] ^= 1
            object_path.write_bytes(corrupted)
            with self.assertRaises(CrystalTamperError):
                store.restore(publication.payload_sha256)
            audit = store.audit()
            self.assertIn(f"{payload.sha256}.crystal", audit.tampered_objects)

        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "bank")
            payload = _payload()
            collision = store.root / "objects" / f"{payload.sha256}.crystal"
            collision.write_bytes(b"different bytes")
            with self.assertRaises(CrystalCollisionError):
                store.publish(payload)

    def test_symlinked_store_and_object_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "real"
            target.mkdir()
            alias = Path(temporary) / "alias"
            alias.symlink_to(target, target_is_directory=True)
            with self.assertRaises(CrystalStoreError):
                CrystalStore(alias)

        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "bank")
            payload = _payload()
            digest = payload.sha256
            target = Path(temporary) / "outside"
            target.write_bytes(payload.to_bytes())
            (store.root / "objects" / f"{digest}.crystal").symlink_to(target)
            with self.assertRaises(OSError):
                store.restore(digest)
            self.assertIn(f"{digest}.crystal", store.audit().tampered_objects)

    def test_orphan_and_crash_staging_reconciliation_are_cas_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "bank")
            orphan = _payload(name="orphan")
            object_path = store.root / "objects" / f"{orphan.sha256}.crystal"
            object_path.write_bytes(orphan.to_bytes())
            os.chmod(object_path, 0o444)
            staged = store.root / "staging" / ".crash.tmp"
            staged.write_bytes(b"partial")

            audit = store.audit()
            self.assertEqual(audit.orphan_objects, (orphan.sha256,))
            self.assertEqual(audit.staged_files, (".crash.tmp",))
            with self.assertRaises(ManifestConflictError):
                store.remove_orphans(expected_generation=1)
            self.assertEqual(
                store.remove_orphans(expected_generation=0), (orphan.sha256,)
            )
            self.assertEqual(
                store.clean_staging(expected_generation=0), (".crash.tmp",)
            )
            self.assertTrue(store.audit().clean)

    def test_oversized_files_and_kernel_descriptors_fail_before_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "bank")
            oversized = b"x" * 129
            digest = hashlib.sha256(oversized).hexdigest()
            object_path = store.root / "objects" / f"{digest}.crystal"
            object_path.write_bytes(oversized)
            with mock.patch(
                "immer.runtimes.ooe.crystal.MAX_CRYSTAL_PAYLOAD_BYTES", 128
            ):
                with self.assertRaisesRegex(CrystalTamperError, "file-size limit"):
                    store.restore(digest)
                with self.assertRaisesRegex(
                    CrystalTamperError, "MAX_CRYSTAL_PAYLOAD_BYTES"
                ):
                    CrystalPayload.from_bytes(oversized)
                self.assertEqual(store.audit().tampered_objects, (f"{digest}.crystal",))

        value = json.loads(_payload().to_bytes())
        value["kernel"]["rows"] = MAX_KERNEL_DIMENSION + 1
        with mock.patch(
            "immer.runtimes.ooe.crystal.base64.b64decode",
            side_effect=AssertionError("decoder must not run"),
        ):
            with self.assertRaisesRegex(CrystalTamperError, "dimensions"):
                CrystalPayload.from_bytes(canonical_json_bytes(value))

        value = json.loads(_payload().to_bytes())
        with (
            mock.patch("immer.runtimes.ooe.crystal.MAX_KERNEL_DECODED_BYTES", 4),
            mock.patch(
                "immer.runtimes.ooe.crystal.base64.b64decode",
                side_effect=AssertionError("decoder must not run"),
            ),
        ):
            with self.assertRaisesRegex(CrystalTamperError, "dimensions"):
                CrystalPayload.from_bytes(canonical_json_bytes(value))

    def test_hash_sealed_controller_state_is_atomic_bounded_and_cas_checked(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = CrystalStore(Path(temporary) / "bank", max_state_bytes=8)
            first = store.publish_state("controller", b"abc")
            self.assertEqual(first.generation, 1)
            self.assertTrue(first.changed)
            self.assertEqual(store.restore_state("controller"), b"abc")
            same = store.publish_state("controller", b"abc")
            self.assertFalse(same.changed)
            self.assertEqual(same.generation, 1)
            with self.assertRaises(ManifestConflictError):
                store.publish_state("controller", b"next", expected_sha256=_sha("f"))
            second = store.publish_state(
                "controller", b"next", expected_sha256=first.payload_sha256
            )
            self.assertEqual(second.generation, 2)
            self.assertEqual(store.restore_state("controller"), b"next")
            with self.assertRaises(ValueError):
                store.publish_state("controller", b"123456789")

            filename = hashlib.sha256(b"controller").hexdigest() + ".state"
            state_path = store.root / "state" / filename
            os.chmod(state_path, 0o644)
            corrupted = state_path.read_bytes().replace(b"bmV4dA==", b"bmV4eA==")
            state_path.write_bytes(corrupted)
            with self.assertRaises(CrystalTamperError):
                store.restore_state("controller")


if __name__ == "__main__":
    unittest.main()
