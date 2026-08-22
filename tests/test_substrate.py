from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from immer.capabilities.organbank import OrganBank
from immer.contracts import Component, ExecutionStatus, Request, Result
from immer.substrate import EventBus, LifeDaemon, LifeStatePort, Priority


class RecordingStream:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def observe(self, text: str) -> None:
        self.seen.append(text)

    def snapshot(self):
        return {"seen": list(self.seen)}

    def restore(self, state) -> None:
        self.seen = list(state.get("seen", ()))


class ExactService:
    name = "test.exact"
    capabilities = frozenset({"exact_math"})

    def handle(self, request: Request) -> Result:
        if request.payload == "1+1":
            return Result(ExecutionStatus.OK, self.name, output=2)
        return Result(ExecutionStatus.ABSTAINED, self.name, reason="weiß ich nicht")


class MetadataService:
    name = "test.metadata"
    capabilities = frozenset({"metadata"})

    def handle(self, request: Request) -> Result:
        return Result(ExecutionStatus.OK, self.name, output=request.metadata.get("trace_id"))


def _make_bank(root: Path) -> OrganBank:
    artifact = root / "arith.organ"
    artifact.write_bytes(b"exact-organ")
    manifest = root / "bank.json"
    manifest.write_text(json.dumps({"organs": [{
        "name": "arith-dual",
        "capability": "arithmetic",
        "group": "R,+",
        "artifact": "arith.organ",
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }]}), encoding="utf-8")
    return OrganBank.from_manifest(manifest)


class BusTests(unittest.TestCase):
    def test_priority_orders_handlers_within_one_publish(self) -> None:
        bus = EventBus()
        order: list[str] = []
        bus.subscribe("life", lambda e: order.append("internal"), priority=Priority.INTERNAL)
        bus.subscribe("life", lambda e: order.append("user"), priority=Priority.USER)
        bus.publish("life", "message", priority=Priority.USER)
        self.assertEqual(order, ["user", "internal"])

    def test_unsubscribe_stops_delivery(self) -> None:
        bus = EventBus()
        hits: list[int] = []
        off = bus.subscribe("life", lambda e: hits.append(e.seq))
        bus.publish("life", "a")
        off()
        bus.publish("life", "b")
        self.assertEqual(len(hits), 1)


class DaemonTests(unittest.TestCase):
    def test_user_message_enters_stream_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "life.json"
            stream = RecordingStream()
            daemon = LifeDaemon(stream=stream, state_path=state)
            daemon.submit_user("hallo")
            self.assertEqual(stream.seen, ["hallo"])
            self.assertTrue(state.is_file())

    def test_restart_restores_memory_and_turn_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "life.json"
            first = LifeDaemon(stream=RecordingStream(), state_path=state)
            first.submit_user("sprich mit mir")
            second = LifeDaemon(stream=RecordingStream(), state_path=state)
            self.assertEqual(second.turns, 1)
            self.assertEqual(second.snapshot()["stream"]["seen"], ["sprich mit mir"])

    def test_mounted_organ_survives_restart_and_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bank = _make_bank(root)
            state = root / "life.json"

            first = LifeDaemon(bank=bank, state_path=state)
            first.rack.mount("arith-dual")
            first.submit_user("tick")

            second = LifeDaemon(bank=bank, state_path=state)
            self.assertEqual(second.rack.mounted(), ("arith-dual",))

            # Tampering kills the organ on the next mount attempt.
            (root / "arith.organ").write_bytes(b"tampered")
            with self.assertRaises(Exception):
                second.rack.mount("arith-dual")

    def test_exact_service_answers_and_abstains(self) -> None:
        daemon = LifeDaemon()
        daemon.register(ExactService())
        ok = daemon.request("exact_math", "1+1")
        self.assertTrue(ok.ok)
        self.assertEqual(ok.output, 2)
        abstain = daemon.request("exact_math", "was ist liebe")
        self.assertIs(abstain.status, ExecutionStatus.ABSTAINED)

    def test_unknown_service_is_unavailable_not_fatal(self) -> None:
        daemon = LifeDaemon()
        result = daemon.request("poetry", "sonett")
        self.assertIs(result.status, ExecutionStatus.UNAVAILABLE)

    def test_service_metadata_crosses_the_daemon_boundary(self) -> None:
        daemon = LifeDaemon()
        daemon.register(MetadataService())
        result = daemon.request("metadata", "payload", metadata={"trace_id": "turn-7"})
        self.assertEqual(result.output, "turn-7")

    def test_duplicate_capability_owner_is_rejected(self) -> None:
        daemon = LifeDaemon()
        daemon.register(ExactService())
        with self.assertRaisesRegex(ValueError, "already owned"):
            daemon.register(ExactService())

    def test_port_roundtrip_without_daemon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            port = LifeStatePort(Path(tmp) / "p.json")
            self.assertIsNone(port.load())
            port.save({"turns": 3})
            self.assertEqual(port.load(), {"turns": 3})
            self.assertEqual(list(Path(tmp).glob(".p.json.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
