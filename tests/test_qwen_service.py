from __future__ import annotations

from pathlib import Path
import socket
import stat
import tempfile
import threading
import unittest

from immer.contracts import ExecutionStatus, Result
from immer.runtimes.qwen3_8.service import (
    QwenChatServiceApplication,
    QwenServiceError,
    QwenServiceEvent,
    QwenServiceRemoteError,
    QwenServiceRequest,
    SnapshotEventBridge,
    UnixQwenServiceClient,
    UnixQwenServiceServer,
)


class _Application:
    def __init__(self) -> None:
        self.requests = []

    def __call__(self, request, emit):
        self.requests.append(request)
        if request.operation == "chat":
            if request.message == "explode":
                raise RuntimeError("simulated failure")
            emit("candidate_delta", {"text": "candidate"})
            emit("progress", {"tokens": 1})
            return Result(
                ExecutionStatus.OK,
                "qwen3.8.fertig-chat",
                output=f"final:{request.message}",
                evidence={
                    "inference_economics": {
                        "status": "recorded",
                        "rollup": {"requests": len(self.requests)},
                    }
                },
            )
        if request.operation == "clear":
            return Result(ExecutionStatus.OK, "qwen.service", output="cleared")
        if request.operation == "stats":
            return Result(
                ExecutionStatus.OK,
                "qwen.service",
                output={"requests": len(self.requests)},
            )
        return Result(ExecutionStatus.OK, "qwen.service", output="pong")


class _FakeQwen:
    def __init__(self, bridge: SnapshotEventBridge) -> None:
        self.bridge = bridge
        self.requests = []
        self.clear_calls = 0

    def handle(self, request):
        self.requests.append(request)
        self.bridge("hel")
        self.bridge("hello")
        self.bridge("hullo")
        return Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output=f"answer:{request.payload}",
            evidence={"generation": {"forward_passes": 1}},
        )

    def clear_conversation(self) -> None:
        self.clear_calls += 1


class _FakeProduct:
    def __init__(self, qwen: _FakeQwen) -> None:
        self.qwen = qwen
        self.requests = []

    def handle(self, request):
        self.requests.append(request)
        return self.qwen.handle(request)


def _chat_application():
    bridge = SnapshotEventBridge()
    qwen = _FakeQwen(bridge)
    product = _FakeProduct(qwen)

    def metadata(message, history, session_id):
        return {"history": history, "session": session_id, "message": message}

    def fit(message, history):
        return history, 0, len(message)

    def economics(message, result, *, request_id):
        evidence = dict(result.evidence)
        evidence["inference_economics"] = {
            "request_id": request_id,
            "status": "recorded",
        }
        return Result(
            result.status,
            result.component,
            output=result.output,
            reason=result.reason,
            evidence=evidence,
        )

    application = QwenChatServiceApplication(
        qwen=qwen,
        component=product,
        snapshot_bridge=bridge,
        request_metadata_for=metadata,
        fit_history=fit,
        attach_inference_economics=economics,
        result_summary=lambda result: f"summary:{result.output}",
        runtime_profile_sha256="a" * 64,
    )
    return application, qwen, product


class QwenServiceProtocolTests(unittest.TestCase):
    def test_request_and_event_roundtrip_and_tamper_rejection(self) -> None:
        request = QwenServiceRequest(
            request_id="request-1",
            operation="chat",
            session_id="session-1",
            message="hello",
        )
        self.assertEqual(QwenServiceRequest.from_bytes(request.to_bytes()), request)
        event = QwenServiceEvent(
            request_id=request.request_id,
            sequence=0,
            event="started",
            body={"operation": "chat"},
        )
        self.assertEqual(QwenServiceEvent.from_bytes(event.to_bytes()), event)
        tampered = bytearray(request.to_bytes())
        tampered[-3] ^= 1
        with self.assertRaises(QwenServiceError):
            QwenServiceRequest.from_bytes(bytes(tampered))


class QwenChatServiceApplicationTests(unittest.TestCase):
    def test_sessions_contextual_raw_qwen_clear_and_snapshot_rewrite(self) -> None:
        application, qwen, product = _chat_application()
        events = []

        def emit(name, body):
            events.append((name, dict(body)))

        first = application(
            QwenServiceRequest("one", "chat", "session-a", "first"),
            emit,
        )
        second = application(
            QwenServiceRequest("two", "chat", "session-a", "second"),
            emit,
        )
        other = application(
            QwenServiceRequest("three", "chat", "session-b", "other"),
            emit,
        )
        stats = application(
            QwenServiceRequest("stats", "stats", "session-a"),
            emit,
        )
        cleared = application(
            QwenServiceRequest("clear", "clear", "session-a"),
            emit,
        )
        after_clear = application(
            QwenServiceRequest("four", "chat", "session-a", "again"),
            emit,
        )

        self.assertEqual(first.output, "answer:first")
        self.assertEqual(second.output, "answer:second")
        self.assertEqual(other.output, "answer:other")
        self.assertEqual(after_clear.output, "answer:again")
        self.assertEqual(len(product.requests), 3)
        self.assertEqual(len(qwen.requests), 4)
        self.assertEqual(
            qwen.requests[1].metadata["history"],
            (("user", "first"), ("assistant", "answer:first")),
        )
        self.assertEqual(qwen.clear_calls, 1)
        self.assertEqual(cleared.output, "Conversation context cleared.")
        self.assertEqual(stats.output, "summary:answer:second")
        candidates = [body for name, body in events if name == "candidate_delta"]
        self.assertEqual(candidates[0]["delta"], "hel")
        self.assertEqual(candidates[1]["delta"], "lo")
        self.assertTrue(candidates[2]["replace"])
        self.assertEqual(candidates[2]["snapshot"], "hullo")
        self.assertEqual(
            first.evidence["inference_economics"]["request_id"],
            "one",
        )

    def test_ping_exposes_profile_without_touching_qwen(self) -> None:
        application, qwen, product = _chat_application()
        result = application(
            QwenServiceRequest("ping", "ping", "session"),
            lambda _name, _body: None,
        )
        self.assertEqual(result.output, "pong")
        self.assertEqual(result.evidence["runtime_profile_sha256"], "a" * 64)
        self.assertFalse(qwen.requests)
        self.assertFalse(product.requests)

    def test_stateless_chat_never_carries_history(self) -> None:
        application, qwen, product = _chat_application()

        def emit(_name, _body):
            return None

        application(QwenServiceRequest("one", "chat", None, "first"), emit)
        application(QwenServiceRequest("two", "chat", None, "second"), emit)

        self.assertEqual(len(product.requests), 2)
        self.assertEqual(len(qwen.requests), 2)
        self.assertEqual(qwen.requests[0].metadata["history"], ())
        self.assertEqual(qwen.requests[1].metadata["history"], ())
        self.assertIsNone(qwen.requests[0].metadata["session"])
        self.assertIsNone(qwen.requests[1].metadata["session"])


class UnixQwenServiceTests(unittest.TestCase):
    def _server(self, root: Path):
        application = _Application()
        server = UnixQwenServiceServer(root / "qwen.sock", application)
        server.open()
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        self.addCleanup(server.close)
        self.addCleanup(thread.join, 2.0)
        self.addCleanup(server.stop)
        return application, server

    def test_persistent_client_receives_candidate_receipt_and_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            application, server = self._server(root)
            self.assertEqual(stat.S_IMODE(server.path.lstat().st_mode), 0o600)
            with UnixQwenServiceClient(server.path) as client:
                first = client.request(
                    "chat",
                    session_id="session",
                    message="one",
                    request_id="one",
                )
                second = client.request(
                    "chat",
                    session_id="session",
                    message="two",
                    request_id="two",
                )
                clear = client.request("clear", session_id="session")
                stats = client.request("stats", session_id="session")
                ping = client.request("ping", session_id="session")
            self.assertEqual(first.result.output, "final:one")
            self.assertEqual(second.result.output, "final:two")
            self.assertEqual(
                [event.event for event in first.events],
                ["started", "candidate_delta", "progress", "receipt", "final"],
            )
            self.assertEqual(clear.result.output, "cleared")
            self.assertEqual(stats.result.output["requests"], 4)
            self.assertEqual(ping.result.output, "pong")
            self.assertEqual(len(application.requests), 5)
            server.stop()

    def test_remote_error_is_terminal_without_killing_the_service(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            application, server = self._server(Path(temporary))
            with UnixQwenServiceClient(server.path) as client:
                with self.assertRaises(QwenServiceRemoteError):
                    client.request(
                        "chat",
                        session_id="session",
                        message="explode",
                    )
                response = client.request("ping", session_id="session")
            self.assertEqual(response.result.output, "pong")
            self.assertEqual(len(application.requests), 2)
            server.stop()

    def test_stale_socket_is_replaced_but_live_socket_is_not(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "qwen.sock"
            stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            stale.bind(str(path))
            stale.close()
            server = UnixQwenServiceServer(path, _Application())
            server.open()
            self.assertTrue(path.exists())
            competing = UnixQwenServiceServer(path, _Application())
            with self.assertRaises(QwenServiceError):
                competing.open()
            server.close()
            self.assertFalse(path.exists())

    def test_non_socket_path_is_never_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "qwen.sock"
            path.write_text("do not remove", encoding="utf-8")
            server = UnixQwenServiceServer(path, _Application())
            with self.assertRaises(QwenServiceError):
                server.open()
            self.assertEqual(path.read_text(encoding="utf-8"), "do not remove")


if __name__ == "__main__":
    unittest.main()
