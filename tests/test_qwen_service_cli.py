from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from immer.cli import main
from immer.contracts import ExecutionStatus, Result
from immer.runtimes.qwen3_8.service import (
    QwenServiceRequest,
    UnixQwenServiceServer,
)


class _RemoteApplication:
    def __init__(self, *, fail_chat: bool = False) -> None:
        self.fail_chat = fail_chat
        self.requests = []

    def __call__(self, request, emit):
        self.requests.append(request)
        if request.operation == "ping":
            return Result(
                ExecutionStatus.OK,
                "qwen3.8.service",
                output="pong",
                evidence={"runtime_profile_sha256": "b" * 64},
            )
        if self.fail_chat:
            raise RuntimeError("chat failed after dispatch")
        emit(
            "candidate_delta",
            {"delta": "remote", "replace": False, "snapshot": "remote"},
        )
        return Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output="remote answer",
            evidence={"transport": "socket"},
        )


class _DirectQwen:
    def __init__(self) -> None:
        self.closed = False

    def handle(self, _request):
        return Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output="direct answer",
        )

    def close(self) -> None:
        self.closed = True


class _Tokenizer:
    def render_no_thinking_prompt(self, system_prompt, message):
        return f"{system_prompt}|{message}"

    def render_no_thinking_messages(self, system_prompt, messages):
        return f"{system_prompt}|{messages!r}"

    def encode(self, text):
        return tuple(range(max(1, len(text.split()))))


class _ServiceQwen:
    def __init__(self, sink) -> None:
        self.sink = sink
        self.requests = []
        self.clear_calls = 0
        self.close_calls = 0

    def handle(self, request):
        self.requests.append(request)
        self.sink("service")
        return Result(
            ExecutionStatus.OK,
            "qwen3.8.causal-chat",
            output=f"answer:{request.payload}",
        )

    def clear_conversation(self) -> None:
        self.clear_calls += 1

    def close(self) -> None:
        self.close_calls += 1


class _InlineServer:
    instance = None

    def __init__(self, path, application) -> None:
        self.path = Path(path)
        self.application = application
        self.events = []
        self.closed = False
        type(self).instance = self

    def serve_forever(self) -> None:
        def emit(name, body):
            self.events.append((name, dict(body)))

        first = self.application(
            QwenServiceRequest("first", "chat", "session", "one"),
            emit,
        )
        second = self.application(
            QwenServiceRequest("second", "chat", "session", "two"),
            emit,
        )
        if first.output != "answer:one" or second.output != "answer:two":
            raise AssertionError("inline service changed the product result")

    def close(self) -> None:
        self.closed = True


class QwenServiceCliTests(unittest.TestCase):
    def _running_server(self, root: Path, *, fail_chat: bool = False):
        application = _RemoteApplication(fail_chat=fail_chat)
        server = UnixQwenServiceServer(root / "qwen.sock", application)
        server.open()
        thread = threading.Thread(target=server.serve_forever)
        thread.start()

        def cleanup() -> None:
            server.stop()
            thread.join(2.0)
            server.close()

        self.addCleanup(cleanup)
        return application, server

    def test_chat_auto_connects_without_constructing_qwen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            application, server = self._running_server(Path(temporary))
            output = io.StringIO()
            with (
                patch("immer.cli._qwen38_service_profile", return_value="b" * 64),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    side_effect=AssertionError("socket path mounted Qwen again"),
                ) as constructor,
                redirect_stdout(output),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--output",
                        "json",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--socket",
                        str(server.path),
                    ]
                )

        self.assertEqual(code, 0)
        constructor.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["output"], "remote answer")
        self.assertEqual(
            [request.operation for request in application.requests],
            ["ping", "chat"],
        )

    def test_failure_after_chat_dispatch_never_falls_back_to_direct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _application, server = self._running_server(
                Path(temporary),
                fail_chat=True,
            )
            output = io.StringIO()
            with (
                patch("immer.cli._qwen38_service_profile", return_value="b" * 64),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    side_effect=AssertionError(
                        "post-dispatch failure retried directly"
                    ),
                ) as constructor,
                redirect_stdout(output),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--output",
                        "json",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--socket",
                        str(server.path),
                    ]
                )

        self.assertEqual(code, 2)
        constructor.assert_not_called()
        self.assertIn("chat failed after dispatch", output.getvalue())

    def test_missing_service_uses_the_existing_direct_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            qwen = _DirectQwen()
            output = io.StringIO()
            with (
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    return_value=qwen,
                ) as constructor,
                redirect_stdout(output),
            ):
                code = main(
                    [
                        "chat",
                        "hello",
                        "--raw-qwen",
                        "--output",
                        "json",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--socket",
                        str(Path(temporary) / "absent.sock"),
                    ]
                )

        self.assertEqual(code, 0)
        constructor.assert_called_once()
        self.assertTrue(qwen.closed)
        self.assertEqual(json.loads(output.getvalue())["output"], "direct answer")

    def test_service_mode_mounts_once_and_handles_multiple_requests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            qwen_holder = {}

            def construct(*_args, **options):
                qwen = _ServiceQwen(options["text_snapshot_sink"])
                qwen_holder["qwen"] = qwen
                return qwen

            with (
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    side_effect=construct,
                ) as constructor,
                patch(
                    "immer.runtimes.qwen3_8.encoding.Qwen38Tokenizer",
                    return_value=_Tokenizer(),
                ),
                patch(
                    "immer.runtimes.qwen3_8.service.UnixQwenServiceServer",
                    _InlineServer,
                ),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "--service",
                        "--raw-qwen",
                        "--qwen38-causal-bundle",
                        "/models/qwen.causal",
                        "--qwen38-tokenizer",
                        "/models/tokenizer.json",
                        "--socket",
                        str(Path(temporary) / "qwen.sock"),
                    ]
                )

        qwen = qwen_holder["qwen"]
        self.assertEqual(code, 0)
        constructor.assert_called_once()
        self.assertEqual(len(qwen.requests), 2)
        self.assertEqual(qwen.close_calls, 1)
        self.assertTrue(_InlineServer.instance.closed)
        self.assertTrue(
            any(
                name == "candidate_delta"
                for name, _body in _InlineServer.instance.events
            )
        )

    def test_service_forwards_the_multilayer_o1_registry_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            q4 = root / "q4"
            atlas = root / "atlas"
            compute = root / "compute"
            q4.mkdir()
            atlas.mkdir()
            compute.mkdir()
            state = root / "layer63.json"

            def construct(*_args, **options):
                return _ServiceQwen(options["text_snapshot_sink"])

            policy = {
                "authorities": {
                    "atlas_revision_sha256": "a" * 64,
                    "compute_revision_sha256": "b" * 64,
                },
                "enabled": True,
                "layers": [0, 63],
                "schema": "immer.qwen3.8-layer-mlp-o1-policy/v2",
            }
            with (
                patch(
                    "immer.cli._qwen38_layer_mlp_o1_policy",
                    return_value=policy,
                ),
                patch(
                    "immer.runtimes.qwen3_8.adapter.Qwen38CausalChat",
                    side_effect=construct,
                ) as constructor,
                patch(
                    "immer.runtimes.qwen3_8.encoding.Qwen38Tokenizer",
                    return_value=_Tokenizer(),
                ),
                patch(
                    "immer.runtimes.qwen3_8.service.UnixQwenServiceServer",
                    _InlineServer,
                ),
                redirect_stdout(io.StringIO()),
                redirect_stderr(io.StringIO()),
            ):
                code = main(
                    [
                        "chat",
                        "--service",
                        "--raw-qwen",
                        "--qwen38-causal-bundle",
                        str(root / "qwen.causal"),
                        "--qwen38-tokenizer",
                        str(root / "tokenizer.json"),
                        "--qwen38-q4",
                        str(q4),
                        "--layer-mlp-o1-state",
                        str(state),
                        "--layer-mlp-o1-atlas",
                        str(atlas),
                        "--layer-mlp-o1-compute-root",
                        str(compute),
                        "--layer-mlp-o1-layers",
                        "0,63",
                        "--socket",
                        str(root / "qwen.sock"),
                    ]
                )

        self.assertEqual(code, 0)
        options = constructor.call_args.kwargs
        self.assertEqual(options["layer_mlp_o1_layers"], (0, 63))
        self.assertEqual(options["layer_mlp_o1_state_path"], str(state))


if __name__ == "__main__":
    unittest.main()
