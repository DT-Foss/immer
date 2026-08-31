"""Local Unix-socket request/event transport for the canonical Qwen runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import stat
import threading
from typing import Any, Literal

from ...contracts import ExecutionStatus, Request, Result
from ..ooe.chat import result_from_document, result_to_document
from ..ooe.identity import canonical_json_bytes


QWEN_SERVICE_REQUEST_SCHEMA = "immer.qwen3.8-service-request/v1"
QWEN_SERVICE_EVENT_SCHEMA = "immer.qwen3.8-service-event/v1"
MAX_SERVICE_REQUEST_BYTES = 1024 * 1024
MAX_SERVICE_EVENT_BYTES = 8 * 1024 * 1024
MAX_SERVICE_ID_CHARS = 128
MAX_SOCKET_PATH_BYTES = 96
ServiceOperation = Literal["chat", "clear", "stats", "ping"]
ServiceEventName = Literal[
    "started",
    "candidate_delta",
    "progress",
    "receipt",
    "final",
    "error",
]
_OPERATIONS = frozenset(("chat", "clear", "stats", "ping"))
_EVENTS = frozenset(
    ("started", "candidate_delta", "progress", "receipt", "final", "error")
)


class QwenServiceError(RuntimeError):
    """The local service transport or protocol failed."""


class QwenServiceRemoteError(QwenServiceError):
    """The service returned a terminal error event."""


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class QwenServiceRequest:
    request_id: str
    operation: ServiceOperation
    session_id: str | None
    message: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_id, str)
            or not self.request_id
            or len(self.request_id) > MAX_SERVICE_ID_CHARS
        ):
            raise ValueError("request_id is invalid")
        if self.session_id is not None and (
            not isinstance(self.session_id, str)
            or not self.session_id
            or len(self.session_id) > MAX_SERVICE_ID_CHARS
        ):
            raise ValueError("session_id is invalid")
        if self.operation not in _OPERATIONS:
            raise ValueError("service operation is invalid")
        if self.operation == "chat":
            if not isinstance(self.message, str) or not self.message.strip():
                raise ValueError("chat request requires a non-empty message")
            if len(self.message.encode("utf-8")) > MAX_SERVICE_REQUEST_BYTES:
                raise ValueError("chat message exceeds its byte limit")
        elif self.message is not None:
            raise ValueError("non-chat service request cannot carry a message")
        if self.operation in {"clear", "stats"} and self.session_id is None:
            raise ValueError(f"{self.operation} requires a session_id")

    def to_document(self) -> dict[str, object]:
        body = {
            "message": self.message,
            "operation": self.operation,
            "request_id": self.request_id,
            "session_id": self.session_id,
        }
        return {
            "body": body,
            "schema": QWEN_SERVICE_REQUEST_SCHEMA,
            "sha256": _digest(body),
        }

    def to_bytes(self) -> bytes:
        payload = canonical_json_bytes(self.to_document()) + b"\n"
        if len(payload) > MAX_SERVICE_REQUEST_BYTES:
            raise QwenServiceError("service request exceeds its byte limit")
        return payload

    @classmethod
    def from_document(cls, value: object) -> "QwenServiceRequest":
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "schema", "sha256"}
            or value.get("schema") != QWEN_SERVICE_REQUEST_SCHEMA
            or not isinstance(value.get("body"), Mapping)
            or value.get("sha256") != _digest(value["body"])
        ):
            raise QwenServiceError("service request envelope is invalid")
        body = value["body"]
        if set(body) != {"message", "operation", "request_id", "session_id"}:
            raise QwenServiceError("service request body is invalid")
        try:
            request = cls(
                request_id=body["request_id"],
                operation=body["operation"],
                session_id=body["session_id"],
                message=body["message"],
            )
        except (TypeError, ValueError) as exc:
            raise QwenServiceError("service request failed validation") from exc
        if request.to_document() != dict(value):
            raise QwenServiceError("service request is not canonical")
        return request

    @classmethod
    def from_bytes(cls, payload: bytes) -> "QwenServiceRequest":
        if (
            not isinstance(payload, bytes)
            or len(payload) > MAX_SERVICE_REQUEST_BYTES
            or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")
        ):
            raise QwenServiceError("service request bytes are invalid")
        document_bytes = payload[:-1]
        try:
            value = json.loads(document_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QwenServiceError("service request is invalid JSON") from exc
        if canonical_json_bytes(value) != document_bytes:
            raise QwenServiceError("service request JSON is not canonical")
        return cls.from_document(value)


@dataclass(frozen=True, slots=True)
class QwenServiceEvent:
    request_id: str
    sequence: int
    event: ServiceEventName
    body: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.request_id, str)
            or not self.request_id
            or len(self.request_id) > MAX_SERVICE_ID_CHARS
        ):
            raise ValueError("event request_id is invalid")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 0
        ):
            raise ValueError("event sequence is invalid")
        if self.event not in _EVENTS:
            raise ValueError("event name is invalid")
        if not isinstance(self.body, Mapping):
            raise TypeError("event body must be a mapping")

    def to_document(self) -> dict[str, object]:
        body = {
            "body": dict(self.body),
            "event": self.event,
            "request_id": self.request_id,
            "sequence": self.sequence,
        }
        return {
            "body": body,
            "schema": QWEN_SERVICE_EVENT_SCHEMA,
            "sha256": _digest(body),
        }

    def to_bytes(self) -> bytes:
        payload = canonical_json_bytes(self.to_document()) + b"\n"
        if len(payload) > MAX_SERVICE_EVENT_BYTES:
            raise QwenServiceError("service event exceeds its byte limit")
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes) -> "QwenServiceEvent":
        if (
            not isinstance(payload, bytes)
            or len(payload) > MAX_SERVICE_EVENT_BYTES
            or not payload.endswith(b"\n")
            or payload.endswith(b"\n\n")
        ):
            raise QwenServiceError("service event bytes are invalid")
        document_bytes = payload[:-1]
        try:
            value = json.loads(document_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QwenServiceError("service event is invalid JSON") from exc
        if canonical_json_bytes(value) != document_bytes:
            raise QwenServiceError("service event JSON is not canonical")
        if (
            not isinstance(value, Mapping)
            or set(value) != {"body", "schema", "sha256"}
            or value.get("schema") != QWEN_SERVICE_EVENT_SCHEMA
            or not isinstance(value.get("body"), Mapping)
            or value.get("sha256") != _digest(value["body"])
        ):
            raise QwenServiceError("service event envelope is invalid")
        body = value["body"]
        if set(body) != {"body", "event", "request_id", "sequence"}:
            raise QwenServiceError("service event body is invalid")
        try:
            event = cls(
                request_id=body["request_id"],
                sequence=body["sequence"],
                event=body["event"],
                body=body["body"],
            )
        except (TypeError, ValueError) as exc:
            raise QwenServiceError("service event failed validation") from exc
        if event.to_document() != dict(value):
            raise QwenServiceError("service event is not canonical")
        return event


ServiceEmitter = Callable[[ServiceEventName, Mapping[str, Any]], None]
ServiceApplication = Callable[[QwenServiceRequest, ServiceEmitter], Result]
RequestMetadataFactory = Callable[
    [str, tuple[tuple[str, str], ...], str | None],
    Mapping[str, object],
]
HistoryFitter = Callable[
    [str, tuple[tuple[str, str], ...]],
    tuple[tuple[tuple[str, str], ...], int, int],
]
EconomicsAttacher = Callable[..., Result]


class SnapshotEventBridge:
    """Route one constructor-fixed Qwen snapshot callback into the active request."""

    def __init__(self) -> None:
        self._emit: ServiceEmitter | None = None
        self._snapshot = ""
        self._updates = 0

    def bind(self, emit: ServiceEmitter) -> None:
        if self._emit is not None:
            raise QwenServiceError("snapshot bridge is already bound")
        self._emit = emit
        self._snapshot = ""
        self._updates = 0

    def unbind(self) -> None:
        self._emit = None
        self._snapshot = ""
        self._updates = 0

    def __call__(self, snapshot: str) -> None:
        if not isinstance(snapshot, str):
            raise TypeError("streaming snapshot must be text")
        emit = self._emit
        if emit is None:
            return
        snapshot = snapshot.strip()
        if snapshot == self._snapshot:
            return
        replace = not snapshot.startswith(self._snapshot)
        delta = snapshot if replace else snapshot[len(self._snapshot) :]
        self._snapshot = snapshot
        self._updates += 1
        emit(
            "candidate_delta",
            {
                "delta": delta,
                "replace": replace,
                "snapshot": snapshot,
            },
        )
        emit(
            "progress",
            {
                "confirmed_updates": self._updates,
                "snapshot_characters": len(snapshot),
            },
        )


@dataclass(slots=True)
class _ChatSession:
    history: tuple[tuple[str, str], ...] = ()
    generation: int = 0
    completed_requests: int = 0
    last_summary: str | None = None
    last_evidence: Mapping[str, Any] = field(default_factory=dict)


class QwenChatServiceApplication:
    """Session-bound application over one persistent Qwen/FERTIG runtime."""

    def __init__(
        self,
        *,
        qwen: Any,
        component: Any,
        snapshot_bridge: SnapshotEventBridge,
        request_metadata_for: RequestMetadataFactory,
        fit_history: HistoryFitter,
        attach_inference_economics: EconomicsAttacher,
        result_summary: Callable[[Result], str | None],
        runtime_profile_sha256: str,
    ) -> None:
        if (
            not isinstance(runtime_profile_sha256, str)
            or len(runtime_profile_sha256) != 64
        ):
            raise ValueError("runtime service profile must be a SHA-256 digest")
        self.qwen = qwen
        self.component = component
        self.snapshot_bridge = snapshot_bridge
        self.request_metadata_for = request_metadata_for
        self.fit_history = fit_history
        self.attach_inference_economics = attach_inference_economics
        self.result_summary = result_summary
        self.runtime_profile_sha256 = runtime_profile_sha256
        self._sessions: dict[str, _ChatSession] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _native_session_id(session_id: str, generation: int) -> str:
        digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()
        return f"socket:{digest}:{generation}"

    def _session(self, session_id: str) -> _ChatSession:
        return self._sessions.setdefault(session_id, _ChatSession())

    @staticmethod
    def _control_result(
        output: Any,
        *,
        evidence: Mapping[str, Any] | None = None,
    ) -> Result:
        return Result(
            ExecutionStatus.OK,
            "qwen3.8.service",
            output=output,
            evidence={} if evidence is None else evidence,
        )

    def __call__(
        self,
        request: QwenServiceRequest,
        emit: ServiceEmitter,
    ) -> Result:
        with self._lock:
            if request.operation == "ping":
                return self._control_result(
                    "pong",
                    evidence={
                        "runtime_profile_sha256": self.runtime_profile_sha256,
                    },
                )
            if request.operation != "chat":
                assert request.session_id is not None
            session = (
                _ChatSession()
                if request.session_id is None
                else self._session(request.session_id)
            )
            if request.operation == "stats":
                return self._control_result(
                    session.last_summary or "No completed Qwen response yet.",
                    evidence={
                        "completed_requests": session.completed_requests,
                        "last_result": dict(session.last_evidence),
                        "runtime_profile_sha256": self.runtime_profile_sha256,
                    },
                )
            if request.operation == "clear":
                session.history = ()
                session.generation += 1
                session.last_summary = None
                session.last_evidence = {}
                assert request.session_id is not None
                self._sessions.pop(request.session_id, None)
                self.qwen.clear_conversation()
                return self._control_result("Conversation context cleared.")

            assert request.message is not None
            retained, dropped_turns, prompt_tokens = self.fit_history(
                request.message,
                session.history,
            )
            native_session_id = (
                None
                if request.session_id is None
                else self._native_session_id(
                    request.session_id,
                    session.generation,
                )
            )
            metadata = self.request_metadata_for(
                request.message,
                retained,
                native_session_id,
            )
            turn_component = self.qwen if session.history else self.component
            self.snapshot_bridge.bind(emit)
            try:
                result = turn_component.handle(
                    Request("chat", request.message, metadata)
                )
            finally:
                self.snapshot_bridge.unbind()
            result = self.attach_inference_economics(
                request.message,
                result,
                request_id=request.request_id,
            )
            session.completed_requests += 1
            if result.ok and isinstance(result.output, str):
                session.history = (
                    *retained,
                    ("user", request.message.strip()),
                    ("assistant", result.output.strip()),
                )
            session.last_summary = self.result_summary(result)
            session.last_evidence = dict(result.evidence)
            evidence = dict(result.evidence)
            evidence["service_session"] = {
                "dropped_turns": dropped_turns,
                "history_turns": len(retained) // 2,
                "prompt_tokens": prompt_tokens,
            }
            return Result(
                result.status,
                result.component,
                output=result.output,
                reason=result.reason,
                evidence=evidence,
            )


@dataclass(frozen=True, slots=True)
class QwenServiceResponse:
    result: Result
    events: tuple[QwenServiceEvent, ...]


class UnixQwenServiceServer:
    """Local service whose serialized application owns one persistent runtime."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        application: ServiceApplication,
    ) -> None:
        self.path = Path(path).expanduser().absolute()
        if len(os.fsencode(self.path)) > MAX_SOCKET_PATH_BYTES:
            raise QwenServiceError("service socket path is too long")
        if not callable(application):
            raise TypeError("service application must be callable")
        self.application = application
        self._socket: socket.socket | None = None
        self._socket_identity: tuple[int, int] | None = None
        self._stop = threading.Event()
        self._application_lock = threading.Lock()
        self._connection_lock = threading.Lock()
        self._connection_threads: set[threading.Thread] = set()

    def _prepare_parent(self) -> None:
        parent = self.path.parent
        try:
            metadata = parent.lstat()
        except FileNotFoundError:
            parent.mkdir(parents=True, mode=0o700)
            metadata = parent.lstat()
        if not stat.S_ISDIR(metadata.st_mode):
            raise QwenServiceError("service socket parent is not a plain directory")

    def _remove_stale_socket(self) -> None:
        try:
            metadata = self.path.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISSOCK(metadata.st_mode):
            raise QwenServiceError("service socket path already exists")
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.2)
            probe.connect(str(self.path))
        except (ConnectionRefusedError, FileNotFoundError):
            self.path.unlink()
            return
        except TimeoutError as exc:
            raise QwenServiceError("service socket is already live") from exc
        finally:
            probe.close()
        raise QwenServiceError("service socket is already live")

    def open(self) -> None:
        if self._socket is not None:
            return
        self._prepare_parent()
        self._remove_stale_socket()
        owner = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            owner.bind(str(self.path))
            os.chmod(self.path, 0o600)
            metadata = self.path.lstat()
            if not stat.S_ISSOCK(metadata.st_mode):
                raise QwenServiceError("bound service path is not a socket")
            self._socket_identity = (metadata.st_dev, metadata.st_ino)
            owner.listen(8)
            owner.settimeout(0.2)
        except Exception:
            owner.close()
            raise
        self._socket = owner

    def stop(self) -> None:
        self._stop.set()

    def _send_event(
        self,
        connection: socket.socket,
        request_id: str,
        sequence: int,
        event: ServiceEventName,
        body: Mapping[str, Any],
    ) -> None:
        connection.sendall(
            QwenServiceEvent(
                request_id=request_id,
                sequence=sequence,
                event=event,
                body=body,
            ).to_bytes()
        )

    def _serve_request(
        self,
        connection: socket.socket,
        request: QwenServiceRequest,
    ) -> None:
        sequence = 0

        def emit(event: ServiceEventName, body: Mapping[str, Any]) -> None:
            nonlocal sequence
            if event not in {"candidate_delta", "progress"}:
                raise QwenServiceError("application emitted an owned terminal event")
            self._send_event(
                connection,
                request.request_id,
                sequence,
                event,
                body,
            )
            sequence += 1

        self._send_event(
            connection,
            request.request_id,
            sequence,
            "started",
            {"operation": request.operation, "session_id": request.session_id},
        )
        sequence += 1
        try:
            with self._application_lock:
                result = self.application(request, emit)
            if not isinstance(result, Result):
                raise TypeError("service application returned a non-Result value")
            economics = result.evidence.get("inference_economics")
            if isinstance(economics, Mapping):
                self._send_event(
                    connection,
                    request.request_id,
                    sequence,
                    "receipt",
                    {"inference_economics": dict(economics)},
                )
                sequence += 1
            self._send_event(
                connection,
                request.request_id,
                sequence,
                "final",
                {"result": result_to_document(result)},
            )
        except Exception as exc:
            self._send_event(
                connection,
                request.request_id,
                sequence,
                "error",
                {
                    "error": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "reason": str(exc)[:512],
                },
            )

    def _serve_connection(self, connection: socket.socket) -> None:
        try:
            with connection, connection.makefile("rb") as reader:
                while not self._stop.is_set():
                    line = reader.readline(MAX_SERVICE_REQUEST_BYTES + 1)
                    if not line:
                        return
                    if len(line) > MAX_SERVICE_REQUEST_BYTES:
                        return
                    try:
                        request = QwenServiceRequest.from_bytes(line)
                    except QwenServiceError:
                        return
                    self._serve_request(connection, request)
        finally:
            current = threading.current_thread()
            with self._connection_lock:
                self._connection_threads.discard(current)

    def serve_forever(self) -> None:
        self.open()
        assert self._socket is not None
        while not self._stop.is_set():
            try:
                connection, _address = self._socket.accept()
            except TimeoutError:
                continue
            worker = threading.Thread(
                target=self._serve_connection,
                args=(connection,),
                daemon=True,
                name="immer-qwen-client",
            )
            with self._connection_lock:
                self._connection_threads.add(worker)
            worker.start()

    def close(self) -> None:
        owner = self._socket
        self._socket = None
        if owner is not None:
            owner.close()
        with self._connection_lock:
            workers = tuple(self._connection_threads)
        for worker in workers:
            worker.join(timeout=1.0)
        identity = self._socket_identity
        self._socket_identity = None
        if identity is None:
            return
        try:
            metadata = self.path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISSOCK(metadata.st_mode) and (metadata.st_dev, metadata.st_ino) == identity:
            self.path.unlink()

    def __enter__(self) -> "UnixQwenServiceServer":
        self.open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class UnixQwenServiceClient:
    """Persistent local client for single, JSONL, and interactive frontends."""

    def __init__(self, path: str | os.PathLike[str], *, timeout: float = 5.0) -> None:
        self.path = Path(path).expanduser().absolute()
        self.timeout = float(timeout)
        self._socket: socket.socket | None = None
        self._reader: Any | None = None

    def connect(self) -> None:
        if self._socket is not None:
            return
        owner = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        owner.settimeout(self.timeout)
        try:
            owner.connect(str(self.path))
        except Exception:
            owner.close()
            raise
        owner.settimeout(None)
        self._socket = owner
        self._reader = owner.makefile("rb")

    def request(
        self,
        operation: ServiceOperation,
        *,
        session_id: str | None = None,
        message: str | None = None,
        request_id: str | None = None,
        event_sink: Callable[[QwenServiceEvent], None] | None = None,
    ) -> QwenServiceResponse:
        self.connect()
        assert self._socket is not None and self._reader is not None
        request = QwenServiceRequest(
            request_id=request_id or secrets.token_hex(16),
            operation=operation,
            session_id=session_id,
            message=message,
        )
        self._socket.sendall(request.to_bytes())
        events = []
        expected_sequence = 0
        while True:
            line = self._reader.readline(MAX_SERVICE_EVENT_BYTES + 1)
            if not line or len(line) > MAX_SERVICE_EVENT_BYTES:
                raise QwenServiceError("service closed before a terminal event")
            event = QwenServiceEvent.from_bytes(line)
            if (
                event.request_id != request.request_id
                or event.sequence != expected_sequence
            ):
                raise QwenServiceError("service event stream lost request ordering")
            expected_sequence += 1
            events.append(event)
            if event_sink is not None:
                event_sink(event)
            if event.event == "error":
                raise QwenServiceRemoteError(
                    str(event.body.get("reason", "remote service error"))
                )
            if event.event == "final":
                try:
                    result = result_from_document(event.body["result"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise QwenServiceError("service final result is invalid") from exc
                return QwenServiceResponse(result=result, events=tuple(events))

    def close(self) -> None:
        reader = self._reader
        self._reader = None
        if reader is not None:
            reader.close()
        owner = self._socket
        self._socket = None
        if owner is not None:
            owner.close()

    def __enter__(self) -> "UnixQwenServiceClient":
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = [
    "MAX_SERVICE_EVENT_BYTES",
    "MAX_SERVICE_REQUEST_BYTES",
    "QWEN_SERVICE_EVENT_SCHEMA",
    "QWEN_SERVICE_REQUEST_SCHEMA",
    "QwenChatServiceApplication",
    "QwenServiceError",
    "QwenServiceEvent",
    "QwenServiceRemoteError",
    "QwenServiceRequest",
    "QwenServiceResponse",
    "SnapshotEventBridge",
    "UnixQwenServiceClient",
    "UnixQwenServiceServer",
]
