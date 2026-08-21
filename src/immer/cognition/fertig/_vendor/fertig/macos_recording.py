"""Passive, local macOS demonstration recording for the desktop apprentice.

The real recorder is intentionally split in two: a tiny listen-only CGEventTap
helper emits JSON Lines, while Python samples the screen concurrently and
turns timestamped observations into :class:`~fertig.desktop.RecordedStep`
objects.  Importing or constructing anything in this module never compiles a
binary, requests a permission, captures a screen, or listens for input.

F8 stops a live recording.  Secure text is never emitted when Accessibility
can identify an ``AXSecureTextField``.  If that check is unavailable, printable
text is suppressed by default; ``allow_unverified_text=True`` is an explicit,
local-only opt-in for environments where Input Monitoring is granted without
Accessibility.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, TextIO

import numpy as np

from .desktop import (
    DesktopAction,
    DesktopBackendError,
    DesktopDemonstration,
    DesktopPermissionError,
    MacOSScreenshotBackend,
    RecordedStep,
    ScreenshotBackend,
    _normalise_frame,
)


STOP_KEY = "f8"
STOP_KEY_CODE = 100
EVENT_TYPES = frozenset(
    {
        "ready",
        "status",
        "mouse_down",
        "mouse_up",
        "key_down",
        "stop",
        "closed",
        "error",
    }
)
MODIFIERS = frozenset(
    {"shift", "control", "option", "command", "caps_lock", "function"}
)

# Apple virtual key codes which the strict DesktopAction API can reproduce.
KEYCODE_TO_DESKTOP_KEY: Mapping[int, str] = {
    36: "return",
    48: "tab",
    49: "space",
    51: "delete",
    53: "escape",
    115: "home",
    116: "page_up",
    119: "end",
    121: "page_down",
    123: "left",
    124: "right",
    125: "down",
    126: "up",
}

_SOURCE = Path(__file__).with_name("_macos") / "EventTap.swift"
_EOF = object()


class PassiveRecordingError(DesktopBackendError):
    """Base error for passive recording, helper builds, and event decoding."""


class EventTapBuildError(PassiveRecordingError):
    """The bundled Swift event-tap helper could not be built."""


class EventTapProtocolError(PassiveRecordingError):
    """The event helper emitted malformed or faithfully unreplayable input."""


@dataclass(frozen=True)
class EventTapEvent:
    """One validated JSONL record emitted by the Swift helper."""

    kind: str
    timestamp_ns: int
    x: float | None = None
    y: float | None = None
    button: int | None = None
    keycode: int | None = None
    modifiers: tuple[str, ...] = ()
    text: str | None = None
    text_suppressed: str | None = None
    repeat: bool = False
    code: str | None = None
    message: str | None = None
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)


@dataclass(frozen=True)
class ScreenSample:
    """An RGB screenshot paired with Python's monotonic nanosecond clock."""

    timestamp_ns: int
    frame: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp_ns, int) or isinstance(
            self.timestamp_ns, bool
        ):
            raise TypeError("sample timestamp_ns must be an integer")
        if self.timestamp_ns < 0:
            raise ValueError("sample timestamp_ns must be non-negative")
        object.__setattr__(self, "frame", _normalise_frame(self.frame))


@dataclass(frozen=True)
class MacOSPermissionStatus:
    """Read-only result of the recorder helper's permission preflight."""

    helper: Path
    input_monitoring: bool
    accessibility: bool
    input_sent: bool = False


class ScreenSampler(Protocol):
    """Injectable interface used to keep live tests completely offline."""

    def start(self) -> None: ...

    def finish(self, settle_seconds: float) -> Sequence[ScreenSample]: ...


def _strict_int(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise EventTapProtocolError(f"event {name} must be an integer >= {minimum}")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise EventTapProtocolError(f"event {name} must be a non-empty string")
    return value


def parse_event_line(line: str | bytes) -> EventTapEvent:
    """Parse and strictly validate one helper JSONL record.

    Unknown top-level fields are retained in ``metadata`` for forward
    compatibility, while every field used for replay remains type checked.
    """

    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EventTapProtocolError("event line is not UTF-8") from exc
    if not isinstance(line, str):
        raise TypeError("event line must be str or bytes")
    if len(line) > 65_536:
        raise EventTapProtocolError("event line exceeds 64 KiB")
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise EventTapProtocolError(f"malformed event JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise EventTapProtocolError("event JSON must be an object")

    kind = value.get("type")
    if not isinstance(kind, str) or kind not in EVENT_TYPES:
        raise EventTapProtocolError(f"unknown event type: {kind!r}")
    timestamp_ns = _strict_int(value.get("timestamp_ns"), "timestamp_ns")

    raw_modifiers = value.get("modifiers", [])
    if not isinstance(raw_modifiers, list) or not all(
        isinstance(item, str) for item in raw_modifiers
    ):
        raise EventTapProtocolError("event modifiers must be a string array")
    unknown_modifiers = set(raw_modifiers) - MODIFIERS
    if unknown_modifiers:
        raise EventTapProtocolError(
            f"event contains unknown modifiers: {sorted(unknown_modifiers)}"
        )
    modifiers = tuple(dict.fromkeys(raw_modifiers))

    x = y = None
    button = None
    if kind in {"mouse_down", "mouse_up"}:
        raw_x, raw_y = value.get("x"), value.get("y")
        if (
            isinstance(raw_x, bool)
            or not isinstance(raw_x, (int, float))
            or isinstance(raw_y, bool)
            or not isinstance(raw_y, (int, float))
            or not math.isfinite(raw_x)
            or not math.isfinite(raw_y)
        ):
            raise EventTapProtocolError("mouse event requires finite x/y coordinates")
        x, y = float(raw_x), float(raw_y)
        button = _strict_int(value.get("button"), "button")

    keycode = None
    if kind in {"key_down", "stop"}:
        keycode = _strict_int(value.get("keycode"), "keycode")
        if keycode > 65_535:
            raise EventTapProtocolError("event keycode must fit in 16 bits")

    text = value.get("text")
    if text is not None:
        if (
            not isinstance(text, str)
            or not text
            or len(text) > 10_000
            or not text.isprintable()
        ):
            raise EventTapProtocolError("event text must be printable and non-empty")
    repeat = value.get("repeat", False)
    if not isinstance(repeat, bool):
        raise EventTapProtocolError("event repeat must be boolean")

    known = {
        "type",
        "timestamp_ns",
        "x",
        "y",
        "button",
        "keycode",
        "modifiers",
        "text",
        "text_suppressed",
        "repeat",
        "code",
        "message",
        "reason",
    }
    return EventTapEvent(
        kind=kind,
        timestamp_ns=timestamp_ns,
        x=x,
        y=y,
        button=button,
        keycode=keycode,
        modifiers=modifiers,
        text=text,
        text_suppressed=_optional_string(
            value.get("text_suppressed"), "text_suppressed"
        ),
        repeat=repeat,
        code=_optional_string(value.get("code"), "code"),
        message=_optional_string(value.get("message"), "message"),
        reason=_optional_string(value.get("reason"), "reason"),
        metadata={key: item for key, item in value.items() if key not in known},
    )


def _default_cache_dir() -> Path:
    return Path.home() / "Library" / "Caches" / "FERTIG" / "event-tap"


def _candidate_sdk_paths(swiftc: str) -> tuple[Path, ...]:
    """Return installed SDKs, preferring the toolchain default then older SDKs."""

    compiler = Path(swiftc).resolve()
    if compiler.parent == Path("/usr/bin"):
        sdk_dir = Path("/Library/Developer/CommandLineTools/SDKs")
    else:
        sdk_dir = (
            compiler.parents[2] / "SDKs"
            if len(compiler.parents) > 2
            else Path("/Library/Developer/CommandLineTools/SDKs")
        )
    candidates: list[Path] = []
    default = sdk_dir / "MacOSX.sdk"
    if default.exists():
        candidates.append(default)
    try:
        installed = sorted(
            sdk_dir.glob("MacOSX*.sdk"),
            key=lambda item: item.name,
            reverse=True,
        )
    except OSError:
        installed = []
    for candidate in installed:
        resolved = candidate.resolve()
        if all(existing.resolve() != resolved for existing in candidates):
            candidates.append(candidate)
    return tuple(candidates)


def _compiler_mismatch(stderr: str) -> bool:
    lower = stderr.lower()
    return "sdk is not supported by the compiler" in lower or (
        "could not build objective-c module" in lower and "swiftshims" in lower
    )


def build_event_tap(
    *,
    cache_dir: str | Path | None = None,
    source: str | Path = _SOURCE,
    swiftc: str = "/usr/bin/swiftc",
    timeout: float = 120.0,
    force: bool = False,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    """Lazily compile the bundled helper and return its cached executable.

    The source hash and machine architecture name the binary.  Compilation is
    atomic, does not invoke a shell, and places compiler module caches beside
    the output instead of mutating the source tree.
    """

    if sys.platform != "darwin":
        raise EventTapBuildError("the passive event-tap helper requires macOS")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("build timeout must be finite and positive")
    source_path = Path(source)
    try:
        source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise EventTapBuildError(f"could not read EventTap.swift: {exc}") from exc
    digest = hashlib.sha256(
        source_bytes + platform.machine().encode("ascii", "replace")
    ).hexdigest()[:16]
    destination_dir = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    destination = destination_dir / f"fertig-event-tap-{digest}"
    if destination.is_file() and os.access(destination, os.X_OK) and not force:
        return destination

    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
        module_cache = destination_dir / "swift-module-cache"
        module_cache.mkdir(exist_ok=True)
        temporary_dir = Path(
            tempfile.mkdtemp(prefix="build-", dir=str(destination_dir))
        )
    except OSError as exc:
        raise EventTapBuildError(f"could not create helper cache: {exc}") from exc
    temporary = temporary_dir / destination.name
    environment = os.environ.copy()
    environment["CLANG_MODULE_CACHE_PATH"] = str(module_cache)
    environment["SWIFT_MODULECACHE_PATH"] = str(module_cache)
    attempts: list[tuple[list[str], subprocess.CompletedProcess[str]]] = []
    sdk_options: tuple[Path | None, ...] = (None, *_candidate_sdk_paths(swiftc))
    result: subprocess.CompletedProcess[str] | None = None
    for sdk in sdk_options:
        command = [swiftc, "-O"]
        if sdk is not None:
            command.extend(("-sdk", str(sdk)))
        command.extend((str(source_path), "-o", str(temporary)))
        try:
            result = runner(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=environment,
            )
        except FileNotFoundError as exc:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise EventTapBuildError(
                "swiftc was not found; install matching Apple Command Line Tools"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise EventTapBuildError("building the event-tap helper timed out") from exc
        except OSError as exc:
            shutil.rmtree(temporary_dir, ignore_errors=True)
            raise EventTapBuildError(f"could not launch swiftc: {exc}") from exc
        attempts.append((command, result))
        if result.returncode == 0 and temporary.is_file():
            break
        if not _compiler_mismatch(result.stderr or result.stdout or ""):
            break

    try:
        if result is None:
            raise EventTapBuildError(
                "no macOS SDK was available to build the event helper"
            )
        if result.returncode != 0 or not temporary.is_file():
            detail = (result.stderr or result.stdout or "unknown swiftc error").strip()
            attempted_sdks = [
                command[command.index("-sdk") + 1]
                for command, _ in attempts
                if "-sdk" in command
            ]
            suffix = f" Attempted SDKs: {attempted_sdks}." if attempted_sdks else ""
            raise EventTapBuildError(
                "could not build the passive event helper; install a Command Line "
                f"Tools/SDK pair from the same Xcode release.{suffix} Detail: {detail}"
            )
        temporary.chmod(0o755)
        os.replace(temporary, destination)
    except OSError as exc:
        raise EventTapBuildError(f"could not cache event-tap helper: {exc}") from exc
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return destination


def check_macos_permissions(
    *,
    cache_dir: str | Path | None = None,
    swiftc: str = "/usr/bin/swiftc",
    timeout: float = 120.0,
) -> MacOSPermissionStatus:
    """Build the helper and inspect permissions without opening an event tap."""

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("permission timeout must be finite and positive")
    helper = build_event_tap(cache_dir=cache_dir, swiftc=swiftc, timeout=timeout)
    try:
        result = subprocess.run(
            [str(helper), "--check"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PassiveRecordingError("permission preflight timed out") from exc
    if result.returncode:
        detail = result.stderr.strip() or f"exit status {result.returncode}"
        raise PassiveRecordingError(f"permission preflight failed: {detail}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise EventTapProtocolError("permission preflight returned no single result")
    event = parse_event_line(lines[0])
    input_monitoring = event.metadata.get("input_monitoring")
    accessibility = event.metadata.get("accessibility")
    input_sent = event.metadata.get("input_sent")
    if (
        event.kind != "status"
        or not isinstance(input_monitoring, bool)
        or not isinstance(accessibility, bool)
        or input_sent is not False
    ):
        raise EventTapProtocolError("permission preflight returned invalid fields")
    return MacOSPermissionStatus(
        helper=helper,
        input_monitoring=input_monitoring,
        accessibility=accessibility,
    )


class ConcurrentScreenSampler:
    """Continuously captures timestamped frames without blocking event intake."""

    def __init__(
        self,
        backend: ScreenshotBackend,
        *,
        sample_hz: float = 8.0,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not math.isfinite(sample_hz) or not 0.1 <= sample_hz <= 60.0:
            raise ValueError("sample_hz must be finite and within [0.1, 60]")
        self.backend = backend
        self.interval = 1.0 / sample_hz
        self.clock_ns = clock_ns
        self._samples: list[ScreenSample] = []
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def _capture(self) -> None:
        started = self.clock_ns()
        frame = self.backend.capture()
        finished = self.clock_ns()
        sample = ScreenSample((started + finished) // 2, frame)
        with self._lock:
            self._samples.append(sample)

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._capture()
            except BaseException as exc:  # surfaced on finish in the caller thread
                with self._lock:
                    self._error = exc
                self._stop.set()
                return

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("screen sampler was already started")
        self._capture()  # Permission and shape failures surface before listening.
        self._thread = threading.Thread(
            target=self._run, name="fertig-screen-sampler", daemon=True
        )
        self._thread.start()

    def finish(self, settle_seconds: float) -> Sequence[ScreenSample]:
        if self._thread is None:
            raise RuntimeError("screen sampler was not started")
        if settle_seconds > 0:
            self._stop.wait(settle_seconds)
        self._stop.set()
        self._thread.join(timeout=20.0)
        if self._thread.is_alive():
            raise PassiveRecordingError("screen capture did not stop within 20 seconds")
        with self._lock:
            error = self._error
        if error is not None:
            if isinstance(error, DesktopBackendError):
                raise error
            raise PassiveRecordingError(f"screen sampling failed: {error}") from error
        # Guarantee a post-action observation even at very low sample rates.
        self._capture()
        with self._lock:
            return tuple(self._samples)


@dataclass(frozen=True)
class _TimedAction:
    action: DesktopAction
    started_ns: int
    ended_ns: int


def _raise_helper_error(event: EventTapEvent) -> None:
    message = event.message or event.code or "unknown event-tap error"
    if event.code in {"input_monitoring_denied", "event_tap_unavailable"}:
        raise DesktopPermissionError(
            "macOS denied passive input observation. Enable Input Monitoring and "
            f"Accessibility in System Settings > Privacy & Security. Detail: {message}"
        )
    raise PassiveRecordingError(f"event-tap helper failed: {message}")


def _timed_actions(
    events: Iterable[EventTapEvent], *, text_gap_seconds: float
) -> list[_TimedAction]:
    mouse_down: dict[int, EventTapEvent] = {}
    result: list[_TimedAction] = []
    gap_ns = round(text_gap_seconds * 1_000_000_000)

    for event in events:
        if event.kind == "error":
            _raise_helper_error(event)
        if event.kind == "stop":
            break
        if event.kind in {"ready", "status", "closed"}:
            continue
        if event.kind == "mouse_down":
            mouse_down[event.button] = event
            continue
        if event.kind == "mouse_up":
            if event.button != 0:
                raise EventTapProtocolError(
                    "the current desktop action layer can replay left clicks only; "
                    f"recording contained mouse button {event.button}"
                )
            unsafe = set(event.modifiers) - {"caps_lock"}
            if unsafe:
                raise EventTapProtocolError(
                    f"modified mouse clicks are not replayable yet: {sorted(unsafe)}"
                )
            down = mouse_down.pop(event.button, None)
            started = down.timestamp_ns if down is not None else event.timestamp_ns
            x, y = round(event.x), round(event.y)
            try:
                action = DesktopAction.click_at(x, y)
            except (TypeError, ValueError) as exc:
                raise EventTapProtocolError(
                    f"mouse coordinate cannot be replayed: ({x}, {y})"
                ) from exc
            result.append(_TimedAction(action, started, event.timestamp_ns))
            continue

        if event.kind != "key_down":
            continue
        unsafe_text_modifiers = set(event.modifiers) & {
            "command",
            "control",
            "option",
            "function",
        }
        if event.text is not None and not unsafe_text_modifiers:
            action = DesktopAction.enter_text(event.text)
        elif event.keycode in KEYCODE_TO_DESKTOP_KEY:
            unsafe_key_modifiers = set(event.modifiers) - {"caps_lock"}
            if unsafe_key_modifiers:
                raise EventTapProtocolError(
                    "modified special keys are not replayable yet: "
                    f"{sorted(unsafe_key_modifiers)}"
                )
            action = DesktopAction.press_key(KEYCODE_TO_DESKTOP_KEY[event.keycode])
        elif event.text_suppressed:
            raise EventTapProtocolError(
                "printable key data was safely suppressed "
                f"({event.text_suppressed}); grant Accessibility for secure-field "
                "detection, or explicitly allow unverified text for a non-sensitive demo"
            )
        elif unsafe_text_modifiers:
            raise EventTapProtocolError(
                "keyboard shortcuts with modifiers are not replayable by the current "
                f"DesktopAction layer: {sorted(unsafe_text_modifiers)} + keycode "
                f"{event.keycode}"
            )
        else:
            raise EventTapProtocolError(
                f"keycode {event.keycode} has neither printable text nor a validated mapping"
            )

        timed = _TimedAction(action, event.timestamp_ns, event.timestamp_ns)
        if (
            action.kind == "text"
            and result
            and result[-1].action.kind == "text"
            and event.timestamp_ns - result[-1].ended_ns <= gap_ns
            and len(result[-1].action.text) + len(action.text) <= 10_000
        ):
            previous = result[-1]
            result[-1] = _TimedAction(
                DesktopAction.enter_text(previous.action.text + action.text),
                previous.started_ns,
                event.timestamp_ns,
            )
        else:
            result.append(timed)
    return result


def correlate_events(
    events: Iterable[EventTapEvent],
    samples: Sequence[ScreenSample],
    *,
    label: str | None = None,
    settle_seconds: float = 0.12,
    text_gap_seconds: float = 0.65,
) -> DesktopDemonstration:
    """Correlate passive input timestamps with the nearest screen states."""

    if not math.isfinite(settle_seconds) or not 0 <= settle_seconds <= 5:
        raise ValueError("settle_seconds must be finite and within [0, 5]")
    if not math.isfinite(text_gap_seconds) or not 0 <= text_gap_seconds <= 5:
        raise ValueError("text_gap_seconds must be finite and within [0, 5]")
    if not samples:
        raise PassiveRecordingError("at least one screen sample is required")
    ordered = sorted(samples, key=lambda item: item.timestamp_ns)
    if any(
        left.timestamp_ns == right.timestamp_ns
        for left, right in zip(ordered, ordered[1:])
    ):
        raise PassiveRecordingError("screen sample timestamps must be unique")
    timestamps = [sample.timestamp_ns for sample in ordered]
    settle_ns = round(settle_seconds * 1_000_000_000)
    actions = _timed_actions(events, text_gap_seconds=text_gap_seconds)
    steps: list[RecordedStep] = []
    for timed in actions:
        before_index = max(0, bisect.bisect_right(timestamps, timed.started_ns) - 1)
        after_index = bisect.bisect_left(timestamps, timed.ended_ns + settle_ns)
        after_index = min(after_index, len(ordered) - 1)
        before = ordered[before_index]
        after = ordered[after_index]
        steps.append(
            RecordedStep(
                action=timed.action,
                before=before.frame,
                after=after.frame,
                elapsed_seconds=max(0.0, (timed.ended_ns - timed.started_ns) / 1e9),
            )
        )
    return DesktopDemonstration(steps=steps, label=label)


def _stream_reader(stream: TextIO, output: queue.Queue[object]) -> None:
    try:
        for line in stream:
            output.put(line)
    except BaseException as exc:
        output.put(exc)
    finally:
        output.put(_EOF)


def _read_text(stream: TextIO | None) -> str:
    if stream is None:
        return ""
    try:
        return stream.read().strip()
    except (OSError, ValueError):
        return ""


class MacOSPassiveRecorder:
    """Record a human demonstration without generating any desktop input.

    ``record`` starts the helper and screen sampler, then returns when F8 is
    pressed, ``stop`` is called from another thread, the optional timeout is
    reached, or stdin delivers ``KeyboardInterrupt``.  Timeout and explicit
    stop keep the partial, valid demonstration and expose the cause through
    ``last_stop_reason``.
    """

    def __init__(
        self,
        screenshot: ScreenshotBackend | None = None,
        *,
        helper_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        swiftc: str = "/usr/bin/swiftc",
        sample_hz: float = 8.0,
        settle_seconds: float = 0.12,
        text_gap_seconds: float = 0.65,
        allow_unverified_text: bool = False,
        process_factory: Callable[..., Any] = subprocess.Popen,
        sampler_factory: Callable[[ScreenshotBackend, float], ScreenSampler]
        | None = None,
    ) -> None:
        if not math.isfinite(sample_hz) or not 0.1 <= sample_hz <= 60:
            raise ValueError("sample_hz must be finite and within [0.1, 60]")
        if not math.isfinite(settle_seconds) or not 0 <= settle_seconds <= 5:
            raise ValueError("settle_seconds must be finite and within [0, 5]")
        if not math.isfinite(text_gap_seconds) or not 0 <= text_gap_seconds <= 5:
            raise ValueError("text_gap_seconds must be finite and within [0, 5]")
        self.screenshot = (
            screenshot if screenshot is not None else MacOSScreenshotBackend()
        )
        self.helper_path = Path(helper_path) if helper_path is not None else None
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.swiftc = swiftc
        self.sample_hz = sample_hz
        self.settle_seconds = settle_seconds
        self.text_gap_seconds = text_gap_seconds
        self.allow_unverified_text = bool(allow_unverified_text)
        self._process_factory = process_factory
        self._sampler_factory = sampler_factory
        self._stop_requested = threading.Event()
        self._active_lock = threading.Lock()
        self._active_process: Any | None = None
        self.last_stop_reason: str | None = None

    def _resolve_helper(self) -> Path:
        if self.helper_path is None:
            return build_event_tap(cache_dir=self.cache_dir, swiftc=self.swiftc)
        if not self.helper_path.is_file():
            raise EventTapBuildError(
                f"event-tap helper does not exist: {self.helper_path}"
            )
        if not os.access(self.helper_path, os.X_OK):
            raise EventTapBuildError(
                f"event-tap helper is not executable: {self.helper_path}"
            )
        return self.helper_path

    def _new_sampler(self) -> ScreenSampler:
        if self._sampler_factory is not None:
            return self._sampler_factory(self.screenshot, self.sample_hz)
        return ConcurrentScreenSampler(self.screenshot, sample_hz=self.sample_hz)

    def stop(self) -> None:
        """Request a graceful stop from another thread."""

        self._stop_requested.set()

    @staticmethod
    def _terminate(process: Any) -> None:
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        except (OSError, ProcessLookupError):
            pass

    def record(
        self, *, label: str | None = None, timeout: float | None = None
    ) -> DesktopDemonstration:
        """Passively record until F8/stop/timeout and return a save-ready demo."""

        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be finite and positive when provided")
        with self._active_lock:
            if self._active_process is not None:
                raise RuntimeError("this recorder is already active")
        helper = self._resolve_helper()
        command = [str(helper)]
        if self.allow_unverified_text:
            command.append("--allow-unverified-text")

        sampler = self._new_sampler()
        events: list[EventTapEvent] = []
        process: Any | None = None
        reader: threading.Thread | None = None
        messages: queue.Queue[object] = queue.Queue()
        self._stop_requested.clear()
        self.last_stop_reason = None
        sampler.start()
        failure: BaseException | None = None
        try:
            try:
                process = self._process_factory(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError as exc:
                raise EventTapBuildError(
                    f"could not launch event-tap helper: {exc}"
                ) from exc
            except OSError as exc:
                raise PassiveRecordingError(
                    f"could not launch event-tap helper: {exc}"
                ) from exc
            if process.stdout is None:
                raise PassiveRecordingError("event-tap helper stdout is unavailable")
            with self._active_lock:
                self._active_process = process
            reader = threading.Thread(
                target=_stream_reader,
                args=(process.stdout, messages),
                name="fertig-event-reader",
                daemon=True,
            )
            reader.start()
            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                if self._stop_requested.is_set():
                    self.last_stop_reason = "requested"
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    self.last_stop_reason = "timeout"
                    break
                try:
                    item = messages.get(timeout=0.05)
                except queue.Empty:
                    if process.poll() is not None and not reader.is_alive():
                        self.last_stop_reason = (
                            "requested"
                            if self._stop_requested.is_set()
                            else "helper_exit"
                        )
                        break
                    continue
                if item is _EOF:
                    self.last_stop_reason = self.last_stop_reason or (
                        "requested" if self._stop_requested.is_set() else "helper_exit"
                    )
                    break
                if isinstance(item, BaseException):
                    raise PassiveRecordingError(
                        f"could not read helper output: {item}"
                    ) from item
                event = parse_event_line(item)
                events.append(event)
                if event.kind == "error":
                    _raise_helper_error(event)
                if event.kind == "stop":
                    self.last_stop_reason = event.reason or STOP_KEY
                    break
        except KeyboardInterrupt:
            self.last_stop_reason = "keyboard_interrupt"
        except BaseException as exc:
            failure = exc
        finally:
            if process is not None:
                self._terminate(process)
            with self._active_lock:
                self._active_process = None
            if reader is not None:
                reader.join(timeout=1.0)

        try:
            samples = sampler.finish(self.settle_seconds)
        except BaseException:
            if failure is None:
                raise
            samples = ()
        if failure is not None:
            raise failure
        if process is not None:
            returncode = process.poll()
            if returncode not in (None, 0, -15) and not any(
                event.kind == "error" for event in events
            ):
                detail = _read_text(getattr(process, "stderr", None))
                if returncode == 77:
                    raise DesktopPermissionError(
                        "macOS denied Input Monitoring/Accessibility for the passive "
                        f"recorder. Detail: {detail or 'helper exit 77'}"
                    )
                raise PassiveRecordingError(
                    f"event-tap helper exited with {returncode}: {detail or 'no detail'}"
                )
        return correlate_events(
            events,
            samples,
            label=label,
            settle_seconds=self.settle_seconds,
            text_gap_seconds=self.text_gap_seconds,
        )


__all__ = [
    "ConcurrentScreenSampler",
    "EVENT_TYPES",
    "EventTapBuildError",
    "EventTapEvent",
    "EventTapProtocolError",
    "KEYCODE_TO_DESKTOP_KEY",
    "MODIFIERS",
    "MacOSPassiveRecorder",
    "MacOSPermissionStatus",
    "PassiveRecordingError",
    "STOP_KEY",
    "STOP_KEY_CODE",
    "ScreenSample",
    "ScreenSampler",
    "build_event_tap",
    "check_macos_permissions",
    "correlate_events",
    "parse_event_line",
]
