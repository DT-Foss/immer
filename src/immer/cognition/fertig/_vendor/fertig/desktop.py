"""Small, testable bridge between demonstrations and a macOS desktop.

The module separates observation and input behind protocols.  Constructing a
real backend has no side effects; screenshots and input happen only when
``capture`` or an input method is called.  Tests and training can therefore use
the in-memory backends without granting macOS permissions.
"""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


ActionKind = Literal["click", "key", "text", "wait"]

ACTION_ALLOWLIST = frozenset({"click", "key", "text", "wait"})
KEY_ALLOWLIST = frozenset(
    {
        "return",
        "tab",
        "escape",
        "space",
        "delete",
        "up",
        "down",
        "left",
        "right",
        "home",
        "end",
        "page_up",
        "page_down",
    }
)

_KEY_CODES = {
    "return": 36,
    "tab": 48,
    "space": 49,
    "delete": 51,
    "escape": 53,
    "home": 115,
    "page_up": 116,
    "delete_forward": 117,
    "end": 119,
    "page_down": 121,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
}

SCHEMA = "fertig.desktop.demonstration"
SCHEMA_VERSION = 1


class DesktopError(RuntimeError):
    """Base error raised by desktop adapters."""


class DesktopPermissionError(DesktopError):
    """A required macOS privacy permission is missing."""


class DesktopBackendError(DesktopError):
    """A desktop backend could not perform an operation."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class DesktopAction:
    """One explicitly allowlisted desktop operation.

    Construct actions through the named factories or the constructor.  Strict
    validation rejects surplus fields, unknown keys, non-finite waits, and
    implausible coordinates before a backend receives anything.
    """

    kind: ActionKind
    x: int | None = None
    y: int | None = None
    key: str | None = None
    text: str | None = None
    seconds: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in ACTION_ALLOWLIST:
            raise ValueError(f"action kind must be one of {sorted(ACTION_ALLOWLIST)}")

        present = {
            name
            for name, value in (
                ("x", self.x),
                ("y", self.y),
                ("key", self.key),
                ("text", self.text),
                ("seconds", self.seconds),
            )
            if value is not None
        }
        expected = {
            "click": {"x", "y"},
            "key": {"key"},
            "text": {"text"},
            "wait": {"seconds"},
        }[self.kind]
        if present != expected:
            raise ValueError(
                f"{self.kind!r} action requires exactly {sorted(expected)}; "
                f"received {sorted(present)}"
            )

        if self.kind == "click":
            if not _is_int(self.x) or not _is_int(self.y):
                raise TypeError("click coordinates must be integers")
            if not (-32768 <= self.x <= 32767 and -32768 <= self.y <= 32767):
                raise ValueError("click coordinates must be within [-32768, 32767]")
        elif self.kind == "key":
            if not isinstance(self.key, str) or self.key not in KEY_ALLOWLIST:
                raise ValueError(f"key must be one of {sorted(KEY_ALLOWLIST)}")
        elif self.kind == "text":
            if not isinstance(self.text, str) or not self.text:
                raise ValueError("text must be a non-empty string")
            if len(self.text) > 10_000:
                raise ValueError("text is limited to 10,000 characters")
        else:
            if isinstance(self.seconds, bool) or not isinstance(
                self.seconds, (int, float)
            ):
                raise TypeError("wait seconds must be numeric")
            if not np.isfinite(self.seconds) or not 0.0 <= self.seconds <= 60.0:
                raise ValueError("wait seconds must be finite and within [0, 60]")

    @classmethod
    def click_at(cls, x: int, y: int) -> DesktopAction:
        return cls("click", x=x, y=y)

    @classmethod
    def press_key(cls, key: str) -> DesktopAction:
        return cls("key", key=key)

    @classmethod
    def enter_text(cls, text: str) -> DesktopAction:
        return cls("text", text=text)

    @classmethod
    def wait_for(cls, seconds: float) -> DesktopAction:
        return cls("wait", seconds=seconds)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind}
        for name in ("x", "y", "key", "text", "seconds"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DesktopAction:
        allowed = {"kind", "x", "y", "key", "text", "seconds"}
        extra = set(value) - allowed
        if extra:
            raise ValueError(f"unknown action fields: {sorted(extra)}")
        if "kind" not in value:
            raise ValueError("action is missing 'kind'")
        return cls(**dict(value))


@runtime_checkable
class ScreenshotBackend(Protocol):
    """Source of RGB screenshots."""

    def capture(self) -> np.ndarray:
        """Return an ``(height, width, 3)`` RGB array."""


@runtime_checkable
class InputBackend(Protocol):
    """Target for allowlisted desktop input."""

    def click(self, x: int, y: int) -> None: ...

    def key(self, key: str) -> None: ...

    def text(self, text: str) -> None: ...

    def wait(self, seconds: float) -> None: ...


def _normalise_frame(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3 or not all(array.shape[:2]):
        raise DesktopBackendError(
            f"screenshot must have shape (height, width, 3), got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise DesktopBackendError("screenshot must contain numeric RGB values")
    if not np.isfinite(array).all():
        raise DesktopBackendError("screenshot contains NaN or infinity")
    if np.issubdtype(array.dtype, np.floating):
        if float(array.min()) < 0.0 or float(array.max()) > 1.0:
            raise DesktopBackendError("floating RGB screenshots must be within [0, 1]")
        array = np.rint(array * 255.0).astype(np.uint8)
    elif float(array.min()) < 0.0 or float(array.max()) > 255.0:
        raise DesktopBackendError("integer RGB screenshots must be within [0, 255]")
    else:
        array = array.astype(np.uint8, copy=False)
    return np.ascontiguousarray(array).copy()


class DesktopEnvironment:
    """Adapter-friendly observe/execute interface for desktop learning."""

    def __init__(self, screenshot: ScreenshotBackend, inputs: InputBackend):
        self.screenshot = screenshot
        self.inputs = inputs

    def observe(self) -> np.ndarray:
        return _normalise_frame(self.screenshot.capture())

    def execute(
        self, action: DesktopAction, *, observe_after: bool = True
    ) -> np.ndarray | None:
        if not isinstance(action, DesktopAction):
            raise TypeError("execute expects a validated DesktopAction")
        if action.kind == "click":
            self.inputs.click(action.x, action.y)
        elif action.kind == "key":
            self.inputs.key(action.key)
        elif action.kind == "text":
            self.inputs.text(action.text)
        else:
            self.inputs.wait(float(action.seconds))
        return self.observe() if observe_after else None


class ArrayScreenshotBackend:
    """Deterministic in-memory screenshot source for tests and replay."""

    def __init__(self, frames: Sequence[np.ndarray], *, repeat_last: bool = False):
        if not frames:
            raise ValueError("at least one frame is required")
        self._frames = [_normalise_frame(frame) for frame in frames]
        self._index = 0
        self.repeat_last = repeat_last

    def capture(self) -> np.ndarray:
        if self._index >= len(self._frames):
            if not self.repeat_last:
                raise DesktopBackendError("in-memory screenshot sequence is exhausted")
            return self._frames[-1].copy()
        frame = self._frames[self._index].copy()
        self._index += 1
        return frame


class DryRunInputBackend:
    """Records actions without sending input or sleeping."""

    def __init__(self) -> None:
        self.actions: list[DesktopAction] = []

    def click(self, x: int, y: int) -> None:
        self.actions.append(DesktopAction.click_at(x, y))

    def key(self, key: str) -> None:
        self.actions.append(DesktopAction.press_key(key))

    def text(self, text: str) -> None:
        self.actions.append(DesktopAction.enter_text(text))

    def wait(self, seconds: float) -> None:
        self.actions.append(DesktopAction.wait_for(seconds))


def _read_bmp(path: Path) -> np.ndarray:
    data = path.read_bytes()
    if len(data) < 54 or data[:2] != b"BM":
        raise DesktopBackendError("macOS screenshot conversion did not produce a BMP")
    pixel_offset = struct.unpack_from("<I", data, 10)[0]
    width = struct.unpack_from("<i", data, 18)[0]
    signed_height = struct.unpack_from("<i", data, 22)[0]
    planes, bits = struct.unpack_from("<HH", data, 26)
    compression = struct.unpack_from("<I", data, 30)[0]
    if width <= 0 or signed_height == 0 or planes != 1 or bits not in (24, 32):
        raise DesktopBackendError("unsupported BMP layout returned by macOS sips")
    if compression != 0:
        raise DesktopBackendError("compressed BMP screenshots are unsupported")
    height = abs(signed_height)
    bytes_per_pixel = bits // 8
    stride = ((width * bits + 31) // 32) * 4
    required = pixel_offset + height * stride
    if len(data) < required:
        raise DesktopBackendError("truncated BMP returned by macOS sips")
    rows = np.frombuffer(
        data, dtype=np.uint8, count=height * stride, offset=pixel_offset
    )
    rows = rows.reshape(height, stride)[:, : width * bytes_per_pixel]
    pixels = rows.reshape(height, width, bytes_per_pixel)[..., :3]
    if signed_height > 0:
        pixels = pixels[::-1]
    return np.ascontiguousarray(pixels[..., ::-1])


def _permission_message(area: str, detail: str) -> str:
    return (
        f"macOS denied {area} access. Enable it for the current terminal/Python "
        f"process in System Settings > Privacy & Security > {area}. Detail: {detail}"
    )


class MacOSScreenshotBackend:
    """Real main-display screenshot backend in macOS input coordinates.

    Retina screenshots contain device pixels while ``CGEvent`` and System
    Events report logical points.  Frames are therefore downsampled to the
    main screen's AppKit frame once, so learned visual targets and replayed
    click coordinates share exactly one coordinate system.
    """

    _GEOMETRY_SCRIPT = """ObjC.import('AppKit')
const screen = $.NSScreen.mainScreen
if (!screen || Number($.NSScreen.screens.count) < 1) {
  throw new Error('no main display is available')
}
const frame = screen.frame
JSON.stringify({
  width: Number(frame.size.width),
  height: Number(frame.size.height),
  scale: Number(screen.backingScaleFactor)
})"""

    def __init__(
        self,
        *,
        screencapture: str = "/usr/sbin/screencapture",
        sips: str = "/usr/bin/sips",
        osascript: str = "/usr/bin/osascript",
        display: int = 1,
        logical_size: tuple[int, int] | None = None,
        timeout: float = 15.0,
    ) -> None:
        if not _is_int(display) or display <= 0:
            raise ValueError("display must be a positive integer")
        if logical_size is not None:
            width, height = logical_size
            if not _is_int(width) or not _is_int(height) or width <= 0 or height <= 0:
                raise ValueError("logical_size must contain positive integers")
        self.screencapture = screencapture
        self.sips = sips
        self.osascript = osascript
        self.display = display
        self._logical_size = logical_size
        self.timeout = timeout

    def _main_logical_size(self) -> tuple[int, int]:
        if self._logical_size is not None:
            return self._logical_size
        try:
            result = subprocess.run(
                [
                    self.osascript,
                    "-l",
                    "JavaScript",
                    "-e",
                    self._GEOMETRY_SCRIPT,
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise DesktopBackendError("macOS osascript was not found") from exc
        except subprocess.TimeoutExpired as exc:
            raise DesktopBackendError("macOS display geometry query timed out") from exc
        try:
            value = json.loads(result.stdout)
            width = int(round(float(value["width"])))
            height = int(round(float(value["height"])))
            scale = float(value["scale"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            detail = result.stderr.strip() or result.stdout.strip() or "no display data"
            raise DesktopBackendError(
                f"macOS could not determine logical display coordinates: {detail}"
            ) from exc
        if result.returncode or width <= 0 or height <= 0 or not np.isfinite(scale):
            detail = (
                result.stderr.strip() or f"invalid display {width}x{height} @ {scale}"
            )
            raise DesktopBackendError(
                f"macOS could not determine logical display coordinates: {detail}"
            )
        self._logical_size = (width, height)
        return self._logical_size

    @staticmethod
    def _to_logical_coordinates(
        frame: np.ndarray, logical_size: tuple[int, int]
    ) -> np.ndarray:
        width, height = logical_size
        if frame.shape[:2] == (height, width):
            return _normalise_frame(frame)
        ys = np.linspace(0, frame.shape[0] - 1, height).round().astype(int)
        xs = np.linspace(0, frame.shape[1] - 1, width).round().astype(int)
        return _normalise_frame(frame[ys[:, None], xs[None, :]])

    def capture(self) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="fertig-desktop-") as directory:
            png = Path(directory) / "screen.png"
            bmp = Path(directory) / "screen.bmp"
            try:
                shot = subprocess.run(
                    [
                        self.screencapture,
                        "-x",
                        "-D",
                        str(self.display),
                        "-t",
                        "png",
                        str(png),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise DesktopBackendError("macOS screencapture was not found") from exc
            except subprocess.TimeoutExpired as exc:
                raise DesktopBackendError("macOS screenshot timed out") from exc
            if shot.returncode or not png.exists():
                detail = shot.stderr.strip() or f"exit status {shot.returncode}"
                raise DesktopPermissionError(
                    _permission_message("Screen Recording", detail)
                )
            try:
                converted = subprocess.run(
                    [self.sips, "-s", "format", "bmp", str(png), "--out", str(bmp)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except FileNotFoundError as exc:
                raise DesktopBackendError("macOS sips was not found") from exc
            except subprocess.TimeoutExpired as exc:
                raise DesktopBackendError(
                    "macOS screenshot conversion timed out"
                ) from exc
            if converted.returncode or not bmp.exists():
                detail = (
                    converted.stderr.strip() or f"exit status {converted.returncode}"
                )
                raise DesktopBackendError(
                    f"macOS could not decode screenshot: {detail}"
                )
            return self._to_logical_coordinates(
                _read_bmp(bmp), self._main_logical_size()
            )


class MacOSInputBackend:
    """Real macOS input backend using fixed AppleScripts and argv data."""

    _CLICK_SCRIPT = """on run argv
set px to item 1 of argv as integer
set py to item 2 of argv as integer
tell application \"System Events\" to click at {px, py}
end run"""
    _KEY_SCRIPT = """on run argv
set codeValue to item 1 of argv as integer
tell application \"System Events\" to key code codeValue
end run"""
    _TEXT_SCRIPT = """on run argv
tell application \"System Events\" to keystroke (item 1 of argv)
end run"""

    def __init__(self, *, osascript: str = "/usr/bin/osascript", timeout: float = 15.0):
        self.osascript = osascript
        self.timeout = timeout

    def _run(self, script: str, *arguments: object) -> None:
        try:
            result = subprocess.run(
                [
                    self.osascript,
                    "-e",
                    script,
                    "--",
                    *(str(value) for value in arguments),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise DesktopBackendError("macOS osascript was not found") from exc
        except subprocess.TimeoutExpired as exc:
            raise DesktopBackendError("macOS input operation timed out") from exc
        if result.returncode:
            detail = result.stderr.strip() or f"exit status {result.returncode}"
            raise DesktopPermissionError(_permission_message("Accessibility", detail))

    def click(self, x: int, y: int) -> None:
        action = DesktopAction.click_at(x, y)
        self._run(self._CLICK_SCRIPT, action.x, action.y)

    def key(self, key: str) -> None:
        action = DesktopAction.press_key(key)
        self._run(self._KEY_SCRIPT, _KEY_CODES[action.key])

    def text(self, text: str) -> None:
        action = DesktopAction.enter_text(text)
        self._run(self._TEXT_SCRIPT, action.text)

    def wait(self, seconds: float) -> None:
        action = DesktopAction.wait_for(seconds)
        time.sleep(float(action.seconds))


@dataclass(frozen=True)
class RecordedStep:
    action: DesktopAction
    before: np.ndarray = field(repr=False)
    after: np.ndarray = field(repr=False)
    elapsed_seconds: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "before", _normalise_frame(self.before))
        object.__setattr__(self, "after", _normalise_frame(self.after))
        if not np.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")


@dataclass(frozen=True)
class DemonstrationPaths:
    metadata: Path
    frames: Path


@dataclass
class DesktopDemonstration:
    """Versioned sequence of observed desktop transitions."""

    steps: list[RecordedStep]
    label: str | None = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ValueError(f"unsupported demonstration version: {self.version}")
        if self.label is not None and (not self.label.strip() or len(self.label) > 256):
            raise ValueError("label must be non-empty and at most 256 characters")

    @staticmethod
    def _paths(path: str | Path) -> DemonstrationPaths:
        base = Path(path)
        if base.suffix in {".json", ".npz"}:
            base = base.with_suffix("")
        return DemonstrationPaths(base.with_suffix(".json"), base.with_suffix(".npz"))

    def save(self, path: str | Path) -> DemonstrationPaths:
        paths = self._paths(path)
        paths.metadata.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        step_metadata = []
        for index, step in enumerate(self.steps):
            before_key = f"before_{index}"
            after_key = f"after_{index}"
            arrays[before_key] = step.before
            arrays[after_key] = step.after
            step_metadata.append(
                {
                    "action": step.action.to_dict(),
                    "before": before_key,
                    "after": after_key,
                    "elapsed_seconds": step.elapsed_seconds,
                }
            )
        np.savez_compressed(paths.frames, **arrays)
        metadata = {
            "schema": SCHEMA,
            "version": self.version,
            "created_at": self.created_at,
            "label": self.label,
            "frames_file": paths.frames.name,
            "steps": step_metadata,
        }
        paths.metadata.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return paths

    @classmethod
    def load(cls, path: str | Path) -> DesktopDemonstration:
        paths = cls._paths(path)
        try:
            metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DesktopBackendError(
                f"could not read demonstration metadata: {exc}"
            ) from exc
        if metadata.get("schema") != SCHEMA:
            raise DesktopBackendError("file is not a FERTIG desktop demonstration")
        if metadata.get("version") != SCHEMA_VERSION:
            raise DesktopBackendError(
                f"unsupported demonstration version: {metadata.get('version')}"
            )
        frames_name = metadata.get("frames_file")
        if not isinstance(frames_name, str) or Path(frames_name).name != frames_name:
            raise DesktopBackendError("invalid frames_file in demonstration metadata")
        frames_path = paths.metadata.parent / frames_name
        try:
            archive = np.load(frames_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise DesktopBackendError(
                f"could not read demonstration frames: {exc}"
            ) from exc
        steps: list[RecordedStep] = []
        try:
            with archive:
                for value in metadata.get("steps", []):
                    if not isinstance(value, dict):
                        raise DesktopBackendError("invalid step metadata")
                    steps.append(
                        RecordedStep(
                            action=DesktopAction.from_dict(value["action"]),
                            before=archive[value["before"]],
                            after=archive[value["after"]],
                            elapsed_seconds=float(value["elapsed_seconds"]),
                        )
                    )
        except (KeyError, TypeError, ValueError) as exc:
            raise DesktopBackendError(f"invalid demonstration step: {exc}") from exc
        return cls(
            steps=steps,
            label=metadata.get("label"),
            created_at=metadata.get("created_at", ""),
            version=metadata["version"],
        )


class RecordingSession:
    """Records before/action/after triples from a desktop environment."""

    def __init__(self, environment: DesktopEnvironment, *, label: str | None = None):
        self.environment = environment
        self.demonstration = DesktopDemonstration([], label=label)

    @property
    def steps(self) -> list[RecordedStep]:
        return self.demonstration.steps

    def record(self, action: DesktopAction) -> RecordedStep:
        before = self.environment.observe()
        started = time.monotonic()
        self.environment.execute(action, observe_after=False)
        after = self.environment.observe()
        step = RecordedStep(action, before, after, time.monotonic() - started)
        self.steps.append(step)
        return step

    def save(self, path: str | Path) -> DemonstrationPaths:
        return self.demonstration.save(path)

    @staticmethod
    def load(path: str | Path) -> DesktopDemonstration:
        return DesktopDemonstration.load(path)


__all__ = [
    "ACTION_ALLOWLIST",
    "KEY_ALLOWLIST",
    "ArrayScreenshotBackend",
    "DemonstrationPaths",
    "DesktopAction",
    "DesktopBackendError",
    "DesktopDemonstration",
    "DesktopEnvironment",
    "DesktopError",
    "DesktopPermissionError",
    "DryRunInputBackend",
    "InputBackend",
    "MacOSInputBackend",
    "MacOSScreenshotBackend",
    "RecordedStep",
    "RecordingSession",
    "SCHEMA",
    "SCHEMA_VERSION",
    "ScreenshotBackend",
]
