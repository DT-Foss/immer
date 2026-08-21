"""Deterministic, zero-OS acceptance demo for the FERTIG product stack.

The demo deliberately exercises the product-facing language surfaces instead
of calling the planner's private methods: three atomic actions are taught to a
``FertigAssistant``, composed and parameterised through natural commands,
persisted, reloaded, and finally invoked through ``FertigChat``.  A small
stateful desktop simulator only advances after the correct input, so every
successful planner step is backed by an actual visible state transition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal, Mapping, cast

import numpy as np

from fertig.assistant import AssistantReply, FertigAssistant
from fertig.chat import ChatReply, FertigChat
from fertig.desktop import (
    DesktopAction,
    DesktopDemonstration,
    DesktopEnvironment,
    RecordedStep,
)


_BACKGROUND = (17, 20, 25)
_EXAMPLE_RECIPIENT = "Ada Lovelace"
_DEFAULT_RUNTIME_RECIPIENT = "Dr. Katherine Johnson"

_OPEN_ORIGINAL = (8, 10, 34, 28)
_SEND_ORIGINAL = (80, 53, 108, 70)
_OPEN_SHIFTED = (74, 8, 100, 26)
_SEND_SHIFTED = (8, 52, 36, 69)
_TEXT_FIELD = (12, 18, 96, 44)


JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True)
class DemoCounts:
    """Auditable counts for one complete acceptance run."""

    atomic_tasks: int
    composed_workflows: int
    templates: int
    taught_steps: int
    executed_steps: int
    verified_steps: int
    persisted_stores: int


@dataclass(frozen=True)
class DemoActionReport:
    """One action that crossed the real ``DesktopEnvironment`` boundary."""

    index: int
    kind: str
    verified: bool
    x: int | None = None
    y: int | None = None
    key: str | None = None
    text: str | None = None
    seconds: float | None = None


@dataclass(frozen=True)
class DemoReplyReport:
    """A normalised assistant or chat reply produced during the demo."""

    stage: str
    command: str
    surface: Literal["assistant", "chat"]
    status: str
    intent_or_route: str
    task: str | None
    text: str
    data: Mapping[str, Any]


@dataclass(frozen=True)
class CoordinateReplayReport:
    """Outcome of replaying absolute demonstration coordinates unchanged."""

    success: bool
    completed_steps: int
    attempted_actions: int
    wrong_clicks: int
    reason: str


@dataclass(frozen=True)
class ProductDemoReport:
    """Typed outcome of the end-to-end, zero-OS FERTIG acceptance demo."""

    success: bool
    counts: DemoCounts
    actions: tuple[DemoActionReport, ...]
    replies: tuple[DemoReplyReport, ...]
    baseline: CoordinateReplayReport
    runtime_value: str
    composed_from: tuple[str, ...]
    task_store: str
    template_store: str
    reloaded_before_execution: bool

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a recursively JSON-compatible copy of the report."""

        return cast(dict[str, JSONValue], _json_value(asdict(self)))

    def to_json(self, *, indent: int | None = 2) -> str:
        """Encode the report without custom JSON encoders."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )


def _json_value(value: object) -> JSONValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"report contains a non-JSON value: {type(value).__name__}")


def _centre(box: tuple[int, int, int, int]) -> tuple[int, int]:
    return ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)


def _inside(box: tuple[int, int, int, int], x: int, y: int) -> bool:
    return box[0] <= x < box[2] and box[1] <= y < box[3]


def _base_screen(
    widgets: tuple[tuple[tuple[int, int, int, int], tuple[int, int, int], int], ...],
    *,
    decoration: int,
) -> np.ndarray:
    frame = np.empty((80, 120, 3), dtype=np.uint8)
    frame[:] = _BACKGROUND
    for (x0, y0, x1, y1), colour, glyph in widgets:
        frame[y0:y1, x0:x1] = colour
        glyph_x = min(x1 - 2, x0 + 2 + glyph)
        frame[y0 + 2 : y1 - 2, glyph_x : glyph_x + 2] = np.maximum(
            np.asarray(colour) - 55, 0
        )
    if decoration == 1:
        frame[4:9, 45:66] = (95, 55, 130)
    elif decoration == 2:
        frame[70:76, 43:72] = (50, 80, 130)
    return frame


def _composer_screen(
    send_box: tuple[int, int, int, int],
    *,
    recipient_width: int = 0,
    decoration: int,
) -> np.ndarray:
    frame = _base_screen(
        (
            (_TEXT_FIELD, (58, 62, 70), 1),
            (send_box, (50, 115, 215), 3),
        ),
        decoration=decoration,
    )
    if recipient_width:
        x0 = _TEXT_FIELD[0] + 8
        x1 = min(_TEXT_FIELD[2] - 5, x0 + recipient_width)
        frame[27:35, x0:x1] = (225, 225, 225)
        for x in range(x0 + 4, x1, 9):
            frame[29:35, x : x + 2] = (115, 125, 145)
    return frame


def _sent_screen(*, shifted: bool) -> np.ndarray:
    receipt = (65, 22, 103, 48) if shifted else (38, 24, 76, 50)
    return _base_screen(((receipt, (155, 65, 190), 5),), decoration=0)


def _demonstrations() -> tuple[DesktopDemonstration, ...]:
    initial = _base_screen(((_OPEN_ORIGINAL, (35, 170, 105), 2),), decoration=1)
    composer = _composer_screen(_SEND_ORIGINAL, decoration=1)
    addressed = _composer_screen(_SEND_ORIGINAL, recipient_width=25, decoration=1)
    sent = _sent_screen(shifted=False)
    return (
        DesktopDemonstration(
            [
                RecordedStep(
                    DesktopAction.click_at(*_centre(_OPEN_ORIGINAL)),
                    initial,
                    composer,
                    0.01,
                )
            ],
            label="open composer",
        ),
        DesktopDemonstration(
            [
                RecordedStep(
                    DesktopAction.enter_text(_EXAMPLE_RECIPIENT),
                    composer,
                    addressed,
                    0.01,
                )
            ],
            label="fill recipient",
        ),
        DesktopDemonstration(
            [
                RecordedStep(
                    DesktopAction.click_at(*_centre(_SEND_ORIGINAL)),
                    addressed,
                    sent,
                    0.01,
                )
            ],
            label="send message",
        ),
    )


class _NamedDemonstrationRecorder:
    """Passive protocol adapter returning one real named demonstration."""

    def __init__(self, demonstration: DesktopDemonstration) -> None:
        self.demonstration = demonstration
        self.requested_label: str | None = None

    def record(self, *, label: str | None = None) -> DesktopDemonstration:
        self.requested_label = label
        if label != self.demonstration.label:
            raise ValueError(
                f"requested label {label!r} does not match demonstration "
                f"{self.demonstration.label!r}"
            )
        return self.demonstration


class _ShiftedDesktop:
    """Stateful screenshot/input adapter that rejects semantically wrong input."""

    def __init__(self, runtime_value: str) -> None:
        self.runtime_value = runtime_value
        self.state = 0
        self.actions: list[DesktopAction] = []
        self.failures: list[str] = []
        self.frames = (
            _base_screen(((_OPEN_SHIFTED, (35, 170, 105), 2),), decoration=2),
            _composer_screen(_SEND_SHIFTED, decoration=2),
            _composer_screen(
                _SEND_SHIFTED,
                recipient_width=max(31, min(70, len(runtime_value) * 3)),
                decoration=2,
            ),
            _sent_screen(shifted=True),
        )

    def capture(self) -> np.ndarray:
        return self.frames[self.state].copy()

    def click(self, x: int, y: int) -> None:
        action = DesktopAction.click_at(x, y)
        self.actions.append(action)
        expected = _OPEN_SHIFTED if self.state == 0 else _SEND_SHIFTED
        if self.state in {0, 2} and _inside(expected, x, y):
            self.state += 1
            return
        self.failures.append(f"wrong click at ({x}, {y}) in state {self.state}")

    def text(self, text: str) -> None:
        action = DesktopAction.enter_text(text)
        self.actions.append(action)
        if self.state == 1 and text == self.runtime_value:
            self.state += 1
            return
        self.failures.append(f"wrong text input in state {self.state}")

    def key(self, key: str) -> None:
        self.actions.append(DesktopAction.press_key(key))
        self.failures.append(f"unexpected key input in state {self.state}")

    def wait(self, seconds: float) -> None:
        self.actions.append(DesktopAction.wait_for(seconds))
        self.failures.append(f"unexpected wait input in state {self.state}")


def _assistant_report(
    stage: str, command: str, reply: AssistantReply
) -> DemoReplyReport:
    return DemoReplyReport(
        stage,
        command,
        "assistant",
        reply.status,
        reply.intent,
        reply.task,
        reply.text,
        dict(reply.data),
    )


def _chat_report(stage: str, command: str, reply: ChatReply) -> DemoReplyReport:
    return DemoReplyReport(
        stage,
        command,
        "chat",
        reply.status,
        reply.route,
        reply.task,
        reply.text,
        dict(reply.data),
    )


def _action_report(raw: Mapping[str, Any]) -> DemoActionReport:
    action = DesktopAction.from_dict(raw["action"])
    return DemoActionReport(
        index=int(raw["index"]),
        kind=action.kind,
        verified=bool(raw["verified"]),
        x=action.x,
        y=action.y,
        key=action.key,
        text=action.text,
        seconds=None if action.seconds is None else float(action.seconds),
    )


def _coordinate_baseline(runtime_value: str) -> CoordinateReplayReport:
    desktop = _ShiftedDesktop(runtime_value)
    environment = DesktopEnvironment(desktop, desktop)
    actions = (
        DesktopAction.click_at(*_centre(_OPEN_ORIGINAL)),
        DesktopAction.enter_text(runtime_value),
        DesktopAction.click_at(*_centre(_SEND_ORIGINAL)),
    )
    for action in actions:
        environment.execute(action, observe_after=True)
    wrong_clicks = sum(
        failure.startswith("wrong click") for failure in desktop.failures
    )
    success = desktop.state == 3 and not desktop.failures
    return CoordinateReplayReport(
        success,
        desktop.state,
        len(desktop.actions),
        wrong_clicks,
        (
            "absolute coordinates happened to complete the shifted workflow"
            if success
            else "absolute demonstration coordinates target background after layout shift"
        ),
    )


def run_product_demo(
    output_dir: str | Path,
    *,
    runtime_value: str = _DEFAULT_RUNTIME_RECIPIENT,
) -> ProductDemoReport:
    """Run the shippable teach/compose/template/reload/chat acceptance path.

    ``output_dir`` must not already contain the two demo stores.  Refusing to
    replace them makes repeated acceptance runs non-destructive; callers can
    simply provide a fresh temporary or artifact directory.
    """

    if not isinstance(runtime_value, str) or not runtime_value:
        raise ValueError("runtime_value must be a non-empty string")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    task_path = root / "product-demo-tasks.json"
    template_path = root / "product-demo-templates.json"
    occupied = [path.name for path in (task_path, template_path) if path.exists()]
    if occupied:
        raise FileExistsError(
            "product demo refuses to replace existing store(s): " + ", ".join(occupied)
        )

    replies: list[DemoReplyReport] = []
    assistant = FertigAssistant(task_path, template_store=template_path)
    atomic_names = ("open composer", "fill recipient", "send message")
    demos = _demonstrations()
    for name, demonstration in zip(atomic_names, demos, strict=True):
        command = f'teach "{name}"'
        reply = assistant.handle(
            command, recorder=_NamedDemonstrationRecorder(demonstration)
        )
        replies.append(_assistant_report(f"teach:{name}", command, reply))

    composition_command = (
        'compose "deliver personal message" = "open composer" + '
        '"fill recipient" + "send message"'
    )
    composed_reply = assistant.handle(composition_command)
    replies.append(_assistant_report("compose", composition_command, composed_reply))

    template_command = (
        'template "deliver message to" from "deliver personal message" recipient=1'
    )
    template_reply = assistant.handle(template_command)
    replies.append(_assistant_report("template", template_command, template_reply))

    stores_persisted = int(task_path.is_file()) + int(template_path.is_file())
    # This new instance is the proof that execution depends only on persisted
    # task/template memories, not on state left in the teaching assistant.
    reloaded = FertigAssistant(task_path, template_store=template_path)
    chat = FertigChat(reloaded)
    shifted = _ShiftedDesktop(runtime_value)
    environment = DesktopEnvironment(shifted, shifted)
    invocation_command = f'do "deliver message to" recipient={json.dumps(runtime_value, ensure_ascii=False)}'
    execution_reply = chat.handle(invocation_command, desktop=environment)
    replies.append(_chat_report("execute", invocation_command, execution_reply))

    raw_steps = execution_reply.data.get("steps", ())
    actions = tuple(
        _action_report(raw) for raw in raw_steps if isinstance(raw, Mapping)
    )
    verified = sum(action.verified for action in actions)
    baseline = _coordinate_baseline(runtime_value)
    counts = DemoCounts(
        atomic_tasks=len(atomic_names),
        composed_workflows=1,
        templates=1,
        taught_steps=sum(len(demo.steps) for demo in demos),
        executed_steps=len(actions),
        verified_steps=verified,
        persisted_stores=stores_persisted,
    )
    management_ok = all(reply.status == "ok" for reply in replies[:-1])
    success = bool(
        management_ok
        and execution_reply.status == "ok"
        and execution_reply.route == "desktop"
        and execution_reply.data.get("success") is True
        and len(actions) == 3
        and verified == len(actions)
        and shifted.state == 3
        and not shifted.failures
        and shifted.actions
        == [
            DesktopAction.click_at(actions[0].x, actions[0].y),
            DesktopAction.enter_text(runtime_value),
            DesktopAction.click_at(actions[2].x, actions[2].y),
        ]
        and not baseline.success
        and baseline.wrong_clicks >= 1
        and stores_persisted == 2
    )
    return ProductDemoReport(
        success,
        counts,
        actions,
        tuple(replies),
        baseline,
        runtime_value,
        atomic_names,
        str(task_path),
        str(template_path),
        True,
    )


__all__ = [
    "CoordinateReplayReport",
    "DemoActionReport",
    "DemoCounts",
    "DemoReplyReport",
    "ProductDemoReport",
    "run_product_demo",
]
