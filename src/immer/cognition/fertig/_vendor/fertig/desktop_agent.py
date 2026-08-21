"""Persistent teach-by-showing task agent for FERTIG desktop adapters.

``DesktopAgent`` is the product-facing loop between recorded demonstrations,
the state-conditioned :mod:`fertig.screen_model`, and the safe action boundary
in :mod:`fertig.desktop`.  It executes one action at a time, captures the
visible result, verifies it, and replans.  Missing or ambiguous targets cause
an honest ``UNKNOWN`` stop before input is sent.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

from fertig.desktop import (
    DesktopAction,
    DesktopDemonstration,
    DesktopEnvironment,
    RecordedStep,
)
from fertig.screen_model import (
    RESOLVED,
    UNKNOWN,
    ActionDecision,
    ActionPrimitive,
    ScreenTaskModel,
    ScreenTransition,
    TaskDemo,
)
from fertig.skill_slots import (
    SkillTemplate,
    SkillTemplateError,
    TemplateApplicationError,
    TemplateInvocation,
    TemplateStore,
    TextSlot,
)


STORE_SCHEMA = "fertig.desktop.task-store"
STORE_VERSION = 2
MAX_COMPOSITION_DEPTH = 32


def _task_name(name: str) -> str:
    value = " ".join(str(name).strip().lower().split())
    if not value:
        raise ValueError("task name must not be empty")
    if len(value) > 256:
        raise ValueError("task name is limited to 256 characters")
    return value


def action_to_primitive(action: DesktopAction) -> ActionPrimitive:
    """Map the existing validated desktop action into the visual planner."""

    if action.kind == "click":
        return ActionPrimitive("click", (float(action.x), float(action.y)))
    if action.kind == "key":
        return ActionPrimitive("key", payload=action.key)
    if action.kind == "text":
        return ActionPrimitive("text", payload=action.text)
    return ActionPrimitive("wait", payload=str(float(action.seconds)))


def primitive_to_action(primitive: ActionPrimitive) -> DesktopAction:
    """Map a replanned primitive back to the one public desktop action API."""

    if primitive.kind == "click":
        if primitive.target is None:
            raise ValueError("click primitive has no resolved target")
        return DesktopAction.click_at(
            int(round(primitive.target[0])), int(round(primitive.target[1]))
        )
    if primitive.kind == "key":
        if primitive.payload is None:
            raise ValueError("key primitive has no key payload")
        return DesktopAction.press_key(primitive.payload)
    if primitive.kind == "text":
        if primitive.payload is None:
            raise ValueError("text primitive has no text payload")
        return DesktopAction.enter_text(primitive.payload)
    if primitive.kind == "wait":
        if primitive.payload is None:
            raise ValueError("wait primitive has no duration payload")
        try:
            seconds = float(primitive.payload)
        except ValueError as exc:
            raise ValueError("wait primitive duration is not numeric") from exc
        return DesktopAction.wait_for(seconds)
    raise ValueError(f"unsupported action primitive {primitive.kind!r}")


def _demonstrated_action(primitive: ActionPrimitive) -> DesktopAction:
    """Materialise a learned action for side-effect-free template checks."""

    if primitive.kind == "click":
        # A demonstrated model stores a relative visual offset, not a current
        # absolute coordinate.  Slots can never target clicks, so a valid
        # placeholder lets SkillTemplate validate the ordered action contract.
        return DesktopAction.click_at(0, 0)
    return primitive_to_action(primitive)


def _screen_transition(step: RecordedStep) -> ScreenTransition:
    return ScreenTransition(
        step.before,
        action_to_primitive(step.action),
        step.after,
        True,
    )


def demonstration_to_task(demo: DesktopDemonstration) -> TaskDemo:
    if not isinstance(demo, DesktopDemonstration):
        raise TypeError("teach_from_demo expects DesktopDemonstration objects")
    return TaskDemo(tuple(_screen_transition(step) for step in demo.steps))


@dataclass(frozen=True)
class AgentStep:
    index: int
    action: DesktopAction
    before_shape: tuple[int, int]
    after_shape: tuple[int, int]
    verified: bool


@dataclass(frozen=True)
class DesktopRunResult:
    task: str
    status: str
    success: bool
    steps: tuple[AgentStep, ...]
    reason: str = ""


class TaskStore:
    """Named screen-task models with optional atomic JSON persistence."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = None if path is None else Path(path)
        self._models: dict[str, ScreenTaskModel] = {}
        self._compositions: dict[str, tuple[str, ...]] = {}
        if self.path is not None and self.path.exists():
            self._load()

    def list(self) -> tuple[str, ...]:
        return tuple(sorted(set(self._models) | set(self._compositions)))

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and _task_name(name) in self.list()

    def get(self, name: str) -> ScreenTaskModel | None:
        return self._models.get(_task_name(name))

    def require(self, name: str) -> ScreenTaskModel:
        normalised = _task_name(name)
        try:
            return self._models[normalised]
        except KeyError as exc:
            raise KeyError(f"unknown desktop task {normalised!r}") from exc

    def put(self, name: str, model: ScreenTaskModel) -> None:
        if not isinstance(model, ScreenTaskModel) or not model.fitted:
            raise ValueError("a stored task must be a fitted ScreenTaskModel")
        task = _task_name(name)
        if task in self._compositions:
            raise ValueError(f"desktop task name {task!r} is already composed")
        self._models[task] = model
        self.save()

    def compose(self, name: str, tasks: Sequence[str]) -> tuple[str, ...]:
        """Create a named hierarchy from existing atomic or composed tasks.

        Compositions are immutable once named.  This keeps demonstrations and
        higher-level references stable across reloads; callers can choose a new
        name when they want a different workflow.
        """

        task = _task_name(name)
        if isinstance(tasks, (str, bytes)):
            raise TypeError("composition tasks must be a sequence of task names")
        children = tuple(_task_name(child) for child in tasks)
        if len(children) < 2:
            raise ValueError("a composed desktop task requires at least two tasks")
        if len(set(children)) != len(children):
            raise ValueError("a composed desktop task cannot contain duplicate names")
        if task in children:
            raise ValueError(f"composition cycle detected: {task!r} references itself")

        known = set(self._models) | set(self._compositions)
        unknown = tuple(child for child in children if child not in known)
        if unknown:
            formatted = ", ".join(repr(child) for child in unknown)
            raise KeyError(f"unknown desktop task reference(s): {formatted}")

        candidate = dict(self._compositions)
        candidate[task] = children
        self._validate_compositions(candidate, known | {task})
        if task in known:
            raise ValueError(f"desktop task name {task!r} already exists")
        self._compositions[task] = children
        self.save()
        return children

    def teach(
        self,
        name: str,
        demos: Sequence[DesktopDemonstration],
    ) -> ScreenTaskModel:
        demonstrations = tuple(demos)
        if not demonstrations:
            raise ValueError("teaching requires at least one demonstration")
        task = _task_name(name)
        if task in self._compositions:
            raise ValueError(f"desktop task name {task!r} is already composed")
        task_demos = tuple(demonstration_to_task(demo) for demo in demonstrations)
        model = self._models.get(task)
        if model is None:
            model = ScreenTaskModel().fit(task_demos)
        else:
            model.add_demonstrations(task_demos)
        self.put(name, model)
        return model

    def explain(self, name: str) -> str:
        task = _task_name(name)
        model = self._models.get(task)
        if model is not None:
            return model.explain()
        if task not in self._compositions:
            return f"UNKNOWN: no desktop task named {task!r}."

        lines = [f"Composed desktop task {task!r}:"]

        def describe(current: str, depth: int, ordinal: str) -> None:
            indent = "  " * depth
            if depth > MAX_COMPOSITION_DEPTH:
                lines.append(
                    f"{indent}{ordinal} UNKNOWN: maximum composition depth "
                    f"({MAX_COMPOSITION_DEPTH}) exceeded at {current!r}."
                )
                return
            current_model = self._models.get(current)
            if current_model is not None:
                lines.append(
                    f"{indent}{ordinal} atomic {current!r}: {current_model.explain()}"
                )
                return
            children = self._compositions.get(current)
            if children is None:
                lines.append(f"{indent}{ordinal} UNKNOWN reference {current!r}.")
                return
            lines.append(f"{indent}{ordinal} composed {current!r}:")
            for index, child in enumerate(children, start=1):
                describe(child, depth + 1, f"{ordinal}{index}.")

        for index, child in enumerate(self._compositions[task], start=1):
            describe(child, 1, f"{index}.")
        return "\n".join(lines)

    def correct(
        self,
        name: str,
        step: RecordedStep,
        *,
        step_index: int = 0,
    ) -> bool:
        task = _task_name(name)
        if task in self._compositions:
            raise ValueError(
                "composed desktop tasks cannot be corrected directly; "
                "correct one of their atomic leaf tasks"
            )
        model = self.require(name)
        if not 0 <= step_index < model.step_count:
            raise IndexError("correction step index is out of range")
        # Set planner progress without exposing mutable internals through the
        # public API.  A synthetic successful prefix selects the intended step.
        model._pending_step = step_index
        accepted = model.observe_result(_screen_transition(step), successful=True)
        if accepted:
            self.save()
        return accepted

    def _payload(self) -> dict:
        models: dict[str, dict] = {}
        for name, model in sorted(self._models.items()):
            # Use the model's versioned encoder without duplicating anchor
            # serialization.  The temporary path never escapes the store.
            with tempfile.TemporaryDirectory() as directory:
                model_path = Path(directory) / "model.json"
                model.save(model_path)
                models[name] = json.loads(model_path.read_text(encoding="utf-8"))
        return {
            "schema": STORE_SCHEMA,
            "version": STORE_VERSION,
            "models": models,
            "compositions": {
                name: list(tasks) for name, tasks in sorted(self._compositions.items())
            },
        }

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self._payload(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        descriptor, raw_temp = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        temp_path = Path(raw_temp)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        finally:
            temp_path.unlink(missing_ok=True)

    def _load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read desktop task store: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema") != STORE_SCHEMA:
            raise ValueError("unsupported desktop task store")
        version = payload.get("version")
        if version not in (1, STORE_VERSION):
            raise ValueError("unsupported desktop task store")
        models = payload.get("models")
        if not isinstance(models, dict):
            raise ValueError("desktop task store has no model mapping")
        loaded: dict[str, ScreenTaskModel] = {}
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            for raw_name, raw_model in models.items():
                name = _task_name(raw_name)
                model_path.write_text(
                    json.dumps(raw_model, allow_nan=False), encoding="utf-8"
                )
                model = ScreenTaskModel.load(model_path)
                if not model.fitted:
                    raise ValueError(f"stored task {name!r} has no steps")
                if name in loaded:
                    raise ValueError(f"duplicate stored task name {name!r}")
                loaded[name] = model
        raw_compositions = {} if version == 1 else payload.get("compositions", {})
        if not isinstance(raw_compositions, dict):
            raise ValueError("desktop task store has no composition mapping")
        compositions: dict[str, tuple[str, ...]] = {}
        for raw_name, raw_tasks in raw_compositions.items():
            name = _task_name(raw_name)
            if name in loaded or name in compositions:
                raise ValueError(f"duplicate stored task name {name!r}")
            if not isinstance(raw_tasks, list):
                raise ValueError(f"composition {name!r} has no task list")
            tasks = tuple(_task_name(item) for item in raw_tasks)
            if len(tasks) < 2:
                raise ValueError(
                    f"composition {name!r} requires at least two task references"
                )
            if len(set(tasks)) != len(tasks):
                raise ValueError(f"composition {name!r} has duplicate references")
            compositions[name] = tasks
        self._validate_compositions(compositions, set(loaded) | set(compositions))
        self._models = loaded
        self._compositions = compositions

    @staticmethod
    def _validate_compositions(
        compositions: dict[str, tuple[str, ...]], known: set[str]
    ) -> None:
        """Validate references, cycles, and pathological hierarchy depth."""

        for name, children in compositions.items():
            unknown = tuple(child for child in children if child not in known)
            if unknown:
                formatted = ", ".join(repr(child) for child in unknown)
                raise ValueError(
                    f"composition {name!r} has unknown reference(s): {formatted}"
                )

        visiting: list[str] = []
        complete: set[str] = set()

        def visit(name: str, depth: int) -> None:
            if depth > MAX_COMPOSITION_DEPTH:
                raise ValueError(
                    "maximum composition depth "
                    f"({MAX_COMPOSITION_DEPTH}) exceeded at {name!r}"
                )
            if name in visiting:
                start = visiting.index(name)
                cycle = " -> ".join((*visiting[start:], name))
                raise ValueError(f"composition cycle detected: {cycle}")
            if name in complete or name not in compositions:
                return
            visiting.append(name)
            for child in compositions[name]:
                visit(child, depth + 1)
            visiting.pop()
            complete.add(name)

        for name in compositions:
            visit(name, 1)

    def leaf_tasks(self, name: str) -> tuple[str, ...]:
        """Return atomic leaves in execution order with a depth safeguard."""

        task = _task_name(name)
        if task not in self._models and task not in self._compositions:
            raise KeyError(f"unknown desktop task {task!r}")
        leaves: list[str] = []

        def expand(current: str, depth: int, active: tuple[str, ...]) -> None:
            if depth > MAX_COMPOSITION_DEPTH:
                raise ValueError(
                    "maximum composition depth "
                    f"({MAX_COMPOSITION_DEPTH}) exceeded at {current!r}"
                )
            if current in active:
                cycle = " -> ".join((*active, current))
                raise ValueError(f"composition cycle detected: {cycle}")
            if current in self._models:
                leaves.append(current)
                return
            children = self._compositions.get(current)
            if children is None:
                raise ValueError(f"composition references unknown task {current!r}")
            for child in children:
                expand(child, depth + 1, (*active, current))

        expand(task, 1, ())
        return tuple(leaves)


class DesktopAgent:
    """Closed-loop task runner: observe, decide, act once, verify, replan."""

    def __init__(
        self,
        store: TaskStore | None = None,
        template_store: TemplateStore | None = None,
    ) -> None:
        self.store = TaskStore() if store is None else store
        self.template_store = (
            TemplateStore() if template_store is None else template_store
        )
        collisions = set(self.store.list()) & set(self.template_store.list())
        if collisions:
            names = ", ".join(repr(name) for name in sorted(collisions))
            raise ValueError(f"desktop task and skill template names collide: {names}")

    def teach_from_demo(
        self,
        name: str,
        demo: DesktopDemonstration | Sequence[DesktopDemonstration],
    ) -> ScreenTaskModel:
        if name in self.template_store:
            raise ValueError(
                f"desktop task name {_task_name(name)!r} is already a skill template"
            )
        demos = (demo,) if isinstance(demo, DesktopDemonstration) else tuple(demo)
        return self.store.teach(name, demos)

    def list(self) -> tuple[str, ...]:
        return self.store.list()

    def list_templates(self) -> tuple[str, ...]:
        """List parameterised skill names separately from executable tasks."""

        return self.template_store.list()

    def compose(self, name: str, tasks: Sequence[str]) -> tuple[str, ...]:
        """Persist a named workflow made from two or more learned skills."""

        if name in self.template_store:
            raise ValueError(
                f"desktop task name {_task_name(name)!r} is already a skill template"
            )
        return self.store.compose(name, tasks)

    def explain(self, name: str) -> str:
        return self.store.explain(name)

    def define_template(
        self,
        name: str,
        base_task: str,
        slots: Sequence[TextSlot],
    ) -> SkillTemplate:
        """Validate and persist a parameterised view of a learned task."""

        return self.store_template(SkillTemplate(name, base_task, slots))

    def declare_template(
        self,
        name: str,
        base_task: str,
        slots: Sequence[tuple[str, int, bool]],
    ) -> SkillTemplate:
        """Define slots by global step index and derive their examples.

        The demonstrated text remains the source of truth.  Product surfaces
        therefore never need to duplicate (and potentially drift from) the
        literal stored in the learned task.
        """

        actions = self._flattened_demonstrated_actions(base_task)
        definitions: list[TextSlot] = []
        for raw in slots:
            if not isinstance(raw, (tuple, list)) or len(raw) != 3:
                raise ValueError(
                    "template slot definitions must be (name, step, required)"
                )
            slot_name, step_index, required = raw
            if isinstance(step_index, bool) or not isinstance(step_index, int):
                raise ValueError("template slot step must be an integer")
            if not isinstance(required, bool):
                raise ValueError("template slot required flag must be boolean")
            if step_index < 0 or step_index >= len(actions):
                raise ValueError(
                    f"template slot step {step_index} is outside the base task"
                )
            action = actions[step_index]
            if action.kind != "text" or action.text is None:
                raise TemplateApplicationError(
                    f"slot step {step_index} targets {action.kind!r}, not a text action"
                )
            definitions.append(
                TextSlot(str(slot_name), step_index, action.text, required)
            )
        if not definitions:
            raise ValueError("a skill template requires at least one text slot")
        return self.define_template(name, base_task, tuple(definitions))

    def store_template(self, template: SkillTemplate) -> SkillTemplate:
        """Persist ``template`` only after its complete base contract matches."""

        if not isinstance(template, SkillTemplate):
            raise TypeError("store_template expects a SkillTemplate")
        if template.name in self.store:
            raise ValueError(
                f"skill template name {template.name!r} is already a desktop task"
            )
        actions = self._flattened_demonstrated_actions(template.base_task)
        template.apply_actions(
            actions,
            {slot.name: slot.example for slot in template.slots},
        )
        self.template_store.put(template)
        return template

    def explain_template(self, name: str) -> str:
        """Describe global slot indexes without conflating them with tasks."""

        try:
            template = self.template_store.get(name)
        except SkillTemplateError:
            template = None
        if template is None:
            return f"UNKNOWN: no skill template named {str(name).strip()!r}."
        base_kind = (
            "composed"
            if len(self.store.leaf_tasks(template.base_task)) > 1
            else "atomic"
        )
        if template.slots:
            slots = ", ".join(
                f"{slot.name!r} at global step {slot.step_index} "
                f"(example={slot.example!r}, "
                f"{'required' if slot.required else 'optional'})"
                for slot in template.slots
            )
        else:
            slots = "no dynamic text slots"
        try:
            actions = self._flattened_demonstrated_actions(template.base_task)
            template.apply_actions(
                actions,
                {slot.name: slot.example for slot in template.slots},
            )
            state = "base contract valid"
        except (KeyError, ValueError) as exc:
            state = f"base contract invalid: {exc}"
        return (
            f"Skill template {template.name!r}: {base_kind} base task "
            f"{template.base_task!r}; {slots}; {state}."
        )

    def correct(
        self,
        name: str,
        step: RecordedStep,
        *,
        step_index: int = 0,
    ) -> bool:
        return self.store.correct(name, step, step_index=step_index)

    def do(
        self,
        name: str,
        environment: DesktopEnvironment,
        *,
        max_steps: int | None = None,
    ) -> DesktopRunResult:
        task = _task_name(name)
        return self._run_task(task, environment, max_steps=max_steps)

    def prepare_template(
        self,
        name: str,
        values: Mapping[str, str] | Iterable[tuple[str, str]],
    ) -> TemplateInvocation:
        """Bind values and verify the current flattened base without input."""

        template = self.template_store.require(name)
        invocation = template.invoke(values)
        return self._preflight_invocation(invocation)

    def resolve_template(self, utterance: str) -> TemplateInvocation:
        """Resolve ``template slot=value`` text and preflight its base task."""

        return self._preflight_invocation(self.template_store.resolve(utterance))

    def do_template(
        self,
        name: str,
        values: Mapping[str, str] | Iterable[tuple[str, str]],
        environment: DesktopEnvironment,
        *,
        max_steps: int | None = None,
    ) -> DesktopRunResult:
        """Run a named template; validation failures return before OS input."""

        if not isinstance(environment, DesktopEnvironment):
            raise TypeError("do_template expects a DesktopEnvironment")
        try:
            invocation = self.prepare_template(name, values)
        except (KeyError, ValueError) as exc:
            result_name = " ".join(str(name).strip().casefold().split())
            return DesktopRunResult(
                result_name or "<template>",
                UNKNOWN,
                False,
                (),
                f"template preflight failed: {exc}",
            )
        return self._run_task(
            invocation.base_task,
            environment,
            max_steps=max_steps,
            invocation=invocation,
            result_task=invocation.template_name,
        )

    def do_template_utterance(
        self,
        utterance: str,
        environment: DesktopEnvironment,
        *,
        max_steps: int | None = None,
    ) -> DesktopRunResult:
        """Resolve and run explicit ``template slot=value`` natural text."""

        if not isinstance(environment, DesktopEnvironment):
            raise TypeError("do_template_utterance expects a DesktopEnvironment")
        try:
            invocation = self.resolve_template(utterance)
        except (KeyError, ValueError) as exc:
            return DesktopRunResult(
                str(utterance).strip(),
                UNKNOWN,
                False,
                (),
                f"template preflight failed: {exc}",
            )
        return self._run_task(
            invocation.base_task,
            environment,
            max_steps=max_steps,
            invocation=invocation,
            result_task=invocation.template_name,
        )

    def invoke_template(
        self,
        invocation: TemplateInvocation,
        environment: DesktopEnvironment,
        *,
        max_steps: int | None = None,
    ) -> DesktopRunResult:
        """Run an invocation after revalidating it against the owned store."""

        if not isinstance(invocation, TemplateInvocation):
            raise TypeError("invoke_template expects a TemplateInvocation")
        if not isinstance(environment, DesktopEnvironment):
            raise TypeError("invoke_template expects a DesktopEnvironment")
        try:
            validated = self._preflight_invocation(invocation)
        except (KeyError, ValueError) as exc:
            return DesktopRunResult(
                invocation.template_name,
                UNKNOWN,
                False,
                (),
                f"template preflight failed: {exc}",
            )
        return self._run_task(
            validated.base_task,
            environment,
            max_steps=max_steps,
            invocation=validated,
            result_task=validated.template_name,
        )

    def _preflight_invocation(
        self, invocation: TemplateInvocation
    ) -> TemplateInvocation:
        stored = self.template_store.require(invocation.template_name)
        if invocation.base_task != stored.base_task:
            raise TemplateApplicationError(
                f"template base changed from {invocation.base_task!r} "
                f"to {stored.base_task!r}"
            )
        # Rebinding makes a hand-built or stale invocation obey the currently
        # persisted schema.  Comparing the resolved map rejects tampering.
        validated = stored.invoke(invocation.values)
        if dict(validated.step_text) != dict(invocation.step_text):
            raise TemplateApplicationError("invocation does not match stored slots")
        actions = self._flattened_demonstrated_actions(validated.base_task)
        validated.apply_actions(actions)
        return validated

    def _flattened_demonstrated_actions(self, task: str) -> tuple[DesktopAction, ...]:
        return tuple(
            _demonstrated_action(action)
            for leaf in self.store.leaf_tasks(task)
            for action in self.store.require(leaf).demonstrated_actions()
        )

    def _run_task(
        self,
        task: str,
        environment: DesktopEnvironment,
        *,
        max_steps: int | None,
        invocation: TemplateInvocation | None = None,
        result_task: str | None = None,
    ) -> DesktopRunResult:
        task = _task_name(task)
        public_task = task if result_task is None else _task_name(result_task)
        if task not in self.store:
            return DesktopRunResult(
                public_task, UNKNOWN, False, (), f"unknown desktop task {task!r}"
            )
        if not isinstance(environment, DesktopEnvironment):
            raise TypeError("do expects a DesktopEnvironment")
        try:
            leaves = self.store.leaf_tasks(task)
        except (KeyError, ValueError) as exc:
            return DesktopRunResult(public_task, UNKNOWN, False, (), str(exc))
        total_steps = sum(self.store.require(leaf).step_count for leaf in leaves)
        limit = total_steps if max_steps is None else int(max_steps)
        if limit <= 0:
            raise ValueError("max_steps must be positive")
        if len(leaves) == 1 and leaves[0] == task:
            result, _ = self._do_atomic(
                public_task,
                self.store.require(task),
                environment,
                limit=limit,
                frame=None,
                index_offset=0,
                invocation=invocation,
            )
            return result

        results: list[AgentStep] = []
        frame = environment.observe()
        remaining = limit
        for leaf in leaves:
            if remaining <= 0:
                return DesktopRunResult(
                    public_task,
                    UNKNOWN,
                    False,
                    tuple(results),
                    "execution stopped at max_steps before composed task completion",
                )
            model = self.store.require(leaf)
            leaf_result, frame = self._do_atomic(
                leaf,
                model,
                environment,
                limit=min(remaining, model.step_count),
                frame=frame,
                index_offset=len(results),
                invocation=invocation,
            )
            results.extend(leaf_result.steps)
            remaining -= len(leaf_result.steps)
            if not leaf_result.success:
                return DesktopRunResult(
                    public_task,
                    UNKNOWN,
                    False,
                    tuple(results),
                    f"leaf task {leaf!r} failed: {leaf_result.reason}",
                )
        return DesktopRunResult(public_task, RESOLVED, True, tuple(results))

    @staticmethod
    def _do_atomic(
        task: str,
        model: ScreenTaskModel,
        environment: DesktopEnvironment,
        *,
        limit: int,
        frame,
        index_offset: int,
        invocation: TemplateInvocation | None,
    ) -> tuple[DesktopRunResult, object]:
        """Run one atomic model, optionally continuing from a prior leaf frame."""

        history: list[ScreenTransition] = []
        results: list[AgentStep] = []
        if frame is None:
            frame = environment.observe()
        while len(results) < limit:
            decision: ActionDecision = model.next_action(frame, history)
            if decision.status != RESOLVED:
                return (
                    DesktopRunResult(
                        task,
                        UNKNOWN,
                        False,
                        tuple(results),
                        decision.reason or "next action is unknown",
                    ),
                    frame,
                )
            if decision.action is None:
                return (
                    DesktopRunResult(task, RESOLVED, True, tuple(results)),
                    frame,
                )
            action = primitive_to_action(decision.action)
            if invocation is not None:
                # The visual planner remains authoritative about the action
                # and its expected effect.  Only the declared text payload is
                # replaced at the last possible seam before execution.
                action = invocation.apply_action(
                    index_offset + decision.step_index, action
                )
            # Exactly one external action crosses the boundary before the
            # visible world is captured and verified.
            after = environment.execute(action, observe_after=True)
            if after is None:  # defensive: observe_after=True promises a frame
                after = environment.observe()
            transition = ScreenTransition(frame, decision.action, after, None)
            verified = model.observe_result(transition)
            results.append(
                AgentStep(
                    index_offset + decision.step_index,
                    action,
                    tuple(frame.shape[:2]),
                    tuple(after.shape[:2]),
                    verified,
                )
            )
            if not verified:
                return (
                    DesktopRunResult(
                        task,
                        UNKNOWN,
                        False,
                        tuple(results),
                        "visible result did not match the demonstrated transition",
                    ),
                    after,
                )
            history.append(ScreenTransition(frame, decision.action, after, True))
            frame = after
        final = model.next_action(frame, history)
        if final.status == RESOLVED and final.action is None:
            return (
                DesktopRunResult(task, RESOLVED, True, tuple(results)),
                frame,
            )
        return (
            DesktopRunResult(
                task,
                UNKNOWN,
                False,
                tuple(results),
                "execution stopped at max_steps before task completion",
            ),
            frame,
        )


__all__ = [
    "STORE_SCHEMA",
    "STORE_VERSION",
    "MAX_COMPOSITION_DEPTH",
    "action_to_primitive",
    "primitive_to_action",
    "demonstration_to_task",
    "AgentStep",
    "DesktopRunResult",
    "TaskStore",
    "DesktopAgent",
]
