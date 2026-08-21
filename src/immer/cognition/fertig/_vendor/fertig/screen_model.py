"""State-conditioned visual action learning for teach-by-showing tasks.

The model learns *where* an action applies from raw RGB demonstrations instead
of memorising screen coordinates.  A visual anchor describes the clicked
component's appearance, shape, normalised patch and local context.  At run
time the component is found again after layout changes and the demonstrated
within-component target offset is projected onto its new bounding box.

This module deliberately stops at the action boundary: it returns an
``ActionPrimitive`` and never controls the operating system.  A desktop,
browser, simulator or robot adapter can execute that primitive and feed the
visible result back through :meth:`ScreenTaskModel.observe_result`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np


UNKNOWN = "unknown"
RESOLVED = "resolved"

_PATCH_SIDE = 10


def _rgb(frame: np.ndarray) -> np.ndarray:
    pixels = np.asarray(frame)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("screen frame must be an HxWx3 RGB array")
    if pixels.shape[0] < 3 or pixels.shape[1] < 3:
        raise ValueError("screen frame is too small")
    if not np.issubdtype(pixels.dtype, np.number):
        raise ValueError("screen frame must contain numeric pixels")
    if not np.all(np.isfinite(pixels)):
        raise ValueError("screen frame contains non-finite pixels")
    return np.clip(pixels, 0, 255).astype(np.uint8, copy=False)


@dataclass(frozen=True)
class ScreenState:
    """One observable RGB state."""

    frame: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _rgb(self.frame).copy())

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.frame.shape[0]), int(self.frame.shape[1]))


@dataclass(frozen=True)
class ActionPrimitive:
    """An adapter-neutral action.

    ``target`` is an absolute ``(x, y)`` observation coordinate.  In a
    demonstration it identifies the visual referent.  In a predicted action
    it is the newly located coordinate.  ``target_offset`` is the invariant
    fractional position inside that referent's bounding box.
    """

    kind: str
    target: tuple[float, float] | None = None
    target_offset: tuple[float, float] = (0.5, 0.5)
    payload: str | None = None

    def __post_init__(self) -> None:
        kind = str(self.kind).strip().lower()
        if not kind:
            raise ValueError("action kind must not be empty")
        object.__setattr__(self, "kind", kind)
        if self.target is not None:
            target = (float(self.target[0]), float(self.target[1]))
            if not np.all(np.isfinite(target)):
                raise ValueError("action target must be finite")
            object.__setattr__(self, "target", target)
        offset = (float(self.target_offset[0]), float(self.target_offset[1]))
        if not np.all(np.isfinite(offset)):
            raise ValueError("action target offset must be finite")
        object.__setattr__(self, "target_offset", offset)
        if self.payload is not None:
            object.__setattr__(self, "payload", str(self.payload))


@dataclass(frozen=True)
class ScreenTransition:
    """One demonstrated or executed screen action and its visible result."""

    before: ScreenState | np.ndarray
    action: ActionPrimitive
    after: ScreenState | np.ndarray
    success: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.before, ScreenState):
            object.__setattr__(self, "before", ScreenState(self.before))
        if not isinstance(self.after, ScreenState):
            object.__setattr__(self, "after", ScreenState(self.after))
        if not isinstance(self.action, ActionPrimitive):
            raise TypeError("transition action must be an ActionPrimitive")
        if self.success is not None:
            object.__setattr__(self, "success", bool(self.success))


@dataclass(frozen=True)
class TaskDemo:
    """A complete ordered demonstration of a task."""

    steps: tuple[ScreenTransition, ...]

    def __init__(self, steps: Sequence[ScreenTransition]) -> None:
        normalised = tuple(steps)
        if not normalised:
            raise ValueError("a task demonstration needs at least one step")
        if not all(isinstance(step, ScreenTransition) for step in normalised):
            raise TypeError("all demonstration steps must be ScreenTransition objects")
        object.__setattr__(self, "steps", normalised)


@dataclass(frozen=True)
class VisualAnchor:
    """Translation-invariant descriptor of one connected visual component."""

    mean_color: tuple[float, float, float]
    color_spread: tuple[float, float, float]
    aspect: float
    fill_ratio: float
    area_fraction: float
    patch: tuple[float, ...]
    context: tuple[float, ...]
    samples: int = 1

    def __post_init__(self) -> None:
        if len(self.mean_color) != 3 or len(self.color_spread) != 3:
            raise ValueError("anchor colours must have three channels")
        if len(self.patch) != _PATCH_SIDE * _PATCH_SIDE * 3:
            raise ValueError("anchor patch has the wrong size")
        if len(self.context) != 7:
            raise ValueError("anchor context has the wrong size")
        numeric = (
            *self.mean_color,
            *self.color_spread,
            self.aspect,
            self.fill_ratio,
            self.area_fraction,
            *self.patch,
            *self.context,
        )
        if not np.all(np.isfinite(numeric)) or self.samples <= 0:
            raise ValueError("invalid visual anchor")


@dataclass(frozen=True)
class ActionDecision:
    """Result of closed-loop visual replanning."""

    status: str
    action: ActionPrimitive | None
    step_index: int
    confidence: float
    reason: str = ""
    target_bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class _Component:
    bbox: tuple[int, int, int, int]
    anchor: VisualAnchor


@dataclass
class _StepModel:
    action: ActionPrimitive
    anchors: list[VisualAnchor]
    success_anchors: list[VisualAnchor]
    change_fraction: float
    change_bbox: tuple[float, float, float, float] | None
    observations: int


def _background(frame: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        (frame[0], frame[-1], frame[1:-1, 0], frame[1:-1, -1]), axis=0
    ).astype(np.float64)
    # The component-wise median tolerates decorations touching a screen edge.
    return np.median(border, axis=0)


def _nearest_resize(crop: np.ndarray, side: int = _PATCH_SIDE) -> np.ndarray:
    ys = np.linspace(0, crop.shape[0] - 1, side).round().astype(int)
    xs = np.linspace(0, crop.shape[1] - 1, side).round().astype(int)
    return crop[ys[:, None], xs[None, :]].astype(np.float64)


def _component_anchor(
    frame: np.ndarray,
    labels: np.ndarray,
    label: int,
    bbox: tuple[int, int, int, int],
) -> VisualAnchor:
    x0, y0, x1, y1 = bbox
    mask = labels[y0:y1, x0:x1] == label
    crop = frame[y0:y1, x0:x1]
    selected = crop[mask].astype(np.float64)
    height, width = crop.shape[:2]
    pad_x = max(2, width // 2)
    pad_y = max(2, height // 2)
    cx0, cy0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    cx1 = min(frame.shape[1], x1 + pad_x)
    cy1 = min(frame.shape[0], y1 + pad_y)
    surroundings = frame[cy0:cy1, cx0:cx1].astype(np.float64)
    ring = np.ones(surroundings.shape[:2], dtype=bool)
    ring[y0 - cy0 : y1 - cy0, x0 - cx0 : x1 - cx0] = False
    context_pixels = surroundings[ring]
    if not len(context_pixels):
        context_pixels = surroundings.reshape(-1, 3)
    background = _background(frame)
    occupancy = float(
        np.mean(np.linalg.norm(context_pixels - background, axis=1) > 24.0)
    )
    context = (
        *np.mean(context_pixels, axis=0).tolist(),
        *np.std(context_pixels, axis=0).tolist(),
        occupancy,
    )
    return VisualAnchor(
        tuple(float(value) for value in np.mean(selected, axis=0)),
        tuple(float(value) for value in np.std(selected, axis=0)),
        float(width / height),
        float(mask.mean()),
        float(mask.sum() / (frame.shape[0] * frame.shape[1])),
        tuple(float(value) for value in _nearest_resize(crop).reshape(-1)),
        tuple(float(value) for value in context),
    )


def _components(frame: np.ndarray) -> tuple[_Component, ...]:
    pixels = _rgb(frame)
    distance = np.linalg.norm(pixels.astype(np.float64) - _background(pixels), axis=2)
    mask = distance > 24.0
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    components: list[_Component] = []
    label = 0
    min_area = max(4, int(round(height * width * 0.0008)))
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x]:
                continue
            label += 1
            stack = [(x, y)]
            labels[y, x] = label
            points: list[tuple[int, int]] = []
            while stack:
                px, py = stack.pop()
                points.append((px, py))
                for nx, ny in (
                    (px - 1, py),
                    (px + 1, py),
                    (px, py - 1),
                    (px, py + 1),
                ):
                    if (
                        0 <= nx < width
                        and 0 <= ny < height
                        and mask[ny, nx]
                        and not labels[ny, nx]
                    ):
                        labels[ny, nx] = label
                        stack.append((nx, ny))
            if len(points) < min_area:
                labels[labels == label] = 0
                continue
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            bbox = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
            components.append(
                _Component(bbox, _component_anchor(pixels, labels, label, bbox))
            )
    return tuple(components)


def _anchor_score(left: VisualAnchor, right: VisualAnchor) -> float:
    color = np.linalg.norm(
        np.asarray(left.mean_color) - np.asarray(right.mean_color)
    ) / (255.0 * np.sqrt(3.0))
    spread = np.linalg.norm(
        np.asarray(left.color_spread) - np.asarray(right.color_spread)
    ) / (255.0 * np.sqrt(3.0))
    patch = (
        np.mean(
            np.abs(
                np.asarray(left.patch, dtype=float)
                - np.asarray(right.patch, dtype=float)
            )
        )
        / 255.0
    )
    aspect = min(1.0, abs(np.log(max(left.aspect, 1e-6) / max(right.aspect, 1e-6))))
    fill = abs(left.fill_ratio - right.fill_ratio)
    area = min(
        1.0,
        abs(np.log(max(left.area_fraction, 1e-9) / max(right.area_fraction, 1e-9)))
        / 2.0,
    )
    left_context = np.asarray(left.context)
    right_context = np.asarray(right.context)
    context = np.mean(np.abs(left_context[:6] - right_context[:6])) / 255.0
    context += abs(left_context[6] - right_context[6])
    return float(
        0.27 * color
        + 0.07 * spread
        + 0.38 * patch
        + 0.12 * aspect
        + 0.06 * fill
        + 0.04 * area
        + 0.03 * context
    )


def _merge_anchor(left: VisualAnchor, right: VisualAnchor) -> VisualAnchor:
    total = left.samples + right.samples

    def average(first: Sequence[float], second: Sequence[float]) -> tuple[float, ...]:
        values = (
            np.asarray(first) * left.samples + np.asarray(second) * right.samples
        ) / total
        return tuple(float(value) for value in values)

    return VisualAnchor(
        average(left.mean_color, right.mean_color),
        average(left.color_spread, right.color_spread),
        float((left.aspect * left.samples + right.aspect * right.samples) / total),
        float(
            (left.fill_ratio * left.samples + right.fill_ratio * right.samples) / total
        ),
        float(
            (left.area_fraction * left.samples + right.area_fraction * right.samples)
            / total
        ),
        average(left.patch, right.patch),
        average(left.context, right.context),
        total,
    )


def _find_target(
    components: Sequence[_Component], target: tuple[float, float]
) -> _Component | None:
    x, y = target
    containing = [
        component
        for component in components
        if component.bbox[0] <= x < component.bbox[2]
        and component.bbox[1] <= y < component.bbox[3]
    ]
    if containing:
        return min(
            containing,
            key=lambda component: (
                (component.bbox[2] - component.bbox[0])
                * (component.bbox[3] - component.bbox[1])
            ),
        )
    if not components:
        return None
    nearest = min(
        components,
        key=lambda component: (
            max(component.bbox[0] - x, 0.0, x - component.bbox[2]) ** 2
            + max(component.bbox[1] - y, 0.0, y - component.bbox[3]) ** 2
        ),
    )
    x0, y0, x1, y1 = nearest.bbox
    distance = np.sqrt(max(x0 - x, 0.0, x - x1) ** 2 + max(y0 - y, 0.0, y - y1) ** 2)
    return nearest if distance <= max(x1 - x0, y1 - y0) * 0.25 else None


def _target_offset(
    target: tuple[float, float], bbox: tuple[int, int, int, int]
) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    width = max(1, x1 - x0 - 1)
    height = max(1, y1 - y0 - 1)
    return ((target[0] - x0) / width, (target[1] - y0) / height)


def _project_target(
    offset: tuple[float, float], bbox: tuple[int, int, int, int]
) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (
        float(x0 + offset[0] * max(1, x1 - x0 - 1)),
        float(y0 + offset[1] * max(1, y1 - y0 - 1)),
    )


def _changed_fraction(transition: ScreenTransition) -> float:
    before = transition.before.frame.astype(np.float64)
    after = transition.after.frame.astype(np.float64)
    if before.shape != after.shape:
        return 1.0
    return float(np.mean(np.linalg.norm(after - before, axis=2) > 24.0))


def _change_bbox(
    transition: ScreenTransition,
) -> tuple[float, float, float, float] | None:
    """Return the normalised bounding box of visibly changed pixels."""

    before = transition.before.frame.astype(np.float64)
    after = transition.after.frame.astype(np.float64)
    if before.shape != after.shape:
        return None
    mask = np.linalg.norm(after - before, axis=2) > 24.0
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    height, width = mask.shape
    return (
        float(xs.min() / width),
        float(ys.min() / height),
        float((xs.max() + 1) / width),
        float((ys.max() + 1) / height),
    )


def _mean_change_bbox(
    transitions: Sequence[ScreenTransition],
) -> tuple[float, float, float, float] | None:
    boxes = [box for item in transitions if (box := _change_bbox(item)) is not None]
    if not boxes:
        return None
    return tuple(float(value) for value in np.mean(boxes, axis=0))


def _spatial_change_matches(
    expected: tuple[float, float, float, float] | None,
    transition: ScreenTransition,
) -> bool:
    """Verify a text effect by visible location, independent of text length."""

    observed = _change_bbox(transition)
    if expected is None or observed is None:
        return False
    left = max(expected[0], observed[0])
    top = max(expected[1], observed[1])
    right = min(expected[2], observed[2])
    bottom = min(expected[3], observed[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    expected_area = max(0.0, expected[2] - expected[0]) * max(
        0.0, expected[3] - expected[1]
    )
    observed_area = max(0.0, observed[2] - observed[0]) * max(
        0.0, observed[3] - observed[1]
    )
    smaller = min(expected_area, observed_area)
    larger = max(expected_area, observed_area)
    # A value may make the changed region much wider than the demonstration,
    # but a one-pixel cursor blink must not masquerade as the learned effect.
    return smaller > 0 and intersection / smaller >= 0.25 and smaller / larger >= 0.05


def _novel_components(transition: ScreenTransition) -> list[VisualAnchor]:
    before = _components(transition.before.frame)
    after = _components(transition.after.frame)
    novel: list[VisualAnchor] = []
    for candidate in after:
        score = min(
            (_anchor_score(candidate.anchor, old.anchor) for old in before),
            default=float("inf"),
        )
        if score > 0.16:
            novel.append(candidate.anchor)
    return novel


def _cluster_anchors(
    anchors: Sequence[VisualAnchor], *, threshold: float = 0.12
) -> list[VisualAnchor]:
    clusters: list[VisualAnchor] = []
    for anchor in anchors:
        if not clusters:
            clusters.append(anchor)
            continue
        index, score = min(
            enumerate(_anchor_score(anchor, cluster) for cluster in clusters),
            key=lambda item: item[1],
        )
        if score <= threshold:
            clusters[index] = _merge_anchor(clusters[index], anchor)
        else:
            clusters.append(anchor)
    return clusters


class ScreenTaskModel:
    """Learn and replay a state-dependent task from RGB demonstrations."""

    def __init__(
        self,
        *,
        match_threshold: float = 0.16,
        ambiguity_margin: float = 0.025,
    ) -> None:
        if match_threshold <= 0 or ambiguity_margin < 0:
            raise ValueError("invalid matching thresholds")
        self.match_threshold = float(match_threshold)
        self.ambiguity_margin = float(ambiguity_margin)
        self._steps: list[_StepModel] = []
        self._pending_step: int | None = None
        self._failures = 0

    @property
    def step_count(self) -> int:
        return len(self._steps)

    @property
    def fitted(self) -> bool:
        return bool(self._steps)

    def demonstrated_actions(self) -> tuple[ActionPrimitive, ...]:
        """Return the immutable learned action contract in task order.

        Predicted pointer coordinates are deliberately absent: they belong to
        the current screen and are resolved by :meth:`next_action`.  The
        returned kind, payload, and relative target offset are the stable part
        of the demonstration and can therefore be used to preflight higher
        level workflows before any external input is sent.
        """

        return tuple(
            ActionPrimitive(
                step.action.kind,
                None,
                step.action.target_offset,
                step.action.payload,
            )
            for step in self._steps
        )

    def fit(
        self,
        demos: Sequence[TaskDemo | Sequence[ScreenTransition]],
    ) -> "ScreenTaskModel":
        """Replace the model with the common ordered task in ``demos``."""

        normalised = [
            demo if isinstance(demo, TaskDemo) else TaskDemo(demo) for demo in demos
        ]
        if not normalised:
            raise ValueError("fit requires at least one task demonstration")
        lengths = {len(demo.steps) for demo in normalised}
        if len(lengths) != 1:
            raise ValueError(
                "all task demonstrations must contain the same number of steps"
            )
        learned: list[_StepModel] = []
        for step_index in range(next(iter(lengths))):
            transitions = [demo.steps[step_index] for demo in normalised]
            kinds = {transition.action.kind for transition in transitions}
            payloads = {transition.action.payload for transition in transitions}
            if len(kinds) != 1 or len(payloads) != 1:
                raise ValueError(
                    f"demonstrations disagree about action at step {step_index}"
                )
            target_anchors: list[VisualAnchor] = []
            offsets: list[tuple[float, float]] = []
            targeted = [
                transition.action.target is not None for transition in transitions
            ]
            if any(targeted) and not all(targeted):
                raise ValueError(
                    f"step {step_index} mixes targeted and targetless actions"
                )
            for transition in transitions:
                if transition.success is False:
                    raise ValueError("failed transitions cannot teach a task")
                if transition.action.target is None:
                    continue
                component = _find_target(
                    _components(transition.before.frame), transition.action.target
                )
                if component is None:
                    raise ValueError(
                        f"no visual component at target for step {step_index}"
                    )
                target_anchors.append(component.anchor)
                offsets.append(_target_offset(transition.action.target, component.bbox))
            anchors = _cluster_anchors(target_anchors)
            offset = (
                tuple(float(value) for value in np.mean(offsets, axis=0))
                if offsets
                else (0.5, 0.5)
            )
            all_novel = [
                anchor
                for transition in transitions
                for anchor in _novel_components(transition)
            ]
            success_clusters = _cluster_anchors(all_novel)
            # A success anchor must recur across demonstrations.  With one
            # demo it remains a useful hypothesis and explicit result labels
            # can still correct it online.
            min_samples = 1 if len(transitions) == 1 else len(transitions)
            success_clusters = [
                anchor for anchor in success_clusters if anchor.samples >= min_samples
            ]
            template = transitions[0].action
            learned.append(
                _StepModel(
                    ActionPrimitive(template.kind, None, offset, template.payload),
                    anchors,
                    success_clusters,
                    float(np.mean([_changed_fraction(item) for item in transitions])),
                    _mean_change_bbox(transitions),
                    len(transitions),
                )
            )
        self._steps = learned
        self._pending_step = None
        self._failures = 0
        return self

    def add_demonstrations(
        self,
        demos: Sequence[TaskDemo | Sequence[ScreenTransition]],
    ) -> "ScreenTaskModel":
        """Merge additional complete demonstrations into a fitted task.

        All demonstrations are validated before any prototype changes.  This
        lets repeated ``fertig teach`` calls improve a named skill instead of
        silently replacing the earlier examples.
        """

        normalised = [
            demo if isinstance(demo, TaskDemo) else TaskDemo(demo) for demo in demos
        ]
        if not normalised:
            raise ValueError("add_demonstrations requires at least one demonstration")
        if not self._steps:
            return self.fit(normalised)
        for demo in normalised:
            if len(demo.steps) != len(self._steps):
                raise ValueError("new demonstration has a different number of steps")
            for index, transition in enumerate(demo.steps):
                expected = self._steps[index]
                if transition.success is False:
                    raise ValueError("failed transitions cannot teach a task")
                if (
                    transition.action.kind != expected.action.kind
                    or transition.action.payload != expected.action.payload
                ):
                    raise ValueError(
                        f"new demonstration disagrees about action at step {index}"
                    )
                targeted = transition.action.target is not None
                if targeted != bool(expected.anchors):
                    raise ValueError(
                        f"new demonstration changes target mode at step {index}"
                    )
                if (
                    targeted
                    and _find_target(
                        _components(transition.before.frame), transition.action.target
                    )
                    is None
                ):
                    raise ValueError(f"no visual component at target for step {index}")
        for demo in normalised:
            for index, transition in enumerate(demo.steps):
                self._pending_step = index
                self.observe_result(transition, successful=True)
        self._pending_step = None
        return self

    def _result_matches(self, step: _StepModel, transition: ScreenTransition) -> bool:
        if transition.success is not None:
            return transition.success
        if step.action.kind == "text":
            # Parameterised text deliberately differs from the demonstrated
            # literal.  Verify that typing changed the same visible region;
            # post-state pixels and changed area may vary with value length.
            if step.change_bbox is not None:
                return _spatial_change_matches(step.change_bbox, transition)
            # Version-1 memories did not persist the spatial envelope.  Keep
            # their former changed-area verification until another successful
            # observation teaches the richer envelope.
            observed = _changed_fraction(transition)
            tolerance = max(0.005, 0.65 * step.change_fraction)
            return observed > 0 and abs(observed - step.change_fraction) <= tolerance
        after = _components(transition.after.frame)
        if step.success_anchors and any(
            min(_anchor_score(expected, component.anchor) for component in after)
            <= self.match_threshold
            for expected in step.success_anchors
        ):
            return True
        observed = _changed_fraction(transition)
        tolerance = max(0.005, 0.65 * step.change_fraction)
        return observed > 0 and abs(observed - step.change_fraction) <= tolerance

    def _progress(self, history: Sequence[ScreenTransition]) -> int:
        index = 0
        for transition in history:
            if not isinstance(transition, ScreenTransition):
                raise TypeError("history must contain ScreenTransition objects")
            if index >= len(self._steps):
                break
            if self._result_matches(self._steps[index], transition):
                index += 1
        return index

    def next_action(
        self,
        frame: ScreenState | np.ndarray,
        history: Sequence[ScreenTransition] = (),
    ) -> ActionDecision:
        """Replan the next task action against the current visible state."""

        if not self._steps:
            return ActionDecision(UNKNOWN, None, 0, 0.0, "model is not fitted")
        state = frame if isinstance(frame, ScreenState) else ScreenState(frame)
        step_index = self._progress(history)
        self._pending_step = min(step_index, len(self._steps) - 1)
        if step_index >= len(self._steps):
            return ActionDecision(
                RESOLVED, None, step_index, 1.0, "task already complete"
            )
        step = self._steps[step_index]
        if not step.anchors:
            action = ActionPrimitive(
                step.action.kind, None, step.action.target_offset, step.action.payload
            )
            return ActionDecision(RESOLVED, action, step_index, 1.0)
        components = _components(state.frame)
        if not components:
            return ActionDecision(
                UNKNOWN, None, step_index, 0.0, "visual target is missing"
            )
        scored = sorted(
            [
                (
                    min(
                        _anchor_score(anchor, component.anchor)
                        for anchor in step.anchors
                    ),
                    component,
                )
                for component in components
            ],
            key=lambda item: item[0],
        )
        best_score, best = scored[0]
        if best_score > self.match_threshold:
            return ActionDecision(
                UNKNOWN,
                None,
                step_index,
                max(0.0, 1.0 - best_score / self.match_threshold),
                "visual target is missing",
            )
        if len(scored) > 1 and scored[1][0] - best_score < self.ambiguity_margin:
            return ActionDecision(
                UNKNOWN,
                None,
                step_index,
                max(0.0, 1.0 - best_score / self.match_threshold),
                "visual target is ambiguous",
            )
        target = _project_target(step.action.target_offset, best.bbox)
        action = ActionPrimitive(
            step.action.kind,
            target,
            step.action.target_offset,
            step.action.payload,
        )
        confidence = max(0.0, min(1.0, 1.0 - best_score / self.match_threshold))
        return ActionDecision(
            RESOLVED, action, step_index, confidence, target_bbox=best.bbox
        )

    def observe_result(
        self,
        transition: ScreenTransition,
        *,
        successful: bool | None = None,
    ) -> bool:
        """Learn from execution feedback, including a human-corrected target."""

        if not self._steps:
            raise ValueError("model is not fitted")
        if not isinstance(transition, ScreenTransition):
            raise TypeError("result must be a ScreenTransition")
        index = 0 if self._pending_step is None else self._pending_step
        step = self._steps[index]
        accepted = (
            self._result_matches(step, transition)
            if successful is None
            else bool(successful)
        )
        if not accepted:
            self._failures += 1
            return False
        action = transition.action
        if action.target is not None:
            component = _find_target(
                _components(transition.before.frame), action.target
            )
            if component is None:
                raise ValueError("successful correction has no visual target")
            nearest = (
                min(
                    (_anchor_score(component.anchor, anchor), idx)
                    for idx, anchor in enumerate(step.anchors)
                )
                if step.anchors
                else None
            )
            if nearest is not None and nearest[0] <= 0.12:
                anchor_index = nearest[1]
                step.anchors[anchor_index] = _merge_anchor(
                    step.anchors[anchor_index], component.anchor
                )
            else:
                step.anchors.append(component.anchor)
            new_offset = np.asarray(_target_offset(action.target, component.bbox))
            old_offset = np.asarray(step.action.target_offset)
            count = step.observations
            offset = tuple(
                float(value)
                for value in (old_offset * count + new_offset) / (count + 1)
            )
        else:
            offset = step.action.target_offset
        # Explicit successful feedback is authoritative correction for action
        # kind/payload as well as target.
        step.action = ActionPrimitive(action.kind, None, offset, action.payload)
        novel = _novel_components(transition)
        for anchor in novel:
            if step.success_anchors:
                score, anchor_index = min(
                    (_anchor_score(anchor, known), idx)
                    for idx, known in enumerate(step.success_anchors)
                )
                if score <= 0.12:
                    step.success_anchors[anchor_index] = _merge_anchor(
                        step.success_anchors[anchor_index], anchor
                    )
                else:
                    step.success_anchors.append(anchor)
            else:
                step.success_anchors.append(anchor)
        step.change_fraction = (
            step.change_fraction * step.observations + _changed_fraction(transition)
        ) / (step.observations + 1)
        observed_bbox = _change_bbox(transition)
        if observed_bbox is not None:
            if step.change_bbox is None:
                step.change_bbox = observed_bbox
            else:
                step.change_bbox = tuple(
                    float(value)
                    for value in (
                        np.asarray(step.change_bbox) * step.observations
                        + np.asarray(observed_bbox)
                    )
                    / (step.observations + 1)
                )
        step.observations += 1
        return True

    def explain(self) -> str:
        if not self._steps:
            return "UNKNOWN: no screen task has been demonstrated."
        descriptions = []
        for index, step in enumerate(self._steps, start=1):
            if step.anchors:
                target = (
                    f"a visual target ({len(step.anchors)} appearance prototype(s)) "
                    f"at relative offset {tuple(round(v, 3) for v in step.action.target_offset)}"
                )
            else:
                target = "the current task state (no pointer target)"
            payload = (
                ""
                if step.action.payload is None
                else f" payload={step.action.payload!r}"
            )
            descriptions.append(
                f"{index}. {step.action.kind}{payload} on {target}; "
                f"learned from {step.observations} successful example(s)"
            )
        failure_text = (
            f"; {self._failures} failed result(s) rejected" if self._failures else ""
        )
        return "Screen task: " + " -> ".join(descriptions) + failure_text + "."

    @staticmethod
    def _anchor_to_dict(anchor: VisualAnchor) -> dict:
        return {
            "mean_color": list(anchor.mean_color),
            "color_spread": list(anchor.color_spread),
            "aspect": anchor.aspect,
            "fill_ratio": anchor.fill_ratio,
            "area_fraction": anchor.area_fraction,
            "patch": list(anchor.patch),
            "context": list(anchor.context),
            "samples": anchor.samples,
        }

    @staticmethod
    def _anchor_from_dict(raw: dict) -> VisualAnchor:
        return VisualAnchor(
            tuple(float(value) for value in raw["mean_color"]),
            tuple(float(value) for value in raw["color_spread"]),
            float(raw["aspect"]),
            float(raw["fill_ratio"]),
            float(raw["area_fraction"]),
            tuple(float(value) for value in raw["patch"]),
            tuple(float(value) for value in raw["context"]),
            int(raw["samples"]),
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "version": 1,
            "match_threshold": self.match_threshold,
            "ambiguity_margin": self.ambiguity_margin,
            "failures": self._failures,
            "steps": [
                {
                    "action": {
                        "kind": step.action.kind,
                        "target_offset": list(step.action.target_offset),
                        "payload": step.action.payload,
                    },
                    "anchors": [self._anchor_to_dict(item) for item in step.anchors],
                    "success_anchors": [
                        self._anchor_to_dict(item) for item in step.success_anchors
                    ],
                    "change_fraction": step.change_fraction,
                    "change_bbox": (
                        None if step.change_bbox is None else list(step.change_bbox)
                    ),
                    "observations": step.observations,
                }
                for step in self._steps
            ],
        }
        Path(path).write_text(
            json.dumps(payload, sort_keys=True, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ScreenTaskModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("unsupported screen task memory version")
        model = cls(
            match_threshold=float(payload["match_threshold"]),
            ambiguity_margin=float(payload["ambiguity_margin"]),
        )
        model._failures = int(payload.get("failures", 0))
        if model._failures < 0:
            raise ValueError("invalid screen task failure count")
        for raw in payload["steps"]:
            action_raw = raw["action"]
            step = _StepModel(
                ActionPrimitive(
                    action_raw["kind"],
                    None,
                    tuple(float(value) for value in action_raw["target_offset"]),
                    action_raw.get("payload"),
                ),
                [model._anchor_from_dict(item) for item in raw["anchors"]],
                [model._anchor_from_dict(item) for item in raw["success_anchors"]],
                float(raw["change_fraction"]),
                (
                    None
                    if raw.get("change_bbox") is None
                    else tuple(float(value) for value in raw["change_bbox"])
                ),
                int(raw["observations"]),
            )
            if (
                not np.isfinite(step.change_fraction)
                or step.change_fraction < 0
                or (
                    step.change_bbox is not None
                    and (
                        len(step.change_bbox) != 4
                        or not np.all(np.isfinite(step.change_bbox))
                        or not (
                            0 <= step.change_bbox[0] < step.change_bbox[2] <= 1
                            and 0 <= step.change_bbox[1] < step.change_bbox[3] <= 1
                        )
                    )
                )
                or step.observations <= 0
            ):
                raise ValueError("invalid saved screen task step")
            model._steps.append(step)
        return model


__all__ = [
    "UNKNOWN",
    "RESOLVED",
    "ScreenState",
    "VisualAnchor",
    "ActionPrimitive",
    "ScreenTransition",
    "TaskDemo",
    "ActionDecision",
    "ScreenTaskModel",
]
