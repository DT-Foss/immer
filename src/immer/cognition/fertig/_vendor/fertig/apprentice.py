"""A small continuous teach-by-showing learner for FERTIG.

The apprentice learns meanings from the visible consequences of actions:

``RGB before --opaque action--> RGB after``

It deliberately has no certificate, freeze, ledger, evaluator protocol or
hidden semantic channel.  A word denotes a measured visual displacement; a
multi-word instruction composes the learned displacements; an online world
model maps the desired consequence back to the currently available opaque
actions.  The included grid world is a runnable kernel demo for approximately
additive, state-independent effects.  Desktop and robot worlds can use the
same ``reset()``/``step(code)``/``action_codes`` boundary, but require a
state-conditioned effect model before this planner is suitable for them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import json
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np


UNKNOWN = "unknown"
RESOLVED = "resolved"


@dataclass(frozen=True)
class Transition:
    """One learner-visible action consequence."""

    before: np.ndarray
    action_code: int
    after: np.ndarray

    def __post_init__(self) -> None:
        before = np.asarray(self.before)
        after = np.asarray(self.after)
        if before.shape != after.shape or before.ndim != 3:
            raise ValueError("transition frames must have the same HxWxC shape")
        if before.shape[2] != 3:
            raise ValueError("transition frames must be RGB")


class ActionEnvironment(Protocol):
    """Small adapter boundary shared by simulator, desktop and robot worlds."""

    @property
    def action_codes(self) -> tuple[int, ...]: ...

    def reset(self) -> np.ndarray: ...

    def step(self, action_code: int) -> Transition: ...


def visual_position(frame: np.ndarray) -> np.ndarray:
    """Centroid of visible foreground, independent of colour and translation.

    The top-left pixel is the background sample, matching the simple stream
    and grounding worlds already used by FERTIG.  Returning NaNs makes an
    unobservable state explicit instead of inventing a position.
    """

    pixels = np.asarray(frame)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("frame must be an HxWx3 array")
    background = pixels[0, 0]
    mask = np.any(pixels != background, axis=2)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.array([np.nan, np.nan], dtype=np.float64)
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def visible_effect(transition: Transition) -> np.ndarray:
    """Measured (dx, dy) consequence from raw RGB only."""

    before = visual_position(transition.before)
    after = visual_position(transition.after)
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        return np.array([np.nan, np.nan], dtype=np.float64)
    return after - before


@dataclass
class _RunningEffect:
    count: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    m2: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))

    def update(self, value: np.ndarray) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean = self.mean + delta / self.count
        self.m2 = self.m2 + delta * (value - self.mean)

    @property
    def variance(self) -> np.ndarray:
        if self.count < 2:
            return np.ones(2, dtype=np.float64)
        return self.m2 / (self.count - 1)


@dataclass(frozen=True)
class EffectPrediction:
    action_code: int
    effect: Optional[tuple[float, float]]
    uncertainty: float
    observations: int


class OnlineWorldModel:
    """Online action-effect model with a tiny, inspectable sufficient state."""

    def __init__(self) -> None:
        self._effects: dict[int, _RunningEffect] = {}

    @property
    def action_codes(self) -> tuple[int, ...]:
        return tuple(sorted(self._effects))

    def observe(self, transition: Transition) -> bool:
        effect = visible_effect(transition)
        if not np.all(np.isfinite(effect)):
            return False
        self._effects.setdefault(int(transition.action_code), _RunningEffect()).update(
            effect
        )
        return True

    def effect(self, action_code: int) -> Optional[np.ndarray]:
        stats = self._effects.get(int(action_code))
        if stats is None or stats.count == 0:
            return None
        return stats.mean.copy()

    def uncertainty(self, action_code: int) -> float:
        stats = self._effects.get(int(action_code))
        if stats is None or stats.count == 0:
            return 1.0
        spread = float(np.sqrt(np.maximum(stats.variance, 0.0)).mean())
        return min(1.0, 1.0 / np.sqrt(stats.count + 1.0) + spread)

    def predict(self, action_code: int) -> EffectPrediction:
        stats = self._effects.get(int(action_code))
        effect = self.effect(action_code)
        return EffectPrediction(
            int(action_code),
            None if effect is None else (float(effect[0]), float(effect[1])),
            self.uncertainty(action_code),
            0 if stats is None else stats.count,
        )

    def choose_exploration(self, action_codes: Iterable[int]) -> int:
        codes = tuple(dict.fromkeys(int(code) for code in action_codes))
        if not codes:
            raise ValueError("the environment exposes no actions")
        return max(codes, key=lambda code: (self.uncertainty(code), -code))

    def to_dict(self) -> dict:
        return {
            str(code): {
                "count": stats.count,
                "mean": stats.mean.tolist(),
                "m2": stats.m2.tolist(),
            }
            for code, stats in sorted(self._effects.items())
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "OnlineWorldModel":
        model = cls()
        for raw_code, raw in payload.items():
            stats = _RunningEffect(
                int(raw["count"]),
                np.asarray(raw["mean"], dtype=np.float64),
                np.asarray(raw["m2"], dtype=np.float64),
            )
            if (
                stats.mean.shape != (2,)
                or stats.m2.shape != (2,)
                or stats.count < 0
                or not np.all(np.isfinite(stats.mean))
                or not np.all(np.isfinite(stats.m2))
            ):
                raise ValueError("invalid saved action-effect statistics")
            model._effects[int(raw_code)] = stats
        return model


@dataclass
class _Meaning:
    count: int = 0
    mean: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))

    def update(self, effect: np.ndarray) -> None:
        self.count += 1
        self.mean = self.mean + (effect - self.mean) / self.count


@dataclass(frozen=True)
class Interpretation:
    utterance: str
    status: str
    effect: Optional[tuple[float, float]]
    tokens: tuple[str, ...]
    unknown_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionPlan:
    utterance: str
    status: str
    actions: tuple[int, ...]
    target_effect: Optional[tuple[float, float]]
    predicted_effect: Optional[tuple[float, float]]
    error: float
    reason: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    utterance: str
    status: str
    success: bool
    actions: tuple[int, ...]
    target_effect: Optional[tuple[float, float]]
    actual_effect: tuple[float, float]
    transitions: tuple[Transition, ...]
    reason: str = ""


class Apprentice:
    """Continuous world learner plus a tiny operational language."""

    def __init__(self, *, tolerance: float = 0.75) -> None:
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        self.tolerance = float(tolerance)
        self.model = OnlineWorldModel()
        self._meanings: dict[str, _Meaning] = {}

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return tuple(sorted(self._meanings))

    def observe(self, transition: Transition) -> bool:
        return self.model.observe(transition)

    def explore(
        self, environment: ActionEnvironment, steps: int = 1
    ) -> tuple[Transition, ...]:
        """Actively sample the least-known available action consequences."""

        if steps < 0:
            raise ValueError("steps must be nonnegative")
        out: list[Transition] = []
        for _ in range(steps):
            # A central restart keeps intrinsic action effects observable in
            # the demo world.  A real adapter may make reset a no-op.
            environment.reset()
            code = self.model.choose_exploration(environment.action_codes)
            transition = environment.step(code)
            self.observe(transition)
            out.append(transition)
        return tuple(out)

    def teach(self, token: str, trace: Sequence[Transition]) -> tuple[float, float]:
        """Bind a name to the cumulative raw visual consequence of a demo."""

        name = " ".join(str(token).strip().lower().split())
        if not name:
            raise ValueError("token must not be empty")
        if not trace:
            raise ValueError("teaching requires at least one transition")
        effects = [visible_effect(transition) for transition in trace]
        if not all(np.all(np.isfinite(effect)) for effect in effects):
            raise ValueError("teaching trace contains an unobservable consequence")
        cumulative = np.sum(np.stack(effects), axis=0)
        self._meanings.setdefault(name, _Meaning()).update(cumulative)
        return (float(cumulative[0]), float(cumulative[1]))

    def interpret(self, utterance: str) -> Interpretation:
        text = " ".join(str(utterance).strip().lower().split())
        if not text:
            return Interpretation(text, UNKNOWN, None, (), ())
        direct = self._meanings.get(text)
        if direct is not None:
            return Interpretation(
                text,
                RESOLVED,
                (float(direct.mean[0]), float(direct.mean[1])),
                (text,),
            )
        tokens = tuple(text.split())
        unknown = tuple(token for token in tokens if token not in self._meanings)
        if unknown:
            return Interpretation(text, UNKNOWN, None, tokens, unknown)
        effect = np.sum(
            np.stack([self._meanings[token].mean for token in tokens]), axis=0
        )
        return Interpretation(
            text, RESOLVED, (float(effect[0]), float(effect[1])), tokens
        )

    understand = interpret

    def plan(
        self,
        utterance: str,
        *,
        max_depth: int = 4,
        action_codes: Optional[Iterable[int]] = None,
    ) -> ActionPlan:
        if max_depth < 0:
            raise ValueError("max_depth must be nonnegative")
        meaning = self.interpret(utterance)
        if meaning.status != RESOLVED or meaning.effect is None:
            names = ", ".join(meaning.unknown_tokens) or "keine beobachtbare Bedeutung"
            return ActionPlan(
                meaning.utterance,
                UNKNOWN,
                (),
                None,
                None,
                float("inf"),
                f"unbekannt: {names}",
            )
        target = np.asarray(meaning.effect, dtype=np.float64)
        if np.linalg.norm(target, ord=1) <= self.tolerance:
            return ActionPlan(
                meaning.utterance,
                RESOLVED,
                (),
                meaning.effect,
                (0.0, 0.0),
                float(np.linalg.norm(target, ord=1)),
            )
        codes = (
            self.model.action_codes
            if action_codes is None
            else tuple(dict.fromkeys(int(code) for code in action_codes))
        )
        effects = {code: self.model.effect(code) for code in codes}
        effects = {
            code: effect for code, effect in effects.items() if effect is not None
        }
        if not effects:
            return ActionPlan(
                meaning.utterance,
                UNKNOWN,
                (),
                meaning.effect,
                None,
                float("inf"),
                "noch kein Aktions-Effekt gelernt",
            )
        best_actions: tuple[int, ...] = ()
        best_sum: Optional[np.ndarray] = None
        best_key = (float("inf"), float("inf"))
        ordered = tuple(sorted(effects))
        for depth in range(1, max_depth + 1):
            for candidate in product(ordered, repeat=depth):
                predicted = np.sum(
                    np.stack([effects[code] for code in candidate]), axis=0
                )
                error = float(np.linalg.norm(predicted - target, ord=1))
                key = (error, depth)
                if key < best_key:
                    best_key = key
                    best_actions = tuple(candidate)
                    best_sum = predicted
            if best_key[0] <= self.tolerance:
                break
        predicted_tuple = (
            None if best_sum is None else (float(best_sum[0]), float(best_sum[1]))
        )
        status = RESOLVED if best_key[0] <= self.tolerance else UNKNOWN
        return ActionPlan(
            meaning.utterance,
            status,
            best_actions if status == RESOLVED else (),
            meaning.effect,
            predicted_tuple,
            best_key[0],
            ""
            if status == RESOLVED
            else "kein gelernter Plan erreicht den Ziel-Effekt",
        )

    def execute(
        self,
        environment: ActionEnvironment,
        utterance: str,
        *,
        max_depth: int = 4,
        learn: bool = True,
    ) -> ExecutionResult:
        """Execute a command closed-loop, replanning after every visible effect."""

        start = environment.reset()
        meaning = self.interpret(utterance)
        if meaning.status != RESOLVED or meaning.effect is None:
            return ExecutionResult(
                meaning.utterance,
                UNKNOWN,
                False,
                (),
                None,
                (0.0, 0.0),
                (),
                "unbekannter Befehl",
            )
        target = np.asarray(meaning.effect, dtype=np.float64)
        achieved = np.zeros(2, dtype=np.float64)
        transitions: list[Transition] = []
        actions: list[int] = []
        for _ in range(max_depth):
            if np.linalg.norm(target - achieved, ord=1) <= self.tolerance:
                break
            # Plan the remaining consequence without changing the lexicon.
            remaining_name = "__remaining__"
            previous = self._meanings.get(remaining_name)
            self._meanings[remaining_name] = _Meaning(1, target - achieved)
            try:
                plan = self.plan(
                    remaining_name,
                    max_depth=max_depth - len(actions),
                    action_codes=environment.action_codes,
                )
            finally:
                if previous is None:
                    self._meanings.pop(remaining_name, None)
                else:
                    self._meanings[remaining_name] = previous
            if plan.status != RESOLVED or not plan.actions:
                break
            transition = environment.step(plan.actions[0])
            effect = visible_effect(transition)
            if not np.all(np.isfinite(effect)):
                break
            transitions.append(transition)
            actions.append(plan.actions[0])
            achieved += effect
            if learn:
                self.observe(transition)
        actual = (
            visual_position(environment.observe()) - visual_position(start)
            if hasattr(environment, "observe")
            else achieved
        )
        if not np.all(np.isfinite(actual)):
            actual = achieved
        success = bool(np.linalg.norm(actual - target, ord=1) <= self.tolerance)
        return ExecutionResult(
            meaning.utterance,
            RESOLVED if success else UNKNOWN,
            success,
            tuple(actions),
            meaning.effect,
            (float(actual[0]), float(actual[1])),
            tuple(transitions),
            "" if success else "sichtbarer Ziel-Effekt nicht erreicht",
        )

    do = execute

    def explain(self, utterance: str) -> str:
        meaning = self.interpret(utterance)
        if meaning.status != RESOLVED or meaning.effect is None:
            missing = ", ".join(meaning.unknown_tokens) or utterance
            return f"UNKNOWN: Für {missing!r} fehlt eine Demonstration."
        parts = []
        for token in meaning.tokens:
            learned = self._meanings[token]
            parts.append(
                f"{token}={tuple(round(float(v), 3) for v in learned.mean)} "
                f"aus {learned.count} Demonstration(en)"
            )
        plan = self.plan(utterance)
        if plan.status == RESOLVED:
            action_text = (
                " -> ".join(str(code) for code in plan.actions) or "keine Aktion"
            )
        else:
            action_text = plan.reason
        return (
            f"{meaning.utterance}: sichtbarer Ziel-Effekt {meaning.effect}; "
            f"{' + '.join(parts)}; Plan: {action_text}."
        )

    def save(self, path: str | Path) -> None:
        payload = {
            "version": 1,
            "tolerance": self.tolerance,
            "model": self.model.to_dict(),
            "meanings": {
                token: {"count": meaning.count, "mean": meaning.mean.tolist()}
                for token, meaning in sorted(self._meanings.items())
            },
        }
        Path(path).write_text(
            json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "Apprentice":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise ValueError("unsupported apprentice memory version")
        apprentice = cls(tolerance=float(payload["tolerance"]))
        apprentice.model = OnlineWorldModel.from_dict(payload["model"])
        for token, raw in payload["meanings"].items():
            mean = np.asarray(raw["mean"], dtype=np.float64)
            count = int(raw["count"])
            if mean.shape != (2,) or count <= 0 or not np.all(np.isfinite(mean)):
                raise ValueError("invalid saved operational meaning")
            apprentice._meanings[str(token)] = _Meaning(count, mean)
        return apprentice


class RGBGridWorld:
    """Tiny RGB world with opaque action codes and renderer translation."""

    _DIRECTIONS = ((3, 0), (-3, 0), (0, 3), (0, -3))

    def __init__(
        self,
        seed: int = 0,
        *,
        codebook_variant: int = 0,
        renderer_offset: tuple[int, int] = (0, 0),
    ) -> None:
        self.seed = int(seed)
        self.codebook_variant = int(codebook_variant)
        self.renderer_offset = (int(renderer_offset[0]), int(renderer_offset[1]))
        rng = np.random.default_rng(17_171 + 1_009 * self.codebook_variant)
        permutation = tuple(int(i) for i in rng.permutation(4))
        codes = tuple(
            100_003 + 10_007 * self.codebook_variant + 7_919 * i for i in range(4)
        )
        self._effect_by_code = {
            codes[index]: self._DIRECTIONS[permutation[index]] for index in range(4)
        }
        self._code_by_effect = {
            effect: code for code, effect in self._effect_by_code.items()
        }
        self._position = np.array([18.0, 18.0], dtype=np.float64)
        self._tick = 0

    @property
    def action_codes(self) -> tuple[int, ...]:
        return tuple(sorted(self._effect_by_code))

    def _render(self) -> np.ndarray:
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        x = int(round(self._position[0] + self.renderer_offset[0]))
        y = int(round(self._position[1] + self.renderer_offset[1]))
        x = max(1, min(frame.shape[1] - 4, x))
        y = max(1, min(frame.shape[0] - 4, y))
        # Changing colour with tick prevents a colour lookup while the mask
        # and geometry remain stable.
        colour = (180 + 15 * (self._tick % 3), 210, 120 + 20 * (self._tick % 2))
        frame[y : y + 3, x : x + 3] = colour
        return frame

    def observe(self) -> np.ndarray:
        return self._render()

    def reset(self) -> np.ndarray:
        self._position = np.array([18.0, 18.0], dtype=np.float64)
        self._tick = 0
        return self._render()

    def step(self, action_code: int) -> Transition:
        code = int(action_code)
        if code not in self._effect_by_code:
            raise ValueError(f"unknown action code {code}")
        before = self._render()
        delta = np.asarray(self._effect_by_code[code], dtype=np.float64)
        self._position = np.clip(self._position + delta, 3.0, 33.0)
        self._tick += 1
        return Transition(before, code, self._render())

    def demonstrate(self, effect: tuple[int, int]) -> Transition:
        """Teacher-side convenience; the learner still receives only RGB."""

        self.reset()
        try:
            code = self._code_by_effect[(int(effect[0]), int(effect[1]))]
        except KeyError as exc:
            raise ValueError(f"no primitive action with effect {effect}") from exc
        return self.step(code)

    def expected_effect(self, action_code: int) -> tuple[int, int]:
        """Evaluation helper, never used by Apprentice decisions."""

        return self._effect_by_code[int(action_code)]


@dataclass(frozen=True)
class LearningCurve:
    steps: int
    task_accuracy: float
    model_accuracy: float
    mean_uncertainty: float


@dataclass(frozen=True)
class ApprenticeReport:
    seed: int
    steps: int
    initial_accuracy: float
    final_accuracy: float
    improvement: float
    composition_accuracy: float
    unknown_abstained: bool
    action_codes: tuple[int, ...]
    shuffled_control: bool
    curve: tuple[LearningCurve, ...]


_WORDS = {
    "right": (3, 0),
    "left": (-3, 0),
    "down": (0, 3),
    "up": (0, -3),
}


def _teach_primitives(apprentice: Apprentice, world: RGBGridWorld) -> None:
    for token, effect in _WORDS.items():
        apprentice.teach(token, (world.demonstrate(effect),))


def _task_accuracy(
    apprentice: Apprentice, world: RGBGridWorld, max_depth: int
) -> tuple[float, float]:
    atomic = tuple(_WORDS)
    composed = (
        "right up",
        "left down",
        "right right",
        "up up left",
        "down right right",
        "left left up down",
    )
    atomic_ok = [
        apprentice.execute(world, task, max_depth=max_depth, learn=False).success
        for task in atomic
    ]
    composed_ok = [
        apprentice.execute(world, task, max_depth=max_depth, learn=False).success
        for task in composed
    ]
    all_results = atomic_ok + composed_ok
    return float(np.mean(all_results)), float(np.mean(composed_ok))


def _model_accuracy(apprentice: Apprentice, world: RGBGridWorld) -> float:
    correct = 0
    for code in world.action_codes:
        predicted = apprentice.model.effect(code)
        expected = np.asarray(world.expected_effect(code), dtype=np.float64)
        if (
            predicted is not None
            and np.linalg.norm(predicted - expected, ord=1) <= 0.25
        ):
            correct += 1
    return correct / len(world.action_codes)


def run_apprentice_experiment(
    seed: int = 0,
    *,
    steps: int = 48,
    checkpoint_every: int = 4,
    max_depth: int = 4,
    codebook_variant: int = 0,
    shuffled_control: bool = False,
) -> ApprenticeReport:
    """Run one visible same-codebook/different-render learning curve.

    Transfer to a different opaque motor alphabet is tested separately by
    relearning only the motor effects while retaining taught meanings.
    """

    if steps < 0 or checkpoint_every <= 0 or max_depth <= 0:
        raise ValueError("steps must be >=0 and checkpoint_every/max_depth positive")
    world = RGBGridWorld(
        seed,
        codebook_variant=codebook_variant,
        renderer_offset=((seed % 3) - 1, ((seed // 3) % 3) - 1),
    )
    apprentice = Apprentice()
    _teach_primitives(apprentice, world)
    curve: list[LearningCurve] = []

    def measure(done: int) -> float:
        evaluation_world = RGBGridWorld(
            seed + 10_000,
            codebook_variant=codebook_variant,
            renderer_offset=(2, -2),
        )
        accuracy, _ = _task_accuracy(apprentice, evaluation_world, max_depth)
        uncertainty = float(
            np.mean([apprentice.model.uncertainty(code) for code in world.action_codes])
        )
        curve.append(
            LearningCurve(
                done, accuracy, _model_accuracy(apprentice, world), uncertainty
            )
        )
        return accuracy

    initial = measure(0)
    codes = world.action_codes
    for index in range(steps):
        world.reset()
        selected = apprentice.model.choose_exploration(codes)
        transition = world.step(selected)
        if shuffled_control:
            # Preserve every frame/effect but break the action-consequence
            # pairing.  This is a diagnostic baseline, not a release gate.
            label = codes[(codes.index(selected) + 1) % len(codes)]
            transition = Transition(transition.before, label, transition.after)
        apprentice.observe(transition)
        if (index + 1) % checkpoint_every == 0 or index + 1 == steps:
            measure(index + 1)
    if steps == 0:
        final = initial
    else:
        final = curve[-1].task_accuracy
    final_world = RGBGridWorld(
        seed + 20_000, codebook_variant=codebook_variant, renderer_offset=(-2, 2)
    )
    _, composition = _task_accuracy(apprentice, final_world, max_depth)
    return ApprenticeReport(
        seed,
        steps,
        initial,
        final,
        final - initial,
        composition,
        apprentice.interpret("never-seen").status == UNKNOWN,
        codes,
        bool(shuffled_control),
        tuple(curve),
    )


__all__ = [
    "UNKNOWN",
    "RESOLVED",
    "Transition",
    "ActionEnvironment",
    "visual_position",
    "visible_effect",
    "EffectPrediction",
    "OnlineWorldModel",
    "Interpretation",
    "ActionPlan",
    "ExecutionResult",
    "Apprentice",
    "RGBGridWorld",
    "LearningCurve",
    "ApprenticeReport",
    "run_apprentice_experiment",
]
