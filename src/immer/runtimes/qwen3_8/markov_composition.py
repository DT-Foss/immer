"""Bounded literal/copy program induction over confirmed token episodes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeAlias


class MarkovCompositionError(ValueError):
    """A compositional token program or induction bound is invalid."""


def _tokens(value: object, *, field: str, allow_empty: bool = False) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{field} must be an integer sequence")
    result = tuple(value)
    if (not result and not allow_empty) or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in result
    ):
        raise MarkovCompositionError(f"{field} contains invalid token IDs")
    return result


@dataclass(frozen=True, slots=True)
class CompositionBounds:
    max_context_tokens: int = 64
    max_output_tokens: int = 64
    max_atoms: int = 8
    max_copy_atoms: int = 4
    max_copy_tokens: int = 8
    max_candidates: int = 256
    max_episodes: int = 64
    max_episode_pairs: int = 1024
    minimum_support: int = 2
    minimum_static_guards: int = 2

    def __post_init__(self) -> None:
        for field in (
            "max_context_tokens",
            "max_output_tokens",
            "max_atoms",
            "max_copy_atoms",
            "max_copy_tokens",
            "max_candidates",
            "max_episodes",
            "max_episode_pairs",
            "minimum_support",
            "minimum_static_guards",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise MarkovCompositionError(f"{field} must be positive")
        if self.max_context_tokens > 64:
            raise MarkovCompositionError("max_context_tokens cannot exceed 64")
        if self.max_output_tokens > 64:
            raise MarkovCompositionError("max_output_tokens cannot exceed 64")
        if self.max_atoms > 8 or self.max_copy_atoms > 4:
            raise MarkovCompositionError("composition atom bounds are too large")
        if self.max_copy_tokens > 8 or self.max_candidates > 256:
            raise MarkovCompositionError("composition content bounds are too large")
        if self.max_episodes > 256 or self.max_episode_pairs > 4096:
            raise MarkovCompositionError("composition search bounds are too large")
        if self.minimum_support < 2:
            raise MarkovCompositionError("minimum_support must be at least two")
        if self.minimum_static_guards < 2:
            raise MarkovCompositionError("minimum_static_guards must be at least two")


@dataclass(frozen=True, slots=True)
class ConfirmedTokenEpisode:
    prompt: tuple[int, ...]
    output: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", _tokens(self.prompt, field="prompt"))
        object.__setattr__(self, "output", _tokens(self.output, field="output"))

    @property
    def canonical_key(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return self.prompt, self.output


@dataclass(frozen=True, slots=True)
class LiteralAtom:
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "token_ids",
            _tokens(self.token_ids, field="literal token_ids"),
        )

    @property
    def width(self) -> int:
        return len(self.token_ids)

    @property
    def canonical_key(self) -> tuple[object, ...]:
        return ("literal", self.token_ids)


@dataclass(frozen=True, slots=True)
class RelativeCopyAtom:
    relative_start: int
    length: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.relative_start, bool)
            or not isinstance(self.relative_start, int)
            or self.relative_start >= 0
            or isinstance(self.length, bool)
            or not isinstance(self.length, int)
            or self.length <= 0
            or self.relative_start + self.length > 0
        ):
            raise MarkovCompositionError("relative copy span is invalid")

    @property
    def width(self) -> int:
        return self.length

    @property
    def canonical_key(self) -> tuple[object, ...]:
        return ("copy", self.relative_start, self.length)


CompositionAtom: TypeAlias = LiteralAtom | RelativeCopyAtom


@dataclass(frozen=True, slots=True)
class MarkovCompositionProgram:
    context_width: int
    guards: tuple[tuple[int, int], ...]
    atoms: tuple[CompositionAtom, ...]
    support: int
    total: int
    distinct_bindings: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.context_width, bool)
            or not isinstance(self.context_width, int)
            or not 1 <= self.context_width <= 64
        ):
            raise MarkovCompositionError("program context width is invalid")
        guards = tuple(self.guards)
        if (
            len(guards) < 2
            or guards != tuple(sorted(guards))
            or len({offset for offset, _token in guards}) != len(guards)
            or any(
                isinstance(offset, bool)
                or not isinstance(offset, int)
                or not -self.context_width <= offset < 0
                or isinstance(token, bool)
                or not isinstance(token, int)
                or token < 0
                for offset, token in guards
            )
        ):
            raise MarkovCompositionError("program guards are invalid")
        atoms = tuple(self.atoms)
        if (
            not atoms
            or len(atoms) > 8
            or any(
                not isinstance(atom, (LiteralAtom, RelativeCopyAtom)) for atom in atoms
            )
            or not any(isinstance(atom, RelativeCopyAtom) for atom in atoms)
            or sum(isinstance(atom, RelativeCopyAtom) for atom in atoms) > 4
            or sum(atom.width for atom in atoms) > 64
            or any(
                isinstance(atom, RelativeCopyAtom)
                and (atom.length > 8 or atom.relative_start < -self.context_width)
                for atom in atoms
            )
        ):
            raise MarkovCompositionError("program atoms are invalid")
        if (
            isinstance(self.support, bool)
            or not isinstance(self.support, int)
            or self.support < 2
            or isinstance(self.total, bool)
            or not isinstance(self.total, int)
            or self.total != self.support
            or isinstance(self.distinct_bindings, bool)
            or not isinstance(self.distinct_bindings, int)
            or not 2 <= self.distinct_bindings <= self.support
        ):
            raise MarkovCompositionError("program support is invalid")
        object.__setattr__(self, "guards", guards)
        object.__setattr__(self, "atoms", atoms)

    @property
    def confidence(self) -> float:
        return self.support / self.total

    @property
    def output_width(self) -> int:
        return sum(atom.width for atom in self.atoms)

    @property
    def copied_tokens(self) -> int:
        return sum(
            atom.length for atom in self.atoms if isinstance(atom, RelativeCopyAtom)
        )

    @property
    def canonical_key(self) -> tuple[object, ...]:
        return (
            self.context_width,
            self.guards,
            tuple(atom.canonical_key for atom in self.atoms),
        )

    def _binding(self, prompt: tuple[int, ...]) -> tuple[tuple[int, ...], ...] | None:
        if len(prompt) < self.context_width:
            return None
        result = []
        for atom in self.atoms:
            if not isinstance(atom, RelativeCopyAtom):
                continue
            start = len(prompt) + atom.relative_start
            stop = start + atom.length
            if start < len(prompt) - self.context_width or stop > len(prompt):
                return None
            result.append(prompt[start:stop])
        return tuple(result)

    def execute(self, prompt: Sequence[int]) -> tuple[int, ...] | None:
        source = _tokens(prompt, field="prompt")
        if len(source) < self.context_width:
            return None
        for relative, token in self.guards:
            if source[len(source) + relative] != token:
                return None
        output: list[int] = []
        for atom in self.atoms:
            if isinstance(atom, LiteralAtom):
                output.extend(atom.token_ids)
            else:
                start = len(source) + atom.relative_start
                stop = start + atom.length
                if start < len(source) - self.context_width or stop > len(source):
                    return None
                output.extend(source[start:stop])
        return tuple(output)

    def match(
        self,
        prompt: Sequence[int],
        confirmed_output: Sequence[int] = (),
        *,
        limit: int | None = None,
    ) -> tuple[int, ...] | None:
        confirmed = _tokens(
            confirmed_output,
            field="confirmed_output",
            allow_empty=True,
        )
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
        ):
            raise MarkovCompositionError("limit must be positive or None")
        output = self.execute(prompt)
        if output is None or len(confirmed) > len(output):
            return None
        if output[: len(confirmed)] != confirmed:
            return None
        remaining = output[len(confirmed) :]
        return remaining if limit is None else remaining[:limit]


@dataclass(frozen=True, slots=True)
class _Plan:
    atoms: tuple[CompositionAtom, ...]
    copied_mask: int
    copied_tokens: int
    copy_atoms: int

    @property
    def canonical_key(self) -> tuple[object, ...]:
        return tuple(atom.canonical_key for atom in self.atoms)


def _prepend(atom: CompositionAtom, plan: _Plan, *, mask: int = 0) -> _Plan:
    atoms = plan.atoms
    if isinstance(atom, LiteralAtom) and atoms and isinstance(atoms[0], LiteralAtom):
        atoms = (LiteralAtom((*atom.token_ids, *atoms[0].token_ids)), *atoms[1:])
    else:
        atoms = (atom, *atoms)
    return _Plan(
        atoms=atoms,
        copied_mask=plan.copied_mask | mask,
        copied_tokens=plan.copied_tokens
        + (atom.length if isinstance(atom, RelativeCopyAtom) else 0),
        copy_atoms=sum(isinstance(row, RelativeCopyAtom) for row in atoms),
    )


def _plan_rank(plan: _Plan) -> tuple[object, ...]:
    return (
        -plan.copied_tokens,
        len(plan.atoms),
        plan.copy_atoms,
        tuple(atom.canonical_key for atom in plan.atoms),
    )


def _pair_plans(
    left: ConfirmedTokenEpisode,
    right: ConfirmedTokenEpisode,
    *,
    context_width: int,
    bounds: CompositionBounds,
) -> tuple[_Plan, ...]:
    if len(left.output) != len(right.output):
        return ()
    left_prompt = left.prompt[-context_width:]
    right_prompt = right.prompt[-context_width:]
    variable_mask = sum(
        1 << index
        for index, (a, b) in enumerate(zip(left_prompt, right_prompt, strict=True))
        if a != b
    )
    if variable_mask == 0:
        return ()
    width = len(left.output)
    keep = min(bounds.max_candidates, 64)

    @lru_cache(maxsize=None)
    def solve(position: int, copies_used: int) -> tuple[_Plan, ...]:
        if position == width:
            return (_Plan((), 0, 0, 0),)
        rows: list[_Plan] = []
        literal_stop = position
        while (
            literal_stop < width
            and left.output[literal_stop] == right.output[literal_stop]
        ):
            literal_stop += 1
        if literal_stop > position:
            literal = LiteralAtom(left.output[position:literal_stop])
            for suffix in solve(literal_stop, copies_used):
                rows.append(_prepend(literal, suffix))
        if copies_used < bounds.max_copy_atoms:
            max_length = min(bounds.max_copy_tokens, width - position)
            for length in range(max_length, 0, -1):
                for start in range(context_width - length + 1):
                    left_span = left_prompt[start : start + length]
                    right_span = right_prompt[start : start + length]
                    if (
                        left_span == right_span
                        or left.output[position : position + length] != left_span
                        or right.output[position : position + length] != right_span
                    ):
                        continue
                    atom = RelativeCopyAtom(start - context_width, length)
                    mask = ((1 << length) - 1) << start
                    for suffix in solve(position + length, copies_used + 1):
                        rows.append(_prepend(atom, suffix, mask=mask))
        unique: dict[tuple[object, ...], _Plan] = {}
        for row in rows:
            if (
                len(row.atoms) > bounds.max_atoms
                or row.copy_atoms > bounds.max_copy_atoms
            ):
                continue
            key = (row.canonical_key, row.copied_mask)
            unique.setdefault(key, row)
        return tuple(sorted(unique.values(), key=_plan_rank)[:keep])

    return tuple(
        plan
        for plan in solve(0, 0)
        if plan.copy_atoms
        and variable_mask & ~plan.copied_mask == 0
        and sum(
            left_prompt[index] == right_prompt[index]
            and not (plan.copied_mask & (1 << index))
            for index in range(context_width)
        )
        >= bounds.minimum_static_guards
    )


def _program_from_plan(
    plan: _Plan,
    *,
    context_width: int,
    left: ConfirmedTokenEpisode,
    right: ConfirmedTokenEpisode,
    episodes: tuple[ConfirmedTokenEpisode, ...],
    bounds: CompositionBounds,
) -> MarkovCompositionProgram | None:
    left_suffix = left.prompt[-context_width:]
    right_suffix = right.prompt[-context_width:]
    guards = tuple(
        (index - context_width, left_suffix[index])
        for index in range(context_width)
        if not (plan.copied_mask & (1 << index))
        and left_suffix[index] == right_suffix[index]
    )
    if len(guards) < bounds.minimum_static_guards:
        return None
    compatible = []
    supporting = []
    bindings = set()
    prototype = MarkovCompositionProgram(
        context_width=context_width,
        guards=guards,
        atoms=plan.atoms,
        support=2,
        total=2,
        distinct_bindings=2,
    )
    for episode in episodes:
        predicted = prototype.execute(episode.prompt)
        if predicted is None:
            continue
        compatible.append(episode)
        if predicted == episode.output:
            supporting.append(episode)
            binding = prototype._binding(episode.prompt)
            if binding is not None:
                bindings.add(binding)
    support = len(supporting)
    total = len(compatible)
    if (
        support != total
        or support < bounds.minimum_support
        or len(bindings) < bounds.minimum_support
    ):
        return None
    return MarkovCompositionProgram(
        context_width=context_width,
        guards=guards,
        atoms=plan.atoms,
        support=support,
        total=total,
        distinct_bindings=len(bindings),
    )


def _program_rank(program: MarkovCompositionProgram) -> tuple[object, ...]:
    return (
        -program.support,
        -program.distinct_bindings,
        -program.copied_tokens,
        len(program.atoms),
        -program.context_width,
        program.canonical_key,
    )


def derive_programs(
    episodes: Sequence[ConfirmedTokenEpisode],
    bounds: CompositionBounds | None = None,
) -> tuple[MarkovCompositionProgram, ...]:
    """Induce deterministic relative-copy programs from confirmed episodes."""

    limits = bounds or CompositionBounds()
    if not isinstance(limits, CompositionBounds):
        raise TypeError("bounds must be CompositionBounds or None")
    if isinstance(episodes, (str, bytes)) or not isinstance(episodes, Sequence):
        raise TypeError("episodes must be a sequence")
    normalized = []
    for episode in episodes:
        if not isinstance(episode, ConfirmedTokenEpisode):
            raise TypeError("episodes must contain ConfirmedTokenEpisode values")
        if (
            len(episode.output) <= limits.max_output_tokens
            and episode.prompt
            and episode.output
        ):
            normalized.append(episode)
    ordered = tuple(
        sorted(normalized, key=lambda row: row.canonical_key)[: limits.max_episodes]
    )
    if len(ordered) < limits.minimum_support:
        return ()

    candidates: dict[tuple[object, ...], MarkovCompositionProgram] = {}
    pair_count = 0
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            if pair_count >= limits.max_episode_pairs:
                break
            if left.prompt == right.prompt:
                continue
            pair_count += 1
            maximum = min(
                limits.max_context_tokens,
                len(left.prompt),
                len(right.prompt),
            )
            for context_width in range(maximum, 1, -1):
                for plan in _pair_plans(
                    left,
                    right,
                    context_width=context_width,
                    bounds=limits,
                ):
                    program = _program_from_plan(
                        plan,
                        context_width=context_width,
                        left=left,
                        right=right,
                        episodes=ordered,
                        bounds=limits,
                    )
                    if program is not None:
                        candidates.setdefault(program.canonical_key, program)
                    if len(candidates) >= limits.max_candidates * 4:
                        break
                if len(candidates) >= limits.max_candidates * 4:
                    break
            if len(candidates) >= limits.max_candidates * 4:
                break
        if len(candidates) >= limits.max_candidates * 4:
            break
        if pair_count >= limits.max_episode_pairs:
            break

    # Syntactically different copy bindings may be extensionally identical for
    # a new prompt (for example two equal repeated slot tokens). Keep them;
    # ``match_programs`` admits execution only when every matching program
    # materializes the same continuation on that concrete prompt.
    return tuple(
        sorted(candidates.values(), key=_program_rank)[: limits.max_candidates]
    )


def match_programs(
    programs: Sequence[MarkovCompositionProgram],
    prompt: Sequence[int],
    confirmed_output: Sequence[int] = (),
    *,
    limit: int | None = None,
) -> tuple[int, ...] | None:
    """Return one unanimous compositional continuation, otherwise abstain."""

    if isinstance(programs, (str, bytes)) or not isinstance(programs, Sequence):
        raise TypeError("programs must be a sequence")
    matches = set()
    for program in programs:
        if not isinstance(program, MarkovCompositionProgram):
            raise TypeError("programs must contain MarkovCompositionProgram values")
        result = program.match(prompt, confirmed_output, limit=limit)
        if result is not None:
            matches.add(result)
    return next(iter(matches)) if len(matches) == 1 else None


__all__ = [
    "CompositionAtom",
    "CompositionBounds",
    "ConfirmedTokenEpisode",
    "LiteralAtom",
    "MarkovCompositionError",
    "MarkovCompositionProgram",
    "RelativeCopyAtom",
    "derive_programs",
    "match_programs",
]
