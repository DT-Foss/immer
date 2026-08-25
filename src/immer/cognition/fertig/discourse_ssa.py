"""Typed discourse binding and single-assignment DAG construction.

The signed-event frontend uses this layer when a calculation crosses sentence
boundaries.  Names are exact identities, pronouns resolve only against one
typed antecedent, and every declared quantitative symbol receives exactly one
definition.  No fuzzy name matching or nearest-noun heuristic participates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from .arithmetic_ir import Span, Unit
from .clause_compiler import SymbolKey
from .signed_expression import (
    AbsoluteExpr,
    CeilingExpr,
    ClosedShareExpr,
    Definition,
    Expression,
    GroundProductExpr,
    IterateExpr,
    LiteralExpr,
    MeanExpr,
    ProductExpr,
    PositivePartExpr,
    QuotientExpr,
    RefExpr,
    SumExpr,
    UnitConversionExpr,
)


class DiscourseNumber(str, Enum):
    SINGULAR = "singular"
    PLURAL = "plural"


class DiscourseSSAError(ValueError):
    """The discourse evidence cannot support one exact SSA program."""

    def __init__(self, reason: str, *, ambiguous: bool = True) -> None:
        self.reason = reason
        self.ambiguous = ambiguous
        super().__init__(reason)


_IDENTIFIER = re.compile(r"[^a-z0-9]+")
_SINGULAR_PRONOUNS = frozenset({"he", "her", "hers", "him", "his", "it", "its", "she"})
_PLURAL_PRONOUNS = frozenset({"their", "theirs", "them", "they"})


def _canonical(surface: str) -> str:
    if not isinstance(surface, str) or not surface.strip():
        raise DiscourseSSAError("referent surface must be non-empty text")
    value = _IDENTIFIER.sub("_", surface.casefold().replace("’", "'")).strip("_")
    if not value:
        raise DiscourseSSAError("referent surface has no stable identity")
    return value


@dataclass(frozen=True, slots=True)
class DiscourseReferent:
    identity: str
    surface: str
    role: str
    number: DiscourseNumber
    members: tuple[str, ...]
    span: Span

    def __post_init__(self) -> None:
        if not self.identity or not self.role:
            raise ValueError("referent identity and role must be non-empty")
        if not isinstance(self.span, Span):
            raise TypeError("referent span must be a Span")
        object.__setattr__(self, "members", tuple(self.members))


@dataclass(frozen=True, slots=True)
class TypedSymbol:
    key: SymbolKey
    unit: Unit
    role: str
    owner: str
    span: Span

    def __post_init__(self) -> None:
        if not isinstance(self.key, SymbolKey):
            raise TypeError("typed symbol key must be a SymbolKey")
        if not isinstance(self.unit, Unit):
            raise TypeError("typed symbol unit must be a Unit")
        if not self.role or not self.owner:
            raise ValueError("typed symbol role and owner must be non-empty")
        if not isinstance(self.span, Span):
            raise TypeError("typed symbol span must be a Span")


@dataclass(frozen=True, slots=True)
class _BoundDefinition:
    symbol: TypedSymbol
    definition: Definition
    relation_id: str


def _references(expr: Expression) -> set[SymbolKey]:
    if isinstance(expr, LiteralExpr):
        return set()
    if isinstance(expr, RefExpr):
        return {expr.symbol}
    if isinstance(expr, (ProductExpr, GroundProductExpr)):
        rows: set[SymbolKey] = set()
        for factor in expr.factors:
            rows.update(_references(factor))
        return rows
    if isinstance(expr, SumExpr):
        rows = set()
        for term in expr.terms:
            rows.update(_references(term.expr))
        return rows
    if isinstance(expr, QuotientExpr):
        return _references(expr.numerator) | _references(expr.denominator)
    if isinstance(expr, MeanExpr):
        rows = set()
        for value in expr.values:
            rows.update(_references(value))
        return rows
    if isinstance(
        expr, (AbsoluteExpr, PositivePartExpr, CeilingExpr, UnitConversionExpr)
    ):
        return _references(expr.value)
    if isinstance(expr, ClosedShareExpr):
        return _references(expr.existing) | _references(expr.share)
    if isinstance(expr, IterateExpr):
        return (
            _references(expr.initial)
            | _references(expr.factor)
            | _references(expr.count)
        )
    raise TypeError(f"unsupported discourse expression {type(expr).__name__}")


class TypedDiscourseSSA:
    """Exact referent registry plus typed, acyclic single-assignment graph."""

    def __init__(self, source: str) -> None:
        if not isinstance(source, str) or not source:
            raise DiscourseSSAError("SSA source must be non-empty text")
        self.source = source
        self._referents: dict[tuple[str, str, DiscourseNumber], DiscourseReferent] = {}
        self._symbols: dict[SymbolKey, TypedSymbol] = {}
        self._definitions: dict[SymbolKey, _BoundDefinition] = {}

    def entity(self, surface: str, role: str, span: Span) -> DiscourseReferent:
        identity = _canonical(surface)
        key = (identity, role, DiscourseNumber.SINGULAR)
        existing = self._referents.get(key)
        row = DiscourseReferent(
            identity,
            surface,
            role,
            DiscourseNumber.SINGULAR,
            (identity,),
            span,
        )
        if existing is not None and existing.surface != surface:
            raise DiscourseSSAError("name surface drifted inside one entity role")
        self._referents.setdefault(key, row)
        return self._referents[key]

    def group(
        self,
        identity: str,
        members: tuple[DiscourseReferent, ...],
        role: str,
        span: Span,
    ) -> DiscourseReferent:
        if len(members) < 2 or len({member.identity for member in members}) != len(
            members
        ):
            raise DiscourseSSAError("group antecedent needs distinct members")
        if any(
            member.number is not DiscourseNumber.SINGULAR or member.role != role
            for member in members
        ):
            raise DiscourseSSAError("group members do not share one typed role")
        canonical = _canonical(identity)
        key = (canonical, role, DiscourseNumber.PLURAL)
        row = DiscourseReferent(
            canonical,
            identity,
            role,
            DiscourseNumber.PLURAL,
            tuple(member.identity for member in members),
            span,
        )
        existing = self._referents.get(key)
        if existing is not None and existing.members != row.members:
            raise DiscourseSSAError("plural antecedent has multiple member sets")
        self._referents.setdefault(key, row)
        return self._referents[key]

    def resolve(
        self,
        surface: str,
        *,
        role: str,
        number: DiscourseNumber | None = None,
        members: tuple[str, ...] | None = None,
    ) -> DiscourseReferent:
        normalized = surface.casefold().replace("’", "'")
        if normalized in _SINGULAR_PRONOUNS | _PLURAL_PRONOUNS:
            required = (
                DiscourseNumber.PLURAL
                if normalized in _PLURAL_PRONOUNS
                else DiscourseNumber.SINGULAR
            )
            if number is not None and number is not required:
                raise DiscourseSSAError("pronoun number conflicts with its role")
            candidates = [
                referent
                for (
                    _,
                    candidate_role,
                    candidate_number,
                ), referent in self._referents.items()
                if candidate_role == role and candidate_number is required
            ]
        else:
            identity = _canonical(surface)
            candidates = [
                referent
                for (
                    candidate_id,
                    candidate_role,
                    candidate_number,
                ), referent in self._referents.items()
                if candidate_id == identity
                and candidate_role == role
                and (number is None or candidate_number is number)
            ]
        if members is not None:
            expected = tuple(_canonical(member) for member in members)
            candidates = [row for row in candidates if row.members == expected]
        if len(candidates) != 1:
            raise DiscourseSSAError(
                "reference has no unique typed antecedent", ambiguous=True
            )
        return candidates[0]

    def symbol(
        self,
        owner: DiscourseReferent,
        *,
        property: str,
        item: str,
        scope: str,
        state: str,
        role: str,
        unit: Unit,
        span: Span,
    ) -> TypedSymbol:
        key = SymbolKey(owner.identity, property, item, scope, state)
        row = TypedSymbol(key, unit, role, owner.identity, span)
        existing = self._symbols.get(key)
        if existing is not None and (
            existing.unit != unit
            or existing.role != role
            or existing.owner != owner.identity
        ):
            raise DiscourseSSAError("symbol identity changed type or role")
        self._symbols.setdefault(key, row)
        return self._symbols[key]

    def ref(
        self,
        symbol: TypedSymbol,
        span: Span,
        *,
        role: str,
        unit: Unit | None = None,
    ) -> RefExpr:
        registered = self._symbols.get(symbol.key)
        if registered != symbol or symbol.role != role:
            raise DiscourseSSAError("reference role does not match its declaration")
        if unit is not None and not unit.compatible(symbol.unit):
            raise DiscourseSSAError("reference unit does not match its declaration")
        return RefExpr(symbol.key, span)

    def define(
        self,
        symbol: TypedSymbol,
        expr: Expression,
        span: Span,
        *,
        relation_id: str,
    ) -> None:
        if self._symbols.get(symbol.key) != symbol:
            raise DiscourseSSAError("definition uses an undeclared typed symbol")
        if symbol.key in self._definitions:
            raise DiscourseSSAError("SSA symbol has multiple definitions")
        if not isinstance(relation_id, str) or not relation_id.strip():
            raise DiscourseSSAError("definition relation id is missing")
        self._definitions[symbol.key] = _BoundDefinition(
            symbol, Definition(symbol.key, expr, span), relation_id
        )

    def finalize(self, target: Expression) -> tuple[Definition, ...]:
        if set(self._symbols) != set(self._definitions):
            missing = set(self._symbols).difference(self._definitions)
            names = ", ".join(sorted(key.variable_name for key in missing))
            raise DiscourseSSAError(f"SSA relations are incomplete: {names}")

        dependencies = {
            key: _references(bound.definition.expr)
            for key, bound in self._definitions.items()
        }
        undefined = (
            set()
            .union(*dependencies.values(), _references(target))
            .difference(self._definitions)
        )
        if undefined:
            names = ", ".join(sorted(key.variable_name for key in undefined))
            raise DiscourseSSAError(f"SSA has undefined dependencies: {names}")

        ordered: list[Definition] = []
        state: dict[SymbolKey, int] = {}

        def visit(key: SymbolKey) -> None:
            marker = state.get(key, 0)
            if marker == 1:
                raise DiscourseSSAError("SSA dependency cycle is non-affine")
            if marker == 2:
                return
            state[key] = 1
            for dependency in sorted(
                dependencies[key], key=lambda row: row.variable_name
            ):
                visit(dependency)
            state[key] = 2
            ordered.append(self._definitions[key].definition)

        for dependency in sorted(
            _references(target), key=lambda row: row.variable_name
        ):
            visit(dependency)
        disconnected = set(self._definitions).difference(state)
        if disconnected:
            names = ", ".join(sorted(key.variable_name for key in disconnected))
            raise DiscourseSSAError(f"SSA relations are outside target scope: {names}")
        return tuple(ordered)


__all__ = [
    "DiscourseNumber",
    "DiscourseReferent",
    "DiscourseSSAError",
    "TypedDiscourseSSA",
    "TypedSymbol",
]
