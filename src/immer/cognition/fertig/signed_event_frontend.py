"""Span-aware event grammar for closed signed contribution problems.

This module is intentionally smaller than the expression compiler it feeds.
It recognizes reusable quantitative clause shapes, binds every accepted numeric
surface once, and emits only the typed AST from :mod:`signed_expression`.
Domain nouns become symbol metadata; they never select arithmetic operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction
import re

from .arithmetic_ir import Span, Unit
from .clause_compiler import SymbolKey
from .signed_expression import (
    ExpressionCompileError,
    ExpressionCompileResult,
    ExpressionProgram,
    ExpressionTarget,
    LiteralExpr,
    NumericEvidence,
    ProductExpr,
    SignedTerm,
    SumExpr,
    compile_expression,
)


class FrontendStatus(str, Enum):
    COMPILED = "compiled"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    norm: str
    span: Span
    number: Fraction | None = None
    money: bool = False


@dataclass(frozen=True, slots=True)
class Clause:
    tokens: tuple[Token, ...]
    span: Span
    question: bool

    @property
    def norms(self) -> tuple[str, ...]:
        return tuple(token.norm for token in self.tokens)


@dataclass(frozen=True, slots=True)
class FrontendResult:
    status: FrontendStatus
    family_matched: bool
    compiled: ExpressionCompileResult | None = None
    family: str | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status is FrontendStatus.COMPILED and self.compiled is not None


_TOKEN = re.compile(
    r"\$\s*(?:\d[\d,]*(?:\.\d+)?|\.\d+)|"
    r"(?:\d[\d,]*(?:\.\d+)?|\.\d+)|"
    r"[A-Za-z]+(?:['’\-][A-Za-z]+)*|[.!?;,:]"
)
_DIGIT = re.compile(r"^\$?\s*(?:\d[\d,]*(?:\.\d+)?|\.\d+)$")
_OPERATOR_NUMBERS = {"once": Fraction(1), "twice": Fraction(2), "half": Fraction(1, 2)}

COUNT = Unit.count()
SCALAR = Unit.scalar()
MONEY = Unit.base("money", symbol="USD")
LENGTH = Unit.base("length", symbol="m")
TIME = Unit.base("time", symbol="day")
PRICE_PER_COUNT = MONEY / COUNT
PRICE_PER_LENGTH = MONEY / LENGTH
COUNT_PER_TIME = COUNT / TIME


class _Reject(ValueError):
    def __init__(self, status: FrontendStatus, reason: str) -> None:
        self.status = status
        self.reason = reason
        super().__init__(reason)


def _fraction(text: str) -> Fraction:
    raw = text.replace("$", "").replace(",", "").strip()
    try:
        return Fraction(Decimal(raw))
    except InvalidOperation as exc:
        raise ValueError(f"invalid numeric token: {text!r}") from exc


def lex(source: str) -> tuple[Token, ...]:
    if not isinstance(source, str):
        raise TypeError("source must be text")
    result: list[Token] = []
    for match in _TOKEN.finditer(source):
        text = match.group()
        norm = text.casefold().replace("’", "'")
        number = None
        money = text.lstrip().startswith("$")
        if _DIGIT.fullmatch(text):
            number = _fraction(text)
        elif norm in _OPERATOR_NUMBERS:
            number = _OPERATOR_NUMBERS[norm]
        result.append(
            Token(text, norm, Span(match.start(), match.end(), source), number, money)
        )
    return tuple(result)


def clauses(source: str) -> tuple[Clause, ...]:
    tokens = lex(source)
    result: list[Clause] = []
    current: list[Token] = []
    for token in tokens:
        if token.norm in {".", "!", "?", ";"}:
            if current:
                result.append(
                    Clause(
                        tuple(current),
                        Span(current[0].span.start, current[-1].span.end, source),
                        token.norm == "?",
                    )
                )
                current = []
            continue
        current.append(token)
    if current:
        result.append(
            Clause(
                tuple(current),
                Span(current[0].span.start, current[-1].span.end, source),
                False,
            )
        )
    if result and not any(clause.question for clause in result):
        last = result[-1]
        result[-1] = Clause(last.tokens, last.span, True)
    return tuple(result)


def _contains(norms: tuple[str, ...], *phrase: str) -> bool:
    width = len(phrase)
    return any(
        norms[index : index + width] == phrase
        for index in range(len(norms) - width + 1)
    )


def _numeric(
    clause_set: tuple[Clause, ...], *, digits_only: bool = True
) -> list[Token]:
    rows = [
        token
        for clause in clause_set
        for token in clause.tokens
        if token.number is not None
    ]
    if digits_only:
        rows = [token for token in rows if _DIGIT.fullmatch(token.text)]
    return rows


def _clause_for(clause_set: tuple[Clause, ...], token: Token) -> Clause:
    matching = [clause for clause in clause_set if token in clause.tokens]
    if len(matching) != 1:
        raise _Reject(FrontendStatus.INVALID, "numeric token has no unique clause")
    return matching[0]


def _nearby(
    norms: tuple[str, ...], token_index: int, words: set[str], radius: int = 5
) -> bool:
    start = max(0, token_index - radius)
    end = min(len(norms), token_index + radius + 1)
    return any(word in words for word in norms[start:end])


def _singular(word: str) -> str:
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("oes") and len(word) > 3:
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us")):
        return word[:-1]
    return word


class _Builder:
    def __init__(self, source: str, clause_set: tuple[Clause, ...]) -> None:
        self.source = source
        self.clauses = clause_set
        self.evidence: list[NumericEvidence] = []
        self.used: set[tuple[int, int]] = set()

    def literal(
        self, token: Token, unit: Unit, *, value: Fraction | None = None
    ) -> LiteralExpr:
        coordinate = (token.span.start, token.span.end)
        if coordinate in self.used:
            raise _Reject(FrontendStatus.INVALID, "numeric surface was consumed twice")
        exact = token.number if value is None else value
        if exact is None:
            raise _Reject(FrontendStatus.INVALID, "non-numeric token used as literal")
        evidence_id = f"surface-{token.span.start:06d}-{token.span.end:06d}"
        evidence = NumericEvidence(evidence_id, exact, unit, token.span)
        self.evidence.append(evidence)
        self.used.add(coordinate)
        return LiteralExpr(exact, unit, token.span, (evidence_id,))

    def lexical_literal(self, token: Token, value: Fraction, unit: Unit) -> LiteralExpr:
        return self.literal(token, unit, value=value)

    def finish(self, expression, family: str) -> FrontendResult:
        explicit = {
            (token.span.start, token.span.end)
            for clause in self.clauses
            for token in clause.tokens
            if token.number is not None
        }
        if explicit != self.used.intersection(explicit):
            missing = sorted(explicit.difference(self.used))
            raise _Reject(
                FrontendStatus.UNSUPPORTED, f"unconsumed numeric surfaces: {missing}"
            )
        target = SymbolKey("question", "answer", "result", family, "current")
        program = ExpressionProgram(
            (),
            ExpressionTarget(
                target, expression, Span(0, len(self.source), self.source)
            ),
            tuple(self.evidence),
        )
        try:
            compiled = compile_expression(program)
        except ExpressionCompileError as exc:
            raise _Reject(FrontendStatus.INVALID, str(exc)) from exc
        return FrontendResult(FrontendStatus.COMPILED, True, compiled, family)


def _signed(sign: int, expr, role: str) -> SignedTerm:
    return SignedTerm(sign, expr, role, expr.span)


def _sum(*terms: SignedTerm) -> SumExpr:
    return SumExpr(
        tuple(terms),
        Span(
            min(term.span.start for term in terms),
            max(term.span.end for term in terms),
        ),
    )


def _product(*factors) -> ProductExpr:
    return ProductExpr(
        tuple(factors),
        Span(
            min(factor.span.start for factor in factors),
            max(factor.span.end for factor in factors),
        ),
    )


def _question(clause_set: tuple[Clause, ...]) -> Clause:
    rows = [clause for clause in clause_set if clause.question]
    if len(rows) != 1:
        raise _Reject(FrontendStatus.AMBIGUOUS, "one explicit question is required")
    return rows[0]


def _rate_length_difference(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "much", "more") and "cost" in question.norms
    ):
        return None
    numbers = _numeric(clause_set)
    rate = [
        token
        for token in numbers
        if token.money and "per" in _clause_for(clause_set, token).norms
    ]
    lengths = [
        token
        for token in numbers
        if not token.money
        and _nearby(
            _clause_for(clause_set, token).norms,
            _clause_for(clause_set, token).tokens.index(token),
            {"meter", "meters"},
            2,
        )
    ]
    if len(rate) != 1 or len(lengths) != 2:
        raise _Reject(
            FrontendStatus.UNSUPPORTED, "rate/length comparison is incomplete"
        )
    rate_clause = _clause_for(clause_set, rate[0])
    per_index = rate_clause.norms.index("per")
    if per_index + 1 >= len(rate_clause.tokens):
        raise _Reject(FrontendStatus.UNSUPPORTED, "rate unit is missing")
    rate_unit = _singular(rate_clause.norms[per_index + 1])
    length_units = set()
    for token in lengths:
        clause = _clause_for(clause_set, token)
        index = clause.tokens.index(token)
        if index + 1 >= len(clause.tokens):
            raise _Reject(FrontendStatus.UNSUPPORTED, "length unit is missing")
        length_units.add(_singular(clause.norms[index + 1]))
    if length_units != {rate_unit}:
        raise _Reject(FrontendStatus.INVALID, "rate and length units differ")

    def owner(token: Token) -> str:
        clause = _clause_for(clause_set, token)
        index = clause.tokens.index(token)
        candidates = [
            row.norm.rstrip("'s")
            for row in clause.tokens[max(0, index - 7) : index]
            if row.text[:1].isupper() and row.norm not in {"how"}
        ]
        if not candidates:
            raise _Reject(FrontendStatus.AMBIGUOUS, "length owner is not explicit")
        return candidates[-1]

    by_owner = {owner(token): token for token in lengths}
    target_owners = [
        token.norm.rstrip("'s")
        for token in question.tokens
        if token.text[:1].isupper() and token.norm not in {"how"}
    ]
    selected = [name for name in target_owners if name in by_owner]
    if len(selected) != 1 or len(by_owner) != 2:
        raise _Reject(FrontendStatus.AMBIGUOUS, "comparison orientation is ambiguous")
    target_length = by_owner[selected[0]]
    baseline_length = next(
        token for name, token in by_owner.items() if name != selected[0]
    )
    builder = _Builder(source, clause_set)
    delta = _sum(
        _signed(1, builder.literal(target_length, LENGTH), "target_length"),
        _signed(-1, builder.literal(baseline_length, LENGTH), "baseline_length"),
    )
    expr = _product(builder.literal(rate[0], PRICE_PER_LENGTH), delta)
    return builder.finish(expr, "rate_length_difference")


def _functioning_chain(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if "functioning" not in question.norms:
        return None
    numbers = _numeric(clause_set)
    losses = []
    for token in numbers:
        clause = _clause_for(clause_set, token)
        index = clause.tokens.index(token)
        following = clause.norms[index + 1 : index + 3]
        if (
            following
            and following[0] in {"streetlight", "streetlights"}
            and _contains(clause.norms, "not", "working")
        ):
            losses.append(token)

    light_counts = []
    pole_rates = []
    intersection_counts = []
    for token in numbers:
        clause = _clause_for(clause_set, token)
        index = clause.tokens.index(token)
        has_poles = any(_singular(word) == "pole" for word in clause.norms)
        has_intersections = any(
            _singular(word) == "intersection" for word in clause.norms
        )
        following = clause.norms[index + 1 : index + 3]
        if (
            index > 0
            and clause.norms[index - 1] in {"has", "have"}
            and "each" in clause.norms[:index]
            and any(_singular(word) == "pole" for word in clause.norms[:index])
            and any(_singular(word) in {"light", "streetlight"} for word in following)
        ):
            light_counts.append(token)
        if (
            index > 0
            and clause.norms[index - 1] == "is"
            and "each" in clause.norms
            and has_poles
            and has_intersections
        ):
            pole_rates.append(token)
        if (
            index + 1 < len(clause.tokens)
            and _singular(clause.norms[index + 1]) == "intersection"
        ):
            intersection_counts.append(token)

    if not (
        len(losses)
        == len(light_counts)
        == len(pole_rates)
        == len(intersection_counts)
        == 1
    ):
        raise _Reject(FrontendStatus.UNSUPPORTED, "functioning chain is incomplete")
    builder = _Builder(source, clause_set)
    positive = _product(
        builder.literal(light_counts[0], SCALAR),
        builder.literal(pole_rates[0], SCALAR),
        builder.literal(intersection_counts[0], COUNT),
    )
    expr = _sum(
        _signed(1, positive, "total"),
        _signed(-1, builder.literal(losses[0], COUNT), "not_working"),
    )
    return builder.finish(expr, "functioning_count")


def _daily_combined(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not ("total" in question.norms and "together" in question.norms):
        return None
    numbers = _numeric(clause_set)
    rates = [
        token
        for token in numbers
        if {"daily", "day"}.intersection(_clause_for(clause_set, token).norms)
        and token not in question.tokens
    ]
    durations = [
        token
        for token in numbers
        if token in question.tokens
        and _nearby(question.norms, question.tokens.index(token), {"day", "days"}, 2)
    ]
    if len(rates) != 2 or len(durations) != 1:
        raise _Reject(FrontendStatus.UNSUPPORTED, "combined daily ledger is incomplete")
    builder = _Builder(source, clause_set)
    rate_sum = _sum(
        *[
            _signed(1, builder.literal(token, COUNT_PER_TIME), "daily_rate")
            for token in rates
        ]
    )
    expr = _product(rate_sum, builder.literal(durations[0], TIME))
    return builder.finish(expr, "combined_daily_total")


def _old_new_savings(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not ("save" in question.norms and "week" in question.norms):
        return None
    numbers = _numeric(clause_set)
    daily = [
        token
        for token in numbers
        if _contains(_clause_for(clause_set, token).norms, "a", "day")
    ]
    prices = [
        token
        for token in numbers
        if token.money and "each" in _clause_for(clause_set, token).norms
    ]
    if len(daily) != 1 or len(prices) != 2:
        raise _Reject(
            FrontendStatus.UNSUPPORTED, "old/new savings facts are incomplete"
        )

    def local_phrase(token: Token, *words: str) -> bool:
        clause = _clause_for(clause_set, token)
        index = clause.tokens.index(token)
        window = clause.norms[max(0, index - 8) : min(len(clause.tokens), index + 3)]
        return _contains(window, *words)

    old = [token for token in prices if local_phrase(token, "used", "to")]
    new = [token for token in prices if local_phrase(token, "new", "vendor")]
    week = [token for token in question.tokens if token.norm == "week"]
    if len(old) != 1 or len(new) != 1 or len(week) != 1:
        raise _Reject(
            FrontendStatus.AMBIGUOUS, "old/new savings orientation is ambiguous"
        )
    daily_clause = _clause_for(clause_set, daily[0])
    daily_index = daily_clause.tokens.index(daily[0])
    if daily_index + 1 >= len(daily_clause.tokens):
        raise _Reject(FrontendStatus.UNSUPPORTED, "daily item is missing")
    daily_item = _singular(daily_clause.norms[daily_index + 1])
    if any(
        daily_item
        not in {_singular(word) for word in _clause_for(clause_set, token).norms}
        and not {"them", "it"}.intersection(_clause_for(clause_set, token).norms)
        for token in (old[0], new[0])
    ):
        raise _Reject(FrontendStatus.INVALID, "old/new rates bind different items")
    builder = _Builder(source, clause_set)
    price_delta = _sum(
        _signed(1, builder.literal(old[0], PRICE_PER_COUNT), "old_price"),
        _signed(-1, builder.literal(new[0], PRICE_PER_COUNT), "new_price"),
    )
    duration = builder.lexical_literal(week[0], Fraction(7), TIME)
    expr = _product(builder.literal(daily[0], COUNT_PER_TIME), duration, price_delta)
    return builder.finish(expr, "old_new_rate_savings")


def _category_sales(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not ("more" in question.norms and "compared" in question.norms):
        return None
    available = next(
        (clause for clause in clause_set if "available" in clause.norms), None
    )
    sales = next(
        (
            clause
            for clause in clause_set
            if ("sell" in clause.norms or "sells" in clause.norms)
            and any(token.number is not None for token in clause.tokens)
        ),
        None,
    )
    if available is None or sales is None or "each" not in sales.norms:
        return None
    available_numbers = [
        token
        for token in available.tokens
        if token.number is not None and _DIGIT.fullmatch(token.text)
    ]
    sales_numbers = [
        token
        for token in sales.tokens
        if token.number is not None and _DIGIT.fullmatch(token.text)
    ]
    if len(available_numbers) != 2 or len(sales_numbers) != 2:
        raise _Reject(FrontendStatus.UNSUPPORTED, "category sales facts are incomplete")
    if any(
        not _contains(
            sales.norms[sales.tokens.index(token) + 1 : sales.tokens.index(token) + 4],
            "of",
            "each",
        )
        for token in sales_numbers
    ):
        raise _Reject(
            FrontendStatus.UNSUPPORTED,
            "each category sale requires an explicit per-item multiplier",
        )
    available_scopes: list[str] = []
    for token in available_numbers:
        index = available.tokens.index(token)
        if index + 1 >= len(available.tokens):
            raise _Reject(FrontendStatus.UNSUPPORTED, "category scope is missing")
        available_scopes.append(_singular(available.norms[index + 1]))
    if len(set(available_scopes)) != 2:
        raise _Reject(FrontendStatus.AMBIGUOUS, "category scopes are not unique")
    sales_by_scope: dict[str, Token] = {}
    for token in sales_numbers:
        index = sales.tokens.index(token)
        matches = [
            scope
            for scope in available_scopes
            if scope in {_singular(word) for word in sales.norms[index + 1 : index + 8]}
        ]
        if len(matches) != 1 or matches[0] in sales_by_scope:
            raise _Reject(
                FrontendStatus.AMBIGUOUS,
                "sales multiplier has no unique category scope",
            )
        sales_by_scope[matches[0]] = token
    target_scopes = [
        scope
        for word in question.norms
        for scope in available_scopes
        if _singular(word) == scope
    ]
    if target_scopes != available_scopes:
        raise _Reject(
            FrontendStatus.AMBIGUOUS, "category comparison orientation differs"
        )
    builder = _Builder(source, clause_set)
    counts_by_scope = dict(zip(available_scopes, available_numbers, strict=True))
    first_scope, second_scope = target_scopes
    first = _product(
        builder.literal(counts_by_scope[first_scope], COUNT),
        builder.literal(sales_by_scope[first_scope], SCALAR),
    )
    second = _product(
        builder.literal(counts_by_scope[second_scope], COUNT),
        builder.literal(sales_by_scope[second_scope], SCALAR),
    )
    expr = _sum(
        _signed(1, first, "target_sales"), _signed(-1, second, "comparison_sales")
    )
    return builder.finish(expr, "category_sales_difference")


def _repeated_duration(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not _contains(question.norms, "how", "long"):
        return None
    multiplicities = [
        token
        for clause in clause_set
        for token in clause.tokens
        if token.norm in {"once", "twice"}
    ]
    durations = [
        token
        for clause in clause_set
        for token in clause.tokens
        if token.number is not None
        and _DIGIT.fullmatch(token.text)
        and _nearby(clause.norms, clause.tokens.index(token), {"second", "seconds"}, 2)
    ]
    if len(multiplicities) != 2 or len(durations) != 2:
        return None

    def multiplicity_scope(token: Token) -> tuple[str, ...]:
        clause = _clause_for(clause_set, token)
        index = clause.tokens.index(token)
        boundaries = [
            offset
            for offset, norm in enumerate(clause.norms[:index])
            if norm in {"sing", "and"}
        ]
        if not boundaries:
            raise _Reject(
                FrontendStatus.AMBIGUOUS,
                "repetition has no explicit item scope",
            )
        scope = tuple(clause.norms[boundaries[-1] + 1 : index])
        if not scope:
            raise _Reject(FrontendStatus.AMBIGUOUS, "repetition scope is empty")
        return scope

    def duration_scope(token: Token) -> tuple[str, ...]:
        clause = _clause_for(clause_set, token)
        index = clause.tokens.index(token)
        if index == 0 or clause.norms[index - 1] != "is":
            raise _Reject(
                FrontendStatus.AMBIGUOUS,
                "duration has no explicit item binding",
            )
        boundaries = [
            offset
            for offset, norm in enumerate(clause.norms[: index - 1])
            if norm in {"if", "and"}
        ]
        start = boundaries[-1] + 1 if boundaries else 0
        scope = tuple(clause.norms[start : index - 1])
        if not scope:
            raise _Reject(FrontendStatus.AMBIGUOUS, "duration scope is empty")
        return scope

    multiplicity_by_scope: dict[tuple[str, ...], Token] = {}
    for token in multiplicities:
        scope = multiplicity_scope(token)
        if scope in multiplicity_by_scope:
            raise _Reject(FrontendStatus.AMBIGUOUS, "duplicate repetition scope")
        multiplicity_by_scope[scope] = token
    duration_by_scope: dict[tuple[str, ...], Token] = {}
    for token in durations:
        scope = duration_scope(token)
        if scope in duration_by_scope:
            raise _Reject(FrontendStatus.AMBIGUOUS, "duplicate duration scope")
        duration_by_scope[scope] = token
    if multiplicity_by_scope.keys() != duration_by_scope.keys():
        raise _Reject(
            FrontendStatus.AMBIGUOUS,
            "repetition and duration scopes do not match",
        )

    builder = _Builder(source, clause_set)
    terms = [
        _product(
            builder.literal(multiplicity, SCALAR),
            builder.literal(duration_by_scope[scope], Unit.base("time", symbol="s")),
        )
        for scope, multiplicity in multiplicity_by_scope.items()
    ]
    expr = _sum(*[_signed(1, term, "duration") for term in terms])
    return builder.finish(expr, "repeated_duration_total")


_PLANNERS = (
    _rate_length_difference,
    _functioning_chain,
    _daily_combined,
    _old_new_savings,
    _category_sales,
    _repeated_duration,
)


def compile_signed_events(source: str) -> FrontendResult:
    """Compile one fully covered event ledger, preserving abstention otherwise."""

    if not isinstance(source, str) or not source.strip():
        return FrontendResult(
            FrontendStatus.INVALID, False, reason="source must be non-empty text"
        )
    try:
        clause_set = clauses(source)
        successes: list[FrontendResult] = []
        matched = False
        failures: list[_Reject] = []
        for planner in _PLANNERS:
            try:
                result = planner(source, clause_set)
            except _Reject as exc:
                matched = True
                failures.append(exc)
                continue
            if result is not None:
                matched = True
                successes.append(result)
        if len(successes) > 1:
            return FrontendResult(
                FrontendStatus.AMBIGUOUS, True, reason="multiple signed-event parses"
            )
        if len(successes) == 1:
            return successes[0]
        if failures:
            status = (
                FrontendStatus.AMBIGUOUS
                if any(row.status is FrontendStatus.AMBIGUOUS for row in failures)
                else failures[0].status
            )
            return FrontendResult(
                status, True, reason="; ".join(row.reason for row in failures)
            )
        return FrontendResult(
            FrontendStatus.UNSUPPORTED, matched, reason="no signed-event family matched"
        )
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return FrontendResult(FrontendStatus.INVALID, True, reason=str(exc))


__all__ = [
    "Clause",
    "FrontendResult",
    "FrontendStatus",
    "Token",
    "clauses",
    "compile_signed_events",
    "lex",
]
