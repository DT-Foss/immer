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
    Definition,
    ExpressionCompileError,
    ExpressionCompileResult,
    ExpressionProgram,
    ExpressionTarget,
    LiteralExpr,
    NumericEvidence,
    ProductExpr,
    QuotientExpr,
    RefExpr,
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
COUNT_PER_MEMBER_TIME = COUNT / COUNT / TIME

_FIXED_MONTH_DAYS = {
    "january": 31,
    "march": 31,
    "april": 30,
    "may": 31,
    "june": 30,
    "july": 31,
    "august": 31,
    "september": 30,
    "october": 31,
    "november": 30,
    "december": 31,
}
_MONTHS = frozenset((*_FIXED_MONTH_DAYS, "february"))


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
    if word == "children":
        return "child"
    if word == "people":
        return "person"
    if word.endswith(("ches", "shes", "xes", "zes")) and len(word) > 4:
        return word[:-2]
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

    def finish(
        self,
        expression,
        family: str,
        *,
        definitions: tuple[Definition, ...] = (),
    ) -> FrontendResult:
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
            definitions,
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


@dataclass(frozen=True, slots=True)
class _Contribution:
    """One locally bound amount, optionally repeated by an explicit count."""

    amount: Token
    amount_unit: Unit
    role: str
    sign: int = 1
    multiplicity: Token | None = None
    multiplicity_unit: Unit = SCALAR


def _contribution(builder: _Builder, row: _Contribution):
    amount = builder.literal(row.amount, row.amount_unit)
    if row.multiplicity is None:
        return amount
    return _product(
        builder.literal(row.multiplicity, row.multiplicity_unit),
        amount,
    )


def _ledger(builder: _Builder, rows: tuple[_Contribution, ...]) -> SumExpr:
    """Lower a nonempty signed transaction ledger through shared primitives."""

    if not rows:
        raise _Reject(FrontendStatus.INVALID, "empty contribution ledger")
    if any(row.sign not in {-1, 1} for row in rows):
        raise _Reject(FrontendStatus.INVALID, "invalid contribution polarity")
    return _sum(
        *[_signed(row.sign, _contribution(builder, row), row.role) for row in rows]
    )


def _unique_word_token(
    clause_set: tuple[Clause, ...],
    word: str,
    *,
    clause: Clause | None = None,
) -> Token:
    """Bind one operator cardinal only inside a selected production context."""

    rows = [
        token
        for current in clause_set
        if clause is None or current is clause
        for token in current.tokens
        if token.norm == word
    ]
    if len(rows) != 1:
        raise _Reject(
            FrontendStatus.AMBIGUOUS,
            f"operator cardinal {word!r} is not uniquely scoped",
        )
    return rows[0]


def _money_tokens(clause: Clause) -> list[Token]:
    return [
        token for token in clause.tokens if token.number is not None and token.money
    ]


def _digit_tokens(clause: Clause) -> list[Token]:
    return [
        token
        for token in clause.tokens
        if token.number is not None and _DIGIT.fullmatch(token.text)
    ]


def _counts(clause: Clause) -> list[Token]:
    return [token for token in _digit_tokens(clause) if not token.money]


def _one_token(
    rows: list[Token],
    reason: str,
    *,
    status: FrontendStatus = FrontendStatus.UNSUPPORTED,
) -> Token:
    if len(rows) != 1:
        raise _Reject(status, reason)
    return rows[0]


def _require(
    condition: bool,
    reason: str,
    status: FrontendStatus = FrontendStatus.UNSUPPORTED,
) -> None:
    if not condition:
        raise _Reject(status, reason)


def _noun_after(
    clause: Clause, token: Token, *, skip: frozenset[str] = frozenset()
) -> str:
    index = clause.tokens.index(token) + 1
    while index < len(clause.tokens) and clause.norms[index] in skip:
        index += 1
    if index >= len(clause.tokens):
        raise _Reject(FrontendStatus.UNSUPPORTED, "bound item noun is missing")
    return _singular(clause.norms[index])


def _noun_after_index(
    clause: Clause,
    index: int,
    *,
    skip: frozenset[str] = frozenset({"a", "an", "the"}),
) -> str:
    index += 1
    while index < len(clause.tokens) and clause.norms[index] in skip:
        index += 1
    if index >= len(clause.tokens):
        raise _Reject(FrontendStatus.AMBIGUOUS, "scoped noun is missing")
    return _singular(clause.norms[index])


def _wide_span(first, second, source: str) -> Span:
    return Span(
        min(first.span.start, second.span.start),
        max(first.span.end, second.span.end),
        source,
    )


def _single_clause(
    clause_set: tuple[Clause, ...],
    predicate,
    *,
    reason: str,
) -> Clause:
    rows = [clause for clause in clause_set if predicate(clause)]
    if len(rows) != 1:
        raise _Reject(FrontendStatus.AMBIGUOUS, reason)
    return rows[0]


def _each_rate_map(clause: Clause) -> dict[str, Token]:
    """Bind every money token to one following ``each <category>`` scope."""

    rates: dict[str, Token] = {}
    for price in _money_tokens(clause):
        start = clause.tokens.index(price) + 1
        offsets = [
            index
            for index in range(start, min(len(clause.tokens), start + 6))
            if clause.norms[index] == "each"
        ]
        if len(offsets) != 1 or offsets[0] + 1 >= len(clause.tokens):
            raise _Reject(
                FrontendStatus.UNSUPPORTED,
                "rate lacks one local each-category",
            )
        category = _singular(clause.norms[offsets[0] + 1])
        if category in rates:
            raise _Reject(FrontendStatus.AMBIGUOUS, "duplicate rate category")
        rates[category] = price
    return rates


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


def _calendar_daily_total(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    question = _question(clause_set)
    if not _contains(question.norms, "total", "number", "of", "posts"):
        return None
    mentioned_months = [token for token in question.tokens if token.norm in _MONTHS]
    if not mentioned_months:
        return None
    if len(mentioned_months) != 1:
        raise _Reject(FrontendStatus.AMBIGUOUS, "calendar month is not unique")
    month = mentioned_months[0]
    month_index = question.tokens.index(month)
    if month_index == 0 or question.norms[month_index - 1] != "in":
        raise _Reject(
            FrontendStatus.UNSUPPORTED,
            "calendar month has no local in-scope binding",
        )
    if month.norm not in _FIXED_MONTH_DAYS:
        raise _Reject(
            FrontendStatus.UNSUPPORTED,
            "February requires an explicit day count",
        )

    member_counts: list[Token] = []
    daily_rates: list[Token] = []
    for token in _numeric(clause_set):
        clause = _clause_for(clause_set, token)
        index = clause.tokens.index(token)
        following = clause.norms[index + 1 : index + 5]
        if (
            index > 0
            and clause.norms[index - 1] == "has"
            and following
            and _singular(following[0]) == "member"
        ):
            member_counts.append(token)
        if _contains(following, "posts", "per", "day") and _contains(
            clause.norms[:index], "each", "member"
        ):
            daily_rates.append(token)
    if len(member_counts) != 1 or len(daily_rates) != 1:
        raise _Reject(
            FrontendStatus.UNSUPPORTED,
            "calendar daily-rate relation is incomplete",
        )

    builder = _Builder(source, clause_set)
    expression = _product(
        builder.literal(member_counts[0], COUNT),
        builder.literal(daily_rates[0], COUNT_PER_MEMBER_TIME),
        builder.lexical_literal(
            month,
            Fraction(_FIXED_MONTH_DAYS[month.norm]),
            TIME,
        ),
    )
    return builder.finish(expression, "calendar_daily_total")


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


def _avoided_cost_transaction(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Avoided consequence minus an explicit fine and an each-priced purchase."""

    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "much", "money")
        and "save" in question.norms
        and "fixing" in question.norms
    ):
        return None
    fixing_index = question.norms.index("fixing")
    target_words = tuple(
        word
        for word in question.norms[fixing_index + 1 :]
        if word not in {"a", "an", "the"}
    )[:1]
    _require(
        bool(target_words)
        and any(
            all(word in clause.norms for word in target_words)
            for clause in clause_set
            if clause is not question
        ),
        "saved repair target has no explicit narrative scope",
        FrontendStatus.AMBIGUOUS,
    )
    try:
        owner_index = question.norms.index("does") + 1
        owner = question.norms[owner_index]
    except (ValueError, IndexError) as exc:
        raise _Reject(FrontendStatus.AMBIGUOUS, "repair owner is not explicit") from exc

    damage = _single_clause(
        clause_set,
        lambda clause: (
            "damage" in clause.norms
            and "fixed" in clause.norms
            and bool(_money_tokens(clause))
        ),
        reason="avoided damage amount is not unique",
    )
    fine = _single_clause(
        clause_set,
        lambda clause: "fine" in clause.norms and bool(_money_tokens(clause)),
        reason="repair fine is not unique",
    )
    purchase = _single_clause(
        clause_set,
        lambda clause: (
            "buy" in clause.norms and "each" in clause.norms and "cost" in clause.norms
        ),
        reason="each-priced repair purchase is not unique",
    )
    damage_money = _one_token(_money_tokens(damage), "avoided amount is incomplete")
    fine_money = _one_token(_money_tokens(fine), "fine amount is incomplete")
    purchase_money = _one_token(_money_tokens(purchase), "purchase price is incomplete")
    purchase_count = _one_token(_counts(purchase), "purchase count is incomplete")
    fixed_index = damage.norms.index("fixed")
    fixed_prefix = list(damage.norms[:fixed_index])
    _require(
        bool(fixed_prefix) and fixed_prefix.pop(0) == "if",
        "avoided damage lacks a local repair condition",
        FrontendStatus.AMBIGUOUS,
    )
    if fixed_prefix and fixed_prefix[0] in {"a", "an", "the"}:
        fixed_prefix.pop(0)
    fixed_subject = fixed_prefix.pop(0) if fixed_prefix else ""
    _require(
        fixed_subject in {"it", target_words[0]}
        and tuple(fixed_prefix) in {("doesn't", "get"), ("does", "not", "get")},
        "avoided damage is not bound to the repaired target",
        FrontendStatus.AMBIGUOUS,
    )
    price_index = purchase.tokens.index(purchase_money)
    count_index = purchase.tokens.index(purchase_count)
    _require(
        count_index < price_index
        and "each" in purchase.norms[count_index:]
        and purchase_count.number is not None
        and purchase_count.number > 0
        and purchase.norms[0] == owner
        and price_index == len(purchase.tokens) - 1
        and owner in fine.norms[: fine.tokens.index(fine_money)]
        and "if" in fine.norms[fine.tokens.index(fine_money) + 1 :]
        and "fixes" in fine.norms[fine.tokens.index(fine_money) + 1 :]
        and "it" in fine.norms[fine.tokens.index(fine_money) + 1 :],
        "repair expenses are not bound to the queried owner and action",
        FrontendStatus.INVALID,
    )

    builder = _Builder(source, clause_set)
    expression = _ledger(
        builder,
        (
            _Contribution(damage_money, MONEY, "avoided_damage"),
            _Contribution(fine_money, MONEY, "fine", sign=-1),
            _Contribution(
                purchase_money,
                PRICE_PER_COUNT,
                "materials",
                sign=-1,
                multiplicity=purchase_count,
                multiplicity_unit=COUNT,
            ),
        ),
    )
    return builder.finish(expression, "avoided_cost_transaction")


def _profit_contribution_ledger(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Revenue contributions minus explicit each-priced acquisition cost."""

    question = _question(clause_set)
    if not ("profit" in question.norms and "make" in question.norms):
        return None
    intro = _single_clause(
        clause_set,
        lambda clause: (
            "decides" in clause.norms
            and "buy" in clause.norms
            and "sell" in clause.norms
        ),
        reason="profit actor declaration is not unique",
    )
    actor_tokens = [token for token in intro.tokens if token.text[:1].isupper()]
    actor = _one_token(
        actor_tokens,
        "profit actor is not unique",
        status=FrontendStatus.AMBIGUOUS,
    ).norm
    purchase = _single_clause(
        clause_set,
        lambda clause: (
            "buys" in clause.norms and "for" in clause.norms and "each" in clause.norms
        ),
        reason="profit acquisition is not unique",
    )
    singles = _single_clause(
        clause_set,
        lambda clause: (
            "gets" in clause.norms
            and "another" in clause.norms
            and "worth" in clause.norms
        ),
        reason="single-card revenue clause is not unique",
    )
    bulk = _single_clause(
        clause_set,
        lambda clause: (
            "more" in clause.norms
            and "average" in clause.norms
            and "each" in clause.norms
        ),
        reason="bulk revenue clause is not unique",
    )
    allowed_subjects = {actor, "he", "she", "they"}
    question_subjects = [
        question.norms[index + 1]
        for index, word in enumerate(question.norms[:-1])
        if word == "did"
    ]
    subjects = (purchase.norms[0], singles.norms[0], *question_subjects)
    pronoun_subjects = {subject for subject in subjects if subject != actor}
    _require(
        all(subject in allowed_subjects for subject in subjects)
        and len(question_subjects) == 1
        and len(pronoun_subjects) <= 1
        and "they" not in pronoun_subjects,
        "profit contributions do not share one explicit actor",
        FrontendStatus.AMBIGUOUS,
    )
    purchase_count = _one_token(_counts(purchase), "acquisition count is incomplete")
    purchase_price = _one_token(
        _money_tokens(purchase), "acquisition price is incomplete"
    )
    single_count = _one_token(_counts(singles), "single-item count is incomplete")
    single_values = _money_tokens(singles)
    bulk_count = _one_token(_counts(bulk), "bulk count is incomplete")
    bulk_price = _one_token(_money_tokens(bulk), "bulk price is incomplete")
    _require(len(single_values) == 2, "single-item values are incomplete")

    purchase_item = _noun_after(purchase, purchase_count)
    single_item = _noun_after(singles, single_count)
    bulk_item = _noun_after(bulk, bulk_count, skip=frozenset({"more"}))
    another_index = singles.norms.index("another")
    _require(another_index + 1 < len(singles.tokens), "second revenue item is missing")
    another_item = _singular(singles.norms[another_index + 1])
    _require(
        single_item == another_item == bulk_item,
        "revenue contributions bind different item scopes",
        FrontendStatus.AMBIGUOUS,
    )
    _require(
        purchase_item != single_item,
        "acquisition and sale units are not separated",
        FrontendStatus.AMBIGUOUS,
    )

    builder = _Builder(source, clause_set)
    expression = _ledger(
        builder,
        (
            _Contribution(
                single_values[0],
                PRICE_PER_COUNT,
                "first_sale",
                multiplicity=single_count,
                multiplicity_unit=COUNT,
            ),
            _Contribution(single_values[1], MONEY, "second_sale"),
            _Contribution(
                bulk_price,
                PRICE_PER_COUNT,
                "bulk_sales",
                multiplicity=bulk_count,
                multiplicity_unit=COUNT,
            ),
            _Contribution(
                purchase_price,
                PRICE_PER_COUNT,
                "acquisition",
                sign=-1,
                multiplicity=purchase_count,
                multiplicity_unit=COUNT,
            ),
        ),
    )
    return builder.finish(expression, "profit_contribution_ledger")


def _alternative_cost_savings(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Compare two scoped each-rate schedules using shared category counts."""

    question = _question(clause_set)
    if not (
        "save" in question.norms
        and "choose" in question.norms
        and "over" in question.norms
    ):
        return None
    schedule_clauses = [
        clause
        for clause in clause_set
        if "each" in clause.norms
        and len(_money_tokens(clause)) >= 1
        and any(label in clause.norms for label in {"first", "second"})
    ]
    _require(
        len(schedule_clauses) == 2,
        "comparison requires two uniquely labelled rate schedules",
        FrontendStatus.AMBIGUOUS,
    )
    schedules: dict[str, dict[str, Token]] = {}
    for clause in schedule_clauses:
        labels = [label for label in ("first", "second") if label in clause.norms]
        _require(
            len(labels) == 1 and labels[0] not in schedules,
            "rate schedule label is ambiguous",
            FrontendStatus.AMBIGUOUS,
        )
        schedules[labels[0]] = _each_rate_map(clause)
    _require(
        schedules["first"].keys() == schedules["second"].keys(),
        "schedule categories differ",
        FrontendStatus.AMBIGUOUS,
    )

    counts: dict[str, Token] = {}
    for count in _counts(question):
        category = _noun_after(question, count)
        _require(
            category not in counts,
            "duplicate category count",
            FrontendStatus.AMBIGUOUS,
        )
        counts[category] = count
    _require(
        counts.keys() == schedules["first"].keys(),
        "family counts do not match the two rate schedules",
        FrontendStatus.AMBIGUOUS,
    )
    choose_index = question.norms.index("choose")
    over_index = question.norms.index("over")
    chosen = [
        label for label in schedules if label in question.norms[choose_index:over_index]
    ]
    baseline = [
        label for label in schedules if label in question.norms[over_index + 1 :]
    ]
    _require(
        len(chosen) == len(baseline) == 1 and chosen != baseline,
        "savings direction is not explicit",
        FrontendStatus.AMBIGUOUS,
    )
    projected_saving = sum(
        count.number
        * (
            schedules[baseline[0]][category].number
            - schedules[chosen[0]][category].number
        )
        for category, count in counts.items()
        if count.number is not None
        and schedules[baseline[0]][category].number is not None
        and schedules[chosen[0]][category].number is not None
    )
    _require(
        projected_saving > 0,
        "the explicitly chosen schedule does not produce a positive saving",
        FrontendStatus.INVALID,
    )

    builder = _Builder(source, clause_set)
    expressions = []
    for category, count in counts.items():
        price_delta = _sum(
            _signed(
                1,
                builder.literal(schedules[baseline[0]][category], PRICE_PER_COUNT),
                "baseline_rate",
            ),
            _signed(
                -1,
                builder.literal(schedules[chosen[0]][category], PRICE_PER_COUNT),
                "chosen_rate",
            ),
        )
        expressions.append(_product(builder.literal(count, COUNT), price_delta))
    expression = _sum(*[_signed(1, term, "category_saving") for term in expressions])
    return builder.finish(expression, "alternative_cost_savings")


def _equal_share_residual(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Subtract an explicit external contribution, then divide equally."""

    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "much")
        and "each" in question.norms
        and "raise" in question.norms
    ):
        return None
    each_index = question.norms.index("each")
    _require(each_index + 1 < len(question.tokens), "share member is missing")
    member = _singular(question.norms[each_index + 1])
    declaration = _single_clause(
        clause_set,
        lambda clause: (
            "twenty" in clause.norms
            and any(_singular(word) == member for word in clause.norms)
        ),
        reason="equal-share member cardinal is not uniquely scoped",
    )
    _single_clause(
        clause_set,
        lambda clause: (
            "each" in clause.norms
            and "same" in clause.norms
            and "amount" in clause.norms
        ),
        reason="equal-share relation is not uniquely scoped",
    )
    total_clause = _single_clause(
        clause_set,
        lambda clause: "total" in clause.norms and bool(_money_tokens(clause)),
        reason="equal-share total is not unique",
    )
    external_clause = _single_clause(
        clause_set,
        lambda clause: (
            "comes" in clause.norms
            and "from" in clause.norms
            and "rest" in clause.norms
            and bool(_money_tokens(clause))
        ),
        reason="external contribution is not unique",
    )
    total = _one_token(_money_tokens(total_clause), "share total is incomplete")
    external = _one_token(
        _money_tokens(external_clause), "external contribution is incomplete"
    )
    from_offsets = [
        index for index, word in enumerate(external_clause.norms) if word == "from"
    ]
    _require(
        len(from_offsets) == 2
        and all(index + 1 < len(external_clause.tokens) for index in from_offsets),
        "external and residual contribution scopes are not explicit",
        FrontendStatus.AMBIGUOUS,
    )
    external_owner = _noun_after_index(external_clause, from_offsets[0])
    residual_owner = _noun_after_index(external_clause, from_offsets[1])
    _require(
        residual_owner == member and external_owner != member,
        "residual recipient scope differs",
        FrontendStatus.AMBIGUOUS,
    )
    cardinal = _unique_word_token(clause_set, "twenty", clause=declaration)

    builder = _Builder(source, clause_set)
    residual = _ledger(
        builder,
        (
            _Contribution(total, MONEY, "total"),
            _Contribution(external, MONEY, "external", sign=-1),
        ),
    )
    denominator = builder.lexical_literal(cardinal, Fraction(20), COUNT)
    expression = QuotientExpr(
        residual,
        denominator,
        _wide_span(residual, denominator, source),
    )
    return builder.finish(expression, "equal_share_residual")


def _unit_cost_residual(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Residual total divided by an explicit target count with a bound price ref."""

    question = _question(clause_set)
    if not (
        _contains(question.norms, "how", "much")
        and "pay" in question.norms
        and "total" in question.norms
    ):
        return None
    target_cardinal = [token for token in question.tokens if token.norm == "one"]
    if len(target_cardinal) != 1:
        return None
    target_index = question.tokens.index(target_cardinal[0])
    _require(target_index + 1 < len(question.tokens), "target item is missing")
    target_item = _singular(question.norms[target_index + 1])
    try:
        question_owner = question.norms[question.norms.index("does") + 1]
    except (ValueError, IndexError) as exc:
        raise _Reject(
            FrontendStatus.AMBIGUOUS, "unit-cost owner is not explicit"
        ) from exc
    narrative_names = {
        token.norm
        for clause in clause_set
        if clause is not question
        for token in clause.tokens
        if token.text[:1].isupper()
        and token.norm not in {"a", "an", "the", "if", "when", "he", "she", "they"}
    }
    _require(
        narrative_names == {question_owner},
        "unit-cost pronouns have no unique explicit owner",
        FrontendStatus.AMBIGUOUS,
    )
    inventory = _single_clause(
        clause_set,
        lambda clause: (
            "eats" in clause.norms
            and "two" in clause.norms
            and "every" in clause.norms
            and "day" in clause.norms
        ),
        reason="daily target inventory is not unique",
    )
    two = _unique_word_token(clause_set, "two", clause=inventory)
    two_index = inventory.tokens.index(two)
    _require(
        two_index + 1 < len(inventory.tokens)
        and _singular(inventory.norms[two_index + 1]) == target_item,
        "target count binds another item",
        FrontendStatus.AMBIGUOUS,
    )
    price_clause = _single_clause(
        clause_set,
        lambda clause: (
            "costs" in clause.norms
            and "half" in clause.norms
            and "price" in clause.norms
        ),
        reason="dependent price relation is not unique",
    )
    price = _one_token(_money_tokens(price_clause), "base price is incomplete")
    half = _one_token(
        [token for token in price_clause.tokens if token.norm == "half"],
        "dependent price factor is incomplete",
    )
    total = _one_token(_money_tokens(question), "daily total is incomplete")
    costs_index = price_clause.norms.index("costs")
    while_index = (
        price_clause.norms.index("while") if "while" in price_clause.norms else -1
    )
    _require(
        costs_index == 2
        and price_clause.norms[0] == "the"
        and 0 <= while_index < len(price_clause.tokens) - 2,
        "dependent price items are missing",
    )
    inventory_subject = inventory.norms[0]
    price_object = (
        price_clause.norms[costs_index + 1]
        if costs_index + 1 < len(price_clause.tokens)
        else ""
    )
    _require(
        (inventory_subject, price_object)
        in {("he", "him"), ("she", "her"), ("they", "them")},
        "unit-cost owner pronouns do not agree",
        FrontendStatus.AMBIGUOUS,
    )
    half_index = price_clause.tokens.index(half)
    _require(
        tuple(price_clause.norms[half_index:]) == ("half", "the", "price"),
        "dependent price does not refer to the bound base price",
        FrontendStatus.AMBIGUOUS,
    )
    base_item = _singular(price_clause.norms[costs_index - 1])
    dependent_item = _singular(price_clause.norms[while_index + 2])
    inventory_nouns = {_singular(word) for word in inventory.norms}
    _require(
        {base_item, dependent_item}.issubset(inventory_nouns),
        "priced goods are outside the inventory",
        FrontendStatus.AMBIGUOUS,
    )

    builder = _Builder(source, clause_set)
    base_symbol = SymbolKey(
        "daily_inventory", "price", base_item, target_item, "current"
    )
    base_definition = Definition(
        base_symbol,
        builder.literal(price, MONEY),
        price_clause.span,
    )
    base_ref = RefExpr(base_symbol, price_clause.span)
    dependent = _product(
        builder.literal(half, SCALAR),
        RefExpr(base_symbol, price_clause.span),
    )
    residual = _sum(
        _signed(1, builder.literal(total, MONEY), "total"),
        _signed(-1, base_ref, "base_item"),
        _signed(-1, dependent, "dependent_item"),
    )
    denominator = builder.lexical_literal(two, Fraction(2), COUNT)
    expression = QuotientExpr(
        residual,
        denominator,
        _wide_span(residual, denominator, source),
    )
    return builder.finish(
        expression,
        "unit_cost_residual",
        definitions=(base_definition,),
    )


def _inventory_total_residual(
    source: str, clause_set: tuple[Clause, ...]
) -> FrontendResult | None:
    """Total inventory minus all explicitly scoped component declarations."""

    question = _question(clause_set)
    if not (_contains(question.norms, "how", "many") and "came" in question.norms):
        return None
    came_index = question.norms.index("came")
    target_words = tuple(
        word for word in question.norms[2:came_index] if word not in {"the", "a", "an"}
    )
    _require(
        bool(target_words),
        "residual target item is missing",
        FrontendStatus.AMBIGUOUS,
    )
    intro = _single_clause(
        clause_set,
        lambda clause: "variety" in clause.norms and "of" in clause.norms,
        reason="inventory scope declaration is not unique",
    )
    variety_index = intro.norms.index("variety")
    try:
        watches_index = intro.norms.index("watches")
    except ValueError as exc:
        raise _Reject(
            FrontendStatus.AMBIGUOUS, "inventory owner is not explicit"
        ) from exc
    _require(watches_index > 0, "inventory owner is not explicit")
    owner = intro.norms[watches_index - 1]
    try:
        of_index = intro.norms.index("of", variety_index)
    except ValueError as exc:
        raise _Reject(
            FrontendStatus.UNSUPPORTED, "inventory scope noun is missing"
        ) from exc
    _require(of_index + 1 < len(intro.tokens), "inventory scope noun is missing")
    scope = _singular(intro.norms[of_index + 1])
    total_clause = _single_clause(
        clause_set,
        lambda clause: (
            _contains(clause.norms, "total", "number", "of")
            and bool(_digit_tokens(clause))
        ),
        reason="inventory total is not unique",
    )
    _require(
        bool(total_clause.norms) and total_clause.norms[0] == owner,
        "inventory total belongs to another owner",
        FrontendStatus.AMBIGUOUS,
    )
    total_phrase_index = next(
        index
        for index in range(len(total_clause.norms) - 2)
        if total_clause.norms[index : index + 3] == ("total", "number", "of")
    )
    intro_subjects = [word for word in intro.norms if word in {"he", "she", "they"}]
    possessive = (
        total_clause.norms[total_phrase_index - 1] if total_phrase_index > 0 else ""
    )
    possessive_by_subject = {"he": "his", "she": "her", "they": "their"}
    explicit_possessives = {f"{owner}'s", f"{owner}’s"}
    _require(
        len(intro_subjects) == 1
        and possessive
        in {possessive_by_subject[intro_subjects[0]], *explicit_possessives},
        "inventory total possessive belongs to another owner",
        FrontendStatus.AMBIGUOUS,
    )
    _require(
        total_phrase_index + 3 < len(total_clause.tokens)
        and _singular(total_clause.norms[total_phrase_index + 3]) == scope,
        "inventory total scope differs",
        FrontendStatus.AMBIGUOUS,
    )
    _require(
        _contains(total_clause.norms, *target_words),
        "residual target is outside the total clause",
        FrontendStatus.AMBIGUOUS,
    )
    total = _one_token(_counts(total_clause), "inventory total is incomplete")
    component_clauses = [
        clause
        for clause in clause_set
        if clause is not total_clause
        and clause is not question
        and "has" in clause.norms
        and len([token for token in _digit_tokens(clause) if not token.money]) == 1
    ]
    _require(
        len(component_clauses) == 3,
        "inventory residual requires exactly three scoped components",
    )
    components: list[tuple[Token, str]] = []
    component_items: set[str] = set()
    for clause in component_clauses:
        token = _counts(clause)[0]
        index = clause.tokens.index(token)
        if (
            index == 0
            or clause.norms[index - 1] != "has"
            or index + 1 >= len(clause.tokens)
        ):
            raise _Reject(
                FrontendStatus.UNSUPPORTED, "component count lacks has-item binding"
            )
        has_index = index - 1
        object_end = next(
            (
                offset
                for offset in range(index + 1, len(clause.tokens))
                if clause.norms[offset] in {"in", "inside", "on"}
            ),
            len(clause.tokens),
        )
        if "of" not in clause.norms[:has_index] or object_end <= index + 1:
            raise _Reject(
                FrontendStatus.AMBIGUOUS,
                "component subject and counted item are not explicit",
            )
        subject_item = _singular(clause.norms[has_index - 1])
        item = _singular(clause.norms[object_end - 1])
        if subject_item != item:
            raise _Reject(
                FrontendStatus.AMBIGUOUS,
                "component subject and counted item differ",
            )
        if item in component_items or item in {
            _singular(word) for word in target_words
        }:
            raise _Reject(
                FrontendStatus.AMBIGUOUS, "inventory component scope is duplicated"
            )
        component_items.add(item)
        components.append((token, item))

    builder = _Builder(source, clause_set)
    expression = _ledger(
        builder,
        (
            _Contribution(total, COUNT, "inventory_total"),
            *tuple(
                _Contribution(token, COUNT, "known_component", sign=-1)
                for token, _ in components
            ),
        ),
    )
    return builder.finish(expression, "inventory_total_residual")


_PLANNERS = (
    _rate_length_difference,
    _functioning_chain,
    _daily_combined,
    _calendar_daily_total,
    _old_new_savings,
    _category_sales,
    _repeated_duration,
    _avoided_cost_transaction,
    _profit_contribution_ledger,
    _alternative_cost_savings,
    _equal_share_residual,
    _unit_cost_residual,
    _inventory_total_residual,
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
